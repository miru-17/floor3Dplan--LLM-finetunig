"""
Inference for 2D Floor Plan Detection.

Engines:
  * hf     - Hugging Face Transformers + (optional) LoRA adapter
  * ollama - local Ollama daemon (base64 image via /api/generate)

Usage:
    python src/inference.py --image data/samples/sample_01.png --output_json out.json
    python src/inference.py --engine ollama --model floorplan-vlm --image data/samples/sample_01.png
"""

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Optional

from schema import FLOORPLAN_PROMPT, PROJECT_ROOT, extract_json_from_response, resolve_path, validate_floorplan_json

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_OLLAMA_MODEL = "qwen2.5vl:3b"
DEFAULT_ADAPTER = "checkpoints/qwen_floorplan_lora/final_adapter"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28


class HFFloorplanDetector:
    """Loads the model once; call .predict(image_path) many times (used by evaluate.py too)."""

    def __init__(self, model_id: str = DEFAULT_HF_MODEL, adapter_path: Optional[str] = DEFAULT_ADAPTER,
                 load_in_4bit: bool = False, max_new_tokens: int = 2048):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        use_cuda = torch.cuda.is_available()
        dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if use_cuda else torch.float32

        kwargs = {"torch_dtype": dtype, "device_map": "auto" if use_cuda else None}
        if load_in_4bit and use_cuda:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True)

        print(f"Loading model {model_id} ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)

        processor_src = model_id
        if adapter_path and str(adapter_path).lower() in ("none", "base", ""):
            print("Running the BASE model without a LoRA adapter.")
            adapter_path = None
        adapter_path = resolve_path(adapter_path) if adapter_path else None
        if adapter_path and os.path.isdir(adapter_path):
            from peft import PeftModel
            print(f"Loading LoRA adapter from {adapter_path} ...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            if os.path.exists(os.path.join(adapter_path, "preprocessor_config.json")):
                processor_src = adapter_path  # same image resolution settings as training
        elif adapter_path:
            print(f"WARNING: adapter '{adapter_path}' not found -> running the BASE model (not fine-tuned).")

        self.processor = AutoProcessor.from_pretrained(processor_src, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        self.model.eval()

    def predict(self, image_path: str) -> str:
        from qwen_vl_utils import process_vision_info

        image_path = resolve_path(image_path, base=PROJECT_ROOT)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": FLOORPLAN_PROMPT},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.model.device)

        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                            do_sample=False, use_cache=True)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]


def run_ollama_inference(image_path: str, model_tag: str = DEFAULT_OLLAMA_MODEL,
                         host: str = "http://localhost:11434", timeout: int = 600) -> str:
    import requests

    image_path = resolve_path(image_path, base=PROJECT_ROOT)
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": model_tag,
        "prompt": FLOORPLAN_PROMPT,
        "images": [img_b64],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 2048},
    }
    url = f"{host.rstrip('/')}/api/generate"
    print(f"Sending request to Ollama: {url} (model: {model_tag}) ...")
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as e:
        raise SystemExit(f"Cannot reach Ollama at {host}. Is `ollama serve` running?") from e
    if resp.status_code == 404:
        raise SystemExit(f"Ollama model '{model_tag}' not found. Run `ollama pull {model_tag}` or `ollama create`.")
    resp.raise_for_status()
    return resp.json().get("response", "")


def main():
    parser = argparse.ArgumentParser(description="Floor Plan Detection Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input floor plan image")
    parser.add_argument("--engine", choices=["hf", "ollama"], default="hf")
    parser.add_argument("--model", type=str, default=None,
                        help=f"HF model id (default {DEFAULT_HF_MODEL}) or Ollama tag (default {DEFAULT_OLLAMA_MODEL})")
    parser.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER)
    parser.add_argument("--host", type=str, default="http://localhost:11434")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save parsed JSON")
    parser.add_argument("--visualize", action="store_true", help="Also save 2D overlay + 3D preview")
    args = parser.parse_args()

    if not Path(resolve_path(args.image)).exists():
        raise SystemExit(f"Image not found: {args.image}")

    if args.engine == "hf":
        detector = HFFloorplanDetector(args.model or DEFAULT_HF_MODEL, args.adapter, args.load_in_4bit)
        raw_output = detector.predict(args.image)
    else:
        raw_output = run_ollama_inference(args.image, args.model or DEFAULT_OLLAMA_MODEL, args.host)

    print("\n--- Raw Model Output ---")
    print(raw_output)

    try:
        parsed = extract_json_from_response(raw_output)
    except ValueError as e:
        print(f"\nFailed to parse JSON from output: {e}")
        raise SystemExit(1)

    is_valid, errors = validate_floorplan_json(parsed)
    print("\n--- JSON Validation ---")
    print("Schema valid:", is_valid)
    for err in errors:
        print("  -", err)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2)
        print(f"Saved structured JSON to {args.output_json}")

    if args.visualize:
        from visualize import draw_2d_detections, render_3d_extrusion
        stem = Path(args.output_json or Path(args.image).stem + "_pred.json").with_suffix("")
        draw_2d_detections(resolve_path(args.image), parsed, f"{stem}_overlay.png")
        render_3d_extrusion(parsed, f"{stem}_3d.png")


if __name__ == "__main__":
    main()
