"""
Merge the fine-tuned LoRA adapter into the base model and generate an Ollama Modelfile.

Usage:
    python src/export_to_ollama.py --merge                     # merge + Modelfile pointing at merged weights
    ollama create floorplan-vlm -f Modelfile
    python src/inference.py --engine ollama --model floorplan-vlm --image data/samples/sample_01.png

Notes:
  * Merging must be done on a NON-quantized (fp16/bf16) base model; the 4-bit weights used during
    QLoRA training cannot be merged into directly.
  * Ollama can only import Safetensors for the architectures its converter supports. If
    `ollama create` rejects Qwen2_5_VLForConditionalGeneration in your Ollama version, convert the merged
    folder to GGUF with llama.cpp (text model + vision projector), then point FROM at the GGUF file(s):
        python llama.cpp/convert_hf_to_gguf.py checkpoints/merged_fp16 --outfile floorplan.gguf
        python llama.cpp/convert_hf_to_gguf.py checkpoints/merged_fp16 --mmproj --outfile floorplan-mmproj.gguf
  * The HF engine (`python src/inference.py`) is the reference path and always works.
"""

import argparse
from pathlib import Path

from schema import PROJECT_ROOT, resolve_path

# No SYSTEM override: training used Qwen's default system prompt, and the task prompt is sent by
# inference.py as the user message - changing either at serving time hurts accuracy.
MODELFILE_TEMPLATE = """# Ollama Modelfile for the fine-tuned 2D floor plan detection VLM
FROM {model_path}

# Deterministic decoding for structured JSON
PARAMETER temperature 0
PARAMETER top_p 1
PARAMETER num_ctx 8192
PARAMETER num_predict 2048
"""


def _abs(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def create_modelfile(model_path: str, output_modelfile: str = "Modelfile"):
    out = _abs(output_modelfile)
    out.write_text(MODELFILE_TEMPLATE.format(model_path=Path(model_path).as_posix()), encoding="utf-8")
    print(f"Generated Ollama Modelfile at {out}")
    print("\nTo build and run in Ollama:")
    print(f"  ollama create floorplan-vlm -f \"{out}\"")
    print("  python src/inference.py --engine ollama --model floorplan-vlm --image data/samples/sample_01.png")


def merge_lora_weights(base_model_id: str, adapter_path: str, output_merged_dir: str) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    adapter = Path(resolve_path(adapter_path))
    if not adapter.is_dir():
        raise SystemExit(f"Adapter not found: {adapter}. Train first (src/train.py) or pass --adapter.")
    out_dir = _abs(output_merged_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model {base_model_id} (fp16, CPU) ...")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model_id, torch_dtype=torch.float16,
                                                              low_cpu_mem_usage=True)
    print(f"Loading adapter {adapter} ...")
    model = PeftModel.from_pretrained(base, str(adapter))
    print("Merging LoRA weights ...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {out_dir} ...")
    merged.save_pretrained(out_dir, safe_serialization=True)
    # Prefer the processor saved with the adapter (same min/max pixels as training)
    proc_src = str(adapter) if (adapter / "preprocessor_config.json").exists() else base_model_id
    AutoProcessor.from_pretrained(proc_src).save_pretrained(out_dir)
    print("Merge complete!")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export fine-tuned model for Ollama")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", type=str, default="checkpoints/qwen_floorplan_lora/final_adapter")
    parser.add_argument("--merged_dir", type=str, default="checkpoints/merged_fp16")
    parser.add_argument("--modelfile", type=str, default="Modelfile")
    parser.add_argument("--merge", action="store_true", help="Merge the LoRA adapter into the base model")
    args = parser.parse_args()

    if args.merge:
        merged_dir = merge_lora_weights(args.base_model, args.adapter, args.merged_dir)
        create_modelfile(str(merged_dir), args.modelfile)
    else:
        print("No --merge given: generating a Modelfile for the BASE model from the Ollama library (not fine-tuned).")
        create_modelfile("qwen2.5vl:3b", args.modelfile)
