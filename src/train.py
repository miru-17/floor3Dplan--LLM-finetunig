"""
QLoRA fine-tuning of Qwen2.5-VL for 2D floor plan element detection.

Fixes vs. the original version:
  * nested chat messages are stored as JSON strings (Arrow cannot hold the mixed nested schema and
    would inject `image: None` keys that break qwen_vl_utils),
  * remove_unused_columns=False (otherwise Trainer silently drops the `messages` column -> empty batches),
  * `eval_strategy` instead of the removed `evaluation_strategy` argument (and v5-safe warmup),
  * LoRA restricted to the language model (the ViT's MLP also has gate/up/down_proj layers),
  * loss is computed ONLY on the assistant answer (prompt, image and padding tokens are masked to -100),
  * image paths are resolved relative to the project root,
  * works on CPU for smoke tests (no bitsandbytes / bf16 calls without CUDA).

Usage:
    python src/train.py --epochs 3
"""

import argparse
import inspect
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

from schema import PROJECT_ROOT, resolve_path

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28
ASSISTANT_HEADER = "<|im_start|>assistant\n"
LORA_TARGET_REGEX = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def load_dataset_from_jsonl(jsonl_path: str) -> Dataset:
    """Loads chat JSONL, resolves image paths, stores messages as a JSON string column."""
    jsonl_path = Path(resolve_path(jsonl_path))
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            for msg in rec["messages"]:
                if isinstance(msg["content"], list):
                    for part in msg["content"]:
                        if part.get("type") == "image":
                            part["image"] = resolve_path(part["image"], base=PROJECT_ROOT)
            rows.append({"id": rec["id"], "messages_json": json.dumps(rec["messages"], ensure_ascii=False)})
    if not rows:
        raise ValueError(f"No records found in {jsonl_path}")
    return Dataset.from_list(rows)


class FloorplanCollator:
    """Tokenizes multimodal conversations and masks everything except the assistant answer."""

    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"
        self.header_ids = processor.tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)
        self.pad_id = processor.tokenizer.pad_token_id

    def _answer_start(self, ids):
        """Index right after the LAST '<|im_start|>assistant\\n' header."""
        n = len(self.header_ids)
        for i in range(len(ids) - n, -1, -1):
            if ids[i:i + n] == self.header_ids:
                return i + n
        return None

    def __call__(self, batch):
        messages_list = [json.loads(item["messages_json"]) for item in batch]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                 for m in messages_list]
        image_inputs, video_inputs = process_vision_info(messages_list)
        inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt")

        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        for i in range(labels.size(0)):
            start = self._answer_start(inputs["input_ids"][i].tolist())
            if start is None:
                raise ValueError("Assistant header not found - check the chat template / data format.")
            labels[i, :start] = -100
        if (labels != -100).sum() == 0:
            raise ValueError("No trainable answer tokens in batch.")
        inputs["labels"] = labels
        return inputs


def train(args):
    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else (torch.float16 if use_cuda else torch.float32)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {args.model_id} (cuda={use_cuda}, dtype={compute_dtype})")
    quant_config = None
    if use_cuda and not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map="auto" if use_cuda else None,
    )
    model.config.use_cache = False  # required with gradient checkpointing

    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

    if quant_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        # LoRA on the language model only: the vision tower's MLP also uses gate/up/down_proj names,
        # so a plain name list would silently adapt the ViT as well.
        target_modules=LORA_TARGET_REGEX,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading datasets...")
    train_ds = load_dataset_from_jsonl(args.train_file)
    val_ds = load_dataset_from_jsonl(args.val_file)
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    # transformers v5 removed `warmup_ratio` (warmup_steps accepts a float ratio instead)
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    warmup = {"warmup_ratio": 0.05} if "warmup_ratio" in ta_params else {"warmup_steps": 0.05}

    training_args = TrainingArguments(
        **warmup,
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bf16,
        fp16=use_cuda and not bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,   # keep `messages_json` for the collator
        dataloader_pin_memory=use_cuda,
        prediction_loss_only=True,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=FloorplanCollator(processor),
    )

    print("Starting fine-tuning...")
    trainer.train()

    adapter_path = output_dir / "final_adapter"
    trainer.model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    print(f"Fine-tuning complete! LoRA adapter saved to {adapter_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL for 2D Floor Plan Detection")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--train_file", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/processed/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/qwen_floorplan_lora")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization on GPU")
    args = parser.parse_args()
    train(args)
