# Floor3DPlan: Fine-Tuning a Vision-Language Model for 2D Floor Plan Detection → 3D

Fine-tunes **Qwen2.5-VL-3B-Instruct** with **QLoRA** to detect architectural elements in 2D floor plan
images and emit **schema-validated JSON**, which a procedural 2D-to-3D step extrudes into a 3D model.

Detected elements (coordinates normalized to integers 0–1000):

| Class | Sub-types | Geometry |
| :-- | :-- | :-- |
| Rooms | living_room, bedroom, kitchen, bathroom, hallway, dining_room | `bbox [ymin, xmin, ymax, xmax]` |
| Walls | interior, exterior | `start [x,y]`, `end [x,y]`, `bbox` |
| Doors | entrance, interior, sliding | `bbox` |
| Windows | – | `bbox` |

---

## Project structure

```
src/
  schema.py            prompt, JSON schema validation, robust JSON extraction, path helpers
  dataset_prep.py      reproducible synthetic CAD floor plan generator + train/val/test JSONL
  train.py             QLoRA fine-tuning (Transformers Trainer + PEFT)
  inference.py         HF (+LoRA) or Ollama inference, optional visualization
  evaluate.py          P/R/F1 @IoU 0.50/0.75, F1@[.50:.95], mean IoU, type accuracy, JSON validity
  visualize.py         2D overlay + 3D extrusion with door/window openings cut into walls
  export_to_ollama.py  merge LoRA into base weights + Ollama Modelfile
tests/test_pipeline.py  CPU-only tests (no model download)
notebooks/floorplan_finetune_colab.ipynb   one-click Colab run of the same src/ code
data/
  raw/images/          generated floor plans        (not committed - run dataset_prep.py)
  processed/           train/val/test JSONL 96/12/12 (not committed - run dataset_prep.py)
  samples/             5 test-split samples (+ overlay / 3D preview)
```

The dataset is generated with a fixed seed, so `python src/dataset_prep.py --samples 120`
recreates exactly the same images and splits. Trained LoRA weights (`checkpoints/`) are not stored in the repo.

---

## Why Qwen2.5-VL-3B-Instruct
* Native visual grounding and good adherence to structured JSON output.
* Fits a free Colab T4 (16 GB) or a 6–8 GB consumer GPU in 4-bit QLoRA.
* Alternatives considered: Florence-2 (fast boxes but weak at nested JSON), PaliGemma-2, YOLOv8-seg
  (excellent masks but no language-conditioned structured output).

---

## Dataset

`src/dataset_prep.py` generates a **reproducible** (seeded) synthetic CAD-style dataset:
randomized margins and wall thicknesses, randomized room-type assignment (labelled in the drawing),
swing and sliding doors, 2–4 windows per plan. Every ground truth is validated against the schema.
All coordinates are normalized: `coord_norm = round(coord_px / dimension × 1000)`.

Each record is a Qwen chat conversation; the assistant target is **compact JSON with no markdown fences**
(~1.3k characters, comfortably inside the 2,048-token generation limit):

```json
{"id": "floorplan_0032", "image": "data/raw/images/floorplan_0032.png",
 "messages": [
   {"role": "user", "content": [{"type": "image", "image": "..."}, {"type": "text", "text": "<FLOORPLAN_PROMPT>"}]},
   {"role": "assistant", "content": [{"type": "text", "text": "{\"image_dimensions\":{...},\"rooms\":[...],...}"}]}
 ]}
```

Real-data extension: CubiCasa5k can be converted into the same schema (not included here).

---

## Fine-tuning setup

| Setting | Value |
| :-- | :-- |
| Base model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Quantization | 4-bit NF4, double quantization (bitsandbytes) |
| LoRA | r = 16, α = 32, dropout 0.05, on the **language model** q/k/v/o/gate/up/down projections (vision tower frozen) |
| Loss | only on the assistant answer (prompt, image and padding tokens masked) |
| Batch | 1 × 8 gradient accumulation |
| LR | 2e-4, cosine, 5 % warmup |
| Image resolution | 256–1024 × 28² pixels |
| Other | gradient checkpointing, best checkpoint by eval loss |

---

## Evaluation

Run on the **held-out test split** (never used for training or model selection):

```bash
python src/evaluate.py --run_inference --load_in_4bit          # fine-tuned model
python src/evaluate.py --run_inference --adapter none --limit 4 --out data/evaluation_baseline.json   # base model
```

Reported per class and macro-averaged: precision / recall / F1 at IoU 0.50, F1 at IoU 0.75,
mean F1 over IoU 0.50:0.95, mean matched IoU, sub-type accuracy, JSON parse rate and schema validity rate.
The model outputs no confidence scores, so a ranked-PR mAP is not defined; F1 at fixed IoU thresholds is used instead.

> **Results:** `data/evaluation_results.json` is written by the command above after training.
> Fill in the table below from that file — do not report numbers that were not produced by it.
> (`python src/evaluate.py --oracle` scores the ground truth against itself and must give 1.0; it only checks the metric code.)

| Metric | Base model | Fine-tuned |
| :-- | :-- | :-- |
| JSON schema validity | _run_ | _run_ |
| Macro F1 @ IoU 0.50 | _run_ | _run_ |
| Macro F1 @ IoU 0.50:0.95 | _run_ | _run_ |

---

## 2D → 3D downstream

`src/visualize.py` turns the JSON into:
1. a color-coded 2D overlay of all detections, and
2. a 3D extrusion: walls extruded to 2.8 m; door openings cut into walls with lintels at 2.1 m;
   window openings with sills at 0.9 m and lintels at 2.1 m plus glass panes; rooms as colored floor slabs.

---

## Quickstart

```bash
git clone https://github.com/miru-17/floor3Dplan--LLM-finetunig.git
cd floor3Dplan--LLM-finetunig
pip install -r requirements.txt
python tests/test_pipeline.py                      # CPU-only sanity tests
python src/dataset_prep.py --samples 120           # dataset (96/12/12)
python src/train.py --epochs 3                     # needs a CUDA GPU (or use the Colab notebook)
python src/evaluate.py --run_inference --load_in_4bit
python src/inference.py --image data/samples/sample_01.png --output_json data/samples/pred_01.json --visualize
python src/visualize.py --image data/samples/sample_01.png --json data/samples/sample_01.json
```

All scripts resolve paths relative to the project root, so they can be run from any folder.
On Colab, open `notebooks/floorplan_finetune_colab.ipynb` and run all cells (T4 GPU).

## Ollama serving (optional)

```bash
python src/export_to_ollama.py --merge             # merges LoRA into fp16 weights, writes Modelfile
ollama create floorplan-vlm -f Modelfile
python src/inference.py --engine ollama --model floorplan-vlm --image data/samples/sample_01.png
```
If your Ollama version cannot import Qwen2.5-VL Safetensors directly, convert the merged folder to GGUF
with llama.cpp first (see the docstring in `export_to_ollama.py`). The HF engine is the reference path.
