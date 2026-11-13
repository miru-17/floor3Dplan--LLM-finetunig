"""
Evaluation for 2D Floor Plan Detection.

Metrics (per class: rooms, walls, doors, windows):
  * JSON parse rate and schema validity rate
  * Precision / Recall / F1 at IoU 0.50 and 0.75 (optimal-greedy one-to-one matching)
  * mean F1 over IoU thresholds 0.50:0.95 (COCO-style threshold sweep)
  * mean IoU of matched pairs
  * type (sub-class) accuracy of matched pairs (e.g. bedroom vs kitchen)

NOTE: the model emits no confidence scores, so a true mAP (area under a ranked PR curve) is not
defined. F1 at fixed IoU thresholds is reported instead.

Modes:
  python src/evaluate.py --run_inference                    # run the fine-tuned model on the test split
  python src/evaluate.py --predictions data/predictions_test.jsonl   # score saved predictions
  python src/evaluate.py --oracle                           # pipeline sanity check (GT vs GT == 1.0)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from schema import CATEGORIES, PROJECT_ROOT, extract_json_from_response, resolve_path, \
    validate_floorplan_json, wall_bbox

IOU_SWEEP = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]


def calculate_iou(box_a: List[float], box_b: List[float]) -> float:
    """IoU for boxes in [ymin, xmin, ymax, xmax] format."""
    y_a, x_a = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    y_b, x_b = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, y_b - y_a) * max(0.0, x_b - x_a)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _box_of(item: Any, category: str):
    if not isinstance(item, dict):
        return None
    box = wall_bbox(item) if category == "walls" else item.get("bbox")
    if isinstance(box, (list, tuple)) and len(box) == 4 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in box):
        return [float(v) for v in box]
    return None


def match_boxes(preds: List[Dict], gts: List[Dict], category: str, iou_threshold: float = 0.5
                ) -> Tuple[int, int, int, List[float], List[bool]]:
    """
    One-to-one matching: all (pred, gt) pairs sorted by IoU, taken greedily (highest first).
    Returns (TP, FP, FN, matched IoUs, type-correct flags for matched pairs).
    """
    p_boxes = [_box_of(p, category) for p in preds]
    g_boxes = [_box_of(g, category) for g in gts]
    pairs = []
    for i, pb in enumerate(p_boxes):
        if pb is None:
            continue
        for j, gb in enumerate(g_boxes):
            if gb is None:
                continue
            iou = calculate_iou(pb, gb)
            if iou >= iou_threshold:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)

    used_p, used_g, ious, type_ok = set(), set(), [], []
    for iou, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        ious.append(iou)
        gt_type = gts[j].get("type") if isinstance(gts[j], dict) else None
        if gt_type is not None:
            type_ok.append(preds[i].get("type") == gt_type)

    tp = len(used_p)
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn, ious, type_ok


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate_predictions(gt_by_id: Dict[str, Dict], pred_by_id: Dict[str, Any]) -> Dict:
    """gt_by_id: id -> ground-truth dict.  pred_by_id: id -> raw text or parsed dict (missing = failure)."""
    n = len(gt_by_id)
    parsed_ok, schema_ok = 0, 0
    parsed_preds: Dict[str, Dict] = {}

    for sid in gt_by_id:
        raw = pred_by_id.get(sid)
        try:
            pred = extract_json_from_response(raw) if isinstance(raw, str) else raw
            if not isinstance(pred, dict):
                raise ValueError("not a dict")
            parsed_ok += 1
            if validate_floorplan_json(pred)[0]:
                schema_ok += 1
            parsed_preds[sid] = pred
        except Exception:
            parsed_preds[sid] = {}  # unparseable -> every GT element is a false negative

    summary: Dict[str, Any] = {
        "total_test_samples": n,
        "json_parse_rate_percent": round(100 * parsed_ok / n, 2) if n else 0.0,
        "schema_validity_rate_percent": round(100 * schema_ok / n, 2) if n else 0.0,
        "per_class_metrics": {},
    }

    macro = {k: [] for k in ["precision@0.50", "recall@0.50", "f1@0.50", "f1@0.75", "mean_f1@[.50:.95]"]}
    for c in CATEGORIES:
        sweep_f1 = []
        stats_50 = stats_75 = None
        all_ious, all_types = [], []
        for thr in IOU_SWEEP:
            tp = fp = fn = 0
            for sid, gt in gt_by_id.items():
                pred_list = parsed_preds[sid].get(c, [])
                pred_list = pred_list if isinstance(pred_list, list) else []
                t, f_p, f_n, ious, types = match_boxes(pred_list, gt.get(c, []), c, thr)
                tp, fp, fn = tp + t, fp + f_p, fn + f_n
                if thr == 0.5:
                    all_ious += ious
                    all_types += types
            prf = _prf(tp, fp, fn)
            sweep_f1.append(prf[2])
            if thr == 0.5:
                stats_50 = (tp, fp, fn, *prf)
            if thr == 0.75:
                stats_75 = prf
        tp, fp, fn, p, r, f = stats_50
        entry = {
            "tp@0.50": tp, "fp@0.50": fp, "fn@0.50": fn,
            "precision@0.50": round(p, 4), "recall@0.50": round(r, 4), "f1@0.50": round(f, 4),
            "f1@0.75": round(stats_75[2], 4),
            "mean_f1@[.50:.95]": round(float(np.mean(sweep_f1)), 4),
            "mean_matched_iou": round(float(np.mean(all_ious)), 4) if all_ious else 0.0,
        }
        if all_types:
            entry["type_accuracy"] = round(sum(all_types) / len(all_types), 4)
        summary["per_class_metrics"][c] = entry
        for k in macro:
            macro[k].append(entry[k])

    summary["macro"] = {k: round(float(np.mean(v)), 4) for k, v in macro.items()}
    return summary


def load_ground_truth(test_jsonl: Path) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    gts, images = {}, {}
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            content = rec["messages"][-1]["content"]
            text = content if isinstance(content, str) else "".join(
                p.get("text", "") for p in content if isinstance(p, dict))
            gts[rec["id"]] = extract_json_from_response(text)
            images[rec["id"]] = rec["image"]
    return gts, images


def load_predictions(path: Path) -> Dict[str, Any]:
    preds = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                preds[rec["id"]] = rec.get("prediction")
    return preds


def print_report(m: Dict, title: str):
    print(f"\n==================== {title} ====================")
    print(f"Samples: {m['total_test_samples']}   JSON parse rate: {m['json_parse_rate_percent']}%   "
          f"Schema valid: {m['schema_validity_rate_percent']}%")
    print(f"{'class':<9}{'P@.5':>8}{'R@.5':>8}{'F1@.5':>8}{'F1@.75':>8}{'F1@.5:.95':>11}{'mIoU':>8}{'typeAcc':>9}")
    for c, v in m["per_class_metrics"].items():
        print(f"{c:<9}{v['precision@0.50']:>8}{v['recall@0.50']:>8}{v['f1@0.50']:>8}{v['f1@0.75']:>8}"
              f"{v['mean_f1@[.50:.95]']:>11}{v['mean_matched_iou']:>8}{str(v.get('type_accuracy', '-')):>9}")
    mm = m["macro"]
    print(f"{'MACRO':<9}{mm['precision@0.50']:>8}{mm['recall@0.50']:>8}{mm['f1@0.50']:>8}{mm['f1@0.75']:>8}"
          f"{mm['mean_f1@[.50:.95]']:>11}")
    print("=" * (42 + len(title)))


def main():
    ap = argparse.ArgumentParser(description="Evaluate floor plan detection on the test split")
    ap.add_argument("--test_file", default="data/processed/test.jsonl")
    ap.add_argument("--predictions", default=None, help="JSONL with {id, prediction} per line")
    ap.add_argument("--run_inference", action="store_true", help="Run the HF model on the test split first")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default="checkpoints/qwen_floorplan_lora/final_adapter")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only the first N test samples")
    ap.add_argument("--oracle", action="store_true", help="Score GT against itself (pipeline sanity check)")
    ap.add_argument("--out", default=None, help="Where to write the metrics JSON")
    args = ap.parse_args()

    test_jsonl = Path(resolve_path(args.test_file))
    if not test_jsonl.exists():
        raise SystemExit(f"{test_jsonl} not found. Run `python src/dataset_prep.py` first.")
    gts, images = load_ground_truth(test_jsonl)
    if args.limit:
        gts = dict(list(gts.items())[:args.limit])

    if args.oracle:
        preds, mode = dict(gts), "oracle_sanity_check (ground truth scored against itself - NOT model performance)"
        default_out = PROJECT_ROOT / "data" / "evaluation_oracle_sanity_check.json"
    elif args.predictions:
        preds, mode = load_predictions(Path(resolve_path(args.predictions))), f"predictions:{args.predictions}"
        default_out = PROJECT_ROOT / "data" / "evaluation_results.json"
    elif args.run_inference:
        from inference import HFFloorplanDetector
        det = HFFloorplanDetector(args.model, args.adapter, args.load_in_4bit)
        pred_path = PROJECT_ROOT / "data" / "predictions_test.jsonl"
        preds = {}
        with open(pred_path, "w", encoding="utf-8") as f:
            for k, sid in enumerate(gts, 1):
                print(f"[{k}/{len(gts)}] {sid}")
                raw = det.predict(images[sid])
                preds[sid] = raw
                f.write(json.dumps({"id": sid, "prediction": raw}, ensure_ascii=False) + "\n")
        print(f"Saved raw predictions to {pred_path}")
        mode = f"model:{args.model} adapter:{args.adapter}"
        default_out = PROJECT_ROOT / "data" / "evaluation_results.json"
    else:
        ap.print_help()
        raise SystemExit("\nChoose one of --run_inference, --predictions FILE or --oracle.")

    metrics = evaluate_predictions(gts, preds)
    metrics = {"mode": mode, **metrics}
    out = Path(args.out) if args.out else default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print_report(metrics, "Evaluation" if not args.oracle else "ORACLE sanity check")
    print(f"Saved metrics to {out}")


if __name__ == "__main__":
    main()
