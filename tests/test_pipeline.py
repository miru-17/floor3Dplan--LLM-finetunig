"""
Lightweight tests (no GPU / no model download needed).
Run:  python -m pytest tests -q      or      python tests/test_pipeline.py
"""

import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from schema import extract_json_from_response, validate_floorplan_json, to_target_text, wall_bbox  # noqa: E402
from evaluate import calculate_iou, match_boxes, evaluate_predictions  # noqa: E402
from dataset_prep import generate_synthetic_floorplan  # noqa: E402
from visualize import build_wall_segments, draw_2d_detections, render_3d_extrusion  # noqa: E402


def _sample():
    tmp = Path(tempfile.mkdtemp())
    return generate_synthetic_floorplan(1, tmp, random.Random(0))


def test_generated_ground_truth_is_valid_and_normalized():
    for seed in range(20):
        item = generate_synthetic_floorplan(seed, Path(tempfile.mkdtemp()), random.Random(seed))
        ok, errs = validate_floorplan_json(item["ground_truth"])
        assert ok, errs


def test_extract_json_variants():
    obj = {"rooms": [], "walls": [], "doors": [], "windows": []}
    s = to_target_text(obj)
    assert extract_json_from_response(s) == obj
    assert extract_json_from_response(f"```json\n{s}\n```") == obj
    assert extract_json_from_response(f"Sure! Here it is:\n{s}\nHope that helps") == obj
    try:
        extract_json_from_response("no json here")
        assert False
    except ValueError:
        pass


def test_validation_catches_bad_boxes():
    bad = {"rooms": [{"id": "r", "bbox": [1, 2, 3]}], "walls": [{}], "doors": [{"bbox": ["a", 1, 2, 3]}],
           "windows": [{"bbox": [0, 0, 2000, 10]}]}
    ok, errs = validate_floorplan_json(bad)
    assert not ok and len(errs) >= 4


def test_iou_and_matching():
    assert calculate_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert calculate_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert abs(calculate_iou([0, 0, 10, 10], [0, 5, 10, 15]) - 1 / 3) < 1e-9
    gts = [{"bbox": [0, 0, 10, 10], "type": "bedroom"}, {"bbox": [20, 20, 30, 30], "type": "kitchen"}]
    preds = [{"bbox": [20, 20, 30, 30], "type": "kitchen"}, {"bbox": [0, 0, 10, 10], "type": "kitchen"},
             {"bbox": [50, 50, 60, 60]}]
    tp, fp, fn, ious, types = match_boxes(preds, gts, "rooms", 0.5)
    assert (tp, fp, fn) == (2, 1, 0)
    assert sorted(types) == [False, True]


def test_walls_without_bbox_are_matched_via_endpoints():
    assert wall_bbox({"start": [100, 50], "end": [100, 500]}) is not None
    gt = [{"start": [100, 50], "end": [100, 500], "bbox": [50, 95, 500, 105]}]
    pred = [{"start": [100, 50], "end": [100, 500]}]
    assert match_boxes(pred, gt, "walls", 0.5)[0] == 1


def test_evaluate_oracle_is_perfect_and_garbage_is_zero():
    gt = _sample()["ground_truth"]
    m = evaluate_predictions({"a": gt}, {"a": gt})
    assert m["macro"]["f1@0.50"] == 1.0 and m["schema_validity_rate_percent"] == 100.0
    m = evaluate_predictions({"a": gt}, {"a": "garbage"})
    assert m["macro"]["f1@0.50"] == 0.0 and m["json_parse_rate_percent"] == 0.0
    m = evaluate_predictions({"a": gt}, {})          # missing prediction
    assert m["macro"]["recall@0.50"] == 0.0


def test_wall_cutting():
    # 10 m horizontal wall with a 1 m door in the middle
    solids = build_wall_segments([(0, 10, 0, 0.2)], [((4.5, 5.5, -0.1, 0.3), "door")], 2.8)
    assert len(solids) == 3                           # left wall, lintel, right wall
    lintel = [s for s in solids if s[4] > 0][0]
    assert abs(lintel[4] - 2.1) < 1e-9


def test_visualization_runs():
    item = _sample()
    out = Path(tempfile.mkdtemp())
    draw_2d_detections(item["abs_image_path"], item["ground_truth"], str(out / "o.png"))
    render_3d_extrusion(item["ground_truth"], str(out / "d.png"))
    assert (out / "o.png").exists() and (out / "d.png").exists()


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} tests passed.")
