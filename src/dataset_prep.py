"""
Dataset Preparation for 2D Floor Plan Detection.

1. Generates a reproducible synthetic CAD-style floor plan dataset (image + ground-truth JSON).
2. Normalizes every coordinate to integers in [0, 1000].
3. Writes Qwen2.5-VL chat-format JSONL splits (train / val / test).

Usage (from anywhere):
    python src/dataset_prep.py --samples 120
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont

from schema import (
    FLOORPLAN_PROMPT, PROJECT_ROOT, ROOM_TYPES, to_target_text, validate_floorplan_json,
)

WALL_COLOR = (30, 30, 30)
WHITE = (255, 255, 255)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _norm(v: float, dim: int) -> int:
    return max(0, min(1000, int(round(v / dim * 1000))))


def _norm_box(box: List[float], w: int, h: int) -> List[int]:
    ymin, xmin, ymax, xmax = box
    return [_norm(ymin, h), _norm(xmin, w), _norm(ymax, h), _norm(xmax, w)]


def _norm_pt(pt: List[float], w: int, h: int) -> List[int]:
    return [_norm(pt[0], w), _norm(pt[1], h)]


def _draw_swing_door(draw, x0, y_wall, length, wall_half, downward=True):
    """Door opening in a horizontal wall at y_wall with a leaf and a 90deg swing arc."""
    draw.rectangle([x0, y_wall - wall_half, x0 + length, y_wall + wall_half], fill=WHITE)
    sign = 1 if downward else -1
    y_face = y_wall + sign * wall_half
    draw.line([x0, y_face, x0, y_face + sign * length], fill=(10, 10, 10), width=2)
    box = [x0 - length, y_face - length, x0 + length, y_face + length]
    draw.arc(box, start=0 if downward else 270, end=90 if downward else 360, fill=(120, 120, 120), width=1)


def _draw_swing_door_vertical(draw, x_wall, y0, length, wall_half):
    """Door opening in a vertical wall at x_wall, swinging to the right."""
    draw.rectangle([x_wall - wall_half, y0, x_wall + wall_half, y0 + length], fill=WHITE)
    x_face = x_wall + wall_half
    draw.line([x_face, y0, x_face + length, y0], fill=(10, 10, 10), width=2)
    draw.arc([x_face - length, y0 - length, x_face + length, y0 + length], start=0, end=90,
             fill=(120, 120, 120), width=1)


def _draw_sliding_door(draw, x0, y_wall, length, wall_half):
    """Sliding door: opening with two offset parallel panels."""
    draw.rectangle([x0, y_wall - wall_half, x0 + length, y_wall + wall_half], fill=WHITE)
    half = length // 2
    draw.line([x0, y_wall - 2, x0 + half + 6, y_wall - 2], fill=(10, 10, 10), width=3)
    draw.line([x0 + half - 6, y_wall + 2, x0 + length, y_wall + 2], fill=(10, 10, 10), width=3)


def generate_synthetic_floorplan(img_idx: int, output_img_dir: Path, rng: random.Random,
                                 canvas_size: int = 1000) -> Dict[str, Any]:
    """Generate one CAD-style floor plan image and its normalized ground-truth JSON."""
    W = H = canvas_size
    img = Image.new("RGB", (W, H), color=WHITE)
    draw = ImageDraw.Draw(img)

    m = rng.randint(60, 110)                # outer margin
    ext_t = rng.choice([12, 14, 16])        # exterior wall thickness
    int_t = rng.choice([6, 8, 10])          # interior wall thickness
    x0, y0, x1, y1 = m, m, W - m, H - m
    sx = rng.randint(int(W * 0.40), int(W * 0.60))   # vertical divider x
    sy = rng.randint(int(H * 0.40), int(H * 0.60))   # horizontal divider y
    ih = int_t // 2

    # ---------------- Walls ----------------
    walls_px = [
        ("wall_ext_north", "exterior", [x0, y0 + ext_t / 2], [x1, y0 + ext_t / 2], [y0, x0, y0 + ext_t, x1]),
        ("wall_ext_south", "exterior", [x0, y1 - ext_t / 2], [x1, y1 - ext_t / 2], [y1 - ext_t, x0, y1, x1]),
        ("wall_ext_west", "exterior", [x0 + ext_t / 2, y0], [x0 + ext_t / 2, y1], [y0, x0, y1, x0 + ext_t]),
        ("wall_ext_east", "exterior", [x1 - ext_t / 2, y0], [x1 - ext_t / 2, y1], [y0, x1 - ext_t, y1, x1]),
        ("wall_int_vertical", "interior", [sx, y0 + ext_t], [sx, y1 - ext_t], [y0 + ext_t, sx - ih, y1 - ext_t, sx + ih]),
        ("wall_int_horizontal", "interior", [x0 + ext_t, sy], [x1 - ext_t, sy], [sy - ih, x0 + ext_t, sy + ih, x1 - ext_t]),
    ]
    for _, _, _, _, (bymin, bxmin, bymax, bxmax) in walls_px:
        draw.rectangle([bxmin, bymin, bxmax, bymax], fill=WALL_COLOR)

    # ---------------- Rooms (random type per quadrant, labelled in the drawing) ----------------
    quadrants = [
        [y0 + ext_t, x0 + ext_t, sy - ih, sx - ih],   # top-left
        [y0 + ext_t, sx + ih, sy - ih, x1 - ext_t],   # top-right
        [sy + ih, x0 + ext_t, y1 - ext_t, sx - ih],   # bottom-left
        [sy + ih, sx + ih, y1 - ext_t, x1 - ext_t],   # bottom-right
    ]
    room_types = rng.sample(ROOM_TYPES, 4)
    tints = {"bathroom": (242, 247, 252), "kitchen": (253, 249, 242)}
    font = _font(22)
    rooms_px = []
    for i, (box, rtype) in enumerate(zip(quadrants, room_types)):
        ymin, xmin, ymax, xmax = box
        if rtype in tints:
            draw.rectangle([xmin + 2, ymin + 2, xmax - 2, ymax - 2], fill=tints[rtype])
        draw.text(((xmin + xmax) // 2, (ymin + ymax) // 2), rtype.replace("_", " ").upper(),
                  fill=(90, 90, 90), font=font, anchor="mm")
        rooms_px.append((f"room_{i + 1}", rtype, box))

    # ---------------- Doors ----------------
    doors_px = []
    pad = 8  # bbox padding around the opening
    # Entrance on the south exterior wall (left half)
    dl = rng.randint(45, 65)
    ex = rng.randint(x0 + ext_t + 30, sx - ih - dl - 30)
    draw.rectangle([ex, y1 - ext_t, ex + dl, y1], fill=WHITE)
    draw.line([ex, y1 - ext_t, ex, y1 - ext_t - dl], fill=(10, 10, 10), width=2)
    draw.arc([ex - dl, y1 - ext_t - dl, ex + dl, y1 - ext_t + dl], start=270, end=360, fill=(120, 120, 120), width=1)
    doors_px.append(("door_entrance", "entrance", [y1 - ext_t - pad, ex, y1 + pad, ex + dl]))

    # Interior door on horizontal divider, left half
    dl = rng.randint(40, 55)
    dx = rng.randint(x0 + ext_t + 25, sx - ih - dl - 25)
    _draw_swing_door(draw, dx, sy, dl, ih, downward=True)
    doors_px.append(("door_2", "interior", [sy - ih - pad, dx, sy + ih + pad, dx + dl]))

    # Door on horizontal divider, right half (interior swing or sliding)
    dl = rng.randint(45, 70)
    dx = rng.randint(sx + ih + 25, x1 - ext_t - dl - 25)
    dtype = rng.choice(["interior", "sliding"])
    if dtype == "sliding":
        _draw_sliding_door(draw, dx, sy, dl, ih)
    else:
        _draw_swing_door(draw, dx, sy, dl, ih, downward=True)
    doors_px.append(("door_3", dtype, [sy - ih - pad, dx, sy + ih + pad, dx + dl]))

    # Interior door on vertical divider (top half)
    dl = rng.randint(40, 55)
    dy = rng.randint(y0 + ext_t + 25, sy - ih - dl - 25)
    _draw_swing_door_vertical(draw, sx, dy, dl, ih)
    doors_px.append(("door_4", "interior", [dy, sx - ih - pad, dy + dl, sx + ih + pad]))

    # ---------------- Windows (2-4, on exterior walls) ----------------
    windows_px = []
    wl = rng.randint(60, 95)
    candidates = [
        ("north", rng.randint(x0 + ext_t + 20, sx - ih - wl - 20)),
        ("north", rng.randint(sx + ih + 20, x1 - ext_t - wl - 20)),
        ("west", rng.randint(sy + ih + 20, y1 - ext_t - wl - 20)),
        ("east", rng.randint(y0 + ext_t + 20, sy - ih - wl - 20)),
    ]
    chosen = rng.sample(candidates, rng.randint(2, 4))
    for i, (side, pos) in enumerate(chosen):
        if side == "north":
            draw.rectangle([pos, y0, pos + wl, y0 + ext_t], fill=WHITE, outline=(40, 40, 40), width=2)
            draw.line([pos, y0 + ext_t // 2, pos + wl, y0 + ext_t // 2], fill=(70, 130, 180), width=2)
            box = [y0 - 4, pos, y0 + ext_t + 4, pos + wl]
        elif side == "west":
            draw.rectangle([x0, pos, x0 + ext_t, pos + wl], fill=WHITE, outline=(40, 40, 40), width=2)
            draw.line([x0 + ext_t // 2, pos, x0 + ext_t // 2, pos + wl], fill=(70, 130, 180), width=2)
            box = [pos, x0 - 4, pos + wl, x0 + ext_t + 4]
        else:  # east
            draw.rectangle([x1 - ext_t, pos, x1, pos + wl], fill=WHITE, outline=(40, 40, 40), width=2)
            draw.line([x1 - ext_t // 2, pos, x1 - ext_t // 2, pos + wl], fill=(70, 130, 180), width=2)
            box = [pos, x1 - ext_t - 4, pos + wl, x1 + 4]
        windows_px.append((f"window_{i + 1}", box))

    # ---------------- Save image ----------------
    img_filename = f"floorplan_{img_idx:04d}.png"
    img_path = output_img_dir / img_filename
    img.save(img_path)

    # ---------------- Normalized ground truth ----------------
    gt = {
        "image_dimensions": {"width": W, "height": H},
        "rooms": [{"id": rid, "type": t, "bbox": _norm_box(b, W, H)} for rid, t, b in rooms_px],
        "walls": [
            {"id": wid, "type": t, "start": _norm_pt(s, W, H), "end": _norm_pt(e, W, H), "bbox": _norm_box(b, W, H)}
            for wid, t, s, e, b in walls_px
        ],
        "doors": [{"id": did, "type": t, "bbox": _norm_box(b, W, H)} for did, t, b in doors_px],
        "windows": [{"id": wid, "bbox": _norm_box(b, W, H)} for wid, b in windows_px],
    }
    ok, errs = validate_floorplan_json(gt)
    assert ok, f"Generated invalid ground truth for {img_filename}: {errs}"

    try:
        rel = img_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel = img_path.resolve().as_posix()
    return {"id": f"floorplan_{img_idx:04d}", "image_path": rel, "abs_image_path": str(img_path), "ground_truth": gt}


def to_chat_record(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item["id"],
        "image": item["image_path"],
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": item["image_path"]},
                {"type": "text", "text": FLOORPLAN_PROMPT},
            ]},
            # Content as a list (same structure as the user turn) keeps HF datasets / Arrow happy.
            {"role": "assistant", "content": [{"type": "text", "text": to_target_text(item["ground_truth"])}]},
        ],
    }


def create_dataset_splits(num_samples: int = 120, output_dir: str = None, train_ratio: float = 0.8,
                          val_ratio: float = 0.1, seed: int = 42, canvas_size: int = 1000):
    base_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "data"
    img_dir, sample_dir, proc_dir = base_dir / "raw" / "images", base_dir / "samples", base_dir / "processed"
    for d in (img_dir, sample_dir, proc_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)  # fully reproducible
    print(f"Generating {num_samples} synthetic floor plans (seed={seed})...")
    items = [generate_synthetic_floorplan(i, img_dir, rng, canvas_size) for i in range(1, num_samples + 1)]

    rng.shuffle(items)
    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)
    splits = {"train": items[:n_train], "val": items[n_train:n_train + n_val], "test": items[n_train + n_val:]}

    # Sample deliverables are taken from the TEST split (never seen in training)
    for i, item in enumerate(splits["test"][:5], start=1):
        with open(sample_dir / f"sample_{i:02d}.json", "w", encoding="utf-8") as f:
            json.dump(item["ground_truth"], f, indent=2)
        Image.open(item["abs_image_path"]).save(sample_dir / f"sample_{i:02d}.png")

    for name, split_items in splits.items():
        path = proc_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in split_items:
                f.write(json.dumps(to_chat_record(item), ensure_ascii=False) + "\n")
        print(f"Wrote {len(split_items):>4} samples to {path}")

    print(f"\nDone. Images: {img_dir}\nSamples (from test split): {sample_dir}\nJSONL: {proc_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Floor Plan Dataset Preprocessing")
    parser.add_argument("--samples", type=int, default=120, help="Number of synthetic samples to generate")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default: <project>/data)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_dataset_splits(num_samples=args.samples, output_dir=args.outdir, seed=args.seed)
