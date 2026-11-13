"""
Visualization for 2D Floor Plan Detection + downstream 2D-to-3D bridge.

1. 2D overlay of detected boxes and labels on the floor plan image.
2. 3D extrusion: walls are extruded to `wall_height`, door/window openings are cut out of the walls
   (lintels above doors, sills + lintels around windows), rooms become floor slabs.

Usage:
    python src/visualize.py --image data/samples/sample_01.png --json data/samples/sample_01.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw, ImageFont

from schema import PROJECT_ROOT, resolve_path, wall_bbox

CLASS_COLORS = {
    "rooms": (46, 204, 113),
    "walls": (52, 73, 94),
    "doors": (230, 126, 34),
    "windows": (52, 152, 219),
}
ROOM_COLORS = {
    "living_room": "#f6d7a7", "bedroom": "#c9e4de", "kitchen": "#faedcb",
    "bathroom": "#c6def1", "hallway": "#e2e2df", "dining_room": "#f2c6de",
}
DOOR_HEIGHT, SILL_HEIGHT, WINDOW_TOP = 2.1, 0.9, 2.1


def _font(size=14):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _item_box(item: Dict, category: str):
    box = wall_bbox(item) if category == "walls" else item.get("bbox")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            return [float(v) for v in box]
        except (TypeError, ValueError):
            return None
    return None


def draw_2d_detections(img_path: str, json_data: dict, output_path: str, coord_max: int = 1000):
    """Overlay boxes + labels (coordinates normalized to 0..coord_max) on the image."""
    img = Image.open(resolve_path(img_path)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    font = _font(14)

    # walls first so smaller elements stay visible on top
    for category in ["walls", "rooms", "doors", "windows"]:
        color = CLASS_COLORS[category]
        for item in json_data.get(category, []) or []:
            if not isinstance(item, dict):
                continue
            box = _item_box(item, category)
            if box is None:
                continue
            ymin, xmin, ymax, xmax = box
            x0, x1 = int(xmin / coord_max * w), int(xmax / coord_max * w)
            y0, y1 = int(ymin / coord_max * h), int(ymax / coord_max * h)
            fill_alpha = 25 if category in ("rooms", "walls") else 70
            draw.rectangle([x0, y0, x1, y1], outline=(*color, 255), fill=(*color, fill_alpha),
                           width=1 if category == "walls" else 3)
            if category == "walls":
                continue
            label = str(item.get("type", category[:-1]))
            tw = int(draw.textlength(label, font=font)) + 6
            ty = y0 - 18 if y0 >= 18 else y0 + 2       # keep the tag inside the image
            draw.rectangle([x0, ty, x0 + tw, ty + 17], fill=(*color, 230))
            draw.text((x0 + 3, ty + 1), label, fill=(255, 255, 255, 255), font=font)

    Image.alpha_composite(img, overlay).convert("RGB").save(output_path)
    print(f"Saved 2D detection overlay to {output_path}")


def _box_faces(x0, x1, y0, y1, z0, z1):
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]


def _overlap(a0, a1, b0, b1):
    return max(a0, b0), min(a1, b1)


def build_wall_segments(walls: List[List[float]], openings: List[Tuple[List[float], str]],
                        wall_height: float) -> List[Tuple[float, float, float, float, float, float]]:
    """
    Split every wall box (meters, x0,x1,y0,y1) along its long axis around the openings that intersect it.
    Returns solid boxes (x0, x1, y0, y1, z0, z1).
    """
    solids = []
    for (x0, x1, y0, y1) in walls:
        horizontal = (x1 - x0) >= (y1 - y0)
        lo, hi = (x0, x1) if horizontal else (y0, y1)
        cuts = []
        for (ox0, ox1, oy0, oy1), kind in openings:
            # opening must intersect the wall's cross-section
            cx0, cx1 = _overlap(x0, x1, ox0, ox1)
            cy0, cy1 = _overlap(y0, y1, oy0, oy1)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            a, b = (cx0, cx1) if horizontal else (cy0, cy1)
            cuts.append((a, b, kind))
        cuts.sort()

        def seg(a, b, z0, z1):
            if b - a <= 1e-6 or z1 <= z0:
                return
            if horizontal:
                solids.append((a, b, y0, y1, z0, z1))
            else:
                solids.append((x0, x1, a, b, z0, z1))

        cursor = lo
        for a, b, kind in cuts:
            a, b = max(a, cursor), max(b, cursor)
            seg(cursor, a, 0, wall_height)                       # full-height wall before the opening
            if kind == "door":
                seg(a, b, DOOR_HEIGHT, wall_height)              # lintel
            else:
                seg(a, b, 0, SILL_HEIGHT)                        # sill
                seg(a, b, WINDOW_TOP, wall_height)               # lintel
            cursor = max(cursor, b)
        seg(cursor, hi, 0, wall_height)
    return solids


def render_3d_extrusion(json_data: dict, output_path: str, wall_height: float = 2.8,
                        plan_size_m: float = 10.0, coord_max: int = 1000):
    """Extrude walls, cut door/window openings, draw room slabs. Plan is scaled to plan_size_m meters."""
    s = plan_size_m / coord_max

    def to_m(box):  # [ymin, xmin, ymax, xmax] -> (x0, x1, y0, y1) with Y flipped (image y points down)
        ymin, xmin, ymax, xmax = box
        return xmin * s, xmax * s, plan_size_m - ymax * s, plan_size_m - ymin * s

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    for r in json_data.get("rooms", []) or []:
        box = _item_box(r, "rooms") if isinstance(r, dict) else None
        if box is None:
            continue
        x0, x1, y0, y1 = to_m(box)
        ax.add_collection3d(Poly3DCollection(
            [[(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)]], alpha=0.55,
            facecolor=ROOM_COLORS.get(r.get("type"), "#dddddd"), edgecolor="#7f8c8d"))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, 0.02, str(r.get("type", "")).replace("_", " "),
                fontsize=8, ha="center", color="#2c3e50")

    openings = []
    for cat, kind in (("doors", "door"), ("windows", "window")):
        for o in json_data.get(cat, []) or []:
            box = _item_box(o, cat) if isinstance(o, dict) else None
            if box is not None:
                openings.append((to_m(box), kind))

    for w in json_data.get("walls", []) or []:
        box = _item_box(w, "walls") if isinstance(w, dict) else None
        if box is None:
            continue
        color = "#2c3e50" if w.get("type") == "exterior" else "#95a5a6"
        for (x0, x1, y0, y1, z0, z1) in build_wall_segments([to_m(box)], openings, wall_height):
            ax.add_collection3d(Poly3DCollection(_box_faces(x0, x1, y0, y1, z0, z1), alpha=0.9,
                                                 facecolor=color, edgecolor="#1a252f", linewidths=0.3))

    # glass panes in window openings
    for (x0, x1, y0, y1), kind in openings:
        if kind == "window":
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            if (x1 - x0) >= (y1 - y0):
                pane = [(x0, ym, SILL_HEIGHT), (x1, ym, SILL_HEIGHT), (x1, ym, WINDOW_TOP), (x0, ym, WINDOW_TOP)]
            else:
                pane = [(xm, y0, SILL_HEIGHT), (xm, y1, SILL_HEIGHT), (xm, y1, WINDOW_TOP), (xm, y0, WINDOW_TOP)]
            ax.add_collection3d(Poly3DCollection([pane], alpha=0.35, facecolor="#3498db", edgecolor="#2980b9"))

    ax.set_xlim(0, plan_size_m)
    ax.set_ylim(0, plan_size_m)
    ax.set_zlim(0, wall_height + 0.5)
    try:
        ax.set_box_aspect((plan_size_m, plan_size_m, (wall_height + 0.5) * 1.5))
    except AttributeError:
        pass
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Height (m)")
    ax.view_init(elev=38, azim=-60)
    plt.title("Downstream 2D-to-3D Extruded Model (doors/windows cut into walls)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved 3D extrusion preview to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Floor Plan Detection & 3D Bridge Visualizer")
    parser.add_argument("--image", type=str, default=str(PROJECT_ROOT / "data/samples/sample_01.png"))
    parser.add_argument("--json", type=str, default=str(PROJECT_ROOT / "data/samples/sample_01.json"))
    parser.add_argument("--out2d", type=str, default=str(PROJECT_ROOT / "data/samples/detection_overlay_01.png"))
    parser.add_argument("--out3d", type=str, default=str(PROJECT_ROOT / "data/samples/extruded_3d_preview_01.png"))
    parser.add_argument("--wall_height", type=float, default=2.8)
    args = parser.parse_args()

    with open(resolve_path(args.json), "r", encoding="utf-8") as f:
        data = json.load(f)
    for p in (args.out2d, args.out3d):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    draw_2d_detections(args.image, data, args.out2d)
    render_3d_extrusion(data, args.out3d, wall_height=args.wall_height)


if __name__ == "__main__":
    main()
