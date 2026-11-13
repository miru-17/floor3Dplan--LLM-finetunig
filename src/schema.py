"""
Floor Plan Schema, Prompt and Validation Module.

Single source of truth for:
  * the instruction prompt used for training AND inference (they must match),
  * the JSON target format (compact JSON, no markdown fences),
  * schema validation of model output,
  * robust JSON extraction from raw model text,
  * project-relative path resolution (so scripts work from any working directory).
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root = parent of the src/ folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROOM_TYPES = ["living_room", "bedroom", "kitchen", "bathroom", "hallway", "dining_room"]
WALL_TYPES = ["interior", "exterior"]
DOOR_TYPES = ["entrance", "interior", "sliding"]
CATEGORIES = ["rooms", "walls", "doors", "windows"]
EXPECTED_KEYS = set(CATEGORIES)

FLOORPLAN_PROMPT = """Analyze this 2D architectural floor plan image. Detect and classify all structural elements:
1. Rooms (living_room, bedroom, kitchen, bathroom, hallway, dining_room)
2. Walls (interior, exterior)
3. Doors (entrance, interior, sliding)
4. Windows

Return ONLY a valid JSON object (no markdown, no explanation). All coordinates are integers normalized to 0-1000.
Bounding boxes use [ymin, xmin, ymax, xmax]. Wall endpoints use [x, y].
Format:
{"image_dimensions": {"width": W, "height": H}, "rooms": [{"id": "room_1", "type": "bedroom", "bbox": [ymin, xmin, ymax, xmax]}], "walls": [{"id": "wall_1", "type": "exterior", "start": [x1, y1], "end": [x2, y2], "bbox": [ymin, xmin, ymax, xmax]}], "doors": [{"id": "door_1", "type": "interior", "bbox": [ymin, xmin, ymax, xmax]}], "windows": [{"id": "window_1", "bbox": [ymin, xmin, ymax, xmax]}]}"""


def to_target_text(ground_truth: Dict[str, Any]) -> str:
    """Serialize ground truth as the exact assistant target string (compact = fewer tokens)."""
    return json.dumps(ground_truth, separators=(",", ":"), ensure_ascii=False)


def resolve_path(path: str, base: Optional[Path] = None) -> str:
    """Resolve a (possibly relative) path: as given -> relative to base -> relative to project root."""
    if path.startswith(("file://", "http://", "https://", "data:")):
        return path
    p = Path(path)
    if p.is_absolute() and p.exists():
        return str(p)
    for root in [Path.cwd(), base, PROJECT_ROOT]:
        if root is not None and (Path(root) / p).exists():
            return str((Path(root) / p).resolve())
    return str(p)


def _is_valid_box(box: Any) -> bool:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box):
        return False
    ymin, xmin, ymax, xmax = box
    return ymin <= ymax and xmin <= xmax


def _is_valid_point(pt: Any) -> bool:
    return (
        isinstance(pt, (list, tuple)) and len(pt) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in pt)
    )


def validate_floorplan_json(data: Any, coord_max: int = 1000) -> Tuple[bool, List[str]]:
    """
    Validates a parsed JSON object against the 2D-to-3D floor plan schema.
    Returns (is_valid, error_messages).
    """
    errors: List[str] = []
    if not isinstance(data, dict):
        return False, ["Root output is not a JSON object/dict."]

    missing = EXPECTED_KEYS - set(data.keys())
    if missing:
        errors.append(f"Missing top-level keys: {sorted(missing)}")

    def check_range(box, label):
        if any(v < 0 or v > coord_max for v in box):
            errors.append(f"{label} has coordinates outside [0, {coord_max}].")

    for cat in CATEGORIES:
        items = data.get(cat, [])
        if not isinstance(items, list):
            errors.append(f"'{cat}' must be a list.")
            continue
        for idx, item in enumerate(items):
            label = f"{cat}[{idx}]"
            if not isinstance(item, dict):
                errors.append(f"{label} is not an object.")
                continue
            label = f"{cat}[{item.get('id', idx)}]"

            if cat == "walls":
                has_bbox = _is_valid_box(item.get("bbox"))
                has_pts = _is_valid_point(item.get("start")) and _is_valid_point(item.get("end"))
                if not (has_bbox or has_pts):
                    errors.append(f"{label} needs a valid 'bbox' or 'start'/'end' endpoints.")
                if has_bbox:
                    check_range(item["bbox"], label)
            else:
                if not _is_valid_box(item.get("bbox")):
                    errors.append(f"{label} missing valid 4-number 'bbox' [ymin, xmin, ymax, xmax].")
                else:
                    check_range(item["bbox"], label)

            if cat == "rooms" and "type" not in item:
                errors.append(f"{label} missing 'type'.")

    return (len(errors) == 0, errors)


def extract_json_from_response(text: str) -> Dict[str, Any]:
    """
    Extracts and parses a JSON object from raw model output.
    Handles markdown fences, leading/trailing chatter and (best-effort) truncated output.
    Raises ValueError if nothing parseable is found.
    """
    if not isinstance(text, str):
        raise ValueError("Model output is not a string.")
    text = text.strip()

    # 1) Markdown fenced block
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    candidates = [m.group(1).strip()] if m else []
    # 2) Whole text
    candidates.append(text)
    # 3) Outermost {...}
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not parse a JSON object from model output.")


def wall_bbox(wall: Dict[str, Any], default_thickness: int = 10) -> Optional[List[float]]:
    """Return a wall's bbox, deriving one from start/end endpoints if needed."""
    if _is_valid_box(wall.get("bbox")):
        return list(wall["bbox"])
    s, e = wall.get("start"), wall.get("end")
    if _is_valid_point(s) and _is_valid_point(e):
        half = default_thickness / 2
        xmin, xmax = min(s[0], e[0]), max(s[0], e[0])
        ymin, ymax = min(s[1], e[1]), max(s[1], e[1])
        if xmax - xmin < default_thickness:
            xmin, xmax = xmin - half, xmax + half
        if ymax - ymin < default_thickness:
            ymin, ymax = ymin - half, ymax + half
        return [ymin, xmin, ymax, xmax]
    return None
