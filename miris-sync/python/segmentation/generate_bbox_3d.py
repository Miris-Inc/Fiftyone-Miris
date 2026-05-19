import json
import pathlib
import re
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# BBox helpers
# ──────────────────────────────────────────────────────────────────────────────


def _discover_frames(dataset_dir: pathlib.Path) -> list[dict[str, pathlib.Path]]:
    """Return list of dicts with keys 'color', 'camera' for each frame."""
    frames: dict[str, dict[str, pathlib.Path]] = {}
    for f in sorted(dataset_dir.iterdir()):
        m = re.match(r"^(.+)_(color|camera)\.(png|json)$", f.name)
        if not m:
            continue
        stem, kind = m.group(1), m.group(2)
        if stem not in frames:
            frames[stem] = {}
        frames[stem][kind] = f
    complete = [v for v in frames.values() if "color" in v and "camera" in v]
    complete.sort(key=lambda d: str(d["camera"]))
    return complete


def _read_frame(frame: dict[str, pathlib.Path]) -> tuple[np.ndarray, dict]:
    """Return (color_bgra uint8 HxWx4, camera_meta dict)."""
    import cv2

    color = cv2.imread(str(frame["color"]), cv2.IMREAD_UNCHANGED)
    with open(frame["camera"]) as f:
        meta = json.load(f)
    return color, meta


def _foreground_mask(color_bgra: np.ndarray, alpha_threshold: float) -> np.ndarray:
    """Bool (H, W) mask from the color PNG's alpha channel."""
    if color_bgra.shape[2] < 4:
        return np.zeros(color_bgra.shape[:2], dtype=bool)
    return color_bgra[:, :, 3].astype(np.float32) / 255.0 >= alpha_threshold


def _backproject_foreground(
    meta: dict,
    fg: np.ndarray,
    pixel_stride: int,
) -> Optional[np.ndarray]:
    """
    Back-project foreground pixels to world space using flat-depth silhouette
    carving (camera-to-origin distance as constant depth).  USD convention:
    camera looks along -Z; image Y points down.

    Returns (N, 3) world-space points or None.
    """
    intrin = np.array(meta["camera_intrinsics"], dtype=np.float64)
    w2c = np.array(meta["world_to_camera"], dtype=np.float64)
    c2w = np.linalg.inv(w2c)
    fx = intrin[0, 0]
    fy = intrin[1, 1]
    cx = intrin[0, 2]
    cy = intrin[1, 2]

    H, W = fg.shape
    ys = np.arange(0, H, pixel_stride)
    xs = np.arange(0, W, pixel_stride)
    uu, vv = np.meshgrid(xs, ys)
    fg_s = fg[vv, uu]
    if fg_s.sum() == 0:
        return None

    u = uu[fg_s].astype(np.float64)
    v = vv[fg_s].astype(np.float64)

    cam_pos = (c2w @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
    z = np.full(u.shape, np.linalg.norm(cam_pos))

    x_cam = (u - cx) / fx * z
    y_cam = -(v - cy) / fy * z
    z_cam = -z
    ones = np.ones_like(z)
    pts_cam = np.stack([x_cam, y_cam, z_cam, ones], axis=-1)
    pts_world = (c2w @ pts_cam.T).T[:, :3]
    return pts_world


def _fit_aabb(pts: np.ndarray, percentile_trim: float = 1.0) -> dict:
    lo = np.percentile(pts, percentile_trim, axis=0)
    hi = np.percentile(pts, 100.0 - percentile_trim, axis=0)
    return {
        "world_min": lo.tolist(),
        "world_max": hi.tolist(),
        "center": ((lo + hi) / 2.0).tolist(),
        "size": (hi - lo).tolist(),
        "num_points": int(pts.shape[0]),
    }


def run_bbox_pipeline(
    frames: list[dict],
    base_dir: Optional[pathlib.Path] = None,
    cam_stride: int = 1,
    pixel_stride: int = 2,
    alpha_threshold: float = 0.5,
    percentile_trim: float = 1.0,
):
    """
    Generator — yields (progress: float, label: str) per processed frame.
    Returns a result dict via StopIteration.value when exhausted, or None if
    no foreground points were found.

    Each entry in `frames` must have 'color_filename' and 'camera_json_filename'
    as paths relative to `base_dir` (or absolute if base_dir is None).
    """
    selected = frames[::cam_stride]
    total = len(selected)
    all_pts: list[np.ndarray] = []
    skipped = 0

    for i, frame_data in enumerate(selected):
        frame = {
            "color":  base_dir / frame_data["color_filename"] if base_dir else pathlib.Path(frame_data["color_filename"]),
            "camera": base_dir / frame_data["camera_json_filename"] if base_dir else pathlib.Path(frame_data["camera_json_filename"]),
        }
        try:
            color, meta = _read_frame(frame)
            fg = _foreground_mask(color, alpha_threshold)
            pts = _backproject_foreground(meta, fg, pixel_stride)
            if pts is not None and pts.shape[0] > 0:
                all_pts.append(pts)
        except Exception:
            skipped += 1

        pct = round((i + 1) / total * 100)
        yield (i + 1) / total, f"Generating labels from Miris stream...{pct}%"

    if not all_pts:
        return None

    pts_all = np.concatenate(all_pts, axis=0)
    bbox = _fit_aabb(pts_all, percentile_trim)
    return {
        "world_min":  bbox["world_min"],
        "world_max":  bbox["world_max"],
        "source":     "visual_hull",
        "num_points": bbox["num_points"],
        "num_frames": total,
        "skipped":    skipped,
    }
