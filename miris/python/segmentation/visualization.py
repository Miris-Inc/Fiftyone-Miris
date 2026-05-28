"""Debug outputs gated behind the pipeline's ``visualize`` flag: SAM2 mask overlays, projected-box PNGs, PLY exports."""
from __future__ import annotations

import pathlib
import re

import cv2
import numpy as np

from .geometry import (
    _BOX_EDGES,
    _aabb_corners,
    _base_label,
    _colour,
    _load_and_make_P,
    _load_rgb_np,
    _PALETTE_BGR,
    _project_cv,
)


_PLY_VERTEX_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("r", "u1"), ("g", "u1"), ("b", "u1"),
])
_PLY_EDGE_DTYPE = np.dtype([("v1", "<i4"), ("v2", "<i4")])


def _draw_box(img, aabb: dict, P: np.ndarray, colour, thickness: int = 2, label: str | None = None):
    pts2d: list[tuple[int, int] | None] = []
    for c in _aabb_corners(aabb):
        res = _project_cv(c, P)
        pts2d.append((int(round(res[0])), int(round(res[1]))) if res else None)
    for i, j in _BOX_EDGES:
        if pts2d[i] is not None and pts2d[j] is not None:
            cv2.line(img, pts2d[i], pts2d[j], colour, thickness, cv2.LINE_AA)
    if label:
        res = _project_cv(np.array(aabb["center"]), P)
        if res:
            cx, cy = int(round(res[0])), int(round(res[1]))
            cv2.putText(img, label, (cx, cy - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.60, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(img, label, (cx, cy - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.60, colour, 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# SAM2 mask overlays
# ──────────────────────────────────────────────────────────────────────────────

def save_sam2_mask_debug(
    frame_list: list[dict],
    all_dets: list[list[dict]],
    label_maps: list[np.ndarray],
    output_dir: pathlib.Path,
) -> None:
    """Per-frame RGB overlay where each SAM2 mask is coloured by instance_id."""
    mask_dir = output_dir / "sam2_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    all_instances = sorted({
        d.get("instance_id", d["label"])
        for dets in all_dets for d in dets
    })
    inst_colour = {name: _colour(i) for i, name in enumerate(all_instances)}

    for frame, dets, lmap in zip(frame_list, all_dets, label_maps):
        rgb = _load_rgb_np(frame["color"])
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
        overlay = bgr.copy()

        # First pass: paint mask fill. Done in a separate loop because all
        # fills must blend with the original bgr before contours/boxes go on top.
        det_colours = [
            inst_colour.get(det.get("instance_id", det["label"]), (128, 128, 128))
            for det in dets
        ]
        for det_idx, colour in enumerate(det_colours):
            mask = lmap == det_idx
            for ch, val in enumerate(colour):
                overlay[:, :, ch] = np.where(mask, val, overlay[:, :, ch])

        vis = cv2.addWeighted(overlay, 0.45, bgr, 0.55, 0).astype(np.uint8)

        for det_idx, (det, colour) in enumerate(zip(dets, det_colours)):
            mask8 = ((lmap == det_idx) * 255).astype(np.uint8)
            cnts, _ = cv2.findContours(mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, colour, 1)
            x1, y1, x2, y2 = [int(v) for v in det["box_xyxy"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
            iid = det.get("instance_id", det["label"])
            text = f"{iid} {det['score']:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, cv2.FILLED)
            cv2.putText(vis, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        stem = re.sub(r"_camera$", "", frame["camera"].stem)
        cv2.imwrite(str(mask_dir / f"{stem}_sam2.png"), vis)

    print(f"[debug] SAM2 mask overlays → {mask_dir}  ({len(frame_list)} images)")


# ──────────────────────────────────────────────────────────────────────────────
# Box overlays on per-frame images
# ──────────────────────────────────────────────────────────────────────────────

def visualise_boxes(
    frames: list[dict],
    label_aabbs: dict,
    label_names: list[str],
    output_dir: pathlib.Path,
    cam_stride: int = 1,
) -> None:
    """Render every label AABB as a wireframe projected onto each frame."""
    vis_dir = output_dir / "visualised"
    vis_dir.mkdir(parents=True, exist_ok=True)
    selected = frames[::cam_stride]
    label_to_lid = {name: lid for lid, name in enumerate(label_names)}

    print(f"\n── Step 6: Visualising on {len(selected)} frames ─────────────────")
    for idx, frame in enumerate(selected):
        bgr = cv2.cvtColor(_load_rgb_np(frame["color"]), cv2.COLOR_RGB2BGR)
        P, _meta = _load_and_make_P(frame)
        for full_name, aabb in label_aabbs.items():
            lid = label_to_lid.get(_base_label(full_name), 0)
            _draw_box(bgr, aabb, P, _colour(lid), thickness=2, label=full_name)

        stem = re.sub(r"_camera$", "", frame["camera"].stem)
        cv2.imwrite(str(vis_dir / f"{stem}_boxes.png"), bgr)
        if (idx + 1) % 10 == 0 or idx == len(selected) - 1:
            print(f"  [{idx+1:3d}/{len(selected)}]")
    print(f"[vis] Done → {vis_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# PLY outputs
# ──────────────────────────────────────────────────────────────────────────────

def _write_ply(
    path: pathlib.Path,
    verts: np.ndarray,
    colours: np.ndarray,
    edges: np.ndarray | None = None,
) -> None:
    """Binary little-endian PLY — ``verts`` (N,3) float, ``colours`` (N,3) uint8 RGB, ``edges`` (E,2) int32 (optional)."""
    n_v = verts.shape[0]
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n_v}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    if edges is not None:
        header += [
            f"element edge {edges.shape[0]}",
            "property int vertex1",
            "property int vertex2",
        ]
    header.append("end_header\n")

    vert_arr = np.empty(n_v, dtype=_PLY_VERTEX_DTYPE)
    vert_arr["x"], vert_arr["y"], vert_arr["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
    vert_arr["r"], vert_arr["g"], vert_arr["b"] = colours[:, 0], colours[:, 1], colours[:, 2]

    with open(path, "wb") as f:
        f.write("\n".join(header).encode())
        vert_arr.tofile(f)
        if edges is not None:
            edge_arr = np.empty(edges.shape[0], dtype=_PLY_EDGE_DTYPE)
            edge_arr["v1"], edge_arr["v2"] = edges[:, 0], edges[:, 1]
            edge_arr.tofile(f)


def save_ply(
    pts: np.ndarray,
    rgb: np.ndarray,
    label_ids: np.ndarray,
    label_names: list[str],
    path: pathlib.Path,
) -> None:
    """Coloured point cloud — palette colour per instance, dimmed original RGB for unlabelled."""
    n = pts.shape[0]
    # Vectorised palette lookup; BGR → RGB swap matches the rest of the file.
    palette = np.array(_PALETTE_BGR, dtype=np.uint8)[:, ::-1]
    lid = label_ids.astype(np.int64)
    has_label = lid >= 0
    colours = np.where(
        has_label[:, None],
        palette[np.mod(lid, len(palette))],
        (rgb.astype(np.float32) * 0.35).astype(np.uint8),
    )
    _write_ply(path, pts, colours)
    print(f"[out] PLY cloud : {path}  ({n:,} pts)")


def save_boxes_ply(
    global_aabb: dict,
    label_aabbs: dict,
    label_names: list[str],
    path: pathlib.Path,
) -> None:
    """Wireframe PLY of the global AABB (white) plus each label AABB (palette colour)."""
    label_to_lid = {name: lid for lid, name in enumerate(label_names)}
    box_specs = [(global_aabb, (255, 255, 255))]
    for full_name, aabb in label_aabbs.items():
        lid = label_to_lid.get(_base_label(full_name), 0)
        box_specs.append((aabb, _colour(lid)[::-1]))   # BGR → RGB

    all_verts: list[np.ndarray] = []
    all_colours: list[np.ndarray] = []
    all_edges: list[np.ndarray] = []
    offset = 0
    edge_template = np.asarray(_BOX_EDGES, dtype=np.int32)
    for aabb, rgb_col in box_specs:
        all_verts.append(_aabb_corners(aabb))
        all_colours.append(np.tile(np.asarray(rgb_col, dtype=np.uint8), (8, 1)))
        all_edges.append(edge_template + offset)
        offset += 8

    verts = np.concatenate(all_verts, axis=0)
    colours = np.concatenate(all_colours, axis=0)
    edges = np.concatenate(all_edges, axis=0)

    _write_ply(path, verts, colours, edges)
    print(f"[out] PLY boxes : {path}  ({len(box_specs)} boxes)")
