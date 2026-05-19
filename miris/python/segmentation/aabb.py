"""AABB fitting + filtering: ``fit_all_aabbs`` (with optional DBSCAN), ``validate_aabbs_by_reprojection``, ``resolve_overlapping_aabbs``."""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from .geometry import (
    CameraCache,
    _UnionFind,
    _aabb_corners,
    _base_label,
)

def _fit_aabb(pts: np.ndarray, percentile_trim: float = 0.5) -> dict:
    lo = np.percentile(pts, percentile_trim, axis=0)
    hi = np.percentile(pts, 100.0 - percentile_trim, axis=0)
    return {
        "world_min": lo.tolist(),
        "world_max": hi.tolist(),
        "center": ((lo + hi) / 2.0).tolist(),
        "size": (hi - lo).tolist(),
        "num_points": int(pts.shape[0]),
    }

def _dbscan_keep_largest(
    pts_label: np.ndarray,
    eps: float,
    min_samples: int,
    name: str,
) -> Optional[np.ndarray]:
    """DBSCAN-cluster ``pts_label``; return the largest dense cluster, or ``None`` if all are noise."""
    from sklearn.cluster import DBSCAN

    clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts_label)
    valid = clusters >= 0
    if not valid.any():
        print(
            f"  [dbscan] '{name}' dropped — all {pts_label.shape[0]} pts are noise "
            f"(eps={eps}, min_samples={min_samples})"
        )
        return None
    cluster_ids, cluster_counts = np.unique(clusters[valid], return_counts=True)
    biggest = int(cluster_ids[int(cluster_counts.argmax())])
    kept = pts_label[clusters == biggest]
    print(
        f"  [dbscan] '{name}' {pts_label.shape[0]:,} → {kept.shape[0]:,} pts "
        f"(kept largest of {len(cluster_ids)} clusters)"
    )
    return kept

def fit_all_aabbs(
    pts: np.ndarray,
    label_ids: np.ndarray,
    label_names: list[str],
    percentile_trim: float = 0.5,
    dbscan_eps: Optional[float] = None,
    dbscan_min_samples: Optional[int] = None,
    max_box_size: float = 0.0,
) -> tuple[dict, dict]:
    """One global AABB + one AABB per instance.

    With both ``dbscan_eps`` and ``dbscan_min_samples`` set, points per label are clustered
    first and only the largest dense cluster survives — this drops residual SAM2 leakage
    (table pixels around the object form sparse clusters that lose the size contest).
    ``max_box_size > 0`` additionally drops instances whose largest AABB dimension exceeds it.
    """
    use_dbscan = dbscan_eps is not None and dbscan_min_samples is not None

    global_aabb = _fit_aabb(pts, percentile_trim)
    label_aabbs: dict[str, dict] = {}

    for lid, name in enumerate(label_names):
        mask = label_ids == lid
        if mask.sum() < 4:
            continue
        pts_label = pts[mask]

        if use_dbscan and pts_label.shape[0] >= dbscan_min_samples:
            kept = _dbscan_keep_largest(
                pts_label, float(dbscan_eps), int(dbscan_min_samples), name,
            )
            if kept is None:
                continue
            pts_label = kept

        aabb = _fit_aabb(pts_label, percentile_trim)
        if max_box_size > 0 and max(aabb["size"]) > max_box_size:
            print(
                f"  [filter] '{name}' dropped — max dim "
                f"{max(aabb['size']):.2f}m > {max_box_size}m"
            )
            continue
        label_aabbs[name] = aabb

    return global_aabb, label_aabbs


def _iou_2d(a, b) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter)


def _project_aabb(aabb: dict, P: np.ndarray, H: int, W: int):
    """2-D AABB ``[x1, y1, x2, y2]`` of the 8 projected corners; ``None`` if behind camera or zero-area."""
    corners_h = np.hstack([_aabb_corners(aabb), np.ones((8, 1))])   # 8 × 4
    proj = (corners_h @ P.T).T                                      # 3 × 8
    if (proj[2] <= 0).all():
        return None
    valid = proj[2] > 0
    proj = proj[:, valid]
    proj /= proj[2:3, :]
    xs, ys = proj[0], proj[1]
    x1, y1 = float(np.clip(xs.min(), 0, W)), float(np.clip(ys.min(), 0, H))
    x2, y2 = float(np.clip(xs.max(), 0, W)), float(np.clip(ys.max(), 0, H))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _per_frame_proj_ious(
    aabb: dict,
    base_label: str,
    cache: CameraCache,
    all_dets: list[list[dict]],
) -> list[float]:
    """Per-frame best-IoU of the AABB's projection vs same-label 2-D dets; frames with no positive IoU are skipped."""
    ious: list[float] = []
    for P, dets, (H, W) in zip(cache.Ps, all_dets, cache.img_sizes):
        proj_box = _project_aabb(aabb, P, H, W)
        if proj_box is None:
            continue
        best = max(
            (_iou_2d(proj_box, det["box_xyxy"])
             for det in dets if det["label"] == base_label),
            default=0.0,
        )
        if best > 0:
            ious.append(best)
    return ious

def validate_aabbs_by_reprojection(
    label_aabbs: dict,
    all_dets: list[list[dict]],
    cache: CameraCache,
    min_proj_matches: int = 3,
    proj_iou_thresh: float = 0.1,
) -> dict:
    """Drop boxes whose projection misses same-label 2-D dets in fewer than ``min_proj_matches`` frames at IoU ≥ ``proj_iou_thresh``."""
    print("\n── Step 5b: Validating 3D boxes by reprojection ─────────────────────")
    valid_aabbs: dict = {}
    n_frames = len(cache.frames)
    for name, aabb in label_aabbs.items():
        ious = _per_frame_proj_ious(aabb, _base_label(name), cache, all_dets)
        frame_matches = sum(1 for iou in ious if iou >= proj_iou_thresh)
        status = "✓" if frame_matches >= min_proj_matches else "✗ DROPPED"
        print(f"  '{name}': {frame_matches}/{n_frames} frame matches  {status}")
        if frame_matches >= min_proj_matches:
            valid_aabbs[name] = aabb

    print(
        f"  Kept {len(valid_aabbs)}/{len(label_aabbs)} instances "
        f"(≥{min_proj_matches} frame matches, IoU≥{proj_iou_thresh})"
    )
    return valid_aabbs


def _aabb_overlap(a: dict, b: dict) -> bool:
    a_min, a_max = np.asarray(a["world_min"]), np.asarray(a["world_max"])
    b_min, b_max = np.asarray(b["world_min"]), np.asarray(b["world_max"])
    return bool(np.all(np.minimum(a_max, b_max) > np.maximum(a_min, b_min)))

def resolve_overlapping_aabbs(
    label_aabbs: dict,
    all_dets: list[list[dict]],
    cache: CameraCache,
) -> dict:
    """For each 3-D-overlapping same-label box group, keep the one with the highest mean reprojection IoU."""
    if len(label_aabbs) <= 1:
        return label_aabbs

    by_label: dict = defaultdict(list)
    for name in label_aabbs:
        by_label[_base_label(name)].append(name)

    print("\n── Step 5c: Resolving overlapping same-label boxes ──────────────────")

    dropped: set[str] = set()
    for base_label, names in by_label.items():
        if len(names) < 2:
            continue

        uf = _UnionFind(names)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if _aabb_overlap(label_aabbs[names[i]], label_aabbs[names[j]]):
                    uf.union(names[i], names[j])

        for members in uf.components():
            if len(members) < 2:
                continue
            scores = {
                n: float(np.mean(_per_frame_proj_ious(label_aabbs[n], base_label, cache, all_dets) or [0.0]))
                for n in members
            }
            best = max(scores, key=scores.get)
            print(f"  [{base_label}] overlap group {sorted(members)}:")
            for n in members:
                tag = "✓ KEPT" if n == best else "✗ DROPPED"
                print(f"    '{n}'  mean_iou={scores[n]:.3f}  {tag}")
            dropped.update(n for n in members if n != best)

    result = {n: a for n, a in label_aabbs.items() if n not in dropped}

    if not dropped:
        print("  No overlapping same-label boxes found.")
    else:
        print(
            f"  {len(label_aabbs) - len(result)} box(es) removed, "
            f"{len(result)} remaining."
        )
    return result
