"""Hull AABB (silhouette carving) and per-pixel depth → world-space point cloud.

The SDK encodes depth as ``1 - (log2(z+1) - log2(zNear+1)) / (log2(zFar+1) - log2(zNear+1))``
written to R==G==B; ``load_depth_map`` inverts that mapping.
"""
from __future__ import annotations

import pathlib
from typing import Optional

import cv2
import numpy as np

from .geometry import _load_camera

# Loose alpha cut-off used when computing the silhouette hull — soft splat edges
# are fine here because the hull only needs to bound the true surface.
_ALPHA_FOREGROUND_LOOSE = 0.5

# Strict alpha cut-off for depth unprojection — alpha-blended splat edges have
# unreliable depth and would paint radial streaks; 0.95 keeps solid interior.
_ALPHA_FOREGROUND_STRICT = 0.95

# Default voxel edge for cloud downsampling (5 mm).
_DEFAULT_VOXEL_SIZE_M = 0.005

def load_depth_map(
    path: pathlib.Path,
    depth_min: float,
    depth_max: float,
    *,
    dither: bool = True,
) -> np.ndarray:
    """Load a depth PNG and decode to float32 (H, W) in meters.

    ``dither`` adds ±½-band uniform jitter per pixel. With only 256 grayscale
    levels through a log curve, on a 1 m scene each byte step is ~18 mm —
    coarser than our 5 mm voxel, which would paint concentric shell bands on
    continuous surfaces. Jitter smears each pixel inside its band so bands
    overlap in world space; the mean is unchanged so surface position is preserved.
    """
    bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgra is None:
        raise IOError(f"Cannot read depth PNG: {path}")
    if bgra.ndim == 2:
        bgra = cv2.cvtColor(bgra, cv2.COLOR_GRAY2BGRA)
    elif bgra.shape[2] == 3:
        bgra = cv2.cvtColor(bgra, cv2.COLOR_BGR2BGRA)

    # cv2 returns BGRA. The SDK writes R==G==B so any channel works; use red.
    norm = bgra[..., 2].astype(np.float32) / 255.0
    log_depth = 1.0 - norm
    L_min = np.log2(depth_min + 1.0).astype(np.float32)
    L_max = np.log2(depth_max + 1.0).astype(np.float32)
    L = log_depth * (L_max - L_min) + L_min
    depth = (np.exp2(L) - 1.0).astype(np.float32)

    if dither:
        # Per-byte log step (in log2-space) over [depth_min, depth_max].
        log_step = (
            float(np.log2(depth_max + 1.0) - np.log2(depth_min + 1.0)) / 255.0
        )
        # Local band width at each pixel: dz/dL = ln(2) * (z + 1).
        band = (np.log(2.0) * (depth + 1.0) * log_step).astype(np.float32)
        rng = np.random.default_rng()
        jitter = (rng.random(depth.shape, dtype=np.float32) - 0.5) * band
        depth = depth + jitter

    return depth

def _flat_camera_pts(
    meta: dict,
    us: np.ndarray,
    vs: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Unproject pixel rays at given view-space depths to world (USD: camera looks along -Z, Y down)."""
    intrin = np.array(meta["camera_intrinsics"], dtype=np.float64)
    w2c = np.array(meta["world_to_camera"], dtype=np.float64)
    c2w = np.linalg.inv(w2c)
    fx = intrin[0, 0]
    fy = intrin[1, 1]
    cx = intrin[0, 2]
    cy = intrin[1, 2]

    x_cam = (us - cx) / fx * z
    y_cam = -(vs - cy) / fy * z
    z_cam = -z
    ones = np.ones_like(z)
    pts_cam = np.stack([x_cam, y_cam, z_cam, ones], axis=-1)
    return (c2w @ pts_cam.T).T[:, :3]


def unproject_depth_to_world(
    depth: np.ndarray,
    meta: dict,
    alpha_mask: np.ndarray,
    max_depth: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a depth map → ``(pts_world (N,3), pixel_index (N,2))`` keyed by the alpha mask.

    The pixel index lets the caller sample the colour frame at the same locations to attach RGB."""
    H, W = depth.shape
    if alpha_mask.shape != (H, W):
        raise ValueError(f"alpha_mask {alpha_mask.shape} != depth {depth.shape}")

    keep = alpha_mask & np.isfinite(depth) & (depth > 0.0)
    if max_depth is not None:
        keep = keep & (depth <= max_depth)

    if not keep.any():
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)

    vs, us = np.where(keep)
    z = depth[vs, us].astype(np.float64)
    pts_world = _flat_camera_pts(meta, us.astype(np.float64), vs.astype(np.float64), z)
    pix = np.column_stack([us.astype(np.int32), vs.astype(np.int32)])
    return pts_world, pix

def compute_hull_aabb(
    frames: list[dict],
    cam_stride: int = 1,
    alpha_threshold: float = _ALPHA_FOREGROUND_LOOSE,
    percentile_trim: float = 1.0,
    aabb_pad: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Coarse world-space AABB from alpha silhouettes (constant view-depth back-projection).

    Loose but never wrong-side of the true surface — used as a spatial filter for the depth cloud."""
    all_pts: list[np.ndarray] = []
    for frame in frames[::cam_stride]:
        color = cv2.imread(str(frame["color"]), cv2.IMREAD_UNCHANGED)
        if color is None or color.ndim != 3 or color.shape[2] < 4:
            continue
        meta = _load_camera(frame["camera"])
        fg = (color[:, :, 3].astype(np.float32) / 255.0) >= alpha_threshold
        H, W = fg.shape
        # 2-pixel stride: hull is loose anyway, so half the rays is plenty.
        uu, vv = np.meshgrid(np.arange(0, W, 2), np.arange(0, H, 2))
        fg_s = fg[vv, uu]
        if fg_s.sum() == 0:
            continue
        c2w = np.linalg.inv(np.array(meta["world_to_camera"], dtype=np.float64))
        cam_pos = (c2w @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
        z = np.full(int(fg_s.sum()), float(np.linalg.norm(cam_pos)))
        pts = _flat_camera_pts(
            meta,
            uu[fg_s].astype(np.float64),
            vv[fg_s].astype(np.float64),
            z,
        )
        all_pts.append(pts)

    if not all_pts:
        raise ValueError("Hull AABB: no foreground points.")

    pts_all = np.concatenate(all_pts, axis=0)
    lo = np.percentile(pts_all, percentile_trim, axis=0)
    hi = np.percentile(pts_all, 100.0 - percentile_trim, axis=0)
    pad = (hi - lo) * aabb_pad
    return lo - pad, hi + pad, {"world_min": lo.tolist(), "world_max": hi.tolist()}

def _voxel_downsample(
    pts: np.ndarray,
    rgbs: np.ndarray,
    voxel_size: float,
    frame_indices: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Returns ``(pts, rgbs, pix_counts, cam_counts)``.

    ``cam_counts`` (distinct source frames per voxel) is the multi-view-consensus signal: it
    separates real surface (many cameras converging) from splat-center streaks (one camera,
    many ray pixels in the same voxel neighbourhood) that pix_count alone can't tell apart.
    """
    if pts.shape[0] == 0:
        z = np.zeros(0, dtype=np.int64)
        return pts, rgbs, z, (z if frame_indices is not None else None)
    idx = np.floor(pts / voxel_size).astype(np.int64)
    _, inv, pix_counts = np.unique(idx, axis=0, return_inverse=True, return_counts=True)
    n = pix_counts.shape[0]
    p_out = np.zeros((n, 3))
    r_out = np.zeros((n, 3))
    np.add.at(p_out, inv, pts)
    np.add.at(r_out, inv, rgbs.astype(np.float64))
    p_out /= pix_counts[:, None]
    r_out /= pix_counts[:, None]

    cam_counts: Optional[np.ndarray] = None
    if frame_indices is not None:
        # Distinct cameras per voxel: dedup (voxel_id, frame_idx) pairs, then
        # bin by voxel_id.
        pair = np.stack([inv, frame_indices.astype(np.int64)], axis=1)
        unique_pairs = np.unique(pair, axis=0)
        cam_counts = np.bincount(unique_pairs[:, 0], minlength=n)

    return p_out, r_out.astype(np.uint8), pix_counts, cam_counts

def build_depth_cloud(
    frames: list[dict],
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    alpha_threshold: float = _ALPHA_FOREGROUND_STRICT,
    voxel_size: float = _DEFAULT_VOXEL_SIZE_M,
    depth_max: Optional[float] = None,
    min_unique_cameras: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject per-frame depth → world, AABB-filter, voxel-downsample, consensus-filter.

    ``min_unique_cameras`` is the consensus filter — voxels with hits from N *distinct* frames
    are real surface; one camera shooting many pixels along a streak hits one voxel many times
    but with cam_count=1.
    """
    all_pts: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    all_cam: list[np.ndarray] = []
    n_skipped = 0

    for i, frame in enumerate(frames):
        depth_path = frame.get("depth")
        if depth_path is None:
            n_skipped += 1
            continue

        bgra = cv2.imread(str(frame["color"]), cv2.IMREAD_UNCHANGED)
        if bgra is None or bgra.shape[2] < 4:
            n_skipped += 1
            continue

        meta = _load_camera(frame["camera"])
        dmin = float(meta.get("depth_min", 0.01))
        dmax = float(meta.get("depth_max", 1000.0))

        try:
            depth = load_depth_map(depth_path, dmin, dmax)
        except IOError as exc:
            print(f"[depth] frame {i}: {exc}")
            n_skipped += 1
            continue

        alpha = bgra[:, :, 3].astype(np.float32) / 255.0
        fg = alpha >= alpha_threshold

        pts, pix = unproject_depth_to_world(depth, meta, fg, max_depth=depth_max)
        if pts.shape[0] == 0:
            continue

        in_box = (pts >= aabb_min).all(axis=1) & (pts <= aabb_max).all(axis=1)
        if not in_box.any():
            continue
        pts = pts[in_box]
        pix = pix[in_box]

        # cv2 reads BGR; reverse the channel slice to RGB.
        rgb = bgra[pix[:, 1], pix[:, 0], 2::-1].astype(np.uint8)

        all_pts.append(pts)
        all_rgb.append(rgb)
        all_cam.append(np.full(pts.shape[0], i, dtype=np.int32))

    if not all_pts:
        raise ValueError(
            f"build_depth_cloud: no points produced "
            f"({len(frames)} frames, {n_skipped} skipped)"
        )

    pts_all = np.concatenate(all_pts, axis=0)
    rgb_all = np.concatenate(all_rgb, axis=0)
    cam_all = np.concatenate(all_cam, axis=0)
    print(f"[depth] Raw: {pts_all.shape[0]:,} pts from {len(frames) - n_skipped} frames")

    pts_all, rgb_all, _pix_counts, cam_counts = _voxel_downsample(
        pts_all, rgb_all, voxel_size, frame_indices=cam_all
    )
    print(f"[depth] After voxel dedup: {pts_all.shape[0]:,} (voxel {voxel_size} m)")

    if min_unique_cameras > 1 and pts_all.shape[0] > 0 and cam_counts is not None:
        keep = cam_counts >= min_unique_cameras
        pts_all = pts_all[keep]
        rgb_all = rgb_all[keep]
        print(
            f"[depth] After consensus filter (>= {min_unique_cameras} unique cameras/voxel): "
            f"{pts_all.shape[0]:,} ({100 * float(keep.mean()):.1f}% retained)"
        )

    return pts_all, rgb_all
