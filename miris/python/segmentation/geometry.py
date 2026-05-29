"""Shared low-level helpers: camera math, projection, device pick, palette, CameraCache, Union-Find."""
from __future__ import annotations

import json
import os
import pathlib
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Miris uses the USD convention (camera looks along -Z, image Y points down).
# This flip converts a Miris world_to_camera matrix into OpenCV's convention
# before composing with the intrinsics.
_MIRIS_TO_CV = np.diag([1.0, -1.0, -1.0])

# Cap on the highest CUDA compute capability we auto-try; newer GPUs sometimes
# need a torch build matching their sm or crash on first launch.
_MAX_SM = 120

_DEFAULT_IMAGE_SIZE = (1080, 1920)

_PALETTE_BGR: list[tuple[int, int, int]] = [
    (75,  25,  230), (75,  180, 60),  (25,  225, 255), (200, 130,   0),
    (48,  130, 245), (180, 30,  145), (240, 240,  70), (230,  50,  240),
    (60,  245, 210), (212, 190, 250), (128, 128,   0), (255, 190,  220),
    (40,  110, 170), (200, 250, 255), (  0,   0, 128), (195, 255,  170),
    (  0, 128, 128), (180, 215, 255), (128,   0,   0), (128, 128,  128),
]

# Index pairs for the 12 edges of an axis-aligned box given 8 corners ordered
# as the Cartesian product of (min_x|max_x) × (min_y|max_y) × (min_z|max_z).
_BOX_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _colour(idx: int) -> tuple[int, int, int]:
    return _PALETTE_BGR[idx % len(_PALETTE_BGR)]


def _base_label(full_name: str) -> str:
    """'wheel_0' → 'wheel',  'front spoiler_1' → 'front spoiler',  'tire' → 'tire'."""
    parts = full_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return full_name


# ──────────────────────────────────────────────────────────────────────────────
# Device selection
# ──────────────────────────────────────────────────────────────────────────────

def _safe_device(requested: str) -> str:
    """Resolve to CUDA / MPS / CPU; ``"cpu"`` and ``"mps"`` are honoured verbatim, ``"cuda"`` falls back."""
    import torch

    if requested == "cpu":
        return "cpu"

    if requested == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return "mps"
        print("[device] MPS not available → using CPU")
        return "cpu"

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        if major * 10 + minor > _MAX_SM:
            print(f"[device] GPU sm_{major*10+minor} > sm_{_MAX_SM} → trying MPS / CPU")
        else:
            print(f"[device] using CUDA (sm_{major*10+minor})")
            return "cuda"

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # Some SAM2 / transformers ops aren't implemented for MPS yet; this
        # env var tells PyTorch to silently fall back to CPU per-op instead
        # of raising.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        print("[device] using MPS (Apple Silicon GPU); unsupported ops fall back to CPU")
        return "mps"

    print("[device] no GPU available → using CPU")
    return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Frame resolution & RGB loading
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_frame_list(
    frames: list[dict],
    base_dir: pathlib.Path | None,
) -> list[dict]:
    """Resolve operator-supplied frame dicts to absolute paths; 'depth' is optional here so
    legacy captures don't error — build_depth_cloud raises if depth is actually needed."""
    def _resolve(p: str) -> pathlib.Path:
        return (base_dir / p) if base_dir else pathlib.Path(p)

    out: list[dict] = []
    for f in frames:
        entry: dict = {
            "color":  _resolve(f["color_filename"]),
            "camera": _resolve(f["camera_json_filename"]),
        }
        depth_filename = f.get("depth_filename")
        if depth_filename:
            entry["depth"] = _resolve(depth_filename)
        out.append(entry)
    return out


def _load_rgb_np(path: pathlib.Path) -> np.ndarray:
    """RGBA *_color.png → uint8 RGB HxWx3 (composite alpha over white)."""
    bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgra is None:
        raise IOError(f"Cannot read {path}")
    if bgra.ndim == 2:
        bgra = cv2.cvtColor(bgra, cv2.COLOR_GRAY2BGR)
    if bgra.shape[2] == 4:
        a = bgra[:, :, 3:4].astype(np.float32) / 255.0
        bgr = (
            bgra[:, :, :3].astype(np.float32) * a
            + np.full_like(bgra[:, :, :3], 255, dtype=np.float32) * (1 - a)
        ).clip(0, 255).astype(np.uint8)
    else:
        bgr = bgra[:, :, :3]
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ──────────────────────────────────────────────────────────────────────────────
# Camera & projection
# ──────────────────────────────────────────────────────────────────────────────

def _load_camera(p: pathlib.Path) -> dict:
    with open(p) as f:
        return json.load(f)


def _make_P(intrin: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """Compose the 3×4 projection matrix from Miris intrinsics + extrinsics."""
    return intrin @ (_MIRIS_TO_CV @ w2c[:3, :])


def _load_and_make_P(frame: dict) -> tuple[np.ndarray, dict]:
    """Load camera JSON for a frame and return (projection matrix P, meta dict)."""
    meta = _load_camera(frame["camera"])
    P = _make_P(
        np.array(meta["camera_intrinsics"], dtype=np.float64),
        np.array(meta["world_to_camera"], dtype=np.float64),
    )
    return P, meta


def _project_cv(pt: np.ndarray, P: np.ndarray):
    """Single-point projection → ``(u, v, z)`` or ``None``; bulk callers use ``_project_points_into_frame``."""
    ph = P @ np.append(pt, 1.0)
    z = ph[2]
    if z <= 1e-6:
        return None
    return ph[0] / z, ph[1] / z, z


def _project_points_into_frame(
    pts: np.ndarray,
    P: np.ndarray,
    lmap: np.ndarray,
    dm: Optional[np.ndarray],
    occlusion_tol: float,
    n_dets: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised hot path: project all ``pts`` through ``P``; return ``(pi, di)`` for points
    that are in-front + in-bounds + visible (if ``dm``) + land on a valid mask pixel."""
    H, W = lmap.shape[:2]
    N = pts.shape[0]
    if N == 0:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, np.empty(0, dtype=np.int32)

    pts_h = np.hstack([pts, np.ones((N, 1))])              # (N, 4)
    ph = pts_h @ P.T                                        # (N, 3)
    z = ph[:, 2]
    in_front = z > 1e-6
    safe_z = np.where(in_front, z, 1.0)
    ui = np.round(ph[:, 0] / safe_z).astype(np.int64)
    vi = np.round(ph[:, 1] / safe_z).astype(np.int64)
    in_bounds = in_front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    if not in_bounds.any():
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, np.empty(0, dtype=np.int32)

    sel = np.flatnonzero(in_bounds)
    di = lmap[vi[sel], ui[sel]]
    hit = (di >= 0) & (di < n_dets)
    if dm is not None:
        z_render = dm[vi[sel], ui[sel]].astype(np.float64)
        hit &= np.abs(z_render - z[sel]) <= occlusion_tol
    return sel[hit], di[hit].astype(np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# AABB corners
# ──────────────────────────────────────────────────────────────────────────────

def _aabb_corners(aabb: dict) -> np.ndarray:
    """8 corners (8, 3) ordered to match ``_BOX_EDGES``.

    Prefers the tight OBB corners (``obb_corners_world``) when the dict was
    produced by ``_fit_box``; falls back to the Cartesian product of
    ``world_min``/``world_max`` for legacy AABB-only dicts.
    """
    if "obb_corners_world" in aabb:
        return np.asarray(aabb["obb_corners_world"], dtype=np.float64)
    mn = np.asarray(aabb["world_min"], dtype=np.float64)
    mx = np.asarray(aabb["world_max"], dtype=np.float64)
    return np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mn[0], mx[1], mn[2]], [mx[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mn[0], mx[1], mx[2]], [mx[0], mx[1], mx[2]],
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Union-Find
# ──────────────────────────────────────────────────────────────────────────────

class _UnionFind:
    """Minimal Union-Find with path compression."""

    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[px] = py

    def components(self) -> list[list]:
        """Members grouped by root; iteration order matches ``parent`` insertion — sort if you need stable order."""
        groups: dict = defaultdict(list)
        for n in self.parent:
            groups[self.find(n)].append(n)
        return list(groups.values())


# ──────────────────────────────────────────────────────────────────────────────
# CameraCache
# ──────────────────────────────────────────────────────────────────────────────

class CameraCache:
    """Lazy per-frame projection-matrix / image-size / depth-map cache; built once, threaded through stages."""

    def __init__(self, frames: list[dict]):
        self.frames = frames
        self._Ps: Optional[list[np.ndarray]] = None
        self._img_sizes: Optional[list[tuple[int, int]]] = None
        self._metas: Optional[list[dict]] = None
        self._depth_maps: Optional[list[Optional[np.ndarray]]] = None

    def _ensure_cameras(self) -> None:
        if self._Ps is not None:
            return
        self._Ps = []
        self._img_sizes = []
        self._metas = []
        for frame in self.frames:
            P, meta = _load_and_make_P(frame)
            self._Ps.append(P)
            self._metas.append(meta)
            # camera JSON carries width/height (see JS buildCameraJson); only
            # fall back to decoding the PNG if those keys are absent.
            h = meta.get("height")
            w = meta.get("width")
            if h is not None and w is not None:
                self._img_sizes.append((int(h), int(w)))
            else:
                img = cv2.imread(str(frame["color"]), cv2.IMREAD_UNCHANGED)
                self._img_sizes.append(
                    img.shape[:2] if img is not None else _DEFAULT_IMAGE_SIZE
                )

    @property
    def Ps(self) -> list[np.ndarray]:
        self._ensure_cameras()
        return self._Ps  # type: ignore[return-value]

    @property
    def img_sizes(self) -> list[tuple[int, int]]:
        self._ensure_cameras()
        return self._img_sizes  # type: ignore[return-value]

    @property
    def metas(self) -> list[dict]:
        self._ensure_cameras()
        return self._metas  # type: ignore[return-value]

    @property
    def depth_maps(self) -> list[Optional[np.ndarray]]:
        if self._depth_maps is not None:
            return self._depth_maps
        # Deferred import: depth_cloud → geometry, can't import at module load.
        from .depth_cloud import load_depth_map

        self._ensure_cameras()
        out: list[Optional[np.ndarray]] = []
        for frame, meta in zip(self.frames, self._metas or []):
            dpath = frame.get("depth")
            if dpath is None:
                out.append(None)
                continue
            out.append(load_depth_map(
                dpath,
                float(meta["depth_min"]),
                float(meta["depth_max"]),
                dither=False,
            ))
        self._depth_maps = out
        return out
