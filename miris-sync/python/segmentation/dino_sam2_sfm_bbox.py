"""
dino_sam2_sfm_bbox.py
=====================
End-to-end pipeline — identical to dino_sfm_bbox.py except label voting uses
SAM2 pixel-wise masks (box-prompted by Grounding DINO) instead of raw DINO boxes.

Pipeline
--------
  1. Hull AABB      – flat-depth silhouette carving for SfM spatial filter
  2. SfM            – SIFT + multi-view triangulation → 3-D point cloud
  3. DINO + SAM2    – Grounding DINO boxes → SAM2 box-prompted masks per detection;
                     each detection keeps its DINO text label + a precise bool mask
  4. Label voting   – project each 3-D point into labelled frames; look up the
                     pixel it lands on in the per-frame label map (mask-derived);
                     majority-vote label wins
  5. AABB fitting   – one global AABB + one per unique semantic label
  6. Visualise      – 3-D boxes projected onto images; saved as annotated PNGs
  7. Save PLY       – coloured point cloud + wireframe box PLY

Usage
-----
    python dino_sam2_sfm_bbox.py                    # all defaults
    python dino_sam2_sfm_bbox.py \\
        --dataset-dir /path/to/frames \\
        --output-dir  /path/to/out \\
        --seg-frames 10 --device cuda

Dependencies
------------
    pip install transformers torch Pillow opencv-python numpy huggingface_hub
    pip install git+https://github.com/facebookresearch/sam2.git
"""
from __future__ import annotations

import json
import pathlib
import re
import struct
import traceback
from collections import Counter, defaultdict

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────

_DINO_MODEL  = "IDEA-Research/grounding-dino-base"
_SAM2_MODEL  = "facebook/sam2.1-hiera-tiny"
_DINO_TEXT   = "Phone. Laptop."
_MAX_SM      = 120

_DEFAULT_IMAGE_SIZE = (1080, 1920)   # (H, W) fallback when image cannot be read

_PALETTE_BGR: list[tuple[int, int, int]] = [
    (75,  25,  230), (75,  180, 60),  (25,  225, 255), (200, 130,   0),
    (48,  130, 245), (180, 30,  145), (240, 240,  70), (230,  50,  240),
    (60,  245, 210), (212, 190, 250), (128, 128,   0), (255, 190,  220),
    (40,  110, 170), (200, 250, 255), (  0,   0, 128), (195, 255,  170),
    (  0, 128, 128), (180, 215, 255), (128,   0,   0), (128, 128,  128),
]

_BOX_EDGES = [
    (0,1),(2,3),(4,5),(6,7),
    (0,2),(1,3),(4,6),(5,7),
    (0,4),(1,5),(2,6),(3,7),
]


def _colour(idx: int) -> tuple[int, int, int]:
    return _PALETTE_BGR[idx % len(_PALETTE_BGR)]


def _base_label(full_name: str) -> str:
    """'wheel_0' → 'wheel',  'front spoiler_1' → 'front spoiler',  'tire' → 'tire'."""
    parts = full_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return full_name


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

    def components(self) -> dict:
        groups: dict = defaultdict(list)
        for n in self.parent:
            groups[self.find(n)].append(n)
        return dict(groups)


# ──────────────────────────────────────────────────────────────────────────────
# Device helper
# ──────────────────────────────────────────────────────────────────────────────

def _safe_device(requested: str) -> str:
    """Pick the best available PyTorch device.

    Resolution order when ``requested`` is ``"cuda"`` (the default):
      1. CUDA, if the GPU is recent enough (sm <= ``_MAX_SM``)
      2. Apple Silicon MPS (Metal), if available — for Macs
      3. CPU

    Explicit ``"cpu"`` or ``"mps"`` requests are honoured unchanged.
    """
    import os
    import torch

    if requested == "cpu":
        return "cpu"

    if requested == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return "mps"
        print("[device] MPS not available → using CPU")
        return "cpu"

    # "cuda" / auto path: try CUDA, then MPS, then CPU.
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
        # of raising. Setting via setdefault is idempotent.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        print("[device] using MPS (Apple Silicon GPU); unsupported ops fall back to CPU")
        return "mps"

    print("[device] no GPU available → using CPU")
    return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Frame discovery
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_frame_list(
    frames: list[dict],
    base_dir: pathlib.Path | None,
) -> list[dict]:
    """Resolve frame list from the operator-provided dict list."""
    return [
        {
            "color":  (base_dir / f["color_filename"])       if base_dir else pathlib.Path(f["color_filename"]),
            "camera": (base_dir / f["camera_json_filename"]) if base_dir else pathlib.Path(f["camera_json_filename"]),
        }
        for f in frames
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Camera helpers
# ──────────────────────────────────────────────────────────────────────────────

_MIRIS_TO_CV = np.diag([1.0, -1.0, -1.0])


def _load_camera(p: pathlib.Path) -> dict:
    with open(p) as f:
        return json.load(f)


def _make_P(intrin: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    return intrin @ (_MIRIS_TO_CV @ w2c[:3, :])


def _load_and_make_P(frame: dict) -> tuple[np.ndarray, dict]:
    """Load camera JSON for a frame and return (projection matrix P, meta dict)."""
    meta = _load_camera(frame["camera"])
    P = _make_P(
        np.array(meta["camera_intrinsics"], dtype=np.float64),
        np.array(meta["world_to_camera"],   dtype=np.float64),
    )
    return P, meta


def _project_cv(pt: np.ndarray, P: np.ndarray):
    ph = P @ np.append(pt, 1.0)
    z  = ph[2]
    if z <= 1e-6:
        return None
    return ph[0]/z, ph[1]/z, z


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Hull AABB
# ──────────────────────────────────────────────────────────────────────────────

def _backproject_flat(meta, fg, stride=2):
    intrin = np.array(meta["camera_intrinsics"], dtype=np.float64)
    w2c    = np.array(meta["world_to_camera"],   dtype=np.float64)
    c2w    = np.linalg.inv(w2c)
    fx=intrin[0,0]; fy=intrin[1,1]; cx=intrin[0,2]; cy=intrin[1,2]
    H,W = fg.shape
    uu,vv = np.meshgrid(np.arange(0,W,stride), np.arange(0,H,stride))
    fg_s  = fg[vv,uu]
    if fg_s.sum()==0: return None
    u=uu[fg_s].astype(np.float64); v=vv[fg_s].astype(np.float64)
    z=np.full(u.shape, np.linalg.norm((c2w@[0.,0.,0.,1.])[:3]))
    pts_cam=np.stack([(u-cx)/fx*z, -(v-cy)/fy*z, -z, np.ones_like(z)], axis=-1)
    return (c2w@pts_cam.T).T[:,:3]


def compute_hull_aabb(frames, cam_stride=1, alpha_threshold=0.5,
                       percentile_trim=1.0, aabb_pad=0.15):
    all_pts=[]
    for frame in frames[::cam_stride]:
        color=cv2.imread(str(frame["color"]), cv2.IMREAD_UNCHANGED)
        if color is None: continue
        meta=_load_camera(frame["camera"])
        fg=(color[:,:,3].astype(np.float32)/255.0)>=alpha_threshold
        pts=_backproject_flat(meta,fg)
        if pts is not None: all_pts.append(pts)
    if not all_pts: raise ValueError("Hull AABB: no foreground points.")
    pts_all=np.concatenate(all_pts,axis=0)
    lo=np.percentile(pts_all,percentile_trim,axis=0)
    hi=np.percentile(pts_all,100.-percentile_trim,axis=0)
    pad=(hi-lo)*aabb_pad
    return lo-pad, hi+pad, {"world_min":lo.tolist(),"world_max":hi.tolist()}


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: SfM triangulation
# ──────────────────────────────────────────────────────────────────────────────

def _extract_features(frame, alpha_threshold=0.5, n_features=4000):
    meta=_load_camera(frame["camera"])
    bgra=cv2.imread(str(frame["color"]),cv2.IMREAD_UNCHANGED)
    if bgra is None: return None
    alpha=bgra[:,:,3].astype(np.float32)/255.0
    fg_mask=(alpha>=alpha_threshold).astype(np.uint8)*255
    gray=cv2.cvtColor(bgra[:,:,:3],cv2.COLOR_BGR2GRAY)
    sift=cv2.SIFT_create(nfeatures=n_features)
    kps,descs=sift.detectAndCompute(gray,mask=fg_mask)
    if kps is None or len(kps)<8: return None
    return np.array([kp.pt for kp in kps],dtype=np.float32), descs, alpha, bgra, meta


def _triangulate_pair(kp1,d1,a1,P1,kp2,d2,a2,P2,c1,
                       aabb_min,aabb_max,max_err=2.0,alpha_thr=0.5,ratio=0.75):
    bf=cv2.BFMatcher(cv2.NORM_L2)
    raw=bf.knnMatch(d1,d2,k=2)
    good=[m for m,n in raw if m.distance<ratio*n.distance]
    if len(good)<8:
        empty=np.zeros((0,3)); empty2=np.zeros((0,2),dtype=np.int32)
        return empty,np.zeros((0,3),dtype=np.uint8),empty2,empty2
    pts1=kp1[[m.queryIdx for m in good]].T
    pts2=kp2[[m.trainIdx for m in good]].T
    pts4d=cv2.triangulatePoints(P1,P2,pts1,pts2)
    pts4d/=pts4d[3:4,:]
    pts_w=pts4d[:3,:].T; M=pts_w.shape[0]
    keep=np.ones(M,bool)
    for P in (P1,P2): keep&=(P[2:3,:]@pts4d)[0]>0.01
    for P_r,pts_ref in ((P1,pts1),(P2,pts2)):
        proj=(P_r@pts4d).T; proj/=proj[:,2:3]
        keep&=np.linalg.norm(proj[:,:2]-pts_ref.T,axis=1)<max_err
    keep&=np.all(pts_w>=aabb_min,axis=1)&np.all(pts_w<=aabb_max,axis=1)
    H1,W1=a1.shape; H2,W2=a2.shape
    u1=pts1[0].astype(int); v1=pts1[1].astype(int)
    u2=pts2[0].astype(int); v2=pts2[1].astype(int)
    in1=(u1>=0)&(u1<W1)&(v1>=0)&(v1<H1)
    in2=(u2>=0)&(u2<W2)&(v2>=0)&(v2<H2)
    keep&=in1&in2
    vi=np.where(keep)[0]
    fg1=np.zeros(M,bool); fg2=np.zeros(M,bool)
    fg1[vi]=a1[v1[vi],u1[vi]]>=alpha_thr
    fg2[vi]=a2[v2[vi],u2[vi]]>=alpha_thr
    keep&=fg1&fg2
    if keep.sum()==0:
        empty=np.zeros((0,3)); empty2=np.zeros((0,2),dtype=np.int32)
        return empty,np.zeros((0,3),dtype=np.uint8),empty2,empty2
    pts_out=pts_w[keep]; u1k,v1k=u1[keep],v1[keep]
    u2k=pts2[0].astype(int)[keep]; v2k=pts2[1].astype(int)[keep]
    colours=np.stack([c1[v1k,u1k,2],c1[v1k,u1k,1],c1[v1k,u1k,0]],axis=-1).astype(np.uint8)
    pix1=np.column_stack([u1k,v1k]).astype(np.int32)
    pix2=np.column_stack([u2k,v2k]).astype(np.int32)
    return pts_out, colours, pix1, pix2


def _voxel_downsample(pts,rgbs,voxel_size):
    if pts.shape[0]==0: return pts,rgbs
    idx=np.floor(pts/voxel_size).astype(np.int64)
    _,inv,counts=np.unique(idx,axis=0,return_inverse=True,return_counts=True)
    n=counts.shape[0]; p_out=np.zeros((n,3)); r_out=np.zeros((n,3))
    np.add.at(p_out,inv,pts); np.add.at(r_out,inv,rgbs.astype(np.float64))
    p_out/=counts[:,None]; r_out/=counts[:,None]
    return p_out, r_out.astype(np.uint8)


def build_sfm_cloud(frames,feat_data,aabb_min,aabb_max,
                    pair_window=5,max_reproj_err=2.0,
                    alpha_threshold=0.5,voxel_size=0.02):
    n=len(frames)
    if pair_window==0:
        pairs=[(i,j) for i in range(n) for j in range(i+1,n)]
    else:
        pairs=[]
        for i in range(n):
            for j in range(i+1,min(i+1+pair_window,n)): pairs.append((i,j))
        for i in range(n-pair_window,n):
            for j in range(0,pair_window-(n-i-1)):
                if j!=i: pairs.append((min(i,j),max(i,j)))
    pairs=list(set(pairs))
    print(f"[sfm] Triangulating {len(pairs)} camera pairs …")
    all_pts=[]; all_rgb=[]; all_obs=[]   # all_obs: (i, j, pix1, pix2)
    for pi,(i,j) in enumerate(sorted(pairs)):
        if feat_data[i] is None or feat_data[j] is None: continue
        kp1,d1,a1,c1,m1=feat_data[i]; kp2,d2,a2,c2,m2=feat_data[j]
        P1=_make_P(np.array(m1["camera_intrinsics"],dtype=np.float64),
                   np.array(m1["world_to_camera"],  dtype=np.float64))
        P2=_make_P(np.array(m2["camera_intrinsics"],dtype=np.float64),
                   np.array(m2["world_to_camera"],  dtype=np.float64))
        pts,rgb,pix1,pix2=_triangulate_pair(kp1,d1,a1,P1,kp2,d2,a2,P2,c1,
                                            aabb_min,aabb_max,max_reproj_err,alpha_threshold)
        if pts.shape[0]>0:
            all_pts.append(pts); all_rgb.append(rgb)
            all_obs.append((i, j, pix1, pix2))   # 2D match pixels, source frame pair
        if (pi+1)%20==0 or pi==len(pairs)-1:
            total=sum(p.shape[0] for p in all_pts)
            print(f"  pairs {pi+1:3d}/{len(pairs)}  running total: {total:,} pts")
    if not all_pts: raise ValueError("No points triangulated.")
    pts_all=np.concatenate(all_pts,axis=0); rgb_all=np.concatenate(all_rgb,axis=0)
    print(f"[sfm] Raw: {pts_all.shape[0]:,} pts")
    pts_all,rgb_all=_voxel_downsample(pts_all,rgb_all,voxel_size)
    print(f"[sfm] After voxel dedup: {pts_all.shape[0]:,} (voxel {voxel_size} m)")
    return pts_all, rgb_all, all_obs


# ──────────────────────────────────────────────────────────────────────────────
# Image loading helper (RGBA splat → RGB)
# ──────────────────────────────────────────────────────────────────────────────

def _load_rgb_np(path: pathlib.Path) -> np.ndarray:
    """RGBA *_color.png → uint8 RGB HxWx3 (composite over white)."""
    bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgra is None:
        raise IOError(f"Cannot read {path}")
    if bgra.ndim == 2:
        bgra = cv2.cvtColor(bgra, cv2.COLOR_GRAY2BGR)
    if bgra.shape[2] == 4:
        a   = bgra[:,:,3:4].astype(np.float32)/255.0
        bgr = (bgra[:,:,:3].astype(np.float32)*a +
               np.full_like(bgra[:,:,:3],255,dtype=np.float32)*(1-a)
               ).clip(0,255).astype(np.uint8)
    else:
        bgr = bgra[:,:,:3]
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3a: Grounding DINO — returns boxes + labels per frame
# ──────────────────────────────────────────────────────────────────────────────

def _parse_prompt_words(dino_text: str) -> list[str]:
    """'Wheel. Phone. Laptop.' → ['wheel', 'phone', 'laptop']"""
    import re
    return [t.strip().lower() for t in re.split(r'[.\s]+', dino_text) if t.strip()]


def _normalize_label(raw: str, prompt_words: list[str]) -> str:
    """
    Map a raw DINO label back to the nearest single prompt word.

    Grounding DINO sometimes returns the full prompt text (e.g. 'wheel phone
    laptop') as the label for every detection instead of the matched token.
    We resolve by finding which prompt word the returned text best represents.
    """
    raw_lower = raw.lower()
    raw_tokens = set(raw_lower.split())

    # 1. Exact token match  ('phone', 'laptop', …)
    for w in prompt_words:
        if w in raw_tokens:
            return w

    # 2. Substring match  ('laptops', 'iphone', …)
    for w in prompt_words:
        if w in raw_lower:
            return w

    # 3. Best character-overlap score
    scores = {w: sum(c in raw_lower for c in w) for w in prompt_words}
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    return raw  # nothing matched — keep original

def run_dino_on_frames(frame_list, device, box_threshold=0.35, text_threshold=0.25,
                       dino_text: str = _DINO_TEXT):
    """
    Returns list (per frame) of dicts:
        {"label": str, "score": float, "box_xyxy": [x1,y1,x2,y2]}
    """
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    print(f"\n── Step 3a: Grounding DINO on {len(frame_list)} frames ─────────────")
    print(f"[dino] Loading {_DINO_MODEL} on {device} …")
    processor = AutoProcessor.from_pretrained(_DINO_MODEL)
    model     = AutoModelForZeroShotObjectDetection.from_pretrained(_DINO_MODEL).to(device)
    print("[dino] Model ready.")

    all_dets: list[list[dict]] = []
    for idx, frame in enumerate(frame_list):
        rgb   = _load_rgb_np(frame["color"])
        image = Image.fromarray(rgb)
        H, W  = rgb.shape[:2]
        inputs = processor(images=image, text=dino_text,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        try:
            results = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                threshold=box_threshold, text_threshold=text_threshold,
                target_sizes=[(H, W)],
            )[0]
        except TypeError:
            results = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                box_threshold=box_threshold, text_threshold=text_threshold,
                target_sizes=[(H, W)],
            )[0]
        boxes  = results["boxes"].cpu().tolist()
        scores = results["scores"].cpu().tolist()
        labels = results.get("text_labels") or results.get("labels") or []
        # Normalise each label back to the nearest prompt word
        _pw = _parse_prompt_words(dino_text)
        if idx == 0 and labels:
            print(f"[dino] Raw labels from model (frame 0): {[str(l) for l in labels]}")
            print(f"[dino] Prompt words: {_pw}")
        labels = [_normalize_label(str(l), _pw) for l in labels]
        dets   = [{"label":str(l),"score":float(s),"box_xyxy":b}
                  for l,s,b in zip(labels, scores, boxes)]

        # Assign per-label instance rank sorted by box center_x (left→right)
        # so "wheel_0" = leftmost wheel consistently across frames
        from collections import defaultdict as _dd
        label_boxes: dict[str, list[int]] = _dd(list)
        for i, d in enumerate(dets):
            label_boxes[d["label"]].append(i)
        for lbl, idxs in label_boxes.items():
            idxs.sort(key=lambda i: (dets[i]["box_xyxy"][0] + dets[i]["box_xyxy"][2]) / 2)
            for rank, i in enumerate(idxs):
                dets[i]["instance_id"] = f"{lbl}_{rank}" if len(idxs) > 1 else lbl

        all_dets.append(dets)
        print(f"  [{idx+1:3d}/{len(frame_list)}] {frame['color'].name[:55]:55s}  "
              f"dets={len(dets)}")
        yield idx + 1, len(frame_list)

    del model
    import gc; gc.collect()
    try:
        if device=="cuda": torch.cuda.empty_cache()
    except Exception: pass

    return all_dets


# ──────────────────────────────────────────────────────────────────────────────
# Step 3b: SAM2 box-prompted — returns per-frame label maps (H×W int, -1=none)
# ──────────────────────────────────────────────────────────────────────────────

def run_sam2_on_frames(frame_list, all_dets, device, sam2_model=_SAM2_MODEL):
    """
    For each detection in each frame run SAM2 box-prompted segmentation.
    Returns list (per frame) of np.ndarray (H, W) int32:
        pixel value = detection index in that frame's det list  (-1 = background)
    When multiple detections cover the same pixel, the last one (highest score
    after sorting by score desc) wins — so higher-confidence masks take priority.
    """
    import torch
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print(f"\n── Step 3b: SAM2 box-prompted on {len(frame_list)} frames ──────────")
    print(f"[sam2] Loading {sam2_model} on {device} …")
    sam2      = build_sam2_hf(sam2_model, device=device)
    predictor = SAM2ImagePredictor(sam2)
    print("[sam2] Model ready.")

    label_maps: list[np.ndarray] = []

    for idx, (frame, dets) in enumerate(zip(frame_list, all_dets)):
        rgb  = _load_rgb_np(frame["color"])
        H, W = rgb.shape[:2]
        lmap = np.full((H, W), -1, dtype=np.int32)

        if not dets:
            label_maps.append(lmap)
            print(f"  [{idx+1:3d}/{len(frame_list)}] {frame['color'].name[:50]:50s}  "
                  f"no detections — skipped")
            continue

        predictor.set_image(rgb)

        # Sort detections by score ascending so high-score masks overwrite low-score
        sorted_dets = sorted(enumerate(dets), key=lambda x: x[1]["score"])

        n_masks = 0
        for det_idx, det in sorted_dets:
            box = np.array(det["box_xyxy"], dtype=np.float32)
            try:
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box[None],          # (1, 4)
                    multimask_output=True,
                )
                best = int(scores.argmax())
                mask = masks[best]
                if hasattr(mask, "cpu"):
                    mask = mask.cpu().numpy()
                lmap[mask.astype(bool)] = det_idx
                n_masks += 1
            except Exception as e:
                print(f"    [sam2 warn] det {det_idx}: {e}")

        # (remove old per-detection loop below — replaced above)
        sorted_dets = []   # sentinel so the old loop does nothing


        label_maps.append(lmap)
        print(f"  [{idx+1:3d}/{len(frame_list)}] {frame['color'].name[:50]:50s}  "
              f"masks={n_masks}")
        yield idx + 1, len(frame_list)

    del predictor, sam2
    import gc; gc.collect()
    try:
        if device=="cuda": torch.cuda.empty_cache()
    except Exception: pass

    return label_maps


def save_sam2_mask_debug(
    frame_list: list[dict],
    all_dets: list[list[dict]],
    label_maps: list[np.ndarray],
    output_dir: pathlib.Path,
) -> None:
    """
    For each segmentation frame save an RGB overlay where each SAM2 mask is
    coloured by its instance_id.  Saved to <output_dir>/sam2_masks/.
    """
    mask_dir = output_dir / "sam2_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Build a stable colour map: instance_id → BGR colour
    all_instances: list[str] = sorted({
        d.get("instance_id", d["label"])
        for dets in all_dets for d in dets
    })
    inst_colour = {name: _colour(i) for i, name in enumerate(all_instances)}

    for idx, (frame, dets, lmap) in enumerate(zip(frame_list, all_dets, label_maps)):
        rgb = _load_rgb_np(frame["color"])
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
        overlay = bgr.copy()
        H, W = lmap.shape

        # Fill each mask colour
        for det_idx, det in enumerate(dets):
            iid    = det.get("instance_id", det["label"])
            colour = inst_colour.get(iid, (128, 128, 128))
            mask   = lmap == det_idx
            for ch, val in enumerate(colour):
                overlay[:, :, ch] = np.where(mask, val, overlay[:, :, ch])

        vis = cv2.addWeighted(overlay, 0.45, bgr, 0.55, 0).astype(np.uint8)

        # Draw contours + DINO boxes + instance labels on top
        for det_idx, det in enumerate(dets):
            iid    = det.get("instance_id", det["label"])
            colour = inst_colour.get(iid, (128, 128, 128))
            mask8  = ((lmap == det_idx) * 255).astype(np.uint8)
            cnts, _ = cv2.findContours(mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, colour, 1)
            # DINO bounding box
            x1, y1, x2, y2 = [int(v) for v in det["box_xyxy"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
            # Label inside box, top-left corner
            score_str = f"{det['score']:.2f}"
            text = f"{iid} {score_str}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, cv2.FILLED)
            cv2.putText(vis, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

        stem = re.sub(r"_camera$", "", frame["camera"].stem)
        out  = mask_dir / f"{stem}_sam2.png"
        cv2.imwrite(str(out), vis)

    print(f"[debug] SAM2 mask overlays → {mask_dir}  ({len(frame_list)} images)")


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Label voting using SAM2 pixel masks
# ──────────────────────────────────────────────────────────────────────────────

def _vote_for_key(
    pts: np.ndarray,
    seg_frame_list: list[dict],
    all_dets: list[list[dict]],
    label_maps: list[np.ndarray],
    key_field: str,     # "label" or "instance_id"
    min_votes: int = 1,
    min_pts: int = 3,
) -> tuple[np.ndarray, list[str]]:
    """Generic voting: for each 3-D point project into frames and vote for det[key_field]."""
    N = pts.shape[0]
    Ps, img_sizes = [], []
    for frame in seg_frame_list:
        P, _meta = _load_and_make_P(frame)
        Ps.append(P)
        probe = cv2.imread(str(frame["color"]), cv2.IMREAD_UNCHANGED)
        h, w  = (probe.shape[0], probe.shape[1]) if probe is not None else _DEFAULT_IMAGE_SIZE
        img_sizes.append((h, w))

    votes:  list[Counter]      = [Counter() for _ in range(N)]
    scores: list[defaultdict]  = [defaultdict(float) for _ in range(N)]

    for P, dets, lmap, (H, W) in zip(Ps, all_dets, label_maps, img_sizes):
        if not dets:
            continue
        for p_idx in range(N):
            res = _project_cv(pts[p_idx], P)
            if res is None:
                continue
            u, v, _ = res
            ui, vi  = int(round(u)), int(round(v))
            if not (0 <= ui < W and 0 <= vi < H):
                continue
            di = int(lmap[vi, ui])
            if di < 0 or di >= len(dets):
                continue
            key = dets[di].get(key_field, dets[di]["label"])
            votes[p_idx][key]  += 1
            scores[p_idx][key] += dets[di]["score"]

    all_keys: set = set()
    for v in votes:
        all_keys.update(v.keys())
    key_names = sorted(all_keys)
    key_to_id = {k: i for i, k in enumerate(key_names)}

    raw = np.full(N, -1, dtype=np.int32)
    for p_idx, v in enumerate(votes):
        if not v or sum(v.values()) < min_votes:
            continue
        best = max(v, key=lambda k: (v[k], scores[p_idx][k]))
        raw[p_idx] = key_to_id[best]

    counts = Counter(raw[raw >= 0])
    valid  = {kid for kid, cnt in counts.items() if cnt >= min_pts}
    remap  = {old: new for new, old in enumerate(sorted(valid))}
    final  = np.array(
        [remap[int(i)] if int(i) in valid else -1 for i in raw],
        dtype=np.int32)
    kept   = [key_names[old] for old in sorted(valid)]
    return final, kept


def _build_graph_from_matches(
    match_obs: list,
    strided_to_seg: dict,
    seg_data: list,            # [(lmap, H, W, dets), ...]
    det_label: dict,
) -> dict:
    """Build co-observation edge_count from 2D SIFT match observations.

    Two detection nodes (fi,di) and (fj,dj) gain weight for each matched SIFT
    pixel pair whose pixels fall inside their SAM2 masks AND share a label.
    Returns ``edge_count: dict[(node_a, node_b), int]`` with ``node_a <= node_b``.
    """
    edge_count: dict = defaultdict(int)
    n_obs_used = 0
    for (si, sj, pix1, pix2) in match_obs:
        fi = strided_to_seg.get(si, -1)
        fj = strided_to_seg.get(sj, -1)
        if fi < 0 or fj < 0:
            continue
        lmap_i, H_i, W_i, dets_i = seg_data[fi]
        lmap_j, H_j, W_j, dets_j = seg_data[fj]
        if not dets_i or not dets_j:
            continue
        for (u1, v1), (u2, v2) in zip(pix1, pix2):
            if not (0 <= u1 < W_i and 0 <= v1 < H_i): continue
            if not (0 <= u2 < W_j and 0 <= v2 < H_j): continue
            di = int(lmap_i[v1, u1])
            dj = int(lmap_j[v2, u2])
            if di < 0 or di >= len(dets_i): continue
            if dj < 0 or dj >= len(dets_j): continue
            if det_label.get((fi, di)) != det_label.get((fj, dj)): continue
            na, nb = (fi, di), (fj, dj)
            key = (na, nb) if na <= nb else (nb, na)
            edge_count[key] += 1
            n_obs_used += 1
    print(f"  {n_obs_used:,} match pixels inside SAM2 masks → "
          f"{len(edge_count)} unique detection-pair edges")
    return edge_count


def _build_graph_from_reprojection(
    pts: np.ndarray,
    seg_frame_list: list[dict],
    all_dets: list[list[dict]],
    label_maps: list[np.ndarray],
    det_label: dict,
    n_points: int,
) -> dict:
    """Build co-observation edge_count by projecting SfM points into SAM2 masks.

    For each SfM point projected into a frame, look up which detection's mask
    covers that pixel. Detection nodes that co-observe ≥2 same-label points
    gain edge weight. Returns ``edge_count: dict[(node_a, node_b), int]``.
    """
    Ps, img_sizes = [], []
    for frame in seg_frame_list:
        P, _meta = _load_and_make_P(frame)
        Ps.append(P)
        img = cv2.imread(str(frame["color"]), cv2.IMREAD_UNCHANGED)
        img_sizes.append(img.shape[:2] if img is not None else _DEFAULT_IMAGE_SIZE)

    edge_count: dict = defaultdict(int)
    point_dets_tmp: list = [[] for _ in range(n_points)]
    for fi, (P, dets, lmap, (H, W)) in enumerate(
            zip(Ps, all_dets, label_maps, img_sizes)):
        if not dets:
            continue
        for pi in range(n_points):
            res = _project_cv(pts[pi], P)
            if res is None:
                continue
            u, v, _ = res
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < W and 0 <= vi < H): continue
            di = int(lmap[vi, ui])
            if di < 0 or di >= len(dets): continue
            node = (fi, di)
            det_label[node] = dets[di]["label"]
            point_dets_tmp[pi].append(node)

    for obs in point_dets_tmp:
        if len(obs) < 2:
            continue
        by_label: dict = defaultdict(list)
        for node in obs:
            by_label[det_label[node]].append(node)
        for nodes in by_label.values():
            for ii in range(len(nodes)):
                for jj in range(ii + 1, len(nodes)):
                    a, b = nodes[ii], nodes[jj]
                    key = (a, b) if a <= b else (b, a)
                    edge_count[key] += 1
    return edge_count


def assign_labels_from_masks(   # noqa: C901  (complex but intentional)
    pts: np.ndarray,
    seg_frame_list: list[dict],
    all_dets: list[list[dict]],   # updated IN-PLACE with instance_id
    label_maps: list[np.ndarray],
    min_votes: int = 1,
    min_label_pts: int = 3,
    anchor_min_pts: int = 2,          # min shared 2D matches to link two detections
    match_obs: list = None,           # [(strided_i, strided_j, pix1, pix2), ...]
    strided_to_seg: dict = None,      # strided frame idx → seg_frame_list idx
) -> tuple[np.ndarray, list[str]]:
    """
    Graph-based instance assignment using SfM point co-observation:

      Feature vector of each SfM point:
        the set of (frame_idx, local_det_idx, base_label) detection observations
        that cover it when projected into each segmentation frame.

      Instance discovery:
        Build a graph where nodes = (frame_idx, det_idx) per-frame detections.
        Two nodes share a directed edge weighted by # SfM points that project
        inside BOTH their SAM2 masks.  Union-Find on same-label nodes with
        edge weight ≥ anchor_min_pts → connected components = physical instances.

      Pass 1:  baseline label vote to log rough coverage.
      Graph:   SfM point ↔ detection memberships built from lmap projections.
      Union-Find: merge detection observations that share enough SfM points.
      Pass 2:  each SfM point votes for the component (instance) that covers it
               most often; majority wins.

      all_dets is updated IN-PLACE so save_sam2_mask_debug shows instance IDs.
    """
    from collections import defaultdict

    N = pts.shape[0]
    min_shared = max(1, anchor_min_pts)

    print(f"\n── Step 4: SfM-graph instance assignment "
          f"({N:,} pts × {len(seg_frame_list)} frames) ──")

    # ── Pass 1: base-label vote (for diagnostics) ─────────────────────────────
    print("[pass 1] Voting for base labels …")
    pass1_dets = [[{**d, "instance_id": d["label"]} for d in fd] for fd in all_dets]
    rough_ids, rough_names = _vote_for_key(
        pts, seg_frame_list, pass1_dets, label_maps,
        key_field="instance_id", min_votes=min_votes, min_pts=min_label_pts)
    print(f"  rough labels: {rough_names}")
    for i, n in enumerate(rough_names):
        print(f"  '{n}': {int((rough_ids==i).sum()):,} pts")
    if not rough_names:
        print("[warn] Pass 1 found no labelled points — returning empty.")
        return np.full(N, -1, dtype=np.int32), []

    # ── Build detection graph from 2D SIFT match observations ─────────────────
    #   Each SfM matched pixel pair (u1,v1 in frame_i) ↔ (u2,v2 in frame_j)
    #   is a direct observation — no 3D reprojection, no occlusion artifacts.
    #   If both pixels fall inside SAM2 tight masks, the two detections share
    #   an observed feature point and are candidates for the same physical wheel.

    #   det_pts[(fi, di)]  = set of (obs_idx) observations (for Pass 2 back-ref)
    #   det_label[(fi,di)] = base_label of that detection
    #   edge_count[(na,nb)] = # 2D matched features shared between two detections
    #   point_dets[pi]     = list of (fi,di) for Pass 2 (still uses 3D projection)

    det_label:  dict = {}
    det_pts:    dict = defaultdict(set)   # reserved for downstream; not populated here

    # Build seg frame direct-access list (indexed by seg_fi)
    seg_data = []   # seg_data[seg_fi] = (lmap, H, W, dets)
    for seg_fi, (frame, dets, lmap) in enumerate(
            zip(seg_frame_list, all_dets, label_maps)):
        H, W = lmap.shape[:2]
        seg_data.append((lmap, H, W, dets))
        for di, det in enumerate(dets):
            det_label[(seg_fi, di)] = det["label"]

    if match_obs and strided_to_seg is not None:
        print("[graph] Building detection graph from 2D SAM2-masked SIFT matches …")
        edge_count = _build_graph_from_matches(match_obs, strided_to_seg, seg_data, det_label)
    else:
        print("[graph] Fallback: projecting SfM points into SAM2 masks …")
        edge_count = _build_graph_from_reprojection(pts, seg_frame_list, all_dets, label_maps, det_label, N)

    all_nodes = sorted({n for key in edge_count for n in key} | set(det_label.keys()))
    # Also include isolated nodes (detections with no shared features)
    for fi, dets in enumerate(all_dets):
        for di in range(len(dets)):
            node = (fi, di)
            if node not in det_label:
                det_label[node] = dets[di]["label"]
            if node not in all_nodes:
                all_nodes.append(node)

    # ── Union-Find over detection nodes ───────────────────────────────────────
    uf = _UnionFind(all_nodes)

    # Print edge-weight histogram to help tune min_shared
    if edge_count:
        weights = sorted(edge_count.values(), reverse=True)
        bins = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        hist = {b: sum(1 for w in weights if w >= b) for b in bins}
        print(f"  Edge weight histogram (# edges with >=N shared pts): "
              + "  ".join(f">={b}:{hist[b]}" for b in bins))

    n_linked = 0
    for (n1, n2), cnt in edge_count.items():
        if cnt >= min_shared and det_label.get(n1) == det_label.get(n2):
            uf.union(n1, n2)
            n_linked += 1

    print(f"  {len(all_nodes)} detection-nodes, "
          f"{len(edge_count)} candidate edges, "
          f"{n_linked} linked (>={min_shared} shared SAM2-mask SfM pts)")

    # ── Connected components → physical instances ──────────────────────────────
    components = uf.components()

    comp_data = []
    for root, members in components.items():
        comp_pts = set()
        for node in members:
            comp_pts.update(det_pts.get(node, set()))
        if len(comp_pts) < min_label_pts and not (match_obs is not None):
            continue
        label_votes = Counter(det_label.get(n, "unknown") for n in members)
        base_lbl = label_votes.most_common(1)[0][0]
        # Placeholder centroid (overwritten by Pass 2 below if we have SfM pts)
        centroid = pts[list(comp_pts)].mean(axis=0) if comp_pts else np.zeros(3)
        comp_data.append({"members": members, "pt_indices": list(comp_pts),
                          "base_label": base_lbl, "centroid": centroid})

    if not comp_data:
        print("[warn] No instances survived min_label_pts filter — returning empty.")
        return np.full(N, -1, dtype=np.int32), []

    # Sort: base-label first, then by XZ azimuth (if centroid known) else member count
    def _sort_key(c):
        centroid = c["centroid"]
        has_centroid = np.any(centroid != 0)
        az = float(np.arctan2(centroid[0], centroid[2])) if has_centroid else 0.0
        return (c["base_label"], az, -len(c["members"]))
    comp_data.sort(key=_sort_key)

    # Assign human-readable IDs: "wheel_0", "wheel_1", "spoiler_0", …
    label_counter: Counter = Counter()
    node_to_inst:  dict = {}
    for comp in comp_data:
        idx     = label_counter[comp["base_label"]]
        inst_id = f"{comp['base_label']}_{idx}"
        comp["instance_id"] = inst_id
        label_counter[comp["base_label"]] += 1
        for node in comp["members"]:
            node_to_inst[node] = inst_id

    # Update all_dets IN-PLACE — debug overlays will now show instance IDs
    for fi, dets in enumerate(all_dets):
        for di, det in enumerate(dets):
            det["instance_id"] = node_to_inst.get((fi, di), det["label"])

    # ── Pass 2: project SfM points into seg frames → vote for instance_id ────────
    #   This uses 3D reprojection (majority vote handles stray occlusion errors).
    print("[pass 2] Assigning SfM points to instances by majority vote …")
    all_inst_names = [c["instance_id"] for c in comp_data]
    inst_to_idx    = {name: i for i, name in enumerate(all_inst_names)}

    # Build projection matrices for seg frames
    Ps2, img_sizes2 = [], []
    for frame in seg_frame_list:
        P, _meta = _load_and_make_P(frame)
        Ps2.append(P)
        img = cv2.imread(str(frame["color"]), cv2.IMREAD_UNCHANGED)
        img_sizes2.append(img.shape[:2] if img is not None else _DEFAULT_IMAGE_SIZE)

    pt_votes: list = [Counter() for _ in range(N)]
    for fi, (P, dets, lmap, (H, W)) in enumerate(
            zip(Ps2, all_dets, label_maps, img_sizes2)):
        if not dets:
            continue
        for pi in range(N):
            res = _project_cv(pts[pi], P)
            if res is None:
                continue
            u, v, _ = res
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < W and 0 <= vi < H):
                continue
            di = int(lmap[vi, ui])
            if di < 0 or di >= len(dets):
                continue
            node = (fi, di)
            if node in node_to_inst:
                pt_votes[pi][node_to_inst[node]] += 1

    final = np.full(N, -1, dtype=np.int32)
    for pi, votes in enumerate(pt_votes):
        if not votes:
            continue
        best = max(votes, key=votes.get)
        if votes[best] >= min_votes:
            final[pi] = inst_to_idx[best]

    # Drop instances that still have too few SfM points after voting
    counts = Counter(int(x) for x in final[final >= 0])
    valid  = {iid for iid, cnt in counts.items() if cnt >= min_label_pts}
    remap  = {old: new for new, old in enumerate(sorted(valid))}
    final  = np.array(
        [remap[int(i)] if int(i) in valid else -1 for i in final],
        dtype=np.int32)
    kept   = [all_inst_names[old] for old in sorted(valid)]

    print(f"[label] Unique instances: {kept}")
    for i, name in enumerate(kept):
        n_frame_dets = sum(len(c["members"]) for c in comp_data if c["instance_id"] == name)
        print(f"  [{i}] '{name}': {int((final==i).sum()):,} pts  "
              f"(seen in {n_frame_dets} frame-detections)")
    print(f"  unlabelled: {int((final==-1).sum()):,} pts")
    return final, kept


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: AABB fitting — one box per instance, no 3-D clustering
# ──────────────────────────────────────────────────────────────────────────────

def _fit_aabb(pts, percentile_trim=0.5):
    lo = np.percentile(pts, percentile_trim, axis=0)
    hi = np.percentile(pts, 100.-percentile_trim, axis=0)
    return {"world_min": lo.tolist(), "world_max": hi.tolist(),
            "center": ((lo+hi)/2).tolist(), "size": (hi-lo).tolist(),
            "num_points": int(pts.shape[0])}


def fit_all_aabbs(pts, label_ids, label_names, percentile_trim=0.5,
                  dbscan_eps=None, dbscan_min_samples=None,
                  max_box_size: float = 0.0):
    """One global AABB + one AABB per instance (label_names entry).
    If max_box_size > 0, instances whose largest AABB dimension exceeds it are dropped."""
    global_aabb = _fit_aabb(pts, percentile_trim)
    label_aabbs: dict[str, dict] = {}
    for lid, name in enumerate(label_names):
        mask = label_ids == lid
        if mask.sum() < 4:
            continue
        aabb = _fit_aabb(pts[mask], percentile_trim)
        if max_box_size > 0 and max(aabb["size"]) > max_box_size:
            print(f"  [filter] '{name}' dropped — max dim "
                  f"{max(aabb['size']):.2f}m > {max_box_size}m")
            continue
        label_aabbs[name] = aabb
    return global_aabb, label_aabbs


# ──────────────────────────────────────────────────────────────────────────────
# Step 5b: Validate 3D boxes by reprojection IoU against 2D DINO detections
# ──────────────────────────────────────────────────────────────────────────────

def _iou_2d(a, b):
    """IoU between two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1])
    ub = (b[2]-b[0])*(b[3]-b[1])
    return inter / (ua + ub - inter)

def _project_aabb(aabb, P, H, W):
    """Return 2D AABB [x1,y1,x2,y2] of the 8 3D corners projected via P, or None."""
    mn = np.array(aabb["world_min"]); mx = np.array(aabb["world_max"])
    corners = np.array([[mn[0],mn[1],mn[2],1],[mx[0],mn[1],mn[2],1],
                         [mn[0],mx[1],mn[2],1],[mx[0],mx[1],mn[2],1],
                         [mn[0],mn[1],mx[2],1],[mx[0],mn[1],mx[2],1],
                         [mn[0],mx[1],mx[2],1],[mx[0],mx[1],mx[2],1]],
                        dtype=np.float64).T       # 4×8
    proj = P @ corners                             # 3×8
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


def validate_aabbs_by_reprojection(
    label_aabbs: dict,          # name → aabb dict from fit_all_aabbs
    seg_frame_list: list[dict],
    all_dets: list[list[dict]], # same order as seg_frame_list
    min_proj_matches: int = 3,  # need this many frames with IoU match
    proj_iou_thresh: float = 0.1,
    Ps: list | None = None,
    img_sizes: list | None = None,
) -> dict:
    """
    For each 3D AABB, project its 8 corners into every seg frame and compute
    the 2D bounding box of the visible silhouette.  Count frames where the
    projected box has IoU ≥ proj_iou_thresh with at least one 2D DINO detection
    of the same label.  Drop boxes with fewer than min_proj_matches such frames.

    Pass cached ``Ps`` / ``img_sizes`` to skip rebuilding them; otherwise they
    are computed from ``seg_frame_list``.
    """
    if Ps is None or img_sizes is None:
        Ps, img_sizes = [], []
        for frame in seg_frame_list:
            P, _meta = _load_and_make_P(frame)
            Ps.append(P)
            img = cv2.imread(str(frame["color"]), cv2.IMREAD_UNCHANGED)
            img_sizes.append(img.shape[:2] if img is not None else _DEFAULT_IMAGE_SIZE)

    valid_aabbs: dict = {}
    print("\n── Step 5b: Validating 3D boxes by reprojection ─────────────────────")
    for name, aabb in label_aabbs.items():
        base_label = _base_label(name)
        frame_matches = 0
        for fi, (P, dets, (H, W)) in enumerate(
                zip(Ps, all_dets, img_sizes)):
            proj_box = _project_aabb(aabb, P, H, W)
            if proj_box is None:
                continue
            # Count frames where the projected 3D box has sufficient IoU
            # with any 2D detection of the same label — label-wide matching
            # means we don't depend on correct instance assignment propagation.
            best_iou = max(
                (_iou_2d(proj_box, det["box_xyxy"])
                 for det in dets if det["label"] == base_label),
                default=0.0,
            )
            if best_iou >= proj_iou_thresh:
                frame_matches += 1

        status = "✓" if frame_matches >= min_proj_matches else "✗ DROPPED"
        print(f"  '{name}': {frame_matches}/{len(seg_frame_list)} frame matches  {status}")
        if frame_matches >= min_proj_matches:
            valid_aabbs[name] = aabb

    print(f"  Kept {len(valid_aabbs)}/{len(label_aabbs)} instances "
          f"(≥{min_proj_matches} frame matches, IoU≥{proj_iou_thresh})")
    return valid_aabbs


def resolve_overlapping_aabbs(
    label_aabbs: dict,
    seg_frame_list: list[dict],
    all_dets: list[list[dict]],
    Ps: list | None = None,
    img_sizes: list | None = None,
) -> dict:
    """
    For each group of same-label 3D boxes that overlap in 3D space, keep only
    the one with the highest mean 2D reprojection IoU against its matched 2D
    detections.  Drop the rest.

    Two AABBs 'overlap' if their axis-aligned intersection has positive volume.
    Pass cached ``Ps`` / ``img_sizes`` to skip rebuilding them.
    """
    if len(label_aabbs) <= 1:
        return label_aabbs

    if Ps is None or img_sizes is None:
        Ps, img_sizes = [], []
        for frame in seg_frame_list:
            P, _meta = _load_and_make_P(frame)
            Ps.append(P)
            img = cv2.imread(str(frame["color"]), cv2.IMREAD_UNCHANGED)
            img_sizes.append(img.shape[:2] if img is not None else _DEFAULT_IMAGE_SIZE)

    def _aabb_overlap(a, b) -> bool:
        """True if two AABBs have any positive-volume intersection."""
        for dim in range(3):
            lo = max(a["world_min"][dim], b["world_min"][dim])
            hi = min(a["world_max"][dim], b["world_max"][dim])
            if hi <= lo:
                return False
        return True

    def _mean_iou_score(name, aabb) -> float:
        """Mean best-IoU of the projected 3D box vs same-label 2D dets."""
        base_label = _base_label(name)
        ious = []
        for P, dets, (H, W) in zip(Ps, all_dets, img_sizes):
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
        return float(np.mean(ious)) if ious else 0.0

    # Group by base label
    from collections import defaultdict as _dd
    by_label: dict = _dd(list)
    for name in label_aabbs:
        base = _base_label(name)
        by_label[base].append(name)

    print("\n── Step 5c: Resolving overlapping same-label boxes ──────────────────")
    result = dict(label_aabbs)

    for base_label, names in by_label.items():
        if len(names) < 2:
            continue

        # Union-Find: merge names that overlap in 3D
        uf = _UnionFind(names)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if _aabb_overlap(label_aabbs[a], label_aabbs[b]):
                    uf.union(a, b)

        groups = uf.components()

        for root, members in groups.items():
            if len(members) < 2:
                continue
            # Score each member and keep the best
            scores = {n: _mean_iou_score(n, label_aabbs[n]) for n in members}
            best = max(scores, key=scores.get)
            dropped = [n for n in members if n != best]
            print(f"  [{base_label}] overlap group {sorted(members)}:")
            for n in members:
                tag = "✓ KEPT" if n == best else "✗ DROPPED"
                print(f"    '{n}'  mean_iou={scores[n]:.3f}  {tag}")
            for n in dropped:
                result.pop(n, None)

    if len(result) == len(label_aabbs):
        print("  No overlapping same-label boxes found.")
    else:
        print(f"  {len(label_aabbs) - len(result)} box(es) removed, "
              f"{len(result)} remaining.")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Step 6: Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def _aabb_corners(aabb):
    mn=np.array(aabb["world_min"]); mx=np.array(aabb["world_max"])
    return np.array([
        [mn[0],mn[1],mn[2]],[mx[0],mn[1],mn[2]],
        [mn[0],mx[1],mn[2]],[mx[0],mx[1],mn[2]],
        [mn[0],mn[1],mx[2]],[mx[0],mn[1],mx[2]],
        [mn[0],mx[1],mx[2]],[mx[0],mx[1],mx[2]],
    ])


def _draw_box(img, aabb, P, colour, thickness=2, label=None):
    corners = _aabb_corners(aabb)
    pts2d   = []
    for c in corners:
        res = _project_cv(c, P)
        pts2d.append((int(round(res[0])), int(round(res[1]))) if res else None)
    for i, j in _BOX_EDGES:
        if pts2d[i] and pts2d[j]:
            cv2.line(img, pts2d[i], pts2d[j], colour, thickness, cv2.LINE_AA)
    if label:
        res = _project_cv(np.array(aabb["center"]), P)
        if res:
            cx, cy = int(round(res[0])), int(round(res[1]))
            cv2.putText(img, label, (cx, cy-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.60, (255,255,255), 3, cv2.LINE_AA)
            cv2.putText(img, label, (cx, cy-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.60, colour, 1, cv2.LINE_AA)


def visualise_boxes(frames, global_aabb, label_aabbs, label_names,
                    output_dir, cam_stride=1):
    vis_dir = output_dir / "visualised"
    vis_dir.mkdir(parents=True, exist_ok=True)
    selected = frames[::cam_stride]
    label_to_lid = {name: lid for lid, name in enumerate(label_names)}
    print(f"\n── Step 6: Visualising on {len(selected)} frames ─────────────────")
    for idx, frame in enumerate(selected):
        raw = cv2.imread(str(frame["color"]), cv2.IMREAD_UNCHANGED)
        if raw is None: continue
        if raw.ndim==3 and raw.shape[2]==4:
            a=raw[:,:,3:4].astype(np.float32)/255.
            bg=np.full_like(raw[:,:,:3],255,dtype=np.float32)
            bgr=(raw[:,:,:3].astype(np.float32)*a+bg*(1-a)).clip(0,255).astype(np.uint8)
        else:
            bgr=raw[:,:,:3].copy() if raw.ndim==3 else raw.copy()
        P, _meta = _load_and_make_P(frame)
        # global_aabb is intentionally not drawn on per-frame images
        for full_name, aabb in label_aabbs.items():
            base = _base_label(full_name)
            lid  = label_to_lid.get(base, 0)
            _draw_box(bgr, aabb, P, _colour(lid), thickness=2, label=full_name)
        stem=re.sub(r"_camera$","",frame["camera"].stem)
        cv2.imwrite(str(vis_dir/f"{stem}_boxes.png"), bgr)
        if (idx+1)%10==0 or idx==len(selected)-1:
            print(f"  [{idx+1:3d}/{len(selected)}]")
    print(f"[vis] Done → {vis_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 7: PLY outputs
# ──────────────────────────────────────────────────────────────────────────────

def save_ply(pts, rgb, label_ids, label_names, path):
    n=pts.shape[0]
    colours=np.zeros((n,3),dtype=np.uint8)
    for i in range(n):
        lid=int(label_ids[i])
        if lid>=0:
            c=_colour(lid); colours[i]=[c[2],c[1],c[0]]
        else:
            colours[i]=(rgb[i].astype(np.float32)*0.35).astype(np.uint8)
    header=(
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode()
    with open(path,"wb") as f:
        f.write(header)
        for i in range(n):
            f.write(struct.pack("<fff",*pts[i].astype(np.float32)))
            f.write(struct.pack("BBB",*colours[i]))
    print(f"[out] PLY cloud : {path}  ({n:,} pts)")


def save_boxes_ply(global_aabb, label_aabbs, label_names, path):
    label_to_lid = {name: lid for lid, name in enumerate(label_names)}
    boxes=[(global_aabb,(255,255,255))]
    for full_name, aabb in label_aabbs.items():
        base = _base_label(full_name)
        lid  = label_to_lid.get(base, 0)
        c    = _colour(lid)
        boxes.append((aabb,(c[2],c[1],c[0])))
    all_verts=[]; all_colors=[]; all_edges=[]; offset=0
    for aabb,rgb_col in boxes:
        corners=_aabb_corners(aabb)
        all_verts.append(corners); all_colors.extend([rgb_col]*8)
        for i,j in _BOX_EDGES: all_edges.append((offset+i,offset+j))
        offset+=8
    verts=np.concatenate(all_verts,axis=0); n_v=verts.shape[0]; n_e=len(all_edges)
    header=(
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n_v}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element edge {n_e}\n"
        "property int vertex1\nproperty int vertex2\n"
        "end_header\n"
    ).encode()
    with open(path,"wb") as f:
        f.write(header)
        for i in range(n_v):
            f.write(struct.pack("<fff",*verts[i].astype(np.float32)))
            f.write(struct.pack("BBB",*all_colors[i]))
        for v1,v2 in all_edges: f.write(struct.pack("<ii",v1,v2))
    print(f"[out] PLY boxes : {path}  ({len(boxes)} boxes)")


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _drive_generator(gen, progress_fn, set_result):
    """Drive a (done, total)-yielding sub-generator, forward progress, capture return.

    ``progress_fn(done, total) -> (frac, label)`` maps each inner yield to an
    outer (progress, label) pair. ``set_result(value)`` is called with the
    inner generator's StopIteration.value.
    """
    try:
        while True:
            done, total = next(gen)
            yield progress_fn(done, total)
    except StopIteration as e:
        set_result(e.value)


class _Pipeline:
    """End-to-end DINO + SAM2 + SfM 3-D bounding box pipeline.

    Stage methods run sequentially in ``run()``. Shared intermediate state
    lives on ``self`` so projection matrices and per-frame caches are computed
    once and reused by later stages.
    """

    def __init__(
        self,
        output_dir,
        frames,
        base_dir=None,
        *,
        device="cuda",
        cam_stride=1,
        seg_frames=20,
        alpha_threshold=0.5,
        pair_window=10,
        n_features=8000,
        max_reproj_err=3.0,
        voxel_size=0.005,
        aabb_pad=0.15,
        box_threshold=0.25,
        text_threshold=0.20,
        min_votes=1,
        min_label_pts=10,
        percentile_trim=0.5,
        dbscan_eps=0.15,
        dbscan_min_samples=10,
        max_box_size=0.0,
        min_proj_matches=5,
        proj_iou_thresh=0.1,
        dino_text=_DINO_TEXT,
        sam2_model=_SAM2_MODEL,
        visualize=False,
    ):
        # Inputs / config
        self.output_dir       = pathlib.Path(output_dir)
        self.frames_in        = frames
        self.base_dir         = base_dir
        self.device           = device
        self.cam_stride       = cam_stride
        self.seg_frames       = seg_frames
        self.alpha_threshold  = alpha_threshold
        self.pair_window      = pair_window
        self.n_features       = n_features
        self.max_reproj_err   = max_reproj_err
        self.voxel_size       = voxel_size
        self.aabb_pad         = aabb_pad
        self.box_threshold    = box_threshold
        self.text_threshold   = text_threshold
        self.min_votes        = min_votes
        self.min_label_pts    = min_label_pts
        self.percentile_trim  = percentile_trim
        self.dbscan_eps       = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.max_box_size     = max_box_size
        self.min_proj_matches = min_proj_matches
        self.proj_iou_thresh  = proj_iou_thresh
        self.dino_text        = dino_text or _DINO_TEXT
        self.sam2_model       = sam2_model
        self.visualize        = visualize

        # Stage outputs (populated by run())
        self.all_frames: list = []
        self.strided: list    = []
        self.feat_data: list  = []
        self.aabb_min = self.aabb_max = None
        self.pts_all = self.rgb_all = self.match_obs = None
        self.seg_frame_list: list  = []
        self.strided_to_seg: dict  = {}
        self.all_dets = self.label_maps = None
        self.label_ids = self.label_names = None
        self.global_aabb: dict = {}
        self.label_aabbs: dict = {}
        self.Ps: list = []          # cached: built once, reused by 4 stages
        self.img_sizes: list = []   # cached alongside Ps

    # ── stage methods ────────────────────────────────────────────────────────

    def _prepare(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = _safe_device(self.device)
        self.all_frames = _resolve_frame_list(self.frames_in, self.base_dir)
        self.strided = self.all_frames[::self.cam_stride]
        if self.all_frames:
            print(f"[data] {len(self.all_frames)} total frames  cam_stride={self.cam_stride} "
                  f"→ {len(self.strided)} used for SfM")

    def _hull_aabb(self) -> bool:
        print("\n── Step 1: Hull AABB ─────────────────────────────────────────────")
        try:
            self.aabb_min, self.aabb_max, _ = compute_hull_aabb(
                self.all_frames, self.cam_stride, self.alpha_threshold, aabb_pad=self.aabb_pad)
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            return False
        print(f"  min={[round(v,3) for v in self.aabb_min.tolist()]}  "
              f"max={[round(v,3) for v in self.aabb_max.tolist()]}")
        return True

    def _extract_sift_features(self):
        """Generator: yields (progress, label) after each frame's SIFT extraction."""
        print("\n── Step 2: SfM triangulation ─────────────────────────────────────")
        n_strided = len(self.strided)
        self.feat_data = []
        for i, frame in enumerate(self.strided):
            fd = _extract_features(frame, self.alpha_threshold, self.n_features)
            self.feat_data.append(fd)
            print(f"  [{i:3d}] {frame['color'].name[:55]:55s}  "
                  f"{len(fd[0]):4d} kps" if fd else f"  [{i:3d}] SKIP")
            p = 0.06 + (i + 1) / n_strided * (0.15 - 0.06)
            yield p, f"Dino+SAM2: SIFT features {i+1}/{n_strided}…"

    def _triangulate(self) -> bool:
        try:
            self.pts_all, self.rgb_all, self.match_obs = build_sfm_cloud(
                self.strided, self.feat_data, self.aabb_min, self.aabb_max,
                self.pair_window, self.max_reproj_err, self.alpha_threshold, self.voxel_size)
            return True
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            return False

    def _select_seg_frames(self):
        if self.seg_frames <= 0 or self.seg_frames >= len(self.all_frames):
            self.seg_frame_list = self.all_frames
        else:
            idx = list(np.linspace(0, len(self.all_frames)-1, self.seg_frames, dtype=int))
            self.seg_frame_list = [self.all_frames[int(i)] for i in idx]
        path_to_si = {str(f["color"]): si for si, f in enumerate(self.seg_frame_list)}
        self.strided_to_seg = {si: path_to_si[str(f["color"])]
                               for si, f in enumerate(self.strided)
                               if str(f["color"]) in path_to_si}
        print(f"\n[seg] Using {len(self.seg_frame_list)} evenly-spaced frames for DINO + SAM2")

    def _build_projection_matrices(self):
        """Build (P, img_size) once for each segmentation frame; downstream
        stages (validate/resolve AABBs) read these instead of rebuilding."""
        self.Ps, self.img_sizes = [], []
        for frame in self.seg_frame_list:
            P, _meta = _load_and_make_P(frame)
            self.Ps.append(P)
            img = cv2.imread(str(frame["color"]), cv2.IMREAD_UNCHANGED)
            self.img_sizes.append(img.shape[:2] if img is not None else _DEFAULT_IMAGE_SIZE)

    def _dino(self):
        """Generator: forwards DINO progress, captures detections to self.all_dets."""
        yield from _drive_generator(
            run_dino_on_frames(self.seg_frame_list, self.device,
                               self.box_threshold, self.text_threshold,
                               dino_text=self.dino_text),
            lambda done, total: (0.22 + done / total * (0.52 - 0.22),
                                 f"Dino+SAM2: DINO detection {done}/{total}…"),
            set_result=lambda v: setattr(self, "all_dets", v),
        )

    def _sam2(self):
        """Generator: forwards SAM2 progress, captures label maps to self.label_maps."""
        yield from _drive_generator(
            run_sam2_on_frames(self.seg_frame_list, self.all_dets,
                               self.device, self.sam2_model),
            lambda done, total: (0.52 + done / total * (0.82 - 0.52),
                                 f"Dino+SAM2: SAM2 mask {done}/{total}…"),
            set_result=lambda v: setattr(self, "label_maps", v),
        )

    def _assign_labels(self):
        self.label_ids, self.label_names = assign_labels_from_masks(
            self.pts_all, self.seg_frame_list, self.all_dets, self.label_maps,
            self.min_votes, self.min_label_pts,
            anchor_min_pts=self.dbscan_min_samples,
            match_obs=self.match_obs,
            strided_to_seg=self.strided_to_seg)
        if self.visualize:
            save_sam2_mask_debug(self.seg_frame_list, self.all_dets, self.label_maps, self.output_dir)

    def _fit_aabbs(self):
        print("\n── Step 5: Fitting AABBs ─────────────────────────────────────────")
        self.global_aabb, self.label_aabbs = fit_all_aabbs(
            self.pts_all, self.label_ids, self.label_names, self.percentile_trim,
            self.dbscan_eps, self.dbscan_min_samples,
            max_box_size=self.max_box_size)
        print(f"  global center : {[round(v,3) for v in self.global_aabb['center']]}")
        for name, aabb in self.label_aabbs.items():
            print(f"  '{name}'  pts={aabb['num_points']:,}  "
                  f"size={[round(v,3) for v in aabb['size']]}")

        if self.min_proj_matches > 0:
            self.label_aabbs = validate_aabbs_by_reprojection(
                self.label_aabbs, self.seg_frame_list, self.all_dets,
                min_proj_matches=self.min_proj_matches,
                proj_iou_thresh=self.proj_iou_thresh,
                Ps=self.Ps, img_sizes=self.img_sizes)

        self.label_aabbs = resolve_overlapping_aabbs(
            self.label_aabbs, self.seg_frame_list, self.all_dets,
            Ps=self.Ps, img_sizes=self.img_sizes)

    def _save(self):
        if self.visualize:
            visualise_boxes(self.all_frames, self.global_aabb, self.label_aabbs,
                            self.label_names, self.output_dir, self.cam_stride)
            save_ply(self.pts_all, self.rgb_all, self.label_ids, self.label_names,
                     self.output_dir/"pointcloud_labelled.ply")
            save_boxes_ply(self.global_aabb, self.label_aabbs, self.label_names,
                           self.output_dir/"boxes_labelled.ply")

    def _final_result(self) -> dict | None:
        if not self.label_aabbs:
            return None
        return {
            "detections": [
                {"label": name, "center": aabb["center"], "dimensions": aabb["size"]}
                for name, aabb in self.label_aabbs.items()
            ]
        }

    def run(self):
        """Drive the pipeline, yielding (progress, label) at each stage boundary."""
        try:
            yield 0.01, "Dino+SAM2: starting"
            self._prepare()
            if not self.all_frames:
                return None
            yield 0.02, "Dino+SAM2: computing scene bounds…"
            if not self._hull_aabb():
                return None
            yield from self._extract_sift_features()
            yield 0.15, "Dino+SAM2: triangulating point cloud…"
            if not self._triangulate():
                return None
            self._select_seg_frames()
            self._build_projection_matrices()
            yield 0.22, "Dino+SAM2: running Grounding DINO…"
            yield from self._dino()
            yield 0.52, "Dino+SAM2: running SAM2 segmentation…"
            yield from self._sam2()
            yield 0.82, "Dino+SAM2: assigning labels to 3-D points…"
            self._assign_labels()
            yield 0.86, "Dino+SAM2: fitting bounding boxes…"
            self._fit_aabbs()
            yield 0.90, "Dino+SAM2: saving visualisations…"
            yield 0.96, "Dino+SAM2: saving outputs…"
            self._save()
            yield 1.0, "Dino+SAM2: complete."
            return self._final_result()
        except Exception:
            traceback.print_exc()
            raise


def run_dino_sam2_pipeline(output_dir, frames, base_dir=None, **kwargs):
    """Generator — yields (progress: float, label: str) between pipeline stages.

    Each entry in ``frames`` must have ``color_filename`` and
    ``camera_json_filename`` as paths relative to ``base_dir`` (or absolute if
    ``base_dir`` is None).

    Returns ``{"detections": [{"label", "center", "dimensions"}]}`` via
    ``StopIteration.value``, or ``None`` if the pipeline produces no detections.
    """
    return _Pipeline(output_dir, frames, base_dir, **kwargs).run()


