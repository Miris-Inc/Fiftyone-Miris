"""SAM3 text-prompted detection + segmentation in one shot.

Drop-in replacement for the DINO + SAM2 pair in ``detection.py``. Produces the same
two downstream structures so the rest of the pipeline (``assign_labels_from_masks``,
``validate_aabbs_by_reprojection``, ``fit_all_aabbs``) is unchanged:

* ``all_dets``  : ``list[list[dict]]``  per frame, each det has
                  ``{label, score, box_xyxy, instance_id}``.
* ``label_maps``: ``list[np.ndarray]``  per frame ``(H, W) int32``, pixel = index
                  into that frame's ``all_dets[fi]`` (or ``-1`` for background).

SAM3 takes a *single concept* per text prompt. To preserve the existing multi-class
prompt syntax (``"Phone. Laptop."``) we split on ``.`` / whitespace (same as the DINO
path) and call SAM3 once per concept per frame, merging results.
"""
from __future__ import annotations

import gc
from collections import defaultdict

import numpy as np

from .detection import _parse_prompt_words
from .geometry import _load_rgb_np

SAM3_MODEL = "facebook/sam3"


def run_sam3_on_frames(
    frame_list: list[dict],
    device: str,
    sam3_text: str,
    score_threshold: float = 0.3,
):
    """Generator → yields ``(done, total)``; returns ``(all_dets, label_maps)`` via ``StopIteration.value``.

    ``sam3_text`` follows the same convention as the old ``dino_text`` — concepts
    are separated by ``.``  (e.g. ``"Phone. Laptop."``). Each concept is submitted
    to SAM3 separately, then merged per-frame.

    ``score_threshold`` filters out low-confidence detections (SAM3 returns scores
    in roughly [0, 1]). Detections are written into ``lmap`` in ascending score order
    so higher-confidence masks overwrite lower ones.
    """
    import torch
    from PIL import Image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    prompt_words = _parse_prompt_words(sam3_text)
    if not prompt_words:
        print("[sam3] empty prompt — returning no detections")
        all_dets: list[list[dict]] = [[] for _ in frame_list]
        label_maps: list[np.ndarray] = []
        for f in frame_list:
            rgb = _load_rgb_np(f["color"])
            label_maps.append(np.full(rgb.shape[:2], -1, dtype=np.int32))
        return all_dets, label_maps

    print(f"\n── SAM3 (text-prompted) on {len(frame_list)} frames ─────────────────")
    print(f"[sam3] Loading {SAM3_MODEL} on {device} …")
    model = build_sam3_image_model()
    # build_sam3_image_model() loads the default checkpoint; move it to the
    # requested device if it isn't there already. Some sam3 builds accept a
    # device kwarg; if not, fall back to a manual ``.to``.
    try:
        model.to(device)
    except Exception:
        pass
    processor = Sam3Processor(model)
    print(f"[sam3] Model ready. Prompt concepts: {prompt_words}")

    all_dets = []
    label_maps = []

    for idx, frame in enumerate(frame_list):
        rgb = _load_rgb_np(frame["color"])
        H, W = rgb.shape[:2]
        image = Image.fromarray(rgb)

        try:
            state = processor.set_image(image)
        except Exception as e:
            print(f"  [sam3 warn] frame {idx}: set_image failed: {e}")
            all_dets.append([])
            label_maps.append(np.full((H, W), -1, dtype=np.int32))
            yield idx + 1, len(frame_list)
            continue

        # Run one inference per concept; each call returns all instances of that concept.
        per_concept: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for concept in prompt_words:
            try:
                out = processor.set_text_prompt(state=state, prompt=concept)
            except Exception as e:
                print(f"  [sam3 warn] frame {idx} concept '{concept}': {e}")
                continue

            masks = _to_numpy(out.get("masks"))
            boxes = _to_numpy(out.get("boxes"))
            scores = _to_numpy(out.get("scores"))
            if masks is None or boxes is None or scores is None:
                continue
            if masks.size == 0:
                continue
            # masks may be (N, H, W) or (N, 1, H, W) — squeeze the channel.
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            # Boolean-ize: SAM3 returns either bool or float in [0, 1].
            masks_bool = masks > 0.5 if masks.dtype != np.bool_ else masks
            per_concept.append((concept, masks_bool, boxes.astype(np.float32), scores.astype(np.float32)))

        # Flatten to a single det list and build the merged label map.
        dets: list[dict] = []
        mask_stack: list[np.ndarray] = []
        for concept, masks_bool, boxes, scores in per_concept:
            for m, b, s in zip(masks_bool, boxes, scores):
                if float(s) < score_threshold:
                    continue
                dets.append({
                    "label": concept,
                    "score": float(s),
                    "box_xyxy": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                })
                mask_stack.append(m)

        lmap = np.full((H, W), -1, dtype=np.int32)
        if dets:
            # Write masks in ascending score order so high-confidence overwrites low.
            order = sorted(range(len(dets)), key=lambda i: dets[i]["score"])
            for det_idx in order:
                mask = mask_stack[det_idx]
                if mask.shape != (H, W):
                    # Defensive: resize if SAM3 returns at a different resolution.
                    import cv2
                    mask = cv2.resize(
                        mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                lmap[mask] = det_idx

        # Provisional per-label instance rank (left→right by box center_x), same
        # convention as the DINO path so visualizations and assign_labels see a
        # sensible default if downstream voting yields nothing.
        label_boxes: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(dets):
            label_boxes[d["label"]].append(i)
        for lbl, idxs in label_boxes.items():
            idxs.sort(key=lambda i: (dets[i]["box_xyxy"][0] + dets[i]["box_xyxy"][2]) / 2)
            for rank, i in enumerate(idxs):
                dets[i]["instance_id"] = f"{lbl}_{rank}" if len(idxs) > 1 else lbl

        all_dets.append(dets)
        label_maps.append(lmap)
        print(
            f"  [{idx+1:3d}/{len(frame_list)}] {frame['color'].name[:50]:50s}  "
            f"dets={len(dets)}"
        )
        yield idx + 1, len(frame_list)

    del processor, model
    gc.collect()
    if device == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return all_dets, label_maps


def _to_numpy(x):
    """Convert a torch tensor / list / array to a numpy ndarray (or None)."""
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)

