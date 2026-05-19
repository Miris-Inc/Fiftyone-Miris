"""DINO 2-D detections + SAM2 box-prompted pixel masks; both are ``(done, total)``-yielding generators."""
from __future__ import annotations

import gc
import pathlib
import re
from collections import defaultdict

import cv2
import numpy as np

from .geometry import _load_rgb_np

DINO_MODEL = "IDEA-Research/grounding-dino-base"
SAM2_MODEL = "facebook/sam2.1-hiera-tiny"

def _parse_prompt_words(dino_text: str) -> list[str]:
    """'Wheel. Phone. Laptop.' → ['wheel', 'phone', 'laptop']"""
    return [t.strip().lower() for t in re.split(r"[.\s]+", dino_text) if t.strip()]


def _normalize_label(raw: str, prompt_words: list[str]) -> str:
    """Map a raw DINO label back to the nearest prompt word — DINO sometimes returns the full
    prompt text instead of the matched token; we resolve via exact-token → substring → char-overlap."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Grounding DINO
# ──────────────────────────────────────────────────────────────────────────────

def run_dino_on_frames(
    frame_list: list[dict],
    device: str,
    dino_text: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
):
    """Generator → per-frame ``[{label, score, box_xyxy, instance_id}]``.

    ``instance_id`` is provisionally left→right per label; ``assign_labels_from_masks`` overwrites
    it with the final ID, but the placeholder lets the SAM2 debug overlay run before assignment.
    """
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    print(f"\n── Step 3a: Grounding DINO on {len(frame_list)} frames ─────────────")
    print(f"[dino] Loading {DINO_MODEL} on {device} …")
    processor = AutoProcessor.from_pretrained(DINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL).to(device)
    print("[dino] Model ready.")

    prompt_words = _parse_prompt_words(dino_text)
    all_dets: list[list[dict]] = []

    for idx, frame in enumerate(frame_list):
        rgb = _load_rgb_np(frame["color"])
        image = Image.fromarray(rgb)
        H, W = rgb.shape[:2]
        inputs = processor(images=image, text=dino_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Newer transformers releases renamed the kwarg from ``box_threshold``
        # to ``threshold``; try both.
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

        boxes = results["boxes"].cpu().tolist()
        scores = results["scores"].cpu().tolist()
        labels = results.get("text_labels") or results.get("labels") or []
        labels = [_normalize_label(str(l), prompt_words) for l in labels]
        if idx == 0 and labels:
            print(f"[dino] Prompt words: {prompt_words}")

        dets = [
            {"label": str(l), "score": float(s), "box_xyxy": b}
            for l, s, b in zip(labels, scores, boxes)
        ]

        # Provisional per-label instance rank, sorted by box center_x
        # (left→right). Overwritten later by assign_labels_from_masks but
        # gives a sensible default when that step is skipped or produces no
        # connected components.
        label_boxes: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(dets):
            label_boxes[d["label"]].append(i)
        for lbl, idxs in label_boxes.items():
            idxs.sort(
                key=lambda i: (dets[i]["box_xyxy"][0] + dets[i]["box_xyxy"][2]) / 2
            )
            for rank, i in enumerate(idxs):
                dets[i]["instance_id"] = f"{lbl}_{rank}" if len(idxs) > 1 else lbl

        all_dets.append(dets)
        print(
            f"  [{idx+1:3d}/{len(frame_list)}] {frame['color'].name[:55]:55s}  "
            f"dets={len(dets)}"
        )
        yield idx + 1, len(frame_list)

    del model
    gc.collect()
    if device == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return all_dets


# ──────────────────────────────────────────────────────────────────────────────
# SAM2 box-prompted segmentation
# ──────────────────────────────────────────────────────────────────────────────

def run_sam2_on_frames(
    frame_list: list[dict],
    all_dets: list[list[dict]],
    device: str,
    sam2_model: str = SAM2_MODEL,
):
    """Generator → per-frame label maps ``(H, W) int32`` (pixel = covering det index, -1 = bg).

    Detections are written in ascending score order so higher-confidence masks overwrite lower."""
    import torch
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print(f"\n── Step 3b: SAM2 box-prompted on {len(frame_list)} frames ──────────")
    print(f"[sam2] Loading {sam2_model} on {device} …")
    sam2 = build_sam2_hf(sam2_model, device=device)
    predictor = SAM2ImagePredictor(sam2)
    print("[sam2] Model ready.")

    label_maps: list[np.ndarray] = []

    for idx, (frame, dets) in enumerate(zip(frame_list, all_dets)):
        rgb = _load_rgb_np(frame["color"])
        H, W = rgb.shape[:2]
        lmap = np.full((H, W), -1, dtype=np.int32)

        if not dets:
            label_maps.append(lmap)
            print(
                f"  [{idx+1:3d}/{len(frame_list)}] {frame['color'].name[:50]:50s}  "
                f"no detections — skipped"
            )
            continue

        predictor.set_image(rgb)
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

        label_maps.append(lmap)
        print(
            f"  [{idx+1:3d}/{len(frame_list)}] {frame['color'].name[:50]:50s}  "
            f"masks={n_masks}"
        )
        yield idx + 1, len(frame_list)

    del predictor, sam2
    gc.collect()
    if device == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return label_maps
