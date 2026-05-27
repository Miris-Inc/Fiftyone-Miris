"""End-to-end DINO + SAM2 + depth pipeline. Public entrypoint: ``run_dino_sam2_pipeline``.

Also exposes ``run_sam3_pipeline``: same pipeline but replaces the DINO + SAM2 pair with a single
SAM3 call (text-promptable detector + segmenter). The geometry / labelling / AABB stages are
identical, so downstream code is untouched.
"""
from __future__ import annotations

import pathlib
import traceback

import numpy as np

from .aabb import (
    fit_all_aabbs,
    resolve_overlapping_aabbs,
    validate_aabbs_by_reprojection,
)
from .depth_cloud import build_depth_cloud, compute_hull_aabb
from .detection import (
    SAM2_MODEL,
    run_dino_on_frames,
    run_sam2_on_frames,
)
from .detection_sam3 import SAM3_MODEL, run_sam3_on_frames
from .geometry import (
    CameraCache,
    _resolve_frame_list,
    _safe_device,
)
from .labels import assign_labels_from_masks
from .visualization import (
    save_boxes_ply,
    save_ply,
    save_sam2_mask_debug,
    visualise_boxes,
)


def _drive_generator(gen, progress_fn, set_result):
    """Forward ``(done, total)`` progress from a sub-generator and capture its return value."""
    try:
        while True:
            done, total = next(gen)
            yield progress_fn(done, total)
    except StopIteration as e:
        set_result(e.value)


class _Pipeline:
    """Stage-by-stage orchestration; per-stage outputs live on ``self`` so later stages can read them."""

    def __init__(
        self,
        output_dir,
        frames,
        base_dir=None,
        *,
        dino_text: str,
        device: str = "cuda",
        cam_stride: int = 1,
        seg_frames: int = 20,
        alpha_threshold: float = 0.5,
        voxel_size: float = 0.005,
        aabb_pad: float = 0.25,
        depth_max=None,
        min_unique_cameras: int = 3,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        min_votes: int = 1,
        min_label_pts: int = 10,
        percentile_trim: float = 0.5,
        dbscan_eps: float = 0.05,
        dbscan_min_samples: int = 20,
        max_box_size: float = 0.0,
        min_proj_matches: int = 5,
        proj_iou_thresh: float = 0.1,
        sam2_model: str = SAM2_MODEL,
        visualize: bool = False,
    ):
        self.output_dir = pathlib.Path(output_dir)
        self.frames_in = frames
        self.base_dir = base_dir
        self.device = device
        self.cam_stride = cam_stride
        self.seg_frames = seg_frames
        self.alpha_threshold = alpha_threshold
        self.voxel_size = voxel_size
        self.aabb_pad = aabb_pad
        self.depth_max = depth_max
        self.min_unique_cameras = min_unique_cameras
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.min_votes = min_votes
        self.min_label_pts = min_label_pts
        self.percentile_trim = percentile_trim
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.max_box_size = max_box_size
        self.min_proj_matches = min_proj_matches
        self.proj_iou_thresh = proj_iou_thresh
        self.dino_text = dino_text
        self.sam2_model = sam2_model
        self.visualize = visualize

        # Stage outputs (populated by run())
        self.all_frames: list = []
        self._aabb_min: np.ndarray | None = None
        self._aabb_max: np.ndarray | None = None
        self.pts_all = self.rgb_all = None
        self.seg_frame_list: list = []
        self.cache: CameraCache | None = None
        self.all_dets = self.label_maps = None
        self.label_ids: np.ndarray | None = None
        self.label_names: list[str] = []
        self.global_aabb: dict = {}
        self.label_aabbs: dict = {}

    # ── stages ────────────────────────────────────────────────────────────────

    def _prepare(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = _safe_device(self.device)
        self.all_frames = _resolve_frame_list(self.frames_in, self.base_dir)
        if self.all_frames:
            print(
                f"[data] {len(self.all_frames)} total frames  cam_stride={self.cam_stride}"
            )

    def _compute_hull(self):
        print("\n── Step 1: Hull AABB ─────────────────────────────────────────────")
        result_box = [None]
        try:
            yield from _drive_generator(
                compute_hull_aabb(
                    self.all_frames, self.cam_stride, self.alpha_threshold,
                    aabb_pad=self.aabb_pad,
                ),
                lambda done, total: (
                    0.02 + done / total * (0.07 - 0.02),
                    f"Dino+SAM2: computing scene bounds… {done}/{total}",
                ),
                set_result=lambda v: result_box.__setitem__(0, v),
            )
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            return
        if result_box[0] is not None:
            self._aabb_min, self._aabb_max, _ = result_box[0]
            print(
                f"  min={[round(v, 3) for v in self._aabb_min.tolist()]}  "
                f"max={[round(v, 3) for v in self._aabb_max.tolist()]}"
            )

    def _build_cloud(self):
        # build_depth_cloud uses a strict alpha_threshold (0.95) by default;
        # soft splat edges have unreliable depth and would paint streaks.
        print("\n── Step 2: Depth → world-space cloud ─────────────────────────────")
        result_box = [None]
        try:
            yield from _drive_generator(
                build_depth_cloud(
                    self.all_frames[::self.cam_stride], self._aabb_min, self._aabb_max,
                    voxel_size=self.voxel_size,
                    depth_max=self.depth_max,
                    min_unique_cameras=self.min_unique_cameras,
                ),
                lambda done, total: (
                    0.07 + done / total * (0.22 - 0.07),
                    f"Dino+SAM2: building depth cloud… {done}/{total}",
                ),
                set_result=lambda v: result_box.__setitem__(0, v),
            )
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            return
        if result_box[0] is not None:
            self.pts_all, self.rgb_all = result_box[0]

    def _select_seg_frames(self):
        if self.seg_frames <= 0 or self.seg_frames >= len(self.all_frames):
            self.seg_frame_list = self.all_frames
        else:
            idx = np.linspace(0, len(self.all_frames) - 1, self.seg_frames, dtype=int)
            self.seg_frame_list = [self.all_frames[int(i)] for i in idx]
        self.cache = CameraCache(self.seg_frame_list)
        print(
            f"\n[seg] Using {len(self.seg_frame_list)} evenly-spaced frames for DINO + SAM2"
        )

    def _dino(self):
        yield from _drive_generator(
            run_dino_on_frames(
                self.seg_frame_list, self.device,
                dino_text=self.dino_text,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            ),
            lambda done, total: (
                0.22 + done / total * (0.52 - 0.22),
                f"Dino+SAM2: DINO detection {done}/{total}…",
            ),
            set_result=lambda v: setattr(self, "all_dets", v),
        )

    def _sam2(self):
        yield from _drive_generator(
            run_sam2_on_frames(
                self.seg_frame_list, self.all_dets, self.device, self.sam2_model,
            ),
            lambda done, total: (
                0.52 + done / total * (0.82 - 0.52),
                f"Dino+SAM2: SAM2 mask {done}/{total}…",
            ),
            set_result=lambda v: setattr(self, "label_maps", v),
        )

    def _assign_labels(self):
        self.label_ids, self.label_names = assign_labels_from_masks(
            self.pts_all, self.cache, self.all_dets, self.label_maps,
            min_votes=self.min_votes,
            min_label_pts=self.min_label_pts,
            anchor_min_pts=self.dbscan_min_samples,
        )
        if self.visualize:
            save_sam2_mask_debug(
                self.seg_frame_list, self.all_dets, self.label_maps, self.output_dir,
            )

    def _fit_aabbs(self):
        print("\n── Step 5: Fitting AABBs ─────────────────────────────────────────")
        self.global_aabb, self.label_aabbs = fit_all_aabbs(
            self.pts_all, self.label_ids, self.label_names, self.percentile_trim,
            self.dbscan_eps, self.dbscan_min_samples,
            max_box_size=self.max_box_size,
        )
        print(f"  global center : {[round(v, 3) for v in self.global_aabb['center']]}")
        for name, aabb in self.label_aabbs.items():
            print(
                f"  '{name}'  pts={aabb['num_points']:,}  "
                f"size={[round(v, 3) for v in aabb['size']]}"
            )

        if self.min_proj_matches > 0:
            self.label_aabbs = validate_aabbs_by_reprojection(
                self.label_aabbs, self.all_dets, self.cache,
                min_proj_matches=self.min_proj_matches,
                proj_iou_thresh=self.proj_iou_thresh,
            )

        self.label_aabbs = resolve_overlapping_aabbs(
            self.label_aabbs, self.all_dets, self.cache,
        )

    def _save(self):
        if not self.visualize:
            return
        visualise_boxes(
            self.all_frames, self.label_aabbs, self.label_names,
            self.output_dir, self.cam_stride,
        )
        save_ply(
            self.pts_all, self.rgb_all, self.label_ids, self.label_names,
            self.output_dir / "pointcloud_labelled.ply",
        )
        save_boxes_ply(
            self.global_aabb, self.label_aabbs, self.label_names,
            self.output_dir / "boxes_labelled.ply",
        )

    def _final_result(self):
        if not self.label_aabbs:
            return None
        return {
            "detections": [
                {"label": name, "center": aabb["center"], "dimensions": aabb["size"]}
                for name, aabb in self.label_aabbs.items()
            ]
        }

    def run(self):
        """Drive the pipeline, yielding ``(progress, label)`` at each stage."""
        try:
            yield 0.01, "Dino+SAM2: starting"
            self._prepare()
            if not self.all_frames:
                return None
            yield from self._compute_hull()
            if self._aabb_min is None:
                return None
            yield from self._build_cloud()
            if self.pts_all is None:
                return None
            self._select_seg_frames()
            yield 0.22, "Dino+SAM2: running Grounding DINO…"
            yield from self._dino()
            yield 0.52, "Dino+SAM2: running SAM2 segmentation…"
            yield from self._sam2()
            yield 0.82, "Dino+SAM2: assigning labels to 3-D points…"
            self._assign_labels()
            yield 0.86, "Dino+SAM2: fitting bounding boxes…"
            self._fit_aabbs()
            yield 0.96, "Dino+SAM2: saving outputs…"
            self._save()
            yield 1.0, "Dino+SAM2: complete."
            return self._final_result()
        except Exception:
            traceback.print_exc()
            raise


def run_dino_sam2_pipeline(output_dir, frames, base_dir=None, *, dino_text: str, **kwargs):
    """Generator → yields ``(progress, label)`` between stages.

    Each ``frames`` entry needs ``color_filename`` + ``camera_json_filename`` (and ideally
    ``depth_filename``), paths relative to ``base_dir`` (or absolute if ``base_dir`` is None).
    Returns ``{"detections": [...]}`` via ``StopIteration.value``, or ``None`` if no detections.
    """
    return _Pipeline(output_dir, frames, base_dir, dino_text=dino_text, **kwargs).run()


# ──────────────────────────────────────────────────────────────────────────────
# SAM3 pipeline (drop-in replacement for DINO + SAM2)
# ──────────────────────────────────────────────────────────────────────────────

class _Sam3Pipeline(_Pipeline):
    """SAM3 variant: text-prompted segmentation produces ``all_dets`` and ``label_maps`` in one stage.

    Reuses every other stage (hull, depth cloud, label assignment, AABB fitting,
    save, final result) from ``_Pipeline`` without modification — only the 2-D
    detection/segmentation stage differs.
    """

    def __init__(
        self,
        output_dir,
        frames,
        base_dir=None,
        *,
        sam3_text: str,
        sam3_model: str = SAM3_MODEL,
        sam3_score_threshold: float = 0.3,
        **kwargs,
    ):
        # ``_Pipeline.__init__`` requires ``dino_text``; satisfy it with the SAM3 prompt
        # so the (unused) DINO knobs don't break instantiation. The actual prompt
        # routed to SAM3 lives in ``self.sam3_text``.
        kwargs.setdefault("dino_text", sam3_text)
        super().__init__(output_dir, frames, base_dir, **kwargs)
        self.sam3_text = sam3_text
        self.sam3_model = sam3_model
        self.sam3_score_threshold = sam3_score_threshold

    def _sam3(self):
        """Single-stage SAM3 detection + segmentation; produces both ``all_dets`` and ``label_maps``."""
        def _set_both(value):
            dets, lmaps = value
            self.all_dets = dets
            self.label_maps = lmaps

        yield from _drive_generator(
            run_sam3_on_frames(
                self.seg_frame_list, self.device,
                sam3_text=self.sam3_text,
                score_threshold=self.sam3_score_threshold,
            ),
            lambda done, total: (
                0.22 + done / total * (0.82 - 0.22),
                f"SAM3: detecting + segmenting {done}/{total}…",
            ),
            set_result=_set_both,
        )

    def run(self):
        """Drive the SAM3 pipeline, yielding ``(progress, label)`` at each stage.

        Mirrors ``_Pipeline.run`` (so we get the same hull + depth-cloud progress
        plumbing introduced on dvj-camera-depth) but replaces the DINO + SAM2
        stages with a single SAM3 call.
        """
        try:
            yield 0.01, "SAM3: starting"
            self._prepare()
            if not self.all_frames:
                return None
            yield from self._compute_hull()
            if self._aabb_min is None:
                return None
            yield from self._build_cloud()
            if self.pts_all is None:
                return None
            self._select_seg_frames()
            yield 0.22, "SAM3: running text-prompted segmentation…"
            yield from self._sam3()
            yield 0.82, "SAM3: assigning labels to 3-D points…"
            self._assign_labels()
            yield 0.86, "SAM3: fitting bounding boxes…"
            self._fit_aabbs()
            yield 0.96, "SAM3: saving outputs…"
            self._save()
            yield 1.0, "SAM3: complete."
            return self._final_result()
        except Exception:
            traceback.print_exc()
            raise


def run_sam3_pipeline(output_dir, frames, base_dir=None, *, sam3_text: str, **kwargs):
    """Generator → yields ``(progress, label)`` between stages.

    Same interface and return shape as :func:`run_dino_sam2_pipeline`, but uses
    SAM3 (single text-prompted call) in place of Grounding DINO + SAM2.
    """
    return _Sam3Pipeline(output_dir, frames, base_dir, sam3_text=sam3_text, **kwargs).run()

