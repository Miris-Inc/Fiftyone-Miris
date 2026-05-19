import pathlib

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types

from .. import _captures_dir
from .segmentation.generate_bbox_3d import run_bbox_pipeline
from .segmentation.dino_sam2_sfm_bbox import run_dino_sam2_pipeline


class SegmentMirisStreamFrames(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="segment_miris_stream_frames",
            label="Miris: Segment stream frames and generate labels",
            unlisted=True,
            execute_as_generator=True,
        )

    def resolve_input(self, _ctx):
        inputs = types.Object()
        inputs.str("dataset_name", required=False)
        inputs.str("asset_name", required=True)
        inputs.str("asset_uuid", required=True)
        inputs.str("algorithm", required=False,
                   label="Algorithm",
                   description="visual_hull (single bounding box, fast) or dino_sam2 (multiple labeled boxes, slow)")
        inputs.str("dino_text", required=False,
                   label="Object classes (dino_sam2 only)",
                   description="Grounding DINO text prompt, e.g. 'Phone. Laptop.'")
        return types.Property(inputs)

    def execute(self, ctx):
        frames       = ctx.params.get("frames", [])
        dataset_name = ctx.dataset.name if ctx.dataset is not None else ctx.params.get("dataset_name", "unknown")
        asset_name   = ctx.params.get("asset_name", "unknown")
        asset_uuid   = ctx.params.get("asset_uuid", "")
        algorithm    = ctx.params.get("algorithm", "visual_hull") or "visual_hull"
        dino_text    = ctx.params.get("dino_text", "") or ""

        if not frames:
            yield ctx.trigger("@voxel51/operators/notify", params={
                "message": f'Segmentation failed for "{asset_name}" — no frames captured.',
                "variant": "error",
            })
            return

        base_dir = pathlib.Path(_captures_dir(dataset_name))

        if algorithm == "dino_sam2":
            # Route segmentation outputs (SAM2 mask overlays, projected-box
            # PNGs, point cloud PLYs) into the same run folder as the input
            # frames so they're easy to browse side-by-side. The run folder
            # is the first path component of any frame's color_filename
            # (JS-side `buildRunFolder`); falls back to a sentinel if missing.
            first_color = frames[0].get("color_filename", "") if frames else ""
            run_folder = first_color.split("/")[0] if "/" in first_color else "unknown_run"
            output_dir = base_dir / run_folder / "segmentation"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Use every captured frame for both SfM triangulation and
            # DINO+SAM2 inference. seg_frames=0 makes _select_seg_frames
            # fall through to the full frame list.
            yield ctx.log(
                f"[segment] dino_sam2: output_dir={output_dir} frames={len(frames)} (all)"
            )
            gen = run_dino_sam2_pipeline(
                output_dir=output_dir,
                frames=frames,
                base_dir=base_dir,
                dino_text=dino_text,
                cam_stride=1,
                seg_frames=0,
                visualize=True,
            )
        else:
            gen = run_bbox_pipeline(
                frames,
                base_dir=base_dir,
                cam_stride=1,
                pixel_stride=2,
                alpha_threshold=0.5,
                percentile_trim=1.0,
            )

        results = None
        try:
            while True:
                progress, label = next(gen)
                yield ctx.ops.set_progress(progress=progress, label=label)
        except StopIteration as e:
            results = e.value
        except Exception as exc:
            import traceback
            yield ctx.log(f"[segment] CRASH: {type(exc).__name__}: {exc}")
            yield ctx.log(f"[segment] traceback:\n{traceback.format_exc()}")
            yield ctx.trigger("@voxel51/operators/notify", params={
                "message": f'Segmentation crashed for "{asset_name}": {type(exc).__name__}: {exc}',
                "variant": "error",
            })
            return

        yield ctx.log(
            f"[segment] pipeline returned: type={type(results).__name__} "
            f"truthy={bool(results)} "
            f"keys={list(results.keys()) if isinstance(results, dict) else None}"
        )

        if not results:
            yield ctx.log(
                f"[segment] {asset_name}: pipeline returned no detections — "
                f"nothing to save. For dino_sam2 this usually means DINO/SAM2 "
                f"found nothing matching the text prompt; for visual_hull it "
                f"means no foreground points were found."
            )
            return

        if algorithm == "dino_sam2":
            yield ctx.log(f"[segment] building {len(results.get('detections', []))} dino_sam2 detections")
            detections = [
                fo.Detection(
                    label=d["label"],
                    location=d["center"],
                    dimensions=d["dimensions"],
                    rotation=[0.0, 0.0, 0.0],
                )
                for d in results["detections"]
            ]
        else:
            yield ctx.log(f"[segment] building 1 visual_hull detection")
            world_min  = results["world_min"]
            world_max  = results["world_max"]
            center     = [(world_min[i] + world_max[i]) / 2 for i in range(3)]
            dimensions = [world_max[i] - world_min[i] for i in range(3)]
            detections = [fo.Detection(
                label="bbox",
                location=center,
                dimensions=dimensions,
                rotation=[0.0, 0.0, 0.0],
            )]

        yield ctx.log(f"[segment] built {len(detections)} Detection object(s)")

        if ctx.dataset is not None and asset_uuid:
            field = "object_detections" if algorithm == "dino_sam2" else "bounding_box"

            # Ensure the dataset schema knows about the field. FiftyOne's
            # dynamic-field path doesn't always create new top-level Detections
            # fields, so newly-introduced fields can save silently without
            # actually persisting. We add the field up front, idempotently.
            schema = ctx.dataset.get_field_schema()
            if field not in schema:
                yield ctx.log(f"[segment] adding {field} to dataset schema (Detections)")
                ctx.dataset.add_sample_field(
                    field,
                    fo.EmbeddedDocumentField,
                    embedded_doc_type=fo.Detections,
                )

            # Make the field visible in the sidebar / 3D viewer by default.
            try:
                app_cfg = ctx.dataset.app_config
                current = list(getattr(app_cfg, "active_fields", []) or [])
                if field not in current:
                    current.append(field)
                    app_cfg.active_fields = current
                    ctx.dataset.save()
            except Exception:
                pass

            # Log the actual world-space placement so we can spot
            # location-out-of-frustum issues quickly.
            for i, d in enumerate(detections[:3]):
                yield ctx.log(
                    f"[segment] det[{i}] label={d.label!r} "
                    f"location={list(d.location)} dimensions={list(d.dimensions)}"
                )

            yield ctx.log(f"[segment] looking up sample miris_asset_uuid={asset_uuid}")
            sample = next(
                iter(ctx.dataset.match(fo.ViewField("miris_asset_uuid") == asset_uuid)),
                None,
            )
            if sample is None:
                yield ctx.log(f"[segment] sample not found for asset_uuid={asset_uuid}")
            else:
                yield ctx.log(f"[segment] writing {len(detections)} detections to sample.{field}")
                sample[field] = fo.Detections(detections=detections)
                sample.save()
                yield ctx.log(f"[segment] sample.save() done")

                # Verify the save actually persisted — fetch the sample fresh
                # from the dataset and report what's there.
                fresh = ctx.dataset[sample.id]
                stored = fresh.get_field(field)
                stored_count = len(stored.detections) if stored else 0
                yield ctx.log(
                    f"[segment] verify after save: sample[{field}] "
                    f"has {stored_count} detection(s) {'✓' if stored_count == len(detections) else '✗'}"
                )

                # reload_dataset (not reload_samples) is required when we
                # introduce a brand-new top-level field — the sidebar needs to
                # refresh its schema view, not just resample row data.
                yield ctx.trigger("@voxel51/operators/reload_dataset", params={})
                yield ctx.log(f"[segment] reload_dataset yielded")
        else:
            yield ctx.log(
                f"[segment] no dataset or asset_uuid; skipping save "
                f"(ctx.dataset={ctx.dataset is not None}, asset_uuid={bool(asset_uuid)})"
            )

        yield ctx.log(f"[segment] triggering success notify")
        yield ctx.trigger("@voxel51/operators/notify", params={
            "message": f'Segmentation complete for "{asset_name}" — {len(frames)} frame{"s" if len(frames) != 1 else ""} processed.',
            "variant": "success",
        })
        yield ctx.log(f"[segment] done")
