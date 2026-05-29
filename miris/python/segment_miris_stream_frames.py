import pathlib

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types

from .. import _captures_dir
from .segmentation import run_dino_sam2_pipeline


def _fail(ctx, message: str, exc: BaseException | None = None, variant: str = "error"):
    """Yield an error notification then raise, so delegated tasks show as failed."""
    yield ctx.trigger("@voxel51/operators/notify", params={
        "message": message,
        "variant": variant,
    })
    if exc is not None:
        raise exc
    raise RuntimeError(message)


class SegmentMirisStreamFrames(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="segment_miris_stream_frames",
            label="Miris: Segment stream frames and generate labels",
            unlisted=True,
            execute_as_generator=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            default_choice_to_delegated=True,
        )

    def resolve_input(self, _ctx):
        inputs = types.Object()
        inputs.str("dataset_name", required=False)
        inputs.str("asset_name", required=True)
        inputs.str("asset_uuid", required=True)
        inputs.str(
            "dino_text",
            required=True,
            label="Object classes",
            description="Grounding DINO text prompt, e.g. 'Phone. Laptop.'",
        )
        return types.Property(inputs)

    def execute(self, ctx):
        frames = ctx.params.get("frames", [])
        dataset_name = (
            ctx.dataset.name if ctx.dataset is not None
            else ctx.params.get("dataset_name", "unknown")
        )
        asset_name = ctx.params.get("asset_name", "unknown")
        asset_uuid = ctx.params.get("asset_uuid", "")
        dino_text = ctx.params["dino_text"]

        if not frames:
            yield from _fail(ctx, f'Segmentation failed for "{asset_name}" — no frames captured.')
            return  # unreachable

        base_dir = pathlib.Path(_captures_dir(dataset_name))

        # Route segmentation outputs (mask overlays, projected-box PNGs, point
        # cloud PLYs — only emitted when visualize=True) into the same run
        # folder as the input frames so they're easy to browse side-by-side.
        # The run folder is the first path component of any frame's
        # color_filename (JS-side `buildRunFolder`); falls back to a sentinel.
        first_color = frames[0].get("color_filename", "") if frames else ""
        run_folder = first_color.split("/")[0] if "/" in first_color else "unknown_run"
        output_dir = base_dir / run_folder / "segmentation"
        output_dir.mkdir(parents=True, exist_ok=True)

        yield from _run_pipeline_generator(
            ctx, frames, base_dir, output_dir, asset_name, asset_uuid, dino_text
        )


def _run_pipeline_generator(ctx, frames, base_dir, output_dir, asset_name, asset_uuid, dino_text):
    # Use every captured frame for both depth-cloud build and DINO+SAM2
    # inference. seg_frames=0 makes the pipeline fall through to the full
    # frame list.
    yield ctx.log(
        f"[segment] output_dir={output_dir} frames={len(frames)} (all)"
    )
    gen = run_dino_sam2_pipeline(
        output_dir=output_dir,
        frames=frames,
        base_dir=base_dir,
        dino_text=dino_text,
        cam_stride=1,
        seg_frames=0,
        visualize=False,
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
        yield from _fail(ctx, f'Segmentation crashed for "{asset_name}": {type(exc).__name__}: {exc}', exc)
        return  # unreachable

    yield ctx.log(
        f"[segment] pipeline returned: type={type(results).__name__} "
        f"truthy={bool(results)} "
        f"keys={list(results.keys()) if isinstance(results, dict) else None}"
    )

    if not results:
        yield ctx.log(
            f"[segment] {asset_name}: pipeline returned no detections — "
            f"nothing to save. Usually means DINO/SAM2 found nothing "
            f"matching the text prompt."
        )
        yield from _fail(ctx,
            f'No detections found for "{asset_name}". '
            f"DINO found no objects matching the text prompt, or the depth cloud "
            f"could not be built (check the console for details).",
            variant="warning",
        )
        return  # unreachable

    yield ctx.log(
        f"[segment] building {len(results.get('detections', []))} detections"
    )
    detections = [
        fo.Detection(
            label=d["label"],
            location=d["center"],
            dimensions=d["dimensions"],
            # Per-instance Euler XYZ (radians). For gravity-locked yaw OBBs
            # only the up-axis component is non-zero; an empty/missing field
            # falls back to identity for compatibility with older pipelines.
            rotation=d.get("rotation", [0.0, 0.0, 0.0]),
        )
        for d in results["detections"]
    ]
    yield ctx.log(f"[segment] built {len(detections)} Detection object(s)")

    if ctx.dataset is not None and asset_uuid:
        field = "object_detections"

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
            yield ctx.log("[segment] sample.save() done")

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
            yield ctx.log("[segment] reload_dataset yielded")
    else:
        yield ctx.log(
            f"[segment] no dataset or asset_uuid; skipping save "
            f"(ctx.dataset={ctx.dataset is not None}, asset_uuid={bool(asset_uuid)})"
        )

    yield ctx.log("[segment] triggering success notify")
    yield ctx.trigger("@voxel51/operators/notify", params={
        "message": f'Segmentation complete for "{asset_name}" — {len(frames)} frame{"s" if len(frames) != 1 else ""} processed.',
        "variant": "success",
    })
    yield ctx.log("[segment] done")


class SegmentMirisStreamFramesFromFolder(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="segment_miris_stream_frames_from_folder",
            label="Miris: Segment stream frames from folder",
            unlisted=False,
            execute_as_generator=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
            default_choice_to_delegated=True,
        )

    def resolve_input(self, _ctx):
        inputs = types.Object()
        inputs.str(
            "dino_text",
            required=True,
            label="Object classes",
            description="Grounding DINO text prompt, e.g. 'Phone. Laptop.'",
        )
        inputs.str(
            "frames_folder",
            required=True,
            label="Frames folder",
            description="Path to folder containing captured frame images",
        )
        return types.Property(inputs)

    def execute(self, ctx):
        dino_text = ctx.params["dino_text"]
        base_dir = pathlib.Path(ctx.params["frames_folder"])

        if not ctx.current_sample:
            yield from _fail(ctx, "No sample is currently loaded. Open a sample in the modal before running this operator.")
            return  # unreachable

        sample = ctx.dataset[ctx.current_sample]
        asset_uuid = sample.get_field("miris_asset_uuid") or ""
        if not asset_uuid:
            yield from _fail(ctx, "The current sample does not have a miris_asset_uuid field. This operator can only be run on Miris samples.")
            return  # unreachable

        asset_name = sample.get_field("miris_asset_name") or "unknown"

        frames = []
        for color_file in sorted(base_dir.glob("*_color.png")):
            base = color_file.name.removesuffix("_color.png")
            camera_file = base_dir / (base + "_camera.json")
            if not camera_file.exists():
                continue
            entry = {
                "color_filename": color_file.name,
                "camera_json_filename": camera_file.name,
            }
            for _depth_ext in ("_depth.npy", "_depth.png"):
                depth_file = base_dir / (base + _depth_ext)
                if depth_file.exists():
                    entry["depth_filename"] = depth_file.name
                    break
            frames.append(entry)

        if not frames:
            yield from _fail(ctx, f'Segmentation failed for "{asset_name}" — no frames found in folder.')
            return  # unreachable

        output_dir = base_dir / "segmentation"
        output_dir.mkdir(parents=True, exist_ok=True)

        yield from _run_pipeline_generator(
            ctx, frames, base_dir, output_dir, asset_name, asset_uuid, dino_text
        )
