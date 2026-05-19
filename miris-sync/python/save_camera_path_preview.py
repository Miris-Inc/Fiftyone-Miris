import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types


def _to_detections(items):
    return fo.Detections(detections=[
        fo.Detection(
            label=item["label"],
            location=item["location"],
            dimensions=item["dimensions"],
            rotation=item.get("rotation", [0.0, 0.0, 0.0]),
        )
        for item in items
    ])


class SaveCameraPathPreview(foo.Operator):
    """Writes the camera-path preview onto the active Miris sample.

    Stores two `fo.Detections` fields:
      - `camera_waypoints`  — one cuboid per user-defined waypoint (labels
                              `wp0`, `wp1`, …). Editable via FiftyOne's
                              annotate-mode TransformControls.
      - `camera_rig_preview` — one tiny cuboid per (frame, rig-camera)
                               sample along the interpolated path (labels
                               `rig_<frame>_<rigname>`). Visualization only.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="save_camera_path_preview",
            label="Save Camera Path Preview",
            unlisted=True,
        )

    def resolve_input(self, _ctx):
        return types.Property(types.Object())

    def execute(self, ctx):
        asset_uuid = ctx.params.get("asset_uuid", "")
        waypoints  = ctx.params.get("waypoints", [])
        rig        = ctx.params.get("rig", [])

        if ctx.dataset is None:
            return {"status": "error", "error": "no dataset loaded"}
        if not asset_uuid:
            return {"status": "error", "error": "missing asset_uuid"}

        sample = next(
            iter(ctx.dataset.match(fo.ViewField("miris_asset_uuid") == asset_uuid)),
            None,
        )
        if sample is None:
            return {"status": "error",
                    "error": f"no sample with miris_asset_uuid {asset_uuid}"}

        # Ensure the dataset schema knows about the preview fields. Without an
        # explicit schema entry the modal can fail to render newly-introduced
        # Detections fields until a full reload.
        schema = ctx.dataset.get_field_schema()
        for field_name in ("camera_waypoints", "camera_rig_preview"):
            if field_name not in schema:
                ctx.dataset.add_sample_field(
                    field_name,
                    fo.EmbeddedDocumentField,
                    embedded_doc_type=fo.Detections,
                )

        sample["camera_waypoints"]   = _to_detections(waypoints)
        sample["camera_rig_preview"] = _to_detections(rig)
        sample.save()

        # Make sure these fields render by default in the sidebar / 3D viewer.
        # FiftyOne's sidebar may hide newly-added fields until they're toggled.
        try:
            app_cfg = ctx.dataset.app_config
            current = list(getattr(app_cfg, "active_fields", []) or [])
            for f in ("camera_waypoints", "camera_rig_preview"):
                if f not in current:
                    current.append(f)
            app_cfg.active_fields = current
            ctx.dataset.save()
        except Exception:
            # Older FiftyOne builds may not support app_config.active_fields —
            # not a hard requirement, the user can toggle them in the sidebar.
            pass

        # Diagnostic peek: confirm the serialized shape looks right.
        first_wp = sample["camera_waypoints"].detections[0] if waypoints else None
        return {
            "status": "ok",
            "waypoints": len(waypoints),
            "rig": len(rig),
            "sample_id": str(sample.id),
            "first_waypoint_preview": (
                {
                    "_cls": first_wp.to_mongo().get("_cls"),
                    "location": first_wp.location,
                    "dimensions": first_wp.dimensions,
                }
                if first_wp is not None else None
            ),
        }
