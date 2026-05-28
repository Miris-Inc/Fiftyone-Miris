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

    Stores three fields:
      - `camera_waypoints`   — one cuboid per user-defined waypoint (labels
                               `wp0`, `wp1`, …). Editable via annotate-mode.
      - `camera_rig_preview` — one tiny cuboid per actual capture position.
      - `camera_path`        — fo.Polylines polyline tracing the waypoint path.
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
        polyline   = ctx.params.get("polyline", [])  # [[x,y,z], ...]

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

        # Ensure the dataset schema knows about the preview fields.
        schema = ctx.dataset.get_field_schema()
        for field_name, doc_type in (
            ("camera_waypoints",   fo.Detections),
            ("camera_rig_preview", fo.Detections),
            ("camera_path",        fo.Polylines),
        ):
            if field_name not in schema:
                ctx.dataset.add_sample_field(
                    field_name,
                    fo.EmbeddedDocumentField,
                    embedded_doc_type=doc_type,
                )

        sample["camera_waypoints"]   = _to_detections(waypoints)
        sample["camera_rig_preview"] = _to_detections(rig)

        if polyline:
            # points3d takes a list-of-lists-of-[x,y,z]; one sub-list per
            # connected component — we have a single open path.
            sample["camera_path"] = fo.Polylines(polylines=[
                fo.Polyline(
                    label="path",
                    points3d=[[[pt[0], pt[1], pt[2]] for pt in polyline]],
                    closed=False,
                    filled=False,
                )
            ])
        else:
            sample["camera_path"] = None

        sample.save()

        # Make sure these fields render by default in the sidebar / 3D viewer.
        try:
            app_cfg = ctx.dataset.app_config
            current = list(getattr(app_cfg, "active_fields", []) or [])
            for f in ("camera_waypoints", "camera_rig_preview", "camera_path"):
                if f not in current:
                    current.append(f)
            app_cfg.active_fields = current
            ctx.dataset.save()
        except Exception:
            pass

        first_wp = sample["camera_waypoints"].detections[0] if waypoints else None
        return {
            "status": "ok",
            "waypoints": len(waypoints),
            "rig": len(rig),
            "polyline_pts": len(polyline),
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
