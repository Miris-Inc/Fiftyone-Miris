import os
import urllib.request

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types

# Explicit submodule import. `fo.MirisStream` can fail to resolve when
# `fiftyone` ends up as a namespace package (e.g. site-packages/fiftyone
# contributed only by fiftyone-brain/db has no __init__.py to run, so
# attributes from fiftyone/__public__.py never get attached). Pulling
# MirisStream directly from its module bypasses the attribute lookup —
# the submodule itself resolves via namespace __path__ regardless.
from fiftyone.core.threed.miris_stream import MirisStream
from fiftyone.utils.utils3d import OrthographicProjectionMetadata


def _dataset_dir(dataset_name: str) -> str:
    return os.path.join(os.path.expanduser("~"), "fiftyone", dataset_name)


def _thumbnails_dir(dataset_name: str) -> str:
    return os.path.join(_dataset_dir(dataset_name), "thumbnails")


def _scenes_dir(dataset_name: str) -> str:
    return os.path.join(_dataset_dir(dataset_name), "scenes")


def _cache_thumbnail(uuid: str, url: str, dataset_name: str) -> str | None:
    """Download thumbnail to ~/fiftyone/<dataset>/thumbnails/ and return the path."""
    if not url:
        return None
    dest_dir = _thumbnails_dir(dataset_name)
    os.makedirs(dest_dir, exist_ok=True)
    path_part = url.split("?")[0]
    ext = path_part.rsplit(".", 1)[-1] if "." in path_part else "png"
    local_path = os.path.join(dest_dir, f"{uuid}.{ext}")
    try:
        urllib.request.urlretrieve(url, local_path)
    except Exception:
        return None
    return local_path


def _write_fo3d_scene(uuid: str, viewer_key: str, dataset_name: str) -> str:
    """Write a single-node fo3d scene referencing the Miris asset.

    ``MirisStream`` is a first-class fo3d type in FiftyOne core; the host's
    looker-3d renderer picks it up automatically.
    """
    dest_dir = _scenes_dir(dataset_name)
    os.makedirs(dest_dir, exist_ok=True)
    fo3d_path = os.path.join(dest_dir, f"{uuid}.fo3d")

    scene = fo.Scene()
    scene.add(
        MirisStream(
            name=uuid,
            asset_uuid=uuid,
            viewer_key=viewer_key or None,
        )
    )
    scene.write(fo3d_path)
    return fo3d_path


def _ensure_dataset_app_config(dataset: fo.Dataset) -> None:
    """Ensure the dataset schema has the OPM field for grid thumbnails."""
    schema = dataset.get_field_schema()
    if "miris_opm" not in schema:
        dataset.add_sample_field(
            "miris_opm",
            fo.EmbeddedDocumentField,
            embedded_doc_type=OrthographicProjectionMetadata,
        )
        dataset.save()


class UpsertMirisAsset(foo.Operator):
    """Bridge operator called by the JS sync operator.

    Writes a fo3d scene file containing a single ``fo.MirisStream`` node
    and adds a sample pointing at it. The sample carries an
    ``OrthographicProjectionMetadata`` field whose filepath is the
    cached Miris thumbnail, so the grid renders the thumbnail image
    instead of the procedural 3D placeholder.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="upsert_miris_asset",
            label="Upsert Miris Asset",
            unlisted=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("uuid", required=True)
        inputs.str("name", required=True)
        inputs.str("thumbnail", required=True)
        inputs.str("viewer_key", required=False)
        return types.Property(inputs)

    def execute(self, ctx):
        uuid = ctx.params["uuid"]
        name = ctx.params.get("name", "")
        thumbnail = ctx.params.get("thumbnail", "")
        viewer_key = ctx.params.get("viewer_key", "")
        tags: list[str] = ctx.params.get("tags", [])

        if not thumbnail:
            return {"uuid": uuid, "action": "skipped", "reason": "no thumbnail URL"}

        dataset = ctx.dataset
        if dataset is None:
            raise ValueError(
                "No dataset is currently loaded. "
                "Open a dataset in FiftyOne before syncing Miris assets."
            )

        if viewer_key:
            dataset.info["miris_viewer_key"] = viewer_key
            dataset.save()

        if "bounding_box" not in dataset.get_field_schema():
            dataset.add_sample_field(
                "bounding_box",
                fo.EmbeddedDocumentField,
                embedded_doc_type=fo.Detections,
            )

        thumbnail_path = _cache_thumbnail(uuid, thumbnail, dataset.name)
        if not thumbnail_path:
            return {"uuid": uuid, "action": "skipped", "reason": "thumbnail download failed"}

        fo3d_path = _write_fo3d_scene(uuid, viewer_key, dataset.name)
        _ensure_dataset_app_config(dataset)

        existing: fo.Sample | None = next(
            iter(dataset.match(fo.ViewField("miris_asset_uuid") == uuid)), None
        )

        opm = OrthographicProjectionMetadata(filepath=thumbnail_path)

        if existing is not None:
            existing["filepath"] = fo3d_path
            existing["miris_opm"] = opm
            existing["miris_asset_name"] = name
            existing["miris_thumbnail_url"] = thumbnail
            existing.save()
            return {"uuid": uuid, "action": "updated"}

        sample = fo.Sample(
            filepath=fo3d_path,
            tags=tags,
            miris_asset_uuid=uuid,
            miris_asset_name=name,
            miris_thumbnail_url=thumbnail,
        )
        sample["miris_opm"] = opm
        dataset.add_sample(sample)
        return {"uuid": uuid, "action": "created"}


def register(p):
    p.register(UpsertMirisAsset)
