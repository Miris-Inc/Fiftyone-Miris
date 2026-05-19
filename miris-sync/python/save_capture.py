import os

import fiftyone.operators as foo
import fiftyone.operators.types as types

from .. import _captures_dir


def _save_artifact(ctx, filename: str, content: bytes) -> dict:
    dataset_name = ctx.dataset.name if ctx.dataset is not None else "unknown"
    dest_dir = _captures_dir(dataset_name)
    file_path = os.path.join(dest_dir, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, "wb") as f:
            f.write(content)
        return {"status": "ok", "path": file_path}
    except Exception as e:
        return {"status": "error", "error": str(e)}


class SaveCaptureBatch(foo.Operator):
    """Receives a batch of PNGs and JSON files from JS and writes them all at once."""

    @property
    def config(self):
        return foo.OperatorConfig(
            name="save_capture_batch",
            label="Save Capture Batch",
            unlisted=True,
        )

    def resolve_input(self, _ctx):
        return types.Property(types.Object())

    def execute(self, ctx):
        import base64
        import json as jsonlib

        png_items = ctx.params.get("png_items") or []
        json_items = ctx.params.get("json_items") or []
        errors: list[str] = []
        saved = 0

        for item in png_items:
            r = _save_artifact(ctx, item["filename"], base64.b64decode(item["png_base64"]))
            if r["status"] == "ok":
                saved += 1
            else:
                errors.append(r["error"])

        for item in json_items:
            content = jsonlib.dumps(jsonlib.loads(item["json_str"]), indent=2).encode()
            r = _save_artifact(ctx, item["filename"], content)
            if r["status"] == "ok":
                saved += 1
            else:
                errors.append(r["error"])

        if errors:
            return {"status": "error", "error": "; ".join(errors), "saved": saved}
        return {"status": "ok", "saved": saved}
