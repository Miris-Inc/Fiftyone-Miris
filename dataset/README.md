# Miris demo dataset (FiftyOne zoo dataset)

A curated snapshot of Miris 3D assets, packaged as a
[remotely-sourced FiftyOne zoo dataset](https://docs.voxel51.com/dataset_zoo/remote.html).
Anyone can load it directly from this GitHub repo — no clone, no plugin
install required:

```python
import fiftyone.zoo as foz
import fiftyone as fo

# Repo shorthand (resolves to the default branch). Or pass the full
# https://github.com/Miris-Inc/Fiftyone-Miris URL.
dataset = foz.load_zoo_dataset("Miris-Inc/Fiftyone-Miris")

session = fo.launch_app(dataset)
```

`load_zoo_dataset` walks the repo, finds this folder's `fiftyone.yml`
(`type: dataset`), copies the folder into your local zoo directory, and
imports the snapshot. The plugin manifest at `miris/fiftyone.yml`
(`type: plugin`) is ignored by the dataset resolver.

## What's in a sample

Each sample mirrors what the `@miris-inc/voxel51` plugin's sync operator
produces:

| Field | Description |
|-------|-------------|
| `filepath` | A `.fo3d` scene referencing a first-class `MirisStream` node, streamed and rendered natively by FiftyOne core's looker-3d. |
| `miris_opm` | `OrthographicProjectionMetadata` whose `filepath` is a cached thumbnail, so the grid renders a 2D preview instead of the procedural 3D placeholder. |
| `miris_asset_uuid` | The Miris asset UUID. |
| `miris_asset_name` | Human-readable asset name. |
| `miris_thumbnail_url` | Source thumbnail URL. |
| `camera_waypoints` | `Detections` marking generated camera waypoints in the scene. |
| `camera_rig_preview` | `Detections` previewing the camera rig. |
| `camera_path` | `Polylines` tracing the generated camera path. |
| `object_detections` | `Detections` from the 3D label-generation pipeline. |

The Miris viewer key is preserved in `dataset.info["miris_viewer_key"]`.

## Files

```
dataset/
├── fiftyone.yml          # Zoo dataset manifest (type: dataset)
├── __init__.py           # download_and_prepare() → FiftyOneDataset importer
├── export_snapshot.py    # Helper to (re)generate the snapshot from a built dataset
├── README.md
├── metadata.json         # ┐
├── samples.json          # ├─ FiftyOneDataset snapshot (generated; commit these)
├── data/                 # │   .fo3d scene files
└── fields/miris_opm/     # ┘   thumbnail images
```

## Regenerating the snapshot

1. Build the dataset in the FiftyOne App (e.g. run the `sync_miris_assets`
   operator to populate samples), and confirm the grid thumbnails render.
2. Export it into this folder:

   ```bash
   python dataset/export_snapshot.py <your_dataset_name>
   ```

3. Update `size_samples` in `fiftyone.yml` to the reported count and commit
   `metadata.json`, `samples.json`, `data/`, and `fields/`.

> Verified on FiftyOne 1.16.0: a `FiftyOneDataset` export with
> `export_media=True` copies both the `.fo3d` scenes and the `miris_opm`
> thumbnails, stores **relative** paths, and re-resolves correctly after the
> folder is copied into the zoo directory — so the grid renders out of the box.
