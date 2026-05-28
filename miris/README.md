# Miris for FiftyOne — Plugin

A FiftyOne plugin that syncs your [Miris](https://miris.com) asset library into a FiftyOne dataset and generates 3D bounding box labels via an offscreen capture + Grounding DINO + SAM2 pipeline.

Rendering of Miris streams is built into FiftyOne core (looker-3d) as the first-class `fo.MirisStream` fo3d type — this plugin populates datasets with samples that point at your Miris assets, and provides the tooling to label them.

## Features

- **One-click sync** — pulls your Miris asset list and upserts one FiftyOne sample per asset.
- **One sample per asset** — `filepath` is a tiny `.fo3d` scene containing a single `MirisStream` node; thumbnails are cached locally and used for the grid.
- **Idempotent** — re-running the sync operator updates existing samples in place (matched by `miris_asset_uuid`).
- **3D label generation** — fly virtual cameras around a live stream, run Grounding DINO + SAM2 on the captured frames, and write 3D bounding box `Detections` back to the sample.
- **Re-run from folder** — if you already have a capture folder, re-run the segmentation pipeline without re-capturing.
- **3D label overlays** — any `Detections` field on a sample is rendered by FiftyOne's native looker-3d.
- **Delegated execution** — label generation runs as a background task via FiftyOne's delegated worker; progress and status are visible in the Runs panel.

## Prerequisites

- **FiftyOne** ≥ 1.14 with built-in `MirisStream` support (`pip install fiftyone`)
- **Node.js** ≥ 18 and **Yarn** (for building the JS bundle)
- **Miris account** with a viewer key — create one at [app.miris.com](https://app.miris.com)
- **HTTPS** — Miris streaming uses WebAssembly which requires a secure context. The FiftyOne dev server runs HTTP, so a reverse proxy is needed.
- **Delegated worker** — label generation operators run as delegated background tasks; a worker process must be running alongside the app (see [Running the delegated worker](#running-the-delegated-worker)).

## Installation

### 1. Symlink the plugin into FiftyOne

**PowerShell (run as Administrator, or with Developer Mode enabled):**

```powershell
$pluginsDir = python -c "import fiftyone as fo; print(fo.config.plugins_dir)"
New-Item -ItemType SymbolicLink -Path "$pluginsDir\miris-viewer" -Target "$PWD\miris"
```

> Windows requires either Administrator privileges or Developer Mode (`Settings → System → For developers → Developer Mode`) to create symlinks.

### 2. Build the JS bundle

```powershell
cd miris\js
yarn install
yarn build
```

The build produces `js/dist/index.umd.js`. React, Three.js, and `@fiftyone/*` are externalized against FiftyOne's runtime globals — only `@miris-inc/three` is bundled.

### 3. Install Python dependencies

The plugin ships a `requirements.txt` with the packages needed by the label generation pipeline (numpy, opencv, torch, transformers, sam2, scikit-learn, …). Install them into the **same Python env that runs FiftyOne**:

```powershell
fiftyone plugins requirements @miris-inc/voxel51 --install
```

| Command | Purpose |
|---|---|
| `fiftyone plugins requirements @miris-inc/voxel51 --print`   | Show requirements without installing. |
| `fiftyone plugins requirements @miris-inc/voxel51 --install` | Run `pip install -r requirements.txt`. |
| `fiftyone plugins requirements @miris-inc/voxel51 --ensure`  | Verify requirements are satisfied; fail if not. |

If you have an NVIDIA GPU, install a CUDA-enabled `torch` **before** running `--install` — see https://pytorch.org/get-started/locally/ for the right command for your CUDA version. Without it, `torch` falls back to CPU and label generation will be very slow.

### 4. Launch FiftyOne

```python
import fiftyone as fo

dataset = fo.Dataset("miris-demo", persistent=True)
fo.launch_app(dataset, port=5151)
```

### 5. Running the delegated worker

Label generation operators (`segment_miris_stream_frames`, `segment_miris_stream_frames_from_folder`) run as delegated background tasks. A worker process must be running in a separate terminal in the same Python environment:

```powershell
fiftyone delegated launch
```

Keep this running alongside the FiftyOne app. Progress and results appear in the **Runs** panel in the FiftyOne sidebar (the clock icon).

## Usage

### Sync your Miris assets

1. Open the operator browser (press `` ` ``) and run **Sync Miris Assets**.
2. Optionally paste a viewer key. If left blank, the bundled demo key is used.
3. The operator fetches your asset list and, for each asset:
   - Downloads the thumbnail to `~/fiftyone/<dataset>/thumbnails/<uuid>.<ext>`
   - Writes a `.fo3d` scene to `~/fiftyone/<dataset>/scenes/<uuid>.fo3d`
   - Creates or updates a sample with `filepath`, `thumbnail_path`, `miris_asset_uuid`, `miris_asset_name`, and `miris_thumbnail_url`

Re-running sync is safe — existing samples are matched by `miris_asset_uuid` and updated in place.

### View an asset

Click any sample in the grid. FiftyOne's modal opens the `.fo3d` scene, core's looker-3d mounts the `MirisStream` node, and the asset streams in.

### Generate 3D labels

1. Open a sample in the modal so the stream is loaded.
2. Run **Generate Miris Labels** from the operator browser.
3. Fill in the fields:
   - **Object classes** — Grounding DINO text prompt, e.g. `Car. Window. Traffic Light.`
   - **Capture duration** and **Capture rate** — how many frames to capture.
   - **Camera path waypoints** (optional) — `[[x,y,z],...]` JSON defining where the cameras fly. If omitted, the operator uses any `wp*` cuboid annotations on the sample, or a single default-framed view.
   - **Points of interest** (optional) — `[[x,y,z],...]` JSON for gaze targets.
   - **Capture mode** — `rig` (multi-camera rig), `nearest_poi`, or `coverage_greedy`.
4. The operator closes immediately; the pipeline runs in the background:
   - **Phase 1 (JS):** Flies virtual cameras through the scene, saving color PNGs, float32 depth NPYs, and camera JSON files to `~/fiftyone/<dataset>/captures/<run_folder>/`.
   - **Phase 2 (delegated Python):** Builds a depth cloud, runs Grounding DINO + SAM2, fits 3D bounding boxes, and writes a `Detections` field (`object_detections`) to the sample.
5. Monitor progress in the **Runs** panel. When complete, the detections appear in the 3D viewer.

### Preview the camera path

Run **Preview Camera Path** to visualise the waypoints and capture positions in the 3D viewer before committing to a full label generation run.

### Re-run segmentation from a capture folder

If you already have a capture folder and want to re-run segmentation with different parameters (e.g., a different text prompt):

1. Open the sample in the modal.
2. Run **Segment Miris Stream Frames from Folder**.
3. Set the path to the existing capture folder and the new text prompt.

This submits a delegated task that skips the capture phase entirely.

### Add labels manually

```python
import fiftyone as fo

dataset = fo.load_dataset("miris-demo")
sample = dataset.first()
sample["object_detections"] = fo.Detections(detections=[
    fo.Detection(
        label="robot_arm",
        location=[0, 5, 0],       # center [x, y, z]
        dimensions=[10, 20, 10],  # size [w, h, d]
        rotation=[0, 0, 0],       # radians [rx, ry, rz]
    ),
])
sample.save()
```

### Viewer key resolution

When core mounts a `MirisStream` node it resolves the viewer key in this order:

1. `viewer_key` on the fo3d node (set by `upsert_miris_asset` if a key was passed to sync)
2. `dataset.info["miris_viewer_key"]` (also set by sync)

If neither is set, core logs a warning and skips streaming.

## Operators

| Operator | Source | Visible | Purpose |
|---|---|---|---|
| `sync_miris_assets` | JS | yes | Fetches the Miris asset list and upserts one sample per asset. |
| `upsert_miris_asset` | Python | unlisted | Bridge called per-asset by `sync_miris_assets`: caches thumbnail, writes `.fo3d`, creates/updates sample. |
| `generate_miris_labels` | JS | yes | Runs the full offscreen-capture → delegated-segmentation pipeline on the current sample. |
| `preview_camera_path` | JS | yes | Renders the planned camera path and waypoints as 3D annotations in the viewer. |
| `save_capture_batch` | Python | unlisted | JS→Python file bridge: decodes and writes color PNGs, depth NPYs, and camera JSONs to the captures directory. |
| `save_camera_path_preview` | Python | unlisted | JS→Python bridge: writes waypoint/rig/polyline detections to the sample for path preview. |
| `segment_miris_stream_frames` | Python | unlisted | **Delegated.** Runs DINO+SAM2 depth pipeline on captured frames and writes `object_detections` to the sample. |
| `segment_miris_stream_frames_from_folder` | Python | yes | **Delegated.** Same pipeline as above, reading frames from an existing folder on disk instead of from a live capture. |

## Architecture

```
Browser (FiftyOne App)
└── looker-3d <Canvas>                        FiftyOne core — renders fo3d scenes
    └── fo3d scene (sample.filepath)
        └── MirisStream node                  First-class fo3d type in core
            └── new MirisStream({ uuid, viewerKey })  Streams via WASM/WebSocket

── Sync flow ──────────────────────────────────────────────────────────────────
JS  SyncMirisAssets
        │  fetch assets via MirisScene.fetchAssets()
        └── for each asset:
              executeOperator("upsert_miris_asset", { uuid, name, thumbnail, viewer_key })
Python  UpsertMirisAsset
        ├── download thumbnail  → ~/fiftyone/<ds>/thumbnails/
        ├── write .fo3d scene   → ~/fiftyone/<ds>/scenes/
        └── create/update sample (matched by miris_asset_uuid)

── Label generation flow ──────────────────────────────────────────────────────
JS  GenerateMirisLabels.execute()
    │
    ├── setupOffscreenScene()           Spin up an offscreen WebGL context
    │
    ├── runCapture()                    Fly cameras, render color + depth
    │   │  for each slot:
    │   │    render color PNG  (RGBA, alpha = object silhouette)
    │   │    render depth NPY  (float32 log-encoded, via SplatDepthColor mode)
    │   │    build camera JSON (intrinsics, world_to_camera, depth_range)
    │   └── SaveCaptureBatch  (Python)  Write files to ~/fiftyone/<ds>/captures/<run>/
    │
    └── executeOperator("segment_miris_stream_frames", { requestDelegation: true })

Python worker  SegmentMirisStreamFrames   (delegated)
    ├── Step 1: Hull AABB               Alpha silhouettes → coarse world-space box
    ├── Step 2: Depth cloud             Unproject depth NPYs → world-space point cloud
    ├── Step 3: Grounding DINO          Detect objects in sampled color frames
    ├── Step 4: SAM2                    Segment masks for each DINO detection
    ├── Step 5: Label assignment        Project cloud points → masks → label each point
    ├── Step 6: Fit AABBs               DBSCAN cluster per label → 3D bounding box
    └── Write fo.Detections             sample["object_detections"] = fo.Detections(...)
```

## Plugin Layout

```
miris/
├── fiftyone.yml                         Plugin manifest (name, version, operators, js_bundle)
├── __init__.py                          Python: UpsertMirisAsset + register()
├── requirements.txt                     Python deps for label generation
├── README.md                            This file
├── python/
│   ├── __init__.py
│   ├── save_capture.py                  SaveCaptureBatch operator
│   ├── save_camera_path_preview.py      SaveCameraPathPreview operator
│   ├── segment_miris_stream_frames.py   SegmentMirisStreamFrames + FromFolder + _fail helper
│   └── segmentation/
│       ├── __init__.py
│       ├── pipeline.py                  _Pipeline orchestrator + run_dino_sam2_pipeline()
│       ├── depth_cloud.py               load_depth_map, build_depth_cloud, compute_hull_aabb
│       ├── detection.py                 run_dino_on_frames, run_sam2_on_frames
│       ├── aabb.py                      fit_all_aabbs, resolve_overlapping_aabbs
│       ├── geometry.py                  CameraCache, _resolve_frame_list, unprojection helpers
│       ├── labels.py                    assign_labels_from_masks
│       └── visualization.py             save_ply, save_boxes_ply, visualise_boxes
└── js/
    ├── package.json
    ├── vite.config.ts                   UMD build, classic JSX, externalized globals
    ├── tsconfig.json
    ├── yarn.lock
    └── src/
        ├── index.tsx                    Plugin entry: boot WASM, register operators
        ├── fiftyone.d.ts                Local type stubs for @fiftyone/operators
        ├── syncMirisAssets.ts           SyncMirisAssets operator
        ├── generateMirisLabels.ts       GenerateMirisLabels operator (orchestrates pipeline)
        ├── previewCameraPath.ts         PreviewCameraPath operator
        ├── captureStreamFrames.ts       runCapture(), buildCameraJson(), CaptureMode enum
        ├── offscreenScene.ts            setupOffscreenScene() — offscreen WebGL context
        ├── cameraRig.ts                 RIG_CAMERAS, createRigCamera(), interpolatePath()
        ├── waypoints.ts                 readWaypointsFromSample()
        └── utils.ts                     executeOperatorAndReturn()
```

## Development

### Watch mode

```powershell
cd miris\js
yarn dev   # vite build --watch
```

Reload the FiftyOne page after each rebuild — FiftyOne picks up the new bundle on reload.

### Build notes (`vite.config.ts`)

- **`minify: false`** — Vite's minifier shadows UMD factory parameter names, breaking the externalized-import lookup.
- **`define: { "process.env.NODE_ENV": ... }`** — Three.js reads `process.env`, which doesn't exist in the browser.
- **`react({ jsxRuntime: "classic" })`** — FiftyOne exposes `window.React` but not `react/jsx-runtime`, so JSX must compile to `React.createElement` calls.
- **Externalized globals:**

  | Module | Global |
  |---|---|
  | `react` | `React` |
  | `react-dom` | `ReactDOM` |
  | `@fiftyone/operators` | `__foo__` |
  | `three` | `__three__` |

  Bundling `three` would create a second instance and break shared state with looker-3d. Only `@miris-inc/three` is bundled.

## Troubleshooting

### "No viewer key resolved" warning in the console

The sample's fo3d node has no `viewerKey` and the dataset has no `miris_viewer_key` in `info`. Re-run **Sync Miris Assets** with a viewer key.

### Miris stream is black or empty

- Check you opened the app over HTTPS — Miris WASM refuses to run in an insecure context.
- Look for `[MirisStream] Failed to construct stream` in the browser console — usually an invalid viewer key or unknown asset UUID.
- Confirm the viewer key has access to the asset at [app.miris.com](https://app.miris.com).

### Label generation task stays in QUEUED state

The delegated worker is not running. Start it in a separate terminal:

```powershell
fiftyone delegated launch
```

### "build_depth_cloud: no points produced (N frames, N skipped)"

All captured frames failed the depth-loading step. Common causes:

- **Depth files not found** — the `save_capture_batch` step may have failed silently. Check the FiftyOne console for batch-save warnings during capture.
- **Alpha channel missing** — color images must be RGBA (4-channel PNG). If they were saved as JPEG or 3-channel PNG, the alpha-mask step produces no foreground pixels.
- **Depth decode error** — each frame prints `[depth] frame N: <error>` to the worker console when `load_depth_map` throws. Check the worker terminal output.

### "No detections found" after a successful depth cloud

Grounding DINO found no objects matching the text prompt. Try:

- Rephrasing the prompt — DINO is sensitive to wording. Use noun phrases separated by `.`, e.g. `Car. Traffic light.` rather than `cars and traffic lights`.
- Lowering `box_threshold` / `text_threshold` in the pipeline (currently hardcoded; requires code change).
- Verifying the segmentation frames contain the objects — check `~/fiftyone/<dataset>/captures/<run>/segmentation/` for debug outputs if `visualize=True` is set.

### Grid shows blank tiles

Thumbnails are populated only after a successful sync. If thumbnails 404, check the operator output for `"action": "skipped", "reason": "thumbnail download failed"`.

## Requirements

| Component | Version |
|---|---|
| FiftyOne | ≥ 1.14 (with built-in `MirisStream` support) |
| Node.js | ≥ 18 |
| `@miris-inc/three` | latest (bundled) |
| Browser | Chrome 90+, Firefox 88+, Safari 14+ |
| HTTPS | Required for Miris streaming |
| GPU (optional) | NVIDIA CUDA recommended for DINO+SAM2 performance |

## License

Apache-2.0
