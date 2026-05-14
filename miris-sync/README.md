# Miris Sync for FiftyOne

A FiftyOne plugin that syncs your [Miris](https://miris.com) asset library into a FiftyOne dataset.

Rendering of Miris streams is built into FiftyOne core (looker-3d) as the first-class `fo.MirisStream` fo3d type — this plugin's only job is to populate datasets with samples that point at your Miris assets.

## Features

- **One-click sync** — pulls your Miris asset list and upserts one FiftyOne sample per asset.
- **One sample per asset** — `filepath` is a tiny `.fo3d` scene containing a single `MirisStream` node; thumbnails are cached locally and used for the grid.
- **Idempotent** — re-running the sync operator updates existing samples in place (matched by `miris_asset_uuid`).
- **3D label overlays** — any `Detections` field on a sample is rendered by FiftyOne's native looker-3d.

## Prerequisites

- **FiftyOne** ≥ 1.0 with built-in `MirisStream` support (`pip install fiftyone`)
- **Node.js** ≥ 18 and **Yarn** (for building the JS bundle)
- **Miris account** with a viewer key — create one at [app.miris.com](https://app.miris.com)
- **HTTPS** — Miris streaming uses WebAssembly which requires a secure context. The FiftyOne dev server runs HTTP, so a reverse proxy is needed.

## Installation

### 1. Symlink the plugin into FiftyOne

```bash
git clone <repo-url>
cd <repo>/fiftyone

PLUGINS_DIR=$(python -c "import fiftyone as fo; print(fo.config.plugins_dir)")
ln -s "$(pwd)/miris-sync" "$PLUGINS_DIR/miris-sync"
```

### 2. Build the JS bundle

```bash
cd miris-sync/js
yarn install
yarn build
```

The build produces `js/dist/index.umd.js` (~14 MB, dominated by the Miris WASM runtime). React, Three.js, and `@fiftyone/*` are externalized against FiftyOne's runtime globals — only `@miris-inc/three` is bundled.

### 3. Launch FiftyOne

```python
import fiftyone as fo

# Any dataset works — the sync operator populates it.
dataset = fo.Dataset("miris-demo", persistent=True)
fo.launch_app(dataset, port=5151)
```

Open the FiftyOne app in your browser.

## Usage

### Sync your Miris assets

1. Open the operator browser (press `` ` ``) and run **Sync Miris Assets**.
2. Optionally paste a viewer key. If left blank, the bundled demo key is used.
3. The operator fetches your asset list from Miris and, for each asset, calls the Python `upsert_miris_asset` bridge which:
   - Downloads the thumbnail to `~/fiftyone/<dataset_name>/thumbnails/<uuid>.<ext>`
   - Writes a `.fo3d` scene to `~/fiftyone/<dataset_name>/scenes/<uuid>.fo3d` containing one `MirisStream` node
   - Stores the viewer key on `dataset.info["miris_viewer_key"]` and adds a `bounding_box` `Detections` field if missing
   - Creates or updates a sample with `filepath`, `thumbnail_path`, `miris_asset_uuid`, `miris_asset_name`, and `miris_thumbnail_url`
   - Configures the dataset's `app_config` so the grid renders `thumbnail_path`

Re-running sync is safe: existing samples are matched by `miris_asset_uuid` and updated in place.

### View an asset

Click any sample in the grid. FiftyOne's modal opens the `.fo3d` scene, core's looker-3d mounts the `MirisStream` node, and the asset streams in.

### Add 3D labels

The plugin doesn't render labels — looker-3d does. Any `Detections` field on the sample is overlaid automatically:

```python
import fiftyone as fo

dataset = fo.load_dataset("miris-demo")
sample = dataset.first()
sample["bounding_box"] = fo.Detections(detections=[
    fo.Detection(
        label="robot_arm",
        location=[0, 5, 0],       # center [x, y, z]
        dimensions=[10, 20, 10],   # size [w, h, d]
        rotation=[0, 0, 0],        # radians [rx, ry, rz]
    ),
])
sample.save()
```

### Viewer key resolution

When core mounts a `MirisStream` node, it resolves the viewer key in this order:

1. `viewer_key` on the fo3d node (set by `upsert_miris_asset` if a key was passed to sync)
2. `dataset.info["miris_viewer_key"]` (also set by sync)

If neither is set, core logs a warning and skips streaming.

## Operators

| Operator | Source | Visible | Purpose |
|---|---|---|---|
| `sync_miris_assets` | JS (`syncMirisAssets.ts`) | yes | Fetches the Miris asset list and upserts a sample per asset, then triggers `reload_dataset`. |
| `upsert_miris_asset` | Python (`__init__.py`) | unlisted | Bridge invoked once per asset by `sync_miris_assets`. Caches the thumbnail, writes the `.fo3d` scene, and creates/updates the sample. |

## Architecture

```
Browser (FiftyOne App)
└── looker-3d <Canvas>                   FiftyOne's native R3F canvas (core)
    └── fo3d scene from sample.filepath
        └── MirisStream (first-class fo3d type — core renders this)
            └── new MirisStream({ uuid, viewerKey })  // streams via WASM/WebSocket

Sync flow (this plugin)
───────────────────────
JS  sync_miris_assets ──► fetch assets via MirisScene.fetchAssets()
            │
            └── for each asset: executeOperator("upsert_miris_asset", { uuid, name, thumbnail, viewer_key, tags })
Python  upsert_miris_asset
            ├── download thumbnail → ~/fiftyone/<ds>/thumbnails/
            ├── write .fo3d scene  → ~/fiftyone/<ds>/scenes/   (one fo.MirisStream node)
            ├── ensure dataset.app_config grid_media_field = "thumbnail_path"
            └── create or update sample (matched by miris_asset_uuid)
```

## Plugin Layout

```
miris-sync/
├── fiftyone.yml         # Plugin manifest (name, version, operators, js_bundle)
├── __init__.py          # Python: UpsertMirisAsset operator + register()
├── README.md            # This file
└── js/
    ├── package.json
    ├── vite.config.ts   # UMD build, classic JSX, no minify
    ├── tsconfig.json
    ├── yarn.lock
    └── src/
        ├── index.tsx              # Boots Miris WASM + registers sync operator
        ├── mirisScene.ts          # MirisScene singleton (used by fetchAssets)
        ├── syncMirisAssets.ts     # SyncMirisAssets operator + DEFAULT_VIEWER_KEY
        └── fiftyone.d.ts          # Local type stubs for @fiftyone/operators
```

## Development

### Watch mode

```bash
cd miris-sync/js
yarn dev   # vite build --watch
```

Reload the FiftyOne page after each rebuild — FiftyOne picks up the new bundle hash on reload.

### Build notes (`vite.config.ts`)

- **`minify: false`** — Vite's minifier shadows UMD factory parameter names, breaking the externalized-import lookup. The unminified bundle is correct.
- **`define: { "process.env.NODE_ENV": ... }`** — Three.js (and some deps) read `process.env`, which doesn't exist in the browser.
- **`react({ jsxRuntime: "classic" })`** — FiftyOne exposes `window.React` as a global but does not provide `react/jsx-runtime`, so the build must compile JSX to `React.createElement` calls.
- **Externalized globals**:
  | Module | Global |
  |---|---|
  | `react` | `React` |
  | `react-dom` | `ReactDOM` |
  | `@fiftyone/operators` | `__foo__` |
  | `three` | `__three__` |

  Bundling `three` would create a second instance and break shared state with looker-3d. Only `@miris-inc/three` is bundled.

## Troubleshooting

### "No viewer key resolved" warning in the console

The sample's fo3d node has no `viewerKey` and the dataset has no `miris_viewer_key` in `info`. Re-run **Sync Miris Assets** with a viewer key to populate both.

### Miris stream is black or empty

- Check you opened the app over HTTPS, not HTTP — Miris WASM refuses to run in an insecure context.
- Look for `[MirisStream] Failed to construct stream` in the console — usually means an invalid viewer key or unknown asset UUID.
- Confirm the viewer key has access to the asset at [app.miris.com](https://app.miris.com).

### "No dataset is currently loaded"

`sync_miris_assets` requires an open dataset. Load or create one (`fo.launch_app(my_dataset)`) before running the operator.

### Grid shows blank tiles

The grid renders `thumbnail_path`, which is populated only after a successful sync. If thumbnails 404, the sync operator's per-asset return payload will say `"action": "skipped", "reason": "thumbnail download failed"` — check the operator output.

## Requirements

| Component | Version |
|---|---|
| FiftyOne | ≥ 1.0 (with built-in `MirisStream` support) |
| Node.js | ≥ 18 |
| `@miris-inc/three` | latest (bundled) |
| Browser | Chrome 90+, Firefox 88+, Safari 14+ |
| HTTPS | Required for Miris streaming |

## License

Apache-2.0
