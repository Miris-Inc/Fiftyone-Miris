# Miris Sync for FiftyOne

Stream full-fidelity 3D assets inside [FiftyOne](https://voxel51.com) using [Miris Spatial Streaming](https://miris.com). Browse, navigate, and label 3D scenes progressively in the browser — no downloads, no desktop tools.

## How It Works

FiftyOne core renders Miris streams natively via the first-class `MirisStream` fo3d node type, so this plugin's only job is to populate your dataset with Miris assets.

The asset registry is a regular FiftyOne dataset:

- The **Sync Miris Assets** operator pulls your Miris asset list (using a viewer key) and creates one sample per asset.
- Each sample's `filepath` points at a small `.fo3d` scene file containing a single `MirisStream` node — no mesh data, just the asset UUID and viewer key.
- A cached thumbnail image is stored on the sample as `thumbnail_path` and used as the grid media field, so the grid loads quickly.
- Opening a sample mounts the `MirisStream` node, which streams the asset directly from Miris into the FiftyOne 3D viewer.

Sync is idempotent — existing samples are updated in place, and re-running the operator is the supported way to refresh your dataset as your Miris library grows.

- **Progressive streaming** via the Miris SDK
- **3D label overlays** from any FiftyOne `Detections` field on the sample (rendered by Looker3D, not the plugin)
- **Idempotent sync** — safe to run repeatedly
- Built on FiftyOne's native Looker3D viewer (React Three Fiber + Three.js)

## Repository Structure

```
.
├── README.md
└── miris-sync/          # The FiftyOne plugin
    ├── fiftyone.yml       # Plugin manifest
    ├── __init__.py        # Python: upsert_miris_asset operator
    ├── README.md          # Installation, usage, and development
    └── js/                # TypeScript source + Vite UMD build
```

## Getting Started

See **[miris-sync/README.md](miris-sync/README.md)** for installation, configuration, and usage.

## License

Apache-2.0
