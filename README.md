# Miris for FiftyOne

Stream full-fidelity 3D assets inside [FiftyOne](https://voxel51.com) using [Miris Spatial Streaming](https://miris.com). Browse, navigate, and label 3D scenes progressively in the browser — no downloads, no desktop tools.

## How It Works

FiftyOne core renders Miris streams natively via the first-class `MirisStream` fo3d node type. This plugin has two jobs:

1. **Sync** — populate your dataset with one sample per Miris asset (viewer key, thumbnail, `.fo3d` scene file).
2. **Label generation** — fly virtual cameras around an asset, run Grounding DINO + SAM2 on the captured frames, and write 3D bounding box detections back to the sample.

## Repository Structure

```
.
├── README.md
└── miris/                              # The FiftyOne plugin
    ├── fiftyone.yml                    # Plugin manifest
    ├── __init__.py                     # Python: UpsertMirisAsset operator
    ├── requirements.txt                # Python deps for label generation
    ├── README.md                       # Installation, usage, and development
    ├── python/
    │   ├── save_capture.py             # SaveCaptureBatch (JS→Python file bridge)
    │   ├── save_camera_path_preview.py # SaveCameraPathPreview
    │   ├── segment_miris_stream_frames.py  # Delegated segmentation operators
    │   └── segmentation/               # DINO + SAM2 + depth-cloud pipeline
    │       ├── pipeline.py
    │       ├── depth_cloud.py
    │       ├── detection.py
    │       ├── aabb.py
    │       ├── geometry.py
    │       ├── labels.py
    │       └── visualization.py
    └── js/
        ├── vite.config.ts
        ├── tsconfig.json
        └── src/
            ├── index.tsx               # Plugin entry point + WASM boot
            ├── syncMirisAssets.ts      # SyncMirisAssets operator
            ├── generateMirisLabels.ts  # GenerateMirisLabels operator
            ├── previewCameraPath.ts    # PreviewCameraPath operator
            ├── captureStreamFrames.ts  # Offscreen capture loop
            ├── offscreenScene.ts       # Offscreen WebGL scene setup
            ├── cameraRig.ts            # Multi-camera rig definitions
            ├── waypoints.ts            # Waypoint read/write helpers
            └── utils.ts
```

## Getting Started

See **[miris/README.md](miris/README.md)** for installation, configuration, and usage.

## License

Apache-2.0
