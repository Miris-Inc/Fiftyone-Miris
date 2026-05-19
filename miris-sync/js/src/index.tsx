import { registerOperator } from "@fiftyone/operators";
import { Miris } from "@miris-inc/three";
import { initMirisScene } from "./mirisScene";
import { SyncMirisAssets, DEFAULT_VIEWER_KEY } from "./syncMirisAssets";
import { GenerateMirisLabels } from "./generateMirisLabels";
import { PreviewCameraPath } from "./previewCameraPath";

// Pass the promise directly — initMirisScene stores it internally so
// getMirisScene() can be awaited later without a top-level await here
// (top-level await is incompatible with the UMD build format).
initMirisScene(Miris.instance(), DEFAULT_VIEWER_KEY);

registerOperator(SyncMirisAssets, "@miris-inc/voxel51");
registerOperator(PreviewCameraPath, "@miris-inc/voxel51");
registerOperator(GenerateMirisLabels, "@miris-inc/voxel51");

// Note: rendering of MirisStream fo3d nodes in the modal is owned by FiftyOne
// core (looker-3d). The GenerateMirisLabels operator does NOT reuse that
// scene — it constructs its own offscreen renderer + MirisStream so capture
// works independently of whatever is currently shown in the modal. See
// offscreenScene.ts.
