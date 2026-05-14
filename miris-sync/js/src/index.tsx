import { registerOperator } from "@fiftyone/operators";
import { Miris } from "@miris-inc/three";
import { initMirisScene } from "./mirisScene";
import { SyncMirisAssets, DEFAULT_VIEWER_KEY } from "./syncMirisAssets";

// Rendering of `MirisStream` fo3d nodes is provided by FiftyOne core's
// looker-3d package. This plugin only ingests Miris assets into the
// current dataset via the sync operator below.
initMirisScene(Miris.instance(), DEFAULT_VIEWER_KEY);
registerOperator(SyncMirisAssets, "@miris-inc/voxel51");
