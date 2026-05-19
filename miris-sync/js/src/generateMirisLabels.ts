import { Operator, OperatorConfig, ExecutionContext, executeOperator, types } from "@fiftyone/operators";
import { useRecoilValue } from "recoil";
import * as fos from "@fiftyone/state";
import { totalStopsForPath } from "./cameraRig";
import { runCapture, sanitizeName, buildRunFolder, type CaptureConfig } from "./captureStreamFrames";
import { setupOffscreenScene } from "./offscreenScene";
import { DEFAULT_VIEWER_KEY } from "./syncMirisAssets";
import { executeOperatorAndReturn } from "./utils";
import { readWaypointsFromSample } from "./waypoints";

const PLUGIN_NAME = "@miris-inc/voxel51";

// The pipeline renders offscreen against a static stream, so wall-clock
// "FPS" is meaningless; this just minimises the leftover sleep in runCapture.
const CAPTURE_FPS = 60;

export class GenerateMirisLabels extends Operator {
  get config(): OperatorConfig {
    return new OperatorConfig({
      name: "generate_miris_labels",
      label: "Miris: Generate labels on current stream",
    });
  }

  useHooks(): Record<string, unknown> {
    return {
      modalSample: useRecoilValue<fos.ModalSample>(fos.modalSample),
      datasetName: useRecoilValue<string | null>(fos.datasetName),
      datasetInfo: useRecoilValue<Record<string, unknown> | null>(
        // dataset atom is a generic with .info; recoil typing varies across
        // FiftyOne versions, so cast through unknown.
        (fos as unknown as { dataset: unknown }).dataset as never,
      ),
    };
  }

  resolveInput(_ctx: ExecutionContext): types.Property {
    const inputs = new types.Object();
    inputs.enum("algorithm", ["visual_hull", "dino_sam2"], {
      label: "Algorithm",
      description: "visual_hull — single bounding box, fast   |   dino_sam2 — multiple labeled boxes, slow",
      default: "visual_hull",
      required: false,
    });
    inputs.str("dino_text", {
      label: "Object classes (dino_sam2 only)",
      description: "Grounding DINO text prompt, e.g. 'Phone. Laptop.'  Leave empty to use the default.",
      default: "Phone. Laptop.",
      required: false,
    });
    inputs.str("path_waypoints", {
      label: "Camera Path Waypoints (JSON)",
      description: 'Optional sequence of 3D points [[x,y,z],...] defining a camera rig path. Each segment between consecutive waypoints contributes 5 capture stops, and 10 rig cameras fire at each stop (e.g. 3 waypoints → 100 images, 5 waypoints → 200 images). Leave empty to re-use wp* cuboids on the sample, or to use a single default-framed view.',
      required: false,
    });
    return new types.Property(inputs);
  }

  execute(ctx: ExecutionContext): void {
    const sample = (ctx.hooks.modalSample as fos.ModalSample | undefined)?.sample;
    const assetUuid = sample?.miris_asset_uuid as string | undefined;
    if (!assetUuid) {
      executeOperator("@voxel51/operators/notify", {
        message: "Select a Miris sample first.",
        variant: "error",
      });
      return;
    }
    const assetName = sanitizeName((sample?.miris_asset_name as string | undefined) ?? "unknown");
    const datasetName = (ctx.hooks.datasetName as string | null) ?? ctx.dataset?.name ?? "unknown";
    const datasetInfo = (ctx.hooks.datasetInfo as { info?: Record<string, unknown> } | null)?.info;
    const viewerKey =
      (datasetInfo?.miris_viewer_key as string | undefined)
      ?? DEFAULT_VIEWER_KEY;

    const timestamp = Date.now();

    const algorithm = ((ctx.params.algorithm as string | undefined) ?? "visual_hull") || "visual_hull";
    const dinoText  = (ctx.params.dino_text  as string | undefined) ?? "";

    let pathWaypoints: [number, number, number][] | undefined;
    const pathRaw = ((ctx.params.path_waypoints as string | undefined) ?? "").trim();
    if (pathRaw) {
      try {
        pathWaypoints = JSON.parse(pathRaw) as [number, number, number][];
      } catch {
        executeOperator("@voxel51/operators/notify", {
          message: "path_waypoints is not valid JSON — falling back to sample waypoints / default view.",
          variant: "warning",
        });
      }
    }
    // If no JSON was supplied (or it was invalid), fall back to the wp* cuboids
    // the user may have positioned via PreviewCameraPath + annotate mode.
    if (!pathWaypoints || pathWaypoints.length === 0) {
      pathWaypoints = readWaypointsFromSample(sample);
    }

    const totalFrames = totalStopsForPath(pathWaypoints);

    // Fire-and-forget: execute() returns immediately so the modal closes and the
    // user can freely interact with the scene while the pipeline runs.
    runPipeline({
      assetUuid, viewerKey, assetName, datasetName,
      totalFrames, fps: CAPTURE_FPS, timestamp, algorithm, dinoText, pathWaypoints,
    }).catch((err: unknown) => {
      console.error("[GenerateMirisLabels]", err);
      executeOperator("@voxel51/operators/notify", {
        message: `Label generation failed: ${(err as Error).message}`,
        variant: "error",
      });
    });
  }
}

// ── pipeline ──────────────────────────────────────────────────────────────────

interface PipelineArgs {
  assetUuid: string;
  viewerKey: string;
  assetName: string;
  datasetName: string;
  totalFrames: number;
  fps: number;
  timestamp: number;
  algorithm: string;
  dinoText: string;
  pathWaypoints?: [number, number, number][];
}

async function runPipeline(args: PipelineArgs): Promise<void> {
  await Promise.resolve(); // let execute() return and the modal close first

  console.log("[GenerateMirisLabels] starting", {
    assetUuid: args.assetUuid,
    assetName: args.assetName,
    totalFrames: args.totalFrames,
    algorithm: args.algorithm,
    hasWaypoints: !!args.pathWaypoints?.length,
    waypointCount: args.pathWaypoints?.length ?? 0,
  });

  await executeOperator("@voxel51/operators/notify", {
    message: `Loading Miris stream ${args.assetName}…`,
    variant: "info",
  });

  console.log("[GenerateMirisLabels] setting up offscreen scene…");
  const off = await setupOffscreenScene({
    assetUuid: args.assetUuid,
    viewerKey: args.viewerKey,
  });
  console.log("[GenerateMirisLabels] offscreen scene ready, beginning capture loop");

  try {
    const folderName = buildRunFolder(args.assetName, args.assetUuid, args.timestamp);

    // Phase 1 — capture frames (runCapture emits per-frame progress notifications)
    const captureConfig: CaptureConfig = {
      gl: off.gl,
      camera: off.defaultCamera,
      scene: off.scene,
      stream: off.stream,
      totalFrames: args.totalFrames,
      fps: args.fps,
      assetUuid: args.assetUuid,
      assetName: args.assetName,
      timestamp: args.timestamp,
      folderName,
      pathWaypoints: args.pathWaypoints,
    };
    const frameData = await runCapture(captureConfig);
    console.log("[GenerateMirisLabels] capture done,", frameData.length, "frame entries; starting segmentation");

    // Phase 2 — segmentation (Python generator emits per-frame set_progress)
    await executeOperatorAndReturn(`${PLUGIN_NAME}/segment_miris_stream_frames`, {
      frames: frameData,
      asset_name: args.assetName,
      asset_uuid: args.assetUuid,
      dataset_name: args.datasetName,
      algorithm: args.algorithm,
      dino_text: args.dinoText,
    });
  } finally {
    off.dispose();
  }
}
