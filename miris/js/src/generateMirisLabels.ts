import { Operator, OperatorConfig, ExecutionContext, executeOperator, types } from "@fiftyone/operators";
import { useRecoilValue } from "recoil";
import * as fos from "@fiftyone/state";
import { runCapture, sanitizeName, buildRunFolder, CaptureMode, type CaptureConfig } from "./captureStreamFrames";
import { setupOffscreenScene } from "./offscreenScene";
import { DEFAULT_VIEWER_KEY } from "./syncMirisAssets";
import { readWaypointsFromSample } from "./waypoints";

const PLUGIN_NAME = "@miris-inc/voxel51";


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
    inputs.str("dino_text", {
      label: "Object classes",
      description: "Grounding DINO text prompt, e.g. 'Phone. Laptop.'",
      required: true,
    });
    inputs.str("path_waypoints", {
      label: "Camera Path Waypoints (JSON)",
      description: 'Optional sequence of 3D points [[x,y,z],...] defining a camera path. Leave empty to re-use wp* cuboids on the sample, or to use a single default-framed view.',
      required: false,
    });
    inputs.str("points_of_interest", {
      label: "Points of Interest (JSON)",
      description: 'Optional list of 3D gaze targets [[x,y,z],...]. In rig mode, a single point overrides the default scene-center gaze. In nearest_poi / coverage_greedy modes the camera gazes at one of these targets each capture slot.',
      required: false,
    });
    inputs.int("capture_duration", {
      label: "Capture duration (seconds)",
      description: "Controls how densely the camera path is sampled together with Capture rate. Ignored when no waypoints are set.",
      default: 30,
      required: false,
    });
    inputs.int("capture_rate", {
      label: "Capture rate (fps)",
      description: "Captures per second of path duration (1–10).",
      default: 2,
      required: false,
    });
    inputs.enum("capture_mode", Object.values(CaptureMode), {
      label: "Capture mode",
      description: "rig = waypoints + 10-camera rig (scene camera gazes at POI or center). nearest_poi = single camera gazes at the closest POI each slot. coverage_greedy = single camera gazes at the least-covered POI each slot.",
      default: CaptureMode.CoverageGreedy,
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

    const dinoText = ctx.params.dino_text as string;

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

    let pointsOfInterest: [number, number, number][] | undefined;
    const poisRaw = ((ctx.params.points_of_interest as string | undefined) ?? "").trim();
    if (poisRaw) {
      try {
        pointsOfInterest = JSON.parse(poisRaw) as [number, number, number][];
      } catch {
        executeOperator("@voxel51/operators/notify", {
          message: "points_of_interest is not valid JSON — ignoring.",
          variant: "warning",
        });
      }
    }

    const captureDuration = Math.min(1000, Math.max(1,  ((ctx.params.capture_duration as number | undefined) ?? 30)));
    const captureRate     = Math.min(10,   Math.max(1,  ((ctx.params.capture_rate     as number | undefined) ?? 2)));
    const captureMode     = (ctx.params.capture_mode as CaptureMode | undefined) ?? CaptureMode.CoverageGreedy;

    // Fire-and-forget: execute() returns immediately so the modal closes and the
    // user can freely interact with the scene while the pipeline runs.
    runPipeline({
      assetUuid, viewerKey, assetName, datasetName,
      captureDuration, fps: captureRate, timestamp, dinoText,
      pathWaypoints, captureMode, pointsOfInterest,
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
  captureDuration: number;
  fps: number;
  timestamp: number;
  dinoText: string;
  pathWaypoints?: [number, number, number][];
  captureMode: CaptureMode;
  pointsOfInterest?: [number, number, number][];
}

async function runPipeline(args: PipelineArgs): Promise<void> {
  await Promise.resolve(); // let execute() return and the modal close first

  console.log("[GenerateMirisLabels] starting", {
    assetUuid: args.assetUuid,
    assetName: args.assetName,
    captureDuration: args.captureDuration,
    fps: args.fps,
    captureMode: args.captureMode,
    hasWaypoints: !!args.pathWaypoints?.length,
    waypointCount: args.pathWaypoints?.length ?? 0,
    poiCount: args.pointsOfInterest?.length ?? 0,
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

    // Phase 1 — capture frames (runCapture emits per-frame set_progress updates)
    const captureConfig: CaptureConfig = {
      gl: off.gl,
      camera: off.defaultCamera,
      scene: off.scene,
      stream: off.stream,
      captureDuration: args.captureDuration,
      fps: args.fps,
      preCaptureDelayMs: 1000,
      assetUuid: args.assetUuid,
      assetName: args.assetName,
      timestamp: args.timestamp,
      folderName,
      pathWaypoints: args.pathWaypoints,
      captureMode: args.captureMode,
      pointsOfInterest: args.pointsOfInterest,
    };
    const frameData = await runCapture(captureConfig);
    console.log("[GenerateMirisLabels] capture done,", frameData.length, "frame entries; starting segmentation");

    // Phase 2 — segmentation (Python generator emits per-frame set_progress)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    await (executeOperator as any)(`${PLUGIN_NAME}/segment_miris_stream_frames`, {
      frames: frameData,
      asset_name: args.assetName,
      asset_uuid: args.assetUuid,
      dataset_name: args.datasetName,
      dino_text: args.dinoText,
    }, { requestDelegation: true });
  } finally {
    off.dispose();
  }
}
