import { Operator, OperatorConfig, ExecutionContext, executeOperator, types } from "@fiftyone/operators";
import { useRecoilValue } from "recoil";
import * as fos from "@fiftyone/state";
import { RIG_CAMERAS, interpolatePath, totalStopsForPath } from "./cameraRig";
import { setupOffscreenScene } from "./offscreenScene";
import { DEFAULT_VIEWER_KEY } from "./syncMirisAssets";
import { executeOperatorAndReturn } from "./utils";
import { readWaypointsFromSample } from "./waypoints";

const PLUGIN_NAME = "@miris-inc/voxel51";

interface PreviewDetection {
  label: string;
  location: [number, number, number];
  dimensions: [number, number, number];
  rotation: [number, number, number];
}

export class PreviewCameraPath extends Operator {
  get config(): OperatorConfig {
    return new OperatorConfig({
      name: "preview_camera_path",
      label: "Miris: Preview camera path",
    });
  }

  useHooks(): Record<string, unknown> {
    return {
      modalSample: useRecoilValue<fos.ModalSample>(fos.modalSample),
      datasetInfo: useRecoilValue<{ info?: Record<string, unknown> } | null>(
        (fos as unknown as { dataset: unknown }).dataset as never,
      ),
    };
  }

  resolveInput(_ctx: ExecutionContext): types.Property {
    const inputs = new types.Object();
    inputs.str("path_waypoints", {
      label: "Camera Path Waypoints (JSON)",
      description: 'Sequence of 3D points [[x,y,z],...]. Leave empty to re-use the wp* cuboids already on the sample.',
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
    const datasetInfo = (ctx.hooks.datasetInfo as { info?: Record<string, unknown> } | null)?.info;
    const viewerKey =
      (datasetInfo?.miris_viewer_key as string | undefined)
      ?? DEFAULT_VIEWER_KEY;

    // Resolve waypoints: explicit JSON wins; otherwise read existing wp* cuboids.
    let waypoints: [number, number, number][] = [];
    const pathRaw = ((ctx.params.path_waypoints as string | undefined) ?? "").trim();
    if (pathRaw) {
      try {
        waypoints = JSON.parse(pathRaw) as [number, number, number][];
      } catch {
        executeOperator("@voxel51/operators/notify", {
          message: "path_waypoints is not valid JSON.",
          variant: "error",
        });
        return;
      }
    } else {
      waypoints = readWaypointsFromSample(sample) ?? [];
    }
    if (waypoints.length < 1) {
      executeOperator("@voxel51/operators/notify", {
        message: "No waypoints provided and no editable wp* cuboids on the sample yet — paste JSON in the field.",
        variant: "error",
      });
      return;
    }

    runPreview({
      assetUuid,
      viewerKey,
      waypoints,
      totalFrames: totalStopsForPath(waypoints),
    }).catch((err: unknown) => {
      console.error("[PreviewCameraPath]", err);
      executeOperator("@voxel51/operators/notify", {
        message: `Preview failed: ${(err as Error).message}`,
        variant: "error",
      });
    });
  }
}

interface PreviewArgs {
  assetUuid: string;
  viewerKey: string;
  waypoints: [number, number, number][];
  totalFrames: number;
}

async function runPreview(args: PreviewArgs): Promise<void> {
  await Promise.resolve();

  console.log("[PreviewCameraPath] waypoints:", args.waypoints);
  await executeOperator("@voxel51/operators/notify", {
    message: "Loading Miris stream to compute preview…",
    variant: "info",
  });

  const off = await setupOffscreenScene({
    assetUuid: args.assetUuid,
    viewerKey: args.viewerKey,
  });

  try {
    // Size cubes to ~1% of the stream's largest bound, with a sane floor.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const bounds = (off.stream as any).getBounds() as { size: number[]; center?: number[] } | undefined;
    console.log("[PreviewCameraPath] stream bounds:", bounds);
    const maxDim = Math.max(...(bounds?.size ?? [1, 1, 1]), 1);
    const wpSize = Math.max(maxDim * 0.01, 0.05);
    const rigSize = wpSize * 0.5;
    console.log("[PreviewCameraPath] cube sizes — waypoint:", wpSize, "rig:", rigSize);

    const waypointDets: PreviewDetection[] = args.waypoints.map((wp, i) => ({
      label: `wp${i}`,
      location: [wp[0], wp[1], wp[2]],
      dimensions: [wpSize, wpSize, wpSize],
      rotation: [0, 0, 0],
    }));

    const rigDets: PreviewDetection[] = [];
    for (let f = 0; f < args.totalFrames; f++) {
      const t = args.totalFrames <= 1 ? 0 : f / (args.totalFrames - 1);
      const pos = interpolatePath(args.waypoints, t);
      for (const rig of RIG_CAMERAS) {
        rigDets.push({
          label: `rig_${String(f).padStart(4, "0")}_${rig.name}`,
          location: [pos.x, pos.y, pos.z],
          dimensions: [rigSize, rigSize, rigSize],
          rotation: [0, 0, 0],
        });
      }
    }
    console.log("[PreviewCameraPath] writing", waypointDets.length, "waypoint(s) and", rigDets.length, "rig marker(s)");

    const result = await executeOperatorAndReturn(`${PLUGIN_NAME}/save_camera_path_preview`, {
      asset_uuid: args.assetUuid,
      waypoints: waypointDets,
      rig: rigDets,
    });
    console.log("[PreviewCameraPath] save result:", result);
    if (result.status === "error") {
      throw new Error(String(result.error ?? "save_camera_path_preview returned error"));
    }

    // reload_samples refreshes the modal viewer; reload_dataset is heavier and
    // sometimes leaves the modal's cached sample stale.
    await executeOperator("@voxel51/operators/reload_samples", {});
    await executeOperator("@voxel51/operators/notify", {
      message: `Preview written — ${waypointDets.length} waypoint(s), ${rigDets.length} rig marker(s). If you don't see cubes: toggle camera_waypoints / camera_rig_preview ON in the left sidebar. Then enter annotate mode to drag wp* cuboids.`,
      variant: "success",
    });
  } finally {
    off.dispose();
  }
}
