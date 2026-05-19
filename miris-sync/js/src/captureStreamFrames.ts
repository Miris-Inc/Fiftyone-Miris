import { executeOperator } from "@fiftyone/operators";
import { Camera, PerspectiveCamera, Scene, WebGLRenderer, WebGLRenderTarget } from "three";
import { RIG_CAMERAS, createRigCamera, interpolatePath } from "./cameraRig";
import { executeOperatorAndReturn } from "./utils";

const PLUGIN_NAME = "@miris-inc/voxel51";
const SAVE_BATCH_OP = `${PLUGIN_NAME}/save_capture_batch`;

// After moving to a new camera, the splat renderer needs time to re-sort
// splats by depth and to stream in chunks that just came into view. If we
// capture immediately, frames can show splats in the wrong z-order or with
// missing chunks. We settle for SETTLE_MS total per camera, calling
// gl.render() at SETTLE_INTERMEDIATE_RENDERS evenly-spaced intervals so the
// renderer keeps ticking and progressively refines.
//
// Cost: ~SETTLE_MS per camera. With the default 10-rig × 5-stop-per-segment
// path and 3 waypoints that's 100 cameras × 1s ≈ 100s extra per run. Tune
// down (e.g. 250ms) if your scene is small/static; tune up if you still see
// sort artifacts.
const SETTLE_MS = 1000;
const SETTLE_INTERMEDIATE_RENDERS = 4;

export interface FrameData {
  frame: number;
  camera_name?: string;
  color_filename: string;
  camera_json_filename: string;
}

export interface CaptureConfig {
  gl: WebGLRenderer;
  camera: Camera;
  scene: Scene;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  stream: any;
  totalFrames: number;
  fps: number;
  assetUuid: string;
  assetName: string;
  timestamp: number;
  folderName: string;
  pathWaypoints?: [number, number, number][];
}

export function sanitizeName(name: string): string {
  return name.replace(/[^a-zA-Z0-9_-]/g, "_");
}

export function buildRunFolder(name: string, uuid: string, timestamp: number): string {
  return `${name}_${uuid}_${timestamp}`;
}

// ── capture loop ─────────────────────────────────────────────────────────────

export async function runCapture(cfg: CaptureConfig): Promise<FrameData[]> {
  const { gl, camera, scene, stream, totalFrames, fps, assetUuid, assetName, timestamp, folderName, pathWaypoints } = cfg;
  const intervalMs = 1000 / fps;
  const glCtx = gl.getContext();
  const w = glCtx.drawingBufferWidth;
  const h = glCtx.drawingBufferHeight;

  const target = new WebGLRenderTarget(w, h, { depthBuffer: true, stencilBuffer: false });
  const encCanvas = document.createElement("canvas");
  encCanvas.width = w;
  encCanvas.height = h;
  const enc2d = encCanvas.getContext("2d")!;

  const frameDataList: FrameData[] = [];
  let skippedEmpty = 0;

  const useRig = pathWaypoints && pathWaypoints.length > 0;

  try {
    for (let frame = 0; frame < totalFrames; frame++) {
      if (frame > 0) await sleep(intervalMs);

      // Determine which cameras to render this time step
      type RenderEntry = { camName: string | undefined; renderCam: Camera };
      let entries: RenderEntry[];

      if (useRig) {
        const t = totalFrames <= 1 ? 0 : frame / (totalFrames - 1);
        const position = interpolatePath(pathWaypoints!, t);
        entries = RIG_CAMERAS.map(rig => ({
          camName: rig.name,
          renderCam: createRigCamera(position, rig.direction, camera),
        }));
      } else {
        entries = [{ camName: undefined, renderCam: camera }];
      }

      const pngItems: { filename: string; png_base64: string }[] = [];
      const jsonItems: { filename: string; json_str: string }[] = [];
      const stepFrameData: FrameData[] = [];

      const frameStr = String(frame).padStart(4, "0");
      const prevTarget = gl.getRenderTarget();

      for (const { camName, renderCam } of entries) {
        // Settle window: render once to kick off the splat depth sort and any
        // async chunk streaming triggered by the camera move, then keep the
        // renderer ticking with intermediate render() calls spread over
        // SETTLE_MS so the sort converges and chunks land before we capture.
        gl.setRenderTarget(target);
        gl.setClearColor(0x000000, 0);
        gl.clear(true, true, false);
        gl.render(scene, renderCam);
        const sliceMs = SETTLE_MS / (SETTLE_INTERMEDIATE_RENDERS + 1);
        for (let s = 0; s < SETTLE_INTERMEDIATE_RENDERS; s++) {
          await sleep(sliceMs);
          gl.render(scene, renderCam);
        }
        await sleep(sliceMs);

        // Final RGB render → readback. Single-pass; the depth-mode pass was
        // removed earlier
        // is asynchronous and contaminated colour frames with depth pixels.
        gl.setClearColor(0x000000, 0);
        gl.clear(true, true, false);
        gl.render(scene, renderCam);
        const rgba = new Uint8Array(w * h * 4);
        gl.readRenderTargetPixels(target, 0, 0, w, h, rgba);

        // Discard frames where no splat covered any pixel — those rig cameras
        // pointed into the void. visual_hull would find no silhouette; DINO
        // would just see black. Saving and segmenting them is pure waste.
        if (isEmpty(rgba)) {
          skippedEmpty++;
          continue;
        }

        const colorFilename   = buildFilename(assetName, assetUuid, timestamp, "color",  frameStr, "png",  camName);
        const camJsonFilename = buildFilename(assetName, assetUuid, timestamp, "camera", frameStr, "json", camName);

        pngItems.push({ filename: `${folderName}/${colorFilename}`,   png_base64: encodePng(enc2d, encCanvas, rgba, w, h) });
        jsonItems.push({ filename: `${folderName}/${camJsonFilename}`, json_str: JSON.stringify(buildCameraJson(renderCam, w, h, assetName, assetUuid, timestamp, frameStr, camName)) });

        stepFrameData.push({
          frame,
          ...(camName !== undefined ? { camera_name: camName } : {}),
          color_filename:       `${folderName}/${colorFilename}`,
          camera_json_filename: `${folderName}/${camJsonFilename}`,
        });
      }

      gl.setRenderTarget(prevTarget);

      if (pngItems.length > 0) {
        const batchResult = await executeOperatorAndReturn(SAVE_BATCH_OP, { png_items: pngItems, json_items: jsonItems });
        if (batchResult.status === "error") {
          await executeOperator("@voxel51/operators/notify", {
            message: `Frame ${frame}: batch save failed — ${batchResult.error as string}`,
            variant: "warning",
          });
        }
      }

      frameDataList.push(...stepFrameData);

      const pct = Math.round((frame + 1) / totalFrames * 100);
      const camLabel = useRig ? `, ${RIG_CAMERAS.length} cameras` : "";
      await executeOperator("@voxel51/operators/notify", {
        message: `Generating labels from Miris stream...capturing frames${camLabel}...${pct}%`,
        variant: "info",
      });
    }
  } finally {
    target.dispose();
  }

  if (skippedEmpty > 0) {
    console.log(`[runCapture] skipped ${skippedEmpty} empty frame(s); kept ${frameDataList.length}`);
  }

  return frameDataList;
}

/**
 * True when no pixel in the RGBA buffer has a non-zero alpha — i.e. the
 * camera saw no splat coverage at all. Early-exits as soon as it finds one
 * covered pixel.
 */
function isEmpty(rgba: Uint8Array): boolean {
  for (let i = 3; i < rgba.length; i += 4) {
    if (rgba[i] !== 0) return false;
  }
  return true;
}

// ── helpers ───────────────────────────────────────────────────────────────────

function buildFilename(
  name: string,
  uuid: string,
  timestamp: number,
  channel: string,
  frame: string,
  ext = "png",
  camName?: string,
): string {
  const camPart = camName ? `_${camName}` : "";
  return `${name}_${uuid}_${timestamp}_${frame}${camPart}_${channel}.${ext}`;
}

function buildCameraJson(
  camera: Camera,
  w: number,
  h: number,
  assetName: string,
  assetUuid: string,
  timestamp: number,
  frameStr: string,
  camName?: string,
): object {
  const cam = camera as PerspectiveCamera;
  const fovRad = (cam.fov ?? 50) * (Math.PI / 180);
  const fy = h / 2 / Math.tan(fovRad / 2);
  const fx = fy; // square pixels — Three.js vFOV drives both axes uniformly
  const cx = w / 2;
  const cy = h / 2;

  // matrixWorldInverse is the view matrix (world→camera), column-major in Three.js.
  // Transpose to row-major for export.
  const e = cam.matrixWorldInverse.elements;
  const worldToCamera = [
    [e[0], e[4], e[8],  e[12]],
    [e[1], e[5], e[9],  e[13]],
    [e[2], e[6], e[10], e[14]],
    [e[3], e[7], e[11], e[15]],
  ];

  return {
    camera_name: buildFilename(assetName, assetUuid, timestamp, "camera", frameStr, "", camName).replace(/\.$/, ""),
    width: w,
    height: h,
    camera_intrinsics: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    world_to_camera: worldToCamera,
    image_path: buildFilename(assetName, assetUuid, timestamp, "color", frameStr, "png", camName),
    depth_range: [cam.near ?? 0.1, cam.far ?? 2000],
    horizontal_aperture: null,
    source_coordinate_system: null,
    meters_per_unit: 1.0,
    frame: parseInt(frameStr, 10),
    clip_near: cam.near ?? 0.1,
    clip_far: cam.far ?? 2000,
  };
}

function encodePng(
  ctx2d: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  pixels: Uint8Array,
  w: number,
  h: number,
): string {
  ctx2d.putImageData(new ImageData(new Uint8ClampedArray(flipVertical(pixels, w, h).buffer as ArrayBuffer), w, h), 0, 0);
  return canvas.toDataURL("image/png").slice("data:image/png;base64,".length);
}

function flipVertical(pixels: Uint8Array, w: number, h: number): Uint8Array {
  const flipped = new Uint8Array(pixels.length);
  const rowBytes = w * 4;
  for (let y = 0; y < h; y++) {
    flipped.set(pixels.subarray(y * rowBytes, (y + 1) * rowBytes), (h - 1 - y) * rowBytes);
  }
  return flipped;
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}
