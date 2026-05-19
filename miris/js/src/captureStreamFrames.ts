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

// Wait after switching the stream into "SplatDepthColor" mode before the
// first depth readback — the splat shader recompile + async chunk streaming
// would otherwise bleed depth pixels into colour frames. If artifacts return,
// raise this before adding intermediate render ticks.
const DEPTH_SETTLE_MS = 1000;
// Per-camera settle inside the depth pass. Chunks are already loaded from the
// preceding colour pass; only the splat sort needs to converge for the new
// camera angle, so this is much shorter than SETTLE_MS.
const DEPTH_PER_CAMERA_MS = 150;

const DEPTH_MODE = "SplatDepthColor";
const SPLAT_MODE = "Splats";

// Factors mirror the SDK's internal _setDepthLimits.
const DEPTH_NEAR_FACTOR = 0.1;
const DEPTH_FAR_FACTOR = 10;
const DEPTH_ENCODING = "single_r";

interface MirisStream {
  getBounds(): { size: number[] };
  _setRenderMode(mode: string): void;
}

function streamControls(stream: unknown) {
  const s = stream as MirisStream;
  return {
    getBounds: () => s.getBounds(),
    async withDepthMode<T>(fn: () => Promise<T>): Promise<T> {
      s._setRenderMode(DEPTH_MODE);
      try {
        return await fn();
      } finally {
        s._setRenderMode(SPLAT_MODE);
      }
    },
  };
}

export interface FrameData {
  frame: number;
  camera_name?: string;
  color_filename: string;
  camera_json_filename: string;
  depth_filename: string;
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

      // Depth limits are stashed on each camera JSON so the Python decoder
      // can reverse the encoding.
      const ctrl = streamControls(stream);
      const bounds = ctrl.getBounds();
      const bound = Math.max(...(bounds?.size ?? [0]));
      const depthMin = Math.max(0.01, bound * DEPTH_NEAR_FACTOR);
      const depthMax = Math.max(depthMin + 1e-3, bound * DEPTH_FAR_FACTOR);

      // PNGs are encoded inline so we don't hold a Uint8Array per camera
      // alive across the depth pass; the readback buffer is reused.
      type CapturedEntry = {
        camName: string | undefined;
        renderCam: Camera;
        colorPng: string;
      };
      const captured: CapturedEntry[] = [];
      const readback = new Uint8Array(w * h * 4);

      for (const { camName, renderCam } of entries) {
        // Settle: kick the splat depth sort + async chunk streaming with one
        // render, then keep the renderer ticking with intermediate render()
        // calls over SETTLE_MS so the sort converges before readback.
        renderToTarget(gl, target, scene, renderCam);
        const sliceMs = SETTLE_MS / (SETTLE_INTERMEDIATE_RENDERS + 1);
        for (let s = 0; s < SETTLE_INTERMEDIATE_RENDERS; s++) {
          await sleep(sliceMs);
          gl.render(scene, renderCam);
        }
        await sleep(sliceMs);

        renderToTarget(gl, target, scene, renderCam);
        gl.readRenderTargetPixels(target, 0, 0, w, h, readback);

        // Rig cameras that pointed into the void show zero alpha everywhere;
        // saving + segmenting them is pure waste.
        if (isEmpty(readback)) {
          skippedEmpty++;
          continue;
        }

        captured.push({ camName, renderCam, colorPng: encodePng(enc2d, encCanvas, readback, w, h) });
      }

      // Switch into depth mode ONCE for the whole frame. Per-camera toggling
      // crashed the browser (Aw Snap / GPU process fault): each
      // `_setRenderMode` walks every splat child and rewrites its material
      // (@miris-inc/three/three.js:27568, setDepthColor at :26926), so
      // hundreds of toggles exhaust GPU resources. One toggle pair per frame
      // is the safe envelope.
      const depthPngs: string[] = new Array(captured.length);
      if (captured.length > 0) {
        await ctrl.withDepthMode(async () => {
          await sleep(DEPTH_SETTLE_MS);
          for (let ci = 0; ci < captured.length; ci++) {
            const { renderCam } = captured[ci];
            renderToTarget(gl, target, scene, renderCam);
            await sleep(DEPTH_PER_CAMERA_MS);
            gl.render(scene, renderCam);

            gl.readRenderTargetPixels(target, 0, 0, w, h, readback);
            depthPngs[ci] = encodePng(enc2d, encCanvas, readback, w, h);
          }
        });
      }

      for (let ci = 0; ci < captured.length; ci++) {
        const { camName, renderCam, colorPng } = captured[ci];
        const depthPng = depthPngs[ci];
        const colorFilename   = buildFilename(assetName, assetUuid, timestamp, "color",  frameStr, "png",  camName);
        const camJsonFilename = buildFilename(assetName, assetUuid, timestamp, "camera", frameStr, "json", camName);
        const depthFilename   = buildFilename(assetName, assetUuid, timestamp, "depth",  frameStr, "png",  camName);

        pngItems.push({ filename: `${folderName}/${colorFilename}`, png_base64: colorPng });
        pngItems.push({ filename: `${folderName}/${depthFilename}`, png_base64: depthPng });
        jsonItems.push({
          filename: `${folderName}/${camJsonFilename}`,
          json_str: JSON.stringify(buildCameraJson(
            renderCam, w, h, assetName, assetUuid, timestamp, frameStr, camName,
            { depth_filename: depthFilename, depth_min: depthMin, depth_max: depthMax, depth_encoding: DEPTH_ENCODING },
          )),
        });

        stepFrameData.push({
          frame,
          ...(camName !== undefined ? { camera_name: camName } : {}),
          color_filename:       `${folderName}/${colorFilename}`,
          camera_json_filename: `${folderName}/${camJsonFilename}`,
          depth_filename:       `${folderName}/${depthFilename}`,
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

function isEmpty(rgba: Uint8Array): boolean {
  // View as uint32s and mask off the alpha byte (high byte in LE). 4× fewer
  // iterations than walking bytes.
  const u32 = new Uint32Array(rgba.buffer, rgba.byteOffset, rgba.length >>> 2);
  for (let i = 0; i < u32.length; i++) {
    if ((u32[i] & 0xff000000) !== 0) return false;
  }
  return true;
}

// ── helpers ───────────────────────────────────────────────────────────────────

function renderToTarget(gl: WebGLRenderer, target: WebGLRenderTarget, scene: Scene, cam: Camera): void {
  gl.setRenderTarget(target);
  gl.setClearColor(0x000000, 0);
  gl.clear(true, true, false);
  gl.render(scene, cam);
}

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

interface DepthMeta {
  depth_filename: string;
  depth_min: number;
  depth_max: number;
  depth_encoding: string;
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
  depthMeta?: DepthMeta,
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
    ...(depthMeta ?? {}),
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
