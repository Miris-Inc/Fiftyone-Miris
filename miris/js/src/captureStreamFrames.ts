import { executeOperator } from "@fiftyone/operators";
import {
  Box3, Camera, Frustum, Matrix4, PerspectiveCamera, Scene, Vector3,
  WebGLRenderer, WebGLRenderTarget, FloatType, RedFormat,
} from "three";
import { RIG_CAMERAS, createRigCamera, interpolatePath } from "./cameraRig";
import { executeOperatorAndReturn } from "./utils";

const PLUGIN_NAME = "@miris-inc/voxel51";
const SAVE_BATCH_OP = `${PLUGIN_NAME}/save_capture_batch`;

export interface FrameData {
  frame: number;
  camera_name?: string;
  color_filename: string;
  depth_filename: string;
  camera_json_filename: string;
}

export interface CaptureConfig {
  gl: WebGLRenderer;
  camera: Camera;
  scene: Scene;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  stream: any;
  captureDuration: number;
  fps: number;
  preCaptureDelayMs: number;
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
  const { gl, camera, scene, stream, captureDuration, fps, preCaptureDelayMs, assetUuid, assetName, timestamp, folderName, pathWaypoints } = cfg;
  const intervalMs        = 1000 / fps;
  const captureDurationMs = captureDuration * 1000;

  const glCtx = gl.getContext();
  const w = glCtx.drawingBufferWidth;
  const h = glCtx.drawingBufferHeight;

  const target      = new WebGLRenderTarget(w, h, { depthBuffer: false, stencilBuffer: false });
  const targetDepth = new WebGLRenderTarget(w, h, { type: FloatType, format: RedFormat });
  const encCanvas   = document.createElement("canvas");
  encCanvas.width   = w;
  encCanvas.height  = h;
  const enc2d = encCanvas.getContext("2d")!;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const bounds = (stream as any).getBounds() as { size: number[]; center: number[] } | null;
  const streamCenter = bounds
    ? new Vector3(bounds.center[0] ?? 0, bounds.center[1] ?? 0, bounds.center[2] ?? 0)
    : new Vector3();

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const modelRoot = (stream as any).children[0];
  const prevMode: string = modelRoot?._renderMode ?? "Splats";

  const frameDataList: FrameData[] = [];
  const capturePositions: Vector3[] = [];
  const hasPath = !!(pathWaypoints && pathWaypoints.length > 0);
  let skippedEmpty = 0;

  // Seed stream refinement at the starting position, then hold still for the
  // pre-capture delay so the stream can refine before the camera starts moving.
  if (hasPath) {
    const [sx = 0, sy = 0, sz = 0] = pathWaypoints![0];
    camera.position.set(sx, sy, sz);
  }
  camera.lookAt(streamCenter);
  camera.updateMatrixWorld(true);
  gl.render(scene, camera);

  if (preCaptureDelayMs > 0) await sleep(preCaptureDelayMs);

  // ── rAF loop ──────────────────────────────────────────────────────────────
  // Moves the offscreen camera continuously along the path for the full capture
  // duration, starting only after the pre-capture delay. This drives MirisStream
  // progressive refinement between captures.
  let rafHandle = -1;
  let rafStartTime: number | null = null;

  if (hasPath) {
    const tick = (now: number) => {
      if (rafStartTime === null) rafStartTime = now;
      const t   = Math.min(1, (now - rafStartTime) / captureDurationMs);
      const pos = interpolatePath(pathWaypoints!, t);
      camera.position.copy(pos);
      camera.lookAt(streamCenter);
      camera.updateMatrixWorld(true);
      gl.render(scene, camera);
      if (t < 1) rafHandle = requestAnimationFrame(tick);
    };
    rafHandle = requestAnimationFrame(tick);
  }

  // ── capture loop ──────────────────────────────────────────────────────────
  // Best-effort capture at `fps` Hz for `captureDuration` seconds.
  // Uses wall-clock slot scheduling: after each capture, skip forward to the
  // next future slot so that slow renders don't cause the run to exceed the
  // requested duration — they simply reduce the number of captures taken.
  const camLabel     = `, ${RIG_CAMERAS.length} cameras`;
  const captureStart = performance.now();
  let captureCount   = 0;
  let slotIndex      = 0;

  try {
    // eslint-disable-next-line no-constant-condition
    while (true) {
      // Wait until the next scheduled slot (or fire immediately if already past).
      const slotTime = captureStart + slotIndex * intervalMs;
      const waitMs   = slotTime - performance.now();
      if (waitMs > 0) await sleep(waitMs);

      // Stop if the capture window has closed.
      const elapsed = performance.now() - captureStart;
      if (elapsed >= captureDurationMs) break;

      const position = camera.position.clone();
      const entries = RIG_CAMERAS.map(rig => ({
        camName:   rig.name,
        renderCam: createRigCamera(position, rig.direction, camera),
      }));

      const visibleEntries = entries.filter(({ renderCam }) =>
        isCameraVisible(renderCam as PerspectiveCamera, bounds),
      );

      if (visibleEntries.length > 0) {
        const frameStr   = String(captureCount + 1).padStart(4, "0");
        const prevTarget = gl.getRenderTarget();

        const pngItems:  { filename: string; png_base64:  string }[] = [];
        const npyItems:  { filename: string; data_base64: string }[] = [];
        const jsonItems: { filename: string; json_str:    string }[] = [];
        const stepFrameData: FrameData[] = [];

        // Pass 1 — color: render each visible camera and isEmpty-filter.
        // Collect survivors into `captured` for the depth pass.
        type CapturedEntry = { camName: string | undefined; renderCam: PerspectiveCamera; colorPng: string };
        const captured: CapturedEntry[] = [];

        for (const { camName, renderCam } of visibleEntries) {
          gl.setRenderTarget(target);
          gl.setClearColor(0x000000, 0);
          gl.clear(true, true, false);
          gl.render(scene, renderCam);
          const rgba = new Uint8Array(w * h * 4);
          gl.readRenderTargetPixels(target, 0, 0, w, h, rgba);

          // Discard frames where no splat covered any pixel (frustum false-positive).
          if (isEmpty(rgba)) { skippedEmpty++; continue; }

          captured.push({ camName, renderCam: renderCam as PerspectiveCamera, colorPng: encodePng(enc2d, encCanvas, rgba, w, h) });
        }

        // Pass 2 — depth: switch render mode ONCE for the whole frame.
        // Per-camera _setRenderMode toggles walk every splat child and rewrite
        // its material, exhausting GPU resources with many cameras per frame.
        const depthData: Float32Array[] = new Array(captured.length);
        if (captured.length > 0) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          (stream as any)._setRenderMode("SplatDepthColor");
          try {
            for (let ci = 0; ci < captured.length; ci++) {
              const { renderCam } = captured[ci];
              gl.setRenderTarget(targetDepth);
              gl.clear(true, true, false);
              gl.render(scene, renderCam);
              const df = new Float32Array(w * h);
              gl.readRenderTargetPixels(targetDepth, 0, 0, w, h, df);
              depthData[ci] = df;
            }
          } finally {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (stream as any)._setRenderMode(prevMode);
          }
        }

        // Build batch items from the captured color + depth pairs.
        for (let ci = 0; ci < captured.length; ci++) {
          const { camName, renderCam, colorPng } = captured[ci];
          const depthFloat = depthData[ci];

          const colorFilename   = buildFilename(assetName, assetUuid, timestamp, "color",  frameStr, "png",  camName);
          const depthFilename   = buildFilename(assetName, assetUuid, timestamp, "depth",  frameStr, "npy",  camName);
          const camJsonFilename = buildFilename(assetName, assetUuid, timestamp, "camera", frameStr, "json", camName);

          pngItems.push ({ filename: `${folderName}/${colorFilename}`,   png_base64:  colorPng });
          npyItems.push ({ filename: `${folderName}/${depthFilename}`,   data_base64: encodeNpy(depthFloat, w, h) });
          jsonItems.push({ filename: `${folderName}/${camJsonFilename}`, json_str:    JSON.stringify(buildCameraJson(renderCam, w, h, assetName, assetUuid, timestamp, frameStr, camName)) });

          stepFrameData.push({
            frame: captureCount,
            ...(camName !== undefined ? { camera_name: camName } : {}),
            color_filename:       `${folderName}/${colorFilename}`,
            depth_filename:       `${folderName}/${depthFilename}`,
            camera_json_filename: `${folderName}/${camJsonFilename}`,
          });
        }

        gl.setRenderTarget(prevTarget);

        if (pngItems.length > 0) {
          const batchResult = await executeOperatorAndReturn(SAVE_BATCH_OP, { png_items: pngItems, npy_items: npyItems, json_items: jsonItems });
          if (batchResult.status === "error") {
            await executeOperator("@voxel51/operators/notify", {
              message: `Frame ${frameStr}: batch save failed — ${batchResult.error as string}`,
              variant: "warning",
            });
          }
        }

        if (stepFrameData.length > 0) {
          capturePositions.push(camera.position.clone());
          frameDataList.push(...stepFrameData);
          captureCount++;
        }
      }

      // Advance to the next future slot, skipping any that the render consumed.
      const elapsedAfter = performance.now() - captureStart;
      const nextSlot     = Math.ceil(elapsedAfter / intervalMs);
      slotIndex          = Math.max(slotIndex + 1, nextSlot);

      await executeOperator("@voxel51/operators/set_progress", {
        progress: Math.min(1, elapsedAfter / captureDurationMs),
        label: `Capturing frames${camLabel}…`,
      });
    }

    // Save actual capture positions and the interpolated path for comparison in
    // the FiftyOne 3D viewer.
    if (hasPath && capturePositions.length > 0) {
      const maxDim = Math.max(...(bounds?.size ?? [1, 1, 1]), 1);
      const wpSize  = Math.max(maxDim * 0.01, 0.05);
      const rigSize = wpSize * 0.5;

      const waypointDets = pathWaypoints!.map((wp, i) => ({
        label:      `wp${i}`,
        location:   [wp[0], wp[1], wp[2]],
        dimensions: [wpSize, wpSize, wpSize],
        rotation:   [0, 0, 0],
      }));

      const rigDets = capturePositions.map((pos, i) => ({
        label:      `cap_${String(i + 1).padStart(4, "0")}`,
        location:   [pos.x, pos.y, pos.z],
        dimensions: [rigSize, rigSize, rigSize],
        rotation:   [0, 0, 0],
      }));

      const polyline = pathWaypoints!.map(wp => [wp[0], wp[1], wp[2]]);

      await executeOperatorAndReturn(`${PLUGIN_NAME}/save_camera_path_preview`, {
        asset_uuid: assetUuid,
        waypoints:  waypointDets,
        rig:        rigDets,
        polyline,
      });
    }
  } finally {
    cancelAnimationFrame(rafHandle);
    target.dispose();
    targetDepth.dispose();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (stream as any)._setRenderMode?.(prevMode);
  }

  if (skippedEmpty > 0) {
    console.log(`[runCapture] skipped ${skippedEmpty} empty frame(s) (frustum false-positives); kept ${frameDataList.length}`);
  }

  return frameDataList;
}

// ── helpers ───────────────────────────────────────────────────────────────────

function isCameraVisible(
  cam: PerspectiveCamera,
  bounds: { size: number[]; center: number[] } | null,
): boolean {
  if (!bounds) return true;
  const frustum = new Frustum();
  frustum.setFromProjectionMatrix(
    new Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse),
  );
  const [cx = 0, cy = 0, cz = 0] = bounds.center;
  const [sx = 0, sy = 0, sz = 0] = bounds.size;
  const box = new Box3(
    new Vector3(cx - sx / 2, cy - sy / 2, cz - sz / 2),
    new Vector3(cx + sx / 2, cy + sy / 2, cz + sz / 2),
  );
  return frustum.intersectsBox(box);
}

function isEmpty(rgba: Uint8Array): boolean {
  // View as uint32s and mask off the alpha byte (high byte in LE) — 4× fewer
  // iterations than checking every 4th byte.
  const u32 = new Uint32Array(rgba.buffer, rgba.byteOffset, rgba.length >>> 2);
  for (let i = 0; i < u32.length; i++) {
    if ((u32[i]! & 0xff000000) !== 0) return false;
  }
  return true;
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

function buildCameraJson(
  cam: PerspectiveCamera,
  w: number,
  h: number,
  assetName: string,
  assetUuid: string,
  timestamp: number,
  frameStr: string,
  camName?: string,
): object {
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
    horizontal_aperture: null,
    source_coordinate_system: null,
    meters_per_unit: 1.0,
    frame: parseInt(frameStr, 10),
    clip_near: cam.near ?? 0.1,
    clip_far: cam.far ?? 2000,
    depth_min: cam.near ?? 0.1,
    depth_max: cam.far ?? 2000,
  };
}

function encodeNpy(data: Float32Array, w: number, h: number): string {
  const flipped   = flipVerticalFloat32(data, w, h);
  const header    = `{'descr': '<f4', 'fortran_order': False, 'shape': (${h}, ${w}), }`;
  const unpaddedTotal = 10 + header.length + 1;
  const paddedTotal   = Math.ceil(unpaddedTotal / 64) * 64;
  const headerPadded  = header.padEnd(header.length + (paddedTotal - unpaddedTotal), " ") + "\n";
  const headerBytes   = new TextEncoder().encode(headerPadded);
  const out  = new Uint8Array(10 + headerBytes.length + flipped.byteLength);
  const view = new DataView(out.buffer);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x01, 0x00]);
  view.setUint16(8, headerBytes.length, true);
  out.set(headerBytes, 10);
  out.set(new Uint8Array(flipped.buffer), 10 + headerBytes.length);
  let binary = "";
  for (let i = 0; i < out.length; i++) binary += String.fromCharCode(out[i]!);
  return btoa(binary);
}

function flipVerticalFloat32(data: Float32Array, w: number, h: number): Float32Array {
  const flipped = new Float32Array(data.length);
  for (let y = 0; y < h; y++) {
    flipped.set(data.subarray(y * w, (y + 1) * w), (h - 1 - y) * w);
  }
  return flipped;
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
  const flipped  = new Uint8Array(pixels.length);
  const rowBytes = w * 4;
  for (let y = 0; y < h; y++) {
    flipped.set(pixels.subarray(y * rowBytes, (y + 1) * rowBytes), (h - 1 - y) * rowBytes);
  }
  return flipped;
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}
