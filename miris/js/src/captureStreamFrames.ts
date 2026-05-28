import { executeOperator } from "@fiftyone/operators";
import {
  Box3, Camera, FloatType, Frustum, Matrix4, PerspectiveCamera, Scene, Vector3,
  WebGLRenderer, WebGLRenderTarget,
} from "three";
import { RIG_CAMERAS, createRigCamera, interpolatePath } from "./cameraRig";
import { executeOperatorAndReturn } from "./utils";

const PLUGIN_NAME = "@miris-inc/voxel51";
const SAVE_BATCH_OP = `${PLUGIN_NAME}/save_capture_batch`;

// Factors mirror the SDK's internal _setDepthLimits.
const DEPTH_NEAR_FACTOR = 0.1;
const DEPTH_FAR_FACTOR = 10;
const DEPTH_ENCODING = "single_r";

/**
 * rig            — Waypoints + 10-camera rig. Scene camera gazes at the first
 *                  POI (or bounds center if no POIs) for octree refinement only;
 *                  rig cameras handle capture.
 * nearest_poi    — Waypoints + POIs. Single camera for both capture and
 *                  refinement; gazes at the spatially closest POI each slot.
 * coverage_greedy— Waypoints + POIs. Single camera; gazes at the POI with the
 *                  least angular-coverage accumulated so far.
 */
export enum CaptureMode {
  NearestPoi    = "nearest_poi",
  CoverageGreedy = "coverage_greedy",
  Rig           = "rig",
}

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
  captureMode?: CaptureMode;
  pointsOfInterest?: [number, number, number][];
}

export function sanitizeName(name: string): string {
  return name.replace(/[^a-zA-Z0-9_-]/g, "_");
}

export function buildRunFolder(name: string, uuid: string, timestamp: number): string {
  return `${name}_${uuid}_${timestamp}`;
}

// ── POI selection helpers ─────────────────────────────────────────────────────

function nearestPoiIndex(position: Vector3, pois: Vector3[]): number {
  let bestIdx = 0;
  let bestDist = position.distanceTo(pois[0]!);
  for (let i = 1; i < pois.length; i++) {
    const d = position.distanceTo(pois[i]!);
    if (d < bestDist) { bestDist = d; bestIdx = i; }
  }
  return bestIdx;
}

function angularBucket(fromPos: Vector3, toPoi: Vector3): string {
  const dir = toPoi.clone().sub(fromPos).normalize();
  const az = Math.atan2(dir.z, dir.x); // -π … π
  const azBucket = Math.floor(((az / Math.PI + 1) * 4) % 8);
  const elBucket = dir.y > 0.33 ? "u" : dir.y < -0.33 ? "d" : "m";
  return `${azBucket}${elBucket}`;
}

function selectCoverageGreedyPoi(
  position: Vector3,
  pois: Vector3[],
  coverage: Map<number, Set<string>>,
): number {
  let bestIdx = 0;
  let bestScore = Infinity;
  for (let i = 0; i < pois.length; i++) {
    const seen = coverage.get(i) ?? new Set<string>();
    const bucket = angularBucket(position, pois[i]!);
    // Prefer the POI that adds a new viewing angle; tie-break by total coverage count.
    const score = seen.has(bucket) ? seen.size + pois.length : seen.size;
    if (score < bestScore) { bestScore = score; bestIdx = i; }
  }
  return bestIdx;
}

function updatePoiCoverage(
  coverage: Map<number, Set<string>>,
  poiIdx: number,
  fromPos: Vector3,
  pois: Vector3[],
): void {
  if (!coverage.has(poiIdx)) coverage.set(poiIdx, new Set());
  coverage.get(poiIdx)!.add(angularBucket(fromPos, pois[poiIdx]!));
}

// ── capture loop ─────────────────────────────────────────────────────────────

export async function runCapture(cfg: CaptureConfig): Promise<FrameData[]> {
  const {
    gl, camera, scene, stream,
    captureDuration, fps, preCaptureDelayMs,
    assetUuid, assetName, timestamp, folderName,
    pathWaypoints,
  } = cfg;
  const captureMode = cfg.captureMode ?? CaptureMode.Rig;
  const pois = cfg.pointsOfInterest ?? [];

  const intervalMs        = 1000 / fps;
  const captureDurationMs = captureDuration * 1000;

  const glCtx = gl.getContext();
  const w = glCtx.drawingBufferWidth;
  const h = glCtx.drawingBufferHeight;

  const target      = new WebGLRenderTarget(w, h, { depthBuffer: false, stencilBuffer: false });
  const targetDepth = new WebGLRenderTarget(w, h, { type: FloatType, depthBuffer: false, stencilBuffer: false });
  const readbackFloat = new Float32Array(w * h * 4); // reused across depth renders
  const encCanvas = document.createElement("canvas");
  encCanvas.width   = w;
  encCanvas.height  = h;
  const enc2d = encCanvas.getContext("2d")!;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const bounds = (stream as any).getBounds() as { size: number[]; center: number[] } | null;
  const streamCenter = bounds
    ? new Vector3(bounds.center[0] ?? 0, bounds.center[1] ?? 0, bounds.center[2] ?? 0)
    : new Vector3();

  // Depth encoding range derived from stream bounds — tuned to the actual scene
  // depth range rather than the full near-far clip span.
  const bound    = Math.max(...(bounds?.size ?? [1, 1, 1]), 1);
  const depthMin = Math.max(0.01, bound * DEPTH_NEAR_FACTOR);
  const depthMax = Math.max(depthMin + 1e-3, bound * DEPTH_FAR_FACTOR);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const modelRoot = (stream as any).children[0];
  const prevMode: string = modelRoot?._renderMode ?? "Splats";

  const poiVectors = pois.map(([x, y, z]) => new Vector3(x, y, z));
  const useRig = captureMode === CaptureMode.Rig;
  // Rig mode: gaze the octree-refinement camera at the first POI if provided,
  // else the scene bounds center.
  const gazeTarget = useRig && poiVectors.length > 0 ? poiVectors[0]! : streamCenter;

  // Coverage map for CoverageGreedy: poiIndex → set of angular-bucket strings seen.
  const coverage = new Map<number, Set<string>>();

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
  camera.lookAt(
    !useRig && poiVectors.length > 0
      ? poiVectors[nearestPoiIndex(camera.position, poiVectors)]!
      : gazeTarget,
  );
  camera.updateMatrixWorld(true);
  gl.render(scene, camera);

  if (preCaptureDelayMs > 0) await sleep(preCaptureDelayMs);

  // ── rAF loop ──────────────────────────────────────────────────────────────
  // Moves the offscreen camera continuously along the path for the full capture
  // duration, starting only after the pre-capture delay. This drives MirisStream
  // progressive refinement between captures.
  // For nearest_poi / coverage_greedy the scene camera is also the capture
  // camera, so the rAF keeps the stream refining toward the nearest POI.
  let rafHandle = -1;
  let rafStartTime: number | null = null;

  if (hasPath) {
    const tick = (now: number) => {
      if (rafStartTime === null) rafStartTime = now;
      const t   = Math.min(1, (now - rafStartTime) / captureDurationMs);
      const pos = interpolatePath(pathWaypoints!, t);
      camera.position.copy(pos);
      camera.lookAt(
        !useRig && poiVectors.length > 0
          ? poiVectors[nearestPoiIndex(camera.position, poiVectors)]!
          : gazeTarget,
      );
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
  const camLabel     = useRig ? `, ${RIG_CAMERAS.length} cameras` : ", 1 camera";
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

      const readback    = new Uint8Array(w * h * 4);
      const pngItems:  { filename: string; png_base64:  string }[] = [];
      const npyItems:  { filename: string; data_base64: string }[] = [];
      const jsonItems: { filename: string; json_str:    string }[] = [];
      const stepFrameData: FrameData[] = [];
      const frameStr   = String(captureCount + 1).padStart(4, "0");
      const prevTarget = gl.getRenderTarget();

      if (useRig) {
        // ── CaptureMode.Rig: 10-camera rig ───────────────────────────────
        const position = camera.position.clone();
        const entries = RIG_CAMERAS.map(rig => ({
          camName:   rig.name,
          renderCam: createRigCamera(position, rig.direction, camera),
        }));
        const visibleEntries = entries.filter(({ renderCam }) =>
          isCameraVisible(renderCam as PerspectiveCamera, bounds),
        );

        if (visibleEntries.length > 0) {
          type CapturedEntry = { camName: string | undefined; renderCam: PerspectiveCamera; colorPng: string };
          const captured: CapturedEntry[] = [];

          // Pass 1 — color
          for (const { camName, renderCam } of visibleEntries) {
            gl.setRenderTarget(target);
            gl.setClearColor(0x000000, 0);
            gl.clear(true, true, false);
            gl.render(scene, renderCam);
            gl.readRenderTargetPixels(target, 0, 0, w, h, readback);
            if (isEmpty(readback)) { skippedEmpty++; continue; }
            captured.push({ camName, renderCam: renderCam as PerspectiveCamera, colorPng: encodePng(enc2d, encCanvas, readback, w, h) });
          }

          // Pass 2 — depth (float32 NPY)
          const depthNpys: string[] = new Array(captured.length);
          if (captured.length > 0) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const streamAny = stream as any;

            streamAny._setRenderMode("SplatDepthColor");
            if (typeof streamAny._setDepthLimits === "function") {
              streamAny._setDepthLimits(depthMin, depthMax);
            }
            try {
              for (let ci = 0; ci < captured.length; ci++) {
                const { renderCam } = captured[ci];
                gl.setRenderTarget(targetDepth);
                gl.setClearColor(0x000000, 0);
                gl.clear(true, true, false);
                gl.render(scene, renderCam);
                gl.readRenderTargetPixels(targetDepth, 0, 0, w, h, readbackFloat);
                depthNpys[ci] = encodeDepthNpy(readbackFloat, w, h);
              }
            } finally {
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              (stream as any)._setRenderMode(prevMode);
            }
          }

          for (let ci = 0; ci < captured.length; ci++) {
            const { camName, renderCam, colorPng } = captured[ci];
            const colorFilename   = buildFilename(assetName, assetUuid, timestamp, "color",  frameStr, "png",  camName);
            const depthFilename   = buildFilename(assetName, assetUuid, timestamp, "depth",  frameStr, "npy",  camName);
            const camJsonFilename = buildFilename(assetName, assetUuid, timestamp, "camera", frameStr, "json", camName);
            pngItems.push({ filename: `${folderName}/${colorFilename}`, png_base64: colorPng });
            npyItems.push({ filename: `${folderName}/${depthFilename}`, data_base64: depthNpys[ci] });
            jsonItems.push({ filename: `${folderName}/${camJsonFilename}`, json_str: JSON.stringify(buildCameraJson(
              renderCam, w, h, assetName, assetUuid, timestamp, frameStr, camName,
              { depth_filename: depthFilename, depth_min: depthMin, depth_max: depthMax, depth_encoding: DEPTH_ENCODING },
            ))});
            stepFrameData.push({
              frame: captureCount,
              ...(camName !== undefined ? { camera_name: camName } : {}),
              color_filename:       `${folderName}/${colorFilename}`,
              depth_filename:       `${folderName}/${depthFilename}`,
              camera_json_filename: `${folderName}/${camJsonFilename}`,
            });
          }
        }
      } else {
        // ── CaptureMode.NearestPoi / CoverageGreedy: single scene camera ──
        const poiIdx = poiVectors.length === 0 ? -1
          : captureMode === CaptureMode.NearestPoi
            ? nearestPoiIndex(camera.position, poiVectors)
            : selectCoverageGreedyPoi(camera.position, poiVectors, coverage);

        camera.lookAt(poiIdx >= 0 ? poiVectors[poiIdx]! : streamCenter);
        camera.updateMatrixWorld(true);

        if (isCameraVisible(camera as PerspectiveCamera, bounds)) {
          // Pass 1 — color
          gl.setRenderTarget(target);
          gl.setClearColor(0x000000, 0);
          gl.clear(true, true, false);
          gl.render(scene, camera);
          gl.readRenderTargetPixels(target, 0, 0, w, h, readback);

          if (isEmpty(readback)) {
            skippedEmpty++;
          } else {
            const colorPng = encodePng(enc2d, encCanvas, readback, w, h);

            // Pass 2 — depth (float32 NPY)
            let depthNpy = "";
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const streamAny = stream as any;
            streamAny._setRenderMode("SplatDepthColor");
            if (typeof streamAny._setDepthLimits === "function") {
              streamAny._setDepthLimits(depthMin, depthMax);
            }
            try {
              gl.setRenderTarget(targetDepth);
              gl.setClearColor(0x000000, 0);
              gl.clear(true, true, false);
              gl.render(scene, camera);
              gl.readRenderTargetPixels(targetDepth, 0, 0, w, h, readbackFloat);
              depthNpy = encodeDepthNpy(readbackFloat, w, h);
            } finally {
              streamAny._setRenderMode(prevMode);
            }

            const colorFilename   = buildFilename(assetName, assetUuid, timestamp, "color",  frameStr, "png");
            const depthFilename   = buildFilename(assetName, assetUuid, timestamp, "depth",  frameStr, "npy");
            const camJsonFilename = buildFilename(assetName, assetUuid, timestamp, "camera", frameStr, "json");
            pngItems.push({ filename: `${folderName}/${colorFilename}`, png_base64: colorPng });
            npyItems.push({ filename: `${folderName}/${depthFilename}`, data_base64: depthNpy });
            jsonItems.push({ filename: `${folderName}/${camJsonFilename}`, json_str: JSON.stringify(buildCameraJson(
              camera as PerspectiveCamera, w, h, assetName, assetUuid, timestamp, frameStr, undefined,
              { depth_filename: depthFilename, depth_min: depthMin, depth_max: depthMax, depth_encoding: DEPTH_ENCODING },
            ))});
            stepFrameData.push({
              frame: captureCount,
              color_filename:       `${folderName}/${colorFilename}`,
              depth_filename:       `${folderName}/${depthFilename}`,
              camera_json_filename: `${folderName}/${camJsonFilename}`,
            });

            if (captureMode === CaptureMode.CoverageGreedy && poiIdx >= 0) {
              updatePoiCoverage(coverage, poiIdx, camera.position, poiVectors);
            }
          }
        }
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

interface DepthMeta {
  depth_filename: string;
  depth_min: number;
  depth_max: number;
  depth_encoding: string;
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
  depthMeta?: DepthMeta,
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

function encodeDepthNpy(readbackFloat: Float32Array, w: number, h: number): string {
  // Extract R channel from RGBA float32 readback (WebGL writes depth to all channels).
  const depth = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) depth[i] = readbackFloat[i * 4]!;

  // Flip rows: WebGL origin is bottom-left; NumPy convention is top-left.
  const flipped = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    flipped.set(depth.subarray(y * w, (y + 1) * w), (h - 1 - y) * w);
  }

  // Build a NumPy v1.0 file: magic(6) + version(2) + header_len(2) + header + data.
  // Total of (10 + header_len) must be a multiple of 64.
  const header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${h}, ${w}), }`;
  const prefixLen = 10;
  const totalPrefix = Math.ceil((prefixLen + header.length + 1) / 64) * 64;
  const headerLen = totalPrefix - prefixLen;
  const headerPadded = header.padEnd(headerLen - 1, " ") + "\n";

  const out = new Uint8Array(totalPrefix + flipped.byteLength);
  out[0]=0x93; out[1]=0x4e; out[2]=0x55; out[3]=0x4d; out[4]=0x50; out[5]=0x59; // \x93NUMPY
  out[6]=0x01; out[7]=0x00;                                                       // version 1.0
  out[8]=headerLen & 0xff; out[9]=(headerLen >> 8) & 0xff;                        // header_len LE
  for (let i = 0; i < headerPadded.length; i++) out[10 + i] = headerPadded.charCodeAt(i);
  out.set(new Uint8Array(flipped.buffer), totalPrefix);

  // Base64-encode in 8 KB chunks to avoid call-stack overflow on large buffers.
  const chunkSize = 8192;
  const parts: string[] = [];
  for (let i = 0; i < out.length; i += chunkSize) {
    // eslint-disable-next-line prefer-spread
    parts.push(String.fromCharCode.apply(null, out.subarray(i, i + chunkSize) as unknown as number[]));
  }
  return btoa(parts.join(""));
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
