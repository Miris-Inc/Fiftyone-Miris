import { Miris, MirisStream } from "@miris-inc/three";
import { PerspectiveCamera, Scene, WebGLRenderer } from "three";

/**
 * Build a self-contained {gl, scene, camera, stream} for a Miris asset.
 *
 * The plugin can't reliably borrow FiftyOne looker-3d's R3F state (R3F's
 * registry is module-internal and `@react-three/fiber` is not externalized as
 * a runtime global by this FiftyOne fork). So we render to our own detached
 * canvas using plain Three.js — the Miris SDK only needs `{uuid, viewerKey}`
 * and any THREE.Scene to render into.
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type MirisStreamLike = MirisStream & { _setRenderMode: (mode: string) => void };

export interface OffscreenScene {
  canvas: HTMLCanvasElement;
  gl: WebGLRenderer;
  scene: Scene;
  stream: MirisStreamLike;
  defaultCamera: PerspectiveCamera;
  dispose: () => void;
}

export interface OffscreenSceneOptions {
  assetUuid: string;
  viewerKey: string;
  /** Output frame width in pixels. Default 1920. */
  width?: number;
  /** Output frame height in pixels. Default 1080. */
  height?: number;
  /** Vertical field of view in degrees for the default camera. Default 50. */
  fov?: number;
  /** Maximum time to wait for the stream's `streamloaded` event. Default 30s. */
  streamLoadTimeoutMs?: number;
}

export async function setupOffscreenScene(
  opts: OffscreenSceneOptions,
): Promise<OffscreenScene> {
  const w = opts.width ?? 1920;
  const h = opts.height ?? 1080;

  // Detached canvas — never attached to the document.
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;

  const gl = new WebGLRenderer({ canvas, alpha: true, antialias: false });
  gl.setSize(w, h, false);
  gl.setClearColor(0x000000, 0);

  // The Miris SDK runtime is a process-wide singleton; safe to await again.
  await Miris.instance();

  const stream = new MirisStream({
    uuid: opts.assetUuid,
    viewerKey: opts.viewerKey,
  }) as MirisStreamLike;

  const scene = new Scene();
  scene.add(stream);

  // Wait for the manifest + bounds to be ready before capture starts.
  // The stream's splat payload may keep loading after this fires — that's
  // tolerable for our capture loop, which proceeds frame-by-frame anyway.
  await new Promise<void>((resolve, reject) => {
    const timeoutMs = opts.streamLoadTimeoutMs ?? 30_000;
    const timer = setTimeout(
      () => reject(new Error(`MirisStream did not load within ${timeoutMs}ms`)),
      timeoutMs,
    );
    stream.addEventListener("streamloaded", () => {
      clearTimeout(timer);
      resolve();
    });
  });

  // Default camera positioned at an orbit-radius offset from the stream's
  // bounds. This gives `runCapture` something sensible when the caller does
  // not supply explicit path waypoints.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const bounds = (stream as any).getBounds() as { size: number[]; center: number[] };
  const [sx = 0, sy = 0, sz = 0] = bounds?.size ?? [];
  const [cx = 0, cy = 0, cz = 0] = bounds?.center ?? [];
  const radius = Math.max(sx, sy, sz) * 1.5 || 5;
  const fov = opts.fov ?? 50;
  const camera = new PerspectiveCamera(fov, w / h, 0.1, Math.max(2000, radius * 100));
  camera.position.set(cx + radius, cy + radius * 0.5, cz + radius);
  camera.lookAt(cx, cy, cz);
  camera.updateMatrixWorld(true);

  const dispose = () => {
    try { scene.remove(stream); } catch { /* ignore */ }
    // The Miris SDK doesn't expose a public dispose; best-effort.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (stream as any).dispose?.();
    gl.dispose();
  };

  return { canvas, gl, scene, stream, defaultCamera: camera, dispose };
}
