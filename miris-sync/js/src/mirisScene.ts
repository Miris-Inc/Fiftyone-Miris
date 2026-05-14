import { MirisScene, Miris } from "@miris-inc/three";

export type { MirisScene };

let _scenePromise: Promise<MirisScene> | null = null;

export function initMirisScene(
  mirisPromise: Promise<Miris>,
  viewerKey?: string | null,
): void {
  _scenePromise = mirisPromise.then(
    (miris) => new MirisScene({ miris, viewerKey }),
  );
}

export async function getMirisScene(): Promise<MirisScene> {
  if (!_scenePromise)
    throw new Error("MirisScene not initialized. Call initMirisScene first.");
  return _scenePromise;
}
