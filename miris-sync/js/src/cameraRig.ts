import { Camera, PerspectiveCamera, Vector3 } from "three";

export interface RigCamera {
  name: string;
  direction: Vector3;
}

export const RIG_CAMERAS: RigCamera[] = [
  { name: "front",           direction: new Vector3( 0,  0,  1) },
  { name: "back",            direction: new Vector3( 0,  0, -1) },
  { name: "right",           direction: new Vector3( 1,  0,  0) },
  { name: "left",            direction: new Vector3(-1,  0,  0) },
  { name: "up",              direction: new Vector3( 0,  1,  0) },
  { name: "down",            direction: new Vector3( 0, -1,  0) },
  { name: "front_right_top", direction: new Vector3( 1,  1,  1) },
  { name: "front_left_top",  direction: new Vector3(-1,  1,  1) },
  { name: "back_right_top",  direction: new Vector3( 1,  1, -1) },
  { name: "back_left_top",   direction: new Vector3(-1,  1, -1) },
];

/**
 * How many capture stops to plant per polyline segment.
 *
 * Total stops along a path of N waypoints = (N - 1) * STOPS_PER_SEGMENT.
 * Stops are distributed uniformly by arc length across the full polyline
 * (longer segments naturally get proportionally more stops). Each stop fires
 * the full {@link RIG_CAMERAS} rig, so total images per run is
 *   (N - 1) * STOPS_PER_SEGMENT * RIG_CAMERAS.length.
 *
 * Adjust here and rebuild the JS bundle to change rig density without
 * touching the operators.
 */
export const STOPS_PER_SEGMENT = 5;

/**
 * Number of capture stops along a polyline of `waypoints`.
 * Returns 1 for 0/1 waypoints (degenerate single-camera fallback).
 */
export function totalStopsForPath(
  waypoints: [number, number, number][] | undefined,
): number {
  if (!waypoints || waypoints.length < 2) return 1;
  return (waypoints.length - 1) * STOPS_PER_SEGMENT;
}

/**
 * Returns the position along `waypoints` at normalized time `t ∈ [0, 1]`
 * using arc-length (constant-speed) parameterization.
 */
export function interpolatePath(
  waypoints: [number, number, number][],
  t: number,
): Vector3 {
  if (waypoints.length === 0) return new Vector3();
  if (waypoints.length === 1) return new Vector3(...waypoints[0]);

  const points = waypoints.map(([x, y, z]) => new Vector3(x, y, z));
  const lengths: number[] = [0];
  for (let i = 1; i < points.length; i++) {
    lengths.push(lengths[i - 1] + points[i].distanceTo(points[i - 1]));
  }
  const total = lengths[lengths.length - 1];
  if (total === 0) return points[0].clone();

  const targetLen = Math.max(0, Math.min(1, t)) * total;
  for (let i = 1; i < points.length; i++) {
    if (lengths[i] >= targetLen || i === points.length - 1) {
      const segLen = lengths[i] - lengths[i - 1];
      const segT = segLen === 0 ? 0 : (targetLen - lengths[i - 1]) / segLen;
      return points[i - 1].clone().lerp(points[i], segT);
    }
  }
  return points[points.length - 1].clone();
}

/**
 * Creates a PerspectiveCamera at `position` looking in `direction`,
 * copying FOV/aspect/near/far from `ref`.
 */
export function createRigCamera(
  position: Vector3,
  direction: Vector3,
  ref: Camera,
): PerspectiveCamera {
  const refCam = ref as PerspectiveCamera;
  const cam = new PerspectiveCamera(
    refCam.fov ?? 50,
    refCam.aspect ?? 1,
    refCam.near ?? 0.1,
    refCam.far ?? 2000,
  );
  cam.position.copy(position);
  const dir = direction.clone().normalize();
  // Avoid degenerate up vector when looking straight up or down
  const isVertical = Math.abs(dir.y) > 0.99;
  cam.up.set(0, isVertical ? 0 : 1, isVertical ? 1 : 0);
  cam.lookAt(position.clone().add(dir));
  cam.updateMatrixWorld(true);
  return cam;
}
