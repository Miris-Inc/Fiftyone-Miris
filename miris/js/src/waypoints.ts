/**
 * Read waypoints from the sample's `camera_waypoints` field, sorting `wp0`,
 * `wp1`, `wp2`, … in natural numeric order. Returns `undefined` if the field
 * is missing or empty.
 *
 * The preview operator writes these cuboids; FiftyOne's annotate-mode makes
 * them draggable so users can reposition waypoints in 3D, then any operator
 * that needs the path reads them back via this helper.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function readWaypointsFromSample(sample: any): [number, number, number][] | undefined {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const detections = sample?.camera_waypoints?.detections as any[] | undefined;
  if (!Array.isArray(detections) || detections.length === 0) return undefined;

  const valid = detections.filter(
    (d) => Array.isArray(d?.location) && d.location.length === 3,
  );
  if (valid.length === 0) return undefined;

  valid.sort((a, b) =>
    String(a?.label ?? "").localeCompare(
      String(b?.label ?? ""),
      undefined,
      { numeric: true, sensitivity: "base" },
    ),
  );

  return valid.map((d) => d.location as [number, number, number]);
}
