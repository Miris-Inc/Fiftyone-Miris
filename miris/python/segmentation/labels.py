"""Stage 4 — instance assignment.

Reprojection graph (same-label detections that co-observe many 3-D points share an edge)
→ Union-Find on those edges → each component is one physical instance → vote each point
into the best-covering instance. Visibility-aware via the rendered-depth occlusion check.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .geometry import (
    CameraCache,
    _UnionFind,
    _project_points_into_frame,
)


# Max disagreement between a 3-D point's expected view-space depth and the
# seg-frame's rendered depth at the projected pixel. Surface scatter
# post-voxel-average is ~1-2 cm and depth quantization is ≤2 cm at z=1 m;
# 5 cm sits comfortably above both while still catching any occluder ≥5 cm in
# front of the point.
_OCCLUSION_TOL_M = 0.05


def _frame_observations(
    pts: np.ndarray,
    cache: CameraCache,
    all_dets: list[list[dict]],
    label_maps: list[np.ndarray],
):
    """Yield ``(fi, dets, pi_array, di_array)`` per frame — vectorised point/det pairs that pass visibility + bounds + mask-coverage."""
    for fi, (P, dets, lmap, (H, W), dm) in enumerate(
        zip(cache.Ps, all_dets, label_maps, cache.img_sizes, cache.depth_maps)
    ):
        if not dets:
            continue
        pi_arr, di_arr = _project_points_into_frame(
            pts, P, lmap, dm, _OCCLUSION_TOL_M, len(dets),
        )
        if pi_arr.size:
            yield fi, dets, pi_arr, di_arr


def _build_graph_from_reprojection(
    pts: np.ndarray,
    cache: CameraCache,
    all_dets: list[list[dict]],
    label_maps: list[np.ndarray],
    det_label: dict,
) -> dict:
    """Co-observation ``edge_count``: nodes ``(fi, di)`` and ``(fj, dj)`` gain weight per 3-D point
    inside both masks with the same base label. Mutates ``det_label`` with ``{(fi, di): label}``."""
    edge_count: dict = defaultdict(int)
    n_points = pts.shape[0]
    point_obs: list[list[tuple[int, int]]] = [[] for _ in range(n_points)]

    for fi, dets, pi_arr, di_arr in _frame_observations(pts, cache, all_dets, label_maps):
        for pi, di in zip(pi_arr.tolist(), di_arr.tolist()):
            node = (fi, di)
            det_label[node] = dets[di]["label"]
            point_obs[pi].append(node)

    for obs in point_obs:
        if len(obs) < 2:
            continue
        by_label: dict = defaultdict(list)
        for node in obs:
            by_label[det_label[node]].append(node)
        for nodes in by_label.values():
            for ii in range(len(nodes)):
                for jj in range(ii + 1, len(nodes)):
                    a, b = nodes[ii], nodes[jj]
                    key = (a, b) if a <= b else (b, a)
                    edge_count[key] += 1
    return edge_count


def assign_labels_from_masks(
    pts: np.ndarray,
    cache: CameraCache,
    all_dets: list[list[dict]],   # updated IN-PLACE with instance_id
    label_maps: list[np.ndarray],
    min_votes: int = 1,
    min_label_pts: int = 3,
    anchor_min_pts: int = 2,
) -> tuple[np.ndarray, list[str]]:
    """Group detections into instances, then vote each 3-D point into the best instance.

    ``all_dets`` is mutated in place with the final ``instance_id`` so debug overlays see them."""
    N = pts.shape[0]
    print(
        f"\n── Step 4: instance assignment ({N:,} pts × {len(cache.frames)} frames) ──"
    )

    if not any(all_dets):
        print("[warn] No detections in any seg frame — returning empty.")
        return np.full(N, -1, dtype=np.int32), []

    # ── Build the co-observation graph from reprojection ───────────────────────
    det_label: dict = {}
    print("[graph] Building detection graph from reprojected SAM2 masks …")
    edge_count = _build_graph_from_reprojection(
        pts, cache, all_dets, label_maps, det_label,
    )

    # Include isolated detections so they don't silently vanish before voting.
    node_set: set = set(det_label.keys())
    for fi, dets in enumerate(all_dets):
        for di in range(len(dets)):
            node = (fi, di)
            if node not in det_label:
                det_label[node] = dets[di]["label"]
            node_set.add(node)
    all_nodes = sorted(node_set)

    # ── Union-Find over same-label detection pairs with enough edge weight ────
    uf = _UnionFind(all_nodes)
    min_shared = max(1, anchor_min_pts)

    if edge_count:
        weights = sorted(edge_count.values(), reverse=True)
        bins = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        hist = {b: sum(1 for w in weights if w >= b) for b in bins}
        print(
            "  Edge weight histogram (# edges with >=N shared pts): "
            + "  ".join(f">={b}:{hist[b]}" for b in bins)
        )

    n_linked = 0
    for (n1, n2), cnt in edge_count.items():
        if cnt >= min_shared and det_label.get(n1) == det_label.get(n2):
            uf.union(n1, n2)
            n_linked += 1
    print(
        f"  {len(all_nodes)} detection-nodes, "
        f"{len(edge_count)} candidate edges, "
        f"{n_linked} linked (>={min_shared} shared pts, same label)"
    )

    # ── Connected components → instances, sorted, named ────────────────────────
    comp_data: list[dict] = []
    for members in uf.components():
        label_votes = Counter(det_label.get(n, "unknown") for n in members)
        base_lbl = label_votes.most_common(1)[0][0]
        comp_data.append({
            "members": members,
            "base_label": base_lbl,
            "n_members": len(members),
        })

    if not comp_data:
        print("[warn] No instances formed — returning empty.")
        return np.full(N, -1, dtype=np.int32), []

    comp_data.sort(key=lambda c: (c["base_label"], -c["n_members"]))

    label_counter: Counter = Counter()
    node_to_inst: dict = {}
    for comp in comp_data:
        idx = label_counter[comp["base_label"]]
        comp["instance_id"] = f"{comp['base_label']}_{idx}"
        label_counter[comp["base_label"]] += 1
        for node in comp["members"]:
            node_to_inst[node] = comp["instance_id"]

    # Update all_dets in place so SAM2 debug overlays show instance IDs.
    for fi, dets in enumerate(all_dets):
        for di, det in enumerate(dets):
            det["instance_id"] = node_to_inst.get((fi, di), det["label"])

    # ── Vote each 3-D point for its instance_id ────────────────────────────────
    print("[vote] Assigning points to instances by majority vote …")
    all_inst_names = [c["instance_id"] for c in comp_data]
    inst_to_idx = {name: i for i, name in enumerate(all_inst_names)}

    pt_votes: list[Counter] = [Counter() for _ in range(N)]
    for fi, _dets, pi_arr, di_arr in _frame_observations(pts, cache, all_dets, label_maps):
        for pi, di in zip(pi_arr.tolist(), di_arr.tolist()):
            inst = node_to_inst.get((fi, di))
            if inst is not None:
                pt_votes[pi][inst] += 1

    final = np.full(N, -1, dtype=np.int32)
    for pi, votes in enumerate(pt_votes):
        if not votes:
            continue
        best = max(votes, key=votes.get)
        if votes[best] >= min_votes:
            final[pi] = inst_to_idx[best]

    # Drop instances with too few points after voting.
    counts = Counter(int(x) for x in final[final >= 0])
    valid = {iid for iid, cnt in counts.items() if cnt >= min_label_pts}
    remap = {old: new for new, old in enumerate(sorted(valid))}
    final = np.array(
        [remap[int(i)] if int(i) in valid else -1 for i in final],
        dtype=np.int32,
    )
    kept = [all_inst_names[old] for old in sorted(valid)]

    print(f"[label] Unique instances: {kept}")
    for i, name in enumerate(kept):
        n_frame_dets = sum(c["n_members"] for c in comp_data if c["instance_id"] == name)
        print(
            f"  [{i}] '{name}': {int((final==i).sum()):,} pts  "
            f"(seen in {n_frame_dets} frame-detections)"
        )
    print(f"  unlabelled: {int((final==-1).sum()):,} pts")
    return final, kept
