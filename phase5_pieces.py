
"""phase5_pieces.py - group tracks into pieces: tracks that lie on the same sheet, side by side.

Every 3rd track point is paired with points of OTHER tracks within r_max voxels. A pair is kept only if the two points sit
at the same depth (offset along the mean normal <= d_perp_max). Each kept pair gets a weight that falls off with the depth
offset (sigma_n), the sideways distance (sigma_par), the angle between the two tracks (alpha) and the verso probability
between them (verso_scale). Track pairs need at least min_support such point pairs. A track pair is dropped if the CT dips
between them (a dark gap: valley test, threshold valley_threshold). The tracks are then grouped by Louvain community
detection on the weighted track graph (louvain_resolution: higher = smaller, purer pieces).

All settings come from run_pipeline.py (P5 there). The maths: docs/MATH.md, section 5.
Results are identical to the earlier version (phase5_affinity_graph_wrap_disambiguation.py).
"""
from __future__ import annotations
import gc
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter


REPORT_MEMORY = True     # print memory after each step (set False to quieten the log)


def _mem(tag):
    """Memory now and the highest so far (whole process), so a run shows which step makes the peak. Linux only."""
    if not REPORT_MEMORY:
        return
    try:
        import resource
        now = int(open("/proc/self/statm").read().split()[1]) * resource.getpagesize() / 1e9
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6      # Linux reports KB
    except Exception:
        return
    print(f"      memory after {tag}: now {now:.1f} GB | highest so far {peak:.1f} GB", flush=True)


def trilinear_sample(arr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    hi = np.array(arr.shape[:3], dtype=float) - 1.0
    p = np.clip(pts, 0.0, hi - 1e-6)
    i0 = np.floor(p).astype(np.int64)
    f = (p - i0).astype(np.float32)
    i1 = np.minimum(i0 + 1, hi.astype(np.int64))
    z0, y0, x0 = i0.T
    z1, y1, x1 = i1.T
    fz, fy, fx = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    c00 = arr[z0, y0, x0] * (1 - fx) + arr[z0, y0, x1] * fx
    c01 = arr[z0, y1, x0] * (1 - fx) + arr[z0, y1, x1] * fx
    c10 = arr[z1, y0, x0] * (1 - fx) + arr[z1, y0, x1] * fx
    c11 = arr[z1, y1, x0] * (1 - fx) + arr[z1, y1, x1] * fx
    return (c00 * (1 - fy) + c01 * fy) * (1 - fz) + (c10 * (1 - fy) + c11 * fy) * fz


def _segment_values(vol4, p_i, p_j, K=9):
    """vol4: (Z,Y,X,1). Returns (M, K) samples along each segment."""
    t = np.linspace(0.0, 1.0, K)[None, :, None]
    pts = p_i[:, None, :] + t * (p_j - p_i)[:, None, :]
    return trilinear_sample(vol4, pts.reshape(-1, 3)).reshape(len(p_i), K)


def build_pieces(
    tracklet_paths,
    field_stack,
    raw_vol=None,               # the tile's CT; needed for the valley test (without it the test is skipped)
    r_max=3.5,
    d_perp_max=1.0,
    sigma_par=8.0,
    sigma_n=1.0,
    alpha=2.5,
    tau_W=0.005,
    min_support=3,
    valley_threshold=0.05,
    verso_scale=0.15,
    verso_mode="integral",
    verso_level=0.15,
    verso_tau=0.04,
    louvain_resolution=3.0,
    subsample=3,
    chunk=1_000_000,
    seed=42,
):
    """tracklet_paths: list of (n, 3) arrays (z, y, x). field_stack: (z, y, x, 9) with normals in 3-5 and the verso
    probability in 8. verso_mode "integral" (used) weighs by verso probability x distance; "level" by how far the mean
    verso probability exceeds verso_level. Returns component_id (piece of every track), the track-pair edge table and counts."""
    N = len(tracklet_paths)
    print(f"  [Phase 5] {N:,} tracklets", flush=True)
    _mem("start of Phase 5 (everything the pipeline already holds)")

    pts_l, tan_l, tid_l = [], [], []
    for i, path in enumerate(tracklet_paths):
        if len(path) < 2:
            continue
        diff = np.diff(path, axis=0)
        tan = np.zeros_like(path)
        tan[:-1] = diff / (np.linalg.norm(diff, axis=1, keepdims=True) + 1e-12)
        tan[-1] = tan[-2]
        pts_l.append(path); tan_l.append(tan); tid_l.append(np.full(len(path), i))
    if not pts_l:
        return None

    all_pts, all_tans, all_tids = np.vstack(pts_l), np.vstack(tan_l), np.concatenate(tid_l)
    del pts_l, tan_l, tid_l
    keep = np.arange(len(all_pts)) % subsample == 0
    P, T, ID = all_pts[keep], all_tans[keep], all_tids[keep]
    del all_pts, all_tans, all_tids, keep

    tree = cKDTree(P)
    pairs = tree.query_pairs(r=r_max, output_type="ndarray")
    del tree
    pairs = pairs.astype(np.int32)                                 # exact: point indices fit in int32
    gc.collect()
    print(f"  [Phase 5] {len(P):,} points, {len(pairs):,} candidate pairs (r_max={r_max})", flush=True)
    _mem("candidate pairs")

    normals = field_stack[..., 3:6]
    n_pairs = len(pairs)
    i_all = np.empty(n_pairs, np.int32); j_all = np.empty(n_pairs, np.int32)
    dperp_all = np.empty(n_pairs, np.float64); dpar_all = np.empty(n_pairs, np.float64)
    n_kept = 0
    for s in range(0, n_pairs, chunk):
        c = pairs[s:s + chunk]
        i, j = c[:, 0], c[:, 1]
        m = ID[i] != ID[j]
        i, j = i[m], j[m]
        n_i = trilinear_sample(normals, P[i]); n_j = trilinear_sample(normals, P[j])
        n_j *= np.sign(np.sum(n_i * n_j, axis=1, keepdims=True) + 1e-12)
        n_m = n_i + n_j
        n_m /= np.linalg.norm(n_m, axis=1, keepdims=True) + 1e-12
        d = P[j] - P[i]
        dperp = np.abs(np.sum(d * n_m, axis=1))
        ok = dperp <= d_perp_max                                   # hard cutoff BEFORE support is counted
        k = int(ok.sum()); sl = slice(n_kept, n_kept + k)
        i_all[sl] = i[ok]; j_all[sl] = j[ok]
        dperp_all[sl] = dperp[ok]
        dpar_all[sl] = np.maximum(0.0, np.sum(d[ok] ** 2, axis=1) - dperp[ok] ** 2)
        n_kept += k
    del pairs
    i, j = i_all[:n_kept].copy(), j_all[:n_kept].copy()
    dperp, dpar = dperp_all[:n_kept].copy(), dpar_all[:n_kept].copy()
    del i_all, j_all, dperp_all, dpar_all
    gc.collect()
    print(f"  [Phase 5] {len(i):,} pairs after d_perp<={d_perp_max}", flush=True)
    _mem("d_perp cut")

    ti, tj = ID[i].astype(np.int64), ID[j].astype(np.int64)
    key = np.minimum(ti, tj) * N + np.maximum(ti, tj)
    del ti, tj
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    m = cnt[inv] >= min_support
    del uniq, inv, cnt
    i, j, dperp, dpar, key = i[m], j[m], dperp[m], dpar[m], key[m]
    del m
    gc.collect()
    print(f"  [Phase 5] {len(i):,} pairs after support>={min_support}", flush=True)
    _mem("support count")

    w0 = np.exp(-dperp ** 2 / (2 * sigma_n ** 2))
    w0 *= np.exp(-dpar / (2 * sigma_par ** 2))
    del dperp, dpar
    for s in range(0, len(i), 2_000_000):          # tangent factor in chunks, no (n,3) temporaries
        sl = slice(s, s + 2_000_000)
        w0[sl] *= np.abs(np.einsum("ij,ij->i", T[i[sl]], T[j[sl]])) ** alpha
    m = w0 >= tau_W
    i, j, key, w0 = i[m], j[m], key[m], w0[m]
    del m
    gc.collect()
    print(f"  [Phase 5] {len(i):,} pairs after weight>={tau_W}", flush=True)
    _mem("weight cut")

    vol_s = None
    if raw_vol is not None and valley_threshold is not None:
        vol_s = gaussian_filter(raw_vol, 1.0)[..., None]
    else:
        print("  [Phase 5] WARNING: valley test disabled (raw_vol not given)", flush=True)
    p_verso = field_stack[..., 8:9]
    _mem("smoothed CT for the valley test")

    W = np.empty(len(i)); V = np.zeros(len(i))
    for s in range(0, len(i), 500_000):
        sl = slice(s, s + 500_000)
        pi, pj = P[i[sl]], P[j[sl]]
        dist = np.linalg.norm(pj - pi, axis=1)
        v_mean = _segment_values(p_verso, pi, pj).mean(axis=1)
        if verso_mode == "level":
            pen = np.exp(-np.maximum(0.0, v_mean - verso_level) / verso_tau)
        else:
            pen = np.exp(-v_mean * dist / verso_scale)
        W[sl] = w0[sl] * pen
        if vol_s is not None:
            v = _segment_values(vol_s, pi, pj)
            ends = np.minimum(v[:, 0], v[:, -1])
            V[sl] = np.maximum(0.0, (ends - v.min(axis=1)) / (ends + 1e-6))
    del i, j, w0, vol_s
    keep = W >= tau_W
    print(f" [Phase 5] {int(keep.sum()):,} pairs after verso penalty (scale={verso_scale})", flush=True)
    _mem("verso penalty and valley samples")

    df = pd.DataFrame({"key": key[keep], "W": W[keep], "V": V[keep]})   # same rows as before, built once
    del key, W, V, keep
    gc.collect()
    g = df.groupby("key").agg(W=("W", "max"), V=("V", "median"), n=("W", "size")).reset_index()
    del df
    gc.collect()
    _mem("grouping pairs into tracklet edges")
    n_before = len(g)
    if raw_vol is not None and valley_threshold is not None:
        g = g[g["V"] <= valley_threshold]
    rejected = n_before - len(g)
    print(f"  [Phase 5] {n_before:,} tracklet-pair edges; valley test rejected {rejected:,}; kept {len(g):,}", flush=True)

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_weighted_edges_from(zip((g["key"] // N).astype(int), (g["key"] % N).astype(int), g["W"].astype(float)))
    comms = nx.community.louvain_communities(G, weight="weight", resolution=louvain_resolution, seed=seed)
    component_id = np.zeros(N, dtype=np.int32)
    for c, nodes in enumerate(comms):
        component_id[list(nodes)] = c
    del G
    print(f"  [Phase 5] Louvain (resolution={louvain_resolution}): {len(comms):,} communities", flush=True)
    _mem("Louvain")

    return {"n_components": len(comms), "component_id": component_id,
            "edges": g, "valley_rejected": rejected}


build_correction_engine = build_pieces      # old name, kept so older notebooks still run
