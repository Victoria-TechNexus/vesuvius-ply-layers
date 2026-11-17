
"""phase6d_stack_layers.py - phase 6, step d (run by run_layers.py): combine tracks into layers using the stacking order.
No reference, no pipeline re-run. (Was stack_layers.py, version 5: ply labels.)

Reads the stage-1 files saved by run_pipeline.py (ckpt_<tile>/) and the ply labels from phase 6c (plies_<tile>/),
and writes one file per layer into layers_<tile>/.

    python phase6d_stack_layers.py --ckpt ckpt_t1 --plies plies_t1/pair_labels.csv --out layers_t1
    python phase6d_stack_layers.py --min_dark 2                 # count thinner dark gaps as a layer boundary
    python phase6d_stack_layers.py --labels component_id.npy    # start from the fine pieces instead of the joined sheets
    python phase6d_stack_layers.py --plies ""                   # dark-gap rule only, no ply labels
Labels default to sheet_id_split.npy (phase 6a). The scroll centre comes from the tile's manifest.json, else config.py;
--core overrides it.

What a layer is here
  A papyrus layer is a bright band in the CT. Two tracks next to each other in a column are in the SAME layer when there is
  no real dark gap between them (longest dark stretch shorter than --min_dark voxels) and they are closer than --max_thick.
  Otherwise they are one layer apart (or more, one per dark gap). Measured ply labels (--plies) overrule this rule for
  pressed pairs (no dark gap, but two layers) and delaminated pairs (a dark gap, but one layer).

How it works
  1. One direction "across the stack" is taken from the surface normals of all tracks.
  2. The chunk is cut into thin columns along that direction. In each column, what the column passes through is listed in order.
  3. Sheet level: every sheet from stage 1 gets a whole-number position that satisfies as many neighbour facts as possible.
  4. Track level: starting from that, every TRACK takes the position its own neighbours in the columns ask for. A sheet that
     sits in two bands is split here, and loose tracks are placed too. Stage-1 sheets are only a starting point.
  5. Check, with no reference: in a column, a layer should appear once, as one unbroken band ("double" near 0).
"""
import argparse, json, os, sys, time, heapq
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from config import scroll_centre, tile_name

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ckpt_t1")
ap.add_argument("--out", default="layers_t1")
ap.add_argument("--labels", default="sheet_id_split.npy", help="per-track labels used as the starting point (sheet_id_split.npy from phase 6a, sheet_id.npy or component_id.npy)")
ap.add_argument("--min_tracks", type=int, default=30, help="sheets smaller than this are not used at sheet level")
ap.add_argument("--cell", type=float, default=3.0, help="column width in voxels")
ap.add_argument("--same_gap", type=float, default=2.5, help="depth gap that separates two passes of one sheet or track through a column")
ap.add_argument("--min_dark", type=float, default=3.0, help="a dark stretch at least this long (voxels) between two tracks is a layer boundary")
ap.add_argument("--max_thick", type=float, default=20.0, help="tracks further apart than this are never the same layer, even with no dark gap")
ap.add_argument("--min_cols", type=int, default=5, help="columns a pair of sheets must share to count (sheet level)")
ap.add_argument("--max_gap", type=float, default=60.0, help="ignore neighbours further apart than this along a column")
ap.add_argument("--no_refine", action="store_true", help="stop at the sheet level (version 2 behaviour)")
ap.add_argument("--bright", default="bright_f16.npy", help="brightness file in the ckpt folder used to find dark gaps (bright_seam_f16.npy = with the seam model's seams darkened)")
ap.add_argument("--verdicts", default="", help="pair_verdicts.csv from train_pairs.py: where there is no dark gap, the model's verdict replaces 'same layer'")
ap.add_argument("--split_at", type=float, default=0.7, help="with --verdicts: a pair the model calls two layers with at least this confidence is put one layer apart")
ap.add_argument("--unsure_weight", type=float, default=0.2, help="with --verdicts: a pair the model is unsure about keeps 'same layer' but with this share of its weight")
ap.add_argument("--plies", default="plies_t1/pair_labels.csv", help="pair_labels.csv from phase6c_ply_label.py: measured same-layer / different-layer facts that overrule the dark-gap rule (\"\" = off)")
ap.add_argument("--ply_reach", type=float, default=30.0, help="with --plies: only neighbours closer than this (voxels) follow the ply label (phase 6c read gaps up to 30)")
ap.add_argument("--same_needs", type=int, default=1, help="with --plies: a same-layer reading (same layer, delaminated, same ply) joins two sheets only if at least this many spots of the pair were readable and agreed (1 = as before, 2 = both spots must agree)")
ap.add_argument("--ply_near", type=float, default=40.0, help="with --plies: a pair's label is used only within this many voxels of the spots it was read at (0 = everywhere)")
ap.add_argument("--core", nargs=2, type=float, default=None, metavar=("CX", "CY"), help="scroll centre; layer 00 is the innermost. Default: from the tile's manifest / config.py")
a = ap.parse_args()
t0 = time.time(); os.makedirs(a.out, exist_ok=True)
for f in os.listdir(a.out):                                       # clear old layer files so stale ones do not linger
    if f.endswith(".npz") and "layer_" in f: os.remove(os.path.join(a.out, f))

# ------------------------------------------------------------------ load, and refuse files that were cut short
if not os.path.exists(os.path.join(a.ckpt, "paths.npy")):
    sys.exit(f"Cannot find {a.ckpt}/paths.npy from {os.getcwd()}. Run this from the folder that holds {a.ckpt}/ (the Drive Python folder), or pass --ckpt and --out with full paths.")
man = json.load(open(os.path.join(a.ckpt, "manifest.json"))) if os.path.exists(os.path.join(a.ckpt, "manifest.json")) else {}
a.core = scroll_centre(man, a.core); TILE = tile_name(a.ckpt, man)
def load(name, mmap=False):
    path = os.path.join(a.ckpt, name); want = man.get("files", {}).get(name, {}).get("bytes")
    if want is not None and os.path.getsize(path) != want:
        sys.exit(f"{path} is {os.path.getsize(path):,} bytes but the manifest says {want:,}: the file was cut short. Re-run stage 1.")
    return np.load(path, mmap_mode="r" if mmap else None)
P = load("paths.npy").astype(np.float64); lens = load("lens.npy"); labels = load(a.labels).astype(np.int64)
bright = np.asarray(load(a.bright), dtype=np.float32); normals = load("normals_f16.npy", mmap=True)
N = len(lens); assert len(labels) == N and lens.sum() == len(P), "labels / lens / paths do not belong to the same run"
tid = np.repeat(np.arange(N), lens); shape = np.array(bright.shape)
ntr = np.bincount(labels); node_ids = np.flatnonzero(ntr >= a.min_tracks); K = len(node_ids)
node_of = -np.ones(len(ntr), np.int64); node_of[node_ids] = np.arange(K)
node_tr = node_of[labels]; node_pt = node_tr[tid]; npts = np.bincount(node_pt[node_pt >= 0], minlength=K)
PLY_SAME = PLY_DIFF = None; SPOT_TREE = None
if a.plies:
    if not os.path.exists(a.plies): sys.exit(f"Cannot find {a.plies}. Run phase6c_ply_label.py first (or pass --plies \"\" for the dark-gap rule only).")
    PLb = pd.read_csv(a.plies)
    same_l = {"same layer", "same layer, delaminated", "same ply"}; diff_l = {"different layers"}
    PLY_SAME = np.zeros((K, K), bool); PLY_DIFF = np.zeros((K, K), bool); miss = 0; weak_same = 0
    for r in PLb.itertuples():
        sa, sb = int(r.sheet_a), int(r.sheet_b)
        if max(sa, sb) >= len(node_of) or node_of[sa] < 0 or node_of[sb] < 0: miss += 1; continue
        x, y = node_of[sa], node_of[sb]
        if r.label in same_l:
            if int(getattr(r, "readable", 1)) < a.same_needs: weak_same += 1; continue      # one readable spot only: not trusted to join layers
            PLY_SAME[x, y] = PLY_SAME[y, x] = True
        elif r.label in diff_l: PLY_DIFF[x, y] = PLY_DIFF[y, x] = True
    SPOT_TREE = None; spf = os.path.join(os.path.dirname(a.plies) or ".", "pair_spots.csv")
    if a.ply_near > 0:
        if not os.path.exists(spf): sys.exit(f"Cannot find {spf} (written by phase6c_ply_label.py next to pair_labels.csv). Or pass --ply_near 0 to use labels everywhere.")
        SPt = pd.read_csv(spf); fam_ = {**{l: "same" for l in same_l}, **{l: "diff" for l in diff_l}}
        SPt = SPt[SPt["label"].map(fam_).notna()]
        sa_, sb_ = SPt["sheet_a"].to_numpy(int), SPt["sheet_b"].to_numpy(int); okk = (np.maximum(sa_, sb_) < len(node_of))
        SPt, sa_, sb_ = SPt[okk], sa_[okk], sb_[okk]; xa, xb = node_of[sa_], node_of[sb_]; okk = (xa >= 0) & (xb >= 0)
        SPt, xa, xb = SPt[okk], xa[okk], xb[okk]
        SPOT_KEY = np.minimum(xa, xb) * K + np.maximum(xa, xb); SPOT_FAM = SPt["label"].map(fam_).to_numpy()
        SPOT_TREE = cKDTree(SPt[["z", "y", "x"]].to_numpy(float))
        print(f"  ply labels used only within {a.ply_near:g} voxels of the {len(SPt):,} spots they were read at (--ply_near)")
    print(f"  ply labels ({a.plies}): same layer {int(PLY_SAME.sum() // 2)} pairs | different layers {int(PLY_DIFF.sum() // 2)} pairs"
          f" (strong only) | not usable here: {miss} (sheets with fewer than --min_tracks tracks, or other labels than phase 6c used)"
          + (f" | same-layer readings left out, fewer than {a.same_needs} agreeing spots (--same_needs): {weak_same}" if a.same_needs > 1 else ""))
def ply_fix(na, nb, apart, gap, where, mid=None):
    """the measured ply label overrules the dark-gap rule for neighbouring tracks of a labelled pair, near where it was read:
    different layers but no dark gap (pressed) -> one layer apart | same layer but one dark gap (delaminated) -> same layer"""
    if PLY_SAME is None: return apart
    ok = (na >= 0) & (nb >= 0) & (na < K) & (nb < K) & (na != nb) & (gap <= a.ply_reach)
    x, y = np.where(ok, na, 0), np.where(ok, nb, 0)
    to_diff = ok & PLY_DIFF[x, y] & (apart == 0); to_same = ok & PLY_SAME[x, y] & (apart == 1)
    if SPOT_TREE is not None and mid is not None:                       # keep only facts near a spot of the same pair that gave the same answer
        key = np.minimum(x, y) * K + np.maximum(x, y)
        for want, fam in ((to_diff, "diff"), (to_same, "same")):
            idx = np.flatnonzero(want)
            if not len(idx): continue
            hits = SPOT_TREE.query_ball_point(mid[idx], r=a.ply_near)
            keep = np.array([any(SPOT_KEY[j] == key[q] and SPOT_FAM[j] == fam for j in h) for q, h in zip(idx, hits)], bool)
            want[idx[~keep]] = False
    new = apart.copy(); new[to_diff] = 1; new[to_same] = 0
    print(f"  plies at {where}: {int(to_diff.sum()):,} neighbour facts changed same -> one layer apart (pressed) | {int(to_same.sum()):,} changed one apart -> same layer (delaminated)")
    return new
print(f"[LAYERS v5] {a.ckpt}: {N:,} tracks, {len(P):,} points | {K} sheets with {a.min_tracks}+ tracks hold {(node_pt >= 0).mean():.1%} of the points | start labels: {a.labels}")

# ------------------------------------------------------------------ 1. the direction across the stack
S = P[::20]; idx = np.clip(np.round(S).astype(int), 0, shape - 1)
n = np.asarray(normals[idx[:, 0], idx[:, 1], idx[:, 2]], dtype=np.float64); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
n0 = np.linalg.eigh(n.T @ n)[1][:, -1]; off = man.get("chunk", {}).get("offset_zyx")
if off is not None:
    out_dir = np.array([0.0, off[1] + shape[1] / 2 - a.core[1], off[2] + shape[2] / 2 - a.core[0]])
    if n0 @ out_dir < 0: n0 = -n0
tilt = np.degrees(np.arccos(np.clip(np.abs(n @ n0), 0, 1)))
print(f"  direction across the stack (z, y, x): {n0.round(3).tolist()} | tracks tilted from it: p50 {np.median(tilt):.0f} deg, p90 {np.percentile(tilt, 90):.0f} deg")
e = np.eye(3)[np.argmin(np.abs(n0))]; u = np.cross(n0, e); u /= np.linalg.norm(u); v = np.cross(n0, u)
ca = np.floor((P @ u - (P @ u).min()) / a.cell).astype(np.int64); cb = np.floor((P @ v - (P @ v).min()) / a.cell).astype(np.int64)
cid = ca * (cb.max() + 1) + cb; t = P @ n0

# ------------------------------------------------------------------ 2. columns: passes, and what lies between neighbours
def passes(lab, min_pts):
    """Group the points of each column by label and depth. Returns the passes sorted by (column, depth)."""
    o = np.lexsort((t, lab, cid)); c_s, l_s, t_s, P_s = cid[o], lab[o], t[o], P[o]
    brk = np.r_[True, (c_s[1:] != c_s[:-1]) | (l_s[1:] != l_s[:-1]) | (t_s[1:] - t_s[:-1] > a.same_gap)]
    st = np.flatnonzero(brk); cnt = np.diff(np.r_[st, len(o)])
    X = dict(cid=c_s[st], lab=l_s[st], n=cnt, t=np.add.reduceat(t_s, st) / cnt, pos=np.add.reduceat(P_s, st, axis=0) / cnt[:, None])
    k = X["n"] >= min_pts; X = {q: x[k] for q, x in X.items()}
    o = np.lexsort((X["t"], X["cid"])); return {q: x[o] for q, x in X.items()}
def tl(vol, q):
    q = np.clip(q, 0, shape - 1 - 1e-6); i0 = np.floor(q).astype(int); f = (q - i0).astype(np.float32); z, y, x = i0.T
    c = lambda dz, dy, dx: vol[z + dz, y + dy, x + dx]
    c00 = c(0, 0, 0) * (1 - f[:, 2]) + c(0, 0, 1) * f[:, 2]; c01 = c(0, 1, 0) * (1 - f[:, 2]) + c(0, 1, 1) * f[:, 2]
    c10 = c(1, 0, 0) * (1 - f[:, 2]) + c(1, 0, 1) * f[:, 2]; c11 = c(1, 1, 0) * (1 - f[:, 2]) + c(1, 1, 1) * f[:, 2]
    return (c00 * (1 - f[:, 1]) + c01 * f[:, 1]) * (1 - f[:, 0]) + (c10 * (1 - f[:, 1]) + c11 * f[:, 1]) * f[:, 0]
def between(X):
    """For each pass and the next one up in the same column: gap, longest dark stretch, number of real dark gaps, layers apart.
    Returns index i of the lower pass (the upper is i + 1) and the measurements."""
    i = np.flatnonzero((X["cid"][1:] == X["cid"][:-1]) & (X["t"][1:] - X["t"][:-1] <= a.max_gap))
    gap = X["t"][i + 1] - X["t"][i]; p1, p2 = X["pos"][i], X["pos"][i + 1]; seg = np.linalg.norm(p2 - p1, axis=1)
    longest = np.zeros(len(i)); nreal = np.zeros(len(i), np.int64); o = np.argsort(seg); need = max(1, int(round(a.min_dark / 0.5)))
    for s in range(0, len(i), 40000):
        k = o[s:s + 40000]; L_ = seg[k]; T = int(L_.max() / 0.5) + 2; step = np.arange(T) * 0.5
        d = (p2[k] - p1[k]) / (L_[:, None] + 1e-12); q = p1[k][:, None, :] + step[None, :, None] * d[:, None, :]
        dark = (tl(bright, q.reshape(-1, 3)).reshape(len(k), T) < 0.5) & (step[None, :] <= L_[:, None])
        c = np.cumsum(dark, axis=1); run = c - np.maximum.accumulate(np.where(~dark, c, 0), axis=1)     # length of the dark run ending here
        longest[k] = run.max(1) * 0.5; nreal[k] = (run == need).sum(1)                                   # each run that reaches min_dark counts once
    apart = np.where((nreal == 0) & (gap < a.max_thick), 0, np.maximum(1, nreal))
    return i, gap, longest, nreal, apart

# ------------------------------------------------------------------ 3. sheet level
Xs = passes(np.where(node_pt >= 0, node_pt, K), 3); n_cells = len(np.unique(Xs["cid"]))
i, gap, longest, nreal, apart = between(Xs); A, B = Xs["lab"][i], Xs["lab"][i + 1]
apart = ply_fix(np.where(A < K, A, -1), np.where(B < K, B, -1), apart, gap, "sheet level", (Xs["pos"][i] + Xs["pos"][i + 1]) / 2)
print(f"  {n_cells:,} columns ({a.cell:g} voxels wide) | sheet passes {len(Xs['cid']):,}")
_e = [0, 0.5, 1, 2, 3, 4, 6, 8, 12, 1e9]; _h = np.histogram(longest[A != B], bins=_e)[0]
print("  longest dark stretch between neighbouring sheets (voxels: share): " + ", ".join(f"{_e[k]:g}-{_e[k + 1]:g}: {_h[k] / max(_h.sum(), 1):.0%}" for k in range(len(_h) - 1)) + f", 12+: {_h[-1] / max(_h.sum(), 1):.0%}"
      + f"\n      -> a layer boundary is a dark stretch of {a.min_dark:g}+ voxels (change with --min_dark); neighbours judged same layer: {(apart[A != B] == 0).mean():.0%}")
m = (A != B) & (A < K) & (B < K); x_, y_ = np.minimum(A[m], B[m]), np.maximum(A[m], B[m]); sgn = np.where(A[m] == x_, 1, -1)
obs = pd.DataFrame({"lo": x_, "hi": y_, "sd": sgn * apart[m], "gap": gap[m], "dark": longest[m], "far": apart[m] > 1, "ord": sgn})
g = obs.groupby(["lo", "hi"]).agg(n=("sd", "size"), med=("sd", "median"), up=("sd", lambda s: int((s > 0).sum())), down=("sd", lambda s: int((s < 0).sum())),
                                  same=("sd", lambda s: int((s == 0).sum())), gap=("gap", "median"), dark=("dark", "median"), far=("far", "mean"), ord=("ord", "mean")).reset_index()
g = g[g["n"] >= a.min_cols].reset_index(drop=True); g["d"] = np.round(g["med"]).astype(int)
g["conflict"] = (g["d"] != 0) & (np.minimum(g["up"], g["down"]) >= 0.2 * g["n"]); g["w"] = g["n"] * np.where(g["far"] >= 0.5, 0.3, 1.0)
model_cut = np.zeros((K, K), bool)                                   # pairs whose 'same layer' the model overruled or doubted (also used at track level)
if a.verdicts:
    if not os.path.exists(a.verdicts): sys.exit(f"Cannot find {a.verdicts}. Run phase6b_pair_table.py and train_pairs.py first.")
    V_ = pd.read_csv(a.verdicts); V_ = V_[(V_["pressed_samples"] >= 10) & V_["model_two_layers"].notna()]
    vd = {(int(r.sheet_a), int(r.sheet_b)): float(r.model_two_layers) for r in V_.itertuples()}
    glo, ghi = g["lo"].to_numpy(), g["hi"].to_numpy()
    pm = np.array([vd.get((int(min(node_ids[x], node_ids[y])), int(max(node_ids[x], node_ids[y]))), np.nan) for x, y in zip(glo, ghi)])
    rule_same = g["d"].to_numpy() == 0; order_clear = np.abs(g["ord"].to_numpy()) >= 0.6      # which of the two is the lower one must be plain
    split = rule_same & (pm >= a.split_at) & order_clear
    unsure = rule_same & ~split & (pm > 0.3)
    g.loc[split, "d"] = np.where(g["ord"].to_numpy()[split] > 0, 1, -1); g.loc[unsure, "w"] = g.loc[unsure, "w"] * a.unsure_weight
    model_cut[glo[split | unsure], ghi[split | unsure]] = True; model_cut |= model_cut.T
    print(f"  model verdicts ({a.verdicts}): of {int(rule_same.sum())} pairs the dark-gap rule calls 'same layer', the model has a verdict on {int((rule_same & ~np.isnan(pm)).sum())} | "
          f"agrees 'same layer' {int((rule_same & (pm <= 0.3)).sum())} | put one layer apart {int(split.sum())} | unsure, weight cut to {a.unsure_weight:g} {int(unsure.sum())}")
E = g[~g["conflict"]].reset_index(drop=True)
lo, hi, dd, ww = E["lo"].to_numpy(), E["hi"].to_numpy(), E["d"].to_numpy(), E["w"].to_numpy()
ncomp, comp = connected_components(coo_matrix((np.ones(len(E)), (lo, hi)), shape=(K, K)), directed=False)
adj = [[] for _ in range(K)]
for x, y, d_, w_ in zip(lo, hi, dd, ww): adj[x].append((y, +d_, w_)); adj[y].append((x, -d_, w_))
Lpos = np.zeros(K, np.int64); placed = np.zeros(K, bool)
for c in range(ncomp):
    mem = np.flatnonzero(comp == c)
    if len(mem) == 1 and not adj[mem[0]]: continue
    root = mem[np.argmax(npts[mem])]; placed[root] = True
    heap = [(-w_, root, y, d_) for y, d_, w_ in adj[root]]; heapq.heapify(heap)
    while heap:
        _, x, y, d_ = heapq.heappop(heap)
        if placed[y]: continue
        Lpos[y] = Lpos[x] + d_; placed[y] = True
        for z, d2, w2 in adj[y]:
            if not placed[z]: heapq.heappush(heap, (-w2, y, z, d2))
for it in range(50):
    changed = 0
    for x in np.argsort(-npts):
        if not placed[x] or not adj[x]: continue
        votes = {}
        for y, d_, w_ in adj[x]: votes[Lpos[y] - d_] = votes.get(Lpos[y] - d_, 0.0) + w_
        best = max(votes, key=votes.get)
        if best != Lpos[x] and votes[best] > votes.get(Lpos[x], 0.0): Lpos[x] = best; changed += 1
    if not changed: break
stack_size = np.array([npts[(comp == c) & placed].sum() for c in range(ncomp)]); main_c = int(np.argmax(stack_size))
in_main = placed & (comp == main_c); Lpos[in_main] -= Lpos[in_main].min()
agree_s = float(ww[(Lpos[hi] - Lpos[lo] == dd) & in_main[lo] & in_main[hi]].sum() / max(ww.sum(), 1e-9))
print(f"  sheet level: {len(g)} neighbouring pairs (same layer {(E['d'] == 0).sum()}, one apart {(E['d'].abs() == 1).sum()}, further {(E['d'].abs() > 1).sum()}) | "
      f"{len(np.unique(Lpos[in_main]))} positions | neighbour facts that hold: {agree_s:.1%} | sheets outside the main stack: {int((~in_main).sum())}")
sheet_layer_tr = np.where((node_tr >= 0) & in_main[np.maximum(node_tr, 0)], Lpos[np.maximum(node_tr, 0)], -1)      # per track, from its sheet

# ------------------------------------------------------------------ 4. track level: every track takes the position its own neighbours ask for
track_layer = sheet_layer_tr.copy(); PAD = 4
if not a.no_refine:
    Xt = passes(tid, 2); i, gap, longest, nreal, apart = between(Xt); tlo, thi = Xt["lab"][i], Xt["lab"][i + 1]
    apart = ply_fix(node_tr[tlo], node_tr[thi], apart, gap, "track level", (Xt["pos"][i] + Xt["pos"][i + 1]) / 2)
    w_obs = np.where(apart > 1, 0.3, 1.0); Bn = int(track_layer.max()) + 1 + 2 * PAD
    if model_cut.any():                                              # tracks of two sheets the model overruled or doubted must not pull each other together
        na_, nb_ = node_tr[tlo], node_tr[thi]; w_obs[(na_ >= 0) & (nb_ >= 0) & model_cut[np.maximum(na_, 0), np.maximum(nb_, 0)] & (apart == 0)] = 0.0
    L = np.where(track_layer >= 0, track_layer + PAD, -1); prior = L.copy(); rs = np.random.default_rng(0)
    def agreement(L):
        k = (L[tlo] >= 0) & (L[thi] >= 0); return float(w_obs[k & (L[thi] - L[tlo] == apart)].sum() / max(w_obs[k].sum(), 1e-9))
    a0 = agreement(L); print(f"  track level: {len(Xt['cid']):,} track passes, {len(i):,} neighbour facts | hold with the sheet positions: {a0:.1%}")
    for it in range(60):
        V = np.zeros((N, Bn))
        k = L[thi] >= 0; c = L[thi][k] - apart[k]; q = (c >= 0) & (c < Bn); np.add.at(V, (tlo[k][q], c[q]), w_obs[k][q])
        k = L[tlo] >= 0; c = L[tlo][k] + apart[k]; q = (c >= 0) & (c < Bn); np.add.at(V, (thi[k][q], c[q]), w_obs[k][q])
        k = prior >= 0; V[np.flatnonzero(k), prior[k]] += 1.0                         # the sheet's position counts as one extra vote
        best = V.argmax(1); has = V.sum(1) > 0; cur = np.where(L >= 0, V[np.arange(N), np.maximum(L, 0)], -1.0)
        want = has & (best != L) & (V[np.arange(N), best] > cur)
        go = want & ((L < 0) | (rs.random(N) < 0.5))                                  # move half at a time so neighbours do not swap back and forth
        L[go] = best[go]
        if want.sum() <= 0.0005 * N: break
    track_layer = np.where(L >= 0, L - PAD, -1); track_layer[track_layer >= 0] -= track_layer[track_layer >= 0].min()
    print(f"      after {it + 1} rounds: hold {agreement(L):.1%} | tracks that left their sheet's position: {int(((sheet_layer_tr >= 0) & (L - PAD != sheet_layer_tr)).sum()):,} "
          f"of {int((sheet_layer_tr >= 0).sum()):,} | loose tracks now placed: {int(((sheet_layer_tr < 0) & (L >= 0)).sum()):,} | still unplaced: {int((L < 0).sum()):,}")
    # how pure each stage-1 sheet is: share of its tracks in its most common layer
    pur = []
    for kk in range(K):
        tl_ = track_layer[node_tr == kk]; tl_ = tl_[tl_ >= 0]
        if len(tl_): b = np.bincount(tl_); pur.append((int(node_ids[kk]), len(tl_), int(b.argmax()), float(b.max() / len(tl_)), " ".join(f"{q}:{b[q]}" for q in np.argsort(-b)[:3] if b[q])))
    PU = pd.DataFrame(pur, columns=["sheet", "tracks", "main_layer", "share_in_main_layer", "layers:tracks"]).sort_values("tracks", ascending=False)
    PU.to_csv(os.path.join(a.out, "sheet_purity.csv"), index=False)
    chk_f = a.plies or os.path.join(os.path.dirname(os.path.normpath(a.ckpt)), f"plies_{TILE}", "pair_labels.csv")   # with --plies "": still check against this tile's ply labels if they exist
    if os.path.exists(chk_f):                                           # reference-free check: do the layers agree with the measured plies?
        main = dict(zip(PU["sheet"], PU["main_layer"])); rows_ = []
        for r in pd.read_csv(chk_f).itertuples():
            la, lb = main.get(int(r.sheet_a)), main.get(int(r.sheet_b))
            if la is None or lb is None: continue
            fam = "same" if r.label in ("same layer", "same layer, delaminated", "same ply") else "diff" if r.label in ("different layers", "likely different layers") else None
            if fam is None: continue
            rows_.append(dict(sheet_a=int(r.sheet_a), sheet_b=int(r.sheet_b), ply_label=r.label, layer_a=int(la), layer_b=int(lb),
                              broken=(fam == "same") != (la == lb), z=r.z, y=r.y, x=r.x))
        if rows_:
            CK = pd.DataFrame(rows_); CK.to_csv(os.path.join(a.out, "ply_check.csv"), index=False); br = CK[CK["broken"]]
            print(f"      ply check ({chk_f}): {len(CK)} labelled pairs | broken {len(br)} ({len(br) / len(CK):.0%}): "
                  f"ply says same but layers differ {int((br['ply_label'].isin(['same layer', 'same layer, delaminated', 'same ply'])).sum())} | "
                  f"ply says different but same layer {int((~br['ply_label'].isin(['same layer', 'same layer, delaminated', 'same ply'])).sum())}")
            if len(br):
                per = pd.concat([br["layer_a"], br["layer_b"]]).value_counts().sort_index()
                print("      broken ply facts per layer (layer: count): " + ", ".join(f"{int(k_)}: {v_}" for k_, v_ in per.items()) + f" | listed in {a.out}/ply_check.csv")
            # ---- contradictions: a 'different layers' fact between two sheets that a chain of 'same layer' facts joins.
            # They cannot all be true if every sheet is one layer. Facts read far apart point to a sheet that crosses layers
            # between them (split it); facts read close together point to one wrong reading.
            from collections import deque, Counter
            SAME = {"same layer", "same layer, delaminated", "same ply"}
            nb = {}
            for r in CK[CK["ply_label"].isin(SAME)].itertuples():
                for p_, q_ in ((r.sheet_a, r.sheet_b), (r.sheet_b, r.sheet_a)):
                    nb.setdefault(p_, []).append((q_, r.ply_label, (r.z, r.y, r.x)))
            def chain(s, t, depth=5):
                prev = {s: None}; dq = deque([(s, 0)])
                while dq:
                    u, d = dq.popleft()
                    if u == t: break
                    if d >= depth: continue
                    for v, lab_, at in nb.get(u, []):
                        if v not in prev: prev[v] = (u, lab_, at); dq.append((v, d + 1))
                if t not in prev: return None
                steps, v = [], t
                while prev[v] is not None: u, lab_, at = prev[v]; steps.append((u, v, lab_, at)); v = u
                return steps[::-1]
            lay = dict(zip(PU["sheet"], PU["main_layer"])); crows = []; cnt_s = Counter(); cnt_f = Counter()
            for r in CK[~CK["ply_label"].isin(SAME)].itertuples():
                st_ = chain(r.sheet_a, r.sheet_b)
                if not st_: continue
                pts = np.array([at for *_, at in st_] + [(r.z, r.y, r.x)], float)
                span = float(np.max(np.linalg.norm(pts[:, None] - pts[None], axis=-1)))
                sheets_ = [st_[0][0]] + [v for _, v, _, _ in st_]
                for s_ in sheets_: cnt_s[s_] += 1
                keys_ = [((min(u, v), max(u, v), lab_), at) for u, v, lab_, at in st_] + [((min(r.sheet_a, r.sheet_b), max(r.sheet_a, r.sheet_b), r.ply_label), (r.z, r.y, r.x))]
                for k_, _ in keys_: cnt_f[k_] += 1
                crows.append(dict(different=f"{r.sheet_a}-{r.sheet_b}", different_label=r.ply_label, different_at=f"z{r.z} y{r.y} x{r.x}",
                                  same_chain=" ".join([str(st_[0][0])] + [f"-[{lab_} z{at[0]} y{at[1]} x{at[2]}]- {v}" for u, v, lab_, at in st_]),
                                  sheets=" ".join(map(str, sheets_)), layers=" ".join(str(lay.get(s_, "?")) for s_ in sheets_),
                                  span=round(span), _keys=keys_))
            if crows:
                for c_ in crows:                                                # suspect = the fact of this contradiction that sits in the most contradictions
                    (k_, at_) = max(c_.pop("_keys"), key=lambda kk: cnt_f[kk[0]])
                    c_["suspect"] = f"{k_[0]}-{k_[1]} {k_[2]}"; c_["suspect_at"] = f"z{at_[0]} y{at_[1]} x{at_[2]}"; c_["suspect_in"] = cnt_f[k_]
                    c_["suspect_layers"] = f"{lay.get(k_[0], '?')}-{lay.get(k_[1], '?')}"
                CR = pd.DataFrame(crows).sort_values(["suspect_in", "span"], ascending=False); CR.to_csv(os.path.join(a.out, "ply_contradictions.csv"), index=False)
                sus = CR.drop_duplicates("suspect")[["suspect", "suspect_at", "suspect_layers", "suspect_in"]].head(12)
                print(f"      contradictions: {len(CR)} 'different layers' facts that a chain of 'same layer' facts joins | "
                      f"explained by {CR['suspect'].nunique()} suspect facts | the top 10 suspects explain {int(CR['suspect'].isin(sus['suspect'].head(10)).sum())} | listed in {a.out}/ply_contradictions.csv")
                print("      suspect facts (wrong reading, or a sheet crossing layers right there): " +
                      "; ".join(f"{r_.suspect} at {r_.suspect_at} [layers {r_.suspect_layers}]: {r_.suspect_in}" for r_ in sus.itertuples()))
                print("      sheets in the most contradictions (sheet [layer]: count): " + ", ".join(f"{s_} [{lay.get(s_, '?')}]: {c_}" for s_, c_ in cnt_s.most_common(12)))
                print("      facts in the most contradictions (a-b label: count): " + ", ".join(f"{u}-{v} {l_}: {c_}" for (u, v, l_), c_ in cnt_f.most_common(10)))
            else:
                print("      contradictions: none (the ply facts can all hold together)")
    print(f"      stage-1 sheets that sit in one layer (90%+ of their tracks): {(PU['share_in_main_layer'] >= 0.9).sum()} of {len(PU)} | "
          f"holding {PU.tracks[PU['share_in_main_layer'] >= 0.9].sum() / PU.tracks.sum():.0%} of the tracks")
    worst = PU[PU["tracks"] >= 150].sort_values("share_in_main_layer").head(10)
    print("      biggest sheets spread over several layers (sheet: share in its main layer | layer:tracks): " + "; ".join(f"{r.sheet}: {r.share_in_main_layer:.0%} | {r._5}" for r in worst.itertuples()))
    Xc = Xt; Xlab = track_layer[Xt["lab"]]; sep = np.zeros(len(Xt["cid"]) - 1, bool); sep[i] = apart >= 1
    sep |= (Xt["cid"][1:] == Xt["cid"][:-1]) & (Xt["t"][1:] - Xt["t"][:-1] > a.max_gap)
else:
    Xc = Xs; Xlab = np.where(Xs["lab"] < K, np.where(in_main[np.minimum(Xs["lab"], K - 1)], Lpos[np.minimum(Xs["lab"], K - 1)], -1), -1)
    sep = np.zeros(len(Xs["cid"]) - 1, bool); sep[i] = apart >= 1; sep |= (Xs["cid"][1:] == Xs["cid"][:-1]) & (Xs["t"][1:] - Xs["t"][:-1] > a.max_gap)

# ------------------------------------------------------------------ 5. check and files
# in a column, a layer should be ONE unbroken band: count the bands each layer forms per column
newband = np.r_[True, (Xc["cid"][1:] != Xc["cid"][:-1]) | (Xlab[1:] != Xlab[:-1]) | sep]
ok = Xlab >= 0; pc = pd.DataFrame({"cid": Xc["cid"][ok & newband], "g": Xlab[ok & newband]}).groupby(["g", "cid"]).size().reset_index(name="bands")
chk = pc.groupby("g").agg(cols=("cid", "size"), double=("bands", lambda s: float((s >= 2).mean())))
pt_layer = track_layer[tid]; rows = []
for l_ in sorted(chk.index):
    tr = np.flatnonzero(track_layer == l_); pm = pt_layer == l_; name = f"layer_{int(l_):02d}"
    np.savez_compressed(os.path.join(a.out, name + ".npz"), points_zyx=P[pm].astype(np.float32), track=tid[pm].astype(np.int32),
                        sheet=labels[tid[pm]].astype(np.int32), offset_zyx=np.array(off if off is not None else [0, 0, 0]))
    sh, shn = np.unique(labels[tr], return_counts=True); sh = sh[np.argsort(-shn)]
    rows.append(dict(file=name + ".npz", layer=int(l_), tracks=len(tr), points=int(pm.sum()), sheets=int((np.sort(shn)[::-1] >= 10).sum()),
                     coverage=round(chk.loc[l_, "cols"] / n_cells, 3), double=round(chk.loc[l_, "double"], 3), biggest_sheets=" ".join(str(int(q)) for q in sh[:6])))
Tb = pd.DataFrame(rows); Tb.to_csv(os.path.join(a.out, "layers.csv"), index=False)
np.save(os.path.join(a.out, "layer_id.npy"), track_layer); np.save(os.path.join(a.out, "layer_id_sheet_level.npy"), sheet_layer_tr)
g.assign(lo_sheet=node_ids[g["lo"]], hi_sheet=node_ids[g["hi"]], lo_pos=Lpos[g["lo"]], hi_pos=Lpos[g["hi"]])[
    ["lo_sheet", "hi_sheet", "n", "up", "down", "same", "d", "gap", "dark", "conflict", "lo_pos", "hi_pos"]].to_csv(os.path.join(a.out, "layer_pairs.csv"), index=False)
print(f"\n[LAYERS RESULT] {TILE}: {len(Tb)} layers (target about 22) | tracks placed {(track_layer >= 0).mean():.1%} | points placed {(pt_layer >= 0).mean():.1%}")
print("  layer | tracks | points | coverage of the chunk | double (0 = one unbroken band per column) | biggest sheets")
for r in Tb.itertuples():
    print(f"   {r.layer:3d}  | {r.tracks:6,} | {r.points:9,} | {r.coverage:6.1%} | {r.double:6.1%} | {r.biggest_sheets}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    vol = np.asarray(load("vol_f16.npy"), dtype=np.float32)
    zs = [int(shape[0] * f) for f in (0.2, 0.5, 0.8)]; fig, axs = plt.subplots(1, 3, figsize=(27, 9.5)); cm = plt.get_cmap("tab20")
    for ax, z in zip(axs, zs):
        ax.imshow(vol[z], cmap="gray"); sl = np.abs(P[:, 0] - z) <= 0.75
        un = sl & (pt_layer < 0); ax.plot(P[un, 2], P[un, 1], ",", color="white", alpha=0.35)
        mk = sl & (pt_layer >= 0); ax.scatter(P[mk, 2], P[mk, 1], s=1.2, c=cm((pt_layer[mk] % 20) / 20.0), linewidths=0)
        for l_ in np.unique(pt_layer[mk]):
            q = P[mk & (pt_layer == l_)]; k = np.argmin(np.abs(q[:, 2] - np.median(q[:, 2])))
            ax.text(q[k, 2], q[k, 1], str(int(l_)), color="yellow", fontsize=9, ha="center", va="center", fontweight="bold")
        ax.set_title(f"{TILE}, z = {z} (local) | colour and number = layer, white = not placed"); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(os.path.join(a.out, f"layers_slices_{TILE}.png"), dpi=100, bbox_inches="tight"); plt.close()
    print(f"  wrote {a.out}/layers_slices_{TILE}.png")
except Exception as ex:
    print(f"  (picture skipped: {ex})")
json.dump(dict(tile=TILE, labels=a.labels, settings=vars(a), direction_zyx=n0.tolist(), layers=int(len(Tb)), minutes=round((time.time() - t0) / 60, 1)),
          open(os.path.join(a.out, "layers.json"), "w"), indent=1)
print(f"  wrote {a.out}/: one .npz per layer, layers.csv, sheet_purity.csv, layer_pairs.csv, layer_id.npy | {time.time() - t0:.0f}s")
