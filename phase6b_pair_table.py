
"""phase6b_pair_table.py - phase 6, step b (run by run_layers.py): measure what lies between neighbouring sheets.

For many points on every sheet it walks outward along the LOCAL surface direction (not one fixed direction for the
whole chunk, so it also works at bends) until it meets the next sheet, and records what lies between the two.

Each such sample gets a class from what is seen at that spot:
    SEP      one real dark gap between the two sheets               -> two layers here, plain to see
    FAR      two or more dark gaps                                  -> more than one layer apart (not used for labels)
    THIN     no dark between, and the bright band holding both is thin with dark on both sides -> one layer here, plain to see
    PRESSED  no dark between, and the band is thick or has no dark edge -> cannot tell here (the hard case)
    AMBIG    a short dark stretch (1.5 to --min_dark voxels)
    EDGE     too close to the chunk wall to measure the band

Each PAIR of sheets then gets a label from its plain samples:
    two layers  if it has clear SEP samples and hardly any THIN ones
    same layer  if it has clear THIN samples and hardly any SEP ones
The PRESSED samples of a labelled pair are the hard examples with a known answer: the answer comes from another
place on the same two sheets, not from the pressed spot itself.

    python phase6b_pair_table.py --ckpt ckpt_t1 --out pairs_t1      (labels default to sheet_id_split.npy from step a)
Writes  <out>/pair_samples.npz, <out>/pair_table.csv, diagnostics/pair_profiles_<tile>.png
"""
import numpy as np, os, sys, json, time, argparse
from config import scroll_centre, tile_name
import pandas as pd
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ckpt_t1")
ap.add_argument("--out", default="pairs_t1")
ap.add_argument("--diag", default="diagnostics")
ap.add_argument("--labels", default="sheet_id_split.npy", help="per-track sheet labels in the ckpt folder (from phase 6a)")
ap.add_argument("--min_tracks", type=int, default=30, help="sheets smaller than this are not used")
ap.add_argument("--samples", type=int, default=120000, help="start points, spread at random over all sheets")
ap.add_argument("--reach", type=float, default=30.0, help="how far to walk outward looking for the next sheet (voxels)")
ap.add_argument("--ray", type=float, default=1.2, help="a track point this close to the walk line counts as met")
ap.add_argument("--min_dark", type=float, default=3.0, help="a dark stretch this long is a real gap between layers (same as stack_layers)")
ap.add_argument("--thin_max", type=float, default=11.0, help="a bright band up to this thick, with dark on both sides, is taken as one layer")
ap.add_argument("--min_n", type=int, default=10, help="plain samples a pair needs before it gets a label")
ap.add_argument("--core", nargs=2, type=float, default=None, metavar=("CX", "CY"), help="scroll centre; default: from the tile's manifest / config.py")
ap.add_argument("--field", default="", help="optional: path to field_stack.npy OF THIS TILE, to add recto / verso measurements")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
t0 = time.time(); os.makedirs(a.out, exist_ok=True); os.makedirs(a.diag, exist_ok=True)

# ------------------------------------------------------------------ load
if not os.path.exists(os.path.join(a.ckpt, "paths.npy")):
    sys.exit(f"Cannot find {a.ckpt}/paths.npy from {os.getcwd()}. Run this from the folder that holds {a.ckpt}/ (the Drive Python folder).")
man = json.load(open(os.path.join(a.ckpt, "manifest.json")))
off = np.array(man["chunk"]["offset_zyx"], dtype=np.float64)
a.core = scroll_centre(man, a.core); TILE = tile_name(a.ckpt, man)
L = lambda name: np.load(os.path.join(a.ckpt, name))
P = L("paths.npy").astype(np.float64); lens = L("lens.npy"); labels = L(a.labels).astype(np.int64)
bright = np.asarray(L("bright_f16.npy"), dtype=np.float32); vol = np.asarray(L("vol_f16.npy"), dtype=np.float32)
normals = np.asarray(L("normals_f16.npy"), dtype=np.float32)
N = len(lens); assert len(labels) == N and lens.sum() == len(P), "labels / lens / paths do not belong to the same run"
tid = np.repeat(np.arange(N), lens); shape = np.array(bright.shape)
ntr = np.bincount(labels); node_ids = np.flatnonzero(ntr >= a.min_tracks); K = len(node_ids)
node_of = -np.ones(len(ntr), np.int64); node_of[node_ids] = np.arange(K); node_pt = node_of[labels][tid]
print(f"[PAIR TABLE] {a.ckpt}: {N:,} tracks, {len(P):,} points | {K} sheets with {a.min_tracks}+ tracks")

# optional recto / verso channels. The file is only used if its surface directions match this tile's (another tile's
# file has the same size, so the size alone proves nothing).
F7 = F8 = None
if a.field:
    if not os.path.exists(a.field): print(f"  --field {a.field} not found: recto / verso left out")
    else:
        fs = np.load(a.field, mmap_mode="r"); rng_ = np.random.default_rng(1)
        ii = np.c_[rng_.integers(0, shape[0], 3000), rng_.integers(0, shape[1], 3000), rng_.integers(0, shape[2], 3000)]
        ii = ii[np.lexsort((ii[:, 2], ii[:, 1], ii[:, 0]))]
        same_tile = tuple(fs.shape[:3]) == tuple(shape) and float(np.abs(np.asarray(fs[ii[:, 0], ii[:, 1], ii[:, 2], 3:6], dtype=np.float32) - normals[ii[:, 0], ii[:, 1], ii[:, 2]]).mean()) < 0.01
        if same_tile:
            F7 = np.asarray(fs[..., 7], dtype=np.float32); F8 = np.asarray(fs[..., 8], dtype=np.float32); print("  recto / verso channels loaded (file matches this tile)")
        else: print(f"  --field {a.field} belongs to a DIFFERENT tile (its surface directions do not match {a.ckpt}): recto / verso left out")

# ------------------------------------------------------------------ helpers
def unit(v): return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
def nrm_at(q):
    i = np.clip(np.round(q).astype(int), 0, shape - 1); return unit(normals[i[..., 0], i[..., 1], i[..., 2]].astype(np.float64))
def tl(v3, q):
    """Trilinear value of volume v3 at points q (n, 3) in (z, y, x). Points outside are read at the wall."""
    q = np.clip(q, 0, shape - 1 - 1e-6); i_ = np.floor(q).astype(int); f = (q - i_).astype(np.float32); z, y, x = i_.T
    c = lambda dz, dy, dx: v3[z + dz, y + dy, x + dx]
    c00 = c(0, 0, 0) * (1 - f[:, 2]) + c(0, 0, 1) * f[:, 2]; c01 = c(0, 1, 0) * (1 - f[:, 2]) + c(0, 1, 1) * f[:, 2]
    c10 = c(1, 0, 0) * (1 - f[:, 2]) + c(1, 0, 1) * f[:, 2]; c11 = c(1, 1, 0) * (1 - f[:, 2]) + c(1, 1, 1) * f[:, 2]
    return (c00 * (1 - f[:, 1]) + c01 * f[:, 1]) * (1 - f[:, 0]) + (c10 * (1 - f[:, 1]) + c11 * f[:, 1]) * f[:, 0]
def first_true(m):
    return np.where(m.any(1), m.argmax(1), -1)
def inside(q): return ((q >= 0) & (q <= shape - 1)).all(-1)

# ------------------------------------------------------------------ start points and their outward direction
rng = np.random.default_rng(a.seed); pool = np.flatnonzero(node_pt >= 0)
pick = rng.choice(pool, min(a.samples, len(pool)), replace=False)
p_all = P[pick]; A_all = node_pt[pick]; n_all = nrm_at(p_all)
radial = np.c_[np.zeros(len(p_all)), p_all[:, 1] + off[1] - a.core[1], p_all[:, 2] + off[2] - a.core[0]]
n_all *= np.where((n_all * radial).sum(1) < 0, -1.0, 1.0)[:, None]                 # outward = away from the scroll centre
tree = cKDTree(P)
# direction each track runs at each of its points (tracks follow the fibres), for the fibre-direction measurements
_end = np.cumsum(lens); _first = _end - lens; _last = _end - 1
tang = np.zeros_like(P); tang[1:-1] = P[2:] - P[:-2]
tang[_first] = P[np.minimum(_first + 1, _last)] - P[_first]; tang[_last] = P[_last] - P[np.maximum(_last - 1, _first)]
tang = unit(tang).astype(np.float32)
print(f"  {len(pick):,} start points | search tree built | {time.time() - t0:.0f}s")

STEP = 0.6; T = int(a.reach / STEP); s_ray = STEP * np.arange(1, T + 1)           # the walk outward
PAD, H = 20.0, 0.5; s_prof = np.arange(-PAD, a.reach + PAD + H / 2, H); M = len(s_prof)   # brightness line: from 20 below the start to 20 past the reach
i0 = int(round(PAD / H)); KP = int(round(PAD / H)); need = max(1, int(round(a.min_dark / H)))
f16 = (np.arange(16) + 0.5) / 16; CTX = slice(i0 - 16, i0 + 64)                    # fixed window: 8 voxels below the start to 32 above
out = {k: [] for k in ("A", "B", "p", "gap", "longest", "nreal", "min_b", "mean_b", "min_v", "dip", "below", "above", "total", "bounded",
                       "edge", "small", "ang", "bend", "b16", "v16", "bctx", "vctx", "rmax", "vmax", "r16", "w16", "az_a", "az_b", "tcross")}
n_start = 0
for c0 in range(0, len(p_all), 4000):
    pc, nc, Ac, ic = p_all[c0:c0 + 4000], n_all[c0:c0 + 4000], A_all[c0:c0 + 4000], pick[c0:c0 + 4000]; n_start += len(pc)
    # --- walk outward until a track point of another sheet is met
    q = pc[:, None, :] + s_ray[None, :, None] * nc[:, None, :]
    d, j = tree.query(q.reshape(-1, 3), k=1, distance_upper_bound=a.ray, workers=-1)
    found = np.isfinite(d); j = np.where(found, j, 0)
    lab = np.where(found, node_pt[j], -2).reshape(len(pc), T); j = j.reshape(len(pc), T)
    k = first_true((lab >= 0) & (lab != Ac[:, None])); ok = k >= 0
    if not ok.any(): continue
    small = ((lab == -1) & (np.arange(T)[None, :] < k[:, None])).any(1)[ok]        # a small piece (under --min_tracks) lies between
    pc, nc, Ac, ic, k, lab, j = pc[ok], nc[ok], Ac[ok], ic[ok], k[ok], lab[ok], j[ok]; r = np.arange(len(pc))
    Bc = lab[r, k]; ph = P[j[r, k]]; gap = np.maximum(((ph - pc) * nc).sum(1), 0.3); qh = pc + gap[:, None] * nc
    tA, tB = tang[ic], tang[j[r, k]]                                 # fibre direction of the lower track and of the upper track
    out["az_a"].append(np.abs(tA[:, 0])); out["az_b"].append(np.abs(tB[:, 0])); out["tcross"].append(np.abs((tA * tB).sum(1)))   # 1 = along the scroll axis | 1 = parallel, 0 = crossing
    # --- brightness along the same line
    qp = (pc[:, None, :] + s_prof[None, :, None] * nc[:, None, :]).reshape(-1, 3)
    b = tl(bright, qp).reshape(len(pc), M); v = tl(vol, qp).reshape(len(pc), M)
    btw = (s_prof[None, :] > 0) & (s_prof[None, :] < gap[:, None]); cnt = btw.sum(1); dark = b < 0.5
    db = dark & btw; cs = np.cumsum(db, 1); run = cs - np.maximum.accumulate(np.where(~db, cs, 0), 1)
    longest = run.max(1) * H; nreal = (run == need).sum(1)
    ih = np.clip(np.round((gap + PAD) / H).astype(int), 0, M - 1)
    min_b = np.where(cnt > 0, np.where(btw, b, np.inf).min(1), b[:, i0]); mean_b = np.where(cnt > 0, (b * btw).sum(1) / np.maximum(cnt, 1), b[:, i0])
    min_v = np.where(cnt > 0, np.where(btw, v, np.inf).min(1), v[:, i0]); dip = min_v / (0.5 * (v[:, i0] + v[r, ih]) + 1e-6)
    # --- the bright band that holds both: how far it runs below the start and above the met sheet before two dark samples in a row
    rev = dark[:, i0::-1]; kb = first_true(rev[:, :-1] & rev[:, 1:]); below = np.where(kb >= 0, kb * H, PAD)
    up = np.take_along_axis(dark, np.minimum(ih[:, None] + np.arange(KP + 1)[None, :], M - 1), 1); ka = first_true(up[:, :-1] & up[:, 1:]); above = np.where(ka >= 0, ka * H, PAD)
    edge = ~inside(pc - PAD * nc) | ~inside(qh + PAD * nc)
    # --- shape: do the two sheets face the same way, and does the surface bend here (direction change over 24 voxels around the scroll)
    ang = np.degrees(np.arccos(np.clip(np.abs((nrm_at(qh) * nc).sum(1)), 0, 1)))
    e = unit(np.cross(nc, np.array([1.0, 0.0, 0.0])))
    bend = np.degrees(np.arccos(np.clip(np.abs((nrm_at(pc + 12 * e) * nrm_at(pc - 12 * e)).sum(1)), 0, 1)))
    ci = np.clip(np.round((f16[None, :] * gap[:, None] + PAD) / H).astype(int), 0, M - 1)
    for key, val in (("A", Ac), ("B", Bc), ("p", pc.astype(np.float32)), ("gap", gap), ("longest", longest), ("nreal", nreal), ("min_b", min_b), ("mean_b", mean_b),
                     ("min_v", min_v), ("dip", dip), ("below", below), ("above", above), ("total", below + gap + above), ("bounded", (kb >= 0) & (ka >= 0)),
                     ("edge", edge), ("small", small), ("ang", ang), ("bend", bend), ("b16", np.take_along_axis(b, ci, 1).astype(np.float16)),
                     ("v16", np.take_along_axis(v, ci, 1).astype(np.float16)), ("bctx", b[:, CTX].astype(np.float16)), ("vctx", v[:, CTX].astype(np.float16))):
        out[key].append(val)
    if F7 is not None:
        rr = tl(F7, qp).reshape(len(pc), M); ww = tl(F8, qp).reshape(len(pc), M)
        out["rmax"].append(np.where(btw, rr, 0).max(1)); out["vmax"].append(np.where(btw, ww, 0).max(1))
        out["r16"].append(np.take_along_axis(rr, ci, 1).astype(np.float16)); out["w16"].append(np.take_along_axis(ww, ci, 1).astype(np.float16))
S = {k: np.concatenate(v) for k, v in out.items() if len(v)}
if "A" not in S: sys.exit("No start point met another sheet. Check --reach and --ray.")
n = len(S["A"]); print(f"  {n:,} of {n_start:,} start points met another sheet within {a.reach:g} voxels | {time.time() - t0:.0f}s")

# ------------------------------------------------------------------ class of each sample, from what is seen at that spot
NAMES = ["SEP", "FAR", "THIN", "PRESSED", "AMBIG", "EDGE"]; SEP, FAR, THIN, PRESSED, AMBIG, EDGE = range(6)
cls = np.full(n, PRESSED)
cls[S["edge"]] = EDGE
cls[~S["edge"] & S["bounded"] & (S["total"] <= a.thin_max)] = THIN
cls[S["longest"] >= 1.5] = AMBIG
cls[S["nreal"] == 1] = SEP
cls[S["nreal"] >= 2] = FAR
print("  what is seen at each spot: " + " | ".join(f"{NAMES[c]} {(cls == c).mean():.0%}" for c in range(6)))

# ------------------------------------------------------------------ label of each pair, from its plain samples
lo, hi = np.minimum(S["A"], S["B"]), np.maximum(S["A"], S["B"]); key, inv = np.unique(lo * K + hi, return_inverse=True)
C = np.bincount(inv * 6 + cls, minlength=len(key) * 6).reshape(len(key), 6); tot = C.sum(1); nS, nT, nP = C[:, SEP], C[:, THIN], C[:, PRESSED]
two = (nS >= a.min_n) & (nS >= 0.15 * tot) & (nT < nS / 3); one = (nT >= a.min_n) & (nT >= 0.15 * tot) & (nS < nT / 3)
conflict = (nS >= a.min_n) & (nT >= a.min_n) & ~two & ~one
plab = np.full(len(key), -1); plab[one] = 0; plab[two] = 1; plab[conflict] = 2          # 0 same layer, 1 two layers, 2 conflicting, -1 not enough plain samples
slab = plab[inv]
med = lambda x: pd.Series(x).groupby(inv).median().to_numpy()
table = pd.DataFrame({"sheet_a": node_ids[key // K], "sheet_b": node_ids[key % K], "samples": tot, "SEP": nS, "THIN": nT, "PRESSED": nP, "AMBIG": C[:, AMBIG], "FAR": C[:, FAR], "EDGE": C[:, EDGE],
                      "label": np.array(["unknown", "same layer", "two layers", "conflict"])[plab + 1], "gap_median": med(S["gap"]).round(2), "band_median": med(S["total"]).round(2),
                      "bend_median_deg": med(S["bend"]).round(1)}).sort_values("samples", ascending=False)
table.to_csv(os.path.join(a.out, "pair_table.csv"), index=False)
big = tot >= 2 * a.min_n
print(f"\n[PAIRS] {len(key):,} neighbouring pairs of sheets, {int(big.sum()):,} with {2 * a.min_n}+ samples")
print(f"  labelled TWO LAYERS: {int(two.sum()):,} | labelled SAME LAYER: {int(one.sum()):,} | conflicting (clear signs of both): {int(conflict.sum()):,} | not enough plain samples: {int((plab == -1).sum()):,}")
print(f"  of the labelled pairs, those that also have pressed spots (10+): two layers {int((two & (nP >= 10)).sum()):,} | same layer {int((one & (nP >= 10)).sum()):,}")

# ------------------------------------------------------------------ the hard examples with a known answer
hard = (cls == PRESSED) & (slab >= 0) & (slab <= 1); y = slab[hard]
print(f"\n[HARD EXAMPLES] pressed spots whose answer is known from elsewhere on the same two sheets: {int(hard.sum()):,}  (two layers {int((y == 1).sum()):,} | same layer {int((y == 0).sum()):,})")
print("  by how much the surface bends there (direction change over 24 voxels):")
for name, b0, b1 in (("flat   (under 10 deg)", 0, 10), ("curved (10-25 deg)   ", 10, 25), ("bent   (over 25 deg) ", 25, 999)):
    m_ = hard & (S["bend"] >= b0) & (S["bend"] < b1); ma = (S["bend"] >= b0) & (S["bend"] < b1)
    print(f"      {name}: two layers {int((slab[m_] == 1).sum()):>7,} | same layer {int((slab[m_] == 0).sum()):>7,} | all samples there {int(ma.sum()):>7,} (pressed {(cls[ma] == PRESSED).mean() if ma.any() else 0:.0%})")
rule_same = (S["nreal"] == 0) & (S["gap"] < 20.0)                                        # today's rule in stack_layers: no real dark gap and under 20 apart -> same layer
if hard.any():
    r_ = rule_same[hard]
    print(f"  today's rule on these: right on {((r_ & (y == 0)) | (~r_ & (y == 1))).mean():.0%} | of the two-layer ones it calls 'same layer': {r_[y == 1].mean() if (y == 1).any() else 0:.0%}   <- the figure a model has to beat")
if hard.any() and (y == 1).any() and (y == 0).any():
    from sklearn.metrics import roc_auc_score
    feats = [("distance between the two sheets", "gap"), ("thickness of the bright band", "total"), ("darkest point between (brightness)", "min_b"),
             ("darkest point between (raw CT)", "min_v"), ("dip in raw CT relative to the two sheets", "dip"), ("angle between the two sheets", "ang"),
             ("lower track runs along the scroll axis", "az_a"), ("upper track runs along the scroll axis", "az_b"), ("the two tracks run parallel", "tcross"),
             ("upper minus lower: runs along the scroll axis", "az_diff")]
    S["az_diff"] = S["az_b"] - S["az_a"]
    if "rmax" in S: feats += [("highest recto probability between", "rmax"), ("highest verso probability between", "vmax")]
    print("  how well ONE measurement alone separates them (0.50 = useless, 1.00 = perfect, either direction counts):")
    for name, k_ in feats:
        au = roc_auc_score(y, S[k_][hard]); print(f"      {name:<44} {max(au, 1 - au):.2f}  ({'higher' if au >= 0.5 else 'lower'} = two layers)")
    print(f"  how tracks run (0 = around the scroll, 1 = along its axis), all samples: lower track p10/p50/p90 {np.percentile(S['az_a'], [10, 50, 90]).round(2).tolist()}"
          f" | share running mostly along the axis (over 0.7): {(S['az_a'] > 0.7).mean():.0%} | mostly around (under 0.3): {(S['az_a'] < 0.3).mean():.0%}")
    print("  note: samples from one pair resemble each other. A model must be tested on pairs it never saw, not on held-back samples.")
else:
    print("  one of the two answers has no pressed examples, so there is nothing to separate yet. Send me this output.")

np.savez_compressed(os.path.join(a.out, "pair_samples.npz"), cls=cls.astype(np.int8), pair_label=slab.astype(np.int8), pair_index=inv.astype(np.int32),
                    sheet_a=node_ids[S["A"]], sheet_b=node_ids[S["B"]], class_names=np.array(NAMES), **{k: v for k, v in S.items() if k not in ("A", "B")})

# ------------------------------------------------------------------ picture: do the two answers look different between the sheets?
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6)); xs = f16
    for col, (arr, title) in enumerate(((S["b16"], "CT brightness between the two sheets"), (S["v16"], "raw CT between the two sheets"))):
        for lab_, colr, nm in ((1, "tab:red", "pressed, known two layers"), (0, "tab:blue", "pressed, known same layer")):
            m_ = hard & (slab == lab_)
            if m_.any():
                q25, q50, q75 = np.percentile(arr[m_].astype(np.float32), [25, 50, 75], axis=0)
                ax[col].plot(xs, q50, color=colr, label=f"{nm} ({int(m_.sum()):,})"); ax[col].fill_between(xs, q25, q75, color=colr, alpha=0.2)
        m_ = cls == SEP
        if m_.any(): ax[col].plot(xs, np.median(arr[m_].astype(np.float32), axis=0), color="grey", ls="--", label="plain dark gap, for scale")
        ax[col].set_title(title + " (middle line = typical, band = middle half)"); ax[col].set_xlabel("0 = lower sheet, 1 = upper sheet"); ax[col].legend(fontsize=8)
    for lab_, colr, nm in ((1, "tab:red", "pressed, known two layers"), (0, "tab:blue", "pressed, known same layer")):
        m_ = hard & (slab == lab_)
        if m_.any(): ax[2].hist(S["total"][m_], bins=np.arange(0, 61, 2), color=colr, alpha=0.5, density=True, label=nm)
    ax[2].set_title("thickness of the bright band holding both sheets (voxels)"); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(a.diag, f"pair_profiles_{TILE}.png"), dpi=110); plt.close()
    print(f"  wrote {a.diag}/pair_profiles_{TILE}.png")
except Exception as ex:
    print(f"  picture skipped: {ex}")
print(f"  wrote {a.out}/pair_samples.npz, {a.out}/pair_table.csv | {time.time() - t0:.0f}s")
