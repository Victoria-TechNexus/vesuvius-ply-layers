
"""
phase6c_ply_label.py  -  phase 6, step c (run by run_layers.py): label every pair of neighbouring sheets as
same layer / different layers, from the fibre plies.

This is ply_count.py run automatically between every pair of neighbouring sheets in the pair table (phase 6b).
For each pair it takes up to --per_pair spots on the inner sheet (A), fits a surface to A there, cuts face-on slices
from inside A to beyond the outer sheet (B), reads the ply at every depth (H = fibres around the scroll, the recto;
V = fibres along the axis, the verso; . = gap) and applies the rules, going outward from A to B:

    A on H, B on V, nothing between            same layer               (recto then verso of one sheet)
    A on H, B on V, a gap between              same layer, delaminated  (one sheet split between its plies)
    a V then later an H between A and B        different layers         (crossed from one sheet's verso to the next one's recto)
    same ply direction across a WIDE gap       likely different layers  (two layers whose other plies are not seen; weaker)
    A and B on the same ply, no gap            same ply                 (two tracks on one ply: same layer, weaker evidence)
    no V anywhere in the window                unreadable               (an H-only region like sheet 359: the rules do not apply)
    anything else                              unreadable               (left out of training)

Same settings as ply_count.py: papyrus-only stripe measuring, grid turned 45 degrees, readings against 33%.
Each spot also records whether its four quarters agree, and whether any V was seen at all (sheet 359 had none).

Run from the folder that holds ckpt_t1/ and pairs_t1/ (after phase6b_pair_table.py):
    python phase6c_ply_label.py --ckpt ckpt_t1 --pairs pairs_t1 --out plies_t1     (labels default to sheet_id_split.npy from step a)
    python phase6c_ply_label.py --limit 50                                          (quick try on the 50 biggest pairs)
The scroll centre comes from the tile's manifest.json, else config.py; --core overrides it.
Writes  <out>/pair_spots.csv (one row per spot), <out>/pair_labels.csv (one row per pair),
        diagnostics/ply_labels_<tile>.png, diagnostics/ply_label_cards_<tile>.png
"""
import numpy as np, os, sys, json, time, argparse
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from itertools import groupby
from config import scroll_centre, tile_name

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ckpt_t1")
ap.add_argument("--labels", default="sheet_id_split.npy", help="per-track sheet labels in the ckpt folder (from phase 6a)")
ap.add_argument("--pairs", default="pairs_t1", help="output folder of phase 6b (holds pair_samples.npz)")
ap.add_argument("--out", default="plies_t1"); ap.add_argument("--diag", default="diagnostics")
ap.add_argument("--states", default="", help="optional: states/pair_states.csv from layer_state.py, for comparison")
ap.add_argument("--per_pair", type=int, default=2, help="spots per pair (spread apart along the pair)")
ap.add_argument("--min_n", type=int, default=20, help="pairs with fewer samples in the pair table are skipped")
ap.add_argument("--limit", type=int, default=0, help="only the N biggest pairs (0 = all)")
ap.add_argument("--radius", type=int, default=24, help="half size of each patch (voxels)")
ap.add_argument("--before", type=int, default=15, help="slices start this far inside sheet A")
ap.add_argument("--after", type=int, default=15, help="slices end this far outside sheet B")
ap.add_argument("--hp", type=float, default=4.0); ap.add_argument("--grad", type=float, default=1.0); ap.add_argument("--rho", type=float, default=4.0)
ap.add_argument("--coh", type=float, default=0.3); ap.add_argument("--v_max", type=float, default=30.0); ap.add_argument("--h_min", type=float, default=60.0)
ap.add_argument("--margin", type=float, default=0.2, help="H or V must rise this far above 33% (and above the other) to name a depth")
ap.add_argument("--dark", type=float, default=0.5, help="a depth where more than this share of the patch is not papyrus is a gap")
ap.add_argument("--papyrus", type=float, default=0.5, help="brightness from which a voxel counts as papyrus")
ap.add_argument("--core", nargs=2, type=float, default=None, metavar=("CX", "CY"), help="scroll centre; default: from the tile's manifest / config.py")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--centre", type=int, default=12, help="the plies are read in a square this far around the spot (the sheets drift apart further out)")
ap.add_argument("--wide", type=float, default=8.0, help="a gap at least this many voxels between two plies of the same direction counts as a layer gap, not a crack")
ap.add_argument("--delam_max", type=float, default=12.0, help="recto, gap, verso with a gap wider than this is only 'possibly delaminated' (could be two layers)")
ap.add_argument("--z", type=int, default=-1, help="slice for the overview picture (-1 = middle)")
ap.add_argument("--show_z", type=int, default=2, help="the overview picture shows spots within this many slices of --z")
ap.add_argument("--cards", type=int, default=8, help="spots shown per label in ply_label_cards_<tile>.png")
ap.add_argument("--ply", type=float, default=7.0, help="thickness of one ply (voxels); measured: recto-only bands 8, verso-only 7")
ap.add_argument("--double", type=float, default=1.7, help="a ply run at least this many plies thick counts as doubled")
ap.add_argument("--search", type=int, default=3, help="look this many slices either side of each sheet for its ply")
a = ap.parse_args()
t0 = time.time(); os.makedirs(a.out, exist_ok=True); os.makedirs(a.diag, exist_ok=True)

# ------------------------------------------------------------------ load
f = os.path.join(a.pairs, "pair_samples.npz")
for x in (os.path.join(a.ckpt, "paths.npy"), f):
    if not os.path.exists(x): sys.exit(f"Cannot find {x} from {os.getcwd()} (run phase6b_pair_table.py first for {f}).")
man = json.load(open(os.path.join(a.ckpt, "manifest.json"))); off = np.array(man["chunk"]["offset_zyx"], dtype=np.float64)
a.core = scroll_centre(man, a.core); TILE = tile_name(a.ckpt, man)
L = lambda name: np.load(os.path.join(a.ckpt, name))
P = L("paths.npy").astype(np.float64); lens = L("lens.npy"); sheet_pt = L(a.labels).astype(np.int64)[np.repeat(np.arange(len(lens)), lens)]
vol = np.asarray(L("vol_f16.npy"), dtype=np.float32); bright = np.asarray(L("bright_f16.npy"), dtype=np.float32)
normals = np.asarray(L("normals_f16.npy"), dtype=np.float32); shape = np.array(vol.shape); tree = cKDTree(P)
d = np.load(f); sp_p = d["p"].astype(np.float64); sp_gap = d["gap"].astype(np.float64); sA = d["sheet_a"]; sB = d["sheet_b"]
pidx = d["pair_index"].astype(np.int64); plab = d["pair_label"].astype(np.int64); scls = d["cls"].astype(np.int64)
print(f"[PLY LABEL] {a.ckpt}: {len(P):,} track points | {len(sp_p):,} pair-table samples in {pidx.max() + 1:,} pairs | {time.time() - t0:.0f}s")

def unit(v): return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
def tl(v3, q):
    q = np.clip(q, 0, shape - 1 - 1e-6); i_ = np.floor(q).astype(int); f_ = (q - i_).astype(np.float32); z, y, x = i_.T
    c = lambda dz, dy, dx: v3[z + dz, y + dy, x + dx]
    c00 = c(0, 0, 0) * (1 - f_[:, 2]) + c(0, 0, 1) * f_[:, 2]; c01 = c(0, 1, 0) * (1 - f_[:, 2]) + c(0, 1, 1) * f_[:, 2]
    c10 = c(1, 0, 0) * (1 - f_[:, 2]) + c(1, 0, 1) * f_[:, 2]; c11 = c(1, 1, 0) * (1 - f_[:, 2]) + c(1, 1, 1) * f_[:, 2]
    return (c00 * (1 - f_[:, 1]) + c01 * f_[:, 1]) * (1 - f_[:, 0]) + (c10 * (1 - f_[:, 1]) + c11 * f_[:, 1]) * f_[:, 0]

R = a.radius; Nn = 2 * R + 1
U, Vv = np.meshgrid(np.arange(Nn) - R, np.arange(Nn) - R, indexing="ij")
QUADS = [(U < 0) & (Vv < 0), (U < 0) & (Vv >= 0), (U >= 0) & (Vv < 0), (U >= 0) & (Vv >= 0)]

def fit_surface(spot, s):
    """surface of sheet s around spot on a grid turned 45 degrees to the scroll axis; None if too little of the sheet is there"""
    ii = np.clip(np.round(spot).astype(int), 0, shape - 1); n0 = unit(normals[ii[0], ii[1], ii[2]].astype(np.float64))
    rad = np.array([0.0, spot[1] + off[1] - a.core[1], spot[2] + off[2] - a.core[0]]); n0 = n0 if n0 @ rad >= 0 else -n0
    ax_ = unit(np.array([1.0, 0, 0]) - n0[0] * n0); ar_ = unit(np.cross(n0, ax_)); eu, ev = unit(ax_ + ar_), unit(ar_ - ax_)
    idx = np.array(tree.query_ball_point(spot, r=R * 1.6), dtype=np.int64)
    idx = idx[sheet_pt[idx] == s] if len(idx) else idx
    if len(idx) < 50: return None
    dd = P[idx] - spot; u, v, h = dd @ eu, dd @ ev, dd @ n0; k = (np.abs(u) <= R) & (np.abs(v) <= R) & (np.abs(h) <= 15)
    if k.sum() < 50: return None
    iu, iv = np.round(u[k]).astype(int) + R, np.round(v[k]).astype(int) + R
    sm = np.zeros((Nn, Nn)); cn = np.zeros((Nn, Nn)); np.add.at(sm, (iu, iv), h[k]); np.add.at(cn, (iu, iv), 1)
    have = cn > 0; hm = np.where(have, sm / np.maximum(cn, 1), 0)
    dist, (ri, ci) = ndi.distance_transform_edt(~have, return_indices=True)
    g1 = ndi.gaussian_filter(np.where(have, hm, 0), 3); g0 = ndi.gaussian_filter(have.astype(float), 3)
    hs = np.where(g0 > 0.05, g1 / np.maximum(g0, 1e-9), hm[ri, ci]); mask = dist <= 8
    S = spot + U[..., None] * eu + Vv[..., None] * ev + hs[..., None] * n0
    dhu, dhv = np.gradient(hs); N = unit(np.cross(eu + dhu[..., None] * n0, ev + dhv[..., None] * n0)); N = np.where((N @ n0)[..., None] < 0, -N, N)
    return dict(S=S, N=N, mask=mask, AX=(float(ax_ @ eu), float(ax_ @ ev)), cover=float(mask.mean()), resid=float(np.std(hm[have] - hs[have])),
                eu=eu, ev=ev, n0=n0, hs=hs)

def depth_of(spot, s, fs):
    """where sheet s really sits, along A's normal, in the centre window (its own track points); None if too few"""
    idx = np.array(tree.query_ball_point(spot, r=R * 1.6), dtype=np.int64)
    idx = idx[sheet_pt[idx] == s] if len(idx) else idx
    if len(idx) < 10: return None
    dd = P[idx] - spot; u, v, h = dd @ fs["eu"], dd @ fs["ev"], dd @ fs["n0"]
    k = (np.abs(u) <= a.centre) & (np.abs(v) <= a.centre)
    if k.sum() < 10: return None
    iu, iv = np.round(u[k]).astype(int) + R, np.round(v[k]).astype(int) + R
    return float(np.median(h[k] - fs["hs"][iu, iv]))

def stripes(img, m, w, AX):
    im = np.where(m, img, img[m].mean()); hp = im - ndi.gaussian_filter(im, a.hp)
    gr = ndi.gaussian_filter(hp, a.grad, order=(1, 0)); gc = ndi.gaussian_filter(hp, a.grad, order=(0, 1)); w = w.astype(np.float32)
    Jrr, Jcc, Jrc = (ndi.gaussian_filter(w * v, a.rho) for v in (gr * gr, gc * gc, gr * gc)); ws = ndi.gaussian_filter(w, a.rho)
    phi = 0.5 * np.arctan2(2 * Jrc, Jrr - Jcc)
    ang = np.degrees(np.arcsin(np.clip(np.abs(np.cos(phi) * AX[0] + np.sin(phi) * AX[1]), 0, 1)))
    coh = 2 * np.sqrt(0.25 * (Jrr - Jcc) ** 2 + Jrc ** 2) / (Jrr + Jcc + 1e-12)
    return ang, coh, ws

def letters(ANG, COH, WS, BR, m):
    """one letter per depth (H, V, . gap, ? unclear), with single-slice runs turned into ?"""
    K = len(ANG); lab = []
    for k in range(K):
        c = m & (COH[k] >= a.coh) & (WS[k] >= 0.3); dk = (BR[k][m] < a.papyrus).mean()
        if dk > a.dark: lab.append("."); continue
        if c.sum() <= 30: lab.append("?"); continue
        h, v = (ANG[k][c] >= a.h_min).mean() - 1 / 3, (ANG[k][c] <= a.v_max).mean() - 1 / 3
        lab.append("H" if (h >= a.margin and h > v) else "V" if (v >= a.margin and v > h) else "?")
    out, k = [], 0
    for ch, g in groupby(lab):
        n = len(list(g)); out += [ch if (n >= 2 or ch == ".") else "?"] * n; k += n
    return out

def bands(lab):
    """split the depth letters into bands of papyrus: a gap is a run of '.' and '?' holding at least 2 '.'"""
    n = len(lab); dark = [False] * n; i = 0
    while i < n:
        if lab[i] in ".?":
            j = i
            while j < n and lab[j] in ".?": j += 1
            if lab[i:j].count(".") >= 2: dark[i:j] = [True] * (j - i)
            i = j
        else: i += 1
    out, i = [], 0
    while i < n:
        if dark[i]: i += 1; continue
        j = i
        while j < n and not dark[j]: j += 1
        out.append((i, j - 1)); i = j
    return out

def ply_runs(lab, s, e):
    """runs of one ply direction inside lab[s..e]; '?' and single '.' do not break a run. Each run: [letter, length, first, last]"""
    out = []
    for k in range(max(s, 0), min(e, len(lab) - 1) + 1):
        c = lab[k]
        if c not in "HV": continue
        if out and out[-1][0] == c: out[-1][1] += 1; out[-1][3] = k
        else: out.append([c, 1, k, k])
    return out

def run_at(q, k, c):
    """length of the run of letter c through depth k in letters q (0 if q has another ply there)"""
    if not (0 <= k < len(q)) or q[k] != c: return 0
    n, j = 0, k
    while j >= 0 and q[j] in (c, "?"): n += q[j] == c; j -= 1
    j = k + 1
    while j < len(q) and q[j] in (c, "?"): n += q[j] == c; j += 1
    return n

def doubled(lab, s, e, qs):
    """doubled ply runs in lab[s..e]: (letter, length, first, last, ply on both sides, seen in fewer than half the quarters)"""
    runs, thr, out = ply_runs(lab, s, e), a.double * a.ply, []
    for i, (c, n, f, l) in enumerate(runs):
        if n < thr: continue
        local = None
        if qs: local = sum(run_at(q, (f + l) // 2, c) >= thr for q in qs) < len(qs) / 2
        out.append((c, n, f, l, 0 < i < len(runs) - 1, local))
    return runs, out

def band_info(lab, b, qs, tag):
    """ply runs, ply count, layer count and doubled-ply flag for one band"""
    empty = {f"runs_{tag}": "", f"plies_{tag}": np.nan, f"layers_{tag}": np.nan, f"double_{tag}": ""}
    if b is None: return empty
    runs, dbl = doubled(lab, b[0], b[1], qs)
    plies = sum(max(1, int(round(n / a.ply))) for _, n, _, _ in runs)
    out = dict(empty); out.update({f"runs_{tag}": " ".join(f"{c}{n}" for c, n, _, _ in runs), f"plies_{tag}": plies, f"layers_{tag}": plies / 2})
    if dbl:
        c, n, f, l, both, local = dbl[0]
        out[f"double_{tag}"] = ("thickening, local" if local else "fold?" if both else "thick ply") + f" ({c}{n})"
    return out

def classify(lab, kA, kB, qs=None):
    """returns label, reason, ply at A, ply at B, run sequence A..B, and band details"""
    def near(k):
        for d_ in range(a.search + 1):
            for j in ((k,) if d_ == 0 else (k - d_, k + d_)):
                if 0 <= j < len(lab) and lab[j] in "HV": return lab[j], j
        return None, None
    (A, jA), (B, jB) = near(kA), near(kB)
    lo = min(kA, jA) if jA is not None else kA; hi = max(kB, jB) if jB is not None else kB    # include the ply each sheet was read on
    seg = [c for c in lab[lo:hi + 1] if c != "?"]; runs = [c for c, _ in groupby(seg)]
    hv = [c for c in runs if c in "HV"]; seq = "".join(runs)
    BD = bands(lab); find = lambda j: next((b for b in BD if b[0] <= j <= b[1]), None)
    bA, bB = (find(jA) if jA is not None else None), (find(jB) if jB is not None else None)
    squash = lambda s: "".join(c for c, _ in groupby(c for c in s if c in "HV"))
    ex = dict(bandA=squash(lab[bA[0]:bA[1] + 1]) if bA else "", bandB=squash(lab[bB[0]:bB[1] + 1]) if bB else "",
              thickA=(bA[1] - bA[0] + 1) if bA else np.nan, thickB=(bB[1] - bB[0] + 1) if bB else np.nan, gap_w=np.nan)
    ex.update(band_info(lab, bA, qs, "A")); ex.update(band_info(lab, bB, qs, "B"))
    R_ = lambda l, why: (l, why, A, B, seq, ex)
    if any(hv[i] == "V" and "H" in hv[i + 1:] for i in range(len(hv))): return R_("different layers", "a verso then a recto between the sheets")
    if "V" not in lab: return R_("unreadable", "no verso anywhere in the window (like sheet 359)")
    if A is None or B is None or bA is None or bB is None: return R_("unreadable", "the ply at one of the two sheets is not clear")
    if bA == bB:                                                       # both sheets inside one band of papyrus
        if A == B:
            if kB - kA >= a.ply:                                       # further apart than one ply is thick: two plies of one direction
                return R_("unreadable", f"one {'recto' if A == 'H' else 'verso'} run, but the sheets are more than one ply apart: two pressed (layers apart) or one thick ply")
            return R_("same ply", "both sheets on one ply")
        if A == "H" and B == "V": return R_("same layer", "recto then verso")
        return R_("unreadable", "no rule fits")
    gw = bB[0] - bA[1] - 1; ex["gap_w"] = gw                           # a gap separates the two bands
    outA = lab[jA:bA[1] + 1]; inB = lab[bB[0]:jB + 1]                  # A's band outward of A | B's band inward of B
    if A == "H" and "V" in outA:
        return R_("likely different layers", "A's band already has its verso before the gap, so the band beyond is another layer")
    if B == "V" and "H" in inB:
        return R_("likely different layers", "B's band has its own recto, so it is a whole layer, not A's peeled verso")
    if A == B:
        if gw >= a.wide: return R_("likely different layers", f"same ply direction on both sides of a wide gap ({gw} voxels): the other plies are not seen")
        return R_("unreadable", "same ply direction on both sides of a narrow gap (crack or missing ply)")
    if A == "H" and B == "V":                                          # recto only | gap | verso only: one layer split
        if gw > a.delam_max: return R_("possibly delaminated", f"recto only, wide gap ({gw} voxels), verso only: a delamination or two layers")
        return R_("same layer, delaminated", "recto only, gap, verso only")
    return R_("unreadable", "no rule fits")

def read_spot(spot, s, g, sb):
    fs = fit_surface(spot, s)
    if fs is None: return None
    S, N, mask = fs["S"], fs["N"], fs["mask"]
    dB = depth_of(spot, sb, fs)                                     # where B really is along A's normal, near the spot
    use = dB if (dB is not None and 0.5 <= dB <= 40) else g; g_end = max(g, use)
    ts = np.arange(-a.before, int(np.ceil(g_end)) + a.after + 1); K = len(ts)
    q = S[None] + ts[:, None, None, None] * N[None]
    CT = tl(vol, q.reshape(-1, 3)).reshape(K, Nn, Nn); BR = tl(bright, q.reshape(-1, 3)).reshape(K, Nn, Nn)
    inner = ndi.binary_erosion(mask, iterations=int(1.5 * a.rho)); inner = inner if inner.sum() >= 150 else mask
    ANG = np.zeros_like(CT); COH = np.zeros_like(CT); WS = np.zeros_like(CT)
    for k in range(K): ANG[k], COH[k], WS[k] = stripes(CT[k], mask, ndi.binary_erosion(BR[k] > a.papyrus, iterations=2) & mask, fs["AX"])
    kA, kB = a.before, a.before + int(round(use))
    centre = inner & (np.abs(U) <= a.centre) & (np.abs(Vv) <= a.centre); centre = centre if centre.sum() >= 150 else inner
    qlabs = [letters(ANG, COH, WS, BR, inner & qm) for qm in QUADS if (inner & qm).sum() >= 100]
    lab = letters(ANG, COH, WS, BR, centre); res = classify(lab, kA, kB, qlabs)
    nq = len(qlabs); agree = sum(classify(q_, kA, kB, qlabs)[0] == res[0] for q_ in qlabs)
    return dict(_cut=CT[:, :, R].astype(np.float16), _kA=kA,                  # a cut across the sheets at the spot, for the check picture
                label=res[0], reason=res[1], depth_B=round(dB, 1) if dB is not None else np.nan, kB=kB,
                ply_A=res[2] or "", ply_B=res[3] or "", between=res[4], **res[5], letters="".join(lab),
                any_V=("V" in lab), quarters_agree=round(agree / nq, 2) if nq else np.nan, cover=round(fs["cover"], 2), scatter=round(fs["resid"], 2))

# ------------------------------------------------------------------ choose spots: per pair, samples spread apart, not at the chunk wall
rng = np.random.default_rng(a.seed); order = np.argsort(pidx, kind="stable"); groups = np.split(order, np.flatnonzero(np.diff(pidx[order])) + 1)
groups = [g for g in groups if len(g) >= a.min_n]; groups.sort(key=len, reverse=True)
if a.limit: groups = groups[:a.limit]
print(f"  {len(groups):,} pairs with {a.min_n}+ samples, up to {a.per_pair} spots each")
rows = []; t1 = time.time()
for gi, g in enumerate(groups):
    g = g[(scls[g] != 5) & (sp_gap[g] <= 30)]                                  # 5 = EDGE in phase6b_pair_table.py
    g = g[rng.permutation(len(g))]; chosen = []
    for i in g:
        if all(np.linalg.norm(sp_p[i] - sp_p[j]) >= R for j in chosen): chosen.append(i)
        if len(chosen) >= a.per_pair: break
    for i in chosen:
        r = read_spot(sp_p[i], int(sA[i]), sp_gap[i], int(sB[i]))
        base = dict(pair=int(pidx[i]), sheet_a=int(sA[i]), sheet_b=int(sB[i]), z=int(sp_p[i][0]), y=int(sp_p[i][1]), x=int(sp_p[i][2]), gap=round(float(sp_gap[i]), 1),
                    pair_table_label={-1: "unknown", 0: "same layer", 1: "two layers", 2: "conflict"}[int(plab[i])])
        rows.append({**base, **(r if r else dict(label="unreadable", reason="too little of sheet A around the spot"))})
    if (gi + 1) % 100 == 0: print(f"    {gi + 1:,} of {len(groups):,} pairs | {time.time() - t1:.0f}s")
SP = pd.DataFrame(rows); CUTS = SP.pop("_cut") if "_cut" in SP else None; KA = SP.pop("_kA") if "_kA" in SP else None
SP.to_csv(os.path.join(a.out, "pair_spots.csv"), index=False)

# ------------------------------------------------------------------ one label per pair
def combine(ls):
    ok = [l for l in ls if l != "unreadable"]
    if not ok: return "unreadable"
    fam = {"same layer": "same", "same layer, delaminated": "same", "same ply": "same", "possibly delaminated": "same",
           "different layers": "diff", "likely different layers": "diff"}
    weak = {"likely different layers", "possibly delaminated"}
    strong = [l for l in ok if l not in weak]; use = strong if strong else ok      # a weak reading gives way to a strong one
    if len({fam[l] for l in use}) > 1: return "conflict"
    if fam[use[0]] == "diff": return "different layers" if "different layers" in use else "likely different layers"
    for l in ("same layer, delaminated", "same layer", "same ply", "possibly delaminated"):
        if l in use: return l
PL = SP.groupby("pair").agg(sheet_a=("sheet_a", "first"), sheet_b=("sheet_b", "first"), spots=("label", "size"),
                            readable=("label", lambda s: int((s != "unreadable").sum())), label=("label", lambda s: combine(list(s))),
                            any_V=("any_V", lambda s: bool(pd.Series(s).fillna(False).astype(bool).any())), quarters_agree=("quarters_agree", "mean"),
                            pair_table_label=("pair_table_label", "first"), z=("z", "first"), y=("y", "first"), x=("x", "first")).reset_index()
lo, hi = np.minimum(PL["sheet_a"], PL["sheet_b"]), np.maximum(PL["sheet_a"], PL["sheet_b"]); PL["sheet_a"], PL["sheet_b"] = lo, hi
if a.states and os.path.exists(a.states):
    st = pd.read_csv(a.states); key = {(min(r.sheet_a, r.sheet_b), max(r.sheet_a, r.sheet_b)): r.state for r in st.itertuples()}
    PL["state"] = [key.get((x_, y_), "") for x_, y_ in zip(PL["sheet_a"], PL["sheet_b"])]
PL.to_csv(os.path.join(a.out, "pair_labels.csv"), index=False)

print(f"\n[SPOTS] {len(SP):,} spots read in {time.time() - t1:.0f}s")
for l, n in SP["label"].value_counts().items(): print(f"  {l:<26} {n:>6,} ({n / len(SP):.0%})")
if "depth_B" in SP:
    dz = (SP["depth_B"] - SP["gap"]).abs().dropna()
    if len(dz): print(f"  outer sheet found from its own tracks at {len(dz):,} spots | differs from the pair-table gap by 3+ voxels at {(dz >= 3).mean():.0%}, 6+ at {(dz >= 6).mean():.0%}")
print("  why spots are unreadable:")
for l, n in SP.loc[SP["label"] == "unreadable", "reason"].value_counts().items(): print(f"    {l:<52} {n:>6,}")
if "bandA" in SP:
    bb = pd.concat([SP[["bandA", "thickA"]].set_axis(["band", "thick"], axis=1), SP[["bandB", "thickB"]].set_axis(["band", "thick"], axis=1)]).dropna()
    bb = bb[bb["band"].isin(["H", "V", "HV", "VH", "HVH", "VHV", "HVHV"])]
    if len(bb):
        print("  band thickness by what the band holds (voxels; one ply should be about half a layer):")
        for k_, g_ in bb.groupby("band"): print(f"    {k_:<6} {len(g_):>5,} bands | median {g_['thick'].median():.0f} | middle half {g_['thick'].quantile(0.25):.0f}-{g_['thick'].quantile(0.75):.0f}")
if "layers_A" in SP:
    lc = pd.concat([SP["layers_A"], SP["layers_B"]]).dropna()
    if len(lc):
        print("  layers per band, counted from plies (0.5 = a single ply, 1 = one layer, 1.5 = a layer plus a ply, ...):")
        print("    " + " | ".join(f"{k_:g}: {v_:,}" for k_, v_ in lc.value_counts().sort_index().items()))
    dbl = pd.concat([SP["double_A"], SP["double_B"]]); dbl = dbl[dbl.fillna("") != ""].str.split(" \\(").str[0]
    print(f"  doubled plies in the sheets' bands: {len(dbl):,}" + ("" if not len(dbl) else " | " + " | ".join(f"{k_}: {v_:,}" for k_, v_ in dbl.value_counts().items())))
    print("    thickening, local = seen in fewer than half the quarters (a lump, overlap or frayed fibre in the papyrus)")
    print("    fold? = the same faces meet over a wide area, with the other ply on both sides | thick ply = doubled, at the edge of a band")
rd = SP[SP["label"] != "unreadable"]
if len(rd): print(f"  readable spots whose four quarters agree: typical {rd['quarters_agree'].median():.0%} | spots with no V anywhere in the window: {(~SP['any_V'].fillna(False).astype(bool)).mean():.0%}")
print(f"\n[PAIRS] {len(PL):,} pairs")
for l, n in PL["label"].value_counts().items(): print(f"  {l:<26} {n:>6,} ({n / len(PL):.0%})")
print("\n  against the pair table's own label (rows: ply reading | columns: pair table):")
print(pd.crosstab(PL["label"], PL["pair_table_label"]).to_string())
if "state" in PL:
    print("\n  against layer_state.py (rows: ply reading | columns: layer_state):")
    print(pd.crosstab(PL["label"], PL["state"]).to_string())
print("\n  note: 'pair table' and 'layer_state' are the old, rule-of-thumb answers. Where they disagree with the plies,")
print("        the plies are the measured one; check a few disagreements with ply_count.py before trusting either.")

# ------------------------------------------------------------------ picture: the labels on the middle slice
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    zc = int(shape[0] // 2) if a.z < 0 else a.z; fig, ax = plt.subplots(1, 2, figsize=(26, 12), gridspec_kw=dict(width_ratios=[2, 1]))
    ax[0].imshow(bright[zc], cmap="gray", vmin=0, vmax=1)
    COL = {"same layer": "cyan", "same ply": "royalblue", "same layer, delaminated": "lime", "possibly delaminated": "yellowgreen", "different layers": "red", "likely different layers": "salmon", "unreadable": "grey"}
    near = SP[np.abs(SP["z"] - zc) <= a.show_z]                 # only spots on (or very near) this slice, so dots sit on their sheet
    for l, c in COL.items():
        m = near["label"] == l
        if m.any(): ax[0].scatter(near.loc[m, "x"], near.loc[m, "y"], s=22, color=c, label=f"{l} ({int(m.sum())})", edgecolors="k", linewidths=0.3)
    ax[0].set_title(f"{TILE}: ply reading at each spot within {a.show_z} slices of z = {zc} (dot = spot on the inner sheet A)"); ax[0].legend(fontsize=9, facecolor="black", labelcolor="white")
    vc = PL["label"].value_counts(); ax[1].bar(vc.index, vc.values, color=[COL.get(l, "orange") for l in vc.index]); ax[1].set_title("pairs by ply reading")
    ax[1].tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.savefig(os.path.join(a.diag, f"ply_labels_{TILE}.png"), dpi=90); plt.close(); print(f"\n  wrote {a.diag}/ply_labels_{TILE}.png")
except Exception as ex:
    print(f"  picture skipped: {ex}")

# ------------------------------------------------------------------ check picture: what each label looks like in the CT
# each card is a cut straight across the sheets at one spot: depth runs down (inner at the top, outer at the bottom),
# along-the-sheet runs across. Cyan line = sheet A, orange line = sheet B, letters down the right = the ply reading per depth.
try:
    if CUTS is not None:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        rng2 = np.random.default_rng(a.seed + 1); show = ["different layers", "likely different layers", "same layer", "same layer, delaminated", "same ply", "unreadable"]
        nc = a.cards; fig, axs = plt.subplots(len(show), nc, figsize=(2.6 * nc, 4.2 * len(show)), squeeze=False)
        for r_, l in enumerate(show):
            ids = np.flatnonzero((SP["label"] == l).to_numpy() & CUTS.notna().to_numpy())
            ids = rng2.choice(ids, min(nc, len(ids)), replace=False) if len(ids) else []
            for c_ in range(nc):
                axc = axs[r_, c_]; axc.axis("off")
                if c_ >= len(ids): continue
                i = ids[c_]; cut = np.asarray(CUTS.iloc[i], dtype=np.float32); kA_, kB_ = int(KA.iloc[i]), int(SP["kB"].iloc[i])
                axc.imshow(cut, cmap="gray", aspect="auto"); axc.axhline(kA_, color="cyan", lw=1); axc.axhline(kB_, color="orange", lw=1)
                axc.axvline(R - a.centre, color="yellow", lw=0.5, ls=":"); axc.axvline(R + a.centre, color="yellow", lw=0.5, ls=":")
                lt = str(SP["letters"].iloc[i])
                for k_, ch in enumerate(lt): axc.text(cut.shape[1] + 1, k_, ch, fontsize=5, va="center", color={"H": "red", "V": "blue"}.get(ch, "grey"))
                axc.set_title(f"{l}\npair {SP['pair'].iloc[i]} z{SP['z'].iloc[i]} y{SP['y'].iloc[i]} x{SP['x'].iloc[i]}\ngap {SP['gap'].iloc[i]}", fontsize=7)
        plt.suptitle(f"{TILE}: cut across the sheets at each spot | cyan = sheet A, orange = sheet B, yellow dotted = centre window | letters: H red, V blue", fontsize=10)
        plt.tight_layout(); plt.savefig(os.path.join(a.diag, f"ply_label_cards_{TILE}.png"), dpi=110); plt.close(); print(f"  wrote {a.diag}/ply_label_cards_{TILE}.png")
except Exception as ex:
    print(f"  check picture skipped: {ex}")
print(f"  wrote {a.out}/pair_spots.csv, {a.out}/pair_labels.csv | {time.time() - t0:.0f}s")
