
"""compare_surfaces.py - a separate checking tool (not part of the pipeline): does each surface follow a real sheet of papyrus in
the CT? OUR surface and the OFFICIAL surface are put through the same tests, over the same papyrus, with the raw CT scan as the
only judge. No human ink labels, and nothing that rewards how our pipeline happens to place its surface.

    python compare_surfaces.py --id 9011
    python compare_surfaces.py --id 9011 --official_dir /content/drive/MyDrive/TechNexus/Vesuvius/Python

Neutral by design
  - Where a surface sits WITHIN its sheet is not scored. A surface on the face of the sheet (as the official surfaces are traced)
    and a surface in the middle of the sheet (as ours is) both count as following the sheet. Only whether the surface is on or
    against papyrus, and whether it stays on the same sheet, is measured.
  - The judge is the raw CT (vol_f16), with its own papyrus threshold from the scan itself (Otsu), not the brightness measure our
    tracker followed. Every test is repeated with the threshold 10% lower and 10% higher; a verdict is only given when all three
    agree, otherwise it says "depends on the threshold".
  - Distances are in voxels for both surfaces (each grid's own point spacing is measured and used), and only the papyrus both
    surfaces cover is compared. How much each covers is reported separately.

The tests (identical for both surfaces)
  1 on or against papyrus   share of points with papyrus within --face voxels along the surface's normal       higher is better
    (the rest run through a dark gap: more than --face voxels from any papyrus)
  2 layer crossings         neighbouring points about --pair voxels apart: does the surface cross a real dark gap (at least
                            --min_dark voxels) onto a different sheet?  per 100 neighbour pairs               fewer is better
  3 stays on the same sheet of the same neighbour pairs, the share that stay on the same sheet of papyrus     higher is better
  4 ink agreement           where both surfaces lie on the same sheet: how closely the two ink predictions match.
                            Reported only, not ranked.

Needs  in the working folder: wrap_<id>_V3.tifxyz, wrap_<id>_mask_V3.tif (and wrap_<id>_ink/prediction.tif for test 4),
           <ckpt>/manifest.json and vol_f16.npy
       in --official_dir: wrap_<official>_V3.tifxyz, wrap_<official>_mask_V3.tif (and wrap_<official>_ink/prediction.tif for test 4)
Writes diagnostics/surfaces_<id>_vs_<official>.png and .json
"""
import argparse, json, os, sys
import numpy as np, tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--id", type=int, required=True, help="our run's id (from run_ink.py)")
ap.add_argument("--ckpt", default="ckpt_t1", help="stage-1 folder our mesh was built from (its raw CT is the judge)")
ap.add_argument("--official", type=int, default=9961, help="id of the official surface grid (9961 = tile 1, from cut_official.py)")
ap.add_argument("--official_dir", default=".", help="folder holding the official surface files; default: the working folder")
ap.add_argument("--official_offset", type=int, nargs=3, default=[15534, 4794, 3895], metavar=("Z", "Y", "X"), help="corner (z y x) of the box the official grid was cut for (tile 1)")
ap.add_argument("--reach", type=float, default=10.0, help="same area: points of the two surfaces closer than this (voxels, about one layer)")
ap.add_argument("--same_sheet", type=float, default=3.0, help="for the ink test: the two surfaces count as on the same sheet when closer than this")
ap.add_argument("--face", type=float, default=2.0, help="a point counts as on or against papyrus when papyrus is within this many voxels along its normal")
ap.add_argument("--min_dark", type=float, default=3.0, help="a dark stretch at least this long (voxels) is a real gap between sheets")
ap.add_argument("--pair", type=float, default=4.0, help="neighbour pairs for tests 2 and 3 are about this many voxels apart along the surface")
ap.add_argument("--out", default="diagnostics")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
od = lambda name: os.path.join(a.official_dir, name)

# ------------------------------------------------------------------ load
need = [f"wrap_{a.id}_V3.tifxyz", f"wrap_{a.id}_mask_V3.tif", od(f"wrap_{a.official}_V3.tifxyz"), od(f"wrap_{a.official}_mask_V3.tif"),
        os.path.join(a.ckpt, "manifest.json"), os.path.join(a.ckpt, "vol_f16.npy")]
miss = [f for f in need if not os.path.exists(f)]
if miss: sys.exit("Missing: " + ", ".join(miss) + f"  (working folder {os.getcwd()}; official files looked for in {a.official_dir}, set with --official_dir)")
man = json.load(open(os.path.join(a.ckpt, "manifest.json"))); ck_off = np.array(man["chunk"]["offset_zyx"], dtype=np.float64)
vol = np.asarray(np.load(os.path.join(a.ckpt, "vol_f16.npy")), dtype=np.float32); shape = np.array(vol.shape)

def otsu(x):
    h, e = np.histogram(x, bins=256); c = (e[:-1] + e[1:]) / 2; w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = ((h * c).sum() - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])
T0 = otsu(vol[::2, ::2, ::2].ravel())
JUDGES = [("raw CT, threshold from the scan (Otsu)", T0), ("raw CT, threshold 10% lower", 0.9 * T0), ("raw CT, threshold 10% higher", 1.1 * T0)]

def load_grid(name, xyz_f, mask_f, offset_zyx):
    """grid (x, y, z) -> points (z, y, x) in the ckpt's local voxels, mask, unit normals, point spacing (voxels per pixel)"""
    g = tifffile.imread(xyz_f).astype(np.float64); m = tifffile.imread(mask_f) > 0
    if g.shape[:2] != m.shape: sys.exit(f"{xyz_f} {g.shape[:2]} and {mask_f} {m.shape} differ in size")
    m &= np.all(np.isfinite(g), axis=-1) & (g[..., 0] >= 0)
    P = g[..., ::-1] + (np.asarray(offset_zyx, dtype=np.float64) - ck_off)
    w = ndi.gaussian_filter(m.astype(float), 2.0); Ps = np.stack([ndi.gaussian_filter(np.where(m, P[..., k], 0), 2.0) for k in range(3)], -1) / np.maximum(w, 1e-6)[..., None]
    du = np.gradient(Ps, axis=1); dv = np.gradient(Ps, axis=0); n = np.cross(du, dv); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    h = np.linalg.norm(P[:, 1:] - P[:, :-1], axis=-1)[m[:, 1:] & m[:, :-1]]; v = np.linalg.norm(P[1:] - P[:-1], axis=-1)[m[1:] & m[:-1]]
    pitch = float(np.median(np.r_[h, v])) if len(h) + len(v) else 1.0
    inside = np.all((P >= 0) & (P <= shape - 1), axis=-1); mm = m & inside & np.all(np.isfinite(n), axis=-1)
    idx = np.argwhere(mm)
    return dict(name=name, P=P, m=mm, n=n, all=int(m.sum()), pitch=pitch, idx=idx, pts=P[mm], gap=max(1, int(round(a.pair / pitch))))

ours = load_grid("ours", f"wrap_{a.id}_V3.tifxyz", f"wrap_{a.id}_mask_V3.tif", ck_off)
offi = load_grid("official", od(f"wrap_{a.official}_V3.tifxyz"), od(f"wrap_{a.official}_mask_V3.tif"), a.official_offset)
if not len(ours["pts"]) or not len(offi["pts"]): sys.exit("one of the surfaces has no points inside this ckpt's box; check --ckpt and --official_offset")

# ------------------------------------------------------------------ same area
for S, O in ((ours, offi), (offi, ours)):
    d, k = cKDTree(O["pts"]).query(S["pts"], distance_upper_bound=a.reach)
    S["over"] = np.isfinite(d); S["dist"] = np.where(np.isfinite(d), d, np.nan); S["near_k"] = np.where(np.isfinite(d), k, -1)
    S["over_img"] = np.zeros(S["m"].shape, bool); S["over_img"][tuple(S["idx"][S["over"]].T)] = True
print(f"[COMPARE SURFACES] ours wrap {a.id} vs official wrap {a.official} | judge: the raw CT of {a.ckpt} (papyrus threshold {T0:.3f} from the scan)")
print(f"  point spacing: ours {ours['pitch']:.2f} voxels, official {offi['pitch']:.2f} voxels | neighbour pairs use {ours['gap']} and {offi['gap']} pixels (about {a.pair:g} voxels)")
print(f"  coverage inside this box: ours {len(ours['pts']):,} points, official {len(offi['pts']):,} | same area (within {a.reach:g} voxels of the other): "
      f"ours {ours['over'].sum():,} ({ours['over'].mean():.0%}), official {offi['over'].sum():,} ({offi['over'].mean():.0%})")

# ------------------------------------------------------------------ tests 1-3
STEP = 0.5; R = 8.0; KS = np.arange(-R, R + STEP / 2, STEP); J0 = int(round(R / STEP)); JJ = np.arange(len(KS))[None, :]
FACE = int(round(a.face / STEP)); need_dark = max(1, int(round(a.min_dark / STEP)))
def tl(v3, q):
    q = np.clip(q, 0, shape - 1 - 1e-6); i_ = np.floor(q).astype(int); f = (q - i_).astype(np.float32); z, y, x = i_.T
    c = lambda dz, dy, dx: v3[z + dz, y + dy, x + dx]
    c00 = c(0, 0, 0) * (1 - f[:, 2]) + c(0, 0, 1) * f[:, 2]; c01 = c(0, 1, 0) * (1 - f[:, 2]) + c(0, 1, 1) * f[:, 2]
    c10 = c(1, 0, 0) * (1 - f[:, 2]) + c(1, 0, 1) * f[:, 2]; c11 = c(1, 1, 0) * (1 - f[:, 2]) + c(1, 1, 1) * f[:, 2]
    return (c00 * (1 - f[:, 1]) + c01 * f[:, 1]) * (1 - f[:, 0]) + (c10 * (1 - f[:, 1]) + c11 * f[:, 1]) * f[:, 0]
def raw_profiles(P, n):
    """raw CT along the normal, from -R to +R voxels, for every point (nan outside the box)"""
    out = np.full((len(P), len(KS)), np.nan, np.float32)
    for s in range(0, len(P), 20000):
        q = P[s:s + 20000, None, :] + KS[None, :, None] * n[s:s + 20000, None, :]
        ok = np.all((q >= 0) & (q <= shape - 1), axis=-1); v = tl(vol, q.reshape(-1, 3)).reshape(-1, len(KS))
        out[s:s + 20000] = np.where(ok, v, np.nan)
    return out
for S in (ours, offi):
    S["sel"] = np.flatnonzero(S["over"]); S["n_sel"] = S["n"][tuple(S["idx"][S["sel"]].T)]
    S["prof"] = raw_profiles(S["pts"][S["sel"]], S["n_sel"])
    pos = -np.ones(S["m"].shape, np.int64); pos[tuple(S["idx"][S["sel"]].T)] = np.arange(len(S["sel"])); pairs = []
    for dy, dx in ((0, S["gap"]), (S["gap"], 0)):
        p_ = pos[:pos.shape[0] - dy, :pos.shape[1] - dx]; q_ = pos[dy:, dx:]; k = (p_ >= 0) & (q_ >= 0); pairs.append(np.c_[p_[k], q_[k]])
    S["pairs"] = np.concatenate(pairs)

def tests(S, thr):
    B = S["prof"] >= thr                                                  # papyrus along the normal (nan counts as not papyrus)
    against = B[:, J0 - FACE:J0 + FACE + 1].any(1)                        # papyrus within --face voxels: on the sheet or against its face
    # the sheet each point belongs to: the papyrus run nearest to it, if within --face voxels
    dist = np.where(B, np.abs(JJ - J0), 10 ** 6); j0 = dist.argmin(1)
    lo = KS[np.clip(np.where(~B & (JJ <= j0[:, None]), JJ, -1).max(1) + 1, 0, len(KS) - 1)]
    hi = KS[np.clip(np.where(~B & (JJ >= j0[:, None]), JJ, len(KS)).min(1) - 1, 0, len(KS) - 1)]
    pr = S["pairs"]; pr = pr[against[pr[:, 0]] & against[pr[:, 1]]]      # both ends on or against papyrus
    r = dict(points=int(len(B)), against=float(against.mean()), through_gap=float(1 - against.mean()))
    if not len(pr):
        r.update(pairs=0, same_sheet=float("nan"), crossings_per100=float("nan"), unclear=float("nan")); r["_maps"] = (against, None); return r
    p, q = pr[:, 0], pr[:, 1]; t = np.einsum("ij,ij->i", S["pts"][S["sel"]][q] - S["pts"][S["sel"]][p], S["n_sel"][p])
    same = (t >= lo[p] - a.face) & (t <= hi[p] + a.face)                 # the neighbour is on (or against) the same run of papyrus
    dark = ~B[p]; c = np.cumsum(dark, 1); run = c - np.maximum.accumulate(np.where(~dark, c, 0), 1)
    beyond_up = (KS[None, :] > hi[p][:, None]) & (KS[None, :] <= t[:, None]); beyond_dn = (KS[None, :] < lo[p][:, None]) & (KS[None, :] >= t[:, None])
    crossed = ~same & (((np.where(beyond_up | beyond_dn, run, 0) >= need_dark).any(1)) | (np.abs(t) > R))
    r.update(pairs=int(len(pr)), same_sheet=float(same.mean()), crossings_per100=float(100 * crossed.mean()), unclear=float((~same & ~crossed).mean()))
    cross_pt = np.zeros(len(B), bool); np.logical_or.at(cross_pt, p[crossed], True); r["_maps"] = (against, cross_pt)
    return r

RES = {j: {S["name"]: tests(S, thr) for S in (ours, offi)} for j, thr in JUDGES}
ROWS = [("1 on or against papyrus (share of points)", "against", True, 0.02, "{:.1%}"),
        ("  runs through a dark gap (share of points)", "through_gap", False, 0.02, "{:.1%}"),
        ("2 layer crossings per 100 neighbour pairs", "crossings_per100", False, 0.3, "{:.2f}"),
        ("3 stays on the same sheet (share of neighbour pairs)", "same_sheet", True, 0.02, "{:.1%}")]
def one_verdict(o, f, higher, tol):
    if not (np.isfinite(o) and np.isfinite(f)): return "n/a"
    if abs(o - f) <= tol: return "about equal"
    return "OURS better" if (o > f) == higher else "OFFICIAL better"
summary = {}
for j, thr in JUDGES:
    o, f = RES[j]["ours"], RES[j]["official"]
    print(f"\n  judged on the {j} = {thr:.3f} | points: ours {o['points']:,}, official {f['points']:,} | neighbour pairs: ours {o['pairs']:,}, official {f['pairs']:,}")
    print(f"    {'test':<56} {'ours':>9} {'official':>9}   result")
    for label, key, higher, tol, fmt in ROWS:
        print(f"    {label:<56} {fmt.format(o[key]):>9} {fmt.format(f[key]):>9}   {one_verdict(o[key], f[key], higher, tol)}")
    print(f"    {'  left the sheet without a clear dark gap (share of pairs)':<56} {o['unclear']:>9.1%} {f['unclear']:>9.1%}   (reported only)")
print("\n  VERDICT (given only where all three thresholds agree)")
for label, key, higher, tol, fmt in ROWS:
    vs = [one_verdict(RES[j]["ours"][key], RES[j]["official"][key], higher, tol) for j, _ in JUDGES]
    final = vs[0] if len(set(vs)) == 1 else "depends on the threshold"
    summary[label.strip()] = dict(ours=[RES[j]["ours"][key] for j, _ in JUDGES], official=[RES[j]["official"][key] for j, _ in JUDGES], verdict=final)
    print(f"    {label:<56} {final}")

# ------------------------------------------------------------------ test 4: ink agreement where both lie on the same sheet
ink_o, ink_f = f"wrap_{a.id}_ink/prediction.tif", od(f"wrap_{a.official}_ink/prediction.tif"); INK = None
def read_pred(f, shp):
    p = np.squeeze(tifffile.imread(f)).astype(np.float32); p = p / 255.0 if p.max() > 1.5 else p
    return p if p.shape == shp else None
if os.path.exists(ink_o) and os.path.exists(ink_f):
    po, pf = read_pred(ink_o, ours["m"].shape), read_pred(ink_f, offi["m"].shape)
    if po is None or pf is None: print("\n  4 ink agreement: skipped (a prediction does not match its grid's size)")
    else:
        same = ours["over"] & (ours["dist"] <= a.same_sheet); k = ours["near_k"][same]
        vo = po[tuple(ours["idx"][same].T)]; vf = pf[tuple(offi["idx"][k].T)]
        if len(vo) > 200:
            agree = float(((vo > 0.5) == (vf > 0.5)).mean()); corr = float(np.corrcoef(vo, vf)[0, 1])
            INK = dict(pixels=int(len(vo)), agree=agree, correlation=corr, ours_ink=float((vo > 0.5).mean()), official_ink=float((vf > 0.5).mean()), _map=(same, vo, vf))
            print(f"\n  4 ink agreement where both lie on the same sheet ({len(vo):,} pixels): both say ink / both say blank on {agree:.0%} | correlation {corr:+.2f} | "
                  f"marked ink: ours {INK['ours_ink']:.0%}, official {INK['official_ink']:.0%}   (reported only, not ranked)")
        else: print("\n  4 ink agreement: too few pixels where the two surfaces lie on the same sheet")
else:
    print(f"\n  4 ink agreement: skipped; needs {ink_o} and {ink_f} (the ink steps run on the official grid too).")
print("\n  'Better' here means: follows a real sheet of papyrus in the scan more closely. Where in the sheet a surface sits is not scored.")

# ------------------------------------------------------------------ pictures and record
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    j0name = JUDGES[0][0]; rows = 3 if INK else 2; fig, ax = plt.subplots(rows, 2, figsize=(22, 6.5 * rows), squeeze=False)
    for col, S in enumerate((ours, offi)):
        against, cross = RES[j0name][S["name"]]["_maps"]; img = np.zeros(S["m"].shape + (3,)); img[S["m"]] = 0.3
        c = np.zeros((len(S["sel"]), 3)); c[against] = (0.1, 0.75, 0.2); c[~against] = (0.85, 0.1, 0.1)
        if cross is not None: c[cross] = (1.0, 0.9, 0.0)
        img[tuple(S["idx"][S["sel"]].T)] = c; ax[0, col].imshow(img); ax[0, col].set_xticks([]); ax[0, col].set_yticks([])
        ax[0, col].set_title(f"{S['name']}: green = on or against papyrus, red = runs through a dark gap, yellow = crosses onto another sheet, grey = outside the shared area", fontsize=8)
    zc = int(np.median(ours["pts"][ours["over"], 0])) if ours["over"].any() else int(shape[0] // 2)
    ax[1, 0].imshow(vol[zc], cmap="gray")
    for S, colr in ((offi, "cyan"), (ours, "red")):
        k = np.abs(S["pts"][:, 0] - zc) <= 0.6; ax[1, 0].plot(S["pts"][k, 2], S["pts"][k, 1], ".", ms=1.5, color=colr, label=S["name"])
    ax[1, 0].legend(markerscale=6); ax[1, 0].set_title(f"raw CT slice z = {zc} (local): red = ours, cyan = official"); ax[1, 0].set_xticks([]); ax[1, 0].set_yticks([])
    o, f = RES[j0name]["ours"], RES[j0name]["official"]; xs = np.arange(2)
    ax[1, 1].bar(xs - 0.2, [o["against"], o["same_sheet"]], 0.4, color="tab:red", label="ours"); ax[1, 1].bar(xs + 0.2, [f["against"], f["same_sheet"]], 0.4, color="tab:cyan", label="official")
    ax[1, 1].set_xticks(xs); ax[1, 1].set_xticklabels(["on or against papyrus", "stays on the same sheet"]); ax[1, 1].set_ylim(0, 1); ax[1, 1].legend()
    ax[1, 1].set_title(f"shares on the shared area ({j0name})")
    if INK:
        same, vo, vf = INK["_map"]; A = np.full(ours["m"].shape, np.nan); B_ = np.full(ours["m"].shape, np.nan); ii = tuple(ours["idx"][same].T)
        A[ii] = vo; B_[ii] = vf
        for x, im, t in ((ax[2, 0], A, "our ink, where both surfaces lie on the same sheet"), (ax[2, 1], B_, "official surface's ink at the same papyrus, in our layout")):
            x.imshow(im, cmap="gray", vmin=0.2, vmax=0.9); x.set_title(t); x.set_xticks([]); x.set_yticks([])
    plt.tight_layout(); png = os.path.join(a.out, f"surfaces_{a.id}_vs_{a.official}.png"); plt.savefig(png, dpi=90, bbox_inches="tight"); plt.close(); print(f"  wrote {png}")
except Exception as ex:
    print(f"  (picture skipped: {ex})")
rec = dict(ours=a.id, official=a.official, ckpt=a.ckpt, settings=vars(a), thresholds={j: thr for j, thr in JUDGES},
           pitch=dict(ours=ours["pitch"], official=offi["pitch"]),
           coverage=dict(ours=len(ours["pts"]), official=len(offi["pts"]), ours_shared=int(ours["over"].sum()), official_shared=int(offi["over"].sum())),
           results={j: {s: {k: v for k, v in r.items() if not k.startswith("_")} for s, r in RES[j].items()} for j in RES}, verdicts=summary,
           ink={k: v for k, v in INK.items() if not k.startswith("_")} if INK else None)
json.dump(rec, open(os.path.join(a.out, f"surfaces_{a.id}_vs_{a.official}.json"), "w"), indent=1); print(f"  wrote {os.path.join(a.out, f'surfaces_{a.id}_vs_{a.official}.json')}")
