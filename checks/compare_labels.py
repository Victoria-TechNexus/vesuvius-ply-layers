
"""compare_labels.py - put the human ink labels of the official segment onto OUR flat image, to compare with our ink prediction by eye.

    python compare_labels.py --id 9011
    python compare_labels.py --id 9011 --official_dir /content/drive/MyDrive/TechNexus/Vesuvius/Python   # official files kept in another folder

For every pixel of our flat grid it finds the nearest point of the official segment's surface (within --reach voxels) and takes the
human label there. So the letters appear in our layout, at our orientation, wherever our layer really is the labelled one.

Needs, in --official_dir (default: the working folder):
    wrap_9961_V3.tifxyz, wrap_9961_mask_V3.tif   the official segment's surface inside this chunk (already in your folder)
    inklabels_9961.npy, supervision_9961.npy     human ink labels and the "checked by a human" map for that surface.
        If these two are not there, the script downloads them from the open-data bucket (about a minute) and keeps them.
and, in the working folder, from run_ink.py:  wrap_<id>_V3.tifxyz, wrap_<id>_mask_V3.tif, wrap_<id>_slices/, wrap_<id>_ink/prediction.tif
Writes diagnostics/labels_<id>.png
"""
import argparse, os, sys
import numpy as np, tifffile
from scipy.spatial import cKDTree
ap = argparse.ArgumentParser()
ap.add_argument("--id", type=int, required=True); ap.add_argument("--reach", type=float, default=6.0); ap.add_argument("--out", default="diagnostics")
ap.add_argument("--official", type=int, default=9961)
ap.add_argument("--official_dir", default=".", help="folder holding the official segment's files (wrap_9961_*, inklabels_9961.npy, supervision_9961.npy); default: the working folder")
ap.add_argument("--r0", type=int, default=628, help="row of the official grid where the 9961 crop starts (from the notebook)")
ap.add_argument("--c0", type=int, default=967, help="column of the official grid where the 9961 crop starts")
ap.add_argument("--ckpt", default="", help="the stage-1 or combined folder our mesh came from (e.g. combo11z). Its manifest gives our grid's corner in the scroll; without it our grid is assumed to share the official grid's corner (true for tile 1 only)")
ap.add_argument("--official_offset", type=int, nargs=3, default=[15534, 4794, 3895], metavar=("Z", "Y", "X"), help="corner (z y x) of the chunk the official surface file was cut for (tile 1)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
import shutil, subprocess
od = lambda name: os.path.join(a.official_dir, name)
surf_f, mask_f = od(f"wrap_{a.official}_V3.tifxyz"), od(f"wrap_{a.official}_mask_V3.tif")
ink_f, sup_f = od(f"inklabels_{a.official}.npy"), od(f"supervision_{a.official}.npy")
need = [surf_f, mask_f, f"wrap_{a.id}_V3.tifxyz", f"wrap_{a.id}_mask_V3.tif", f"wrap_{a.id}_ink/prediction.tif"]
miss = [f for f in need if not os.path.exists(f)]
if miss: sys.exit("Missing: " + ", ".join(miss) + f"  (working folder {os.getcwd()}; official files looked for in {a.official_dir}, set with --official_dir)")
om = tifffile.imread(mask_f) > 0; Ho, Wo = om.shape
for t, dst in (("inklabels", ink_f), ("supervision", sup_f)):
    if os.path.exists(dst): continue
    if os.path.exists(f"/tmp/{t}_{a.official}.npy"): shutil.copy(f"/tmp/{t}_{a.official}.npy", dst); continue
    print(f"  downloading {t} for the official segment ...", flush=True)
    if not os.path.exists("/tmp/zv3_pkgs/zarr"):
        subprocess.run([sys.executable, "-m", "pip", "install", "--target", "/tmp/zv3_pkgs", "zarr>=3", "s3fs", "-q"], check=True)
    open("/tmp/read_labels_worker.py", "w").write('''
import sys
sys.path.insert(0, "/tmp/zv3_pkgs")
import numpy as np, zarr, s3fs
target, level = sys.argv[1], int(sys.argv[2]); r0, r1, c0, c1 = map(int, sys.argv[3:7]); out = sys.argv[7]
path = f"vesuvius-challenge-open-data/PHercParis4/segments/20231016151002/ink-labels/2.4um-volume-20260411134726/20260918/{target}.zarr"
try:
    g = zarr.open_group("s3://" + path, mode="r", storage_options={"anon": True})
except Exception:
    g = zarr.open_group(store=zarr.storage.FsspecStore(s3fs.S3FileSystem(anon=True), path=path), mode="r")
arr = g[str(level)]; np.save(out, np.asarray(arr[max(r0, 0):min(r1, arr.shape[0]), max(c0, 0):min(c1, arr.shape[1])]))
''')
    r = subprocess.run([sys.executable, "/tmp/read_labels_worker.py", t, "2", str(a.r0 * 5), str(a.r0 * 5 + Ho), str(a.c0 * 5), str(a.c0 * 5 + Wo), dst], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(dst): sys.exit(f"could not download {t}:\n{r.stderr[-1500:]}")
og = tifffile.imread(surf_f).astype(np.float64); ink = np.load(ink_f) > 0; sup = np.load(sup_f) > 0
if ink.shape != om.shape: sys.exit(f"label image {ink.shape} does not match the official surface grid {om.shape}; check --r0/--c0 (delete {ink_f} and {sup_f} to download again)")
om &= np.all(np.isfinite(og), axis=-1) & (og[..., 0] >= 0); f = ink.shape[0] / Ho
if abs(f - round(f)) > 0.05 or abs(ink.shape[1] / Wo - f) > 0.05: sys.exit(f"label image {ink.shape} does not match the official grid {om.shape}")
f = int(round(f)); oi, oj = np.nonzero(om); tree = cKDTree(og[oi, oj])
blk = lambda im: im[:Ho * f, :Wo * f].reshape(Ho, f, Wo, f).mean(axis=(1, 3))           # label share per official surface point
ink_c, sup_c = blk(ink.astype(float)), blk(sup.astype(float))
g = tifffile.imread(need[2]).astype(np.float64); m = tifffile.imread(need[3]) > 0
if a.ckpt:                                                        # put our grid in the official grid's frame (grids hold x, y, z)
    import json
    man = json.load(open(os.path.join(a.ckpt, "manifest.json")))
    ours = np.array(man["chunk"]["offset_zyx"], dtype=np.float64); shift_zyx = ours - np.array(a.official_offset, dtype=np.float64)
    g = g + shift_zyx[::-1]
    print(f"  our grid moved by (z, y, x) {shift_zyx.astype(int).tolist()} into the official surface's frame")
pred = np.squeeze(tifffile.imread(need[4])).astype(np.float32); pred = pred / 255.0 if pred.max() > 1.5 else pred
d, k = tree.query(g[m], distance_upper_bound=a.reach); hit = np.isfinite(d)
INK = np.zeros(m.shape); SUP = np.zeros(m.shape); HIT = np.zeros(m.shape, bool); mi, mj = np.nonzero(m)
INK[mi[hit], mj[hit]] = ink_c[oi[k[hit]], oj[k[hit]]]; SUP[mi[hit], mj[hit]] = sup_c[oi[k[hit]], oj[k[hit]]]; HIT[mi[hit], mj[hit]] = True
print(f"[LABELS ON OUR IMAGE] wrap {a.id}: {int(m.sum()):,} pixels | official surface within {a.reach:g} voxels under {hit.mean():.1%} of them "
      f"(distance p50 {np.median(d[hit]) if hit.any() else float('nan'):.1f}) | human-checked: {(SUP[m] > 0.5).mean():.1%} | labelled ink: {(INK[m] > 0.5).mean():.1%}")
ev = m & (SUP > 0.5)
if ev.sum() > 200 and 0 < (INK[ev] > 0.5).mean() < 1:
    pi, pn = pred[ev & (INK > 0.5)], pred[ev & (INK <= 0.5)]
    print(f"  on human-checked pixels: our prediction averages {pi.mean():.2f} where the label says ink and {pn.mean():.2f} where it says no ink "
          f"| predicted ink (above 0.5) covers {(pi > 0.5).mean():.0%} of labelled ink and {(pn > 0.5).mean():.0%} of labelled blank")
    # chance check: slide the label map sideways / up and down by 60+ pixels and repeat. Smooth blobs can agree by luck.
    gap = float((pi > 0.5).mean() - (pn > 0.5).mean()); rs = np.random.default_rng(0); null = []
    for _ in range(300):
        dy, dx = rs.integers(-120, 121, 2)
        if abs(dy) < 60 and abs(dx) < 60: continue
        I2 = np.roll(INK, (dy, dx), (0, 1)); S2 = np.roll(SUP, (dy, dx), (0, 1)); e2 = m & (S2 > 0.5)
        a1, b1 = e2 & (I2 > 0.5), e2 & (I2 <= 0.5)
        if a1.sum() > 500 and b1.sum() > 500: null.append(float((pred[a1] > 0.5).mean() - (pred[b1] > 0.5).mean()))
    if len(null) >= 30:
        null = np.array(null)
        print(f"  chance check: 'ink covered minus blank covered' is {gap:+.2f} with the labels in place; with the labels slid out of place it is "
              f"{np.median(null):+.2f} typically, {np.percentile(null, 95):+.2f} at the 95th percentile ({len(null)} slides) -> "
              f"{'BEATS chance' if gap > np.percentile(null, 95) else 'NOT clearly better than chance'}")
    print("  (rough figures only: the official surface cuts corners, so judge the picture)")
else:
    print("  too little human-checked area under this layer to compare; this layer is probably not the labelled one. Try another --id.")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = [("our ink prediction (white = ink)", np.where(m, pred, 0), dict(cmap="gray", vmin=0.2, vmax=0.9))]
lab = np.zeros(m.shape + (3,)); lab[m] = 0.12; lab[HIT] = 0.25; lab[ev] = 0.45; lab[ev & (INK > 0.5)] = 1.0
rows.append(("human labels placed on our image: white = ink, mid grey = checked, no ink, dark grey = official surface here but unchecked, darkest = no official surface", lab, {}))
ov = np.zeros(m.shape + (3,)); ov[..., 0] = np.where(m, np.clip((pred - 0.25) / 0.5, 0, 1), 0); ov[..., 1] = np.where(ev, (INK > 0.5) * 1.0, 0)
rows.append(("overlay: red = our prediction, green = labelled ink, yellow = both", ov, {}))
sl = f"wrap_{a.id}_slices/layers"
if os.path.isdir(sl):
    fs = sorted(x for x in os.listdir(sl) if x.endswith(".tif")); rows.insert(0, ("middle slice through our sheet", np.where(m, tifffile.imread(os.path.join(sl, fs[len(fs) // 2])), 0), dict(cmap="gray")))
fig, ax = plt.subplots(len(rows), 1, figsize=(20, 8.2 * len(rows)))
for x, (t, im, kw) in zip(np.atleast_1d(ax), rows): x.imshow(im, **kw); x.set_title(t); x.set_xticks([]); x.set_yticks([])
plt.tight_layout(); png = os.path.join(a.out, f"labels_{a.id}.png"); plt.savefig(png, dpi=100, bbox_inches="tight"); plt.close(); print(f"  wrote {png}")
