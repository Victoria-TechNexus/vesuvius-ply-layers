
"""run_ink.py - phases 8, 9 and 10 for ONE layer's mesh, in one command: from the mesh to an ink picture you can look at.
(Was read_ink.py.)

Runs, in order, each as its own script (each can still be run on its own):
    8   phase8_flat_grid.py                 the mesh from phase 7b -> a flat grid       wrap_<id>_V3.tifxyz, wrap_<id>_mask_V3.tif
    9   phase9_ink_volume_extraction.py     21 slices through the sheet along the grid  wrap_<id>_slices/
        (then everything outside the sheet is filled with the sheet's average grey: the ink model misbehaves next to black)
    10  INK_SCRIPT (phase10_ink.py)         the ink model on those slices                wrap_<id>_ink/prediction.tif
Writes diagnostics/ink_<id>.png: the middle slice on top, the ink prediction underneath, same size, for judging by eye.

    python run_ink.py                                              # the layer in SETTINGS below
    python run_ink.py --name layer11 --id 9011                     # layer 11 of tile 1 (mesh_t1_layer11/wraps)
    python run_ink.py --name layer11 --id 9011 --ckpt mesh_t1-t2_layer11      # a layer combined over tiles by phase 7a
    python run_ink.py --name layer11 --id 9011 --step -1           # read the other face of the sheet
    python run_ink.py --name official --id 9971 --no_grid --offset 15382 4794 3895   # read an existing grid (e.g. the official surface)

Phase 10 is the step to swap for another ink model: set INK_SCRIPT and MODEL below (phase10_ink.py says what a replacement must do).
The mesh folder, the scan and the scroll centre come from config.py / the stage-1 folder's manifest.json unless given.
"""
import argparse, json, os, shutil, subprocess, sys
import numpy as np, tifffile
from config import scroll_centre

# =============================================================================================== SETTINGS
CKPT = "ckpt_t1"                      # stage-1 folder the mesh was built from (a tile's ckpt_<tile>, or a combined folder from phase 7a)
NAME = "layer11"                      # mesh name as phase 7b wrote it (layerNN)
ID = 9011                             # number for the output files (wrap_<id>_...)
WRAPS = None                          # folder holding the mesh; None = mesh_<tile>_layerNN/wraps from config.py
OUT = "diagnostics"                   # folder for the pictures

# ---- 8: flat grid
GRID = dict(
    fill_holes=400,                   # enclosed holes up to this many pixels are filled from their rim (0 = leave all holes)
    up="high_z",                      # which end of the scroll axis goes at the top: high_z = writing upright (checked on the official segment for PHerc. Paris 4)
)

# ---- 9: slices through the sheet
SHIFT = 5.0                           # where the 21 slices are centred, in voxels from the mesh along the normal
STEP = 1.0                            # spacing between slices (voxels); -1 reads the other face of the sheet
SNAP = True                           # move the grid onto the brightest papyrus within 6 voxels before slicing
FLIP = False                          # turn the normals around (read from the other side)

# ---- 10: ink model (the step to swap)
INK_SCRIPT = "phase10_ink.py"         # script that turns wrap_<id>_slices into wrap_<id>_ink/prediction.tif
MODEL = "step-075000.pth"             # model file it uses
DEVICE = "cpu"                        # cpu, cuda or mps
INK_EXTRA = []                        # extra arguments for INK_SCRIPT, e.g. ["--batch-size", "8"]
# =============================================================================================== end of SETTINGS

ap = argparse.ArgumentParser()
ap.add_argument("--name", default=NAME, help="mesh name as written by phase 7b, e.g. layer11")
ap.add_argument("--id", type=int, default=ID, help="number for the output files, e.g. 9011")
ap.add_argument("--shift", type=float, default=SHIFT); ap.add_argument("--step", type=float, default=STEP)
ap.add_argument("--model", default=MODEL); ap.add_argument("--device", default=DEVICE)
ap.add_argument("--core", nargs=2, type=float, default=None, metavar=("CX", "CY"), help="scroll centre; default: from the manifest / config.py")
ap.add_argument("--out", default=OUT)
ap.add_argument("--ckpt", default=CKPT, help="stage-1 folder the mesh was built from")
ap.add_argument("--wraps", default=WRAPS, help="folder holding the mesh; default: mesh_<tile>_layerNN/wraps from config.py")
ap.add_argument("--no_grid", action="store_true", help="skip phase 8 and read an existing wrap_<id>_V3.tifxyz (e.g. the official surface from cut_official.py)")
ap.add_argument("--offset", nargs=3, default=None, metavar=("Z", "Y", "X"), help="corner of the box the grid is relative to; overrides the --ckpt manifest")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); here = os.path.dirname(os.path.abspath(__file__))
man = json.load(open(os.path.join(a.ckpt, "manifest.json"))) if os.path.exists(os.path.join(a.ckpt, "manifest.json")) else {}
off = a.offset or [str(v) for v in man.get("chunk", {}).get("offset_zyx", [15534, 4794, 3895])]
core = [f"{v:g}" for v in scroll_centre(man, a.core)]
def run(cmd, quiet=False):
    print("  $ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, stdout=subprocess.PIPE if quiet else None, stderr=subprocess.STDOUT if quiet else None, text=True)
    if r.returncode != 0:
        if quiet and r.stdout: print(r.stdout[-3000:])
        sys.exit(f"step failed: {' '.join(cmd)}")
print(f"[RUN INK] {a.name} from {a.ckpt} -> id {a.id} | shift {a.shift:g}, step {a.step:g} | ink model {INK_SCRIPT} + {a.model}")
sl, ink = f"wrap_{a.id}_slices", f"wrap_{a.id}_ink"
shutil.rmtree(sl, ignore_errors=True); shutil.rmtree(ink, ignore_errors=True)
if not a.no_grid:
    print("\n[8] flat grid")
    run([sys.executable, os.path.join(here, "phase8_flat_grid.py"), "--name", a.name, "--id", str(a.id), "--ckpt", a.ckpt, "--core", *core,
         "--fill_holes", str(GRID["fill_holes"]), "--up", GRID["up"], "--out", a.out] + (["--wraps", a.wraps] if a.wraps else []))
print("\n[9] slices through the sheet")
run([sys.executable, os.path.join(here, "phase9_ink_volume_extraction.py"), str(a.id), "--ckpt", a.ckpt, "--core", *core, "--offset", *off,
     "--shift", str(a.shift), "--step", str(a.step)] + ([] if SNAP else ["--no-snap"]) + (["--flip"] if FLIP else []))
mask = tifffile.imread(f"wrap_{a.id}_mask_V3.tif") > 0; layers = sorted(f for f in os.listdir(os.path.join(sl, "layers")) if f.endswith(".tif"))
for f in layers:                                                  # grey fill outside the sheet
    p = os.path.join(sl, "layers", f); im = tifffile.imread(p); m = mask & (im > 0) if im.shape == mask.shape else im > 0
    tifffile.imwrite(p, np.where(m, im, int(im[m].mean()) if m.any() else 0).astype(im.dtype))
print(f"  filled outside the sheet with average grey in {len(layers)} slices")
print(f"\n[10] ink model ({INK_SCRIPT})")
run([sys.executable, os.path.join(here, INK_SCRIPT), sl, a.model, ink, "--device", a.device] + list(INK_EXTRA), quiet=True)
mid = tifffile.imread(os.path.join(sl, "layers", layers[len(layers) // 2])); pred = np.squeeze(tifffile.imread(os.path.join(ink, "prediction.tif"))).astype(np.float32)
if pred.max() > 1.5: pred /= 255.0
inside = mask if pred.shape == mask.shape else np.ones(pred.shape, bool)
print(f"  ink prediction on the sheet: p5/p50/p95 {np.percentile(pred[inside], [5, 50, 95]).round(2).tolist()} | above 0.5: {(pred[inside] > 0.5).mean():.0%}")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(20, 17))
ax[0].imshow(np.where(inside, mid, 0), cmap="gray"); ax[0].set_title(f"{a.name}: middle slice through the sheet (flattened)")
ax[1].imshow(np.where(inside, pred, 0), cmap="gray", vmin=0.2, vmax=0.9); ax[1].set_title(f"ink prediction (shift {a.shift:g}, step {a.step:g}); white = ink")
for x in ax: x.set_xticks([]); x.set_yticks([])
plt.tight_layout(); png = os.path.join(a.out, f"ink_{a.id}.png"); plt.savefig(png, dpi=110, bbox_inches="tight"); plt.close(); print(f"  wrote {png}")
