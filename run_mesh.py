
"""run_mesh.py - phase 7 in one command: follow ONE layer (across one or more tiles) and build its mesh.

Runs, in order, each as its own script (each can still be run on its own):
    7a  phase7a_combine_tiles.py   only when more than one tile is listed: match the layer across the tiles and combine them
    7b  phase7b_mesh.py            build the layer's mesh

From phase 7 on, ONE layer is followed. The layer number is the one in the FIRST tile listed (layer numbers differ between tiles;
7a matches them through the overlap and prints the match).

    python run_mesh.py                          # the tiles and layer in SETTINGS below
    python run_mesh.py --tiles t1 --layer 11    # one tile: no combining, the mesh is built straight from ckpt_t1 / layers_t1
    python run_mesh.py --tiles t1 t2 t3 t4 t5 t6 --layer 11       # several tiles: combine first, then mesh
    python run_mesh.py --tiles t1 t2 --layer 11 --from b          # rerun only the mesh (after changing its settings)

Every tile listed must have been through run_pipeline.py and run_layers.py.
Output folder (from config.py): mesh_<tiles>_layerNN/ (e.g. mesh_t1_layer11/), the mesh in its wraps/ folder.
Next: run_ink.py, which prints its command at the end of this run.
"""
import argparse, os, subprocess, sys, time
from config import tile_dirs, mesh_dir

# =============================================================================================== SETTINGS
TILES_TO_USE = ["t1"]                     # one tile, or several neighbouring tiles, e.g. ["t1", "t2", "t3", "t4", "t5", "t6"]
LAYER = 11                                # the layer to follow, numbered as in the FIRST tile listed

# ---- 7a: combine tiles (only used with more than one tile)
COMBINE = dict(
    radius=3.0,                           # two track points closer than this (voxels) count as the same band when matching the layer
)

# ---- 7b: mesh
MESH = dict(
    node=2.5,                             # thinning: one point per this many voxels
    link=6.0,                             # link points closer than this along the sheet
    across=4.0,                           # ...and no further than this across the sheet (keeps fold limbs apart)
    landmarks=200,                        # landmarks for the first flat layout
    local=40.0,                           # the flat layout keeps walking distances up to this far (voxels)
    max_stretch=2.5,                      # mesh cells stretched more than this (or squashed below 1/this) are cut out
    fit=1.2,                              # radius deciding where the sheet has tracks (larger grows the mesh past the tracks)
    smooth=3.0,                           # radius over which the band middle is smoothed
    base=5.0,                             # radius of the first smooth surface through all the layer's tracks
    keep=5.0,                             # where tracks spread wider than one band, keep those within this of the clean surface (0 = keep all)
    clean_sd=3.0,                         # an area is clean (one band) when its tracks spread less than this (standard deviation, voxels)
    reach=12.0,                           # how far across the sheet to look for the papyrus band in the CT
    bridge=6.0,                           # holes narrower than about twice this are bridged (and marked)
    step=2,                               # mesh vertex spacing (voxels)
    compare_old=False,                    # also run the old plane/cylinder mesher on the same points (needs its script)
)
# =============================================================================================== end of SETTINGS

ap = argparse.ArgumentParser()
ap.add_argument("--tiles", nargs="+", default=TILES_TO_USE, help="tile names from config.py; the first is the reference")
ap.add_argument("--layer", type=int, default=LAYER, help="layer number in the first tile")
ap.add_argument("--from", dest="start", choices=["a", "b"], default="a", help="b = skip combining and rerun only the mesh")
a = ap.parse_args()
for t in a.tiles:
    D = tile_dirs(t)
    for f in (os.path.join(D["ckpt"], "paths.npy"), os.path.join(D["layers"], "layer_id.npy")):
        if not os.path.exists(f): sys.exit(f"Cannot find {f}. Run run_pipeline.py and run_layers.py for tile {t} first.")


def args(d):
    """settings dict -> command-line arguments (None / False = leave out, True = flag)"""
    out = []
    for k, v in d.items():
        if v is None or v is False: continue
        out += [f"--{k}"] if v is True else [f"--{k}", str(v)]
    return out


out_dir = mesh_dir(a.tiles, a.layer)
if len(a.tiles) == 1:                                          # one tile: the mesher reads the tile directly
    D = tile_dirs(a.tiles[0]); ckpt, layers_dir = D["ckpt"], D["layers"]
    steps = [("b", ["phase7b_mesh.py"] + args(dict(ckpt=ckpt, layers_dir=layers_dir, out=os.path.join(out_dir, "wraps"), layer=a.layer, **MESH)))]
else:                                                          # several tiles: combine into out_dir, then mesh from there
    ckpt, layers_dir = out_dir, os.path.join(out_dir, "layers")
    steps = [("a", ["phase7a_combine_tiles.py", "--tiles", *a.tiles] + args(dict(layer=a.layer, out=out_dir, **COMBINE))),
             ("b", ["phase7b_mesh.py"] + args(dict(ckpt=ckpt, layers_dir=layers_dir, out=os.path.join(out_dir, "wraps"), layer=a.layer, **MESH)))]
    if a.start == "b":
        if not os.path.exists(os.path.join(out_dir, "paths.npy")): sys.exit(f"{out_dir} has not been combined yet; run without --from b first.")
        steps = steps[1:]

print(f"[RUN MESH] layer {a.layer} of {a.tiles[0]} | tiles {', '.join(a.tiles)} | " + ("one tile: no combining" if len(a.tiles) == 1 else "combining first") + f" | -> {out_dir}/")
t0 = time.time()
for s, cmd in steps:
    print(f"\n{'=' * 100}\n[7{s}] python " + " ".join(cmd) + f"\n{'=' * 100}", flush=True)
    t1 = time.time(); r = subprocess.run([sys.executable] + cmd)
    if r.returncode != 0:
        sys.exit(f"\n[RUN MESH] step 7{s} stopped with an error (code {r.returncode}). Fix it, then rerun" + (" with --from b" if s == "b" and len(a.tiles) > 1 else "") + ".")
    print(f"[7{s}] done in {time.time() - t1:.0f}s", flush=True)
print(f"\n[RUN MESH] done in {(time.time() - t0) / 60:.1f} min | mesh in {os.path.join(out_dir, 'wraps')}/")
print(f"  next:  python run_ink.py --name layer{a.layer:02d} --id <any number, e.g. 9{a.layer:03d}> --ckpt {ckpt}")
