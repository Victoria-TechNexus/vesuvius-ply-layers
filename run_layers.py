
"""run_layers.py - phase 6 for one tile, in one command: every sheet of the tile numbered by layer.

Runs the four steps in order, each as its own script (each can still be run on its own):
    6a  phase6a_split_sheets.py   split joined sheets that stack on themselves  -> ckpt_<tile>/sheet_id_split.npy
    6b  phase6b_pair_table.py     measure what lies between neighbouring sheets -> pairs_<tile>/
    6c  phase6c_ply_label.py      read the fibre plies between them             -> plies_<tile>/
    6d  phase6d_stack_layers.py   number every track by layer                   -> layers_<tile>/

Phase 6 works on ALL the papyrus in the tile. Picking one layer comes after, in run_mesh.py (phase 7).

    python run_layers.py                 # the tile in SETTINGS below
    python run_layers.py t2              # another tile (must be in TILES in config.py, and run_pipeline.py done for it)
    python run_layers.py t1 --from c     # rerun only steps c and d (after changing their settings)
    python run_layers.py t1 --only d     # rerun only step d

Folders come from config.py. Every adjustable setting of the four steps is below; the values are the steps' own defaults.
"""
import argparse, os, subprocess, sys, time
from config import tile_dirs

# =============================================================================================== SETTINGS
TILE = "t1"
DIAG = "diagnostics"                      # folder for the check pictures (named per tile)

LABELS = "sheet_id_split.npy"             # sheet labels written by 6a and read by 6b, 6c, 6d (in the ckpt folder)
MIN_TRACKS = 30                           # sheets with fewer tracks are left out of the pair table and the sheet level of 6d
MIN_DARK = 3.0                            # a dark stretch this long (voxels) is a real gap between layers (6b and 6d)
CORE = None                               # scroll centre (x, y); None = from the tile's manifest / config.py

# ---- 6a: split joined sheets back into their pieces
SPLIT = dict(
    split=[],                             # joined sheet numbers to split by hand (from sheets.csv), e.g. [186, 329]
    stacked=0.2,                          # also split every joined sheet whose stacked share is at least this (0 = off)
)

# ---- 6b: pair table
PAIRS = dict(
    samples=120000,                       # start points, spread at random over all sheets
    reach=30.0,                           # how far to walk outward looking for the next sheet (voxels)
    ray=1.2,                              # a track point this close to the walk line counts as met
    thin_max=11.0,                        # a bright band up to this thick, with dark on both sides, is one layer
    min_n=10,                             # plain samples a pair needs before it gets a label
    field="",                             # optional: field_stack.npy of THIS tile, adds recto / verso measurements
    seed=0,
)

# ---- 6c: ply labels
PLIES = dict(
    per_pair=6,                           # spots read per pair (spread apart along the pair). Was 2; 6 cut the worst doubled layer on t2 / t3 (v1.1), 6c takes ~9 min
    min_n=20,                             # pairs with fewer pair-table samples are skipped
    limit=0,                              # only the N biggest pairs (0 = all)
    radius=24,                            # half size of each patch (voxels)
    before=15, after=15,                  # slices start this far inside sheet A and end this far outside sheet B
    hp=4.0, grad=1.0, rho=4.0,            # stripe measurement: high-pass, gradient and smoothing scales
    coh=0.3,                              # minimum stripe coherence
    v_max=30.0, h_min=60.0,               # angle to the scroll axis: under v_max = V (verso), over h_min = H (recto)
    margin=0.2,                           # H or V must rise this far above 33% (and above the other) to name a depth
    dark=0.5,                             # a depth where more than this share of the patch is not papyrus is a gap
    papyrus=0.5,                          # brightness from which a voxel counts as papyrus
    centre=12,                            # plies are read in a square this far around the spot
    wide=8.0,                             # a gap this wide between two plies of one direction is a layer gap, not a crack
    delam_max=12.0,                       # recto, gap, verso with a wider gap is only 'possibly delaminated'
    ply=7.0,                              # thickness of one ply (voxels)
    double=1.7,                           # a ply run at least this many plies thick counts as doubled
    search=3,                             # look this many slices either side of each sheet for its ply
    states="",                            # optional: layer_state.py output, for comparison only
    z=-1, show_z=2, cards=8,              # pictures: slice (-1 = middle), spots within this many slices, cards per label
    seed=0,
)

# ---- 6d: stack layers
LAYERS = dict(
    use_plies=True,                       # measured ply labels (6c) overrule the dark-gap rule; False = dark-gap rule only
    ply_reach=30.0,                       # only neighbours closer than this follow a ply label
    ply_near=80.0,                        # a pair's label is used only this close to where it was read (0 = everywhere). Was 40
    same_needs=1,                         # a same-layer ply reading joins two sheets only if this many spots agreed (1 = as before, 2 = both)
    cell=3.0,                             # column width (voxels)
    same_gap=2.5,                         # depth gap that separates two passes of one sheet or track through a column
    max_thick=20.0,                       # tracks further apart than this are never the same layer
    min_cols=5,                           # columns a pair of sheets must share to count
    max_gap=60.0,                         # neighbours further apart than this along a column are ignored
    no_refine=False,                      # True = stop at the sheet level (no track-level step)
    bright="bright_f16.npy",              # brightness file used to find dark gaps
    verdicts="",                          # optional: pair_verdicts.csv from train_pairs.py
    split_at=0.7, unsure_weight=0.2,      # only with verdicts
)
# =============================================================================================== end of SETTINGS

STEPS = "abcd"
ap = argparse.ArgumentParser()
ap.add_argument("tile", nargs="?", default=TILE)
ap.add_argument("--from", dest="start", choices=list(STEPS), default="a", help="first step to run")
ap.add_argument("--only", choices=list(STEPS), default=None, help="run just this step")
a = ap.parse_args()
D = tile_dirs(a.tile)
if not os.path.exists(os.path.join(D["ckpt"], "paths.npy")):
    sys.exit(f"Cannot find {D['ckpt']}/paths.npy. Run run_pipeline.py for tile {a.tile} first.")


def args(d):
    """settings dict -> command-line arguments (None = leave out, True = flag, list = several values)"""
    out = []
    for k, v in d.items():
        if v is None or v is False: continue
        if v is True: out.append(f"--{k}"); continue
        if isinstance(v, (list, tuple)):
            if len(v): out += [f"--{k}"] + [str(x) for x in v]
            continue
        out += [f"--{k}", str(v)]
    return out


core = dict(core=list(CORE) if CORE is not None else None)
lay = dict(LAYERS); use_plies = lay.pop("use_plies")
CMDS = {
    "a": ["phase6a_split_sheets.py"] + args(dict(ckpt=D["ckpt"], out=LABELS, min_tracks=MIN_TRACKS, **SPLIT)),
    "b": ["phase6b_pair_table.py"] + args(dict(ckpt=D["ckpt"], out=D["pairs"], diag=DIAG, labels=LABELS, min_tracks=MIN_TRACKS, min_dark=MIN_DARK, **core, **PAIRS)),
    "c": ["phase6c_ply_label.py"] + args(dict(ckpt=D["ckpt"], pairs=D["pairs"], out=D["plies"], diag=DIAG, labels=LABELS, **core, **PLIES)),
    "d": ["phase6d_stack_layers.py"] + args(dict(ckpt=D["ckpt"], out=D["layers"], labels=LABELS, min_tracks=MIN_TRACKS, min_dark=MIN_DARK, **core, **lay))
          + ["--plies", os.path.join(D["plies"], "pair_labels.csv") if use_plies else ""],
}
todo = [a.only] if a.only else STEPS[STEPS.index(a.start):]
print(f"[RUN LAYERS] tile {a.tile} | steps {', '.join('6' + s for s in todo)}")
t0 = time.time()
for s in todo:
    cmd = [sys.executable] + CMDS[s]
    print(f"\n{'=' * 100}\n[6{s}] python " + " ".join(f'"{x}"' if x == "" else x for x in CMDS[s]) + f"\n{'=' * 100}", flush=True)
    t1 = time.time(); r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"\n[RUN LAYERS] step 6{s} stopped with an error (code {r.returncode}). Fix it, then: python run_layers.py {a.tile} --from {s}")
    print(f"[6{s}] done in {time.time() - t1:.0f}s", flush=True)
print(f"\n[RUN LAYERS] tile {a.tile} done in {(time.time() - t0) / 60:.1f} min | layers in {D['layers']}/ | next: run_mesh.py (one layer)")
