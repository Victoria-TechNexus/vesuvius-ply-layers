
"""config.py - WHERE the data is: the scan, the scroll centre and the tiles. Read by every step.

The handlers hold HOW each stage runs (their own SETTINGS):
    run_pipeline.py  phases 1-5   one tile, all of its papyrus        -> tracks, pieces, sheets
    run_layers.py    phase 6      one tile, all of its papyrus        -> every sheet numbered by layer
    run_mesh.py      phase 7      ONE LAYER, followed across tiles    -> its mesh, in mesh_<tiles>_layerNN/
    run_ink.py       phases 8-10  that one layer                      -> flat image and ink

Phases 1-6 work on all the data in a tile. From phase 7 on you pick one layer and everything after follows that layer.
"""
import os

# ---- the scan ----
SCROLL = "PHercParis4"                                          # folder name in the open-data bucket
SCAN = "20260411134726-2.400um-0.2m-78keV-masked.zarr"          # the scan volume in that folder
SCAN_SOURCE = None                                              # or a full path / URL to a .zarr; overrides the two above
LEVEL = 2                                                       # resolution level (2 of a 2.4 um scan = 9.6 um, which all defaults assume)
# To list a scan's levels and sizes: python phase1_stream_ct.py --scroll <name> --scan <volume>

# ---- the scroll's centre (umbilicus) in level-LEVEL voxels, x then y. Used to tell inward (recto side) from outward. ----
SCROLL_CENTRE_XY = (4087.0, 4087.0)                             # PHerc. Paris 4; MUST be changed for another scroll

# ---- tiles: z, y, x ranges in level-LEVEL voxels. Neighbouring tiles should share about 40 voxels. ----
TILES = {
    "t1": dict(z=(15534, 15726), y=(4794, 5210), x=(3895, 4311)),
    "t2": dict(z=(15534, 15726), y=(4794, 5210), x=(4271, 4687)),   # beside t1
    "t3": dict(z=(15382, 15574), y=(4794, 5210), x=(3895, 4311)),   # above t1
    "t4": dict(z=(15686, 15878), y=(4794, 5210), x=(3895, 4311)),   # below t1
    "t5": dict(z=(15382, 15574), y=(4794, 5210), x=(4271, 4687)),   # above t2
    "t6": dict(z=(15686, 15878), y=(4794, 5210), x=(4271, 4687)),   # below t2
}

# ---- where each tile's results go (all folders inside OUTPUT_ROOT) ----
OUTPUT_ROOT = "."                                               # e.g. "/content/drive/MyDrive/vesuvius_runs"


def tile_dirs(tile):
    """The folders of one tile: stage 1 (ckpt), pair table (pairs), ply readings (plies), layers (layers)."""
    if tile not in TILES:
        raise SystemExit(f"unknown tile '{tile}'; known tiles: {', '.join(TILES)} (add it to TILES in config.py)")
    return {k: os.path.join(OUTPUT_ROOT, f"{k}_{tile}") for k in ("ckpt", "pairs", "plies", "layers")}


def scroll_centre(manifest=None, override=None):
    """The scroll centre (x, y) a step should use: an explicit --core if given, else the one saved in the tile's
    manifest.json by run_pipeline.py, else SCROLL_CENTRE_XY above."""
    if override is not None:
        return [float(v) for v in override]
    if manifest and manifest.get("scroll_centre_xy"):
        return [float(v) for v in manifest["scroll_centre_xy"]]
    return [float(v) for v in SCROLL_CENTRE_XY]


def mesh_dir(tiles, layer):
    """Folder for one layer's mesh (phase 7) and everything after it: mesh_t1_layer11, or mesh_t1-t2-t3_layer11 when
    tiles are combined. The layer number is the one in the FIRST tile listed. tiles: a list of tile names or one name."""
    name = tiles if isinstance(tiles, str) else "-".join(tiles)
    return os.path.join(OUTPUT_ROOT, f"mesh_{name}_layer{int(layer):02d}")


def tile_name(ckpt_dir, manifest=None):
    """Short name of a tile for file names: from manifest.json, else from the folder name (ckpt_t1 -> t1)."""
    if manifest and manifest.get("tile"):
        return manifest["tile"]
    base = os.path.basename(os.path.normpath(ckpt_dir))
    return base[5:] if base.startswith("ckpt_") else base
