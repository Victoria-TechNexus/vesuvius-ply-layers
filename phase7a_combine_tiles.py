
"""phase7a_combine_tiles.py - phase 7, step a (run by run_mesh.py when more than one tile is used): match one layer across
neighbouring, overlapping tiles and combine it into a single set for the mesher (phase 7b). (Was combine_tiles.py.)

From here on ONE layer is followed. With a single tile this step is skipped: the mesher reads the tile directly.

    python phase7a_combine_tiles.py --tiles t1 t2 t3 t4 t5 t6 --layer 11
    python phase7a_combine_tiles.py --tiles ckpt_t1:layers_t1 ckpt_t2:layers_t2 --layer 11 --out my_folder

--tiles   one entry per tile: a tile name from config.py (t1, t2, ...) or "stage-1 folder:layers folder".
          The FIRST tile is the reference; --layer is a layer number in it.
          Every later tile must overlap the tiles before it (about 40 voxels of overlap is the intention).
--layer   the layer to follow, numbered as in the first tile.
--out     folder to write; default from config.py: mesh_<tiles>_layerNN (e.g. mesh_t1-t2-t3-t4-t5-t6_layer11).
          It is laid out like a stage-1 folder, so the mesher can read it directly:
              paths.npy, lens.npy                   the layer's tracks from all tiles, in one coordinate frame
              layers/layer_id.npy                   every track labelled with --layer
              bright_f16.npy, normals_f16.npy, vol_f16.npy   the CT volumes stitched over the combined region
              manifest.json                         the combined region's offset in the scroll, the scan and the scroll centre
              tile_of_track.npy                     which tile each track came from

How layers are matched: layer numbers are separate in every tile. In the overlap the same papyrus is in both tiles, so the
tracks of the reference layer and the tracks of the matching layer lie in the same band. For each layer of the new tile, the
script counts its track points in the overlap that sit within --radius voxels of a reference-layer point, and takes the layer
with the most. It prints the table so the match can be judged.
"""
import argparse, json, os, sys
import numpy as np
from scipy.spatial import cKDTree
from config import TILES, tile_dirs, mesh_dir, scroll_centre, tile_name

ap = argparse.ArgumentParser()
ap.add_argument("--tiles", nargs="+", required=True, help='tile names (t1 t2 ...) or entries like "ckpt_t1:layers_t1"; the first is the reference tile')
ap.add_argument("--layer", type=int, required=True, help="layer number in the first tile")
ap.add_argument("--out", default=None, help="default: mesh_<tiles>_layerNN from config.py")
ap.add_argument("--radius", type=float, default=3.0, help="two track points closer than this (voxels) count as the same band")
a = ap.parse_args()

def folders(spec):
    """tile name -> its folders from config.py | 'ckpt:layers' -> as given"""
    if ":" in spec: ck, _, ly = spec.partition(":"); return ck, ly
    if spec in TILES: D = tile_dirs(spec); return D["ckpt"], D["layers"]
    sys.exit(f"tile '{spec}': give a tile name from config.py ({', '.join(TILES)}) or 'stage-1 folder:layers folder'")

def load_tile(spec):
    ck, ly = folders(spec)
    need = [os.path.join(ck, f) for f in ("manifest.json", "paths.npy", "lens.npy", "bright_f16.npy", "normals_f16.npy", "vol_f16.npy")] + [os.path.join(ly, "layer_id.npy")]
    miss = [f for f in need if not os.path.exists(f)]
    if miss: sys.exit(f"tile '{spec}': missing " + ", ".join(miss) + f"  (looked from {os.getcwd()}; run run_pipeline.py and run_layers.py for it first)")
    man = json.load(open(os.path.join(ck, "manifest.json")))
    off = np.array(man["chunk"]["offset_zyx"], dtype=np.int64); shp = np.array(man["chunk"]["shape"], dtype=np.int64)
    lens = np.load(os.path.join(ck, "lens.npy")); lay = np.load(os.path.join(ly, "layer_id.npy"))
    if len(lay) != len(lens): sys.exit(f"tile '{spec}': {ly}/layer_id.npy has {len(lay)} tracks but {ck}/lens.npy has {len(lens)}; the layers folder is from a different run")
    P = np.load(os.path.join(ck, "paths.npy")).astype(np.float64) + off                     # scroll coordinates (z, y, x)
    return dict(spec=spec, name=tile_name(ck, man), man=man, ck=ck, off=off, shp=shp, lens=lens, lay=lay, P=P, tid=np.repeat(np.arange(len(lens)), lens))

tiles = [load_tile(s) for s in a.tiles]
if a.out is None: a.out = mesh_dir([t["name"] for t in tiles], a.layer)
print(f"[COMBINE] {len(tiles)} tiles | following layer {a.layer} of '{tiles[0]['spec']}' | writing {a.out}/")
for t in tiles: print(f"  {t['spec']}: offset (z, y, x) {t['off'].tolist()}, size {t['shp'].tolist()}, {len(t['lens']):,} tracks")

# ------------------------------------------------------------------ match the layer tile by tile
t0 = tiles[0]; t0["sel"] = t0["lay"] == a.layer
if not t0["sel"].any(): sys.exit(f"layer {a.layer} has no tracks in the first tile")
ref = t0["P"][t0["sel"][t0["tid"]]][::3]                                                    # reference cloud, grows as tiles are added
boxes = [(t0["off"], t0["off"] + t0["shp"])]; matched = {t0["name"]: [a.layer]}
for t in tiles[1:]:
    inside_prev = np.zeros(len(t["P"]), bool)
    for lo, hi in boxes: inside_prev |= np.all((t["P"] >= lo) & (t["P"] < hi), axis=1)       # this tile's points that lie in an earlier tile
    if not inside_prev.any(): sys.exit(f"tile '{t['spec']}' does not overlap the tiles before it; check the tile ranges in config.py")
    idx = np.flatnonzero(inside_prev); lay_pt = t["lay"][t["tid"][idx]]
    d, _ = cKDTree(ref).query(t["P"][idx], distance_upper_bound=a.radius); near = np.isfinite(d)
    ok = lay_pt >= 0; nL = int(t["lay"].max()) + 1
    n_ov = np.bincount(lay_pt[ok], minlength=nL); n_near = np.bincount(lay_pt[ok & near], minlength=nL); share = n_near / np.maximum(n_ov, 1)
    best = int(np.argmax(n_near))
    if n_near[best] == 0: sys.exit(f"tile '{t['spec']}': no layer has tracks within {a.radius:g} voxels of the reference layer in the overlap")
    chosen = sorted({best} | {int(k) for k in range(nL) if n_near[k] >= 0.3 * n_near[best] and share[k] >= 0.5})
    matched[t["name"]] = chosen
    print(f"  '{t['spec']}': {len(idx):,} of its track points lie in the overlap | its layers there, by points within {a.radius:g} voxels of the reference layer:")
    for k in np.argsort(-n_near)[:5]:
        if n_near[k]: print(f"      layer {int(k):3d}: {int(n_near[k]):8,} points close ({share[k]:.0%} of that layer's points in the overlap)" + ("   <- matched" if int(k) in chosen else ""))
    t["sel"] = np.isin(t["lay"], chosen); selpt = t["sel"][t["tid"]]
    # how well the two tiles agree in the overlap: reference points inside this tile, distance to the matched layer's points
    lo, hi = t["off"], t["off"] + t["shp"]; r_in = ref[np.all((ref >= lo) & (ref < hi), axis=1)]; m_in = t["P"][selpt & inside_prev][::3]
    if len(r_in) and len(m_in):
        dd = cKDTree(m_in).query(r_in)[0]
        print(f"      agreement in the overlap: reference track points -> nearest matched track point: p50 {np.percentile(dd, 50):.1f}, p95 {np.percentile(dd, 95):.1f} voxels "
              f"| within {a.radius:g}: {(dd <= a.radius).mean():.0%}   (tracks differ between tiles; a layer is about 9 voxels thick)")
    if share[best] < 0.5: print(f"      WARNING: only {share[best]:.0%} of the matched layer's overlap points are close to the reference layer. The match is doubtful.")
    ref = np.vstack([ref, t["P"][selpt][::3]]); boxes.append((lo, hi))

# ------------------------------------------------------------------ combine the tracks
lo = np.min([t["off"] for t in tiles], axis=0); hi = np.max([t["off"] + t["shp"] for t in tiles], axis=0); shp = (hi - lo).astype(int)
os.makedirs(os.path.join(a.out, "layers"), exist_ok=True)
P_out = np.vstack([t["P"][t["sel"][t["tid"]]] - lo for t in tiles]).astype(np.float32); lens_out = np.concatenate([t["lens"][t["sel"]] for t in tiles])
tile_of = np.concatenate([np.full(int(t["sel"].sum()), i) for i, t in enumerate(tiles)])
assert lens_out.sum() == len(P_out), "track lengths and points do not add up"
np.save(os.path.join(a.out, "paths.npy"), P_out); np.save(os.path.join(a.out, "lens.npy"), lens_out); np.save(os.path.join(a.out, "tile_of_track.npy"), tile_of)
np.save(os.path.join(a.out, "layers", "layer_id.npy"), np.full(len(lens_out), a.layer, dtype=np.int64))
print(f"  combined layer: {len(lens_out):,} tracks, {len(P_out):,} points | region offset (z, y, x) {lo.tolist()}, size {shp.tolist()}")

# ------------------------------------------------------------------ stitch the CT volumes; in an overlap, each voxel comes from the tile whose edge is furthest away
bright = np.zeros(tuple(shp), np.float16); vol = np.zeros(tuple(shp), np.float16); nrm = np.zeros(tuple(shp) + (3,), np.float16)
wbest = np.full(tuple(shp), -1, np.int16)
for t in tiles:
    n = t["shp"]; ramp = [np.minimum(np.arange(k), k - 1 - np.arange(k)) for k in n]
    w = np.minimum(np.minimum(ramp[0][:, None, None], ramp[1][None, :, None]), ramp[2][None, None, :]).astype(np.int16)   # distance to this tile's nearest face
    o = (t["off"] - lo).astype(int); sl = (slice(o[0], o[0] + n[0]), slice(o[1], o[1] + n[1]), slice(o[2], o[2] + n[2]))
    take = w > wbest[sl]
    for name, dst in (("bright_f16.npy", bright), ("vol_f16.npy", vol), ("normals_f16.npy", nrm)):
        src = np.load(os.path.join(t["ck"], name))
        if tuple(src.shape[:3]) != tuple(n): sys.exit(f"{t['ck']}/{name} has shape {src.shape[:3]}, the manifest says {tuple(n)}")
        view = dst[sl]; view[take] = src[take]; del src
    wv = wbest[sl]; wv[take] = w[take]
covered = float((wbest >= 0).mean())
np.save(os.path.join(a.out, "bright_f16.npy"), bright); np.save(os.path.join(a.out, "vol_f16.npy"), vol); np.save(os.path.join(a.out, "normals_f16.npy"), nrm)
m0 = tiles[0]["man"]; combo_name = "-".join(t["name"] for t in tiles)
json.dump(dict(chunk=dict(offset_zyx=[int(v) for v in lo], shape=[int(v) for v in shp], z=[int(lo[0]), int(hi[0])], y=[int(lo[1]), int(hi[1])], x=[int(lo[2]), int(hi[2])]),
               tile=combo_name, scan=m0.get("scan"), scroll_centre_xy=scroll_centre(m0),
               combined_from=[t["spec"] for t in tiles], matched_layers=matched, layer=a.layer, tracks=int(len(lens_out)), points=int(len(P_out)),
               coordinates="paths.npy is (z, y, x) in voxels local to the combined region; add offset_zyx for scroll coordinates"),
          open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
print(f"  stitched CT volumes: {covered:.0%} of the combined box is covered by a tile")
print(f"  wrote {a.out}/ | next:")
print(f"    python phase7b_mesh.py --ckpt {a.out} --layers_dir {os.path.join(a.out, 'layers')} --out {os.path.join(a.out, 'wraps')} --layer {a.layer}")
print(f"    python run_ink.py --name layer{a.layer:02d} --id {9100 + a.layer} --ckpt {a.out} --wraps {os.path.join(a.out, 'wraps')}")
