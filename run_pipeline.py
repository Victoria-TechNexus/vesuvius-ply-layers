"""run_pipeline.py - stage 1 for one tile: CT -> structure tensor + U-Net -> tracks -> pieces -> sheets.

    python run_pipeline.py t2          # run the tile named t2 in config.py

Writes <ckpt>/ : paths.npy, lens.npy, component_id.npy, sheet_id.npy, sheets.csv, join_pairs.csv, the compact CT, brightness
and normals that the later steps read, and manifest.json with every setting used.

Where the data is (scan, tiles) is in config.py; how stage 1 runs is in the SETTINGS block below. docs/TUNING.md says which setting to try for which problem.
Nothing in this file uses the official segment or its labels; comparing with them is done afterwards (cut_official.py,
compare_labels.py).
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from config import SCROLL, SCAN, SCAN_SOURCE, LEVEL, SCROLL_CENTRE_XY, TILES, tile_dirs
from phase1_stream_ct import ScrollStreamer
from phase2_structure_tensor import compute_structure_tensor, build_phase4_field_stack
from phase3_unet import load_unet, predict_volume
from phase4_tracking import Tracer, Rules, ridge_seeds
from phase5_pieces import build_pieces, trilinear_sample

_T0 = time.time()

# ================================================================ SETTINGS ================================================================
# The scan, the tiles and the scroll centre are in config.py (shared by every step).
TILE = "t1"                                                     # used when no tile is named on the command line

# ---- brightness: B = blurred CT rescaled between these percentiles; B >= 0.5 counts as papyrus ----
BRIGHT_PERCENTILES = (20, 90)

# ---- phase 2: structure tensor | phase 3: U-Net ----
ST = dict(sigma=1.0, rho=6.0)                                   # gradient scale, integration scale (voxels)
UNET_CHECKPOINT = "phase3_unet_checkpoint.pt"                # without it, tensor-based stand-ins are used (results differ)

# ---- phase 4: tracking ----
TRACK = dict(
    integrator="rk4",    # "rk4" (4 samples per step, follows bends closely) or "euler" (1 sample, faster, drifts on bends)
    step=0.5,            # step length (voxels)
    max_len=600.0,       # longest track in each direction (voxels)
    tau_sheet=0.5,       # stop where brightness B drops below this (leaving papyrus)
    tau_verso=0.6,       # stop where the U-Net's verso probability rises above this
    tau_recto=0.0,       # stop where the U-Net's recto probability drops below this (0 = off)
    use_classes=True,    # apply the verso / recto stops at all
    kappa_max=0.5,       # stop where the track turns more than this (radians per voxel)
    trim=1.0)            # cut this many voxels off an end stopped by any rule except the tile edge or length
SEED = dict(offset=1.5,  # a seed must be at least as bright as the points this far either side along the normal (a ridge)
            spacing=2.0, # at most one seed per cell of this size (voxels)
            cover_r=2.0) # no seed within this distance of an earlier track
SEED_ROUNDS = [(45000, 0.6), (15000, 0.5)]                      # (seed cap for a 192 x 416 x 416 tile, brightness threshold)
SEED_CAP_SCALES_WITH_TILE = True                                # caps grow with tile volume, so bigger tiles keep the same density
MIN_TRACK_POINTS = 30                                           # shorter tracks are dropped

# ---- phase 5: pieces ----
P5 = dict(
    subsample=3,             # use every 3rd track point
    r_max=3.5,               # pair points of different tracks closer than this (voxels)
    d_perp_max=1.0,          # ...that sit at the same depth: offset along the normal at most this (voxels)
    min_support=3,           # a track pair needs at least this many point pairs
    sigma_n=1.0,             # weight falls off with depth offset (voxels)
    sigma_par=8.0,           # ...and with sideways distance (voxels)
    alpha=2.5,               # ...and with the angle between the tracks (|cos|^alpha)
    verso_mode="integral",   # verso penalty: "integral" (verso probability x distance) or "level"
    verso_scale=0.15,        # ...strength of the "integral" penalty (smaller = stricter)
    tau_W=0.005,             # point pairs weighing less than this are dropped
    valley_threshold=0.05,   # drop a track pair if the CT between them dips by more than this share (a dark gap); None = off
    louvain_resolution=3.0,  # grouping: higher = smaller, purer pieces; lower = bigger pieces, more risk of spanning two layers
    seed=42)                 # random seed of the grouping

# ---- joining pieces into sheets ----
JOIN = dict(close=8.0, min_pts=100, max_apart=0.20, min_bright=0.90, apart_depth=2.5, cap=4000)
MIN_TRACKS = 30                                                 # pieces / sheets with fewer tracks are not used

# ---- output ----
LOCAL_DIR = "vesuvius_local"                                    # full-precision stacks (about 2.5 GB per tile)
SAVE_FULL_STACKS = False                                        # nothing later reads them; True keeps them for inspection
WRITE_POINT_CLOUD = True                                        # diagnostics/phase5_wraps.ply, one colour per piece
DIAG_DIR = "diagnostics"
# =========================================================================================================================================

REF_TILE_VOXELS = 192 * 416 * 416
if len(sys.argv) > 1: TILE = sys.argv[1]
CKPT_DIR = tile_dirs(TILE)["ckpt"]
print(f"[TILE] {TILE}: z {TILES[TILE]['z']}, y {TILES[TILE]['y']}, x {TILES[TILE]['x']} -> {CKPT_DIR}/")
for _d in (CKPT_DIR, DIAG_DIR) + ((LOCAL_DIR,) if SAVE_FULL_STACKS else ()): os.makedirs(_d, exist_ok=True)

_saved = {}
def save_arr(folder, name, arr):
    """np.save, force it to disk, read the header back, and record the exact byte size for the manifest."""
    path = os.path.join(folder, name)
    with open(path, "wb") as f:
        np.save(f, arr); f.flush(); os.fsync(f.fileno())
    back = np.load(path, mmap_mode="r")
    assert back.shape == arr.shape and os.path.getsize(path) >= arr.nbytes, f"{path} was cut short"
    _saved[name] = dict(folder=folder, bytes=int(os.path.getsize(path)), shape=list(arr.shape), dtype=str(arr.dtype)); del back
    return path

def sign_safe_sample(arr3, pts):
    """Trilinear interpolation of a sign-ambiguous unit-vector field: the 8 corner vectors are turned to agree with the
    nearest voxel's vector before averaging, so opposite signs do not cancel."""
    hi = np.array(arr3.shape[:3]) - 1
    p = np.clip(pts, 0, hi - 1e-6); i0 = np.floor(p).astype(int); f = p - i0
    ref = arr3[tuple(np.clip(np.round(p).astype(int), 0, hi).T)].astype(np.float64)
    out = np.zeros((len(p), 3))
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                c = arr3[i0[:, 0] + dz, i0[:, 1] + dy, i0[:, 2] + dx].astype(np.float64)
                c *= np.sign(np.sum(c * ref, 1, keepdims=True) + 1e-12)
                w = ((f[:, 0] if dz else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dx else 1 - f[:, 2]))
                out += w[:, None] * c
    return out

# ---------------------------------------------------------------------------------------------------------------- 1. CT
print("\n[PHASE 1] Streaming the CT tile")
streamer = ScrollStreamer(scroll_name=SCROLL, scan_folder=SCAN, level=LEVEL, source=SCAN_SOURCE)
(z_start, z_end), (y_start, y_end), (x_start, x_end) = TILES[TILE]["z"], TILES[TILE]["y"], TILES[TILE]["x"]
vol = streamer.fetch_chunk(z_start, z_end, y_start, y_end, x_start, x_end)
print(f"  volume {vol.shape}, {vol.dtype}")

# ---------------------------------------------------------------------------------------------------------------- 2. orientation fields
print("\n[PHASE 2] Structure tensor")
evals, evecs, sheet_prob, polarity = compute_structure_tensor(vol, **ST)
field_stack = build_phase4_field_stack(evecs, sheet_prob, polarity)
del evals, evecs
if field_stack.shape[-1] < 9:
    field_stack = np.pad(field_stack, ((0, 0), (0, 0), (0, 0), (0, 9 - field_stack.shape[-1])), mode="constant")

print("[PHASE 3] U-Net")
if os.path.exists(UNET_CHECKPOINT):
    model, device = load_unet(UNET_CHECKPOINT)
    probs = predict_volume(model, vol, device)                     # (4, z, y, x): sheet, recto, verso, polarity
    field_stack[..., 6], field_stack[..., 7], field_stack[..., 8] = probs[0], probs[1], probs[2]
    print(f"  U-Net loaded ({device}) | mean polarity strength {np.mean(np.abs(probs[3])):.3f}")
    del model, probs
    if torch.cuda.is_available(): torch.cuda.empty_cache()
else:
    print(f"  WARNING: '{UNET_CHECKPOINT}' not found; using the tensor's own stand-ins (results will differ)")
    field_stack[..., 6] = sheet_prob
    vz = np.abs(field_stack[..., 0]); vin = np.sqrt(field_stack[..., 1] ** 2 + field_stack[..., 2] ** 2 + 1e-12); tot = vz + vin
    field_stack[..., 7] = sheet_prob * vin / tot; field_stack[..., 8] = sheet_prob * vz / tot
del sheet_prob, polarity

# ---------------------------------------------------------------------------------------------------------------- 4. tracking
print("\n[PHASE 4] Tracking")
_ct = ndi.gaussian_filter(vol, 1.0); _slo, _shi = np.percentile(_ct[::3, ::3, ::3], BRIGHT_PERCENTILES)
trace_stack = field_stack.copy()
trace_stack[..., 6] = np.clip((_ct - _slo) / (_shi - _slo), 0, 1)    # the tracker follows brightness B, not the U-Net sheet channel
del _ct

tracer = Tracer(trace_stack, sign_safe=True)
rules = Rules(**TRACK)
rng = np.random.default_rng(0)
seed_scale = vol.size / REF_TILE_VOXELS if SEED_CAP_SCALES_WITH_TILE else 1.0
all_traced, valid_paths, end_reasons, covered = [], [], [], None
for rnd, (cap, thr) in enumerate(SEED_ROUNDS):
    cap = int(round(cap * seed_scale))
    seeds = ridge_seeds(trace_stack, covered, thr=thr, **SEED)
    print(f"  round {rnd}: {len(seeds):,} ridge seeds (B > {thr})")
    if len(seeds) < 500:
        print("  coverage sufficient; no more rounds"); break
    if len(seeds) > cap:
        print(f"  capping to {cap:,} seeds (random, seed 0)"); seeds = seeds[rng.choice(len(seeds), cap, replace=False)]
    paths, counts, _ = tracer.trace(seeds, rules)
    _r1, _r2 = tracer.last_reasons
    end_reasons += [(_r2[i], _r1[i]) for i, p in enumerate(paths) if len(p) >= MIN_TRACK_POINTS]
    print(f"    stop reasons {counts} | traced {len(paths):,}, kept {sum(len(p) >= MIN_TRACK_POINTS for p in paths):,}")
    all_traced += [p for p in paths if len(p) >= 2]
    valid_paths += [p for p in paths if len(p) >= MIN_TRACK_POINTS]
    covered = np.vstack(all_traced)[::2]
print(f"  {len(valid_paths):,} tracks kept ({len(all_traced):,} traced)")
del all_traced, covered

# ---------------------------------------------------------------------------------------------------------------- 5. pieces
print("\n[PHASE 5] Grouping tracks into pieces")
p5 = build_pieces(tracklet_paths=valid_paths, field_stack=field_stack, raw_vol=vol, **P5)
if not p5: sys.exit("phase 5 found no track pairs; check the tile and the tracking settings")
component_id = p5["component_id"]
lens_all = np.array([len(p) for p in valid_paths])

save_arr(CKPT_DIR, "paths.npy", np.vstack(valid_paths).astype(np.float32)); save_arr(CKPT_DIR, "lens.npy", lens_all)
save_arr(CKPT_DIR, "component_id.npy", component_id); save_arr(CKPT_DIR, "end_reasons.npy", np.array(end_reasons))
p5["edges"].to_pickle(os.path.join(CKPT_DIR, "edges.pkl"))
save_arr(CKPT_DIR, "normals_f16.npy", field_stack[..., 3:6].astype(np.float16))
save_arr(CKPT_DIR, "bright_f16.npy", trace_stack[..., 6].astype(np.float16))
save_arr(CKPT_DIR, "vol_f16.npy", vol.astype(np.float16))
if SAVE_FULL_STACKS:
    save_arr(LOCAL_DIR, "field_stack.npy", field_stack); save_arr(LOCAL_DIR, "trace_stack.npy", trace_stack); save_arr(LOCAL_DIR, "vol.npy", vol)
print(f"  checkpoint written to {CKPT_DIR}/")

N = len(valid_paths); g = p5["edges"]
_A = coo_matrix((np.ones(len(g)), ((g["key"] // N).astype(int), (g["key"] % N).astype(int))), shape=(N, N))
_, _lab = connected_components(_A, directed=False)
sizes = np.bincount(component_id)
print(f"  {p5['n_components']:,} pieces | with {MIN_TRACKS}+ tracks: {(sizes >= MIN_TRACKS).sum()} | "
      f"connected groups with {MIN_TRACKS}+ tracks: {(np.bincount(_lab) >= MIN_TRACKS).sum()} | largest pieces: {np.sort(sizes)[::-1][:10].tolist()}")

if WRITE_POINT_CLOUD:
    _P = np.vstack(valid_paths)[::4]; _cid = np.repeat(component_id, lens_all)[::4]
    _keep = sizes[_cid] >= MIN_TRACKS; _P, _cid = _P[_keep], _cid[_keep]
    _rgb = np.random.default_rng(7).integers(60, 256, size=(component_id.max() + 1, 3))[_cid]
    _xyz = _P[:, ::-1] + np.array([x_start, y_start, z_start])           # local (z, y, x) -> scroll (x, y, z)
    with open(os.path.join(DIAG_DIR, f"phase5_pieces_{TILE}.ply"), "w") as _f:
        _f.write("ply\nformat ascii 1.0\n" + f"element vertex {len(_xyz)}\n" + "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        np.savetxt(_f, np.c_[_xyz, _rgb], fmt="%.2f %.2f %.2f %d %d %d")
    print(f"  wrote {DIAG_DIR}/phase5_pieces_{TILE}.ply ({len(_xyz):,} points)")

# ---------------------------------------------------------------------------------------------------------------- join pieces into sheets
def join_sheets(paths, comp, bright, close=8.0, min_pts=100, max_apart=0.20, min_bright=0.90, apart_depth=2.5, cap=4000, min_tracks=30):
    """Join pieces of the same layer (end to end or side by side at the same depth), never two pieces a layer apart."""
    t0 = time.time(); lens_ = np.array([len(p) for p in paths]); P_all = np.vstack(paths); cpt = np.repeat(comp, lens_)
    comps = [int(c) for c in np.nonzero(np.bincount(comp) >= min_tracks)[0]]
    pts = {c: P_all[cpt == c][::4].astype(float) for c in comps}; trees = {c: cKDTree(pts[c]) for c in comps}
    def nrm(p):
        n = sign_safe_sample(field_stack[..., 3:6], p); return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    def apart(y, tree, q):                                  # share of points whose neighbours in the other piece sit a layer away
        ny = nrm(y); out = np.zeros(len(y), bool)
        for k, nb in enumerate(tree.query_ball_point(y, float(np.hypot(8.0, 2.0)))):
            if not nb: continue
            dd = q[nb] - y[k]; dep = np.abs(dd @ ny[k]); m = np.sqrt(np.maximum((dd ** 2).sum(1) - dep ** 2, 0)) <= 2.0
            out[k] = bool(m.any() and np.median(dep[m]) >= apart_depth)
        return float(out.mean())
    box = {c: (pts[c].min(0) - close, pts[c].max(0) + close) for c in comps}; rows = []
    for ia, a in enumerate(comps):
        for b in comps[ia + 1:]:
            if not (np.all(box[a][0] <= box[b][1]) and np.all(box[b][0] <= box[a][1])): continue
            X, Y = (a, b) if len(pts[a]) <= len(pts[b]) else (b, a)
            d, j = trees[Y].query(pts[X], distance_upper_bound=close); m = np.isfinite(d)
            if m.sum() < min_pts: continue
            xi = np.nonzero(m)[0]; xi = xi[::max(1, len(xi) // cap)][:cap]; x, y = pts[X][xi], pts[Y][j[xi]]
            st = max(apart(y, trees[X], pts[X]), apart(x, trees[Y], pts[Y]))
            t = np.linspace(0, 1, 9)[None, :, None]
            bm = trilinear_sample(bright, (x[:, None, :] + t * (y - x)[:, None, :]).reshape(-1, 3)).reshape(-1, 9).min(1)
            rows.append((a, b, int(m.sum()), st, float((bm >= 0.5).mean())))
    bad = {(r[0], r[1]) for r in rows if r[3] >= 0.5}
    ok = sorted([r for r in rows if r[3] <= max_apart and r[4] >= min_bright], key=lambda r: -r[2])
    root = {c: c for c in comps}; mem = {c: {c} for c in comps}; done = refused = 0; joined = set()
    def find(x):
        while root[x] != x: root[x] = root[root[x]]; x = root[x]
        return x
    for a, b, *_ in ok:
        x, y = find(a), find(b)
        if x == y: continue
        if any((min(p, q), max(p, q)) in bad for p in mem[x] for q in mem[y]): refused += 1; continue
        root[y] = x; mem[x] |= mem.pop(y); done += 1; joined.add((a, b))
    out = np.array([find(int(c)) if int(c) in root else int(c) for c in comp], dtype=np.int64)
    print(f"\n[JOIN] {len(comps)} pieces | touching pairs {len(rows)} | a layer apart {len(bad)} | pass the rule {len(ok)} | "
          f"joins made {done}, refused {refused} | {time.time() - t0:.0f}s")
    table = pd.DataFrame(rows, columns=["a", "b", "contact", "apart", "bright"])
    table["a_layer_apart"] = table["apart"] >= 0.5
    table["passes"] = (table["apart"] <= max_apart) & (table["bright"] >= min_bright)
    table["grey_zone"] = (table["apart"] > max_apart) & (table["apart"] < 0.5)
    table["joined"] = [(a, b) in joined for a, b in zip(table["a"], table["b"])]
    return out, table

sheet_id, join_table = join_sheets(valid_paths, component_id, trace_stack[..., 6:7], min_tracks=MIN_TRACKS, **JOIN)

def stacked_share(P, n_sub=1500):
    """Share of a sheet's points with another point OF THE SAME SHEET 6-14 voxels away along the normal and within 4 voxels
    sideways. About 0 for one layer; high when two layers are stuck together (split_sheets.py uses it)."""
    rs = np.random.default_rng(0); S = P[rs.choice(len(P), min(n_sub, len(P)), replace=False)]
    n = sign_safe_sample(field_stack[..., 3:6], S); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    hits = 0
    for p, nv, nb in zip(S, n, cKDTree(P).query_ball_point(S, float(np.hypot(14.0, 4.0)))):
        q = P[nb] - p; dep = np.abs(q @ nv); lat = np.sqrt(np.maximum((q ** 2).sum(1) - dep ** 2, 0.0))
        hits += bool(np.any((dep >= 6) & (dep <= 14) & (lat <= 4)))
    return hits / len(S)

_Pall = np.vstack(valid_paths); _sheet_pt = np.repeat(sheet_id, lens_all)
_sid, _inv = np.unique(sheet_id, return_inverse=True)
_s_tracks = np.bincount(_inv); _s_points = np.bincount(_inv, weights=lens_all).astype(np.int64)
_rows = []
for _k in np.argsort(-_s_points)[:60]:
    if _s_tracks[_k] < MIN_TRACKS: continue
    _lab = int(_sid[_k]); _parts = np.unique(component_id[sheet_id == _lab]); _parts = _parts[sizes[_parts] >= MIN_TRACKS]
    _rows.append(dict(sheet=_lab, points=int(_s_points[_k]), tracks=int(_s_tracks[_k]), parts=len(_parts),
                      stacked=round(stacked_share(_Pall[_sheet_pt == _lab][::4].astype(float)), 3),
                      pieces=" ".join(str(int(c)) for c in sorted(_parts, key=lambda c: -sizes[c]))))
sheet_table = pd.DataFrame(_rows, columns=["sheet", "points", "tracks", "parts", "stacked", "pieces"])

save_arr(CKPT_DIR, "sheet_id.npy", sheet_id)
join_table.to_csv(os.path.join(CKPT_DIR, "join_pairs.csv"), index=False)
sheet_table.to_csv(os.path.join(CKPT_DIR, "sheets.csv"), index=False)
import hashlib                                                   # fingerprint of the U-Net file: says which model made this run, whatever the file is called
_unet_sha = hashlib.sha256(open(UNET_CHECKPOINT, "rb").read()).hexdigest() if os.path.exists(UNET_CHECKPOINT) else None
json.dump(dict(
    scan=dict(scroll=SCROLL, volume=SCAN, source=streamer.path, level=LEVEL),
    tile=TILE, scroll_centre_xy=list(SCROLL_CENTRE_XY),
    chunk=dict(z=[z_start, z_end], y=[y_start, y_end], x=[x_start, x_end], offset_zyx=[z_start, y_start, x_start], shape=list(vol.shape)),
    bright_percentiles=list(BRIGHT_PERCENTILES), structure_tensor=ST, unet=UNET_CHECKPOINT if os.path.exists(UNET_CHECKPOINT) else None, unet_sha256=_unet_sha,
    tracer=rules.as_dict(), seed=SEED, seed_rounds=SEED_ROUNDS, seed_cap_scale=round(seed_scale, 3), min_track_points=MIN_TRACK_POINTS,
    phase5=P5, join=JOIN, min_tracks=MIN_TRACKS,
    counts=dict(tracks=len(valid_paths), points=int(lens_all.sum()), pieces=int((sizes >= MIN_TRACKS).sum()),
                sheets=int((_s_tracks >= MIN_TRACKS).sum()), grey_zone_pairs=int(join_table["grey_zone"].sum())),
    coordinates="paths.npy is (z, y, x) in tile-local voxels; add offset_zyx for scroll coordinates",
    files=_saved, minutes=round((time.time() - _T0) / 60, 1)),
    open(os.path.join(CKPT_DIR, "manifest.json"), "w"), indent=1)

print(f"\n[STAGE 1 RESULT] {int((_s_tracks >= MIN_TRACKS).sum())} sheets from {int((sizes >= MIN_TRACKS).sum())} pieces | "
      f"grey-zone pairs (neither joined nor barred): {int(join_table['grey_zone'].sum())}")
print("  biggest sheets: label | points | tracks | parts | stacked (0 = one layer)")
for _r in sheet_table.head(12).itertuples():
    print(f"    {_r.sheet:6d} | {_r.points:8,} | {_r.tracks:6,} | {_r.parts:3d} | {_r.stacked:.2f}")
print(f"  saved to {CKPT_DIR}/ | {(time.time() - _T0) / 60:.0f} min | next: run_layers.py {TILE}")
