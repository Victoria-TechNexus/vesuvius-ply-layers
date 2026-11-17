
"""phase7b_mesh.py - phase 7, step b (run by run_mesh.py): build ONE layer's surface from its own tracks, so folds and
wrinkles are kept. (Was phase6_track_surface.py.)

Reads a stage-1 folder (one tile, ckpt_<tile>/, or the combined folder from phase 7a) and its layer labels, writes the mesh
into wraps/ inside the layer's mesh folder (mesh_<tiles>_layerNN/wraps from config.py, unless --out is given).

    python phase7b_mesh.py --ckpt ckpt_t1 --layers_dir layers_t1 --layer 11          # one layer of one tile
    python phase7b_mesh.py --ckpt mesh_t1-t2_layer11 --layers_dir mesh_t1-t2_layer11/layers --layer 11   # combined by phase 7a
    python phase7b_mesh.py --sheet 722                                                # one stage-1 sheet (for checking)

How it works
  1. The layer's track points are thinned to one point per --node voxels and linked to near neighbours ON THE SAME SURFACE
     (close along the sheet, not across it). Distances are then measured by walking along the sheet, not through the air.
  2. From those walking distances every point gets a flat (u, v) position: the sheet is unrolled first. A fold is just more sheet.
  3. On that flat map, each mesh vertex is fitted to the track points around it (a local straight-line fit, no blurring),
     so the mesh passes through the tracks and keeps wrinkles. Small holes are bridged and marked; big holes stay holes.
  4. The checks print PASS or FAIL against fixed limits (see the EXPECTATIONS block at the end of the output).
"""
import argparse, json, os, sys, time
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from config import OUTPUT_ROOT, mesh_dir, tile_name

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ckpt_t1", help="stage-1 folder: a tile's ckpt_<tile>, or the combined folder from phase 7a")
ap.add_argument("--out", default=None, help="default: mesh_<tile>_layerNN/wraps (or mesh_<tile>_sheetN/wraps) from config.py")
ap.add_argument("--layers_dir", default="layers_t1", help="folder holding layer_id.npy (phase 6d output, or <combined>/layers)")
ap.add_argument("--sheet", type=int, default=None, help="stage-1 sheet label (from --labels in the ckpt folder)")
ap.add_argument("--layer", type=int, default=None, help="layer number (from layer_id.npy in --layers_dir)")
ap.add_argument("--labels", default="sheet_id_split.npy", help="with --sheet: the sheet labels file in the ckpt folder")
ap.add_argument("--node", type=float, default=2.5, help="thinning: one point per this many voxels")
ap.add_argument("--link", type=float, default=6.0, help="link points closer than this along the sheet")
ap.add_argument("--across", type=float, default=4.0, help="...and no further than this across the sheet (spans the thickness of one layer, keeps fold limbs apart)")
ap.add_argument("--landmarks", type=int, default=200)
ap.add_argument("--local", type=float, default=40.0, help="the flat layout is refined so walking distances up to this far (voxels) are kept")
ap.add_argument("--max_stretch", type=float, default=2.5, help="mesh cells stretched more than this (or squashed below 1/this) are cut out and reported")
ap.add_argument("--fit", type=float, default=1.2, help="radius (voxels) used to decide where the sheet has tracks (larger grows the mesh past the edge of the tracks)")
ap.add_argument("--smooth", type=float, default=3.0, help="radius (voxels) over which the band middle is smoothed; removes sharp steps in depth without growing the edges")
ap.add_argument("--base", type=float, default=5.0, help="radius (voxels) of the first, smooth surface through ALL the sheet's tracks")
ap.add_argument("--keep", type=float, default=5.0, help="where the tracks spread wider than one band, keep only tracks within this many voxels of the surface carried over from the clean areas (0 = keep everything, the old behaviour)")
ap.add_argument("--clean_sd", type=float, default=3.0, help="an area counts as clean (one band) when its tracks spread less than this across the sheet (standard deviation, voxels)")
ap.add_argument("--reach", type=float, default=12.0, help="how far across the sheet to look for the papyrus band in the CT")
ap.add_argument("--bridge", type=float, default=6.0, help="holes narrower than about twice this are bridged (and marked)")
ap.add_argument("--step", type=int, default=2, help="mesh vertex spacing in voxels")
ap.add_argument("--compare_old", action="store_true", help="also run the old plane/cylinder mesher on the same points")
a = ap.parse_args()
t0 = time.time()
if not os.path.exists(os.path.join(a.ckpt, "paths.npy")):
    sys.exit(f"Cannot find {a.ckpt}/paths.npy from {os.getcwd()}. Run this from the folder that holds {a.ckpt}/, or pass --ckpt and --out with full paths.")
_mf = os.path.join(a.ckpt, "manifest.json"); TILE = tile_name(a.ckpt, json.load(open(_mf)) if os.path.exists(_mf) else None)
if a.out is None:
    a.out = os.path.join(mesh_dir(TILE, a.layer) if a.layer is not None else os.path.join(OUTPUT_ROOT, f"mesh_{TILE}_sheet{722 if a.sheet is None else a.sheet}"), "wraps")
os.makedirs(a.out, exist_ok=True)

P_all = np.load(os.path.join(a.ckpt, "paths.npy")).astype(np.float64); lens = np.load(os.path.join(a.ckpt, "lens.npy")); tid = np.repeat(np.arange(len(lens)), lens)
if a.layer is not None:
    lab = np.load(os.path.join(a.layers_dir, "layer_id.npy")); pick = lab == a.layer; name = f"layer{a.layer:02d}"
else:
    sid = 722 if a.sheet is None else a.sheet; lab = np.load(os.path.join(a.ckpt, a.labels)); pick = lab == sid; name = str(sid)
P = P_all[pick[tid]]; tid_p = tid[pick[tid]]; del P_all           # (z, y, x), chunk-local voxels; tid_p = track of each point
if len(P) < 1000: sys.exit(f"'{name}' has only {len(P)} points; nothing to mesh.")
normals = np.load(os.path.join(a.ckpt, "normals_f16.npy"), mmap_mode="r"); bright = np.load(os.path.join(a.ckpt, "bright_f16.npy"), mmap_mode="r"); shape = np.array(bright.shape)
print(f"[PHASE 7b - MESH] {a.ckpt} '{name}': {int(pick.sum()):,} tracks, {len(P):,} points | writing {a.out}/")

# ------------------------------------------------------------------ 1. thin the points and link neighbours on the same surface
key = np.floor(P / a.node).astype(np.int64); _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
cnt = np.bincount(inv); Q = np.stack([np.bincount(inv, weights=P[:, k]) / cnt for k in range(3)], 1)       # nodes
ix = np.clip(np.round(Q).astype(int), 0, shape - 1); nq = np.asarray(normals[ix[:, 0], ix[:, 1], ix[:, 2]], dtype=np.float64)
nq /= np.linalg.norm(nq, axis=1, keepdims=True) + 1e-12
pr = cKDTree(Q).query_pairs(a.link, output_type="ndarray"); d = Q[pr[:, 1]] - Q[pr[:, 0]]
okp = (np.abs(np.einsum("ij,ij->i", d, nq[pr[:, 0]])) <= a.across) & (np.abs(np.einsum("ij,ij->i", d, nq[pr[:, 1]])) <= a.across)
pr, w = pr[okp], np.linalg.norm(d[okp], axis=1); M = len(Q)
G = coo_matrix((np.r_[w, w], (np.r_[pr[:, 0], pr[:, 1]], np.r_[pr[:, 1], pr[:, 0]])), shape=(M, M)).tocsr()
nc, comp = connected_components(G, directed=False); big = np.argmax(np.bincount(comp)); keep = comp == big
main_share = float(cnt[keep].sum() / cnt.sum())
print(f"  {M:,} thinned points, {len(pr):,} links | largest connected piece holds {main_share:.1%} of the sheet's points ({nc - 1} smaller pieces left out)")
node_new = -np.ones(M, np.int64); node_new[keep] = np.arange(keep.sum()); Q, nq, G = Q[keep], nq[keep], G[keep][:, keep]; M = len(Q)
pt_node = node_new[inv]; in_main = pt_node >= 0

# ------------------------------------------------------------------ 2. unroll: flat (u, v) from walking distances
lm = [int(np.argmin(np.linalg.norm(Q - Q.mean(0), axis=1)))]
D = np.empty((min(a.landmarks, M), M)); D[0] = dijkstra(G, indices=lm[0]); dmin = D[0].copy()
for k_ in range(1, len(D)):                                        # landmarks spread as far apart as possible
    nxt = int(np.argmax(dmin)); lm.append(nxt); D[k_] = dijkstra(G, indices=nxt); np.minimum(dmin, D[k_], out=dmin)
# (same result as before; the table is filled in place and the nearest-landmark distance is kept as it goes,
#  instead of copying the growing table and re-taking its minimum every round)
L = len(lm); D2 = D[:, lm] ** 2; J = np.eye(L) - 1.0 / L; Bm = -0.5 * J @ D2 @ J
ev, evec = np.linalg.eigh(Bm); o = np.argsort(-ev); ev, evec = ev[o], evec[:, o]
flat3 = float(max(ev[2], 0) / ev[0])                              # 0 = unrolls perfectly flat
UV = (-0.5 * (evec[:, :2] / np.sqrt(ev[:2])).T @ (D ** 2 - D2.mean(1, keepdims=True))).T                   # (M, 2)
# Refine the flat layout. The first layout comes from far-apart landmarks and crumples sheets that are not a simple
# rectangle (arms, notches, holes), because walking around a notch is longer than the straight line on the flat map.
# Here every point is moved so that its walking distance to points within --local voxels is kept; far distances are ignored.
ck = np.unique(np.floor(Q / 8.0).astype(np.int64), axis=0, return_index=True)[1]           # one source every ~8 voxels
pi_, pj_, pd_ = [], [], []
tq_ = cKDTree(Q)
for s_ in range(0, len(ck), 256):
    # A walk of at most --local voxels never leaves the ball of that radius around its start, so each batch only needs
    # the nodes within --local of its sources. Same distances as searching the whole sheet, but the work per batch no
    # longer grows with the size of the sheet (this was the part that grew with the square of the size).
    src = ck[s_:s_ + 256]
    near = np.unique(np.concatenate(tq_.query_ball_point(Q[src], a.local + 1e-6)))
    Dl = dijkstra(G[near][:, near], indices=np.searchsorted(near, src), limit=a.local)
    ii, jj = np.nonzero(np.isfinite(Dl) & (Dl > 0))
    pi_.append(src[ii]); pj_.append(near[jj]); pd_.append(Dl[ii, jj])
del tq_
r0, c0_ = G.nonzero(); pi_.append(r0); pj_.append(c0_); pd_.append(np.asarray(G[r0, c0_]).ravel())
pi_, pj_, pd_ = np.concatenate(pi_), np.concatenate(pj_), np.concatenate(pd_)
si, sj, sd_ = np.r_[pi_, pj_], np.r_[pj_, pi_], np.r_[pd_, pd_]; sw = 1.0 / sd_ ** 2; wsum = np.bincount(si, weights=sw, minlength=M)
def layout_error(U): return float(np.median(np.abs(np.linalg.norm(U[pi_] - U[pj_], axis=1) - pd_) / pd_))
err0 = layout_error(UV)
for it in range(150):
    dv = UV[si] - UV[sj]; ln = np.linalg.norm(dv, axis=1) + 1e-9; tgt = UV[sj] + dv * (sd_ / ln)[:, None]
    UV = np.stack([np.bincount(si, weights=sw * tgt[:, k], minlength=M) for k in range(2)], 1) / wsum[:, None]
print(f"  flat layout refined over {len(pd_):,} local distances: typical error {err0:.1%} -> {layout_error(UV):.1%}")
# sub-node accuracy: local map from 3D offsets to (u, v) offsets at every node, from its links
r, c = G.nonzero(); dX = Q[c] - Q[r]; dU = UV[c] - UV[r]
A_ = np.zeros((M, 3, 3)); B_ = np.zeros((M, 3, 2)); np.add.at(A_, r, dX[:, :, None] * dX[:, None, :]); np.add.at(B_, r, dX[:, :, None] * dU[:, None, :])
A_ += (1e-3 * np.trace(A_, axis1=1, axis2=2) + 1e-9)[:, None, None] * np.eye(3); Tm = np.linalg.solve(A_, B_)   # (M, 3, 2)
Pm = P[in_main]; tid_m = tid_p[in_main]; nd = pt_node[in_main]; uv_p = UV[nd] + np.einsum("ij,ijk->ik", Pm - Q[nd], Tm[nd])
print(f"  unrolled with {L} landmarks | flat size {np.ptp(uv_p[:, 0]):.0f} x {np.ptp(uv_p[:, 1]):.0f} voxels | third-direction share {flat3:.3f} (0 = unrolls flat)")

# ------------------------------------------------------------------ 3. surface: local straight-line fit of (x, y, z) over the flat map
pad = int(3 * a.bridge) + 4; u0, v0 = uv_p.min(0) - pad; W_ = int(np.ptp(uv_p[:, 0])) + 2 * pad + 2; H_ = int(np.ptp(uv_p[:, 1])) + 2 * pad + 2
pu, pv = uv_p[:, 0] - u0, uv_p[:, 1] - v0; iu, iv = np.round(pu).astype(int), np.round(pv).astype(int)
def moments(sig, w0=None):
    w0 = np.ones(len(pu)) if w0 is None else np.asarray(w0, dtype=np.float64)     # optional weight per track point
    def img(wt): return ndi.gaussian_filter(np.bincount(iv * W_ + iu, weights=wt * w0, minlength=H_ * W_).reshape(H_, W_), sig, mode="constant")
    S = {k: img(wt) for k, wt in (("1", np.ones(len(pu))), ("u", pu), ("v", pv), ("uu", pu * pu), ("uv", pu * pv), ("vv", pv * pv))}
    for k in range(3): S[f"c{k}"] = img(Pm[:, k]); S[f"c{k}u"] = img(Pm[:, k] * pu); S[f"c{k}v"] = img(Pm[:, k] * pv)
    return S
def fit(S):
    """Value of the local straight-line fit at each pixel centre; returns (H, W, 3) positions and the support image."""
    V_, U_ = np.mgrid[0:H_, 0:W_].astype(np.float64); s1 = S["1"]
    su, sv = S["u"] - U_ * s1, S["v"] - V_ * s1
    suu = S["uu"] - 2 * U_ * S["u"] + U_ * U_ * s1; svv = S["vv"] - 2 * V_ * S["v"] + V_ * V_ * s1; suv = S["uv"] - U_ * S["v"] - V_ * S["u"] + U_ * V_ * s1
    lam = 0.05 * s1 + 1e-9                                        # keeps the fit stable where points are few or in a line
    Mx = np.stack([np.stack([s1 + 1e-12, su, sv], -1), np.stack([su, suu + lam, suv], -1), np.stack([sv, suv, svv + lam], -1)], -2)
    out = np.zeros((H_, W_, 3))
    for k in range(3):
        rhs = np.stack([S[f"c{k}"], S[f"c{k}u"] - U_ * S[f"c{k}"], S[f"c{k}v"] - V_ * S[f"c{k}"]], -1)
        out[..., k] = np.linalg.solve(Mx, rhs[..., None])[..., 0, 0]
    return out, s1
# A stage-1 sheet is not a thin surface: its tracks run at several depths through the thickness of one papyrus layer.
# So: (a) a smooth surface through all of them, (b) across that surface, find the papyrus band in the CT itself,
# (c) put the mesh at the middle of the band. Wrinkles then come from the CT, not from which track happens to be nearest.
_, s1 = fit(moments(a.fit)); has = s1 >= 0.08
S0, sb = fit(moments(a.base))
near = ndi.distance_transform_edt(~has) <= a.bridge
closed = ndi.binary_closing(has, structure=np.ones((3, 3), bool), iterations=int(a.bridge)) | has
bridged = (~has) & near & closed & (sb >= 0.01); valid = has | bridged
N0 = np.cross(np.gradient(S0, axis=1), np.gradient(S0, axis=0)); N0 /= np.linalg.norm(N0, axis=-1, keepdims=True) + 1e-12
w_pt = np.einsum("ij,ij->i", Pm - S0[iv, iu], N0[iv, iu])                       # each track point's depth across the sheet
def smooth_img(wt, sig): return ndi.gaussian_filter(np.bincount(iv * W_ + iu, weights=wt, minlength=H_ * W_).reshape(H_, W_), sig, mode="constant")
c0 = np.maximum(smooth_img(np.ones(len(pu)), 3.0), 1e-9); m_img = smooth_img(w_pt, 3.0) / c0
sd_img = np.sqrt(np.maximum(smooth_img(w_pt ** 2, 3.0) / c0 - m_img ** 2, 0.0)); half = np.clip(2.5 * sd_img + 2.0, 4.0, 10.0)
# Where the layer's tracks spread wider than one band (a neighbouring layer is mixed in, or the two are pressed together),
# the smooth surface above gets pulled between the two. So: (a) find the CLEAN areas, where the tracks form one band;
# (b) rebuild the smooth surface from the clean areas only, carried across the unclean areas as a straight continuation;
# (c) in the unclean areas keep only the tracks within --keep voxels of that continuation, and set the rest aside.
kept_pt = np.ones(len(w_pt), bool); clean = valid.copy(); clean_share = 1.0
if a.keep > 0:
    has_pts = valid & (c0 > 0.02); sd_here = sd_img[has_pts]
    lim = a.clean_sd if (sd_here <= a.clean_sd).mean() >= 0.3 else float(np.percentile(sd_here, 40))      # if little is clean, take the cleanest 40%
    clean = has_pts & (sd_img <= lim); clean_share = float(clean.sum() / max(has_pts.sum(), 1)); wc = clean[iv, iu].astype(np.float64)
    Sc, sc = fit(moments(a.base, wc)); Sf, sf = fit(moments(3.0 * a.base, wc)); dens = float(np.median(sb[has_pts]))
    S0 = np.where((sc >= 0.05 * dens)[..., None], Sc, np.where((sf >= 0.01 * dens)[..., None], Sf, S0))   # near clean areas / further away / no clean area in reach
    N0 = np.cross(np.gradient(S0, axis=1), np.gradient(S0, axis=0)); N0 /= np.linalg.norm(N0, axis=-1, keepdims=True) + 1e-12
    w_pt = np.einsum("ij,ij->i", Pm - S0[iv, iu], N0[iv, iu])
    kept_pt = clean[iv, iu] | (np.abs(w_pt) <= a.keep)
    kf = kept_pt.astype(np.float64); ck_ = np.maximum(smooth_img(kf, 3.0), 1e-9)
    m_img = np.where(ck_ > 0.02, smooth_img(w_pt * kf, 3.0) / ck_, 0.0)
    sd_img = np.sqrt(np.maximum(smooth_img(w_pt ** 2 * kf, 3.0) / ck_ - m_img ** 2, 0.0))
    half = np.clip(2.5 * sd_img + 2.0, 4.0, 10.0); half = np.where(clean, half, np.minimum(half, a.keep + 1.5))
    tr_n = np.bincount(tid_m, minlength=len(lens)); tr_drop = np.bincount(tid_m, weights=(~kept_pt).astype(float), minlength=len(lens))
    aside = np.flatnonzero((tr_n > 0) & (tr_drop > 0.5 * tr_n)); np.save(os.path.join(a.out, f"wrap_{name}_set_aside_tracks.npy"), aside)
    print(f"  one-band check: {clean_share:.1%} of the sheet is clean (tracks spread under {lim:.1f} voxels) | in the rest, track points more than {a.keep:g} voxels "
          f"from the clean layer's continuation were set aside: {1 - kept_pt.mean():.1%} of all points, {len(aside):,} whole tracks (saved as wrap_{name}_set_aside_tracks.npy)")
bright32 = np.asarray(bright, dtype=np.float32)
def tl(vol, q):
    q = np.clip(q, 0, shape - 1 - 1e-6); i0 = np.floor(q).astype(int); f = (q - i0).astype(np.float32); z, y, x = i0.T
    c = lambda dz, dy, dx: vol[z + dz, y + dy, x + dx]
    c00 = c(0, 0, 0) * (1 - f[:, 2]) + c(0, 0, 1) * f[:, 2]; c01 = c(0, 1, 0) * (1 - f[:, 2]) + c(0, 1, 1) * f[:, 2]
    c10 = c(1, 0, 0) * (1 - f[:, 2]) + c(1, 0, 1) * f[:, 2]; c11 = c(1, 1, 0) * (1 - f[:, 2]) + c(1, 1, 1) * f[:, 2]
    return (c00 * (1 - f[:, 1]) + c01 * f[:, 1]) * (1 - f[:, 0]) + (c10 * (1 - f[:, 1]) + c11 * f[:, 1]) * f[:, 0]
KS = np.arange(-a.reach, a.reach + 0.01, 0.5); JJ = np.arange(len(KS))[None, :]
def band(base, nrm, centre, halfw):
    """Along the normal at each position: the bright run of CT nearest `centre`, limited to centre +- halfw.
    Returns its middle, its thickness, whether a band was found, and whether it ran past the limit (touching a neighbour)."""
    mid = np.zeros(len(base)); thick = np.zeros(len(base)); found = np.zeros(len(base), bool); cut = np.zeros(len(base), bool)
    for s_ in range(0, len(base), 20000):
        sl = slice(s_, s_ + 20000); q = base[sl][:, None, :] + KS[None, :, None] * nrm[sl][:, None, :]
        Bt = tl(bright32, q.reshape(-1, 3)).reshape(-1, len(KS)) >= 0.5
        dist = np.where(Bt, np.abs(KS[None, :] - centre[sl][:, None]), 1e9); j0 = dist.argmin(1); ok = dist[np.arange(len(j0)), j0] <= 3.0
        lo_j = np.where(~Bt & (JJ <= j0[:, None]), JJ, -1).max(1) + 1; hi_j = np.where(~Bt & (JJ >= j0[:, None]), JJ, len(KS)).min(1) - 1
        k1, k2 = KS[np.clip(lo_j, 0, len(KS) - 1)], KS[np.clip(hi_j, 0, len(KS) - 1)]
        c1, c2 = np.maximum(k1, centre[sl] - halfw[sl]), np.minimum(k2, centre[sl] + halfw[sl])
        mid[sl] = np.where(ok, 0.5 * (c1 + c2), centre[sl]); thick[sl] = np.where(ok, k2 - k1, 0); found[sl] = ok
        cut[sl] = ok & ((k1 < centre[sl] - halfw[sl]) | (k2 > centre[sl] + halfw[sl]))
    return mid, thick, found, cut
vi, vu = np.nonzero(valid); mid_v, thick_v, found_v, cut_v = band(S0[vi, vu], N0[vi, vu], m_img[vi, vu], half[vi, vu])
C = np.zeros((H_, W_)); C[vi, vu] = mid_v; C = ndi.median_filter(C, size=3)
C = ndi.gaussian_filter(np.where(valid, C, 0), a.smooth) / np.maximum(ndi.gaussian_filter(valid.astype(float), a.smooth), 1e-6)
surf = S0 + C[..., None] * N0
print(f"  sheet thickness: the tracks spread {np.percentile(w_pt, 5):+.1f} to {np.percentile(w_pt, 95):+.1f} voxels across the sheet (5%-95%) | "
      f"CT band found under {found_v.mean():.1%} of the surface, thickness p10/p50/p90 {np.percentile(thick_v[found_v], [10, 50, 90]).round(1).tolist()} voxels | "
      f"band runs into a neighbour (pressed): {cut_v.mean():.1%}")
# mesh: one vertex every --step pixels
sub = valid[::a.step, ::a.step]; vx = surf[::a.step, ::a.step]; br = bridged[::a.step, ::a.step]
quad = sub[:-1, :-1] & sub[:-1, 1:] & sub[1:, :-1] & sub[1:, 1:]
_e1 = np.linalg.norm(vx[:-1, 1:] - vx[:-1, :-1], axis=-1) / a.step; _e2 = np.linalg.norm(vx[1:, :-1] - vx[:-1, :-1], axis=-1) / a.step
_e3 = np.linalg.norm(vx[1:, 1:] - vx[:-1, :-1], axis=-1) / (a.step * np.sqrt(2)); _e4 = np.linalg.norm(vx[1:, :-1] - vx[:-1, 1:], axis=-1) / (a.step * np.sqrt(2))
_mx = np.maximum.reduce([_e1, _e2, _e3, _e4]); _mn = np.minimum.reduce([_e1, _e2, _e3, _e4])
bad_cell = quad & ((_mx > a.max_stretch) | (_mn < 1.0 / a.max_stretch)); cut_share = float(bad_cell.sum() / max(quad.sum(), 1)); quad &= ~bad_cell
labq, nqd = ndi.label(quad)
if nqd == 0: sys.exit("no surface could be built (no connected area on the flat map)")
bigq = labq == (np.argmax(np.bincount(labq.ravel())[1:]) + 1)
vid = -np.ones(sub.shape, np.int64); rr, cc = np.nonzero(bigq); used = np.zeros(sub.shape, bool)
for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)): used[rr + dr, cc + dc] = True
vid[used] = np.arange(used.sum()); V_zyx = vx[used]; interp_v = br[used]
uv_v = np.stack([np.nonzero(used)[1] * a.step + u0, np.nonzero(used)[0] * a.step + v0], 1).astype(np.float64)
q00, q01, q10, q11 = vid[rr, cc], vid[rr, cc + 1], vid[rr + 1, cc], vid[rr + 1, cc + 1]
F = np.concatenate([np.stack([q00, q01, q10], 1), np.stack([q01, q11, q10], 1)])
V_xyz = V_zyx[:, ::-1]                                            # the mesh is written in (x, y, z), like the old Phase 6
obj = os.path.join(a.out, f"wrap_{name}_mesh_final.obj")
with open(obj, "w") as f:
    np.savetxt(f, V_xyz, fmt="v %.4f %.4f %.4f"); np.savetxt(f, F + 1, fmt="f %d %d %d")
np.save(obj[:-4] + "_interp.npy", np.c_[V_xyz, interp_v.astype(float)]); np.save(obj[:-4] + "_uv.npy", uv_v)
print(f"  mesh: {len(V_xyz):,} vertices, {len(F):,} triangles -> {obj}")

# ------------------------------------------------------------------ 4. checks
def topo(F, nV):
    E = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1); ue, ce = np.unique(E, axis=0, return_counts=True); b = ue[ce == 1]
    loops = 0
    if len(b):
        _, labb = connected_components(coo_matrix((np.ones(len(b)), (b[:, 0], b[:, 1])), shape=(nV, nV)), directed=False); loops = len(np.unique(labb[np.unique(b)]))
    return loops, int((ce > 2).sum()), len(np.unique(F)) - len(ue) + len(F)
loops, nonman, chi = topo(F, len(V_xyz))
Pk = Pm[kept_pt]; rs = np.random.default_rng(0); samp = Pk[rs.choice(len(Pk), min(6000, len(Pk)), replace=False)]     # kept track points only
def dist_to_mesh(Vz, Fz, pts):
    try:
        import trimesh
        return trimesh.proximity.closest_point(trimesh.Trimesh(Vz, Fz, process=False), pts)[1], "exact"
    except Exception:
        return cKDTree(Vz).query(pts)[0], "to nearest vertex (trimesh/rtree not available; reads about 0.7 voxel high)"
d_pm, how = dist_to_mesh(V_zyx, F, samp)
d_vp = cKDTree(Pm[::3]).query(V_zyx)[0]                           # vertex -> nearest track point (is the mesh inventing surface?)
ixv = np.clip(np.round(V_zyx).astype(int), 0, shape - 1); bv = np.asarray(bright[ixv[:, 0], ixv[:, 1], ixv[:, 2]], dtype=np.float32) >= 0.5
e = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
stretch = np.linalg.norm(V_zyx[e[:, 0]] - V_zyx[e[:, 1]], axis=1) / (np.linalg.norm(uv_v[e[:, 0]] - uv_v[e[:, 1]], axis=1) + 1e-12)
# each track point against the surface at its own flat position, measured across the sheet
ok_pix = valid[iv, iu]; Nf = np.cross(np.gradient(surf, axis=1), np.gradient(surf, axis=0)); Nf /= np.linalg.norm(Nf, axis=-1, keepdims=True) + 1e-12
off_pt = np.einsum("ij,ij->i", Pm[ok_pix] - surf[iv[ok_pix], iu[ok_pix]], Nf[iv[ok_pix], iu[ok_pix]])
res = np.linalg.norm(surf[iv[ok_pix], iu[ok_pix]] - Pm[ok_pix], axis=1); mixed = float((res > 8.0).mean())
kp = kept_pt[ok_pix]; off_keep = off_pt[kp]; mixed_keep = float((res[kp] > 8.0).mean()); aside_share = float(1.0 - kept_pt.mean())
# independent re-measurement: from the finished mesh, where is the middle of the CT band along the mesh's own normal?
def vertex_normals(Vz, Fz):
    fn_ = np.cross(Vz[Fz[:, 1]] - Vz[Fz[:, 0]], Vz[Fz[:, 2]] - Vz[Fz[:, 0]]); vn = np.zeros_like(Vz)
    for k in range(3): np.add.at(vn, Fz[:, k], fn_)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)
def band_offset(Vz, Fz):
    mo, th, fo, cu = band(Vz, vertex_normals(Vz, Fz), np.zeros(len(Vz)), np.full(len(Vz), 10.0)); return mo[fo & ~cu], float(fo.mean())
boff, bfound = band_offset(V_zyx, F)
# shape: how much does the surface turn, and could a single flat or curved height map have held it?
fn = np.cross(V_zyx[F[:, 1]] - V_zyx[F[:, 0]], V_zyx[F[:, 2]] - V_zyx[F[:, 0]]); fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
n_mean = np.linalg.eigh(fn.T @ fn)[1][:, -1]; ang = np.degrees(np.arccos(np.clip(np.abs(fn @ n_mean), 0, 1)))
e1 = np.cross(n_mean, np.eye(3)[np.argmin(np.abs(n_mean))]); e1 /= np.linalg.norm(e1); e2 = np.cross(n_mean, e1)
cell = np.floor(np.c_[V_zyx @ e1, V_zyx @ e2] / 3.0).astype(np.int64); cellid = cell[:, 0] * 100003 + cell[:, 1]; depth = V_zyx @ n_mean
o = np.lexsort((depth, cellid)); twice = np.zeros(len(o), bool); twice[1:] = (cellid[o][1:] == cellid[o][:-1]) & (np.diff(depth[o]) >= 4.0)
fold_share = float(len(np.unique(cellid[o][twice])) / len(np.unique(cellid)))
# wrinkles: how far the mesh departs from the smooth first surface
wr_mesh = float(np.std(C[valid]))

old = None
if a.compare_old:
    try:
        sys.path.insert(0, os.getcwd()); import phase6_continuous_2_manifold_mesh as p6
        pts_xyz = Pm[:, ::-1].astype(np.float32); ixp = np.clip(np.round(Pm).astype(int), 0, shape - 1)
        nr = np.asarray(normals[ixp[:, 0], ixp[:, 1], ixp[:, 2]], dtype=np.float32)[:, ::-1]
        m_old = p6.patch_to_grid_mesh(pts_xyz, nr, p6.MeshConfig()) or p6.patch_to_cylinder_mesh(pts_xyz, nr, p6.MeshConfig())
        if m_old is None: old = "old mesher: produced NO mesh for these points"
        else:
            do, _ = dist_to_mesh(np.asarray(m_old.vertices)[:, ::-1], np.asarray(m_old.faces), samp)
            ob, of_ = band_offset(np.asarray(m_old.vertices)[:, ::-1].astype(np.float64), np.asarray(m_old.faces))
            ixo = np.clip(np.round(np.asarray(m_old.vertices)[:, ::-1]).astype(int), 0, shape - 1); bo = np.asarray(bright[ixo[:, 0], ixo[:, 1], ixo[:, 2]], dtype=np.float32) >= 0.5
            old = (f"old mesher on the same points: {len(m_old.vertices):,} vertices | distance to the middle of the CT band p50 {np.percentile(np.abs(ob), 50):.2f}, p95 {np.percentile(np.abs(ob), 95):.2f} "
                   f"| on papyrus {bo.mean():.1%} | track -> mesh p50 {np.percentile(do, 50):.2f}, p95 {np.percentile(do, 95):.2f}")
    except Exception as ex:
        old = f"old mesher comparison skipped ({ex})"

p50, p95, p99 = np.percentile(d_pm, [50, 95, 99]); T = []
def chk(label, value, passed, limit): T.append((label, value, "PASS" if passed else "FAIL", limit))
ab = np.abs(boff); width = float(np.percentile(off_pt, 95) - np.percentile(off_pt, 5))
chk("1 mesh runs down the middle of the papyrus band (mesh -> band middle, re-measured from the finished mesh, voxels)", f"p50 {np.percentile(ab, 50):.2f} | p95 {np.percentile(ab, 95):.2f} | band found under {bfound:.1%} of vertices", np.percentile(ab, 50) <= 0.5 and np.percentile(ab, 95) <= 2.0 and bfound >= 0.9, "p50 <= 0.5, p95 <= 2.0, found >= 90%")
chk("2 whole sheet is covered", f"{main_share:.1%} in one connected piece | {(d_pm <= 6).mean():.1%} of those within 6 voxels of the mesh", main_share >= 0.90 and (d_pm <= 6).mean() >= 0.97, "piece >= 90%, within 6 voxels >= 97%")
chk("3 mesh does not invent surface", f"bridged vertices {interp_v.mean():.1%} | vertex -> nearest track p95 {np.percentile(d_vp, 95):.1f}, worst 1% {np.percentile(d_vp, 99):.1f} voxels", interp_v.mean() <= 0.25 and np.percentile(d_vp, 99) <= a.bridge + 4, f"bridged <= 25%, worst 1% <= {a.bridge + 4:g}")
chk("4 one clean surface", f"non-manifold edges {nonman} | boundary loops {loops} (1 outer + {max(loops - 1, 0)} holes)", nonman == 0, "non-manifold edges = 0 (holes are reported, not failed)")
chk("5 mesh sits on papyrus in the CT", f"bright under {bv.mean():.1%} of vertices (fitted {bv[~interp_v].mean():.1%}, bridged {bv[interp_v].mean() if interp_v.any() else float('nan'):.1%})", bv[~interp_v].mean() >= 0.90, "fitted vertices >= 90% bright")
kept_share = float(bigq.sum() / max(quad.sum() + bad_cell.sum(), 1))
chk("6 unrolling does not stretch", f"3D length / flat length of mesh edges: p5 {np.percentile(stretch, 5):.3f} | p50 {np.percentile(stretch, 50):.3f} | p95 {np.percentile(stretch, 95):.3f} | cells cut out for stretch: {cut_share:.1%} | surface kept in the final piece: {kept_share:.1%}", np.percentile(stretch, 5) >= 0.90 and np.percentile(stretch, 95) <= 1.10 and cut_share <= 0.03 and kept_share >= 0.95, "p5 >= 0.90, p95 <= 1.10, cut <= 3%, kept >= 95%")
width_keep = float(np.percentile(off_keep, 95) - np.percentile(off_keep, 5))
chk("7 the mesh follows ONE layer", f"kept tracks sit {np.percentile(off_keep, 5):+.1f} to {np.percentile(off_keep, 95):+.1f} voxels from the mesh (5%-95%, width {width_keep:.1f}), further than 8 voxels: {mixed_keep:.2%} | "
    f"set aside as a neighbouring layer: {aside_share:.1%} of points | all tracks, before setting aside: width {width:.1f}, further than 8 voxels {mixed:.2%}",
    width_keep <= 12.0 and mixed_keep <= 0.02 and aside_share <= 0.20, "kept: width <= 12 and <= 2% beyond 8; set aside <= 20%")
chk("8 mesh keeps its own shape (not just the smooth first surface)", f"mesh departs from the smooth surface by {wr_mesh:.2f} voxels (std); tracks' own depth spread {np.std(w_pt):.2f}", True, "reported; judge wrinkles in the picture")
print("\n[EXPECTATIONS]  (distance measured " + how + ")")
for label, value, verdict, limit in T: print(f"  {verdict} | {label}: {value}   [limit: {limit}]")
print(f"  INFO | shape: surface turns up to {np.percentile(ang, 99):.0f} deg from its average direction | area a single height map could NOT hold (surface passes twice): {fold_share:.1%}")
if old: print("  INFO | " + old)
print(f"  RESULT: {sum(v == 'PASS' for _, _, v, _ in T)} of {len(T)} passed | {time.time() - t0:.0f}s")
json.dump(dict(name=name, ckpt=a.ckpt, layers_dir=a.layers_dir, settings=vars(a), vertices=int(len(V_xyz)), triangles=int(len(F)),
               checks=[dict(check=l, value=v, verdict=r, limit=m) for l, v, r, m in T], fold_share=fold_share, old=old), open(obj[:-4] + "_report.json", "w"), indent=1)

# ------------------------------------------------------------------ picture: CT slices with the tracks and the mesh
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    vol = np.asarray(np.load(os.path.join(a.ckpt, "vol_f16.npy"), mmap_mode="r"), dtype=np.float32); S_all = surf[valid]; B_all = bridged[valid]
    zs = np.percentile(Pm[:, 0], [20, 50, 80]).astype(int); fig, axs = plt.subplots(1, 3, figsize=(27, 9.5))
    for ax, z in zip(axs, zs):
        ax.imshow(vol[z], cmap="gray"); k = (np.abs(Pm[:, 0] - z) <= 0.6) & kept_pt; ax.plot(Pm[k, 2], Pm[k, 1], ".", color="cyan", ms=2.5, label="tracks")
        k = (np.abs(Pm[:, 0] - z) <= 0.6) & ~kept_pt; ax.plot(Pm[k, 2], Pm[k, 1], ".", color="magenta", ms=2.5, label="set aside")
        k = (np.abs(S_all[:, 0] - z) <= 0.6) & ~B_all; ax.plot(S_all[k, 2], S_all[k, 1], ".", color="red", ms=1.2, label="mesh")
        k = (np.abs(S_all[:, 0] - z) <= 0.6) & B_all; ax.plot(S_all[k, 2], S_all[k, 1], ".", color="yellow", ms=1.2, label="mesh, bridged")
        lo_, hi_ = Pm[np.abs(Pm[:, 0] - z) <= 3][:, 1:].min(0) - 15, Pm[np.abs(Pm[:, 0] - z) <= 3][:, 1:].max(0) + 15
        ax.set_xlim(max(lo_[1], 0), min(hi_[1], shape[2])); ax.set_ylim(min(hi_[0], shape[1]), max(lo_[0], 0))
        ax.set_title(f"{TILE} {name}, z = {z} | cyan = tracks kept, magenta = tracks set aside, red = mesh, yellow = bridged"); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); png = obj[:-4] + "_check.png"; plt.savefig(png, dpi=100, bbox_inches="tight"); plt.close(); print(f"  wrote {png}")
except Exception as ex:
    print(f"  (picture skipped: {ex})")
