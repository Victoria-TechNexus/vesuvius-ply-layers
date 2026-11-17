
"""phase8_flat_grid.py - phase 8 (run by run_ink.py): turn one layer's mesh from phase 7b straight into the flat grid the ink
steps read. (Was mesh_to_grid.py.)

The mesher already builds its mesh on a flat layout, one vertex every 2 voxels, so the flat grid is that layout filled in to
1-voxel pitch. No second flattening.

    python phase8_flat_grid.py --name layer11 --id 9011                         # mesh of ckpt_t1's layer 11 (mesh_t1_layer11/wraps)
    python phase8_flat_grid.py --name layer11 --id 9011 --ckpt mesh_t1-t2_layer11   # a layer combined by phase 7a
    then (run_ink.py does this):  python phase9_ink_volume_extraction.py 9011 --core 4087 4087 --offset 15534 4794 3895 --shift 5 --step 1

Reads  <wraps>/wrap_<name>_mesh_final.obj, ..._interp.npy, ..._uv.npy
       (wraps defaults to the mesh folder from config.py: mesh_<tile>_layerNN/wraps, as phase 7b writes it)
Writes wrap_<id>_V3.tifxyz, wrap_<id>_mask_V3.tif, wrap_<id>_footprint_V3.tif (in the working folder, where phase 9 looks)
       and diagnostics/grid_<id>_stretch.png
The scroll centre comes from the stage-1 folder's manifest.json, else config.py; --core overrides it.
"""
import argparse, json, os, re, sys
import numpy as np, tifffile
from scipy import ndimage as ndi
from config import OUTPUT_ROOT, mesh_dir, scroll_centre, tile_name
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, help="mesh name as written by the mesher, e.g. layer11 or 722")
ap.add_argument("--id", type=int, required=True, help="number for the output files, e.g. 9011")
ap.add_argument("--ckpt", default="ckpt_t1", help="stage-1 folder the mesh was built from (a tile's ckpt_<tile>, or a combined folder from phase 7a)")
ap.add_argument("--wraps", default=None, help="folder holding the mesh; default: mesh_<tile>_layerNN/wraps from config.py")
ap.add_argument("--out", default="diagnostics")
ap.add_argument("--fill_holes", type=int, default=400, help="enclosed holes up to this many pixels are filled from their rim (0 = leave all holes)")
ap.add_argument("--core", nargs=2, type=float, default=None, metavar=("CX", "CY"), help="scroll centre; default: from the manifest / config.py")
ap.add_argument("--up", choices=["low_z", "high_z"], default="high_z", help="which end of the scroll axis goes at the top of every picture (same for all layers, so pictures never flip); high_z = writing upright for PHerc. Paris 4 (checked against the official segment, whose labels were drawn upright)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
man = json.load(open(os.path.join(a.ckpt, "manifest.json"))) if os.path.exists(os.path.join(a.ckpt, "manifest.json")) else {}
a.core = scroll_centre(man, a.core); TILE = tile_name(a.ckpt, man)
if a.wraps is None:
    m_ = re.fullmatch(r"layer(\d+)", a.name)
    a.wraps = os.path.join(mesh_dir(TILE, int(m_.group(1))) if m_ else os.path.join(OUTPUT_ROOT, f"mesh_{TILE}_sheet{a.name}"), "wraps")
base = os.path.join(a.wraps, f"wrap_{a.name}_mesh_final")
for f in (base + ".obj", base + "_interp.npy", base + "_uv.npy"):
    if not os.path.exists(f): sys.exit(f"Cannot find {f} from {os.getcwd()}. Run phase7b_mesh.py first (or pass --wraps).")
V = np.load(base + "_interp.npy")[:, :3].astype(np.float64)       # (x, y, z), chunk-local
uv = np.load(base + "_uv.npy")
F = np.array([[int(t) for t in l.split()[1:4]] for l in open(base + ".obj") if l.startswith("f ")], dtype=np.int64) - 1
du = np.unique(np.round(np.diff(np.unique(uv[:, 0])), 6)); step = int(round(float(du.min())))
col = np.round((uv[:, 0] - uv[:, 0].min()) / step).astype(int); row = np.round((uv[:, 1] - uv[:, 1].min()) / step).astype(int)
R, C = row.max() + 1, col.max() + 1; G = np.full((R, C, 3), np.nan); G[row, col] = V
cell = np.zeros((R - 1, C - 1), bool); cell[row[F].min(1), col[F].min(1)] = True          # the mesh's own cells (cut-out cells stay out)

# fill each cell in to 1-voxel pitch (straight-line blend of its four corners)
H, W = (R - 1) * step + 1, (C - 1) * step + 1; grid = np.full((H, W, 3), -1.0, np.float32); mask = np.zeros((H, W), bool)
rr, cc = np.nonzero(cell); p00, p01, p10, p11 = G[rr, cc], G[rr, cc + 1], G[rr + 1, cc], G[rr + 1, cc + 1]
for i in range(step + 1):
    for j in range(step + 1):
        fi, fj = i / step, j / step
        grid[rr * step + i, cc * step + j] = ((1 - fi) * ((1 - fj) * p00 + fj * p01) + fi * ((1 - fj) * p10 + fj * p11)).astype(np.float32)
        mask[rr * step + i, cc * step + j] = True

# small enclosed holes (cells the mesher cut out) are filled from their rim, so the ink model does not see black specks
footprint = mask.copy(); n_filled = 0
if a.fill_holes > 0:
    holes = ndi.binary_fill_holes(mask) & ~mask; lab, nh = ndi.label(holes)
    if nh:
        small = np.isin(lab, np.flatnonzero(np.bincount(lab.ravel()) <= a.fill_holes)[1:]) & holes
        _, (ri, ci) = ndi.distance_transform_edt(~mask, return_indices=True)
        grid[small] = grid[ri[small], ci[small]]; mask = mask | small; n_filled = int(small.sum())

# reading side: the image should be seen from the scroll centre looking outward, or the text comes out mirrored
off = man.get("chunk", {}).get("offset_zyx", [0, 0, 0])
ok = mask[1:-1, 1:-1] & mask[1:-1, 2:] & mask[1:-1, :-2] & mask[2:, 1:-1] & mask[:-2, 1:-1]
gu = (grid[1:-1, 2:] - grid[1:-1, :-2]).astype(np.float64) / 2; gv = (grid[2:, 1:-1] - grid[:-2, 1:-1]).astype(np.float64) / 2
nrm = np.cross(gu, gv)[ok]; pc = grid[1:-1, 1:-1][ok].astype(np.float64)
outward = np.c_[pc[:, 0] + off[2] - a.core[0], pc[:, 1] + off[1] - a.core[1]]
share_out = float((np.einsum("ij,ij->i", nrm[:, :2], outward) > 0).mean()); mirrored = share_out < 0.5
if mirrored: grid, mask, footprint = grid[:, ::-1].copy(), mask[:, ::-1].copy(), footprint[:, ::-1].copy()
# fixed "up": the scroll axis always runs down the image, with the same end of it at the top, so pictures never come out
# turned. Only quarter turns and half turns are used: they never mirror, so the reading side fixed above is kept.
def zslope(g, m, axis):                                            # how z changes per pixel along rows (axis 0) or columns (axis 1)
    zi = np.where(m, g[..., 2], np.nan); return float(np.nanmedian(np.diff(zi, axis=axis)))
turned = []
if abs(zslope(grid, mask, 1)) > abs(zslope(grid, mask, 0)):        # the axis runs across: a quarter turn puts it down the image
    grid, mask, footprint = np.rot90(grid).copy(), np.rot90(mask).copy(), np.rot90(footprint).copy(); turned.append("quarter turn")
want_down = 1.0 if a.up == "low_z" else -1.0
if np.sign(zslope(grid, mask, 0)) != want_down:                    # wrong end at the top: a half turn
    grid, mask, footprint = grid[::-1, ::-1].copy(), mask[::-1, ::-1].copy(), footprint[::-1, ::-1].copy(); turned.append("half turn")
H, W = mask.shape
# stretch of the grid: one pixel should be one voxel on the surface, in both directions
ok = footprint[:, 1:] & footprint[:, :-1]; su = np.linalg.norm(grid[:, 1:] - grid[:, :-1], axis=-1)[ok]
ok2 = footprint[1:, :] & footprint[:-1, :]; sv = np.linalg.norm(grid[1:, :] - grid[:-1, :], axis=-1)[ok2]
tifffile.imwrite(f"wrap_{a.id}_V3.tifxyz", grid, photometric="rgb")
tifffile.imwrite(f"wrap_{a.id}_mask_V3.tif", (mask * 255).astype(np.uint8)); tifffile.imwrite(f"wrap_{a.id}_footprint_V3.tif", (footprint * 255).astype(np.uint8))
p = lambda x: np.percentile(x, [5, 50, 95]).round(2).tolist()
print(f"[PHASE 8 - FLAT GRID] {base}.obj -> wrap_{a.id}_V3.tifxyz | {H} x {W} pixels, {int(mask.sum()):,} on the sheet ({mask.mean():.0%} of the rectangle) | small holes filled: {n_filled:,} pixels")
print(f"  surface distance per pixel (should be 1.0): across p5/p50/p95 {p(su)} | down p5/p50/p95 {p(sv)}")
good = (np.percentile(su, 5) >= 0.8) and (np.percentile(su, 95) <= 1.25) and (np.percentile(sv, 5) >= 0.8) and (np.percentile(sv, 95) <= 1.25)
print(f"  {'PASS' if good else 'FAIL'} | flat grid does not stretch   [limit: 5%-95% within 0.80-1.25 in both directions]")
print(f"  reading side: {'mirrored left-right so it is' if mirrored else 'already'} seen from the scroll centre looking outward (normals outward: {max(share_out, 1 - share_out):.0%})")
zimg = np.where(mask, grid[..., 2], np.nan); dz_across = np.nanmedian(np.abs(np.diff(zimg, axis=1))); dz_down = np.nanmedian(np.abs(np.diff(zimg, axis=0)))
print(f"  up: scroll axis runs down the image with {'the LOW' if a.up == 'low_z' else 'the HIGH'} z end at the top (--up {a.up}) | "
      f"{' + '.join(turned) if turned else 'already that way'} | z change per pixel: down {dz_down:.2f}, across {dz_across:.2f}")
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    S = np.full((H, W), np.nan); S[:, 1:][ok] = su; fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    im = ax[0].imshow(S, cmap="coolwarm", vmin=0.6, vmax=1.4); ax[0].set_title(f"{TILE} {a.name}: surface distance per pixel across (white = 1.0, red = stretched, blue = squashed)"); plt.colorbar(im, ax=ax[0], fraction=0.03)
    im = ax[1].imshow(zimg, cmap="viridis"); ax[1].set_title("z (scroll axis) at each pixel"); plt.colorbar(im, ax=ax[1], fraction=0.03)
    for x in ax: x.set_xticks([]); x.set_yticks([])
    plt.tight_layout(); png = os.path.join(a.out, f"grid_{a.id}_stretch.png"); plt.savefig(png, dpi=90, bbox_inches="tight"); plt.close(); print(f"  wrote {png}")
except Exception as ex:
    print(f"  (picture skipped: {ex})")
