
"""phase9_ink_volume_extraction.py - phase 9 (run by run_ink.py): cut 21 slices through the sheet along the flat grid from
phase 8, ready for an ink model (phase 10).

    python phase9_ink_volume_extraction.py 9011 --ckpt ckpt_t1 --shift 5 --step 1

Reads  wrap_<id>_V3.tifxyz, wrap_<id>_mask_V3.tif (phase 8, in the working folder)
Writes wrap_<id>_slices/layers/00.tif ... 20.tif, mask.png, extrusion.json (and wrap_<id>_snapped_xyz.npy)

The scan, the box offset and the scroll centre are read from the stage-1 folder's manifest.json (--ckpt), so the slices come
from the same scan the mesh was built on (any scroll). Without a scan in the manifest, config.py's scan is used.
--core and --offset override the manifest.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import tifffile
from phase1_stream_ct import ScrollStreamer
from config import SCROLL, SCAN, SCAN_SOURCE, LEVEL, scroll_centre

INK9UM = {"native_depths": 21, "pool": 1}

def read_tifxyz_custom(wrap_id: int) -> tuple[np.ndarray, np.ndarray]:
    xyz_path = f"wrap_{wrap_id}_V3.tifxyz"
    mask_path = f"wrap_{wrap_id}_mask_V3.tif"

    if not Path(xyz_path).exists() or not Path(mask_path).exists():
        raise FileNotFoundError(f"Missing {xyz_path} or {mask_path}. Run phase 8 (phase8_flat_grid.py) first.")

    xyz = tifffile.imread(xyz_path).astype(np.float32)
    mask = tifffile.imread(mask_path)
    valid = mask > 0
    xyz[~valid] = np.nan
    return xyz, valid

def open_scan(man: dict) -> ScrollStreamer:
    """the scan the stage-1 folder was built from (manifest.json), else the one in config.py"""
    s = man.get("scan") or {}
    if s.get("volume") or s.get("source"):
        print(f"Scan from the manifest: {s.get('source') or (str(s.get('scroll')) + '/' + str(s.get('volume')))} (level {s.get('level', LEVEL)})")
        return ScrollStreamer(scroll_name=s.get("scroll", SCROLL), scan_folder=s.get("volume", SCAN), level=s.get("level", LEVEL), source=s.get("source"))
    print(f"Scan from config.py: {SCAN_SOURCE or (SCROLL + '/' + SCAN)} (level {LEVEL})")
    return ScrollStreamer(scroll_name=SCROLL, scan_folder=SCAN, level=LEVEL, source=SCAN_SOURCE)

def smooth_surface(xyz: np.ndarray, valid: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    xyz_filled = np.where(valid[..., None], xyz, 0.0)
    xyz_smooth = np.empty_like(xyz_filled)
    for c in range(3):
        xyz_smooth[..., c] = ndi.gaussian_filter(xyz_filled[..., c], sigma=sigma)
    mask_smooth = ndi.gaussian_filter(valid.astype(np.float32), sigma=sigma)
    mask_smooth = np.where(mask_smooth > 1e-6, mask_smooth, 1.0)
    xyz_smoothed = xyz_smooth / mask_smooth[..., None]
    return np.where(valid[..., None], xyz_smoothed, np.nan)

def normals(xyz: np.ndarray, valid: np.ndarray) -> np.ndarray:
    n = np.full(xyz.shape, np.nan, dtype=np.float32)
    if min(valid.shape) < 3:
        return n
    du = (xyz[1:-1, 2:] - xyz[1:-1, :-2]) / 2.0
    dv = (xyz[2:, 1:-1] - xyz[:-2, 1:-1]) / 2.0
    c = np.cross(du, dv)
    length = np.linalg.norm(c, axis=-1, keepdims=True)
    ok = (valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1] & (length[..., 0] > 1e-9))
    inner = np.where(ok[..., None], c / np.where(length > 1e-9, length, 1.0), np.nan)
    n[1:-1, 1:-1] = inner
    for (dst, src) in (((0, slice(None)), (1, slice(None))), ((-1, slice(None)), (-2, slice(None))),
                       ((slice(None), 0), (slice(None), 1)), ((slice(None), -1), (slice(None), -2))):
        n[dst] = np.where(valid[dst][..., None] & np.isnan(n[dst]), n[src], n[dst])
    return n

def orient(n: np.ndarray, xyz: np.ndarray, axis: tuple[float, float], toward: str = "core", offset: tuple[int, int, int] = (0, 0, 0)) -> tuple[np.ndarray, float]:
    ok = np.all(np.isfinite(xyz), axis=-1) & np.all(np.isfinite(n), axis=-1)
    if not ok.any(): return n, float("nan")
    p, m = xyz[ok], n[ok]
    cx, cy = axis
    z_off, y_off, x_off = offset
    p_x, p_y = p[:, 0] + x_off, p[:, 1] + y_off
    to_core = np.stack([np.asarray(cx) - p_x, np.asarray(cy) - p_y], axis=-1)
    dots = np.sum(m[:, :2] * to_core, axis=-1)
    share_core = float(np.mean(dots > 0))
    if (share_core < 0.5 if toward == "core" else share_core > 0.5):
        n, dots = -n, -dots
    return n, float(np.mean(dots > 0)) if toward == "core" else float(np.mean(dots < 0))

def extrude_aws(xyz: np.ndarray, valid: np.ndarray, n: np.ndarray, depths: np.ndarray, streamer: ScrollStreamer, offset: tuple[int, int, int] = (0, 0, 0),
                snap: bool = True, wrap_id_val: int = 0) -> np.ndarray:
    depths = np.asarray(depths, dtype=np.float32)
    H, W = xyz.shape[:2]
    z_offset, y_offset, x_offset = offset

    good = valid & np.all(np.isfinite(xyz), axis=-1) & np.all(np.isfinite(n), axis=-1)
    if not np.any(good): raise ValueError("No finite coordinates and surface normals found to extrude!")

    pts = xyz[good]
    norms = n[good]

    pts_min_ext = pts + norms * (depths.min() - 6.0)
    pts_max_ext = pts + norms * (depths.max() + 6.0)

    min_coords = np.minimum(np.nanmin(pts_min_ext, axis=0), np.nanmin(pts_max_ext, axis=0))
    max_coords = np.maximum(np.nanmax(pts_min_ext, axis=0), np.nanmax(pts_max_ext, axis=0))

    x_min = int(np.floor(min_coords[0] - 2)) + x_offset
    x_max = int(np.ceil(max_coords[0] + 3)) + x_offset
    y_min = int(np.floor(min_coords[1] - 2)) + y_offset
    y_max = int(np.ceil(max_coords[1] + 3)) + y_offset
    z_min = int(np.floor(min_coords[2] - 2)) + z_offset
    z_max = int(np.ceil(max_coords[2] + 3)) + z_offset

    print(f"Streaming required volume block: Z[{z_min}:{z_max}] Y[{y_min}:{y_max}] X[{x_min}:{x_max}]")
    volume_chunk = streamer.fetch_chunk(z_min, z_max, y_min, y_max, x_min, x_max)

    if snap:
        print("Snapping floating mesh to the true papyrus peak...")
        search_ds = np.arange(-6.0, 6.5, 0.5)
        prof = np.empty((len(search_ds), H, W), dtype=np.float32)
        for i, d in enumerate(search_ds):
            t_pts = xyz + (n * d)
            coords = [
                np.nan_to_num(((t_pts[..., 2] + z_offset) - z_min).ravel(), nan=-9999.0),
                np.nan_to_num(((t_pts[..., 1] + y_offset) - y_min).ravel(), nan=-9999.0),
                np.nan_to_num(((t_pts[..., 0] + x_offset) - x_min).ravel(), nan=-9999.0)
            ]
            prof[i] = ndi.map_coordinates(volume_chunk, coords, order=1, mode="constant", cval=0.0).reshape(H, W)

        prof = ndi.gaussian_filter1d(prof, 1.5, axis=0)
        prior = np.exp(-0.5 * (search_ds / 3.0)**2)[:, None, None]
        off = np.where(valid, search_ds[np.argmax(prof * prior, axis=0)], 0.0)
        off = ndi.median_filter(off, size=9)
        off = ndi.gaussian_filter(off, sigma=4.0)

        xyz = xyz + n * off[..., None]
        np.save(f"wrap_{wrap_id_val}_snapped_xyz.npy", xyz)

    out = np.empty((len(depths), H, W), dtype=np.float32)
    for i, d in enumerate(depths):
        target_pts = xyz + (n * d)
        coords = [
            np.nan_to_num(((target_pts[..., 2] + z_offset) - z_min).ravel(), nan=-9999.0),
            np.nan_to_num(((target_pts[..., 1] + y_offset) - y_min).ravel(), nan=-9999.0),
            np.nan_to_num(((target_pts[..., 0] + x_offset) - x_min).ravel(), nan=-9999.0)
        ]
        sampled = ndi.map_coordinates(volume_chunk, coords, order=1, mode="constant", cval=0.0).reshape(H, W)
        sampled[~good] = np.nan
        out[i] = sampled

    return out

def pool(stack: np.ndarray, f: int) -> np.ndarray:
    if f == 1: return stack
    d, h, w = (s // f for s in stack.shape)
    s = stack[:d * f, :h * f, :w * f].reshape(d, f, h, f, w, f)
    with np.errstate(all="ignore"): return np.nanmean(s, axis=(1, 3, 5))

def ink9um_depths(native: int = INK9UM["native_depths"], shift: float = 0.0, step: float = 1.0) -> np.ndarray:
    return (np.arange(native, dtype=np.float32) - (native - 1) / 2.0) * step + shift

def to_uint8(stack: np.ndarray, lo: float = 0.0, hi: float = 255.0) -> np.ndarray:
    s = np.nan_to_num(stack, nan=0.0)
    max_val = float(s.max()) if s.size > 0 else 0.0
    if 0.0 < max_val <= 1.5: s = s * 255.0
    elif max_val > 255.0: s = s * (255.0 / max_val)
    elif (lo, hi) != (0.0, 255.0): s = (s - lo) * (255.0 / (hi - lo))
    return np.clip(np.rint(s), 0, 255).astype(np.uint8)

def write_output(out_dir: str, stack: np.ndarray, mask: np.ndarray, agree: float, mode: str, toward: str) -> None:
    from PIL import Image
    out_dir = Path(out_dir)
    (out_dir / "layers").mkdir(parents=True, exist_ok=True)
    width = max(2, len(str(len(stack) - 1)))
    for i, plane in enumerate(stack):
        tifffile.imwrite(out_dir / "layers" / f"{i:0{width}d}.tif", plane)
    Image.fromarray((mask * 255).astype(np.uint8)).save(out_dir / "mask.png")
    record = {"tool": "integrated_phase9_aws", "slices": int(len(stack)), "surface_slice": int(len(stack) // 2), "shape": list(stack.shape[1:]), "normal_agreement": round(float(agree), 4), "mode": mode, "toward": toward}
    (out_dir / "extrusion.json").write_text(json.dumps(record, indent=1))

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 9: cut slices through the sheet along the flat grid, for the ink model")
    parser.add_argument("wrap_id", type=int, help="ID of the wrap to extract")
    parser.add_argument("--ckpt", default="ckpt_t1", help="stage-1 folder the mesh was built from (scan, offset and scroll centre are read from its manifest.json)")
    parser.add_argument("--core", nargs=2, type=float, default=None, metavar=("CX", "CY"), help="scroll centre; default: from the manifest / config.py")
    parser.add_argument("--offset", nargs=3, type=int, default=None, metavar=("Z", "Y", "X"), help="corner of the box the grid is relative to; default: from the manifest")
    parser.add_argument("--shift", type=float, default=.0)
    parser.add_argument("--step", type=float, default=1.0)
    parser.add_argument("--no-snap", action="store_true")
    parser.add_argument("--flip", action="store_true", help="negate normals after orient()")
    parser.add_argument("--normals_from", type=int, default=None, help="Optional wrap_id to borrow normals from")
    args = parser.parse_args(argv)

    mf = os.path.join(args.ckpt, "manifest.json")
    man = json.load(open(mf)) if os.path.exists(mf) else {}
    core = scroll_centre(man, args.core)
    offset = tuple(args.offset) if args.offset is not None else tuple(int(v) for v in man.get("chunk", {}).get("offset_zyx", [0, 0, 0]))
    out_dir = f"wrap_{args.wrap_id}_slices"

    print(f"\n--- [PHASE 9] INK VOLUME EXTRACTION: WRAP {args.wrap_id} (Shift: {args.shift}, Step: {args.step}) | offset {list(offset)} | centre {core} ---")
    xyz, valid = read_tifxyz_custom(args.wrap_id)

    xyz = smooth_surface(xyz, valid, sigma=3.0)

    if args.normals_from is not None:
        xyz_n, valid_n = read_tifxyz_custom(args.normals_from)
        xyz_n = smooth_surface(xyz_n, valid_n, sigma=3.0)
        n = normals(xyz_n, valid_n)
    else:
        n = normals(xyz, valid)

    n, agree = orient(n, xyz, core, toward="core", offset=offset)
    if args.flip:
        n = -n

    native = extrude_aws(xyz, valid, n, ink9um_depths(shift=args.shift, step=args.step), open_scan(man), offset=offset, snap=not args.no_snap, wrap_id_val=args.wrap_id)
    stack = pool(native, INK9UM["pool"])

    mask = np.any(np.isfinite(stack), axis=0) & (np.nan_to_num(stack, nan=0.0).max(axis=0) > 0)
    stack_uint8 = to_uint8(stack)
    write_output(out_dir, stack_uint8, mask, agree, "ink9um", "core")

    print(f"Successfully generated {stack_uint8.shape[0]} slices of size {stack_uint8.shape[1]}x{stack_uint8.shape[2]}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
