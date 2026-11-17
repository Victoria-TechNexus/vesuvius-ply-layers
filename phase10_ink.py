
"""phase10_ink.py - phase 10 (run by run_ink.py): ink detection on phase 9's slices, with the published 9 um models (rev 2.2)

    python phase10_ink.py PHASE9_OUT CHECKPOINT OUT [--device mps|cuda|cpu] [--sha256 HEX] [--batch-size N]

This is the one step meant to be swapped for another ink model. run_ink.py calls it as
    python <INK_SCRIPT> wrap_<id>_slices <MODEL> wrap_<id>_ink --device <DEVICE>
and then reads wrap_<id>_ink/prediction.tif (one value per pixel of the flat grid; 0..1, or 0..255 as uint8).
A replacement only has to take the same three arguments and --device, and write that file. Set INK_SCRIPT and MODEL in run_ink.py.

Needs the vesuvius package (villa's ink_detection) on top of the pipeline's other packages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "1")   # before torch is imported anywhere

import numpy as np

SLICES = 21

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def read_phase9(folder) -> tuple[np.ndarray, np.ndarray, dict]:
    """phase 9's stack (21, H, W) uint8, its mask (H, W) bool and its record."""
    import tifffile
    from PIL import Image

    folder = Path(folder)
    layers = sorted((folder / "layers").glob("*.tif"))
    if len(layers) != SLICES:
        raise SystemExit(f"{folder}: {len(layers)} layers; the 9 um models want {SLICES} (phase 9 in ink9um mode)")
    stack = np.stack([tifffile.imread(p) for p in layers])
    if stack.dtype != np.uint8:
        raise SystemExit(f"{folder}: layers are {stack.dtype}; phase 9 writes uint8")
    mask = np.asarray(Image.open(folder / "mask.png")) > 0 if (folder / "mask.png").is_file() else np.ones(stack.shape[1:], bool)
    record = json.loads((folder / "extrusion.json").read_text()) if (folder / "extrusion.json").is_file() else {}
    if record and record.get("mode") != "ink9um":
        raise SystemExit(f"{folder}: phase 9 ran in {record.get('mode')!r} mode; use ink9um for these models")
    return stack, mask, record

def write_input_zarr(path, stack: np.ndarray) -> Path:
    """the layout villa's inference opens: a zarr group with level "0", (slices, rows, columns) uint8."""
    import zarr

    path = Path(path)
    root = zarr.open_group(str(path), mode="w")
    arr = root.create_array("0", shape=stack.shape, chunks=(SLICES, 128, 128), dtype="uint8") \
        if hasattr(root, "create_array") else root.create_dataset("0", shape=stack.shape, chunks=(SLICES, 128, 128), dtype="uint8")
    arr[:] = stack
    return path

def preview(prob: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(p - 0.25) / 0.5, clipped to 0..1, as uint8; black outside the mask. p is 0..1 (a uint8 file is /255)."""
    p = prob.astype(np.float32)
    if prob.dtype == np.uint8:
        p = p / 255.0
    v = np.clip((p - 0.25) / 0.5, 0.0, 1.0)
    if mask.shape == v.shape:
        v = np.where(mask, v, 0.0)
    return np.rint(v * 255).astype(np.uint8)

def villa_infer(argv: list[str]) -> int:   # one place to swap in a stand-in for the tests
    from vesuvius.ink_detection.inference import infer
    return infer.main(argv)

def run(phase9_out, checkpoint, out, device: str = "mps", sha256: str | None = None, batch_size: int = 16,
        workers: int = 4, infer_fn=None) -> dict:
    import tifffile
    from PIL import Image

    out = Path(out)
    if (out / "prediction.tif").exists():
        raise SystemExit(f"{out}/prediction.tif is already there; pick a new output folder")
    checkpoint = Path(checkpoint)
    got = sha256_file(checkpoint)
    if sha256 and got != sha256.lower():
        raise SystemExit(f"refused: {checkpoint.name} has sha256 {got}, not the pinned {sha256}")
    stack, mask, p9 = read_phase9(phase9_out)
    out.mkdir(parents=True, exist_ok=True)
    zarr_path = write_input_zarr(out / "input.zarr", stack)
    argv = [str(zarr_path), str(checkpoint), str(out / "prediction.tif"), "--device", device, "--no-compile",
            "--overlap", "0.5", "--blend-mode", "hann", "--batch-size", str(batch_size), "--num-workers", str(workers)]
    t0 = time.monotonic()
    code = (infer_fn or villa_infer)(argv)
    seconds = round(time.monotonic() - t0, 1)
    record = {"tool": "phase10_ink rev2.2", "status": "ok" if code == 0 else f"exit {code}", "seconds": seconds,
              "device": device, "villa_argv": argv[3:], "checkpoint": {"name": checkpoint.name, "sha256": got,
              "pinned": bool(sha256)}, "weights_only_switch": os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD"),
              "input": {"shape": list(stack.shape), "phase9": p9}}
    if code == 0 and (out / "prediction.tif").exists():
        prob = np.squeeze(tifffile.imread(out / "prediction.tif"))
        Image.fromarray(preview(prob, mask)).save(out / "preview.png")
        record["prediction"] = {"shape": list(prob.shape), "dtype": str(prob.dtype),
                                "sha256": sha256_file(out / "prediction.tif")}
    (out / "phase10.json").write_text(json.dumps(record, indent=1))
    return record

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="phase 10: the published 9 um ink model on phase 9's slices")
    ap.add_argument("phase9_out")
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--device", choices=("mps", "cuda", "cpu"), default="cpu")
    ap.add_argument("--sha256", help="the checkpoint's sha256, as you checked it at download")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    record = run(a.phase9_out, a.checkpoint, a.out, a.device, a.sha256, a.batch_size, a.workers)
    print(f"phase 10: {record['status']} in {record['seconds']} s on {a.device} -> {a.out}")
    return 0 if record["status"] == "ok" else 1

if __name__ == "__main__":
    sys.exit(main())
