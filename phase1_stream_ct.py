
"""phase1_stream_ct.py - stream a block of CT from an OME-Zarr scan volume.

Works with any scroll in the Vesuvius Challenge open-data bucket, another bucket or web address, or a local copy.

    from phase1_stream_ct import ScrollStreamer
    s = ScrollStreamer()                                                    # PHerc. Paris 4, 2.4 um scan, level 2 (9.6 um)
    s = ScrollStreamer(scroll_name="PHerc0332", scan_folder="<volume>.zarr") # another scroll in the same bucket
    s = ScrollStreamer(source="/content/drive/MyDrive/scans/my_scan.zarr")  # a local copy or a full URL
    vol = s.fetch_chunk(z0, z1, y0, y1, x0, x1)                             # float32, scaled to 0..1

To see what a scan holds (its resolution levels and their sizes) before choosing tiles:

    python phase1_stream_ct.py --scroll PHercParis4 --scan 20260411134726-2.400um-0.2m-78keV-masked.zarr

Level 0 is the full resolution; each level up halves it (level 2 of a 2.4 um scan = 9.6 um). The pipeline's distances are
set for 9.6 um voxels: at another voxel size, see docs/TUNING.md.
"""
import argparse
import numpy as np
import zarr

DEFAULT_BUCKET = "s3://vesuvius-challenge-open-data"
DEFAULT_SCROLL = "PHercParis4"
DEFAULT_SCAN = "20260411134726-2.400um-0.2m-78keV-masked.zarr"
DEFAULT_LEVEL = 2


class ScrollStreamer:
    def __init__(self, scroll_name=DEFAULT_SCROLL, scan_folder=DEFAULT_SCAN, level=DEFAULT_LEVEL,
                 source=None, bucket=DEFAULT_BUCKET, anon=True):
        """
        scroll_name, scan_folder  the scan inside `bucket`: <bucket>/<scroll_name>/volumes/<scan_folder>
        source                    instead: a full path or URL to the .zarr (s3://..., https://..., or a local folder)
        level                     resolution level to read (0 = full resolution)
        anon                      anonymous access for remote storage (the open-data bucket needs no account)
        """
        self.path = source if source else f"{bucket.rstrip('/')}/{scroll_name}/volumes/{scan_folder}"
        remote = "://" in self.path
        print(f"Connecting to {self.path}")
        opts = dict(storage_options={"anon": anon}) if remote and self.path.startswith("s3://") else {}
        self.dataset = zarr.open(self.path, mode="r", **opts)

        if isinstance(self.dataset, zarr.Array):                          # a bare array, not an OME-Zarr pyramid
            if level not in (0, None):
                print(f"  note: this volume has no resolution levels; reading it as it is (level {level} ignored)")
            self.target_level, self.vol_raw = None, self.dataset
        else:
            levels = sorted(k for k in self.dataset.array_keys() if k.isdigit()) if hasattr(self.dataset, "array_keys") \
                else sorted(k for k in self.dataset.keys() if k.isdigit())
            if str(level) not in levels:
                raise ValueError(f"level {level} not found; this volume has levels {', '.join(levels) or 'none'}")
            self.levels = levels
            self.target_level = str(level)
            self.vol_raw = self.dataset[self.target_level]
        self.shape = tuple(self.vol_raw.shape)
        print(f"Connected. Level {self.target_level} shape (z, y, x): {self.shape}, dtype {self.vol_raw.dtype}")

    def fetch_chunk(self, z_start, z_end, y_start, y_end, x_start, x_end):
        """Read one block (end values excluded) and scale it to 0..1 as float32."""
        for name, a, b, n in (("z", z_start, z_end, self.shape[0]), ("y", y_start, y_end, self.shape[1]), ("x", x_start, x_end, self.shape[2])):
            if not (0 <= a < b <= n):
                raise ValueError(f"{name} range {a}-{b} is outside this volume (0-{n}) at level {self.target_level}")
        print(f"Streaming chunk Z:{z_start}-{z_end}, Y:{y_start}-{y_end}, X:{x_start}-{x_end}...")
        chunk = np.asarray(self.vol_raw[z_start:z_end, y_start:y_end, x_start:x_end])
        if np.issubdtype(chunk.dtype, np.integer):
            out = chunk.astype(np.float32) / float(np.iinfo(chunk.dtype).max)    # 8-bit /255, 16-bit /65535
        else:
            out = chunk.astype(np.float32)
            top = float(np.nanmax(out)) if out.size else 0.0
            if top > 1.5:                                                    # float data not already in 0..1
                out /= top
                print(f"  note: float data scaled by its maximum ({top:.4g}) in this block")
        print(f"Chunk loaded and scaled to 0..1 (dtype: {out.dtype}).")
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Show the resolution levels of a scan volume (no data is downloaded).")
    ap.add_argument("--scroll", default=DEFAULT_SCROLL); ap.add_argument("--scan", default=DEFAULT_SCAN)
    ap.add_argument("--source", default=None, help="full path or URL to the .zarr instead of --scroll/--scan")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    a = ap.parse_args()
    s = ScrollStreamer(a.scroll, a.scan, level=0, source=a.source, bucket=a.bucket)
    if getattr(s, "levels", None):
        for lv in s.levels:
            print(f"  level {lv}: shape (z, y, x) {tuple(s.dataset[lv].shape)}")
