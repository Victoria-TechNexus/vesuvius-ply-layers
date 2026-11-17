
"""phase6a_split_sheets.py - phase 6, step a (run by run_layers.py): undo the stage-1 join for joined sheets that lie on
top of themselves (span two layers), and keep every other joined sheet as it is.

Why: a joined sheet such as tile 2's sheet 198 (9 pieces, 23% stacked) glues two layers into one. Starting from all the
pieces instead (component_id.npy) fixes that, but loses the joins that were right and makes other layers worse. This keeps
the joins, except for the sheets flagged here (or named with --split), whose tracks go back to their own pieces.

    python phase6a_split_sheets.py --ckpt ckpt_t1                      # split joined sheets with stacked >= 0.2
    python phase6a_split_sheets.py --ckpt ckpt_t1 --split 198 56 247   # also split exactly these

Reads <ckpt>/sheet_id.npy, component_id.npy, sheets.csv. Writes <ckpt>/sheet_id_split.npy: one label per track, like
sheet_id.npy. Tracks of a split sheet get their piece's number plus an offset, so they never clash with another sheet.
If nothing needs splitting, the file is an unchanged copy, so the next steps always find it.
"""
import argparse, os, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ckpt_t1", help="stage-1 folder of the tile")
ap.add_argument("--split", type=int, nargs="*", default=[], help="joined sheet numbers to split back into their pieces")
ap.add_argument("--stacked", type=float, default=0.2, help="also split every joined sheet (2+ pieces) whose stacked share in sheets.csv is at least this (0 = off)")
ap.add_argument("--min_tracks", type=int, default=30, help="only for the printout: pieces this size count as real pieces")
ap.add_argument("--out", default="sheet_id_split.npy")
a = ap.parse_args()

f = lambda n: os.path.join(a.ckpt, n)
for n in ("sheet_id.npy", "component_id.npy"):
    if not os.path.exists(f(n)):
        sys.exit(f"Cannot find {f(n)} from {os.getcwd()}")
sid = np.load(f("sheet_id.npy")).astype(np.int64); cid = np.load(f("component_id.npy")).astype(np.int64)
if len(sid) != len(cid):
    sys.exit("sheet_id.npy and component_id.npy have different lengths: not from the same run")

pick = set(a.split); t = None
if a.stacked > 0:
    if os.path.exists(f("sheets.csv")):
        t = pd.read_csv(f("sheets.csv"))
        pick |= set(t.loc[(t["parts"] > 1) & (t["stacked"] >= a.stacked), "sheet"].astype(int))
    else:
        print(f"  no {f('sheets.csv')}: only the sheets given with --split are split")
missing = [s for s in sorted(pick) if not np.any(sid == s)]
if missing:
    print(f"  not found in sheet_id.npy (skipped): {missing}")
pick = [s for s in sorted(pick) if s not in missing]
if not pick:
    np.save(f(a.out), sid)
    print(f"[SPLIT SHEETS] {a.ckpt}: no sheet to split; wrote {f(a.out)} unchanged (a copy of sheet_id.npy)")
    sys.exit(0)

off = int(max(sid.max(), cid.max())) + 1
new = sid.copy()
ntr_piece = np.bincount(cid)
print(f"[SPLIT SHEETS] {a.ckpt}: {len(sid):,} tracks | splitting {len(pick)} joined sheets back into their pieces (new labels = piece + {off})")
for s in pick:
    m = sid == s
    new[m] = cid[m] + off
    pcs = np.unique(cid[m])
    big = pcs[ntr_piece[pcs] >= a.min_tracks]
    st = ""
    if t is not None:
        row = t[t["sheet"] == s]
        if len(row):
            st = f" | stacked {float(row['stacked'].iloc[0]):.2f}"
    print(f"  sheet {s}: {int(m.sum()):,} tracks -> {len(pcs)} pieces ({len(big)} with {a.min_tracks}+ tracks: "
          f"{', '.join(str(int(p)) for p in big[np.argsort(-ntr_piece[big])][:10])}){st}")
np.save(f(a.out), new)
print(f"  wrote {f(a.out)}: {len(np.unique(sid)):,} labels -> {len(np.unique(new)):,}")
