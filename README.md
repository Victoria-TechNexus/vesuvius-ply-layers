# Automatic layer separation and meshing for Herculaneum scroll CT (draft)

From raw CT to a flattened papyrus layer with ink, with no manual tracing.
Tested on PHerc. Paris 4 (ESRF 2.4 µm scan, read at level 2 = 9.6 µm).

**Status:** research code. Works end to end on a 2 × 3 block of tiles (about 4.8 × 7.6 mm of one layer).
Written and run in Google Colab.

## What it does

1. **Tracks the papyrus.** Ridge tracking through the CT, guided by a structure tensor and a 3D U-Net,
   gives ~50,000 short tracks per tile; phase 5 groups them into pieces and joins pieces into sheets.
2. **Separates the layers using the papyrus itself.** Each papyrus layer has two plies: the recto with fibres
   running around the scroll and the verso with fibres along its axis. `ply_label.py` reads the ply order
   between neighbouring sheets and labels them *same layer* / *different layers* / *delaminated* / *same ply*,
   which resolves pressed spots where a dark-gap rule fails.
3. **Undoes bad joins.** `split_sheets.py` returns joined sheets that lie on top of themselves (span two layers)
   to their pieces.
4. **Numbers the layers.** `stack_layers.py` orders all sheets through the stack, with the ply facts overruling
   the dark-gap rule near where they were read, and reports every contradiction.
5. **Joins tiles.** `combine_tiles.py` matches the same layer across overlapping tiles (40-voxel overlap) and
   stitches it into one set.
6. **Meshes and flattens one layer.** `phase6_track_surface.py` builds the surface from the layer's own tracks
   (folds and wrinkles kept), sets aside points that belong to a neighbouring layer, unrolls it, and runs eight
   self-checks. `mesh_to_grid.py` turns it into the flat grid the ink step reads.
7. **Reads ink.** `read_ink.py` samples 21 slices through the sheet and runs an ink model.

## Results (PHerc. Paris 4, layer of the official segment)
Six tiles, each 192 × 416 × 416 voxels (z × y × x, 9.6 µm), each sharing 40 voxels with its neighbours.
Each tile was run on its own, then the same layer was matched across all of them.
| Tile | Position | z | x | Matching Layer | Overlap Points Within 3 Voxels |
|---|---|---|---|---|---|
| t1 | test chunk | 15534–15726 | 3895–4311 | 11 | reference |
| t2 | beside t1 | 15534–15726 | 4271–4687 | 12 | 93% |
| t3 | above t1 | 15382–15574 | 3895–4311 | 9 | 98% |
| t4 | below t1 | 15686–15878 | 3895–4311 | 10 | 95% |
| t5 | above t2 | 15382–15574 | 4271–4687 | 12 | 96% |
| t6 | below t2 | 15686–15878 | 4271–4687 | 12 | 99% |

Each tile numbers its own layers, so the same layer has a different number in each.

**The joined layer.** Tiles were joined step by step. Side by side, two tiles make 416 + 416 − 40 = 792 voxels
wide; three stacked make 192 × 3 − 2 × 40 = 496 voxels tall.

| Tiles Joined | Joined Size (z × y × x) | Ink Score vs. Human Labels* | Labelled Ink Found | Labelled Blank Marked as Ink |
|---|---|---|---|---|
| t1 alone | 192 × 416 × 416 | +0.81 | 95% | 14% |
| t1 to t2 | 192 × 416 × 792 | +0.88 | 96% | 8% |
| t1 to t4 | 496 × 416 × 792 | +0.89 | 97% | 8% |
| t1 to t6 | 496 × 416 × 792 | +0.89 | 97% | 8% |

**Against the human-made surface.** The same ink model and reading (shift 5, 21 slices, 9.6 µm) on the official
segment's surface and on ours, scored against the same human labels:

| surface | ink score vs. human labels* | labelled ink found | labelled blank marked as ink |
|---|---|---|---|
| official segment (human-made) | +0.80 | 87% | 6% |
| ours (automatic, all six tiles) | +0.73 | 78% | 5% |

Our surface is fully automatic. With the same ink model and reading, it scores +0.73 against the human labels;
the hand-made official surface, on which the labels were drawn, scores +0.80. Ours is scored only where the
official surface lies within 6 voxels of it (about half of our sheet).

\* Score = share of labelled ink marked as ink minus share of labelled blank marked as ink, on human-checked
pixels (0 = guessing, +1 = perfect). Sliding the labels out of place (230 trials) gives about −0.1, and +0.3 at
the 95th percentile.

**Matching across tiles:** in every tile overlap, 93–99% of the layer's points lie within 3 voxels of the
neighbouring tile's copy of the layer.

**Compute** (Colab, per 192 × 416 × 416 tile): about 13–14 min and 11–12 GB peak RAM for stages 1–2; meshing
about 7–8 min per million track points.

## Requirements

`pip install -r requirements.txt`. A GPU helps for the U-Net (phase 2); everything else runs on CPU.
The CT is streamed from the Vesuvius Challenge open-data bucket
(`s3://vesuvius-challenge-open-data/PHercParis4/volumes/...`); nothing has to be downloaded by hand.

Model files (not in this repository):
- `phase2_stage_b_checkpoint.pt`: U-Net for phase 2 — _source / licence to fill in_
- `step-075000.pth`: ink model for phase 10 — _source / licence to fill in_

## How to run

Run everything from the folder holding these scripts. Tiles are listed in `TILES` at the top of
`run_pipeline.py` (each 192 × 416 × 416, with 40 voxels of overlap with its neighbours).

**Per tile** (example: tile `t2`)
```
python run_pipeline.py t2                                    # phases 1-5 + join  -> ckpt_t2/
python split_sheets.py --ckpt ckpt_t2
python pair_table.py   --ckpt ckpt_t2 --labels sheet_id_split.npy --out pairs_t2s
python ply_label.py    --ckpt ckpt_t2 --labels sheet_id_split.npy --pairs pairs_t2s --out plies_t2s
python stack_layers.py --ckpt ckpt_t2 --labels sheet_id_split.npy --plies plies_t2s/pair_labels.csv --out layers_t2s
```

**Per layer** (example: layer 11 of tile 1, across six tiles)
```
python combine_tiles.py --tiles ckpt:layers_s ckpt_t2:layers_t2s ckpt_t3:layers_t3s ckpt_t4:layers_t4s ckpt_t5:layers_t5s ckpt_t6:layers_t6s --layer 11 --out combo11r
python phase6_track_surface.py --ckpt combo11r --layers_dir combo11r/layers --out combo11r/wraps --layer 11 --node 2.5 --link 6
python read_ink.py --name layer11 --id 9511 --ckpt combo11r --wraps combo11r/wraps
```

**Checks** (optional)
```
python compare_labels.py --id 9511 --ckpt combo11r     # score against the official segment's human labels
python ink_overlap.py --a 9511 --b 9510                # is the ink on two layers the same ink, in 3D?
python layer_report.py                                 # table of the phase 6 checks for every meshed layer
python layer_map.py --old layers --new layers_s        # match layer numbers between two stackings
```

## Outputs

| step | writes |
|---|---|
| run_pipeline.py | `ckpt*/`: tracks, pieces, joined sheets, compact CT and normals |
| stack_layers.py | `layers*/layer_id.npy` (layer of every track), per-layer files, `ply_check.csv`, `ply_contradictions.csv` |
| phase6_track_surface.py | `wraps/wrap_layerNN_mesh_final.obj`, check picture and report |
| read_ink.py | `wrap_<id>_V3.tifxyz`, `diagnostics/ink_<id>.png` |

## Known limits

- Single pieces from phase 5 that span two layers (e.g. one sheet per tile with "stacked" above 0.3) are not
  split yet; they cause most of the remaining tangled layers.
- The flat grid is saved as one 3-channel TIFF, not yet as VC3D's tifxyz folder.
- Ink is read at 9.6 µm; strokes are about twice their real width.
- Validated against human labels on one layer only (the official segment's).

## Licence

MIT (see LICENSE).
