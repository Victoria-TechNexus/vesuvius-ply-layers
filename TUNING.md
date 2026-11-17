# Tuning guide

The defaults were set on PHerc. Paris 4 (level 2, 9.6 µm). On another scroll, another part of this one, or another
resolution, some may need changing. This page lists what to look at, by symptom.

- **Where the data is** (scan, scroll centre, tiles) is set in `config.py`. On another scroll, the scroll centre
  must be changed there.
- **How each stage runs** is set in the `SETTINGS` block at the top of its runner: `run_pipeline.py` (phases 1–5),
  `run_layers.py` (phase 6), `run_mesh.py` (phase 7) and `run_ink.py` (phases 8–10). In Colab, open the file from
  the folder pane on the left; changes save straight away.
- Each phase script also takes the same settings as command-line options (`python <script>.py -h` lists them), for
  running one step by hand.
- Every run writes the settings it used: `manifest.json` for phases 1–5, and the printed log for the rest.

Change one setting at a time and rerun only the stage it belongs to. A change in phases 1–5 needs every later stage
rerun; a change in phase 6 needs phase 6 and the stages after it.

**Distances are in voxels.** All lengths below assume 9.6 µm voxels. On a scan read at another resolution, scale
every distance (radii, gaps, thicknesses, `close`, `reach` and so on) by 9.6 / (new voxel size in µm).

---

## Phases 1–5: tracks, pieces, sheets (`run_pipeline.py`)

| what you see | likely cause | try |
|---|---|---|
| Few tracks, or tracks stop short on clear papyrus | brightness threshold too strict for this scan's contrast | lower `TRACK["tau_sheet"]` (0.5 → 0.4), or widen `BRIGHT_PERCENTILES` (e.g. `(10, 90)`) |
| Tracks cross from one layer into the next | tracker turns too easily | lower `TRACK["kappa_max"]` (0.5 → 0.35); raise `TRACK["tau_verso"]` only if tracks stop too early at verso |
| Verso stops almost disappear (the `stop reasons` line shows verso in the hundreds, not thousands, per tile) | the U-Net's verso output is on a different scale, e.g. after retraining | keep the original training set (more data compressed the verso output; see the write-up, section 4), or lower `TRACK["tau_verso"]` until verso stops return |
| Large areas with no tracks at all | too few seeds | raise the seed caps in `SEED_ROUNDS`, or lower the second round's threshold (0.5 → 0.45). Caps already grow with tile volume (`SEED_CAP_SCALES_WITH_TILE`) |
| Run is slow or runs out of memory | tile too big for the session | use smaller tiles; keep `SAVE_FULL_STACKS = False`. A 192 × 416 × 416 tile peaks at about 9 GB, in phase 5 (the pair search); memory grows roughly with the tile's volume |
| Choosing a new tile size | | make each side a multiple of 8 (192 × 416 × 416 is). The U-Net halves the volume three times; other sizes are padded and trimmed, which works but differs slightly at the edges from the tested behaviour. Tiles need not be cubes; neighbouring tiles should share about 40 voxels |
| Pieces are tiny and many (the log lists thousands of pieces with 30+ tracks) | grouping too strict | lower `P5["louvain_resolution"]` (3.0 → 2.0) |
| A piece spans two layers (high `stacked` in `sheets.csv`) | grouping too loose | raise `P5["louvain_resolution"]` (3.0 → 4.0); lower `P5["d_perp_max"]` only with care |
| Joined sheets span two layers (high `stacked` with `parts` > 1) | join rule too loose | lower `JOIN["max_apart"]` (0.20 → 0.10), or raise `JOIN["min_bright"]`; phase 6a also undoes these afterwards |
| Pieces of one layer are left unjoined (many `grey_zone` rows in `join_pairs.csv`) | join rule too strict | raise `JOIN["max_apart"]` a little (0.20 → 0.25), or `JOIN["close"]` for pieces that meet end to end |
| Layers separate better with one U-Net on some tiles and with the other on others | the two checkpoints stop at verso plies at different rates in different parts of the scroll | both are in the repository: `phase3_unet_checkpoint.pt` (retrained, default) and `phase3_unet_checkpoint_v1.pt` (original). Set `UNET_CHECKPOINT`, rerun the tile, and keep the one whose phase 6 breaks fewer ply facts; the better one is usually the one with more verso stops on that tile. Tiles from both line up when joined |
| No U-Net checkpoint warning | `UNET_CHECKPOINT` not found | put `phase3_unet_checkpoint.pt` in the working folder, or retrain with `train_phase3_unet.py` |

## Phase 6: layers (`run_layers.py`)

| what you see | likely cause | try |
|---|---|---|
| A layer is "double" because a joined sheet spans two layers | a stacked joined sheet was kept | `SPLIT["stacked"]` 0.15 (default 0.2), or list a known sheet in `SPLIT["split"]` |
| A layer is "double" where two layers are pressed together | too few ply readings reach that contact | more spots per pair and a longer reach: on tiles 2 and 3, going from 2 to 6 spots (`PLIES["per_pair"]`) and from 40 to 80 voxels (`LAYERS["ply_near"]`) cut the worst doubled layer from 46–48% to 35–38%. On tiles 1 and 4 it made little difference either way. More spots make phase 6c slower (about 9 minutes per tile at 6) |
| Many spots read *unreadable* in phase 6c | stripes too faint or too few clear pixels | `PLIES["coh"]` 0.25 (default 0.3) or `PLIES["margin"]` 0.15 (default 0.2) |
| Plies read as doubled or the wrong thickness | ply thickness differs on this scroll | read it from the band-thickness table phase 6c prints (an H or a V band is one ply), then set `PLIES["ply"]` (default 7) and `PLIES["wide"]` (default 8) |
| *Same layer, delaminated* where the layers look separate | delamination gap allowed too wide | `PLIES["delam_max"]` 10 (default 12) |
| Layers merge across thin dark gaps | gap needed for a layer boundary too long | `MIN_DARK` 2 (default 3) |
| Layers split inside one thick band | thick papyrus taken as two | `LAYERS["max_thick"]` 24 (default 20) |
| Many broken ply facts in `ply_check.csv` near one sheet | that sheet crosses layers | list it in `SPLIT["split"]`, then rerun phase 6 |
| Ply readings ignored far from where they were read, or applied too far away | override range | `LAYERS["ply_near"]` (default 80) and `LAYERS["ply_reach"]` (default 30) |

## Joining tiles (`run_mesh.py`, phase 7a)

| what you see | likely cause | try |
|---|---|---|
| "The match is doubtful" warning, or two layers both close | overlap too thin, or layers pressed in the overlap | widen the tile overlap in `config.py` (40 → 60 voxels); check the table phase 7a prints for each tile |
| A layer splits in the new tile | the new tile numbered it as two layers | expected: both are taken when each is close enough; check the printed table |
| Too few overlap points match | the radius is too tight for this scan | `COMBINE["radius"]` 4 (default 3) |

## Mesh (`run_mesh.py`, phase 7b)

| what you see (check numbers) | likely cause | try |
|---|---|---|
| Check 7 fails: kept tracks spread too wide | a neighbouring layer is mixed in | lower `MESH["keep"]` (5 → 4), or fix the layer in phase 6 first |
| Over 20% of points set aside | layer pressed against a neighbour, or `keep` too tight | raise `MESH["keep"]` (5 → 6); look at the magenta in the check picture |
| Check 6 fails: stretch | the sheet has sharp folds, or links jump between layers | lower `MESH["link"]` (6 → 5); a smaller `MESH["node"]` keeps more detail but runs slower |
| Check 1 fails: mesh off the band middle | band search too wide or too narrow | `MESH["reach"]` (default 12) |
| Many holes | tracks sparse | more seeds in phases 1–5, or `MESH["bridge"]` 8 (default 6) to bridge wider holes (bridged areas are marked) |
| Meshing is slow | too many points; the flat layout is the slow step | keep `MESH["node"]` 2.5 and `MESH["link"]` 6 (the defaults); four tiles take about 23 minutes, six about 28 |
| Meshing stops with error code −9 | the session ran out of memory, usually while refining the flat layout | restart the session (Runtime → Restart session) and rerun with `--from b`; if it still stops, lower `MESH["local"]` (40 → 30), which makes the unrolling slightly less exact over long distances. Six tiles fit a 12 GB session at the defaults |

## Ink (`run_ink.py`, phases 8–10)

| what you see | likely cause | try |
|---|---|---|
| Ink faint, or ink from the neighbouring layer shows | reading window at the wrong depth | `SHIFT` (5 is inward, toward the recto); try 3 and 7 and compare |
| Text is upside down | the scroll axis runs the other way in this scan | `GRID["up"]`: `high_z` (default, upright on PHerc. Paris 4) or `low_z` |
| Text is mirrored | surface seen from the wrong side | check the "reading side" line printed by phase 8; `FLIP = True` reads from the other side |
| Want a different ink model | | set `INK_SCRIPT` and `MODEL`; `phase10_ink.py` states what a replacement must read and write (the 21-slice, 9 µm layout) |

## Retraining the U-Net for another scroll (`train_phase3_unet.py`)

The U-Net was trained on four blocks of PHerc. Paris 4 and one of PHerc. 332. A scroll that looks different in the
CT (more crushed, darker, another scan setting) may do better with a U-Net trained on blocks of its own. Training
needs no human labels and takes about 6 minutes on a T4.

**Setting it up.** Edit `SOURCES` at the top of `train_phase3_unet.py`: each block is a scroll, a scan and z, y, x
ranges at level 2. Set `VALIDATION` to one block of the same scroll that is never trained on, away from the training
blocks and from the tiles you will run. Rename the old checkpoint first; the script will not overwrite it.

**Keep the training set small and even.** On PHerc. Paris 4, adding six blocks to the original five made the
pipeline worse: the network learned to hedge, its verso output stayed near the middle, and the tracker's verso stop
almost never fired, so tracks ran across layer contacts. Start from about five blocks of the original size
(64 × 128 × 128), chosen to cover the kinds of papyrus in the scroll (tight and loose wraps, different heights), and
replace blocks rather than add them.

**Checking that it trained well.** The class loss barely moves during training whatever the data (its floor is
about 0.05), so do not judge by it. Check, in this order:

| where | what to look for | a warning sign |
|---|---|---|
| `python train_phase3_unet.py --check` (before training) | every block 100% inside the scan, with plenty of papyrus | a block flagged "little or no papyrus": move it |
| the held-out lines during training | held-out recto and verso error well below "answering the average", and held-out loss falling with the training loss | held-out error near the average, or rising while the training loss falls (the network memorised its blocks) |
| phase 3 in `run_pipeline.py` | `mean polarity strength` near the previous model's on the same tile (about 0.10–0.12 on PHerc. Paris 4) | a clearly lower value (0.07 for the over-trained version) |
| phase 4 `stop reasons` | verso stops in the thousands per tile (about 3,000–15,000 on PHerc. Paris 4) | verso stops in the hundreds: the verso output is compressed; use less data, or lower `TRACK["tau_verso"]` until they return |
| phase 6 on the same tile, against the previous model | ply check broken % and doubled layers no worse | more broken facts or more layers over 30% doubled |

The last row is the one that counts: the U-Net only matters through how well the layers separate.
