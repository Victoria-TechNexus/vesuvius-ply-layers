# The pipeline in equations

This page states what each step computes, with the pipeline's current default settings. Coordinates are voxels of
the level-2 volume (9.6 µm); arrays are indexed $`(z, y, x)`$. Every section was checked line by line against the
code; file names and defaults are those of the current version.

**Notation.** $`I(\mathbf{x}) \in [0,1]`$ is the normalised CT; $`B(\mathbf{x}) \in [0,1]`$ is CT brightness rescaled
between its 20th and 90th percentile after a Gaussian blur ($`\sigma = 1`$), with $`B \ge 0.5`$ counted as papyrus.
$`\mathbf{n}(\mathbf{x})`$ is the unit surface normal field, $`\mathbf{t}(\mathbf{x})`$ the unit fibre (track) direction.
A *track* is a polyline $`\gamma_i`$; a *piece* is a group of tracks from phase 5; a *sheet* is one piece or several
joined pieces; a *layer* is one turn of papyrus (recto ply + verso ply).

---

## Stage 1: tracks, pieces, sheets (`run_pipeline.py`, one tile)

### 1. CT (`phase1_stream_ct.py`)
A tile of $`192 \times 416 \times 416`$ voxels is read from level 2 (9.6 µm) of the OME-Zarr volume of the 2.4 µm scan
(`PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr`, public S3 bucket) and scaled to $`[0,1]`$ by
dividing by the type's maximum ($`I = v/255`$ for 8-bit data, $`v/65535`$ for 16-bit). No other normalisation is
applied; $`B`$ (see Notation) is the percentile-based rescaling used where brightness thresholds are needed.

### 2. Orientation fields (`phase2_structure_tensor.py`)
Structure tensor of the smoothed gradient,

```math
J_\rho(\mathbf{x}) = K_\rho * \left(\nabla I_\sigma \, \nabla I_\sigma^{\mathsf T}\right), \qquad \sigma = 1,\ \rho = 6 ,
```

with eigenvalues $`\lambda_1 \ge \lambda_2 \ge \lambda_3`$ and eigenvectors $`\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3`$.
The normal is $`\mathbf{n} = \mathbf{v}_1`$, turned to point from dark to bright ($`\mathbf{n}\cdot\nabla I_\sigma \ge 0`$);
the fibre (track) direction is $`\mathbf{t} = \mathbf{v}_3`$, the direction of least change; and the tensor's sheet
measure is $`(\lambda_1 - \lambda_2)/(\lambda_1 + \lambda_2 + \lambda_3)`$, clipped to $`[0,1]`$.

### 3. Sheet, recto and verso probabilities (`phase3_unet.py`)
A small 3D U-Net (three levels, 16–128 channels) predicts four per-voxel probabilities: sheet, recto, verso and
polarity. It was trained (`train_phase3_unet.py`; 60 epochs, 64³ patches from four regions of PHerc. Paris 4 and one
of PHerc. 332) on targets made from the structure tensor itself:

```math
p_{\text{verso}}^{\text{target}} = s\,\frac{|t_z|}{|t_z| + \lVert (t_y, t_x)\rVert}, \qquad
p_{\text{recto}}^{\text{target}} = s\,\frac{\lVert (t_y, t_x)\rVert}{|t_z| + \lVert (t_y, t_x)\rVert},
```

with $`s`$ the tensor's sheet measure: verso where the fibres run along the scroll axis ($`z`$), recto where they run
around it. For these targets the tensor is computed at a finer scale than in phase 2 ($`\sigma = 1,\ \rho = 1.2`$),
so that one ply's fibre direction is not averaged with its neighbour's. The U-Net's input is the CT alone. The field
stack holds channels 0–2 $`\mathbf{t}`$, 3–5 $`\mathbf{n}`$ (both from the tensor) and 6–8 the U-Net's sheet / recto /
verso probabilities; the verso channel is $`p_V(\mathbf{x})`$.

### 4. Tracking (`phase4_tracking.py`)
For tracking, channel 6 is replaced by the brightness $`B`$, so the tracker follows bright papyrus in the CT rather
than the U-Net's sheet probability. Seeds are ridge points of $`B`$: voxels with $`B`$ above the threshold that are at
least as bright as the points 1.5 voxels either side along $`\mathbf{n}`$, more than 2 voxels from existing tracks,
one per 2-voxel cell. Round 0 uses $`B > 0.6`$ (capped at 45,000, chosen at random), round 1 $`B > 0.5`$ (capped at
15,000); rounds stop early if fewer than 500 seeds are left.

*Integration (RK4).* From each seed, a track is integrated in both directions along the fibre field $`\mathbf{t}`$:

```math
\frac{d\gamma}{ds} = \pm\,\mathbf{t}(\gamma(s)) .
```

Each step uses the fourth-order Runge–Kutta method (RK4) with step $`h = 0.5`$ voxels:

```math
\mathbf{k}_1 = \mathbf{t}(\mathbf{x}), \qquad
\mathbf{k}_2 = \mathbf{t}\!\left(\mathbf{x} + \tfrac{h}{2}\mathbf{k}_1\right), \qquad
\mathbf{k}_3 = \mathbf{t}\!\left(\mathbf{x} + \tfrac{h}{2}\mathbf{k}_2\right), \qquad
\mathbf{k}_4 = \mathbf{t}\!\left(\mathbf{x} + h\,\mathbf{k}_3\right),
```

```math
\mathbf{x}_{\text{new}} = \mathbf{x} + \frac{h}{6}\left(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\right).
```

RK4 samples the field four times per step, so the track follows a curving sheet much more closely than a single
straight step (Euler) would, without drifting off the papyrus on bends. The fibre field has no fixed sign
($`\mathbf{t}`$ and $`-\mathbf{t}`$ are the same direction), so $`\mathbf{k}_1`$ is turned to agree with the previous
step and $`\mathbf{k}_2, \mathbf{k}_3, \mathbf{k}_4`$ to agree with $`\mathbf{k}_1`$; the first step of each direction
starts from a fixed reference, $`\pm(0, 1, 0)`$.

Between voxels, $`\mathbf{t}`$ is interpolated trilinearly after turning each corner vector to agree with the
direction being followed, so the sign never flips mid-step. A direction stops at the tile edge, after 600 voxels, where $`B < 0.5`$, where
$`p_V > 0.6`$ (verso reached), or where the turn per voxel exceeds $`\kappa_{\max} = 0.5`$ radians; a direction stopped
by anything but the edge or length loses its last 1 voxel. The two directions are joined, and tracks with 30+ points
are kept.

### 5. Pieces: affinity graph between tracks (`phase5_pieces.py`)
Every third track point is used. For point pairs $`(i, j)`$ on different tracks within $`r_{\max} = 3.5`$ voxels, with
mean normal $`\bar{\mathbf{n}} = (\mathbf{n}_i \pm \mathbf{n}_j)/\lVert\cdot\rVert`$ (sign made consistent) and offset
$`\mathbf{d} = \mathbf{p}_j - \mathbf{p}_i`$:

```math
d_\perp = |\mathbf{d}\cdot\bar{\mathbf{n}}|, \qquad d_\parallel^2 = \lVert\mathbf{d}\rVert^2 - d_\perp^2 .
```

Pairs with $`d_\perp > 1`$ are dropped, as are track pairs with fewer than 3 such point pairs. The weight is

```math
w_{ij} = \exp\!\left(-\frac{d_\perp^2}{2\sigma_n^2}\right)\exp\!\left(-\frac{d_\parallel^2}{2\sigma_\parallel^2}\right)
|\mathbf{t}_i\cdot\mathbf{t}_j|^{\alpha}\;\exp\!\left(-\frac{\lVert\mathbf{d}\rVert\,\overline{p_V}}{s_V}\right),
```

with $`\sigma_n = 1`$, $`\sigma_\parallel = 8`$, $`\alpha = 2.5`$, $`s_V = 0.15`$, and $`\overline{p_V}`$ the mean verso
probability at 9 points along the segment. Pairs with $`w_{ij} < 0.005`$ are dropped, both before and after the verso factor is applied.

*Valley test.* Along each segment, with $`v_k`$ the blurred CT at 9 points and $`v_e = \min(v_0, v_8)`$,

```math
V_{ij} = \max\!\left(0,\ \frac{v_e - \min_k v_k}{v_e}\right).
```

Point pairs are grouped per track pair (weight = max, valley = median); a track pair with median valley $`> 0.05`$
(a dark dip between them) is removed. Tracks are then grouped into pieces by Louvain community detection on the
weighted track graph (resolution 3.0).

### Joining pieces into sheets
Only pieces with 30+ tracks take part, each represented by every 4th track point. For two pieces $`X, Y`$ with at
least 100 points within 8 voxels of each other, up to 4,000 sampled points $`\mathbf{x}\in X`$ and
their nearest $`\mathbf{y}\in Y`$ give two measures:

- *Apart share* $`a_{XY}`$: the fraction of points whose neighbours in the other piece (within 2 voxels sideways)
  sit at median depth $`\ge 2.5`$ voxels along the normal, i.e. a layer away. The larger of the two directions is used.
- *Bright share* $`b_{XY}`$: the fraction of segments $`\mathbf{x}\to\mathbf{y}`$ whose darkest of 9 samples has $`B \ge 0.5`$.

Pieces are joined, largest overlap first, if $`a_{XY} \le 0.20`$ and $`b_{XY} \ge 0.90`$, unless the join would put two
pieces with $`a \ge 0.5`$ into one sheet (union-find with a veto).

*Stacked share* of a sheet (reported per sheet, used in stage 2): the fraction of 1,500 sampled points that have
another point of the same sheet 6–14 voxels away along the normal and within 4 voxels sideways. About 0 for a
single layer; high when two layers are stuck together.

---

## Stage 2: layers (`run_layers.py`, one tile)

### 6a. Splitting stacked joins (`phase6a_split_sheets.py`)
Every joined sheet with 2+ pieces and stacked share $`\ge 0.2`$ is returned to its pieces (new label = piece +
offset); all other sheets are kept.

### 6b. Pair table (`phase6b_pair_table.py`)
From up to 120,000 start points, walk outward along the local normal to the next sheet within 30 voxels and record
the gap, the darkest CT between, the band thickness and the angle between the two sheets. Each sample is classed
(separate / far / thin / pressed / ambiguous / edge); pairs of sheets with 20+ samples go to the ply reader.

### 6c. Reading the plies (`phase6c_ply_label.py`)
At up to 6 spots per pair (set in `run_layers.py`; the script's own default is 2), a surface is fitted to the inner
sheet $`A`$ on a $`49 \times 49`$ grid turned 45° to the scroll axis, and face-on CT slices are cut at depths $`k`$ from
15 voxels inside $`A`$ to 15 beyond the outer sheet $`B`$.

*Fibre direction per slice.* In each slice $`S_k`$ (high-passed by subtracting a $`\sigma=4`$ blur), a 2D structure
tensor with gradient scale 1 and integration scale $`\rho = 4`$, restricted to papyrus pixels, gives at every pixel
the orientation $`\phi`$ and coherence

```math
c = \frac{2\sqrt{\tfrac14 (J_{rr}-J_{cc})^2 + J_{rc}^2}}{J_{rr}+J_{cc}} .
```

The angle $`\theta`$ between the stripe direction and the scroll axis is computed. Among coherent pixels
($`c \ge 0.3`$), with $`h = P(\theta \ge 60°) - \tfrac13`$ and $`v = P(\theta \le 30°) - \tfrac13`$, depth $`k`$ is labelled

```math
\ell_k = \begin{cases} \texttt{.} & \text{if more than half the patch has } B < 0.5\ (\text{gap})\\
\texttt{H} & \text{if } h \ge 0.2 \text{ and } h > v\ (\text{fibres around the scroll: recto})\\
\texttt{V} & \text{if } v \ge 0.2 \text{ and } v > h\ (\text{fibres along the axis: verso})\\
\texttt{?} & \text{otherwise}\end{cases}
```

where a depth with 30 or fewer coherent pixels is `?`; single-slice runs of H or V then become `?`. This is done for
the central $`25\times25`$ window and separately for each quarter of the patch (quarter agreement is recorded).

*Bands.* The letters are split into bands of papyrus: a gap is a run of `.` and `?` holding at least two `.`.

*Rules.* Let $`A`$, $`B`$ be the plies at the two sheets' depths (searching ±3 slices), and the run sequence between them
read from $`A`$ outward to $`B`$:

| condition (checked in this order) | label |
|---|---|
| some V is followed later by an H | different layers |
| no V anywhere in the window | unreadable |
| no ply found at $`A`$ or at $`B`$ | unreadable |
| both sheets in one papyrus band, $`A=B`$, depths $`< 7`$ voxels apart | same ply |
| both sheets in one band, $`A=B`$, depths $`\ge 7`$ voxels apart | unreadable (two plies of one direction) |
| both sheets in one band, $`A=\mathrm{H}`$, $`B=\mathrm{V}`$ | same layer |
| separate bands; $`A`$'s band already holds a V outward of $`A`$, or $`B`$'s band holds an H inward of $`B`$ | likely different layers |
| separate bands, $`A=B`$, gap $`\ge 8`$ | likely different layers |
| separate bands, $`A=B`$, gap $`< 8`$ | unreadable (crack or missing ply) |
| separate bands, $`A=\mathrm{H}`$, $`B=\mathrm{V}`$, gap $`\le 12`$ | same layer, delaminated |
| separate bands, $`A=\mathrm{H}`$, $`B=\mathrm{V}`$, gap $`> 12`$ | possibly delaminated |
| anything else | unreadable |

Per pair, readings are combined: weak labels (*likely different layers*, *possibly delaminated*) give way to strong
ones; a mix of "same" and "different" families is a *conflict*.

### 6d. Stacking (`phase6d_stack_layers.py`)
1. *Direction across the stack:* $`\mathbf{n}_0`$ = leading eigenvector of $`\sum \mathbf{n}\mathbf{n}^{\mathsf T}`$ over
   sampled track points, oriented away from the scroll centre.
2. *Columns:* the tile is cut into columns 3 voxels wide along $`\mathbf{n}_0`$. Within each column, the points of one
   sheet (or track) form a *pass*, split where they are more than 2.5 voxels apart in depth; passes are ordered by
   depth $`t`$.
3. *Neighbour facts:* for consecutive passes with gap $`\le 60`$, sample $`B`$ every 0.5 voxels along the segment and
   count dark runs ($`B < 0.5`$) of length $`\ge 3`$ voxels, $`n_{\text{dark}}`$. The number of layers between the
   two passes is $`\Delta = 0`$ if $`n_{\text{dark}} = 0`$ and the gap is under 20 voxels, and
   $`\Delta = \max(1, n_{\text{dark}})`$ otherwise.

   *Ply override:* for passes of a labelled pair within 30 voxels of each other and within 80 voxels of a spot that
   gave that answer (set in `run_layers.py`; the script's own default is 40), "different layers" turns $`\Delta = 0`$
   into 1 (pressed) and "same layer / delaminated / same ply" turns $`\Delta = 1`$ into 0 (delaminated).
4. *Sheet level:* per sheet pair, $`d`$ = rounded median of signed $`\Delta`$ over shared columns (5+ columns);
   pairs with $`d \ne 0`$ where both directions occur in $`\ge 20\%`$ of columns are dropped as conflicts; weight $`w`$ = number of columns
   (×0.3 if mostly more than one layer apart). Integer positions $`L_s`$ are found by a maximum-weight spanning
   placement from the largest sheet, then up to 50 rounds in which each sheet takes the position with the largest
   summed weight of votes $`L_{s'} - d_{ss'}`$.
5. *Track level:* each track starts at its sheet's position and, for up to 60 rounds, moves to the position most of
   its column neighbours vote for ($`L_{\text{nb}} \pm \Delta`$, weight 0.3 when $`\Delta > 1`$; its sheet's position
   counts as one extra vote; half the movers move per round).
6. *Self-check:* each ply fact the final layers break is listed; each "different layers" fact whose two sheets are
   joined by a chain of "same" facts is a contradiction, traced to the "same" fact appearing in the most chains.

---

## Stage 3: one layer across tiles (`run_mesh.py`, phase 7a: `phase7a_combine_tiles.py`)
In the overlap between the layer so far and a new tile, for each layer $`\lambda`$ of the new tile:

```math
m_\lambda = \#\{\text{points of } \lambda \text{ in the overlap within 3 voxels of a reference-layer point}\},
```

and the layer with the largest $`m_\lambda`$ is taken (its share of overlap points is reported). Another layer is
taken as well if its $`m_\lambda`$ is at least 30% of the best and at least half of its overlap points are close (a
layer the new tile split in two). Tracks are moved into
one coordinate frame; the CT, brightness and normals are stitched, each voxel taken from the tile whose edge is
furthest away. With a single tile this step is skipped.

---

## Stage 4: mesh, flat grid, ink (one layer)

### 7b. Mesh and unrolling (`run_mesh.py`, phase 7b: `phase7b_mesh.py`)
Settings: `--node 2.5 --link 6`, the script's defaults.

1. *Graph:* track points are averaged into nodes on a grid of 2.5 voxels; nodes within 6 voxels are linked if their
   offset along each node's normal is $`\le 4`$ voxels (stays within one layer). The largest connected part is kept.
2. *First flat layout (landmark MDS):* 200 landmarks chosen farthest-first by graph (walking) distance; with
   $`D`$ the landmark-to-node walking distances and $`D_2`$ the landmark-to-landmark squared distances,
   $`B_m = -\tfrac12 J D_2 J`$ ($`J`$ the centring matrix), and nodes are placed by
   $`\mathbf{u} = -\tfrac12\,\Lambda^{-1/2} E^{\mathsf T}\left(D^{\circ 2} - \overline{D_2}\right)`$ using the top two
   eigenpairs $`(E, \Lambda)`$ of $`B_m`$. The third eigenvalue over the first is reported as the share that does not unroll flat.
3. *Refinement:* walking distances $`g_{ij}`$ up to 40 voxels from a source every ~8 voxels (plus all links) are kept;
   150 rounds of $`\mathbf{u}_i \leftarrow \frac{\sum_j w_{ij}\left(\mathbf{u}_j + g_{ij}\,\frac{\mathbf{u}_i-\mathbf{u}_j}{\lVert\mathbf{u}_i-\mathbf{u}_j\rVert}\right)}{\sum_j w_{ij}}, \qquad w_{ij} = g_{ij}^{-2},`$
   reduce the median relative error $`\big|\lVert\mathbf{u}_i-\mathbf{u}_j\rVert - g_{ij}\big|/g_{ij}`$ (typically 7–8% → 5–6%).
   Each track point gets $`(u,v)`$ through a local linear map from its node's links.
4. *Surface over the flat map:* a locally linear (Gaussian-weighted, $`\sigma = 5`$) fit of $`(x,y,z)`$ over $`(u,v)`$.
   Each point's depth across the sheet is $`w = (\mathbf{p} - S_0(u,v))\cdot\mathbf{N}_0(u,v)`$.
5. *Clean areas and set-aside:* where the local spread of $`w`$ (standard deviation, Gaussian window 3) is $`\le 3`$
   voxels the layer is *clean*. The surface is refitted from clean areas only, carried across the rest, and in the
   rest points with $`|w| > 5`$ are set aside as a neighbouring layer; a track with more than half its points set
   aside is set aside whole.
6. *Band middle:* along $`\mathbf{N}_0`$ at each pixel (searching $`\pm 12`$ voxels in steps of 0.5), the bright run
   ($`B \ge 0.5`$) nearest the kept tracks' mean depth (within 3 voxels of it) is found, limited to
   $`\pm(2.5\,\mathrm{sd}_w + 2)`$ clipped to 4–10 voxels (and to at most 6.5 outside clean areas); the mesh is placed at its middle
   (median filter 3, Gaussian smoothing 3). A run that reaches the limit is reported as *pressed*.
7. *Grid mesh* with vertices every 2 voxels; cells whose 3D/flat length ratio is beyond 2.5 (or below 1/2.5) are cut.
8. *Checks:* band-middle distance (p50 ≤ 0.5, p95 ≤ 2.0); coverage (one piece ≥ 90%, 97% of points within 6 voxels);
   bridged area ≤ 25%; no non-manifold edges; 90% of fitted vertices on papyrus; stretch p5 ≥ 0.90, p95 ≤ 1.10,
   cut ≤ 3%, kept ≥ 95%; kept tracks within width 12 and ≤ 2% beyond 8 voxels, set aside ≤ 20%; departure from
   the smooth surface reported.

### 8. Flat grid (`run_ink.py`, phase 8: `phase8_flat_grid.py`)
The mesh's flat layout is filled in to a 1-voxel grid by bilinear blending of each cell's four corners; holes up to
400 pixels are filled from their rim. Distance per pixel is reported in both directions (limit 0.80–1.25 for p5–p95).
The image is mirrored if needed so that it is seen from the scroll centre looking outward (normals outward), then
turned by quarter or half turns only (no mirroring) so the scroll axis runs down the image with high $`z`$ at the top,
which keeps the writing upright.

### 9–10. Ink (`run_ink.py`, phase 9: `phase9_ink_volume_extraction.py`, phase 10: `phase10_ink.py`)
The flat grid's 3D positions are smoothed (Gaussian, $`\sigma = 3`$, within the sheet) and normals are taken from the
grid's central differences, turned to point towards the scroll centre (inward). *Snapping:* along each normal the CT
is sampled from $`-6`$ to $`+6`$ voxels in steps of 0.5, smoothed along depth ($`\sigma = 1.5`$), weighted by a Gaussian
prior $`e^{-d^2/(2\cdot 3^2)}`$, and the surface is moved to the brightest depth (offsets median-filtered over 9 pixels
and smoothed with $`\sigma = 4`$). Then 21 slices are sampled (trilinear) at

```math
d_k = s + (k - 10)\,\Delta, \qquad k = 0,\dots,20, \qquad s = 5,\ \Delta = 1 ,
```

i.e. from 5 voxels outward to 15 voxels inward of the snapped surface, with positive $`d`$ towards the scroll centre
(the recto side). Slices are stored as 8-bit; outside the sheet they are filled with the sheet's mean grey.
Phase 10 runs a published 9 µm ink model (`step-075000.pth`) on the 21-slice stack through the challenge's
`vesuvius.ink_detection` inference (tiles overlapping by half, Hann blending), giving one ink probability per pixel.

### Scoring against human labels (`checks/compare_labels.py`)
Each pixel takes the label of the nearest official-surface point within 6 voxels (after moving our grid into the
official surface's frame). On human-checked pixels,

```math
\text{score} = P(\hat p > 0.5 \mid \text{ink}) - P(\hat p > 0.5 \mid \text{no ink}).
```

*Chance:* the label map is shifted by a random offset of up to 120 pixels in each direction, at least 60 in one
(300 draws; about 230 leave enough labelled pixels), and the score recomputed; the result beats chance if it exceeds the 95th percentile of these.
