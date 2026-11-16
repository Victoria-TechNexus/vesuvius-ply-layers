# The pipeline in equations

This page states what each step computes, with the default settings used for every result in the
[write-up](WRITEUP.md). Coordinates are voxels of the level-2 volume (9.6 µm); arrays are indexed $(z, y, x)$.
Sections marked *(to verify)* describe code not yet cross-checked line by line against this page.

**Notation.** $I(\mathbf{x}) \in [0,1]$ is the normalised CT; $B(\mathbf{x}) \in [0,1]$ is CT brightness rescaled
between its 20th and 90th percentile after a Gaussian blur ($\sigma = 1$), with $B \ge 0.5$ counted as papyrus.
$\mathbf{n}(\mathbf{x})$ is the unit surface normal field, $\mathbf{t}(\mathbf{x})$ the unit fibre (track) direction.
A *track* is a polyline $\gamma_i$; a *piece* is a group of tracks from phase 5; a *sheet* is one piece or several
joined pieces; a *layer* is one turn of papyrus (recto ply + verso ply).

---

## Stage 1: tracks, pieces, sheets (`run_pipeline.py`, one tile)

### 1. CT *(to verify)*
A tile of $192 \times 416 \times 416$ voxels is streamed from the OME-Zarr volume and normalised to $[0,1]$.

### 2. Orientation fields *(to verify)*
Structure tensor of the smoothed gradient,
$$J_\rho(\mathbf{x}) = K_\rho * \left(\nabla I_\sigma \, \nabla I_\sigma^{\mathsf T}\right), \qquad \sigma = 1,\ \rho = 6 .$$
Its leading eigenvector gives the sheet normal $\mathbf{n}$; the in-plane fibre direction gives $\mathbf{t}$.
A 3D U-Net adds per-voxel probabilities, including a verso probability $p_V(\mathbf{x})$.
All are stored in one field stack (channels 0–2 $\mathbf{t}$, 3–5 $\mathbf{n}$, 6–8 probabilities).

### 4. Tracking *(to verify)*
Seeds are ridge points with sheet probability $\ge 0.6$ (round 0, capped at 45,000) and then $\ge 0.5$ among points
not yet covered (round 1, capped at 15,000). Each seed is integrated along the fibre direction with RK4,
$$\frac{d\gamma}{ds} = \pm\,\mathbf{t}(\gamma(s)), \qquad \Delta s = 0.5,$$
the sign kept consistent from step to step. A track stops at the tile edge, where sheet probability drops below
0.5, where $p_V > 0.6$ (verso reached), or where curvature exceeds $\kappa_{\max} = 0.5$; stopped tracks are
trimmed by 1 voxel. Tracks with 30+ points are kept.

### 5. Pieces: affinity graph between tracks (`phase5_affinity_graph_wrap_disambiguation.py`)
Every third track point is used. For point pairs $(i, j)$ on different tracks within $r_{\max} = 3.5$ voxels, with
mean normal $\bar{\mathbf{n}} = (\mathbf{n}_i \pm \mathbf{n}_j)/\lVert\cdot\rVert$ (sign made consistent) and offset
$\mathbf{d} = \mathbf{p}_j - \mathbf{p}_i$:
$$d_\perp = |\mathbf{d}\cdot\bar{\mathbf{n}}|, \qquad d_\parallel^2 = \lVert\mathbf{d}\rVert^2 - d_\perp^2 .$$
Pairs with $d_\perp > 1$ are dropped, as are track pairs with fewer than 3 such point pairs. The weight is
$$w_{ij} = \exp\!\left(-\frac{d_\perp^2}{2\sigma_n^2}\right)\exp\!\left(-\frac{d_\parallel^2}{2\sigma_\parallel^2}\right)
|\mathbf{t}_i\cdot\mathbf{t}_j|^{\alpha}\;\exp\!\left(-\frac{\lVert\mathbf{d}\rVert\,\overline{p_V}}{s_V}\right),$$
with $\sigma_n = 1$, $\sigma_\parallel = 8$, $\alpha = 2.5$, $s_V = 0.15$, and $\overline{p_V}$ the mean verso
probability at 9 points along the segment. Pairs with $w_{ij} < 0.005$ are dropped.

*Valley test.* Along each segment, with $v_k$ the blurred CT at 9 points and $v_e = \min(v_0, v_8)$,
$$V_{ij} = \max\!\left(0,\ \frac{v_e - \min_k v_k}{v_e}\right).$$
Point pairs are grouped per track pair (weight = max, valley = median); a track pair with median valley $> 0.05$
(a dark dip between them) is removed. Tracks are then grouped into pieces by Louvain community detection on the
weighted track graph (resolution 3.0).

### Joining pieces into sheets
For two pieces $X, Y$ with at least 100 points within 8 voxels of each other, sampled points $\mathbf{x}\in X$ and
their nearest $\mathbf{y}\in Y$ give two measures:

- *Apart share* $a_{XY}$: the fraction of points whose neighbours in the other piece (within 2 voxels sideways)
  sit at median depth $\ge 2.5$ voxels along the normal, i.e. a layer away. The larger of the two directions is used.
- *Bright share* $b_{XY}$: the fraction of segments $\mathbf{x}\to\mathbf{y}$ whose darkest of 9 samples has $B \ge 0.5$.

Pieces are joined, largest overlap first, if $a_{XY} \le 0.20$ and $b_{XY} \ge 0.90$, unless the join would put two
pieces with $a \ge 0.5$ into one sheet (union-find with a veto).

*Stacked share* of a sheet (reported per sheet, used in stage 2): the fraction of 1,500 sampled points that have
another point of the same sheet 6–14 voxels away along the normal and within 4 voxels sideways. About 0 for a
single layer; high when two layers are stuck together.

---

## Stage 2: layers (one tile)

### Splitting stacked joins (`split_sheets.py`)
Every joined sheet with 2+ pieces and stacked share $\ge 0.2$ is returned to its pieces (new label = piece +
offset); all other sheets are kept.

### Pair table (`pair_table.py`)
From up to 120,000 start points, walk outward along the local normal to the next sheet within 30 voxels and record
the gap, the darkest CT between, the band thickness and the angle between the two sheets. Each sample is classed
(separate / far / thin / pressed / ambiguous / edge); pairs of sheets with 20+ samples go to the ply reader.

### Reading the plies (`ply_label.py`)
At up to 2 spots per pair, a surface is fitted to the inner sheet $A$ on a $49 \times 49$ grid turned 45° to the
scroll axis, and face-on CT slices are cut at depths $k$ from 15 voxels inside $A$ to 15 beyond the outer sheet $B$.

*Fibre direction per slice.* In each slice $S_k$ (high-passed by subtracting a $\sigma=4$ blur), a 2D structure
tensor with gradient scale 1 and integration scale $\rho = 4$, restricted to papyrus pixels, gives at every pixel
the orientation $\phi$ and coherence
$$c = \frac{2\sqrt{\tfrac14 (J_{rr}-J_{cc})^2 + J_{rc}^2}}{J_{rr}+J_{cc}} .$$
The angle $\theta$ between the stripe direction and the scroll axis is computed. Among coherent pixels
($c \ge 0.3$), with $h = P(\theta \ge 60°) - \tfrac13$ and $v = P(\theta \le 30°) - \tfrac13$, depth $k$ is labelled
$$\ell_k = \begin{cases} \texttt{.} & \text{if more than half the patch has } B < 0.5\ (\text{gap})\\
\texttt{H} & \text{if } h \ge 0.2 \text{ and } h > v\ (\text{fibres around the scroll: recto})\\
\texttt{V} & \text{if } v \ge 0.2 \text{ and } v > h\ (\text{fibres along the axis: verso})\\
\texttt{?} & \text{otherwise}\end{cases}$$
and single-slice runs of H or V become `?`. This is done for the central $25\times25$ window and separately for
each quarter of the patch (quarter agreement is recorded).

*Rules.* Let $A$, $B$ be the plies at the two sheets' depths (searching ±3 slices), and the run sequence between them
read from $A$ outward to $B$:

| condition | label |
|---|---|
| some V is followed later by an H | different layers |
| no V anywhere in the window | unreadable |
| both sheets in one papyrus band, $A=B$, depths $< 7$ voxels apart | same ply |
| both sheets in one band, $A=\mathrm{H}$, $B=\mathrm{V}$ | same layer |
| separate bands; $A$'s band already holds a V outward of $A$, or $B$'s band holds an H inward of $B$ | likely different layers |
| separate bands, $A=B$, gap $\ge 8$ | likely different layers |
| separate bands, $A=\mathrm{H}$, $B=\mathrm{V}$, gap $\le 12$ | same layer, delaminated |
| separate bands, $A=\mathrm{H}$, $B=\mathrm{V}$, gap $> 12$ | possibly delaminated |

Per pair, readings are combined: weak labels (*likely different layers*, *possibly delaminated*) give way to strong
ones; a mix of "same" and "different" families is a *conflict*.

### Stacking (`stack_layers.py`)
1. *Direction across the stack:* $\mathbf{n}_0$ = leading eigenvector of $\sum \mathbf{n}\mathbf{n}^{\mathsf T}$ over
   sampled track points, oriented away from the scroll centre.
2. *Columns:* the tile is cut into columns 3 voxels wide along $\mathbf{n}_0$; within each column the passes of sheets
   (or tracks) are ordered by depth $t$.
3. *Neighbour facts:* for consecutive passes with gap $\le 60$, sample $B$ every 0.5 voxels along the segment and
   count dark runs ($B < 0.5$) of length $\ge 3$ voxels, $n_{\text{dark}}$. Then
   $$\Delta = \begin{cases} 0 & n_{\text{dark}} = 0 \text{ and gap} < 20 \\ \max(1, n_{\text{dark}}) & \text{otherwise.}\end{cases}$$
   *Ply override:* for passes of a labelled pair within 30 voxels of each other and within 40 voxels of a spot that
   gave that answer, "different layers" turns $\Delta = 0$ into 1 (pressed) and "same layer / delaminated / same ply"
   turns $\Delta = 1$ into 0 (delaminated).
4. *Sheet level:* per sheet pair, $d$ = rounded median of signed $\Delta$ over shared columns (5+ columns);
   pairs where both directions occur in $\ge 20\%$ of columns are dropped as conflicts; weight $w$ = number of columns
   (×0.3 if mostly more than one layer apart). Integer positions $L_s$ are found by a maximum-weight spanning
   placement from the largest sheet, then up to 50 rounds in which each sheet takes the position with the largest
   summed weight of votes $L_{s'} - d_{ss'}$.
5. *Track level:* each track starts at its sheet's position and, for up to 60 rounds, moves to the position most of
   its column neighbours vote for ($L_{\text{nb}} \pm \Delta$, its sheet's position counting one extra vote; half the
   movers move per round).
6. *Self-check:* each ply fact the final layers break is listed; each "different layers" fact whose two sheets are
   joined by a chain of "same" facts is a contradiction, traced to the "same" fact appearing in the most chains.

---

## Stage 3: one layer across tiles (`combine_tiles.py`)
In the overlap between the layer so far and a new tile, for each layer $\lambda$ of the new tile:
$$m_\lambda = \#\{\text{points of } \lambda \text{ in the overlap within 3 voxels of a reference-layer point}\},$$
and the layer with the largest $m_\lambda$ is taken (its share of overlap points is reported). Tracks are moved into
one coordinate frame; the CT, brightness and normals are stitched, each voxel taken from the tile whose edge is
furthest away.

---

## Stage 4: mesh, flat grid, ink (one layer)

### Mesh and unrolling (`phase6_track_surface.py`)
1. *Graph:* track points are averaged into nodes on a grid of 2.5 voxels; nodes within 6 voxels are linked if their
   offset along each node's normal is $\le 4$ voxels (stays within one layer). The largest connected part is kept.
2. *First flat layout (landmark MDS):* 200 landmarks chosen farthest-first by graph (walking) distance; with
   $D$ the landmark-to-node walking distances and $D_2$ the landmark-to-landmark squared distances,
   $B_m = -\tfrac12 J D_2 J$ ($J$ the centring matrix), and nodes are placed by
   $\mathbf{u} = -\tfrac12\,\Lambda^{-1/2} E^{\mathsf T}\left(D^{\circ 2} - \overline{D_2}\right)$ using the top two
   eigenpairs $(E, \Lambda)$ of $B_m$. The third eigenvalue over the first is reported as the share that does not unroll flat.
3. *Refinement:* walking distances $g_{ij}$ up to 40 voxels from a source every ~8 voxels (plus all links) are kept;
   150 rounds of
   $$\mathbf{u}_i \leftarrow \frac{\sum_j w_{ij}\left(\mathbf{u}_j + g_{ij}\,\frac{\mathbf{u}_i-\mathbf{u}_j}{\lVert\mathbf{u}_i-\mathbf{u}_j\rVert}\right)}{\sum_j w_{ij}}, \qquad w_{ij} = g_{ij}^{-2},$$
   reduce the median relative error $\big|\lVert\mathbf{u}_i-\mathbf{u}_j\rVert - g_{ij}\big|/g_{ij}$ (typically 7–8% → 5–6%).
   Each track point gets $(u,v)$ through a local linear map from its node's links.
4. *Surface over the flat map:* a locally linear (Gaussian-weighted, $\sigma = 5$) fit of $(x,y,z)$ over $(u,v)$.
   Each point's depth across the sheet is $w = (\mathbf{p} - S_0(u,v))\cdot\mathbf{N}_0(u,v)$.
5. *Clean areas and set-aside:* where the local spread of $w$ is $\le 3$ voxels the layer is *clean*. The surface is
   refitted from clean areas only, carried across the rest, and in the rest points with $|w| > 5$ are set aside as a
   neighbouring layer.
6. *Band middle:* along $\mathbf{N}_0$ at each pixel, the bright run ($B \ge 0.5$) nearest the kept tracks' mean depth,
   limited to $\pm(2.5\,\mathrm{sd}_w + 2)$ clipped to 4–10 voxels, is found; the mesh is placed at its middle
   (median filter 3, Gaussian smoothing 3). A run that reaches the limit is reported as *pressed*.
7. *Grid mesh* with vertices every 2 voxels; cells whose 3D/flat length ratio is beyond 2.5 (or below 1/2.5) are cut.
8. *Checks:* band-middle distance (p50 ≤ 0.5, p95 ≤ 2.0); coverage (one piece ≥ 90%, 97% of points within 6 voxels);
   bridged area ≤ 25%; no non-manifold edges; 90% of fitted vertices on papyrus; stretch p5 ≥ 0.90, p95 ≤ 1.10,
   cut ≤ 3%, kept ≥ 95%; kept tracks within width 12 and ≤ 2% beyond 8 voxels, set aside ≤ 20%; departure from
   the smooth surface reported.

### Flat grid (`mesh_to_grid.py`)
The mesh's flat layout is filled in to a 1-voxel grid by bilinear blending of each cell's four corners; holes up to
400 pixels are filled from their rim. Distance per pixel is reported in both directions (limit 0.80–1.25 for p5–p95).
The image is mirrored if needed so that it is seen from the scroll centre looking outward (normals outward), then
turned by quarter or half turns only (no mirroring) so the scroll axis runs down the image with low $z$ at the top.

### Ink (`read_ink.py`, phase 9 and 10) *(to verify)*
For each pixel, 21 samples are taken along the surface normal at offsets $d = s + k$, $k \in \{-10,\dots,10\}$,
with shift $s = 5$ voxels (towards the recto) after snapping the surface to the papyrus peak. Outside the sheet the
slices are filled with the sheet's mean grey. The ink model maps the 21-slice stack to a 2D ink probability.

### Scoring against human labels (`compare_labels.py`)
Each pixel takes the label of the nearest official-surface point within 6 voxels (after moving our grid into the
official surface's frame). On human-checked pixels,
$$\text{score} = P(\hat p > 0.5 \mid \text{ink}) - P(\hat p > 0.5 \mid \text{no ink}).$$
*Chance:* the label map is shifted by a random offset of 60–120 pixels (300 draws, about 230 valid) and the score
recomputed; the result beats chance if it exceeds the 95th percentile of these.
