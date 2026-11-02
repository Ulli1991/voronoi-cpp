# 1D, 2D and 3D: design note (2026-11-02, draft for the owner)

Goal: native 1D and 2D Voronoi meshes and the finite-volume hydro on top of them,
so that LUNA's standard tests can run in their own dimension: 1D Sod, Shu-Osher and
Woodward-Colella; 2D Kelvin-Helmholtz after McNally, Lyra & Passy 2012; 3D Sedov
(which runs today). Base: main 7a7adb8. Nothing in this note has been implemented
or measured. Every number for 1D or 2D is an **estimate**, marked as one, and
reasoned from the measured 3D numbers in README.md and the research notes.

Order of the criteria (owner, 2026-11-02): **throughput first**, then accuracy (as
good as LUNA at the same resolution and order), then exactness, memory and lean
code. Templates must cost 3D nothing. Wherever a generic design would cost speed in
any D, this note picks a specialised implementation and says why.

## 0. Summary

- **D is a template parameter only where the 3D instructions stay the same**: the
  box and snapping (`BoxN<D>`), the snapped `Points`, Morton keys and the cell list,
  the face and cell arrays (`Geometry`, `CellFaces`, the per-cell face lists of
  `tet_mesh::cells`), and the hydro (`State` with D + 2 entries, `Vec<D>`). The
  current names stay as 3D aliases (`Box = BoxN<3>`,
  `FiniteVolume<E, R> = FiniteVolumeD<3, E, R>`), so 3D callers compile unchanged.
- **The mesh engines are specialised, not templated**. D = 1 is a sorted order with
  no predicates, clipping or simplices. D = 2 is a new triangle engine: polygon
  clipping, triangles with Lawson 2-2 flips, faces with exactly two vertices. D = 3
  is today's tetrahedral code, unchanged. In each D, a generic simplex engine would
  carry 3D's costs (ring pools, edge walks, 2-3/3-2 bookkeeping, free slots) into 1D
  and 2D, where none of them is needed.
- **Boundaries**: the library stays periodic. Walls (reflective, zero-gradient,
  inflow) belong to the hydro layer. In 1D they are explicit boundary faces (two per
  run). In 2D and 3D they would use mirror particles in a doubled periodic box;
  none of the target 2D/3D tests needs walls.
- **Stages**: 0 (plumbing; 3D bit-identical, within 1 %), 1 (1D mesh, hydro and walls,
  with Sod, Shu-Osher and WC against LUNA), 2 (2D mesh), 3 (2D hydro and McNally KH),
  4 (optional: a faster 2D kernel, a 2D host fallback, walls in 2D/3D). Effort:
  about 3,000-3,500 new lines including tests, about 9-14 agent-days in all.
- **Refinement and derefinement** (section 11):
  - criteria in the dt kernel, so there is no cost when nothing is flagged;
  - 1D by re-sort; 2D/3D by point location and split, then the existing flip repair,
    or cavity insertion (choice pending the literature report);
  - removal by a device local Bowyer-Watson of the star (3D) and the ear rule (2D);
  - an exact remap from the mesh's volumes before and after (no polyhedron
    intersection), under an independence rule;
  - stable ids with holes.

  Stages R0-R5, ~1,650 lines, 8-12 agent-days.

## 1. What exists, and where it is 3D

| piece | file:line | 3D-specific? |
|---|---|---|
| `Box`: isotropic step = longest/2^30, `period[3]` | `exact_predicates.hpp:48-70` | arrays of 3; the logic is generic in D |
| `snap`, `delta`, `Offset` | `exact_predicates.hpp:30-34, 75-94` | `Offset` is `i64 v[3]` |
| `orient` (det3, 128-bit), `in_sphere` (256-bit `Wide`, perturbation by id) | `exact_predicates.hpp:140-243` | 3D formulas; 2D needs orient2d/incircle, 1D needs neither |
| `Points` (int*[4], padded) and `snap(exec, ...)` | `exact_predicates.hpp:248-273` | padding of 4 |
| `CellList`: tiles, Morton keys `spread3`, octree levels (`count / 8^k`) | `neighbor_search.hpp:49-64, 67-90, 154-285` | `3 *` in the key bits, `/ 3.0` in the level |
| clipping per thread: `RoundedCell`, `ExactCell`, `clip`, `construct_pass` | `shell_voronoi.hpp:435-735, 751` | polyhedra |
| warp clipping: `Polyhedron`, `Kernel`, `construct_warp` | `shell_voronoi_warp.hpp:44, 309, 634` | polyhedra |
| tetrahedra: `Mesh` (vertex/adjacent quads, centre, edge bits), `from_rings`, `FaceRings`, `update_face_rings`, `circumcentres`, `faces` | `tet_mesh.hpp:34-45, 181, 384, 503, 656, 681` | simplices of 4, rings of variable length |
| per-cell face lists, volumes, centroids: `cells`, `pyramid` (\|A.d\|/6, centroid at 3/4) | `tet_mesh.hpp:729-926; 752, 779-786` | only the constants 1/6 and 3/4 (in D: 1/(2D) and D/(D+1)) |
| exact start: `check`, `build` | `tet_start.hpp:72, 153` | tetrahedra |
| device repair: screening, 2-3/3-2 plans, votes by hashed index | `tet_repair.hpp:384, 199-276, 283, 446` | 2-3/3-2 |
| host fallback: local Bowyer-Watson | `tet_flips.hpp:110, 250, 399` | tetrahedra |
| `voronoi::Mesh` (update, rebuild, clip, scratch, update_cells, accessors) | `voronoi_mesh.hpp:100-342` | owns the 3D engines; its interface is dimension-free apart from `[3]` |
| Euler: `State` = 5, `Vec3`, `half_step` loops d < 3, HLL/HLLC in box axes | `euler.hpp:14-181` | count 3 only; the form is dimension-free (`euler.hpp:84-92`) |
| `FiniteVolume`: gradients 4x4 LSQ, `double*[15]`, time step `cbrt(3V/4pi)`, face angle `sqrt(A/pi)/(d/2)`, drift wrap with `box()` | `finite_volume.hpp:74-559; 147, 283, 305-311, 335-418` | the constants and sizes listed |

## 2. Architecture

### 2.1 Shared and templated on D (zero cost to 3D, checked by bits and timing)

1. **`exact_predicates::BoxN<D>`**: `side[D]`, `inverse[D]`, `step`, `period[D]`, the
   same rounding rule (`exact_predicates.hpp:56-66`), plus `using Box = BoxN<3>`
   with the implicit conversion from a double kept (box3 note, "Interface").
   `snap(x, box, axis)` and `delta` are unchanged: they are per axis already.
   `Offset` becomes `OffsetN<D>`, with `Offset = OffsetN<3>`.
2. **Snapped points**: `Points<Memory, D>`, padded to 4 ints for D = 3 (as now: one
   aligned 16-byte load) and to 2 for D = 2 (8 bytes). D = 1 uses a plain `int*`.
   Double positions: `Points4` for D = 3 (as now, 32 bytes), `double*[2]` for D = 2
   (16 bytes, one aligned load), `double*` for D = 1.
3. **Cell list for D = 2 and 3**: `CellList<Exec, D>` with `spread2`/`spread3`
   chosen by `if constexpr`, key bits `tiles + D (levels + fine_levels)`, level
   `count / 2^(D k)`, and a stencil of offsets in (2·4+1)^D cells
   (`shell_voronoi.hpp:146-178`). For D = 3 every expression is textually today's
   (`3` written as `D` with D a compile-time constant), so the instructions are the
   same. 1D does not use the cell list (2.3).
4. **Face and cell arrays**: `Geometry` and `CellFaces` with `double*[D]`, and the
   per-cell face lists and ordered sums of `tet_mesh::cells` (`tet_mesh.hpp:869-924`)
   with `pyramid` = \|A.d\| / (2D) and the pyramid centroid at D/(D+1) of the way
   from the generator to the face centroid (3D: 1/6 and 3/4,
   `tet_mesh.hpp:779-786`). The warp sort variant `CellGeometryWarp` serves 2D
   unchanged. 1D gets its own trivial cell pass (2.3), because the generic one pays
   two atomic passes and a sort for lists that are always two long.
5. **Hydro**: `FiniteVolumeD<D, Exec, Riemann>` (section 5).
6. **`voronoi::Mesh`**: one interface, three engines (section 6).

How "zero cost" is enforced: for D = 3 every templated function must compile to the
same arithmetic in the same order. CUDA contracts `a*b + c` into FMA by default, so a
reordered expression changes bits. The rule is therefore textual: loops `for (a < D)`
replace exactly the existing 3-loops (`exact_predicates.hpp:271`,
`neighbor_search.hpp:205`, `voronoi_mesh.hpp:53`), and `dot`/`cross` of `Vec<3>` keep
today's left-to-right sum (`shell_voronoi.hpp:55`, `euler.hpp:22`). The acceptance
gate is the bench's `cells_checksum` (box3 note: 9227075244623139925 on jittered
2.1M) and a hydro snapshot checksum, both bit-identical, plus the 1 % rule (stage 0).

### 2.2 Specialised per D (where generic code would cost speed)

| job | 1D | 2D | 3D (today) |
|---|---|---|---|
| exact predicates | integer compare of snapped keys, ties by id | orient2d in i64 (exact, no filter needed), incircle with a double filter and then i128, perturbation by id | orient (i128), in_sphere (filter, 256-bit), perturbation |
| construction | radix sort of (key, id) | polygon clipping per thread (rounded + exact cells), rings = ordered neighbour cycle of a cell | polyhedron clipping, warp per cell, face rings |
| persistent mesh | the sorted order | triangles: vertex[3], adjacent[3], centre (x, y, r²), 3 edge bits | tetrahedra, ring pools |
| start | the sort is exact | `from_rings2` + exact check + exact patch (as tet_start) | `from_rings` + `tet_start::build` |
| repair | re-sort (or a check pass, then sort only if needed) | 2-2 Lawson flips, votes as in tet_repair, fixed slot count | 2-3/3-2 flips, free slots, growth |
| host fallback | none needed | none at first (device rebuild); host Bowyer-Watson only if measurements want it | tet_flips |
| faces | n faces at the midpoints | one per triangulation edge: the two circumcentres of its triangles | rings of circumcentres |

Why a separate engine per D:

- **1D**: the Delaunay "triangulation" is the cyclic sorted order. Clipping,
  predicates and simplices would cost launches and memory for nothing.
- **2D**: a face has exactly two vertices, the circumcentres of the two triangles
  across an edge, which `adjacent` already gives. So 2D needs no ring pool, no
  `walk_edge` (`tet_mesh.hpp:126`), no compaction (`update_face_rings`,
  `tet_mesh.hpp:503-630`), and no free slots or growth. A 2-2 flip rewrites two
  triangles in place, and on a torus the triangle count is fixed at 2n (Euler).
  Running 3D's machinery with "rings of length 2" would keep its memory (ring pools
  are a large part of 2221 B/cell; hydro_stage3: pools at 688 B/cell at 256³) and its
  extra passes.
- **3D**: unchanged files. The 3D engine keeps its names (`tet_mesh`, `tet_repair`, ...);
  the 2D engine gets new headers (`tri_mesh.hpp`, `tri_repair.hpp`, `tri_start.hpp`,
  `polygon_clip.hpp`), and 1D one header (`sorted_mesh.hpp`).

## 3. 1D: sorted generators, the minimal exact design

**Construction (exact)**:

1. Snap: k(i) = `snap(x(i), box, 0)` in [0, period). This is the same snapping as
   now, so a 1D box has 2^30 steps.
2. Sort the pairs (k, i) stably by k (CUB radix over `bits(period)` = 30 bits, as
   `neighbor_search.hpp:217-223`; host `std::sort` elsewhere). Equal keys keep id
   order: this is the 1D symbolic perturbation.
3. The mesh is the cyclic order `order[0..n)`. Face s lies between slots s and
   s + 1 mod n (n faces), with owner the lower id and neighbour the higher.
   With d = (k(next) − k(this)) mod period (always ≥ 0 in the sorted cyclic order):
   - area vector `±1` (+1 if the neighbour lies at +d from the owner);
   - face centroid d/2 from the owner, or −d/2 if the owner is the right-hand one;
   - all in grid steps × `step`, i.e. exact rationals of the snapped points.

   Cell volume = (d_left + d_right) / 2; centroid relative to the generator =
   (d_right − d_left) / 4.
4. Per-cell face lists: always two entries, at `cell_face_start(i) = 2 i`, ordered
   by the other particle's id (the 3D contract, `voronoi_mesh.hpp:28-31`). Written
   directly by the face kernel, without atomics or sorting, so bit-reproducible by
   construction.

This is AREPO's 1D mesh too: sorted cells, faces at the midpoints, volume = the
distance between the faces (`/u/uli/codes/arepo/src/voronoi_1d.c:184-213`, the sort
in `voronoi_1D_order`, `:229-260`).

**Moving mesh**: `update(x)` snaps and sorts again, in one CUB sort and two
element-wise kernels (fused where possible: snap+keys, then faces+cells). A cheaper
path, "check whether the previous order is still sorted, sort only if not", is an
option after measurement. Particles wrapping through x = L rotate the cyclic order,
which the check must allow. Crossings are unlikely at Courant 0.3: the pair criterion
`(d + r) / vsig` (`finite_volume.hpp:151-159`) limits the approach of neighbours per
step to about 0.3 (d + r) < d. They are not impossible with regularisation, and the
sort handles them exactly.

**Limitations kept from 3D**: snapped duplicates make zero-length cells
(README "Limitations"). n ≥ 3, so that no pair has two faces.

**Walls (1D only, needed by the tests)**: a non-periodic 1D box [0, L] with the end
cells extended to the walls: first cell [0, (x₀+x₁)/2], last cell up to L. There are
two boundary faces, with `neighbour = -1 - side` (−1 left, −2 right). The mesh code is
the same minus the wrap. The hydro gives each boundary face a ghost state:

- reflective: the cell's own state with the normal velocity negated;
- zero-gradient/outflow: the cell's own state (LUNA BOUNDARY = 1);
- inflow: a fixed state.

This is AREPO's mirror ghost in 1D, without the ghost particle
(`voronoi_1d.c:124-176`: mirror at −x₀ and 2L − x_{n−1}; the velocity is flipped in
`face_boundary_check`, `finite_volume_solver.c:540-542, 2206-2280`).

**Throughput, 1D (estimates)**:

- At n ≥ 2M the work is bandwidth-bound: keys 8 B, ids 4 B, about 6 passes of
  ≈ 20-30 B/cell. With the A100 40 GB's ~1.5 TB/s, a rebuild = update estimates to
  **~1-3 G cells/s**. CUB's radix sort is the largest single cost; its rate is not
  measured here.
- LUNA's 1D tests use N = 128-4096 (8192-65536 for references). There a step is
  **launch-bound**: about 10-15 kernels of ~5 µs gives 50-80 µs per step, i.e.
  **~50 M cell updates/s at N = 4096**, whatever the kernels do.
- So 1D design rule 1 is the fewest launches: fuse the mesh update into one sort plus
  one kernel, and the hydro into gradients + fluxes/update (2 kernels: in 1D a cell's
  neighbours are slots s ± 1, so no face list gather is needed; see 5.3).
- Design rule 2 is CUDA graphs (Kokkos::Experimental::Graph) or a multi-step kernel
  for the small ladders. To measure, not assume.

## 4. 2D: polygons, triangles and 2-2 flips

### 4.1 Exact predicates (`exact_predicates_2d.hpp`, ~150 lines)

- Offsets are nearest-image differences of 30-bit snapped coordinates, so
  \|d\| ≤ 2^29 (`delta`, `exact_predicates.hpp:89-94`).
- **orient2d** = d1x d2y − d1y d2x: products ≤ 2^58, sum ≤ 2^59, exact in `i64`
  always. No filter: one 64-bit multiply pair is cheaper than a filter and fallback
  on the GPU (to measure: the A100 emulates 64-bit integer multiplies).
- **incircle** (expanded along the lift column like `in_sphere`,
  `exact_predicates.hpp:197-243`): lifts ≤ 2^59, cofactors (orient2d) ≤ 2^59,
  products ≤ 2^118, sum of 3 terms ≤ 2^120, so exact in `i128`. No 256-bit `Wide`.
  First a double filter with a permanent bound (as `in_sphere:204-215`), then i128,
  then the symbolic perturbation: the cofactor of lift i is ± the orient2d of the
  other three, taken in increasing id (`:224-241` with 4 points instead of 5).
- **Exact circumcentre** (sliver triangles, exact cells):
  n = \|a\|² b⊥ − \|b\|² a⊥ in i128, denominator 2·orient2d (as
  `exact_predicates.hpp:158-166`).
- Tests: as `exact_predicates_test` (an independent integer evaluation,
  near-degenerate and 2^30-scale inputs, ties consistent from every corner order,
  thin slabs).

### 4.2 Clipping (`polygon_clip.hpp`, ~450 lines)

- A cell is a convex polygon: a cyclic list of edges, each with the particle beyond
  it. Vertex m is where edges m and m+1 meet. A bisector cuts off a **contiguous arc**
  of vertices (convexity), which is replaced by two new vertices. There is no hole
  topology, no adjacency bytes and no face walk, unlike `clip`
  (`shell_voronoi.hpp:628-735`).
- The walk (stencil, nearest-first candidates, the 2R certification, the level fitted
  to density, one periodic image) is today's, with the 2D stencil and the starting
  square of half-width ¼ shortest side bounded by 4 virtual particles
  (`start_half_width`, `start_cube`, `shell_voronoi.hpp:409, 556`).
- Two cell kinds, as in 3D (`shell_voronoi.hpp:435-496`):
  - `RoundedPolygon`: double vertices, cut with a tolerance;
  - `ExactPolygon`: a vertex = the circumcentre of the generator and its two edge
    particles as exact (w, o), inside = the filtered incircle with the exact
    perturbed fallback, as `inside` (`:530-544`).
- **Ring output for the start** = the cell's ordered neighbour cycle (in 3D a ring
  belongs to a face; in 2D it belongs to a cell). One pool block per cell, as
  `Cells::with_rings`.
- **Kernel shape (throughput-first; to measure)**:
  - (a) thread per cell. A 2D Voronoi cell has 6 edges on average (the mean Delaunay
    degree in 2D), and the polygon fits in ~32 entries of registers/local memory.
  - (b) a team of 8 lanes per cell (4 cells per warp), testing candidates 8 at a time
    with a sub-warp sort, as the 3D warp kernel does with 32
    (`shell_voronoi_warp.hpp:12-21`).

  In 3D, teams of 16 or 8 lanes were 2x and 3.6x slower than a warp, but a polyhedron
  of ~30 vertices is a warp's work and a polygon of ~6 is not. Start with (a),
  because it is also the exact patch kernel (one code path); write (b) only if (a) is
  far below the estimate below.
- Overflow (more than the capacity of edges): rebuild with a big polygon, as `Pass`
  (`shell_voronoi.hpp:745`).

### 4.3 Persistent triangle mesh (`tri_mesh.hpp`, ~350 lines)

- `Triangles`: `vertex` int*[3] and `adjacent` int*[3] (across the edge opposite
  corner k), and `centre` double*[4] (x, y, r², pad: one 32-byte sector, as
  `Points4`). `edges` uint8 holds 3 bits: the representative of edge k, as
  `tet_mesh::Mesh::edges` (`tet_mesh.hpp:41-44`). Whether int*[3] or a padded int*[4]
  is faster is to be measured.
- **Faces**: a scan over the edge bits numbers the faces, as `number_faces`
  (`tet_mesh.hpp:444-488`) without the ring branch. Face f stores t·3 + k (one int).
  Its vertices are `centre(t)` and `centre(adjacent(t, k))`; its owner and neighbour
  are the two particles of edge k.
- **Face geometry**: area vector = (c₂ − c₁)⊥, oriented by its dot product with
  d_ij; centroid = the midpoint of c₁, c₂ relative to the owner.
- **Face renumbering after flips is cheap**: in 3D only the faces of touched
  tetrahedra are walked again and their rings appended (`update_face_rings`). In 2D a
  face's geometry depends only on its two triangles' centres, so after a step:
  1. centres of the touched triangles;
  2. the face scan;
  3. one face-geometry pass.

  No pool, no compaction.
- **Memory per cell (estimate, 2 triangles and 3 faces per cell)**:
  - triangles: 2·(12 + 12 + 32 + 1) ≈ 114 B;
  - faces: 3·(owner, neighbour, t·3+k = 12 B, area 16 B, centroid 16 B) ≈ 132 B;
  - repair state (alive is not needed: slots are fixed; touched, dirty, lock 8 B,
    listed) ≈ 2·(1 + 4 + 8 + 4) ≈ 34 B;
  - cell lists: 6 entries × 2 ints + start ≈ 52 B; positions, snapped points,
    volumes, centroids ≈ 16 + 8 + 8 + 16 B;
  - cell list for rebuilds ≈ 40 B.

  Total **~430-500 B/cell**, against 2221 B/cell in 3D. A 2D mesh of ~60-80M cells
  would fit in the A100's 40 GB with the hydro's ~100-150 B/cell (estimate).

### 4.4 Exact start (`tri_start.hpp`, ~250 lines)

- `from_rings2`: triangle (i, q_m, q_{m+1}) is emitted by its smallest particle, as
  in 3D (`tet_mesh.hpp:175-181`). Adjacency across (i, q_m) is the neighbouring
  triangle in i's own ring; across (q_m, q_{m+1}) it is found in the smaller of the
  two particles' rings.
- `check` (as `tet_start::check`, `tet_start.hpp:62-104`): no flat triangle; mutual
  adjacency; apexes on opposite sides; local Delaunay under the perturbed incircle;
  every particle used; areas summing to the box. Together these make it the Delaunay
  triangulation.
- `build`: patch the cells around bad triangles with the exact polygon kernel, and
  check again (`tet_start.hpp:105-215`).

### 4.5 Repair (`tri_repair.hpp`, ~300 lines)

- Screen, as `move_screened` (`tet_repair.hpp:384-441`): the far vertex of a
  neighbour inside or near the circumcircle, or a badly shaped triangle.
- Plan a 2-2 flip for every failing edge. It is valid when the quadrilateral is
  strictly convex (two orient2d signs), and it must "close its cavity": both new
  triangles positively oriented (as `closes`, `tet_repair.hpp:102-136`).
- Vote with the hashed index on the 2 triangles and their 4 outer neighbours
  (`priority`, `:279-289`, the one-thread livelock fix), apply the winners, and mark
  the new triangles dirty.
- Slots are reused in place: no free list, no `reserve`/growth (`:309-323`), no
  `touched_entries` overflow mode.
- Lawson's theorem: from a valid (non-inverted) triangulation, 2-2 flips reach the
  Delaunay triangulation. After a step, inverted triangles can only come from points
  that moved through an edge. They are listed as stuck (as `inverted`,
  `tet_repair.hpp:42`).
- **Fallback in 2D**: a device rebuild only (`Fallback::rebuild`). The 2D rebuild is
  estimated several times cheaper per cell than 3D's, and the 3D host path
  (tet_flips, 577 lines) won only for a handful of stuck entries at 2.1M (cleanup2,
  D2). A 2D host fallback is stage 4, if measurements show frequent small stuck
  counts.
- Degenerate input (a square lattice: 4 points on every circle) is decided by the
  perturbation alone, as the 3D lattice is. It is a must-have test, and the first
  step out of it is the 2D analogue of the wave-lattice regression.

### 4.6 Throughput, 2D (estimates, reasoned from 3D)

Per cell, 2D has 2 triangles against ~6.5 tetrahedra, 3 stored faces against ~8, 6
face sides against ~16, and two vertices per face against rings of ~5. Polygon
clipping has ~6 edges against ~16 faces and ~30 vertices.

| | 3D measured (2.1M) | 2D estimate | limited by |
|---|---|---|---|
| rebuild (clipping, plain) | 40 M cells/s (warp) | 150-400 M cells/s | candidate loads and cuts per cell; to be measured with kernels (a) and (b) |
| exact start | 17.7 M cells/s | 60-150 M cells/s | rings + check + few patches |
| moving update, 0.03 spacings | 201 M cells/s | 500-1000 M cells/s | screening pass over 2n triangles (~70 B each), the face scan and geometry: bandwidth |
| update_cells | 326 M cells/s | 600-900 M cells/s | the two atomic passes and the warp sort over 6 entries |

## 5. Hydro in D dimensions

### 5.1 What changes (all compile-time)

| item | now (3D) | in D | file:line |
|---|---|---|---|
| state | `Kokkos::Array<double, 5>` | `D + 2`: (rho, rho v_1..v_D, E). This is LUNA's layout: 1D (3, Nx), 2D (4, Nx, Ny), `ic.py:141` | `euler.hpp:14` |
| vectors | `Vec3` | `Vec<D>` with the same left-to-right `dot` | `euler.hpp:16-22` |
| primitive/conserved | 3 velocity terms | D terms | `euler.hpp:33-48` |
| half step | `for d < 3`, `k < 3` | `< D` | `euler.hpp:58-75` |
| HLL/HLLC | box axes, v_t = v − (v.n) n | unchanged in form. For D = 1, n = ±1 and v_t = 0 | `euler.hpp:127-181` |
| gradients | 4x4 Cholesky, `double*[15]` | (D+1)², `double*[(D+2) D]` | `finite_volume.hpp:82, 335-418` |
| time step radius | `cbrt(3V/4pi)` | 2D `sqrt(V/pi)`, 1D `V/2` (AREPO `get_cell_radius`, `voronoi.c:1053-1071`) | `finite_volume.hpp:147` |
| face angle (regularisation) | `sqrt(A/pi)/(d/2)` | 2D `0.5 L/(d/2)` (AREPO `voronoi_2d.c:1309`), 1D none (AREPO's 1D sets none, `voronoi_1d.c`) | `finite_volume.hpp:283` |
| drift wrap | `mesh_->box()` (cube only) | per-axis `sides()` | `finite_volume.hpp:305-311` |
| face neighbour j < 0 | none | boundary face: a ghost state (5.4) | new |

`FiniteVolumeD<D, Exec, Riemann>` with `using FiniteVolume = FiniteVolumeD<3, ...>`.
D = 3 keeps today's instruction stream under the textual rule of 2.1. The boundary
branch is `if constexpr (Bounded)`, a separate template flag, so periodic 3D carries
no test for j < 0.

### 5.2 Accuracy bar: as good as LUNA at the same resolution and order

- On a **static uniform 1D lattice** our scheme should reproduce LUNA's MUSCL-Hancock
  with LIMITER = 1 (MC). The derivation here is unverified, to be tested:
  - the LSQ gradient with an intercept over the two neighbours is the central
    difference;
  - AREPO's face limiter keeps W_i ± g dx/2 within [min, max] of the neighbours
    (`finite_volume.hpp:362-366, 393-411`). That is g ≤ 2Δ_R/dx and g ≤ 2Δ_L/dx for
    monotone data and g = 0 at extrema, i.e. MC's minmod(2Δ_L, 2Δ_R, central).
  - The half step is LUNA's (`euler.hpp:55-57`) and HLLC is LUNA's (CTest within
    1e-14).

  So a fixed-dt run should equal LUNA's to rounding. This is the strongest oracle we
  can have, and a mismatch is a bug in one of the two codes.
- In 2D on a square lattice, AREPO's limiter scales the whole gradient per variable,
  while LUNA limits per axis. The results differ (comparable, not equal).
- Order 0 (Godunov) against LUNA ORDER = 0 (`muscl_hancock.py:485-494`,
  `first_order_divergence`) on a static lattice: equal to rounding in 1D and 2D if
  LUNA's ORDER = 0 time integration is forward Euler. That is to verify
  (`time_integration/integrators.py`), not assumed.
- Time step: LUNA's tests run at CFL 0.8 with its own dt rule (`template_config.py:32`;
  the McNally bounds "hold for CFL 0.8 only", `test_hydro_hard_mcnally_kh.py:195`).
  Ours is AREPO's rule at 0.3. For matched comparisons, add a fixed-dt or LUNA-rule
  option (5 lines).

### 5.3 1D hydro: a specialised step

The generic face-list path works for D = 1, but the 1D mesh is a sorted order, and
`order`, `left(i)`, `right(i)` give the two neighbours of every cell directly.

A specialised `FiniteVolumeD<1>` step:

1. Kernel 1: gradients (reading 3 cells).
2. Kernel 2: per cell, its two face fluxes and the update. Each face is computed from
   both sides with the same arguments, so both get the same bits: the per-cell loop
   of `finite_volume.hpp:477-493`. There is no face flux array.
3. Mesh: one sort + one kernel.

That is about 4-5 launches per step instead of ~12. This matters only because small
1D runs are launch-bound (section 3). The specialisation is worth its ~150 lines only
if the generic path, measured, is launch-bound at LUNA's N. Decide after stage 1's
first measurement.

### 5.4 Boundaries in the hydro

- **1D**: explicit boundary faces (section 3).
- **2D/3D (not needed by the target tests)**: mirror particles in a doubled periodic
  box. For each walled axis, the caller passes 2n particles in a box of twice that
  side, with mirror m(i) at 2L − x_i. The mirrors are computed from the snapped
  integer, 2K − s, so the pair is exactly symmetric about the wall.
  - The mesh library needs no change: it stays periodic, the particle count is fixed
    (no rebuild for ghosts entering or leaving), and repair works.
  - Faces between a real cell and a mirror are wall faces. Their ghost state is the
    mirror's state with the normal velocity flipped (reflective) or not (outflow), or
    a fixed state (inflow). These are AREPO's `REFLECTIVE_X` = 1 / 2 ghosts
    (`voronoi.c:319-330`, `finite_volume_solver.c:1011-1030`).
  - The hydro updates only i < n.
  - Cost: 2^k times the mesh memory and time for k walled axes, and none for the
    hydro. Rejected alternative: clipping cells at walls in the persistent mesh. That
    needs the hull of a non-periodic Delaunay triangulation with an "infinite vertex",
    and polygons on the walls: a new engine.
- **Inflow on a moving mesh without refinement**: the cell next to an inflow wall
  grows without bound, because its generator moves away with the flow. In Shu-Osher
  the inflow region stays exactly the constant post-shock state (all characteristics
  move right: u − c = 2.63 − 1.94 > 0 with LEFT = (3.857, 2.629, 10.33)), so one large
  uniform cell is exact there. In general, inflow needs refinement or a static layer
  near the wall. **Open**.

## 6. `voronoi::Mesh<D>` and the tests

### 6.1 Interface

`template <int D, class Exec = Default> class MeshD`, with
`template <class Exec = Default> using Mesh = MeshD<3, Exec>`, so `voronoi::Mesh<Exec>`
(8 uses: `hydro_sedov.cpp:217`, `hydro_test.cpp`, `support.hpp:489`,
`finite_volume.hpp:78`) is unchanged.

The specialisations `MeshD<1>`, `MeshD<2>` and `MeshD<3>` (today's class body) share
the public contract (`voronoi_mesh.hpp:100-276`):

- `update(x)`, `rebuild(x)`, `update_cells()`, `scratch(bytes)`;
- `faces()`, `owner()`, `neighbour()`;
- `area()`, `face_centroid()`, `centroid()` as `View<const double*[D]>`;
- `volume()`, `cell_face_start()`, `cell_face()`, `positions()`, `sides()`.

They differ in this:

- `clip(x, cells)` exists for D = 2, 3; D = 1 would rebuild everything (it is
  cheaper).
- `set_fallback` has no effect in 1D. In 2D it allows `rebuild` only, at first.
- `Report` keeps its fields; flips are 2-2 in 2D, and in 1D there are none (the
  "sorted again" flag is enough).

Bounded 1D: `BoxN<1>` gains `bool periodic = true`. The 1D engine honours false.
`MeshD<2>`/`MeshD<3>` throw on it (2D/3D walls are the hydro layer's mirrors, 5.4).

### 6.2 Tests, bench, CTest

- `support.hpp` gets `make_points<D>(kind, n, box)`: jittered, uniform, lattice, wave,
  clustered, and in 2D sc/hex lattices and shells; plus brute-force oracles per D:
  - **1D**: the sorted order by (snapped key, id), midpoints, volumes summing to L
    exactly in grid steps;
  - **2D**: every triangle's circumcircle empty (exact, perturbed) against every
    particle with x within the radius (the 3D start oracle's sweep,
    `start_test.cpp`); every edge twice with apexes on opposite sides; areas summing
    to the box; and the clipping's `check_cells` in 2D (face centroids on bisectors,
    no closer generator).

  O(n²) stays in tests only.
- CTest:
  - 1D: `exact_predicates` covers the 1D sort ties, and a new `mesh1d` covers
    construction, the moving update against a rebuild after every step, walls and
    the contract;
  - 2D: `exact_predicates_2d`, `clipping_2d`, `start_2d` (layouts × boxes 1, 2, 3,
    slab, 1:2), `repair_2d` (jittered, square lattice, wave; 0.03 and 0.3 spacings),
    and `voronoi_mesh` parametrised on D;
  - `hydro`: parametrised on D for the existing invariants (uniform state, moving
    bulk, conservation, bit reproducibility, Galilean blast), plus the LUNA
    static-lattice oracles (5.2) and 1D exact Riemann (Toro 1 = Sod, 3, 4; a ~100-line
    exact solver in test support).
- Bench:
  - `voronoi_bench --dim=1|2|3` (default 3: the 3D command lines and output are
    unchanged);
  - new drivers `hydro_1d` (Sod, Shu-Osher, WC, LUNA ICs) and `hydro_kh2d` (McNally),
    writing LUNA-layout HDF5 with `spatial_dimension` = D (`snapshot.hpp:92` hard-codes
    3).

### 6.3 Keeping 3D bit-identical and as fast

The gate for stage 0 and for every later merge is one job interleaving main and the
branch at 2.1M, as in the box3 and shrink notes:

- warp rebuild × 6 inputs;
- exact start;
- steps at 0.03 / 0.1 / 0.3;
- `update_cells`;
- peak memory;
- static and moving hydro step;
- `cells_checksum` and the hydro checksum.

Pass: identical bits and memory, and every timing within 1 %. The 3D engine files
(`shell_voronoi*.hpp`, `tet_*.hpp`) are not edited beyond `Box` → `BoxN<3>` spellings,
which are type aliases (no change at all).

## 7. The test plan against LUNA

The setups are exactly LUNA's:

| test | LUNA setup (file:line) | N ladder, t | boundary in LUNA | reference | ours |
|---|---|---|---|---|---|
| Sod (Toro 1) | (1, 0, 1) \| (0.125, 0, 0.1), x₀ = 0.5, γ = 1.4, Lx = 1 (`exact_riemann.py:8-9`; `sod.py:7-33` puts the jump at Lx/2) | 128, 256, 512; t = 0.25 (`test_hydro_accuracy_shocktube.py:9, 98-99`) | 1 zero-gradient; the waves stay inside | exact cell averages (`exact_riemann.cell_average`); L1/L2 of rho, u, p; front positions; budget `check_tube_budget` (`tests/_helpers.py:344-357`) | 1D walls (zero-gradient). The exact solution needs no LUNA run. Also the periodic doubled tube as a cross-check |
| Shu-Osher | LEFT = (3.857143, 2.629369, 10.33333) for x < −4, (1 + 0.2 sin 5x, 0, 1) beyond, on [−5, 5], γ = 1.4, exact cell averages (`shu_osher.py:34-85`) | 256-2048; t = 1.8 (`test_hydro_hard_shu_osher.py:29-30`) | 1 zero-gradient (supersonic inflow at the left) | AthenaK PPMX 16384 `athenak/shu_osher/so_ppmx_rk4_16384.npz` (external gate); LUNA MC 65536 `luna/shu_osher/shu_osher_mc.npz` (regression); L1/L2 of rho against the reference averaged onto N cells | 1D walls: zero-gradient (left inflow, right outflow). A periodic box does not work: the seam sends a wave at u + c ≈ 4.57 that reaches x ≈ 3.2 by t = 1.8 |
| Woodward-Colella | rho = 1, u = 0, p = 1000 \| 0.01 \| 100 split at 0.1, 0.9, γ = 1.4 (`woodward_colella.py:86-112`) | 256-4096; t = 0.038 (`test_hydro_hard_woodward_colella.py:26-27`) | 2 reflective | AthenaK PPMX 16384 `athenak/woodward_colella/a9_ppmx_rk4_16384.npz`; LUNA MC 65536 `luna/woodward_colella/woodward_colella_mc.npz`; L1/L2 of rho | 1D reflective walls; or exactly the mirror-doubled periodic box (no code) |
| McNally KH | ρ 1/2/1, V_x 0.5/−0.5, P₀ = 2.5, L = 0.025, V_y = 0.01 sin 4πx, γ = 5/3, unit box, point values at cell centres (`kh_mcnally.py:122-171`) | 64-2048; M(t) at t = 0.02k to 1.5; fields at 1.5 on 256² (`test_hydro_hard_mcnally_kh.py:28-30, 187-230`) | 0 periodic | Pencil 4096² M(t), KE(t) `pencil/mcnally_kh/*.txt`; AthenaK PPMX 4096² field `athenak/mcnally_kh/a5_ppmx_rk4_4096.npz`; LUNA MC 4096² (regression) | native periodic 2D. IC on a square lattice (= LUNA's cell centres) and on jittered points (the real test of the mesh) |
| Sedov 3D | as now | 128³, 256³ | periodic | analytic, LUNA | runs today |

Notes for the analysis scripts (Python, importing LUNA's own metric code where
possible):

- 1D L1 against a fine reference: (i) LUNA's metric, our cells deposited
  conservatively onto LUNA's N uniform cells; (ii) the exact ∫\|ρ − ρ_ref\| dx of the
  two piecewise-constant fields (sorted breakpoints).
- KH M(t) (`mode_amplitude`, `test_hydro_hard_mcnally_kh.py:52-61`) is written for
  cell centres on a grid. For Voronoi cells, use volume-weighted sums at the cell
  centroids. Check against McNally's eqs. 6-9 before the numbers are compared.
  Fields at t = 1.5 by conservative deposit onto 256².
- Limiters: LUNA's ladders are minmod/MC/van Leer. Ours is AREPO's face limiter (= MC
  in 1D uniform, 5.2). Compare against LUNA MC.

## 8. Stages, effort, acceptance

Line counts are estimates of library + tests/drivers. "Agent-days" are working days
of one agent including A100 jobs.

| stage | content | lines | agent-days | acceptance |
|---|---|---|---|---|
| 0 | `BoxN<D>`, `OffsetN`, `Points<D>`, `CellList<D>` (D = 2, 3), `Geometry`/`CellFaces`/`cells` in D, `Vec<D>`, `State<D+2>`, `FiniteVolumeD`; aliases for 3D | +250 / −80 | 1-2 | 3D: ctest 7/7 (A100, CPU), checksums identical, the interleaved 1 % job (6.3), clang-tidy clean |
| 1 | `MeshD<1>` (sort, faces, cells, walls), hydro D = 1 incl. boundary faces, `hydro_1d` driver, comparison script | lib ~350, tests+drivers ~600 | 2-3 | `mesh1d` oracle; hydro invariants in 1D; static uniform lattice = LUNA MC (fixed dt) and ORDER 0 to ~1e-13; Sod L1 ≤ LUNA's bounds at 128-512 (`test_hydro_accuracy_shocktube.py:13-21`); Shu-Osher and WC L1 against AthenaK ≤ LUNA MC's at the same N (moving and static mesh); throughput table at N = 4096 and 2M |
| 2 | 2D predicates, polygon clipping (rounded + exact), `from_rings2`, start, triangle mesh, faces, repair, `MeshD<2>` | lib ~1,500, tests ~700 | 4-6 | the 2D oracles of 6.2 on 13 layouts × boxes; repair = rebuild after every step (jittered, uniform, clustered, square lattice, wave; slab, 1:2 box); bench rebuild/start/step/update_cells/memory table at 2.1M and 16.8M |
| 3 | hydro D = 2, `hydro_kh2d`, KH analysis | lib ~100, drivers+scripts ~450 | 1-2 | 2D hydro invariants; static square lattice ORDER 0 = LUNA's; McNally M(t) and fields within LUNA MC's bounds at the same N (`test_hydro_hard_mcnally_kh.py:196-216`); accuracy per GPU-second against LUNA 2D |
| 4 (optional) | 8-lane 2D clipping kernel; 2D host fallback; mirror walls in 2D/3D; a fused 1D step | 300-600 | 2-3 | each only if a measurement asks for it |

Total stages 0-3: ~3,000-3,500 lines, 8-13 agent-days.

Risks:

1. FMA contraction or reordering makes 3D bits change after templating (stage 0
   catches it; fix by keeping the expressions textually).
2. Small 1D runs are launch-bound: LUNA's own times at N ≤ 4096 are probably also
   overhead-bound, so compare GPU time per run, not peak rates.
3. 2D repair from degenerate starts (square lattice) and inverted triangles at large
   steps: no data yet. The device rebuild is the fallback.
4. No refinement, so inflow boundaries grow cells (5.4).
5. KH and L1 metrics on unstructured cells need definitions agreed before the numbers
   are compared.
6. The 2D clipping kernel shape (thread vs 8-lane team) decides the 2D rebuild rate;
   the estimate range is wide.
7. Our time-step rule and limiter differ from LUNA's ladders: matched-method options
   are needed for exact same-order comparisons.

## 9. Expected throughput per D on the A100 (all estimates, labelled)

Measured 3D anchors (README; hydro_stage3): rebuild 40 M cells/s, moving update 201
M cells/s (0.03 spacings), update_cells 326 M cells/s, hydro moving 55, static
per-face 112, first-order per-cell 288 M cell updates/s, 2221-2418 B/cell.

| | 1D (n ≥ 2M) | 1D (N = 4096) | 2D (2.1M) | 3D (measured) |
|---|---|---|---|---|
| rebuild | ~1-3 G cells/s (radix sort, bandwidth) | launch-bound | ~150-400 M cells/s | 40 |
| moving update + cell lists | ~1-3 G cells/s | launch-bound | ~300-600 M cells/s | 124 (update + update_cells) |
| hydro static, MUSCL-Hancock HLLC | ~1-2 G updates/s (1 stored face per cell, 3 variables) | ~50 M/s (≈ 12 launches × ~5 µs per step) | ~300-400 M updates/s (3 vs ~8 stored faces per cell, 4 vs 5 variables, 8 vs 15 gradients) | 112 |
| hydro moving (all phases) | ~0.5-1 G updates/s | ~30-50 M/s | ~120-200 M updates/s | 55 |
| memory | ~150-250 B/cell | | ~550-650 B/cell with hydro | 2370-2418 |

Reasoning: the hydro phases scale with face sides per cell × variables (bandwidth
and FLOPs alike). The mesh update scales with simplices per cell plus face work, and
2D removes the ring pools. The 1D small-N column is 1/(launch count × launch latency),
independent of the kernels. Every entry is to be replaced by a measurement in its
stage.

## 10. Open decisions for the owner

1. Naming: `MeshD<D, Exec>` + `Mesh<Exec> = MeshD<3, Exec>` (no caller changes), or
   a breaking `Mesh<D, Exec>`.
2. 1D: the specialised 2-kernel hydro step (5.3) now, or only if the generic path
   measures launch-bound at LUNA's N.
3. 1D moving update: always sort, or check-then-sort.
4. 2D clipping kernel: thread per cell first (recommended), 8-lane teams only if
   needed.
5. 2D fallback: device rebuild only (recommended) vs a host port of tet_flips.
6. Walls: 1D explicit faces now; 2D/3D via mirror doubling (2^k mesh cost) when a test
   needs them, or not at all for now.
7. Inflow on the moving mesh: accept the growing inflow cell (exact for Shu-Osher), or
   hold generators static near inflow walls.
8. Matched-method comparisons: add a fixed-dt / LUNA-CFL option and, if wanted,
   LUNA's per-axis limiters on lattices, or compare only "our scheme vs LUNA MC".

## 11. Refinement and derefinement (inserting and removing generators), D = 1, 2, 3

Added at the owner's request: it touches the persistent mesh as much as the
dimension does, so both are designed together. **The choice of insertion and removal
algorithm per D is pending the literature report** (`reports/GPU Delaunay insertion and
removal.md`, being written separately). Below are the constraints from our code that
any choice must meet, a default per D that already meets them, and the parts that do
not depend on the choice: criteria, remap, ids, memory, interface and tests. No web
search was done for this section. Papers named here are cited from memory, and the
report should confirm them.

### 11.1 Criteria (a)

From AREPO next generation (`origin/Arepo2-nextgeneration-SoA:src/voronoi/voronoi_criteria_refine_and_derefine.cc`,
read with `git show`):

- **Split** when `can_this_cell_be_split` and (volume criterion or mass criterion).
  `can_this_cell_be_split` requires a round cell:
  `MaxFaceAngle < 1.5 CellMaxAngleFactor` with `REGULARIZE_MESH_FACE_ANGLE`, else
  centroid offset < 2 CellShapingFactor · radius (`:101-116`). The mass criterion is
  `m > 2 TargetGasMass` (`:128-133`). The volume criteria are bounds on MinVolume and
  MaxVolume (`:136-150`). Order and prevention flags: `:81-99`.
- **Merge** when `m < 0.5 TargetGasMass` (`:121-126`) or `V < 0.5 MinVolume`
  (`:144-149`), unless `V > 0.5 MaxVolume` (`:137-142`) (`:64-79`).

Here, criteria are the hydro's: one per-cell flag (split / merge / none).

- Throughput rule: the flag is computed inside the existing time-step kernel
  (`finite_volume.hpp:142-162`), which already has W, V and the face loop (for the
  face angle).
- The number of flagged cells comes back with dt as a second reduction value, so no
  extra launch and no extra host synchronisation when nothing refines. The
  zero-refinement step is then the step of today, up to one extra integer in a
  reduction (within the 1 % rule, to be measured).
- Mass target as AREPO for the moving mesh. Optionally LUNA's AMR criterion (relative
  ∇P, or ∇ρ gated by overdensity, `LUNA/docs/AMR.md`, Sedov paragraph) for a
  like-for-like comparison against LUNA AMR.

### 11.2 Where the new point goes

AREPO-ng puts it at the cell's farthest Voronoi vertex,
`np->IntPos = p->IntPos + MaxVertexPosition`
(`origin/Arepo2-nextgeneration:src/voronoi/voronoi_execute_refinements.cc:775-783`).

A Voronoi vertex is the circumcentre of a Delaunay simplex, so the new point lies
**exactly on** the circumsphere of the D + 1 generators of that simplex: an exact tie.
The symbolic perturbation decides it consistently, but the new faces start with zero
area.

Default here: the point 1/2 of the way from the generator to its farthest vertex,
snapped to the grid. It lies strictly inside the parent's cell in exact arithmetic,
while snapping may put it on a bisector, which the perturbation decides. **Open:**
the fraction (AREPO-ng uses 1).

### 11.3 Insertion into the persistent mesh (b)

**1D**:

- Insert = a splice in the sorted order.
- A batch of k inserts and removals is one stable radix sort of n' keys, the same
  kernel as `update()` (section 3). Exact, no conflicts, no independence needed.
- Rate (estimate): the sort's, ~1 G points/s at n ≥ 2M, so a batch costs the same as
  a 1D update.

**2D and 3D (default, pending the report)**: point location, a split, then the
existing flip repair.

1. **Locate**: walk from a simplex of the parent (its generator is a vertex of the
   simplex whose circumcentre is the target vertex: `Rmax` simplex) towards the new
   point, using exact orient tests (a visibility walk). The point is inside or next
   to the parent's star, so the walk is a few steps.
2. **Split**: 1-3 (2D) or 1-4 (3D) of the containing simplex, or a 2-4 / 2-6 edge or
   face split when an orient test is 0 (perturbation: the new id decides on which
   side). New simplices take free slots (3D: `tet_repair` free list and `reserve`,
   `tet_repair.hpp:309-323`; 2D: a slot pool of 2 per insert).
3. **Repair**: mark the new simplices dirty and run the existing device repair
   (`tet_repair::repair`, `tet_repair.hpp:446`; the 2D 2-2 repair of 4.5).
   - For a single insertion into a Delaunay mesh, flipping always succeeds in 2D
     (Lawson) and in 3D (Joe 1991).
   - For many concurrent insertions, the flip graph can get stuck in 3D. GPU work on
     this (gDel3D: Cao, Nanjappa, Tan et al., about 2014) reports stuck flips and
     repairs them by star splaying (**to confirm in the report**). Ours lists them as
     stuck and finishes with the existing fallback (`tet_flips::finish`, or a
     rebuild).
4. **Independence (conflict-free batches)**: two inserts conflict if their splits
   touch the same simplex or their remaps (11.5) share a natural neighbour.
   - Rule: the parents must be at Delaunay distance ≥ 3 (no shared natural neighbour
     after insertion).
   - Selected by the hashed-index vote of `tet_repair` (`priority`,
     `tet_repair.hpp:279-289`) on the parents' 2-rings: the losers wait one step.
   - At 1 % of cells per step, almost all candidates win at once (estimate).
   - The flips themselves need no independence: the vote/lock scheme already
     serialises conflicting flips round by round.
5. **Alternative**: cavity (Bowyer-Watson) insertion. Delete the simplices whose
   circumsphere holds the point (exact perturbed in-sphere) and star the cavity
   boundary from the point. One kernel per insert, and it touches each cavity simplex
   once (3D cavities ~20-30 tetrahedra, estimate). The cavities must be disjoint,
   locked by the same vote. Exact by construction: a cavity is star-shaped from the
   point under the same perturbed predicate.

   This is likely fewer memory round trips than split + flip rounds. Which one is
   faster on the A100 is the report's question; both reuse our predicates and votes.

### 11.4 Removal (c)

- **1D**: drop from the sorted order (the same sort).
- **2D**: the link of the removed point is a polygon of ~6 vertices (estimate: the
  mean degree). Re-triangulate it by the Delaunay ear rule: an ear is kept if no other
  link vertex is in its circle (Devillers 2002, O(k log k); with k ~ 6 an O(k²) ear
  scan in registers is cheaper). Exact with the perturbed incircle. One thread per
  removal. The 2D link is always triangulable, and the result is the Delaunay
  triangulation of the hole.
- **3D**: removal by flips alone is not always possible: some stars cannot be
  flipped down to degree 4. The general exact method re-tetrahedralises the star:
  the hole is filled by those Delaunay tetrahedra of the link vertices that lie inside
  it. Our host code already does this kind of local rebuild:
  - `tet_flips::detail::delaunay`, Bowyer-Watson of a few points in a local frame
    with exact offsets and perturbation, `tet_flips.hpp:105-205`;
  - `rebuild`, accepted only if it matches the old boundary face for face,
    `tet_flips.hpp:246-306`.

  Default here: a device port of `delaunay` for the link of a removed point (~15
  vertices, ~27 tetrahedra in the star: estimates from the ~16 faces per cell), one
  thread (or warp) per removal, fixed-size local arrays. Accept if the boundary
  matches; else list the removal as stuck and retry it in the next batch.
- **Independence**: removals must not share a natural neighbour (their remap, 11.5),
  and must not be adjacent to an insertion in the same batch. Same vote.

### 11.5 Conservative remap (d), exact and without polyhedron intersection

AREPO-ng:

- Refinement intersects the new cell with each old neighbour's cell, clipping with the
  old neighbour's planes (`voronoi_execute_refinements.cc:415-523` per
  `research_notes/arepo2_mesh_survey.md:170-180`, including a volume bug fixed on a
  fork).
- Derefinement builds each neighbour q's cell twice from p's neighbour set, and q gains
  `(VolAfter − VolBefore) / Vol_p` of p (`voronoi_execute_derefinements.cc:233-340`
  per the survey `:182-189`).

Here, both cases come from the mesh's own volumes before and after, with no extra
cell builds:

- **Insert x**: every natural neighbour q of x loses exactly the region that x's new
  cell takes from it: Voronoi cells only shrink when a generator is added, and what
  they lose goes to x. With V_q before and V'_q after (positions of all old points
  unchanged), x receives `(V_q − V'_q) / V_q` of q's mass, momentum and energy, and q
  keeps the rest. The fractions are Sibson's natural-neighbour coordinates times
  V_x / V_q.
- **Remove p**: each natural neighbour q gains `(V'_q − V_q) / V_p` of p's
  conserved quantities (AREPO-ng's rule, from volumes we already have).

Conditions and properties:

- Exact in exact arithmetic when no two changes share a natural neighbour (the
  independence rule of 11.3/11.4).
- Conservation holds to rounding: every transfer is one value subtracted from one
  cell and added to another.
- The cost is one pass over the changed cells' face lists.
- It requires that refinement runs at fixed positions: after `update(x)` and
  `update_cells()` of the step (V known), insert/remove, geometry and volumes again
  (V'), then transfer. The volumes before are kept in a scratch copy (8 B/cell).
- First-order (piecewise-constant) transfer, as AREPO-ng. A second-order transfer
  with the gradients at the region centroids is an option, not in stage R1.

### 11.6 Incremental faces, rings and update_cells (e)

- **3D**: insertion and removal produce touched tetrahedra, exactly like flips, and
  enter `state_.touched_list`. `update_face_rings` then walks only faces on touched
  tetrahedra and appends their rings (`tet_mesh.hpp:492-630`). Circumcentres only for
  created tetrahedra (`voronoi_mesh.hpp:309-315`). Nothing new.
- **2D**: the face scan and two-centre geometry (4.3) are already "dirty only".
- **1D**: recomputed in full (cheap).
- `update_cells()` stays a full pass (6.4 ms at 2.1M in 3D, box3 table). An
  incremental version is the open item of cleanup2 ("Open") and would serve
  refinement and steps alike.

### 11.7 Ids, interface and memory (f, g)

**Ids (D3: particles are the caller's indices, and ties are broken by them)**:

- Inserted points get new ids. Removed ids leave holes.
- Renaming (moving id n−1 into a hole) changes the perturbation order, so a mesh that
  was Delaunay under the old ids may fail an exact tie under the new ones. Every
  renamed particle's star must be marked dirty and repaired, or ids must not be
  renamed.

Default: stable ids with holes.

- `alive(i)` per particle, a device free list of ids, and new points take free ids
  first, else n, n+1, ...
- The hydro skips dead ids: one predictable branch per cell, and dead cells have no
  faces, so face loops are untouched.
- A full rebuild (fallback, or on request) compacts ids and returns the permutation
  `new_id(old)`. The hydro applies it to Q (one gather).

**Interface (sketch; D = 1, 2, 3)**:

```cpp
struct Changes { View<const int*> inserted, removed, deferred; };   // ids
// Positions of the cell's new points (one per split parent) and the ids to merge away.
Changes refine(const View<const double*[D]>& new_points, const View<const int*>& parents,
               const View<const int*>& merge);  // after update(x) and update_cells()
View<const double*> volume_before() const;     // V for the remap (11.5)
View<const char*>   alive() const;
```

- `refine` does locate/split/repair (or cavity insertion), removal, geometry and
  `update_cells`, and returns which requests were deferred by the independence vote.
- The hydro does the transfer with `volume_before()`, `volume()` and the face lists.
- The `update(x)` path is unchanged and never touches the refinement state, so 3D
  steps without refinement are today's code (gate: 6.3).

**Memory growth policy (f)**:

- Per-particle arrays get capacity n + n/8, as tetrahedra and faces have today
  (`voronoi_mesh.hpp:175-183`), grown by 1/4 when exceeded (realloc + copy; the rings
  already grow this way, `tet_mesh.hpp:497-520`).
- Simplices: 3D ~6.5 tetrahedra per point (free list, growth); 2D exactly 2 per point
  (+2 per insert, −2 per removal, so the pool is sized from n's capacity); 1D none.
- Ring pools (3D) take appended rings as in a step and compact as today.
- Peak in 3D is 2221 B/cell (README) plus ~13 B/cell of capacity slack per 1 % of
  growth (estimate). The 256³ moving run had 0.25 GB to spare (hydro_stage3 note), so
  refinement at 256³ needs n below about 16.5M, or the face-list memory saved first.

### 11.8 Cost (h), estimates pending the report

| | per change | at 1 % of 2.1M cells per step (21k inserts + some removals) | limited by |
|---|---|---|---|
| 1D | one sort for all | ~= one 1D update (<< 1 ms at 2.1M) | the radix sort |
| 2D insert (split + 2-2 flips; ~3-6 flips each, estimate) | ~0.05-0.1 µs amortised | ~1-3 ms | repair rounds (launch + scan per round) |
| 2D remove (ear rule, k ~ 6) | thread-local | < 1 ms | predicate count |
| 3D insert (split + flips) | ~10-20 flips each (estimate). The measured repair does 61k flips in a 10.3 ms step at 0.03 spacings, including screening every tetrahedron (box3 note), an upper bound of ~0.17 µs per flip | ~20-60 ms extra, i.e. ~0.5-1 M inserts per 2.1M step, ~10-40 M inserts/s | flip rounds, ring appends |
| 3D remove (device local Bowyer-Watson, ~15 points) | ~5-20 µs per thread (estimate), in parallel | ~2-10 ms | per-thread predicate work, register/local memory |
| remap | one pass over the changed cells' faces | < 0.5 ms | gathers |
| criteria | inside the dt kernel | ~0 | |

**1 % per step is a high rate.** In 3D it would add 2-6x a 0.03-spacing step
(estimate), so the rate in real runs (Sedov with a mass target) is the first thing to
measure. When nothing is flagged, the cost is zero by construction (11.1, 11.7).

### 11.9 Stages and acceptance (after the dimension stages they depend on)

| stage | content | lines | agent-days | acceptance |
|---|---|---|---|---|
| R0 | criteria in the dt kernel, `refine()` skeleton, ids with holes, `volume_before`, remap | ~250 | 1 | 3D gate 6.3 unchanged with refinement off; conservation to rounding (long double sums ≤ 1e-14) over random insert/remove sequences on a static mesh |
| R1 | 1D refine/merge (sort) | ~100 | 0.5 | sorted-order oracle after every batch; Sod/Shu-Osher with a mass target against the exact solution / AthenaK, same L1 at fewer cells than uniform |
| R2 | 3D insertion (split + device flips, independence vote) or the report's choice | ~400 | 2-3 | after every batch the mesh equals an exact rebuild (`support::compare_with_rebuild`: no failing triangle, same Delaunay neighbours, volumes within 1e-10) on jittered, uniform, lattice and wave (inserts at exact ties); remap fractions equal brute-force overlap volumes (clipping each new cell against the old cells, tests only) to 1e-12 |
| R3 | 3D removal (device local Bowyer-Watson; the host `tet_flips` as fallback) | ~400 | 2-3 | as R2, with removals and mixed batches; stuck removals retried and counted |
| R4 | 2D insertion/removal | ~300 | 1-2 | as R2 in 2D, after stage 2 |
| R5 | refining Sedov (3D) and KH (2D) | ~200 drivers | 1-2 | Sedov with a mass target against our uniform 128³/256³ runs and against LUNA block-AMR Sedov (`scripts/validation/runs/sedov3d_amr_run.py`; `docs/AMR.md`: 64³ base → 512³ effective, "~400 s on a single A100" in its physical-unit variant): L1 rho, shock radius, accuracy per GPU-second; conservation to rounding over the run |

Open decisions for the owner (refinement):

- (R-a) the insertion algorithm per D (split + flips vs cavity insertion, and star
  splaying for stuck 3D flips): pending the literature report;
- (R-b) the new point's position (AREPO-ng: at the farthest vertex; here halfway);
- (R-c) stable ids with holes (recommended) vs compaction every step;
- (R-d) the remap's independence rule (no shared natural neighbour) vs exact
  overlaps for neighbouring changes (polyhedron intersection, AREPO-ng's way);
- (R-e) the refinement cadence: every step, or every k steps / at synchronisation
  points with local time steps.

## Owner's decisions (2026-11-02)

1. Naming: one breaking `Mesh<D, Exec>` now (the hydro is the only user).
2. 1D: a fused hydro step from the start (launch-bound at LUNA's 1D sizes).
3. 1D moving update: check order first, sort only when points cross.
4. 2D clipping: one thread per cell first; measure teams later.
5. 2D fallback: device rebuild only.
6. Walls in 2D/3D: none for now (1D gets explicit wall faces).
7. Inflow on a moving mesh without refinement: accept the growing inflow cell.
8. LUNA comparisons: add a fixed-dt / LUNA-CFL option; keep AREPO-ng's limiter.
9. Refinement details: after the literature report (reports/GPU Delaunay insertion and removal.md).
10. Non-cubic boxes in every dimension: `BoxN<D>` carries one side per axis (as box3 does in 3D), with the same isotropic snapping step.
