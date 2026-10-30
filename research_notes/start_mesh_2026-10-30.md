# Exact starting mesh, 2026-10-30 (branch from main 3074c1e)

Goal: the starting tetrahedralization is exactly the Delaunay
tetrahedralization defined by the exact predicates (positions snapped to
2^30 steps, ties broken by symbolic perturbation by particle index), for
any input, independent of how well the double-precision clipping does;
steps from such starts stay exact.

## Failure inventory before (main)

`voronoi_bench step 4096 <input> 0.03 2 --verify`, OpenMP, main code:

| input | from_rings unresolved | steps |
|---|---|---|
| jittered, clustered, bcc@0, planes@0 | 0 | exact |
| uniform | 0 (but inconsistent) | volume error 1.6e-3, 4.1e-3: FAIL |
| lattice (= sc@0) | 49646, 4180 instead of 24576 tetrahedra | total volume 0.004: FAIL |
| wave | 45 | 84 failing triangles, volume error 0.34: FAIL |
| sc@1e-6 | 585 | 864 failing, host fallback 16 s per step: FAIL |
| fcc@1e-6 | 434 | 114 failing, volume error 0.57: FAIL |
| shells@1e-9 | - | timed out (> 300 s) |
| pairs@1e-6 | 0 | volume error 0.039: FAIL |

At 2.1M cells on the A100, main's `from_rings` on *jittered* input leaves 4
ring slots unresolved (the clipping's rings disagree in a few places).
Causes: rounding decides the clipping's near-ties (a vertex is cut only
beyond a tolerance, rounding-level faces are dropped), so neighbouring
cells pick different triangulations of cospherical groups; `from_rings`
trusts the rings and leaves holes, overlaps and wrong adjacency.
The failcheck job output named in the task was not on disk (only its
script).

## Design

`tet_start.hpp`, `tet_start::build(exec, grid, cells_with_rings, mesh)`:

1. `from_rings` as before (the guess).
2. `check`, one thread per tetrahedron, exact: not flat; across every
   face a tetrahedron that shares the face and points back; the two
   apexes on opposite sides (orient); the far vertex outside the
   circumsphere (perturbed in-sphere); every particle used; volumes sum
   to the box (a double sum suffices: a closed, consistently oriented
   complex covers the box an integer number of times). Together this is a
   certificate: a locally Delaunay triangulation of the box is the
   Delaunay triangulation, unique under the perturbation. Orientation and
   in-sphere run once per triangle. Particles of bad tetrahedra are marked.
3. Patch: grow the marks by one ring of tetrahedra and rebuild those cells
   by **exact clipping** (`detail::exact_cells`): the clipping kernel's
   walk and hole-closing, but every vertex is named by its three planes'
   particles and stores the exact integer coefficients w, o of the
   in-sphere value e.w + |e|^2 o (computed in 128-bit, rounded once), so a
   cut test costs ~12 flops with a rigorous 16 eps filter; uncertain cases
   go to `exact_predicates::in_sphere` with the particle ids. The
   perturbed predicate is a genuine power diagram, so every cell is a
   convex polytope and neighbouring cells agree by construction. The
   reach (2R) is a rigorous bound from w, o, widened by 8 grid steps for
   snapping; box planes are six virtual particles. Then `from_rings` and
   `check` again. If more than n/8 cells are marked (lattices), or the
   second check fails, every cell is rebuilt exactly (third check).
4. The result is the mesh `tet_repair::prepare` starts from.

Why not fix `from_rings` alone: on a lattice every Voronoi vertex is
degenerate and the rings carry no consistent information; deciding them
needs the cell's candidates, i.e. clipping again. Why not host Bowyer-
Watson for the holes: a lattice is all hole (2.1M cells), and the host
path took seconds per thousand cells.

Also changed, found on the way:

- `tet_flips::rebuild` (host fallback) widened its seeds to the whole ring
  after a failed fit and gave up after 3 tries (5 -> 42 -> 237 seeds);
  on shells it left unrepaired triangles. It now adds only the particles
  of the faces that did not fit, up to 8 tries: no unrepaired triangle in
  repeated shells/sc/wave/fcc runs.
- Inputs on the snapping grid (`support::to_grid`, in `make_points` and
  `Flow::advect`). The topology is decided on snapped positions, the
  geometry from the positions: off the grid, steps from a lattice showed
  volume differences of 1e-9 to 2e-8 against clipping (pure snapping
  inconsistency: on the grid 1e-14). The solver must keep positions on
  the grid too.
- `voronoi_bench step --verify` accepts mesh faces the clipping lacks if
  their area is below 1e-9 spacings^2 (clipping drops rounding-level
  faces), requires no missing clipping face.
- Test bug fixed: the repair test now takes n from the layout (shells has
  4116 points; a fixed 4096 overflowed the neighbour sets).

## Verification

Tests (CPU ctest 5/5, 75 s on one core; CUDA 5/5 on the A100):

- new `start`: 13 inputs at ~4096 (jittered, uniform, clustered, lattice,
  wave, sc@1e-6, bcc@0, fcc@0, fcc@1e-6, shells@0, shells@1e-9, planes@0,
  pairs@1e-6). Independent host oracle, brute force: every triangle in
  exactly two tetrahedra with apexes on opposite sides and matching
  adjacency, no flat tetrahedron, all particles used, exact 128-bit volume
  sum equal to the box, and every circumsphere empty of all particles
  (exact in-sphere; a double prefilter on circumcentres from exact
  integers). Negative control: the oracle rejects `from_rings` alone on
  the lattice. Jittered needs no patch.
- `repair`: also lattice and wave starts at 0.03 and 0.3 spacings (4
  steps) and shells@1e-9 at 0.03 (2 steps): every step equals an exact
  clipping rebuild (neighbours, faces, volumes to 1e-10, total volume).

A100 40 GB, gpu1 jobs 31685559 / 31685562 (binaries v3 = HEAD before the
notes), double. Start at 2,097,152 cells (`voronoi_bench step`, medians of
3; clip = thread clipping with rings):

| input | clip ms | start ms | first check bad tets | cells patched | rounds | exact |
|---|---|---|---|---|---|---|
| jittered (main: clip + from_rings = 340.5 + 52.6) | 340.5 | 462.6 | 2 | 42 | 2 | yes |
| uniform | 883.6 | 1029.8 | 10 | 142 | 2 | yes |
| clustered | 4220 | 4432 | 4 | 37 | 2 | yes |
| lattice | 20.8 | 285.1 | 2,097,908 | all | 2 | yes |
| wave | 274.4 | 872.8 | 35,229 | all | 2 | yes |
| sc@1e-6 | 274.4 | 866.9 | 394,795 | all | 2 | yes |
| shells@1e-9 | 910 | 4587 | 7,223,451 | all | 2 | yes |

Cost on jittered: +70 ms over main's clip + from_rings, 20.5 % of the
clipping (18 % of main's start). Kernel profile (nsys, per start): check
6.6 ms (x2), exact clipping of 42 cells 2.4 ms (one slow thread per
cell), the second `from_rings` ~45 ms (emit 19 ms, adjacency 17 ms), grow
and snap < 0.5 ms. With nothing to patch (262k jittered) the extra is the
check alone: 57.0 vs 46.9 + 8.4 ms. Exact clipping of every cell: ~230 ms
for 2.1M lattice cells, ~500 ms for wave/sc@1e-6, ~3.6 s for shells
(cospherical groups of 12 fall back to the exact in-sphere often).

Steps at 262,144 cells, every step checked against an exact rebuild
(`--verify`: no failing triangle, no missing face, extra faces below 1e-9
spacings^2, volumes within 1e-10):

| input, spacings/step, steps | result | max volume difference |
|---|---|---|
| lattice 0.03 x5, 0.3 x5 | PASS all | 1.1e-14 |
| wave 0.03 x5, 0.3 x5 | PASS all | 1.2e-14 |
| jittered, uniform, sc@0, sc@1e-6, bcc@0, fcc@0, fcc@1e-6 0.03 x3 | PASS all | 1.9e-14 |
| shells@0, shells@1e-9 0.03 x3 | PASS all (5-29 extra faces of area <= 3e-14) | 1.2e-13 |
| planes@0 0.03 x3 | PASS all | 5.6e-14 |
| clustered 0.03 x2 | topology exact; one face short, volume 7e-8: FAIL | 2.9e-7 |
| pairs@1e-6 0.03 x3 | start not exact (see below): FAIL | 1 |

Main for comparison at 4096 (above): lattice, wave, sc@1e-6, fcc@1e-6,
uniform, pairs failed; shells timed out.

## What is left

- **Cells with more than 32 owned faces** cannot be represented in
  `shell_voronoi::Cells` (`max_owned_faces`); pairs 1e-6 spacings apart
  (16 grid steps) produce one such cell in 262k, the start stays inexact
  and `Stats::exact` is false. Fix: carry the neighbour inside the ring
  rows (ring format `k, neighbour, ring...`) so the exact kernel is not
  bound by the 32-column arrays, or a CSR face list for patched cells.
- **First step out of a degenerate start** at 2.1M cells: the flow
  rearranges most tetrahedra (lattice 10.7M flips), the device repair
  leaves 1.9e3 (lattice), 8e5 (wave), 1.1e6 (sc@1e-6), 1.9e5 (shells)
  triangles stuck, and the host fallback takes 7 s, 29 s, 30 s, 79 s.
  Exact, but slow; later steps cost the same as jittered (host round trip
  ~6.8 s when even a few triangles stick). A device rebuild guard
  (clipping + `tet_start::build`, < 1 s) when the stuck count is large
  would be the cheap fix; it belongs to the step driver.
- **Clustered, 262k**: after step 1, one Delaunay edge has no face (faces =
  clipping - 1, Delaunay neighbours equal): the edge-representative
  bookkeeping of the step loses one edge. Not seen at 32k. Open.
- The jittered patch cost is dominated by a full second `from_rings`
  (45 ms of 70); a local reassembly of the patched cells, or a faster
  `from_rings` (emit/adjacency are random-access lookups), would bring it
  under 10 %.
- Snapped duplicates (two particles in one grid cell) make every
  predicate flat; the check reports them, nothing repairs them.
