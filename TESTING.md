# Testing voronoi-cpp

Ten CTest programs in `src/test/` specify the library (`mesh_cases`
runs as ten groups, each case also on its own by name); `voronoi_bench`
(`src/bench/`) adds the large-scale checks that are too slow for CTest.
Every test prints one `ok` or `FAIL` line per expectation and exits non-zero
on any failure. All tests run on the default Kokkos execution space, so a
CUDA build checks the device code and an OpenMP or Serial build the host
code.

## Running them

```sh
ctest --test-dir <build> --output-on-failure            # all of them
OMP_NUM_THREADS=1 ctest --test-dir <build-omp>          # CPU, one thread
```

On Raven, `scripts/bench_a100.sh <build-cuda>` (submit with sbatch, see the
README) runs the CUDA tests and the A100 checks below in one gpu1 job.

## The tests

CPU times: OpenMP build (gcc 15), one thread of a Raven login node
(Xeon Platinum 8360Y), 2026-10-31. On the A100 the seven 3D programs take about 35 s, the two 1D ones a few seconds.

| Test | n | CPU, 1 thread | What it specifies |
|---|---|---|---|
| `exact_predicates` | random + constructed | 0.1 s | orientation and in-sphere signs equal an independent integer evaluation (near-degenerate and 2³⁰-scale inputs included); ties under the symbolic perturbation are consistent from every corner order and from both sides of a triangle; periodic offsets take the nearest image; `snap(x, box)` maps any box onto [0, 2³⁰) and wraps the edge; `Box(lx, ly, lz)`: one step for every axis, sides whole numbers of steps (2 x 2 x 1/16 exact, 1 x 2 x 3 rounded), a cubic `Box` snaps as before, per-axis wrap and periods, grid positions of a 1 x 2 x 3 box snap to themselves; orientation and in-sphere across the thin side of a slab; snapping k * step gives k back in 200 boxes of random side (4e6 positions) |
| `clipping` | 4096 | 15 s | on jittered, uniform, lattice, clustered and halo points: every cell certified, volumes positive and summing to the box, area vectors closing each cell, every face seen from both sides, every face centroid on the bisector with no generator closer (`support::check_cells`, brute force); the thread (rounded), warp and exact kernels give the same cells; a coarser cell list gives the same cells; a too small face pool grows; a particle at NaN and one at infinity are never candidates (every other cell has the neighbours and, to 1e-12, the volume of the build without them; bit-identical on CPU, on CUDA two cells differ in the last bits because the extra particles change grid levels nearby) |
| `mesh` | 4096 | 0.6 s | `from_rings` gives the same tetrahedra as a host construction from the rings, each once, with mutual adjacency and one representative per edge; the rings are canonical (faces by neighbour id, each ring from its least id) and entry for entry those of the exact thread kernel; volumes from stored and from walked rings equal clipping to 1e-12; the exact check passes; a small ring pool grows; the circumcentre of a sliver is exact; the 2D pieces of the dimension templates: `CellList<Exec, 2>` puts the slots in (tile, Morton key, id) order of a host sort (576 points, 2 x 1 tiles), 2D `tet_mesh::cells` on a hand-made square lattice gives 4 faces per cell, area h² and the centroid at the generator (1e-15) |
| `start` | ~4096 (13 layouts), 32768 | 12 s | the start (`tet_start::build`) is the Delaunay tetrahedralization of the snapped positions for jittered, uniform, clustered, lattice, wave, sc/bcc/fcc, icosahedral shells, planes and pairs, in boxes of side 1, 2 and 3, and in non-cubic boxes: jittered and an exact sc lattice in a 1:1:0.125 slab (64 x 64 x 8 = 32768), uniform and an exact bcc lattice in a 1:2:3 box (~4096): a brute-force oracle checks that it triangulates the box (every triangle twice with apexes on opposite sides, adjacency, all particles used, exact volume sum) and that every circumsphere is empty (exact, perturbed); every edge has one representative (also wave at 32768, the dropped-face regression); jittered needs no patch; the oracle rejects a mesh with one corner moved |
| `repair` | ~4096 | 10 s | steps of a smooth flow through `voronoi::Mesh`: jittered (0.03 and 0.3 spacings, 10 steps), lattice and wave (0.03 and 0.3, 4 steps; their starts are decided by the perturbation alone, wave's first step is the step-1 regression), shells@1e-9 (0.03, 2 steps), jittered in a box of side 2 (0.3, 3 steps), uniform in a 1:2:3 box (0.3, 3 steps), jittered in a 1:1:0.125 slab (32768, 0.3, 2 steps); after every step no failing triangle (brute force), the same Delaunay neighbours and face count as an exact rebuild by clipping, volumes within 1e-10, total volume the box, per-cell lists (`update_cells`, incremental after few flips) naming each face in both its cells in increasing id of the other particle |
| `voronoi_mesh` | 4096 | 2 s | the contract of `voronoi::Mesh` that luna-cpp relies on, on off-grid positions in a box of side 2: the first update builds an exact mesh; faces once with owner < neighbour, closing every cell; volumes fill the box; per-cell face lists name each face twice, in increasing id of the other particle; cell centroids equal clipping's; the same input gives the same volumes and centroids bit for bit; a permuted input gives the permuted faces; a local rebuild returns each face of the listed cells once with the mesh's area; updates repair and stay exact; on icosahedral shells@1e-9, where the device repair leaves stuck triangles, the step is exact with `Fallback::host` (local rebuilds on the host) and with `Fallback::rebuild`; in a 1:2:3 box the mesh is exact, its volumes fill the box and `box()` refuses it; a slab 4 cells thin is refused with the "cannot certify" error |
| `order` | 4096 | | the memory layout (#34): on an 8³ grid Hilbert and Moore move to a neighbouring cell at every step and Moore closes its loop, Gray-coded Morton changes one Morton bit, row-major runs x fastest, every curve visits every cell; for each of the six orderings, re-sorted at the build and every second update over six steps of 0.3 spacings, the permutation is one, `permute()` carries the caller's rows and the mesh's positions are those rows, every particle keeps its neighbours (by its first number) and volumes and centroids equal an unsorted mesh's to 1e-12; faces by owner with the owner bit over six steps of 0.03 spacings (the incremental lists): faces in owner order, `owned_start` right, areas, volumes and centroids the same bits as without, entries carry the bit |
| `mesh1d` | 2048 | under 1 s | the 1D mesh (`voronoi::Mesh<1>`, `sorted_mesh.hpp`) against a host oracle, periodic and with walls, lines of length 1 and 3, uniform, lattice and clumps of snapped duplicates (ties by id): the sorted order, volumes and centroids exact in quarter grid steps of the 2⁵² grid, volumes summing to the line in integers, faces once with owner < neighbour, area +-1, per-cell face lists by id of the other particle, the end cells reaching the walls; moving: after every step (small ones that keep the order, large ones that cross, points wrapping through L) the mesh equals a fresh build bit for bit, and the order is sorted again only when points crossed; insertion and removal over random batches, on a rotated periodic line and with walls, on generators and twice in a gap: the oracle, a fresh build bit for bit, the reported overlaps against the host's integers, growth, a transfer by the report conserves to rounding |
| `mesh2d` | ~4096, 2²⁰ | 3 min | the 2D mesh (`voronoi::Mesh<2>`) against oracles that share no code with it: in-circle signs against a 4 x 4 integer determinant, the perturbed test consistent under every permutation; on random, jittered, clustered, lattice, hexagonal and wave inputs in square and non-square boxes: every triangle positive, every edge in two triangles, signed areas summing to the box in exact integers, no particle strictly inside a circumcircle, every cell clipped in long double by the bisectors of all particles in all images (volumes, centroids, neighbours, face lengths and centroids), face lists in order of the other id (brute force, O(n²), oracle only); after steps of 0.03 to 1 spacings a repaired mesh equals a fresh build bit for bit, with both fallbacks; repeated runs give the same hash; particles on one position refused by name at the start and in a step, no mesh after a refusal and the next update builds, no particles, a mapping of degree 4 and kicks of order the box |

### Special cases of the mesh

`mesh_cases_test` names every case (`mesh_cases_test list`) and runs those
whose name starts with a given prefix; ctest runs one group per prefix. A case
requires the exact mesh, or accepts a refusal (an exception) in its place; it
never accepts a wrong mesh without an error. A case of an open problem would
carry its issue, be skipped by default and run with `--known`: `repro.serial`
(#27).
ctest runs a subset within its time budget;
`MESH_CASES_FULL=1` runs the whole matrix, as the A100 job does. On the A100 a
failing case of a rare outcome needs repeats: `large.random_walk.10` is wrong
in about 2 % of steps, and rounds vary from run to run on that path.

| Group | What it specifies |
|---|---|
| `coincident` | two, three, 1 % and all particles on one grid position at the start (exact doubles and doubles apart inside one grid step, random and lattice): refused; one grid step apart: exact; a pair, 40 pairs and a quarter of the particles landing on one position during a step, separations from 1e-3 to below a grid step, two particles swapping through one point, every other particle at one grid step from a neighbour; each fallback with 1, 4 and 64 pairs; a refusal, at the start and at each fallback, names the pair by id (`coincident.message.*`) |
| `few` | n from 0 to 512, uniform, jittered, lattice: built and exact, or refused, three steps after a build; every cell against brute clipping; no particles refused, and a mesh it was given stays |
| `contrast` | density ratios 1e2 to 1e8, balls and shells of radius 1e-2 to 1e-8, with and without a centre particle, clumps on 512³ to 2⁶⁰ grid positions, 5 to 1000 particles in an empty box, a line, a plane, two skew lines, a line winding around the torus, a log-uniform cusp over 6 decades; shells down to radius 1e-6 and a hub of 2000 faces must build exact, below 1e-6 the sphere has real duplicates; what each clipping status is reported as (`contrast.refusal_messages`) |
| `box` | slabs 1 to 8 cells thin, 1:1:1/32 to 1:1:1/1024, long boxes to 1:1:65536, voids of radius 0.1 to 0.45, empty or with a centre particle, particles on and across every boundary (snapping equal to floor(x / step) mod period), positions run past the side, cubes of side 1e-9 to 1e9 give the same tetrahedra and scaled volumes, NaN, infinity and 1e300 at the start and in a step |
| `degenerate` | sc, bcc, fcc: rigid motion in whole grid steps (no flip) and in doubles, shear by layers that recurs as an exact lattice, smooth shear, fresh noise of 1, 16, 1024 grid steps every step; zero-area faces finite (#13); two lattices sliding through each other, coplanar every 8 steps; one particle walking through face centres and edge midpoints; octahedra pulsing through exact cospherical; 1014 grid points exactly on one sphere |
| `valence` | a face of 16 to 1000 corners at the start and created by the repair; spheres and shells of 50 to 1600 particles; a particle moved into a shell, a cell of up to 1600 faces no build saw; all must be exact, none refused |
| `large` | displacements of 0.5 to 20 spacings, random walks of 0.3 to 10, every particle somewhere else, permuted ids, no motion (no flip, the same bits), another n, 1000 steps of a solenoidal flow without drift in faces, slots or memory, there and back 8 times to the same tetrahedra and the same bits; a refused update leaves no mesh, no crash, and the next update builds, at the start and after the repair and the host (`large.after_refusal.*`); `update_cells()` without a mesh says so |
| `fallback` | a repair out of rounds (limit lowered to 1 and 2) never returns as a mesh; the cheapest fallback does no host work at 32768 cells (`fallback.cheapest.small_n`); all three fallbacks on shells, kicks of 2 spacings, and a jump from a lattice to two skew segments; growth in small steps; `clip()` of a hub with its pool of 10 faces per cell |
| `predicates` | the library's orient and in-sphere on the quintuples real meshes ask about (eight point sets, lattices with +-1 and +-3 grid steps of noise, a clump on 2048³ positions) equal the oracle's integers |
| `repro` | two runs give the same bits, the default space and Serial the same bits; `repro.dump.*` with `MESH_REPRO_DUMP=<file>` writes hashes per step to compare builds (measured: Serial, OpenMP at 1 and 8 threads and CUDA agree on every hash of faces, volumes and centroids, 4 inputs, 21 steps) |

`hostcost.*` (not in ctest) measures one step of `Fallback::host` against the
stuck entries it gets.

`distortion_test` drives `voronoi::Mesh` with prescribed velocity fields, no
hydro (`src/test/distortion.hpp`): shear, sawtooth, vortex, taylor_green,
turbulence, collapse, noh, expansion, pancake, counter (interpenetrating), thin
(long thin cells), and collapse_wide, pancake_wide, streams, which empty a
region as wide as the box and end in a refusal. Without regularisation, then on
three flows with a drift to the centroids in the driver: the library layer has
none. Every sampled step passes the oracle, or the run ends in a refusal. It
prints per step flips, rounds, stuck entries, fallbacks, M cells/s, faces,
B/cell, worst aspect ratio (largest circumradius at a cell over half the
distance to its nearest neighbour), reach, faces per cell and corners per face.
`scripts/mesh_tests_a100.sh` runs the cases on the A100, the flows at 128³ and
256³, the drift, and the full oracle at 262144.

### Random cases

`mesh_fuzz_test` (`src/test/mesh_fuzz_test.cpp`, generators in `mesh_fuzz.hpp`)
draws whole cases from one integer seed: box, points, the `Mesh` type and every
call, as a plain `Script` that can be saved and replayed. Digits of the seed above
10¹² choose the profile: 0 small, 1 long call sequences, 2 medium, 3 large, 4 huge
n, 5 1D, 6 2D, 7 giant n (3 to 6 million before the added parts, for the A100). A case ends `exact`, `refused` (a cause the input really has),
`valid` (a refusal of input that has a mesh, or with no clear cause), `bits`,
`rough` (exact tetrahedra, geometry past rounding), `WRONG`, `CRASH` or `HANG`.
Checked on top of `mesh_check::check`: snapping, face order and direction, centroids
on bisectors and against brute force, per-cell lists, float rows,
`clip()`, and a refusal for reach against the measured largest cell.

```sh
mesh_fuzz_test smoke --cases=src/test/fuzz_cases             # ctest: saved cases, 16 fixed seeds
mesh_fuzz_test run --first=S --count=N --results=FILE        # resumes from FILE; crashes and hangs are logged
mesh_fuzz_test one SEED [--dump=FILE]                        # one seed, verbose
mesh_fuzz_test case FILE                                     # a saved case
mesh_fuzz_test min SEED|FILE --out=FILE                      # delta debugging, keeps the signature (3D)
```

`src/test/fuzz_cases/*.case` are minimised failures; a line `known <issue>` skips one
until the issue is fixed (`--known=1` runs it), a line `expect exact` accepts nothing but exact
(`repeated_face_ratio.case`, #48). A build refused for its face count (`face_count`) is WRONG. ctest skips seeds in boxes of
non-dyadic step and of extreme side (#29; extreme sides are refused since #32), and counts `rough` without failing,
so most of the checking is in the long runs: `scripts/mesh_fuzz_cpu.sh` (`small`),
`scripts/mesh_fuzz_gpudev.sh` (15-minute chunks; resubmit the same command to go on),
`scripts/mesh_fuzz_a100.sh` (CUDA, profiles large, huge and giant on the four A100 of a node, 15-minute chunks).
Above 4096 particles a case is checked after its last call only, against brute force on 8 sampled cells and on all
cells for closure, total volume, face pairing and the tetrahedra's empty spheres.
The seeds of a run are tied to the generator: a changed generator needs a new base.
With a coverage build a worker writes its counters before `_exit`.

### Oracles

Brute force exists only in tests and the benchmark, never in the library:

- `support::check_cells`: each sampled face centroid against every
  generator, about 2.5e8 distances (at n = 4096 every cell is sampled).
- The start oracle (`start_test.cpp`): every circumsphere against every
  particle whose x lies within the circumradius (sorted by x, all
  particles if that spans the box), exact in-sphere where a double
  prefilter is unsure; n ~ 4096, 32768 for the slab.
- `support::failing`: every shared triangle of the whole mesh, exact.
- `support::clipped` (the reference of `compare_with_rebuild`): an
  independent build of the same positions by clipping.
- `oracle.hpp`, through `mesh_check::check`: shares no code with the library.
  Every periodic image, not only the nearest; orientation and in-sphere signs
  in its own 128-bit arithmetic, no tie-break (points on a sphere are allowed);
  corners that agree on each other's images; a closed cover with exact volumes;
  cells by half-space clipping in long double against every image of every
  generator (a sample of cells). Ties are compared with a fresh build:
  the tetrahedra must be equal as sets, since the perturbation makes them
  unique. Volumes of a cell h grid steps wide round at about 4e-8 / h.

### What PASS means

- `voronoi_bench rebuild ... ` (`check_cells`): every cell certified, total
  volume within 1e-12 of the box, closure (|sum of area vectors| / sum of
  |area|) at most 1e-12, faces = 2 x owned faces, face centroids on the
  bisectors within 1e-9 (relative). Uniform random input at 2.1M fails
  "faces = 2 x owned" by a handful of faces: the double-precision clipping
  alone is not watertight there (the start repairs it; known, expected).
- Wound-up meshes: past aspect ratios of about 1e5 the plain clipping can return
  cells `degenerate` with volume 0 (125 of 262144 in `thin` at t = 1.9), and
  `compare_with_rebuild` reports faces the reference lacks, up to 18 spacings²,
  while brute force agrees with the mesh to every digit. `distortion_test`
  then asks `mesh_check::arbitrate` (`rebuild=PASS_BRUTE`).
- `voronoi_bench step ... --verify` (`support::compare_with_rebuild`):
  no failing triangle, no clipping face the mesh lacks, mesh faces the
  clipping lacks only below 1e-9 spacings² (clipping drops rounding-level
  faces, the mesh keeps every Delaunay edge), largest relative volume
  difference below 1e-10.
- The start: `tet_start::Stats::exact` (the exact check passed).

## Regression tests kept

- Wave lattice, step 1 (`repair`: wave at 0.3 spacings): the device repair
  once dropped failing triangles beyond its work capacity; since the
  device tries every flip until one closes its cavity, it repairs the step
  alone (nothing sticks).
- Clustered input, missing face (`start`: every edge has one
  representative, wave at 32768): an edge whose owner cell dropped a
  rounding-level face had no representative and no Voronoi face.
- One-thread repair livelock (`repair` on one CPU thread): plans voted
  with their index, which on one thread ordered the winners in space and
  stalled the lattice's first step for 1000 rounds.

## The A100 checks (not in CTest)

`scripts/bench_a100.sh` runs, besides ctest:

```sh
voronoi_bench rebuild 2097152 <kind> 3                       # check_cells PASS/FAIL per kind
voronoi_bench step 262144 <kind> 0.3 3 --verify              # jittered uniform clustered halo lattice wave
```

and on non-cubic boxes (not in `bench_a100.sh`; job script in
research_notes/box3_2026-10-31.md):

```sh
voronoi_bench step 262144 <kind> 0.3 3 --verify --box=2,2,0.25   # 128 x 128 x 16: jittered uniform clustered sc@1e-6
voronoi_bench step 262144 <kind> 0.3 3 --verify --box=1,2,3      # uniform jittered
```

Each `verify:` line must end in `PASS`.

## Solver-level checks

The finite-volume solver is not in this repository. Its tests, the 1D comparisons against LUNA and the 3D gate runs
are in [luna-cpp](https://github.com/Ulli1991/luna-cpp), which calls this library for the mesh. Their earlier
descriptions are in research_notes/stage1_1d_2026-11-04.md and the git history.
