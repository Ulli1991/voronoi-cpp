# voronoi-cpp

Kokkos C++17 library for exact local 3D Voronoi meshes in a periodic box on GPUs
(CUDA validated on A100; OpenMP/Serial builds for checks). It is the geometry
base for a planned moving-mesh finite-volume (MFV/MFM) hydro solver. Each face
is stored once per particle pair, by the cell with the lower slot index (its
owner), so a flux computed on that face and applied with opposite signs
conserves exactly. The hydro solver itself is not started; this repository
builds and maintains the mesh. Multi-GPU execution is future work.

## How the pieces fit

The library is header-only (`src/neighbour_search/`, `src/voronoi_construct/`),
one job per header:

| Header | Job |
|---|---|
| `neighbor_search.hpp` | `CellList`: counting sort of the particles into cubic grid cells (optionally Morton-ordered) |
| `shell_voronoi.hpp` | full rebuild by clipping, thread per cell: `construct`, the portable kernel, the overflow pass, and the only kernel that records face rings |
| `shell_voronoi_warp.hpp` | full rebuild, warp per cell on CUDA: `construct_warp`, the default |
| `exact_predicates.hpp` | exact orientation and in-sphere signs on snapped positions |
| `tet_mesh.hpp` | persistent Delaunay tetrahedra: `from_rings`, `circumcentres`, `faces`, `check`, face rings |
| `tet_start.hpp` | the exact starting tetrahedralization: exact check of `from_rings`, exact clipping of the cells around errors |
| `tet_repair.hpp` | per-step repair by flips on the device |
| `tet_flips.hpp` | host fallback of the repair, including local Bowyer-Watson rebuilds |

**Full rebuild.** `neighbor_search::build` sorts the particles into a cell
list with about one particle per grid cell; `shell_voronoi::make_grid` hands it
to the kernels. Slots, positions and cell offsets follow the cell order;
`list.order(slot)` is the original particle ID. Each cell is built by walking
grid cells around its generator and clipping a polyhedron with every particle
that can still cut it: a particle at distance d cuts only if d < 2R (R is the
largest generator-to-vertex distance), and the walk stops at the first grid
cell entirely beyond 2R, which certifies the cell. Output (`Cells`) is lean:
volume and centroid per cell, faces stored once by their owner.
`construct_warp` builds one cell per warp in shared memory (lane l owns
vertices and planes l and l + 32; ring particles are tested 32 at a time and
clipped nearest-first after a warp sort) with persistent warps taking cells
from a counter. Cells that overflow its capacities, or need more than its
stencil of grid cells, are rebuilt by the thread kernel; cells that overflow
that too get a big-capacity pass. On non-CUDA backends, and when face rings
are requested, `construct_warp` is the thread kernel.

**Starting mesh.** After a rebuild with face rings, `tet_start::build`
assembles the tetrahedra and their adjacency on the device
(`tet_mesh::from_rings`: each tetrahedron emitted by the face of its two
smallest particles), then checks them exactly: no flat tetrahedron, mutual
adjacency, apexes on opposite sides of every shared triangle, local
Delaunay under the perturbed in-sphere test, every particle used, volumes
summing to the box. These together certify the unique Delaunay
tetrahedralization of the snapped positions (ties broken by the symbolic
perturbation). Where the check fails, the cells around the bad tetrahedra
are rebuilt by exact clipping (vertices named by particle triples, every
cut an exact in-sphere sign behind a filter) and the mesh is assembled and
checked again; if many cells are bad (lattices) or a second check fails,
every cell is rebuilt exactly. Rounding decides the clipping's near-ties,
so its rings are wrong on lattices and other cospherical layouts and, at
2.1M cells, at a few places on jittered and random input too.

**Persistent mesh.** Each tetrahedron is a
Voronoi vertex (its circumcentre), and a Voronoi face is the ring of
circumcentres around a Delaunay edge; every edge has one representative
tetrahedron (6 bits per tetrahedron). Per step:

1. `tet_mesh::circumcentres` (slivers fall back to exact integer arithmetic);
2. `tet_repair::move_screened` snaps the new positions and lists the
   tetrahedra whose local Delaunay test could fail; `tet_repair::repair` flips
   them on the device (failing triangles planned as 2->3 or 3->2 flips; plans
   vote on the tetrahedra they rewrite, winners apply, losers retry; a flip is
   only accepted if it closes its cavity exactly);
3. if the device repair reports stuck entries, `tet_flips::finish` repairs a
   host copy (flips, then local rebuilds) and copies it back;
4. `tet_mesh::circumcentres` again, `tet_mesh::update_face_rings` (a face
   keeps its ring unless a flip touched it, else its edge is walked) and
   `tet_mesh::faces` (area vectors, centroids, cell volumes).

Every repair decision uses `exact_predicates.hpp`: positions snapped to a
2^30 grid, a filtered double evaluation with an error bound, and an exact
fallback (128-bit orientation, 256-bit in-sphere), with symbolic perturbation
for ties, so the device and the host agree on one triangulation.

## Measured performance

One A100 40 GB, double precision, 2,097,152 cells (128^3), periodic unit box.
Rates in million cells per second. The gpudev node shares its GPU, so compare
variants only within one job; numbers below are best-node values.

| Stage | Rate |
|---|---|
| Full rebuild, warp kernel, jittered | 18.7 |
| Full rebuild, warp kernel, uniform random | ~10.4 |
| Full rebuild, warp kernel, clustered | ~0.9 |
| Tetrahedra from rings (once after a rebuild) | 40 |
| Exact start (ring clipping + `tet_start::build`), jittered | 4.5 (ring clipping alone 6.2; the patch of 42 cells costs 70 ms, mostly a second `from_rings`) |
| Exact start, perfect lattice (every cell clipped exactly) | 7.4 |

A thread-per-cell kernel with the polyhedron in shared memory (the removed
"lean" kernel) reached 10.4 on jittered input.

Per step at 0.03 spacings of motion (jittered), complete mesh update (repair
with screening, circumcentres, face rings and faces): about 16.8 ms, i.e.
~124 M cells/s; ~22 ms at 0.1 spacings and ~35 ms at 0.3.

Comparisons on the same A100 (research_notes/arepo_wd_gpu_benchmark.md and
the status notes):

| Code | Mesh | Full step |
|---|---|---|
| this work | 18.7 M cells/s rebuild; ~124 M cells/s persistent step | no hydro yet |
| PROTEUS | 1.6-6 M cells/s | Sedov, 5 M updates/s |
| AREPO-WD GPU per-cell builder | 1.4-1.6 M cells/s | Sedov, 0.59 M updates/s |
| LUNA (fixed grid) | n/a | 1,100 M updates/s |

Nsight on the 2.1M jittered case shows the rebuild is divergence-bound, not
memory-bound (DRAM <1 %, L2 hit 99 %). Morton cell order, coarser cell lists,
fast-math, octant culling and 16- or 8-lane teams per cell gave no gain.
Details: `research_notes/status_2026-10-29.md`; the cleanup that produced this
layout: `research_notes/cleanup_2026-10-30.md`.

## Known limitations and open items

- Clustered input is 16-20x slower than jittered: a single cell-list size cannot
  fit clusters and voids (a multi-level cell list is planned).
- Clipping alone is not watertight on uniform random input in double at 2.1M
  cells (some face pairs inconsistent), and its rings are wrong on lattices
  and cospherical layouts; `tet_start::build` makes the starting mesh exact
  for all inputs of `support.hpp` except below.
- A cell can store at most 32 owned faces (`shell_voronoi::max_owned_faces`);
  a cell with more (pairs 1e-6 spacings apart, one cell in 262k) cannot be
  patched and `tet_start::Stats::exact` stays false.
- Particles must sit on the 2^30 snapping grid of the exact predicates: the
  topology is decided on snapped positions, the geometry computed from the
  positions. Off the grid, near-degenerate cells differ from clipping by
  ~1e-9 in volume and particles closer than ~1e3 grid steps can get wrong
  geometry. The tests and benchmark keep positions on the grid.
- The first step out of a degenerate start (lattice, wave, sc@1e-6, shells)
  rearranges most tetrahedra; the device repair leaves 2e3 (lattice) to 1e6
  (wave, sc@1e-6) triangles stuck at 2.1M cells and the host fallback takes
  7-80 s. Exact, but a device rebuild (about 1 s) would be the faster guard.
- Clustered input at 262k cells: after the first step one Delaunay edge has
  no face (face count one below clipping), a 7e-8 volume error in one tiny
  cell; edge-representative bookkeeping, open.
- GPU repair breaks down near one spacing of motion per step; it needs
  substepping or a rebuild guard.
- The float variants and degenerate layouts still fail some checks of
  `voronoi_bench rebuild`; the tests cover double precision.
- Periodic box only. MFV/MFM hydro is not started.

## Build, test and run

Raven, CUDA build:

```sh
module load gcc/14 cuda/13.0 cmake
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda \
  -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper
cmake --build build-cuda -j4
```

OpenMP build: `module load gcc/15`, Kokkos `/u/uli/kokkos-gcc15`, same cmake
call with a different build directory. Elsewhere, point `CMAKE_PREFIX_PATH` at
a Kokkos installation with the desired backend. Build large configurations
and run GPU jobs through Slurm, not on the login node;
`scripts/bench_a100.sh` is an example job.

Tests (`ctest --test-dir build-cuda --output-on-failure`, about 75 s on one
CPU core), each a short specification in `src/test/`:

| Test | Specifies |
|---|---|
| `exact_predicates` | signs equal an independent integer evaluation, near-degenerate and large inputs included; ties consistent; periodic offsets |
| `clipping` | jittered, uniform, lattice, clustered: every cell certified, volumes fill the box, cells close, faces consistent and on the bisectors; thread and warp kernels and Morton order agree |
| `mesh` | `from_rings` equals a host construction, adjacency and edge representatives consistent, volumes from stored and walked rings equal clipping, sliver circumcentres exact |
| `start` | the starting mesh of random, clustered, lattice and degenerate inputs is a triangulation of the box with every circumsphere empty (brute-force host oracle, exact predicates), i.e. the Delaunay tetrahedralization |
| `repair` | steps of a smooth flow at 0.03 and 0.3 spacings from jittered, lattice and wave starts (and shells at 0.03): the repaired mesh equals an exact rebuild (neighbours, volumes) every step |

`voronoi_bench` (`src/bench/`) times the pipeline and helps debugging:

```text
voronoi_bench rebuild <n> <distribution> [variants=all] [repeats=5] [--width=<factor>] [--morton]
voronoi_bench step <n> <distribution> <displacement> [steps=10] [--verify] [--debug]
```

Distributions are uniform, jittered, lattice, wave, clustered, or
`<sc|bcc|fcc|shells|planes|pairs>@<perturbation in spacings>`. `rebuild`
times the thread-/warp- kernels in -double/-float (selected by substring)
and checks every variant's cells. `step` times the exact start (clipping
with rings, `from_rings`, `tet_start::build`) and the per-step kernels, then steps the flow `displacement` spacings at a time;
`--verify` compares every step with an exact rebuild, `--debug` adds
adjacency, stuck and worst-cell details.

```sh
# full rebuild, warp kernel, double, 3 repeats
voronoi_bench rebuild 2097152 jittered warp-double 3
# persistent mesh: 10 steps of 0.03 spacings
voronoi_bench step 2097152 jittered 0.03 10
# the same at 0.3 spacings, checked against an exact rebuild every step
voronoi_bench step 262144 jittered 0.3 10 --verify
```

## Repository layout

- `src/neighbour_search/`, `src/voronoi_construct/`: the library (above).
- `src/test/`: the tests, `support.hpp` (inputs, moving-mesh driver,
  reference results; shared with the benchmark) and `check.hpp`.
- `src/bench/`: `voronoi_bench`.
- `scripts/`: `lint_cpp.sh`, `bench_a100.sh`.
- `research_notes/`, `reports/`: status notes, AREPO/PROTEUS/GIZMO/SPH-EXA
  surveys, Kokkos CUDA audit, literature reports.

CMake exposes `voronoi_library` with both include directories. The earlier
table-based pipeline (`local_voronoi*.hpp`) and its tools are on the branch
`archive/codex-era`.

## Linting and continuous integration

GitHub Actions (`.github/workflows/checks.yml`) runs clang-tidy and the CPU
tests in ordinary and address/undefined-behavior sanitizer builds against a
pinned Serial Kokkos 5.0.2. It does not run GPU jobs.

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/kokkos -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j 2
bash scripts/lint_cpp.sh build
ctest --test-dir build --output-on-failure
```

Add `-DVORONOI_SANITIZERS=ON` in a separate host build for sanitizers.
