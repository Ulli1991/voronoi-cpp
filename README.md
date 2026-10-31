# voronoi-cpp

Kokkos C++ library for exact 3D Voronoi meshes of moving particles in a
periodic box (a cube or any three sides) on GPUs (CUDA, validated on an A100 40 GB; OpenMP and Serial
builds run the same code for checks). It is the geometry base of a planned
moving-mesh finite-volume hydro solver: the solver hands over particle
positions and gets back faces (owner, neighbour, area vector, centroid),
cell volumes and centroids, and per-cell face lists. The hydro solver
itself is not part of this repository.

## Install and build

Dependencies:

- CMake 3.16 or later, a C++17 compiler (gcc 14/15 tested).
- Kokkos 5.0 (5.0.2 in CI, 5.0.99 on Raven) with the backend you want:
  CUDA (A100: `Kokkos_ARCH_AMPERE80`, compiled through Kokkos'
  `nvcc_wrapper`; CUDA 13.0 tested), OpenMP or Serial. The CUDA backend
  needs Kokkos built with `Kokkos_ENABLE_CUDA_LAMBDA` (the default in 5.x).
- Nothing else: the library is header-only (`src/neighbour_search/`,
  `src/voronoi_construct/`); CMake exports it as the target
  `voronoi_library`.

CUDA (A100), on Raven:

```sh
module load gcc/14 cuda/13.0 cmake
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda \
  -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper
cmake --build build-cuda -j4
```

OpenMP (CPU), on Raven:

```sh
module load gcc/15 cmake
cmake -S . -B build-omp -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/u/uli/kokkos-gcc15 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-omp -j4
```

Elsewhere, point `CMAKE_PREFIX_PATH` at a Kokkos installation with the
desired backend. On Raven, build at most with `-j4` on the login node, or
inside a Slurm job; run GPU work through Slurm (`scripts/bench_a100.sh`).

### Compiler flags in use

- Release: `-O3 -DNDEBUG` (CMake's default), C++17 required (Kokkos sets
  `-std=c++20`), warnings `-Wall -Wextra -Wpedantic` on the programs.
- CUDA: `nvcc_wrapper` with `-extended-lambda -arch=sm_80` (from Kokkos'
  AMPERE80). The warp clipping kernel uses
  `Kokkos::LaunchBounds<128, 5>` (4 warps per block, 5 blocks per SM),
  which caps it at 96 registers; 18 KB of shared memory per block.
- OpenMP: `-fopenmp` from Kokkos; no `-march` is set by the project
  (`-DCMAKE_CXX_FLAGS=-march=icelake-server` is worth adding on Raven's
  CPUs).
- `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` (always on in `CMakeLists.txt`)
  writes the compilation database clang-tidy needs:
  `bash scripts/lint_cpp.sh build-omp` (Raven: `module load clang/20`).
- `-DVORONOI_SANITIZERS=ON` adds address and undefined-behaviour
  sanitizers to a host build.

## Using it: `voronoi::Mesh`

```cpp
#include "voronoi_mesh.hpp"

voronoi::Mesh<> mesh(Kokkos::DefaultExecutionSpace{}, box);   // a cube of side box, or
// voronoi::Mesh<> mesh(exec, exact_predicates::Box(lx, ly, lz)); // any three sides
mesh.update(x);          // x(i, a): positions in [0, side a), particle i = index i
// ... move the particles ...
voronoi::Report r = mesh.update(x);   // repairs the mesh; exact on return
mesh.update_cells();     // per-cell face lists, ordered volumes and centroids
// Faces f < mesh.faces(): owner()(f) < neighbour()(f), area()(f, a) from
// owner to neighbour, face_centroid()(f, a) relative to the owner.
// Cells i: volume()(i), centroid()(i, a) relative to the generator, faces
// cell_face()(k) for k in [cell_face_start()(i), cell_face_start()(i + 1)).
```

- Particles are the caller's indices; ties between cospherical particles
  are broken by them (symbolic perturbation), at the start and in every
  repair alike. A permuted input gives the permuted mesh.
- Positions are snapped to the grid of the exact predicates inside, 2^30
  steps across the longest side, the same step on every axis (so the
  integer in-sphere test is the test of the snapped points); topology and
  geometry both come from the snapped positions. A side that is not a
  whole number of steps is rounded to one (relative change below 2^-31
  longest / side; none for sides in power-of-two ratios such as
  2 x 2 x 1/16); `sides()` gives the box used (`box()` is a cube's side).
- Every cell must be certified within one periodic image: a cell's radius
  must stay below about 3/16 of the shortest side (in practice about 8 or
  more particles across it; a slab of 16 cells is fine). Otherwise
  `update()` throws, saying so.
- `update()` repairs (the first call, or a change of n, builds); exact on
  return. When the device repair leaves stuck triangles it finishes on
  the host or rebuilds on the device, whichever is cheaper
  (`voronoi::Fallback`).
- Faces are stored once, by the lower id (the owner rule), so a flux
  applied with opposite signs conserves exactly. Cell volumes after
  `update()` are summed with atomics; `update_cells()` sums volumes and
  centroids over each cell's faces in a fixed order, the same bits every
  run, and gives the per-cell face lists for gathers.
- `clip(x, cells)` rebuilds the listed cells by clipping, independent of
  the persistent mesh, and returns each of their faces once (a local mode
  for substeps of active cells).
- `rebuild(x)` builds the exact mesh from scratch on demand.
- `scratch(bytes)` lends device memory between updates (the spare ring
  pool).

## How it works

| Header | Job |
|---|---|
| `neighbor_search.hpp` | `CellList`: particles sorted along a Morton curve, an implicit octree over a dense base grid (per tile: the box is cut into near-cubes of its shortest side, numbered row-major above the Morton key), a grid level per particle fitted to the local density |
| `exact_predicates.hpp` | the periodic `Box` and snapping (`snap(x, box, axis)`), exact orientation and in-sphere signs (filtered double, then 128/256-bit integers), symbolic perturbation by particle id |
| `shell_voronoi.hpp` | clipping, thread per cell: the walk, `Cells` output (owned faces or face rings in pools), rounded and exact cells |
| `shell_voronoi_warp.hpp` | clipping, warp per cell on CUDA (`construct_warp`), with faces or face rings |
| `tet_mesh.hpp` | persistent tetrahedra: `from_rings`, circumcentres, face rings kept in place, faces, per-cell face lists |
| `tet_start.hpp` | the exact start: exact check of `from_rings`, exact clipping of the cells around errors |
| `tet_repair.hpp` | per-step repair by 2-3 / 3-2 flips on the device |
| `tet_flips.hpp` | host fallback of the repair (local Bowyer-Watson rebuilds) |
| `voronoi_mesh.hpp` | `voronoi::Mesh`: the pipeline below behind one interface |

**Clipping.** Each cell is built by walking the grid cells of its level
around its generator, nearest first, clipping with every particle that
can still cut it: a particle at distance d cuts only if d < 2R (R the
largest generator-to-vertex distance); the walk stops at the first grid
cell beyond 2R, which certifies the cell, and goes one level coarser when
the stencil ends first. The warp kernel builds one cell per warp in shared
memory and writes owned faces (neighbour, area, centroid) into a pool, 256
entries claimed per warp at a time, or face rings. Cells it cannot hold or
certify are built by the per-thread kernel with rounded cells; its exact
cells (vertices named by particle triples, every cut an exact perturbed
in-sphere sign) build the start's patches.

**Exact start.** Face rings from the warp kernel give the tetrahedra and
their adjacency (`tet_mesh::from_rings`). `tet_start::build` checks them
exactly (no flat tetrahedron, mutual adjacency, apexes on opposite sides,
local Delaunay under the perturbed in-sphere test, every particle used,
volumes summing to the box: together the Delaunay tetrahedralization) and
rebuilds the cells around anything wrong by exact clipping.

**Persistent mesh.** Per step: snap; screen the tetrahedra whose local
Delaunay test may fail (double, with a margin) and repair them by flips on
the device (plans vote by a hash of their index; every flip decided by
exact predicates and checked to close its cavity, the next candidate
flip if it does not); rebuild the surroundings of stuck triangles on the
host, or rebuild the mesh; circumcentres of the created tetrahedra; walk only
the faces whose ring changed; face geometry.

## Measured performance

One A100 40 GB (Raven gpu1), double precision, periodic unit box, job
31694457, commit f32c909. Rates in million cells per second.

| | 2,097,152 cells (128^3) | 16,777,216 cells (256^3) |
|---|---|---|
| Warp rebuild, jittered | 34.7 (60.4 ms) | 34.8 (482 ms) |
| Warp rebuild, uniform random | 16.5 | |
| Warp rebuild, clustered (Gaussian blobs, ~1:500) | 13.0 | |
| Warp rebuild, halo (1:10^4 core to edge) | 12.1 | |
| Warp rebuild, lattice / wave | 46.0 / 33.5 | |
| Exact start (warp with rings + `tet_start::build`), jittered | 120 ms | 1.43 s |
| `voronoi::Mesh` first update (start, state, all faces) | 146 ms | 2.14 s |
| Step at 0.03 spacings (repair, centres, rings, faces) | 11.1 ms (189 M/s) | 64 ms (262 M/s) |
| Step at 0.1 spacings | 18.9 ms | |
| Step at 0.3 spacings | 37.1 ms | 180 ms |
| `update_cells()` (face lists, ordered volumes and centroids; job 31697589) | 6.9 ms | 54 ms |
| Device memory, plain warp rebuild | 701 B/cell | 656 B/cell |
| Device memory peak, start and steps (with face lists) | 2221 B/cell | 2200 B/cell (37.9 GB in use) |

Non-cubic: a Kelvin-Helmholtz-like slab (Lx = Ly = 2, Lz = 16 spacings,
`--box=2,2,0.0625`, 512 x 512 x 16 = 4.2M, and `--box=2,2,0.03125`,
1024 x 1024 x 16 = 16.8M; job 31702953) rebuilds at the cube's rate:
warp rebuild jittered 39.9 / 39.9 M cells/s, uniform 19.9 / 20.0 M
cells/s; exact start 0.36 / 1.40 s; steps at 0.03 spacings 39.9 / 155 ms
(the bench flow, stretched with the box, shears across the 16-cell
thickness: 4x the flips per cell of the cube); peak 2192 B/cell at 16.8M
(research_notes/box3_2026-10-31.md). The cubic case is bit-identical to
the cubic-only code and within 0.3-0.7 % of its rebuild speed.

Steps are medians of steps 2-10 on jittered points moved by a smooth flow;
the step at 256^3 moves particles by the same number of spacings, but the
flow varies on the box scale, so neighbours shear half as much per step
(half the flips per cell). On jittered, uniform and clustered input at
0.1 and 0.3 spacings no triangle ever sticks; the first step out of a
degenerate start (wave lattice, 2.1M) is repaired on the device alone in
0.81 s (job 31701509; research_notes/shrink_2026-10-31.md), while
icosahedral shells@1e-9 still leave ~8k of 262k stuck (host or rebuild). The per-cell face lists account for most of the step peak above
2fd8b50's 2080 B/cell (about 140 B/cell; not measured separately).

## Tests

Six CTest programs specify the library; details, oracles, run times and
what PASS means are in [TESTING.md](TESTING.md).

| Test | Specifies |
|---|---|
| `exact_predicates` | signs equal an independent integer evaluation; ties consistent; nearest periodic image; snapping for any box |
| `clipping` | cells certified, closing, consistent and on the bisectors (brute force); thread, warp and exact kernels agree; pools grow; NaN/infinite particles never cut |
| `mesh` | `from_rings` against a host construction; adjacency, edge representatives, exact check, volumes from rings; exact sliver centres |
| `start` | the start is the Delaunay tetrahedralization (brute-force oracle) for 13 layouts and boxes of side 1, 2, 3, a 1:1:0.125 slab and a 1:2:3 box |
| `repair` | after every step of a flow the mesh equals an exact rebuild (lattice, wave, shells, jittered; box 2, a 1:2:3 box, a 1:1:0.125 slab) |
| `voronoi_mesh` | the hydro contract of `voronoi::Mesh` |

```sh
ctest --test-dir build-cuda --output-on-failure         # A100: ~35 s
OMP_NUM_THREADS=1 ctest --test-dir build-omp            # one CPU core: ~45 s
sbatch -o build-cuda/bench-%j.out scripts/bench_a100.sh build-cuda   # ctest + A100 checks
```

`voronoi_bench` (`src/bench/`) times the pipeline and checks it at scale:

```text
voronoi_bench rebuild <n> <distribution> [variants=thread,exact,warp] [repeats=5] [--width=f] [--no-check] [--memory] [--box=Lx,Ly,Lz]
voronoi_bench step <n> <distribution> <displacement> [steps=10] [--verify] [--debug] [--fallback=host|rebuild] [--memory] [--box=Lx,Ly,Lz]
```

`--box` (default 1,1,1) sets the periodic box; lattices get about equal
spacings along every axis (`--box=2,2,0.0625` with 4194304 jittered
points is a 512 x 512 x 16 slab), the other distributions stretch with
the box.

Distributions: uniform, jittered, lattice, wave, clustered, halo, or
`<sc|bcc|fcc|shells|planes|pairs>@<perturbation in spacings>`.

## Limitations and open items

- The warp rebuild is 1.5-4 % slower than on 2fd8b50 (uniform 16.5 vs
  17.2 M cells/s), the cost of the pooled face output and of particle ids
  in the clipping output (research_notes/cleanup2_2026-10-31.md).
- Clipping alone is not watertight on uniform random input at 2.1M cells;
  the exact start makes the mesh exact.
- Snapped duplicates (two particles in one grid cell) make every predicate
  flat; nothing repairs them.
- The device repair breaks down near one spacing of motion per step
  (stuck triangles, host or rebuild fallback); substepping helps.
- Ghost particles (multi-GPU) and repair limited to active regions are not
  implemented; `update()` takes an `active` list for later.
- Periodic boxes only (any three sides; see the certification limit
  above for thin slabs).

## Repository layout

- `src/neighbour_search/`, `src/voronoi_construct/`: the library.
- `src/test/`: the tests, `support.hpp` (inputs, references, the
  comparison with an exact rebuild), `check.hpp`.
- `src/bench/`: `voronoi_bench`.
- `scripts/`: `lint_cpp.sh`, `bench_a100.sh`.
- `research_notes/`, `reports/`: status notes and surveys.

CI (`.github/workflows/checks.yml`) runs clang-tidy and the CPU tests in
ordinary and sanitizer builds against a pinned Serial Kokkos 5.0.2; it does
not run GPU jobs.
