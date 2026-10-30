# voronoi-cpp

Kokkos C++ library for exact 3D Voronoi meshes of moving particles in a
periodic box on GPUs (CUDA, validated on an A100 40 GB; OpenMP and Serial
builds run the same code for checks). It is the geometry base of a planned
moving-mesh finite-volume hydro solver: the solver hands over particle
positions and gets back faces (owner, neighbour, area vector, centroid),
cell volumes and centroids, and per-cell face lists. Stage 1 of the hydro
solver (static mesh, MUSCL-Hancock, HLLC) is in `src/hydro/` (see "Hydro").

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

voronoi::Mesh<> mesh(Kokkos::DefaultExecutionSpace{}, box);
mesh.update(x);          // x(i, a): positions in [0, box), particle i = index i
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
- Positions are snapped to the 2^30 grid of the exact predicates inside;
  topology and geometry both come from the snapped positions.
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
| `neighbor_search.hpp` | `CellList`: particles sorted along a Morton curve, an implicit octree over a dense base grid, a grid level per particle fitted to the local density |
| `exact_predicates.hpp` | snapping (`snap(x, box)`), exact orientation and in-sphere signs (filtered double, then 128/256-bit integers), symbolic perturbation by particle id |
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

One A100 40 GB (Raven gpu1), double precision, periodic unit box. Rates
in million cells per second. 2.1M column: job 31701509 (commit 2311937's
code); 256^3 column: job 31694457 (commit f32c909, before the plume-derived
speedups, so the current code is faster there).

Throughput in million cells per second (cells processed per second of GPU time).

| Million cells per second | 2,097,152 cells (128^3) | 16,777,216 cells (256^3) |
|---|---|---|
| Full rebuild (warp clipping), jittered | **40.1** | 34.8 |
| Full rebuild, uniform random | **20.0** | |
| Full rebuild, clustered (Gaussian blobs, ~1:500) | **15.6** | |
| Full rebuild, halo (1:10^4 core to edge) | **14.3** | |
| Full rebuild, lattice / wave | **50.2 / 38.5** | |
| Exact start (warp with rings + `tet_start::build`), jittered | 17.7 | 11.7 |
| Moving-mesh update, 0.03 spacings per step | **201** | 262 |
| Moving-mesh update, 0.1 spacings per step | 115 | |
| Moving-mesh update, 0.3 spacings per step | 57 | 93 |
| `update_cells()` alone (face lists, ordered volumes and centroids) | 326 | 311 |
| Moving-mesh update + `update_cells()`, 0.03 spacings (bit-reproducible) | **124** | |

Inputs (all in the periodic unit box, `src/test/support.hpp`):

| Name | Points | Tests |
|---|---|---|
| jittered | cubic lattice, each point moved randomly by up to ±0.1 lattice spacing per axis | the typical hydro start; nearly regular, ~16 faces per cell |
| uniform random | independent uniform points | unstructured cells of very different size and shape |
| clustered | 8 Gaussian blobs (σ = 0.04) with 80 % of the points, 20 % uniform background | density contrast ~1:500 (blob centre to void) |
| halo | 4 halos with ρ ∝ 1/(r² + c²) to r = 0.2, c = 0.002, 90 % of the points, 10 % background | density contrast ~1:10^4 (core to edge) |
| lattice | perfect cubic lattice | fully degenerate: 8 points on every circumsphere |
| wave (moved lattice) | the lattice with every point moved by up to 1e-5 spacing | nearly degenerate: ties broken only by tiny offsets |

Moving-mesh update rates are per step of a smooth flow that moves the
points by the given number of mean interparticle spacings per step
(spacing = (box volume / n)^(1/3)): 0.03 spacings is a typical hydro
step, 0.3 a very large one (more flips per step, lower throughput).

Device memory: plain warp rebuild 701 B/cell (656 at 256^3); peak during
start and steps, with face lists, 2221 B/cell (2200 at 256^3, 37.9 GB in use).

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

Seven CTest programs specify the library and the hydro; details, oracles, run times and
what PASS means are in [TESTING.md](TESTING.md).

| Test | Specifies |
|---|---|
| `exact_predicates` | signs equal an independent integer evaluation; ties consistent; nearest periodic image; snapping for any box |
| `clipping` | cells certified, closing, consistent and on the bisectors (brute force); thread, warp and exact kernels agree; pools grow; NaN/infinite particles never cut |
| `mesh` | `from_rings` against a host construction; adjacency, edge representatives, exact check, volumes from rings; exact sliver centres |
| `start` | the start is the Delaunay tetrahedralization (brute-force oracle) for 13 layouts and boxes of side 1, 2, 3 |
| `repair` | after every step of a flow the mesh equals an exact rebuild (lattice, wave, shells, jittered, box 2) |
| `voronoi_mesh` | the hydro contract of `voronoi::Mesh` |
| `hydro` | HLLC and HLL against reference fluxes and under rotation; uniform states stay uniform, conservation to rounding, bit-identical runs, the two update loops agree to rounding on jittered and uniform meshes; moving mesh: uniform state with a bulk velocity, conservation, Galilean invariance of a moving blast |

```sh
ctest --test-dir build-cuda --output-on-failure         # A100: ~35 s
OMP_NUM_THREADS=1 ctest --test-dir build-omp            # one CPU core: ~45 s
sbatch -o build-cuda/bench-%j.out scripts/bench_a100.sh build-cuda   # ctest + A100 checks
```

`voronoi_bench` (`src/bench/`) times the pipeline and checks it at scale:

```text
voronoi_bench rebuild <n> <distribution> [variants=thread,exact,warp] [repeats=5] [--width=f] [--no-check] [--memory]
voronoi_bench step <n> <distribution> <displacement> [steps=10] [--verify] [--debug] [--fallback=host|rebuild] [--memory]
```

Distributions: uniform, jittered, lattice, wave, clustered, halo, or
`<sc|bcc|fcc|shells|planes|pairs>@<perturbation in spacings>`.

## Hydro

`hydro::FiniteVolume` (`src/hydro/`) runs finite volumes on a static or
moving `voronoi::Mesh` (moving: the points follow the flow with a drift
towards the cell centroids, the mesh is repaired every step, fluxes go
through the moving faces): extensive conserved state only between steps, least-squares
gradients of the primitives limited at the face centroids, MUSCL-Hancock
half step, HLLC (default) or HLL along each face normal, one global time
step (Courant factor 0.3 on each cell's radius and signal speeds); order 0
is first-order Godunov. Every cell sums over its face list in a fixed
order, so runs are bit-reproducible. Two update loops: per face (a face
flux array, then a per-cell gather; default at second order) or per cell
(each face flux computed from both sides; default at first order).

```text
hydro_sedov <side> <jittered|uniform> [--order=0|1] [--update=cell|face] [--limiter=face|cell|none]
            [--riemann=hllc|hll] [--moving] [--bulk=km/s] [--out=dir] [--snapshots=10] [--steps=max] [--input-order] [--memory]
sbatch -o <logs>/hydro-%j.out scripts/hydro_a100.sh build-cuda <out> <mode> (see the script)
```

`hydro_sedov` (needs HDF5; on Raven `module load hdf5-serial`) runs a
Sedov blast (1e51 erg, n = 1 cm^-3, 50 pc box, 1e4 yr) and writes HDF5
snapshots. Measured on one A100 (jobs 31699864, 31699865, 31700852), double,
jittered points in Morton order, median of 40 steps:

| Million cell updates per second | 2,097,152 cells | 16,777,216 cells |
|---|---|---|
| moving mesh, MUSCL-Hancock HLLC, per-face loop (mesh update and face lists included) | **55** | 55 (whole run with the per-cell loop, which fits in memory: 43) |
| static mesh, MUSCL-Hancock HLLC, per-face loop | **112** | 105 |
| static mesh, MUSCL-Hancock HLL, per-face loop | 116 | 114 |
| static mesh, first order HLLC, per-cell loop | 288 | 290 |
| device memory, mesh + hydro (static, per-face, second order) | 2370 B/cell | 2371 B/cell (41.2 GB in use) |
| device memory, moving mesh + hydro (per-face) | 2418 B/cell | whole run: out of memory (job 31701027); per-cell loop fits |

Sedov at 128^3 (job 31699863), HLLC: shock radius 0.978 of the
Sedov-Taylor value at 1e4 yr, L1 density error 0.075 rho0 (first order:
0.104); mass and energy conserved to 2e-16. On the moving mesh (jobs
31700850-51): 354 steps instead of 1642, L1 density error 0.058 rho0, the
same numbers with the whole box moving at 500 km/s. Details:
research_notes/hydro_stage1_2026-10-31.md, hydro_stage3_2026-10-31.md.

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
- Periodic box only.

## Repository layout

- `src/neighbour_search/`, `src/voronoi_construct/`: the library.
- `src/test/`: the tests, `support.hpp` (inputs, references, the
  comparison with an exact rebuild), `check.hpp`.
- `src/hydro/`: the hydro (header-only). `src/bench/`: `voronoi_bench`, `hydro_sedov`.
- `scripts/`: `lint_cpp.sh`, `bench_a100.sh`, `hydro_a100.sh`, `hydro_sedov_compare.py`.
- `research_notes/`, `reports/`: status notes and surveys.

CI (`.github/workflows/checks.yml`) runs clang-tidy and the CPU tests in
ordinary and sanitizer builds against a pinned Serial Kokkos 5.0.2; it does
not run GPU jobs.
