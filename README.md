# voronoi-cpp

Kokkos C++ library for exact Voronoi meshes of moving particles in a periodic
box (any sides) on GPUs: CUDA, validated on an A100 40 GB; OpenMP and Serial
builds run the same code for checks. A solver hands over particle positions
and gets back faces (owner, neighbour, area vector, centroid), cell volumes and
centroids, and per-cell face lists; the mesh is kept exact as the particles
move. `voronoi::Mesh<D, Exec>` is complete for D = 3, 2 and 1.

The hydro moved to [luna-cpp](https://github.com/Ulli1991/luna-cpp), which uses
this library as a submodule through `voronoi::Mesh` alone. The copy in
`src/hydro/` still builds here, as the bit check of mesh changes, and goes when
the split is finished (#16); nothing is developed in it.

BSD 3-Clause, see [LICENSE](LICENSE).

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

voronoi::Mesh<3> mesh(Kokkos::DefaultExecutionSpace{}, box);   // a cube of side box, or
// voronoi::Mesh<3> mesh(exec, exact_predicates::Box(lx, ly, lz)); // any three sides
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
- Positions are snapped to the grid of the exact predicates inside, 2³⁰
  steps across the longest side, the same step on every axis (so the
  integer in-sphere test is the test of the snapped points); topology and
  geometry both come from the snapped positions. A side that is not a
  whole number of steps is rounded to one (relative change below 2⁻³¹
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
- The dimension is a template parameter, `voronoi::Mesh<D, Exec>`
  (`exact_predicates::BoxN<D>`); D = 3, 2 and 1 are implemented
  (research_notes/dimension_design_2026-11-02.md). `Box` is `BoxN<3>`.
- `voronoi::Mesh<2>` has the contract of `Mesh<3>`, on triangles
  (`voronoi_mesh_2d.hpp`, `exact_predicates::BoxN<2>`), with two differences:
  `volume()` is current after `update_cells()` only, and there are always exactly
  3n faces, numbered for good (a flip gives its face new particles, and a face can
  have zero length on a lattice). The geometry is a function of the positions
  alone: a repaired mesh equals a fresh build bit for bit.

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
| `sorted_mesh.hpp` | `voronoi::Mesh<1>`: generators sorted on a 2⁵² grid, ties by id, periodic or with walls; the order is checked every step and sorted only when points crossed |

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

One A100 40 GB (Raven gpu1), double precision, periodic unit box. Rates in
million cells per second, cells processed per second of GPU time. Current
at 2.1M cells: job 31952136 (main 16e7769 and this tree, interleaved) for the
rebuild rows, the moving-mesh update and `update_cells()`. The exact start and the
sum row at 2.1M: job 31701509 (commit 2311937). The 256³ column: job 31694457 (commit
f32c909), older than the current code, which is faster there.

| Million cells per second | 2,097,152 cells (128³) | 16,777,216 cells (256³) |
|---|---|---|
| Full rebuild (warp clipping), jittered | **39.4** | 34.8 |
| Full rebuild, uniform random | **19.7** | |
| Full rebuild, clustered (Gaussian blobs, ~1:500) | **15.4** | |
| Full rebuild, halo (1:10⁴ core to edge) | **14.1** | |
| Full rebuild, lattice / wave | **49.2 / 37.9** | |
| Exact start (warp with rings + `tet_start::build`), jittered | 17.7 | 11.7 |
| Moving-mesh update, 0.03 spacings per step | **208** | 291 |
| Moving-mesh update, 0.1 spacings per step | 119 | 188 |
| Moving-mesh update, 0.3 spacings per step | 57 | 98 |
| `update_cells()` alone, 0.03 / 0.1 / 0.3 spacings | 574 / 448 / 436 | 666 / 543 / 458 |
| Moving-mesh update + `update_cells()`, 0.03 / 0.1 / 0.3 spacings (bit-reproducible) | **154** / 94 / 51 | 202 / 140 / 81 |

Inputs (all in the periodic unit box, `src/test/support.hpp`):

| Name | Points | Tests |
|---|---|---|
| jittered | cubic lattice, each point moved randomly by up to ±0.1 lattice spacing per axis | the typical hydro start; nearly regular, ~16 faces per cell |
| uniform random | independent uniform points | unstructured cells of very different size and shape |
| clustered | 8 Gaussian blobs (σ = 0.04) with 80 % of the points, 20 % uniform background | density contrast ~1:500 (blob centre to void) |
| halo | 4 halos with ρ ∝ 1/(r² + c²) to r = 0.2, c = 0.002, 90 % of the points, 10 % background | density contrast ~1:10⁴ (core to edge) |
| lattice | perfect cubic lattice | fully degenerate: 8 points on every circumsphere |
| wave (moved lattice) | the lattice with every point moved by up to 1e-5 spacing | nearly degenerate: ties broken only by tiny offsets |

Moving-mesh update rates are per step of a smooth flow that moves the
points by the given number of mean interparticle spacings per step
(spacing = (box volume / n)^(1/3)): 0.03 spacings is a typical hydro
step, 0.3 a very large one (more flips per step, lower throughput).

Device memory: plain warp rebuild 701 B/cell (656 at 256³); peak during
start and steps, with face lists, 2028 B/cell (2007 at 256³, 34.8 GB in use).
Moving-mesh rows: jobs 31886231, 31886232 (research_notes/mesh_speed_2026-11-04.md).

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
the step at 256³ moves particles by the same number of spacings, but the
flow varies on the box scale, so neighbours shear half as much per step
(half the flips per cell). On jittered, uniform and clustered input at
0.1 and 0.3 spacings no triangle ever sticks; the first step out of a
degenerate start (wave lattice, 2.1M) is repaired on the device alone in
0.81 s (job 31701509; research_notes/shrink_2026-10-31.md), while
icosahedral shells@1e-9 still leave ~8k of 262k stuck (host or rebuild). The per-cell face lists account for most of the step peak above
2fd8b50's 2080 B/cell (about 140 B/cell; not measured separately).

### GPU against a full CPU node, same code

The identical code (commit 57201bc) built with the Kokkos OpenMP backend on
one Raven CPU node (2x Xeon Platinum 8360Y, 72 cores, `OMP_PROC_BIND=spread`,
job 31796079) against one A100 40 GB (jobs 31795767, 31795035), 2,097,152
cells, double. Throughput in million cells (or cell updates) per second.

| | 72-core CPU node | A100 | A100 / CPU |
|---|---|---|---|
| Full rebuild, jittered / uniform / clustered | 4.1 / 2.8 / 2.0 | 39.9 / 19.9 / 15.5 | 9.7 / 7.0 / 7.9 |
| Moving-mesh update, 0.03 / 0.3 spacings | 11.5 / 1.7 | 202 / 57 | 17.5 / 33 |
| `update_cells()` | 43 | 343 | 7.9 |

For scale, the hardware ratios A100 / node are about 3.9 in memory bandwidth
(1.55 TB/s against ~0.4 TB/s) and 1.8 in FP64 peak. Every phase gains more
than the bandwidth ratio: the clipping is latency- and divergence-bound
(DRAM traffic under 1 % on the GPU), the flip repair is many small parallel
rounds with atomics, and even the bandwidth-like hydro passes gain 1.6-3x
beyond bandwidth. The CPU build is the generic Kokkos OpenMP path (per-thread
clipping, no hand vectorisation), so a CPU-tuned code would narrow the gap.

### 2D and 1D

`voronoi::Mesh<2>`, one A100 at 1305 MHz, double, Morton order, 0.03 spacings per step (job 31952137):

| M cells/s | jittered | random | clustered | lattice |
|---|---|---|---|---|
| Build, 2²⁰ cells | 100 | 50 | 44 | 115 |
| Step (repair, faces, cell lists), 2²⁰ | 765 | 791 | 789 | 749 |
| Build, 2²⁴ cells | 134 | 58 | 52 | 168 |
| Step, 2²⁴ | 1056 | 1059 | 1061 | 1023 |

Jittered at 0.1 / 0.3 spacings: 770 / 744 at 2²⁰, 1024 / 1015 at 2²⁴. Peak device
memory 537-574 B/cell (3D: 1998-2028 B/cell at 2.1M, job 31952136). Details: research_notes/mesh_2d_2026-11-05.md.

`voronoi::Mesh<1>`: the mesh phases of a moving Sod run at 1095 MHz (job 31888427, older
than this branch, research_notes/stage1_1d_2026-11-04.md): 18,905 M cells/s at 2²⁰, 39,351 at 2²⁴.

## Tests

Twelve CTest programs specify the library and, until the split ends, the hydro copy; details, oracles, run times and
what PASS means are in [TESTING.md](TESTING.md).

| Test | Specifies |
|---|---|
| `exact_predicates` | signs equal an independent integer evaluation; ties consistent; nearest periodic image; snapping for any box |
| `clipping` | cells certified, closing, consistent and on the bisectors (brute force); thread, warp and exact kernels agree; pools grow; NaN/infinite particles never cut |
| `mesh` | `from_rings` against a host construction; adjacency, edge representatives, exact check, volumes from rings; exact sliver centres |
| `start` | the start is the Delaunay tetrahedralization (brute-force oracle) for 13 layouts and boxes of side 1, 2, 3, a 1:1:0.125 slab and a 1:2:3 box |
| `repair` | after every step of a flow the mesh equals an exact rebuild (lattice, wave, shells, jittered; box 2, a 1:2:3 box, a 1:1:0.125 slab) |
| `voronoi_mesh` | the hydro contract of `voronoi::Mesh` |
| `mesh_cases.*` | 3D special cases against an oracle that shares no code with the library: coincident points, few points, extreme contrast, thin and long boxes, degeneracy in motion, high valence, large motion, fallbacks, predicates on real point sets, bits across backends |
| `distortion` | flows that wind the mesh up, no hydro: shear, vortex, Taylor-Green, turbulence, collapse, expansion, sheets, interpenetrating streams, long thin cells |
| `mesh2d` | the 2D mesh against oracles that share no code with it: predicates against an integer determinant, on six inputs in four boxes empty circumcircles in exact integers, areas summing to the box exactly, every cell clipped in long double against all particles; a repaired mesh equals a fresh build bit for bit, with both fallbacks |
| `hydro` | HLLC and HLL against reference fluxes and under rotation; uniform states stay uniform, conservation to rounding, bit-identical runs, the two update loops agree to rounding on jittered and uniform meshes; moving mesh: uniform state with a bulk velocity, conservation, Galilean invariance of a moving blast |

```sh
ctest --test-dir build-cuda --output-on-failure         # A100: ~5 minutes
OMP_NUM_THREADS=1 ctest --test-dir build-omp            # one CPU core
sbatch -o build-cuda/bench-%j.out scripts/bench_a100.sh build-cuda   # ctest + A100 checks
```

`voronoi_bench` (`src/bench/`) times the pipeline and checks it at scale:

```text
voronoi_bench rebuild <n> <distribution> [variants=thread,exact,warp] [repeats=5] [--width=f] [--no-check] [--memory] [--box=Lx,Ly,Lz]
voronoi_bench step <n> <distribution> <displacement> [steps=10] [--verify] [--debug] [--fallback=host|rebuild] [--memory] [--kernels] [--touched] [--checksum] [--box=Lx,Ly,Lz]
```

`--box` (default 1,1,1) sets the periodic box; lattices get about equal
spacings along every axis (`--box=2,2,0.0625` with 4194304 jittered
points is a 512 x 512 x 16 slab), the other distributions stretch with
the box.

Distributions: uniform, jittered, lattice, wave, clustered, halo, or
`<sc|bcc|fcc|shells|planes|pairs>@<perturbation in spacings>`.

## Limitations and open items

- Bit-reproducible moving runs cost 2.5-4.7 % on the mesh step alone
  (canonical circumcentres and face-ring order).
- CUDA and Serial give different last bits of volumes and centroids (#27).
- Clipping alone is not watertight on uniform random input at 2.1M cells;
  the exact start makes the mesh exact.
- Particles on one grid position (grid step box / 2³⁰) have no Voronoi cell, so
  `update()` refuses them on every path, with a `std::runtime_error` that names
  a pair by id; the library does not merge them. It takes a rebuild to find out,
  so such a step costs one. After a refusal there is no mesh (no faces, no
  particles), and the next `update()` builds from scratch. Zero particles are a
  `std::invalid_argument` and change nothing.
- `update()` does not look at its input: a NaN or infinite coordinate is snapped
  to some grid position, with no error.
- Faces may have any number of corners. A cell may hold up to 16000 planes and
  32000 vertices during its exact clipping: past 128 planes or 256 vertices it is
  built on the host, with the grid and the ring pool copied there and back. Past those limits
  `update()` refuses, naming them. Particles around a void cost O(k²) at the
  start, every cell of the shell clipping all k of them: about 1 s at k = 2000,
  2 minutes at k = 20000 with a particle in the middle (8 CPU threads).
- A step of order the box (random kicks of 0.6 box) can end the device repair
  with nothing stuck on a mesh that covers the torus many times. `update()` sees
  it in the total volume of the cells, off the box by more than 1e-6, and rebuilds
  (`Report::overlapped`). That check is on the volume alone: a wrong mesh whose
  volumes still add up to the box within 1e-6 would pass it.
- On CUDA a build in a cube of side 3.086e24 aborts with an illegal device
  address (#26, open); the other sides tested and OpenMP do not.
- `Fallback::cheapest` takes the host path only up to n / 200000 stuck entries
  (10 at 2.1M cells, none below 200000), the rest is a rebuild, which needs the
  memory of a start.
- Geometry rounds with the positions: the relative volume of a cell h grid
  steps wide is good to about 4e-8 / h. Exact down to clumps of 512 steps
  (error 8e-10), and particles one grid step apart are still exact.
- The device repair breaks down near one spacing of motion per step
  (stuck triangles, host or rebuild fallback); substepping helps.
- Ghost particles (multi-GPU), repair limited to active regions, and
  refinement/derefinement are not implemented yet; `update()` takes an
  `active` list for later.
- Periodic boxes only (any three sides; see the certification limit
  above for thin slabs); only 1D has walls.
- `voronoi::Mesh<2>` refuses as `Mesh<3>` does: particles on one grid position
  by name, cells out of certified reach, no particles; a refused update leaves no
  mesh and the next builds. Its repair is checked by the sum of the signed areas
  (`Report::overlapped`). `volume()` is current after `update_cells()` only.
- Memory limits 3D on one A100 to about 256³ today (#28).

## Roadmap

Done: exact 3D, 2D and 1D meshes, incremental cell lists, special-case tests, the
refusals above.

1. Cut memory to 384³ on one A100 (#28): float faces, `tet_centre`, repair lists, start peak.
2. Point insertion and removal (#25), 1D first.
3. Mesh speed: repair rounds, the fixed cost per step.
4. Clustered rebuild tuning.
5. Multi-GPU: overlapped against blocking communication across two nodes first.
6. Refinement and derefinement in 3D.

## Repository layout

- `src/neighbour_search/`, `src/voronoi_construct/`: the library.
- `src/test/`: the tests, `support.hpp` (inputs, references, the
  comparison with an exact rebuild), `check.hpp`.
- `src/bench/`: `voronoi_bench`, `mesh2d_bench`. `src/hydro/`, `hydro_sedov`, `hydro_1d`: the copy that goes (#16).
- `scripts/`: `lint_cpp.sh`, `bench_a100.sh`, `dim_gate_a100.sh`, `dim2d_a100.sh`, `mesh_tests_a100.sh`, `mesh_speed_a100.sh`.
- `research_notes/`, `reports/`: status notes and surveys.

CI (`.github/workflows/checks.yml`) runs clang-tidy and the CPU tests in
ordinary and sanitizer builds against a pinned Serial Kokkos 5.0.2; it does
not run GPU jobs.
