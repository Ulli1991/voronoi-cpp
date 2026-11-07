# voronoi-cpp

A C++ library that builds exact Voronoi meshes of moving particles on GPUs. You hand it particle positions in a periodic box and get back the faces, cell volumes and centroids a finite-volume code needs. When the particles move, it repairs the mesh instead of rebuilding it, and the result stays exact. It is written in Kokkos: CUDA (validated on an A100 40 GB), plus OpenMP and Serial builds that run the same code for checks. `voronoi::Mesh<D, Exec>` is complete for D = 3, 2 and 1.

The topology is exact. The cell geometry is not yet right in every case: two open bugs give wrong volumes with no error, see [Limitations](#limitations-and-open-items).

The hydro lives in [luna-cpp](https://github.com/Ulli1991/luna-cpp), which uses this library as a submodule. The old copy in `src/hydro/` still builds here as a bit check of mesh changes and goes when the split is finished ([#16](https://github.com/Ulli1991/voronoi-cpp/issues/16)).

BSD 3-Clause, see [LICENSE](LICENSE).

## Install and build

Dependencies:

- CMake 3.16 or later and a C++17 compiler (gcc 14/15 tested).
- Kokkos 5.0 (5.0.2 in CI, 5.0.99 on Raven) with CUDA, OpenMP or Serial. CUDA uses `Kokkos_ARCH_AMPERE80` on an A100, Kokkos' `nvcc_wrapper`, CUDA 13.0, and `Kokkos_ENABLE_CUDA_LAMBDA` (the default in 5.x).
- Nothing else: the library is header-only (`src/neighbour_search/`, `src/voronoi_construct/`) and CMake exports it as `voronoi_library`.

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

Elsewhere, point `CMAKE_PREFIX_PATH` at a Kokkos installation with the backend you want. On Raven, build with at most `-j4` on the login node, or inside a Slurm job, and run GPU work through Slurm (`scripts/bench_a100.sh`).

### Compiler flags in use

- Release: `-O3 -DNDEBUG` (CMake's default), C++17 required (Kokkos sets `-std=c++20`), warnings `-Wall -Wextra -Wpedantic` on the programs.
- CUDA: `nvcc_wrapper` with `-extended-lambda -arch=sm_80`. The warp clipping kernel uses `Kokkos::LaunchBounds<128, 5>`, which caps it at 96 registers; 18 KB of shared memory per block.
- OpenMP: `-fopenmp` from Kokkos. The project sets no `-march`; `-DCMAKE_CXX_FLAGS=-march=icelake-server` is worth adding on Raven's CPUs.
- `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` (always on in `CMakeLists.txt`) writes the database clang-tidy needs: `bash scripts/lint_cpp.sh build-omp` (Raven: `module load clang/20`).
- `-DVORONOI_SANITIZERS=ON` adds address and undefined-behaviour sanitizers to a host build.

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

What a caller can rely on:

- Particles are the caller's indices. Ties between cospherical particles are broken by index (symbolic perturbation), at the start and in every repair. A permuted input gives the permuted mesh.
- Positions are snapped to the grid of the exact predicates, 2³⁰ steps across the longest side, the same step on every axis; topology and geometry both come from the snapped positions. A side that is not a whole number of steps is rounded to one (relative change below 2⁻³¹ longest / side; none for sides in power-of-two ratios such as 2 x 2 x 1/16). `sides()` gives the box used; `box()` is a cube's side.
- Every cell must be certified within one periodic image: its radius must stay below about 3/16 of the shortest side (in practice about 8 or more particles across it; a slab of 16 cells is fine). Otherwise `update()` throws, saying so. It can throw below 3/16 too ([#31](https://github.com/Ulli1991/voronoi-cpp/issues/31)).
- `update()` repairs the mesh; the first call, or a change of n, builds it. If the device repair leaves stuck triangles, it finishes on the host or rebuilds on the device, whichever is cheaper (`voronoi::Fallback`).
- Faces are stored once, by the lower id, so a flux applied with opposite signs conserves exactly. Cell volumes after `update()` are summed with atomics. `update_cells()` sums volumes and centroids over each cell's faces in a fixed order, giving the same bits every run, and builds the per-cell face lists for gathers.
- `clip(x, cells)` rebuilds the listed cells by clipping, independent of the persistent mesh, and returns each face once (a local mode for substeps of active cells). It can return a wrong volume ([#30](https://github.com/Ulli1991/voronoi-cpp/issues/30)).
- `rebuild(x)` builds the exact mesh from scratch. `scratch(bytes)` lends device memory between updates.
- The dimension is a template parameter, `voronoi::Mesh<D, Exec>` with `exact_predicates::BoxN<D>`; `Box` is `BoxN<3>` (research_notes/dimension_design_2026-11-02.md).
- `voronoi::Mesh<2>` has the contract of `Mesh<3>` on triangles (`voronoi_mesh_2d.hpp`), with two differences. `volume()` is current after `update_cells()` only. There are always exactly 3n faces, numbered for good: a flip gives its face new particles, and a face can have zero length on a lattice. A repaired mesh equals a fresh build bit for bit.

## How it works

![Control flow of Mesh::update](docs/figures/flow_mesh.png)

What `update()` does with new positions. Normally the mesh is repaired by exact flips on the GPU, which is the green path. If a repair can't finish, or the first time, the mesh is rebuilt from scratch by clipping, the blue path. Input that can't give an exact mesh is refused with an error.

Source map, in the order `Mesh<3>::update` calls it (stage numbers as in the file headers; each file's header also says who calls it and what it feeds):

| Stage | Header | Job |
|---|---|---|
| all | `exact_predicates.hpp` | the periodic `Box`, snapping, exact orientation and in-sphere signs (filtered double, then 128/256-bit integers), symbolic perturbation by particle id |
| 0 | `voronoi_mesh.hpp` | `voronoi::Mesh`: the entry point, `update`, `rebuild`, `update_cells`, `clip`; the pipeline behind one interface |
| 1 | snapping | `Mesh::snap` in `voronoi_mesh.hpp`: positions to the 2³⁰ grid |
| 2 | `neighbor_search.hpp` | `CellList`: particles sorted along a Morton curve, an implicit octree over a dense base grid, a grid level per particle fitted to the local density |
| 3 | `shell_voronoi.hpp` | clipping with a thread per cell: the walk, rounded and exact cells |
| 3 | `shell_voronoi_warp.hpp` | clipping with a warp per cell on CUDA, writing faces or face rings |
| 3 | `tet_start.hpp` | the exact start: exact check of `from_rings`, exact clipping of cells around errors |
| 4a | `tet_repair.hpp` | per-step repair by 2-3 / 3-2 flips on the device |
| 4b | `tet_flips.hpp` | host fallback of the repair (local Bowyer-Watson rebuilds) |
| 5 | `tet_mesh.hpp` | persistent tetrahedra: `from_rings`, circumcentres, face rings, faces |
| 5b | `tet_cells.hpp` | per-cell face lists, volumes, centroids and gradient weights (`update_cells`) |

The other dimensions follow the same order. 2D: `voronoi_mesh_2d.hpp` with `polygon_clip.hpp` (clipping), `tri_mesh.hpp` (start and faces), `tri_repair.hpp` (device repair), `tri_finish.hpp` (host fallback) and `exact_predicates_2d.hpp`. 1D: `sorted_mesh.hpp`, `voronoi::Mesh<1>`: generators sorted on a 2⁵² grid, ties by id, periodic or with walls; `insert()` and `remove()` change the line between updates and report what changed.

Clipping builds each cell by walking the grid cells around its generator, nearest first, and clipping with every particle within twice the cell's radius; the walk ends when no further particle can cut, which certifies the cell. The exact start turns the resulting face rings into tetrahedra and checks that together they are the Delaunay tetrahedralization, rebuilding any cell that fails. Each step then snaps the positions and repairs the tetrahedra whose local Delaunay test may fail by flips on the device, every flip decided by exact predicates. Triangles that stay stuck are rebuilt on the host, or the whole mesh is rebuilt.

## Measured performance

One A100 40 GB, double precision, periodic unit box. Rates are million cells per second of GPU time. Jobs, commits and further setup are in [docs/measurements.md](docs/measurements.md). The 256³ column is from older code, which is faster there now.

| Million cells per second | 2,097,152 cells (128³) | 16,777,216 cells (256³) |
|---|---|---|
| Full rebuild (warp clipping), jittered | 39.4 | 34.8 |
| Full rebuild, uniform random | 19.7 | |
| Full rebuild, clustered (Gaussian blobs, ~1:500) | 15.4 | |
| Full rebuild, halo (1:10⁴ core to edge) | 14.1 | |
| Full rebuild, lattice / wave | 49.2 / 37.9 | |
| Exact start (warp with rings + `tet_start::build`), jittered | 17.7 | 11.7 |
| Moving-mesh update, 0.03 spacings per step | 208 | 291 |
| Moving-mesh update, 0.1 spacings per step | 119 | 188 |
| Moving-mesh update, 0.3 spacings per step | 57 | 98 |
| `update_cells()` alone, 0.03 / 0.1 / 0.3 spacings | 574 / 448 / 436 | 666 / 543 / 458 |
| Update + `update_cells()`, 0.03 / 0.1 / 0.3 spacings (bit-reproducible) | 154 / 94 / 51 | 202 / 140 / 81 |

Spacings per step is how far a smooth flow moves the points per step, in mean interparticle spacings. 0.03 is a typical hydro step and 0.3 a very large one, with more flips per step and lower throughput. The six inputs (jittered lattice, uniform random, clustered, halo, lattice, wave) are described in the measurements file.

Device memory: a plain warp rebuild takes 701 B/cell (656 at 256³); the peak during start and steps, with face lists, is 2028 B/cell (2007 at 256³, 34.8 GB in use).

A thin slab (Lz = 16 spacings, up to 16.8M cells) rebuilds at the cube's rate, 39.9 M cells/s for jittered input.

### Long runs

The rates above are early steps. On a smooth flow where every cell moves, the mesh loses its memory order and the step slows from 118 M cells/s at step 100 to 59 at step 500 (128³, 0.1 spacings, [#34](https://github.com/Ulli1991/voronoi-cpp/issues/34)). The mesh has no re-sort yet. Flows with little bulk motion do not show it: on Sedov the loss is at most 4.5 % over a run.

### GPU against a full CPU node, same code

The same code built with the OpenMP backend on one Raven CPU node (2x Xeon Platinum 8360Y, 72 cores), against one A100, at 2,097,152 cells in double. Million cells (or cell updates) per second:

| | 72-core CPU node | A100 | A100 / CPU |
|---|---|---|---|
| Full rebuild, jittered / uniform / clustered | 4.1 / 2.8 / 2.0 | 39.9 / 19.9 / 15.5 | 9.7 / 7.0 / 7.9 |
| Moving-mesh update, 0.03 / 0.3 spacings | 11.5 / 1.7 | 202 / 57 | 17.5 / 33 |
| `update_cells()` | 43 | 343 | 7.9 |

Every phase gains more than the A100's 3.9 times the node's memory bandwidth. The CPU build is the generic Kokkos path with no hand vectorisation, so a CPU-tuned code would narrow the gap.

### 2D and 1D

`voronoi::Mesh<2>` on one A100 at 1305 MHz, double, 0.03 spacings per step:

| M cells/s | jittered | random | clustered | lattice |
|---|---|---|---|---|
| Build, 2²⁰ cells | 100 | 50 | 44 | 115 |
| Step (repair, faces, cell lists), 2²⁰ | 765 | 791 | 789 | 749 |
| Build, 2²⁴ cells | 134 | 58 | 52 | 168 |
| Step, 2²⁴ | 1056 | 1059 | 1061 | 1023 |

Jittered at 0.1 / 0.3 spacings: 770 / 744 at 2²⁰, 1024 / 1015 at 2²⁴. Peak device memory is 537 to 574 B/cell.

`voronoi::Mesh<1>`, mesh phases of a moving Sod run: 18,905 M cells/s at 2²⁰ and 39,351 at 2²⁴ (1095 MHz, older than this branch).

## Tests

Twelve CTest programs specify the library and, until the split ends, the hydro copy. Oracles, run times and what PASS means are in [TESTING.md](TESTING.md).

| Test | Specifies |
|---|---|
| `exact_predicates` | signs equal an independent integer evaluation; ties; periodic images; snapping for any box |
| `clipping` | cells certified, closing and on the bisectors (brute force); thread, warp and exact kernels agree |
| `mesh` | `from_rings` against a host construction; adjacency, exact check, volumes |
| `start` | the start is the Delaunay tetrahedralization (brute-force oracle) for 13 layouts and several boxes |
| `repair` | after every step of a flow the mesh equals an exact rebuild |
| `voronoi_mesh` | the hydro contract of `voronoi::Mesh` |
| `mesh_cases.*` | 3D special cases against an oracle that shares no code with the library: coincident points, extreme contrast, thin boxes, high valence, large motion, fallbacks, bits across backends |
| `distortion` | flows that wind the mesh up, no hydro: shear, vortex, Taylor-Green, turbulence, collapse, expansion |
| `mesh2d` | the 2D mesh against independent oracles; a repaired mesh equals a fresh build bit for bit |
| `hydro` | the hydro copy: HLLC and HLL fluxes, conservation, bit-identical runs, moving-mesh invariances |

```sh
ctest --test-dir build-cuda --output-on-failure         # A100: ~5 minutes
OMP_NUM_THREADS=1 ctest --test-dir build-omp            # one CPU core
sbatch -o build-cuda/bench-%j.out scripts/bench_a100.sh build-cuda   # ctest + A100 checks
```

`voronoi_bench` (`src/bench/`) times and checks the pipeline:

```text
voronoi_bench rebuild <n> <distribution> [variants=thread,exact,warp] [repeats=5] [--width=f] [--no-check] [--memory] [--box=Lx,Ly,Lz]
voronoi_bench step <n> <distribution> <displacement> [steps=10] [--verify] [--debug] [--fallback=host|rebuild] [--memory] [--kernels] [--touched] [--checksum] [--box=Lx,Ly,Lz]
```

`--box` (default 1,1,1) sets the periodic box; `--box=2,2,0.0625` with 4194304 jittered points is a 512 x 512 x 16 slab. Distributions: uniform, jittered, lattice, wave, clustered, halo, or `<sc|bcc|fcc|shells|planes|pairs>@<perturbation in spacings>`.

## Limitations and open items

Open bugs, found by fuzzing the mesh against an oracle that shares no code with it:

- **Wrong cell volumes with no error, up to 9.4 %,** on valid input in boxes whose grid step is not a power of two ([#29](https://github.com/Ulli1991/voronoi-cpp/issues/29)). The topology is right and the volumes still sum to the box, so the check in `update()` does not catch it. How common it is on realistic input is not known.
- **`clip()` returns a volume 5 % low with status ok,** in a unit cube ([#30](https://github.com/Ulli1991/voronoi-cpp/issues/30)). The cause is not known. The persistent mesh was exact in those runs.
- **Non-finite or negative geometry with no error** at extreme box sizes ([#32](https://github.com/Ulli1991/voronoi-cpp/issues/32)). The fuzz drew sides from 1e-140 to 1e141; the usable range is not measured or stated yet.
- Valid input can be refused below the stated 3/16 limit, with largest radii from 0.112 of the shortest side ([#31](https://github.com/Ulli1991/voronoi-cpp/issues/31)).
- `Fallback::host` has no bound on its work: minutes on hundreds of particles ([#33](https://github.com/Ulli1991/voronoi-cpp/issues/33)).
- `Mesh<2>::update` can hang: the 2D repair flips the same entries forever and never returns ([#36](https://github.com/Ulli1991/voronoi-cpp/issues/36)).
- `tet_flips::rebuild` reads particle positions at index -1; the effect on meshes is not known ([#35](https://github.com/Ulli1991/voronoi-cpp/issues/35)).
- A periodic `Mesh<1>` with all particles on one grid step gives volumes summing to 0, silently ([#37](https://github.com/Ulli1991/voronoi-cpp/issues/37)).

Other limits:

- Bit-reproducible moving runs cost 2.5 to 4.7 % on the mesh step alone.
- CUDA and Serial give different last bits of volumes and centroids ([#27](https://github.com/Ulli1991/voronoi-cpp/issues/27)).
- On CUDA a build in a cube of side 3.086e24 aborts with an illegal device address ([#26](https://github.com/Ulli1991/voronoi-cpp/issues/26), open); the other sides tested and OpenMP do not.
- Clipping alone is not watertight on uniform random input at 2.1M cells; the exact start makes the mesh exact.
- Particles on one grid position (grid step box / 2³⁰) have no Voronoi cell. `update()` refuses them on every path with a `std::runtime_error` naming a pair by id, and does not merge them. Finding out takes a rebuild, so such a step costs one. After a refusal there is no mesh (no faces, no particles) and the next `update()` builds from scratch. Zero particles are a `std::invalid_argument` and change nothing.
- `update()` does not look at its input: a NaN or infinite coordinate is snapped to some grid position, with no error.
- A cell may hold up to 16000 planes and 32000 vertices during its exact clipping; past 128 planes or 256 vertices it is built on the host. Past those limits `update()` refuses, naming them. Particles around a void cost O(k²) at the start: about 1 s at k = 2000, 2 minutes at k = 20000 with a particle in the middle (8 CPU threads).
- A step of order the box (random kicks of 0.6 box) can end the device repair with nothing stuck on a mesh that covers the torus many times. `update()` sees it in the total volume of the cells, off the box by more than 1e-6, and rebuilds (`Report::overlapped`). That check is on the volume alone: a wrong mesh whose volumes still add up to the box within 1e-6 would pass it.
- `Fallback::cheapest` takes the host path only up to n / 200000 stuck entries (10 at 2.1M cells, none below 200000); the rest is a rebuild, which needs the memory of a start.
- Geometry rounds with the positions: with a power-of-two step, the relative volume of a cell h grid steps wide is good to about 4e-8 / h (other steps: [#29](https://github.com/Ulli1991/voronoi-cpp/issues/29)). It is exact down to clumps of 512 steps (error 8e-10), and particles one grid step apart are still exact.
- The device repair breaks down near one spacing of motion per step (stuck triangles, host or rebuild fallback); substepping helps.
- Ghost particles (multi-GPU), repair limited to active regions, and refinement and derefinement are not implemented yet; `update()` takes an `active` list for later.
- Periodic boxes only (any three sides; see the certification limit above for thin slabs). Only 1D has walls.
- `voronoi::Mesh<2>` refuses as `Mesh<3>` does: particles on one grid position by name, cells out of certified reach, no particles. A refused update leaves no mesh and the next builds. Its repair is checked by the sum of the signed areas (`Report::overlapped`).
- Memory limits 3D on one A100 to about 256³ today: 2319 B/cell with the hydro, of which the start peak alone is 1728. The target is 384³, about 720 B/cell ([#28](https://github.com/Ulli1991/voronoi-cpp/issues/28)).

## Roadmap

Done: exact 3D, 2D and 1D meshes, incremental cell lists, special-case tests, the refusals above.

1. Fix the wrong volumes ([#29](https://github.com/Ulli1991/voronoi-cpp/issues/29), [#30](https://github.com/Ulli1991/voronoi-cpp/issues/30)) and the 2D hang ([#36](https://github.com/Ulli1991/voronoi-cpp/issues/36)), then state the box range ([#32](https://github.com/Ulli1991/voronoi-cpp/issues/32)), the reach limit ([#31](https://github.com/Ulli1991/voronoi-cpp/issues/31)) and a bound on the host path ([#33](https://github.com/Ulli1991/voronoi-cpp/issues/33)).
2. Cut memory to 384³ on one A100 ([#28](https://github.com/Ulli1991/voronoi-cpp/issues/28)): float faces, `tet_centre`, repair lists, start peak.
3. Point insertion and removal ([#25](https://github.com/Ulli1991/voronoi-cpp/issues/25)): done in 1D (`Mesh<1>::insert`, `remove`), then 2D and 3D.
4. Mesh speed: a re-sort when the order has decayed ([#34](https://github.com/Ulli1991/voronoi-cpp/issues/34)), faces numbered by owner, then repair rounds and the fixed cost per step.
5. Clustered rebuild tuning.
6. Multi-GPU: overlapped against blocking communication across two nodes first.
7. Refinement and derefinement in 3D.

## Repository layout

- `src/neighbour_search/`, `src/voronoi_construct/`: the library.
- `src/test/`: the tests and their support code; `mesh_cases_test` has its groups in `src/test/mesh_cases/` (one file per group, shared helpers in `common.hpp`). `src/bench/`: `voronoi_bench`, `mesh2d_bench`.
- `src/hydro/`, `hydro_sedov`, `hydro_1d`: the copy that goes ([#16](https://github.com/Ulli1991/voronoi-cpp/issues/16)).
- `scripts/`: lint and A100 benchmark and test scripts.
- `research_notes/`, `reports/`: status notes and surveys.

CI (`.github/workflows/checks.yml`) checks the comment rules and the formatting of changed lines (`.clang-format`, `scripts/check_format.py`), runs clang-tidy and the CPU tests in ordinary and sanitizer builds against a pinned Serial Kokkos 5.0.2. It does not run GPU jobs.
