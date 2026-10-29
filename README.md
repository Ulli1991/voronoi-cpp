# voronoi-cpp

Kokkos C++17 library for exact local 3D Voronoi meshes in a periodic box on GPUs
(CUDA validated on A100; OpenMP/Serial builds for checks). It is the geometry
base for a planned moving-mesh finite-volume (MFV/MFM) hydro solver. Each face
is stored once per particle pair, by the cell with the lower slot index (its
owner), so a flux computed on that face and applied with opposite signs
conserves exactly. The hydro solver itself is not started; this repository
builds and maintains the mesh. Multi-GPU execution is future work.

## Architecture

**(a) Full rebuild by clipping** (`shell_voronoi*.hpp`, search in
`src/neighbour_search/neighbor_search.hpp`). `DenseCellSearch` builds a cell
list with about one particle per grid cell (optionally Morton-ordered cells).
Each cell is built by walking rings of grid cells around its generator and
clipping a polyhedron with every particle that can still cut it. A particle at
distance d cuts only if d < 2R (R is the largest generator-to-vertex distance),
and the walk stops at the first ring entirely beyond 2R; this security radius
certifies the cell. Output is lean: volume and centroid per cell, faces stored
once by their owner. Slots, positions and cell offsets follow the search order;
`search.order(slot)` is the original particle ID.

| Kernel | File | Notes |
|---|---|---|
| `shell_voronoi::construct` | `shell_voronoi.hpp` | thread-per-cell reference and fallback |
| `construct_lean` | `shell_voronoi_lean.hpp` | thread-per-cell, compact polyhedron in shared memory (about 1.2 KB/cell float, 1.8 KB double) |
| `construct_warp` | `shell_voronoi_warp.hpp` | warp-per-cell, default; polyhedron in shared memory, lane l owns vertices/planes l and l+32; ring particles tested 32 at a time and clipped nearest-first after a warp sort |

Cells that overflow the kernel capacities are rebuilt in a big-capacity
overflow pass (warp falls back to the thread kernel; on non-CUDA backends
`construct_warp` calls that kernel directly).

**(b) Persistent Delaunay tetrahedralization** (`tet_mesh.hpp`,
`tet_repair.hpp`, `tet_flips.hpp`, `face_geometry.hpp`,
`exact_predicates.hpp`). After a rebuild, `tet_mesh::from_rings` builds the
tetrahedra and adjacency on the device from the clipping rings. Each tetrahedron
is a Voronoi vertex (its circumcentre), and a Voronoi face is the ring of
circumcentres around an edge. Per step only three things are needed:
circumcentres, face geometry and volumes from them (`face_geometry.hpp`, one
thread per face), and a local Delaunay check with repair where it fails.
`tet_repair` repairs by bistellar flips on the device: failing triangles are
planned as 2->3 or 3->2 flips, plans vote on the tetrahedra they rewrite, the
winners are applied and losers retry. Triangles no flip can repair, and
inverted tetrahedra, are returned as stuck for `tet_flips`, the host
local-rebuild fallback. Every decision uses `exact_predicates.hpp`: positions
snapped to a 2^30 grid, a filtered double evaluation with an error bound, and
an exact fallback (128-bit integer orientation, 256-bit in-sphere), with
symbolic perturbation for ties so all callers agree on one triangulation.

## Measured performance

One A100 40 GB, double precision, 2,097,152 cells (128^3), periodic unit box.
Rates in million cells per second. The gpudev node shares its GPU, so compare
variants only within one job; numbers below are best-node values.

| Stage | Rate |
|---|---|
| Full rebuild, warp kernel, jittered | 18.7 |
| Full rebuild, warp kernel, uniform random | ~10.4 |
| Full rebuild, warp kernel, clustered | ~0.9 |
| Full rebuild, lean kernel, jittered | 10.4 |
| Tetrahedra from rings (once after a rebuild) | 40 |

Per step at 0.03 spacings of motion (jittered): screened repair 5.8-6.6 ms,
circumcentres 0.91 ms, faces 3.45 ms, about 10.4 ms in total, i.e. about
200 M cells/s. The face-list rebuild after flips is not implemented yet and is
not included.

Comparisons on the same A100 (research_notes/arepo_wd_gpu_benchmark.md and
the status notes):

| Code | Mesh | Full step |
|---|---|---|
| this work | 18.7 M cells/s rebuild; ~200 M cells/s persistent step | no hydro yet |
| PROTEUS | 1.6-6 M cells/s | Sedov, 5 M updates/s |
| AREPO-WD GPU per-cell builder | 1.4-1.6 M cells/s | Sedov, 0.59 M updates/s |
| LUNA (fixed grid) | n/a | 1,100 M updates/s |

Nsight on the 2.1M jittered case shows the rebuild is divergence-bound, not
memory-bound (DRAM <1 %, L2 hit 99 %). Morton cell order, coarser cell lists,
fast-math and octant culling gave no gain and were reverted or left off.
Details: `research_notes/status_2026-10-29.md`.

## Known limitations and open items

- Clustered input is 16-20x slower than jittered: a single cell-list size cannot
  fit clusters and voids (a multi-level cell list is planned).
- Clipping alone is not watertight on uniform random input in double (some face
  pairs inconsistent); the exact tet repair fixes it.
- GPU repair breaks down near one spacing of motion per step; it needs
  substepping or a rebuild guard.
- Near-degenerate simple-cubic lattice (sc@1e-6): the local rebuild fallback
  often fails (hundreds to thousands of unrepaired triangles per step).
- The `shells@1e-9` degenerate test timed out.
- Face-list rebuild after flips is not done.
- The float variants and clustered or degenerate inputs still fail some checks
  of `shell_voronoi_check`; only double on uniform/jittered/lattice is in ctest.
- Periodic box only. MFV/MFM hydro is not started.

## Build and run

Raven, CUDA build:

```sh
module load gcc/14 cuda/13.0 cmake
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda \
  -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper
cmake --build build-cuda -j
```

OpenMP build: `module load gcc/15`, Kokkos `/u/uli/kokkos-gcc15`, same cmake
call with a different build directory. Elsewhere, point `CMAKE_PREFIX_PATH` at
a Kokkos installation with the desired backend. Run GPU jobs through Slurm
(`scripts/` holds example job scripts; submit from the repository root).

Tests: `ctest --test-dir build-cuda --output-on-failure`. This includes the
older table-pipeline checks and `shell_voronoi_check 4096 {uniform,jittered,lattice}
double 1`.

`shell_voronoi_check` validates and times the construction:

```text
shell_voronoi_check [n=32768] [uniform|jittered|lattice|wave|clustered|<layout>@<eps>]
                    [variants=all] [repeats=5] [width_factor=1] [displacement=0] [mode]
```

`variants` selects by substring of thread-/warp-/lean- with -double/-float.
`<layout>` is one of the degenerate layouts sc, bcc, fcc, shells, planes, pairs
with perturbation `eps` in spacings. Besides variant timing it has modes
`faces`, `topology`, `evolve`, `tets`, `cullstats` (CPU replay of cheaper
candidate bounds); in `tets` mode a trailing `flips` (host) or `gpuflips`
(device) repairs after moving particles by `displacement` spacings. Checks:
every cell certified, positive volumes summing to the box, closed cells, every
face seen from both sides, and face centroids equidistant from their two
generators with no closer generator.

```sh
# full rebuild, warp kernel, double, 3 repeats
shell_voronoi_check 2097152 jittered warp-double 3

# persistent tets: displace by 0.03 spacings, GPU flip repair, timing only
GPUFLIPS_FAST=1 GPUFLIPS_SCREEN=1 \
  shell_voronoi_check 2097152 jittered tets 1 1 0.03 gpuflips
```

Environment switches: `SHELL_MORTON` (Morton cell order), `SHELL_PHASES` (warp
phase timers, needs the compile-time `SHELL_WARP_PHASES`), `SHELL_WARP_BLOCKS`
(non-persistent launch), `SHELL_LEAN_LEVEL` / `SHELL_LEAN_THREADS` (lean-kernel
scratch level and block size), `GPUFLIPS_FAST` (no host check, timing only),
`GPUFLIPS_SCREEN` (screen for failing triangles before repair), `TETS_DEBUG`
(print mismatch details). Compile-time: `SHELL_WARP_TEAM` (threads per cell, 32,
16 or 8; default 32), `SHELL_WARP_MIN_BLOCKS` (register cap, default 5 blocks/SM),
`SHELL_WARP_PHASES`.

## Repository layout

- `src/neighbour_search/`: `neighbor_search.hpp` (dense cell grid, radius
  queries), `fixed_point_positions.hpp`.
- `src/voronoi_construct/`: `shell_voronoi*.hpp` (clipping), `tet_mesh.hpp`,
  `tet_repair.hpp`, `tet_flips.hpp`, `face_geometry.hpp`, `exact_predicates.hpp`
  (persistent mesh), and the earlier table pipeline (`local_voronoi*.hpp`).
- `src/test/`: `shell_voronoi_check.cpp` and the older checks and benchmarks.
- `research_notes/`: status notes (`status_2026-10-29.md`), AREPO/PROTEUS/
  GIZMO/SPH-EXA surveys, Kokkos CUDA audit, and literature reports.
- `docs/`: design notes for the earlier table pipeline (GPU optimization plan,
  memory budget, layout experiments).
- `scripts/`, `python/`: Slurm jobs, comparison and plotting tools.

CMake exposes `voronoi_library` with both include directories.

## Earlier table-based pipeline

The first pipeline (`local_voronoi*.hpp`, `table_mesh_check`,
`construction_scaling`, `mesh_benchmark`) builds each cell from a precomputed
neighbor table with fixed-capacity plane workspaces and floating-point
tolerances, not exact predicates. It remains in the repository with its tests
and its measurements (about 4.2-4.8 million particles/s combined search plus
reconstruction at 195^3 on an A100). It is superseded for speed by the shell
kernels above. See `HANDOFF.md`, `docs/`, and
`docs/README-before-offline-scaling.md` for its details.

## Linting and continuous integration

GitHub Actions (`.github/workflows/checks.yml`) runs Ruff lint/format checks,
clang-tidy, and CPU correctness tests in ordinary and address/undefined-behavior
sanitizer builds against a pinned Serial Kokkos 5.0.2. It does not run GPU jobs.

```sh
python3 -m pip install ruff==0.11.13
ruff check python
ruff format --check python
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/kokkos -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j 2
bash scripts/lint_cpp.sh build
ctest --test-dir build --output-on-failure
```

Add `-DVORONOI_SANITIZERS=ON` in a separate host build for sanitizers.
