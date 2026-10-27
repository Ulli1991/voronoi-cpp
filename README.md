# voronoi-cpp

Workspace profiling is available in `local_voronoi_workspace.hpp` through
`profile_workspace(...)`, writing one `WorkspaceUsage` record per row. It runs
separately from production timing; normal calls compile out the counters.
`local_voronoi_retry.hpp` provides `reconstruct_table_retry` and
`reconstruct_hydro_retry`: a 64-plane first pass, then the retained 160-plane
path for rows that overflow, with no extra production buffers or host read.
The ordinary interfaces and their defaults remain available.

For the experimental scale-aware path, use
`PlaneWorkspace<64,RelativeClippingTolerance>` as the retry interface's second
template argument. It uses an arithmetic-scaled plane test and a pivoted
intersection solve. Local stress checks cover scales 1e-8 through 1e8, rotated
slivers up to aspect ratio 1e8, distinct near-coincident sites, and a clustered
periodic cloud with a 4,900-fold radius contrast. An independent long-double
plane-intersection enumeration checks the small cells. This is not an exact
predicate implementation; unresolved degeneracies still need further work.
GPU runtime validation passes (job 31585499): analytic/stress checks and all
120 CPU/GPU, small/full and numerical-path output comparisons pass. At 32,768
particles, the relative/pivoted 64-plane GPU path takes 6.52 ms to reconstruct
and 8.68 ms including search, versus 5.88/8.12 ms for cached legacy/full.
The smaller workspace reduces common-path private stack about 44%; no material
speedup from capacity reduction alone was measured. The clustered-cloud test
certifies 102/128 cells; 26 explicitly lack sufficient reach and are not yet
automatically recovered. These options remain explicit; defaults are unchanged.

The benchmark accepts two further arguments after the distribution:
`... cached random 64 relative` selects the smaller workspace with the new
numerical path. Distributions now include `random` and `gradient` as well as
the existing three. `construction_stress_check` exercises the difficult cases.

An experimental vertex cache is available without changing existing calls:
`reconstruct_table<CachedVertexPositions>(...)` and
`reconstruct_hydro<CachedVertexPositions>(...)`. It reuses each computed vertex
through clipping, compaction, certification and output. The default remains
`RecomputedVertexPositions`, preserving the reference path. Cached positions
add 1,536 bytes of private temporary storage per cell, with no new persistent
buffers. CPU/GPU checks pass; at 32,768 particles cached clipping takes 5.88 ms
on an A100 versus 15.29 ms for recomputation, and total construction takes
8.13 versus 17.58 ms with the parallel neighbor sort. Geometry agrees exactly
between paths in the tested cases. The private stack grows from 7,104 to 8,640
bytes/thread while register use falls; choose the cache with this tradeoff in
mind. NVCC builds can enable `VORONOI_CUDA_RESOURCE_REPORT` to report
construction-kernel register, stack and spill usage.

The extended check syntax is
`table_mesh_check output.csv side insertion|team_bitonic recomputed|cached lattice|disordered|regular`.
Checks include all-particle nearest-site tests on sampled vertices and the
face-centroid divergence identity, outside the timed sections. The analytic
check now covers rotated symmetric/asymmetric cells at five scales.

An experimental parallel nearest-first sort is selectable with
`search.row_sort = neighbor_search::RowSort::team_bitonic`. It sorts each row's
`(distance squared, slot)` keys in team scratch, preserving deterministic ties
and avoiding repeated insertion-sort shifts through global memory. Table
capacities above 2,048 use the retained insertion fallback. The option persists
across capacity growth; insertion remains the default for now.
Update: job 31583755 validated both sorters on CPU/GPU. At 32,768 particles,
the team sort reduces A100 neighbor rebuild from 22.51 to 2.26 ms and full
diagnostic construction from 37.76 to 17.52 ms, with identical geometry.
It remains explicitly selectable; CPU insertion is faster than the host team
network. The detailed results and current construction-only scope are in HANDOFF.md.
`neighbor_sort_check` covers ties, padding and overflow truncation; invoke
`table_mesh_check output.csv 12 team_bitonic` to compare exact row ordering
against insertion and run the geometry checks using the new sort.

`local_voronoi_hydro.hpp` adds `reconstruct_hydro`, a compact table-input path
returning cell volumes, cell centroid offsets, face area vectors and face
centroid offsets. It integrates moments during the same clipping/face traversal
as the diagnostic interface and requires no face-vertex storage. Centroids are
relative to each generator in its periodic image; only successful rows and
faces below their row count are valid. The fixed output footprint is 1,704 bytes
per particle versus 25,608 bytes for the diagnostic interface (excluding neighbor
storage and clipping workspace). CPU/GPU analytic and integration checks pass
through 32,768 particles. Compact hydro reconstruction takes 14.9 ms on an A100
versus 189.4 ms on 18 CPU threads at that size; GPU nearest-first ordering
dominates the pipeline at 21.4 ms. See HANDOFF.md for the measured scope.

Pass an optional `neighbor_search::StageProfile*` to `rebuild` for synchronized
grid key/count, scan, scatter, cell ordering/gather, candidate fill, nearest-first
ordering and overflow-check timings. Profiling introduces per-stage fences;
use ordinary calls to measure full pipeline performance. The extended
`table_mesh_check` reports both profiles and unprofiled diagnostic/hydro timings.

The direct neighbor-table integration is available in `local_voronoi_table.hpp`
through `local_voronoi::reconstruct_table`. Pass the completed dense search's
`positions`, `counts`, `table`, and `radii` views, using nearest-first candidate
ordering. Results and face neighbor IDs use search slot order;
`search.order(slot)` recovers the original particle ID. This path avoids a CSR
copy and preserves the existing reconstruction interface. `table_mesh_check`
is the new end-to-end correctness executable. gpudev CPU/GPU checks pass through
32,768 particles; separate search, reconstruction and pipeline timings are
recorded in HANDOFF.md.
See [HANDOFF.md](HANDOFF.md) for its checks and remaining work. Descriptions of
`mesh_benchmark` below still refer to its existing host-built graph path.

A small Kokkos prototype for reconstructing local 3D Voronoi cells from particle positions and a candidate-neighbor graph. The goal is to test the geometric reconstruction independently before connecting it to a meshless finite-mass or finite-volume solver.

## What is here

Candidate order can affect temporary workspace use. The mesh stress test orders candidates by distance from the cell center, which avoids transient-capacity failures in the tested lattice; a production neighbor-list path should preserve or provide this ordering, or increase and validate the workspace capacities.

`local_voronoi.hpp` clips a temporary cube by the perpendicular-bisector planes associated with each candidate neighbor. For each cell it returns face areas, neighbor IDs, and face vertices. It uses fixed-capacity storage so a device kernel can report overflow or an uncertified cell rather than silently truncating the geometry. A successful row is certified against the supplied search radius. The caller must provide every particle close enough to affect that cell.

`local_voronoi_example` makes a reproducible 128-particle periodic example, uses an all-to-all candidate graph, checks cell closure and pairwise face consistency, and writes CSV files. `plot_mesh.py` renders those files. This is a correctness example, not a scalable neighbor search: the all-to-all graph is only for a tiny demonstration. The reconstruction currently uses double precision; periodic image handling, degenerate configurations, and capacity limits need further validation.

`neighbor_search.hpp` is the radius search intended for production use. It sorts particles into a dense cell grid with a deterministic counting sort, copies them into cell order, and writes each particle's neighbors into a fixed-capacity table with a per-row count. Buffers persist across rebuilds and are reused without zeroing, so a rebuild allocates nothing and needs one host read, the overflow check; on overflow, `rebuild` grows the table and repeats. It supports periodic boxes (radii below half the box), per-particle radii, optional nearest-first row ordering, and, for small particle counts on a GPU, several vector lanes per particle. The rebuild can be captured as one Kokkos graph.

`neighbor_benchmark` times that search and compares it with a sorted-cell search over occupied cells and a median-split BVH whose build is host-side. It uses uniform, jittered-lattice, clustered, and density-gradient point sets whose radii scale with n^(-1/3), so the mean neighbor count is independent of n, and it reports the median of repeated runs after a warm-up. Results are checked against an all-pairs oracle up to 32,768 particles and against the BVH above that. Usage: `neighbor_benchmark [n] [radius_scale] [repeats] [all|dense|timing] [periodic]`; `timing` skips all checks.

`mesh_benchmark` exercises periodic neighbor construction and Voronoi geometry together on a jittered lattice. It checks the candidate graph against an all-pairs periodic search for up to 8,192 particles, then runs the geometry kernel and checks certification, area closure, shared-face consistency, and volume conservation: the cell volumes must sum to the box volume to within 1e-12, and every cell volume must be positive. Its neighbor graph is generated on the host from a structured cell stencil, so it is a geometry stress test rather than a GPU neighbor-search benchmark.

## Build

Build Kokkos with the backend you want to use, then point CMake at that installation. For example, for an existing Kokkos installation:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/kokkos
cmake --build build -j
```

Run the small geometry example and neighbor-search benchmark:

```sh
./build/local_voronoi_example
./build/neighbor_benchmark 8192
./build/mesh_benchmark 16
```

The example writes `particles_128.csv` and `faces_128.csv` to the current directory. To plot them, install Matplotlib and run `python3 plot_mesh.py` from the repository root.

For GPU experiments, first build or load Kokkos with the relevant CUDA or HIP backend and a compatible compiler/toolchain. Configure this project against that Kokkos installation, then confirm the CMake output lists `CUDA` or `HIP` under enabled Kokkos devices. `neighbor_benchmark` exercises cell search and CSR construction on the selected execution space; `mesh_benchmark` exercises cell reconstruction there. A useful first GPU run is:

```sh
./build/neighbor_benchmark 4096
./build/mesh_benchmark 16
```

The mesh benchmark allocates fixed-capacity face output arrays, about 100 MB at 4,096 particles.

### Measured on Raven

On one NVIDIA A100-SXM4-40GB (Kokkos 5.0.99, CUDA 13.0, GCC 14.1) and one node with two Intel Xeon Platinum 8360Y processors (72 cores, OpenMP, GCC 15.1), both benchmarks pass all of their checks. The mesh benchmark certifies every cell at 1,728, 4,096, and 32,768 particles, and the total volume matches the box to within 3e-15. Neighbor-search rows match the reference exactly for every distribution, variant, periodic and non-periodic box, and lane count tested.

Median time for a full neighbor-search rebuild, uniform points, non-periodic box:

| particles | A100, ~30 neighbors | 72 cores, ~30 neighbors | A100, ~60 neighbors | 72 cores, ~60 neighbors |
|---|---|---|---|---|
| 512 | 0.062 ms | 0.145 ms | 0.070 ms | 0.110 ms |
| 8,192 | 0.120 ms | 0.54 ms | 0.136 ms | 0.91 ms |
| 32,768 | 0.145 ms | 2.3 ms | 0.195 ms | 3.4 ms |
| 1,048,576 | 2.39 ms | 52 ms | 3.68 ms | 74 ms |

- **Periodic boxes** cost 15–20% more on the GPU.
- **Clustered points** are 9–17× faster on the GPU from 32k particles up.
- **Nearest-first row ordering** is slow as a separate pass; at 1M particles it costs several times the search itself, so it should move into the consumer of the rows.
- **Cabana and ArborX** neighbor lists were slower than this search in every case tested.

## Current scope and limitations

This is early research code. The local geometry routine has been exercised on a small synthetic periodic set and the search prototype has been checked against an all-pairs oracle on synthetic distributions. These checks do not establish robustness for all particle configurations, nor do they validate conservation or stability in a hydrodynamics solver. There is no Cabana dependency. The mesh benchmark still builds its candidate graph on the host, so the GPU search does not yet feed the mesh reconstruction.
