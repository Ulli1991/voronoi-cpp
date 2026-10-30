# voronoi-cpp

Kokkos C++17 library for parallel neighbor finding and fully local 3D Voronoi
cell reconstruction. The current focus is construction speed, numerical
robustness and memory use, before any hydrodynamics operations. CPU/OpenMP and
single-GPU/CUDA paths have been validated. Multi-GPU execution is future work.

## Construction interfaces

| File | Purpose |
|---|---|
| `src/neighbour_search/neighbor_search.hpp` | Dense counting-sort cell grid, radius queries and persistent neighbor tables |
| `src/voronoi_construct/local_voronoi.hpp` | Local plane clipping, fixed-capacity workspaces and CSR input |
| `src/voronoi_construct/local_voronoi_table.hpp` | Direct reconstruction from the search table, with diagnostic face vertices |
| `src/voronoi_construct/local_voronoi_hydro.hpp` | Compact geometry: volumes, cell centroids, face areas and face centroids |
| `src/voronoi_construct/local_voronoi_retry.hpp` | Small-workspace pass followed by full-workspace recovery of plane overflows |
| `src/voronoi_construct/local_voronoi_workspace.hpp` | Separate instrumented kernel for workspace and operation counts |

The search feeds reconstruction directly, without a CSR conversion. Outputs and
face neighbor IDs use search slot order; `search.order(slot)` gives the original
particle ID. Centroid offsets are relative to each generator in its local
periodic image. Only successful rows and faces below their row count are valid.

Each cell clips a temporary bounding cube against candidate bisector planes.
Certification requires removal of the initial boundary and sufficient search
reach for the resulting vertices. The caller must supply a complete candidate
list within that reach. Periodic search radii must be below half the box size.
Capacity overflow, coincidence, degeneracy and insufficient reach are explicit
statuses; failed cells must not be consumed as valid geometry.

The compact output occupies **1,704 bytes/particle**, compared with **25,608**
for diagnostic output, excluding inputs, neighbor tables and private workspace.
Despite its `hydro` name, the compact interface performs geometry only.

## Performance options

Existing reference paths and defaults remain available:

- Search defaults to insertion sorting. Select
  `search.row_sort = neighbor_search::RowSort::team_bitonic` for the parallel
  nearest-first sort; capacities above 2,048 retain insertion fallback.
- Reconstruction defaults to `RecomputedVertexPositions`. Select
  `CachedVertexPositions` to reuse intersections through clipping and output.
- `reconstruct_table_retry` and `reconstruct_hydro_retry` default to cached
  vertices and a 64-plane first pass, retrying plane overflows with 160 planes.
- `PlaneWorkspace<64,RelativeClippingTolerance>` selects arithmetic-scaled
  clipping tolerances and pivoted intersection solves. Legacy tolerances and
  determinant-ratio intersections remain the ordinary default/reference.

For example, the compact experimental path is
`reconstruct_hydro_retry<CachedVertexPositions, PlaneWorkspace<64,RelativeClippingTolerance>>(...)`
within the `local_voronoi` namespace. See `src/test/construction_scaling.cpp` for a full
search-to-geometry example with output allocation and validation.

An optional `neighbor_search::StageProfile*` passed to `rebuild` records search
stages with extra fences. `profile_workspace(...)` records per-row peaks and
operation counts in a separate kernel. Both are excluded from production
benchmark timing. `VORONOI_CUDA_RESOURCE_REPORT=ON` enables NVCC resource reports
for `table_mesh_check`.

## Build and checks

Configure against a Kokkos installation with the desired backend:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/kokkos
cmake --build build -j
./build/hydro_geometry_check
./build/construction_stress_check
./build/neighbor_sort_check
./build/table_mesh_check faces.csv 12 team_bitonic cached lattice 64 relative
```

CUDA builds require a compatible compiler/toolchain and CUDA-enabled Kokkos
(e.g. its `nvcc_wrapper` as `CMAKE_CXX_COMPILER`). These commands execute locally;
cluster runs must use an approved allocation.

`table_mesh_check` arguments are:

```text
table_mesh_check output.csv side insertion|team_bitonic recomputed|cached lattice|disordered|regular|random|gradient 160|64 legacy|relative
```

Its test sizes are 12–32 per dimension; the relative path requires cached
vertices. It checks candidate completeness, certification, volumes, closure,
shared faces, compact moments and sampled all-site nearest-vertex conditions.
`python/compare_faces.py` compares geometry CSVs across paths/backends.

`hydro_geometry_check` covers analytic rotated and asymmetric cells, scales,
invalid inputs and forced workspace recovery. `construction_stress_check` uses
an independent long-double plane-triple oracle for difficult small cells.
`neighbor_sort_check` covers ties, padding and the oversized-table fallback.

The original `local_voronoi_example`, `neighbor_benchmark`, `mesh_benchmark`
and `python/plot_mesh.py` remain available. The tiny example uses an all-to-all graph;
`mesh_benchmark` uses a host-generated graph. Use the table integration or
scaling benchmark for the actual search-plus-construction pipeline. Plotting
requires Matplotlib.

## Validated performance and limits

Job **31585499** passed 194 internal result rows and 120 output comparisons on
Raven: CPU/GPU, small/full workspaces and legacy/relative numerics. The table
below uses 32,768 jittered particles, five warmed repetitions, one A100 40GB
versus 18 OpenMP threads. CPU uses insertion sorting; GPU uses team sorting.
These are diagnostic-output timings, in milliseconds:

| Cached path | CPU construction | CPU combined | GPU construction | GPU combined |
|---|---:|---:|---:|---:|
| Legacy, 160 planes | 46.786 | 74.627 | 5.880 | 8.120 |
| Legacy, 64 + retry | 47.118 | 75.029 | 5.988 | 8.141 |
| Relative/pivoted, 160 | 60.260 | 87.950 | 6.439 | 8.658 |
| Relative/pivoted, 64 + retry | 60.263 | 88.182 | 6.524 | 8.684 |

The last path with **compact output** takes **6.071 ms construction / 8.360 ms
combined** on the GPU. These are earlier integration measurements, not results
from the new large scaling sweep.

The 64-plane workspace reduces the cached relative path's compiler-reported
private frame from 8,736 to 4,896 bytes; capacity reduction alone did not give a
material speedup. The full fallback retains its larger frame. No register spills
were reported, but private arrays still use local memory. These frame sizes are
not total GPU memory usage or measurements of runtime memory traffic.

Stress coverage includes scales 1e-8 through 1e8, rotated slivers up to aspect
ratio 1e8, distinct near-coincident sites and a periodic clustered cloud with a
4,900-fold radius contrast. Sliver errors reach about 6e-9 under
conditioning-aware bounds. This is floating-point geometry, not exact predicates
or arbitrary-degeneracy support. The clustered test certifies 102/128 cells;
26 explicitly lack sufficient reach. Automatic reach enlargement remains to be
implemented. No particle merging, splitting, relaxation or hydro updates occur.

## Offline single-GPU scaling

`construction_scaling` measures repeated complete reconstruction of fixed input:

```sh
./build/construction_scaling 32 5 relative scaling.csv
```

Arguments are side length (12–1024), repetitions (1–101), policy
(`relative` or `legacy`) and optional append-only CSV path. The CPU build allows
only sizes through 32 cubed for smoke checks. The scaling workload is a periodic
jittered lattice generated on-device, radius `3.5/side`, approximately constant
neighbor count, fixed neighbor capacity 250, team sorting, cached vertices,
64-plane retry and compact output. It does not represent irregular-cloud scaling.

Reported median times and throughput are separate:

| Stage | Throughput |
|---|---|
| Neighbor rebuild | Million particles processed/s |
| Local reconstruction | Million cells reconstructed/s |
| Combined rebuild + reconstruction | Million particles processed/s |

Each rate is `N / elapsed_seconds / 1e6`; these are construction rates, not cell
updates or hydro steps. Input generation, two warmups, correctness checks and
CSV output are outside timing. Validation checks every cell's status, positive
volume, finite moments, closure and centroid tensor, global volume, and sampled
neighbor lists against a device all-pairs oracle.

`scripts/run_scaling_gpudev.sh` prepares a 15-minute, one-A100 job with four CPU cores.
It tries sides 32, 48, 64, 96, 128, 160, 192, 256, 512 and 1024 until the first
memory exclusion, then bisects to the largest integer side admitted by its
memory budget. Every size runs in a fresh process. Before array allocation it
reserves the larger of 2 GiB or 15% of device capacity; oversized cases append
`SKIP_MEMORY` and return 77. Other errors stop the sweep.

The current live array payload is approximately 4,780 bytes/particle plus the
dense grid. Thus 1024 cubed needs roughly 5 TB, and a 40GB GPU is expected to
reach between 128 cubed and 256 cubed. The sweep determines a conservative
budget ceiling, not the absolute allocation limit. CSV memory fields distinguish
estimated array payload from sampled total device usage; the latter includes
runtime/context allocations and is not a guaranteed peak measurement.

**Scaling job 31589055 completed successfully in 2m38s.** All nine measured
sizes passed with zero validation errors. The largest admitted size was
195 cubed (7,414,875 particles); 196 cubed exceeded the conservative memory
budget and was skipped before allocation. At 195 cubed, search took 555.645 ms,
reconstruction 1,196.984 ms and the combined pipeline 1,751.970 ms: respectively
13.345 million particles/s, 6.195 million cells/s and 4.232 million particles/s.
Maximum sampled device usage was 37.70 GB (35.11 GiB), not a guaranteed peak.
The skipped sizes are budget exclusions, not geometry failures. Ask before
submitting any further cluster test job. Jobs run independently of the console. Results go to
`build-gpu-scaling/<jobid>/scaling.csv`, with log
`build-table-validation/scaling-offline-<jobid>.out`.

Optional plotting (requires Matplotlib):

```sh
python3 python/plot_construction_scaling.py build-gpu-scaling/<jobid>/scaling.csv
```

This produces throughput, timing and memory panels in PNG and PDF. CSV collection
does not depend on plotting packages.

## Development record

See [HANDOFF.md](HANDOFF.md) for validation logs, historical measurements and
remaining work. The previous README is preserved in
[docs/README-before-offline-scaling.md](docs/README-before-offline-scaling.md).
Existing code, reference paths and artifacts are retained.

## Experimental candidate early stop

`PlaneWorkspace<64,RelativeClippingTolerance,true>` enables a conservative
candidate-loop stop; the third template argument defaults to false. The cell's
vertex radius bounds the bisectors that can still cut it. The implementation
checks every remaining candidate distance before stopping, so it also handles
unsorted rows and late coincident sites. Existing final certification still runs.
No persistent buffers are added; the full-workspace retry preserves the option.

Use `relative_early` instead of `relative` in `table_mesh_check` or
`construction_scaling` to compare paths. `construction_early_stress_check` runs
the independent stress oracle with this option and includes unsorted late-cutter,
late-coincidence and radius-boundary fixtures. Local CPU comparisons on five
small distributions match reference geometry. The jittered case reduces
vertex-plane tests from 8,057,728 to 620,608; the suffix-distance checks are not
included in that counter. This operation-count reduction is not itself a GPU speedup or roofline result;
measured GPU timings are reported below.

`scripts/validate_early_stop_gpudev.sh` prepares stress checks, five distribution
comparisons and paired scaling runs at 32, 128 and 195 cubed on one A100.
Job **31589308 completed successfully** in 4m13s. All stress, geometry and
scaling checks passed. At 195 cubed, early stopping reduced reconstruction
from 1,196.39 to 1,000.87 ms (1.20x throughput), and combined time from
1,750.65 to 1,556.38 ms (1.12x throughput). Reconstruction achieves 7.41
million cells/s; combined throughput is 4.76 million particles/s. Sampled
device usage is unchanged. At 128 cubed the respective speedups are 1.25x
and 1.16x; at 32 cubed they are only about 1.02x. Obtain approval before further submissions. Hardware memory traffic and
occupancy profiling remain separate work. LUNA's user-reported ~1 billion full
hydro cell updates/s is a useful performance target, not a directly equivalent
workload or a measured roofline for this construction kernel.

## Experimental nearest-batch candidates

`PlaneWorkspace<64,RelativeClippingTolerance,true,B>` with batch size `B>0`
reads unsorted rows: configure the search with `nearest_first=false`, which
skips the row sort and the per-entry distance table (8 bytes per table entry).
The fourth template argument defaults to 0 (off). Each cell visits its
candidates in (distance squared, slot) order, the insertion sort's order,
selecting the `B` nearest remaining candidates per scan of its row. Because this
order is exact, early stopping ends at the first candidate beyond the cutoff
without the suffix check. The full-workspace retry preserves the option.
`default_nearest_batch<ExecSpace>()` gives 8 on GPUs and 32 on host backends.

Use `relative_nearest_batch` in `table_mesh_check` (with `insertion`, which is
ignored) or `relative_early_nearest_batch` in `construction_scaling`; both use
the backend default. `construction_nearest_batch_stress_check` runs the stress
oracle with batches of 4, so its small fixtures rescan rows.

Each output was byte-compared with `relative_early` under insertion sorting,
for the five `table_mesh_check` distributions at 12 and 32 cubed, at the
default batch and at batch 4. On OpenMP all 20 comparisons are byte-identical.
On CUDA 18 of 20 are. The exceptions are the unjittered `regular` lattice at
32 cubed, at both batch sizes:
- The face sets match exactly.
- 2,016 of the 32,768 cells list their faces in a different order. Matched by
  face, every diagnostic value is bit-identical.
- In the compact output, those cells' cell centroids differ by at most
  1.08e-19. The values are near zero, about 5e-20.

The likely cause is the lattice's many exact distance ties. The search and
the reconstruction compute distances with different, separately contracted
expressions, so tied neighbours can be visited in another order.

Local measurements were taken on a Quadro RTX 6000 with the SM clock fixed at
1,440 MHz, on the jittered lattice of `construction_scaling`. Each figure is
the median of 3 alternating rounds, each round the median of 5 repeats. Times
are per particle, as search / reconstruction / pipeline:

| Side | Baseline (team bitonic) | Nearest batch (8) |
|---:|---|---|
| 32 | 258 / 650 / 906 ns | 91 / 571 / 662 ns |
| 48 | 245 / 584 / 831 ns | 83 / 584 / 670 ns |
| 64 | 246 / 553 / 803 ns | 82 / 585 / 663 ns |
| 80 | 247 / 533 / 781 ns | 81 / 558 / 639 ns |

The nearest-batch cost per particle is flat with size. Reconstruction is about
5% slower than the baseline from 64 cubed up, while search is about 3× faster.
Peak device memory at 64 cubed falls from 1,950 to 1,456 MiB, about 2.8 instead
of 4.9 KB per particle.

At the same fixed clock, GPU batches of 4, 6, 8, 12, 16, 24 and 32 gave
reconstruction times of 629, 577, 573, 594, 631, 687 and 646 ns per particle
at 80 cubed, and batch 8 was also fastest at 32 and 48 cubed. The host default
of 32 has not been measured at a fixed clock. A100 performance is not yet
measured.

## Repository layout

- `src/neighbour_search/`: neighbor search and fixed-point position inputs.
- `src/voronoi_construct/`: local cell reconstruction and geometry workspaces.
- `src/test/`: examples, correctness checks and benchmarks.
- `scripts/`: Slurm validation, scaling and profiling jobs. Submit from the repository root using `sbatch scripts/<job>.sh`.
- `python/`: output comparisons and plotting tools.

CMake exposes `voronoi_library` with both library include directories.
Executable names and build output locations are unchanged.

## Linting and continuous integration

GitHub Actions (`.github/workflows/checks.yml`) runs Ruff lint/format checks,
clang-tidy, and CPU correctness tests in ordinary and address/undefined-behavior
sanitizer builds. CI builds a pinned Serial Kokkos 5.0.2. It does not submit
cluster jobs or replace A100 validation.

```sh
python3 -m pip install ruff==0.11.13
ruff check python
ruff format --check python
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/kokkos -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build -j 2
bash scripts/lint_cpp.sh build
ctest --test-dir build --output-on-failure
```

Use a host compilation database for clang-tidy. Add
`-DVORONOI_SANITIZERS=ON` in a separate host build to enable sanitizers.
Tests include analytic and irregular geometry, sorting edge cases, periodic and
nonperiodic search oracles, cross-path geometry/moment comparisons over five
distributions, coalesced and fixed-point input checks, fixed-point wrapping and
quantization contracts, and rejection tests for malformed geometry CSVs.

The clang-tidy configuration excludes `clang-analyzer-cplusplus.NewDeleteLeaks`
because it does not model Kokkos reference-counted allocations; CI retains
runtime leak detection. Ruff changes only formatting/import order.

## Bounded coalesced-cache experiment

The optional `relative_early_batched` scaling policy reuses a coalesced vertex
cache across batches of 65,536 cells, capping persistent cache storage at 96 MiB.
The full-size `relative_early_coalesced` policy remains available. Launches on the
same execution instance reuse storage in order; overflow recovery runs afterward.
Plane and topology arrays remain private. `relative_early_batched_test` uses
257-row batches to exercise uneven boundaries in the host correctness suite.

Job31590387 validated all six previous comparison rows. At128 cubed, full-cache
reconstruction took259.60ms versus279.07ms for private cached vertices (~1.075x),
but added3GiB. Fixed32 input-only storage gave no measured speedup. The bounded
variant has passed local checks; GPU performance is not yet measured.
`scripts/compare_batched_gpudev.sh` is prepared, not submitted; approval required.
