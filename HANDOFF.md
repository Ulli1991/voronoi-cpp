# Local reconstruction integration

The previous stopping point is recorded in README.md and commit c75db57:
the dense-cell neighbor search and Voronoi reconstruction were validated
separately, while mesh_benchmark still builds candidates on the host.

The new local_voronoi_table.hpp provides reconstruct_table(), a direct adapter
from fixed-capacity neighbor tables to the existing reconstruction kernel.
No CSR graph is allocated or copied. The existing CSR interface and benchmarks
are preserved. Pass search.positions, search.counts, search.table and
search.radii after a completed nearest-first search rebuild.

Outputs and neighbor IDs use search slot order. Map a slot back to the original
particle ID with search.order(slot). Vertices remain relative to each generator.
Invalid counts report invalid_candidate_count (8); unsuccessful rows expose zero
faces. View extent mismatches throw before launch. Candidate IDs and geometric
input validity remain caller responsibilities, as for the CSR interface.

table_mesh_check exercises periodic search-to-reconstruction integration with
permuted input IDs, forced search capacity growth, an all-pairs neighbor oracle,
positive volumes, total volume, area closure, shared faces, invalid counts,
incomplete candidates and mismatched extents.

Validation is pending. Do not submit cluster test jobs without asking the user.
Do not delete files or remove existing code without asking the user.

Compilation succeeded with GCC 15.1 and the existing Kokkos OpenMP/Serial
installation in a separate /tmp/voronoi-table-build directory. The executable
has not been run. CPU and GPU runtime checks remain pending user approval for
cluster test submission.

Further work: run the integration check on CPU and GPU; measure the table path;
add analytic and degenerate geometry cases; investigate scale-aware clipping
tolerances and configurable workspace capacities. Nearest-first ordering is
still a separate search pass, and workspace/degeneracy limits are unchanged.

## Completed gpudev validation, 2026-10-27

The pending-validation notes above describe the initial draft. Jobs 31582631
and 31582789 ran on ravg1002, comparing one A100-SXM4-40GB against 18 OpenMP
threads on the same node. Every check passed for 1,728, 4,096, 13,824 and
32,768 particles. Original-ID face sets agree; maximum CPU/GPU area-component
difference is below 2.5e-17 and total-volume error below 3.2e-15.

Median milliseconds, five repetitions after warm-up, persistent allocations:

| Particles | CPU search | GPU search | CPU reconstruction | GPU reconstruction | CPU pipeline | GPU pipeline |
|---|---:|---:|---:|---:|---:|---:|
| 1,728 | 3.312 | 5.059 | 10.187 | 5.996 | 13.530 | 10.789 |
| 4,096 | 3.507 | 5.133 | 23.745 | 6.199 | 27.347 | 11.320 |
| 13,824 | 11.402 | 5.773 | 79.899 | 6.332 | 91.449 | 12.079 |
| 32,768 | 27.447 | 19.901 | 189.205 | 13.229 | 216.909 | 33.187 |

Search includes the complete rebuild, nearest-first ordering and its overflow
check, with already-grown buffers. Reconstruction includes its completion fence.
Pipeline is measured independently. All-pairs oracle, geometry checks and CSV
output occur outside the timed sections. Inputs are periodic jittered lattices
with radius 3.5/side, substantially more candidates than the README's ~30/~60
neighbor search examples. These are narrow baseline measurements, not a claim
about arbitrary hydro particle distributions or full-node CPU scaling.

Performance direction: optimize both neighbor preparation and reconstruction
for repeated hydro mesh rebuilds. At 32k, search including sorting dominates
the GPU pipeline; isolate sorting before choosing a replacement. Keep reusable
device storage and avoid CSR transfers. Full vertex output reserves 24 KiB per
particle by itself; investigate a compact hydro output (volume, cell centroid,
face area vectors and face centroids) without removing the diagnostic interface.
Do not trade certification or conservation checks for speed.

Reproduction: validate_gpudev.sh builds/runs the first comparison;
validate_scaling_gpudev.sh uses those binaries for sides 16, 24 and 32.
Both must be submitted from the repository root. Local logs and face dumps are
in build-table-validation/. Cluster submission requires user permission; the
user explicitly authorized these gpudev comparison runs.

## Compact hydro output and profiling draft

local_voronoi_hydro.hpp adds reconstruct_hydro() with table input. It shares
reconstruct_row's clipping and face traversal using a compile-time geometry
observer. The default observer preserves diagnostic output; the hydro observer
integrates face/cell moments from fan triangles without storing polygons.
Output storage is 1,704 bytes/particle versus 25,608 bytes/particle for full
diagnostic output; neighbor storage and the existing clipping workspace are
additional. Moment arithmetic adds work, so no speedup is claimed before timing.
Both APIs still use one serial clipping task per cell, parallel across cells.

Hydro outputs: face counts and status [N], neighbor IDs [N,max_faces], outward
area vectors and face centroid offsets [N,max_faces,3], volumes [N], and cell
centroid offsets [N,3]. Offsets are relative to the generator in its local
periodic image. Slots/ID mapping follow reconstruct_table. Failures clear face
count, volume and cell centroid; stale face entries must not be consumed.

StageProfile is optional on neighbor_search::build/fill/rebuild. Its explicit
fences isolate key/count, scan, scatter, cell ordering/gather, candidate fill,
nearest-first sort and overflow read. Normal calls add no profiling fences.
Stage medians are diagnostic; use unprofiled timings for total performance.
Build/fill accumulate profile values; rebuild resets and accumulates retries.

hydro_geometry_check tests translated symmetric/asymmetric rectangular cells
at scales 0.1, 1 and 10, checking analytic volumes, cell/face centroids, areas,
shape rejection and clearing moments after coincident-particle failures.
These six cases passed locally on one CPU thread. All CPU and CUDA targets
compile (GCC 15.1/OpenMP and GCC 14.1/CUDA). The extended table_mesh_check
compares compact output against diagnostic polygons, including shared periodic
face centroids; it also writes .hydro.csv comparison files when given a path.

validate_hydro_gpudev.sh is prepared but NOT SUBMITTED. It requests one A100,
18 CPU threads, 15 minutes; builds both backends, runs analytic checks, compares
diagnostic and hydro outputs at 1,728/4,096/32,768 particles, and reports stage,
reconstruction and pipeline timings. Each job uses a separate output directory
under build-hydro-validation/<jobid>. Ask the user before submitting this job.
Earlier recorded GPU baselines predate the compact output implementation.

The extended 1,728-particle integration check also passed locally on one CPU
thread: errors=0, volume error 4.45e-16, compact moments agree with diagnostic
polygons, and periodic shared-face centroids agree. Local one-thread timings
are correctness-run diagnostics, not a substitute for the pending cluster
comparison. Existing cluster baselines and their output files are retained.

## Completed compact hydro validation, job 31583267

User approved submission; job ran on gpudev/ravg1002 and completed with exit
code 0 in 2m51s. All six analytic cases pass on both OpenMP and CUDA. All
diagnostic/hydro comparisons pass at 1,728, 4,096 and 32,768 particles,
including moment checks against diagnostic polygons, periodic shared-face
centroids, invalid/incomplete neighbor rows, and CPU/GPU comparison files.
Maximum CPU/GPU difference across hydro area/centroid/volume fields is
3.36e-15. Total-volume error is at most 2.23e-15 for these runs.

One A100-SXM4-40GB versus 18 OpenMP threads; medians of five warmed runs,
same periodic jittered lattices and fixed search radius 3.5/side as before:

| Particles | CPU hydro build ms | GPU hydro build ms | CPU hydro pipeline ms | GPU hydro pipeline ms |
|---|---:|---:|---:|---:|
| 1,728 | 10.229 | 7.034 | 13.614 | 12.964 |
| 4,096 | 23.845 | 7.350 | 27.353 | 13.384 |
| 32,768 | 189.379 | 14.878 | 216.890 | 37.378 |

At 32,768 particles, synchronized stage-profile medians:

| Neighbor stage | CPU ms | GPU ms |
|---|---:|---:|
| Key/count | 0.0306 | 0.0152 |
| Scan | 0.0095 | 0.0208 |
| Scatter | 0.0296 | 0.0138 |
| Cell order/gather | 0.0344 | 0.1033 |
| Candidate fill | 9.0269 | 0.9805 |
| Nearest-first ordering | 18.2875 | 21.4138 |
| Overflow check | 0.0013 | 0.0122 |

Unprofiled GPU search rebuild was 22.520 ms. Nearest-first ordering accounts
for about 95% of the profiled search cost, confirming it as the next target:
evaluate a parallel per-row sort while retaining the original path for
comparison. Preserve distance ordering and deterministic tie-breaking, which
limit transient clipping capacity failures. Candidate fill is already fast;
the current evidence does not justify replacing the radius-search grid.

Compact reconstruction is close in speed to diagnostic reconstruction in this
run (GPU 14.878 vs 15.292 ms; CPU 189.379 vs 188.731 ms), while returning
additional moments. The robust improvement is output memory: 55,836,672 bytes
versus 839,122,944 bytes at 32k, about 15x less. This excludes neighbor storage
and per-cell clipping workspace. Narrow synthetic results do not establish
production hydro robustness, full-node CPU scaling or multi-GPU performance.

Log: build-table-validation/hydro-31583267.out. Builds and comparison files:
build-hydro-validation/31583267/. Earlier baseline data are preserved. No
further cluster job is submitted or authorized by this result alone; ask before
submitting the next test job, following the user's standing ground rule.

## Parallel sorting draft

User authorized work on the sorting bottleneck. DenseCellSearch now has an
explicit RowSort option, defaulting to the preserved insertion path. Set
search.row_sort=neighbor_search::RowSort::team_bitonic to enable the experimental
team sort. configure() leaves that option intact, including during overflow
growth. No new persistent buffers are allocated. The bitonic network stages
one row's distance/slot pairs through team scratch and writes them back once.
GPU teams use 128 threads; host teams use one thread to validate the algorithm.
Scratch is bounded at about 24 KiB/team for capacities up to 2,048; larger
capacities fall back to the original insertion sort. Padding is based on actual
row length (scratch reservation on table capacity). Sorting is lexicographic
by exact squared distance then slot, matching the existing path.

neighbor_sort_check passed locally with one CPU thread for capacities
0/1/7/33/128/257/513/1025/2048 and correctly declines direct team sorting at
2049. It checks all-equal/random/reversed distances, tied IDs, short/empty rows,
counts above capacity, unchanged padding and row counts, and insertion fallback
through a real rebuild. All CPU and CUDA targets compiled; a CUDA warning about
standard-library padding constants was corrected by capturing host constants.

The extended table_mesh_check accepts insertion/team_bitonic as a fourth
argument. For team_bitonic it first verifies exact neighbor IDs and distances
against insertion, then runs all diagnostic/hydro geometry and moment checks.
The initial small capacity also exercises growth with the selected sorter.

validate_sort_gpudev.sh is prepared but NOT SUBMITTED. It requests one A100,
18 CPU threads and 15 minutes, builds both backends, checks sorting edge cases,
and compares insertion/team_bitonic at 1,728/4,096/32,768 particles. It validates
both CPU/GPU and old/new geometry and reports stage and pipeline timings.
Fresh artifacts go to build-sort-validation/<jobid>; logs to
build-table-validation/sort-<jobid>.out. Ask the user before submission.

Terminology clarification agreed with user: hydro based on these explicit
Voronoi volumes and moving faces would be moving-mesh finite volume, in the
same numerical family as AREPO. It is not conventional kernel-partition MFV.
This repository currently supplies geometry, not a complete hydro solver.

The 1,728-particle local integration check with team_bitonic has now passed:
errors=0, exact IDs/distances agree with insertion after capacity growth, and
all diagnostic/hydro geometry checks pass (volume error 4.45e-16). The host
team network is a correctness reference and is slower than insertion in the
one-thread run; CPU insertion remains the default. GPU runtime correctness
and speed remain pending the prepared job's approval and execution.

## Parallel-sort validation complete, job 31583755

User approved submission. The gpudev job completed with exit code 0 in 5m13s.
All sorting edge cases and exact neighbor ID/distance comparisons passed on
CPU and GPU. Both sorter paths pass all construction checks at 1,728, 4,096
and 32,768 particles. Original/new diagnostic and compact geometry files agree
exactly (maximum difference 0), on both CPU and GPU. CPU/GPU area differences
remain below 2.5e-17, and combined centroid/volume differences below 3.4e-15.
The largest-case total-volume error is 2.23e-15. These are the synthetic cases
described earlier, not proof of robustness for all degenerate configurations.

Same-allocation A100 timings, median of five warmed runs (milliseconds):

| Particles | Insertion search | Team search | Insertion diagnostic pipeline | Team diagnostic pipeline |
|---|---:|---:|---:|---:|
| 4,096 | 6.085 | 0.437 | 13.474 | 7.843 |
| 32,768 | 22.511 | 2.264 | 37.762 | 17.520 |

At 32k, GPU ordering alone drops from 21.412 to 1.183 ms, an 18x improvement;
full search improves about 10x and construction about 2.16x. Reconstruction
itself remains ~15.26 ms. Compact geometry pipeline improves from 37.364 to
17.144 ms. Compact outputs are geometry only; no hydro update is performed.
On 18 CPU threads the team network is slower: 60.382 ms search versus insertion's
27.728 ms. Both implementations remain available; insertion is still the
default and team_bitonic is explicitly selectable on the search state.

Artifacts: build-sort-validation/31583755/. Log:
build-table-validation/sort-31583755.out. No further test job is submitted.

## Current user priorities and scope

Maximize speed AND accuracy of the local construction library before adding
any hydrodynamics operations. The defining advantage is independent cell
construction from complete local candidate lists, without a global Delaunay
triangulation or shared mutable mesh connectivity. Certification verifies
completeness beyond the supplied radius; obtaining local candidates still
requires neighbor search and, eventually, ghost exchange across domains.
Keep this local independence when optimizing. No fluxes, Riemann solver or
time integration are in the current work scope. Existing compact geometry
outputs are retained. Do not delete files/code; ask before cluster submissions.

The user requested an informational comparison with rennehan/plume, reviewed
at commit c2481c8 in a separate /tmp checkout. No Plume code or new algorithm
was imported. The user explicitly does not want to follow its design by
default: use measurements and construction accuracy to choose improvements.
Next bottleneck from our experiment is clipping/reconstruction, now ~87% of
the diagnostic GPU pipeline. Broader accuracy checks and measurements of
workspace/occupancy should accompany any optimization there.

## Cached vertex experiment and independent accuracy checks

User reiterated the scope: maximize speed, robustness and memory efficiency of
neighbor finding and local construction, before hydro. Comparisons with other
projects are informational, not instructions to copy or follow their design.
The LUNA repository was mentioned as experience available in the user's home;
no LUNA source has been read and no dependency on it was introduced.

The existing clipping kernel now accepts a compile-time vertex storage policy.
RecomputedVertexPositions remains the default. CachedVertexPositions retains
coordinates for the 64 possible live vertices, moves them with vertex triples
during compaction, and computes each new intersection once. Certification and
face output use the same coordinates. It adds 1,536 private bytes per cell,
with no extra persistent arrays. No clipping tolerance, capacity, ordering or
certificate condition was changed. Both reference and optimized paths remain.
The new policy is selected on reconstruct_table/reconstruct_hydro as the first
explicit template argument; inferred existing calls continue to work.

Accuracy improvements to checks (not a new numerical tolerance policy):
- 40 analytic cases across both policies, five scales 1e-3 through 1e3,
  symmetric/asymmetric boxes, translated and rotated, all pass locally.
- Periodic jittered, strongly disordered (jitter 0.45 spacings) and regular
  lattices at 1,728 particles pass on one CPU thread in both policies.
- Cache/reference diagnostic and compact geometry files match exactly for
  all three distributions. Existing completeness, certification, positive
  volume, shared-face and total-volume checks remain enabled.
- Added all-particle nearest-site inequality checks at vertices of 32 sampled
  cells, including sites outside the candidate radius, and the independent
  face-centroid divergence identity sum_f c_f A_f^T = V I for every cell.

The local single-thread jittered-lattice clipping timings were ~185 ms
reference versus ~44 ms cached (not cluster benchmark claims). Regular-lattice
total-volume error was 1.91e-14 in both paths; disordered error 2.23e-16.
CPU and CUDA targets compile. Path defaults have not changed. Tiny-scale
tolerance robustness, arbitrary degeneracies and capacity recovery still need
further work; these checks do not establish all-configuration correctness.

validate_clipping_gpudev.sh is prepared but NOT SUBMITTED. Request: one A100,
18 CPU threads, 15 minutes; builds both paths, runs the 40 analytic cases on
both backends, validates all three distributions at 1,728 particles and the
jittered case at 32,768, compares geometry and timings. CPU uses insertion,
GPU uses the measured team_bitonic sort, so each backend's reference/cache
comparison holds neighbor preparation fixed. NVCC resource reports are enabled
to expose registers, stack and spills alongside performance measurements.
Logs: build-table-validation/clipping-<jobid>.out. Fresh artifacts:
build-clipping-validation/<jobid>/. Ask user before submitting this job.

## Cached vertex validation complete, job 31584361

User approved submission. gpudev/ravg1002 job completed successfully in 5m38s.
All 40 analytic cases passed on both CPU and GPU. Both vertex paths pass
all-pairs neighbor checks, certification, vertex nearest-site inequalities,
the face-centroid divergence theorem, centroid/volume checks and shared-face
checks for jittered, strongly disordered and regular periodic inputs at
1,728 particles, plus the jittered input at 32,768. All reference/cache
diagnostic and compact output comparisons are exactly equal on each backend.
CPU/GPU outputs agree within rounding; maximum tested face-area discrepancy
is below 4.8e-17 and combined volume/centroid discrepancy below 3.4e-15.
Total-volume error is 2.23e-15 at 32k and 1.91e-14 on the regular lattice.

Median milliseconds of five warmed repetitions, same allocation and inputs:

| Case | CPU clipping reference | CPU clipping cached | A100 clipping reference | A100 clipping cached |
|---|---:|---:|---:|---:|
| Jittered, 1,728 | 10.076 | 2.520 | 7.039 | 2.499 |
| Disordered, 1,728 | 9.930 | 2.599 | 8.107 | 3.181 |
| Regular, 1,728 | 3.176 | 1.041 | 1.334 | 0.509 |
| Jittered, 32,768 | 186.373 | 47.016 | 15.287 | 5.877 |

At 32k, the full diagnostic pipeline is 17.579 -> 8.127 ms on GPU and
213.645 -> 75.159 ms on 18 CPU threads. The compact-geometry pipeline is
17.076 -> 7.909 ms on GPU and 217.162 -> 74.232 ms on CPU. These are geometry
construction timings, with no hydro operations. GPU search is team_bitonic;
CPU search is insertion. Timed sections exclude validation and file output.

NVCC sm_80 resource reports (no compiler-reported register spills):

| Path | Private stack bytes/thread | Registers/thread |
|---|---:|---:|
| Reference, diagnostic | 7,104 | 86 |
| Cached, diagnostic | 8,640 | 78 |
| Reference, compact | 7,104 | 114 |
| Cached, compact | 8,640 | 95 |

The stack is device local memory even though compiler register-spill counters
are zero; do not describe this as a register-resident construction. Persistent
buffer sizes are unchanged by caching. The extra 1,536 bytes/cell are private
temporary coordinates. Both reference and cached paths remain selectable;
defaults are unchanged. Fastest tested setup is cached positions with team
sorting on CUDA, cached positions with insertion on CPU.

Artifacts: build-clipping-validation/31584361/. Log and resource report:
build-table-validation/clipping-31584361.out. A local resource-only compiler
report also lives at /tmp/voronoi-clipping-resources.4R6pB0.
No additional cluster job has been submitted. Source still needs deeper checks
of extreme scales, arbitrary random/dense configurations, near-degeneracies,
and capacity recovery; passing these test families is not universal robustness.

Next measured optimization direction: clipping remains the largest GPU stage
(5.88 ms versus ~2.26 ms search at 32k). Its 7-9 KiB private frame is dominated
by the 160-plane array. Measure actual workspace high-water marks and runtime
local-memory traffic before changing capacities or storage. Keep larger/reference
paths and failure reporting; don't silently trim geometry for speed.

## Workspace measurements and irregular-distribution work (not yet GPU-tested)

User approved workspace measurement/memory reduction, then explicitly emphasized
highly irregular point distributions and poor cells. They raised split/merge
as a possible remedy, agreed with first fixing construction numerics, and asked
to continue. No particle positions/counts have been changed by the library.
No new cluster job has been submitted in this phase; permission is still needed.

Added optional WorkspaceUsage instrumentation in local_voronoi_workspace.hpp.
Normal calls instantiate NoWorkspaceTrace, with no counters or output writes.
Separate profile_workspace launch returns per-row peaks (including the next
required entry on a capacity failure), candidate/vertex tests, coordinate reads,
created vertices, logical intersection solve counts and final clipping status.
Instrumentation runtime is not production reconstruction runtime.

Local one-thread peak measurements for 1,728 points (planes include initial 6):

| Distribution | Planes | Vertices | Intermediate hole edges | Faces | Face vertices |
|---|---:|---:|---:|---:|---:|
| Jittered | 28 | 40 | 9 | 22 | 12 |
| Strongly disordered | 31 | 44 | 13 | 24 | 13 |
| Regular | 12 | 8 | 6 | 6 | 4 |
| Uniform random | 37 | 50 | 13 | 27 | 12 |
| Smooth 4:1 spacing gradient | 29 | 42 | 12 | 23 | 12 |

PlaneWorkspace<64> reduces plane storage from 6,400 to 2,560 bytes. All original
capacities/defaults remain; the plane array is now templated. New retry wrappers
perform the smaller launch and a second device launch which rebuilds only rows
whose status is plane_capacity. The second launch scans statuses but allocates
no buffers and reads no host flag. Other failures remain explicit. Fallback
uses 160 planes and the SAME numerical policy as the small pass. All 60 analytic
cases now include deliberately forced 8-plane failure and successful full-size
recovery for both diagnostic and compact APIs. Ordinary 64-plane local cases
needed zero fallbacks and match the 160-plane geometry exactly under legacy math.

Important: the new irregular stress suite exposed actual numerical limitations:
- Original absolute tolerance rejects a valid scaled 1e-8 cell as uncertified.
- Original coincidence floor rejects a distinct site at distance about 1e-15.
- Arithmetic-scaled classification alone still left ~9.03e-7 relative geometric
  errors in a dense-cluster/void cloud, despite correct neighbors and good closure.
- Adding equilibrated partial-pivoting elimination for plane intersections
  reduced that cloud's worst nearest-site/bisector residuals to ~1.54e-15 with
  the same test thresholds and fixed positions. No merge, split or relaxation.

RelativeClippingTolerance is an experimental opt-in numerical policy. It removes
the absolute length floor, recognizes zero-distance coincidence, and selects the
pivoted solver. LegacyClippingTolerance and the original determinant-ratio
intersection remain the default/reference. This is NOT exact geometry; nearly
singular pivots still fail explicitly. Do not claim arbitrary-degeneracy support.

construction_stress_check uses an independent CPU long-double oracle: enumerate
all plane triples, retain feasible vertices, group by planes and order polygons
by angle (not incremental dual clipping). It checks volumes, area vectors and
centroids over scales 1e-8..1e8, rotated slivers with aspect ratios 1e2..1e8,
dense/void local geometry, near-coincident distinct sites and true coincidence.
Bounds for slivers account for their conditioning and are printed; at aspect
1e8 measured volume error is ~6.41e-9, not machine precision. The periodic cloud
has 128 points, 64 in a 1e-5-wide cluster, and search radii 1e-4 vs 0.49. Its
all-pairs neighbor check passes; 102 cells certify and pass nearest-site/bisector
and closure checks, 26 report insufficient reach, no other failures. Automatic
radius enlargement is not yet implemented. Those 26 are not claimed recovered.

Local CPU checks passed: 60 analytic cases, all stress cases, and five periodic
datasets comparing legacy/full with relative/small (both geometry and moments).
All CPU/CUDA targets compile. No GPU runtime check of these additions yet.

NVCC sm_80 private stacks, no compiler-reported register spills:
- Cached legacy initial: 8,640 bytes at 160 planes, 4,800 at 64 planes.
- Cached relative/pivoted: 8,736 at 160 planes, 4,896 at 64 planes.
- Legacy cached register counts unchanged by capacity: diagnostic 78, compact 95.
- Relative/pivoted: diagnostic 80, compact 104.
The full fallback retains its full-size frame. This measures common-path private
frame reduction, not total peak GPU memory reduction; runtime local-memory
traffic/hardware counters have NOT been measured. No production buffers added.
Local compile report: /tmp/voronoi-irregular-resources.n1Og2p.

validate_workspace_gpudev.sh is prepared, NOT SUBMITTED. It requests one A100,
18 CPU threads and 15 minutes. It builds both backends; runs analytic and
independent irregular stress tests; compares 64/160 planes and legacy/relative
numerics on five 1,728-point distributions plus jittered 32,768; validates
CPU/GPU, small/full and legacy/relative geometry; reports production timings,
workspace peaks, fallback counts and NVCC resources. CPU uses insertion, GPU
team sorting. Artifacts use build-workspace-validation/<jobid>, with log
build-table-validation/workspace-<jobid>.out. Ask before submitting this job.

## Workspace and irregular geometry validation complete, job 31585499

User approved submission; job completed on gpudev/ravg1002 with exit code 0
in 13m31s. All 194 internal result rows have errors=0: 60 analytic cases and
13 stress cases per backend, plus 48 periodic benchmark runs. All 120 output
comparisons pass (CPU/GPU, small/full workspaces, legacy/relative numerics).
No further cluster job is submitted. Existing files and reference paths remain.

One A100 versus 18 OpenMP threads, median of five warmed runs, 32,768 particles:

| Numerical path | Planes | CPU build ms | CPU pipeline ms | A100 build ms | A100 pipeline ms |
|---|---:|---:|---:|---:|---:|
| Cached legacy | 160 | 46.786 | 74.627 | 5.880 | 8.120 |
| Cached legacy with retry | 64 | 47.118 | 75.029 | 5.988 | 8.141 |
| Cached relative/pivoted | 160 | 60.260 | 87.950 | 6.439 | 8.658 |
| Cached relative/pivoted with retry | 64 | 60.263 | 88.182 | 6.524 | 8.684 |

The last path's compact-geometry GPU build/pipeline is 6.071/8.360 ms. Runtime
includes retry launch and fences. Workspace profiling, all-pairs checks, vertex
oracles and output are outside production timings. CPU uses insertion sort;
GPU uses team_bitonic. No hydro operations are performed.

Capacity reduction alone did not materially improve speed: it is a common-path
private-stack improvement, not a measured throughput win. Relative/pivoted
numerics cost ~7% GPU pipeline time against cached legacy/full in this test,
and ~18% CPU pipeline time. Smaller workspaces do not eliminate the full-size
fallback's frame. Actual total peak GPU memory and hardware local-memory
traffic were NOT measured; Slurm MaxRSS is host memory, not GPU memory.

GPU irregular-cloud result: radius ratio 4,900, exact all-pairs neighbor match,
102 certified cells, 26 uncertified for insufficient reach, no other failures.
Worst nearest-site/bisector residuals are 1.51e-15/1.77e-15; closure 1.21e-15.
The improvement from the earlier ~9e-7 failure occurred with fixed positions.
Scaled 1e-8 cells and distinct ~1e-15 sites pass the new numerical path; true
coincidence is rejected. Rotated slivers at aspect ratio 1e8 have GPU relative
volume/area errors 6.03e-9/5.04e-9, within the printed conditioning-aware bound;
do not claim machine precision for all slivers or arbitrary exact degeneracies.

Workspace peaks agree on CPU/GPU. Across the five small distributions maxima
are 37 planes, 50 vertices, 13 intermediate hole edges, 27 faces, 13 vertices
on a face. The 32k jittered case peaks at 30/44/10/24/13. All ordinary 64-plane
benchmark rows fit; forced 8-plane analytic tests separately validate recovery
through the full workspace under both numerical policies.

The operation counters identify remaining work: the 32k jittered case performs
153,382,526 vertex-plane tests and 159,762,914 coordinate reads, but only
2,074,357 new-vertex intersection solves. Many sorted candidates therefore get
tested even after they can no longer affect the cell. A conservative early-stop
optimization based on the shrinking cell radius is a candidate for measurement;
it must preserve correctness for unsorted input and numerical edge cases.
Automatic reach growth for the 26 uncertified cloud cells is still outstanding.
Those cells must not be used as valid geometry. Splitting/merging/relaxation has
not been added or used to make the test pass.

Artifacts: build-workspace-validation/31585499/. Log, resources and raw results:
build-table-validation/workspace-31585499.out. Numerical and capacity options
remain opt-in, with existing defaults unchanged pending broader policy choices.
At the user's source-size query, code totals 3,919 physical lines (3,706 nonblank),
including 1,282 library-header lines; excludes docs, builds and data files.

## Offline scaling preparation and README refresh, 2026-10-27

Reviewed construction_scaling.cpp and run_scaling_gpudev.sh after resuming.
The CUDA construction_scaling target compiles successfully in
/tmp/voronoi-hydro-cuda. Local single-thread CPU smoke checks at 12^3 with one
repetition pass for both relative and legacy policies: zero validation errors,
volume error 1.11e-15, closure below 5.5e-16, centroid tensor below 1.5e-15.
These smoke timings are not CPU/GPU performance comparisons. Shell syntax and
plot Python syntax pass; plot rendering has not been validated.

README.md now describes the current interfaces, opt-in policies, validated
31585499 results, limitations and offline scaling workflow. Previous content
remains in docs/README-before-offline-scaling.md; no files/code paths deleted.
The scaling job is prepared but NOT SUBMITTED. Obtain approval before sbatch.
Run one A100/four CPU cores/15 minutes, 32^3 upward with a conservative memory
preflight and integer-side bisection after the first exclusion. Throughput units
are million particles/s for search and combined, million reconstructed cells/s
for geometry. Results append under build-gpu-scaling/<jobid>/scaling.csv.
No large-size GPU runtime results or true peak device-memory measurements yet.

## Offline scaling submitted: job 31589055

User approved the prepared job. Submitted run_scaling_gpudev.sh to gpudev;
Slurm returned 31589055. Last checked: PENDING, no log yet. Do not resubmit.
Read build-table-validation/scaling-offline-31589055.out and
build-gpu-scaling/31589055/scaling.csv after it starts. Results remain pending.

## Offline scaling completed: job 31589055

COMPLETED, exit 0, elapsed 2m38s. Nine PASS sizes: 32,48,64,96,128,160,192,194,195.
All validation_errors=0; largest 195^3 = 7,414,875 particles. At that size:
search 555.644875 ms (13.34463 Mparticles/s), reconstruction 1196.983629 ms
(6.19463 Mcells/s), combined 1751.969679 ms (4.23231 Mparticles/s).
Maximum sampled device usage 37,702,664,192 bytes; estimated arrays
35,446,477,508 bytes. 196^3 and larger tested cases returned intentional
SKIP_MEMORY/77 before allocation. Slurm marks those steps FAILED due to exit77;
the driver handles them and the overall job completed successfully. This is
the conservative budget ceiling, not the absolute GPU allocation limit.
Raw CSV: build-gpu-scaling/31589055/scaling.csv. README updated with results.

## Optional conservative early stop prepared (not GPU runtime validated)

User authorized optimizing construction before hydro and emphasized LUNA's
user-reported ~1000 million full hydro cell updates/s as a performance target.
This is not a directly equivalent workload or our measured hardware roofline.

PlaneWorkspace now has a third bool template argument (default false).
When enabled, reconstruct_row tracks an enclosing vertex radius during existing
classification and includes newly computed vertices in the bound. A candidate
with d^2 > 4 R^2 (1+1e-10) triggers a suffix-distance check; stopping requires
all remaining candidates outside the same bound and no coincidence. Unsorted
input therefore preserves correctness. Final reach/bounding-face certification
still runs. Retry forwards the flag to the 160-plane path. No persistent arrays
or default changes; original paths remain available.

CLI policy relative_early selects the option in table_mesh_check and
construction_scaling. construction_early_stress_check is a second target of the
existing stress source with the option enabled. Added late-cutter, late-zero
coincidence and radius-boundary fixtures with deliberately unsorted rows.
CPU stress checks all pass, as does final side12 scaling smoke validation.
Five distributions (lattice, disordered, regular, random, gradient) compare
against the old path: all ten geometry/moment comparisons pass, area component
errors zero. Files/logs: /tmp/voronoi-early-validation/. Jittered1728 vertex
plane tests fall 8,057,728 -> 620,608 (13x), coordinate reads 8,392,888 ->955,768,
created vertices stay109,221. Suffix distance reads are NOT counted in these
counters, and this reduction is NOT a GPU speedup claim.

CPU and CUDA targets compile. CUDA resource log:
/tmp/voronoi-early-cuda-build.log. An overlapping CPU build retained an older
stress object during editing; forced recompilation and reran final stress and
five-distribution comparisons afterward. Final tests include all three new cases.

validate_early_stop_gpudev.sh prepared, syntax checked, NOT SUBMITTED. Requests
one A100/four CPU cores/15min; builds CUDA targets, runs early stress, compares
five small distributions, then same-allocation paired relative/relative_early
scaling at sides32,128,195 with five repeats. Output build-early-stop/<jobid>/;
log build-table-validation/early-stop-<jobid>.out. Ask user before submission.
Actual roofline/DRAM traffic/occupancy profiling remains outstanding.

## Early-stop GPU comparison submitted: 31589308

User approved the prepared comparison job. sbatch validate_early_stop_gpudev.sh
returned job31589308. Do not resubmit. Track
build-table-validation/early-stop-31589308.out and
build-early-stop/31589308/scaling.csv. GPU runtime results pending.

## Early-stop GPU comparison complete: 31589308

COMPLETED exit0, elapsed4m13s. All16 stress cases, ten distribution runs,
ten output comparisons and six scaling rows pass (zero validation errors).
At195^3 baseline/early: reconstruction1196.385524/1000.872116 ms;
pipeline1750.654425/1556.377197 ms. Early rates7.408414 Mcells/s and
4.764189 Mparticles/s; ~1.20x construction/~1.12x combined throughput.
Search unchanged555.68/555.79 ms. Sampled device usage unchanged
37,702,664,192 bytes. At128^3 gains1.25x construction/1.16x combined;
32^3 gains~1.02x. Opt-in remains; no defaults changed. Hardware profiling
needed before assigning remaining bottleneck to bandwidth, occupancy or compute.
Results build-early-stop/31589308/scaling.csv; README updated.

## Coalesced-coordinate and fixed-point experiments, 2026-10-28

local_voronoi_coalesced.hpp adds LayoutLeft double[N,64,3] explicit coordinate
cache, 1536 persistent bytes/row. Core reconstruct_row accepts optional injected
VertexStorage object; reference call sites/defaults unchanged. Coalesced wrapper
uses 64-plane relative/early clipping and separate 160-plane private fallback.
Planes/topology still private: this is a coordinate-cache experiment only.
Side12 CPU compact geometry/volume/neighbor validation passes, zero errors.
construction_scaling policy relative_early_coalesced selects it; memory budget
includes added cache. Fixed32 policy relative_early_fixed32 reduces input positions
only, sorted positions remain double. Small CPU check passes. Classification mask
change passes early stress tests. docs/local-construction-literature.md records
primary-source search and precision caveats. No GPU performance claims yet.

First CUDA compile found NVCC cannot first-capture positions inside constexpr-if;
added explicit capture use before branch and started rebuild. Log
/tmp/voronoi-coalesced-cuda.log. compare_coalesced_gpudev.sh prepared (NOT SUBMITTED)
for baseline early/coalesced/fixed32 at32^3 and128^3, five repeats. Obtain user
approval before cluster submission. Existing profiler artifacts retained.

CUDA compile succeeded. User approved submission; compare_coalesced_gpudev.sh
submitted as job31590387. Confirmed PENDING after interrupted turn. Do not
resubmit. Log: build-table-validation/coalesced-31590387.out. Results:
build-gpu-scaling/31590387/comparison.csv. GPU measurements pending.

## Source organization, 2026-10-28

Moved headers into src/neighbour_search and src/voronoi_construct; all C++
examples/checks/benchmarks into src/test; shell scripts into scripts and Python
tools into python. All files retained. CMake paths and include interface updated;
Slurm scripts reference python/compare_faces.py. Submit scripts from repo root.
Job31590387 was already spooled by Slurm and builds via root CMake, so the
source relocation does not require resubmission. Historical handoff paths refer
to the former layout; consult README for current locations.

## Ruff, clang-tidy and extensive host CI, 2026-10-28

Added pyproject.toml (Ruff0.11.13 lint/import/format), .clang-tidy, host lint runner
scripts/lint_cpp.sh, CTest registration and .github/workflows/checks.yml. GitHub
push/PR/manual workflow builds pinned Kokkos5.0.2 Serial; ordinary and ASan/UBSan
matrix; full correctness suite; clang-tidy; Python lint/format. No pushes/commits
or cluster jobs submitted for this work. CPU executable names unchanged.

21 CTest entries: analytic, stress/early stress, sort, five table distributions,
three scaling policies, Python comparison contract, five multi-path equivalence
checks, two search-oracle modes, fixed-point contract. Each equivalence check
compares full cached relative, reduced early, legacy recomputed and team-sort
paths including geometry and hydro-moment CSVs. Python unit suite covers nine
contracts incl malformed/duplicate/nonfinite/dimension/missing/incorrect rows.
Fixed-point checks wrap boundaries, signed shortest deltas, quantization error,
exact-tick roundtrip and acknowledged sub-tick coincidence. All21 host tests pass.

Temporary lint tools installed via Python3.11 under /tmp/voronoi-lint-tools.
Ruff lint/format pass. Local clang-tidy uses Serial Kokkos to avoid clang18 vs
GCC15 OpenMP header incompatibility. NewDeleteLeaks reported false leak inside
Kokkos View allocation; exclude only that analyzer model and retain runtime
leak checks. Full clang-tidy run ongoing at /tmp/voronoi-clang-tidy-serial.log.
Three selected ASan/UBSan tests pass with detect_leaks=0. LeakSanitizer cannot
run under this traced environment (fatal ptrace message), so local leak checking
is NOT claimed; GitHub enables it. Logs /tmp/voronoi-sanitizer-tests.log (failed
LSan attempt), later console run confirms address/UB pass. CI not yet run remotely.
Plot_mesh now resolves data at repo root after relocation; fixed during tool audit.

Final clang-tidy pass completed successfully (exit0) for every src/test/*.cpp
translation unit against Serial Kokkos. No project diagnostics; external Kokkos
warnings suppressed. Ruff lint and format checks pass on all five Python files.
All21 ordinary CTest entries pass (20-suite run plus fixed-point contract).
Three selected ASan/UBSan tests pass with local LeakSanitizer disabled as noted.
GitHub workflow configured for the entire21-test suite in normal and sanitizer
builds, with leak detection enabled and UBSan halt_on_error. No remote CI run yet.

## Next memory step: bounded coalesced cache

Read completed job31590387: all6rowsPASS. At128^3 private/explicit fullcache
reconstruction279.073/259.596ms, pipeline428.557/409.385ms. Fullcacheadds3GiB.
Fixed32 input-only policy no gain:279.335ms reconstruction/428.699combined.

Added relative_early_batched using65536row coalesced cache,96MiBmaximum.
Wrapper supports smaller cache.extent0 thanN and launches sequential row ranges
on the supplied execution instance; row cache index is row-begin, fallback after
all chunks. Reference/fullcache policy remains available. Test policy uses257row
cache for1728cells (partial final chunk). All4scaling CTests pass. CUDA build
in progress at /tmp/voronoi-batched-cuda.log. Prepared scripts/compare_batched_gpudev.sh
for baseline/ bounded at32/128/195 sides, NOTSUBMITTED; ask approval first.
No claimed GPU speedup for bounded version yet. Next potential tuning is cache
batch size versus L2 working set; first obtain timings for bounded design.

CUDA bounded-cache compile succeeded. User approved submission; submitted
scripts/compare_batched_gpudev.sh as job31590674. Do not resubmit. Log:
build-table-validation/batched-31590674.out; CSV:
build-gpu-scaling/31590674/comparison.csv. GPU results pending.

## Bounded-cache comparison complete:31590674

COMPLETED exit0,2m01s; all6rowsPASS/zero validation errors. At128^3 baseline/
bounded reconstruction279.056/281.249ms, pipeline428.762/430.487ms. At195^3
992.616/1010.427ms, pipeline1548.380/1566.260ms. At32^3 6.121/6.071ms.
96MiBcache cap works, but bounded variant shows no throughput improvement;
195^3 reconstruction~1.8%slower. Keep experimental; do not promote default.
Fullcache7.5%gain at128^3 did not carry over to bounded chunks. Cause not yet
measured; kernel occupancy/grid size/cache layout may differ. Next experiment
should address remaining private planes/topology or cooperative cell execution,
with counters and correctness comparisons rather than assuming cache-only moves
solve the traffic issue. Raw build-gpu-scaling/31590674/comparison.csv.

## Cooperative clipping primitive started

src/voronoi_construct/cooperative_clip.hpp: classify_and_compact, TeamPolicy
one team per cell (32CUDAthreads,1host), TeamThreadRange vertex classification,
max-radius reduction and stable prefix-scan compaction. Caller-supplied distinct
input/outputbuffers; no allocationinside. Invalidcounts yieldkept_count=-1.
src/test/cooperative_clip_check.cpp validates everycount0..64 plusinvalidcount,
allinside/alloutside/plane-boundary, stable outputordering andradius. CPUcontract
passes. AddedCTestentry23. CUDAcompile started /tmp/voronoi-cooperative-cuda.log.
THIS IS NOT YET A FULL COOPERATIVE CELL RECONSTRUCTOR: topology/hole updates,
intersection creation, certification andmoments still needintegration. No GPUjob
submitted andnothroughputclaim. Existing productionkernel unchanged.

User repeated request to finish full warp-per-cell reconstruction. Existing
cooperative_clip.hpp is only a classification primitive and is not integrated
with topology updates. No full cooperative reconstructor has been implemented;
do not advertise it as a team-per-cell speed path. Required work remains a
shared per-team planes/vertices/hole state and coordinated clipping, then full
face geometry and retry validation.

## Team scratch storage bridge

CellClippingState owns arrays; reconstruct_row accepts injected descriptor.
team_cell_storage.hpp adds unmanaged scratch owners and lightweight pointer
arrays plus cached coordinates. edge_add_or_cancel now accepts generic arrays.
local_voronoi_shared.hpp contains full serial-in-team compact reconstruction
using this scratch state; it is explicitly a scratch-storage baseline, NOT a
cooperative warp implementation. It has not yet been instantiated or validated.
Cooperative classification remains standalone. Full integration still required.

## Full shared-workspace path instantiated and checked

construction_scaling relative_early_shared instantiates the complete compact
reconstruction using team_cell_storage.hpp/local_voronoi_shared.hpp. Host build
succeeds; side12 all1728cells pass volumes/centroids/areas/closure/tensor/neighbor
checks, validation_errors0. Added scaling policy to CTest. CUDA compilation
ongoing at /tmp/voronoi-shared-cuda.log. This remains single-thread-per-team
scratch baseline, NOT cooperative clipping, and does not yet retry plane overflow.
Next work is team-cooperative stages with shared counters/control and barriers.

## Cooperative full-cell validation and optimization plan (2026-10-28)

The full local_voronoi_team.hpp path now has public view-extent validation,
explicit classification-mask broadcast, and a separate 160-plane reference retry
for 64-plane overflow. Classification and intersection creation cooperate across
32 CUDA threads; topology edits, compaction and final face/moment traversal are
still on the leader. The cooperative loop currently processes all candidates
(no radius early exit); do not label it as an early-stop implementation.

Added cooperative table_mesh_check policy: compact team geometry is compared
with the 160-plane relative diagnostic reference, including face IDs, areas,
centroids, volume, invalid counts and incomplete-list certification. Five
particle distributions and cooperative scaling pass locally. Added independent
long-double stress-oracle target construction_team_stress_check; host test passes.
CPU and CUDA compilation succeeded for all three affected executables.

User optimization notes are incorporated in docs/gpu-optimization-plan.md:
measured profiling/locality, workspace, grid tuning, mixed precision, fusion,
and separately measured temporal reuse. Warm face lists alone cannot certify
absence of omitted sites within 2R. Keep cold-build and warm throughput distinct.

Prepared scripts/compare_cooperative_gpudev.sh: one A100 on gpudev, four CPUs,
15 minutes, stress and five-distribution checks, then paired full reconstruction
and search timings at 32^3/128^3/195^3 for reference early-stop, serial shared and
cooperative paths. NOT SUBMITTED; ask before submission. No measured GPU speedup
for the cooperative path yet. Full host regression and modified-source lint
results are recorded below when complete.

Validation complete: all 31 host CTests pass (125 seconds). clang-tidy on
modified table/stress translation units exits 0. CUDA builds for scaling,
table comparison and team stress all succeed. Script bash syntax and
git diff whitespace checks pass. GPU execution remains pending approval.

User approved cooperative GPU comparison. Submitted scripts/compare_cooperative_gpudev.sh
as job 31600967 on gpudev (one A100, 15 minutes). Do not resubmit.
Log: build-table-validation/team-31600967.out; results:
build-gpu-scaling/31600967/comparison.csv. Results pending.

User requested the same comparison on gpu1 as well, with 15 minutes maximum.
Submitted job 31601151 using --partition=gpu1 --time=00:15:00 --mem=32G,
one A100. Initial submission with default memory was rejected (no job created)
for exceeding the single-GPU host-memory share. gpudev job 31600967 retained.
GPU1 log: build-table-validation/team-31601151.out; CSV:
build-gpu-scaling/31601151/comparison.csv. Do not resubmit either job.

GPU comparison complete: 31601151 COMPLETED exit0 in 6m01s on ravg1024.
User requested cancellation of gpudev 31600967; cancellation confirmed.
All nine scaling rows PASS, zero validation errors; preceding geometry/stress
steps exited0. At195^3 reconstruction: reference896.964ms (8.267M cells/s),
serial shared2897.095ms, cooperative4023.174ms (1.843M cells/s).
Cooperative is 4.49x slower, not a production improvement. Search ~478ms
across all policies. At128^3 reference252.116ms vs cooperative1139.914ms.
Keep reference default. Cooperative processes all candidates whereas reference
has early exit; serial topology/face work and team overhead remain hypotheses
requiring profiling, not established attribution. Next: equalize certified
early-stop work and profile before further cooperative performance claims.

## Isolated memory experiment after cooperative slowdown

User agrees to measured, small changes; notes LUNA regular-grid capacity may
have been256^3 rather than512^3. Do not report an exact LUNA memory footprint.
Added docs/memory-budget.md: current payload ~4780B/particle, dominated by
2000B double candidate distances,1000B candidate IDs,1704B geometry outputs.

Opt-in RowSort::team_bitonic_recompute computes double candidate distances in
team scratch and writes only sorted IDs. Fresh configure skips the global
distance table for capacities<=2048; larger rows preserve insertion fallback
and allocate distances. Existing allocations/reference paths retained. Select
this mode before configure. Periodic recomputation can differ at last bits from
shifted-query search arithmetic; certify full geometry, not just row ordering.
No reconstruction algorithm or precision change. Benchmark policy
relative_early_scratch_sort subtracts2000B/particle from its preflight estimate.
Expected saving41.8%,13.81GiB at195^3. Still insufficient for256^3 on40GB.

Eight targeted CPU CTests pass: legacy sort, scratch allocation/order contract
(24 cases: periodic/nonperiodic, lanes1/4, growth/cap250/oversized2049,
wide/narrow grids and repeated builds), scaling and five geometry distributions.
Prepared scripts/compare_scratch_sort_gpu1.sh (one A100,32GiB host,15min):
contracts, five geometries, paired reference/scratch-sort32^3/128^3/195^3.
NOT SUBMITTED; ask approval. CUDA build/lint completion recorded below.

Scratch-sort validation: CUDA compilation succeeds for construction_scaling,
table_mesh_check and scratch_sort_check. clang-tidy on the new contract and
scaling translation units exits0; git diff --check and job script syntax pass.
GPU timing remains pending submission approval.

User approved scratch-sort GPU comparison. Submitted job31601918 via
scripts/compare_scratch_sort_gpu1.sh: gpu1,one A100,32GiB host,15min maximum.
Do not resubmit. Log: build-table-validation/scratch-sort-31601918.out;
results: build-gpu-scaling/31601918/comparison.csv. Results pending.

## GPU scratch-sort result: 31601918

Completed successfully on ravg1063 in3m53s. All contract/geometry steps exit0;
all6scaling rowsPASS with0validation errors. At195^3: search477.661->314.987ms,
reconstruction896.917->897.179ms (unchanged), pipeline1374.229->1211.418ms.
Search throughput15.523->23.540M particles/s; pipeline5.396->6.121M/s.
Observed device usage37,687,984,128->22,856,925,184bytes (~35.10->21.29GiB).
Array estimate35,446,477,508->20,616,727,508bytes (~41.8% reduction); observed
usage reduction~39.4% includes runtime/workspace overhead. At128^3 search
127.894->82.635ms and pipeline379.944->334.664ms; reconstruction unchanged.
This is a validated memory and search improvement, not a reconstruction speedup.
Reference remains available. Results build-gpu-scaling/31601918/comparison.csv.

## Focused reference instrumentation (user agreed after both paths deemed slow)

construction_scaling now accepts optional SEARCH_STAGE_CSV after timing CSV.
Separate post-timing passes use StageProfile; baseline search/reconstruction/
pipeline measurements remain unfenced beyond their original completion points.
Reports seven per-stage medians and instrumented wall median. Reconstructs once
again after profiling so the resulting candidate table gets geometry validation.
Added Python/CTest contract covering both sort policies, appended CSV schemas,
finite/nonnegative stage times and successful final geometry.

Prepared scripts/profile_reference_gpu1.sh: one A100,gpu1,32GiB host,15min max.
Collects uninstrumented paired timings and separate fenced search stages at
128^3/195^3. Builds with -lineinfo; captures one warmed main reference clipping
kernel at128^3 with LaunchStats,Occupancy,MemoryWorkloadAnalysis/tables,
WarpStateStats,SourceCounters. Saves raw counters and source report separately
from throughput CSV. Sections verified with local ncu --list-sections. Retry
cost remains in throughput; counter capture isolates the main kernel. This is
NOT phase-by-phase reconstruction wall timing. NOT SUBMITTED; ask approval.

Reference instrumentation validated: three targeted host CTests pass, CUDA
compilation succeeds, clang-tidy exits0 (only suppressed non-user warnings),
Ruff lint/format, shell syntax and git diff whitespace checks pass.
Profiling job remains prepared and unsubmitted pending user approval.

User approved reference profiling. Submitted job31603868 via
scripts/profile_reference_gpu1.sh: gpu1,one A100,32GiB host,15min maximum.
Do not resubmit. Log: build-table-validation/reference-profile-31603868.out;
reports and CSVs: build-roofline/31603868/. Results pending.

## Profiling job31603868: useful search results, Nsight resource failure

Four uninstrumented/stage benchmark runs PASS with0validation errors.
At195^3 scratch-sort stage medians: key/count0.187ms,scan0.045ms,
scatter0.152ms,cell order/gather4.436ms,candidate fill116.155ms,
nearest ordering194.031ms,overflow check0.012ms; instrumented total314.952ms.
Sorting ~62%, candidate collection ~37%; grid setup ~1.5%. At128^3 sorting
56.933ms vs candidate fill24.648ms (total82.846ms). Reconstruction ~899ms
at195^3, unchanged. Data: build-roofline/31603868/search-stages.csv and timings.csv.

Nsight step failed exit9: driver profiling resource unavailable; log suggests
another profiler/DCGM as possible cause, not confirmed. No new reconstruction
hardware counters captured. Job FAILED overall after2m06s on ravg1075; ordinary
benchmark data remains valid. Do not confuse profiler failure with geometry
failure. No retry submitted. Next search target is sorting rather than grid
build; reconstruction source-counter capture remains outstanding.

## User authorized experiments1(no full candidate sort) and2(Morton locality)

Implemented opt-in Morton GRID CELL BLOCK ordering in neighbor_search.hpp,
with cached Cartesian-cell-to-Morton-rank map, complete query coverage, per-cell
Morton ranges instead of merged Cartesian z ranges. Input-ID ordering inside
cells retained. Configure-time host map construction is outside steady-state
benchmarks;4bytes/gridcell added to device payload estimate. No default change.

Benchmark modes relative_early_unsorted,relative_early_morton,
relative_early_morton_unsorted plus existing relative_early_scratch_sort baseline.
Raw unsorted failed:568/1728invalid cells in scaling lattice; intermediate
vertex/plane growth and regular-grid coplanar degeneracies. Preserve this finding.
Added local_voronoi_order_retry.hpp: sort failed rows IN PLACE, retry full160plane
reference, compact path also retries tiny order-sensitive faces (area^2 ratio
<=1e-24). No face removal/tolerance relaxation. Every timed scaling reconstruction
starts after a fresh search fill and includes recovery. Diagnostic table oracle
is refreshed after compact recovery changes candidate ordering.

Added layout_search_check.cpp independent neighbour oracle and
python/check_layout_geometry.py comparisons against canonical particle IDs.
Four nonregular distribution comparisons pass with failure recovery; regular
retest after tiny-face trigger is pending. Five reference regressions pass.
CUDA initial retry build and clang-tidy initially passed; final edit revalidation
pending. See docs/layout-experiments.md for limitations and timing semantics.
Prepared scripts/compare_layout_gpu1.sh (oneA100,32GiBhost,15min); NOTSUBMITTED.
Ask approval after final validation. Preserve original algorithms/artifacts.

Layout experiments final validation complete: all14 targeted CPU CTests pass
(153.65s), including all5cross-layout geometry distributions and reference
regressions. CUDA build succeeds for construction_scaling,table_mesh_check,
layout_search_check. Final clang-tidy exits0; Ruff and script syntax/whitespace
checks pass. Regular topology now matches after tiny-face reference-order retry.
No GPU job submitted for these experiments. Prepared compare_layout_gpu1.sh
compares baseline sorted Cartesian,unsorted-first,Morton sorted,Morton unsorted
at32/128/195,5repeats,with full recovery included. Ask user approval before sbatch.

User approved four-way layout comparison. Submitted job31612191 via
scripts/compare_layout_gpu1.sh: gpu1,one A100,32GiB host,15min maximum.
Do not resubmit. Log: build-table-validation/layout-31612191.out;
results: build-gpu-scaling/31612191/comparison.csv and search-stages.csv.
Results pending.

User approved focused reconstruction profiling. Submitted job 31624020 via
scripts/profile_reconstruction_gpudev.sh: gpudev, one A100, 15-minute limit.
Confirmed RUNNING on ravg1002; initial configure completed, build underway.
Log: build-table-validation/clip-profile-31624020.out.
Artifacts: build-roofline/31624020/. Counter preflight and captures pending.
Do not duplicate or automatically resubmit this job.

Job 31624020 FAILED after 57 seconds: hardware counter access succeeded, but
check_ncu_capture.py assumed long-format CSV whereas Nsight raw export is wide
with a units row. Fixed parser to accept both formats, retaining strict checks.
Seven parser tests and Ruff pass; saved actual preflight.csv now validates.
Recovered small 32^3 capture: 6.085696 ms kernel duration, 886.045312 Mbyte DRAM
reads, 100 registers/thread, 8672-byte stack. This is a profiler capture, not
baseline throughput. Full hardware/source captures did not execute.
No resubmission yet; ask approval for another gpudev job (15-minute maximum).

User approved corrected profiling resubmission. Submitted job 31624143 via
scripts/profile_reconstruction_gpudev.sh: gpudev, one A100, 15-minute limit.
Uses corrected wide/long CSV parser validated against job 31624020 capture.
Log: build-table-validation/clip-profile-31624143.out.
Artifacts: build-roofline/31624143/. Results pending; do not duplicate submission.

Job 31624143 completed baseline and verified hardware capture; FAILED at source
capture after 62 seconds because --source-folders requires --import-source on.
Corrected script to include that option; bash syntax passes. No resubmission.
128^3 baseline: search97.956ms, reconstruction286.955ms, pipeline384.499ms,
all geometry validation passed. Profile main kernel287.044ms, DRAM reads110.567GB,
writes46.273GB, L2traffic503.308GB, local load sectors11450311675/store4800448203,
occupancy23.376%, eligiblewarps0.233575, longscoreboard67.170%, active threads
per instruction9.54. About74.8KB DRAM traffic/cell,546GB/s aggregate DRAM.
Work counts/cell:23.011 candidates processed,154.432 suffix tests,359.662 vertex
classifications,63.311 intersections,293.770 edge comparisons,453.647 face seed
checks,1230.835 face link checks. Counts are not timings; source hotspots pending.
Hardware evidence supports local workspace traffic/latency as leading target,
not a claim of saturating DRAM bandwidth. All artifacts retained in
build-roofline/31624143/. Obtain approval before another cluster submission.

Implemented opt-in FloatCachedVertexPositions and scaling policy
relative_early_float_cache. FP32 seed-relative cache halves vertex cache from
1536 to768 bytes; double planes retained, uncertain clipping tests recompute
in double, radius inflated for cache rounding, certification/output recompute
double intersections. Reference path preserved. No GPU speed claim.
Added 20 mixed-cache analytic cases and float_cache_scaling_smoke (12^3).
Initial analytic/periodic CPU checks pass; CUDA analytic compilation passes.
Final rebuild after profiling-counter bookkeeping/label adjustment underway;
logs /tmp/voronoi-float-build.log and /tmp/voronoi-float-tests.log.
Details: docs/mixed-precision-experiment.md. Integer particle storage and FP16
remain unimplemented. Irregular-distribution testing needed before promotion.
Source-only script scripts/profile_source_gpudev.sh prepared with corrected
--import-source on; local ncu --list-metrics option validation passes.
No new cluster job submitted. Ask approval before submission.
Final CPU rebuild completed; geometry_analytic and float_cache_scaling_smoke
both pass after the final changes. git diff --check passes.

User approved source-only profiling job. Submitted 31624563 using
scripts/profile_source_gpudev.sh: gpudev, one A100, 15-minute maximum.
Profiles the existing double-reference executable from job31624143, not the
new FP32 cache, to complete source attribution of the measured baseline.
Log: build-table-validation/source-profile-31624563.out.
Artifacts: build-roofline/31624563/. Do not duplicate or automatically resubmit.

Source job31624563 COMPLETED52s, capture gates passed. Critical interpretation:
old executable imported newer FP32-edited source; displayed source text is
misaligned. Reversed known FP32 edits into separate reconstructed baseline
snapshot within result directory. Leading baseline line441 face adjacency search
754657 samples, line442 edge match182699, line310 suffix position gather635510,
line225 hole edge matching258666. Not walltime percentages; inline accessor
attribution duplicates counts. See build-roofline/31624563/findings.md.
Next target should include explicit face adjacency / removing repeated scans;
FP32 cache alone does not address the largest source hotspots.

Implemented opt-in PlaneWorkspace fourth flag direct_face_links and benchmark
relative_early_face_links. Byte lookup per face replaces repeated all-vertex
adjacency scans; preserves reference order, 64byte usual/160byte retry map.
Retry policy propagation included. Analytic108->24 face-link checks verified.
Four CPU CTests pass: geometry_analytic, float_cache_scaling_smoke,
face_links_scaling_smoke, geometry_face_links_stress (includes irregular inputs).
CUDA analytic target build in /tmp/voronoi-links-cuda.log. Prepared (NOT submitted)
scripts/compare_workspace_gpudev.sh:gpudev1A10015min, baseline/float/directlinks,
32/128/195^3, five repeats plus GPU geometry checks. Ask approval before sbatch.

User approved workspace comparison. Submitted job31624791 using
scripts/compare_workspace_gpudev.sh:gpudev,one A100,15-minute maximum.
Runs GPU analytic and direct-face-link stress checks, then reference/FP32cache/
directlinks at32^3,128^3,195^3, five repeats. Log:
build-table-validation/workspace-31624791.out. Results:
build-gpu-scaling/31624791/comparison.csv. Do not duplicate submission.
CUDA analytic compilation for directlinks completed successfully before submission.

Integer reconstruction implemented (user explicitly requested integer positions,
not FP32 vertex caching). coordinate_displacement overload on FixedPointPositions32
subtracts unsigned coordinates before double conversion for unit periodic box.
Both clipping candidate and early-stop suffix loads use it. New benchmark mode
relative_early_integer32 consumes slot-ordered uint32 positions, keeping double
planes/vertices/output. Search stays double; additional integer gather included
in search/pipeline timing. Net persistent arrays unchanged versus double-input
baseline (input -12bytes/particle, integer slots +12). No GPU speed claim.
New integer_positions_check compares complete geometry exactly with decoded
quantized input: regular/jittered/periodically shifted125particles plus boundary
and half-box displacement tests. CPU/CUDA builds underway. Documentation:
docs/integer-reconstruction.md. No new cluster job submitted.
Prior GPU job31624791 final195^3: reference1020.510ms, FP32cache1418.670ms,
directlinks862.361ms reconstruction; all geometry validations PASS.
Integer initial validation: five CPU CTests passed, CUDA integer_positions_check
compiled successfully. Added matched relative_early_integer32_decoded control
(same quantized input/gather/search; reconstruction reads doubles). Final control
rebuild running /tmp/voronoi-integer-control-build.log, tests afterwards in
/tmp/voronoi-integer-tests.log. Prepared scripts/compare_integer_gpudev.sh:
one gpudev A100,15min, integer geometry test then original/decoded/integer
32/128/195^3 five-repeat comparison. NOT SUBMITTED; ask approval first.
