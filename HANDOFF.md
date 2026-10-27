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
