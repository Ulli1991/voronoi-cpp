# Construction memory budget

2026-10-28. User reports LUNA fits a regular grid of at least roughly 256^3
(the earlier 512^3 recollection is uncertain). Do not assume identical fields,
precision or GPU capacity; use this as motivation to budget bytes per particle.

The current scaling executable reserves 250 candidates and 32 faces per row.
Its array estimate, excluding CUDA runtime/workspace and allocator overhead, is:

| Allocation | Bytes/particle | GiB at 256^3 |
|---|---:|---:|
| Candidate distances, double[250] | 2000 | 31.25 |
| Candidate IDs, int[250] | 1000 | 15.625 |
| Face geometry and cell outputs | 1704 | 26.625 |
| Input, sorted positions/radii, metadata | 76 | 1.1875 |
| Dense grid counts/offsets | size dependent | additional |
| Subtotal excluding dense grid | 4780 | 74.6875 |

First isolated experiment: `team_bitonic_recompute` / scaling policy
`relative_early_scratch_sort`. Recompute double distances from positions in
team scratch, preserve candidate IDs, sorting/tie-breaking and the existing
reconstructor. Fresh searches with capacity <=2048 allocate no distance table;
larger capacities retain the insertion fallback and distance allocation.
Existing allocations are retained when changing modes. Periodic distance
recomputation can differ in last bits from the search's shifted-query arithmetic;
full geometry checks are required as well as sorted-ID tests with exact ties.

Expected fresh-allocation subtotal: 2780 bytes/particle, a 41.8% reduction.
At195^3 this saves 13.81 GiB; at256^3 it still needs 43.44 GiB plus dense grid
and runtime overhead. This change alone does NOT make 256^3 fit on a 40GB GPU.
No GPU throughput claim until paired measurements pass.

Next memory decisions must separately budget candidate IDs and geometry output.
Batched/streamed construction or fused search avoids an all-particle candidate
table. Compact face storage or streaming consumption can reduce output padding.
A regular grid has implicit connectivity; arbitrary Voronoi cells require more
information. Keep that distinction explicit without accepting avoidable overhead.

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
