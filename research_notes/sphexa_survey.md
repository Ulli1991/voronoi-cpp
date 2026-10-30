# SPH-EXA / Cornerstone survey: what is useful for the GPU Voronoi mesh

Date: 2026-10-29. Read-only research: nothing was built or run.

**Source.** `/raven/u/uli/sphexa`, `git clone --depth 50` of https://github.com/sphexa-org/sphexa at commit
`a8e6bfa` (16 Oct 2026). There is one submodule, `extern/grackle/grackle_repo` (cooling chemistry). It was not fetched
because it is not relevant here.

**Paths.** All paths are relative to `/raven/u/uli/sphexa`. `cstone/` means `domain/include/cstone/`.

**Tags.** (V) = verified in the source or in a paper, with the location given. (I) = my inference, not measured.

**Our baseline** (from `status_2026-10-29.md`, `kokkos_cuda_features_audit.md`):

| | jittered | uniform random | clustered |
|---|---|---|---|
| Full rebuild, A100, 2.1 M cells | 15-18 M cells/s | ~10 M cells/s | 0.9 M cells/s |

On clustered input each cell runs 169 candidate tests and 37 cuts, and dense cell-list cells overflow the 32-entry
candidate buffer. The clipping kernel is divergence-bound: 5-16 of 32 lanes are active.

---

## 0. What SPH-EXA is, in one paragraph (V)

- **Language and portability.** C++20 plus plain CUDA. There is no Kokkos anywhere: `grep Kokkos` finds nothing.
  - HIP is supported by running `hipify-perl` on the `.cu`/`.cuh` files (`domain/README.md`, "Compilation").
  - Warp width is a compile-time constant, 32 on NVIDIA and 64 on AMD (`cstone/cuda/gpu_config.cuh:29-43`). Lane masks
    are `uint32_t` or `uint64_t` (`ThreadMask`).
  - It uses no CUDA graphs: `grep cudaGraph` finds nothing. Work runs on one stream per `execution::Gpu` object, with
    `cudaMallocAsync` (`cstone/cuda/memory.cuh:72`).
- **Domain library (cstone).** Builds an octree from a sorted space-filling-curve key array, decomposes the domain,
  discovers halos and exchanges them. It is the "Cornerstone" library of Keller et al., PASC'23.
- **Physics.**
  - `sph/` holds the SPH kernels.
  - `ryoanji/` is a Barnes-Hut gravity solver with a Bonsai-style warp traversal.
  - The neighbour loops (`cstone/traversal/ijloop/`) are by Felix Thaler (CSCS). They are a GROMACS-style cluster-pair
    neighbour list ("supercluster") built from octree traversal.
- **Space-filling curve.** The default is Hilbert, 64-bit (`cstone/sfc/sfc.hpp:40` `using SfcKind = HilbertKey<IntegerType>`;
  `sph/include/sph/types.hpp:41` `using KeyType = uint64_t`).

---

## 1. Neighbour search on the GPU: the Cornerstone octree

### 1.1 Octree construction from the space-filling curve (V)

**Format.** A "cornerstone" tree is a sorted array of SFC keys, one per leaf (`cstone/tree/csarray.hpp:11-28`).
Its invariants:
- "code sequence contains code 0 and the maximum code";
- "difference between consecutive elements must be a power of 8";
- "only leaf nodes are stored".

So the leaves tile all of space with no holes, and leaf `i` is the key range `[tree[i], tree[i+1])`.

**Per-leaf particle counts.** Each leaf does two `lower_bound`s into the sorted particle keys. The search is
accelerated with a guess plus an exponential search (`csarray.hpp:100-135`, `findSearchBounds`). GPU kernel:
`computeNodeCountsKernel` (`csarray_gpu.cu:36`).

**Rebalance step.** One thread per leaf (`calculateNodeOp`, `csarray.hpp:268-290`):
- merge if the 8 siblings together hold `<= bucketSize`;
- split into 8 / 64 / 512 / 4096 if `count > bucketSize * {1, 8, 64, 512}`. The quote:
  `if (counts[nodeIdx] > bucketSize * 512 && level + 3 < maxTreeLevel) return 4096;`

An exclusive scan over the operations gives the new leaf positions, and `processNodes` writes the new keys
(`csarray_gpu.cu:151-205`).

**Iteration.** Repeat until nothing changes (`computeOctree`, `csarray.hpp:430-440`). Between time steps the previous
tree is the starting guess, so usually one pass is enough. `domain/test/performance/octree.cu:78-94` times
"build time from scratch" against "build time with guess".

**Internal nodes.** "cornerstone octree (leaves) -> internal binary radix tree -> internal octree" … "fully parallel
and non-recursive and non-iterative" (`cstone/tree/octree.hpp:16-20`). The build (`octree_gpu.cu:41-166`):
- `createUnsortedLayout`, a Karras-style prefix test: an internal node exists where `commonPrefix % 3 == 0`;
- a radix sort of the Warren-Salmon placeholder-bit prefixes;
- `linkTree`, which sets `childOffsets` and `parents` by binary search.

**Geometry.** Node boxes are geometric (from the keys), not particle bounding boxes
(`cstone/domain/domain.hpp:389-392`, `geoCentersAcc()`, `geoSizesAcc()`).

**Production settings.** `bucketSizeFocus = 64` (`main/src/sphexa/sphexa.cpp:151`). The global decomposition tree uses
`max(64, N/(100*numRanks))`: "~100 global nodes per rank" (`sphexa.cpp:152-153`).

### 1.2 Measured cost (V, Keller et al. PASC'23, arXiv:2307.06345, Table 1)

Setup: Gaussian (clustered) particle distribution, N_crit = 64, times in ms.

| A100 | 1 M | 32 M | 512 M |
|---|---|---|---|
| 64-bit Morton keys | 0.018 | 0.31 | 7.9 |
| 64-bit Hilbert keys | 0.084 | 1.9 | 30 |
| Radix sort | 0.483 | 5.0 | 73 |
| Leaf update (one rebalance pass) | 0.11 | 0.51 | 2.5 |
| Internal nodes | 0.19 | 0.65 | 5.4 |

On the claim for clustered input, the paper says the faster construction "outweighs the penalty of a slightly larger
tree, even for highly clustered distributions" (abstract and §4, as quoted by arXiv HTML).

**Cost for our 2.1 M cells (I).** About 0.1 ms for keys, 0.6-0.8 ms for the sort, about 0.15 ms per leaf pass, and
about 0.25 ms for internal nodes: **roughly 1-1.5 ms in total** with a previous tree as the guess, and about 2-3 ms from
scratch (3-5 passes).

At our clustered rate of 0.9 M cells/s a rebuild takes about 2.3 s, so the tree is noise there. It is also small next to
the about 11 ms per-step pipeline (repair plus geometry). The per-step path needs no neighbour search anyway, because
the tets carry the topology. The tree is only needed for rebuilds and, later, for decomposing the domain.

### 1.3 How neighbour/halo searches traverse the tree on the GPU (V)

There are three traversal styles.

**(a) Stackless depth-first walk, one thread per query.** `singleTraversal` (`cstone/traversal/traversal.hpp:26-69`)
walks with `childOffsets` and `parents` only, with no stack:
- descend if `continuationCriterion(node)`;
- otherwise move to the next sibling;
- otherwise go up to the parent with `backtrack = true`.

For a fixed radius, `findNeighbors` (`cstone/findneighbors.hpp`) uses as the criterion "minimum distance from the
query point to the node box < 2h", and as the leaf action a scan of the particles of `layout[leaf]..layout[leaf+1]`.
The halo search (`cstone/traversal/collisions_gpu.cu:24-55`, `findHalosKernel`) is the same walk, one thread per leaf.

**(b) Warp-cooperative walk for a group of 32 particles: the supercluster build.**
`collectNeighborJClusters` (`cstone/traversal/ijloop/gpu_superclusternblist/build.cuh:324-469`):
- **One warp per group.** A warp owns 32 SFC-consecutive particles and computes their bounding box (`:258-289`).
- **Node test.** The warp runs the same `singleTraversal`, but the node test is
  `return bool(anySync(overlaps));` (`:372`). The walk is therefore warp-uniform: no divergence in control flow, and
  every lane tests its own particle against the node.
- **Leaf work.** At a leaf, the leaf's "j-clusters" (4 or 8 particles each) are tested 32 at a time, one per lane,
  against the group box (`:386-411`). Each surviving cluster is then tested particle by particle, and a 32-bit
  "which i-cluster × which j-particle" interaction mask is formed with `warpBitwiseOr` (`:413-438`).
- **Output.** A list of j-cluster indices plus bitmasks, compressed with nibble delta encoding
  (`compressneighbors.cuh`).
- **Scheduling.** Persistent warps take groups from an atomic counter: `if (laneIdx == 0) index = atomicAdd(&globalBuildData->index, 1);`
  (`build.cuh:559`). Launch size is `smCount * 40 warps / 2` (`:613-617`).
- **Load balance.** After the build the superclusters are sorted by descending neighbour count:
  "sort supercluster array by descending neighbor count for load balancing (schedule large work packages first)"
  (`gpu_superclusternblist.cuh:327-328`).

**(c) Breadth-first warp traversal with compacted queues (Ryoanji, gravity).**
`traverseWarp` (`ryoanji/src/ryoanji/nbody/traversal_gpu.cu:168-338`), "inspired by the original Bonsai
implementation" (`:11`). The warp holds a per-level queue of nodes in a ring buffer in global memory (`cellQueue`). In
each pass:
- 4 nodes are tested per step, 8 lanes each (`spreadSeg8`, `:202`);
- nodes to split are appended by a warp exclusive scan (`:216-221`);
- accepted far nodes are compacted into a 32-entry `approxQueue` (`streamCompact`, `:232-245`).

Leaf particles come out as a flat stream through a segmented scan (`inclusiveSegscanInt(tempQueue[laneIdx], prevBodyIdx)`,
`:264`). They are buffered in a 32-entry `bodyQueue` and evaluated only when **all 32 lanes have a body**
(`:270-300`). This is the key anti-divergence device: irregular leaf sizes are turned into full 32-wide batches.

**Target groups.** `groupSplitsKernel` (`cstone/traversal/groups_gpu.cuh:149-220`) cuts the SFC order into groups of
up to 32 or 64 particles. It splits wherever two consecutive particles are farther apart than
`tolFactor * cbrt(smallest leaf volume)` (`findSplits`, `:38-78`, a ballot over shuffled neighbour distances), with
`tolFactor = 2` (`sph/include/sph/groups.hpp:45`). On clustered input the groups therefore stay spatially compact; a
group never straddles a void.

### 1.4 How clustered input is handled (V, plus I)

- **(V)** Adaptivity comes entirely from the bucket rule. Leaves hold at most `bucketSize` (64) particles at any depth,
  up to 21 levels with 64-bit keys. A cluster gets deep, small leaves and a void gets one big leaf. There is no
  per-density tuning.
- **(V)** The neighbour list has a fixed capacity `ncmax = 32*ngmax/2` j-clusters (`sph/neighborhood_gpu.hpp:73-75`).
  When it overflows the build only warns: "WARNING: overflow in cluster neighbor list … Missing neighbors!"
  (`build.cuh:646`). We cannot accept that; see §5.
- **(I) Why our clustered case is slow.** One grid cell size gives two problems:
  - in a cluster, a grid cell holds tens to hundreds of points, so the initial stencil ring already yields far more
    candidates than needed (169 tests for 37 cuts), and the 32-slot buffer overflows;
  - in a void, the ring has to be walked out many cells.

  An octree with leaves of ≤ 8-16 points gives candidate sets matched to the local density in both regions.

### 1.5 Could it replace or complement our cell list? (I)

**Replace.** For full rebuilds, a cornerstone tree with a small bucket (8-16) instead of the uniform cell list. All
pieces are a few hundred lines of Kokkos: Morton or Hilbert keys, a sort (we already sort), the leaf count and
rebalance kernels, scans, and the Karras-style internal-node build.

**But nearest-first matters.** Our clipping wants candidates in **increasing distance** so the security radius shrinks
early. cstone's walk is a plain depth-first walk with a fixed radius; it has no nearest-first order. Two ways to adapt
it:
1. **Two phases.** First seed with the points of the cell's own leaf and its touching leaves. They are close, about 16-64
   points, which fits our sorted 32-batch. Then do one tree walk with the final radius `2*sqrt(radius2)` to catch the
   rest. The radius only shrinks, so the walk can re-test `minDist(node) < 2R` with the current R as it goes.
2. **Best-first.** A short priority list of nodes per warp (a k-NN-style walk). This is more complex and probably not
   needed.

**Complement.** Keep the uniform grid for jittered or uniform input, where it is already at 15-18 M cells/s, and switch
to the tree when the occupancy histogram is skewed. For example, a maximum cell count above 4× the mean, or our
existing overflow counter, would trigger the switch.

**Cost per step.** About 1-1.5 ms only when a rebuild is triggered. Nothing in the per-step tet pipeline.

---

## 2. GPU kernel engineering patterns (V unless marked)

| Pattern | Where | Note for us |
|---|---|---|
| **Persistent warps with an atomic work counter** | `gpu_alwaystraverse.cuh:60-65`, `build.cuh:556-561`, `traversal_gpu.cu:420` | Their default for all irregular kernels. Grid = `smCount × 40 warps`. This is exactly item 1 of our audit; it is in production in SPH-EXA. |
| **Sort work by cost, largest first** | `gpu_superclusternblist.cuh:327-328` (`thrust::stable_sort` by neighbour count) | Cheap tail-balancing. No measured gain is published in the repo. |
| **Warp-uniform traversal with `anySync`** | `build.cuh:372` | One tree walk per 32 queries. Control flow stays uniform. |
| **Compacted 32-wide work queues** (`streamCompact`, segmented scan, top-up with `shflUp`/`shflDown`) | `traversal_gpu.cu:232-300`, primitives in `cstone/primitives/warpscan.cuh` | Directly reusable for candidate streams: never test fewer than 32 candidates at once. |
| **Target groups by SFC with distance splits** | `groups_gpu.cuh:38-78,149-220` | Groups of ≤ 32 nearby cells for shared candidate fetch. |
| **Cluster-pair layout 8×4 per warp** (iSize = 8, jSize = 4, supercluster = 32) | `sph/neighborhood_gpu.hpp:38-40`; table in `gpu_superclusternblist.cuh:38-53` | Lane = (i in cluster, j in cluster). i data is staged in shared memory once per supercluster (`loop.cuh:245-281`); j data is loaded once per lane. |
| **Interaction bitmasks** per (j-cluster, i-cluster) | `build.cuh:419-455`, `loop.cuh:374-380` | Skip empty sub-tiles with no branch per pair. |
| **Neighbour index compression** (nibble delta) | `ijloop/compressneighbors.cuh` | Lowers memory for 100-neighbour lists. Of little value for us at about 15 faces per cell. |
| **AoS vs SoA shared layout, switched by field count** | `loop.cuh:236-243`: "AoS layout suffers from bank conflicts … cutoff based on measurements on GH200", `useSoaSharedLayout = tuple_size > 20` | A rule of thumb for the MFV face kernel. |
| **Warp-uniform hints** | `loop.cuh:348-350`: `shflSync(x, 0)` "Show NVCC that variables are warp-uniform", after GROMACS MR 5715 | Cheap trick for our per-cell loop bounds. |
| **Oversized virtual allocation for an output of unknown size** | `memory.cuh:77-85` (`cudaMallocManaged` as reserve-only), `gpu_superclusternblist.cuh:301-304`, with `atomicAdd` "allocation" per warp (`build.cuh:219`) | An option for our ring and face buffers instead of the count-then-fill passes. |
| **`cuda::discard_memory` on scratch** | `build.cuh:576-578` | Avoids L2 write-backs of dead scratch. Minor. |
| **`__grid_constant__`** on tree and box arguments | `gpu_alwaystraverse.cuh:42-43`, `build.cuh:520-521` | Not reachable through Kokkos lambdas; skip. |
| **Two neighbour paths kept side by side** | `neighborhood_gpu.hpp:62-78`: no-list "always traverse" (lane per particle, `ngmax` buffer per lane) vs cluster list | They keep the simple path as a fallback (`--disableNeighborLists`). |

**Divergence in pair loops.** They avoid it structurally:
- a fixed 8×4 tile per warp;
- an overlap bitmask decides whole sub-tiles;
- inside a tile the only branch is `if (iClose | jClose)` around the interaction (`loop.cuh:390-421`).

For SPH, the interaction is cheap and the same for all pairs, so the cost of a lane-wasting test is low. Our clipping
is the opposite: the interaction (topology update) is the expensive part and has a variable cost.

**Not present.** CUDA graphs, cooperative groups `tiled_partition`, dynamic parallelism, and Kokkos.

---

## 3. Hydro on the GPU

### 3.1 How pair interactions are computed (V)

**Symmetric (each pair once).** The default when all particles are active (`neighborhood_gpu.hpp:75`,
`ClusteredNeighborhoodBuilder<true>`). Each pair is evaluated once and gives an i-result and a j-result:
- i-results are kept in registers per lane and reduced along j by shuffle (`storeTupleISum`, `loop.cuh:76-125`);
- j-results are reduced along i by shuffle and then added with **atomics** to global memory (`storeTupleJSum`,
  `loop.cuh:135-168`, `atomicUpdatePtr`).

The cost of the atomics is extra passes:
- `initResult` before the loop;
- a separate `applyPostamble` pass after it (`gpu_superclusternblist.cuh:169-187`);
- a stream synchronize.

To balance which side owns a pair, `includeNbSymmetric` uses a checkerboard of 32-particle blocks (`build.cuh:183-190`).

**Gather (each pair twice).** Used with sub-groups, i.e. block or individual time steps: `subgroups ->
ClusteredNeighborhoodBuilder<false>` (`neighborhood_gpu.hpp:72-73`). The symmetric list explicitly "does not support
local time stepping" (`:55`). No atomics; every i writes its own result.

**Build once per step, reuse.** The list is built once per step and reused by 5-6 loops: XMass, VE, IAD/divv/curlv, AV
switches, momentum-energy (`main/src/propagator/ve_hydro.hpp:141-186`).

**Mixed precision.** The benchmark driver runs double coordinates with double or float values, kernel table or direct
evaluation (`domain/test/performance/sph_density_driver.cu:174-183`). It reports "Giga Particle Updates / s"
(`nbbenchmark.cuh:273`). The repo stores no results.

### 3.2 Throughput (V)

| Source | Rate | Setup |
|---|---|---|
| Kubli et al. 2025, arXiv:2510.26663 §II | "1×10^7 to 2.5×10^7 particle timesteps per GPU per second" | GH200 (Alps), SPH plus Barnes-Hut self-gravity, about 100 neighbours, polytropic EOS with AV. Production TDE runs of 10^6-10^10 particles. |
| Keller et al. PASC'23 | 6.7 Tflop/s FP64 per MI250X GCD for gravity | 8 × 10^12 particles on 16416 GCDs of LUMI-G |
| Kubli et al. 2025 | Scaling up to 2.8 × 10^12 particles | Alps and LUMI-G |

The repo contains no A100 SPH number; I found none published.

Comparison points from our notes, on an A100:
- PROTEUS Sedov: 5 M updates/s;
- AREPO-WD: 0.59 M updates/s;
- LUNA fixed grid: 1.1 G updates/s.

**(I)** SPH-EXA's 10-25 M updates/s per GH200 is the bar a GPU Lagrangian code reaches with about 100 neighbours. MFV
on Voronoi faces has about 15 faces per cell but needs gradients and a Riemann solver per face. Our "gate"
(≥ 200 M cells/s for mesh upkeep) leaves room for the whole step to reach 10-20 M cells/s if the hydro kernels are as
tight as SPH-EXA's.

### 3.3 Lessons for our MFV face-flux kernel (I, each from a V pattern)

1. **Face-centric, each face once, two writes.** This is SPH-EXA's symmetric mode. Faces are explicit for us (owner
   sorted), so a thread or lane per face computes the flux once and adds ±F to both cells.
   - Atomics: about 15/2 faces per cell × 5 conserved quantities × 2 atomics ≈ 75 FP64 atomics per cell.
   - Alternative: write per-face fluxes, then a segmented reduction over owner-sorted faces plus one over
     neighbour-sorted faces. That is our audit item 6, with no atomics, deterministic, and better for reproducibility.

   SPH-EXA chose atomics plus an extra init and postamble pass. We already know the faces' owners, so the segmented
   reduction is cheap.
2. **Gather mode for individual time steps.** If we adopt hierarchical time steps (AREPO-style), use per-cell gather
   (each face twice, no atomics) for active cells only. This mirrors SPH-EXA's switch to `ClusteredNeighborhoodBuilder<false>`
   with sub-groups. Plan both paths from the start.
3. **Build connectivity once per step, reuse in all loops** (gradients, limiter, fluxes, time step). For us these are
   the persistent tets and faces.
4. **Stage the target cluster in shared memory, AoS for few fields and SoA for many** (cutoff about 20 fields on GH200).
   With 8 cells per i-cluster and their face partners as j, this is the GROMACS/SPH-EXA tile. It is mainly useful if
   per-face work is short. Our Riemann solver is heavier, so a thread per face in SFC order is probably enough
   (memory-bound gather of 2 × about 10 doubles per face).
5. **Order cells and faces by SFC** so that face partners are close in memory. SPH-EXA reorders every particle field by
   Hilbert key at each `sync` (`domain.hpp:171-188`). Our geometry kernels are 99 % L2-hit already at 2.1 M cells; at
   tens of millions of cells this matters more.

---

## 4. Load balancing and domain decomposition on many GPUs (V)

**Assignment.** A global coarse octree is built with about 100 nodes per rank (`sphexa.cpp:152-153`). Ranks get
contiguous SFC ranges of leaves with equal particle counts (`uniformBins`, `cstone/domain/domaindecomp.hpp:33-55`;
`makeSfcAssignment`, `:113-127`). Load balance is by particle count only; there is no cost weighting in this version.

**Sync sequence** (`domain.hpp:156-169`), 10 steps:
1. bounding box;
2. global octree;
3. max h per node;
4. assignment;
5. halo discovery;
6. layout;
7. resize;
8. exchange of assigned particles;
9. SFC sort;
10. halo exchange.

The previous tree and assignment are the starting guess each time.

**Focus tree / LET** (`cstone/focus/octree_focus_mpi.hpp`). A locally essential tree: fine inside the rank's own
range, coarse outside it, and refined near the boundary according to a MAC. Halo discovery (`discoverHalos`,
`:493-575`, then `findHalosKernel`) does the following for each local leaf:
- take a search box (leaf box extended by 2·h_max of the leaf);
- collide it with the tree outside the own SFC range;
- flag the colliding nodes as halos.

Leaves fully inside the own range are skipped by `containedIn(lowestKey, highestKey, tC, tS, box)`
(`collisions_gpu.cu:44`).

**Halo exchange.** By index ranges of flagged leaves, with send and receive lists stored and replayed by
`exchangeHalos` for any field (`cstone/halos/halos.hpp:52-130`, `exchange_halos_gpu.cuh`). GPU-aware MPI is optional
(`CSTONE_WITH_GPU_AWARE_MPI`).

**(I) Mapping to Voronoi.** Set "h" per cell to half its security radius, i.e. the circumradius R_i of the cell. Then
cstone's halo criterion "node within 2h" is exactly the Voronoi security criterion "a point within 2R can still cut
cell i". The halos are then exactly the points needed to build the boundary cells.

R is known only after construction, but:
- in the persistent-mesh path the previous step's R plus a drift margin works (SPH-EXA does the same with
  `searchExtFactor`, e.g. `*= 1.012`, `ve_hydro_bdt.hpp:205`);
- after a full rebuild, iterate once if a boundary cell fails certification.

For MFV, the halo depth must also cover the gradient stencil (a second layer of neighbours).

---

## 5. Ranked ideas

### (a) Neighbour search for clustered input (Voronoi rebuild)

1. **A cornerstone-format adaptive leaf array as our "multi-level cell list"** (I, high value). Port only:
   - keys (Morton is fine; Hilbert is better for decomposition later);
   - `calculateNodeOp` rebalance (`csarray.hpp:268-290`) with bucket 8-16;
   - the leaf counts by binary search;
   - the `childOffsets`/`parents` internal tree (`octree_gpu.cu`).

   About 1-1.5 ms per build at 2.1 M (from PASC'23 Table 1). It removes the overflowing 32-slot buffer and the
   169-test cells. Expected: clustered approaches the uniform-random rate, which means 5-10× (I, to be measured; our
   uniform-random rate is the ceiling the tree can reach).
2. **Seed then certify** (I). Start with the cell's own leaf plus touching leaves, sorted nearest-first in our existing
   32-batch. Then do one stackless walk (`singleTraversal` style) with the shrinking `2R` test on node boxes, feeding
   leaf points through a **compacted 32-wide queue** (Ryoanji `bodyQueue`, `traversal_gpu.cu:250-300`). Every
   plane-test batch is then full: a fix for both the overflow and the low lane activity in candidate tests.
3. **Keep the uniform grid where it wins** (jittered or uniform input), and choose by the occupancy histogram (I).
4. **What not to copy.** cstone's `ncmax` overflow policy ("Missing neighbors!" warning). A Voronoi cell must never
   drop a candidate: on overflow, fall back to the walk, never truncate.

### (b) Less divergence in per-cell construction

1. **Persistent warps with an atomic cell counter plus descending-cost order.** It is in production at SPH-EXA
   (`build.cuh:556-561`, `gpu_superclusternblist.cuh:327-328`); it is our audit item 1. The cost key can be last step's
   cut count per cell, or the number of points in the leaf (I).
2. **Target groups of 32 SFC-consecutive cells sharing one tree walk** (`groups_gpu.cuh` splits plus `anySync`
   traversal). The warp walks once for the group's bounding box extended by max 2R and writes the union candidate list
   to shared or scratch memory. Each cell then filters that list.
   - With a warp per cell, this saves the candidate fetch and grid walk (about 20-30 % of the walk and fetch phases?
     to be measured with the per-phase timers).
   - It does not fix the topology-update divergence (52-63 % of time), which is intrinsic to clipping.
   - (I) Most valuable on clustered input, where many cells share the same dense leaves.
3. **Full 32-candidate batches** (from a.2). Today a ring step can hand the warp fewer than 32 candidates; a top-up
   queue keeps the test phase at 32 of 32 lanes. (I) The test phase is cheap, so the gain is small on jittered input
   and larger on clustered input.
4. **Not applicable.** SPH-EXA's 8×4 cluster-pair tiling. Our "interaction" (a cut) changes per-cell state
   sequentially, so it cannot be tiled across cells the way a commutative sum can.

### (c) MFV hydro kernel

1. **One face, one flux, written to both cells.** Atomics as SPH-EXA does, or better a segmented reduce over
   owner-sorted and neighbour-sorted face lists (deterministic).
2. **Keep a gather variant for hierarchical time steps** (SPH-EXA disables symmetry for sub-groups).
3. **Build faces once per step** and reuse for gradients, limiter, flux and time-step reduction. Do the time-step
   reduction per target group with `warpMin` and then one atomic (`momentum_energy_gpu.cu:reduceDt`).
4. **SFC-ordered cells and faces; AoS staging for ≤ 20 fields per cell.**
5. **Benchmark target.** SPH-EXA reaches 10-25 M particle updates per GH200 second with gravity. Aim for ≥ 10 M cells/s
   for a full MFV step on an A100 (I).

### (d) Multi-GPU (later)

1. **Reuse the cstone design whole**, or even link the library: it is MIT-licensed and plain CUDA/HIP, so it would sit
   beside our Kokkos code with raw pointers. The design: SFC-range assignment by `uniformBins` on a coarse global tree;
   a focus tree; halo discovery by box collision with h = R_cell; recorded index ranges for repeated halo exchange.
2. **Weight `uniformBins` by measured cost per leaf** (cuts or faces per cell), not by count (I). SPH-EXA uses counts;
   clustered Voronoi cost varies more.
3. **Halo depth.** Security radius for construction, plus one more face layer for the MFV gradients.

### What does not apply

- Smoothing-length iteration (`updateHIterative`, `updateSmoothingLengthGpu`, ng0/ngmax targets).
- Kernel tables, IAD, AV switches, the generalised volume elements (VE/XMass), turbulence stirring.
- Barnes-Hut multipoles, MAC and Ewald (only if we add self-gravity).
- Neighbour-list compression (our lists are about 15 long).
- Truncation at `ncmax`.
- The 8×4 cluster-pair tiling for clipping; it may apply to a light face loop.
- The HIP build via hipify; Kokkos already covers portability for us.

---

## Sources

- Keller, Cavelan, Cabezón, Mayer, Ciorba, "Cornerstone: Octree Construction Algorithms for Scalable Particle
  Simulations", PASC'23, [arXiv:2307.06345](https://arxiv.org/html/2307.06345v1) (Table 1 timings; 8 × 10^12-particle
  LUMI-G run).
- Kubli et al., "Tidal disruption events with SPH-EXA", [arXiv:2510.26663](https://arxiv.org/html/2510.26663v1) §II and
  App. A (GH200: 1-2.5 × 10^7 particle timesteps/GPU/s; 100 neighbours; scaling to 2.8 × 10^12).
- Páll and Hess 2013, "A flexible algorithm for calculating pair interactions on SIMD architectures" (cited in
  `gpu_superclusternblist.cuh:12-13`).
- Code: https://github.com/sphexa-org/sphexa at `a8e6bfa`, local copy `/raven/u/uli/sphexa`.
