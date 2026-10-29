# Cooperative granularity for irregular per-item GPU work: sub-warp teams, warp-per-item, hybrid/adaptive

Scope: how many lanes to assign per work item (1 / 2-16 / 32 / CTA) when items vary in size and have data-dependent control flow; measured evidence; applicability to the voronoi-cpp cell-clipping kernel (Kokkos/CUDA, A100, FP64; lean = thread-per-cell, 5.4 active lanes/warp, 8% occupancy; warp kernel = warp-per-cell, 15.6 active lanes/warp, 17% occupancy, 130 regs/thread; ~16-24 cuts per cell, ~30 vertices; parallel vertex test + mostly serial topology rewiring; not memory-bound).

Source-access note: full text was read (via local PDF-to-text) for Hong et al. 2011, WarpCore (Jünger et al. 2020), Osama et al. PPoPP'23 and Taveira et al. 2026. Other sources (LightSpMV, Gunrock, Merrill BFS / merge SpMV, Better GPU Hash Tables, TCF filters, Lier et al. HPG'18, CoopRT, CSR-Adaptive) were seen only as abstracts or secondary summaries; numbers from those are abstract-level.

## Q1. What speedups do sub-warp teams (4/8/16 lanes) give over thread-per-item and warp-per-item for items with ~10-50 elements of work and data-dependent loops?

### Takeaway
The best-documented measurements (Hong et al., BFS) show sub-warp "virtual warps" beat both thread-per-item and full warps, and the best team size is roughly the item's average parallel width (about the average degree) for regular inputs. For heavily skewed inputs, 32 lanes won. The gains over thread-per-item are large (up to 9x over the earlier GPU BFS). The gains over a correctly sized full warp are modest, in the tens of percent. No paper I found measures sub-warp teams on geometric clipping with a serial topology phase. For our kernel, sub-warp gains over the existing warp kernel have to be inferred, and a 2x gain is plausible but not demonstrated.

### Cited Findings
- **Virtual warps (Hong, Kim, Oguntebi, Olukotun, PPoPP 2011, "Accelerating CUDA Graph Algorithms at Maximum Warp", pp. 267-276):**
  - The method splits a physical warp into "virtual warps" of 4, 8 or 16 lanes, with each virtual warp processing a different task. All lockstep and coalescing assumptions still hold inside a virtual warp. [Hong et al. 2011 PDF](https://ppl.stanford.edu/papers/ppopp070a-hong.pdf)
  - The authors state the trade-off explicitly: "the virtual warp scheme can be viewed as a trade-off between execution-path divergence and ALU underutilization by varying a single parameter, the virtual warp size." They also say the parallelism of the SISD phase increases when a physical warp holds several virtual warps. [Hong et al. 2011](https://ppl.stanford.edu/papers/ppopp070a-hong.pdf)
  - They name two drawbacks of a full warp per task. (1) If the task's native SIMD width is small, lanes sit idle. (2) "The ratio of the SIMD phase duration to the SISD phase duration imposes an Amdahl's limit on performance." [Hong et al. 2011](https://ppl.stanford.edu/papers/ppopp070a-hong.pdf)
  - Measured result (Fig. 7b; GTX 260; graphs with 4M nodes and 48M edges; RMAT, Random, LiveJournal, Patents). Regular graphs (Random, Patents) "achieve maximum average utilization of the SIMD lanes when the virtual warp size is set to approximately the average degree value" (average degree about 6-12). For the skewed RMAT graph, "a virtual warp size of 32 yielded optimal results." With a properly chosen size, the method beats the thread-per-vertex baseline on every graph. [Hong et al. 2011](https://ppl.stanford.edu/papers/ppopp070a-hong.pdf)
  - When average degree is below the warp size, full warps underuse SIMD lanes and "a smaller virtual warp size (i.e. warp16) provides meaningful benefit" (Fig. 8d). [Hong et al. 2011](https://ppl.stanford.edu/papers/ppopp070a-hong.pdf)
  - Headline numbers: up to 9x over previous GPU algorithms and 12x over single-thread CPU on irregular graphs. Up to 30% over previous GPU algorithms on regular graphs. 1.3x to 15.1x when applied to other GPU applications. [Hong et al. 2011 PDF](https://ppl.stanford.edu/papers/ppopp070a-hong.pdf); [hgpu abstract](https://hgpu.org/?p=2995)
- **Hash tables with tile size as a parameter:**
  - WarpCore (Jünger et al., IEEE HiPC 2020) uses cooperative-groups tiles of 1, 2, 4, 8, 16 or 32 threads per key. Each lane loads one slot coalesced, a group vote (ballot) finds candidates, and the lowest-index candidate lane is elected to do the atomic CAS. The authors motivate sub-warp tiles because at 90% load "it would be likely that an empty slot can be found by a group of less than 32 threads at the first probing attempt." Result: up to 1.6 G inserts/s and 4.3 G retrievals/s on a GV100. [WarpCore arXiv 2009.07914](https://arxiv.org/abs/2009.07914)
  - The TCF / GPU filters paper (McCoy et al., "High-Performance Filters for GPUs", PPoPP 2023) reports "an optimal cooperative group sizing for each filter variation. For the majority of the configurations, this size is 4." The stated reason is that a larger group gives less divergence and better compute throughput but fewer memory operations in flight. [arXiv 2212.09005](https://arxiv.org/abs/2212.09005)
  - Better GPU Hash Tables (Awad et al., 2023) recommend a bucketed cuckoo table with bucket size 16 and 3 hash functions, processed by a tile of matching width, as the best general configuration. Average insertion probe count at load factor 0.99 is 1.43. [arXiv 2108.07232](https://arxiv.org/abs/2108.07232)
  - SlabHash (Ashkiani, Farach-Colton, Owens, IPDPS 2018) introduced a "warp-cooperative work sharing" strategy (WCWS) to reduce branch divergence. [arXiv 1710.11246](https://arxiv.org/abs/1710.11246)
- **SpMV vector width:**
  - LightSpMV (Liu & Schmidt, J. Signal Proc. Systems 2016) uses dynamic, atomic-based distribution of rows to "vectors" (sub-warp groups) or warps, with shuffle-based dot products. On a K40c it reports up to 2.63x/2.65x (SP/DP) over CUSP and up to 1.94x/1.79x over cuSPARSE. [DOI 10.1007/s11265-016-1216-4](https://doi.org/10.1007/s11265-016-1216-4); [hgpu](https://hgpu.org/?p=13445)
  - Bell & Garland (SC09) CSR-vector assigns a warp per row. A later source notes it performs poorly for rows with too few nonzeros, which motivates smaller vectors and row binning. [DOI 10.1145/1654059.1654078](https://doi.org/10.1145/1654059.1654078); [secondary: patent text](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9697176)
- **BVH traversal with several lanes per ray:**
  - Lier, Stamminger, Selgrad (HPG 2018) spread nodes of moderately wide BVHs (branching factor up to 8) across several lanes of a warp, i.e. a sub-warp per ray. They report improved traversal "especially for increasingly incoherent rays"; the abstract gives no numbers. [Eurographics DL / DOI 10.1145/3231578.3231583](https://diglib.eg.org/handle/10.1145/3231578-3231583)
  - CoopRT (hardware proposal, simulator-only) lets idle lanes help busy lanes traverse the BVH. Reported: up to 5.11x and geomean 2.15x in Vulkan-sim, at 3.0% area. This is not a software technique. [CoopRT, UMD](https://rqs.umd.edu/publications/cooprt-accelerating-bvh-traversal-ray-tracing-cooperative-threads)
- **Closest Voronoi baseline, thread-per-cell:** Taveira et al. 2026 ("Scalable GPU Construction of 3D Voronoi and Power Diagrams", SIGGRAPH Conf. Papers '26, arXiv 2605.06408) use "one convex polyhedron per thread", single precision. Per-thread state lives in strided SoA global memory rather than shared memory, because the shared-memory footprint "severely limit[s] SM occupancy". White-noise (uniform) end-to-end times are 1.216 s for 10M points on an RTX 5090 and 1.208 s on an H200, about 8.2 M cells/s including BVH build. They report a 3-6x speedup over their own CUDA reimplementation of Basselin et al. 2021 (white noise: 3.753 s / 3.294 s at 10M). [arXiv 2605.06408](https://arxiv.org/abs/2605.06408)

### Inferences
- Hong's rule ("team size ≈ parallel width of the item") maps to our case as follows. The parallel phase has about 30 vertices, which suggests 16-32 lanes. The serial rewiring phase has width 1, which pushes toward smaller teams. Hong's Amdahl argument (SISD/SIMD ratio) is the key point. If topology rewiring takes most of the per-cut time, a warp per cell leaves 31 lanes idle during it. Four 8-lane teams per warp run four cells' serial phases in the same instructions, provided their control paths coincide. Under that condition the serial-phase throughput per warp rises by up to 4x, while the parallel phase per cell takes about 4 iterations instead of 1.
- Our numbers fit this picture. The warp kernel (15.6/32 active lanes) is about 2.9x more SIMD-efficient than lean (5.4/32), yet throughput is only ~15 M/s jittered. That suggests the remaining limit is the serial rewiring chain and its latency at 17% occupancy, not the vertex test.
- For Taveira, a single-precision thread-per-cell code reaching ~8 M cells/s end-to-end on uniform data is in the same range as our 8 M/s uniform FP64 warp kernel on A100. This is a cross-precision, cross-GPU comparison and only indicative. It hints that thread-per-cell with high occupancy (global-memory SoA instead of shared memory) is a competitive alternative axis, not only finer teams.
- None of the cited speedups transfer numerically. BFS, hashing and SpMV have trivial per-element work and no serial pointer rewiring. The 2x+ target is not supported by direct evidence for clipping.

### Gaps
- I found no paper that measures 4/8/16-lane teams per convex polyhedron (clipping or hull) against thread-per-item and warp-per-item.
- Per-tile-size throughput tables for WarpCore and BCHT were not extracted. Only the existence of a tile-size optimum ("4" for TCF) is cited.
- I did not obtain Lier et al.'s numeric speedups.

## Q2. How is the per-item serial part (editing a small shared data structure) handled cooperatively?

### Takeaway
There are three established patterns. (a) Replicated SISD: every lane of the team executes the serial code on the same data (Hong). (b) Elect a leader with a ballot or vote, the leader alone does the mutation, and the result is broadcast with a shuffle (WarpCore, SlabHash). (c) Restructure the serial edit into a data-parallel form via ballot, popc and prefix-sum compaction (Merrill-style scan-based gather; merge-path). For polyhedron rewiring, (c) applies to compacting surviving vertices and allocating new ones. Walking the new face boundary remains a short serial chain.

### Cited Findings
- **Replicated SISD (Hong et al. 2011).** During the SISD phase "every thread in a warp executes exactly the same instruction on exactly the same data", enforced by giving all lanes the same warp ID. This costs no extra memory bandwidth because identical addresses in a warp merge. Atomics in SISD regions must be guarded (`if (WARP_OFFSET==0)`) so that only one lane executes them. Each virtual warp gets a private shared-memory partition. [Hong et al. 2011](https://ppl.stanford.edu/papers/ppopp070a-hong.pdf)
- **Vote, elect leader, CAS, broadcast (WarpCore).** Lanes test their slots, "communicate[] via a fast in-register group vote", the lowest candidate lane is selected and does `atomicCAS`, and on failure the group re-elects among the remaining candidates. [WarpCore](https://arxiv.org/abs/2009.07914)
- **Warp-cooperative work sharing (SlabHash).** Lanes hold different keys, but the warp processes one lane's operation at a time cooperatively, with a ballot-selected queue of pending operations. [Ashkiani et al. 2018](https://arxiv.org/abs/1710.11246)
- **Scan-based gather (Merrill, Garland, Grimshaw, PPoPP 2012, "Scalable GPU Graph Traversal").** Fine-grained task management "using efficient prefix sum" gives O(|V|+|E|) work. Reported: 3.3 G TEPS on one GPU and 8.3 G TEPS on four. [NVIDIA PDF](https://research.nvidia.com/sites/default/files/pubs/2012-02_Scalable-GPU-Graph/ppo213s-merrill.pdf). Gunrock describes Merrill's mapping as assigning each vertex's work to a thread, warp or CTA by neighbor-list size. [Gunrock arXiv 1501.05387](https://arxiv.org/abs/1501.05387)
- **Cooperative groups mechanics (NVIDIA docs, not peer-reviewed).** `tiled_partition<N>` gives static tiles of N = 1, 2, 4, 8, 16 or 32 lanes, with tile-scoped `shfl`, `ballot`, `any/all`, `sync` and reduce/scan collectives. [CUDA C++ Programming Guide: Cooperative Groups](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)

### Inferences
- **Mapping to clip-and-rewire.** The vertex/plane test is a natural SIMD phase: each lane tests its ~4 vertices and the team takes a `ballot` per lane. The "any vertex cut?" early-out becomes `tile.any()`.
- **Data-parallel parts of rewiring.** Removing dead vertices and assigning new-vertex slots are a stream compaction (popc and exclusive-scan over the ballot masks), which is data-parallel.
- **Finding the boundary edges** can also be done in parallel. Each lane checks its alive vertices for neighbors that are dead, and each such edge creates one new vertex.
- **What stays serial.** Ordering the new face cycle is the inherently serial piece. It is short: the new face has typically ~5-7 edges.
- **Ordering the cycle in parallel.** It can be done with each lane computing its edge's successor locally (the next edge around the dead region via triangle/plane-index adjacency), then shuffling indices. Alternatively, a single elected lane can walk the cycle (WarpCore-style leader) while the other lanes wait. The replicated-SISD variant (Hong) avoids the broadcast but keeps the whole team busy on redundant work.
- **Representation.** Encoding vertices as plane-index triples, as in Taveira, makes the "which edges cross" test local to each vertex. That is the property needed for the parallel boundary-edge discovery above.

### Gaps
- I found no published warp- or tile-cooperative implementation of the polyhedron rewiring step to cite. The design above is inference.

## Q3. Evidence for hybrid schemes (route large items to bigger teams), especially on Ampere-class GPUs

### Takeaway
Size-binned thread/warp/CTA routing (Merrill TWC, as used in Gunrock), dynamic CG group sizes (Osama et al. PPoPP'23) and row-binning SpMV (CSR-Adaptive and follow-ups) are well established and consistently beat any single granularity on skewed inputs. Gains shrink or turn negative on uniform inputs, where the binning overhead is not repaid. Ampere-specific evidence is thin. The Osama et al. results are on V100.

### Cited Findings
- **Gunrock TWC (Wang et al., TOPC 2017).** Neighbor lists fall into three size classes (larger than a block, between warp and block, smaller than a warp), each with its own strategy. The paper claims this addresses SIMD-lane underutilization and intra-warp imbalance. Gunrock also offers a merge-based (load-balanced partition) advance. [Gunrock arXiv 1701.01170](https://arxiv.org/abs/1701.01170); [arXiv 1501.05387](https://arxiv.org/abs/1501.05387). A later adaptive-balancer paper notes TWC bins "may result in load imbalance if degree distributions within a bin vary significantly". [arXiv 1911.09135](https://arxiv.org/abs/1911.09135)
- **Osama, Porumbescu, Owens, "A Programming Model for GPU Load Balancing" (PPoPP 2023, DOI 10.1145/3572848.3577434).**
  - The paper offers thread-mapped, group-mapped (CG groups of arbitrary programmer-specified size, generalizing thread, warp and block) and merge-path schedules.
  - A heuristic picks merge-path unless rows or columns are below 500 and nnz is below 10000, in which case it uses thread- or group-mapped.
  - Result: geomean 2.7x and peak 39x over cuSPARSE on SuiteSparse, on a V100 with CUDA 11.7.
  - Their merge-path SpMV has a 2.5% geomean slowdown vs CUB's.
  - [arXiv 2301.04792](https://arxiv.org/abs/2301.04792)
- **Hong et al.** "Deferring outliers" (pushing large items to a later pass) gave large gains alone but "only a marginal performance gain" combined with virtual warps. Dynamic workload distribution helped skewed RMAT but "yields more overhead than benefit" on evenly distributed graphs. [Hong et al. 2011](https://ppl.stanford.edu/papers/ppopp070a-hong.pdf)
- **Merge-based SpMV (Merrill & Garland, SC16, DOI 10.1109/SC.2016.57).** Equitable 2-D merge-path decomposition bounds per-PE work regardless of row distribution. Conventional CsrMV shows "order-of-magnitude performance variation across similarly-sized datasets". [SC16 page](https://sc16.supercomputing.org/presentation/index-id=pap369&sess=sess147.html); [code](https://github.com/dumerrill/merge-spmv). Merge path carries more overhead than row-split and lost for SpMM. [Yang, Buluç, Owens, arXiv 1803.08601](https://arxiv.org/abs/1803.08601)
- **Row binning SpMV.**
  - CSR-Adaptive (Greathouse & Daga, SC14) reports 14.7x average over prior CSR kernels and 2.3x over clSpMV. [SC14](https://sc14.supercomputing.org/schedule/event_detail-evid=pap569.html)
  - An auto-tuned binning plus per-bin kernel selection (Hou, Feng, Che, ASHES 2017) gives up to 1.9x over the state of the art. [PDF](https://seec.cs.vt.edu/pubs/papers/hou-spmv-ashes17.pdf)
- **A100 sparse evaluation (Anzt et al. 2020).** CSR kernels gain ~1.7x from V100 to A100, which tracks bandwidth. Rows with high variance (small reads) perform relatively poorly on A100. [arXiv 2008.08478](https://arxiv.org/abs/2008.08478)

### Inferences
- Hybrid routing pays off when item size is heavy-tailed and size is known before processing (degree, row length).
- For Voronoi cells, the final size (number of cuts/vertices) is not known up front. It can be predicted cheaply: by neighbor-count within the certification radius, or by a first pass that flags overflow (our kernel already rebuilds overflowing cells with larger capacity, i.e. a "defer outliers" scheme).
- Hong's result that deferring outliers adds little once teams are used suggests the tail is not where 2x comes from. On uniform or jittered points, cell sizes are fairly concentrated (~16-24 cuts), so per-cell size skew is small. Our divergence is primarily control-flow divergence across cells (different branch outcomes per cut), not size imbalance.

### Gaps
- I found no hybrid granularity study with measurements on A100 or Ampere, and none for computational-geometry items.

## Q4. Applicability to our kernel (8 lanes/cell, or thread-per-cell plus warp for tail cells)

### Takeaway
Inference only, since no direct literature exists. An 8-lane team per cell (4 cells per warp) is the best-motivated variant by Hong's trade-off. It addresses the actual bottleneck (the serial rewiring idles 31 lanes in warp-per-cell) only if (i) rewiring is made largely data-parallel (ballot/scan compaction plus a short cycle walk) and (ii) the four teams stay in convergent control flow. Otherwise, independent teams in one warp serialize and you get warp-kernel behavior with more registers. Thread-per-cell plus a warp for the tail targets size skew, which our inputs barely have. It is unlikely to give 2x on uniform or jittered data.

### Cited Findings
- Optimal team size tracks the item's parallel width for regular inputs. Full warps win only under heavy skew. [Hong et al. 2011](https://ppl.stanford.edu/papers/ppopp070a-hong.pdf)
- Small-tile optima of 4 lanes appear when larger tiles reduce the number of independent items in flight. [McCoy et al. arXiv 2212.09005](https://arxiv.org/abs/2212.09005)
- A thread-per-cell Voronoi/power-diagram clipper gains throughput by moving per-thread polyhedra to coalesced global SoA to raise occupancy. It also caches vertex positions and encodes vertices as plane-index triples. [Taveira et al. 2026](https://arxiv.org/abs/2605.06408)

### Inferences
**Why 8 lanes:**
- About 30 vertices over 8 lanes gives ~4 per lane: 4 short loop iterations of the plane test, then `ballot`.
- Four cells share each warp-instruction during serial or replicated phases, versus one cell in the warp kernel.
- At equal occupancy, the expected gain is up to ~2-4x on the rewiring-dominated portion and a slight loss on the vertex-test portion. That is consistent with a 2x target only if rewiring is more than ~60% of the time. Profile instruction counts per phase first.

**Risks:**
- *Register pressure.* Each lane holding ~4 vertices (FP64 positions = 12 doubles = 24 regs, plus indices) on top of the current 130 regs may push spills. Keep vertex positions in shared memory: 4 cells × ~1.7 KB = ~7 KB per warp, about the same as the warp kernel per cell×4. Keep only indices and flags in registers.
- *Convergence.* The data-dependent number of cuts per cell (16-24) means teams finish at different times. Use a persistent loop in which a team that finishes grabs the next cell (an atomic counter per team or per warp, as in LightSpMV's dynamic row distribution) so lanes do not idle until the warp's slowest cell completes.
- *Per-cut branch divergence across teams.* "Plane misses cell" versus "plane cuts cell" still diverges across teams. Mitigation: evaluate the cheap early-reject test for all four cells, then run the expensive rewiring path only once, with all teams that need it active together (masking idle teams). This is the same principle as Hong's coherent SISD/SIMD split.

**Thread-per-cell plus warp-for-tail (Gunrock/TWC-style):**
- The lean kernel's 5.4/32 activity comes from control-flow divergence per cut, not from a few huge cells. Routing tail cells to warps fixes the wrong problem for uniform or jittered points.
- It is worth having only for clustered inputs, where cell neighbor counts are heavy-tailed. Our existing overflow-rebuild path already acts as Hong's "deferred outliers".

**Orthogonal lever with literature backing:** occupancy. Both of our kernels are at 8-17%. Taveira's argument for global-memory SoA over shared memory to raise occupancy applies to the lean kernel. Pairing tiles with lower per-cell register and shared-memory use could compound with any team-size gain.

**Suggested experiment order:**
1. Instrument phase cost (vertex test vs. rewiring vs. neighbor fetch) in the warp kernel.
2. Template the warp kernel on a tile size T ∈ {4, 8, 16, 32}, using `cooperative_groups::tiled_partition<T>` or Kokkos TeamPolicy vector length (Kokkos `ThreadVectorRange` with vector_length 4/8/16 maps to sub-warp lanes on CUDA).
3. Measure cells/s on jittered and uniform data. Hong's sweep is the methodological precedent.

### Gaps
- I found no measured data on sub-warp teams for convex-polyhedron clipping, convex hull per small team, or Delaunay star construction. Any 2x expectation is speculative.
- Kokkos-specific overheads of a team-of-T (TeamPolicy with vector_length < 32) versus raw CG tiles were not researched here.
