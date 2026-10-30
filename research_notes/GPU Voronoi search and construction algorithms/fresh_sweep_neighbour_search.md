# Fresh sweep: GPU neighbour search for per-point 3D Voronoi cell construction (2010 – Oct 2026)

Scope note: this is an independent sweep, done after reading only the task brief (I did not read the existing `neighbour_search.md`). It is judged by what a per-cell Voronoi reconstruction for moving-mesh hydro needs: candidates delivered nearest-first, a certification/stop rule, and a memory pattern that fits a fused or split search+clip kernel. Items marked **[via construction agent, not re-verified by me]** come from agent abea32190e16c99ae's message. Items marked **[unverified]** are from memory with no source fetched in this sweep. ProteusGPU was not used.

Fetch failures to note: HAL (Ray et al. 2018) was blocked by bot protection, the Eurographics LBVH-kNN paper (diglib.eg.org) refused the connection, and the Wald 2022 PDF could not be rendered locally (it was read through ar5iv instead).

## Q1. Which neighbour-search organizations achieve the highest measured throughput on GPUs for fixed-radius or kNN queries in 3D, and why?

### Takeaway
No single organization wins everywhere. With small radii or small k on near-uniform data, coherent (Morton-sorted) per-query traversal of a BVH or k-d tree, or RT-core BVH traversal, reaches 10^8 to 10^9 queries/s, and query order alone is worth 4–6x. With large or variable radii on clustered data, cell lists win. For Voronoi construction specifically, the fastest published system (Taveira et al., SIGGRAPH 2026) uses a fused BVH best-first traversal. Its search costs only about 4% of the kernel, so search throughput is no longer the bottleneck there. Clipping is.

### Cited Findings
- **Wald (2022), stack-free traversal of left-balanced k-d trees.** RTX 3090 Ti, uniform float4 points in [0,1]^4, M = 10,000,000 queries per launch, averaged over 100 launches. Measured queries/s (the table leaves the unit implicit):
  - Unbounded radius, N = 1M points: find-closest-point (fcp) 100M; kNN k=4: 42.1M; k=8: 26.6M; k=20: 7.5M; k=50: 2.9M.
  - Max radius 0.01, N = 1M: k=4: 334M; k=8: 338M; k=20: 282M; k=50: 255M.
  - Max radius 0.01, N = 1k: up to 6.6G.
  - N = 10M: fcp 44M; unbounded k=8: 12.3M; radius 0.01, k=8: 58.5M.
  - N = 200M: fcp 34.8M; unbounded k=8: 10.6M.
  - The author: "We make no claims that our sample implementation be the fastest possible." The data are 4D, not 3D. — [Wald, arXiv 2210.12859](https://ar5iv.labs.arxiv.org/html/2210.12859)
- **Throughput depends strongly on bounding the search.** In the same table, a finite max radius raises k=8 throughput at 1M points from 26.6M to 338M queries/s (about 13x). — [Wald, arXiv 2210.12859](https://ar5iv.labs.arxiv.org/html/2210.12859)
- **RTNN (Zhu, 2022), RT cores, RTX 2080.**
  - Geomean speedups: range search 2.2x over PCLOctree and 44.0x over cuNSearch; kNN 3.5x over FRNN and 65.0x over FastRNN.
  - Inputs: KITTI LiDAR (1M/12M/25M), Stanford scans (Bunny 360K, Dragon 3.6M, Buddha 4.6M), and Millennium N-body (9M, 10M; strongly non-uniform).
  - Absolute times appear only in the figures, which I could not read. — [RTNN, arXiv 2201.01366](https://ar5iv.labs.arxiv.org/html/2201.01366)
- **Query order matters a lot in RTNN.**
  - Randomly ordered queries are "consistently 5 times slower" than coherent ones (the Fig. 5 caption says 4x).
  - Scheduling, which sorts queries in Morton order, gives 1.8x (kNN) and 5.9x (range) on KITTI-12M, and 4.7x (kNN) and 2.9x (range) on NBody-9M.
  - Query partitioning (a per-partition search radius) gives 154.4x for kNN on KITTI-12M but hurts on NBody-9M. — [RTNN](https://ar5iv.labs.arxiv.org/html/2201.01366)
- **RT-core vs GPU cell list, fixed-radius (Meneses et al., 2026).** RTX Pro 6000 Blackwell Server Edition, Lennard-Jones step, time per step:
  - Lattice, r=1, 1M particles: GPU-CELL 2.97 ms; RT-REF 0.599 ms; ORCS-persé 0.506 ms.
  - Lattice, r=160 (about 25,000 neighbours per particle): GPU-CELL 3185 ms vs RT-REF 2235 ms.
  - Cluster, r=160, 1M: GPU-CELL 7119 ms; ORCS-forces 19150 ms; RT-REF ran out of memory.
  - The paper concludes that clustered data with common large radii is where RT cores become inefficient, and that GPU-CELL is the most energy-efficient choice when many particles have large radii.
  - GPU-CELL uses Z-ordering via a radix sort and no fixed-size neighbour list. — [Meneses et al., arXiv 2601.15633](https://arxiv.org/html/2601.15633v1)
- **ArborX.** One thread per query on a binary BVH. Queries are pre-sorted so neighbouring threads get geometrically close queries, "which reduces data and thread divergence". No queries/s numbers were found in this sweep. — [ArborX, arXiv 1908.11807](https://ar5iv.labs.arxiv.org/html/1908.11807); [ArborX 2024, arXiv 2409.10743](https://arxiv.org/pdf/2409.10743)
- **Taveira et al. (SIGGRAPH 2026) fused search+clip, the Voronoi-specific state of the art.** BVH construction is 4.8% of total time and the fused kernel 95.2%. Within the kernel, cell clipping takes 96.2%, and the rest is BVH traversal and culling. Unweighted Voronoi, white noise (Table 7), RTX 5090: 1.0M points 0.112 s; 10M 1.078 s (about 9.3M cells/s). H200: 15M in 1.582 s. (The weighted power-diagram case, Table 11, is slower: RTX 5090 1M 0.123 s, 10M 1.216 s; H200 15M 1.811 s.) — [Taveira et al., arXiv 2605.06408](https://arxiv.org/html/2605.06408v1)
- **Fused-kernel details [via construction agent, not re-verified by me].**
  - One thread per site, BVH built with cuBQL, optional K=8 kNN warm start.
  - Unsorted stack with linear-scan pop of the nearest node.
  - Per-thread cell state sits in SoA global memory strided across the warp rather than in shared memory, for occupancy.
  - Leaf size 17 on H200 and 10 on RTX 5090; positions are packed in float4 for 128-bit loads (the leaf sizes and float4 packing are also confirmed in my fetch). — [Taveira et al.](https://arxiv.org/html/2605.06408v1)
- **Voronoi on an RTX 3080 [via construction agent, not re-verified by me].** Xiao, Cao, Chen (arXiv 2509.07175) store no kNN list and use an edge-based search. 5M white-noise sites: 0.49 s vs Ray et al. 3.12 s and Basselin 2.81 s. — [arXiv 2509.07175](https://arxiv.org/abs/2509.07175)

### Inferences
- For the reconstruction use case, the governing rate is cells/s including clipping, not neighbour queries/s. Taveira's roughly 9–10M cells/s (unweighted Voronoi) on a 5090/H200 is bound by clipping, so further search speedups buy at most about 4–5% there.
- Gains in search come from (a) bounding the query (a radius or certificate that ends the traversal early) and (b) coherence (sorting queries along a space-filling curve). Both give order-of-magnitude effects: about 13x for bounding in Wald, and 4–6x for coherence in RTNN.
- RT cores win for small fixed radii on near-uniform data. They are a poor fit for Voronoi, which needs an adaptive and shrinking bound, ordered delivery and integration with clipping state, and they degrade on clustered data.

### Gaps
- No apples-to-apples 3D kNN queries/s comparison of grid vs BVH vs k-d tree on the same modern GPU was found. Wald's numbers are 4D, and RTNN gives absolute times only in figures.
- No paper found states an achieved fraction of DRAM bandwidth or of the RT-core peak for neighbour search, so "how close to hardware limits" cannot be answered from a source.
- The FRNN/cuNSearch absolute numbers and the Eurographics LBVH-kNN paper ("queries per ms for 16 NN") were not retrieved.

## Q2. Is there measured evidence that processing groups of spatially neighbouring query points together beats independent per-point queries?

### Takeaway
Yes, but mostly indirectly. The strongest evidence is for coherence through ordering (RTNN: 4–6x from Morton-sorted queries) and for cluster-pair processing in MD (GROMACS). Explicit "packet" or shared-candidate schemes for Voronoi/kNN queries have no published head-to-head numbers. The best Voronoi code (Taveira) deliberately stays one thread per site with independent traversal.

### Cited Findings
- **GROMACS cluster-pair scheme.**
  - NVIDIA GPUs use an 8×4 i/j cluster setup (8×8 on 64-wide AMD).
  - Clusters are grouped eight-way from a 2×2×2 cell grouping into super-clusters that share one joint j-cluster list. One thread block handles a super-cluster, and i-forces accumulate in registers.
  - About half of the inner-loop checks are skips ("approximately four of the eight i-clusters contain interactions"), costing an estimated 8–12% overhead.
  - Pruning on cluster bounding boxes, then particle pairs, "reduces the list size by 50–75%".
  - The kernel reaches "close to 100 Ginteractions/s".
  - The 8×4 cluster scheme needs a 0.105 nm buffer vs 0.218 nm for 1×1 at equal error. — [Páll et al. 2020, arXiv 2006.09167](https://ar5iv.labs.arxiv.org/html/2006.09167)
- **Compressed cluster neighbour list (Thaler & Keller, 2026).** It uses an SFC memory layout and particle clusters, "4 bytes per particle to store approximately 200 neighbors". It reports "comparable cluster overhead and ... similar performance" to GROMACS's list, and "performs well for systems with high density contrasts". No numbers appear in the abstract. — [arXiv 2602.19873](https://arxiv.org/abs/2602.19873)
- **RTNN bundling.** Bundling queries adds 18.8% and 18.6% on range search for the two evaluated inputs and has little effect on kNN. Morton scheduling is the main coherence win. — [RTNN](https://ar5iv.labs.arxiv.org/html/2201.01366)
- **Hierarchical SPH.** A hierarchical cell-based GPU SPH method with shared memory and merged tasks reports 1.73x over other GPGPU SPH frameworks. This is a framework-level comparison, not packet vs per-point. — [Stony Brook record](https://researchconnect.stonybrook.edu/en/publications/novel-hierarchical-strategies-for-sph-centric-algorithms-on-gpgpu/)
- **Verlet vs cell lists on GPUs.** Winkler et al. (2018, DualSPHysics) report per-particle Verlet lists "may be significantly faster" than cell linked lists, at much higher memory, with speedups that depend on hardware, dimensionality and precision. — [Mendeley dataset page](https://data.mendeley.com/datasets/nsx8hfcycv)
- **Taveira's per-thread layout [via construction agent, not re-verified by me].** Per-thread cell state is kept in warp-strided SoA global memory (not shared memory) for occupancy. They report no packet variant. — [Taveira et al.](https://arxiv.org/html/2605.06408v1)

### Inferences
- In MD/SPH the interaction itself is cheap and identical for all pairs, so sharing j-loads across an i-cluster pays. In Voronoi each candidate triggers a divergent, state-dependent clip of a polyhedron, and Taveira shows clipping is about 96% of the time. A packet scheme can therefore at best save part of the about 4% search share, unless it also cuts the number of wasted clips.
- A plausible place for a tile/cluster scheme is building a conservative shared candidate list per spatial tile (a GROMACS-style bounding-box-pruned cluster list). Each thread would then consume that list nearest-first. I found no published measurement of this for Voronoi.

### Gaps
- No paper found measures packet/warp-cooperative traversal vs per-thread traversal for kNN or Voronoi on GPU.
- The GROMACS j-data placement (shared memory vs registers) was not stated in the text read.

## Q3. What is the minimum candidate set needed to certify a Voronoi cell, and which pruning rules get closest to it?

### Takeaway
The minimum is the set of true Voronoi neighbours, plus whatever is needed to prove no other site cuts the cell. The standard certificate is the security radius: a sphere of radius 2·max|v − p_i| over the cell vertices v contains every possible cutter. Best-first BVH traversal with directional (per-octant) bisector-distance bounds certifies when the stack empties, and it is markedly better than fixed-k kNN plus a security-radius check on non-uniform data.

### Cited Findings
- **Security radius [via construction agent, not re-verified by me].** Ray et al. 2018 / Basselin et al. 2021 take a fixed-k kNN from a grid and certify the cell when twice the farthest cell-vertex distance is below the distance to the k-th neighbour. Cells that fail are retried with larger k or sent to the CPU. — [Ray et al. 2018, HAL](https://hal.archives-ouvertes.fr/hal-01927559); [Basselin et al. 2021](https://diglib7.eg.org/bitstream/handle/10.1111/cgf142610/v40i2pp001-012.pdf)
- **Security-radius origin.** The security-radius idea for terminating per-element processing comes from Lévy & Bonnel 2012 (restricted Voronoi diagrams), as summarised in a Pacific Graphics 2017 paper. — [PG 2017](https://diglib7.eg.org/bitstream/handle/10.2312/pg20171320/023-026.pdf)
- **Taveira culling rule.** Their culling criterion is "based on directional geometric bounds of the evolving cell". Ablations on RTX 5090 (median slowdown vs the full method):
  - Isotropic single-radius bound: 1.26x clustered, 1.33x Poisson, 1.40x real-world.
  - Depth-first instead of best-first: 27.4x (range 3.96–80.3) clustered, 1.30x Poisson, 37.2x real-world. — [Taveira et al.](https://arxiv.org/html/2605.06408v1)
- **Bisector-distance bound [via construction agent].** Children are pruned when the lower bound of the bisector distance d_ij = (|pi−pj|² + wi − wj)/(2|pi−pj|) over the child box exceeds a directional radius r_i (taken from the cell AABB corners in that child's octants). The test is re-run on pop.
- **Failures of fixed-k.** Taveira's port of Basselin "fails" on all real-world and clustered inputs, while it runs on white noise and the density gradient. — [Taveira et al.](https://arxiv.org/html/2605.06408v1)
- **Precision.** Taveira runs in FP32. Against CGAL exact predicates, about 0.1–0.2% of adjacencies mismatch on real data, caused by near-coincident planes. — [Taveira et al.](https://arxiv.org/html/2605.06408v1)

### Inferences
- The fixed-k sphere certificate is loose. The sphere of radius 2R_max around p_i covers a volume about 8x the cell's circumscribed ball, so many candidates must be fetched that cannot cut. Directional bounds tighten this (1.26–1.40x end to end).
- Best-first (nearest box first) delivers candidates roughly nearest-first. Nearest-first matters for clipping cost, since early true neighbours shrink the cell fast and make later tests prune. This is consistent with the 27–37x depth-first penalty on non-uniform data.
- For a moving-mesh hydro code, the previous step's neighbour list is a natural warm start. It plays the role of Taveira's K=8 kNN warm start and of GROMACS's dual/rolling lists. No paper found measures this for Voronoi.
- **[unverified]** The expected number of faces of a Poisson–Voronoi cell in 3D is about 15.54 (Meijering 1953). That is the lower bound on useful candidates per cell for random points. No sweep source reports candidates examined per cell for GPU methods.

### Gaps
- No source found reports the number of candidates fetched or clipped per cell (e.g. a wasted-cut ratio) for Ray, Basselin or Taveira. The Taveira appendix text read did not include it.
- No GPU paper found evaluates an exact (filtered/exact-predicate) certificate.

## Q4. How do methods cope with strongly clustered point sets (density contrasts 1e3–1e6)?

### Takeaway
Uniform grids and fixed-k kNN break down on clustered data: memory blow-up, failures, and a need for CPU fallback. Hierarchies (BVH with best-first traversal, octree/SFC cluster lists) cope. On "10 clusters" inputs Taveira's method is 2–3x slower than on white noise at the same N but completes. gDel3D (GPU Delaunay) is faster at small N but fails beyond about 2–5M clustered points.

### Cited Findings
- **Taveira, 10 clusters, RTX 5090.**
  - Ours: 0.1M 0.175 s; 1M 0.386 s; 5M 1.224 s; 10M 1.423 s.
  - gDel3D: 2M 3.120 s, failing at 5M.
  - On H200, gDel3D takes 45.423 s at 5M. Basselin* fails.
  - Unweighted white noise for comparison (Table 7): 1M 0.112 s; 10M 1.078 s. — [Taveira et al.](https://arxiv.org/html/2605.06408v1)
- **Taveira, density gradient, RTX 5090.** 1M 0.118 s and 10M 1.102 s. Basselin*: 1M 0.387 s and 10M 3.602 s. — [Taveira et al.](https://arxiv.org/html/2605.06408v1)
- **Taveira, real-world scenes (Gaussian-splat point sets), RTX 5090.** bicycle 4.2M 0.570 s vs gDel3D 1.204 s. — [Taveira et al.](https://arxiv.org/html/2605.06408v1)
- **RT cores and the GPU cell list.** RT-core FRNN degrades or runs out of memory on clustered data with large radii. The Z-ordered GPU cell list without a fixed-size neighbour list handles it but is slow: 7119 ms per step for 1M particles, cluster, r=160. — [Meneses et al.](https://arxiv.org/html/2601.15633v1)
- **Query partitioning on N-body data.** RTNN's query partitioning (per-region radius) helps LiDAR kNN by 154x but degrades on N-body data because of partitioning and BVH-rebuild overhead. — [RTNN](https://ar5iv.labs.arxiv.org/html/2201.01366)
- **SFC/cluster list on high density contrast.** The SFC/octree cluster neighbour list targets astrophysics, claims good performance at "high density contrasts", and supports per-particle radii. — [Thaler & Keller, arXiv 2602.19873](https://arxiv.org/abs/2602.19873)
- **Ray et al. at 10M points [via construction agent].** Ray et al. 2018 could not reach 10M white-noise points on a GTX 1080, per the Taveira slides.

### Inferences
- For contrasts of 1e3–1e6, a uniform grid cell size must either suit the dense regions (wasting memory and empty-cell visits in sparse regions) or the sparse ones (overloading dense cells). A BVH or octree over Morton-sorted sites adapts automatically, and best-first traversal with a shrinking directional bound is the published approach that stays robust.
- None of the clustered benchmarks found quantify the density contrast. The "10 clusters" sets are not characterised in that way in the text read, so robustness at 1e6 contrast is unproven in the literature.

### Gaps
- No GPU Voronoi benchmark found reports density contrast explicitly or uses cosmological/hydro particle sets for Voronoi construction (RTNN's Millennium data was used only for kNN/range).
- Load imbalance within warps on clustered inputs (the spread of per-cell candidate counts) is not reported.
