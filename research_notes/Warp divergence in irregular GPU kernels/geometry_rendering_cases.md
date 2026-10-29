# Divergence and load imbalance in GPU geometry, ray tracing and particle codes: case studies

Scope: GPU per-cell Voronoi, Delaunay and power diagram codes; ray and path tracing; Barnes-Hut and MD/SPH pair loops. The question is what each code did about SIMT divergence and long-tail per-item work, and what it measured. Each case is mapped to our candidate fixes: (1) persistent threads or dynamic fetching, (2) tail deferral with work caps, (3) sub-warp teams per cell, (4) stage splitting or wavefront, (5) sorting cells by predicted cost, (6) branch-free formulations.

Method note: I made about 20 tool calls. Some sources were blocked. Both HAL PDFs (Ray et al. 2018 and Basselin et al. 2021) returned Anubis bot-blocks, eg.org refused the connection, and the Laine 2013 PDF could not be text-extracted on this machine. Details of those papers come from abstracts or from descriptions in later papers, and are flagged as such. ProteusGPU was not used, as requested.

## Q1: Which techniques gave 2x or more on irregular geometric per-item work on modern GPUs, and which did not help?

### Takeaway
The largest documented wins on long-tail, variable-length per-item work come from regrouping live work so idle lanes are refilled. Examples are persistent threads with a global work pool in BVH traversal, wavefront splitting in path tracing, and hardware Shader Execution Reordering (SER), which gave 16.5x on a photon-lifetime long tail. Software sorting of rays is the clearest case of "helps in the kernel, loses end-to-end": reordering speeds up the trace phase 1.3-2.0x, but the sort overhead is hard to recover. In GPU Voronoi and Delaunay codes, the tail is handled by capacity limits plus deferral, retry or a CPU repair stage, not by regrouping.

### Cited findings

**GPU Voronoi, power diagram and Delaunay codes**

- **Taveira, Lindström, Fatemi, Hammarstrand, Kahl, "Scalable GPU Construction of 3D Voronoi and Power Diagrams", arXiv:2605.06408 (7 May 2026; preprint, not yet peer reviewed).** [arXiv abs](https://arxiv.org/abs/2605.06408), [HTML](https://arxiv.org/html/2605.06408)
  - Each cell starts from an initial volume and is clipped by bisector planes from candidate neighbors. Candidates come from a culling criterion based on directional geometric bounds of the evolving cell, plus a best-first traversal of a BVH. [arXiv abs](https://arxiv.org/abs/2605.06408)
  - Mapping: the construction "decomposes naturally into N independent volume-clipping tasks, which we map one-to-one onto GPU threads", i.e. one thread per cell, written in plain CUDA. [HTML](https://arxiv.org/html/2605.06408)
  - Occupancy: per-thread clipping data (planes as float4, vertices as uchar3 triples of plane indices, cached vertex positions) is kept in coalesced global memory, not shared memory. The authors "found the resulting per-thread footprint to severely limit streaming multiprocessor (SM) occupancy", and global memory "allows more warps to reside on each SM". [HTML](https://arxiv.org/html/2605.06408)
  - Capacity: plane and vertex arrays have fixed size. When the plane array passes 85% fill, it is compacted by discarding fully clipped planes, with no reallocation. [HTML](https://arxiv.org/html/2605.06408)
  - Traversal: a per-thread stack where pop finds the nearest entry by linear scan. An optional KNN warm start (K=8) pre-clips each cell. It is on for the RTX 5090 and off for the H200: "The warm start is therefore disabled on such hardware." [HTML](https://arxiv.org/html/2605.06408)
  - Divergence and load imbalance are not analyzed directly. No lane-utilization or warp-efficiency figures were found in the text read (first ~100k of 112k characters). [HTML](https://arxiv.org/html/2605.06408)
  - Timings, end-to-end, single precision:

    | Case | GPU | This method | gDel3D | Radiant Foam Delaunay |
    |---|---|---|---|---|
    | Garden, 4.1M pts | H200 | 0.574 s | 0.862 s | 39.551 s |
    | Garden, 4.1M pts | RTX 5090 | 0.496 s | 1.168 s | 40.413 s |
    | Clustered K=10, 10M | H200 | 1.848 s | no result | – |
    | Clustered K=10, 15M | H200 | 2.530 s | no result | – |
    | Clustered K=10, 5M | H200 | – | 45.423 s | – |

    These correspond to roughly 5.9-9.4 M cells/s on the H200 (the cells/s figures are computed from the reported times). [HTML](https://arxiv.org/html/2605.06408)
- **Basselin, Alonso, Ray, Sokolov, Lefebvre, Lévy, "Restricted Power Diagrams on the GPU", Computer Graphics Forum 40(2), 2021.** [EG PDF](https://diglib7.eg.org/bitstream/handle/10.1111/cgf142610/v40i2pp001-012.pdf), [HAL](https://hal.univ-grenoble-alpes.fr/INRIA/hal-03169575v1)
  - Reported: 3D Voronoi diagrams of 10 million points restricted to a mesh in 300 ms on an NVIDIA V100 (search-result summary of the paper). [HAL](https://hal.univ-grenoble-alpes.fr/INRIA/hal-03169575v1)
  - As described by Taveira et al. 2026: on overflow it relaunches the kernel with larger arrays and restarts construction from scratch. Taveira et al. call this "a primary limitation on non-uniform point distributions". Their reimplementation of Basselin completes only the Poisson-distributed cases (e.g. 1.810 s at 5M points on an RTX 5090). [arXiv HTML 2605.06408](https://arxiv.org/html/2605.06408)
- **Ray, Sokolov, Lefebvre, Lévy, "Meshless Voronoi on the GPU", ACM TOG 37(6), 2018 (SIGGRAPH Asia).** [HAL](https://hal.archives-ouvertes.fr/hal-01927559), [SIGGRAPH archive](https://history.siggraph.org/?p=198036)
  - Reported: each cell is computed independently in parallel from its nearest neighbors. Memory per thread is constant and no synchronization primitives are used. It is about an order of magnitude faster than the fastest multi-core CPU implementations of the time. Source code is in the supplemental material. [HAL](https://hal.archives-ouvertes.fr/hal-01927559)
- **Singh, Byrohl, Nelson, "votess: A multi-target, GPU-capable, parallel Voronoi tessellator", arXiv:2412.04514 (Dec 2024; submitted to JOSS).** [arXiv](https://arxiv.org/abs/2412.04514), [HTML v2](https://arxiv.org/html/2412.04514v2)
  - Uses SYCL. Points are "sorted into a grid", then a k-nearest-neighbor search runs. A "security radius" condition certifies each cell, and "If a cell cannot be validated, an automatic CPU fallback mechanism ensures robustness." Its CPU version beats other implementations "by a factor of 10 to 100". "votess performs best on GPUs with large datasets", but no GPU numbers appear in the text. Divergence and load imbalance are not discussed. [HTML v2](https://arxiv.org/html/2412.04514v2)
- **gDel3D: Cao, Nanjappa, Qi, Tan, "A GPU accelerated algorithm for 3D Delaunay triangulation", I3D 2014.** [DOI 10.1145/2556700.2556710](https://www.doi.org/10.1145/2556700.2556710), [project page](https://www.comp.nus.edu.sg/~tants/gdel3d.html)
  - Massively parallel insertion and flipping on the GPU give a nearly-Delaunay mesh. "When many points are inserted in parallel, flipping gets stuck long before reaching the DT", so insertion and flipping are alternated. The CPU finishes with conservative star splaying, building stars only for points in non-locally-Delaunay facets. [project page](https://www.comp.nus.edu.sg/~tants/gdel3d.html), [Nanjappa thesis](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf)
  - Up to 10x faster than CGAL's 3D Delaunay. [project page](https://www.comp.nus.edu.sg/~tants/gdel3d.html)
  - On strongly clustered input (K=10) it fails beyond 5M points (45.4 s at 5M on an H200) in Taveira et al.'s benchmark. [arXiv HTML 2605.06408](https://arxiv.org/html/2605.06408)
- **Radiant Foam: Govindarajan, Rebain, Yi, Tagliasacchi, ICCV 2025.** [arXiv 2502.01157](https://arxiv.org/pdf/2502.01157), [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Govindarajan_Radiant_Foam_Real-Time_Differentiable_Ray_Tracing_ICCV_2025_paper.html)
  - Builds a Voronoi/Delaunay structure with custom CUDA kernels and traverses it for rendering. This comes from a secondary summary only. [radiancefields.com](https://radiancefields.com/radiant-foam-radfoam)
  - Its Delaunay construction takes about 40 s on the Garden scene, against 0.5-0.6 s for Taveira et al. [arXiv HTML 2605.06408](https://arxiv.org/html/2605.06408)

**Ray and path tracing**

- **Aila & Laine, "Understanding the Efficiency of Ray Traversal on GPUs", HPG 2009.** [NVIDIA PDF](https://research.nvidia.com/sites/default/files/pubs/2009-08_Understanding-the-Efficiency/aila2009hpg_paper.pdf), [EG DL](https://diglib.eg.org/handle/10.2312/EGGH.HPG09.145-150)
  - Earlier methods were "1.5–2.5X off from theoretical optimum". Most of the gap was due to "inefficiencies in hardware work distribution", not memory bandwidth. [NVIDIA PDF](https://research.nvidia.com/sites/default/files/pubs/2009-08_Understanding-the-Efficiency/aila2009hpg_paper.pdf)
  - The fix: persistent threads that fetch rays from a global pool, while-while traversal and speculative traversal. Per a secondary summary, this reached about 80-90% of the simulated optimum, and speculative traversal raised SIMD efficiency. [student summary, secondary](https://folk.idi.ntnu.no/elster/tdt24/tdt24-f09/holger-summary.pdf)
  - The Fermi/Kepler addendum (Aila, Laine, Karras 2012) found that ray tracing is still "almost entirely limited by the available FLOPS". [NVIDIA poster](https://research.nvidia.com/sites/default/files/pubs/2012-06_Understanding-the-Efficiency/nvr-2012-02-poster.pdf)
- **Laine, Karras, Aila, "Megakernels Considered Harmful: Wavefront Path Tracing on GPUs", HPG 2013.** [NVIDIA PDF](https://research.nvidia.com/sites/default/files/publications/laine2013hpg_paper.pdf), [EG DL](https://diglib.eg.org/handle/10.2312/14940)
  - A large pool of paths is kept alive at all times. The path tracer is split into specialized kernels so ray casts and material evaluations run in coherent chunks. This is "much better suited for real-world use cases where multiple complex materials are present". [NVIDIA publication page](https://research.nvidia.com/publication/megakernels-considered-harmful-wavefront-path-tracing-GPUs)
  - Exact speedup numbers could not be extracted (see Gaps).
- **Meister, Bokšanský, Guthe, Bittner, "On Ray Reordering Techniques for Faster GPU Ray Tracing", I3D 2020 (arXiv 2506.11273).** [arXiv HTML](https://arxiv.org/html/2506.11273v1)
  - In wavefront path tracing with RTX trace kernels, reordering gives a 1.3-2.0x trace speedup, but it is a problem to recover the reordering overhead within the hardware-accelerated trace phase. [arXiv HTML](https://arxiv.org/html/2506.11273v1)
  - A later paper reads this as large overhead "often leads to low speedup or even negative speedup on RTX platforms". [Hierarchy Cut Code, arXiv 2305.16652](https://arxiv.org/pdf/2305.16652)
- **NVIDIA Shader Execution Reordering (Ada and later).** [SER whitepaper](https://developer.nvidia.cn/sites/default/files/akamai/gameworks/ser-whitepaper.pdf)
  - Claims "up to 2x" in real-world ray tracing. [SER whitepaper](https://developer.nvidia.cn/sites/default/files/akamai/gameworks/ser-whitepaper.pdf)
  - Khronos-reported game gains are +24% (Indiana Jones), +39% (Alan Wake 2) and +370% (Black Myth: Wukong); the hardware is not stated clearly. [TechSpot](https://www.techspot.com/news/111536-directx-made-shader-execution-reordering-official-boosting-ray.html)
  - Analysis of the mechanism: [Chips and Cheese](https://chipsandcheese.com/p/shader-execution-reordering-nvidia-tackles-divergence)
- **Galgoczi, "Mitigating warp divergence in GPU optical photon Monte Carlo: an order of magnitude speedup with Shader Execution Reordering", arXiv:2608.21396 (Aug 2026; preprint).** [arXiv](https://arxiv.org/abs/2608.21396)
  - The problem: photons have widely varying lifetimes, so a few long-lived photons hold their warp while other threads idle. This is the same long-tail shape as our cells.
  - SER regroups surviving photons. Propagation-kernel cycles drop 16.55x and end-to-end time drops 14.72x, with bit-identical output.
  - Breakdown: 3.2x from launch-configuration changes that come with SER, 5.1x from the reorder itself, and 1.13x from a volume-coherent hint.
  - The dominant mechanism is "active lane recovery". Branch efficiency and memory bandwidth "change little". [arXiv](https://arxiv.org/abs/2608.21396)

**Particle methods**

- **Burtscher & Pingali, "An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body Algorithm", GPU Computing Gems Emerald, 2011.** [PDF](https://userweb.cs.txstate.edu/~mb92/papers/gcg11.pdf)
  - 5M bodies take 5.2 s per step on a Quadro FX 5800, 74x faster than an optimized serial code. Similar work is grouped together to minimize divergence. [O'Reilly chapter page](https://www.oreilly.com/library/view/gpu-computing-gems/9780123849885/B9780123849885000061.xhtml)
  - Bodies are sorted spatially so warp-neighbors traverse similar tree parts. A warp vote on the opening test makes all lanes take the same decision; it "avoids divergence but the beneficial effect of the opening angle test is lost for some threads". [FLASH4 BH, arXiv 1509.07370](https://arxiv.org/pdf/1509.07370), [DynaSOAr paper, arXiv 1908.05845](https://arxiv.org/pdf/1908.05845)
- **Páll & Hess, "A flexible algorithm for calculating pair interactions on SIMD architectures", CPC 184 (2013) 2641-2650, DOI 10.1016/j.cpc.2013.06.003 (arXiv 1306.1737).** [abstract](https://www.lammps.org/abstracts/abstract.3709.html)
  - Particles are grouped into fixed-size spatial clusters (2, 4 or 8). All pairs between two clusters are computed, with the cluster size matched to the SIMD width, including GPUs. Variable neighbor counts become fixed-shape work units. [abstract](https://www.lammps.org/abstracts/abstract.3709.html)
- **HOOMD-blue.** [arXiv 1308.5587](https://arxiv.org/pdf/1308.5587), [NVIDIA forum](https://forums.developer.nvidia.com/t/md-neighbour-list-update-how-to-re-construct-update-neighbour-lists-on-gpu/23594)
  - The cell-list build is one thread per particle with atomics, about 10x faster than a host build on GF100. [arXiv 1308.5587](https://arxiv.org/pdf/1308.5587)
  - An early neighbor-list design needed the block size to be at least the largest cell occupancy, which "resulted in many wasted threads". It was then rewritten. [NVIDIA forum](https://forums.developer.nvidia.com/t/md-neighbour-list-update-how-to-re-construct-update-neighbour-lists-on-gpu/23594)
  - Kernel launch parameters are autotuned at runtime. [HOOMD tune docs](https://hoomd-blue.readthedocs.io/en/v4.0.0/module-md-tune.html)

### Mapping to our candidate fixes (1-6)

| Case | Fix | What it implies for us (inference) |
|---|---|---|
| Aila-Laine persistent threads, global pool, while-while | (1) | The canonical 1.5-2.5x case. It worked because the hardware block scheduler of that era distributed work badly; the 2012 addendum says traversal later became FLOP-bound. With 15M cells our grid is huge. Dynamic fetching only helps if a warp can refill lanes whose cell has finished; one cell per warp-lifetime does not do that. While-while with "fetch a new cell when my lane finishes" is the transferable idea. |
| Galgoczi SER photon MC (16.5x) | (5)/(1) in hardware | Strongest evidence that refilling idle lanes on a long tail pays off by an order of magnitude. A100 has no SER or OptiX-style reorder. The software equivalent is lane refill (1) or a compaction pass between rounds (4). |
| Laine 2013 wavefront | (4) | Pays when stages have very different register or occupancy footprints (heavy materials). Our clipping loop is one homogeneous stage, so the gain would come from compacting surviving cells between rounds, not from specialization. |
| Meister 2020 ray sorting | (5) | 1.3-2.0x in-kernel, but overhead often cancels it. Our cells take only ~65 ns each (15M/s), so a global sort per pass is a large relative cost. Use cheap keys (Morton, which we already have) or no sort. |
| Burtscher-Pingali BH: spatial sort + warp vote | (5)+(6)/(3) | Spatial sorting puts similar-cost items in one warp. Warp-uniform decisions trade wasted work for zero divergence. Analog: lanes in a warp jointly decide "keep clipping" (warp-level termination test). |
| Páll-Hess cluster pairs | (3)/(6) | Reshape variable work into fixed SIMD-width tiles. Analog: a warp clips 4-8 cells together, each lane testing one (cell, plane) pair or one vertex. |
| HOOMD block ≥ max occupancy | anti-pattern for (3) | Sizing a team to the worst-case cell wastes lanes. Our 15.6/32 warp-per-cell utilization is this pathology. |
| Taveira 2026: global-memory per-thread state, 85% compaction | occupancy (not 1-6) | Their fix for low occupancy was to move per-cell state out of the register and shared footprint into coalesced global memory. Our occupancy is 8-17% and L2 hit is 99%, so the same move is plausible: trade some L2 traffic for 2-4x more resident warps to hide the latency of the dependent clipping chain. |
| Basselin 2021: overflow → relaunch with larger arrays from scratch | (2) | Deferral works but restart-from-scratch is costly. We already rebuild overflowing cells with larger capacity (commit 0a53f4b), the same pattern. |
| votess: CPU fallback for uncertified cells | (2) | Same pattern. |
| gDel3D: GPU near-Delaunay + CPU star splaying | (2) | Same pattern. |

### Inferences
- Across the cases, the techniques with 2x or more all share one property: they let a SIMD unit pick up new work when some lanes finish early. That is persistent threads with a pool and while-while, wavefront compaction, or hardware reordering. Our diagnosis (5.4/32 active lanes per thread-per-cell warp, half the throughput on uniform-random input) matches the photon-lifetime problem in Galgoczi 2026, where active-lane recovery gave 5.1x from the reorder alone.
- Sorting by predicted cost (5) helps only if the key is cheap and predictive. The ray-sorting evidence says a heavyweight sort per pass rarely pays end-to-end on modern GPUs.
- Tail deferral (2) is universal in the GPU Voronoi and Delaunay literature (Basselin relaunch, votess CPU fallback, gDel3D CPU repair, Taveira compaction). None of them report it as a throughput optimization; it is used for correctness and robustness. Using a work cap deliberately to cut the tail, then finishing deferred cells in a compacted second pass, would be a new use of an established mechanism.
- On A100 FP64, branch-free formulations (6) mostly cut divergence inside the clipping step. They do not fix the variable trip count, which is the dominant loss for us.

### Gaps
- Exact megakernel-vs-wavefront speedups in Laine et al. 2013: the PDF was fetched but could not be text-extracted on this machine.
- Exact persistent-thread gains per GPU generation in Aila-Laine 2009/2012: abstract and secondary summary only. I recall the 2012 addendum reporting smaller benefits from persistent threads on Kepler, but could not verify it, so it is not stated as fact.
- Not researched within the tool budget, so no claims are made:
  - Cycles X wavefront and shader sorting results; pbrt-v4 wavefront.
  - gHull, CudaHull, GPU QuickHull.
  - GPU polygon clipping and cut-cell methods; GPU Delaunay refinement.
  - DEM contact detection; SPH-specific codes (DualSPHysics etc.).
  - Liu et al. 2020 3D Voronoi GPU; Gao & Chen "Local-DeWall".
  - Treelet scheduling (Aila & Karras 2010).

## Q2: Do any GPU Voronoi or Delaunay papers report lane utilization or explicitly attack divergence?

### Takeaway
None of the sources I could read report lane utilization, warp efficiency or branch efficiency for per-cell Voronoi or Delaunay construction. The newest and fastest (Taveira et al. 2026) explicitly maps one thread per cell and does not analyze divergence. It attacks occupancy (per-thread state moved to global memory) and the amount of work per cell (directional culling bounds plus best-first BVH traversal). Our measured 5.4/32 and 15.6/32 active lanes would be new data in this literature.

### Cited findings
- Taveira et al. 2026:
  - One thread per cell ("map one-to-one onto GPU threads").
  - Per-thread footprint "severely limit[ed] SM occupancy", fixed by moving that state to coalesced global memory.
  - Plane-array compaction at 85% fill.
  - No divergence or load-imbalance analysis in the text read. [arXiv HTML](https://arxiv.org/html/2605.06408)
- Ray et al. 2018 abstract: constant memory per thread and no synchronization. Divergence is not mentioned in the abstract; the full text could not be accessed. [HAL](https://hal.archives-ouvertes.fr/hal-01927559)
- votess: divergence and load imbalance are not discussed; it relies on grid sorting of points and a CPU fallback. [arXiv HTML v2](https://arxiv.org/html/2412.04514v2)
- Basselin 2021: overflow is handled by relaunching with larger arrays (as described by Taveira et al.). [arXiv HTML 2605.06408](https://arxiv.org/html/2605.06408)
- gDel3D: parallel flipping "gets stuck", leaving a tail that is resolved on the CPU. This is tail deferral at algorithm level, not a SIMT-divergence fix. [project page](https://www.comp.nus.edu.sg/~tants/gdel3d.html)

### Inferences
- Our ~15M cells/s in FP64 on an A100 (jittered input) is in the same range as Taveira et al.'s 5.9-9.4M cells/s. Theirs is FP32 on an H200, end-to-end including the BVH build, on harder distributions. The comparison is not like-for-like.
- Because no Voronoi or Delaunay code reports lane utilization, the closest transferable evidence for fixes (1), (3) and (4) comes from ray traversal (Aila-Laine), wavefront path tracing (Laine 2013), SER photon transport (Galgoczi 2026), Barnes-Hut (Burtscher-Pingali) and MD cluster pairs (Páll-Hess).
- Taveira's occupancy-first approach is the one directly applicable Voronoi-domain lesson for our 8-17% occupancy.

### Gaps
- Full text of Ray et al. 2018 and Basselin et al. 2021 (HAL and EG blocked automated access). Unverified: their k values, per-thread capacity limits, block sizes, failure fractions, and any CPU fallback in Ray 2018.
- Whether Taveira et al. discuss divergence in the last ~12% of the HTML, which was not read.
- Liu et al. 2020 (3D Voronoi GPU) and Gao & Chen's Local-DeWall were not checked.
- No 2024-2026 GPU Voronoi or Delaunay paper beyond Taveira, votess and Radiant Foam was found in the searches run.
