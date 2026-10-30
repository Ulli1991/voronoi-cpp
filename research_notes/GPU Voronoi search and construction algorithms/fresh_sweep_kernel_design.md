# GPU kernel design for many small independent geometric problems (2010 to Oct 2026): fresh independent sweep

Scope note: this sweep ran about 20 tool calls. Several primary PDFs could not be rendered (EG digital library refused the connection, HAL blocked by bot protection). Where a number comes from an abstract or a search snippet rather than the full text, it is marked "(abstract)" or "(unverified, secondary)". ProteusGPU was not used as a source.

## Q1. Which mappings and state layouts gave the largest measured speedups for batched small geometric problems?

### Takeaway
The best-measured evidence comes from GPU ray tracing, the most heavily benchmarked "many small independent traversals" workload. There, three things helped: one thread per element with an independent per-element stack, persistent threads that fetch work from a global counter, and fewer bytes per step (compressed nodes, a compressed or shared-memory stack). Persistent threads alone gave about 1.5-2x on a GTX285. Compressed wide BVHs with a compressed stack gave a further 1.9-2.1x on Pascal and Maxwell. For per-cell Voronoi clipping, the newest paper (2026) deliberately moves per-cell state out of shared memory into coalesced global memory to raise occupancy. It replaces overflow-and-relaunch with in-place compaction. Splitting a divergent kernel into stages joined by stream compaction is the other repeatedly measured win: 1.31x to 2.73x in GPU narrow-phase collision.

### Cited Findings
- **Aila & Laine 2009 (HPG; foundational, slightly before 2010), NVIDIA GTX285, 1024x768, Conference scene (282K tris):**
  - Non-persistent per-ray "while-while" traversal: 88.0 Mrays/s primary, 86.3 AO, 44.5 diffuse.
  - Warp-packet traversal (shared stack, Günther 2007): only 63.6 / 39.4 / 16.6 Mrays/s.
  - Persistent while-while: 135.6 / 130.7 / 62.4 Mrays/s.
  - Persistent packet: 122.1 / 86.1 / 32.3 Mrays/s.
  - Persistent speculative while-while: 142.2 / 134.5 / 60.9 Mrays/s.
  - Measured performance rose from about 43-64% to about 77-92% of the simulated upper bound.
  - The authors attribute the 1.5-2.5x gap from optimum mainly to "inefficiencies in hardware work distribution", not memory bandwidth.
  - Per-ray stacks were kept in thread-local memory.
  - — [Aila & Laine 2009, Table 1 and Sections 3.1-3.4](https://users.aalto.fi/~laines9/publications/aila2009hpg_paper.pdf)
- **Packets lose in that study.** A warp-sized packet shares one traversal stack and votes with `__any()`. Each node is fetched once per packet, so memory access is coherent. But rays visit nodes they do not intersect, and packet traversal measured 1.7-2.4x below simulation. Per-ray traversal "is faster than packet traversal in all cases" once persistent threads are used. With diffuse rays the gap is about 2x (Conference: 62.4 vs 32.3 Mrays/s). — [Aila & Laine 2009](https://users.aalto.fi/~laines9/publications/aila2009hpg_paper.pdf)
- **SIMD-efficiency tuning (Aila & Laine 2009):**
  - "if-if" loop organisation has higher traversal SIMD efficiency than "while-while" (e.g. 81/56% vs 78/63% trav/isect for Conference primary rays). Its slowest warps are about 30% faster.
  - Speculative traversal improves primary rays by up to 5% and AO by up to 10%, but gives no average gain for diffuse rays.
  - — [Aila & Laine 2009](https://users.aalto.fi/~laines9/publications/aila2009hpg_paper.pdf)
- **Ylitie, Karras & Laine 2017 (HPG), NVIDIA Titan X (Pascal) and GTX Titan X (Maxwell):**
  - Method: compressed 8-wide BVH, compressed 64-bit traversal-stack entries, octant-ordered child traversal, and a 12-entry shared-memory stack. It also uses persistent threads and dynamic ray fetching, adopted from Aila & Laine.
  - Result: "1.9-2.1x improvement in incoherent ray traversal performance compared to the current state of the art" (Aila et al. 2012).
  - Hierarchy memory is 35-60% of a typical uncompressed BVH.
  - Their key argument is that GPUs "are highly sensitive to the amount of memory traffic".
  - — [Ylitie et al. 2017](https://users.aalto.fi/~laines9/publications/ylitie2017hpg_paper.pdf)
- **Aila, Laine & Karras 2012 addendum (Fermi/Kepler):** ray tracing "is still, even with incoherent rays and more complex scenes, almost entirely limited by the available FLOPS". — [NVR-2012-02 poster](https://research.nvidia.com/sites/default/files/pubs/2012-06_Understanding-the-Efficiency/nvr-2012-02-poster.pdf) (abstract-level, via search snippet)
- **Scalable GPU Construction of 3D Voronoi and Power Diagrams (arXiv 2605.06408, 2026):**
  - Mapping and state:
    - One thread per cell. Construction "decomposes naturally into N independent volume-clipping tasks, which we map one-to-one onto GPU threads".
    - A single fused kernel does best-first BVH traversal, directional culling and clipping.
    - Sites are packed as float4 (position plus weight) for 128-bit coalesced loads.
    - Per-cell state lives in coalesced global memory, not shared memory. The authors say Basselin et al.'s shared-memory footprint "severely limit[s]" SM occupancy.
    - Planes are stored as float4. Vertices are uchar3 triples of plane indices, with cached vertex positions.
    - The fixed plane array is compacted ("garbage collection") at 85% occupancy, instead of Basselin's overflow-and-relaunch-with-larger-arrays.
  - Search:
    - A per-thread BVH node stack, where a pop is a linear scan for the nearest node.
    - Optional KNN (K=8) warm-start pre-clip: on for the RTX 5090, off for the H200.
    - BVH leaf size is 17 on the H200 and 10 on the 5090.
  - Measured times:
    - 4.2M-point "bicycle" scene: 0.570 s on the RTX 5090, 0.684 s on the H200.
    - 10M clustered points: 1.423 s on the 5090, 1.848 s on the H200.
    - 15M clustered points: 2.530 s on the H200.
    - That is roughly 6M cells/s on the H200 (my derivation; the paper does not state a rate).
  - Versus gDel3D:
    - gDel3D is faster at small clustered sizes (H200, 2M points: 0.576 s vs 1.104 s).
    - gDel3D fails beyond about 5M points.
    - On real-world scenes the new method is on average about 55% faster on the H200 and about 137% faster on the 5090.
  - Versus Basselin et al. 2021 (CUDA port): it completed only the Poisson-like distributions.
  - — [arXiv 2605.06408 (HTML full text, first ~100K chars read)](https://arxiv.org/html/2605.06408v1)
- **Ray, Sokolov, Lefebvre & Lévy 2018, "Meshless Voronoi on the GPU" (ACM TOG 37(6)):**
  - Each cell is computed independently from its nearest neighbours, with "constant memory usage per thread and no synchronization primitives".
  - Claimed speedup is "one order of magnitude" over the fastest multi-core CPU implementations (abstract).
  - Cells are integrated by on-the-fly tetrahedral decomposition.
  - — [HAL hal-01927559](https://hal.archives-ouvertes.fr/hal-01927559) (abstract only; GPU model and per-cell layout not verified here)
- **Basselin et al. 2021, "Restricted Power Diagrams on the GPU" (CGF 40(2)):** reported 10M points restricted to a mesh in 300 ms on an NVIDIA V100. — (unverified, secondary: via [arXiv 2412.04514 citation/search snippet](https://arxiv.org/html/2412.04514v2); primary [EG PDF](https://diglib7.eg.org/bitstream/handle/10.1111/cgf142610/v40i2pp001-012.pdf) could not be fetched)
- **Lehericey, Gouranton & Arnaldi 2015 (GRAPP), GPU ray-traced narrow-phase collision, OpenCL in Bullet 3.x, AMD HD 7990:**
  - Splitting one monolithic ray-tracing kernel into fine-grained stages, joined by append buffers / stream compaction, ensures "a dense input to the following kernel".
  - Putting different algorithms in different kernels reduces branch divergence.
  - Measured average speedups: 1.31x with the full BVH ray-tracing algorithm only, and 2.73x with combined full and iterative ray-tracing.
  - In mixed scenes: 1.84x average, and 2.0-2.84 ms per time-step faster.
  - — [Lehericey et al. 2015 PDF](https://www.scitepress.org/Papers/2015/52996/52996.pdf)
- **McGuire et al. 2011, triangle/quad clipping within shaders:** reported about 5x faster than direct Sutherland-Hodgman, and a 45% net throughput increase in the host algorithm on two GPU architectures. — [McGuire 2011 project page](https://casual-effects.com/research/McGuire2011Clipping/index.html) (search-snippet level; GPU models not verified)
- **Kaehler 2017 (Astronomy & Computing 20), tetrahedron vs cubical-cell clipping for phase-space density deposits:**
  - Claimed "almost an order of magnitude" over an optimised CPU version (abstract).
  - Scaled to 256 GPUs and more than 400 billion tetrahedra.
  - Design note from search summary (unverified): each intermediate tetrahedron is clipped against all cell faces rather than caching intermediates in shared memory, trading redundant arithmetic for less data movement.
  - — [arXiv 1612.09491](https://arxiv.org/abs/1612.09491)
- **Zhang et al. 2015, GPU QuickHull for one large planar set:** built from Thrust primitives such as segmented scan; up to 10.98x over Qhull. — [arXiv 1501.04706](https://arxiv.org/pdf/1501.04706) (abstract-level)

### Inferences
- Across the ray-tracing literature, the largest single measured win for "many small irregular traversals" was persistent threads plus a global work counter: about 1.5-2x at fixed algorithm on GTX285. The next largest was cutting bytes moved per step: about 2x from compressed nodes and stack. Choosing per-thread vs per-warp mapping mattered less than these, as long as per-element state was not forcing low occupancy.
- The 2026 Voronoi paper and Kaehler's design note both point the same way. For half-space clipping with variable vertex counts, keeping a big per-element polyhedron in shared memory is an occupancy trap. Coalesced global (L1/L2-cached) or register state, plus recomputation, is the current preference. This is inferred from design statements, not from a published head-to-head ablation in those papers.
- Kernel splitting with compaction (Lehericey) is the measured answer to per-element branch divergence when there are distinct algorithmic phases. Compacting element lists between "search" and "clip" stages is the analogous move for Voronoi construction.

### Gaps
- No ablation was found that directly compares thread-per-cell, warp-per-cell and block-per-cell for 3D Voronoi half-space clipping on the same hardware.
- I could not extract Ray et al. 2018's per-thread array sizes, GPU model or cells/s from the primary text. Basselin 2021's kernel details (shared memory size, retry mechanism) also remain unverified from the primary text.
- I found no batched "many tiny convex hulls" GPU paper with measured numbers. Published GPU hull work targets one large set: QuickHull and CudaChain, plus filtering such as [Carrasco et al. 2024](https://meshinglab.dcc.uchile.cl/publication/carrasco-2024104793/carrasco-2024104793.pdf).
- For Bullet 3 OpenCL SAT and contact clipping, the GPU pipeline is documented ([bullet3 fork README](https://github.com/RimmerM/bullet3); [DeepWiki](https://deepwiki.com/bulletphysics/bullet3/7.4-opencl-gpu-acceleration)), but no throughput numbers were found.

## Q2. Is there evidence that processing groups of spatially neighbouring elements together (sharing candidate/neighbour data, packets) beats independent per-element processing?

### Takeaway
The best controlled evidence (Aila & Laine 2009) says no, at least for warp-sized packets that share one traversal stack. Independent per-ray traversal with persistent threads beat persistent packet traversal in every measured case on GTX285. The margin was about 1.1x for primary rays and about 2x for incoherent diffuse rays, because packets visit nodes that individual members do not need. Later state of the art (Ylitie 2017) keeps per-ray traversal and gains instead by cutting memory traffic. No reviewed Voronoi/clipping paper shares candidate lists between neighbouring cells. Each cell does its own KNN or BVH search.

### Cited Findings
- Packet (shared-stack) vs independent per-ray traversal, persistent variants, GTX285, Mrays/s (packet vs per-ray while-while):
  - Conference: primary 122.1 vs 135.6, AO 86.1 vs 130.7, diffuse 32.3 vs 62.4.
  - Sibenik: primary 100.6 vs 117.5, AO 83.7 vs 113.4, diffuse 27.3 vs 48.1.
  - — [Aila & Laine 2009, Table 1](https://users.aalto.fi/~laines9/publications/aila2009hpg_paper.pdf)
- Packets give coherent memory access ("each node is fetched only once per packet"), yet were "a factor of 1.7-2.4 off from simulations" even for coherent primary rays. — [Aila & Laine 2009](https://users.aalto.fi/~laines9/publications/aila2009hpg_paper.pdf)
- Ylitie et al. 2017 keep per-thread rays with persistent threads and dynamic ray fetching, using a 12-entry shared-memory stack. Their gain (1.9-2.1x) comes from memory-traffic reduction, not ray grouping. — [Ylitie et al. 2017](https://users.aalto.fi/~laines9/publications/ylitie2017hpg_paper.pdf)
- 2026 GPU power diagrams: each thread runs its own best-first BVH traversal. The paper describes no candidate sharing between neighbouring cells. — [arXiv 2605.06408](https://arxiv.org/html/2605.06408v1)
- Ray et al. 2018 compute each cell "independently, in parallel, using only its nearest neighbors". — [HAL hal-01927559](https://hal.archives-ouvertes.fr/hal-01927559)

### Inferences
- In these studies, "packet" sharing helps only when the elements' working sets overlap strongly. For Voronoi cells, adjacent cells' candidate sets do overlap heavily. So a cheaper form of sharing could still pay off without forcing lockstep traversal. An example is a block cooperatively loading a spatial tile's points into shared memory or L1, with each thread still clipping independently. I found no direct measurement of this for 3D Voronoi in the reviewed sources.
- The practical lesson from ray tracing is to sort or order elements spatially (coherence through ordering and caches) while keeping per-element control flow independent. This is inference, not a measured Voronoi result.

### Gaps
- No paper was found that measures tile-shared candidate lists vs per-cell independent neighbour search for GPU Voronoi or half-space intersection.
- I did not retrieve post-2017 packet studies (ray streams, e.g. Barringer & Akenine-Möller "Dynamic ray stream traversal", 2014, [SIGGRAPH archive](https://history.siggraph.org/?p=103631)) with numbers. That work is CPU-oriented.

## Q3. What do mesh-intersection/remap codes on GPUs (e.g. Portage/Tangram, R3D/R2D, Kokkos-based remap) report for per-element clipping throughput?

### Takeaway
Published per-element GPU clipping throughput for remap codes is very thin. Portage documents Thrust-based on-node parallelism, with OpenMP as the default and "no support yet for CUDA" on its site. A 2018 LANL report on a simpler volume-fraction/centroid kernel quotes 10-100x GPU speedup without per-element rates. The LANL FANG note (Jones, Shevitz, Garimella; IMR 2024/25) explicitly argues that clip-and-cap with R3D-style variable-complexity kernels is hard on GPUs. It proposes decomposing polyhedra into tetrahedra and running fixed-complexity tet-tet intersections with 5 precomputed tet-plane cut cases, static arrays and bounding-box pruning. But it reports CPU results only, with GPU porting "planned as the next steps". No measured R3D/R2D GPU throughput was found.

### Cited Findings
- Portage: on-node parallelism is through Thrust, with OpenMP the default and no CUDA support yet (site statement). The SC17 poster says it is fully tested with OpenMP, an early first-order remap was tested with CUDA, and the lack of on-node scaling was likely due to poor search scaling. — [portage.lanl.gov](https://portage.lanl.gov); [SC17 poster P07](https://sc17.supercomputing.org/SC17%20Archive/tech_poster/tech_poster_pages/post169.html)
- Q.-T. Ha (LANL, 2018), "GPU Acceleration of Volume Fraction and Centroid Computation from General Shapes on Unstructured Mesh":
  - Uses Thrust with the CUDA backend.
  - Speedup "pronounced (10-100 times)" vs CPU.
  - Thrust needed substantial refactoring and gave less fine-grained control than raw CUDA.
  - Hardware not given on the landing page.
  - — [OSTI 1471309](https://www.osti.gov/biblio/1471309)
- FANG (Jones, Shevitz, Garimella), "Fast Approach for intersecting Numerous Polyhedra with scalability on GPUs":
  - Argues that clip-and-cap is hard to port because "the computational complexity of a kernel that clips an arbitrary polyhedron with a plane can be highly variable whereas GPUs ... prefer simple kernels with fixed complexity".
  - Method: tet decomposition, bounding-box pre-pruning, "static arrays to avoid dynamic memory allocation", five tet-plane geometric cases with precomputed topologies.
  - Results are on CPUs, with GPU testing future work.
  - A search snippet says a Kokkos prototype "outperforms the r3d library as the number of tetrahedra increases". I could not verify that in the extracted text.
  - — [IMR33 research note PDF](https://internationalmeshingroundtable.com/assets/research-notes/imr33/2011-compressed.pdf)
- R3D lineage: LLNL PolyClipper reimplements R3D's plane clipping (Powell & Abel 2015) and removes R3D's hard-coded vertex limits. No GPU numbers were found. — [PolyClipper docs](https://polyclipper.readthedocs.io/en/latest/intro.html); [LLNL news](https://software.llnl.gov/news/2020/06/19/polyclipper-new/)
- A 2026 arXiv field-transfer paper builds local supermeshes with "the R3D algorithm as implemented in the Omega_h library" and says this is "naturally suited to GPU execution". Throughput numbers were not verified. — [arXiv 2603.00538](https://arxiv.org/pdf/2603.00538) (search snippet)
- MOAB developers (2021) gave up making MOAB's AdaptiveKdTree GPU-compatible and used ArborX (Kokkos) for element queries. Even on OpenMP, ArborX was "orders of magnitude faster" than the MOAB kd-tree. This concerns search, not clipping. — [moab-dev mailing list](https://lists.mcs.anl.gov/pipermail/moab-dev/2021/007604.html)
- Kaehler 2017: tet-vs-cube clipping on GPU, "almost an order of magnitude" over optimised CPU (abstract). — [arXiv 1612.09491](https://arxiv.org/abs/1612.09491)

### Inferences
- The remap community's documented GPU design response to variable clip complexity is to bound per-element work. This means fixed-size static arrays, decomposition into fixed-topology primitives (tets) and lookup tables for cut cases. That is a different answer from the Voronoi community's: one thread per cell, a variable-size polyhedron in global memory, and compaction.
- For remap codes, search/candidate finding (not clipping) was repeatedly identified as the on-node bottleneck (Portage poster, MOAB thread). This mirrors Voronoi construction, where neighbour search often dominates.

### Gaps
- No measured per-element GPU clipping throughput (polyhedra/s or clips/s) with named hardware was found for Portage/Tangram, R3D/R2D, PolyClipper, Omega_h or any Kokkos ALE remap code as of Oct 2026.
- The FANG Kokkos-vs-r3d comparison numbers and hardware could not be extracted; only the first ~700 words of the PDF were readable.
- I found no GPU numbers for Tangram (interface reconstruction).
