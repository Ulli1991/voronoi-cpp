# Local (cell-by-cell) Voronoi construction and GPU 3D Voronoi/Delaunay: literature catalogue (to Oct 2026)

Search date: 2026-10-29. Several primary PDFs (HAL copy of Ray et al. 2018, Eurographics digital library copy of Basselin et al. 2021, ResearchGate) were blocked by bot protection (Anubis / HTTP 403), so some details of those two papers come from search snippets or third-party summaries. Where that happens it is marked **[snippet]** or **[unverified]**.

## Q1. Verify and extract numbers: Ray et al. 2018, votess 2024, Taveira et al. 2026, voro++

### Takeaway
All four works exist and are confirmed. The best published 3D GPU throughput figures are:
- **Ray et al. 2018**: 10M points in 800 ms on a V100, about 12.5M cells/s.
- **Basselin et al. 2021**: 10M points restricted to a mesh in 300 ms on a V100.
- **Taveira et al. 2026**: 10M points in about 1.1 s end-to-end on an RTX 5090 or H200, about 9M cells/s. These are single-precision runs on non-uniform data and include the BVH build. Unlike the first two, this method also handles strongly clustered data.

votess gives only a log-scale chart, at roughly 1e-6 s per point on an A100 at N = 1e6. Serial voro++ does about 4.8e4 cells/s in 3D.

### Cited Findings

**Ray, Sokolov, Lefebvre, Lévy 2018, "Meshless Voronoi on the GPU"** (ACM TOG 37(6), Art. 265, SIGGRAPH Asia 2018, doi 10.1145/3272127.3275092)
- Each 3D Voronoi cell is computed independently from its k nearest neighbours, with no synchronization, by clipping against bisector planes. The paper claims about one order of magnitude speed-up over the fastest multi-core CPU implementations of the time. — [HAL record](https://hal.archives-ouvertes.fr/hal-01927559); [ACM DL](https://dl.acm.org/doi/abs/10.1145/3272127.3275092)
- Teaser figure: "Voronoi diagram of 10 million points computed on the GPU in 800 ms (NVidia V100)… corresponds to 84 million Delaunay tetrahedra per second." Tetrahedra are not computed explicitly. **[snippet]** — [ACM DL](https://dl.acm.org/doi/abs/10.1145/3272127.3275092)
  - My arithmetic: 10M / 0.8 s = 12.5M cells/s.
- k is "relatively low (typically 35 to 180)". Cells whose Delaunay degree exceeds the threshold "need special treatment on the CPU". **[snippet]** — [ResearchGate copy](https://www.researchgate.net/publication/329139164_Meshless_Voronoi_on_the_GPU)
- Robustness: the meshless approach "will just produce sometimes a facet with zero area". It does not use consistent exact predicates as classical Delaunay codes do. **[snippet]** — [ACM DL](https://dl.acm.org/doi/abs/10.1145/3272127.3275092)
- Not retrieved: the exact GPU precision (float vs double), the failure/fallback percentage, the per-thread workspace sizes, and the benchmark tables. **[unverified]**

**Basselin, Alonso, Ray, Sokolov, Lefebvre, Lévy 2021, "Restricted Power Diagrams on the GPU"** (CGF 40(2), Eurographics 2021, doi 10.1111/cgf.142610)
- Figure 1: 3D Voronoi diagrams of 10 million points restricted to a mesh in 300 ms on an Nvidia V100. **[snippet]** — [EG diglib](https://diglib.eg.org/handle/10.1111/cgf142610); [HAL](https://hal.archives-ouvertes.fr/hal-03169575)
- The abstract claims a 2 to 3x speed-up over state-of-the-art methods for unrestricted Voronoi, and x50 over CPU implementations for restricted power diagrams. **[snippet]** — [EG diglib](https://diglib.eg.org/handle/10.1111/cgf142610)
- How Taveira et al. 2026 describe the method:
  - kNN with a growing radius; the search "terminates at a fixed multiple of the current cell radius".
  - It assumes near-uniform sites.
  - Per-thread intermediate data sits in shared memory.
  - On overflow, "the kernel is relaunched with larger array sizes, reattempting construction from scratch".
  - It "fails to complete" clustered test cases.
  - Source: [Taveira et al. arXiv HTML](https://arxiv.org/html/2605.06408)
- Third-party re-implementation ("Basselin\*", single precision):
  - Density gradient, 10M points: 3.602 s on an RTX 5090 and 3.311 s on an H200.
  - White noise: 0.355 s at 1M and 3.608 s at 10M on the RTX 5090.
  - Source: [Taveira et al.](https://arxiv.org/html/2605.06408)
  - These are much slower than the original 300 ms. Hardware and configuration differ, so the numbers are not directly comparable.

**Taveira, Lindström, Fatemi, Hammarstrand, Kahl 2026, "Scalable GPU Construction of 3D Voronoi and Power Diagrams"** (arXiv 2605.06408, submitted 7 May 2026, cs.CG; ACM doi 10.1145/3799902.3811229). **Verified that it exists.** — [arXiv abs](https://arxiv.org/abs/2605.06408); [ACM DL](https://dl.acm.org/doi/full/10.1145/3799902.3811229); [Chalmers](https://research.chalmers.se/en/publication/553368)
- Algorithm:
  - Each cell is built by progressively clipping an initial volume against bisecting planes of candidate neighbours.
  - Neighbour search is a best-first traversal of a BVH with a "directional culling" criterion. It does not use kNN.
  - A candidate or BVH node is discarded when the lower bound on its bisector distance exceeds r_i. Here r_i is the maximum distance from the site to the cell-AABB corners in the candidate's octant.
  - The method extends to power diagrams.
  - Source: [arXiv HTML](https://arxiv.org/html/2605.06408)
- Implementation:
  - One GPU thread per site.
  - Sites are packed as float4 (xyz plus weight, 128-bit aligned) so loads coalesce.
  - Planes are stored as float4. Vertices are stored as uchar3 triples of plane indices, and vertex positions are cached in global memory, not shared memory.
  - The plane array has a fixed size and is compacted (fully clipped planes dropped) at 85% occupancy. This replaces Basselin's relaunch-on-overflow.
  - KNN warm start with K=8 is enabled on the RTX 5090 and disabled on the H200.
  - BVH leaf size is 17 on the H200 and 10 on the RTX 5090.
  - Source: [arXiv HTML](https://arxiv.org/html/2605.06408)
- Precision and consistency:
  - Single precision. The method "relies on floating-point arithmetic and does not target the exact geometric guarantees of CPU-based libraries".
  - Against CGAL exact predicates, 0.1–0.2% of adjacencies mismatch on real data, attributed to near-coincident candidate planes.
  - Source: [arXiv HTML](https://arxiv.org/html/2605.06408)
- Hardware and timing protocol:
  - GPUs: RTX 5090 and H200. CPU baselines run on 16 cores of an AMD EPYC 9534, in double precision.
  - Timing is end-to-end, with three warm-up runs and the mean of ten timed runs.
  - Source: [arXiv HTML](https://arxiv.org/html/2605.06408)
- Real scenes, Table 1 (Voronoi; seconds, RTX 5090 / H200):

| Scene | Ours | gDel3D | Geogram (CPU) | CGAL (CPU) |
|---|---|---|---|---|
| bicycle (4.2M) | 0.570 / 0.684 | 1.204 / 0.893 | 1.593 | 20.445 |
| garden (4.1M) | 0.496 / 0.574 | 1.168 / 0.862 | 1.732 | 19.205 |
| room (1.9M) | 0.240 / 0.293 | 0.542 / 0.424 | 0.812 | 8.877 |

  Source: [arXiv HTML](https://arxiv.org/html/2605.06408)
- Density gradient, Table 6 (seconds, RTX 5090 / H200):

| Points | Ours | gDel3D | HXT (CPU) | Basselin\* |
|---|---|---|---|---|
| 1M | 0.118 / 0.132 | 0.283 / 0.226 | 0.307 | 0.387 / 0.561 |
| 10M | 1.102 / 1.084 | fails | 1.670 | 3.602 / 3.311 |

  - My arithmetic: about 9.1M cells/s at 10M points on the RTX 5090.
  - White noise, Table 7: gDel3D 0.281 s at 1M; HXT 0.303 s at 1M and 1.545 s at 10M. The "Ours" row was cut off in my fetch. **[gap]**
  - Source: [arXiv HTML](https://arxiv.org/html/2605.06408)
- Clustered, K=10, Table 4 (seconds, RTX 5090 / H200):

| Points | Ours | gDel3D |
|---|---|---|
| 1M | 0.386 / 0.720 | 0.284 / 0.229 |
| 2M | 0.675 / 1.027 | 3.120 / 1.196 |
| 5M | 1.224 / 1.732 | fails / 45.423 |

  - At 1M clustered points, gDel3D is faster than Ours.
  - Source: [arXiv HTML](https://arxiv.org/html/2605.06408)
- Power diagrams (RTX 5090 unless noted):
  - Table 9, clustered: 0.180 s at 0.1M and 1.913 s at 10M; 3.210 s at 15M on the H200, versus Geogram 10.965 s.
  - Table 12, white noise: 1.216 s at 10M.
  - Basselin\* fails on every clustered power-diagram case.
  - Source: [arXiv HTML](https://arxiv.org/html/2605.06408)
- Profile:
  - BVH build is 4.8% of the time and the fused kernel 95.2%.
  - Cell clipping is 96.2% of the fused kernel; traversal and culling are the rest.
  - Ablations: isotropic instead of directional culling is 1.26–1.40x slower (median). Depth-first instead of best-first traversal is 27.4x slower on clustered data and 37.2x on real data, but only 1.30x on Poisson data.
  - Source: [arXiv HTML](https://arxiv.org/html/2605.06408)
- Not reported: memory per cell; votess and Ray et al. are not benchmarked. — [arXiv HTML](https://arxiv.org/html/2605.06408)

**votess (Singh, Byrohl, Nelson 2024)** (arXiv 2412.04514v2, JOSS-style paper, SYCL)
- Algorithm:
  1. Sort points into a grid and run a k-NN search.
  2. Clip a bounding box by the bisectors of the neighbours.
  3. A "security radius" condition validates each cell; "If a cell cannot be validated, an automatic CPU fallback mechanism ensures robustness."
  - Output is cell geometry plus a jagged neighbour list, not a combinatorial mesh.
  - Source: [arXiv PDF](https://arxiv.org/pdf/2412.04514v2)
- Hardware in Fig. 1: two Intel Xeon Platinum 8360Y CPUs (36 cores each) and an NVIDIA A100-40GB. The distribution is uniform random with N = 1e3 to 1e6. — [arXiv PDF](https://arxiv.org/pdf/2412.04514v2)
- Fig. 1 gives no numbers in the text. Values read off the log-scale chart, so approximate **[my reading]**: at N = 1e6, votess-GPU is about 1.2e-6 s/point (≈0.8M cells/s), votess-CPU about 1.7e-6 s/point, Qhull about 2.5e-5 and voro++ about 5.5e-5 s/point.
- Text claims "The CPU implementation can outperform other implementations by a factor of 10 to 100". This is against single-threaded Qhull and voro++.
- Not stated: k, precision, fallback rate, and memory. — [arXiv PDF](https://arxiv.org/pdf/2412.04514v2)

**voro++ (Rycroft 2009, Chaos 19, 041111)** and the multithreaded extension (Lu, Lazar, Rycroft, CPC 2023, arXiv 2209.11606)
- Cell-based: each cell is built by plane cutting with a radius-based termination over a block grid. — [voro++ overview](https://math.lbl.gov/voro++/about.html)
- Serial throughput on a dual Xeon E5-2650L v4 (28 cores, 1.7 GHz), homogeneous case: 664,000 cells/s in 2D and 47,600 cells/s in 3D. — [arXiv 2209.11606](https://arxiv.org/html/2209.11606v2)
- Table 4, 3D:

| Case | 1 thread | 28 threads |
|---|---|---|
| Homogeneous, 10^8 particles | 2099.72 s | 93.34 s (≈1.07M cells/s, my arithmetic) |
| Localized, ~10^7 particles | 589.62 s | 26.85 s |
| Extreme clustering | 10368.70 s | 464.96 s |

  - Parallel efficiency is above 95% in many cases.
  - Source: [arXiv 2209.11606](https://arxiv.org/html/2209.11606v2)
- The extension refers to double precision. It gives no benchmark against other Voronoi codes. — [arXiv 2209.11606](https://arxiv.org/html/2209.11606v2)

### Inferences
- Peak 3D GPU throughput published, uniform-ish data, about 1e7 points:
  - Ray et al.: about 12.5M cells/s on a V100.
  - Basselin et al.: 10M points in 0.3 s on a V100 (restricted diagram).
  - Taveira et al.: about 9M cells/s end-to-end on an RTX 5090 or H200.
- Taveira's newer-GPU number is not higher than Ray's 2018 V100 number. Plausible reasons are end-to-end timing that includes the BVH build, harder distributions, and different measurement conventions. Taveira's own re-implementation of Basselin is about 3.3x slower than "Ours". None of these are like-for-like.
- Per-cell GPU clipping beats GPU Delaunay (gDel3D) on uniform and real data by about 1.3–2.4x. gDel3D can still win on small clustered sets.

### Gaps
- Not retrieved for Ray et al. 2018 (HAL/ACM blocked): the precision used, the exact k, the per-thread arrays, and the CPU-fallback rate.
- Not obtained for Basselin 2021: its internal tables.
- votess publishes only a chart.
- The original 2009 voro++ paper's timing numbers were not obtained.

## Q2. GPU Delaunay: gDel3D, gFlip3D, star splaying, newer GPU 3D Delaunay (2018–2026)

### Takeaway
gDel3D (2014) is still the standard GPU 3D Delaunay baseline in 2026. It is hybrid: the GPU does insertion and flipping, then the CPU runs a star-splaying repair, which makes it exact. It claims up to 10x over CGAL. Taveira 2026 reports it runs out of memory beyond a few million points on consumer GPUs. I found no peer-reviewed 2018–2026 GPU 3D Delaunay that clearly beats it on throughput.

### Cited Findings
- gDel3D (Cao, Nanjappa, Gao, Tan, I3D 2014, doi 10.1145/2556700.2556710):
  - The GPU does parallel insertion plus flipping, which gives a near-Delaunay result. The CPU then repairs it with "conservative star splaying".
  - Claims up to 10x over CGAL; hgpu/thesis summaries say up to 6x, so the sources disagree.
  - Sibling projects: gStar4D (fully GPU star splaying of lifted points in 4D) reports up to 5x over CGAL; gReg3D (regular triangulation) up to 4x.
  - Source: [NUS project page](https://www.comp.nus.edu.sg/~tants/gdel3d.html); [gDel3D PDF](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf); [hgpu](https://hgpu.org/?p=9609)
- gDel3D in 2026 benchmarks:
  - Double precision by default; single precision changes runtime by less than 0.5% (median).
  - Its "resource footprint limits scalability to a few million sites on consumer hardware". It fails at 10M density-gradient points on both the 5090 and the H200, and at 5M clustered points on the 5090.
  - Timings: 0.281–0.283 s for 1M points on an RTX 5090, and 0.226 s on an H200.
  - Source: [Taveira et al.](https://arxiv.org/html/2605.06408)
- HXT (Marot, Remacle et al., CPU, parallel Delaunay): described as "the current CPU state of the art". It tetrahedralized 3 billion points on one workstation in under a minute (Marot et al. 2019). It takes 1.545–1.670 s for 10M points on 16 EPYC cores. — [Taveira et al.](https://arxiv.org/html/2605.06408)
- Radiant Foam's GPU Delaunay ("RF Del", single precision) is very slow in these tables. Bicycle 4.2M points takes 124.9 s on the 5090. — [Taveira et al.](https://arxiv.org/html/2605.06408)
- An AsianCOMNET 2025 conference abstract on "gFlip3D"-style parallel insertion claims up to 24x over CPU and 4–5M points/s on more than 100M points. It is not peer-reviewed in the journal sense and has no gDel3D comparison. **[unverified, low confidence]** — [AsianCOMNET 2025](https://asiancomnet2025.aconf.org/presentation/212.html)
- GPU constrained Delaunay refinement in 3D (gQM3D) exists from the same NUS group. It covers refinement, not raw throughput. — [arXiv 1903.03406](https://arxiv.org/pdf/1903.03406)

### Inferences
- Throughput ranking for about 1M uniform points in 3D, from Taveira's tables:

| Method | Time |
|---|---|
| Per-cell clipping (Taveira) | ≈0.12 s |
| gDel3D | ≈0.23–0.28 s |
| HXT on 16 CPU cores | ≈0.3 s |
| Basselin\* re-implementation | ≈0.36–0.56 s |
| Geogram (CPU) | ≈0.8–1.7 s for 2–4M real points |

- GPU Delaunay codes are limited by memory more than by speed.

### Gaps
- gFlip3D (Gao et al. 2013) original numbers were not retrieved; the gDel3D PDF could not be parsed here (no pdftotext).
- No 2018–2026 peer-reviewed GPU 3D Delaunay with a verified throughput above gDel3D was found.

## Q3. Other local/meshless Voronoi and power-diagram work

### Takeaway
The Lévy/Ray/Sokolov line runs from Ray 2018 to Basselin 2021 (restricted power diagrams, GPU) and on to Geogram (CPU). It is the main GPU lineage, and votess and Taveira build on it. On the CPU, voro++, its multithreaded extension, ParVoro++ and the Rust meshless_voronoi crate are the main cell-based codes. Only meshless_voronoi documents exact-arithmetic fallback.

### Cited Findings
- Geogram (Lévy, Inria) is the fastest CPU baseline in Taveira's tables:
  - Real-scene Voronoi: 1.593 s for 4.2M points.
  - Power diagram, 15M clustered points: 10.965 s.
  - Source: [Taveira et al.](https://arxiv.org/html/2605.06408)
- ParVoro++ (Wu, Tian, Lu, Wang 2023, Parallel Computing 115, 102995): a kd-tree-decomposed parallel voro++. Its numbers were not retrieved. — cited in [votess](https://arxiv.org/pdf/2412.04514v2)
- meshless_voronoi (Rust, Yuyttenhove):
  - Implements Ray et al. on the CPU (rayon parallel), with rectangular or periodic boundaries.
  - "arbitrary precision arithmetic is used to treat degeneracies" so that local geometry stays globally consistent.
  - On a 64³ grid it takes 2.062 s for a perfect (degenerate) grid and about 1.30 s for a perturbed grid, with the rug backend. Hardware is not stated.
  - Source: [GitHub](https://github.com/yuyttenhove/meshless_voro)
- Other users of the same machinery: restricted power diagrams for medial axes (MATTopo), and higher-dimensional power diagrams for semi-discrete optimal transport. — [MATTopo arXiv 2403.18761](https://arxiv.org/pdf/2403.18761); [arXiv 2106.14730](https://arxiv.org/pdf/2106.14730)

### Gaps
- GPU CVT/Lloyd papers, PARAVT, and CGAL-parallel absolute numbers were not researched within the budget.

## Q4. Techniques reported to matter for GPU speed

### Takeaway
The best-documented ablations are in Taveira 2026:
- **Neighbour search** must adapt to the point distribution. Best-first BVH traversal with directional culling beats kNN with a growing radius on non-uniform data.
- **Clipping dominates the time**, at about 96% of the kernel.
- **Overflow handling**: compacting the plane array in place beats relaunching with larger arrays.
- **Precision**: float suffices for adjacency at a 0.1–0.2% mismatch level.

### Cited Findings
- Clipping takes 96.2% of the fused kernel; BVH build is 4.8% of total time. — [Taveira et al.](https://arxiv.org/html/2605.06408)
- Best-first versus depth-first traversal matters 27–37x on clustered and real data, but only 1.3x on Poisson data. Directional versus isotropic culling is worth 1.26–1.40x. — [Taveira et al.](https://arxiv.org/html/2605.06408)
- Memory layout:
  - Sites are float4-aligned for coalesced loads.
  - Each vertex is three uchar plane indices (compact combinatorial vertices).
  - Vertex positions are cached in global memory rather than shared memory.
  - Basselin et al. instead keep per-thread data in shared memory.
  - Source: [Taveira et al.](https://arxiv.org/html/2605.06408)
- Plane "garbage collection": fully clipped planes are compacted at 85% occupancy, which replaces Basselin's relaunch-from-scratch with larger arrays. — [Taveira et al.](https://arxiv.org/html/2605.06408)
- The kNN warm start (K=8) helps on the RTX 5090 but is disabled on the H200, so it is hardware-dependent. — [Taveira et al.](https://arxiv.org/html/2605.06408)
- Thread-per-cell is used in Taveira, and implicitly in Ray, votess and Basselin. No published warp-per-cell 3D Voronoi ablation was found. — [Taveira et al.](https://arxiv.org/html/2605.06408)
- gDel3D's runtime barely changes between double and single precision (median under 0.5%). — [Taveira et al.](https://arxiv.org/html/2605.06408)

### Gaps
- No published float-versus-double ablation for a per-cell clipping kernel was found.
- No warp-cooperative versus thread-per-cell comparison was found.
- No memory-per-cell numbers in bytes are reported by any of these GPU works.

## Q5. Consistency between independently computed neighbouring cells

### Takeaway
Floating-point per-cell methods accept small topological inconsistencies:
- Ray 2018 accepts zero-area facets.
- Taveira 2026 measures a 0.1–0.2% adjacency mismatch against CGAL exact predicates.

Only CPU codes (meshless_voronoi with arbitrary precision; gDel3D's CPU star-splaying repair, which makes it a true Delaunay) guarantee consistency.

### Cited Findings
- Ray et al.: the meshless approach "will just produce sometimes a facet with zero area" instead of using consistent exact predicates. **[snippet]** — [ACM DL](https://dl.acm.org/doi/abs/10.1145/3272127.3275092)
- Taveira et al. (single precision):
  - 0.1–0.2% of adjacencies mismatch CGAL on real data, attributed to near-coincident candidate planes.
  - Volume rendering over 100k rays gives a mean accumulated-density difference below 1.7e-5, with a median of 0.
  - Source: [Taveira et al.](https://arxiv.org/html/2605.06408)
- votess: cells that fail the security-radius check are recomputed on the CPU. No rate is reported. — [votess](https://arxiv.org/pdf/2412.04514v2)
- meshless_voronoi: arbitrary-precision fallback for degeneracies keeps global consistency. Degenerate inputs (a perfect grid) cost about 1.6x the runtime of perturbed ones (2.06 s vs 1.31 s). — [GitHub](https://github.com/yuyttenhove/meshless_voro)
- gDel3D: the CPU star-splaying repair produces an exact Delaunay result. — [NUS](https://www.comp.nus.edu.sg/~tants/gdel3d.html)

### Inferences
- A security radius certifies that each cell is complete given its own arithmetic. It does not certify that two neighbours agree on a shared face when evaluated in floating point. Symmetric adjacency needs either exact or filtered predicates, or a post-pass that symmetrizes the faces.
- None of the GPU papers found report a symmetrization pass.

### Gaps
- No published GPU fallback rate (fraction of cells failing security-radius certification) was found for Ray, votess or Basselin.
- No GPU paper reporting exact or filtered predicates inside a per-cell clipping kernel was found.
