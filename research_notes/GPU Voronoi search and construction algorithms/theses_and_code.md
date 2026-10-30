# GPU Voronoi/Delaunay construction: theses, technical reports and open-source code (to Oct 2026)

Scope: details that the papers leave out, taken from theses and source code: (a) how each method finds neighbours for a per-cell Voronoi cell, (b) the cell/star construction step, (c) measured throughput. The ProteusGPU code was excluded as the user asked, and was not read.
Method: full-text reading of the Nanjappa PhD thesis and the Paragram paper (arXiv 2605.06408 v1, including its appendix and supplementary tables). Source code was read from shallow clones of paragram, peclet-voro, votess, GPU-Restricted-Power-Diagrams (Basselin), 3D-Voronoi-GPU (Liu et al. 2020) and Local-DeWall (Gao & Chen 2026).
Derived throughputs (cells/s = N / time) are my arithmetic on the quoted times and are labelled "derived".

---

## Q1. Theses and habilitations: what do they say about neighbour discovery and star/cell construction?

### Takeaway
Ashwin Nanjappa's NUS PhD thesis (2012) is the main "hidden" source. Its gStar4D chapter is a full GPU method that builds Delaunay stars from an approximate neighbour set: it lifts each point to R⁴ and runs beneath-beyond insertion into a per-point star, without clipping. It used about 15.8 initial candidates per point plus about 2.2 repair insertions per point on uniform input. Even so it was slower than the flip-based gFlip3D/gDel3D, and on real data sometimes slower than single-core CGAL, because a fixed digital-Voronoi grid gives poor working sets for non-uniform input and the 4D exact/SoS predicates are expensive. I could not obtain the Basselin, Gao and Cao theses, the Ray HDR, the Sokolov habilitation, or any GPU moving-mesh astrophysics thesis (see Gaps).

### Cited findings

**Nanjappa, "Delaunay triangulation in R3 on the GPU", PhD, NUS 2012 (advisor Tan Tiow Seng)** — [thesis PDF](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf)
- Hardware for every experiment: Intel i7 2600K, 16 GB DDR3, NVIDIA GTX 580 (Fermi, 3 GB), CUDA 4.2, Windows 7, CGAL 3.8 as baseline (Sec. 4.5.1). — [thesis](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf)
- **gFlip3D** (parallel insertion plus flipping, near-Delaunay only):
  - Predicates are arithmetic-filtered Shewchuk predicates split into a "fast kernel" and an "exact kernel". Items that need exact evaluation are marked and handled by a second kernel launched with fewer threads, because "the exact variant of the insphere predicate needs local storage of up to 14880 floating point numbers".
  - SoS perturbation handles degeneracies.
  - Speedup over CGAL: up to 6.5× for synthetic distributions and 3× for the grid distribution.
  - Real models, CGAL s / gFlip3D s: Armadillo 172,974 pts 2.01/0.87; Brain 294,012 3.45/0.90; Dragon 437,645 5.35/1.75; Happy Buddha 543,652 6.65/2.19; Blade 882,954 10.47/5.44 (Table 4.3).
  - — [thesis](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf)
- **gStar4D (Ch. 6): star splaying in R⁴. This is the main documented GPU method that gets natural neighbours without clipping a candidate set.** The pipeline:
  1. Snap points to an N³ voxel grid (512³ in the experiments). If several points fall in one voxel, only one keeps it; the rest become "missing points".
  2. Compute a digital Voronoi diagram with the Parallel Banding Algorithm (PBA).
  3. Dualise it under a "perturbed grid" interpretation, so that only four colours can meet at a grid vertex; each voxel thread handles 6 of its 24 vertices. Colour pairs give each point a **working set** (its approximate Voronoi neighbours).
  4. Missing points get their "leader" point, and the leader gets the missing points. Working sets smaller than a minimum are "pumped" from neighbours' sets. All insertions are **reciprocated** (if p is in s's set, s goes into p's set).
  5. For each point, build an initial star from 4 working-set points (a 4-simplex), then insert the remaining points by **beneath-beyond** in the lifted space (w = x²+y²+z²).
  6. Iterate consistency repair: find "drowned" insertions, compute confinement-proof points, insert them, until no insertions remain. Extract tetrahedra from the lower hull.
  - Data structure: per star, a link triangulation of `Tri{int _v[3]}` + `TriOpp{int _t[3]}` stored in per-star contiguous **chunks** of one global array. Parallelism is one thread per star, or per star tetrahedron, with no atomics or locks.
  - — [thesis](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf)
- gStar4D insertion counts at 10⁶ points, grid 512³ (Table 6.3), given as star creation / splaying / rounds:
  - Uniform: 15,789,280 / 2,228,302 / 12 rounds.
  - Gaussian: 15,786,564 / 7,410,146 / 7.
  - Ball: 15,973,062 / 2,941,722 / 10.
  - Sphere: 15,775,530 / 5,202,922 / 10.
  - Grid: 15,790,644 / 1,520,024 / 14.
  - The thesis comments that the "average working set size of 15.8 is close to the 15.535 expected number of faces of a typical Voronoi cell". Per uniform star, about 15.7 points are inserted at creation and 2.2 during splaying. Splaying still costs more time than creation, "because the cost of a point insertion is proportional to the number of link triangles in the star".
  - — [thesis](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf)
- gStar4D performance:
  - Up to 5× over CGAL for uniform and ball, up to 4× for sphere and Gaussian. The grid distribution "has no speedup, taking double the time of CGAL for 10⁶ points"; its splaying stage alone takes 24.4 s because of exact and SoS 4D orientation tests.
  - Real inputs, gStar4D s (CGAL s): Armadillo 1.74 (2.01), Brain 2.42 (3.45), Dragon 5.98 (5.35), Happy Buddha 9.12 (6.65). Blade could not run because the hardware "cannot handle the number of exact and SoS checks".
  - With a 256³ grid, "more than 90% of the points are missing points" for real inputs.
  - The author's own verdict: star splaying "is enormously wasteful since it constructs the stars for all points and every tetrahedron in the triangulation needs to be constructed in its four stars."
  - — [thesis](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf)
- **gDel3D (Ch. 7)**: gFlip3D followed by a CPU repair using conservative star splaying, with stars built on demand only for failed points.
  - 10⁶ uniform points in 2 s (GTX 580); about 6× over CGAL, and about 50% faster than gStar4D.
  - Real inputs, gDel3D s with repair s in brackets: Armadillo 1.14 (0.17), Brain 1.16 (0.18), Dragon 2.55 (0.77), Happy Buddha 2.72 (0.34), Blade 6.92 (1.04).
  - — [thesis](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf); [gDel3D project page](https://www.comp.nus.edu.sg/~tants/gdel3d.html)
- Earlier NUS GPU-hull work (GHull, as described in the thesis): working sets come from dualising a digital Voronoi diagram ("for every triangle abc dualized … b and c are added to the working set of a"), followed by star splaying in R³. Reported as 10× over QHull on an i7 2600K plus GTX 580. — [thesis](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf)

**Justine Basselin (Inria Nancy / LORIA): GPU restricted power diagrams**
- Published as a Computer Graphics Forum 2021 paper (hal-03169575), with Alonso, Ray, Sokolov, Lefebvre and Lévy. The abstract claims "×50" versus CPU implementations, and the search snippet reports 10 M points restricted to a mesh in 300 ms on a V100 (not verified from the full text). — [HAL](https://hal.archives-ouvertes.fr/hal-03169575)
- Her slides note "GTX1080: can't reach 10M points for white noise", which is the fixed-k/fixed-capacity limit. — slides PDF (local copy only; not re-verified online)

### Inferences
- gStar4D is the closest existing GPU analogue to "build each star from a small candidate set, then fix inconsistencies". Its weak points were (i) the quality of the candidate set (a digital Voronoi diagram on a fixed grid) and (ii) 4D exact/SoS predicates on Fermi.
- A modern version would use kNN or grid candidates instead of PBA, and float filters with rare exact fallbacks. That might do better, but the 2012 evidence is that per-star beneath-beyond plus a global consistency iteration loses to direct flip-based methods.
- Per point, the work (15.8 insertions into a star, each costing O(link size)) is comparable to clipping about 16–30 planes into a convex cell. So the lifted-star route does not reduce work per cell. It only replaces "clip a half-space" with "insert a lifted point".

### Gaps
- **Not found or read:** Mingcen Gao's NUS thesis (gFlip3D/flip-flop hulls), Thanh-Tung Cao's thesis, Basselin's PhD thesis (no HAL "tel-" record turned up), Nicolas Ray's HDR, Dmitry Sokolov's habilitation, and Lévy's semi-discrete OT notes on GPU. HAL PDFs were blocked by bot protection (Anubis) during this session.
- I found no thesis on GPU Voronoi moving-mesh hydro for astrophysics (an Arepo/Shadowfax-like code on GPU) in the time available.

---

## Q2. Open-source code: neighbour discovery, per-cell data structure, neighbours/clips per cell, measured cells/s

### Takeaway
Every production GPU per-cell Voronoi code I found does incremental half-space clipping into a small fixed-capacity convex cell. The cell is stored as planes plus vertex triples of plane indices, at 32–64 planes. The codes differ mainly in neighbour discovery:
- fixed-k kNN plus a security-radius check (Ray 2018, Liu 2020, Basselin 2021, votess);
- a grid "worklist" walked closest-first (voro++-style; peclet-voro);
- a best-first BVH traversal with directional culling (Paragram, SIGGRAPH 2026).

Paragram is the fastest published: about 9.5 M cells/s (derived) in FP32 on an H200 at 15 M uniform points, about 2× gDel3D. It reports 96.2% of fused-kernel time in clipping. Its gains on non-uniform data come from the neighbour search (best-first versus DFS: up to 80× on clustered data).

### Cited findings

**Paragram — Taveira, Lindström, Fatemi, Hammarstrand, Kahl, "Scalable GPU Construction of 3D Voronoi and Power Diagrams", SIGGRAPH 2026 / ACM TOG** — [arXiv 2605.06408](https://arxiv.org/pdf/2605.06408); code linked from [research.zenseact.com/publications/paragram](https://research.zenseact.com/publications/paragram)

Algorithm:
- One thread per site, with the clipping procedure taken from Basselin 2021.
- Neighbour discovery is a **best-first traversal of a binary BVH** built with cuBQL. Leaf size is 17 on H200 and 10 on RTX 5090. Each node also stores max weight, for power diagrams.
- Each node is pruned with a **directional cell radius**: the maximum distance from the site to the corners of the current cell's AABB, taken only in the octants the node occupies. This is compared with a lower bound on the bisector distance.
- The traversal uses a per-thread stack that is unsorted on push; pop does a linear scan for the nearest entry. The stack is `MAX_STACK_K 128` in `pwr_bvh.cu`.
- Nodes are ordered by a signed "penetration depth" into the cell AABB, not by distance to the seed.
- An optional **kNN warm start with K = 8** pre-clips the cell. It is enabled on the RTX 5090 (large L2) and disabled on the H200 because "the overhead of the additional pass is not recovered".
- The cell is initialised as the AABB of the whole point set.
- — [arXiv 2605.06408](https://arxiv.org/pdf/2605.06408)

Per-cell memory:
- Per-thread cell storage lives in **coalesced global memory, not shared memory**: SoA with `CELL_BLOCK_STRIDE 32`, `MAX_PLANES 64`, `MAX_VERTS 64`.
- Per thread: `uchar4` vertex (3 plane indices), cached `float4` vertex position, `uchar` boundary-next, `float4` plane, `int` neighbour id. That totals about 64·(4+16) + 64·(1+16+4) = 2,624 B/cell of scratch (derived from `required_bytes` in `faster_convex_cell.cuh`).
- The paper explains the choice: "we found the resulting per-thread footprint [in shared memory] to severely limit … occupancy". Vertex positions are cached rather than recomputed.
- The plane list is garbage-collected when it is more than 85% full, instead of Basselin's overflow-and-relaunch.
- — [arXiv 2605.06408](https://arxiv.org/pdf/2605.06408); local code read of `paragram/csrc/laguerre/faster_convex_cell.cuh`

Clip step:
- Classify vertices against the new plane.
- Remove the outside vertices one at a time, keeping the hole boundary as a circular list of plane-index pairs.
- Add one new vertex triple (h, a, b) per boundary edge, then update the cell AABB and the radii.
- — [arXiv 2605.06408](https://arxiv.org/pdf/2605.06408)

Measured results (end-to-end; FP32; 3 warm-up and 10 timed runs):
- White noise (Table 7), RTX 5090 at 0.1/0.5/1/2/5/10 M: 0.020/0.060/0.112/0.214/0.530/1.078 s. 15 M does not fit.
- White noise, H200 at 0.1/…/15 M: 0.017/0.063/0.123/0.227/0.537/1.064/1.582 s. → about **9.5 M cells/s at 15 M, H200** (derived) and about 9.3 M cells/s at 10 M on the 5090 (derived).
- gDel3D (double) on the 5090 at 1/5 M: 0.281/1.471 s, out of memory at ≥10 M. On H200: 0.225/1.086 s.
- Their own Basselin re-implementation (FP32), H200, 15 M: 4.619 s.
- CPU baselines on 16 Epyc 9534 cores, 15 M: Geogram 6.138 s; HXT 2.259 s; voro++ 41.136 s; CGAL 71.999 s.
- — [arXiv 2605.06408](https://arxiv.org/pdf/2605.06408)
- Real Radiant Foam point sets (Table 1), 2.0–4.2 M points, "Ours" on the 5090: 0.235–0.570 s. gDel3D: 0.542–1.204 s. HXT: 0.675–1.375 s. — [arXiv 2605.06408](https://arxiv.org/pdf/2605.06408)
- Clustered data (10 clusters, σ = 0.1, Table 4): gDel3D is *faster* at small N. On the H200 at 1 M, gDel3D takes 0.229 s against Paragram's 0.720 s. Paragram wins at ≥5 M because gDel3D runs out of memory or takes 45.4 s. — [arXiv 2605.06408](https://arxiv.org/pdf/2605.06408)

Time breakdown:
- BVH build is 4.8%. Of the fused kernel, **96.2% is cell clipping** and the remainder is traversal and culling. — [arXiv 2605.06408](https://arxiv.org/pdf/2605.06408)

Ablations (RTX 5090, median slowdown):
- Isotropic culling: 1.26× (clustered), 1.33× (Poisson), 1.40× (real).
- DFS instead of best-first: 27.4× (3.96–80.3), 1.30× (1.13–22.8), 37.2× (34.8–53.1).
- — [arXiv 2605.06408](https://arxiv.org/pdf/2605.06408)

Accuracy:
- Single precision. "0.1–0.2%" of adjacencies differ from CGAL exact on real data, "traced to near-coincident candidate planes". The authors note the method "does not target the exact geometric guarantees".
- Measured gDel3D single versus double precision: "median < 0.5%" difference.
- Radiant Foam's own Delaunay rebuild ("RF Del", FP32) is very slow at scale: 124.9 s on bicycle 4.2 M on the 5090.
- — [arXiv 2605.06408](https://arxiv.org/pdf/2605.06408)

**Basselin et al. 2021, GPU-Restricted-Power-Diagrams (OpenCL)** — [HAL](https://hal.archives-ouvertes.fr/hal-03169575); code github.com/basselin7u/GPU-Restricted-Power-Diagrams (local read)
- Neighbours come from a fixed-K kNN on a grid (`knearests.cl`). The cell lives in shared/local memory with capacities P (planes) and V (vertices). Precision is float or double at compile time (`USE_DOUBLE`).
- README-recommended parameters for 100 K points: blue noise `--K=20 --P=25 --V=30`; perturbed grid `--K=50 --P=30 --V=40`; white noise `--K=70 --P=30 --V=40`.
- On overflow it relaunches with larger arrays. Paragram reports that this "overflow-and-retry" and the fixed kNN "cannot produce valid results" on clustered or real data.
- — code README (local); [arXiv 2605.06408](https://arxiv.org/pdf/2605.06408)

**Ray, Sokolov, Lefebvre, Lévy, "Meshless Voronoi on the GPU", TOG 2018** and **Liu, Ma, Guo, Yan, "Parallel Computation of 3D Clipped Voronoi Diagrams", TVCG 2020** (code 3D-Voronoi-GPU, CUDA)
- Each cell is computed independently from its k nearest neighbours, with a radius-of-security check. The paper claims "one order of magnitude" over the fastest multi-core CPU. — [HAL](https://hal.archives-ouvertes.fr/hal-01927559)
- Liu 2020 `params.h` presets:
  - White noise: `_K_ 90`, `_MAX_P_ 64`, `_MAX_T_ 96`, `VORO_BLOCK_SIZE 16`.
  - Perturbed grid: K 90, MAX_P 50.
  - Blue noise: K 35, MAX_P 32.
  - Status code `security_radius_not_reached`; arithmetic filter optional (`USE_ARITHMETIC_FILTER`, off by default).
  - — local code read (github repo of Liu et al.; README links [IEEE 9151267](https://ieeexplore.ieee.org/document/9151267/))
- No timings could be verified (the HAL PDF was bot-blocked). **Unverified.**

**votess (Samridh Dev, SYCL; JOSS / arXiv 2412.04514)** — [arXiv](https://arxiv.org/html/2412.04514v2)
- Points are sorted into a grid, followed by a k-NN search, then clipping from a bounding box. Cells that fail the security-radius check go to a CPU fallback. — [arXiv](https://arxiv.org/html/2412.04514v2)
- Defaults (`arguments.hpp`): grid resolution 16, `cc_p_maxsize 32`, `cc_t_maxsize 32`, chunk size 8196. — local code read
- The JOSS paper concedes that compared with CGAL "both the CPU and GPU version falls short by around a factor of 6". No absolute GPU numbers are given in the text. — votess `joss/paper.md` (local read); [arXiv](https://arxiv.org/html/2412.04514v2)

**peclet-voro (computational-chemical-engineering/peclet-voro; Kokkos CUDA/HIP/OpenMP)** — repo [github.com/computational-chemical-engineering/peclet-voro](https://github.com/computational-chemical-engineering/peclet-voro), `docs/performance_report.md`
- Neighbour discovery: a **voro++-derived presorted worklist** that walks neighbouring grid cells closest-first and stops at the security radius. The search window defaults to `kSearchWindow = 4`.
- Cell: a compact **dual-triangle `ConvexCell`** (vertex = triple of plane indices) with `MAXP = 64` planes and `MAXT = 112` triangles. It is clipped "GEOGRAM/Ray-style". Geometry comes from a per-vertex divergence-theorem scatter that needs no sorting and no adjacency.
- Measured (FP64, uniform Poisson, periodic, NVIDIA RTX 5080, CUDA 13.2), cold build: 2.99 / 3.25 / 3.54 / **3.61 M cells/s** at N = 100 k / 200 k / 1 M / 2 M. OpenMP on 8 cores: 0.33 M cells/s. Serial: 0.059 M cells/s.
- Construct from cached neighbours (no gather) runs at **8.5 M cells/s FP64**. The report states "the neighbour gather costs roughly as much as the entire construct + geometry". It also self-reports an FP32 construct of about 14–17 M cells/s that "beat the Liu-2020 GPU code on the same hardware" (a prior internal measurement; **unverified**).
- Resident store for moving-mesh repair: 1,260 B/cell. Repair uses a Lawson-style local convexity certificate ("poke4" planes per dual vertex) and regathers only the flagged cells.
- — [peclet-voro repo](https://github.com/computational-chemical-engineering/peclet-voro) (local read of docs/performance_report.md and include/peclet/voro/params.hpp)

**meshless_voronoi (Rust, Y. Uyttenhove)** — [docs.rs](https://docs.rs/meshless_voronoi)
- CPU (rayon) implementation of Ray 2018. Cells are intersections of `HalfSpace`s. It depends on `rstar` (presumably used for neighbour queries; not documented). Degeneracies are handled with arbitrary-precision integer backends.
- 64³ seeds take 2.062 s (perfect grid, rug backend) and 1.308 s (perturbed grid). Hardware is not stated.
- — [docs.rs](https://docs.rs/meshless_voronoi)

**Radiant Foam (theialab/radfoam)**
- Custom CUDA Delaunay. Adjacency is updated "by performing an incremental Delaunay triangulation", and rebuilds are done every 1 to 100 optimiser steps. — [arXiv 2502.01157](https://arxiv.org/pdf/2502.01157)
- Paragram measured it as by far the slowest GPU code on large sets. — [arXiv 2605.06408](https://arxiv.org/pdf/2605.06408)

**Geogram / HXT (CPU reference points)**
- On 16 cores of an Epyc 9534, Geogram (PeriodicDelaunay3d/Delaunay_psm lineage) does 15 M uniform points in 6.138 s (≈2.4 M/s, derived). HXT does it in 2.259 s (≈6.6 M/s, derived). — [arXiv 2605.06408](https://arxiv.org/pdf/2605.06408)

### Inferences
- Across all codes, the clip kernel looks the same (planes plus vertex-as-plane-triple at 32–64 capacity). The distinguishing lever is **candidate ordering and culling**, which is what reduces the number of clips per cell.
- Paragram's 96.2%-clipping breakdown, together with best-first ordering, suggests near-optimal candidate order leaves clipping as the floor. peclet's data (gather ≈ construct) shows that a grid worklist without tight culling is a large overhead. Those two figures are not directly comparable: FP32 versus FP64, a different GPU, and a non-periodic AABB start versus periodic.
- Paragram stores scratch in global memory, not shared memory or registers. That choice conflicts with register-resident designs (peclet). Which wins likely depends on GPU (L2 size) and capacity (64 planes).

### Gaps
- None of the codes report the actual **mean number of clips per cell** or the number of rejected candidates. Paragram gives only percentages; Ray, Basselin and votess give K only.
- No verified absolute timings for Ray 2018, Liu 2020 or votess GPU runs.
- I did not inspect Geogram's GPU code path (if any) or CGAL parallel internals.

---

## Q3. Methods that get natural neighbours without clipping a full candidate set — and their measured efficiency

### Takeaway
Three families exist:
1. **Lifted-star construction from an approximate neighbour set**: gStar4D (2012). Exact, but slower than flip-based methods.
2. **Per-vertex Delaunay "walking" / gift-wrapping with empty-sphere queries on a grid**: Gao & Chen's "Parallel 3D Delaunay Triangulation on the GPU" (CAD 2025) and "Local DeWall" (CAD 2026). They claim 2× speedup and 1.2 billion points on one consumer GPU, but I could not access the full-text numbers.
3. **Global insert-and-flip**: gDel3D.

Only Local DeWall is a credible, recent "no-clipping" per-point alternative. It is unverified against Paragram-class clipping on the same hardware.

### Cited findings
- **Local DeWall (Wuheng Gao, Renjie Chen; Computer-Aided Design vol. 199, Oct 2026)**:
  - Abstract: "the complete one-ring is not strictly necessary". Each vertex stores a "Local Delaunay Wall", a striped subset of its one-ring, as an ordered point list. This cuts per-vertex memory "by over 2/3".
  - Method: an "incremental traversal [that] walks across adjacent tetrahedra using only point-based operations, replacing expensive cell clipping with local walking".
  - Claims: 2× speedup (baseline not given in the snippet) and scaling to 1.2 billion points on a single consumer GPU.
  - — [ScienceDirect abstract](https://www.sciencedirect.com/science/article/abs/pii/S0010448526000771) (full text 403; via search snippet)
- Local DeWall code (MIT; [github.com/WuhengGao/Local-DeWall](https://github.com/WuhengGao/Local-DeWall); local read of `include/delaunay_kernels.h`, `src/delaunay_kernels.cu`):
  - **Two lanes per vertex** (paired via `__shfl_xor_sync(…,1)`), each holding up to `MAX_P = 32` wall vertices.
  - The main phase runs in `float3` with Shewchuk-style filtered predicates and SoS (`InSphereSoSOnly`).
  - Each step takes the active triangle (seed, a, b) and a tentative fourth point c. It computes the circumsphere, scans every uniform-grid cell overlapping the sphere's AABB (`cell_range` + `ballsearch`), and looks for any point inside the sphere.
  - If a point is found, it replaces c. If not, c is accepted as the next wall vertex and the walk advances. This is a gift-wrapping step: no half-space clipping, just repeated empty-sphere queries.
  - The walk is capped at `MAX_CYCLES = 100`.
  - Failures (`need_exact_predict`, overflow) go to a **post phase** where a whole block cooperates on one vertex (`BLOCK_SIZE_POST`, warp reductions, `MAX_POST_P = 128`, `MAX_POST_CYCLES2 = 500`), using `REAL` (double optional via `USINGDOUBLE`).
  - The grid is built by counting sort (`countpoints`/`storepoints`). Output is int32 tetrahedra.
  - — [Local-DeWall repo](https://github.com/WuhengGao/Local-DeWall)
- **Gao & Chen, "Parallel 3D Delaunay Triangulation on the GPU" (CAD 2025)**: extends Chen–Gotsman's Local Delaunay Lemma to R³. Delaunay neighbours of a point are confined to the union of circumspheres of (point, consecutive ring neighbours), so each point is processed independently by half-space intersection and the output is a list of tetrahedra. No numbers accessible. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010448525000946); 2D lemma origin: [Chen & Gotsman, "Localizing the Delaunay Triangulation and its Parallel Implementation"](http://staff.ustc.edu.cn/~renjiec/papers/pdt.pdf)
- **gStar4D**: about 15.8 working-set insertions plus 2.2–7.4 splaying insertions per point. On real models it ran at 0.73–1.43× CGAL single-core on a GTX 580 (see Q1). — [Nanjappa thesis](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf)
- An independent benchmark harness (omarbesbes/Delaunay; A100-SXM4 40 GB, CUDA 12.8) compares CGAL, gDel3D, gStar4D, Local DeWall, Geogram and Paragram. The authors state that "Detailed experimental results … are intentionally not included". It lists Local DeWall as float32 with exact predicates. — [omarbesbes/Delaunay](https://github.com/omarbesbes/Delaunay)

### Inferences
- Local DeWall's inner loop is "find any point inside a circumsphere via a grid AABB scan". Each wall step costs a grid range scan of O(points in the sphere's AABB) insphere tests. A clipping step instead costs O(cell vertices) plane tests per candidate.
- For uniform input with roughly 15.5 neighbours and about 27 Voronoi vertices per cell, the walk needs at least about 27 successful sphere queries per vertex, each scanning several grid cells (each vertex's tetrahedra are shared across 4 vertices). Whether this beats about 30–40 clips per cell is an empirical question. The authors claim 2×, but the baseline is unknown (probably gDel3D).
- The method is exact (filtered predicates plus SoS and a double post phase). That is an advantage over FP32 Paragram, which has 0.1–0.2% adjacency mismatches.
- For the user's goal (exact 3D Voronoi on GPU, per-cell), the most promising non-clipping idea to test is Local DeWall-style **circumsphere-emptiness walking around each site** (dual of walking the Voronoi vertices of the cell), with the candidate search done by grid range queries. A hybrid seems natural: a kNN warm start, then sphere-emptiness certification per Voronoi vertex, which is also the natural exactness certificate.
- No source shows a lifted convex hull of a small neighbourhood beating clipping on a GPU. gStar4D is the negative data point.

### Gaps
- I could not access the full text or numbers of Local DeWall or Gao & Chen 2025: Elsevier returned 403. The 2× speedup and the 1.2 B-point claim come from the abstract only. The hardware is only implied by the repo's `sm_86` target (RTX 30-series).
- No direct same-hardware comparison of Local DeWall against Paragram or per-cell clipping was found (the omarbesbes harness withholds its results).
- I found no measured GPU "edge-based" or "Delaunay walk from kNN" Voronoi method other than those above.
