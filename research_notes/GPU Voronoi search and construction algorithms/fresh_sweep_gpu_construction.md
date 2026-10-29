# Fastest GPU 3D Voronoi / Delaunay construction methods (2015 – Oct 2026), independent sweep

Scope note: these are notes from an independent sweep (web search, arXiv full texts where text extraction worked). ScienceDirect (Gao & Chen 2025/2026), HAL (Ray et al. 2018) and the Semantic Scholar API refused access (403, Anubis bot wall, 429), so those papers are covered only from abstracts, highlights and search snippets. ProteusGPU was deliberately not used. Throughput figures labelled "derived" are my own arithmetic (N / reported seconds) and are not quoted from the papers.

## Which methods report the highest 3D throughput, and under what conditions?

### Takeaway
The highest 3D throughput with complete, end-to-end tables is Taveira et al. (SIGGRAPH 2026). It clips one cell per thread with a fused BVH best-first search, in FP32. Derived throughput is about 9–10 M cells/s for 1–15 M Poisson points on an RTX 5090 or H200, and about 7–9 M cells/s on real Radiant-Foam point sets. On real and Poisson data it beats gDel3D, which is double precision and still the reference GPU Delaunay code, by about 1.3–2.7×. gDel3D stays faster on small clustered sets but runs out of memory at about 5–10 M points. Gao & Chen's Local DeWall (CAD 2026) claims the largest scale: 1.2 B points (8.4 B tets) on one consumer GPU and "2× speedup". I could not get its full text, so its GPU model, precision and absolute throughput are not verified.

### Cited Findings

**A. Taveira, Lindström, Fatemi, Hammarstrand, Kahl, "Scalable GPU Construction of 3D Voronoi and Power Diagrams", SIGGRAPH Conference Papers '26 (DOI 10.1145/3799902.3811229), arXiv:2605.06408**
- Algorithm family: per-cell convex clipping by bisector half-spaces, following Basselin et al. 2021. A directional culling criterion and best-first BVH traversal find the candidates. It also handles power diagrams. — [arXiv:2605.06408](https://arxiv.org/abs/2605.06408); [Chalmers record](https://research.chalmers.se/en/publication/553368)
- Code: the authors say it is released at research.zenseact.com/publications/paragram. I did not check the repository. — [Chalmers record](https://research.chalmers.se/en/publication/553368)
- Hardware and precision: NVIDIA RTX 5090 and NVIDIA H200, "We run our method using single precision". CPU baselines ran on 16 cores of an AMD Epyc 9534 in double precision. gDel3D used its default double-precision setup; single precision changed its time by a median of <0.5%. Radiant Foam's Delaunay ("RF Del") and their own Basselin port are single precision. — [arXiv:2605.06408 PDF](https://arxiv.org/pdf/2605.06408)
- Timing protocol: 3 warm-up runs, then 10 timed runs averaged, 300 s timeout. "All reported timing results are end-to-end; no preprocessing or preallocation is excluded." — [arXiv PDF](https://arxiv.org/pdf/2605.06408)
- White noise (Table 7), seconds for 0.1M/0.5M/1M/2M/5M/10M/15M points:
  - Ours on RTX 5090: 0.020/0.060/0.112/0.214/0.530/1.078/– (15M did not fit).
  - Ours on H200: 0.017/0.063/0.123/0.227/0.537/1.064/1.582.
  - gDel3D on 5090: 0.027/0.149/0.281/0.581/1.471/–/–.
  - gDel3D on H200: 0.036/0.128/0.225/0.442/1.086/–/–.
  - Basselin* on 5090: 0.032/0.186/0.355/0.676/1.755/3.608/5.598.
  - HXT (CPU, 16 cores): 0.097/0.200/0.303/0.478/0.909/1.545/2.259.
  - Geogram (CPU): 0.060/0.267/0.420/0.791/1.920/4.202/6.138.
  - Source: [arXiv PDF](https://arxiv.org/pdf/2605.06408)
- Density gradient (Table 6): Ours on 5090 took 0.020/0.066/0.118/0.221/0.540/1.102/– s; Ours on H200 took 0.019/0.071/0.132/0.245/0.561/1.084/1.588 s; gDel3D on 5090 took 0.027/0.149/0.283/0.590/1.476/–/– s. — [arXiv PDF](https://arxiv.org/pdf/2605.06408)
- Clustered data with 10 clusters (Table 4):
  - gDel3D on 5090: 0.056/0.148/0.284/3.120/–/–/– s.
  - Ours on 5090: 0.175/0.341/0.386/0.675/1.224/1.423/– s.
  - Ours on H200: 0.456/0.677/0.720/1.027/1.732/1.848/2.530 s.
  - Basselin* failed every clustered case.
  - Source: [arXiv PDF](https://arxiv.org/pdf/2605.06408)
- Real Radiant-Foam / Mip-NeRF360 point sets (Table 1):
  - bicycle (4.2M): Ours 0.570 s and gDel3D 1.204 s on 5090; Ours 0.684 s and gDel3D 0.893 s on H200; HXT 1.243 s; Geogram 1.593 s.
  - room (1.9M): Ours 0.240 s and gDel3D 0.542 s on 5090.
  - RF Del took 16.9–250.7 s.
  - The paper's summary: gDel3D is "∼55% slower than our method on the H200 and ∼137% slower on the 5090" on real scenes.
  - Source: [arXiv PDF](https://arxiv.org/pdf/2605.06408)
- Derived throughput (my arithmetic):
  - White noise, 10M points on RTX 5090: about 9.3 M cells/s (1.078 s).
  - White noise, 15M points on H200: about 9.5 M cells/s (1.582 s).
  - gDel3D, 5M white noise on 5090: about 3.4 M pts/s.
  - bicycle on 5090: about 7.4 M cells/s.
  - Clustered, 15M points on H200: about 5.9 M cells/s.
- Power diagrams: runtimes are similar to the unweighted case. White noise with weights took 1.216 s at 10M (5090) and 1.811 s at 15M (H200). Runtime grows with weight spread: bicycle goes from 0.595 s to 2.243 s as the weight ratio goes from 0 to 1e-1. — [arXiv PDF](https://arxiv.org/pdf/2605.06408)

**B. gDel3D (Cao, Nanjappa, Gao, Tan, I3D 2014; code github.com/ashwin/gDel3D)** — pre-2015, but it is still the GPU Delaunay baseline that 2025–2026 papers compare against.
- Hybrid algorithm. On the GPU, massively parallel point insertion and flipping produce a nearly-Delaunay triangulation. A CPU pass then repairs it with conservative star splaying. — [NUS gDel3D page](https://www.comp.nus.edu.sg/~tants/gdel3d.html); [ACM DL](https://dl.acm.org/doi/10.1145/2556700.2556710)
- Speedup claims over CGAL vary by source: "up to 6 times" in the thesis, "up to 10 times" on the project page, and in the paper 4–6× rising to 8–10× as N grows. — [hgpu summary](https://hgpu.org/?p=9609); [NUS page](https://www.comp.nus.edu.sg/~tants/gdel3d.html); [ACM DL PDF](https://dl.acm.org/doi/pdf/10.1145/2556700.2556710)
- In 2026 measurements it exhausts memory beyond about 5M points on an RTX 5090 (32 GB) and even on an H200 for clustered 10M points. On clustered sets at or below 1M it is the fastest GPU method (0.053–0.285 s). — [Taveira et al. arXiv PDF](https://arxiv.org/pdf/2605.06408)

**C. Gao & Chen, "Local DeWall: Billion-scale 3D Delaunay triangulation on a single GPU", Computer-Aided Design 199 (2026) 104107, DOI 10.1016/j.cad.2026.104107** (full text not accessible)
- The method computes only a striped subset of each vertex's one-ring, the "Local Delaunay Wall", and stores it as an ordered point list. This cuts per-vertex memory by more than two-thirds. — [ScienceDirect abstract](https://www.sciencedirect.com/science/article/abs/pii/S0010448526000771) (seen via search snippet; the page itself returned 403)
- Highlights claim a "2× speedup". Search snippets also say it scales to 1.2 billion points (8.4 billion tetrahedra) on one consumer GPU, "2.4× the previous GPU capacity". The snippets give no GPU model, precision, baseline for the 2×, or throughput. Unverified. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010448526000771); [dblp CAD vol. 199](https://dblp.org/db/journals/cad/cad199.html); bibliography confirmed via [Crossref](https://api.crossref.org/works?query.bibliographic=Local+DeWall+Billion-scale+3D+Delaunay+triangulation&rows=3)
- One search snippet says it "walks across adjacent tetrahedra in a compact point-based representation". Unverified secondary paraphrase. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010448526000771)

**D. Gao & Chen, "Parallel 3D Delaunay Triangulation on the GPU", Computer-Aided Design 189 (Dec 2025), DOI 10.1016/j.cad.2025.103933** (full text not accessible)
- The paper describes itself as the "first fully GPU-parallel algorithm" for 3D DT. It extends Chen & Gotsman's Local Delaunay Lemma to R^3, uses a half-space-intersection approach, and outputs lists of Delaunay tetrahedra. Numbers were not obtainable. — [ScienceDirect abstract](https://www.sciencedirect.com/science/article/abs/pii/S0010448525000946) (via snippets); DOI via [Crossref](https://api.crossref.org/works?query.bibliographic=Parallel+3D+Delaunay+Triangulation+on+the+GPU+Gao+Chen&rows=3)

**E. Xiao, Cao, Chen, "Efficient Computation of Voronoi Diagrams Using Point-in-Cell Tests", arXiv:2509.07175 (v2 Feb 2026, TVCG-format preprint, says it was first submitted to SIGGRAPH Asia 2025)**
- Computes clipped/restricted VDs. An edge-based search with point-in-cell tests picks the bisectors, and a cell is clipped only when an edge straddles it. Runs on an RTX 3080 (10 GB), all GPU codes in 32-bit floats. — [arXiv:2509.07175](https://arxiv.org/abs/2509.07175)
- 5M 3D cells in a cube on the RTX 3080:
  - White noise: Ray et al. 3.12 s, Basselin et al. 2.81 s, ours 0.49 s.
  - Blue noise: 0.34 s, 0.42 s and 0.26 s respectively.
  - Timings cover cell construction plus centroids and exclude GPU allocation and transfer.
  - Derived: about 10 M cells/s (white noise), not end-to-end.
  - Source: [arXiv PDF](https://arxiv.org/pdf/2509.07175)
- Ray et al.'s code ran out of GPU memory above 7M white-noise sites on 10 GB. — [arXiv PDF](https://arxiv.org/pdf/2509.07175)

**F. Basselin, Alonso, Ray, Sokolov, Lefebvre, Lévy, "Restricted Power Diagrams on the GPU", CGF 2021; code github.com/basselin7u/GPU-Restricted-Power-Diagrams**
- Per-cell clipping, kNN plus security radius, multi-pass memory strategy, shared-memory per-thread arrays that are relaunched with larger sizes on overflow. — [Taveira et al. App. A.4](https://arxiv.org/pdf/2605.06408); [Xiao et al.](https://arxiv.org/pdf/2509.07175)
- The authors' presentation slides (local copy; source URL not recorded, so treat as unverified) claim "Process 4300K seeds per second / 60K seeds per second" against Liu et al. (parallel 3D clipped Voronoi), and "One order of magnitude faster than CPU".

**G. Ray, Sokolov, Lefebvre, Lévy, "Meshless Voronoi on the GPU", ACM TOG 37(6) 2018**
- Each cell is computed independently from its k nearest neighbours, with constant memory per thread and no synchronization. The authors report about one order of magnitude over the fastest multicore CPU codes. — [HAL record](https://hal.archives-ouvertes.fr/hal-01927559); [SIGGRAPH history](https://history.siggraph.org/?p=198036)
- Absolute numbers were not retrieved (HAL blocked). The Basselin slides (unverified copy) say a GTX 1080 "can't reach 10M points for white noise".

**H. Other items**
- votess (Singh, Byrohl, Nelson, arXiv:2412.04514): a SYCL implementation of grid kNN plus clipping with a security radius and CPU fallback. The text gives no GPU throughput numbers. — [arXiv:2412.04514](https://arxiv.org/html/2412.04514v2)
- Radiant Foam's GPU Delaunay ("RF Del", Govindarajan et al. ICCV 2025) takes tens to hundreds of seconds for 2–4M points in Taveira's tests. — [arXiv:2605.06408](https://arxiv.org/pdf/2605.06408)
- CPU reference points: HXT/Marot et al. report about 10^7 pts/s and 3 B tets in under 1 minute on a 64-core EPYC. — [arXiv:1805.08831](https://arxiv.org/pdf/1805.08831). Lo's CPU code reports >5 M tets/s for 1 B points. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0168874X15000785)

### Inferences
- At 1–15 M points, a well-engineered FP32 one-thread-per-cell clipping code on a 2025 GPU reaches about 10^7 cells/s end-to-end. That is roughly the same throughput as the best 16-core CPU code (HXT, about 4–6.6 M pts/s derived at 5–15M) and 2–3× it on Poisson data. GPU advantages over HXT are therefore modest, not orders of magnitude.
- gDel3D's limits are memory and clustered or large inputs, not raw speed at small N.
- Local DeWall leads on reported scale (1.2 B points). Whether it leads on throughput cannot be decided without its tables.

### Gaps
- Local DeWall and the CAD 2025 predecessor: GPU model, precision, timings, distributions, robustness and code availability were all inaccessible (ScienceDirect 403). Only abstract and highlight snippets were seen.
- Ray et al. 2018 absolute timings were not retrieved.
- No 2015–2026 GPU paper reporting >2×10^7 3D cells/s end-to-end was found.

## Are there non-clipping strategies (empty-circumsphere tests, Delaunay walks, star construction, DeWall-style) on GPU, and what do they report?

### Takeaway
Yes. Star-based and Local-Delaunay-Lemma methods exist: gStar4D (star splaying in R^4, before 2015), gDel3D (insertion and flipping plus star splaying), and Gao & Chen's 2025 Local Delaunay Lemma method and 2026 Local DeWall, both of which build each vertex's star / one-ring in parallel. Xiao et al. replace kNN-plus-security-radius with point-in-cell tests. Only gDel3D among these has independent 2026 benchmark numbers.

### Cited Findings
- gStar4D lifts the points to R^4 and uses massively parallel star construction and star splaying. It reports up to 5× over sequential 3D Delaunay codes; gDel3D reports up to 6×, and the thesis says gDel3D "is shown to be a better choice than gStar4D for certain point distributions". gReg3D extends the approach to regular triangulations, reporting up to 4×. — [Nanjappa thesis (NUS 2012)](https://codeyarns.com/daariga/papers/gdel3d_thesis.pdf); [ResearchGate](https://www.researchgate.net/publication/277570650_Delaunay_triangulation_in_R3_on_the_GPU)
- Gao & Chen 2025 generalize the Local Delaunay Lemma to R^3, so each vertex's Delaunay star can be computed locally. They use half-space intersection and output tetrahedron lists. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010448525000946) (snippet-level)
- Local DeWall 2026 keeps only a "striped subset of each vertex's one-ring", the Local Delaunay Wall, as an ordered point list, with more than two-thirds less memory per vertex. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010448526000771) (snippet-level)
- Xiao et al. use an edge-based search with point-in-cell tests and only one nearest-neighbour structure, so no kNN lists are stored. They report lower peak GPU memory than Ray et al. and Basselin et al. across white noise, perturbed grid and blue noise. — [arXiv:2509.07175](https://arxiv.org/pdf/2509.07175)

### Inferences
- The Local-Delaunay-Lemma family (Gao & Chen) is the most direct published GPU analogue to "find each point's Delaunay star by local empty-sphere reasoning". Its throughput relative to clipping-based Taveira et al. is unknown. Taveira et al. do not benchmark Gao & Chen, although Xiao et al. cite the 2025 paper.

### Gaps
- No full-text details are available for the Gao & Chen thread-per-point organization or walk strategy.
- I found no GPU "parallel DeWall" divide-and-conquer paper from 2015–2026 other than Local DeWall.

## How do these methods discover neighbours, certify completeness, and organize per-point work (fused or separate search)?

### Takeaway
Clipping methods split into two kinds:
- kNN plus security radius, with search separate from clipping and retry/fallback (Ray 2018, Basselin 2021, votess). This fails or blows up memory on non-uniform data.
- Fused, ordered search (Taveira 2026: best-first BVH with directional conservative bound; Xiao 2025/26: point-in-cell edge search). These are robust to clustered data.

Everything is one thread per site. Delaunay methods (gDel3D) use global kernels per insertion and flip round, one thread per point, tetrahedron or flip.

### Cited Findings
- Taveira et al., fused one-thread-per-site kernel:
  - Neighbour discovery: the BVH is built with cuBQL (leaf size 17 on H200, 10 on RTX 5090). An optional K=8 kNN warm start pre-clips the cell; it is enabled on the 5090 (large L2) and disabled on the H200.
  - Order: best-first traversal with an unsorted stack (append on push, linear-scan min on pop).
  - Completeness: a child is pruned when the lower-bound bisector distance from its box exceeds a directional radius taken from the cell's AABB corners in that octant. Deferred nodes are re-tested on pop, and the search terminates when the stack is empty.
  - Ablation on RTX 5090: depth-first traversal is 27.4× slower (3.96–80.3×) on clustered data, 1.30× on Poisson and 37.2× on real-world data; isotropic culling is 1.26–1.40× slower.
  - Cost split: BVH build is 4.8% of the time; within the fused kernel, 96.2% goes to clipping.
  - Source: [arXiv:2605.06408](https://arxiv.org/pdf/2605.06408)
- Taveira et al. memory layout: per-thread cell state lives in coalesced global memory in an SoA layout strided across warp lanes, not in shared memory, for occupancy. Planes are float4, vertices are uchar3 plane-index triples with cached positions, and the plane array is compacted at 85% occupancy. Sites are packed as float4 and loaded with LDG.128. — [arXiv PDF](https://arxiv.org/pdf/2605.06408)
- Basselin et al.: kNN, then clipping in shared-memory arrays. When an array overflows, the kernel is relaunched with larger arrays and the construction is redone from scratch. Taveira et al. call this "a primary limitation on non-uniform point distributions", and their CUDA port fails all non-Poisson cases. — [arXiv PDF](https://arxiv.org/pdf/2605.06408)
- votess: grid sort and kNN, then clipping; a security radius validates each cell and failures fall back to CPU. — [arXiv:2412.04514](https://arxiv.org/html/2412.04514v2)
- Xiao et al.: one thread per site, or per simplex–cell pair for restricted diagrams, "without inter-thread data synchronization". — [arXiv PDF](https://arxiv.org/pdf/2509.07175)
- gDel3D/gFlip3D: global CUDA kernels, "one thread per point" for distance and location, "one thread per active tetrahedron", "one thread per flip". — [Nanjappa thesis](https://codeyarns.com/daariga/papers/gdel3d_thesis.pdf)

### Inferences
- For a moving-mesh hydro pipeline, the evidence favours fused, ordered (near-to-far) search with a conservative per-cell bound over fixed-k kNN lists. Fixed-k lists cost memory and need retries on clustered data.

### Gaps
- No paper found uses multiple threads (warp or team) per cell for 3D construction with published throughput. Everything found is thread-per-cell or global-kernel Delaunay.

## What exactness/robustness strategies are used on GPUs?

### Takeaway
Most fast GPU Voronoi codes are FP32 with no exactness guarantee. Exact or robust predicates on GPU appear mainly in the gDel3D lineage, which uses fast filtered predicates plus a separate exact kernel, symbolic perturbation and a CPU repair pass. Xiao et al. avoid one floating-point topological failure by construction. Taveira et al. measure 0.1–0.2% adjacency mismatches against CGAL exact predicates.

### Cited Findings
- gDel3D/gFlip3D:
  - Uses Shewchuk-style adaptive predicates split into two kernels. A fast kernel marks the items needing exact checks, and an exact kernel with fewer threads processes them, because the exact insphere needs "hundreds of registers" and up to 14880 floats of local storage.
  - Degeneracies are handled by symbolic perturbation in the exact kernel.
  - The CPU star-splaying pass guarantees a Delaunay output.
  - Source: [Nanjappa thesis](https://codeyarns.com/daariga/papers/gdel3d_thesis.pdf); [NUS page](https://www.comp.nus.edu.sg/~tants/gdel3d.html)
- Taveira et al.: FP32 only. Compared with CGAL exact predicates, 0.1–0.2% of adjacencies mismatch on real data, "traced to near-coincident candidate planes"; ray-integrated density differences are mean <1.7e-5. The conclusion says it "does not target the exact geometric guarantees". — [arXiv PDF](https://arxiv.org/pdf/2605.06408)
- Xiao et al. note that FP clipping in kNN-based GPU codes can misclassify triangles and produce a disconnected removal boundary. Their algorithm avoids this by always clipping from a corner certified to lie outside, using only the connected component R_p. They also claim the "parallel version... still ensures exact results" in the sense of a complete neighbour set. — [arXiv PDF](https://arxiv.org/pdf/2509.07175)
- Ray/Basselin/votess use a security-radius certificate for completeness, which is a geometric rather than arithmetic guarantee, with retry or CPU fallback. — [votess](https://arxiv.org/html/2412.04514v2); [Taveira et al.](https://arxiv.org/pdf/2605.06408)

### Inferences
- No fast GPU Voronoi-cell (clipping) method found in 2015–2026 uses exact or integer arithmetic. The two-pass "filter, then exact on flagged items" idea is documented only for Delaunay flipping codes (gDel3D).

### Gaps
- I found no GPU 3D construction paper using integer or fixed-point coordinates with exact predicates and reported throughput.
- The Local DeWall robustness strategy is unknown (no full-text access).
