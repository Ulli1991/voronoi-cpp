# Astrophysical moving-mesh / Voronoi finite-volume codes: mesh construction strategy, reuse, parallelism, cost, robustness

Research cutoff: searches done 2026-10-29. About 26 tool calls. Several PDFs could not be machine-read (raw PDF streams), so some numbers come from ar5iv HTML renderings that were summarised by a fetch tool. Numbers marked "from ar5iv" were quoted by that tool. Spot-check them against the PDFs before you publish.

## Q1. AREPO: rebuild policy, algorithm, exact arithmetic, cost, GPU port

### Takeaway
AREPO rebuilds a global per-MPI-domain Delaunay tessellation every timestep. It uses incremental insertion in Peano-Hilbert order with jump-and-walk point location and 3D flips (2-3, 3-2, 4-4, plus degenerate cases). Its predicates are filtered and fall back to exact GMP integer arithmetic on coordinates mapped to [1,2) (53-bit mantissa). The only published absolute speed is "several tens of thousands of tetrahedra per second", which Springel calls comparable to an SPH neighbour search. No GPU port of AREPO's mesh construction was found. The only GPU work found is AREPO-RT, which ports the radiative transfer part.

### Cited Findings
- Algorithm: incremental insertion, Delaunay first and Voronoi as its dual. Worst case is O(N^2), but random or space-filling insertion order gives roughly O(N log N). Insertion splits a tetrahedron and local flips restore the Delaunay property: 2→3, 3→2, 4→4, and degenerate 2→6 and n→2n. Points are inserted in Peano-Hilbert order and located by "jump and walk" from the last tetrahedron. — [Springel 2010, MNRAS 401, 791 (arXiv:0901.4107, ar5iv)](https://ar5iv.labs.arxiv.org/html/0901.4107)
- Robustness: round-off error is estimated for every in-sphere and orientation test. If the sign is uncertain, the test is redone with exact long-integer arithmetic (GMP) on coordinates mapped to 53-bit integers in [1,2). — [Springel 2010 (ar5iv)](https://ar5iv.labs.arxiv.org/html/0901.4107). The 2020 public-release paper confirms GMP is used "for robustly resolving degeneracies in the mesh construction". — [Weinberger, Springel & Pakmor 2020, ApJS 248, 32 (arXiv:1909.04667)](https://ar5iv.labs.arxiv.org/html/1909.04667)
- Rebuild policy: the mesh is rebuilt every timestep. Springel writes "even if the mesh is reconstructed each timestep, as in our current approach", and the first step of each hydro step is a new tessellation. — [Springel 2010 (ar5iv)](https://ar5iv.labs.arxiv.org/html/0901.4107). The public-release paper says the time integrator needs "only a single mesh construction ... for each time step". — [Weinberger+2020](https://ar5iv.labs.arxiv.org/html/1909.04667)
- Speed: "tessellation speeds on the order of several tens of thousands of tetrahedra per second". The paper gives no CPU model. It says this is "comparable to or only slightly more than the work needed for SPH neighbour search". — [Springel 2010 (ar5iv)](https://ar5iv.labs.arxiv.org/html/0901.4107)
- Large run: "The mesh construction took 516 seconds on 1024 CPUs of an SGI Altix 4700" for about 5 billion points (about 34 billion tetrahedra). — [Springel 2010 (ar5iv)](https://ar5iv.labs.arxiv.org/html/0901.4107). (My inference: this is about 9.5e6 points/s aggregate, or about 9.3e3 points/s per CPU. I computed this; it is not quoted in the paper.)
- Memory: about 36 B per tetrahedron (32-bit indices) or 68 B (64-bit). There are about 6.77 tetrahedra per point, giving about 244 B per point (about 460 B with 64-bit). — [Springel 2010 (ar5iv)](https://ar5iv.labs.arxiv.org/html/0901.4107)
- MPI: Peano-Hilbert domain decomposition with imported ghost points. Method 1 grows a per-point search radius h_i until h_i exceeds twice the largest circumsphere radius of the point's tetrahedra. Method 2 searches the circumspheres of local tetrahedra directly and is better with individual timesteps. Ghosts add "typically 3−10%" more cells per process. — [Springel 2010 (ar5iv)](https://ar5iv.labs.arxiv.org/html/0901.4107). Each MPI domain builds its own local mesh plus imported ghosts. Parallelism is MPI and deterministic. — [Weinberger+2020](https://ar5iv.labs.arxiv.org/html/1909.04667)
- Local refinement and derefinement: a point is split into two points offset by 0.025 r_cell in a random direction. This needs no global load balancing. — [Weinberger+2020](https://ar5iv.labs.arxiv.org/html/1909.04667)
- AREPO is cited as scaling to about 100,000 cores (Pakmor et al. 2023). This was secondary context in the AREPO-RT GPU paper and was not verified in the primary source. — [Zier et al. 2024, arXiv:2404.17630](https://arxiv.org/pdf/2404.17630)
- GPU: "Adapting AREPO-RT for Exascale Computing" moves "all RT-related calculations to GPUs". It reports about 15x speedup on standard benchmarks versus the CPU version and about 3x overall in an Epoch-of-Reionization run, because RT becomes subdominant. The abstract does not say the Voronoi mesh construction was ported, which implies it stays on CPU. — [arXiv:2404.17630](https://arxiv.org/abs/2404.17630)
- A 2026 paper calls the code "AREPO-2". No GPU mesh content appeared in the retrieved text. — [arXiv:2607.05504](https://arxiv.org/pdf/2607.05504)

### Inferences
- AREPO does not reuse topology between steps. It retriangulates every step, and uses local timestepping and active-interface fluxes to limit the cost of the work around each rebuild.
- If the Altix-era rate of about 1e4 points/s per core still roughly holds on modern cores, it is several times faster today (unverified). It is still orders of magnitude below GPU structured-grid hydro rates (see Q5).

### Gaps
- I found no published runtime fraction for AREPO mesh construction in production runs. The public-release paper's performance section (beyond 100k characters of ar5iv) was not read. Pakmor et al. 2016 ("Improving the convergence properties of AREPO", gradient and time-integration changes) was not fetched.
- No AREPO GPU mesh construction was found for 2020-2026, but the absence is unconfirmed (the search only covered arXiv abstracts).

## Q2. Shadowfax and ShadowSWIFT (SWIFT moving mesh)

### Takeaway
Shadowfax (2016) uses a Springel-style incremental Delaunay construction with exact integer predicates (Boost.Multiprecision). It rebuilds "the relevant parts" of the mesh every step. Serially it builds about 9e4 cells/s in 2D and about 1.5e4 cells/s in 3D, and in 3D grid construction costs about twice the hydro integration. ShadowSWIFT is SWIFT's moving-mesh scheme on the `moving_mesh` branch. Its documentation confirms it exists, with MPI support for 1D/2D/3D, but no algorithm paper with timings could be retrieved.

### Cited Findings
- Shadowfax builds Voronoi from the dual Delaunay with incremental construction, and "the relevant parts of the mesh are reconstructed for every time step". A mesh-evolution (topology-reuse) algorithm was tried but is not in the public version ("Vandenbroucke & De Rijcke, in preparation"). — [Vandenbroucke & De Rijcke 2016, A&C 16, 109 (arXiv:1605.03576, ar5iv)](https://ar5iv.labs.arxiv.org/html/1605.03576)
- Exact arithmetic: all geometric tests are exact. Coordinates are mapped to [1,2] and integer mantissas are used. The maximum integer size is precomputed, and long integers come from Boost Multiprecision. Shewchuk's adaptive predicates were rejected because their FPU assumptions "are not met on all hardware". — [Shadowfax (ar5iv)](https://ar5iv.labs.arxiv.org/html/1605.03576)
- Speed (serial, CPU not stated for this figure): about 90,000 cells/s in 2D and about 15,000 cells/s in 3D. This includes boundary handling and computing centroids, volumes, face midpoints and areas. In 3D, grid construction takes about 2x the hydro integration time. Gravity dominates total runtime at large N. Scaling tests ran on 4× 2.7 GHz Xeon (8 cores each), GCC 4.8.4, OpenMPI 1.6.5. — [Shadowfax (ar5iv)](https://ar5iv.labs.arxiv.org/html/1605.03576)
- Parallelism is MPI only, with Hilbert-key decomposition by particle count and ghost generators. Strong and weak scaling are described as "rather poor". Memory was 137 MB for a 20,000-cell Evrard test, with Delaunay and Voronoi stored together. — [Shadowfax (ar5iv)](https://ar5iv.labs.arxiv.org/html/1605.03576)
- ShadowSWIFT: "implementation of the moving-mesh finite-volume method for hydrodynamics in SWIFT". It needs a Riemann solver (e.g. `--with-hydro="shadowswift" --with-riemann-solver="hllc"`) and lives on the `moving_mesh` branch while being merged into master. It supports 1D/2D/3D hydro over MPI, gravity, cosmology and cooling. Cells must tile the whole volume, so vacuum needs zero- or low-mass particles. There is no MHD, RT or star-formation feedback yet. — [SWIFT docs: ShadowSWIFT](https://swift.strw.leidenuniv.nl/docs/HydroSchemes/shadowswift.html)
- Merge request "Draft: Merge the moving mesh hydro scheme in master" (!1749) exists on the SWIFT GitLab. The content needs a login, so it was not readable. — [SWIFT GitLab MR 1749](https://gitlab.cosma.dur.ac.uk/swift/swiftsim/-/merge_requests/1749)
- The SWIFT code paper (Schaller et al. 2024) lists a moving-mesh solver "akin to Vandenbroucke & De Rijcke (2016)" as in development. — [Schaller et al., arXiv:2305.13380](https://arxiv.org/html/2305.13380)
- SWIFT's GPU work (2025) targets SPH via task offload, not moving mesh. — [arXiv:2505.14538](https://arxiv.org/pdf/2505.14538)

### Inferences
- SWIFT's task-based design (cell pairs as tasks) suggests ShadowSWIFT builds Delaunay or Voronoi locally per SWIFT top-level or leaf cell, with neighbour-cell particles as ghosts. This is unverified because no paper or algorithm description was retrievable.

### Gaps
- I could not find a refereed ShadowSWIFT paper (likely authors Y. Uyttenhove, B. Vandenbroucke et al., Ghent; unverified) with algorithm details or timings. The user's "~2023-2025 task-based local Delaunay" paper could not be located with web search. Check ADS directly for "Uyttenhove" or "ShadowSWIFT".

## Q3. Other Voronoi / moving-mesh codes (MANGA, TESS/Tess2, RICH, DISCO, GIZMO, votess, PARAVT)

### Takeaway
MANGA (ChaNGa) is the clearest published example of local per-cell Voronoi construction in an astrophysical moving-mesh code. It clips each cell with voro++ against distance-sorted neighbours, rebuilds every step, and reports about 1e4 cells/s per core, against about 1e5 cells/s per core for voro++ with its native containers. votess (2024) is a SYCL CPU/GPU cell-by-cell Voronoi library from the IllustrisTNG/AREPO community, explicitly aimed at on-the-fly use in hydro codes. Tess/Tess2 are distributed-memory Delaunay/Voronoi codes for analysis, built on Qhull or CGAL.

### Cited Findings
- MANGA: "largely based on AREPO's algorithm", but it computes the tessellation "directly" with voro++ instead of a Delaunay dual. Each cell starts from a large box and is clipped by bisecting planes of neighbours sorted by distance. Clipping stops when the remaining neighbours are farther than twice the distance to the farthest cell vertex. Cells are built one at a time because voro++ containers are "expensive to instantiate". The tessellation is recomputed every step. — [Chang, Wadsley & Quinn 2017, MNRAS (arXiv:1707.05333, ar5iv)](https://ar5iv.labs.arxiv.org/html/1707.05333)
- MANGA speed: voro++ with native containers reaches "100K 3-D Voronoi cells per second" on one core of an Intel Core i7-3770 @ 3.40 GHz. MANGA's cell-by-cell use reaches "10K cell per second per core", which the authors call adequate. They blame part of the gap on a domain decomposition not tuned for tessellation. Parallelism is Charm++ over-decomposition, and because each Voronoi calculation is local it is "straightforward to parallelize". — [Chang+2017 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1707.05333)
- MANGA time integration rebuilds Voronoi cells at the start and at the half step, so there are multiple tessellations per step. Pakmor et al. (2016) use fewer tessellations but two Riemann solves. — [Prust & Chang 2019, arXiv:1904.09256](https://arxiv.org/pdf/1904.09256); [Prust 2020, arXiv:2002.04287](https://arxiv.org/pdf/2002.04287)
- votess: "Voronoi cell-by-cell computation algorithm" that outputs cell geometry and neighbour connectivity rather than a combinatorial mesh. The authors argue this suits data-parallel hardware better than sequential insertion or Bowyer-Watson. It uses SYCL for CPUs and GPUs and claims "significant performance improvements over established single-threaded programs". The abstract gives no numbers. Authors: S. D. Singh, C. Byrohl, D. Nelson (Dec 2024, JOSS). — [arXiv:2412.04514](https://arxiv.org/abs/2412.04514)
- TESS (Duffell & MacFadyen 2011) is a relativistic hydro code on a moving Voronoi mesh. — [arXiv:1104.3562](https://arxiv.org/abs/1104.3562). Duffell & MacFadyen also studied grid noise from rapid topology changes. — [arXiv:1407.7300](https://arxiv.org/abs/1407.7300)
- Tess / Tess2 (Peterka, Morozov, Phillips, SC14): a parallel Delaunay/Voronoi algorithm that finds which points to exchange between subdomains automatically, with periodic and wall boundaries. — [mrzv.org SC14 page, DOI 10.1109/SC.2014.86](https://mrzv.org/publications/distributed-delaunay/). Tess2 wraps Qhull or CGAL with the DIY block-parallel library. — [GitHub diatomic/tess2](https://github.com/diatomic/tess2). k-d tree decomposition is "up to 50 times faster" than regular-grid decomposition on late-stage cosmology data. Scaling tests reached 4096^3 points and 128Ki MPI processes. — [Morozov & Peterka SC16](https://mrzv.org/publications/distributed-delaunay-kdtree/); [LBNL-1007265](https://vis.lbl.gov/archive/Publications/2016/LBNL-1007265.pdf)
- PARAVT: an MPI plus Qhull parallel Voronoi code for cosmology analysis. — [arXiv:1601.06429](https://ar5iv.arxiv.org/html/1601.06429)
- Density-gridding from Voronoi Lagrangian data: [arXiv:1710.07108](https://arxiv.org/pdf/1710.07108) (not read in detail).
- Arbitrary high-order ALE FV/DG on moving 2D Voronoi meshes with topology changes. — [Gaburro et al., arXiv:1905.00967](https://arxiv.org/abs/1905.00967)

### Inferences
- The two families split cleanly. Delaunay-dual codes (AREPO, Shadowfax, TESS, Tess2) get globally consistent topology for free but need serial-ish insertion per domain. Per-cell clipping codes (MANGA via voro++, votess) parallelize trivially but must make sure the faces of neighbouring cells agree. None of the per-cell sources retrieved describe an explicit cross-cell face-consistency check or exact predicates. That is a robustness gap relative to AREPO and Shadowfax.
- The voro++ native-container rate (about 1e5 cells/s per core, i7-3770) is about 7x the Shadowfax 3D serial rate (1.5e4 cells/s). That 1.5e4 figure includes geometry, though.

### Gaps
- RICH (Yalinewich, Steinberg & Sari 2015), DISCO (Duffell 2016), FVMHD3D, and GIZMO's MFV/MFM (Hopkins 2015) were not fetched in this pass. No construction timings are recorded for them. From memory and unverified: RICH is a 2D Voronoi code using a Delaunay-dual construction, DISCO uses a moving 1D-in-azimuth (shearing cylindrical) mesh with no Voronoi, and GIZMO is meshless (no tessellation; faces come from kernel-weighted partitions).
- Hess & Springel 2010 (Voronoi-SPH) and DTFE GPU implementations were not found or verified. No GPU DTFE paper from 2020-2026 with throughput figures turned up.
- votess absolute throughput (cells/s on a given GPU) was not in the abstract. The JOSS paper or GitHub README needs reading.

## Q4. GPU moving-mesh / Voronoi hydro codes (2018-2026)

### Takeaway
I found no production astrophysical moving-mesh Voronoi hydro code that builds its mesh on the GPU. The only GPU items found are AREPO-RT (RT on GPU, mesh on CPU) and votess (a GPU-capable cell-by-cell Voronoi library meant for future in-situ use).

### Cited Findings
- AREPO-RT GPU: ports the RT calculations only, about 15x faster on RT benchmarks and about 3x faster overall in an EoR run. — [arXiv:2404.17630](https://arxiv.org/abs/2404.17630)
- votess targets "heterogeneous platforms, from CPUs and GPUs" via SYCL and positions itself for use in hydro codes. — [arXiv:2412.04514](https://arxiv.org/abs/2412.04514)
- SWIFT's GPU paper is about SPH task offload, not moving mesh. — [arXiv:2505.14538](https://arxiv.org/pdf/2505.14538)

### Gaps
- Searches for "GPU moving mesh hydrodynamics Voronoi" and "Arepo GPU" found nothing beyond these. This is not a proof of absence, especially for 2025-2026 preprints and talks.

## Q5. Reference GPU hydro throughput (structured or AMR grid codes)

### Takeaway
The only verified figure is AthenaK at more than 1e9 cell-updates/s for 3D hydro on a single NVIDIA Grace Hopper (GH200), with about 80% parallel efficiency on 65,536 AMD GPUs on Frontier. The abstract does not state the precision.

### Cited Findings
- AthenaK: "over one billion cell updates per second for hydrodynamics in three-dimensions on a single NVIDIA Grace Hopper processor", and "typical parallel efficiency of 80% on 65536 AMD GPUs on the OLCF Frontier system". — [Stone et al. 2024, arXiv:2409.16053](https://arxiv.org/abs/2409.16053)

### Inferences
- For scale, against the CPU Voronoi construction rates above (about 1e4 to 1e5 cells/s per core), rebuilding a Voronoi mesh every step on CPU is about 4 to 5 orders of magnitude slower per device than a GPU Cartesian hydro update. GPU mesh construction is therefore the key enabler for a GPU moving-mesh code. This is an order-of-magnitude comparison only: the hardware generations and precisions differ.

### Gaps
- No double-precision A100 or H100 zone-cycles/s figures were retrieved for AthenaK, K-Athena, Cholla, IDEFIX or ARK. The AthenaK full paper (tables), the K-Athena paper (Grete et al. 2021, IEEE TPDS), the Cholla papers (Schneider & Robertson 2015), and the IDEFIX paper (Lesur et al. 2023, A&A) contain them but were not fetched. Do not use remembered values without checking.
