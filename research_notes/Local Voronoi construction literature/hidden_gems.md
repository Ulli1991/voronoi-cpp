# Hidden gems: fast Voronoi/Delaunay construction and update for moving points (out-of-field / low-citation), up to Oct 2026

Scope note: about 27 search/fetch calls. Coverage is uneven. Active matter, graphics/OT, radiance-field and Kokkos/GitHub sources gave concrete results. MD per-atom GPU Voronoi, DTFE-on-GPU, 3D SPV codes, free-Lagrange reconnection details and non-English literature gave little or nothing usable (see Gaps). Target used for every relevance verdict: 3D, double precision, GPU, about 100 M cells/s per timestep.

## Q1. Per-timestep topology maintenance: who does rebuild vs. local test-and-repair, and how?

### Takeaway
Out-of-field codes really do use the "certify last step's topology, repair only failures" pattern. The best examples are cellGPU (2D, GPU test, CPU repair) and peclet-voro (3D, Kokkos GPU, Verlet-skin worklist plus near-miss certificate). peclet-voro's own measurements show the gain collapses once per-step displacement reaches about 1% of the interparticle spacing. Above that a full rebuild is as fast or faster, which is the main quantitative lesson for a hydro code.

### Cited Findings
**peclet-voro (Kokkos, CUDA/HIP/OpenMP; MIT; Zenodo, v1.0.4 dated 27 Sep 2026). The most relevant find, and very recent and uncited.**
- It is a Kokkos C++20 engine for dynamic 3D Voronoi and power (Laguerre) tessellation of moving particles in periodic and Lees–Edwards boxes, with incremental cell repair. — [GitHub](https://github.com/computational-chemical-engineering/peclet-voro); [Zenodo](https://zenodo.org/records/23001953)
- **Cell algorithm:** a GEOGRAM / Ray et al.-style "ConvexCell" in dual form. Cells are half-space intersections, one plane per neighbour plus 6 box planes. Each vertex is a triple of 1-byte plane indices with a cached dual-vertex position. Clipping marks triangles whose vertex lies outside the plane, finds the horizon, and re-triangulates it. Cuts are applied closest-first with a security-radius early exit. No adjacency is stored. Volume, facet areas and first moments come from a sort-free per-vertex scatter of signed determinants. The cold build is counting-sort grid, then worklist gather, then clip, then CSR publish. — [GitHub](https://github.com/computational-chemical-engineering/peclet-voro)
- **Repair (`MovingTessellation`):** a two-pass update driven by a persistent Verlet-skin worklist, with a per-cell reference position `xRef`. Seeds that leave their skin trigger re-gather, and the subset gather reuses the cold-build kernel on an index list. A purely seed-local certificate missed new faces from seeds outside the stored topology. That was about 0.1% of neighbour relations per step and a 1.6e-3 volume error over 400 steps. The fix stores "near-miss" candidates at build time (margin `nearMarginFrac` × skin, default 0.5) and re-tests them every step with `ConvexCell::planeGap`. The repair is then exact to 1e-11 (default tolerance) and 1e-15 (tight) over 400 steps, at about 80% of the previous speedup. — [GitHub](https://github.com/computational-chemical-engineering/peclet-voro)
- **Performance report** (FP64 unless noted, VORF_TOL = 1e-7; 48-core x86 CPU; NVIDIA GeForce RTX 5080, sm_120, CUDA 13.2). — [performance_report.md](https://raw.githubusercontent.com/computational-chemical-engineering/peclet-voro/main/docs/performance_report.md)
  - Cold build, uniform Poisson:
    - Serial, 1 core: 0.059 Mcells/s.
    - OpenMP, 8 cores: 0.336 / 0.323 / 0.328 / 0.331 Mcells/s at 20k / 100k / 200k / 1M.
    - CUDA, RTX 5080: 2.99 / 3.25 / 3.54 / 3.61 Mcells/s at 100k / 200k / 1M / 2M.
    - The "repair-ready" build, which also stores the certificate data, is 15–26% slower.
  - Peak repair throughput at δ/h = 1e-4: 21.8 Mcells/s on the GPU and 3.86 Mcells/s on 8 CPU cores.
  - Repair speedup over cold rebuild, CUDA at N = 500k:

    | δ/h | Speedup |
    |---|---|
    | 1e-4 | 6.5× |
    | 1e-3 | 2.9× |
    | 5e-3 | 1.1× |
    | 1e-2 | 0.86× |
    | ≥ 2e-2 | 0.84× |

  - Share of cells flagged per step, OpenMP at N = 200k:

    | δ/h | Cells flagged | Steps routed to full rebuild |
    |---|---|---|
    | 1e-4 | 1.3% | – |
    | 1e-3 | 12.4% | – |
    | 5e-3 | 46.7% | – |
    | 1e-2 | 71.5% | 83% |
    | 5e-2 | 99.7% | 100% |

  - Volume conservation:
    - Cold build: within 4e-15.
    - Repair: total volume to a few ×1e-15.
    - A few marginal cells at δ/h ≈ 5e-4–5e-3 reach worst-cell errors of about 5e-5 (CPU) and 8e-4 (GPU).
  - MPI, 50k cells/rank: cold build 846–866 ms per step and repair 149–156 ms on 1–8 ranks.
  - It cites about 1.0–1.05× voro++ for the single-core cold build.

**cellGPU (Sussman, Comput. Phys. Commun. 2017; arXiv:1702.02939). 2D SPV/Voronoi and AVM tissue models.**
- Each step a GPU kernel runs one thread per Delaunay triangle and tests the empty circumcircle. A single flag is copied to the host. If no test fails, the next step starts on the GPU with no memory copy. If some fail, the flagged cells are repaired on the CPU. A locality lemma uses the circumcircles of consecutive Voronoi vertices to bound each cell's candidate "Delaunay 1-ring", and CGAL reduces the candidates to the true neighbours. The data then goes back to the GPU. — [arXiv/ar5iv](https://ar5iv.labs.arxiv.org/html/1702.02939)
- A full retriangulation (CGAL by default, Bowyer–Watson optional) runs at initialization or when many neighbours need updating. A `globalOnly` flag forces a CGAL retriangulation every step. — [cellGPU docs](https://dmsussman.gitlab.io/cellGPUdocumentation/basicinfo.html); [voronoiModelBase](https://dmsussman.gitlab.io/cellGPUdocumentation/classvoronoiModelBase.html)
- **Performance and hardware:** up to "three orders of magnitude faster" than existing CPU implementations at large N, in the solid-like regime (P0 = 3.8, v0 = 0.01, Δt = 0.05). Benchmarks ran in double precision, except a dashed float curve on a GeForce 980Ti. Hardware was a Xeon E5-1620 v3 with Tesla K40, Quadro K620 or 980Ti, and a laptop with a GeForce 950M. The CPU branch is single-threaded. The article text gives no ms-per-step values; the numbers are only in figures. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1702.02939)
- **Weakness:** towards the fluid-like regime (P0 = 3.9 to 4.0) the CPU repair is "invoked more and more frequently" and performance degrades. The paper names an edge-flipping or star-splaying GPU repair as future work. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1702.02939); [GitHub](https://github.com/sussmanLab/cellGPU)

**TESS (Duffell & MacFadyen 2011; relativistic moving-mesh hydro; less known than AREPO).**
- It reuses last step's connectivity. Candidates for the next step are the neighbours of neighbours ("friends of friends"), justified because the Courant-limited step changes the true neighbour set only slightly. Each cell is then built locally: sort the candidates by angle and prune them with in-circle tests. — [arXiv 1104.3562](https://arxiv.org/html/1104.3562)

**Bernaschi, Lulli & Sbragaglia, "GPU Based Detection of Topological Changes in Voronoi Diagrams" (arXiv:1607.00908, 2016; soft-glassy lattice Boltzmann).**
- It is 2D and lattice-based. Droplets are labelled on the GPU with Swendsen–Wang-style clustering. A digital Voronoi diagram is built with one thread per lattice site. Delaunay adjacency comes from counting colour "gaps" around each site. T1 (plastic) events are found by comparing the past and current adjacency matrices.
- The test case is a 2048×1024 lattice with about 2,000 droplets, compared every 100 of 4×10^6 steps. The overhead is "within the typical running time fluctuations". No hardware or precision is given. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1607.00908)

**Radiance-field graphics ("Radiant Foam" lineage).**
- Radiant Foam (ICCV 2025) represents scenes as differentiable 3D Voronoi diagrams. The Delaunay triangulation must be recomputed as sites move during gradient-descent training, and that repeated reconstruction is called the scaling bottleneck. — [arXiv 2502.01157](https://arxiv.org/pdf/2502.01157); [CVF](https://openaccess.thecvf.com/content/ICCV2025/papers/Govindarajan_Radiant_Foam_Real-Time_Differentiable_Ray_Tracing_ICCV_2025_paper.pdf)
- Radiance Meshes (Mai et al., arXiv 2512.04076) uses Delaunay tetrahedra and notes that moving vertices causes topological discontinuities (edge flips). It handles them with a smooth field backbone rather than any update algorithm. — [arXiv 2512.04076](https://arxiv.org/abs/2512.04076)

**Theory and old literature.**
- Guibas–Mitchell–Roos kinetic Voronoi for moving points takes O(log n) per topological event. — [Stony Brook record](https://researchconnect.stonybrook.edu/en/publications/voronoi-diagrams-of-moving-points-in-the-plane/)
- Gavrilova & Rokne, "Swap conditions for dynamic Voronoi diagrams for circles and line segments", CAGD 16:89–106 (1999). Citation only; not read.
- A 2003 arXiv paper on dynamic and kinetic 3D Delaunay was seen only as a search snippet and not verified. — [search hit](https://www.arxiv.org/pdf/physics/0302018)
- Fritts (1985), "Three-dimensional algorithms for grid restructuring in Free-Lagrangian calculations", covers 3D Delaunay restructuring including moving and colliding surfaces. The abstract has no cost figures. — [NASA NTRS](https://ntrs.nasa.gov/citations/19870024133)
- The free-Lagrange lineage (Crowley, Fritts, Trease, Clark) is documented in Lecture Notes in Physics vols 238 (1985) and 395 (1991). — [Trease LANL 1987](https://sgp.fas.org/othergov/doe/lanl/lib-www/la-pubs/00228697.pdf); [LNP 395](https://link.springer.com/book/10.1007/3-540-54960-9)
- SimpleX2 (Paardekooper, Kruip & Icke, A&A 2010) does radiative transfer on a dynamic Delaunay grid with updates during the run. It uses CPU and MPI only. — [arXiv 0912.4273](https://arxiv.org/pdf/0912.4273)

### Inferences
- peclet-voro's table is the most directly useful number set found. Repair beats rebuild only when the per-step displacement is below about 5e-3 of the spacing. My reading is that δ/h means per-step displacement over mean spacing; the report's exact definition was not checked. A CFL-limited moving-mesh hydro step (Δt ≈ C·h/(c_s+|v−w|), with mesh velocity w ≈ local fluid velocity) typically moves points by about 1e-2–1e-1 of h in active regions. In that regime peclet-voro's own data show repair at 0.84–0.86× of rebuild speed. Topology reuse is therefore worth it mostly as candidate seeding (TESS-style friends-of-friends, or near-miss lists), not as skip-if-certified. Subsonic or quiescent regions and individual-timestep schemes, where most cells are inactive, are the exception.
- Absolute GPU throughput in these codes is far below 100 M cells/s. peclet-voro's cold build reaches 3.6 Mcells/s FP64 on an RTX 5080, a consumer Blackwell card with heavily reduced FP64 rate, so data-centre GPUs could be several times faster. That is unverified. Its repair peak of 21.8 Mcells/s applies only at δ/h = 1e-4.
- cellGPU's design (GPU certificate kernel, one host flag, repair only on failure) carries over to 3D. Its CPU repair is exactly the bottleneck seen in fluid-like regimes, so a 3D hydro code needs the repair on the GPU.
- peclet-voro's bug history is a useful warning. A purely seed-local certificate misses faces gained from seeds not in the stored topology (about 0.1% of relations per step). Any 3D test-and-repair needs a candidate margin, such as near-miss lists or a security radius, not only checks against existing neighbours.

### Gaps
- peclet-voro has no paper and no named authors. The exact definition of δ/h and the numeric skin were not stated; numbers come from the repo's own report and are not peer-reviewed.
- cellGPU ms-per-step values and the fraction of steps needing repair appear only in figures (Figs. 2–4) that could not be read numerically.
- No 3D SPV / Merkel–Manning (NJP 2018) per-step update strategy was found. The methods section was not reached; the model appears to retessellate 3D Voronoi each step, which is unverified. — [Manning lab blog](https://mmanning.expressions.syr.edu/blog/2018/02/06/article-by-matthias-merkel-and-lisa-manning-published-as-fast-track-communication-in-new-journal-of-physics); [arXiv 1802.09149](https://arxiv.org/pdf/1802.09149)
- SAMoS, AVM extensions and Gavrilova's papers: not read.
- The algorithmic details of the free-Lagrange reconnection in LNP 238/395 were not accessible.

## Q2. Out-of-field GPU 3D construction throughput (full cells), precision, robustness, code

### Takeaway
The fastest documented GPU 3D Voronoi/power constructors outside astrophysics are graphics/vision codes, mostly single precision. Taveira et al. 2026 ("paragram") reaches about 7–9 M points/s end-to-end on an RTX 5090 or H200. Basselin et al. 2021 reports 10 M points in 300 ms on a V100 for near-uniform data. None reaches 100 M cells/s, and none is in documented double precision with exact predicates.

### Cited Findings
**Taveira, Lindström, Fatemi, Hammarstrand & Kahl, "Scalable GPU Construction of 3D Voronoi and Power Diagrams" (arXiv 2605.06408, May 2026; Chalmers/Zenseact).**
- **Method:** per-cell clipping by bisector planes. Candidates are culled with directional bounds of the evolving cell plus a hierarchical best-first traversal of a bounding-volume hierarchy, so the method is robust to non-uniform and clustered data. It works for power diagrams.
- **Timings (end-to-end):**

  | Case | RTX 5090 | H200 |
  |---|---|---|
  | 1M, density gradient | 0.118 s | 0.132 s |
  | 10M, density gradient | 1.102 s | 1.084 s |
  | 10M, clustered (K=10) | 1.423 s | 1.848 s |
  | 15M, clustered (K=10) | – | 2.530 s |
  | 4.1M "Garden" (real data) | 0.496 s | 0.574 s |

- **Baselines (density gradient 1M / Garden):**

  | Baseline | Density gradient, 1M | Garden |
  |---|---|---|
  | gDel3D (double), RTX 5090 | 0.283 s (fails at ≥10M) | 1.168 s |
  | Geogram, 16 cores of an Epyc 9534 | 0.432 s | 1.732 s |
  | CGAL single-thread | 4.655 s | 19.205 s |

  A CUDA re-implementation of Basselin et al. 2021 failed on almost all real and clustered cases.
- Precision is "floating-point" without exact guarantees. The paper's appendix states that the Basselin re-implementation and Radiant Foam run in single precision, gDel3D in double. Code is at research.zenseact.com/publications/paragram. Replacing Radiant Foam's Voronoi step with this method scaled it to more than 20 M cells (5× the previous limit). — [arXiv HTML](https://arxiv.org/html/2605.06408); [abs](https://arxiv.org/abs/2605.06408); [Chalmers](https://research.chalmers.se/publication/553368)
- A follow-up, VoroTracing (arXiv 2608.17682, Aug 2026), renders differentiable Voronoi ray tracing at 623 FPS on an RTX 5090. Its update strategy was not reported in the abstract. — [arXiv 2608.17682](https://arxiv.org/abs/2608.17682)

**Basselin, Alonso, Ray, Sokolov, Lefebvre & Lévy, "Restricted Power Diagrams on the GPU" (Eurographics/CGF 40(2), 2021).** Reported as 10 million points in 300 ms on an NVIDIA V100 (about 33 M points/s). The number comes from a search snippet and was not verified in the paper. Precision is single, per Taveira et al.'s description of their re-implementation. — [EG diglib](https://diglib.eg.org/items/43d82919-34bf-4096-a2fd-5784207f8624/full); [PDF](https://diglib7.eg.org/bitstream/handle/10.1111/cgf142610/v40i2pp001-012.pdf)

**Ray, Sokolov, Lefebvre & Lévy, "Meshless Voronoi on the GPU" (ACM TOG 37(6), 2018).** Each cell is built independently from its k nearest neighbours (kd-tree) with a security-radius test. Integrals come from on-the-fly tetrahedral decomposition. It is reported as about 10× faster than the fastest multicore CPU codes, and code is in the supplemental material. It assumes near-uniform site density. — [HAL](https://hal.archives-ouvertes.fr/hal-01927559); [ACM](https://dl.acm.org/doi/abs/10.1145/3272127.3275092); [Taveira et al.](https://arxiv.org/html/2605.06408)

**Plateau–Holleville & Lévy, "More Power to the Particles" (arXiv 2601.05765, Jan 2026; partial-OT fluids, 3D restricted Laguerre cells).**
- Hardware: CUDA on an RTX A4000. Per-step seconds (CPU M1 Pro / GPU), averaged over 100 steps:

  | Case | Laguerre build (CPU / GPU) | Full step (CPU / GPU) |
  |---|---|---|
  | 10k sites | 0.662 / 0.335 | 1.921 / 0.750 |
  | 50k sites | 5.603 / 4.779 | 19.850 / 9.774 |

- The diagram is rebuilt inside the Newton iterations on the weights, with no topology reuse reported. The GPU path lacks exact predicates and is "less robust in some complex cases". Code is promised after publication. — [arXiv 2601.05765](https://arxiv.org/html/2601.05765)
- Background: de Goes et al. 2015 Power Particles is a CPU solver. — [PDF](https://geometry.caltech.edu/pubs/dGWH+15.pdf). Caplan et al. report about 29 s on CPU for 100k sites in 3D. — [arXiv 2106.14730](https://arxiv.org/pdf/2106.14730)

**Others (lower relevance).**
- Xiao, Cao & Chen, "Efficient Computation of Voronoi Diagrams Using Point-in-Cell Tests" (arXiv 2509.07175). Clipped/restricted Voronoi with an edge-based bisector search, where only edges with one endpoint inside and one outside trigger clipping. A GPU extension is claimed; no numbers on the abstract page. — [arXiv](https://arxiv.org/abs/2509.07175)
- Voro3D (Levet 2023, microscopy) is hybrid CPU/GPU: "millions of localizations in a few minutes". — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10436483/)
- "GPU Voronoi Diagrams for Random Moving Seeds" (arXiv 2209.00117) is a 2D raster jump-flooding variant, up to about 5.3× over JFA. Not applicable to exact 3D geometry. — [arXiv](https://arxiv.org/pdf/2209.00117)
- gDel3D / gFlip3D / gStar4D (Cao, Nanjappa et al.) are GPU flip plus star-splaying 3D Delaunay. Flipping alone stalls in 3D, so star splaying repairs the leftovers. They are static constructors, but the flip/splay machinery is the natural GPU local-repair primitive. — [gDel3D GitHub](https://github.com/ashwin/gDel3D); [ACM](https://dl.acm.org/doi/10.1145/2556700.2556710)
- The user's own voronoi-cpp README (CUDA) appeared in a search: 7.4 M particles (195³) with search 555.645 ms plus reconstruction 1196.984 ms, 1751.970 ms total, about 4.2 M cells/s. — [GitHub](https://github.com/Ulli1991/voronoi-cpp)

### Inferences
- Taveira et al.'s per-cell clipping is algorithmically the same family as Ray et al., peclet-voro and the user's code. Its contribution is robust candidate culling (directional bounds of the evolving cell plus BVH best-first traversal) for clustered data, which is directly transferable to cosmological clustering. At about 8–9 M pts/s (precision unclear, likely float) it is roughly 10× below the 100 M cells/s target.
- The Basselin figure (about 33 M pts/s, V100, single precision, near-uniform) is the highest reported per-cell GPU rate found. It comes from a uniform-density, float, minimal-output setting, so it is an upper envelope, not a double-precision hydro rate.
- No out-of-field source demonstrates double-precision 3D Voronoi on GPU above about 10 M cells/s. Reaching 100 M cells/s probably requires both topology-seeded candidates (to skip kNN/BVH search) and cheaper clipping arithmetic. That inference rests on the measured gaps above.

### Gaps
- Exact Ray et al. 2018 timings, GPU and precision were not retrieved. Basselin 2021's 300 ms / V100 figure was not checked against the paper.
- Taveira et al.'s own precision subsection was not read (the fetch was truncated).
- No GPU per-atom Voronoi for MD (OVITO, LAMMPS voronoi/atom, freud) was found. These appear to be CPU / voro++-based; freud relies on voro++, which is unverified. — [freud](https://arxiv.org/pdf/1906.06317); [voro++ multithreaded](https://arxiv.org/pdf/2209.11606)
- No GPU DTFE or ZOBOV/VIDE implementation was found. The public DTFE code is CPU-only. — [DTFE](https://www.arxiv.org/abs/1105.0370)
- Theses, GTC/SC posters, patents and Chinese/Japanese/Russian literature: none surfaced in these searches; not covered.
