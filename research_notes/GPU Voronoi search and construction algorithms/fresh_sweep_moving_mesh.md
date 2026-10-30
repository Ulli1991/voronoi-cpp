# GPU-accelerated moving-mesh / Voronoi-based simulation codes (2015 to Oct 2026), independent sweep

Scope note: this is a fresh sweep of about 20 tool calls (web search and fetch), done on 2026-10-29. Several PDFs (Zhai et al. TVCG 2020; Ray et al. 2018 on HAL) could not be text-extracted or were blocked by bot protection, so their numbers are marked as gaps. ProteusGPU appears only with its README-stated facts. Its algorithms were not analysed, as instructed.

## Which simulation codes build Voronoi/Delaunay meshes on GPUs today, and how fast?

### Takeaway
Very few time-stepping codes build Voronoi or Delaunay meshes on the GPU, and almost none publish per-cell mesh-build rates. The verified GPU-resident examples are:
- ProteusGPU (astro moving-mesh hydro, GH200)
- peclet-voro (Kokkos, dynamic 3D Voronoi for CFD)
- cellGPU (2D active-matter Voronoi model; the triangulation repair runs on the CPU)
- graphics power-particle and optimal-transport fluid solvers (Zhai et al. 2020; Plateau Holleville and Lévy 2026)

The best-documented GPU construction speeds come from static-point builders: Taveira et al. (SIGGRAPH 2026) and gDel3D. Neither has been used inside a hydro time loop. Production astro moving-mesh codes (AREPO, RICH, TESS, ChaNGa-MM) still build the mesh on CPUs.

### Cited Findings
**ProteusGPU** (github.com/lucas56098/ProteusGPU), README-stated only:
- It offers "2D/3D static/moving mesh generation on CPU and GPU" and is "optimized for NVIDIA's GH200 superchips". The hydro solver is "similar to" AREPO (Springel 2010), and mesh generation is credited to Ray et al. 2018 — [ProteusGPU README](https://github.com/lucas56098/ProteusGPU)
- The README reports "Weak scaling on JUPITER booster for an acoustic wave test with up to 3600^3 and 1000 GH200 GPUs retains 75% efficiency" — [ProteusGPU README](https://github.com/lucas56098/ProteusGPU)
- Per-step time ("Per step time for a 454^3 acoustic wave test on two GH200 chips") and a per-step breakdown appear only as figures. No numeric values or precision are given in the README text — [ProteusGPU README](https://github.com/lucas56098/ProteusGPU)

**peclet-voro** (computational-chemical-engineering, Kokkos; CUDA/HIP/OpenMP; dated entries up to 2026-09-03; Zenodo DOI 10.5281/zenodo.21132443; no paper):
- Described as a "Dynamic 3D Voronoi tessellation of moving particles (Kokkos GPU)" for compressible and incompressible CFD with surface tension — [GitHub](https://github.com/computational-chemical-engineering/peclet-voro); [PyPI](https://pypi.org/project/peclet-voro-cu13/)
- The README lists no cells/s, ms per step or GPU model. It points to `docs/performance_report.md`, which was not retrieved — [GitHub](https://github.com/computational-chemical-engineering/peclet-voro)

**cellGPU** (Sussman, CPC 2017, arXiv:1702.02939):
- GPU code for the 2D self-propelled Voronoi (SPV) model and the active vertex model (AVM) — [arXiv](https://ar5iv.labs.arxiv.org/html/1702.02939)
- Claimed speed: SPV is "as much as three orders of magnitude faster than existing CPU-based implementations for large system sizes"; AVM is "more than an order of magnitude faster than its own CPU-based implementation" — [arXiv](https://ar5iv.labs.arxiv.org/html/1702.02939)
- Hardware: Tesla K40, Quadro K620 and GeForce 980Ti with a Xeon E5-1620 v3, plus a GeForce 950M laptop. Benchmarks are double precision, with one 980Ti curve in float. Time per step against N is shown only in figures — [arXiv](https://ar5iv.labs.arxiv.org/html/1702.02939)
- A topological-flocking (Vicsek-type) code reuses cellGPU's GPU Voronoi computation — [cellGPU repo / search summary](https://github.com/sussmanLab/cellGPU)

**Zhai, Hou, Qin, Hao, "Fluid simulation with adaptive staggered power particles on GPUs"** (IEEE TVCG 26(6):2234–2246, 2020):
- The whole method runs on GPUs. A "facet-based power diagram construction algorithm suitable for parallelization" gives about an order-of-magnitude speedup over "the existing code library" (abstract-level claim; tables not retrieved) — [BUAA record](https://research.buaa.edu.cn/en/publications/fluid-simulation-with-adaptive-staggered-power-particles-on-gpus/); [Stony Brook record](https://researchconnect.stonybrook.edu/en/publications/fluid-simulation-with-adaptive-staggered-power-particles-on-gpus/)

**Plateau Holleville and Lévy, "More Power to the Particles"** (arXiv:2601.05765, Jan 2026):
- The optimal-transport solver, fluid simulation and rendering are "implemented on the GPU". Laguerre cells are built analytically from intersections with spheres rather than by convex clipping — [arXiv](https://arxiv.org/abs/2601.05765)
- Reported numbers: 162 ms on average versus 177 ms for the previous method, "computed with 500 samples of a simulation composed of 1000 sites". The simulation ran on an Apple M1 Pro; rendering used an RTX A4000 at 9 ms for 100,000 sites. The paper does not say whether 162 ms is per step — [arXiv HTML](https://arxiv.org/html/2601.05765v1)

**Lévy, JCP 451 (2022), "Partial optimal transport for a constant-volume Lagrangian mesh with free boundaries"** (CPU baseline in the same family):
- About 98 s per timestep for 500,000 cells, 12 s for 25,000 cells and 2.8 s for 5,000 cells, on an Intel Core i9-9880H, multithreaded (CPU, not GPU) — [arXiv 2106.03936](https://arxiv.org/pdf/2106.03936)

**Power Plastics** (Qu, Li, Yang, Jiang, de Goes; SIGGRAPH Asia 2023):
- A hybrid Lagrangian/Eulerian method built on the Power Particle-In-Cell representation. No GPU implementation details were found — [SIGGRAPH history](https://history.siggraph.org/?p=214400)

**Static or standalone GPU builders** (not used in a hydro time loop):
- Taveira et al., "Scalable GPU Construction of 3D Voronoi and Power Diagrams" (SIGGRAPH 2026 / TOG, arXiv:2605.06408), end-to-end Voronoi times:
  - 4.2M points ("bicycle"): 0.570 s on RTX 5090 and 0.684 s on H200; gDel3D 1.204 s / 0.893 s; Geogram (CPU) 1.593 s; CGAL 20.445 s
  - 1.0M density-gradient points: 0.118 s on RTX 5090
  - 10-cluster 1.0M points: 0.386 s (5090) versus gDel3D 0.284 s; gDel3D fails above 5M points on the 5090
  - Power diagram for "bicycle": 1.134 s on 5090
  - Scales to more than 20M cells
  - Precision of the method itself is not stated. gDel3D runs double by default, and float made "negligible" difference (median < 0.5%) — [arXiv HTML](https://arxiv.org/html/2605.06408); [Chalmers](https://research.chalmers.se/en/publication/553368)
- votess (Singh, Byrohl, Nelson; arXiv:2412.04514; SYCL; cell-by-cell clipping, outputs cell geometry and neighbour lists):
  - Its authors say no general-purpose public GPU 3D Voronoi code existed before it
  - The CPU version reportedly beats other implementations by 10–100x
  - It is aimed at "on-the-fly computations in hydrodynamical codes", but the abstract shows no use inside a time loop — [arXiv](https://arxiv.org/abs/2412.04514); [search summary](https://arxiv.org/html/2412.04514v2)
- Radiant Foam (arXiv:2502.01157): a differentiable Voronoi ray tracer whose sites move under gradient descent, so the Delaunay mesh is rebuilt during training — [arXiv](https://arxiv.org/pdf/2502.01157)

**CPU-only astro moving-mesh codes** (for context):
- AREPO-RT GPU port (Zier et al., arXiv:2404.17630): moved "all RT-related calculations to GPUs" for about 15x on RT benchmarks and 3x overall in a reionization run. The abstract does not say the mesh moved to the GPU — [arXiv](https://arxiv.org/abs/2404.17630)
- RICH/STORM (arXiv:2610.25174, Oct 2026): a 3D Voronoi moving mesh with radiation; scaling quoted on 13,440 CPU cores — [arXiv](https://arxiv.org/abs/2610.25174)
- ChaNGa moving-mesh paper: Voro++ computes about 100K 3D Voronoi cells/s on a single core — [MNRAS 471, 3577](https://academic.oup.com/mnras/article/471/3/3577/3979023)

### Inferences
- In astrophysics, ProteusGPU appears to be the only public code doing full 3D moving-mesh hydro with GPU mesh generation at scale. AREPO's GPU work (as far as found) covers radiative transfer, not the mesh.
- The fastest verified 3D GPU build rate is about 7–8.5M cells/s (4.2M points in 0.570 s, and 1M in 0.118 s, on an RTX 5090, end-to-end including allocation). This is for a static point set, so it is an upper-bound reference, not a per-step rate.
- Graphics power-particle solvers run small N (10^3–10^5) and do not report Mcells/s.

### Gaps
- ProteusGPU per-step times and precision: the values exist only in README figures.
- peclet-voro throughput: `docs/performance_report.md` was not retrieved.
- Zhai et al. 2020 tables: the PDF could not be text-extracted.
- Ray et al. 2018 "Meshless Voronoi on the GPU": the HAL page was blocked, so its numbers (cited elsewhere as a power-diagram fluid application) are unverified here.
- votess GPU timings are not in the abstract.
- No GPU numbers were found for Power Plastics.

## How do they handle topology changes between steps on GPUs?

### Takeaway
Three strategies appear:
1. **Full rebuild every step or iteration** by cell-wise clipping: Radiant Foam, Taveira et al., and implied for clipping-based fluid solvers.
2. **Check and locally repair**: cellGPU tests the Delaunay empty circumcircle on the GPU and repairs flagged cells on a single CPU thread with CGAL.
3. **Certificate or Verlet-skin incremental repair, fully on the GPU**: peclet-voro.

No source found reports a GPU-resident local Delaunay flip-repair inside a 3D hydro code with published timings.

### Cited Findings
- **cellGPU (SPV)**: after positions move, a GPU kernel tests the empty-circumcircle property. "For each of these flagged cells, the triangulation is locally repaired". Candidate 1-rings come from the CPU, and CGAL reduces them to the true neighbour set. The repair is "strictly single-threaded", after which data is copied back to the GPU. The AVM part does T1 transitions fully on the GPU, with "only a single topology update per cell per kernel call" — [arXiv 1702.02939](https://ar5iv.labs.arxiv.org/html/1702.02939)
- **peclet-voro**:
  - "Incremental cell updates under particle motion (persistent Verlet-skin worklist)"
  - `repair.hpp`, "MovingTessellation: incremental two-pass repair update"
  - "the certificate re-tests those planes each step"
  - Repair is "exact to 1e-11 (default tolerance) / 1e-15 (tight) over 400 steps" at "~80 % of the previous speedup"
  - Neighbour discovery uses a counting-sort grid plus a presorted worklist and a Z-order index
  - Python arrays are float64 — [GitHub](https://github.com/computational-chemical-engineering/peclet-voro)
- **Taveira et al. 2026**: the method "targets single-GPU execution and static point sets"; incremental updates are listed as future work. In the Radiant Foam experiment they "benchmark full reconstruction" every time — [arXiv HTML](https://arxiv.org/html/2605.06408)
- **Radiant Foam**: site optimisation by gradient descent "demands the mesh to be recomputed exhaustively". A Dec 2025 follow-up notes that moving Delaunay vertices causes topological discontinuities (edge flips) — [search summary of arXiv 2605.06408](https://arxiv.org/pdf/2605.06408); [arXiv 2512.04076](https://arxiv.org/pdf/2512.04076v1)
- **ProteusGPU**: the README mentions a "two-tier mesh generation" with "fast and slow-tier" parts. Not analysed further, per instruction — [README](https://github.com/lucas56098/ProteusGPU)
- **CPU Lagrangian Voronoi schemes for context**: some regenerate the mesh "anew from Voronoi seeds ... at every time step" ("Lagrangian Voronoi meshes and particle dynamics with shocks") — [ResearchGate](https://www.researchgate.net/publication/377036562_Lagrangian_Voronoi_meshes_and_particle_dynamics_with_shocks). Others use direct ALE schemes on moving Voronoi meshes with topology changes (Gaburro et al.) — [ResearchGate](https://www.researchgate.net/publication/332873476_High_order_direct_Arbitrary-Lagrangian-Eulerian_schemes_on_moving_Voronoi_meshes_with_topology_changes)

### Inferences
- Cell-wise clipping (the Ray et al. / votess / Taveira family) makes a full rebuild every step natural on GPUs, because no global Delaunay structure has to be kept consistent.
- Incremental approaches reuse the previous step's neighbour candidates and re-validate them through a certificate or skin, instead of doing Delaunay flips.
- Flip-based repair is still CPU-bound in the only documented case (cellGPU).

### Gaps
- No quantitative comparison of rebuild cost against repair cost was found for any GPU code. peclet-voro's "~80 %" figure is relative and has no absolute baseline.

## What fraction of a step is mesh construction, and what full-step cell-update rates are reported?

### Takeaway
None of the GPU codes found publish a numeric fraction of step time spent in mesh construction, or full-step cell updates per second, in text. ProteusGPU shows them only as figures. The only numeric breakdown found is a CPU code: in Lévy 2022, Laguerre diagram construction was about 9.7% of the step, while Hessian assembly and gradient evaluation dominated. Those percentages come from a garbled extraction and are unverified.

### Cited Findings
- **ProteusGPU**: a "Breakdown of where the per step time is spent" figure and a per-step-time figure (454^3 on two GH200), with no numbers in the text. Weak scaling is 75% at 1000 GH200 GPUs up to 3600^3 cells — [README](https://github.com/lucas56098/ProteusGPU)
- **Lévy 2022 (CPU, i9-9880H)**: crown splash with 500k cells takes about 98 s per step. Hessian assembly is about 41.5%, the gradient about 32.8% and the Laguerre step about 9.7% (unverified extraction) — [arXiv 2106.03936](https://arxiv.org/pdf/2106.03936)
- **AREPO-RT on GPU**: the overall gain falls to 3x from 15x on RT benchmarks, because non-RT parts (including CPU mesh work) then dominate. No numeric mesh fraction is reported in the abstract — [arXiv 2404.17630](https://arxiv.org/abs/2404.17630)
- **cellGPU**: no triangulation-time fraction is reported. The timing data are in figures — [arXiv 1702.02939](https://ar5iv.labs.arxiv.org/html/1702.02939)
- **Plateau Holleville and Lévy 2026**: 162 ms for a 1000-site simulation on an M1 Pro, with no split between construction and solve — [arXiv HTML](https://arxiv.org/html/2601.05765v1)

### Inferences
- Without published fractions, the static-build rate of about 7–8.5M cells/s on an RTX 5090 (Taveira et al.) is the best verified anchor for GPU 3D Voronoi build cost. A GPU moving-mesh step that includes a full rebuild would be bounded by roughly that rate plus the hydro work.
- The AREPO-RT experience (RT 15x faster but only 3x overall) suggests that once physics moves to the GPU, a CPU-resident mesh becomes the limiter. This is indirect evidence for moving mesh construction to the GPU.

### Gaps
No numeric values were found for:
- full-step cells/s for any GPU moving-mesh code
- the mesh-construction fraction on any GPU code
- strong-scaling data for GPU Voronoi construction inside a simulation

Retrieving ProteusGPU's figure data or an associated paper, peclet-voro's performance_report.md, and the Zhai et al. 2020 tables would close these.
