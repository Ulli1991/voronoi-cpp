# CFD / applied-math / computational-geometry literature on moving Voronoi meshes (to Oct 2026)

Scope note: research was budget-limited (about 20 tool calls). Several primary PDFs could not be fetched (Stanford geometry server refused the connection, Eurographics diglib returned 403, HAL was behind a bot wall, the LANL PDF came back empty). Items marked **[UNVERIFIED, from background knowledge]** were not confirmed against a fetched source in this session. The report writer should treat them as leads, not facts.

## Free-Lagrange methods (Crowley, Fritts, Trease X3D, Ball): mesh reconnection strategies and cost

### Takeaway
Free-Lagrange (FL) codes of the 1980s and 1990s were the first CFD schemes to use Voronoi cells of moving generators, with connectivity taken from the Delaunay/Voronoi structure. Trease carried the method from 2D to 3D at Los Alamos, and later work concentrated on making 3D reconnection cheaper. I found no quantitative per-step cost figures in a source I could read.

### Cited Findings
- Trease, "Three-Dimensional, Free-Lagrange Hydrodynamics" (Los Alamos report, 1987), states that the 2D free-Lagrange method was pioneered by Crowley and Fritts and describes a 3D extension from LANL's X-7 division — [LANL report LA-UR / fas.org mirror](https://sgp.fas.org/othergov/doe/lanl/lib-www/la-pubs/00228697.pdf) (summary via search snippet; the PDF text was empty when fetched).
- A NASA abstract on 3D grid restructuring for Free-Lagrange calculations describes algorithms that "reduce the cost of three-dimensional Free-Lagrange calculations". The same algorithms can efficiently determine Voronoi connections for initial grids and handle colliding surfaces — [NTRS 19870024133](https://ntrs.nasa.gov/citations/19870024133).
- The main collections are the proceedings *The Free-Lagrange Method* (Hilton Head 1985, eds. Fritts, Crowley, Trease; Springer LNP 238) and *Advances in the Free-Lagrange Method* (1991; covers adaptive gridding and SPH) — [Springer BFb0032238](https://link.springer.com/10.1007/BFb0032238); [Springer 3-540-54960-9](https://link.springer.com/book/10.1007/3-540-54960-9).
- In the vortex-method setting, one review says that building the Voronoi diagram of the vortices costs "of the same order as the convection of the vortices using fast algorithms" — [FSU thesis section 1.3.6](https://web1.eng.famu.fsu.edu/~dommelen/papers/subram/style_a/node11.html).

### Inferences
- From the NASA abstract, 3D reconnection was the known cost bottleneck of FL codes, and the work went into restructuring the existing grid (local reconnection) rather than rebuilding it globally.

### Gaps
- I found no source on G. Ball's "FLAG" in the free-Lagrange context. FLAG is also the name of a LANL ALE code (Burton), so the attribution in the brief may be mixed up. Not verified.
- I found no explicit per-step reconnection cost for X3D. The algorithmic details of Trease's 3D reconnection (local Delaunay flips versus re-tessellation) are not confirmed here. **[UNVERIFIED, from background knowledge]:** FL codes typically repaired Delaunay connectivity locally after each step by checking the empty-circumsphere condition for neighbours-of-neighbours.

## Recent (2010–2026) Lagrangian / ALE Voronoi CFD methods: mesh construction, topology changes, cost

### Takeaway
Modern moving-Voronoi CFD schemes fall into three groups by how they build the mesh:
- **Global Delaunay rebuilt every step**, in the AREPO style (Springel; Gaburro/Dumbser ALE).
- **Partial or local reconstruction** (Shadowfax rebuilds the relevant parts of the mesh; MANGA uses Voro++ per cell).
- **Direct per-cell half-plane clipping with a cell list, warm-started from the previous step's neighbours**, with no Delaunay dual (SILVA, Kincl/Peshkov/Boscheri 2024).

None of the sources I read gives a quantitative comparison of full rebuild versus incremental update inside a CFD solver.

### Cited Findings
**Gaburro, Boscheri, Chiocchetti, Klingenberg, Springel, Dumbser — "High order direct ALE schemes on moving Voronoi meshes with topology changes", J. Comput. Phys. 407 (2020) 109167, arXiv:1905.00967**
- Generators move with the local fluid velocity, and the mesh is regenerated at every time step with an AREPO-type approach that "rapidly rebuilds a new high quality mesh rearranging the element shapes and neighbors" — [arXiv abs](https://arxiv.org/abs/1905.00967v2).
- Old and new cells that share a generator are joined into closed space-time control volumes. Their bottom and top faces can be polygons with different numbers of sides. Degenerate space-time **sliver elements** fill the "space–time holes that arise because of topology changes" — [arXiv abs](https://arxiv.org/abs/1905.00967v2); [arXiv PDF](https://arxiv.org/pdf/1905.00967).
- The scheme is conservative and satisfies the geometric conservation law (GCL) by construction, through integration over closed space-time control volumes. It is 2D and high-order (FV and DG), and the authors report "substantial improvements" over direct ALE on conforming meshes without topology change — [arXiv abs](https://arxiv.org/abs/1905.00967v2).
- The abstract gives no runtime or cost figures for mesh generation. The full PDF exceeded the fetch size limit, so I could not check it — [arXiv abs](https://arxiv.org/abs/1905.00967v2).
- A later unified review by Gaburro covers this family, together with a local non-conforming approach for sliding lines — [IRIS UniVR "Unified ACME"](https://iris.univr.it/retrieve/8622e869-3922-486f-aff3-6bf765f5ba66/13_Unified_ACME.pdf); also [IRIS "Arepo" item](https://iris.univr.it/retrieve/06dbc3ae-618a-42cd-9a74-7d10bdac8f4a/8_Arepo.pdf).

**SILVA — Kincl, Peshkov, Boscheri, "Semi-Implicit Lagrangian Voronoi Approximation" (incompressible Euler/NS), Int. J. Numer. Methods Fluids 97(1), arXiv:2405.04116; compressible extension arXiv:2410.14564**
- Particles carry the solution and act as Voronoi generators. A semi-implicit step projects the velocity onto a divergence-free field. The authors say the scheme needs no remapping stage, unlike reconnection-based ALE — [arXiv abs](https://arxiv.org/abs/2405.04116v1).
- **Mesh construction is local and per-cell, with no Delaunay dual.** In the authors' words, most generators "rely on constructing the Delaunay triangulation," but they "found that a direct approach [46, 35] is preferable, where no dual mesh is constructed" — [arXiv HTML](https://arxiv.org/html/2405.04116).
- The algorithm builds a cell list with partitions of side 2δr. For each seed it visits partitions in order of distance and cuts the cell with one half-plane per neighbouring seed. It stops when R_i^k < ½ dist(P_{i,0}, P_{i,k+1}), since the remaining seeds are "provably too remote". This is a security-radius criterion — [arXiv HTML](https://arxiv.org/html/2405.04116).
- **Topology reuse via warm start:** "Starting from the second time step, the process of mesh generation is further accelerated by initializing" each cell with the previous step's neighbour set (their Eq. 17) — [arXiv HTML](https://arxiv.org/html/2405.04116).
- The authors claim "O(N) complexity is guaranteed" if diam(ω_i/δr) is bounded, and that the method is "easily parallelized" because "cell constructions are independent processes". It works in 2D and 3D. No timings or hardware are reported — [arXiv HTML](https://arxiv.org/html/2405.04116).
- The compressible follow-up adds an auxiliary elliptic pressure equation for low-Mach flows and is validated across Mach regimes, shocks and multiphase flows — [arXiv 2410.14564](https://arxiv.org/html/2410.14564v1).

**Other Lagrangian Voronoi / particle-Voronoi methods**
- Heß and Springel (2010), "Particle hydrodynamics with tessellation techniques" (MNRAS 406:2289–2311), introduce Voronoi Particle Hydrodynamics (VPH). Density comes from the Voronoi tessellation of the particles instead of an SPH kernel. It gives results comparable to SPH in shocks and better ones in turbulent regimes, and adds correction forces that keep the mesh regular — [arXiv 0912.0629](https://ar5iv.labs.arxiv.org/html/0912.0629).
- Duque (2022/23), "A unified derivation of Voronoi, power, and finite-element Lagrangian CFD" (Eur. J. Mech. B/Fluids, DOI 10.1016/j.euromechflu.2022.12.009):
  - Zeroth-order consistency gives the Voronoi diagram. Adding an incompressibility pressure recovers Gallouët–Mérigot's power-diagram method.
  - First-order consistency gives the particle FEM (pFEM), plus a spring-like term.
  - Tested on the Gresho vortex and the Taylor–Green vortex — [arXiv 2212.12323](https://arxiv.org/abs/2212.12323).
- A Lagrangian power-diagram method for the spherical shallow-water equations (arXiv:2508.08129, 2025) belongs to the moving power/Voronoi line for geophysical flows — [arXiv 2508.08129](https://arxiv.org/pdf/2508.08129). Details were not fetched.
- Fu, Hu and Adams' "centroidal Voronoi particle" (CVP) work (JCP 335:718–735, 2017) and the Lagrangian Inertial CVP (Ji, Fu, Hu, Adams, Comput. Phys. Commun. 239:53–63, 2019) use Voronoi particles for **domain decomposition / dynamic load balancing**, not as flux-carrying FV cells — [arXiv 1802.04539](https://arxiv.org/pdf/1802.04539).
- Després has a first-order scheme on Lagrangian and moving Voronoi meshes for compressible flows with shocks — [HAL Sorbonne hal-04166811](https://hal.sorbonne-universite.fr/hal-04166811/document) (seen in a search snippet only).
- Free-ALE (University of Auckland) is a collocated Voronoi FV Navier–Stokes solver with Rhie–Chow interpolation — [Auckland ResearchSpace](https://researchspace.auckland.ac.nz/handle/2292/52299).

**Astrophysics moving-mesh codes (reference implementations adopted by CFD)**
- AREPO builds its Voronoi mesh from the dual Delaunay triangulation, using Bowyer–Watson incremental insertion extended to a point set distributed over many cores (Springel 2010). The worst case is about N², but random insertion order gives N log N in practice — [Subsweep arXiv 2412.00950](https://arxiv.org/pdf/2412.00950); [Springel 2010 arXiv 0901.4107](https://arxiv.org/pdf/0901.4107).
- Shadowfax reconstructs the relevant parts of the mesh at every step — [arXiv 1605.03576](https://arxiv.org/pdf/1605.03576).
- MANGA, the ChaNGa moving-mesh solver, computes cells with Voro++ (per-cell) and reports about 10K cells per second per core — [arXiv 1707.05333](https://arxiv.org/pdf/1707.05333) (via search snippet; hardware not confirmed).
- A GPU Delaunay project page states that in recent cosmological runs the mesh was the most expensive CPU component. This is an informal, secondary source — [ArepoCUDA](https://nelson.tng-project.org/ArepoCUDA).
- Other moving-Voronoi codes: RICH (open source, 2D) — [ApJS 216:35](https://authors.library.caltech.edu/56525/1/0067-0049_216_2_35.pdf); TESS (relativistic) — [arXiv 1104.3562](https://arxiv.org/pdf/1104.3562).

### Inferences
- Across CFD and astrophysics, the dominant practice is to **rebuild the tessellation every step** (globally through Delaunay, or locally per cell) rather than repair topology incrementally. Topology changes are then handled in the flux formulation (Gaburro's space-time slivers) or simply ignored at the level of a single step (Godunov-type Voronoi FV). SILVA's warm start from the previous neighbour list is the clearest published example of **topology reuse inside a per-cell (non-Delaunay) construction**, and it matches the "local per-cell + security radius" design.

### Gaps
- I found no paper titled "Lagrangian Voronoi Meshes (LVM)" by Fu or Sun. The term may refer to SILVA or Després. Not verified.
- Gaburro et al. 2020: mesh-generation timing and fraction of runtime not obtained (PDF too large to fetch). Later Gaburro/Dumbser work on 3D or space-time-DG Voronoi with topology changes (2021–2023) was not checked.
- No AREPO mesh-construction share of runtime was confirmed from a primary source. **[UNVERIFIED, from background knowledge]:** Springel 2010 (MNRAS 401:791) reports that Voronoi mesh construction takes a sizeable but not dominant fraction of the time step in 3D. The exact numbers need checking.
- No ocean or atmosphere moving-Voronoi model (beyond the power-diagram shallow-water paper) was found within budget.

## Kinetic and incremental Delaunay maintenance for moving points: performance versus rebuild

### Takeaway
There are three strategies: kinetic data structures (KDS, which need known trajectories), delete-and-reinsert relocation, and filtered or "almost-Delaunay" updates. For small displacements from a physics integrator, flip-based repair is reported as up to about 20× faster than rebuilding (Schaller and Meyer-Hermann, 3D). The CGAL-group studies find that rebuilding from scratch is "surprisingly" competitive unless points move only slightly or only a few points move.

### Cited Findings
- **Schaller & Meyer-Hermann, "Kinetic and Dynamic Delaunay tetrahedralizations in three dimensions" (arXiv:physics/0302018, 2003; later Comput. Phys. Commun. [UNVERIFIED venue])** — [ar5iv](https://ar5iv.arxiv.org/html/physics/0302018):
  - Each vertex gets a maximum step size from the signed volumes of its incident tetrahedra, so that no orientation flips sign. After each step, Delaunay is restored with 2→3 and 3→2 flips only; 1→4 and 4→1 flips are not needed for movement.
  - "In the ideal case, this update method is about 20 times as fast as computing a new triangulation." The step-size check roughly doubles the restoration time.
  - Hardware: 1.533 GHz AMD Athlon, 1 GB RAM, g++ 3.3.
  - Mixed insert/delete/move test: about 0.26 s per step at 20,000 points and about 3.21 s per step at 200,000 points.
  - Flip totals over 100 steps (excluding deletion flips): 426 at 20k points and 9,766 at 200k points.
  - Robustness: very flat simplices give tiny step bounds and possible rounding errors. Exact arithmetic is suggested but was never needed in their tests. Unperturbed cubic lattices show quadratic insertion cost.
- **Guibas & Russel, "An empirical comparison of techniques for updating Delaunay triangulations" (SoCG 2004, DOI 10.1145/997817.997846):**
  - Motivated by small displacements from physics integrators (3D molecular simulation).
  - The authors implemented many update algorithms, most of them new, timed them on a corpus of data sets, and concluded that "updating can be significantly faster than recomputing".
  - Sources: [Springer Professional abstract](https://www.springerprofessional.de/doi/10.1145/997817.997846); [PDF](https://geometry.stanford.edu/lgl_2024/papers/gr-ectudt-04/gr-ectudt-04.pdf) (could not be fetched, so no exact numbers).
- **Survey: Manhães de Castro, Devillers et al., "State of the Art: Updating Delaunay Triangulations for Moving Points" (INRIA report, about 2008):** covers KDS, delete-and-reinsert relocation, and "almost-Delaunay" structures that postpone repairs — [HAL inria-00325816](https://hal.uca.fr/INRIA/inria-00325816).
- **Manhães de Castro, Tournois, Alliez, Devillers, "Filtering relocations on a Delaunay triangulation", Computer Graphics Forum 28(5), 2009 (SGP), DOI 10.1111/j.1467-8659.2009.01523.x:**
  - Introduces per-vertex **tolerances**, a safe-region certificate. While a vertex moves less than its tolerance, no combinatorial change is needed.
  - Filtered relocation beats rebuilding only under certain conditions. Rebuilding loses its edge when all points move slightly or only a few points move. The group also calls rebuilding from scratch "surprisingly a very viable option".
  - Sources: [HAL inria-00413344](https://hal-univ-tlse3.archives-ouvertes.fr/INRIA/inria-00413344); [Eurographics diglib](https://diglib.eg.org/items/4658d391-2b48-4b3a-b6ad-ab03ea4fea60/full).
- **Companion paper, "Fast Delaunay triangulation for converging point relocation sequences":** in 2D Lloyd iterations it runs "several times faster" than rebuilding — [HAL inria-00413351](https://hal-univ-tlse3.archives-ouvertes.fr/INRIA/inria-00413351v1).
- **Zhou et al., "Fast updating of Delaunay triangulation of moving points by bi-cell filtering", Computer Graphics Forum 29(7):2233–2242 (2010):**
  - Identifies bi-cells whose connectivity survives a perturbation, in 2D and 3D.
  - A 2011 follow-up (J. CAD & CG) fixes flipped bi-cells locally and reports 20–30% speedups when rebuilding is frequent.
  - Source: [Eurographics diglib](https://diglib.eg.org/handle/10.2312/CGF.v29i7pp2233-2242).
- **Theory:**
  - For points in continuous Delaunay motion, each co-circularity event is repaired by a single edge flip. Rubin (FOCS 2013) proves a nearly tight O(n^{2+ε}) bound on the number of discrete changes for unit-speed linear motion in 2D — [Rubin FOCS 2013](https://ieee-focs.org/FOCS-2013-Papers/5135a519.pdf).
  - KDS assumes the motions are known in advance, which does not hold for physics simulation (de Berg, Roeloffzen, Speckmann) — [JoCG 2012](https://doaj.org/article/27e2a48d0c2a4e3da7cddb46f8a6b7cf).
  - Kinetic meshing of moving points: [Acar, Hudson, Türkoğlu et al., SoCG 2011](https://condor.depaul.edu/dturkogl/research/kinetic-mesh-socg11.pdf) (not fetched).
- In mesh updating, point removal is the most expensive operation and flips the cheapest — [Mitchell UNM course notes](https://www.sandia.gov/-samitch/scott-a-mitchell/unm-math-579/).

### Inferences
- The published evidence on full rebuild versus update is mixed and depends on the displacement size. Filtered or flip-based updates help most when displacements are small relative to local spacing, which is the CFD CFL regime: Courant numbers below 1 imply sub-cell motion. Rebuilds stay competitive when every point moves by an O(spacing) amount. All of these numbers are CPU-era (2003–2011). I found no GPU comparison.

### Gaps
- No exact speedup tables were obtained from Guibas & Russel 2004, Manhães de Castro et al. 2009 or Zhou et al. 2010, because the PDFs were inaccessible.
- **[UNVERIFIED, from background knowledge]:**
  - Guibas, Mitchell and Roos (1991) and Albers, Guibas, Mitchell and Roos (1998) give the classical kinetic Delaunay/Voronoi bounds.
  - Acar et al. treat kinetic convex hulls via self-adjusting computation.
  - CGAL provides `Triangulation_3::move()` / `move_if_no_collision()` (relocation by local remove-and-insert).
  - None of these were checked in this session.
- No explicit study was found that compares per-step full reconstruction with incremental topology update **inside a moving-Voronoi CFD solver**. This appears to be a genuine gap in the literature, since the closest evidence comes from computational-geometry benchmarks and physics-simulation-motivated geometry papers.

## Neighbour-count bounds, local-cell certificates and robustness

### Takeaway
A 3D Poisson-Voronoi cell has on average 48π²/35 + 2 ≈ 15.535 faces, with the mode at 15. There is no theoretical maximum, but sampled maxima grow with sample size: 36 in older studies and 41 in a 10⁶-cell simulation. A per-cell construction can be certified locally with a security-radius stopping rule (SILVA uses one). For robustness, the standard tools are exact or filtered predicates and symbolic perturbation.

### Cited Findings
- Meijering (1953): the mean face count is c_F = 48π²/35 + 2 = 15.535. The distribution peaks at n = 15 in 3D and n = 6 in 2D, is roughly symmetric, and has no cells with fewer than 4 faces. Kumar et al. and Tanemura saw at most 36 faces, while Lazar et al. found cells with up to 41 faces in a 1-million-cell sample. Kumar's discrete Gamma fit has a = 21.6292 and b = 0.7199 — [Lazar, Mason, MacPherson, Srolovitz, arXiv 1401.1736](https://arxiv.org/pdf/1401.1736).
- Many-faced cells have been analysed asymptotically (heuristic theory for large n in d dimensions) — [Hilhorst, arXiv 0906.4449](https://arxiv.org/pdf/0906.4449). The distribution of n-edged faces in 3D is treated in [arXiv 1601.06252](https://arxiv.org/pdf/1601.06252).
- SILVA's per-cell security criterion (stop when the cell radius is below half the distance to the next-nearest unprocessed seed) is a published certificate of local cell correctness used in a CFD code — [arXiv HTML 2405.04116](https://arxiv.org/html/2405.04116).
- Schaller and Meyer-Hermann bound vertex step sizes so that no simplex inverts. This is an orientation-based certificate that keeps the topology valid between flip repairs — [ar5iv physics/0302018](https://ar5iv.arxiv.org/html/physics/0302018).

### Inferences
- For per-cell construction of moving Voronoi cells, the face-count statistics suggest sizing per-cell buffers for about 15–16 faces on average with a long tail (above 40 for N ≥ 10⁶ random points). Real particle distributions can also contain non-Poisson clustering that makes the tail worse.

### Gaps
- **[UNVERIFIED, from background knowledge]:**
  - Shewchuk (1997) adaptive exact predicates.
  - Edelsbrunner and Mücke (1990) Simulation of Simplicity.
  - Devillers and Teillaud's symbolic perturbation for 3D Delaunay in CGAL.
  - Lévy's geogram / "Meshless Voronoi on the GPU" (Ray, Sokolov, Lefebvre, Lévy, ACM TOG 2018), which uses the security radius for per-cell GPU construction.
  - Rycroft's Voro++ (Chaos 2009), per-cell construction.
  - None of these were checked here; they likely belong to other researchers' scope.
- I found no paper that analyses the **consistency of independently computed neighbour cells** (whether i lists j as a neighbour exactly when j lists i) in floating point for CFD use. **[UNVERIFIED]:** Ray et al. 2018 discuss this issue for GPU per-cell Voronoi.
