# Mesh refinement and derefinement in moving-mesh / Lagrangian Voronoi hydro codes (adding and removing generators, conservative remap)

Scope note: per the coordinator, depth is on (1) AREPO, (2) GIZMO MFM/MFV split/merge, (3) ALE/ReALE and remap literature (incl. r3d). Other codes are brief. Where I read the public AREPO source (GPL, gitlab.mpcdf.mpg.de/vrs/arepo, files `src/mesh/criterion_refinement.c`, `criterion_derefinement.c`, `refinement.c`, `voronoi/voronoi_refinement.c`, `voronoi/voronoi_derefinement.c`, `src/main/run.c`, public release 2019/2020), it is cited as "AREPO public source".

## Q1. AREPO (Springel 2010; Vogelsberger 2012; Pakmor 2016; Weinberger 2020; AREPO-2): criteria, split, merge, remap, mesh update, restrictions

### Takeaway
AREPO refines by splitting one active cell: a second generator is placed 0.025 r_cell from the parent in a random direction. Conserved quantities are divided by the exact volume ratio of the two daughter cells, computed on a small local Delaunay mesh. AREPO derefines by deleting one active generator. Its mass, momentum and energy go to its Voronoi neighbours in proportion to the exact volume each neighbour gains, found by clipping the dead cell's polyhedron (as tetrahedra) with neighbour bisector planes. Two neighbouring cells are never derefined in the same step, and a cell next to a derefined one is not refined in that step. Only "round" cells may be split. After the changes the mesh of active cells is rebuilt at the next sync point; mesh regularisation then separates the split pair over a few steps.

### Cited Findings
**Basic mechanism (Springel 2010, MNRAS 401, 791)**
- Springel 2010 says that splitting a single Voronoi cell is the one easy case. Inserting a new generator "at almost exactly the same location as the cell's original point" splits the cell in two and "will leave all surrounding Voronoi cells unchanged." This is "the basic mechanism for mesh-refinement." — [Springel 2010, arXiv:0901.4107 §6](https://arxiv.org/abs/0901.4107)
- On a split, mass, energy and momentum are divided "either simply by weighting with the relative fractions of the volumes occupied by the two new cells, or by using the estimated linear gradient for a conservative reconstruction combined with a volume integration." Mesh regularisation then separates the two nearby points "over the course of a few timesteps." Placing the new point "in the direction of fluid gradients" can steer the direction in which resolution is gained. — [Springel 2010](https://arxiv.org/abs/0901.4107)
- The Lagrangian mesh "largely eliminates the need for 'mesh derefinements'", because "advection alone does not generate a need for refinement or derefinement, in contrast to AMR codes." — [Springel 2010](https://arxiv.org/abs/0901.4107)
- Rejected derefinement option: slowly moving two points together and then merging them. It was rejected because the decision involves two cells at once and needs several steps (the criteria may change in the meantime). Adopted option: a cell "is dissolved instantly by simply removing its mesh-generating point." Its volume "will be claimed by the surrounding Voronoi cells", and its mass, energy and momentum are distributed "in proportion to the claimed volume fractions". This "requires the construction of the Voronoi diagram of the neighbouring cells with and without the point that is removed." — [Springel 2010](https://arxiv.org/abs/0901.4107)
- Ordering rule: derefinement criteria must allow a priority ordering. Cells are flagged in order of urgency, and any cell that already has a neighbour flagged in the same step is skipped. So "two neighbouring cells are never derefined in the same timestep." Reason given: removing one cell enlarges its neighbour, which can change that neighbour's criterion. — [Springel 2010](https://arxiv.org/abs/0901.4107)
- Noh 2D test: cells split if m_i > 1.5 m_0 and removed if V_i < 0.25 V_0, which caps the effective resolution at N_eff = 50. The derefinement "tends to smooth out high frequency noise" in the post-shock region, and the scheme works "without introducing any artefacts." — [Springel 2010 §8](https://arxiv.org/abs/0901.4107)
- Noh 3D test: cells created if m_i > 1.5 m_0 and dissolved if V_i < 0.05 V_0. In the low-resolution run (~16.7³ initial cells in the octant), "about 812620 cells have been created, and 225209 were destroyed" by t = 2. The high-resolution run (~50³ cells) had about 27× more. "Some limited postshock oscillations are present." — [Springel 2010 §8](https://arxiv.org/abs/0901.4107)

**Cosmological practice (Vogelsberger et al. 2012, MNRAS 425, 3024)**
- Cell mass is held in the range 1/2 ≲ m_cell/m_target ≲ 2, where m_target is the mean cell mass for equal-volume cells (the same as the SPH particle mass in the matching GADGET run). — [Vogelsberger et al. 2012, arXiv:1109.1281 §2](https://arxiv.org/abs/1109.1281)
- Only "roundish" cells are refined: "we only refine a cell if the maximum face angle is smaller than 3.38." This "protects against introducing large local errors by splitting very irregular cells (which are usually cells that have just been split in the previous timestep)." As a result, a few cells can temporarily exceed 2 m_target. — [Vogelsberger et al. 2012](https://arxiv.org/abs/1109.1281)
- Runs that allowed derefinement only for cells with a shallow temperature gradient "did not make any significant difference." — [Vogelsberger et al. 2012](https://arxiv.org/abs/1109.1281)
- "Thanks to the Lagrangian mesh motion in AREPO, these re-/derefinement operations are invoked only rarely." The resulting cell-mass distribution is roughly log-normal within the bounds. Because neighbours cannot both be derefined in one step, some cells sit below m_target/2 "for a few time steps". Star formation is therefore barred in cells with m < m_target/4. — [Vogelsberger et al. 2012, App. A](https://arxiv.org/abs/1109.1281)

**Public code paper (Weinberger, Springel & Pakmor 2020, ApJS 248, 32)**
- "To refine a cell, the mesh-generating point is split into a pair of two very close points, offsetting their location in a random direction by 0.025 r_cell", where r_cell = (3V/4π)^{1/3}. Conserved quantities "are then subdivided conservatively among the two new cells according to their volume ratio." — [Weinberger et al. 2020, arXiv:1909.04667 §5.2](https://arxiv.org/abs/1909.04667)
- For derefinement, conserved quantities "are conservatively distributed among these neighbors, in proportion to the volume overlap they realize with the removed cell." — [Weinberger et al. 2020](https://arxiv.org/abs/1909.04667)
- Default criterion: derefine if m < 0.5 m_target and refine if m > 2 m_target. An optional criterion resolves the Jeans length with a given number of cells. "Normally highly distorted cells are excluded from refinement ... After a refinement event, the mesh steering motions take a couple of time steps to reestablish a locally regular mesh geometry." — [Weinberger et al. 2020](https://arxiv.org/abs/1909.04667)
- "The refinement/de-refinement operations are local to a given cell and do not trigger any global load-balancing operations." — [Weinberger et al. 2020, footnote 4](https://arxiv.org/abs/1909.04667)

**Implementation details (AREPO public source, read 2026-11-02)**
- Placement in the step: `do_derefinements_and_refinements()` runs after the first half of the flux computation and before `find_next_sync_point()`, the domain decomposition and `create_mesh()`. The new and removed cells are therefore absorbed by the next mesh construction, not by local Delaunay insertion or removal into a persistent mesh. Derefinement runs first, then refinement. Timer: CPU_REFINE. — [AREPO public source, src/main/run.c, src/mesh/refinement.c](https://gitlab.mpcdf.mpg.de/vrs/arepo)
- Only active cells (`TimeBinsHydro.ActiveParticleList`) are tested. A split daughter is added to the parent's hydro and gravity time bins (`timebin_add_particle(..., P[i].TimeBinHydro ...)`). It is a copy of the parent with a new ID, and both carry `SepVector = dir`. — [AREPO public source, voronoi_refinement.c](https://gitlab.mpcdf.mpg.de/vrs/arepo)
- Split volumes: AREPO builds an auxiliary local Delaunay mesh (`DeRefMesh`) from the parent's Voronoi neighbours, the parent and the new point, computes the two daughter volumes, and scales mass, energy and momentum by V_i/(V_i+V_j) and V_j/(V_i+V_j). With MHD, B is extrapolated to the new position with the gradient. The offset is 0.025·rad with an isotropic random direction (random angle in 2D). — [AREPO public source, voronoi_refinement.c](https://gitlab.mpcdf.mpg.de/vrs/arepo)
- Derefinement geometry: the dead cell's Voronoi cell is collected as tetrahedra ("triangles" in the code) from its Delaunay connections. Each piece is assigned to the first neighbour, then clipped successively by each other neighbour (`derefine_add_point_and_split_tri`). Summing the pieces gives the exact volume each neighbour claims. Mass, momentum and energy (and scalars and B) go to neighbour q with weight fac = vol_q / Σvol. Neighbours whose shared face area is below DEREF_SA_FAC = 1e-4 × the cell's surface area are excluded. Updates to remote neighbours go through an MPI "flux list". This is exact intersection of the dead cell with the new neighbour cells, but the remap is piecewise-constant: no gradient weighting, and neighbours' own contents are not remapped. — [AREPO public source, voronoi_derefinement.c](https://gitlab.mpcdf.mpg.de/vrs/arepo)
- Independent set: candidates are sorted globally (by ID, with a random tie-breaker). A cell is eligible only if no active neighbour is already flagged, and a cell whose neighbour has the same ID (a periodic mirror) is skipped. A second check runs across MPI ranks. Every neighbour that receives mass sets `FlagDoNotRefine[p] = 1`, so it cannot be split in the same step. — [AREPO public source, voronoi_derefinement.c](https://gitlab.mpcdf.mpg.de/vrs/arepo)
- Other criteria in `REFINEMENT_VOLUME_LIMIT`:
  - refine if V > 2·MaxVolume;
  - never refine if V < 2·MinVolume;
  - refine if V > MaxVolumeDiff × (smallest neighbour volume), which limits the volume ratio between neighbouring cells;
  - derefine if V < 0.5·MinVolume;
  - never derefine if V > 0.5·MaxVolume, or if V > 0.3·MaxVolumeDiff·MinNgbVolume.
  
  The "round enough" test is either MaxFaceAngle < 1.5·CellMaxAngleFactor, or a distance from the centroid below 2·CellShapingFactor·r_cell. — [AREPO public source, criterion_refinement.c, criterion_derefinement.c](https://gitlab.mpcdf.mpg.de/vrs/arepo)
- 1D: the refinement and derefinement code is compiled only `#if ... !defined(ONEDIMS)`, so the public AREPO has no 1D (de)refinement. — [AREPO public source](https://gitlab.mpcdf.mpg.de/vrs/arepo)

**Custom criteria (Curtis & Sijacki 2015)**
- "Super-Lagrangian" refinement around black holes: inside R_ref (= the BH smoothing length h_BH), cells are forced into a radius corridor that grows linearly with distance, between R_cell,min (= the Bondi radius) and R_cell,max = 0.5 h_BH. A slack factor c = 2 is used. Setting c too low means "cells are constantly being refined and de-refined", which costs a lot. — [Curtis & Sijacki 2015, arXiv:1502.03445 §2.2](https://arxiv.org/abs/1502.03445)
- Cost: a 16³-cell Bondi run with refinement reproduced the Bondi rate at a CPU time "between the costs for the 16³ and 32³ runs without refinement." With aggressive parameters, a 32³ run matched a run with 8× more cells "at somewhat lower CPU cost." — [Curtis & Sijacki 2015 §3](https://arxiv.org/abs/1502.03445)
- That paper also notes that "angular momentum conservation is not guaranteed in AREPO." — [Curtis & Sijacki 2015](https://arxiv.org/abs/1502.03445)

**AREPO-2 / next generation**
- A 2026 paper describes AREPO-2 as keeping "an approximately constant baryonic mass resolution ... by refining (splitting) cells whose masses become too large and derefining (merging) those whose masses become too small." The paper cites Pakmor et al. 2023 (MPI-3 shared memory, hierarchical domain decomposition, more than 10⁵ MPI tasks), and Zier et al. 2024 (GPU radiative transfer plus "an abstraction layer that will facilitate porting further parts of the code to GPUs"). — [Zier et al. 2026, arXiv:2607.05504 §3](https://arxiv.org/abs/2607.05504)

### Inferences
- AREPO uses exact geometry in both operations, done locally on a small auxiliary Delaunay mesh. The split uses two exact daughter volumes, and the merge uses an exact clip of the dead polyhedron by neighbour bisectors. The remap itself is first order (volume fractions of conserved totals). Springel 2010 mentions a gradient-based option for splits, but the public split code scales by volume ratio only.
- Because the global mesh is rebuilt every step for active cells, AREPO never needs persistent incremental Delaunay insertion or deletion for (de)refinement. A code that keeps a persistent mesh would need either local insert/delete or a cheap rebuild.
- The 0.025 r_cell offset plus regularisation, the "only round cells" test and the no-refine-next-to-merge rule together prevent repeated splits of the same cell and the slivers that a near-degenerate split creates.

### Gaps
- Pakmor et al. 2016 (MNRAS 455, 1134) mainly changes gradients and time integration. I found nothing in it that changes (de)refinement, but I did not read its full text here.
- I found no published per-step fraction of refined or derefined cells, and no CPU_REFINE share for large AREPO runs (Illustris, TNG, Auriga). Vogelsberger 2012 says only "invoked only rarely".
- I found no published AREPO-2 change to the refinement algorithm, and no GPU (de)refinement in AREPO as of 2026.

## Q2. GIZMO (Hopkins 2015) MFM/MFV particle splitting and merging, for comparison

### Takeaway
GIZMO's split and merge are mass-threshold based and deliberately simple. A merge folds the lightest particle into its second-lightest neighbour at the pair's centre of mass. A split makes two half-mass copies offset ±min(h/8, r_near/3) perpendicular to the number-density gradient. Hopkins calls both "noisy and diffusive" and recommends them only when necessary, but FIRE-feedback MFV discs crash without them.

### Cited Findings
- Merge: if a particle falls below ε_min·m_min(t0) (ε_min = 0.5, "ad-hoc") and is the least massive in its neighbour list, it merges with the second-least-massive neighbour. That neighbour inherits the summed mass, momentum and energy and their time derivatives, and moves to the centre of mass of the pair. Merges happen "only on timesteps where the neighbor/gravity tree is being reconstructed." — [Hopkins 2015, arXiv:1409.7395 App. E](https://arxiv.org/abs/1409.7395)
- Split: if a particle exceeds ε_max·m_max(t0) and is the most massive among its neighbours, it splits into two particles with half the mass each and the parent's specific properties. They are separated by min(h_i/8, |r_near|/3), moved in opposite directions "along an axis perpendicular to the particle number density gradient (to minimize the perturbation to volumetric quantities)." Doing this without seeding fluctuations "is highly non-trivial." — [Hopkins 2015 App. E](https://arxiv.org/abs/1409.7395)
- "These operations are both noisy and diffusive, and we recommend against particle merging/splitting unless absolutely necessary." The test problems show "very little difference" with or without them, because very few particles qualify. But the FIRE isolated disc with MFV "simply can[not]" run without them: winds bring particles of very different mass together and cause N-body problems. — [Hopkins 2015 App. E](https://arxiv.org/abs/1409.7395)
- For badly conditioned particle configurations, GIZMO avoids splitting. It expands the neighbour search until N_cond ≤ N_cond^crit (≈100–1000), and falls back to SPH gradients in rare cases, because splitting is "highly diffusive". — [Hopkins 2015 App. C](https://arxiv.org/abs/1409.7395)

### Inferences
- The perpendicular-to-gradient placement is the meshless counterpart of AREPO's small offset. Both aim to leave the local volume partition almost unchanged at the moment of the split.

### Gaps
- The current GIZMO user guide (parameters such as MaxMassForParticleSplit, MinMassForParticleMerger, and the hyper-refinement options) could not be fetched: the tapir.caltech.edu certificate had expired and the Bitbucket raw source is behind HTML. Later changes to the defaults are therefore not documented here.

## Q3. ALE / ReALE / Voronoi ALE and conservative remap methods (r3d/r2d, swept region, intersection) including GPU cost

### Takeaway
ReALE keeps the number of generators fixed and rebuilds the Voronoi mesh each rezone. It then remaps by cell intersection or simplified swept faces, so it changes topology but adds and removes no points. Exact polyhedron intersection (r3d) is robust and conserves to round-off in double precision, at about 0.2–0.8 µs per clip on a 2015 CPU core. Swept-region remap is cheaper but less accurate when the mesh moves badly, and hybrid schemes use intersection only near interfaces. Direct ALE (Gaburro, Springel, Dumbser) avoids remap by integrating over space-time control volumes with "sliver" or "hole-like" elements at topology changes: in 2D in 2020, extended to 3D polyhedra in 2026.

### Cited Findings
- ReALE (Loubère, Maire, Shashkov, Breil, Galera 2010, JCP 229, 4724) is standard ALE (Lagrangian step, rezone, remap) whose rezone also allows connectivity changes through Voronoi diagrams. The remap partitions each rezoned cell by overlap with Lagrangian cells, using either a simplified swept-face method or a cell-intersection method. — [ReALE, JCP 2010](https://www.sciencedirect.com/science/article/abs/pii/S002199911000121X); [Gaburro et al. 2020 review section](https://arxiv.org/abs/1905.00967)
- R-ReALE (Bo & Shashkov 2015) adds r-adaptivity: the generators are moved using a monitor function based on the Hessian of flow variables, plus a centroidal Voronoi tessellation. This changes resolution by redistributing a fixed number of points, not by adding points. — [R-ReALE, J. Math. Study](https://www.global-sci.com/jms/article/view/13240)
- Klima, Kucharik & Shashkov (CiCP 21, 2017) compare "the intuitive, but resource intensive method utilizing intersections of computational cells with the faster and simpler swept-region-based method." Swept remap can distort symmetric distributions more when the mesh moves in an unsuitable direction. — [Klima et al. 2017](https://ojs.global-sci.org/index.php/cicp/article/view/6412)
- Hybrid remap (Kucharik & Shashkov, JCP 231, 2851, 2012): "perform the expensive intersections only in the vicinity of material interfaces, while cheaper swept-based remap is done in the rest of the domain." — [Kucharik/Shashkov hybrid remap notes](https://cnls.lanl.gov/~shashkov/papers/hybrid_milan.pdf)
- r3d (Powell & Abel 2015, JCP 297, 340): exact integration of polynomials (up to quadratic moments) over the intersection of convex polyhedra, by clipping one polyhedron against the other's faces. It follows Sugihara's planar-graph method for topological robustness. Timings, per 10⁶ clips on a 2.2 GHz Xeon E5-2660: 220 ms with 1 plane, 760 ms with 4 planes (≈0.22–0.76 µs per clip, "faster by a factor of 3-4" than López & Hernández). Reduction to volume and moments costs 150/290 ms per 10⁶ tetrahedra and 690/1850 ms per 10⁶ dodecahedra. Double-precision voxelization error is about 1.6e-12 rms and 5e-10 max. Single precision gives about 5e-4 rms and 0.19 max. 2D remesh conserves mass to machine precision. — [Powell & Abel 2015, arXiv:1412.4941](https://arxiv.org/abs/1412.4941)
- r3d is the default 2D/3D intersection kernel in LANL's Portage remap framework. Portage threads with NVIDIA Thrust, not Kokkos. A LANL study of GPU volume-fraction and centroid computation found a "10-100 times" speedup but "significant effort is required to re-factor the code." — [Portage SC17 poster](https://sc17.supercomputing.org/SC17%20Archive/tech_poster/tech_poster_pages/post169.html); [OSTI 1471309](https://www.osti.gov/biblio/1471309) (search snippet only; full report not read)
- Direct ALE on moving 2D Voronoi meshes (Gaburro, Boscheri, Chiocchetti, Klingenberg, Springel, Dumbser 2020). The mesh is regenerated each step AREPO-style. Old and new cells of the same generator are joined into space-time control volumes. "Degenerate space–time sliver elements" fill the holes left by topology changes. The scheme is conservative and satisfies the GCL by construction, with no remap. — [Gaburro et al. 2020, arXiv:1905.00967](https://arxiv.org/abs/1905.00967)
- 3D extension (Gaburro, Klima, Bonafini, Tavelli, May–Aug 2026): 4D space-time "hole-like elements" for 2-3, 3-2 and 4-4 flips on a Delaunay tetrahedralization. The authors report it is fully conservative, satisfies the GCL, and keeps the order of accuracy "even with frequent topology changes." The abstract gives no cost data. — [arXiv:2605.23506](https://arxiv.org/abs/2605.23506)

### Inferences
- For a Voronoi code, local (de)refinement does not need a general old-to-new remap. Old and new cells differ only in the parent's (split) or dead cell's (merge) region, so intersection is needed for one polyhedron against at most ~15–20 bisector planes. At r3d-like rates of ≈0.2 µs per plane-clip on one 2015 CPU core, one merge costs roughly a few µs. That is estimated, not measured.
- A split needs only the two daughter volumes. These come directly from building the Voronoi cell of each daughter against the parent's neighbours, without any intersection.

### Gaps
- I found no published GPU throughput for r3d-style polyhedron intersection inside a hydro (de)refinement step. I also found no ReALE cost breakdown (rezone versus remap share) in sources I could open.

## Q4. Other codes (brief): TESS, Shadowfax, RICH, DISCO, MANGA, GPU codes

### Takeaway
Outside AREPO, adding and removing Voronoi generators is rare and simple. RICH uses AREPO-like volume-threshold split and coarsen in tests. TESS listed refinement as future work. I found no published (de)refinement for Shadowfax, DISCO, MANGA or any GPU moving-mesh code.

### Cited Findings
- RICH (2D, Yalinewich, Steinberg & Sari 2015): in the Noh test it uses "adaptive mesh refinement, like in AREPO": cells split above 150% of their initial volume and coarsen below 25%. Derefining cells around the shock front contributed to worse errors than Eulerian. For extreme neighbour size ratios it suggests "splitting cells when they become much larger than their neighbors, or coarsening cells when they get much smaller." — [RICH, arXiv:1410.3219](https://arxiv.org/abs/1410.3219)
- Steinberg, Yalinewich & Sari 2016 trace Voronoi moving-mesh grid noise to a volume-inconsistency error: the volume exchanged between cells disagrees with the hydro update, which prevents second-order convergence. A consistent-volume fix restores second order. — [arXiv:1603.06753](https://arxiv.org/abs/1603.06753)
- TESS (Duffell & MacFadyen 2011): generators "can be added or removed to enable adaptive refinement" in principle. The conclusions list "mesh refinement, and a local time step" as not yet implemented ("implemented in AREPO and are thus a solved problem"). — [TESS, arXiv:1104.3562](https://arxiv.org/abs/1104.3562)
- The Shadowfax (2016) abstract does not mention refinement or derefinement. — [Shadowfax, arXiv:1605.03576](https://arxiv.org/abs/1605.03576)
- DISCO (Duffell 2016) uses a shearing cylindrical moving mesh. I found no refinement or derefinement description. — [DISCO, arXiv:1605.03577](https://arxiv.org/abs/1605.03577)

### Inferences
- In practice, AREPO's split/merge is the de facto reference design for Voronoi codes. Others copy its thresholds (mass factor 2, or volume ratios such as 1.5 and 0.25).

### Gaps
- MANGA/ChaNGa moving mesh (Chang et al. 2017; Prust & Chang 2019), FVMHD3D and AREPO-RT: not checked for (de)refinement details.
- No published GPU moving-mesh code with dynamic generator insertion and removal was found (2024–2026).

## Q5. How often refinement happens and what it costs

### Takeaway
Hard numbers are scarce. Quasi-Lagrangian motion keeps (de)refinement "rare" in cosmological runs. Problems with strong compression, or with refinement regions that stay in place, can create many cells over a run: in the 3D Noh test, about 175 created per initial cell at low resolution. Targeted refinement costs about what it saves: a refined 16³ run cost between unrefined 16³ and 32³ runs.

### Cited Findings
- Cosmology: "these re-/derefinement operations are invoked only rarely." — [Vogelsberger et al. 2012](https://arxiv.org/abs/1109.1281)
- 3D Noh, low resolution: 812,620 cells created and 225,209 destroyed by t = 2, from ~4.7×10³ initial cells in the octant (16.7³). At ~3× linear resolution the counts are ~27× higher. — [Springel 2010](https://arxiv.org/abs/0901.4107)
- Black hole refinement cost: see the Q1 cost entries. — [Curtis & Sijacki 2015](https://arxiv.org/abs/1502.03445)
- AREPO (de)refinement is local and "does not trigger any global load-balancing operations." — [Weinberger et al. 2020](https://arxiv.org/abs/1909.04667)

### Inferences
- Because only active cells are tested and AREPO's step-to-step mass change comes from small mesh-motion mass fluxes, the per-step refined fraction in cosmological runs is presumably well below 1% of active cells. This is an estimate, not measured.
- The Noh counts imply that in compressive flows, total split events over a run can exceed the initial cell count by a factor of ~100. A code with refinement should therefore not treat the operation as a rare slow path.

### Gaps
- No published per-step fraction or CPU_REFINE share was found for Illustris, TNG, Auriga or FIRE. Getting these would need run logs (AREPO's cpu.txt) or direct contact with the authors.

## Q6. Lessons: what goes wrong and best practice

### Takeaway
Use exact local volumes for the conservative transfer. Split only well-shaped cells, with a tiny offset, and let regularisation separate the pair. Derefine an independent set with no adjacent removals. Block refining next to a merge. Use a hysteresis band (factor 4 between the split and merge thresholds). Keep the neighbour volume ratio bounded. Expect some diffusion and noise, especially from merges and from particle splits in meshless codes.

### Cited Findings
- Hysteresis: AREPO uses m > 2 m_target to split and m < 0.5 m_target to merge. Curtis & Sijacki found that a narrow corridor (c too small) makes cells cycle between refinement and derefinement, at high cost. — [Weinberger et al. 2020](https://arxiv.org/abs/1909.04667); [Curtis & Sijacki 2015](https://arxiv.org/abs/1502.03445)
- Splitting irregular or just-split cells gives "large local errors", hence the shape gate (maximum face angle < 3.38). — [Vogelsberger et al. 2012](https://arxiv.org/abs/1109.1281); [Weinberger et al. 2020](https://arxiv.org/abs/1909.04667)
- The order of derefinement matters, because removing one cell changes its neighbours' criteria. Hence the priority ordering and no adjacent removals. As a side effect, some cells stay below threshold for a few steps. — [Springel 2010](https://arxiv.org/abs/0901.4107); [Vogelsberger et al. 2012](https://arxiv.org/abs/1109.1281)
- Large neighbour volume ratios cause errors and crashes, and a size-ratio criterion is the fix: AREPO's MaxVolumeDiff and RICH's suggestion. — [AREPO public source](https://gitlab.mpcdf.mpg.de/vrs/arepo); [RICH](https://arxiv.org/abs/1410.3219)
- Merges are diffusive but can smooth noise (Noh post-shock). Derefinement near shocks can hurt accuracy (RICH Noh). — [Springel 2010](https://arxiv.org/abs/0901.4107); [RICH](https://arxiv.org/abs/1410.3219)
- Particle splitting and merging in meshless codes is "noisy and diffusive". — [Hopkins 2015](https://arxiv.org/abs/1409.7395)
- Mass-spectrum side effects: very unequal masses cause N-body heating, so star formation is restricted in low-mass or refined cells. — [Vogelsberger et al. 2012](https://arxiv.org/abs/1109.1281); [Curtis & Sijacki 2015](https://arxiv.org/abs/1502.03445); [Hopkins 2015](https://arxiv.org/abs/1409.7395)
- Time-step aspect: in AREPO, refinement creates the daughter in the parent's time bin, and only active cells are split or merged. Small cells shorten the CFL step, which is one reason for MinVolume limits (the Noh volume floor). — [AREPO public source](https://gitlab.mpcdf.mpg.de/vrs/arepo); [Springel 2010](https://arxiv.org/abs/0901.4107)

### Inferences
Suggested minimal recipe for a new GPU Voronoi code; these are recommendations, not sourced facts.
1. **Flagging:** flag active cells by the mass band [0.5, 2]·m_target, an optional Jeans criterion and a neighbour volume ratio. Allow splits only for cells that pass a shape test.
2. **Merge selection:** choose merges as a maximal independent set with a deterministic priority (ID or hash). On a GPU this is a Luby-style parallel independent set. Forbid splitting neighbours of merged cells.
3. **Splits:** place the daughter at 0.025 r_cell in a random direction, or along the density gradient. Divide mass, momentum and energy by the exact daughter volumes.
4. **Merges:** give the dead cell's conserved totals to its neighbours in proportion to the exact volume each gains, by clipping the dead polyhedron with bisector planes.
5. **Mesh update:** rebuild the mesh, or update it locally, before the next flux computation.

## Sources list (all used above)
- Springel 2010, MNRAS 401, 791 — https://arxiv.org/abs/0901.4107
- Vogelsberger et al. 2012, MNRAS 425, 3024 — https://arxiv.org/abs/1109.1281
- Weinberger, Springel & Pakmor 2020, ApJS 248, 32 — https://arxiv.org/abs/1909.04667
- AREPO public source — https://gitlab.mpcdf.mpg.de/vrs/arepo
- Curtis & Sijacki 2015, MNRAS 454, 3445 — https://arxiv.org/abs/1502.03445
- Zier et al. 2026 (AREPO-2 SIDM) — https://arxiv.org/abs/2607.05504
- Hopkins 2015, MNRAS 450, 53 — https://arxiv.org/abs/1409.7395
- Loubère et al. 2010 ReALE — https://www.sciencedirect.com/science/article/abs/pii/S002199911000121X
- Bo & Shashkov R-ReALE — https://www.global-sci.com/jms/article/view/13240
- Klima, Kucharik & Shashkov 2017 — https://ojs.global-sci.org/index.php/cicp/article/view/6412
- Kucharik & Shashkov hybrid remap — https://cnls.lanl.gov/~shashkov/papers/hybrid_milan.pdf
- Powell & Abel 2015 r3d — https://arxiv.org/abs/1412.4941
- Portage SC17 poster — https://sc17.supercomputing.org/SC17%20Archive/tech_poster/tech_poster_pages/post169.html
- Gaburro et al. 2020 — https://arxiv.org/abs/1905.00967 ; Gaburro et al. 2026 — https://arxiv.org/abs/2605.23506
- RICH — https://arxiv.org/abs/1410.3219 ; Steinberg et al. 2016 — https://arxiv.org/abs/1603.06753
- TESS — https://arxiv.org/abs/1104.3562 ; Shadowfax — https://arxiv.org/abs/1605.03576 ; DISCO — https://arxiv.org/abs/1605.03577
