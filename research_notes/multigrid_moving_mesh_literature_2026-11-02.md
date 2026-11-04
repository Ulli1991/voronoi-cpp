# Multigrid self-gravity on a moving mesh: literature check (2026-11-02)

Question: has anyone coupled a multigrid Poisson solver to moving-mesh
(moving Voronoi / Lagrangian / ALE) hydrodynamics?

Method: web search (arXiv, ar5iv, ADS, journal pages) on 2026-11-02. Primary
sources were opened where marked "[read]". Entries marked "[secondary]" are
taken from abstracts or citing papers only. Search coverage is good for the
named astrophysics codes and weaker for CFD/plasma ALE codes. "Not found"
means not found in these searches; it does not prove absence.

## Bottom line

**(b) Partly done.** "Multigrid gravity on a moving mesh" has been done
before, but only on two kinds of mesh:

1. **Structured curvilinear moving mesh, multigrid on the deformed grid,
   with hydro:** Pen 1998 (MMH), building on Pen 1995. This is the clearest
   prior art for the general claim. Its hydro grid is a potential-flow
   deformation of a Cartesian grid, not a Voronoi tessellation.
2. **Unstructured ALE mesh (2D, arbitrary polygons), with a multigrid-type
   gravity solve on the moving hydro grid:** BETHE-Hydro (Murphy & Burrows
   2008). Gravity is solved directly on the moving grid with
   conjugate gradients preconditioned by algebraic multigrid (AMG1R6).
   It is 2D and runs on CPU, and multigrid serves as the preconditioner
   rather than as the solver.

**Not found:** any multigrid Poisson solver, whether geometric, agglomeration
or algebraic, used for self-gravity in a **moving Voronoi** (AREPO-family)
code. That holds whether the solve runs on the Voronoi cells or on a
Cartesian/AMR grid that the cells deposit onto. Every Voronoi moving-mesh
code checked uses a tree, TreePM or FFT-PM. A GPU multigrid gravity solver
on any moving unstructured mesh was also not found.

So a claim of "first multigrid self-gravity in a moving Voronoi-mesh code" is
defensible after this search, with the usual caveat. A claim of "first
multigrid gravity on a moving mesh" is **false**: Pen 1995/1998, BETHE-Hydro
2008 and Horowitz 2026 (N-body only) predate it.

## 1. Gravity solvers in moving-mesh / Lagrangian / ALE codes

| Code | Mesh | Gravity solver | Grid for gravity | Source |
|---|---|---|---|---|
| AREPO (Springel 2010, MNRAS 401, 791; arXiv:0901.4107) | moving Voronoi | Tree or TreePM (GADGET-derived); cells treated as softened point masses, softening tied to cell size | particles + Cartesian FFT mesh for the PM part | [read] arXiv:1109.2218: "we use a similar TreePM solver for gravity as employed in the GADGET code", "Voronoi cells are treated effectively as point masses" |
| AREPO public release (Weinberger, Springel & Pakmor 2020, ApJS 248, 32; arXiv:1909.04667) | moving Voronoi | TreePM ("tree-particle-mesh algorithm for the Poisson equation") | tree + FFT mesh | [secondary, abstract] |
| AREPO-2 (referenced in Zier et al. 2026, arXiv:2607.05504) | moving Voronoi | Calls itself only "optimized gravity solver" for the Poisson equation; no multigrid named. GPU port so far covers RT only (Zier et al. 2024, arXiv:2404.17630) | not stated | [read abstract] No AREPO-2 method paper found. |
| TESS (Duffell & MacFadyen 2011, ApJS 197, 15; arXiv:1104.3562) | moving Voronoi | none; external or accelerated-frame gravity only | n/a | [read] |
| Shadowfax (Vandenbroucke & De Rijcke 2016, A&C 16, 109; arXiv:1605.03576) | moving Voronoi | Barnes-Hut tree with relative opening criterion and Ewald summation; gas cells treated as particles with fixed spline softening | particles | [read] |
| RICH 2D (Yalinewich, Steinberg & Sari 2015, ApJS 216, 35; arXiv:1410.3219) | moving Voronoi | No self-gravity solver in the method paper (source terms only) | n/a | [secondary] |
| RICH 3D (Steinberg et al. 2019, arXiv:1903.03898; Steinberg & Stone 2024, Nature 625, 463; Giron et al. 2026, arXiv:2601.05120) | moving Voronoi 3D | Tree up to quadrupole moments (stated for the 2019 TDE paper) | particles | [secondary] |
| DISCO (Duffell 2016, ApJS 226, 2; arXiv:1605.03577) | moving cylindrical annuli (shearing) | No self-gravity solver found in the method paper; point-mass potentials | n/a | [secondary]; not exhaustively checked |
| Sprout (Mandal & Duffell 2023, arXiv:2307.13785) | expanding Cartesian | none found | n/a | [secondary] |
| MANGA / ChaNGa moving mesh (Chang, Wadsley & Quinn 2017, MNRAS 471, 3577; arXiv:1707.05333; Chang, Davis & Jiang 2020, MNRAS 493, 5397) | moving Voronoi (VORO++ / own) | ChaNGa tree with hexadecapole moments; cells treated as spline-softened particles | particles | [read] Only performance numbers given: Voronoi build at about 0.1 M cells/s/core (VORO++) and 0.01 M cells/s/core (MANGA), i7-3770 |
| FVMHD3D (Gaburov, Johansen & Levin 2012) | moving Voronoi (MHD) | not determined in this search | n/a | gap |
| GIZMO (Hopkins 2015, MNRAS 450, 53) | meshless MFM/MFV (no mesh) | TreePM from GADGET-3 with adaptive softening | particles + FFT | [secondary] |
| PROTEUS | n/a | no published paper found under this name | n/a | per instruction, only published papers considered |
| Pen MMH (Pen 1998, ApJS 115, 19; arXiv:astro-ph/9704258) | structured curvilinear moving grid (potential-flow deformation of Cartesian) | **iterative multigrid on the curvilinear (deformed) grid**; Poisson written with metric terms (eq. 11). A second multigrid elliptic solve handles grid deformation. | the moving hydro grid itself | [read] details below |
| Pen 1995 (ApJS 100, 269), "Linear Moving Adaptive PM" | curvilinear moving grid, N-body only | multigrid on the deformed grid | moving grid | [secondary] |
| Gnedin SLH (Gnedin 1995, ApJS 97, 231) + Gnedin & Bertschinger 1996 (ApJ 470, 115; arXiv:astro-ph/9602063) | softened Lagrangian deformable mesh | Originally "Moving Mesh Gravity": FD Poisson on the deformed mesh (iterative method not named). **Abandoned** because on strongly deformed meshes it "does not necessarily converge to the true solution, giving the incorrect force law". Replaced by P3M on a Cartesian mesh. | deformed mesh, then Cartesian | [read] Relevant warning for Poisson on distorted cells. |
| BETHE-Hydro (Murphy & Burrows 2008, ApJS 179, 209; arXiv:0805.3356) | 2D ALE, arbitrary unstructured polygonal grid | Support-operator discretisation (Morel et al. 1998) solved "explicitly on the arbitrary hydro grid, eliminating an interpolation step". **CG with AMG1R6 algebraic multigrid preconditioner** (Ruge & Stüben 1987), warm-started from the previous step and called in predictor and corrector. Outer BC from a multipole expansion. 1D case uses a tridiagonal solve. | the moving ALE hydro grid | [read] No timings in the part read. |
| VULCAN/2D (Livne et al. 2004, ApJ 609, 277) | 2D ALE, structured or unstructured | "direct grid-based solution of the Newtonian Poisson equation" (Burrows et al. 2007); solver type not determined | not determined | [secondary] gap |
| Horowitz 2026, "Fast(er)PM and Moving Mesh: JAX-native Geometric Multigrid Methods" (arXiv:2607.10983, 13 Jul 2026) | Pen-style curvilinear deforming Cartesian grid, **N-body PM only, no hydro** | Geometric multigrid V-cycles on a variable-coefficient Laplace-Beltrami operator in flux form; coarse operators rebuilt from restricted metric fields; warm start, about 6 V-cycles late; Jacobi smoothing beats Chebyshev on the curved mesh | moving grid | [read] **GPU (A100, Perlmutter, JAX)**. Details in section 2. |

Searches for "moving mesh" + "multigrid" + Voronoi (2024–2026) found no other
hits. Note that one search-engine summary claimed AREPO solves "Poisson on
the moving baryonic mesh with a relaxation technique". **Primary sources do
not support this**; AREPO uses TreePM.

### Pen 1998 MMH details [read]
- Eq. 11 is Poisson with curvilinear metric, solved "using the multigrid
  algorithm described in Pen (1995)". The grid-deformation potential needs a
  second elliptic multigrid solve ("two elliptic equations ... instead of
  one").
- Cost is about 19,000 flop/cell/Courant step, roughly half hydro and half
  multigrid. Multigrid costs about 1,419 flop/cell per call (hardware counter,
  64³) and is called 4 times per Courant step.
- Hardware: Cray C90 at about 300 Mflop sustained on 256³; SGI Power
  Challenge R8000 at 48 Mflop/CPU on 64³; speedup about 6 on 8 CPUs. No
  wall-clock throughput is given. A rough estimate is about 300e6/19e3 ≈
  0.016 M cells/s per C90 CPU (derived, not quoted).
- Santa Barbara comparison (Frenk et al. 1999, ApJ 525, 554;
  arXiv:astro-ph/9906160): "The gravity and grid deformation equations are
  solved using a hierarchical multigrid algorithm."

## 2. Multigrid / AMG on unstructured, Voronoi or polyhedral meshes

- **OpenFOAM GAMG**: geometric agglomeration (pairwise, merging across the
  largest-area face) with algebraic coarse operators. It is the standard
  pressure-Poisson solver on polyhedral meshes, including dynamic or moving
  meshes (`dynamicFvMesh`), and has agglomeration caching and update-interval
  options. CFD has done agglomeration multigrid Poisson on moving polyhedral
  meshes routinely, but not for self-gravity and not on Voronoi meshes
  regenerated each step. [doc.cfd.direct, OpenFOAM API]
- **Agglomeration geometric MG on polytopal meshes**: Pan & Persson,
  arXiv:2012.08024 (CDG Poisson, agglomeration coarsening); Badia & Manyer,
  arXiv:2603.00860 (2026, GMG for HHO on agglomerated polytopal meshes, 2D
  and 3D). These cover static meshes only.
- **Lagrangian Voronoi CFD**: SILVA (Kincl, Peshkov & Boscheri 2024,
  arXiv:2405.04116; compressible version arXiv:2410.14564) regenerates the
  Voronoi mesh every step and solves the pressure Poisson with MINRES/CG with
  no AMG mentioned. Only 2D benchmarks are reported. The authors say the
  pressure solve, not the mesh rebuild, is the bottleneck. This is the
  closest non-astro analogue (Poisson on a moving Voronoi mesh), and it does
  not use multigrid.
- **AMG for particle-method Poisson**: arXiv:0905.3005 studies AMG for the
  non-symmetric Poisson matrices of Lagrangian particle methods. On a Voronoi
  mesh the two-point-flux Laplacian (A_f/d_f) is symmetric, which avoids that
  problem.
- **AMG setup cost on moving meshes**: Sandia wind-turbine study (2019,
  classical vs aggregation AMG). Mesh motion forces matrix and preconditioner
  rebuild every step, and cheaper setup at the cost of slower solves wins.
  Partial-reuse AMG (arXiv:2108.02054) cuts setup 40–200% and total time up
  to 20%. Both bear directly on a Voronoi mesh that changes every step.
- **GPU AMG throughput (published numbers are sparse and not comparable)**:
  - AmgX (Naumov et al. 2015, SIAM SISC): 2–5x over a CPU code on 1 GPU; no
    throughput figure.
  - hypre BoomerAMG GPU (ECP highlight; Li et al. LLNL slides): 3.1–3.8x
    total and 7–8x solve-phase over CPU. One 7-pt Laplacian AMG-PCG case on
    4 A100s takes 0.156 s setup + 0.137 s solve, problem size not legible.
  - PETSc GAMG on Summit (Mills et al. 2021, arXiv:2011.00715): Fig. 8 gives
    times for a 3D Q2 Laplacian; no throughput in text.
  - **Horowitz 2026 (geometric MG, GPU, structured grid)**: 256³ on 1 A100
    at 16.5 / 20.1 / 23.3 ms per step for 1 / 2 / 3 warm V-cycles,
    vs 17.9 ms for FFT. That is about 0.72–1.0 G cells/s per A100 for the
    whole PM step (derived). At 1024³, 1-cycle MG beats FFT by 1.64x on 32
    GPUs and 2.38x on 128 GPUs. These are fixed-mesh numbers; no per-step
    timing is given for the moving mesh.
- **Astro Cartesian/AMR multigrid (reference points, not moving mesh)**:
  Tomida & Stone 2023 (ApJS 266, 7; arXiv:2302.13903), Athena++ multigrid:
  "small fraction" of MHD cost and faster than FFT on a uniform grid (numbers
  only in the full text). Others: Guillet & Teyssier 2011 (RAMSES), Ricker
  2008 (FLASH), Mandal et al. 2023 (PLUTO), Yen 2020 (nested grids).

## 3. Moving mesh deposited to Cartesian/AMR, then multigrid

- No code found that deposits moving-Voronoi cell masses onto a
  Cartesian/AMR grid, solves with multigrid, and interpolates back every step.
- AREPO's TreePM does deposit onto a Cartesian mesh (CIC), but the long-range
  part is solved with **FFT**. Zoom runs use two FFT levels (e.g. Hestia,
  arXiv:2008.04926).
- VorAMR (Lewis et al. 2025, ApJ 994, 69; arXiv:2511.14697) interpolates an
  AREPO snapshot onto a FLASH AMR grid once, for a hand-off to Torch. It is
  not a coupled gravity solve.
- Pen 1998 is the inverse case: its PM N-body particles are deposited onto the
  moving curvilinear grid, and multigrid runs on that grid.

## 4. Momentum-conserving gravity as face fluxes

- Grid codes: the gravitational stress tensor
  T_g = (1/4πG)[∇φ∇φ − ½|∇φ|² I], with fluxes at faces, is used in Athena and
  Athena++. Jiang, Belyaev, Goodman & Stone 2013, New Astron. 19, 48
  (arXiv:1208.1795), handles total energy and notes that the momentum tensor
  form is already standard in Athena. Mullen, Hanawa & Gammie 2021, ApJS 252,
  30 (arXiv:2012.01340), is fully conservative in Athena++: the common tensor
  discretisation is not curl-free and gives anomalous accelerations, and is
  fixed by changing the diagonal terms. Higher-order follow-up:
  arXiv:2502.18794.
- **Voronoi / unstructured: not found.** No paper writes self-gravity as a
  stress-tensor face flux on Voronoi, polyhedral or tetrahedral cells. AREPO
  applies gravity as a source term and couples energy through mass fluxes
  (Springel 2010 §5), and tree forces do not conserve momentum or angular
  momentum exactly.
- BETHE-Hydro's support-operator (mimetic) discretisation on polygons is the
  nearest analogue to a compatible face-based gravity discretisation on an
  unstructured moving mesh. It uses a mimetic potential and gradient, not a
  stress tensor.

## Closest prior work (ranked)

1. **Murphy & Burrows 2008, BETHE-Hydro** (ApJS 179, 209; arXiv:0805.3356):
   2D unstructured polygonal ALE hydro, gravity solved on the moving grid by
   AMG-preconditioned CG. Closest in mesh type (unstructured, moving, Poisson
   on hydro cells). It differs in being 2D, ALE rather than Voronoi, AMG as
   preconditioner, and CPU.
2. **Pen 1998, MMH** (ApJS 115, 19; Pen 1995 ApJS 100, 269): 3D
   self-gravitating hydro, multigrid gravity on the moving (curvilinear) grid.
   Closest in "multigrid + moving mesh + 3D hydro". The mesh is structured,
   not Voronoi.
3. **Horowitz 2026** (arXiv:2607.10983): GPU geometric multigrid on a
   Pen-style moving mesh in JAX on A100, with measured timings. N-body only,
   no hydro, structured.
4. **Gnedin & Bertschinger 1996**: Poisson on a deformed Lagrangian mesh was
   tried and abandoned for force-law errors. This is a cautionary precedent.
5. **SILVA 2024** and **OpenFOAM GAMG**: a Poisson solve on a moving
   Voronoi or polyhedral mesh in CFD (pressure, not gravity; SILVA without
   multigrid).

## Open gaps in this search
- FVMHD3D and VULCAN/2D gravity solver types were not determined.
- No AREPO-2 method paper was found; its gravity solver is unknown beyond
  "optimized gravity solver".
- The DISCO follow-up papers were not exhaustively checked for self-gravity.
- CFD/plasma ALE codes (e.g. LLNL/LANL ALE codes with electrostatics) were
  not surveyed beyond the items above.
