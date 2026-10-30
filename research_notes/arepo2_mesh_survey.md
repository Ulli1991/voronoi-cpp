# AREPO branches: how the Voronoi mesh is built and updated, compared with voronoi-cpp

Survey date: 2026-10-29. Repository: `/u/uli/codes/arepo`, read only (`git show`, `git grep`, `git log`). No branch was checked out or modified.
Reference copies of the nextgeneration files are in
`/tmp/claude-28632/-raven-u-uli-voronoi-cpp/2fada6d8-a9b5-4727-917f-6d817979c190/scratchpad/arepo-survey/`.

**Labels.**
- **[V]** = verified in code at the cited `branch:path:line`. Line numbers are for the branch head on the survey date.
- **[I]** = my inference.

Short branch names used below:
- `ng` = `origin/Arepo2-nextgeneration`
- `soa` = `origin/Arepo2-nextgeneration-SoA`

---

## (a) Branches and lineages

| Lineage | Branches | Mesh algorithm | Status (last mesh commit) |
|---|---|---|---|
| **A. Classic global Delaunay** | `master`, `Arepo2`, `Arepo2_dev`, `Arepo2_dev_integer`, `arepo2_integer_coord`, `Arepo2-Vec3`, `Arepo2-NewGPU`, `Arepo2_GPU_manager`, `CUDA_Implementation`, and most physics branches | Incremental Delaunay tetrahedralization with exact predicates. Voronoi faces are derived from Delaunay edges. Ghost points across MPI domains. | Production |
| **B. newmesh prototype** | `Arepo2newmesh` (Springel, 2024-01-13 to 2024-02-01) | Per-cell clipping of a cube by bisector half-spaces (Voro++ style), in plain double. | Test only. It runs after the classic mesh and only compares volumes. |
| **D. GPU per-cell clipping inside classic AREPO** | `Arepo2_WD` (Pakmor, 2026-08-10 to 10-29; also `Arepo2-IDORT-dev`) | Port of C's clipper (`VORONOI_CELL`): canonical sort, closure certificate, retirement, graph (previous-CSR) candidates, CUDA/HIP warp-per-cell kernels; per-cell hydro on GPU | Most advanced; production-oriented (WD runs, zoom tests) |
| **C. nextgeneration per-cell clipping** | `ng` (head 2026-08-24) and `soa` (2026-10-28, under active refactoring) | Same clipper as B, matured. Integer positions, tree-walk gathering, retry with a larger radius. No Delaunay structure anywhere. | The production mesh of the new code architecture. The classic Delaunay code is absent from these branches. |
| C, side branches | `-generalmesh` (Migelo 2025-05) | Refactor into a generic `mesh` base class plus `voronoi`/`cartesian` subclasses. Mainline later adopted it (`src/mesh/`). | Merged in spirit. |
| | `-mhd1`, `-pureA`, `-bininvhydro` | MHD vector-potential fits and flux changes. No change to construction. | Physics |
| | `-starformation` (Sunmyon, 2026-09/10) | Same clipper plus fixes: drift before search, free-stack sizing, overlap volume from polyhedron vertices, overflow diagnostics, remote refinement donors. | Most robustness-oriented fork |
| | `-gputree`, `-gpusum(-bench)`, `-hipfft(-bench)`, `-heffte` (Hecker 2025) | GPU gravity tree, direct summation and PM FFT (HIP/CUDA). Nothing for the mesh. | GPU work, but not on the mesh |
| | `-anttitest`, `-mihatest`, `-pureA` | No mesh-construction changes (empty diff in `src/voronoi`). | |

Diff evidence for C side branches [V]:
- `git diff --stat $(git merge-base X ng) X -- src/voronoi src/mesh src/gpu`.
- For example, `-gputree` changes only `src/gpu/gpu.hip`, `src/gpu/gpu.h` and `voronoi.h`; its commits are "Get tree running on unified memory GPU" and "Get direct summation on GPU working".

---

## (b) Answers to questions 1–7

### Lineage C (nextgeneration / SoA): the important one

#### 1. Algorithm, predicates, degeneracies

**Construction is per-cell clipping [V].** `soa:src/voronoi/cell.cc:19-132` `construct_cell`:
- Sort all gathered neighbours by `r2` (`:26`).
- Initialise a cube of half-width `Rmax` = search radius (`cell.h:126-315`). The cube faces carry `pindex = -1..-6` and `CubeDist[]`.
- Clip nearest first with `clip_with_point` until the security criterion holds:
  ```cpp
  if(Rmax2 > 0 && r2 >= 4.0 * Rmax2)  return 1;          // cell.cc:139
  ```
  Here `Rmax2` is the current largest squared vertex distance, recomputed after every clip (`cell.cc:446-467`).
- If the neighbour list runs out before the criterion fires, the build fails: `return false; // we did not succeed - need larger neighbor region` (`cell.cc:131`).

**Data structure [V]** (`cell.h:55-123`):
- A winged-edge polyhedron with `edge{v1,v2,fl,fr,next_edge_l/r,prev_edge_l/r}`, `facet{pindex, ei}` and `vertex{vector<double> xyz; float r2; deleted}`.
- `locindex` is `unsigned short`. Capacities: `MAX_NFACES/EDGES/VERTICES = 1024`; free stacks of 100.
- The cell object is created per call: `cell CellConsruct(Sp);` (`soa:voronoi.cc:191`). That is about 1024 × (faces + edges + vertices) of stack, i.e. tens of kB per thread [I].
- Commit `ng` 2026-07-26 says why `locindex` was widened: "in complex 3D runs, it happened that NEdges >= 255 was reached (causing the mesh construction to malfunction), for now internal locindex has been promoted from unsigned char to unsigned short".

**Clip kernel [V]** (`cell.cc:135-470`):
- A vertex is outside if `dx * v - d2 > 0` with `d2 = 0.5*r2` (`:144`, `:164`). This is a plain double test with **no tolerance**.
- Faces whose vertices are all outside are deleted first (`:181-221`).
- Then every cut ring is processed (`while(einit < NEdges)`, `:227`). This handles several disjoint cut loops for one plane.
- Each new vertex comes from edge interpolation, `t1=(d2-s2)/(s1-s2)` (`:316-321`), with `Terminate` if `t1` is outside [0,1].

**No exact predicates, no symbolic perturbation, no tolerance [V].**
- `git grep -iE 'orient3d|insphere|mpz|long double|perturb'` on `soa`/`ng` `src/voronoi src/mesh` finds only Riemann-solver and least-squares hits.
- Commit `d2fbd6a02b` (ng, 2026-04-21, "more robust handling of degenrate Voronoi cells") **removed** the earlier tolerance. Before, a cut was cancelled if `out2max < ROUND_OFF_THRESHOLD * d2`, and the stop test was `r2 > 4/(1+eps)^2 Rmax2`. The commit replaced this with the exact-compare version above, plus deletion of fully cut faces and multi-ring cutting.
- [I] The design choice is to let tiny or degenerate faces appear (area ≈ 0, see Q3) instead of guarding them.

**Coordinates [V].**
- Positions are integer (`IntPos`, `MyIntPosType`).
- Every neighbour offset is computed as an exact integer periodic difference and then converted to double: `Sp->nearest_image_intpos_to_pos(ngb->IntPos, walk_data.IntPos, posdiff)` (`soa:voronoi.cc:230`).
- So `dx` is exact up to the double conversion, but all clipping arithmetic is floating point.

**Tie-breaking [V].** For equal `r2` the sort compares `dx` components, with a direction set by the generator's octant (`IntPos[0] & MSB`, `IntPos[1] & MSB`), the "sector" (`cell.cc:26-94`).
- In 2D the initial square is also oriented by sector (`cell.h:318-347`). In 3D `sector` is ignored by `init_cell` (`cell.h:126`).
- The purpose, from commits `a3faa2a1d0` and `d3a26a3bda` ("changes to make static results in 2D symmetric", "now also symmetric for moving mesh", Dec 2025), is mirror symmetry of the solution about the box centre in test problems.
- This is **not** a combinatorial consistency rule: two cells are not guaranteed to agree on a degenerate (cospherical) vertex [I].

#### 2. Topology reuse and local timesteps

**Nothing topological is reused [V].**
- The only state carried between steps is the scalar `MaxVertexDistance` per cell (written at `soa:voronoi.cc:197`). It predicts the search radius: `search_radius = SEARCH_ENLARGE_FACTOR * MaxVertexDistance` with `SEARCH_ENLARGE_FACTOR 2.3` (`soa:src/data/constants.h:56`, `voronoi.cc:154,183`).
- On failure the cell gets `Volume = 0`, then `MaxVertexDistance *= 1.26` (a volume factor of 2), and it goes into a compacted retry list (`voronoi.cc:101-114`). This repeats until no cells remain, with `MAX_VORONOI_ITER 100` (`voronoi.h:33`).
- Initial `MaxVertexDistance` is `len / partcount^(1/3)` from the first tree node with more than 25 particles (`soa:src/main/init.cc:490-516`).

**Rebuilt cells [V].**
- `Mesh->construct_mesh(C, All.HighestSynchronizedTimeBin)` (`soa:src/main/run.cc:145`) rebuilds only cells in synchronised (active) time bins: `gas_parallel_up_to_timebin` (`voronoi.cc:49`).
- Inactive cells keep their stale neighbour list and geometry. They receive fluxes from active neighbours (Q3).
- Neighbour positions are drifted to the current time; the starformation fork had to add an explicit drift (`b5ba8f8a67` "Drift mesh search geometry before Voronoi construction").
- After construction, ghost-cell copies of `Volume/Center` are refreshed (`run.cc:153`).

`VORONOI_DYNAMIC_UPDATE` still appears in `examples/*/Config.sh` ("keeps track of mesh connectivity, which speeds up mesh construction"). These are leftovers from lineage A and are unused by the C code [V: no reference in `soa:src`].

#### 3. Faces: once or twice, consistency

**Each cell computes all its own faces [V].** `cell_geometry` (`cell.cc:472-624`) gives, per face:
- the area from a triangle fan over the face's vertices;
- the face centroid, stored as an **integer** position `FaceCenter = IntPos + delta` (`:594-597`);
- the neighbour index.

The cell volume is a sum of pyramids with the **analytic height** `0.5*sqrt(r2)` (or `CubeDist` for cube faces) (`:564-566`), not a height derived from the vertices.

Faces with `area == 0` are silently dropped (`if(area > 0)`, `:560`). Storage is `MAX_NGB_PER_CELL 32` faces per cell (`soa:src/data/dtypes.h:146`); exceeding it calls `Terminate`.

So a face shared by two active cells is **built twice**, independently. There is **no symmetry check and no repair** [V: no such code in `src/voronoi` or `src/mesh`].

**Fluxes depend on the branch.**
- **`ng`** (`ng:src/mesh/mesh_fluxes.cc:84-91`, 161-166). By default the face is processed once when both sides are active:
  ```cpp
  // Both are active, so face will be appearing twice. We only do it here if the left side has the smaller ID.
  if(particle_L->ID.get() >= particle_R->ID.get()) continue;
  ```
  The flux is applied with opposite signs to both cells under per-particle locks (`access.get_lock()`). This is conservative to round-off.
  - With `COMPUTE_CELL_FLUXES_INDEPENDENTLY` ("each cell computes all its cell fluxes (i.e. no locking required, but faces are computed twice)", `Template-Config.sh:25`), each active cell computes its own flux. A pair of active cells then computes the Riemann problem twice, from two independently built geometries.
  - [I] Consequence of the default mode: if roundoff makes the neighbour relation asymmetric (L lists R, R does not list L, and ID_L > ID_R), that face gets no flux at all. This stays conservative but loses the face.
- **`soa`**: the face-once branch is gone. The code always runs in independent mode (`soa:src/mesh/mesh_fluxes.cc:176-195`):
  - "if the right cell is active as well, it will compute its own flux, otherwise we need to add this in an order fashion to the inactive cells".
  - Fluxes for inactive or remote cells are buffered and **sorted by (origin, ID) before being applied**, which makes the result deterministic (`:576-588`).
  - [I] For active–active pairs, conservation holds only to the extent that both cells built the same area, face centre and states. This is not bitwise.
  - [I] The motivation is lock-free, race-free SoA/GPU-ready code ("retired some more race conditions", 2026-10-27).
- Face normal: from the generator difference (`prepare_face_geometry`).
  - **Bug in `soa` head [V]:** `soa:src/mesh/mesh_fluxes.cc:246` reads `nearest_image_intpos_to_pos(pR->IntPos[iL].da, pL->IntPos[iR].da, geo.n.da)`. The indices `iL`/`iR` are swapped, which looks like a slip from the SoA refactoring.
  - The face velocity uses the stored face centre and the vertex velocities, with a limiter: `if(cc > 0.9*geo.dist)` (`:294-331`).

#### 4. Neighbour gathering and cost

**Tree walk [V].**
- A generic `tree_walk<ngbtree, ngb_accumulator, ...>` (`soa:voronoi.h:53`) over the gas neighbour tree, in integer coordinates.
- Nodes are rejected per axis with integer box tests against the node's tight `center_offset_min/max` extent (`soa:voronoi.cc:250-298`).
- Every particle inside the sphere of radius `2.3 × MaxVertexDistance` is appended to a `ngblist` of capacity `MAX_VORONOI_NGB_NUMBER 100000` (`voronoi.cc:216-248`).
- The **whole list** is then `std::sort`ed by `r2` with the tie-break comparator (`cell.cc:26`).
- MPI: remote neighbours are imported ghost cells that sit in the shared-memory tree. One rank per node manages shared lists (`MPI_Bcast` of list pointers, `voronoi.cc:40-47`).

**Cost model [I].**
- A sphere of radius 2.3 R_max holds about (2.3)^3 × (R_max/r_cell)^3 × ... ≈ 80–150 candidates. Most are never needed for clipping: the stop fires around 2 R_max.
- The full sort is O(k log k), but clipping cost is dominated by the ~40–60 planes tested.
- Each clip is O(#vertices + #faces + #edges), with linear scans over all slots including deleted ones (`for vi < NVertices`, `for fi < NFaces`, `for ei < NEdges`).
- A missed opportunity: the walk stops only when a *listed* neighbour exceeds 2 R_max. If all listed neighbours are closer than 2 R_max (possible when 2 R_max is only slightly above the search radius), the build fails, even if 2 R_max ≤ search radius would already certify the cell. [V for the logic at `cell.cc:102-131`; the impact is I.]
- `Vertices[].r2` is a `float` (`cell.h:68`), and the stop test `r2 >= 4*Rmax2` has no safety margin. [I] There is a ~1e-7 relative risk of stopping one plane too early. Our code uses a `1.0001f` factor (`shell_voronoi.hpp:329`).

**Timers and logs [V]:**
- "VORONOI: %lld cells constructed (took %g sec), %lld cells remaining" (`voronoi.cc:129`)
- "Total constructed cell volume = %g, Cell construction took = %g secs" (`:146`)
- The starformation fork adds `CPU_MESH` timer coverage (`4b71ae759b`).
- No throughput numbers were found in code, docs or commits [V: grep for speed/fast/per sec].
- High-water counters `max_NFaces` etc. are tracked but not printed (`cell.cc:108-124`).

#### 5. GPU work

**Mesh construction, fluxes and gradients: no GPU kernels in any C branch [V].**
- `soa:src/gpu/` is a portability layer: CUDA/HIP/SYCL/CPU backends, `gpu_manager`, `gpu_memory_manager`, and `gpu_algorithms.h` with `gpu_parallel_for` and `gpu_parallel_for_blocks` ("one BLOCK per index ... e.g. one stiff-ODE cell per block"). It was added 2026-07-31/08-01 ("included GPU manager", "fixed compilation for A100/FreyaGPU", "compilation on ViperGPU").
- Only `src/main/main.cc` and `simulation.h` reference it [V].

**The SoA refactor (Oct 2026) is clearly GPU preparation [V/I].**
- Particles are stored in blocks of `PDATA_SIZE = 1<<16` (`dtypes.h:143-144`).
- The neighbour table is per block: `ngb_base<PDATA_SIZE*MAX_NGB_PER_CELL>` holds `pindex`, `FaceCenter` (int), `area` and `gradweight[]` (`sph_particle_data.h:61-78`).
- Its index layout switches with the target:
  ```cpp
  inline size_t get_ngb_index(int i, int n) {
  #ifdef GPU
    return i + ((size_t)n) * PDATA_SIZE;      // neighbour-major: coalesced over cells
  #else
    return MAX_NGB_PER_CELL * ((size_t)i) + n;
  #endif }                                     // sph_particle_data.h:411-418
  ```
- Hecker's branches (`-gputree`, `-gpusum`, `-hipfft`, `-heffte`) put gravity (tree, direct sum) and PM FFT on GPU (HIP, unified memory on MI300). They contain no mesh code [V].

#### 6. Refinement and derefinement

**Refinement** (`ng:src/voronoi/voronoi_execute_refinements.cc`, overview comment `:1-55`).
- The new point goes **at the cell's farthest Voronoi vertex**: `np->IntPos = p->IntPos + MaxVertexPosition` (`:775-783`). `MaxVertexPosition` is the vertex found by `construct_cell` (`Rmax_vertex_index`).
- Neighbouring cells are never refined together. Conflicts are resolved with a random number (`refinement_identify_sites`).
- Sequence:
  1. Insert the new points into the neighbour tree.
  2. Build the new cells with **only old points** as neighbours, and keep those neighbour lists.
  3. Rebuild the new cells with new neighbours included.
- **Overlap computation** (`:415-523`):
  - For each old neighbour of a new cell, rebuild the new cell, then clip it further with the **old neighbour's own planes**.
  - The old planes are offset to the old generator through `clip_with_point(n, &c)`, i.e. `d2 = 0.5 r2 + c·dx` (`cell.cc:146-147`).
  - The intersection polyhedron's volume gives the fraction `fac = vol_overlap / Volume_old` of mass, momentum and energy to transfer.
  - **Bug [V/I]:** the volume uses `cell_geometry()`, which assumes the faces are bisectors of the origin, with pyramid height `0.5*sqrt(r2)`. That is wrong for the shifted planes. The starformation fork fixed it ("Compute refinement overlap volume from polyhedron vertices", `11ae8b1d39`, a new `polyhedron_volume()` that tetrahedralises from the vertex mean). Conservation still holds because donors lose exactly what the new cell gains; only the split fractions are wrong.
- Old donor cells that are active are then rebuilt, and their gradients and primitives recomputed (`:126-195`). Everything is local and nothing is rebuilt globally.
- `soa` has not yet ported refinement: it still calls the old `construct_mesh(todolist, ntarget)` API, and `voronoi.cc:199` references a `Vertices` not in scope [V; I that `REFINEMENT` does not compile on `soa` yet].

**Derefinement** (`ng:src/voronoi/voronoi_execute_derefinements.cc`).
- Allowed only if no neighbour is also derefining and **all neighbours are active** (`:151-200`).
- Redistribution (`derefine_cell`, `:233-340`): for each Voronoi neighbour q of the removed point p, q's cell is built twice **from p's neighbour set only**, once without p (`VolAfter`) and once with p (`VolBefore`). It starts from a cube of half-width `2*maxlen`, recentred on q via `init_cell(2*maxlen, ngbpos)` (`cell.h:396-410`).
  - q gains the fraction `(VolAfter−VolBefore)/Vol_p` of p's conserved quantities.
  - [I] This is exact, because the part of p's old cell that q inherits is bounded only by p's natural neighbours.
- The affected active neighbours are then rebuilt (`:75-116`).
- 2026-08-09: "augmented derefinement slightly so that conserved quantities of dissolved cells can land on different nodes".

**Gradients [V].** `prepare_gradient_operator` (`soa:src/mesh/mesh_gradients.cc:150-290`) runs once per mesh build.
- It solves an area-weighted least-squares problem (4 unknowns including a constant) on cell centroids, and stores a per-face **gradient weight vector** `gradweight[nidx]` (`:265-273`).
- Any field gradient is then `grad += (ValueNgb - Value) * gradweight[nidx]` (`:59`): one fused multiply-add per face per field.
- Faces with `area < MIN_AREA_FRACTION_FOR_SLOPE_LIMITING*Surface` are ignored in the limiter (`:61-62`).

#### 7. Clever or unusual points (lineage C)

1. A **radius predictor from last step's `MaxVertexDistance`**, with retry ×1.26 and compaction of only the failed cells. The cube half-width equals the search radius, so failure detection is automatic: a surviving cube face pushes `Rmax` beyond the limit.
2. **Integer generator coordinates and exact integer offsets.** Face centres and cell centroids are stored back as integers relative to `IntPos`.
3. **Analytic pyramid heights** (`0.5|d|`) for volumes, avoiding vertex roundoff in the height.
4. **Geometric conservative remap by polyhedron intersection** for refinement, and natural-neighbour-local "with and without" cell builds for derefinement.
5. **Precomputed per-face gradient weights** in the neighbour table: one linear operator reused for all fields and all limiter passes.
6. A **deterministic, sorted application** of fluxes to inactive and remote cells.
7. The **neighbour table in SoA blocks** with a switchable layout, `i + n*PDATA_SIZE` on GPU.

### Lineage B (Arepo2newmesh, 2024): the prototype of C

All [V], from a sub-survey of `origin/Arepo2newmesh:src/voronoi/voronoi_newmesh.cc` and `voronoi_newmesh_search.cc`.

**What the branch is.**
- Six commits by Springel, 2024-01-13 to 02-01: "added a neighbor search in first preparations for testing a new mesh construction algorithm", and others.
- **`predicates.cc` and `mpz_extension.cc` are not new.** They belong to the legacy Delaunay code and are not called by the new code.

**Algorithm.** Cube init and winged-edge clipping, identical in structure to C.
- `locindex` is `unsigned char`; capacities are 255; there are no `prev_edge` links.
- Only one cut ring is handled per plane.
- Stop rule: `if(r > 2.0 * Rmax) return 1;` (`:505-506`).
- Neighbours come from the classic tree, `ngb_treefind_variable_threads` with `h = 1.01*MaxDelaunayRadius`, i.e. from the *old Delaunay mesh* (`search.cc:65,235`).

**Status: test harness only.**
- Behind `VORONOI_NEW_MESH`, `create_new_mesh()` runs after the classic `create_mesh()`.
- It only prints volume mismatches, `if(fabs(SphP[i].Volume - vol) > 1.0e-5*vol)` (`voronoi_newmesh.cc:92-93`), and then frees the data.
- There is no exact arithmetic.
- A bug: when a cube face survives, `check_cell` reads `NgbTab.ngbs[p<0]`. C fixed this with `CubeDist`.

**Path from B to C.**
- The code moved into `ng:src/voronoi/voronoi.cc` ("Initial commit of new (still experimental) code architecture", 2025-01-13).
- It was split into `soa:src/voronoi/cell.{h,cc}` in Oct 2026.

### Lineage A (classic global Delaunay: `master` C, `Arepo2*` C++)

Source: a sub-survey that read the code, plus my spot checks. File:line references are `master:src/...` unless noted.
- `Arepo2:src/voronoi/*.cc` is the same algorithm as master, with only cosmetic C→C++ differences (`printf`→`arepo::print`).

**1. Construction [V].** `create_mesh()` (`voronoi.c:82-513`).
- **Starting tetrahedron.** One bounding tetrahedron (`tetra_incircle = 1.5*box`) and a point at infinity (`voronoi_3d.c:83,123`).
- **Insertion.** Serial incremental insertion in particle order (Peano–Hilbert sorted). Each point-location walk starts at the previous insert's tetrahedron: `tlast = insert_point(&Mesh, skip+i, tlast)` (`voronoi.c:269`).
- **Point location.** `InTetra` (`voronoi_3d.c:2905`) solves for barycentric coordinates in double, using an `INSIDE_EPS` band. Only near zero does it fall back to `Orient3d_Exact`.
- **Exact predicates.** Coordinates are mapped into [1,2), so all 52 mantissa bits form an integer grid:
  - `xx = (x - CentralOffset)*ConversionFac + 1.0; ix = double_to_voronoiint(xx); xx = mask_voronoi_int(xx)` (`voronoi_3d.c:3894-3910`, `USEDBITS 52`).
  - Because the doubles are masked to that grid, the double filter and the exact test see identical points.
  - `InSphere_Errorbound` and `Orient3d` are double determinants with a static bound `1e-14*sizelimit`. When undecided they go to the GMP `mpz` versions `InSphere_Exact` and `Orient3d_Exact`.
  - Circumcentres fall back to an exact `mpz` determinant if the double solve fails (`voronoi_3d.c:3695-3699`).
  - Shewchuk's `predicates.cc` exists in some branches but is never called. It was removed in Arepo2 on 2026-07-10 ("remove predicates", Zier).
- **Degeneracies [V].** Handled by flips, not by perturbation (`insert_point`, `voronoi_3d.c:1480-1871`):
  - 1→4 for a point inside a tetrahedron, a face split for a point on a face, an edge split for a point on an edge;
  - legalisation by 2-3, 3-2 and 4-4 flips (`convex_edge_test`), driven by a check stack and a free stack of tetrahedra;
  - 2D uses 1-3 and 2-4 flips.
- **MPI and periodic ghosts: "repeated searching"** (`voronoi_ghost_search.c`).
  - For each tetrahedron with a local active vertex that is not yet certified, a tree search runs around its circumcentre with radius `1.0001*R_cc`, bounded by `Hsml` (`:173-256, 482-516`).
  - The points found are inserted in the next round.
  - `Hsml` starts at `1.01*MaxDelaunayRadius` (twice the largest previous circumradius) and grows ×1.3 while any tetrahedron is undecided. The loop allows at most `MAX_VORONOI_ITERATIONS=500` rounds.

**2. Reuse and timesteps [V].**
- **No tetrahedra survive a step.** The mesh is built again at every (sub)step (`run.c:419`) and freed after fluxes and (de)refinement (`run.c:336-342`).
- **Sub-steps build a partial mesh.** It contains the active cells plus whatever the circumsphere search pulls in. Geometry is accumulated only for active cells (`TimeBinSynchronized` checks, `voronoi_3d.c:605-678`).
- **`VORONOI_DYNAMIC_UPDATE`** keeps only the per-cell **connection lists `DC`** (`voronoi_dynamic_update.c`).
  - They seed round 0 with each active cell's previous neighbours, including foreign ones, which cuts the number of ghost rounds.
  - The LSF gradients require them (`voronoi_gradients_lsf.c:21`).
- `TIMESTEPS_WAKEUP` (`Arepo2_dev*`, `voronoi_wakeup.cc`) promotes woken cells mid-build and reruns convergence.

**3. Faces [V].**
- **One face per Delaunay edge per task.** `process_edge_faces_and_volumes` (`voronoi_3d.c:358-756`) walks the ring of tetrahedra around each edge. An `Edge_visited` mask ensures each face is made once, and an "inconsistency" check is part of the walk.
- **Volume** is the pyramid sum `(1/3)*area*h` with the analytic height.
- **Across tasks** the same face can exist on two tasks. `face_check_responsibility_of_this_task` (`finite_volume_solver.c:2306-2360`) makes exactly one task compute it by an ID rule.
- **Flux is applied once with ±fac** (`:974-996`). Foreign partners go through `FluxList` over MPI.
- **Consistency by construction:** one Delaunay tessellation implies one face set.

**4. Costs [V].**
- Timers `CPU_MESH` (`insert`, `findpoints`, `cellcheck`, `geometry`, `exchange`, `dynamic`).
- Log line `"VORONOI: iter=%d: %llu local points, points/sec/task = %g"` (`voronoi.c:251`).
- No runtime fraction is stated in the code.

**5. GPU [V].** `Arepo2`, `Arepo2_dev`, `Arepo2-NewGPU`, `Arepo2_GPU_manager` (Zier, 2024-09 to 2025-12).
- **The GPU only consumes a CPU-built mesh.** `construct_mesh_for_gpu` copies the VF/DP topology to the device as CSR plus SoA (`offset_interfaces`, `interfaces_list`, `nx[]`, `face_area[]`, `face_dt[]`) (`src/gpu/gpu_finite_volume/gpu_finite_volume_helper.cu:30-72`).
- **Kernels are for M1 radiative transfer only:** Rusanov interface fluxes, LSF gradients, thermochemistry (`src/MRT/gpu/gpu_files/*.cu`).
- **Notable flux reduction:** a per-cell **gather over its CSR face list with sign `fac = (left != cell) ? 1 : -1`, without atomics** (`RT_gpu_finite_volume.cu:289-315`).
- Faces whose active partner is on another task are computed by both tasks, which avoids sending results back.
- `CUDA_Implementation` (everbeek, 2023-10 to 12, last commit "BROKEN | WIP"): RT face kernels only, abandoned.

**6. Refinement [V]** (`voronoi_refinement.c`, `voronoi_derefinement.c`).
- **Refine.** The new point is offset `0.025*rad` from the parent, in a random direction or toward the most distant neighbour (`:207-251`).
- A local mini-Delaunay (`DeRefMesh`) gives the split volumes. Conserved quantities are split by `voli/(voli+volj)`.
- **Derefine.** The neighbourhood mesh is built with and without the cell, and the conserved quantities go to neighbours by volume gain.
- **The global mesh is not updated locally.** It is rebuilt at the next step.

**7. Integer-coordinate branches** (`arepo2_integer_coord`, Zier 2026-06; `Arepo2_dev_integer`, Zier/Pakmor 2026-07/08) [V].
- Positions become N-bit unsigned `IntPos`, with periodic wrap by unsigned overflow (`src/data/intposconvert.h:9-50`).
- Mesh changes are plumbing only: DP coordinates come from `IntPos`, and centres are stored as `IntCenter`.
- The predicates are unchanged (52-bit mantissa map plus GMP).

### Lineage D (`origin/Arepo2_WD`: `VORONOI_CELL` + `FINITE_VOLUME_SOLVER_PER_CELL[_GPU]`)

Author: Ruediger Pakmor (alias "Varan"). 118 commits, 2026-08-10 to 2026-10-23; also contained in `Arepo2-IDORT-dev`. Not in `AMR_GRID`, as checked with `git branch -r --contains`; see section (e).

[V] from `src/voronoi_cell/README_VORONOI_CELL.md` (707 lines), `src/hydro_per_cell/README_FINITE_VOLUME_SOLVER_PER_CELL.md` and `voronoi_cell_gpu.cu`. I read the READMEs and kernel headers, not all 1 MB of code.

**This is the closest relative of voronoi-cpp: a GPU per-cell clipping builder inside the classic AREPO.**
- **Algorithm.** It ports lineage C's clipping kernel ("ported from the nextgen Arepo prototype", README:8) into the classic code. The Delaunay tessellation is never built (`Mesh.DT/DTC/DTF stay empty`, README:12-16). The classic `Mesh.DP/VF` outputs are filled, so the downstream hydro runs unchanged.
- **Rounds** (README:27-59):
  1. Exact candidate **count** inside `rcut` (strict `r2 < rcut²`).
  2. Cells with more than `K = MeshCellCandidateLimit` candidates are **deferred**.
  3. Gather, then a **canonical sort** (distance, then position/ID tie-breaks evaluated at the cell's sector), then clipping until `r2 >= 4*Rmax2`.
  4. **Closure certificate:** if the list runs out, the cell is accepted when the seed cube is gone and `rcut` already implies the stop criterion. Otherwise `rcut` grows ×1.26. This fixes the missed early acceptance noted for C in Q4.
  5. **Retirement:** points whose cell and all face neighbours are built are dropped from later gathers. In a zoom run this cut the largest candidate count from ~290,000 to ~7,700 (README:494).
- **Graph candidate source** (`MeshCandidateSource 1/2`, README:61-137). This is **topology reuse for gathering**.
  - Candidates come from the previous mesh's **connectivity CSR**: the depth-3 neighbour ball, raised to depth 4 if a face partner lands on the outermost level. No tree walk is needed.
  - "a cell can gain a face partner that the previous mesh's graph places three hops away" (seen in a zoom run), so depth 2 is never accepted.
  - Verification knobs rebuild every n-th cell by tree walk and compare.
- **GPU** (`voronoi_cell_gpu.cu`).
  - Kernels: `vcell_gpu_clip_block_kernel` ("One cell per warp, the cell's scratch in shared memory and the lanes sharing its work ... 164 kB per SM, so ~7 cells: fewer cells resident, each far faster", `:220-250`), `vcell_gpu_clip_smem_kernel` (one thread per block), and `vcell_gpu_build_block_kernel` / `_tail_kernel`.
  - Device capacities are a compact 128/384/256 faces/edges/vertices, about 16–22 kB of scratch per cell (README:505, 593).
  - Cells go in **waves** (`AREPO_VCELL_WAVE`, default 256; going 256→4096 "cut the feeder's clip 5.9x and the device's cost over the run from +21.7 % to +0.4 %", README:188-192), with a 2-slot pipeline that overlaps the host walk with device clipping.
  - A **device tail** takes larger capacities, and remaining failures go home to the host.
  - The neighbour tree can be built and sorted on the device.
- **Determinism.** The mesh is bitwise invariant to rerun, K, schedule and host vs device. It is pinned with `-ffp-contract=off` and `-fmad=false`, and per-build **order-independent XOR checksums** of volume and Rmax are logged (README:425-455).
- **Faces.**
  - The lower-ID side emits the VF. In 3D "a face exists iff its fan-summed area is > 0".
  - "Exact-degeneracy slivers (planes touching a cell exactly; areas ~1e-30) may survive on one side only; they are below every consumer threshold" (README:400-413). So it is **not** topologically watertight, by design.
- **Hydro per cell** (FVS README:16-34).
  - "A face between two active cells is computed twice, once per cell, with the fixed convention L = lower ID, R = higher ID; each cell uses the area and centroid of the face it built". Conservation holds to round-off.
  - Inactive partners get flux records, which is exactly conservative.
  - Commit 2026-10-19: "Give up on exact face symmetry, massively simplifies hydro code".
  - GPU keeps the mirror of primitives, gradients and fluxes on the device. The host path costs +10% (hydro 696 s vs 487 s).
- **Performance.** "Construction cost is within ~1.3× of the classic Delaunay construction on a 400-rank cosmological zoom"; (de)refinement volumes are "3–7× cheaper" (README:572-577). The remaining cost is dominated by gather and point exchange.
  - No device throughput in cells/s is given anywhere [V: grep].
  - [I] Warp-per-cell with ~7 resident cells per SM, plus double-precision winged-edge scratch, is likely well below our 12–16 M cells/s. This is unmeasured.

---

## (c) Same as us / different from us

C = `ng`/`soa`; D = `Arepo2_WD`; A = classic.

| Aspect | AREPO | voronoi-cpp | Same / different |
|---|---|---|---|
| Construction | C, D: per-cell clipping, nearest first, stop at r ≥ 2 R_max. A: global incremental Delaunay. | Per-cell clipping (`shell_voronoi*.hpp`), stop at 2R; plus a persistent Delaunay (`tet_mesh.hpp`) | **Same as C/D** for clipping. Our tet path resembles A, but persistent and with flips. |
| Polyhedron | C/D: winged edge (`fl, fr, next/prev_edge`), free stacks, linear scans. Capacities: C 1024; D device 128/384/256 (~20 kB). | Vertex adjacency (3 planes + 3 neighbours), 48–64 vertices, 64-bit outside mask; ~1.2–1.8 kB per cell, interleaved in shared memory | **Different**: ours is ~10× smaller, so many more cells resident per SM |
| GPU mapping | D: warp-per-cell in shared memory (~7 cells/SM) or thread-per-cell with global scratch; waves of 256–4096 cells staged from the host walk | Thread-per-cell in shared memory (lean), warp-per-cell variant (`shell_voronoi_warp.hpp`); everything on device | Same idea; ours is fully device-resident |
| Neighbour gathering | C: tree walk of the full 2.3 R_max sphere, then `std::sort`. D: exact count, deferral above K, canonical sort, or graph ball from the previous CSR. | Uniform grid; rings or a sorted stencil; 16–32-entry nearest-first buffer; no full sort | **Different**: our gather is streaming. D's graph source is topology reuse that we do not use for clipping. |
| Certification | C: needs a listed neighbour beyond 2 R_max, otherwise retry ×1.26. D: adds a **closure certificate** (rcut ≥ 2 R_max after the list is exhausted). | The walk stops at the first ring or stencil entry beyond 2R; self-certifying in one pass | Equivalent to D; better than C |
| Predicates | C/D: plain double, no tolerance (D pins FP contraction for bitwise CPU/GPU). A: 52-bit mantissa grid, static filter, GMP fallback. | Clipping: float/double with tolerance `64 ε sqrt(d² R²)`. Tets: exact integer + SoS (`tet_flips.hpp`). | Our tet path ≈ A's robustness, plus SoS; clipping is similar to C/D |
| Degeneracies / ties | C/D: sector-dependent sort (mirror symmetry); D: canonical total order (determinism); zero-area faces dropped, one-sided slivers tolerated. A: flips 2-3/3-2/4-4. | Owner rule; SoS flips; double fallback | Different goals: symmetry/determinism vs consistency |
| Topology reuse | C: only the scalar R_max. D: R_max and the **previous CSR as the candidate source**. A: DC lists seed the ghost search. | `tet_mesh` persistent tets + flip repair | Ours goes furthest (full topology); D's graph gather is the lighter middle ground |
| Faces | A: one per Delaunay edge. C `ng`: once (lower ID), with locks. C `soa`, D-hydro: computed twice, once per cell (L = lower ID), conservative to round-off. D-mesh VF: lower-ID side emits. | Stored once by the owner (lower slot); one flux ±; `face_geometry.hpp` one thread per face | Ours equals A's guarantee at C/D's speed |
| Flux reduction on GPU | A-RT: per-cell CSR gather with sign, no atomics. D: `fvs_gather/scatter` kernels. | (not yet) | Adoptable |
| Local timesteps | All: rebuild active cells only (A: plus the ghost and neighbour closure). D: native wakeup. | All cells | Theirs is the target behaviour |
| Refinement | C/D: point at the farthest vertex; polyhedron-intersection remap; D "3–7× cheaper" than A's mini-Delaunay | n/a | n/a |
| Determinism tooling | D: XOR checksums per build, GPU-vs-host verify tiers, FP pinning | Flip repair checked against an exact rebuild | Adopt D's checksums |
| Numbers | A: none stated. C: none. D: ~1.3× of classic Delaunay cost on a 400-rank zoom; no cells/s. | A100: clip 12–16 M cells/s; geometry from rings ~300 M; persistent-tet step ~200 M cells/s | We are faster as far as can be compared (I) |

---

## (d) Ranked ideas worth adopting

Ranked by expected value for speed or robustness. The source is in parentheses.

1. **Graph-ball candidate source from the previous mesh (D)**, for the clipping path.
   - Take candidates from the depth-3 ball over last step's neighbour CSR instead of the grid stencil. Accept the cell only if no face partner sits at the outermost ring; otherwise go to depth 4, else fall back to the grid.
   - It needs no grid rebuild and gives far fewer candidates than a 2R sphere, and it is naturally sorted by graph distance.
   - It also covers adaptive / high-dynamic-range point sets, where a uniform grid degrades.
   - We already have `tet_mesh` neighbour rings, so the CSR is free.
   - D's empirical warning applies: depth 2 is **not** enough, because new face partners at hop 3 occur.
2. **Retirement of finished points (D)** for clustered or zoom distributions. Once a point's own cell and all its face neighbours are built, drop it from later candidate lists. Exact, and it bounded the worst candidate count from 290k to 7.7k in their zoom test.
3. **Per-face precomputed gradient weights in the face table (C, `mesh_gradients.cc:150-290`).**
   - Compute the 3- or 4-vector of LSQ weights per face side in the face-geometry kernel.
   - All field gradients and the limiter then become a gather plus FMA.
   - Combine it with A-RT's **per-cell CSR gather with sign (no atomics)** for the flux reduction (`RT_gpu_finite_volume.cu:289-315`).
4. **Radius predictor plus closure certificate plus compaction retry (C, D).**
   - Store R_max per cell. Use it to size the stencil and buffer, and to route cells to the float or double kernel up front.
   - Seed the box at the predicted radius, so a surviving box plane flags "uncertified" for free.
   - Accept when the list is exhausted and the gathered radius ≥ 2 R_max (D's certificate).
   - Retry only the compacted failures.
5. **Determinism tooling (D).** A canonical candidate order (distance, then ID), `-ffp-contract=off` / `-fmad=false` where bitwise CPU/GPU equality matters, and **order-independent XOR checksums** of volume and R_max per build. These are cheap regression guards for our three kernels (shell, lean, warp) and for flip repair vs rebuild.
6. **Analytic pyramid heights (A, C)**: `V = Σ A_f |d_f|/6`.
   - Our `shell_voronoi.hpp:429` sums vertex tetrahedra (`dot(first, cross(previous, x))/6`) instead.
   - Using `|area|·|d|/6` per face saves work and removes vertex error from the height.
7. **Active-only rebuild for local timesteps (A, C, D)**, with deterministic sorted scatter of fluxes to inactive cells (`soa:mesh_fluxes.cc:576-588`). For `tet_mesh`: recompute circumcentres and flip-repair only around moved or active particles.
8. **Refinement and derefinement with the clipper (C, D).**
   - Place the new point at the farthest vertex.
   - Remap conserved quantities by polyhedron intersection, using offset planes `d2 = 0.5|d|² + c·d`.
   - Derefine with "with/without p" builds over p's natural neighbours. D measured 3–7× cheaper than mini-Delaunay.
   - Pitfall (C bug, fixed in `-starformation` `11ae8b1d39`): volumes of offset-plane polyhedra must come from the vertices, not from bisector heights.
9. **Wave / batch pipelining (D)** only matters if construction is host-fed. Their lesson, "small batches are latency-bound: 256→4096 cut cost 5.9×", supports keeping big launches.
10. **Integer face and cell centres relative to integer generators (C, A-integer)**, if we move to integer positions: exact periodic wrap, and 32-bit offsets halve storage.

**Keep what we have (lessons from their failures):**
- C hit `NEdges ≥ 255` in production (2026-07-26 commit). Our capacity overflow → double fallback is right; keep counting overflows. D does the same with a device tail and then the host.
- C and D accept one-sided sliver faces (area ~1e-30) and dropped zero-area faces, and gave up on exact face symmetry in hydro. Our owner rule plus SoS flips gives watertight topology. If we ever build faces from both sides, add a j ∈ N(i) ⇔ i ∈ N(j) check.
- C's stop test uses a `float` vertex r² with no margin. Keep our `1.0001f` margin in every kernel.
- A's 52-bit mantissa grid (filter and exact test on identical coordinates) is the same principle as our snapped integer predicates; keep it.

**Not worth adopting:**
- C's full candidate list plus `std::sort`;
- 1024-slot winged-edge cells and linear scans over deleted slots;
- warp-per-cell with ~20 kB shared memory (D: about 7 cells per SM);
- tree walks for near-uniform point sets on the GPU;
- the mirror-symmetry sector tie-break, unless bitwise-symmetric test problems are wanted.

---

## (e) GPU mesh work by person (Pakmor alias "Varan", O. Zier, plus Hecker for completeness)

Survey of 2026-10-29, read only. **[V]** means verified in code, docs or git metadata; **[I]** means inferred.

### Identities [V]

`git log --all --format='%an <%ae>'`:
- **"Varan" is Rüdiger Pakmor.** The alias commits with his addresses: `Varan <ruediger.pakmor@h-its.org>` (490 commits), `Varan/varan <rpakmor@mpa-garching.mpg.de>` (99), and `varan <ruediger.pakmor@h-its.org>` (44).
  - Varan commits run from 2015-09-19 to 2026-08-05. In 2025-26 they are wakeup, merge and fix commits on `Arepo2_dev`, `Arepo2_SIDM`, `AGN-CGM` and `Arepo2_generalzied_tree`, with no separate GPU line.
  - All of his GPU mesh work is committed as "Ruediger Pakmor" (`rpakmor@mpa-garching.mpg.de`, `rdp@orion01...`).
- **Oliver Zier**: `ozier@mit.edu` (1214), `o.zier@web.de`, `ozier@mpa-garching.mpg.de`, `ozier@login01.frontier.olcf.ornl.gov` (Frontier).

**Branches with 2025-26 commits by them touching mesh, voronoi, gpu, hydro or ngbtree [V].**
The counts come from `git log --all --since=2025-01-01 --source --author=…`, which attributes each commit to one ref only, so they are lower bounds.

| Person | Branches (≈ commit count) | Mesh-relevant content |
|---|---|---|
| Pakmor / Varan | `Arepo2-IDORT-dev` (107), `Arepo2_WD` (28 by first-ref, ~118 in `src/voronoi_cell` + `src/hydro_per_cell`), `Arepo2_mf_agn`/`Arepo2_dev`/`Arepo2_dev_integer` (wakeup, 2026-07), `grwp_caerdydd` (clean-ups), `long_term_disks`, `new_bh`, `Arepo2_SIDM` | **VORONOI_CELL (+ _GPU)** and **FINITE_VOLUME_SOLVER_PER_CELL (+ _GPU)**; `TIMESTEPS_WAKEUP`; `NGBTREE_NODELOCAL` halo tree and graph; `FORCETREE_NODELOCAL_GPU`; `NUCLEAR_NETWORK_GPU` |
| Zier | `MRT_transparent_EOS` (39), `MRT_GPU_cleaned_up` (32), `Arepo2-NewGPU` (30), `grwp_caerdydd` (29), `AMR_GRID` (15), `Arepo2_generalzied_tree`, `Arepo2_dev_integer`, `arepo2_integer_coord`, `Arepo2` | GPU M1 radiative transfer on the CPU-built mesh; GPU abstraction and stack memory manager; integer coordinates and integer trees; `AMR_RT_GPU` (Cartesian AMR, not Voronoi) |

`git branch -r --contains` for key commits:
- VORONOI_CELL infrastructure `c9b08584bd`, GPU clip `03e9f13c47` and warp-per-cell `b689ead0c4` are only in **`Arepo2_WD` and `Arepo2-IDORT-dev`**.
- Hydro-per-cell `3221f27b4d` and the HIP build `ca278bac15` are only in `Arepo2_WD`.
- Correction to section A/D above: `AMR_GRID` does **not** contain VORONOI_CELL (no `src/voronoi_cell`). Its GPU work is Zier's `AMR_RT_GPU` (`33b8aae9e6`, AMR_GRID only).
- Zier's RT-GPU commits (`9506b1a75c`, `09b6624280`) are in `Arepo2` and therefore in almost every Arepo2 child.

### E1. Pakmor: `Arepo2_WD` VORONOI_CELL_GPU and FINITE_VOLUME_SOLVER_PER_CELL_GPU

**Timeline [V]** (commit subjects, 2026):
- 08-10 "VORONOI_CELL: actual mesh construction, first version".
- 08-13 "Make mesh construction more robust. 20 edges max per faces it not enough...".
- 08-14 to 09-16 the node-local halo tree and **graph** (`NGBTREE_NODELOCAL`, "Graph tree: full link symmetry" 08-26).
- 09-17 "one island per GPU" (`7a27babbec`).
- 09-18 "Core of mesh construction: clipping on GPU".
- 09-19 "clipping now in waves", then "Some optimisations, but **still super slow on GPU**. Needs tree as well on there..." (`c66502f9e4`), then "**Building a cell by one warp instead of one thread**" (`b689ead0c4`).
- 09-20 "Treewalk on the GPU".
- 09-23 "ngbtree now fully build on device".
- 09-26/27 "Candidate ball from graph only", "Tail on the device".
- 09-28 "mode == 1 becomes the default".
- 10-06 "Persistent halo", "Reduce traffic from and to the device".
- 10-18 to 10-20 per-cell hydro: gradients, fluxes and face list on the device.
- 10-19 "Give up on exact face symmetry, massively simplifies hydro code".
- 10-23 "Compile on VIPER-GPU (with GPU_HIP)".

**Where things run [V]** (`src/voronoi_cell/README_VORONOI_CELL.md`, abbreviated `VC`; `voronoi_cell_gpu.h`; `voronoi_cell_build_kernel.h`).
- **Domain layout.** There is one "island" per GPU. Island rank 0 is the "feeder" that owns the device. The other ranks of the island walk and stage their cells into island-shared (MPI shared-memory) slices (VC:167-177).
- **Device build path** (`vcell_gpu_build_block_kernel`, `voronoi_cell_gpu.cu:3840`). One launch per cell runs the whole retry ladder: walk, sort, clip, geometry, grow radius, repeat. "so that a cell costs one launch however many attempts it takes and its candidates never cross the bus" (`voronoi_cell_build_kernel.h:9-17`).
- **Neighbour tree on the device.** Built and sorted there (`AREPO_VCELL_GPU_TREEBUILD`, default 1). A device `(key, ID)` sort is proved against the host order with a strictly-ascending check (VC:200-260).
- **Stays on the host:**
  - the classic AREPO mesh structures (`Mesh.DP/VF`), filled by the adapter `voronoi_cell_arepo.cc`;
  - MPI halo exchange;
  - the **host tail**: cells handed back for overflow, a structural error, or a radius past half the box. These are rebuilt by the host ladder "from the radius and attempt count the device reached" (`build_kernel.h:30-37`);
  - small islands: the device is used only above `MeshDeviceMinCells` (20000 in the example log, VC:520-524) or `MeshDeviceMinGraphCells`.

**Warp-per-cell kernel design [V]** (`voronoi_cell_gpu.cu:211-250`, `voronoi_cell_gpu.h:52-96`, `voronoi_cell_kernel.h:796-830, 1042-1110`).
- **Mapping.** One block of 32 threads per cell. The whole polyhedron scratch (`vcell_work_t<128,384,256>`, i.e. faces/edges/vertices) sits in shared memory as raw words, zero-filled by the lanes:
  > "One cell per warp, the cell's scratch in shared memory and the lanes sharing its work. ... several hundred accesses per candidate into a 16-21 kB linked structure cost a dependent-load latency each in global memory and an on-chip hit in shared memory ... Shared memory bounds the occupancy (164 kB per SM, so ~7 cells): fewer cells resident, each far faster."
- **Size and occupancy.** "22 424 B + an 88 B clip record per block is 7 blocks per SM on an A100" (`voronoi_cell_gpu.h:57-59`). That is **7 warps per SM, about 11 % of A100 warp slots** [I: 7/64].
- **Capacities.** The winged-edge structure from nextgen is unchanged: `unsigned short` indices, `fl/fr/next/prev` edge links, a `float r2` per vertex, double positions, and free stacks equal to the array sizes.
  - Production high-water is "**36 faces / 105 edges / 72 vertices per cell**", so the 128/384/256 defaults are "~3x the worst cell" (`gpu.h:52-55`).
  - Shrinking them buys occupancy and costs escalations. Capacity is a compile-time template parameter (`VCELL_GPU_ARENA_NF/NE/NV`).
  - Removing an unused embedded face buffer saved "5120 B of shared memory ... 19 % of the blocks an SM can hold" (`gpu.h:76-80`).
- **Lane work.** `GPU_BLOCK_FOR` loops spread the per-element steps over lanes: vertex classification, the fully-cut-face test, face fans in `cell_geometry`. Counts are reduced with `gpu_block_sum_int`.
  - Order-dependent steps (free-stack pushes, ring splicing) run in `GPU_BLOCK_ORDERED_BEGIN` or `GPU_BLOCK_SINGLE` "in the order the serial code produces, so the result is bitwise the serial one" (`kernel.h:796-805`).
  - So the cut-ring splice is **serial on one lane** [V: "One lane. Returns 0 or an error", `kernel.h:832`].
- **Alternatives.** `vcell_gpu_clip_smem_kernel` (one thread per block with shared scratch, env `AREPO_VCELL_GPU_SMEM`) and a global-memory "arena" form (one thread per cell, latency-bound).
- **Device tail.** Cells that outgrow the compact scratch go to `vcell_gpu_build_tail_kernel` at full 1024 capacity, with about 100 kB of scratch and ~3 MB of slices per block in global memory. It "is rare by construction (a few hundred cells of ten million over a whole cosmological zoom run, none at all on compact meshes)" (`gpu.h:84-97`).

**Candidate gathering on the GPU [V].**
- **Tree source** (`MeshCandidateSource 0/1`). The device walks the node-local halo tree (`ngbnl_walk_view_cands`, shared with the host). It counts every candidate inside `rcut` (an exact strict `r2 < rcut²`) and stores those that fit the slice.
  - The candidate slice is padded to a power of two and sorted on the device by a **bitonic network** over the strict total order `vcell_cand_less`: distance, then position/ID tie-breaks evaluated at the cell's sector. The network's pairs are spread over the lanes (`build_kernel.h:205-215`).
  - Cells above the candidate cap K are deferred, or go to the graph ladder.
- **Graph source** (`MeshCandidateSource 1` = tree first with graph backup, the default since 09-28; `2` = graph only).
  - The candidates are the cell's **depth-3 ball over the previous mesh's connectivity CSR**, raised to depth 4 if a face partner lands at the outermost level. A truncated ball goes to the tree walk.
  - "a cell can gain a face partner that the previous mesh's graph places three hops away (`NGBNL_GRAPH_MINDEPTH`; a cosmological zoom run shows one within a few hundred builds)" (VC:61-84).
  - On the device: "`vcell_build_dev_one` with `VCELL_SRC_GRAPH` runs the same ladder over a device copy of the CSR, the ball in a per-block slice and its visited table in the cell's idle candidate slice" (VC:100-110).
  - The CSR upload is lazy, at the first device graph batch.
  - Verification knobs rebuild every n-th graph cell by tree walk, requiring agreement to 1e-10 in volume and Rmax, and test outside points against the vertices.

**Exact acceptance rule [V]** (`voronoi_cell_clip.h:140-147`, `kernel.h:120-160`).
- **Termination:** `r2 >= 4.0*Rmax2` on the next sorted candidate.
- **Closure certificate** on an exhausted list:
  ```cpp
  __host__ __device__ inline int vcell_closure_certified(double rmax2, double rcut)
  { if(VCELL_NUMDIMS == 2) return 4.01 * rmax2 < rcut * rcut;
    return 4.0 * rmax2 <= rcut * rcut; }
  ```
  It is applied only if `seed_fully_clipped()`, i.e. no cube face survives. "A certified cell is bitwise what a terminating candidate at this rcut would have produced". This rests on the gather predicate being strict (`r2 < rcut²`).
- **Retry growth** `vcell_grow_rcut`: ×1.26. Once the seed cube is gone, the radius is raised to at least `1.01 * 2 * Rmax` instead ("twice that distance bounds what the termination criterion can ask for"), so a second attempt nearly always closes.
- **No exact arithmetic or tolerance** in the clip: plain double, the `dx*v - d2 > 0` test, tri-state deleted marks, and a `t∈[0,1]` guard that returns an error code instead of aborting (`kernel.h:44-52`).
- **Coordinates:** "physical doubles relative to the cell's mesh-generating point ... No integer positions" (`kernel.h:24-27`).

**Bitwise CPU/GPU identity [V]** (VC:425-455).
- The same source runs on host and device: `__host__ __device__`, with the `GPU_BLOCK_*` macros becoming serial loops on the host.
- The whole directory is built with `-ffp-contract=off` and the device unit with `-fmad=false` ("so neither side fuses a*b+c on its own"). Under LTO, entry points carry `VCELL_PINNED_ENTRY`.
- Candidate sets are geometric and the sort is a strict total order, so results are independent of gather order, K and schedule. Per-build order-independent **XOR checksums** of volume and Rmax are logged.
- `VORONOI_CELL_GPU_VERIFY` re-clips every batch on the host and compares field by field.
- The same approach was used earlier for `NUCLEAR_NETWORK_GPU`: "GPU offload v1 - bitwise-identical to the CPU", "pin FP contraction, explicit fma in the kernels" (2026-07-12/13).

**Per-cell hydro on the GPU [V]** (`src/hydro_per_cell/README_FINITE_VOLUME_SOLVER_PER_CELL.md`, abbreviated `FVS`).
- **Data on the device.** Mirrors of the primitive records and gradients of every island cell. The face list is built on the device right after mesh construction (`243bf20c24`).
- **Gradients.** A per-cell LSQ kernel (`fvs_per_cell_grad_kernel.h`).
- **Fluxes.** "the same per-cell kernel (`fvs_cell_fluxes`) as a thread per island cell over the mirror. (**A warp per cell, its lanes sharing the scalars of a face, was slower at every scalar count**: the lanes repeat the hydrodynamics of the face, which is most of the work.)" (FVS:298-305).
- **Face computation.** Each side computes the face with its own geometry: L = lower ID, R = higher ID, conservative to round-off. Inactive partners get records, sorted by (target, source ID) on the host.
- The host per-cell path costs +10 % (hydro 696 s vs 487 s) and doubles face work (fluxes ~173 s of 1878 s, +9 %). A face-flux cache was rejected because it would break bit identity (FVS:229, 383-392).
- Requires MHD + Powell + HLLD.

**Measured numbers [V]. This is everything I found; none of it is a cells/s device throughput.**
- Wave size 256 → 4096 "cut the feeder's clip 5.9x and the device's cost over the run from +21.7 % to +0.4 %" (VC:188-192).
- Whole construction "within ~1.3× of the classic Delaunay construction on a 400-rank cosmological zoom"; (de)refinement volumes "3–7× cheaper" (VC:572-577).
- Example log, Auriga zoom partial build on 384 ranks / 32 islands, **CPU**: "built 4849 cells in 2 rounds (0.0236496 s) ... 105.4 candidates/cell avg of 1174 max" (VC:520). That is ~0.2 M cells/s for a tiny partial build, which says little about throughput [I].
- Candidate counts in a zoom: largest ~290,000 without retirement and ~7,700 with (VC:494).
- No job scripts or benchmark files exist for VORONOI_CELL [V: `git ls-tree` shows only `examples/voronoi_cell/*` regression tests: degenerate BCC and lattice, Evrard, CPU/CUDA/GPU variants].

### E2. Oliver Zier: GPU finite volume for radiative transfer on the classic mesh

**[V]** (`origin/Arepo2`, `Arepo2-NewGPU`, `MRT_GPU_cleaned_up`; files `src/gpu/gpu_finite_volume/gpu_finite_volume_helper.cu`, `src/MRT/gpu/gpu_files/*.cu`).
- **No mesh construction or geometry on the GPU.** `construct_mesh_for_gpu` copies the CPU-built classic Delaunay/VF topology to the device as CSR plus SoA (`offset_interfaces`, `interfaces_list`, `particles_left/right`, `nx[]`, `face_area[]`, `face_dt[]`) (`gpu_finite_volume_helper.cu:30-72`). It is rebuilt every step after the CPU mesh build.
- **Kernels**, via `gpu_parallel_for` over CUDA/HIP/SYCL backends:
  - interface fluxes: Rusanov, M1 RT, "Each thread processes one interface but loops over all MRT_BINS to reduce register pressure" (`RT_gpu_finite_volume.cu:61-63`); "launch one thread per interface, not per bin" (`9506b1a75c`, 2026-01-01);
  - LSF gradients: "calculate gradients for one cell per thread" (`09b6624280`);
  - the flux reduction: a **per-cell gather over the CSR interface list with sign, no atomics** (`RT_gpu_finite_volume.cu:289-315`);
  - thermochemistry and primitive update. "rewrite data to save one array per conserved quantity" (`f82d61617e`).
- **Responsibility.** Faces follow the classic lower-ID responsibility rule. Faces whose active partner is on another task are computed by both tasks to avoid sending results back.
- **CPU-side mesh changes by Zier that affect GPU readiness:**
  - integer coordinates and integer neighbour/gravity trees (`arepo2_integer_coord` 2026-06, `Arepo2_dev_integer` 2026-07/08: "replace Pos by IntPos", "introduce integer based neighbour tree", "add buckets of particles in tree, allow bottom up tree construction");
  - "remove predicates" (unused Shewchuk file, 2026-07-10);
  - "change gradient calculation by calculating inverse matrix once" (`e61a013d0c`, 2026-07-16, `voronoi_gradients_lsf.cc`: "the least-squares matrix depends only on the interface geometry ... invert the shared matrix once and obtain the gradient of each quantity by multiplying its right-hand side with the inverse");
  - a stack-based GPU memory manager (2026-07-15).
- **Timings:** none in the code or commit messages. Commit subjects only say "optimize".
- `AMR_GRID` (Zier, 2026-07/09): `AMR_RT_GPU` on a Cartesian AMR grid ("Implement AMR_RT_GPU production binding"). This is not Voronoi.

### E3. Others, for completeness [V]
- Christopher Hecker (`-nextgeneration-gputree/-gpusum/-hipfft/-heffte`): gravity and PM on GPU (HIP, unified memory on MI300). No mesh work.
- Springel (`-nextgeneration-SoA`): SoA neighbour table with a GPU-coalesced index (`i + n*PDATA_SIZE`) and a GPU backend layer. No mesh kernels yet (section C5).

### E4. Design points versus voronoi-cpp, and what would speed up our full rebuild on an A100

| Design point | Pakmor VORONOI_CELL_GPU | voronoi-cpp | Assessment |
|---|---|---|---|
| Cell representation | Winged edge, `ushort` indices, double, 128F/384E/256V, **22.4 kB/cell** | Lean: vertex adjacency (3 planes + 3 neighbours), 48V/32P, ~1.2 kB (float) / 1.8 kB (double), interleaved per thread. Warp: 64V/64P SoA per warp. | Ours is **~12–18× smaller**. On an A100 (164 kB shared memory per SM) that is ~90–130 cells per SM in lean versus their 7. For a full rebuild of a near-uniform set this is the dominant factor. Keep ours. |
| Thread mapping | Warp per cell; per-element loops on lanes; ring splice on one lane | Lean: thread per cell. Warp: warp per cell with all lanes in the cut (constant work per vertex via adjacency). | Their warp kernel serialises the topology update; ours parallelises it. Their own hydro finding ("warp per cell slower: lanes repeat work") matches the general lesson: warp-per-cell pays only if the per-element step is truly parallel. |
| Capacity versus real cells | High-water in zooms: **36 faces / 105 edges / 72 vertices** | Lean 48 vertices / 32 planes; warp 64/64 | [I] **Our lean capacity would overflow on the worst zoom cells (72 vertices), and those go to our double per-thread fallback.** For uniform/glass inputs it is fine. Measure the overflow rate on a clustered set before trusting the fast path. |
| Candidate gather | Device tree walk (exact count), then **bitonic sort** of the padded slice (~105 candidates average); or a graph ball (depth 3–4) over the previous CSR | Grid rings or a sorted stencil, nearest-first 16-entry buffer, no sort of the full set | Ours is cheaper for uniform inputs (no full sort, no tree). Their graph ball is the main **new idea** (below). |
| Acceptance | Stop at `r2 >= 4 Rmax2`; **closure certificate** `4 Rmax2 <= rcut²` with a strict gather; growth to ≥ 1.01·2·Rmax once bounded | Stop at the first ring or stencil entry beyond 2R (with a 1.0001 margin) | Equivalent in spirit. Our walk is already self-certifying. Their "grow to 2.02 Rmax once bounded" rule matches our ring walk's natural behaviour. |
| Exactness | Plain double, no tolerance; bit-identity by FP pinning; one-sided slivers (area ~1e-30) accepted | Tolerance in clipping; exact integer + SoS in `tet_flips`; owner rule | Ours is more robust topologically. Theirs is more reproducible bit for bit. |
| Topology reuse | Only R_max and the previous CSR (as the candidate source) | `tet_mesh` persistent tets + flips (~200 M cells/s per step) | Our persistent path is ~10–15× faster than any clipping rebuild. Their graph ball is the clipping-side counterpart. |
| Host/device split | Host feeds waves; tail on host; MPI islands | Fully device-resident (Kokkos) | Ours avoids their dominant cost (gather and exchange) for single-GPU runs. |
| Hydro | Thread per cell, faces computed twice | Face once by owner (`face_geometry.hpp`, one thread per face) | Ours does half the face work and is exactly conservative. |

**Ideas from this person-level dig that would speed up our *full rebuild* on an A100, ranked.**
1. **Graph-ball candidates for the rebuild after motion** (Pakmor graph mode).
   - When a full rebuild is needed (e.g. flip repair gives up, or after refinement), seed each cell's candidates from the previous step's neighbour CSR at depth 3, falling back to the grid if a face partner sits at the outermost hop.
   - We have this CSR from `tet_mesh` and the face lists. It replaces the grid build and stencil walk, and the candidates come almost sorted by hop.
   - Expected gain [I]: fewer candidates tested (their average is ~105 for the tree source), and no grid. It needs a cheap distance sort of ~60–100 entries per cell; a warp-level bitonic sort, as they do, fits.
2. **Exhausted-list certificate plus "grow to 2.02 R_max once bounded"** for our float fast path. A cell whose first attempt exhausts the 16-entry buffer or stencil, with the box gone, can be certified or retried with a provably sufficient radius in **one** extra pass instead of an escalation ladder. This cuts double-fallback launches [I].
3. **Size capacity from measured high-water, and log it** (their `record_highwater`, "~3× the worst cell"). Print max vertices/planes per run for lean and warp, and tune `max_vertices`/`max_planes` per data set as template parameters; they report one binary per point. The fewer bytes per cell, the more resident cells. Their 5120 B removal raised occupancy 19 %.
4. **Lazy rare-cell tail with big scratch in global memory.** They keep the fast kernel small and send "a few hundred cells of ten million" to a big-capacity tail kernel. This is the same as our double per-thread fallback; keep the fast-path scratch minimal and do not grow it for outliers.
5. **Per-face LSQ gradient weights, or the inverse matrix once per cell** (Zier `e61a013d0c`, Springel nextgen) and a **CSR gather-with-sign flux reduction, no atomics** (Zier RT-GPU). This does not speed construction, but it is the cheapest way to make geometry outputs immediately useful for a GPU hydro step.
6. **Determinism tooling:** XOR checksums of volume and R_max per build, FP-contraction pinning in a "reference" build, and a VERIFY mode that rebuilds every n-th cell with the reference kernel. These are cheap guards while optimising lean/warp further.

**Not to copy for speed:**
- the 22 kB winged-edge cell and 7 cells per SM;
- a serial one-lane ring splice in a warp kernel;
- host-fed waves for a single-GPU code;
- computing every face twice.
