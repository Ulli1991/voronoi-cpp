# Hierarchical timestepping in GADGET-4 and AREPO-2, and how to adopt it on the GPU

2026-10-29. Read-only survey. Nothing was built or run.

Sources and abbreviations:

- **G4**: `/raven/u/uli/gadget4`, master `5ab1527b` (Springel, 2026-08-24). Paths are relative to `src/`.
- **soa**: `origin/Arepo2-nextgeneration-SoA` `51af37f569` (Springel, 2026-10-28), read with `git show`.
- **wd**: `origin/Arepo2_WD` `def96e2ab9` (Pakmor, 2026-10-29). This is the classic AREPO lineage plus the GPU per-cell builder and solver.
- **ng**: `origin/Arepo2-nextgeneration`. **sf**: `origin/Arepo2-nextgeneration-starformation`.

Tags: **[V]** means verified in the code at the cited line. **[I]** means inferred, or a design proposal.

This note does not repeat earlier surveys (arepo2_mesh_survey.md §2-3 covers active-only rebuild and the face-twice policy; gizmo_kokkos_survey.md and sphexa_survey.md cover the gather mode for sub-groups).

---

## 0. Summary

1. **Both codes use the same skeleton** [V]:
   - an integer timeline `Ti_Current` in `[0, TIMEBASE]`;
   - power-of-two bins, so bin `b` has step `2^b` ticks;
   - a sync point is the next multiple of `2^b` over the occupied bins;
   - bin `b` is active iff `Ti_Current % 2^b == 0`. The active set is therefore always "all bins ≤ HighestSynchronizedTimeBin".
   - A step may grow only into a bin that is synchronised now, and it may shrink at any sync point.
2. **Active lists.**
   - G4 keeps doubly linked lists per bin and walks the synchronised bins to fill `ActiveParticleList` [V].
   - soa replaced this with **one index list sorted by bin, descending**. The active set is then a contiguous suffix, and only that suffix is re-sorted after new timesteps are assigned [V]. This is exactly the GPU pattern to copy.
3. **Passive particles are drifted lazily**:
   - "drift on touch" under a per-particle spin lock (G4 `predict.cc:317-439`; soa `predict.cc:205-275`) [V];
   - tree nodes are drifted with their velocity bounds [V];
   - every particle is drifted only before a domain decomposition [V].
4. **Timestep limiting.**
   - G4 and soa have **no** Saitoh–Makino wakeup and **no** neighbour-bin limiter [V: no hits for `wakeup`, `TimebinDifference`].
   - Both rely on a **tree-based signal-arrival criterion** instead: `dt_i ≤ (r_ij + 2h_i) / v_sig,ij` over all particles, pruned by drifted node boxes with `vmin/vmax/MaxCsnd` [V].
   - wd adds a neighbour limiter (max 1 bin difference) and a full wakeup scheme with "rewind" fluxes. The rewind fluxes have negative `face_dt` and need reversed dissipation in HLLD [V]. That is complex; do not start there.
5. **AREPO-2 mesh with local steps** [V]:
   - one mesh build per sync point, for active cells only;
   - two flux passes on that mesh: a closing half for the steps that end now, with states extrapolated in time, then an opening half with new bins and new vertex velocities;
   - `face_dt = dt(min bin)` and `fac = 0.5*area*face_dt`;
   - each face is computed from the active side;
   - fluxes to inactive cells are buffered, sorted and applied deterministically;
   - passive states are extrapolated in time by `All.Time - TimeLastGradUpdate`;
   - passive mesh points move with the constant `VelVertex` of their own step.
6. **Domain decomposition (G4, inherited by soa).**
   - It is redone only when the active fraction exceeds `ActivePartFracForNewDomainDecomp` [V].
   - The balance is **multi-constraint**: memory load plus one normalised cost per listed bin (the cost of all particles with bin ≤ b). Peano-Hilbert top-leaf segments are merged greedily over round-robin cost queues [V].
   - Hydro cost per cell is a constant 1, not measured [V].
7. **For us** [I]:
   - copy the integer timeline and the soa sorted-suffix list;
   - compute each firing face **once** (owner rule) and apply it ± by a deterministic gather. This is better than AREPO-2's twice-per-face;
   - drift all positions eagerly each sync point (O(N), trivial);
   - restrict Delaunay re-checking to tets whose vertex velocities changed (incident to active points), plus tets whose kinetic certificate expired;
   - compute geometry, gradients and fluxes for the active set only;
   - cap the hierarchy depth, or use CUDA Graphs, because sub-steps with fewer than about 10^4–10^5 active cells are latency-bound.

---

## 1. GADGET-4 timestep hierarchy

### 1.1 Integer timeline and bins [V]

- `integertime` is `long long` with `TIMEBINS 60` under `ENLARGE_DYNAMIC_RANGE_IN_TIME`, else `int` with `TIMEBINS 29` and `TIMEBASE (1<<TIMEBINS)` (`data/dtypes.h:117-123`).
- Next sync point (`time_integration/predict.cc:124-157`): `ti_next_for_bin = (Ti_Current/dt_bin)*dt_bin + dt_bin` for each occupied bin, then the minimum, with `MPI_Allreduce MIN`.
- `mark_active_timebins` (`predict.cc:159-276`):
  - sets `TimeBinSynchronized[n] = (Ti_Current % 2^n == 0)` and `Ti_begstep[n]`;
  - computes the Lowest/Highest Occupied, Active and Synchronized bins;
  - decides whether a domain decomposition is due (§3).
- Bin assignment (`timestep.cc:493-536`, `timebins_get_bin_and_do_validity_checks`): the step is rounded down to a power of two. If the step wants to grow, `while(TimeBinSynchronized[bin]==0 && bin>bin_old) bin--`. So a step grows only into a bin that is in phase now, and shrinking is always allowed.
- **Two hierarchies.** `TimeBinsHydro` and `TimeBinsGravity` are separate `TimeBinData` objects (`timestep.h:15-43`). Hydro is nested inside gravity: `if(bin > P[target].TimeBinGrav) bin = P[target].TimeBinGrav` (`timestep.cc:72-76`).

### 1.2 Active lists and keeping them cheap [V]

- Each bin is a doubly linked list (`FirstInTimeBin/LastInTimeBin/NextInTimeBin/PrevInTimeBin`, `timestep.h:21-25`).
- Moving a particle between bins is O(1) (`timestep.cc:538-571`).
- `make_list_of_active_particles` (`predict.cc:441-492`) walks the lists of synchronised bins into a flat `ActiveParticleList`. Every pass after that iterates the flat list.
- The hydro list built at the sync point is reused by the density, the second-half force, the tree timestep and the first-half force (`run.cc:141-202`).
- Changing bins (`find_hydro_timesteps`) "will not change the set of active hydro particles at this synchronization point" (`run.cc:197-198`).
- `reconstruct_timebins` (`predict.cc:43-113`) rebuilds all lists in O(N) after a domain exchange.
- Linked lists are a CPU idiom. soa dropped them (§2.1).

### 1.3 Drift/kick split (KDK) [V]

Order of one sync point (`main/run.cc:72-205`):

1. find the sync point;
2. mark bins and make the lists (this drifts the active particles);
3. optional domain decomposition (`drift_all_particles` + `domain_decomposition` + tree build, `:117-139`);
4. `compute_densities`;
5. `do_hydro_step_second_half`;
6. `do_gravity_step_second_half`;
7. find timesteps, then the gravity first half-kick;
8. `find_hydro_timesteps`;
9. `do_hydro_step_first_half`;
10. `NgbTree.update_velocities`.

Details:

- `hydro_force(step)` (`kicks.cc:479-593`) kicks only `ActiveParticleList` targets.
  - Half-kick interval: `[Ti_Current - ti_step/2, Ti_Current]` for the second half, `[Ti_Current, +ti_step/2]` for the first.
  - In the second half it corrects `VelPred += (a_new - a_old)*dt`.
  - SPH forces are **gather-only**: only `SphP_i` is written (`sph/hydra.cc:781-784, 973-976`). Hence momentum is **not** conserved exactly across bins [V for the writes; I for the consequence].
- **Prediction of passive particles.** `drift_particle` (`predict.cc:317-439`):
  - integer position `+= Vel*dt_drift`;
  - `VelPred += HydroAccel*dt + GravAccel*dt`, `EntropyPred += DtEntropy*dt`, `Density += DtDensity*dt`, `Hsml += DtHsml*dt`, then EOS.
  - So passive neighbours are *predicted*, not frozen.
  - It runs under `P->access.test_and_set` and is idempotent per `Ti_Current`.
- **Lazy drift sites.** A neighbour is drifted when touched: `sph/hydra.cc:101-102`, `density.cc:94-95`, `timestep_treebased.cc:156-157`. Tree nodes are drifted with `drift_node` (`ngbtree/ngbtree.h:60-104`), which moves `center_offset_min/max` by `vmin/vmax*dt` and grows `MaxHsml += MaxDtHsml*dt`.
- **Tree maintenance between decompositions** is incremental, not a rebuild. `update_maxhsml` and `update_velocities` walk `Father[]` upward from active particles only and stop once the bounds already hold (`ngbtree/ngbtree_build.cc:664-751`).
- **HIERARCHICAL_GRAVITY** (`kicks.cc:80-222, 302-368`):
  - a cascade over bins, from the highest synchronised bin down;
  - forces are recomputed for each sub-hierarchy;
  - kick differences between level `b` and `b+1`.
  - This does not matter for hydro.

### 1.4 Timestep criteria and signal arrival [V]

- `get_timestep_hydro` (`timestep.cc:274-375`) takes the minimum of:
  - kinematic `sqrt(2 η h / a)`;
  - Courant `C·2h/MaxSignalVel`;
  - `CurrentMaxTiStep` from the tree walk;
  - `C·h/|dh/dt|`.
- **Tree-based "signal arrival" step** (`timestep_treebased.cc`):
  - Initialised as `CurrentMaxTiStep = 2h/MaxSignalVel` for each active particle (`:375-392`).
  - **Particle–particle** (`:139-216`): `v_sig = c_i + c_j - (v_j - v_i)·r̂`. If `v_sig > 0`, then `dt = (r + 2h_i)/v_sig` and `CurrentMaxTiStep = min(...)`. `j` runs over **all** gas particles, passive included, drifted lazily (`:156-157`).
  - **Node pruning** (`:36-137`) uses the drifted node box and `vsig = c_i + MaxCsnd`, with `vleft = -vsig + (vmin - v_i)` and `vright = vsig + (vmax - v_i)`. The node is discarded if the box swept over `CurrentMaxTiStep` cannot reach `i`. This is a conservative arrival-time test, applied hierarchically.
  - **No wake-up exists.** A passive particle on a long step is not shortened later. Its own step was already bounded at its last sync point by the earliest possible arrival of any signal, with velocities and sound speeds frozen at that time.

---

## 2. AREPO-2 nextgeneration-SoA: local timesteps on a moving mesh

### 2.1 Active set as a sorted suffix [V]

- `ListOfParticles` per type is kept sorted by `TimeBin`, **descending** (`soa:src/time_integration/predict.cc:301-367`).
- `general_part_parallel(minbin, maxbin)` (`soa:src/data/simparticles.h:654-713`):
  - computes `nstart = Σ_{b>maxbin} (count+free)`;
  - processes `[nstart, nstart + Σ_{b≤maxbin})`;
  - splits that range evenly over the node's ranks with `subdivide_evenly`;
  - skips `FLAG_DELETED` slots.
- After new bins are assigned, only the active part is re-sorted: `reorder_ListOfParticles_in_timebin_sequence(C, All.HighestSynchronizedTimeBin)` (`soa:timestep.cc:133`). The sort covers only bins ≤ `timebin_max` (`predict.cc:345-366`).
  - This works because an active cell's new bin is ≤ `HighestSynchronizedTimeBin`. Steps grow only into synchronised bins: `Terminate("timebin > All.HighestSynchronizedTimeBin")`, `timestep.cc:110-111`.
  - So the passive prefix never needs touching. (A full re-sort with `TIMEBINS` happens only after a domain exchange, `soa:domain_exchange.cc:277`.)
- Neighbour storage per block is slot-major on the GPU, `i + n*PDATA_SIZE` (coalesced), and cell-major on the CPU (`soa:src/data/sph_particle_data.h:413-420`).

### 2.2 Order of one sync point [V] (`soa:src/main/run.cc:54-317`)

1. `find_next_sync_point`, then `mark_active_timebins`. If there is no decomposition, `NgbTree->update_velocities` runs over the *previous* active set (`:77-96`).
2. Optional decomposition (`:110-130`): `drift_all_particles`, then a tree rebuild.
3. `Mesh->construct_mesh(C, HighestSynchronizedTimeBin)` (`:145`). Only active cells are rebuilt. Volume and centre of ghosts are refreshed (`:153`).
4. `prepare_gradient_operator` over active cells (`:156-158`).
5. **First flux pass** `compute_interface_fluxes` (`:166`), labelled "second half step flux computation". It closes the steps that end now.
6. Refinement (`:174`).
7. `update_primitive_variables` (active only; `soa:src/mesh/mesh_update_primitives.cc:193-201`).
8. Gravity second half-kick.
9. Ghost primitives exchanged, then `calculate_gradients`. This stamps `TimeLastGradUpdate = All.Time` and `Grad.Center = Center` for active cells (`soa:mesh_gradients.cc:26-36`).
10. Timesteps: `assign_hydro_timesteps` (tree walk) and `assign_timesteps`, which re-sorts the suffix (`run.cc:268-271`).
11. Gravity first half-kick.
12. `Mesh->set_vertex_velocities` (active only: `VelVertex = Vel` + regularisation; `soa:voronoi_vertex_velocities.cc:20-27`).
13. **Second flux pass** "doing fluxes at the beginning of the timestep" (`:285`). It opens the new steps.
14. The mesh is destroyed (`:288-289`).

So **one mesh build per sync point serves two flux passes**: the closing half of the old step and the opening half of the new step. This is the trapezoidal (Heun-type) time integration of Pakmor et al. 2016, split KDK-style across sync points [I: the scheme name; V: the structure].

### 2.3 Which faces get fluxes, with what dt [V] (`soa:src/mesh/mesh_fluxes.cc`)

- The loop runs over active cells only (`:82`). Each one loops over its own neighbour list, i.e. its own faces.
- `face_timestep` (`:218-237`) takes the **minimum** of the two bins: `timeBin = min(TimeBin_L, TimeBin_R)` and `face_dt = 2^timeBin * Timebase_interval`.
- Applied amount: `fac = 0.5 * area * face_dt` (`:158`). The 0.5 is there because each sync point contributes one half (closing or opening).
- The left cell is updated directly (`:160-164`).
- If the right cell is inactive, the flux goes into a per-node buffer (`:176-197`): "if the right cell is active as well, it will compute its own flux, otherwise we need to add this in an order fashion to the inactice cells".
- Buffers are exchanged, **sorted by ID** and applied (`:576-584`, `mycxxsort ... return p1.ID < p2.ID`). The application is deterministic.
- So in soa an active–active face is computed **twice**, with independently built geometry (see arepo2_mesh_survey.md §3). It is conservative only to the extent that both geometries agree.
- [I] **A passive neighbour always has the larger bin.** A smaller bin would be synchronised too, by power-of-two nesting. So for an active–passive face, `face_dt` is always the active cell's dt. A passive cell receives `0.5*dt_A*area` per pass from every active neighbour. Summed over its step, that integrates its whole interval.

### 2.4 States of passive cells: time extrapolation [V] (`mesh_fluxes.cc:333-480`)

- `dtExtrapolation = All.Time - p->TimeLastGradUpdate[i]` (`:356`). This covers:
  - active cells in the closing pass, extrapolated over their full step;
  - passive cells in either pass, extrapolated over the time since their last gradient.
- Time derivative from primitive-variable Euler equations using the stored gradients:
  - `delta_rho = -dt ρ ∇·v`;
  - `delta_vel = -dt ∇P/ρ`;
  - `delta_press = -dt γ P ∇·v`;
  - plus MHD terms (`:376-421`).
- Moving mesh: instead of an advective term, the spatial vector is shifted, `d -= dt*vel` (`:441`). Here `d` = face centre − `Grad.Center` (`:371`), with `Grad.Center` the centre at gradient time.
- Extrapolations are applied only if ρ and P stay positive (`:466-475`).
- Primitives and gradients of passive cells are not updated until their own sync point. Their conserved quantities do accumulate fluxes.

### 2.5 Mesh points of passive cells [V/I]

- `drift_particle` (`soa:predict.cc:205-275`) moves gas by `VelVertex*dt` (`:246-248`) and moves `Center` by the same integer delta (`:260-266`).
- `VelVertex` is set only for active cells at the start of their step. So a passive generator moves on a straight line with constant velocity over its step [V].
- Face velocity (`mesh_fluxes.cc:296-331`): midpoint `0.5(w_L+w_R)` plus a correction from the face-centre offset, limited when `|c| > 0.9 d`.
- **Who drifts passive neighbours before the mesh build?**
  - In soa, `voronoi.cc` reads `ngbp->IntPos` without drifting (`soa:src/voronoi/voronoi.cc:220-232`) [V].
  - `drift_all_particles` runs only before a decomposition (`run.cc:117`) [V].
  - Gradients drift neighbours lazily (`mesh_gradients.cc:193-194`), but that is after `construct_mesh`.
  - [I] So in pure-hydro runs, passive neighbours can enter the mesh build at stale positions.
  - The `sf` fork fixed this in `b5ba8f8a67` (Sunmyon, 2026-10-07, "Drift mesh search geometry before Voronoi construction"). The fix drifts targets, neighbours and nodes inside the mesh walk, with locks around snapshot copies. It is not in ng or soa [V].
  - Lesson: **every position that enters geometry must be at `Ti_Current`.**
- [I] Possible slip at `soa:timestep_treebased.cc:108`: `if(p->Ti_Current[i] != All.Ti_Current) MyTree->drift_node(...)` tests the *target particle's* time, not the node's. Active targets are current, so nodes may not be drifted in this walk.

### 2.6 Tree-based timestep in soa [V]

- Same algorithm as G4, written as a generic `tree_walk` accumulator (`soa:src/time_integration/timestep_treebased.cc:37-140`).
- The minimum distance is `cellrad` instead of `2h` (`:84-86`).
- The initial value is `rad / (c + |v - w|)` (`soa:timestep.cc:148-171`).
- `get_timestep_hydro` (`soa:src/data/sph_particle_data.h:305-355`) returns `C · a · min(rad/(c [+|v|]), CurrentMaxTiStep)`.
- Note the FIXME (`voronoi_vertex_velocities.cc:20-21`): the vertex velocity uses the timestep, and the vertex velocity also enters the timestep criterion, which is circular.

### 2.7 What wd adds (Pakmor; classic lineage) [V]

- **Neighbour limiter**, `static const int maxAllowedTimebinDifference = 1` (`wd:src/time_integration/timestep.h:39`).
  - Active cells are limited to within 1 bin of every direct Voronoi neighbour, iterated to convergence.
  - It runs in all builds unless `TIMESTEPS_WAKEUP_IGNORE_NGB_LIMIT` is set (`wd:timestep_wakeup.cc:12-41`).
  - It needs `VORONOI_DYNAMIC_UPDATE` connectivity.
- **Wakeup** (`TIMESTEPS_WAKEUP`). Woken cells move to a fine bin, and their original bin is kept in `SphP[].Wakeup`.
  - `face_timestep` (`wd:src/hydro/finite_volume_kernel.h:128-180`) gives faces of a woken cell `face_dt = 2*fvs_half_step_from_bin(...)`, i.e. `dt_true - 0.5*dt_planned`. That "can be negative when the wake falls in the first half of the face's cadence step".
  - HLLD then runs with **reversed dissipation** for negative faces (`finite_volume_solver.cc:304-313`).
  - This is powerful but intricate. It conflicts with a simple owner-face kernel.
- **GPU thresholds for small sync points.**
  - `PerCellDeviceMinCells`: "an island with fewer active cells at a sync point runs the per-cell kernels on the host (the device's fixed cost per launch against the host's cost per cell)" (`wd:src/data/allvars.h:1865-1866`; default 1, `parameters.cc:419`).
  - `MeshDeviceMinCells`, **default 20000** (`parameters.cc:429`; `allvars.h:1872`).
  - The example log reads "0 of 32 islands with cells on the device (32 below MeshDeviceMinCells 20000…)" (`README_VORONOI_CELL.md:523`).
  - Wave size: "`AREPO_VCELL_WAVE` 256 -> 4096 cut the feeder's clip 5.9x" because small batches are latency-bound (`README_VORONOI_CELL.md:180-192`).
- **Flux records for inactive partners on the GPU.** Each cell writes them at a per-cell offset from a scan. The host applies them in (target, source ID) order (`README_FINITE_VOLUME_SOLVER_PER_CELL.md:300-320`).
- **Cost of double face work.** Computing both-active faces once per cell "roughly doubl[es] the face work (fluxes ~173 s of the 1878 s WD production row, +9 %)", accepted for CPU/GPU bit identity (`:380-395`).
- **Incremental domain decomposition**: `DOMAIN_SMALL_DECOMPOSITION` tries a cheap incremental decomposition before a full one (`wd:src/main/run.cc:474-498`).

### 2.8 GADGET-4 vs AREPO-2

| | GADGET-4 (SPH) | AREPO-2 soa (moving mesh) |
|---|---|---|
| Timeline, bins, sync | integer, 2^b, grow only into synced bins | identical (`soa:timestep.cc:295-340`) |
| Hierarchies | separate hydro and gravity; hydro ≤ grav | one `TimeBin` = min(grav, hydro) (`soa:timestep.cc:100-104`) |
| Active list | linked lists → flat list | **bin-sorted index list, active = suffix** |
| Passive neighbours | predicted (pos, VelPred, ρ, h, entropy) on touch | positions drifted on touch (not in the mesh build, §2.5); states time-extrapolated from gradients per face |
| Interaction | gather; i-only writes; not conservative across bins | face flux with dt = min bin; passive side gets sorted, buffered flux; conservative to geometry agreement |
| Time integration | KDK half-kicks | trapezoidal flux halves at both ends of a step, one mesh per sync point |
| Step limiter | tree signal-arrival | tree signal-arrival (wd: + neighbour ≤1 bin + wakeup) |
| Domain decomposition | per-bin multi-constraint, at large sync points only | same code |

---

## 3. Work and domain decomposition

### 3.1 GADGET-4 [V]

- **When.** At every sync point, `SmallestTimeBinWithDomainDecomposition` is the smallest bin `n` such that the cumulative count `Σ_{b≤n} N_b > ActivePartFracForNewDomainDecomp · N` (`predict.cc:268-275`).
  - Decomposition runs only if `HighestActiveTimeBin ≥` that bin (`run.cc:117`). Small sync points reuse the old domains and update the tree incrementally (§1.3).
- **What is balanced** (`domain/domain_balance.cc:653-725`, `domain.cc:213-330`):
  - Always the highest active bin. Without hierarchical gravity, also every occupied bin below `SmallestTimeBinWithDomainDecomposition`.
  - For each listed bin `b` there are two cost vectors per top-leaf: `binGravCost` (Σ `GravCost` of particles with grav bin ≤ b, where `GravCost` is the interaction count from the walk, `gravtree/gwalk.cc:163,431`) and `binHydroCost` (**1 per gas particle** with hydro bin ≤ b).
  - Plus `load` (particle count) and `loadsph` (gas count).
  - Each vector is normalised to sum 1 and capped per leaf at `1/(TopNodeFactor·NTask)` (`domain.cc:305-330`).
  - `MultipleDomains` is the number of non-empty criteria. The top tree is refined into `MultipleDomains·NTask` pieces.
- **How.** Pieces are contiguous Peano-Hilbert ranges of top leaves (`domain_balance.cc:268-305`). They are assigned to tasks greedily: round-robin over per-criterion cost queues sorted by cost, and the max over all criteria is checked against `try_balance` (`:320-460`).
  - Many (`nextra`, `try_balance`) combinations are tried, one per MPI rank in parallel, and the best is kept (`:201-260`).
  - Particles are finally put in PH order (`domain_exchange.cc:318-360`).
- **Effect.** Each task gets about the same work *in every sub-step*, not just on average. This is the main defence against "the small bin all sits on one task".

### 3.2 AREPO-2 soa

- Domain code is the G4 scheme, ported: `domain_init_sum_cost`, `GravCostPerListedTimeBin`, `HydroCostPerListedTimeBin += 1.0` (`soa:src/domain/domain.cc:227-395`) [V].
- Within a node there is no decomposition. The ranks share node memory and split the active suffix evenly (`subdivide_evenly`, §2.1) [V].
- **GPU in soa.** Only infrastructure exists [V]:
  - `src/gpu/` backends for CUDA/HIP/SYCL/CPU;
  - `gpu_manager`;
  - `gpu_algorithms.h`, with "Tier 2" `gpu_parallel_for(stream, n, GPU_LAMBDA body)` and a block-cooperative `parallel_for_blocks`.
  - Commits "gpu prep stuff" and "fixed compilation for A100" date from 2026-07/08.
  - Outside `src/gpu` only `main.cc` and `simulation.h` refer to it.
  - No mesh, flux or timestep kernels yet.
  - The design signals are the bin-sorted suffix, block-SoA storage and the slot-major neighbour table.

---

## 4. Proposal for our GPU design [I unless cited]

Our state: persistent Delaunay tets with GPU flip repair (~6 ms per 2.1M cells per step), face geometry (~4.4 ms incl. circumcentres), and faces stored once with an owner rule. The full rebuild runs at ~19 M cells/s only when needed. Target: ~100 M cell updates/s.

### 4.1 Time integration skeleton (copy, small)

- Integer timeline, power-of-two bins, and `TimeBinSynchronized` as above. Grow only into synchronised bins, as in `timebins_get_bin_and_do_validity_checks`.
- Per cell store: `bin`, `t_beg` (integer), `x_beg` (snapped position at step start), `w` (vertex velocity, constant over the step), `t_grad`, gradients, `Grad.Center`.
- Use AREPO-2's two-pass scheme. Per sync point: one mesh update, then a **closing** flux pass (old bins, states extrapolated by `t - t_grad`), then primitives, gradients, timesteps and `w`, then an **opening** flux pass (new bins, `dtExtrapolation = 0` for active cells).
- This costs two Riemann solves per firing face per step, like AREPO-2. A MUSCL-Hancock single pass would need a mid-step mesh. Keep the AREPO-2 split.

### 4.2 Which faces get fluxes

- Firing faces F = faces with at least one active endpoint. By the nesting argument (§2.3), `face_dt = dt` of the active side.
- **Compute each face once**, by our owner rule, and apply `∓0.5·area·dt·F` to both cells.
  - This is exactly conservative to round-off: one number with two signs. AREPO-2 soa and wd instead compute an active–active face twice from independent geometries.
  - Passive endpoints just accumulate conserved quantities. Their primitives are not touched until their own sync point.
- Generate F from the active set rather than by compaction over all faces. For each active cell, emit its incident Delaunay edges and keep each edge if (other end passive) or (both active and this cell is the owner). Then do a prefix-sum compaction into a face list. Work scales with |A|·~15.
- **Apply deterministically.** Prefer a per-cell gather over a cell→face CSR with sign (the AREPO-2 Zier RT GPU pattern, `fac = (left != cell) ? 1 : -1`; arepo2_mesh_survey.md §"GPU") over `atomicAdd(double)`, which is native on A100 but non-deterministic.
  - The CSR has to be rebuilt after flips anyway (the face-list rebuild is still open in the README).
  - Gather only over cells in A ∪ N1(A).

### 4.3 Mesh repair when only some points move

**Option E: eager drift + kinetic certificates (recommended).**

- Each sync point, set `x_i(t) = snap(x_beg_i + w_i·(t - t_beg_i))` for **all** points in one O(N) kernel. That is about 0.1 ms for 2.1M points: 100 MB at roughly 1.5 TB/s.
- Compute the drift from the step-start position, not incrementally. Then snapped coordinates do not depend on the drift path, and the exact predicates see the same points however often a cell was touched. G4 and AREPO add integer deltas incrementally, so their result depends on the drift path (`predict.cc:383-387`).
- All positions are current. This avoids the stale-neighbour problem of §2.5 and keeps the triangulation Galilean invariant: a bulk flow moves everything together and breaks nothing.
- Re-check only:
  - (i) tets with a vertex in A, because their vertex velocities just changed;
  - (ii) tets whose certificate expired.
- A certificate is a slack stored at the last check:
  - `s_t` = the minimum over the 4 facets of the in-sphere margin of the opposite far vertex, together with a flatness margin;
  - `u_t` = the maximum relative speed among the tet's vertices and the far vertices;
  - `t_fail = t_check + s_t / u_t`.
  - Velocities are constant between a vertex's own sync points, so `t_fail` stays valid until a vertex in the tet changes `w`. That only happens at that vertex's sync point, which is case (i).
- Then screen and repair only the dirty set. Today `move_screened` screens every tet (`tet_repair.hpp:272-300`), so it needs a list-driven variant.
- Circumcentres and Voronoi face geometry are needed only for tets incident to A:
  - the faces of A are edge rings around active vertices;
  - volumes and centroids are needed for A only, because passive volumes enter only when a passive cell becomes active.
- Cost per sync point is then about O(|A|) + O(distorted tets) + the O(N) drift and the certificate test. The certificate test can be one fused, memory-bound kernel over a tet array of compact `t_fail` values, about 0.1–0.2 ms, or a min-heap/bucket list by expiry time.

**Option L: lazy drift (AREPO-style "drift on touch").**

- Drift only A ∪ N1(A) and repair only the tets incident to them.
- It is cheaper per sub-step, but the triangulation mixes times. In a fast bulk flow, a drifted vertex and a stale 2-ring vertex are offset by `|w|·lag`. That can be many cell sizes, because the Courant criterion uses `|v − w|`, not `|w|`.
- So it gives large spurious flips or repair failures. Our repair already degrades near 1 spacing of motion (README). **Avoid.**

**Option G: global per sub-step.**

- Drift all, then screen, repair and compute geometry for everything (~10 ms per 2.1M).
- It is simple, but the mesh cost does not shrink with |A|. With deep hierarchies it dominates.
- It is a good first implementation and fallback, together with:
- **A switch.** When |A ∪ N1| / N is above about 0.3 (a guess, to be measured), run the global kernels without lists.

**Rebuild by clipping** only at large sync points, the ones where G4/AREPO would decompose, or on repair failure.

### 4.4 Active lists and per-bin compaction on the GPU

- **Bin-sorted permutation** (soa §2.1):
  - `order[N]` sorted by bin descending, plus `count[TIMEBINS]` on host and device;
  - active range = `[N - Σ_{b≤hs} count[b], N)`, so a sub-step's kernels launch over a contiguous index range with no compaction;
  - after timesteps, **re-sort only the suffix** with a stable counting sort over ≤ ~10 bins: per-block shared-memory histogram, exclusive scan, scatter;
  - stability keeps the Morton order within each bin, so gathers stay semi-coalesced;
  - keep the cell data in Morton order and permute only the indices. Re-sort the data in Morton order only at large sync points.
- **Host knowledge.** The host needs `count[]` once per sync point for launch sizes. Use one pinned readback (we already do pinned counter readbacks). Better, use device-side sizing so that sub-steps can be captured in a CUDA Graph (kokkos_cuda_features_audit.md §3: about 1–2 µs between kernels in a graph; a graph freezes grid sizes, so launch at capacity with an early exit or a grid-stride loop).
- **Lists derived from A, built by flag + scan:**
  - firing faces (§4.2);
  - `N1(A)` for gathers (mark neighbours of A in a bitmask, then compact);
  - dirty tets (incident to A, plus expired certificates).
  - Use `Kokkos::parallel_scan`, or CUB through Kokkos interop if the scan dominates.

### 4.5 Timestep criteria on the GPU

- **Courant** per active cell: `C·R/(c + |v - w|)`, AREPO-2 style.
- **Neighbour limiter, max 1 bin difference** (wd): over the faces of A, iterate `bin_i = min(bin_i, bin_j + 1)` until nothing changes, typically 1–3 sweeps.
  - It is cheap, local and GPU-friendly.
  - Use it as the primary protection against a shock running into sleeping cells.
- **Signal-arrival tree criterion** (G4/soa): later and optional. It needs a hierarchy with drifted node boxes, `vmin/vmax` and `MaxCsnd`. A BVH over Morton blocks of 32–64 cells would do.
  - Test with Sedov first. If the neighbour limiter already keeps the shock front sharp, skip this.
- **No wakeup or rewind initially.** wd's negative-`face_dt` correction with reversed HLLD dissipation is the most intricate part of either code (§2.7).
- **Timestep circularity** (soa FIXME): `w` depends on dt via the regularisation, and dt depends on `w`. Assign the bin first, then compute `w` with the new dt, then check `|v-w|` against the bin. If it fails, drop one bin. AREPO accepts the circularity.

### 4.6 Expected gain vs overhead

- **Cell updates per top-level step.** Let `f_b` be the fraction of cells in bin `b`, `b_min` the smallest occupied bin and `b_max` the largest. Then
  - `U_local / N = Σ_b f_b · 2^{b_max - b}`;
  - `U_global / N = 2^{b_max - b_min}`.
- **Illustrative distribution** [I], top-heavy like a Sedov blast late in time: 80/10/5/3/2 % over 5 bins.
  - `U_local/N = 0.8 + 0.2 + 0.2 + 0.24 + 0.32 = 1.76`, against `U_global/N = 16`, i.e. **9x** fewer cell updates.
  - **Overhead.** 16 sub-steps × a per-sub-step floor. The floor is drift + certificate pass + ~15 launches + 1 readback, roughly 0.3–0.5 ms without graphs and ~0.15 ms with graphs [I].
  - At 100 M updates/s, a full 2.1M step is ~21 ms. So local ≈ 1.76·21 + 16·0.4 ≈ 43 ms, against global ≈ 336 ms, which is **~8x**.
  - With 8 bins and a long fine tail, the floor term (2^7 × 0.4 ms ≈ 51 ms) starts to matter. That is why the hierarchy should be capped or graphs used.
- **Gains reported by others.**
  - The AREPO-2 MHD paper's ~3x on a blast test (and more in galaxy formation) was reported by the user. It is **not verified here**: there is no paper in the repository.
  - The ratio depends entirely on the bin histogram. **Measure it first.** Add a bin-histogram diagnostic to the global-step code (compute each cell's desired bin, then the formula above) and log the theoretical speedup per step on Sedov and the clustered tests before building any of §4.3–4.4.
- **Mesh share.** Under Option G the mesh costs ~10 ms per sub-step whatever |A| is. In the example that is 16 × 10 = 160 ms per top step and wipes out most of the gain. Option E makes the mesh scale with activity.

### 4.7 Pitfalls

1. **Small sub-steps leave the GPU idle.** An A100 holds ~221k resident threads, so a sub-step with 10^3–10^4 active cells is pure latency.
   - wd moves islands below 20000 cells to the host (`MeshDeviceMinCells`). For us, with device-resident data, the remedies are: CUDA Graphs for the whole sub-step, fusing small-bin kernels, and/or a `MinBinOnDevice`-style cap that merges the finest bins upward (shorter steps for a few cells cost little).
2. **Conservation at bin boundaries.** It is exact with owner faces applied ±.
   - Keep the face flux in one kernel, and never let a passive cell add its own copy of a face.
   - With gather application, the sign comes from the owner rule, not from IDs.
   - After flips the face set changes, so rebuild the CSR before the closing pass.
3. **Face velocity consistency.**
   - The face velocity must use the same `w` that drives the drift, constant over each generator's step. GIZMO-Kokkos has a `VelPred` vs `ParticleVel` mismatch (gizmo_kokkos_survey.md).
   - The passive state extrapolation must shift by `w` as well (`d -= dt*vel` in the gas frame, §2.4).
   - With eager drift, the active–passive face is built from current positions, which matches AREPO-2 once its drift fix is in.
4. **The passive cell's volume changes** while its neighbours move, but its primitives are only updated at its sync point.
   - That is fine for finite volume, because conserved quantities are integrated.
   - But the state used in extrapolation is ρ from the old volume. AREPO does the same.
   - Do not recompute passive primitives from the new volume mid-step: that would break the time extrapolation.
5. **Determinism.** Atomics plus a varying active set give bit-different runs. AREPO went to great lengths to stay bitwise (sorted record application; the CPU-GPU bit-identity contract in wd). Prefer gather.
6. **Load imbalance across GPUs** (later, multi-GPU). Balance per bin (G4 multi-constraint, §3.1), not on total work. Otherwise the fine-bin sub-steps leave GPUs idle. Use measured per-cell cost: hydro cost "1" is crude, and clustered cells cost 16–20x more in the rebuild.
7. **Integer time underflow.** `Terminate("time-step of integer size 1 not allowed")` (`timestep.cc:507`). Use `long long` with 60 bins.
8. **Refinement and derefinement** must happen only among active cells, and AREPO requires all neighbours of a derefined cell to be active (arepo2_mesh_survey.md §"Derefine"). That fits the A ∪ N1 lists.

---

## 5. Other GADGET-4 pieces, ranked by use to us

1. **Integer timeline, bin validity and `Ti_begstep[]`** (`timestep.cc:419-536`, `predict.cc:124-276`). Adopt as is. It is small, exact and well tested.
2. **The soa replacement of linked lists by a bin-sorted suffix** (not G4 itself; `soa:predict.cc:301-367`, `simparticles.h:654-713`). Adopt; this is the GPU active list.
3. **Domain decomposition only when enough is active** (`ActivePartFracForNewDomainDecomp`), plus incremental tree updates in between (`ngbtree_build.cc:664-751`). Adopt the trigger, also for Morton re-sorting and for deciding when to rebuild by clipping. The Father-chain updates matter only if we keep a BVH for the signal criterion.
4. **Per-bin multi-constraint domain balance on PH segments** (`domain_balance.cc`, `domain.cc:213-330`). Use it for the multi-GPU phase. On one GPU it does not apply.
5. **Tree signal-arrival timestep** (`timestep_treebased.cc`). Optional, after the neighbour limiter (§4.5).
6. **Peano-Hilbert keys** (`sort/peano.cc`). Marginal. Morton is enough on one GPU. PH gives slightly smaller domain surfaces for multi-GPU.
7. **Parallel sort that keeps element counts per rank** (`sort/parallel_sort.h`). Multi-GPU only. On one GPU use Kokkos sort, BinSort or CUB.
8. **Shared-memory node-level parallelism** (`mpi_utils/shared_mem_handler.*`: MPI-3 shared windows, "ghost" ranks serving remote tree requests). **Does not apply.** Node-level sharing on a GPU node would be NVLink peer access or CUDA IPC, a different design. The one useful idea is a dedicated communication thread or rank serving requests asynchronously, which matters later for multi-GPU halo fetches.
9. **vectorclass explicit SIMD** (`EXPLICIT_VECTORIZATION`, `Vec4d` in `sph/hydra.cc:523+`, 3 uses). **CPU-only, does not apply.**
10. **Drift and kick factors for cosmology** (`driftfac.*`), and HIERARCHICAL_GRAVITY kick cascades. **Not relevant now** (no gravity, no cosmology).

---

## 6. Concrete next steps [I]

1. Bin-histogram diagnostic in the current global-step code. Report the theoretical speedup for Sedov, jittered and clustered.
2. Integer timeline + bin-sorted suffix + counting sort of the suffix (host-checked against a reference sort).
3. MFV with local steps under **Option G** (global mesh each sub-step) and owner faces with gather. Validate conservation to round-off and Sedov against global stepping.
4. Option E: per-tet certificates and a list-driven screen/repair/geometry over the dirty set. Measure mesh cost against |A|.
5. CUDA Graph capture of one sub-step (needs device-side sizing). Set a minimum-active-cells or bin-depth cap from measurements.
6. Later: signal-arrival criterion, and per-bin domain balance for multi-GPU.
