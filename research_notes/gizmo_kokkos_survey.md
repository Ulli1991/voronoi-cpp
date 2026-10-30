# GIZMO Kokkos GPU port: what it offers our GPU Voronoi MFV

Date: 2026-10-29. Repository: `/raven/u/uli/gizmo`, branch `gpu_kokkos_unified`, HEAD b653e863, 7404 commits. This was a read-only survey; nothing was built or run.

Paths are relative to `gizmo/src/` unless stated. **V** = verified in code or a commit message; **I** = inference.

The Kokkos programming patterns and the ranked list of speed items for our kernels are in **`gizmo_kokkos_paradigms.md`**. This file covers the architecture, neighbour search, hydro, performance numbers and tooling, kept short.

## 1. Architecture

**Memory**
- **Particle data (V).** `P[]` and `CellP[]` are AoS structs of about 963 B per particle. They are allocated once in `Kokkos::SharedSpace`, i.e. UVM (system/allocate.cc:144,167; gpu_particles_arena.h:186-205).
  - Host and device mutate them in place. The "arena" is a pointer alias (commit cff2510c).
  - Placement is set by `cudaMemAdvise` / `hipMemAdvise` according to the per-rank footprint (gpu_particles_arena.cc:240-310).
- **Hot device-only arrays (V).** These live in `GIZMO_KOKKOS_DEVICE_SPACE` = CudaSpace / HIPSpace (declarations/macros.h:189-200) and are moved by `deep_copy` through Unmanaged Views:
  - neighbour index: tiles, BVH, pool, compact rows;
  - CSR lists;
  - staged active data;
  - scratch.
- **Global parameters (V).** Device code reads them through a per-TU `__managed__` mirror of `All` (declarations/gpu_all_mirror.h).

**Launches**
- **Kernels (V).** Mostly flat `parallel_for(N, lambda)` + `fence` + error check (declarations/gpu_dispatch_templates.h:31-36).
- **Team kernels (V).** Used for:
  - neighbour-row reductions (mesh/neighbor_loop_runner.cc:2148-2204);
  - tile folds (mesh/gpu_neighbor_list.cc:1215-1249);
  - gravity packet walks (gravity/gpu_gravtree.cc).

**Host vs device**
- **Device (V):** density, gradients, hydro force, gravity tree build and walk, drift, cooling and feedback.
- **Host (V):**
  - MPI;
  - ghost writeback;
  - Mode A trimming and after-iteration steps;
  - inverting the gradient matrix (hydro/density_loop.cc:432-462);
  - kicks and timestep selection (core/kicks.cc, core/timestep.cc).
- The docs say "almost all numerically-intensive work runs on the device" (docs/gizmo_documentation.md:822-856). Kicks and timesteps staying on the host contradicts that for those two pieces.

## 2. Neighbour search

**The three paths (V).** One is chosen per call (mesh/neighbor_loop_runner.cc:261-299).
- **Mode A:** CSR lists built from an index of Morton-ordered tiles of 64, with a BVH over the tiles (mesh/sfc_tiles.h:3-33; gpu_neighbor_list.cc:426-442, 1138-1273).
  - Build: 128-bit Morton keys, then `Kokkos::Experimental::sort_by_key`, which falls back to merge sort on CUDA (gravity/gpu_morton.cc:150).
- **Mode B:** a host walk.
- **Mode D:** a device walk of the gravity octree with the pair kernel fused in, so no list is stored. Queries are grouped in packets of 8 that share one traversal (mesh/device_packet_walk.h).

**CSR build (V).** gpu_neighbor_list.cc:2211-2323.
- One fused pass writes each query's neighbours into a per-query scratch of 512 slots, counting past the cap.
- `parallel_scan` turns counts into offsets, then a compact pass copies the lists.
- Rows that overflowed re-walk directly into the final array.
- Lists contain both directions: there is no i<j halving.
- The symmetric radius is `max(h_i, h_j)` (mesh/neighbor_list.h:45-46).

**Smoothing-length iteration (V).**
- A bracketed, log-Newton update, at most 150 iterations (hydro/density_loop.h:303, 842-900).
- Mode A builds the list with 1.3× buffered radii and rebuilds only when a radius outgrows its buffer (density_loop.h:307; runner.cc:7392-7408).
- Converged particles drop out of a compacted active set (runner.cc:2171, 6807-6825).

**Ordering (V).**
- Particle arrays are put in Peano-Hilbert order on the host during domain decomposition.
- The GPU index sorts only keys and indices, and skips the sort when the order is still valid (gpu_neighbor_list.cc:244, 432, 2079).

**Dirty tracking (V).**
- mesh/gpu_dirty_tracker.h keeps a per-field bitset, promoted to "all dirty" above a threshold.
- The index is kept across drifts by widening each tile by its velocity range times dt (sfc_tiles.h:52-60).
- It is rebuilt when the box growth exceeds 1.0 or 1% of particles are active (gpu_neighbor_list.cc:2026-2072).

**For us (I).**
- Our uniform cell list with a counting sort is simpler and cheaper for a fixed-resolution periodic box than tiles plus BVH.
- GIZMO's structures solve adaptive h and multiple MPI ranks, which we do not have.
- Two ideas carry over:
  - **velocity-bounded reuse** of a spatial index across steps, i.e. keep the index or mesh and only repair it;
  - **skip re-sorting while the order is still valid**.

## 3. Hydro (MFM/MFV) on the GPU

**Faces (V).** Kernel-based effective areas: `A = (w_i V_i NV_T_i + w_j V_j NV_T_j)·dp` (hydro/compute_finitevol_faces_functions.h:76).
- Fallback to an SPH-style face when the condition number squared exceeds about 1e12, or when A·dp < 0 (:129-134).
- **Does not apply to us:** we have exact Voronoi faces.

**Gradients (V).**
- A least-squares matrix from the density pass, regularised until its condition number is below 10×DANGER (density_loop.cc:432-462).
- A per-cell min/max limiter with a_lim from 0.25 to 0.5 (gradients_loop.cc:75-94).
- A pairwise face limiter, `reconstruct_face_states`, with fac_minmax 0.5 and fac_meddev 0.375 (hydro/reimann.h:121-238).
- **May carry over to our LUNA-style reconstruction as a robustness fallback (I).**

**Riemann solver (V).**
- HLLC first. If the star pressure is negative or NaN it falls back to Kurganov–Tadmor, then Rusanov or the exact solver (reimann.h:278-371). MHD uses HLLD.
- Outer chain: reconstructed states → first-order states → zero-velocity fallback → abort (hydro/hydro_core_meshless_functions.h:275-345).
- MFM fluxes carry no mass (reimann.h:387-391).
- **The fallback chain is worth copying (I).**

**Pair loop (V).** Each active-active pair is computed twice, once from each side.
- The only j-side writes are atomic `dMass` (MFV) and `wakeup` (hydro/hydro_functions.h:14-16, 436-470).
- So momentum and energy conserve only to roundoff, and not at all across timebins (I, from the commented-out j-writes in core/kicks.cc:301).
- **Our owner rule is strictly better:** half the Riemann solves, and exact conservation.
- One team per row, 64 lanes for symmetric loops, with a custom reducer (neighbor_loop_runner.cc:2148-2252).

**Timesteps (V).**
- Rows are the active gas cells. Inactive neighbours are drifted lazily.
- The pair dt is `max(dt_i, dt_j)` (hydro_functions.h:75).
- Kicks and timestep selection run on the host.

**Precision (V).**
- Everything is double (declarations/typedefs.h:42-43). Only gravity has optional float.
- Fast-math is rejected with `#error`.
- Neighbour positions were moved from float to double "float absolute positions are invalid" (commit d412ea76; gpu_neighbor_list.h:50-53).

**Possible GIZMO bug (I).** The GPU flux frame uses `VelPred` for `ParticleVel` (hydro/hydro_force_loop.h:241; hydro_functions.h:92), while the drift uses `CellP.ParticleVel` (core/timestep_functions.h:537). The two differ for HYDRO_FIX_MESH_MOTION 4/5/6.
- Lesson for us: the face velocity used in the flux must be the same one used to move the mesh.

## 4. Performance numbers (V, from docs and commit messages)

**Docs (docs/gizmo_documentation.md:876-886), GH200 against the same node's CPU build:**

| Test | Particles | Time per step | Speedup |
|---|---|---|---|
| poisson_box | 125k | 0.27 s | 7.2× |
| poisson_box | 1M | 2.14 s | 7.9× |
| gmc_cooling | 512k | 4.8 s | 5.4× |

- Per phase: hydro force kernel ~32×, gradient 6.7×, density 3.2×.
- **Derived (I):** about 0.5 M particle-steps/s for a full hydro+gravity step at 1M on GH200, more than two orders of magnitude below our 100 M cell-updates/s target.
- GIZMO is not a throughput reference.

**Frontier MI250X, commit bfb3de05** (managed → device memory):
- device kernel total 471.8 → 281.3 ms/step;
- neighbour build 102.8 → 40.6 s;
- "the device arm is still behind the production CPU configuration overall".

**Other commit numbers:**
- UVM page faults "~0.65 ms/page ... ~1.4 s fixed overhead per kernel call" (b4b7ad1e).
- Staging array moved to device memory: 3178 → 100 ms/step (8bc158af).
- Single-pass tiling: 34.1 → 18.4 ms per million particles (dacc8a0e).
- Cooperative packet evaluation: tree+gravity 521.75 → 257.68 s (35450597).
- Spilled per-lane state moved to scratch: 698.3 → 603.4 s (4cb6f26e).
- Below about 4096 queries per rank, the host beats the device by 3-7× (neighbor_loop_runner.cc:261-265).

**Hardware:** Vista GH200, Frontier MI250X, Lonestar6 A100 (PCIe), Mac OpenMP.

## 5. Correctness and robustness tooling worth copying

All verified; details in the paradigms file, §6.
1. **Kokkos Tools allocation callbacks:** label-prefix buckets, current and high-water memory, and a ledger (system/kokkos_mem_telemetry.cc:111-136; system/memory_ledger.cc).
2. **Device error sentinel:** first-set-wins code and line, consumed by the host after the fence (declarations/gpu_device_error_sentinel.h).
3. **Bitwise gates in every performance commit.** Snapshot-identical A/B runs, flat-versus-hierarchical oracles behind environment flags, and planted faults that prove a comparison can fail (commits 1b05ea38, efb49ccf, dacc8a0e).
4. **Dirty bitset tracker** (mesh/gpu_dirty_tracker.h). Also `static_assert`s on struct sizes for shuffle reductions (neighbor_loop_runner.cc:2108-2111).
5. **`team_size_max` probed with the same scratch request as the launch,** and launch shapes reported per call (gpu_dispatch_templates.h:80-105; commit ae813606).

## 6. What helps us, ranked by expected value

**(a) Neighbour search and mesh (details in paradigms R1-R3)**
1. Flip repair: persistent pinned counter mirror and device-sized launches, so one readback per round instead of 4 `create_mirror_view_and_copy`. Optionally a Kokkos Graph afterwards. The pinned staging pattern is GIZMO's; device-sized loop control is our own design.
2. Pad positions and circumcentres to LayoutRight `double[4]`. Our `x` is default-layout `View<double*[3]>`, which is **LayoutLeft on CUDA**: 3 sectors per random gather. GIZMO's analogue is its narrow `compact_xyzh` rows (386ecb91).
3. Lean clipper: put the polyhedron in level-1 scratch so blocks can grow past 16 threads. Kokkos 5 caps level 0 at 48 KB with no opt-in (kokkos-src Cuda_Parallel_Team.hpp:579). GIZMO has no workaround either.
4. Warp clipper: move the per-lane staging arrays out of registers. This is GIZMO's scratch-slot lesson; our warp kernel already has the ballot/scan cut that GIZMO's packet engine approximates.
5. Velocity-bounded reuse of the index or mesh across steps, re-sorting only when order decays.

**(b) MFV hydro on exact faces**
1. Keep the owner rule (one Riemann solve per face). GIZMO's both-sides evaluation is the thing to avoid.
2. Prototype both reductions, ± atomics per face and a deterministic per-cell gather through a cell→face CSR. GIZMO's team-per-row with a custom reducer (static_assert size % 4 == 0) is the template for the gather. Use narrow sub-warp teams (4-8 lanes), since we have about 15 faces per cell against GIZMO's 100.
3. Copy the Riemann fallback chain and the pairwise face limiter as a safety net under LUNA-style reconstruction.
4. Keep the timestep and kick on the device. GIZMO's host kicks are a known gap, not a model.

**(c) Memory and host/device movement**
1. Device memory only for anything a kernel reads; never UVM. Persistent workspaces carved from one block, as GIZMO does (9fe3a3d9).
2. Allocation telemetry via Kokkos Tools callbacks, to track memory per stage.
3. Pinned staging plus a copy execution-space instance plus events for the few host readbacks (snapshots, diagnostics).

**Does not apply:**
- kernel effective areas, the NV_T matrix and condition-number face fallbacks;
- adaptive-h iteration;
- tiles plus BVH (our uniform grid is enough);
- UVM placement policies;
- per-TU managed mirrors;
- MPI ghost and writeback machinery;
- Mode A/B/D routing.
