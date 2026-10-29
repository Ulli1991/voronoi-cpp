# GIZMO `gpu_kokkos_unified`: Kokkos programming paradigms, and what we should take for speed

Date: 2026-10-29. Source: `/raven/u/uli/gizmo`, branch `gpu_kokkos_unified`, HEAD b653e863 (P. Hopkins, 7404 commits). Read-only survey: nothing was built or run.
Unless a path says otherwise, it is relative to `gizmo/src/`.
**V** = verified in code or in a commit message. **I** = my inference.
Our numbers come from the brief (A100, 2.1M cells, double). I did not re-measure any of them.

Companion file, with the physics and neighbour search kept short: `gizmo_kokkos_survey.md`.

---

## 0. Bottom line

GIZMO's port has a different shape from ours:

- It is a large, MPI-first, AoS particle code. P[] is about 963 B per particle (commit 386ecb91).
- It runs on GH200 (Vista), MI250X (Frontier) and A100 (Lonestar6).
- Its Kokkos usage is conservative:
  - flat `RangePolicy` plus `Kokkos::fence()` after almost every launch;
  - raw pointers carved out of `kokkos_malloc` instead of Views;
  - no `RandomAccess`, `LaunchBounds`, MDRange, Graph, ScatterView, SIMD or level-1 scratch.
- By its own account, the device arm is still behind the production CPU configuration overall. Commit bfb3de05: "the device arm is still behind the production CPU configuration overall even though it now wins the all-active bin".

So it is **not** a throughput reference for our kernels. Its value to us is a handful of patterns that were measured, mostly with A/B numbers in commit messages:

1. **Team-per-row with a custom reducer.**
2. **Record-then-evaluate packets.** Divergent decisions go into a per-team chunk in scratch; then all lanes evaluate that chunk.
3. **Device memory, not managed memory, for every hot read path.**
4. **Persistent pinned staging + execution-space instances + events**, to overlap copies with work.
5. **Compile-time policies.** One kernel body per schedule, so one schedule's code never bloats another's registers.
6. **A diagnostics stack:** Kokkos-Tools allocation telemetry, a device error sentinel, bitwise gates.

---

## 1. Speed opportunities for voronoi-cpp, ranked

Expected gains are estimates (I) unless stated. Every one needs an A/B run on gpudev.

### R1. Flip repair: replace the per-round `create_mirror_view_and_copy` readbacks

**Expected gain:** roughly 1-3 ms of the 7-8 ms (I).

**Our code (V).** `tet_repair.hpp` does 4 host readbacks per round:
- :358 `create_mirror_view_and_copy(HostSpace{}, counters)(3)`
- :392 the same
- :438 the same
- :527 the same

Each call:
- allocates a fresh host View;
- does a pageable `cudaMemcpy`, which is synchronous and staged;
- serialises the stream.

With 5-15 rounds that is 20-60 blocking round trips plus the allocations. This matches "cudaMemcpy dominated the API time".

**GIZMO pattern (V).**
- A persistent page-locked host space: `#define GIZMO_KOKKOS_PINNED_HOST_SPACE Kokkos::SharedHostPinnedSpace` (declarations/macros.h:188).
- A one-time pinned slab, allocated once per run because "pinning costs about half a second per 3 GB ... far too much to repeat per call" (commit a7314014; mesh/gpu_neighbor_list.cc:3030-3045).
- Copies on their own execution-space instance (`partition_space`, gpu_neighbor_list.cc:3048), with completion polled by events (`GizmoGpuEvent`, declarations/gpu_dispatch_templates.h:168-200).

GIZMO does **not** solve device-side loop control itself. Its list build still reads totals back through `parallel_scan(..., total)` + `fence` (gpu_neighbor_list.cc:2259-2280), and it has 103 `Kokkos::fence()` calls in 20 files. The concrete changes below are therefore ours, with GIZMO as partial evidence.

**Concrete changes:**

a) **Hoist the counters mirror.** Make one `Kokkos::View<int[8], Kokkos::SharedHostPinnedSpace>` (or a `HostPinned` mirror) once in `State`. Each round:
   ```
   Kokkos::deep_copy(exec, pinned, counters); exec.fence();
   ```
   This removes the host allocation and the pageable path. It is cheap and certain, but it saves only the per-call overhead (about 10-30 µs per call) (I).

b) **Size launches from the device and cut syncs from 4 per round to 1.**
   - `find` is already bounded by `listed`. Launch it over `work_capacity`, or over last round's `dirty_count` as an upper bound, and read `counters(3)` inside the kernel (`if (i >= counters(3)) return;`).
   - Do the same for `plan` and `apply`, using `counters(0)`.
   - The `planned_stuck` readback at :438 only feeds host statistics. Accumulate it on the device and read it at the end.
   - The `free_top` / `count` bookkeeping at :527 can live in device counters, with `release` reading its base from them.
   - Only the termination test (`items == 0`) needs the host. Read it with (a). Optionally test only every 2nd or 3rd round: an extra round on an empty list costs one launch of empty kernels.

c) **Once (b) is done, the round is a fixed kernel sequence with no host decisions.** It can then be captured as a `Kokkos::Experimental::Graph` (Kokkos 5 needs `#include <Kokkos_Graph.hpp>`, per our memory notes) and replayed per round, which removes launch latency for the ~6 small kernels per round. GIZMO does not use Graph (0 hits), so this item is ours alone.

### R2. Per-cell clipping: level-1 scratch for the lean kernel, register relief for the warp kernel

**Expected gain:** uncertain; up to about 1.5-2x on the lean kernel if occupancy is the limiter (I). GIZMO offers no trick for getting more shared memory per block.

**Scratch limit facts (V).**
- In our Kokkos 5.0.99, CUDA level-0 scratch is capped at `cudaDeviceProp::sharedMemPerBlock` = 48 KB. There is no opt-in: `kokkos-src/core/src/Cuda/Kokkos_Cuda_Parallel_Team.hpp:579-588` and `Kokkos_Cuda_KernelLaunch.hpp:133`.
- Kokkos only sets the carveout preference (`KernelLaunch.hpp:237`).
- GIZMO does not get past this either:
  - every scratch request is level 0 (`set_scratch_size(0, …)`, 4 sites);
  - there is no `cudaFuncSetAttribute`, `MaxDynamicSharedMemorySize` or level-1 use anywhere in GIZMO.
  - GIZMO instead *shrinks* scratch to protect occupancy. Commit 4cb6f26e: HIP packet kernels went from 1000 to 632 B/lane scratch "at unchanged registers and occupancy" and gave -14% total wall. `DevicePacketScratchLayout` (mesh/device_packet_walk.h:93-104) gives zero bytes to unused regions because "an unused region is a real cost in occupancy".

**Options:**

(i) **Lean kernel.** Today: 16-thread blocks × 1.7 KB is about 27 KB per block. That means half-empty warps by construction, plus few resident warps.
   - Try putting the polyhedron in **level-1 scratch**: `policy.set_scratch_size(1, PerThread(bytes))` or `PerTeam`. Level 1 is global memory from a Kokkos pool, capped at 20 MB per team (`Kokkos_Cuda_Parallel_Team.hpp:197-200`).
   - Then use 128-thread blocks. Our measured L2 hit rate is 99% and DRAM traffic is under 1%, so per-thread polyhedra in global memory would mostly stay in L1/L2.
   - The trade is shared-memory latency for L1/L2 latency, in exchange for 8x more resident threads and full warps.
   - A cheaper variant needs no Kokkos change: a per-cell workspace View carved once with `WithoutInitializing`, indexed `[thread][slot]` interleaved so a warp's accesses coalesce.

(ii) **Raw-CUDA opt-in, warp kernel only.** It is already CUDA-specific (`__device__` bitonic sort at shell_voronoi_warp.hpp:279). Launch it as a plain `__global__` with `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, 100*1024)`.
   - A100 allows up to 163 KB per block.
   - This helps only if the warp kernel is scratch-limited. At 23 KB per block it is register-limited (130 registers), so it is probably not worth it there (I).

(iii) **Record-then-evaluate (GIZMO's packet engine): we already have it.** GIZMO: gravity/gpu_gravtree.cc:1780-1880; device_packet_walk.h:1-32.
   - Each member makes its divergent decisions and reserves records in a team-scratch chunk with an atomic counter.
   - When the chunk is full, all lanes evaluate the records, then fold.
   - Measured result (commit 35450597): "tree+gravity 521.75 -> 257.68 s ... small-N cooperative calls about 6.9x faster per call".

   **Our warp kernel already does the clipping equivalent (V).** `shell_voronoi_warp.hpp:183-275`:
   - ballot-classifies the vertices (`vertex_ballot`);
   - numbers the crossings with a warp `exclusive_scan`;
   - builds the new vertices in parallel;
   - writes a compacted polyhedron.

   It also uses `__ballot_sync`/`__shfl_*`/`__popc`, which GIZMO never does (0 warp intrinsics in its walkers). The leftover divergence is structural: only lanes that own removed vertices do work in pass 1 and in the kept-vertex renumbering. GIZMO has nothing beyond this to offer the cut itself.

(iv) **Register pressure → per-lane scratch slots.** This is the GIZMO lesson that does apply to the warp kernel.
   - The cut stages `new_x/new_y/new_z/new_corner/new_adjacent[3*per_lane]` and `keep_*[per_lane]` in registers (shell_voronoi_warp.hpp:216-250).
   - With `per_lane = 2` that is about 20 doubles plus 16 words per lane. It is a plausible main contributor to the 130 registers that hold occupancy at 8-17% (I).
   - GIZMO moved exactly this kind of per-lane state into one team-scratch slot per lane, gated at compile time (commits 78d8667e, 4cb6f26e):
     - "the dense step-0 walk 0.63x";
     - "HIP scratch 792 -> 712 B per lane with occupancy held";
     - "Frontier FIRE-big ... total 698.3 -> 603.4 s".
   - For us the 48 KB level-0 cap binds: 4 warps × 5.75 KB polyhedron + about 8 KB per warp of staging comes to about 55 KB.
   - The experiment is therefore one of these:
     - `warps_per_block = 2` with staging in shared memory;
     - or ping-pong between two polyhedron copies, so new and kept vertices go straight to the other copy with no register staging.
   - Then check registers (`-Xptxas -v` / ncu) and occupancy. Gain: unknown (I). It pays only if registers fall enough to admit more warps per SM.

(v) **Compile-time flavours, not runtime switches.**
   - Commit 35450597 / gpu_gravtree.cc:1855-1863: "one kernel serving both schedules ran a quarter slower per call at large N".
   - Our warp kernel already fixes the team width at compile time (`SHELL_WARP_TEAM`).
   - Keep the lean, warp and retry paths as separate instantiations (I).

### R3. Geometry kernels: AoS-padded positions and centres, and Morton-ordered faces

**Expected gain:** perhaps 1.3-2x on `faces` (4.1 ms) and `circumcentres` (1.1 ms) (I).

**Layout finding (V, our code).** `x` reaches `tet_mesh::circumcentres` / `faces` as `Kokkos::View<double* [3], memory_space>` with **no layout argument** (tet_mesh.hpp:209, 230; neighbor_search.hpp:40 `using Positions = Kokkos::View<double*[3], memory_space>`).
- On CUDA that is **LayoutLeft**: `kokkos-src/core/src/Cuda/Kokkos_Cuda.hpp:98: using array_layout = LayoutLeft;` (OpenMP: LayoutRight).
- So every random gather `x(v,0..2)` touches 3 separate 32-B sectors.
- The tetrahedra were switched to LayoutRight, but the points were not.

**Changes:**
- Give positions `double[4]` LayoutRight, padded to 32 B. That is one sector per gather, versus 3 now. Plain `[3]` LayoutRight would be about 1.5 sectors on average.
- Do the same for `centre`. Its 4th slot can carry the circumradius² that `check` recomputes (tet_mesh.hpp:283-287) (I).
- GIZMO's own move was to pack exactly the hot fields into a narrow row and leave the 963-B struct alone. `compact_xyzh`, 5 doubles per member, is "sized to fit in the H100 L2 cache" (commit 386ecb91; mesh/sfc_tiles.h:33 `SIDX_ROW_WIDTH 5`). The fused-pass rewrite that came with it took the tiling term from 34.1 to 18.4 ms per million particles (commit dacc8a0e).

**Gather order.**
- `faces` iterates owned faces and gathers ring tetrahedra.
- If face order follows owner order (Morton-sorted cells) and tetrahedra are numbered by their emitting face (which `emit` already does: `tet_start(g)`), gathers stay local.
- Verify that the repair's free-slot reuse does not scramble this over time. If it does, renumber the tetrahedra periodically, as GIZMO does: it re-sorts by key only when order has decayed (`sidx_particles_still_ordered`, gpu_neighbor_list.cc:244; `presorted` :2079).

**`RandomAccess` / `__ldg`:** no evidence from GIZMO (0 uses). On A100, nvcc emits `ld.global.nc` when it can prove a pointer is read-only. Marking the input Views `const` + `Kokkos::MemoryTraits<Kokkos::RandomAccess>` is a cheap experiment, probably worth under 10% (I).

**Atomics in `faces`:** the two double `atomic_add`s per face on `volume` are native on A100 and probably fine. For hydro, see R5.

### R4. Allocation churn: carve scratch from one persistent block

**Expected gain:** small for steady state, but it removes jitter (I).

**GIZMO evidence:**
- Commit 9fe3a3d9 / 74b20722: "carve the gravity-walk scratch arrays from one allocation ... every call paid one allocation, free and page advice per array".
- Commit 8bc158af: moving a staging array from managed to device memory took "3178 to 100 ms per step".
- Commit b4b7ad1e: UVM page faults cost "~0.65 ms/page ... ~1.4 s fixed overhead per kernel call".

**For us:**
- `tet_repair` already persists `State`.
- Audit per-step `Kokkos::View(...)` constructions in `tet_mesh::build` and the shell builders. For example, :95 allocates `face_particles` per call, and the `stencil`/`bound2` mirrors are rebuilt every call at shell_voronoi_warp.hpp:610-617 and shell_voronoi_lean.hpp:67-74.
- Hoist these into a persistent workspace with a capacity, and grow it geometrically.
- **Never use SharedSpace/UVM for anything a kernel reads** (GIZMO's single most-measured lesson; macros.h:189-200).

### R5. MFV flux loop: one team per owner row with a custom reducer; avoid double-computing pairs

This is the design rule for the next stage.

**GIZMO hydro, for contrast (V):**
- It computes every active-active pair **twice**, once from each side, with no j-writes except atomic `dMass` and `wakeup` (hydro/hydro_functions.h:14-16, 436-470).
- Momentum and energy therefore cancel only to roundoff, and across timebins they are not conserved at all (subagent finding).
- Our owner rule (one face, applied ±) halves the Riemann work. Keep it.

**Patterns to copy:**
- `NlrModeATeamPairKernel` (mesh/neighbor_loop_runner.cc:2148-2204): one team per row, `TeamThreadRange` + `parallel_reduce` with `NlrAccumReducer` (join = merge_accum), and `Kokkos::single(PerTeam)` to publish.
- Widths come from a measured row-length census (neighbor_loop_runner.h:578-621): 32 one-way, 64 symmetric, 16 when the accumulator exceeds 512 B ("SPILL HEURISTIC").
- The reducer `static_assert`s `sizeof(AccumData) % sizeof(int) == 0`, because shuffles truncate (:2108-2111). That is worth copying verbatim.
- Commit 9136b318 explains why: a thread per row cannot coalesce `neighbors[]`, and "a wavefront runs at the longest row".

**For our faces:** a Voronoi cell has about 15 faces, not 100.
- A team of 32 per cell is too wide. A sub-warp of 4-8 lanes per cell, as a team or by hand with `__shfl`, is the analogue (I).
- Simpler still: one thread per face with ± scatter through atomics (A100 has native double atomic add), compared against a gather variant. In that variant each cell sums its owned and non-owned faces via a CSR of incident faces, with no atomics and deterministic results.
- GIZMO avoids order-dependent atomics where it can and labels the rest (`EpsilonAtomic`, hydro_force_loop.h:136-145). For bitwise reproducibility, choose the gather (I).

---

## 2. Census of Kokkos idioms in GIZMO (V)

Counts are from `grep -rE` over `src/*.cc|*.h`.

| Idiom | Uses (files) | Notes |
|---|---|---|
| `KOKKOS_INLINE_FUNCTION` | 1201 (137) | physics lives in headers ("AthenaK-style", docs/gizmo_documentation.md:858) |
| `KOKKOS_FUNCTION` | 61 (11) | |
| `KOKKOS_LAMBDA` | 140 (26) | lambdas dominate; functors for team kernels (`GravFlatTeamWalk`, `NlrModeATeamPairKernel`) |
| `parallel_for` / `reduce` / `scan` | 110 / 14 / 13 | scans are used for compaction (`nlr_mode_d_compact`, runner.cc:6820) and CSR offsets |
| `RangePolicy` explicit | 47 (6) | most launches pass a bare `int N` (`gizmo_gpu_kernel_launch`, gpu_dispatch_templates.h:31-36) |
| `TeamPolicy` | 22 (6) | always `vector_length = 1`; width carried on the team dimension for CUDA/HIP portability (gpu_dispatch_templates.h:44-62) |
| `TeamThreadRange` | 5 | no `ThreadVectorRange` / `TeamVectorRange` |
| `set_scratch_size` | 4, all level 0 | team scratch for spill-prone sums and packet stacks |
| `team_barrier` | 13 (3) | packet walk rounds |
| `Kokkos::single(PerTeam)` | 3 | |
| `Kokkos::fence()` | 103 (20) | after nearly every launch (wrapper bakes it in) |
| `deep_copy` | 81 (11) | almost always through **Unmanaged** Views wrapping raw pointers (e.g. gpu_particles_arena.cc:540-549) |
| `MemoryTraits<Unmanaged>` | 70 | raw-pointer world; Views are mostly adaptors |
| `kokkos_malloc` / `kokkos_free` | 30 / 264 | |
| `ViewAllocateWithoutInitializing` | 4 (1) | gpu_pseudo_update.cc:54-81 |
| `SharedSpace` | 144 (38) | P/CellP are canonical UVM (allocate.cc:144,167) |
| `SharedHostPinnedSpace` | 1 macro | pinned staging |
| `partition_space` | 1 | gpu_neighbor_list.cc:3048 (copy queue + eval queue) |
| `Experimental::sort_by_key` | 1 | gpu_neighbor_list.cc:1173, Morton128 comparator (merge sort on CUDA, gpu_morton.cc:150) |
| atomics | add 145, min/max 63, CAS 9, or 10 | CAS loops for double max (gpu_force_update.cc:75-89) |
| custom reducers | 2 | `NlrAccumReducer` (runner.cc:2094), `SidxBuildTally` + `reduction_identity` specialisation (gpu_neighbor_list.cc:944) |
| **zero uses** | | LayoutLeft/Right/Stride, `RandomAccess`, `Atomic` trait, DualView, `create_mirror`, MDRange, `LaunchBounds`, `Schedule<Dynamic>`, Graph, ScatterView, `Kokkos::sort`, BinSort, SIMD, `Kokkos::Array`, Profiling regions, ThreadVectorRange |

Math: a trivial-aggregate `Vec3<T>` that is layout-identical to `T[3]` (math_types/vec3.h:12-55), plus `mat3.h` and `symmetric_tensor2.h`. Plain `sqrt` is used in device code, and there are no `Kokkos::sqrt` calls (I: they rely on the CUDA math overloads). Fast-math is forbidden with an `#error` (declarations/typedefs.h:18-27).

---

## 3. Memory model: what "unified" means (V)

- **Canonical particle arrays in UVM.** `P[]` and `CellP[]` are allocated with `kokkos_malloc<SharedSpace>` (`gpu_particles_uvm_alloc`, system/gpu_particles_arena.h:186-205; allocate.cc:144,167).
  - Host and device read and write the same AoS structs in place. The "arena" became a pointer alias (commit cff2510c).
  - The labels say `particle_soa_*`, but the layout is AoS: `struct particle_data` (declarations/particle_data.h:6), about 963 B.
- **Placement policy by footprint** (gpu_particles_arena.cc:240-310):
  - `cudaMemAdvise` / `hipMemAdvise`, PreferredLocation + AccessedBy.
  - Host-preferred above 1 GiB per rank on AMD ("a 291.8-million-particle restart ... gains a factor of 4.2").
  - Device-preferred on PCIe A100 ("device-preferred placement ran 5% faster than no hint" on Lonestar6).
  - No hint on GH200.
  - The comment flags this as a "STOPGAP, NOT A DESIGN".
- **Device-only space for hot read paths.** `GIZMO_KOKKOS_DEVICE_SPACE` = CudaSpace / HIPSpace (macros.h:189-200).
  - The index arrays (tiles, BVH, pool, `compact_xyzh`) are built in a stage and `deep_copy`'d.
  - On HIP: per-call times 20228 → 126 µs and 30399 → 5277 µs; device kernel total 471.8 → 281.3 ms/step (commit bfb3de05).
- **MPI into managed memory on a PCIe A100 loses data.** "imported and exchanged particles arrived still holding what their slots held before" (gpu_particles_arena.cc:449-455). Staged receives are forced there.
- **Global parameters.** A per-TU `static __managed__` mirror of `All`, synced before each dispatch, because there is no `-rdc` (declarations/gpu_all_mirror.h:1-60).
  - Pitfall they hit: an unregistered mirror left `BoxSize = 0` on the device, which made the BVH prune always pass. That was an O(N) walk and a "1.48s/call floor" (commit b67e8111).
- **Staging batches** for subset kernels: host gather → explicit device copy → kernel → scatter (`ParticleStagingBatch`, gpu_particles_arena.h:93-170). They deliberately avoid managed memory here: "a host-side gather reads them from the side they live on".

**For us:** we are a single GPU and own all our data, so none of the UVM machinery applies. Keep everything in `CudaSpace`. The transferable lessons are the measured costs of managed memory on the read path, and of per-call allocation.

---

## 4. Execution model (V)

- **Launch wrapper.** `gizmo_gpu_kernel_launch(tag, N, f)` = `parallel_for` + `fence` + `gizmo_gpu_check_last_error` (gpu_dispatch_templates.h:31-36). Team variants probe `team_size_max` with the *same scratch request* before launching (:80-105).
  - Avoid copying the unconditional fence: it is the opposite of what our flip repair needs.
  - Copy the probe-with-scratch idea instead: our lean launch should derive its thread count from `team_size_max(f, scratch)`, not from a constant.
- **Async path.** `gizmo_gpu_kernel_launch_async(space, …)` with no fence, plus `GizmoGpuEvent::record/query` using raw `cudaEventRecord` / `cudaEventQuery` on `space.cuda_stream()` (:136-200), because "Kokkos has no portable query of whether an instance's work has finished".
  - Used in Mode D to overlap segmented device→pinned copies with host writeback (commit 3b51aa8d).
- **Team widths from data, launch shapes as named rows.** "A row the backend cannot launch selects the next named row ... never narrowed silently" (commit ae813606). Each call reports the shape it ran with.
- **Spill control.** When per-lane state spills, put it in team scratch as one slot per lane, gated at compile time (commits 78d8667e, 4cb6f26e).
  - "the dense step-0 walk 0.63x"
  - "FIRE-3 packet kernels 1000 -> 632 B/lane scratch at unchanged registers".
  - This is the opposite of our worry. Our warp kernel's 130 registers is register-limited occupancy, and moving cold state into scratch could lower registers (I).
- **Team reduction in place of atomics** for tile bounds: one team per tile, a custom reducer, one writer (commit 04157331; gpu_neighbor_list.cc:1215-1249).
- **No** `LaunchBounds` anywhere. Our warp kernel already uses it (shell_voronoi_warp.hpp:620).

---

## 5. Reductions, scans, compaction, sort (V)

- **Compaction idiom.** An exclusive `parallel_scan` with `if(final) out[i] = sum` writes offsets, and the total comes back to the host (gpu_neighbor_list.cc:2266-2280). The same shape appears in `mode_d_export.h:181` (a stable LSB radix sort built from scans), mg_gradient_correction.cc:424, 947, and gpu_topology_build.cc:397.
- **Bounded scratch with exact overflow.**
  - Neighbours are written to a per-query 512-slot scratch and counted past the cap.
  - Scan, then compact.
  - Overflowed rows re-walk straight into the final array (gpu_neighbor_list.cc:2241-2323).
  - This is a one-pass alternative to count-then-fill, and fits our cell-list or face emission (I).
- **Custom reducer for struct accumulators** (§1 R5). It checks for shuffle truncation and needs a `reduction_identity` specialisation.
- **Atomic CAS loops** for float max/min where native atomics do not exist (gpu_force_update.cc:75-95). They are not needed on A100 for int64 `atomic_max`.
- **Sort.** `Kokkos::Experimental::sort_by_key` with a 128-bit Morton comparator. Their note: on CUDA it "falls back from radix to merge-sort" (gpu_morton.cc:150-152). The sort is skipped when the order is still valid.
  - Our counting sort on a uniform grid is cheaper. Keep it.

---

## 6. Portability and robustness tooling worth copying (V)

1. **Kokkos Tools allocation telemetry** (system/kokkos_mem_telemetry.cc:111-136): `Kokkos::Tools::Experimental::set_allocate_data_callback` / `set_deallocate_data_callback`.
   - Classifies every allocation by **label prefix** and memory space.
   - Tracks current and high-water marks.
   - Has an UNCLASSIFIED remainder.
   - Feeds a memory ledger (system/memory_ledger.cc).
   - About 100 lines, and gives us a per-stage memory high-water for free. **High value for "memory-lean" goals.**
2. **Device error sentinel** (declarations/gpu_device_error_sentinel.h): a device `atomicCAS` first-set-wins records code+line into managed memory, then the host consumes it after the fence. This replaces `Kokkos::abort` for recoverable stops.
   - We currently call `Kokkos::abort("tet_repair: capacity exceeded")` on the host side (tet_repair.hpp:531); device-side overflow guards are silent (`if (at < capacity)`).
   - A counter of dropped writes, read with the termination readback, would make those visible.
3. **Bitwise A/B gates in every perf commit.** "every one of 960 snapshot datasets is identical" (1b05ea38). Flat-vs-hierarchical oracles kept behind env flags (efb49ccf). Planted faults to prove the comparison can fail (dacc8a0e).
   - Copy the habit: each optimisation gets an exact compare against the previous kernel on fixed inputs, plus one planted fault.
4. **Dirty bitset tracker** (mesh/gpu_dirty_tracker.h): per-field dirty bits, consumed by `ctzll`, promoted to "all dirty" above a threshold. This is relevant to incremental mesh updates (only re-clip cells whose neighbourhood moved) (I).
5. `static_assert`s on layout and size (Vec3, AccumData), and `#error` gates for unsupported configurations (commit 80354891).

---

## 7. What NOT to copy

- **Fence after every launch** (gpu_dispatch_templates.h:34). It costs us exactly in R1.
- **UVM / SharedSpace for anything hot.** GIZMO needs it for MPI and host physics; we do not.
- **AoS mega-structs** (about 963 B per particle). GIZMO itself had to build narrow rows (`compact_xyzh`) to get bandwidth.
- **Per-TU `__managed__` global mirrors.** These only exist because there is no `-rdc`. Pass parameters by value in functors, as we already do.
- **Kernel-based effective faces, both-sides flux evaluation and the pairwise min/max limiter tuned for kernel faces.** We have exact faces and the owner rule.
- **Host-side h/after-iteration steps and host kicks.** They are not relevant: we have no h-iteration, and our timestep logic should stay on the device.
- **`sort_by_key` with a comparator on CUDA** (merge sort). Our counting sort is better.

---

## 8. Suggested experiment order (cheapest first)

1. **R3 layout.** Pad `Positions` and `centre` to `double[4]` LayoutRight, and time `circumcentres` and `faces`. About one hour of work.
2. **R1a + R1b.** Use a persistent pinned counters mirror and device-bounded launch sizes, so there is one readback per round. Profile with nsys: the cudaMemcpy count should fall by about 4x.
3. **R2(i).** Lean kernel with the polyhedron in level-1 scratch, or in a carved global workspace, at 64/128 threads per block. Compare cells/s and achieved occupancy.
4. **R2(iv).** Move the warp kernel's per-lane staging arrays into per-lane shared-memory slots, or ping-pong polyhedra, with 2 warps per block. Measure registers and occupancy.
5. **R1c.** Use a Kokkos Graph for the repair round once it is host-decision-free.
6. **§6.1.** Add the allocation telemetry callbacks to the test drivers.
