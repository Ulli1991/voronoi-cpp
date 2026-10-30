# GIZMO Kokkos GPU on a PCIe-attached A100: from 6.23 to 2.9 s/step

Date: 2026-10-30. Code: GIZMO `gpu_kokkos_unified` at b653e863, the same export as in `gizmo_gpu_benchmark.md`. Machine: Raven, 1x A100-SXM4-40GB behind an x86 Xeon 8360Y. The GPU's managed memory is not host-coherent (`cudaDevAttrDirectManagedMemAccessFromHost = 0`), so for this purpose it behaves like a PCIe A100. Test: Sedov 128^3 (2,097,152 particles), MFM, no gravity, global timesteps, 1 MPI rank with 18 OpenMP threads; s/step is the mean of steps 2-17 of `cpu.txt`, as in the benchmark note.

Everything is in `/raven/u/uli/gizmo-bench/`:
- `src-opt/`: the modified export. It is a local git repo whose first commit is the baseline export; `git diff` there is the patch.
- `build-opt/`: GPU build; binaries `GIZMO.v1` ... `GIZMO.v7`, one per measured state.
- `build-opt-cpu/`: the same source, Kokkos-OpenMP build.
- `gizmo_gpu_pcie_optimizations.patch`: the patch to send upstream. Everything in it is behind the Config option `GPU_PCIE_OPTIMIZATIONS`, off by default; see (e). The first version, without the option, is `gizmo_pcie_speedup.patch`. It applies with `git apply` to b653e863 (checked with `git apply --check`).
- `analysis/`: the nsys sqlite scripts used below (`um_attrib.py`, `um_timeline.py`, `cpu_window.py`, `cpu_samples.py`, `hot_lines.py`) and `snapdiff.py`.
- `runs/<case>/`: logs, `cpu.txt`, final snapshots; nsys profiles in `runs/mfm_sedov128_gpu_nsys` (baseline), `runs/opt3_hd_nsys`, `runs/opt5_hd_nsys` and `runs/opt6_hd_nsys`.
- `src/`, `build-mfm/` and the baseline runs are unchanged, and `/raven/u/uli/gizmo` was not touched.

## TL;DR

| Run (Sedov 128^3, 1 A100, 1 rank x 18 threads unless noted) | s/step | vs baseline |
|---|---|---|
| GPU baseline (b653e863) | **6.23** (repeat 6.19) | 1.00 |
| CPU, same source, 18 ranks x 1 thread | **3.96** | 1.57x |
| GPU baseline, 2 ranks x 9 threads on the one GPU (no code change) | 5.58 | 1.12x |
| GPU baseline + `GPUDriftOffloadMinCount 1000000000` (drift on host; parameter only) | 6.06 | 1.03x |
| GPU baseline, `GPU_PARTICLE_STORAGE_PLACEMENT=1` (host-preferred) | 10.33 | 0.60x |
| **GPU + patch (v7, default parameters)** | **2.89** (repeat of the same code path on another node: 2.96) | **2.16x** |
| GPU + patch, bulk prefetch switched off (`GIZMO_UVM_PREFETCH=0`; this also returns the drift to the device) | 3.34 | 1.87x |
| CPU + patch, 18 ranks x 1 thread / 1 rank x 18 threads | 3.87 / 4.12 (baseline 3.96 / 5.96) | |

- With the patch the GPU build runs a step in 2.89 s against 3.96 s for the same source on 18 CPU cores, 1.37x faster than the CPU. It was 1.57x slower than the CPU before.
- Physics is unchanged:
  - with drift on the device (the code path the baseline takes), the final snapshot is bitwise identical to the baseline GPU run;
  - the patch's default moves the drift to the host when the particles are there (below). That changes 846 of 6.3M velocity components and 180 of 2.1M internal energies at the 1e-16 relative level (FMA contraction on the device, none on the host). The total energy agrees to 1e-17.
  - The CPU builds with the patch are bitwise identical to the CPU baselines.
- Where the time went:
  - Fault-driven unified-memory traffic fell from 11.9 GB/step to 1.3 GB/step.
  - The remaining traffic is about 10.5 GB/step of explicit bulk prefetches. They run at 17-18 GB/s and take about 0.47 s of copy-engine time per step.
  - GPU kernel time per step fell from 1.65 s to 0.58 s: the kernels no longer stall on GPU page faults.

**Upstream form:** everything is behind the Config.sh option `GPU_PCIE_OPTIMIZATIONS`, off by default; see (e).
- Without the option the GPU and CPU builds are bitwise identical to the baseline.
- With it the GPU build runs at 2.92-3.02 s/step and is bitwise identical to v7.
- **Only a Sedov blast wave was tested:** 128^3, MFM, 1 rank x 18 threads on one A100 with non-coherent managed memory, no gravity, global timesteps.

## (a) Where the migration came from (baseline, nsys)

Source: the nsys profile of the baseline (`runs/mfm_sedov128_gpu_nsys`, 8 steps, 7.1 s/step under the profiler), steady window t = 20-76 s (7.9 steps).

Method:
- Each unified-memory D2H transfer is attributed to the host function whose CPU page fault came last before it (`CUDA_UM_CPU_PAGE_FAULT_EVENTS` records the faulting function).
- Each H2D transfer is attributed to the kernel running when it happened.
- Host time comes from nsys CPU sampling, resolved to source lines with `addr2line`.
- Scripts: `analysis/um_attrib.py`, `um_timeline.py`, `cpu_window.py`, `hot_lines.py`.

Almost all of the traffic (92.5 GB of the 94 GB) is driver "speculative prefetch" triggered by faults; explicit memcpys are negligible.

**D2H, 8.3 GB/step: the host touching particles that a kernel had pulled to the device.**

| Host phase (function) | GB/step | What the host does there |
|---|---|---|
| Density h-iteration control (`DensitySpec::after_iter_body`, called from the single-rank `run_neighbor_loop_iterative`) | 1.43 | Per-active convergence check after each Mode D pass. It ran on **one thread**: about 0.6 s of faulting per step. |
| Gradient writeback (`GradientsSpec::apply_active_writeback`) | 1.19 | Six interleaved chunks, because of `NLR_MODE_A_STAGING_BYTES_CAP` = 128 MB: writeback, then kernel, then writeback, so the same pages ping-ponged six times. |
| Force writeback + `hydro_final_operations_and_cleanup` | 0.83 + 0.68 | Per-active finalize. |
| Drift: host scan after the device drift kernel (`drift_particles_batch`) + `merge_and_split_particles` | 0.68 + 0.75 | The device drift kernel sat between two host phases: kicks before, merge/split and domain after. |
| `hydro_gradient_calc` + `construct_gradient` | 0.69 + 0.31 | Gradient finalize. |
| Neighbour-list query setup (`gpu_ngb_list_build`, host loop over `P[i].Pos`) | 0.68 | Host loop sitting between two device phases. |
| Tree per-type hmax refresh (`force_refresh_hmax_per_type_host`, one thread) | 0.44 + 0.27 | Runs right after the device tree build. |
| Mode D single-rank result readback | 0.22 | |
| Tree moment writeback, finalize, misc | ~0.1 | |

**H2D, 3.75 GB/step: kernels faulting particles back.**

| Kernel | GB/step |
|---|---|
| Density, Mode D: team pair kernel, local walk, reduce | 0.88 + 0.56 + 0.46 |
| Gradients, `run_mode_a` | 0.81 |
| Drift kernel | ~0.5 (in the reduce figure above; seen directly in the timeline) |
| Tree build: `gx_leaf_mirror_current`, `gpu_topology_prepare` | 0.38 + 0.19 |
| Force, `run_mode_a` | 0.24 |
| Spatial index, neighbour list | 0.18 |

**Host work that was slow for reasons other than faults:**
- **`schedule(dynamic)` with chunk 1** on per-particle loops over 2M particles, in kicks, `do_box_wrapping`, `hydro_final_operations`, the force preloop, the density finalize, the gradient finalize and `force_update_hmax`. Example: `do_the_kick` took 0.36 s/step on 18 threads with no faults at all. The CPU 1x18 build pays the same: 0.25 s.
- **Serial Peano-Hilbert key loops** in `domain_Decomposition_light`: two passes over all particles, about 0.5 s/step on one thread.
- **Serial loops after the tree build:** `force_refresh_hmax_per_type_host`, the active-gas count in `hydro_corridor.cc`, `density_build_active_list`.
- **Tree passes on one host thread.** Two tree passes (`topo_father_init_parts`, `node_reset_ephemeral`) use `RangePolicy<DefaultHostExecutionSpace>`. That is `Kokkos::Serial` in a CUDA build, so they ran on one host thread over managed memory between device passes. The source comment says this was measured on AMD only.

GIZMO's placement hint is no help here. Device-preferred placement does not stop a host access from migrating the page: an x86 CPU cannot map device memory. Host-preferred placement (`GPU_PARTICLE_STORAGE_PLACEMENT=1`) is much worse, at 10.3 s/step, because every kernel then reads particles across PCIe.

## (b) Changes, where they are, and measured s/step

All s/step values are Sedov 128^3 on one A100, mean of steps 2-17. "hd" means the drift ran on the host: in v1-v6 via the parameter `GPUDriftOffloadMinCount 1000000000`, and in v7 by default.

| State | Contents (cumulative) | s/step (drift on device) | s/step (hd) |
|---|---|---|---|
| baseline | | 6.23 | 6.06 |
| v1 | Changes 1-8 below | 3.92 | **3.36** (prefetch off: 3.51) |
| v3 | + 9, 10 | | 3.11 (repeat on a shared gpudev node: 3.18) |
| v5 | + 11-13 | 3.36 | 3.04 |
| v6 | + 14-16 | 3.18 | **2.89** (prefetch off: 3.05) |
| v7 | + 17 (host drift by default) | 3.34 (prefetch off, which also returns the drift to the device) | **2.89** (default parameters) |

**The changes**, by file, with the measured effect where it could be isolated.

Host-loop fixes. All are bitwise neutral: the work is per particle, and reductions are integer, min or max.

1. **Chunked dynamic schedules.** `schedule(dynamic)` became `schedule(dynamic, 256)` in:
   - `core/kicks.cc` (both kicks);
   - `core/predict.cc` (`do_box_wrapping`);
   - `core/timestep.cc` (the fast-wave loop);
   - `hydro/hydro_toplevel.cc` (`hydro_final_operations_and_cleanup`, `hydro_force_initial_operations_preloop`);
   - `hydro/density.cc` (`cellcorrections_final_operations_and_cleanup`);
   - `hydro/density_loop.cc` (`density_finalize_post_runner`);
   - `hydro/gradients_loop.cc` (the `construct_gradient` loop);
   - `gravity/forcetree_update.cc` (`force_update_hmax`; added in v3).

   Kicks went from 0.36 to 0.06 s/step and `hmaxupdate` from 0.15 to 0.03. The CPU 1x18 build gains the same way.
2. **Threaded Peano keys and the move-flag loop** in `domain_Decomposition_light` (`domain/domain.cc`), as the full decomposition already does. Domain went from 0.50 to 0.19 s/step together with 1 and 7.
3. **Threaded host `after_iter`** (`mesh/neighbor_loop_runner.cc`, the iterative runner's host branch):
   - It applies to Specs that can iterate on the device (`nlr_mode_d_iterates_on_device_v`). Their `after_iter` touches only its own active, by contract.
   - The results are computed on several threads into a per-slot array. The compaction of the live list stays serial and in order, so the lists and radii are identical to the one-thread loop.
   - Single-rank runs hit this branch because the device iteration is gated on `NTask > 1`.
4. **Mode A staging cap sized to the device** (`neighbor_loop_runner.cc`, `nlr_mode_a_staging_bytes_cap()`). The cap is max(128 MB, 1/8 of free device memory). A full-active gradient loop now runs in one chunk instead of six, which removes five writeback/kernel ping-pongs. Chunking is documented as result-neutral (K = N reproduces the unchunked path).

Bulk prefetch. These are placement hints only.

5. **Bulk prefetch API** (`system/gpu_particles_arena.{h,cc}`):
   - `gizmo_particles_prefetch_to_device/_to_host/_P_to_device(site)` does a `cudaMemPrefetchAsync` of `P[0,NumPart)` and `CellP[0,N_gas)`, fenced and synchronous.
   - `gizmo_managed_prefetch_to_host(ptr, bytes, site)` does the same for one managed buffer.
   - `gizmo_particles_prefetch_side()` reports where the last bulk move went.
   - On by default only when the particle storage is CUDA UVM and not host-coherent. It is a no-op on GH200, HIP and CPU builds. `GIZMO_UVM_PREFETCH=0|1` overrides it.
   - A prefetch of pages that are already resident is nearly free, so a redundant call costs little.
6. **Where the prefetches are called:**
   - **to the device:**
     - in the runner, before each of the five `gpu_particles_arena_acquire` calls (Mode A/D neighbour loops), when `num_active * 8 >= NumPart`;
     - in `gravity/gpu_topology_build.cc` `topo_acquire_` before the tree build (P only from v6, since the build reads no cell data);
   - **to the host:**
     - at `nlr_apply_active_writebacks`, together with the kernel's accumulator buffer (v5);
     - at the threaded `after_iter`;
     - in v5, `force_refresh_hmax_per_type_host` (`gravity/forcetree.cc`), which starts the host stretch between the tree build and density;
     - in v6, the Mode D single-rank result readback.

   With everything else in place, prefetching is worth 0.15 s/step (v6 hd: 3.05 without, 2.89 with). On its own it is not the main lever: the main lever is that the host phases no longer fault one page at a time on one thread.
7. **Device-side neighbour-list query setup** (`mesh/gpu_neighbor_list.cc`, `gpu_ngb_list_build`). The per-active radius/position/`Ti_current` loop runs as a Kokkos `parallel_reduce` when:
   - the particles were last moved to the device (v6; in v1-v5 it ran whenever the inputs were device-visible);
   - the positions are the particles' own;
   - there are at least 4096 queries.

   Host-only caller radii are staged into the managed radius array first (v5). The values written are the same.
8. **Tree topology passes on the device for CUDA** (`gravity/gpu_topology_finalize.cc`): `GxTreeHostRange = RangePolicy<>` under `KOKKOS_ENABLE_CUDA`, unchanged elsewhere. This answers the open "measure it on NVIDIA" note in that file. Tree build went from 0.60 to 0.28-0.30 s/step together with 6.
9. **`force_refresh_hmax_per_type_host`** seed loop threaded with an atomic max (CAS on `MyFloat`). A maximum is order-independent, so the bands are identical. (v3)
10. **`domain_record_step_work`** hoisted out of the serial timestep-bin loop in `find_timesteps` into a threaded pass. It writes only its own particle and reads its pre-change `TimeBin`. (v3)
11. **Threaded active-gas counts** in `hydro/hydro_corridor.cc` (integer reduction) and **threaded `density_build_active_list`** in `hydro/density_loop.cc` (predicate threaded, ordered serial compaction). (v5)
12. Prefetch to the host at the start of the post-tree host stretch (see 6). (v5)
13. Accumulator buffer prefetched with the writeback (see 6). (v5)
14. Neighbour-list setup runs on the device only when the particles are there (see 7). (v6)
15. Mode D single-rank accumulator readback prefetched (see 6). (v6)
16. Tree build prefetches P only (see 6). (v6)
17. **Drift where the particles are** (`core/gpu_drift.cc`). `drift_particles_batch` takes the host path when the last bulk move sent the particles to the host (`gizmo_particles_prefetch_side() == 2`). In the step order that is always true at the main drift, which sits between the force writeback and kicks on one side and merge/split and domain on the other.
    - With the device drift, the kernel faulted 0.5 GB to the device and the next host phase faulted 1.4 GB back.
    - The same can be had without the patch from the parameter `GPUDriftOffloadMinCount` (baseline: 6.23 to 6.06 s/step).
    - This is the only change that alters arithmetic: the host drift has no FMA contraction.
    - `GIZMO_UVM_PREFETCH=0` restores the device drift.

**Per-bucket times** (s/step, mean of steps 2-17). "1x18" is 1 rank x 18 threads; "18x1" is 18 ranks x 1 thread.

| bucket | GPU baseline | GPU v6, device drift | GPU v6 hd | GPU v6 hd, prefetch off | CPU 18x1 baseline | CPU 1x18 baseline | CPU 1x18 + patch |
|---|---|---|---|---|---|---|---|
| total | 6.234 | 3.183 | 2.894 | 3.048 | 3.964 | 5.956 | 4.124 |
| treebuild | 0.599 | 0.282 | 0.301 | 0.314 | 0.070 | 0.353 | 0.305 |
| treewalk | 0.089 | 0.092 | 0.091 | 0.081 | 0.004 | 0.031 | 0.031 |
| density (host side of the h iteration) | 1.484 | 0.747 | 0.746 | 0.763 | 0.623 | 1.257 | 0.874 |
| hmaxupdate | 0.152 | 0.034 | 0.034 | 0.033 | 0.012 | 0.147 | 0.034 |
| ngb build | 0.181 | 0.184 | 0.193 | 0.183 | 1.299 | 0.812 | 0.802 |
| gradients | 0.608 | 0.353 | 0.352 | 0.444 | 0.173 | 0.179 | 0.080 |
| hydro_frc | 0.678 | 0.278 | 0.276 | 0.349 | 0.138 | 0.287 | 0.086 |
| pair kernel (device) | 0.307 | 0.146 | 0.151 | 0.146 | 1.199 | 1.206 | 1.199 |
| misc_hydro | 0.091 | 0.094 | 0.090 | 0.091 | 0.011 | 0.077 | 0.078 |
| domain | 0.497 | 0.228 | 0.194 | 0.198 | 0.046 | 0.441 | 0.184 |
| drift/splitmg | 0.546 | 0.331 | 0.096 | 0.092 | 0.050 | 0.309 | 0.097 |
| kicks | 0.363 | 0.063 | 0.061 | 0.059 | 0.044 | 0.251 | 0.040 |
| find_timestep | 0.239 | 0.215 | 0.211 | 0.203 | 0.022 | 0.236 | 0.214 |
| misc | 0.398 | 0.134 | 0.098 | 0.094 | 0.037 | 0.366 | 0.099 |

The pair kernels themselves got 2x faster (0.31 to 0.15 s): they no longer take GPU page faults.

**Migration after the patch** (v6 hd under nsys, 3.17 s/step, window t = 18-37 s, `runs/opt6_hd_nsys`):

| | baseline GB/step | v6 GB/step |
|---|---|---|
| D2H, fault-driven | 8.1 | 0.84 (0.67 of it the host neighbour-list setup of the corridor's symmetric list) |
| H2D, fault-driven | 3.7 | 0.46 |
| D2H, explicit prefetch | 0 | 5.6 |
| H2D, explicit prefetch | 0 | 4.9 |
| copy-engine time | 0.76 s | 0.55 s (0.47 s of it prefetch) |
| GPU kernel time | 1.65 s | 0.58 s |

There are four host/device round trips of the particle arrays per step:
- density (device), then h-iteration control and finalize (host);
- gradients (device), then gradient finalize and force preloop (host);
- force (device), then writeback, finalize, kicks, timestep, drift and domain (host);
- tree build (device), then hmax refresh and active lists (host).

Each one-way move of 1.4 GB takes about 80 ms.

## (c) Correctness

Final snapshots at t = 3e-5 (17 steps) were compared particle by particle, matched by ID (`analysis/snapdiff.py`). "Differing" counts elements that are not bitwise equal.

| A vs B | Coordinates | Velocities | Density | InternalEnergy | Etot (A / B) |
|---|---|---|---|---|---|
| GPU baseline vs its repeat | 0 differ | 0 | 0 | 0 | identical |
| GPU baseline vs v1 / v6, device drift | **0 differ** | **0** | **0** | **0** | identical |
| GPU baseline vs v6 hd / v7 default | 0 differ | 846 of 6.3M, max rel 3.2e-16 | 1 of 2.1M, 6e-17 | 180, 2.9e-16 | 0.9998443121825042 / same; Ekin differs in the 16th digit |
| CPU 18x1 baseline vs v6 hd | 0 differ | 5.08M, max rel 4.8e-16 (as GPU baseline vs CPU: 5.08M, 5.6e-16) | 622k, 8.7e-16 | 615, 2.9e-16 | same to 1e-17 |
| CPU 18x1 baseline vs CPU + patch | 0 | 0 | 0 | 0 | identical |
| CPU 1x18 baseline vs CPU + patch | 0 | 0 | 0 | 0 | identical |

- Mass is exactly 1 in all runs.
- The total momentum magnitude stays at 8.5-8.6e-18.
- The patched GPU run is as close to the CPU run as the baseline GPU run was.

## (d) The patch

`/raven/u/uli/gizmo-bench/gizmo_pcie_speedup.patch` (the first version, always on; superseded by the Config-option patch in (e)): 18 files, 308 lines added and 24 removed, made with `git diff` against the baseline export (paths `src/...`, applies at the repository root with `git apply`). The SYSTYPE blocks for Raven are in the baseline export commit and are not part of the patch.

What to tell Phil, in short:
1. **On non-coherent managed memory (PCIe A100/H100 on x86), most of the loss was host-side, not PCIe bandwidth.**
   - Host loops faulted pages one at a time, often on one thread (h-iteration control, Peano keys, hmax refresh, the tree "host-range" passes).
   - `schedule(dynamic)` with chunk 1 cost 0.1-0.3 s per loop at 2M particles.
   - Those fixes alone (prefetch off) take the step from 6.2 to 3.0-3.5 s, and they help the 1-rank CPU build too (5.96 to 4.12 s).
2. **Bulk prefetch at the existing arena acquire points** (neighbour loops, tree build) and at the writebacks turns the remaining fault storms into about 80 ms copies. It is gated to non-coherent CUDA UVM, so GH200 and Frontier are untouched.
3. **The Mode A staging cap of 128 MB splits a 2M-active gradient loop into six chunks,** each paying a host/device page round trip. Sizing the cap to device memory removes that.
4. **The main drift should run where the particles are.** Between two host phases that is the host.
5. **Open:**
   - the device iteration for single-rank Mode D (`NTask > 1` gate at `nlr_mode_d_iteration_begin`);
   - the serial `get_timestep` loop in `find_timesteps` (0.2 s/step);
   - the remaining four round trips per step, which disappear only if the host phases between the neighbour loops move to the device.

## (e) The Config option `GPU_PCIE_OPTIMIZATIONS` (the patch to send upstream)

**Patch to send:** `/raven/u/uli/gizmo-bench/gizmo_gpu_pcie_optimizations.patch`.
- It supersedes `gizmo_pcie_speedup.patch`.
- It has a header that says what was and was not tested.
- 21 files, 329 lines added and 23 removed. `git apply --check` passes against b653e863.
- Source: `src-opt` at commit 173cb83 (the baseline) plus the working tree.

Every change in (b) now takes effect only when `GPU_PCIE_OPTIMIZATIONS` is in Config.sh. The option is off by default. The gating is done in four ways:
- **Two helpers in `core/proto.h`:**
  - `gizmo_pcie_opt()` returns 1 with the option and 0 without.
  - `gizmo_omp_particle_chunk()` returns 256 with the option and 1 without; `dynamic` with chunk 1 is OpenMP's default dynamic schedule.
  - The changed loops read them in their OpenMP clauses: `schedule(dynamic, gizmo_omp_particle_chunk())` and `if(gizmo_pcie_opt() && ...)`. Without the option those loops keep their original schedule, and the newly threaded ones run on one thread in the original order.
  - Functions rather than macros are used because the pragmas are also compiled by nvcc.
- **Small `#ifdef GPU_PCIE_OPTIMIZATIONS` regions:**
  - the prefetch enable in `gpu_particles_arena.cc`, including the `GIZMO_UVM_PREFETCH` override and the log line;
  - the device-sized staging cap;
  - the device range of the tree-topology passes;
  - the device neighbour-list setup;
  - the hoisted `domain_record_step_work`; without the option the call stays inside the bin loop, as before.
- **Calls that become no-ops:** every prefetch call, and `gizmo_particles_prefetch_side()`, which then always returns 0. The drift placement and the neighbour-list setup route depend on that side, so they fall back to the baseline path.
- **Documentation:** the option is described in `src/Template_Config.sh` and `docs/gizmo_documentation.md`, next to `GPU_PARTICLE_STORAGE_PLACEMENT`. The description says what the option does and that it helps only on GPUs with non-coherent managed memory. It also states that **only a Sedov blast wave was tested** (128^3, MFM, 1 rank x 18 threads on one A100 with non-coherent managed memory, no gravity, global timesteps) and lists the untested paths:
  - other problems and ICs;
  - MFV and the other hydro methods;
  - gravity;
  - multiple ranks per GPU, and multi-node runs;
  - local timesteps;
  - all other physics modules;
  - GH200 and other coherent systems, and AMD GPUs, with the option on.

**Verification.** All builds are full rebuilds of the final source: `build-pcie-off`, `build-pcie-on`, `build-pcie-off-cpu`. All runs are Sedov 128^3 to t = 3e-5 with final snapshots compared bitwise, field by field, matched by particle ID (`analysis/snapdiff.py`).

| Build | s/step | Final snapshot |
|---|---|---|
| GPU, option off | 6.37 (ravg1075; baseline 6.23 / 6.19 / 6.37) | **bitwise identical** to the unpatched GPU run, all 9 fields |
| GPU, option on | **3.02** (ravg1075, sharing the node with the option-off run); repeat **2.92** (ravg1131, also bitwise identical to v7) | **bitwise identical** to the earlier patched run (v7, 2.89 s) |
| CPU, option off, 18 ranks x 1 thread | 3.86 (baseline 3.96) | **bitwise identical** to the unpatched CPU run |
| CPU, option off, 1 rank x 18 threads | 5.89 (baseline 5.96) | **bitwise identical** to the unpatched CPU run |

The option-off GPU build does not print the prefetch line. The option-on build prints "bulk prefetch ... is on".

## Caveats

- **Node and noise.**
  - The baseline was measured on gpudev (ravg1002, 6.23 s) and repeated on gpu1 (6.19 s).
  - Most patched runs ran on gpu1 nodes: ravg1041, 1052, 1063, 1131, 1151, 1166, 1186, 1196, 1201.
  - Repeats:
    - v3 on the shared gpudev node: 3.18 s against 3.11 s on gpu1;
    - v6 hd: 2.89 / 2.96 s;
    - the baseline on a gpu1 node: 6.37 s against 6.23 / 6.19 s.
  - Expect about 2-3 % node-to-node variation.
- **Only this configuration was tested:** MFM Sedov, one rank per GPU, global timesteps, no gravity. Multi-rank, gravity, MFV, sinks and individual timesteps take other code paths that the patch touches only through the shared runner, arena and drift code.
  - The runner changes are gated on `num_active * 8 >= NumPart`.
  - The device-iteration-eligible Specs are AGS density, DM dispersion and the radiation-pressure loop besides hydro density. Their threaded host `after_iter` relies on the same per-active contract their device iteration already relies on.
- **The 2-ranks-per-GPU baseline** (5.58 s) is shown only as a no-code-change reference. 4 ranks x 4 threads ran out of device memory: the 64 KB CUDA thread stack in each of four contexts.
- **`GPU_PARTICLE_STORAGE_PLACEMENT=1`** (host-preferred) was built and run once (10.3 s/step) to confirm that placement alone does not fix this.

## Update 2026-10-30/31: current head, dd6dc7b8 backport, turbulent box (GIZMO issue #78)

**Head check.** On 2026-10-30, `git ls-remote https://github.com/pfhopkins/gizmo gpu_kokkos_unified` returned b653e863. That is the commit the patch was made against, checked again right before finishing. `gizmo_gpu_pcie_optimizations.patch` passes `git apply --check` on a fresh `git archive` of b653e863 (`ext/head-b653e863`).

**Patches and branch** (all in `/raven/u/uli/gizmo-bench/`):
- `gizmo_gpu_pcie_optimizations.patch`: for b653e863.
  - Code unchanged from (e).
  - The header and the option text in `Template_Config.sh`/`docs` now carry the turbulence results and the tested/untested scope.
  - The source is `src-opt` 173cb83..b677dc8. The previous Sedov-only version is kept as `ext/gizmo_gpu_pcie_optimizations.patch.2026-10-30-sedov-only`.
- `gizmo_gpu_pcie_optimizations_dd6dc7b8.patch`: the backport.
  - Source: `src-dd6` aea6726..ddd4b09. aea6726 = the dd6dc7b8 export plus Raven SYSTYPE blocks, which are not in the patch.
  - It passes `git apply --check` on a fresh `git archive dd6dc7b8`.
- `ext/gizmo-pr`: a fresh clone of `gpu_kokkos_unified`.
  - Branch `gpu-pcie-optimizations` holds one commit, 7472ec01, on top of b653e863. It is authored by Ulli, has no trailer, and was not pushed.
  - Its diff is identical to the b653e863 patch.
- PR text draft: `pr_description_draft.md`. Issue comment draft: `issue78_comment_draft.md`. Nothing was posted.

**Backport to dd6dc7b8** (Mike Grudić's tree, 131 commits older). 19 of the 21 files applied as they were. Three hunks were redone by hand:
- `core/proto.h`: the two helpers, inserted at a new anchor.
- `core/gpu_drift.cc`: dd6dc7b8 gates the drift on `n_need < GPU_MIN_PARTICLES_FOR_DRIFT_OFFLOAD`. `|| gizmo_particles_prefetch_side() == 2` was added to that condition.
- `mesh/neighbor_loop_runner.cc`:
  - dd6dc7b8 has no `nlr_apply_active_writebacks`. The writeback prefetch became a helper, `nlr_prefetch_for_writeback<Spec>`, called before the four host writeback loops: Mode B local, Mode B remote, Mode D, and the Mode A chunk. The writeback loops themselves stay serial, as in dd6dc7b8.
  - dd6dc7b8 has no device iteration (`after_iter_device`), so the threaded host `after_iter` cannot key on it. It uses a new opt-in trait instead, `Spec::after_iter_own_active_only`, set only in `DensitySpec` (`hydro/density_loop.h`). The audit found that `DensitySpec::after_iter` reads P/CellP/accum of its own active and writes only its own scratch.
  - The `AfterIterContext` of dd6dc7b8 has 8 fields, without call slot and device context.
- The cleanly applied hunks were checked by hand: `gpu_neighbor_list` has the same host loop body, the runner acquires are at the same places, and the drift placement default of dd6dc7b8 on PCIe CUDA is also device-preferred.

**Turbulent box setup.**
- IC generator, Config and params: `test/turbulent_box` of `starforge_dev` 63eb32e, exported to `ext/starforge_dev-63eb32e`.
  - The upstream generator imports matplotlib/pytest through `gizmo.test`. `scripts/make_turbbox_ics.py` is a verbatim copy of `make_ics` + `write_ics` without those imports.
  - The ICs are `ics/turbulent_box_128.hdf5` (M = 3875.33 Msun, as in the README).
- Config: Mike's, i.e. `BOX_PERIODIC SELFGRAVITY_OFF HYDRO_MESHLESS_FINITE_MASS EOS_GAMMA=1.001 EOS_ENFORCE_ADIABAT=0.04 TURB_DRIVING`, plus `OUTPUT_IN_DOUBLEPRECISION` for the bitwise checks (I/O only).
- Params (`scripts/param_turbbox_template.txt`): Mike's changes (TimeMax 0.4, MaxMemSize 4000, PartAllocFactor 2.0), plus:
  - `TimeOfFirstSnapshot 100`, so only the final snapshot is written;
  - `MaxSizeTimestep 0.0125`, so the run is exactly 32 steps with all 2,097,152 particles active every step. Confirmed from `Nactive` in `cpu.txt`.
- Builds: `build-tb-{base,off,on,base-cpu}` and `build-dd6-{base,off,on}`, all on gpu1 with make -j18, modules as before (gcc/14, openmpi/4, hdf5-serial/1.14.1, cuda/12.8 to match the Kokkos 4.6.01 install).
- Run script: `scripts/run_turb.sh`. It samples `nvidia-smi` every second into `gpu_util.csv`. The util figure averages the whole job, including init and the final snapshot.
- s/step is the mean of steps 2-32 from `cpu.txt` (`scripts/extract.py`).
- Job scripts: `scripts/job_i78_*.sh`, jobs 31692758/60/61/65/66/67 and 31693502. Logs are in `runs/i78-<jobid>.out`.

**Turbulent box 128^3, b653e863, one A100**

| run | node | s/step | GPU util | vs GPU baseline |
|---|---|---|---|---|
| CPU build (Kokkos OpenMP), 18 ranks x 1 thread | ravg1159 | **3.96** | - | |
| GPU baseline, 1 x 18 | ravg1038 | **8.01** (repeat on ravg1010: 7.85) | 32 % | 1.00 |
| GPU, patch with option off, 1 x 18 | ravg1055 | 8.11 | 30 % | |
| **GPU, option on, 1 x 18** | ravg1038 (same node as baseline) | **4.43** | 33 % | **1.81x** |
| GPU baseline, 2 x 9 on the one GPU | ravg1005 | 6.62 (repeat on ravg1007: 6.60) | 50 % | |
| GPU, option off, 2 x 9 | ravg1007 | 6.59 | | |
| GPU, option on, 2 x 9 | ravg1005 | 4.34 | 56 % | 1.52x vs 2x9 baseline |

- Per-bucket times (1 x 18, baseline → option on, s/step):

  | bucket | baseline | option on |
  |---|---|---|
  | density | 1.64 | 0.84 |
  | hydro_frc | 1.95 | 1.54 |
  | gradients | 0.67 | 0.38 |
  | treebuild | 0.61 | 0.28 |
  | drift | 0.76 | 0.26 |
  | domain | 0.51 | 0.20 |
  | misc | 0.63 | 0.12 |
  | kicks | 0.16 | 0.04 |
  | pair kernel | 0.33 | 0.14 |

- With the option on, `hydro_frc` (1.54 s) is the largest bucket. In this Config it contains the turbulence driving acceleration: `turb/turb_driving.cc` gathers whole `particle_data` + `gas_cell_data` records of all actives into compact SharedSpace arrays on one host thread, runs the kernel, and scatters them back. This is the same path Mike's issue lists, and the patch does not touch it. It is the obvious next target: the CPU build spends 0.48 s in `hydro_frc`.
- The CPU 18x1 build is still about 10 % faster on this problem (3.96 vs 4.43 s). On Sedov the patched GPU build is 1.34x faster than the CPU build.

**Sedov 128^3 control, same node** (ravg1055, job 31692765; binaries `build-pcie-off`/`build-pcie-on` from (e)):

| | s/step |
|---|---|
| option off | 6.26 |
| option on | **2.95** |

Both agree with (e): 6.37 / 3.02 and 2.92 there.

**Physics checks** (`analysis/snapdiff_box.py`: matched by ID, periodic box of any size, counts of elements that are not bitwise equal, max relative diff, totals). Final snapshots are at t = 0.4 for the turbulent box and t = 3e-5 for Sedov.

| A vs B | Velocities | Density | InternalEnergy | Coordinates | totals |
|---|---|---|---|---|---|
| turb GPU base vs base repeat (1x18, other node) | 0 differ | 0 | 0 | 0 | identical |
| turb GPU base vs option off (1x18) | **0** | **0** | **0** | **0** | identical |
| turb GPU base vs option on (1x18) | 25,348 of 6.3M, max 7.0e-16 of max\|v\| | 4,517 of 2.1M, max rel 6.2e-15 | 2,057, 5.3e-16 | 240, 1.8e-16 of box | M, Ekin, Eth, Etot, \|p\| agree to all 16 printed digits |
| turb CPU 18x1 vs GPU base (reference scale) | 4.2M, 4.2e-15 of max\|v\| | 1.4M, 1.6e-14 | 611k, 5.3e-16 | 239k | same 16 digits |
| turb GPU base 2x9 vs base 2x9 repeat | 2.3M, 4.1e-15 | 695k, 1.3e-14 | 298k | | same 16 digits |
| turb GPU base 2x9 vs option off 2x9 | 2.35M, 4.0e-15 | 724k, 2.0e-14 | 311k | | same |
| turb GPU base 2x9 vs option on 2x9 | 2.23M, 4.3e-15 | 653k, 1.4e-14 | 280k | | same |
| Sedov GPU base (Oct 29) vs option off (today) | 0 | 0 | 0 | 0 | identical |
| Sedov GPU base vs option on | 846, 3.2e-16 | 1, 6e-17 | 180, 2.9e-16 | 0 | Etot identical; \|p\| 8.63e-18 vs 8.52e-18 |
| Sedov option on (Oct 30 run) vs option on (today) | 0 | 0 | 0 | 0 | identical |

- At 1 rank, the GPU runs are deterministic. Option off is bitwise identical to the baseline. Option on differs only by the host drift (no FMA contraction on the host), at round-off level, and the growth over 32 steps of a driven flow is to 1e-15 relative.
- At 2 ranks, the baseline is **not** bitwise reproducible between two runs (domain/communication order). Option on and option off differ from the baseline by the same amount as a baseline repeat. So the 2-rank checks can only say "within the baseline's run-to-run scatter".

**dd6dc7b8 backport, turbulent box 128^3, 8 steps (dt = 0.0125), 1 x 18, one node** (ravg1005, job 31692760):

| | s/step (steps 2-8) | vs base |
|---|---|---|
| base | 8.96 | |
| option off | 9.14 | **bitwise identical** to base (all fields) |
| option on | **5.19** | 27 velocity components, 7 densities, 2 energies differ; max 9e-15 relative, 5e-17 of max\|v\|; totals identical to 16 digits |

**Not tested:**
- real PCIe H100s, which are Mike's hardware;
- more than 2 ranks per GPU, MPS, multi-node;
- gravity, MFV and the other hydro methods, MHD (Mike also ran `MAGNETIC`), all other physics modules;
- deep timestep hierarchies (both problems were all-active every step);
- the dd6dc7b8 backport beyond the 8-step turbulence run (no Sedov, no 2 ranks, no CPU build);
- the PCIe-default placement 1 (host-preferred) that Mike found best on his node. All runs used the default placement, which is device-preferred on both trees for a non-coherent CUDA GPU;
- GH200 and AMD GPUs with the option on.

## Update 2026-10-30 (evening): Mike Grudić's multi-rank additions (PR #80), tested on the A100

**What was tested.** Mike's second comment on PR #80 reports PCIe H100 numbers (Rusty, 8 ranks x 12 threads, 2 ranks per GPU under MPS: option on 2.41 s/step, #80 + items 1-8 0.82) and adds eight commits. Branch `gpu-pcie-optimizations-multirank` (tip 7d9f0cb5) = exactly 8 commits on our 7472ec01, fetched into `ext/gizmo-pr` (local branch `mike-multirank`). Diff: 8 files, +80/-24.

| item | commit | change (all prefetches gated by the option via `particles_prefetch_enabled()` and, where an active count exists, by `num_active*8 >= NumPart`) |
|---|---|---|
| 1 | c3d4b45a | `neighbor_loop_runner.cc`: P/CellP to host after a Mode D density call that iterated on the device to its end, and at the Mode D -> Mode B demotion |
| 2 | 60179d02 | `ghost_exchange.cc`: to host before `gx_pack_and_forward_particle_exchange` (receiver walk ran on device) |
| 3 | 5db2c151 | `gpu_particles_arena.cc`: CellP prefetch range `[0, min(NumPart, MaxPartGas))` while ghosts are imported, else `[0, N_gas)` |
| 4 | 96c262d2 | `gpu_neighbor_list.cc`: to host after the CSR kernels in `gpu_ngb_list_build` |
| 5 | 37084286 | `gpu_force_update.cc`: P-only to device before the reused-tree kick, P-only to host before the `dp` zeroing; new `gizmo_particles_prefetch_P_to_host` |
| 6 | 49533dc9 | `gpu_neighbor_list.cc`: to device before the owned-index rebuild (`sidx_build`) |
| 7 | 81b40dae | `gpu_particles_uvm_alloc`: `cudaMemPrefetchAsync(..., cudaCpuDeviceId)` before the memset of new storage; `resize_particle_storage`: to host before the capacity-change copies |
| 8 | 7d9f0cb5 | `turb_driving.cc` `add_turb_accel`: where SharedSpace is device-accessible, run the kernel in place on P/CellP over an int index list instead of the serial whole-record gather/scatter (bulk prefetch to device/host around it when the option is on). **Not behind the option.** |

**Builds** (`scripts/setup_pr80_builds.sh`, `scripts/job_p80_build.sh`, jobs 31698576/7; `git archive` of the commit's `src/` + the Raven SYSTYPE blocks `scripts/raven_makefile_blocks.diff` + Config; same modules and Kokkos 4.6.01 CUDA as before; make -j18 in Slurm):
`build-p80-{tb,sd}-{a,b,c,d}`: a = 7472ec01 option off, b = 7472ec01 on, c = 7d9f0cb5 on, d = 7d9f0cb5 off. `tb` = turbulent-box Config (`scripts/Config_turbbox.sh`), `sd` = the Sedov MFM Config of (e) (`scripts/Config_sedov_mfm.sh`).

**Runs** (`scripts/job_p80_run.sh`, jobs 31698578-82, logs `runs/p80-<jobid>.out`, run dirs `runs/p80_<tag>_<variant>`). Same ICs/params as before (turbulent box 32 steps to t = 0.4, all 2,097,152 active; Sedov 17 steps to 3e-5). One A100 per job; all arms of one layout back to back in one job. 2 ranks x 9 threads ran **under MPS** (user-started `nvidia-cuda-mps-control -d` in the job, compute mode Default; `get_server_list` and `nvidia-smi` confirm both ranks on one MPS server). Earlier 2x9 numbers in this note were without MPS. s/step = mean of steps 2..last of `cpu.txt` (`scripts/extract.py`); wall = `/usr/bin/time` of srun.

| case, layout (node) | (a) off | (b) on | (c) Mike on | (c) vs (b) |
|---|---|---|---|---|
| turb box, 1x18 (ravg1056) | 7.95 s, 4:36 | 4.42 s, 2:37 | **3.59 s, 2:09** | -19 % |
| turb box, 2x9 MPS (ravg1156) | 5.65 s, 3:16 | 3.53 s, 2:04 | **3.09 s, 1:50** | -13 % |
| Sedov, 1x18 (ravg1158) | 6.36 s, 2:08 | 2.95 s, 1:04 | 2.99 s, 1:06 | +1.7 % (noise) |
| Sedov, 2x9 MPS (ravg1158) | 4.67 s, 1:35 | 2.57 s, 0:54 | 2.50 s, 0:53 | -2.6 % |
| turb box, 1x18, option **off** (ravg1153) | 7.71 s, 4:40 | | (d) Mike off: **7.29 s, 4:14** | (d) vs (a): -5.6 % |

- Sedov 1x18 and 2x9 ran at the same time on ravg1158 (different GPUs, shared host). The Sedov (d) arm, whose code path equals (a) (no TURB_DRIVING), ran 6.15 vs 6.36: about 3 % same-job noise.
- Rows that changed (s/step):
  - turb 1x18, b -> c: `hydro_frc` 1.54 -> **0.59**; misc 0.12 -> 0.14; everything else within 0.02.
  - turb 2x9, b -> c: `hydro_frc` 1.03 -> **0.55**; log/stats 0.04 -> 0.005; misc_hydro 0.07 -> 0.05; gradients 0.30 -> 0.33; density 0.46 -> 0.44; treebuild 0.21 -> 0.22.
  - turb 1x18 off, a -> d: `hydro_frc` 1.89 -> **0.95**, misc 0.61 -> **1.02** (presumably: without the bulk prefetch the in-place kernel faults the actives' pages to the device and a later host phase faults them back; not profiled), treebuild +0.03. Net -0.43.
  - Sedov 2x9, b -> c: domain 0.16 -> 0.13, density 0.38 -> 0.36, gi_corridor 0.43 -> 0.41, misc_hydro 0.07 -> 0.05, gradients 0.28 -> 0.30.
- So on one A100, almost all of the gain is item 8 (turbulence driving). Items 1-7 are neutral within noise at 1x18 (expected: those paths are multi-rank) and at most a few % at 2x9 under MPS. No profile was taken to check how often items 1-6 fire at 2 ranks here; with 2 ranks on one GPU the problem per rank is 4x larger than Mike's 8-rank runs.
- MPS vs no MPS, 2x9 turb box: option off 5.65 (MPS) vs 6.60 (earlier, no MPS, other node); option on 3.53 vs 4.34.

**Physics** (`analysis/snapdiff_box.py`, run with `module load python-waterboa/2025.06` for h5py; final snapshots):

| A vs B | result |
|---|---|
| turb 1x18: (c) vs (b) | **bitwise identical, all fields** |
| turb 1x18: (d) vs (a), same node; (d) vs (a) other node; (a) vs (a) other node; (a) vs Oct-30 `tb128_gpu_off` | **bitwise identical, all fields** |
| turb 1x18: (a) vs (b) and (a) vs (c) | identical diff in both: 25,348 velocity components (max 7.0e-16 of max\|v\|), 4,517 densities (6.2e-15 rel), 2,057 u, 240 coordinates; totals to 16 digits. Same as the Oct-30 base-vs-on numbers (host drift without FMA). |
| Sedov 1x18: (c) vs (b), (d) vs (a) | **bitwise identical, all fields** |
| Sedov 1x18: (a) vs (b) = (a) vs (c) | 846 velocities (3.2e-16), 1 density, 180 u; Etot identical (as in (c) of this note) |
| turb 2x9: (a)-(b), (a)-(c), (b)-(c) | 2.20-2.27M velocity components differ, max 3.2e-15 of max\|v\|; 0.65-0.67M densities, max 1.65e-14 rel; Etot to 16 digits (last digit 8 vs 9) |
| turb 2x9: Oct-30 off (`tb128_gpu_off_2x9_b`, no MPS) vs today (a) | 2.32M, 3.2e-15; 0.70M, 1.5e-14: the same scatter |
| Sedov 2x9: (a)-(b), (a)-(c), (b)-(c) | 4.0-4.1M velocity components, max 1e-15 of max\|v\|; 0.18-0.20M densities, 2e-15; Etot identical |
| Sedov 2x9: Oct-29 baseline 2x9 vs today (a) | 4.03M, 9e-16; 0.17M, 2e-15: the same scatter |

- Item 8 is bitwise neutral with the option on and off at 1 rank, on both Configs (Sedov does not call it: control).
- Our A100 does not reproduce Mike's "bitwise identical to option off" for velocities/densities at 1 rank: option on (with or without his commits) differs from option off by the host drift, as on Oct 30. Mike's 8-rank comparison is within the multi-rank run-to-run scatter, where this cannot be seen.

**Draft reply**: `/raven/u/uli/gizmo-bench/pr80_reply_draft.md` (not posted). Nothing pushed; `/raven/u/uli/gizmo` untouched.

**Not tested**: >2 ranks per GPU, more than one GPU, H100/GH200, item 8 on a host-coherent GPU (GH200, where it runs without the prefetches), nsys of items 1-6 at 2 ranks.
