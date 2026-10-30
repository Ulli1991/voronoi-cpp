# Kokkos / CUDA features we do not use yet: speed audit for the A100 kernels

Date: 2026-10-29.

**Scope.** This note is read-only research. Nothing was built or run.

**Inputs.**
- Our code: `src/voronoi_construct/{shell_voronoi_warp,shell_voronoi_lean,shell_voronoi,tet_mesh,tet_repair,exact_predicates}.hpp`, `src/neighbour_search/neighbor_search.hpp`, including the uncommitted `SHELL_WARP_TEAM` diff.
- Kokkos source `/u/uli/kokkos-src` (5.0.99, commit e2d872a, Feb 2026), install `/u/uli/kokkos-cuda` (AMPERE80, no RDC, no MallocAsync).
- CUDA 13.0.1 headers (`/mpcdf/soft/SLE_15/packages/x86_64/cuda/13.0.1/include`).
- NVIDIA and Kokkos documentation (links at the end of each item).

**Related note.** `gizmo_kokkos_paradigms.md` already covers several of these items: pinned readbacks and device-side sizing for repair (R1), level-1 scratch for the lean kernel and register staging (R2), AoS/Morton for the geometry kernels (R3), and allocation churn (R4). Those are cited here, not repeated.

**Tags.** (V) = verified in source, headers or our measurements. (I) = inferred, to be measured.

**Measured bottlenecks.**
- Clipping:
  - divergence-bound: 5-16 of 32 lanes active;
  - occupancy 8-17%, issue 42%;
  - DRAM <1%, L2 hit 99%;
  - the topology update takes 52-63% of the time.
- Repair: 6-8 ms per step, 5-15 rounds × 5-6 launches, and 2 pinned readbacks + fences per round.
- Geometry: faces 4.1 ms, circumcentres 1.1 ms (2.1M cells).

---

## 0. A bug to fix before any sub-warp experiment (V)

The uncommitted `SHELL_WARP_TEAM` change keeps this launch:
```
TeamPolicy<Cuda, LaunchBounds<warps_per_block * lanes, MIN>> policy(exec, blocks, warps_per_block, lanes);
```
The Kokkos CUDA backend launches `block(vector_length, team_size)`: `Kokkos_Cuda_Parallel_Team.hpp`, `execute()`, `const dim3 block(int(m_vector_size), int(m_team_size), 1)`.

So with `lanes = 16` the block is (16, 4), which is 64 threads in 2 warps. The kernel, however, assumes `blockDim.x == 32`:
- `team = team_rank * cells_per_warp + threadIdx.x / lanes`, and `threadIdx.x / 16` is always 0. Only the even teams of each block are built. The odd cells never run, and the `only_failed` fallback may silently rebuild them, which looks like a slowdown, not an error.
- `team_mask()` uses `threadIdx.x - lane()`, which is always 0. The threads with `threadIdx.y = 1` are hardware lanes 16-31, but they pass mask `0x0000ffff` to `__shfl_sync` / `__ballot_sync`. A mask that does not include the calling lane is undefined behaviour (it can hang or give wrong votes).
- With `lanes = 8` the scratch request is 16 cells × 5408 B = 86.5 KB. That is above the 48 KB level-0 cap, so Kokkos throws at launch ("Requested too much scratch memory on level 0").

**Fix.**
- Always launch `policy(exec, blocks, warps_per_block, 32)` with `LaunchBounds<warps_per_block * 32, MIN>`, and keep `lanes` as the logical team width inside the warp.
- For `lanes = 8`, use `warps_per_block = 2`: 8 cells × 5408 = 43 KB fits.
- Better still, use `cooperative_groups::tiled_partition<lanes>` (item 8), so the masks come from the hardware lane id (`threadIdx.x + 32 * threadIdx.y`, i.e. `%laneid`), not from `threadIdx.x`.

**Polyhedron sizes in double (computed from the struct).**

| team width | per cell | cells per block (4 warps) | block |
|---|---|---|---|
| 32 | 5600 B | 4 | 22.4 KB |
| 16 | 5504 B | 8 | 44.0 KB, fits just under 48 KB |
| 8 | 5408 B | 16 | 86.5 KB, does not fit |

---

## Ranked list (highest expected speed value first)

| # | Feature | Target kernel | Expected gain | Kokkos support |
|---|---|---|---|---|
| 1 | Decouple warps from blocks: 1 warp per block, or persistent warps with an atomic cell queue | warp clipping | 1.1-1.4x (I) | native (TeamPolicy, LaunchBounds); CUDA ignores `Schedule<Dynamic>` (V), so the queue is manual |
| 2 | FP32 polyhedron (`Real=float`) in the warp kernel, with double input and the exact repair downstream | warp clipping | 1.2-1.6x (I) | native (template already exists) |
| 3 | CUDA Graph with a conditional WHILE node for the repair rounds (zero host round trips) | tet_repair | 1-2 ms of 6-8 ms (I) | raw CUDA graph + stream capture of Kokkos launches; `Kokkos::Experimental::Graph` can wrap a native `cudaGraph_t` (V) |
| 4 | Register budget sweep + spill check (`LaunchBounds` MIN 4..8, `-Xptxas -v,--warn-on-spills`, cold paths out of line) | warp, lean | 0-10% (I) | `Kokkos::LaunchBounds<T,B>` only; no per-kernel `__maxnreg__` through Kokkos |
| 5 | Nsight Compute metric set (stall reasons, divergence, bank conflicts, achieved vs theoretical occupancy) | all | enables 1-2 | tool |
| 6 | Atomic-free volume accumulation in `faces` (segmented reduce over the owner-sorted faces) + `RandomAccess`/`const` gathers | tet_mesh faces | 10-30% of 4.1 ms (I) | native (`MemoryTraits<RandomAccess>`); warp intrinsics via `__CUDA_ARCH__` |
| 7 | >48 KB dynamic shared opt-in | lean (full warps), warp team 8 | small (I) | **not exposed** in Kokkos 5.0.99 (V): needs a raw `__global__` launch |
| 8 | `cooperative_groups::tiled_partition<N>` for sub-warp teams | warp (team 16/8) | correctness and clarity; ~0 speed | raw CUDA inside `#ifdef __CUDA_ARCH__` |
| 9 | Warp-aggregated appends (`__reduce_add_sync`, `cg::reduce`, `__match_any_sync`) | repair find/plan/apply | <5% of repair (I) | raw CUDA; desul already maps to `atomicAdd` (V) |
| 10 | Filter statistics for exact predicates; FP64-FMA expansions instead of int128/256 if the exact path is hot | repair on lattices | 0 on random input, large on sc lattice (I) | n/a |
| 11 | L2 persistence window for positions / centres | geometry kernels | 0-10% (I) | raw CUDA on `exec.cuda_stream()` |
| 12 | Carveout / `DesiredOccupancy` | lean, fallback | ~0 (I) | `Kokkos::Experimental::prefer(policy, DesiredOccupancy{N})` (V) |
| 13 | cudaMallocAsync / memory pools | per-step allocations | ~0 if we preallocate | `Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC` (OFF in our build) (V) |
| 14 | Persistent cooperative kernel with `grid.sync()` / dynamic parallelism | repair | ≤ item 3, more work | raw CUDA; dynamic parallelism needs RDC (not in our install) (V) |
| 15 | `Kokkos::Experimental::simd`, `ScatterView`, `Kokkos::sort`/`BinSort`, `StaticBatchSize` | - | none on CUDA for us | see below |
| 16 | Compiler flags (`-use_fast_math`, `-dlcm`, `-lineinfo`) | - | 0 speed; `-lineinfo` for profiling | nvcc |

---

## 1. Decouple warps from blocks in the warp kernel

**What.** A CUDA block keeps its resources (shared memory, block slot, register file share) until its *last* warp exits.

Today 4 independent cells share a block, and per-cell cost varies a lot: jittered 24 tests / 16 cuts, uniform 45 / 18, clustered 169 / 37. A block therefore holds its 22 KB and 4 warp slots while 1-3 of its warps are already finished.

Theoretical occupancy:
- `LaunchBounds<128,5>`: 5 blocks × 4 warps = 20 of 64 warps = 31%.
- Measured achieved: 16.7%.

The gap between theoretical and achieved is the signature of tail and imbalance (V for the numbers; the cause is I).

**Options.**

a) `TeamPolicy(exec, n_cells, 1, 32)` with `LaunchBounds<32, 20>`. One cell per block (5.6 KB scratch; Kokkos adds `(team_size + 2) * 8` B), and up to 32 blocks/SM on A100.
- The register cap is the same as today: 65536 / (20 × 32) = 102.
- Warps now retire independently.
- One Kokkos-only line.

b) **Persistent warps.**
- Launch league = SMs × resident warps, i.e. 108 × 20.
- Each warp loops `t = atomic_fetch_add(&next, 1)` (lane 0, then `__shfl_sync`) until `t >= n`.
- This removes the grid tail and balances clustered input.
- The Kokkos CUDA TeamPolicy runs a static `league_rank += gridDim.x` loop and has no dynamic schedule. `Schedule<Dynamic>` is ignored: no reference in `Kokkos_Cuda_Parallel_Team.hpp` (V). So the queue must be our own atomic.
- Optional: hand out cells in decreasing expected cost (e.g. by cell-list occupancy) to shrink the tail further.

**Why it should pay.** We are latency/divergence-bound with issue at 42%. More resident warps hide shared-memory and dependency latency even at the same lane efficiency. Achieved occupancy can rise toward the theoretical 31%.

**Kokkos.** Native (TeamPolicy, LaunchBounds, `Kokkos::atomic_fetch_add`).

**Experiment.**
1. Build `warps_per_block` ∈ {1, 2, 4}, with MIN chosen so that warps/SM stays 20.
2. Add a persistent-warp variant with an atomic queue.
3. On gpu1, jittered / uniform / clustered, 2.1M cells: measure M cells/s.
4. In ncu, compare `sm__warps_active.avg.pct_of_peak_sustained_active` (achieved) against `sm__maximum_warps_per_active_cycle_pct` (theoretical).
5. Success means achieved within 80% of theoretical, and clustered gaining the most.

**Docs.**
- Occupancy and block retirement: CUDA Best Practices Guide, §Occupancy, https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Kokkos TeamPolicy: https://kokkos.org/kokkos-core-wiki/API/core/policies/TeamPolicy.html

## 2. FP32 polyhedron in the warp kernel

**What.** Instantiate `warp::Kernel<float>` and keep positions and the certificate distance in double, or make the cell-local coordinates relative to the generator.

**Hardware facts.**
- A100 FP64 (non-tensor) is 9.7 TFLOP/s and FP32 is 19.5 TFLOP/s: only 2:1 (Ampere tuning guide / A100 whitepaper). So the arithmetic gain is at most 2x.
- The larger effects are registers and shared bytes:
  - every double costs 2 registers;
  - `new_x/new_y/new_z[3*per_lane]` plus `keep_*` alone is about 24 doubles = 48 registers per lane at team 32.
  - The polyhedron drops from 5.6 KB to about 3.3 KB per cell.
- Fewer registers is exactly what holds occupancy down (130 registers before the cap, and +4% from the cap alone).

**Why it is safe.** The status note says clipping alone is not watertight even in double. Exact tet repair from the rings already makes the topology exact, so float clipping only changes how many cells the repair must fix.

**Risk.** More `failed` cells routed to the per-thread double fallback, and more flips. Measure both.

**Kokkos.** Native: `construct_warp<float>`. The status note has float numbers only for the lean kernel ("40-slot lean double worse"), not for the warp kernel.

**Experiment.**
- `shell_voronoi_check` with `Real=float` vs `double`, warp kernel, the three layouts.
- Record: M cells/s, registers (`-Xptxas -v` or the ncu `launch__registers_per_thread`), achieved occupancy, failed-cell fraction, and repair flips/rounds afterwards.
- Accept if the end-to-end time (clip + fallback + repair) drops.

**Docs.** Ampere tuning guide https://docs.nvidia.com/cuda/ampere-tuning-guide/ ; A100 whitepaper https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf

## 3. Repair rounds as a CUDA Graph with a conditional WHILE node

**What.** CUDA 12.4+ conditional nodes (IF/WHILE; SWITCH in 12.8) run a body graph repeatedly while a device-set condition is non-zero.

Verified in our CUDA 13.0.1 headers (V):
- `cudaGraphConditionalHandleCreate` (`cuda_runtime_api.h:13013`);
- `cudaGraphCondTypeWhile` / `cudaGraphCondTypeSwitch` (`driver_types.h:3352`);
- the device function `cudaGraphSetConditional` (`cuda_device_runtime_api.h:480`).

The body is filled with `cudaStreamBeginCaptureToGraph` (CUDA 12.3+), so we can capture our Kokkos `parallel_for`s launched on a `Kokkos::Cuda` instance built on the capture stream.

**What Kokkos 5.0.99 offers (V).**
- `Kokkos::Experimental::Graph` with `then`, `then_parallel_for`, `then_host`, `when_all`, `instantiate`, `submit`.
- `cuda_capture(exec, functor)`: stream capture into a child node, `Kokkos_GraphNode.hpp:331`.
- Construction *from* a native `cudaGraph_t`: `Kokkos_Graph.hpp:81`, PR #7664.
- **No conditional nodes** (no hits in `Kokkos_Cuda_Graph*.hpp`). So the WHILE node is raw CUDA, and Kokkos supplies the kernels.

**Prerequisite.** Device-side sizing, i.e. GIZMO note R1(b): a graph freezes grid sizes.
- Launch `find`, `plan`, `apply` and `release` over a fixed grid, e.g. `RangePolicy(0, 108 * 2048)` with an inner stride loop to `counters(k)`, or over `work_capacity` with an early exit.
- Move the `free_count` / `count` / statistics bookkeeping into a one-thread kernel at the end of the body.
- That tail kernel also calls `cudaGraphSetConditional(handle, items > 0 || retry_progress)`.
- The `retry_stuck` branch becomes either an IF node or a few lines inside the tail kernel.

**Expected gain.** Today each round pays 2 × (D2H copy + `exec.fence()`), and after each fence the GPU idles until the next host launch arrives. Each of the 5-6 later-round kernels has only hundreds to thousands of items, so it is launch-latency-sized (3-5 µs).
- Estimate: about 50-80 µs of overhead per round × 5-15 rounds = 0.4-1.2 ms, plus launch gaps (I).
- An in-graph kernel-to-kernel gap is about 1-2 µs, and there are no host syncs.
- Expect 1-2 ms of the 6-8 ms. More if nsys shows the GPU idle for a large part of the repair.

**Open check (I).** Whether kernels that call `cudaGraphSetConditional` need `-rdc=true` / `cudadevrt`. It is declared `__cudart_builtin__`, which suggests not. Our Kokkos install has no `KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE`, so test it in a 30-line standalone .cu first.

Fallback if RDC is needed: replay a non-conditional `Kokkos::Experimental::Graph` for k rounds (e.g. 4) per host check. The extra rounds run on empty lists and cost about 6 × 2 µs each.

**Capture restrictions.**
- No `fence()`, no `cudaMalloc`, and no pageable copies inside the captured body.
- Kokkos's `cuda_capture` uses `cudaStreamCaptureModeGlobal` (V).
- Level-1 scratch would resize a pool, so it is not allowed in captured kernels. Repair uses none.
- Kokkos-side `deep_copy(exec, subview, 0)` becomes a memset node, which is allowed.
- Body node types allowed: kernel, memset, memcpy (device-reachable memory), child graph, conditional. Cooperative kernels are not allowed in the body.

**Experiment.**
1. nsys the current repair (with the NVTX connector, §Tools). Sum kernel time vs repair wall time: the gap is the upper bound on the gain.
2. Implement R1(b) device sizing alone and measure.
3. Wrap the round in the WHILE node; build the graph once per mesh capacity and launch it per step (`cudaGraphLaunch` on `exec.cuda_stream()`).
4. Compare ms/step and rounds.

**Docs.**
- https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes
- CUDA Programming Guide, CUDA Graphs → Conditional graph nodes, and "Prohibited and unhandled operations" under stream capture: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Kokkos Graph: https://kokkos.org/kokkos-core-wiki/API/core/Graph.html ; Kokkos CHANGELOG 4.5 (#7240), 4.6 (#7664), 4.7 (#7552, `cuda_capture`; #8075, `then_host`).

## 4. Register budget: sweep, spills, cold paths

**Kokkos route.** The only per-kernel register control through Kokkos is `Kokkos::LaunchBounds<MaxThreads, MinBlocks>`, which becomes `__launch_bounds__` on the generated `__global__` (`Kokkos_Cuda_KernelLaunch.hpp:73-118`) (V).
- CUDA 12.4 added a per-kernel `__maxnreg__`, but we cannot attach it to the Kokkos kernel.
- `-maxrregcount` is global and would also hit the repair and geometry kernels, so do not use it.

**Do.**
- Sweep `SHELL_WARP_MIN_BLOCKS` 4..8, and the per-block MIN for item 1.
- Compile once with `-Xptxas -v,--warn-on-spills` (or `VORONOI_CUDA_RESOURCE_REPORT`) and read the spill bytes.
- In ncu, local traffic shows as `l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum` / `..._st.sum`. Spill loads hit L1, which is cheap while L1 is not thrashed.
- Move rare paths out of line (plane/vertex overflow handling, the certificate failure path) with `__noinline__` device functions. This keeps their live ranges out of the hot loop's allocation (I).

**Gain.** 0-10% (I). It matters mostly in combination with items 1 and 2.

**Docs.** nvcc `--resource-usage`, `-Xptxas`: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/ ; launch bounds: CUDA Programming Guide §Launch Bounds.

## 5. Nsight Compute: what to collect next (no speed by itself; it decides 1, 2, 4, 6)

Build with `-lineinfo`; it has no runtime cost.

Run `ncu --set full --import-source yes -k regex:construct_warp` and add these sections: `WarpStateStats`, `SourceCounters`, `SchedulerStats`, `Occupancy`, `LaunchStats`, `MemoryWorkloadAnalysis_Tables`.

| question | metric |
|---|---|
| lanes active per instruction | `smsp__thread_inst_executed_per_inst_executed.ratio` (and `.pct`) |
| divergent branches | `smsp__sass_branch_targets_threads_divergent.sum`, and per line in SourceCounters |
| why warps wait | `smsp__average_warp_latency_issue_stalled_*` / `smsp__pcsamp_warps_issue_stalled_{short_scoreboard,long_scoreboard,mio_throttle,barrier,membar,branch_resolving,wait,not_selected}` |
| shared bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`, `..._op_st.sum` vs `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` |
| achieved vs theoretical | `sm__warps_active.avg.pct_of_peak_sustained_active` vs `sm__maximum_warps_per_active_cycle_pct`; limiter: `launch__occupancy_limit_{registers,shared_mem,blocks,warps}` |
| spills | `l1tex__t_bytes_pipe_lsu_mem_local_op_{ld,st}.sum` |
| geometry kernels | `dram__throughput.avg.pct_of_peak_sustained_elapsed`, `lts__t_sector_hit_rate.pct`, `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum / l1tex__t_requests_...` (sectors per request) |
| atomics | `l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum`, `lts__t_requests_srcunit_l1_op_atom.sum` |

**Expected stall picture for the topology update (I).** `short_scoreboard` (shared-memory and dependent loads in the adjacency walk) and `branch_resolving` / `wait`.

- If `mio_throttle` is high, the shared-memory pipe is saturated, and a smaller (float) polyhedron helps.
- Bank conflicts:
  - the SoA double arrays indexed by lane are conflict-free (two wavefronts per 256 B is normal);
  - the gathers `x[adjacent]`, `corner`/`replacement` lookups, and the `uint8` arrays `starts/ends/first` are where conflicts can appear.

**Docs.** https://docs.nvidia.com/nsight-compute/ProfilingGuide/ (Metrics Reference; Warp Scheduler States).

## 6. Geometry kernels: atomic-free volumes, read-only gathers

**What.** `faces` ends with two `Kokkos::atomic_add(double)` per face (`out.volume(i)`, `out.volume(j)`) (V).
- Faces are grouped by owner. Consecutive lanes therefore often hit the same `volume(i)`, and same-address FP64 atomics serialise in L2.
- Desul maps `fetch_add(double)` to native `atomicAdd(double*)` (V, `desul/atomics/Fetch_Op_CUDA.hpp:29`).

**Options.**
- Owner side: a warp segmented reduction (`__match_any_sync` on `i`, then a shuffle-reduce and one atomic per segment). Or, since faces are owner-contiguous, a second pass `volume(i) = sum over faces [face_start(i), face_start(i+1))`, which needs no atomics.
- Neighbour side: keep the atomic, or run a second pass over a neighbour-sorted face index.
- Mark `x`, `centre` and `ring` as `View<const T, ..., MemoryTraits<RandomAccess>>`. That gives `__ldg` / `ld.global.nc` (`Kokkos_Cuda_View.hpp:23`) (V), but only matters if nvcc cannot already prove read-only (I).

**Gain.** 10-30% of 4.1 ms (I). First check whether `faces` is DRAM-bound:
- centres alone are about 14M tets × 32 B = 450 MB, more than the 40 MB L2;
- if `dram__throughput` > 70%, only fewer bytes or better locality help (float centres, Morton tet order = GIZMO R3).

**Docs.** https://kokkos.org/kokkos-core-wiki/API/core/view/view.html (Memory traits); CUDA Programming Guide §Warp Match Functions.

## 7. >48 KB dynamic shared memory opt-in

**CUDA.** `cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` allows up to 163 KB per block on A100; 164 KB per SM.

**Kokkos 5.0.99 (V).**
- `scratch_size_max(0)` = `sharedMemPerBlock` (48 KB) minus a reserve (`Kokkos_Cuda_Parallel_Team.hpp:188-205`).
- `ParallelFor` throws if `m_shmem_begin + m_shmem_size > sharedMemPerBlock` (:579-588).
- No call to `MaxDynamicSharedMemorySize` anywhere in `core/src/Cuda` (V).
- It cannot be enabled from user code without patching Kokkos, because the check runs before launch.

**Value for us.**
- Warp kernel: register-limited, and 5 × 22 KB = 110 KB/SM fits today. The opt-in only matters for team 8 with 4 warps per block (86.5 KB). Two warps per block avoids it. → No gain.
- Lean kernel: 1.7 KB per cell. Per-SM shared memory (164 KB / 1.7 KB = 96 cells = 3 warps) is the real cap, whatever the block size.
  - The opt-in only turns today's 6 half-warps (16-thread blocks) into 3 full warps: the same threads in half the warp slots.
  - That could help issue efficiency, but it does not help latency hiding. Level-1 scratch (GIZMO R2(i)) is the lever that adds threads.
- If we still want it: a raw `__global__` that calls the same device body with `extern __shared__`, `cudaFuncSetAttribute`, and `<<<grid, block, shmem, exec.cuda_stream()>>>`. The warp kernel already uses only `threadIdx`, `team_rank`, `league_rank` and `team_shmem`, so this is easy.

**Docs.** CUDA Programming Guide, Compute Capability 8.x → Shared Memory; Ampere tuning guide §Shared memory: https://docs.nvidia.com/cuda/ampere-tuning-guide/

## 8. Cooperative-groups tiles for sub-warp teams; Independent Thread Scheduling

**Tiles.** `auto team = cg::tiled_partition<lanes>(cg::this_thread_block());` gives:
- `team.thread_rank()`, `team.meta_group_rank()` (the cell index within the block);
- `team.shfl / shfl_xor / ballot / any / sync()`;
- `cg::reduce(team, v, cg::plus<int>())` (sm_80 `redux.sync` for int);
- `cg::inclusive_scan` / `exclusive_scan` (CUDA 11+).

Static tiles compile to the same SHFL/VOTE instructions as our hand-written ones, with masks derived from the real lane id. That removes the item 0 class of bug. `labeled_partition` / `binary_partition` are dynamic (`match_any`-based) and cost more; we do not need them.

**ITS caveat.** Two 16-lane teams in one warp running different cells diverge almost everywhere. The warp then issues each team's path separately: at most 16 lanes per issued instruction, and the two teams share one issue slot.

So sub-warp teams pay only if a 16-lane team keeps its lanes busier than a 32-lane team keeps its 32. With 5-16 active of 32 today, that is plausible but unproven. The literature note found no 2x evidence.

After any divergent region, use `team.sync()` (= `__syncwarp(mask)`) before reading shared data written by other team lanes. We already do this with `__syncwarp(team_mask())`.

**Docs.** CUDA Programming Guide §Cooperative Groups (Thread Block Tile, Group Collectives) and §Independent Thread Scheduling (Compute Capability 7.x).

## 9. Warp-aggregated appends in repair

**What we use.** `Kokkos::atomic_fetch_add` (relaxed, device scope) lowers to plain `atomicAdd` (V, desul). nvcc can already warp-aggregate some same-address `atomicAdd` patterns automatically (I; check the SASS for `VOTE` + `POPC` + one `ATOMG`). If it does not:

```
const unsigned m = __activemask();
const int leader = __ffs(m) - 1;
const int rank = __popc(m & ((1u << lane) - 1));
int base;
if (lane == leader) base = atomicAdd(&counter, __popc(m));
base = __shfl_sync(m, base, leader) + rank;
```

On sm_80, `__reduce_add_sync` handles variable counts per lane.

**Gain.** Small (I). The `find` appends happen only for failing triangles (few after round 1), and `mark` happens per new tet.

The `atomic_max` voting on 64-bit `lock` (5 tets × 2 atomics per plan) is more traffic. Hardware `atomicMax(ull)` is native, so the cost is L2 throughput, not locks.

Measure `lts__t_requests_srcunit_l1_op_atom.sum` per repair kernel before changing anything.

**Docs.** https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/ ; CUDA Programming Guide §Warp Reduce Functions (`__reduce_*_sync`, sm_80+).

## 10. Exact predicates: integer 128/256-bit on Ampere

**Hardware.** Ampere has no native 64×64→128 multiply. One 64×64 product is about 4 `IMAD.WIDE.U32` plus carries (`IADD3.X`), so our `u128 × u128` limb products in `add_product` cost about 16-30 integer instructions each (I). INT32 IMAD issues at half the FP32 rate on GA100.

**Where it matters.** Only when the double filter fails:
- for random and jittered input that is rare;
- for the sc lattice (cospherical points) *every* in-sphere test falls through to the 256-bit path plus symbolic perturbation. This is the failing robustness case in the status note.

**Experiment.** Add a device counter for filter failures per kernel (find/plan). If more than ~1% of tests on a layout fall through, consider:
- a staged filter (a semi-static bound, then a double-double/FMA expansion, which runs on FP64 units at 9.7 TF);
- reordering so that lattice degeneracies are detected once per tet rather than per triangle.

Otherwise leave it as is.

**Docs.** Shewchuk, "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates"; CUDA Programming Guide §Arithmetic Instructions (throughput table, cc 8.0).

## 11. L2 persistence (`cudaAccessPolicyWindow`)

On A100 up to 30 MB of the 40 MB L2 can be set aside (`cudaLimitPersistingL2CacheSize`). The window is set with `cudaStreamSetAttribute(exec.cuda_stream(), cudaStreamAttributeAccessPolicyWindow, …)`: raw CUDA on the Kokkos stream (V, headers).

- Clipping already has 99% L2 hits, so there is no gain there.
- Geometry kernels: positions are 2.1M × 24-32 B = 50-67 MB, larger than the window. A hitRatio < 1 could still keep part resident while centres stream (I).
- Try it only if item 6's ncu shows DRAM-bound `faces` with poor L2 hit rates on `x`.

**Docs.** CUDA Programming Guide §L2 Cache Access Management.

## 12. Shared/L1 carveout and `DesiredOccupancy`

Kokkos sets `cudaFuncAttributePreferredSharedMemoryCarveout` **only** when the policy carries `DesiredOccupancy`, i.e. `Kokkos::Experimental::prefer(policy, Kokkos::Experimental::DesiredOccupancy{N})` (`Kokkos_Cuda_KernelLaunch.hpp:161-245`, used at :410/:514/:700) (V).

Otherwise the driver picks the carveout per launch. On cc 8.x it "configures the shared memory capacity to avoid shared memory occupancy bottlenecks". The options are 0/8/16/32/64/100/132/164 KB of the 192 KB.

- Warp kernel: about 110 KB/SM, so the driver picks 132 KB and leaves 60 KB of L1.
- `shell_voronoi.hpp` fallback: no shared memory, so its local-memory polyhedron gets the maximum L1.

**Gain.** About 0 (I).

`DesiredOccupancy` is useful to cap occupancy on purpose, e.g. for the lean kernel with level-1 scratch, to keep the per-thread polyhedra L1-resident.

**Docs.** https://kokkos.org/kokkos-core-wiki/API/core/policies/ExecutionPolicyConcept.html (prefer / occupancy hints); Ampere tuning guide §L1/shared.

## 13. cudaMallocAsync / pools

`Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC` is OFF in our install (V). It is used in `Kokkos_CudaSpace.cpp:192/353` when enabled.

It only matters if Views are allocated per step. Repair allocates `stuck` once, but `from_rings` allocates about 8 Views per rebuild (`tet_mesh.hpp:56-125`).

Prefer persistent buffers (GIZMO R4). Turn this on only if nsys shows `cudaMalloc`/`cudaFree` in the step.

## 14. Persistent cooperative kernel / dynamic parallelism

**Cooperative kernel.** A single cooperative kernel (`cudaLaunchCooperativeKernel`, `cg::this_grid().sync()`) could run all repair rounds with grid barriers (a few µs each).
- Kokkos does not launch cooperatively, so this is raw CUDA.
- All blocks must be co-resident, and the phases must be rewritten as grid-stride loops.
- It beats item 3 only if we also fuse phases (e.g. plan+vote, or keep `items` in registers).
- Do item 3 first.

**Dynamic parallelism and device graph launch.** These need relocatable device code (`-rdc`, `cudadevrt`). Our Kokkos install was built without `KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE` (V, `KokkosCore_config.h`), so they are not available without a Kokkos rebuild. Not recommended.

## 15. Kokkos features with no gain for us on CUDA (V unless noted)

- **`Kokkos::Experimental::simd`.** On device the ABI is `scalar` (`simd/src/Kokkos_SIMD.hpp:90-92`): width 1, no gain.
- **`ScatterView`.** On CUDA it defaults to `ScatterAtomic`, the same atomics we already use.
- **`Kokkos::sort` / `BinSort`.** `Kokkos::sort` dispatches to Thrust for contiguous CUDA views (`algorithms/src/sorting/impl/Kokkos_SortImpl.hpp:21-42`). Our counting-sort cell list is not a bottleneck.
- **`parallel_scan`.** Fine. 4.6 improved `inclusive_scan` for CUDA.
- **`Kokkos::Experimental::StaticBatchSize<N>`** (new in 5.0, "explicit loop unrolling", #8164). Each RangePolicy thread processes N items. Possibly useful for the very light `unlist` / `release` / `move` kernels; negligible overall (I).
- **`__grid_constant__` launch** (`KOKKOS_IMPL_CUDA_USE_GRID_CONSTANT`, `Kokkos_Cuda_Instance.hpp:27-30`). Already active with nvcc on Ampere. Functors are passed by parameter, with no constant-memory copy and event.
- **Schedule.** `Schedule<Dynamic>` is ignored by the CUDA Range/Team backends (no references), hence the manual queue in item 1.
- **Programmatic dependent launch** (`cudaLaunchAttributeProgrammaticStreamSerialization`, `griddepcontrol`). Hopper+ only, not A100.

## 16. Compiler flags

- **`-use_fast_math`.** Affects only float: FTZ, approximate div/sqrt/rcp, `__sinf`, etc. Double is untouched. This matches our measured "no gain"; revisit only together with item 2.
- **`-Xptxas -dlcm=ca`.** No-op on Ampere: global loads are already L1-cached.
- **`-lineinfo`.** Zero runtime cost; enable it in the profiling builds of `shell_voronoi_check` (today only `reconstruction_profile` has it).
- **`-Xptxas -v,--warn-on-spills`.** Use in one diagnostic build.
- **`--extra-device-vectorization`.** Negligible for this code.

## Tools (Kokkos side)

- **kokkos-tools `kp_kernel_timer`.** Per-label kernel times. Our kernels are already labelled (`tet_repair::find`, …).
- **`kp_space_time_stack`.** Per-region time and memory high-water mark.
- **`kp_nvtx_connector`.** Puts Kokkos labels and regions into nsys timelines. Wrap each repair round in `Kokkos::Profiling::pushRegion("repair::round")` to see the host gaps for item 3.
- Load a tool with `--kokkos-tools-libs=…/libkp_nvtx_connector.so`, or with `KOKKOS_TOOLS_LIBS`.
- Docs: https://github.com/kokkos/kokkos-tools/wiki

---

## Suggested order

1. **Fix item 0** (team launch config), then the planned `SHELL_WARP_TEAM` 16/8 comparison becomes valid.
2. **One ncu run per clipping variant with the §5 metric set.** In particular, check whether achieved occupancy is far below theoretical (it supports item 1) and whether registers are the limiter (it supports items 2 and 4).
3. **Item 1** (warps per block 1/2/4, then a persistent warp queue).
4. **Item 2** (`Real=float` warp kernel, end to end including repair).
5. **nsys of repair** with the NVTX connector → R1(b) device sizing → **item 3** WHILE graph. Run the standalone RDC test first.
6. **Item 6** for faces after a DRAM-throughput check.
7. **Item 10** counters for the lattice case.
