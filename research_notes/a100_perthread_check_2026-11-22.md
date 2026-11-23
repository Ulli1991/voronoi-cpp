# Brief for the A100 (Raven) agent: is the MI300A really slower at the voronoi-cpp full rebuild, or only on a different kernel?

Written 2026-11-22 on Viper. This brief is self-contained: you need nothing from the Viper session.

## 1. The question

Uli's voronoi-cpp full rebuild on one MI300A (Viper, HIP) runs at 4.5-21.6 M cells/s. The README's A100 numbers (CUDA) are
2-5x higher. The MI300A has about 6x the A100's FP64 rate and 2.5x its memory bandwidth, so this is suspicious ("sus", Uli).

The two runs do not use the same kernel:
- **A100 / CUDA:** `shell_voronoi::construct_warp` runs the warp kernel (one 32-lane warp per cell,
  `src/voronoi_construct/shell_voronoi_warp.hpp`, guarded by `#if defined(KOKKOS_ENABLE_CUDA)` and
  `if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>)`). The per-thread kernel rebuilds only the cells the warp kernel gives up.
- **MI300A / HIP:** the same function falls through to `construct(exec, grid, out, {}, false, rounded);`: the per-thread kernel for every
  cell (README: "The fast warp kernel is CUDA only; other backends (OpenMP, Serial, HIP) run the per-thread kernel, the same cells").

**Hypothesis to test (not yet shown):** the gap is the kernel, not the hardware. On the same per-thread kernel, the MI300A should match or
beat the A100.

What supports it so far:
- The moving step, where neither machine uses the warp kernel (except the CUDA cell sums), matches: about 225 / 134 / 71 M/s on MI300A
  vs 223 / 130 / 65.9 on A100.
- The cell-list build is 1.3-1.9x faster on the MI300A.

Why it is still only a hypothesis:
- The A100 numbers are from the repo's docs and were not rerun.
- The only per-thread A100 numbers (9.0 / 2.6 / 2.2 / 2.1 M cells/s, `research_notes/adaptive_cell_list_2026-10-30.md`) are from older
  code that still had rings. That is not like for like.

## 2. What to run on one A100 (Raven)

### Version
Use voronoi-cpp main at **1741a1a** (the base of the MI300A runs). The MI300A build adds three local commits on top: hipCUB radix sorts
and two benchmark programs. None of them touches the CUDA path or `voronoi_bench`.

### Build A: stock CUDA build, warp kernel
Build as in the README:

```sh
module load gcc/14 cuda/13.0 cmake
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper
cmake --build build-cuda -j4
```

### Build B: same, per-thread kernel only
This is a measurement build only. Keep it local, in a separate worktree or build directory. Never commit or push it.

In `construct_warp` (shell_voronoi_warp.hpp, the `#if defined(KOKKOS_ENABLE_CUDA)` just inside the function), change the guard to:

```cpp
#if defined(KOKKOS_ENABLE_CUDA) && !defined(VORONOI_PER_THREAD_ONLY)
```

Then build into `build-cuda-pt` with `-DCMAKE_CXX_FLAGS=-DVORONOI_PER_THREAD_ONLY`. CUDA then takes the same fall-through as HIP.

Change only that one guard; the other `KOKKOS_ENABLE_CUDA` sites, such as the cell sums, stay as they are. Check with
`grep -n VORONOI_PER_THREAD_ONLY` that the guard is the only change.

### Runs
Run on one A100, both builds, with exactly the MI300A commands:

```sh
for kind in jittered uniform clustered halo; do ./voronoi_bench rebuild 2097152  $kind 3;            done   # 128^3
for kind in jittered uniform clustered halo; do ./voronoi_bench rebuild 16777216 $kind 3 --no-check; done   # 256^3
```

Record from each run:
- the output lines `warp: build_ms=... construct_ms=... construct_Mcells_s=... pipeline_Mcells_s=...`. The label says "warp" on every
  backend because it names the function, not the kernel.
- the GPU clock: the README uses 1305 MHz. Say whether it was locked.

Correctness: build B must certify every cell. Run once without `--no-check` at 128³, and run `ctest` on build B if time allows.

## 3. MI300A numbers to compare against

Measured on Viper, voronoi-cpp 1741a1a plus hipCUB sorts, Kokkos 5.0.0 HIP, ROCm 7.1, GPU 0 of APU 0, double precision.
Values are `construct_Mcells_s`.

| input | MI300A 128³ (per-thread) | A100 128³ (README, warp) | MI300A 256³ (per-thread) | A100 256³ (README, warp) |
|---|---|---|---|---|
| jittered | 20.7 | 42.6 | 21.6 | 42.8 |
| white noise (uniform) | 4.57 | 22.2 | 4.51 | 22.5 |
| clustered | 3.58 | 17.7 | 4.12 | 18.5 |
| halo | 3.00 | 16.9 | 4.18 | 18.3 |

Logs on Viper: `/ptmp/uli/gku_viper/voronoi_mi300a/logs/benchA-13409857.out` (128³) and `benchC-13419427.out` (256³).

## 4. How to read the result

| A100 build B (per-thread) vs MI300A | Conclusion | Next step |
|---|---|---|
| A100 at or below the MI300A | The gap is the kernel. Hypothesis confirmed. | Port the warp kernel to HIP with 64-lane wavefronts (about 60 intrinsic sites: 32-bit masks, `__ballot_sync`, `__popc`, `__syncwarp`, `cuda_device_prop`). |
| A100 clearly above the MI300A | The hardware or the HIP build loses even on the same kernel. Hypothesis wrong. | Profile the MI300A per-thread kernel: occupancy, registers, scratch spills, FP64 rate. Spills are likely, given the large per-thread polyhedron. |

Also report:
- the build A / build B ratio on the A100 (what the warp kernel buys there);
- any change in build A against the README table. If build A is far from the README, flag it: the clock or code may differ.

## 5. Rules

- These are Uli's repos and Uli's runs. Do not push, open issues or PRs, or post anything; Uli decides.
- No env-var switches or debug code in commits. The `VORONOI_PER_THREAD_ONLY` guard stays local.
- Each claim needs the run output and the exact command. Label anything else as a hypothesis.

## 6. Report back

Write the following to a markdown file for Uli to carry back to Viper:
- the two-build table (128³ and 256³, four inputs);
- the clock and GPU used;
- the ctest / certification result for build B;
- your reading per section 4;
- the job IDs and log paths.
