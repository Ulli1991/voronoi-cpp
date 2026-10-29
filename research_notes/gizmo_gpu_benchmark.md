# GIZMO `gpu_kokkos_unified` (Kokkos CUDA) Sedov benchmark on one A100

Date: 2026-10-29. Machine: Raven, 1x A100-SXM4-40GB (nodes ravg1002 on `gpudev`; ravg1128 and ravg1202 on `gpu1`), Xeon Platinum 8360Y, 18 cores per job.
Everything lives in `/raven/u/uli/gizmo-bench/`. `/raven/u/uli/gizmo` was not modified: the source was exported with `git archive gpu_kokkos_unified` (HEAD b653e863, 2026-10-29, recorded in `src/EXPORTED_FROM_COMMIT`).

## TL;DR

N = 2,097,152 (128^3), periodic unit box, MFM (GIZMO's standard meshless method), no gravity, global timesteps. The time step is the mean of steady steps 2-17 from `cpu.txt`.

| Run | s/step | M updates/s | Notes |
|---|---|---|---|
| **GPU, MFM, 1 rank + 18 OpenMP threads (default placement)** | **6.23** (repeat on another node: 6.19) | **0.336** | job 31666473 / 31666785 |
| GPU, MFM, particle-storage placement hint off (`GPU_PARTICLE_STORAGE_PLACEMENT=0`) | 6.31 | 0.333 | the placement hint makes no difference |
| GPU, MFV (`HYDRO_MESHLESS_FINITE_VOLUME`) | 7.42 | 0.283 | |
| CPU, same source, Kokkos-OpenMP, 1 rank x 18 threads | 5.96 | 0.352 | |
| CPU, same source, Kokkos-OpenMP, 18 ranks x 1 thread | **3.96** | **0.529** | faster than the GPU |
| 64^3 GPU smoke test (262k) | 0.78 | 0.336 | same rate as at 2.1M |

Comparison on the same A100 (2.1M cells, Sedov 128^3):

| Code | s/step | updates/s |
|---|---|---|
| LUNA, fixed grid | - | 1,100 M/s |
| PROTEUS | 0.422 | 5 M/s |
| AREPO-WD GPU | 3.58 | 0.59 M/s |
| GIZMO GPU MFM | 6.23 | 0.34 M/s |

So GIZMO GPU is **15x slower than PROTEUS** and **1.7x slower than AREPO-WD** per step, even though GIZMO has no mesh construction. On this x86 + A100 node the GPU build is **slower than the same code on 18 CPU cores** (3.96 s with 18 MPI ranks).

The nsys profile shows why:
- **GPU kernels run only ~23 % of the wall time.**
- Every step moves about 12 GB of unified memory as page faults (8 GB device-to-host, 4 GB host-to-device).
- The particle arrays `P[]`/`CellP[]` live in CUDA managed memory. Host-side passes over them stall on page migration over PCIe: kicks, timestep, drift bookkeeping, domain, density h-iteration control and tree build.

GIZMO's own GH200 numbers (docs: 1M particles, 2.14 s/step with gravity) rely on the coherent Grace-Hopper memory, which this node does not have.

## Build

- **Kokkos 4.6.01** (GIZMO's documented version for the Lonestar6 A100 block; our Kokkos 5.0.99 requires C++20 while GIZMO builds with `-std=c++17`). Cloned from GitHub into `ext/kokkos-4.6.01`, built per the Makefile's Lonestar6 recipe:
  - `-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON -DCMAKE_CXX_STANDARD=17`, with `nvcc_wrapper`;
  - install prefix `ext/kokkos-4.6.01-cuda-a100`; script `scripts/build_kokkos.sh`.
- **OpenMP Kokkos 4.6.01** for the CPU reference: `ext/kokkos-4.6.01-omp` (`-DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_ICX=ON`, g++).
- **Modules:** `module load gcc/14; module load openmpi/4; module load hdf5-serial/1.14.1; module load cuda/12.8` (OpenMPI 4.1, HDF5 1.14.1, CUDA 12.8.1). There is no FFTW or GSL: neither is needed without gravity/PM.
- **Makefile:** I added two SYSTYPE blocks to the exported `src/src/Makefile` (export only), copied from `Lonestar6` and `Lonestar6_CPU`:
  - `Raven_A100`: `nvcc_wrapper --std=c++17` for the GPU TUs with `--expt-relaxed-constexpr --expt-extended-lambda -arch=sm_80`, `mpicxx -std=c++17 -O2 -g -march=icelake-server -Wall -fopenmp` for host TUs, linking `-lkokkoscore -lkokkoscontainers -lcudart -lcuda`;
  - `Raven_CPU`: the same with the OpenMP Kokkos and `mpicxx` for all TUs.
- The GPU build compiled and ran first time, with no source changes.
- Build script: `scripts/build_gizmo.sh BUILDDIR [SYSTYPE]`.
- Build directories: `build-mfm`, `build-mfm-cpu`, `build-mfv`, `build-mfm-pnone`.
- **Note:** the first builds used `make -j48` on the login node, which overloaded it; the script now defaults to `-j4`.

`Config.sh` (`build-mfm`; it follows GIZMO's own `test/sedov/Config.sh` plus input precision and global steps):
```
HYDRO_MESHLESS_FINITE_MASS
BOX_PERIODIC
SELFGRAVITY_OFF
EOS_GAMMA=(5.0/3.0)
INPUT_IN_DOUBLEPRECISION
OUTPUT_IN_DOUBLEPRECISION
DEVELOPER_MODE
FORCE_EQUAL_TIMESTEPS
```
- The code reports these derived options: `HYDRO_FIX_MESH_MOTION=5`, `DOMAIN_LIGHTWEIGHT_REPARTITION`, `SLOPE_LIMITER_TOLERANCE=1`, `STOP_WHEN_BELOW_MINTIMESTEP`.
- `build-mfv` is the same with `HYDRO_MESHLESS_FINITE_VOLUME`.
- `build-mfm-pnone` adds `GPU_PARTICLE_STORAGE_PLACEMENT=0`.
- GPU vs CPU is chosen only by the Kokkos backend (SYSTYPE), not by Config flags. The GPU path needs no gravity or other module.
- The double-precision hydro is GIZMO's only mode; fast-math is forbidden by the code.

## Initial conditions

`scripts/make_ics.py` is the AREPO-WD script reused unchanged; GIZMO reads the same HDF5 layout. The Sedov setup:
- `ics/sedov_128.hdf5`: 128^3 lattice at cell centres, jittered by ±0.25 dx per axis (seed 12345), periodic BoxSize 1;
- ρ = 1 (m = dx^3), v = 0, γ = 5/3, P = 1e-5;
- E = 1 as thermal energy in the 35 particles with r < 2 dx of the centre.

It is identical to the AREPO-WD `sedov_128` IC. `ics/sedov_64.hdf5` is the 64^3 analogue (34 hot particles).

## Parameters and runs

`scripts/param_template.txt`:
- code units of 1, G = 1 (unused);
- `DesNumNgb 32`, `MaxNumNgbDeviation 0.1`, `CourantFac 0.3`, `MinGasTemp 0`, `InitGasTemp 0`;
- `DomainBuild_ActiveFraction 0.01`;
- no snapshots except the final one;
- `TimeMax 3e-5`, `MaxSizeTimestep = TimeMax/4`.

**Pitfall:** `MaxSizeTimestep` larger than `TimeMax` overflows the integer timeline, and the run stops with "Timestep wants to be below the limit MinSizeTimestep".

**Timesteps:** global, via `FORCE_EQUAL_TIMESTEPS`, so all 2,097,152 particles are active every step.
- dt = 9.4e-7 for the first 4 steps, then 1.875e-6. The 17 steps reach t = 3e-5.
- So, as for AREPO-WD, the blast is only a few cells across, and the cost is that of a full all-active step.

Runner: `scripts/run_case.sh NAME BINARY IC TMAX [NRANKS] [NTHREADS]` sets `OMP_NUM_THREADS=18 OMP_PROC_BIND=spread OMP_PLACES=cores` and runs `srun --cpu-bind=cores ./GIZMO param.txt 0`. Job scripts:
```
#SBATCH --partition=gpudev --nodes=1 --ntasks=1 --cpus-per-task=18 --gres=gpu:a100:1 --mem=120000 --time=00:15:00
module purge; module load gcc/14; module load openmpi/4; module load hdf5-serial/1.14.1; module load cuda/12.8
```
- `scripts/job_smoke64.sh`: job 31666457, 64^3.
- `scripts/job_sedov128.sh`: parametrised by CASE/BUILD/TMAX/NR/NT.
  - 31666473: GPU MFM.
  - 31666474: CPU 1x18.
  - 31666558: CPU 18x1, with `--ntasks=18 --cpus-per-task=1`.
  - 31666783: MFV.
  - 31666784: pnone.
  - 31666785: GPU repeat.
- `scripts/job_nsys128.sh`: job 31666555, 8 steps under `nsys --trace=cuda,nvtx --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true`.

Logs are in `runs/<case>/log.txt` and `runs/<case>/output/cpu.txt`. Per-step numbers come from `scripts/extract.py RUNDIR`. `cpu.txt` is **cumulative** per bucket, so the script differences steps 1 and 17 and divides by 16.

The GPU run uses 1 MPI rank, the model GIZMO documents (one rank per GPU, host threads for the remaining host loops). The defaults print "Neighbor loops: ... Mode A otherwise". The offload thresholds (4096 etc.) are far below N, so every loop goes to the device.

## Raw timer lines (GPU MFM, `runs/mfm_sedov128_gpu`)

cpu.txt at step 10 (cumulative since start, init included):
```
Step 10, Time: 1.5e-05, CPUs: 1
Nactive=2097152, Imbal(Max/Mean)=1
total              73.19  100.0%
tree+gravity        8.19   11.2%
   treebuild        7.12    9.7%
   treewalk         1.07    1.5%
hydro/fluids       33.19   45.3%
   density         16.33   22.3%
   gradients        6.69    9.1%
   hydro_frc        7.46   10.2%
   hmaxupdate       1.68    2.3%
   misc_hydro       1.03    1.4%
ngb build           2.01    2.7%
pair kernel         3.40    4.6%
domain              7.26    9.9%
peano               0.79    1.1%
drift/splitmg       5.56    7.6%
kicks               3.63    5.0%
find_timestep       2.45    3.3%
io/snapshots        1.53    2.1%
misc                5.18    7.1%
```
stdout per step:
```
 X  bin=56               0     2097152     0.000001875000          2097152 < *        6.25    100.0%
  density computation done (1.4856 s) via runner
Symmetric neighbor list: 2097152 active, 64415732 pairs (0.2191 s)
```
That is 30.7 neighbours per particle in the symmetric CSR list. Init, step 0, took 10.8 s; the first density pass alone took 3.0 s.

### Per-step breakdown, mean of steps 2-17 (s/step)

The buckets are disjoint: GIZMO subtracts child buckets, so `pair kernel` and `ngb build` are not inside `density`/`gradients`/`hydro_frc`.

| Bucket | GPU MFM | GPU MFV | CPU 1x18 | CPU 18x1 |
|---|---|---|---|---|
| total | **6.234** | 7.421 | 5.956 | 3.964 |
| treebuild + treewalk (tree built even with SELFGRAVITY_OFF) | 0.688 | 0.874 | 0.384 | 0.074 |
| density (h iteration, host side of the runner) | 1.484 | 1.912 | 1.257 | 0.623 |
| hmaxupdate | 0.153 | 0.154 | 0.148 | 0.012 |
| ngb build (CSR neighbour list) | 0.181 | 0.322 | 0.813 | 1.299 |
| gradients | 0.608 | 0.715 | 0.179 | 0.173 |
| hydro_frc | 0.678 | 0.772 | 0.288 | 0.138 |
| pair kernel (device pair-kernel launches + fence; density, gradient and force loops) | 0.307 | 0.384 | 1.206 | 1.199 |
| misc_hydro | 0.091 | 0.097 | 0.078 | 0.011 |
| ghost import (corridor) | 0 | 0 | 0 | 0.234 |
| domain | 0.497 | 0.526 | 0.441 | 0.046 |
| drift/splitmg | 0.546 | 0.589 | 0.309 | 0.050 |
| kicks | 0.363 | 0.384 | 0.251 | 0.044 |
| find_timestep | 0.239 | 0.255 | 0.236 | 0.023 |
| misc | 0.398 | 0.437 | 0.366 | 0.037 |
| updates/s | **0.336 M** | 0.283 M | 0.352 M | 0.529 M |

Shares for GPU MFM:

| Group | Buckets | s/step | Share |
|---|---|---|---|
| Neighbour search + density | density + hmaxupdate + ngb build | 1.82 | **29 %** |
| Tree build/walk (only a spatial index for the neighbour loops, since gravity is off) | treebuild + treewalk | 0.69 | 11 % |
| Hydro | gradients + hydro_frc | 1.29 | 21 % |
| Device pair kernels of all hydro loops | pair kernel | 0.31 | 5 % |
| Integration and bookkeeping | drift, kicks, find_timestep, domain, misc, misc_hydro | 2.13 | **34 %** |

- Density plus neighbour structures, tree included: **40 %**.
- Hydro gradients plus fluxes: ~21-26 %.
- The rest is integration and bookkeeping.

Note the inversion against the CPU run: the device pair kernels (0.31 s) are 4x faster than on 18 threads (1.21 s), and the CSR build is 4.5x faster. But every bucket dominated by host loops over `P[]`/`CellP[]` is 1.5-3.5x slower on the GPU build: gradients, hydro_frc, drift, kicks, tree, domain.

## nsys (job 31666555, 8 steps, 7.1 s/step under the profiler)

Steady window, t = 20-76 s of the timeline, ~7.9 steps:
- **GPU kernel busy: 13.1 s of 56 s = 23 %**, i.e. ~1.65 s of kernels per step.
- **Unified-memory migration:** 63.9 GB device-to-host and 29.6 GB host-to-device in the window. That is ~8.1 GB + 3.7 GB per step, in ~1 M transfers, with 3.3 + 2.7 s of copy-engine time.
- Over the whole run: 1.06 M CPU page faults and 24k GPU page faults.
- Explicit memcpys are negligible: 1.3 GB H2D, 2.1 GB D2H.
- CUDA API time is 73 % `cudaDeviceSynchronize`, i.e. the host waiting on fences.
- Top kernels are anonymous Kokkos `parallel_for`s of 55-230 ms each. Raw CSVs: `runs/mfm_sedov128_gpu_nsys/stats_*.csv`.

The particle data are 2.7-2.9 GB in managed memory; on this node GIZMO advises it "device-preferred" (1.3-1.5 GB) with host access mapped. Turning the hint off (`build-mfm-pnone`) gives 6.31 vs 6.23 s/step, i.e. no change. The migrations come from the host touching the arrays every step, not from the hint.

## Caveats

- **What is timed:** `cpu.txt` `total` per step (wall time of rank 0), averaged over steps 2-17.
  - Init excluded: IC read, first domain decomposition, first density pass, 10.8 s.
  - Step 1 excluded (warm-up).
  - The final snapshot and restart write come after the last step and are excluded.
  - Steady steps vary by < 1 % (6.16-6.29 s).
  - Repeat on another node: 6.19 s.
- **Host-side work is included,** and it dominates. GIZMO keeps kicks, timestep selection, domain bookkeeping and the h-iteration control on the host. The particle arrays are UVM, and on a PCIe-attached A100 each step pays ~12 GB of page migration. On a Grace-Hopper node (GIZMO's Vista development machine) the managed memory is coherent, and this cost is much smaller. So this number is specific to x86 + A100, and is how GIZMO's own A100 (Lonestar6) configuration would behave.
- **Same physics as AREPO-WD?** The physics is comparable, but the discretisation is not: meshless MFM with 32 effective neighbours (≈31 per particle in the list) and both sides of each pair evaluated, against AREPO's Voronoi faces (~15.5 per cell). There is no mesh construction; the "mesh-like" cost is the neighbour search, density h-iteration and tree, about 40 %. With MFV: 7.42 s/step.
- **The tree is built every step even with `SELFGRAVITY_OFF`** (0.6 s/step). It serves as the spatial index for the neighbour loops.
- **CPU reference:**
  - The same export, built with Kokkos-OpenMP (`Raven_CPU`), on 18 cores of a GPU node (the GPU is unused).
  - 18 MPI ranks x 1 thread (3.96 s) beats 1 rank x 18 threads (5.96 s). The 1x18 runs the same host-serial bookkeeping as the GPU build.
  - GIZMO's docs compare against a non-Kokkos CPU build (`Vista_CPU`), which no longer exists on this branch.
- **Not tuned:** offload thresholds (`GPUElementOffloadMinCount` etc.), multiple ranks per GPU, `NeighborLoopModeD*` routing. All defaults were used.
- The `gpudev` node ravg1002 was shared with other jobs on its other GPUs, as in the AREPO-WD runs. The `gpu1` repeat on ravg1202 agrees to 1 %.
- Early-time Sedov only (t ≤ 3e-5). The cost per step is that of an all-active step, not of a developed blast.
