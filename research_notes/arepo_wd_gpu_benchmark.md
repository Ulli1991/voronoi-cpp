# AREPO `Arepo2_WD` (Pakmor) VORONOI_CELL / VORONOI_CELL_GPU benchmark on one A100

Date: 2026-10-29. Machine: Raven `gpudev` node ravg1002, 1x A100-SXM4-40GB, Xeon Platinum 8360Y, 18 MPI ranks (1 core each).
Everything lives in `/raven/u/uli/arepo-wd-bench/`. The AREPO repository was not modified.

## TL;DR

N = 2,097,152 (128^3), periodic unit box, double precision. Steady-state rebuilds (every step is a full rebuild under `FORCE_EQUAL_TIMESTEPS`):

| Run | `VCELL: built` time (median, builds 3+) | cells/s (built) | device clip wall only | full `voronoi` timer per step | cells/s (`voronoi`) |
|---|---|---|---|---|---|
| GPU, jittered ±0.25 dx | 1.540 s | **1.36 M/s** | 1.12 s (1.87 M/s) | 2.46 s | 0.85 M/s |
| GPU, uniform random | 2.429 s | **0.86 M/s** | 1.79 s (1.17 M/s) | 3.46 s | 0.61 M/s |
| CPU path, same binary (`AREPO_VCELL_GPU=0`), 18 ranks, jittered | 3.208 s | **0.65 M/s** | n/a | 4.18 s | 0.50 M/s |
| CPU path, uniform random | 4.496 s | **0.47 M/s** | n/a | 5.51 s | 0.38 M/s |
| GPU, first build from scratch (init, no previous mesh), jittered | 6.32 s | 0.33 M/s | | | |
| GPU, first build from scratch, uniform | 7.74 s | 0.27 M/s | | | |

Our numbers for comparison (A100, construction only, 2.1M cells): warp-double 14.8 M/s jittered and 8.2 M/s uniform; lean-double 7.9 and 3.6 M/s.
So our warp-double is **~11x (jittered) and ~10x (uniform) faster** than their steady-state `built` time. Against their device clip wall alone (the most favourable reading for them) we are 8x and 7x faster.
The GPU path is only ~2.1x (jittered) and 1.85x (uniform) faster than their own CPU path on 18 cores. GPU and CPU builds give bitwise identical meshes (same `build checksums`).

Sedov (2.1M cells, 20 global steps):
- **Everything on the GPU** (mesh plus per-cell gradients and fluxes): **3.58 s per full step = 0.585 M cell-updates/s**. The mesh is 71 % of the step.
- Mesh on the GPU with the classic per-face hydro on 18 CPU ranks: 4.24 s per step = 0.49 M/s, with the mesh at 55 %.
- PROTEUS: 0.422 s per step (75 % mesh). AREPO-WD is **8.5x slower per step**.

Run-to-run spread: the same jittered build took 1.54 s in job 31653699 and 1.34 s in job 31654033 (`gpu_jitter128_bigmem`, and Sedov at 1.35-1.40 s). So the jittered GPU rate is **1.36-1.57 M cells/s**. The node was shared with other jobs on the other 3 GPUs.

## Build

Source: `git -C /u/uli/codes/arepo archive origin/Arepo2_WD | tar -x -C /raven/u/uli/arepo-wd-bench/src`. Head commit as exported: the code reports the git hash of a local throw-away `git init` in the export, not the upstream hash.
Build fixes, all in the export only:
- The Makefile needs a git repository (submodule check, pre-commit hook). I ran `git init` and made one local commit in `src/`, then `touch build/git_submodules.check`.
- The `lib/safeintegral` submodule is empty in an archive. I cloned it from GitHub and checked out the pinned commit `49441eb392c5`.
- `Makefile.systype`: `SYSTYPE="Raven-OpenMPI"` (already present in `buildsystem/systypes.make`, including the CUDA wiring: `nvcc -ccbin=mpicxx -arch=compute_80 -code=sm_80`).
- **CUDA 13.0 does not compile.** `src/gpu/backends/cuda_backend.h:111/113` calls the old `cudaMemAdvise(ptr, n, advice, int device)` signature, which CUDA 13 removed (it now takes a `cudaMemLocation`). I used **cuda/12.8** instead.

Modules (login node build and runs): `module load gcc/14 openmpi/4 gsl fftw-serial hdf5-serial hwloc cuda/12.8` (openmpi 4.1.8, gsl 2.7, fftw 3.3.10, hdf5 2.2, hwloc 2.12).

Compiler lines (from `make.log`; the systype's `-O3` is overridden by the build system's defaults):
```
mpicxx -std=c++11 -g -march=native -O2 -flto -Wall -Wextra ... -DH5_USE_16_API=1
nvcc -ccbin=mpicxx -g -O2 -std=c++11 -arch=compute_80 -code=sm_80 --extended-lambda ... -fmad=false   (voronoi_cell_gpu.cu; FP contraction pinned off for bitwise host/device parity)
```

### Build A: mesh on the GPU, classic hydro (`build-gpu/Config.sh`)
```
VORONOI
VORONOI_CELL
SHAREDNODEMEMORY
CHUNKING
REGULARIZE_MESH_CM_DRIFT
REGULARIZE_MESH_CM_DRIFT_USE_SOUNDSPEED
REGULARIZE_MESH_FACE_ANGLE
VORONOI_DYNAMIC_UPDATE
FORCE_EQUAL_TIMESTEPS        # VORONOI_CELL needs this or TIMESTEPS_WAKEUP; gives global steps = full rebuild each step
DOUBLEPRECISION=1
OUTPUT_CENTER_OF_MASS
HAVE_HDF5
GPU
GPU_CUDA
VORONOI_CELL_GPU
```
No gravity and no MHD. GAMMA is the default 5/3. Pure hydro uses the classic per-face finite-volume solver on the CPU.

### Build B: everything on the GPU (`build-percell/Config.sh`)
This is build A plus `MHD`, `RIEMANN_HLLD`, `MHD_POWELL`, `FINITE_VOLUME_SOLVER_PER_CELL` and `FINITE_VOLUME_SOLVER_PER_CELL_GPU`. The per-cell solver requires the MHD/HLLD/Powell flux path (README_FINITE_VOLUME_SOLVER_PER_CELL "Requirements"). B = 0 in the ICs, so the physics is pure hydro through an HLLD solver.

## Initial conditions

`scripts/make_ics.py` (run with `/u/uli/jax-env/bin/python3 -I`) writes HDF5 (ICFormat 3), double precision, BoxSize 1, ρ = 1 (mass = dx^3), v = 0, γ = 5/3, seed 12345.
- `jitter_128`: 128^3 lattice at cell centres, each coordinate displaced uniformly in **±0.25 dx**, wrapped periodically.
- `uniform_128`: 2,097,152 i.i.d. uniform points.
- Both use P = 0.6 (c_s = 1). There are also 64^3 versions (`jitter_64`, `uniform_64`).
- `sedov_128`: the same ±0.25 jittered lattice, P = 1e-5, total E = 1 put as thermal energy into the 35 cells with r < 2 dx of the box centre. Periodic unit box.

## Runs

The parameter template is `scripts/param_template.txt`. Main settings: periodic, `MaxMemSize 3000`, `MaxMemSizeNode 8000`, `CourantFac 0.3`, `CellShapingSpeed 0.5`, no snapshots except the final one, and `NumberOfGlobalTimestepsToStopAfter` to stop the run.
VORONOI_CELL parameters are at their defaults: `MeshCandidateSource 1` (hybrid: tree walk capped at 1024 candidates, then a graph ball), `MeshCellCandidateLimit 4096`, `MeshCellRetirement 1`, `MeshDeviceMinCells 20000`, `GpuMaxMemSize 0` (automatic).
The runner `scripts/run_case.sh` launches `srun --cpu-bind=cores ./Arepo param.txt` with 18 ranks. That gives one shared-memory island and one GPU; island rank 0 is the "feeder" that owns the device, and the other 17 ranks walk and stage.

Job scripts:
- `scripts/job_mesh128.sh` (job 31653699): `gpu_jitter128`, `gpu_uniform128` (5 steps); `cpu_jitter128`, `cpu_uniform128` (3 steps, `AREPO_VCELL_GPU=0`).
- `scripts/job_sedov128.sh` (job 31653700): `sedov_meshgpu` (build A, 20 steps); `sedov_percell_gpu` (build B), which died, see below.
- `scripts/job_round2.sh` (job 31654033): `sedov_percell_gpu` with `CPUsUsedPerGPU 1`, and `gpu_jitter128_bigmem` (the same, mesh only).
- `scripts/job_test64.sh` / `job_test64b.sh`: 64^3 smoke tests.

Header used:
```
#SBATCH --partition=gpudev --nodes=1 --ntasks=18 --cpus-per-task=1 --gres=gpu:a100:1 --mem=120000 --time=00:15:00
module purge; module load gcc/14 openmpi/4 gsl fftw-serial hdf5-serial hwloc cuda/12.8
```
Logs are in `runs/<name>/log.txt`, `runs/<name>/output/{cpu.txt,voronoi.txt}`. Numbers come from `scripts/extract.py RUNDIR N`.

## Raw timer lines (2.1M cells)

Startup configuration line (all runs):
```
VCELL: mesh construction: candidate source 1 (hybrid, tree walk capped at 1024 then graph ball), node-local tree with halo imports, 1 islands, device on above 20000 cells per island
GPU MEMORY: GpuMaxMemSize is automatic: task 0's device reports 32835 MB free and is addressed by 18 rank(s), so 90 % of it is 1640 MB per task
```

GPU, jittered (`runs/gpu_jitter128`). Builds: 6.321 (init) 1.800 1.549 1.541 1.540 1.538 1.538 s.
```
VCELL: built 2097152 cells in 1 rounds (6.3214 s): 32610002 faces (both sides), 92.8 candidates/cell avg of 130 max, 0 deferrals, 0 radius retries, 0 K relaxations (final K=4096)
VCELL: built 2097152 cells in 1 rounds (1.54105 s): 32596240 faces (both sides), 33.7 candidates/cell avg of 69 max, 0 deferrals, 0 radius retries, 0 K relaxations (final K=1024)
VCELL: build checksums (must be invariant under reruns, K and scheduling): vol=0002dbf3b013dbf2 rmax=000bf24392456696
VCELL: waves: 512 waves, 29 launches, 2097152 cells clipped in batches, 0 radius retries, host tail 0 (arena 0, face slot 0, list 0, long list 0); face slot 34
VCELL: device: 1 islands on the device (0 below MeshDeviceMinCells 20000, 0 on the host for lack of device memory), island build sets 2097152 to 2097152 cells over 1 islands with cells, wave 4096 of 4096 requested, 29 island batches, 29 launches of up to 73728 cells, arena 0.0 MB + staging 701.6 MB per device, H2D 136.0 MB, D2H 1549.4 MB, feeder clip 1.121 s (...), longest wait 1.135 s, 0 cells verified against the host mirror
VCELL: devbuild: 1 islands build on the device, pool 1024 blocks x 4096 candidate slots, walk list 273.1 MB per island (...); host tail: long list 0, no drift/kernel 0; 0 long rings
VCELL: graph: cells 0 (forced 0) | ... | device built 0 home 0
VCELL: split: critical rank 0 (feeder) total 1310.6 ms = begin 2.3 + loop 1307.7 (stage 2.4 bar1 3.2 enq 6.0 wait 1115.6 bar2 0.2 verify 0.0 resolve 88.3 commit 83.4; 31 iterations) + end 0.5
```

GPU, uniform (`runs/gpu_uniform128`). Builds: 7.738 (init, 3 rounds, 36111 radius retries) 2.416 2.560 2.519 2.412 2.399 2.429 s.
```
VCELL: built 2097152 cells in 3 rounds (7.73761 s): 32580670 faces (both sides), 84.1 candidates/cell avg of 243 max, 0 deferrals, 36111 radius retries, 0 K relaxations (final K=4096)
VCELL: built 2097152 cells in 1 rounds (2.51878 s): 32421910 faces (both sides), 63.5 candidates/cell avg of 264 max, 0 deferrals, 13 radius retries, 0 K relaxations (final K=1024)
VCELL: build checksums (must be invariant under reruns, K and scheduling): vol=00e41e8acae4575e rmax=0011c28b43a6b4ec
VCELL: waves: 522 waves, 29 launches, 2097152 cells clipped in batches, 0 radius retries, host tail 0 (...); face slot 46
VCELL: device: ... 29 launches of up to 73728 cells, staging 766.7 MB per device, H2D 136.0 MB, D2H 1535.4 MB, feeder clip 1.793 s, longest wait 1.805 s
VCELL: split: critical rank 0 (feeder) total 1981.9 ms = ... loop 1978.8 (stage 2.5 bar1 8.1 enq 6.1 wait 1787.2 ... resolve 86.7 commit 79.7)
```

CPU path, jittered (`runs/cpu_jitter128`, `AREPO_VCELL_GPU=0`). Builds: 6.775 (init) 3.438 3.245 3.189 3.208 s.
CPU path, uniform (`runs/cpu_uniform128`). Builds: 7.751 4.445 4.577 4.496 4.348 s.
```
VCELL: built 2097152 cells in 1 rounds (3.18912 s): 32596240 faces (both sides), 33.7 candidates/cell avg of 69 max, ...
VCELL: build checksums (...): vol=0002dbf3b013dbf2 rmax=000bf24392456696          <- identical to the GPU run
VCELL: built 2097152 cells in 1 rounds (4.49569 s): 32421910 faces (both sides), 63.5 candidates/cell avg of 264 max, ...
VCELL: build checksums (...): vol=00e41e8acae4575e rmax=0011c28b43a6b4ec          <- identical to the GPU run
```

cpu.txt per-step `diff` (mean of the steady steps, seconds):

| run | total step | voronoi | of which cellrounds+cellclip | dynamic | graphmaint | cellfill | hydro (grad / flux) | domain |
|---|---|---|---|---|---|---|---|---|
| gpu_jitter128 | 4.34 | 2.46 | 0.41 + 1.13 | 0.46 | 0.22 | 0.13 | 1.57 (0.37 / 1.09) | 0.15 |
| gpu_uniform128 | 5.49 | 3.46 | 0.42 + 1.85 | 0.54 | 0.38 | 0.13 | 1.58 (0.38 / 1.14) | 0.25 |
| cpu_jitter128 | 6.27 | 4.18 | 0.48 + 1.91 (+ cellgather 0.59) | 0.46 | 0.25 | 0.14 | 1.74 | 0.17 |
| cpu_uniform128 | 7.55 | 5.51 | 0.40 + 2.61 (+ cellgather 0.91) | 0.53 | 0.43 | 0.13 | 1.60 | 0.25 |

The CPU path has only one steady step in its average, but its per-build `built` times are stable (above).

## What is timed

- **`VCELL: built ... (X s)`** (`voronoi_cell_build()`, `voronoi_cell.cc:1603-2205`) is wall time from the start of the rounds driver to convergence and stats reduction. It includes:
  - the graph build begin;
  - drifting and initialising the cell states;
  - the node-local tree / halo setup (`crhalo`);
  - the whole device path: candidate walk, sort and clip on the GPU (`devbuild`), staging, H2D/D2H (D2H is ~1.5 GB per 2.1M-cell build: the face records), and host-side resolve and commit of the face records;
  - the allreduce.

  This is the number used for "cells/s (built)". It roughly equals `cellrounds + cellclip` in cpu.txt.
- **Not in `built`, but in the `voronoi` timer:**
  - `dynamic`: the VORONOI_DYNAMIC_UPDATE bookkeeping, 0.46-0.54 s;
  - `graphmaint`/`graphend`: building the connectivity CSR for the next build;
  - `cellfill`: filling the classic `Mesh.DP/VF` structures;
  - mesh `exchange`;
  - the node-local tree build (`ngbnl*`).

  The `voronoi` column is the full mesh cost inside a time step as AREPO uses it.
- **Device clip wall** (`feeder clip` on the `VCELL: device:` line) is the feeder's wait for the device batches. It is the closest thing to a pure-GPU construction time. On the device it includes the tree walk, the candidate bitonic sort and the clip with cell geometry. It excludes the host commit and resolve (~0.17 s) and staging.
- **Every steady build is a full rebuild** of all 2.1M cells (global timesteps). The first build of the run (at init) has no previous mesh and is 4-5x slower: it uses the uncapped tree walk at K=4096 with ~90 candidates/cell, and for uniform points also 3 rounds and 36k radius retries. Later builds have ~34 (jittered) / ~63 (uniform) candidates per cell.
- Candidate source: the startup line says hybrid (`MeshCandidateSource 1`). In every steady build the `VCELL: graph:` line shows `cells 0`, so **all cells took the capped tree walk on the device. The graph ball was never used**, and there was no host tail (`host tail 0`) and no radius retry in steady state (0-39 in the uniform builds).
- The `VCELL: device:` line shows the device budget limiting the launches to "29 launches of up to 73728 cells". That is at the automatic budget of 1640 MB per rank (32.8 GB free / 18 ranks, because every rank creates a CUDA context). Round 2 (`gpu_jitter128_bigmem`, `CPUsUsedPerGPU 1` → 29.5 GB budget for the feeder) still shows "29 launches of up to 73728 cells". The launch width is not set by the budget. That run built in 1.339 s (1.57 M/s; feeder clip 0.925 s), faster than the 1.540 s of job 31653699 with an otherwise identical setup. I put the difference down to run-to-run or node variance, not to the budget.
- Faces/cell: 32.6M faces (both sides) / 2.1M = **15.5 faces per cell** (jittered) and 15.5 (uniform). Kernel high-water at 64^3 was NFaces=24, NEdges=67, NVertices=46.

## 64^3 (262,144 cells) smoke tests

These used build B binary with the mesh on the GPU or the CPU (`runs/pc_gpu_jitter64`, `runs/pc_cpu_jitter64`, jittered, 15 steps):
- GPU: steady built 0.179 s, **1.46 M cells/s**; feeder clip 0.125 s; full step 0.455 s; `voronoi` 0.31 s.
- CPU path: built 0.346 s, **0.76 M cells/s**.

Throughput is about the same as at 2.1M, so the device is not starved at 262k either.

## Sedov full step

Setup:
- 128^3 = 2,097,152 cells, ±0.25 dx jittered lattice, periodic unit box;
- ρ = 1, P = 1e-5, E = 1 thermal in the 35 central cells (r < 2 dx), γ = 5/3;
- moving mesh with the regularisation options of the repo's `examples/sedov_3d/sedov_globaltstp_3d` (CM drift, face-angle);
- `FORCE_EQUAL_TIMESTEPS` (global steps, every cell active, full mesh rebuild every step);
- 20 steps (`NumberOfGlobalTimestepsToStopAfter 20`).

The global step is fixed by the hot centre at dt = 3.05e-6, so 20 steps cover t = 6e-5: the blast is only a few cells big and most of the mesh barely moves. The cost per step is therefore representative of a full rebuild plus full hydro on 2.1M cells, not of a developed Sedov.

### A. Mesh on GPU, classic per-face hydro on 18 CPU ranks (`runs/sedov_meshgpu`, build A)

Builds: 6.394 (init) 1.603 1.354 1.387 1.354 1.346 1.347 ... 1.348 s. Median **1.348 s → 1.56 M cells/s** (built). Feeder clip 0.937 s.

cpu.txt, Step 10 (representative; the mean over steps 2-19 is total 4.239 s, voronoi 2.331, hydro 1.545):
```
Step 10, Time: 3.05176e-05, CPUs: 18, MultiDomains: 4, HighestActiveTimeBin: 14
                          diff            max          cumulative
total                     4.25  100.0%   0.00      76.75  100.0%
  ngbtreebuild            0.02    0.4%   0.02       0.20         0.3%
  voronoi                 2.33   54.9%   0.00      33.04        43.0%
    exchange              0.05    1.3%   0.07       0.59              0.8%
    dynamic               0.52   12.2%   0.54       5.25              6.8%
    cellrounds            0.40    9.5%   0.00       5.03              6.6%
    cellclip              0.94   22.2%   0.95      12.00             15.6%
    cellfill              0.13    3.1%   0.13       2.03              2.6%
    graphmaint            0.22    5.1%   0.00       2.77              3.6%
  hydro                   1.55   36.5%   0.00      16.92        22.0%
    gradients             0.38    8.8%   0.39       4.54              5.9%
    fluxes                1.10   25.9%   1.14      11.53             15.0%
    fluxcomm              0.05    1.2%   0.07       0.54              0.7%
  domain                  0.19    4.4%   0.03       2.02         2.6%
  misc                    0.10    2.2%   0.10      20.25        26.4%
```

| quantity | value |
|---|---|
| wall per full step (cpu.txt `total`, mean of steps 2-19) | **4.24 s** |
| cell updates/s = N / step | **0.495 M/s (0.0005 G/s)** |
| mesh share (`voronoi` / total) | **55 %** (2.33 s; of which `built` region 1.35 s = 32 %) |
| hydro share (gradients + fluxes) | 36 % (1.55 s) |
| domain + other | 9 % |
| PROTEUS on the same A100 | 0.422 s/step, 0.005 G/s, 75 % mesh, i.e. **10x faster per step** |

### B. Mesh and per-cell hydro both on the GPU (build B)

First attempt (`runs/sedov_percell_gpu_oom_auto_budget`): the init mesh built (6.32 s), then the run died at the first per-cell hydro call:
```
TERMINATE: ... gpumalloc_fullinfo(), file src/gpu/gpu_memory_manager.cu, line 384: GPU MEMORY: out of device memory for 'FvsMirStage' at fvs_gpu_refresh_open()/src/hydro_per_cell/fvs_per_cell_gpu.cu/line 369: requested 256 MB, allocated 1466.67 MB of 1640 MB (GpuMaxMemSize)
```
The automatic budget gives each of the 18 ranks 1/18 of the device, but only the feeder rank allocates. Retry with `CPUsUsedPerGPU 1` (the whole 90 % budget to the feeder): it ran 20 steps (`runs/sedov_percell_gpu`). Device high-water was 5.1 GB.

Builds: 6.637 (init) 1.693 1.407 1.408 1.399 ... 1.399 s. Median **1.403 s → 1.50 M cells/s** (built). Feeder clip 0.953 s.

cpu.txt, Step 10 (the mean over steps 2-19 is total 3.584 s, voronoi 2.550, hydro 0.621):
```
total                     3.60  100.0%
  voronoi                 2.56   71.0%
    exchange              0.17    4.6%
    dynamic               0.54   15.0%
    cellrounds            0.41   11.5%
    cellclip              0.97   26.9%
    cellfill              0.15    4.1%
    graphmaint            0.24    6.7%
  hydro                   0.62   17.2%
    gradients             0.20    5.5%
    fluxes                0.36   10.0%
    updates               0.04    1.0%
  domain                  0.21    6.0%
  misc                    0.10    2.8%
```

| quantity | value |
|---|---|
| wall per full step (mean of steps 2-19) | **3.58 s** |
| cell updates/s | **0.585 M/s (0.0006 G/s)** |
| mesh share (`voronoi` / total) | **71 %** (2.55 s; `built` region 1.40 s = 39 %) |
| hydro share | 17 % (0.62 s: GPU gradients 0.20, GPU fluxes 0.36) |
| PROTEUS on the same A100 | 0.422 s/step, i.e. **8.5x faster per step** |

Moving the hydro to the GPU saves 0.92 s per step (1.55 → 0.62 s). The mesh, still ~2.5 s per step, is then 71 % of the step, close to PROTEUS's 75 % share, but at ~8x PROTEUS's absolute mesh time. Of AREPO's own mesh time, only ~1 s is the device clip; the rest is host-side work:
- `dynamic` 0.54 s;
- face-record resolve and commit ~0.17 s;
- the graph CSR for the next build 0.24 s;
- `cellfill` into `Mesh.DP/VF` 0.15 s;
- halo setup and exchange.

## Caveats

- **Our numbers vs theirs measure different scopes.** Ours: construction kernel(s) only, positions already on the device. Theirs (`built`): a full production build inside AREPO with 18 MPI ranks, including host staging, ~1.5 GB of D2H face records per build, host commit, and the halo/tree setup. Their pure device clip (feeder clip) is the fairer kernel-only comparison, at 1.87 M/s jittered and 1.17 M/s uniform. It still includes their on-device tree walk and sort.
- **Their design favours the 2-slot host/device pipeline, not raw throughput.** The README's warp-per-cell kernel uses ~22 kB of shared memory per cell (7 blocks/SM on A100). The `devbuild` pool is 1024 blocks.
- **Wave/launch knobs were not tuned:**
  - `AREPO_VCELL_WAVE` was already 4096 (the requested and observed value);
  - `AREPO_VCELL_GPU_SLOTS`, `AREPO_VCELL_GPU_LAUNCH` were left at defaults;
  - only `CPUsUsedPerGPU 1` was tried (round 2).
- **18 ranks per GPU** is my choice: a quarter of the 72-core node per A100, matching the 4-GPU node. The README does not prescribe a rank count, only one island per device.
- **The jitter amplitude is ±0.25 dx**, not ±0.5. Uniform random is i.i.d.
- **The CPU-path reference** is the same binary with the device disarmed (`AREPO_VCELL_GPU=0`). It runs the host wave layer on all 18 ranks, and its meshes are bitwise identical to the GPU ones.
- **Builds use CUDA 12.8**, because the branch does not compile with CUDA 13.0 (`cudaMemAdvise` signature).
- **The Sedov steps are very early-time** (t ≤ 6e-5), so the mesh barely moves between builds. Each step is still a full rebuild. Build A's hydro is the classic CPU per-face solver; it is not GPU hydro.
- No `VERIFY` options were compiled in, so the timings are production-mode.
