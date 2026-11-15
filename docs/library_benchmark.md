# Library benchmark: methods and results

voronoi-cpp against public Voronoi and Delaunay libraries on the same points, the same box, the same outputs and the
same hardware, with every number regenerated from raw records by one script. This file is the methods section; the
README carries the headline table and the figure. Tables in full: [bench/libraries/results/summary.md](../bench/libraries/results/summary.md).

## Results in brief

The README's table and figure (section "Against other libraries") show the 128³ build rates. This section adds 256³ and
the observations behind the numbers; every number is in [summary.md](../bench/libraries/results/summary.md).

RESULTS_DETAIL

## Inputs

Points are written once by `library_bench points` ([src/bench/library_bench.cpp](../src/bench/library_bench.cpp)) from
the generators of `src/test/support.hpp` (fixed seed 20261029) and read by every library from the same file
(`bench/libraries/make_points_a100.sh`, job 32225761):

| file | points | SHA-256 (first 16 hex) |
|---|---|---|
| jittered_2097152.pts | 128³ lattice, each point moved up to ±0.1 spacing per axis | f7fc0e00d13844c4 |
| uniform_2097152.pts | independent uniform | d65cd75f4e20157a |
| clustered_2097152.pts | 8 Gaussian blobs (σ 0.04) with 80 %, 20 % uniform; about 1:500 in density | 10c91b377f64b790 |
| halo_2097152.pts | 4 halos ρ ∝ 1/(r² + c²), c = 0.002, 90 %, 10 % uniform; about 1:10⁴ | cf81db50496a43f7 |
| *_16777216.pts | the same at 256³ | d902c5b8…, 033d9416…, dcb3094c…, a6ed08e4… |
| jittered_2097152_step0.03.trj | jittered 128³ in cell-list (slot) order, and 10 steps of the bench flow, each moving the points 0.03 mean spacings | 2cb9d68fe4e2e940 |

Format (`bench/libraries/libbench.hpp`): uint64 n, uint64 frames, then frames × n × 3 doubles. Positions are in the
periodic unit box [0, 1)³ on the grid of 2⁻³⁰ that voronoi-cpp snaps to, so every library gets the identical doubles
voronoi-cpp computes on. The files (4.6 GB) are not in the repository: the generator and seed are.

## What is timed

The clock (`std::chrono::steady_clock`, wall time) runs from the points in the library's natural memory (host for CPU
codes, device for GPU codes unless the API takes host points, as stated per row) to the library's output in its own
memory: cell volumes, and faces with areas where the library gives them. No file I/O, no copy of the result back. A
library that only builds a Delaunay tetrahedralization is timed twice: the Delaunay alone, and the Delaunay plus cell
volumes from it by our own dual computation (`images_dual.hpp` on the CPU, a CUDA kernel for gDel3D); both rows are
shown. Rates are M cells/s: the n cells of the input (images not counted) per second.

Protocol (`run_protocol` in `libbench.hpp`): one untimed run that measures the peak memory and writes the volumes for
the accuracy check, untimed warm-up runs until 2 s have passed (the A100's clock ramps up from 210 MHz; a 2M build of
60 ms is otherwise timed cold), then 3 timed runs. Tables give the median and the min–max of the 3. Moving-mesh steps:
each of the 10 trajectory steps is timed; the tables give the median and range of steps 2 to 10.

Peak memory, in B/cell of the n input cells: on the host the resident set's high-water mark above the resident set at
the start of the run (`/proc/self/clear_refs` reset, `VmHWM`); on the GPU the device use (`cudaMemGetInfo`, total
minus free) sampled every 200 µs on a second thread, above the use at the start of the run. Both count everything the
process allocates during the build, including library pools and, for padded codes, the images. For voronoi-cpp the
Kokkos arrays alone are recorded as well (Kokkos Tools callbacks; side records in summary.md).

## Like for like, and what cannot be matched

- Same points, same box: native periodic unit box where the library supports it (voronoi-cpp, voro++, CGAL
  `Periodic_3_Delaunay_triangulation_3`, geogram `PeriodicDelaunay3d`, peclet-voro). The others are padded with periodic
  images in a shell around the box (width per input in each row's `periodic` field), wide enough that every original
  cell is exact; the accuracy check confirms it. The time includes building and triangulating the images.
- Same outputs: volumes for every row; faces with areas where the library gives them (`outputs` per row). Delaunay-only
  codes get our dual volumes, timed separately.
- Same precision: double everywhere a library allows it. Ray et al.'s VoroGPU and Liu et al.'s code compute in float
  (cells, volumes) and Paragram's adjacency is float32; they are run in float and marked so.
- Same hardware: GPU codes on one A100-SXM4-40GB (Raven `gpu1`, 18 host cores of the node allotted, SM clock logged
  before and after each run: 1305 MHz under load, 1410 MHz maximum); CPU codes on one full Raven node (2× Xeon
  Platinum 8360Y, 72 cores, `--exclusive`), at 72 threads (parallel codes) and at 1 thread. OpenMP codes run with
  `OMP_PROC_BIND=spread OMP_PLACES=cores`; geogram runs unbound (its thread pool is otherwise pinned to one core,
  job 32226520).
- Robustness differs and is part of the result: voronoi-cpp's predicates are exact on the 2⁻³⁰ grid; CGAL and geogram
  use exact (filtered) predicates; voro++, peclet-voro, VoroGPU, Liu et al. and Paragram use floating-point tests.
- Not matched: library tuning beyond documented defaults (block counts, neighbour counts K and similar are listed per
  row); GPU codes that take host input include the upload; published numbers are on other hardware and points.

## Accuracy

`bench/libraries/oracle_check.cpp` compares each library's volumes with the brute-force oracle of
`src/test/oracle.hpp` (exact 128-bit predicates and long-double clipping against all periodic images; it shares no
code with voronoi-cpp) on 1000 cells drawn with a fixed seed from each point file, and checks all n cells for missing or
non-positive volumes and the sum against the box. Oracle cost: 60 s at 2.1M and 500 s at 16.8M on 72 cores (job
32225958).

## Reproducing

```
# inputs (one A100 job), then each library: build script, then run script (Slurm; see each script's header)
sbatch -o <log> bench/libraries/make_points_a100.sh <voronoi-cpp CUDA build> <data dir>
sbatch -o bench/libraries/results/<lib>-<device>-%j.jsonl bench/libraries/run_<lib>_<device>.sh ...
sbatch -o bench/libraries/results/accuracy-<lib>-%j.jsonl bench/libraries/oracle_cpu.sh <data dir> <volumes>...
python3 scripts/library_benchmark.py   # summary.md and docs/figures/library_benchmark.png from the records
```

Third-party sources are fetched by the build scripts into a directory outside the repository, at the versions in the
provenance table; local patches needed to build the older GPU codes with current CUDA are in `bench/libraries/*.patch`.

## Provenance

Versions and builds (the build scripts `bench/libraries/build_<lib>.sh` fetch and build each exactly so; every record
carries its version, threads, host and job id):

| library | version | build | runs (jobs) |
|---|---|---|---|
| voronoi-cpp (A100 rows) | 09d31d0 = main 9e9ec03 + this benchmark | Kokkos 5.0.99 CUDA (AMPERE80), gcc 14, CUDA 13.0, nvcc_wrapper, Release | 32232716 |
| voronoi-cpp (CPU rows) | 799bc38 = main 45db7fa + this benchmark (the step kernels of 9e9ec03 are GPU-only tuning) | Kokkos OpenMP (gcc 15), `-march=icelake-server`, Release | 32226238 |
| voro++ | master b0dac57; multithreaded: dev 61099f6 (Lu, Lazar and Rycroft 2023) | gcc 14 `-O3 -march=native` (dev `-fopenmp`) | 32230277, 32230747, 32230748 |
| CGAL | 6.2.1, Boost 1.74, oneTBB 2023.2; Epick kernel | gcc 14 `-O3 -march=native -DNDEBUG`, C++20 | 32226051, 32226052 |
| geogram | 8d4b20af24b1 | its Release flags (`-O3 -fopenmp -ffp-contract=off`), gcc 14 | 32230300, 32230303 |
| Qhull | 2020.2 (v8.0.2, 613debe), `qh_new_qhull("d Qbb Qz Qt")` | CMake Release `-O3 -march=native`, gcc 14 | 32230279 |
| TetGen | 1.6.0 (535f9c4), `tetrahedralize("Qz")` | `-O3 -march=native` (predicates as its makefile), gcc 14 | 32230278 |
| gDel3D | a429be8 + `gdel3d.patch` (build fixes for current CUDA), double, star splaying on | `-O3`, sm_80, gcc 13, CUDA 12.6 | 32230284 |
| VoroGPU (Ray et al.) | f4d7e7e + `vorogpu.patch`; K 190 (PRESET 0) and K 90 (PRESET 1), float as upstream | `-O3`, sm_80, gcc 13, CUDA 12.6 | 32230285, 32230286 |
| Liu et al. | b2c64f5 + `voronoi-gpu-liu.patch`; K 90, float | `-O3`, sm_80, gcc 13, CUDA 12.6 | 32230287, 32230288 |
| Basselin et al. | 196e7d7, OpenCL on the NVIDIA driver; K 70, P 30, V 40 (README settings), relaunch × 1.5 | gcc 13 `-O3`, CUDA 12.6 runtime | 32230958, 32230959 |
| peclet-voro | v1.0.4-1-gd7371c9, Kokkos 5.1.1 (CUDA AMPERE80; OpenMP) | its CMake Release, gcc 14, CUDA 13.0 | 32230289, 32230290, 32230291 |
| Paragram | 4f36b1f (0.1.0), torch 2.11 + cu128, its default build (`-use_fast_math`) | lazily built CUDA extension, sm_80, gcc 14, CUDA 12.8 | 32230794, 32230795, 32230957 |
| votess | f0f805f; A100: AdaptiveCpp v25.02.0 (generic SSCP, clang 18.1.8, CUDA 12.2); CPU: icpx 2025.3 `-fsycl` | `-O3` | 32231706; 32231009, 32231211, 32231821, 32231822 |

Inputs: job 32225761. Oracle caches: job 32225958. Accuracy checks: the `accuracy-*.jsonl` files. The first runs of
voronoi-cpp (timed without warm-up, and on the code before 9e9ec03) are kept in `bench/libraries/results/superseded/`.
GPU clocks: our A100 jobs ran at 1305 MHz under load; the other libraries' A100 jobs at 1095 or 1305 MHz (logged per
run in `/raven/u/uli/libbench-data/logs`, the SM clock of each job's node is not ours to set).
