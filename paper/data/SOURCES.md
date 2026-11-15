# Sources of paper/data

| file | what | from | jobs, commit, clock |
|---|---|---|---|
| `oracle_errors.csv` | per-cell relative volume error of the 1000 sampled cells, every volumes file of the library benchmark | `paper/scripts/extract_oracle.py` (+ `oracle_errors.cpp`, same sample as `bench/libraries/oracle_check.cpp`) on `/raven/u/uli/libbench-data/volumes/*.f64` and the oracle caches `points/*.oracle1000` | oracle jobs 32225958, 32234618 (caches); checks 32269164 and the login run `accuracy-main-login.jsonl`; voronoi-cpp volumes from ad62997 (main b2faee8 + benchmark). The extractor checks every median against those records (all match) |
| `ceiling.json` | DRAM bytes, bandwidth, occupancy per kernel label (Nsight Compute), 128³ | `scripts/ceiling_tables.py` on `/raven/u/uli/voronoi-cpp-joblogs/voronoi-cpp-ceiling/build-nvtx/jobs` | jobs 32222161, 32222162, 32222164, 32222165, 32222228; main 69de1d3; 1279-1305 MHz per kernel |
| `raw/weak-<job>.jsonl` | one JSON line per timed run (library_bench) | `paper/scripts/weak_scaling_a100.sh` via `submit_sweep.sh` | see `weak_scaling.csv` (jobs, hosts and logged SM clock per row) |
| `raw/check-<job>.txt` | `voronoi_bench rebuild` with its cell checks and faces per cell, per input and n | same jobs | |
| `weak_scaling.csv` | median / min / max M cells/s per commit, input, n and path; peak B/cell; SM clock under load | `paper/scripts/extract_sweep.py` | |
| `raw/kernels-<job>.txt`, `stages.csv`, `kernels.csv` | ms per step per stage and per kernel label, 128³ | `stages_a100.sh kernels`, `extract_stages.py` | |
| `raw/memory-<job>.txt` | Kokkos arrays at the peak of a rebuild and a step | `stages_a100.sh memory` | |
| `raw/long128-<job>.txt`, `raw/long256-<job>.txt` | 500 steps at 0.1 spacings with and without re-sort | `stages_a100.sh long128`, `long256` | |
| `clocks/clock-<job>.csv` | nvidia-smi every second: SM clock, application clock, utilisation, temperature, power | every A100 job above | |

Library-benchmark rates (`library_comparison`) are read in place from `bench/libraries/results/*.jsonl` and
`bench/libraries/published.csv` through `scripts/library_benchmark.py` (voronoi-cpp rows ad62997, node ravg1002 at
1095 MHz; provenance table in `docs/library_benchmark.md`).

Point files: `/ptmp/uli/voronoi-paper/points` (128³ and 256³ linked to the library benchmark's files, same generators
and seed; the rest made by the sweep with `library_bench points` and, for blue noise, 100 Lloyd iterations).
Binaries: `/ptmp/uli/voronoi-paper/bin/<commit>/` (built by `paper/scripts/build_cuda.sh`: gcc 14, CUDA 13.0,
Kokkos 5.0.99 AMPERE80, Release).
