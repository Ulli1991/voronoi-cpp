# Sources of paper/data

| file | what | from | jobs, commit, clock |
|---|---|---|---|
| `oracle_errors.csv` | per-cell relative volume error of the 1000 sampled cells, every volumes file of the library benchmark | `paper/scripts/extract_oracle.py` (+ `oracle_errors.cpp`, same sample as `bench/libraries/oracle_check.cpp`) on `/raven/u/uli/libbench-data/volumes/*.f64` and the oracle caches `points/*.oracle1000` | oracle jobs 32225958, 32234618 (caches); checks 32269164 and the login run `accuracy-main-login.jsonl`; voronoi-cpp volumes from ad62997 (main b2faee8 + benchmark). The extractor checks every median against those records (all match) |
| `raw/sweep-<job>.jsonl` | one JSON line per timed run (library_bench) | `paper/scripts/throughput_vs_n_a100.sh` via `submit_sweep.sh` | see `throughput_vs_n.csv` (jobs, hosts and logged SM clock per row) |
| `raw/check-<job>.txt` | `voronoi_bench rebuild` with its cell checks and faces per cell, per input and n | same jobs | |
| `throughput_vs_n.csv` | median / min / max M cells/s per commit, input, n and path; peak B/cell; SM clock under load | `paper/scripts/extract_sweep.py` | |
| `raw/kernels-<job>.txt`, `stages.csv`, `kernels.csv` | ms per step per stage and per kernel label, 128³ | `stages_a100.sh kernels`, `extract_stages.py` | |
| `raw/memory-<job>.txt` | Kokkos arrays at the peak of a rebuild and a step | `stages_a100.sh memory` | |
| `raw/long128-<job>.txt`, `raw/long256-<job>.txt` | 500 steps at 0.1 spacings with and without re-sort | `stages_a100.sh long128`, `long256` | |
| `clocks/clock-<job>.csv` | nvidia-smi every second: SM clock, application clock, utilisation, temperature, power | every A100 job above | |

Application-input rates (`application`) are read in place from the same results files (our rows 631af46, jobs 32367524,
32367883, 32367884 at 1305 MHz; the other A100 codes chains 32367907–32367911 and 32370411; CPU jobs 32367538–32367544),
completeness from `bench/libraries/results/accuracy-inputs-{32369335,32370726}.jsonl`; crashes (votess) as recorded in
`docs/measurements.md`, "Application inputs".

Library-benchmark rates (`library_comparison`) are read in place from `bench/libraries/results/*.jsonl` and
`bench/libraries/published.csv` through `scripts/library_benchmark.py` (voronoi-cpp rows ad62997, node ravg1002 at
1095 MHz; provenance table in `docs/library_benchmark.md`).

Point files: `/ptmp/uli/voronoi-paper/points` (128³ and 256³ linked to the library benchmark's files, same generators
and seed; the rest made by the sweep with `library_bench points` and, for blue noise, 100 Lloyd iterations).
Binaries: `/ptmp/uli/voronoi-paper/bin/<commit>/` (built by `paper/scripts/build_cuda.sh`: gcc 14, CUDA 13.0,
Kokkos 5.0.99 AMPERE80, Release).
| `luna/yee_vortex_convergence.npz` | LUNA MC 3D Yee vortex convergence | `git show yee-vortex:luna/yee_vortex/yee_vortex_convergence.npz` in luna-reference-solutions (99a52d0) | its own `slurm_job_id`, `gpu_clock_mhz` fields |
| `luna/vortex_norms.csv` | luna-cpp moving (two passes) and static vortex L1 | parsed from `voronoi-cpp/build-1ddiag/logs/{n-vortex-31920446,fin-norm-31922276}.out` (same regex as luna-cpp's `make_figures.py`) | jobs 31920446, 31922276 (library of early November) |

| `raw/libbench-vcpp-32366449.jsonl` | voronoi-cpp rows of the library benchmark rerun at 1305 MHz (ravg1034), 128³, warp / host / host-vc / exact / step | `bench/libraries/run_voronoi_cpp_a100.sh` with binaries 55cd6b1, points of the benchmark; volumes in `/ptmp/uli/voronoi-paper/libbench/volumes` | job 32366449 |
| `oracle_errors_vcpp_rerun.csv` | per-cell errors of those volumes against the exact reference (same sample and caches) | `ORACLE_OUT=... LIBBENCH_DATA=/ptmp/uli/voronoi-paper/libbench extract_oracle.py` | medians equal the benchmark's records |
| `memory_by_array.csv` | Kokkos arrays at each phase's peak | `extract_memory.py` on `raw/memory-32362454.txt` | 55cd6b1, 1305 MHz |
| `inputs.npz` | 64³ slabs, neighbour-distance histograms, density contrast (eight inputs and lattice, wave) | `extract_inputs.py` on `/ptmp/uli/voronoi-paper/points` | same generators and seed as the benchmark; `cosmicweb_262144.pts`, `surface_262144.pts`, `fccsolid_131072.pts` made on the login node with `library_bench points` built with the Kokkos OpenMP backend from c4e6249 (main 04eb9e4; `build-paper-omp`), which reproduces `halo_262144.pts` byte for byte |
| `oracle_errors_application.csv` | per-cell errors of the 1000 sampled cells of every volumes file of cosmic web 128³, surface 128³ and fcc solid 2²⁰, against the exact reference | `ORACLE_ONLY='_(cosmicweb\|surface)_2097152\.f64\|_fccsolid_1048576\.f64' ORACLE_OUT=oracle_errors_application.csv extract_oracle.py` on `/raven/u/uli/libbench-data` | oracle caches of jobs 32369335, 32370726; medians equal the records; voronoi-cpp 631af46 |
| `raw/libstages-a100-32368965.jsonl`, `raw/libstages-cpu-32368966.jsonl` | staged runs of the library benchmark (ns per stage) | copies of `bench/libraries/results/stages-*.jsonl` on branch library-stages (16393a3) | A100 job 32368965 (ravg1013, 1305 MHz), CPU job 32368966 (ravl1045, 72 threads) |
| `raw/gaps-chain-a100-32368078.jsonl`, `raw/gaps-cpu-pair-32367276.jsonl` | our volumes-only path with Basselin, VoroGPU, Liu on blue noise (one A100 job); our CPU path with voro++ (one node) | copies of `bench/libraries/results/{chain-a100-32368078,cpu-pair-32367276}.jsonl` on branch close-gaps (2be7448), voronoi-cpp 77055a4 | jobs 32368078 (ravg1004), 32367276 (ravc4079) |

**Stage mapping of `where_time`** (`paper/scripts/fig_time.py`, from `kernels.csv` and the wall totals of `stages.csv`;
the bench's position copy `Kokkos::View::initialization`, `deep_copy` left out):
check = voronoi::snap, exact_predicates::snap, tet_repair::screen;
flip = tet_repair::find, plan, apply, unlist, release, retry_stuck;
reconnect = tet_mesh::face_touched, face_release, face_offsets, face_assign, face_move, face_low, face_numbered,
face_numbers, face_ids, dirty_faces, append_rings, compact_rings, ring_entries;
geometry = tet_mesh::circumcentres, tet_mesh::faces, voronoi::total_volume;
per-cell results = tet_mesh::cell_face_counts, cell_face_fill, cell_face_start, cell_sort, cell_sums, fresh_counts,
fresh_fill, mark_cells, marked_counts, marked_fill;
host and other = the step's wall time minus the kernels above (any other label, e.g. Kokkos::ViewCopy-1D, falls here).
| `luna/kh/mode_<mesh>_<N>.txt`, `luna/kh/density_t2_1024.npz` | luna-cpp `hydro_kh2d <N> [--moving] --t-end=2.5`: t, mode amplitude M, peak kinetic energy density every 0.02; the density at t = 2 on 1024² pixels (`fig_kh.py <dump prefix>`) | Viper, one MI300A, job 13683847 (`kh_job.sh`); luna-cpp d209db7 (branch consolidate), voronoi-cpp c677926 (branch mi300a-hip), `-ffp-contract=off` | |
