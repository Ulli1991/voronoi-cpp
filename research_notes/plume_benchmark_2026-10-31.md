# plume's local Voronoi build vs ours, PROTEUS and voro++ (2026-10-31)

Benchmark only. No plume, PROTEUS or voro++ code or algorithm went into
voronoi-cpp; nothing was pushed or changed in the plume repo (a local copy
took a timing harness, see below).

## Versions and jobs

| code | version | build |
|---|---|---|
| plume | rennehan/plume main `cf55509a` (2026-10-30), local copy `/raven/u/uli/plume-bench/plume-work` | cuda-oxide `6921d3ee`, rustup nightly-2026-08-28 in ~/.rustup, ~/.cargo (user space), `cargo oxide build --release`, kernels to PTX by the nightly's llc, `default-arch = sm_80`; CUDA 13.2 (module cuda/13.2: plume's CUB wrappers need CCCL's `cuda/iterator`, absent in 13.0; README asks 13.1), clang/21 for bindgen, hdf5-serial/1.14.1 |
| ours | voronoi-cpp main `77a903b` (git archive in build-plumecmp/src-tree) | gcc/14 cuda/13.0, Kokkos /u/uli/kokkos-cuda, nvcc_wrapper; build job 31699972 |
| PROTEUS | /raven/u/uli/proteus-bench/ProteusGPU (not re-run) | numbers of job 31694621 (cleanup2_2026-10-31.md) |
| voro++ | chr1shr/voro master `b0dac57` (2026-03-04), /raven/u/uli/plume-bench/voro | gcc/15, `-O3 -march=icelake-server`, static lib; harness tools/voro_bench.cc |

GPU: A100-SXM4-40GB, driver 580.178.04 (jobs 31700277, 31700427). CPU:
Xeon Platinum 8360Y, 72 cores (general partition).

Inputs: our `make_points` (src/test/support.hpp, on the 2^30 grid), dumped
to raw f64 by /raven/u/uli/plume-bench/tools/dump_points.cpp (the generator
lines copied verbatim from our support.hpp, same libstdc++), 2,097,152 points
per kind, and 262,144 for the size comparisons; SWIFT's glassCube_64 (plume's
Sedov start) for the README comparison. All codes see the same positions
(plume: words = floor(x 2^32) << 32, exact for 2^-30 points).

## How each was timed

- **plume**: plume ships no stand-alone benchmark for the build (only its
  `#[ignore]` GPU tests and the full Sedov example programs), so a harness
  example `crates/plume/examples/vbench/main.rs` was added to the local copy.
  It calls plume's public API only, in the same order as its own tests
  (`local_voronoi/src/pass.rs`) and partition (`partition/src/cell_rounds.rs`):
  Morton sort and positions as words, `Topology::rebuild` and `NodeRadii::build`
  (the "tree" number), `find_neighbors` over every row (the "search"), then
  `build_face_table` over every row (the "build": exact-sign cell, certificate,
  volume, centroid offset, face table with neighbour row, area vector and face
  centre). Smoothing lengths: host distance to the 32nd neighbour (the length a
  running simulation would have predicted; plume's Sedov `neighbor_target = 32`),
  search radius = reach = 1.15 h (`search_margin = 1.15`). Rows left
  uncertified are searched again at 1.15 x the radius their cell asks for and
  built again, up to 3 rounds, as the partition does (on host for the radius
  update; untimed). Median of 5 after one warm-up run; double precision (plume's
  default `Real`). Only build changes in the local copy otherwise:
  `.cargo/cuda-oxide.toml` default-arch sm_75 -> sm_80, and the CUB wrapper
  build scripts (columns, neighbors) to sm_80 SASS, because nvcc 13.2 PTX for
  compute_75 can't be JIT-compiled by the 580 driver.
- **ours**: `voronoi_bench rebuild <n> <kind> warp 3 --no-check` (median of 3):
  construct_ms = the warp kernel, which includes the per-cell neighbour walk of
  the cell list (shell search), clipping, the certificate, volume, centroid
  and faces (neighbour, area vector, face centroid), double precision;
  build_ms = the cell list (0.2-1 ms). The table uses construct_ms.
- **PROTEUS**: not re-run; mesh time per step from job 31694621 (its whole
  mesh step including its own search and data movement).
- **voro++**: `container_periodic`, the unit box scaled to 2^16 (exact) because
  `put()` aborts on two particles closer than 1e-5 absolute units ("Duplicate"),
  which clustered and halo have in the unit box; `nx=ny=nz=75` (4.97 per
  block, the recommended ~5). Per cell, `voronoicell_neighbor`: volume, centroid,
  neighbour ids, face areas, and face centroids computed from `vertices` and
  `face_vertices` (area-weighted fan; voro++ has no face-centroid call); no
  area vectors (normals not requested). Parallel: one container shared
  read-only (`create_all_images()` first), one `voro_compute` per OpenMP thread,
  (j,k) block rows handed out dynamically (`schedule(dynamic,1) collapse(2)`),
  72 threads, `OMP_PROC_BIND=spread`. "cells" = the cell loop; "setup" = the
  container fill (serial; its per-insert duplicate check is quadratic in block
  occupancy). Volumes tile the box to 1e-15 in every run.

## Results, M cells/s (2,097,152 cells, A100 for GPU codes)

| input | plume build (search+tree+build) | ours construct (incl. search) | PROTEUS mesh/step | voro++ 72 cores, cells (with setup) | voro++ 1 core |
|---|---|---|---|---|---|
| jittered | **7.17** (6.12) | **34.7** | 6.81 | 2.58 (0.93) | 0.039 |
| uniform | fails: "needs more than 64 vertices" | **16.5** | 4.10 | 2.34 (0.87) | 0.035 |
| clustered | no number (see below) | **13.0** | 2.00 | 0.81 (0.46) | 0.014 |
| halo | fails: "needs more than 64 vertices" | **12.1** | 1.57 | 0.080 (0.070) | 0.009 |

plume jittered 2.1M in ms: tree 1.6, search 48.2, build 292.4, no row left
uncertified (volume sum 1 - 3e-15). Its build is 4.8x slower than ours, and
4.2x slower if its search is counted too.

Jobs: plume and ours 31700277 and 31700427 (A100, driver 580.178.04); voro++
31700541 (72 threads), 31700542 and 31700543 (1 core); PROTEUS 31694621.

### 262,144 cells (and the README's 64^3 Sedov figure)

| input | plume build (with search+tree) | ours construct |
|---|---|---|
| SWIFT glassCube_64 (plume's Sedov start) | **11.4** (8.34): build 23.1 ms, search 7.9 ms | (not run: our bench generates its own points) |
| jittered 64^3 | 7.01 (5.97): build 37.4 ms | 33.6 (7.8 ms) |
| uniform | full neighbour list after searching again (harness) | 13.8 |
| clustered | full neighbour list after searching again (harness) | 3.85 |
| halo | "needs more than 64 vertices" | 6.65 |

The README's 64^3 Sedov figure is 262,144 cells in about 55 ms on a Quadro
RTX 6000 (4.8 M cells/s, at t = 0.02). The same count on the A100, on the
t = 0 glass, builds in 23 ms (11.4 M cells/s), about 2.4x faster. So the A100
is roughly 2.4x the RTX 6000 for plume's build. The evolved state at t = 0.02
differs slightly from the glass (51 listed and 16 examined neighbours per row,
per the README).

## Why plume fails on uniform, halo and clustered

- **Fixed capacities.** In 3D a cell can hold at most 64 vertices, 48 planes,
  32 faces and a 32-edge hole (`capacity.rs`). The README sizes these at 1.5-2x
  the most measured on glasses, lattices, Sedov and Kelvin-Helmholtz. The
  64-vertex limit is hard: `clip.rs` asserts `VERTICES <= u64::BITS` (vertex
  bitmasks), so a copy with doubled capacities did not compile. A Poisson cell
  (uniform random) or a cell in the halo's density gradient sometimes passes 64
  vertices while being built, and plume stops with a crash naming the row.
  This happens at 2.1M on uniform and halo, and at 262k on halo.
- **Full lists.** The clustered runs (2.1M and 262k) and uniform 262k made it
  past the first build. Searching uncertified rows again at 1.15 x the radius
  they ask for (symmetric max(h_i, h_j) pairs, 352 slots) then filled a list.
  plume's partition would shrink such a radius 0.7x and search again
  (`full_lists.rs`), but my harness doesn't, so these runs give no number. A
  vertex overflow could still follow.

## What each number means: precision and guarantees

- **plume**: double precision. Every cut is decided by the exact sign of an
  in-sphere test on positions snapped to a 2^-48 grid: a floating-point filter
  first, then 64-bit or 320-bit integers. So two cells meeting the same
  particles decide alike, and a face of positive area is in both tables. Every
  written cell is certified to be the exact Voronoi cell. Faces of zero or
  nearly zero area may appear on one side only. It crashes instead of truncating
  when a capacity is exceeded. It builds only rows whose radius certifies. The
  time above excludes the host smoothing-length step.
- **ours (warp rebuild)**: double precision. Cuts are rounded with a
  tolerance, and cells are certified by the walk. Cells over capacity, or that
  can't be certified, are rebuilt by the per-thread kernel. These timings don't
  include the exactness step: exact topology comes from the exact start and
  repair (`tet_start`, predicates on the 2^30 grid, about 2x the rebuild cost,
  120 ms at 2.1M jittered).
- **PROTEUS**: per-step mesh time of its own pipeline; its exactness wasn't
  examined (source not read).
- **voro++**: double precision, its own tolerances (`tolerance =
  10 eps x max_len_sq`), no exact predicates and no cross-cell consistency
  guarantee. Its fixed block grid makes dense regions costly: per-cell cost grows
  with block occupancy, so halo is 29x slower than jittered on one core. With 72
  threads, halo scales only 9x because a few block rows through the cores
  dominate. Setup is serial and isn't parallelised here.

## Caveats

- plume's harness is mine, not plume's. Its search follows plume's API but
  leaves out late-partner walks (list symmetry after searching again), which
  don't change any cell.
- plume needs CUDA 13.1+ (the README says 13.1). cuda/13.0 failed on
  `cuda/iterator`, so plume ran on cuda/13.2 while ours ran on cuda/13.0.
- The A100 runs clock-unlocked; no nsys.
- Paths: harness /raven/u/uli/plume-bench/plume-work/crates/plume/examples/vbench/main.rs,
  binaries /raven/u/uli/plume-bench/bin, logs /raven/u/uli/plume-bench/logs
  (gpu-31700277, gpu2-31700427, vL72/vL1a/vL1b), points
  /raven/u/uli/plume-bench/data.
