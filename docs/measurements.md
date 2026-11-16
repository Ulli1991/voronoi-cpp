# Measurements: provenance

Where each number in the [README](../README.md) comes from: jobs, commits, clocks and the setup details left out of the README. Sections follow the README headings.

## Measured performance

One A100 40 GB (Raven gpu1), double precision, periodic unit box. Rates are million cells per second, cells processed per second of GPU time.

| Rows in the README table | Job | Commit |
|---|---|---|
| Moving-mesh update, update plus `update_cells()`, 128³ and 256³ | 32269165, 32237516 (interleaved with main 9e9ec03; [fewer bytes per step](#fewer-bytes-per-step-incremental-face-numbers-e-and-the-ring-re-walk-k2-2026-11-12)) | `step-bytes` (the ring re-walk commit) |
| `update_cells()` alone | 32231679 ([step kernels](#step-kernels-at-todays-bytes-2026-11-12)) | `step-kernels` |
| Full rebuilds at 2.1M and 16.8M, exact start at 2.1M | 32269868, 32269874, 32269869 ([lean cut](#canonical-rings-and-a-lean-cut-2026-11-13)) | `clip-canonical` |
| Exact start, the sum row, at 2.1M | 31701509 | 2311937 |
| The 256³ column, except the moving-mesh rows | 31694457 | f32c909 (older than the current code, which is faster there) |
| Moving-mesh rows (also in research_notes/mesh_speed_2026-11-04.md) | 31886231, 31886232 | |

Moving-mesh update rates are per step of a smooth flow that moves the points by the given number of mean interparticle spacings, spacing = (box volume / n)^(1/3). Steps are medians of steps 2 to 10 on jittered points. At 256³ the step moves particles by the same number of spacings, but the flow varies on the box scale, so neighbours shear half as much per step (half the flips per cell).

On jittered, uniform and clustered input at 0.1 and 0.3 spacings no triangle ever sticks. The first step out of a degenerate start (wave lattice, 2.1M) is repaired on the device alone in 0.81 s (job 31701509, research_notes/shrink_2026-10-31.md). Icosahedral shells at 1e-9 still leave about 8k of 262k stuck (host or rebuild).

Device memory: plain warp rebuild 701 B/cell (656 at 256³). Peak during start and steps, with face lists, 2028 B/cell (2007 at 256³, 34.8 GB in use), older code; the current step peak is 1998 B/cell at 128³ and 256³ alike, 1950 for the mesh's arrays (job 32145254, [measure_2026-11-10.md](measure_2026-11-10.md)). The per-cell face lists account for most of the step peak above the 2080 B/cell of commit 2fd8b50 (about 140 B/cell; not measured separately).

### Inputs

All in the periodic unit box, generated in `src/test/support.hpp`.

| Name | Points | Tests |
|---|---|---|
| jittered | cubic lattice, each point moved randomly by up to ±0.1 lattice spacing per axis | the typical hydro start; nearly regular, about 16 faces per cell |
| uniform random | independent uniform points | unstructured cells of very different size and shape |
| clustered | 8 Gaussian blobs (σ = 0.04) with 80 % of the points, 20 % uniform background | density contrast about 1:500 (blob centre to void) |
| halo | 4 halos with ρ ∝ 1/(r² + c²) to r = 0.2, c = 0.002, 90 % of the points, 10 % background | density contrast about 1:10⁴ (core to edge) |
| lattice | perfect cubic lattice | fully degenerate: 8 points on every circumsphere |
| wave (moved lattice) | the lattice with every point moved by up to 1e-5 spacing | nearly degenerate: ties broken only by tiny offsets |
| cosmic web | Zel'dovich approximation x = q + D ψ(q) from a cubic lattice: Gaussian random field with the BBKS CDM spectrum (Bardeen et al. 1986; Γ = 0.21, n_s = 0.96), σ₈ = 0.8 at D = 1, 100 Mpc/h box, all modes below the Nyquist number of the lattice; growth factor D = 0.5 (rms displacement 4.3 spacings at 128³) | filaments, walls and voids: density contrast about 1:40 on a 2-spacing grid, 4 % of that grid empty; weakly deformed lattice in the voids |
| fcc solid | fcc lattice of l³ unit cells (4 l³ atoms; l = 64, 128: 2²⁰, 2²³), each atom moved by an independent Gaussian of rms 0.0499 lattice constants per axis: the measured rms of the LAMMPS Lennard-Jones solid at ρ* = 1.05, T* = 1.0 near melting (MSD 0.018 σ², Lindemann ratio 0.12; research_notes/lammps_voronoi_2026-11-12.md on branch lammps-scout) | an MD crystal: near-ties of the rhombic dodecahedra broken by thermal noise |
| surface | 16 spherical shells of radius 0.2 on a bcc lattice of side 0.5 (8 cross the box faces), each around a torus of radii 0.1 and 0.05; uniform in area, moved along the normal by a Gaussian of rms 0.1 surface spacings √(area / n) | points on closed surfaces: thin needle cells reaching about 0.1 box into empty space, the graphics-benchmark case |

### Non-cubic box

Job 31702953 (research_notes/box3_2026-10-31.md). Kelvin-Helmholtz-like slab, Lx = Ly = 2, Lz = 16 spacings: `--box=2,2,0.0625` with 512 x 512 x 16 = 4.2M cells, and `--box=2,2,0.03125` with 1024 x 1024 x 16 = 16.8M cells.

| M cells/s | 4.2M | 16.8M |
|---|---|---|
| Warp rebuild, jittered | 39.9 | 39.9 |
| Warp rebuild, uniform | 19.9 | 20.0 |
| Exact start | 11.6 | 12.0 |
| Steps at 0.03 spacings | 105 | 108 |

The bench flow, stretched with the box, shears across the 16-cell thickness: 4x the flips per cell of the cube. Peak 2192 B/cell at 16.8M. The cubic case is bit-identical to the cubic-only code and within 0.3 to 0.7 % of its rebuild speed.

### Against other libraries (2026-11-12 to 2026-11-14)

The README's library table: methods, every setting, 256³ and the per-code versions, compilers and jobs in
[library_benchmark.md](library_benchmark.md); raw records in `bench/libraries/results/` and the tables generated from
them by `scripts/library_benchmark.py` (`summary.md`). voronoi-cpp rows: ad62997 (main b2faee8 plus the benchmark),
A100 jobs 32272549, 32272550 and 32319068 on `ravg1002` (gpudev, 1095 MHz under load), CPU job 32272650 (Kokkos OpenMP,
72 cores and 1). Every other GPU code ran on the same node in chained jobs (`bench/libraries/run_chain_a100.sh`), each
with our warp rebuild as a same-GPU reference. Inputs: jobs 32225761 (generators) and 32234292 (blue noise, 100 Lloyd
iterations). Oracle checks: job 32269164 and accuracy-main-login.jsonl (one core, login node).

### Application inputs: cosmic web, fcc solid, surface (2026-11-15)

The three application inputs of the Inputs table, 128³ and 256³ points (fcc solid: 2²⁰ and 2²³ atoms), point files from
`bench/libraries/make_points_a100.sh` (job 32367403). voronoi-cpp 631af46 (main c42c230 + the generators), one A100 at
1305 MHz, jobs 32367524, 32367883, 32367884; M cells/s, median of 3 (steps: of steps 2 to 10 at 0.03 spacings).

| M cells/s | cosmic web 128³ / 256³ | surface 128³ | fcc solid 2²⁰ / 2²³ |
|---|---|---|---|
| warp rebuild (cell list + clipping) | 12.0 / 12.8 | 0.036, 476 cells refused | 30.6 / 32.4 |
| exact start (`Mesh::update` + `update_cells`) | 5.1 / 4.9 | 0.026 | 12.9 / 9.5 |
| moving step, 0.03 spacings | 201 | 139 | 188 |

Peak device memory at 128³: 740 B/cell for the warp rebuild, 1869 B/cell for the exact start. Oracle checks (job
32369335, 1000 sampled cells each): every cell present and a largest relative volume error of 2.3e-13 for the exact
start, the moving steps, and the warp rebuild on cosmic web and fcc solid; the warp rebuild on the surface misses 476
cells (sum of volumes − 1 = −1.1e-3). `voronoi_bench rebuild --check` (job 32367880) passes on cosmic web 128³ and the
fcc solid (all certified, closure below 3e-16, volume sum within 5e-14); cosmic web 256³ fails only `faces=2*owned`,
the known one-sided zero-area faces of clipping alone. `step --verify` passes all 10 steps on cosmic web at 64³ and on
the fcc solid at 32³ unit cells.

Surface: clipping refuses cells (status degenerate: 69 at 64³, 476 at 128³, and one at 32³ on the OpenMP backend) and
runs about 300 times slower than on jittered points: its needle cells reach about 0.1 box into empty space, so each
clips about 6000 neighbours instead of 75. The exact start inherits the cost, since it clips first. Issue
[#60](https://github.com/Ulli1991/voronoi-cpp/issues/60), with a 953-point case. The mesh itself is right: Delaunay
exact in every step (`failing=0`), volumes within 2.3e-13 of the oracle, moving steps at 139 M cells/s.

The other libraries (`bench/libraries/results/summary.md`, section "Application inputs"; A100 chains 32367907 to
32367911 and 32370411, CPU jobs 32367538 to 32367544): the kNN codes lose cells wherever a cell has more neighbours
than K. On cosmic web VoroGPU K 190 misses 2.4 % of the cells (sum of volumes − 1 = −0.038), K 90 and Liu et al. 19 %,
Basselin et al. (white-noise setting) 673 cells; on the surface they lose almost every cell. On the fcc solid every
code is complete, the float codes with relative errors up to 4e-5. peclet-voro returns wrong volumes on the surface
(sum − 1 = +31 to +62) and runs out of memory at 256³; votess fails on cosmic web and surface (A100: CUDA error 700;
CPU: `std::length_error`); gDel3D's 8-spacing padding is too thin for the surface (82627 certificate violations, one
sampled cell off by 1.3 %); Paragram leaves 647 surface cells without a volume (22 of 1000 sampled cells off by more
than 1 %) and 5 at cosmic web 256³. The exact Delaunay codes (CGAL, geogram, Qhull, TetGen) and voro++ are right on
all three (oracle jobs 32369335 and 32370726). Fastest full cells besides ours, cosmic web 128³: gDel3D + our faces 0.87
and voro++ (72 cores) 0.83 M cells/s; surface 128³: geogram (72 cores) 0.69, gDel3D 0.68 (too thin padding); fcc solid
2²⁰: peclet-voro 3.7 (A100).

### Long runs

Job 31953232 (128³, 0.1 spacings per step): the step falls from 118 M cells/s at step 100 to 59 at step 500. Issue: https://github.com/Ulli1991/voronoi-cpp/issues/34. On Sedov the loss is at most 4.5 % over a run.

### GPU against a full CPU node, same code

Commit 57201bc built with the Kokkos OpenMP backend on one Raven CPU node (2x Xeon Platinum 8360Y, 72 cores, `OMP_PROC_BIND=spread`, job 31796079), against one A100 40 GB (jobs 31795767, 31795035). 2,097,152 cells, double.

Hardware ratios A100 / node: about 3.9 in memory bandwidth (1.55 TB/s against about 0.4 TB/s) and 1.8 in FP64 peak. Every phase gains more than the bandwidth ratio. The clipping is latency- and divergence-bound (DRAM traffic under 1 % on the GPU), the flip repair is many small parallel rounds with atomics, and even the bandwidth-like hydro passes gain 1.6 to 3x beyond bandwidth. The CPU build is the generic Kokkos OpenMP path (per-thread clipping, no hand vectorisation), so a CPU-tuned code would narrow the gap.

### 2D and 1D

`voronoi::Mesh<2>`: job 31952137, one A100 at 1305 MHz, double, Morton order, 0.03 spacings per step. Jittered at 0.1 / 0.3 spacings: 770 / 744 at 2²⁰, 1024 / 1015 at 2²⁴. Peak device memory 537 to 574 B/cell (3D: 1998 to 2028 B/cell at 2.1M, job 31952136). Details: research_notes/mesh_2d_2026-11-05.md.

`voronoi::Mesh<1>`: job 31888427, mesh phases of a moving Sod run at 1095 MHz (older than this branch, research_notes/stage1_1d_2026-11-04.md).

### After the fixes of #29, #30, #35, #36 (2026-11-09)

128³ rows of the README, one A100 at 1305 MHz, origin/main 11745ea against the fix branch in the same jobs: rebuilds job 32094782, moving steps job 32094783 (median of steps 2 to 9). Steps 0.03 / 0.1 / 0.3: 204-208 / 117-118 / 57.0 before, 198-200 / 115-116 / 56.5 after. Rebuilds unchanged at 39.4 / 19.7 / 15.4. Geometry checksums identical in the unit cube at 0.03 and 0.3, different in a cube of side 0.05 (job 32095015).

256³ column: job 32093328 at 1095 MHz, voronoi-cpp 2190882 (before the fixes); steps include `update_cells()`.

### After the fix of #45 (2026-11-10)

128³, one A100 at 1305 MHz, origin/main 61e9914 against branch `fuzz-45` in the same job 32146167 (median of steps 2 to 9, two rounds each). Steps 0.03 / 0.1 / 0.3: 198.4 / 115.5 / 56.2 before, 195.5 / 113.5 / 55.7 after (1.5 / 1.7 / 0.9 % slower); `tet_mesh::faces` unchanged at 2.60 ms, `tet_repair::screen` 2.11 to 2.28 ms. Rebuilds unchanged (38.65 / 19.15 jittered / uniform). Geometry checksums differ in the unit cube and in a cube of side 0.05: centres stored from a corner in another periodic image now round at their wrapped position. A first version with a division per face vertex cost 4 to 10 % (jobs 32145669, 32145809).

### Mesh fuzz campaign (2026-11-06)

Driver `src/test/mesh_fuzz_test.cpp`, campaigns `scripts/mesh_fuzz_cpu.sh` and `scripts/mesh_fuzz_gpudev.sh` (the A100 campaign at millions of points is below), results in `build-fuzz/` on Raven: 3D on the CPU 115,598 cases (`camp2`, `camp3`), 1D 1,978,498, 2D 31,389, with sanitizers and bounds checks 731 (`asan_out`), on the A100 316 (`cuda_out`), none of them with millions of points. The 423 unexplained 3D failures rerun after the fix of #29 left 24 (#45).

### A100 mesh fuzz at 3e4 to 1.6e7 particles (2026-11-10)

`scripts/mesh_fuzz_a100.sh`, branch `fuzz-45` (the #45 fix), jobs 32145583, 32146360 and 32146363 (11 node-jobs of 15 minutes, four A100 each); results in `build-cuda/a100/results/` of the `voronoi-cpp-fuzz45` worktree on Raven, HANG rechecks in `build-cuda/a100/hang/`. The first jobs ran the fix with a division per face vertex, the rest the committed code (same values but for exact half-period ties). 604 cases: 273 exact, 239 refused (130 coincident, 94 reach, 15 box range), 2 rough, 2 valid (more than 16000 planes in one cell), 3 WRONG, 85 HANG.

| Particles | Cases | exact | refused | other |
|---|---:|---:|---:|---|
| 3e4 to 1e5 | 239 | 116 | 121 | 1 rough, 1 WRONG |
| 1e5 to 1e6 | 167 | 88 | 76 | 2 valid, 1 WRONG |
| 1e6 to 4e6 | 68 | 37 | 30 | 1 rough |
| 4e6 to 1.6e7 | 45 | 32 | 12 | 1 WRONG |
| out of time | 85 | | | |

Profiles large / huge / giant: 432 / 56 / 116 cases. Every box family (cube, pow2, 123, ratio, ratiolong, hcp, long, rod, slab, thin, extreme) and generator (lattices sc, bcc, fcc, hcp, diamond; uniform, Poisson disc, quasi; jitter; Gaussian, power-law, shell, Fibonacci, filament, sheet, spiral, cusp clouds; density steps, voids, points on box faces, near and doubled pairs, coplanar and cospherical sets) had at least 22 cases. The 3 WRONG are all #48 (seeds 3000000000159, 3000000000237, 7000000000041); with its fix (edges represented by their lowest tetrahedron) 3000000000159 and 7000000000041 are exact and 3000000000237 is refused for 66 coincident particles, as the input has them. The cases out of time are #49: four rerun alone were still in their first `update()` after 830 s; two rerun on 36 CPU threads refused for reach after 353 and 713 s. The generator `quasi` puts every point on one plane (1/g² + 1/g³ = 1 for the plastic number g), so its cases test only that.

### Per-vertex balls (P1) and directional boxes (R8) in the clipping (2026-11-12)

Branch `clip-bounds`; the bound with the measurement build (`-DVORONOI_MEASURE`, counters `rounds` to `clips_r8`), timings with the plain build, all on one A100 in jobs on `gpu1` and `gpudev`, 2,097,152 cells. Script `scripts/clip_bounds_a100.sh`; logs archived on Raven in `voronoi-cpp-joblogs/voronoi-cpp-clip/jobs/`.

**The bound** (job 32221981, the walk of main unchanged, the tests evaluated on the side). P1 for a grid cell: it meets the ball of a live vertex (with the margins of the skip); R8 for a grid cell: it meets the box around those balls. P1 for a clip: the particle lies in a vertex ball of the cell at its turn; R8 for a clip: the cell's bounding box does not rule it out. Rounds are the warp's particle rounds (one particle per lane each). Mean, p99 and max per cell:

| input | particles read: main, P1, R8 | rounds: main, P1, R8 | clips: main, P1, R8 |
|---|---|---|---|
| jittered | 76, 80, 80 / 32, 36, 47 / 32, 41, 63 | 3.0, 3, 3 / 1.04, 2, 3 / 1.06, 3, 3 | 21.9, 25, 28 / 16.0, 21, 26 / 21.7, 25, 26 |
| uniform | 103, 176, 344 / 57, 84, 121 / 76, 125, 206 | 15.1, 27, 49 / 9.8, 18, 30 / 12.2, 22, 37 | 48.0, 92, 188 / 17.9, 28, 45 / 39.1, 70, 121 |
| clustered | 112, 238, 1133 / 63, 141, 683 / 82, 167, 687 | 16.7, 38, 278 / 10.4, 20, 112 / 13.1, 27, 167 | 50.7, 106, 448 / 19.8, 34, 59 / 41.6, 75, 165 |
| halo | 115, 440, 3812 / 61, 130, 636 / 80, 180, 1184 | 18.0, 78, 535 / 10.5, 24, 138 / 13.4, 37, 259 | 53.6, 210, 2096 / 19.4, 34, 59 / 41.9, 88, 583 |

P1 leaves only the clips that cut (it is the clip's own test) and halves the particles read and the rounds on unstructured input, and cuts the tails most (halo p99 clips 210 to 34). R8 keeps 75 to 80 % of the rounds and 78 to 82 % of the clips there. P1 contains R8, so both together, or R8 cascaded before P1, leave what P1 leaves. On jittered input P1 and R8 remove two thirds of the reads and rounds; clips that cut nothing are 27 % there.

**Implementations, all slower** (median of 5 rebuilds, `construct_ms`, main and the variant in the same job, 1305 MHz on `gpu1` unless noted):

| variant | jittered | uniform | clustered | halo | lattice | wave |
|---|---|---|---|---|---|---|
| main 69de1d3, ms | 54.2 | 109.5 | 139.6 | 152.4 | 43.1 | 56.4 |
| P1 per candidate: each lane tests its buffered particle against every live vertex at the flush, then only new vertices after each cut (job 32222094, 1095 MHz) | +22 % | +15 % | +12 % | +10 % | +17 % | +18 % |
| P1 per grid cell, each lane against every live vertex, plus the above (job 32222186, 1095 MHz) | +64 % | +42 % | +41 % | +38 % | +36 % | +60 % |
| R8 box around the balls in double, per batch; grid cells and particles outside it skipped (jobs 32222688, 32222998, 32223108) | +18 to +24 % | +26 to +32 % | +26 to +34 % | +22 to +30 % | +14 to +22 % | +17 to +23 % |
| the same, grid cells only / particles only (jobs 32223854, 32223855) | +17 % / +19 % | +26 % / +26 % | +26 % / +25 % | +23 % / +21 % | +14 % / +16 % | +16 % / +18 % |
| R8 box in float by `__reduce_max_sync` (jobs 32224849, 32225604) | +14 % | +19 % | +22 % | +18 % | +10 % | +13 % |
| the bounds compiled out, only the clip's check for a vertex past its kept end (job 32224057) | +3 % | +5 % | +3 % | +3 % | +2 % | +3 % |

With the R8 box in use the work falls as counted (job 32222999: particles read 76 to 31 jittered, 103 to 75 uniform; clips 48 to 39 uniform; no cell sent to the thread kernel), yet the time rises. Nsight Compute (job 32224827, uniform): 4.6 % more instructions, but stalls for instruction fetch (`no_instruction`) rise from 28 % to 43 % of samples, and local loads from 175M to 258M (stack frames of 176 to 208 bytes become 192 to 256, registers capped at 96). The warp kernel is limited by its instruction cache and registers, so any test added to the walk costs more than the work it removes; even the variant with the bounds compiled out is 2 to 5 % slower. Lattice input, where no grid cell is skipped, shows the fixed cost: 10 to 22 %.

Rigour of the variants: a bound skipped a grid cell or particle only with a margin above rounding (the cell widened by 64 ε of the longest side; the ball box widened by 2^-14 R, above the rounding of 64 later cuts, which stay in the hull of the current vertices); a cut whose new vertex lies past its kept end (that end within the tolerance outside) would send a cell that skipped anything to the thread kernel. None of this is in the library: the branch keeps the counters only, and the plain build's kernel is the same SASS as main's.

### Canonical rings and a lean cut (2026-11-13)

Branch `clip-canonical` against main (9e9ec03, and after the rebase 6ead543; with the bench's checksums only, `ref`), one A100, every comparison in one job. Script `scripts/clip_canonical_a100.sh`, tables by `scripts/clip_canonical_tables.py` and `scripts/clip_canonical_ncu.py`; logs in `build-jobs/` of the `voronoi-cpp-canon` worktree on Raven. `canon` is commit a53ed48 (canonical rings only), `final` the branch head.

**Canonical rings.** Each cell writes its owned rings in increasing neighbour id, each from its least particle, in both kernels; the warp kernel also starts each face's fan there. `mesh_test` checks the form and that the exact thread kernel writes the same entries as the warp kernel (0 of 4096 cells differ, jittered and uniform). Cost: 0 to 1 % on the rebuild (job 32237666, 1305 MHz) and 0.3 % more instructions per cell (9611 to 9642 jittered, job 32269328).

**The lean cut.** A new vertex goes into the lowest slot free before the cut, built by the lane that owns it: no stack of free slots, no pushes and no 64-bit shifts for the slots made. The starting cube's adjacency is a formula (the search took 4 % of the kernel), the radius uses a select instead of `Kokkos::max` (which branched), the candidate sort and the side test have no branches, a face walk finds its corner with one load, and the reach of a candidate is recomputed only after a cut. The cut's decisions and vertex bits are main's.

Rate of the warp clipping (`rebuild`, median of 2 × 5, M cells/s and change against main; job 32238284 at 1305 MHz for 128³, the head's source but for formatting; job 32269162 at 1305 MHz for 256³, one round; job 32269153 repeats 128³ at 1095 MHz with the same changes):

| | jittered | uniform | clustered | halo | lattice | wave |
|---|---|---|---|---|---|---|
| main, 128³ | 38.7 | 19.2 | 15.0 | 13.8 | 48.6 | 37.2 |
| canonical rings, 128³ (job 32237666) | 38.4 (−1 %) | 19.0 (−1 %) | 15.0 (0 %) | 13.7 (0 %) | 48.7 (0 %) | 36.9 (−1 %) |
| lean cut, 128³ | 42.6 (+10 %) | 21.9 (+14 %) | 17.1 (+14 %) | 15.5 (+12 %) | 55.1 (+13 %) | 40.8 (+10 %) |
| main, 256³ | 38.8 | 19.5 | 16.3 | 15.8 | 50.9 | 37.3 |
| lean cut, 256³ | 42.8 (+10 %) | 22.4 (+15 %) | 18.5 (+14 %) | 17.9 (+13 %) | 57.6 (+13 %) | 41.0 (+10 %) |

Exact start (`start_ms`, median of 2, job 32269154, 1305 MHz), M cells/s: main 17.5, 8.3, 7.2, 9.9, 6.6, 4.0 against 18.5, 8.9, 7.3, 10.3, 7.6, 4.3 (jittered, uniform, clustered, halo, lattice, wave); the start's own kernels and host waits vary by 5 to 8 % between runs.

Nsight Compute of the warp kernel (job 32269328, 1095 MHz), warp instructions per cell, kernel time per clip over the whole GPU (21.9 clips a cell jittered, 48.0 uniform), and stall shares; no spills, 96 registers in all three:

| | jittered: instructions, ns per clip | stalls: fixed latency, shared memory, instruction fetch, issue | uniform: instructions, ns per clip | stalls |
|---|---|---|---|---|
| main | 9611, 1.40 | 35, 14, 13, 24 % | 15128, 1.26 | 29, 10, 35, 16 % |
| canonical rings | 9642, 1.41 | 35, 14, 14, 23 % | 15159, 1.27 | 29, 10, 36, 16 % |
| lean cut | 8802, 1.27 | 36, 16, 11, 24 % | 14020, 1.10 | 31, 12, 29, 18 % |

**Mesh.** After the build and two steps at 0.1 spacings, `tets_checksum` (the start's tetrahedra as particle sets), `geometry_checksum`, `faces_checksum` and `lists_checksum` equal main's on all six inputs at 128³ (job 32269155) and 256³ (jobs 32269156 to 32269160; wave at 256³ runs out of the A100's 40 GB in this bench's step on main too, so for it the start alone, job 32269402: the same tetrahedra). The bad tetrahedra and patched cells of the start are main's. What changed once: `rings_checksum` and the build's `cells_checksum` (the tetrahedra and faces are numbered from canonical rings now); the step's `cells_checksum` differs between runs of main as well. Canonical rings and the lean cut give the same rings and the same face geometry of a plain rebuild (`clip_checksum`) bit for bit on all six inputs, so the cut's slots no longer reach any output. luna-cpp (a82092c) builds against the branch; `hydro_sedov` 128³ jittered, moving mesh, 50 steps, gives the same bits of U, x and volume as against main (job 32238689).

**After the rebase on 6ead543** (E and K2 in the repair): rebuild 42.6, 22.0, 17.0, 15.5, 55.1, 40.9 against main's 38.7, 19.2, 15.0, 13.7, 48.7, 37.2 at 128³ (+10 to +15 %, job 32269868, 1305 MHz) and 42.8, 22.4, 18.5, 17.9, 57.7, 41.0 against 38.9, 19.5, 16.3, 15.8, 50.9, 37.3 at 256³ (job 32269874); exact start 18.6, 8.8, 7.9, 10.3, 7.4, 4.4 against 17.6, 8.2, 7.2, 9.6, 7.1, 4.0 (job 32269869); the same checksums against main as above at 128³ and 256³ (jobs 32269867, 32269870 to 32269873); `hydro_sedov` the same bits (job 32269941).

**A fuzz crash, fixed.** A 1e6 A100 fuzz batch (profile huge, seed 4000000000091: 1.3M particles, 38 of them coincident, an update after the refused build) hit an illegal address: a rounded cell with two faces toward one neighbour (#48) gave both rings the same place in the canonical order, the end of the block stayed unwritten, and `assemble` read a garbage length. Equal neighbours now go in lane order. Main refuses that case cleanly.

**What it says.** The bookkeeping that slots and cut order forced was worth 10 to 15 %, not several times: the cut that is left is mostly the work of finding and building the new vertices, and the kernel still waits on that chain's latency and, on uniform and clustered input, on instruction fetch. A rebuild several times faster needs fewer clips (B, the rounds of `clip-split`) or no clipping (D).

### kNN-first clipping, hybrids and a vertex-ball filter (2026-11-15)

Branch `fewer-clips`; the code is in commit c192c49 (`shell_knn.hpp`, the `listed` and `walk_todo` modes of the warp kernel, chosen by `VORONOI_KNN` and `VORONOI_BALLS`) and was taken out again, so the library is main's. One A100 at 1305 MHz on `gpu1`, main b2faee8 (`ref`) and the branch in the same job. Script `scripts/fewer_clips_a100.sh`, tables by `scripts/fewer_clips_tables.py`; logs in `build-jobs/` of the `voronoi-cpp-fewer` worktree on Raven.

**The designs.** *kNN pass*: one thread per particle walks the cell list as a linear octree (the particle's level, stencil cells nearest first, coarser levels as the walk does) and keeps the k nearest in registers, sorted by a float key; it stops when the next cell lies beyond the k-th. *kNN-first*: the warp clips by the list, nearest first, and stops when the next entry lies beyond 2R; past the list every particle lies at least as far as its last entry (with a margin of 2^-20 for the float keys), so a cell is certified if that entry lies beyond 2R. A cell the list cannot certify is built again by the walk (*fallback*). Hybrids: `:resume` goes on with the walk in the same warp, skipping the listed particles; `k+k2` lists the uncertified cells again with k2 before the walk; `:even` lists only cells at the base grid level (even density) and walks the rest; `:balls8` and `w8` (the walk) skip a candidate, from the 8th of a batch on, that lies outside every vertex ball by a margin above the rounding of later cuts (2^-36 (|d| R + |d|²); later vertices stay in the hull of the current ones), which leaves every cut and every vertex bit as without it. Memory of the lists: 4k + 4 B/cell during the rebuild (k = 24, 32, 48: 100, 132, 196 B/cell).

**Rate of the rebuild** (`construct_ms`, median of 5; 128³ two rounds, job 32319474; 256³ one round, jobs 32319849, 32319851), M cells/s and against main; fallbacks are the cells the walk built after the list, as a share of all:

| design | jittered | uniform | clustered | halo | lattice | wave |
|---|---|---|---|---|---|---|
| main's walk, 128³ | 42.6 | 21.9 | 17.0 | 15.5 | 55.1 | 40.8 |
| kNN-first k = 24 | 36.1 (−15 %) | 12.5 (−43 %) | 10.5 (−38 %) | 10.0 (−36 %) | 28.1 (−49 %) | 36.4 (−11 %) |
| fallbacks | 14.3 % | 95.2 % | 95.3 % | 95.3 % | 100 % | 12.7 % |
| kNN-first k = 32 | 37.7 (−12 %) | 12.6 (−43 %) | 7.8 (−54 %) | 6.6 (−58 %) | 49.8 (−10 %) | 38.4 (−6 %) |
| fallbacks | 0 | 79.1 % | 79.7 % | 79.6 % | 0 | 0 |
| kNN-first k = 48 | 33.4 (−22 %) | 14.8 (−33 %) | 11.4 (−33 %) | 10.8 (−31 %) | 41.8 (−24 %) | 33.4 (−18 %) |
| fallbacks | 0 | 31.8 % | 34.2 % | 34.2 % | 0 | 0 |
| k = 32, resume in the warp | 38.0 (−11 %) | 17.3 (−21 %) | 14.3 (−16 %) | 12.1 (−22 %) | 50.2 (−9 %) | 38.5 (−6 %) |
| k = 24, then 48, then the walk | 34.9 (−18 %) | 11.0 (−50 %) | 8.8 (−49 %) | 8.3 (−47 %) | 25.0 (−55 %) | 35.4 (−13 %) |
| k = 32 at the base level, resume | 38.0 (−11 %) | 17.2 (−21 %) | 13.5 (−21 %) | 12.3 (−21 %) | 50.2 (−9 %) | 38.5 (−6 %) |
| k = 32, resume, balls | 37.9 (−11 %) | 16.9 (−23 %) | 14.1 (−17 %) | 12.5 (−20 %) | 48.7 (−12 %) | 38.4 (−6 %) |
| walk with balls (`w8`) | 40.8 (−4 %) | 18.4 (−16 %) | 14.5 (−15 %) | 13.4 (−14 %) | 49.9 (−9 %) | 40.0 (−2 %) |
| main's walk, 256³ | 42.8 | 22.4 | 18.5 | 17.9 | 57.6 | 41.0 |
| kNN-first k = 24 | 36.8 (−14 %) | 12.7 (−43 %) | 11.5 (−38 %) | 11.2 (−37 %) | 29.2 (−49 %) | 37.2 (−9 %) |
| kNN-first k = 32 | 38.4 (−10 %) | 12.8 (−43 %) | 11.5 (−38 %) | 11.1 (−38 %) | 51.9 (−10 %) | 39.0 (−5 %) |
| kNN-first k = 48 | 33.8 (−21 %) | 15.2 (−32 %) | 12.9 (−30 %) | 12.3 (−31 %) | 43.6 (−24 %) | 34.0 (−17 %) |
| k = 32, resume in the warp | 38.5 (−10 %) | 17.7 (−21 %) | 15.3 (−18 %) | 14.1 (−21 %) | 52.9 (−8 %) | 39.3 (−4 %) |
| k = 24, then 48, then the walk | 35.8 (−16 %) | 11.3 (−50 %) | 9.8 (−47 %) | 9.3 (−48 %) | 26.5 (−54 %) | 36.2 (−12 %) |
| k = 32 at the base level, resume | 38.7 (−10 %) | 17.7 (−21 %) | 15.0 (−19 %) | 14.8 (−17 %) | 53.0 (−8 %) | 39.3 (−4 %) |
| k = 32, resume, balls | 38.5 (−10 %) | 17.4 (−22 %) | 15.0 (−19 %) | 14.1 (−21 %) | 51.1 (−11 %) | 38.9 (−5 %) |
| walk with balls (`w8`) | 41.0 (−4 %) | 18.7 (−17 %) | 15.7 (−15 %) | 15.3 (−14 %) | 52.2 (−9 %) | 40.2 (−2 %) |
| the branch's walk, balls off (256³) | 41.1 (−4 %) | 18.8 (−16 %) | 16.2 (−13 %) | 15.8 (−12 %) | 54.0 (−6 %) | 39.3 (−4 %) |

Fallbacks at 256³ are the same shares as at 128³ (k = 24: 14.3, 95.2, 95.2, 95.2, 100, 12.8 %; k = 32: 0, 79.0, 79.2, 79.2, 0, 0 %; k = 48: 0, 31.8, 32.6, 32.8, 0, 0 %). With resume, 26 to 110 cells of 2.1M (235 to 582 of 16.8M) still went to the walk kernel: incomplete lists.

**Where the time goes** at 128³, k = 32 (ms of the kNN pass, the listed clipping and the walk for the fallbacks, from the same job; main's whole kernel for comparison):

| | jittered | uniform | clustered | halo | lattice | wave |
|---|---|---|---|---|---|---|
| main, the walk | 49.2 | 95.8 | 123.4 | 135.3 | 38.1 | 51.4 |
| kNN pass | 15.4 | 20.1 | 34.8 | 33.0 | 8.8 | 9.4 |
| listed clipping | 45.4 | 44.2 | 44.1 | 44.1 | 32.3 | 44.4 |
| walk of the fallbacks | 0.2 | 98.9 | 114.0 | 117.5 | 0.2 | 0.2 |
| with resume: pass, clipping | 10.4, 43.9 | 20.1, 97.6 | 31.7, 110.7 | 33.0, 122.6 | 8.7, 32.0 | 13.9, 53.8 |

**Work per cell** at 128³ (jobs 32319383, 32319476; candidate reads from the measurement build, job 32319475: particles whose distance was computed, the kNN pass included), mean (p99):

| design | jittered: clips, cuts, reads | uniform | clustered | halo | lattice | wave |
|---|---|---|---|---|---|---|
| main's walk | 21.9 (25), 16.0, 76 | 48.0 (92), 17.9, 103 | 50.7 (106), 19.8, 112 | 53.7 (211), 19.4, 115 | 26.0, 6.0, 80 | 26.0, 16.1, 80 |
| kNN-first k = 32 | 21.9 (25), 16.0, 117 | 47.1 (92), 17.7, 181 | 49.2 (106), 19.1, 191 | 52.3 (211), 18.8, 192 | 26.0, 6.0, 112 | 21.8 (25), 16.1, 112 |
| kNN-first k = 48 | 21.9 (25), 16.0, 146 | 44.5 (92), 17.0, 183 | 46.1 (105), 17.7, 197 | 49.5 (211), 17.6, 198 | 26.0, 6.0, 148 | 21.8 (25), 16.1, 148 |
| k = 32, resume | 21.9 (25), 16.0, 117 | 44.3 (89), 16.6, 204 | 45.6 (100), 16.6, 214 | 49.1 (208), 16.6, 215 | 26.0, 6.0, 112 | 21.8 (25), 16.1, 112 |
| k = 32, resume, balls | 16.8 (22), 16.0, 117 | 25.8 (40), 16.6, 204 | 26.0 (40), 16.6, 214 | 25.8 (40), 16.6, 215 | 26.0, 6.0, 112 | 16.8 (22), 16.1, 112 |
| walk with balls | 16.8 (22), 16.0, 76 | 23.8 (41), 17.9, 103 | 25.7 (46), 19.8, 112 | 25.1 (46), 19.4, 115 | 26.0, 6.0, 80 | 17.5 (23), 16.1, 80 |

**Instructions** (Nsight Compute, job 32360847, warp instructions per cell, 128³): main's kernel 8802 jittered, 14020 uniform, 16751 halo; kNN pass (k = 32) 1404, 2764, 3897 (136 registers); listed clipping 8110 jittered and, with resume, 13580 uniform and 15336 halo; the walk with balls 9085, 15368, 18380 (the filter's lane loop over the vertices costs more instructions than the clips it removes); the branch's walk with the balls compiled in but off 9090, 14740, 17548, and 4 to 19 % slower than main at the same 96 registers (51.0 against 48.9 ms jittered, 110.7 against 92.8 uniform): the kernel's instruction-cache limit of the P1 section again.

**Exact start** (`start_ms`, median of 2 × 3, job 32319853), M cells/s: main 18.5, 8.9, 8.0, 10.4, 7.7, 4.4 against k = 32 with resume 18.4, 8.5, 7.4, 9.0, 7.6, 4.4 and the walk with balls 18.1, 8.2, 7.0, 9.3, 7.5, 4.3 (jittered, uniform, clustered, halo, lattice, wave).

**Mesh.** After the build and two steps at 0.1 spacings (job 32319852, 128³) `tets_checksum`, `geometry_checksum`, `faces_checksum` and `lists_checksum` equal main's on all six inputs for the branch's walk, k = 32 with resume (with and without balls) and the walk with balls, with main's bad tetrahedra and patched cells. The rings' topology (`rings_topology`, the bench's ring checksum without geometry bits) is the same in all of them. `rings_checksum` and a plain rebuild's `clip_checksum` are main's for the walk with balls (the filter changes no cut); a kNN order cuts in another order, so the clipped volumes and centroids differ in the last bits (not the mesh's, which come from the tetrahedra).

**What it says.** kNN-first is slower than the walk on every input at both sizes: 4 to 11 % on jittered and wave input in its best form (k = 32, resume), 16 to 22 % on unstructured input. Two reasons. (1) A list certifies a cell only when its last entry lies beyond 2R, as the walk's stop does; k = 32 does that for jittered, lattice and wave input, but on uniform, clustered and halo input 79 % of the cells need more (k = 48: 32 %, k = 64 at 64³: 8 %), so they are clipped twice or resumed. (2) A list clips the same particles as the walk: everything within 2R at its turn, nearest first. Its global order saves 8 to 16 % of the clips (uniform 48.0 to 44.3), and its cuts fall from 17.9 to 16.6 on unstructured input, but the clipping without any walk (45 ms jittered) is already 92 % of main's whole kernel (49 ms), so the gather the list replaces is worth at most 8 % there, less than the kNN pass costs (10 to 15 ms). A per-region choice changes nothing: the octree level does not tell which cells a list certifies. The vertex balls remove half the clips on unstructured input (48 to 24 uniform, 54 to 25 halo) and a quarter on jittered, with every cut and bit the same, but they save no time: a clip that cuts nothing is a side test and a ballot, about as dear as the filter's own test, and the cuts (16 to 20 a cell, about a face each) are the cost. The lower bound of about one cut per face is already met within 1 to 25 % (cuts against 15.9 faces), so no candidate order, list or bound can make the rebuild several times faster; that needs a cheaper cut (C, R16) or no clipping (D).

### Step kernels at today's bytes (2026-11-12)

Branch `step-kernels` against main 5d23b26 (with the re-sort, layout off, and the #48 fix), one A100 at 1305 MHz application clock, double, jittered input, median of steps 2 to 10 over two interleaved runs of each build (scripts `scripts/ab_step_a100.sh`, `scripts/ab_step_tables.py`; logs archived on Raven in `voronoi-cpp-joblogs/voronoi-cpp-step/build-cuda/jobs/`). The target was the bandwidth ceiling of [measure_2026-11-12.md](measure_2026-11-12.md) at today's bytes: 434 M cells/s for the step at 0.03 spacings, 320 with `update_cells()`, 259 at 0.1. Nothing stored changed.

| M cells/s | main | step-kernels | gain | ceiling (share reached) |
|---|---|---|---|---|
| step, 0.03 / 0.1 / 0.3 spacings, 128³ (job 32231679) | 194.0 / 113.9 / 56.4 | 204.5 / 126.4 / 63.6 | +5.4 / +11.0 / +12.8 % | 434 / 259 / – (47 / 49 %) |
| step + `update_cells()`, same | 144.0 / 90.4 / 49.9 | 149.6 / 98.2 / 55.5 | +3.9 / +8.6 / +11.2 % | 320 at 0.03 (47 %) |
| step, 0.03 / 0.1 / 0.3, 256³ (job 32231432) | 264.2 / 177.6 / 93.6 | 280.7 / 194.1 / 107.6 | +6.3 / +9.3 / +15.0 % | |
| step + `update_cells()`, 256³ | 187.3 / 133.2 / 77.6 | 195.5 / 142.2 / 86.9 | +4.4 / +6.8 / +12.0 % | |

On main c505cc6 (before #48) the same changes gave +5.3 / +10.6 / +12.9 % (job 32230977) and +6.8 / +10.4 / +12.8 % at 256³ (job 32230698); on a `gpudev` node that ran slower for both builds +5.8 / +9.8 / +11.7 % (job 32231522). Bits: `geometry_checksum` (cell volumes and centroids) and a numbering-free face checksum (owner, neighbour, area and centroid bits, faces sorted by pair) are identical to main at every step of 0.03, 0.1 and 0.3 spacings, jittered and clustered, 128³ (job 32231523 on 5d23b26, and 32230464, 32230931 on c505cc6; 30 steps each); the start's `cells_checksum` too. The per-step `cells_checksum` hashes face numbers, which follow the repair's slot order (atomics): it differs between two runs of main itself (job 32225953), so it cannot compare builds. Peak device memory is unchanged: 1998.29 B/cell over steps at 0.03 spacings and 2028.37 at 0.3, both builds (job 32231678). A100 `ctest`: 22 of 22 pass (job 32231298).

Each lever on its own, 128³, against the build before it in the same job (step rate at 0.03 / 0.1 spacings):

| lever | job | step | kernel |
|---|---|---|---|
| One wait per flip round: when the work and stuck lists hold 4 failing triangles per listed tetrahedron, `plan` and `apply` run over that bound and their threads past the device count return; the count is read with the apply step's counters. The screen's count read is reused for the repair's first round | 32225960 (Nsight Systems), 32225565 | `update()` wall 11.15 → 10.96 ms (+1.7 %) at 0.03, 19.43 → 19.33 ms (+0.5 %) at 0.1; synchronisations 26.6 → 20.7 and 31.7 → 25.6 | |
| Launch bounds: `faces` at most 80 registers (128 threads, 6 blocks per SM), `screen` at most 64 (128, 8) | 32225763 | +3.8 / +3.8 % | `faces` 2.95 → 2.74 ms, `screen` 2.65 → 2.50 ms at 0.03 |
| Warp-aggregated counters (`tet_mesh::claim`): one atomic per group of threads on a shared counter (dirty lists, work list, stuck list, freed slots, created tetrahedra, ring pool) | 32225953 | −0.1 / +2.6 % | `apply` 3.24 → 2.83 ms at 0.1 |
| `dirty_faces` with a thread per edge of each touched tetrahedron (was a thread per tetrahedron, six walks each) | 32225953 | +1.9 / +4.4 % | `dirty_faces` 0.68 → 0.42 ms at 0.03, 2.38 → 1.22 ms at 0.1 |

Tried and left out:

| lever | job | result |
|---|---|---|
| `find`, `plan`, `apply` capped at 64 or 80 registers | 32225408 | slower: they spill; `find` 0.46 → 0.72 to 0.76 ms at 0.03, the step 3 to 7 % slower. At 128 registers no change |
| `faces` at 64 registers | 32225408 | `faces` 4 % slower (spills) |
| Winners of the vote compacted (a kernel lists them, 2→3 and 3→2 flips apart, in the work list's room) before `apply` | 32225565, 32230610 | +1.0 / +0.9 % alone, but +0.3 / −2.5 % on top of the levers above: `apply` 1.08 → 0.89 ms, the extra kernel 0.27 ms |
| The dirty-face count kept on the device: `append_rings` over 6 per touched tetrahedron, one wait instead of two in `update_face_rings` | 32226509 (Nsight Systems) | `update()` wall 0.8 % (0.03) and 1.8 % (0.1) slower: the wider launch costs more than the wait |

What is left between the step and its ceiling (0.03 spacings, 128³). Nsight Systems (job 32231002, a `gpudev` node slower for both builds): per `update()` 44.4 → 48.0 launches, 26.6 → 20.7 synchronisations, kernel time 11.21 → 10.72 ms, host gap 1.12 → 0.97 ms (at 0.1: 19.97 → 18.29 ms of kernels, gap 1.30 → 1.12 ms). Per kernel, the time now (Kokkos Tools, job 32230610) against its bytes at 80 % of peak ([measure_2026-11-12.md](measure_2026-11-12.md)): `screen` 2.17 against 0.89 ms, `faces` 2.42 against 1.72, `apply` 1.08 against 0.24, `plan` 0.91 against 0.13, `face_numbers` 1.16 against 0.71, `append_rings` 0.71 against 0.35, `find` 0.48 against 0.12, `dirty_faces` 0.40 against 0.36: about 4.8 ms of kernel time and 1 ms of host gaps over the ceiling's 4.8 ms. Why it stays: `screen` and `faces` gather a 32-byte row per corner or ring entry from scattered tetrahedra and are bound by L1/L2 transactions and latency, not by registers any more; the repair's `plan`, `find` and `apply` are one thread per flip with long, divergent, dependent chains (the exact predicates' 128- and 256-bit fallbacks keep `plan` and `find` at 152 to 168 registers, and capping them spills); `face_numbers` is a scan whose result (the face count) the host must read before the next launches, as are the counts of each flip round. The next steps are structural: a warp per flip or per ring walk (K1), fewer bytes per flip (K2), face numbers kept for untouched tetrahedra (E, needs `number_faces`), and the count read-backs replaced by a device-side queue (R1).

### Fewer bytes per step: incremental face numbers (E) and the ring re-walk (K2) (2026-11-12)

Branch `step-bytes` on main 9e9ec03: `cbf2da6` the incremental face numbering (E, [#53](https://github.com/Ulli1991/voronoi-cpp/issues/53)), `d3cc1e2` memory to pay for it, then the ring re-walk (K2) with these docs. Builds: main (base), E (E and memory), K2 (K2 alone on main, not committed: `voronoi-cpp-joblogs/voronoi-cpp-bytes/k2_alone.patch` on Raven), E+K2 (the branch). One A100 at 1305 MHz application clock (the SM clock each job logs is in the table), double, jittered input, median of steps 2 to 10 over two interleaved runs of each build, all builds in one job (`scripts/ab_step_a100.sh`, `scripts/ab_step_tables.py`; per-kernel bytes `scripts/bytes_a100.sh`, `scripts/bytes_tables.py`; logs archived on Raven in `voronoi-cpp-joblogs/voronoi-cpp-bytes/build-cuda/jobs/`). The levers' shares from [measure_2026-11-12.md](measure_2026-11-12.md) were upper bounds: E −380 B/cell (1.15×), K2 1.28× at 0.03 and 1.85× at 0.1.

**What changed.** E: a tetrahedron's faces are `ids(start(t) + rank)` in a pool of numbers; after a repair with at most count/16 created tetrahedra only the touched ones release and take numbers (in place when they have no more faces than before; the other freed numbers go to new faces in slot order by a scan, and faces above the new count move down), the rest keep theirs; with more created tetrahedra every face is numbered in slot order as before, which also restores the order. K2: a dirty face's ring is walked once into a buffer and written in canonical order from it (was: walked twice), into its old room when it is no longer than before (the ring pool compacts about half as often).

| M cells/s, step (with `update_cells()`) | base | E | K2 | E+K2 | E+K2 against base |
|---|---|---|---|---|---|
| 128³, 0.03 spacings (job 32269165) | 205.5 (150.3) | 221.8 (160.7) | 205.6 (150.3) | 222.7 (161.1) | +8.4 % (+7.2 %) |
| 128³, 0.1 | 126.2 (98.0) | 126.1 (97.1) | 129.7 (100.1) | 129.7 (99.3) | +2.8 % (+1.3 %) |
| 128³, 0.3 | 63.6 (55.5) | 62.6 (54.6) | 66.2 (57.4) | 65.9 (57.2) | +3.5 % (+3.1 %) |
| 256³, 0.03 spacings (job 32237516) | 280.7 (195.4) | 313.6 (212.9) | 284.0 (197.0) | 318.2 (214.9) | +13.4 % (+10.0 %) |
| 256³, 0.1 | 194.8 (142.6) | 203.5 (148.4) | 200.1 (145.3) | 208.8 (151.2) | +7.2 % (+6.0 %) |
| 256³, 0.3 | 108.0 (87.0) | 104.2 (81.7) | 113.6 (90.6) | 109.5 (85.6) | +1.5 % (−1.6 %) |
| 500 steps at 0.1, Hilbert re-sort every 50, 128³ (job 32238599): step, step + cells, all with re-sorts | 148.6, 113.3, 111.9 | 150.9, 112.8, 111.1 | | 154.3, 114.8, 112.3 | +3.8 %, +1.3 %, +0.4 % |

E gains most where few tetrahedra change (0.03 spacings, and 256³, where the flow shears half as much per step). At 0.3 it costs at 256³ and is neutral at 128³: once the flips fall below count/16, and the faces that take new numbers lose the slot order (`faces` and `cell_sums` read them scattered), and every face lookup goes through `ids` (`dirty_faces` +15 % at 0.1). K2 gains 0 to 5.2 % on its own (most at large spacings), mostly from `append_rings` (−23 % time at 0.1) and fewer ring compactions.

DRAM bytes per cell of step 3, Nsight Compute, 128³, caches flushed (job 32237298; the E and E+K2 binaries then had the same kernels as the final ones):

| B/cell, `update()` (`update_cells()`) | base | E | K2 | E+K2 |
|---|---|---|---|---|
| 0.03 spacings | 2717 (964) | 2398 (916) | 2707 (964) | 2392 (913) |
| 0.1 spacings | 4292 (1083) | 4180 (1133) | 4239 (1083) | 4133 (1132) |
| `face_numbers` / its replacement at 0.03 | 422 | 63 (`face_touched` 14, `face_offsets` 11, `face_release` 20, `face_assign` 13, `face_numbered` 5) | 421 | 63 |
| repair and ring re-walk (`find`, `plan`, `apply`, `dirty_faces`, `append_rings`), per flip at 0.03 / 0.1 | 20.4 / 21.1 kB | 21.2 / 22.1 kB | 20.3 / 20.8 kB | 21.3 / 21.8 kB |

E removes 359 B/cell of numbering at 0.03 and 263 at 0.1, but gives back 40 (0.03) and 151 (0.1) in `faces`, `dirty_faces` and `cell_sums`: net −319 and −112 B/cell. K2 cuts the re-walk's bytes per flip by 2 % only: one walk instead of two halves the rows read in `append_rings`, but those rows mostly hit L2 already (the second walk re-reads what the first just read); the DRAM bytes per flip stay about 21 kB against a target of 3 kB. What is left there is the repair's rounds (`find`, `plan`, `apply`: 9.7 kB per flip at 0.1, unchanged) and the partial walks of `dirty_faces` from every touched tetrahedron.

Bits (jobs 32238014, 32238070, 32238146 jittered, 32238147, 32238148, 32238322 clustered; 30 steps each at 0.03, 0.1 and 0.3 spacings, 128³): `geometry_checksum`, `faces_checksum` and the new `lists_checksum` (each cell's list as other particles and owner bits) identical to main at every step for E and E+K2; `cells_checksum` hashes face numbers and differs, as it did between two runs of main. With faces by owner, the owner bit and a re-sort every 2 steps (job 32238180, 262,144 cells, 6 steps) the same checksums equal main's; `--verify` against a fresh rebuild passes at 0.3 spacings. A100 `ctest`: 22 of 22 pass (job 32238334). CI's host steps (gcc 15, sanitizers off and on, clang-tidy, ctest; host Kokkos 5.0.99 with OpenMP and 2 threads, where CI builds a Serial Kokkos 5.0.2): 22 of 22 pass with sanitizers off and on (jobs 32238509, 32238627); clang-tidy 21 stops on `mesh_fuzz_test.cpp`, unchanged here and on main ([#56](https://github.com/Ulli1991/voronoi-cpp/issues/56)). In one clustered 0.3 run E+K2 left 6 triangles stuck at step 2, which the host finish repaired (the repair's flip order follows atomics; clustered 0.3 sticks in some steps on main too, [measure_2026-11-12.md](measure_2026-11-12.md)); its checksums equal main's.

Memory: peak over 10 steps, 128³ (job 32238100): 1998.29 → 1979.05 B/cell at 0.03 spacings and 2028.38 → 2010.04 at 0.3 for E and E+K2; K2 alone is unchanged. E adds `face_rings_ids` (41 B/cell, a quarter spare) and `face_rings_numbered` (7.8); the repair's free list sized to its use (31 → under 2 B/cell) and the cell list freed after a build (37 B/cell, absent at the step's peak) pay for it.

luna-cpp (branch `resort` a82092c, the first that builds against main's API without gradient weights), moving Sedov 128³, 50 steps, two runs of each build (jobs 32238510, 32238511): the final `Q`, `x` and `volume` hashes equal those on main's voronoi-cpp, with the default layout and with faces by owner, the owner bit and a re-sort every 10 steps. luna-cpp reads `cell_face` lists, `area`, `face_centroid`, `owner` and `neighbour` as before; only the face numbers differ (see below). Mesh throughput there 274 → 311 M cells/s (+13.6 %), with faces by owner 198.8 → 221.7 (+11.5 %); the whole hydro step 71.4 → 73.7 M cells/s (+3.2 %) and 70.5 → 73.1 (+3.7 %).

**Face numbering, for callers.** As before, faces are numbered 0 .. faces() − 1, `owner(f) < neighbour(f)`, each cell's list is sorted by the other particle, and with `Layout::owner_faces` the numbers go by owner. What changed: without the owner layout a face's number no longer follows the slot order of the tetrahedra after every update; it stays the same from update to update while the tetrahedron representing its edge is not touched by a flip, and a new face takes a freed number. Numbers were never stable across updates before; a caller that keyed data by face number across an update (luna-cpp does not) now sees most numbers kept, not all.


**Left to the 5× budget.** The step now moves 2392 B/cell at 0.03 spacings and 4133 at 0.1 (step 3, 128³; main 69de1d3 moved 2866 and 4795, main 9e9ec03 2717 and 4292). The 5× budgets are 1269 and 2182 B/cell: 1123 and 1951 B/cell still to go, with every kernel at 80 % of peak besides. By kernel at 0.03: `faces` 1005 and `cell_sums` 737 (R14 and float centres, R6), `screen` 528 (K4), the repair and ring re-walk 617 (the repair's rounds and the `dirty_faces` walks, K2), the numbering now 63.

### CPU and GPU together: where the GPU waits (2026-11-15)

Phase 1 of "overlapped CPU tail" ([ideas.md](ideas.md#cpu-and-gpu-together-2026-11-15)): measured only, no library change. Branch `cpu-overlap` on main 04eb9e4; one A100 40 GB on `gpu1` (18 of the node's 72 cores), 1305 MHz SM clock under load in every job; double precision, periodic unit box. Nsight Systems 2026 traces CUDA and the bench's NVTX phase ranges (`--nvtx=phases`, fenced at both ends, no per-kernel callbacks); `scripts/overlap_tables.py` splits each phase into GPU busy (the union of kernels, copies and memsets) and idle, and each idle interval by what the issuing host thread did meanwhile. Tail counts come from the measurement build (`-DVORONOI_MEASURE`, `warp_status`). Jobs 32373756 (128³), 32373758 and 32373759 (256³), 32373757 (surface 128³), 32373991 (volumes only); scripts `scripts/overlap_a100.sh`, `scripts/overlap_volumes_a100.sh`. The step's numbers include the bench's own fences (`Timings`, one per phase): three or four of the read-backs are those.

Rebuild (warp clipping, median of 3):

| input | N | clipping ms | GPU idle ms (%) | tail cells | tail pass ms (% of clipping) |
|---|---|---|---|---|---|
| jittered | 128³ | 49.3 | 0.32 (0.6 %) | 0 | 0.07 (0.1 %) |
| uniform | 128³ | 95.7 | 0.32 (0.3 %) | 33 | 2.54 (2.7 %) |
| clustered | 128³ | 123.1 | 0.41 (0.3 %) | 68 | 6.28 (5.1 %) |
| halo | 128³ | 135.2 | 0.32 (0.2 %) | 261 | 15.63 (11.6 %) |
| cosmicweb | 128³ | 161.5 | 0.32 (0.2 %) | 976 | 17.11 (10.6 %) |
| jittered | 256³ | 392.3 | 0.91 (0.2 %) | 0 | 0.50 (0.1 %) |
| uniform | 256³ | 748.3 | 0.57 (0.1 %) | 306 | 5.65 (0.8 %) |
| clustered | 256³ | 904.6 | 0.36 (0.0 %) | 315 | 6.31 (0.7 %) |
| halo | 256³ | 937.9 | 0.47 (0.1 %) | 1235 | 35.52 (3.8 %) |
| cosmicweb | 256³ | 1308.3 | 0.56 (0.0 %) | 8501 | 113.65 (8.7 %) |

Exact start (Mesh::update from scratch, one run):

| input | N | start ms | GPU idle ms (%) | of which allocation | read-back trips ms | tail pass ms | exact re-clipped cells | warp kernel ends, first check ends (ms) |
|---|---|---|---|---|---|---|---|---|
| jittered | 128³ | 140.4 | 12.6 (9.0 %) | 9.0 | 0.95 | 0.07 | 0 | 60, 109 |
| uniform | 128³ | 261.6 | 16.7 (6.4 %) | 12.6 | 1.34 | 15.43 | 129 | 118, 168 |
| clustered | 128³ | 303.6 | 31.2 (10.3 %) | 26.9 | 1.25 | 23.32 | 134 | 149, 204 |
| halo | 128³ | 269.8 | 52.7 (19.5 %) | 49.1 | 1.06 | 14.95 | 0 | 137, 202 |
| cosmicweb | 128³ | 343.7 | 16.6 (4.8 %) | 12.6 | 1.46 | 52.91 | 108 | 166, 231 |
| jittered | 256³ | 2046.3 | 629.4 (30.8 %) | 623.7 | 6.99 | 3.12 | 830 | 462, 853 |
| uniform | 256³ | 2422.9 | 525.9 (21.7 %) | 520.3 | 18.60 | 16.46 | 277 | 928, 1349 |
| clustered | 256³ | 2600.0 | 639.3 (24.6 %) | 633.3 | 10.74 | 22.30 | 203 | 1019, 1426 |
| halo | 256³ | 2524.6 | 487.6 (19.3 %) | 482.2 | 5.10 | 67.34 | 393 | 1019, 1432 |
| cosmicweb | 256³ | 2974.2 | 541.4 (18.2 %) | 534.9 | 18.21 | 154.37 | 156 | 1362, 1876 |

Moving step (update + update_cells, median of steps 2 to the last):

| input | N | spacings | step ms | GPU idle ms (%) | read-backs | trips ms | launches | host code ms | stuck, host-fallback ms |
|---|---|---|---|---|---|---|---|---|---|
| jittered | 128³ | 0.03 | 13.06 | 0.62 (4.7 %) | 13 | 0.22 | 67 | 0.17 | 0, 0.0 |
| jittered | 128³ | 0.1 | 21.15 | 0.73 (3.5 %) | 17 | 0.28 | 74 | 0.21 | 0, 0.0 |
| jittered | 128³ | 0.3 | 37.81 | 0.89 (2.4 %) | 19 | 0.35 | 82 | 0.24 | 0, 0.0 |
| uniform | 128³ | 0.03 | 11.02 | 0.56 (5.1 %) | 11 | 0.19 | 60 | 0.16 | 0, 0.0 |
| uniform | 128³ | 0.1 | 14.75 | 0.68 (4.6 %) | 15 | 0.25 | 71 | 0.19 | 0, 0.0 |
| uniform | 128³ | 0.3 | 24.21 | 0.81 (3.4 %) | 19 | 0.31 | 84 | 0.22 | 0, 0.0 |
| clustered | 128³ | 0.03 | 10.70 | 0.53 (5.0 %) | 10 | 0.18 | 57 | 0.16 | 0, 0.0 |
| clustered | 128³ | 0.1 | 14.25 | 0.68 (4.8 %) | 15 | 0.25 | 70 | 0.19 | 0, 0.0 |
| clustered | 128³ | 0.3 | 23.03 | 0.80 (3.5 %) | 19 | 0.31 | 82 | 0.23 | 0, 0.0 |
| halo | 128³ | 0.03 | 10.92 | 0.53 (4.9 %) | 10 | 0.18 | 57 | 0.15 | 0, 0.0 |
| halo | 128³ | 0.1 | 14.50 | 0.70 (4.9 %) | 15 | 0.25 | 72 | 0.19 | 0, 0.0 |
| halo | 128³ | 0.3 | 23.95 | 0.85 (3.6 %) | 20 | 0.33 | 86 | 0.24 | 0, 0.0 |
| cosmicweb | 128³ | 0.03 | 10.67 | 0.52 (4.9 %) | 10 | 0.18 | 56 | 0.16 | 0, 0.0 |
| cosmicweb | 128³ | 0.1 | 14.18 | 0.68 (4.8 %) | 15 | 0.25 | 71 | 0.19 | 0, 0.0 |
| cosmicweb | 128³ | 0.3 | 22.91 | 0.81 (3.5 %) | 18 | 0.32 | 81 | 0.22 | 0, 0.0 |
| jittered | 256³ | 0.03 | 77.42 | 0.67 (0.9 %) | 11 | 0.23 | 62 | 0.19 | 0, 0.0 |
| jittered | 256³ | 0.1 | 109.45 | 0.80 (0.7 %) | 15 | 0.28 | 72 | 0.22 | 0, 0.0 |
| jittered | 256³ | 0.3 | 193.62 | 1.09 (0.6 %) | 20 | 0.42 | 89 | 0.29 | 0, 0.0 |
| uniform | 256³ | 0.03 | 69.74 | 0.57 (0.8 %) | 10 | 0.19 | 57 | 0.16 | 0, 0.0 |
| uniform | 256³ | 0.1 | 83.66 | 0.76 (0.9 %) | 14 | 0.29 | 70 | 0.21 | 0, 0.0 |
| uniform | 256³ | 0.3 | 121.65 | 1.10 (0.9 %) | 17 | 0.41 | 81 | 0.30 | 0, 0.0 |
| clustered | 256³ | 0.03 | 69.00 | 0.58 (0.8 %) | 10 | 0.19 | 57 | 0.17 | 0, 0.0 |
| clustered | 256³ | 0.1 | 82.08 | 0.69 (0.8 %) | 13 | 0.26 | 67 | 0.19 | 0, 0.0 |
| clustered | 256³ | 0.3 | 117.51 | 0.79 (0.7 %) | 17 | 0.29 | 82 | 0.21 | 0, 0.0 |
| halo | 256³ | 0.03 | 69.30 | 0.54 (0.8 %) | 10 | 0.18 | 57 | 0.15 | 0, 0.0 |
| halo | 256³ | 0.1 | 83.40 | 0.70 (0.8 %) | 13 | 0.26 | 67 | 0.19 | 0, 0.0 |
| halo | 256³ | 0.3 | 121.14 | 0.82 (0.7 %) | 17 | 0.31 | 81 | 0.22 | 0, 0.0 |
| cosmicweb | 256³ | 0.03 | 68.46 | 0.57 (0.8 %) | 10 | 0.19 | 57 | 0.16 | 0, 0.0 |
| cosmicweb | 256³ | 0.1 | 81.32 | 0.69 (0.8 %) | 13 | 0.25 | 67 | 0.20 | 0, 0.0 |
| cosmicweb | 256³ | 0.3 | 117.64 | 0.87 (0.7 %) | 17 | 0.35 | 82 | 0.24 | 0, 0.0 |

What the numbers say:

- **The host does nothing while kernels run.** In every phase it spends the kernels blocked in a synchronising call; host code outside the CUDA API is 0.2 ms per step and under 1.5 ms per start.
- **Rebuild:** the GPU never waits (idle under 1 %). The tail is the per-thread pass over the cells the warp kernel overflows: up to 11.6 % of the clipping at 128³ (halo, cosmic web) and 8.7 % at 256³ (cosmic web, 8501 cells). A tail cell is known on the device when its warp finishes it, on the host only after the warp kernel.
- **Exact start:** the GPU waits 5 to 20 % of the start at 128³ and 18 to 31 % at 256³, nearly all of it in synchronous `cudaMalloc` and `cudaFree` (81 and 43 calls; at 256³ three single `cudaMalloc` calls of 100 to 210 ms after the assembly). Read-backs cost about 1 ms at 128³ (6 round trips), 5 to 19 ms at 256³. The cells for exact re-clipping are known only after the first check (60 to 65 ms after the warp kernel at 128³, about 400 ms at 256³).
- **Moving step:** the GPU waits 2.4 to 5.1 % at 128³ and under 1 % at 256³ (0.5 to 1.1 ms a step): 10 to 20 read-backs (counters of the repair rounds, face counts, the volume check) of 0.02 ms each and the launch gaps between them. In none of the 240 steps (30 runs of 8) did a triangle stick, so no host fallback ran.
- **Copies (pageable, as the benches do them):** host to device 2.2 to 2.7 GB/s in library_bench and 10.7 GB/s in voronoi_bench, device to host 4.3 GB/s.

Volumes only (`library_bench build --path=warp-host-vc`: points in host memory to volumes and centroids in host memory; medians of the timed builds, ms):

| input | N | build | kernels | points in | volumes and centroids out | GPU idle |
|---|---|---|---|---|---|---|
| jittered | 128³ | 96.5 | 49.2 | 19.4 | 25.0 | 3.3 (3.4 %) |
| uniform | 128³ | 136.8 | 95.0 | 24.9 | 14.2 | 2.8 (2.0 %) |
| clustered | 128³ | 165.0 | 122.4 | 19.8 | 14.3 | 2.9 (1.7 %) |
| halo | 128³ | 176.2 | 135.0 | 24.9 | 13.3 | 2.7 (1.5 %) |
| cosmic web | 128³ | 201.2 | 159.8 | 24.0 | 14.1 | 2.7 (1.4 %) |
| jittered | 256³ | 719.2 | 392.5 | 188.3 | 132.0 | 6.6 (0.9 %) |
| uniform | 256³ | 1070.7 | 746.1 | 188.5 | 129.5 | 6.7 (0.6 %) |

The copies are 20 to 46 % of this path: 50 MB of points in at 2.6 GB/s and 67 MB out, from and to pageable host memory. (The GPU counts as busy during a copy, so the idle column misses that the copies and the clipping run one after the other.)

Bulk cells on the CPU: the same code on a full 72-core node clips 2.6 M cells/s against 33.7 on the A100 (13×) and steps 8.5 against 144 M cells/s (17×) (README, library benchmark). Giving the CPU its share of bulk cells could save at most 7 % of a rebuild, and copying the positions over alone costs 4.7 ms at 128³ (50 MB at 10.7 GB/s), about 10 % of a jittered clipping; a step would need the mesh (about 1950 B/cell) mirrored on the host every step. So no candidate splits bulk cells.

### CPU and GPU from one build (2026-11-16)

S0 and S2 of [ideas.md](ideas.md#cpu-and-gpu-together-2026-11-15): one build on Kokkos with CUDA 13.0 and OpenMP (`/u/uli/kokkos-cuda-omp`), and `voronoi::Engine` choosing cpu, gpu or hybrid at run time. One A100 40 GB on `gpu1`, 18 host threads; jobs 32378021 (build cost, CI host steps, first speed run), 32407747 (speed with `--fmad=false`, bits), 32410447 (fixed `modes_test`, bits, second speed run).

Build cost, `voronoi_bench` at -j1: 143 s in both builds; binary 8,905,800 bytes (CUDA only) against 8,947,864 (+0.5 %). `mesh_test`: 99 s in both; 6,058,536 against 6,100,688 bytes. CI's host steps (gcc 15 OpenMP build, sanitizers off and on): 23 of 23; clang-tidy on `modes_test` (with `voronoi_engine.hpp`): 0 warnings. A100 ctest after the Engine fix: 23 of 23 in both builds.

Speed of the two-backend build against the CUDA-only one (job 32407747; three interleaved rounds, median; rebuild = warp clipping median of 5, start = exact start, steps = `update` + `update_cells` median of steps 2 to 10 (2 to 6 at 256³)): all 28 cases within 1 %, geometric mean 0.9996, range 0.993 to 1.009.

| input, N | rebuild | start 0.03 / 0.3 | steps 0.03 / 0.3 |
|---|---|---|---|
| jittered 128³ | 1.000 | 0.997 / 1.001 | 0.993 / 1.009 |
| uniform 128³ | 1.001 | 0.998 / 1.000 | 1.002 / 0.996 |
| clustered 128³ | 1.000 | 0.999 / 1.000 | 1.005 / 1.000 |
| halo 128³ | 1.002 | 0.995 / 1.001 | 1.003 / 1.000 |
| cosmicweb 128³ | 0.999 | 0.997 / 0.995 | 0.999 / 1.000 |
| jittered 256³ | 1.000 | 0.999 / – | 0.999 / – |

Time ratios, two backends / CUDA only. Two more runs (jobs 32378021 and 32410447) agree for rebuild (0.998 to 1.005) and steps (0.996 to 1.010). The exact start is one sample per round: in those two runs single rounds of either build were 10 to 25 % (once 2×) slower than the others, so their start medians range 0.86 to 1.11; only 32407747, where the rounds agree to 1 %, resolves the start to 1 %. A start claim at 1 % needs repeats per round. The bench's host advection is slower in the two-backend build (260 against 30 ms at 128³: OpenMP binds the main thread to one core, and the bench's 16 `std::thread`s inherit it); it is outside every timed phase. Tetrahedron counts differ from run to run on CUDA in both builds (flip scheduling), faces never.

Unfused device arithmetic (`-DVORONOI_FMAD=OFF`, `--fmad=false`), same job, against the default two-backend build:

| input, N | rebuild | start 0.03 / 0.3 | steps 0.03 / 0.3 |
|---|---|---|---|
| jittered 128³ | 1.046 | 1.021 / 1.020 | 1.001 / 0.995 |
| uniform 128³ | 1.107 | 1.042 / 1.041 | 0.989 / 1.009 |
| clustered 128³ | 1.086 | 1.036 / 1.038 | 0.990 / 0.993 |
| halo 128³ | 1.077 | 1.052 / 1.053 | 0.985 / 1.003 |
| cosmicweb 128³ | 1.113 | 1.053 / 1.055 | 0.993 / 1.000 |
| jittered 256³ | 1.046 | 1.014 / – | 0.994 / – |

The cost is in the clipping (rebuild +4.6 to +11.3 %, start +1.4 to +5.5 %); steps are unchanged (−1.5 to +0.9 %). What it buys: CPU and GPU meshes equal in every bit (TESTING.md, "Bits across backends").

Host against device clipping (`backends_probe`, 128³, all cells on 18 threads with the per-thread kernel, against the A100's warp kernel): 22.6 to 28.7× slower; the tail cells alone cost 118 to 275 µs per cell per core.

### Unfused mesh geometry: the same bits on CUDA and the host (2026-11-16, job 32413666)

Branch `unfused-on-main` (main 8cb1475 plus the change) against main 8cb1475, one A100 40 GB on `gpu1`, 1410 MHz max SM clock. The geometry kernels round every multiply and add on their own (`tet_mesh::rounded`; TESTING.md, "Bits across backends"); the clipping is unchanged.

Bits, per cell after a build and 3 steps of 0.1 spacings at 128³ (`voronoi_bench step --dump-cells`; volume and centroid of every particle, and a hash of its neighbours):

| input | CUDA vs OpenMP (unfused) | CUDA, main vs unfused: volumes / centroids changed | largest relative volume change | OpenMP, main vs unfused |
|---|---|---|---|---|
| uniform | 0 cells differ | 448,132 / 2,037,882 of 2,097,152 | 7.9e-16 | 0 |
| clustered | 0 | 448,662 / 2,037,347 | 6.9e-16 | 0 |
| halo | 0 | 449,762 / 2,036,167 | 6.9e-16 | 0 |
| cosmicweb | 0 | 448,599 / 2,020,103 | 6.9e-16 | 0 |
| jittered | 0 | 402,791 / 2,095,384 | 8.0e-16 | 0 |

Neighbours are the same in every comparison. main's CUDA differs from main's OpenMP exactly as from the unfused CUDA: the unfused device now gives the host's bits. Accuracy against a long-double reference (every 16th cell, `backends_probe` on `cpu-overlap`): unchanged, median relative volume error 1.28e-15 (uniform), 2.61e-15 (clustered), 2.55e-15 (halo), 1.55e-15 (cosmicweb), at most 2.0e-14 to 2.8e-13.

Speed, three interleaved rounds, medians (steps: `update` + `update_cells`, steps 2 to 10, 2 to 6 at 256³):

| input, N | rebuild | steps 0.03 | steps 0.1 | steps 0.3 |
|---|---|---|---|---|
| jittered 128³ | 1.000 | 1.032 | 1.022 | 1.015 |
| uniform 128³ | 1.000 | 1.042 | 1.033 | 1.022 |
| clustered 128³ | 1.000 | 1.041 | 1.030 | 1.014 |
| halo 128³ | 1.003 | 1.042 | 1.030 | 1.013 |
| cosmicweb 128³ | 0.998 | 1.044 | 1.031 | 1.021 |
| jittered 256³ | 1.000 | 1.044 | 1.029 | 1.014 |
| cosmicweb 256³ | 1.000 | 1.049 | 1.043 | 1.027 |

Time ratios, unfused / main; per-round ratios agree to 1 % for rebuild and steps. Where it goes: the cell sums (`update_cells`) +5 to +11 %, centres and faces +0.5 to +4 %, flips unchanged. Exact start: its clipping and assembly unchanged (build_ms), geometry +0.6 to +1.1 % and cells +5 to +6 %, together about +0.4 ms of 110 to 265 ms at 128³ and +3 ms of 1.6 to 2.5 s at 256³; the start totals are one sample per round and vary by up to 20 % from round to round in both builds.

## Removed, and where to find it

Scripts and files nothing maintained uses; each line names the last commit that had it (`git show <commit>:<path>`).

- `scripts/flow_table.py`, tables of `distortion_test --print` logs for research_notes/mesh_tests_2026-11-05.md: 61e9914.
- `reports/hydro_stage3_blast128.png`, `hydro_stage3_sedov128.png`, `hydro_stage3_sedov256.png`, hydro figures from before the split that nothing referenced: 61e9914.
