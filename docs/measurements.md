# Measurements: provenance

Where each number in the [README](../README.md) comes from: jobs, commits, clocks and the setup details left out of the README. Sections follow the README headings.

## Measured performance

One A100 40 GB (Raven gpu1), double precision, periodic unit box. Rates are million cells per second, cells processed per second of GPU time.

| Rows in the README table | Job | Commit |
|---|---|---|
| Full rebuilds, moving-mesh update, `update_cells()`, update plus `update_cells()` at 2.1M | 31952136 (interleaved runs of main and this tree) | main 16e7769 |
| Exact start, the sum row, at 2.1M | 31701509 | 2311937 |
| The 256³ column | 31694457 | f32c909 (older than the current code, which is faster there) |
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

### Non-cubic box

Job 31702953 (research_notes/box3_2026-10-31.md). Kelvin-Helmholtz-like slab, Lx = Ly = 2, Lz = 16 spacings: `--box=2,2,0.0625` with 512 x 512 x 16 = 4.2M cells, and `--box=2,2,0.03125` with 1024 x 1024 x 16 = 16.8M cells.

| M cells/s | 4.2M | 16.8M |
|---|---|---|
| Warp rebuild, jittered | 39.9 | 39.9 |
| Warp rebuild, uniform | 19.9 | 20.0 |
| Exact start | 11.6 | 12.0 |
| Steps at 0.03 spacings | 105 | 108 |

The bench flow, stretched with the box, shears across the 16-cell thickness: 4x the flips per cell of the cube. Peak 2192 B/cell at 16.8M. The cubic case is bit-identical to the cubic-only code and within 0.3 to 0.7 % of its rebuild speed.

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

Profiles large / huge / giant: 432 / 56 / 116 cases. Every box family (cube, pow2, 123, ratio, ratiolong, hcp, long, rod, slab, thin, extreme) and generator (lattices sc, bcc, fcc, hcp, diamond; uniform, Poisson disc, quasi; jitter; Gaussian, power-law, shell, Fibonacci, filament, sheet, spiral, cusp clouds; density steps, voids, points on box faces, near and doubled pairs, coplanar and cospherical sets) had at least 22 cases. The 3 WRONG are all #48 (seeds 3000000000159, 3000000000237, 7000000000041). The cases out of time are #49: four rerun alone were still in their first `update()` after 830 s; two rerun on 36 CPU threads refused for reach after 353 and 713 s. The generator `quasi` puts every point on one plane (1/g² + 1/g³ = 1 for the plastic number g), so its cases test only that.

### Per-vertex balls (P1) and directional boxes (R8) in the clipping (2026-11-12)

Branch `clip-bounds`; the bound with the measurement build (`-DVORONOI_MEASURE`, counters `rounds` to `clips_r8`), timings with the plain build, all on one A100 in jobs on `gpu1` and `gpudev`, 2,097,152 cells. Script `scripts/clip_bounds_a100.sh`; logs in `jobs/` of the `voronoi-cpp-clip` worktree on Raven.

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

## Removed, and where to find it

Scripts and files nothing maintained uses; each line names the last commit that had it (`git show <commit>:<path>`).

- `scripts/flow_table.py`, tables of `distortion_test --print` logs for research_notes/mesh_tests_2026-11-05.md: 61e9914.
- `reports/hydro_stage3_blast128.png`, `hydro_stage3_sedov128.png`, `hydro_stage3_sedov256.png`, hydro figures from before the split that nothing referenced: 61e9914.
