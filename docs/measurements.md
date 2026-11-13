# Measurements: provenance

Where each number in the [README](../README.md) comes from: jobs, commits, clocks and the setup details left out of the README. Sections follow the README headings.

## Measured performance

One A100 40 GB (Raven gpu1), double precision, periodic unit box. Rates are million cells per second, cells processed per second of GPU time.

| Rows in the README table | Job | Commit |
|---|---|---|
| Moving-mesh update, update plus `update_cells()`, 128³ and 256³ | 32269165, 32237516 (interleaved with main 9e9ec03; [fewer bytes per step](#fewer-bytes-per-step-incremental-face-numbers-e-and-the-ring-re-walk-k2-2026-11-12)) | `step-bytes` (the ring re-walk commit) |
| `update_cells()` alone | 32231679 ([step kernels](#step-kernels-at-todays-bytes-2026-11-12)) | `step-kernels` |
| Full rebuilds at 2.1M and 16.8M, exact start at 2.1M | 32238284, 32269162, 32269154 ([lean cut](#canonical-rings-and-a-lean-cut-2026-11-13)) | `clip-canonical` |
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

Profiles large / huge / giant: 432 / 56 / 116 cases. Every box family (cube, pow2, 123, ratio, ratiolong, hcp, long, rod, slab, thin, extreme) and generator (lattices sc, bcc, fcc, hcp, diamond; uniform, Poisson disc, quasi; jitter; Gaussian, power-law, shell, Fibonacci, filament, sheet, spiral, cusp clouds; density steps, voids, points on box faces, near and doubled pairs, coplanar and cospherical sets) had at least 22 cases. The 3 WRONG are all #48 (seeds 3000000000159, 3000000000237, 7000000000041); with its fix (edges represented by their lowest tetrahedron) 3000000000159 and 7000000000041 are exact and 3000000000237 is refused for 66 coincident particles, as the input has them. The cases out of time are #49: four rerun alone were still in their first `update()` after 830 s; two rerun on 36 CPU threads refused for reach after 353 and 713 s. The generator `quasi` puts every point on one plane (1/g² + 1/g³ = 1 for the plastic number g), so its cases test only that.

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

### Canonical rings and a lean cut (2026-11-13)

Branch `clip-canonical` against main 9e9ec03 (with the bench's checksums only, `ref`), one A100, every comparison in one job. Script `scripts/clip_canonical_a100.sh`, tables by `scripts/clip_canonical_tables.py` and `scripts/clip_canonical_ncu.py`; logs in `build-jobs/` of the `voronoi-cpp-canon` worktree on Raven. `canon` is commit a53ed48 (canonical rings only), `final` the branch head.

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

**What it says.** The bookkeeping that slots and cut order forced was worth 10 to 15 %, not several times: the cut that is left is mostly the work of finding and building the new vertices, and the kernel still waits on that chain's latency and, on uniform and clustered input, on instruction fetch. A rebuild several times faster needs fewer clips (B, the rounds of `clip-split`) or no clipping (D).

### Step kernels at today's bytes (2026-11-12)

Branch `step-kernels` against main 5d23b26 (with the re-sort, layout off, and the #48 fix), one A100 at 1305 MHz application clock, double, jittered input, median of steps 2 to 10 over two interleaved runs of each build (scripts `scripts/ab_step_a100.sh`, `scripts/ab_step_tables.py`; logs in `build-cuda/jobs` of the `voronoi-cpp-step` worktree on Raven). The target was the bandwidth ceiling of [measure_2026-11-12.md](measure_2026-11-12.md) at today's bytes: 434 M cells/s for the step at 0.03 spacings, 320 with `update_cells()`, 259 at 0.1. Nothing stored changed.

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

Branch `step-bytes` on main 9e9ec03: `cbf2da6` the incremental face numbering (E, [#53](https://github.com/Ulli1991/voronoi-cpp/issues/53)), `d3cc1e2` memory to pay for it, then the ring re-walk (K2) with these docs. Builds: main (base), E (E and memory), K2 (K2 alone on main, not committed: worktree `voronoi-cpp-bytes-k2`), E+K2 (the branch). One A100 at 1305 MHz application clock (the SM clock each job logs is in the table), double, jittered input, median of steps 2 to 10 over two interleaved runs of each build, all builds in one job (`scripts/ab_step_a100.sh`, `scripts/ab_step_tables.py`; per-kernel bytes `scripts/bytes_a100.sh`, `scripts/bytes_tables.py`; logs in `build-cuda/jobs` of the `voronoi-cpp-bytes` worktree on Raven). The levers' shares from [measure_2026-11-12.md](measure_2026-11-12.md) were upper bounds: E −380 B/cell (1.15×), K2 1.28× at 0.03 and 1.85× at 0.1.

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

## Removed, and where to find it

Scripts and files nothing maintained uses; each line names the last commit that had it (`git show <commit>:<path>`).

- `scripts/flow_table.py`, tables of `distortion_test --print` logs for research_notes/mesh_tests_2026-11-05.md: 61e9914.
- `reports/hydro_stage3_blast128.png`, `hydro_stage3_sedov128.png`, `hydro_stage3_sedov256.png`, hydro figures from before the split that nothing referenced: 61e9914.
