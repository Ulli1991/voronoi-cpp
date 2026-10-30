# Adaptive cell list for the full rebuild (2026-10-30)

Goal: make the full Voronoi rebuild fast on clustered input (what gravity
produces) without slowing near-uniform input, on an A100 40 GB in double
precision, with exact results.

All numbers: one A100 40 GB (Raven gpu1), 2,097,152 cells, periodic unit box,
`voronoi_bench rebuild <n> <kind> warp-double 5`, main and the branch
interleaved in one job (logs in `/raven/u/uli/voronoi-cpp/build-adaptive/logs/`).

## Inputs

- `jittered`, `uniform`, `clustered` as before. `clustered` is 8 Gaussian
  blobs (sigma 0.04) over a 20 % uniform background: peak density ~100x the
  mean, ~500x the background.
- New `halo`: four halos with density 1 / (r^2 + c^2) out to r = 0.2,
  c = 0.002 (core to edge 1:10^4) over a 10 % uniform background, so ~1:10^5
  from the voids to the cores. With 2.1M points a base cell (one mean spacing)
  in a core holds up to ~2x10^4 particles.

## Baseline (main 3074c1e)

| input | M cells/s | `rebuild` checks |
|---|---|---|
| jittered | 18.7 | PASS |
| uniform | 10.47 | FAIL (faces=2*owned, known: clipping alone is not watertight) |
| clustered | 0.93 | PASS |
| halo | 0.58 | FAIL (one cell face_overflow, volume 0) |

Cell-list build: 0.4 / 0.8 / 1.2 ms, and 28 ms on halo (the per-cell insertion
sort is quadratic in the cell occupancy).

Per-cell work against the ideal (instrumented build, 40k sampled cells; the
host counts the particles within 2 R_final of each generator, which any exact
security-radius method must test; `beyond` = tested candidates outside
2 R_final). On main the clustered and halo rows mix in cells finished by the
thread kernel, whose R is the warp attempt's, so their "within" column is
too high:

| input | tested | cuts | particles read | within 2R_final | beyond |
|---|---|---|---|---|---|
| jittered | 24.3 | 16.2 | 65 | 21.9 | 2.4 |
| uniform | 48.0 | 17.8 | 103 | 42.8 | 5.2 |
| clustered | 168 | 37.2 | 754 | 59.1 | 109 |
| halo | 236 | 37.0 | 711 | 151 | 85 |

On clustered input, cells whose home grid cell holds 16-63 particles test 202
candidates for 45 cuts (ideal ~59 and ~18); cells in single-occupancy cells
behave like uniform input (48 tests, 18 cuts). Halo cells in grid cells with
more than 256 particles test ~1400. The extra cuts come from candidates
that are not near-first: a 3x3x3 block of overfull grid cells delivers its
particles 32 at a time in storage order, so the cell is first cut by far
particles and then re-cut by nearer ones.

Where the time went on main (status after the warp kernel, before the
fallbacks; the coordinator's nsys run agrees: the thread pass is 84 % of the
clustered and 83 % of the halo time):

| input | warp kernel ok | uncertified (4-cell reach) | plane overflow (>64 planes) | vertex overflow | big pass |
|---|---|---|---|---|---|
| clustered | 1,455,588 | 85,211 | 556,349 | 4 | 711 |
| halo | 1,512,528 | 162,875 | 421,720 | 29 | 1,638 |
| uniform | 2,097,148 | 0 | 0 | 4 | 4 |
| jittered | 2,097,152 | 0 | 0 | 0 | 0 |

So 30 % of the clustered cells (28 % of halo) went to the thread kernel:
plane overflows from the many transient cuts in blobs, uncertified cells in
voids where four cell widths do not reach 2R.

## Design

Implicit octree from a Morton sort, per-particle levels, the existing
nearest-first stencil walk at that level.

1. `neighbor_search::build` sorts the particles by a 3 x (levels + 10)-bit
   Morton key (cub radix sort on CUDA, std::sort on host backends; stable, so
   ties keep particle-ID order and the slot order is deterministic). The base
   grid (2^levels per side, the power of two nearest to one particle per cell)
   keeps a dense offset table in Morton order, so every coarser node (a block
   of 8^k base cells) is a contiguous slot range found with two loads, and
   every finer node (down to 2^-10 base widths) is a contiguous part of one
   base cell found by binary search on the stored 30-bit fine key.
2. Per slot, `level` = the grid level whose cells hold about one particle
   there: take the deepest node around the particle that still holds at least
   64 particles (a density estimate with ~12 % noise) and refine by
   round(log8(count)). On uniform input every particle gets the base level
   (count 64-512 at the node two or three levels up), so the walk is the old
   one.
3. Both kernels walk the stencil (cells sorted by their smallest possible
   distance, up to 4 widths) at that level, 32 grid cells per batch, fetch 32
   candidates at a time, sort them nearest-first and clip, and certify when
   the next cell's lower bound exceeds 2R, exactly as before. If the stencil
   ends first, the walk continues one level coarser and skips the coarse cells
   whose eight children the finer walk covered (partially covered ones are
   walked again; their particles re-test without cutting). Coarsening goes
   down to 4 cells per box side (reach 1), which is as far as the start cube
   (half-width 0.25 box) allows. When every particle is at the base level
   (near-uniform input), the warp kernel is instantiated without level
   changes and fine-cell searches, which is the old walk on a Morton-ordered
   grid (see below for why).
4. Exactness: a grid cell is only skipped when its lower bound is at least
   2R or the finer walk already covered all of it; a cell is certified only
   when every unvisited cell's bound is at least 2R. Bounds are rounded down
   (float nextafter) as before. Positions are mapped to cells with one
   function (`grid_coordinate`, x / box * 2^k) everywhere, so the node a
   particle is sorted into and the node the walk computes for it agree
   bit for bit.

Why this one: the measurement says the problem is order (candidates not
nearest-first) and occupancy (many particles per grid cell, or many empty
cells), not the certification. A per-particle level restores ~1 particle per
grid cell everywhere, which is the regime in which the existing kernel is
already near the ideal (uniform: 48 tests for 43 within 2R_final). It keeps
the warp kernel's structure (32 grid cells per lane batch, batched fetch of
32, warp sort, nearest-first clipping) and its fast path on uniform input is
the old walk. A best-first tree traversal (LBVH, Karras 2012; best-first BVH
as in Taveira et al. 2026) needs a per-warp priority queue and per-node
bounding boxes; it would also handle sharp density jumps within one stencil,
but the per-particle level already brings clustered and halo to the uniform
regime (below), so the remaining gap is clipping, not search.

## Results

Final gate, job 31686774 (gpu1, ravg1009), main 3074c1e (+ the halo input)
against the branch, two interleaved rounds (both rounds agree to 0.1 %):

| input | main M cells/s | branch M cells/s | change | tested / cuts per cell, main -> branch | `rebuild` checks main / branch |
|---|---|---|---|---|---|
| jittered | 18.74 | 21.92 | +17 % | 24.2 / 16.2 -> 21.9 / 16.0 | PASS / PASS |
| uniform | 10.46 | 10.54 | +0.7 % | 47.9 / 17.8 -> 48.0 / 17.9 | FAIL / FAIL (known faces=2*owned) |
| clustered | 0.93 | 8.71 | 9.4x | 169 / 37.2 -> 50.7 / 19.8 | PASS / PASS |
| halo | 0.58 | 8.30 | 14x | 234 / 37.0 -> 53.7 / 19.4 | FAIL / PASS |

Cell-list build (included in pipeline rates, not in the table): 0.69 / 0.97 /
1.01 / 1.02 ms (main 0.40 / 0.83 / 1.23 / 28 ms). The jittered gain is the
base grid: main uses ceil(cbrt n) = 129 cells per side for 2^21 points, the
branch the nearest power of two, 128, so each jittered particle has its own
cell (main with 128 cells: 21.7 M cells/s, job 31684227).

Thread kernel (`thread-double`, which also builds the face rings for the
persistent mesh; job 31686845): jittered 6.43 -> 8.97, uniform 2.42 -> 2.56,
clustered 0.50 -> 2.15, halo 0.38 -> 2.10 M cells/s.

Per-cell work after the change (same instrumentation; all sampled cells built by the warp kernel):

| input | tested | cuts | particles read | within 2R_final | beyond |
|---|---|---|---|---|---|
| jittered | 21.9 | 16.0 | 76 | 21.9 | 0.0 |
| uniform | 48.0 | 17.9 | 103 | 42.7 | 5.3 |
| clustered | 50.7 | 19.8 | 112 | 43.9 | 6.8 |
| halo | 53.9 | 19.4 | 116 | 47.3 | 6.6 |

Clustered and halo cells now cost what uniform random cells cost; the
remaining difference to uniform is ~2 more cuts per cell in void cells
next to blobs. Fallback passes after the change:

| input | warp kernel ok | uncertified | plane overflow | vertex overflow | big pass |
|---|---|---|---|---|---|
| clustered | 2,097,129 | 0 | 21 | 2 | 2 |
| halo | 2,096,978 | 0 | 138 | 35 | 42 |
| uniform | 2,097,148 | 0 | 0 | 4 | 4 |
| jittered | 2,097,152 | 0 | 0 | 0 | 0 |

Exactness on the A100 (2.1M, final branch against main, by particle ID,
`clipped()` cells dumped from both and compared on the host):

| input | neighbour sets differing | max relative volume difference | cells failed main / branch |
|---|---|---|---|
| clustered | 0 | 1.4e-13 | 0 / 0 |
| uniform | 0 | 5.9e-14 | 0 / 0 |
| halo | 35 | 4.6e-13 | 1 / 0 |

The 35 halo sets are the one cell main fails (face_overflow, volume 0) and its
35 neighbours: main drops that cell's faces, the branch builds it. Volumes now
sum to 1 - 3e-16 on halo (main: 1 - 8.2e-6). `rebuild` checks: jittered,
clustered PASS on both; uniform the known faces=2*owned FAIL of clipping
alone on both; halo FAIL on main, PASS on the branch. ctest 4/4 on CPU
(OpenMP, 1 thread) and on the A100.

## Face output overflow (halo)

On halo, main leaves one cell with status face_overflow (more than 32 owned
faces; volume 0, faces=2*owned NO). The branch keeps the face arrays at 32
owned faces per cell by default but makes the capacity a run-time size of
`Cells`; a build that finds a cell owning more counts them, widens the face
arrays to the largest count (copying the cells already built) and builds
those cells again with the big-capacity thread kernel. On the 2.1M halo this
grows the arrays once (to 33 owned faces). Specified in the clipping test
(an output for 4 owned faces is widened, same cells). `construct` and
`construct_warp` now take `Cells&`.

## Memory

Bytes per particle at 2^21 particles (positions, 24 B, in both):

| | main | branch |
|---|---|---|
| slot order | 4 | 4 |
| base-cell offsets | 4.1 (129^3 cells) | 4.0 (128^3 cells) |
| fine key | - | 4 |
| level | - | 1 |
| build scratch | 8.1 (keys, counts) | 0.27 (cub temporary, 568 KB); keys and IDs live in the positions buffer before it is filled |
| total | 16.2 | 13.3 |

The base grid is the power of two nearest to one particle per cell, so the
offset table holds 0.35-2.8 cells per particle (1.4-11 B) for other n. The
widened face output costs n x 52 B per extra owned face (halo: +109 MB once,
for one cell); see open items.

## Kept / dropped

- First version kept the walk state in a struct and the generator's cell on
  the finest grid: clustered 8.5, but uniform 9.86 (-6 %). Nsight Compute: the
  coordinates went to local memory (spill requests 90M -> 153M).
- Home cell per level in registers: uniform -3.9 % (spills 122M; the "no
  instruction" stall went from 5.3 to 5.9 cycles per issue: the larger code
  thrashes the instruction cache).
- Binary search of fine cells not inlined (`noinline`): uniform -3.3 %,
  clustered -0.8 %. Dropped.
- Kept: a template flag selects the warp kernel. When every particle is at the
  base level (an occupancy statistic computed with the cell list), the kernel
  without level changes and fine-cell searches runs: uniform +0.7 %. Cells it
  cannot certify go to the thread kernel, which always walks adaptively.
- Not tried: a best-first LBVH traversal (Karras 2012; Taveira et al. 2026
  report 27-37x for best-first over depth-first on clustered data). With
  per-particle levels the clustered and halo cells already test ~51-54
  candidates for ~44-47 within 2 R_final, so a tree traversal could save at
  most ~10 % of the tests; the cuts (19-20 vs 18 on uniform) are clipping.
- Also new: the cell list is built by one radix sort instead of a counting
  sort plus a per-cell insertion sort (28 ms -> 1 ms on halo), and the old
  `--morton` option and `cell_rank` table are gone (the base grid is always in
  Morton order).

Literature used: Karras 2012 (Morton codes as an implicit hierarchy), Ray et
al. 2018 and votess (security-radius certification of kNN cells), Taveira et
al. 2026 (best-first order matters most on clustered input); see
`GPU Voronoi search and construction algorithms/neighbour_search.md`.

## Open items

- Void cells next to blobs make ~22 cuts (uniform 18): their level comes
  from a 64-particle node that mixes densities. A smaller node or a bias
  toward finer levels is untested.
- Inputs with every particle at the base level use the lean kernel; a single
  particle elsewhere switches the whole build to the adaptive kernel (~4 %
  slower on the uniform part). A threshold could keep the lean kernel for
  almost uniform input.
- Only n = 2^21 measured on the GPU. For other n the base grid has 0.35-2.8
  particles per cell; uniform input may then get levels on both sides of the
  base and the adaptive kernel.
- The Morton key resolves 10 levels below the base cell (2^30 per base
  cell); denser clumps are walked at that depth, correct but slower.
- The sort is cub on CUDA and std::sort on host-accessible memory; HIP/SYCL
  builds need a sort (Kokkos::Experimental::sort_by_key plus a tie fix).
- Halo cells still overflow the warp capacities (138 plane, 35 vertex of 2.1M)
  and are rebuilt by the thread kernel; the thread kernel alone on halo shows
  inconsistent face pairs (faces=2*owned NO), like uniform.
- Face output widening copies all n rows for one big cell. A spill list for
  the few cells above 32 owned faces would be leaner (memory work).
