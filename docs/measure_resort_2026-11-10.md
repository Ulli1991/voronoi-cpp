# Re-sort, faces by owner, owner bit: measured (2026-11-10)

Issue [#34](https://github.com/Ulli1991/voronoi-cpp/issues/34) and ideas R15 and "faces numbered by owner" of [ideas.md](ideas.md). What the layouts of [how_it_works.md](how_it_works.md#12-memory-order-and-numbering) cost and buy.

## Setup

One A100 40 GB on Raven (`gpu1`), 1305 MHz SM clock in every job (from `nvidia-smi` every 5 s), double precision, periodic unit box. `voronoi_bench step <n> jittered 0.1 500`: the bench's smooth flow at 0.1 mean spacings per step, 500 steps, points first in cell-list (Morton) order. Branch `resort`. Scripts: `scripts/resort_a100.sh` (jobs), `scripts/resort_tables.py` (tables), `scripts/resort_bits.py` (bits); job logs in `build-cuda/jobs` on Raven.

A step's time below is everything `update()` and `update_cells()` take: repair, host fallback, rebuild, re-sort, geometry, lists. "Over the run" is cells times steps over the sum of those times, so the re-sorts are paid in it.

| Jobs | What |
|---|---|
| 32146518, 32147166 | decay on today's library (main 1316317, bench with `--order`), 128³ and 256³ |
| 32147124, 32147675, 32147125 | 128³: every ordering, at the build and re-sorted; re-sort interval; trigger |
| 32147676 | 256³: re-sort every 50 and 100 |
| 32147679 | bits: cells by particle after 500 steps, three orders |
| 32149022, 32149023 | 128³: faces by owner and the owner bit; kernel times and memory |
| 32149024 (luna-cpp) | moving Sedov 128³ through luna-cpp's hydro |

## The decay, on today's library

128³ (2,097,152 cells), median of 11 steps around each:

| step | repair, ms | geometry, ms | lists, ms | M cells/s | flips | owner step p50 | owner step ≤ 64 | \|owner − neighbour\| p50 | list span p50 | 128-B lines per list entry | tetrahedra step ≤ 64 | cell from its Morton place, p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 9.0 | 9.0 | 4.8 | 92.2 | 201,483 | 3 | 0.817 | 18 | 863 | 0.637 | 1.000 | 0 |
| 100 | 6.6 | 6.0 | 4.3 | 124.3 | 51,145 | 11 | 0.646 | 18 | 6.3e6 | 0.722 | 0.550 | 8,053 |
| 200 | 8.6 | 8.9 | 6.4 | 87.9 | 87,746 | 51 | 0.519 | 28 | 9.9e6 | 0.770 | 0.314 | 29,196 |
| 300 | 9.5 | 10.5 | 8.3 | 74.4 | 84,859 | 878 | 0.417 | 49 | 1.2e7 | 0.803 | 0.170 | 69,176 |
| 400 | 10.4 | 11.7 | 9.7 | 66.1 | 82,736 | 4.0e4 | 0.353 | 82 | 1.3e7 | 0.820 | 0.096 | 128,534 |
| 500 | 11.0 | 12.5 | 10.7 | 61.4 | 82,941 | 1.1e5 | 0.312 | 111 | 1.3e7 | 0.828 | 0.056 | 177,458 |

(The order columns at step 2 are those of step 0.) Owner step: |owner(f + 1) − owner(f)|; list span: the largest minus the smallest face of a cell's list; lines per entry: distinct 128-byte lines of `area` rows a cell's list touches, per entry; tetrahedra step: the lowest corner of consecutive live tetrahedra. The first steps flip 200,000 times while the jittered lattice relaxes; from step 200 the flips are flat at 83,000, and the rate still halves between step 100 and 500. The faces and tetrahedra lose their order first (tetrahedra within 64 cells of each other: 55 % by step 100, 6 % by 500), the cells later (the median distance between a face's cells 18 to 111). This reproduces the earlier profile (job 31953232: 118 to 59 M cells/s).

256³ (16,777,216 cells): the flow varies on the box scale, so cells shear half as much per step, and the decay is slower. Over 500 steps 126.5 M cells/s; steps 2 to 11 134.8, the last 10 95.8 (−29 %). At step 500 the order is that of 128³ near step 250 (owner step ≤ 64: 0.464; tetrahedra within 64: 0.233).

## Speed by ordering (128³, 500 steps)

| ordering | re-sort | M cells/s over the run | steps 2-11 | last 10 | re-sort, ms | its step, ms over the steps around | rebuild after 500 steps, M cells/s |
|---|---|---|---|---|---|---|---|
| today (cell-list Morton) | none | 81.6 | 91.8 | 61.5 | | | 6.7 |
| Morton | at the build | 81.5 | 91.8 | 61.4 | | | 6.6 |
| Morton | every 100 | 115.0 | 91.8 | 105.7 | 16.4 | 24.1 | 7.2 |
| Hilbert | at the build | 81.3 | 92.5 | 61.1 | | | 7.8 |
| Hilbert | every 100 | 115.9 | 92.1 | 106.6 | 11.0 | 18.6 | 7.3 |
| Moore | at the build | 81.3 | 91.4 | 61.3 | | | 6.1 |
| Moore | every 100 | 115.5 | 91.6 | 105.9 | 13.5 | 21.1 | 7.2 |
| Gray-coded Morton | at the build | 81.4 | 91.7 | 61.5 | | | 5.9 |
| Gray-coded Morton | every 100 | 114.3 | 91.8 | 106.2 | 22.8 | 30.5 | 7.3 |
| row-major | at the build | 70.8 | 79.8 | 53.7 | | | 6.7 |
| row-major | every 100 | 102.2 | 79.9 | 95.3 | 15.4 | 27.0 | 5.0 |
| reverse Cuthill-McKee | at the build | 72.3 | 86.7 | 55.0 | | | 5.8 |
| reverse Cuthill-McKee | every 100 | 101.5 | 86.6 | 93.8 | 85.7 | 93.8 | 5.3 |

Job 32147124; a second job (32147675) repeated row-major, RCM and Hilbert every 100 within 0.3 %. The re-sort time is the mean of its five; "its step" is what the step with the re-sort took beyond the median of the six steps around it, the re-sort included (it walks every face ring again and rebuilds every list). The rebuild column is `Mesh::rebuild` of the final positions, allocations included, in the order the run left them; it does not depend on the ordering beyond noise.

Hilbert, Moore and Morton are within 1 % of each other; Gray-coded Morton's re-sort costs more for nothing. Row-major is the floor (−13 % at the build), RCM is no better (−11 %) and its re-sort costs 80 to 86 ms at 128³, eight times a curve's: it runs on the GPU, level by level (two searches of about 200 levels), which is reasonable time but buys nothing here. **Default: Hilbert**, with the cheapest re-sort of the three best curves.

Interval and trigger (Hilbert):

| re-sort | M cells/s over the run | last 10 | re-sorts |
|---|---|---|---|
| every 25 | 122.3 (121.4 in a second job) | 124.0 | 20 |
| every 50 | 121.0 (120.1) | 117.7 | 10 |
| every 100 | 115.9 | 106.6 | 5 |
| every 200 | 106.5 | 106.0 | 2 |
| when more than 40 % of faces join cells over 64 ids apart | 104.0 | 113.5 | 2 |

Every 25 to 50 updates gives 1.48 to 1.50 times today's rate over the run; a re-sort costs about one step (11 ms plus 8 ms of extra walking on its step, 25 ms a step). The `far` trigger reacts late: the cells' own order (the share of far faces) decays more slowly than the order of tetrahedra and faces, which is what costs. A fixed interval is the better trigger on this flow.

256³, Hilbert (job 32147676; today's rate from job 32147166, same clock):

| re-sort | M cells/s over the run | steps 2-11 | last 10 | re-sort, ms | its step, ms over the steps around |
|---|---|---|---|---|---|
| none (today) | 126.5 | 134.8 | 95.8 | | |
| every 50 | 165.8 | 134.4 | 164.2 | 114 | 188 |
| every 100 | 162.5 | 134.1 | 154.3 | 124 | 196 |

+31 % over 500 steps at 256³; a re-sort costs about two steps there.

## Bits

Cells compared particle by particle after 500 steps (job 32147679, 2,097,152 cells), against the run without a re-sort:

| run | neighbour sets that differ | volumes whose bits differ | largest relative volume difference | largest centroid difference / cell size |
|---|---|---|---|---|
| Hilbert every 50 | 0 | 1,505,465 | 3.2e-14 | 2.4e-14 |
| RCM every 100, faces by owner, owner bit | 0 | 1,483,508 | 2.3e-14 | 1.3e-14 |

The mesh as a set is the same: every particle keeps exactly its neighbours. The geometry agrees to rounding, not to the bit, because the orders that fix rounding (the corners of a circumcentre, the first vertex of a face's fan, the order of a cell's sums) follow the ids, which a re-sort changes. Making them bit-identical would mean ordering by a carried label instead of the id everywhere the geometry sorts. Faces by owner and the owner bit alone change no bits (the `order` test; luna-cpp's Sedov checksums below).

## Faces by owner and the owner bit

The numbering by owner (`tet_mesh::number_by_owner`) is a counting sort over the faces every update: counts, start, fill, a sort of each owner's few faces by slot, the numbers both ways, and the rewrite of the incremental lists' table. Kernel times at 128³, 0.1 spacings (job 32149023): 0.66 + 0.46 + 0.44 + 0.28 + 0.23 + 0.20 ms, 2.3 ms a step; `faces` takes 0.13 ms more (it reads its rings through the slot), `cell_sums` 0.22 and the list fill 0.11 less. Memory: 1998 B/cell at the peak of a step without, 2166 with (owner and neighbour by number, two slot tables, the inverse, `owned_start`: +168 B/cell). The owner bit costs nothing in the mesh.

The mesh step alone, 128³, 500 steps (job 32149022):

| layout | M cells/s over the run | steps 2-11 | last 10 |
|---|---|---|---|
| none | 81.5 | 92.1 | 61.3 |
| faces by owner | 83.4 | 83.8 | 67.6 |
| faces by owner, owner bit | 83.5 | 84.1 | 67.7 |
| Hilbert every 50 | 119.4 | 91.8 | 117.2 |
| Hilbert every 50, faces by owner, owner bit | 107.5 | 84.0 | 105.5 |

In the mesh alone faces by owner pay their 2.3 ms (−9 % early) and win some back once the order has decayed; with a re-sort they cost 10 %.

Through luna-cpp (branch `resort`, which follows a re-sort and reads the owner bit): moving Sedov 128³, two passes, 349 steps to t_end, medians per step, job 32149024. Phases in ms, rates in M cells/s.

| layout | hydro, ms | mesh, ms | lists, ms | whole step, ms | hydro M cells/s | step M cells/s | GPU s to t_end | checksum of Q |
|---|---|---|---|---|---|---|---|---|
| none | 19.16 | 8.19 | 3.07 | 30.58 | 109.5 | 68.6 | 10.70 | `ac03cb4f` |
| Hilbert every 50 | 18.55 | 8.35 | 3.03 | 30.10 | 113.0 | 69.7 | 10.73 | `6f140cdb` |
| faces by owner | 17.63 | 10.70 | 2.89 | 31.39 | 119.0 | 66.8 | 10.96 | `ac03cb4f` |
| faces by owner, owner bit | 16.52 | 10.71 | 2.90 | 30.28 | 126.9 | 69.3 | 10.57 | `ac03cb4f` |
| all three | 16.08 | 10.97 | 2.88 | 30.10 | 130.4 | 69.7 | 10.66 | `6f140cdb` |

The hydro gains what the profile predicted (faces by owner +9 %, with the bit +16 %, with the re-sort +19 %), but the numbering pass gives most of it back in the mesh: the whole step gains 1 to 1.6 %, the run to t_end 1.2 % at best. Sedov moves little, so the re-sort alone buys 1.6 % a step and costs as much in its six steps. The faces' own bits do not change (the checksum of Q is the same without and with faces by owner and the bit); a re-sort changes them, as above.

## What it says

- On a flow where every cell moves, a Hilbert re-sort every 25 to 50 updates is worth 1.5x over 500 steps at 128³ and 1.3x at 256³, after paying for itself. Hilbert, Moore and Morton are equal; the default is Hilbert. It is off by default, because it changes the cells' ids: a caller must follow `permutation()`.
- On Sedov the order barely decays and the re-sort buys about nothing either way.
- Faces by owner and the owner bit buy the hydro 16 %, but cost the mesh 2.3 ms a step and 168 B/cell at 128³; the whole step gains 1 %. A numbering that keeps the numbers of untouched faces (most of them on Sedov) instead of sorting every face every update would keep the hydro's gain: not done.
