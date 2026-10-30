# Ideas from Doug Rennehan's plume: results (2026-10-31, branch `plume-ideas`)

Ideas 1-4 of `plume_ideas_2026-10-31.md`, each tried in its own commit on
top of main 77a903b. plume is Doug Rennehan's (private repo rennehan/plume,
read at cf55509). Each commit names the plume commit it took the idea from.

All numbers come from the A100 (Raven gpu1) at 2,097,152 cells unless
stated. Every table compares binaries run interleaved in the same job
against 77a903b ("base"), built with the same compiler and Kokkos. A
value is the median over the rounds (two rounds in reversed order unless
stated). Rebuild figures are construct M cells/s (higher is better); the
other times are ms (lower is better). Steps are 2.1M jittered, the median
of steps 2-10; "cells" is `update_cells`. Logs are in
`/raven/u/uli/voronoi-cpp/build-plumeideas/logs/` (file name = job tag and
job id), binaries in `build-plumeideas/bins/<tag>`, and register counts in
`logs/regs-<tag>.txt` (from `cuobjdump -res-usage`).

## Result

| | base 77a903b | plume-ideas (ideas 1-3) | |
|---|---|---|---|
| warp rebuild jittered / uniform / clustered | 34.69 / 16.50 / 13.00 | 40.10 / 20.01 / 15.57 | +15.6 / +21.3 / +19.8 % |
| warp rebuild halo / lattice / wave | 12.10 / 46.01 / 33.51 | 14.24 / 50.15 / 38.49 | +17.7 / +9.0 / +14.9 % |
| exact start, 2.1M jittered | 119.1 | 118.1 | -0.9 % |
| exact start, 262k lattice / wave | 49.3 / 98.7 | 42.4 / 70.8 | -13.9 / -28.3 % |
| step 0.03 / 0.1 / 0.3 spacings | 11.09 / 18.93 / 37.21 | 10.45 / 18.13 / 36.53 | -5.8 / -4.2 / -1.8 % |
| update_cells (0.03 / 0.1 / 0.3) | 6.89 / 6.99 / 7.30 | 6.43 / 6.54 / 6.81 | -6.6 / -6.5 / -6.7 % |
| peak, start + steps at 0.3 (B/cell) | 2221 | 2221 | 0 |
| voronoi::Mesh first update (build_ms) | 179 | 132 | |

Sources: job 31701208 (one round, base / fin3 / fin4 interleaved) and job
31701207 (memory). Idea 4 (fin4) was not kept.

Exactness:
- ctest passes on CUDA (6/6, job 31701209) and on the CPU (OpenMP, one
  thread, 6/6).
- `step 262144 <jittered|uniform|clustered|halo|lattice|wave> 0.3 3
  --verify` passes every step, 18/18, for ideas 1-3 (job 31701207) and
  for 1-4 (jobs 31701209 and 31700805).
- The rebuild checks (`rebuild 2097152 <kind> warp 1`) all pass (6/6), with
  every cell certified and faces per cell equal to base.
- clang-tidy is clean.
- The topology is identical to base: the start gives the same tetrahedra
  (14583412 at 2.1M; 262k lattice 1572864 and wave 1849047), the same
  bad-tetrahedra and patch counts, and the same mesh face count (16680564).
- Volumes, centroids and face geometry change in the last bits (ideas 2
  and 3), so `cells_checksum` differs from base.

## Idea 1: vertex adjacency and slot reuse in the per-thread kernel (kept, 064dac3)

plume cc3a81a. Changes to `RoundedCell` and `ExactCell`:
- Each vertex holds the three vertices across its edges.
- A cut finds its hole edges through those links.
- Removed vertices are marked, and their slots go on a stack for reuse.
- The face walk follows the links.

This replaces three things in the old kernel: the quadratic twin matching
of hole edges, the copy of every kept vertex in every cut, and the O(V)
searches for a face's start and each next vertex. If a hole touches a
face twice (possible under rounding), the cell is left degenerate rather
than linked wrongly (plume's `OPEN_FACE`).

The first version checked whether each vertex was live in the cut test,
the hottest loop. On the A100 that made the halo fallback passes about
1.5x slower. The kept version moves a removed rounded vertex to the
generator instead: no bisector cuts it off, so the test needs no check.
One more trap: the back link of a kept vertex has to be found before any
slot is reused within the same cut, otherwise it may match a reused slot.

| | base | first version (jobs 31700484/620) | kept (job 31700902) |
|---|---|---|---|
| rebuild jittered | 34.69 | 34.69 | 34.69 (0.0 %) |
| rebuild uniform | 16.51 | 16.55 | 16.64 (+0.8 %) |
| rebuild clustered | 13.00 | 12.92 | 13.06 (+0.5 %) |
| rebuild halo | 12.11 | 11.52 (-4.7 %) | 12.13 (+0.2 %) |
| rebuild lattice / wave | 46.03 / 33.51 | 46.02 / 33.51 | 46.02 / 33.54 |
| start 262k lattice | 49.55 | 42.81 | 42.93 (-13.4 %) |
| start 262k wave | 98.56 | 67.14 | 71.19 (-27.8 %) |
| fallback passes, uniform (nsys, 48/64 + 128/256) | 1.81 + 1.66 | 1.79 + 1.48 | 1.33 + 1.15 |
| fallback passes, halo | 9.29 + 7.44 | 11.96 + 12.35 | 8.95 + 7.04 |
| thread variant (every cell per-thread), uniform / halo | 833 / 1002 | 932 / 1187 | 833 / 1021 |
| steps 0.03 / 0.1 / 0.3 (first version) | 11.13 / 18.99 / 37.31 | 11.11 / 18.91 / 37.29 | |

Registers and local memory (stack, bytes):
- RoundedCell: 112 -> 110 registers, stack 4208 -> 4768 (48/64) and
  12816 -> 14752 (128/256).
- ExactCell: 186 registers, stack 4896 -> 5456 (48/64) and 15888 -> 17824
  (128/256).

On the CPU (OpenMP, 4 threads of the login node, 262k cells, noisy), the
per-thread kernel was 1.4x faster on uniform (2967 -> 2130 ms) and 1.7x on
halo (4115 -> 2423 ms), and the lattice start went 4772 -> 4128 ms. On
lattices the exact in-sphere ties dominate the patch.

## Idea 2: warp-kernel face output (kept, 6cda567)

plume e4ccbfb and 6cbb429. The face walk sums, undivided, twice the area
vector and the vertex pairs weighted by their projected area, and divides
once per face. Variant (b) also takes the cell volume and centroid from
one pyramid per face: volume A.first / 3, with the centroid at 3/4 of the
face centroid. That drops the per-edge tetrahedron, which was a cross
product, a dot product and a moment update. Plane normals stay float. The
area vector is still the cross-product sum.

Jobs 31700484 and 31700620:

| | base | (a) divisions once per face | (a)+(b) pyramids (kept) |
|---|---|---|---|
| rebuild jittered | 34.70 | 39.35 (+13.4 %) | 40.10 (+15.6 %) |
| rebuild uniform | 16.52 | 19.51 (+18.1 %) | 19.85 (+20.2 %) |
| rebuild clustered | 12.99 | 15.12 (+16.4 %) | 15.38 (+18.4 %) |
| rebuild halo | 12.09 | 13.89 (+15.0 %) | 14.20 (+17.5 %) |
| rebuild lattice | 46.02 | 49.31 (+7.2 %) | 50.15 (+9.0 %) |
| rebuild wave | 33.51 | 37.78 (+12.8 %) | 38.47 (+14.8 %) |
| start 2.1M jittered | 119.6 | 117.7 | 118.0 (-1.3 %) |
| steps 0.03 / 0.1 / 0.3 | 11.13 / 18.99 / 37.31 | 11.18 / 18.94 / 37.22 | 11.18 / 18.98 / 37.31 |

The warp kernels stay at the 96-register cap. Their stack shrinks from 224
to 192 bytes (plain), and the ring kernel's from 72 to 64 bytes with (a).
Shared memory is unchanged, because (b) needs no per-plane |d|: the
pyramid volume is A.first / 3. The gain is several times the note's
estimate of 3-8 %. The face loop sat under the register cap, and removing
the per-edge chain removed spills from the walk.

## Idea 3: mesh faces and cells without square roots or per-triangle divisions (kept, f688f9b)

plume 6cbb429. Changes to `tet_mesh::faces`:
- Each ring triangle is weighted by |twice.dj| (its area projected on the
  separation) instead of |twice|. The two are equal for a planar ring,
  and the absolute value keeps a rounded, non-planar sliver's centroid a
  convex combination of its triangles.
- The vertices are summed undivided and divided once per face.
- The pyramid volume is |A.dj| / 6 instead of |A| |dj| / 6.

`update_cells` sums |A.d| and divides once per cell. The area vector
stays the cross-product sum, so the faces still close each cell to
rounding.

Jobs 31700484 and 31700620:

| | base | idea 3 |
|---|---|---|
| step 0.03 / 0.1 / 0.3 | 11.13 / 18.99 / 37.31 | 10.48 / 18.36 / 36.67 (-5.8 / -3.3 / -1.7 %) |
| update_cells 0.03 / 0.1 / 0.3 | 6.87 / 6.99 / 7.25 | 6.42 / 6.54 / 6.79 (-6.6 / -6.4 / -6.3 %) |
| rebuilds, starts | | within ±0.5 % |

Registers: `faces` drops from 92 to 80, the cell kernel from 56 to 48 and
`CellGeometryWarp` from 74 to 72. The step gain (0.65 ms at 0.03) exceeds
the note's estimate (0.1-0.5 ms). Most of it is in `update_cells`, which
the note did not count.

## Idea 4: exact arithmetic out of the repair's find and plan kernels (not kept: 7c7b285, reverted in 6f056cf)

First, a probe (job 31700861): the exact branches were removed from
`orient` and `in_sphere`, which gives wrong results on ties. Registers
dropped from 152 to 70 in find and from 134 to 90 in plan. Even so, the
steps sped up by at most 1.6 %.

The implementation added `ep::orient_filter` and `ep::in_sphere_filter`:
- find and plan decide with the filters alone and defer the undecided
  cases (`Plan::deferred`) to a deferred list.
- `find_exact` decides the deferred find cases. The first version always
  launched it; the second launched it only when the list was not empty,
  since the counts are read after find anyway.
- `plan_exact` re-plans the deferred plans exactly and then votes.

The decisions equal the base's, and verify passes 18/18. Registers:
find 70, plan 89, and the exact kernels 116 and 130.

| | base | probe (no exact code) | first version | second version |
|---|---|---|---|---|
| step 0.03 | 11.16 / 11.20 | 10.98 (-1.6 %) | 11.17 (+0.1 %) | 11.23 (+0.3 %) |
| step 0.1 | 18.86 / 19.02 | 18.84 (-0.2 %) | 18.91 (+0.3 %) | 18.92 (-0.5 %) |
| step 0.3 | 37.06 / 37.24 | 36.47 (-1.6 %) | 36.60 (-1.2 %) | 36.96 (-0.7 %) |
| peak (B/cell) | 2221 | | | 2227 |

Sources: jobs 31700861 (probe and first version) and 31701207 (second
version). The base column lists both jobs' base values.

Over three comparisons, the gain at 0.3 is -0.7 to -1.2 %, and the result
at 0.03 and 0.1 is mixed (±1 %). The repair time is set by the number of
rounds and the gathers, not by find and plan's registers. The gain stays
within noise, and the change adds two kernels and a deferred list
(+6 B/cell), so it was reverted.

## Notes

- Workflow: each variant was a source tree of base plus one idea's diff
  (`build-plumeideas/src-<tag>`; diffs `build-plumeideas/<tag>.diff`), so
  each idea was measured alone against base. The kept ideas touch
  separate files: idea 1 `shell_voronoi.hpp`, idea 2
  `shell_voronoi_warp.hpp`, idea 3 `tet_mesh.hpp`.
- The README's rebuild and step figures predate this branch; they are
  not updated here.
