# Mesh-step speed and the 256^3 memory fix, 2026-11-04

One A100 40 GB (Raven), jittered, smooth bench flow, medians of steps 2-9,
mean over rounds. "main" is 86fdae5, run interleaved with this branch in the
same job on the same node. M cells/s; total = mesh update + `update_cells`
per step. Binaries in build-meshspeed/bin, logs in build-meshspeed/logs.

## Result

Job 31886231 (128^3, ravg1010, 3 rounds), job 31886232 (256^3, ravg1016):

| n, spacings | main update | branch | main update_cells | branch | main total | branch | total / main | peak B/cell main -> branch |
|---|---|---|---|---|---|---|---|---|
| 128^3, 0.03 | 201.9 | 209.8 | 330 | 574 | 125.4 | 153.6 | 1.225 | 2191 -> 1998 |
| 128^3, 0.1 | 112.8 | 119.0 | 326 | 448 | 83.7 | 94.0 | 1.123 | 2191 -> 1998 |
| 128^3, 0.3 | 54.6 | 57.5 | 316 | 436 | 46.5 | 50.8 | 1.091 | 2221 -> 2028 |
| 256^3, 0.03 | 279.5 | 290.9 | 339 | 666 | 153.2 | 202.4 | 1.322 | 2191 -> 1998 |
| 256^3, 0.1 | 181.6 | 188.3 | 336 | 543 | 117.8 | 139.8 | 1.187 | 2191 -> 1998 |
| 256^3, 0.3 | 93.3 | 98.2 | 328 | 458 | 72.6 | 80.8 | 1.113 | 2200 -> 2007 |

Moving Sedov (HLLC, per face, 40 steps), M cell updates/s:

| | main | branch |
|---|---|---|
| 128^3 whole step | 73.4 | 83.7 |
| 128^3 mesh / cells phase | 292 / 333 | 296 / 725 |
| 128^3 hydro phase | 140.4 | 140.4 |
| 256^3 whole step | out of memory (job 31886232: `face_rings_pool` 7.847 GiB) | 90.3 |
| 256^3 mesh / cells phase | | 376 / 777 |

The sweep's speed is unchanged. No phase got slower against main; the memory
work did cost part of the speed gain, see below.

## Bits and checks

- Bits: final U, x, volume by sha256, 128^3, 50 steps, on five paths (moving
  per face, moving per cell, moving HLL, static, order 0): equal to main
  86fdae5 on all five (job 31886251). Face numbers were never reproducible
  after flips on main (slots are claimed by racy counters) and are not
  compared; the geometry is.
- Incremental against full lists: per-step checksums of the bits of volumes
  and centroids equal at 0.03/0.1/0.3, 5 steps at 2.1M (job 31881856, on the
  version before the memory work, with every step incremental).
- `step 262144 <kind> 0.3 3 --verify`, six inputs: 18/18 PASS (job 31886259).
- The list oracle (each face in both its cells, in increasing id of the other
  particle) is new in `support::compare_with_rebuild`, and `repair_test` now
  calls `update_cells()`, so its steps cover the incremental path.
- ctest 7/7 on the A100 (job 31885644) and on CPU (OpenMP, gcc 15, one
  thread, job 31885797). No warning in that CPU build's log. clang-tidy not run.

Three verify steps show failing=0, mismatched_cells=2, extra_faces=1:
clustered step 1 (area 6.839e-16 spacings^2), lattice step 3 (1.04029e-14),
wave step 3 (1.93586e-15). Main gives the same numbers to the last printed
digit on the same steps (job 31913960). The mesh is exact (no failing
triangle, nothing missing); the mesh keeps one Delaunay edge whose face has
rounding-level area, and clipping dropped that face, so the two cells of that
one face count as mismatched. This is the documented oracle behaviour
(TESTING.md, "What PASS means": extra faces below 1e-9 spacings^2). I did not
look up the three particle pairs; the attribution rests on missing=0, the
areas and the equality with main.

## Memory: what grew, and the fix

gpu_total is 42405 MB and the CUDA context takes about 1570 MB, so views may
hold about 40800 MB, 2432 B/cell at 256^3. On main moving 256^3 needed more:

- after hydro_setup 2230 B/cell (37426 MB, job 31879294), of which 48 B/cell
  are the moving arrays;
- `borrow_scratch` (src/hydro/finite_volume.hpp:317) asks for 488 B/cell:
  primitives 48 (new in the hydro rounds), gradients 120, face fluxes 40 B per
  face = 318. `Mesh::scratch` reallocates the spare ring pool to that, from 244
  B/cell. 2230 + 244 = 2474 B/cell, 41.5 GB, plus the context 43.1 GB. Static
  still fitted (42285 of 42405 MB, job 31879296), which is why only moving
  broke; the 48 B/cell of primitives were enough to push it over.
- mixed: `face_rows`, 32 B per face with 1/8 capacity, 288 B/cell, on top of
  the double area and centroid; it fell over at the first `cell_face`.

It was not a transient double allocation in `scratch()`: `Kokkos::realloc`
frees first when no other view refers to the allocation
(Kokkos_CopyViews.hpp:2974-2975). But such transients did exist on the rarer
paths, and are closed too (below).

Fix (src/voronoi_construct/tet_mesh.hpp, `FaceRings`, `update_face_rings`):

- one `start`, `ring_start`, `ring_length` instead of two: the last step's copy
  is needed only while numbering, so it now sits in the spare pool, copied at
  the start of the update (the hydro's scratch is free then). -207 B/cell at
  1/8 capacity.
- the dirty list in the spare pool. -36 B/cell.
- capacity of face arrays 1/32 instead of 1/8, of `cell_face` 1/32 of a list
  entry count instead of 1/8. A growth no longer copies ring arrays.
- no transient at growth: `mesh.owner` etc. dropped before the reallocations;
  the hydro drops its `geometry_` before `mesh_->update`
  (src/hydro/finite_volume.hpp, `move`), one line, the only hydro change;
  `rebuild()` frees the old pools first; after a compaction into a grown spare
  the rings always move back into the smaller pool, grown only to hold them,
  instead of growing it to the scratch's size (both pools at 8 GB).
- `scratch()` grows with 1/64 room, so not at every face added.

Peak, B/cell, device views, moving 256^3:

| | main | branch | GPU in use, of 42405 MB |
|---|---|---|---|
| per face, all double | 2474 (does not fit) | 2298 | 40125 MB |
| per face, `--precision=mixed` | does not fit | 2334 | 40727 MB |
| per cell | 2231 | 2038 | 35767 MB |

Runs to t_end, 256^3 moving Sedov, 923 steps: all double 177 s, 88.0 M
updates/s (job 31885794); mixed 169 s, 92.0 M updates/s (job 31885795). The
README's 288 s (job 31702077) is now 177 s. Energy conserved to 1.7e-16
in both. Peak flat over the run. Built from the commit before the two
reverted switches were removed; those were off, so the code that ran is the same.

Speed cost of the memory fix: copying the last numbers each step. Mesh update
at 0.03, 128^3: 215.0 (job 31884957, before) -> 209.8 (job 31886231, after),
different nodes, so about -2.4 %; hydro's mesh phase 309 -> 296, about -4 %. It
buys every moving 256^3 run. Not compared on one node.

## Candidates, each measured

Fraction of cells touched (2.1M, job 31878971; cells at a corner of a created
tetrahedron / cells whose set of faces changed): 0.03: 13.1 % / 5.6 %. 0.1:
36 % / 17 %. 0.3: 65-71 % / 37-43 %. Not measured in the Sedov flow, which flips far less (128^3: 9272 device flips in 40 steps, job 31886231).

What can be reused: an untouched cell keeps the set and sorted order of its
faces, but the face numbers change every step, because they are a prefix sum
over tetrahedra (tet_mesh.hpp `number_faces`): a flip anywhere renumbers all
later faces. So the list is rewritten through a table old number -> new, and
the CSR offsets shift. Areas and centroids change every step, so every cell
sums again; only sort and fill are saved.

| variant | update | update_cells | total / main, at 0.03 | verdict |
|---|---|---|---|---|
| no atomic volumes in `faces()` | +1.4 % | | 1.008 | dropped: needs an API switch, buys <1 % of the mesh step |
| 1 incremental, warp per cell as main | -0.8 % | 393 | 1.058 | sort is not the main cost |
| 1 incremental, thread per cell | -1.4 % | 253 | 0.889 | worse: reads chained per thread (hydro 56.4 vs 64.4). Reverted |
| 1 incremental, 8 lanes per cell, sums shared | -0.0 % | 524 | 1.163 | see below |
| same, 4 lanes / 16 lanes | | 579 / 444 | 1.192 / 1.107 | 4 lanes kept |
| full rebuild of the lists with the 4-lane sums | | 453 | 1.106 | job 31881855; beats main at all spacings |
| 1 final: incremental below n/5 created tetrahedra, sort only new entries | | 574 | | default |
| 2 ring lengths and append in one kernel | +2.5 % | | 1.180 vs 1.159 | default; job 31880691 |
| 3 one-sided screening | +3 % | | 1.018 | default; job 31880049 |
| 3 skip plan's repeat in-sphere, no dirty clear in the screen | +0.0-0.3 % | 0 | 1.000 against variants 1-3 together | buys nothing, reverted |

Likely cause (inferred from these timings and an instruction count, not
measured directly): a warp per cell ran each cell's sequential sum on its own,
16 entries of 4 shuffles and 4 FP64 operations. Dropping the sort alone gave
little (393), sharing the sum gave a lot (579). Thread per cell shares the
instructions over 32 cells but each thread reads its chain of 16 entries one
after the other, and memory waits decide. Four lanes per cell, 8 cells per warp,
gathers a batch of 4 entries in parallel and shares the sum: `cell_sums` 2.16 ms
against 4.24 for the old kernel, sort 0.61 (blocking launches, kernel timer).
The same splitting and sums help the full path, so `CellGeometryWarp` is gone;
one sort, one sum, on both paths and in D = 2, 3.

One-sided screening: a triangle fails from both sides or from neither, so only
the lower index tests it; `screen` 2.41 -> 1.94 ms. What it lists differs, the
exact tests decide the same mesh, hence the same bits.

ncu (job 31886256, kernels at base clocks; 128^3 moving, launches of the Sedov):
`cell_sums` 1.67 ms, 52 regs, warps 53 %, long-scoreboard 83 %, FP64 20 %, DRAM
1.35 GB. `cell_sort` 1.23 ms, 40 regs, warps 69 %, long-scoreboard 60 %. So
`cell_sums` is now memory-waits-bound. `tet_mesh::faces` is unchanged (2.15 ms, 78
regs, long-scoreboard 69 %, DRAM 64 % of peak), `move_screened` 1.72 ms (was 2.49, 62
regs, 69 %). These two and `face_numbers` (1.17 ms) are what's left.

## Not done

- Stable face numbers (a table per face, freed slots) would drop `face_numbers`
  and the table, up to 1.2 + 0.4 ms of about 11, but cost the "faces nearly
  sorted by owner" order and need a rewrite of edge representatives. Not tried.
- Fusing circumcentres into faces: the fan sum over a ring is ordered, a
  tetrahedron-major pass would change bits. Not tried.
- Fewer repair rounds: not tried. Rounds are 6 at 0.03.
- `Kokkos::resize` of tetrahedra at growth (tet_repair.hpp `reserve`) still
  copies, so holds two copies of those arrays for a moment. Their 1/8 capacity was
  not reached in these runs.
