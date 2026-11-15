# What we tried, what it did, and why we kept or dropped it

A running list of every change measured for speed, memory or accuracy since the speed goal was set on 2026-11-12 (baseline main `69de1d3`). One A100 40 GB, double precision, 1305 MHz unless a row says otherwise, 128³ (2,097,152 cells) unless stated. The effect is measured against the code just before the change, in the same job. Full tables and job ids are in [measurements.md](measurements.md) and the notes linked in each row.

Each new measurement adds its row here, in the same commit as its result.

Reason gives the structural cause, not "method X is worse".

## Rebuild (clipping)

| Change | Effect | Kept? | Reason | Where |
|---|---|---|---|---|
| Canonical face rings (each ring from its least id, faces by neighbour id) | 0 to −1 % | kept | frees the cut from slot order; same mesh | measurements.md, "Canonical rings and a lean cut"; `b5d1526` |
| Lean cut (lowest free slot by its own lane, cube adjacency by formula, branchless selects) | +10 to +15 % on all six inputs, 128³ and 256³; exact start +3 to +11 % | kept | 8–9 % fewer instructions per cell, fewer fetch stalls | same |
| P1 per-vertex balls inside the warp kernel | −10 to −64 % | dropped | adding code to the walk spills registers (local loads 175M → 258M) and raises instruction-fetch stalls (28 → 43 %) | measurements.md, "Per-vertex balls (P1) and directional boxes (R8)" |
| R8 directional box, float, in the warp kernel | −10 to −22 % | dropped | same: the kernel is at its size limit | same |
| Split clipping: gather, then clip in a small kernel | uniform +2 %, clustered +7 %, halo +4 %; jittered −10 %, lattice −5 %, wave −9 % | not adopted (opt-in branch `clip-split`) | spills and fetch stalls gone, but keeping the cell in memory between kernels costs what P1 saves | branch `clip-split`, docs/measurements.md there |
| Batched rounds of 32 with the R8 box | uniform +3 %, clustered +11 %, halo +8 %; jittered −10 %, lattice −5 %, wave −9 % | not adopted (opt-in branch `clip-split`) | same; best on hard input only | same |
| Float side test with a proven bound and double fallback | −20 to −26 % | dropped | the filter's own code costs more than double arithmetic, which was never the limit (13 % of instructions) | branch `clip-arith`, measurements.md there |
| Leaner double cut (before canonical rings) | −1 to −4 % | dropped | bounded by the fixed slot and cut order | same |
| Compact cell, one slot per lane up to 32 | −4 to −14 % | dropped | a second copy of the cut raises fetch stalls | same |
| Plane-index triplets, topology templates (R16) | not built | — | per-line costs show they can't win: about 30 more double operations per side test to save 25 per cut; templates could save at most the 16–18 % spent on branches | same |
| kNN-first, k = 24 / 32 / 48, on our octree | −6 to −58 % | dropped | order of operations: find-all-then-clip with a fixed k against the interleaved walk, which clips as it goes and stops once certified. At k = 32, 79 % of uniform, clustered and halo cells cannot be certified from the list and walk anyway; clipping from the list alone is already 92 % of the walk | measurements.md, "kNN-first clipping, hybrids and a vertex-ball filter" |
| kNN hybrids (continue with the walk; by density; k = 24 then 48 then the walk) | −6 to −55 % | dropped | same order-of-operations cost | same |
| Vertex-ball filter on the walk | −2 to −16 % | dropped | halves the candidate tests (uniform 48 → 24, halo p99 211 → 46), but the filter costs as many instructions as it saves | same |
| Delaunay-first: lattice plus motion through `update()` | jittered 3.5 against 17.4 (start); stuck on uniform, clustered, halo | dropped | lattice ties force a full re-triangulation; large shifts stall the repair | branch `delaunay-first`, research_notes/delaunay_first_2026-11-12.md |
| Delaunay-first: batched insertion with exact flips | 0.38 when it finishes; falls back on most inputs | dropped | floored by flips: about 13.5 flips per cell at 17–21 M flips/s | same |

## Moving step

| Change | Effect | Kept? | Reason | Where |
|---|---|---|---|---|
| Register caps on `faces` (80) and `screen` (64) | step +3.8 % | kept | better occupancy without spills | measurements.md, "Step kernels at today's bytes" |
| One thread per edge in `dirty_faces` | step +2 to +4 %; the kernel −39 to −49 % | kept | finer, balanced work | same |
| Warp-aggregated counters | −0.1 % at 0.03, +2.6 % at 0.1 | kept (owner's call) | fewer atomics on busy steps | same |
| One host wait per flip round | about +0.2 %; syncs 27 → 21 per step | kept (owner's call) | fewer host round trips | same |
| Register caps on `find`, `plan`, `apply` | −3 to −7 % | dropped | exact-arithmetic fallbacks need the registers; caps spill | same |
| Compacting vote winners before `apply` | +0.3 / −2.5 % on top of the rest | dropped | the extra pass costs what it saves | same |
| E: incremental face numbering (#53) | step +8 % at 0.03 (128³), +13 % (256³); about 0 at 0.3 | kept | face numbering 422 → 63 B/cell; renumbered faces lose memory order, giving some back | measurements.md, "Fewer bytes per step" |
| K2: one ring walk per dirty face, rewritten in place | +1 to +5 % | kept | the second walk mostly hit L2, so few DRAM bytes saved | same |
| Repair free list sized to use; cell list freed after a build | memory −19 B/cell | kept | pays for E's arrays | same |
| Re-sort along a curve every 50 steps (Hilbert) | 500-step run 1.31–1.48× | kept, opt-in | restores memory order the flow scrambles | measure_resort_2026-11-10.md |
| Re-sort by Morton, Moore, Gray | about equal to Hilbert | not default | Hilbert sorts fastest | same |
| Re-sort by row-major, reverse Cuthill–McKee | 10–12 % worse than the curves | dropped | worse locality | same |
| Faces by owner + owner bit | hydro +16 %, whole step about +1 % | kept, opt-in | numbering cost eats the hydro gain until it is incremental (#53) | same |
| R14: face geometry not stored by `update()`; `update_cells()` computes each face from its ring (twice), callers through `Mesh::face_geometry()`; the update's checks from tetrahedron volumes | job 32370682, 128³, `update()` + `update_cells()`: −22 % at 0.03 spacings (162.8 → 127.7 M cells/s), −13 % at 0.1, −8 % at 0.3; `update()` alone +24 %, +14 %, +6 %; step 3 moves 2552 instead of 3304 B/cell at 0.03 (−753), 4464 instead of 5263 at 0.1; bits equal to main (30 of 30 checksums); in luna-cpp the hydro's sweep 7.5 → 26 ms | kept, opt-in (`Layout::faces_on_read`), off by default | fewer bytes but slower: `cell_sums` gathers each ring twice (126 registers, 23 % occupancy, 7.4 ms against 2.2 for `cell_sums` plus 2.3 for `faces`), and every caller that reads a face more than once pays the ring again | branch `step-v1-r14`, measurements.md (V1 and R14) |
| V1: quiet tetrahedra skip the screen's neighbour tests under a certified motion bound (cofactors + Hadamard, one byte per tetrahedron) | job 32370682, V1 alone (faces stored), `update()`: 128³ 3.1× slower at 0.03 spacings (225.8 → 72.8 M cells/s), 2.3× at 0.1, 1.8× at 0.3; 256³ 3.7×, 3.0×, 2.1×; `screen` 1.96 → 19.9 ms, 528 → 10984 B/cell (step 3, 0.03); 4–26 % of tetrahedra skipped (most at 256³, 0.03); the motion measure alone (certificates off) 1.27× slower, 1733 B/cell in `screen`; topology and geometry bits equal to main | dropped | the certificate (4×4 cofactors per tested pair) spills from the screen's 64 registers to local memory, and every step reads the old snapped points of four corners per tetrahedron to measure the motion; the skipped neighbour tests were at most 111 B/cell × the skipped share, so even without the spill it could save under 30 B/cell | same; caused luna-cpp's 23.8 ms mesh step |
| Face visitor `Mesh::for_each_face` with `Layout::faces_on_read`: each face computed once from its ring in the caller's kernel, no face rows stored | job 32375192, `update()` + a closure functor: 128³ +10 % at 0.03 spacings (182.4 → 201.4 M cells/s), +6 % at 0.1; 256³ +14 %, +11 %; 2697 → 2199 B/cell at 0.03, 4445 → 3950 at 0.1; memory 1979 → 1585 B/cell; the visitor's values are the stored rows' bits; with `update_cells()` as well 23 % (128³) to 27 % (256³) slower | kept, opt-in | the faces kernel (1005 B/cell) goes and the visitor reads the rings once (663 B/cell), but `update_cells()` still computes every face twice from its ring | measurements.md (Face visitor and chunked passes) |
| Chunked passes: a two-pass functor over Hilbert chunks of C cells with a halo layer, both passes per chunk while it is in L2 | job 32375192: DRAM 943 → 598 B/cell (128³) and 831 → 598 (256³) at C = 65536, outputs bit for bit; but 2 % (128³) and 23 % (256³) slower at 64 to 512 chunk launches, and slower at every C; halo 0.16 cells per cell (one layer) at 65536, 0.26 at 16384 | not adopted (bench only) | each chunk's grid is too small to fill the GPU and its tail repeats per chunk, so the summed kernel time rises above the whole mesh's even with fewer bytes; needs chunks run concurrently, one per block or SM, not one after another | same |

## Precision

| Change | Effect | Kept? | Reason | Where |
|---|---|---|---|---|
| Float face rows as default | breaks cell closure (2–4e-8); small-amplitude waves 4–770× worse | rejected as default | closure must be exact for conservation | measure_2026-11-12.md, part 2 |
| Float circumcentres as offsets from a corner | geometry error 4e-8 to 1e-7 of h²; closure exact | yes geometrically; not yet adopted | needs a hydro check | same |
| Float circumcentres in absolute coordinates; float face arithmetic | error grows with N; closure 1e-7 | rejected | precision lost | same |
| `Layout::float_centroids`: face centroids stored as float (relative to the owner), areas double; face row 56 → 44 B | job 32373869, 128³ step 3: 3300 → 3096 B/cell at 0.03 spacings (`faces` −95, `cell_sums` −110), 5247 → 5036 at 0.1; `update()` + `update_cells()` +1.8 % (162.8 → 165.8 M cells/s) at 0.03, +1.1 % at 0.1, +0.7 % at 0.3; 256³ +1.9 %, +1.6 %, +1.3 %; step peak memory 1979 → 1881 B/cell; areas, volumes and topology bit for bit; cell centroids within 8e-9 spacings of double (mean 1.8e-9) | kept, opt-in, off by default; default is the owner's decision | closure and volumes use only the areas, which stay double; a float centroid relative to the owner rounds at 6e-8 of a face's offset, below every error the method has | measurements.md (Float face centroids) |
| Hydro face rows with float centroids (offsets from the owner), double areas, 44 instead of 56 B a face (luna-cpp `-DLUNA_FACE_ROWS=1`, repacked once per mesh) | luna-cpp Sedov 128³, 1410 MHz (job 32372277): moving hydro +2.6 % `grad32` (122.1 → 125.3 M cells/s), +2.0 % double; static +4.2 %, +4.3 %; moving step equal (repack 0.43 ms); hydro 809 → 779 B a face; Sedov, sound wave and vortex norms equal to 4–5 digits (job 32373383) | luna-cpp study (branch `face-rows`), adoption is a later luna-cpp task | closure stays exact with double areas; the centroid feeds only the limiter and the face velocity, where 1e-8 of a face is below the method's error; the gain is small because the gathers are sector-bound, not row-bound | luna-cpp docs/measurements.md, Face rows |
| Hydro face rows as one 48-byte row (double area, float centroid, owner, neighbour) | luna-cpp study: moving hydro −12 % `grad32`, static −16 %; hydro 1008 B a face; `face_step` 29 → 73 B a face | dropped | a kernel that needs 8 bytes of the row reads its whole sector, and the sweep's gathers in cell order touch more sectors | same |

## Correctness fixes that cost speed

| Change | Effect | Kept? | Reason | Where |
|---|---|---|---|---|
| #45: circumcentres in grid steps | step −0.9 to −1.7 % | kept | equal centres must round equally | measurements.md, "After the fix of #45" |
| #48: edge representatives from the tetrahedra, plus the face-count check | step −0.9 % | kept | a doubled face gave wrong cells; the check makes such errors impossible to miss | issue #48 |
