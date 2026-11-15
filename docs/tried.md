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
| Overflowed warp cells straight to big thread cells (skip the 48/64 thread pass they overflow again) | rebuild 128³: halo +5.3 %, clustered +1.2 %, uniform +1.2 %, jittered, lattice, wave, blue noise −0.1 to 0 %; 256³: 0 to +0.7 %; start within its 5–8 % spread | kept | the thread pass for the 33–261 cells the warp gives up was twice the work it needed; it is still the time of its slowest cell on one thread (10 ms of 129 on halo) | measurements.md, "Hard cells on clustered and halo input, and #49"; `cbfd8b3` |
| Queue of hard cells (below the base level) first, their overflows on a concurrent stream | halo −6 %, clustered −2 % against the row above | dropped | the persistent warp kernel holds every SM slot until it ends, so the overlapped thread pass cannot start; a third of the halo overflows are on finer levels anyway; the tail is under 1 % of the warp cycles | same; `f6535fc` |
| Overflowing rounded cells on the host | clustered and halo 2.1–2.3× slower; other last bits | dropped | copies to and from the host, and the host's rounding differs from the device's | same |
| A larger-capacity warp kernel (or a block) for big cells | not built | — | the thread and warp kernels give different bits, so the overflowing cells of uniform, clustered and halo input would change against main; it is what #49's inputs with most cells overflowing need | same |
| Another choice of the adaptive grid level | not built | — | the level decides which particles share a batch, so the cut order and last bits | same |
| Exact huge cells on every host core (#49) | a fuzz case with 5525 huge cells: over 830 s → 110 s (18 cores) | kept | the CUDA build's host space is Serial, so they ran on one core; now one Serial instance per core, same bits | same |

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

## Exact start (`tet_start::build`)

Kernel times fenced per kernel (`--kernels`), 128³ jittered unless stated, against main c42c230 in the same job; job ids in [measurements.md](measurements.md#the-exact-start-assembly-from-the-ring-blocks-and-an-incremental-patch-round-2026-11-15).

| Change | Effect | Kept? | Reason | Where |
|---|---|---|---|---|
| Lookups read the clipping's ring blocks; no face-ordered copy of the rings | `face_particles`, `face_lengths`, `first_tet` and the scans (12.7 ms) gone | kept | the copy moved 2353 B/cell for 167 B of rings; the blocks hold the same data in place | `tet_mesh::assemble` |
| One thread per ring slot for emit and lookups, first try (per-slot face map, modulo wraps, counting earlier faces) | emit 15.7 → 20.1 ms | changed | integer division and re-counting the owner's earlier faces per lookup cost more than the divergence saved | job 32370011 |
| The same with `after()` wraps and the face's first tetrahedron read, not counted | emit 15.7 → 5.0 ms | kept | one lookup per thread, 14 of 32 lanes busy against 9.8 | job 32370305 |
| One thread per tetrahedron for the adjacency | 13.0 → 3.0 ms | kept | main ran one thread per face, 4.9 lanes busy | same |
| Face arrays by one thread per cell | 6.6 ms, 2210 + 1392 B/cell at 74 % of DRAM peak | changed | uncoalesced writes of six arrays: partial sectors read back from DRAM | ncu job 32370306 |
| Face arrays by one thread per face (owner by bisection) | 6.6 → 2.6 ms | kept | coalesced writes | job 32370596 |
| Check by the double filters, undecided tetrahedra by the exact predicates in a second kernel | 5.1 → 3.2 ms; lattice: 8.2 → 2.8 + 7.5 (all ties go exact) | kept | 91 registers instead of 168: 12 → about 25 % occupancy | same |
| A lookup reads one 32-byte cell record (block, faces, first face, slot, tetrahedron) | not measured on its own | kept | one memory sector per lookup instead of three | |
| Per-slot face map against bisection on `ring_start` | map: emit 5.0 ms; bisection 6.6 ms | bisection kept | the map costs 167 B/cell at the start's peak; owner's call | job 32373386 |
| #59: incremental round after a partial re-clip | 256³ jittered: second round 407 → 108 ms of kernels (main → branch), the first 381 → 129 | kept | only cells naming a re-clipped particle look up again; the passes over all cells, faces, slots and tetrahedra that shift the rest (about 70 ms at 256³) remain | nsys job 32372190 |
| Freeing the old mesh and temporaries before allocating | `cudaMalloc` 534 → 297 ms in one 256³ start | changed | a large `cudaMalloc` right after large frees stalls (76 to 103 ms each) | nsys jobs 32371022, 32371295 |
| Storage kept from round to round (two sets for the incremental round) | host gap of one 256³ start 559 → 126 ms | kept | no frees between rounds | nsys job 32371594 |
| Slack of 1/16 on kept storage, check lists of their own | start peak through `Mesh::update` 1955 → 2059 B/cell (256³) | changed to 1/64 and lists in idle assembly storage | the peak equals main's again (1955.2) | memory job 32373385 |

## Precision

| Change | Effect | Kept? | Reason | Where |
|---|---|---|---|---|
| Float face rows as default | breaks cell closure (2–4e-8); small-amplitude waves 4–770× worse | rejected as default | closure must be exact for conservation | measure_2026-11-12.md, part 2 |
| Float circumcentres as offsets from a corner | geometry error 4e-8 to 1e-7 of h²; closure exact | yes geometrically; not yet adopted | needs a hydro check | same |
| Float circumcentres in absolute coordinates; float face arithmetic | error grows with N; closure 1e-7 | rejected | precision lost | same |

## Correctness fixes that cost speed

| Change | Effect | Kept? | Reason | Where |
|---|---|---|---|---|
| #45: circumcentres in grid steps | step −0.9 to −1.7 % | kept | equal centres must round equally | measurements.md, "After the fix of #45" |
| #48: edge representatives from the tetrahedra, plus the face-count check | step −0.9 % | kept | a doubled face gave wrong cells; the check makes such errors impossible to miss | issue #48 |
