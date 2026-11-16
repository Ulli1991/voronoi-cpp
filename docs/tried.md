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

| Volumes-only output (`volumes_and_centroids`: no face sums, face writes or pool) | device +16 % on blue noise (34.8 → 40.4), +12 to +27 % on jittered, uniform and halo, +1 % clustered; host to host 18.4 → 33.2 (blue noise, 128³); 671 → 101 B/cell; same bits | kept (new option) | the face rows were few bytes but the 1.1 GB face pool, its allocation and the host mirrors dominated host-to-host time; the walk and cuts are unchanged | measurements.md, "Volumes-only output and the CPU path"; jobs 32368078–80 |
| Pinned host buffers for the volumes-only copies | 128³: within the run spread (31 to 37 against 33 to 35); 256³: +5 to +11 % | option in the bench only; tables use pageable memory | the copies are 56 B/cell, a few ms at 128³; pinned memory pays only once the copies are a larger share | same |

## CPU path

| Change | Effect | Kept? | Reason | Where |
|---|---|---|---|---|
| Parallel stable radix sort of the host cell list (was `std::sort` on one thread) | cell list 370–450 → 11 ms; build 2.2 → 3.6 M cells/s on blue noise (72 cores); same slot order and bits | kept | the sort was serial while everything else used 72 threads | measurements.md, "Volumes-only output and the CPU path"; jobs 32361898, 32362120 |
| Dynamic chunks of 16 cells in the host thread kernel | +7 % (blue noise) to +17 % (halo) | kept (host backends only; CUDA unchanged) | cell costs vary, static blocks leave threads idle at the end | same; job 32362198 |
| Dynamic chunks of 128 | about equal to 16 | not chosen | load balance is already even at 16 | job 32362198 |
| Close instead of spread binding | 0 to +3 % (within noise) | not changed | the kernel is not bound by where threads sit | same |
| 144 hyperthreads instead of 72 cores | −18 % (blue noise) to 0 % | dropped | clipping is compute-bound per core; a second thread per core only competes for it | same |

## Surface input, issue #60 (2026-11-15)

Branch `fix-60` against `new-inputs` `c4e6249`, one A100 at 1305 MHz, the same gpu1 job each; `scripts/issue60_a100.sh`, logs in `build-jobs/logs/` of the worktree. Surface rates are of `voronoi_bench rebuild` (warp rebuild) or the library bench's exact start, M cells/s.

| Change | Effect | Kept? | Reason | Where |
|---|---|---|---|---|
| Cells a rounded build leaves neither ok nor out of pool room built again exactly (`exact_where_failed`) | surface refused cells 476 → 0 (128³), 69 → 0 (64³), 1 → 0 (32³ OpenMP; the 953-point case); other inputs: no such cells, same bits | kept | every refusal was a rounded cut: a coarser level re-reads a grid cell the finer walk did not cover and clips a particle a second time; in a needle cell the vertices of its face, made by nearly parallel planes, round past 64 ε \|d\| R, so the twin plane "cuts" and meets its twin nowhere (thread kernel: det 0). The exact cell decides by sign | how_it_works.md §3; clipping_test |
| Base-level warp walk's uncertified cells to a warp kernel over a list, not the thread kernel; rings kernel without levels on base-level input | surface 64³ 0.129 → 0.35 | kept | 74 % of surface cells (64³) left the base-level kernel uncertified and went to one thread per cell: the cell list puts surface points at the base level (64 of them fill a node of mean density), so the shortcut for uniform input took them | measure job 32371403 |
| Clip budget (4096) with a filtered warp kernel: cylinder bound on grid cells and particles, vertex-ball test per particle | surface 64³ 0.35 → 0.59 with a budget of 256, 0.34 with 4096; ball test alone: no change | kept, budget 4096 | the filter removes 85 to 95 % of the clips, but the time went to the thread kernel's overflow cells (phase timers: 0.5 of 0.8 s at 64³, 10.7 of 14.5 s at 128³) | jobs 32371732, 32372787 |
| Plane compaction in the filtered warp kernel; its overflows stay deferred and go filtered to the thread kernel's big capacity, skipping the small one | surface 64³ 0.33 → 0.45, 128³ 0.10 → 0.12 | kept | the warp kernel never dropped planes that stopped being faces; needle cells collect them. Vertex overflow (more than 64 live vertices, 2.5 % of 128³ cells) still goes to one thread per cell | jobs 32372958, 32373139 |
| Budget checked every batch in the warp and thread kernels | other inputs: rebuild −0.5 to −1.5 %, exact start of lattice and wave −9 to −10 % | dropped | a compare per batch in the hot loops changes their code; the exact thread kernel rebuilds every lattice cell in the start | job 32373656 |
| Budget at level changes in the warp kernel and in exact thread cells, every batch in rounded thread cells; filtered thread pass as its own template | other inputs: rebuild −0.1 to −0.5 %, exact start −1 to +3 % (noise ±3 %), lattice 128³ +0.8 %, same bits; surface 128³ 0.12 → 0.10 | kept | a needle in the adaptive warp kernel now stops only after its first level; a check every batch in exact cells cost the lattice start 10 % (job 32374842) | jobs 32374842 to 32374852, 32375358 |
| Filtered warp kernel with 128 vertex and plane slots (4 blocks per SM) for the filtered kernel's cells past 64 (variant B) | surface 64³ 0.41 → 0.51, 128³ 0.100 → 0.104, 256³ 0.041 → 0.045; other inputs same bits, rebuild −0.07 to −0.73 % | kept | it takes 12,173 cells at 128³ off the thread kernel; the 121,189 overflows of the unfiltered warp kernel still go there and are now most of the time | job 32406803 |
| Variant B2: the unfiltered warp kernel's overflows to the filtered kernels too | surface 64³ / 128³ / 256³ 1.01 / 0.29 / 0.100; cosmic web, halo, clustered rebuild +1.5 to +5.7 %; bits change on uniform, clustered, halo, cosmic web; `clipping_test` fails | dropped (bits; the owner may choose it) | the filtered kernel clips in another order, so the overflowing cells round differently; volumes-only and full rebuilds then differ in a few cells | 4497a6c; jobs 32378800 to 32378817 |
| Budget 256 instead of 4096 | surface 128³ 0.10 → 0.16; halo and cosmic web rebuild −5 to −6 %, exact start −4 to −11 %, bits differ on halo and cosmic web | not adopted (owner's call) | their tail cells (p99 210 to 268 clips) pass 256 and restart filtered | jobs 32374384, 32374385 |

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
| `Layout::float_centroids`: face centroids stored as float (relative to the owner), areas double; face row 56 → 44 B | job 32373869, 128³ step 3: 3300 → 3096 B/cell at 0.03 spacings (`faces` −95, `cell_sums` −110), 5247 → 5036 at 0.1; `update()` + `update_cells()` +1.8 % (162.8 → 165.8 M cells/s) at 0.03, +1.1 % at 0.1, +0.7 % at 0.3; 256³ +1.9 %, +1.6 %, +1.3 %; step peak memory 1979 → 1881 B/cell; areas, volumes and topology bit for bit; cell centroids within 8e-9 spacings of double (mean 1.8e-9) | kept, opt-in, off by default; default is the owner's decision | closure and volumes use only the areas, which stay double; a float centroid relative to the owner rounds at 6e-8 of a face's offset, below every error the method has | measurements.md (Float face centroids) |
| Hydro face rows with float centroids (offsets from the owner), double areas, 44 instead of 56 B a face (luna-cpp `-DLUNA_FACE_ROWS=1`, repacked once per mesh) | luna-cpp Sedov 128³, 1410 MHz (job 32372277): moving hydro +2.6 % `grad32` (122.1 → 125.3 M cells/s), +2.0 % double; static +4.2 %, +4.3 %; moving step equal (repack 0.43 ms); hydro 809 → 779 B a face; Sedov, sound wave and vortex norms equal to 4–5 digits (job 32373383) | luna-cpp study (branch `face-rows`), adoption is a later luna-cpp task | closure stays exact with double areas; the centroid feeds only the limiter and the face velocity, where 1e-8 of a face is below the method's error; the gain is small because the gathers are sector-bound, not row-bound | luna-cpp docs/measurements.md, Face rows |
| Hydro face rows as one 48-byte row (double area, float centroid, owner, neighbour) | luna-cpp study: moving hydro −12 % `grad32`, static −16 %; hydro 1008 B a face; `face_step` 29 → 73 B a face | dropped | a kernel that needs 8 bytes of the row reads its whole sector, and the sweep's gathers in cell order touch more sectors | same |
| Mesh geometry unfused on the device (`tet_mesh::rounded`: circumcentres, faces, cell sums, 2D centres; clipping still fused) | CUDA bits equal the host's (5 inputs at 128³; `repro.serial` on CUDA); steps +3.2 to +4.9 / +2.2 to +4.3 / +1.3 to +2.7 % at 0.03 / 0.1 / 0.3 spacings (128³, 256³), rebuild 0, exact start +0.4 ms at 128³; accuracy unchanged; 19 to 21 % of CUDA volumes change by at most 8e-16 relative against main 8cb1475 | kept on branch `unfused-on-main`; the owner decides on main | the fused a * b + c rounds once where the host rounds twice; unfused, the cell sums (`update_cells`, +5 to +11 %) lose a DFMA per component of each pyramid's moment | measurements.md, "Unfused mesh geometry"; TESTING.md, "Bits across backends" |

## Correctness fixes that cost speed

| Change | Effect | Kept? | Reason | Where |
|---|---|---|---|---|
| #45: circumcentres in grid steps | step −0.9 to −1.7 % | kept | equal centres must round equally | measurements.md, "After the fix of #45" |
| #48: edge representatives from the tetrahedra, plus the face-count check | step −0.9 % | kept | a doubled face gave wrong cells; the check makes such errors impossible to miss | issue #48 |

## CPU and GPU together (measured, not built)

Upper bounds from Nsight traces (measurements.md, "CPU and GPU together: where the GPU waits"); none of these was implemented.

| Change | Effect | Kept? | Reason | Where |
|---|---|---|---|---|
| CPU finishes stuck flips while the GPU goes on (H5) | 0: no triangle stuck in 240 steps, six inputs, 0.03 to 0.3 spacings, 128³ and 256³ | dropped | nothing to hide; the host fallback stays as the rare path it is | ideas.md, "CPU and GPU together" |
| CPU runs the start's patch rounds while the GPU writes faces (H6) | 0 | dropped | the faces are computed from the patched tetrahedra, so the two are sequential by data; the re-clipping inside the rounds is the tail (H2) | same |
| Hilbert re-sort keys on the CPU (H7) | below 0: the positions copy (about 12 ms at 128³) costs more than the key kernel | dropped | the keys need every position on the host; the re-sort is off by default | same |
| CPU takes a share of bulk cells (H8) | at most 7 % of a rebuild before copies; the positions copy alone is about 10 % of a jittered clipping | dropped | a 72-core node is 13× slower clipping and 17× slower stepping than the A100 with the same code; a step would need the mesh mirrored on the host | same |

## CPU and GPU from one build

| Change | Effect | Kept? | Reason | Where |
|---|---|---|---|---|
| One build on Kokkos CUDA + OpenMP instead of CUDA + Serial (S0) | rebuild, exact start and steps 0.993 to 1.009 of the CUDA-only build (28 cases); compile time the same, binary +0.5 % | kept (option of the Kokkos install; the library needs no change) | only the host space changes; device kernels are the same code | measurements.md, "CPU and GPU from one build" |
| Device arithmetic unfused (`-DVORONOI_FMAD=OFF`) | rebuild +4.6 to +11.3 %, start +1.4 to +5.5 %, steps −1.5 to +0.9 %; CPU and GPU meshes equal in every bit (default: 21 % of volumes differ by at most 8e-16 relative) | option, off by default; the owner decides | the clipping is dense multiply-add arithmetic (plane cuts, vertex and volume sums) and DFMA halves its instructions; the step's geometry is bound by memory | same; TESTING.md, "Bits across backends" |
| Mesh geometry unfused on the device, option (c) (`tet_mesh::rounded` in circumcentres, faces, cell sums, 2D centres; clipping still fused) | CPU and GPU meshes equal in every bit with the default build; on main's code steps +1.3 to +4.9 %, rebuild 0, accuracy unchanged | kept (on main: the owner decides) | the cell sums lose a DFMA per moment component of each pyramid | measurements.md, "Unfused mesh geometry"; branch `unfused-on-main` |
