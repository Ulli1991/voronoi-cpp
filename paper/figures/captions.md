# Caption drafts (facts the captions must state)

**walkthrough (Fig. 1, full page).** One periodic 2D example, computed by the library's own 2D mesh (`Mesh<2>`),
which shares the 3D mesh's structure. Rows 1–2, *Build*: **(1) Quantise positions.** Each coordinate is rounded down to an integer k in [0, 2³⁰), and the position is k × step: positions are quantised to integer coordinates on a 2³⁰ grid (inset: one grid cell, much enlarged). Every later decision is an exact test on these integer positions; exact ties break by particle id. **(2) Morton cell list.** Sorted along a Morton curve (line, slot 0 to 95): each tree node is one run of slots. Each particle's level has about one particle per cell: 4 (16²) in the patch, 3 (8²) around. **(3) Clip one cell.** From a square of half-width L/4, cut by bisectors nearest first (numbers). The walk stops at the first grid cell 2R away (dashed), R the farthest corner: 7 cuts, 1 miss (hollow), 87 never read. **(4) All cells: face rings.** Every cell is clipped on its own, in parallel, and keeps its ring: its neighbours in order (q1 … q7 here), in one pool. The rings are the whole topology; no corner is stored. **(5) Exact start.** Consecutive ring entries give triangles (i, qₘ, qₘ₊₁), each once: 192 = 2n, with 3n = 288 faces. Checked: the exact screen and flips change 0 of them; the areas add up to the box once. **(6) Points move.** The solver calls update(x) with new positions, here up to 0.25 mean spacings (arrows ×3). The triangles are kept (grey); a step needs no cell list and no clipping. Rows 3–4, *Step*: **(7) Screen.** One pass over the old triangles at the new points: the in-circle test of each edge in double, with a rounding bound. Failing, unsure, and (in so small a box) long edges are listed: 42 of 288. **(8) Flip.** The exact test decides each listed edge. Flips (dashed: gone, green: new) run in rounds; votes settle conflicts, so thread order is moot. 13 flips in 6 rounds; volumes equal a fresh build's. **(9) Geometry: the dual.** Circumcentres (blue dots) are the cell corners: the Voronoi mesh is the dual of the triangles. One thread per face writes its area vector (arrows) and centroid (orange). **(10) update_cells().** Per cell, a list of its faces sorted by the other particle's id (1 … 7). Volume and centroid (×) are summed over the pyramids in that order: the same bits on every run. Σ volume = L² exactly here. **(11) Fallbacks.** Two particles jump (arrows); flips leave 1 inverted triangle (red). Few stuck: the host redoes a patch (11 triangles), the GPU flips again. Many: a rebuild (1–5). Two at one integer position: refused. **(12) Re-sort (3D).** Over many steps ids drift apart in space (grey: id order). Renumbering along a Hilbert curve (colour) restores locality: the mean id gap over faces is 13 at the start, 24 now, 13 after. Badges: where each part
runs (GPU, host, or the caller). In 3D the same: panels 1, 6 and 10; 2D stand-ins for 3D: a quadtree for the octree
(2), a polygon cut by bisector lines for a polyhedron cut by bisector planes, the 2R circle for a sphere (3), one
neighbour ring per cell for one ring per face (4), triangles for tetrahedra (5), in-circle tests and edge flips for
in-sphere tests and 2-3/3-2 flips (7, 8), polygon edges for polygonal faces (9), an inverted triangle for an inverted
tetrahedron (11); panel 12 shows in 2D a step only the 3D mesh takes. The code behind each panel:
`walkthrough_captions.md`.

**inputs.** The five benchmark inputs (top row and first panel of the bottom row) and the three application inputs
(bottom row: cosmic web, fcc solid, surface), with the generators and seed 20261029 of the benchmarks
(`src/test/support.hpp`, positions quantised to integer coordinates on a 2³⁰ grid; `library_bench points`), at 64³ = 262,144 points; the fcc solid at
32³ unit cells = 131,072 atoms (the generator rounds n to 4 l³, and 2¹⁸ atoms would need l = 40.3). The benchmarks run
at 128³ and 256³ (fcc solid: 2²⁰ and 2²³ atoms). Each panel: the points of a slab one mean spacing thick, projected to
x–y; jittered, blue noise, white noise and the fcc solid as a zoom of 16 × 16 mean spacings, the other four as the whole
box; clustered and halo shifted periodically so that the most isolated dense blob or halo sits in the middle, with the
slab through its centre; the fcc slab centred on an atomic (001) plane, the surface slab through the centres of four
shells; halo points drawn at a third of the common marker area and half transparent: at the common size the whole halo (12,115 slab points) is solid ink and its core does not show. Jittered: a cubic lattice with uniform shifts
up to ±0.1 spacing per axis. Blue noise: the uniform points after 100 Lloyd iterations (`library_bench lloyd`). White
noise: independent uniform points. Clustered: 8 Gaussian blobs, σ = 0.04 of the box, holding 80 % of the points, over
20 % uniform; nominal peak-to-background 1:500. Halo: 4 profiles ∝ 1/(r² + c²), c = 0.002, out to r = 0.2, holding 90 %
of the points, over 10 % uniform; 1:10⁴ from core to edge within a halo. Cosmic web: Zel'dovich positions x = q + D ψ(q)
of a cubic lattice at growth factor D = 0.5, ψ from a Gaussian field with the BBKS CDM spectrum (Bardeen et al. 1986;
Γ = 0.21, n_s = 0.96, σ₈ = 0.8 at D = 1, 100 Mpc/h box), all modes below the lattice's Nyquist number: filaments, walls
and voids, nominal contrast about 1:40 on a 2-spacing grid (at 128³). Fcc solid: an fcc lattice with independent
Gaussian displacements of rms 0.0499 lattice constants per axis, the rms of a Lennard-Jones solid near melting
(ρ* = 1.05, T* = 1.0; Lindemann ratio 0.12). Surface: 16 spherical shells of radius 0.2 on a bcc lattice of side 0.5,
each around a torus of radii 0.1 and 0.05 (the inner circles), points uniform in area and moved along the normal by
0.1 rms surface spacings. The ratio under each panel is the measured density contrast: the 99.9th over the 0.1th
percentile of 1/r₃₂³ (r₃₂ the distance to the 32nd neighbour), which also counts the background and Poisson scatter;
for the surface it measures only the density along the shells (the space between them holds no points at all).
Each input has one colour, used for its points, its curve and the same input in every other figure. Bottom: the
nearest-neighbour distances of all eight inputs, as the fraction of points per bin (40 bins per decade) against the
distance in mean spacings (n^(−1/3)); dashed: the three application inputs; dotted: one mean spacing. Two degenerate test cases are not shown: an exact cubic lattice (every point
cospherical with its neighbours) and the lattice shifted by 10⁻⁵ spacing ("wave").

**application.** Completeness against build rate on the three application inputs, one point per code at 128³
(fcc solid: 2²⁰ atoms), double precision but where marked float, periodic unit box, median of 3 timed runs, full cells
but for the volumes-only codes (Basselin et al., Ray et al., Liu et al.; their README setting per input class, as in the
quality figure: white noise for cosmic web and surface, perturbed grid for the fcc solid; Ray et al. = VoroGPU, K = 190
on cosmic web and surface, K = 90 on the fcc solid; Liu et al. tet-centric, K 90 (PRESET 0 and 1); Basselin et al.
K = 70 and 50). Completeness: cells missing or with non-positive
volume, of all n; band: no cell missing. Colour: the code; circles: one A100, diamonds: 72 cores (votess runs on both). Complete codes are spread over the band,
each at a fixed height, so that none hides another. Open markers: wrong volumes, i.e. a sampled cell that has a volume is off by
more than 1 % from the exact reference (1000 sampled cells), or the cells cover more than the box (Σ V − box > 10⁻³):
peclet-voro on the surface (Σ V − box = +54), Paragram + faces on the surface (22 sampled cells off by more than 1 %,
up to 76 %), gDel3D + faces on the surface (its padding of 8 spacings is too thin: one sampled cell off by 1.3 %), Liu
et al. on cosmic web and surface (besides the cells missing, cells that get a volume are off by up to 6 % and 22 %). Crashed, no
volumes: votess + faces on the A100 on cosmic web and surface (CUDA error 700), votess + faces on 72 cores on the surface
(`std::length_error`). voronoi-cpp: 631af46 (library as on main c42c230), one A100 at 1305 MHz, jobs 32367524,
32367883, 32367884. The surface rows of voronoi-cpp are marked "#60 open": clipping refuses 476 cells on the surface
(the rebuild's missing cells) and runs about 300 times slower than on jittered points, since its needle cells reach about
0.1 box into empty space and clip about 6000 neighbours each instead of 75; the exact start inherits the cost (it clips
first) but ends complete and exact. Final surface numbers come after the fix. Other A100 codes: chains 32367907 to
32367911 and 32370411 (nodes ravg1131, ravg1162, ravg1166); voronoi-cpp's warp rebuild ran in each as a same-GPU
reference at 10.3 to 12.8 M cells/s on cosmic web (12.0 in our own job) and 17 to 31 on the fcc solid (30.6). "72 cores": one node of 2 × Xeon Platinum 8360Y,
72 threads, jobs 32367538 to 32367544. Exact-reference checks: jobs 32369335 and
32370726. Not shown: peclet-voro runs out of memory at 256³ and gDel3D cannot index 256³.

**Library headline (full cells) and library_more.** One A100 40 GB. voronoi-cpp: main 55cd6b1 (library code as on main
c42c230), 1305 MHz SM clock under load, node ravg1034, job 32366449. Other GPU codes: node ravg1002 at 1095 MHz (the
benchmark's jobs, docs/library_benchmark.md); their rates at 1305 MHz would be up to about 15 % higher, double precision, 2,097,152 cells (128³), periodic unit box, median of 3 timed runs
(steps: frames 2 to 10). "+ faces": the code's tetrahedra or neighbour lists turned into full cells by our faces step,
timed in. Only measured rows are drawn (every code on the same points and hardware); the rates the papers report on
their own hardware and points are in the table "Published rates" below, for the text.

**library_more, additions (close-gaps, voronoi-cpp 77055a4).** (a) "ours, full cells, host to host": the warp rebuild with volumes and centroids copied out (faces stay on the
device). Star: our volumes-only path
(`shell_voronoi::volumes_and_centroids`), host to host with pageable buffers like the three volumes-only codes, blue
noise only: 33.2 M cells/s in A100 job 32368078 (node ravg1004), which also ran Basselin et al. (10.4), Ray et al.
(18.1) and Liu et al. (18.1) on blue noise in their README setting; the ratios printed are from that job alone (3.2×,
1.8×, 1.8×). With pinned host buffers 36.1, on the device 40.2. The curves are the benchmark's rows as before.
(b) voronoi-cpp's warp rebuild and voro++ (multithreaded, Lu, Lazar and Rycroft 2023) from one CPU job on one node (32367276, ravc4079, 72 threads,
77055a4): ours 3.4 / 3.8 / 2.6 / 2.0 / 2.0 against voro++ 2.4 / 2.5 / 1.8 / 0.38 / 0.085 M cells/s on jittered, blue
noise, white noise, clustered, halo; the other codes from the benchmark's CPU jobs (other nodes). (c) ours CPU: the
moving step at 72 threads from the same job (8.5 M cells/s, jittered, 0.03 spacings).

**Memory.** (a) Kokkos device arrays at the peak of each phase, jittered input, main 55cd6b1, job 32362454; the 256³
values, measured in the same job, are 655.8, 1955.2 and 1979.2 B/cell (the arrays scale with n: the same B/cell as
at 128³ to 0.1 B but for the rebuild's face pool), so the bars show 128³ only. (b) Process-wide device use sampled
every 200 µs (library benchmark, voronoi-cpp ad62997), except the moving step (Kokkos arrays, 55cd6b1); dark part:
the lowest input, light part: up to the highest of the five inputs. Dashed: the B/cell at which 256³, 384³ and 512³
fill one A100 40 GB. Besides memory, the face-ring pool's 32-bit offsets overflow between 256³ and 384³.

**oracle_errors.** Against the exact reference `src/test/oracle.hpp`: 128-bit integer predicates and long-double
clipping over all periodic images, brute force per sampled cell; it includes only the C++ standard library and shares
no code with the mesh (its driver shares the point-file reader). 1000 sampled cells per input. All at 128³ (2,097,152 cells). (a) voronoi-cpp's
warp rebuild and (b) its exact start, one A100 at 1305 MHz (55cd6b1, job 32366449), per input; (c) every code on halo
input, voronoi-cpp's warp rebuild included. Left of the axis break: cells whose volume equals the exact reference's
(error 0). Cells without a volume are not in the histograms; on halo: peclet-voro 33,819, Basselin et al. 2,023 and
Ray et al. (VoroGPU, float, K = 190) 21,471 of 2,097,152 cells; the other codes none.

**throughput_vs_n, step_vs_displacement, resort, stages.** One A100 40 GB at 1305 MHz SM clock (every job's log), main
55cd6b1, double precision, periodic unit box. Rates are cells over wall time of the timed region, median of 5 runs
(rebuild, exact start) or of steps 2 to 10 (steps: points moved by the bench's smooth flow, `update` + `update_cells`).
Wave at 256³ runs out of the 40 GB in the step. Re-sort: 500 steps at 0.1 spacings, jittered, Hilbert order every 50
updates; the legend gives cells × steps over the summed step times, re-sorts included.

**quality.** Build rates at 128³ as in the library figures; error against the exact reference on 1000 sampled cells
(median to max); completeness over all cells; total sampled volume |Σ V − box| (a tiling check, not accuracy).
Neighbour-set agreement and closure are not shown: the benchmark runs stored volumes only.

**Note for throughput_vs_n (b), #59 fixed (not yet redrawn).** On branch start-fast (to be merged), the exact start
builds the same arrays 1.07 to 1.74× faster; at 256³ jittered it goes from 12.7 to 22.2 M cells/s (start-fast's own
measurement against main c42c230; README of close-gaps). The figure keeps 55cd6b1 until the full sweep reruns after
the merges; the note below explains the old dip.

**Note for throughput_vs_n (b).** The exact start of jittered input slows from 128³ to 256³ (9.5 → 6.6 M cells/s) because
at 256³ its check finds 44 bad tetrahedra and patches 830 cells in a second round (job 32366447: clipping with rings
455 ms, from_rings 378 ms, start 1333 ms in all, so about 500 ms, 30 ns a cell, for the second round); at 128³ the start
has one round, no bad tetrahedra, and costs clipping 57 + from_rings 50 ≈ start 117 ms. Per cell, clipping (27 ns)
and from_rings (23 ns) are the same at both sizes.

**library_stages.** Where each code's build time goes, ns per cell at 128³ (2,097,152 cells) on jittered and white-noise
input. Each code's staged runs repeat its timed runs with a lap at the end of each stage (device synchronized first) and,
inside a library call, the code's own timers or its kernels by label (method per code: docs/library_benchmark.md,
"Stages", branch library-stages). Markers on one log axis: neighbour search (cell list, kNN grid and query, BVH,
spatial sort), cell kernel with its relaunches, exact Delaunay start (ours only), faces (the code's own or our faces
step), copies host ↔ device, and the rest (preparation, padding with periodic images, time between laps); stages under
0.05 ns are not drawn; markers sit above or below the row's centre so equal times both show. Bar: the plain (unstaged)
runs' median total. kNN codes in their README setting (jittered: perturbed grid, white noise: white noise). gDel3D's
own phase timers are inconsistent (init reads about 1.4·10⁵ s, star splaying about 1.9 s in runs of 0.5 to 1.4 s),
so only its total is shown. One A100 at 1305 MHz, job 32368965 (node ravg1013; voronoi-cpp de57e65+local); one CPU
node, 72 threads, job 32368966 (ravl1045; voronoi-cpp d06ddc4). Double precision but Ray et al. and Liu et al. (float).
Our two rows split the cell list differently, so their neighbour searches do not compare: the rebuild calls the
cell list itself and its lap holds the whole call (configure, Morton keys, radix sort, slots, levels: 114 ns per cell on
72 cores, 0.6 ns on the A100), while inside `Mesh::update` only the cell-list kernels are labelled (keys, slots, levels,
positions: 2.4 ns on 72 cores); the sort and the host work around them fall outside labelled kernels and are counted in
the exact Delaunay start. Our rebuild's rest on the A100 (1.4 ns) is the cell arrays' allocation (0.5) and the time
between laps (0.9); it has no copies because it is timed from points on the device to cells on the device.

**Published rates (for the text; not drawn).** As the papers report them, on their own hardware and points; from
`bench/libraries/published.csv`, where precision, outputs, timing and source locations are recorded per row.

| code | reference | hardware | points | distribution | precision | outputs | M cells/s | source |
|---|---|---|---|---|---|---|---|---|
| peclet-voro (cold build) | peters2026peclet | GeForce RTX 5080 | 2·10⁶ | uniform Poisson | double | topology, volumes, facet areas | 3.61 | repository docs |
| Basselin et al. 2021 | basselin2021restricted | Tesla V100 | 10⁷ | blue noise | double | volumes and barycentres | 34.5 | CGF 40(2), Fig. 10 (top) |
| Basselin et al. 2021 | basselin2021restricted | Tesla V100 | 10⁷ | white noise | double | volumes and barycentres | 4.65 | CGF 40(2), Fig. 10 (bottom) |
| Ray et al. 2018 | ray2018meshless | Tesla V100 | 10⁷ | blue noise | not stated | cell geometry | 17.4 | ACM TOG 37(6):265, Table 2 |
| Paragram | taveira2026paragram | RTX 5090 | 2·10⁶ | white noise | float | connectivity | 9.35 | SIGGRAPH '26 (arXiv:2605.06408), suppl. Table 7 |
| votess (GPU) | singh2024votess | A100 40 GB | 10⁶ | uniform | not stated | neighbour lists | 0.85 (read from a bar chart) | arXiv:2412.04514v2, Fig. 1 |
| voro++ multithreaded | lu2023voro | 2 × Xeon E5-2650L v4, 28 threads | 10⁸ | homogeneous random | double | full cells | 1.07 | CPC 291, 108832, Table 4 |

Basselin et al.'s 34.5 matches their cell kernel alone (docs/library_benchmark.md, Stages: 38.4 M cells/s on our A100,
33.3 scaled to a V100), not the full build host to host (10.4 on our A100).

**luna_vortex.** 3D Yee vortex, HLLC Riemann solver, MC limiter; L1 errors of density and velocity against n_x
(n_x³ cells). See PLAN.md (Fig. 13) for the runs.

**memory (b), legend.** Dark bars: voronoi-cpp, grey: the other codes; the darker part is the lowest of the five inputs,
the lighter part up to the highest.

**where_time (Fig. 4, "where the time goes").** (a) The control flow of `Mesh::update()`: a built mesh with the same n
is checked, flipped, reconnected, measured and summed; few stuck flips are finished on the host, more, or a failed
volume check, go to a rebuild (dashed: quantise, cell list, clip, exact start), which refuses with a clear error when
the start is not exact. Box colours are the stages of (b) and (c). (b) Wall time per cell of one moving step
(`update` + `update_cells`, median of steps 2 to 10) in six stages, per input at 0.03, 0.1 and 0.3 mean spacings per
step; the number is the rate in M cells/s (152 M cells/s = 6.6 ns per cell). (c) Jittered input across displacements;
the rows at 1.0 spacings and clustered 0.3, where the host finish and the rebuild occur, come with the post-merge rerun.
One A100 at 1305 MHz, 128³, main 55cd6b1, job 32362452. Stages:
*check*: quantise the new positions and screen every tetrahedron in double for those that may fail;
*flip*: exact in-sphere tests on the listed tetrahedra, then plan, vote and apply 2-3 and 3-2 flips, in rounds;
*reconnect*: number the faces of created tetrahedra and walk their rings again;
*geometry*: circumcentres of created tetrahedra, face areas and centroids, cell volumes and the volume check;
*per-cell results*: `update_cells()`, the per-cell face lists sorted by neighbour and the volume and centroid sums;
*host and other*: wall time no kernel accounts for (host waits, copies, launches), and the host finish and rebuild when
they occur.
