# Caption drafts (facts the captions must state)

**Figs. 1–2, walk-through.** Panel texts: `walkthrough_captions.md`. Every panel is computed by the library's 2D mesh
(`Mesh<2>`); 2D stand-ins for 3D as listed in `PLAN.md`.

**inputs.** The five benchmark inputs at 64³ (262,144 points; the benchmarks run at 128³ and 256³ with the same
generators and seed 20261029, `src/test/support.hpp`, snapped to the 2⁻³⁰ grid). Top: the points of a slab one mean
spacing thick, projected to x–y; jittered, blue noise and white noise as a zoom of 16 × 16 mean spacings, clustered and halo
as the whole box, shifted periodically so that the most isolated dense blob or halo sits in the middle, with the slab
through its centre; halo points drawn smaller and half transparent so its core shows. Jittered: a cubic lattice with uniform shifts up to ±0.1 spacing per axis. Blue noise: the uniform points after
100 Lloyd iterations (`library_bench lloyd`). White noise: independent uniform points. Clustered: 8 Gaussian blobs,
σ = 0.04 of the box, holding 80 % of the points, over 20 % uniform; nominal peak-to-background 1:500. Halo: 4 profiles
∝ 1/(r² + c²), c = 0.002, out to r = 0.2, holding 90 % of the points, over 10 % uniform; 1:10⁴ from core to edge within
a halo. The ratio under each panel is the measured density contrast: the 99.9th over the 0.1th percentile of 1/r₃₂³
(r₃₂ the distance to the 32nd neighbour), which also counts the background and Poisson scatter. Bottom: nearest-neighbour
distances in mean spacings (each strip scaled to its peak; dotted: one spacing). Two degenerate test cases are not
shown: an exact cubic lattice (every point cospherical with its neighbours) and the lattice shifted by 10⁻⁵ spacing
("wave").

**Library headline (full cells) and library_more.** One A100 40 GB. voronoi-cpp: main 55cd6b1 (library code as on main
c42c230), 1305 MHz SM clock under load, node ravg1034, job 32366449. Other GPU codes: node ravg1002 at 1095 MHz (the
benchmark's jobs, docs/library_benchmark.md); their rates at 1305 MHz would be up to about 15 % higher, double precision, 2,097,152 cells (128³), periodic unit box, median of 3 timed runs
(steps: frames 2 to 10). "+ faces": the code's tetrahedra or neighbour lists turned into full cells by our faces step,
timed in. Open markers: published rates on other hardware and points (peclet-voro; Basselin et al. 2021; Ray et al.
2018; Paragram; votess; Voro++, Lu et al. 2023), cited per marker. To be regenerated when the close-gaps CPU rows and
our rerun at 1305 MHz exist.

**Memory.** (a) Kokkos device arrays at the peak of each phase, jittered input, main 55cd6b1, job 32362454; the 256³
values, measured in the same job, are 655.8, 1955.2 and 1979.2 B/cell (the arrays scale with n: the same B/cell as
at 128³ to 0.1 B but for the rebuild's face pool), so the bars show 128³ only. (b) Process-wide device use sampled
every 200 µs (library benchmark, voronoi-cpp ad62997), except the moving step (Kokkos arrays, 55cd6b1); dark part:
the lowest input, light part: up to the highest of the five inputs. Dashed: the B/cell at which 256³, 384³ and 512³
fill one A100 40 GB. Besides memory, the face-ring pool's 32-bit offsets overflow between 256³ and 384³.

**oracle_errors.** Against the exact reference `src/test/oracle.hpp`: 128-bit integer predicates and long-double
clipping over all periodic images, brute force per sampled cell; it includes only the C++ standard library and shares
no code with the mesh (its driver shares the point-file reader). 1000 sampled cells per input.

**weak_scaling, step_vs_displacement, resort, stages.** One A100 40 GB at 1305 MHz SM clock (every job's log), main
55cd6b1, double precision, periodic unit box. Rates are cells over wall time of the timed region, median of 5 runs
(rebuild, exact start) or of steps 2 to 10 (steps: points moved by the bench's smooth flow, `update` + `update_cells`).
Wave at 256³ runs out of the 40 GB in the step. Re-sort: 500 steps at 0.1 spacings, jittered, Hilbert order every 50
updates; the legend gives cells × steps over the summed step times, re-sorts included.

**quality.** Build rates at 128³ as in the library figures; error against the exact reference on 1000 sampled cells
(median to max); completeness over all cells; total sampled volume |Σ V − box| (a tiling check, not accuracy).
Neighbour-set agreement and closure are not shown: the benchmark runs stored volumes only.

**Note for weak_scaling (b).** The exact start of jittered input slows from 128³ to 256³ (9.5 → 6.6 M cells/s) because
at 256³ its check finds 44 bad tetrahedra and patches 830 cells in a second round (job 32366447: clipping with rings
455 ms, from_rings 378 ms, start 1333 ms in all, so about 500 ms, 30 ns a cell, for the second round); at 128³ the start
has one round, no bad tetrahedra, and costs clipping 57 + from_rings 50 ≈ start 117 ms. Per cell, clipping (27 ns)
and from_rings (23 ns) are the same at both sizes.
