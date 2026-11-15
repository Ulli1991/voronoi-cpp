# Caption drafts (facts the captions must state)

**Figs. 1–2, walk-through.** Panel texts: `walkthrough_captions.md`. Every panel is computed by the library's 2D mesh
(`Mesh<2>`); 2D stand-ins for 3D as listed in `PLAN.md`.

**Library headline (full cells) and library_more.** One A100 40 GB (node ravg1002, SM clock 1095 MHz under load in
every job of these rows; voronoi-cpp rows are commit ad62997 = main b2faee8 + the benchmark, which runs 10–15 % slower
than main 55cd6b1 at 1305 MHz), double precision, 2,097,152 cells (128³), periodic unit box, median of 3 timed runs
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

**tried_key.** Rate change against the code just before the change, in the same job, one A100 at 1305 MHz unless the
row's source says otherwise, 128³ unless stated; from `docs/tried.md` (full list: appendix table `tried.md`).

**oracle_errors.** Against the exact reference `src/test/oracle.hpp`: 128-bit integer predicates and long-double
clipping over all periodic images, brute force per sampled cell; it includes only the C++ standard library and shares
no code with the mesh (its driver shares the point-file reader). 1000 sampled cells per input.
