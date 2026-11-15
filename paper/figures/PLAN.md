# Figures of the CPC paper (plan)

"Computer Programs in Physics": voronoi-cpp on its own terms, then a showcase of the moving-mesh code built on it.
Owner's budget: part 1 (the library) at most 6–7 figures, part 2 (luna-cpp) 3–4. Every figure is drawn by one script
in `paper/scripts/` from stored data, so it can be redrawn after each speed merge (`paper/scripts/make_all.sh`).

**Format.** Vector PDF + 600 dpi PNG, 90 mm (column) or 190 mm (page width). LUNA's `scripts/figstyle.py` unchanged;
`paper/scripts/paperfig.py` draws at design size and scales the PDF to the print width, so text prints at 7–9.6 pt.
Curves 2.2 pt at design size, legends inside, no dark versions. Sizes powers of two or 3 × 2^k; M cells/s, B/cell;
captions state the A100's SM clock. Figures draw measured rows only (every code on the same points and hardware); published rates of other codes go into the captions' table with citations and hardware.

**Layout rule for every figure (owner, 2026-11-15).** Legends always inside the axes, in clear empty space next to
related data; room is made by axis limits (headroom on a log axis), by columns, or by moving long details (missing
counts, hardware, clocks, commits) into the caption. A multi-panel figure with repeated entries has one shared legend,
inside the panel with the most room. Legends are compact (`paperfig.LEGEND`: short entries such as "ours", tight
handles and spacing, an opaque white box without an edge so no grid or data line crosses their text). Nothing crosses
text and nothing touches the frame (value labels get a white box where lines pass behind them). Panel titles are short
("(a) rebuild"); details go into the caption. Text at print size is at least 7 pt (fontsize 11 at design size).

## Part 1: the library (6–7 figures)

**Owner's 7-figure plan (2026-11-15).** 1 `walkthrough` (done, one full page); 2 `inputs`; 3 speed against the
libraries (today's Figs. 4 `library_headline` + 5 `library_more`); 4 where the time goes (`library_stages` + the step's
`stages`); 5 quality (Fig. 7 `quality` + `oracle_errors` + `application`, today's 3b); 6 throughput against problem size,
step against displacement, re-sort; 7 memory. Only the walkthrough merge is done now; the rest is combined at the
post-merge rerun. The table below keeps today's figure files.

| Fig. | file | message | data | status |
|---|---|---|---|---|
| 1 | `walkthrough` (panels 1–12, full page) | Rows 1–2, Build: a periodic point set goes through quantisation to integer coordinates on a 2³⁰ grid, the Morton cell list, clipping one cell, all cells with their face rings, the exact start, and the first move. Rows 3–4, Step: screening the old triangles, flips in rounds, the dual's faces, the cells summed in a fixed order, fallbacks to the host or a rebuild, the re-sort. | the library's own 2D arrays (`scripts/overview_mesh2d.cpp`, `Mesh<2>`) | **done**, 190 × 260 mm (`fig_walkthrough.py`): titles and drawings only; panel texts in the caption, code names in `walkthrough_captions.md` |
| 3 | `inputs` (before the benchmark figures) | The five benchmark inputs and the three application inputs (cosmic web, fcc solid, surface), one panel each (a 16-spacing zoom for the even inputs and the fcc solid, the whole box for clustered, halo, cosmic web and surface), with their measured density contrast and nearest-neighbour distances on one log axis; lattice and wave named in the caption as degenerate test cases. | `paper/data/inputs.npz` (`extract_inputs.py` on the 64³ point files of `library_bench points` / `lloyd`, same generators and seed; fcc solid 2¹⁷ atoms, l = 32) | **done**, 190 × 136 mm, two rows of four (`fig_inputs.py`) |
| 3b | `application` (panel set or own figure, next to `quality`) | Completeness against build rate per code on cosmic web, fcc solid and surface: of the codes shown only voronoi-cpp's exact start, CGAL, geogram and voro++ (the last three on 72 CPU cores) are complete and right on all three; the kNN codes lose cells on cosmic web and almost all on the surface; crashes and wrong volumes marked, not dropped; our surface rows marked "#60 open". | `bench/libraries/results` (the "Application inputs" tables of `scripts/library_benchmark.py`; ours 631af46), `accuracy-inputs-*.jsonl`, `paper/data/oracle_errors_application.csv` | **done, surface preliminary**, 190 × 80 mm (`fig_application.py`); redraw the surface after the fix of #60 (rerun `run_voronoi_cpp_a100.sh` on the surface, its oracle check, then `extract_oracle.py` with `ORACLE_ONLY`) |
| 4 | `library_headline` | Full cells on one A100, like for like: voronoi-cpp's rebuild is several times faster than the fastest other GPU code with full cells, more so the less even the input; measured rows only (published rates in a table in captions.md). | `bench/libraries/results/*.jsonl`, `published.csv`; our rows from the paper's rerun `paper/data/raw/libbench-vcpp-32366449.jsonl` (55cd6b1, 1305 MHz, ravg1034) | **done**, 90 × 84 mm (`fig_library.py`); the other codes' GPU rows are at 1095 MHz (ravg1002), as the caption says |
| 5 | `library_more` (appendix or second figure) | (a) volumes-only codes on the A100, (b) one 72-core CPU node, (c) the moving step. | same | **done**, 190 × 78 mm; (a) adds our volumes-only path on blue noise (star, ratios from job 32368078), (b) ours and voro++ from one CPU job (32367276), (c) ours CPU step from that job (close-gaps, 77055a4) |
| 5b | `library_stages` | Where the time goes, per code: on the A100 our cell kernel takes about 23 (jittered) to 44 ns (white noise) a cell with a neighbour search under 1 ns, while the kNN codes spend 20 to 55 % of their time in the neighbour search; on the CPU ours spends about 110–180 ns in the cell list. | `paper/data/raw/libstages-{a100-32368965,cpu-32368966}.jsonl` (branch library-stages, `scripts/library_stages.py`) | **done**, 190 × 120 mm (`fig_library_stages.py`); gDel3D totals only (own timers inconsistent) |
| 4 (new) | `where_time` | Where the time goes: (a) flowchart of `Mesh::update()` with its decisions and the rebuild path; (b) six stages per input at 0.03/0.1/0.3 spacings; (c) jittered across displacements, slots for the host-finish and rebuild rows (1.0 spacings, clustered 0.3; branch fallback-cases). Same names and colours in flowchart and bars; wall time per cell. | `stages.csv`, `kernels.csv` (job 32362452, 55cd6b1); mapping in SOURCES.md | **draft**, 190 × 167 mm (`fig_time.py`); `library_stages` joins it in the 7-figure plan; fallback rows and the rerun after the merges |
| 6 | `memory` | (a) Device B/cell by array role at the peak of the rebuild (701), exact start (1955) and moving step (1979), 128³ jittered, Kokkos arrays; (b) peak device B/cell per code in the library benchmark against the B/cell at which 256³ (2384), 384³ (706) and 512³ (298) fit one A100 40 GB. | `paper/data/memory_by_array.csv` (job 32362454, 55cd6b1), `bench/libraries/results` | **done**, 190 × 80 mm (`fig_memory.py`). Our rows in (b) are process-wide device use (benchmark, ad62997) except the step (Kokkos arrays, 55cd6b1); the ring pool's 32-bit offsets also overflow between 256³ and 384³ |
| 7 | `quality` | Speed against quality, one point per library and input at 128³: (a) relative volume error against the exact reference on 1000 sampled cells (median to max), (b) completeness (fraction of cells missing), (c) total sampled volume (|Σ V − box|, a tiling check). | benchmark rates (`bench/libraries/results`), `paper/data/oracle_errors.csv`, `accuracy-*.jsonl` | **done**, 190 × 148 mm (`fig_quality.py`). Not measurable from what the benchmark runs wrote: exact neighbour sets and closure (only volumes were stored); they need each driver to dump faces or neighbour lists and a rerun of every code |
| 8 (optional) | `oracle_errors` | Relative volume error per cell against the exact reference, per input for voronoi-cpp (warp rebuild, exact start; the 1305 MHz rerun, 128³) and per code on halo input. | `paper/data/oracle_errors.csv`, `oracle_errors_vcpp_rerun.csv` | **done**, 190 × 79 mm |
| 9 (optional) | `throughput_vs_n`, `step_vs_displacement`, `resort` | Throughput against problem size, 32³–256³ on one A100, of rebuild, exact start, step (0.1 spacings, + update_cells) and peak memory on seven inputs; step rate against displacement (0.01 to 1 spacings, 128³); 500 steps with and without a Hilbert re-sort every 50 (128³ 66 → 124, 256³ 88 → 174 M cells/s). | `paper/data/throughput_vs_n.csv` (jobs 32362448–51, 32366446), `raw/long128-32362455.txt`, `raw/long256-32366447.txt`; 55cd6b1, 1305 MHz | **done**: 190 × 119, 90 × 74, 90 × 69 mm; rerun `submit_sweep.sh` after the speed merges |
| 10 (optional) | `stages` (and `ceiling`) | Where a step's time goes: stages per input and spacing, kernels grouped (jittered); the ceiling figure gives the bytes against the 5× and 10× budgets. | `paper/data/stages.csv`, `kernels.csv` (job 32362452, 55cd6b1, 1305 MHz); `ceiling.json` (main 69de1d3) | `stages` **done**, 190 × 90 mm; `ceiling` still on 69de1d3 data (needs an Nsight rerun) |

Long runs with and without re-sort (jobs 32362455, and 32362456 cancelled mid-way; `fig_resort.py`) and memory by array
(job 32362454) ran before the hold; raw output kept, held for a rerun after the speed merges; candidates for an appendix.

Design choices and what was tried (`docs/tried.md`) are for the team; the paper describes the design choices in the text, with no figure or table from that list.

**Caption of the walkthrough, 2D stand-ins.** Every panel is computed by the library's 2D mesh (`Mesh<2>`), which shares the
3D mesh's structure; the code line under each panel names the 3D counterpart. The same in 3D: panel 1 (quantisation, per
coordinate), 6 (the caller moves the points), 10 (update_cells is the same code in 2D and 3D; triangles over edges
stand for pyramids over faces). 2D stand-ins for 3D: 2 (a quadtree cell list for the octree), 3 (a polygon cut by
bisector lines for a polyhedron cut by bisector planes; the 2R circle is a sphere), 4 (one neighbour ring per cell for
one ring per face), 5 (triangles for tetrahedra), 7 and 8 (in-circle tests and edge flips for in-sphere tests and 2-3/3-2
flips), 9 (polygon edges for polygonal faces), 11 (an inverted triangle for an inverted tetrahedron). Panel 12 shows in
2D a step only the 3D mesh takes (the re-sort).

**Terms (owner):** *exact reference* = the per-cell exact computation of `src/test/oracle.hpp`; only errors against it are accuracy. *Total sampled volume* = the sum of the cell volumes against the box: shows the tiling has no gaps or overlaps, not per-cell accuracy. *Closure* = the sum of each cell's area vectors: a per-cell consistency check.

**Fig. 4: measurements needed per library and input, against the exact reference** (jittered, blue noise, white noise, clustered, halo; 128³,
and 256³ where the code runs), against the exact reference (`src/test/oracle.hpp`), the same way for every code:
1. completeness: cells missing or with non-positive volume, of all n (exists for volumes, `accuracy-*.jsonl`);
2. exact neighbour sets: per sampled cell, the code's neighbour set against the exact reference's (needs each code's faces or
   neighbour lists dumped; not applicable to volumes-only codes, said so in the figure);
3. volume error: per-cell relative error on the 1000-cell sample (exists, `paper/data/oracle_errors.csv`);
4. closure (consistency, not accuracy): per cell |Σ area vectors| / Σ |area| (codes with area vectors: voronoi-cpp, voro++, peclet-voro, our faces
   on the Delaunay codes);
5. total sampled volume: Σ V − box volume, a tiling check, not accuracy (exists, `accuracy-*.jsonl`);
6. face agreement: each face seen from both cells, with equal and opposite area vectors and the same centroid.
Speed on the other axis from the same jobs (M cells/s, the output group stated).

**Waiting for the post-merge sweep** (start-fast, close-gaps and the speed branches merged, then `submit_sweep.sh`,
`stages_a100.sh`, our library rows rerun): Fig. 4 `library_headline` (our rows 55cd6b1), Fig. 5 `library_more` (our
volumes-only and CPU rows on 77055a4, the rest of ours on 55cd6b1), Fig. 6 `memory`, Fig. 7 `quality` and Fig. 8
`oracle_errors` (our rows), Fig. 9 `throughput_vs_n` / `step_vs_displacement` / `resort` (#59: exact start at 256³
jittered 12.7 → 22.2 M cells/s on start-fast), Fig. 10 `stages` and `ceiling` (Nsight rerun), `library_stages`
(our rows de57e65/d06ddc4), and the surface points of Fig. 3b `application` (after the fix of #60). Figs. 1–3 do not
depend on speed.

To do at that redraw: `library_headline` breaks each line at a missing point and marks the gap ("crashed": votess on
blue noise, illegal address), and its exact-start rows are pre-start-fast (main now has 25.0 M cells/s jittered at
128³, so they will rise); `library_more` (a) and (b) trim the y axes to the data (all below about 40 M cells/s, the
legend placed to fit).

## Inputs and tests, in plain words (draft for the results section)

| input | what the points look like | why it is in |
|---|---|---|
| jittered | a cubic lattice with each point moved up to ±0.1 spacing per axis | the even, nearly regular case of most simulations; the fastest input |
| blue noise | uniform points after 100 Lloyd iterations: even spacing without a lattice | the standard input of the GPU Voronoi papers (Ray et al. 2018, Basselin et al. 2021) |
| white noise | independent uniform random points | the textbook Poisson–Voronoi case, with close pairs and large voids |
| cosmic web | Zel'dovich positions of a cubic lattice, BBKS CDM spectrum, growth factor D = 0.5; about 1:40 on a 2-spacing grid | filaments, walls and voids, as in cosmological simulations |
| fcc solid | an fcc lattice with thermal Gaussian displacements (Lindemann ratio 0.12, a Lennard-Jones solid near melting); 4 l³ atoms | a molecular-dynamics crystal: near-ties broken only by thermal noise |
| surface | 16 noisy spherical shells, each around a torus; points uniform in area | points on closed surfaces with empty space between: needle cells, the graphics case |
| clustered | 8 Gaussian blobs (σ = 0.04 of the box) holding 80 % of the points over a 20 % uniform background; about 1:500 in density | strong but smooth density contrast, as in galaxies or droplets |
| halo | 4 halos with density ∝ 1/(r² + c²) out to r = 0.2, c = 0.002, holding 90 % of the points over a 10 % background; about 1:10⁴ from core to edge | extreme contrast, as in cosmological haloes; the hardest input |
| lattice (secondary) | an exact cubic lattice | every point cospherical with its neighbours: the worst case for ties and exact predicates |
| wave (secondary) | the lattice with shifts of 10⁻⁵ spacing | near-degenerate: ties broken by tiny perturbations, the hardest case for floating-point codes |

| test | what is timed or checked |
|---|---|
| rebuild | cells from scratch: cell list and clipping of every cell, points on the device to cells (volumes, centroids, faces with areas and neighbours) on the device |
| exact start | the full mesh from scratch: clipping with face rings, exact Delaunay tetrahedra, faces and per-cell lists (`Mesh::update` + `update_cells` on new points) |
| moving step | one time step of a moving mesh: points moved by a smooth flow by 0.03, 0.1 or 0.3 mean spacings, tetrahedra repaired by exact flips, faces and cells recomputed (`update` + `update_cells`) |
| against other libraries | the same point files, box, outputs and hardware for every code, each in its documented best configuration; full cells, volumes only, or topology, timed from points in the code's memory to all outputs |
| accuracy | each cell volume against the exact reference (`src/test/oracle.hpp`, brute force with exact predicates, sharing no code with the mesh) on 1000 sampled cells; checks: the total sampled volume (sum of volumes against the box: no gaps or overlaps) and closure (each cell's area vectors sum to zero) |
| memory peak | the largest device memory during a rebuild, an exact start or a step, by array (Kokkos) and process-wide (sampled), in B/cell |
| fuzz robustness | many random and adversarial inputs (lattices, ties, coincident points, clusters, 3·10⁴ to 1.6·10⁷ points); each must end exact, or be refused with a reason, never wrong |

## Part 2: luna-cpp on the library (4 figures, all five showcases)

Layout drafts (raster tiles of luna-cpp's own figures at 190 mm, `paper/scripts/draft_luna.py`, luna-cpp read only):
`paper/figures/draft_luna_{sedov,kh,vortex,1d}.png`. Final versions are to be redrawn from luna-cpp's stored npz with
`paperfig.py` (the tiles keep luna-cpp's design sizes, so some text prints under 7 pt).

| Fig. | content | luna-cpp source | status |
|---|---|---|---|
| 11 | **Sedov–Taylor in 3D**: (a) density slice with the moving mesh, zoom on the shock moving vs static (b, c); (d) radial n, v, P, T against Sedov–Taylor; (e) shock radius against t | `sedov3d_mesh`, `sedov3d_radial`, `sedov3d_shock` (`make_figures.py`: fig_sedov_shock, fig_sedov_radial) | draft; redraw at 190 mm, radial panels need larger text |
| 12 | **Kelvin–Helmholtz in 2D**: density at t = 0, 1, 2.5, 4, 5, moving vs LUNA MC at N = 2048; the moving mesh zoomed to single cells | `kh_2d_evolution`, `kh_2d_mesh` (fig_kh, fig_kh_density) | draft; redraw |
| 13 | **3D vortex (Yee)**: L1(ρ), L1(v) against n_x; moving mesh (two flux passes, today's scheme), static, LUNA MC; the removed one-pass run left out | `paper/data/luna/` (luna-cpp jobs 31920446, 31922276; LUNA MC from luna-reference-solutions branch yee-vortex) | **done**, `luna_vortex` 190 × 70 mm (`fig_vortex.py`). The runs predate today's library; a rerun with today's luna-cpp and voronoi-cpp is pending. Moving L1(ρ) flattens at 256³ (3.3e-5 against LUNA MC's 1.5e-5) |
| 14 | **1D**: (a) ladder summary, L1 against GPU time for Sod, Shu–Osher, Woodward–Colella; (b) the super-hard tube | `time_to_target_1d` (fig_time), `super_tube_t2_luna` | draft for (a); **(b) pending**: the existing tube figures show LUNA MC against a 2²⁰ reference only, no moving-mesh run; it needs a luna-cpp moving-mesh run of the tube |

Seen in the drafts, for the owner: in 1D the moving mesh needs 3 to 10× more GPU time than LUNA MC and the static
mesh for the same L1 on Sod and Shu–Osher (it catches up only on Woodward–Colella at the finest levels); the vortex
with one pass stalls at first order (two passes converge at second order).

## Data and refresh

`paper/data/SOURCES.md` lists every data file, job id, commit and clock; raw job output in `paper/data/raw/`.

```
bash paper/scripts/build_cuda.sh   # or sbatch; binaries in /ptmp/uli/voronoi-paper/bin/<commit>
bash paper/scripts/submit_sweep.sh /ptmp/uli/voronoi-paper/bin/<commit>
sbatch -o paper/data/raw/<mode>-%j.txt paper/scripts/stages_a100.sh <bin> kernels|memory|long128|long256
python3 paper/scripts/extract_sweep.py; python3 paper/scripts/extract_stages.py
bash paper/scripts/make_all.sh
```

"Weak scaling" and "strong scaling" are kept for the multi-GPU work; on one A100 the figure is throughput against problem size (`throughput_vs_n`).
