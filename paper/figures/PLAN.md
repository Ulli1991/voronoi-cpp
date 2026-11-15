# Figures of the CPC paper (plan)

"Computer Programs in Physics": voronoi-cpp on its own terms, then a showcase of the moving-mesh code built on it.
Owner's budget: part 1 (the library) at most 6–7 figures, part 2 (luna-cpp) 3–4. Every figure is drawn by one script
in `paper/scripts/` from stored data, so it can be redrawn after each speed merge (`paper/scripts/make_all.sh`).

**Format.** Vector PDF + 600 dpi PNG, 90 mm (column) or 190 mm (page width). LUNA's `scripts/figstyle.py` unchanged;
`paper/scripts/paperfig.py` draws at design size and scales the PDF to the print width, so text prints at 7–9.6 pt.
Curves 2.2 pt at design size, legends inside, no dark versions. Sizes powers of two or 3 × 2^k; M cells/s, B/cell;
captions state the A100's SM clock. Other codes only from the like-for-like benchmark or cited published numbers.

## Part 1: the library (6–7 figures)

| Fig. | file | message | data | status |
|---|---|---|---|---|
| 1 | `walkthrough_build` (panels 1–6) | A periodic point set goes through snapping, the Morton cell list, clipping one cell, all cells with their face rings, the exact start, and the first move. | the library's own 2D arrays (`scripts/overview_mesh2d.cpp`, `Mesh<2>`) | **done**, 190 × 193 mm (`paper/scripts/fig_walkthrough.py` wraps `scripts/overview_figure.py` unchanged) |
| 2 | `walkthrough_step` (panels 7–12) | A step screens the old triangles, flips the failing ones in rounds, writes the dual's faces, sums the cells in a fixed order, falls back to the host or a rebuild when stuck, and re-sorts. | same | **done**, 190 × 194 mm |
| 3 | `library_comparison` | Like for like on the same points and A100, voronoi-cpp builds full cells 4 to 55× faster than the fastest other GPU code with full cells (a); volumes-only codes (b), one CPU node (c) and the moving step (d) as panels; published numbers as open markers, cited in the caption. | `bench/libraries/results/*.jsonl`, `published.csv`, through `scripts/library_benchmark.py`'s readers | **done**, 190 × 149 mm (`fig_library.py`); regenerable when close-gaps adds rows. voronoi-cpp rows are ad62997 at 1095 MHz (ravg1002); rerun on the final commit before submission |
| 3b | `tried` (design decisions) | Every alternative tried since 2026-11-12, its measured rate change against the code before (dots: the values in the row; markers: the inputs it names), kept or dropped, and the structural reason (e.g. kNN: find-then-clip with a fixed k against the interleaved find-and-clip walk, −6 to −58 %). | `docs/tried.md` (parsed; the figure follows the list as rows are added); appendix table `figures/tried.md` from the same parse | **done**, 190 × 213 mm (`fig_tried.py`); rows without a baseline (Delaunay-first insertion: 0.38 M cells/s "when it finishes") and the precision rows are text only |
| 4 | `quality` | Speed against quality per library and input, every code measured against our oracle the same way. | the oracle measurements listed below | **proposal**: measured by another agent |
| 5 | (optional) `oracle_errors` | Relative volume error per input for voronoi-cpp (warp rebuild, exact start), and per code on halo input. | `paper/data/oracle_errors.csv` (per-cell errors from the benchmark's volumes, the same 1000-cell sample as `oracle_check`) | draft drawn (190 mm); may fold into Fig. 4 |
| 6 | (optional) `performance` | Weak scaling 32³–256³ of rebuild, exact start and step on seven inputs, and the step rate against the displacement per step. | `paper/scripts/weak_scaling_a100.sh` (55cd6b1, gpu1) | **held** for gpu1 (speed work): jobs 32362447–51 cancelled before they ran; resubmit with `submit_sweep.sh` after the speed branches merge; `extract_sweep.py`, `fig_sweep.py` ready |
| 7 | (optional) `ceiling` or `stages` | Where the step's time and bytes go per kernel, against the bandwidth budget for 5× and 10×. | `paper/data/ceiling.json` (main 69de1d3), or job 32362452 | `ceiling` draft on old data; `stages` held (job 32362452 ran, raw output kept) |

Long runs with and without re-sort (jobs 32362455, and 32362456 cancelled mid-way; `fig_resort.py`) and memory by array
(job 32362454) ran before the hold; raw output kept, held for a rerun after the speed merges; candidates for an appendix.

**Captions of Figs. 1–2, 2D stand-ins.** Every panel is computed by the library's 2D mesh (`Mesh<2>`), which shares the
3D mesh's structure; the code line under each panel names the 3D counterpart. The same in 3D: panel 1 (snapping, per
coordinate), 6 (the caller moves the points), 10 (update_cells is the same code in 2D and 3D; triangles over edges
stand for pyramids over faces). 2D stand-ins for 3D: 2 (a quadtree cell list for the octree), 3 (a polygon cut by
bisector lines for a polyhedron cut by bisector planes; the 2R circle is a sphere), 4 (one neighbour ring per cell for
one ring per face), 5 (triangles for tetrahedra), 7 and 8 (in-circle tests and edge flips for in-sphere tests and 2-3/3-2
flips), 9 (polygon edges for polygonal faces), 11 (an inverted triangle for an inverted tetrahedron). Panel 12 shows in
2D a step only the 3D mesh takes (the re-sort).

**Fig. 4: oracle measurements needed per library and input** (jittered, blue noise, uniform, clustered, halo; 128³,
and 256³ where the code runs), against the brute-force oracle of `src/test/oracle.hpp`, the same way for every code:
1. completeness: cells missing or with non-positive volume, of all n (exists for volumes, `accuracy-*.jsonl`);
2. exact neighbour sets: per sampled cell, the code's neighbour set against the oracle's (needs each code's faces or
   neighbour lists dumped; not applicable to volumes-only codes, said so in the figure);
3. volume error: per-cell relative error on the 1000-cell sample (exists, `paper/data/oracle_errors.csv`);
4. closure: per cell |Σ area vectors| / Σ |area| (codes with area vectors: voronoi-cpp, voro++, peclet-voro, our faces
   on the Delaunay codes);
5. total volume: Σ V − box volume (exists, `accuracy-*.jsonl`);
6. face agreement: each face seen from both cells, with equal and opposite area vectors and the same centroid.
Speed on the other axis from the same jobs (M cells/s, the output group stated).

## Part 2: luna-cpp on the library (4 figures, all five showcases)

Layout drafts (raster tiles of luna-cpp's own figures at 190 mm, `paper/scripts/draft_luna.py`, luna-cpp read only):
`paper/figures/draft_luna_{sedov,kh,vortex,1d}.png`. Final versions are to be redrawn from luna-cpp's stored npz with
`paperfig.py` (the tiles keep luna-cpp's design sizes, so some text prints under 7 pt).

| Fig. | content | luna-cpp source | status |
|---|---|---|---|
| 8 | **Sedov–Taylor in 3D**: (a) density slice with the moving mesh, zoom on the shock moving vs static (b, c); (d) radial n, v, P, T against Sedov–Taylor; (e) shock radius against t | `sedov3d_mesh`, `sedov3d_radial`, `sedov3d_shock` (`make_figures.py`: fig_sedov_shock, fig_sedov_radial) | draft; redraw at 190 mm, radial panels need larger text |
| 9 | **Kelvin–Helmholtz in 2D**: density at t = 0, 1, 2.5, 4, 5, moving vs LUNA MC at N = 2048; the moving mesh zoomed to single cells | `kh_2d_evolution`, `kh_2d_mesh` (fig_kh, fig_kh_density) | draft; redraw |
| 10 | **3D vortex (Yee)**: L1(ρ), L1(v) against n_x; moving mesh (two flux passes, today's scheme), static, LUNA MC; the removed one-pass run left out | `paper/data/luna/` (luna-cpp jobs 31920446, 31922276; LUNA MC from luna-reference-solutions branch yee-vortex) | **done**, `luna_vortex` 190 × 70 mm (`fig_vortex.py`). The runs predate today's library; a rerun with today's luna-cpp and voronoi-cpp is pending. Moving L1(ρ) flattens at 256³ (3.3e-5 against LUNA MC's 1.5e-5) |
| 11 | **1D**: (a) ladder summary, L1 against GPU time for Sod, Shu–Osher, Woodward–Colella; (b) the super-hard tube | `time_to_target_1d` (fig_time), `super_tube_t2_luna` | draft for (a); **(b) pending**: the existing tube figures show LUNA MC against a 2²⁰ reference only, no moving-mesh run; it needs a luna-cpp moving-mesh run of the tube |

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
