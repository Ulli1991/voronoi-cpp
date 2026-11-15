# Figures of the CPC paper (plan)

"Computer Programs in Physics": voronoi-cpp on its own terms (algorithm, exactness, robustness, performance, memory,
API), with applications. Every figure is drawn by one script in `paper/scripts/` from data in `paper/data/` (or, for
diagrams, from the library's own rules); `paper/scripts/make_all.sh` redraws them all, so they follow the speed work.

**Format.** Vector PDF plus a 600 dpi PNG, scaled to CPC's column (90 mm) or page width (190 mm). Style: LUNA's
`scripts/figstyle.py` unchanged; `paper/scripts/paperfig.py` draws at design size (print / 0.633) and scales the PDF,
so figstyle's 12 to 14 pt print at 7.6 to 8.9 pt. Curves 2.2 pt (1.4 pt printed), legends inside, no dark versions,
one colour and marker per input in every figure. Sizes only powers of two or 3 × 2^k; speed in M cells/s, memory in
B/cell; every A100 caption states the SM clock its jobs logged. Other codes only from the like-for-like library
benchmark or published numbers with citations.

Status: **done** (paper version in this folder), **redraw** (a docs/figures version exists, the paper needs a
smaller, print-sized one), **runs** (needs the listed jobs; scripts exist), **later** (needs work not yet done).

## Main text

| # | figure | message (one sentence) | section | data | status |
|---|---|---|---|---|---|
| 1 | `overview` (build and step, 2 panels or 2 figures) | A periodic point set goes through snapping, the cell list, clipping, exact tetrahedra and faces in a build, and through screening, flips and geometry in a step. | 2 Algorithm | library arrays of `Mesh<2>` (`scripts/overview_mesh2d.cpp`) | redraw: `docs/figures/overview_{build,step}.pdf` are 9.7 in wide with 9.5 to 12.5 pt text, i.e. 7.3 to 9.6 pt at 190 mm; a paper copy needs only the scaled PDF and one font step less in titles |
| 2 | `exact` (predicates and ties) | Snapping to the 2⁻³⁰ grid makes every orientation and in-sphere test exact in 128-bit integers, and symbolic perturbation breaks cospherical ties the same way on every device. | 3.1 Exact predicates | port of `exact_predicates_2d.hpp` | redraw from `construction_exact.png` (12 in, 2×2) at 190 mm |
| 3 | `cell_list_clipping` | Each cell is cut from a cube by bisectors in nearest-first order from a Morton cell list, and the walk stops once the security radius certifies it. | 3.2 Cell list and clipping | ports in `scripts/construction_figures.py` | redraw: merge `construction_cell_list` and `construction_clipping` (+ rings) into one 190 mm figure |
| 4 | `flips` (persistent tetrahedra) | Between steps the tetrahedra persist; only those whose in-sphere test fails after the move are repaired by 2-3/3-2 flips on the GPU, with a host fallback and a rebuild as last resort. | 3.3 Moving mesh | `construction_flip` + `flow_step` | redraw: one 190 mm figure from `construction_flip` and a slim `flow_step` |
| 5 | `memory_layout` | Cells, tetrahedra and faces live in flat pools ordered along a space-filling curve, sized by measured bounds and grown on overflow. | 4 Data layout | `data_arrays`, `memory_pools`, `gpu_access_*` | redraw: one 190 mm figure (arrays and pools), access patterns to the appendix |
| 6 | `oracle_errors` | Every sampled cell volume is within 2·10⁻¹³ of a brute-force oracle with exact predicates on all inputs, no cell is missing, while float and inexact codes miss cells or err by up to 10⁻⁴. | 5 Correctness | `paper/data/oracle_errors.csv` | **done** (190 mm) |
| 7 | `fuzz` | Fuzzed inputs (lattices, ties, clusters, 3·10⁴ to 1.6·10⁷ points) end exact or are refused with a reason; none is wrong. | 5 Robustness | `build-fuzz/` campaign results, A100 campaign `mesh_fuzz_a100.sh` | runs: the A100 campaign of 2026-11-10 ran on branch fuzz-45 (604 cases, 3 WRONG, 85 HANG); rerun on current main before drawing |
| 8 | `weak_scaling` | Rebuild, exact start and steps keep their rate from 32³ to 256³ on every input; nonuniform input costs 2-3× in the rebuild and start, not in the step. | 6 Performance | `paper/data/weak_scaling.csv` from `weak_scaling_a100.sh` | runs (submitted); script ready |
| 9 | `step_vs_displacement` | The step's rate falls with the displacement per step (flips per cell), and stays above the rebuild up to about 0.3 to 1 mean spacings. | 6 Performance | same sweep, + 0.01 and 1 spacings at 128³ | runs (submitted) |
| 10 | `stages` (time per stage and per kernel) | Most of a step is the repair and the face geometry; most of a rebuild is clipping. | 6 Performance | `stages_a100.sh kernels` | runs (script ready) |
| 11 | `ceiling` | Kernel by kernel the step moves N bytes per cell, so at 80 % of the A100's bandwidth it could run at X M cells/s, the room the speed work aims for. | 6 Performance | `scripts/ceiling_tables.py` output (Nsight Compute, docs/measure_2026-11-12.md) | redraw from `measure_ceiling.png`: data are in job logs on Raven; extract to paper/data/ceiling.json, then redraw at 190 mm; refresh after the speed merges |
| 12 | `memory` (B/cell by array) | A moving mesh needs about N B/cell at the peak of a step, the arrays split as shown; 512³ fits on one 40 GB A100. | 4/6 | `voronoi_bench step --memory` (docs/measure_2026-11-10.md section C) | runs: one `--memory` job on 55cd6b1 (the numbers in `data_figure.py` predate the step-bytes work) |
| 13 | `library_comparison` | Like for like on the same points and A100, voronoi-cpp builds full cells 4 to 55× faster than the fastest other GPU code with full cells, with exact topology. | 7 Comparison | `bench/libraries/results/*.jsonl`, `published.csv` | **done** (190 mm, from the benchmark's raw records) |
| 14 | `applications` (4 panels) | MD (fcc melt vs voro++), cosmological density (DTFE-like), Lloyd relaxation, moving-mesh hydro (luna-cpp). | 8 Applications | application codes | later: placeholders until the applications run |

## Appendix

| # | figure | message | data | status |
|---|---|---|---|---|
| A1 | `resort` (long runs) | Without re-sort the step's rate decays by a third over 500 steps as the faces scatter in memory; a Hilbert re-sort every 50 updates keeps it. | `stages_a100.sh long128`, `long256` | runs (script ready; the 2026-11-10 data are on an older library) |
| A2 | `clip_tails` | Clipping work per cell has short tails on all inputs (cuts per cell, p99). | `measure_a100.sh clip` (VORONOI_MEASURE build) | redraw from `measure_clip_tails.png` (data in build-measure/jobs); refresh on 55cd6b1 |
| A3 | `faces_per_cell` | Every rebuild passes the cell checks (closure, both-sided faces), and faces per cell match Poisson-Voronoi theory (15.535, Meijering 1953) on uniform input at every N. | `check-*.txt` of the sweep | runs (in the sweep) |
| A4 | `gpu_access` | Warps read neighbours along the Morton order; flips and faces touch tetrahedra in pool order. | rules in `scripts/access_figures.py` | redraw at 190 mm |
| A5 | `flow` (control flow of update and rebuild) | Which kernel runs where, and where the host waits. | `flow_mesh`, `flow_rebuild` | redraw at 190 mm; times from the stages run |
| A6 | `oracle_by_size` | Errors do not grow from 128³ to 256³ (median ≈ 2× with twice the faces per volume ulp). | `oracle_errors.csv` | can be drawn now; optional |

## Data

`paper/data/` holds the extracted data (CSV/JSON) with their sources; raw job output in `paper/data/raw/`.
`paper/data/SOURCES.md` lists every file, the job ids, the commit and the clock.
