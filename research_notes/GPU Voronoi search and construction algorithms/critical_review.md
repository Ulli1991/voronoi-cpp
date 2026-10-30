# Critical review of the GPU Voronoi research notes (2026-10-29)

## Scope and method

I read all 13 files: the 8 notes in `GPU Voronoi search and construction algorithms/`, the 5 notes in `Local Voronoi construction literature/`, and `reports/Local Voronoi construction literature.md`.

Claims were checked against these primary sources:
- **Taveira et al. 2026.** Full text of arXiv 2605.06408, Tables 1–12 and Appendices A–D. Extracted locally at `scratchpad/pdf/2605.06408.txt`, source https://arxiv.org/pdf/2605.06408.
- **Xiao, Cao & Chen.** Full text of arXiv 2509.07175v2, https://arxiv.org/pdf/2509.07175.
- **paragram source code.** https://github.com/zenseact/paragram (local clone).
- **peclet-voro.** `docs/performance_report.md` (local clone).
- **votess.** `joss/loglog.png`.
- **Nanjappa thesis and gDel3D paper.** Text extracts.
- **Basselin D1-Days slides.** https://members.loria.fr/Olivier.Devillers/seminaires/Slides/2021-D1-Days/Basselin.pdf
- **Radiant Foam.** https://arxiv.org/html/2502.01157
- **ArepoCUDA page.** https://nelson.tng-project.org/ArepoCUDA

Ray et al. 2018 and Basselin et al. 2021 are still unreadable: HAL returns an Anubis bot block, and EG diglib refuses the connection. Their numbers keep the [unverified] flag.

Abbreviations:
- WN = white noise (uniform Poisson)
- DG = density gradient
- T*n* = Taveira Table *n*

---

## (a) Checked claims

### A. Taveira et al. 2026 ("paragram"): throughput numbers

Key facts from the paper:
- All runs are fp32 ("We run our method using single precision").
- Timings are end-to-end, with 3 warm-up runs and 10 timed runs.
- Tables 4–7 are **Voronoi**; Tables 8–11 are **power diagrams**.
- Table 7 is WN Voronoi; Table 11 is WN power diagram.

| # | Claim | File(s) | Verdict | Correct value and source |
|---|---|---|---|---|
| A1 | "White noise, H200: 0.132 s for 1M, 1.208 s for 10M, 1.811 s for 15M (≈8.3 M cells/s). RTX 5090: 1.216 s for 10M" | construction_algorithms.md Q4 | **WRONG** (these are T11 power-diagram numbers) | Voronoi WN (T7): H200 0.123 / 1.064 / 1.582 s; RTX 5090 10M 1.078 s. That is **9.5 M cells/s at 15M on H200** and 9.3 M/s at 10M on the 5090. |
| A2 | "runs at about 8 M cells/s on uniform points"; "Taveira about 8 M/s fp32 on H200" | construction_algorithms.md Q4 Takeaway and Inferences | **CORRECTED** | 9.3–9.5 M/s for Voronoi WN at 5–15M (T7). 8.1 M/s only at 1M on H200 (0.123 s). |
| A3 | "white noise at 10M took 1.216 s (5090) / 1.208 s (H200)" (labelled unweighted Voronoi) | gpu_techniques.md §2 | **WRONG** (T11 power diagram) | T7 Voronoi: 1.078 s (5090) / 1.064 s (H200). |
| A4 | "Density-gradient data, H200: 1.588 s for 15M (≈9.4 M/s). Basselin* reimplementation: 5.574 s" | construction_algorithms.md Q4 | **CORRECTED** (GPUs mixed) | 5.574 s is the RTX 5090 Basselin* time. On H200 at 15M DG, Basselin* takes **4.829 s** (T6). "Ours" has no 15M result on the 5090. |
| A5 | Table of WN timings: 5090 0.020…1.078; H200 0.017…1.582 | neighbour_search.md §2; theses_and_code.md; fresh_sweep_gpu_construction.md; deep_dive_gpu_kernels.md | **CONFIRMED** | T7 |
| A6 | "about 9.3M cells/s at 10M on H200" | neighbour_search.md §2 | **CORRECTED** (minor) | 10M / 1.064 s = **9.4 M/s**. The 9.3 M/s figure is the 5090 at 10M (1.078 s). |
| A7 | Weighted (T11) values "5090 1M 0.123, 10M 1.216; H200 15M 1.811", explicitly labelled as power diagram | fresh_sweep_neighbour_search.md Q1; fresh_sweep_gpu_construction.md | **CONFIRMED** (correctly separated) | T11 |
| A8 | "Power diagrams: Table 9, clustered: 0.180 s at 0.1M and 1.913 s at 10M; 3.210 s at 15M H200 vs Geogram 10.965" | geometry_gpu_voronoi.md Q1 | **CORRECTED** (table label) | Those values are **Table 8** (10 clusters, weighted). Table 9 is 5 clusters: 0.153 s at 0.1M, 1.718 s at 10M, 2.879 s at 15M on H200. |
| A9 | "Table 12, white noise: 1.216 s at 10M" (power) | geometry_gpu_voronoi.md Q1 | **CORRECTED** (table label) | That value is **Table 11**. Table 12 is the ablation study. |
| A10 | DG (T6): 5090 0.118 / 1.102; H200 0.132 / 1.084 / 1.588; gDel3D 0.283 / 0.226; Basselin* 0.387 / 0.561, 3.602 / 3.311; HXT 0.307 / 1.670 | gpu_techniques.md, geometry_gpu_voronoi.md, hidden_gems.md, fresh_sweep_* | **CONFIRMED** | T6 |
| A11 | 10 clusters (T4): Ours 5090 0.175…1.423; H200 0.456…2.530; gDel3D 5090 0.056…3.120; gDel3D H200 1M 0.229, 2M 1.196, 5M 45.423 | several | **CONFIRMED** | T4 |
| A12 | "gDel3D faster at small clustered sizes (H200, 2M points: 0.576 s vs 1.104 s)" | fresh_sweep_kernel_design.md Q1 | **CORRECTED** (dataset label) | These are **5-cluster** values (Table 5). For 10 clusters at 2M on H200, Ours (1.027 s) beats gDel3D (1.196 s) (T4). |
| A13 | Real scenes (T1): bicycle 0.570 / 0.684; garden 0.496 / 0.574; room 0.240 / 0.293; gDel3D 1.204 / 0.893 etc.; RF Del 16.9–250.7 s; CGAL 20.445 s | several | **CONFIRMED** | T1 |
| A14 | "1.5–2.4x faster than gDel3D on real scenes" | construction_algorithms.md Q4 | **CORRECTED** | Per-scene ratios from T1 are **1.31–1.83× on H200** and **2.11–2.59× on the 5090**. The paper's averages are "∼55% slower (H200), ∼137% slower (5090)" (Sec. 5.2). |
| A15 | "On real-world scenes the new method is on average about 55% faster on the H200 and about 137% faster on the 5090" | fresh_sweep_kernel_design.md Q1 | **CORRECTED** (wording) | The paper says gDel3D is "∼55% slower … and ∼137% slower". That means gDel3D takes 1.55× / 2.37× the time. It is not equivalent to "Ours is 55% faster". |
| A16 | "HXT about 55% slower on H200 for Poisson data" | neighbour_search.md §2 | **CONFIRMED** | Sec. 5.2. The same passage gives HXT as "∼129% slower on real-world scenes". |
| A17 | "On real-world scenes the 5090 was typically 5–20% faster" than the H200 | gpu_techniques.md §5 | **CORRECTED** (minor) | 3–26% faster per scene (T1). |
| A18 | "roughly the same throughput as the best 16-core CPU code (HXT) … and 2–3× it on Poisson data" | fresh_sweep_gpu_construction.md Inferences | **CORRECTED** (matters for decision 2) | Ours/HXT on T7 is 2.7× at 1M (0.112 vs 0.303 s) but only **1.45× at 10M** (1.064 vs 1.545 s) and **1.43× at 15M** (1.582 vs 2.259 s). The paper's own average is ~55%. At scale the GPU lead over a 16-core CPU is about 1.4–1.5×, not 2–3×. |
| A19 | CPU 15M WN: Geogram 6.138, HXT 2.259, voro++ 41.136, CGAL 71.999 (serial), CGAL Parallel 6.903 | theses_and_code.md, deep_dive_gpu_kernels.md | **CONFIRMED** | T7. Do not mix the serial CGAL figure (72.0 s) with CGAL Parallel (6.9 s). |
| A20 | Basselin* H200 WN 0.386 / 1.624 / 3.186 / 4.619 s (≈2.6–3.2 M/s); 5090 3.608 s at 10M | deep_dive_gpu_kernels.md, theses_and_code.md | **CONFIRMED** | T7 |
| A21 | Clustered 15M H200: 5.9 M/s (10 clusters, 2.530 s); 6.8 M/s (5 clusters, 2.194 s) | deep_dive_gpu_kernels.md, fresh_sweep_gpu_construction.md | **CONFIRMED** | T4, T5 |
| A22 | "On '10 clusters' inputs Taveira's method is 2–3x slower than on white noise at the same N" | fresh_sweep_neighbour_search.md Q4 | **CORRECTED** | Clustered/WN ratio varies strongly with N: 8.8× at 0.1M (0.175/0.020), 3.4× at 1M, 1.3× at 10M on the 5090; 1.6× at 15M on H200. |
| A23 | BVH build 4.8% of time; fused kernel 95.2%, of which 96.2% is clipping | all | **CONFIRMED** | App. D "Timing breakdown". "Clipping" here is the authors' undefined attribution; no profiler method is given. |
| A24 | Ablation: isotropic culling 1.26 / 1.33 / 1.40×; DFS 27.4 / 1.30 / 37.2× (5090) | all | **CONFIRMED** | T12. Text: isotropic culling costs "1.03–1.61× across all datasets". |
| A25 | Taveira precision "not stated" (fresh_sweep_moving_mesh.md) or "unclear, likely float" (hidden_gems.md) | fresh_sweep_moving_mesh.md, hidden_gems.md Q2 | **WRONG** | App. D: "We run our method using single precision". The code builds with `-use_fast_math` by default (`paragram/_backend.py` l.159). |
| A26 | 0.1–0.2% adjacency mismatch vs CGAL on real data; density difference mean <1.7e-5 | all | **CONFIRMED** | App. D "Precision" |
| A27 | gDel3D fp32 vs fp64 difference "median <0.5% (H200/RTX 5090)" | gpu_techniques.md §4 | **CORRECTED** (minor) | App. C gives "median < 0.5%" but names no GPU. |
| A28 | "Replacing Radiant Foam's step … scaled it to more than 20 M cells (5×)" | hidden_gems.md | **CONFIRMED with caveat** | The abstract says "5x, to over 20 million points". The experiment in Sec. 5.3 varies the final site count from 3M to 17M. |
| A29 | Power diagram bicycle (T3): 0.595 → 2.243 s for weight ratio 0 → 1e-1 | fresh_sweep_gpu_construction.md, deep_dive | **CONFIRMED** | T3. The GPU is not stated in T3. |
| A30 | "Power diagram for bicycle: 1.134 s on 5090" | fresh_sweep_moving_mesh.md | **CONFIRMED** | T2 |

### B. Taveira / paragram: algorithm and code claims

| # | Claim | File(s) | Verdict | Correct value and source |
|---|---|---|---|---|
| B1 | "The cell is final when the stack is empty. This is an exact termination certificate under the culling bound" | deep_dive_gpu_kernels.md Q3; report ("exact termination certificate") | **CORRECTED** (important for decision 1) | In code the traversal stack is capped at 64 entries for Voronoi (`voronoi_bvh.cuh`, `const int stackSize = 64`) and 128 for power diagrams (`pwr_bvh.cu`, `MAX_STACK_K 128`). On overflow, `voronoi_stack_push` **drops the farthest entry** (or the new node, if it is farther). Pruned nodes are never revisited. The certificate is therefore conditional on no stack overflow, which the code does not report. Combined with fp32 arithmetic, the paper's term "robust" means robust to distribution in practice, not certified. |
| B2 | Scratch memory: 2,624 B/cell; `MAX_PLANES 64`, `MAX_VERTS 64`, `CELL_BLOCK_STRIDE 32`; SoA layout `slot*32+lane` | theses_and_code.md (`faster_convex_cell.cuh`), deep_dive (`voronoi_convex_cell.cuh`) | **CONFIRMED**, and the two files are not in conflict | Both headers exist: `faster_convex_cell.cuh` is used by the Laguerre/power path, `voronoi_convex_cell.cuh` by the Voronoi path. |
| B3 | Stack `MAX_STACK_K 128` (theses_and_code) vs a 64-entry stack (deep_dive) | theses_and_code.md vs deep_dive | **Both correct** | 128 is the power-diagram path (`pwr_bvh.cu`); 64 is the Voronoi path (`voronoi_bvh.cuh`). |
| B4 | 64-bit Bloom-filter plane mask (hash `vid*2654435761u>>26`) | deep_dive, report | **CONFIRMED** | `voronoi_convex_cell.cuh` l.121, l.162 |
| B5 | Output is symmetrised adjacency CSR | report | **CONFIRMED** | `voronoi_ultra.cu` l.318ff (`populate_symmetric_adjacency`). Symmetrisation hides one-sided mismatches; it does not resolve them. |
| B6 | Leaf size 17 (H200) / 10 (5090); K=8 warm start off on H200 | all | **CONFIRMED** | App. A.2, A.6; `_backend.py` |
| B7 | "Taveira: one thread per cell … decomposes into N independent tasks" | all | **CONFIRMED** | App. A.1 |
| B8 | "paragram apparently does not reorder sites (no Morton sort)" | deep_dive, report | **Plausible, unverified** | A grep of `voronoi_ultra.cu` finds no sort or Morton/Hilbert call. The cuBQL BVH build may permute internally, but thread order follows input order. Keep the hedge. |

### C. Xiao, Cao & Chen (arXiv 2509.07175v2)

| # | Claim | File(s) | Verdict | Correct value and source |
|---|---|---|---|---|
| C1 | Fig. 1, 5M cells in a cube, RTX 3080: WN 3.12 / 2.81 / 0.49 s; blue noise 0.34 / 0.42 / 0.26 s, attributed to Ray [24], Basselin [27], Ours | deep_dive, fresh_sweep_gpu_construction, report | **CONFIRMED** | Fig. 1 caption: "two parallel implementations [24], [27]". [24] = Ray, [27] = Basselin. |
| C2 | Derived 10.2 M/s (WN) and 19.2 M/s (blue noise), fp32, including centroids, excluding allocation and transfer | several | **CONFIRMED** (arithmetic) | Sec. VII-B: "32-bit floating-point numbers"; "covers cells construction and centroids computation, excluding GPU memory allocation and data transfer". This is **not end-to-end** in Taveira's sense. |
| C3 | "Claimed exact" | report ranking table; deep_dive Q5 | **CORRECTED** (over-claim if read as arithmetic exactness) | The paper's "exact results" means **combinatorial completeness**: the iteration "continues until all corner points are in the target Voronoi cell" (Sec. V-B). All GPU runs are fp32. Integer flags settle ties only at corners. No exact predicates are used. |
| C4 | Ray code OOM above 7M WN sites on 10 GB; Liu code tested ≤100K | several | **CONFIRMED** | Sec. VII-B |
| C5 | ~20 NN queries per cell ("over 20.54 millions searches" for 1M cells) | construction_algorithms, deep_dive | **CONFIRMED** | Text l.757. This count is for the **serial** method. The parallel Alg. 3 "needs more point-in-cell tests" (authors), so the GPU per-cell query count is higher and unreported. |
| C6 | No code link | several | **CONFIRMED** | Footnotes link only the baselines' code. |

### D. Ray 2018, Basselin 2021, votess, peclet-voro, gDel3D/gStar4D, Radiant Foam

| # | Claim | File(s) | Verdict | Correct value and source |
|---|---|---|---|---|
| D1 | Ray 2018 teaser: "10 million points … in 800 ms (NVidia V100) … 84 million Delaunay tetrahedra per second" → 12.5 M cells/s | geometry_gpu_voronoi.md; report ("~12.5 M on V100"; "The 2018 V100 headline has not clearly been beaten on uniform data in eight years") | **UNVERIFIABLE** (keep the flag; downgrade the report's conclusion) | HAL is bot-blocked and two searches found no copy of the caption. 84M / 12.5M = 6.7 tets/point is internally consistent. Point distribution and precision are unknown. Basselin's slides list a Ray-2018 limitation "GTX1080: can't reach 10M points for white noise", so a 10M teaser was probably blue noise or near-uniform. The report's "not beaten in eight years" rests entirely on this unverified snippet and should be dropped. |
| D2 | Basselin 2021 Fig. 1: 10M points restricted to a mesh in 300 ms on a V100; hidden_gems derives "≈33 M pts/s, the highest reported per-cell GPU rate" | hidden_gems.md Q2; geometry_gpu_voronoi.md; deep_dive | **UNVERIFIABLE; the 33 M/s derivation is an over-claim** | The PDF is still unreachable. Basselin's own slides (local copy, URL above) give "Process 4300K seeds per second / 60K seeds per second" for the restricted case vs Liu et al.: **4.3 M seeds/s**, an order of magnitude below 33 M/s. The 300 ms caption may cover a different setting (e.g. Voronoi only, no integration, or pre-sorted input). Do not use 33 M/s as an anchor. |
| D3 | votess on A100: ~1.8 s at 1e6, ~13 s at ~8e6 → ~0.6 M/s (deep_dive) vs "≈1.2e-6 s/point at 1e6 (≈0.8 M/s)" (geometry_gpu_voronoi) | deep_dive vs geometry_gpu_voronoi | **Both are plot read-offs; resolved below** | `joss/loglog.png` (2× Xeon 8360Y, A100-40GB) shows the GPU curve at ≈1.8–1.9 s at 1e6 and ≈13 s at ≈8e6, i.e. **0.55–0.6 M/s**. CGAL on 72 cores (≈2.5 s at 8e6, about 3 M/s) is **faster** than votess-GPU. The 0.8 M/s value comes from a different arXiv figure and is a coarser reading. |
| D4 | peclet-voro (RTX 5080, FP64): cold build 2.99 / 3.25 / 3.54 / 3.61 Mcells/s; construct-only 8.5; FP32 construct 14–17 (self-reported); repair 21.8 Mcells/s at δ/h=1e-4; repair/rebuild speedups 6.5 / 2.9 / 1.1 / 0.86 / 0.84 | theses_and_code, deep_dive, hidden_gems, report | **CONFIRMED** against the repository report (not peer-reviewed) | `docs/performance_report.md` l.95, 105, 132–134, 214–226 |
| D5 | "definition of δ/h not verified; we read it as per-step displacement over mean spacing" | hidden_gems.md, report | **RESOLVED (confirmed)** | performance_report.md l.10: "per-step displacement δ/h (fraction of the mean inter-particle spacing)". |
| D6 | "The repair became exact only after it started storing near-miss candidates" / "exact to 1e-11 / 1e-15" | report, hidden_gems, fresh_sweep_moving_mesh | **CORRECTED** (over-claim) | The same report (l.271–281) says total volume is conserved to ~1e-15. In the local-repair band δ/h ≈ 5e-4…5e-3, however, "a handful of cells" (up to ~23 on the GPU) carry worst-cell volume errors up to **~8e-4 (GPU)**. "Exact" holds only for total volume, or at very small or very large δ/h. |
| D7 | gDel3D Phase 2: "only about 500 stars … <200 more … <1.5%" stated for "about 1e6 points" | construction_algorithms.md Q1 | **CORRECTED** (minor) | The experiment used **5×10⁵** synthetic points (gDel3D paper Sec. 5.2, Fig. 8). |
| D8 | gDel3D on a GTX Titan is memory-bound, with ~30% speedup over the GTX 580; speedup over CGAL 4–6× rising to 8–10× | construction_algorithms.md | **CONFIRMED** | gDel3D paper Sec. 5.1 and 5.5 |
| D9 | gFlip3D exact insphere "hundreds of registers", "local storage of up to 14880 floating point numbers"; fast and exact kernels split | theses_and_code, fresh_sweep_gpu_construction | **CONFIRMED** | Nanjappa thesis Sec. 4 (text l.2688–2690) |
| D10 | gStar4D on real inputs 0.73–1.43× CGAL; 2–4× slower than gFlip3D; Blade could not run | theses_and_code.md | **CONFIRMED** | Thesis Table 6.2 and Sec. 6.4. The per-distribution insertion counts (Table 6.3) were not re-extracted, since the tables were lost in text extraction; they remain unverified. |
| D11 | Radiant Foam is "full rebuild every step or iteration" (strategy 1); Taveira "benchmark full reconstruction" | fresh_sweep_moving_mesh.md Q2 | **WRONG** | Radiant Foam: "we must update the adjacency by performing an incremental Delaunay triangulation" in "custom CUDA kernels", with the rebuild ratio going from 1:1 to 1:100 optimiser steps (https://arxiv.org/html/2502.01157). Taveira: "Radiant Foam uses incremental updates when possible, we benchmark full reconstruction for a controlled comparison" (Sec. 5.3). theses_and_code.md states this correctly. |
| D12 | "Ray et al. could not reach 10M on a GTX 1080, per the Taveira slides" | fresh_sweep_neighbour_search.md Q4 | **CORRECTED** (source) | The statement is in **Basselin's** D1-Days 2021 slides, not Taveira's. |
| D13 | Local DeWall: 2× speedup, 1.2 B points on one consumer GPU | theses_and_code, fresh_sweep_gpu_construction | **UNVERIFIABLE** (ScienceDirect 403) | Keep the flag. The code-level description (2 lanes per vertex, block-per-vertex post phase) comes from a local repo read and is plausible. |
| D14 | Menezes et al. GPU exact predicates: "up to 289×" | construction_algorithms.md Q5 | **CONFIRMED, but incomplete** | 289× is the IMR 2019 version (2D segment intersection). The CAD 2022 journal version ("Fast Parallel Evaluation of Exact Geometric Predicates on GPUs", 3D triangle intersection) reports up to **1936×**. Both use an interval filter on the GPU with **CPU** rational re-evaluation. Sources: https://internationalmeshingroundtable.com/assets/papers/2019/14-Menezes.pdf; https://www.academia.edu/82417124/Fast_Parallel_Evaluation_of_Exact_Geometric_Predicates_on_GPUs |
| D15 | User's README: 7.4M particles, 555.6 + 1197.0 = 1752.0 ms → 4.2 M cells/s | hidden_gems.md | **CONFIRMED** (arithmetic) | This **conflicts** with the report's "Ours: 5–7 M/s at 7.4M points". The report should state which measurement it uses. |

### E. Own-derivation and inference checks

| # | Claim | File | Verdict |
|---|---|---|---|
| E1 | A nearest-first walk with the "next d ≥ 2R_current" stop tests exactly N(B(g, 2R_max,final)) points | neighbour_search.md §1 | **CONFIRMED (logic)**. Every natural neighbour has d ≤ 2R_final, so all are clipped before the walk passes 2R_final. After that R_current = R_final, and the stop fires at the first point beyond 2R_final. The count is a property of the point set (plus one stop-trigger point). |
| E2 | Mean faces 15.535 = 2 + 48π²/35 | several | **CONFIRMED** (Meijering; cited via Lazar et al. arXiv 1401.1736 in cfd_math note). |
| E3 | A100 issue budget: 108·4·1.41 GHz ≈ 6.1e11 warp-instr/s → ≈53 M cells/s cap at 100k thread-instr/cell and 8.7/32 lanes active | report | **CONFIRMED** (arithmetic; it is an upper bound that ignores dual-issue limits and stalls). |
| E4 | 1.2 KB/thread shared → ≈136 threads/SM → ≈6% occupancy on A100 (164 KB shared) | gpu_techniques.md | **CONFIRMED** (arithmetic: 136/2048 = 6.6%). |
| E5 | "Per-star work comparable to clipping" (lifted star vs clip) | construction_algorithms, theses_and_code | **Inference only.** It is supported qualitatively by Nanjappa's verdict ("enormously wasteful … every tetrahedron … constructed in its four stars") and gStar4D's real-input speedups below gFlip3D. |

---

## (b) Contradictions between files, resolved

1. **Taveira white-noise numbers.** construction_algorithms.md and gpu_techniques.md use T11 (power diagram) values. neighbour_search.md, theses_and_code.md, fresh_sweep_* and deep_dive use T7 (Voronoi) values. **T7 is correct for Voronoi**: H200 0.123 / 1.064 / 1.582 s at 1 / 10 / 15M and 5090 1.078 s at 10M, i.e. 9.3–9.5 M/s. The T11 power-diagram values are 0.132 / 1.208 / 1.811 s (H200) and 1.216 s (5090 10M), i.e. 8.3 M/s.
2. **Taveira precision.** Most notes say fp32. fresh_sweep_moving_mesh says "not stated" and hidden_gems says "unclear". **It is fp32** (App. D), with `-use_fast_math` in the code.
3. **"Best published GPU rate".** The candidates were Ray's 12.5 M/s on a V100 (geometry note, report), Basselin's 33 M/s (hidden_gems), Taveira's 8 M/s (construction note) and Xiao's 19 M/s blue noise. Resolution:
   - Verified end-to-end plain Voronoi: Taveira, **9.5 M/s** (fp32, H200, WN 15M).
   - Verified kernel-only (construction plus centroids, no allocation): Xiao, **10.2 M/s WN / 19.2 M/s blue noise** (fp32, RTX 3080, 5M).
   - Ray 12.5 M/s and Basselin 33 M/s are unverified snippets. Basselin's own slides imply ~4.3 M seeds/s for the restricted case.
4. **Radiant Foam update strategy.** fresh_sweep_moving_mesh says full rebuild; theses_and_code says incremental. **It is incremental GPU Delaunay** (primary source above).
5. **Stack size (128 vs 64) and convex-cell header name.** Not a contradiction: the power path and the Voronoi path use different files (B2, B3).
6. **votess rate (0.6 vs 0.8 M/s).** Both are plot read-offs. The JOSS log-log plot gives about 0.55–0.6 M/s on an A100 at 1–8M points.
7. **"Ray could not reach 10M" source.** It comes from the Basselin slides, not the Taveira slides.
8. **Our own throughput.** The report says 5–7 M/s uniform at 7.4M. hidden_gems quotes the repo README: 4.2 M/s total (search plus reconstruction). construction_algorithms says "peaks at ~16 M cells/s". These are different measurements and configurations; the report should name them explicitly.
9. **Is a GPU moving-mesh hydro code available?** The report says "No GPU Voronoi moving-mesh hydro code exists". fresh_sweep_moving_mesh lists ProteusGPU (README: 2D/3D moving-mesh generation on GPU, AREPO-like hydro, weak scaling on 1000 GH200) and peclet-voro (GPU moving Voronoi for CFD). **The report's statement is wrong as written.** Correct wording: "no peer-reviewed paper with published per-step GPU Voronoi mesh-construction rates was found".

---

## (c) Most important corrections that change conclusions

1. **Published GPU throughput (decision 2).** Use these anchors, and stop quoting the following: 8 M/s (it is the power-diagram table), 12.5 M/s for Ray (snippet), and 33 M/s for Basselin (snippet, contradicted by the authors' slides).

   | Method | Rate | Conditions |
   |---|---|---|
   | Taveira | **9.3–9.5 M cells/s** | Plain Voronoi, fp32 with fast-math, end-to-end, H200 / RTX 5090, uniform 5–15M (T7). Output is adjacency only. |
   | Taveira, power diagram | 8.3 M/s | T11 |
   | Xiao | **10.2 M/s WN, 19.2 M/s blue noise** | Plain Voronoi in a cube, fp32, RTX 3080, 5M. Construction plus centroids; excludes allocation and transfer. |
   | gDel3D | 3.4–4.7 M points/s | Double precision, topology only |

   At 10–15M points, Taveira's GPU method is only **~1.4–1.5× faster than 16-core HXT on the CPU** (T7), not 2–3×.

2. **The paragram termination is not a certificate in code (decisions 1 and 4).** The BVH stack (64 entries) silently drops the farthest nodes on overflow. The arithmetic is fp32 with fast-math, and 0.1–0.2% of adjacencies mismatch. It is then post-hoc symmetrised.
   - No published GPU per-cell method gives an exact guarantee.
   - For the user's design, a nearest-first grid walk with a 2R security stop and exact double/integer predicates is a stricter certificate than any published GPU method. E1 shows the 2R walk's candidate count is at the optimum for that criterion.
   - The literature supports per-cell clipping with ordered, culled candidates as the dominant GPU design. "Exact termination certificate" should be removed from the report's description of paragram.

3. **Topology reuse on GPUs exists (decision 3).** The report's list of "five results new to the literature" overstates novelty.
   - Radiant Foam does **incremental GPU (CUDA) Delaunay updates across optimiser steps** (arXiv 2502.01157).
   - peclet-voro does **GPU (Kokkos) Verlet-skin and near-miss certificate repair** of 3D Voronoi cells, reusing topology across timesteps. It is not peer-reviewed, and its numbers are self-reported.
   - cellGPU does GPU certificate testing with CPU repair (2D).
   - ProteusGPU (README only) is a GPU moving-mesh code.
   - What remains unpublished, as far as found:
     - peer-reviewed GPU **3D Voronoi** topology reuse with timings;
     - **sharing candidate data across neighbouring cells** in a GPU per-cell clipper. The closest analogue is gStar4D's reciprocated working sets, shared via a digital Voronoi diagram (Nanjappa thesis Ch. 6, GPU, 2012). Taveira, Ray, Basselin and votess are all strictly independent per cell.
   - **Splitting topology from geometry** is common, not novel:
     - Taveira outputs topology only.
     - peclet-voro computes geometry in a separate per-vertex scatter.
     - gDel3D and Radiant Foam produce topology first.
   - "Warp-cooperative per-cell construction" is not entirely new:
     - Local DeWall pairs 2 lanes per vertex and runs a block-per-vertex post phase (repo code, unverified paper).
     - A 2018 Brno student paper on brittle-body fracture uses workgroup-per-cell Voronoi, at ≤100 cells: https://excel.fit.vutbr.cz/submissions/2018/005/5.pdf
   - No **large-scale, measured** warp-per-cell 3D Voronoi result was found.

4. **Exact predicates on GPUs (decision 4).** The verified evidence is:
   - gDel3D/gFlip3D split fast and exact kernels. The exact insphere needs "hundreds of registers" and up to 14,880 floats of local storage (Nanjappa thesis). Exact work is confined to a compacted second launch.
   - GPredicates is a CUDA port of Shewchuk's predicates (no numbers extracted).
   - Menezes et al. use a GPU interval filter with CPU rational fallback: 289× (IMR 2019) and 1936× (CAD 2022) on predicate evaluation.
   - Ray/Liu have an optional arithmetic filter that flags `needs_exact_predicates`, off by default. Basselin relaunches flagged cells in **double**, which is not exact.
   - Local DeWall uses filtered predicates plus SoS with a double post phase.
   - **No GPU cost measurement for exact orient/insphere inside a per-cell clipper, or for 128-bit integer predicates, was found.**
   - Xiao's "exact" is combinatorial completeness in fp32, not exact arithmetic (C3).

5. **Moving-mesh repair economics.** Use peclet-voro's numbers as stated:
   - δ/h is per-step displacement over the mean spacing (verified).
   - The break-even point is around δ/h ≈ 5e-3–1e-2.
   - At δ/h ≥ 1e-2 the repair is slower than a rebuild (0.84–0.86×).
   - "Exact" applies to total volume only. Worst cells reach about 8e-4 volume error on the GPU in the local-repair band (D6).

## Not resolved (flags kept)

- Ray 2018 and Basselin 2021 own timings, GPU, precision and k: HAL and EG are bot-blocked.
- Local DeWall and Gao & Chen 2025 numbers: ScienceDirect returns 403.
- gStar4D Table 6.3 insertion counts: the table was lost in text extraction.
- Galgoczi 2026 SER numbers (abstract only).
- Wald 2011 and Osama 2022 full numbers.
- Lévy 2022 step-fraction percentages: garbled extraction, flagged in the note.
- Observation, not a note error: the session scratchpad contains a **ProteusGPU clone, build and run directory** (`scratchpad/ext/ProteusGPU`, `proteus-build`, timestamps 12:23–12:30). No note analyses ProteusGPU's algorithms beyond its README; all six mentions are disclaimers or README facts. Confirm that this build was not used as a source of ideas.
