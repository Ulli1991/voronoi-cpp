# Neighbour search for per-generator 3D Voronoi cell construction on GPUs

Scope: how a thread or warp finds the particles that can cut its cell, in what order, and how that choice drives total cost. Context: A100, FP64, uniform grid with about 1 particle per cell, distance-sorted stencil, 2R security-radius stop, nearest-first clipping through a 16-entry sorted buffer. Measured per cell: about 75-115 grid cells, 24-44 candidates and 16-17 cuts.

Method note: some numbers below come from my own Monte Carlo, run with scipy Qhull Delaunay at density λ=1 (mean spacing 1). The script is `/tmp/claude-28632/-raven-u-uli-voronoi-cpp/2fada6d8-a9b5-4727-917f-6d817979c190/scratchpad/mc/pv.py`. There were about 13.9k interior cells per point set; boundary cells were excluded with a margin of 6. These numbers are marked **[own MC]**. They are derived results, not published ones. The PROTEUS/ProteusGPU code was not consulted.

## 1. Candidate counts: what published methods need vs the theoretical minimum

### Takeaway
For Poisson points, a nearest-first walk with the 2R_max stop must test on average **42.8 points**: every point inside B(g, 2R_max). Only **15.56** of them are natural neighbours. Your 24-44 candidates and 75-115 grid cells match this prediction almost exactly, from a glass-like distribution (jitter 0.25: 23.2 / 72.5) up to Poisson (42.8 / 99.8). The current walk is therefore already at the optimum allowed by the 2R ball criterion. Further reductions need a different stop or prune geometry, not better bookkeeping.

### Cited findings
- The tail of the circumradius of the typical Poisson-Voronoi cell (the largest distance from a vertex to the nucleus) has sharp asymptotics, proved by Calka, D'Errico and Enriquez (arXiv Nov 2025). The constant is the mean volume of a random simplex with vertices uniform on the unit sphere, under a conditioning — [Calka, D'Errico, Enriquez, arXiv:2511.11189](https://arxiv.org/abs/2511.11189v1).
- D'Errico's talk slides give the form P(R_circ ≥ t) ≈ C_d (dκ_d) t^{d(d−1)} e^{−κ_d t^d} as t→∞, with κ_d the volume of the unit ball. **Not checked against the paper.** For d=3 the tail is ~ t^6 e^{−(4π/3)t^3} — [D'Errico slides](https://www.imo.universite-paris-saclay.fr/media/filer_public/4b/1b/4b1b434e-b740-4ca8-ab8f-1f92b106caa8/derrico_presentation.pdf).
- The inradius has an exact law: P(R_in ≥ t) = e^{−κ_d (2t)^d} — [Calka & Chenavier, arXiv:1304.0170](https://pdf.arxiv.org/pdf/1304.0170).
- Calka and Chenavier show that the extremes of the inradius and circumradius over all cells in a window converge to Gumbel or Weibull laws after rescaling. This is relevant to the worst cell per kernel launch — [arXiv:1304.0170](https://pdf.arxiv.org/pdf/1304.0170).
- **[own MC]** Measured statistics. Distances are in units of mean spacing, λ=1; percentiles are p50/p90/p99/p99.9/max.

  | Quantity | Poisson | Jittered lattice ±0.5 | Jittered lattice ±0.25 (glass-like) |
  |---|---|---|---|
  | Natural neighbours (faces), mean ± sd | 15.56 ± 3.35 (max 29) | 15.47 ± 2.66 | 15.57 ± 2.20 |
  | R_max (farthest vertex), mean ± sd | 1.067 ± 0.129 | 0.992 ± 0.068 | 0.894 ± 0.038 |
  | R_max percentiles | 1.06/1.23/1.37/1.48/1.51 | 0.99/1.08/1.14/1.20/1.22 | 0.89/0.94/0.99/1.02/1.03 |
  | kNN rank of farthest natural neighbour (min k that contains all faces), mean | 30.7 | 24.4 | 19.5 |
  | Same, percentiles | 29/46/66/84/121 | 24/33/42/50/56 | 19/24/27/31/37 |
  | Points inside B(g, 2R_max) (= candidates tested by an exact nearest-first 2R walk), mean | **42.8** | 32.4 | **23.2** |
  | Same, percentiles | 41/61/83/105/127 | 32/42/50/57/60 | 23/27/31/34/39 |
  | Unit grid cells hit by B(g, 2R_max), mean | **99.8** | 83.3 | **72.5** |
  | Same, percentiles | 97/137/175/196/225 | 83/100/115/128/136 | 74/81/86/90/92 |
  | vol(union of final vertex spheres) / vol(B(2R_max)) | 0.37 | 0.43 | 0.53 |
  | Unit grid cells hit by the union of final vertex spheres, mean | 50.2 (p99 81) | 46.3 | 35.4 |

  The MC mean of 15.56 natural neighbours agrees with the known exact value E[faces] = 2 + 48π²/35 ≈ 15.535 for the 3D Poisson-Voronoi typical cell. That formula is classical (Meijering 1953) but no citation was fetched in this session.
- Why the walk count equals N(B(2R_max)) **[own derivation]**: nearest-first clipping with the stop "next candidate distance ≥ 2R_current" cannot stop before every natural neighbour has been clipped. At that point R_current = R_max(final), so the walk ends exactly when it leaves B(g, 2R_max). The tested-candidate count is therefore a property of the point set alone, independent of the implementation.
- Published kNN-based methods: votess sorts points into a grid, runs a k-nearest-neighbour search, clips a bounding box with bisectors, checks validity with a security radius, and falls back to the CPU when a cell is not certified. The fetched HTML gives no k value or failure rate — [votess, arXiv:2412.04514v2](https://arxiv.org/html/2412.04514v2).
- Paragram (SIGGRAPH 2026) uses an optional K=8 kNN warm start to pre-clip each cell and bound its initial search radius. It is enabled on RTX 5090 and disabled on H200, where the extra pass does not pay for itself — [Taveira et al., arXiv:2605.06408](https://arxiv.org/html/2605.06408v1).

### Inferences
- A fixed-k kNN that certifies cells by the security radius needs k ≥ N(B(2R_max)). For Poisson that means k ≈ 61 (p90), 83 (p99) or 105 (p99.9), against 15.5 faces. A kNN that only needs to *contain* all faces (no certification) needs k ≈ 46 / 66 / 84. Either way a fixed-k kNN tests ≥ 2-4× more points than the faces and gains nothing over your adaptive walk. It also adds a k-sized buffer per thread and a retry path for the tail.
- Your 16-entry sorted buffer is close to the mean face count but well below N(2R) (p50 = 41 for Poisson). The buffer works only because candidates stream through it nearest-first. If streaming is a sliding window, check how often the window overflows on p99 cells (83 candidates).
- The theoretical floor, with an oracle stop, is about 15.5 candidates tested and about 15.5 cuts. You are at 24-44 tested and 16-17 cuts. Cuts are already near-optimal. The waste is (a) 8-28 tested non-cutting candidates and (b) about 75-115 grid cells visited for only 24-44 points, where about 55-60% of the visited cells hold nothing useful.
- The search cost is dominated by the volume of B(2R_max) ∝ 8·E[R_max³] ≈ 8×1.27 for Poisson. Relaxing the particle distribution toward a glass (jitter 0.25) cuts candidates from 43 to 23 and grid cells from 100 to 73 with no algorithm change. Real hydro particle distributions are usually much more regular than Poisson, so measure R_max/spacing on your actual inputs.

### Gaps
- No closed-form mean or quantiles of the 3D Poisson-Voronoi circumradius were found, only tail asymptotics (Calka-D'Errico-Enriquez 2025). The MC values above fill that gap.
- I could not retrieve the k values used by Ray et al. 2018 (HAL blocked the fetch), Basselin et al. 2021 (diglib refused the connection) or votess (not in the HTML text). The code is in the Ray et al. supplementary material and at github.com/basselin7u/GPU-Restricted-Power-Diagrams, per [arXiv:2509.07175](https://arxiv.org/pdf/2509.07175), if exact k values are needed.
- Hug/Schneider/Lautensack/Møller results on circumradius moments were not located in this session.

## 2. kNN vs uniform grid vs hierarchical grid vs BVH best-first vs incremental expanding search

### Takeaway
For near-uniform data, the best quantitative evidence (Paragram's ablation) shows that the search *structure* matters little on Poisson input. Best-first ordering plus directional culling buys about 1.3× each. The large wins (27-80×) appear only on clustered or real-world data. On uniform data a GPU BVH best-first method (0.112 s for 1M points, FP32, RTX 5090, end to end) beats a kNN-grid method (Basselin*, 0.355 s for 1M, FP32, RTX 5090) by about 3.2×. That result does not separate search from clipping.

### Cited findings
- Paragram builds a binary BVH over the sites with cuBQL (leaf size 17 on H200 and 10 on RTX 5090) and traverses it best-first. Child nodes are ordered by a signed "penetration depth" into the cell's current validity region, not by Euclidean distance. The per-thread stack is an array scanned linearly at pop to find the nearest entry. Traversal ends when the stack is empty; there is no separate security-radius criterion — [arXiv:2605.06408](https://arxiv.org/html/2605.06408v1).
- Paragram culling has two parts. Directional culling takes the radius from the cell AABB corners in the octant that contains the candidate, and discards the candidate if its bisector distance exceeds that radius. Node culling discards a whole node when the lower bound d/2 (+ a weight term) exceeds r_i — [arXiv:2605.06408](https://arxiv.org/html/2605.06408v1).
- Paragram white-noise timings (FP32, end to end, warm-up runs excluded) — [arXiv:2605.06408](https://arxiv.org/html/2605.06408v1):

  | GPU | 0.1M | 0.5M | 1M | 2M | 5M | 10M | 15M |
  |---|---|---|---|---|---|---|---|
  | RTX 5090 | 0.020 s | 0.060 | 0.112 | 0.214 | 0.530 | 1.078 | – |
  | H200 | 0.017 s | 0.063 | 0.123 | 0.227 | 0.537 | 1.064 | 1.582 |

  That is about 9.3M cells/s at 10M points on H200.
- Basselin et al. 2021, reimplemented as Basselin* (kNN with a growing radius, FP32), on RTX 5090 with white noise: 0.032 s (0.1M), 0.355 s (1M), 3.608 s (10M), 5.598 s (15M). Paragram says it "works well for homogeneous point distributions" but "fails to complete any other test cases" — [arXiv:2605.06408](https://arxiv.org/html/2605.06408v1).
- In Paragram's time breakdown, BVH construction is 4.8% of the total on large sets. Inside the fused kernel, cell clipping is 96.2% and traversal plus culling the remainder — [arXiv:2605.06408](https://arxiv.org/html/2605.06408v1).
- Paragram reports gDel3D about 55% slower on H200 and about 137% slower on RTX 5090 (real-world scenes), and HXT about 55% slower on H200 for Poisson data — [arXiv:2605.06408](https://arxiv.org/html/2605.06408v1).
- Basselin et al. 2021 report a 2-3× improvement over prior GPU work (Ray et al.) for unrestricted Voronoi diagrams and 50× over CPU for restricted power diagrams — [Basselin et al., CGF 2021](https://diglib.eg.org/handle/10.1111/cgf142610) (from the search-result abstract; the PDF could not be fetched).
- Ray et al. 2018 compute each cell independently from its nearest neighbours, with no global combinatorial mesh, and decompose cells into tetrahedra on the fly for integrals. The source code is in the supplemental material — [Ray et al., TOG 2018 (HAL)](https://hal.archives-ouvertes.fr/hal-01927559).

### Inferences
- Paragram's 96% clipping / ~4% traversal split conflicts with your measured ~40% neighbour walk. Paragram works in FP32, fuses traversal and clipping, and clips nearest-first by penetration depth. A 40% walk share is therefore not inherent to the problem. It suggests your stencil walk (75-115 grid-cell visits, each with an index lookup and a branch) costs more than the BVH node visits. These are different hardware (RTX 5090/H200 vs A100) and precisions, so treat this as indicative only.
- On uniform data, the walk's dominant cost is visiting about 100 grid cells to find about 43 points. Options, roughly in order of expected payoff:
  1. Use coarser grid cells (about 2-4 particles each) so fewer cells hold the same ball (fewer index loads, more points per load).
  2. Cull grid cells with a directional bound, not the isotropic 2R ball. Paragram's ablation suggests about 1.33× on Poisson for the whole pipeline.
  3. Cull grid cells against the union of current-vertex spheres (section 3).
- A BVH is unlikely to beat a well-tuned uniform grid on near-uniform input. Its 27-80× advantage is for clustered and real-world inputs. Hydro simulations with strong density contrasts would benefit.

### Gaps
- No published per-cell candidate counts or node-visit counts for Paragram, votess, Ray or Basselin were found.
- No source compared GPU kNN libraries (FAISS, cuKNN, Garcia, Kim & Kim) specifically as Voronoi candidate generators. General GPU kNN benchmarks were out of scope for the remaining tool budget.
- No FP64 numbers exist for Paragram (its limitations section says it is floating point and not exact). Adjacency mismatches against CGAL exact predicates are 0.1-0.2% on real data — [arXiv:2605.06408](https://arxiv.org/html/2605.06408v1).

## 3. Tighter pruning than the 2R ball (vertex empty-sphere, directional, AABB bounds)

### Takeaway
A point p can cut the current cell only if it lies inside some vertex sphere B(v, |v−g|). The union of these spheres for the *final* cell covers only **37% (Poisson) to 53% (glass)** of B(2R_max) and touches about **50 instead of 100** unit grid cells. During construction the union is larger, so realised savings are smaller. The only published measurement of a related, cheaper bound is Paragram's octant-wise AABB directional culling: removing it costs 1.26-1.40× (median).

### Cited findings
- Paragram ablation (RTX 5090, median slowdown vs the full method, range in parentheses) — [arXiv:2605.06408](https://arxiv.org/html/2605.06408v1):

  | Variant | Clustered | Poisson | Real-world |
  |---|---|---|---|
  | Isotropic culling (single radius instead of directional) | 1.26× (1.03-1.52) | 1.33× (1.25-1.44) | 1.40× (1.34-1.61) |
  | Depth-first instead of best-first | 27.4× (3.96-80.3) | 1.30× (1.13-22.8) | 37.2× (34.8-53.1) |

- Edge-based clipping with point-in-cell tests (arXiv Sep 2025, restricted Voronoi): for each edge of the current cell it searches for the bisector that cuts it, using "a few" point-in-cell tests. It clips only when an edge's endpoints lie on opposite sides, so "only clippings that contribute to the final result are performed". The abstract gives no numbers — [arXiv:2509.07175](https://arxiv.org/abs/2509.07175).
- **[own MC]** Final-cell union of vertex spheres vs B(2R_max): volume ratio 0.37 / 0.43 / 0.53 and unit grid cells 50.2 / 46.3 / 35.4 (Poisson / jitter 0.5 / jitter 0.25), against 99.8 / 83.3 / 72.5 for the 2R ball. By the Delaunay empty-sphere property, the union of the final spheres contains no points except the natural neighbours on its boundary. A union-of-spheres test applied after the cell is complete would therefore reject *every* non-neighbour candidate (about 27 per cell for Poisson).

### Inferences
- In nearest-first order the vertex-sphere test cannot shorten the walk's *stop*: the farthest point of the union from g is exactly 2R_max, so the stop distance is unchanged. It can skip (a) the clip and plane test for a candidate, which is a cheap early-out, and (b) whole grid cells whose box misses every current vertex sphere, which saves up to about 50% of cell visits at the final state.
- The cost of (b) is O(#vertices ≈ 27) sphere-box tests per grid cell, which is too expensive per cell on GPU. A cheaper proxy is Paragram-style per-octant radii (8 numbers, updated after each cut) or per-direction bounds from the cell AABB. Paragram measured 1.25-1.44× from this proxy on Poisson.
- A conservative cheap variant: keep R_oct[8], the maximum vertex distance per octant around g. Accept a grid cell in octant o only if its min distance is below 2·max over the octants it touches. This is untested; I found no published number for it.

### Gaps
- No published cost/benefit numbers were found for a true per-vertex circumsphere (empty-sphere) culling of grid cells or BVH nodes during construction.

## 4. Using previous-timestep topology for moving points

### Takeaway
I found no GPU implementation with published numbers that seeds candidate discovery from previous-step natural neighbours (or neighbours of neighbours) for exact 3D Voronoi cells. Paragram explicitly lists static point sets as a limitation.

### Cited findings
- Paragram's stated limitations include single-GPU execution and *static* point sets — [arXiv:2605.06408](https://arxiv.org/abs/2605.06408).

### Inferences
- **[own derivation]** A warm start from the previous step's ~15.5 neighbours gives a nearly final cell after about 16 cuts, so R_current ≈ R_max from the start. The 2R ball walk still has to visit B(2R_max), roughly 43 points and 100 grid cells for Poisson. The warm start saves *transient* cuts (you already have only 0.5-1.5 transient cuts), not walk length.
- To shorten the walk itself, combine the warm start with certification by a Delaunay-type test: are all old neighbours' neighbours outside every vertex sphere, plus a check that no particle entered any sphere since the last step. That needs extra per-particle data and correctness reasoning about displacement bounds (a kinetic/Verlet-skin argument).

### Gaps
- No numbers were found for TESS, SILVA, cellGPU, peclet-voro or kinetic-Delaunay approaches in this session; the tool budget ran out. AREPO-style moving-mesh codes are covered in sibling notes (`Local Voronoi construction literature/astro_moving_mesh.md`).

## 5. Ordering effects (transient cuts and workspace)

### Takeaway
Proximity-first ordering is critical in general (depth-first is up to 80× slower on clustered data) but only 1.13-1.30× on Poisson (median 1.30×). Your measured 16-17 cuts against about 15.5 faces show that nearest-first ordering already leaves only about 1 transient cut per cell.

### Cited findings
- Without proximity-first ordering, "the method clips against distant, non-contributing neighbors first." Depth-first traversal median slowdown is 1.30× on Poisson, 27.4× on clustered and 37.2× on real-world data (RTX 5090) — [arXiv:2605.06408](https://arxiv.org/html/2605.06408v1).
- Paragram orders by signed penetration depth so that "the neighbors most likely to induce a cut on the current bounding volume" come first and volume reduction is maximised — [arXiv:2605.06408](https://arxiv.org/html/2605.06408v1).
- Paragram keeps per-thread planes in coalesced global memory, not shared memory, and compacts when the plane array exceeds 85% occupancy — [arXiv:2605.06408](https://arxiv.org/html/2605.06408v1).

### Inferences
- With 16-17 cuts against 15.5 faces, ordering is not where your time goes. Penetration-depth ordering could at best remove about 1 cut per cell. Its value would come from making the per-octant bounds shrink faster, which improves culling.

### Gaps
- Quantitative transient-cut statistics from Ray et al. or Lévy for random vs nearest-first ordering were not retrieved.

## 6. Divergence-aware search (work sorting, persistent threads, warp cooperation)

### Takeaway
No source fetched in this session reports divergence-aware neighbour search for Voronoi cells. The MC spread gives the size of the imbalance: on Poisson input the walk length spans 41 (p50) to 83 (p99) candidates and 97 to 175 grid cells. That is about 2× within a warp, and much tighter for glass-like input (23 to 31).

### Cited findings
- **[own MC]** Spread of B(2R_max) candidates, p50/p90/p99: Poisson 41/61/83; jitter 0.5 32/42/50; jitter 0.25 23/27/31.
- Paragram stores per-thread data in coalesced global memory and uses one thread per cell with a linear-scan priority stack — [arXiv:2605.06408](https://arxiv.org/html/2605.06408v1).

### Inferences
- With 8.7/32 active lanes, the loss is not explained by walk-length variance alone, which is at most about 2× on Poisson. Most divergence probably comes from branchy per-grid-cell control flow: empty or occupied cells, buffer insert positions and clip branches. Two directions follow:
  - Warp-cooperative walking: 32 lanes each load one stencil cell or candidate and do the distance test, then ballot and compact into the sorted buffer. This turns the walk into dense SIMD work and leaves a serial clip.
  - Coarser grid cells (several particles per cell) to cut per-cell overhead.
- Sorting cells by a cheap work estimate, such as local density or the previous step's R_max, matters mainly for non-uniform input.

### Gaps
- No published measurements were found for persistent-thread or work-queue schemes, or for warp-cooperative candidate testing, in GPU Voronoi construction. Searches for these were not reached within the tool budget.
