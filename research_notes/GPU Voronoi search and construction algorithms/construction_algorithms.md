# Construction algorithms for 3D Voronoi cells / Delaunay stars on GPUs (vs. incremental half-space clipping)

Context for the reader: our kernel does exact per-cell clipping on an A100 in double precision. It costs about 100k thread instructions per cell, makes about 16 cuts per cell, stores vertices as 3 plane ids plus adjacency, and peaks at about 16 M cells/s. The target is about 100 M cells/s. We also need face area vectors, face centroids and volumes.
Throughputs marked "derived" are my arithmetic from the paper's tables. They are not figures the authors state.

## Q1. Per-point Delaunay "star" construction (star splaying, gStar4D, lifted kNN hulls)

### Takeaway
Star splaying is a repair algorithm. It runs in linear time only when the input is already nearly Delaunay. On GPUs the NUS group used it only as a fix-up stage (gStar4D, and Phase 2 of gDel3D), never as a faster replacement for per-cell construction. I found no published GPU measurement in which per-point star or lifted-hull construction beats cell clipping in cells/s.

### Cited Findings
- Shewchuk (2005), "Star Splaying: An Algorithm for Repairing Delaunay Triangulations and Convex Hulls". It works in any dimension and takes a triangulation or approximate convex hull as input. The output is the Delaunay triangulation, weighted DT, or convex hull. "If the input is nearly Delaunay or nearly convex … and it is sparse (each input vertex adjoins only a constant number of edges), star splaying runs in time linear in the number of vertices." It is described as akin to Lawson flips but valid in any dimension. — [Shewchuk, star.pdf](https://people.eecs.berkeley.edu/~jrs/papers/star.pdf)
- gStar4D (NUS: Nanjappa, Cao, Gao, Tan). It uses neighbourhood information from a *digital* (grid-rasterised) Voronoi diagram to approximate the DT. It then builds, in massive parallel, the star of each input point lifted to 4D, and splays the stars until they are consistent. The 3D DT is the lower hull of the lifted points. The CUDA code is "up to 5x faster" than CGAL's 3D Delaunay. Code: gStar4D.zip on the project page. — [NUS gDel3D project page](https://www.comp.nus.edu.sg/~tants/gdel3d.html); [Nanjappa PhD thesis (NUS 2012)](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf)
- The gDel3D paper explains why pure star splaying is not used as the main GPU method: "directly using this approach leads to a lot of time spent constructing the convex stars of all the vertices and checking for the [consistency]". gDel3D therefore runs star splaying adaptively, on the CPU, and only on the few failed vertices. — [Cao, Nanjappa, Gao, Tan, gDel3D, I3D 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- In gDel3D's Phase 2 on uniform, Gaussian and grid inputs (about 1e6 points), only about 500 stars are built initially and fewer than 200 more are involved in splaying. On thin-sphere and sphere inputs, fewer than 1.5% of points are involved. — [gDel3D paper](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- gReg3D (regular/power triangulations) uses the same star-splaying ideas and is "up to 4x" faster than the CGAL regular triangulator. — [NUS project page](https://www.comp.nus.edu.sg/~tants/gdel3d.html)

### Inferences
- A per-point star in 3D is the convex hull of the lifted neighbours in 4D around the point. It has the same combinatorial size as our cell, since star triangles are dual to Voronoi vertices, and it needs in-sphere-degree predicates (4D orient) instead of orient3d on planes. So it does not reduce work per cell. It also shifts the costly predicate to insphere, where filters fail more often (see Q5).
- Stars also require a consistency step across points, which brings back inter-cell communication. Our independent-cell model avoids that.
- Voronoi geometry from a star costs one circumcentre per star tetrahedron (≈ our vertex count) plus a face-ring traversal. That is similar work to what clipping already produces.

### Gaps
- I found no gStar4D absolute timings, GPU model or precision. The thesis PDF could not be text-extracted within budget.
- I found no source for GPU "kNN + lifted convex hull per point" (gHull/CudaHull/QuickHull per cell) with per-cell cost numbers. gHull (Gao et al. 2013) is cited by gDel3D only as the source of the flipping approach for 3D hulls.

## Q2. Point-in-cell / edge-based, output-sensitive construction (Xiao, Cao, Chen, arXiv 2509.07175)

### Takeaway
Xiao et al. never build a kNN list. They walk polytope edges and use nearest-site queries to find exactly the bisector that bounds the cell, so every clip they perform survives into the final cell. On an RTX 3080 in float32 this gives about 10 M cells/s (derived) for 5M white-noise sites and about 19 M cells/s for blue noise, including centroids. That is 6x faster than the kNN-clipping baseline on white noise but only about 1.3–1.6x faster on blue noise. Exactness is claimed combinatorially, through integer flags, but the GPU runs used float32.

### Cited Findings
- Paper: "Efficient Computation of Voronoi Diagrams Using Point-in-Cell Tests", Yanyang Xiao, Juan Cao, Zhonggui Chen. v1 is dated 8 Sep 2025 and v2 16 Feb 2026 (submitted to IEEE TVCG). No code is linked. — [arXiv abs](https://arxiv.org/abs/2509.07175)
- **Point-in-cell test.** A point q is in the cell Ω_i if its nearest site N_q is v_i, or if ‖q−v_i‖ = ‖q−N_q‖. Each test costs one nearest-site query on a uniform grid with about 3–4 sites per grid cell in 3D. A KD-tree was rejected. There are no kNN lists. Because the equality test is unreliable in floating point, ties at polytope corners are resolved with integer flags on the incident edges and faces: if N_q's index is among the flags, q is in Ω_i. Only polytope corners and bisector–edge intersections are ever tested. — [arXiv 2509.07175v2 HTML](https://arxiv.org/html/2509.07175v2)
- **Edge-based bisector search.** Take an edge p_s→p_t with p_s ∈ Ω_i and p_t ∈ Ω_j. Intersect bisector B_ij with the edge at q, then test q. If q ∉ Ω_i, set j ← nearest site of q, p_t ← q, and repeat. If q ∈ Ω_i, B_ij is a true face and the clip is applied. The paper claims "at most one clipping is involved for each edge" and that every clip contributes. A speed-up starts the search at g = p_s + λδ·dir with δ = half the distance from v_i to its nearest neighbour, and increments λ while g's nearest site is still v_i. — [arXiv 2509.07175v2](https://arxiv.org/html/2509.07175v2)
- **Parallel variant (Alg. 3).** It builds "virtual edges" from v_i to each outside corner and finds a bisector on each. The authors say it needs more point-in-cell tests and is not advised for serial use. — [arXiv 2509.07175v2](https://arxiv.org/html/2509.07175v2)
- **GPU cell data structure.** It follows Ray et al.: an array of planes plus an array of corners, each corner being the intersection of 3 planes. Clipping removes the outside triangles (in the dual representation) and reconnects to the new plane. Float misclassification is handled by removing only R_p, the connected component that contains the chosen corner. Memory is only the NN grid, with no kNN lists. Peak memory appears only in a figure (Fig. 17), with no numbers in the text. — [arXiv 2509.07175v2](https://arxiv.org/html/2509.07175v2)
- **Cost.** There is no formal complexity bound. Producing 1M cells in a cube with random sites took "over 20.54 millions searches", about 20 NN queries per cell. Clip counts per cell are not reported. — [arXiv 2509.07175v2](https://arxiv.org/html/2509.07175v2)
- **GPU hardware and precision.** RTX 3080 (10 GB) in "32-bit floating-point". Cube domains use one thread per site. Surfaces use one thread per triangle–cell pair. Timings include construction plus centroid computation and exclude allocation and transfers. — [arXiv 2509.07175v2](https://arxiv.org/html/2509.07175v2)
- **GPU numbers (Fig. 1, 5M 3D cells in a cube, RTX 3080).**
  - White noise: 3.12 s and 2.81 s for two kNN implementations, versus 0.49 s for the proposed method.
  - Blue noise: 0.34 s and 0.42 s for kNN, versus 0.26 s for the proposed method.
  - GPU CVT on 5M white-noise sites over 100 iterations: 31.07 s versus 54.8 s for the kNN implementation of Ray et al.
  - The Ray et al. code ran out of memory at 7M white-noise sites.
  - — [arXiv 2509.07175v2](https://arxiv.org/html/2509.07175v2)
- **Serial 3D numbers.** Proposed 0.462 s, DT-based 0.6 s, kNN-based 0.684 s on white noise. On 3D CVT (10K sites, 200 iterations), Voronoi totals were 22.5 s versus 39.1 s (DT) and 85.3 s (kNN). — [arXiv 2509.07175v2](https://arxiv.org/html/2509.07175v2)

### Inferences
- Derived throughput is 5M/0.49 s ≈ 10 M cells/s for white noise and 5M/0.26 s ≈ 19 M cells/s for blue noise, on RTX 3080 in fp32 with centroids. Our 16 M cells/s in fp64 on A100 is already in the same range. The method's main gain is robustness to bad distributions (no oversized kNN), not raw speed on well-spaced points.
- The trade is one clip per *final* face in exchange for about 20 grid NN queries per cell. Each NN query is a divergent, memory-bound grid walk. With about 16 faces per cell and good kNN ordering, our wasted clips may be few. The method wins only if our profile shows many non-contributing cuts or kNN-overflow retries.
- Exactness depends on the NN query being exact (comparing squared distances). That is easy to do exactly with integer or fixed-point coordinates, which suits an integer-position pipeline.

### Gaps
- Clips per cell, instructions per cell and per-cell memory are not reported. The last ~22k characters of the v2 HTML (tail of the experiments and limitations) were not read.
- I found no other published output-sensitive GPU Voronoi method. The closest related CPU idea is "corner validation" by Sainlot et al. (2017), which I did not verify in this session.

## Q3. Half-space intersection via duality, gift-wrapping, Delaunay walk, radial sweep

### Takeaway
I found no GPU paper with per-cell cost numbers for dual convex hull, gift-wrapping or Delaunay-walk construction. Every GPU per-cell Voronoi code I found (Ray 2018, Basselin 2021, votess 2024, Taveira 2026, Xiao 2025) uses incremental clipping of a convex polytope stored in the dual triangle form ("vertex = triple of plane ids"). In effect that already is an incremental dual convex hull.

### Cited Findings
- Taveira et al. (2026) describe each cell as "computationally independent, defined solely by the intersection of half-spaces derived from its neighbors". They implement it as clipping an AABB by bisectors in proximity order, with each vertex stored as a triple of plane indices. — [arXiv 2605.06408](https://arxiv.org/html/2605.06408)
- votess (Singh, Byrohl, Nelson, 2024–25) uses a grid, then kNN, then iterative clipping of a bounding box by bisectors. Validity is checked with a "security radius", and cells that cannot be validated fall back to the CPU. — [arXiv 2412.04514v2](https://arxiv.org/html/2412.04514v2)
- Ray, Sokolov, Lefebvre, Lévy (TOG 37(6) 2018) compute each cell independently from its kNN, with "constant memory usage per thread" and no synchronisation. They report "one order of magnitude" gain over the fastest multicore CPU codes. — [HAL hal-01927559](https://hal.archives-ouvertes.fr/hal-01927559) (abstract only; full text blocked by bot protection)

### Inferences
- Incremental half-space intersection in the dual (triangle) representation *is* an online dual convex hull. Dual gift-wrapping would need all candidate planes up front and an O(h·k) wrap, so it has no obvious per-cell advantage at k ≈ 30–50.
- A "Delaunay walk" around the generator (rotating about Delaunay edges) needs a robust insphere and a neighbour oracle. Xiao et al.'s bisector search is the closest practical form of this idea (Q2).

### Gaps
- No GPU measurements found for gift-wrapping, radial sweep or Delaunay-walk per-cell construction.

## Q4. Global GPU Delaunay (gFlip3D, gDel3D, Radiant Foam, newer work) and Voronoi-from-Delaunay cost

### Takeaway
Global GPU Delaunay codes run at about 4–5 M points/s on modern GPUs (gDel3D, double). They produce topology only. In 2026 the fastest published GPU 3D Voronoi builder (Taveira et al., SIGGRAPH 2026) is again per-cell clipping, in float32. It is about 1.5–2.4x faster than gDel3D on real scenes and runs at about 8 M cells/s on uniform points. Our 16 M cells/s in double on A100 is already competitive with all published numbers I found.

### Cited Findings
- **gFlip3D** (Nanjappa thesis, 2012) uses one insertion and one flip per thread. It is up to 6x faster than CGAL, and fewer than 0.0001 of the output facets are not locally Delaunay. — [Nanjappa thesis](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf) (via search snippet)
- A secondary, unverified claim: "4 to 5 million points per second on datasets larger than 100 million points" and up to 24x over CPU for gFlip3D-style insertion. Hardware is not stated. — [ASIAN COMNET 2025 abstract](https://asiancomnet2025.aconf.org/presentation/212.html)
- **gDel3D** (I3D 2014) works in two phases. Phase 1 runs on the GPU and alternates parallel batch insertion with bilateral flipping. Phase 2 runs adaptive star splaying on the CPU. The original hardware was a GTX 580 (3 GB) against CGAL 4.2 on an i7-2600K. Speedups reach "an order of magnitude" and climb to 8–10x as n grows from 1e5 to 1.5e6. On a GTX Titan, gDel3D is "memory bound", with only about 30% speedup over the GTX 580. — [gDel3D paper](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **gDel3D on modern GPUs** (from Taveira et al. Table 1, double precision):
  - bicycle (4.2M points): 0.893 s on H200 and 1.204 s on RTX 5090.
  - room (1.9M points): 0.424 s on H200.
  - Derived rate: about 4.5–4.7 M points/s on H200.
  - Running gDel3D in single precision was only about 0.5% faster, so precision is not its bottleneck.
  - — [Taveira et al., arXiv 2605.06408](https://arxiv.org/html/2605.06408)
- **Taveira, Lindström, Fatemi, Hammarstrand, Kahl, "Scalable GPU Construction of 3D Voronoi and Power Diagrams"** (SIGGRAPH 2026). Code at research.zenseact.com/publications/paragram.
  - It uses one thread per cell, clipping in proximity order.
  - Candidates come from best-first traversal of a GPU BVH (cuBQL). Candidates are culled when the bisector distance exceeds the cell's directional radius, using per-octant AABB corner distances.
  - An optional K=8 kNN warm start is used.
  - Fully clipped planes are compacted at 85% occupancy, which replaces Basselin's overflow-and-relaunch.
  - Per-thread data sits in coalesced global memory rather than shared memory, for occupancy.
  - Precision is float32, with "no exact guarantees". On real data, 0.1–0.2% of adjacencies mismatch CGAL, traced to near-coincident planes.
  - — [arXiv 2605.06408](https://arxiv.org/html/2605.06408)
  - Timing breakdown: BVH build is 4.8% of time. Inside the fused kernel, cell clipping is 96.2% and traversal plus culling about 3.6% of total. Clipping dominates even in the state of the art. — [arXiv 2605.06408](https://arxiv.org/html/2605.06408)
  - White noise, H200: 0.132 s for 1M, 1.208 s for 10M, 1.811 s for 15M (derived ≈ 8.3 M cells/s). RTX 5090: 1.216 s for 10M. — [arXiv 2605.06408](https://arxiv.org/html/2605.06408)
  - Density-gradient data, H200: 1.588 s for 15M (≈ 9.4 M/s). Basselin* reimplementation: 5.574 s. — [arXiv 2605.06408](https://arxiv.org/html/2605.06408)
  - On clustered data with small n, gDel3D beats it (1M points: 0.229 s for gDel3D versus 0.720 s on H200). At 5M clustered points gDel3D degrades to 45.4 s. — [arXiv 2605.06408](https://arxiv.org/html/2605.06408)
- **Radiant Foam** (arXiv 2502.01157) rebuilds the Delaunay/Voronoi structure as sites move. Its own GPU Delaunay is far slower than gDel3D or Taveira: 16–250 s for 2–4M-point scenes, per Taveira Table 1. — [arXiv 2605.06408](https://arxiv.org/html/2605.06408); [Radiant Foam](https://arxiv.org/pdf/2502.01157)
- **Radiance Meshes** (arXiv 2512.04076) reports about 92 ms for Delaunay on a scene with 1.2M tetrahedra, with topology updated only every 10 iterations (search snippet, not verified). — [arXiv 2512.04076](https://arxiv.org/pdf/2512.04076)
- **Basselin et al.**, "Restricted Power Diagrams on the GPU" (CGF 40(2) 2021). Per a search snippet, its figure shows 10M-point 3D Voronoi restricted to a mesh in 300 ms on a V100 (unverified; the PDF host refused connection). In Taveira et al.'s tests, the Basselin-style code produced no valid output on real or clustered power-diagram inputs. — [Basselin et al. EG diglib](https://diglib.eg.org/items/43d82919-34bf-4096-a2fd-5784207f8624/full); [arXiv 2605.06408](https://arxiv.org/html/2605.06408)

### Inferences
- If topology is maintained (Delaunay kept between steps and repaired by flips), Voronoi geometry costs one circumcentre per tetrahedron. That is about 6.5 tets per point in uniform 3D, a standard figure not re-verified here. Each tet is shared by 4 cells, so each Voronoi vertex is computed once rather than about 4 times as in per-cell clipping. Face rings per Delaunay edge then give area vectors and centroids.
  - This could undercut per-cell clipping by up to about 4x in vertex computations.
  - The cost is a global, synchronised structure, flips with exact insphere, and irregular memory access. gDel3D was already memory-bound on a Titan.
  - No paper I found measures "flip-repair + geometry" throughput per step on GPU.
- Published per-cell GPU numbers top out at about 8–19 M cells/s: Taveira about 8 M/s fp32 on H200; Xiao about 10–19 M/s fp32 on RTX 3080 with centroids. None reaches 100 M cells/s. Reaching that target would need a large constant-factor gain or amortising topology across time steps, not a different known algorithm.

### Gaps
- No public GPU number for incremental Delaunay repair per time step (kinetic or flip-based) with geometry extraction.
- Taveira's Poisson table and per-cell memory were not extracted.
- Liu et al. "parallel 3D clipped Voronoi" (cited by Xiao et al. as a baseline, tested to ≤100K sites) was not located.

## Q5. Exact predicates on GPU (filters, expansions, integer/fixed point, SoS)

### Takeaway
The proven GPU pattern is a float filter followed by a separately launched exact pass over a compacted set of marked threads, as in gDel3D. It keeps divergence and register/memory pressure out of the fast path. I found no published GPU measurement of 128-bit-integer orient or insphere overhead.

### Cited Findings
- **gDel3D.**
  - Uses Shewchuk's adaptive predicates on the GPU: a fast floating-point check, then an exact check with floating-point expansions.
  - Mixing both checks in one kernel causes divergence. The exact path also needs "a lot more temporary memory, and hence the number of threads we can launch is much smaller".
  - So each predicate kernel is split in two. The first does fast checks and marks failures. The second runs exact checks only on marked threads, compacted on the fly in shared memory with one atomicAdd per block.
  - Exact computation is used only in the final flipping iterations.
  - Thanks to this, performance on thin-sphere and grid (degenerate) inputs is "very close" to uniform, while the sphere distribution needs more exact computation.
  - — [gDel3D paper](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **GPredicates** (IEEE Access, 2019) is a CUDA port of Shewchuk's adaptive orient2d/3d and incircle that also splits fast and exact kernels. No numbers were extracted. — [DOAJ record](https://doaj.org/article/c2df885fc0ed43cca44cea9e50b89ff6)
- **Menezes et al.** (IMR 2019) run interval-arithmetic filters on a CUDA GPU with rational re-evaluation on the CPU. They report up to 289x speedup in predicate evaluation and up to 40x for the whole algorithm (2D segment intersection). — [Menezes et al., IMR 2019](https://internationalmeshingroundtable.com/assets/papers/2019/14-Menezes.pdf)
- **Devillers–Preparata analysis.** Orientation is rarely near-degenerate, with Θ(V) probability. Insphere has sublinear bounds, so it falls to the exact path more often. — [arXiv cs/9907028](https://arxiv.org/pdf/cs/9907028)
- **Simulation of Simplicity.** An implementation paper exists (arXiv 2212.08226), but its content was not retrieved. Lévy's exact mesh-CSG paper (arXiv 2405.12949) uses SoS-style consistent tie-breaking on the CPU. — [arXiv 2212.08226](https://arxiv.org/pdf/2212.08226); [arXiv 2405.12949](https://arxiv.org/pdf/2405.12949)
- **Float-only clipping in practice.** Taveira's fp32 per-cell clipping gives 0.1–0.2% adjacency mismatches against CGAL on real data. Xiao et al. avoid disconnected clip results in fp32 through the connected-component (R_p) rule plus integer flags. — [arXiv 2605.06408](https://arxiv.org/html/2605.06408); [arXiv 2509.07175v2](https://arxiv.org/html/2509.07175v2)

### Inferences
- For our per-cell kernel, the gDel3D lesson carries over: keep the exact or integer path out of the main kernel and re-run flagged cells in a second, compacted launch. A "retry" kernel along these lines appears to be what local_voronoi_retry.hpp already does.
- Clipping only needs the sign of a plane–vertex test, which is an orient-like degree-4 determinant in the coordinates when vertices are implicit triples of planes. On a GPU, which lacks native 128-bit multiply, a mostly-pass float filter is likely cheaper than always-on wide-integer arithmetic. This is inference, not measured.

### Gaps
- No measured GPU cost for 128-bit integer orient or insphere, or for SoS on GPU, was found.
- GPredicates timing numbers were not extracted.

## Q6. Memory per cell, divergence, thread-per-cell vs warp-per-cell (cross-cutting)

### Takeaway
Every high-performing per-cell GPU method in the literature uses thread-per-cell with bounded per-thread polytope storage. Recent work moves that storage to coalesced global memory for occupancy (Taveira) or drops kNN lists for an NN grid (Xiao). None reports warp-per-cell.

### Cited Findings
- Ray et al.: "constant memory usage per thread", no synchronisation. — [HAL abstract](https://hal.archives-ouvertes.fr/hal-01927559)
- Taveira: one thread per cell. Per-thread intermediate data sits in coalesced global memory, not shared memory, to improve occupancy. Scale is limited by device memory, not compute. Plane-array garbage collection runs at 85% occupancy. Inputs are packed as float4. — [arXiv 2605.06408](https://arxiv.org/html/2605.06408)
- Xiao: one thread per site, no kNN lists, only an NN grid. Atomics are used for queues. The Ray et al. kNN implementation ran out of memory at 7M white-noise sites on 10 GB. — [arXiv 2509.07175v2](https://arxiv.org/html/2509.07175v2)
- gDel3D: memory-bound on newer GPUs. The exact-predicate path is split out to limit register and memory pressure. — [gDel3D paper](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)

### Inferences
- Taveira's profile (96% in clipping) and Xiao's (white noise costs 6x more than blue noise for the kNN baseline) suggest one conclusion about the bottleneck. It lies in the clipping inner loop and in wasted clips on irregular inputs, not in the choice of algorithm family.
- On this evidence, incremental clipping remains the right core. The levers with published support are:
  1. Fewer wasted clips: output-sensitive bisector search (Xiao) or directional-radius culling (Taveira).
  2. Exact work deferred to a compacted second pass (gDel3D).
  3. Amortising topology across time steps, the only route that removes the ~4x duplicated vertex work. This one is unmeasured.

### Gaps
- No published warp-per-cell or cooperative-group Voronoi construction numbers were found.
- No per-cell instruction counts were published by any of these papers, so a direct comparison with our ~100k instructions per cell is not possible.
