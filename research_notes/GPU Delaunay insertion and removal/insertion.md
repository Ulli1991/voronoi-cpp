# Parallel and GPU point insertion into Delaunay triangulations (2D/3D), with emphasis on small batches into a persistent mesh

Scope note: these notes cover insertion only. All throughput figures marked "(derived)" are my arithmetic on the published timings (points / seconds). Figures marked "(estimate)" were read off plots or inferred and are not stated in the source. Primary PDFs were downloaded and their text extracted. That covers gDel3D I3D 2014, Nanjappa's thesis, Marot et al. 2019, Blandford et al. 2006, gQM3D 2019, gDP2d/gQM 2017, Radiant Foam 2025, and Taveira et al. 2026.

## 1. gDel3D / gDel2D (Cao, Nanjappa, Gao, Tan; I3D 2014) and Nanjappa's thesis (2012/2015): algorithm, speed, robustness, failure modes

### Takeaway
gDel3D builds the triangulation in rounds. In each round, every tetrahedron that contains uninserted points inserts one of them with a plain 1-4 split, and then the mesh is flipped in parallel (2-3/3-2) until no flippable non-Delaunay facet is left. A few facets remain non-Delaunay but unflippable, so 3D flipping does get stuck. The CPU repairs these with adaptive star splaying. In 2014 (GTX 580, double precision, Shewchuk-style exact predicates plus Simulation of Simplicity) it ran 6-10x faster than CGAL 4.2. On a 2026 RTX 5090 or H200 it still builds a 2-4M-point DT in about 0.4-1.2 s, i.e. about 3.5-4.7 M points/s (derived). Clustered or co-spherical inputs slow it down badly.

### Cited Findings
- **Phase 1 (GPU).** Each round, "each tetrahedron with points inside picks one of them to insert". The tetrahedron is split, the new facets are labeled, and flipping is repeated until no locally non-Delaunay *flippable* facet remains. Then the locations of the uninserted points are updated. "Point insertion refers to splitting the tetrahedron containing the point, without any further modification." This is a 1-4 split, not a Bowyer-Watson cavity. — [Cao et al., I3D 2014 (PDF)](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **Why insertion and flipping alternate (InsFlip).** Flipping after every insertion round, instead of inserting everything first (InsAll), cuts the number of failed vertices by "nearly two orders of magnitude" and the number of flips by about 40%. It also raises the tet count from 3m toward the DT value of about 6.67m, so the next round can insert more points in parallel. — [Cao et al. 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **Which point each tet inserts.** Each tet inserts the point nearest its circumcenter, which is the point furthest from the lifted face. Compared with random choice, this cuts failed vertices by more than 2x on uniform data and by 2-4x on real models, cuts flips by 20%, and roughly halves total time on real models. — [Cao et al. 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **Phase 2 (CPU, adaptive star splaying, after Shewchuk 2005).** Stars are rebuilt only for "failed vertices", i.e. vertices incident to a locally non-Delaunay facet. Each lifted star is made convex with Flip-Flop, made consistent by splaying, and patched back into T. On uniform, Gaussian and grid inputs (5×10^5 points) only about 500 initial stars plus fewer than 200 extra stars are needed. Thin-sphere, sphere and real inputs need under 1.5% of the point count. — [Cao et al. 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **Point location for uninserted points.** After a split, points are relocated into the 4 children with orient3d tests. After flipping, they are relocated through a *history DAG* of flips (each node stores 5 vertices and 3 tets, plus up to 3 child pointers). Without the DAG, relocation "would take nearly two times longer". — [Cao et al. 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **Flip scheduling.** The flip loop is one thread per active tet, deduplicated by tet index. It uses stream compaction of active tets while the active set is large, then switches to collecting the up to 3 modified tets per flip into a pre-allocated 3× array. Tets are re-sorted by smallest vertex index after each insertion round, and points are Hilbert-sorted. — [Cao et al. 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **Robustness.**
  - Shewchuk's adaptive exact predicates (floating-point filter, then floating-point expansion) were ported to the GPU, with Simulation of Simplicity for degeneracies.
  - Each predicate kernel is split in two: a fast-check kernel marks the uncertain threads, and a second kernel runs the exact checks, with shared-memory on-the-fly compaction plus one atomicAdd per block.
  - Exact computation is used only in the flipping iterations near the end, "pushing all the flips that involve degenerate configurations to be processed together".
  - — [Cao et al. 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **Precision.** Double by default. Single precision is a compile option (`REAL_TYPE_FP32`). It was tested on GTX 450/460/470/580 and GTX Titan. — [gDel3D README, GitHub](https://github.com/ashwin/gDel3D)
- **2014 speed (i7-2600K + GTX 580 3 GB, vs CGAL 4.2).**
  - Synthetic inputs of 10^5 to 1.5×10^6 points: speedup "starts at 4–6 times, and quickly rises to 8–10 times".
  - Real models: Armadillo 172,974 points in 0.4 s (6.1x); Happy Buddha 543,652 points in 1.1 s (7.2x); Blade 882,954 points in 1.8 s (6.7x); Asian Dragon 3,609,600 points in 5.4 s (9.2x).
  - That is about 0.43-0.67 M points/s on a GTX 580 (derived).
  - Time grows linearly with input size.
  - — [Cao et al. 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **Time breakdown.** On uniform, Gaussian and grid inputs, most of the time goes to flipping. Splaying takes a larger share on thin-sphere inputs and dominates on co-spherical inputs. — [Cao et al. 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **GPU scaling and memory-boundedness.**
  - The GTS 450 to GTX 580 step gives 3x.
  - The GTX Titan is only about 30% faster than the GTX 580 because "gDel3D is actually memory bound".
  - The paper's conclusion says the implementation "is currently memory bound … will benefit from improvements in the data structure".
  - — [Cao et al. 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **Stated limitation.** On pathological inputs, such as points on two non-intersecting lines in 3D or on a circle in 2D, "very few flips can be performed in parallel" because one flip must happen before the next can be discovered. — [Cao et al. 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **gDel2D** (section 5.6 of the same paper). Only Phase 1 is needed, because 2D flipping never gets stuck (Lawson 1977). On uniform inputs up to 10^7 points it is up to 10x faster than Triangle, more than 6x faster than CGAL, and 2x faster than GPU-DT. On thin-circle inputs it is more than 5x faster than GPU-DT, even when GPU-DT uses an 8192² grid. Sorting gives more than 2x faster insertion and nearly 4x faster relocation. InsFlip cuts flip time by about 30% in 2D. — [Cao et al. 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **Thesis (earlier versions).**
  - gFlip3D (GPU insertion plus flipping, near-Delaunay output) reaches up to 6.5x over CGAL on synthetic inputs, 3x on grid, and 2.16-3.87x on real inputs.
  - The thesis-era gDel3D is up to 6x over CGAL. Its real-input times are slower than the I3D paper's: Armadillo 1.14 s, Happy Buddha 2.72 s, Blade 6.92 s. Repair took 0.17-1.04 s of that.
  - gStar4D reaches up to 5x and gReg3D (regular triangulation) up to 4x over CGAL.
  - — [Nanjappa PhD thesis, NUS (PDF)](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf); [gDel3D project page](https://www.comp.nus.edu.sg/~tants/gdel3d.html) (the page claims "up to 10 times")
- **Thesis statement.** "It is well known that flipping to Delaunay works in 3D only with incremental insertion." This means one point inserted into a DT and then flipped (Joe 1991). — [Nanjappa thesis](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf)
- **gDel3D on 2026 hardware** (Taveira et al., SIGGRAPH 2026, arXiv 2605.06408). Full-rebuild times on Radiant Foam point sets:

  | Scene | Points | RTX 5090 | H200 | Rate (derived) |
  |---|---|---|---|---|
  | bicycle | 4.2M | 1.204 s | 0.893 s | 3.5 / 4.7 M pts/s |
  | bonsai | 2.0M | 0.570 s | 0.428 s | 3.5 / 4.7 M pts/s |
  | garden | 4.1M | 1.168 s | 0.862 s | |
  | stump | 4.2M | 1.197 s | 0.896 s | |

  - CGAL on 16 EPYC 9534 cores: 20.4 s (bicycle). Geogram: 1.59 s. HXT: 1.24 s.
  - **Synthetic "10 clusters", 5M points:** gDel3D takes 45.4 s on the H200, about 0.11 M pts/s (derived). That is a roughly 40x collapse on clustered input.
  - **Synthetic "10 clusters", 0.1M points:** gDel3D takes 0.056 s.
  - — [Taveira et al. 2026, arXiv 2605.06408](https://arxiv.org/html/2605.06408v1)
- **Independent benchmark (single author, provisional).** On LiDAR clouds whose points sit on a sensor grid (many exactly co-spherical), gDel3D falls back to exact arithmetic often enough to become much slower and to produce **wrong output**. After perturbation it was the fastest method tested (0.046 s for 100,000 points, identical to CGAL). — summarized from search results for [DelaunayBench (GitHub)](https://github.com/omarbesbes/DelaunayBench); I could not fetch the repo page itself (404), so this is unverified.

### Inferences
- gDel3D's 1-4 split plus 2-3/3-2 flipping, with flips chosen per tet in parallel, is structurally the same machinery as the user's flip repair. Its insertion step is just "split the containing tet". So gDel3D's Phase 1, minus the from-scratch scaffolding (big tet, per-tet point lists, history DAG), is essentially "batch-insert by 1-4 split, then run the existing flip kernel".
- What gDel3D adds that the user's pipeline lacks is a guaranteed exit when flips get stuck (star splaying, on the CPU). That exit is needed only for simultaneous multi-point insertion. A single point inserted into a DT and flipped never gets stuck (Joe 1991, as cited in the thesis).
- gDel3D's failure rate gives a sense of scale. Fewer than about 700 stars per 5×10^5 uniform points, i.e. under 0.15% of vertices, needed repair even when *all* points were inserted in rounds. A 1% batch into an already Delaunay mesh should trigger far fewer stuck configurations. This is an estimate and has not been measured.
- The 2026 RTX 5090/H200 figure (about 3.5-4.7 M pts/s, double, exact plus SoS) is the best published *GPU Delaunay construction* rate I found for gDel3D. It is a full rebuild, not incremental insertion.

### Gaps
- No standalone gDel2D paper with absolute 2D timings was found. The gDel2D repository README could not be retrieved (GitHub API 404). Absolute gDel2D points/s on modern GPUs is therefore unknown; only speedup ratios are available.
- The I3D paper's synthetic absolute timings are only in plots. On the GTX 580 at 10^6 uniform points I estimate about 1.5-2 s from Fig. 10, i.e. 0.5-0.7 M pts/s (estimate).
- I did not find gDel3D figures for how much of the time is spent in the exact-predicate path on degenerate (integer-grid) inputs.

## 2. Other GPU and parallel Delaunay builders

### Takeaway
- **Fastest published CPU builder:** Marot, Pellerin and Remacle's HXT (2019). It uses sequential Bowyer-Watson that is about 3x faster than CGAL/Geogram/TetGen, and parallelizes it lock-free through Moore-curve partitions, with aborts and repartitioning. It reaches about 10^7 points/s (about 65M tets/s) on a 64-core EPYC.
- **Earlier GPU 2D methods** (GPU-DT 2008, GPU-CDT 2012) used digital Voronoi diagrams and do not extend to 3D.
- **GPU refinement (gQM3D 2019):** the most directly relevant design for inserting points into an *existing* GPU mesh. It uses Bowyer-Watson cavities claimed by atomicMin priority marking.
- **2024-2026 GPU work** has mostly moved to per-cell clipping (no topology, no exactness), e.g. Taveira et al. 2026, or to incremental Delaunay inside neural rendering (Radiant Foam 2025).

### Cited Findings
- **Marot, Pellerin, Remacle, "One machine, one minute, three billion tetrahedra"** (arXiv 2018, IJNME 2019).
  - **Sequential speed** (i7-6700HQ, Bowyer-Watson):

    | Points | HXT | Geogram 1.5.4 | CGAL 4.12 | TetGen |
    |---|---|---|---|---|
    | 10^6 | 2.03 s | 5.53 s | 6.65 s | 5.89 s |
    | 10^7 | 21.66 s | 56.02 s | 66.24 s | 63.99 s |

    That is about 0.46-0.49 M pts/s for HXT and about 0.15 M pts/s for CGAL (derived).
  - **Breakdown for 5M points** (HXT vs CGAL): WALK 1.0 vs 1.4 s; CAVITY 6.2 vs 14.9 s (of which inSphere 3.2 vs 10.5 s); DELAUNAYBALL 4.5 vs 15.3 s.
  - **Per-insertion cost:** about 20 tets deleted, about 26 created, about 46 inSphere calls, and about 15 vertices on the cavity boundary.
  - **Speed tricks:**
    - Store 4 inSphere sub-determinants per tet, which gives a 32% inSphere speedup. This is dropped in the parallel version because bandwidth is saturated.
    - Use an n×n (n_max = 32) edge lookup table for cavity adjacency.
    - Order points with BRIO plus a Moore curve.
  - **Robustness:** Shewchuk filtered predicates plus Edelsbrunner symbolic perturbation.
  - — [arXiv 1805.08831v3](https://arxiv.org/abs/1805.08831v3)
- **Marot et al., parallel scheme.**
  - **Ownership:** "A tetrahedron belongs to a thread if at least three of its vertices are in that thread's partition".
  - **Aborts:** a thread aborts an insertion if its WALK leaves the partition or its CAVITY reaches the buffer zone. Aborted points are re-partitioned by shifting the Moore curve.
  - **Thread count:** halved when the success ratio ρ < 1/5 or when fewer than 3000 points remain per thread. The first BRIO round and the final tail run sequentially.
  - **Success rates (1M points, 8 threads):** BRIO round 4 inserted 503,730 of 602,112 points (84%) with 8 threads. Then 46%, 59%, 36% on the retries. Overall 94.5% of points were inserted with 8 threads.
  - **Synchronization:** memory reallocation happens between OpenMP barriers; nothing else synchronizes.
  - — [arXiv 1805.08831v3](https://arxiv.org/abs/1805.08831v3)
- **Marot et al., parallel speed.**

  | Machine | Points | HXT | Geogram | CGAL |
  |---|---|---|---|---|
  | 4-core i7 | 10^7 | 7.40 s (≈1.35 M pts/s, derived) | 17.11 s | 23.37 s |
  | 64-core Xeon Phi 7210 | 10^8 | 28.95 s (≈3.5 M pts/s, derived) | – | – |
  | 64-core Xeon Phi 7210 | 10^7 | – | 43.70 s | 20.15 s |

  - On the 64-core AMD EPYC 7551: 10^9 tets in 17.8 s and 3×10^9 tets in 53 s, which the paper describes as "about 10^7 points per second". The average rate is 65 M tets/s.
  - The tet rate grows with mesh size "because the proportion of conflicts is lower".
  - — [arXiv 1805.08831v3](https://arxiv.org/abs/1805.08831v3)
- **Blandford, Blelloch, Kadow** (SODA/SoCG-era 2006, "Engineering a compact parallel Delaunay algorithm in 3D").
  - Parallel Bowyer-Watson on 64 × 1.15 GHz EV67 (HP GS1280): 1.51×10^9 vertices and more than 10^10 tets in 5512 s, about 0.27 M pts/s (derived). One processor inserting 1/64 as many vertices took 3202 s.
  - Compressed mesh: 197 GB instead of an estimated 720 GB for pointer-based connectivity.
  - Conflicts are handled with "test-locks" on vertices: on failure the thread "aborts the insertion: it releases all of its locks and returns the tetrahedron to the work queue".
  - Contention when the mesh is small is avoided by allowing at most one thread per 2^14 vertices.
  - — [Blandford et al. 2006 (PDF)](https://www.cs.cmu.edu/~./blelloch/papers/BBK06.pdf)
- **Batista, Millman, Pion, Singler 2010** (CGAL multi-core). Bowyer-Watson with vertex locks or cell locks. A sequential bootstrap builds the initial triangulation, then points are space-filling-curve sorted and partitioned, with work stealing. A thread that loses a lock "tries insertion of a different point". Speedup was up to 7x on 8 cores at 10^7 uniform points. On ellipsoid inputs there was no speedup with vertex locks and 4x with cell locks. — [Nanjappa thesis, related-work chapter](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf) (secondary summary of Batista et al., Comput. Geom. 43(8), 2010)
- **CGAL today** ships parallel insertion through TBB with a lock data structure over a bounding box (`parallel_insertion_in_delaunay_3.cpp`, CGAL 5.2-6.1). The example is a correctness demo, not a benchmark. — [CGAL 6.1 docs](https://doc.cgal.org/latest/Triangulation_3/Triangulation_3_2parallel_insertion_in_delaunay_3_8cpp-example.html)
- **Kohout, Kolingerová, Žára 2005.** Parallel flip-based insertion on a history DAG with optimistic locking and per-thread journals for rollback: 3.6x on 4 cores at 0.5M points in 3D. The sequential speed was "an order of magnitude lower than the CGAL implementation". — [Nanjappa thesis](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf)
- **Foteinos and Chrisochoides 2012 (PDT, dynamic).** Bowyer-Watson with vertex locks and rollback, and jump-and-walk point location: 6.59x with 12 threads and 13.22x with 48 threads at 12M points. — [Nanjappa thesis](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf)
- **GPU-DT (Rong, Tan, Cao, Stephanus, I3D 2008).**
  1. Compute a digital Voronoi diagram with the Jump Flooding Algorithm on the GPU.
  2. Dualize.
  3. On the CPU, fix the hull, shift sites back to their real coordinates, and re-insert missing sites by point location plus 1-to-3 splits.
  4. Edge-flip on the CPU.

  It was 1.53x faster than Triangle on an 8800 GTX, and the CPU stages took "sometimes … more than 75%" of the time. — [Nanjappa thesis](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf)
- **GPU-CDT (Qi, Cao, Tan, I3D 2012).** Runs entirely on the GPU with the Parallel Banding Algorithm. Missing sites are inserted on the GPU "using the atomic operations of CUDA to ensure that no two insertions can simultaneously modify one triangle", followed by massively parallel edge flipping. It was 4.5x faster than CGAL on a GTX 580. The digital-Voronoi approach "is not extensible to 3D". — [Nanjappa thesis](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf); [Cao et al. 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **gQM3D / gQM3D+ (Chen and Tan, PACT 2019): GPU 3D constrained Delaunay refinement**, i.e. inserting Steiner points into an existing mesh.
  - **Pipeline (InsertPoints):** (a) compute and locate splitting points, (b) grow cavities, (c) shrink cavities, (d) remesh cavities.
  - **Conflict handling:** "an element e is to first mark a tetrahedron t, using the atomic operation to set t to the priority of e … After all markings … are synchronized, those who successfully marked their tetrahedra can then add the corresponding tuples". If e loses a mark, its cavity is aborted for this round. If e wins a tet that q already held, q's cavity is aborted. The survivors are "Delaunay independent" and are inserted concurrently.
  - **Cavity growth:** one GPU thread per (element, tet, face) tuple runs the insphere test on the neighbor across that face.
  - **Remeshing:** creates one tet per boundary facet. Internal adjacencies are found by rotating around an edge outside the cavity.
  - **Robustness:** double precision, Shewchuk predicates ported to the GPU, and Simulation of Simplicity. Cavity growth stops after "a large number of iterations" on degenerate tets, and such tets are marked unsplittable.
  - **Speed:** about an order of magnitude over TetGen on an i7-7700K plus GTX 1080 Ti. For example, 4.29M output points and 27.2M tets in 2.5 min, versus 53.4 min for TetGen. This is end-to-end refinement including the initial CDT, about 29k inserted pts/s (derived); it is not comparable to raw insertion throughput.
  - — [Chen and Tan, gQM3D, PACT 2019 (PDF)](https://www.comp.nus.edu.sg/~tants/gqm3d_files/gqm3d.pdf)
- **gDP2d / gQM (Chen, Qi, Tan, I3D 2017): 2D GPU refinement.**
  - New points are inserted by splitting triangles and then running FlipFlop: "flip to Delaunay; flop to remove point".
  - Independence is approximated with atomicMin marking, with priority given to the shortest edge. Points that turn out not to be Delaunay independent "are recognized and removed from the mesh by FlipFlop".
  - The design rule: concurrent independent insertions "can lead to redundant ones".
  - Hardware: GTX 980 Ti.
  - — [Chen, Qi, Tan 2017 (PDF)](https://www.comp.nus.edu.sg/~tants/gqm_files/11-0018-chen.pdf); [gQM project page](https://www.comp.nus.edu.sg/~tants/gqm.html)
- **Radiant Foam (Govindarajan et al., ICCV 2025).**
  - Maintains a 3D Delaunay/Voronoi adjacency during training with custom CUDA kernels. "Whenever the primal vertex positions are changed we must update the adjacency by performing an incremental Delaunay triangulation. While much faster than a complete rebuild, the incremental update is still computationally expensive."
  - The mesh is rebuilt every optimizer step after densification at first, then every 100 steps. Training takes 70 min on an RTX 4090 for "bonsai".
  - — [Radiant Foam, arXiv 2502.01157](https://arxiv.org/pdf/2502.01157)
  - Used as a *full* builder, Radiant Foam's Delaunay code took 124.9 s for 4.2M points on an RTX 5090, versus 1.2 s for gDel3D. — [Taveira et al. 2026](https://arxiv.org/html/2605.06408v1)
- **Taveira, Lindström, Fatemi, Hammarstrand, Kahl (SIGGRAPH 2026), "Scalable GPU Construction of 3D Voronoi and Power Diagrams".**
  - Each cell is clipped by bisector planes of candidates found with a power-augmented BVH. There is no insertion or flipping, floating point only, and no exact guarantees.
  - Speed: 4.2M points in 0.570 s on an RTX 5090 (about 7.4 M pts/s, derived), and 15M clustered points in 2.53 s on an H200.
  - Incremental updates are explicitly future work.
  - — [arXiv 2605.06408](https://arxiv.org/html/2605.06408v1)
- **Parallelism in randomized incremental algorithms (Blelloch, Gu, Shun, Sun; SPAA 2016 / JACM 2020).** Gives the first incremental Delaunay algorithm with optimal work and polylogarithmic depth. — [ar5iv 1810.05303](https://ar5iv.arxiv.org/html/1810.05303)
- **Deterministic reservations (Blelloch, Fineman, Gibbons, Shun, PPoPP 2012).** Each point "writes its ID (with a write-min) to the endpoints of the triangles it affects". A point commits only if it still holds its ID on all of them. Shun's lecture reports 21x on 32 cores, with the 1-thread version 1.4x slower than serial. — [Shun MIT 6.886 lecture slides](https://jshun.csail.mit.edu/6886-s19/lectures/lecture10-2.pdf); [union-find paper describing the scheme, arXiv 2304.09331](https://arxiv.org/pdf/2304.09331)

### Inferences
- Several conflict schemes have been used. They share one property: a loser defers, and no undo log is needed if claims are made before any write.
  - Blandford: vertex test-locks with abort.
  - Batista: vertex or cell locks with "try another point".
  - Marot: ownership by spatial partition with abort.
  - gQM3D: atomicMin priority marks on tets in a two-step claim-then-commit.
  - Deterministic reservations: write-min, then commit.
- The gQM3D and deterministic-reservations pattern is the one that maps onto the user's existing vote-based flip locking.
- No published GPU method I found targets *incremental batch insertion into a persistent exact 3D DT* as its main use case, except refinement codes (gQM3D) and Radiant Foam, which publishes no insertion throughput.

### Gaps
- I found no measured GPU *insertion-only* rate (points inserted per second into an existing mesh) for gQM3D, gDP2d or Radiant Foam. Their reported times include refinement logic or training.
- I did not retrieve the original Batista et al. 2010 or Kohout et al. 2005 papers. Their numbers above come from the Nanjappa thesis's summaries.
- The "21x on 32 cores" figure from Shun's slides may refer to Delaunay refinement or triangulation in PBBS generally; the snippet did not say which.
- I did not search the CUDA cuDT (IIETA JESA) paper in depth. Its year and reliability are unknown.

## 3. Bowyer-Watson cavity insertion vs flip-based insertion

### Takeaway
- **Bowyer-Watson** costs about 46 inSphere tests, about 20 tets deleted and about 26 created per 3D insertion (Marot). It is exact by construction, so a single insertion never gets stuck. It needs a cavity-boundary adjacency rebuild, which takes about 1/3 of the time.
- **Flip insertion** (1-4 split, then 2-3/3-2 flips) touches the same final set of tets but through a sequence of local rewrites. It parallelizes naturally on the GPU (gDel3D) and reuses flip kernels. When many points are inserted at once it can get stuck in 3D and needs a fallback.
- **For a persistent mesh that already has an exact parallel flip engine,** flip insertion is the cheaper integration. Bowyer-Watson with priority-claimed cavities is the robust alternative or fallback.

### Cited Findings
- **Bowyer-Watson cost per 3D insertion:** about 20 cavity tets deleted, 26 created, about 46 inSphere evaluations (20 positive, 26 negative), and about 15 cavity-boundary vertices. — [Marot et al. 2019](https://arxiv.org/abs/1805.08831v3)
- **Time split (HXT, 5M points, 12.2 s of insertion):** WALK 1.0 s, CAVITY 6.2 s, DELAUNAYBALL (retriangulation plus adjacency) 4.5 s. In CGAL, DELAUNAYBALL is 15.3 s of 32.5 s. Marot attributes the speedup mainly to specializing DELAUNAYBALL for one interior vertex. — [Marot et al. 2019](https://arxiv.org/abs/1805.08831v3)
- **Flip-based insertion cost (3D, gDel3D):** flipping dominates total time on uniform, Gaussian and grid inputs. InsAll needs about 40% more flips than InsFlip, and random insertion order about 20% more flips than circumcenter order. — [Cao et al. 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **Getting stuck:** "in 3D when many points are inserted in parallel, flipping gets stuck long before reaching the DT." Guibas and Russel 2004 found that flipping seldom gets stuck when vertices of a DT move slightly. Inserting all points first and then flipping "usually leads to a stuck configuration". — [Cao et al. 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **GPU suitability:** "locking becomes very inefficient on the GPU, while conflicts during point insertions become uncontrollable", whereas flipping is "very amenable to parallel processing". — [Cao et al. 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **Bowyer-Watson on the GPU is feasible** with tuple lists, one thread per (cavity, tet, face). gQM3D does this for refinement. — [gQM3D 2019](https://www.comp.nus.edu.sg/~tants/gqm3d_files/gqm3d.pdf)
- **Memory traffic:** HXT dropped the cached sub-determinants in its parallel version because "the bandwidth usage with the parallel insertions is already near its maximum". gDel3D is memory-bound on a Titan. — [Marot et al. 2019](https://arxiv.org/abs/1805.08831v3); [Cao et al. 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)

### Inferences
- **Flip count per insertion (estimate).** In exact arithmetic the final star of the new point is the same either way: about 26 tets created on average. A flip sequence that removes about 20 cavity tets from an initial 1-4 split needs roughly (cavity size − 1)/1 net-removal steps. Each 2-3 flip adds a tet and each 3-2 removes one, so expect on the order of 10-20 flips per inserted point, each with its own insphere check. This is an estimate; no source gives flips per insertion in a Delaunay mesh. Total predicate work is probably comparable to Bowyer-Watson's about 46 inSphere calls, but spread over many more kernel iterations and global-memory rewrites.
- **Recommendation for 1% batches into a persistent DT.** The user already has exact integer predicates plus SoS, and parallel 2-3/3-2 flips with vote locking.
  - **Flip path:** (1) locate the containing tet, (2) 1-4 split (or 2-6 / 1-3 edge/face splits for degenerate locations under SoS, which SoS removes), (3) mark new facets and run the existing flip loop. This is the least new code.
  - **Fallback for stuck facets (rare):** either re-insert the affected point by Bowyer-Watson after removing it, or apply a local cavity re-triangulation: claim the cavity tets by priority, then retriangulate. This avoids a CPU star-splaying stage.
- **Alternatively, Bowyer-Watson insertion** with gQM3D-style priority claims gives exact results in one pass with no stuck states. Its cost is a new cavity-growth/retriangulation kernel and a variable-size cavity, which hurts GPU load balance.

### Gaps
- I found no published head-to-head of the GPU cost per insertion for Bowyer-Watson vs flip insertion on the same hardware and mesh.
- I found no published measurement of how often batched flip-insertion gets stuck when the batch is small (about 1%) and the base mesh is already Delaunay.

## 4. Batched insertion strategies: independent sets, colouring, locks and votes, BRIO and spatial sorting, point location

### Takeaway
There are three conflict strategies:
1. Spatial ownership with aborts (Marot).
2. Fine-grained locks with abort and retry (Blandford, Batista, PDT).
3. Priority claims followed by a commit phase (gQM3D atomicMin; deterministic reservations write-min).

Option 3 is the GPU-native one and matches vote-based flip locking. Ordering (BRIO plus Hilbert/Moore) keeps walks constant-length and cache-friendly. Point location uses either a walk from a hint (visibility walk) or, in from-scratch GPU builders, point-in-tet bookkeeping per tet with relocation (orient3d into children, or a history DAG).

### Cited Findings
- **BRIO plus space-filling curve:** with points sorted, "the number of walking steps is essentially constant". BRIO rounds also "reduce the average size of cavities". The visibility walk is guaranteed to terminate in a Delaunay triangulation. — [Marot et al. 2019](https://arxiv.org/abs/1805.08831v3)
- **Success rate per parallel round falls on retries:** "If 80 out of 100 vertices are successfully inserted … less than 16 of the 20 remaining" will be inserted on the next attempt, because leftover points sit in thin buffer zones. The fix is to halve the thread count when ρ < 1/5. — [Marot et al. 2019](https://arxiv.org/abs/1805.08831v3)
- **One insertion per tet per round (gDel3D)** is an implicit independent set for 1-4 splits: two points inside one tet can never be split at once. — [Cao et al. 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **GPU-CDT** inserts missing sites with CUDA atomics "to ensure that no two insertions can simultaneously modify one triangle". — [Nanjappa thesis](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf)
- **gQM3D two-step claims:** each candidate atomically writes its priority into every tet it wants, synchronizes, then checks ownership. Losers are filtered out and winners are "Delaunay independent". Priority favours lower-dimensional elements (subsegments, then subfaces, then tets). Reviving the cavity of a loser after the winner releases its tets was tried and "yet to find any good one which improves performance". — [gQM3D 2019](https://www.comp.nus.edu.sg/~tants/gqm3d_files/gqm3d.pdf)
- **Two Delaunay-dependent points inserted concurrently into cavities that only share a non-subface facet** produce a non-Delaunay facet: "the circumsphere of the new tetrahedron bcdp encloses q". The cavities must therefore be strictly disjoint, including their boundaries, or a flip repair must follow. — [gQM3D 2019, Fig. 6](https://www.comp.nus.edu.sg/~tants/gqm3d_files/gqm3d.pdf)
- **Blandford contention rule:** at most one thread per 2^14 mesh vertices early on. — [Blandford et al. 2006](https://www.cs.cmu.edu/~./blelloch/papers/BBK06.pdf)
- **Deterministic reservations** write-min a point's ID onto affected elements and commit if it holds all of them. The result is deterministic regardless of thread schedule. — [Shun lecture](https://jshun.csail.mit.edu/6886-s19/lectures/lecture10-2.pdf)
- **gDel3D point location:**
  - After a split, uninserted points are re-located into the 4 children with orient3d.
  - After flipping, a history DAG replays the flips (1-2 orient tests per DAG node).
  - Hilbert sorting means adjacent threads follow the same DAG path.
  - — [Cao et al. 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **PDT (Foteinos) location:** "jump and walk" to choose a start, then a visibility walk. — [Nanjappa thesis](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf)

### Inferences
- For a 1% batch, cavities rarely overlap in uniform refinement: about 20 tets per cavity against a mesh of about 6.5N tets with 0.01N new points gives about 3% of tets claimed. So one or two claim rounds should insert nearly all points (estimate). Clustered refinement, e.g. many children of one parent, will collide much more, as Marot observed for thin leftover regions.
- **Sibling points.** If refinement inserts several children per parent, siblings are mutually dependent. Two ways to handle this:
  - Order siblings into successive rounds, giving each parent at most one child per round. This is a colouring by sibling index.
  - Treat them as competing priorities.
- **Priority choice.** A fixed priority, e.g. a hash of point ID or Morton key, gives deterministic, reproducible meshes. This is the deterministic-reservations property.
- **Sorting the batch** by Morton/Hilbert key before insertion is cheap at 1% and improves memory coherence, as in gDel3D and HXT.

### Gaps
- I found no published conflict-rate measurements specifically for GPU batches of about 1% inserted into a large existing 3D DT.

## 5. Inserting a point inside a known cell (refinement near the parent generator): point location and fast paths

### Takeaway
No source I found describes a dedicated "parent-hint" fast path for GPU Delaunay insertion. Standard practice is to walk from a good hint: the last cavity (HXT), a jump-and-walk start (PDT), or tracked per-tet containment (gDel3D, BBK06). For a point inside its parent's Voronoi cell, the parent's star is an excellent hint (see Inferences). Location then costs a few orient3d tests, which is small next to the about 46 inSphere calls of the insertion itself.

### Cited Findings
- **HXT** starts each WALK from a tet in the previous point's Delaunay ball. With sorted points, walk length is essentially constant. WALK was only 1.0 s of 12.7 s for 5M points (about 8%). — [Marot et al. 2019](https://arxiv.org/abs/1805.08831v3)
- **BBK06** keeps "an association of every point p not yet inserted" with the tet that contains it, so no walk is needed. After each insertion, planeside tests redistribute the points held by deleted tets. — [Blandford et al. 2006](https://www.cs.cmu.edu/~./blelloch/papers/BBK06.pdf)
- **gDel3D** locates by containment bookkeeping plus a DAG replay; there is no walk. — [Cao et al. 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **gQM3D** seeds each cavity with "all tetrahedra intersecting p". The splitting point is computed from its element (the circumcenter of a bad tet, or the midpoint of a segment), so the element serves as the location hint. — [gQM3D 2019](https://www.comp.nus.edu.sg/~tants/gqm3d_files/gqm3d.pdf)

### Inferences
- **Key geometric fact (standard, not from the cited sources).** The nearest-neighbour graph is a subgraph of the Delaunay triangulation. If a new point x lies in the Voronoi cell of parent p, then p is x's nearest existing site, so after insertion xp is a Delaunay edge. Therefore at least one tet of p's current star has a circumsphere containing x, i.e. it is in x's Bowyer-Watson cavity.
- **Bowyer-Watson fast path:** test insphere(x) on p's star (about 25 tets in 3D, about 6 in 2D), seed the cavity from the hits, and grow. No walk is needed.
- **Flip/1-4 fast path:** the tet *containing* x need not be incident to p. A visibility walk started from the star tet that x "sees" (orient3d against p's link faces) reaches it in very few steps (estimate: 0-2 steps typical).
- **Using the existing Voronoi-cell bookkeeping.** The user's code already has p's Voronoi cell (from the dual). Choosing the tet whose circumcenter is a Voronoi vertex near x is another cheap seed.
- **Degenerate placements.** Placing children exactly on symmetric positions, e.g. on a face or edge of the parent's cell, creates exact co-spherical or co-planar configurations with integer coordinates. SoS resolves them. gDel3D's experience is that exact-predicate fallbacks then dominate run time on degenerate data and can expose bugs (DelaunayBench report). A small deterministic jitter in child placement could avoid systematic degeneracy, if the physics allows it.

### Gaps
- I found no published measurement of point-location cost per insertion on the GPU with parent hints. HXT's 8% WALK share (CPU, sorted) is the closest proxy.

## 6. Best published throughput numbers (points/s), with hardware and precision

### Takeaway
- **3D GPU exact DT construction:** best is gDel3D on an H200 or RTX 5090 (2026 measurement), about 4.7 M and 3.5 M points/s respectively (derived), double precision with exact predicates plus SoS. It collapses to about 0.1 M points/s on clustered data.
- **3D CPU:** HXT reaches about 10^7 points/s on a 64-core EPYC (2019, exact plus symbolic perturbation).
- **Inexact GPU Voronoi construction (clipping):** about 7.4 M points/s on an RTX 5090, with no exactness.
- **2D GPU:** only speedup ratios are published (gDel2D about 2x GPU-DT and more than 6x CGAL on a GTX 580). I found no 2D points/s number.
- I found no published *incremental insertion* throughput for any GPU code.

### Cited Findings
| System (year) | Hardware | Size / distribution | Time | Rate (derived) | Precision / exactness | Source |
|---|---|---|---|---|---|---|
| gDel3D (meas. 2026) | NVIDIA H200 | 4.2M real (bicycle) | 0.893 s | ≈4.7 M pts/s | double, exact + SoS | [Taveira 2026](https://arxiv.org/html/2605.06408v1) |
| gDel3D (meas. 2026) | RTX 5090 | 4.2M real | 1.204 s | ≈3.5 M pts/s | same | [Taveira 2026](https://arxiv.org/html/2605.06408v1) |
| gDel3D (meas. 2026) | H200 | 5M, 10 clusters | 45.4 s | ≈0.11 M pts/s | same | [Taveira 2026](https://arxiv.org/html/2605.06408v1) |
| gDel3D (2014) | GTX 580 | 3.61M Asian Dragon | 5.4 s | ≈0.67 M pts/s | same | [Cao 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf) |
| Clipping Voronoi (2026) | RTX 5090 | 4.2M real | 0.570 s | ≈7.4 M pts/s | float, not exact, no topology updates | [Taveira 2026](https://arxiv.org/html/2605.06408v1) |
| HXT parallel (2019) | AMD EPYC 7551, 64 cores | 3×10^9 tets, uniform | 53 s | "about 10^7 points per second", 65 M tets/s avg | double, Shewchuk + symbolic perturbation | [Marot 2019](https://arxiv.org/abs/1805.08831v3) |
| HXT parallel (2019) | Xeon Phi 7210 | 10^8 uniform | 28.95 s | ≈3.5 M pts/s | same | [Marot 2019](https://arxiv.org/abs/1805.08831v3) |
| HXT parallel (2019) | i7-6700HQ, 4 cores | 10^7 uniform | 7.40 s | ≈1.35 M pts/s | same | [Marot 2019](https://arxiv.org/abs/1805.08831v3) |
| HXT serial (2019) | i7-6700HQ, 1 core | 10^7 uniform | 21.66 s | ≈0.46 M pts/s | same | [Marot 2019](https://arxiv.org/abs/1805.08831v3) |
| CGAL 4.12 serial | i7-6700HQ | 10^7 uniform | 66.24 s | ≈0.15 M pts/s | exact (filtered) | [Marot 2019](https://arxiv.org/abs/1805.08831v3) |
| HXT (meas. 2026) | 16 cores EPYC 9534 | 4.2M real | 1.243 s | ≈3.4 M pts/s | exact | [Taveira 2026](https://arxiv.org/html/2605.06408v1) |
| Geogram (meas. 2026) | 16 cores EPYC 9534 | 4.2M real | 1.593 s | ≈2.6 M pts/s | exact | [Taveira 2026](https://arxiv.org/html/2605.06408v1) |
| BBK06 | 64 × EV67 1.15 GHz | 1.51×10^9 uniform | 5512 s | ≈0.27 M pts/s | – | [Blandford 2006](https://www.cs.cmu.edu/~./blelloch/papers/BBK06.pdf) |
| gQM3D refinement (2019) | GTX 1080 Ti | 4.29M output points | 2.5 min end-to-end | ≈29 k pts/s (includes refinement and initial CDT) | double, exact + SoS | [gQM3D 2019](https://www.comp.nus.edu.sg/~tants/gqm3d_files/gqm3d.pdf) |
| Radiant Foam Delaunay as full builder | RTX 5090 | 4.2M | 124.9 s | ≈34 k pts/s | – | [Taveira 2026](https://arxiv.org/html/2605.06408v1) |

- **2D:** gDel2D is up to 10x faster than Triangle, more than 6x faster than CGAL and 2x faster than GPU-DT (GTX 580, up to 10^7 points). GPU-CDT is 4.5x faster than CGAL on a GTX 580. GPU-DT is 1.53x faster than Triangle on an 8800 GTX. — [Cao 2014](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf); [Nanjappa thesis](https://www.comp.nus.edu.sg/~tants/gdel3d_files/AshwinNanjappaThesis.pdf)

### Inferences
- Implied time for a 1% batch (estimate, uniform data). A Bowyer-Watson insertion does about 46 inSpheres and creates about 26 tets. If GPU insertion into an existing mesh sustained even gDel3D's full-build rate of about 4 M pts/s, a 1% batch on a 10M-point mesh (10^5 points) would take about 25 ms.
- That rate is probably pessimistic for incremental use, because gDel3D's rate includes tracking all uninserted points and the DAG. It is optimistic if the batch needs several conflict rounds, each with a full kernel-launch sequence. At 10^5 points per batch, kernel-launch latency (about 5-10 µs per launch × tens of rounds) is negligible.
- The real risk is the long tail: rounds where only a few stuck or conflicting points remain, as in Marot's dropping success ratio and gQM3D's "long tail".
- In throughput terms, the about 4 M pts/s exact GPU build rate is roughly one-tenth of the user's flip-repair update throughput in M cells/s. This is unverified and depends on the user's current numbers. So insertion would likely not dominate a step that touches only 1% of cells.

### Gaps
- I found no published GPU throughput for *incremental* exact insertion into a persistent 2D or 3D mesh: not for Radiant Foam, gQM3D or anyone else.
- I found no 2D GPU points/s figures on modern hardware.
- I found no measurements of gDel3D in single precision, or with integer/fixed-point predicates, which is the user's representation.
