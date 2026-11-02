# Vertex removal (deletion) from Delaunay triangulations: 2D/3D, sequential/parallel/GPU, exactness, suitability for derefining a moving Voronoi mesh

Scope: deleting generating points (about 1% per step) from a persistent GPU tetrahedral Delaunay mesh. That mesh is repaired by parallel 2-3/3-2 flips with vote-based locking, using exact integer predicates with symbolic perturbation (SoS).

Source access: some PDFs could not be fetched. Ledoux/Gold/Baciu 2005 (TU Delft server refused the connection; ResearchGate returned HTML), Devillers–Teillaud 2003/2011 (font-encoded, unreadable), HAL pages (bot wall). Statements about those papers that come only from abstracts or citing papers are marked as such. Throughout, "[inference]" marks my own reasoning or arithmetic.

## 1. 2D removal: Devillers ear queue, Chew, flips, CGAL 2D

### Takeaway
2D removal is a solved problem. For realistic degrees (mean 6), the winner is not the asymptotically optimal algorithm but hard-wired small-degree code. CGAL 3.9 removes a 2D vertex in about 1.7–3.2 µs (100K–100M points), i.e. roughly 0.3–0.6 M deletions/s on one core. The cost rises with triangulation size because of cache misses, not algorithmic work.

### Cited Findings
- **Devillers, "On deletion in Delaunay triangulations"** (SoCG 1999; IJCGA 12:193–205, 2002).
  - Removes p by filling the star-shaped hole with "ears" q_i q_{i+1} q_{i+2}, prioritised by |power of p| with respect to each ear's circumcircle (space-of-spheres / lifting duality). The ear of minimal |power| is Delaunay.
  - Cost is O(k log k) for degree k; "time consuming computations are only done a linear number of times." The basic version needs 3k−8 power computations; the optimised version 2k−4 power computations plus k−4 updates.
  - One power evaluation is 14 additions, 15 multiplications and 1 division.
  - Gives an Ω(k log k) lower bound for any ear-queue method (it is equivalent to sorting).
  - [Source](https://ar5iv.labs.arxiv.org/html/cs/9907023) / [arXiv cs/9907023](https://arxiv.org/abs/cs/9907023)
- **Comparison counts from the same paper:**
  - Flip-based removal (flip the degree down to 3, then remove and re-flip) needs at worst (k−2)(k−3)/2 incircle tests. The naive queue-based flip variant needs more.
  - Edge completion (gift wrapping) needs k−3 tests per triangle and has the same worst case.
  - Chew's randomized algorithm needs an expected 5 incircle tests per reinserted point, about 5k + O(1).
  - Aggarwal–Guibas–Saxe–Shor (1989) and Chew achieve O(k) but are "a little bit too intricate" with big constants.
  - For k < 9, flipping or edge completion uses fewer incircle tests. The best hybrid uses ears for degree ≥ d_limit ≈ 9 and flips below.
  - [Source](https://ar5iv.labs.arxiv.org/html/cs/9907023)
- **Devillers 1999 experiment:**
  - 2,000,000 uniform random points in a square; the initial DT took 104 s on a Sun Ultra10 at 300 MHz. All points were then removed in random order. Timings appear only in Fig. 4, whose axis tops at 40 s.
  - The implementation uses 24-bit integer coordinates, static and semi-static filters, and exact fallback. It notes that "Degeneracies can be solved using perturbation techniques."
  - With floating point, the method terminates but may produce a non-Delaunay or even non-planar result.
  - [Source](https://arxiv.org/pdf/cs/9907023)
- **Heller's (and Midtbø's) "smallest circumradius ear is Delaunay" rule is false.** The paper gives a counterexample. [Source](https://ar5iv.labs.arxiv.org/html/cs/9907023)
- **Devillers, "Vertex removal in two-dimensional Delaunay triangulation: speed-up by low degrees optimization"** (CGTA 44:169–177, 2011; INRIA RR 7104, 2009, "Asymptotic complexity is pointless").
  - Hand-designed decision-tree code for degrees 3 ≤ d ≤ 7 improves global removal time for random points by a factor of about 2.
  - [Source (HAL record)](https://hal-univ-tlse3.archives-ouvertes.fr/INRIA/inria-00433107); citation in [Devillers EuroCG 2012](https://homepages.loria.fr/ODevillers/EuroCG2012/EuroCG12-devillers.pdf)
- **CGAL `Delaunay_triangulation_2::remove`** removes the vertex and re-triangulates the hole Delaunay-wise. Removing a vertex of degree d costs O(d²), and d = O(1) for a random vertex. [CGAL doc](https://doc.cgal.org/5.6/Triangulation_2/classCGAL_1_1Delaunay__triangulation__2.html)
- **Measured CGAL 3.9 numbers**: 2.3 GHz, 16 GB, Epick kernel; "deletion" removes all points in random order; µs per point.

  | Dimension | Points | Static insert (µs) | Dynamic insert (µs) | Delete (µs) |
  |---|---|---|---|---|
  | 2D | 100K | 0.90 | 2.8 | 1.7 |
  | 2D | 1M | 0.92 | 5.8 | 2.5 |
  | 2D | 10M | 1.06 | 9.0 | 3.0 |
  | 2D | 100M | 1.15 | 13 | 3.2 |
  | 3D | 100K | 7.8 | 18 | 102 |
  | 3D | 1M | 8 | 25 | 106 |
  | 3D | 10M | 8.2 | 33 | 109 |

  Devillers: "Although theory claims that the deletion time does not depend on the triangulation size, we observe the contrary, especially in two dimensions... due the load of the cache memory." [Source](https://homepages.loria.fr/ODevillers/EuroCG2012/EuroCG12-devillers.pdf)

### Inferences
- [inference] 2D CGAL deletion throughput is about 0.59 M/s at 100K points and 0.31 M/s at 100M points, single core. Deletion costs about 2–3x a spatially sorted insertion.
- [inference] For a flip-based GPU code, the 2D analogue (flip the degree down to 3, then a 3→1 removal) is the natural fit. Mean degree is 6, so about 3 flips plus re-Delaunay flips per removal, which is far below the k ≥ 9 crossover where ear queues pay off.

### Gaps
- Fig. 4 of Devillers 1999 and the speed-up tables of Devillers 2011 were not readable numerically.
- No published GPU 2D Delaunay removal throughput was found.

## 2. 3D removal: Devillers–Teillaud perturbation, CGAL 3D, flip-based removal (Ledoux et al.), ESA 2013, Shewchuk

### Takeaway
In 3D, the practical standard is "triangulate and sew" (CGAL). Build the Delaunay triangulation of the link vertices P, take its tetrahedra in conflict with q, and glue them into the cavity. Degenerate (cospherical) stars are handled by a symbolic perturbation that must be the same one that defines the outer triangulation.

Measured cost in CGAL is about 100 µs per removal (about 0.01 M removals/s per core). That is 12–13x the cost of a sorted insertion. Pure flip-based removal (Lawson flips in 3D) can get stuck, so robust flip removal needs ear ordering and perturbation (Ledoux et al. 2005).

### Cited Findings
- **The star-shaped cavity always has a Delaunay tetrahedralization.** The hole left by removing a Delaunay vertex can always be tetrahedralized, as part of DT(S∖{q}). This is not true for constrained DTs. [Schaller & Meyer-Hermann 2003, citing Shewchuk](https://ar5iv.arxiv.org/html/physics/0302018)
- **Devillers & Teillaud, "Perturbations and vertex removal in a 3D Delaunay triangulation"** (SODA 2003, pp. 313–319; INRIA RR 4624, 2002; extended as RR 5968, 2006, "...in Delaunay and regular 3D triangulations"; journal version "Perturbations for Delaunay and weighted Delaunay 3D triangulations," CGTA 44:160–168, 2011).
  - States that robust 3D removal "remains a practical problem" and proposes a simple method that removes any vertex even in highly degenerate point sets.
  - This method made CGAL removal robust (CGAL 2.3/2.4).
  - [Teillaud bibliography](https://members.loria.fr/MTeillaud/biblio/Author/DEVILLERS-O.html); [record](https://science.ulysseus.eu/records/9j2np-k2m82)
- **The core difficulty (from Devillers–Teillaud via Ledoux et al.):** not every polyhedron can be tetrahedralized. When the link vertices are cospherical, DT(P) is not unique. An arbitrary choice inside the hole may not match the facets on the hole boundary, which are already fixed by the outer triangulation.
  - Devillers–Teillaud resolve this with a symbolic perturbation of the in-sphere predicate that is consistent between the outside and the hole.
  - Ledoux et al. note that the Devillers–Teillaud paper is largely devoted to the perturbation machinery.
  - [Search summary of Ledoux et al. 2005](https://link.springer.com/chapter/10.1007/11424758_76)
  - The exact form of the perturbation (a lexicographic or index order on points that perturbs the lifted coordinate) is recalled from memory and was not verified from the unreadable PDF.
- **CGAL Triangulation_3:**
  - Delaunay is "fully dynamic" (insertions, vertex removals, displacements).
  - Removal first appeared in CGAL 2.2 and "became really robust only in CGAL 2.3" after the degenerate-case research. The faster algorithm was proposed by Mariette Yvinec and implemented by Andreas Fabri.
  - Regular-triangulation removal (Christophe Delage, CGAL 3.2, 2005) uses "the symbolic perturbation proposed in [Devillers–Teillaud]."
  - [CGAL Triangulation_3 manual](https://doc.cgal.org/latest/Triangulation_3/index.html)
- **CGAL 3D removal timing** (CGAL 3.6, 3 GHz 64-bit Xeon, GCC 4.3.2 -O3, random points in a unit cube; average per removal):
  - Delaunay 1.0e-4 s
  - Delaunay with fast location 1.03e-4 s
  - Regular 1.42e-4 s
  - Regular with no hidden points 1.38e-4 s
  - For comparison, construction of 10^6 points took 8.5 s.
  - [CGAL manual, Fig. 48.7](https://doc.cgal.org/latest/Triangulation_3/index.html)
- **CGAL 3.9 (2012):** 3D deletion of all points in random order costs 102 / 106 / 109 µs per point at 100K / 1M / 10M points, against 7.8–8.2 µs for static insert and 18–33 µs for dynamic insert. [Devillers EuroCG 2012](https://homepages.loria.fr/ODevillers/EuroCG2012/EuroCG12-devillers.pdf)
- **CGAL's 3D approach:** "compute from scratch the (small) Delaunay triangulation of the neighbors of the removed point, and plug the relevant simplices inside the hole." [Devillers EuroCG 2012](https://homepages.loria.fr/ODevillers/EuroCG2012/EuroCG12-devillers.pdf)
  - CGAL falls back to a lower-dimensional DT whenever general position is violated. [Buchin et al. ESA 2013](https://page.mi.fu-berlin.de/mulzer/pubs/dtDel3DESA.pdf)
- **Expected work per removal:** for a 3D Poisson point set, the expected number of tetrahedra deleted is 96π²/35 ≈ 27. A good algorithm must therefore be fast on small stars. [Devillers 1999](https://arxiv.org/pdf/cs/9907023)
- **Buchin, Devillers, Mulzer, Schrijvers, Shewchuk, "Vertex deletion for 3D Delaunay triangulations"** (ESA 2013; [PDF](https://page.mi.fu-berlin.de/mulzer/pubs/dtDel3DESA.pdf)).
  - Survey of 3D methods:
    - Gift wrapping is Θ(d³) in 3D.
    - The ear queue is O(k log d), where k is the number of new tetrahedra, but needs a new "compare powers of two ears" predicate. This makes it "less generic and more difficult to make robust."
    - The flip algorithm "does not always work; it can get stuck in a non-Delaunay triangulation from which no flip can make further progress."
    - Triangulate-and-sew runs in expected O(d log d + C(P)).
  - New algorithm: expected O(C⊗(P)) time, where C⊗(P) is the expected number of tetrahedra in conflict with q created during the randomized incremental construction of DT(P). It builds only the part of DT(P) that conflicts with q, uses guided point location, and needs only standard predicates.
  - Experiments (CGAL 4.2, 2.53 GHz i5, MSVC):
    - "Similar running times for all methods" on bounded-degree data (random distributions, real 3D models).
    - Significantly faster than CGAL for high degree (moment curve and helix, degrees up to 250, timings in ms).
    - The prototype failed on stability "a few dozen times" out of 182,051 deletions. It assumed general position and the failures were discarded.
- **Ledoux, Gold, Baciu, "Flipping to robustly delete a vertex in a Delaunay tetrahedralization"** (ICCSA 2005, LNCS 3480).
  - Deletes v by a sequence of bistellar flips that progressively shrink star(v) without opening a hole. Some tetrahedra outside star(v) are modified. The final step is a flip41 (the inverse of a 1→4 insertion).
  - Proposes "two mechanisms for ensuring robustness." Degenerate flat tetrahedra (4 coplanar vertices) can be removed immediately by a flip32.
  - [Springer](https://link.springer.com/chapter/10.1007/11424758_76); [PDF (unreachable from here)](https://3d.bk.tudelft.nl/hledoux/pdfs/05_iccsa.pdf)
  - Not verified (from memory of the paper and Ledoux's 2006 thesis): ears of the star's boundary are processed in power order, à la Devillers, using flip23/flip32/flip44, with symbolic perturbation for cospherical ears.
- **Shewchuk, "Star splaying: an algorithm for repairing Delaunay triangulations and convex hulls"** (SoCG 2005).
  - Works in any dimension and runs in linear time if the input is nearly Delaunay and sparse.
  - "In three or more dimensions ... hill-climbing [flipping] can get stuck in a local optimum that is not Delaunay. Three-dimensional flipping gets stuck easily in practice."
  - Recommends: "A Delaunay repair algorithm can start with flipping, then fall back on star splaying or star flipping to complete the repair when flipping gets stuck."
  - Cites Guibas & Russel: for small vertex movements, flipping gets stuck "only occasionally," and repair is up to 3x faster than rebuilding.
  - Suggests star splaying for parallel and out-of-core meshing.
  - [PDF](https://people.eecs.berkeley.edu/~jrs/papers/star.pdf)

### Inferences
- [inference] SoS is the key enabler for our code. With exact integer predicates and SoS keyed on a fixed per-point index (not insertion time), the perturbed DT is a unique function of the point set. DT(S∖{q}) restricted to the cavity then equals the conflict part of DT(P) under the same perturbation. Triangulate-and-sew (or Bowyer–Watson reinsertion of the link points) is therefore consistent with the outer boundary by construction, which is exactly the property Devillers–Teillaud had to engineer.
  - The subtlety: the perturbation must be applied identically in the global mesh and in the local rebuild. That means the same index order and the same lifted-coordinate convention, and no geometric "infinite vertex" tricks that differ between the two.
  - If the predicate perturbs by insertion order, removal breaks that consistency.
- [inference] Removal by flips is also sound under SoS: in a perturbed (generic) configuration there are no flat/flip44 cases. It can still get stuck in principle, and no polynomial bound on the number of flips needed is known. So a "local rebuild" fallback (what our host fallback does now) remains necessary.
- [inference] CGAL 3D removal (~100 µs, ~0.01 M/s/core) is about 12x slower than insertion. For 1% removal of N = 10^7 points (10^5 removals) this is about 10 s single core on CPU. A GPU path is therefore justified if removal sits on the critical path every step.

### Gaps
- No published per-removal µs for flip-based 3D removal (Ledoux et al.) against CGAL; the PDF was not accessible.
- CGAL's parallel removal speed-ups (Fig. 48.8, 100K of 1M points removed) exist only as an image.
- The exact Devillers–Teillaud perturbation formula was not verified.

## 3. Kinetic/dynamic Delaunay maintenance and removal

### Takeaway
Kinetic DT codes maintain the mesh by 2-3/3-2 flips as points move, which is the same machinery as our repair. Removal there is done either by a local cavity re-triangulation or by "moving" the vertex onto a neighbour with flips and then merging. Removal is cheaper than insertion because no point location is needed.

### Cited Findings
- **Schaller & Meyer-Hermann, "Kinetic and dynamic Delaunay tetrahedralizations in three dimensions"** (Comput. Phys. Commun., 2004; arXiv physics/0302018).
  - Deletes vertex A by repeatedly moving it toward its nearest neighbour A′, with flips after each step (step size limited by a safety criterion). It then deletes the simplices containing both A and A′, relabels A to A′, and flips.
  - Deletion is "more efficient than insertion" (no location cost) and scales linearly.
  - Kinetic restoration is about 20x faster than re-triangulation for small displacements.
  - Mixed insert/delete step: 0.26 s at 20,000 points and 3.21 s at 200,000 points, with 426 and 9,766 flips. Hardware: 1.533 GHz Athlon, double precision.
  - Rounding errors in skinny tetrahedra are flagged and exact arithmetic is recommended; "in our experiences ... such situations actually never occurred."
  - [Source](https://ar5iv.arxiv.org/html/physics/0302018)
- **Guibas & Russel (via Shewchuk 2005):** flipping repair after small motions is up to 3x faster than rebuilding and gets stuck only occasionally. [Source](https://people.eecs.berkeley.edu/~jrs/papers/star.pdf)
- **AREPO (Springel 2010):** derefinement dissolves a cell by deleting its generating point so that neighbours absorb its volume. To avoid interacting decisions, candidates are processed in priority order and cells adjacent to an already-flagged cell are skipped, i.e. an independent set per step. [arXiv 0901.4107](https://arxiv.org/pdf/0901.4107)

### Inferences
- [inference] The "move toward neighbour, then merge" removal reuses an existing kinetic flip engine. On our GPU it would be: displace q toward a link neighbour, run the same vote-locked 2-3/3-2 flip repair, then collapse the edge q–q′. This is attractive because it needs no new kernels. However, a degenerate collapse (the final step) still requires a topological edge-collapse kernel and exact handling of zero-volume tetrahedra.

### Gaps
- No per-deletion µs numbers in Schaller & Meyer-Hermann (figure only).
- The AREPO public paper gives no measured derefinement cost.

## 4. Parallel and GPU removal

### Takeaway
I found no published GPU implementation of exact Delaunay vertex deletion in 3D. Parallel removal exists only on multicore CPUs: CGAL with TBB uses per-vertex locks with retry, and the same machinery serves insertion. GPU tetrahedral coarsening work (edge collapse, HPG 2023) uses conflict-free independent sub-meshes, not Delaunay deletion. The standard parallel pattern is: select an independent set of removable vertices whose stars do not overlap, process them concurrently, and retry or defer conflicts.

### Cited Findings
- **CGAL 3D parallel mode** (`Parallel_tag` plus a lock data structure, TBB): "the insertion/removal of a range of points will be performed in parallel."
  - Locking is per vertex: "Locking a facet (resp. a cell) boils down to locking all its 3 (resp. 4) incident vertices."
  - The parallel mode requires the compact location policy.
  - Speed-ups for removing 100K of 1M points are shown in Fig. 48.8 (values not given in the text).
  - The regular-triangulation example uses `rtr.remove(begin,end)` in parallel.
  - [CGAL manual](https://doc.cgal.org/latest/Triangulation_3/index.html)
- **Ströter, Stork, Fellner, "Massively parallel adaptive collapsing of edges for unstructured tetrahedral meshes"** (HPG 2023). GPU edge-collapse coarsening; finds a compact set of conflict-free sub-meshes and needs up to 59% fewer parallel collapsing iterations than the prior state of the art. It is not Delaunay-preserving and not exact. [EG digital library](https://diglib.eg.org/handle/10.2312/hpg20231139)
- **Independent sets for parallel decimation:** deleting an independent set of vertices (no two joined by an edge) "can be run efficiently in parallel" (surface meshes, 2000). [Source](https://emis.univie.ac.at/journals/AMUC/_contributed/algo2000/francskala.pdf)
- **Parallel Delaunay refinement** (Spielman, Teng, Üngör; arXiv cs/0207063) uses the same independent-set principle for insertion: each independent set can be inserted sequentially without loss of guarantees. [Source](https://ar5iv.labs.arxiv.org/html/cs/0207063)
- **Parallel tetrahedral coarsening** (de Cougny & Shephard 1999) uses edge collapse with repartitioning. [Source](https://scorec.rpi.edu/reports/view_report.php?id=60)
- **GPU 3D Delaunay construction** (gDel3D: Cao, Nanjappa, Gao, Tan) uses parallel insertion plus parallel flipping, with exact predicates and SoS. It does no removal. [gDel3D PDF](https://www.comp.nus.edu.sg/~tants/gdel3d_files/gDel3D.pdf)
- **A GPU linear conforming DT** (OSTI 1120828) treats vertex add/remove as implicit edge flips in a second pass. It is 2D and conforming, not a general 3D removal. [Source](https://www.osti.gov/servlets/purl/1120828)

### Inferences
- [inference] A GPU removal design consistent with our mesh:
  1. Select the removal set (1% of N) and reduce it to an independent set in which no two removed vertices share a tetrahedron. Being stricter (star-disjoint plus a one-ring buffer, as AREPO does) avoids races in the re-triangulated cavities. At 1% density in a mesh where each vertex touches about 27 tetrahedra, conflicts are rare and one to two rounds should suffice.
  2. Per vertex, run one of two options. (a) Rebuild: gather the link P (|P| ≈ 15 on average for Poisson; expected 27 tetrahedra in the star) in shared memory, build the conflict part of DT(P) by Bowyer–Watson with the same exact SoS predicate, and sew. (b) Flip down: drive star(q) to 4 tetrahedra using 2-3/3-2 flips ordered by an ear priority, then flip41.
  3. Run the global flip repair (not strictly needed if step 2 is exact, but harmless).
- [inference] The rebuild (triangulate-and-sew) is the more robust choice. It needs only orient3d and insphere with SoS, never gets stuck, and its cost is bounded by C(P). Flip-down reuses kernels but can stall in 3D, so it needs the rebuild as a fallback anyway (Shewchuk's "flip first, then fall back").
- [inference] Throughput estimate (not measured). Rebuild work is about d log d + C(P), roughly a few hundred insphere tests per removal. At an exact-filtered rate of 10^9–10^10 predicates/s on a modern GPU, that suggests 10^6–10^7 removals/s, i.e. 100–1000x a single CGAL core. This is an order-of-magnitude estimate only.

### Gaps
- I found no published GPU Delaunay vertex deletion, nor any measured GPU removal throughput in 2D or 3D, up to Nov 2026. Several searches (2024–2026) returned only off-topic GPU work.
- No numbers were obtained for Chernikov/Chrisochoides parallel coarsening, Foteinos, Marot, or fTetWild coarsening (searches did not surface removal-specific results).
- Ströter et al.'s throughput numbers (403 on fetch) are not recorded.

## 5. Removal vs local rebuild by Bowyer–Watson insertion of the star vertices

### Takeaway
"Rebuild the star by inserting the link vertices, then glue" is what CGAL does: triangulate-and-sew, expected O(d log d + C(P)). The ESA 2013 refinement builds only the q-conflict part, expected O(C⊗(P)), and is competitive at low degree and much faster at high degree. Under consistent SoS it is the most robust option. Its costs are a small local DT build plus gluing, and it needs only standard predicates.

### Cited Findings
- Triangulate-and-sew: triangulate P with randomized incremental construction, take the tetrahedra adjoining q in DT(P), and sew them into the cavity. Expected O(d log d + C(P)), where Ω(d) ≤ k ≤ C(P) ≤ O(d²). [Buchin et al. 2013](https://page.mi.fu-berlin.de/mulzer/pubs/dtDel3DESA.pdf)
- The ESA 2013 variant (DT⊗q only) avoids building the non-conflicting part of DT(P). The authors identify this as "the source of a big part of the improvement," while guided point location matters less. [Buchin et al. 2013](https://page.mi.fu-berlin.de/mulzer/pubs/dtDel3DESA.pdf)
- The ear queue needs a non-standard "compare powers" predicate, which is harder to make exact. Triangulate-and-sew "uses less complicated numerical predicates ... easier to implement robustly." [Buchin et al. 2013](https://page.mi.fu-berlin.de/mulzer/pubs/dtDel3DESA.pdf)
- On bounded-degree inputs, all methods (CGAL, randomized CGAL, and the new variants) had similar running times. [Buchin et al. 2013](https://page.mi.fu-berlin.de/mulzer/pubs/dtDel3DESA.pdf)

### Inferences
- [inference] For Voronoi-mesh derefinement (bounded degree, quasi-uniform), the host fallback's local Bowyer–Watson rebuild is algorithmically close to optimal and should be ported to the GPU rather than replaced.
  - Gain: keep only tetrahedra in conflict with q (the ESA "DT⊗q-only" trick), which cuts work and memory.
  - Requirement: the SoS order must be identical to the global mesh's so the sewn facets match.
- [inference] Since the ear queue requires a power-comparison predicate (a degree-higher determinant), it is a poor fit for exact integer predicates.

### Gaps
- No direct published timing comparing flip-based removal, the ear queue and rebuild on the same 3D mesh with exact SoS predicates.

## 6. Best published throughput numbers

### Takeaway
Best measured: 2D about 0.3–0.6 M deletions/s and 3D about 0.01 M deletions/s, single-core CGAL. No GPU or parallel removal throughput numbers are published in text form.

### Cited Findings
- **2D, CGAL 3.9, 2.3 GHz, one core:**
  - 1.7 µs/deletion at 100K points (about 0.59 M/s).
  - 3.2 µs/deletion at 100M points (about 0.31 M/s).
  - [Devillers EuroCG 2012](https://homepages.loria.fr/ODevillers/EuroCG2012/EuroCG12-devillers.pdf)
- **3D, CGAL 3.9:**
  - 102 µs at 100K points (about 0.0098 M/s).
  - 109 µs at 10M points (about 0.0092 M/s).
  - [Devillers EuroCG 2012](https://homepages.loria.fr/ODevillers/EuroCG2012/EuroCG12-devillers.pdf)
- **3D, CGAL 3.6, 3 GHz Xeon:** 100 µs/removal (Delaunay), 142 µs (regular). [CGAL manual](https://doc.cgal.org/latest/Triangulation_3/index.html)
- **Low-degree-optimised 2D removal** gives about a 2x speed-up over generic code (Devillers 2011). [HAL](https://hal-univ-tlse3.archives-ouvertes.fr/INRIA/inria-00433107)
- **Multicore CGAL removal:** speed-ups exist only in a figure. [CGAL manual](https://doc.cgal.org/latest/Triangulation_3/index.html)

### Inferences
- [inference] Rates in this section are 1/µs conversions of the cited figures.
- [inference] With 3D removal at about 100 µs against about 8 µs for insertion, a moving-mesh code removing 1% of points per step spends about as much CPU time on removal as on inserting about 12% of points. This is why moving removal to the GPU (or batching it with the flip repair) matters.

### Gaps
- No GPU removal benchmarks were found anywhere (2D or 3D, any precision).
- CGAL timings are older hardware (2009–2012); modern single-core numbers are probably 2–3x faster [inference, unmeasured].
