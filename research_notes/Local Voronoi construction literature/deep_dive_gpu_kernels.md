# Deep dive: fastest published local (per-generator) 3D Voronoi cell construction on GPUs (as of Oct 2026)

How I checked things: I read the full arXiv text of Taveira et al. 2026, including the appendices. I cloned and read the source of paragram (Taveira), GPU-Restricted-Power-Diagrams (Basselin 2021, OpenCL), 3D-Voronoi-GPU (Liu et al. 2020, which is a direct fork of Ray et al. 2018's CUDA code and credits Ray et al. for it), votess (SYCL) and peclet-voro (Kokkos). I also read the arXiv text of votess/JOSS and of Xiao, Cao & Chen (arXiv 2509.07175).
Some sources could not be read:
- The full text of Ray et al. 2018: HAL is behind Anubis bot protection, and ACM DL and ResearchGate returned 403.
- The full text of Basselin et al. 2021: the EG diglib PDF connection was refused, and HAL was blocked.
For these two papers, numbers come from their code, the abstract/caption snippets, a slide deck, and third-party benchmarks. Anything not checked against the primary text is marked UNVERIFIED.

Code paths below are inside the cloned repositories:
- paragram: https://github.com/zenseact/paragram, files under `paragram/csrc/laguerre/`
- Ray-derived code: https://github.com/xh-liu-tech/3D-Voronoi-GPU
- Basselin: https://github.com/basselin7u/GPU-Restricted-Power-Diagrams
- votess: https://github.com/samridh-dev/votess
- peclet-voro: https://github.com/computational-chemical-engineering/peclet-voro

---

## Q1. Ray, Sokolov, Lefebvre, Lévy 2018, "Meshless Voronoi on the GPU" (ACM TOG 37(6), art. 265): how does it work at kernel level?

### Takeaway
The design is one thread per cell, with the cell stored in shared memory as up to 64 float4 planes plus up to 96 dual triangles (uchar3 plane-index triples), about 1.4 KB per thread. The block size is only 16 threads. Neighbours come from a fixed-K kNN (K = 90 for white noise, 35 for blue noise) computed beforehand on a uniform grid with about 3.1 points per bin. After every clip, a "2R" security-radius test recomputes all vertex positions. Overflow or an insufficient K is only flagged in a status code. The method is fast on blue noise and slow on white noise: in a third-party benchmark on an RTX 3080 (fp32), 5M points took 0.34 s for blue noise and 3.12 s for white noise.

### Cited Findings
- The paper exists: Ray, Sokolov, Lefebvre, Lévy, "Meshless voronoi on the GPU", ACM TOG 37(6), Article 265, Dec 2018, doi 10.1145/3272127.3275092. The code was published as supplemental material — [Taveira et al. reference list](https://arxiv.org/pdf/2605.06408); [HAL record hal-01927559](https://hal.archives-ouvertes.fr/hal-01927559); [search summary of HAL/ACM](https://history.siggraph.org/?p=198036)
- Liu et al. (TVCG 2020) state in their README: "We thank Ray et al. for sharing their code". Their `voronoi.cu`/`knearests.cu` is the Ray CUDA kernel extended with tet clipping, and it is used below as a proxy for Ray's code — [3D-Voronoi-GPU README](https://github.com/xh-liu-tech/3D-Voronoi-GPU)
- **Parameters** (`params.h`). The comments tie each preset to a point distribution:
  - Preset 0, "conservative settings (white noise)": `VORO_BLOCK_SIZE 16`, `KNN_BLOCK_SIZE 32`, `_K_ 90`, `_MAX_P_ 64`, `_MAX_T_ 96`.
  - Preset 1, "perturbed grid": K=90, MAX_P=50.
  - Preset 2, "blue noise": K=35, MAX_P=32, KNN block 64.
  — [params.h](https://github.com/xh-liu-tech/3D-Voronoi-GPU/blob/master/params.h)
- **Per-cell data structure** (`voronoi.h`): it lives entirely in shared memory.
  - `__shared__ uchar3 tr_data[VORO_BLOCK_SIZE*_MAX_T_]` holds the dual triangles. Each Voronoi vertex is three plane indices.
  - `__shared__ uchar boundary_next_data[VORO_BLOCK_SIZE*_MAX_P_]` and `__shared__ float4 clip_data[VORO_BLOCK_SIZE*_MAX_P_]` hold the planes.
  - Indexing is `threadIdx.x*_MAX_T_ + t`, i.e. contiguous per thread (AoS per thread, not interleaved across lanes).
  - With preset 0 that is 96·3 + 64 + 64·16 = 1,376 B per cell. Vertex positions are not stored.
  — [voronoi.h](https://github.com/xh-liu-tech/3D-Voronoi-GPU/blob/master/voronoi.h)
- **Clipping** (`clip_by_plane`):
  1. Every triangle is tested with `triangle_is_in_conflict(tr(i), eqn)`, which evaluates a float 4×4 determinant of the three vertex planes plus the new plane. Positions are never cached.
  2. Conflicting triangles are swapped to the end.
  3. If none conflict, the plane is popped.
  4. Otherwise `compute_boundary()` walks the cavity boundary using `boundary_next` per plane.
  5. The hole is fanned with new triangles `(cur_v, cir, boundary_next(cir))`.
  — [voronoi.cu](https://github.com/xh-liu-tech/3D-Voronoi-GPU/blob/master/voronoi.cu)
- **Termination**: `is_security_radius_reached(last_neig)` loops over all `nb_t` triangles, recomputes each vertex with three 3×3 determinants and a divide, takes the max squared distance `v_dist`, and returns `d2 > 4*v_dist`. In other words, stop when the next kNN neighbour is farther than twice the farthest vertex. It is called after every clip. If the K-th neighbour is reached without passing the test, the status is set to `security_radius_not_reached` — [voronoi.cu](https://github.com/xh-liu-tech/3D-Voronoi-GPU/blob/master/voronoi.cu)
- **Precision**: float by default. There is an optional `USE_ARITHMETIC_FILTER` (with a constant `6.6876506e-05` produced by a predicate generator). It sets `needs_exact_predicates` when `|det| < eps`, and there is a `triangle_is_in_conflict_double` variant with eps `1.2466136531027298e-13`. The status codes are `triangle_overflow, vertex_overflow, inconsistent_boundary, security_radius_not_reached, success, needs_exact_predicates, no_intersection` — [params.h](https://github.com/xh-liu-tech/3D-Voronoi-GPU/blob/master/params.h), [voronoi_defs.h](https://github.com/xh-liu-tech/3D-Voronoi-GPU/blob/master/voronoi_defs.h)
- **kNN** (`knearests.cu`):
  - Uniform grid with side `sz = round((N/3.1)^(1/3))`, i.e. about 3.1 points per bin. It refuses to run when sz < 16. Points are assumed to lie in [0,1000]^3.
  - Points are counting-sorted into bin order (`stored_points`, `permutation`), so threads process points in grid order.
  - Each thread keeps a K-entry max-heap (keys plus distances) in shared memory (`__shared__ knearests[_K_*KNN_BLOCK_SIZE]`).
  - It visits precomputed shell offsets (rings up to Nmax=16) ordered by their lower-bound distance and stops when the heap max is below the next ring's minimum distance. The final heap is heapsorted, so neighbours are sorted by distance.
  — [knearests.cu](https://github.com/xh-liu-tech/3D-Voronoi-GPU/blob/master/knearests.cu)
- **Third-party throughput** (Xiao, Cao, Chen, arXiv 2509.07175v2, Fig. 1). The workload is 5M Voronoi cells bounded by a cube on an NVIDIA RTX 3080, fp32, including centroid computation and excluding allocation and transfer.
  - Ray et al.: 3.12 s for white noise (about 1.6 M cells/s) and 0.34 s for blue noise (about 14.7 M cells/s).
  - Basselin et al.: 2.81 s and 0.42 s.
  - The same paper notes that Ray's code "requires GPU memory beyond the test machine's capacity when the number of sites exceeds 7M" for white noise.
  — [Xiao et al. 2025/26](https://arxiv.org/pdf/2509.07175)
- Basselin's slides on Ray 2018 list as limitations: "Strong priors on input point-sets", "Homogeneous distribution point-set", "GTX1080: can't reach 10M points for white noise", "No support for power diagram", "No support for restricted domain" — [Basselin slides, D1 days 2021](https://members.loria.fr/Olivier.Devillers/seminaires/Slides/2021-D1-Days/Basselin.pdf)
- The search summary of the abstract says the method runs about an order of magnitude faster than multi-core CPU implementations and integrates over cells by splitting them into tetrahedra on the fly — [search summary of HAL record](https://hal.archives-ouvertes.fr/hal-01927559)

### Inferences
- The white-noise vs blue-noise gap (about 9×) comes straight from the fixed-K design. White noise needs K≈90, so the kNN heap and the per-clip security check (which recomputes every vertex) dominate. Blue noise gets by with K≈35.
- The kernel recomputes vertex coordinates (determinants) both in each conflict test and in each security check. That is high per-cell instruction work, exactly the cost that paragram later removed by caching positions.

### Gaps
- The paper's own timings (GPU model, point counts, its ms per M points) and any occupancy or profiling statements could not be read (HAL, ACM and ResearchGate all blocked). UNVERIFIED.
- Whether the original supplemental code differs from the Liu fork (for example in K presets or block size) is unverified.

---

## Q2. Basselin, Alonso, Ray, Sokolov, Lefebvre, Lévy 2021, "Restricted Power Diagrams on the GPU" (CGF 40(2), EG 2021): what changed?

### Takeaway
The kernel is OpenCL, still one thread per cell, with the workspace in `__local` (shared) memory. Unlike Ray's code, the workspace is now interleaved across the work-group: the stride is the local size, i.e. SoA across lanes. Vertices are stored as uchar4 plane triples, and the size of every array is a run-time parameter. Termination is the same 2R test on fixed-K kNN. Failures are handled by relaunching the failed cells on the GPU with K, P, V or T multiplied by 1.5, and by relaunching in double precision for `needs_exact_predicates`. The headline figure is 10M points restricted to a mesh in 300 ms on a V100. When Taveira et al. reimplemented it in CUDA fp32 as a baseline, it ran at about 2.8–3.3 M cells/s on uniform data and failed on clustered data.

### Cited Findings
- Figure 1 caption: "We compute 3D Voronoi diagrams of 10 million points restricted to a mesh in 300 ms on the GPU (Nvidia V100)". The abstract reports "a factor 2 to 3 for (unrestricted) Voronoi diagrams and a ×50 speed-up with respect to CPU implementations for restricted power diagrams" — [EG diglib PDF, via search snippet](https://diglib7.eg.org/bitstream/handle/10.1111/cgf142610/v40i2pp001-012.pdf); [HAL hal-03169575](https://hal.archives-ouvertes.fr/hal-03169575)
- The README's example parameters are `--K=20 --P=25 --V=30` for 100K blue noise, `--K=50 --P=30 --V=40` for the perturbed grid and `--K=70 --P=30 --V=40` for white noise. Float or double is chosen at compile time (`-DUSE_DOUBLE`). The output is cell barycenters/integrals — [README](https://github.com/basselin7u/GPU-Restricted-Power-Diagrams)
- **Kernel** (`ConvexCell.cl`, `compute_voro_cell`):
  - The cell arrays are `__local uchar4 *vertices, __local real *planes, __local uchar *next_boundaries, __local int *planesNeighbour`. Accessors index them as `&cc->m_planes[4*p*localSize]` with base `&planes[lid]`, i.e. interleaved by local id.
  - The main loop is `for i<K: CCNewPlane; CCClipByPlane; if CCIsSecurityRadiusReached(cc, pts[z]) break;`.
  - `CCIsSecurityRadiusReached` recomputes every vertex (`CCComputeVertexCoordinates(..., true)`) and returns `d2 > 4*v_dist`. A source comment says power diagrams should use `6*v_dist`.
  — [ConvexCell.cl](https://github.com/basselin7u/GPU-Restricted-Power-Diagrams/blob/main/ConvexCell.cl)
- **Retry logic** (`voronoi.cpp`): `if (status[vertex_overflow]) m_V *= 1.5; if (status[plane_overflow]) m_P *= 1.5; if (status[security_radius_not_reached]) m_K *= 1.5; if (status[triangle_overflow]) m_T *= 1.5;`, followed by a relaunch of only the failed ids (`recomputeIds`). The `needs_exact_predicates` cells are relaunched in double when the device supports it — [voronoi.cpp](https://github.com/basselin7u/GPU-Restricted-Power-Diagrams/blob/main/voronoi.cpp)
- The kNN is OpenCL `build_knearests` with the heap in `__local` memory, interleaved by `get_local_size(0)` — [knearests.cl](https://github.com/basselin7u/GPU-Restricted-Power-Diagrams/blob/main/knearests.cl)
- Taveira et al. on Basselin's design:
  - Neighbour search is "a k-nearest-neighbor heuristic and terminates at a fixed multiple of the current cell radius".
  - "intermediate per-thread data is held in shared memory".
  - "the shared-memory arrays are never compacted. Instead, when an array overflows, the kernel is relaunched with larger array sizes, reattempting construction from scratch".
  — [Taveira et al. 2026, Sec. 2.1 and App. A.3–A.4](https://arxiv.org/pdf/2605.06408)
- **Taveira's own CUDA fp32 reimplementation ("Basselin\*")**, uniform white noise. Throughput is end-to-end.
  - H200: 1M in 0.386 s, 5M in 1.624 s, 10M in 3.186 s, 15M in 4.619 s, i.e. about 2.6–3.2 M cells/s.
  - RTX 5090: 5M in 1.755 s, 10M in 3.608 s, i.e. about 2.8 M cells/s.
  - It failed ("-") on every clustered and real-world dataset.
  — [Taveira et al. Tables 4–7](https://arxiv.org/pdf/2605.06408)

### Inferences
- Interleaving the shared arrays by lane avoids bank conflicts when lanes touch the same slot index. paragram then took the same SoA-by-lane layout from shared into global memory.

### Gaps
- Basselin's own throughput for unrestricted Voronoi (V100, point distribution, ms) and any profiling or occupancy statements were not readable (PDF blocked). UNVERIFIED beyond the abstract and caption.
- The "factor 2 to 3" in the abstract is presumably relative to Ray 2018, but I could not confirm the baseline.

---

## Q3. Taveira, Lindström, Fatemi, Hammarstrand, Kahl 2026, "Scalable GPU Construction of 3D Voronoi and Power Diagrams" (arXiv 2605.06408, SIGGRAPH 2026, code "paragram"): exactly how does it reach its throughput?

### Takeaway
The paper exists and the code is public. It is CUDA, one thread per site, fp32 with `-use_fast_math`. Its main design choices:
- Blocks of 32 threads.
- The cell workspace (64 float4 planes, 64 cached homogeneous float4 vertex positions, 64 uchar4 vertex plane triples, plane-neighbour ids and a boundary list) is about 2.6 KB per cell. It lives in coalesced global memory, interleaved across the 32 lanes, so occupancy is not limited by shared memory.
- Neighbours are found by best-first traversal of a cuBQL BVH (leaf size 10–17), pruned by a directional radius computed from the cell's AABB.
- Planes are garbage-collected at 85% full.

The output is only adjacency (CSR), with no geometry. Measured end-to-end throughput is about 9.3–9.5 M cells/s for 5–15M uniform points on both an H200 and an RTX 5090, about 6–7 M cells/s for clustered data at 15M points, and about 7.4–8.3 M cells/s for real Radiant Foam scenes. Adjacency mismatches against CGAL are 0.1–0.2%.

### Cited Findings
**Existence, venue, code**
- arXiv:2605.06408v1 [cs.CG], 7 May 2026. SIGGRAPH Conference Papers '26, doi 10.1145/3799902.3811229. Authors: Taveira and Lindström (equal contribution), Fatemi, Hammarstrand, Kahl (Zenseact and Chalmers). The code is released at research.zenseact.com/publications/paragram — [arXiv PDF](https://arxiv.org/pdf/2605.06408); [project page](https://research.zenseact.com/publications/paragram/)
- The repository is a "standalone PyTorch CUDA package". After the paper it gained automatic chunking of the convex-cell scratch memory, and the README says paper timings should be compared with the unchunked path — [paragram README](https://github.com/zenseact/paragram)

**Parallel mapping and memory**
- "decomposes naturally into N independent volume-clipping tasks, which we map one-to-one onto GPU threads." Sites are packed as float4 (xyz plus weight) "with a single LDG.128 instruction" — [App. A, A.1](https://arxiv.org/pdf/2605.06408)
- Quote from App. A.3: "Convex cell clipping is primarily memory-bound ... In [Basselin et al. 2021], intermediate per-thread data is held in shared memory. Instead, we allocate it in coalesced global memory. While shared memory offers lower latency, we found the resulting per-thread footprint to severely limit streaming multiprocessor (SM) occupancy ... Coalesced global memory allows more warps to reside on each SM, yielding higher effective throughput in practice."
  - Planes are stored as float4. Vertices are uchar triples of plane indices.
  - "We additionally cache the explicit vertex positions in global memory rather than recomputing them each time ... caching eliminates repeated intersection computations and substantially reduces arithmetic overhead."
  - Fig. 8 describes the layout as "Structure-of-Arrays layout strided across warp threads for coalesced access".
  — [App. A.3, Fig. 8](https://arxiv.org/pdf/2605.06408)
- In the code (`voronoi_convex_cell.cuh`), `MAX_PLANES 64`, `MAX_VERTS 64` and `CELL_BLOCK_STRIDE 32`.
  - `required_bytes = MAX_VERTS*(4+16) + MAX_PLANES*(1+16+4)`, which is 2,624 B per cell in global memory.
  - The accessors are `m_planes[p*CELL_BLOCK_STRIDE + local_tid]`, `m_vertex_pos[v*32+lane]` and so on. So for a given slot index, the 32 lanes of a warp touch 32 consecutive float4 values.
  - Vertex positions are stored homogeneous (`float4` x, y, z, w with no divide). The half-space test is `dot3(v,eqn) + v.w*eqn.w > 0`.
  - The kernel is launched as `<<<launch_blocks, CELL_BLOCK_STRIDE>>>`, i.e. 32 threads per block.
  — [voronoi_convex_cell.cuh](https://github.com/zenseact/paragram/blob/main/paragram/csrc/laguerre/voronoi_convex_cell.cuh), [voronoi_ultra.cu](https://github.com/zenseact/paragram/blob/main/paragram/csrc/laguerre/voronoi_ultra.cu)
- The traversal stack holds 64 entries (`uint32_t stack_idx[64]; float stack_dist[64]`), 512 B per thread, dynamically indexed, so it presumably lives in local memory. When the stack is full, the farthest entry is replaced — [voronoi_bvh.cuh](https://github.com/zenseact/paragram/blob/main/paragram/csrc/laguerre/voronoi_bvh.cuh)
- Build flags: on compute capability ≥ 10 (Blackwell) the code uses `-Xptxas -dlcm=ca --maxrregcount=128 -DMODERN_CUDA_CARDS`, and `-use_fast_math` is on by default. On other GPUs (e.g. H200, cc 9.0) the default leaf size is 17 and kNN warm start is forced off (`knn = 0`). On modern cards the default leaf size is 10 — [_backend.py](https://github.com/zenseact/paragram/blob/main/paragram/_backend.py), [voronoi_ultra.cu](https://github.com/zenseact/paragram/blob/main/paragram/csrc/laguerre/voronoi_ultra.cu)
- The paper's App. A.2 says leaf size ℓ=17 for the H200 and 10 for the RTX 5090, using the cuBQL GPU builder. The BVH is augmented with a per-node max weight — [App. A.2](https://arxiv.org/pdf/2605.06408)

**Clipping algorithm** (App. B and code)
- Initialisation and each clip:
  - The cell starts as the AABB of the whole point set (6 planes, 8 vertices).
  - For each candidate, the vertices are classified, the outside vertices are removed "one at a time, in an order that keeps each removed vertex adjacent to the current hole boundary", and the boundary is kept "as a circular list of half-space indices".
  - New vertices `(h,a,b)` are added for each boundary pair. If nothing is removed, no clip happens.
  — [App. B](https://arxiv.org/pdf/2605.06408)
- Cheap rejections in the code, applied before any clipping work:
  1. `CCIsPointTooFar`: `|p_i-p_j|² > 4·max_radius²`, the isotropic 2R test.
  2. On modern cards, `CCHasPlane(vid)`, which checks a **64-bit Bloom-filter bitmask** of plane-neighbour ids, hashed with `vid*2654435761u >> 26`, and falls back to a linear scan on a hit.
  3. On modern cards, an "intersects" pre-scan over the cached vertex positions that exits on the first vertex outside, before the plane is appended.
  — [voronoi_convex_cell.cuh](https://github.com/zenseact/paragram/blob/main/paragram/csrc/laguerre/voronoi_convex_cell.cuh)
- After each clip that changes the cell, `CCUpdateBounds()` loops over all vertices (one reciprocal per vertex for the homogeneous divide) to recompute the AABB relative to the seed and `max_radius`. Garbage collection runs when `nb_plane >= 0.85*MAX_PLANES`. In the paper: "When the array exceeds 85% occupancy, we compact it by discarding fully clipped planes" — [voronoi_ultra.cu](https://github.com/zenseact/paragram/blob/main/paragram/csrc/laguerre/voronoi_ultra.cu); [App. A.4](https://arxiv.org/pdf/2605.06408)

**Neighbour discovery and termination**
- **Directional culling**: the bound is "the distances from the cell site to the corners of an axis-aligned bounding box (AABB) enclosing the current cell vertices ... the octant of space in which the neighbor p_j lies relative to the cell site ... determines which subset of AABB corners can possibly intersect". For a BVH box, the octants it spans are max-reduced. A site is discarded if d_ij > r_i, where d_ij is the distance from the site to the bisector (|p_i−p_j|/2 for Voronoi) — [Sec. 4.2.1–4.2.2](https://arxiv.org/pdf/2605.06408)
- In the code, `BoxRadii` holds 8 squared corner distances. `get_directional_safety_radius` combines 6 comparisons and 8 conditional selects into a max-tree, which is branch-free. The node test is `fSqrDistance(p, node.bounds)*0.25 - r² > 0`, meaning prune. This is the 2R criterion with a direction-dependent R — [voronoi_bvh.cuh](https://github.com/zenseact/paragram/blob/main/paragram/csrc/laguerre/voronoi_bvh.cuh)
- **Best-first traversal**:
  - The stack is a "priority queue ordered by distance". Children are ordered by the signed "penetration depth" δ = d − r (Alg. 1).
  - Pop does a linear scan for the minimum and then swap-removes it ("trades a small amount of extra work at pop time for significantly fewer writes at push time").
  - Popped nodes are re-tested against the shrunken radius.
  - The cell is final when the stack is empty. This is an exact termination certificate under the culling bound, and there is no fixed K.
  — [Sec. 4.3, App. A.5](https://arxiv.org/pdf/2605.06408)
- **kNN warm start** (optional, K=8, run on the same BVH): it pre-clips the cell so that pruning works from the root. It "is beneficial on GPUs with large L2 caches (e.g., ... RTX 5090) ... On datacenter GPUs such as the H200 ... the overhead of the additional pass is not recovered. The warm start is therefore disabled on such hardware" — [App. A.6](https://arxiv.org/pdf/2605.06408)

**Output, precision, accuracy**
- The output is the power-diagram adjacency in CSR form. It is obtained by recording the three defining planes of every final vertex, then compacting and symmetrising. No volumes or faces are produced — [Fig. 8](https://arxiv.org/pdf/2605.06408); [README](https://github.com/zenseact/paragram)
- Quote: "We run our method using single precision ... we note a small fraction of mismatching adjacencies (0.1–0.2%) on real data, traced to near-coincident candidate planes. Volume rendering validation across 100k rays confirm negligible differences ... (mean < 1.7e-5, median = 0.0)". The limitations section says the method "relies on floating-point arithmetic and does not target the exact geometric guarantees" — [App. D, Sec. 6](https://arxiv.org/pdf/2605.06408)
- Failures are reported only as per-site status codes: `plane_overflow, vertex_overflow, empty_cell, security_radius_not_reached, inconsistent_boundary`. There is no CPU fallback or retry in the code — [voronoi_convex_cell.cuh](https://github.com/zenseact/paragram/blob/main/paragram/csrc/laguerre/voronoi_convex_cell.cuh)

**Throughput** (all end-to-end, including BVH build, adjacency compaction and allocation; 3 warm-up runs and 10 timed runs; domain [−10,10]³; fp32; Tables 1 and 4–7)

| Dataset | GPU | Points | Time | Throughput |
|---|---|---|---|---|
| White noise | H200 | 1M | 0.123 s | 8.1 M/s |
| White noise | H200 | 5M | 0.537 s | 9.3 M/s |
| White noise | H200 | 10M | 1.064 s | 9.4 M/s |
| White noise | H200 | 15M | 1.582 s | 9.5 M/s |
| White noise | RTX 5090 | 5M | 0.530 s | 9.4 M/s |
| White noise | RTX 5090 | 10M | 1.078 s | 9.3 M/s |
| Density gradient | H200 | 15M | 1.588 s | 9.4 M/s |
| 10 Gaussian clusters (σ=0.1) | H200 | 15M | 2.530 s | 5.9 M/s |
| 5 Gaussian clusters | H200 | 15M | 2.194 s | 6.8 M/s |
| 10 clusters | RTX 5090 | 10M | 1.423 s | 7.0 M/s |
| Real scene "bicycle" | RTX 5090 | 4.2M | 0.570 s | 7.4 M/s |
| Real scene "garden" | RTX 5090 | 4.1M | 0.496 s | 8.3 M/s |
| Real scene "bicycle" | H200 | 4.2M | 0.684 s | 6.1 M/s |

Notes on the table:
- The RTX 5090 runs at 15M are "-", i.e. out of memory or failed.
- On the real scenes the RTX 5090 was faster than the H200.

— [Taveira et al. Tables 1, 4–7](https://arxiv.org/pdf/2605.06408)

- Baselines from the same tables (white noise, 15M points; CPU runs use 16 cores of an AMD Epyc 9534):
  - HXT: 2.259 s, about 6.6 M/s.
  - Geogram: 6.138 s.
  - Voro++: 41.136 s.
  - CGAL Parallel: 6.903 s.
  - gDel3D on the 5090: 1.471 s for 5M (about 3.4 M/s), and out of memory beyond that.
  — [Table 7](https://arxiv.org/pdf/2605.06408)
- **Profiling and timing breakdown**: "BVH construction ... 4.8% of the total time on large point sets. The remaining 95.2% is our fused kernel, of which the vast majority of the time is spent on cell clipping operations (96.2%) and the rest is BVH traversal and culling." The paper gives no occupancy, warp-efficiency or instruction counts — [App. D](https://arxiv.org/pdf/2605.06408)
- **Ablations** on the RTX 5090, median slowdown versus the full method:
  - Isotropic culling: 1.26× clustered, 1.33× Poisson, 1.40× real-world.
  - Depth-first traversal: 27.4× clustered (range 3.96–80.3), 1.30× Poisson (1.13–22.8), 37.2× real-world.
  — [Table 12](https://arxiv.org/pdf/2605.06408)
- **Power diagrams** (Table 3, bicycle scene): runtime is unchanged at 0.595–0.598 s for weight ratios up to 1e-4 and rises to 2.243 s at a weight ratio of 1e-1 (46.8% empty cells) — [Table 3](https://arxiv.org/pdf/2605.06408)

### Inferences
- Since 96% of the time is spent clipping, paragram is limited by the clip kernel, not by the search. Its gains come from three things: (a) higher occupancy, from dropping shared memory and using 32-thread blocks plus L1-cached global loads (`-dlcm=ca` on Blackwell); (b) cached vertex positions, which avoid determinant recomputation; and (c) cheap rejection tests (2R distance, Bloom filter, early-exit intersect scan) before any write.
- The kernel does not reorder sites spatially before launch: thread idx is the input order (`pack_points_kernel`, no Morton sort). Coherence therefore depends on the input ordering. This is my reading of the code, not something the paper states.
- `CCUpdateBounds()` after every successful clip is an O(nb_vertex) pass, with one reciprocal per vertex, at every cut.

### Gaps
- The paper has no occupancy, achieved bandwidth, warp-divergence or instruction-per-cell numbers.
- There are no fp64 results.
- Per-cell counts of candidates tested and clips are not reported.
- The paper does not say whether the supplementary material contains more than the arXiv appendices.

---

## Q4. votess (Singh, Byrohl, Nelson; arXiv 2412.04514 / JOSS): kernel design and speed

### Takeaway
votess is a SYCL port of Ray's algorithm. One fused per-point work-item runs a grid kNN (default k=64) followed by clipping. The plane and triangle arrays (P, T; defaults P=32, T=32) live in global memory, AoS per cell. Only the boundary array is in work-group local memory. Failed cells are recomputed on the CPU with k doubled and capacities doubled. On an A100-40GB it is slower than 72-core CGAL: about 13 s for about 8M float white-noise points, versus about 2.5 s for CGAL (read off the log–log plot). That is roughly 0.6 M cells/s, an order of magnitude below the other GPU codes.

### Cited Findings
- The paper: "`votess` is a library, implementing Ray's meshless algorithm ... the input set of points is sorted into a grid, and a k-nearest neighbors search is performed ... A *security radius* condition ensures that the resulting Voronoi cell is valid, and if the cell cannot be validated, an CPU fallback mechanism is used." Also: "when compared CGAL, both the CPU and GPU version falls short by around a factor of 6." Tests use "float32 white-noise". "The use of clustured datasets lies outside of both `votess` and the reference implementation" — [arXiv 2412.04514 / JOSS paper.md](https://arxiv.org/abs/2412.04514), [joss/paper.md](https://github.com/samridh-dev/votess/blob/main/joss/paper.md)
- Benchmark plot (`joss/loglog.png`): two Intel Xeon Platinum 8360Y CPUs (72 cores) and an NVIDIA A100-40GB. The GPU curve is flat at about 0.6 s up to about 6×10⁴ points (launch and transfer overhead), about 1.8 s at 10⁶, and about 13 s at the largest size (about 8×10⁶). CGAL is about 2.5 s there and Voro++ about 1,300 s. Values are read approximately off the log plot — [loglog.png](https://github.com/samridh-dev/votess/blob/main/joss/loglog.png)
- **Kernel** (`src/votess.ipp`): a single `parallel_for` over an `nd_range` (`gpu_ndsize`, default 1) calls `knni::compute` and then `cci::compute`.
  - The heap is in global memory with a `[ki*subsize + si]` layout (strided across items).
  - `P` and `T` are global buffers indexed `4*p_maxsize*i + 4*j` and `3*t_maxsize*i + 3*j` (contiguous per cell).
  - `dR` (the boundary) is a `sycl::local_accessor`.
  — [votess.ipp](https://github.com/samridh-dev/votess/blob/main/src/votess.ipp), [cc.ipp](https://github.com/samridh-dev/votess/blob/main/src/cc.ipp)
- Defaults: `ARGS_DEFAULT_K 64`, `GRID_RESOLUTION 16`, `P_MAXSIZE 32`, `T_MAXSIZE 32`, `USE_RECOMPUTE false`, `GPU_NDWORKSIZE 1` — [arguments.hpp](https://github.com/samridh-dev/votess/blob/main/include/arguments.hpp)
- Fallback (`__cpu__recompute`): cells without `security_radius_reached` are recomputed on CPU threads with `k = min(2k, N-1)`; `p_maxsize*=2` on P overflow and `t_maxsize*=2` on T overflow; this loops until every cell is certified. The security test is `radius > 4.0f * security_radius` (squared, i.e. 2R) — [votess.ipp](https://github.com/samridh-dev/votess/blob/main/src/votess.ipp), [sradius.ipp](https://github.com/samridh-dev/votess/blob/main/src/sradius.ipp)
- Output: a "jagged 2 dimensional array" of neighbour ids (natural neighbours); the input order is not preserved — [README](https://github.com/samridh-dev/votess)

### Inferences
- votess adds no GPU-specific optimisation over Ray. It has no shared-memory cell, no cached positions, a default work-group size of 1 and a non-coalesced per-cell AoS layout. It is useful as a lower bound and offers no technique worth copying.

### Gaps
- The arXiv/JOSS text gives no numeric table. Numbers are plot read-offs (approximate).

---

## Q5. Other GPU per-cell Voronoi implementations with code or numbers

### Takeaway
Two further works are relevant. (a) Xiao, Cao & Chen (arXiv 2509.07175, submitted to TVCG in Feb 2026) replace kNN plus a security radius with an *edge-based point-in-cell search* that only performs clips that contribute to the final cell. On an RTX 3080 (fp32) they report 5M white-noise cells in 0.49 s (about 10 M/s) and blue noise in 0.26 s (about 19 M/s), versus 3.12 s and 0.34 s for Ray. (b) peclet-voro is a Kokkos FP64 dual-triangle ConvexCell, the closest analogue to our code. Its repository reports 3.6 M cells/s for the full FP64 build on an RTX 5080 (gather-bound) and 8.5 M cells/s for FP64 construct-only. Liu et al. 2020 (TVCG) is a fork of Ray's code for clipped Voronoi.

### Cited Findings
- **Xiao, Cao, Chen, "Efficient Computation of Voronoi Diagrams Using Point-in-Cell Tests"**, arXiv 2509.07175v2 (16 Feb 2026), "submitted to IEEE TVCG in February 2026".
  - Point-in-cell test: q ∈ Ω_i iff the nearest site of q is v_i, decided with integer flags for robustness. It needs one nearest-neighbour query.
  - Clipping only picks an edge p_s p_t with p_s inside and p_t outside. The bisector B_ij is then found iteratively: intersect with B_ij, NN-query the intersection point, and if the nearest site is a third site k, set j←k and repeat. "our method only involves the clippings that contribute to the final results".
  - Parallel version: one thread per site iteratively clips the domain (Alg. 3, corners outside the cell found via virtual edges from the site). It uses Ray's cell data structure (planes plus corners as plane triples).
  - The grid has about 3–4 sites per bin in 3D. One million cells needed "over 20.54 millions searches" (about 20 NN queries per cell).
  - All codes were run in fp32 on an RTX 3080 (10 GB). Timings include centroid computation and exclude allocation and transfer.
  - Fig. 1, 5M white noise: Ray 3.12 s, Basselin 2.81 s, theirs 0.49 s. Blue noise: 0.34, 0.42 and 0.26 s.
  - Results are claimed exact. No code link was found in the paper.
  — [arXiv 2509.07175](https://arxiv.org/pdf/2509.07175)
- **peclet-voro (Kokkos; CUDA/HIP/OpenMP)**, "dynamic Voronoi tessellation of moving particles". Last commit seen was 1 Oct 2026. It has no peer-reviewed paper; the numbers come from the repository's own `docs/performance_report.md`.
  - Design: "One thread per cell. Each cell is built in a compact dual representation (ConvexCell): a Voronoi vertex is a triple of plane indices, the cell a small dual-triangle list that lives in registers / a tiny per-thread frame (this is what lifts GPU occupancy far above the retired half-edge engine)."
  - Gather: "a voro++-derived presorted worklist walks neighbour grid cells closest-first and stops at the security radius."
  - Capacities: MAXP=64, MAXT=112.
  - Throughput, uniform Poisson, FP64, RTX 5080 (a consumer Blackwell card with weak FP64): full build 2.99–3.61 Mcells/s for N = 100k–2M; construct-from-cache (no gather) 8.52 Mcells/s.
  - "~14–17 Mcells/s FP32 construct, which beat the Liu-2020 GPU code on the same hardware".
  - "The gather dominates ... the neighbour gather costs roughly as much as the entire construct + geometry."
  - Volume conservation ≤ 4×10⁻¹⁵.
  — [peclet-voro README](https://github.com/computational-chemical-engineering/peclet-voro), [docs/performance_report.md](https://github.com/computational-chemical-engineering/peclet-voro/blob/main/docs/performance_report.md), [params.hpp](https://github.com/computational-chemical-engineering/peclet-voro/blob/main/include/peclet/voro/params.hpp)
- **Liu, Ma, Guo, Yan 2020**, "Parallel Computation of 3D Clipped Voronoi Diagrams" (IEEE TVCG 28(2)), is CUDA code built on Ray's. Xiao et al. could only run it on up to 100K sites on the complex domain — [3D-Voronoi-GPU](https://github.com/xh-liu-tech/3D-Voronoi-GPU); [Xiao et al.](https://arxiv.org/pdf/2509.07175)
- Xiao, Li, Xu 2023, "Parallel Computation of 3D Accurate Power Diagrams on GPU" (J. CAD & CG 35(12)): extends the meshless method to power diagrams by lifting to 4D. Seen only in search results; content unverified — [jcad.cn](https://jcad.cn/en/article/doi/10.3724/SP.J.1089.2023.2023-00018)

### Inferences
- Xiao et al.'s "only contributing clips" idea targets the same waste as our ~50 candidate tests for ~17–24 cuts. But it replaces candidate tests with about 20 nearest-neighbour queries per cell, each a grid walk, so on a GPU it trades clip work for search divergence. Their white-noise speedup comes mostly from avoiding the large fixed K that Ray needs.

### Gaps
- I found no public code for Xiao et al. 2025/26.
- peclet-voro numbers are self-reported and have no A100 data.
- No CGAL or Geogram GPU per-cell Voronoi kernel was found. Geogram's per-cell code (VBW::ConvexCell) is CPU-only, as far as I could see in this search.
- No published GPU per-cell kernel from SPH or astrophysics codes was found besides votess.

---

## Q6. Comparison with our kernel (A100, Kokkos/CUDA, exact double; ~14 M cells/s jittered, ~5–7 M cells/s uniform random at 7.4M points; ~6% occupancy; ~1.2 KB/cell shared; ~100k instructions/cell; ~8.7/32 lanes active)

### Takeaway
On uniform random points, the fastest published local GPU kernel (paragram) reaches about 9.3–9.5 M cells/s end-to-end in **fp32**, on an H200 or RTX 5090, outputting adjacency only. Our 5–7 M cells/s in **exact double** on an A100, with full cells, is in the same range. Adjusted for hardware (the H200 has more SMs and bandwidth, the 5090 runs fp32 only) it is probably at parity or better. On jittered or blue-noise-like inputs, our ~14 M/s matches the best published blue-noise number (Ray: about 14.7 M/s, fp32, RTX 3080, 5M) and is below Xiao's 19 M/s (fp32, centroids only).

The published techniques that target our specific bottlenecks:
1. Occupancy: move the workspace out of shared memory into global memory interleaved by lane (SoA, stride 32) with L1 caching, plus 32-thread blocks (paragram).
2. Instruction count: cheap pre-rejection before any write (the 2R distance test; a 64-bit Bloom filter for duplicate planes; an early-exit "any vertex outside" scan on cached positions), cached homogeneous positions, and directional AABB radii instead of an isotropic R.
3. Ordering: best-first ordering by penetration depth, and a K=8 kNN warm start (paragram).

None of the papers measures warp divergence. None reduces it explicitly beyond these, and all use one thread per cell.

### Cited Findings
- Throughputs for comparable sizes:
  - paragram, white noise, fp32: 9.3 M/s at 5M (H200, 0.537 s); 9.4–9.5 M/s at 10–15M (H200); 9.3–9.4 M/s at 5–10M (RTX 5090). Real scenes: 6.1–8.3 M/s. Clustered: 5.9–7.0 M/s — [Taveira et al. Tables 1, 4–7](https://arxiv.org/pdf/2605.06408)
  - Basselin-style kNN in CUDA fp32: about 2.8–3.2 M/s on uniform data — [Taveira et al. Table 7](https://arxiv.org/pdf/2605.06408)
  - Ray fp32 on an RTX 3080 at 5M points: about 1.6 M/s white noise and about 14.7 M/s blue noise. Xiao et al.: about 10.2 M/s white noise and about 19 M/s blue noise — [Xiao et al. Fig. 1](https://arxiv.org/pdf/2509.07175)
  - votess on an A100, white noise fp32: about 0.6 M/s — [votess loglog.png](https://github.com/samridh-dev/votess/blob/main/joss/loglog.png)
  - peclet-voro FP64 on an RTX 5080: 3.6 M/s full build and 8.5 M/s construct-only. FP32 construct-only: about 14–17 M/s — [peclet-voro performance report](https://github.com/computational-chemical-engineering/peclet-voro/blob/main/docs/performance_report.md)
- **Occupancy versus shared memory.** paragram explicitly moved from shared to "coalesced global memory" because "the resulting per-thread footprint [severely limits] SM occupancy". It uses an SoA layout with stride 32 per warp (`p*32 + lane`), 32-thread blocks, `-dlcm=ca` (cache global loads in L1) and `--maxrregcount=128` on Blackwell. Its footprint is 2,624 B per cell in global scratch plus a 64-entry traversal stack — [App. A.3](https://arxiv.org/pdf/2605.06408); [voronoi_convex_cell.cuh](https://github.com/zenseact/paragram/blob/main/paragram/csrc/laguerre/voronoi_convex_cell.cuh); [_backend.py](https://github.com/zenseact/paragram/blob/main/paragram/_backend.py)
- **Instruction count reducers in paragram**:
  - Cached homogeneous vertex positions: the conflict test is a 4-term dot product, with no determinant and no divide.
  - The early-out chain `CCIsPointTooFar` (2R) → `CCHasPlane` (Bloom bitmask) → "intersects" scan, run before appending the plane.
  - Plane garbage collection at 85%, which keeps loops short.
  - A branch-free 8-corner directional radius.
  - The min-scan pop stack.
  — [voronoi_convex_cell.cuh](https://github.com/zenseact/paragram/blob/main/paragram/csrc/laguerre/voronoi_convex_cell.cuh), [voronoi_bvh.cuh](https://github.com/zenseact/paragram/blob/main/paragram/csrc/laguerre/voronoi_bvh.cuh)
- **Ordering effects**: best-first versus DFS is worth 1.13–22.8× on Poisson data and up to 80× on clustered data. Directional versus isotropic culling is worth 1.25–1.44× on Poisson data. The K=8 warm start helps only on the RTX 5090 (large L2) and was disabled on the H200 — [Table 12, App. A.6](https://arxiv.org/pdf/2605.06408)
- **Precision**: every published GPU kernel benchmarked here runs in fp32 (paragram also with fast-math; Ray and Basselin with optional filters and a double relaunch; Xiao fp32 with integer-flag robustness). paragram reports 0.1–0.2% adjacency mismatches versus CGAL. Only peclet-voro reports FP64 GPU numbers — [Taveira App. D](https://arxiv.org/pdf/2605.06408); [Xiao et al. Sec. VII-B](https://arxiv.org/pdf/2509.07175); [peclet-voro report](https://github.com/computational-chemical-engineering/peclet-voro/blob/main/docs/performance_report.md)
- **Divergence and profiling**: the only profiling statement published is paragram's breakdown (95.2% fused kernel; 96.2% of that clipping). No paper reports active lanes, occupancy figures or instructions per cell — [Taveira App. D](https://arxiv.org/pdf/2605.06408)

### Inferences
- **Occupancy (our ~6%)**: paragram's evidence is that, for this workload, latency hiding through more resident warps beats shared-memory latency. Our 1.2 KB float shared per cell maps naturally onto a global SoA layout `[slot][lane]` with L1 caching (`-dlcm=ca`, or `__ldca`/`ld.global.ca`). This is the most direct transferable change.
  - Caveat: with only about 8.7 of 32 lanes active, lanes in a warp often access different slot indices, so "coalescing" degrades to L1-sector hits. paragram does not quantify this either.
  - Their 32-thread blocks also suggest testing small blocks with many blocks per SM.
- **Instruction count (~100k per cell)**: paragram avoids per-candidate determinant work completely (cached positions) and rejects most candidates with O(1) work (the 2R distance test and the Bloom filter) before an O(nv) scan, which itself exits early.
  - We already store positions and use 2R. The remaining paragram ideas are the directional AABB radius (8 corner radii; about 1.3× fewer candidates on uniform data) and doing the "any vertex outside" scan with early exit before any topology work.
  - Their per-clip `CCUpdateBounds` is O(nv). An incremental max-radius update is something none of the papers do (my inference).
- **Divergence (8.7/32 lanes)**: no published kernel uses warp or team per cell, or any cooperative scheme. The literature gives no direct remedy beyond reducing per-cell variance in work: best-first ordering, warm start with K=8, and spatial ordering of threads (Ray sorts points into grid order; paragram apparently does not reorder).
  - Our own team/warp experiments go beyond the published state of the art. We should not expect the literature to have solved this.
- **Certification**: paragram's termination (stack empty under a directional bound) is the same family as 2R, but anisotropic and hierarchical, so it costs little on uniform data. Our ~20 M cells/s ceiling without certification is above every published end-to-end number except Xiao's blue-noise result (fp32, centroids only).
- **Fair-comparison caveats**: paragram is fp32 with fast-math, outputs adjacency only, and its timings include BVH build and CSR compaction. Ours is exact double with full cells (vertices and faces). Ray, Basselin and Xiao report fp32 on consumer GPUs, and Xiao's numbers include centroid computation. A100 FP64 is half of its non-tensor FP32 rate, so precision alone should cost us at most about 2× in arithmetic-bound phases.

### Gaps
- The original Ray 2018 and Basselin 2021 papers' own throughput, occupancy and profiling statements could not be read (blocked).
- No published A100 numbers exist for paragram, Ray, Basselin or Xiao, so the cross-GPU normalisation above is an estimate.
- None of the works reports candidates tested per cell, cuts per cell, faces per cell or bytes moved per cell, so our ~50 candidates and ~17–24 cuts cannot be compared directly.
