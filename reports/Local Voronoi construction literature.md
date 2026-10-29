# Per-cell clipping wins GPU Voronoi meshing

On GPUs, **per-cell clipping is the efficient algorithm for 3D Voronoi meshing**. Each cell is built independently by cutting a box with the bisector planes of nearby points and certified with a security radius. That family holds every top position in published benchmarks. Global GPU Delaunay (gDel3D) is 1.3–2.4× slower and runs out of memory beyond a few million sites. Local test-and-repair pays off only when points barely move.

The algorithms rank as follows:

1. **Taveira et al. 2026 ("paragram")** is the verified leader. It runs **9.3–9.5 M cells/s end-to-end on uniform random points** on an H200 or RTX 5090 and stays at 6–7 M/s on clustered data.
2. **Xiao, Cao & Chen 2025/26** reports the fastest raw numbers on the weakest hardware: about **10 M/s on white noise and 19 M/s on blue noise** on an RTX 3080. It has no public code.
3. **Ray et al. 2018** is fast only on near-uniform input: about 14.7 M/s on blue noise, but 1.6 M/s on white noise.
4. Basselin 2021, peclet-voro, gDel3D and votess follow well behind.

Every leading kernel is **fp32 and one thread per cell**, and none guarantees that neighbouring cells agree on shared faces. Their speed comes from five design choices:

- neighbour search that adapts to the local point distribution (best-first BVH or edge-driven search, not fixed-K kNN);
- a compact per-cell workspace held where it does not limit occupancy;
- cached vertex positions;
- cheap rejection of candidates before any clipping work;
- in-kernel overflow handling.

Our exact-double A100 kernel already matches this plateau. It runs at ~5–7 M cells/s on uniform random points and ~14 M/s on jittered points, with a ~20 M/s ceiling without certification. The 100 M cells/s double target is therefore 5–10× beyond anything published. Reaching it requires solving warp divergence, which no published kernel addresses, on top of adopting paragram's occupancy and instruction-count techniques.

## Ranked comparison of GPU Voronoi construction algorithms

The ranking weighs throughput on hard (uniform random, "white noise") input first, then robustness, then verifiability. "Easy" input means blue noise or perturbed grids, which corresponds to our jittered case.

| Rank | Algorithm | Family | GPU | Precision | Hard input (cells/s) | Easy input (cells/s) | Robustness / consistency |
|---|---|---|---|---|---|---|---|
| 1 | **Taveira et al. 2026 (paragram)** | per-cell clip, best-first BVH | H200, RTX 5090 | fp32, fast-math | **9.3–9.5 M** (5–15M pts, end-to-end) | ~9.4 M (density gradient) | Handles clustered data (5.9–7.0 M/s). 0.1–0.2% adjacency mismatch vs CGAL, symmetrised after the kernel. No fallback. |
| 2 | **Xiao, Cao & Chen 2025/26** | per-cell clip, edge-based point-in-cell search | RTX 3080 | fp32 + integer flags | **~10 M** (5M) | **~19 M** (5M) | Claimed exact. No code found. Clustered data not reported. |
| 3 | **Ray et al. 2018** | per-cell clip, fixed-K kNN | V100 / RTX 3080 | fp32 (optional filter) | ~1.6 M (5M, RTX 3080) | ~14.7 M (RTX 3080); ~12.5 M on V100 (teaser, snippet) | Accepts zero-area facets. Flags cells needing exact predicates. Runs out of memory above 7M white-noise points. |
| 4 | **Basselin et al. 2021** | per-cell clip, fixed-K kNN, relaunch | V100 | fp32, double relaunch | ~2.6–3.2 M (Taveira's CUDA re-implementation) | 10M restricted in 0.3 s (snippet, unverified) | Relaunches failed cells with 1.5× capacity. Fails on all clustered cases. |
| 5 | **peclet-voro 2026 (cold build)** | per-cell clip, presorted grid worklist | RTX 5080 | **fp64** | 3.6 M full; 8.5 M construct-only | — | Volume to 4e-15. Consumer card with weak FP64, so not representative of an A100. |
| 6 | **gDel3D 2014** | global GPU Delaunay + CPU star splaying | RTX 5090, H200 | double | ~3.5–4.4 M (1M); 3.4 M (5M) | — | Exact after CPU repair. Out of memory at ≥10M points on both GPUs. |
| 7 | **votess 2024** | per-cell clip, kNN (port of Ray) | A100-40GB | fp32 | ~0.6 M (~8M, plot read-off) | — | CPU fallback until every cell is certified. |
| — | **peclet-voro repair** | local test-and-repair (3D) | RTX 5080 | fp64 | 21.8 M at δ/h = 1e-4; slower than rebuild at δ/h ≥ 1e-2 | — | Exact only after adding near-miss candidate lists. |
| — | **cellGPU 2017** | local test-and-repair (2D) | K40, 980Ti | double | figures only | — | GPU test, CPU repair. Slows down in fluid-like regimes. |
| — | **Ours** | per-cell clip, grid search, exact | **A100** | **exact fp64** | **5–7 M** (7.4M pts) | **~14 M** (ceiling ~20 M uncertified) | Exact; full cells, ~16 faces/cell. |

Sources: [Taveira et al. Tables 1, 4–7, App. D](https://arxiv.org/pdf/2605.06408); [Xiao et al. Fig. 1](https://arxiv.org/pdf/2509.07175); [Ray et al. ACM DL](https://dl.acm.org/doi/abs/10.1145/3272127.3275092); [Ray-derived code](https://github.com/xh-liu-tech/3D-Voronoi-GPU); [Basselin et al. EG](https://diglib.eg.org/handle/10.1111/cgf142610); [Basselin code](https://github.com/basselin7u/GPU-Restricted-Power-Diagrams); [peclet-voro performance report](https://github.com/computational-chemical-engineering/peclet-voro/blob/main/docs/performance_report.md); [gDel3D](https://www.comp.nus.edu.sg/~tants/gdel3d.html); [votess](https://github.com/samridh-dev/votess/blob/main/joss/loglog.png); [cellGPU](https://ar5iv.labs.arxiv.org/html/1702.02939).

paragram ranks first despite Xiao's higher raw numbers. Its results are verified against public code, cover clustered and real-world data, and include the BVH build (4.8% of time) and adjacency compaction in the timing ([Taveira et al.](https://arxiv.org/pdf/2605.06408)). Xiao's results come from a three-generation-older consumer GPU and include centroid computation. Per unit of hardware, they are probably the fastest published, but without code or clustered-data tests that cannot be confirmed ([Xiao et al.](https://arxiv.org/pdf/2509.07175)).

The ranking yields three robust conclusions:

- **The fixed-K kNN design collapses on white noise.** Ray's code runs 9× slower on white noise than on blue noise. White noise needs K ≈ 90 neighbours per cell, blue noise only K ≈ 35 ([params.h](https://github.com/xh-liu-tech/3D-Voronoi-GPU/blob/master/params.h)).
- **Global GPU Delaunay loses to per-cell clipping on speed and memory.** The 2026 benchmarks find no exception except small clustered sets, where gDel3D wins at 1M points ([Taveira et al.](https://arxiv.org/html/2605.06408)).
- **The 2018 V100 headline has not clearly been beaten on uniform data in eight years.** Progress since then has gone into robustness on clustered data, not peak speed.

## Why some designs are fast on GPUs and others are not

### Neighbour search is where non-uniform data is won or lost

The three published search strategies differ sharply.

- **Fixed-K kNN** (Ray, Basselin, votess) runs a separate kernel. A per-thread heap of K neighbours sits in shared memory over a grid with ~3.1 points per bin ([knearests.cu](https://github.com/xh-liu-tech/3D-Voronoi-GPU/blob/master/knearests.cu)). K must be sized for the worst cell, so typical cells waste work and outlier cells fail the security radius and need a relaunch.
- **Presorted grid shells walked closest-first until the security radius is reached** (peclet-voro, and our code) adapts K per cell. peclet-voro reports that this gather "costs roughly as much as the entire construct + geometry" ([README](https://github.com/computational-chemical-engineering/peclet-voro)).
- **Best-first BVH traversal with directional culling** (paragram) has no K at all. The cell is final when the priority stack empties under a culling bound, which is an exact termination certificate. Best-first traversal beats depth-first by **27.4× on clustered data and 37.2× on real-world data**, but by only 1.30× on Poisson data. Directional culling against the 8 corners of the cell's bounding box, chosen by octant, beats an isotropic radius by **1.26–1.40×** ([Taveira et al. Table 12](https://arxiv.org/pdf/2605.06408)).
- **Xiao's edge-based search** is a fourth option. It clips only along edges with one endpoint inside and one outside, finding the true cutting site by nearest-neighbour queries. It thus performs "only the clippings that contribute to the final results", at about 20 NN queries per cell ([Xiao et al.](https://arxiv.org/pdf/2509.07175)).

For near-uniform hydro meshes, grid search with an adaptive stop is adequate. BVH best-first matters for cosmological clustering.

### Workspace placement sets occupancy

Workspace placement is the most consequential GPU-specific choice.

- **Ray** keeps the whole cell in shared memory, about 1.4 KB per thread, laid out contiguously per thread, with blocks of 16 threads ([voronoi.h](https://github.com/xh-liu-tech/3D-Voronoi-GPU/blob/master/voronoi.h)).
- **Basselin** interleaves the shared arrays by lane (stride = work-group size) to avoid bank conflicts ([ConvexCell.cl](https://github.com/basselin7u/GPU-Restricted-Power-Diagrams/blob/main/ConvexCell.cl)).
- **paragram** abandons shared memory. The authors write that the per-thread footprint "severely limit[s] streaming multiprocessor (SM) occupancy … Coalesced global memory allows more warps to reside on each SM, yielding higher effective throughput in practice." They store 2,624 B per cell in global memory, interleaved as `slot*32 + lane`, use 32-thread blocks, and on Blackwell add L1 caching (`-dlcm=ca`) and `--maxrregcount=128` ([App. A.3](https://arxiv.org/pdf/2605.06408); [voronoi_convex_cell.cuh](https://github.com/zenseact/paragram/blob/main/paragram/csrc/laguerre/voronoi_convex_cell.cuh); [_backend.py](https://github.com/zenseact/paragram/blob/main/paragram/_backend.py)).
- **peclet-voro** says its compact dual-triangle cell, held in "registers / a tiny per-thread frame", lifted occupancy "far above the retired half-edge engine" ([README](https://github.com/computational-chemical-engineering/peclet-voro)).
- **votess** keeps an AoS-per-cell global layout with a default work-group size of 1, and it is the slowest code in the comparison ([votess.ipp](https://github.com/samridh-dev/votess/blob/main/src/votess.ipp)).

Two compactness tricks recur across the fast codes: each vertex is stored as three 1-byte plane indices, and planes are garbage-collected at 85% fill ([Taveira et al. App. A.4](https://arxiv.org/pdf/2605.06408)).

### Work per cell separates fast clipping from slow clipping

Clipping takes **96.2% of paragram's fused kernel** ([App. D](https://arxiv.org/pdf/2605.06408)), so arithmetic per clip decides throughput. Ray's code recomputes 3×3 determinants in every conflict test, and its security check recomputes every vertex after every clip ([voronoi.cu](https://github.com/xh-liu-tech/3D-Voronoi-GPU/blob/master/voronoi.cu)). Basselin and votess keep the same pattern. paragram removes this cost in three ways:

- It caches homogeneous vertex positions, so the half-space test is a 4-term dot product with no divide.
- It runs a cheap rejection chain before any write: a 2R distance test, then a 64-bit Bloom-filter bitmask for duplicate planes, then an "any vertex outside" scan that exits on the first hit.
- It compacts dead planes ([voronoi_convex_cell.cuh](https://github.com/zenseact/paragram/blob/main/paragram/csrc/laguerre/voronoi_convex_cell.cuh)).

Xiao's search attacks the other waste, candidate planes that never cut the cell. A 3D Poisson-Voronoi cell has only **15.535 faces on average**, with maxima around 41 in 1e6 cells ([Lazar et al.](https://arxiv.org/pdf/1401.1736)). Every candidate beyond ~16 is overhead.

### Divergence, precision and failure handling

**Divergence is unmeasured territory.** Every published kernel maps one thread to one cell. None reports active lanes, occupancy or instructions per cell; the only published profile is paragram's time split. The only levers in the literature are indirect: spatial ordering of threads (Ray counting-sorts into grid order; paragram apparently does not), best-first ordering, and a K=8 kNN warm start. The warm start helps on the RTX 5090 and is disabled on the H200 because "the overhead of the additional pass is not recovered" ([App. A.6](https://arxiv.org/pdf/2605.06408)).

**Precision is uniformly fp32 in the fast codes.** Ray's code has an optional arithmetic filter that flags cells for exact predicates, and Basselin relaunches those cells in double ([voronoi_defs.h](https://github.com/xh-liu-tech/3D-Voronoi-GPU/blob/master/voronoi_defs.h); [voronoi.cpp](https://github.com/basselin7u/GPU-Restricted-Power-Diagrams/blob/main/voronoi.cpp)). gDel3D's runtime changes by less than 0.5% between float and double ([Taveira et al.](https://arxiv.org/html/2605.06408)). No published float-versus-double ablation exists for a per-cell clipping kernel.

**Failure handling** ranges widely:

- relaunch from scratch with arrays 1.5× larger (Basselin);
- a CPU loop that doubles K until every cell is certified (votess);
- in-kernel plane compaction plus status codes and no fallback (paragram);
- CPU star splaying (gDel3D).

In-kernel handling is the GPU-friendly choice. A rare GPU retry is acceptable. Host round-trips are not.

### What a fast GPU algorithm should combine

The evidence supports a specific recipe:

- adaptive neighbour discovery with a certified stop (grid shells for near-uniform data, best-first BVH for clustered data), with directional bounding-box culling;
- one compact dual-triangle cell, stored in global memory interleaved by lane, or in registers if small enough, never in a large shared-memory block;
- cached homogeneous vertex positions;
- an O(1)-first rejection chain before any topology edit;
- plane compaction and a rare on-GPU retry instead of worst-case sizing;
- spatially sorted thread order.

Two further elements are missing from all published kernels and are needed for hydro: exact or filtered predicates that guarantee symmetric faces, and some remedy for lane divergence.

## Implications for our kernel and the 100 M cells/s double target

**We are at the published frontier, so the remaining factor must come from new work.** Our 5–7 M/s on uniform random points, in exact double with full geometry on an A100, sits in the same band as paragram's fp32, adjacency-only 9.3 M/s on newer GPUs. Our ~14 M/s on jittered points matches Ray's blue-noise figure. Our ~20 M/s ceiling without certification matches Xiao's best fp32 number. Since certification removal only reaches 20 M/s, the bottleneck is clipping, which matches paragram's 96% profile.

**The A100's instruction issue rate makes divergence and instruction count jointly decisive.** This is our arithmetic from spec-sheet figures, not a literature result. It assumes 108 SMs, 4 schedulers per SM, 1.41 GHz, and one warp instruction per scheduler per cycle:

- Peak issue rate is about **6e11 warp instructions/s**.
- At 100 M cells/s, that is ~6,100 warp issues per cell. If one warp builds 32 cells, that is ~195k lane slots per cell.
- At 8.7/32 lanes active, only ~53k of those slots do useful work.
- A kernel needing ~100k thread instructions per cell is therefore capped near 50 M cells/s even with perfect issue and no memory stalls.

So the target needs **both roughly halving instructions and roughly doubling lane utilisation**. The literature addresses only the first.

The literature-backed changes, in the order we'd expect them to pay off (our estimate: 1.3–2× in total on the phases they touch):

1. Move the ~1.2 KB shared workspace to global memory interleaved by lane, with L1 caching and small blocks, to fix ~6% occupancy. Caveat: with 8.7/32 lanes active, lanes touch different slot indices, so "coalesced" access degrades to L1 sector hits.
2. Add directional culling and an early-exit "any vertex outside" pre-scan to cut our ~50 candidates per ~17–24 cuts.
3. Update the bounding radius incrementally after each clip. paragram recomputes it in O(nv) every time.
4. Size the fast path for the mean of ~16 faces and push the tail to an on-GPU retry.

Divergence needs approaches no paper has published. Our team and warp-per-cell experiments, persistent work queues and lane-coherent clipping are all beyond the state of the art.

**Precision and consistency are our differentiator.** paragram's 0.1–0.2% adjacency mismatch would mean about 1e5 broken faces per step in a 1e8-cell mesh, which is unacceptable for conservative fluxes ([Taveira et al. App. D](https://arxiv.org/pdf/2605.06408)). No GPU paper delivers exact, symmetric per-cell faces. The fp32 numbers in the ranking do not include the cost of making their output hydro-safe.

**Topology reuse should replace search, not skip rebuilds.** peclet-voro's repair runs 6.5× faster than a rebuild at δ/h = 1e-4, but only 0.84–0.86× as fast at δ/h ≥ 1e-2, where 71.5% of cells are flagged each step. The repair became exact only after it started storing "near-miss" candidates ([performance report](https://raw.githubusercontent.com/computational-chemical-engineering/peclet-voro/main/docs/performance_report.md); [README](https://github.com/computational-chemical-engineering/peclet-voro)). The repository's definition of δ/h was not verified; we read it as per-step displacement over mean spacing. Moving-mesh generators travel roughly C·Mach·h per step, about 0.03–0.3 h (our estimate), which is beyond break-even. Seeding candidates from previous neighbours plus near-miss lists, then running the normal clip-and-certify, removes the gather, which peclet-voro measures at about half the cost. It does not remove clipping.

## Supporting context from CPU, astrophysics and CFD codes

Production moving-mesh codes confirm that the mesh is rebuilt every step and that per-cell construction is a legitimate hydro design. **AREPO** rebuilds a global per-domain Delaunay mesh each timestep, with filtered predicates and exact GMP fallback. The large run took 516 s on 1024 CPUs for 5 billion points, about 9e3 points/s per CPU (our arithmetic) ([Springel 2010](https://ar5iv.labs.arxiv.org/html/0901.4107); [Weinberger et al. 2020](https://ar5iv.labs.arxiv.org/html/1909.04667)). **Shadowfax** uses exact integer predicates and builds ~1.5e4 cells/s serially in 3D. Its grid construction costs **about 2× the hydro integration** ([Vandenbroucke & De Rijcke 2016](https://ar5iv.labs.arxiv.org/html/1605.03576)). If that ratio holds on GPUs, a 100 M cells/s hydro step needs construction well above 100 M/s.

Two codes use per-cell clipping inside a hydro scheme:

- **MANGA** clips each cell with voro++ and stops when the remaining neighbours are farther than twice the farthest vertex. It reaches ~1e4 cells/s per core ([Chang, Wadsley & Quinn 2017](https://ar5iv.labs.arxiv.org/html/1707.05333)).
- **SILVA** (Lagrangian Voronoi CFD) prefers the "direct approach … where no dual mesh is constructed". It uses the same half-distance certificate and warm-starts each cell from the previous step's neighbours ([Kincl, Peshkov & Boscheri 2024](https://arxiv.org/html/2405.04116)).

TESS seeds candidates with neighbours-of-neighbours ([Duffell & MacFadyen 2011](https://arxiv.org/html/1104.3562)).

Update-versus-rebuild studies from computational geometry agree with peclet-voro. Flip repair is up to ~20× faster than rebuilding for tiny 3D motions ([Schaller & Meyer-Hermann 2003](https://ar5iv.arxiv.org/html/physics/0302018)). Yet rebuilding from scratch is "surprisingly a very viable option" unless points move only slightly ([Manhães de Castro et al. 2009](https://hal-univ-tlse3.archives-ouvertes.fr/INRIA/inria-00413344)).

No GPU Voronoi moving-mesh hydro code exists. AREPO's GPU port covers radiative transfer only ([Zier et al. 2024](https://arxiv.org/abs/2404.17630)). The GPU-hydro reference remains Cartesian: AthenaK exceeds 1e9 cell updates/s on one GH200 ([Stone et al. 2024](https://arxiv.org/abs/2409.16053)).

## Conclusion

The question of which algorithm is efficient on GPUs has a clear answer: per-cell clipping with adaptive, certified neighbour discovery, a compact occupancy-friendly workspace and cheap early rejection. paragram is the best verified instance of that recipe, and Xiao's edge-driven search is the most promising idea for cutting wasted clips. That recipe has plateaued at 10–20 M cells/s in fp32. Our exact kernel already sits on the plateau, so 100 M/s in double precision cannot come from adopting it alone.

The decisive new ingredient is lane-coherent execution. On the A100's issue budget, instruction count and divergence must both fall by about 2× together. A second ingredient is topology-seeded candidates that remove the search phase. Five results would be new to the literature: warp-cooperative per-cell construction, exact fp64 GPU construction, symmetric face geometry produced directly by independent cells, topology-seeded GPU candidate generation for moving meshes, and the first GPU Voronoi moving-mesh hydro throughput figure.
