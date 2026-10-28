# Local Voronoi construction: literature and implementation notes

2026-10-28. Scope: local geometry from candidate neighbors, before hydro.

* Voro++ (Rycroft): convex edge traversal and plane clipping; avoids a global
  Delaunay mesh. https://escholarship.org/content/qt8sf4t5x8/qt8sf4t5x8.pdf
* Ray, Sokolov, Lefebvre, Levy (2018), Meshless Voronoi on the GPU:
  independent nearest-neighbor-based cell computation and streamed integrals.
  https://doi.org/10.1145/3272127.3275092
* Singh, Byrohl, Nelson (2024), votess: nearest-neighbor lists, cell clipping,
  security-radius validation and CPU fallback. https://arxiv.org/html/2412.04514v1
* Taveira et al. (2026), Scalable GPU Construction of 3D Voronoi and Power
  Diagrams: compact cell data in coalesced global memory; cached positions,
  plane garbage collection, directional culling and best-first BVH traversal.
  https://arxiv.org/html/2605.06408v1 (Appendix A.3, A.4, D precision).

Implications for this library (our interpretation): first compare explicitly
coalesced workspace arrays against compiler-managed per-thread local arrays.
Shared-memory storage is not automatically better: the 2026 paper reports
occupancy costs large enough to favor coalesced global workspace. This argues
for measuring both layouts before committing to warp-per-cell reconstruction.
Directional AABB rejection may avoid more classification than spherical bounds.
BVH results concern integrated search plus clipping; they do not directly
establish speed from an already-computed candidate list. Exact topology matters:
the 2026 single-precision implementation reports 0.1-0.2% adjacency mismatches
on real data, which we should not assume acceptable for hydro geometry.

Current first experiments: compact classification bit mask in local_voronoi.hpp;
fixed_point_positions.hpp unit-box uint32 input adapter, selected with
construction_scaling policy relative_early_fixed32. Coordinates use floor(x*2^32)
encoding; error per coordinate is below one unit-box tick (2^-32). Distinct sites
can quantize together. Integer differences use int64 before periodic wrapping.
DenseCellSearch still gathers doubles and reconstruction remains double, so the
first experiment only saves 12 input bytes/particle. No speedup is claimed.
It is not yet integer storage throughout search or reconstruction.

195^3 profiler: baseline/early DRAM reads713.90/401.27GB, writes151.79/136GB,
L2 traffic3.16/1.74TB, occupancy23.74/23.58%, long-scoreboard74.81/67.71%.
Reports build-roofline/31589997/. These counters alone do not prove saturated
DRAM bandwidth; memory latency and access layout remain candidates.
