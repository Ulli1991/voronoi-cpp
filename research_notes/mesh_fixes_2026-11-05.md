# Mesh library fixes, 2026-11-05 (branch mesh-fixes)

Bugs found by the mesh tests (branch mesh-tests). Build dir and logs:
build-meshfix/. Before-binaries: build-meshfix/before/.

## Findings as they come

- #21: exact clipping of a shell (k = 800, 2000; 512 background) ends with
  only plane_overflow cells (3 and 20, at big_planes = 128). Every
  `degenerate` of the issue comes from the rounded first pass; the exact pass
  recovers all of them. Measured with `shell_voronoi::construct(..., exact=true)`.

- #5, the pairs (same run as `voronoi_bench step 262144 <kind> 0.3 3 --verify`,
  OpenMP, same numbers as on the A100). clustered step 1: pair 50684-129028,
  lattice step 3: pair 170295-174392, area 6.8e-16 and 1.0e-14 spacings^2 as
  the issue says. Each is a triangle (edge of three tetrahedra). Python integers
  on the dumped grid positions: all three circumspheres empty of every particle
  within 6 spacings, none on a sphere, exact area 0.19 and 2.9 grid steps^2
  (non-zero), corners 1.5 and 22 grid steps apart. The exact clipping
  (`construct(..., exact = true)`) has the face too. So the mesh is right and the
  rounded clipping drops it (magnitude <= sliver, shell_voronoi_warp.hpp:526), as
  TESTING.md says. wave step 3: pair 132213-136373, area 1.9e-15, the same:
  empty spheres, exact area 0.54 grid steps^2. No defect, nothing to fix.

- #18: the walks around an edge need no room of their own but for two stack
  arrays (from_rings' dropped faces, tet_flips' region edges). Both now walk
  with lowest_around_edge and keep nothing. The bound only stops a walk on
  broken adjacency; the one walk that ran before the check (from_rings) moved
  after it (tet_mesh::represent_dropped). Options: (a) the bound far above n,
  chosen: no memory (65 ints of stack less), no extra instruction on a ring of
  usual length; (b) 256 or 1024 with fixed arrays: 1-4 KB of GPU stack per
  thread in two kernels and still a limit; (c) an overflow pass on the host:
  a report path for nothing, since nothing needs the arrays. Speed: see the job.
