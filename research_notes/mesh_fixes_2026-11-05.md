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

- Correction, #18: commit 79c8de2 has the message of the fix but only its notes
  and test expectations: the edit of the sources was refused with the command
  that held it, and the check I then ran skipped the cases (still tagged known),
  so it passed for nothing. The fix itself is commit 9bdca28. Found through
  contrast.skew_segments: after the #21 fix that input built, with closure 0.97
  and rings of exactly 65. Rings of 1000 corners run in valence.ring now (294
  checks, none skipped).

- #21: a third tier of exact cells, huge_planes = 16000, huge_vertices = 32000,
  on the host for the few cells that overflow big ones (16-bit indices, 1.9 MB
  each, on the heap). Options: big_planes up to 255 (the byte limit): 3 cells of
  k = 800 might fit, 20 of k = 2000 perhaps not, and every device thread of the
  big pass pays the stack; a host tier: nothing on the common path (one more
  reduce only after an overflow), grid and pools copied to the host and back
  once in that rare path. Chosen: the host tier.
  Cost that stays: every shell cell clips all k particles of the shell (the
  void's interior is in reach), so the start is O(k^2) in the particles around
  one void: k = 2000 about 1 s, k = 20000 with a hub 129 s (8 CPU threads).
  Old behaviour, now reachable. A limit, not a slowdown.

- #13 moved to luna-cpp (#6) with the hydro, after the guard was committed here.
  Commit c0b405b holds it and its hydro_test cases, for the port; its one line in
  mesh_cases_test.cpp is the #21 tag and has nothing to do with it. Without the
  guard the static lattice cases gave NaN, until the run aborted in the fifth.

- Found while on it, not an issue: tet_repair::repair stops after 1000 rounds, and
  with nothing listed as stuck at that moment update() returned a mesh with failing
  triangles. Lowering the limit shows it (fallback.out_of_rounds, displacement 3e-3
  spacings, 1 and 2 rounds). Stats::unfinished sends such a step to a rebuild.

- #22: snap is the largest k with k * step <= x, the product rounded as positions
  are. Before, in 200 boxes of random side, 221037 of 4e6 positions k * step moved
  (5.5 %; the issue's 5 of 12288 were at side 1e-3 alone, which has its own mantissa).
  Option not taken: a tolerance on x / step would move off-grid inputs near a grid
  line in every box, the unit cube too, and change bits.

- Hashes, job 31924558 (main against mesh-fixes with the snap change): all five
  paths differ, x the same on the static paths, volume not. hydro_sedov's box is
  0.05, a long mantissa, where main's own snap moves positions k * step (#22). To
  isolate it: the same code with the old snap, build-meshfix/nosnap.

- #24: the sum of the cell volumes comes out of tet_mesh::faces (a reduction in
  place of a parallel_for, no new kernel), compared with the box to 1e-6. Not
  reachable on OpenMP, where that case ends at the round limit: only an A100 loop
  shows it. Options: exact sum of |det| per tetrahedron (an exact covering test,
  about one screening kernel per step, not tried); the cell sum, chosen.

- #23: rule stuck * 200000 <= n from the issue's 50 ms per entry and 0.24 us per
  cell for a rebuild. No timer: path and bits stay the same run to run. A
  deterministic rule stays a guess at a cost that varies 400 times by input.
