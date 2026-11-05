# Mesh library fixes, 2026-11-05 (branch mesh-fixes)

Bugs found by the mesh tests (branch mesh-tests). Build dir and logs:
build-meshfix/. Before-binaries: build-meshfix/before/.

## Findings as they come

- #21: exact clipping of a shell (k = 800, 2000; 512 background) ends with
  only plane_overflow cells (3 and 20, at big_planes = 128). Every
  `degenerate` of the issue comes from the rounded first pass; the exact pass
  recovers all of them. Measured with `shell_voronoi::construct(..., exact=true)`.
