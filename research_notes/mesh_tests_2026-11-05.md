# 3D special-case tests of the mesh, 2026-11-05 (issue #15)

Branch `mesh-tests`. Build and logs: `/raven/u/uli/voronoi-cpp/build-meshtests/`
(`omp`, `cuda`, `a100/` for job logs, `sweep*/` per-case logs of the OpenMP runs).

## What was added

- `src/test/oracle.hpp`: an oracle that shares no code with the library: own
  periodic differences, orientation and in-sphere in 128-bit arithmetic, every
  periodic image, no tie-break, cells by half-space clipping in long double.
- `src/test/mesh_check.hpp`: a `voronoi::Mesh` against it, and against a fresh
  build of the same positions (the tetrahedra must be equal as sets).
- `src/test/mesh_cases_test.cpp`: named cases, ten ctest groups; a case of an
  open problem runs with `--known`; `MESH_CASES_FULL=1` runs the whole matrix.
- `src/test/distortion.hpp`, `distortion_test.cpp`: prescribed flows, no hydro.
- `scripts/mesh_tests_a100.sh`: cases, reproducibility, flows, drift, oracle.

## Failures

| Issue | Class | What |
|---|---|---|
| #17 | library bug, silent | a coincident pair in a step drops a particle on the host path: default fallback from n about 9000 |
| #18 | library bug, silent | a face of more than 64 corners: wrong area, volume, closure, at the start and in the repair |
| #19 | library bug | segfault on the update after a refused update |
| #20 | library bug | segfault on zero particles (abort on CUDA) |
| #21 | limit, poor message | valid shells of 800+ particles refused with no cause given |
| #22 | library bug, small | snap not idempotent at side 1e-3: 5 of 12288 coordinates |

Test bugs found on the way, for the record: `degenerate.mover` walked onto a
lattice site; `support::Flow` is compressive, so the first long run ended in
a correct refusal; `box.edges` put two particles on one wrapped position; the
first `predicates.wide` used constants with a period of 500 (11 real
duplicates); the oracle first counted a refused fresh build as a wrong mesh;
`box.scale` re-snapped `positions()` (see #22).

A warning for whoever reads the sweeps: an early summary of sweep 2 was read
before the box cases had finished, which hid `box.edges` and `box.scale`.

## Results that passed

- Degeneracy in motion: sc, bcc, fcc rigid, sheared by layers through recurring
  exact lattices, smooth shear, noise of 1 to 1024 grid steps per step, sliding
  lattices, octahedra pulsing through exact cospherical: all exact.
- Predicates: orient and in-sphere equal the oracle's integers on real point sets.
- Bits: Serial, OpenMP at 1 and 8 threads and CUDA give the same hashes of
  faces, volumes and centroids at every step of 4 inputs (84 hashes, all
  distinct, 21 steps each). There and back gives the same tetrahedra and bits.
- Zero-area faces (#13): 16384 of 28672 on sc, 2048 of 14336 on fcc, 0 on bcc;
  every centroid finite. The NaN of #13 is in the hydro, not in the mesh.
- Precision floor: the relative volume of a cell h grid steps wide rounds at
  about 4e-8 / h (clumps 512 to 2^20 steps wide, 7.9e-10 to 7.3e-13).
- Limits by n: uniform refused at 216, built at 512; jittered and lattice
  refused at 125, built at 216. Past reach 3/8 a repaired mesh stays exact; it
  ends at the first stuck step, whose rebuild refuses.
