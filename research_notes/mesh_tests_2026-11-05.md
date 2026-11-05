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
| #23 | library cost | the host fallback costs about 50 ms per stuck entry, not 0.12 ms; `cheapest` picks it: steps of 36 s at 128^3 |
| #24 | library bug, silent | after a step of 0.6 box the device repair can end with nothing stuck on a mesh of 500 box volumes: 27 of 27 such steps wrong, 1.8 % of steps |
| #4 (comment) | limit | at 256^3 a rebuild does not fit beside the mesh on one A100: exception |
| #21 (comment) | limit | half of 512k+ particles in a small ball: the start refuses, 2 to 21 cells overflow |

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

## Distorting flows on one A100

`distortion_test`, d = 0.3 initial spacings per step at speed 1, no regularisation.
Rates are M cells/s of the whole update (repair, geometry, cells, fallbacks).
Smooth reference (`voronoi_bench step`, the same definition): 46 to 64 at 128^3 and
72 to 82 at 256^3 over the first steps of jittered points, the rate following
flips per step. Tables: `python3 scripts/flow_table.py --stages <log>`.

128^3 (jobs 31923398, 31923401, 31924250; 45 to 120 s per flow):

| Flow | Steps, t | Rate early -> late | Flips/step | Stuck, rebuilds | Aspect | Max faces | Reach | Ends |
|---|---|---|---|---|---|---|---|---|
| shear | 800, 1.9 | 109 -> 53 | 0.11M | 0 | 36 | 31 | 0.02 | limit |
| sawtooth | 800, 1.9 | 132 -> 50 | 0.11M | 0 | 22 | 28 | 0.02 | limit |
| vortex | 1169, 2.7 | 147 -> 84 | 0.05M | 0 | 33 | 38 | 0.03 | limit |
| taylor_green | 800, 1.9 | 77 -> 47 | 0.14M | 12 | 143 | 61 | 0.07 | limit |
| turbulence | 800, 1.9 | 47 -> 36 | 0.31M | 18 | 56 | 38 | 0.03 | limit |
| thin | 800, 1.9 | 60 -> 40 | 0.2M | 18 | 7.5e5 | 77 | 0.25 | see below |
| collapse | 130, 0.3 | 181, worst 0.2 | 0.01M | 4049 | 23 | 101 | 0.11 | host path |
| noh | 10, 0.02 | 99, worst 0.06 | 0.01M | 1877 | 2 | 26 | 0.02 | host path |
| pancake | 70, 0.16 | 4, rebuilds | 1.9M | 8e7, 69 | 1393 | 46 | 0.09 | limit |
| counter | 85, 0.2 | 4.2 | 7.2M | 28 | 2 | 26 | 0.02 | limit |
| streams | 55, 0.13 | 1.7, rebuilds | 3.0M | 7e7, 52 | 2305 | 38 | 0.24 | limit |
| collapse_wide | 130, 0.3 | 113, worst 0.3 | 0.09M | 4060 | 26 | 124 | 0.15 | refused, step 131 |
| expansion | 0 | | | | | | | refused at the start (#21) |

256^3 (job 31924251; 240 s per flow, no rebuild check, it does not fit):

| Flow | Steps, t | Rate by quarter | Flips/step | Stuck | Aspect | Max faces | Reach |
|---|---|---|---|---|---|---|---|
| thin | 816, 0.96 | 127, 58, 47, 46 | 1.05M | 6 | 454 | 59 | 0.23 |
| shear | 1099, 1.29 | 117, 80, 67, 62 | 0.44M | 0 | 24 | 29 | 0.01 |

Turbulence went 600 steps (median 52), then the rebuild check ran out of device
memory (the test's check, not the mesh). Pancake at 256^3 ends at step 2 in a
rebuild that does not fit (#4).

Memory: 1978 B/cell at rest, to 2021 in turbulence, 2294 in counter, on
the mesh arrays, no creep over 800 steps; faces per cell fall from 7.95 to 7.3
in thin, nothing grows. Corners per face stayed at 27 or below, against the limit
of #18 at 64.

What degrades the rate:
- Shear at constant flips per step: 117 to 62 at 256^3. The work per step does
  not change. Inference, not measured: lost memory locality as particles leave
  their sorted order.
- Flips: counter at 7.2M flips per step, 3.4 per cell, 4.2 M cells/s.
- Rebuilds: pancake, streams from step 2, one rebuild per step, 2 to 4 M cells/s.
- The host path: collapse, noh, a thousand times slower on the steps it runs (#23).
  Stuck entries start at t about 0.25 in collapse, step 3 in noh, step 2 in sheets.

Correctness: 29 full oracle passes at 128^3 and 256^3 at low strain (steps 40, 80:
jobs 31922269, 31922270, 31922271, 31922272), and at 262144 on turbulence, thin and
pancake on the last of up to 800 steps (31923406). Rebuild checks passed on
every flow they reached at 128^3, thin excepted: at step 800 the check reported
6307 mesh faces the clipping lacks, up to 18.5 spacings^2, with volumes within 8e-10.
At 64^3 at the same strain (job 31925079) brute-force clipping sides with
the mesh on all 8 largest such faces, to every printed digit, and the reference has
125 cells `degenerate` with volume 0: the plain clipping is wrong, not the mesh, a limit
of this reference past aspect ratios of about 1e5 (`mesh_check::arbitrate`). The job
that would have arbitrated at 128^3 itself ran out of time (31924249).

Not checked: whether `collapse` carried a dropped particle (#17) in its last steps
before the refusal at step 131, when neighbours were 3 grid steps apart; those
steps went through the host path and had no oracle.

Drift to the centroids, in this driver (the library layer has none), 128^3, chi 0.1,
0.5 per step (job 31923405):

| Flow | Rate, no drift -> 0.1 -> 0.5 | Aspect | Faces per cell at the end | Stuck |
|---|---|---|---|---|
| shear | 53 -> 60 -> 58 | 36 -> 2.6 -> 2.4 | 7.81 -> 7.22 -> 7.16 | 0 |
| turbulence | 36 -> 39 -> 40 | 56 -> 4.3 -> 2.8 | 7.78 -> 7.51 -> 7.32 | 18 -> 0 -> 0 |
| thin | 40 -> 52 -> 54 | 7.5e5 -> 4.6 -> 2.8 | 7.30 -> 7.27 -> 7.17 | 18 -> 0 -> 0 |
| pancake | 2.5 -> 3.1 -> 3.7 | 1393 -> 281 -> 381 | | still rebuilds every step |
| counter | 4.2 -> 4.3 -> 4.4 | 2.3 -> 2.2 -> 2.2 | | 28 -> 0 -> 0 |

The drift alone takes the aspect ratios of winding flows from hundreds of thousands
to single digits and stops the stuck entries, at no cost to the rate. It does not
help where the flow itself drives one spacing per step across a sheet.
