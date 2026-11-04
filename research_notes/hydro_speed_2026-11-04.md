# Hydro speed, 2026-11-02 to 11-04

128^3 Sedov on one A100 40 GB (Raven), jittered, Morton order, 40 steps, medians.
M cell updates/s: hydro-only static / hydro-only moving / whole moving step;
G face updates/s: hydro-only static (faces stored once). Logs:
build-hydrospeed/logs, jobs 31802042 (bits), 31802043 (speed), 31802044-46 (ncu).

| build | hydro static | hydro moving | whole moving | G face/s |
|---|---|---|---|---|
| main 3fd11aa | 107 | 88 | 56 | 0.85 |
| hydro-speed, default (double) | 123 | 111 | 64 | 0.98 |
| `--precision=mixed` | 166 | 154 | 76 | 1.32 |
| `--precision=low` | 179 | 162 | 78 | 1.42 |

Phases of the default (M cells/s): sweep 187, fluxes 649, update 797; moving:
sweep 169, fluxes 547, mesh 288, cells 333. Sweep = primitives + time step +
point velocities + gradients. What changed from main: per-cell and per-face
data in rows (LayoutRight); one sweep over each cell's face list instead of
three kernels; a primitives pass (W and c once per cell, step in place);
precision policy (src/hydro/precision.hpp). Morton order was already the default:
+13 % against `--input-order` (sweep DRAM 1432 vs 3174 B/cell).

Stored gradient weights (`weight32`): sweep 186 -> 207, but `cells` 332 -> 143
on a moving mesh, whole moving step 64 -> 51: off by default.

## Bits of the default


Six paths at 128^3, 50 steps, final U, x, volume by sha256 (job 31878302):
static, moving, order 0 (per cell, per face), per-cell moving, HLL moving: all
equal main's. Reduced-precision policies are not main's bits by construction.

Order 0 differed for a while (job 31802042, U cab0b2432e74fb5e against main's
4992580c6319900f; x and volumes equal). `--order=0` defaults to the per-cell
loop (src/bench/hydro_sedov.cpp, `o.loop`). The cause: after the primitives pass
FaceFlux::side returned W as loads from the primitive rows, and at order 0 this
W goes straight into Riemann::flux (src/hydro/euler.hpp:254-261). Main computes
W in the kernel; the products of `primitive()` (euler.hpp:104, 108: v = U
inv_rho, P = (gamma - 1) e) can then be contracted into the sums of the Riemann
solve. The individual FMA sites were not isolated. Evidence:
- W inline again (Q as input, per cell into next_ as main): main's hash at 2,
  3, 5, 50 steps (32^3) and at 128^3, per cell and per face (job 31878102);
- compiled with `-fmad=false`, W from loads and W inline give the same hash
  at 5 and 50 steps (job 31878302), so the difference is the contraction;
- order 1 hashes were equal throughout: there W reaches the solver only through
  the half step and extrapolation, and a select between that and W.
Size while it differed (50 steps, 128^3; arrays of 5 x n): rounding. Density:
0.03 % of the cells, at most 8 ulp, largest relative 1.2e-15. Energy: 0.09 %,
at most 19 ulp, 2.7e-15. Momentum components: 32-34 % of the cells, largest
difference 6e-16 of the largest |momentum|; relatively up to 3e-7 only in cells
at rest whose momentum is itself rounding noise (~1e-15). Any variable: 69 % of
the cells. Sums over the box: 0 for mass and energy, ~1e-14 for momentum.

Measured on the CPU (OpenMP, gcc): U hashes equal on all five paths before
the fix too (runs/cpu), so this was an nvcc-only difference.

Fix: order 0 computes W from Q in FaceFlux::side (src/hydro/finite_volume.hpp),
per cell out of place (next_, 40 B/cell, allocated only for order 0 per cell).
Order 1, the fast path, is unchanged.

Checks at the merge: main merged by a merge commit (README and notes only);
ctest 7/7 on CPU (gcc 15, OpenMP) and A100 (job 31878773); warnings 0 in both
builds; clang-tidy 21 (scripts/lint_cpp.sh) clean.
