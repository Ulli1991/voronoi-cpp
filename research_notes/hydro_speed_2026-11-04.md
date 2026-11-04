# Hydro speed, 2026-11-02 to 11-04

Current default (rows, split sweep, time step per face): hydro 152 M cell
updates/s static, 140 moving, whole moving step 73, 1.21 G face updates/s
(job 31879405). The table below is the state at the first merge; the sweep
section further down took it to these numbers.

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

## The sweep: splits, gathers, index layouts (128^3, same job against its baseline)

Different GPUs of the node ran at different speeds (fluxes 650 or 564 M/s on the
same binary): only numbers of one job are compared. Rates: hydro-only static /
hydro-only moving / whole moving step, M cell updates/s, and the change against
the job's baseline. ncu: Sweep kernel, static, `--clock-control none`.

What the fused sweep was (ncu): 12.8 ms, 164 registers, 12 % active warps, 62 %
long-scoreboard, FP64 pipe 21 %, DRAM 1651 B/cell, L2 4437 B/cell. Not DRAM
bandwidth (17 % of peak): each memory wait stalls the few warps there are.
All loads are 64-bit (SASS: LDG.E.64 and 32, no 128): a double, a load.

| variant | what | change | ncu | bits |
|---|---|---|---|---|
| s1 | time step + w, then gradients | 110/99/57 -> 126/96/56 | fit 152 regs, 18 % warps; dt 56 regs, 50 %, 85 % stall | base |
| s2 | + limiter its own kernel | 110/99/57 -> 112/88/53 | fit 5.2 ms + limit 5.0 ms; DRAM 3287 B/cell; stall 70-85 % | base |
| s3 | + Cholesky factor pass | 110/99/57 -> 97/79/50 | worse | base |
| s4 | gradients + w, then dt reading w | 123/111/64 -> 140/128/70 | fit 7.1 ms, 18 %; dt 2.3 ms, 963 B/cell, 87 % stall | base |
| m1 | s4, dt as cell + face reductions | 126/115/61 -> 135/123/64 | FaceStep 0.7 ms, 228 B/cell; CellStep 0.1 ms | base |
| m2 | m1, lo/hi on the stack (volatile) | 126/115/61 -> 127/111/60 | 128 regs, 24 % warps, but 9.2 ms: L2 11858 B/cell, 77 % stall | base |
| m3 | m1, surface in the main loop | 150/138/72.5 -> 152/140/73 | 148 regs; L2 5851 -> 4452 B/cell | base |
| g1 | x and centroid in the primitive row | 110/99/57 -> 106/98/56 | sweep 12.5 -> 12.4 ms, pass +0.7 ms | base |
| g2 | g1 + list (f, j) | 110/99/57 -> 87/82/51 | sweep 11.6 ms; built here at 6.2 ms a step | base |
| r1 | 64-byte face rows (area, |A|, centroid, ids) | 110/99/57 -> 84/79/50 | sweep -6 %; update 800 -> 421; pass 3.7 ms | base |
| w0 | 32-byte aligned rows (2 x LDG.128) | 123/111/64 -> 122/111/64 | no change | base |

Registers: warps follow them in steps (164: 12 %, 152: 18 %, 128: 24 %, 80: 34 %,
56: 50 %, 40: 70 %), but m2 shows they are not the goal by themselves. Row packing
(g1, r1, w0) buys nothing: the waits are not bandwidth; dependent loads and
repeated loads of the same data are. What paid: not reading things twice. s2 and
s3 split the loops that share their loads and doubled DRAM. s4 and m1 split what
shares little, and the time step turned out to be a global minimum of a quantity
symmetric per face: half the pairs, sequential face reads, 963 -> 295 B/cell.

Rejected, reverted (experiments in build-hydrospeed/exp, never on the branch):
s2, s3, m2, g1, g2, r1, w0.

## Index layouts (each against s4, 139/128/70, with the same list)

| variant | result | layout numbers | bits |
|---|---|---|---|
| (a) SELL-32 lists, no sorting | 139/127/69, -0.1 to -0.5 %; update 798 -> 978, sweep 8.96 -> 9.10 ms, list pass 0.37 ms | padding 4.4 entries, 17.5 B/cell; sigma 256: 0.67 entries; 4096: 0.06 (needs a cell permutation, not built) | base |
| (b) 16-bit offsets with 32-bit escape | 125/116/66, -10 %; update 798 -> 712 | escape: neighbour - owner 4.5 %; list entry - first entry 8-9 %; saves 2 B per entry and face | base |
| (c) shared-memory tiles, fit only, static | T = 128: 96 (-29 %); T = 256: 73 (-46 %) against m1's 135 | outside refs 27 % (T = 256), 38 % (T = 64), 17 % (1024); occupancy 45 %, 32 %, 58 %; 27 KB of rows per tile at T = 256 | base |
| (d) face-major, atomic adds | 125/116/62, -1 % against s4 | not reproducible: every run other bits | no |
| (d) face-major, colours (greedy, host, static) | 66, -48 % | reproducible, not main's bits; one kernel per colour; a moving mesh would recolour each step | no |

(a) rebuilt the list every step in a hydro pass (0.37 ms); kept by the mesh instead
it would be free, then +22 % on update, ~+3.5 % hydro static. Prototype only.

## Adopted: gradients + w, then the time step per cell and per face, surface in the main loop

Same job 31879405, the build before this round -> now:

| | hydro static | hydro moving | whole moving | G face/s | sweep | B/cell |
|---|---|---|---|---|---|---|
| before | 122.7 | 111.3 | 64.5 | 0.98 | 186.8 | 2426 |
| now | 152.2 | 140.0 | 73.2 | 1.21 | 264.6 | 2434 |

Fluxes 650, update 798 (static); moving: sweep 246, fluxes 550, mesh 291, cells
332, mesh + cells ~155: unchanged. ncu, static, sweep pass only: 7.07 ms, 148
registers, 17 % warps, 66 % long-scoreboard, FP64 27 %, DRAM 1045 B/cell (time step
kernels +295), L2 4456 B/cell; before 11.2 ms, 164, 12 %, 63 %, 20 %, 1592, 4377.
+8 B/cell (the cell's radius in the primitive row).

Bits: all equal main's U, x, volume: the six paths of job 31878302 and 64^3
`uniform` lattices static, moving and `--limiter=cell` (job 31879404). Reproducible
run to run (checksums of job 31879405). Whether the uniform lattice took the
tiny-face path was not counted; the path is equal to the old code by construction.

Same job 31879471, a slower GPU (fluxes 568), default -> mixed -> low, hydro static
137 / 151 / 163, hydro moving 125 / 139 / 148, whole moving 64 / 67 / 69; main in that
job 99 / 82 / 51. The gain of mixed over double fell from +35 % to +10 %, consistent with: much of it
was the registers of the fused double sweep (mixed: 108 registers, 23 % warps,
6.0 ms, DRAM 827 B/cell, job 31879474).

Open: an owner bit in the list would spare update its 16 scattered owner loads
(not measured); SELL lists are worth it only if the mesh writes them.
