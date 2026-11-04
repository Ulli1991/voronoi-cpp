# Accuracy score for hydro codes: exploration on real data, 2026-11-04

The owner's proposal: score = 1 / (T L1^((d+1)/p)), p fixed per test. Tested on
LUNA and voronoi-cpp (main c742e83, static and moving mesh), one A100 40 GB,
FP64 unless stated, Pallas kernels for LUNA at second order, the XLA step at
order 0. Tags: **[M]** measured, **[D]** derived arithmetically from measured
numbers, **[I]** inference, not tested.

Scratch: `build-metric/` (scripts `driver/`, `analysis/`, logs `logs/`, the
drivers are not in `src/`). States: `/ptmp/uli/metric/<job>/` (27 GB, can go).

## Answer in brief

1. The score is not flat on Sedov with any p near 1: over 128^3 to 448^3 LUNA
   falls 11.5x at p = 1 **[M/D]**. The assumption T ~ n^(d+1) fails: steps
   grow as n^1.9 on Eulerian meshes (hot cavity), so T ~ n^4.6-4.9. The right
   exponent on L1 is about 7, not 4 **[D]**.
2. It is flat where its assumptions hold: smooth wave at p = 2; Sod in L1(P)
   at p = 1, both in cell updates. Max/min 1.19-1.28 **[D]**. A GPU spoils T
   at small n: per-step latency dominates below about 1e5 cells.
3. At equal error the score ratio equals the T ratio for every p: the proposal
   is exact there and p does nothing. At unequal error the moving/LUNA order
   flips between p = 0.85 and p = 1 **[D]**.
4. The norm matters more than p. Moving 256^3 against LUNA at equal error:
   1.0x to 9x, depending on the norm **[D]**.
5. A first-order moving mesh beats the second-order one on the Sedov score at
   192^3 **[M/D]**, and HLL beats HLLC 5x on Sedov for a reason unrelated to
   shock quality **[M/D]**.
6. Recommendation: T(eps0), GPU-seconds to a card's error, interpolated between
   two of the code's own runs. The score is that number's single-run estimate,
   with the exponent on the card and a limit on how far from eps0 it may be used.
   Publish T, L1, M with it (owner's decision).

## Data

Existing: `research_notes/hydro_stage3_2026-10-31.md:147-157` and final states
in `/ptmp/uli/voronoi-hydro/`, `build-hydro/runs6/`. New jobs (A100):

| job | node | what |
|---|---|---|
| 31879151 | ravg1156 | LUNA 1D Sod and sound wave, FP64 and FP32 |
| 31879152 | ravg1156 | LUNA 3D tube, wave, shear; ours tube (timed out before wave, shear) |
| 31879806 | ravg1156 | ours wave, shear (same node as 31879152, not the same job) |
| 31879294 | ravg1172 | Sedov: LUNA 64-256, ours moving 64-192, static 64-192; moving 256 ran out of memory |
| 31879295 | ravg1172 | LUNA Sedov variants; ours order 0 (timed out before precision and moving 192) |
| 31879807 | ravg1198 | ours moving 256, per-cell loop; mixed precision ran out of memory |
| 31879808 | ravg1198 | ours moving, mixed, low, HLL, order 0 at 192 |
| 31879296 | ravg1183 | capacity: LUNA to 640^3, ours to 268^3, 40 steps |
| 31880401, 31880402 | ravg1198, ravg1002 | LUNA and ours calibration |

Jobs 31879809/10/11 and 31880346/47/48: norm analysis (CPU only, on GPU nodes).
Two norm runs did not finish: LUNA 64^3 and 96^3 (`l64`, `l96`) are not in
`norms_luna.json`, so LUNA's norm tables start at 128^3. LUNA 320-448 are the
older jobs of the stage 3 note, so they are not same-node with the rest. Sedov
T is the time-stepping loop only: no snapshots, no LUNA compile (see Gaming).
Ours adds the mesh build. Smooth-wave and tube T for LUNA has compile spikes.

Sedov, L1(rho) volume-weighted, r < 1.5 R_s, cell-sampled
(`scripts/hydro_sedov_compare.py:102-104`), units rho0. M: device peak
(XLA `peak_bytes_in_use`, Kokkos peak).

| series | n | steps | T (s) | M (GB) | L1 rho | local p | local T exponent |
|---|---|---|---|---|---|---|---|
| LUNA | 128 | 1075 | 3.19 | 0.40 | 0.0766 | | |
| LUNA | 192 | 2329 | 16.8 | 1.17 | 0.0595 | 0.62 | 4.09 |
| LUNA | 256 | 4031 | 65.0 | 2.82 | 0.0495 | 0.64 | 4.71 |
| LUNA | 320 | 6167 | 188 | ~5.4 | 0.0425 | 0.68 | 4.77 |
| LUNA | 384 | 8727 | 460 | ~9.3 | 0.0376 | 0.67 | 4.90 |
| LUNA | 448 | 11703 | 933 | 14.6 | 0.0341 | 0.63 | 4.59 |
| moving | 64 | 138 | 0.99 | 0.65 | 0.0913 | | |
| moving | 96 | 235 | 3.84 | 2.19 | 0.0711 | 0.62 | 3.34 |
| moving | 128 | 348 | 12.2 | 5.19 | 0.0580 | 0.71 | 4.00 |
| moving | 192 | 637 | 70.2 | 17.6 | 0.0429 | 0.74 | 4.33 |
| moving, per-cell loop | 256 | 918 | 351 | 37.4 | 0.0341 | 0.80 | 5.60 (the loop changes here) |
| static | 64 | 438 | 1.21 | 0.64 | 0.1079 | | |
| static | 96 | 924 | 6.82 | 2.15 | 0.0881 | 0.50 | 4.26 |
| static | 128 | 1636 | 28.4 | 5.09 | 0.0750 | 0.56 | 4.96 |
| static | 192 | 3549 | 207 | 17.2 | 0.0582 | 0.63 | 4.90 |

LUNA 320, 384: M interpolated at 165 B/cell, not measured.

## Q1. Is the score flat?

Max/min of the score, n >= 128 (`build-metric/analysis/sedov_score.py`) **[D]**:

| exponent on L1 | LUNA (128-448) | moving (128-256) | static (128-192) |
|---|---|---|---|
| 4 (p = 1, the proposal) | 11.5 | 3.5 | 2.6 |
| 5 (p = 0.8) | 5.1 | 2.0 | 2.1 |
| 6.15 (p = 0.65) | 2.0 | 1.2 | 1.5 |
| fitted per code (7.0 / 6.3 / 7.8) | 1.12 | 1.18 | ~1.0 (3 points) |

No flat range appears at any n reached: local p and local T exponent do not
settle by 448^3 (R_s = 120 cells) **[M]**.

Why, measured **[M]**:
- Steps. LUNA and static: steps ~ n^1.91; moving: n^1.41.
- Smallest density falls as n^-1.78 on LUNA (1.45e-3 to 1.55e-4 rho0, 128 to
  448). Then c_max ~ n^0.89 and steps ~ n^1.89, which matches **[D]**. So the
  hot, emptied centre sets dt on Eulerian meshes. On a moving mesh the cells move
  with the gas, so |v - w| is small there (stage 3 note).
- Order. 89-93 % of L1 comes from the shell, 0.85-1.1 R_s, at every resolution.
  The local order is 0.62-0.68, not 1, up to 448^3.
- Why p stays low is **[I]**: the post-shock spike is not resolved, and the
  blast is put into 8 cells, so the initial radius shrinks with n and each n
  is a slightly different problem.

1D LUNA Sod, 64 to 65536 **[M/D]**: in raw T the score rises 100x at p = 1,
because T goes as n, not n^2: the step stays at 0.7-1.0 ms from n = 2048 to
65536 (`median_perf` 96 M cells/s at 65536). Below about 1e5 cells a GPU is in
its latency regime, with occasional compile spikes (256 cells took 0.47 s
against 0.12 s). Replacing T by work W = n x steps:

| test, 1D LUNA minmod | W L1^x | max/min | range |
|---|---|---|---|
| Sod, L1(P), x = 2 (p = 1) | 0.153-0.181 | 1.19 | 64-65536 |
| Sod, L1(rho), x = 2 | 0.24-5.5 | 22 | 64-65536, drifts throughout |
| Sod, L1(rho), x = 2.67 (p = 0.75) | 5.97e-3 - 6.79e-3 | 1.14 | 2048-65536 (1.8 from 64) |

3D wave, W L1^2 (p = 2): LUNA MC 0.073-0.093 (1.28, 32-128), ours static
3.8-4.8 (1.28), moving 2.7-4.8 (1.8, order rising to 2.5 at 128 **[M]**).
In raw T at n <= 128, LUNA is not flat (latency); ours static is 1.54 from 64
to 128 and 1.13 from 96 to 128.

## Q2. What p? Fixed p? Steps? Non-integer order?

Measured order, L1, global fit **[M]**:

| test | LUNA | ours static | ours moving |
|---|---|---|---|
| Sedov rho (n >= 128) | 0.65 | 0.63 | 0.76 |
| Sedov v_r | 0.80 | | 0.97 |
| Sedov P | 0.67 | | 0.74 |
| Sod rho (MC, 3D tube, 64-2048) | 0.85-0.93 | 0.84-0.91 | 0.80-0.97 |
| Sod P | 0.96-1.05 | 0.95-1.04 | 0.96-1.03 |
| Sod rho, order 0, 1D | 0.63-0.67 | | |
| wave, minmod | 1.88 (1.96, 1D to 4096) | | |
| wave, MC / ours | 2.02-2.14 | 2.03-2.17 | 2.04-2.55 |
| wave, order 0, 1D | 1.00 | | |
| shear, MC / ours | 3.00 | 2.16-2.28 | 2.22-2.28 |

Sod by region (density, 1D minmod to 65536) **[M]**: rarefaction 1.00, shock
~1, contact 0.664 from n = 1024 onwards. The contact takes over as n grows, so
overall p in L1(rho) falls from 0.83 to 0.71 and has no constant value; the
theory value for a second-order contact is 2/3. L1(u) and L1(P) contain no
contact: order 1. So Sod in pressure has a real p, Sod in density does not.
Contact width in cells grows on LUNA and ours static (minmod 4 to 15, MC 2 to 8,
ours static 2 to 8), and stays about 2 on the moving mesh **[M]**.

Is a fixed p defensible? For ranking, only if every code is compared near the
same error. Sensitivity, S(moving)/S(LUNA) at the same n **[D]**:

| p | 128^3 | 192^3 | 256^3 |
|---|---|---|---|
| 0.45 | 3.10 | 4.37 | 5.01 |
| 0.65 | 1.45 | 1.79 | 1.81 |
| 0.85 | 0.97 | 1.11 | 1.06 |
| 1.0 | 0.80 | 0.88 | 0.82 |
| 1.2 | 0.66 | 0.71 | 0.64 |

The winner changes within p = 1 +- 0.2. Time to equal error says the moving
mesh is ahead at all of these, so p = 1 ranks them wrong. Static against
LUNA is stable (0.12-0.14 at 128^3), since their L1 nearly agree. Moving 256^3
against LUNA 448^3 (equal error) is 2.62-2.64 for every p: p is irrelevant.

Steps: yes, T should be read as T = W / R, with W = cells x steps and R
updates/s, because W is what scales and it carries no node noise. Publish
steps with every score. The score needs d + z instead of d + 1, with
steps ~ n^z: z = 1.9 Eulerian, 1.4 moving on Sedov, 1 on the tube and wave.
Non-integer order: then x = (d + z) / p is a test constant measured on a
reference code, fine as a number on a card. It does not have to be derived.

The reference form W_ref = C / eps^((d+1)/p) does not fit: C from LUNA 448^3
at x = 4 predicts 18x too much work at 128^3 **[D]**. A table of W_ref(eps)
measured on one code works and gives (b) below.

## Q3. Does it rank sensibly?

Time to equal L1(rho), T(LUNA)/T(moving) by log-log interpolation **[D]**,
new runs:

| eps | LUNA T | moving T | ratio | static T |
|---|---|---|---|---|
| 0.075 | 3.65 | 2.88 | 1.27 | 28.4 (LUNA 7.8x faster) |
| 0.058 | 20.2 | 12.1 | 1.66 | |
| 0.050 | 59.9 | 28.8 | 2.08 | |
| 0.045 | 125 | 53.1 | 2.36 | |
| 0.0341 | 921 | 351 | 2.62 | |

The owner's check: at p = 1, 288 s against 933 s gives 3.24, the stated 3.2x,
by construction. That 288 s is the per-face loop of 2026-10-31. Today's main
needs the per-cell loop at 256^3, 351 s, so today's figure is 2.6x (see Memory).
The ratio grows as eps falls, so a single number does not describe the pair.

LUNA variants, same n **[M/D]**:

| variant | n | T | L1 rho | S / S(HLLC minmod), p = 1 | same, p = 0.65 |
|---|---|---|---|---|---|
| HLLC MC | 256 | 71.4 | 0.0419 | 1.78 | 2.54 |
| HLLC MC | 128 | 3.41 | 0.0668 | 1.61 | 2.16 |
| HLLC van Leer | 192 | 23.9 | 0.0537 | 1.05 (its run was at 688 M cells/s, maybe the slow mode below) | |
| HLL minmod | 256 | 17.3 | 0.0460 | 5.03 | 5.89 |
| HLL minmod | 128 | 0.85 | 0.0708 | 5.15 | 6.11 |
| ORDER=0 HLLC | 256 | 53.3 | 0.0599 | 0.57 | 0.38 |
| ORDER=0 HLLC | 128 | 2.23 | 0.1039 | 0.42 | 0.22 |
| CFL 0.4 | 192 | 32.8 | 0.0597 | 0.5 | |

As expected: MC beats minmod, order 0 loses, halving CFL only costs T (L1 does
not move). Not as expected: HLL wins 5x. It takes 1215 steps against 4031,
because its centre stays dense (min rho 5.9e-3 against 4.3e-4 rho0), and L1
hardly sees the centre: centre L1 is equal at 256^3 (0.0063, 0.0066) and worse
at 128^3 (0.0187, 0.0058). So this is the dt mechanism of Q1 and not a better
solver. On Sod HLL is about 9 % worse in L1(rho), MC against minmod about 2x
better at n = 16384 (T unchanged) **[M]**.

## Q4. How to break it

| attack | result | verdict |
|---|---|---|
| First-order code, Sedov | moving order 0 at 192^3: L1 0.0425, T 64.7 s; order 1: 0.0429, 70.2 s. Order 0 wins by 12 %. L1 P 0.0092 against 0.0110; L1 v_r 0.0174 against 0.0124 | **Breaks Sedov alone** |
| First-order code, wave (p = 2) | LUNA at 4096: L1 6.1e-4 against 1.5e-6, same T | The smooth test catches it, decisively |
| Lower precision, Sedov | LUNA FP32: same L1 to 4 digits, T 2.2x less at 256^3, M 1.68 against 2.82 GB. Ours mixed/low: same L1, T 1.18x less, memory slightly up (2543 against 2480 B/cell) | Legitimate gain, must be stated |
| Lower precision, wave | FP32 at amp 1e-4: L1(P) stuck at 1.7e-3 from n = 256, L1(rho) grows with n to 6.7e-2 of the amplitude. Sod sees nothing (4 digits equal) | The wave exposes precision, shocks do not |
| Outside the asymptotic range | At p = 1 on Sedov a coarse run always looks better: moving 64^3 (S = 14500) is 18x LUNA 448^3 (S = 796) | **Breaks it**: coarse runs win |
| Mass-weighted L1 | doubles L1 (0.0766 to 0.156) and order drops to 0.53 | Changes the ranking, table below |
| Region | Whole box against r < 1.5 R_s: scale factor only, same order, same ratios | Harmless but the number changes 3.7x |
| Variable | L1 v_r order 0.80-0.97; P 0.67-0.74 | Moving's lead depends on it, table below |
| Generator sampling on a moving or jittered mesh | Wave on static jittered: order 1.82 from generators, 2.17 from centroids; uniform random 1.38 against 1.89 | **Breaks smooth tests** unless centroids (or integrals) are used |
| Wave amplitude | Linear exact solution has a floor at 2.67e-4 of the amplitude for amp 1e-4: exactly (gamma+1)/2 x amp x k t / pi, nonlinear steepening. p falls to 0 by n = 2048; score peaks then falls. Simple-wave exact solution removes it | **Silent trap**: a floor in the reference, not the code |
| Compile and start-up | LUNA 64^3: loop 0.66 s, whole run 42 s, cold cache; 128^3: 3.2 and 7.8 s. Ours: no compile | Say which T |
| Wall-time noise | Below | 1.8x from nowhere |

### The norm decides the moving mesh's lead

T(LUNA)/T(moving) at equal error, moving 256^3 (per-cell loop) and 128^3,
LUNA interpolated (`build-metric/analysis/extra.py`) **[D]**. Fine lattice 1024^3:
each lattice point takes the value of its Voronoi cell (nearest generator) or
LUNA cell, inside r < 1.5 R_s.

| norm | moving 256 | moving 128 | p LUNA | p moving |
|---|---|---|---|---|
| cell-sampled at generators (the current one) | 2.62 | 1.66 | 0.65 | 0.77 |
| cell-sampled at centroids | 2.20 | 1.53 | 0.65 | 0.75 |
| field: integral of abs(rho_h - rho) over the lattice | 1.81 | 1.13 | 0.66 | 0.78 |
| conservative: abs(rho_i - exact mean over cell i), weighted by volume | 1.55 | 0.90 | 0.55 | 0.67 |
| projection onto a common 128^3 grid | 1.05 | 0.89 | 0.72 | 0.78 |
| projection onto a common 64^3 grid | 0.99 | 0.64 | 0.70 | 0.80 |
| P, generators | 2.56 | 1.74 | 0.67 | 0.74 |
| mass-weighted rho | 4.10 (LUNA extrapolated) | 2.69 | 0.53 | 0.65 |
| v_r, generators | 8.95 (extrapolated) | 5.34 | 0.80 | 0.97 |

So the 3.2x (today 2.6x) is real for one norm. Reading **[I]**: sampling at the
generator favours a mesh that concentrates cells in the shell; a coarse common
grid discards exactly that resolution and lets errors of opposite sign cancel
(so does the conservative form, within a cell). Mass-weighted and v_r show what
the moving mesh does better: it has the mass and the post-shock flow right.
None of these is "the true" norm; the choice decides the headline number.
The radial-shell L1 was unusable as computed (non-monotonic with n) and is omitted.

### Wall-time noise

Node speed is not the main thing on LUNA here. LUNA Sedov 128^3, median M
cells/s, identical code **[M]**:

| job | node | LUNA | ours static 64^3, s |
|---|---|---|---|
| 31879151 | ravg1156 | 833 | |
| 31879152 | ravg1156 | 836 | |
| 31879806 | ravg1156 | 832 | 1.136 |
| 31879294 | ravg1172 | 832 | 1.129 |
| 31879295 | ravg1172 | 833 | |
| 31880402 | ravg1002 | 832, 834 | 1.137 |
| 31879808 | ravg1198 | **497** | 1.131 |
| 31880401 | ravg1198 | 819, then **458** in the same job | 1.135 |
| 31802206 (older) | ravg1002 | **~460** | |

LUNA is bimodal, about 830 or about 470, 1.8x, and the slow mode occurred on
ravg1198 and on ravg1002, which also gave 832. Each was its own process. Ours moved 0.7 % over the same nodes, including
the job where LUNA fell to 497. Cause not isolated. So:
- not clock alone **[D]**: ours did not move, LUNA did twice inside one node;
  the owner's finding of different application clocks (ravg1002 1095 MHz,
  ravg1199 1305 MHz) was not cross-checked here: my jobs logged only max clocks
  (1410/1215 MHz), and ravg1199 is not in my data. Ours took 6.82 s (ravg1198)
  and 6.84 s (ravg1002) for 400 steps at 128^3.
- Record the clock and repeat each timed run: take the faster of 2-3.
- Effect on the score: at x = 4 a 1.8x T shift equals a 16 % L1 change.

## Memory

Owner's decision: the score stays time and error only; memory is a separate
metric. Results of the form he asked to test, for the record **[M/D]**:

Max/min flatness of score_m = 1 / (T M L1^(7/p)), n >= 128: 37x LUNA, 5x moving
at p = 1; 1.7x and 1.5x at p = 0.65. It is worse than the base score at p = 1,
and p propagates more strongly: a factor 2 in L1 between codes moves S by 2^(4/p) and
score_m by 2^(7/p). Moving/LUNA at 256^3, from p = 0.45 to 1.2: 4.47 to 0.12 for
score_m, 5.0 to 0.64 for S. M ~ n^3 holds (exponent 2.9-3.07) above a fixed
offset: `nvidia-smi` adds ~1.4 GB on ours and 0.5 GB on LUNA, which dominates
at 64^3.

T x M at fixed error, GB s **[D]**:

| eps | LUNA | moving | LUNA better by |
|---|---|---|---|
| 0.058 | 26.5 | 63 | 2.4 |
| 0.050 | 160 | 272 | 1.7 |
| 0.045 | 529 | 767 | 1.45 |
| 0.0341 | 13640 | 13150 | 0.96 (tie) |

Score / (bytes per cell) ignores that moving needs fewer cells: at equal error
LUNA 163 B/cell against ours 2231 gives LUNA 5x, though T x M says a tie. It does
not measure what memory costs; dropped.

M itself is ambiguous: LUNA at 448^3 peaks at 14.6 GB by XLA stats and
33.2 GB by `nvidia-smi` (the pool grows by large steps even with
`XLA_PYTHON_CLIENT_PREALLOCATE=false`). Printed `mem = ...` is neither. With
33.2 GB, T x M is 31000 and moving wins 2.4x. M needs a definition.

Capacity on one 40 GB A100, Sedov **[M]**, 40 steps (job 31879296):
- LUNA FP64 fits 512^3 (21.8 GB), 576^3 fails. The note's "512^3 does not fit"
  (`hydro_stage3_2026-10-31.md:143-144`, job 31701766) was the default
  preallocation limit, not the code's size. Not run to t_end.
- LUNA FP32 fits 640^3 (21.4 GB), with 448^3 at 7.4 GB.
- Ours, moving: 256^3 with the **per-cell loop only** (37.4 GB, 2231 B/cell,
  whole run). Per-face at 256^3 and `--precision=mixed` per-face run out of memory on
  main at the first steps ("failed to allocate 7.722 GiB (face_rings_pool)",
  31879294, 31879296; 1.119 GiB `cell_face`, 31879807), though 2418.6 B/cell
  ran on 2026-10-31. Main's moving per-face peak is now 2475-2480 B/cell.
  **This is a regression of main against the stage 3 note and the README.**
- Ours static 256^3 fits (40.7 GB), 268^3 fails.
- Smallest error reachable: ours 0.0341; LUNA 512^3 about 0.031 and FP32 640^3
  about 0.027, extrapolated at p = 0.65 **[D]**, 30-40 min of GPU time, not run.

## More metrics, tested on the same data

| metric | result here | rewards | gaming | reference code? | standard? |
|---|---|---|---|---|---|
| (a) GPU-s to fixed error | table in Q3 | what a user pays | low; norm, T definition | no | **yes, first choice** |
| (a) Energy to fixed error | P 190-245 W during the loop, E ~ 200 W x T to 10 %; moving/LUNA 2.15 at 0.05 against 2.08 in T | same as T within one GPU | power cap | no | only across hardware |
| (b) Effective updates/s from W_ref(eps) | LUNA as reference: moving 3.0 G/s at 0.0341 (real 44 M/s), 1.7 G/s at 0.058; LUNA real 1.0-1.1 G/s | a familiar unit | as (a), plus choice of reference | yes | local; a restatement of (a) |
| (c) Smallest error on one GPU | ours 0.0341, LUNA ~0.031 FP64, ~0.027 FP32 **[D]** | memory efficiency | FP32, allocator | no | **yes, with T** |
| (c) GB at fixed error | 0.0341: 37.4 against 14.6; M ambiguous | same | M definition | no | yes, once M is defined |
| (d) Effective resolution | moving 128^3 = LUNA 200^3 (3.8x cells); 256^3 = 448^3 (5.4x); static 128 = 132 | scheme, ignoring cost | norm | yes | good second line |
| (e) Dissipation per cost (shear) | below | low numerical nu | limiter, compressive | no | **yes, as the no-exact-solution test** |
| (f) Mixed norms | L1(rho) + L1(P) + L1(v_r): not separate from its parts; contact and shock widths below | specific structures | weights | no | widths yes, mixes no |
| (g) Galilean score ratio | below | invariance | test choice | no | **yes, as one extra line** |

(e) Stationary diagonal shear wave, 1 time unit, nu from the kinetic energy decay
(job 31879806, 31879152). Re_eff = U / (k nu) **[M/D]**:

| code | n = 64 | n = 128 | M updates/s (128) |
|---|---|---|---|
| LUNA minmod | 77 | 329 | 824 |
| LUNA MC | 1160 | 9290 | 763 |
| ours static | 825 | 7590 | 120 |
| ours moving | 1020 | 8900 | 59 |

Minmod loses 28x at 128^3, because it clips the extrema; at equal resolution,
Re_eff of LUNA MC and ours agree within 25 %. Re_eff grows about n^3 here.
This makes LUNA's L1 wave score depend on the limiter: wave 128 L1 3.5e-4 (MC),
3.9e-3 (minmod), ours static 2.0e-3. Rank depends on a run-time flag.

KH and turbulence were not run. The shear wave replaces them as a cheap proxy;
it has no vortex or shock physics.

(f) Width (10-90 %) in cells at n = 2048, tube: contact LUNA MC 8, ours static 8,
moving 2; shock about 1 in all three. Moving's L1 near the contact is 40 % below
static, its rarefaction 38 % above, and its shock order falls to 0.5-0.6 at n >=
1024 (cause not isolated; 8 stuck triangles at 2048). Tube score at 2048:
static 1.2x moving by L1 rho, p = 1, with T 2x less.

(g) Score(bulk)/score(rest):

| test | LUNA | static | moving |
|---|---|---|---|
| wave, v = 3c, n = 96, by work (W L1^2) | 0.20 | 0.040 | 1.00 |
| wave, by L1 alone | L1 x1.57 | x2.51 | x1.00 (0.999-1.000) |
| Sedov 128^3, bulk 500 km/s, by T, p = 1 | 0.94 | 1.06 | 0.97 |

The wave shows invariance; Sedov L1 does not (it rises above 1 for the static
mesh). Static and LUNA steps double or quadruple. Use the wave. LUNA's T there has
compile spikes, so use work or repeat.

## Ranking of metrics

1. **T(eps0)** on each card. Needs no p, no reference, no extrapolation.
2. **Smallest error on one GPU** with M, as decided. One number that T(eps0) cannot show.
3. **Re_eff per cost** (shear wave) where no exact solution is wanted.
4. **Bulk ratio on the wave**: the moving-mesh claim in one figure.
5. Effective resolution as an explanatory line. Effective updates/s only as a
   restatement. Energy only across hardware. Reject: score_m, score / (B per cell),
   weighted mixes of variables.

## Recommendation

**Primary:** T(eps0): GPU-seconds to reach eps0 at t_end, from two of the code's
own runs bracketing eps0, n ratio <= 1.5, log-log interpolation. No p.

**Score**, the single-run form, normalised:

    S = 1000 s / t*,   t* = T (L1 / eps0)^x

t* is the estimated GPU-seconds to eps0, so S has a clear meaning: S = 1000 means
1 s. x is on the card. Valid only if 0.75 eps0 <= L1 <= 1.33 eps0: there an
error of 1 in x moves S by at most 33 %. Units and awkward magnitudes disappear.

**Report:** S, T, L1, M, n, cells x steps, code version, node, GPU model, application
clock, kernel and options (solver, limiter, precision, CFL), and whether start-up
is included. T: loop plus mesh, no snapshots; start-up and compile separately.
Two or three repeats, fastest. Same node for anything compared; otherwise calibrate.

**L1:** field L1: the integral of |f_h - f_exact| dV of the code's own
piecewise-constant field on a fixed sample lattice, reference solution exact to
well below eps0. Never at generators. Report the alternative (mass-weighted) as a
second line. This field norm is the one comparable between mesh types without
erasing what a mesh resolves.

**Cards** (x, eps0 proposals from measured data):

| test | d | t_end | variable, region | order seen | x | eps0 | resolutions |
|---|---|---|---|---|---|---|---|
| Sedov 3D (LUNA set-up, 1e4 yr; energy in fixed physical radius, not 8 cells) | 3 | 1e4 yr | field rho and P, r < 1.5 R_s | 0.65-0.77 | 7 | rho 0.05 | 128-384, code's choice |
| Sod, thin tube, Lx = 1, 8 x 8 | 3, scales as 1 | 0.25 | P (shock, rarefaction) and rho (contact, separately) | P 1, rho 0.66-0.85 | P 2 | P 2e-4 | n ~ 2048 (131k cells, still near the latency regime; a wider tube is better) |
| Sound wave, (1,1,1) | 3 | 1 period | rho at centroids, simple-wave exact, or amp 1e-6 | 2 | 2 | 1e-3 of amplitude | 64-256 |
| Shear wave, k (1,1,0) | 3 | t = 1 | v, and Re_eff | 2-3 | 2 | | 64-128 |
| Wave with bulk 3c | 3 | 1 period | ratio to wave | | | | 64-96 |

Sedov x = 7 was measured on LUNA; moving's 6.3 is why x only applies near eps0. A
code is valid for a card if two resolutions give local order within 0.1 of each
other and S within 20 %. All four tests together: Sedov alone is broken by order
0; the wave catches order 0 and precision, the shear wave the limiter.

Changes to the proposal, in short: x measured, not (d+1)/p; near eps0 only;
normalised to seconds; steps published; clock recorded; norm fixed and mesh-agnostic;
a smooth test required beside each shock test.

## Owner decides

1. Primary norm for Sedov. Field L1 (moving lead 1.8x) or something else: it
   changes the lead from 1.0x to 9x, and the headline 3.2x would change with it.
2. Whether a Sedov with a fixed physical energy radius replaces 8 cells. It would
   need a new baseline; the effect on p was not measured.
3. Whether to accept T(eps0) as primary and S as its estimate, with limits on L1.
4. eps0 per card.
5. Whether the cell-loop fallback at 256^3 is acceptable, or the memory regression
   (per-face moving no longer fits) gets fixed first. README's 256^3 line
   (`README.md:323`) describes a run that main no longer reproduces per-face.
6. LUNA's bimodal timing (830 or 470 M cells/s) needs a cause before LUNA is used
   as reference, as does correcting the "512^3 does not fit" line in the stage 3 note.
