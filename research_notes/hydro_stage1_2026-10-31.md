# Hydro stage 1, 2026-10-31 (branch `hydro-stage1` from main 77a903b)

Static Voronoi mesh, MUSCL-Hancock with HLL (owner's correction of the
brief: MUSCL-Hancock, not forward Euler; the zero-gradient path is the
first-order Godunov special case), one global time step, ideal gas.
Code: `src/hydro/` (header library: `euler.hpp`, `finite_volume.hpp`,
`snapshot.hpp`), driver `src/bench/hydro_sedov.cpp`, CTest `hydro`,
A100 script `scripts/hydro_a100.sh`, analysis `scripts/hydro_sedov_compare.py`.
Logs: `/raven/u/uli/voronoi-cpp/build-hydro/logs/`, runs and snapshots
`build-hydro/runs2/` (final), `build-hydro/runs/` (first round).

References read for every scheme detail: LUNA at /raven/u/uli/LUNA (paths
below relative to `src/luna/` unless they start with `scripts/`); AREPO
next generation, read only, from /u/uli/codes/arepo:
`origin/Arepo2-nextgeneration-SoA` (51af37f5, "SoA") and
`origin/Arepo2-nextgeneration` (63107335, "ng"); AREPO line numbers are
SoA unless marked ng. For gradients and limiter the two branches have the
same algorithm (`diff` of `src/mesh/mesh_gradients.cc`: only the data
layout and the loop over cells differ); for the face extrapolation the
same code at other lines. AREPO master was read first and is not used
(owner: next generation is the gold standard).

## What a step does, and where each piece comes from

| Stage | LUNA (what it does, file:line) | Voronoi version (what we do) |
|---|---|---|
| state | U = (rho, rho v, E) per volume between steps; W = (rho, vx, vy, vz, P) from U at the start of the step (`time_integration/integrators.py:202-222`: `to_prim`, divergence, `U + dt dU`) | same; U as `View<double*[5]>` (SoA on CUDA), W never stored |
| U -> W | `cons_to_prim` (`hydro/conversions.py:15-25`); with `preserve_face_states` `cons_to_prim_floored` (`:28-36`: rho >= smallr, e >= smallc^2/(gamma (gamma-1)) rho; U not changed), smallr = smallc = 1e-10 (`template_config.py:41-42`) | always the floored conversion (brief). In the Sedov runs the smallest cell density is 8.4e-6, so the density floor never acts on cell states |
| time step | `cell_dt` = CFL / sum_d (abs(v_d) + c) / dx_d, CFL 0.8 (`integrators.py:11-13, 23-28`, `template_config.py:32`) | the brief's rule C min r/(c + abs(v)), r = (3V/4pi)^(1/3), C = 0.3, which is AREPO-ng's criterion (`src/time_integration/timestep.cc:156-165`: rad/(c + abs(v)) on a static mesh, times CourantFac). On a cubic cell ours is 0.186 dx/(c+abs(v)), LUNA's 0.267 dx/(c+abs(v)) for flow along an axis: 1.43x more steps from the rule alone |
| first order | ORDER=0: HLL of the cell states (`hydro/muscl_hancock.py:34-64, 485-495`) | order 0: no gradients, face states = cell states |
| gradients | per axis, limited one-sided differences of W (`hydro/slopes.py:150-159`; LIMITER 0 minmod `:8-10`, 1 MC `:22-24`, 2 van Leer `:26-33`, `slope_limiter` `:98-108`) | cannot carry over (no axes): AREPO-ng's least-squares fit (`src/mesh/mesh_gradients.cc:150-282`): W_j - W_i = a + g.d over all face neighbours, weight = face area (`:217`, `:263`), d between cell centroids (`:197`, `:243`), unknowns (a, g) (`:204`, `:250`); AREPO inverts by GSL LU (`:229-232`), we use Cholesky of the same symmetric positive definite 4x4 matrix, once for all five variables |
| limiter | inside the slope (minmod of the one-sided differences) | AREPO-ng: per variable, min/max over the neighbours whose face is at least 1e-6 of the cell surface (`mesh_gradients.cc:61-70`, `src/data/constants.h:285`), then for each such face in order the gradient is scaled so the value at the face centroid stays in [min, max] (`:75-97`, `limit_gradient` `:120-148`; face centroid unless `USE_MINMOD_SLOPE_LIMITING`, `:85-94`). Master's velocity limiter `limit_vel_gradient` is declared in SoA `src/mesh/mesh.h:167` but never called: not used |
| half step | W_half = W - dt/2 sum_d A_d(W) dW/dx_d, x then y then z (`hydro/muscl_hancock.py:105-116`, rows `hydro/jacobian.py:84-97`) | the same formula with the least-squares gradient as dW/dx_d (`euler.hpp: half_step`). AREPO-ng with `VORONOI_STATIC_MESH` adds the same advective terms to its convective extrapolation (`src/mesh/mesh_fluxes.cc:423-437`), so the two agree on a static mesh |
| to the face | W_half +- dW/2, half a cell along the axis (`muscl_hancock.py:121-122`) | W_half + g.(face centroid - cell centroid), as AREPO-ng (`mesh_fluxes.cc:372`: FaceCenter - Grad.Center) |
| bad face state | LUNA's Sedov run uses the defaults, floors off and `allow_first_order_faces=False`: an unphysical face (rho or P <= 0) becomes NaN and the run stops (`hydro/slopes.py:208-228`, `template_config.py:40, 43`); with floors the RAMSES rule (`slopes.py:181-186`) | LUNA's RAMSES rule (`euler.hpp: face_guard`). **Disagreement, the owner's call:** AREPO-ng drops the whole extrapolation when rho or P would not be positive (`mesh_fluxes.cc:467-479`, ng `:731`). How often the guard acts was not counted |
| Riemann solver | HLL `_hll_core` (`hydro/fluxes.py:135-167`), Davis speeds S_L = min(vn_L - c_L, vn_R - c_R), S_R = max(vn_L + c_L, vn_R + c_R) (`:146-147`), safe sqrt for c (`:142-145`), in the rotated normal frame (`:42-48`) | the same formulas along n = A/abs(A) without rotation (HLL is linear in the vector components, so this equals rotate, solve, rotate back up to rounding: 8e-16 from LUNA's float64 values in CTest `hydro`); a face-velocity argument w for stage 3 with the lab-frame conversion of AREPO-ng's `flux_convert_to_lab_frame` (`mesh_fluxes.cc:492-498`). AREPO-ng has no HLL |
| update | U + dt dU (`integrators.py:222`) | U_i += -dt / V_i sum over the cell's face list of the outgoing flux times abs(A), in the list's fixed order. AREPO-ng applies two half-weighted flux passes per step (`mesh_fluxes.cc:158`, `src/main/run.cc:166, 285`); not used (owner: MUSCL-Hancock) |

Our own choices (in neither code): Cholesky instead of LU (same matrix,
rounding only); the deposit set and the L1 metric (below).

## Sedov set-up (`scripts/validation/runs/sedov3d_run.py`, `ic_generation/sedov3d.py`)

AREPO units kpc, 1e10 Msun, km/s (`sedov3d_run.py:19-27`), 50 pc box,
gamma 5/3, t_end = 1e4 yr = 1.0227e-5 (`:28-29, 39-45`); rho = 1 m_p cm^-3,
P = n k 1e4 K, E = 1e51 erg in code units (`sedov3d.py:9-14, 41-46`):
rho = 2.4708e-3, P = 0.20395, E = 5.0279e-3. LUNA adds p_inject =
(gamma-1) E / (8 dx^3) to P in the 8 central cells (`sedov3d.py:79-106`).
Ours: cells sorted by the distance of their generator from the box centre
(periodic nearest image); the deposit set is the prefix of that order
whose volume sum is closest to 8 (L/128)^3; p_inject = (gamma-1) E /
V_deposit with V_deposit its actual volume, so exactly E goes in.
Jittered: 8 cells, 4.690e-10 vs LUNA's 4.768e-10, generators within 0.38
pc of the centre; uniform random: 7 cells, 4.787e-10, within 0.41 pc.
LUNA's boundary is zero-gradient (BOUNDARY=1, `sedov3d_run.py:39`), ours
periodic: at t_end the farthest cell whose density or pressure differs
from the background by more than 1e-6 is at 15.1-15.9 pc (table below) in
both codes, the half box is 25 pc: the boundaries never see the blast.

Points: `support::make_points_off_grid` (jittered: lattice moved by up to
0.1 spacings; uniform random), scaled to the box and renumbered in Morton
order by the driver (the mesh keeps the caller's numbering; see throughput).

## Validation at 128^3 (2,097,152 cells), A100, job 31698375

Every run to t_end with 10 snapshots. LUNA: /raven/u/uli/LUNA with
`~/jax-env` (jax 0.10.1, as LUNA's `submit/` scripts), `ORDER=1 LIMITER=0`
or `ORDER=0`, `riemann_solver=hll enable_x64=1`. Analysis:
`scripts/hydro_sedov_compare.py` (analytic: LUNA's
`hydro/exact_sedov.py:solution`). R_s: peak of the radial density profile
as LUNA's `scripts/validation/plots/sedov3d_plot.py:78-85` with
`metrics.radial_profile` (64 bins of 0.67 pc; both codes land in the same
bins, so R_s resolves nothing finer). L1: volume-weighted mean of
abs(f - f_exact) over r < 1.5 R_s(analytic), divided by rho0, by the
largest analytic v_r and by the largest analytic P (not a LUNA metric).

| run | steps | R_s / analytic (13.395 pc) | max cell rho/rho0 | profile peak rho/rho0 | L1 rho | L1 v_r | L1 P | disturbed radius |
|---|---|---|---|---|---|---|---|---|
| Voronoi MUSCL-Hancock, jittered | 1110 | 0.978 | 2.843 | 2.587 | 0.0698 | 0.0236 | 0.0193 | 15.07 pc |
| LUNA ORDER=1 (minmod) | 386 | 0.977 | 2.735 | 2.523 | 0.0708 | 0.0235 | 0.0190 | 15.24 pc |
| Voronoi first order, jittered | 369 | 0.978 | 2.218 | 2.090 | 0.1010 | 0.0414 | 0.0307 | 15.92 pc |
| LUNA ORDER=0 | 237 | 0.977 | 2.270 | 2.134 | 0.0974 | 0.0386 | 0.0290 | 15.86 pc |
| Voronoi MUSCL-Hancock, uniform random | 1953 | 0.980 | 3.060 | 2.576 | 0.0719 | 0.0248 | 0.0207 | 15.66 pc |

R_s(t) at the 10 snapshot times: the two codes fall in the same bin at
every time. Plot: `reports/hydro_stage1_sedov128.png` (profiles at 1e4 yr,
R_s(t)). Conservation over the full runs: total mass and energy change by
at most 1.7e-16 relative; total momentum stays below 1.1e-22 (scale 5.6e-5).

Steps: first order 369 against LUNA's 237 is the time-step rule (x1.43).
MUSCL-Hancock takes 1110 against 386: the hot centre is thinner in our run
(at t_end min rho 8.4e-6 vs LUNA's 3.8e-5, max c + abs(v) 5556 vs 2564
km/s; job 31698083 snapshots), so the global step is a further 2.2x
smaller. The analytic centre density goes to zero; whether the limiter
(AREPO's face limiter vs minmod) or the gradient causes it was not isolated.

## Throughput, A100 40 GB, double, job 31698376 (gpu1)

Median over 40 steps from the Sedov start (phases fenced; "time step" =
the dt reduction); cell updates per second = n / sum of the phase
medians. Order 1 = MUSCL-Hancock; (b) = per-cell loop (each face flux
computed from both sides), (c) = per-face flux array plus per-cell gather.
Jittered points in Morton order unless stated.

| n | order | loop | time step ms | gradients ms | fluxes ms | gather ms | step ms | M cell updates/s |
|---|---|---|---|---|---|---|---|---|
| 128^3 | 1 | (b) | 0.12 | 7.06 | 11.95 (with update) | | 19.13 | 110 |
| 128^3 | 1 | (c) | 0.12 | 7.07 | 2.26 | 5.65 | 15.09 | 139 |
| 128^3 | 0 | (b) | 0.12 | | 3.37 (with update) | | 3.48 | 602 |
| 128^3 | 0 | (c) | 0.12 | | 1.52 | 5.65 | 7.29 | 288 |
| 256^3 | 1 | (b) | 0.69 | 56.66 | 96.21 (with update) | | 153.6 | 109 |
| 256^3 | 1 | (c) | 0.69 | 56.72 | 17.96 | 45.30 | 120.7 | 139 |
| 256^3 | 0 | (b) | 0.69 | | 26.50 (with update) | | 27.2 | 617 |
| 256^3 | 0 | (c) | 0.69 | | 12.14 | 45.31 | 58.1 | 289 |
| 128^3 uniform random | 1 | (b) | 0.12 | 8.08 | 11.97 (with update) | | 20.17 | 104 |
| 128^3 uniform random | 1 | (c) | 0.12 | 8.10 | 2.25 | 5.43 | 15.89 | 132 |

Particle order, same job: with the generator's order (lattice rows;
uniform: random) instead of Morton, order 1: 128^3 jittered (b) 115, (c)
133 M/s (rows are as good as Morton at this size); 256^3 jittered (b) 67,
(c) 119 M/s; 128^3 uniform random (b) 16, (c) 23 M/s. The driver uses
Morton order.

Choice: for MUSCL-Hancock (c) is 1.26x (128^3) and 1.28x (256^3) faster
than (b): the Riemann solve and the face states are done once per face, at
the price of a 5.6 ms gather. At first order it is the reverse, (b) is
2.1x faster (cheap flux; the gather dominates). (c) needs the face flux
array (5 doubles per face, 318 B/cell) and still fits at 256^3 with 1.2 GB
to spare (memory below). Default in `hydro_sedov`: (c) at order 1; (b) stays
for first order and as the low-memory option.

Whole Sedov runs (validation job 31698375, 11 HDF5 snapshots, loop (b)):
MUSCL-Hancock 25.0 s, first order 5.1 s.

### LUNA on the same problem and GPU type (reference; not for the README)

| | steps | ms/step | M cells/s | job |
|---|---|---|---|---|
| LUNA ORDER=1 HLL minmod, no snapshots | 383 | 6.34 | 331 | 31698376 (same job as the table above) |
| LUNA ORDER=0 HLL, no snapshots | 234 | 3.68 | 570 | 31698376 |
| LUNA ORDER=1, no snapshots | 383 | 7.77 | 270 | 31698086 (earlier job) |
| LUNA ORDER=0, no snapshots | 234 | 5.54 | 378 | 31698086 |
| LUNA ORDER=1, 11 snapshots | 386 | 13.11 / 18.19 | 160 / 115 | 31698375 / 31698083 |

LUNA's numbers are its "Performance after step 1 (no compile, output
included)" line; they vary by 20-40 % between jobs. Per step our
MUSCL-Hancock (c) (15.1 ms) is 2.4x LUNA's 6.3 ms; per run (steps x
ms/step: 1110 x 15.1 ms = 16.8 s against 383 x 6.3 ms = 2.4 s) about 7x,
at the same accuracy (validation table). The extra steps (above) are half
of it.

## Memory (Kokkos allocations on the device, `--memory`, job 31698376)

| | 128^3 B/cell | 256^3 B/cell | 256^3 GPU in use (of 42.4 GB) |
|---|---|---|---|
| mesh after `update()` + `update_cells()` (current) | 2143 | 2143 | 37.4 GB |
| peak while building the mesh | 2167 | 2167 | |
| + hydro (b), order 0 or 1: U and a second U (80 B); gradients in the mesh's spare ring pool (no growth) | 2223 | 2223 | 38.7 GB |
| + hydro (c), order 1: U (40 B); gradients (120 B) and face fluxes (318 B) in the spare pool, which grows by 188 B/cell | 2370 | 2371 | 41.2 GB |
| + hydro (c), order 0: U; face fluxes in the spare pool | 2250 | 2251 | 39.2 GB |

Gradients are stored (15 doubles per cell), not recomputed: recomputing
means a least-squares fit for each side of every face, about 16 x 16
neighbour loads per cell, against 120 B/cell that cost no new memory
here because they live in the mesh's spare ring pool (`Mesh::scratch`),
idle while the mesh does not change (asked for at every step, since
`update()` and `update_cells()` invalidate it). Left at 256^3: 3.7 GB
(~220 B/cell) with (b), 1.2 GB (~70 B/cell) with (c).

## Tests and checks

CTest `hydro` (TESTING.md): 2.3 s on one CPU thread, 0.85 s on the A100;
all 7 CTest programs pass on the CPU (OpenMP, one thread) and on the A100
(job 31698375). clang-tidy (`scripts/lint_cpp.sh`, the header filter now
includes `src/hydro`) is clean. The two loops give the same bits, and two
runs give the same bits (CPU and A100).

## Open items

- Unphysical face states: LUNA's RAMSES floors (now) or AREPO-ng's
  dropped extrapolation (owner's call); count them per step.
- Thinner centre than LUNA at second order (2.2x more steps): isolate
  limiter against gradient (AREPO's `USE_MINMOD_SLOPE_LIMITING` variant,
  or a minmod-type face limiter).
- The (c) gather (5.6 ms at 2.1M) reads 5 face rows strided by the face
  count, in face-list order; an interleaved face flux or a face numbering
  that follows the cell lists may cut it.
- Sound wave convergence and Sod (design note) not run yet.
- HLLC (stage 2), moving mesh (stage 3).
