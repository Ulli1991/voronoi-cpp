# Hydro stage 1, 2026-10-31 (branch `hydro-stage1` from main 77a903b)

Static Voronoi mesh, MUSCL-Hancock with HLLC (default) or HLL (owner's
decisions: MUSCL-Hancock, not forward Euler; HLLC as LUNA's default; the
zero-gradient path is first-order Godunov), one global time step, ideal
gas. Code: `src/hydro/` (header library: `euler.hpp`, `finite_volume.hpp`,
`snapshot.hpp`), driver `src/bench/hydro_sedov.cpp`, CTest `hydro`, A100
script `scripts/hydro_a100.sh`, analysis `scripts/hydro_sedov_compare.py`.
Logs: `/raven/u/uli/voronoi-cpp/build-hydro/logs/`, final runs and
snapshots `build-hydro/runs5/` (limiter study `runs4/`).

References for every scheme detail: LUNA at /raven/u/uli/LUNA (paths below
relative to `src/luna/` unless they start with `scripts/` or `docs/`);
AREPO next generation, read only, from /u/uli/codes/arepo:
`origin/Arepo2-nextgeneration-SoA` (51af37f5, "SoA") and
`origin/Arepo2-nextgeneration` (63107335, "ng"); AREPO line numbers are
SoA unless marked ng. The two branches have the same gradient, limiter and
face-state algorithm (`diff` of `src/mesh/mesh_gradients.cc`: data layout
and the cell loop only; `mesh_fluxes.cc`: the same code at other lines);
SoA is followed. AREPO master was read first and is not used.

## What a step does, and where each piece comes from

| Stage | LUNA (file:line) | AREPO-ng (file:line) | Voronoi version |
|---|---|---|---|
| state | U = (rho, rho v, E) per volume; W = (rho, vx, vy, vz, P) from U at the start of the step (`time_integration/integrators.py:202-222`) | | same; U as `View<double*[5]>`, W never stored |
| U -> W | `cons_to_prim`, with `preserve_face_states` `cons_to_prim_floored` (`hydro/conversions.py:15-36`; smallr = smallc = 1e-10, `template_config.py:41-42`) | Utherm floored at MinEgySpec, U changed (`src/mesh/mesh_update_primitives.cc:161-178`) | LUNA's floored conversion, U unchanged (brief); the smallest cell density in the runs is 3.0e-6, the floor 1e-10 |
| time step | CFL / sum_d (abs(v_d) + c) / dx_d, CFL 0.8 (`integrators.py:11-13, 23-28`, `template_config.py:32`) | per cell CourantFac * min(r / (c + abs(v)) [static mesh], min over particles j a signal can reach of (d_ij + r_i) / (c_i + c_j - (v_j - v_i).d_ij/d_ij)) (`src/data/sph_particle_data.h:305-334`; `src/time_integration/timestep.cc:148-171`; `timestep_treebased.cc:72-91`); r = (3V/4pi)^(1/3) (`sph_particle_data.h:280-287`); with `FORCE_EQUAL_TIMESTEPS` the global minimum (`timestep.cc:39-63`), rounded down to the power-of-two time bins; CourantFac is a parameter (`src/data/allvars.cc:103`), 0.3 in every Sedov example (`examples/sedov_3d/sedov_globaltstp_3d/param.txt:52`, the global-step Sedov), 0.4 in the documentation's example (`documentation/05_parameterfile.md:638`) | **AREPO-ng** (owner): its cell criterion, the pair criterion over the face neighbours only (AREPO walks the tree; a non-neighbour j cannot set the global minimum when d_ij >= r_j, since then (d_ij + r_i)/(A_i + A_j) >= min(r_i/A_i, r_j/A_j), A = c + abs(v)); no time-bin rounding; CourantFac 0.3 |
| first order | ORDER=0 (`hydro/muscl_hancock.py:34-64, 485-495`) | | order 0: face states = cell states |
| gradients | per axis, limited one-sided differences (`hydro/slopes.py:150-159`, minmod `:8-10`) | least squares W_j - W_i = a + g.d over all face neighbours, weight = face area (`src/mesh/mesh_gradients.cc:217, 263`), d between cell centroids (`:197, 243`), unknowns (a, g) (`:204, 250`), GSL LU inverse (`:229-232`) | AREPO-ng's fit (LUNA's axes do not exist here); Cholesky of the same SPD 4x4 matrix, once for all five variables |
| limiter | minmod inside the slope | per variable, min/max over neighbours whose face is at least 1e-6 of the cell surface (`mesh_gradients.cc:61-70`, `src/data/constants.h:285`); for each such face in order, `limit_gradient` keeps W_i + g.d within [min, max] (`:75-97, 120-148`) with d to the face centroid, or to the neighbour's centroid with `USE_MINMOD_SLOPE_LIMITING` (`:85-94`); master's velocity limiter is declared (`src/mesh/mesh.h:167`) but not called | AREPO-ng, face centroids (default); `hydro::Limiter` also offers the cell-centroid variant and none (study below) |
| half step | W - dt/2 sum_d A_d(W) dW/dx_d (`muscl_hancock.py:105-116`, `hydro/jacobian.py:84-97`) | in the gas frame, plus the advective terms on a static mesh: the same terms (`mesh_fluxes.cc:376-449`); time integration is two half-weighted flux passes (`:158`, `src/main/run.cc:166, 285`) | LUNA's formula (owner: MUSCL-Hancock) |
| to the face | W_half +- dW/2 (`muscl_hancock.py:121-122`) | W + time and space extrapolation to the face centroid from the gradient's centroid (`mesh_fluxes.cc:372, 451-465`) | W_half + g.(face centroid - cell centroid) |
| bad face state | Sedov defaults: an unphysical face becomes NaN and the run stops (`hydro/slopes.py:208-228`, `template_config.py:40, 43`) | if rho or P of the extrapolated state is not positive, neither extrapolation is added: the cell-centre W (`mesh_fluxes.cc:466-479`, ng `:731`) | **AREPO-ng** (owner): the cell's W; counted per step (per-face loop) |
| Riemann solver | HLLC `_hllc_core` (`hydro/fluxes.py:51-106`), the default (`template_config.py:36`); HLL `_hll_core` (`:135-167`); Davis speeds (`:65-66, 146-147`); rotated normal frame (`:9-48`) | no HLL; exact, HLLD (`src/riemann/`) | HLLC (default) and HLL with LUNA's formulas and guards, along n in box axes; lab-frame conversion for a moving face as AREPO-ng (`mesh_fluxes.cc:492-498`) |
| update | U + dt dU (`integrators.py:222`) | | U_i += -dt / V_i sum over the cell's face list, fixed order |

Solving along n without rotation: in HLL and HLLC every branch is chosen
by scalars (S_L, S_M, S_R) that do not depend on the frame, and within a
branch each flux component is the same linear combination of the matching
components of F, U and U*; HLLC's star momentum rho* (S_M, vt1, vt2) is
rho* (v_t + S_M n) in box axes with v_t = v - (v.n) n. Rotating is then
the identity up to rounding; CTest `hydro` checks it under 1000 random
rotations (3.5e-14) and against LUNA's float64 values (HLL 12 faces,
8e-16; HLLC 15 faces, 5e-16).

## Sedov set-up (`scripts/validation/runs/sedov3d_run.py`, `ic_generation/sedov3d.py`)

AREPO units kpc, 1e10 Msun, km/s (`sedov3d_run.py:19-27`), 50 pc box,
gamma 5/3, t_end = 1e4 yr = 1.0227e-5 (`:28-29, 39-45`); rho = 1 m_p
cm^-3, P = n k 1e4 K, E = 1e51 erg in code units (`sedov3d.py:9-14,
41-46`): rho = 2.4708e-3, P = 0.20395, E = 5.0279e-3. LUNA adds
(gamma-1) E / (8 dx^3) to P in the 8 central cells (`sedov3d.py:79-106`).
Ours: cells by distance of their generator from the centre; the deposit
set is the prefix whose volume sum is closest to 8 (L/128)^3, with
p_inject = (gamma-1) E / (its actual volume): 8 cells, 4.690e-10 vs
4.768e-10, within 0.38 pc (jittered); 7 cells (uniform random). LUNA's
boundary is zero-gradient, ours periodic: at t_end no cell beyond 16.3 pc
differs from the background by more than 1e-6 (both codes; half box 25 pc).
Points: jittered lattice (0.1 spacings) or uniform random, in Morton order.

## Validation at 128^3 (2,097,152 cells), A100, job 31699863

All runs to t_end with 10 snapshots; LUNA `ORDER=1 LIMITER=0` or
`ORDER=0`, `enable_x64=1`, `~/jax-env`. R_s: peak of the radial density
profile as LUNA's `scripts/validation/plots/sedov3d_plot.py:78-85` (64
bins of 0.67 pc; both codes fall in the same bin at all 10 times, so R_s
resolves nothing finer). L1: volume-weighted mean abs(f - f_exact) over
r < 1.5 R_s, divided by rho0, the largest analytic v_r and P; "centre":
the same for density over r < R_s/2 (not LUNA metrics). Analytic: LUNA's
`hydro/exact_sedov.py`.

| run | steps | R_s/analytic | profile peak rho/rho0 | L1 rho | L1 v_r | L1 P | min rho/rho0 | L1 rho, centre |
|---|---|---|---|---|---|---|---|---|
| Voronoi MUSCL-Hancock HLLC | 1642 | 0.978 | 2.608 | 0.0750 | 0.0289 | 0.0203 | 1.43e-3 | 0.0077 |
| LUNA ORDER=1 HLLC | 1079 | 0.977 | 2.528 | 0.0766 | 0.0311 | 0.0203 | 1.45e-3 | 0.0058 |
| Voronoi MUSCL-Hancock HLL | 1110 | 0.978 | 2.587 | 0.0698 | 0.0236 | 0.0193 | 3.42e-3 | 0.0022 |
| LUNA ORDER=1 HLL | 386 | 0.977 | 2.523 | 0.0708 | 0.0235 | 0.0190 | 1.53e-2 | 0.0187 |
| Voronoi first order HLLC | 1698 | 0.978 | 2.136 | 0.1035 | 0.0446 | 0.0298 | 1.23e-3 | 0.0182 |
| LUNA ORDER=0 HLLC | 1098 | 0.977 | 2.159 | 0.1039 | 0.0454 | 0.0296 | 1.35e-3 | 0.0172 |
| Voronoi first order HLL | 369 | 0.978 | 2.090 | 0.1010 | 0.0414 | 0.0307 | 5.19e-2 | 0.0745 |
| LUNA ORDER=0 HLL | 237 | 0.977 | 2.134 | 0.0974 | 0.0386 | 0.0290 | 5.10e-2 | 0.0734 |

Plot: `reports/hydro_stage1_sedov128.png`. Conservation over every full
run: mass and energy change by at most 1.7e-16 relative.

Face fallbacks (AREPO-ng rule), per-face loop, MUSCL-Hancock: HLLC 122,333
face sides over 1642 steps (in 1638 steps, at most 143 in one step, of
33.4M face sides per step); HLL 84,310 over 1110 steps (at most 146).

### The thin hot centre and the step count (owner's item 3)

Line by line against `mesh_gradients.cc` (weights, unknowns, centroid
distances, all neighbours in the fit, the 1e-6 surface threshold for
min/max and limiter, `limit_gradient` face by face in order, face-centroid
extrapolation): no deviation found. Variants, 128^3 jittered (jobs
31699165 with HLL, 31699546 with HLLC, both with the AREPO-ng time step
and fallback; first order from job 31699863):

| solver | limiter | steps | min rho/rho0 | L1 rho | L1 rho, centre |
|---|---|---|---|---|---|
| HLL | face centroids (AREPO-ng default) | 1110 | 3.4e-3 | 0.0698 | 0.0022 |
| HLL | cell centroids (`USE_MINMOD_SLOPE_LIMITING`) | 538 | 1.9e-2 | 0.0785 | 0.0251 |
| HLL | none | NaN after 112 steps | | | |
| HLL | first order | 369 | 5.2e-2 | 0.1010 | 0.0745 |
| LUNA HLL | minmod | 386 | 1.5e-2 | 0.0708 | 0.0187 |
| HLLC | face centroids | 1642 | 1.4e-3 | 0.0750 | 0.0077 |
| HLLC | cell centroids | 1638 | 1.4e-3 | 0.0841 | 0.0066 |
| HLLC | none | NaN after 158 steps | | | |
| LUNA HLLC | minmod | 1079 | 1.5e-3 | 0.0766 | 0.0058 |

Cause: with HLL, the face-centroid limiter (AREPO-ng's default) lets the
centre empty further than minmod (closer to the analytic zero: centre L1
0.0022 vs 0.0187); the cell-centroid variant gives LUNA's centre and half
the steps. With HLLC (the default now) both codes empty the centre to
1.4e-3 rho0 and agree. Not a deviation from AREPO-ng; nothing changed.
The remaining step ratio with HLLC, 1642/1079 = 1.52, is the time-step
rule (1.43 on a cubic cell for flow along an axis, above).

## Throughput, A100 40 GB, double, jobs 31699864 (128^3) and 31699865 (256^3)

Median over 40 steps from the Sedov start; phases fenced; "time step" =
the AREPO-ng criterion (one pass over the face neighbours); cell updates
per second = n / sum of the phase medians. (c) = per-face flux array +
gather (default at second order), (b) = per-cell loop (default at first
order). Jittered, Morton order.

| n | order | solver | loop | time step | gradients | fluxes | gather | step ms | M cell updates/s |
|---|---|---|---|---|---|---|---|---|---|
| 128^3 | 1 | HLLC | (c) | 2.27 | 7.30 | 3.44 | 5.65 | 18.66 | 112 |
| 128^3 | 1 | HLLC | (b) | 2.27 | 7.27 | 18.21 (with update) | | 27.76 | 76 |
| 128^3 | 1 | HLL | (c) | 2.27 | 7.27 | 2.84 | 5.64 | 18.03 | 116 |
| 128^3 | 1 | HLL | (b) | 2.27 | 7.28 | 12.06 | | 21.61 | 97 |
| 128^3 | 0 | HLLC | (b) | 2.28 | | 5.01 | | 7.29 | 288 |
| 128^3 | 0 | HLL | (b) | 2.27 | | 3.45 | | 5.73 | 366 |
| 256^3 | 1 | HLLC | (c) | 18.69 | 58.78 | 37.49 | 45.25 | 160.2 | 105 |
| 256^3 | 1 | HLLC | (b) | 18.70 | 58.78 | 146.72 | | 224.2 | 75 |
| 256^3 | 1 | HLL | (c) | 18.72 | 58.80 | 24.53 | 45.25 | 147.3 | 114 |
| 256^3 | 1 | HLL | (b) | 18.72 | 58.75 | 95.63 | | 173.1 | 97 |
| 256^3 | 0 | HLLC | (b) | 18.67 | | 39.15 | | 57.8 | 290 |
| 256^3 | 0 | HLL | (b) | 18.71 | | 27.38 | | 46.1 | 364 |

Earlier jobs for the record: with the brief's local rule only (no pair
criterion) the time step took 0.12 ms (128^3); counting fallbacks inside
the per-cell loop by a reduction made it 2x slower (job 31699166), so (b)
does not count them. Particle order (job 31698376, HLL, older time step):
in generator order instead of Morton, 256^3 (b) 67 vs 109 M/s, uniform
random 128^3 (b) 16 vs 104 M/s.

Whole MUSCL-Hancock runs, 128^3, no snapshots (job 31699864): HLLC 1642
steps, 30.6 s of GPU phases (18.6 ms/step); HLL 1110 steps, 20.1 s.

### LUNA on the same problem and GPU type (reference; not for the README)

LUNA's own 10-step "perf" windows (steady: the median over the windows
after step 10), and its "Performance after step 1" line (whole run,
compile excluded). Kernels: `hydro_kernel='xla'` (default) and
`'pallas'` (`docs/CONFIG.md:33, 652-655`; LUNA's docs report the Pallas
split step on an A100 at 1108 Mcell/s, Sedov 512^3 FP64,
`docs/HYDRO_KERNEL_PATHS.md:145-150`; the fused XLA kernel was slower
here, jobs 31699166 and 31699167: 359 vs 413 M/s at 128^3, 509 vs 682 at 256^3).

| n | solver | kernel | steady M cells/s | whole run ms/step | steps | job |
|---|---|---|---|---|---|---|
| 128^3 | HLLC | Pallas | 833 | 2.97 | 1075 | 31699864 |
| 128^3 | HLLC | XLA | 481 | 4.81 | 1075 | 31699864 |
| 128^3 | HLL | Pallas | 942 | 3.49 | 383 | 31699864 |
| 128^3 | HLL | XLA | 413 | 6.31 | 383 | 31699864 |
| 256^3 | HLLC | Pallas | 1050 | 20.21 | 100 | 31699865 |
| 256^3 | HLLC | XLA | 633 | 30.77 | 100 | 31699865 |
| 256^3 | HLL | Pallas | 1249 | 17.51 | 100 | 31699865 |
| 256^3 | HLL | XLA | 666 | 29.43 | 100 | 31699865 |

Against ours (same jobs): at 256^3 HLLC, ours 105 M/s vs LUNA's 1050
(Pallas) / 633 (XLA): 10x / 6x per step. Whole 128^3 Sedov run, HLLC:
ours 30.6 s, LUNA 1074 x 2.97 ms = 3.2 s (Pallas) / 5.2 s (XLA): 9.6x /
5.9x GPU time, at the accuracy of the validation table (1.52x of it from
the step count).

## Memory (Kokkos allocations on the device, `--memory`, job 31699865)

| | 128^3 B/cell | 256^3 B/cell | 256^3 GPU in use (of 42.4 GB) |
|---|---|---|---|
| mesh after `update()` + `update_cells()` | 2143 | 2143 | 37.4 GB |
| peak while building the mesh | 2167 | 2167 | |
| + hydro (b): U and a second U (80 B); gradients in the mesh's spare ring pool (no growth) | 2223 | 2223 | 38.7 GB |
| + hydro (c), order 1: U (40 B); gradients (120 B) and face fluxes (318 B) in the spare pool, grown by 188 B/cell | 2370 | 2371 | 41.2 GB |

Gradients are stored (15 doubles per cell), not recomputed (recomputing
means a least-squares fit for each side of every face, about 16 x 16
neighbour loads per cell); they cost no new memory because they live in the
mesh's spare ring pool (`Mesh::scratch`), asked for at every step. Left at
256^3: 3.7 GB (b), 1.2 GB (c).

## Tests and checks

CTest `hydro` (TESTING.md): 4.0 s on one CPU thread; all 7 CTest programs
pass on the CPU (OpenMP, one thread, 38.9 s) and on the A100 (job
31699863). clang-tidy (`scripts/lint_cpp.sh`, header filter includes
`src/hydro`) clean. Two runs give the same bits; the two loops agree to
4e-16 (HLL: same bits; HLLC: last bits differ between the two kernels,
cause not isolated).

## Open items

- The time step's neighbour pass (2.3 ms at 128^3, 12 % of a step) could
  be fused into the gradient kernel (it loads the same neighbours) if dt is
  taken after the gradients.
- The (c) gather (5.6 ms at 128^3) reads 5 face rows strided by the face
  count; an interleaved face flux may cut it.
- HLLC per-cell loop (b) is 2.4x its per-face cost; (b) is only the
  low-memory option at second order.
- Sound wave convergence and Sod (design note) not run yet; moving mesh
  (stage 3).
