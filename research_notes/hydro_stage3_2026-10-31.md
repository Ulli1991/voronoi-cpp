# Hydro stage 3, 2026-10-31: the moving mesh (branch `hydro-stage1`)

The mesh-generating points move with the flow plus AREPO next
generation's regularisation; the mesh is repaired every step through
`voronoi::Mesh::update(x)` and `update_cells()`; fluxes are taken through
the moving faces. The static mesh of stage 1
(research_notes/hydro_stage1_2026-10-31.md) is the special case
`hydro::Motion::static_mesh` and gives the same results as before (same
steps, same accuracy table to 4 digits; the state is now extensive, so
the last bits differ). Code: `src/hydro/finite_volume.hpp`, driver
`src/bench/hydro_sedov.cpp --moving [--bulk=]`, `scripts/luna_sedov_bulk.py`,
`scripts/hydro_a100.sh` (modes `sedov_moving`, `blast_moving`, `moving256`,
`static256`, `moving256full`, `luna256`). Logs
`/raven/u/uli/voronoi-cpp/build-hydro/logs/`, runs `build-hydro/runs6/`.

References: AREPO next generation, read only, /u/uli/codes/arepo,
`origin/Arepo2-nextgeneration-SoA` (51af37f5, "SoA", followed) and
`origin/Arepo2-nextgeneration` (63107335, "ng"); lines are SoA unless
marked. LUNA has no moving mesh; it covers only what stage 1 took from it.

## The scheme, piece by piece

| Piece | AREPO-ng (file:line) | Here |
|---|---|---|
| state | extensive mass, momentum, energy per cell, updated by the face fluxes times area and dt (`src/mesh/mesh_fluxes.cc:158-164`); density = mass / volume, velocity = momentum / mass (`src/mesh/mesh_update_primitives.cc:129-131`) | Q = (mass, momentum, energy) per cell; W = primitive(Q / V) where needed (LUNA's floored conversion as in stage 1); Q -= dt sum of outgoing fluxes, so the totals change only by rounding |
| point velocities | `set_vertex_velocities` (`src/voronoi/voronoi_vertex_velocities.cc:26-139`): the fluid velocity (`:31`), plus with `REGULARIZE_MESH_CM_DRIFT` a drift towards the cell's centroid; the Sedov examples set `REGULARIZE_MESH_CM_DRIFT`, `_USE_SOUNDSPEED`, `_FACE_ANGLE` (`examples/sedov_3d/sedov_globaltstp_3d/Config.sh:11-13`): fraction = 0 below 0.75 CellMaxAngleFactor of the largest face angle, rising linearly to CellShapingSpeed at CellMaxAngleFactor (`:100-106`); speed = fraction * c (`:111-118`, no gravity here), along the generator-to-centroid direction (`:137`); face angle sqrt(A/pi) / (d_ij/2) (`src/voronoi/cell.cc:563, 576-578`); CellShapingSpeed 0.5, CellMaxAngleFactor 2.25 (`sedov_globaltstp_3d/param.txt:93-94`) | the same; the condition `dt > 0` (`:100`) holds every step. SoA uses `dt` there without declaring it (its definition, `:45`, sits in a comment); ng defines it as the cell's step (ng `:28-32`); it enters only as `dt > 0` |
| drift | the points move with the vertex velocity over the step | x += dt w, wrapped into [0, box) |
| mesh | rebuilt or updated every step | `Mesh::update(x)` (repair; host fallback or rebuild when stuck), `update_cells()` |
| face velocity | `prepare_face_velocity` (`mesh_fluxes.cc:294-331`): the mean of the two point velocities plus, along the generator direction n, c.(w_L - w_R) / d with c the face centroid relative to the generators' midpoint, abs(c) limited to 0.9 d | the same |
| Riemann problem | in the face frame, velocities minus the face velocity (`mesh_fluxes.cc:18-22`), flux back to the lab frame (`:492-498`) | the `w` argument of stage 1 (HLLC default; the same lab-frame conversion) |
| face states | gradients at the start of the step from the cell centroids (`mesh_gradients.cc:36`); in the second flux pass a time extrapolation in the frame of the gas with the spatial extrapolation vector shifted by -dt v (`mesh_fluxes.cc:376-441`) | MUSCL-Hancock (owner): W - dt/2 sum_d A_d(W) dW/dx_d + g.(r + dt/2 w_face), r the face centroid relative to the cell centroid at the start: the state at the point where the face centroid is at the half step. This equals the half step in the frame of the cell (velocities relative to w_i in the advective terms) plus the extrapolation from the cell centroid to the face centroid both moved to the half step, (r + dt/2 (w_face - w_i)); see below |
| geometry of the flux | first pass on the old mesh, second on the new one, each with half weight (`mesh_fluxes.cc:158`; `src/main/run.cc:166, 285`) | one flux pass with the areas and normals of the mesh at the start of the step (MUSCL-Hancock has one flux evaluation); volumes of the new mesh after the update |
| time step | `CurrentMaxTiStep` = rad / (c + abs(v - w)) for a moving mesh (`src/time_integration/timestep.cc:156-165`), lowered by the signal-speed tree walk (`timestep_treebased.cc:72-91`), then min with rad / c and times CourantFac (`src/data/sph_particle_data.h:305-334`) | courant * min(r/(c + abs(v - w)), face-neighbour pair criterion), global minimum, CourantFac 0.3 (`sedov_globaltstp_3d/param.txt:52`) |
| time integration | two half-weighted flux passes per step (Heun) | MUSCL-Hancock, one pass (owner's decision); the difference is recorded here |

The half step in the moving frame (no paper at hand; derived here and
checked by the Galilean test): Springel's moving-frame prediction
W - dt/2 A(W; v - w_i) grad W extrapolated over (r + dt/2 (w_face - w_i))
equals, term by term (A(W; v - w_i) grad W = A(W; v) grad W - (w_i . grad) W),
W - dt/2 A(W; v) grad W + (r + dt/2 w_face) . grad W, which is what the code
computes (`FaceFlux::side`).

Not exact: the face areas are those at the start of the step while the
volumes after it come from the new mesh, so the geometric conservation
law holds only to O(dt^2) where the point velocities differ between
neighbours. A uniform translation keeps it exactly (CTest).

## Tests (CTest `hydro`, 7.95 s on one CPU thread; A100 job 31700850)

- A uniform state with a bulk velocity (1, 0.5, -0.25) on a moving
  jittered mesh, 5 steps whose displacement is a multiple of the snapping
  grid: uniform to 4.7e-16.
- Mass, momentum and energy over 10 steps with moving cells (a blast plus
  30 % noise): conserved to 6.7e-18 (8115 device flips, no rebuild).
- Galilean invariance: a blast at rest and the same blast with the box
  moving at 16 along x (Mach 12 in the background, c = 1.29), 40 steps of 2^-12: on the moving mesh the two agree cell by cell
  to 6.8e-8 (rounding grown through the blast) and their radial density
  profiles to 1.2e-13; on the static mesh the profiles differ by 6.9e-2.

## Sedov at 128^3, A100 (jobs 31700850, 31700851)

HLLC, MUSCL-Hancock, the analysis of stage 1 (`scripts/hydro_sedov_compare.py`;
`--bulk=500` for the moving blast: the centre moves with the box,
velocities relative to it).

| run | steps | R_s/analytic | profile peak rho/rho0 | max cell rho/rho0 | L1 rho | L1 v_r | L1 P | centre L1 rho | disturbed radius |
|---|---|---|---|---|---|---|---|---|---|
| ours, moving mesh | 354 | 0.978 | 2.844 | 4.018 | 0.0580 | 0.0181 | 0.0147 | 0.0068 | 14.9 pc |
| ours, static mesh | 1642 | 0.978 | 2.608 | 3.111 | 0.0750 | 0.0289 | 0.0203 | 0.0077 | 15.1 pc |
| LUNA ORDER=1 HLLC | 1079 | 0.977 | 2.528 | 2.804 | 0.0766 | 0.0311 | 0.0203 | 0.0058 | 15.2 pc |
| moving blast, 500 km/s: ours, moving mesh | 354 | 0.978 | 2.844 | 4.018 | 0.0580 | 0.0181 | 0.0147 | 0.0068 | 14.9 pc |
| moving blast: ours, static mesh | 1694 | 0.979 | 2.600 | 3.926 | 0.0733 | 0.0288 | 0.0203 | 0.0074 | 16.4 pc |
| moving blast: LUNA (periodic) | 1041 | 0.978 | 2.515 | 3.043 | 0.0745 | 0.0309 | 0.0200 | 0.0058 | 17.5 pc |

The moving-mesh blast gives the same numbers with and without the bulk
velocity (Galilean invariance); on the static mesh and in LUNA the
disturbed region spreads further ahead (16.4 and 17.5 pc against 14.9 pc).
At rest, the moving mesh has the sharpest shell (profile peak 2.84 rho0
against 2.61 static and 2.53 LUNA) and the smallest L1 errors. It takes
354 steps against 1642 static: in the hot centre the cells move with the
gas, so abs(v - w) is small in the time-step criterion.

Mesh updates (moving, at rest): 354 updates, no rebuild; 114 stuck
triangles in all, finished by the host fallback (61 host flips); 1.81M
device flips. Face fallbacks (AREPO-ng rule): 18,770 face sides in 351 of
354 steps (at most 138 in one step; 33.2M face sides per step).
Conservation: mass and energy to 1.7e-16 relative over the whole run.

## Throughput (A100, double, jittered, Morton order, median of 40 steps; job 31700852)

| n | mesh | time step | point velocities | gradients | fluxes | gather | drift | mesh update | update_cells | step ms | M cell updates/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 128^3 | moving | 2.67 | 3.26 | 7.61 | 4.27 | 5.69 | 0.11 | 7.88 | 6.84 | 38.3 | 54.7 |
| 256^3 | moving | 22.76 | 23.42 | 60.59 | 49.71 | 45.66 | 0.88 | 51.03 | 53.46 | 307.5 | 54.6 |
| 256^3 | static | 22.34 | | 59.50 | 42.09 | 45.68 | | | | 169.6 | 98.9 |

Whole 128^3 Sedov (job 31700850): moving 14.3 s of GPU phases (354
steps), static 31.9 s (1642 steps), LUNA 1078 x 6.96 ms = 7.5 s (XLA kernel
with 11 snapshots; Pallas, stage 1: 3.2 s without snapshots).

Whole Sedov at 256^3 (16.8M cells), HLLC:

| run | steps | GPU time | R_s/analytic | profile peak rho/rho0 | L1 rho | L1 v_r | L1 P | centre L1 rho | job |
|---|---|---|---|---|---|---|---|---|---|
| ours, moving mesh, per-cell loop | 918 | 364 s (394 ms/step) | 0.994 | 3.277 | 0.0341 | 0.0093 | 0.0088 | 0.0045 | 31701234 |
| LUNA ORDER=1 HLLC, Pallas | 4031 | 4030 x 16.65 ms = 67 s | 0.994 | 2.971 | 0.0495 | 0.0181 | 0.0126 | 0.0066 | 31700854 |
| ours, static mesh, per-face loop | did not finish in the 15-min job (job 31700853) | | | | | | | | |

The moving 256^3 run used the per-cell loop: with the per-face loop it
ran out of device memory after 3.7 min (job 31701027: "failed to
allocate 6.847 GiB (face_rings_pool)", the repair growing its ring pool
while the face flux array had grown the spare one). Mesh updates of the
moving 256^3 run: 918, no rebuild, 767 stuck triangles finished by the
host fallback (389 host flips), 15.3M device flips; mass and energy
conserved exactly in the printed digits. At 256^3 the moving mesh is
more accurate than LUNA (L1 rho 0.034 vs 0.050) at 5.4x LUNA's GPU time.

## Memory at 256^3 (Kokkos allocations, `--memory`, job 31700852)

Moving, per-face loop, first 40 steps: 2418.6 B/cell at the peak (mesh
with its repair state, face lists and the hydro's Q, positions, point
velocities, and the gradients and face fluxes in the spare ring pool);
42.03 GB of 42.40 GB in use. Over a whole run this did not fit: a later
repair grew the ring pool by 6.8 GiB and failed (job 31701027; fixed by
7082aad, see "The 256^3 moving run with the per-face loop"). With the
per-cell loop (no face flux array; the gradients fit in the spare pool
without growing it) the whole moving 256^3 run completed (job 31701234;
its peak was not recorded). Static: 2370.6 B/cell, 41.22 GB.

## Accuracy per GPU-second against LUNA (2026-10-31, after merging main 9159978)

LUNA's Sedov (`scripts/validation/runs/sedov3d_run.py`), ORDER=1 minmod,
HLLC, FP64, `hydro_kernel=pallas`, at increasing resolution until its L1
density error reaches our moving mesh at 256^3. One analysis for both codes
(`scripts/hydro_sedov_compare.py`, the final snapshot at 1e4 yr). Runs
longer than one 15-min job continue from LUNA's own checkpoints: a
checkpoint every 330 s of wall time (`checkpoint_wall_time_interval`,
`src/luna/setup/outputs.py:325-360`) and `restart=1` in a chained job,
which restarts from the newest checkpoint (`docs/CONFIG.md:672-674`;
`scripts/hydro_a100.sh` mode `luna_resolution`). GPU time, both codes,
without start-up, compilation and output: LUNA steps x N^3 / (median of
its 10-step "perf" windows after step 10); ours the sum of the fenced
phases (`gpu_s`). LUNA at 512^3 does not fit the A100 40 GB in FP64 with
Pallas ("Out of memory while trying to allocate 10.30GiB", job 31701766).

| code | N^3 | steps | GPU time | L1 rho | L1 v_r | L1 P | job |
|---|---|---|---|---|---|---|---|
| LUNA | 128 | 1075 | 2.7 s | 0.0766 | 0.0311 | 0.0203 | 31701998 |
| LUNA | 192 | 2329 | 16.3 s | 0.0595 | 0.0231 | 0.0152 | 31701761 |
| LUNA | 256 | 4031 | 64.6 s | 0.0495 | 0.0181 | 0.0126 | 31701999 |
| LUNA | 320 | 6167 | 188.2 s | 0.0425 | 0.0148 | 0.0109 | 31701762 |
| LUNA | 384 | 8727 | 459.7 s | 0.0376 | 0.0128 | 0.0097 | 31701763 |
| LUNA | 448 | 11703 | 932.8 s | 0.0341 | 0.0114 | 0.0088 | 31701765 + 31701767 (restart at step 7775) |
| ours, static | 128 | 1642 | 31.8 s | 0.0750 | 0.0289 | 0.0203 | 31701848 |
| ours, moving | 128 | 354 | 13.9 s | 0.0580 | 0.0181 | 0.0147 | 31701848 |
| ours, moving, per-face loop | 256 | 918 / 923 | 287.2 / 288.5 s | 0.0341 | 0.0093 | 0.0088 | 31701847 / 31702077 |

Equal accuracy (L1 rho): LUNA's GPU time interpolated linearly in log(t)
against log(L1) between its two neighbouring resolutions:
- 0.0341 (ours moving 256^3): LUNA 448^3 reaches it exactly, 932.8 s
  against our 288 s: the moving mesh needs 3.2x less GPU time (its v_r
  error is lower too, 0.0093 vs 0.0114; P equal).
- 0.0580 (ours moving 128^3): LUNA 19.7 s (between 192^3 and 256^3)
  against our 13.9 s: 1.4x less.
- 0.0750 (ours static 128^3): LUNA 3.2 s (between 128^3 and 192^3)
  against our 31.8 s: the static mesh needs 10x more.

Two moving 256^3 runs of the same build took 918 and 923 steps (jobs
31701847, 31702077): the moving-mesh run is not bit-reproducible run to
run (the static one is, CTest); the cause is not isolated (the mesh
update is the candidate). Their accuracy agrees to the digits shown.

## The 256^3 moving run with the per-face loop (2026-10-31)

What grew: the hydro's scratch (gradients 120 B/cell and face fluxes
318 B/cell) grows the mesh's spare ring pool to 438 B/cell. When the
appended rings overflow the pool in use, `tet_mesh::update_face_rings`
compacts them into the spare pool and then grows the emptied pool to the
size of the one in use, "the next spare: as large as this one" (the
library before this change): both pools at 438 B/cell, the second
allocation of 6.847 GiB fails. Fix (commit 7082aad, `tet_mesh.hpp`): when
the emptied pool is the smaller one and holds the rings with a quarter to
append, the compacted rings are copied back into it and the grown pool
stays the spare (one device copy of the ring entries per compaction);
otherwise it grows as before. With pools of equal size (no scratch growth)
nothing changes. CTest `hydro`: after 60 moving steps with the per-face
loop the mesh equals an exact rebuild of the snapped points.

Same job (31702077), the merged library without (BEFORE) and with (AFTER)
the change, moving 256^3, per-face loop, whole Sedov:

| | BEFORE | AFTER |
|---|---|---|
| result | "failed to allocate 6.847 GiB (face_rings_pool)" | 923 steps, t_end reached |
| Kokkos peak, mesh build | 2166.8 B/cell | 2166.8 B/cell |
| peak after the hydro set-up | 2230.8 B/cell | 2230.8 B/cell |
| peak over the steps | (failed) | 2418.6 B/cell, 42.15 GB of 42.40 GB in use; ring pools 688.5 B/cell |
| median step, first 100 steps (job 31701846) | 298.5 ms | 298.6 ms |

The whole moving 256^3 Sedov with the per-face loop takes 288 s of GPU
phases (median step 308 ms: time step 23.0, point velocities 24.3,
gradients 61.7, fluxes 52.7, gather 46.2, drift 0.9, mesh update 49.3,
update_cells 50.3), against 364 s with the per-cell loop (job 31701234).
761 stuck triangles over the run, finished by the host fallback (382 host
flips), no rebuild.

## Open items

- The face areas at the start of the step and the new volumes: GCL only to
  O(dt^2) for non-uniform point velocities; AREPO-ng's two passes average
  old and new geometry. Measure a uniform state on a strongly regularised
  (uniform random) moving mesh.
- The point-velocity kernel (3.3 ms) and the time-step kernel (2.7 ms) both
  loop over the face neighbours; they could share one pass.
- (Done: the per-face loop fits a whole moving 256^3 run since 7082aad,
  see above; 0.25 GB to spare.)
- Moving runs are not bit-reproducible run to run (918 vs 923 steps at
  256^3); find where the mesh update or the hydro loses determinism.
- The radial profiles of the moving mesh have empty innermost bins (the
  cells have moved out of the hot centre, as in a Lagrangian scheme); the
  plots show them as zero (`reports/hydro_stage3_*.png`).
- The static 256^3 Sedov needs more than one 15-min job.
