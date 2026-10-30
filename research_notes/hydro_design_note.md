# Hydro on the Voronoi mesh: design note (draft, 2026-10-30)

Written with Ulli. Decisions are marked **decided**, open points **open**.

## Goal and yardstick

A moving-mesh finite-volume code on the exact Voronoi faces of voronoi-cpp, one A100,
double. It is judged by accuracy per GPU-second against LUNA (fixed grid and AMR) on
Kelvin-Helmholtz and a moving blast wave, not by raw cell updates per second: the mesh will
never match fixed-grid throughput (LUNA: 1.1 G cell updates/s).

## Division of work (**decided**)

voronoi-cpp handles all the mesh work; the hydro code never sees tetrahedra, flips, rings or
clipping. The mesh gives, per face, owner, neighbour, area vector and centroid; per cell, the
volume (and later the centroid); and one call that takes new point positions and returns an
exact face list again (repair or rebuild decided inside). The hydro code owns the cell states,
gradients, fluxes, the update and the time step, and in stage 3 the point velocities.

## Stages

1. **Static mesh, MUSCL-Hancock, HLL** (**decided**, amended 2026-10-30: MUSCL-Hancock as in LUNA,
   not forward Euler). Points do not move. AREPO-style least-squares gradients of the primitives
   with a slope limiter, half-step prediction, HLL flux on every face, global time step; the
   zero-gradient case (first-order Godunov) is kept as a test path.
   This tests the mesh geometry, the face loop, conservation and the time step, with nothing
   else that can be wrong.
2. Static mesh, second order: AREPO-style gradients (**decided**: least squares over the face
   neighbours, Springel 2010 / Pakmor et al. 2016 improvement, slope limiter at face
   centroids), MUSCL-Hancock half step; HLLC.
3. Moving mesh: points move with the flow plus a regularisation term; face velocities from
   the point velocities; fluxes in the face frame; per-step tetrahedra repair (already on GPU).
4. Comparison against LUNA: KH, moving blast wave, accuracy per GPU-second.

## Stage 1 in detail

**Mesh input** (**decided**): the faces of the persistent tetrahedra (`tet_mesh`), not the
clipping output. Each face is stored once, with owner, neighbour, area vector (owner to
neighbour) and centroid; each cell has a volume. Exact topology means every face is seen
from both sides and the sum of area vectors of a cell closes to rounding. Gate: the exact
starting tetrahedralisation (agent at work) passes on lattice, jittered, uniform, clustered.

**State**: conserved U = (rho, rho v, E) per cell, structure of arrays, double.
5 doubles per cell, plus a second copy or an accumulator for the update.

**Flux** (**decided**: HLL). Computed directly along the face normal n = A/|A|:
v_n = v.n, F = (rho v_n, rho v v_n + P n, (E + P) v_n), wave speeds
S_L = min(v_nL - c_L, v_nR - c_R), S_R = max(v_nL + c_L, v_nR + c_R) (Davis), so no
rotation into a face frame is needed. The flux takes a face velocity w from the start
(w = 0 in stages 1-2), so stage 3 does not change its signature: the Riemann problem is
solved in the frame moving with w and the flux carries -w U.

**Update loop** (**decided**: bit-wise reproducible). Results must be identical run to run,
so no floating-point atomics in the update. Two candidates, measure both:
- (b) one thread per cell over its faces (needs a per-cell face list, ~16 ints per cell);
  each flux is computed twice, once from each side, with the same arguments in the same
  order so both sides get the same bits.
- (c) one thread per face writes its flux into a face array, then a per-cell gather in a
  fixed face order: one flux per face, 5 extra doubles per face (~40 doubles per cell, since
  faces are stored once at ~8 per cell).
Pick by speed; (c) is the default guess (one Riemann solve per face).

**Time step**: global, dt = C min_i r_i / (c_i + |v_i|), r_i = (3 V_i / 4 pi)^(1/3),
C = 0.3 to start. Equation of state: ideal gas, gamma a parameter.

**Code location** (**decided**): `src/hydro/` inside voronoi-cpp, header library like the mesh,
with its own CTest specifications and a `hydro_bench`/run tool.

**Snapshots** (**decided**): HDF5 in LUNA's layout, so LUNA's tools read them with small
changes: `snapdir_NNN/snapshot_NNN.0.hdf5`, group `cells` with attributes `time`, `step`,
`gamma`, `spatial_dimension`, `Lx/Ly/Lz`, plus `mesh = "voronoi"` and `N_cells`; datasets
1-D over cells: `x, y, z` (generating points), `volume`, `rho, vx, vy, vz, P, e_int`. LUNA's
reader stitches grid shards by `x_start`, so it needs a branch for `mesh = "voronoi"`
(unstructured, no Nx/dx); comparisons on grids go through a deposit/interpolation step.
HDF5 from Raven's module (serial is enough on one GPU).

**Boundaries**: periodic only (the mesh is periodic). Others later.

**What a step costs**: the mesh does not change, so a step is one pass over ~8 faces per
cell (16 faces per cell, stored once) plus one pass over cells. Should be far faster than the
mesh step; measure it to know what the moving mesh adds.

## Throughput: every stage and every option measured (**decided**)

Every stage, and every alternative inside a stage (update loop (b) vs (c), stored variables,
gradient variants, time integrators, regularisation), is measured in **cell updates per
second** on the A100, double, at 2.1M cells (jittered, uniform random, clustered), with the
step split into phases: mesh (when it moves), gradients, fluxes, update, time step. Reported
next to LUNA fixed grid (1.1 G/s), PROTEUS (5 M/s), AREPO-WD (0.59 M/s), GIZMO GPU (0.34 M/s)
on the same problem where one exists. A choice is made by these numbers; a variant that is
slower without a measured accuracy gain is dropped. The hydro tool prints them the same way
`voronoi_bench` does (median of repeats, one line per phase), so before/after comparisons are
interleaved within one job.

## Local time steps and active cells (requirement for stages 3-4)

With hierarchical time steps only the active cells are updated in a substep. AREPO's next
generation (`Arepo2-nextgeneration-SoA`, `run.cc:145`, `voronoi.cc:34-52`, verified
2026-10-30) rebuilds only cells up to `HighestSynchronizedTimeBin` with its per-cell clipper,
which keeps its cost proportional to the active cells. Our repair is 26-120x faster than a
per-cell rebuild per full step (2.1M cells, A100), but today its face, ring and geometry passes
run over the whole mesh. On a deep hierarchy (5-10 % active) that would cut our lead to about
10x. Requirement: the mesh update, faces, gradients and fluxes run over the active region
(active cells, their faces and the tetrahedra touched by moved points) only, with cost
proportional to it, and inactive cells keep their geometry until they are active again.
Measure it in cell updates per second per substep against a full step.

Option (Ulli, 2026-10-30): in substeps, rebuild only the active cells locally by clipping (warp
kernel, 12-35 M cells/s on the A100) instead of repairing; the persistent mesh is repaired at
synchronisation points. Faces between an active and an inactive cell follow the owner rule.
Crossover with today's whole-mesh repair (~178 M cells/s per full mesh): local rebuild is
cheaper below about 10-20 % active cells; an active-only repair would move that line. Requires
the clipping to keep its face output (owner, area, centroid), sized like the ring pool rather
than 32 slots per cell (cleanup decision D1 amended).

## Validation for stage 1

- Conservation: mass, momentum, energy to rounding every step (periodic box).
- Uniform state on an irregular mesh stays exactly uniform (fluxes cancel because faces close).
- Sound wave convergence (first order: error ~ N^(-1/3)), on jittered and on uniform random
  points.
- Sod shock tube in 3D (plane-parallel on a jittered mesh) against the exact solution.
- Sedov 128^3 against the analytic solution and LUNA, with s/step next to PROTEUS, AREPO-WD,
  GIZMO GPU (same A100).
- Reference data: luna-reference-solutions (`reference_path`), exact Riemann solutions.

## From LUNA

LUNA is JAX (Python); there is no code to link. Reuse: the HLL and HLLC formulas and their
wave-speed estimates (src/luna/hydro/fluxes.py, `_hll_core`, `_hllc_core`), the problem set-ups
and test metrics (scripts/validation), the reference solutions repository. Same problems
with the same parameters make the comparison direct.

## Decided 2026-10-30

Static mesh with MUSCL-Hancock and HLL first (as LUNA; amended from first order); bit-wise reproducible update; code in `src/hydro/`; HDF5
snapshots in LUNA's layout; AREPO-style least-squares gradients for stage 2.

## Still open

- Mesh regularisation for stage 3 (AREPO's centroid steering): after the static stages.

Decided as well: time integration is MUSCL-Hancock throughout. The GPU state between steps is
the conserved variables only (5 doubles per cell); primitives are computed where needed and
not kept in GPU memory, as in LUNA. Snapshots write primitives.
Reconstruction is always on the primitives (rho, v, P): gradients, limiter, extrapolation to
the face centroids and the MUSCL-Hancock half step act on W, fluxes and the update on U.
