# Ideas: what we will try, and why

A review of research ideas for the mesh library, 2026-11-09, checked against how the code works today and what Raven has installed. Hydro ideas are in luna-cpp's `docs/ideas.md`.

## Already done

| Idea | Where it is |
|---|---|
| Filtered predicates with an exact fallback | `exact_predicates.hpp`: a double filter with an error bound decides first, then 128-bit orientation or 256-bit in-sphere on the 2³⁰ grid |
| Shared faces computed once | the owner rule: each face is stored once, by the lower id, and both cells read it |
| Differential testing against an exact reference | `oracle.hpp`, the special-case tests, and the fuzz driver's rational arbiter |
| Deterministic sums | `update_cells()` sums in a fixed order, so runs give the same bits |
| Repair instead of rebuild for moving points | the persistent tetrahedra and GPU flips, about 10 times the rebuild rate |
| Geometry refreshed only where topology changed | only faces whose ring changed are walked again |

## Planned, in order

1. **GPU memory and race checking.** Run the CUDA tests under `compute-sanitizer` (memcheck, racecheck, synccheck; in the CUDA 13.0 module on Raven). Cheap, and the read at index -1 (#35) shows it is needed.
2. **Metamorphic tests.** On the 2³⁰ grid these can be exact: translation by whole grid steps, axis swaps and 90° rotations, doubling the box, and permuting particle ids must give the same mesh, transformed. Cheap, and needs no reference.
3. **Where clipping time goes.** A histogram of planes, vertices and time per cell, for jittered, uniform, clustered and halo input. This decides the next two items.
4. **Does clipping order matter?** Clip the same candidates in random orders and measure the spread in work. If it is small, drop learned ordering. If it is large, try cheap scores first.
5. **Bounds that skip a clip.** A directional test that proves a bisector cannot cut the current cell, as in Taveira et al. (2026). Aimed at clustered input, where we are slowest.
6. **Split kernels by cell size.** A lean kernel for ordinary cells and a separate path for the large tail, if item 3 shows a tail that dominates.

## Worth trying later

- **Autotuning** of launch bounds, warps per cell and pool sizes, scored on speed and on the exact checks together.
- **Proving the filter's error bound** with Gappa. Today it is a hand bound, which the exact fallback backs up, so a proof adds confidence, not speed.
- **SMT search (Z3) for counterexamples** to the reach rule (#31). Z3 installs with pip.

## Not now, and why

- **Predicting flips with derivatives (Enzyme) or kinetic certificates.** The step's screening pass is not where the time goes; the repair, faces and cell lists are. For straight-line motion the in-sphere test is a polynomial in time, so exact event times need no AD. Revisit if screening becomes a cost.
- **Replacing clipping by local Delaunay construction** ("local DeWall"). Steps already use a local Delaunay repair, and only the start clips. A 2026 paper by that name was not found.
- **Herbie, Alive2, stochastic arithmetic.** The decisions are exact integer predicates, so float rewrites cannot change topology. Herbie also needs Racket, which Raven lacks.
- **Space-time cells.** A research project of its own, for after the hydro needs it.
