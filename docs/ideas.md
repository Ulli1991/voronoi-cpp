# Ideas for a faster, leaner and more accurate mesh

Every idea from the research review of 2026-11-08, checked on 2026-11-09 against the code and against what Raven has installed. Nothing is dismissed: each idea has what it could buy (speed, memory, accuracy, robustness), a first test that decides whether to go on, and a status. Hydro ideas are in luna-cpp's `docs/ideas.md`, with links where an idea spans both.

Where we stand, for scale: one A100, double precision, 2.1M cells. Full rebuild 19.7 M cells/s on uniform random points and 39.4 on jittered ones; exact start 17.7; moving step with cell lists 154 M cells/s at 0.03 spacings. Memory 2319 B/cell with the hydro, so about 256³ fits on a 40 GB A100, against a target of 384³ (about 720 B/cell). We think both speed and memory are still well short of what the hardware allows.

Status: **done** (in the code), **next** (first in line), **open** (a real candidate, not started), **research** (bigger, uncertain).

## Measure first

| ID | Idea | Buys | First test | Status |
|---|---|---|---|---|
| A | Cost attribution per cell: candidates tested, planes kept, clips, vertices, topology edits, repair rounds, scratch size, register spills, bytes written; tails, not only means; Nsight Compute stalls and divergence | decides every speed item below | histograms on jittered, uniform, clustered, halo and anisotropic input | next |
| R3 | The gap between what we store and what the hydro needs: neighbours, area vectors, face centroids, volumes, centroids, face motion | memory | list every array by B/cell and its reader (the memory profile has most of it, #28) | next |
| R9 | Time to solution at fixed error as the final score, not cells/s | right decisions | already used in luna-cpp; apply to every mesh change that affects the hydro | done, keep |

## Speed of construction (rebuild and start)

| ID | Idea | Buys | First test | Status |
|---|---|---|---|---|
| B | Candidate order: nearest first (ours), best-first with geometric bounds, directional support bounds, predicted cut strength; previous neighbours first so the cell shrinks early | fewer clips, mostly on clustered input | V3's spread test below | open |
| V3 | Learned or tuned clipping order: offline orders, derivative-free tuning first, then differentiable ranking (NeuralSort, Blondel et al.), Enzyme on the scoring surrogate only | fewer clips | shuffle the same candidates, measure the spread in work; drop it if the spread is small | open |
| R8 | Bounds that prove a bisector cannot cut the current cell: support function or directional box, cascaded before the exact clip, as in Taveira et al. 2026 | fewer clips, clustered input | microbenchmark: rejection rate, false positives, cost per test against a clip | next |
| C | Cell representation: explicit vertices and edges (ours), plane-index triplets, compact fixed capacity, a hybrid for small and large cells; same plane sequence for all | speed, registers | replay recorded plane sequences through each representation | open |
| O7, R7 | Kernel portfolio: a lean kernel for ordinary cells and a separate queue for the large tail; classifier cost included | speed on clustered input | needs A's tail histogram; then end-to-end time with dispatch | open |
| F | Execution: thread per cell against warp or block per cell (ours: warp), buckets by last step's cell size, spatial reordering, face-centric against cell-centric output | speed | one variable at a time on the same inputs | open |
| T5 | Autotuning of launch bounds, warps per cell, pool sizes, batch sizes and precision thresholds (Kernel Tuner, Optuna or our own), with correctness as a hard constraint (R5) and held-out inputs | speed | a harness over two or three knobs first | open |
| D | No clipping: build the start as a Delaunay triangulation directly (a GPU star or walking method, a fresh look at DeWall), then take the Voronoi faces from it; compare all outputs the hydro needs | faster start; the start is our slowest path (11.7 M cells/s at 256³) | the start's phase costs, then a prototype on one input | research |
| R16 | Reusable topology templates: cache common connectivity patterns of cells and their evaluation order, with generated kernels for the frequent ones | less branching in the clip kernel | topology signatures on jittered and clustered input: how few templates cover how many cells | research |

## Speed of the moving step (repair)

| ID | Idea | Buys | First test | Status |
|---|---|---|---|---|
| E | Three paths: geometry-only refresh where topology is unchanged, local repair, full rebuild; validation and launch costs counted | speed | how much of a step goes to tetrahedra that did not change; we already re-walk only faces whose ring changed | open |
| V1 | Predict topology events: forward-mode derivative of the in-sphere test along the motion (Enzyme or by hand), time to the zero crossing; never a certificate, the exact test stays | could skip screening and refresh work on quiet tetrahedra | log predictions against real flips over many steps: recall, false positives, cost | open |
| O1 | Certified kinetic events: for straight-line motion each in-sphere test is a polynomial in time; isolate its roots exactly on the grid and schedule work by event | the same as V1, with a certificate | cost of keeping all certificates against today's screening pass | research |
| R1 | Repair driven by a queue of invalidated relationships, against today's rounds over lists | fewer launches, less duplicate work | today's step has 66 launches and 42 fences; measure what a queue saves | open |
| #34 | Re-sort cells, tetrahedra and faces when memory order has decayed | up to 2x on long smooth runs | sort cost against the measured halving of the step rate by step 500 | next |
| — | Faces numbered by owner, an owner bit in list entries (from the profile) | about +5 % whole step, +8 % hydro | measured with a copy in the hydro already | next |
| R13 | Voronoi geometry from the Delaunay dual: vertices are circumcentres, faces are the rings of circumcentres around Delaunay edges | no clipping in steps | done: steps already work this way; the open part is the start (D) | done |
| S2 | Second and third rings of neighbours, built on demand from the tetrahedra (not stored), as candidates to rebuild stuck or fast-moving cells on the GPU by exact clipping | replaces most host fallbacks (#33) and full rebuilds; local clips for active cells | how often cells get stuck or move about a spacing, and how often a 2- or 3-ring list holds every new neighbour (`Arepo2_WD` needs depth 3) | open |
| R12 | Predict new neighbours from relative positions and velocities before the step, to prefill candidate lists in hard regions | fewer repair rounds under strong deformation | share of new neighbours predicted, cost, list growth | open |
| R11 | Hydro-aware prediction: trajectories with the acceleration from the hydro, x + τw + τ²a/2, to rank event checks | better event estimates in shear and compression | linear against accelerated prediction, with real flips as truth | research |
| R15 | Memory order by Morton or Hilbert, or by the graph, possibly on predicted positions; cost of reordering counted | locality in repair and hydro (see #34) | static, periodic, predicted and graph orderings on a long run | next, with #34 |

## Memory

| ID | Idea | Buys | First test | Status |
|---|---|---|---|---|
| #28 | Ranked savings from the memory profile: float faces (197 B/cell), `tet_centre` float or not stored (125 to 250), repair lists (112), tetrahedron spare 1/32 (56), one position array (56), ring pool (42), idle cell list (41), 16-bit ring lengths (25); lower start peak | 384³ on one A100 | apply in that order, measuring B/cell and speed at each | next |
| R3 | Store only what the hydro reads; compute face geometry on the fly where it is cheaper than storing it | memory, maybe speed | the reader list from R3 above | open |
| R6 | Precision by condition: float where geometry is well conditioned, double or exact near uncertain decisions; topology is never approximate | memory and speed | error of float face moments against double across inputs, and the hydro error it causes | open |
| — | 64-bit offsets past 325³ (ring pool) and 508³ (cell faces) | correctness at large sizes | needed before 384³ | open |
| R14 | Fuse face geometry with the hydro's flux: compute a face's geometry and use it at once, instead of storing faces and reading them back | memory (face arrays are 197 B/cell each for area and centroid) and memory traffic | geometry write and read traffic in the profile, then a small fused face and flux kernel; MUSCL-Hancock needs cell gradients first, so full fusion may not be possible | research |

## Accuracy of geometry

| ID | Idea | Buys | First test | Status |
|---|---|---|---|---|
| #29, #30 | Fix the silent volume errors: geometry from exact integer differences times the step, and the `clip()` loss | correct volumes | in progress on branch mesh-fixes | next |
| T3 | Herbie on circumcentres, plane intersections, face areas and moments | accuracy of geometry in double, relevant to #29 | Herbie needs Racket, which installs in user space; one expression against an mpmath reference | open |
| O5 | Stochastic arithmetic (Verificarlo, CADNA) to find where geometry is sensitive to rounding | finds the next #29 | a run of the fuzz cases under random rounding | open |
| V2, V8 | Derivatives of vertices, face moments and volumes with respect to positions on a fixed topology; checked against hand derivatives and finite differences | sensitivity and checks, and the hydro's GCL test | Enzyme on the host build (see luna-cpp's ideas) | open |
| V7 | Margins of the predicates near zero, to see how often the filter falls to the exact path | speed of predicates | count filter failures per input | open |

## Correctness tools

| ID | Idea | Buys | First test | Status |
|---|---|---|---|---|
| T7 | `compute-sanitizer` memcheck, racecheck, initcheck and synccheck on the CUDA tests (in the CUDA 13.0 module) | finds GPU races and bad reads such as #35 | run it on the existing ctests | next |
| T1 | Metamorphic tests, exact on the 2³⁰ grid: shift by whole grid steps, axis swaps and 90° rotations, doubling the box, permuted ids | robustness without a reference | add to the special-case tests | next |
| T6 | Differential testing against an independent exact reference | robustness | `oracle.hpp`, special-case tests, the fuzz driver's rational arbiter | done, extend |
| T2 | Filtered predicates with an exact fallback | robust and fast | a double filter, then 128-bit or 256-bit integers | done |
| T2b | Prove the filter's error bound with Gappa | confidence in the fast path | one predicate's bound | open |
| T4 | SMT search (Z3) for counterexamples on small geometric models: the reach rule (#31), ordering and neighbour sufficiency | finds edge cases, as regression fixtures | Z3 installs with pip; the reach rule first | open |
| T7b | Alive2 on chosen LLVM transformations of the predicates | confidence in compiled code | only if a miscompile is suspected | research |
| O6 | Deterministic sums | reproducible bits | `update_cells()` sums in a fixed order | done |
| R2 | Each shared face computed once, both cells reading it | consistency and speed | the owner rule | done |
| R18 | Adversarial inputs by active learning: a model of which point sets make clipping slow, repair stall or geometry wrong, steering the fuzz generator | rare failures and a hard benchmark set | against the fuzz generators at the same budget | open |

## Between the libraries

| ID | Idea | Buys | First test | Status |
|---|---|---|---|---|
| O8 | A predictive interface: geometry, fixed-topology derivatives, event estimates and validity flags for the hydro | better time-centring and mesh motion | design only, after V1 and V2 show value | research |
| J1, O3 | Mesh motion that avoids needless flips while keeping accuracy | fewer repairs | see luna-cpp's ideas | research |
| J2, J3 | Regularisation aware of hydro error and of mesh cost | accuracy per GPU second | see luna-cpp's ideas | research |
| J4, O2 | Differentiable moving mesh; space-time cells | adjoints, exact time-centring | see luna-cpp's ideas | research |
| R19 | Recompute only what changed: track which geometry, reconstructions and fluxes a step invalidates | less work per step | the share of quantities unchanged between steps, as an upper bound | research |

## Benchmark discipline

For every speed claim: GPU and clock, compiler, precision, dimension, boundaries, inputs, number of points, mean and tail neighbour counts, which outputs are produced, correctness checks passed, and what is timed. End-to-end hydro time to a target error decides.

## Sources to check

voro++; Taveira et al. 2026 (arXiv 2605.06408); gDel3D; Ray et al. 2018; ParVoro++; kinetic Delaunay and certificate frameworks; differentiable Voronoi (arXiv 2312.16192); NeuralSort and Blondel et al.; Enzyme (no Kokkos support yet, CUDA experimental); Herbie and FPBench; Gappa; Z3; Alive2; NVIDIA Compute Sanitizer. "Local DeWall (2026)" was not found in a first search.
