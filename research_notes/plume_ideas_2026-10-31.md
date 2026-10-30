# Ideas from Doug Rennehan's plume for voronoi-cpp (2026-10-31)

Source: plume (private, github.com/rennehan/plume, Rust + cuda-oxide),
read-only clone `/raven/u/uli/plume-bench/plume` at cf55509 (2026-10-30).
Mainly `crates/partitions/local_voronoi` (README with its table of
decisions and measurements, `src/`), also `neighbors`, `partition` and
the commits since 2026-10-24. Ideas taken from it should credit Doug
Rennehan. Nothing was run for this note. Each "expected gain" is an
estimate unless it says "measured". plume's numbers are from its own
runs (Quadro RTX 6000, thread per cell, 3D Sedov 64^3 unless stated). I
did not reproduce them.

## How plume's situation differs from ours

| | plume `local_voronoi` | voronoi-cpp |
|---|---|---|
| Parallelism | one thread per cell, blocks of 32 threads (`src/pass.rs:8`); cell in registers and local memory (~11 kB, 255 registers, README l. 60) | one warp per cell in shared memory (4.5 KB), 96-register cap; per-thread kernel only as fallback, exact patch and CPU path |
| GPU | Quadro RTX 6000: FP64 at 1/32 of FP32, so double divisions and square roots cost far more than on an A100 (FP64 at 1/2 of FP32) | A100, latency- and divergence-bound (cuts note: 39 % of cycles with no eligible warp) |
| Inputs | each row's own SPH neighbour list (~51 entries), nearest first from a 16-entry batch (`cell/nearest.rs:5,42-65`) | adaptive cell list, warp-sorted batches of 32 |
| Exactness | each cut decision is the exact sign of an in-sphere test. An exact zero removes nothing (no symbolic perturbation); degenerate points become zero-length edges, and faces of near-zero area may exist on one side only (README l. 69) | the warp clipping uses rounded decisions. The start is made exact afterwards (check, then exact clipping with symbolic perturbation, SoS), and the repair is exact with SoS |
| Grid | 2^-48 of the box (`exact/grid.rs:30`), offsets up to 2^47 | 2^30, offsets below 2^29 |
| Output | fixed 32 face places per row, ~0.9-1.7 kB per particle, each face stored by both rows | owner-rule face pool (701 B/cell for the plain rebuild) or persistent mesh faces |

So any idea that rests on "FP64 is very slow" will gain less here. Any
idea that rests on "one thread holds the whole cell" applies to our
per-thread kernel, not to the warp kernel.

## Ranked candidates

### 1. Vertex adjacency and slot reuse in the per-thread kernel

- **What plume does.** It applied this change to its own per-thread
  kernel. Each vertex holds the three vertices across its edges
  (`cell/polyhedron/cell.rs:30`). A cut finds its hole edges through
  those links (`clip.rs:115-138`). Removed slots go on a free stack and
  are reused (`clip.rs:144-162`), so no vertex moves. Faces are walked by
  one lookup per step (`faces.rs:49-73`). Commit cc3a81a.
- **Measured in plume.** "Finding the hole's boundary and each face's
  next vertex by searching every vertex was about 45 % of the build's
  stalls with about 2 of 32 lanes active". Builds took 2.61 s instead of
  5.60 s on 64^3 Sedov to t = 0.02 (README l. 102). They rejected two
  alternatives: compacting vertices after every cut, and a 48x48
  directed-edge table, which was 60 % slower.
- **Our code.** The warp kernel already has this structure (cuts note,
  commits 1-2). The per-thread kernel (`shell_voronoi.hpp`; RoundedCell
  and ExactCell share it) still does the old version of all three steps:
  - It matches hole edges against their twins by a quadratic search
    (`shell_voronoi.hpp:593-616`).
  - It copies every kept vertex in every cut (`:622-659`).
  - Its face walk searches all vertices to find a face's start (`:822`)
    and to find every next vertex (`:842`). That costs O(V) per face
    edge, and V is up to 256 in the big pass.
- **Where it would help.**
  - The warp fallback passes: uniform 7.1 + 6.6 ms of ~127 ms; halo
    9.0 + 7.4 ms of ~174 ms (cleanup2 note). Both passes wait for their
    slowest cell. The big pass (128 planes, 256 vertices) is where the
    O(V) searches cost most.
  - The exact start of degenerate input. At 262k cells (logs
    `build-cleanup2/logs/chk-*`), lattice takes 43 ms against a 5.5 ms
    clip, and wave 80-105 ms against a 7.6 ms clip. Every cell is
    patched by ExactCell clipping in both cases.
  - Local `clip(x, cells)` with rings, which runs the per-thread kernel.
  - The D2 device-rebuild fallback: 0.17 s for the wave's first step at
    2.1M.
  - The CPU/OpenMP backend.
- **Expected gain.**
  - Jittered rebuild: none (it has no fallback cells).
  - Uniform, clustered and halo rebuilds: about 2-6 %, if the fallback
    passes speed up by about 1.5-2x.
  - Lattice and wave starts: about 1.3-2x on the exact-clipping part.
    I have not measured how that part splits between face walk, cuts and
    in-sphere tests. On lattices the exact in-sphere ties may dominate
    instead (see idea 6).
- **Risk.** Exactness: none; decisions are unchanged and only the
  topology bookkeeping changes. Rounding can make a hole touch one face
  twice in RoundedCell; plume flags this as `OPEN_FACE` (`clip.rs:250-264`,
  test `hole_touching_face_twice_is_open`), and we would need the same
  check. Memory: 3 bytes of adjacency plus 1 byte of free-slot stack per
  vertex in local memory, about 1 KB extra for the big cell.
- **Effort.** Medium. The warp kernel's `clip()` is a template to follow,
  but the per-thread version must stay generic over RoundedCell and
  ExactCell, and `compact_planes` must keep working.

### 2. Warp-kernel face output without per-triangle divisions, with pyramids per face

- **What plume does.**
  - e4ccbfb: "face centers, centroids and volumes divide once per face
    or cell instead of once per triangle". On the RTX 6000, "fp64
    divisions by 3 and 6 were 80 % of its stalls". Measured 32.3 s ->
    12.9 s, together with two other changes in the same commit.
  - 6cbb429: each fan triangle's area vector is parallel to the
    bisector normal, so the face uses one component of it, along the
    normal's largest axis (`faces.rs:21-24, 61-64`). The face's area
    vector, centre and pyramid weight then come from two sums per face
    (`faces.rs:94-108`). The cell volume is
    Σ_f 2·offset_f·(projected area_f / n_axis), and the centroid is the
    volume-weighted 3/4 of the face centres (`moments.rs:19-32`,
    `build.rs:81-99`). No tetrahedron is formed per triangle. Measured
    12.9 s -> 8.7 s.
- **Our code.** `shell_voronoi_warp.hpp:497-507` does, per face edge, two
  double divisions (`w / 3.0`, `/ 6.0`), one cross product and dot for
  the tetrahedron, and three moment updates. The per-thread kernel does
  the same (`shell_voronoi.hpp:850-857`). Face output is 21 % of warp
  cycles on jittered, 16 % on uniform and 9 % on clustered (cuts note,
  after the cut work).
- **Proposal.**
  - (a) Accumulate `w` and `tetra` undivided, and multiply by 1/3 and 1/6
    once per face (or once per cell). This is trivial and needs no
    memory.
  - (b) Compute the volume and centroid from pyramids per face:
    V_f = |A_f|·|d_f|/6, centroid 3/4·c_f. This is what our
    `tet_mesh::cell_geometry` already does (`tet_mesh.hpp:723-729`). It
    drops the per-edge tetrahedron entirely. It needs |d| in double per
    plane: one more `double[64]` (512 B) in the Polyhedron, which takes a
    block from 18 to 20 KB. Shared memory is not the occupancy limiter
    (cuts note: 18 KB would allow 8 blocks). Alternatively, reload the
    neighbour's position once per face.
  - Do not adopt plume's projected area vector here. Our plane normals
    are float (cuts commit 5), so the area vector would be float-accurate.
    Keep the cross-product area and use the projection only for the
    centroid weight, as we do now with `dot(twice, normal)`.
- **Why it may help.** Fewer instructions and registers in a phase that
  is a fifth of the jittered cycles. Lower register pressure matters at
  the 96-register cap; the cut work showed that small chain shortenings
  gave 2-7 %. On the A100, though, a double division is a short FMA-based
  sequence, not the stall source it is on the RTX 6000.
- **Expected gain.** (a) about 1-3 %; (a)+(b) about 3-8 % on the
  jittered warp rebuild, less on uniform and clustered. Both are
  unmeasured.
- **Risk.** Exactness and topology: none. Volumes and centroids change
  in the last bits. The cuts note's `cut_dump` comparison (relative
  volume differences ~1e-15) is the check. Memory: +512 B shared per
  warp with (b).
- **Effort.** Small: (a) in an hour, (b) in a day including the A/B job.

### 3. Step geometry: no square root or division per ring triangle

- **What plume does.** As in idea 2 (6cbb429, `faces.rs:21-24`): one
  component along the normal's largest axis serves as the triangle
  weight, and there is one division per face.
- **Our code.** In `tet_mesh::faces` (`tet_mesh.hpp:676-685`), each ring
  triangle takes `sqrt(dot(twice, twice))` and `w / 3.0`. Each face
  takes `sqrt(dot(area,area)·dot(dj,dj)) / 6.0` (`:695`). `update_cells`
  takes the same square root and /6 per list entry (`:729`). The faces
  kernel is about 2.9-3.0 ms of the 11 ms step. Our speed note already
  found it "partly bound by double divisions": replacing three divisions
  in `wrap` gave 3.37 -> 2.86 ms.
- **Proposal.**
  - Triangle weight w = twice[k]·sign(dj[k]), where k is the largest
    axis of dj. This is exact for planar rings. For rings made non-planar
    by rounding of the circumcentres, the centroid weights are off only
    by rounding.
  - Divide by 3 once per face.
  - Pyramid volume as |A·dj|/6, without the square root. A ∥ dj up to
    rounding, and the sign is already fixed by the `dot(area, dj) < 0`
    flip.
  - Keep the full cross-product area vector. plume's projected area
    (area ∥ d by construction) would not close exactly over non-planar
    rounded rings; the cross-product sum does, up to summation rounding.
    Our hydro needs closure to rounding.
- **Expected gain.** About 0.1-0.5 ms per step (1-4 % of the 11 ms step
  at 0.03 spacings), plus a little in `update_cells`. Unmeasured.
- **Risk.** Face centroids and volumes change in the last bits. The
  `update_cells` bit reproducibility (run to run) is unaffected.
- **Effort.** Small.

### 4. Exact arithmetic out of the repair's find and plan kernels

- **What plume says.** On a glass, the exact arithmetic compiled into the
  kernel caused most of the 10 % cost of exactness: it "raises every
  row's registers and spills though no row uses it" (README l. 62). The
  open question is to build cells that need exact arithmetic in a second
  kernel, flagged by the first. On 32^3 Sedov, the build without the
  exact code ran within 6 % of the build before exactness (README
  l. 120). plume has not implemented this.
- **Our code.** `tet_repair::find` calls `ep::in_sphere` inline
  (`tet_repair.hpp:501`), and `plan` calls `detail::plan` (`:195-246`).
  Both inline the 256-bit Wide arithmetic and the symbolic-perturbation
  orient loop. find and plan use 132-152 registers (speed note). Repair
  rounds take 2.2 ms per step at 0.03 spacings and ~20 ms at 0.3, in
  15-19 rounds.
- **Proposal.** First measure: build find and plan with the exact branch
  replaced by a stub, and compare registers and time. If they drop a lot,
  run the double filter in find and plan, list the undecided predicates,
  and decide them exactly in a small second kernel. Do not use
  `noinline` calls: our own trials of `noinline` cost 10 % (warp output)
  and 3.3 % (binary search).
- **Expected gain.** Unknown until measured. If registers fall to
  ~64-80, perhaps 10-30 % of those kernels' time: up to ~0.5 ms per step
  at 0.03 spacings and several ms at 0.3.
- **Risk.** Exactness: none if every undecided case goes to the exact
  kernel. It adds a launch per round and another code path in the flip
  logic.
- **Effort.** Medium, with a cheap first measurement.

### 5. Certified cut decisions in the warp kernel that records rings

- **What plume does.** Commit 88a374d.
  - Each cut computes one coarse bound for all its vertices:
    live_bound·|n|_1 + slack (`clip.rs:33-45`). Vertices inside it get
    their own bound (`clip.rs:62-95`), and only vertices inside their own
    bound go to the exact in-sphere test (`predicate.rs:280-291`). The
    exact test re-reads the offsets by slot rather than storing per-plane
    integers.
  - New vertices are solved from their three planes. The error bound
    comes from the absolute terms over the determinant
    (`new_vertex.rs:60-86`), so it does not compound from cut to cut. When
    that bound exceeds the hole edge, the vertex is placed on the edge
    instead (`clip.rs:225-237`, `new_vertex.rs:88-106`).
  - The radius bound only shrinks (`clip.rs:267-269`).
- **Measured in plume.**
  - Placing every vertex on its edge compounded the bound about fourfold
    per cut, from 3e-14 to 1e-5 of the cell over 18 cuts (README l. 112).
    Our warp kernel interpolates along the edge
    (`shell_voronoi_warp.hpp:237`).
  - Exact decisions cost about 10 % on a glass and 1.86x against kernel
    faces on the lattice sound wave (README l. 62).
- **Where it would help.** Only the start, and only on input where the
  rounded rings are wrong:
  - Jittered, uniform and halo already give 0 bad tetrahedra and one
    check round, at 262k and 2.1M (`build-cleanup2/logs`). For them this
    gains nothing.
  - Clustered at 262k: 2 bad tetrahedra, 48 cells patched, 100 ms
    against a 64 ms clip.
  - Lattice at 262k: every cell patched, 43 ms against a 5.5 ms clip.
  - Wave at 262k: every cell patched, 80-105 ms against a 7.6 ms clip.
  - Also the D2 device rebuild out of degenerate steps.
- **Expected gain.** Lattice, wave and clustered starts could approach
  clip + `from_rings` + check (about 15-20 ms at 262k), but only if
  in-kernel exact ties are cheap (idea 6). The plain rebuild should not
  be touched: plume attributes its glass cost to registers, and we are
  at the cap. Instantiate this only in `Kernel<..., with_rings = true>`.
- **Risk.**
  - Ties must be broken by our symbolic perturbation (`ep::in_sphere`
    with ids), not by plume's "zero removes nothing". Otherwise the rings
    are not the perturbed Delaunay tetrahedralization.
  - The bounds must be rigorous for the whole chain: the float candidate
    key, float normals and interpolated vertices. In practice this means
    adopting the 3-plane solve.
  - Keep `tet_start`'s check as the guard.
  - Memory: a bound per vertex (512 B shared per warp).
- **Effort.** High. Do it after ideas 1 and 6, which make the
  alternative (patching by ExactCell) cheaper anyway.

### 6. A 64-bit exact path after dividing out common powers of two; no second filter on ties

- **What plume does.**
  - Before an exact sign, it shifts out the smallest trailing-zero count
    of all coordinates. If the largest coordinate is then below 2^10, it
    decides in plain i64; otherwise it uses 320-bit integers built from
    128-bit products (`predicate.rs:169-189, 245-262, 280-291`).
  - Measured: on its lattice wave (32^3), 128-bit products against
    32-bit limbs gave 52.5 vs 17.6 s.
  - A second floating-point filter on the in-sphere determinant "decided
    none of the wave's rounding-level decisions and slowed its run from
    52.5 to 85.7 s" (README l. 111).
- **Our code.**
  - ExactCell's `inside()` filters with w and o (`shell_voronoi.hpp:490`)
    and then calls `ep::in_sphere`. That runs `orient`, then a second
    double filter (four `det3_float` with permanents), then 256-bit Wide,
    then up to five SoS orients (`exact_predicates.hpp:137-183`). On exact
    ties, which are the lattice case, both filters always fail.
  - `set_vertex` uses i128 crosses for every new vertex
    (`shell_voronoi.hpp:454-470`).
  - Our 2^30 grid holds power-of-two lattices exactly. Lattice offsets at
    128^3 are multiples of 2^23 steps and would reduce to small integers.
- **Proposal.**
  - (a) Add an entry to `in_sphere` that skips its double filter when the
    caller's filter has just failed.
  - (b) Add a reduced 64-bit path, plume's `reduced()` and
    `small_outside()`, for in_sphere, orient and the SoS orients.
- **Expected gain.** Only for exactly degenerate input that shares powers
  of two: sc, bcc and fcc lattices with N = 2^k (lattice start: 262k
  43 ms, 2.1M ~230 ms of exact clipping). Wave and perturbed lattices
  (sc@1e-6) get (a) at most, because their offsets do not share powers of
  two. Jittered, uniform and the steps: nothing measurable, since their
  filters decide.
- **Risk.** It needs tests that the 64-bit and wide paths agree, as
  plume's `small_and_wide_paths_agree` and
  `common_powers_of_two_are_divided_out` do.
- **Effort.** Small to medium.

## Checked and not proposed

- **Blocks of one warp for partial builds** (README l. 107: 7.06 vs
  8.70 s for blocks of 32 vs 256). This fixes plume's thread-per-cell
  tick of a few hundred cells. We already build a warp per cell with
  persistent warps, so a small selection already spreads over the SMs;
  one cell per block would not shorten a cell's latency. plume names a
  warp per cell as its own next fix (README l. 60).
- **Shape in single precision, measured in double** (README l. 60, a fix
  plume's precision rule forbids). It pays off on FP64-poor GPUs. On the
  A100 we are latency-bound, and fast-math was already rejected.
- **The 2^-48 grid.** Its purpose is to keep lattice degeneracies exact.
  Our 2^30 grid already does that for power-of-two lattices. Its 1-step
  near-ties sit about 2^-23 relative to the terms, far above the filters'
  ~2^-48 error bounds, so the double filter decides them.
- **Nearest-first batch from a list scan** (`nearest.rs`). Already tried
  in our per-thread kernel and rejected (3-15 % slower).
- **The per-row face table, found by scanning for the neighbour row.**
  It holds 32 places per row (0.9-1.7 kB per particle) and stores every
  face twice. We moved from 32 slots (1792 B/cell) to a pool (701 B/cell).
  For hydro it would also give two roundings of one face, which plume
  settles by always reading the canonical row's face (`partition/src/
  implementations/cell_faces.rs:17-24`); our owner rule already does that.
- **Evaluating only pairs that share a face** (6e469a4: 70 % of pair
  evaluations had no face; 1.32 -> 0.35 s). This is specific to hydro
  over SPH neighbour lists. We iterate faces.
- **Neighbour search** (`neighbors` README). The tree with 32-row
  buckets, active-row clusters and transposed lists for touched rows
  (47 -> 9.4 ms at 1/1000 active) targets SPH lists of sparse active
  rows. Our adaptive cell list already brings clustered and halo input to
  ~51-54 tests per cell. It might matter for the local substep mode at
  very low active fractions; that is unmeasured, so it is not ranked.

## Not speed, but worth taking for the hydro

- **Clamp each face centre to its vertices' bounding box, and drop faces
  of exactly zero area** (`faces.rs:94-103`, README l. 114). On their 64^3
  sound wave, a face cut down to an edge put its area-weighted centre
  outside the cell. Through the moving-face correction this collapsed the
  timestep. Our `tet_mesh::faces` falls back to `first` only when
  `weight == 0`; near-zero weights can still give such outliers.
- **The moving-face frame correction from the face centre's offset from
  the pair midpoint** (Springel 2010 eq. 33; `cell_faces.rs:35-46`).
  Measured on their 128^2 Kelvin–Helmholtz run: 121 vs -3.8 vs 122 mean
  cell areas swept per unit time (README l. 104). We already store face
  centroids, so hydro stage 3 needs nothing new from the mesh.
