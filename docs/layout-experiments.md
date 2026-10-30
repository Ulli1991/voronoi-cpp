# Candidate order and Morton cell-layout experiments

These are opt-in experiments toward the user's target of at least 10x faster
search and reconstruction. Neither has a measured GPU speedup yet. Relative to job31603868 at195^3,
the 10x targets are approximately31.5ms for search and89.9ms for reconstruction.
Removing all194ms of candidate sorting alone would still leave about121ms of
search and899ms of reconstruction; these experiments cannot be assumed to
achieve the target on their own.

| Scaling policy | Particle layout | Initial candidate order |
|---|---|---|
| relative_early_scratch_sort | Cartesian cell blocks | Full distance sort |
| relative_early_unsorted | Cartesian cell blocks | Search encounter order |
| relative_early_morton | Morton cell blocks | Full distance sort |
| relative_early_morton_unsorted | Morton cell blocks | Search encounter order |

All paths use complete radius queries and the same double-precision reference
reconstruction, including its conservative suffix check before early exit.
Morton keys are not distance bounds and never certify omitted neighbours.

## Morton scope and costs

Set `DenseCellSearch::morton_cells=true` before configure. This implementation
orders GRID CELL BLOCKS by a 64-bit Morton key; particles within each cell retain
deterministic input-ID order. It is not a fine-grained Morton sort of every
particle coordinate. A Cartesian-key-to-Morton-rank map preserves access to
all spatial query cells, including periodic images and wide stencils. Morton
queries use per-cell ranges where the Cartesian implementation can merge z
segments, so locality gains may be offset by additional range lookups.

The map costs four device bytes per grid cell and is cached across rebuilds and
candidate-capacity growth. Map construction uses a temporary host key/rank vector
and occurs once per grid shape during configure, outside steady-state benchmark
timing, as with the existing allocations. Rebuild/search timings include mapping
lookups and physical particle gather. The scaling memory estimate includes the
persistent device map. Existing maps/reference paths are retained.

## Unsorted result and explicit recovery

Simply disabling sorting FAILED CPU geometry validation: the deterministic12^3
jittered scaling fixture had568 invalid cells. Intermediate polyhedra exceeded
the64-vertex workspace; the table fixture observed up to118 clipping planes.
The regular lattice also exposed order-sensitive coplanar/degenerate faces.
This is useful negative evidence, not an accepted fast path.

The unsorted-first experiment now has an explicit recovery step:

1. Perform the existing reconstruction and plane-capacity retry.
2. For compact geometry, flag numerically tiny faces relative to the largest
   face (squared-area ratio <=1e-24) for reference-order retry. This does not
   remove faces or loosen predicates; legitimate tiny faces survive in the
   resulting sorted reconstruction.
3. Sort failed/order-sensitive candidate rows IN PLACE using double distances
   in team scratch, then reconstruct only those rows with the160-plane reference.
   Invalid-count and coincident-particle failures remain explicit. Candidate
   capacities above the team-sort limit2048 retain their failure status.

A subsequent search fill restores encounter order. The scaling benchmark
rebuilds before each timed reconstruction, so every sample includes the full
unsorted attempt and recovery. All recovery kernels count toward reconstruction
and pipeline time. Search-stage nearest-order time is zero in unsorted mode;
sorting moved into recovery is not a free speedup.

The table diagnostic checker refreshes its polygon oracle after compact recovery
has changed row order. Its repeated-reconstruction timing may see already repaired
rows; use construction_scaling for cold reconstruction performance comparisons.

## Validation and GPU experiment

Independent brute-force neighbour tests cover periodic/nonperiodic domains,
vector lengths1/4, sorted/unsorted and Cartesian/Morton variants, capacity growth,
repeated rebuilds, changed grid shapes, and non-power-of-two anisotropic grids.
Cross-layout geometry tests compare original-particle-ID keyed face areas,
centroids and cell moments for lattice, disordered, regular, random and gradient
fixtures. These tests do not establish correctness for every degenerate input.

`scripts/compare_layout_gpu1.sh` prepares one A100 on gpu1,32GiB host memory and a
15-minute maximum. Runs the contracts and cross-layout geometry checks, then all
four policies at32^3,128^3,195^3 with five repeats and separate search-stage CSVs.
Submission requires user approval. No automatic submission or default promotion.
