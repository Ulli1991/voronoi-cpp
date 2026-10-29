# GPU neighbour search and local Voronoi optimization plan

Updated 2026-10-28 from Ulrich Steinwandel's GPU optimization notes.
Primary objective: fast, robust, memory-lean local construction before hydro.
Retain reference implementations and artifacts. Ask before each cluster job.

## Measurement and acceptance

Measure neighbour search, full reconstruction (including face geometry, moments,
certification and retries), and combined throughput separately. Report cells/s,
median time, particle count/distribution, precision, device, peak allocation,
status counts and numerical errors. Compare identical input and output contracts.
Use 32^3, 128^3 and 195^3 initially; increase only within measured memory limits.
Include irregular, clustered, near-degenerate and scale-varied distributions.
A faster result is acceptable only with equivalent certification and geometry.

The suggested 50–100 million cells/s is an aspirational target, not a measured
bound. FLOP estimates must account for precision, intersections and face walks.
Neighbour input bytes alone omit output, table, topology and spill traffic;
640 bytes/cell at 1.5–2 TB/s would imply 2.3–3.1 billion cells/s for that input
alone, not a complete reconstruction roofline. Low compute utilization alone
cannot distinguish bandwidth, latency, divergence and instruction overhead.

## Ordered experiments

1. **Profile and establish baselines.** Existing Nsight runs measured high traffic,
   long scoreboard stalls and about 24% occupancy. Next capture local-memory
   load/store traffic, shared usage, occupancy and branch/active-lane efficiency
   for the actual cooperative kernel and reference. Keep profiler runs separate
   from uninstrumented throughput measurements.
2. **Particle locality.** Counting-sort by grid cell already exists. Measure gather
   locality before adding Morton order or deferred sorting. Deferred sorting is
   a later moving-particle feature with explicit validity conditions.
3. **Reduce per-cell workspace traffic.** Finish and validate warp-per-cell shared
   storage versus thread-per-cell and serial shared-storage baselines. Parallel
   classification and intersections exist; topology and final face traversal
   remain serial within each team. Include full-capacity fallback in timings.
   Then test uint8 plane triplets (capacity <=256), plane storage layout, reduced
   capacity and compaction. Small arrays alone do not guarantee no spills.
4. **Tune grid resolution.** Sweep occupancy/cell and search cost with irregular
   inputs. A 27-cell stencil is complete only when the query radius is no larger
   than cell width; finer grids require a larger stencil or adaptive traversal.
   Keep certification/completeness unchanged when reducing candidate counts.
5. **Seed-relative mixed precision.** Compare FP32 planes/coordinates with FP64
   reference. Use justified error bounds and double recomputation for ambiguous
   decisions; a double fallback cannot recover information already quantized
   away. Preserve original positions for fallback. Measure topology disagreement,
   volumes, face areas/centroids and failure rates, not volume alone.
6. **Cell-centric tiles and search/clip fusion.** Share candidate loads among
   nearby seeds, then prototype fused search and clipping without the persistent
   candidate-table round trip. Compare scratch occupancy, reuse and memory at
   fixed correctness. Consider count/prefix/write compact face storage against
   current fixed-capacity output; include both passes in timing.
7. **Temporal reuse, separately benchmarked.** Persist face neighbours; test them
   plus neighbours-of-neighbours as warm-start candidates. Add skin lists and
   displacement tracking. Reuse sorted order only under a proven validity rule.
   Report cold and warm throughput separately on controlled motion sequences.

## Certification contract

Clip nearest candidates first from a conservative local bounding cube. The
security-radius test uses R = maximum seed-to-cell-vertex distance. A distance
beyond 2R permits termination only if all omitted sites are known to be farther
away. The current table path conservatively checks the remaining suffix rather
than assuming arbitrary caller lists are sorted.

A previous face-neighbour list, a Morton window, or neighbours-of-neighbours
alone does not prove completeness. Warm-start cells require a spatial exclusion
query or a proved motion/skin bound covering omitted particles. When coverage is
insufficient, extend search for that seed and continue/rebuild clipping. Never
mark incomplete geometry successful or silently truncate capacity failures.

## Additional options after the measured bottleneck is addressed

- Warp-level selection when an explicit nearest-neighbour list is needed.
- LBVH for strongly clustered data, compared against the tuned grid including
  construction cost; Morton windows require certified spatial fallback.
- Exact/adaptive predicates or consistent symbolic perturbation for ambiguous
  topology; benchmark robustness separately from mixed precision.
- RT-core search only on compatible hardware, not A100/H100; it is outside the
  portable Kokkos baseline.
- Multi-GPU domain decomposition: include halo completeness, exchange cost,
  strong/weak scaling and identical local certification before claiming scaling.

## Current next gate

Validate full cooperative geometry and fallback, compile CUDA, prepare paired
GPU timings and request submission approval. No cooperative speedup is claimed
until the GPU run completes. Warm starts are not required for this cold-build gate.

## Revision after GPU job 31601151 and LUNA memory discussion

Cooperative reconstruction is 4.49x slower at195^3; retain it as experimental.
The reference remains the baseline. Make one isolated change before further
mapping rewrites. The user corrected the LUNA grid-size recollection to possibly
256^3; no precise LUNA memory comparison is established.

First memory experiment: eliminate the 2000-byte/particle persistent distance
table from team sorting, recomputing double distances in scratch instead. Keep
candidate IDs and the fast reference reconstructor unchanged. Validate periodic
ordering, ties, growth/fallback and geometry; measure both search and full
pipeline time. See [allocation budget](memory-budget.md). Expected array saving
is 41.8%, not a predicted speedup and not sufficient alone for256^3 on40GB.
