# Filtered FP32 vertex cache

Opt-in benchmark policy: `relative_early_float_cache`. Uses the same scratch
candidate sort and double plane workspace as `relative_early_scratch_sort`.
The per-cell coordinate cache is 768 bytes rather than 1536 bytes. Public
geometry and particle storage remain double; this is not integer position storage.

Clipping reads FP32 cached seed-relative vertices. Tests within a cache-rounding
margin of the clipping boundary recompute the intersection from double planes.
Nonfinite cached values also take that path. Radius estimates are inflated for
cache rounding. Certification and final geometry always recompute double
intersections. This retains the reference's numerical limitations; it is not
an exact-predicate implementation. No tolerance relaxation in final geometry.

The tradeoff is extra double intersection solves during output versus less
private cache traffic. Neither fewer spills nor a speedup is established yet.
Initial CPU validation covers 20 analytic configurations spanning five scales,
rotations and asymmetry, plus the existing reference cases and a 12^3 periodic
construction with geometry checks. CUDA analytic target compiles. Clustered,
near-degenerate and GPU execution comparisons remain necessary before promotion.

FP16 planes/vertices and GADGET-style integer particle storage are not included
in this first isolated experiment. Half precision needs its own scale and error
bounds; it is not safe to substitute it directly in clipping decisions.
