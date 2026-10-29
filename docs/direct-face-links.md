# Direct per-face vertex lookup experiment

`relative_early_face_links` keeps the reference candidate sorting, clipping,
double geometry and output ordering. While finding a face's starting vertex,
it builds a byte-valued map from the following plane ID to vertex index.
The face walk looks up the next vertex using its preceding plane ID, replacing
a full vertex scan at each edge. First matching entries and last seed selection
preserve reference behavior. Missing entries retain the degenerate status.

The map is reused per face and adds64 bytes for the usual workspace or160 bytes
on plane-capacity fallback. It does not remove face-seed scans or hole-edge
cancellation scans. Register count, local memory placement and runtime must be
measured: a reduction in comparisons is not proof of a speedup.

Analytic box face-link checks drop from108 to24. CPU analytic, periodic scaling,
and irregular stress tests pass, including slivers, nearly coincident particles,
clusters, and unsorted candidate cases. Policy propagates through plane retry.
Prepared scripts/compare_workspace_gpudev.sh compares reference, filtered FP32
cache, and direct links at32^3,128^3,195^3, five repeats each; requires approval.
