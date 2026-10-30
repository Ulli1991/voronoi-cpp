# Integer particle positions in reconstruction

`relative_early_integer32` uses uint32 coordinates over the unit periodic box.
Each coordinate represents `integer * 2^-32`. Input conversion truncates, so
coordinate error is less than one lattice spacing (2^-32 box lengths).
Distinct input points closer than that spacing can become identical; this path
must not be assumed suitable for arbitrary spatial dynamic range. Existing
coincident-particle failure reporting is retained. No FP32 vertex cache is used.

The reconstruction consumes a slot-ordered integer array. It subtracts unsigned
coordinates modulo2^32, maps the high half to negative displacements using int64
arithmetic without signed overflow or implementation-defined casts, and only
then converts to double. The half-box tie maps to -0.5, matching the reference.
Both the clipping candidate loop and the early-stop suffix scan use this path.
Planes, intersections, vertex caching and final geometry remain double precision.

This first experiment isolates reconstruction: DenseCellSearch still gathers
positions as doubles and searches/sorts them. Integer input is gathered by the
same slot order into an additional12-byte/particle reconstruction buffer.
That gather is included in total search/pipeline timing, but not the existing
individual search-stage counters. Relative to the original double-input path,
12bytes saved in input storage offsets the12-byte slot buffer; no net persistent
array saving is claimed. Retaining integers throughout search is future work.

Validation compares complete face IDs, areas, centroids, cell volumes and cell
centroids exactly against the same quantized coordinates decoded to doubles.
Regular and jittered125-particle boxes are checked with and without a periodic
translation. Additional device checks cover integer extremes, one-tick boundary
crossings, half-box ties, and nonperiodic displacements. This establishes
agreement for quantized input, not equality to arbitrary original doubles.

Implementation reference: GADGET-4 intposconvert.h nearest-image integer
coordinate differences:
https://gitlab.mpcdf.mpg.de/vrs/gadget4/-/blob/master/src/data/intposconvert.h

The matched `relative_early_integer32_decoded` control uses the same quantized
input, search and integer gather, but reconstruction reads the decoded doubles.
Prepared `scripts/compare_integer_gpudev.sh` compares this control, integer
reconstruction and the original double-input reference at32^3,128^3,195^3 with
five repeats. It runs integer geometry checks on GPU first. Not submitted.
