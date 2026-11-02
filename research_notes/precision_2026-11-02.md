# Precision study, 2026-11-02 — what to carry over (summary)

Kept on main so the findings are not lost; the code is on branch `precision-study` (59ff9d1), all switches off by default.

To carry over into the dimension-templated hydro, as compile-time precision OPTIONS (owner's decision: investigate more; default stays all-double):
1. **Biggest win: store face fluxes in float, sum them in double.** Gather 2.8x faster (bandwidth-bound: main moves 3626 B/cell, 7.3x the needed bytes, 5 sectors per flux read; float: 1117 B/cell), static hydro step +33 % on its own; conservation unchanged (mass/energy to 1.8e-16). Also try one packed row per face.
2. Packed 32-byte float rows for face geometry and gradients, gradients and limiter computed in float from double differences (gradients 1.5x: latency-bound, active warps 18 % -> 24 %; fluxes 1.4x). The mesh should write these rows itself (the copy in the study costs 1.06 ms per moving step and a second set of geometry). Keep volumes summed in double.
3. FP16 gradients (owner's idea): accurate (L1 change 1e-6, no overflow with one exponent per variable group) but no faster than float; they save memory only.
4. Keep the Riemann solve in double (float gains 2-3 % and loses 2e-4 of the pressure in a Mach-43 uniform flow on a static mesh; LUNA's recon32 keeps it double too).
5. Drop float cut decisions in the clipping (exact but 10-15 % slower from register spills).
Open: free-stream error with float geometry (1e-7 over 40 steps vs 2e-12), the float Riemann failure's cause, accuracy on the 1D tests and KH, memory at 256^3 when the mesh writes float rows. All combinations: static 128^3 108.6 (main) -> 133 (float recon) -> 182 (FP16 grad + float Riemann + float face flux) M cell updates/s; moving 56.7 -> 70.7; mesh update + update_cells (~13 ms/step) untouched, so moving gains need pass fusion too.

---

# Reduced precision in voronoi-cpp, 2026-11-02 (branch `precision-study` from main 57201bc)

A measurement study, not for merging as is: what lower precision buys on the
A100 (Raven gpu1, A100-SXM4-40GB) without losing exactness of the mesh or
accuracy of the hydro. Everything is behind compile-time switches or separate
kernels; the default build is main's code. Builds in
`/raven/u/uli/voronoi-cpp/build-precision/` (`main` = 57201bc, `study` =
this branch, `fcut`/`fcut2` = this branch with `-DVORONOI_FLOAT_CUTS=ON`
and level 1/2); logs in `build-precision/logs/`, runs in `build-precision/runs/`;
job script `scripts/precision_a100.sh` (modes below), analysis
`scripts/precision_compare.py`, `scripts/precision_ncu.py`.

Throughput is in million cells per second (rebuild) or million cell updates
per second (hydro step); phase times in ms are given only to show where the
time goes.

## LUNA's recon32 (read, not changed; LUNA at 8aa51e2)

- The 3-D MUSCL-Hancock step casts the padded primitives W to FP32 before
  the slopes (`src/luna/hydro/muscl_hancock.py:103`, `Wp = W_pad.astype(jnp.float32) if recon32`),
  so cell-centre values, slopes, predictor and face states are FP32; the
  face states are cast back to the state dtype before the guard and the
  Riemann solver (`muscl_hancock.py:121-122`, `.astype(W_pad.dtype)`), so
  fluxes and dU are FP64 (docstring `muscl_hancock.py:94`).
- The Pallas kernel does the same: `rdtype = float32 if recon32`
  (`src/luna/hydro/muscl_hancock_pallas.py:403`), `mixed = rdtype != dtype`
  (`:224`), and `_face` casts both face states up before the Riemann core
  (`:63-64`). docs/HYDRO_KERNEL_PATHS.md:132-133 states it: "slopes,
  predictor and face states in FP32, cast up before the solver; fluxes and
  dU in FP64". 3-D only (`muscl_hancock_pallas.py:394-395`).
- Measured gain quoted by LUNA: GH200 512^3, FP32 2193, recon32 2104, FP64
  1333 M cells/s (docs/GALLERY.md:88); A100 256^3 Pallas FP64 1194 vs
  recon32 1414 M cells/s (docs/HYDRO_KERNEL_PATHS.md:162, 164).
- Here (job 31797215, A100, 128^3 Sedov, ORDER=1 HLLC minmod Pallas):
  FP64 827, recon32 ~1010 M cells/s in the steady perf windows (+22 %);
  whole run 3.21 vs 2.71 ms/step. Accuracy identical to 6 digits (L1 rho
  0.076552 both; cell-wise L1 difference 5e-10 rho0).

What differs here: LUNA's fixed grid is bandwidth-bound and its state is
read once per cell; our Voronoi kernels gather through face lists and are
latency-bound (below), so the same change buys less unless it also shrinks
the gathered rows. Our "r32" variant is the analogue of recon32 (plus FP32
geometry and gradient storage); cell values W are computed in FP64 from Q
and rounded to FP32 (as LUNA casts W).

## 1. Mixed-precision cut decisions in the warp clipping kernel

`src/voronoi_construct/shell_voronoi_warp.hpp`, `#if VORONOI_FLOAT_CUTS`.
Each polyhedron vertex keeps a float copy (xf, yf, zf); the plane test
computes side_f = fma chain in float from d rounded to float, with the bound
B = 16 u (|d| R + offset), u = 2^-24 (the error is at most 5 u (|d||x| +
offset) + O(u^2): three input roundings per product, three fma roundings;
|x| <= R for every vertex in use). side_f > B: cut off (then side > 11 u |d| R,
far above the 64 eps |d| R tolerance); side_f < -B: kept; otherwise (or NaN,
or B outside float's normal range) the side is computed in double exactly as
main does and decided as main does. Builders recompute the two double sides
of their crossed edge from the double coordinates (main's expression), so
new vertices are main's bits. Level 1 stores each vertex's squared radius in
double (kept radius without reloading coordinates; +768 B per polyhedron);
level 2 does not (+256 B; the kept radius from the double coordinates after
a cut).

Jobs 31797030 (MAIN / STUDY / FCUT) and 31797896 (MAIN / FCUT / FCUT2),
2,097,152 cells, `rebuild <kind> warp 5 --no-check`, two interleaved rounds,
warp kernel M cells/s (rounds agree to 0.2 %):

| input | main | float cuts, level 1 | level 2 | change |
|---|---|---|---|---|
| jittered | 39.94 | 35.81 | 35.65 | -10 % |
| uniform | 19.87 | 16.84 | 16.79 | -15 % |
| clustered | 15.50 | 13.68 | 13.71 | -12 % |
| halo | 14.18 | 12.64 | 12.65 | -11 % |
| lattice | 49.88 | 42.71 | 42.80 | -14 % |
| wave | 38.36 | 34.13 | 33.96 | -11 % |

(The study build with the switch off equals main within 0.4 %.)
Exactness: dumps of every cell (status, face count, owned count, hash of the
sorted owned neighbours, volume) against the switch-off build: all six inputs
IDENTICAL, volumes bit-identical (largest relative difference 0) for level 1
and 2; `step 262144 <6 inputs> 0.3 3 --verify` 18/18 PASS; exact start
121.3 -> 126.7 ms (jittered 2.1M).

Why slower: the kernel is capped at 96 registers (5 blocks per SM); the float
path adds live values (float d, bound) and the second (kept-radius) loop after
the ballot, and the stack (spills) grows from 176-192 B to 208-224 B per
thread (`cuobjdump --dump-resource-usage`); the plane test it speeds up was
8-18 % of the warp cycles (cuts_2026-10-30.md profile), and the cut chain
(ballots, shuffles, divisions) is untouched. FP64 throughput was never the
limit. Verdict: **drop**; the earlier full-float attempt's ~10 % came from
halving the polyhedron, not from the decisions.

## 2. Hydro mixed precision

`src/hydro/precision.hpp` (variants), `src/hydro/euler.hpp` (half step,
Riemann solvers templated on the real type, literals `R(...)`),
`src/hydro/finite_volume.hpp` (`FiniteVolume<Exec, Riemann, P>`; `P = Main`
is main's code unchanged; the packed kernels `gradients_packed`,
`FaceFluxPacked`, `advance_packed`, `pack_geometry`), `hydro_sedov
--precision=<v>`. In every variant Q, volumes, W = primitive(Q/V) and the
flux accumulation are double; differences W_j - W_i and separations are
formed in double and rounded once; the limiter works on the differences.
All packed variants read 32-byte rows (LayoutRight): per face (area, |area|,
face centroid, 0), per cell (centroid, 0), gradients (15 + 1 pad), held in
the mesh's scratch (spare ring pool) and filled after every mesh update
("pack").

| variant | gradients (compute / store) | face states | Riemann | geometry rows | face flux store |
|---|---|---|---|---|---|
| main | double, LayoutLeft | double | double | mesh's double arrays | double |
| d64 | double / double rows | double | double | double rows | double |
| geo32 | double / double | double | double | float | double |
| g32 | float / float | double | double | double | double |
| r32 (LUNA recon32 analogue) | float / float | float | double | float | double |
| r32s | float / float | float | float | float | double |
| h16 | float / FP16 | float | double | float | double |
| h16s | float / FP16 | float | float | float | double |
| h16sf | float / FP16 | float | float | float | float |

FP16 gradients: IEEE half (`__float2half_rn`), 15 values per cell plus one
int8 power-of-two exponent per variable group (rho, v, P) in a separate
32-bit word, chosen so the group's largest component sits in [2^14, 2^15):
no overflow at any pressure ratio (Sedov reaches 3.5e7), 11-bit precision
relative to the group's largest component, 36 B per cell (main 120 B).
Float Riemann: the face states are shifted to the face frame in float, solved
with w = 0, and brought to the lab frame in double. FP64 instructions per
kernel (SASS): flux kernel 455 (d64) -> 374 (r32) -> 201 (r32s); gradient
kernel 882 -> 193; what remains is the double primitive conversion, geometry
sums and the lab-frame conversion.

### Throughput, A100, jittered, Morton order, per-face loop, median of 40 steps

128^3 (job 31798111, two interleaved rounds, mean; MAIN = main's binary):

| variant | static: gradients / fluxes / gather ms | static M cell updates/s | moving: M cell updates/s (pack ms) | peak B/cell static / moving |
|---|---|---|---|---|
| MAIN | 7.42 / 3.58 / 5.68 | **108.6** | **56.7** | 2370 / 2418 |
| d64 | 7.17 / 3.23 / 5.68 | 112.0 (+3 %) | 55.5 (1.92) | 2920 / 2968 |
| geo32 | 5.83 / 2.90 / 5.68 | 123.0 (+13 %) | 59.5 (1.06) | 2649 / 2697 |
| g32 | 6.65 / 2.80 / 5.68 | 118.1 (+9 %) | 57.0 (1.92) | 2856 / 2904 |
| r32 | 4.91 / 2.53 / 5.69 | 133.1 (+23 %) | 61.6 (1.06) | 2585 / 2633 |
| r32s | 4.89 / 2.15 / 5.68 | 136.5 (+26 %) | 62.6 (1.06) | 2585 / 2633 |
| h16 | 5.00 / 2.48 / 5.68 | 132.8 (+22 %) | 61.8 (1.06) | 2557 / 2605 |
| h16s | 4.98 / 2.05 / 5.68 | 136.6 (+26 %) | 62.7 (1.06) | 2557 / 2605 |
| h16sf | 4.98 / 1.92 / 2.01 | **181.6 (+67 %)** | **70.7 (+25 %)**; 73.3 without pack | 2398 / 2446 |

Time step (2.65 ms), point velocities (3.25), mesh update (7.2) and
update_cells (6.05) are unchanged (double, not touched). In the moving step
the hydro phases are 17.8 of 37.0 ms on main, so precision can at most
halve that part: h16sf takes them to 9.4 ms (+1.06 pack).

Per phase at 128^3 (static, M cells/s per phase, main -> best): gradients
283 -> 429 (float geometry rows and float compute; FP16 storage adds nothing
over float), fluxes 586 -> 829 (r32, Riemann double) -> 1023 (h16s, Riemann
float) -> 1092 (h16sf; FP16 gradient rows read 36 instead of 64 B but the
time equals float's within 3 %), gather 369 -> 1043 (only with float face
fluxes), mesh unchanged (update 7.2 + update_cells 6.05 ms = 158 M cells/s).

256^3 (jobs 31798112/3, one round): MAIN static 99.8 M cell updates/s
(gradients 59.5, fluxes 40.4, gather 45.7 ms), **h16sf static 179.4**
(39.8 / 15.1 / 16.1 ms, 1.80x); MAIN moving 57.5. Every other variant ran out
of device memory at 256^3 ("failed to allocate 9.8-15.4 GiB
(face_rings_pool)"): in this study the packed rows come on top of the mesh's
double geometry and grow the spare ring pool; h16sf moving also failed
(7.3 GiB). Per cell, the 256^3 phases equal 8x the 128^3 ones for both
measured builds (no change of regime).

### Accuracy (Sedov 128^3, HLLC, MUSCL-Hancock, whole run to 1e4 yr; jobs 31797214, 31797215)

Analysis `scripts/hydro_sedov_compare.py` (L1 over r < 1.5 R_s), more
digits and cell-by-cell differences against MAIN with
`scripts/precision_compare.py`:

| variant | static: L1 rho / v_r / P, R_s/R_a | static: L1 of (rho - rho_MAIN) / rho0 | moving: L1 rho / v_r / P | moving: L1 of difference | conservation (mass, energy) |
|---|---|---|---|---|---|
| MAIN | 0.075006 / 0.028902 / 0.020262, 0.97815 | | 0.057994 / 0.018080 / 0.014712 | | <= 1.8e-16 |
| d64 | same 6 digits | 7e-17 | same | 5e-13 | <= 1.8e-16 |
| geo32 | same | 2.3e-9 | same | 1.9e-7 | <= 1.8e-16 |
| g32 | same | 4.8e-10 | same | 2.7e-7 | <= 1.8e-16 |
| r32 | same | 2.7e-9 | same | 2.5e-7 | <= 1.8e-16 |
| r32s | same | 6.8e-9 | same | 2.2e-7 | <= 1.8e-16 |
| h16 | same | 1.8e-7 | 0.057995 / 0.018080 / 0.014712 | 1.35e-5 | <= 1.8e-16 |
| h16s | same | 1.8e-7 | 0.057993 / 0.018079 / 0.014712 | 1.34e-5 | <= 1.8e-16 |
| h16sf | same | 1.8e-7 | 0.057995 / 0.018080 / 0.014712 | 1.32e-5 | <= 1.8e-16 |
| LUNA FP64 (Pallas) | 0.076552 / 0.031100 / 0.020252, 0.97726 | | | | |
| LUNA recon32 | same 6 digits | 5e-10 (vs LUNA FP64) | | | |

Steps: 1636 static, 348 moving, the same for every variant; face fallbacks
121856-121860 static, 18170-18198 moving. Largest single-cell differences:
static float variants 4e-5 (rho), FP16 7e-4; moving runs diverge cell by cell
near the shock (mesh topology follows the flow; even g32 reaches 1e-2 in one
cell), while their L1 errors agree to 1e-6.

PASS bar: L1 rho, v_r, P each at most LUNA FP64's 128^3 value plus 2 %
(0.0781 / 0.0317 / 0.0207), R_s/R_a within 0.5 % of our double run, mass and
energy conserved to 1e-15. **Every variant passes, static and moving** (the
FP16 variants change L1 rho by 1e-6 against an error of 0.058-0.075).

Free stream (`--no-blast --bulk=500`, Mach 43, 40 static steps / 10 moving;
job 31797201), largest relative deviation of P from uniform:

| | MAIN, d64, g32 | geo32, r32, h16 | r32s, h16s | h16sf |
|---|---|---|---|---|
| static mesh | 2.3e-12 | 0.9-1.0e-7 | 2.0e-4 | 2.8e-4 |
| moving mesh (10 steps) | 1.9e-7 | 1.8-1.9e-7 | 1.8e-7 | 4.9e-7 |

Float face geometry breaks free-stream preservation at the 1e-7 level per
40 steps (sum of rounded areas no longer closes); the float Riemann solve
on a static mesh at Mach 43 loses 2e-4 of the pressure (the kinetic energy
flux is 1000x the thermal one and is rounded to float before the
difference). On the moving mesh the face frame removes the bulk velocity
before the solve and main itself is at 1.9e-7 (geometry at the start of the
step, see hydro_stage3). Sedov does not see either effect. CTest `hydro`
(study build, A100): no FAIL line.

### What limits each phase (bytes per cell update against achieved GB/s)

Nsight Compute 2026 (job 31798354, after `dcgmi profile --pause`: without
it Raven's DCGM holds the counters, "a driver resource was unavailable",
job 31798114), 128^3, 20 launches per kernel (static) or 3 (moving), medians;
kernel times under ncu are 5-15 % above the fenced ones. DRAM = bytes read +
written per cell; the A100 40 GB peaks are ~1555 GB/s DRAM:

| kernel | main: DRAM B/cell | GB/s | L2 B/cell | active warps | long-scoreboard stall | h16sf: DRAM B/cell | GB/s | active warps |
|---|---|---|---|---|---|---|---|---|
| gradients | 1130 | 275 (18 %) | 7963 | 18 % | 66 % | 963 | 345 (22 %) | 24 % |
| fluxes | 1608 | 877 (56 %) | 4337 | 25 % | 65 % | 797 | 785 (50 %) | 34 % |
| gather (update) | 3626 | 1247 (80 %) | 7669 | 70 % | 70 % | 1117 | 1094 (70 %) | 70 % |
| time step (unchanged) | 667 | 455 | 2784 | 34 % | 71 % | 660 | 448 | 34 % |
| r32 gradients / fluxes | 956 / 1012 | 350 / 746 | 5249 / 2986 | 24 % / 25 % | 73 % / 65 % | | | |

Compulsory bytes (each face's data read once although both sides gather it;
F = 7.95 faces per cell): gradients 737 (main) / 518 (h16sf), fluxes 987 /
609, gather 498 / 339 B/cell.

- Gather: **bandwidth-bound**. Main moves 3626 B/cell from DRAM, 7.3x the
  compulsory bytes, at 80 % of peak: each neighbour-side read of a face flux
  touches five 32-byte sectors (five LayoutLeft component arrays) for 40
  useful bytes, and the 667 MB flux array does not stay in L2 between the
  two sides' reads. Float fluxes (334 MB) cut DRAM traffic 3.2x (more than
  the 2x in size: a larger share of the second reads hits L2), hence 2.8x.
  Packing the five components into one row (not tried) would be the next
  step for the same reason.
- Gradients: **latency/occupancy-bound**. 18 % active warps (register-heavy
  4x4 least squares per cell, 142 FP64 instructions per cell), 66 % of
  issue slots stalled on long-scoreboard (dependent gathers), 18 % of DRAM
  peak. Float compute raises active warps to 24 % (fewer registers) and cuts
  FP64 instructions to 38 per cell; DRAM bytes drop only 15 %. That is where
  the 1.5x comes from, not bandwidth.
- Fluxes: mixed, at 50-56 % of DRAM peak with 25-34 % active warps. Float
  geometry rows and float gradients halve the DRAM bytes (1608 -> 797); the
  float Riemann solve raises active warps 25 -> 34 %.
- Point velocities (moving, unchanged): 1918 B/cell at 1135 GB/s (73 %),
  90 % long-scoreboard: also a gather of LayoutLeft rows, the same cure.

## 3. Memory

Measured device peaks (Kokkos allocations, `--memory`) are in the table
above. In this study the packed rows are copies (32 B per face + 16 B per
cell in float, double twice that) on top of the mesh's double geometry, so
every variant except h16sf uses more memory than main, and nothing but main
and h16sf static fits 256^3.

What carrying the formats into the library would save, computed from the
array sizes (not measured; F = 7.95 faces per cell, face arrays allocated
with 1/8 spare, `voronoi_mesh.hpp:179-185`):

| item | double | reduced | saved B/cell |
|---|---|---|---|
| face area + centroid in the mesh (2 x 3 values, x 1.125 F) | 429 | 215 (float) | 215 |
| cell centroid | 24 | 12 | 12 |
| gradients (hydro scratch) | 120 | 64 (float) / 36 (FP16 + exponents) | 56 / 84 |
| face fluxes (hydro scratch) | 318 | 159 (float) | 159 |

The hydro scratch (gradients + fluxes) is what grows the spare ring pool
(438 B/cell on main at 256^3, hydro_stage3 notes); FP16 gradients and float
fluxes cut it to 195 B/cell, below the ring pool's own size, so the pool
would no longer grow for the hydro (main: +94 B/cell growth above the ring
pool, estimated from 438 - 344). With float face geometry in the mesh as
well, main's static 256^3 peak of 2371 B/cell (39.8 GB) would drop by about
215 + 12 + 94 = 321 B/cell to ~2050 B/cell, 5.4 GB less at 256^3, i.e. room
for about 19-20M cells (~270^3) instead of 17M on the 40 GB card (estimate).
Caveat: update_cells() sums volumes and cell centroids over the stored face
geometry; from float areas the volumes would be good to ~1e-7 only, and the
CTest contracts (volumes fill the box to 1e-12, uniform states stay uniform
to 1e-13) would fail. Keeping the volumes exact needs the double geometry
until the volumes are summed (e.g. the faces kernel adds the pyramid
volumes and moments in double before rounding the rows), which the
library's current face-then-cells split does not do.

## Recommendation (what to carry over to the dim-templated code)

1. Float face fluxes (stored float, summed in double): the largest single
   gain (gather 2.8x, its DRAM traffic 3.2x lower; static step +33 % on its
   own, h16s -> h16sf), conservation unchanged (the same value is added and
   subtracted), Sedov L1 unchanged. Carry over; also try one packed row per
   face for the five components (the gather is bandwidth-bound by sector
   over-fetch).
2. Packed 32-byte rows for the face geometry (area, |area|, centroid) and the
   gradients, in float, with gradients and limiter computed in float from
   double differences W_j - W_i (r32/h16 path): gradients 1.5x, fluxes 1.4x;
   Sedov unchanged; free stream preserved only to ~1e-7 per 40 steps (float
   areas do not close). Carry over, with the mesh writing the float rows
   itself (the copy here costs 1.06 ms per moving step and a second
   geometry), and with volumes and cell centroids summed from double
   geometry.
3. FP16 gradients (the owner's idea): accurate (L1 change 1e-6, no overflow
   with one exponent per variable group) but no faster than FP32 (64-byte
   float rows are already one line); memory only (28 B/cell measured,
   36 vs 64 B/cell). Optional; FP16 for nothing else.
4. Float Riemann solve: +2-3 % of the static step; loses 2e-4 of the
   pressure in a Mach-43 free stream on a static mesh; keep double (LUNA's
   recon32 does too).
5. Float cut decisions in the warp clipping kernel: exact (cells bit-
   identical) but 10-15 % slower; drop.

Combined (h16sf): static 128^3 108.6 -> 181.6 M cell updates/s (1.67x), 256^3
99.8 -> 179.4 (1.80x); moving 128^3 56.7 -> 70.7 (1.25x; 73.3 without the
pack copy), where the unchanged mesh update and update_cells take most of the
rest. Every variant passes the Sedov bar against LUNA (L1 rho 0.0750 static
/ 0.0580 moving vs LUNA's 0.0766).
