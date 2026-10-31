# Testing voronoi-cpp

Seven CTest programs in `src/test/` specify the library and the hydro; `voronoi_bench`
(`src/bench/`) adds the large-scale checks that are too slow for CTest.
Every test prints one `ok` or `FAIL` line per expectation and exits non-zero
on any failure. All tests run on the default Kokkos execution space, so a
CUDA build checks the device code and an OpenMP or Serial build the host
code.

## Running them

```sh
ctest --test-dir <build> --output-on-failure            # all seven
OMP_NUM_THREADS=1 ctest --test-dir <build-omp>          # CPU, one thread
```

On Raven, `scripts/bench_a100.sh <build-cuda>` (submit with sbatch, see the
README) runs the CUDA tests and the A100 checks below in one gpu1 job.

## The tests

CPU times: OpenMP build (gcc 15), one thread of a Raven login node
(Xeon Platinum 8360Y), 2026-10-31. On the A100 all seven take about 35 s.

| Test | n | CPU, 1 thread | What it specifies |
|---|---|---|---|
| `exact_predicates` | random + constructed | 0.1 s | orientation and in-sphere signs equal an independent integer evaluation (near-degenerate and 2^30-scale inputs included); ties under the symbolic perturbation are consistent from every corner order and from both sides of a triangle; periodic offsets take the nearest image; `snap(x, box)` maps any box onto [0, 2^30) and wraps the edge |
| `clipping` | 4096 | 15 s | on jittered, uniform, lattice, clustered and halo points: every cell certified, volumes positive and summing to the box, area vectors closing each cell, every face seen from both sides, every face centroid on the bisector with no generator closer (`support::check_cells`, brute force); the thread (rounded), warp and exact kernels give the same cells; a coarser cell list gives the same cells; a too small face pool grows; a particle at NaN and one at infinity are never candidates (every other cell has the neighbours and, to 1e-12, the volume of the build without them; bit-identical on CPU, on CUDA two cells differ in the last bits because the extra particles change grid levels nearby) |
| `mesh` | 4096 | 0.6 s | `from_rings` gives the same tetrahedra as a host construction from the rings, each once, with mutual adjacency and one representative per edge; volumes from stored and from walked rings equal clipping to 1e-12; the exact check passes; a small ring pool grows; the circumcentre of a sliver is exact |
| `start` | ~4096 (13 layouts), 32768 | 19 s | the start (`tet_start::build`) is the Delaunay tetrahedralization of the snapped positions for jittered, uniform, clustered, lattice, wave, sc/bcc/fcc, icosahedral shells, planes and pairs, in boxes of side 1, 2 and 3: a brute-force oracle checks that it triangulates the box (every triangle twice with apexes on opposite sides, adjacency, all particles used, exact volume sum) and that every circumsphere is empty (exact, perturbed); every edge has one representative (also wave at 32768, the dropped-face regression); jittered needs no patch; the oracle rejects a mesh with one corner moved |
| `repair` | ~4096 | 10 s | steps of a smooth flow through `voronoi::Mesh`: jittered (0.03 and 0.3 spacings, 10 steps), lattice and wave (0.03 and 0.3, 4 steps; their starts are decided by the perturbation alone, wave's first step is the step-1 regression), shells@1e-9 (0.03, 2 steps), jittered in a box of side 2 (0.3, 3 steps); after every step no failing triangle (brute force), the same Delaunay neighbours and face count as an exact rebuild by clipping, volumes within 1e-10, total volume the box |
| `voronoi_mesh` | 4096 | 2 s | the hydro contract of `voronoi::Mesh` on off-grid positions in a box of side 2: the first update builds an exact mesh; faces once with owner < neighbour, closing every cell; volumes fill the box; per-cell face lists name each face twice, in increasing id of the other particle; cell centroids equal clipping's; the same input gives the same volumes and centroids bit for bit; a permuted input gives the permuted faces; a local rebuild returns each face of the listed cells once with the mesh's area; updates repair and stay exact; on icosahedral shells@1e-9, where the device repair leaves stuck triangles, the step is exact with `Fallback::host` (local rebuilds on the host) and with `Fallback::rebuild` |
| `hydro` | 4096 (jittered, uniform) | 8.0 s | the finite-volume hydro (`src/hydro/`): HLLC and HLL equal LUNA's `hllc_flux_primitive` / `hll_flux_primitive` on 15 / 12 sample faces (values printed by LUNA, float64; within 1e-14), equal states give the exact flux, also through a moving face (F(U) - (w.n) U), swapping states and normal flips the flux, rotating states, normal and face velocity rotates the flux (1000 random rotations, 1e-13); HLLC on jittered and uniform random meshes, HLL on jittered, first and second order, per-cell and per-face loops: a uniform moving state stays uniform to rounding (3 steps, within 1e-13), mass, momentum and energy of a rough state (a blast plus 30 % random perturbations) are conserved over 10 steps within 1e-14 (long double sums); two runs give the same bits; the two loops agree within 1e-12 (measured 4e-16 with HLLC, the same bits with HLL); moving mesh (points on the snapping grid, steps whose displacement is a multiple of a grid step): a uniform state with a bulk velocity stays uniform over 5 steps (1e-13; measured 5e-16), mass, momentum and energy are conserved over 10 steps with moving cells (1e-14), a blast moving with the box at Mach 12 equals the blast at rest cell by cell after 40 steps (1e-6; measured 7e-8, rounding grown through the blast) and in its radial density profile (1e-10; measured 1e-13), while on a static mesh the profiles differ by more than 1000 times that (measured 7e-2) |

### Oracles

Brute force exists only in tests and the benchmark, never in the library:

- `support::check_cells`: each sampled face centroid against every
  generator, about 2.5e8 distances (at n = 4096 every cell is sampled).
- The start oracle (`start_test.cpp`): every circumsphere against every
  particle, O(tetrahedra x particles), exact in-sphere where a double
  prefilter is unsure; n ~ 4096.
- `support::failing`: every shared triangle of the whole mesh, exact.
- `support::clipped` (the reference of `compare_with_rebuild`): an
  independent build of the same positions by clipping.

### What PASS means

- `voronoi_bench rebuild ... ` (`check_cells`): every cell certified, total
  volume within 1e-12 of the box, closure (|sum of area vectors| / sum of
  |area|) at most 1e-12, faces = 2 x owned faces, face centroids on the
  bisectors within 1e-9 (relative). Uniform random input at 2.1M fails
  "faces = 2 x owned" by a handful of faces: the double-precision clipping
  alone is not watertight there (the start repairs it; known, expected).
- `voronoi_bench step ... --verify` (`support::compare_with_rebuild`):
  no failing triangle, no clipping face the mesh lacks, mesh faces the
  clipping lacks only below 1e-9 spacings^2 (clipping drops rounding-level
  faces, the mesh keeps every Delaunay edge), largest relative volume
  difference below 1e-10.
- The start: `tet_start::Stats::exact` (the exact check passed).

## Regression tests kept

- Wave lattice, step 1 (`repair`: wave at 0.3 spacings): the device repair
  once dropped failing triangles beyond its work capacity; since the
  device tries every flip until one closes its cavity, it repairs the step
  alone (nothing sticks).
- Clustered input, missing face (`start`: every edge has one
  representative, wave at 32768): an edge whose owner cell dropped a
  rounding-level face had no representative and no Voronoi face.
- One-thread repair livelock (`repair` on one CPU thread): plans voted
  with their index, which on one thread ordered the winners in space and
  stalled the lattice's first step for 1000 rounds.

## The A100 checks (not in CTest)

`scripts/bench_a100.sh` runs, besides ctest:

```sh
voronoi_bench rebuild 2097152 <kind> warp 3                  # check_cells PASS/FAIL per kind
voronoi_bench step 262144 <kind> 0.3 3 --verify              # jittered uniform clustered halo lattice wave
```

Each `verify:` line must end in `PASS`.
