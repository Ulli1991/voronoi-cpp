# Measurements: provenance

Where each number in the [README](../README.md) comes from: jobs, commits, clocks and the setup details left out of the README. The full tables of rebuild and step rates, long runs, GPU against a CPU node, and 2D and 1D are here; the comparison with other libraries is in [library_benchmark.md](library_benchmark.md#the-full-table).

## Measured performance

One A100 40 GB (Raven gpu1), double precision, periodic unit box. Rates are million cells per second, cells processed per second of GPU time.

### Rebuild, start and step rates

Rates are M cells/s: million cells built or updated per second of GPU time. luna-cpp quotes its hydro in G cell updates/s; 154 M cells/s is 0.154 G. Memory is in bytes per cell and GB of 10⁹ bytes. Every change tried since 2026-11-12, with its measured effect and why it was kept or dropped, is in [tried.md](tried.md). The rebuild rows are the lean cut of 2026-11-13 (1305 MHz, 128³ and 256³), the white-noise rebuild with the hard-cell change of 2026-11-15, the clustered and halo rebuilds with the overflow routing of 2026-11-16 (#60, default clocks); the step rows, at 128³ and 256³, are the incremental face numbering and ring re-walk of 2026-11-12 (`step-bytes`, 1305 MHz); the exact start is `start-fast` of 2026-11-15 (1305 MHz, 128³ and 256³).

| M cells/s | 2,097,152 cells (128³) | 16,777,216 cells (256³) |
|---|---|---|
| Full rebuild (warp clipping), jittered | 42.6 | 42.8 |
| Full rebuild, white noise | 22.2 | 22.5 |
| Full rebuild, clustered (Gaussian blobs, ~1:500) | 17.7 | 18.5 |
| Full rebuild, halo (1:10⁴ core to edge) | 16.9 | 18.3 |
| Full rebuild, lattice / wave | 55.1 / 40.9 | 57.7 / 41.0 |
| Full rebuild, surface (shells and tori, 2026-11-16, [#60](https://github.com/Ulli1991/voronoi-cpp/issues/60)) | 0.29 | 0.100 |
| Exact start (warp with rings + `tet_start::build`), jittered / blue noise | 25.3 / 23.1 | 22.2 / 20.6 |
| Exact start, white noise / clustered / halo | 12.3 / 10.3 / 11.9 | 12.6 / 12.5 / 12.1 |
| Exact start, lattice / wave | 8.0 / 5.0 | 8.4 / 5.1 |
| Moving-mesh update, 0.03 spacings per step | 223 | 318 (215 with `update_cells()`) |
| Moving-mesh update, 0.1 spacings per step | 130 | 209 (151 with `update_cells()`) |
| Moving-mesh update, 0.3 spacings per step | 65.9 | 110 (85.6 with `update_cells()`) |
| `update_cells()` alone, 0.03 / 0.1 / 0.3 spacings | 559 / 440 / 432 | |
| Update + `update_cells()`, 0.03 / 0.1 / 0.3 spacings (bit-reproducible; 2026-11-18, [step-2x](#the-2-step-branch-step-2x-2026-11-18)) | 179 / 109 / 61.9 | 244 / 170 / 93.6 |
| The same over 500 steps of the flow, re-sorts paid (default every 25) | 194 / 142 / 84.4 | – / 196 / – |

With `Layout::same_bits_as_host` (the host's bits on the GPU; `Engine` hybrid) update + `update_cells()` is 3.4 to 5.1 / 1.7 to 4.6 / 0.7 to 2.8 % slower at 0.03 / 0.1 / 0.3 spacings (128³ and 256³), the rebuild at most 1 % and the exact start at most 1.5 %, 5 % on cosmicweb at 256³ ([below](#rounding-per-mode-2026-11-16-jobs-32422876-and-32429908)). Without it the GPU keeps main's arithmetic and speed.

Spacings per step is how far a smooth flow moves the points per step, in mean interparticle spacings. 0.03 is a typical hydro step and 0.3 a very large one, with more flips per step and lower throughput. The six inputs (jittered lattice, white noise, clustered, halo, lattice, wave) and the three application inputs of the library benchmark (cosmic web: Zel'dovich positions of a CDM field; fcc solid: a thermal crystal near melting; surface: points on closed shells and tori) are described [below](#inputs).

The step kernels of 2026-11-12 (register limits on `faces` and `screen`, warp-aggregated counters, a thread per edge in `dirty_faces`, one host wait per flip round) made the step 5.4 / 11.0 / 12.8 % faster at 0.03 / 0.1 / 0.3 spacings at 128³ (6.3 / 9.3 / 15.0 % at 256³) with the same bits and arrays. The step at 0.03 now runs at 47 % of what its memory traffic allows on the A100 (434 M cells/s with every kernel at 80 % of peak bandwidth; [measure_2026-11-12.md](measure_2026-11-12.md)). Keeping the face numbers of faces whose tetrahedron no flip touched ([#53](https://github.com/Ulli1991/voronoi-cpp/issues/53)) and walking each changed face ring once then cut the step's memory traffic from 2717 to 2392 B/cell at 0.03 spacings (4292 to 4133 at 0.1) and made it 8.4 / 2.8 / 3.5 % faster at 0.03 / 0.1 / 0.3 spacings at 128³ (13.4 / 7.2 / 1.5 % at 256³), with the same geometry bits and a lower memory peak ([below](#fewer-bytes-per-step-incremental-face-numbers-e-and-the-ring-re-walk-k2-2026-11-12)).

GPU memory, as the peak of all arrays the mesh allocates: a plain warp rebuild takes 701 B/cell (656 at 256³); a step with face lists peaks at 1979 B/cell (2010 at 0.3 spacings; 128³): about 1930 for the mesh's arrays, the rest the bench's positions. The arrays and pools are drawn in [how_it_works.md](how_it_works.md#13-where-the-data-lives-and-how-the-gpu-touches-it).

A thin slab (Lz = 16 spacings, up to 16.8M cells) rebuilt at the cube's rate before the lean cut, 39.9 M cells/s for jittered input ([non-cubic box](#non-cubic-box)).

The lean cut (2026-11-13) made the rebuild 10 to 15 % faster than main 6ead543 on all six inputs at 128³ and 256³, and the exact start 6 % on jittered input, with the same tetrahedra, volumes, centroids and per-cell lists. It needed canonical face rings, which cost under 1 %: a new vertex goes into the lowest free slot, built by the lane that owns it, without a stack of free slots; the starting cube is set by formula; the candidate sort and the side test have no branches. That is 7 to 8 % fewer warp instructions per cell (9611 to 8802 on jittered input, 15128 to 14020 on white noise). The kernel still waits on the latency of each cut and on instruction fetch, so a leaner cut is not enough ([lean cut](#canonical-rings-and-a-lean-cut-2026-11-13)). Fewer clips are not enough either (2026-11-15): vertex balls halve the clips on unstructured input with the same cuts and the same time, and clipping by k-nearest lists (k = 24, 32, 48, and hybrids with the walk) is 4 to 55 % slower than the walk on all six inputs at 128³ and 256³, with the same mesh. The cuts, about one per face, are the cost; a several times faster rebuild needs a cheaper cut or no clipping ([kNN-first clipping](#knn-first-clipping-hybrids-and-a-vertex-ball-filter-2026-11-15)).

Clustered and halo input (2026-11-15) are slower than white noise because of more clips per cell (50.7 and 53.6 against 48.0) and the few cells the warp kernel gives up (68 and 261 of 2.1M, more than 64 planes or vertices), not because of a tail of slow cells: persistent warps take cells from a counter, and the slowest 1 % of cells hold 2 to 4 % of the warp's cycles. Those few cells used to be built twice more on one thread each, first at the thread kernel's usual capacities, which they overflow again; they now go straight to its big cells. That made the halo rebuild 5.3 % faster at 128³ (clustered and white noise 1.2 %, 0 to 0.7 % at 256³) with the same bits. A queue of hard cells, cells on the host, and larger warp cells were measured or ruled out ([tried.md](tried.md), [hard cells](#hard-cells-on-clustered-and-halo-input-and-49-2026-11-15)).

The exact start of 2026-11-15 (`start-fast`) builds the same arrays as main c42c230, entry for entry, 1.07 to 1.74× faster on all seven inputs at 128³ and 256³ (jittered 18.6 → 25.3 and 12.7 → 22.2 M cells/s; blue noise 18.6 → 23.1 and 13.6 → 20.6). `tet_start::build` alone, without the clipping, is 2.1× faster at 128³ and 2.8× at 256³ on jittered input. The assembly reads the clipping's ring blocks in place with one thread per ring slot, tetrahedron or face (main ran one per face and copied every ring first), the check decides by double filters and sends only undecided tetrahedra to the exact predicates, and the round after a partial re-clip, which most inputs need at 256³, looks up only around the re-clipped cells ([#59](https://github.com/Ulli1991/voronoi-cpp/issues/59)). The clipping, unchanged (with rings, and on lattice and wave the exact re-clip of every cell), is now 60 to 90 % of the start. The start's memory peak through `Mesh::update` is main's (1955 B/cell at 256³ jittered). Kernel by kernel: [the exact start](#the-exact-start-assembly-from-the-ring-blocks-and-an-incremental-patch-round-2026-11-15).

### Provenance of the rates

| Rows in the table above | Job | Commit |
|---|---|---|
| Moving-mesh update, update plus `update_cells()`, 128³ and 256³ | 32269165, 32237516 (interleaved with main 9e9ec03; [fewer bytes per step](#fewer-bytes-per-step-incremental-face-numbers-e-and-the-ring-re-walk-k2-2026-11-12)) | `step-bytes` (the ring re-walk commit) |
| `update_cells()` alone | 32231679 ([step kernels](#step-kernels-at-todays-bytes-2026-11-12)) | `step-kernels` |
| Full rebuilds at 2.1M and 16.8M, exact start at 2.1M | 32269868, 32269874, 32269869 ([lean cut](#canonical-rings-and-a-lean-cut-2026-11-13)) | `clip-canonical` |
| Full rebuilds, uniform, clustered and halo, at 2.1M and 16.8M | 32366435, 32366438 ([hard cells](#hard-cells-on-clustered-and-halo-input-and-49-2026-11-15)) | `clustered-build` |
| Exact start, the sum row, at 2.1M | 31701509 | 2311937 |
| The 256³ column, except the moving-mesh rows | 31694457 | f32c909 (older than the current code, which is faster there) |
| Moving-mesh rows (also in research_notes/mesh_speed_2026-11-04.md) | 31886231, 31886232 | |

Moving-mesh update rates are per step of a smooth flow that moves the points by the given number of mean interparticle spacings, spacing = (box volume / n)^(1/3). Steps are medians of steps 2 to 10 on jittered points. At 256³ the step moves particles by the same number of spacings, but the flow varies on the box scale, so neighbours shear half as much per step (half the flips per cell).

On jittered, uniform and clustered input at 0.1 and 0.3 spacings no triangle ever sticks. The first step out of a degenerate start (wave lattice, 2.1M) is repaired on the device alone in 0.81 s (job 31701509, research_notes/shrink_2026-10-31.md). Icosahedral shells at 1e-9 still leave about 8k of 262k stuck (host or rebuild).

Device memory: plain warp rebuild 701 B/cell (656 at 256³). Peak during start and steps, with face lists, 2028 B/cell (2007 at 256³, 34.8 GB in use), older code; the current step peak is 1998 B/cell at 128³ and 256³ alike, 1950 for the mesh's arrays (job 32145254, [measure_2026-11-10.md](measure_2026-11-10.md)). The per-cell face lists account for most of the step peak above the 2080 B/cell of commit 2fd8b50 (about 140 B/cell; not measured separately).

### Inputs

All in the periodic unit box, generated in `src/test/support.hpp`.

| Name | Points | Tests |
|---|---|---|
| jittered | cubic lattice, each point moved randomly by up to ±0.1 lattice spacing per axis | the typical hydro start; nearly regular, about 16 faces per cell |
| white noise (`uniform`) | independent uniform random points | unstructured cells of very different size and shape |
| clustered | 8 Gaussian blobs (σ = 0.04) with 80 % of the points, 20 % uniform background | density contrast about 1:500 (blob centre to void) |
| halo | 4 halos with ρ ∝ 1/(r² + c²) to r = 0.2, c = 0.002, 90 % of the points, 10 % background | density contrast about 1:10⁴ (core to edge) |
| lattice | perfect cubic lattice | fully degenerate: 8 points on every circumsphere |
| wave (moved lattice) | the lattice with every point moved by up to 1e-5 spacing | nearly degenerate: ties broken only by tiny offsets |
| cosmic web | Zel'dovich approximation x = q + D ψ(q) from a cubic lattice: Gaussian random field with the BBKS CDM spectrum (Bardeen et al. 1986; Γ = 0.21, n_s = 0.96), σ₈ = 0.8 at D = 1, 100 Mpc/h box, all modes below the Nyquist number of the lattice; growth factor D = 0.5 (rms displacement 4.3 spacings at 128³) | filaments, walls and voids: density contrast about 1:40 on a 2-spacing grid, 4 % of that grid empty; weakly deformed lattice in the voids |
| fcc solid | fcc lattice of l³ unit cells (4 l³ atoms; l = 64, 128: 2²⁰, 2²³), each atom moved by an independent Gaussian of rms 0.0499 lattice constants per axis: the measured rms of the LAMMPS Lennard-Jones solid at ρ* = 1.05, T* = 1.0 near melting (MSD 0.018 σ², Lindemann ratio 0.12; research_notes/lammps_voronoi_2026-11-12.md on branch lammps-scout) | an MD crystal: near-ties of the rhombic dodecahedra broken by thermal noise |
| surface | 16 spherical shells of radius 0.2 on a bcc lattice of side 0.5 (8 cross the box faces), each around a torus of radii 0.1 and 0.05; uniform in area, moved along the normal by a Gaussian of rms 0.1 surface spacings √(area / n) | points on closed surfaces: thin needle cells reaching about 0.1 box into empty space, the graphics-benchmark case |

### Non-cubic box

Job 31702953 (research_notes/box3_2026-10-31.md). Kelvin-Helmholtz-like slab, Lx = Ly = 2, Lz = 16 spacings: `--box=2,2,0.0625` with 512 x 512 x 16 = 4.2M cells, and `--box=2,2,0.03125` with 1024 x 1024 x 16 = 16.8M cells.

| M cells/s | 4.2M | 16.8M |
|---|---|---|
| Warp rebuild, jittered | 39.9 | 39.9 |
| Warp rebuild, uniform | 19.9 | 20.0 |
| Exact start | 11.6 | 12.0 |
| Steps at 0.03 spacings | 105 | 108 |

The bench flow, stretched with the box, shears across the 16-cell thickness: 4x the flips per cell of the cube. Peak 2192 B/cell at 16.8M. The cubic case is bit-identical to the cubic-only code and within 0.3 to 0.7 % of its rebuild speed.

### Against other libraries (2026-11-12 to 2026-11-15)

The library table ([library_benchmark.md](library_benchmark.md#the-full-table)): methods, every setting, 256³ and the
per-code versions, compilers and jobs in [library_benchmark.md](library_benchmark.md); raw records in `bench/libraries/results/` and the tables generated from
them by `scripts/library_benchmark.py` (`summary.md`). voronoi-cpp rows: 77055a4 (close-gaps, below), A100 jobs
32368078, 32368079, 32368080 (gpu1, 1305 MHz), CPU job 32367276 (with voro++ in the same job); before that ad62997
(main b2faee8 plus the benchmark), A100 jobs 32272549, 32272550 and 32319068 on `ravg1002` (gpudev, 1095 MHz under
load), CPU job 32272650 (Kokkos OpenMP, 72 cores and 1). Every other GPU code ran on the same node in chained jobs (`bench/libraries/run_chain_a100.sh`), each
with our warp rebuild as a same-GPU reference. Inputs: jobs 32225761 (generators) and 32234292 (blue noise, 100 Lloyd
iterations). Oracle checks: job 32269164 and accuracy-main-login.jsonl (one core, login node).

### Volumes-only output and the CPU path (2026-11-15)

Branch close-gaps, commit 77055a4 (main c42c230 plus the changes below). Library benchmark inputs and scripts
([library_benchmark.md](library_benchmark.md)), 128³ unless stated, median of 3 timed runs, M cells/s.

**Volumes and centroids only** (`shell_voronoi::volumes_and_centroids`, `Cells::volumes_only`): the warp kernel with
`warp::Output::volumes` records status, volume and centroid and skips the face sums, face writes and pool claim; the
walk, certification, canonical face starts (so the bits of the volume sums) and fallbacks are unchanged. Volumes and
centroids are bit-identical to the full rebuild's on blue noise, jittered and halo input (`cmp` of the volume files,
job 32361903); the oracle checks of every new file give at most 1.6e-13, no cell missing (job 32371853).

| A100, 1305 MHz | jittered | blue noise | uniform | clustered | halo | B/cell |
|---|---|---|---|---|---|---|
| full warp rebuild, device | 37.8 | 34.8 | 19.1 | 16.6 | 15.2 | 671 |
| volumes only, device | 43.1 | 40.4 | 24.3 | 16.7 | 17.0 | 101 |
| full rebuild, host to host, volumes and centroids copied out (before) | 19.7 | 18.4 | 13.6 | 11.6 | 11.0 | 696 |
| volumes only, host to host, pageable | 34.5 | 33.2 | 21.5 | 16.5 | 15.6 | 126 |
| volumes only, host to host, pinned buffers | 31.4 | 34.4 | 22.4 | 17.9 | 16.1 | 126 |
| volumes only, host to host, pageable, 256³ | 35.1 | 33.0 | 21.9 | 18.5 | 17.7 | |

Jobs 32368079 (128³) and 32368080 (256³), and 32368078 (blue noise with Liu et al., VoroGPU and Basselin et al. in the
same job: 18.1, 18.1 and 10.4 at 128³; 24.4, 19.7 and 11.6 at 256³). Pinned buffers are within the run-to-run spread of
pageable ones at 128³ and 5 to 11 % faster at 256³.

**CPU** (72 cores, 2x Xeon Platinum 8360Y, `OMP_PROC_BIND=spread OMP_PLACES=cores`, full cells with the cell list):

| | jittered | blue noise | uniform | clustered | halo |
|---|---|---|---|---|---|
| before (ad62997, job 32272650) | 2.6 | 2.2 | 1.7 | 1.4 | 1.3 |
| parallel radix sort of the host cell list (job 32362120) | 3.4 | 3.6 | 2.4 | 1.9 | 1.7 |
| + dynamic chunks of 16 (77055a4, job 32367276) | 3.4 | 3.8 | 2.6 | 2.0 | 2.0 |
| voro++ multithreaded, same job | 2.4 | 2.5 | 1.8 | 0.38 | 0.085 |

Phases on blue noise before (job 32361898): cell list 370 to 450 ms (one thread's `std::sort` of 2.1M keys), clipping
575 to 600 ms. The radix sort (8 bits a pass, per-chunk counts, stable, so the slot order is the same) takes 11 ms.
Dynamic chunks of 16 or 128 cut the clipping by 7 % (blue noise) to 17 % (halo); close binding changes nothing; 144
hyperthreads are 0 to 18 % slower (job 32362198). Volumes bit-identical to ad62997's on all five inputs.

**Default paths unchanged** (job 32368117, main 55cd6b1 against close-gaps, alternating, 1305 MHz, 128³): rebuild
42.6 / 21.9 M cells/s jittered / uniform on both; update + `update_cells()` 161.2 / 162.4 M cells/s at 0.03 spacings
and 99.8 / 99.6 at 0.1; clip, tetrahedra, ring, geometry, face and list checksums identical (the cells checksum hashes
face numbers and differs between any two runs, as before). A100 `ctest` 22 of 22 (job 32368083); CI's host steps (gcc 15,
host Kokkos 5.0.99 OpenMP at 2 threads, sanitizers off and on): 22 of 22 each, clang-tidy stops on `mesh_fuzz_test.cpp`
as on main (#56) (job 32367277).

### Application inputs: cosmic web, fcc solid, surface (2026-11-15)

The three application inputs of the Inputs table, 128³ and 256³ points (fcc solid: 2²⁰ and 2²³ atoms), point files from
`bench/libraries/make_points_a100.sh` (job 32367403). voronoi-cpp 631af46 (main c42c230 + the generators), one A100 at
1305 MHz, jobs 32367524, 32367883, 32367884; M cells/s, median of 3 (steps: of steps 2 to 10 at 0.03 spacings).

| M cells/s | cosmic web 128³ / 256³ | surface 128³ | fcc solid 2²⁰ / 2²³ |
|---|---|---|---|
| warp rebuild (cell list + clipping) | 12.0 / 12.8 | 0.036, 476 cells refused | 30.6 / 32.4 |
| exact start (`Mesh::update` + `update_cells`) | 5.1 / 4.9 | 0.026 | 12.9 / 9.5 |
| moving step, 0.03 spacings | 201 | 139 | 188 |

Peak device memory at 128³: 740 B/cell for the warp rebuild, 1869 B/cell for the exact start. Oracle checks (job
32369335, 1000 sampled cells each): every cell present and a largest relative volume error of 2.3e-13 for the exact
start, the moving steps, and the warp rebuild on cosmic web and fcc solid; the warp rebuild on the surface misses 476
cells (sum of volumes − 1 = −1.1e-3). `voronoi_bench rebuild --check` (job 32367880) passes on cosmic web 128³ and the
fcc solid (all certified, closure below 3e-16, volume sum within 5e-14); cosmic web 256³ fails only `faces=2*owned`,
the known one-sided zero-area faces of clipping alone. `step --verify` passes all 10 steps on cosmic web at 64³ and on
the fcc solid at 32³ unit cells.

Surface: clipping refuses cells (status degenerate: 69 at 64³, 476 at 128³, and one at 32³ on the OpenMP backend) and
runs about 300 times slower than on jittered points: its needle cells reach about 0.1 box into empty space, so each
clips about 6000 neighbours instead of 75. The exact start inherits the cost, since it clips first. Issue
[#60](https://github.com/Ulli1991/voronoi-cpp/issues/60), with a 953-point case. The mesh itself is right: Delaunay
exact in every step (`failing=0`), volumes within 2.3e-13 of the oracle, moving steps at 139 M cells/s.

### Surface after the fix of #60 (2026-11-15)

Branch `fix-60` against `new-inputs` c4e6249 (ref), one A100 at 1305 MHz, ref and new in the same gpu1 jobs
(`scripts/issue60_a100.sh`; jobs 32374838 to 32374852, 32375358 to 32375364). Every row M cells/s.

| surface | 64³ | 128³ | 256³ |
|---|---|---|---|
| warp rebuild, ref | 0.129, 69 refused | 0.036, 475 degenerate + 1 plane overflow | did not finish in 14 min (below 0.020) |
| warp rebuild, new | 0.41, none refused | 0.100, none refused | 0.041, none refused |
| exact start (library bench), ref / new | | 0.026 / 0.095 | |
| exact start: bad tetrahedra, patched cells, ref / new | 302, 1819 / 4, 101 | 11800, 75598 / 20, 404 | |

Oracle check (`oracle_check`, 1000 sampled cells, job 32375451): the new warp rebuild and exact start at 128³ have
every cell, sum of volumes − 1 = −7e-17 / −2e-17, largest relative volume error 3.9e-14 / 1.4e-13; ref's warp rebuild
misses 476 cells (sum − 1 = −1.1e-3). `voronoi_bench rebuild --check`: every cell certified at 64³, 128³ and 256³,
closure at most 3.1e-15, volume sum within 4e-14; 128³ and 256³ fail only `faces=2*owned`, the one-sided zero-area
faces of clipping alone, as cosmic web 256³ does on ref.

Causes. (1) Every refusal was a rounded cut: when the walk goes one level coarser it re-reads a grid cell the finer
walk did not cover and clips a particle it already clipped; the twin plane should cut nothing, but in a needle cell the
vertices of its face come from nearly parallel planes and round past the 64 ε |d| R tolerance, so the twin "cuts" its
own face and the cut cannot close (thread kernel: the two planes meet nowhere, det 0). The 953-point case of the issue
is this, for particle 255 and its neighbour 672 in slot order. Fix: cells a rounded build leaves neither ok nor out of
pool room are built again exactly. (2) The slowness: the cell list puts every surface point at the base level (64 of
them fill a node of mean density), so the warp kernel took its shortcut for uniform input and walked the base level
only; it left 74 % of the cells uncertified at 64³ (measurement build, job 32371403) to the thread kernel, one thread
per needle with thousands of candidates. Those cells now go to a filtered warp kernel (a cylinder around the cell rules
out grid cells and particles, a vertex-ball test the rest, planes are compacted when full), as do cells whose walk
passes 4096 clips. What is left: 2.5 % of the cells at 128³ grow past the warp kernel's 64 vertices and are built by the
thread kernel (filtered, from its big capacity), and that pass is most of the time (phase timers: 10.7 of 14.5 s at 128³
before the plane compaction).

The other inputs (jittered, blue noise, uniform, clustered, halo, lattice, wave, cosmic web, fcc solid), 128³ and 256³:
tets, rings, rings topology, start cells, geometry, faces and lists checksums of 3 steps and the warp rebuild's clip
checksums identical to ref; blue noise by the library bench's volume files, identical bit for bit. The per-step
`cells_checksum` differs between two runs of ref itself (job 32373870), so it is not compared. Wave at 256³ runs out of
device memory in the step bench in both builds; its rebuild and start rates and its rebuild checksum are compared.
Rates, ref → new, rebuild at 128³ / 256³: −0.06 to −0.30 % / −0.07 to −0.46 %; exact start within the ±3 % noise of
its repeats (lattice and wave within 1 %). Blue noise through the library bench: within 0.5 %.

Tests: A100 ctest 22 of 22 (job 32375356), with the 953-point case of the issue (thread and warp kernels against the
exact one) and the surface input at 64³ (32³ on the host) in `clipping_test`; OpenMP `clipping_test` passes on the login
node. Fuzz on the A100 (job 32375357, profile huge, seeds 4e12 + 101 to 108): one exact, five refused for causes the
inputs have (coincident particles, reach), two out of time at 280 s; those two (105, 107) also run past 13 minutes on ref
(jobs 32376473, 32376490), so they are #49, not this change.

Not adopted, the owner's call: a clip budget of 256 instead of 4096 makes the surface 1.6 times faster at 128³ (0.16),
but changes the bits of the halo and cosmic-web tail cells (p99 210 to 268 clips), which restart filtered, and slows
those inputs by 5 to 6 % (rebuild) and 4 to 11 % (exact start).

### Surface: a 128-slot filtered warp kernel (2026-11-16)

Branch `fix-60` at 343f653 (main 8d7ac1f merged in) against main 8d7ac1f, one A100 SXM4 40 GB at its default clocks
(boost up to 1410 MHz), main and branch in the same gpu1 job (job 32406803, `build-jobs/final_B.sh`). Cells the
filtered warp kernel leaves past its 64 vertex or plane slots are built again by the same kernel with 128 slots (twice
the shared memory, 4 blocks per SM) before any reach the thread kernel. M cells/s:

| surface | 64³ | 128³ | 256³ |
|---|---|---|---|
| warp rebuild, main | 0.129, 69 refused | 0.038, 476 refused | did not finish in 14 min |
| warp rebuild, after 05e44e1 | 0.41 | 0.100 | 0.041 |
| warp rebuild, 128-slot kernel | 0.51 | 0.104 | 0.045 |
| exact start (library bench), main / branch | | 0.027 / 0.098 | |

Main's rates: job 32378814 to 32378815. Cells per path at 64³ / 128³ / 256³ (counted in an instrumented build of the
same commit): the unfiltered warp kernel certifies 24 / 65 / 52 %; the filtered warp kernel, from the cells it left
uncertified or past the budget, 73 / 29 / 43 %; the 128-slot kernel 0.6 / 0.6 / 1.5 %; the thread kernel builds the
unfiltered kernel's overflows, 1.7 / 5.8 / 3.2 %, of which 6 / 357 / 4261 cells go on to the exact kernel. That last
pass is now most of the time.

Check: `voronoi_bench rebuild --check` certifies every cell at all three sizes (closure at most 3.1e-15, volume sum
within 4e-14; 128³ and 256³ fail only `faces=2*owned`, as before). Oracle check, 1000 sampled cells at 128³: every cell
present for the warp rebuild and the exact start, sum − 1 = −7.7e-17 / −2.4e-17, largest relative error 3.9e-14 /
1.4e-13.

The other inputs, 128³ and 256³, against main: tets, rings, rings topology, start cells, geometry, faces and lists
checksums of 3 steps and the warp rebuild's clip checksum and topology identical on all eight of the step bench (wave
256³: start and rebuild only, as its step runs out of memory on main too); the library bench's warp and exact volumes
identical bit for bit on blue noise and uniform (128³) and blue noise and cosmic web (256³). Rates, main → branch:
rebuild −0.07 to −0.73 %; exact start −1.7 to +3.3 % (min of 3 repeats; uniform 128³ +9.6 %, main's slow repeat).
Library bench within its repeat noise.

Dropped, variant B2: sending the unfiltered warp kernel's own overflows to the filtered kernels too (4497a6c, jobs
32378800 to 32378816). Surface 1.01 / 0.29 / 0.100 at 64³ / 128³ / 256³ (2 to 2.7 times the kept variant), and the
other inputs' rebuild 0 to +5.7 % faster (cosmic web +5.2 / +5.7 %, halo +3.6 / +1.5 %, clustered +2.5 %), but the
filtered kernel cuts in another order, so `rings_checksum`, `clip_checksum` and `clip_topology` change on uniform,
clustered, halo and cosmic web (128³ and 256³), and `clipping_test` fails (volumes-only and full rebuild differ in 2 of
32768 clustered and 3 halo cells).

Tests on 343f653: A100 ctest 22 of 22; CI's host steps (GCC 15, OpenMP Kokkos) 22 of 22 with sanitizers off and on;
comment and format checks pass; clang-tidy 18 clean on `clipping_test.cpp`, `voronoi_bench.cpp` and
`library_bench.cpp`. A100 fuzz, profile huge, seeds 4e12 + 101 to 108: one exact, five refused for causes the inputs
have, 105 and 107 out of time at 280 s as on main (#49).

### Surface: the warp kernel's overflows to the filtered kernels (2026-11-16)

Branch `fix-60-b2` (proof job on 109b329; the kernel is unchanged since) against main fbe6a0c (the 128-slot kernel above), one A100 SXM4 40 GB at its default
clocks (boost up to 1410 MHz), main and branch in the same gpu1 job 32416930 (`build-jobs/final_B2.sh`; jittered 256³
start again in job 32422214). Cells the first, unfiltered warp kernel leaves past its 64 vertex or plane slots now go
to the filtered warp kernels (64 slots, then 128) instead of the thread kernel. A filtered cell's overflow is now
`deferred` in every output: the volumes-only output used to keep it as an overflow, so its cells went to the thread
kernel where a faces build used the 128-slot kernel. M cells/s:

| surface | 64³ | 128³ | 256³ |
|---|---|---|---|
| warp rebuild, main | 0.52 | 0.105 | 0.045 |
| warp rebuild, branch | 1.01 | 0.287 | 0.100 |
| exact start (library bench), main (job 32406803) / branch | | 0.098 / 0.246 | |

Cells per path at 64³ / 128³ / 256³ (instrumented build of the same routing, job 32378783): the unfiltered warp kernel
24 / 65 / 52 %, the filtered 64-slot kernel 74 / 32 / 43 %, the 128-slot kernel 2.1 / 3.4 / 4.1 %, the thread kernel 20
/ 988 / 14655 cells (0 / 0.05 / 0.09 %), and 0 / 5 / 182 cells exactly. Check: every cell certified at all three sizes
(128³ and 256³ fail only `faces=2*owned`, as before); oracle check, 1000 cells at 128³: every cell present, sum − 1 =
−8.7e-17 (warp rebuild) and −2.4e-17 (exact start), largest relative error 3.9e-14 and 1.4e-13.

Bits against main, 128³ and 256³:
- jittered, lattice, wave, fcc solid and blue noise: every checksum identical (blue noise: the library bench's volumes).
- uniform, clustered, halo and cosmic web: `clip_checksum`, `clip_topology` and `rings_checksum` change. Identical:
  `tets_checksum`, `rings_topology`, `assembly_checksum`, `start_arrays_checksum`, the build's `cells_checksum` and
  after 3 steps `geometry_checksum`, `faces_checksum` and `lists_checksum`, and the exact start's volumes. These are the
  cells that overflow the unfiltered warp kernel, now clipped in another order. Per-cell dumps of the rebuild at 128³
  (job 32414966): 32 uniform, 62 clustered, 224 halo and 898 cosmic-web cells differ (114,544 on the surface), all with
  the same status, face count, owned count and set of neighbours; volumes within 4.8e-15, 1.9e-14, 1.8e-14 and 5.5e-15
  relative (surface 2.0e-13). `clip_topology` changes although no face does: the faces output stores a cell's faces in
  plane order, which is the clipping order. The rings are canonical (increasing neighbour id), so `rings_topology` stays.

Routing is now the same in every output: `clipping_test` compares a volumes-only, a rings and a faces build at 32³ on
nine inputs, to the bit on the A100. Main fails it: the rings build differs from the faces build in 72 clustered and 17
halo cells (the thread kernel sums a rings cell's faces in neighbour-id order), and on the surface the volumes-only
build in 39 cells and the rings build in 142. On the host, where the thread kernel builds every cell and sums
a rings cell's faces in id order, only the volumes-only build is compared (to the bit).

Rates, main → branch, rebuild at 128³ / 256³: cosmic web +5.5 / +6.1 %, halo +3.9 / +2.2 %, clustered +2.8 / −0.2 %,
uniform +0.4 / −0.3 %, the rest within 0.1 %. Exact start: cosmic web +4.0 / +4.1 %, halo +3.3 / +1.2 %, clustered
+1.7 / +0.8 %, the rest within −2.3 to +1.3 % except uniform 256³ +7.3 % (one round); jittered 256³ −11.5 % in one
round, −1.0 % in three (job 32422214), with the same bits. Library bench: cosmic web 256³ warp +6.0 %, exact +11 %
(repeat noise about ±5 %); blue noise and uniform within 0.6 % (warp) and +0.4 to +5.3 % (exact).

Tests on the branch: A100 ctest 22 of 22; CI's host steps 22 of 22 with sanitizers off and on (job 32423951; `clipping_test` on the A100 there too);
comment and format checks pass; clang-tidy 18 clean on `clipping_test.cpp`, `voronoi_bench.cpp` and
`library_bench.cpp`. A100 fuzz, profile huge, seeds 4e12 + 101 to 108: as on main (one exact, five refused for causes
the inputs have, 105 and 107 out of time, #49).

The other libraries (`bench/libraries/results/summary.md`, section "Application inputs"; A100 chains 32367907 to
32367911 and 32370411, CPU jobs 32367538 to 32367544): the kNN codes lose cells wherever a cell has more neighbours
than K. On cosmic web VoroGPU K 190 misses 2.4 % of the cells (sum of volumes − 1 = −0.038), K 90 and Liu et al. 19 %,
Basselin et al. (white-noise setting) 673 cells; on the surface they lose almost every cell. On the fcc solid every
code is complete, the float codes with relative errors up to 4e-5. peclet-voro returns wrong volumes on the surface
(sum − 1 = +31 to +62) and runs out of memory at 256³; votess fails on cosmic web and surface (A100: CUDA error 700;
CPU: `std::length_error`); gDel3D's 8-spacing padding is too thin for the surface (82627 certificate violations, one
sampled cell off by 1.3 %); Paragram leaves 647 surface cells without a volume (22 of 1000 sampled cells off by more
than 1 %) and 5 at cosmic web 256³. The exact Delaunay codes (CGAL, geogram, Qhull, TetGen) and voro++ are right on
all three (oracle jobs 32369335 and 32370726). Fastest full cells besides ours, cosmic web 128³: gDel3D + our faces 0.87
and voro++ (72 cores) 0.83 M cells/s; surface 128³: geogram (72 cores) 0.69, gDel3D 0.68 (too thin padding); fcc solid
2²⁰: peclet-voro 3.7 (A100).

### When the step falls back: host finish and rebuild (2026-11-15)

No run of the stage figure reached the host finish or the rebuild. These runs look for motions that do.

**Triggers** (main 8d7ac1f):
- `Mesh::update` rebuilds from scratch on the first call or when the particle count changes (`voronoi_mesh.hpp:219`).
- The device repair (`tet_repair::repair`, at most 1000 rounds, `voronoi_mesh.hpp:591`) lists a triangle as stuck
  when its tetrahedron is inverted after the move (`tet_repair.hpp:509`) or no flip repairs it (`:574`). Out of rounds
  with triangles left is `unfinished` (`:730`).
- **Host finish** (`tet_flips::finish`) runs if some entries are stuck, the repair finished, and the policy is `host`,
  or the default `cheapest` with stuck × 200000 ≤ n (`voronoi_mesh.hpp:232`). At 128³ that is at most 10 entries;
  below 200000 cells it never runs. It repairs a region of 8, then 16 hops around the entries on the host
  (`tet_flips.hpp:403`). It gives up after 2²⁰ tetrahedra of local Delaunay builds (`host_budget`,
  `voronoi_mesh.hpp:161`; `tet_flips.hpp:461`) or on an exception (`voronoi_mesh.hpp:237`).
- **Rebuild** runs if entries are still stuck (more than the policy allows, or the host gave up), the repair ran out
  of rounds, the volumes do not sum to the box within 1e-6 (`overlapped`, `:252`), or the face count fails Euler's
  formula (`miscounted`, `:253`). A rebuild whose exact start fails is a refusal (`runtime_error`, `:297`).

**Motions** (`voronoi_bench step`, branch `fallback-cases`; new options are in the bench only):
- the smooth flow at 0.5, 1 and 2 spacings per step;
- a share of the points jumping a fixed distance each step in a random direction (`--jump-share`,
  `--jump-spacings`), on top of the flow at 0.1 spacing. Two particles (share 1e-6) to 2 % of them, 1 to 4 spacings;
- clustered, halo, cosmic web, lattice, wave and surface inputs under the flow;
- `--fallback=host|rebuild` to set the policy; `--no-start` skips the bench's own start timing.

`scripts/fallback_cases_a100.sh`, one A100 40 GB (gpu1) at 1305 MHz (nvidia-smi, every job), 128³ (2,097,152
cells), 12 steps (30 for the frequency runs, 6 on the surface). Jobs 32378332 to 32378334, 32378770 to 32378773 and
32378981. Means over steps 2 to the last, not medians, so that rare fallbacks count. The rate is cells over the mean
of the whole step (repair, host, rebuild, geometry, cells). Data: [docs/data/fallback_cases.csv](data/fallback_cases.csv)
(columns of the paper's stages.csv, plus steps, steps used, stuck entries and the rate) and
[fallback_cases_steps.jsonl](data/fallback_cases_steps.jsonl) (every step).

ns per cell per step (mean of steps 2 on; ms per step = ns per cell × 2.097):

| case (128³, policy `cheapest` unless noted) | repair | host | rebuild | geometry | cells | steps host / rebuild | M cells/s |
|---|---|---|---|---|---|---|---|
| jittered, flow 1 spacing | 13.9 | 0 | 0 | 8.34 | 2.39 | 0 / 0 of 11 | 40.6 |
| jittered, flow 2 spacings | 29 | 0 | 0 | 14 | 2.86 | 0 / 0 of 11 | 21.8 |
| clustered, flow 0.3 spacings | 4.83 | 0 | 0 | 4.08 | 2.46 | 0 / 0 of 11 | 88 |
| clustered, flow 0.5 spacings | 7.06 | 0.346 | 0 | 5.53 | 2.42 | 1 / 0 of 11 | 65 |
| cosmicweb, flow 1 spacing | 15.4 | 0.228 | 0 | 10.2 | 3.09 | 3 / 0 of 29 | 34.6 |
| jittered, jumps, share 1e-6, 1 spacing | 4.62 | 0.811 | 37.3 | 5.01 | 2.33 | 3 / 7 of 11 | 20 |
| jittered, jumps, share 1e-6, 1 spacing, `host` | 5.05 | 50.1 | 0 | 3.62 | 2.37 | 10 / 0 of 11 | 16.3 |
| jittered, jumps, share 1e-6, 4 spacings | 5.44 | 0 | 59.6 | 5.96 | 2.37 | 0 / 11 of 11 | 13.6 |
| jittered, jumps, share 1e-3, 4 spacings | 15.7 | 0 | 60.6 | 6.01 | 2.37 | 0 / 11 of 11 | 11.8 |
| jittered, jumps, share 1e-2, 4 spacings | 47.3 | 0 | 68.2 | 6.29 | 2.41 | 0 / 11 of 11 | 8.05 |
| jittered, jumps, share 1e-3, 4 spacings, `host` | 15.5 | 2.12 µs | 61 | 6.06 | 2.38 | 11 / 11 of 11 | 0.453 |
| surface, flow 0.3 spacings | 12.3 | 0 | 8.39 µs | 5.91 | 2.75 | 0 / 1 of 5 | 0.119 |
| surface, flow 1 spacing | 39.3 | 83.9 | 40.8 µs | 5.91 | 2.82 | 1 / 5 of 5 | 0.0244 |

What the runs show:
- **The smooth flow rarely sticks.** Jittered at 0.5, 1 and 2 spacings, uniform at 1, halo at 0.3, lattice and
  wave at 0.5 and 1: no stuck entry in any step; the repair alone grows (29 ms at 1 spacing, 61 ms at 2).
  (An older build stuck in most steps at 1 spacing, docs/figures/flow_step.png; main 8d7ac1f does not.)
  Clustered at 0.3 (2 of 6 steps on an older build) no longer sticks. Clustered at 0.5 stuck in 1 of 11 steps (6 entries, host 8 ms) in one run and in none of 29 in another (the
  flip order follows atomics). Cosmic web at 1 spacing: 3 of 29 steps, 5 or 6 entries each, host 4 to 8 ms.
- **A few particles jumping go to a rebuild.** Two particles jumping 2 or 4 spacings leave 20 to 50 stuck entries,
  more than the 10 the default allows at 128³, so every step is a rebuild (118 to 125 ms against 11 ms of repair).
  At 1 spacing two jumpers leave 5 to 21 entries: 3 of 11 steps go to the host (4 to 8 ms), 7 to a rebuild.
- **The host is not cheaper there.** With `--fallback=host` the same two jumpers at 1 spacing take 4 to 572 ms of
  host time per step (a second, 16-hop round costs 66 to 572 ms), 105 ms on average against 78 ms of rebuild with
  `cheapest`. At 2 and 4 spacings the host takes 0.4 to 1.1 s per step and still gives up in half the steps or more,
  then rebuilds. At 1e-5 and 1e-4 jumpers it always gives up (1.1 to 1.3 s, then the rebuild); at 1e-3 (38000 entries) after 4.4 s.
- **On the surface a rebuild costs minutes.** At 0.3 spacings the first two steps stick (65 and 14 entries) and
  rebuild in 82 and 88 s; at 1 spacing every step sticks (10 to 1498 entries) and rebuilds in 79 to 90 s, the
  clipping's needle cells (#60). One step went to the host with 10 entries (882 ms), which gave up.
- **No step overlapped, miscounted, ran out of rounds or was refused.** `--verify` (each step against a fresh
  rebuild; job 32378981) passes all 48 steps of cosmic web at 1 spacing (2 host finishes), clustered at 0.5 (1),
  two jumpers at 1 spacing (3 host finishes, 8 rebuilds) and two jumpers at 2 spacings with `host` (12 host
  finishes, 4 of them then rebuilt). No wrong mesh, crash or hang, so no issue filed.

For the stage figure: host finish shows in the stage bars only for a few particles jumping about one spacing
(two jumpers at 1 spacing: 0.81 ns per cell on average, 4 to 8 ms in the steps that use it) or with `host` forced;
in the smooth flow it shows in single steps (cosmic web at 1 spacing, clustered at 0.5) and is under 1 % of the mean.
Rebuild shows whenever a few particles jump 2 spacings or more (60 ns per cell; the rate falls from about 96 M cells/s at 0.1 spacings, paper stages.csv, to 14),
and on the surface at large steps, where it is the whole cost (8 to 41 µs per cell).

### Local repair of stuck regions on the device (S2, 2026-11-18)

`tet_local::repair` (branch `local-repair`, `Fallback::local`, and `cheapest` up to n / 64 stuck entries) replaces
the tetrahedra around the stuck entries instead of rebuilding the mesh:
- **Affected set.** The particles of the stuck tetrahedra and of those across their stuck triangles (seeds), their
  first ring in the current tetrahedra, and the seeds' new neighbours: a jumped particle's old and new
  neighbourhoods.
- **Cells.** Those cells are clipped on the cell list of the current positions with the existing thread kernel
  (`shell_voronoi::detail::construct_cells`, face rings, the 2R certification): rounded cells first, exact ones
  (`ExactCell`) on a second attempt, exact and one ring wider (old and new neighbours of every cell) on a third.
- **Stitch.** The old tetrahedra with an affected corner go; the cells emit each tetrahedron with an affected corner
  once (by its lowest affected particle); a hash table of triangles checks that every new face meets exactly one other
  new face or one kept boundary face. The tetrahedra are written back by `tet_flips::write_region` (shared with the
  host finish: border, edge representatives, touched list for the incremental face numbering of #53).
- **Check.** The device repair runs on the new tetrahedra and their neighbours with the exact predicates; an entry left
  stuck starts the next attempt, and after the third, or past n / 8 affected cells, the step rebuilds. An allocation
  failure before the write-back also rebuilds.

One A100 40 GB (gpu1), 1305 MHz application clock (nvidia-smi in every job). main is dcb9b3d with the bench options of
`fallback-cases` (`voronoi_bench_ref`), the branch 4dd3ab8 (`voronoi_bench_new`), in the same jobs. Means over steps 2
to 12 of a flow at 0.1 spacings per step with a share of points jumping each step (`--jump-share`, `--jump-spacings`),
or of the smooth flow; the rate is cells over the whole step (repair, fallback, rebuild, geometry, cells).
`scripts/local_repair_a100.sh`; raw logs in `docs/data/raw/local-repair-*`.

**Correctness.** Every step compared with a fresh exact rebuild of the same points (`--verify`: no failing
triangle, Delaunay neighbours and faces equal, cell volumes equal to 1e-10; the few faces of zero area that the
check's clipping drops are main's too):
- job 32513179, local repair forced, 6 steps each: two jumpers at 1, 2 and 4 spacings, shares 1e-4, 1e-3 and 1e-2 at
  4 spacings, cosmic web at 1 spacing and clustered at 0.5: 48 of 48 steps pass; 29 steps were repaired locally (up to
  174254 cells), 1e-2 rebuilt (past n / 8), cosmic web and clustered did not stick in these 6 steps;
- job 32513187, default policy, 16 steps of cosmic web at 1 spacing (2 local repairs) and of clustered at 0.5 (none
  stuck): 32 of 32 pass; jobs 32515533 and 32515534, clustered at 0.5 and 0.7 spacings, default policy, 16 steps each: 2 and 4 local repairs (6 stuck entries, 55 to 60 cells each), 32 of 32 pass;
- job 32513186, surface 64³ at 0.3 spacings, local forced: 6 of 6 pass (one local repair: 8 stuck entries, 101 cells, 119 ms);
- earlier binaries of the branch: jobs 32417473 and 32417793, 81 of 81 steps.

**Speed, 128³** (job 32513180; M cells/s; main's default against the branch's default, `cheapest`):

| case | main | branch | branch, local forced | stuck entries | cells repaired | local repair ms (index, clip, stitch, check) |
|---|---|---|---|---|---|---|
| 2 jumpers, 1 spacing | 18.4 (host 3, rebuild 7 of 11 steps) | 62.3 | 58.2 | 13 | 81 | 10.5 (2.9, 2.8, 1.4, 0.9) |
| 2 jumpers, 2 spacings | 13.2 (rebuild every step) | 54.3 | 56.7 | 30 | 132 | 13.7 (3.1, 4.1, 1.6, 1.1) |
| 2 jumpers, 4 spacings | 13.5 | 51.2 | 51.6 | 39 | 177 | 15.9 (2.0, 5.8, 1.9, 1.1) |
| share 1e-4, 4 spacings | 12.2 | 35.6 | 37.1 | 3790 | 18000 | 21.8 (2.2, 9.9, 2.6, 2.6) |
| share 1e-3, 4 spacings | 11.1 | 11.5 (rebuild) | 13.4 | 37800 | 157000 | 83 (2.3, 47, 7.1, 18) |
| share 1e-2, 4 spacings | 7.6 | 7.9 (rebuild) | 7.7 (rebuild, past n / 8) | 376000 | — | — |
| cosmic web, flow 1 spacing | 39.2 | 37.3 | 36.8 | 1 (2 of 11 steps) | 10 | 3.2 |
| clustered, flow 0.5 spacings | 65.5 (host in 1 step) | 62.8 | 66.6 | 1 (1 of 11 steps) | 5 | 1.8 |

The flows stick in different steps from run to run (the flips follow atomics), so their rows compare runs, not steps.
After a local repair the geometry pass is cheaper than after a rebuild (7.8 against 12.6 ms), which only renumbers
the touched faces.

**Speed, 256³** (job 32513182; M cells/s):

| case | main | branch | branch, local forced | cells repaired |
|---|---|---|---|---|
| 2 jumpers, 1 spacing | 73.4 (host finish in 9 of 11 steps, 117 ms mean) | 124.5 | 122.9 | 56 |
| 2 jumpers, 2 spacings | 15.6 (host every step, 610 ms mean; 2 rebuilt) | 118.6 | 119.1 | 130 |
| 2 jumpers, 4 spacings | 5.5 (host 1.0 s, then a rebuild of 1.84 s, every step) | 115.8 | 115.8 | 174 |
| share 1e-4, 4 spacings | 8.5 | 58.9 | 60.3 | 208000 |
| share 1e-3, 4 spacings | 8.0 | 8.1 (rebuild) | 8.1 (55 % of steps rebuilt) | 631000 |
| share 1e-2, 4 spacings | 6.5 | 6.4 (rebuild) | 6.2 (rebuild, past n / 8) | — |
| cosmic web, flow 1 spacing | 69.9 | 67.7 | 69.1 | 17 |
| clustered, flow 0.5 spacings | 105.8 | 105.8 | 106.4 | — |

At 256³ a local repair of a few cells costs 22 to 30 ms, of which the cell list (5 ms) and the stitch's scan of all
110 million tetrahedra (8 ms) do not depend on the cells repaired. At share 1e-3 (1.39 million cells) the rounded
cells' stitch fails in most steps and the exact second attempt (0.7 s) closes it in most, but not all
(job 32513188: 4 of 6 steps local, the rest rebuilt after a third attempt would pass n / 8).

**Crossover, 128³** (job 32513181; local forced against rebuild forced, M cells/s):

| share (4 spacings) | stuck entries | cells repaired | local | rebuild |
|---|---|---|---|---|
| 1e-5 | 370 | 1806 | 45.6 | 11.8 |
| 1e-4 | 3750 | 17955 | 28.8 | 12.2 |
| 3e-4 | 11400 | 53607 | 26.5 | 11.7 |
| 1e-3 | 37800 | 173101 | 14.4 | 10.9 |
| 3e-3 | 113000 | (797000, past n / 8: rebuilt) | 10.1 | 10.1 |
| 1e-2, 3e-2 | 0.38 M, 1.1 M | past n / 8: rebuilt | 7.6, 4.9 | 7.7, 5.0 |

Before the n / 8 limit (job 32417474, same code otherwise) local took 4.5 M cells/s at 3e-3 (798000 cells) and 2.1 at
1e-2 (1.6 million) against 10.7 and 8.1 for a rebuild: the thread kernel clips a cell several times slower than the
rebuild's warp kernel, so the crossover is near 10 % of the cells repaired (about 200000 at 128³), between shares of
1e-3 and 3e-3. `cheapest` takes the local path up to n / 64 stuck entries (32768 at 128³; at 1e-3, 37800, it rebuilds,
where local would be 1.3× faster); past n / 8 affected cells the local path itself rebuilds.

**Default path unchanged.** With nothing stuck no new code runs. Seven inputs (jittered, uniform, clustered, halo,
lattice, wave, blue noise) at 0.03, 0.1 and 0.3 spacings, 10 steps, main and branch twice each, alternating (job
32513184): `geometry_checksum`, `faces_checksum` and `lists_checksum` equal in all 21 cases and all 10 steps. Median
step rates differ by −0.56 to +1.24 % (lattice at 0.3, faster), all others within ±0.9 %; means −0.86 to +2.53 %.

**Tests.** A100 ctest 22 of 22 (job 32513189), with six new cases `fallback.local.jumpers.{2,33}.{1,2,4}` (2 or 33
particles jumping 1 to 4 spacings at 32³, each step repaired locally without a rebuild and checked against an oracle
and a fresh build) and `Fallback::local` in the shells, large and growth cases. CI's host steps on gpu1 (job
32513191): comments and format pass (format fixed after the job, `ae4c431`), ctest 22 of 22 with sanitizers off and on,
clang-tidy 18 clean on `voronoi_bench.cpp`, `mesh_cases/fallback.cpp` and `repair_test.cpp` with the headers. A100
fuzz, profile huge, 15 minutes (job 32513190): 10 cases, 2 exact, 6 refused for causes the input has, 2 out of time
(seeds 4000000000002 and 4000000000010, which run out of 900 s alone on main as well, job 32428400), no WRONG or CRASH.

**Re-proven against main 3f27c51** (the cpu, gpu and hybrid modes; the local repair's rounded cells follow
`Layout::same_bits_as_host`), all in one gpu1 job (32515907, `scripts/local_repair_proof_a100.sh`, branch c5d2ed8):
- A100 ctest 23 of 23. `modes` has a new case, jittered 32³ at 0.1 spacings with 4 particles jumping 4 spacings:
  gpu and hybrid each repair 4 of 5 updates locally, and hybrid equals cpu in every bit (163840 of 163840 volumes).
- `--verify`, local forced, the eight cases above: 48 of 48 steps pass.
- Rates, M cells/s, main → branch (`cheapest`): 128³ two jumpers at 1, 2, 4 spacings 18.7, 11.8, 12.7 → 55.4, 51.1,
  52.3; share 1e-4 11.8 → 29.5; 1e-3 10.3 → 11.2 (both rebuild). 256³: 71.6, 13.8, 6.3 → 130.7, 119.2, 119.8; share
  1e-4 8.5 → 68.7; 1e-3 8.2 → 8.2 (local forced 14.3).
- Default path, seven inputs at 0.03, 0.1, 0.3 spacings: checksums equal in all 21 cases. Median rates within
  −0.65 to +0.61 % in 20; clustered at 0.3 −1.35 % (mean −5.5 %): in both of the branch's runs step 2 left 6 stuck
  entries, repaired locally in 12 ms, where main's runs did not stick. The repair code is main's; which steps stick
  follows the flips' atomics and differs between binaries (as between runs of main above).
- CI's host steps: comments and format pass, ctest 23 of 23 with sanitizers off and on, clang-tidy 18 clean on
  `voronoi_bench.cpp`, `mesh_cases/fallback.cpp`, `repair_test.cpp` and `modes_test.cpp`.

### Long runs

The rates above are early steps. On a smooth flow where every cell moves, the mesh loses its memory order and the step slows from 124 M cells/s at step 100 to 61 at step 500 (128³, 0.1 spacings, update and `update_cells()`, [#34](https://github.com/Ulli1991/voronoi-cpp/issues/34)). A Hilbert re-sort every 50 updates (`Layout`) keeps it: 121 M cells/s over 500 steps against 81.6 without, re-sorts paid (each about one step's time); at 256³ 166 against 127. Hilbert, Moore and Morton are within 1 %, row-major and reverse Cuthill-McKee 11 to 13 % slower. Flows with little bulk motion do not show the decay: on Sedov the loss is at most 4.5 % over a run. Tables in [measure_resort_2026-11-10.md](measure_resort_2026-11-10.md).

Since 2026-11-18 the re-sort is on by default, every 25 updates: over 500 steps 142 M cells/s against main's 65.9 at 128³, 0.1 spacings (2.16×), 196 against 88.7 at 256³ (2.20×), 1.65× at 0.03 and 2.15× at 0.3 ([step-2x](#the-2-step-branch-step-2x-2026-11-18)).

Job 31953232 (128³, 0.1 spacings per step): the step falls from 118 M cells/s at step 100 to 59 at step 500. Issue: https://github.com/Ulli1991/voronoi-cpp/issues/34. On Sedov the loss is at most 4.5 % over a run.

### GPU against a full CPU node, same code

Older code built with the OpenMP backend on one Raven CPU node (2x Xeon Platinum 8360Y, 72 cores), against one A100, at 2,097,152 cells in double, in M cells/s:

| | 72-core CPU node | A100 | A100 / CPU |
|---|---|---|---|
| Full rebuild, jittered / white noise / clustered | 4.1 / 2.8 / 2.0 | 39.9 / 19.9 / 15.5 | 9.7 / 7.0 / 7.9 |
| Moving-mesh update, 0.03 / 0.3 spacings | 11.5 / 1.7 | 202 / 57 | 17.5 / 33 |
| `update_cells()` | 43 | 343 | 7.9 |

Every phase gains more than the A100's 3.9 times the node's memory bandwidth. The CPU build is the generic Kokkos path with no hand vectorisation, so a CPU-tuned code would narrow the gap. Since 2026-11-15 the host cell list sorts with all threads (a stable radix sort; one thread's `std::sort` took about 400 of 1000 ms) and the thread kernel takes cells in dynamic chunks of 16: the rebuild with the cell list on 72 cores went from 2.2 to 3.8 M cells/s on blue noise and from 1.7 to 2.6 on white noise, the same bits ([library_benchmark.md](library_benchmark.md)).

Commit 57201bc built with the Kokkos OpenMP backend on one Raven CPU node (2x Xeon Platinum 8360Y, 72 cores, `OMP_PROC_BIND=spread`, job 31796079), against one A100 40 GB (jobs 31795767, 31795035). 2,097,152 cells, double.

Hardware ratios A100 / node: about 3.9 in memory bandwidth (1.55 TB/s against about 0.4 TB/s) and 1.8 in FP64 peak. Every phase gains more than the bandwidth ratio. The clipping is latency- and divergence-bound (DRAM traffic under 1 % on the GPU), the flip repair is many small parallel rounds with atomics, and even the bandwidth-like hydro passes gain 1.6 to 3x beyond bandwidth. The CPU build is the generic Kokkos OpenMP path (per-thread clipping, no hand vectorisation), so a CPU-tuned code would narrow the gap.

### 2D and 1D

`voronoi::Mesh<2>` on one A100 at 1305 MHz, double, 0.03 spacings per step:

| M cells/s | jittered | white noise | clustered | lattice |
|---|---|---|---|---|
| Build, 2²⁰ cells | 100 | 50 | 44 | 115 |
| Step (repair, faces, cell lists), 2²⁰ | 765 | 791 | 789 | 749 |
| Build, 2²⁴ cells | 134 | 58 | 52 | 168 |
| Step, 2²⁴ | 1056 | 1059 | 1061 | 1023 |

`voronoi::Mesh<1>`, mesh phases of a moving Sod run: 18,905 M cells/s at 2²⁰ and 39,351 at 2²⁴ (1095 MHz, older code).

`voronoi::Mesh<2>`: job 31952137, one A100 at 1305 MHz, double, Morton order, 0.03 spacings per step. Jittered at 0.1 / 0.3 spacings: 770 / 744 at 2²⁰, 1024 / 1015 at 2²⁴. Peak device memory 537 to 574 B/cell (3D: 1998 to 2028 B/cell at 2.1M, job 31952136). Details: research_notes/mesh_2d_2026-11-05.md.

`voronoi::Mesh<1>`: job 31888427, mesh phases of a moving Sod run at 1095 MHz (older than this branch, research_notes/stage1_1d_2026-11-04.md).

### After the fixes of #29, #30, #35, #36 (2026-11-09)

128³ rows of the README, one A100 at 1305 MHz, origin/main 11745ea against the fix branch in the same jobs: rebuilds job 32094782, moving steps job 32094783 (median of steps 2 to 9). Steps 0.03 / 0.1 / 0.3: 204-208 / 117-118 / 57.0 before, 198-200 / 115-116 / 56.5 after. Rebuilds unchanged at 39.4 / 19.7 / 15.4. Geometry checksums identical in the unit cube at 0.03 and 0.3, different in a cube of side 0.05 (job 32095015).

256³ column: job 32093328 at 1095 MHz, voronoi-cpp 2190882 (before the fixes); steps include `update_cells()`.

### After the fix of #45 (2026-11-10)

128³, one A100 at 1305 MHz, origin/main 61e9914 against branch `fuzz-45` in the same job 32146167 (median of steps 2 to 9, two rounds each). Steps 0.03 / 0.1 / 0.3: 198.4 / 115.5 / 56.2 before, 195.5 / 113.5 / 55.7 after (1.5 / 1.7 / 0.9 % slower); `tet_mesh::faces` unchanged at 2.60 ms, `tet_repair::screen` 2.11 to 2.28 ms. Rebuilds unchanged (38.65 / 19.15 jittered / uniform). Geometry checksums differ in the unit cube and in a cube of side 0.05: centres stored from a corner in another periodic image now round at their wrapped position. A first version with a division per face vertex cost 4 to 10 % (jobs 32145669, 32145809).

### Mesh fuzz campaign (2026-11-06)

Driver `src/test/mesh_fuzz_test.cpp`, campaigns `scripts/mesh_fuzz_cpu.sh` and `scripts/mesh_fuzz_gpudev.sh` (the A100 campaign at millions of points is below), results in `build-fuzz/` on Raven: 3D on the CPU 115,598 cases (`camp2`, `camp3`), 1D 1,978,498, 2D 31,389, with sanitizers and bounds checks 731 (`asan_out`), on the A100 316 (`cuda_out`), none of them with millions of points. The 423 unexplained 3D failures rerun after the fix of #29 left 24 (#45).

### A100 mesh fuzz at 3e4 to 1.6e7 particles (2026-11-10)

`scripts/mesh_fuzz_a100.sh`, branch `fuzz-45` (the #45 fix), jobs 32145583, 32146360 and 32146363 (11 node-jobs of 15 minutes, four A100 each); results in `build-cuda/a100/results/` of the `voronoi-cpp-fuzz45` worktree on Raven, HANG rechecks in `build-cuda/a100/hang/`. The first jobs ran the fix with a division per face vertex, the rest the committed code (same values but for exact half-period ties). 604 cases: 273 exact, 239 refused (130 coincident, 94 reach, 15 box range), 2 rough, 2 valid (more than 16000 planes in one cell), 3 WRONG, 85 HANG.

| Particles | Cases | exact | refused | other |
|---|---:|---:|---:|---|
| 3e4 to 1e5 | 239 | 116 | 121 | 1 rough, 1 WRONG |
| 1e5 to 1e6 | 167 | 88 | 76 | 2 valid, 1 WRONG |
| 1e6 to 4e6 | 68 | 37 | 30 | 1 rough |
| 4e6 to 1.6e7 | 45 | 32 | 12 | 1 WRONG |
| out of time | 85 | | | |

Profiles large / huge / giant: 432 / 56 / 116 cases. Every box family (cube, pow2, 123, ratio, ratiolong, hcp, long, rod, slab, thin, extreme) and generator (lattices sc, bcc, fcc, hcp, diamond; uniform, Poisson disc, quasi; jitter; Gaussian, power-law, shell, Fibonacci, filament, sheet, spiral, cusp clouds; density steps, voids, points on box faces, near and doubled pairs, coplanar and cospherical sets) had at least 22 cases. The 3 WRONG are all #48 (seeds 3000000000159, 3000000000237, 7000000000041); with its fix (edges represented by their lowest tetrahedron) 3000000000159 and 7000000000041 are exact and 3000000000237 is refused for 66 coincident particles, as the input has them. The cases out of time are #49: four rerun alone were still in their first `update()` after 830 s; two rerun on 36 CPU threads refused for reach after 353 and 713 s. The generator `quasi` puts every point on one plane (1/g² + 1/g³ = 1 for the plastic number g), so its cases test only that.

### Per-vertex balls (P1) and directional boxes (R8) in the clipping (2026-11-12)

Branch `clip-bounds`; the bound with the measurement build (`-DVORONOI_MEASURE`, counters `rounds` to `clips_r8`), timings with the plain build, all on one A100 in jobs on `gpu1` and `gpudev`, 2,097,152 cells. Script `scripts/clip_bounds_a100.sh`; logs archived on Raven in `voronoi-cpp-joblogs/voronoi-cpp-clip/jobs/`.

**The bound** (job 32221981, the walk of main unchanged, the tests evaluated on the side). P1 for a grid cell: it meets the ball of a live vertex (with the margins of the skip); R8 for a grid cell: it meets the box around those balls. P1 for a clip: the particle lies in a vertex ball of the cell at its turn; R8 for a clip: the cell's bounding box does not rule it out. Rounds are the warp's particle rounds (one particle per lane each). Mean, p99 and max per cell:

| input | particles read: main, P1, R8 | rounds: main, P1, R8 | clips: main, P1, R8 |
|---|---|---|---|
| jittered | 76, 80, 80 / 32, 36, 47 / 32, 41, 63 | 3.0, 3, 3 / 1.04, 2, 3 / 1.06, 3, 3 | 21.9, 25, 28 / 16.0, 21, 26 / 21.7, 25, 26 |
| uniform | 103, 176, 344 / 57, 84, 121 / 76, 125, 206 | 15.1, 27, 49 / 9.8, 18, 30 / 12.2, 22, 37 | 48.0, 92, 188 / 17.9, 28, 45 / 39.1, 70, 121 |
| clustered | 112, 238, 1133 / 63, 141, 683 / 82, 167, 687 | 16.7, 38, 278 / 10.4, 20, 112 / 13.1, 27, 167 | 50.7, 106, 448 / 19.8, 34, 59 / 41.6, 75, 165 |
| halo | 115, 440, 3812 / 61, 130, 636 / 80, 180, 1184 | 18.0, 78, 535 / 10.5, 24, 138 / 13.4, 37, 259 | 53.6, 210, 2096 / 19.4, 34, 59 / 41.9, 88, 583 |

P1 leaves only the clips that cut (it is the clip's own test) and halves the particles read and the rounds on unstructured input, and cuts the tails most (halo p99 clips 210 to 34). R8 keeps 75 to 80 % of the rounds and 78 to 82 % of the clips there. P1 contains R8, so both together, or R8 cascaded before P1, leave what P1 leaves. On jittered input P1 and R8 remove two thirds of the reads and rounds; clips that cut nothing are 27 % there.

**Implementations, all slower** (median of 5 rebuilds, `construct_ms`, main and the variant in the same job, 1305 MHz on `gpu1` unless noted):

| variant | jittered | uniform | clustered | halo | lattice | wave |
|---|---|---|---|---|---|---|
| main 69de1d3, ms | 54.2 | 109.5 | 139.6 | 152.4 | 43.1 | 56.4 |
| P1 per candidate: each lane tests its buffered particle against every live vertex at the flush, then only new vertices after each cut (job 32222094, 1095 MHz) | +22 % | +15 % | +12 % | +10 % | +17 % | +18 % |
| P1 per grid cell, each lane against every live vertex, plus the above (job 32222186, 1095 MHz) | +64 % | +42 % | +41 % | +38 % | +36 % | +60 % |
| R8 box around the balls in double, per batch; grid cells and particles outside it skipped (jobs 32222688, 32222998, 32223108) | +18 to +24 % | +26 to +32 % | +26 to +34 % | +22 to +30 % | +14 to +22 % | +17 to +23 % |
| the same, grid cells only / particles only (jobs 32223854, 32223855) | +17 % / +19 % | +26 % / +26 % | +26 % / +25 % | +23 % / +21 % | +14 % / +16 % | +16 % / +18 % |
| R8 box in float by `__reduce_max_sync` (jobs 32224849, 32225604) | +14 % | +19 % | +22 % | +18 % | +10 % | +13 % |
| the bounds compiled out, only the clip's check for a vertex past its kept end (job 32224057) | +3 % | +5 % | +3 % | +3 % | +2 % | +3 % |

With the R8 box in use the work falls as counted (job 32222999: particles read 76 to 31 jittered, 103 to 75 uniform; clips 48 to 39 uniform; no cell sent to the thread kernel), yet the time rises. Nsight Compute (job 32224827, uniform): 4.6 % more instructions, but stalls for instruction fetch (`no_instruction`) rise from 28 % to 43 % of samples, and local loads from 175M to 258M (stack frames of 176 to 208 bytes become 192 to 256, registers capped at 96). The warp kernel is limited by its instruction cache and registers, so any test added to the walk costs more than the work it removes; even the variant with the bounds compiled out is 2 to 5 % slower. Lattice input, where no grid cell is skipped, shows the fixed cost: 10 to 22 %.

Rigour of the variants: a bound skipped a grid cell or particle only with a margin above rounding (the cell widened by 64 ε of the longest side; the ball box widened by 2^-14 R, above the rounding of 64 later cuts, which stay in the hull of the current vertices); a cut whose new vertex lies past its kept end (that end within the tolerance outside) would send a cell that skipped anything to the thread kernel. None of this is in the library: the branch keeps the counters only, and the plain build's kernel is the same SASS as main's.

### Canonical rings and a lean cut (2026-11-13)

Branch `clip-canonical` against main (9e9ec03, and after the rebase 6ead543; with the bench's checksums only, `ref`), one A100, every comparison in one job. Script `scripts/clip_canonical_a100.sh`, tables by `scripts/clip_canonical_tables.py` and `scripts/clip_canonical_ncu.py`; logs in `build-jobs/` of the `voronoi-cpp-canon` worktree on Raven. `canon` is commit a53ed48 (canonical rings only), `final` the branch head.

**Canonical rings.** Each cell writes its owned rings in increasing neighbour id, each from its least particle, in both kernels; the warp kernel also starts each face's fan there. `mesh_test` checks the form and that the exact thread kernel writes the same entries as the warp kernel (0 of 4096 cells differ, jittered and uniform). Cost: 0 to 1 % on the rebuild (job 32237666, 1305 MHz) and 0.3 % more instructions per cell (9611 to 9642 jittered, job 32269328).

**The lean cut.** A new vertex goes into the lowest slot free before the cut, built by the lane that owns it: no stack of free slots, no pushes and no 64-bit shifts for the slots made. The starting cube's adjacency is a formula (the search took 4 % of the kernel), the radius uses a select instead of `Kokkos::max` (which branched), the candidate sort and the side test have no branches, a face walk finds its corner with one load, and the reach of a candidate is recomputed only after a cut. The cut's decisions and vertex bits are main's.

Rate of the warp clipping (`rebuild`, median of 2 × 5, M cells/s and change against main; job 32238284 at 1305 MHz for 128³, the head's source but for formatting; job 32269162 at 1305 MHz for 256³, one round; job 32269153 repeats 128³ at 1095 MHz with the same changes):

| | jittered | uniform | clustered | halo | lattice | wave |
|---|---|---|---|---|---|---|
| main, 128³ | 38.7 | 19.2 | 15.0 | 13.8 | 48.6 | 37.2 |
| canonical rings, 128³ (job 32237666) | 38.4 (−1 %) | 19.0 (−1 %) | 15.0 (0 %) | 13.7 (0 %) | 48.7 (0 %) | 36.9 (−1 %) |
| lean cut, 128³ | 42.6 (+10 %) | 21.9 (+14 %) | 17.1 (+14 %) | 15.5 (+12 %) | 55.1 (+13 %) | 40.8 (+10 %) |
| main, 256³ | 38.8 | 19.5 | 16.3 | 15.8 | 50.9 | 37.3 |
| lean cut, 256³ | 42.8 (+10 %) | 22.4 (+15 %) | 18.5 (+14 %) | 17.9 (+13 %) | 57.6 (+13 %) | 41.0 (+10 %) |

Exact start (`start_ms`, median of 2, job 32269154, 1305 MHz), M cells/s: main 17.5, 8.3, 7.2, 9.9, 6.6, 4.0 against 18.5, 8.9, 7.3, 10.3, 7.6, 4.3 (jittered, uniform, clustered, halo, lattice, wave); the start's own kernels and host waits vary by 5 to 8 % between runs.

Nsight Compute of the warp kernel (job 32269328, 1095 MHz), warp instructions per cell, kernel time per clip over the whole GPU (21.9 clips a cell jittered, 48.0 uniform), and stall shares; no spills, 96 registers in all three:

| | jittered: instructions, ns per clip | stalls: fixed latency, shared memory, instruction fetch, issue | uniform: instructions, ns per clip | stalls |
|---|---|---|---|---|
| main | 9611, 1.40 | 35, 14, 13, 24 % | 15128, 1.26 | 29, 10, 35, 16 % |
| canonical rings | 9642, 1.41 | 35, 14, 14, 23 % | 15159, 1.27 | 29, 10, 36, 16 % |
| lean cut | 8802, 1.27 | 36, 16, 11, 24 % | 14020, 1.10 | 31, 12, 29, 18 % |

**Mesh.** After the build and two steps at 0.1 spacings, `tets_checksum` (the start's tetrahedra as particle sets), `geometry_checksum`, `faces_checksum` and `lists_checksum` equal main's on all six inputs at 128³ (job 32269155) and 256³ (jobs 32269156 to 32269160; wave at 256³ runs out of the A100's 40 GB in this bench's step on main too, so for it the start alone, job 32269402: the same tetrahedra). The bad tetrahedra and patched cells of the start are main's. What changed once: `rings_checksum` and the build's `cells_checksum` (the tetrahedra and faces are numbered from canonical rings now); the step's `cells_checksum` differs between runs of main as well. Canonical rings and the lean cut give the same rings and the same face geometry of a plain rebuild (`clip_checksum`) bit for bit on all six inputs, so the cut's slots no longer reach any output. luna-cpp (a82092c) builds against the branch; `hydro_sedov` 128³ jittered, moving mesh, 50 steps, gives the same bits of U, x and volume as against main (job 32238689).

**After the rebase on 6ead543** (E and K2 in the repair): rebuild 42.6, 22.0, 17.0, 15.5, 55.1, 40.9 against main's 38.7, 19.2, 15.0, 13.7, 48.7, 37.2 at 128³ (+10 to +15 %, job 32269868, 1305 MHz) and 42.8, 22.4, 18.5, 17.9, 57.7, 41.0 against 38.9, 19.5, 16.3, 15.8, 50.9, 37.3 at 256³ (job 32269874); exact start 18.6, 8.8, 7.9, 10.3, 7.4, 4.4 against 17.6, 8.2, 7.2, 9.6, 7.1, 4.0 (job 32269869); the same checksums against main as above at 128³ and 256³ (jobs 32269867, 32269870 to 32269873); `hydro_sedov` the same bits (job 32269941).

**A fuzz crash, fixed.** A 1e6 A100 fuzz batch (profile huge, seed 4000000000091: 1.3M particles, 38 of them coincident, an update after the refused build) hit an illegal address: a rounded cell with two faces toward one neighbour (#48) gave both rings the same place in the canonical order, the end of the block stayed unwritten, and `assemble` read a garbage length. Equal neighbours now go in lane order. Main refuses that case cleanly.

**What it says.** The bookkeeping that slots and cut order forced was worth 10 to 15 %, not several times: the cut that is left is mostly the work of finding and building the new vertices, and the kernel still waits on that chain's latency and, on uniform and clustered input, on instruction fetch. A rebuild several times faster needs fewer clips (B, the rounds of `clip-split`) or no clipping (D).

### kNN-first clipping, hybrids and a vertex-ball filter (2026-11-15)

Branch `fewer-clips`; the code is in commit c192c49 (`shell_knn.hpp`, the `listed` and `walk_todo` modes of the warp kernel, chosen by `VORONOI_KNN` and `VORONOI_BALLS`) and was taken out again, so the library is main's. One A100 at 1305 MHz on `gpu1`, main b2faee8 (`ref`) and the branch in the same job. Script `scripts/fewer_clips_a100.sh`, tables by `scripts/fewer_clips_tables.py`; logs in `build-jobs/` of the `voronoi-cpp-fewer` worktree on Raven.

**The designs.** *kNN pass*: one thread per particle walks the cell list as a linear octree (the particle's level, stencil cells nearest first, coarser levels as the walk does) and keeps the k nearest in registers, sorted by a float key; it stops when the next cell lies beyond the k-th. *kNN-first*: the warp clips by the list, nearest first, and stops when the next entry lies beyond 2R; past the list every particle lies at least as far as its last entry (with a margin of 2^-20 for the float keys), so a cell is certified if that entry lies beyond 2R. A cell the list cannot certify is built again by the walk (*fallback*). Hybrids: `:resume` goes on with the walk in the same warp, skipping the listed particles; `k+k2` lists the uncertified cells again with k2 before the walk; `:even` lists only cells at the base grid level (even density) and walks the rest; `:balls8` and `w8` (the walk) skip a candidate, from the 8th of a batch on, that lies outside every vertex ball by a margin above the rounding of later cuts (2^-36 (|d| R + |d|²); later vertices stay in the hull of the current ones), which leaves every cut and every vertex bit as without it. Memory of the lists: 4k + 4 B/cell during the rebuild (k = 24, 32, 48: 100, 132, 196 B/cell).

**Rate of the rebuild** (`construct_ms`, median of 5; 128³ two rounds, job 32319474; 256³ one round, jobs 32319849, 32319851), M cells/s and against main; fallbacks are the cells the walk built after the list, as a share of all:

| design | jittered | uniform | clustered | halo | lattice | wave |
|---|---|---|---|---|---|---|
| main's walk, 128³ | 42.6 | 21.9 | 17.0 | 15.5 | 55.1 | 40.8 |
| kNN-first k = 24 | 36.1 (−15 %) | 12.5 (−43 %) | 10.5 (−38 %) | 10.0 (−36 %) | 28.1 (−49 %) | 36.4 (−11 %) |
| fallbacks | 14.3 % | 95.2 % | 95.3 % | 95.3 % | 100 % | 12.7 % |
| kNN-first k = 32 | 37.7 (−12 %) | 12.6 (−43 %) | 7.8 (−54 %) | 6.6 (−58 %) | 49.8 (−10 %) | 38.4 (−6 %) |
| fallbacks | 0 | 79.1 % | 79.7 % | 79.6 % | 0 | 0 |
| kNN-first k = 48 | 33.4 (−22 %) | 14.8 (−33 %) | 11.4 (−33 %) | 10.8 (−31 %) | 41.8 (−24 %) | 33.4 (−18 %) |
| fallbacks | 0 | 31.8 % | 34.2 % | 34.2 % | 0 | 0 |
| k = 32, resume in the warp | 38.0 (−11 %) | 17.3 (−21 %) | 14.3 (−16 %) | 12.1 (−22 %) | 50.2 (−9 %) | 38.5 (−6 %) |
| k = 24, then 48, then the walk | 34.9 (−18 %) | 11.0 (−50 %) | 8.8 (−49 %) | 8.3 (−47 %) | 25.0 (−55 %) | 35.4 (−13 %) |
| k = 32 at the base level, resume | 38.0 (−11 %) | 17.2 (−21 %) | 13.5 (−21 %) | 12.3 (−21 %) | 50.2 (−9 %) | 38.5 (−6 %) |
| k = 32, resume, balls | 37.9 (−11 %) | 16.9 (−23 %) | 14.1 (−17 %) | 12.5 (−20 %) | 48.7 (−12 %) | 38.4 (−6 %) |
| walk with balls (`w8`) | 40.8 (−4 %) | 18.4 (−16 %) | 14.5 (−15 %) | 13.4 (−14 %) | 49.9 (−9 %) | 40.0 (−2 %) |
| main's walk, 256³ | 42.8 | 22.4 | 18.5 | 17.9 | 57.6 | 41.0 |
| kNN-first k = 24 | 36.8 (−14 %) | 12.7 (−43 %) | 11.5 (−38 %) | 11.2 (−37 %) | 29.2 (−49 %) | 37.2 (−9 %) |
| kNN-first k = 32 | 38.4 (−10 %) | 12.8 (−43 %) | 11.5 (−38 %) | 11.1 (−38 %) | 51.9 (−10 %) | 39.0 (−5 %) |
| kNN-first k = 48 | 33.8 (−21 %) | 15.2 (−32 %) | 12.9 (−30 %) | 12.3 (−31 %) | 43.6 (−24 %) | 34.0 (−17 %) |
| k = 32, resume in the warp | 38.5 (−10 %) | 17.7 (−21 %) | 15.3 (−18 %) | 14.1 (−21 %) | 52.9 (−8 %) | 39.3 (−4 %) |
| k = 24, then 48, then the walk | 35.8 (−16 %) | 11.3 (−50 %) | 9.8 (−47 %) | 9.3 (−48 %) | 26.5 (−54 %) | 36.2 (−12 %) |
| k = 32 at the base level, resume | 38.7 (−10 %) | 17.7 (−21 %) | 15.0 (−19 %) | 14.8 (−17 %) | 53.0 (−8 %) | 39.3 (−4 %) |
| k = 32, resume, balls | 38.5 (−10 %) | 17.4 (−22 %) | 15.0 (−19 %) | 14.1 (−21 %) | 51.1 (−11 %) | 38.9 (−5 %) |
| walk with balls (`w8`) | 41.0 (−4 %) | 18.7 (−17 %) | 15.7 (−15 %) | 15.3 (−14 %) | 52.2 (−9 %) | 40.2 (−2 %) |
| the branch's walk, balls off (256³) | 41.1 (−4 %) | 18.8 (−16 %) | 16.2 (−13 %) | 15.8 (−12 %) | 54.0 (−6 %) | 39.3 (−4 %) |

Fallbacks at 256³ are the same shares as at 128³ (k = 24: 14.3, 95.2, 95.2, 95.2, 100, 12.8 %; k = 32: 0, 79.0, 79.2, 79.2, 0, 0 %; k = 48: 0, 31.8, 32.6, 32.8, 0, 0 %). With resume, 26 to 110 cells of 2.1M (235 to 582 of 16.8M) still went to the walk kernel: incomplete lists.

**Where the time goes** at 128³, k = 32 (ms of the kNN pass, the listed clipping and the walk for the fallbacks, from the same job; main's whole kernel for comparison):

| | jittered | uniform | clustered | halo | lattice | wave |
|---|---|---|---|---|---|---|
| main, the walk | 49.2 | 95.8 | 123.4 | 135.3 | 38.1 | 51.4 |
| kNN pass | 15.4 | 20.1 | 34.8 | 33.0 | 8.8 | 9.4 |
| listed clipping | 45.4 | 44.2 | 44.1 | 44.1 | 32.3 | 44.4 |
| walk of the fallbacks | 0.2 | 98.9 | 114.0 | 117.5 | 0.2 | 0.2 |
| with resume: pass, clipping | 10.4, 43.9 | 20.1, 97.6 | 31.7, 110.7 | 33.0, 122.6 | 8.7, 32.0 | 13.9, 53.8 |

**Work per cell** at 128³ (jobs 32319383, 32319476; candidate reads from the measurement build, job 32319475: particles whose distance was computed, the kNN pass included), mean (p99):

| design | jittered: clips, cuts, reads | uniform | clustered | halo | lattice | wave |
|---|---|---|---|---|---|---|
| main's walk | 21.9 (25), 16.0, 76 | 48.0 (92), 17.9, 103 | 50.7 (106), 19.8, 112 | 53.7 (211), 19.4, 115 | 26.0, 6.0, 80 | 26.0, 16.1, 80 |
| kNN-first k = 32 | 21.9 (25), 16.0, 117 | 47.1 (92), 17.7, 181 | 49.2 (106), 19.1, 191 | 52.3 (211), 18.8, 192 | 26.0, 6.0, 112 | 21.8 (25), 16.1, 112 |
| kNN-first k = 48 | 21.9 (25), 16.0, 146 | 44.5 (92), 17.0, 183 | 46.1 (105), 17.7, 197 | 49.5 (211), 17.6, 198 | 26.0, 6.0, 148 | 21.8 (25), 16.1, 148 |
| k = 32, resume | 21.9 (25), 16.0, 117 | 44.3 (89), 16.6, 204 | 45.6 (100), 16.6, 214 | 49.1 (208), 16.6, 215 | 26.0, 6.0, 112 | 21.8 (25), 16.1, 112 |
| k = 32, resume, balls | 16.8 (22), 16.0, 117 | 25.8 (40), 16.6, 204 | 26.0 (40), 16.6, 214 | 25.8 (40), 16.6, 215 | 26.0, 6.0, 112 | 16.8 (22), 16.1, 112 |
| walk with balls | 16.8 (22), 16.0, 76 | 23.8 (41), 17.9, 103 | 25.7 (46), 19.8, 112 | 25.1 (46), 19.4, 115 | 26.0, 6.0, 80 | 17.5 (23), 16.1, 80 |

**Instructions** (Nsight Compute, job 32360847, warp instructions per cell, 128³): main's kernel 8802 jittered, 14020 uniform, 16751 halo; kNN pass (k = 32) 1404, 2764, 3897 (136 registers); listed clipping 8110 jittered and, with resume, 13580 uniform and 15336 halo; the walk with balls 9085, 15368, 18380 (the filter's lane loop over the vertices costs more instructions than the clips it removes); the branch's walk with the balls compiled in but off 9090, 14740, 17548, and 4 to 19 % slower than main at the same 96 registers (51.0 against 48.9 ms jittered, 110.7 against 92.8 uniform): the kernel's instruction-cache limit of the P1 section again.

**Exact start** (`start_ms`, median of 2 × 3, job 32319853), M cells/s: main 18.5, 8.9, 8.0, 10.4, 7.7, 4.4 against k = 32 with resume 18.4, 8.5, 7.4, 9.0, 7.6, 4.4 and the walk with balls 18.1, 8.2, 7.0, 9.3, 7.5, 4.3 (jittered, uniform, clustered, halo, lattice, wave).

**Mesh.** After the build and two steps at 0.1 spacings (job 32319852, 128³) `tets_checksum`, `geometry_checksum`, `faces_checksum` and `lists_checksum` equal main's on all six inputs for the branch's walk, k = 32 with resume (with and without balls) and the walk with balls, with main's bad tetrahedra and patched cells. The rings' topology (`rings_topology`, the bench's ring checksum without geometry bits) is the same in all of them. `rings_checksum` and a plain rebuild's `clip_checksum` are main's for the walk with balls (the filter changes no cut); a kNN order cuts in another order, so the clipped volumes and centroids differ in the last bits (not the mesh's, which come from the tetrahedra).

**What it says.** kNN-first is slower than the walk on every input at both sizes: 4 to 11 % on jittered and wave input in its best form (k = 32, resume), 16 to 22 % on unstructured input. Two reasons. (1) A list certifies a cell only when its last entry lies beyond 2R, as the walk's stop does; k = 32 does that for jittered, lattice and wave input, but on uniform, clustered and halo input 79 % of the cells need more (k = 48: 32 %, k = 64 at 64³: 8 %), so they are clipped twice or resumed. (2) A list clips the same particles as the walk: everything within 2R at its turn, nearest first. Its global order saves 8 to 16 % of the clips (uniform 48.0 to 44.3), and its cuts fall from 17.9 to 16.6 on unstructured input, but the clipping without any walk (45 ms jittered) is already 92 % of main's whole kernel (49 ms), so the gather the list replaces is worth at most 8 % there, less than the kNN pass costs (10 to 15 ms). A per-region choice changes nothing: the octree level does not tell which cells a list certifies. The vertex balls remove half the clips on unstructured input (48 to 24 uniform, 54 to 25 halo) and a quarter on jittered, with every cut and bit the same, but they save no time: a clip that cuts nothing is a side test and a ballot, about as dear as the filter's own test, and the cuts (16 to 20 a cell, about a face each) are the cost. The lower bound of about one cut per face is already met within 1 to 25 % (cuts against 15.9 faces), so no candidate order, list or bound can make the rebuild several times faster; that needs a cheaper cut (C, R16) or no clipping (D).

### Hard cells on clustered and halo input, and #49 (2026-11-15)

Branch `clustered-build` against main c42c230 (`ref`; the library of 55cd6b1), one A100 at 1305 MHz on `gpu1`, every comparison in one job. Script `scripts/clustered_build_a100.sh` (bench binaries in `build-jobs/bin`, blue noise the library benchmark's point file), tables by `scripts/clustered_build_tables.py`; logs in `build-jobs/` of the `voronoi-cpp-clustered` worktree on Raven. The bench's `--kernels` now also splits the rebuild and the start by kernel label (it fences every kernel).

**Where the time of a clustered or halo rebuild goes** (128³, job 32362275; counters from the measurement build, job 32365230). Per kernel: the warp kernel takes 93.0 ms on uniform, 116.6 on clustered and 119.3 on halo input; the thread-kernel pass for the cells the warp gives up takes 3.1, 6.7 and 17.2 ms (12 % of a halo rebuild). Those cells are few: 33, 68 and 261 of 2.1M (more than 64 planes or 64 vertices; none on jittered, lattice, wave and blue noise). The tail of the warp kernel is thin: the slowest 0.1 % of cells hold 0.2, 0.3 and 0.8 % of its cycles and the slowest 1 % hold 1.8, 2.2 and 4.0 %, since persistent warps take cells from a counter and a slow cell holds only its own warp. Clips per cell, mean, p99 and max: uniform 48.0, 92, 188; clustered 50.7, 106, 448; halo 53.6, 210, 2096; blue noise 19.6, 36, 75 (jittered, lattice and wave as before). The clipping of every cell is the same in every variant below, so these counts hold for all of them.

By grid level against the base (halo): level −1 is 8 % of the cells and 7 % of the cycles (clips mean 54.6, p99 267) and holds 170 of the 261 overflowing cells; the base level is 10 % of the cells, 13 % of the cycles (clips mean 91.4, p99 590) and 64 overflows; the finer levels are 82 % of the cells, 80 % of the cycles and 27 overflows. On clustered input the overflows are spread over all four levels (27, 5, 20, 16).

**Lever 1, overflowed cells straight to big thread cells (kept).** Main rebuilt a cell the warp gave up first in a thread cell of 48 planes and 64 vertices and, when that overflowed too, in one of 128 and 256. A cell that overflows the warp's 64 planes or 64 vertices overflows the thread's 48 and 64 as well, so that first pass was wasted; uncertified cells still take it. A cell that fits gives the same bits at either capacity. Thread pass 17.2 → 10.2 ms on halo, 6.7 → 5.3 clustered, unchanged elsewhere (job 32366435).

Rebuild (`construct_ms`, median of 5, M cells/s; 128³ two rounds, job 32366435; 256³ one round, job 32366438):

| | jittered | uniform | clustered | halo | lattice | wave | blue noise |
|---|---|---|---|---|---|---|---|
| main, 128³ | 42.5 | 21.9 | 17.1 | 15.5 | 55.1 | 40.8 | 37.0 |
| branch, 128³ | 42.5 (−0.1 %) | 22.2 (+1.2 %) | 17.3 (+1.2 %) | 16.3 (+5.3 %) | 55.0 (−0.1 %) | 40.8 (−0.1 %) | 36.9 (−0.1 %) |
| main, 256³ | 42.8 | 22.4 | 18.6 | 17.9 | 57.7 | 41.0 | 37.0 |
| branch, 256³ | 42.8 (0 %) | 22.5 (+0.3 %) | 18.6 (+0.3 %) | 18.0 (+0.7 %) | 57.7 (0 %) | 41.1 (0 %) | 37.1 (+0.2 %) |

At 256³ the pass is the time of its slowest cell on one thread, much as at 128³, against an eight times longer warp kernel. Exact start (`start_ms`, median of 3, M cells/s, one round; jobs 32366435, 32366439): 128³ main 18.6, 8.9, 7.8, 10.3, 7.7, 4.4, 18.8 against 18.5, 9.0, 8.0, 10.7, 7.5, 4.2, 18.7 (jittered, uniform, clustered, halo, lattice, wave, blue noise); 256³ main 12.7, 9.0, 8.7, 8.7, 7.8, 4.4, 13.8 against 12.9, 8.9, 9.1, 8.3, 8.0, 4.4, 13.7, and again over three rounds (job 32367236) uniform 9.0 → 9.5, clustered 9.1 → 9.0, halo 8.3 → 8.8. The start varies by 5 to 8 % from run to run; lattice and wave, where no cell overflows, run the same kernels as main.

**Lever 1, a queue of hard cells first (dropped).** The cells below the base level (halo 168k, clustered 379k) built first by their own launch of the warp kernel, the rest by a second launch on another stream that fills the SMs they free, and the big thread pass for the hard cells' overflows on a third stream while the rest is built (`VORONOI_HARD=2` in commit f6535fc, a level more with `=3`). Halo 136.6 ms against 128.6 with lever 1 alone, clustered 124.4 against 121.5 (job 32365816). The timeline (Nsight Systems, job 32366007) shows why: the second launch is a persistent kernel that keeps every SM slot until it ends, so the overlapped thread pass starts only after it (at 126 ms of 137); and a third of the halo overflows (91 of 261, 5.5 ms of thread pass) lie on finer levels the queue does not hold. A hard-first order cannot shorten the drain either, which is under 1 % of the cycles. The vertex balls (P1) and the R8 box were meant to pay in this queue; with the thread kernel for the hard cells they would only skip clips that cut nothing in a pass whose time is that of one cell, and a warp kernel of larger capacity would change bits (next paragraph).

**Lever 2, balancing by expected work (not built).** The warp kernel's persistent warps already balance the load (above). A block or several warps per huge cell needs a warp kernel of larger capacity in place of the thread kernel for the cells that overflow, and the two kernels do not give the same bits: building every cell with the thread kernel (`VORONOI_HARD=9`, commit cdf04d4, job 32365528) gives other `clip_checksum` and `clip_topology` on jittered, uniform and halo input than the warp kernel. So the overflowing cells of uniform, clustered and halo input would change their last bits against main.

**Lever 3, overflowing cells on the host (dropped).** The big rounded cells built on the host (Kokkos Serial, the arrays copied there and back; `VORONOI_HARD=8`, commit f6535fc) while nothing else runs: 276.7 ms against 122.9 on clustered and 283.4 against 135.2 on halo (job 32365816), and the host's rounding differs: `clip_checksum` changes on clustered and halo input, with the same topology. Run beside the warp kernel it would need the copies to overlap too, and it would still change bits. Exact huge cells were already built on the host; they now use every core (#49, below).

**Adaptive levels.** A different level for a slot changes which particles share a batch of the walk and so the order of the cuts and the last bits of the cell; with the mesh held bit for bit to main, the level choice stays.

**#49.** Seeds that ran out of time, with the passes timed (`VORONOI_TRACE` in commit f6535fc; jobs 32365817 to 32365819, Nsight Systems jobs 32362474 and 32362475). Seed 3000000000009 (156,273 particles, a Fibonacci sphere with a diamond lattice and clumps): per update the warp kernel 0.12 s, 42,543 rounded thread cells 0.7 s, 34,355 big ones 5.4 s, then the exact patch: 35,488 big exact cells 15.5 s and 5525 huge ones on the host, 408 s on one core, since the CUDA build's host space is Kokkos Serial. Seed 4000000000002 (1,551,620 particles, no background: a cusp, a spiral, polytopes): 97 % of the cells overflow the warp; the rounded thread pass took 40.6 s and the big one 412 s for 1,485,690 cells (3600 cells/s), then 1,416,857 exact cells 105 s. So two causes: (a) inputs where most cells have more than 64 faces send nearly every cell to the thread kernel, which builds big cells at a few thousand per second; (b) cells past 128 planes or 256 vertices go to the host one at a time. (b) is fixed: the host builds huge cells one at a time on every core the process may use, each thread with its own Serial instance, and gives the same bits as before. Seed 3000000000009 now ends in 110 s (job 32366437, 18 cores), with the right refusal (5 cells need more than 16000 planes); main was still in its first `update()` after 830 s. (a) is not fixed: lever 1 drops the 40.6 s of doomed small cells (16.8 s left for the uncertified ones), but the 412 s of big thread cells stay. It needs a warp kernel of larger capacity for such cells, which changes the warp kernel's output code and, on inputs with few overflowing cells, bits (above).

**Mesh.** After the build and two steps at 0.1 spacings, `tets_checksum`, `rings_checksum`, `rings_topology`, the build's `cells_checksum`, `geometry_checksum`, `faces_checksum` and `lists_checksum`, and a plain rebuild's `clip_checksum` and `clip_topology` equal main's on all seven inputs at 128³ (job 32366436) and 256³ (jobs 32366440 to 32366442; wave at 256³ the start's and the rebuild's checksums, as its step runs out of memory on main too), with main's bad tetrahedra and patched cells. The step's `cells_checksum` differs between runs of main as well. Tests: A100 ctest 22 of 22 (job 32367231, with the fuzz smoke); CI's host steps with GCC 15 and the OpenMP Kokkos, sanitizers off and on, 22 of 22 each (jobs 32367232, 32367233; lint reports one clang-analyzer warning in `mesh_fuzz_test.cpp` that main has too). A100 fuzz, profile huge (1e6 to 4.5e6 particles, with its Gaussian, power-law and void families; jobs 32367234, 32367235): 18 cases, 8 exact, 7 refused correctly, 3 out of time (280 s), no WRONG or CRASH. The three (4000000000098, 105, 107: spirals, lines and coplanar sets, cause (a) of #49) run out of time alone with main as well (260 s; jobs 32367945, 32367946).

### Step kernels at today's bytes (2026-11-12)

Branch `step-kernels` against main 5d23b26 (with the re-sort, layout off, and the #48 fix), one A100 at 1305 MHz application clock, double, jittered input, median of steps 2 to 10 over two interleaved runs of each build (scripts `scripts/ab_step_a100.sh`, `scripts/ab_step_tables.py`; logs archived on Raven in `voronoi-cpp-joblogs/voronoi-cpp-step/build-cuda/jobs/`). The target was the bandwidth ceiling of [measure_2026-11-12.md](measure_2026-11-12.md) at today's bytes: 434 M cells/s for the step at 0.03 spacings, 320 with `update_cells()`, 259 at 0.1. Nothing stored changed.

| M cells/s | main | step-kernels | gain | ceiling (share reached) |
|---|---|---|---|---|
| step, 0.03 / 0.1 / 0.3 spacings, 128³ (job 32231679) | 194.0 / 113.9 / 56.4 | 204.5 / 126.4 / 63.6 | +5.4 / +11.0 / +12.8 % | 434 / 259 / – (47 / 49 %) |
| step + `update_cells()`, same | 144.0 / 90.4 / 49.9 | 149.6 / 98.2 / 55.5 | +3.9 / +8.6 / +11.2 % | 320 at 0.03 (47 %) |
| step, 0.03 / 0.1 / 0.3, 256³ (job 32231432) | 264.2 / 177.6 / 93.6 | 280.7 / 194.1 / 107.6 | +6.3 / +9.3 / +15.0 % | |
| step + `update_cells()`, 256³ | 187.3 / 133.2 / 77.6 | 195.5 / 142.2 / 86.9 | +4.4 / +6.8 / +12.0 % | |

On main c505cc6 (before #48) the same changes gave +5.3 / +10.6 / +12.9 % (job 32230977) and +6.8 / +10.4 / +12.8 % at 256³ (job 32230698); on a `gpudev` node that ran slower for both builds +5.8 / +9.8 / +11.7 % (job 32231522). Bits: `geometry_checksum` (cell volumes and centroids) and a numbering-free face checksum (owner, neighbour, area and centroid bits, faces sorted by pair) are identical to main at every step of 0.03, 0.1 and 0.3 spacings, jittered and clustered, 128³ (job 32231523 on 5d23b26, and 32230464, 32230931 on c505cc6; 30 steps each); the start's `cells_checksum` too. The per-step `cells_checksum` hashes face numbers, which follow the repair's slot order (atomics): it differs between two runs of main itself (job 32225953), so it cannot compare builds. Peak device memory is unchanged: 1998.29 B/cell over steps at 0.03 spacings and 2028.37 at 0.3, both builds (job 32231678). A100 `ctest`: 22 of 22 pass (job 32231298).

Each lever on its own, 128³, against the build before it in the same job (step rate at 0.03 / 0.1 spacings):

| lever | job | step | kernel |
|---|---|---|---|
| One wait per flip round: when the work and stuck lists hold 4 failing triangles per listed tetrahedron, `plan` and `apply` run over that bound and their threads past the device count return; the count is read with the apply step's counters. The screen's count read is reused for the repair's first round | 32225960 (Nsight Systems), 32225565 | `update()` wall 11.15 → 10.96 ms (+1.7 %) at 0.03, 19.43 → 19.33 ms (+0.5 %) at 0.1; synchronisations 26.6 → 20.7 and 31.7 → 25.6 | |
| Launch bounds: `faces` at most 80 registers (128 threads, 6 blocks per SM), `screen` at most 64 (128, 8) | 32225763 | +3.8 / +3.8 % | `faces` 2.95 → 2.74 ms, `screen` 2.65 → 2.50 ms at 0.03 |
| Warp-aggregated counters (`tet_mesh::claim`): one atomic per group of threads on a shared counter (dirty lists, work list, stuck list, freed slots, created tetrahedra, ring pool) | 32225953 | −0.1 / +2.6 % | `apply` 3.24 → 2.83 ms at 0.1 |
| `dirty_faces` with a thread per edge of each touched tetrahedron (was a thread per tetrahedron, six walks each) | 32225953 | +1.9 / +4.4 % | `dirty_faces` 0.68 → 0.42 ms at 0.03, 2.38 → 1.22 ms at 0.1 |

Tried and left out:

| lever | job | result |
|---|---|---|
| `find`, `plan`, `apply` capped at 64 or 80 registers | 32225408 | slower: they spill; `find` 0.46 → 0.72 to 0.76 ms at 0.03, the step 3 to 7 % slower. At 128 registers no change |
| `faces` at 64 registers | 32225408 | `faces` 4 % slower (spills) |
| Winners of the vote compacted (a kernel lists them, 2→3 and 3→2 flips apart, in the work list's room) before `apply` | 32225565, 32230610 | +1.0 / +0.9 % alone, but +0.3 / −2.5 % on top of the levers above: `apply` 1.08 → 0.89 ms, the extra kernel 0.27 ms |
| The dirty-face count kept on the device: `append_rings` over 6 per touched tetrahedron, one wait instead of two in `update_face_rings` | 32226509 (Nsight Systems) | `update()` wall 0.8 % (0.03) and 1.8 % (0.1) slower: the wider launch costs more than the wait |

What is left between the step and its ceiling (0.03 spacings, 128³). Nsight Systems (job 32231002, a `gpudev` node slower for both builds): per `update()` 44.4 → 48.0 launches, 26.6 → 20.7 synchronisations, kernel time 11.21 → 10.72 ms, host gap 1.12 → 0.97 ms (at 0.1: 19.97 → 18.29 ms of kernels, gap 1.30 → 1.12 ms). Per kernel, the time now (Kokkos Tools, job 32230610) against its bytes at 80 % of peak ([measure_2026-11-12.md](measure_2026-11-12.md)): `screen` 2.17 against 0.89 ms, `faces` 2.42 against 1.72, `apply` 1.08 against 0.24, `plan` 0.91 against 0.13, `face_numbers` 1.16 against 0.71, `append_rings` 0.71 against 0.35, `find` 0.48 against 0.12, `dirty_faces` 0.40 against 0.36: about 4.8 ms of kernel time and 1 ms of host gaps over the ceiling's 4.8 ms. Why it stays: `screen` and `faces` gather a 32-byte row per corner or ring entry from scattered tetrahedra and are bound by L1/L2 transactions and latency, not by registers any more; the repair's `plan`, `find` and `apply` are one thread per flip with long, divergent, dependent chains (the exact predicates' 128- and 256-bit fallbacks keep `plan` and `find` at 152 to 168 registers, and capping them spills); `face_numbers` is a scan whose result (the face count) the host must read before the next launches, as are the counts of each flip round. The next steps are structural: a warp per flip or per ring walk (K1), fewer bytes per flip (K2), face numbers kept for untouched tetrahedra (E, needs `number_faces`), and the count read-backs replaced by a device-side queue (R1).

### Fewer bytes per step: incremental face numbers (E) and the ring re-walk (K2) (2026-11-12)

Branch `step-bytes` on main 9e9ec03: `cbf2da6` the incremental face numbering (E, [#53](https://github.com/Ulli1991/voronoi-cpp/issues/53)), `d3cc1e2` memory to pay for it, then the ring re-walk (K2) with these docs. Builds: main (base), E (E and memory), K2 (K2 alone on main, not committed: `voronoi-cpp-joblogs/voronoi-cpp-bytes/k2_alone.patch` on Raven), E+K2 (the branch). One A100 at 1305 MHz application clock (the SM clock each job logs is in the table), double, jittered input, median of steps 2 to 10 over two interleaved runs of each build, all builds in one job (`scripts/ab_step_a100.sh`, `scripts/ab_step_tables.py`; per-kernel bytes `scripts/bytes_a100.sh`, `scripts/bytes_tables.py`; logs archived on Raven in `voronoi-cpp-joblogs/voronoi-cpp-bytes/build-cuda/jobs/`). The levers' shares from [measure_2026-11-12.md](measure_2026-11-12.md) were upper bounds: E −380 B/cell (1.15×), K2 1.28× at 0.03 and 1.85× at 0.1.

**What changed.** E: a tetrahedron's faces are `ids(start(t) + rank)` in a pool of numbers; after a repair with at most count/16 created tetrahedra only the touched ones release and take numbers (in place when they have no more faces than before; the other freed numbers go to new faces in slot order by a scan, and faces above the new count move down), the rest keep theirs; with more created tetrahedra every face is numbered in slot order as before, which also restores the order. K2: a dirty face's ring is walked once into a buffer and written in canonical order from it (was: walked twice), into its old room when it is no longer than before (the ring pool compacts about half as often).

| M cells/s, step (with `update_cells()`) | base | E | K2 | E+K2 | E+K2 against base |
|---|---|---|---|---|---|
| 128³, 0.03 spacings (job 32269165) | 205.5 (150.3) | 221.8 (160.7) | 205.6 (150.3) | 222.7 (161.1) | +8.4 % (+7.2 %) |
| 128³, 0.1 | 126.2 (98.0) | 126.1 (97.1) | 129.7 (100.1) | 129.7 (99.3) | +2.8 % (+1.3 %) |
| 128³, 0.3 | 63.6 (55.5) | 62.6 (54.6) | 66.2 (57.4) | 65.9 (57.2) | +3.5 % (+3.1 %) |
| 256³, 0.03 spacings (job 32237516) | 280.7 (195.4) | 313.6 (212.9) | 284.0 (197.0) | 318.2 (214.9) | +13.4 % (+10.0 %) |
| 256³, 0.1 | 194.8 (142.6) | 203.5 (148.4) | 200.1 (145.3) | 208.8 (151.2) | +7.2 % (+6.0 %) |
| 256³, 0.3 | 108.0 (87.0) | 104.2 (81.7) | 113.6 (90.6) | 109.5 (85.6) | +1.5 % (−1.6 %) |
| 500 steps at 0.1, Hilbert re-sort every 50, 128³ (job 32238599): step, step + cells, all with re-sorts | 148.6, 113.3, 111.9 | 150.9, 112.8, 111.1 | | 154.3, 114.8, 112.3 | +3.8 %, +1.3 %, +0.4 % |

E gains most where few tetrahedra change (0.03 spacings, and 256³, where the flow shears half as much per step). At 0.3 it costs at 256³ and is neutral at 128³: once the flips fall below count/16, and the faces that take new numbers lose the slot order (`faces` and `cell_sums` read them scattered), and every face lookup goes through `ids` (`dirty_faces` +15 % at 0.1). K2 gains 0 to 5.2 % on its own (most at large spacings), mostly from `append_rings` (−23 % time at 0.1) and fewer ring compactions.

DRAM bytes per cell of step 3, Nsight Compute, 128³, caches flushed (job 32237298; the E and E+K2 binaries then had the same kernels as the final ones):

| B/cell, `update()` (`update_cells()`) | base | E | K2 | E+K2 |
|---|---|---|---|---|
| 0.03 spacings | 2717 (964) | 2398 (916) | 2707 (964) | 2392 (913) |
| 0.1 spacings | 4292 (1083) | 4180 (1133) | 4239 (1083) | 4133 (1132) |
| `face_numbers` / its replacement at 0.03 | 422 | 63 (`face_touched` 14, `face_offsets` 11, `face_release` 20, `face_assign` 13, `face_numbered` 5) | 421 | 63 |
| repair and ring re-walk (`find`, `plan`, `apply`, `dirty_faces`, `append_rings`), per flip at 0.03 / 0.1 | 20.4 / 21.1 kB | 21.2 / 22.1 kB | 20.3 / 20.8 kB | 21.3 / 21.8 kB |

E removes 359 B/cell of numbering at 0.03 and 263 at 0.1, but gives back 40 (0.03) and 151 (0.1) in `faces`, `dirty_faces` and `cell_sums`: net −319 and −112 B/cell. K2 cuts the re-walk's bytes per flip by 2 % only: one walk instead of two halves the rows read in `append_rings`, but those rows mostly hit L2 already (the second walk re-reads what the first just read); the DRAM bytes per flip stay about 21 kB against a target of 3 kB. What is left there is the repair's rounds (`find`, `plan`, `apply`: 9.7 kB per flip at 0.1, unchanged) and the partial walks of `dirty_faces` from every touched tetrahedron.

Bits (jobs 32238014, 32238070, 32238146 jittered, 32238147, 32238148, 32238322 clustered; 30 steps each at 0.03, 0.1 and 0.3 spacings, 128³): `geometry_checksum`, `faces_checksum` and the new `lists_checksum` (each cell's list as other particles and owner bits) identical to main at every step for E and E+K2; `cells_checksum` hashes face numbers and differs, as it did between two runs of main. With faces by owner, the owner bit and a re-sort every 2 steps (job 32238180, 262,144 cells, 6 steps) the same checksums equal main's; `--verify` against a fresh rebuild passes at 0.3 spacings. A100 `ctest`: 22 of 22 pass (job 32238334). CI's host steps (gcc 15, sanitizers off and on, clang-tidy, ctest; host Kokkos 5.0.99 with OpenMP and 2 threads, where CI builds a Serial Kokkos 5.0.2): 22 of 22 pass with sanitizers off and on (jobs 32238509, 32238627); clang-tidy 21 stops on `mesh_fuzz_test.cpp`, unchanged here and on main ([#56](https://github.com/Ulli1991/voronoi-cpp/issues/56)). In one clustered 0.3 run E+K2 left 6 triangles stuck at step 2, which the host finish repaired (the repair's flip order follows atomics; clustered 0.3 sticks in some steps on main too, [measure_2026-11-12.md](measure_2026-11-12.md)); its checksums equal main's.

Memory: peak over 10 steps, 128³ (job 32238100): 1998.29 → 1979.05 B/cell at 0.03 spacings and 2028.38 → 2010.04 at 0.3 for E and E+K2; K2 alone is unchanged. E adds `face_rings_ids` (41 B/cell, a quarter spare) and `face_rings_numbered` (7.8); the repair's free list sized to its use (31 → under 2 B/cell) and the cell list freed after a build (37 B/cell, absent at the step's peak) pay for it.

luna-cpp (branch `resort` a82092c, the first that builds against main's API without gradient weights), moving Sedov 128³, 50 steps, two runs of each build (jobs 32238510, 32238511): the final `Q`, `x` and `volume` hashes equal those on main's voronoi-cpp, with the default layout and with faces by owner, the owner bit and a re-sort every 10 steps. luna-cpp reads `cell_face` lists, `area`, `face_centroid`, `owner` and `neighbour` as before; only the face numbers differ (see below). Mesh throughput there 274 → 311 M cells/s (+13.6 %), with faces by owner 198.8 → 221.7 (+11.5 %); the whole hydro step 71.4 → 73.7 M cells/s (+3.2 %) and 70.5 → 73.1 (+3.7 %).

**Face numbering, for callers.** As before, faces are numbered 0 .. faces() − 1, `owner(f) < neighbour(f)`, each cell's list is sorted by the other particle, and with `Layout::owner_faces` the numbers go by owner. What changed: without the owner layout a face's number no longer follows the slot order of the tetrahedra after every update; it stays the same from update to update while the tetrahedron representing its edge is not touched by a flip, and a new face takes a freed number. Numbers were never stable across updates before; a caller that keyed data by face number across an update (luna-cpp does not) now sees most numbers kept, not all.


**Left to the 5× budget.** The step now moves 2392 B/cell at 0.03 spacings and 4133 at 0.1 (step 3, 128³; main 69de1d3 moved 2866 and 4795, main 9e9ec03 2717 and 4292). The 5× budgets are 1269 and 2182 B/cell: 1123 and 1951 B/cell still to go, with every kernel at 80 % of peak besides. By kernel at 0.03: `faces` 1005 and `cell_sums` 737 (R14 measured below: −753 B/cell but slower; float centres, R6), `screen` 528 (K4), the repair and ring re-walk 617 (the repair's rounds and the `dirty_faces` walks, K2), the numbering now 63.

### V1 and R14 (branch `step-v1-r14`, job 32370682)

One A100 40 GB at 1305 MHz, jittered, 10 steps twice per build, steps 2–10 averaged; M cells/s of `update()` and, in brackets, of `update()` + `update_cells()`. main is c42c230; head is the branch with V1 dropped and R14 opt-in; V1 is the quiet screen alone (faces stored); R14 is head with `Layout::faces_on_read`.

| | main | head | V1 | R14 |
|---|---|---|---|---|
| 128³, 0.03 | 225.8 (162.8) | 224.4 (162.1) | 72.8 (64.7) | 280.5 (127.7) |
| 128³, 0.1 | 128.5 (98.8) | 128.0 (98.4) | 54.7 (48.5) | 146.4 (86.1) |
| 128³, 0.3 | 66.8 (57.8) | 66.6 (57.6) | 36.5 (33.7) | 71.1 (53.3) |
| 256³, 0.03 | 325.3 (218.3) | 324.0 (217.8) | 87.1 (77.0) | |
| 256³, 0.1 | 212.7 (153.2) | 211.2 (152.5) | 72.1 (63.7) | |
| 256³, 0.3 | 108.4 (85.3) | 108.0 (85.0) | 50.7 (45.0) | |

DRAM bytes of step 3 (Nsight Compute, 128³, `update()` + `update_cells()`): head 3304 B/cell at 0.03 spacings and 5263 at 0.1, as main; R14 2552 and 4464 (no `faces` kernel, 1005 B/cell, but `cell_sums` 834 B/cell in 7.4 ms against 736 in 2.2, and `tet_volumes` 170); V1 13777 and 16370, of which `screen` 10984 and 11621 (main 528 and 553). With the certificates off, V1's motion measure alone leaves `screen` at 1733 B/cell and 4.2 ms.

Bits (5 steps at 0.03, 0.1, 0.3, jittered and clustered, 128³): `geometry_checksum`, `faces_checksum` (face areas and centroids) and `lists_checksum` equal to main at all 30 checks for head, V1, R14 and V1's motion measure. The centroid of a face computed where it is read rounds its last product on its own (c39b548), so the bits are those of the stored rows; the stored rows did not change. `voronoi_mesh_test` checks `faces_on_read` against the stored faces and cell sums bit for bit over four updates.

luna-cpp's mesh step on this branch before these changes, 23.8 ms against 6.75 on its pin 6ead543 (128³, `grad32`, 1410 MHz), was V1: on by default, its screen 3–4× slower. R14 was then off in practice, since luna-cpp reads `area()`, which stores the faces after the update.

### Float face centroids (`Layout::float_centroids`, job 32373869)

A stored face is owner (4 B), neighbour (4), area (24) and centroid (24), one array per field: 56 B. The library's index arrays are already 32-bit, with the owner bit in the cell lists (`Layout::owner_bit`), and the rows are already split by field, so the layout left to shrink is the centroid: float, relative to the owner, 12 B, 44 B a face. Areas stay double: cell closure, volumes and the topology keep their bits. One A100 40 GB at 1305 MHz, jittered, M cells/s of `update()` + `update_cells()` (head is the branch's default layout, same build):

| | main | head | float centroids |
|---|---|---|---|
| 128³, 0.03 | 161.9 | 162.8 | 165.8 |
| 128³, 0.1 | 98.8 | 98.6 | 99.7 |
| 128³, 0.3 | 56.4 | 58.0 | 58.4 |
| 256³, 0.03 | 218.3 | 220.2 | 224.3 |
| 256³, 0.1 | 152.9 | 153.8 | 156.2 |
| 256³, 0.3 | 85.3 | 84.9 | 86.0 |

DRAM of step 3 (128³): 3300 → 3096 B/cell at 0.03 spacings and 5247 → 5036 at 0.1; `faces` writes 95 B/cell less, `cell_sums` reads 110 to 113 less. Per face, 415 → 389 B at 0.03. Peak memory of the steps 1979 → 1881 B/cell at 0.03 and 2010 → 1912 at 0.3. The build's geometry time is unchanged (12.9 ms at 128³).

Geometry: `voronoi_mesh_test` (4096 cells) finds cell centroids within 7e-9 spacings of double and of clipping (double and clipping agree to 6e-16), volumes and areas bit for bit. At 128³ after 3 steps (`--dump-cells`), at 0.03 and 0.1 spacings: the same neighbours for every cell, volumes bit for bit, centroids within 8.1e-9 spacings (mean 1.8e-9). `lists_checksum` equals main's at all 30 checks; `geometry_checksum` and `faces_checksum` differ (centroids).

### Face visitor and chunked passes (`Mesh::for_each_face`, `voronoi_bench chunks`, job 32375192)

`for_each_face(visit)` runs one kernel over the faces and calls `visit(f, owner, neighbour, area, centroid)` with the geometry in registers: read from the stored rows, or, with `Layout::faces_on_read` and no rows stored, computed once from the face's ring (unlike R14's `update_cells()`, which computes each face twice). The values are the stored rows' bits (`voronoi_mesh_test`, four updates). The generic functor here is the closure check: each face adds its area vector to its owner and subtracts it from its neighbour (six atomics); the largest |sum| was the same, 1.6e-14 to 3.8e-14 spacings², either way. One A100 40 GB at 1305 MHz, jittered, 10 steps twice, M cells/s of `update()` + the visitor (with `update_cells()` too in brackets):

| | stored rows | visitor, no rows |
|---|---|---|
| 128³, 0.03 | 182.4 (139.9) | 201.4 (108.2) |
| 128³, 0.1 | 113.5 (89.9) | 120.1 (76.3) |
| 256³, 0.03 | 244.2 (180.2) | 278.7 (131.6) |
| 256³, 0.1 | 174.0 (133.0) | 192.4 (103.8) |

DRAM of step 3 (128³), `update()` + visitor: 2697 → 2199 B/cell at 0.03 spacings, 4445 → 3950 at 0.1 (the visitor reads 663 and 764 B/cell from the rings against 311 and 324 from the rows, but `faces`' 1005 and 1106 go). Peak memory of the steps 1979 → 1585 B/cell at 128³ and 256³. A caller that also needs `update_cells()` loses: its sums still compute each face from the ring, twice (9.0 against 3.5 ms at 128³, 0.03).

Chunked passes, a two-pass functor on a static 128³ and 256³ mesh sorted along the Hilbert curve, faces by owner: pass 1 per cell sums |A| over its list in list order, pass 2 per face uses both cells' sums. Whole mesh: two launches. Chunked: cells [c C, (c + 1) C), pass 1 on the chunk and its first halo layer, then pass 2 on the faces the chunk owns, two launches per chunk. Outputs equal bit for bit in every case (the sums run in list order whatever the chunk). Halo cells per cell, one layer (two layers): C = 16384: 0.26 (0.57); 32768: 0.20 (0.42); 65536: 0.16 (0.34); 131072: 0.13 (0.27).

| C | launches 128³ / 256³ | ms 128³ / 256³ | DRAM B/cell 128³ / 256³ |
|---|---|---|---|
| whole | 2 / 2 | 1.73 / 11.5 | 943 / 831 |
| 16384 | 256 / 2048 | 4.51 / 36.0 | 643 / 643 |
| 32768 | 128 / 1024 | 2.66 / 21.3 | 612 / 612 |
| 65536 | 64 / 512 | 1.77 / 14.1 | 598 / 598 |
| 131072 | 32 / 256 | 2.01 / 16.1 | 974 / 977 |

Chunks of 32768 to 65536 cells keep pass 1's output and the rows in L2 for pass 2: 28 to 37 % fewer DRAM bytes. They are still 2 % (128³) to 23 % (256³) slower: the summed kernel time is above the whole mesh's too (1.79 against 1.60 ms at 128³, 14.3 against 11.5 at 256³), so a CUDA graph would remove the launch gaps but not the small grids' tails; 131072 cells overflow the 40 MB L2.

### The exact start: assembly from the ring blocks and an incremental patch round (2026-11-15)

Branch `start-fast` against main c42c230, both in the same `gpu1` job (one A100 at the 1305 MHz application clock; `scripts/start_fast_a100.sh proof`, tables by `scripts/start_fast_tables.py`). The start is the bench's `start_ms`: the warp clipping with rings and `tet_start::build`, median of 3 after a warm-up. "Exact" is `tet_start::build` alone: assembly, check, exact re-clips and further rounds.

What changed, all in `tet_mesh::assemble` and `tet_start.hpp`:

- **Lookups read the clipping's ring blocks** in place. Main copied every ring into a face-ordered array (`face_particles`) and looked a tetrahedron up through three face arrays and that copy; now a lookup reads one 32-byte record of the owner cell, then the cell's block. No `face_particles`, `face_lengths`, `first_tet` scan or `ring_start` scan.
- **One thread per ring slot** for emitting and naming (main: one per face, 9 threads busy per warp instruction), **one per tetrahedron** for the adjacency (main: one per face, 4.9), **one per face** for the face arrays (coalesced writes; a thread per cell wrote six arrays at 74 % of DRAM peak but 8× the bytes needed). Ring wraps without integer division.
- **The check by the double filters** (91 registers, 168 before); tetrahedra a filter cannot decide go to a second kernel with the exact predicates, so the outcome is the exact one.
- **#59: the round after a partial re-clip is incremental.** Cells whose ring block names a re-clipped particle look up again; every other lookup and adjacency is the last round's, shifted by the change in its cell's first tetrahedron. The check tests only the tetrahedra looked up again and their neighbours. The arrays are those of a full round, entry for entry (`start_arrays_checksum`).
- **Storage kept between rounds.** Freeing gigabytes and allocating again right after made single `cudaMalloc` calls take 76 to 103 ms at 256³ (Nsight Systems); the assembly and the check now keep their arrays, with a second set of tetrahedra and slots for the incremental round.

**Rates.** `start_ms`, median of 3 unfenced runs, job 32373387 (ravg1082, 1305 MHz); M cells/s and ns per cell, main → start-fast:

| input | 128³ M cells/s | 128³ ns/cell | 256³ M cells/s | 256³ ns/cell | rounds, bad tetrahedra, re-clipped cells at 256³ |
|---|---|---|---|---|---|
| jittered | 18.6 → 25.3 | 53.8 → 39.5 | 12.7 → 22.2 | 78.5 → 45.1 | 2, 44, 830 |
| blue noise | 18.6 → 23.1 | 53.7 → 43.3 | 13.6 → 20.6 | 73.3 → 48.5 | 2, 10, 127 |
| uniform | 8.6 → 12.3 | 116.5 → 81.3 | 9.4 → 12.6 | 106.1 → 79.3 | 2, 18, 277 |
| clustered | 8.0 → 10.3 | 125.0 → 97.0 | 8.9 → 12.5 | 112.5 → 79.9 | 2, 10, 203 |
| halo | 9.8 → 11.9 | 102.0 → 83.9 | 8.6 → 12.1 | 116.5 → 82.4 | 2, 24, 393 |
| lattice | 7.5 → 8.0 | 133.4 → 124.5 | 7.8 → 8.4 | 128.0 → 119.3 | 2, all, all |
| wave | 4.4 → 5.0 | 229.7 → 198.3 | 4.3 → 5.1 | 231.0 → 196.4 | 2, 481311, all |

At 128³ jittered, uniform and clustered need 1, 2 and 2 rounds (as on main). Lattice and wave re-clip every cell exactly in the second round (more than n/8 marked), which is unchanged and is most of their time.

**Kernels by group**, ms per start, fenced per kernel (`--kernels`, same job), main → start-fast. Clipping: the warp kernel with rings; re-clip: the exact re-clips; assembly: `assemble` (main: face_start, face_lengths, ring_start, face_particles, first_tet, emit, adjacency; branch: cell_counts, cell_starts, faces_of_cells, emit, adjacency); check: check, its exact pass, the incremental list, unused, the carried volume.

| input, 128³ | clipping | re-clip | assembly | check | represent |
|---|---|---|---|---|---|
| jittered | 56.6 → 56.7 | 0.1 → 0.1 | 41.7 → 13.9 | 5.1 → 3.3 | 3.3 → 3.3 |
| blue noise | 68.3 → 68.3 | 0.1 → 0.1 | 31.1 → 11.4 | 4.5 → 2.9 | 2.5 → 2.4 |
| uniform | 114.8 → 114.8 | 15.8 → 15.8 | 80.2 → 24.4 | 10.1 → 4.5 | 3.2 → 3.2 |
| clustered | 131.3 → 132.1 | 23.9 → 23.7 | 81.8 → 24.3 | 10.0 → 4.5 | 3.2 → 3.2 |
| halo | 133.8 → 133.9 | 16.6 → 16.5 | 39.6 → 13.6 | 5.1 → 3.2 | 3.2 → 3.2 |
| lattice | 39.7 → 39.7 | 182.0 → 182.0 | 33.7 → 15.2 | 8.2 → 10.4 | 2.3 → 2.3 |
| wave | 58.6 → 58.6 | 313.4 → 313.0 | 86.1 → 28.0 | 10.3 → 6.6 | 3.3 → 3.4 |

| input, 256³ | clipping | re-clip | assembly | check | represent |
|---|---|---|---|---|---|
| jittered | 453 → 453 | 4 → 4 | 681 → 197 | 79 → 35 | 26 → 26 |
| blue noise | 546 → 546 | 6 → 5 | 529 → 164 | 69 → 31 | 19 → 19 |
| uniform | 918 → 918 | 17 → 17 | 687 → 192 | 77 → 34 | 25 → 25 |
| clustered | 1008 → 1015 | 23 → 23 | 653 → 192 | 77 → 34 | 25 → 25 |
| halo | 1009 → 1010 | 69 → 69 | 684 → 192 | 77 → 35 | 25 → 25 |
| lattice | 303 → 304 | 1438 → 1437 | 281 → 118 | 63 → 80 | 18 → 18 |
| wave | 468 → 468 | 2494 → 2496 | 731 → 222 | 80 → 51 | 27 → 27 |

256³ jittered by kernel, both rounds: main emit 265.6, adjacency 218.3, face_particles 116.4, check 78.3, first_tet 38.2, face_lengths 33.2, represent 25.8, ring_start 7.7 ms; branch emit 92.7, faces_of_cells 41.0, adjacency 34.6, represent 25.9, cell_counts 25.5, check 25.4, near_redone 7.5, cell_starts 3.7 ms. On lattice the check is slower (63 → 80 ms at 256³): every tetrahedron has a cospherical tie, so all go through the exact pass after the filters. One 128³ jittered start under Nsight Compute (job 32370306, before the incremental round and the per-face face arrays): main's assembly kernels ran at 4.9 to 9.8 of 32 lanes per instruction and 13 to 44 % of DRAM peak; the branch's emit 14.2 lanes, 52 % issue active, 668 + 304 B/cell; the adjacency 14.7 lanes, 874 + 158 B/cell at 48 % of DRAM peak; the filtered check 91 registers (main 168, 12 % occupancy).

**Host side.** One 256³ jittered `tet_start::build` (the bench's last start, Nsight Systems): main 840 ms wall, 791 of kernels; the branch at v7 366 ms wall, 240 of kernels, the rest `cudaMalloc` and `cudaFree` (jobs 32371594, 32372190). Syncs and launches per start: 92 and 32 on main, 72 to 74 and 29 on the branch.

**Memory.** Peak device memory through the first `Mesh::update` (`--memory`, job 32373385): 1955.2 B/cell at 256³ jittered and 1719.5 blue noise, both main's to 0.01 B/cell; the peak is in `prepare` and the geometry, not the start. The bench's own start block, which keeps the previous run's mesh alive while the next start builds, peaks at 1973 against 1696 B/cell (jittered 256³). A per-slot face map instead of bisection on `ring_start` makes emit 1.6 ms faster at 128³ (5.0 against 6.6 ms) for 167 B/cell more at the start's peak; not taken.

**Bits.** `tets_checksum`, `rings_checksum`, `rings_topology`, `assembly_checksum` (one assembly of the first clipping, entry for entry, also where the check rejects it), `start_arrays_checksum` (the final start's vertex, adjacency, edge bits, faces and ring slots, entry for entry), the build's `cells_checksum`, and after two steps at 0.1 spacings `geometry_checksum`, `faces_checksum` and `lists_checksum`: identical to main on all seven inputs at 128³ and 256³ (job 32373387; the final binary, with the round-3 volume fix below, again in job 32374526). Wave at 256³ runs out of memory in the bench's step on main and the branch alike, so for it the start's checksums only. The step's `cells_checksum` hashes face numbers that follow the repair's atomics and differs between runs of main, so it is not compared. The patched cells and bad tetrahedra per round are main's.

**Found by the tests.** The first full build of the branch failed `mesh_cases.valence` (`valence.ring.step.128` and `.129`): after an incremental second round failed, the third, full round still added the second round's carried volume (1.94 instead of 1) and refused the mesh. None of the seven bench inputs reach a third round. Fixed before the proof's final binary; A100 ctest 22 of 22 (job 32373935).

**Tests.** A100 ctest 22 of 22, the fuzz smoke included (job 32373935). CI's host steps (gcc 15, host Kokkos 5.0.99 OpenMP at 2 threads; CI builds a Serial Kokkos 5.0.2): comment and format checks pass, ctest 22 of 22 with sanitizers off and on, clang-tidy 18 clean on `voronoi_bench.cpp`, `start_test.cpp` and `mesh_test.cpp` with the library headers (job 32374568). A100 fuzz batch, profiles large, huge and giant on four A100 for 10 minutes (job 32373936): 52 cases, 26 exact, 17 refused for causes the input has, 9 out of time and no WRONG or CRASH; main's campaign of 2026-11-10 had 85 of 604 out of time (#49), and seed 4000000000005 rerun alone times out at 900 s on main and on the branch alike (job 32374773).

**Face output (about 17.5 ns per cell of the start).** It cannot come out of the check's pass: faces are numbered by the edge representatives, which `represent` sets only once the check has passed, and most inputs need a second round at 256³. Where its time goes (main, 128³ jittered, Nsight Systems job 32369633, one build): about 22 ms of kernels, of which the two walks around every edge in `number_faces` and `update_face_rings` are 9.0 ms (40 %), `tet_repair::prepare`'s resize copies of the start's arrays 5.0 ms, `faces` 2.2, the cell sums and sort 3.0, circumcentres 1.1. The assembly already holds every face's ring of tetrahedra (`mesh.ring`, in owner order); numbering the build's faces from it (a face's representative is its lowest tetrahedron; its ring rotated to the canonical start) would replace both walks, and sizing the start's arrays with `prepare`'s room would drop the copies. Both change `update_face_rings`, `number_faces` and `Mesh::rebuild` (voronoi_mesh.hpp), so they are left for the owner of those files; #48 (a rounded cell naming a face twice) needs the walk kept for cells with repeated neighbours.

**Allocation.** The first `Mesh::update` at 256³ spends 528 ms of its 1992 in `cudaMalloc` on main (Nsight Systems job 32372190): mapping fresh device memory costs about as much as the start's kernels. A pool for the mesh's arrays would remove it for every caller; not done here.

### Unfused mesh geometry: the same bits on CUDA and the host (2026-11-16, job 32413666)

Branch `unfused-on-main` (main 8cb1475 plus the change) against main 8cb1475, one A100 40 GB on `gpu1`, 1410 MHz max SM clock. The geometry kernels round every multiply and add on their own (`tet_mesh::rounded`; TESTING.md, "Bits across backends"); the clipping is unchanged.

Bits, per cell after a build and 3 steps of 0.1 spacings at 128³ (`voronoi_bench step --dump-cells`; volume and centroid of every particle, and a hash of its neighbours):

| input | CUDA vs OpenMP (unfused) | CUDA, main vs unfused: volumes / centroids changed | largest relative volume change | OpenMP, main vs unfused |
|---|---|---|---|---|
| uniform | 0 cells differ | 448,132 / 2,037,882 of 2,097,152 | 7.9e-16 | 0 |
| clustered | 0 | 448,662 / 2,037,347 | 6.9e-16 | 0 |
| halo | 0 | 449,762 / 2,036,167 | 6.9e-16 | 0 |
| cosmicweb | 0 | 448,599 / 2,020,103 | 6.9e-16 | 0 |
| jittered | 0 | 402,791 / 2,095,384 | 8.0e-16 | 0 |

Neighbours are the same in every comparison. main's CUDA differs from main's OpenMP exactly as from the unfused CUDA: the unfused device now gives the host's bits. Accuracy against a long-double reference (every 16th cell, `backends_probe` on `cpu-overlap`): unchanged, median relative volume error 1.28e-15 (uniform), 2.61e-15 (clustered), 2.55e-15 (halo), 1.55e-15 (cosmicweb), at most 2.0e-14 to 2.8e-13.

Speed, three interleaved rounds, medians (steps: `update` + `update_cells`, steps 2 to 10, 2 to 6 at 256³):

| input, N | rebuild | steps 0.03 | steps 0.1 | steps 0.3 |
|---|---|---|---|---|
| jittered 128³ | 1.000 | 1.032 | 1.022 | 1.015 |
| uniform 128³ | 1.000 | 1.042 | 1.033 | 1.022 |
| clustered 128³ | 1.000 | 1.041 | 1.030 | 1.014 |
| halo 128³ | 1.003 | 1.042 | 1.030 | 1.013 |
| cosmicweb 128³ | 0.998 | 1.044 | 1.031 | 1.021 |
| jittered 256³ | 1.000 | 1.044 | 1.029 | 1.014 |
| cosmicweb 256³ | 1.000 | 1.049 | 1.043 | 1.027 |

Time ratios, unfused / main; per-round ratios agree to 1 % for rebuild and steps. Where it goes: the cell sums (`update_cells`) +5 to +11 %, centres and faces +0.5 to +4 %, flips unchanged. Exact start: its clipping and assembly unchanged (build_ms), geometry +0.6 to +1.1 % and cells +5 to +6 %, together about +0.4 ms of 110 to 265 ms at 128³ and +3 ms of 1.6 to 2.5 s at 256³; the start totals are one sample per round and vary by up to 20 % from round to round in both builds.

### CPU and GPU together: where the GPU waits (2026-11-15)

Phase 1 of "overlapped CPU tail" ([ideas.md](ideas.md#cpu-and-gpu-together-2026-11-15)): measured only, no library change. Branch `cpu-overlap` on main 04eb9e4; one A100 40 GB on `gpu1` (18 of the node's 72 cores), 1305 MHz SM clock under load in every job; double precision, periodic unit box. Nsight Systems 2026 traces CUDA and the bench's NVTX phase ranges (`--nvtx=phases`, fenced at both ends, no per-kernel callbacks); `scripts/overlap_tables.py` splits each phase into GPU busy (the union of kernels, copies and memsets) and idle, and each idle interval by what the issuing host thread did meanwhile. Tail counts come from the measurement build (`-DVORONOI_MEASURE`, `warp_status`). Jobs 32373756 (128³), 32373758 and 32373759 (256³), 32373757 (surface 128³), 32373991 (volumes only); scripts `scripts/overlap_a100.sh`, `scripts/overlap_volumes_a100.sh`. The step's numbers include the bench's own fences (`Timings`, one per phase): three or four of the read-backs are those.

Rebuild (warp clipping, median of 3):

| input | N | clipping ms | GPU idle ms (%) | tail cells | tail pass ms (% of clipping) |
|---|---|---|---|---|---|
| jittered | 128³ | 49.3 | 0.32 (0.6 %) | 0 | 0.07 (0.1 %) |
| uniform | 128³ | 95.7 | 0.32 (0.3 %) | 33 | 2.54 (2.7 %) |
| clustered | 128³ | 123.1 | 0.41 (0.3 %) | 68 | 6.28 (5.1 %) |
| halo | 128³ | 135.2 | 0.32 (0.2 %) | 261 | 15.63 (11.6 %) |
| cosmicweb | 128³ | 161.5 | 0.32 (0.2 %) | 976 | 17.11 (10.6 %) |
| jittered | 256³ | 392.3 | 0.91 (0.2 %) | 0 | 0.50 (0.1 %) |
| uniform | 256³ | 748.3 | 0.57 (0.1 %) | 306 | 5.65 (0.8 %) |
| clustered | 256³ | 904.6 | 0.36 (0.0 %) | 315 | 6.31 (0.7 %) |
| halo | 256³ | 937.9 | 0.47 (0.1 %) | 1235 | 35.52 (3.8 %) |
| cosmicweb | 256³ | 1308.3 | 0.56 (0.0 %) | 8501 | 113.65 (8.7 %) |

Exact start (Mesh::update from scratch, one run):

| input | N | start ms | GPU idle ms (%) | of which allocation | read-back trips ms | tail pass ms | exact re-clipped cells | warp kernel ends, first check ends (ms) |
|---|---|---|---|---|---|---|---|---|
| jittered | 128³ | 140.4 | 12.6 (9.0 %) | 9.0 | 0.95 | 0.07 | 0 | 60, 109 |
| uniform | 128³ | 261.6 | 16.7 (6.4 %) | 12.6 | 1.34 | 15.43 | 129 | 118, 168 |
| clustered | 128³ | 303.6 | 31.2 (10.3 %) | 26.9 | 1.25 | 23.32 | 134 | 149, 204 |
| halo | 128³ | 269.8 | 52.7 (19.5 %) | 49.1 | 1.06 | 14.95 | 0 | 137, 202 |
| cosmicweb | 128³ | 343.7 | 16.6 (4.8 %) | 12.6 | 1.46 | 52.91 | 108 | 166, 231 |
| jittered | 256³ | 2046.3 | 629.4 (30.8 %) | 623.7 | 6.99 | 3.12 | 830 | 462, 853 |
| uniform | 256³ | 2422.9 | 525.9 (21.7 %) | 520.3 | 18.60 | 16.46 | 277 | 928, 1349 |
| clustered | 256³ | 2600.0 | 639.3 (24.6 %) | 633.3 | 10.74 | 22.30 | 203 | 1019, 1426 |
| halo | 256³ | 2524.6 | 487.6 (19.3 %) | 482.2 | 5.10 | 67.34 | 393 | 1019, 1432 |
| cosmicweb | 256³ | 2974.2 | 541.4 (18.2 %) | 534.9 | 18.21 | 154.37 | 156 | 1362, 1876 |

Moving step (update + update_cells, median of steps 2 to the last):

| input | N | spacings | step ms | GPU idle ms (%) | read-backs | trips ms | launches | host code ms | stuck, host-fallback ms |
|---|---|---|---|---|---|---|---|---|---|
| jittered | 128³ | 0.03 | 13.06 | 0.62 (4.7 %) | 13 | 0.22 | 67 | 0.17 | 0, 0.0 |
| jittered | 128³ | 0.1 | 21.15 | 0.73 (3.5 %) | 17 | 0.28 | 74 | 0.21 | 0, 0.0 |
| jittered | 128³ | 0.3 | 37.81 | 0.89 (2.4 %) | 19 | 0.35 | 82 | 0.24 | 0, 0.0 |
| uniform | 128³ | 0.03 | 11.02 | 0.56 (5.1 %) | 11 | 0.19 | 60 | 0.16 | 0, 0.0 |
| uniform | 128³ | 0.1 | 14.75 | 0.68 (4.6 %) | 15 | 0.25 | 71 | 0.19 | 0, 0.0 |
| uniform | 128³ | 0.3 | 24.21 | 0.81 (3.4 %) | 19 | 0.31 | 84 | 0.22 | 0, 0.0 |
| clustered | 128³ | 0.03 | 10.70 | 0.53 (5.0 %) | 10 | 0.18 | 57 | 0.16 | 0, 0.0 |
| clustered | 128³ | 0.1 | 14.25 | 0.68 (4.8 %) | 15 | 0.25 | 70 | 0.19 | 0, 0.0 |
| clustered | 128³ | 0.3 | 23.03 | 0.80 (3.5 %) | 19 | 0.31 | 82 | 0.23 | 0, 0.0 |
| halo | 128³ | 0.03 | 10.92 | 0.53 (4.9 %) | 10 | 0.18 | 57 | 0.15 | 0, 0.0 |
| halo | 128³ | 0.1 | 14.50 | 0.70 (4.9 %) | 15 | 0.25 | 72 | 0.19 | 0, 0.0 |
| halo | 128³ | 0.3 | 23.95 | 0.85 (3.6 %) | 20 | 0.33 | 86 | 0.24 | 0, 0.0 |
| cosmicweb | 128³ | 0.03 | 10.67 | 0.52 (4.9 %) | 10 | 0.18 | 56 | 0.16 | 0, 0.0 |
| cosmicweb | 128³ | 0.1 | 14.18 | 0.68 (4.8 %) | 15 | 0.25 | 71 | 0.19 | 0, 0.0 |
| cosmicweb | 128³ | 0.3 | 22.91 | 0.81 (3.5 %) | 18 | 0.32 | 81 | 0.22 | 0, 0.0 |
| jittered | 256³ | 0.03 | 77.42 | 0.67 (0.9 %) | 11 | 0.23 | 62 | 0.19 | 0, 0.0 |
| jittered | 256³ | 0.1 | 109.45 | 0.80 (0.7 %) | 15 | 0.28 | 72 | 0.22 | 0, 0.0 |
| jittered | 256³ | 0.3 | 193.62 | 1.09 (0.6 %) | 20 | 0.42 | 89 | 0.29 | 0, 0.0 |
| uniform | 256³ | 0.03 | 69.74 | 0.57 (0.8 %) | 10 | 0.19 | 57 | 0.16 | 0, 0.0 |
| uniform | 256³ | 0.1 | 83.66 | 0.76 (0.9 %) | 14 | 0.29 | 70 | 0.21 | 0, 0.0 |
| uniform | 256³ | 0.3 | 121.65 | 1.10 (0.9 %) | 17 | 0.41 | 81 | 0.30 | 0, 0.0 |
| clustered | 256³ | 0.03 | 69.00 | 0.58 (0.8 %) | 10 | 0.19 | 57 | 0.17 | 0, 0.0 |
| clustered | 256³ | 0.1 | 82.08 | 0.69 (0.8 %) | 13 | 0.26 | 67 | 0.19 | 0, 0.0 |
| clustered | 256³ | 0.3 | 117.51 | 0.79 (0.7 %) | 17 | 0.29 | 82 | 0.21 | 0, 0.0 |
| halo | 256³ | 0.03 | 69.30 | 0.54 (0.8 %) | 10 | 0.18 | 57 | 0.15 | 0, 0.0 |
| halo | 256³ | 0.1 | 83.40 | 0.70 (0.8 %) | 13 | 0.26 | 67 | 0.19 | 0, 0.0 |
| halo | 256³ | 0.3 | 121.14 | 0.82 (0.7 %) | 17 | 0.31 | 81 | 0.22 | 0, 0.0 |
| cosmicweb | 256³ | 0.03 | 68.46 | 0.57 (0.8 %) | 10 | 0.19 | 57 | 0.16 | 0, 0.0 |
| cosmicweb | 256³ | 0.1 | 81.32 | 0.69 (0.8 %) | 13 | 0.25 | 67 | 0.20 | 0, 0.0 |
| cosmicweb | 256³ | 0.3 | 117.64 | 0.87 (0.7 %) | 17 | 0.35 | 82 | 0.24 | 0, 0.0 |

What the numbers say:

- **The host does nothing while kernels run.** In every phase it spends the kernels blocked in a synchronising call; host code outside the CUDA API is 0.2 ms per step and under 1.5 ms per start.
- **Rebuild:** the GPU never waits (idle under 1 %). The tail is the per-thread pass over the cells the warp kernel overflows: up to 11.6 % of the clipping at 128³ (halo, cosmic web) and 8.7 % at 256³ (cosmic web, 8501 cells). A tail cell is known on the device when its warp finishes it, on the host only after the warp kernel.
- **Exact start:** the GPU waits 5 to 20 % of the start at 128³ and 18 to 31 % at 256³, nearly all of it in synchronous `cudaMalloc` and `cudaFree` (81 and 43 calls; at 256³ three single `cudaMalloc` calls of 100 to 210 ms after the assembly). Read-backs cost about 1 ms at 128³ (6 round trips), 5 to 19 ms at 256³. The cells for exact re-clipping are known only after the first check (60 to 65 ms after the warp kernel at 128³, about 400 ms at 256³).
- **Moving step:** the GPU waits 2.4 to 5.1 % at 128³ and under 1 % at 256³ (0.5 to 1.1 ms a step): 10 to 20 read-backs (counters of the repair rounds, face counts, the volume check) of 0.02 ms each and the launch gaps between them. In none of the 240 steps (30 runs of 8) did a triangle stick, so no host fallback ran.
- **Copies (pageable, as the benches do them):** host to device 2.2 to 2.7 GB/s in library_bench and 10.7 GB/s in voronoi_bench, device to host 4.3 GB/s.

Volumes only (`library_bench build --path=warp-host-vc`: points in host memory to volumes and centroids in host memory; medians of the timed builds, ms):

| input | N | build | kernels | points in | volumes and centroids out | GPU idle |
|---|---|---|---|---|---|---|
| jittered | 128³ | 96.5 | 49.2 | 19.4 | 25.0 | 3.3 (3.4 %) |
| uniform | 128³ | 136.8 | 95.0 | 24.9 | 14.2 | 2.8 (2.0 %) |
| clustered | 128³ | 165.0 | 122.4 | 19.8 | 14.3 | 2.9 (1.7 %) |
| halo | 128³ | 176.2 | 135.0 | 24.9 | 13.3 | 2.7 (1.5 %) |
| cosmic web | 128³ | 201.2 | 159.8 | 24.0 | 14.1 | 2.7 (1.4 %) |
| jittered | 256³ | 719.2 | 392.5 | 188.3 | 132.0 | 6.6 (0.9 %) |
| uniform | 256³ | 1070.7 | 746.1 | 188.5 | 129.5 | 6.7 (0.6 %) |

The copies are 20 to 46 % of this path: 50 MB of points in at 2.6 GB/s and 67 MB out, from and to pageable host memory. (The GPU counts as busy during a copy, so the idle column misses that the copies and the clipping run one after the other.)

Bulk cells on the CPU: the same code on a full 72-core node clips 2.6 M cells/s against 33.7 on the A100 (13×) and steps 8.5 against 144 M cells/s (17×) (README, library benchmark). Giving the CPU its share of bulk cells could save at most 7 % of a rebuild, and copying the positions over alone costs 4.7 ms at 128³ (50 MB at 10.7 GB/s), about 10 % of a jittered clipping; a step would need the mesh (about 1950 B/cell) mirrored on the host every step. So no candidate splits bulk cells.

### CPU and GPU from one build (2026-11-16)

S0 and S2 of [ideas.md](ideas.md#cpu-and-gpu-together-2026-11-15): one build on Kokkos with CUDA 13.0 and OpenMP (`/u/uli/kokkos-cuda-omp`), and `voronoi::Engine` choosing cpu, gpu or hybrid at run time. One A100 40 GB on `gpu1`, 18 host threads; jobs 32378021 (build cost, CI host steps, first speed run), 32407747 (speed with `--fmad=false`, bits), 32410447 (fixed `modes_test`, bits, second speed run).

Build cost, `voronoi_bench` at -j1: 143 s in both builds; binary 8,905,800 bytes (CUDA only) against 8,947,864 (+0.5 %). `mesh_test`: 99 s in both; 6,058,536 against 6,100,688 bytes. CI's host steps (gcc 15 OpenMP build, sanitizers off and on): 23 of 23; clang-tidy on `modes_test` (with `voronoi_engine.hpp`): 0 warnings. A100 ctest after the Engine fix: 23 of 23 in both builds.

Speed of the two-backend build against the CUDA-only one (job 32407747; three interleaved rounds, median; rebuild = warp clipping median of 5, start = exact start, steps = `update` + `update_cells` median of steps 2 to 10 (2 to 6 at 256³)): all 28 cases within 1 %, geometric mean 0.9996, range 0.993 to 1.009.

| input, N | rebuild | start 0.03 / 0.3 | steps 0.03 / 0.3 |
|---|---|---|---|
| jittered 128³ | 1.000 | 0.997 / 1.001 | 0.993 / 1.009 |
| uniform 128³ | 1.001 | 0.998 / 1.000 | 1.002 / 0.996 |
| clustered 128³ | 1.000 | 0.999 / 1.000 | 1.005 / 1.000 |
| halo 128³ | 1.002 | 0.995 / 1.001 | 1.003 / 1.000 |
| cosmicweb 128³ | 0.999 | 0.997 / 0.995 | 0.999 / 1.000 |
| jittered 256³ | 1.000 | 0.999 / – | 0.999 / – |

Time ratios, two backends / CUDA only. Two more runs (jobs 32378021 and 32410447) agree for rebuild (0.998 to 1.005) and steps (0.996 to 1.010). The exact start is one sample per round: in those two runs single rounds of either build were 10 to 25 % (once 2×) slower than the others, so their start medians range 0.86 to 1.11; only 32407747, where the rounds agree to 1 %, resolves the start to 1 %. A start claim at 1 % needs repeats per round. The bench's host advection is slower in the two-backend build (260 against 30 ms at 128³: OpenMP binds the main thread to one core, and the bench's 16 `std::thread`s inherit it); it is outside every timed phase. Tetrahedron counts differ from run to run on CUDA in both builds (flip scheduling), faces never.

Unfused device arithmetic (`-DVORONOI_FMAD=OFF`, `--fmad=false`), same job, against the default two-backend build:

| input, N | rebuild | start 0.03 / 0.3 | steps 0.03 / 0.3 |
|---|---|---|---|
| jittered 128³ | 1.046 | 1.021 / 1.020 | 1.001 / 0.995 |
| uniform 128³ | 1.107 | 1.042 / 1.041 | 0.989 / 1.009 |
| clustered 128³ | 1.086 | 1.036 / 1.038 | 0.990 / 0.993 |
| halo 128³ | 1.077 | 1.052 / 1.053 | 0.985 / 1.003 |
| cosmicweb 128³ | 1.113 | 1.053 / 1.055 | 0.993 / 1.000 |
| jittered 256³ | 1.046 | 1.014 / – | 0.994 / – |

The cost is in the clipping (rebuild +4.6 to +11.3 %, start +1.4 to +5.5 %); steps are unchanged (−1.5 to +0.9 %). What it buys: CPU and GPU meshes equal in every bit (TESTING.md, "Bits across backends").

Host against device clipping (`backends_probe`, 128³, all cells on 18 threads with the per-thread kernel, against the A100's warp kernel): 22.6 to 28.7× slower; the tail cells alone cost 118 to 275 µs per cell per core.

### Rounding per mode (2026-11-16, jobs 32422876 and 32429908)

Branch `cpu-gpu-modes` (main fbe6a0c, S0/S2, and both arithmetics compiled in, `Layout::same_bits_as_host` off by default; TESTING.md, "Bits across backends") against main fbe6a0c, one A100 40 GB on `gpu1`, 1410 MHz max SM clock. Three builds compared: main, the branch's default (gpu mode), and the branch with `--same-bits` (hybrid's arithmetic).

Bits. gpu mode equals main in every bit: per cell (volume, centroid, neighbours) after a build and 3 steps of 0.1 spacings on uniform, clustered, halo, cosmicweb and jittered at 128³, and a 2D mesh of 2²⁰ cells over 5 updates. hybrid equals cpu in every bit (`modes_test`, 3D, 2D, 1D; `backends_probe` at 128³ on uniform, clustered, halo and cosmicweb: all 2,097,152 volumes in 3 of 3 updates; the per-thread clipping kernel in Rounded on the device equals the host's in every cell). Accuracy against a long-double reference is the same in all three.

Speed, time ratios against main (job 32422876: three interleaved rounds, median; steps: `update` + `update_cells`, steps 2 to 10, 2 to 6 at 256³):

| input, N | gpu: rebuild / steps 0.03 / 0.1 / 0.3 | same_bits: rebuild / steps 0.03 / 0.1 / 0.3 |
|---|---|---|
| jittered 128³ | 1.000 / 1.007 / 0.999 / 1.006 | 1.000 / 1.034 / 1.017 / 1.007 |
| uniform 128³ | 1.000 / 1.005 / 0.996 / 1.001 | 1.000 / 1.051 / 1.026 / 1.019 |
| clustered 128³ | 1.000 / 0.999 / 1.000 / 0.999 | 1.002 / 1.040 / 1.031 / 1.023 |
| halo 128³ | 0.999 / 1.004 / 1.001 / 1.000 | 1.000 / 1.039 / 1.034 / 1.012 |
| cosmicweb 128³ | 1.000 / 0.997 / 1.012 / 1.006 | 1.007 / 1.036 / 1.029 / 1.016 |
| jittered 256³ | 1.000 / 1.001 / 1.000 / 0.999 | 1.000 / 1.043 / 1.030 / 1.012 |
| cosmicweb 256³ | 1.000 / 1.002 / 1.001 / 1.002 | 1.010 / 1.050 / 1.046 / 1.028 |

The one gpu case above 1 % (cosmicweb 128³ at 0.1) was 0.998 in six more rounds (job 32429908); there jittered 256³ at 0.3 was 1.007. The exact start's totals are one sample per round and vary by up to 20 % between rounds in all three builds, so the start is read from its phases (median over every run): gpu 0.998 to 1.004 of main in clipping and assembly (`build_ms`), geometry and cells on all seven cases except one round-to-round outlier (jittered 256³ `build_ms` 1.068 in job 32422876, 0.999 in six rounds of job 32429908). same_bits: start clipping and assembly 1.000 to 1.015, except cosmicweb 256³ at 1.054 (both jobs: its tail of per-thread cells is long), geometry +0.6 to +1.2 %, cells +4 to +6 %. In the steps the cost of same_bits is the cell sums (+5 to +11 %) and centres and faces (+0.5 to +4 %); flips are unchanged.

The clipping's per-thread kernel alone (`backends_probe`, all cells on the device): Rounded costs +0.1 to +1.1 % (832.7 / 840.2 ms uniform, 1471 / 1483 ms cosmicweb at 128³); a warp build whose tail runs in Rounded: −0.2 to +1.2 %.

Again after merging main 5aac4bf (#60 B2: the warp kernel's overflow cells to the filtered warp kernels; job 32431927, same three builds, surface added): gpu mode equals main in every bit on all six inputs at 128³ (surface included) and in 2D; rebuild 0.999 to 1.004 and steps 0.995 to 1.010 of main (eight inputs and sizes, three rounds); hybrid equals cpu in every bit (`modes_test`, `backends_probe` on four inputs); same_bits steps +0.7 to +5.1 %, rebuild within 0.3 %; the rounded per-thread kernel +0.2 to +0.9 % and equal to the host in every cell. The exact start's totals and its `build_ms` swung by up to ±15 % between rounds in all three builds in this job (jittered 128³ `build_ms` 0.937, jittered 256³ 1.059, with the same bits), so the start's speed rests on job 32429908. B2's `clipping_test` (volumes only, rings and faces give the same cells) passes in both builds.

Build: `voronoi_bench` at -j1 in 214 s against main's 204 s (+4.9 %; 216 against 206 s in job 32431927), binary 15,327,536 against 14,417,232 bytes (+6.3 %), both variants of the geometry and per-thread kernels instantiated. A100 ctest 23 of 23 in the CUDA and the CUDA + OpenMP build; CI host steps (gcc 15, sanitizers off and on) 23 of 23; clang-tidy 18: 0 warnings.

### The 2× step (branch `step-2x`, 2026-11-18)

Goal: the moving step (`update()` + `update_cells()`) about 2× faster. One A100 SXM4 40 GB on Raven `gpu1` at 1305 MHz (application clock, from `nvidia-smi` in every job), double precision, periodic unit box, the bench's smooth flow (`voronoi_bench step`). Main is 3f27c51; head is 31b41fd, measured in the same job as main. Scripts: `scripts/step2x_a100.sh` (jobs), `scripts/step2x_tables.py` (tables), `scripts/resort_bits.py` (cells by particle); logs in `build-jobs/logs` of the worktree on Raven. Each lever's row, against the code before it, is in [tried.md](tried.md#moving-step).

What changed:

1. **Re-sort on by default.** `Layout::every = 25`: every 25 updates the cells are renumbered along a Hilbert curve, the tetrahedra and faces after them. Particle ids change: `Report::resorted`, `permutation()`, `permute()`; `order_test` checks that a caller who follows the permutation sees, particle by particle, the mesh of a run that never re-sorts. A trigger on flips per cell (`Layout::flips`, opt-in) is as fast at 0.03 and 0.1 spacings and 5 % faster at 0.3, but flip counts vary from run to run (the votes of a flip round follow the order in which threads list their work), so it would re-sort on different updates in two runs of one input.
2. **Per-cell sums from a stored |A·d|.** `faces` keeps |A·d| per face (8 B a face, about 62 B/cell); `update_cells()` reads it and the face centroid instead of the area and both positions (the neighbour's side still reads the owner's position for its centroid). `cell_sums` −28 %. Incremental lists up to n/2 created tetrahedra (was n/5).
3. **Fewer launches and host waits (S4): not done.** Bound 2.4–5.1 % at 128³, under 1 % at 256³ (H3); its places are in `tet_repair.hpp`, which branch `local-repair` was editing at the time (merged as abd3d39).
4. **Reconnect in one kernel.** `dirty_rings` finds each dirty face, marks it, and the marking thread walks and writes its ring; the starts are set after. Was `dirty_faces` then `append_rings`.
5. **CUB.** Renumber's touched slots by `cub::DeviceSelect::Flagged` (was a Kokkos scan, 0.49 ms at 128³).

Rates, ten steps (steps 2–10 averaged, two rounds), update + `update_cells()`, M cells/s and ns per cell (job 32516595). Ten steps hold no re-sort, so this is levers 2 to 5:

| n | input | spacings | main M cells/s | head M cells/s | head / main | main ns/cell | head ns/cell |
|---|---|---|---|---|---|---|---|
| 128³ | jittered | 0.03 / 0.1 / 0.3 | 163.8 / 99.0 / 58.2 | 179.1 / 109.1 / 61.9 | 1.093 / 1.101 / 1.063 | 6.11 / 10.10 / 17.18 | 5.58 / 9.17 / 16.16 |
| 128³ | uniform | 0.03 / 0.1 / 0.3 | 197.5 / 146.8 / 85.7 | 218.2 / 160.6 / 93.3 | 1.105 / 1.094 / 1.089 | 5.06 / 6.81 / 11.67 | 4.58 / 6.23 / 10.71 |
| 128³ | clustered | 0.03 / 0.1 / 0.3 | 201.6 / 151.1 / 90.4 | 221.5 / 165.1 / 96.6 | 1.099 / 1.093 / 1.068 | 4.96 / 6.62 / 11.06 | 4.51 / 6.06 / 10.35 |
| 128³ | halo | 0.03 / 0.1 / 0.3 | 198.4 / 148.2 / 87.2 | 219.5 / 162.1 / 95.2 | 1.106 / 1.094 / 1.091 | 5.04 / 6.75 / 11.46 | 4.56 / 6.17 / 10.50 |
| 256³ | jittered | 0.03 / 0.1 / 0.3 | 220.5 / 153.7 / 84.8 | 244.4 / 169.9 / 93.6 | 1.109 / 1.106 / 1.104 | 4.54 / 6.51 / 11.79 | 4.09 / 5.88 / 10.68 |
| 256³ | uniform | 0.03 / 0.1 / 0.3 | 246.6 / 204.0 / 135.6 | 274.6 / 226.0 / 149.2 | 1.114 / 1.107 / 1.101 | 4.06 / 4.90 / 7.38 | 3.64 / 4.43 / 6.70 |
| 256³ | clustered | 0.03 / 0.1 / 0.3 | 248.3 / 207.6 / 143.4 | 277.3 / 230.7 / 158.6 | 1.117 / 1.111 / 1.106 | 4.03 / 4.82 / 6.97 | 3.61 / 4.33 / 6.31 |
| 256³ | halo | 0.03 / 0.1 / 0.3 | 246.8 / 204.5 / 139.0 | 275.3 / 227.0 / 153.4 | 1.115 / 1.110 / 1.104 | 4.05 / 4.89 / 7.20 | 3.63 / 4.41 / 6.52 |

500 steps of the smooth flow, jittered start, the rate over the whole run with the re-sorts paid (jobs 32516596–99):

| n | spacings | main M cells/s | head M cells/s | head / main | head: steps 2–11 / last 10 | main: steps 2–11 / last 10 | re-sorts, ms each |
|---|---|---|---|---|---|---|---|
| 128³ | 0.03 | 117.9 | 194.3 | 1.65 | 178.4 / 178.9 | 162.4 / 92.8 | 20, 9.9 |
| 128³ | 0.1 | 65.9 | 142.1 | 2.16 | 108.9 / 125.3 | 98.8 / 51.6 | 20, 13.3 |
| 128³ | 0.3 | 39.3 | 84.4 | 2.15 | 62.9 / 74.2 | 58.0 / 34.7 | 20, 12.3 |
| 256³ | 0.1 | 88.7 | 195.5 | 2.20 | 169.0 / 163.5 | 152.8 / 65.9 | 20, 111.6 |

Re-sort triggers, 500 steps, on lever 1 alone (jobs 32515066–69; M cells/s over the run; main without a re-sort for scale):

| trigger | 128³ 0.03 | 128³ 0.1 | 128³ 0.3 | 256³ 0.1 |
|---|---|---|---|---|
| none (main) | 117.8 | 65.7 | 39.2 | 88.9 |
| every 25 | 179.0 | 130.8 | 77.5 | 180.0 |
| every 50 | 180.8 | 124.5 | 68.6 | 176.7 |
| every 100 | 176.0 | 109.8 | 56.9 | 159.5 |
| 1 flip per cell | 178.8 | 131.0 | 81.4 | |
| 2 / 4 / 8 flips per cell | 164.9 / 147.6 / 117.9 | 126.9 / 111.3 / 92.2 | 80.0 / 74.3 / 64.2 | |

Bits. Topology is exact in every run. Levers 2 to 5 change no bit: the geometry, faces and lists checksums of head equal main's on seven inputs (jittered, uniform, clustered, halo, lattice, wave, cosmicweb) at 0.03, 0.1 and 0.3 spacings, five steps each, 105 of 105 (job 32516600; the layout cannot re-sort in five steps). `cells_checksum`, which also hashes face numbers, differs between two runs of main itself (job 32515833): face numbers depend on the order of atomics, as before. The re-sort changes ids by design; per particle after 500 steps at 128³, 0.1 spacings (job 32516923), against main: 0 of 2,097,152 neighbour sets differ, 1,495,265 volumes differ in their bits, by at most 3.2e-14 relative, centroids by at most 2.4e-14 cell sizes, as in [measure_resort_2026-11-10.md](measure_resort_2026-11-10.md#bits).

Cost. 8 B a face for |A·d| (computed: about 62 B/cell at 7.7 faces per cell); a re-sort is about one step's time every 25 updates, paid also on a flow whose order does not decay (at 128³ 10–13 ms once in 25 steps of 12–34 ms, 2–4 %). gpu stays the default mode; `modes_test` (hybrid equals cpu in every bit) passes.

Tests: A100 ctest 23 of 23 on head (job 32516982); CI's host steps in a gpu1 job (comments, format, gcc 15 with sanitizers off and on 23 of 23 each, clang-tidy 18 without warnings on the changed programs; job 32516922, then its warnings fixed); fuzz smoke in ctest, and a CPU batch of the small profile (below).

**Again after merging main abd3d39 (S2, local repair), in one gpu1 job (32520263, `scripts/step2x_proof_a100.sh`, ref abd3d39, new b7be4c1).** A100 ctest 23 of 23 (`modes`: hybrid equals cpu); S2's verified jump and flow cases with `--fallback=local`, every step against a fresh build: 48 of 48; S2's jump rates under the default `cheapest` fallback −2.7 to +17 % in rate against abd3d39 (12 steps each, most of them rebuild- or repair-bound); geometry, faces and lists checksums equal on seven inputs at three spacings, 105 of 105; per particle after 500 steps (128³, 0.1): 0 neighbour sets differ, volumes within 3.2e-14, centroids 2.4e-14 cell sizes; CI host steps 23 of 23 with sanitizers off and on, comments and format ok, clang-tidy 18 without warnings. Rates, ten steps, M cells/s abd3d39 → head: 128³ jittered 163.2 / 98.6 / 57.7 → 178.5 / 108.7 / 61.5 (0.03 / 0.1 / 0.3 spacings; 1.094 / 1.102 / 1.066), uniform 1.104 / 1.097 / 1.091, clustered 1.105 / 1.094 / 1.055, halo 1.100 / 1.091 / 1.091; 256³ jittered 219.7 / 153.4 / 85.2 → 244.2 / 169.4 / 93.4 (1.112 / 1.105 / 1.097), the other three inputs 1.101 to 1.118. 500 steps: 128³ 0.1 65.7 → 143.2 (2.18×), 0.03 117.9 → 194.1 (1.65×), 0.3 39.2 → 84.3 (2.15×); 256³ 0.1 88.8 → 196.0 (2.21×).

Fuzz, CPU (OpenMP, gcc 15, small profile): 236,785 cases on 31b41fd and (so far) 233,128 on the merged head; 8 and 9 WRONG, every one of them also WRONG with main 3f27c51's fuzz program ([#62](https://github.com/Ulli1991/voronoi-cpp/issues/62)); the HANGs (256 and 299) are cases killed at the 120 s limit before their first call on a loaded node (one rerun: refused for reach, as on main).

Against the 2× goal: 1.06–1.12× on ten-step runs (levers 2 to 5), 1.65× (0.03 spacings) to 2.2× (0.1 and 0.3) over 500 steps with the re-sort. The rest of a ten-step run is the flips (`plan`, `apply`, `find`: 5.8 ms of 21 at 128³, 0.1), the screen (2.4 ms), the faces (2.7 ms) and the ring walks (3.3 ms), which these levers do not touch.

### The step levers (branch `step-levers`, 2026-11-18)

Guided by the profile of 2026-11-18 (branch `profile-2026-11-18`, research_notes/profile_2026-11-18.md: per-kernel speed of light and memory of light). One A100 SXM4 40 GB on Raven (`gpu1`, `gpudev`) at 1305 MHz (application clock, from `nvidia-smi` in every job), double precision, periodic unit box, jittered input, the bench's smooth flow (`step` mode of the bench), update + `update_cells()`. Base: step-2x 1741a1a (= main). Scripts: `scripts/levers_a100.sh` (modes kernels, rates128, rates256, long, bits, ncu, memory, verify), `scripts/levers_build.sh` (registers and stack per kernel, from `cuobjdump`), `scripts/levers_proof_a100.sh`; Nsight Compute through the Kokkos Tools connector `scripts/kp_profile.cpp` and `scripts/profile_ncu.py` (from the profile branch). Logs in `build-jobs/logs` of the worktree on Raven. Each change's row, kept or dropped, is in [tried.md](tried.md#moving-step).

What changed:

1. **Lever 1: the repair at its registers** (`3cc282a`). The plan step searches its first valid flip, then checks that one flip's cavity: the threads of a warp run one search and one check together (one unrolled copy per candidate left 4 of 32 threads active). The cavity check (`closes23`, `closes32`) and the apply step (`apply_flip`, one code path for 2->3 and 3->2) work by formula from the rows of the old tetrahedra; every array has a fixed size and stays in registers. Rounds of a dirty list of at most 2048 tetrahedra run back to back on one team (`small_rounds`), with barriers instead of launches and one host wait for all of them; when the lists hold a round, the plan step also unlists.
2. **Lever 2: the screen keeps its centre in registers** (`ecb9992`). `store_centre` returns the row it stores and the neighbour tests read it from registers; each far vertex is found from t's corners in registers. The 64-register cap stays: without it the screen has no spills but is 4 to 7 % slower.

Per kernel at 128³, 0.1 spacings, Nsight Compute on step 3 (jobs 32527071, 32529634; step-2x → levers):

| kernel | ms | registers | local B/cell | threads active of 32 |
|---|---|---|---|---|
| `apply` | 2.54 → 1.43 | 58 → 72 | 526 → 27 | 7.5 → 10.9 |
| `plan` | 2.21 → 1.22 | 156 → 96 | 369 → 0 | 15.5 → 15.0 |
| `find` | 1.03 → 0.75 | 168 → 160 | 35 → 35 | 19.3 → 22.6 |
| `small_rounds` (the tail rounds) | – → 0.29 | 146 | 0 | 18.9 |
| `unlist` | 0.12 → 0.10 | 16 | 0 | 32 |
| `screen` (lever 2) | 2.17 → 1.78 | 64 | 168 → 985 | 20.9 → 24.8 |

The repair (`find`, `plan`, `apply`, `unlist`, `release`, small rounds) went from 5.85 to 3.79 ms, about 1.0 ns/cell of the up to 2.6 the profile bounded.

500 steps, the rate over the run with the re-sorts paid (jobs 32529712, 32530154), M cells/s:

| n | spacings | step-2x | lever 1 | levers 1 + 2 | against step-2x |
|---|---|---|---|---|---|
| 128³ | 0.03 | 193.5 | 204.2 | 211.3 | 1.09 |
| 128³ | 0.1 | 139.9 | 151.7 | 161.4 | 1.15 |
| 128³ | 0.3 | 84.2 | 93.8 | 96.9 | 1.15 |
| 256³ | 0.1 | 194.8 | 200.5 | 216.7 | 1.11 |

Rates, ten steps (steps 2–10 averaged, two rounds, no fences; jobs 32530145, 32530148):

| n | input | spacings | step-2x M cells/s | lever 1 | levers 1 + 2 | lever 1 / step-2x | levers 1 + 2 / step-2x | step-2x ns/cell | lever 1 | levers 1 + 2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 128³ | jittered | 0.03 | 178.6 | 191.2 | 197.2 | 1.071 | 1.104 | 5.60 | 5.23 | 5.07 |
| 128³ | jittered | 0.1 | 109.1 | 121.6 | 125.1 | 1.115 | 1.146 | 9.17 | 8.22 | 8.00 |
| 128³ | jittered | 0.3 | 61.4 | 72.5 | 73.9 | 1.180 | 1.203 | 16.27 | 13.79 | 13.53 |
| 256³ | jittered | 0.03 | 243.9 | 251.0 | 259.5 | 1.029 | 1.064 | 4.10 | 3.98 | 3.85 |
| 256³ | jittered | 0.1 | 170.1 | 181.0 | 186.7 | 1.064 | 1.097 | 5.88 | 5.52 | 5.36 |
| 256³ | jittered | 0.3 | 93.7 | 104.1 | 106.6 | 1.111 | 1.138 | 10.68 | 9.61 | 9.38 |

Bits. Topology is exact in every run. The geometry, faces and lists checksums of both levers equal step-2x's on seven inputs (jittered, uniform, clustered, halo, lattice, wave, cosmicweb) at 0.03, 0.1 and 0.3 spacings, five steps each: 105 of 105 (job 32530150, lever 1 and levers 1 + 2; again in the proof, job 32530159). Which flips win a round still follows the work list's order, so `cells_checksum` (face numbers) differs between runs, as before.

Memory. 2045 B/cell for the moving step at 128³ (the arrays at the peak, the bench's 48 included; job 32530157), unchanged by both levers: `small_rounds` adds 80 bytes in all.

Tests (job 32530159 on the A100, 32530162 on a CPU node, on `ecb9992`): A100 ctest 23 of 23 (`modes`: hybrid equals cpu); every step against a fresh exact build on seven inputs at three spacings (262,144 cells, four steps) and on jittered input at 128³: 93 of 93; S2's jump and flow cases with `--fallback=local`: 48 of 48; CI's host steps: comments and format ok, gcc 15 with sanitizers off and on 23 of 23 each, clang-tidy 18 without warnings on `voronoi_bench.cpp`, `repair_test.cpp`, `modes_test.cpp`.

Few flips (Sedov-like steps; 128³ jittered, kernels fenced, job 32530305): at 0.003 spacings (6,122 flips a step) the repair takes 3.43 → 2.80 ms and the step 8.99 → 8.35 ms; at 0.01 (20,320 flips) 4.13 → 3.19 ms and 10.15 → 9.14 ms. The screen (1.6 ms) and the full-mesh passes are now most of such a repair.

**After levers 1 and 2** (head `d4f61d1`: lever 5a, CUB's scan for the per-cell list starts; M1, 32-bit repair locks; M2, the repair's dirty flags as bytes). Lever 3 (`dirty_rings` with the ring in registers, or one walk per thread) was slower and dropped; see [tried.md](tried.md#moving-step).

| 128³ jittered, M cells/s | 0.03 | 0.1 | 0.3 |
|---|---|---|---|
| ten steps, levers 1 + 2 → head (job 32540488) | 198.5 → 208.0 | 125.4 → 131.1 | 74.4 → 78.6 |
| 500 steps, levers 1 + 2 → head (job 32540489) | 217.1 → 227.8 | 163.6 → 169.3 | 97.6 → 102.4 |

At 256³ (0.03 / 0.1 / 0.3 spacings) ten steps 259.6 / 187.0 / 107.0 → 263.9 / 191.1 / 109.8 M cells/s; 500 steps at 0.1 217.7 → 221.1 (job 32540490). Against step-2x over 500 steps: 1.18 / 1.21 / 1.22× at 128³ (the step-2x rates of job 32529712; runs of the same build differ by up to 5 % at 0.03 between jobs), 1.14× at 256³. Peak memory of the moving step at 128³: 2045 → 1990 B/cell (the bench's 48 included); M1 was reverted afterwards (CI's sanitizer build failed now and then with it, tried.md), which gives back its 31 B/cell: 2021 B/cell by the arrays' sizes. Bits: geometry, faces and lists checksums equal to step-2x on seven inputs at three spacings (105 of 105, job 32530569); every step against a fresh build at 262,144 cells, seven inputs, three spacings, six steps: 18 of 18 runs pass (job 32540506).

The screen's register cap, measured (Nsight Compute, job 32540491; 64 against 76 registers): 43.8 against 32.5 % of warps active, 6.98 against 5.19 warps per scheduler; the spills are 594 B/cell of local loads (98 % L1 hits) and 391 B/cell of local stores (to L2), and add 8 B/cell of DRAM traffic. More warps hide the gathers' latency, which is what the screen waits on (long scoreboard 68 %).

**The re-sort in place** (after the head above; asked for by luna-cpp, whose moving 256³ run ran out of the A100's memory at its first re-sort). The re-sort allocated about 330 B/cell for a moment: the cell keys and the sort's double buffers, the rank, a second copy of the positions and quantised points, the tetrahedra's keys and slots (64 bits each) and their sort, and new vertex, adjacency and edge arrays beside the old. The ring pools are free while it runs (the next update walks every ring again), so it now carves its keys, sort buffers and rank out of one pool (`mesh_order::Arena`), stages each permuted array in the other and copies it back, and sorts the tetrahedra by 32-bit keys. Only the permutation itself (4 B/cell) is kept, reused from the last re-sort. Callers follow with `permute_in_place(view)`, staged in the spare ring pool (`order_test` covers it beside `permute`).

| jittered, 0.1 spacings | 128³ | 256³ |
|---|---|---|
| peak of a run with re-sorts, B/cell (jobs 32540920, 32540922; a re-sort every 3 updates) | 2320 → 1994 | 2320 → 1994 (38.9 → 33.5 GB) |
| peak without a re-sort, B/cell | 1990 | 1990 |
| one re-sort, ms (ns/cell) (jobs 32540913, 32540918) | 10.4 → 4.8 (5.0 → 2.3) | 108.8 → 34.3 (6.5 → 2.0) |
| 500 steps, re-sort every 25, M cells/s | 170.3 → 173.7 | 220.7 → 230.0 |
| 500 steps, re-sort every 100, M cells/s (jobs 32540914, 32540919) | 137.5 → 144.3 | 199.3 → 201.1 |

At 0.3 spacings (128³) one re-sort is 11.6 → 5.4 ms, and 500 steps 102.4 → 103.9 M cells/s (every 25) and 73.4 → 73.6 (every 100). The step after a re-sort still walks every ring and builds every list; that is the rest of what luna-cpp measured as about one step. Bits: the same geometry, faces and lists checksums as before on seven inputs at three spacings with a re-sort every 2 updates (job 32540927); A100 ctest 23 of 23 (job 32540912).

**The branch's head** (`6f87d0b`: levers 1, 2 and 5a, M2, the re-sort in place; M1 reverted). Against levers 1 + 2 in one job each (jobs 32543851–53), M cells/s:

| | 128³ 0.03 / 0.1 / 0.3 | 256³ 0.03 / 0.1 / 0.3 |
|---|---|---|
| ten steps | 197.7 / 125.1 / 73.7 → 201.6 / 127.4 / 74.7 | 259.5 / 186.7 / 107.0 → 262.7 / 189.7 / 108.5 |
| 500 steps, re-sort every 25 | 214.1 / 160.5 / 98.0 → 229.7 / 169.6 / 101.2 | |

Tests on this head (jobs 32543848, 32543849, 32544828): A100 test suite 23 of 23; every step against a fresh exact build, 93 of 93 and S2's 48 of 48; geometry, faces and lists checksums equal to step-2x, 105 of 105; CI's host steps clean (comments, format, gcc 15 with sanitizers off and on 23 of 23, clang-tidy 18 without warnings), and the sanitizer suite again three times as CI runs it, 23 of 23 each. Against step-2x over 500 steps at 128³ (job 32529712): 1.19 / 1.21 / 1.20×; at 256³, 0.1 spacings, in one job (32544490): 196.3 → 228.8 M cells/s (1.17×), a re-sort 101 → 34 ms. Peak of a run with re-sorts at 128³: 2375 → 2025 B/cell (job 32543853).

## Removed, and where to find it

Scripts and files nothing maintained uses; each line names the last commit that had it (`git show <commit>:<path>`).

- `scripts/flow_table.py`, tables of `distortion_test --print` logs for research_notes/mesh_tests_2026-11-05.md: 61e9914.
- `reports/hydro_stage3_blast128.png`, `hydro_stage3_sedov128.png`, `hydro_stage3_sedov256.png`, hydro figures from before the split that nothing referenced: 61e9914.
