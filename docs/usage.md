# Using voronoi-cpp: flags, contract, bench and layout

The details behind the README's [Install and build](../README.md#install-and-build) and
[Using it](../README.md#using-it-voronoimesh): the compiler flags in use, everything a caller of `voronoi::Mesh` can
rely on, the options of `voronoi_bench`, and the layout of the repository and its CI.

## Compiler flags

- Release: `-O3 -DNDEBUG` (CMake's default), C++17 required (Kokkos sets `-std=c++20`), warnings `-Wall -Wextra -Wpedantic` on the programs.
- CUDA: `nvcc_wrapper` with `-extended-lambda -arch=sm_80`. The warp clipping kernel uses `Kokkos::LaunchBounds<128, 5>`, which caps it at 96 registers; 18 KB of shared memory per block.
- OpenMP: `-fopenmp` from Kokkos. The project sets no `-march`; `-DCMAKE_CXX_FLAGS=-march=icelake-server` is worth adding on Raven's CPUs.
- `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` (always on in `CMakeLists.txt`) writes the database clang-tidy needs: `bash scripts/lint_cpp.sh build-omp` (Raven: `module load clang/20`).
- `-DVORONOI_SANITIZERS=ON` adds address and undefined-behaviour sanitizers to a host build.
- No flag picks the floating-point arithmetic: nvcc fuses multiply-add by default (`--fmad=true`) and the project keeps that; both variants of the geometry kernels are compiled in and `Layout::same_bits_as_host` picks one at run time (below). `-march=icelake-server` lets gcc fuse on the host too, so host results then differ in the last bits from a build without it.

CPU and GPU in one build (Kokkos with CUDA and OpenMP, `scripts/kokkos_cuda_omp.sh`); `voronoi::Engine` (`voronoi_engine.hpp`) then picks cpu, gpu or hybrid at run time. Rebuild and steps run within 1 % of the CUDA-only build and the binary is 0.5 % larger ([measurements](measurements.md#cpu-and-gpu-from-one-build-2026-11-16)):

```sh
module load gcc/14 cuda/13.0 cmake
cmake -S . -B build-both -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/u/uli/kokkos-cuda-omp \
  -DCMAKE_CXX_COMPILER=/u/uli/kokkos-src/bin/nvcc_wrapper
cmake --build build-both -j4
```

On Raven, build with at most `-j4` on the login node, or inside a Slurm job, and run GPU work through Slurm (`scripts/bench_a100.sh`).

## What a caller can rely on

- Particles are the caller's indices; in 3D a re-sort (by default every 25 updates, below) renumbers them and hands the caller the permutation. Ties between cospherical particles are broken by index (symbolic perturbation), at the start and in every repair. A permuted input gives the permuted mesh, except that on exact ties faces of zero area can differ.
- Positions are quantised to integer coordinates on a 2³⁰ grid (rounded down): 2³⁰ steps across the longest side, the same step on every axis; topology and geometry both come from the quantised positions. A side that is not a whole number of steps is rounded to one (relative change below 2⁻³¹ longest / side; none for sides in power-of-two ratios such as 2 x 2 x 1/16). `sides()` gives the box used; `box()` is a cube's side.
- Every cell must be certified within one periodic image: its radius must stay below 3/16 of the shortest tile side, and below 1/8 of it where a tile holds fewer than about 180 particles (32 in 2D). The box is cut into tiles of side / round(side / shortest side) along each axis; a cube is one tile. In practice that is about 6 or more particles across the shortest side (a 5³ lattice is refused, 6³ is not; a slab of 16 cells is fine). Otherwise `update()` throws, stating this rule ([#31](https://github.com/Ulli1991/voronoi-cpp/issues/31)).
- `update()` repairs the mesh; the first call, or a change of n, builds it. If the device repair leaves stuck triangles, it repairs the cells around them on the device (local repair, `tet_local`) or rebuilds, chosen by how many entries are stuck, not by a clock (`voronoi::Fallback`); the host finish is opt-in (`Fallback::host`).
- Faces are stored once, by the lower id, so a flux applied with opposite signs conserves exactly. Cell volumes after `update()` are summed with atomics. Opt-in, 3D (`Layout::faces_on_read`): `update()` does not store face geometry; `face_geometry().face(f, area, centroid)` computes face f from its ring of circumcentres where it is read, with the bits the stored rows would hold, and `area()`, `face_centroid()` or `volume()` store them on first use. It saves the faces kernel (about 1000 B/cell) but each read gathers the ring, so it is slower wherever faces are read more than once ([tried.md](tried.md), R14). `for_each_face(visit)` runs one kernel over the faces and calls `visit(f, owner, neighbour, area, centroid)` with each face's geometry, from the stored rows or, with `faces_on_read`, computed once from the ring; with no rows stored, `update()` plus one such pass is 6 to 14 % faster and takes 394 B/cell less memory ([measurements](measurements.md#face-visitor-and-chunked-passes-meshfor_each_face-voronoi_bench-chunks-job-32375192)). `update_cells()` sums volumes and centroids over each cell's faces in a fixed order, giving the same bits every run, and builds the per-cell face lists for gathers.
- Faces are numbered 0 .. `faces()` − 1 after every update. In 3D a face keeps its number from one update to the next while no flip touches the tetrahedron that represents it; a new face takes a freed number, and an update with many flips (more than 1/16 of the tetrahedra created), a build or a re-sort numbers every face again ([#53](https://github.com/Ulli1991/voronoi-cpp/issues/53)). With `Layout::owner_faces` the numbers go by owner, as before.
- `clip(x, cells)` rebuilds the listed cells by clipping, independent of the persistent mesh, and returns each face once (a local mode for substeps of active cells). It builds exact cells.
- `rebuild(x)` builds the exact mesh from scratch. `scratch(bytes)` lends device memory between updates.
- Volumes and centroids only, without a mesh: `shell_voronoi::volumes_and_centroids(exec, box, x, volume, centroid)` (`shell_voronoi_warp.hpp`) fills `volume(i)` and `centroid(i, a)` (relative to particle i) for every particle of `x`, quantised to the box's grid as above. It is the clipping rebuild with no faces, face counts or rings recorded (`shell_voronoi::Cells::volumes_only(n)` for the arrays alone), with the same certification and fallbacks; it throws if a cell is left unbuilt. Its volumes and centroids are bit-identical to a full rebuild's, at 101 B/cell on the device instead of 671.
- **3D re-sorts by default: particle order and ids change.** Every 25 updates (`Layout::every = 25`) `Mesh<3>` renumbers the cells along a Hilbert curve, and the tetrahedra and faces after them, which keeps the memory order on long moving runs ([#34](https://github.com/Ulli1991/voronoi-cpp/issues/34); 2.0× over 500 steps of the bench's flow at 128³ and 256³, 0.1 spacings per step, [measurements](measurements.md#the-2-step-branch-step-2x-2026-11-18)). When `Report::resorted` is set, cell i is the particle the caller passed as `permutation()(i)` (also `Engine<3>::permutation()`), and the caller renumbers every per-particle array it keeps, positions too, before the next update (`permute(view)` does it into a new view; `permute_in_place(view)` without allocating, staged in the spare ring pool). The re-sort itself allocates nothing: its keys, sorts and staged arrays live in the ring pools, which the next update walks again anyway. Each particle keeps its neighbours; its geometry agrees to rounding (1e-14), not to the bit. A caller that needs fixed ids sets `every = 0`. A build keeps the caller's order unless `at_build` is set.
- `set_layout(voronoi::Layout)` (3D) sets the rest. `curve`: Hilbert (default), Morton, Moore, Gray-coded Morton, row-major, reverse Cuthill-McKee. Other triggers, off by default: `flips = f` re-sorts once the device flips since the last re-sort reach f per cell (it follows the flow, but flip counts vary from run to run, so the update it fires on can too), and `far` when that share of faces joins cells over 64 ids apart. `owner_faces` numbers the faces by owner (`owned_start()(i)` is cell i's first), `owner_bit` sets bit 31 in a list entry whose cell does not own the face (`Mesh::list_face(e)`, `Mesh::list_owned(e)`); neither changes a bit of the geometry. Together they make luna-cpp's hydro 16 % faster on moving Sedov 128³ and cost the mesh 2.3 ms a step and 168 B/cell: +1 % a whole step. `float_centroids` stores the face centroids as float (`face_centroid32()`, relative to the owner; `face_centroid()` is then empty): 44 instead of 56 B a face, about 200 B/cell less moved per step and 98 B/cell less memory at 128³; areas, volumes and topology keep their bits, cell centroids move by under 1e-8 spacings ([measurements](measurements.md#float-face-centroids-layoutfloat_centroids-job-32373869)). `same_bits_as_host` (also `set_same_bits_as_host(bool)` on `Mesh<3>` and `Mesh<2>`; in 2D set it before the first update) rounds every multiply and add of the geometry and of the clipping's per-thread kernel on its own, so a GPU gives the host's bits; off, nvcc fuses them, which is faster and differs from the host in the last bits (20 to 23 % of volumes, at most 8e-16 relative). It costs 1 to 5 % of a step (most at 0.03 spacings) and at most 1 % of a rebuild ([measurements](measurements.md#rounding-per-mode-2026-11-16-jobs-32422876-and-32429908), [TESTING.md](../TESTING.md#bits-across-backends)).
- `voronoi::Engine<D>` (`voronoi_engine.hpp`) holds a `Mesh<D>` on the host or the device, chosen at run time by `voronoi::Options{backend, tail, host_threads}`: `cpu` (`Kokkos::DefaultHostExecutionSpace`, on `host_threads` OpenMP threads), `gpu` (the device, fused arithmetic, main's bits) or `hybrid` (the device with `same_bits_as_host`, so work the host takes gives the same bits; until the host takes a share, hybrid runs on the device alone). It has `Mesh`'s calls, takes positions in any memory space and layout, and returns results as host views; `visit(f)` hands the mesh itself to code that wants the views where they are.
- The dimension is a template parameter, `voronoi::Mesh<D, Exec>` with `exact_predicates::BoxN<D>`; `Box` is `BoxN<3>` (research_notes/dimension_design_2026-11-02.md).
- `voronoi::Mesh<2>` has the contract of `Mesh<3>` on triangles (`voronoi_mesh_2d.hpp`), with two differences. `volume()` is current after `update_cells()` only. There are always exactly 3n faces, numbered for good: a flip gives its face new particles, and a face can have zero length on a lattice. A repaired mesh equals a fresh build bit for bit.

## The benchmark driver `voronoi_bench`

`voronoi_bench` (`src/bench/`) times and checks the pipeline:

```text
voronoi_bench rebuild <n> <distribution> [repeats=5] [--no-check] [--memory] [--kernels] [--box=Lx,Ly,Lz] [--nvtx | --nvtx=phases]
voronoi_bench step <n> <distribution> <displacement> [steps=10] [--verify] [--memory] [--kernels] [--checksum] [--box=Lx,Ly,Lz] [--nvtx | --nvtx=phases]
                    [--order=every] [--curve=morton|hilbert|moore|gray|row|rcm] [--every=k] [--flips=f] [--far=share] [--sort-at-build] [--no-resort]
                    [--owner-faces] [--owner-bit] [--dump-cells=path]
```

`rebuild` times the warp clipping. `--order=k` prints the memory order every k steps; `--curve`, `--every`, `--flips`, `--far`, `--sort-at-build`, `--no-resort` (every trigger off, as main before 2026-11-18), `--owner-faces` and `--owner-bit` change the default `Layout`, and the bench's points follow each re-sort; `--dump-cells` writes each particle's cell for `scripts/resort_bits.py`. `--box` (default 1,1,1) sets the periodic box; `--box=2,2,0.0625` with 4194304 jittered points is a 512 x 512 x 16 slab. Distributions: `uniform` (white noise), `jittered`, `lattice`, `wave`, `clustered`, `halo`, `<sc|bcc|fcc|shells|planes|pairs>@<perturbation in spacings>`, or a library-benchmark point file (`*.pts`, e.g. blue noise; n is the file's). `--kernels` prints the time of each kernel label, of the rebuild's clipping and of `step`'s start and steps.

## Repository layout

- `src/neighbour_search/`, `src/voronoi_construct/`: the library.
- `src/test/`: the tests and their support code; `mesh_cases_test` has its groups in `src/test/mesh_cases/` (one file per group, shared helpers in `common.hpp`). `src/bench/`: `voronoi_bench`, `mesh2d_bench`.
- Used by luna-cpp, keep: `voronoi_library`, `src/test/check.hpp`, `src/test/support.hpp` (and what it includes), `src/bench/device_memory.hpp`.
- `scripts/`: lint and A100 benchmark, test and `compute-sanitizer` (`sanitize_a100.sh <build> <memcheck|racecheck|initcheck|synccheck>`) scripts.
- `research_notes/`, `reports/`: status notes and surveys.

## CI

CI (`.github/workflows/checks.yml`) checks the comment rules (a header per file, a docstring on every multi-line library definition, comment runs of 3 lines at most) and the formatting of changed lines (`.clang-format`, `scripts/check_format.py`), runs clang-tidy (ordinary build) and the CPU tests (ordinary and sanitizer builds) against a pinned Serial Kokkos 5.0.2. It does not run GPU jobs.
