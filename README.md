# voronoi-cpp

A small Kokkos prototype for reconstructing local 3D Voronoi cells from particle positions and a candidate-neighbor graph. The goal is to test the geometric reconstruction independently before connecting it to a meshless finite-mass or finite-volume solver.

## What is here

Candidate order can affect temporary workspace use. The mesh stress test orders candidates by distance from the cell center, which avoids transient-capacity failures in the tested lattice; a production neighbor-list path should preserve or provide this ordering, or increase and validate the workspace capacities.

`local_voronoi.hpp` clips a temporary cube by the perpendicular-bisector planes associated with each candidate neighbor. For each cell it returns face areas, neighbor IDs, and face vertices. It uses fixed-capacity storage so a device kernel can report overflow or an uncertified cell rather than silently truncating the geometry. A successful row is certified against the supplied search radius. The caller must provide every particle close enough to affect that cell.

`local_voronoi_example` makes a reproducible 128-particle periodic example, uses an all-to-all candidate graph, checks cell closure and pairwise face consistency, and writes CSV files. `plot_mesh.py` renders those files. This is a correctness example, not a scalable neighbor search: the all-to-all graph is only for a tiny demonstration. The reconstruction currently uses double precision; periodic image handling, degenerate configurations, and capacity limits need further validation.

`neighbor_search.hpp` is the radius search intended for production use. It sorts particles into a dense cell grid with a deterministic counting sort, copies them into cell order, and writes each particle's neighbors into a fixed-capacity table with a per-row count. Buffers persist across rebuilds and are reused without zeroing, so a rebuild allocates nothing and needs one host read, the overflow check; on overflow, `rebuild` grows the table and repeats. It supports periodic boxes (radii below half the box), per-particle radii, optional nearest-first row ordering, and, for small particle counts on a GPU, several vector lanes per particle. The rebuild can be captured as one Kokkos graph.

`neighbor_benchmark` times that search and compares it with a sorted-cell search over occupied cells and a median-split BVH whose build is host-side. It uses uniform, jittered-lattice, clustered, and density-gradient point sets whose radii scale with n^(-1/3), so the mean neighbor count is independent of n, and it reports the median of repeated runs after a warm-up. Results are checked against an all-pairs oracle up to 32,768 particles and against the BVH above that. Usage: `neighbor_benchmark [n] [radius_scale] [repeats] [all|dense|timing] [periodic]`; `timing` skips all checks.

`mesh_benchmark` exercises periodic neighbor construction and Voronoi geometry together on a jittered lattice. It checks the candidate graph against an all-pairs periodic search for up to 8,192 particles, then runs the geometry kernel and checks certification, area closure, shared-face consistency, and volume conservation: the cell volumes must sum to the box volume to within 1e-12, and every cell volume must be positive. Its neighbor graph is generated on the host from a structured cell stencil, so it is a geometry stress test rather than a GPU neighbor-search benchmark.

## Build

Build Kokkos with the backend you want to use, then point CMake at that installation. For example, for an existing Kokkos installation:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/kokkos
cmake --build build -j
```

Run the small geometry example and neighbor-search benchmark:

```sh
./build/local_voronoi_example
./build/neighbor_benchmark 8192
./build/mesh_benchmark 16
```

The example writes `particles_128.csv` and `faces_128.csv` to the current directory. To plot them, install Matplotlib and run `python3 plot_mesh.py` from the repository root.

For GPU experiments, first build or load Kokkos with the relevant CUDA or HIP backend and a compatible compiler/toolchain. Configure this project against that Kokkos installation, then confirm the CMake output lists `CUDA` or `HIP` under enabled Kokkos devices. `neighbor_benchmark` exercises cell search and CSR construction on the selected execution space; `mesh_benchmark` exercises cell reconstruction there. A useful first GPU run is:

```sh
./build/neighbor_benchmark 4096
./build/mesh_benchmark 16
```

The mesh benchmark allocates fixed-capacity face output arrays, about 100 MB at 4,096 particles.

### Measured on Raven

On one NVIDIA A100-SXM4-40GB (Kokkos 5.0.99, CUDA 13.0, GCC 14.1) and one node with two Intel Xeon Platinum 8360Y processors (72 cores, OpenMP, GCC 15.1), both benchmarks pass all of their checks. The mesh benchmark certifies every cell at 1,728, 4,096, and 32,768 particles, and the total volume matches the box to within 3e-15. Neighbor-search rows match the reference exactly for every distribution, variant, periodic and non-periodic box, and lane count tested.

Median time for a full neighbor-search rebuild, uniform points, non-periodic box:

| particles | A100, ~30 neighbors | 72 cores, ~30 neighbors | A100, ~60 neighbors | 72 cores, ~60 neighbors |
|---|---|---|---|---|
| 512 | 0.062 ms | 0.145 ms | 0.070 ms | 0.110 ms |
| 8,192 | 0.120 ms | 0.54 ms | 0.136 ms | 0.91 ms |
| 32,768 | 0.145 ms | 2.3 ms | 0.195 ms | 3.4 ms |
| 1,048,576 | 2.39 ms | 52 ms | 3.68 ms | 74 ms |

- **Periodic boxes** cost 15–20% more on the GPU.
- **Clustered points** are 9–17× faster on the GPU from 32k particles up.
- **Nearest-first row ordering** is slow as a separate pass; at 1M particles it costs several times the search itself, so it should move into the consumer of the rows.
- **Cabana and ArborX** neighbor lists were slower than this search in every case tested.

## Current scope and limitations

This is early research code. The local geometry routine has been exercised on a small synthetic periodic set and the search prototype has been checked against an all-pairs oracle on synthetic distributions. These checks do not establish robustness for all particle configurations, nor do they validate conservation or stability in a hydrodynamics solver. There is no Cabana dependency. The mesh benchmark still builds its candidate graph on the host, so the GPU search does not yet feed the mesh reconstruction.
