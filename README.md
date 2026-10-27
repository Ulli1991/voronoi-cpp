# voronoi-cpp

A small Kokkos prototype for reconstructing local 3D Voronoi cells from particle positions and a candidate-neighbor graph. The goal is to test the geometric reconstruction independently before connecting it to a meshless finite-mass or finite-volume solver.

## What is here

`local_voronoi.hpp` clips a temporary cube by the perpendicular-bisector planes associated with each candidate neighbor. For each cell it returns face areas, neighbor IDs, and face vertices. It uses fixed-capacity storage so a device kernel can report overflow or an uncertified cell rather than silently truncating the geometry. A successful row is certified against the supplied search radius. The caller must provide every particle close enough to affect that cell.

`local_voronoi_example` makes a reproducible 128-particle periodic example, uses an all-to-all candidate graph, checks cell closure and pairwise face consistency, and writes CSV files. `plot_mesh.py` renders those files. This is a correctness example, not a scalable neighbor search: the all-to-all graph is only for a tiny demonstration. The reconstruction currently uses double precision; periodic image handling, degenerate configurations, and capacity limits need further validation.

`neighbor_benchmark` compares a compact sorted-cell radius search with a median-split BVH reference. It checks each candidate graph against an all-pairs oracle on uniform, jittered-lattice, clustered, and density-gradient point sets. This is a radius-search microbenchmark; it does not yet compare GPU performance or integrate neighbor search with Voronoi clipping.

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
```

The example writes `particles_128.csv` and `faces_128.csv` to the current directory. To plot them, install Matplotlib and run `python3 plot_mesh.py` from the repository root.

For GPU experiments, Kokkos must be built with the relevant CUDA or HIP backend and a compatible compiler/toolchain. The current prototype has not yet been benchmarked on a GPU. No GPU speedup claims are made here.

## Current scope and limitations

This is early research code. The local geometry routine has been exercised on a small synthetic periodic set and the search prototype has been checked against an all-pairs oracle on synthetic distributions. These checks do not establish robustness for all particle configurations, nor do they validate conservation or stability in a hydrodynamics solver. There is no Cabana dependency yet; the caller supplies the CSR neighbor graph.
