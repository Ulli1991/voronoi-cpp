# voronoi-cpp

A small Kokkos prototype for reconstructing local 3D Voronoi cells from particle positions and a candidate-neighbor graph. The goal is to test the geometric reconstruction independently before connecting it to a meshless finite-mass or finite-volume solver.

## What is here

Candidate order can affect temporary workspace use. The mesh stress test orders candidates by distance from the cell center, which avoids transient-capacity failures in the tested lattice; a production neighbor-list path should preserve or provide this ordering, or increase and validate the workspace capacities.

`local_voronoi.hpp` clips a temporary cube by the perpendicular-bisector planes associated with each candidate neighbor. For each cell it returns face areas, neighbor IDs, and face vertices. It uses fixed-capacity storage so a device kernel can report overflow or an uncertified cell rather than silently truncating the geometry. A successful row is certified against the supplied search radius. The caller must provide every particle close enough to affect that cell.

`local_voronoi_example` makes a reproducible 128-particle periodic example, uses an all-to-all candidate graph, checks cell closure and pairwise face consistency, and writes CSV files. `plot_mesh.py` renders those files. This is a correctness example, not a scalable neighbor search: the all-to-all graph is only for a tiny demonstration. The reconstruction currently uses double precision; periodic image handling, degenerate configurations, and capacity limits need further validation.

`neighbor_benchmark` compares a compact sorted-cell radius search with a median-split BVH reference. It checks each candidate graph against an all-pairs oracle on uniform, jittered-lattice, clustered, and density-gradient point sets. It exercises Kokkos cell-list queries and CSR construction on the selected execution space, although the BVH build is host-side.

`mesh_benchmark` exercises periodic neighbor construction and Voronoi geometry together on a jittered lattice. It checks the candidate graph against an all-pairs periodic search for up to 8,192 particles, then runs the geometry kernel and checks certification, area closure, and shared-face consistency. Its neighbor graph is generated on the host from a structured cell stencil, so it is a geometry stress test rather than a GPU neighbor-search benchmark.

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

For GPU experiments, Kokkos must be built with the relevant CUDA or HIP backend and a compatible compiler/toolchain. `neighbor_benchmark` exercises cell search and CSR construction on that execution space; `mesh_benchmark` exercises cell reconstruction there. Neither has yet been benchmarked on a GPU, and no GPU speedup claims are made here.

## Current scope and limitations

This is early research code. The local geometry routine has been exercised on a small synthetic periodic set and the search prototype has been checked against an all-pairs oracle on synthetic distributions. These checks do not establish robustness for all particle configurations, nor do they validate conservation or stability in a hydrodynamics solver. There is no Cabana dependency yet; the caller supplies the CSR neighbor graph.
