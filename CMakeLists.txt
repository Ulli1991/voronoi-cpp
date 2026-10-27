cmake_minimum_required(VERSION 3.16)
project(voronoi_cpp LANGUAGES CXX)

find_package(Kokkos CONFIG REQUIRED)

add_executable(local_voronoi_example example.cpp)
target_compile_features(local_voronoi_example PRIVATE cxx_std_17)
target_link_libraries(local_voronoi_example PRIVATE Kokkos::kokkos)
target_compile_options(local_voronoi_example PRIVATE -Wall -Wextra -Wpedantic)

add_executable(neighbor_benchmark neighbor_benchmark.cpp)
target_compile_features(neighbor_benchmark PRIVATE cxx_std_17)
target_link_libraries(neighbor_benchmark PRIVATE Kokkos::kokkos)
target_compile_options(neighbor_benchmark PRIVATE -Wall -Wextra -Wpedantic)

add_executable(mesh_benchmark mesh_benchmark.cpp)
target_compile_features(mesh_benchmark PRIVATE cxx_std_17)
target_link_libraries(mesh_benchmark PRIVATE Kokkos::kokkos)
target_compile_options(mesh_benchmark PRIVATE -Wall -Wextra -Wpedantic)
