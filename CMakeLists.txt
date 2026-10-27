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

add_executable(table_mesh_check table_mesh_check.cpp)
target_compile_features(table_mesh_check PRIVATE cxx_std_17)
target_link_libraries(table_mesh_check PRIVATE Kokkos::kokkos)
target_compile_options(table_mesh_check PRIVATE -Wall -Wextra -Wpedantic)

add_executable(hydro_geometry_check hydro_geometry_check.cpp)
target_compile_features(hydro_geometry_check PRIVATE cxx_std_17)
target_link_libraries(hydro_geometry_check PRIVATE Kokkos::kokkos)
target_compile_options(hydro_geometry_check PRIVATE -Wall -Wextra -Wpedantic)

add_executable(neighbor_sort_check neighbor_sort_check.cpp)
target_compile_features(neighbor_sort_check PRIVATE cxx_std_17)
target_link_libraries(neighbor_sort_check PRIVATE Kokkos::kokkos)
target_compile_options(neighbor_sort_check PRIVATE -Wall -Wextra -Wpedantic)

add_executable(construction_stress_check construction_stress_check.cpp)
target_compile_features(construction_stress_check PRIVATE cxx_std_17)
target_link_libraries(construction_stress_check PRIVATE Kokkos::kokkos)
target_compile_options(construction_stress_check PRIVATE -Wall -Wextra -Wpedantic)

add_executable(construction_scaling construction_scaling.cpp)
target_compile_features(construction_scaling PRIVATE cxx_std_17)
target_link_libraries(construction_scaling PRIVATE Kokkos::kokkos)
target_compile_options(construction_scaling PRIVATE -Wall -Wextra -Wpedantic)

option(VORONOI_CUDA_RESOURCE_REPORT "Report NVCC registers, stack and spills for construction kernels" OFF)
if(VORONOI_CUDA_RESOURCE_REPORT)
    if(NOT Kokkos_ENABLE_CUDA)
        message(FATAL_ERROR "VORONOI_CUDA_RESOURCE_REPORT requires a CUDA Kokkos build")
    endif()
    target_compile_options(table_mesh_check PRIVATE --resource-usage)
endif()
