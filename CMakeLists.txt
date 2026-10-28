cmake_minimum_required(VERSION 3.16)
project(voronoi_cpp LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(VORONOI_SANITIZERS "Enable host address and undefined-behavior sanitizers" OFF)
if(VORONOI_SANITIZERS)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()
find_package(Kokkos CONFIG REQUIRED)

add_library(voronoi_library INTERFACE)
target_link_libraries(voronoi_library INTERFACE Kokkos::kokkos)
target_compile_features(voronoi_library INTERFACE cxx_std_17)
target_include_directories(voronoi_library INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/neighbour_search
    ${CMAKE_CURRENT_SOURCE_DIR}/src/voronoi_construct)


add_executable(local_voronoi_example src/test/example.cpp)
target_compile_features(local_voronoi_example PRIVATE cxx_std_17)
target_link_libraries(local_voronoi_example PRIVATE voronoi_library)
target_compile_options(local_voronoi_example PRIVATE -Wall -Wextra -Wpedantic)

add_executable(neighbor_benchmark src/test/neighbor_benchmark.cpp)
target_compile_features(neighbor_benchmark PRIVATE cxx_std_17)
target_link_libraries(neighbor_benchmark PRIVATE voronoi_library)
target_compile_options(neighbor_benchmark PRIVATE -Wall -Wextra -Wpedantic)

add_executable(mesh_benchmark src/test/mesh_benchmark.cpp)
target_compile_features(mesh_benchmark PRIVATE cxx_std_17)
target_link_libraries(mesh_benchmark PRIVATE voronoi_library)
target_compile_options(mesh_benchmark PRIVATE -Wall -Wextra -Wpedantic)

add_executable(table_mesh_check src/test/table_mesh_check.cpp)
target_compile_features(table_mesh_check PRIVATE cxx_std_17)
target_link_libraries(table_mesh_check PRIVATE voronoi_library)
target_compile_options(table_mesh_check PRIVATE -Wall -Wextra -Wpedantic)

add_executable(hydro_geometry_check src/test/hydro_geometry_check.cpp)
target_compile_features(hydro_geometry_check PRIVATE cxx_std_17)
target_link_libraries(hydro_geometry_check PRIVATE voronoi_library)
target_compile_options(hydro_geometry_check PRIVATE -Wall -Wextra -Wpedantic)

add_executable(neighbor_sort_check src/test/neighbor_sort_check.cpp)
target_compile_features(neighbor_sort_check PRIVATE cxx_std_17)
target_link_libraries(neighbor_sort_check PRIVATE voronoi_library)
target_compile_options(neighbor_sort_check PRIVATE -Wall -Wextra -Wpedantic)

add_executable(construction_stress_check src/test/construction_stress_check.cpp)
target_compile_features(construction_stress_check PRIVATE cxx_std_17)
target_link_libraries(construction_stress_check PRIVATE voronoi_library)
target_compile_options(construction_stress_check PRIVATE -Wall -Wextra -Wpedantic)

add_executable(construction_scaling src/test/construction_scaling.cpp)
target_compile_features(construction_scaling PRIVATE cxx_std_17)
target_link_libraries(construction_scaling PRIVATE voronoi_library)
target_compile_options(construction_scaling PRIVATE -Wall -Wextra -Wpedantic)

option(VORONOI_CUDA_RESOURCE_REPORT "Report NVCC registers, stack and spills for construction kernels" OFF)
if(VORONOI_CUDA_RESOURCE_REPORT)
    if(NOT Kokkos_ENABLE_CUDA)
        message(FATAL_ERROR "VORONOI_CUDA_RESOURCE_REPORT requires a CUDA Kokkos build")
    endif()
    target_compile_options(table_mesh_check PRIVATE --resource-usage)
endif()

add_executable(construction_early_stress_check src/test/construction_stress_check.cpp)
target_compile_features(construction_early_stress_check PRIVATE cxx_std_17)
target_compile_definitions(construction_early_stress_check PRIVATE VORONOI_TEST_EARLY_STOP=true)
target_link_libraries(construction_early_stress_check PRIVATE voronoi_library)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME geometry_analytic COMMAND hydro_geometry_check)
    add_test(NAME geometry_stress COMMAND construction_stress_check)
    add_test(NAME geometry_early_stress COMMAND construction_early_stress_check)
    add_test(NAME neighbor_sort COMMAND neighbor_sort_check)
    foreach(distribution lattice disordered regular random gradient)
        add_test(NAME table_${distribution} COMMAND table_mesh_check
            ${CMAKE_CURRENT_BINARY_DIR}/test-${distribution}.csv
            12 insertion cached ${distribution} 64 relative_early)
    endforeach()
    foreach(policy relative_early relative_early_coalesced relative_early_fixed32 relative_early_batched_test relative_early_shared)
        add_test(NAME scaling_${policy} COMMAND construction_scaling 12 1 ${policy})
    endforeach()
    set_tests_properties(geometry_analytic geometry_stress geometry_early_stress neighbor_sort
        PROPERTIES TIMEOUT 120)
    set_tests_properties(table_lattice table_disordered table_regular table_random table_gradient
        scaling_relative_early scaling_relative_early_coalesced scaling_relative_early_fixed32
        PROPERTIES TIMEOUT 180)
endif()

if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME python_comparison_contract COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_compare_faces.py)
    foreach(distribution lattice disordered regular random gradient)
        add_test(NAME equivalence_${distribution} COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/python/check_geometry_paths.py
            $<TARGET_FILE:table_mesh_check> ${distribution})
        set_tests_properties(equivalence_${distribution} PROPERTIES TIMEOUT 300)
    endforeach()
    foreach(periodic 0 1)
        add_test(NAME search_oracles_periodic_${periodic}
            COMMAND neighbor_benchmark 512 2.0 1 all ${periodic})
        set_tests_properties(search_oracles_periodic_${periodic} PROPERTIES TIMEOUT 120)
    endforeach()
endif()

if(BUILD_TESTING)
    add_executable(fixed_point_check src/test/fixed_point_check.cpp)
    target_link_libraries(fixed_point_check PRIVATE voronoi_library)
    add_test(NAME fixed_point_contract COMMAND fixed_point_check)
endif()

if(BUILD_TESTING)
    add_executable(cooperative_clip_check src/test/cooperative_clip_check.cpp)
    target_link_libraries(cooperative_clip_check PRIVATE voronoi_library)
    add_test(NAME cooperative_clip_contract COMMAND cooperative_clip_check)
endif()
