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
    foreach(policy relative_early relative_early_coalesced relative_early_fixed32 relative_early_batched_test relative_early_shared cooperative)
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

if(BUILD_TESTING)
    foreach(distribution lattice disordered regular random gradient)
        add_test(NAME team_${distribution} COMMAND table_mesh_check
            ${CMAKE_CURRENT_BINARY_DIR}/test-team-${distribution}.csv
            12 insertion cached ${distribution} 64 cooperative)
        set_tests_properties(team_${distribution} PROPERTIES TIMEOUT 180)
    endforeach()
endif()

if(BUILD_TESTING)
    add_executable(construction_team_stress_check src/test/construction_stress_check.cpp)
    target_link_libraries(construction_team_stress_check PRIVATE voronoi_library)
    target_compile_definitions(construction_team_stress_check PRIVATE VORONOI_TEST_TEAM=true)
    add_test(NAME geometry_team_stress COMMAND construction_team_stress_check)
    set_tests_properties(geometry_team_stress PROPERTIES TIMEOUT 180)
endif()

if(BUILD_TESTING)
    add_test(NAME scaling_scratch_sort COMMAND construction_scaling 12 1 relative_early_scratch_sort)
    foreach(distribution lattice disordered regular random gradient)
        add_test(NAME scratch_sort_${distribution} COMMAND table_mesh_check
            ${CMAKE_CURRENT_BINARY_DIR}/test-scratch-sort-${distribution}.csv
            12 team_bitonic_recompute cached ${distribution} 64 relative_early)
        set_tests_properties(scratch_sort_${distribution} PROPERTIES TIMEOUT 180)
    endforeach()
endif()

if(BUILD_TESTING)
    add_executable(scratch_sort_check src/test/scratch_sort_check.cpp)
    target_link_libraries(scratch_sort_check PRIVATE voronoi_library)
    add_test(NAME scratch_sort_contract COMMAND scratch_sort_check)
    set_tests_properties(scratch_sort_contract PROPERTIES TIMEOUT 120)
endif()

if(BUILD_TESTING)
    add_test(NAME search_stage_profile_contract COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/python/check_search_stage_profile.py
        $<TARGET_FILE:construction_scaling>)
    set_tests_properties(search_stage_profile_contract PROPERTIES TIMEOUT 180)
endif()

if(BUILD_TESTING)
    add_executable(layout_search_check src/test/layout_search_check.cpp)
    target_link_libraries(layout_search_check PRIVATE voronoi_library)
    add_test(NAME layout_search_contract COMMAND layout_search_check)
    set_tests_properties(layout_search_contract PROPERTIES TIMEOUT 120)
    foreach(policy relative_early_unsorted relative_early_morton relative_early_morton_unsorted)
        add_test(NAME scaling_${policy} COMMAND construction_scaling 12 1 ${policy})
    endforeach()
    foreach(distribution lattice disordered regular random gradient)
        add_test(NAME layout_geometry_${distribution} COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/python/check_layout_geometry.py
            $<TARGET_FILE:table_mesh_check> ${distribution})
        set_tests_properties(layout_geometry_${distribution} PROPERTIES TIMEOUT 180)
    endforeach()
endif()

add_executable(reconstruction_profile src/test/construction_scaling.cpp)
target_link_libraries(reconstruction_profile PRIVATE voronoi_library)
target_compile_definitions(reconstruction_profile PRIVATE VORONOI_REFERENCE_PROFILE=1)
if(Kokkos_ENABLE_CUDA)
    target_compile_options(reconstruction_profile PRIVATE -lineinfo)
endif()
if(BUILD_TESTING)
    add_test(NAME reconstruction_profile_smoke COMMAND reconstruction_profile 12 1 relative_early_scratch_sort)
    set_tests_properties(reconstruction_profile_smoke PROPERTIES TIMEOUT 180)
endif()

if(BUILD_TESTING)
    add_test(NAME ncu_capture_contract COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/python/test_ncu_capture.py)
endif()

add_test(NAME float_cache_scaling_smoke COMMAND construction_scaling 12 1 relative_early_float_cache)

add_test(NAME face_links_scaling_smoke COMMAND construction_scaling 12 1 relative_early_face_links)

add_executable(construction_face_links_stress_check src/test/construction_stress_check.cpp)
target_link_libraries(construction_face_links_stress_check PRIVATE voronoi_library)
target_compile_definitions(construction_face_links_stress_check PRIVATE
    VORONOI_TEST_EARLY_STOP=true VORONOI_TEST_FACE_LINKS=true)
add_test(NAME geometry_face_links_stress COMMAND construction_face_links_stress_check)

add_executable(integer_positions_check src/test/integer_positions_check.cpp)
target_link_libraries(integer_positions_check PRIVATE voronoi_library)
add_test(NAME integer_positions_geometry COMMAND integer_positions_check)
add_test(NAME integer_positions_scaling_smoke COMMAND construction_scaling 12 1 relative_early_integer32)

add_test(NAME integer_positions_decoded_smoke COMMAND construction_scaling 12 1 relative_early_integer32_decoded)

# Fused natural-neighbour construction: validation and timing.
add_executable(shell_voronoi_check src/test/shell_voronoi_check.cpp)
target_link_libraries(shell_voronoi_check PRIVATE voronoi_library)
target_compile_options(shell_voronoi_check PRIVATE -Wall -Wextra -Wpedantic)
# Double-precision kernels only: float and clustered or degenerate inputs
# still fail some checks (see the degenerate layouts in the program).
foreach(distribution uniform jittered lattice)
    add_test(NAME shell_${distribution} COMMAND shell_voronoi_check 4096 ${distribution} double 1)
endforeach()
