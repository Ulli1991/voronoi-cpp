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

function(voronoi_program name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE voronoi_library)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/test)
    target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
endfunction()

voronoi_program(voronoi_bench src/bench/voronoi_bench.cpp)
voronoi_program(mesh2d_bench src/bench/mesh2d_bench.cpp)
voronoi_program(library_bench src/bench/library_bench.cpp)
voronoi_program(geometry_derivative_bench src/bench/geometry_derivative_bench.cpp)
# CPU and GPU in one build (Kokkos with OpenMP and CUDA): the probe of docs/ideas.md, "CPU and GPU together".
if(Kokkos_ENABLE_OPENMP AND Kokkos_ENABLE_CUDA)
    voronoi_program(backends_probe src/bench/backends_probe.cpp)
endif()

include(CTest)
if(BUILD_TESTING)
    foreach(test exact_predicates clipping mesh start repair voronoi_mesh order mesh1d mesh2d distortion metamorphic modes geometry_derivative)
        voronoi_program(${test}_test src/test/${test}_test.cpp)
        add_test(NAME ${test} COMMAND ${test}_test)
        set_tests_properties(${test} PROPERTIES TIMEOUT 600)
    endforeach()
    if(VORONOI_SANITIZERS)
        set_tests_properties(mesh2d PROPERTIES TIMEOUT 3000)  # its brute-force oracles, about 11x slower
    endif()
    # Optional: the 2D geometry derivatives against luna-cpp's test-only reference (not a ctest; TESTING.md).
    set(VORONOI_LUNA_DIR "" CACHE PATH "A luna-cpp checkout with src/test/reference_geometry_2d.hpp")
    if(VORONOI_LUNA_DIR)
        voronoi_program(geometry_derivative_luna src/test/geometry_derivative_luna.cpp)
        # luna-cpp's interface (src/hydro/mesh_derivative.hpp, which includes the library's header), searched after ours.
        target_include_directories(geometry_derivative_luna AFTER PRIVATE ${VORONOI_LUNA_DIR}/src/hydro)
        target_compile_definitions(geometry_derivative_luna PRIVATE
            LUNA_GEOMETRY_DERIVATIVE="${VORONOI_LUNA_DIR}/src/hydro/mesh_derivative.hpp"
            LUNA_REFERENCE_2D="${VORONOI_LUNA_DIR}/src/test/reference_geometry_2d.hpp")
    endif()
    voronoi_program(mesh_cases_test src/test/mesh_cases_test.cpp)
    file(GLOB mesh_cases_groups CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/test/mesh_cases/*.cpp)
    target_sources(mesh_cases_test PRIVATE ${mesh_cases_groups})
    voronoi_program(mesh_fuzz_test src/test/mesh_fuzz_test.cpp)
    add_test(NAME mesh_fuzz COMMAND mesh_fuzz_test smoke --cases=${CMAKE_CURRENT_SOURCE_DIR}/src/test/fuzz_cases)
    set_tests_properties(mesh_fuzz PROPERTIES TIMEOUT 600)
    foreach(group coincident few contrast box degenerate valence large fallback predicates repro)
        add_test(NAME mesh_cases.${group} COMMAND mesh_cases_test ${group}.)
        set_tests_properties(mesh_cases.${group} PROPERTIES TIMEOUT 600)
    endforeach()
    if(VORONOI_SANITIZERS)
        set_tests_properties(mesh_cases.large PROPERTIES TIMEOUT 3000)  # its 1000-step long run, about 11x slower
    endif()
endif()
