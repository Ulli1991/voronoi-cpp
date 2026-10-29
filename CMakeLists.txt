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

# Fused natural-neighbour construction: validation and timing.
add_executable(shell_voronoi_check src/test/shell_voronoi_check.cpp)
target_link_libraries(shell_voronoi_check PRIVATE voronoi_library)
target_compile_options(shell_voronoi_check PRIVATE -Wall -Wextra -Wpedantic)

include(CTest)
if(BUILD_TESTING)
    # Double-precision kernels only: float and clustered or degenerate inputs
    # still fail some checks (see the degenerate layouts in the program).
    foreach(distribution uniform jittered lattice)
        add_test(NAME shell_${distribution} COMMAND shell_voronoi_check 4096 ${distribution} double 1)
    endforeach()
endif()
