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

# The finite-volume hydro on the mesh (header-only); snapshots need HDF5.
add_library(hydro_library INTERFACE)
target_link_libraries(hydro_library INTERFACE voronoi_library)
target_include_directories(hydro_library INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/hydro)
find_package(HDF5 COMPONENTS C)
if(HDF5_FOUND)
    voronoi_program(hydro_sedov src/bench/hydro_sedov.cpp)
    target_link_libraries(hydro_sedov PRIVATE hydro_library HDF5::HDF5)
else()
    message(STATUS "HDF5 not found: hydro_sedov is not built")
endif()

include(CTest)
if(BUILD_TESTING)
    foreach(test exact_predicates clipping mesh start repair voronoi_mesh hydro)
        voronoi_program(${test}_test src/test/${test}_test.cpp)
        add_test(NAME ${test} COMMAND ${test}_test)
        set_tests_properties(${test} PROPERTIES TIMEOUT 600)
    endforeach()
    target_link_libraries(hydro_test PRIVATE hydro_library)
endif()
