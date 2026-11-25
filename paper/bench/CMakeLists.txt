# The paper's helper programs, built against the repository's headers: cmake -S paper/bench -B <dir>
cmake_minimum_required(VERSION 3.16)
project(voronoi_paper LANGUAGES CXX)
find_package(Kokkos CONFIG REQUIRED)
set(root ${CMAKE_CURRENT_SOURCE_DIR}/../..)
add_executable(paper_trajectory paper_trajectory.cpp)
target_link_libraries(paper_trajectory PRIVATE Kokkos::kokkos)
target_compile_features(paper_trajectory PRIVATE cxx_std_17)
target_include_directories(paper_trajectory PRIVATE ${root}/src/neighbour_search ${root}/src/voronoi_construct
    ${root}/src/test)
if(Kokkos_ENABLE_HIP)  # as the library's CMakeLists.txt
    target_compile_options(paper_trajectory PRIVATE -ffp-contract=off)
endif()
