#pragma once

// Clipping stage, shared constants (3D): the Status codes of a cell and the capacities of its plane and vertex arrays,
// read by shell_cell.hpp, shell_voronoi.hpp and shell_voronoi_warp.hpp.
namespace shell_voronoi {

// Per-cell outcome code of the clipping walk (ok = certified and within capacity).
// Used by: shell_voronoi::construct_pass, shell_voronoi_warp build.
enum Status : int {
    ok = 0,
    uncertified = 1,  // the walk reached the coarsest grid the box allows
    plane_overflow = 2,
    vertex_overflow = 3,
    face_overflow = 4,  // the cell's chunk of the face or ring pool is full
    degenerate = 5,
};
constexpr int status_count = 6;

constexpr int max_planes = 48;                             // cutting planes, including the 6 box planes
constexpr int max_vertices = 64;                           // one 64-bit mask marks vertices to remove
constexpr int big_planes = 128, big_vertices = 256;        // for the rare cells that overflow those
constexpr int huge_planes = 16000, huge_vertices = 32000;  // exact, on the host: the few that overflow those
constexpr int ring_buffer = 32;
constexpr int box_planes = 6;
constexpr int face_block = 256;  // face pool entries a warp claims at a time (shell_voronoi_warp.hpp)

}  // namespace shell_voronoi
