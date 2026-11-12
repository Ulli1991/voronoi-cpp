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

#if defined(VORONOI_MEASURE)
namespace measure {
// Measurement build only: the columns of Cells::measure, per cell of the warp kernel.
enum : int {
    tested,           // particles whose distance the walk computed
    taken,            // of those, within 2R: buffered
    clipped,          // clip calls
    cuts,             // clips that cut: planes kept
    vertices_end,     // live vertices of the finished cell
    vertices_peak,    // most live vertices after any clip
    walk_cells,       // stencil grid cells the walk checked
    walk_open,        // of those, within 2R and not empty
    tested_outside,   // tested particles outside every vertex ball when tested
    taken_outside,    // taken particles outside every vertex ball
    open_outside,     // opened grid cells meeting no vertex ball
    in_open_outside,  // particles in those cells
    warp_status,      // status before the thread kernel took overflow cells
    cycles,           // clock64 cycles of the warp, lane 0
    batches,          // batches of 32 stencil entries
    rounds,           // particle rounds of the walk (one particle per lane each)
    open_p1,          // opened grid cells meeting a vertex ball (P1, with the skip's margin)
    in_open_p1,       // particles in those cells
    rounds_p1,        // rounds if only those cells were read
    open_r8,          // opened grid cells meeting the bounding box of the vertex balls (R8)
    in_open_r8,       // particles in those cells
    rounds_r8,        // rounds if only those cells were read
    clips_p1,         // clips whose particle lies in a vertex ball of the cell at its turn
    clips_r8,         // clips the bounding box of the cell at their turn does not rule out
    count
};
}  // namespace measure
#endif

}  // namespace shell_voronoi
