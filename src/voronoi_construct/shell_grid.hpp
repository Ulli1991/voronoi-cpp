#pragma once

#include "neighbor_search.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

// Clipping stage, the grid and the walk (3D): the vectors, Grid, Stencil and the per-cell stencil walk helpers that
// shell_voronoi.hpp (construct_pass) and shell_voronoi_warp.hpp (warp kernel) share.
namespace shell_voronoi {

// Double-precision 3D vector of the rounded cell geometry and the face arrays.
// Used by: RoundedCell, Cells, detail::construct_pass.
struct Vec3 {
    double x, y, z;
};
KOKKOS_INLINE_FUNCTION Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
KOKKOS_INLINE_FUNCTION Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
KOKKOS_INLINE_FUNCTION Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
KOKKOS_INLINE_FUNCTION double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
KOKKOS_INLINE_FUNCTION Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
// The 2D vector of the face and cell arrays (tet_mesh::cells in 2D).
struct Vec2 {
    double x, y;
};
KOKKOS_INLINE_FUNCTION Vec2 operator+(Vec2 a, Vec2 b) { return {a.x + b.x, a.y + b.y}; }
KOKKOS_INLINE_FUNCTION Vec2 operator-(Vec2 a, Vec2 b) { return {a.x - b.x, a.y - b.y}; }
KOKKOS_INLINE_FUNCTION Vec2 operator*(Vec2 a, double s) { return {a.x * s, a.y * s}; }
KOKKOS_INLINE_FUNCTION double dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
template <int D>
using VecN = std::conditional_t<D == 2, Vec2, Vec3>;

// Read-only inputs, all in slot order, from a built neighbor_search::CellList.
// Used by: every walk below (construct_pass, warp kernel).
template <class MemorySpace>
struct Grid {
    Kokkos::View<double* [3], MemorySpace> positions;
    Kokkos::View<int*, MemorySpace> order;  // the particle in each slot
    Kokkos::View<int*, MemorySpace> cell_offsets;  // base cells in Morton order
    Kokkos::View<std::uint32_t*, MemorySpace> fine;
    Kokkos::View<std::uint8_t*, MemorySpace> level;
    int n = 0;
    int levels = 2;  // base grid: 2^levels cells per tile side
    exact_predicates::Box box;
    int tiles[3] = {1, 1, 1};  // tiles per axis (neighbor_search::CellList)
    double width[3] = {1, 1, 1};  // tile widths, box.side[a] / tiles[a]
    double narrowest = 1;  // the smallest width
    int fewest = 1;        // the fewest tiles along an axis

    // One tile, a cube: the walk can take the cube's shorter arithmetic.
    bool cube() const { return tiles[0] * tiles[1] * tiles[2] == 1 && box.cubic(); }

    // The first base cell, in cell_offsets, of grid cell (x, y, z) at
    // `level` <= levels (tiles[a] 2^level cells along axis a).
    template <bool cubic = false>
    KOKKOS_INLINE_FUNCTION int cell(int level, int x, int y, int z) const {
        if constexpr (cubic) return static_cast<int>(neighbor_search::morton_short(x, y, z)) << (3 * (levels - level));
        const int mask = (1 << level) - 1;
        const int tile = ((x >> level) * tiles[1] + (y >> level)) * tiles[2] + (z >> level);
        return tile << (3 * levels) |
               static_cast<int>(neighbor_search::morton_short(x & mask, y & mask, z & mask)) << (3 * (levels - level));
    }

    // The slots [begin, end) of grid cell (x, y, z) at `level` (2^level
    // cells per tile side): a block of base cells, or part of one base cell.
    template <bool cubic = false>
    KOKKOS_INLINE_FUNCTION void range(int level, int x, int y, int z, int& begin, int& end) const {
        const int below = level - levels;
        if (below <= 0) {
            const int span = -3 * below;
            const int first = cell<cubic>(level, x, y, z);
            begin = cell_offsets(first);
            end = cell_offsets(first + (1 << span));
            return;
        }
        const int base = cell<cubic>(levels, x >> below, y >> below, z >> below);
        begin = cell_offsets(base);
        end = cell_offsets(base + 1);
        if (begin == end) return;
        const int mask = (1 << below) - 1, drop = 3 * (neighbor_search::fine_levels - below);
        const std::uint32_t key = neighbor_search::morton_short(x & mask, y & mask, z & mask) << drop;
        begin = neighbor_search::lower_bound(fine, begin, end, key);
        end = neighbor_search::lower_bound(fine, begin, end, key + (std::uint32_t(1) << drop));
    }
};

// Views the cell list's sorted positions and cell offsets as a Grid.
// Used by: Mesh<3>::update (start), Mesh<3>::clip, tet_start::build.
template <class ExecSpace>
Grid<typename ExecSpace::memory_space> make_grid(const neighbor_search::CellList<ExecSpace>& list) {
    Grid<typename ExecSpace::memory_space> g{list.positions, list.order, list.cell_offsets, list.fine, list.level,
                                             list.n, list.levels, list.box};
    g.narrowest = list.box.side[0];
    g.fewest = list.tiles[0];
    for (int a = 0; a < 3; ++a) {
        g.tiles[a] = list.tiles[a];
        g.width[a] = list.box.side[a] / list.tiles[a];
        g.narrowest = std::min(g.narrowest, g.width[a]);
        g.fewest = std::min(g.fewest, g.tiles[a]);
    }
    return g;
}

// The walk goes over grid cells on a level whose cells hold about one particle; if the stencil ends uncertified it
// goes one level coarser, skipping covered cells. Stencil: offsets (4 bits per axis, biased by 8) sorted by squared
// smallest distance in widths from the home cell; ends(r) counts those closer than r.
constexpr int max_reach = 4;
template <class MemorySpace>
struct Stencil {
    Kokkos::View<int*, MemorySpace> offsets;
    Kokkos::View<float*, MemorySpace> bound2;
    Kokkos::View<int*, MemorySpace> ends;
};

// Builds the sorted stencil table (see above).
// Used by: construct, construct_huge, construct_warp, tet_start::build.
template <class ExecSpace>
Stencil<typename ExecSpace::memory_space> make_stencil(const ExecSpace& exec) {
    std::vector<std::pair<int, int>> entries;
    auto beyond = [](int d) { return std::max(std::abs(d) - 1, 0); };
    for (int dx = -max_reach; dx <= max_reach; ++dx)
        for (int dy = -max_reach; dy <= max_reach; ++dy)
            for (int dz = -max_reach; dz <= max_reach; ++dz) {
                const int gaps = beyond(dx) * beyond(dx) + beyond(dy) * beyond(dy) + beyond(dz) * beyond(dz);
                if (gaps < max_reach * max_reach) entries.push_back({gaps, (dx + 8) | (dy + 8) << 4 | (dz + 8) << 8});
            }
    std::sort(entries.begin(), entries.end());
    using Memory = typename ExecSpace::memory_space;
    Stencil<Memory> s{Kokkos::View<int*, Memory>("shell_stencil", entries.size()),
                      Kokkos::View<float*, Memory>("shell_stencil_bound2", entries.size()),
                      Kokkos::View<int*, Memory>("shell_stencil_ends", max_reach + 1)};
    auto offsets = Kokkos::create_mirror_view(s.offsets);
    auto bound2 = Kokkos::create_mirror_view(s.bound2);
    auto ends = Kokkos::create_mirror_view(s.ends);
    Kokkos::deep_copy(ends, 0);
    for (std::size_t i = 0; i < entries.size(); ++i) {
        offsets(i) = entries[i].second;
        // Round down so that scaling never overstates a distance.
        bound2(i) = std::nextafter(float(entries[i].first), 0.0f);
        for (int r = 0; r <= max_reach; ++r)
            if (entries[i].first < r * r) ends(r) = static_cast<int>(i) + 1;
    }
    Kokkos::deep_copy(exec, s.offsets, offsets);
    Kokkos::deep_copy(exec, s.bound2, bound2);
    Kokkos::deep_copy(exec, s.ends, ends);
    return s;
}

// Cells closer than this many widths are walked at `level`: max_reach, or less where 2 reach + 1 cells
// along the shortest axis (`fewest` tiles) would visit a cell twice. Certification therefore holds within one
// periodic image only: 2R must stay below 3/8 of the shortest side, else the cell is uncertified.
// Used by: the walk in construct_pass and its warp twin.
KOKKOS_INLINE_FUNCTION int level_reach(int level, int fewest = 1) {
    return Kokkos::min(int{max_reach}, ((fewest << level) - 1) / 2);
}

KOKKOS_INLINE_FUNCTION int beyond2(int d) {
    const int b = Kokkos::max(Kokkos::abs(d) - 1, 0);
    return b * b;
}

// Periodic image (-1, 0, 1 per axis) packed into 6 bits.
KOKKOS_INLINE_FUNCTION int pack_image(int x, int y, int z) { return (x + 1) | (y + 1) << 2 | (z + 1) << 4; }
KOKKOS_INLINE_FUNCTION int image(int packed, int axis) { return ((packed >> (2 * axis)) & 3) - 1; }

// The home cell of generator p at `level` (2^level cells per tile side).
// cubic (Grid::cube(), known at compile time): one tile, equal sides.
// Used by: construct_pass, warp kernel.
template <bool cubic = false, class MemorySpace>
KOKKOS_INLINE_FUNCTION void home_cell(const Vec3& p, const Grid<MemorySpace>& grid, int level, int (&home)[3]) {
    const double* side = grid.box.side;
    home[0] = neighbor_search::grid_coordinate(p.x, side[0], cubic ? 1 << level : grid.tiles[0] << level);
    home[1] = neighbor_search::grid_coordinate(p.y, side[cubic ? 0 : 1], cubic ? 1 << level : grid.tiles[1] << level);
    home[2] = neighbor_search::grid_coordinate(p.z, side[cubic ? 0 : 2], cubic ? 1 << level : grid.tiles[2] << level);
}

// The grid cell at stencil offset `code` of the walk at `level` around
// generator p in cell `home`: false if it lies beyond 2R (reach2 = 4R^2) or
// inside the finished walk one level finer, whose home cell is 2 home + bit
// a of `finer` along axis a (finer < 0: none); else its slots and the
// periodic image they are seen in.
// Used by: the walk in construct_pass and the warp kernel.
template <bool adaptive = true, bool cubic = false, class MemorySpace>
KOKKOS_INLINE_FUNCTION bool stencil_cell(const Grid<MemorySpace>& grid, int level, const int (&home)[3], int finer,
                                         int code, const Vec3& p, double reach2, int& begin, int& end,
                                         int& packed) {
    const double fraction = 1.0 / (1 << level);  // of a tile
    const int u[3] = {home[0] + (code & 15) - 8, home[1] + ((code >> 4) & 15) - 8, home[2] + ((code >> 8) & 15) - 8};
    const double q[3] = {p.x, p.y, p.z};
    double gap2 = 0;
    for (int a = 0; a < 3; ++a) {
        const double width = cubic ? grid.width[0] / (1 << level) : grid.width[a] * fraction;
        const double g = Kokkos::max(0.0, Kokkos::max(u[a] * width - q[a], q[a] - (u[a] + 1) * width));
        gap2 += g * g;
    }
    if (gap2 >= reach2) return false;
    if (finer >= 0) {
        // Covered if all 8 children of u are: per axis the farther child, seen from the finer home cell.
        int gaps = 0;
        for (int a = 0; a < 3; ++a) {
            const int child = 2 * (u[a] - home[a]) - ((finer >> a) & 1);
            gaps += Kokkos::max(beyond2(child), beyond2(child + 1));
        }
        const int covered = level_reach(level + 1, cubic ? 1 : grid.fewest);
        if (gaps < covered * covered) return false;
    }
    int shift[3], w[3];  // periodic image and wrapped coordinate
    for (int a = 0; a < 3; ++a) {
        const int side = cubic ? 1 << level : grid.tiles[a] << level;
        shift[a] = u[a] < 0 ? -1 : u[a] >= side ? 1 : 0;
        w[a] = cubic ? u[a] & (side - 1) : u[a] - shift[a] * side;
    }
    packed = pack_image(shift[0], shift[1], shift[2]);
    if constexpr (adaptive) {
        grid.template range<cubic>(level, w[0], w[1], w[2], begin, end);
    } else {
        const int key = grid.template cell<cubic>(level, w[0], w[1], w[2]);
        begin = grid.cell_offsets(key);
        end = grid.cell_offsets(key + 1);
    }
    return begin < end;
}

}  // namespace shell_voronoi
