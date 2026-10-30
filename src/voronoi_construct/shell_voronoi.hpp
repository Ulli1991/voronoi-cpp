#pragma once

#include "neighbor_search.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Fused natural-neighbour Voronoi construction in a periodic box.
//
// Each cell is built by walking grid cells around its generator, nearest
// first, and clipping with every particle that can still cut it. A particle
// at distance d cuts the cell only if d < 2R, where R is the largest distance
// from the generator to a current vertex; the walk stops at the first grid
// cell lying entirely beyond 2R, which certifies the cell. The grid level
// fits the local density (see Stencil). There is no candidate table: the
// particles are clipped nearest-first from a small private buffer.
//
// Output is lean: every cell stores its volume and centroid, and each face is
// stored once, by the cell with the lower slot index (its owner). A flux
// computed on that single face and applied with opposite signs conserves
// exactly. Slots, positions and cell offsets come from a built
// neighbor_search::CellList.
namespace shell_voronoi {

enum Status : int {
    ok = 0,
    uncertified = 1,      // the walk reached the coarsest grid the box allows
    plane_overflow = 2,
    vertex_overflow = 3,
    face_overflow = 4,    // more owned faces than the output holds (construct widens it), or rings
    degenerate = 5,
};
constexpr int status_count = 6;

constexpr int max_planes = 48;    // cutting planes, including the 6 box planes
constexpr int max_vertices = 64;  // one 64-bit mask marks vertices to remove
constexpr int big_planes = 128, big_vertices = 256;  // for the rare cells that overflow those
constexpr int ring_buffer = 32;
constexpr int max_owned_faces = 32;  // owned faces per cell the output holds at first
constexpr int box_planes = 6;

template <class Real>
struct Vec3 {
    Real x, y, z;
};

template <class Real>
KOKKOS_INLINE_FUNCTION Vec3<Real> operator+(Vec3<Real> a, Vec3<Real> b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
template <class Real>
KOKKOS_INLINE_FUNCTION Vec3<Real> operator-(Vec3<Real> a, Vec3<Real> b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
template <class Real>
KOKKOS_INLINE_FUNCTION Vec3<Real> operator*(Vec3<Real> a, Real s) {
    return {a.x * s, a.y * s, a.z * s};
}
template <class Real>
KOKKOS_INLINE_FUNCTION Real dot(Vec3<Real> a, Vec3<Real> b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
template <class Real>
KOKKOS_INLINE_FUNCTION Vec3<Real> cross(Vec3<Real> a, Vec3<Real> b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
template <class To, class From>
KOKKOS_INLINE_FUNCTION Vec3<To> convert(Vec3<From> a) {
    return {static_cast<To>(a.x), static_cast<To>(a.y), static_cast<To>(a.z)};
}

// Read-only inputs, all in slot order, from a built neighbor_search::CellList.
template <class MemorySpace>
struct Grid {
    Kokkos::View<double* [3], MemorySpace> positions;
    Kokkos::View<int*, MemorySpace> cell_offsets;  // base cells in Morton order
    Kokkos::View<std::uint32_t*, MemorySpace> fine;
    Kokkos::View<std::uint8_t*, MemorySpace> level;
    int n = 0;
    int levels = 2;  // base grid: 2^levels cells per side
    double box = 1.0;

    // Place of base cell (x, y, z) in cell_offsets.
    KOKKOS_INLINE_FUNCTION int cell(int x, int y, int z) const {
        return static_cast<int>(neighbor_search::morton_short(x, y, z));
    }

    // The slots [begin, end) of grid cell (x, y, z) at `level` (2^level
    // cells per side): a block of base cells, or part of one base cell.
    KOKKOS_INLINE_FUNCTION void range(int level, int x, int y, int z, int& begin, int& end) const {
        const int below = level - levels;
        if (below <= 0) {
            const int span = -3 * below;
            const int first = cell(x, y, z) << span;
            begin = cell_offsets(first);
            end = cell_offsets(first + (1 << span));
            return;
        }
        const int base = cell(x >> below, y >> below, z >> below);
        begin = cell_offsets(base);
        end = cell_offsets(base + 1);
        if (begin == end) return;
        const int mask = (1 << below) - 1, drop = 3 * (neighbor_search::fine_levels - below);
        const std::uint32_t key = neighbor_search::morton_short(x & mask, y & mask, z & mask) << drop;
        begin = neighbor_search::lower_bound(fine, begin, end, key);
        end = neighbor_search::lower_bound(fine, begin, end, key + (std::uint32_t(1) << drop));
    }
};

template <class ExecSpace>
Grid<typename ExecSpace::memory_space> make_grid(const neighbor_search::CellList<ExecSpace>& list) {
    return {list.positions, list.cell_offsets, list.fine, list.level, list.n, list.levels, list.box};
}

// The walk around a generator. Each cell is walked on the grid level whose
// cells hold about one particle near it (CellList::level), grid cell by grid
// cell in the order of a stencil. When the stencil of that level ends before
// the cell is certified, the walk goes on one level coarser and skips the
// grid cells the finer walk covered.
//
// Stencil: grid-cell offsets (4 bits per axis, biased by 8) sorted by the
// squared smallest distance, in cell widths, that any point of the home cell
// can have to them. The first ends(r) entries are the cells closer than r
// widths.
constexpr int max_reach = 4;
template <class MemorySpace>
struct Stencil {
    Kokkos::View<int*, MemorySpace> offsets;
    Kokkos::View<float*, MemorySpace> bound2;
    Kokkos::View<int*, MemorySpace> ends;
};

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

// Cells closer than this many widths are walked at `level`: max_reach, or
// less where fewer than 2 reach + 1 cells per side would visit a cell twice.
KOKKOS_INLINE_FUNCTION int level_reach(int level) { return Kokkos::min(int{max_reach}, ((1 << level) - 1) / 2); }

KOKKOS_INLINE_FUNCTION int beyond2(int d) {
    const int b = Kokkos::max(Kokkos::abs(d) - 1, 0);
    return b * b;
}

// Periodic image (-1, 0, 1 per axis) packed into 6 bits.
KOKKOS_INLINE_FUNCTION int pack_image(int x, int y, int z) { return (x + 1) | (y + 1) << 2 | (z + 1) << 4; }
KOKKOS_INLINE_FUNCTION int image(int packed, int axis) { return ((packed >> (2 * axis)) & 3) - 1; }

// The home cell of generator p at `level` (2^level cells per side).
KOKKOS_INLINE_FUNCTION void home_cell(const Vec3<double>& p, double box, int level, int (&home)[3]) {
    home[0] = neighbor_search::grid_coordinate(p.x, box, 1 << level);
    home[1] = neighbor_search::grid_coordinate(p.y, box, 1 << level);
    home[2] = neighbor_search::grid_coordinate(p.z, box, 1 << level);
}

// The grid cell at stencil offset `code` of the walk at `level` around
// generator p in cell `home`: false if it lies beyond 2R (reach2 = 4R^2) or
// inside the finished walk one level finer, whose home cell is 2 home + bit
// a of `finer` along axis a (finer < 0: none); else its slots and the
// periodic image they are seen in.
template <bool adaptive = true, class MemorySpace>
KOKKOS_INLINE_FUNCTION bool stencil_cell(const Grid<MemorySpace>& grid, int level, const int (&home)[3], int finer,
                                         int code, const Vec3<double>& p, double reach2, int& begin, int& end,
                                         int& packed) {
    const int side = 1 << level;
    const double width = grid.box / side;
    const int u[3] = {home[0] + (code & 15) - 8, home[1] + ((code >> 4) & 15) - 8, home[2] + ((code >> 8) & 15) - 8};
    const double q[3] = {p.x, p.y, p.z};
    double gap2 = 0;
    for (int a = 0; a < 3; ++a) {
        const double g = Kokkos::max(0.0, Kokkos::max(u[a] * width - q[a], q[a] - (u[a] + 1) * width));
        gap2 += g * g;
    }
    if (gap2 >= reach2) return false;
    if (finer >= 0) {
        // Covered if all 8 children of u are: per axis the farther child,
        // seen from the finer home cell.
        int gaps = 0;
        for (int a = 0; a < 3; ++a) {
            const int child = 2 * (u[a] - home[a]) - ((finer >> a) & 1);
            gaps += Kokkos::max(beyond2(child), beyond2(child + 1));
        }
        const int covered = level_reach(level + 1);
        if (gaps < covered * covered) return false;
    }
    packed = pack_image(u[0] < 0 ? -1 : u[0] >= side ? 1 : 0, u[1] < 0 ? -1 : u[1] >= side ? 1 : 0,
                        u[2] < 0 ? -1 : u[2] >= side ? 1 : 0);
    if constexpr (adaptive) {
        grid.range(level, u[0] & (side - 1), u[1] & (side - 1), u[2] & (side - 1), begin, end);
    } else {
        const int key = grid.cell(u[0] & (side - 1), u[1] & (side - 1), u[2] & (side - 1));
        begin = grid.cell_offsets(key);
        end = grid.cell_offsets(key + 1);
    }
    return begin < end;
}

// Per-cell results. Face arrays are [cell, owned face]; area vectors point
// out of the owning cell, centroids are relative to the owner's generator.
// They hold owned_faces per cell at first; a build that finds a cell with more
// widens them to that cell's count.
template <class MemorySpace>
struct Cells {
    Kokkos::View<int*, MemorySpace> status;
    Kokkos::View<int*, MemorySpace> face_count;   // all faces of the cell
    Kokkos::View<int*, MemorySpace> owned_count;  // faces stored below
    // Size 0 in a build with face rings.
    Kokkos::View<int**, MemorySpace> neighbour;
    Kokkos::View<double** [3], MemorySpace> area;
    Kokkos::View<double** [3], MemorySpace> face_centroid;
    Kokkos::View<double*, MemorySpace> volume;
    Kokkos::View<double* [3], MemorySpace> centroid;
    // Optional, size 0 = off (thread kernel only): the face rings. Cell t's
    // owned faces, in order, start at rings(ring_start(t)), each as its
    // neighbour, its ring length k and the k slots whose bisectors bound it,
    // so that vertex m of the face is equidistant from the cell, the
    // neighbour, ring[m] and ring[m + 1]. The pool is split into
    // ring_used.extent(0) equal chunks; cell t claims its block in chunk
    // ring_chunk(t) (one counter per chunk: a single counter serializes the
    // build; hashed so that no chunk collects one side of the lattice).
    // Cells that did not fit are face_overflow; ring_entries_needed gives a
    // pool size that fits (see with_rings).
    Kokkos::View<int*, MemorySpace> rings;
    Kokkos::View<long long*, MemorySpace> ring_start;
    Kokkos::View<unsigned*, MemorySpace> ring_used;
    // Optional diagnostics, size 0 = off: |sum of area vectors| / sum of
    // |area|, and work per cell (batches of grid cells walked, candidates
    // clipped, cuts).
    Kokkos::View<double*, MemorySpace> closure;
    Kokkos::View<int* [3], MemorySpace> work;

    Cells() = default;
    Cells(int n, bool with_diagnostics, int owned_faces = max_owned_faces)
        : status("cell_status", n),
          face_count("cell_face_count", n),
          owned_count("cell_owned_count", n),
          neighbour("face_neighbour", n, owned_faces),
          area("face_area", n, owned_faces),
          face_centroid("face_centroid", n, owned_faces),
          volume("cell_volume", n),
          centroid("cell_centroid", n),
          closure("cell_closure", with_diagnostics ? n : 0),
          work("cell_work", with_diagnostics ? n : 0) {}

    KOKKOS_INLINE_FUNCTION int owned_capacity() const { return static_cast<int>(neighbour.extent(1)); }

    // Cells that record face rings, room for `entries` (about 60 per cell),
    // and no per-face arrays. If ring_entries_needed exceeds that after a
    // build, build again with that many.
    static Cells with_rings(int n, long long entries) {
        Cells c;
        c.status = Kokkos::View<int*, MemorySpace>("cell_status", n);
        c.face_count = Kokkos::View<int*, MemorySpace>("cell_face_count", n);
        c.owned_count = Kokkos::View<int*, MemorySpace>("cell_owned_count", n);
        c.volume = Kokkos::View<double*, MemorySpace>("cell_volume", n);
        c.centroid = Kokkos::View<double* [3], MemorySpace>("cell_centroid", n);
        c.rings = Kokkos::View<int*, MemorySpace>(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("cell_rings")), entries);
        c.ring_start = Kokkos::View<long long*, MemorySpace>(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("cell_ring_start")), n);
        c.ring_used = Kokkos::View<unsigned*, MemorySpace>("cell_ring_used", Kokkos::max(1, n / 1024));
        return c;
    }

    // Pool entries a build with rings needed: every chunk as large as the fullest.
    long long ring_entries_needed() const {
        const auto used = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ring_used);
        unsigned most = 0;
        for (std::size_t c = 0; c < used.extent(0); ++c) most = Kokkos::max(most, used(c));
        return static_cast<long long>(most) * static_cast<long long>(used.extent(0));
    }
};

// The ring-pool chunk of cell t: a multiplicative hash, as slot order is
// spatial and t % chunks can pick, say, only the cells at one box face.
KOKKOS_INLINE_FUNCTION unsigned ring_chunk(int t, unsigned chunks) {
    return static_cast<unsigned>((static_cast<unsigned long long>(static_cast<unsigned>(t) * 2654435761u) * chunks) >> 32);
}

// Claim `need` ring entries for cell t: sets ring_start(t), returns false
// if its chunk is full.
template <class MemorySpace>
KOKKOS_INLINE_FUNCTION bool claim_rings(const Cells<MemorySpace>& out, int t, int need) {
    const unsigned chunks = static_cast<unsigned>(out.ring_used.extent(0));
    const long long chunk = static_cast<long long>(out.rings.extent(0)) / chunks;
    const unsigned c = ring_chunk(t, chunks);
    const unsigned at = Kokkos::atomic_fetch_add(&out.ring_used(c), static_cast<unsigned>(need));
    out.ring_start(t) = c * chunk + at;
    return static_cast<long long>(at) + need <= chunk;
}

namespace detail {

// Private state of one cell under construction. A vertex is the meeting point
// of three planes, stored in an order that walks each face consistently.
// Planes and Vertices are capacities: the defaults fit almost every cell;
// the rare larger ones are rebuilt with big ones (big_planes, big_vertices).
template <class Real, int Planes = max_planes, int Vertices = max_vertices>
struct Polyhedron {
    static_assert(Planes <= 255, "plane indices are bytes");
    static constexpr int planes_capacity = Planes, vertices_capacity = Vertices;
    Vec3<Real> normal[Planes];  // the cell is {x : normal . x <= offset}
    Real offset[Planes];
    int neighbour[Planes];
    Vec3<Real> vertex[Vertices];
    std::uint8_t corner[Vertices][3];
    int planes = 0;
    int vertices = 0;
    Real radius2 = 0;  // largest squared vertex distance from the generator
};

template <class Cell, class Real>
KOKKOS_INLINE_FUNCTION bool intersect(const Cell& cell, int a, int b, int c,
                                      Vec3<Real>& x) {
    const Vec3<Real> bc = cross(cell.normal[b], cell.normal[c]);
    const Vec3<Real> ca = cross(cell.normal[c], cell.normal[a]);
    const Vec3<Real> ab = cross(cell.normal[a], cell.normal[b]);
    const Real det = dot(cell.normal[a], bc);
    if (!(Kokkos::abs(det) > Real(0))) return false;
    x = (bc * cell.offset[a] + ca * cell.offset[b] + ab * cell.offset[c]) * (Real(1) / det);
    return true;
}

// The starting cube must lie well inside half the box so that every
// candidate that can cut the cell is unique in the periodic image.
template <class Real>
Real start_half_width(double box) {
    return static_cast<Real>(0.25 * box);
}

// Cube of half-width h around the generator. Each corner lists its three
// planes in the order that walks every face the same way round, so the order
// flips at corners of odd parity.
template <class Cell, class Real>
KOKKOS_INLINE_FUNCTION void start_cube(Cell& cell, Real h) {
    const Vec3<Real> axes[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int a = 0; a < 3; ++a) {
        cell.normal[2 * a] = axes[a];
        cell.normal[2 * a + 1] = axes[a] * Real(-1);
        cell.offset[2 * a] = cell.offset[2 * a + 1] = h;
        cell.neighbour[2 * a] = cell.neighbour[2 * a + 1] = -1;
    }
    cell.planes = box_planes;
    for (int c = 0; c < 8; ++c) {
        const int sx = c & 1, sy = (c >> 1) & 1, sz = (c >> 2) & 1;
        const bool odd = (sx + sy + sz) % 2 == 1;
        const std::uint8_t px = sx, py = 2 + sy, pz = 4 + sz;
        cell.corner[c][0] = px;
        cell.corner[c][1] = odd ? pz : py;
        cell.corner[c][2] = odd ? py : pz;
        cell.vertex[c] = {sx ? -h : h, sy ? -h : h, sz ? -h : h};
    }
    cell.vertices = 8;
    cell.radius2 = 3 * h * h;
}

// Drop planes no vertex uses any more (cut away by later planes) and renumber
// the rest. The six box planes keep their indices. Returns the freed count.
template <class Cell>
KOKKOS_INLINE_FUNCTION int compact_planes(Cell& cell) {
    constexpr int capacity = Cell::planes_capacity;
    bool used[capacity];
    for (int p = 0; p < cell.planes; ++p) used[p] = p < box_planes;
    for (int v = 0; v < cell.vertices; ++v)
        for (int k = 0; k < 3; ++k) used[cell.corner[v][k]] = true;
    std::uint8_t renumber[capacity];
    int planes = 0;
    for (int p = 0; p < cell.planes; ++p) {
        if (!used[p]) continue;
        renumber[p] = static_cast<std::uint8_t>(planes);
        cell.normal[planes] = cell.normal[p];
        cell.offset[planes] = cell.offset[p];
        cell.neighbour[planes] = cell.neighbour[p];
        ++planes;
    }
    for (int v = 0; v < cell.vertices; ++v)
        for (int k = 0; k < 3; ++k) cell.corner[v][k] = renumber[cell.corner[v][k]];
    const int freed = cell.planes - planes;
    cell.planes = planes;
    return freed;
}

// Clip by the bisector of the generator and a particle at offset d. Returns
// ok when the plane does not cut, or after the cut has been applied.
template <class Cell, class Real>
KOKKOS_INLINE_FUNCTION int clip(Cell& cell, Vec3<Real> d, int neighbour) {
    constexpr int max_holes = Cell::vertices_capacity / 2;
    const Real offset = Real(0.5) * dot(d, d);
    const Real tolerance =
        Real(64) * Kokkos::Experimental::epsilon_v<Real> * Kokkos::sqrt(dot(d, d) * cell.radius2);
    bool outside[Cell::vertices_capacity];
    bool any = false;
    for (int v = 0; v < cell.vertices; ++v) {
        outside[v] = dot(d, cell.vertex[v]) - offset > tolerance;
        any |= outside[v];
    }
    if (!any) return ok;
    if (cell.planes == Cell::planes_capacity && compact_planes(cell) == 0) return plane_overflow;

    // The removed vertices' directed edges that are not cancelled by a
    // reversed twin form the boundary of the hole the new face closes.
    std::uint8_t hole[max_holes][2];
    int holes = 0;
    for (int v = 0; v < cell.vertices; ++v) {
        if (!outside[v]) continue;
        for (int e = 0; e < 3; ++e) {
            const std::uint8_t a = cell.corner[v][e], b = cell.corner[v][(e + 1) % 3];
            int twin = -1;
            for (int h = 0; h < holes; ++h)
                if (hole[h][0] == b && hole[h][1] == a) twin = h;
            if (twin >= 0) {
                hole[twin][0] = hole[holes - 1][0];
                hole[twin][1] = hole[holes - 1][1];
                --holes;
            } else {
                if (holes == max_holes) return vertex_overflow;
                hole[holes][0] = a;
                hole[holes][1] = b;
                ++holes;
            }
        }
    }

    const int plane = cell.planes++;
    cell.normal[plane] = d;
    cell.offset[plane] = offset;
    cell.neighbour[plane] = neighbour;

    int kept = 0;
    for (int v = 0; v < cell.vertices; ++v) {
        if (outside[v]) continue;
        cell.vertex[kept] = cell.vertex[v];
        for (int k = 0; k < 3; ++k) cell.corner[kept][k] = cell.corner[v][k];
        ++kept;
    }
    if (kept + holes > Cell::vertices_capacity) return vertex_overflow;
    if (kept + holes < 4) return degenerate;
    Real radius2 = 0;
    for (int v = 0; v < kept; ++v) radius2 = Kokkos::max(radius2, dot(cell.vertex[v], cell.vertex[v]));
    for (int h = 0; h < holes; ++h) {
        Vec3<Real> x;
        if (!intersect(cell, hole[h][0], hole[h][1], plane, x)) return degenerate;
        cell.vertex[kept] = x;
        cell.corner[kept][0] = hole[h][0];
        cell.corner[kept][1] = hole[h][1];
        cell.corner[kept][2] = static_cast<std::uint8_t>(plane);
        radius2 = Kokkos::max(radius2, dot(x, x));
        ++kept;
    }
    cell.vertices = kept;
    cell.radius2 = radius2;
    return ok;
}

struct Candidate {
    float distance2;  // sort key only; clipping recomputes the offset exactly
    int slot;
    std::int8_t image[3];  // periodic image of the slot, in box lengths
};

// Which cells a pass builds.
enum class Pass { all, failed, overflowed, too_many_faces };

template <class Real, int Planes, int Vertices, class ExecSpace>
void construct_pass(const ExecSpace& exec, const Grid<typename ExecSpace::memory_space>& grid,
                    const Stencil<typename ExecSpace::memory_space>& stencil,
                    const Cells<typename ExecSpace::memory_space>& out, Pass pass) {
    const int n = grid.n;
    const double box = grid.box;
    if (n <= 0 || out.status.extent(0) != static_cast<std::size_t>(n))
        throw std::invalid_argument("shell_voronoi::construct: invalid grid or output size");
    const auto positions = grid.positions;
    const bool with_closure = out.closure.extent(0) > 0;
    const int owned_capacity = out.owned_capacity();
    const Real half_width = start_half_width<Real>(box);

    Kokkos::parallel_for(
        "shell_voronoi::construct", Kokkos::RangePolicy<ExecSpace>(exec, 0, n),
        KOKKOS_LAMBDA(int t) {
            const int before = out.status(t);
            if ((pass == Pass::failed && before == ok) ||
                (pass == Pass::overflowed && before != vertex_overflow && before != plane_overflow) ||
                (pass == Pass::too_many_faces && before != face_overflow))
                return;
            const Vec3<double> p{positions(t, 0), positions(t, 1), positions(t, 2)};

            Polyhedron<Real, Planes, Vertices> cell;
            detail::start_cube(cell, half_width);
            int status = ok;
            bool certified = false;
            Candidate buffer[ring_buffer];
            int buffered = 0;
            int rings = 0, clipped = 0, cuts = 0;

            // Clip with the buffered particles nearest-first; stop as soon as
            // the remaining ones are too far to cut.
            auto flush = [&]() {
                for (int i = 1; i < buffered; ++i) {
                    const Candidate c = buffer[i];
                    int j = i;
                    for (; j > 0 && buffer[j - 1].distance2 > c.distance2; --j) buffer[j] = buffer[j - 1];
                    buffer[j] = c;
                }
                for (int i = 0; i < buffered && status == ok; ++i) {
                    const Candidate& c = buffer[i];
                    if (c.distance2 >= 4 * static_cast<float>(cell.radius2) * 1.0001f) break;
                    const Vec3<double> q{positions(c.slot, 0) + c.image[0] * box,
                                         positions(c.slot, 1) + c.image[1] * box,
                                         positions(c.slot, 2) + c.image[2] * box};
                    const int planes_before = cell.planes;
                    status = detail::clip(cell, convert<Real>(q - p), c.slot);
                    ++clipped;
                    cuts += cell.planes > planes_before;
                }
                buffered = 0;
            };

            // The walk (see Stencil), ring_buffer grid cells at a time.
            int level = grid.level(t), home[3], finer = -1;
            home_cell(p, box, level, home);
            while (status == ok) {
                const double width = box / (1 << level);
                const int reach = level_reach(level), entries = stencil.ends(reach);
                for (int first = 0; first < entries && status == ok && !certified; first += ring_buffer) {
                    ++rings;
                    for (int e = first; e < Kokkos::min(first + ring_buffer, entries) && status == ok; ++e) {
                        const double reach2 = 4.0 * static_cast<double>(cell.radius2);
                        if (double(stencil.bound2(e)) * width * width >= reach2) {
                            certified = true;
                            break;
                        }
                        int begin = 0, end = 0, packed = 0;
                        if (!stencil_cell(grid, level, home, finer, stencil.offsets(e), p, reach2, begin, end, packed))
                            continue;
                        const int shift[3] = {image(packed, 0), image(packed, 1), image(packed, 2)};
                        for (int q = begin; q < end; ++q) {
                            if (q == t) continue;
                            const double ddx = positions(q, 0) + shift[0] * box - p.x;
                            const double ddy = positions(q, 1) + shift[1] * box - p.y;
                            const double ddz = positions(q, 2) + shift[2] * box - p.z;
                            const double d2 = ddx * ddx + ddy * ddy + ddz * ddz;
                            if (d2 >= 4.0 * static_cast<double>(cell.radius2)) continue;
                            if (buffered == ring_buffer) flush();
                            using Image = std::int8_t;
                            buffer[buffered++] = {static_cast<float>(d2), q,
                                                  {Image(shift[0]), Image(shift[1]), Image(shift[2])}};
                        }
                    }
                    flush();
                }
                if (status != ok || certified) break;
                // Cells outside the stencil are at least `reach` widths away.
                if (reach * width * reach * width >= 4.0 * static_cast<double>(cell.radius2)) {
                    certified = true;
                    break;
                }
                if (level == 2) break;  // no coarser grid fits the box
                finer = (home[0] & 1) | (home[1] & 1) << 1 | (home[2] & 1) << 2;
                for (int a = 0; a < 3; ++a) home[a] >>= 1;
                --level;
            }
            if (status == ok && !certified) status = uncertified;
            for (int v = 0; v < cell.vertices && status == ok; ++v)
                for (int k = 0; k < 3; ++k)
                    if (cell.corner[v][k] < box_planes) status = uncertified;

            // Rings: claim a block for the owned faces, at most a neighbour,
            // a length and one entry per vertex each.
            const bool with_faces = out.neighbour.extent(0) > 0, with_rings = out.rings.extent(0) > 0;
            long long ring_at = 0;
            if (with_rings && status == ok) {
                detail::compact_planes(cell);  // only planes that bound a face
                int need = 0;
                for (int plane = box_planes; plane < cell.planes; ++plane) need += 2 * (cell.neighbour[plane] > t);
                for (int v = 0; v < cell.vertices; ++v)
                    for (int k = 0; k < 3; ++k) need += cell.neighbour[cell.corner[v][k]] > t;
                if (!claim_rings(out, t, need)) status = face_overflow;
                ring_at = out.ring_start(t);
            }

            // Faces: walk each plane's vertices in order, integrating the face
            // area vector and centroid and the cell volume and centroid.
            int faces = 0, owned = 0, ring_used = 0;
            double volume = 0;
            Vec3<double> moment{0, 0, 0}, area_sum{0, 0, 0};
            double area_magnitude = 0;
            for (int plane = box_planes; plane < cell.planes && status == ok; ++plane) {
                int start = -1;
                for (int v = 0; v < cell.vertices && start < 0; ++v)
                    for (int k = 0; k < 3; ++k)
                        if (cell.corner[v][k] == plane) start = v;
                if (start < 0) continue;  // plane later cut away completely

                const Vec3<double> normal = convert<double>(cell.normal[plane]);
                const Vec3<double> first = convert<double>(cell.vertex[start]);
                Vec3<double> previous = first, area{0, 0, 0}, face_moment{0, 0, 0};
                double weight = 0;
                int current = start, steps = 0, ring_length = 0;
                const bool record = with_rings && cell.neighbour[plane] > t;
                do {
                    int at = 0;
                    while (cell.corner[current][at] != plane) ++at;
                    const std::uint8_t last = cell.corner[current][(at + 2) % 3];
                    // The plane shared with the next vertex bounds the face there.
                    if (record) out.rings(ring_at + ring_used + 2 + ring_length) = cell.neighbour[last];
                    ++ring_length;
                    int next = -1;
                    for (int w = 0; w < cell.vertices && next < 0; ++w)
                        for (int k = 0; k < 3; ++k)
                            if (cell.corner[w][k] == plane && cell.corner[w][(k + 1) % 3] == last) next = w;
                    if (next < 0 || ++steps > Vertices) {
                        status = degenerate;
                        break;
                    }
                    const Vec3<double> x = convert<double>(cell.vertex[next]);
                    const Vec3<double> twice = cross(previous - first, x - first);
                    area = area + twice * 0.5;
                    const double w = dot(twice, normal);
                    weight += w;
                    face_moment = face_moment + (first + previous + x) * (w / 3.0);
                    const double tetra = dot(first, cross(previous, x)) / 6.0;
                    volume += tetra;
                    moment = moment + (first + previous + x) * (tetra / 4.0);
                    previous = x;
                    current = next;
                } while (current != start);
                if (status != ok) break;

                // Faces below rounding level (degenerate, e.g. cospherical
                // generators) carry no flux and are not reported; their
                // volume contribution is kept.
                area_sum = area_sum + area;
                const double sliver = 64 * Kokkos::Experimental::epsilon_v<Real> *
                                      static_cast<double>(cell.radius2);
                if (Kokkos::sqrt(dot(area, area)) <= sliver) continue;
                ++faces;
                area_magnitude += Kokkos::sqrt(dot(area, area));
                const int neighbour = cell.neighbour[plane];
                if (neighbour > t) {
                    if (with_rings) {
                        out.rings(ring_at + ring_used) = neighbour;
                        out.rings(ring_at + ring_used + 1) = ring_length;
                        ring_used += 2 + ring_length;
                    }
                    // Beyond the face arrays' capacity only counted, to widen them.
                    if (with_faces && owned < owned_capacity) {
                        // A sliver face has no meaningful centroid; use a vertex.
                        const Vec3<double> c = weight > 0 ? face_moment * (1.0 / weight) : first;
                        out.neighbour(t, owned) = neighbour;
                        out.area(t, owned, 0) = area.x;
                        out.area(t, owned, 1) = area.y;
                        out.area(t, owned, 2) = area.z;
                        out.face_centroid(t, owned, 0) = c.x;
                        out.face_centroid(t, owned, 1) = c.y;
                        out.face_centroid(t, owned, 2) = c.z;
                    }
                    ++owned;
                }
            }
            if (status == ok && with_faces && owned > owned_capacity) status = face_overflow;
            if (status == ok && !(volume > 0)) status = degenerate;

            out.status(t) = status;
            out.face_count(t) = status == ok ? faces : 0;
            out.owned_count(t) = status == ok || status == face_overflow ? owned : 0;
            out.volume(t) = status == ok ? volume : 0.0;
            const Vec3<double> c = status == ok ? moment * (1.0 / volume) : Vec3<double>{0, 0, 0};
            out.centroid(t, 0) = c.x;
            out.centroid(t, 1) = c.y;
            out.centroid(t, 2) = c.z;
            if (with_closure) {
                out.closure(t) = status == ok ? Kokkos::sqrt(dot(area_sum, area_sum)) / area_magnitude : 0.0;
                out.work(t, 0) = rings;
                out.work(t, 1) = clipped;
                out.work(t, 2) = cuts;
            }
        });
}

// Face arrays of `capacity` owned faces per cell, the old ones copied in.
template <class MemorySpace>
void widen_faces(Cells<MemorySpace>& out, int capacity) {
    const int n = static_cast<int>(out.status.extent(0)), old = out.owned_capacity();
    const auto kept = Kokkos::pair<int, int>(0, old);
    Kokkos::View<int**, MemorySpace> neighbour("face_neighbour", n, capacity);
    Kokkos::View<double** [3], MemorySpace> area("face_area", n, capacity), centroid("face_centroid", n, capacity);
    Kokkos::deep_copy(Kokkos::subview(neighbour, Kokkos::ALL, kept), out.neighbour);
    Kokkos::deep_copy(Kokkos::subview(area, Kokkos::ALL, kept, Kokkos::ALL), out.area);
    Kokkos::deep_copy(Kokkos::subview(centroid, Kokkos::ALL, kept, Kokkos::ALL), out.face_centroid);
    out.neighbour = neighbour;
    out.area = area;
    out.face_centroid = centroid;
}

// Build the cells of `pass`, then rebuild those that overflowed the usual
// capacities once more, in double with big ones. If a cell owns more faces
// than the output holds, widen the output to the largest such count and
// build those cells again.
template <class Real, class ExecSpace>
void construct_with_overflow(const ExecSpace& exec, const Grid<typename ExecSpace::memory_space>& grid,
                             const Stencil<typename ExecSpace::memory_space>& stencil,
                             Cells<typename ExecSpace::memory_space>& out, Pass pass) {
    construct_pass<Real, max_planes, max_vertices>(exec, grid, stencil, out, pass);
    int overflowed = 0, owned = 0;
    const auto status = out.status;
    const auto owned_count = out.owned_count;
    Kokkos::parallel_reduce(
        "shell_voronoi::count_overflows", Kokkos::RangePolicy<ExecSpace>(exec, 0, grid.n),
        KOKKOS_LAMBDA(int t, int& count) { count += status(t) == vertex_overflow || status(t) == plane_overflow; },
        overflowed);
    if (overflowed > 0) construct_pass<double, big_planes, big_vertices>(exec, grid, stencil, out, Pass::overflowed);
    Kokkos::parallel_reduce(
        "shell_voronoi::most_owned_faces", Kokkos::RangePolicy<ExecSpace>(exec, 0, grid.n),
        KOKKOS_LAMBDA(int t, int& most) {
            if (status(t) == face_overflow) most = Kokkos::max(most, owned_count(t));
        },
        Kokkos::Max<int>(owned));
    if (owned > out.owned_capacity()) {
        widen_faces(out, owned);
        construct_pass<double, big_planes, big_vertices>(exec, grid, stencil, out, Pass::too_many_faces);
    }
}

}  // namespace detail

// Build every cell with one thread per cell: the portable kernel, and the
// only one that records face rings. Real is the precision of the private
// polyhedron; inputs and outputs are double.
template <class Real, class ExecSpace>
void construct(const ExecSpace& exec, const Grid<typename ExecSpace::memory_space>& grid,
               Cells<typename ExecSpace::memory_space>& out) {
    if (out.rings.extent(0) > 0) Kokkos::deep_copy(exec, out.ring_used, 0u);
    detail::construct_with_overflow<Real>(exec, grid, make_stencil(exec), out, detail::Pass::all);
}

}  // namespace shell_voronoi
