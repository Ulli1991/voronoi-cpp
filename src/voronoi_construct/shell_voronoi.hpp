#pragma once

#include "exact_predicates.hpp"
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
    face_overflow = 4,    // the cell's chunk of the face or ring pool is full
    degenerate = 5,
};
constexpr int status_count = 6;

constexpr int max_planes = 48;    // cutting planes, including the 6 box planes
constexpr int max_vertices = 64;  // one 64-bit mask marks vertices to remove
constexpr int big_planes = 128, big_vertices = 256;  // for the rare cells that overflow those
constexpr int ring_buffer = 32;
constexpr int box_planes = 6;

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
KOKKOS_INLINE_FUNCTION void home_cell(const Vec3& p, double box, int level, int (&home)[3]) {
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
                                         int code, const Vec3& p, double reach2, int& begin, int& end,
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

// Per-cell results. Owned faces live in a pool of blocks, one per cell:
// cell t's faces are face_start(t) + [0, owned_count(t)), each with its
// neighbour, area vector (out of t) and centroid (relative to t's
// generator). A build with face rings records rings in a second pool
// instead (see rings) and no face geometry.
//
// A pool is split into equal chunks; cell t claims its block in chunk
// pool_chunk(t) (one counter per chunk: a single counter serializes the
// build; hashed so that no chunk collects one side of a lattice). Cells
// whose chunk is full are face_overflow; the construct functions then
// build again with a pool of needed() entries.
template <class MemorySpace>
struct Cells {
    Kokkos::View<int*, MemorySpace> status;
    Kokkos::View<int*, MemorySpace> face_count;   // all faces of the cell
    Kokkos::View<int*, MemorySpace> owned_count;  // faces stored (or ring faces)
    Kokkos::View<double*, MemorySpace> volume;
    Kokkos::View<double* [3], MemorySpace> centroid;
    // Owned faces; size 0 in a build with face rings.
    Kokkos::View<int*, MemorySpace> face_start;
    Kokkos::View<int*, MemorySpace> neighbour;
    Kokkos::View<double* [3], MemorySpace> area, face_centroid;
    Kokkos::View<unsigned*, MemorySpace> face_used;  // per chunk
    // Face rings; size 0 = off. Cell t's owned faces, in order, start at
    // rings(ring_start(t)), each as its neighbour, its ring length k and the
    // k particles whose bisectors bound it, so that vertex m of the face is
    // equidistant from the cell, the neighbour, ring[m] and ring[m + 1].
    Kokkos::View<int*, MemorySpace> rings;
    Kokkos::View<long long*, MemorySpace> ring_start;
    Kokkos::View<unsigned*, MemorySpace> ring_used;  // per chunk
    // Optional diagnostics, size 0 = off: |sum of area vectors| / sum of
    // |area|, and work per cell (batches of grid cells walked, candidates
    // clipped, cuts).
    Kokkos::View<double*, MemorySpace> closure;
    Kokkos::View<int* [3], MemorySpace> work;

    Cells() = default;

    // Cells with owned faces, room for faces_per_cell per cell on average
    // (about 8 are needed).
    Cells(int n, bool with_diagnostics, int faces_per_cell = 10) : Cells(n) {
        face_start = Kokkos::View<int*, MemorySpace>(uninitialized("cell_face_start"), n);
        face_used = Kokkos::View<unsigned*, MemorySpace>("cell_face_used", chunks(n));
        reallocate(static_cast<long long>(faces_per_cell) * n);
        closure = Kokkos::View<double*, MemorySpace>("cell_closure", with_diagnostics ? n : 0);
        work = Kokkos::View<int* [3], MemorySpace>("cell_work", with_diagnostics ? n : 0);
    }

    // Cells that record face rings, room for `entries` (about 60 per cell).
    static Cells with_rings(int n, long long entries) {
        Cells c(n);
        c.ring_start = Kokkos::View<long long*, MemorySpace>(uninitialized("cell_ring_start"), n);
        c.ring_used = Kokkos::View<unsigned*, MemorySpace>("cell_ring_used", chunks(n));
        c.reallocate(entries);
        return c;
    }

    bool with_rings() const { return ring_used.extent(0) > 0; }

    // Entries of the pool in use (faces or ring entries) the last build
    // needed: every chunk as large as the fullest.
    long long needed() const {
        const auto used =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, with_rings() ? ring_used : face_used);
        unsigned most = 0;
        for (std::size_t c = 0; c < used.extent(0); ++c) most = Kokkos::max(most, used(c));
        return static_cast<long long>(most) * static_cast<long long>(used.extent(0));
    }
    long long capacity() const {
        return static_cast<long long>(with_rings() ? rings.extent(0) : neighbour.extent(0));
    }

    // A pool of `entries`; the contents are dropped.
    void reallocate(long long entries) {
        if (with_rings()) {
            rings = {};
            rings = Kokkos::View<int*, MemorySpace>(uninitialized("cell_rings"), entries);
            return;
        }
        neighbour = {};
        area = face_centroid = {};
        neighbour = Kokkos::View<int*, MemorySpace>(uninitialized("face_neighbour"), entries);
        area = Kokkos::View<double* [3], MemorySpace>(uninitialized("face_area"), entries);
        face_centroid = Kokkos::View<double* [3], MemorySpace>(uninitialized("face_centroid"), entries);
    }

private:
    static int chunks(int n) { return Kokkos::max(1, n / 1024); }
    static auto uninitialized(const char* label) {
        return Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label));
    }
    explicit Cells(int n)
        : status("cell_status", n),
          face_count("cell_face_count", n),
          owned_count("cell_owned_count", n),
          volume("cell_volume", n),
          centroid("cell_centroid", n) {}
};

// The pool chunk of cell t: a multiplicative hash, as slot order is spatial
// and t % chunks can pick, say, only the cells at one box face.
KOKKOS_INLINE_FUNCTION unsigned pool_chunk(int t, unsigned chunks) {
    return static_cast<unsigned>((static_cast<unsigned long long>(static_cast<unsigned>(t) * 2654435761u) * chunks) >> 32);
}

// Claim `need` entries of a pool of `capacity` for cell t: sets start,
// returns false if its chunk is full.
template <class Used, class Index>
KOKKOS_INLINE_FUNCTION bool claim(const Used& used, long long capacity, int t, int need, Index& start) {
    const unsigned chunks = static_cast<unsigned>(used.extent(0));
    const long long chunk = capacity / chunks;
    const unsigned c = pool_chunk(t, chunks);
    const unsigned at = Kokkos::atomic_fetch_add(&used(c), static_cast<unsigned>(need));
    start = static_cast<Index>(c * chunk + at);
    return static_cast<long long>(at) + need <= chunk;
}

template <class MemorySpace>
KOKKOS_INLINE_FUNCTION bool claim_rings(const Cells<MemorySpace>& out, int t, int need) {
    return claim(out.ring_used, static_cast<long long>(out.rings.extent(0)), t, need, out.ring_start(t));
}
template <class MemorySpace>
KOKKOS_INLINE_FUNCTION bool claim_faces(const Cells<MemorySpace>& out, int t, int need) {
    return claim(out.face_used, static_cast<long long>(out.neighbour.extent(0)), t, need, out.face_start(t));
}
namespace detail {

namespace ep = exact_predicates;

// The starting cube must lie well inside half the box so that every
// candidate that can cut the cell is unique in the periodic image.
inline double start_half_width(double box) { return 0.25 * box; }

// One cell under construction, in grid steps relative to its snapped
// generator. Vertex v is the circumcentre of the generator and the
// particles of its three planes, -w / (2 o); a particle at e lies inside
// that sphere iff e.w + |e|^2 o has the opposite sign of o (w and o are
// exact integers, stored rounded once). Corners list a vertex's planes in
// the order that walks every face the same way round. Planes and Vertices
// are capacities: the defaults fit almost every cell; the rare larger ones
// are built again with big ones.
template <int Planes, int Vertices>
struct Cell {
    static_assert(Planes <= 255, "plane indices are bytes");
    static constexpr int planes_capacity = Planes, vertices_capacity = Vertices;
    ep::Offset offset[Planes];  // generator to the plane's particle
    int id[Planes];             // that particle; box planes are virtual particles from n on
    std::uint8_t corner[Vertices][3];
    double w[Vertices][3], o[Vertices];
    int planes = 0, vertices = 0;
    double radius2 = 0;  // bound on the squared distance of every vertex
};

// Vertex v at the meeting point of planes a, b, c (in walking order);
// false if they do not meet in one point.
template <class C>
KOKKOS_INLINE_FUNCTION bool set_vertex(C& cell, int v, int a, int b, int c) {
    const ep::Offset &A = cell.offset[a], &B = cell.offset[b], &D = cell.offset[c];
    ep::i128 bd[3], ad[3], ab[3];
    ep::cross(B, D, bd);
    ep::cross(A, D, ad);
    ep::cross(A, B, ab);
    const ep::i128 o = A[0] * bd[0] + A[1] * bd[1] + A[2] * bd[2];
    if (o == 0) return false;
    const ep::i128 la = ep::lift(A), lb = ep::lift(B), ld = ep::lift(D);
    for (int k = 0; k < 3; ++k) cell.w[v][k] = double(-la * bd[k] + lb * ad[k] - ld * ab[k]);
    cell.o[v] = double(o);
    cell.corner[v][0] = static_cast<std::uint8_t>(a);
    cell.corner[v][1] = static_cast<std::uint8_t>(b);
    cell.corner[v][2] = static_cast<std::uint8_t>(c);
    return true;
}

// Vertex v relative to the generator, in grid steps.
template <class C>
KOKKOS_INLINE_FUNCTION Vec3 position(const C& cell, int v) {
    const double scale = -0.5 / cell.o[v];
    return {cell.w[v][0] * scale, cell.w[v][1] * scale, cell.w[v][2] * scale};
}

// Bound on the squared distance of vertex v from the generator.
template <class C>
KOKKOS_INLINE_FUNCTION double radius2(const C& cell, int v) {
    const double* w = cell.w[v];
    return (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]) / (4 * cell.o[v] * cell.o[v]) * (1 + 1e-12);
}

// Whether particle `id` at e lies inside the circumsphere of vertex v; the
// filtered value decides unless it is within its rounding bound.
template <class C>
KOKKOS_INLINE_FUNCTION bool inside(const C& cell, int v, const ep::Offset& e, int id, int self) {
    const double ex = double(e[0]), ey = double(e[1]), ez = double(e[2]);
    const double le = ex * ex + ey * ey + ez * ez;
    const double t0 = ex * cell.w[v][0], t1 = ey * cell.w[v][1], t2 = ez * cell.w[v][2], t3 = le * cell.o[v];
    const double value = t0 + t1 + t2 + t3;
    const double bound =
        16 * ep::epsilon * (Kokkos::abs(t0) + Kokkos::abs(t1) + Kokkos::abs(t2) + Kokkos::abs(t3));
    if (value > bound) return cell.o[v] < 0;
    if (value < -bound) return cell.o[v] > 0;
    const int a = cell.corner[v][0], b = cell.corner[v][1], c = cell.corner[v][2];
    const ep::Offset p[5] = {{0, 0, 0}, cell.offset[a], cell.offset[b], cell.offset[c], e};
    const int ids[5] = {self, cell.id[a], cell.id[b], cell.id[c], id};
    return ep::in_sphere(p, ids);
}

// The cube of half-width h around the generator, bounded by six virtual
// particles at distance 2h (ids n..n+5). Corner order flips at corners of
// odd parity, so that every face is walked the same way round.
template <class C>
KOKKOS_INLINE_FUNCTION void start_cube(C& cell, ep::i64 h, int n) {
    for (int a = 0; a < 3; ++a)
        for (int s = 0; s < 2; ++s) {
            ep::Offset& d = cell.offset[2 * a + s];
            d = {0, 0, 0};
            d[a] = s ? -2 * h : 2 * h;
            cell.id[2 * a + s] = n + 2 * a + s;
        }
    cell.planes = box_planes;
    cell.radius2 = 0;
    for (int c = 0; c < 8; ++c) {
        const int sx = c & 1, sy = (c >> 1) & 1, sz = (c >> 2) & 1;
        const bool odd = (sx + sy + sz) % 2 == 1;
        set_vertex(cell, c, sx, odd ? 4 + sz : 2 + sy, odd ? 2 + sy : 4 + sz);
        cell.radius2 = Kokkos::max(cell.radius2, radius2(cell, c));
    }
    cell.vertices = 8;
}

// Drop planes no vertex uses any more and renumber the rest (box planes keep
// their indices). Returns the freed count.
template <class C>
KOKKOS_INLINE_FUNCTION int compact_planes(C& cell) {
    constexpr int capacity = C::planes_capacity;
    bool used[capacity];
    for (int p = 0; p < cell.planes; ++p) used[p] = p < box_planes;
    for (int v = 0; v < cell.vertices; ++v)
        for (int k = 0; k < 3; ++k) used[cell.corner[v][k]] = true;
    std::uint8_t renumber[capacity];
    int planes = 0;
    for (int p = 0; p < cell.planes; ++p) {
        if (!used[p]) continue;
        renumber[p] = static_cast<std::uint8_t>(planes);
        cell.offset[planes] = cell.offset[p];
        cell.id[planes] = cell.id[p];
        ++planes;
    }
    for (int v = 0; v < cell.vertices; ++v)
        for (int k = 0; k < 3; ++k) cell.corner[v][k] = renumber[cell.corner[v][k]];
    const int freed = cell.planes - planes;
    cell.planes = planes;
    return freed;
}

// Clip by the bisector of the generator `self` and particle `id` at offset
// e: remove the vertices whose circumsphere holds it and close the hole
// with a new face. Returns ok when the plane does not cut, or after the cut.
template <class C>
KOKKOS_INLINE_FUNCTION int clip(C& cell, const ep::Offset& e, int id, int self) {
    constexpr int max_holes = C::vertices_capacity / 2;
    bool cut[C::vertices_capacity];
    bool any = false;
    for (int v = 0; v < cell.vertices; ++v) {
        cut[v] = inside(cell, v, e, id, self);
        any |= cut[v];
    }
    if (!any) return ok;
    if (cell.planes == C::planes_capacity && compact_planes(cell) == 0) return plane_overflow;

    // The removed vertices' directed edges that are not cancelled by a
    // reversed twin bound the hole the new face closes.
    std::uint8_t hole[max_holes][2];
    int holes = 0;
    for (int v = 0; v < cell.vertices; ++v) {
        if (!cut[v]) continue;
        for (int k = 0; k < 3; ++k) {
            const std::uint8_t a = cell.corner[v][k], b = cell.corner[v][(k + 1) % 3];
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
    cell.offset[plane] = e;
    cell.id[plane] = id;
    int kept = 0;
    for (int v = 0; v < cell.vertices; ++v) {
        if (cut[v]) continue;
        for (int k = 0; k < 3; ++k) {
            cell.corner[kept][k] = cell.corner[v][k];
            cell.w[kept][k] = cell.w[v][k];
        }
        cell.o[kept] = cell.o[v];
        ++kept;
    }
    if (kept + holes > C::vertices_capacity) return vertex_overflow;
    double r2 = 0;
    for (int v = 0; v < kept; ++v) r2 = Kokkos::max(r2, radius2(cell, v));
    for (int h = 0; h < holes; ++h) {
        if (!set_vertex(cell, kept, hole[h][0], hole[h][1], plane)) return degenerate;
        r2 = Kokkos::max(r2, radius2(cell, kept));
        ++kept;
    }
    cell.vertices = kept;
    cell.radius2 = r2;
    return ok;
}

struct Candidate {
    double distance2;  // sort key and reach
    int slot;
    std::int8_t image[3];  // periodic image of the slot, in box lengths
};

// Which cells a pass builds: all, those not ok, those that overflowed the
// usual capacities, or the slots list(0..count).
enum class Pass { all, failed, overflowed, listed };

// The exact per-thread kernel: the walk of the clipping (see Stencil), with
// every cut an exact (perturbed) in-sphere sign on the snapped positions,
// so that neighbouring cells agree by construction and the cells are those
// of the unique perturbed Delaunay tetrahedralization. The reach 2R is a
// rigorous bound from w and o, widened by 8 grid steps for the snapping.
template <int Planes, int Vertices, class ExecSpace>
void construct_pass(const ExecSpace& exec, const Grid<typename ExecSpace::memory_space>& grid,
                    const Stencil<typename ExecSpace::memory_space>& stencil,
                    const Cells<typename ExecSpace::memory_space>& out, Pass pass,
                    const Kokkos::View<int*, typename ExecSpace::memory_space>& list = {}, int count = 0) {
    const int n = grid.n;
    const double box = grid.box, unit = box / ep::scale;
    if (n <= 0 || out.status.extent(0) != static_cast<std::size_t>(n))
        throw std::invalid_argument("shell_voronoi::construct: invalid grid or output size");
    const auto positions = grid.positions;
    const bool with_closure = out.closure.extent(0) > 0;
    const bool with_rings = out.with_rings();
    const auto half = static_cast<ep::i64>(start_half_width(box) / unit);

    Kokkos::parallel_for(
        "shell_voronoi::construct", Kokkos::RangePolicy<ExecSpace>(exec, 0, pass == Pass::listed ? count : n),
        KOKKOS_LAMBDA(int i) {
            const int t = pass == Pass::listed ? list(i) : i;
            const int before = out.status(t);
            // A full pool stays full: those cells wait for a bigger one.
            if ((pass == Pass::failed && (before == ok || before == face_overflow)) ||
                (pass == Pass::overflowed && before != vertex_overflow && before != plane_overflow))
                return;
            const Vec3 p{positions(t, 0), positions(t, 1), positions(t, 2)};
            const ep::i64 self[3] = {ep::snap(p.x, box), ep::snap(p.y, box), ep::snap(p.z, box)};

            Cell<Planes, Vertices> cell;
            start_cube(cell, half, n);
            // A particle cuts only within twice the largest vertex distance;
            // 8 grid steps cover the snapping of both ends.
            auto reach2 = [&] {
                const double r = (2 * Kokkos::sqrt(cell.radius2) + 8) * unit;
                return r * r;
            };
            int status = ok;
            bool certified = false;
            Candidate buffer[ring_buffer];
            int buffered = 0, batches = 0, clipped = 0, cuts = 0;

            // Clip with the buffered particles nearest-first; stop as soon as
            // the remaining ones are too far to cut.
            auto flush = [&]() {
                for (int a = 1; a < buffered; ++a) {
                    const Candidate c = buffer[a];
                    int b = a;
                    for (; b > 0 && buffer[b - 1].distance2 > c.distance2; --b) buffer[b] = buffer[b - 1];
                    buffer[b] = c;
                }
                for (int a = 0; a < buffered && status == ok; ++a) {
                    const Candidate& c = buffer[a];
                    if (c.distance2 >= reach2()) break;
                    ep::Offset e;
                    for (int k = 0; k < 3; ++k)
                        e[k] = ep::snap(positions(c.slot, k), box) + c.image[k] * ep::period - self[k];
                    const int planes_before = cell.planes;
                    status = clip(cell, e, c.slot, t);
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
                    ++batches;
                    for (int e = first; e < Kokkos::min(first + ring_buffer, entries) && status == ok; ++e) {
                        if (double(stencil.bound2(e)) * width * width >= reach2()) {
                            certified = true;
                            break;
                        }
                        int begin = 0, end = 0, packed = 0;
                        if (!stencil_cell(grid, level, home, finer, stencil.offsets(e), p, reach2(), begin, end, packed))
                            continue;
                        const int shift[3] = {image(packed, 0), image(packed, 1), image(packed, 2)};
                        for (int q = begin; q < end; ++q) {
                            if (q == t) continue;
                            const double dx = positions(q, 0) + shift[0] * box - p.x;
                            const double dy = positions(q, 1) + shift[1] * box - p.y;
                            const double dz = positions(q, 2) + shift[2] * box - p.z;
                            const double d2 = dx * dx + dy * dy + dz * dz;
                            if (d2 >= reach2()) continue;
                            if (buffered == ring_buffer) flush();
                            using Image = std::int8_t;
                            buffer[buffered++] = {d2, q, {Image(shift[0]), Image(shift[1]), Image(shift[2])}};
                        }
                    }
                    flush();
                }
                if (status != ok || certified) break;
                // Cells outside the stencil are at least `reach` widths away.
                if (reach * width * reach * width >= reach2()) {
                    certified = true;
                    break;
                }
                if (level == 2) break;  // no coarser grid fits the box
                finer = (home[0] & 1) | (home[1] & 1) << 1 | (home[2] & 1) << 2;
                for (int k = 0; k < 3; ++k) home[k] >>= 1;
                --level;
            }
            if (status == ok && !certified) status = uncertified;
            for (int v = 0; v < cell.vertices && status == ok; ++v)
                for (int k = 0; k < 3; ++k)
                    if (cell.corner[v][k] < box_planes) status = uncertified;

            // A block for the owned faces: one entry per owned plane, or with
            // rings a neighbour, a length and one entry per vertex each.
            long long ring_at = 0;
            int face_at = 0;
            if (status == ok) {
                compact_planes(cell);  // only planes that bound a face
                int need = 0;
                for (int plane = box_planes; plane < cell.planes; ++plane)
                    need += (with_rings ? 2 : 1) * (cell.id[plane] > t);
                if (with_rings)
                    for (int v = 0; v < cell.vertices; ++v)
                        for (int k = 0; k < 3; ++k) need += cell.id[cell.corner[v][k]] > t;
                if (!(with_rings ? claim_rings(out, t, need) : claim_faces(out, t, need))) status = face_overflow;
                if (with_rings)
                    ring_at = out.ring_start(t);
                else
                    face_at = out.face_start(t);
            }

            // Faces: walk each plane's vertices in order, integrating the face
            // area vector and centroid and the cell volume and centroid (in
            // grid steps, scaled at the end).
            int faces = 0, owned = 0, ring_used = 0;
            double volume = 0, area_magnitude = 0;
            Vec3 moment{0, 0, 0}, area_sum{0, 0, 0};
            const double sliver = 64 * Kokkos::Experimental::epsilon_v<double> * cell.radius2;
            for (int plane = box_planes; plane < cell.planes && status == ok; ++plane) {
                int start = -1;
                for (int v = 0; v < cell.vertices && start < 0; ++v)
                    for (int k = 0; k < 3; ++k)
                        if (cell.corner[v][k] == plane) start = v;
                if (start < 0) continue;  // plane later cut away completely

                const Vec3 normal{double(cell.offset[plane][0]), double(cell.offset[plane][1]),
                                  double(cell.offset[plane][2])};
                const Vec3 first = position(cell, start);
                Vec3 previous = first, area{0, 0, 0}, face_moment{0, 0, 0};
                double weight = 0;
                int current = start, length = 0;
                const int neighbour = cell.id[plane];
                const bool record = with_rings && neighbour > t;
                do {
                    int at = 0;
                    while (cell.corner[current][at] != plane) ++at;
                    const std::uint8_t last = cell.corner[current][(at + 2) % 3];
                    // The plane shared with the next vertex bounds the face there.
                    if (record) out.rings(ring_at + ring_used + 2 + length) = cell.id[last];
                    ++length;
                    int next = -1;
                    for (int w = 0; w < cell.vertices && next < 0; ++w)
                        for (int k = 0; k < 3; ++k)
                            if (cell.corner[w][k] == plane && cell.corner[w][(k + 1) % 3] == last) next = w;
                    if (next < 0 || length > Vertices) {
                        status = degenerate;
                        break;
                    }
                    const Vec3 x = position(cell, next);
                    const Vec3 twice = cross(previous - first, x - first);
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
                area_sum = area_sum + area;
                // Rings hold every face (topology). Face geometry leaves out
                // faces below rounding level, as the warp kernel does; their
                // volume contribution is kept.
                const double magnitude = Kokkos::sqrt(dot(area, area));
                if (!with_rings && magnitude <= sliver) continue;
                ++faces;
                area_magnitude += magnitude;
                if (neighbour <= t) continue;
                if (with_rings) {
                    out.rings(ring_at + ring_used) = neighbour;
                    out.rings(ring_at + ring_used + 1) = length;
                    ring_used += 2 + length;
                } else {
                    // A sliver face has no meaningful centroid; use a vertex.
                    const Vec3 c = (weight > 0 ? face_moment * (1.0 / weight) : first) * unit;
                    const int f = face_at + owned;
                    out.neighbour(f) = neighbour;
                    out.area(f, 0) = area.x * unit * unit;
                    out.area(f, 1) = area.y * unit * unit;
                    out.area(f, 2) = area.z * unit * unit;
                    out.face_centroid(f, 0) = c.x;
                    out.face_centroid(f, 1) = c.y;
                    out.face_centroid(f, 2) = c.z;
                }
                ++owned;
            }
            if (status == ok && !(volume > 0)) status = degenerate;

            out.status(t) = status;
            out.face_count(t) = status == ok ? faces : 0;
            out.owned_count(t) = status == ok ? owned : 0;
            out.volume(t) = status == ok ? volume * unit * unit * unit : 0.0;
            const Vec3 c = status == ok ? moment * (unit / volume) : Vec3{0, 0, 0};
            out.centroid(t, 0) = c.x;
            out.centroid(t, 1) = c.y;
            out.centroid(t, 2) = c.z;
            if (with_closure) {
                out.closure(t) = status == ok ? Kokkos::sqrt(dot(area_sum, area_sum)) / area_magnitude : 0.0;
                out.work(t, 0) = batches;
                out.work(t, 1) = clipped;
                out.work(t, 2) = cuts;
            }
        });
}

// Build the cells of `pass`, then those that overflowed the usual
// capacities once more with big ones.
template <class ExecSpace>
void construct_cells(const ExecSpace& exec, const Grid<typename ExecSpace::memory_space>& grid,
                     const Stencil<typename ExecSpace::memory_space>& stencil,
                     const Cells<typename ExecSpace::memory_space>& out, Pass pass,
                     const Kokkos::View<int*, typename ExecSpace::memory_space>& list = {}, int count = 0) {
    construct_pass<max_planes, max_vertices>(exec, grid, stencil, out, pass, list, count);
    int overflowed = 0;
    const auto status = out.status;
    Kokkos::parallel_reduce(
        "shell_voronoi::count_overflows", Kokkos::RangePolicy<ExecSpace>(exec, 0, grid.n),
        KOKKOS_LAMBDA(int t, int& sum) { sum += status(t) == vertex_overflow || status(t) == plane_overflow; },
        overflowed);
    if (overflowed > 0) construct_pass<big_planes, big_vertices>(exec, grid, stencil, out, Pass::overflowed);
}

// Run build() on an empty pool; if the pool turned out too small (cells
// face_overflow), again on one that fits.
template <class ExecSpace, class MemorySpace, class Build>
void until_pool_fits(const ExecSpace& exec, Cells<MemorySpace>& out, const Build& build) {
    for (int attempt = 0; attempt < 2; ++attempt) {
        Kokkos::deep_copy(exec, out.with_rings() ? out.ring_used : out.face_used, 0u);
        build();
        const long long needed = out.needed();
        if (needed <= out.capacity()) return;
        out.reallocate(needed + needed / 16);
    }
}

}  // namespace detail

// Build every cell with one thread per cell, exactly (detail::construct_pass):
// the portable kernel, and the fallback of the warp kernel.
template <class ExecSpace>
void construct(const ExecSpace& exec, const Grid<typename ExecSpace::memory_space>& grid,
               Cells<typename ExecSpace::memory_space>& out) {
    const auto stencil = make_stencil(exec);
    detail::until_pool_fits(exec, out, [&] { detail::construct_cells(exec, grid, stencil, out, detail::Pass::all); });
}

}  // namespace shell_voronoi
