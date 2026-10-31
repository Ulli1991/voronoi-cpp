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
// stored once, by the cell with the lower particle id (its owner), as face
// geometry or as a face ring (Cells). Slots, positions and cell offsets come
// from a built neighbor_search::CellList.
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
constexpr int face_block = 256;  // face pool entries a warp claims at a time (shell_voronoi_warp.hpp)

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

    // The first base cell, in cell_offsets, of grid cell (x, y, z) at
    // `level` <= levels (tiles[a] 2^level cells along axis a).
    KOKKOS_INLINE_FUNCTION int cell(int level, int x, int y, int z) const {
        const int mask = (1 << level) - 1;
        const int tile = ((x >> level) * tiles[1] + (y >> level)) * tiles[2] + (z >> level);
        return tile << (3 * levels) |
               static_cast<int>(neighbor_search::morton_short(x & mask, y & mask, z & mask)) << (3 * (levels - level));
    }

    // The slots [begin, end) of grid cell (x, y, z) at `level` (2^level
    // cells per tile side): a block of base cells, or part of one base cell.
    KOKKOS_INLINE_FUNCTION void range(int level, int x, int y, int z, int& begin, int& end) const {
        const int below = level - levels;
        if (below <= 0) {
            const int span = -3 * below;
            const int first = cell(level, x, y, z);
            begin = cell_offsets(first);
            end = cell_offsets(first + (1 << span));
            return;
        }
        const int base = cell(levels, x >> below, y >> below, z >> below);
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
// less where fewer than 2 reach + 1 cells along the shortest axis (`fewest`
// tiles) would visit a cell twice. So a cell is certified only within one
// periodic image: its reach 2R must stay below 3/8 of the shortest side
// (reach 3 at 8 cells across), else it is uncertified (too few particles
// across a thin slab; voronoi::Mesh refuses such a box).
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
template <class MemorySpace>
KOKKOS_INLINE_FUNCTION void home_cell(const Vec3& p, const Grid<MemorySpace>& grid, int level, int (&home)[3]) {
    home[0] = neighbor_search::grid_coordinate(p.x, grid.box.side[0], grid.tiles[0] << level);
    home[1] = neighbor_search::grid_coordinate(p.y, grid.box.side[1], grid.tiles[1] << level);
    home[2] = neighbor_search::grid_coordinate(p.z, grid.box.side[2], grid.tiles[2] << level);
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
    const double fraction = 1.0 / (1 << level);  // of a tile
    const int u[3] = {home[0] + (code & 15) - 8, home[1] + ((code >> 4) & 15) - 8, home[2] + ((code >> 8) & 15) - 8};
    const double q[3] = {p.x, p.y, p.z};
    double gap2 = 0;
    for (int a = 0; a < 3; ++a) {
        const double width = grid.width[a] * fraction;
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
        const int covered = level_reach(level + 1, grid.fewest);
        if (gaps < covered * covered) return false;
    }
    int shift[3], w[3];  // periodic image and wrapped coordinate
    for (int a = 0; a < 3; ++a) {
        const int side = grid.tiles[a] << level;
        shift[a] = u[a] < 0 ? -1 : u[a] >= side ? 1 : 0;
        w[a] = u[a] - shift[a] * side;
    }
    packed = pack_image(shift[0], shift[1], shift[2]);
    if constexpr (adaptive) {
        grid.range(level, w[0], w[1], w[2], begin, end);
    } else {
        const int key = grid.cell(level, w[0], w[1], w[2]);
        begin = grid.cell_offsets(key);
        end = grid.cell_offsets(key + 1);
    }
    return begin < end;
}

// Per-cell results, indexed by particle (CellList::order), as are the
// neighbours and rings. Owned faces live in a pool of blocks, one per cell:
// cell t's faces are face_start(t) + [0, owned_count(t)), each with its
// neighbour, area vector (out of t) and centroid (relative to t's
// generator). A build with face rings records rings in a second pool
// instead (see rings) and no face geometry.
//
// The face pool has one counter; the warp kernel claims blocks of many
// cells' faces at a time. The ring pool is split into equal chunks; cell t
// claims its block in chunk pool_chunk(t) (one counter per chunk: a single
// counter serializes the build when every cell claims; hashed so that no
// chunk collects one side of a lattice). Cells whose chunk is full are
// face_overflow; the construct functions then build again with a pool of
// needed() entries.
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

    // Cells of n particles with owned faces, room for faces_per_cell per
    // built cell on average (about 8 are needed) and a part-used block per
    // warp; `built` cells are built (all by default; fewer for a local
    // rebuild).
    Cells(int n, bool with_diagnostics, int faces_per_cell = 10, int built = -1) : Cells(n) {
        if (built < 0) built = n;
        face_start = Kokkos::View<int*, MemorySpace>(uninitialized("cell_face_start"), n);
        face_used = Kokkos::View<unsigned*, MemorySpace>("cell_face_used", 1);
        reallocate(static_cast<long long>(faces_per_cell) * built +
                   static_cast<long long>(face_block) * Kokkos::min(built, 8192));
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

// The cells a build constructs: every cell, or the slots `slots` only.
// With `listed` (flags by particle), a listed cell also stores the faces it
// shares with unlisted particles, so a local rebuild returns each face of
// the listed cells once.
template <class MemorySpace>
struct Selection {
    Kokkos::View<const int*, MemorySpace> slots;
    Kokkos::View<const char*, MemorySpace> listed;
    KOKKOS_INLINE_FUNCTION bool all() const { return slots.data() == nullptr; }
    KOKKOS_INLINE_FUNCTION int count(int n) const { return all() ? n : static_cast<int>(slots.extent(0)); }
    KOKKOS_INLINE_FUNCTION int slot(int i) const { return all() ? i : slots(i); }
    // Whether cell `id` stores its face with particle `other` (the owner rule).
    KOKKOS_INLINE_FUNCTION bool owns(int id, int other) const {
        return other > id || (listed.data() != nullptr && !listed(other));
    }
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

// The starting cube must lie well inside half the shortest side so that
// every candidate that can cut the cell is unique in the periodic image.
inline double start_half_width(const ep::Box& box) { return 0.25 * box.shortest(); }

// Two kinds of cells under construction share the walk, the cut topology
// and the face walk of construct_pass. Both keep planes (each with the
// particle id beyond it; box planes are virtual particles from n on) and
// vertices, each listing its three planes in the order that walks every
// face the same way round and the three vertices across its edges
// (adjacent[v][k] lies across the edge from corner k to corner k + 1). A cut
// finds its hole and a face walk its next vertex through those links, and
// vertices stay in their slots: removed ones are marked and their slots
// reused (idea from Doug Rennehan's plume). Planes and Vertices are
// capacities: the defaults fit almost every cell; the rare larger ones are
// built again with big ones.
//
// RoundedCell: planes and vertices in double, relative to the generator; a
// vertex is cut when it lies beyond the bisector by more than a rounding
// tolerance. The warp kernel's fallback on CUDA, and the portable kernel.
//
// ExactCell: in grid steps relative to the snapped generator. Vertex v is
// the circumcentre of the generator and the particles of its three planes,
// -w / (2 o); a particle at e lies inside that sphere iff e.w + |e|^2 o has
// the opposite sign of o (w and o are exact integers, stored rounded once),
// decided by a filter and, when that is uncertain, by the exact perturbed
// in-sphere sign. Neighbouring cells agree by construction: these are the
// cells of the unique perturbed Delaunay tetrahedralization. The start's
// patch (tet_start).
template <int Planes, int Vertices>
struct RoundedCell {
    static_assert(Planes <= 255, "plane indices are bytes");
    static constexpr int planes_capacity = Planes, vertices_capacity = Vertices;
    static constexpr bool exact = false;
    Vec3 normal[Planes];  // the cell is {x : normal . x <= offset}
    double offset[Planes];
    int id[Planes];
    Vec3 vertex[Vertices];
    std::uint8_t corner[Vertices][3], adjacent[Vertices][3];
    std::uint8_t free_slot[Vertices];           // stack of the removed vertices' slots
    std::uint8_t starts[Planes], ends[Planes];  // a cut's new vertex whose hole edge starts / ends at a plane
    int planes = 0, vertices = 0, free_count = 0;  // vertices: slots ever used, free_count of them on the stack
    double radius2 = 0;  // largest squared vertex distance from the generator

    KOKKOS_INLINE_FUNCTION Vec3 position(int v, double) const { return vertex[v]; }
    KOKKOS_INLINE_FUNCTION Vec3 face_normal(int plane) const { return normal[plane]; }
    KOKKOS_INLINE_FUNCTION double reach2(double) const { return 4 * radius2; }
    KOKKOS_INLINE_FUNCTION double radius2_in_box(double) const { return radius2; }
    KOKKOS_INLINE_FUNCTION void move_plane(int to, int from) {
        normal[to] = normal[from];
        offset[to] = offset[from];
        id[to] = id[from];
    }
};

template <int Planes, int Vertices>
struct ExactCell {
    static_assert(Planes <= 255, "plane indices are bytes");
    static constexpr int planes_capacity = Planes, vertices_capacity = Vertices;
    static constexpr bool exact = true;
    ep::Offset offset[Planes];  // generator to the plane's particle
    int id[Planes];
    std::uint8_t corner[Vertices][3], adjacent[Vertices][3];
    std::uint8_t free_slot[Vertices];
    std::uint8_t starts[Planes], ends[Planes];
    double w[Vertices][3], o[Vertices];
    double r2[Vertices];  // bound on the squared distance of each vertex
    int planes = 0, vertices = 0, free_count = 0;
    double radius2 = 0;  // bound on the squared distance of every vertex

    KOKKOS_INLINE_FUNCTION Vec3 position(int v, double unit) const {
        const double scale = -0.5 * unit / o[v];
        return {w[v][0] * scale, w[v][1] * scale, w[v][2] * scale};
    }
    KOKKOS_INLINE_FUNCTION Vec3 face_normal(int plane) const {
        return {double(offset[plane][0]), double(offset[plane][1]), double(offset[plane][2])};
    }
    // A particle cuts only within twice the largest vertex distance; 8 grid
    // steps cover the snapping of both ends.
    KOKKOS_INLINE_FUNCTION double reach2(double unit) const {
        const double r = (2 * Kokkos::sqrt(radius2) + 8) * unit;
        return r * r;
    }
    KOKKOS_INLINE_FUNCTION double radius2_in_box(double unit) const { return radius2 * unit * unit; }
    KOKKOS_INLINE_FUNCTION void move_plane(int to, int from) {
        offset[to] = offset[from];
        id[to] = id[from];
    }
};

// Vertex v of an exact cell at the meeting point of planes a, b, c (in
// walking order); false if they do not meet in one point.
template <class C>
KOKKOS_INLINE_FUNCTION bool set_vertex(C& cell, int v, int a, int b, int c) {
    ep::i128 n[3];  // the vertex is n / (2 o) = -w / (2 o)
    const ep::i128 o = ep::circumcentre(cell.offset[a], cell.offset[b], cell.offset[c], n);
    if (o == 0) return false;
    for (int k = 0; k < 3; ++k) cell.w[v][k] = -double(n[k]);
    cell.o[v] = double(o);
    const double* w = cell.w[v];
    cell.r2[v] = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]) / (4 * cell.o[v] * cell.o[v]) * (1 + 1e-12);
    cell.corner[v][0] = static_cast<std::uint8_t>(a);
    cell.corner[v][1] = static_cast<std::uint8_t>(b);
    cell.corner[v][2] = static_cast<std::uint8_t>(c);
    return true;
}

// The vertex of a rounded cell where planes a, b, c meet; false if they do
// not meet in one point.
template <class C>
KOKKOS_INLINE_FUNCTION bool intersect(const C& cell, int a, int b, int c, Vec3& x) {
    const Vec3 bc = cross(cell.normal[b], cell.normal[c]);
    const Vec3 ca = cross(cell.normal[c], cell.normal[a]);
    const Vec3 ab = cross(cell.normal[a], cell.normal[b]);
    const double det = dot(cell.normal[a], bc);
    if (!(Kokkos::abs(det) > 0.0)) return false;
    x = (bc * cell.offset[a] + ca * cell.offset[b] + ab * cell.offset[c]) * (1.0 / det);
    return true;
}

// Whether particle `id` at e lies inside the circumsphere of vertex v of an
// exact cell; the filtered value decides unless it is within its rounding
// bound.
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

// corner[v][0] of a removed vertex (no plane index reaches it).
constexpr std::uint8_t removed = 255;
template <class C>
KOKKOS_INLINE_FUNCTION bool live(const C& cell, int v) {
    return cell.corner[v][0] != removed;
}

// The cube of half-width h (box units) around the generator, bounded by six
// virtual particles at distance 2h (ids n..n+5). Corner order flips at
// corners of odd parity, so that every face is walked the same way round.
template <class C>
KOKKOS_INLINE_FUNCTION void start_cube(C& cell, double h, double unit, int n) {
    for (int a = 0; a < 3; ++a)
        for (int s = 0; s < 2; ++s) {
            const int plane = 2 * a + s;
            cell.id[plane] = n + plane;
            if constexpr (C::exact) {
                cell.offset[plane] = {0, 0, 0};
                cell.offset[plane][a] = (s ? -2 : 2) * static_cast<ep::i64>(h / unit);
            } else {
                cell.normal[plane] = {0, 0, 0};
                (a == 0 ? cell.normal[plane].x : a == 1 ? cell.normal[plane].y : cell.normal[plane].z) = s ? -1 : 1;
                cell.offset[plane] = h;
            }
        }
    cell.planes = box_planes;
    cell.radius2 = 0;
    for (int c = 0; c < 8; ++c) {
        const int sx = c & 1, sy = (c >> 1) & 1, sz = (c >> 2) & 1;
        const bool odd = (sx + sy + sz) % 2 == 1;
        const int a = sx, b = odd ? 4 + sz : 2 + sy, d = odd ? 2 + sy : 4 + sz;
        if constexpr (C::exact) {
            set_vertex(cell, c, a, b, d);
            cell.radius2 = Kokkos::max(cell.radius2, cell.r2[c]);
        } else {
            cell.corner[c][0] = static_cast<std::uint8_t>(a);
            cell.corner[c][1] = static_cast<std::uint8_t>(b);
            cell.corner[c][2] = static_cast<std::uint8_t>(d);
            cell.vertex[c] = {sx ? -h : h, sy ? -h : h, sz ? -h : h};
        }
    }
    // Across the edge from corner k to k + 1 lies the corner that differs
    // along the axis of corner k + 2.
    for (int c = 0; c < 8; ++c)
        for (int k = 0; k < 3; ++k)
            cell.adjacent[c][k] = static_cast<std::uint8_t>(c ^ (1 << (cell.corner[c][(k + 2) % 3] / 2)));
    for (int p = 0; p < C::planes_capacity; ++p) cell.starts[p] = cell.ends[p] = 0;
    if constexpr (!C::exact) cell.radius2 = 3 * h * h;
    cell.vertices = 8;
    cell.free_count = 0;
}

// Drop planes no vertex uses any more and renumber the rest (box planes keep
// their indices). Returns the freed count.
template <class C>
KOKKOS_INLINE_FUNCTION int compact_planes(C& cell) {
    constexpr int capacity = C::planes_capacity;
    bool used[capacity];
    for (int p = 0; p < cell.planes; ++p) used[p] = p < box_planes;
    for (int v = 0; v < cell.vertices; ++v)
        if (live(cell, v))
            for (int k = 0; k < 3; ++k) used[cell.corner[v][k]] = true;
    std::uint8_t renumber[capacity];
    int planes = 0;
    for (int p = 0; p < cell.planes; ++p) {
        if (!used[p]) continue;
        renumber[p] = static_cast<std::uint8_t>(planes);
        cell.move_plane(planes, p);
        ++planes;
    }
    for (int v = 0; v < cell.vertices; ++v)
        if (live(cell, v))
            for (int k = 0; k < 3; ++k) cell.corner[v][k] = renumber[cell.corner[v][k]];
    const int freed = cell.planes - planes;
    cell.planes = planes;
    return freed;
}

// Clip by the bisector of the generator `self` and particle `id` (at e in
// grid steps for an exact cell, at d in box units for a rounded one):
// remove the vertices it cuts off and close the hole with a new face.
// Returns ok when the plane does not cut, or after the cut.
template <class C>
KOKKOS_INLINE_FUNCTION int clip(C& cell, const ep::Offset& e, const Vec3& d, int id, int self) {
    constexpr int max_holes = C::vertices_capacity / 2;
    bool cut[C::vertices_capacity];
    int cuts = 0;
    double offset = 0, tolerance = 0;
    if constexpr (!C::exact) {
        offset = 0.5 * dot(d, d);
        tolerance = 64 * Kokkos::Experimental::epsilon_v<double> * Kokkos::sqrt(dot(d, d) * cell.radius2);
    }
    // A removed vertex of a rounded cell sits at the generator, which no
    // bisector cuts off.
    for (int v = 0; v < cell.vertices; ++v) {
        if constexpr (C::exact)
            cut[v] = live(cell, v) && inside(cell, v, e, id, self);
        else
            cut[v] = dot(d, cell.vertex[v]) - offset > tolerance;
        cuts += cut[v];
    }
    if (cuts == 0) return ok;
    if (cell.planes == C::planes_capacity && compact_planes(cell) == 0) return plane_overflow;

    // The removed vertices' edges to kept vertices bound the hole the new
    // face closes: edge (a, b) of a removed vertex, with kept u across it,
    // whose edge `back` leads back. Slot s will hold its new vertex.
    struct Hole {
        std::uint8_t a, b, u, back, s;
    };
    Hole hole[max_holes];
    int holes = 0;
    for (int w = 0; w < cell.vertices; ++w) {
        if (!cut[w]) continue;
        for (int k = 0; k < 3; ++k) {
            const std::uint8_t u = cell.adjacent[w][k];
            if (cut[u]) continue;
            if (holes == max_holes) return vertex_overflow;
            const std::uint8_t* across = cell.adjacent[u];
            const std::uint8_t back = across[0] == w ? 0 : across[1] == w ? 1 : 2;
            hole[holes++] = {cell.corner[w][k], cell.corner[w][(k + 1) % 3], u, back, 0};
        }
    }
    const int kept = cell.vertices - cell.free_count - cuts;
    if (kept + holes > C::vertices_capacity) return vertex_overflow;
    if (kept + holes < 4) return degenerate;
    const int plane = cell.planes++;
    cell.id[plane] = id;
    if constexpr (C::exact) {
        cell.offset[plane] = e;
    } else {
        cell.normal[plane] = d;
        cell.offset[plane] = offset;
    }
    // r2: the largest squared distance of a kept vertex (removed ones count
    // 0), then of any.
    double r2 = 0;
    for (int w = 0; w < cell.vertices; ++w) {
        if (cut[w]) {
            cell.corner[w][0] = removed;
            cell.free_slot[cell.free_count++] = static_cast<std::uint8_t>(w);
            if constexpr (C::exact)
                cell.r2[w] = 0;
            else
                cell.vertex[w] = {0, 0, 0};
        }
        if constexpr (C::exact)
            r2 = Kokkos::max(r2, cell.r2[w]);
        else
            r2 = Kokkos::max(r2, dot(cell.vertex[w], cell.vertex[w]));
    }

    // New vertex (a, b, plane) on each hole edge, in a free slot; the edge
    // back from u now ends at it.
    for (int h = 0; h < holes; ++h) {
        const int a = hole[h].a, b = hole[h].b, u = hole[h].u;
        const int s = cell.free_count > 0 ? cell.free_slot[--cell.free_count] : cell.vertices++;
        if constexpr (C::exact) {
            if (!set_vertex(cell, s, a, b, plane)) return degenerate;
            r2 = Kokkos::max(r2, cell.r2[s]);
        } else {
            Vec3 x;
            if (!intersect(cell, a, b, plane, x)) return degenerate;
            cell.vertex[s] = x;
            cell.corner[s][0] = static_cast<std::uint8_t>(a);
            cell.corner[s][1] = static_cast<std::uint8_t>(b);
            cell.corner[s][2] = static_cast<std::uint8_t>(plane);
            r2 = Kokkos::max(r2, dot(x, x));
        }
        cell.adjacent[u][hole[h].back] = static_cast<std::uint8_t>(s);
        cell.adjacent[s][0] = static_cast<std::uint8_t>(u);
        cell.starts[a] = cell.ends[b] = static_cast<std::uint8_t>(s);
        hole[h].s = static_cast<std::uint8_t>(s);
    }
    // Across (b, plane) lies the new vertex whose hole edge starts at b,
    // across (plane, a) the one whose hole edge ends at a. Rounding can make
    // a hole touch a face twice; then these are not unique, and the cell is
    // left degenerate rather than linked wrongly.
    for (int h = 0; h < holes; ++h) {
        const int a = hole[h].a, b = hole[h].b, s = hole[h].s;
        const int next = cell.starts[b], previous = cell.ends[a];
        if (cell.starts[a] != s || cell.ends[b] != s || !live(cell, next) || cell.corner[next][0] != b ||
            cell.corner[next][2] != plane || !live(cell, previous) || cell.corner[previous][1] != a ||
            cell.corner[previous][2] != plane)
            return degenerate;
        cell.adjacent[s][1] = static_cast<std::uint8_t>(next);
        cell.adjacent[s][2] = static_cast<std::uint8_t>(previous);
    }
    cell.radius2 = r2;
    return ok;
}

struct Candidate {
    double distance2;  // sort key and reach
    int slot;
    std::int8_t image[3];  // periodic image of the slot, in box lengths
};

// Which of the selected cells a pass builds: all, those not ok, or those
// that overflowed the usual capacities.
enum class Pass { all, failed, overflowed };

// The per-thread kernel, with cells of type Cell (RoundedCell or
// ExactCell): the walk (see Stencil), then the face walk that writes the
// owned faces or rings and integrates the geometry.
template <class Cell, class ExecSpace>
void construct_pass(const ExecSpace& exec, const Grid<typename ExecSpace::memory_space>& grid,
                    const Stencil<typename ExecSpace::memory_space>& stencil,
                    const Cells<typename ExecSpace::memory_space>& out, Pass pass,
                    const Selection<typename ExecSpace::memory_space>& selection) {
    const int n = grid.n;
    const ep::Box box = grid.box;
    const double unit = box.step, half = start_half_width(box);
    if (n <= 0 || out.status.extent(0) != static_cast<std::size_t>(n))
        throw std::invalid_argument("shell_voronoi::construct: invalid grid or output size");
    const auto positions = grid.positions;
    const bool with_closure = out.closure.extent(0) > 0;
    const bool with_rings = out.with_rings();

    Kokkos::parallel_for(
        "shell_voronoi::construct", Kokkos::RangePolicy<ExecSpace>(exec, 0, selection.count(n)),
        KOKKOS_LAMBDA(int i) {
            const int t = selection.slot(i);
            const int id = grid.order(t);  // the particle
            const int before = out.status(id);
            // A full pool stays full: those cells wait for a bigger one.
            if ((pass == Pass::failed && (before == ok || before == face_overflow)) ||
                (pass == Pass::overflowed && before != vertex_overflow && before != plane_overflow))
                return;
            const Vec3 p{positions(t, 0), positions(t, 1), positions(t, 2)};
            ep::i64 self[3] = {0, 0, 0};
            if (Cell::exact)
                for (int k = 0; k < 3; ++k) self[k] = ep::snap(positions(t, k), box, k);

            Cell cell;
            start_cube(cell, half, unit, n);
            double reach2 = cell.reach2(unit);
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
                    if (c.distance2 >= reach2) break;
                    ep::Offset e{};
                    Vec3 d{0, 0, 0};
                    if constexpr (Cell::exact)
                        for (int k = 0; k < 3; ++k)
                            e[k] = ep::snap(positions(c.slot, k), box, k) + c.image[k] * box.period[k] - self[k];
                    else
                        d = {positions(c.slot, 0) + c.image[0] * box.side[0] - p.x,
                             positions(c.slot, 1) + c.image[1] * box.side[1] - p.y,
                             positions(c.slot, 2) + c.image[2] * box.side[2] - p.z};
                    const int planes_before = cell.planes, vertices_before = cell.vertices;
                    // Exact cuts break ties by particle id; rounded cells keep
                    // the slot until the walk is done (one load fewer per cut).
                    status = clip(cell, e, d, Cell::exact ? grid.order(c.slot) : c.slot, id);
                    ++clipped;
                    if (cell.planes == planes_before && cell.vertices == vertices_before) continue;
                    ++cuts;
                    reach2 = cell.reach2(unit);
                }
                buffered = 0;
            };

            // The walk (see Stencil), ring_buffer grid cells at a time.
            int level = grid.level(t), home[3], finer = -1;
            home_cell(p, grid, level, home);
            while (status == ok) {
                const double width = grid.narrowest / (1 << level);
                const int reach = level_reach(level, grid.fewest), entries = stencil.ends(reach);
                for (int first = 0; first < entries && status == ok && !certified; first += ring_buffer) {
                    ++batches;
                    for (int e = first; e < Kokkos::min(first + ring_buffer, entries) && status == ok; ++e) {
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
                            const double dx = positions(q, 0) + shift[0] * box.side[0] - p.x;
                            const double dy = positions(q, 1) + shift[1] * box.side[1] - p.y;
                            const double dz = positions(q, 2) + shift[2] * box.side[2] - p.z;
                            const double d2 = dx * dx + dy * dy + dz * dz;
                            if (!(d2 < reach2)) continue;  // never a non-finite distance
                            if (buffered == ring_buffer) flush();
                            using Image = std::int8_t;
                            buffer[buffered++] = {d2, q, {Image(shift[0]), Image(shift[1]), Image(shift[2])}};
                        }
                    }
                    flush();
                }
                if (status != ok || certified) break;
                // Cells outside the stencil are at least `reach` widths away.
                if (reach * width * reach * width >= reach2) {
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
                    if (live(cell, v) && cell.corner[v][k] < box_planes) status = uncertified;

            if (!Cell::exact)
                for (int plane = box_planes; plane < cell.planes; ++plane) cell.id[plane] = grid.order(cell.id[plane]);

            // A block for the owned faces: one entry per owned plane, or with
            // rings a neighbour, a length and one entry per vertex each.
            long long ring_at = 0;
            int face_at = 0;
            if (status == ok) {
                compact_planes(cell);  // only planes that bound a face
                int need = 0;
                for (int plane = box_planes; plane < cell.planes; ++plane)
                    need += (with_rings ? 2 : 1) * selection.owns(id, cell.id[plane]);
                if (with_rings)
                    for (int v = 0; v < cell.vertices; ++v)
                        if (live(cell, v))
                            for (int k = 0; k < 3; ++k) need += selection.owns(id, cell.id[cell.corner[v][k]]);
                if (!(with_rings ? claim_rings(out, id, need) : claim_faces(out, id, need))) status = face_overflow;
                if (with_rings)
                    ring_at = out.ring_start(id);
                else
                    face_at = out.face_start(id);
            }

            // Faces: walk each plane's vertices in order, integrating the face
            // area vector and centroid and the cell volume and centroid.
            int faces = 0, owned = 0, ring_used = 0;
            double volume = 0, area_magnitude = 0;
            Vec3 moment{0, 0, 0}, area_sum{0, 0, 0};
            const double sliver = 64 * Kokkos::Experimental::epsilon_v<double> * cell.radius2_in_box(unit);
            std::int16_t first_vertex[Cell::planes_capacity];  // a vertex on each face, or -1
            for (int plane = 0; plane < cell.planes; ++plane) first_vertex[plane] = -1;
            for (int v = 0; v < cell.vertices; ++v)
                if (live(cell, v))
                    for (int k = 0; k < 3; ++k) first_vertex[cell.corner[v][k]] = static_cast<std::int16_t>(v);
            for (int plane = box_planes; plane < cell.planes && status == ok; ++plane) {
                const int start = first_vertex[plane];
                if (start < 0) continue;  // plane later cut away completely

                const Vec3 normal = cell.face_normal(plane);
                const Vec3 first = cell.position(start, unit);
                Vec3 previous = first, area{0, 0, 0}, face_moment{0, 0, 0};
                double weight = 0;
                int current = start, length = 0;
                const int neighbour = cell.id[plane];
                const bool own = selection.owns(id, neighbour), record = with_rings && own;
                do {
                    int at = 0;
                    while (at < 3 && cell.corner[current][at] != plane) ++at;
                    if (at == 3 || length == Cell::vertices_capacity) {
                        status = degenerate;
                        break;
                    }
                    // The plane shared with the next vertex bounds the face there.
                    if (record) out.rings(ring_at + ring_used + 2 + length) = cell.id[cell.corner[current][(at + 2) % 3]];
                    ++length;
                    const int next = cell.adjacent[current][(at + 2) % 3];
                    const Vec3 x = cell.position(next, unit);
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
                // Faces below rounding level (degenerate, e.g. cospherical
                // generators) carry no flux and are left out of the face
                // geometry, as the warp kernel does (their volume is kept).
                // Rounded rings leave them out too; exact rings hold every
                // face (topology).
                const double magnitude = Kokkos::sqrt(dot(area, area));
                if (!(Cell::exact && with_rings) && magnitude <= sliver) continue;
                ++faces;
                area_magnitude += magnitude;
                if (!own) continue;
                if (with_rings) {
                    out.rings(ring_at + ring_used) = neighbour;
                    out.rings(ring_at + ring_used + 1) = length;
                    ring_used += 2 + length;
                } else {
                    // A sliver face has no meaningful centroid; use a vertex.
                    const Vec3 c = weight > 0 ? face_moment * (1.0 / weight) : first;
                    const int f = face_at + owned;
                    out.neighbour(f) = neighbour;
                    out.area(f, 0) = area.x;
                    out.area(f, 1) = area.y;
                    out.area(f, 2) = area.z;
                    out.face_centroid(f, 0) = c.x;
                    out.face_centroid(f, 1) = c.y;
                    out.face_centroid(f, 2) = c.z;
                }
                ++owned;
            }
            if (status == ok && !(volume > 0)) status = degenerate;

            out.status(id) = status;
            out.face_count(id) = status == ok ? faces : 0;
            out.owned_count(id) = status == ok ? owned : 0;
            out.volume(id) = status == ok ? volume : 0.0;
            const Vec3 c = status == ok ? moment * (1.0 / volume) : Vec3{0, 0, 0};
            out.centroid(id, 0) = c.x;
            out.centroid(id, 1) = c.y;
            out.centroid(id, 2) = c.z;
            if (with_closure) {
                out.closure(id) = status == ok ? Kokkos::sqrt(dot(area_sum, area_sum)) / area_magnitude : 0.0;
                out.work(id, 0) = batches;
                out.work(id, 1) = clipped;
                out.work(id, 2) = cuts;
            }
        });
}

// Build the cells of `pass` with cells of kind Cell (RoundedCell or
// ExactCell), then those that overflowed the usual capacities once more
// with big ones.
template <template <int, int> class Cell, class ExecSpace>
void construct_cells(const ExecSpace& exec, const Grid<typename ExecSpace::memory_space>& grid,
                     const Stencil<typename ExecSpace::memory_space>& stencil,
                     const Cells<typename ExecSpace::memory_space>& out, Pass pass,
                     const Selection<typename ExecSpace::memory_space>& selection = {}) {
    construct_pass<Cell<max_planes, max_vertices>>(exec, grid, stencil, out, pass, selection);
    int overflowed = 0;
    const auto status = out.status;
    const auto order = grid.order;
    Kokkos::parallel_reduce(
        "shell_voronoi::count_overflows", Kokkos::RangePolicy<ExecSpace>(exec, 0, selection.count(grid.n)),
        KOKKOS_LAMBDA(int i, int& sum) {
            const int s = status(order(selection.slot(i)));
            sum += s == vertex_overflow || s == plane_overflow;
        },
        overflowed);
    if (overflowed > 0)
        construct_pass<Cell<big_planes, big_vertices>>(exec, grid, stencil, out, Pass::overflowed, selection);
}

// Run build() on an empty pool; if the pool turned out too small (cells
// face_overflow), again on one that fits.
template <class ExecSpace, class MemorySpace, class Build>
void until_pool_fits(const ExecSpace& exec, Cells<MemorySpace>& out, const Build& build) {
    for (int attempt = 0; attempt < 3; ++attempt) {
        Kokkos::deep_copy(exec, out.with_rings() ? out.ring_used : out.face_used, 0u);
        build();
        const long long needed = out.needed();
        if (needed <= out.capacity()) return;
        out.reallocate(needed + needed / 8);
    }
}

}  // namespace detail

// Build the selected cells (default: all) with one thread per cell: the
// portable kernel (rounded cells, as the warp kernel), or exactly
// (detail::ExactCell) if `exact`.
template <class ExecSpace>
void construct(const ExecSpace& exec, const Grid<typename ExecSpace::memory_space>& grid,
               Cells<typename ExecSpace::memory_space>& out,
               const Selection<typename ExecSpace::memory_space>& selection = {}, bool exact = false) {
    const auto stencil = make_stencil(exec);
    detail::until_pool_fits(exec, out, [&] {
        if (exact)
            detail::construct_cells<detail::ExactCell>(exec, grid, stencil, out, detail::Pass::all, selection);
        else
            detail::construct_cells<detail::RoundedCell>(exec, grid, stencil, out, detail::Pass::all, selection);
    });
}

}  // namespace shell_voronoi
