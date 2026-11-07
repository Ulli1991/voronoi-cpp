#pragma once

#include "exact_predicates.hpp"
#include "neighbor_search.hpp"
#include "shell_cell.hpp"
#include "shell_grid.hpp"
#include "shell_limits.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Clipping, the thread-per-cell kernel (3D): feeds the start (tet_start::build) on rebuilds and Mesh::clip, and runs on
// host for overflow cells. Each cell is walked on a grid, clipped nearest-first, and its volume, centroid and owned
// faces (or face rings) are stored. Callers: voronoi_mesh.hpp via shell_voronoi_warp.hpp, tet_start.hpp.
namespace shell_voronoi {

// Per-cell results indexed by particle (CellList::order), as are the neighbours and rings. Cell t's owned
// faces are face_start(t) + [0, owned_count(t)), each with neighbour, area vector (out of t) and centroid
// (relative to t's generator); a ring build records rings in a second pool and no face geometry. A cell whose
// pool chunk is full is face_overflow, and the construct functions build again with a pool of needed() entries.
// Used by: construct*, tet_start::build (rings), Mesh::clip (faces).
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
    // Face rings, size 0 = off.
    Kokkos::View<int*, MemorySpace> rings;
    Kokkos::View<long long*, MemorySpace> ring_start;
    Kokkos::View<unsigned*, MemorySpace> ring_used;  // per chunk
    // Optional diagnostics, size 0 = off: |sum of area vectors| / sum of |area|, and work per cell (batches of grid
    // cells walked, candidates clipped, cuts).
    Kokkos::View<double*, MemorySpace> closure;
    Kokkos::View<int* [3], MemorySpace> work;

    Cells() = default;

    // Cells of n particles with owned faces, room for faces_per_cell per built cell on average (about 8 are needed) and
    // a part-used block per warp; built: cells to build (all by default, fewer for a local rebuild).
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
// Used by: construct_pass, Mesh::clip.
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
// Used by: claim_rings, claim_faces, the warp kernel, polygon_clip::construct_pass.
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


// A particle that may cut the cell, sorted by distance.
// Used by: construct_pass.
struct Candidate {
    double distance2;  // sort key and reach
    int slot;
    std::int8_t image[3];  // periodic image of the slot, in box lengths
};

// Which of the selected cells a pass builds: all, those not ok, or those
// that overflowed the usual capacities.
// Used by: construct_cells, construct_huge.
enum class Pass { all, failed, overflowed };

// A thread's cell: on its stack, the host's huge ones (over a megabyte) on
// the heap, as an OpenMP thread may have little more than that.
template <class Cell, bool on_heap = (sizeof(Cell) > (1 << 20))>
struct Held {
    Cell cell;
};
template <class Cell>
struct Held<Cell, true> {
    Cell* held;
    Cell& cell;
    KOKKOS_FUNCTION static Cell* make() {
        Cell* made = nullptr;
        KOKKOS_IF_ON_HOST((made = new Cell;))
        return made;
    }
    KOKKOS_FUNCTION Held() : held(make()), cell(*held) {}
    KOKKOS_FUNCTION ~Held() { KOKKOS_IF_ON_HOST((delete held;)) }
    Held(const Held&) = delete;
    Held& operator=(const Held&) = delete;
};

// The per-thread kernel, with cells of type Cell (RoundedCell or
// ExactCell): the walk (see Stencil), then the face walk that writes the
// owned faces or rings and integrates the geometry.
// Used by: construct_cells, construct_huge.
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
            const int id = grid.order(t);
            const int before = out.status(id);
            // A full pool stays full: those cells wait for a bigger one.
            if ((pass == Pass::failed && (before == ok || before == face_overflow)) ||
                (pass == Pass::overflowed && before != vertex_overflow && before != plane_overflow))
                return;
            const Vec3 p{positions(t, 0), positions(t, 1), positions(t, 2)};
            ep::i64 self[3] = {0, 0, 0};
            if (Cell::exact)
                for (int k = 0; k < 3; ++k) self[k] = ep::snap(positions(t, k), box, k);

            Held<Cell> holder;
            Cell& cell = holder.cell;
            start_cube(cell, half, unit, n);
            double reach2 = cell.reach2(unit);
            int status = ok;
            bool certified = false;
            Candidate buffer[ring_buffer];
            int buffered = 0, batches = 0, clipped = 0, cuts = 0;

            // Clip with the buffered particles nearest-first, until the rest are too far to cut.
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
                    // Exact cuts break ties by particle id; rounded cells keep the slot until the walk is done (one
                    // load fewer per cut).
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
            // NOLINTNEXTLINE(bugprone-infinite-loop): `status` is set inside the loop body (by the lambdas above and below).
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
                if (level == 2) break;
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

            // A block for the owned faces: one entry per owned plane, or with rings a neighbour, a length and one entry
            // per vertex each.
            long long ring_at = 0;
            int face_at = 0;
            if (status == ok) {
                compact_planes(cell);
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

            // Faces: walk each plane's vertices in order, integrating the face area vector and centroid and the cell
            // volume and centroid.
            int faces = 0, owned = 0, ring_used = 0;
            double volume = 0, area_magnitude = 0;
            Vec3 moment{0, 0, 0}, area_sum{0, 0, 0};
            const double sliver = 64 * Kokkos::Experimental::epsilon_v<double> * cell.radius2_in_box(unit);
            std::int16_t first_vertex[Cell::planes_capacity];
            if (status == ok) {
                for (int plane = 0; plane < cell.planes; ++plane) first_vertex[plane] = -1;
                for (int v = 0; v < cell.vertices; ++v)
                    if (live(cell, v))
                        for (int k = 0; k < 3; ++k) first_vertex[cell.corner[v][k]] = static_cast<std::int16_t>(v);
            }
            for (int plane = box_planes; plane < cell.planes && status == ok; ++plane) {
                const int start = first_vertex[plane];
                if (start < 0) continue;

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
                // Faces below rounding level (cospherical generators) carry no flux and are left out of the face
                // geometry; exact rings keep every face (topology).
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

// A host copy of view `from`, of type To (the host's layout); an empty view
// stays empty.
// Used by: construct_huge.
template <class To, class From>
To host_copy(const From& from) {
    if (from.data() == nullptr) return To{};
    const auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, from);
    To to(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(from.label())), from.extent(0));
    Kokkos::deep_copy(to, mirror);
    return to;
}

// Device view `to` = host view `from`.
// Used by: construct_huge.
template <class From, class To>
void copy_back(const From& from, const To& to) {
    if (to.data() == nullptr) return;
    const auto mirror = Kokkos::create_mirror_view(Kokkos::HostSpace{}, to);
    Kokkos::deep_copy(mirror, from);
    Kokkos::deep_copy(to, mirror);
}

// The cells that overflowed big exact cells: exact cells of huge capacity, on
// the host, with the grid and the results copied there and back unless they
// are there already. A rare path: its cost per cell grows as faces x vertices.
// Used by: construct_cells.
template <class ExecSpace>
void construct_huge(const Grid<typename ExecSpace::memory_space>& grid,
                    const Cells<typename ExecSpace::memory_space>& out,
                    const Selection<typename ExecSpace::memory_space>& selection) {
    using Memory = typename ExecSpace::memory_space;
    using Host = Kokkos::DefaultHostExecutionSpace;
    using HostMemory = typename Host::memory_space;
    using Huge = ExactCell<huge_planes, huge_vertices>;
    if constexpr (std::is_same_v<Memory, HostMemory>) {
        construct_pass<Huge>(Host{}, grid, make_stencil(Host{}), out, Pass::overflowed, selection);
    } else {
        Grid<HostMemory> g;
        g.positions = host_copy<decltype(g.positions)>(grid.positions);
        g.order = host_copy<decltype(g.order)>(grid.order);
        g.cell_offsets = host_copy<decltype(g.cell_offsets)>(grid.cell_offsets);
        g.fine = host_copy<decltype(g.fine)>(grid.fine);
        g.level = host_copy<decltype(g.level)>(grid.level);
        g.n = grid.n;
        g.levels = grid.levels;
        g.box = grid.box;
        g.narrowest = grid.narrowest;
        g.fewest = grid.fewest;
        for (int a = 0; a < 3; ++a) {
            g.tiles[a] = grid.tiles[a];
            g.width[a] = grid.width[a];
        }
        Cells<HostMemory> c;
        c.status = host_copy<decltype(c.status)>(out.status);
        c.face_count = host_copy<decltype(c.face_count)>(out.face_count);
        c.owned_count = host_copy<decltype(c.owned_count)>(out.owned_count);
        c.volume = host_copy<decltype(c.volume)>(out.volume);
        c.centroid = host_copy<decltype(c.centroid)>(out.centroid);
        c.face_start = host_copy<decltype(c.face_start)>(out.face_start);
        c.neighbour = host_copy<decltype(c.neighbour)>(out.neighbour);
        c.area = host_copy<decltype(c.area)>(out.area);
        c.face_centroid = host_copy<decltype(c.face_centroid)>(out.face_centroid);
        c.face_used = host_copy<decltype(c.face_used)>(out.face_used);
        c.rings = host_copy<decltype(c.rings)>(out.rings);
        c.ring_start = host_copy<decltype(c.ring_start)>(out.ring_start);
        c.ring_used = host_copy<decltype(c.ring_used)>(out.ring_used);
        const Selection<HostMemory> s{host_copy<Kokkos::View<int*, HostMemory>>(selection.slots),
                                      host_copy<Kokkos::View<char*, HostMemory>>(selection.listed)};
        construct_pass<Huge>(Host{}, g, make_stencil(Host{}), c, Pass::overflowed, s);
        copy_back(c.status, out.status);
        copy_back(c.face_count, out.face_count);
        copy_back(c.owned_count, out.owned_count);
        copy_back(c.volume, out.volume);
        copy_back(c.centroid, out.centroid);
        copy_back(c.face_start, out.face_start);
        copy_back(c.neighbour, out.neighbour);
        copy_back(c.area, out.area);
        copy_back(c.face_centroid, out.face_centroid);
        copy_back(c.face_used, out.face_used);
        copy_back(c.rings, out.rings);
        copy_back(c.ring_start, out.ring_start);
        copy_back(c.ring_used, out.ring_used);
    }
}

// Build the cells of `pass` with cells of kind Cell (RoundedCell or
// ExactCell), then those that overflowed the usual capacities once more
// with big ones, and exact cells that overflowed those with huge ones.
// Used by: construct, tet_start::build.
template <template <int, int> class Cell, class ExecSpace>
void construct_cells(const ExecSpace& exec, const Grid<typename ExecSpace::memory_space>& grid,
                     const Stencil<typename ExecSpace::memory_space>& stencil,
                     const Cells<typename ExecSpace::memory_space>& out, Pass pass,
                     const Selection<typename ExecSpace::memory_space>& selection = {}) {
    construct_pass<Cell<max_planes, max_vertices>>(exec, grid, stencil, out, pass, selection);
    const auto status = out.status;
    const auto order = grid.order;
    auto overflowed = [&] {
        int count = 0;
        Kokkos::parallel_reduce(
            "shell_voronoi::count_overflows", Kokkos::RangePolicy<ExecSpace>(exec, 0, selection.count(grid.n)),
            KOKKOS_LAMBDA(int i, int& sum) {
                const int s = status(order(selection.slot(i)));
                sum += s == vertex_overflow || s == plane_overflow;
            },
            count);
        return count > 0;
    };
    if (!overflowed()) return;
    construct_pass<Cell<big_planes, big_vertices>>(exec, grid, stencil, out, Pass::overflowed, selection);
    if constexpr (Cell<big_planes, big_vertices>::exact)
        if (overflowed()) construct_huge<ExecSpace>(grid, out, selection);
}

// Run build() on an empty pool; if the pool turned out too small (cells
// face_overflow), again on one that fits.
// Used by: construct, construct_warp.
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
// Used by: construct_warp (non-CUDA), tests.
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
