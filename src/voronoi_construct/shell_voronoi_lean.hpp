#pragma once

#include "shell_voronoi.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

// Thread-per-cell construction with a compact polyhedron in shared memory.
//
// Same algorithm and output as shell_voronoi::construct, laid out for speed
// and a small footprint:
//  - Vertices store their position and topology (three planes, three
//    neighbouring vertices); planes store only the neighbour's slot. A cut
//    creates each new vertex by interpolating along the edge it crosses, so no
//    plane equation is kept.
//  - Each thread's polyhedron is interleaved with its block neighbours in
//    shared memory (element k of thread r at k * stride + r), so threads
//    touching the same element index never share a bank.
//  - Neighbouring grid cells are visited in the order of their smallest
//    possible distance (a fixed stencil) and the walk stops at the first one
//    beyond 2R.
// About 1.2 KB per cell in float, 1.8 KB in double. Cells that overflow the
// capacities are rebuilt in double precision by the per-thread kernel.
namespace shell_voronoi {
namespace lean {

constexpr int max_vertices = 48;
constexpr int max_planes = 32;  // box planes included; unused planes are dropped when full
constexpr std::uint8_t none = 255;

// Grid-cell offsets (4 bits per axis, biased by 8) sorted by the squared
// smallest distance any point of the home cell can have to them.
template <class MemorySpace>
struct Stencil {
    Kokkos::View<int*, MemorySpace> offsets;
    Kokkos::View<float*, MemorySpace> bound2;
    double reach = 0;  // cells outside the stencil are at least this far
};

template <class ExecSpace>
Stencil<typename ExecSpace::memory_space> make_stencil(const ExecSpace& exec, int cells_per_side,
                                                       double cell_width) {
    const int reach = std::min((cells_per_side - 1) / 2, 4);
    std::vector<std::pair<float, int>> entries;
    auto beyond = [](int d) { return std::max(std::abs(d) - 1, 0); };
    for (int dx = -reach; dx <= reach; ++dx)
        for (int dy = -reach; dy <= reach; ++dy)
            for (int dz = -reach; dz <= reach; ++dz) {
                // Only cells up to `reach` widths away: beyond that the cube's
                // corners alone would claim distances the stencil cannot vouch for.
                if (beyond(dx) * beyond(dx) + beyond(dy) * beyond(dy) + beyond(dz) * beyond(dz) > reach * reach)
                    continue;
                const double bound = cell_width * std::sqrt(double(beyond(dx) * beyond(dx) +
                                                                   beyond(dy) * beyond(dy) +
                                                                   beyond(dz) * beyond(dz)));
                // Round down so float storage never overstates a distance.
                entries.push_back({std::nextafter(float(bound * bound), 0.0f),
                                   (dx + 8) | (dy + 8) << 4 | (dz + 8) << 8});
            }
    std::sort(entries.begin(), entries.end());
    Stencil<typename ExecSpace::memory_space> s{
        Kokkos::View<int*, typename ExecSpace::memory_space>("stencil_offsets", entries.size()),
        Kokkos::View<float*, typename ExecSpace::memory_space>("stencil_bound2", entries.size()),
        reach * cell_width};
    auto ho = Kokkos::create_mirror_view(s.offsets);
    auto hb = Kokkos::create_mirror_view(s.bound2);
    for (std::size_t i = 0; i < entries.size(); ++i) {
        ho(i) = entries[i].second;
        hb(i) = entries[i].first;
    }
    Kokkos::deep_copy(exec, s.offsets, ho);
    Kokkos::deep_copy(exec, s.bound2, hb);
    return s;
}

// Topology of a vertex in one word: corner k (a plane) in byte k, the vertex
// across the edge from corner k to corner k + 1 in byte 3 + k.
KOKKOS_INLINE_FUNCTION int corner(std::uint64_t t, int k) { return int(t >> (8 * k)) & 255; }
KOKKOS_INLINE_FUNCTION int adjacent(std::uint64_t t, int k) { return int(t >> (24 + 8 * k)) & 255; }
KOKKOS_INLINE_FUNCTION std::uint64_t with_adjacent(std::uint64_t t, int k, int v) {
    const int shift = 24 + 8 * k;
    return (t & ~(std::uint64_t(255) << shift)) | std::uint64_t(v) << shift;
}
KOKKOS_INLINE_FUNCTION std::uint64_t topology(int c0, int c1, int c2, int a0, int a1, int a2) {
    return std::uint64_t(c0) | std::uint64_t(c1) << 8 | std::uint64_t(c2) << 16 | std::uint64_t(a0) << 24 |
           std::uint64_t(a1) << 32 | std::uint64_t(a2) << 40;
}

// One thread's polyhedron: element k lives at base[k * stride].
template <class Real>
struct Polyhedron {
    Real* x;
    Real* y;
    Real* z;
    std::uint64_t* links;
    int* neighbour;
    std::uint8_t* starts;  // per plane: new vertex whose crossing edge starts there
    std::uint8_t* ends;    // ... and ends there
    int stride;

    static constexpr std::size_t bytes = max_vertices * (3 * sizeof(Real) + sizeof(std::uint64_t)) +
                                         max_planes * (sizeof(int) + 2);

    // Carve one thread's slice out of a block's shared memory.
    KOKKOS_INLINE_FUNCTION Polyhedron(char* block, int threads, int rank) : stride(threads) {
        char* p = block;
        auto take = [&](auto*& array, std::size_t element, int count) {
            array = reinterpret_cast<std::remove_reference_t<decltype(array)>>(p) + rank;
            p += element * count * threads;
        };
        take(links, sizeof(std::uint64_t), max_vertices);
        take(x, sizeof(Real), max_vertices);
        take(y, sizeof(Real), max_vertices);
        take(z, sizeof(Real), max_vertices);
        take(neighbour, sizeof(int), max_planes);
        take(starts, 1, max_planes);
        take(ends, 1, max_planes);
    }

    KOKKOS_INLINE_FUNCTION Vec3<Real> vertex(int v) const { return {x[v * stride], y[v * stride], z[v * stride]}; }
    KOKKOS_INLINE_FUNCTION void set_vertex(int v, Vec3<Real> p) const {
        x[v * stride] = p.x;
        y[v * stride] = p.y;
        z[v * stride] = p.z;
    }
    KOKKOS_INLINE_FUNCTION std::uint64_t& link(int v) const { return links[v * stride]; }
    KOKKOS_INLINE_FUNCTION int& plane_neighbour(int p) const { return neighbour[p * stride]; }
    KOKKOS_INLINE_FUNCTION std::uint8_t& start(int p) const { return starts[p * stride]; }
    KOKKOS_INLINE_FUNCTION std::uint8_t& end(int p) const { return ends[p * stride]; }
};

KOKKOS_INLINE_FUNCTION int lowest_bit(std::uint64_t bits) { return Kokkos::Experimental::countr_zero_builtin(bits); }
KOKKOS_INLINE_FUNCTION int bit_count(std::uint64_t bits) { return Kokkos::Experimental::popcount_builtin(bits); }

template <class Real>
KOKKOS_INLINE_FUNCTION Real squared_radius(const Polyhedron<Real>& c, int vertices) {
    Real r2 = 0;
    for (int v = 0; v < vertices; ++v) {
        const Vec3<Real> x = c.vertex(v);
        r2 = Kokkos::max(r2, dot(x, x));
    }
    return r2;
}

// Cube of half-width h; corners and adjacency follow local_voronoi.hpp.
template <class Real>
KOKKOS_INLINE_FUNCTION void start_cube(const Polyhedron<Real>& c, Real h) {
    constexpr int corners[8][3] = {{0, 2, 4}, {1, 4, 2}, {0, 4, 3}, {1, 3, 4},
                                   {0, 5, 2}, {1, 2, 5}, {0, 3, 5}, {1, 5, 3}};
    constexpr int across[8][3] = {{4, 1, 2}, {3, 0, 5}, {0, 3, 6}, {7, 2, 1},
                                  {6, 5, 0}, {1, 4, 7}, {2, 7, 4}, {5, 6, 3}};
    for (int v = 0; v < 8; ++v) {
        c.set_vertex(v, {v & 1 ? -h : h, v & 2 ? -h : h, v & 4 ? -h : h});
        c.link(v) = topology(corners[v][0], corners[v][1], corners[v][2], across[v][0], across[v][1],
                             across[v][2]);
    }
    for (int p = 0; p < box_planes; ++p) c.plane_neighbour(p) = -1;
}

// Drop planes no vertex uses any more and renumber the rest; box planes keep
// their indices. Returns the number of planes freed.
template <class Real>
KOKKOS_INLINE_FUNCTION int compact_planes(const Polyhedron<Real>& c, int vertices, int& planes) {
    std::uint64_t used = (std::uint64_t(1) << box_planes) - 1;
    for (int v = 0; v < vertices; ++v)
        for (int k = 0; k < 3; ++k) used |= std::uint64_t(1) << corner(c.link(v), k);
    std::uint8_t renumber[max_planes];
    int kept = 0;
    for (int p = 0; p < planes; ++p) {
        if (!(used >> p & 1)) continue;
        renumber[p] = static_cast<std::uint8_t>(kept);
        c.plane_neighbour(kept) = c.plane_neighbour(p);
        ++kept;
    }
    for (int v = 0; v < vertices; ++v) {
        const std::uint64_t t = c.link(v);
        c.link(v) = (t & ~std::uint64_t(0xffffff)) | std::uint64_t(renumber[corner(t, 0)]) |
                    std::uint64_t(renumber[corner(t, 1)]) << 8 | std::uint64_t(renumber[corner(t, 2)]) << 16;
    }
    const int freed = planes - kept;
    planes = kept;
    return freed;
}

// Clip by the bisector with a particle at offset d. Returns ok when the plane
// does not cut, or after the cut has been applied.
template <class Real>
KOKKOS_INLINE_FUNCTION int clip(const Polyhedron<Real>& c, int& vertices, int& planes, Real& radius2,
                                Vec3<Real> d, int neighbour) {
    const Real offset = Real(0.5) * dot(d, d);
    const Real tolerance = Real(64) * Kokkos::Experimental::epsilon_v<Real> * Kokkos::sqrt(dot(d, d) * radius2);
    std::uint64_t removed = 0;
    for (int v = 0; v < vertices; ++v)
        if (dot(d, c.vertex(v)) - offset > tolerance) removed |= std::uint64_t(1) << v;
    if (removed == 0) return ok;
    if (planes == max_planes && compact_planes(c, vertices, planes) == 0) return plane_overflow;
    const int plane = planes;

    // New vertices, one per edge from a removed vertex to a kept one. A
    // removed vertex's slot is reused once its data has been read.
    std::uint64_t free_slots = 0, created = 0;
    int end = vertices;
    for (std::uint64_t pending = removed; pending; pending &= pending - 1) {
        const int v = lowest_bit(pending);
        const std::uint64_t tv = c.link(v);
        const Vec3<Real> xv = c.vertex(v);
        const Real sv = dot(d, xv) - offset;
        free_slots |= std::uint64_t(1) << v;
        for (int e = 0; e < 3; ++e) {
            const int u = adjacent(tv, e);
            if (removed >> u & 1) continue;
            int slot;
            if (free_slots) {
                slot = lowest_bit(free_slots);
                free_slots &= free_slots - 1;
            } else {
                if (end == max_vertices) return vertex_overflow;
                slot = end++;
            }
            const Vec3<Real> xu = c.vertex(u);
            const Real su = dot(d, xu) - offset;
            c.set_vertex(slot, xv + (xu - xv) * (sv / (sv - su)));
            const int a = corner(tv, e), b = corner(tv, (e + 1) % 3);
            c.link(slot) = topology(a, b, plane, u, 0, 0);
            std::uint64_t tu = c.link(u);
            for (int k = 0; k < 3; ++k)
                if (adjacent(tu, k) == v) tu = with_adjacent(tu, k, slot);
            c.link(u) = tu;
            c.start(a) = static_cast<std::uint8_t>(slot);
            c.end(b) = static_cast<std::uint8_t>(slot);
            created |= std::uint64_t(1) << slot;
        }
    }
    // Around the new face: across (b, plane) is the new vertex whose crossing
    // edge starts at b, across (plane, a) the one whose edge ends at a.
    // In a degenerate cut a link can be missing; then the cell goes to the
    // double-precision fallback instead of following a stale index.
    for (std::uint64_t bits = created; bits; bits &= bits - 1) {
        const int v = lowest_bit(bits);
        const std::uint64_t t = c.link(v);
        const int next = c.start(corner(t, 1)), previous = c.end(corner(t, 0));
        if (next >= max_vertices || previous >= max_vertices || !(created >> next & 1) ||
            !(created >> previous & 1))
            return degenerate;
        c.link(v) = with_adjacent(with_adjacent(t, 1, next), 2, previous);
    }

    // Close the remaining holes by moving the last vertices into them.
    const int count = end - bit_count(free_slots);
    if (count < 4) return degenerate;
    for (std::uint64_t holes = free_slots & ((std::uint64_t(1) << count) - 1); holes; holes &= holes - 1) {
        const int h = lowest_bit(holes);
        do --end;
        while (free_slots >> end & 1);
        const std::uint64_t t = c.link(end);
        c.set_vertex(h, c.vertex(end));
        c.link(h) = t;
        for (int k = 0; k < 3; ++k) {
            const int n = adjacent(t, k);
            std::uint64_t tn = c.link(n);
            for (int j = 0; j < 3; ++j)
                if (adjacent(tn, j) == end) tn = with_adjacent(tn, j, h);
            c.link(n) = tn;
        }
    }
    vertices = count;
    c.plane_neighbour(plane) = neighbour;
    planes = plane + 1;
    radius2 = squared_radius(c, vertices);
    return ok;
}

// Warm start: each cell is first clipped by its natural neighbours from the
// previous build (particle IDs), then certified by the usual walk. order maps
// slot -> particle and rank particle -> slot for the current grid. Empty
// views mean a cold build.
template <class MemorySpace>
struct Seeds {
    Kokkos::View<int**, MemorySpace> ids;
    Kokkos::View<int*, MemorySpace> count;
    Kokkos::View<int*, MemorySpace> order;
    Kokkos::View<int*, MemorySpace> rank;
};

template <class Real, class ExecSpace>
struct Kernel {
    using MemorySpace = typename ExecSpace::memory_space;
    using Member = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
    Grid<MemorySpace> grid;
    Cells<MemorySpace> out;
    Stencil<MemorySpace> stencil;
    Seeds<MemorySpace> seeds;
    Real half_width;
    int threads;
    bool certify;  // false only for timing experiments: skips the walk

    KOKKOS_FUNCTION void operator()(const Member& member) const {
        char* block = static_cast<char*>(member.team_shmem().get_shmem(threads * Polyhedron<Real>::bytes));
        const int t = member.league_rank() * threads + member.team_rank();
        if (t >= grid.n) return;
        const Polyhedron<Real> c(block, threads, member.team_rank());
        const auto& positions = grid.positions;
        const auto& cell_offsets = grid.cell_offsets;
        const int side = grid.cells_per_side;
        const double width = grid.cell_width, box = grid.box;
        const int stencil_size = static_cast<int>(stencil.offsets.extent(0));

        const Vec3<double> p{positions(t, 0), positions(t, 1), positions(t, 2)};
        const double q[3] = {p.x, p.y, p.z};
        int home[3];
        for (int a = 0; a < 3; ++a) home[a] = Kokkos::clamp(static_cast<int>(q[a] / width), 0, side - 1);

        start_cube(c, half_width);
        int vertices = 8, planes = box_planes, status = ok, visited = 0, clipped = 0, cuts = 0;
        Real radius2 = 3 * half_width * half_width;
        bool certified = false;

        if (seeds.count.extent(0) > 0) {
            const int particle = seeds.order(t);
            for (int j = 0; j < seeds.count(particle) && status == ok; ++j) {
                const int s = seeds.rank(seeds.ids(particle, j));
                if (s == t) continue;
                Vec3<double> d{positions(s, 0) - p.x, positions(s, 1) - p.y, positions(s, 2) - p.z};
                d = {d.x - box * Kokkos::nearbyint(d.x / box), d.y - box * Kokkos::nearbyint(d.y / box),
                     d.z - box * Kokkos::nearbyint(d.z / box)};
                const int before = planes;
                status = clip(c, vertices, planes, radius2, convert<Real>(d), s);
                ++clipped;
                cuts += planes != before || status != ok;
            }
        }

        // Candidates wait in a small buffer sorted by distance. Before a grid
        // cell is visited, every waiting candidate nearer than that cell's
        // smallest possible distance is clipped, so candidates are clipped
        // in exact distance order whatever order the grid cells come in.
        struct Pending {
            float distance2;
            int slot;
            int shift;  // periodic image, (sx + 1) + 3 (sy + 1) + 9 (sz + 1)
        };
        constexpr int pending_capacity = 16;
        Pending pending[pending_capacity];
        int waiting = 0;
        auto clip_nearest = [&]() {
            const Pending next = pending[0];
            for (int i = 1; i < waiting; ++i) pending[i - 1] = pending[i];
            --waiting;
            if (double(next.distance2) >= 4.0 * static_cast<double>(radius2)) {
                waiting = 0;  // sorted: none of the rest can cut either
                return;
            }
            const int sx = next.shift % 3 - 1, sy = next.shift / 3 % 3 - 1, sz = next.shift / 9 - 1;
            const Vec3<double> d{positions(next.slot, 0) + sx * box - p.x, positions(next.slot, 1) + sy * box - p.y,
                                 positions(next.slot, 2) + sz * box - p.z};
            const int before = planes;
            status = clip(c, vertices, planes, radius2, convert<Real>(d), next.slot);
            ++clipped;
            cuts += planes != before || status != ok;
        };
        auto clip_nearer_than = [&](double limit2) {
            while (waiting > 0 && status == ok && double(pending[0].distance2) <= limit2) clip_nearest();
        };

        for (int e = 0; e < (certify ? stencil_size : 0) && status == ok; ++e) {
            const double bound2 = double(stencil.bound2(e));
            clip_nearer_than(bound2);
            if (status != ok) break;
            const double reach2 = 4.0 * static_cast<double>(radius2);
            if (bound2 >= reach2) {
                certified = true;  // waiting candidates are beyond bound2, so beyond 2R
                break;
            }
            const int code = stencil.offsets(e);
            const int u[3] = {home[0] + (code & 15) - 8, home[1] + ((code >> 4) & 15) - 8,
                              home[2] + ((code >> 8) & 15) - 8};
            double gap2 = 0;
            int key = 0, shift = 0, scale = 1;
            double offset[3];
            for (int a = 0; a < 3; ++a) {
                const double g = Kokkos::max(0.0, Kokkos::max(u[a] * width - q[a], q[a] - (u[a] + 1) * width));
                gap2 += g * g;
                key = key * side + (u[a] + side) % side;
                const int image = u[a] < 0 ? -1 : u[a] >= side ? 1 : 0;
                offset[a] = image * box;
                shift += (image + 1) * scale;
                scale *= 3;
            }
            if (gap2 >= reach2) continue;
            ++visited;
            key = grid.rank(key);
            const int last = cell_offsets(key + 1);
            for (int s = cell_offsets(key); s < last && status == ok; ++s) {
                if (s == t) continue;
                const Vec3<double> d{positions(s, 0) + offset[0] - p.x, positions(s, 1) + offset[1] - p.y,
                                     positions(s, 2) + offset[2] - p.z};
                const double d2 = dot(d, d);
                if (d2 >= 4.0 * static_cast<double>(radius2)) continue;
                if (waiting == pending_capacity) clip_nearest();
                int at = waiting++;
                const float key2 = static_cast<float>(d2);
                for (; at > 0 && pending[at - 1].distance2 > key2; --at) pending[at] = pending[at - 1];
                pending[at] = {key2, s, shift};
            }
        }
        // The stencil ran out before the bound passed 2R: clip what waits.
        while (waiting > 0 && status == ok) clip_nearest();
        if (status == ok && !certified && stencil.reach * stencil.reach >= 4.0 * static_cast<double>(radius2))
            certified = true;
        if (status == ok && !certified && certify) status = uncertified;
        for (int v = 0; v < vertices && status == ok; ++v)
            for (int k = 0; k < 3; ++k)
                if (corner(c.link(v), k) < box_planes) status = uncertified;

        // Faces: start each plane's walk at one of its vertices (start()
        // is reused as that table), then follow the vertex adjacency.
        for (int p = 0; p < planes; ++p) c.start(p) = none;
        for (int v = 0; v < vertices; ++v)
            for (int k = 0; k < 3; ++k) c.start(corner(c.link(v), k)) = static_cast<std::uint8_t>(v);

        int faces = 0, owned = 0, ring_used = 0;
        const bool with_rings = out.rings.extent(0) > 0;
        const int ring_capacity = with_rings ? static_cast<int>(out.rings.extent(1)) : 0;
        double volume = 0, area_magnitude = 0;
        Vec3<double> moment{0, 0, 0}, area_sum{0, 0, 0};
        const double sliver = 64 * Kokkos::Experimental::epsilon_v<Real> * static_cast<double>(radius2);
        for (int plane = box_planes; plane < planes && status == ok; ++plane) {
            const int first_vertex = c.start(plane);
            if (first_vertex == none) continue;
            const Vec3<double> first = convert<double>(c.vertex(first_vertex));
            Vec3<double> previous = first, area{0, 0, 0}, face_moment{0, 0, 0};
            double weight = 0;
            int current = first_vertex, steps = 0, ring_length = 0;
            do {
                const std::uint64_t link = c.link(current);
                int at = 0;
                while (at < 3 && corner(link, at) != plane) ++at;
                if (at == 3 || ++steps > max_vertices) {
                    status = degenerate;
                    break;
                }
                const int next = adjacent(link, (at + 2) % 3);
                if (next >= vertices) {
                    status = degenerate;
                    break;
                }
                // The plane shared with the next vertex bounds the face there.
                if (with_rings && ring_used + 1 + ring_length < ring_capacity)
                    out.rings(t, ring_used + 1 + ring_length) = c.plane_neighbour(corner(link, (at + 2) % 3));
                ++ring_length;
                const Vec3<double> x = convert<double>(c.vertex(next));
                const Vec3<double> twice = cross(previous - first, x - first);
                area = area + twice * 0.5;
                const double w = Kokkos::sqrt(dot(twice, twice));
                weight += w;
                face_moment = face_moment + (first + previous + x) * (w / 3.0);
                const double tetra = dot(first, cross(previous, x)) / 6.0;
                volume += tetra;
                moment = moment + (first + previous + x) * (tetra / 4.0);
                previous = x;
                current = next;
            } while (current != first_vertex);
            if (status != ok) break;
            area_sum = area_sum + area;
            const double magnitude = Kokkos::sqrt(dot(area, area));
            if (magnitude <= sliver) continue;
            const int neighbour = c.plane_neighbour(plane);
            if (out.all_neighbours.extent(0) > 0) {
                if (faces == static_cast<int>(out.all_neighbours.extent(1))) {
                    status = face_overflow;
                    break;
                }
                out.all_neighbours(t, faces) = neighbour;
            }
            ++faces;
            area_magnitude += magnitude;
            if (neighbour > t) {
                if (owned == max_owned_faces) {
                    status = face_overflow;
                    break;
                }
                const Vec3<double> centroid = weight > 0 ? face_moment * (1.0 / weight) : first;
                if (with_rings) {
                    if (ring_used + 1 + ring_length > ring_capacity) {
                        status = face_overflow;
                        break;
                    }
                    out.rings(t, ring_used) = ring_length;
                    ring_used += 1 + ring_length;
                }
                out.neighbour(t, owned) = neighbour;
                for (int a = 0; a < 3; ++a) {
                    out.area(t, owned, a) = a == 0 ? area.x : a == 1 ? area.y : area.z;
                    out.face_centroid(t, owned, a) = a == 0 ? centroid.x : a == 1 ? centroid.y : centroid.z;
                }
                ++owned;
            }
        }
        if (status == ok && !(volume > 0)) status = degenerate;

        const bool good = status == ok;
        out.status(t) = status;
        out.face_count(t) = good ? faces : 0;
        out.owned_count(t) = good ? owned : 0;
        out.volume(t) = good ? volume : 0.0;
        out.centroid(t, 0) = good ? moment.x / volume : 0.0;
        out.centroid(t, 1) = good ? moment.y / volume : 0.0;
        out.centroid(t, 2) = good ? moment.z / volume : 0.0;
        if (out.closure.extent(0) > 0) {
            out.closure(t) = good ? Kokkos::sqrt(dot(area_sum, area_sum)) / area_magnitude : 0.0;
            out.work(t, 0) = visited;
            out.work(t, 1) = clipped;
            out.work(t, 2) = cuts;
        }
    }
};

}  // namespace lean

// Build every cell with one thread per cell and a shared-memory polyhedron,
// then rebuild failed cells in double with the per-thread kernel. GPU blocks
// hold one warp; host backends use one thread per team.
template <class Real, class ExecSpace>
void construct_lean(const ExecSpace& exec, const Grid<typename ExecSpace::memory_space>& grid,
                    const Cells<typename ExecSpace::memory_space>& out,
                    const lean::Stencil<typename ExecSpace::memory_space>& stencil,
                    const lean::Seeds<typename ExecSpace::memory_space>& seeds = {},
                    bool certify = true) {
    const int side = grid.cells_per_side;
    if (grid.n <= 0 || side < 3 || out.status.extent(0) != static_cast<std::size_t>(grid.n))
        throw std::invalid_argument("shell_voronoi::construct_lean: invalid grid or output size");
    // One warp per block when it fits the 48 KB block scratch limit (float),
    // otherwise the largest power of two that does (double).
    const bool host = Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename ExecSpace::memory_space>::accessible;
    int threads = host ? 1 : 32;
    while (threads > 1 && threads * lean::Polyhedron<Real>::bytes > 48 * 1024) threads /= 2;
    const int max_ring = (side - 1) / 2;
    const Real half_width = static_cast<Real>(std::min(0.25 * grid.box, (max_ring + 1) * grid.cell_width));
    Kokkos::TeamPolicy<ExecSpace> policy(exec, (grid.n + threads - 1) / threads, threads);
    policy.set_scratch_size(0, Kokkos::PerTeam(threads * lean::Polyhedron<Real>::bytes));
    Kokkos::parallel_for("shell_voronoi::construct_lean", policy,
                         lean::Kernel<Real, ExecSpace>{grid, out, stencil, seeds, half_width, threads, certify});
    if (certify) construct<double>(exec, grid, out, /*only_failed=*/true);
}

}  // namespace shell_voronoi
