#pragma once

#include "exact_predicates.hpp"
#include "shell_grid.hpp"
#include "shell_limits.hpp"

#include <Kokkos_Core.hpp>

#include <cstdint>

// Clipping stage, the cell under construction (3D): RoundedCell and ExactCell and the clip step on them, used by
// construct_pass in shell_voronoi.hpp and by the warp kernel.
namespace shell_voronoi::detail {

namespace ep = exact_predicates;

// The starting cube must lie well inside half the shortest side so that
// every candidate that can cut the cell is unique in the periodic image.
inline double start_half_width(const ep::Box& box) { return 0.25 * box.shortest(); }

// A cell under construction, in double relative to the generator: a vertex is cut when it lies beyond the
// bisector by more than a rounding tolerance. Planes and Vertices are capacities; larger cells are built
// again with bigger ones. Layout of both cell kinds: docs/design.md.
// Used by: construct_cells (portable and warp-equivalent path).
template <int Planes, int Vertices>
struct RoundedCell {
    static_assert(Planes <= 255, "plane indices are bytes");
    using Index = std::uint8_t;
    static constexpr int planes_capacity = Planes, vertices_capacity = Vertices;
    static constexpr bool exact = false;
    static constexpr Index removed = 255;  // corner[v][0] of a removed vertex: no plane index reaches it
    Vec3 normal[Planes];  // the cell is {x : normal . x <= offset}
    double offset[Planes];
    int id[Planes];
    Vec3 vertex[Vertices];
    Index corner[Vertices][3], adjacent[Vertices][3];
    Index free_slot[Vertices];           // stack of the removed vertices' slots
    Index starts[Planes], ends[Planes];  // a cut's new vertex whose hole edge starts / ends at a plane
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

// ExactCell: in grid steps relative to the snapped generator. Vertex v is the circumcentre -w / (2 o) of the
// generator and its three planes' particles; a particle at e is inside iff e.w + |e|^2 o has the opposite sign of o,
// decided by a filter, else by the exact perturbed in-sphere sign, so neighbouring cells agree (the start's patch is tet_start).
// Used by: construct_cells from tet_start::build (exact cells around failures) and construct(exact).
template <int Planes, int Vertices>
struct ExactCell {
    using Index = std::conditional_t<(Planes <= 255 && Vertices <= 256), std::uint8_t, std::uint16_t>;
    static_assert(Planes < 65535 && Vertices <= 32767, "plane indices fit an Index, vertex numbers an int16");
    static constexpr int planes_capacity = Planes, vertices_capacity = Vertices;
    static constexpr bool exact = true;
    static constexpr Index removed = static_cast<Index>(-1);  // no plane index reaches it
    ep::Offset offset[Planes];  // generator to the plane's particle
    int id[Planes];
    Index corner[Vertices][3], adjacent[Vertices][3];
    Index free_slot[Vertices];
    Index starts[Planes], ends[Planes];
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
// Used by: start_cube.
template <class C>
KOKKOS_INLINE_FUNCTION bool set_vertex(C& cell, int v, int a, int b, int c) {
    ep::i128 n[3];  // the vertex is n / (2 o) = -w / (2 o)
    const ep::i128 o = ep::circumcentre(cell.offset[a], cell.offset[b], cell.offset[c], n);
    if (o == 0) return false;
    for (int k = 0; k < 3; ++k) cell.w[v][k] = -double(n[k]);
    cell.o[v] = double(o);
    const double* w = cell.w[v];
    cell.r2[v] = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]) / (4 * cell.o[v] * cell.o[v]) * (1 + 1e-12);
    using Index = typename C::Index;
    cell.corner[v][0] = static_cast<Index>(a);
    cell.corner[v][1] = static_cast<Index>(b);
    cell.corner[v][2] = static_cast<Index>(c);
    return true;
}

// The vertex of a rounded cell where planes a, b, c meet; false if they do
// not meet in one point.
// Used by: clip (rounded cells).
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
// Used by: clip (exact cells).
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

template <class C>
KOKKOS_INLINE_FUNCTION bool live(const C& cell, int v) {
    return cell.corner[v][0] != C::removed;
}

// The cube of half-width h (box units) around the generator, bounded by six
// virtual particles at distance 2h (ids n..n+5). Corner order flips at
// corners of odd parity, so that every face is walked the same way round.
// Used by: construct_pass, at the start of each cell.
template <class C>
KOKKOS_INLINE_FUNCTION void start_cube(C& cell, double h, double unit, int n) {
    using Index = typename C::Index;
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
            cell.corner[c][0] = static_cast<Index>(a);
            cell.corner[c][1] = static_cast<Index>(b);
            cell.corner[c][2] = static_cast<Index>(d);
            cell.vertex[c] = {sx ? -h : h, sy ? -h : h, sz ? -h : h};
        }
    }
    // Across the edge from corner k to k + 1 lies the corner that differs along the axis of corner k + 2.
    for (int c = 0; c < 8; ++c)
        for (int k = 0; k < 3; ++k)
            cell.adjacent[c][k] = static_cast<Index>(c ^ (1 << (cell.corner[c][(k + 2) % 3] / 2)));
    for (int p = 0; p < C::planes_capacity; ++p) cell.starts[p] = cell.ends[p] = 0;
    if constexpr (!C::exact) cell.radius2 = 3 * h * h;
    cell.vertices = 8;
    cell.free_count = 0;
}

// Drop planes no vertex uses any more and renumber the rest (box planes keep
// their indices). Returns the freed count.
// Used by: clip when the planes are full, construct_pass.
template <class C>
KOKKOS_INLINE_FUNCTION int compact_planes(C& cell) {
    constexpr int capacity = C::planes_capacity;
    bool used[capacity];
    for (int p = 0; p < cell.planes; ++p) used[p] = p < box_planes;
    for (int v = 0; v < cell.vertices; ++v)
        if (live(cell, v))
            for (int k = 0; k < 3; ++k) used[cell.corner[v][k]] = true;
    typename C::Index renumber[capacity];
    int planes = 0;
    for (int p = 0; p < cell.planes; ++p) {
        if (!used[p]) continue;
        renumber[p] = static_cast<typename C::Index>(planes);
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

// Clip by the bisector of the generator `self` and particle `id` (at e in grid steps, or d in box units):
// removes the cut vertices, closes the hole with a new face. Returns ok when the plane does not cut or after the cut;
// after anything else the cell must not be read again.
// Used by: the nearest-first loop in construct_pass.
template <class C>
KOKKOS_INLINE_FUNCTION int clip(C& cell, const ep::Offset& e, const Vec3& d, int id, int self) {
    using Index = typename C::Index;
    constexpr int max_holes = C::vertices_capacity / 2;
    bool cut[C::vertices_capacity];
    int cuts = 0;
    double offset = 0, tolerance = 0;
    if constexpr (!C::exact) {
        offset = 0.5 * dot(d, d);
        tolerance = 64 * Kokkos::Experimental::epsilon_v<double> * Kokkos::sqrt(dot(d, d) * cell.radius2);
    }
    // A removed vertex of a rounded cell sits at the generator, which no bisector cuts off.
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
        Index a, b, u, back, s;
    };
    Hole hole[max_holes];
    int holes = 0;
    for (int w = 0; w < cell.vertices; ++w) {
        if (!cut[w]) continue;
        for (int k = 0; k < 3; ++k) {
            const Index u = cell.adjacent[w][k];
            if (cut[u]) continue;
            if (holes == max_holes) return vertex_overflow;
            const Index* across = cell.adjacent[u];
            hole[holes++] = {cell.corner[w][k], cell.corner[w][(k + 1) % 3], u,
                             static_cast<Index>(across[0] == w ? 0 : across[1] == w ? 1 : 2), 0};
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
    // r2: the largest squared distance of a kept vertex (removed ones count 0), then of any.
    double r2 = 0;
    for (int w = 0; w < cell.vertices; ++w) {
        if (cut[w]) {
            cell.corner[w][0] = C::removed;
            cell.free_slot[cell.free_count++] = static_cast<Index>(w);
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

    // New vertex (a, b, plane) on each hole edge, in a free slot; the edge back from u now ends at it.
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
            cell.corner[s][0] = static_cast<Index>(a);
            cell.corner[s][1] = static_cast<Index>(b);
            cell.corner[s][2] = static_cast<Index>(plane);
            r2 = Kokkos::max(r2, dot(x, x));
        }
        cell.adjacent[u][hole[h].back] = static_cast<Index>(s);
        cell.adjacent[s][0] = static_cast<Index>(u);
        cell.starts[a] = cell.ends[b] = static_cast<Index>(s);
        hole[h].s = static_cast<Index>(s);
    }
    // Across (b, plane) lies the new vertex whose hole edge starts at b, across (plane, a) the one whose hole edge ends
    // at a.
    for (int h = 0; h < holes; ++h) {
        const int a = hole[h].a, b = hole[h].b, s = hole[h].s;
        const int next = cell.starts[b], previous = cell.ends[a];
        if (cell.starts[a] != s || cell.ends[b] != s || !live(cell, next) || cell.corner[next][0] != b ||
            cell.corner[next][2] != plane || !live(cell, previous) || cell.corner[previous][1] != a ||
            cell.corner[previous][2] != plane)
            return degenerate;
        cell.adjacent[s][1] = static_cast<Index>(next);
        cell.adjacent[s][2] = static_cast<Index>(previous);
    }
    cell.radius2 = r2;
    return ok;
}

}  // namespace shell_voronoi::detail
