#pragma once

#include "polygon_clip.hpp"
#include "tet_mesh.hpp"

#include <Kokkos_Core.hpp>

#include <stdexcept>
#include <string>

// 2D start and geometry stages: from_rings turns polygon_clip's rings into triangles (Mesh<2>::rebuild); faces() computes
// the geometry (Mesh<2>::geometry). tri_repair/tri_finish keep the triangles Delaunay in between.
// Persistent 2D Delaunay triangulation on the torus: 2n triangles and 3n edges, so slots and face numbers never change.
// Triangles are counter-clockwise, lowest particle id first, so centres and faces depend on the positions alone.
namespace tri_mesh {

namespace ep = exact_predicates;
using tet_mesh::Vec2;

// Row-major view of one 6-int row per triangle (corners and links).
// Used by: Mesh<2>, tri_mesh::Mesh (corners, links).
template <class MemorySpace>
using Sixes = Kokkos::View<int* [6], Kokkos::LayoutRight, MemorySpace>;
// Row-major view of one 2D point per row (circumcentres, positions).
// Used by: tri_mesh::Mesh (centre), tri_repair::State, Mesh<2> (Positions).
template <class MemorySpace>
using Rows2 = Kokkos::View<double* [2], Kokkos::LayoutRight, MemorySpace>;

// Used by: Mesh<2>, tri_repair, tri_finish, faces.
template <class MemorySpace>
struct Triangles {
    // Row t: corners 0, 1, 2, then the apex across the edge opposite corner k (3 + k).
    Sixes<MemorySpace> corners;
    // Row t: the triangle across the edge opposite corner k, then its face number (3 + k).
    Sixes<MemorySpace> links;
    Rows2<MemorySpace> centre;  // circumcentre, unwrapped near corner 0
    Kokkos::View<int*, MemorySpace> owner, neighbour;  // of face f, owner < neighbour

    int count() const { return static_cast<int>(corners.extent(0)); }
};

namespace detail {
// Used by: from_rings.
template <class Memory>
struct RingLookup {
    polygon_clip::Rings<Memory> rings;
    Kokkos::View<const int*, Memory> first;  // the first triangle of a particle

    // The particle after q in the ring of i, or -1.
    KOKKOS_INLINE_FUNCTION int after(int i, int q) const {
        const int begin = rings.start(i), degree = rings.degree(i);
        for (int m = 0; m < degree; ++m)
            if (rings.ring(begin + m) == q) return rings.ring(begin + (m + 1 == degree ? 0 : m + 1));
        return -1;
    }
    // The triangle (a, b, c), counter-clockwise, in any rotation, or -1.
    KOKKOS_INLINE_FUNCTION int triangle(int a, int b, int c) const {
        int o = a, u = b, v = c;
        if (b < o && b < c) {
            o = b;
            u = c;
            v = a;
        } else if (c < o) {
            o = c;
            u = a;
            v = b;
        }
        const int begin = rings.start(o), degree = rings.degree(o);
        int rank = 0;
        for (int m = 0; m < degree; ++m) {
            const int q = rings.ring(begin + m), r = rings.ring(begin + (m + 1 == degree ? 0 : m + 1));
            if (q == u && r == v) return first(o) + rank;
            rank += o < q && o < r;
        }
        return -1;
    }
};
}  // namespace detail

// The triangles and faces of the rings of an exact clipping. Throws if the
// rings do not close (never for exact rings on distinct points).
// Used by: Mesh<2>::rebuild (start).
template <class ExecSpace>
Triangles<typename ExecSpace::memory_space> from_rings(const ExecSpace& exec,
                                                       const polygon_clip::Rings<typename ExecSpace::memory_space>& rings) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const int n = static_cast<int>(rings.degree.extent(0));
    Kokkos::View<int*, Memory> first(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("tri_first")), n + 1);
    int total = 0;
    Kokkos::parallel_scan(
        "tri_mesh::count", Range(exec, 0, n + 1),
        KOKKOS_LAMBDA(int i, int& sum, bool final) {
            int owned = 0;
            if (i < n) {
                const int begin = rings.start(i), degree = rings.degree(i);
                for (int m = 0; m < degree; ++m) {
                    const int q = rings.ring(begin + m), r = rings.ring(begin + (m + 1 == degree ? 0 : m + 1));
                    owned += i < q && i < r;
                }
            }
            if (final) first(i) = sum;
            sum += owned;
        },
        total);
    if (total != 2 * n)
        throw std::runtime_error("tri_mesh::from_rings: " + std::to_string(total) + " triangles of " + std::to_string(n) +
                                 " particles (2n on a torus): too few particles across the box, or duplicates");
    Triangles<Memory> mesh;
    const auto no_init = [](const char* label) { return Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)); };
    mesh.corners = Sixes<Memory>(no_init("tri_corners"), total);
    mesh.links = Sixes<Memory>(no_init("tri_links"), total);
    mesh.centre = Rows2<Memory>(no_init("tri_centre"), total);
    mesh.owner = Kokkos::View<int*, Memory>(no_init("face_owner"), 3 * n);
    mesh.neighbour = Kokkos::View<int*, Memory>(no_init("face_neighbour"), 3 * n);
    const auto corners = mesh.corners, links = mesh.links;
    const detail::RingLookup<Memory> lookup{rings, first};
    int broken = 0;
    Kokkos::parallel_reduce(
        "tri_mesh::from_rings", Range(exec, 0, n),
        KOKKOS_LAMBDA(int i, int& bad) {
            const int begin = rings.start(i), degree = rings.degree(i);
            int t = first(i);
            for (int m = 0; m < degree; ++m) {
                const int q = rings.ring(begin + m), r = rings.ring(begin + (m + 1 == degree ? 0 : m + 1));
                if (!(i < q && i < r)) continue;
                const int v[3] = {i, q, r};
                for (int k = 0; k < 3; ++k) {
                    const int a = v[(k + 1) % 3], b = v[(k + 2) % 3];
                    const int w = lookup.after(b, a);
                    const int across = w < 0 ? -1 : lookup.triangle(b, a, w);
                    corners(t, k) = v[k];
                    corners(t, 3 + k) = w;
                    links(t, k) = across;
                    bad += across < 0;
                }
                ++t;
            }
        },
        broken);
    if (broken > 0) throw std::runtime_error("tri_mesh::from_rings: rings that do not close");

    // Faces are numbered by the triangle with the lower apex, in triangle order; the other side reads it across.
    const auto owner = mesh.owner, neighbour = mesh.neighbour;
    int faces = 0;
    Kokkos::parallel_scan(
        "tri_mesh::number_faces", Range(exec, 0, total),
        KOKKOS_LAMBDA(int t, int& sum, bool final) {
            for (int k = 0; k < 3; ++k) {
                if (corners(t, k) >= corners(t, 3 + k)) continue;
                if (final) {
                    const int a = corners(t, (k + 1) % 3), b = corners(t, (k + 2) % 3);
                    links(t, 3 + k) = sum;
                    owner(sum) = a < b ? a : b;
                    neighbour(sum) = a < b ? b : a;
                }
                ++sum;
            }
        },
        faces);
    if (faces != 3 * n) throw std::runtime_error("tri_mesh::from_rings: an edge without one lower apex");
    Kokkos::parallel_for(
        "tri_mesh::faces_across", Range(exec, 0, total), KOKKOS_LAMBDA(int t) {
            for (int k = 0; k < 3; ++k) {
                if (corners(t, k) < corners(t, 3 + k)) continue;
                const int u = links(t, k), w = corners(t, 3 + k);
                for (int j = 0; j < 3; ++j)
                    if (corners(u, j) == w) links(t, 3 + k) = links(u, 3 + j);
            }
        });
    return mesh;
}

// Products of a triangle with corners 1, 2 at offsets a, b from corner 0 (grid steps), shared by the centre and the
// in-circle filter so every kernel gets the same bits.
// Used by: centres in tri_repair, the screen, total_area, centre of each triangle.
struct Circle {
    double la, lb, mx, my, det;
    KOKKOS_INLINE_FUNCTION Circle(double ax, double ay, double bx, double by)
        : la(ax * ax + ay * ay), lb(bx * bx + by * by), mx(la * bx - lb * ax), my(la * by - lb * ay), det(ax * by - ay * bx) {}
    // The circumcentre of corner 0 at grid point (x0, y0), in box units.
    KOKKOS_INLINE_FUNCTION Vec2 centre(int x0, int y0, double step) const {
        const double scale = 0.5 / det;
        return {(x0 + my * scale) * step, (y0 - mx * scale) * step};
    }
};

// Offset from grid coordinate `from` to `to` on a period, to the nearest image.
KOKKOS_INLINE_FUNCTION double steps(int from, int to, int period) {
    int d = to - from;
    if (d >= period / 2) d -= period;
    if (d < -(period / 2)) d += period;
    return d;
}

// Face geometry, one thread per triangle: the area vector (owner to neighbour) is the edge between the two centres turned
// a quarter, the centroid its midpoint, both relative to the owner. A face belongs to the side of the lower apex.
// Used by: Mesh<2>::geometry.
template <class ExecSpace, class X>
void faces(const ExecSpace& exec, const X& x, const ep::BoxN<2>& box,
           const Triangles<typename ExecSpace::memory_space>& mesh,
           const tet_mesh::Geometry<typename ExecSpace::memory_space, 2>& out) {
    const auto corners = mesh.corners, links = mesh.links;
    const auto centre = mesh.centre;
    const auto area = out.area, centroid = out.centroid;
    const auto rows = out.rows;
    const bool with_rows = rows.extent(0) > 0;
    Kokkos::parallel_for(
        "tri_mesh::faces", Kokkos::RangePolicy<ExecSpace>(exec, 0, mesh.count()), KOKKOS_LAMBDA(int t) {
            const Vec2 ct = tet_mesh::row<2>(centre, t);
            for (int k = 0; k < 3; ++k) {
                if (corners(t, k) >= corners(t, 3 + k)) continue;
                const int a = corners(t, (k + 1) % 3), b = corners(t, (k + 2) % 3), f = links(t, 3 + k);
                const Vec2 e = tet_mesh::wrap(ct - tet_mesh::row<2>(centre, links(t, k)), box);
                const int o = a < b ? a : b, j = a < b ? b : a;
                const double sign = a < b ? 1.0 : -1.0;
                const Vec2 A{sign * e.y, -sign * e.x};
                const Vec2 c = tet_mesh::wrap(ct - tet_mesh::row<2>(x, o), box) - e * 0.5;
                tet_mesh::store(area, f, A);
                tet_mesh::store(centroid, f, c);
                if (with_rows) {
                    const float row[8] = {float(A.x), float(A.y), 0, float(c.x), float(c.y), 0,
                                          Kokkos::bit_cast<float>(o), Kokkos::bit_cast<float>(j)};
                    for (int s = 0; s < 8; ++s) rows(f, s) = row[s];
                }
            }
        });
}

}  // namespace tri_mesh
