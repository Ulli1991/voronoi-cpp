#pragma once

#include "face_geometry.hpp"

// A persistent Delaunay tetrahedralization and the Voronoi geometry it implies.
//
// Each tetrahedron is four particles whose circumsphere holds no other
// particle; its circumcentre is a Voronoi vertex shared by the four cells.
// A Voronoi face between particles i and j is the ring of circumcentres of
// the tetrahedra around the edge i-j. Kept from step to step, the mesh needs
// per step only
//   1. one circumcentre per tetrahedron,
//   2. face geometry from those circumcentres,
//   3. a check that each tetrahedron's circumsphere is still empty of the
//      far vertices of its neighbours (the local Delaunay condition),
// and repair only where the check fails.
namespace tet_mesh {

using shell_voronoi::Vec3;

template <class MemorySpace>
struct Mesh {
    Kokkos::View<int* [4], MemorySpace> vertex;     // particles
    Kokkos::View<int* [4], MemorySpace> adjacent;   // tetrahedron across the face opposite vertex k, -1 if none
    Kokkos::View<double* [3], MemorySpace> centre;  // circumcentre, unwrapped near vertex 0
    // Owned faces (owner < neighbour), each a ring of tetrahedra in order.
    Kokkos::View<int*, MemorySpace> owner, neighbour, ring_start, ring;
};

namespace detail {
using face_geometry::detail::circumcentre;
using face_geometry::detail::wrap;
}  // namespace detail

// 1. Circumcentres, one thread per tetrahedron.
template <class ExecSpace>
void circumcentres(const ExecSpace& exec, const Kokkos::View<double* [3], typename ExecSpace::memory_space>& x,
                   double box, const Mesh<typename ExecSpace::memory_space>& mesh) {
    Kokkos::parallel_for(
        "tet_mesh::circumcentres", Kokkos::RangePolicy<ExecSpace>(exec, 0, mesh.vertex.extent(0)),
        KOKKOS_LAMBDA(int t) {
            const int v0 = mesh.vertex(t, 0);
            Vec3<double> d[3];
            for (int k = 0; k < 3; ++k) {
                const int v = mesh.vertex(t, k + 1);
                d[k] = {detail::wrap(x(v, 0) - x(v0, 0), box), detail::wrap(x(v, 1) - x(v0, 1), box),
                        detail::wrap(x(v, 2) - x(v0, 2), box)};
            }
            const Vec3<double> c = detail::circumcentre(d[0], d[1], d[2]);
            mesh.centre(t, 0) = x(v0, 0) + c.x;
            mesh.centre(t, 1) = x(v0, 1) + c.y;
            mesh.centre(t, 2) = x(v0, 2) + c.z;
        });
}

// 2. Face area vectors (owner to neighbour), centroids (relative to the
// owner) and cell volumes, one thread per owned face.
template <class ExecSpace>
void faces(const ExecSpace& exec, const Kokkos::View<double* [3], typename ExecSpace::memory_space>& x, double box,
           const Mesh<typename ExecSpace::memory_space>& mesh,
           const face_geometry::Geometry<typename ExecSpace::memory_space>& out) {
    Kokkos::deep_copy(exec, out.volume, 0.0);
    Kokkos::parallel_for(
        "tet_mesh::faces", Kokkos::RangePolicy<ExecSpace>(exec, 0, mesh.owner.extent(0)), KOKKOS_LAMBDA(int f) {
            const int i = mesh.owner(f), j = mesh.neighbour(f);
            const int begin = mesh.ring_start(f), end = mesh.ring_start(f + 1);
            auto vertex = [&](int m) {
                const int t = mesh.ring(m);
                return Vec3<double>{detail::wrap(mesh.centre(t, 0) - x(i, 0), box),
                                    detail::wrap(mesh.centre(t, 1) - x(i, 1), box),
                                    detail::wrap(mesh.centre(t, 2) - x(i, 2), box)};
            };
            const Vec3<double> dj{detail::wrap(x(j, 0) - x(i, 0), box), detail::wrap(x(j, 1) - x(i, 1), box),
                                  detail::wrap(x(j, 2) - x(i, 2), box)};
            const Vec3<double> first = vertex(begin);
            Vec3<double> previous = vertex(begin + 1), area{0, 0, 0}, moment{0, 0, 0};
            double weight = 0;
            for (int m = begin + 2; m < end; ++m) {
                const Vec3<double> v = vertex(m);
                const Vec3<double> twice = cross(previous - first, v - first);
                area = area + twice * 0.5;
                const double w = Kokkos::sqrt(dot(twice, twice));
                weight += w;
                moment = moment + (first + previous + v) * (w / 3.0);
                previous = v;
            }
            if (dot(area, dj) < 0) area = area * -1.0;
            const Vec3<double> centroid = weight > 0 ? moment * (1.0 / weight) : first;
            out.area(f, 0) = area.x;
            out.area(f, 1) = area.y;
            out.area(f, 2) = area.z;
            out.centroid(f, 0) = centroid.x;
            out.centroid(f, 1) = centroid.y;
            out.centroid(f, 2) = centroid.z;
            const double pyramid = Kokkos::sqrt(dot(area, area) * dot(dj, dj)) / 6.0;
            Kokkos::atomic_add(&out.volume(i), pyramid);
            Kokkos::atomic_add(&out.volume(j), pyramid);
        });
}

// 3. Local Delaunay check, one thread per tetrahedron: the far vertex of each
// neighbour must not lie inside this circumsphere. Writes 1 per failing
// tetrahedron into `invalid` and returns how many failed. Rounding-level
// cases count as valid here; deciding them exactly is the repair's job.
template <class ExecSpace>
int check(const ExecSpace& exec, const Kokkos::View<double* [3], typename ExecSpace::memory_space>& x, double box,
          const Mesh<typename ExecSpace::memory_space>& mesh,
          const Kokkos::View<int*, typename ExecSpace::memory_space>& invalid) {
    int failed = 0;
    Kokkos::parallel_reduce(
        "tet_mesh::check", Kokkos::RangePolicy<ExecSpace>(exec, 0, mesh.vertex.extent(0)),
        KOKKOS_LAMBDA(int t, int& count) {
            const Vec3<double> c{mesh.centre(t, 0), mesh.centre(t, 1), mesh.centre(t, 2)};
            auto from_centre = [&](int v) {
                return Vec3<double>{detail::wrap(x(v, 0) - c.x, box), detail::wrap(x(v, 1) - c.y, box),
                                    detail::wrap(x(v, 2) - c.z, box)};
            };
            const Vec3<double> r = from_centre(mesh.vertex(t, 0));
            const double r2 = dot(r, r);
            int bad = 0;
            for (int k = 0; k < 4 && !bad; ++k) {
                const int other = mesh.adjacent(t, k);
                if (other < 0) continue;
                // The far vertex of the neighbour is the one this tetrahedron lacks.
                int far = -1;
                for (int a = 0; a < 4; ++a) {
                    const int v = mesh.vertex(other, a);
                    bool shared = false;
                    for (int b = 0; b < 4; ++b) shared |= v == mesh.vertex(t, b);
                    if (!shared) far = v;
                }
                if (far < 0) continue;
                const Vec3<double> d = from_centre(far);
                bad = dot(d, d) < r2 * (1.0 - 1e-10);
            }
            invalid(t) = bad;
            count += bad;
        },
        failed);
    return failed;
}

// Orientation sign of every tetrahedron (+1, -1, or 0 when flat to
// rounding). A tetrahedron whose sign differs from the build's has inverted,
// and the local Delaunay check is only conclusive without inversions.
template <class ExecSpace>
void orientations(const ExecSpace& exec, const Kokkos::View<double* [3], typename ExecSpace::memory_space>& x,
                  double box, const Mesh<typename ExecSpace::memory_space>& mesh,
                  const Kokkos::View<int*, typename ExecSpace::memory_space>& sign) {
    Kokkos::parallel_for(
        "tet_mesh::orientations", Kokkos::RangePolicy<ExecSpace>(exec, 0, mesh.vertex.extent(0)),
        KOKKOS_LAMBDA(int t) {
            const int v0 = mesh.vertex(t, 0);
            Vec3<double> d[3];
            for (int k = 0; k < 3; ++k) {
                const int v = mesh.vertex(t, k + 1);
                d[k] = {detail::wrap(x(v, 0) - x(v0, 0), box), detail::wrap(x(v, 1) - x(v0, 1), box),
                        detail::wrap(x(v, 2) - x(v0, 2), box)};
            }
            const double volume6 = dot(d[0], cross(d[1], d[2]));
            const double scale = Kokkos::sqrt(dot(d[0], d[0]) * dot(d[1], d[1]) * dot(d[2], d[2]));
            sign(t) = volume6 > 1e-12 * scale ? 1 : volume6 < -1e-12 * scale ? -1 : 0;
        });
}

}  // namespace tet_mesh
