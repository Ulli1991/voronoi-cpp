#pragma once

#include "shell_voronoi.hpp"

// Face geometry from a known topology, one thread per face.
//
// A face between particles i (its owner) and j is bounded by a ring of
// particles q_0 ... q_{k-1}: its vertex m is the point equidistant from i, j,
// q_m and q_{m+1}, the circumcentre of that tetrahedron. Every face thus
// costs about the same, its vertices come straight from particle
// coordinates in double precision, and no cell is clipped. Each face adds
// its pyramid to both cells' volumes: |A| |x_j - x_i| / 6 on either side.
namespace face_geometry {

using shell_voronoi::Vec3;

// Faces in CSR form: face f has ring entries [ring_start(f), ring_start(f + 1)).
template <class MemorySpace>
struct Faces {
    Kokkos::View<int*, MemorySpace> owner, neighbour, ring_start, ring;
};

// Area vectors point from owner to neighbour; centroids are relative to the
// owner. volume must hold one entry per particle.
template <class MemorySpace>
struct Geometry {
    Kokkos::View<double* [3], MemorySpace> area, centroid;
    Kokkos::View<double*, MemorySpace> volume;
};

namespace detail {
KOKKOS_INLINE_FUNCTION double wrap(double d, double box) { return d - box * Kokkos::nearbyint(d / box); }

// The point equidistant from the origin and a, b, c.
KOKKOS_INLINE_FUNCTION Vec3<double> circumcentre(Vec3<double> a, Vec3<double> b, Vec3<double> c) {
    const Vec3<double> bc = cross(b, c), ca = cross(c, a), ab = cross(a, b);
    const double scale = 0.5 / dot(a, bc);
    return (bc * dot(a, a) + ca * dot(b, b) + ab * dot(c, c)) * scale;
}
}  // namespace detail

template <class ExecSpace>
void compute(const ExecSpace& exec, const Kokkos::View<double* [3], typename ExecSpace::memory_space>& positions,
             double box, const Faces<typename ExecSpace::memory_space>& faces,
             const Geometry<typename ExecSpace::memory_space>& out) {
    const int count = static_cast<int>(faces.owner.extent(0));
    Kokkos::deep_copy(exec, out.volume, 0.0);
    Kokkos::parallel_for(
        "face_geometry::compute", Kokkos::RangePolicy<ExecSpace>(exec, 0, count), KOKKOS_LAMBDA(int f) {
            const int i = faces.owner(f), j = faces.neighbour(f);
            const int begin = faces.ring_start(f), end = faces.ring_start(f + 1);
            auto offset = [&](int k) {
                return Vec3<double>{detail::wrap(positions(k, 0) - positions(i, 0), box),
                                    detail::wrap(positions(k, 1) - positions(i, 1), box),
                                    detail::wrap(positions(k, 2) - positions(i, 2), box)};
            };
            const Vec3<double> dj = offset(j);
            // Vertex m sits between ring[m] and ring[m + 1]; walk the ring once.
            const Vec3<double> q0 = offset(faces.ring(begin));
            Vec3<double> q = q0;
            Vec3<double> first{0, 0, 0}, previous{0, 0, 0}, area{0, 0, 0}, moment{0, 0, 0};
            double weight = 0;
            for (int m = begin; m < end; ++m) {
                const Vec3<double> next = m + 1 < end ? offset(faces.ring(m + 1)) : q0;
                const Vec3<double> x = detail::circumcentre(dj, q, next);
                if (m == begin) {
                    first = x;
                } else if (m > begin + 1) {
                    const Vec3<double> twice = cross(previous - first, x - first);
                    area = area + twice * 0.5;
                    const double w = Kokkos::sqrt(dot(twice, twice));
                    weight += w;
                    moment = moment + (first + previous + x) * (w / 3.0);
                }
                previous = x;
                q = next;
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

}  // namespace face_geometry
