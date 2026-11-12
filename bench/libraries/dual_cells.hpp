#pragma once

// "+ faces (ours)" for the Delaunay libraries: the full Voronoi cells of the n original points (volume, centroid, and
// per face the neighbour and the outward area vector) from a Delaunay tetrahedralization, on the host with OpenMP.
// Used by images_dual.hpp (Qhull, TetGen), cgal.cpp, geogram.cpp (PDEL) and gdel3d.cu, inside the clock.
//
// The cell of vertex i is the union over its tetrahedra T and the ordered pairs (j, k) of the other vertices of T of the
// signed simplex (p_i, m_ij, c_ijk, c_T): m the edge midpoint, c_ijk the circumcentre of face ijk, c_T that of T, the sign
// the orientation of (i, j, k, l). The triangle (m_ij, c_ijk, c_T) lies in the bisector of ij: summed over the
// tetrahedra around edge ij with the same signs it is the face between i and j, so its area vector A satisfies
// m_ij . A / 3 = the simplex volume and points from i to j. The centroid is the volume-weighted mean of the simplices'.
//
// Source interface (Tets): tets() count; owner(t, a): whether corner a of tetrahedron t is an original whose cell is
// wanted (padded sets: index < n; periodic: every corner); cell(t, a): that original's index; key(t, a, b): a 64-bit id
// of corner b as seen from corner a (the same image of the same neighbour gives the same key); label(t, a, b): corner
// b's original index; relative(t, a, b, out): corner b's position minus corner a's.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace dual {

struct Cells {
    std::vector<double> volume, centroid;     // n, n * 3 (centroid relative to the origin of the coordinates)
    std::vector<std::uint64_t> face_start;    // n + 1
    std::vector<std::int64_t> neighbour;      // per face: the neighbour's original index
    std::vector<double> area;                 // per face: outward area vector (3)
    std::uint64_t flat = 0;                   // flat tetrahedra skipped
};

struct D3 {
    double x, y, z;
};
inline D3 operator-(const D3& a, const D3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline D3 cross(const D3& a, const D3& b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double dot(const D3& a, const D3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// Circumcentre of (0, u, v, w); false if flat.
inline bool tet_centre(const D3& u, const D3& v, const D3& w, D3& c) {
    const double d = dot(u, cross(v, w));
    if (d == 0 || !std::isfinite(d)) return false;
    const D3 vw = cross(v, w), wu = cross(w, u), uv = cross(u, v);
    const double uu = dot(u, u), vv = dot(v, v), ww = dot(w, w), s = 0.5 / d;
    c = {(uu * vw.x + vv * wu.x + ww * uv.x) * s, (uu * vw.y + vv * wu.y + ww * uv.y) * s,
         (uu * vw.z + vv * wu.z + ww * uv.z) * s};
    return true;
}

// Circumcentre of (0, a, b).
inline D3 tri_centre(const D3& a, const D3& b) {
    const D3 n = cross(a, b);
    const double aa = dot(a, a), bb = dot(b, b), s = 0.5 / dot(n, n);
    const D3 t = {aa * b.x - bb * a.x, aa * b.y - bb * a.y, aa * b.z - bb * a.z};
    const D3 f = cross(t, n);
    return {f.x * s, f.y * s, f.z * s};
}

// The cells of the n originals. position(i, out): original i's coordinates (for the centroids).
template <class Tets, class Position>
Cells build(const Tets& T, std::uint64_t n, const Position& position) {
    Cells out;
    const std::uint64_t m = T.tets();
    // Incidences of the originals: CSR of (tetrahedron, corner) per cell.
    std::vector<std::uint32_t> count(n + 1, 0);
    for (std::uint64_t t = 0; t < m; ++t)
        for (int a = 0; a < 4; ++a)
            if (T.owner(t, a)) ++count[T.cell(t, a) + 1];
    for (std::uint64_t i = 0; i < n; ++i) count[i + 1] += count[i];
    std::vector<std::uint64_t> inc(count[n]);
    {
        std::vector<std::uint32_t> at(count.begin(), count.end() - 1);
        for (std::uint64_t t = 0; t < m; ++t)
            for (int a = 0; a < 4; ++a)
                if (T.owner(t, a)) inc[at[T.cell(t, a)]++] = t * 4 + a;
    }
    // Circumcentre of every tetrahedron once, relative to its corner 0 (parallel); flat ones flagged.
    std::vector<D3> centre(m);
    std::vector<unsigned char> flat(m, 0);
#pragma omp parallel for schedule(static)
    for (std::int64_t t = 0; t < std::int64_t(m); ++t) {
        double r[3];
        D3 q[3];
        for (int b = 1; b < 4; ++b) {
            T.relative(std::uint64_t(t), 0, b, r);
            q[b - 1] = {r[0], r[1], r[2]};
        }
        flat[t] = !tet_centre(q[0], q[1], q[2], centre[t]);
    }
    out.volume.assign(n, 0.0);
    out.centroid.assign(3 * n, 0.0);
    out.face_start.assign(n + 1, 0);
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    // Each thread fills its own face lists over a contiguous range of cells; concatenated in cell order after.
    struct Part {
        std::vector<std::int64_t> neighbour;
        std::vector<double> area;
        std::uint64_t flat = 0;
    };
    std::vector<Part> parts(threads);
    const std::uint64_t chunk = (n + threads - 1) / threads;
#pragma omp parallel num_threads(threads)
    {
        int me = 0;
#ifdef _OPENMP
        me = omp_get_thread_num();
#endif
        Part& part = parts[me];
        std::vector<std::uint64_t> keys;
        std::vector<std::int64_t> labels;
        std::vector<D3> sums;
        const std::uint64_t lo = std::min<std::uint64_t>(n, me * chunk), hi = std::min<std::uint64_t>(n, lo + chunk);
        for (std::uint64_t i = lo; i < hi; ++i) {
            keys.clear(), labels.clear(), sums.clear();
            double volume = 0;
            D3 moment{0, 0, 0};
            for (std::uint32_t e = count[i]; e < count[i + 1]; ++e) {
                const std::uint64_t t = inc[e] / 4;
                const int a = int(inc[e] % 4);
                if (flat[t]) {
                    ++part.flat;
                    continue;
                }
                int other[3], o = 0;
                for (int b = 0; b < 4; ++b)
                    if (b != a) other[o++] = b;
                D3 p[4];
                double r[3];
                for (int b : other) {
                    T.relative(t, a, b, r);
                    p[b] = {r[0], r[1], r[2]};
                }
                D3 c = centre[t];  // relative to corner 0; corner 0 relative to corner a is p[0]
                if (a != 0) c = {c.x + p[0].x, c.y + p[0].y, c.z + p[0].z};
                // The three faces at corner a: circumcentres f[pair], pair (other[u], other[v]).
                D3 f[3];
                f[0] = tri_centre(p[other[0]], p[other[1]]);
                f[1] = tri_centre(p[other[0]], p[other[2]]);
                f[2] = tri_centre(p[other[1]], p[other[2]]);
                auto face_of = [&](int u, int v) { return u + v == 1 ? f[0] : (u + v == 2 ? f[1] : f[2]); };
                for (int u = 0; u < 3; ++u) {
                    const int b = other[u];
                    const D3 mid = {0.5 * p[b].x, 0.5 * p[b].y, 0.5 * p[b].z};
                    D3 face{0, 0, 0};
                    for (int v = 0; v < 3; ++v) {
                        if (v == u) continue;
                        const int k = other[v], l = other[3 - u - v];
                        const double orient = dot(p[b], cross(p[k], p[l]));
                        const D3 fc = face_of(u, v);
                        const D3 half = cross(fc - mid, c - mid);
                        const double s = orient > 0 ? 0.5 : -0.5;
                        face = {face.x + s * half.x, face.y + s * half.y, face.z + s * half.z};
                        const double w = s * dot(mid, half) / 3;  // signed simplex (0, mid, fc, c): mid . A / 3
                        volume += w;
                        moment = {moment.x + 0.25 * w * (mid.x + fc.x + c.x), moment.y + 0.25 * w * (mid.y + fc.y + c.y),
                                  moment.z + 0.25 * w * (mid.z + fc.z + c.z)};
                    }
                    const std::uint64_t key = T.key(t, a, b);
                    std::size_t slot = 0;
                    while (slot < keys.size() && keys[slot] != key) ++slot;
                    if (slot == keys.size()) {
                        keys.push_back(key);
                        labels.push_back(T.label(t, a, b));
                        sums.push_back({0, 0, 0});
                    }
                    sums[slot] = {sums[slot].x + face.x, sums[slot].y + face.y, sums[slot].z + face.z};
                }
            }
            double x[3];
            position(i, x);
            out.volume[i] = volume;
            out.centroid[3 * i + 0] = x[0] + moment.x / volume;
            out.centroid[3 * i + 1] = x[1] + moment.y / volume;
            out.centroid[3 * i + 2] = x[2] + moment.z / volume;
            std::uint64_t faces = 0;
            for (std::size_t s = 0; s < keys.size(); ++s) {
                if (sums[s].x == 0 && sums[s].y == 0 && sums[s].z == 0) continue;  // a degenerate (zero) face
                part.neighbour.push_back(labels[s]);
                part.area.insert(part.area.end(), {sums[s].x, sums[s].y, sums[s].z});
                ++faces;
            }
            out.face_start[i + 1] = faces;
        }
    }
    for (std::uint64_t i = 0; i < n; ++i) out.face_start[i + 1] += out.face_start[i];
    out.neighbour.reserve(out.face_start[n]);
    out.area.reserve(3 * out.face_start[n]);
    for (const Part& p : parts) {
        out.neighbour.insert(out.neighbour.end(), p.neighbour.begin(), p.neighbour.end());
        out.area.insert(out.area.end(), p.area.begin(), p.area.end());
        out.flat += p.flat;
    }
    return out;
}

// Area-vector closure of the cells: the largest |sum of a cell's area vectors| / sum of their lengths (0 for a closed
// cell up to rounding). An untimed check.
inline double worst_closure(const Cells& c) {
    double worst = 0;
    for (std::size_t i = 0; i + 1 < c.face_start.size(); ++i) {
        double s[3] = {0, 0, 0}, total = 0;
        for (std::uint64_t f = c.face_start[i]; f < c.face_start[i + 1]; ++f) {
            for (int a = 0; a < 3; ++a) s[a] += c.area[3 * f + a];
            total += std::sqrt(c.area[3 * f] * c.area[3 * f] + c.area[3 * f + 1] * c.area[3 * f + 1] +
                               c.area[3 * f + 2] * c.area[3 * f + 2]);
        }
        if (total > 0) worst = std::max(worst, std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]) / total);
    }
    return worst;
}

}  // namespace dual
