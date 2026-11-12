#pragma once

// Shared by the drivers of the non-periodic Delaunay libraries (qhull.cpp, tetgen.cpp): the periodic box emulated by
// padding with periodic images, Voronoi volumes from the Delaunay tetrahedra ("dual volumes (ours)"), and the
// exactness certificate of the padding.
//
// Padding: the n points, then every periodic image x + s (s in {-1, 0, 1}^3, s != 0) that lies in [-w, 1 + w]^3.
// Index i < n is original point i.
//
// Certificate (outside the clock): the cell of an original point is exact when every tetrahedron incident to it is a
// tetrahedron of the periodic Delaunay triangulation. A tetrahedron of the padded triangulation whose circumsphere
// lies inside [-w, 1 + w]^3 is one: every periodic image inside that sphere is in the padded set, and the padded set
// has none inside it. required_width() returns the largest excursion of such a circumsphere beyond the unit box over
// the tetrahedra incident to an original point; the padding is exact when it is <= w (up to the library's own
// predicates: TetGen's are exact, Qhull's are floating point).
//
// Dual volumes: the Voronoi cell of vertex i is the union over its tetrahedra T and the ordered pairs (j, k) of the
// other vertices of T of the signed simplex (p_i, midpoint of ij, circumcentre of ijk, circumcentre of T), signed by
// the orientation of (i, j, k, l). The signs make the sum exact for tetrahedra whose circumcentre lies outside them.

#include "libbench.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

namespace images {

struct Padded {
    std::vector<double> xyz;  // (n + images) * 3
    std::uint64_t n = 0;      // originals first
    std::uint64_t size() const { return xyz.size() / 3; }
};

// All periodic images within w of the unit box; w < 1.
inline Padded pad(const double* xyz, std::uint64_t n, double w) {
    Padded p;
    p.n = n;
    p.xyz.reserve(std::size_t(3 * n * std::pow(1 + 2 * w, 3) * 1.02) + 64);
    p.xyz.assign(xyz, xyz + 3 * n);
    for (std::uint64_t i = 0; i < n; ++i) {
        const double* x = xyz + 3 * i;
        int lo[3], hi[3];
        for (int a = 0; a < 3; ++a) {
            lo[a] = x[a] > 1 - w ? -1 : 0;  // x - 1 >= -w
            hi[a] = x[a] < w ? 1 : 0;       // x + 1 <= 1 + w
        }
        for (int sx = lo[0]; sx <= hi[0]; ++sx)
            for (int sy = lo[1]; sy <= hi[1]; ++sy)
                for (int sz = lo[2]; sz <= hi[2]; ++sz) {
                    if (sx == 0 && sy == 0 && sz == 0) continue;
                    p.xyz.push_back(x[0] + sx);
                    p.xyz.push_back(x[1] + sy);
                    p.xyz.push_back(x[2] + sz);
                }
    }
    return p;
}

struct V3 {
    double x, y, z;
};
inline V3 sub(const double* a, const double* b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline V3 cross(const V3& a, const V3& b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double dot(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline double det(const V3& a, const V3& b, const V3& c) { return dot(a, cross(b, c)); }

// Circumcentre of the tetrahedron (0, u, v, w) relative to its first vertex; false if flat.
inline bool tet_centre(const V3& u, const V3& v, const V3& w, V3& c) {
    const double d = det(u, v, w);
    if (d == 0 || !std::isfinite(d)) return false;
    const V3 vw = cross(v, w), wu = cross(w, u), uv = cross(u, v);
    const double uu = dot(u, u), vv = dot(v, v), ww = dot(w, w), s = 0.5 / d;
    c = {(uu * vw.x + vv * wu.x + ww * uv.x) * s, (uu * vw.y + vv * wu.y + ww * uv.y) * s,
         (uu * vw.z + vv * wu.z + ww * uv.z) * s};
    return true;
}

// Circumcentre of the triangle (0, a, b) relative to its first vertex.
inline V3 tri_centre(const V3& a, const V3& b) {
    const V3 n = cross(a, b);
    const double aa = dot(a, a), bb = dot(b, b), s = 0.5 / dot(n, n);
    const V3 t = {aa * b.x - bb * a.x, aa * b.y - bb * a.y, aa * b.z - bb * a.z};
    const V3 f = cross(t, n);
    return {f.x * s, f.y * s, f.z * s};
}

// Adds the dual-volume contributions of tetrahedron (t[0..3]) to its original vertices (index < n). Returns false
// (and adds nothing) for a flat tetrahedron.
inline bool add_tet(const double* xyz, std::uint64_t n, const std::uint64_t t[4], double* volume) {
    if (t[0] >= n && t[1] >= n && t[2] >= n && t[3] >= n) return true;
    const double* p0 = xyz + 3 * t[0];
    V3 c;
    if (!tet_centre(sub(xyz + 3 * t[1], p0), sub(xyz + 3 * t[2], p0), sub(xyz + 3 * t[3], p0), c)) return false;
    const double cc[3] = {p0[0] + c.x, p0[1] + c.y, p0[2] + c.z};
    for (int a = 0; a < 4; ++a) {
        if (t[a] >= n) continue;
        const double* pi = xyz + 3 * t[a];
        const V3 ci = sub(cc, pi);
        double sum = 0;
        for (int b = 0; b < 4; ++b) {
            if (b == a) continue;
            for (int k = 0; k < 4; ++k) {
                if (k == a || k == b) continue;
                const int l = 6 - a - b - k;
                const V3 pj = sub(xyz + 3 * t[b], pi), pk = sub(xyz + 3 * t[k], pi), pl = sub(xyz + 3 * t[l], pi);
                const double o = det(pj, pk, pl);
                const V3 m = {0.5 * pj.x, 0.5 * pj.y, 0.5 * pj.z};
                const V3 f = tri_centre(pj, pk);
                const double v = det(m, f, ci);
                sum += o > 0 ? v : -v;
            }
        }
        volume[t[a]] += sum / 6;
    }
    return true;
}

// Excursion of the circumsphere of tetrahedron t beyond the unit box (<= 0 inside), or 0 if no original vertex.
inline double excursion(const double* xyz, std::uint64_t n, const std::uint64_t t[4]) {
    if (t[0] >= n && t[1] >= n && t[2] >= n && t[3] >= n) return 0;
    const double* p0 = xyz + 3 * t[0];
    V3 c;
    if (!tet_centre(sub(xyz + 3 * t[1], p0), sub(xyz + 3 * t[2], p0), sub(xyz + 3 * t[3], p0), c)) return INFINITY;
    const double r = std::sqrt(dot(c, c));
    const double cc[3] = {p0[0] + c.x, p0[1] + c.y, p0[2] + c.z};
    double e = -INFINITY;
    for (int a = 0; a < 3; ++a) e = std::fmax(e, std::fmax(r - cc[a], cc[a] + r - 1));
    return e;
}

// The protocol of a padded Delaunay library: Lib() builds the triangulation of a Padded set in Lib::build and lists
// its finite tetrahedra in Lib::each_tet(f) (f(const uint64_t[4])); the destructor frees it.
//   <driver> <points file> --variant=delaunay|dual --width=<w in mean spacings> [--repeats=3] [--volumes=file]
// "delaunay" times padding + the library's Delaunay; "dual" times padding + Delaunay + dual volumes of the n originals.
// The untimed first run measures peak memory, then (outside the peak and the clock) checks the certificate and,
// for "dual", writes the volumes.
template <class Lib>
int drive(int argc, char** argv, libbench::Setup s) {
    const std::string file = argv[1];
    const std::string variant = libbench::option(argc, argv, "variant", "dual");
    const double spacings = std::stod(libbench::option(argc, argv, "width", "4"));
    const int repeats = std::stoi(libbench::option(argc, argv, "repeats", "3"));
    const auto frames = libbench::read_frames(file);
    const std::uint64_t n = frames.n;
    const double h = std::cbrt(1.0 / double(n)), w = spacings * h;
    const bool dual = variant == "dual";
    s.variant = dual ? s.variant + "; Delaunay + dual volumes (ours)" : s.variant + "; Delaunay";
    s.outputs = dual ? "tetrahedra, volumes" : "tetrahedra";
    s.timed = dual ? "points in host memory to padded set, tetrahedra and volumes of the n originals in host memory"
                   : "points in host memory to padded set and tetrahedra in the library's host memory";
    s.periodic = "images: every periodic image within w of the box";
    std::vector<double> volume;
    std::uint64_t flat = 0;
    // The padded set and the library's triangulation live on after the clock stops (freeing them is not timed).
    std::unique_ptr<Padded> held_points;
    std::unique_ptr<Lib> held_lib;
    std::vector<double> v;
    const auto body = [&]() {
        held_points = std::make_unique<Padded>(pad(frames.frame(0), n, w));
        held_lib = std::make_unique<Lib>();
        held_lib->build(*held_points);
        if (dual) {
            v.assign(n, 0.0);
            flat = 0;
            const double* xyz = held_points->xyz.data();
            held_lib->each_tet([&](const std::uint64_t t[4]) { flat += !add_tet(xyz, n, t, v.data()); });
        }
    };
    const auto release = [&]() {
        held_lib.reset();
        held_points.reset();
        v.clear();
        v.shrink_to_fit();
    };
    const auto memory_run = [&]() {
        libbench::HostPeak peak;
        peak.start();
        body();
        const double bytes = peak.stop();
        const std::uint64_t padded = held_points->size();
        // Certificate and volume sum, outside the peak and the clock.
        volume = v;
        const double* xyz = held_points->xyz.data();
        double need = -INFINITY;
        std::uint64_t tets = 0;
        held_lib->each_tet([&](const std::uint64_t t[4]) {
            need = std::fmax(need, excursion(xyz, n, t));
            ++tets;
        });
        release();
        double sum = 0;
        std::uint64_t missing = 0;
        for (double x : volume) {
            sum += x;
            missing += !(x > 0);
        }
        char note[512];
        std::snprintf(note, sizeof note,
                      "%sw = %g mean spacings = %.6g, %llu images (%.3f n); certificate: circumspheres of tetrahedra "
                      "at originals reach %.4g mean spacings beyond the box (%s); %llu tetrahedra; flat skipped %llu; "
                      "sum of volumes - 1 = %.3g; nonpositive volumes %llu",
                      s.note.empty() ? "" : (s.note + "; ").c_str(), spacings, w, (unsigned long long)(padded - n), double(padded - n) / double(n),
                      need / h, need <= w ? "exact" : "NOT exact", (unsigned long long)tets,
                      (unsigned long long)flat, dual ? sum - 1 : NAN, (unsigned long long)(dual ? missing : 0));
        std::fprintf(stderr, "# %s %s %s\n", s.library.c_str(), file.c_str(), note);
        s.note = note;
        if (dual) libbench::write_volumes(libbench::option(argc, argv, "volumes"), volume);
        volume.clear();
        volume.shrink_to_fit();
        return bytes;
    };
    const auto timed_run = [&]() {
        const double t0 = libbench::now();
        body();
        const double t1 = libbench::now();
        release();
        return t1 - t0;
    };
    libbench::run_protocol(s, file, n, "build", repeats, memory_run, timed_run);
    return 0;
}

}  // namespace images
