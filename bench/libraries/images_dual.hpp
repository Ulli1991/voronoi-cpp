#pragma once

// Shared by the drivers of the non-periodic Delaunay libraries (qhull.cpp, tetgen.cpp): the periodic box emulated by
// padding with periodic images, the full Voronoi cells from the Delaunay tetrahedra ("faces (ours)", dual_cells.hpp), and the
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

#include "dual_cells.hpp"
#include "libbench.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <array>
#include <memory>
#include <vector>

namespace images {

struct Padded {
    std::vector<double> xyz;            // (n + images) * 3
    std::vector<std::uint32_t> source;  // the original of each point (images after the n originals)
    std::uint64_t n = 0;                // originals first
    std::uint64_t size() const { return xyz.size() / 3; }
};

// All periodic images within w of the unit box; w < 1.
inline Padded pad(const double* xyz, std::uint64_t n, double w) {
    Padded p;
    p.n = n;
    p.xyz.reserve(std::size_t(3 * n * std::pow(1 + 2 * w, 3) * 1.02) + 64);
    p.xyz.assign(xyz, xyz + 3 * n);
    p.source.resize(n);
    for (std::uint64_t i = 0; i < n; ++i) p.source[i] = static_cast<std::uint32_t>(i);
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
                    p.source.push_back(static_cast<std::uint32_t>(i));
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

// The finite tetrahedra of a padded set as dual::build's source: the cells of the originals (index < n); neighbours
// keyed by their padded index (each image its own key), labelled by their original.
struct PaddedTets {
    const Padded& p;
    const std::vector<std::array<std::uint64_t, 4>>& t;
    std::uint64_t tets() const { return t.size(); }
    bool owner(std::uint64_t k, int a) const { return t[k][a] < p.n; }
    std::uint64_t cell(std::uint64_t k, int a) const { return t[k][a]; }
    std::uint64_t key(std::uint64_t k, int, int b) const { return t[k][b]; }
    std::int64_t label(std::uint64_t k, int, int b) const { return p.source[t[k][b]]; }
    void relative(std::uint64_t k, int a, int b, double* r) const {
        for (int d = 0; d < 3; ++d) r[d] = p.xyz[3 * t[k][b] + d] - p.xyz[3 * t[k][a] + d];
    }
};

// The full cells of the originals from the library's tetrahedra (copied out first, as part of the timed step).
template <class Lib>
dual::Cells padded_cells(const Padded& p, Lib& lib) {
    std::vector<std::array<std::uint64_t, 4>> tets;
    lib.each_tet([&](const std::uint64_t t[4]) {
        if (t[0] < p.n || t[1] < p.n || t[2] < p.n || t[3] < p.n) tets.push_back({t[0], t[1], t[2], t[3]});
    });
    return dual::build(PaddedTets{p, tets}, p.n, [&](std::uint64_t i, double* x) {
        for (int d = 0; d < 3; ++d) x[d] = p.xyz[3 * i + d];
    });
}

// The protocol of a padded Delaunay library: Lib() builds the triangulation of a Padded set in Lib::build and lists
// its finite tetrahedra in Lib::each_tet(f) (f(const uint64_t[4])); the destructor frees it.
//   <driver> <points file> --variant=delaunay|faces --width=<w in mean spacings> [--repeats=3] [--volumes=file]
// "delaunay" times padding + the library's Delaunay; "faces" times padding + Delaunay + the full cells of the n
// originals (dual::build: volumes, centroids, faces with neighbours and area vectors).
// The untimed first run measures peak memory, then (outside the peak and the clock) checks the certificate and,
// for "faces", writes the volumes.
template <class Lib>
int drive(int argc, char** argv, libbench::Setup s) {
    const std::string file = argv[1];
    const std::string variant = libbench::option(argc, argv, "variant", "faces");
    const double spacings = std::stod(libbench::option(argc, argv, "width", "4"));
    const int repeats = std::stoi(libbench::option(argc, argv, "repeats", "3"));
    const auto frames = libbench::read_frames(file);
    const std::uint64_t n = frames.n;
    const double h = std::cbrt(1.0 / double(n)), w = spacings * h;
    const bool faces = variant == "faces";
    s.variant = faces ? s.variant + "; Delaunay + faces (ours)" : s.variant + "; Delaunay";
    s.outputs = faces ? "tetrahedra; cells (ours): volumes centroids faces areas neighbours" : "tetrahedra";
    s.timed = faces ? "points in host memory to padded set, tetrahedra and the full cells of the n originals in host "
                      "memory (ours: dual_cells.hpp, OpenMP)"
                    : "points in host memory to padded set and tetrahedra in the library's host memory";
    s.periodic = "images: every periodic image within w of the box";
    std::vector<double> volume;
    std::uint64_t flat = 0;
    // The padded set and the library's triangulation live on after the clock stops (freeing them is not timed).
    std::unique_ptr<Padded> held_points;
    std::unique_ptr<Lib> held_lib;
    std::vector<double> v;
    dual::Cells cells;
    double closure = 0;
    const auto body = [&]() {
        held_points = std::make_unique<Padded>(pad(frames.frame(0), n, w));
        held_lib = std::make_unique<Lib>();
        held_lib->build(*held_points);
        if (faces) cells = padded_cells(*held_points, *held_lib);
    };
    const auto release = [&]() {
        held_lib.reset();
        held_points.reset();
        cells = dual::Cells{};
    };
    const auto memory_run = [&]() {
        libbench::HostPeak peak;
        peak.start();
        body();
        const double bytes = peak.stop();
        const std::uint64_t padded = held_points->size();
        // Certificate, volume sum and closure, outside the peak and the clock.
        volume = cells.volume;
        flat = cells.flat;
        closure = faces ? dual::worst_closure(cells) : 0;
        const double* xyz = held_points->xyz.data();
        double need = -INFINITY;
        std::uint64_t tets = 0;
        held_lib->each_tet([&](const std::uint64_t t[4]) {
            need = std::fmax(need, excursion(xyz, n, t));
            ++tets;
        });
        const std::uint64_t faces_total = cells.face_start.empty() ? 0 : cells.face_start.back();
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
                      "sum of volumes - 1 = %.3g; nonpositive volumes %llu; faces per cell %.3f; worst closure %.2g",
                      s.note.empty() ? "" : (s.note + "; ").c_str(), spacings, w, (unsigned long long)(padded - n), double(padded - n) / double(n),
                      need / h, need <= w ? "exact" : "NOT exact", (unsigned long long)tets,
                      (unsigned long long)flat, faces ? sum - 1 : NAN, (unsigned long long)(faces ? missing : 0),
                      faces ? double(faces_total) / double(n) : 0.0, closure);
        std::fprintf(stderr, "# %s %s %s\n", s.library.c_str(), file.c_str(), note);
        s.note = note;
        if (faces) libbench::write_volumes(libbench::option(argc, argv, "volumes"), volume);
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
