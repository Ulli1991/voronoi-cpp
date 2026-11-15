// gDel3D (Cao, Nanjappa, Gao, Tan, I3D 2014) in the library benchmark on one A100: GPU 3D Delaunay triangulation,
// double precision, exact/SoS predicates, CPU star splaying fix-up on (the library's defaults, GDelParams()).
// Not periodic: the points are padded with periodic images (gpu_images.hpp) before the call; the time includes it.
//   gdel3d <points file> [--repeats=3] [--width=<shell width in mean spacings>] [--volumes=file]
// Two variants, each 1 untimed run (device peak memory, volumes, checks) + `repeats` timed runs:
//   "Delaunay": host points -> padded host points -> GpuDel::compute -> tetrahedra (+ adjacency) in host memory
//               (gDel3D's API: Point3HVec in, GDelOutput out; it copies in, triangulates on the GPU, copies the
//               tetrahedra back, then repairs failed stars on the CPU).
//   "Delaunay + faces (ours)": the same, then the full Voronoi cells of the n original points (volumes, centroids,
//               faces with neighbours and areas) from the tetrahedra on the host (dual_cells.hpp, OpenMP).
// The untimed run also checks the padding: every finite tetrahedron with an original vertex must have its circumball
// inside the padded box [-w, 1 + w]^3; it prints the width that would have sufficed (needed_width) and the number of
// tetrahedra that violate it (width_violations; must be 0).

#include "dual_cells.hpp"
#include "libbench.hpp"
#include "gpu_images.hpp"

#include "gDel3D/GpuDelaunay.h"

#include <array>
#include <cstring>

namespace {

using V3 = std::array<double, 3>;
inline V3 sub(const V3& a, const V3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline V3 cross(const V3& a, const V3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline double dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline double det(const V3& a, const V3& b, const V3& c) { return dot(a, cross(b, c)); }

// Circumcentre of the tetrahedron (0, a, b, c), relative to its first vertex.
inline V3 tet_circumcentre(const V3& a, const V3& b, const V3& c) {
    const V3 bc = cross(b, c), ca = cross(c, a), ab = cross(a, b);
    const double aa = dot(a, a), bb = dot(b, b), cc = dot(c, c), d = 2.0 * dot(a, bc);
    return {(aa * bc[0] + bb * ca[0] + cc * ab[0]) / d, (aa * bc[1] + bb * ca[1] + cc * ab[1]) / d,
            (aa * bc[2] + bb * ca[2] + cc * ab[2]) / d};
}
struct Run {
    gpu_images::Padded padded;
    Point3HVec input;
    GDelOutput output;
};

// Pad, convert to gDel3D's Point3 (double), triangulate. GpuDel is constructed per call (its state is per call).
void delaunay(const double* x, std::uint64_t n, double w, Run& r) {
    gpu_images::pad(x, n, w, r.padded);
    const std::int64_t m = std::int64_t(r.padded.size());
    r.input.resize(m);
    Point3* p = thrust::raw_pointer_cast(r.input.data());
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < m; ++i)
        for (int a = 0; a < 3; ++a) p[i]._p[a] = r.padded.xyz[3 * i + a];
    r.output = GDelOutput();
    GpuDel triangulator;  // GDelParams(): insertion by circumcentre, sorting, star splaying on
    triangulator.compute(r.input, &r.output);
}

// The live finite tetrahedra at an original vertex as dual::build's source: cells of the originals (index < n),
// neighbours keyed by padded index and labelled by their original.
struct GTets {
    const Run& r;
    std::vector<std::int64_t> t;
    std::uint64_t tets() const { return t.size(); }
    int id(std::uint64_t k, int a) const { return thrust::raw_pointer_cast(r.output.tetVec.data())[t[k]]._v[a]; }
    bool owner(std::uint64_t k, int a) const { return std::uint64_t(id(k, a)) < r.padded.n; }
    std::uint64_t cell(std::uint64_t k, int a) const { return std::uint64_t(id(k, a)); }
    std::uint64_t key(std::uint64_t k, int, int b) const { return std::uint64_t(id(k, b)); }
    std::int64_t label(std::uint64_t k, int, int b) const {
        const std::uint64_t i = std::uint64_t(id(k, b));
        return i < r.padded.n ? std::int64_t(i) : std::int64_t(r.padded.source[i - r.padded.n]);
    }
    void relative(std::uint64_t k, int a, int b, double* q) const {
        const double* X = r.padded.xyz.data();
        for (int d = 0; d < 3; ++d) q[d] = X[3 * id(k, b) + d] - X[3 * id(k, a) + d];
    }
};

// The full cells of the originals from gDel3D's tetrahedra (ours: dual_cells.hpp on the host, OpenMP).
dual::Cells cells_of(const Run& r) {
    const std::uint64_t n = r.padded.n;
    const int inf = int(r.padded.size());  // gDel3D's infinity vertex: index = number of input points
    const Tet* tets = thrust::raw_pointer_cast(r.output.tetVec.data());
    const char* info = thrust::raw_pointer_cast(r.output.tetInfoVec.data());
    GTets g{r, {}};
    for (std::int64_t t = 0; t < std::int64_t(r.output.tetVec.size()); ++t) {
        if (!isTetAlive(info[t])) continue;
        const int* v = tets[t]._v;
        if (v[0] == inf || v[1] == inf || v[2] == inf || v[3] == inf) continue;
        if (v[0] >= int(n) && v[1] >= int(n) && v[2] >= int(n) && v[3] >= int(n)) continue;
        g.t.push_back(t);
    }
    return dual::build(g, n, [&](std::uint64_t i, double* x) {
        for (int d = 0; d < 3; ++d) x[d] = r.padded.xyz[3 * i + d];
    });
}

// The padding check: the width each finite tetrahedron with an original vertex needs (its circumball beyond [0,1]^3).
void check_width(const Run& r, double& needed, long long& violations) {
    const std::uint64_t n = r.padded.n;
    const int inf = int(r.padded.size());
    const double* X = r.padded.xyz.data();
    const Tet* tets = thrust::raw_pointer_cast(r.output.tetVec.data());
    const char* info = thrust::raw_pointer_cast(r.output.tetInfoVec.data());
    const std::int64_t T = std::int64_t(r.output.tetVec.size());
    const double w = r.padded.w;
    double need = 0;
    long long bad = 0;
#pragma omp parallel for schedule(static) reduction(max : need) reduction(+ : bad)
    for (std::int64_t t = 0; t < T; ++t) {
        if (!isTetAlive(info[t])) continue;
        const int* v = tets[t]._v;
        if (v[0] == inf || v[1] == inf || v[2] == inf || v[3] == inf) continue;
        if (v[0] >= int(n) && v[1] >= int(n) && v[2] >= int(n) && v[3] >= int(n)) continue;
        const V3 o = {X[3 * v[0]], X[3 * v[0] + 1], X[3 * v[0] + 2]};
        V3 q[3];
        for (int j = 1; j < 4; ++j) q[j - 1] = sub({X[3 * v[j]], X[3 * v[j] + 1], X[3 * v[j] + 2]}, o);
        const V3 c = tet_circumcentre(q[0], q[1], q[2]);
        const double R = std::sqrt(dot(c, c));
        double e = 0;
        for (int a = 0; a < 3; ++a) e = std::max(e, std::max(R - (o[a] + c[a]), o[a] + c[a] + R - 1.0));
        need = std::max(need, e);
        if (e > w) ++bad;
    }
    needed = need;
    violations = bad;
}

std::string stats_json(const Run& r) {
    const Statistics& s = r.output.stats;
    std::ostringstream o;
    o.precision(6);
    o << ",\"gdel3d_ms\":{\"total\":" << s.totalTime << ",\"init\":" << s.initTime << ",\"split\":" << s.splitTime
      << ",\"flip\":" << s.flipTime << ",\"relocate\":" << s.relocateTime << ",\"sort\":" << s.sortTime
      << ",\"out\":" << s.outTime << ",\"splaying\":" << s.splayingTime << "},\"failed_verts\":" << s.failVertNum
      << ",\"final_stars\":" << s.finalStarNum << ",\"flips\":" << s.totalFlipNum
      << ",\"tetrahedra\":" << r.output.tetVec.size() << ",\"padded_points\":" << r.padded.size();
    return o.str();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: gdel3d <points file> [--repeats=3] [--width=spacings] [--volumes=file]\n";
        return 1;
    }
    const std::string file = argv[1];
    const int repeats = std::stoi(libbench::option(argc, argv, "repeats", "3"));
    const auto frames = libbench::read_frames(file);
    const std::uint64_t n = frames.n;
    const std::string input = libbench::input_name(file);
    const double spacings = std::stod(libbench::option(argc, argv, "width", "0"));
    const double ws = spacings > 0 ? spacings : gpu_images::default_width_spacings(input);
    const double w = ws * std::cbrt(1.0 / double(n));
    const double* x = frames.frame(0);
    CudaSafeCall(cudaSetDevice(0));
    cudaFree(nullptr);  // context creation outside the clock

    libbench::Setup s;
    s.library = "gDel3D";
    s.version = libbench::env("LIBBENCH_LIB_VERSION", "a429be8 + gdel3d.patch");
    s.device = "A100-SXM4-40GB";
    s.threads = 1;
    std::ostringstream per;
    per << "images: shell of width " << ws << " mean spacings (" << w << ") around the unit box";
    s.periodic = per.str();
    s.precision = "double, exact/SoS predicates on the GPU, CPU star splaying fix-up";

    for (int variant = 0; variant < 2; ++variant) {
        const bool faces = variant == 1;
        s.variant = faces ? "Delaunay + faces (ours)" : "Delaunay";
        s.outputs = faces ? "tetrahedra; cells (ours): volumes centroids faces areas neighbours"
                          : "tetrahedra (Delaunay only), adjacency";
        s.timed = faces ? "points in host memory to full cells in host memory: padding, GpuDel::compute (H2D, GPU "
                          "Delaunay, D2H, CPU star splaying), our cells on the CPU (dual_cells.hpp, OpenMP, 18 threads)"
                       : "points in host memory to tetrahedra in host memory: padding, GpuDel::compute (H2D, GPU "
                         "Delaunay, D2H, CPU star splaying)";
        dual::Cells cells;
        std::string extra;
        double peak = 0;
        const double start = libbench::now();  // as run_protocol: warm-up runs until 2 s after this
        {
            Run r;
            libbench::DevicePeak dp;
            dp.start();
            delaunay(x, n, w, r);
            peak = dp.stop();
            double needed = 0;
            long long violations = 0;
            check_width(r, needed, violations);
            std::ostringstream o;
            o << ",\"width\":" << w << ",\"needed_width\":" << needed << ",\"width_violations\":" << violations;
            extra = o.str();
            std::cerr << "# gdel3d " << input << " " << n << " " << s.variant << ": width " << w << " needed "
                      << needed << " (" << needed * std::cbrt(double(n)) << " spacings), violations " << violations
                      << stats_json(r) << std::endl;
            if (faces) {
                cells = cells_of(r);
                double sum = 0;
                for (double v : cells.volume) sum += v;
                std::cerr << "# sum of volumes - 1 = " << sum - 1.0 << ", faces per cell "
                          << double(cells.face_start.back()) / double(n) << ", worst closure "
                          << dual::worst_closure(cells) << ", flat " << cells.flat << std::endl;
                libbench::write_volumes(libbench::option(argc, argv, "volumes"), cells.volume);
            }
        }
        s.note = "untimed run: device peak (cudaMemGetInfo) over the whole call; host star splaying and our cells on " +
                 std::to_string(gpu_images::threads()) + " OpenMP threads (padding too)";
        while (libbench::now() - start < 2.0) {
            Run r;
            delaunay(x, n, w, r);
            if (faces) cells = cells_of(r);
        }
        for (int rep = 1; rep <= repeats; ++rep) {
            Run r;
            const double t0 = libbench::now();
            delaunay(x, n, w, r);
            if (faces) cells = cells_of(r);
            const double t = libbench::now() - t0;
            libbench::record(s, file, input, n, "build", rep, t, peak, extra + stats_json(r));
        }
    }
    return 0;
}
