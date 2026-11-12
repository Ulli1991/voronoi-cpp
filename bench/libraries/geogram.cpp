// geogram (Bruno Levy, https://github.com/BrunoLevy/geogram) in the library benchmark (CONTRACT.md, libbench.hpp).
//   geogram_bench <points file> --variant=pdel|pdel-cells|pdel-nonper|pdel-nonper-faces --threads=T [--repeats=3]
//                 [--volumes=file] [--width=w]
//
// Variants:
//   pdel              GEO::PeriodicDelaunay3d(periodic=true): native periodic unit cube, multithreaded Bowyer-Watson
//                     with BRIO order, exact (PCK, symbolic perturbation) predicates. Delaunay only.
//   pdel-cells        the same, then every Voronoi cell from the Delaunay: copy_Laguerre_cell_from_Delaunay (zero weights,
//                     so Laguerre = Voronoi) into a ConvexCell, compute_geometry, volume and centroid (compute_mg) and
//                     per face the neighbour (v_global_index -> periodic_vertex_real) and area (facet_area), stored as
//                     per-thread face lists. This is how geogram's own demo_Delaunay3d and save_cells get the cells;
//                     parallel over cells with GEO::parallel_for_slice (one slice per thread).
//   pdel-nonper       GEO::Delaunay::create(3, "PDEL") (ParallelDelaunay3d, non-periodic) on the points padded with
//                     periodic images in a shell of width w (built inside the clock). Delaunay only.
//   pdel-nonper-faces the same + the full cells of the n originals from the tetrahedra (ours: dual_cells.hpp, volumes,
//                     centroids, faces with neighbours and areas). Untimed check: a cell is exact if every tetrahedron
//                     incident to it has its circumball inside the padded cube; others get NaN volumes.
//
// Threads: geogram runs its thread groups through OpenMP (the library is built with -fopenmp, so GEO_OPENMP and the
// OMPThreadManager); the run script sets OMP_NUM_THREADS = T and this driver Process::set_max_threads(T). The count
// geogram uses (Process::maximum_concurrent_threads()) is what the record says and is checked to equal T.

#include "dual_cells.hpp"
#include "libbench.hpp"

#include <geogram/basic/common.h>
#include <geogram/basic/logger.h>
#include <geogram/basic/process.h>
#include <geogram/delaunay/delaunay.h>
#include <geogram/delaunay/periodic_delaunay_3d.h>
#include <geogram/voronoi/convex_cell.h>

#include <atomic>
#include <cmath>
#include <limits>

namespace {

// geogram's Logger writes to stdout, which carries the records: send its messages to stderr instead.
class StderrLogger : public GEO::LoggerClient {
public:
    void div(const std::string& s) override { std::cerr << s << '\n'; }
    void out(const std::string& s) override { std::cerr << s; }
    void warn(const std::string& s) override { std::cerr << s; }
    void err(const std::string& s) override { std::cerr << s; }
    void status(const std::string& s) override { std::cerr << s; }
};

std::string version() { return libbench::env("GEOGRAM_COMMIT", "unknown"); }

// Per cell: volume, centroid, number of faces; per thread slice: the faces (neighbour, area) of its cells in order.
struct Cells {
    std::vector<double> volume, centroid;
    std::vector<GEO::index_t> nfaces;
    std::vector<std::vector<GEO::index_t>> neighbour;
    std::vector<std::vector<double>> area;
};

// The Voronoi cells of a computed PeriodicDelaunay3d, in parallel over the n real vertices.
void voronoi_cells(const GEO::PeriodicDelaunay3d& del, GEO::index_t n, Cells& c) {
    c.volume.assign(n, 0.0);
    c.centroid.assign(3 * std::size_t(n), 0.0);
    c.nfaces.assign(n, 0);
    const GEO::index_t slices = GEO::Process::maximum_concurrent_threads();
    c.neighbour.assign(slices, {});
    c.area.assign(slices, {});
    std::atomic<GEO::index_t> next_slice{0};
    GEO::parallel_for_slice(0, n, [&](GEO::index_t b, GEO::index_t e) {
        const GEO::index_t s = next_slice++;
        auto& nb = c.neighbour[s];
        auto& ar = c.area[s];
        nb.reserve(16 * std::size_t(e - b));
        ar.reserve(16 * std::size_t(e - b));
        GEO::ConvexCell C;
        C.use_exact_predicates(true);
        GEO::PeriodicDelaunay3d::IncidentTetrahedra W;
        for (GEO::index_t i = b; i < e; ++i) {
            del.copy_Laguerre_cell_from_Delaunay(i, C, W);
            C.compute_geometry();
            double m = 0;
            GEO::vec3 mg;
            C.compute_mg(m, mg);
            c.volume[i] = m;
            for (int d = 0; d < 3; ++d) c.centroid[3 * std::size_t(i) + d] = m > 0 ? mg[d] / m : 0.0;
            GEO::index_t k = 0;
            for (GEO::index_t lv = 0; lv < C.nb_v(); ++lv) {
                const GEO::index_t g = C.v_global_index(lv);
                if (g == GEO::NO_INDEX) continue;  // the plane at infinity
                const double a = C.facet_area(lv);
                if (!(a > 0)) continue;
                nb.push_back(del.periodic_vertex_real(g));
                ar.push_back(a);
                ++k;
            }
            c.nfaces[i] = k;
        }
    });
    if (next_slice > slices) throw std::runtime_error("more slices than threads");
}

// Periodic images of the points within w of a face (edges and corners included), appended after the n originals.
std::vector<double> padded(const double* xyz, std::uint64_t n, double w, std::vector<std::uint32_t>* source = nullptr) {
    std::vector<double> p(xyz, xyz + 3 * n);
    for (std::uint64_t i = 0; i < n; ++i) {
        const double* x = xyz + 3 * i;
        int lo[3], hi[3];
        for (int d = 0; d < 3; ++d) {
            lo[d] = x[d] > 1.0 - w ? -1 : 0;
            hi[d] = x[d] < w ? 1 : 0;
        }
        for (int a = lo[0]; a <= hi[0]; ++a)
            for (int b = lo[1]; b <= hi[1]; ++b)
                for (int c = lo[2]; c <= hi[2]; ++c) {
                    if (a == 0 && b == 0 && c == 0) continue;
                    p.push_back(x[0] + a);
                    p.push_back(x[1] + b);
                    p.push_back(x[2] + c);
                    if (source) source->push_back(std::uint32_t(i));
                }
    }
    return p;
}

inline double det3(const double* a, const double* b, const double* c) {
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
}

// Circumcentre of tetrahedron (a, b, c, d) (double, from the tetrahedron's frame at a).
void circumcentre4(const double* a, const double* b, const double* c, const double* e, double* o) {
    double u[3], v[3], w[3];
    for (int d = 0; d < 3; ++d) u[d] = b[d] - a[d], v[d] = c[d] - a[d], w[d] = e[d] - a[d];
    const double uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2], vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2],
                 ww = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    const double D = 2 * det3(u, v, w);
    double vxw[3] = {v[1] * w[2] - v[2] * w[1], v[2] * w[0] - v[0] * w[2], v[0] * w[1] - v[1] * w[0]};
    double wxu[3] = {w[1] * u[2] - w[2] * u[1], w[2] * u[0] - w[0] * u[2], w[0] * u[1] - w[1] * u[0]};
    double uxv[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
    for (int d = 0; d < 3; ++d) o[d] = a[d] + (uu * vxw[d] + vv * wxu[d] + ww * uxv[d]) / D;
}

// The tetrahedra of a padded Delaunay as dual::build's source: cells of the originals (index < n); neighbours keyed by
// padded index, labelled by their original (source: original of each image).
struct GeoTets {
    const GEO::Delaunay& del;
    const std::vector<GEO::index_t>& t;  // tetrahedra with an original corner
    const std::vector<std::uint32_t>& source;
    GEO::index_t n;
    std::uint64_t tets() const { return t.size(); }
    GEO::index_t id(std::uint64_t k, int a) const { return del.cell_vertex(t[k], GEO::index_t(a)); }
    bool owner(std::uint64_t k, int a) const { return id(k, a) < n; }
    std::uint64_t cell(std::uint64_t k, int a) const { return id(k, a); }
    std::uint64_t key(std::uint64_t k, int, int b) const { return id(k, b); }
    std::int64_t label(std::uint64_t k, int, int b) const {
        const GEO::index_t i = id(k, b);
        return i < n ? std::int64_t(i) : std::int64_t(source[i - n]);
    }
    void relative(std::uint64_t k, int a, int b, double* r) const {
        const double *p = del.vertex_ptr(id(k, a)), *q = del.vertex_ptr(id(k, b));
        for (int d = 0; d < 3; ++d) r[d] = q[d] - p[d];
    }
};

// The full cells of the n originals (ours, dual_cells.hpp, OpenMP).
dual::Cells padded_cells(const GEO::Delaunay& del, GEO::index_t n, const std::vector<std::uint32_t>& source) {
    std::vector<GEO::index_t> t;
    for (GEO::index_t c = 0; c < del.nb_cells(); ++c)
        for (GEO::index_t a = 0; a < 4; ++a)
            if (del.cell_vertex(c, a) < n) {
                t.push_back(c);
                break;
            }
    return dual::build(GeoTets{del, t, source, n}, n, [&](std::uint64_t i, double* x) {
        for (int d = 0; d < 3; ++d) x[d] = del.vertex_ptr(GEO::index_t(i))[d];
    });
}

std::uint64_t mark_inexact(const GEO::Delaunay& del, GEO::index_t n, double w, std::vector<double>& volume) {
    std::uint64_t bad = 0;
    for (GEO::index_t t = 0; t < del.nb_cells(); ++t) {
        GEO::index_t v[4];
        const double* p[4];
        bool any = false;
        for (int q = 0; q < 4; ++q) {
            v[q] = del.cell_vertex(t, GEO::index_t(q));
            p[q] = del.vertex_ptr(v[q]);
            any = any || v[q] < n;
        }
        if (!any) continue;
        double c[3];
        circumcentre4(p[0], p[1], p[2], p[3], c);
        double r2 = 0;
        for (int d = 0; d < 3; ++d) r2 += (p[0][d] - c[d]) * (p[0][d] - c[d]);
        const double r = std::sqrt(r2);
        bool inside = true;
        for (int d = 0; d < 3; ++d) inside = inside && c[d] - r > -w && c[d] + r < 1.0 + w;
        if (inside) continue;
        for (int q = 0; q < 4; ++q)
            if (v[q] < n && !std::isnan(volume[v[q]])) volume[v[q]] = std::numeric_limits<double>::quiet_NaN(), ++bad;
    }
    // Cells of originals that touch no tetrahedron (cannot happen for a valid padding) also NaN.
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: geogram_bench <points file> --variant=... --threads=T [--repeats=3] [--volumes=file]"
                  << " [--width=w]\n";
        return 1;
    }
    const std::string file = argv[1];
    const std::string variant = libbench::option(argc, argv, "variant", "pdel-cells");
    const int threads = std::stoi(libbench::option(argc, argv, "threads", "1"));
    const int repeats = std::stoi(libbench::option(argc, argv, "repeats", "3"));
    const double width = std::stod(libbench::option(argc, argv, "width", "0.0625"));

    GEO::initialize(GEO::GEOGRAM_INSTALL_NONE);
    GEO::Logger::instance()->unregister_all_clients();
    GEO::Logger::instance()->register_client(new StderrLogger);
    GEO::Process::set_max_threads(GEO::index_t(threads));
    const int used = int(GEO::Process::maximum_concurrent_threads());
    std::cerr << "# geogram threads: requested " << threads << ", Process::maximum_concurrent_threads() " << used
              << ", number_of_cores() " << GEO::Process::number_of_cores() << ", OMP_NUM_THREADS "
              << libbench::env("OMP_NUM_THREADS", "unset") << std::endl;
    if (used != threads) throw std::runtime_error("geogram does not use the requested thread count");

    const auto frames = libbench::read_frames(file);
    const std::uint64_t n = frames.n;
    const double* xyz = frames.frame(0);
    const GEO::index_t ni = GEO::index_t(n);

    libbench::Setup s;
    s.library = "geogram";
    s.version = version();
    s.device = "2x Xeon Platinum 8360Y";
    s.threads = used;
    s.precision = "double, exact predicates (PCK: arithmetic filters, expansions, symbolic perturbation)";

    std::vector<double> volume(n, std::numeric_limits<double>::quiet_NaN());
    std::function<void(bool)> body;  // one build; with keep, the volumes to `volume` (outside the clock)
    double t_end = 0;                 // set by body right after the timed work (before destructors)
    std::string extra;

    if (variant == "pdel" || variant == "pdel-cells") {
        const bool cells = variant == "pdel-cells";
        s.variant = cells ? "PeriodicDelaunay3d + Voronoi cells (copy_Laguerre_cell_from_Delaunay, ConvexCell)"
                          : "PeriodicDelaunay3d (Delaunay)";
        s.outputs = cells ? "Delaunay tetrahedra, volumes centroids, faces (neighbour, area)" : "Delaunay tetrahedra";
        s.periodic = "native";
        s.timed = cells ? "points in host memory to tetrahedra and cells in host memory: PeriodicDelaunay3d "
                          "set_vertices + compute, then per cell ConvexCell + compute_geometry, volume, centroid, faces"
                        : "points in host memory to periodic tetrahedra in host memory: set_vertices + compute";
        s.note = "geogram threads through OpenMP (OMPThreadManager); cells in parallel_for_slice";
        body = [&, cells](bool keep) {
            GEO::SmartPointer<GEO::PeriodicDelaunay3d> del = new GEO::PeriodicDelaunay3d(true, 1.0);
            del->use_exact_predicates_for_convex_cell(true);
            del->set_vertices(ni, xyz);
            del->compute();
            if (del->has_empty_cells()) throw std::runtime_error("empty cells");
            Cells c;
            if (cells) voronoi_cells(*del, ni, c);
            t_end = libbench::now();
            if (keep) {
                std::size_t faces = 0;
                for (auto& a : c.area) faces += a.size();
                std::ostringstream o;
                o << ",\"tetrahedra\":" << del->nb_cells() << ",\"faces\":" << faces;
                extra = o.str();
                if (cells) volume = std::move(c.volume);
            }
        };
    } else if (variant == "pdel-nonper" || variant == "pdel-nonper-faces") {
        const bool faces = variant == "pdel-nonper-faces";
        s.variant = faces ? "ParallelDelaunay3d PDEL on padded points + faces (ours)"
                          : "ParallelDelaunay3d PDEL on padded points (Delaunay)";
        s.outputs = faces ? "Delaunay tetrahedra; cells (ours): volumes centroids faces areas neighbours"
                          : "Delaunay tetrahedra";
        std::ostringstream pw;
        pw << "images: shell of width " << width;
        s.periodic = pw.str();
        s.timed = std::string("points in host memory to tetrahedra") + (faces ? " and full cells" : "") +
                  " in host memory: padding, Delaunay::create(3, PDEL), set_vertices" +
                  (faces ? ", our cells (OpenMP)" : "");
        body = [&, faces](bool keep) {
            std::vector<std::uint32_t> source;
            const std::vector<double> p = padded(xyz, n, width, &source);
            GEO::Delaunay_var del = GEO::Delaunay::create(3, "PDEL");
            del->set_vertices(GEO::index_t(p.size() / 3), p.data());
            dual::Cells c;
            if (faces) c = padded_cells(*del, ni, source);
            t_end = libbench::now();
            if (keep) {
                std::uint64_t bad = 0;
                double closure = 0;
                if (faces) {
                    volume = c.volume;
                    bad = mark_inexact(*del, ni, width, volume);
                    closure = dual::worst_closure(c);
                }
                std::ostringstream o;
                o << ",\"padded_points\":" << p.size() / 3 << ",\"tetrahedra\":" << del->nb_cells()
                  << ",\"cells_not_exact\":" << bad;
                if (faces) o << ",\"faces\":" << c.face_start.back() << ",\"worst_closure\":" << closure;
                extra = o.str();
                std::ostringstream note;
                note << "padded set " << p.size() / 3 << " points (" << p.size() / 3 - n
                     << " images); M cells/s counts the n originals; " << bad
                     << " originals with a circumball leaving the padded cube (NaN volume)";
                s.note = note.str();
            }
        };
    } else {
        throw std::runtime_error("unknown variant " + variant);
    }

    libbench::run_protocol(
        s, file, n, "build", repeats,
        [&] {
            libbench::HostPeak hp;
            hp.start();
            body(true);  // the untimed run: peak memory, and the volumes and counts (kept after the clock)
            return hp.stop();
        },
        [&] {
            const double t0 = libbench::now();
            body(false);
            return t_end - t0;
        });
    if (variant == "pdel-cells" || variant == "pdel-nonper-faces")
        libbench::write_volumes(libbench::option(argc, argv, "volumes"), volume);
    // Sizes of the output (from the first, untimed run), one extra line per invocation as library_bench does.
    std::cout << "{\"library\":\"geogram\",\"variant\":\"" << s.variant << "\",\"input\":\""
              << libbench::input_name(file) << "\",\"n\":" << n << ",\"threads\":" << used << extra << "}"
              << std::endl;
    return 0;
}
