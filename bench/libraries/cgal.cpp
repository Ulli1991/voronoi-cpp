// CGAL in the library benchmark (CONTRACT: /raven/u/uli/libbench-src/CONTRACT.md; records: libbench.hpp).
//
//   cgal_bench p3dt  <points file> [--repeats=3] [--volumes=file]          Periodic_3_Delaunay_triangulation_3, serial
//   cgal_bench dt3   <points file> --threads=T [--repeats=3] [--volumes=file]   Delaunay_triangulation_3, Parallel_tag
//   cgal_bench step  <trajectory file> [--steps=10] [--budget=600] [--volumes=file]   P3DT moving mesh
//
// Each build mode records two variants on the same points: "Delaunay" (the triangulation alone) and
// "Delaunay + faces (ours)" (the triangulation, then the full Voronoi cells from its tetrahedra: volumes, centroids,
// faces with neighbours and area vectors, by dual_cells.hpp on OpenMP threads, as many as the library's). CGAL gives
// Delaunay tetrahedra only (dual(v) returns a cell's polygons, not its volume or areas), so the cells step is ours.
//
// Kernel: Exact_predicates_inexact_constructions_kernel (filtered exact orientation and in-sphere predicates,
// double constructions). Points are exactly the doubles of the point file (on the 2^-30 grid).

#include "dual_cells.hpp"
#include "libbench.hpp"

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Periodic_3_Delaunay_triangulation_3.h>
#include <CGAL/Periodic_3_Delaunay_triangulation_traits_3.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/version.h>

#include <boost/version.hpp>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/version.h>

#include <array>
#include <atomic>
#include <memory>
#include <unordered_map>
#include <omp.h>

namespace {

using K = CGAL::Exact_predicates_inexact_constructions_kernel;

const char* const device_name = "2x Xeon Platinum 8360Y";

// ---------------------------------------------------------------------------------------------------------------
// Dual volumes. The Voronoi cell of a vertex a is the union, over the Delaunay tetrahedra T at a, the edges (a, x)
// of T and the two faces (a, x, y) of T at that edge, of the signed tetrahedra (p_a, midpoint(a, x),
// circumcentre(a, x, y), circumcentre(T)) (the flags of the Delaunay complex mapped to the dual). Signed, the sum is
// exact also where a circumcentre lies outside its simplex. Per tetrahedron: 1 circumcentre, 4 face circumcentres,
// 6 midpoints and 24 signed volumes, all relative to vertex 0 (doubles of points within a box of edge 2: the
// differences are exact on the 2^-30 grid). The result goes to add(local vertex 0..3, volume).

struct V {
    double x, y, z;
};
inline V operator-(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V operator+(V a, V b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V operator*(double s, V a) { return {s * a.x, s * a.y, s * a.z}; }
inline double dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V cross(V a, V b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double det(V a, V b, V c) { return dot(a, cross(b, c)); }

// Circumcentre of the triangle (0, b, c).
inline V tri_centre(V b, V c) {
    const V n = cross(b, c);
    return (0.5 / dot(n, n)) * cross(dot(b, b) * c - dot(c, c) * b, n);
}

// p[0..3] are the tetrahedron's vertices (any orientation). Also returns the circumcentre and radius (squared) in
// `centre` and `r2` (absolute coordinates) for the padding certificate.
template <class Add>
inline void dual_volumes(const V* p, Add&& add, V* centre = nullptr, double* r2 = nullptr) {
    const V b = p[1] - p[0], c = p[2] - p[0], d = p[3] - p[0];
    const double six = det(b, c, d);  // 6 x signed volume
    const V C = (0.5 / six) * (dot(b, b) * cross(c, d) + dot(c, c) * cross(d, b) + dot(d, d) * cross(b, c));
    if (centre) {
        *centre = p[0] + C;
        *r2 = dot(C, C);
    }
    const V q[4] = {{0, 0, 0}, b, c, d};
    // F[z]: circumcentre of the face opposite vertex z.
    V F[4];
    F[0] = b + tri_centre(c - b, d - b);
    F[1] = tri_centre(c, d);
    F[2] = tri_centre(b, d);
    F[3] = tri_centre(b, c);
    // For vertex a, (a, base[a]) is an even permutation of (0, 1, 2, 3).
    static constexpr int base[4][3] = {{1, 2, 3}, {2, 0, 3}, {0, 1, 3}, {2, 1, 0}};
    const double sign = six > 0 ? 1.0 / 6.0 : -1.0 / 6.0;
    for (int a = 0; a < 4; ++a) {
        const int i = base[a][0], j = base[a][1], k = base[a][2];
        const V pa = q[a];
        const V ca = C - pa;
        const V mi = 0.5 * (q[i] - pa), mj = 0.5 * (q[j] - pa), mk = 0.5 * (q[k] - pa);
        const V fi = F[i] - pa, fj = F[j] - pa, fk = F[k] - pa;
        // (x, y, z) over the permutations of (i, j, k); term det(m_x, F_z, C), sign of the permutation.
        const double v = det(mi, fk, ca) + det(mj, fi, ca) + det(mk, fj, ca)   // (i,j,k) (j,k,i) (k,i,j)
                         - det(mj, fk, ca) - det(mi, fj, ca) - det(mk, fi, ca);  // (j,i,k) (i,k,j) (k,j,i)
        add(a, sign * v);
    }
}

std::string version() { return CGAL_VERSION_STR; }

libbench::Setup setup(const std::string& library, const std::string& variant, int threads) {
    libbench::Setup s;
    s.library = library;
    s.variant = variant;
    s.version = "CGAL " + version() + ", Boost " + std::to_string(BOOST_VERSION / 100000) + "." +
                std::to_string(BOOST_VERSION / 100 % 1000) + ", oneTBB " + std::to_string(TBB_VERSION_MAJOR) + "." +
                std::to_string(TBB_VERSION_MINOR);
    s.device = device_name;
    s.threads = threads;
    s.precision = "double, Epick (exact predicates, inexact constructions); our cells in double";
    return s;
}

// ---------------------------------------------------------------------------------------------------------------
// Periodic_3_Delaunay_triangulation_3 (Caroli and Teillaud): the periodic unit cube natively, serial.

using PGT = CGAL::Periodic_3_Delaunay_triangulation_traits_3<K>;
struct PInfo {
    double volume;
    std::uint32_t index;
};
using PVb = CGAL::Triangulation_vertex_base_with_info_3<
    PInfo, PGT, CGAL::Triangulation_vertex_base_3<PGT, CGAL::Periodic_3_triangulation_ds_vertex_base_3<>>>;
using PCb = CGAL::Triangulation_cell_base_3<PGT, CGAL::Periodic_3_triangulation_ds_cell_base_3<>>;
using PTds = CGAL::Triangulation_data_structure_3<PVb, PCb>;
using P3DT = CGAL::Periodic_3_Delaunay_triangulation_3<PGT, PTds>;

std::unique_ptr<P3DT> p3dt_build(const double* xyz, std::uint64_t n) {
    std::vector<P3DT::Point> points(n);
    for (std::uint64_t i = 0; i < n; ++i) points[i] = P3DT::Point(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
    auto tr = std::make_unique<P3DT>(P3DT::Iso_cuboid(0, 0, 0, 1, 1, 1));
    tr->insert(points.begin(), points.end(), /*is_large_point_set=*/true);  // dummy points, spatial sort
    return tr;
}

// The periodic tetrahedra as dual::build's source: every corner is a cell (vertex ids are numbered in the clock, in
// vertex order: info().index); neighbours keyed by id and periodic offset relative to the corner, unwrapped positions.
struct PTet {
    std::uint32_t id[4];
    std::int8_t off[4][3];
    double x[4][3];
};
struct PTets {
    const std::vector<PTet>& t;
    std::uint64_t tets() const { return t.size(); }
    bool owner(std::uint64_t, int) const { return true; }
    std::uint64_t cell(std::uint64_t k, int a) const { return t[k].id[a]; }
    std::uint64_t key(std::uint64_t k, int a, int b) const {
        std::uint64_t code = 0;
        for (int d = 0; d < 3; ++d) code = code * 5 + std::uint64_t(t[k].off[b][d] - t[k].off[a][d] + 2);
        return std::uint64_t(t[k].id[b]) * 125 + code;
    }
    std::int64_t label(std::uint64_t k, int, int b) const { return t[k].id[b]; }
    void relative(std::uint64_t k, int a, int b, double* r) const {
        for (int d = 0; d < 3; ++d) r[d] = t[k].x[b][d] - t[k].x[a][d];
    }
};

// The full cells of a periodic triangulation, cells in the triangulation's vertex numbering (info().index, set
// here when `number` is true; the step keeps the input numbering).
dual::Cells p3dt_cells(P3DT& tr, bool number) {
    std::uint32_t k = 0;
    std::vector<std::array<double, 3>> position(tr.number_of_vertices());
    for (auto v = tr.vertices_begin(); v != tr.vertices_end(); ++v) {
        if (number) v->info().index = k++;
        const auto& q = v->point();
        position[v->info().index] = {q.x(), q.y(), q.z()};
    }
    std::vector<PTet> tets;
    tets.reserve(tr.number_of_cells());
    for (auto c = tr.cells_begin(); c != tr.cells_end(); ++c) {
        PTet t;
        for (int i = 0; i < 4; ++i) {
            t.id[i] = c->vertex(i)->info().index;
            const auto o = tr.periodic_point(c, i).second;
            t.off[i][0] = std::int8_t(o.x()), t.off[i][1] = std::int8_t(o.y()), t.off[i][2] = std::int8_t(o.z());
            const auto q = tr.point(c, i);
            t.x[i][0] = q.x(), t.x[i][1] = q.y(), t.x[i][2] = q.z();
        }
        tets.push_back(t);
    }
    return dual::build(PTets{tets}, position.size(), [&](std::uint64_t i, double* x) {
        for (int d = 0; d < 3; ++d) x[d] = position[i][d];
    });
}

// Input index of every vertex, by its coordinates (outside the clock).
void p3dt_index(P3DT& tr, const double* xyz, std::uint64_t n) {
    struct H {
        std::size_t operator()(const std::array<double, 3>& a) const {
            std::size_t h = 0;
            for (double d : a) h = h * 0x9E3779B97F4A7C15ull ^ std::hash<double>()(d);
            return h;
        }
    };
    std::unordered_map<std::array<double, 3>, std::uint32_t, H> where;
    where.reserve(n);
    for (std::uint64_t i = 0; i < n; ++i) where[{xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]}] = std::uint32_t(i);
    for (auto v = tr.vertices_begin(); v != tr.vertices_end(); ++v) {
        const auto& q = v->point();
        const auto it = where.find({q.x(), q.y(), q.z()});
        if (it == where.end()) throw std::runtime_error("vertex not among the input points");
        v->info().index = it->second;
    }
}

const char* const p3dt_name = "CGAL Periodic_3_Delaunay_triangulation_3";

int p3dt(int argc, char** argv) {
    const std::string file = argv[2];
    const int repeats = std::stoi(libbench::option(argc, argv, "repeats", "3"));
    const auto frames = libbench::read_frames(file);
    const std::uint64_t n = frames.n;
    const double* xyz = frames.frame(0);
    omp_set_num_threads(1);  // a serial library: our cells step on one core too
    for (const bool volumes : {false, true}) {
        auto s = setup(p3dt_name, volumes ? "Delaunay + faces (ours)" : "Delaunay", 1);
        s.outputs = volumes ? "Delaunay tetrahedra; cells (ours): volumes centroids faces areas neighbours"
                            : "Delaunay tetrahedra";
        s.periodic = "native (periodic unit cube, 1-sheeted cover)";
        s.timed = volumes ? "points in host memory to triangulation and full cells in host memory (Point conversion, "
                            "range insert with spatial sort, vertex numbering, our cells from the periodic tetrahedra)"
                          : "points in host memory to triangulation in host memory (Point conversion, range insert "
                            "with spatial sort and dummy points)";
        s.note = "range insert, is_large_point_set=true; serial (class not parallel); cells numbered in the "
                 "triangulation's vertex order; destruction outside the clock";
        dual::Cells cells;
        auto body = [&](bool copy) {
            auto tr = p3dt_build(xyz, n);
            if (volumes) cells = p3dt_cells(*tr, true);
            if (copy) {
                if (!tr->is_1_cover()) throw std::runtime_error("not 1-sheeted");
                std::cerr << "# p3dt " << file << ": " << tr->number_of_vertices() << " vertices, "
                          << tr->number_of_cells() << " cells\n";
                if (volumes) {
                    // Cell k is the k-th vertex in iteration order; the volumes file is in input order.
                    p3dt_index(*tr, xyz, n);
                    std::vector<double> v(n, std::nan(""));
                    std::size_t k = 0;
                    for (auto u = tr->vertices_begin(); u != tr->vertices_end(); ++u) v[u->info().index] = cells.volume[k++];
                    long double sum = 0;
                    for (double x : v) sum += x;
                    std::cerr << "# p3dt volume sum - 1 = " << double(sum - 1.0L) << ", faces per cell "
                              << double(cells.face_start.back()) / double(n) << ", worst closure "
                              << dual::worst_closure(cells) << ", flat " << cells.flat << "\n";
                    libbench::write_volumes(libbench::option(argc, argv, "volumes"), v);
                }
            }
            return tr;  // destroyed by the caller after the clock
        };
        libbench::run_protocol(
            s, file, n, "build", repeats,
            [&] {
                libbench::HostPeak peak;
                peak.start();
                auto tr = body(true);
                const double p = peak.stop();
                return p;
            },
            [&] {
                const double t0 = libbench::now();
                auto tr = body(false);
                const double t = libbench::now() - t0;
                return t;
            });
    }
    return 0;
}

// Moving mesh: build frame 0 (untimed), then per frame f = 1.. move every vertex (remove + insert with a hint: the
// cell of a former neighbour; P3DT::move_point is remove + insert without a hint) and recompute the dual volumes,
// timed per step. Then, for comparison, a full rebuild (range insert + volumes) per frame.
int step(int argc, char** argv) {
    const std::string file = argv[2];
    const auto frames = libbench::read_frames(file);
    const std::uint64_t n = frames.n;
    const int steps = std::min<int>(std::stoi(libbench::option(argc, argv, "steps", "10")), int(frames.frames) - 1);
    const double budget = std::stod(libbench::option(argc, argv, "budget", "600"));
    auto wrapped = [&](std::uint64_t f) {
        std::vector<double> x(frames.frame(f), frames.frame(f) + 3 * n);
        for (double& c : x) c -= std::floor(c);
        for (double& c : x)
            if (c >= 1.0) c = 0.0;
        return x;
    };
    omp_set_num_threads(1);
    auto s = setup(p3dt_name, "moving mesh: remove + insert (hinted) per point, then faces (ours)", 1);
    s.outputs = "Delaunay tetrahedra; cells (ours): volumes centroids faces areas neighbours";
    s.periodic = "native (periodic unit cube, 1-sheeted cover)";
    s.timed = "new positions in host memory to updated triangulation and full cells, per step";
    s.note = "every vertex moved each step (no test for unchanged stars); hint = cell of a former neighbour";
    const std::vector<double> x0 = wrapped(0);
    auto tr = p3dt_build(x0.data(), n);
    p3dt_index(*tr, x0.data(), n);
    std::vector<P3DT::Vertex_handle> handle(n);
    for (auto v = tr->vertices_begin(); v != tr->vertices_end(); ++v) handle[v->info().index] = v;
    double total = 0;
    int done = 0;
    dual::Cells last;
    for (int f = 1; f <= steps; ++f) {
        const std::vector<double> x = wrapped(f);
        const double t0 = libbench::now();
        std::vector<P3DT::Vertex_handle> around;
        for (std::uint64_t i = 0; i < n; ++i) {
            around.clear();
            tr->adjacent_vertices(handle[i], std::back_inserter(around));
            const P3DT::Vertex_handle near = around.front();
            tr->remove(handle[i]);
            handle[i] = tr->insert(P3DT::Point(x[3 * i], x[3 * i + 1], x[3 * i + 2]), near->cell());
            handle[i]->info().index = std::uint32_t(i);
        }
        last = p3dt_cells(*tr, false);
        const double seconds = libbench::now() - t0;
        libbench::record(s, file, libbench::input_name(file), n, "step", f, seconds, -1);
        total += seconds;
        done = f;
        if (total + seconds > budget && f < steps) {
            std::cerr << "# step: stopped after " << f << " steps, " << total << " s (budget " << budget << " s)\n";
            break;
        }
    }
    if (!tr->is_1_cover()) throw std::runtime_error("not 1-sheeted after moves");
    {
        const std::vector<double>& v = last.volume;  // in input order (the step keeps it)
        long double sum = 0;
        for (double c : v) sum += c;
        std::cerr << "# step volume sum - 1 = " << double(sum - 1.0L) << " after " << done << " steps\n";
        std::string out = libbench::option(argc, argv, "volumes");
        if (!out.empty()) libbench::write_volumes(out + "_step" + std::to_string(done) + ".f64", v);
    }
    tr.reset();
    auto r = setup(p3dt_name, "moving mesh: full rebuild per step (range insert + faces (ours))", 1);
    r.outputs = s.outputs;
    r.periodic = s.periodic;
    r.timed = "points in host memory to triangulation and full cells, a new triangulation per step";
    r.note = "is_large_point_set=true; destruction of the previous triangulation outside the clock";
    for (int f = 1; f <= steps; ++f) {
        const std::vector<double> x = wrapped(f);
        const double t0 = libbench::now();
        auto t = p3dt_build(x.data(), n);
        const dual::Cells c = p3dt_cells(*t, true);
        const double seconds = libbench::now() - t0;
        libbench::record(r, file, libbench::input_name(file), n, "step", f, seconds, -1);
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------
// Delaunay_triangulation_3 with Parallel_tag (TBB, Spatial_lock_grid_3), non-periodic: the points and their periodic
// images within a shell of width w around the unit cube. The cells of the original points are exact when every
// tetrahedron at an original point has its circumsphere inside [-w, 1+w]^3 (the sphere is then empty of every
// periodic image, since all images in that region are in the set): checked on the untimed run; w starts at 4 mean
// spacings and doubles until the check passes.

using DVb = CGAL::Triangulation_vertex_base_with_info_3<std::uint32_t, K>;
using DCb = CGAL::Delaunay_triangulation_cell_base_3<K>;
using DTds = CGAL::Triangulation_data_structure_3<DVb, DCb, CGAL::Parallel_tag>;
using DT = CGAL::Delaunay_triangulation_3<K, DTds>;  // Compact_location: the hierarchy (Fast_location) is serial

struct Padded {
    std::vector<std::pair<DT::Point, std::uint32_t>> points;  // info: index < n original, else n + image number
    std::vector<std::uint32_t> source;                        // image number -> original index
};

Padded pad(const double* xyz, std::uint64_t n, double w) {
    // Images: per point and axis the shifts {0} plus +1 if x < w, -1 if x >= 1 - w; all combinations but (0,0,0).
    auto shifts = [&](std::uint64_t i, int a, int* s) {
        const double x = xyz[3 * i + a];
        int m = 0;
        s[m++] = 0;
        if (x < w) s[m++] = 1;
        if (x >= 1 - w) s[m++] = -1;
        return m;
    };
    std::vector<std::uint32_t> count(n + 1, 0);
    tbb::parallel_for(tbb::blocked_range<std::uint64_t>(0, n), [&](const auto& r) {
        int s[3];
        for (std::uint64_t i = r.begin(); i != r.end(); ++i)
            count[i + 1] = shifts(i, 0, s) * shifts(i, 1, s) * shifts(i, 2, s) - 1;
    });
    for (std::uint64_t i = 0; i < n; ++i) count[i + 1] += count[i];
    Padded p;
    p.points.resize(n + count[n]);
    p.source.resize(count[n]);
    tbb::parallel_for(tbb::blocked_range<std::uint64_t>(0, n), [&](const auto& r) {
        int sx[3], sy[3], sz[3];
        for (std::uint64_t i = r.begin(); i != r.end(); ++i) {
            const double x = xyz[3 * i], y = xyz[3 * i + 1], z = xyz[3 * i + 2];
            p.points[i] = {DT::Point(x, y, z), std::uint32_t(i)};
            const int mx = shifts(i, 0, sx), my = shifts(i, 1, sy), mz = shifts(i, 2, sz);
            std::uint64_t k = count[i];
            for (int a = 0; a < mx; ++a)
                for (int b = 0; b < my; ++b)
                    for (int c = 0; c < mz; ++c) {
                        if (a == 0 && b == 0 && c == 0) continue;
                        p.points[n + k] = {DT::Point(x + sx[a], y + sy[b], z + sz[c]), std::uint32_t(n + k)};
                        p.source[k] = std::uint32_t(i);
                        ++k;
                    }
        }
    });
    return p;
}

std::unique_ptr<DT> dt_build(const Padded& p, double w, std::unique_ptr<DT::Lock_data_structure>& lock) {
    lock = std::make_unique<DT::Lock_data_structure>(CGAL::Bbox_3(-w, -w, -w, 1 + w, 1 + w, 1 + w), 50);
    auto dt = std::make_unique<DT>(lock.get());
    dt->insert(p.points.begin(), p.points.end());  // insert_with_info: spatial sort, then TBB parallel_for
    return dt;
}

// Volumes of the original cells (index < n). The finite cells are listed serially (the cell container has no
// parallel range), then summed in parallel with atomic adds. With `check`, also counts tetrahedra at an original
// point whose circumsphere leaves [-w, 1+w]^3.
std::uint64_t dt_volumes(DT& dt, std::uint64_t n, std::vector<double>& volume, bool check, double w) {
    std::vector<DT::Cell_handle> cells;
    cells.reserve(dt.number_of_finite_cells());
    for (auto c = dt.finite_cells_begin(); c != dt.finite_cells_end(); ++c) cells.push_back(c);
    std::fill(volume.begin(), volume.end(), 0.0);
    std::atomic<std::uint64_t> outside{0};
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, cells.size()), [&](const auto& r) {
        std::uint64_t out = 0;
        for (std::size_t k = r.begin(); k != r.end(); ++k) {
            const DT::Cell_handle c = cells[k];
            std::uint32_t idx[4];
            bool any = false;
            V p[4];
            for (int i = 0; i < 4; ++i) {
                idx[i] = c->vertex(i)->info();
                any |= idx[i] < n;
                const auto& q = c->vertex(i)->point();
                p[i] = {q.x(), q.y(), q.z()};
            }
            if (!any) continue;
            V centre;
            double r2;
            dual_volumes(
                p,
                [&](int a, double v) {
                    if (idx[a] < n) std::atomic_ref<double>(volume[idx[a]]).fetch_add(v, std::memory_order_relaxed);
                },
                check ? &centre : nullptr, &r2);
            if (check) {
                const double rad = std::sqrt(r2);
                if (centre.x - rad < -w || centre.y - rad < -w || centre.z - rad < -w || centre.x + rad > 1 + w ||
                    centre.y + rad > 1 + w || centre.z + rad > 1 + w)
                    ++out;
            }
        }
        outside += out;
    });
    return outside;
}

// The padded triangulation's finite tetrahedra at an original point as dual::build's source: cells of the originals
// (info < n), neighbours keyed by padded index and labelled by their original.
struct DTets {
    const std::vector<DT::Cell_handle>& c;
    const std::vector<std::uint32_t>& source;
    std::uint64_t n;
    std::uint64_t tets() const { return c.size(); }
    std::uint32_t id(std::uint64_t k, int a) const { return c[k]->vertex(a)->info(); }
    bool owner(std::uint64_t k, int a) const { return id(k, a) < n; }
    std::uint64_t cell(std::uint64_t k, int a) const { return id(k, a); }
    std::uint64_t key(std::uint64_t k, int, int b) const { return id(k, b); }
    std::int64_t label(std::uint64_t k, int, int b) const {
        const std::uint32_t i = id(k, b);
        return i < n ? i : source[i - n];
    }
    void relative(std::uint64_t k, int a, int b, double* r) const {
        const auto &p = c[k]->vertex(a)->point(), &q = c[k]->vertex(b)->point();
        r[0] = q.x() - p.x(), r[1] = q.y() - p.y(), r[2] = q.z() - p.z();
    }
};

// The full cells of the originals (cell list serial: the container has no parallel range; cells on OpenMP threads).
dual::Cells dt_cells(DT& dt, const Padded& p, std::uint64_t n, const double* xyz) {
    std::vector<DT::Cell_handle> cells;
    cells.reserve(dt.number_of_finite_cells());
    for (auto c = dt.finite_cells_begin(); c != dt.finite_cells_end(); ++c)
        for (int i = 0; i < 4; ++i)
            if (c->vertex(i)->info() < n) {
                cells.push_back(c);
                break;
            }
    return dual::build(DTets{cells, p.source, n}, n, [&](std::uint64_t i, double* x) {
        for (int d = 0; d < 3; ++d) x[d] = xyz[3 * i + d];
    });
}

int dt3(int argc, char** argv) {
    const std::string file = argv[2];
    const int repeats = std::stoi(libbench::option(argc, argv, "repeats", "3"));
    const int threads = std::stoi(libbench::option(argc, argv, "threads", "72"));
    tbb::global_control control(tbb::global_control::max_allowed_parallelism, threads);
    omp_set_num_threads(threads);  // our cells step on as many threads as CGAL's
    const auto frames = libbench::read_frames(file);
    const std::uint64_t n = frames.n;
    const double* xyz = frames.frame(0);
    const double h = std::cbrt(1.0 / double(n));
    std::vector<double> volume(n);
    // The shell width, by the certificate (untimed).
    double w = 4 * h;
    std::uint64_t extra = 0;
    for (;;) {
        const Padded p = pad(xyz, n, w);
        std::unique_ptr<DT::Lock_data_structure> lock;
        auto dt = dt_build(p, w, lock);
        const std::uint64_t outside = dt_volumes(*dt, n, volume, true, w);
        extra = p.source.size();
        std::cerr << "# dt3 " << file << ": w = " << w / h << " spacings, " << extra << " images, " << outside
                  << " tetrahedra at original points with circumspheres outside the padded box\n";
        if (outside == 0) break;
        w *= 2;
        if (w > 0.5) throw std::runtime_error("shell wider than half the box");
    }
    {
        long double sum = 0;
        for (double v : volume) sum += v;
        std::cerr << "# dt3 volume sum - 1 = " << double(sum - 1.0L) << "\n";
    }
    std::ostringstream note;
    note.precision(4);
    note << "periodic images in a shell of width " << w / h << " mean spacings (" << w << "), " << extra
         << " extra points (" << 100.0 * double(extra) / double(n)
         << "% of n), certified: every tetrahedron at an original point has its circumsphere inside the padded box; "
            "Parallel_tag with Spatial_lock_grid_3 (50^3), Compact_location (Fast_location hierarchy is not "
            "thread-safe); destruction outside the clock";
    for (const bool volumes : {false, true}) {
        auto s = setup("CGAL Delaunay_triangulation_3 (Parallel_tag, TBB)",
                       volumes ? "Delaunay + faces (ours)" : "Delaunay", threads);
        s.outputs = volumes ? "Delaunay tetrahedra; cells (ours): volumes centroids faces areas neighbours"
                            : "Delaunay tetrahedra";
        s.periodic = "images: shell of width " + std::to_string(w) + " (" + std::to_string(extra) + " extra points)";
        s.timed = std::string("points in host memory to ") +
                  (volumes ? "triangulation and full cells of the n originals in host memory (padding, insert "
                             "with spatial sort, our cells: serial cell list, OpenMP cells)"
                           : "triangulation in host memory (padding, insert with spatial sort, parallel insertion)");
        s.note = note.str();
        dual::Cells cells;
        auto body = [&](std::unique_ptr<DT::Lock_data_structure>& lock) {
            const Padded p = pad(xyz, n, w);
            auto dt = dt_build(p, w, lock);
            if (volumes) cells = dt_cells(*dt, p, n, xyz);
            return dt;
        };
        libbench::run_protocol(
            s, file, n, "build", repeats,
            [&] {
                libbench::HostPeak peak;
                peak.start();
                std::unique_ptr<DT::Lock_data_structure> lock;
                auto dt = body(lock);
                const double b = peak.stop();
                if (volumes) {
                    std::cerr << "# dt3 cells: faces per cell " << double(cells.face_start.back()) / double(n)
                              << ", worst closure " << dual::worst_closure(cells) << ", flat " << cells.flat << "\n";
                    libbench::write_volumes(libbench::option(argc, argv, "volumes"), cells.volume);
                }
                return b;
            },
            [&] {
                std::unique_ptr<DT::Lock_data_structure> lock;
                const double t0 = libbench::now();
                auto dt = body(lock);
                return libbench::now() - t0;
            });
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 2 ? argv[1] : "";
    try {
        if (mode == "p3dt") return p3dt(argc, argv);
        if (mode == "dt3") return dt3(argc, argv);
        if (mode == "step") return step(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "cgal_bench: " << e.what() << "\n";
        return 1;
    }
    std::cerr << "usage: cgal_bench p3dt <points> [--repeats=3] [--volumes=file]\n"
                 "       cgal_bench dt3 <points> --threads=T [--repeats=3] [--volumes=file]\n"
                 "       cgal_bench step <trajectory> [--steps=10] [--budget=600] [--volumes=prefix]\n";
    return 2;
}
