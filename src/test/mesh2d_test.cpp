// The 2D mesh (voronoi::Mesh<2>) against oracles that share no code with it: predicates, brute-force triangulation and
// cells, moving meshes equal to fresh builds bit for bit, and refusals. Exercises voronoi::Mesh<2> (update,
// update_cells, clip); self-contained (no support.hpp), with its own oracles.

#include "check.hpp"
#include "voronoi_mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

// Sanitizers cost about 11x: the longest runs shrink there, the same code paths.
#ifdef __SANITIZE_ADDRESS__
constexpr int large_n = 1 << 17, moving_steps = 3;
#else
constexpr int large_n = 1 << 20, moving_steps = 10;
#endif

using Exec = Kokkos::DefaultExecutionSpace;
using Memory = Exec::memory_space;
using Mesh = voronoi::Mesh<2, Exec>;
using Box = exact_predicates::BoxN<2>;
using X = Kokkos::View<double* [2], Memory>;
using Points = std::vector<std::array<double, 2>>;
namespace ep = exact_predicates;
using i128 = ep::i128;
using Long = long double;

// Host copy of a device view. Used by: every checker that reads the mesh.
template <class V>
auto host(const V& v) {
    auto h = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h, v);
    return h;
}

X device(const Points& p) {
    X v("x", p.size());
    auto h = Kokkos::create_mirror_view(v);
    for (std::size_t i = 0; i < p.size(); ++i)
        for (int a = 0; a < 2; ++a) h(i, a) = p[i][a];
    Kokkos::deep_copy(v, h);
    return v;
}

// Points in the box, snapped as the mesh snaps them.
Points on_grid(Points p, const Box& box) {
    for (auto& q : p)
        for (int a = 0; a < 2; ++a) {
            const double wrapped = q[a] - box.side[a] * std::floor(q[a] / box.side[a]);
            q[a] = ep::snap(wrapped, box, a) * box.step;
        }
    return p;
}

// Points of the given kind (random, jittered, clustered, lattice, hexagonal, wave) in the box, on its grid.
// Used by: test_static, test_large, test_refusals, test_clip, test_moving (not support::make_points).
Points make_points(const std::string& kind, int n, const Box& box) {
    std::mt19937_64 rng(20261105);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double> normal;
    const double lx = box.side[0], ly = box.side[1];
    const double spacing = std::sqrt(lx * ly / n);
    const int nx = std::max(4, static_cast<int>(std::lround(lx / spacing))), ny = std::max(4, static_cast<int>(std::lround(ly / spacing)));
    Points p;
    if (kind == "random") {
        for (int i = 0; i < n; ++i) p.push_back({unit(rng) * lx, unit(rng) * ly});
    } else if (kind == "clustered") {
        std::vector<std::array<double, 2>> centres;
        for (int c = 0; c < 8; ++c) centres.push_back({unit(rng) * lx, unit(rng) * ly});
        for (int i = 0; i < n; ++i) {
            if (i % 2 == 0) {
                p.push_back({unit(rng) * lx, unit(rng) * ly});
            } else {
                const auto& c = centres[(i / 2) % 8];
                const double x = c[0] + 0.1 * std::min(lx, ly) * normal(rng), y = c[1] + 0.1 * std::min(lx, ly) * normal(rng);
                p.push_back({x, y});
            }
        }
    } else if (kind == "hexagonal") {
        const int rows = 2 * std::max(2, ny / 2);
        for (int j = 0; j < rows; ++j)
            for (int i = 0; i < nx; ++i) p.push_back({(i + 0.5 * (j % 2)) * lx / nx, j * ly / rows});
    } else {  // lattices: jittered, wave, lattice
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j) {
                double x = (i + 0.5) * lx / nx, y = (j + 0.5) * ly / ny;
                if (kind == "jittered") {
                    x += 0.4 * lx / nx * (unit(rng) - 0.5);
                    y += 0.4 * ly / ny * (unit(rng) - 0.5);
                } else if (kind == "wave") {
                    x += 1e-5 * lx / nx * std::sin(2 * M_PI * y / ly);
                }
                p.push_back({x, y});
            }
    }
    return on_grid(p, box);
}

// 3x3 determinant in 128 bits. Used by: textbook_in_circle.
i128 det3(i128 a, i128 b, i128 c, i128 d, i128 e, i128 f, i128 g, i128 h, i128 i) {
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

// In-circle of a, b, c (counter-clockwise), d: > 0 inside, by the textbook
// determinant over differences from d. Small coordinates only. Used by: test_predicates, check_triangles.
i128 textbook_in_circle(const ep::Offset2* p) {
    i128 m[3][3];
    for (int r = 0; r < 3; ++r) {
        const i128 x = p[r][0] - p[3][0], y = p[r][1] - p[3][1];
        m[r][0] = x;
        m[r][1] = y;
        m[r][2] = x * x + y * y;
    }
    return det3(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]);
}

// ep::in_circle (and cross) against the textbook determinant, every point ordering, and cocircular ties.
void test_predicates(Check& check) {
    std::mt19937_64 rng(7);
    long long wrong = 0, ties = 0, inconsistent = 0, cases = 0;
    for (int scale : {3, 16, 1 << 20, 1 << 29}) {
        std::uniform_int_distribution<int> digit(-scale, scale);
        for (int trial = 0; trial < 200000; ++trial) {
            ep::Offset2 p[4];
            for (auto& q : p) q = {digit(rng), digit(rng)};
            const ep::i64 o = ep::cross(ep::minus(p[1], p[0]), ep::minus(p[2], p[0]));
            if (o == 0 || (p[3][0] == p[0][0] && p[3][1] == p[0][1])) continue;
            if ((p[3][0] == p[1][0] && p[3][1] == p[1][1]) || (p[3][0] == p[2][0] && p[3][1] == p[2][1])) continue;
            int id[4] = {5, 1, 7, 3};
            const bool inside = ep::in_circle(p, id);
            ++cases;
            if (std::abs(scale) <= 1 << 20) {
                const i128 det = textbook_in_circle(p) * (o > 0 ? 1 : -1);
                if (det == 0) {
                    ++ties;
                    // Every ordering of the same four points agrees (flat orderings excepted).
                    int order[4] = {0, 1, 2, 3};
                    do {
                        const ep::Offset2 q[4] = {p[order[0]], p[order[1]], p[order[2]], p[order[3]]};
                        const int ids[4] = {id[order[0]], id[order[1]], id[order[2]], id[order[3]]};
                        const ep::i64 flat = ep::cross(ep::minus(q[1], q[0]), ep::minus(q[2], q[0]));
                        if (flat == 0) continue;
                        // Swapping the tested point (d in (a, b, c) versus a in (b, c, d)) follows the lifted facet's
                        // orientation rule, but a permutation of the triangle alone must not change the answer.
                        if (order[3] == 3 && ep::in_circle(q, ids) != inside) ++inconsistent;
                    } while (std::next_permutation(order, order + 4));
                } else if ((det > 0) != inside) {
                    ++wrong;
                }
            }
        }
    }
    check.expect(wrong == 0, "in_circle agrees with an integer determinant on " + std::to_string(cases) + " cases");
    check.expect(ties > 1000, "ties are exercised (" + std::to_string(ties) + ")");
    check.expect(inconsistent == 0, "the perturbed test does not depend on the triangle's corner order");
    // 4 cocircular points: diagonal ac is judged alike from its two triangles, and exactly one of the two diagonals is
    // Delaunay under the perturbation.
    const ep::Offset2 a{0, 0}, b{2, 0}, c{2, 2}, d{0, 2};
    const ep::Offset2 abc_d[4] = {a, b, c, d}, acd_b[4] = {a, c, d, b}, abd_c[4] = {a, b, d, c};
    const int id1[4] = {0, 1, 2, 3}, id2[4] = {0, 2, 3, 1}, id3[4] = {0, 1, 3, 2};
    check.expect(ep::in_circle(abc_d, id1) == ep::in_circle(acd_b, id2), "cocircular quad: one verdict per diagonal");
    check.expect(ep::in_circle(abc_d, id1) != ep::in_circle(abd_c, id3), "cocircular quad: one diagonal wins");
}

// One brute-force cell: volume, centroid and faces keyed by the other particle. Used by: brute_cell, check_cells.
struct Cell {
    Long volume = 0, cx = 0, cy = 0;
    std::map<int, std::array<Long, 3>> faces;  // other particle: length, centroid relative to the generator
};

// Cell i of the particles by clipping a big square in long double with
// every bisector of every image of every particle. Used by: check_cells.
Cell brute_cell(int i, const Points& p, const Box& box) {
    struct Edge {
        Long x, y;
        int other;
    };
    const Long lx = box.side[0], ly = box.side[1], extent = 2 * (lx + ly);
    std::vector<Edge> poly = {{-extent, -extent, -1}, {extent, -extent, -1}, {extent, extent, -1}, {-extent, extent, -1}};
    // Nearest neighbours first keeps the polygon small: sort by distance.
    std::vector<std::array<Long, 4>> others;
    for (std::size_t j = 0; j < p.size(); ++j)
        for (int sx = -1; sx <= 1; ++sx)
            for (int sy = -1; sy <= 1; ++sy) {
                if (static_cast<int>(j) == i && sx == 0 && sy == 0) continue;
                const Long dx = Long(p[j][0]) - p[i][0] + sx * lx, dy = Long(p[j][1]) - p[i][1] + sy * ly;
                others.push_back({dx * dx + dy * dy, dx, dy, Long(j)});
            }
    std::sort(others.begin(), others.end());
    Long reach = 0;
    for (const auto& o : others) {
        if (!poly.empty() && reach > 0 && o[0] > 4 * reach * (1 + 1e-9L)) break;
        const Long dx = o[1], dy = o[2], half = o[0] / 2;
        std::vector<Edge> kept;
        const int count = static_cast<int>(poly.size());
        for (int m = 0; m < count; ++m) {
            const Edge& a = poly[m];
            const Edge& b = poly[(m + 1) % count];
            const Long sa = a.x * dx + a.y * dy - half, sb = b.x * dx + b.y * dy - half;
            if (sa <= 0) kept.push_back(a);
            if ((sa <= 0) != (sb <= 0)) {
                const Long t = sa / (sa - sb);
                kept.push_back({a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), sa <= 0 ? static_cast<int>(o[3]) : a.other});
            }
        }
        poly = kept;
        reach = 0;
        for (const Edge& e : poly) reach = std::max(reach, e.x * e.x + e.y * e.y);
    }
    Cell cell;
    const int count = static_cast<int>(poly.size());
    for (int m = 0; m < count; ++m) {
        const Edge& a = poly[m];
        const Edge& b = poly[(m + 1) % count];
        const Long twice = a.x * b.y - a.y * b.x, length = std::hypot(b.x - a.x, b.y - a.y);
        cell.volume += twice / 2;
        cell.cx += twice * (a.x + b.x) / 6;
        cell.cy += twice * (a.y + b.y) / 6;
        if (length > 0) {
            auto& f = cell.faces[a.other];
            if (length > f[0]) f = {length, (a.x + b.x) / 2, (a.y + b.y) / 2};
        }
    }
    cell.cx /= cell.volume;
    cell.cy /= cell.volume;
    return cell;
}

// Oracle (a): the triangulation over all particles (exact integers).
void check_triangles(Check& check, const Mesh& mesh, const Box& box, const std::string& name) {
    const auto corners = host(mesh.triangles().corners), links = host(mesh.triangles().links);
    const auto point = host(mesh.points().view);
    const int n = mesh.particles(), triangles = mesh.triangles().count();
    const ep::i64 px = box.period[0], py = box.period[1];
    auto off = [&](int from, int to) {
        return ep::Offset2{ep::delta(point(from, 0), point(to, 0), px), ep::delta(point(from, 1), point(to, 1), py)};
    };
    long long negative = 0, unpaired = 0, wrong_links = 0;
    i128 twice = 0;
    std::map<std::pair<int, int>, int> edges;
    long long inside = 0;
    for (int t = 0; t < triangles; ++t) {
        const int v[3] = {corners(t, 0), corners(t, 1), corners(t, 2)};
        const ep::Offset2 a = off(v[0], v[1]), b = off(v[0], v[2]);
        const ep::i64 o = ep::cross(a, b);
        negative += o <= 0;
        twice += o;
        wrong_links += v[0] > v[1] || v[0] > v[2];
        for (int k = 0; k < 3; ++k) {
            ++edges[{v[(k + 1) % 3], v[(k + 2) % 3]}];
            const int u = links(t, k);
            int back = 0;
            for (int j = 0; j < 3; ++j) back += links(u, j) == t && links(u, 3 + j) == links(t, 3 + k);
            wrong_links += back != 1;
        }
        // No particle strictly inside the circle, nearest image of the corners' frame.
        const Long la = Long(a[0]) * a[0] + Long(a[1]) * a[1], lb = Long(b[0]) * b[0] + Long(b[1]) * b[1];
        const Long ccx = (la * b[1] - lb * a[1]) / (2 * Long(o)), ccy = (lb * a[0] - la * b[0]) / (2 * Long(o));
        for (int e = 0; e < n; ++e) {
            if (e == v[0] || e == v[1] || e == v[2]) continue;
            ep::Offset2 d = off(v[0], e);
            const Long ex = d[0] - ccx, ey = d[1] - ccy;
            const Long r2 = ccx * ccx + ccy * ccy;
            if (ex * ex + ey * ey > r2 * (1 + 1e-6L)) continue;  // far from the circle: exact only near it
            const ep::Offset2 q[4] = {{0, 0}, a, b, d};
            const i128 det = textbook_in_circle(q);
            inside += det > 0;
        }
    }
    for (const auto& [edge, count] : edges) unpaired += count != 1 || edges.count({edge.second, edge.first}) != 1;
    check.expect(triangles == 2 * n && mesh.faces() == 3 * n, name + ": 2n triangles, 3n faces");
    check.expect(negative == 0, name + ": every triangle positive");
    check.expect(unpaired == 0 && wrong_links == 0, name + ": every edge shared by two triangles, links mutual");
    check.expect(twice == i128(2) * px * py, name + ": areas sum to the box exactly");
    check.expect(inside == 0, name + ": no particle inside a circumcircle");
}

// Oracles (b), (c): against brute-force cells.
void check_cells(Check& check, const Mesh& mesh, const Points& p, const Box& box, const std::string& name) {
    const auto volume = host(mesh.volume());
    const auto centroid = host(mesh.centroid());
    const auto start = host(mesh.cell_face_start()), list = host(mesh.cell_face());
    const auto owner = host(mesh.owner()), neighbour = host(mesh.neighbour());
    const auto area = host(mesh.area()), fc = host(mesh.face_centroid());
    const auto positions = host(mesh.positions());
    const int n = mesh.particles();
    const double scale = std::sqrt(box.volume() / n);
    double worst_volume = 0, worst_centroid = 0, worst_face = 0, sum = 0, closure = 0;
    long long missing = 0, disorder = 0, moved = 0;
    for (int i = 0; i < n; ++i) {
        const Cell cell = brute_cell(i, p, box);
        worst_volume = std::max(worst_volume, double(std::fabs(cell.volume - volume(i)) / cell.volume));
        worst_centroid = std::max({worst_centroid, double(std::fabs(cell.cx - centroid(i, 0)) / scale),
                                   double(std::fabs(cell.cy - centroid(i, 1)) / scale)});
        sum += volume(i);
        moved += positions(i, 0) != p[i][0] || positions(i, 1) != p[i][1];
        int last = -1;
        double ax = 0, ay = 0;
        std::map<int, int> at;
        for (int k = start(i); k < start(i + 1); ++k) {
            const int f = list(k), o = owner(f), m = neighbour(f), other = o == i ? m : o;
            disorder += other <= last;
            last = other;
            at[other] = f;
            const double sign = o == i ? 1.0 : -1.0;
            ax += sign * area(f, 0);
            ay += sign * area(f, 1);
        }
        closure = std::max(closure, std::hypot(ax, ay) / scale);
        const double tiny = 1e-9 * scale;
        for (const auto& [other, face] : cell.faces) {
            if (double(face[0]) < tiny) continue;
            const auto it = at.find(other);
            if (it == at.end()) {
                ++missing;
                continue;
            }
            const int f = it->second;
            const double length = std::hypot(area(f, 0), area(f, 1));
            worst_face = std::max(worst_face, double(std::fabs(face[0] - length)) / scale);
            const int o = owner(f);
            double cx = fc(f, 0), cy = fc(f, 1);
            if (o != i) {
                cx += ep::delta(ep::snap(p[i][0], box, 0), ep::snap(p[o][0], box, 0), box.period[0]) * box.step;
                cy += ep::delta(ep::snap(p[i][1], box, 1), ep::snap(p[o][1], box, 1), box.period[1]) * box.step;
            }
            worst_face = std::max({worst_face, double(std::fabs(face[1] - cx)) / scale, double(std::fabs(face[2] - cy)) / scale});
        }
    }
    check.expect(moved == 0, name + ": positions are the snapped input");
    check.expect(worst_volume < 1e-9, name + ": volumes match brute force, worst " + sci(worst_volume));
    check.expect(worst_centroid < 1e-9, name + ": centroids match, worst " + sci(worst_centroid));
    check.expect(missing == 0 && worst_face < 1e-9, name + ": faces of nonzero length match, worst " + sci(worst_face));
    check.expect(disorder == 0, name + ": lists in increasing id of the other particle");
    check.expect(std::fabs(sum / box.volume() - 1) < 1e-12, name + ": volumes sum to the box");
    check.expect(closure < 1e-10, name + ": cells closed, worst " + sci(closure));
}

// Everything a run reads, bit for bit: faces by (owner, neighbour). Used by: test_moving (fresh build comparison).
struct Bits {
    std::vector<std::uint64_t> values;
};
// Appends the bit pattern of v. Used by: bits_of.
void add(std::vector<std::uint64_t>& out, double v) {
    std::uint64_t u;
    std::memcpy(&u, &v, 8);
    out.push_back(u);
}
// Reads the mesh into Bits. Used by: test_moving.
Bits bits_of(const Mesh& mesh) {
    Bits b;
    const auto volume = host(mesh.volume());
    const auto centroid = host(mesh.centroid());
    const auto start = host(mesh.cell_face_start()), list = host(mesh.cell_face());
    const auto owner = host(mesh.owner()), neighbour = host(mesh.neighbour());
    const auto area = host(mesh.area()), fc = host(mesh.face_centroid());
    for (int i = 0; i < mesh.particles(); ++i) {
        add(b.values, volume(i));
        add(b.values, centroid(i, 0));
        add(b.values, centroid(i, 1));
        for (int k = start(i); k < start(i + 1); ++k) {
            const int f = list(k);
            if (owner(f) != i) continue;
            b.values.push_back(static_cast<std::uint64_t>(neighbour(f)));
            add(b.values, area(f, 0));
            add(b.values, area(f, 1));
            add(b.values, fc(f, 0));
            add(b.values, fc(f, 1));
        }
    }
    return b;
}

// Large: deep levels of the cell list (10 bits per axis in Morton numbers).
void test_large(Check& check) {
    const Box box(1.0);
    const int n = large_n;
    for (const char* kind : {"random", "lattice"}) {
        Mesh mesh(Exec(), box);
        const auto report = mesh.update(device(make_points(kind, n, box)));
        mesh.update_cells();
        const auto volume = host(mesh.volume());
        double sum = 0;
        for (int i = 0; i < mesh.particles(); ++i) sum += volume(i);
        check.expect(report.flips == 0 && std::abs(sum - 1) < 1e-11, std::string(kind) + " large: exact start, volumes sum to the box");
    }
}

// Whether f() throws E with `text` in its message.
template <class E, class F>
bool refused(F&& f, const std::string& text) {
    try {
        f();
    } catch (const E& e) {
        return std::string(e.what()).find(text) != std::string::npos;
    }
    return false;
}

// The same refusals and the same safety as Mesh<3>.
void test_refusals(Check& check) {
    const Box box(1.0);
    const Points good = make_points("jittered", 2500, box);
    {
        Points p = good;
        p[1700] = p[300];
        Mesh mesh(Exec(), box);
        check.expect(refused<std::runtime_error>([&] { mesh.update(device(p)); }, "particles 300 and 1700"), "coincident at the start: refused, the pair by name");
        check.expect(mesh.particles() == 0 && mesh.faces() == 0, "a refused start leaves no mesh");
        check.expect(refused<std::logic_error>([&] { mesh.update_cells(); }, "no mesh"), "update_cells() after a refusal says so");
        const auto report = mesh.update(device(good));
        mesh.update_cells();
        check.expect(report.rebuilt && mesh.faces() == 3 * int(good.size()), "after a refusal the next update builds");
        check_triangles(check, mesh, box, "after a refusal");
    }
    for (const auto fallback : {voronoi::Fallback::cheapest, voronoi::Fallback::host, voronoi::Fallback::rebuild}) {
        const std::string name = fallback == voronoi::Fallback::host ? "host" : fallback == voronoi::Fallback::rebuild ? "rebuild" : "cheapest";
        for (const int gap : {7, 1100}) {
            Points p = on_grid(good, box);
            Mesh mesh(Exec(), box);
            mesh.set_fallback(fallback);
            mesh.update(device(p));
            mesh.update_cells();
            p[200 + gap] = p[200];
            const std::string what = name + " gap " + std::to_string(gap);
            check.expect(refused<std::runtime_error>([&] { mesh.update(device(p)); }, "one position of the snapping grid"), what + ": a pair landing on one position is refused");
            check.expect(mesh.particles() == 0, what + ": a refused step leaves no mesh");
            p[200 + gap][0] = std::fmod(p[200 + gap][0] + 0.013, 1.0);
            mesh.update(device(p));
            mesh.update_cells();
            check_triangles(check, mesh, box, what + " after the refusal");
        }
    }
    {
        Mesh mesh(Exec(), box);
        const X none("none", 0);
        check.expect(refused<std::invalid_argument>([&] { mesh.update(none); }, "no particles"), "no particles: refused");
        mesh.update(device(good));
        mesh.update_cells();
        const int faces = mesh.faces();
        check.expect(refused<std::invalid_argument>([&] { mesh.update(none); }, "no particles") && mesh.faces() == faces && mesh.cells_current(),
                     "no particles leave a mesh as it was");
    }
    {   // Another n after a build, then steps: the lists start over.
        Mesh mesh(Exec(), box);
        mesh.update(device(good));
        mesh.update_cells();
        Points p = make_points("jittered", 1600, box);
        mesh.update(device(p));
        mesh.update_cells();
        for (auto& q : p) q[0] = std::fmod(q[0] + 2e-3, 1.0);
        mesh.update(device(p));
        mesh.update_cells();
        check_triangles(check, mesh, box, "another n, then a step");
        check_cells(check, mesh, on_grid(p, box), box, "another n, then a step");
    }
    // x -> 2 x: every triangle stays positive and locally Delaunay, and the mesh would cover the torus four times; the
    // areas tell.
    {
        Points p = on_grid(make_points("hexagonal", 2500, box), box);
        Mesh mesh(Exec(), box);
        mesh.update(device(p));
        mesh.update_cells();
        for (auto& q : p) {
            q[0] = std::fmod(2 * q[0], 1.0);
            q[1] = std::fmod(2 * q[1], 1.0);
        }
        bool done = true;
        voronoi::Report report;
        try {
            report = mesh.update(device(p));
        } catch (const std::runtime_error&) {
            done = false;
        }
        if (done) {
            mesh.update_cells();
            check.expect(report.rebuilt, "a mapping of degree 4 is rebuilt (overlapped " + std::to_string(report.overlapped) + ")");
            check_triangles(check, mesh, box, "degree 4");
        } else {
            check.expect(mesh.particles() == 0, "a mapping of degree 4 refused: no mesh");
        }
    }
    {   // Kicks of order the box, random: exact on return, or refused.
        std::mt19937_64 rng(5);
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        for (double kick : {0.1, 0.5}) {
            Points p = on_grid(good, box);
            Mesh mesh(Exec(), box);
            mesh.update(device(p));
            mesh.update_cells();
            for (auto& q : p) {
                q[0] = std::fmod(q[0] + kick * unit(rng), 1.0);
                q[1] = std::fmod(q[1] + kick * unit(rng), 1.0);
            }
            mesh.update(device(p));
            mesh.update_cells();
            const std::string name = "kicks of " + std::to_string(kick) + " of the box";
            check_triangles(check, mesh, box, name);
            check_cells(check, mesh, on_grid(p, box), box, name);
        }
    }
}

// clip(): the rings of a few cells are the mesh's neighbours in each list.
void test_clip(Check& check) {
    const Box box(1.0);
    const Points p = make_points("random", 2000, box);
    const X x = device(p);
    Mesh mesh(Exec(), box);
    mesh.update(x);
    mesh.update_cells();
    Kokkos::View<int*, Memory> cells("cells", 200);
    auto h = Kokkos::create_mirror_view(cells);
    for (int i = 0; i < 200; ++i) h(i) = 7 * i;
    Kokkos::deep_copy(cells, h);
    const auto rings = mesh.clip(x, cells);
    const auto degree = host(rings.degree), start = host(rings.start), ring = host(rings.ring), status = host(rings.status);
    const auto first = host(mesh.cell_face_start()), list = host(mesh.cell_face());
    const auto owner = host(mesh.owner()), neighbour = host(mesh.neighbour());
    long long different = 0;
    for (int c = 0; c < 200; ++c) {
        const int i = h(c);
        std::vector<int> a, b;
        for (int m = 0; m < degree(i); ++m) a.push_back(ring(start(i) + m));
        for (int k = first(i); k < first(i + 1); ++k) b.push_back(owner(list(k)) == i ? neighbour(list(k)) : owner(list(k)));
        std::sort(a.begin(), a.end());
        different += a != b || status(i) != polygon_clip::ok;
    }
    check.expect(different == 0, "clip: the rings of listed cells are their Delaunay neighbours");
}

// Static builds in four box shapes and six point kinds: no repair needed, triangulation and cells right.
void test_static(Check& check) {
    const std::pair<const char*, Box> boxes[] = {{"1x1", Box(1.0)}, {"2x1", Box(2.0, 1.0)}, {"3x1", Box(3.0, 1.0)}, {"1x1/16", Box(1.0, 1.0 / 16)}};
    for (const char* kind : {"random", "jittered", "clustered", "lattice", "hexagonal", "wave"})
        for (const auto& [label, box] : boxes) {
            const int n = label == std::string("1x1/16") ? (kind == std::string("clustered") ? 4000 : 1600) : 800;  // a slab: no cell reaches 3/16 of its width
            const std::string name = std::string(kind) + " " + label;
            const Points p = make_points(kind, n, box);
            Mesh mesh(Exec(), box);
            const auto report = mesh.update(device(p));
            mesh.update_cells();
            check.expect(report.rebuilt && report.flips == 0, name + ": the clipping needs no repair");
            check_triangles(check, mesh, box, name);
            check_cells(check, mesh, p, box, name);
        }
}

// Moving points in a 2 x 1 box with the given fallback: oracle checks and bit-identity with fresh builds.
void test_moving(Check& check, voronoi::Fallback fallback) {
    const Box box(2.0, 1.0);
    std::mt19937_64 rng(11);
    std::normal_distribution<double> normal;
    for (const char* kind : {"random", "jittered", "clustered", "lattice", "hexagonal"})
        for (double amount : {0.03, 0.3, 1.0}) {
            const std::string name = std::string(kind) + " moved " + std::to_string(amount) + (fallback == voronoi::Fallback::host ? " host" : " rebuild");
            Points p = make_points(kind, 4000, box);
            const double spacing = std::sqrt(box.volume() / p.size());
            Mesh mesh(Exec(), box);
            mesh.set_fallback(fallback);
            mesh.update(device(p));
            mesh.update_cells();
            long long flips = 0, stuck = 0, patches = 0, rebuilt = 0;
            bool same = true;
            for (int step = 0; step < moving_steps; ++step) {
                for (std::size_t i = 0; i < p.size(); ++i) {
                    const double phase = 2 * M_PI * p[i][1];
                    p[i][0] += amount * spacing * (0.6 * std::sin(phase) + 0.4 * normal(rng));
                    p[i][1] += amount * spacing * (0.6 * std::cos(2 * phase) + 0.4 * normal(rng));
                }
                p = on_grid(p, box);
                const X x = device(p);
                const auto report = mesh.update(x);
                mesh.update_cells();
                flips += report.flips;
                stuck += report.stuck;
                patches += report.host_rebuilds;
                rebuilt += report.rebuilt;
                Mesh fresh(Exec(), box);
                fresh.update(x);
                fresh.update_cells();
                same = same && bits_of(mesh).values == bits_of(fresh).values;
            }
            check.expect(same, name + ": equal to a build from nothing, bit for bit (" + std::to_string(flips) + " flips, " +
                                   std::to_string(stuck) + " stuck, " + std::to_string(patches) + " patches, " +
                                   std::to_string(rebuilt) + " rebuilds)");
            check_triangles(check, mesh, box, name);
        }
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    std::cout << std::unitbuf;
    Check check;
    test_predicates(check);
    test_static(check);
    test_large(check);
    test_clip(check);
    test_refusals(check);
    test_moving(check, voronoi::Fallback::host);
    test_moving(check, voronoi::Fallback::rebuild);
    return check.failed() != 0;
}
