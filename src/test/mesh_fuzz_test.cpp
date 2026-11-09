// Random cases of voronoi::Mesh against the independent oracle: smoke (ctest), run (long, resumes, survives crashes), one, case, min.
// One seed is one whole case, ending exact, refused, valid, bits, rough, WRONG, CRASH or HANG; usage and classes are in TESTING.md.

#include "check.hpp"
#include "mesh_check.hpp"
#include "mesh_fuzz.hpp"

#include <fcntl.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cinttypes>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <set>

// NOLINTNEXTLINE(bugprone-reserved-identifier): gcov's own name, defined in a coverage build
extern "C" void __gcov_dump() __attribute__((weak));

namespace {

using namespace mesh_fuzz;
using support::Exec;
using support::Memory;
using support::Positions;

enum Class { exact, refused, valid, bits, rough, wrong, crash, hang, classes };
const char* const class_name[classes] = {"exact", "refused", "valid", "bits", "rough", "WRONG", "CRASH", "HANG"};

struct Outcome {
    int kind = exact;
    std::string signature = "exact", detail;
    int op = -1;
    long long calls = 0, exact_calls = 0, refusals = 0, flips = 0, stuck = 0, host = 0, rebuilt = 0;
    std::uint64_t hash = 14695981039346656037ULL;  // over tetrahedra, faces and bits of what the calls gave
    void hit(int k, const std::string& sig, const std::string& text, int at) {
        if (k <= kind) return;
        kind = k;
        signature = std::string(class_name[k]) + ":" + sig;
        detail = text;
        op = at;
    }
};

// Long doubles of the volume and centroid checks: errors in units of the
// cell's own size, above what position rounding gives (mesh_check::Result).
constexpr double centroid_floor = 1e-6;

void mix(std::uint64_t& h, const void* data, std::size_t size) {
    const auto* b = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) h = (h ^ b[i]) * 1099511628211ULL;
}

// Positions of a call as a view for the mesh's memory.
Positions device(const Points& x) { return support::to_device(x); }

std::string category(const std::string& what) {
    auto has = [&](const char* s) { return what.find(s) != std::string::npos; };
    if (has("no particles")) return "no_particles";
    if (has("one position of the snapping grid")) return "coincident";
    if (has("reach too far")) return "reach";
    if (has("need more than")) return "overflow";
    if (has("not exact")) return "not_exact";
    if (has("too elongated")) return "elongated";
    if (has("Allocation") || has("bad_alloc") || has("allocation")) return "memory";
    return "unknown";
}

// The snap of the mesh against long double arithmetic of its own.
template <class M>
void check_snap(const M& mesh, const Points& x, Outcome& out, int at) {
    const auto p = support::to_host(mesh.positions());
    const auto& box = mesh.sides();
    long long bad = 0;
    for (std::size_t i = 0; i < x.size(); ++i)
        for (int a = 0; a < 3; ++a) {
            if (!std::isfinite(x[i][a]) || std::abs(x[i][a]) > 1e15 * box.side[a]) continue;
            const long double side = (long double)(box.period[a]) * box.step;
            long double y = (long double)x[i][a];
            y -= side * floorl(y / side);
            const long double k = y / box.step;
            const long double got = (long double)p(i, a) / box.step;
            long double d = fabsl(k - got - 0.5L);  // the grid step that holds x: within half of a step of it
            d = std::min(d, fabsl(d - (long double)box.period[a]));
            bad += d > 1.5L;
        }
    if (bad > 0) out.hit(wrong, "snap", std::to_string(bad) + " coordinates away from their grid step", at);
}

// Face data no check of mesh_check reads: order of each pair, faces against
// the separation, duplicates of a pair, each centroid on its bisector, float rows.
template <class M>
void check_faces(const M& mesh, const voronoi::Report& report, Outcome& out, int at) {
    const auto owner = support::to_host(mesh.owner());
    const auto neighbour = support::to_host(mesh.neighbour());
    const auto area = support::face_areas(mesh);
    const auto centroid = support::to_host(mesh.face_centroid());
    const auto g = mesh_check::grid_of(mesh);
    const int faces = mesh.faces(), n = mesh.particles();
    if (report.faces != faces)
        out.hit(wrong, "report_faces", std::to_string(report.faces) + " of " + std::to_string(faces), at);
    if (static_cast<int>(owner.extent(0)) != faces) out.hit(wrong, "owner_extent", "", at);
    std::vector<std::pair<int, int>> pairs;
    pairs.reserve(faces);
    long long order = 0, direction = 0, bisector = 0, range = 0;
    double worst = 0;
    for (int f = 0; f < faces; ++f) {
        const int a = owner(f), b = neighbour(f);
        if (a < 0 || b >= n || a >= n || b < 0) {
            ++range;
            continue;
        }
        order += a >= b;
        pairs.push_back({a, b});
        const auto d = oracle::difference(g, a, b);
        double along = 0, size = 0, offset = 0, d2 = 0;
        for (int k = 0; k < 3; ++k) {
            const double dk = double(d[k]) * g.step;
            along += area(f, k) * dk;
            size += area(f, k) * area(f, k);
            offset += centroid(f, k) * dk;
            d2 += dk * dk;
        }
        size = std::sqrt(size);
        if (size == 0) continue;  // a zero-area face has no direction
        if (along < -1e-9 * size * std::sqrt(d2)) ++direction;
        // On the bisector: c.d = |d|^2 / 2. Only faces of real size: a rounding-level
        // polygon has a centroid of rounding quality.
        const double spacing = std::sqrt(d2);
        if (size > 1e-12 * spacing * spacing) {
            const double e = std::abs(offset / spacing - spacing / 2) / g.step;  // grid steps
            worst = std::max(worst, e);
            bisector += e > 1;
        }
    }
    std::sort(pairs.begin(), pairs.end());
    const long long repeated = pairs.end() - std::unique(pairs.begin(), pairs.end());
    if (range > 0) out.hit(wrong, "face_range", std::to_string(range), at);
    if (order > 0) out.hit(wrong, "face_order", std::to_string(order) + " faces with owner >= neighbour", at);
    if (repeated > 0) out.hit(wrong, "face_repeated", std::to_string(repeated) + " pairs with two faces", at);
    if (direction > 0) out.hit(wrong, "face_direction", std::to_string(direction) + " faces pointing back", at);
    if (bisector > 0)
        out.hit(worst > 1e3 ? wrong : rough, "face_bisector",
                std::to_string(bisector) + " faces, worst " + std::to_string(worst) + " grid steps", at);
    if constexpr (M::float_rows) {
        auto rows = mesh.face_rows();
        const auto r = support::to_host(rows);
        long long differ = 0;
        for (int f = 0; f < faces; ++f)
            for (int k = 0; k < 3; ++k)
                differ +=
                    r(f, k) != static_cast<float>(area(f, k)) || r(f, 3 + k) != static_cast<float>(centroid(f, k));
        if (differ > 0) out.hit(wrong, "float_rows", std::to_string(differ) + " values", at);
    }
}

// Centroids of sampled cells' faces and of the cells against brute clipping.
template <class M>
void check_cell_geometry(const M& mesh, const oracle::Grid& g, int sample, Outcome& out, int at) {
    const int n = mesh.particles();
    const auto owner = support::to_host(mesh.owner());
    const auto neighbour = support::to_host(mesh.neighbour());
    const auto fc = support::to_host(mesh.face_centroid());
    const auto start = support::to_host(mesh.cell_face_start());
    const auto list = support::to_host(mesh.cell_face());
    const auto centroid = support::to_host(mesh.centroid());
    const int stride = std::max(1, n / std::max(sample, 1));
    double worst_cell = 0, worst_face = 0;
    for (int i = 0; i < n; i += stride) {
        const oracle::Cell cell = oracle::voronoi_cell(g, i);
        const long double size = cbrtl(cell.volume);
        for (int a = 0; a < 3; ++a) {
            const long double brute = cell.moment[a] / cell.volume;
            worst_cell = std::max(worst_cell, double(fabsl((long double)centroid(i, a) / g.step - brute) / size));
        }
        // Faces of one image of the neighbour, of real area, sit where brute force has them.
        for (int k = start(i); k < start(i + 1); ++k) {
            const int f = list(k), j = owner(f) == i ? neighbour(f) : owner(f);
            const auto images = cell.images.find(j);
            if (images == cell.images.end() || images->second != 1) continue;
            for (const auto& polygon : cell.faces) {
                if (polygon.neighbour != j) continue;
                const oracle::Vec a = oracle::area_vector(polygon);
                const long double area = sqrtl(oracle::dot(a, a));
                if (area < 1e-6L * size * size) continue;
                oracle::Vec m{0, 0, 0};
                for (std::size_t c = 1; c + 1 < polygon.corners.size(); ++c) {
                    const oracle::Vec t = oracle::cross(oracle::minus(polygon.corners[c], polygon.corners[0]),
                                                        oracle::minus(polygon.corners[c + 1], polygon.corners[0]));
                    const long double w = sqrtl(oracle::dot(t, t));
                    for (int d = 0; d < 3; ++d)
                        m[d] += w * (polygon.corners[0][d] + polygon.corners[c][d] + polygon.corners[c + 1][d]) / 3;
                }
                long double total = 0;
                for (std::size_t c = 1; c + 1 < polygon.corners.size(); ++c) {
                    const oracle::Vec t = oracle::cross(oracle::minus(polygon.corners[c], polygon.corners[0]),
                                                        oracle::minus(polygon.corners[c + 1], polygon.corners[0]));
                    total += sqrtl(oracle::dot(t, t));
                }
                // Stored relative to the owner, brute force relative to i.
                const oracle::Offset d = oracle::difference(g, i, j);
                for (int k2 = 0; k2 < 3; ++k2) {
                    const long double brute = m[k2] / total - (owner(f) == i ? 0.0L : (long double)d[k2]);
                    worst_face = std::max(worst_face, double(fabsl((long double)fc(f, k2) / g.step - brute) / size));
                }
            }
        }
    }
    if (worst_cell > centroid_floor)
        out.hit(worst_cell > 1e-3 ? wrong : rough, "cell_centroid", std::to_string(worst_cell) + " cell sizes", at);
    if (worst_face > centroid_floor)
        out.hit(worst_face > 1e-3 ? wrong : rough, "face_centroid", std::to_string(worst_face) + " cell sizes", at);
}

// The per-cell lists: bounds, each face in both its cells, order by the other id.
template <class M>
void check_lists(const M& mesh, Outcome& out, int at) {
    const int n = mesh.particles(), faces = mesh.faces();
    const auto start = support::to_host(mesh.cell_face_start());
    const auto list = support::to_host(mesh.cell_face());
    const auto owner = support::to_host(mesh.owner());
    const auto neighbour = support::to_host(mesh.neighbour());
    if (start(0) != 0 || start(n) != 2 * faces) {
        out.hit(wrong, "list_bounds", "start(n) " + std::to_string(start(n)) + " faces " + std::to_string(faces), at);
        return;
    }
    std::vector<int> seen(faces, 0);
    for (int i = 0; i < n; ++i) {
        if (start(i) > start(i + 1)) {
            out.hit(wrong, "list_bounds", "not monotone at " + std::to_string(i), at);
            return;
        }
        int last = -1;
        for (int k = start(i); k < start(i + 1); ++k) {
            const int f = list(k);
            if (f < 0 || f >= faces) {
                out.hit(wrong, "list_face", "face " + std::to_string(f), at);
                return;
            }
            ++seen[f];
            if (owner(f) != i && neighbour(f) != i) {
                out.hit(wrong, "list_face", "cell " + std::to_string(i) + " lists a face of others", at);
                return;
            }
            const int other = owner(f) == i ? neighbour(f) : owner(f);
            if (other <= last) {
                out.hit(wrong, "list_order", "cell " + std::to_string(i), at);
                return;
            }
            last = other;
        }
    }
    for (int f = 0; f < faces; ++f)
        if (seen[f] != 2) {
            out.hit(wrong, "list_twice", "face " + std::to_string(f) + " listed " + std::to_string(seen[f]) + " times",
                    at);
            return;
        }
}

// Least-squares weights give back the gradient of a linear field: the fit
// is exact there, with no intercept.
template <class M>
void check_weights(const M& mesh, const oracle::Grid& g, Outcome& out, int at) {
    if constexpr (M::with_weights) {
        const int n = mesh.particles();
        const auto start = support::to_host(mesh.cell_face_start());
        const auto list = support::to_host(mesh.cell_face());
        const auto owner = support::to_host(mesh.owner());
        const auto neighbour = support::to_host(mesh.neighbour());
        const auto centroid = support::to_host(mesh.centroid());
        const auto weights = support::to_host(mesh.gradient_weights());
        const double tolerance = std::is_same_v<typename M::WeightReal, float> ? 5e-2 : 1e-6;
        const double gradient[3] = {1, -2, 3};
        long long not_finite = 0, off = 0;
        double worst = 0;
        for (int i = 0; i < n; ++i) {
            double sum[3] = {0, 0, 0};
            if (start(i + 1) - start(i) < 4) continue;  // no fit
            for (int k = start(i); k < start(i + 1); ++k) {
                const int f = list(k), j = owner(f) == i ? neighbour(f) : owner(f);
                const oracle::Offset d = oracle::difference(g, i, j);
                double du = 0;
                for (int a = 0; a < 3; ++a)
                    du += gradient[a] * (double(d[a]) * g.step + centroid(j, a) - centroid(i, a));
                for (int a = 0; a < 3; ++a) {
                    not_finite += !std::isfinite(weights(k, a));
                    sum[a] += static_cast<double>(weights(k, a)) * du;
                }
            }
            double e = 0;
            for (int a = 0; a < 3; ++a) e = std::max(e, std::abs(sum[a] - gradient[a]));
            worst = std::max(worst, e);
            off += e > tolerance;
        }
        if (not_finite > 0) out.hit(wrong, "weights_finite", std::to_string(not_finite) + " weights", at);
        if (off > 0) out.hit(bits, "weights_fit", std::to_string(off) + " cells, worst " + std::to_string(worst), at);
    } else {
        (void)mesh;
        (void)g;
        (void)out;
        (void)at;
    }
}

// update_cells of a repaired mesh: the same bits as from a fresh build.
template <class M>
void check_cell_bits(const M& mesh, Outcome& out, int at) {
    const int n = mesh.particles();
    const auto volume = support::to_host(mesh.volume());
    const auto centroid = support::to_host(mesh.centroid());
    const auto g = mesh_check::grid_of(mesh);
    const auto& box = mesh.sides();
    Positions p("fresh", n);
    auto h = Kokkos::create_mirror_view(p);
    for (int i = 0; i < n; ++i)
        for (int a = 0; a < 3; ++a) h(i, a) = (double(g.p[i][a]) + 0.5) * box.step;
    Kokkos::deep_copy(p, h);
    voronoi::Mesh<3, Exec> fresh(Exec{}, box);
    try {
        fresh.update(p);
    } catch (const std::exception&) {
        return;  // mesh_check says what a refused fresh build means
    }
    fresh.update_cells();
    const auto fv = support::to_host(fresh.volume());
    const auto fc = support::to_host(fresh.centroid());
    long long differ = 0;
    double most = 0;
    for (int i = 0; i < n; ++i) {
        differ += fv(i) != volume(i);
        most = std::max(most, std::abs(fv(i) / volume(i) - 1));
        for (int a = 0; a < 3; ++a) differ += fc(i, a) != centroid(i, a);
    }
    if (differ > 0)
        out.hit(bits, "fresh_cells", std::to_string(differ) + " values, volume off by " + std::to_string(most), at);
}

// Tetrahedra as a sorted set and faces by pair, volumes and centroids: what a backend must agree on.
template <class M>
void hash_mesh(const M& mesh, Outcome& out) {
    auto t = mesh_check::tets_of(mesh);
    std::vector<std::array<int, 4>> q;
    for (int i = 0; i < t.count(); ++i)
        if (t.alive[i]) {
            q.push_back(t.vertex[i]);
            std::sort(q.back().begin(), q.back().end());
        }
    std::sort(q.begin(), q.end());
    if (!q.empty()) mix(out.hash, q.data(), q.size() * sizeof(q[0]));
}

// clip(): each listed cell against brute force.
template <class M>
void check_clip(M& mesh, const Op& o, const Box& box, Outcome& out, int at, bool verbose) {
    using Slots = typename M::template View<int*>;
    Slots list("clip_list", o.list.size());
    auto h = Kokkos::create_mirror_view(list);
    for (std::size_t i = 0; i < o.list.size(); ++i) h(i) = o.list[i];
    Kokkos::deep_copy(list, h);
    const auto& cells = mesh.clip(device(o.x), list);
    const auto status = support::to_host(cells.status);
    const auto volume = support::to_host(cells.volume);
    oracle::Grid g;
    g.step = box.step;
    for (int a = 0; a < 3; ++a) g.period[a] = box.period[a];
    g.p.resize(o.x.size());
    for (std::size_t i = 0; i < o.x.size(); ++i)
        for (int a = 0; a < 3; ++a) g.p[i][a] = exact_predicates::snap(o.x[i][a], box, a);
    {  // A doubled particle has no cell, and what clip gives it means nothing.
        auto sorted = g.p;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) return;
    }
    int checked = 0;
    for (const int i : o.list) {
        if (checked++ >= 8) break;
        if (status(i) != shell_voronoi::ok) continue;
        const oracle::Cell cell = oracle::voronoi_cell(g, i);
        const double brute = double(cell.volume * g.step * g.step * g.step);
        const double e = std::abs(volume(i) - brute) / brute;
        if (verbose) {
            char text[120];
            std::snprintf(text, sizeof text, "  clip cell %d status %d volume %.17g brute %.17g", i, status(i),
                          volume(i), brute);
            std::cout << text << '\n' << std::flush;
        }
        // Scaled by the cell's width in steps, with a floor for doubles summed over dozens of faces.
        if (e * std::cbrt(double(cell.volume)) > 1e-6 && e > 1e-13)
            out.hit(e > 1e-6 ? wrong : rough, "clip_volume", "cell " + std::to_string(i) + " off by " + sci(e), at);
    }
}

struct Limits {
    int every_call = 4096;   // n up to here is checked after every call
    int sample = 8;          // cells against brute force
    int reach_check = 1024;  // n up to here: a refusal for reach is measured against brute force
    bool verbose = false;
};

template <class M>
void run(const Script& s, Outcome& out, const Limits& limits) {
    const Box box(s.sides[0], s.sides[1], s.sides[2]);
    M mesh(Exec{}, box);
    bool timed = false, reach_measured = false;
    Points last;  // the positions the mesh is of
    auto say = [&](const std::string& text) {
        if (limits.verbose) std::cout << "  " << text << '\n' << std::flush;
    };
    int at = -1;
    for (const Op& o : s.ops) {
        ++at;
        ++out.calls;
        say(std::to_string(at) + ": " + op_name(o.kind) + " " + std::to_string(o.arg) +
            " x=" + std::to_string(o.x.size()));
        voronoi::Report report;
        bool had_mesh = mesh.faces() > 0;
        switch (o.kind) {
            case Op::fallback:
                mesh.set_fallback(static_cast<voronoi::Fallback>(o.arg));
                continue;
            case Op::rounds:
                mesh.set_repair_rounds(o.arg);
                continue;
            case Op::timed:
                timed = true;
                continue;
            case Op::scribble: {
                if (!had_mesh) continue;
                const auto scratch = mesh.scratch(static_cast<std::size_t>(o.arg));
                const char value = static_cast<char>(0xa5);
                Kokkos::deep_copy(Exec{}, scratch, value);
                Kokkos::fence();
                continue;
            }
            case Op::clip: {
                if (static_cast<int>(o.x.size()) != mesh.particles() && mesh.particles() > 0) continue;
                try {
                    check_clip(mesh, o, box, out, at, limits.verbose);
                } catch (const std::exception& e) {
                    say(std::string("clip: ") + e.what());
                }
                continue;
            }
            case Op::cells: {
                try {
                    mesh.update_cells();
                } catch (const std::logic_error& e) {
                    if (had_mesh) out.hit(wrong, "cells_refused", e.what(), at);
                    continue;
                }
                if (!had_mesh) {
                    out.hit(wrong, "cells_no_mesh", "update_cells without a mesh did not throw", at);
                    continue;
                }
                check_lists(mesh, out, at);
                if (out.kind >= wrong) return;
                const auto g = mesh_check::grid_of(mesh);
                if (mesh.particles() <= limits.every_call) {
                    check_cell_geometry(mesh, g, limits.sample, out, at);
                    check_weights(mesh, g, out, at);
                    check_cell_bits(mesh, out, at);
                }
                if (out.kind >= wrong) return;
                continue;
            }
            default:
                break;
        }
        // update or rebuild.
        voronoi::Timings timings;
        try {
            const Positions x = device(o.x);
            report = o.kind == Op::rebuild ? mesh.rebuild(x, timed ? &timings : nullptr)
                                           : mesh.update(x, {}, timed ? &timings : nullptr);
            timed = false;
        } catch (const std::exception& e) {
            timed = false;
            ++out.refusals;
            const std::string what = e.what(), c = category(what);
            say(std::string(op_name(o.kind)) + " refused: " + what);
            // The input really has the cause named?
            if (c == "coincident") {
                std::vector<std::array<int, 3>> grid;
                grid.reserve(o.x.size());
                for (const Point& q : o.x)
                    grid.push_back({exact_predicates::snap(q[0], box, 0), exact_predicates::snap(q[1], box, 1),
                                    exact_predicates::snap(q[2], box, 2)});
                std::sort(grid.begin(), grid.end());
                if (std::adjacent_find(grid.begin(), grid.end()) == grid.end())
                    out.hit(valid, "coincident_false", what, at);
                else
                    out.hit(refused, c, what, at);
            } else if (c == "reach") {
                // The cause claimed: a cell of radius 3/16 of the shortest side or more.
                // Brute force over all cells, once a case, on a mesh small enough.
                const int count = static_cast<int>(o.x.size());
                double largest = -1;
                if (!reach_measured && count <= limits.reach_check && !box.cubic()) {
                    reach_measured = true;
                    oracle::Grid g;
                    g.step = box.step;
                    for (int a = 0; a < 3; ++a) g.period[a] = box.period[a];
                    g.p.resize(count);
                    for (int i = 0; i < count; ++i)
                        for (int a = 0; a < 3; ++a) g.p[i][a] = exact_predicates::snap(o.x[i][a], box, a);
                    auto sorted = g.p;
                    std::sort(sorted.begin(), sorted.end());
                    if (std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end()) {
                        for (int i = 0; i < count; ++i)
                            largest = std::max(largest, double(oracle::voronoi_cell(g, i).reach * g.step));
                        largest /= box.shortest();
                        say("largest cell radius / shortest side: " + std::to_string(largest));
                    }
                }
                if (largest >= 0 && largest < 3.0 / 16 * 0.9)
                    out.hit(valid, "reach_false", "largest radius " + sci(largest) + " of the shortest side; " + what,
                            at);
                else
                    out.hit(refused, c, what, at);
            } else if (c == "elongated" || c == "no_particles" || c == "memory") {
                out.hit(refused, c, what, at);
            } else {
                out.hit(valid, c, what, at);
            }
            if (mesh.faces() != 0 || mesh.particles() != 0)
                out.hit(wrong, "after_refusal", "a refused update left a mesh", at);
            bool thrown = false;
            try {
                mesh.update_cells();
            } catch (const std::logic_error&) {
                thrown = true;
            }
            if (!thrown) out.hit(wrong, "after_refusal", "update_cells after a refusal did not throw", at);
            last.clear();
            if (out.refusals >= 3) return;  // the rest of the calls would say the same, at a build each
            continue;
        }
        out.flips += report.flips + report.host_flips;
        out.stuck += report.stuck;
        out.host += report.host_flips + report.host_rebuilds;
        out.rebuilt += report.rebuilt;
        last = o.x;
        hash_mesh(mesh, out);
        if (limits.verbose) {  // to find the first call where two builds differ
            char text[64];
            std::snprintf(text, sizeof text, "hash %d %016" PRIx64, at, out.hash);
            std::cout << "  " << text << '\n' << std::flush;
        }
        check_snap(mesh, o.x, out, at);
        check_faces(mesh, report, out, at);
        const int n = mesh.particles();
        if (n <= limits.every_call || at + 1 == static_cast<int>(s.ops.size())) {
            const auto r = mesh_check::check(mesh, limits.sample, !report.rebuilt);
            say(std::string(op_name(o.kind)) + " n=" + std::to_string(n) + " faces=" + std::to_string(mesh.faces()) +
                " flips=" + std::to_string(report.flips) + " stuck=" + std::to_string(report.stuck) +
                " host=" + std::to_string(report.host_flips + report.host_rebuilds) +
                " rebuilt=" + std::to_string(report.rebuilt) + (r.ok() ? " exact" : " NOT EXACT " + r.text()));
            if (!r.ok()) {
                if (limits.verbose) {  // for arbitration by exact arithmetic outside this program
                    const auto volume = support::to_host(mesh.volume());
                    const auto g = mesh_check::grid_of(mesh);
                    const int stride = std::max(1, n / limits.sample);
                    for (int i = 0; i < n; i += stride) {
                        char text[96];
                        std::snprintf(text, sizeof text, "cell %d volume %.17g brute %.17g", i, volume(i),
                                      double(oracle::voronoi_cell(g, i).volume * g.step * g.step * g.step));
                        say(text);
                    }
                }
                // What failed, in the order a reader would look.
                const auto& t = r.tets;
                const char* what = t.ambiguous                      ? "ambiguous"
                                   : t.flat                         ? "flat"
                                   : t.inverted                     ? "inverted"
                                   : t.unpaired                     ? "unpaired"
                                   : t.folded                       ? "folded"
                                   : t.wrong_adjacency              ? "adjacency"
                                   : t.unused                       ? "unused"
                                   : t.not_empty                    ? "not_empty"
                                   : t.oversized                    ? "oversized"
                                   : !t.box_volume                  ? "box_volume"
                                   : r.not_finite                   ? "not_finite"
                                   : r.fresh_differs > 0            ? "fresh_differs"
                                   : r.brute_missing                ? "brute_missing"
                                   : r.brute_extra                  ? "brute_extra"
                                   : r.negative                     ? "negative"
                                   : r.duplicates                   ? "duplicates"
                                   : r.closure >= 1e-11             ? "closure"
                                   : std::abs(r.total - 1) >= 1e-11 ? "total"
                                   : r.fresh_volume >= 1e-10        ? "fresh_volume"
                                                                    : "brute_volume";
                // Rounding-level geometry on the exact tetrahedra: not a wrong mesh.
                const bool small = t.ok() && r.not_finite == 0 && r.brute_missing == 0 && r.brute_extra == 0 &&
                                   r.negative == 0 && r.fresh_differs <= 0 && r.duplicates == 0 &&
                                   r.brute_volume < 1e-6 && r.brute_area < 1e-6 && r.closure < 1e-7 &&
                                   std::abs(r.total - 1) < 1e-7 && r.fresh_volume < 1e-6;
                out.hit(small ? rough : wrong, what, r.text(), at);
            }
            if (r.fresh_differs < 0) out.hit(valid, "fresh_refused", r.fresh_error, at);
        }
        if (out.kind >= wrong) return;
        ++out.exact_calls;
    }
}

void run_script(const Script& s, Outcome& out, const Limits& limits) {
    switch (s.variant) {
        case float_rows:
            run<voronoi::Mesh<3, Exec, float, void>>(s, out, limits);
            break;
        case float_weights:
            run<voronoi::Mesh<3, Exec, double, float>>(s, out, limits);
            break;
        case double_weights:
            run<voronoi::Mesh<3, Exec, double, double>>(s, out, limits);
            break;
        default:
            run<voronoi::Mesh<3, Exec>>(s, out, limits);
            break;
    }
}

// --------------------------------------------------------------- 1D and 2D

// Seeds of profile digits 5 (1D) and 6 (2D) have no Script: they reproduce
// from the seed and are not shrunk.
int dimension_of(std::uint64_t seed) {
    const std::uint64_t digit = seed / 1000000000000ULL;
    return digit == 5 ? 1 : digit == 6 ? 2 : 3;
}

void run_1d(std::uint64_t seed, Outcome& out, std::string& label, bool verbose) {
    using Mesh1 = voronoi::Mesh<1, Exec>;
    using X1 = Kokkos::View<double* [1], Memory>;
    Rng rng(seed);
    const bool periodic = rng.chance(0.6);
    const double side = rng.chance(0.5) ? 1.0 : rng.between(1, 10) * std::pow(10.0, rng.range(-9, 9));
    const int n = static_cast<int>(rng.log_between(3, 20000));
    label = std::string(periodic ? "periodic" : "walls") + " n=" + std::to_string(n);
    const sorted_mesh::Grid grid(side);
    std::vector<double> x(n);
    const int kind = rng.below(5);
    for (int i = 0; i < n; ++i) {
        double u = rng.unit();
        if (kind == 1) u = (i + 0.5) / n;                                               // a lattice
        if (kind == 2) u = 0.5 + 1e-6 * (rng.unit() - 0.5);                             // a clump
        if (kind == 3) u = rng.below(std::max(2, n / 4)) / double(std::max(2, n / 4));  // doubled points
        if (kind == 4 && i % 2) u = rng.chance(0.5) ? 0 : std::nextafter(1.0, 0.0);     // at the ends
        x[i] = std::min(u * side, std::nextafter(side, 0.0));
    }
    const auto ends = periodic ? voronoi::Ends::periodic : voronoi::Ends::walls;
    Mesh1 mesh(Exec{}, side, ends);
    auto to_view = [&](const std::vector<double>& v) {
        X1 d("x", v.size());
        auto h = Kokkos::create_mirror_view(d);
        for (std::size_t i = 0; i < v.size(); ++i) h(i, 0) = v[i];
        Kokkos::deep_copy(d, h);
        return d;
    };
    const int steps = rng.range(1, 8);
    for (int t = 0; t <= steps; ++t) {
        ++out.calls;
        mesh.update(to_view(x));
        mesh.update_cells();
        const long long P = sorted_mesh::Grid::period;
        std::vector<std::pair<long long, int>> order(n);
        for (int i = 0; i < n; ++i) {
            const long long k = static_cast<long long>(std::floor(x[i] / grid.step));
            order[i] = {periodic ? ((k % P) + P) % P : std::clamp(k, 0LL, P - 1), i};
        }
        std::sort(order.begin(), order.end());
        const auto volume = support::to_host(mesh.volume());
        long long differ = 0;
        for (int s = 0; s < n; ++s) {
            // Twice the cell, in grid steps, from the integers.
            long long left, right;
            if (periodic) {
                left = s > 0 ? order[s].first - order[s - 1].first : order[0].first + P - order[n - 1].first;
                right = s + 1 < n ? order[s + 1].first - order[s].first : order[0].first + P - order[n - 1].first;
            } else {
                left = s > 0 ? order[s].first - order[s - 1].first : 2 * order[0].first;
                right = s + 1 < n ? order[s + 1].first - order[s].first : 2 * (P - order[n - 1].first);
            }
            differ += volume(order[s].second) != static_cast<double>(left + right) * (0.5 * grid.step);
        }
        if (differ > 0) {
            double total = 0;
            for (int i = 0; i < n; ++i) total += volume(i);
            out.hit(wrong, "volume_1d",
                    std::to_string(differ) + " cells, step " + std::to_string(t) + ", volumes sum to " +
                        sci(total / side) + " of the line",
                    t);
            return;
        }
        Mesh1 fresh(Exec{}, side, ends);
        fresh.update(to_view(x));
        fresh.update_cells();
        const auto again = support::to_host(fresh.volume());
        for (int i = 0; i < n; ++i)
            if (again(i) != volume(i)) {
                out.hit(bits, "fresh_1d", "cell " + std::to_string(i), t);
                break;
            }
        ++out.exact_calls;
        const double d = rng.log_between(grid.step, side);
        const bool all = rng.chance(0.5);
        for (double& q : x) {
            if (all || rng.chance(0.1)) q += rng.between(-d, d);
            q = periodic ? q - side * std::floor(q / side) : std::clamp(q, 0.0, std::nextafter(side, 0.0));
            if (q >= side) q = 0;
        }
    }
    if (verbose) std::cout << label << '\n' << std::flush;
}

// The cell of particle i in integer grid steps by half planes, long double, 25 images.
long double area_2d(const std::vector<std::array<long long, 2>>& p, int i, const long long* period) {
    using V = std::array<long double, 2>;
    const long double h = 2.0L * (period[0] + period[1]);
    std::vector<V> poly{{-h, -h}, {h, -h}, {h, h}, {-h, h}};
    std::vector<std::array<long double, 3>> cuts;
    for (std::size_t j = 0; j < p.size(); ++j)
        for (int u = -2; u <= 2; ++u)
            for (int v = -2; v <= 2; ++v) {
                if (static_cast<int>(j) == i && u == 0 && v == 0) continue;
                const long double dx = (long double)(p[j][0] - p[i][0] + u * period[0]);
                const long double dy = (long double)(p[j][1] - p[i][1] + v * period[1]);
                cuts.push_back({dx * dx + dy * dy, dx, dy});
            }
    std::sort(cuts.begin(), cuts.end());
    long double reach = h * h * 4;
    for (const auto& c : cuts) {
        if (c[0] > 4 * reach) break;
        std::vector<V> kept;
        const std::size_t m = poly.size();
        for (std::size_t k = 0; k < m; ++k) {
            const V& a = poly[k];
            const V& b = poly[(k + 1) % m];
            const long double sa = a[0] * c[1] + a[1] * c[2] - c[0] / 2, sb = b[0] * c[1] + b[1] * c[2] - c[0] / 2;
            if (sa <= 0) kept.push_back(a);
            if ((sa < 0 && sb > 0) || (sa > 0 && sb < 0)) {
                const long double t = sa / (sa - sb);
                kept.push_back({a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])});
            }
        }
        poly.swap(kept);
        reach = 0;
        for (const V& q : poly) reach = std::max(reach, q[0] * q[0] + q[1] * q[1]);
    }
    long double twice = 0;
    for (std::size_t k = 0; k < poly.size(); ++k) {
        const V& a = poly[k];
        const V& b = poly[(k + 1) % poly.size()];
        twice += a[0] * b[1] - a[1] * b[0];
    }
    return twice / 2;
}

void run_2d(std::uint64_t seed, Outcome& out, std::string& label, bool verbose) {
    using Mesh2 = voronoi::Mesh<2, Exec>;
    using Box2 = exact_predicates::BoxN<2>;
    using X2 = Kokkos::View<double* [2], Memory>;
    Rng rng(seed);
    const double lx = rng.chance(0.5) ? 1.0 : rng.between(1, 10) * std::pow(10.0, rng.range(-9, 9));
    const double ratio = rng.chance(0.6) ? 1.0 : rng.chance(0.5) ? 0.5 : rng.between(0.2, 1);
    const Box2 box(lx, lx * ratio);
    const int n = static_cast<int>(rng.log_between(rng.chance(0.1) ? 4 : 200, 4000));
    const int kind = rng.below(7);
    label = "2d kind" + std::to_string(kind) + " n=" + std::to_string(n) + " ratio=" + sci(ratio);
    using P2 = std::array<double, 2>;
    std::vector<P2> x(n);
    const int rows = std::max(2, static_cast<int>(std::sqrt(double(n))));
    for (int i = 0; i < n; ++i) {
        double u = rng.unit(), v = rng.unit();
        switch (kind) {
            case 1:  // a lattice
                u = ((i % rows) + 0.5) / rows;
                v = (((i / rows) % rows) + 0.5) / rows;
                break;
            case 2:  // hexagonal
                u = frac(((i % rows) + ((i / rows) & 1) * 0.5) / rows);
                v = (((i / rows) % rows) + 0.5) / rows;
                break;
            case 3:  // a clump
                if (i % 2) {
                    u = 0.5 + 1e-5 * rng.normal();
                    v = 0.3 + 1e-5 * rng.normal();
                }
                break;
            case 4:  // a circle
                if (i % 3 == 0) {
                    const double a = i * 2 * M_PI / n;
                    u = 0.5 + 0.2 * std::cos(a);
                    v = 0.5 + 0.2 * std::sin(a);
                }
                break;
            case 5:  // on the edges
                if (i % 2) u = rng.chance(0.5) ? 0 : 1 - 1e-9;
                break;
            case 6:  // near pairs
                if (i > 0 && i % 4 == 0) {
                    u = x[i - 1][0] / box.side[0] + rng.range(-3, 3) * box.step / box.side[0];
                    v = x[i - 1][1] / box.side[1];
                }
                break;
            default:
                break;
        }
        x[i] = {frac(u) * box.side[0], frac(v) * box.side[1]};
        for (int a = 0; a < 2; ++a) x[i][a] = std::min(x[i][a], std::nextafter(box.side[a], 0.0));
    }
    Mesh2 mesh(Exec{}, box);
    if (rng.chance(0.2)) mesh.set_fallback(rng.chance(0.5) ? voronoi::Fallback::rebuild : voronoi::Fallback::cheapest);
    auto to_view = [&](const std::vector<P2>& v) {
        X2 d("x", v.size());
        auto h = Kokkos::create_mirror_view(d);
        for (std::size_t i = 0; i < v.size(); ++i)
            for (int a = 0; a < 2; ++a) h(i, a) = v[i][a];
        Kokkos::deep_copy(d, h);
        return d;
    };
    const int steps = rng.range(1, 6);
    for (int t = 0; t <= steps; ++t) {
        ++out.calls;
        voronoi::Report report;
        if (verbose) {  // before the call: it may not return
            std::vector<std::array<int, 2>> at;
            for (const P2& p : x)
                at.push_back({exact_predicates::snap(p[0], box, 0), exact_predicates::snap(p[1], box, 1)});
            std::sort(at.begin(), at.end());
            const long long coincident = at.end() - std::unique(at.begin(), at.end());
            std::cout << label << " box " << box.side[0] << " x " << box.side[1] << " step " << t << " coincident "
                      << coincident << '\n'
                      << std::flush;
        }
        try {
            report = mesh.update(to_view(x));
        } catch (const std::exception& e) {
            ++out.refusals;
            const std::string what = e.what(), c = category(what);
            if (verbose) std::cout << "refused: " << what << '\n' << std::flush;
            out.hit(c == "unknown" ? valid : refused, c, what, t);
            if (mesh.faces() != 0) out.hit(wrong, "after_refusal", "a refused update left a mesh", t);
            if (out.refusals >= 3) return;
        }
        if (mesh.particles() > 0) {
            mesh.update_cells();
            const auto volume = support::to_host(mesh.volume());
            const auto q = support::to_host(mesh.positions());
            const long long period[2] = {box.period[0], box.period[1]};
            std::vector<std::array<long long, 2>> g(n);
            for (int i = 0; i < n; ++i) g[i] = {std::llround(q(i, 0) / box.step), std::llround(q(i, 1) / box.step)};
            double total = 0, worst = 0;
            for (int i = 0; i < n; ++i) {
                total += volume(i);
                if (!std::isfinite(volume(i)) || !(volume(i) > 0))
                    out.hit(wrong, "volume_2d", "cell " + std::to_string(i), t);
            }
            if (std::abs(total / (box.side[0] * box.side[1]) - 1) > 1e-10) out.hit(wrong, "total_2d", sci(total), t);
            const int stride = std::max(1, n / 16);
            for (int i = 0; i < n; i += stride) {
                const double brute = double(area_2d(g, i, period) * box.step * box.step);
                worst = std::max(worst, std::abs(volume(i) / brute - 1));
            }
            if (worst > 1e-6)
                out.hit(wrong, "brute_2d", "cell off by " + sci(worst), t);
            else if (worst > 1e-9)
                out.hit(rough, "brute_2d", "cell off by " + sci(worst), t);
            if (!report.rebuilt) {
                Mesh2 fresh(Exec{}, box);
                try {
                    fresh.update(to_view(x));
                    fresh.update_cells();
                    const auto again = support::to_host(fresh.volume());
                    for (int i = 0; i < n; ++i)
                        if (again(i) != volume(i)) {
                            out.hit(bits, "fresh_2d", "cell " + std::to_string(i), t);
                            break;
                        }
                } catch (const std::exception&) {  // a refused fresh build: nothing to compare
                    out.hit(valid, "fresh_2d_refused", "", t);
                }
            }
            if (out.kind >= wrong) return;
            ++out.exact_calls;
            out.flips += report.flips;
        }
        const double spacing = std::sqrt(box.side[0] * box.side[1] / n);
        const double d =
            rng.chance(0.07) ? rng.between(0.2, 1.1) * box.shortest() : rng.log_between(box.step, 3 * spacing);
        const int motion = rng.below(4);
        for (auto& p : x) {
            if (motion == 0 || (motion == 1 && rng.chance(0.1))) {
                p[0] += rng.between(-d, d);
                p[1] += rng.between(-d, d);
            } else if (motion == 2) {
                p[0] += d * std::sin(2 * M_PI * p[1] / box.side[1]);
            }
            for (int a = 0; a < 2; ++a) {
                p[a] -= box.side[a] * std::floor(p[a] / box.side[a]);
                if (p[a] >= box.side[a]) p[a] = 0;
            }
        }
    }
}

void run_low(std::uint64_t seed, Outcome& out, std::string& label, bool verbose) {
    try {
        if (dimension_of(seed) == 1)
            run_1d(seed, out, label, verbose);
        else
            run_2d(seed, out, label, verbose);
    } catch (const std::exception& e) {
        out.hit(valid, category(e.what()), e.what(), 0);
    }
}

// ------------------------------------------------------------------ long runs

struct Shared {
    volatile long long seed;  // the seed being run
    volatile double beat;     // when it started
};

double now_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void line(FILE* f, long long seed, const char* kind, const Script* s, const Outcome* o, double seconds) {
    std::fprintf(
        f,
        "%lld %s %s n=%d calls=%lld refusals=%lld flips=%lld stuck=%lld host=%lld rebuilt=%lld hash=%016" PRIx64
        " t=%.3f op=%d | %s | %s\n",
        seed, kind, o ? o->signature.c_str() : "-", s ? s->particles() : 0, o ? o->calls : 0, o ? o->refusals : 0,
        o ? o->flips : 0, o ? o->stuck : 0, o ? o->host : 0, o ? o->rebuilt : 0, o ? o->hash : 0, seconds,
        o ? o->op : -1, s ? s->label.c_str() : "", o ? o->detail.substr(0, 300).c_str() : "");
    std::fflush(f);
}

// The worker: seeds from *shared on, a line for each.
void work(Shared* shared, const std::string& results, long long last, long long stride, double deadline) {
    FILE* f = std::fopen(results.c_str(), "a");
    if (!f) std::exit(3);
    const Limits limits;
    for (long long seed = shared->seed; seed <= last && now_seconds() < deadline; seed += stride) {
        shared->seed = seed;
        shared->beat = now_seconds();
        const double t0 = now_seconds();
        if (dimension_of(static_cast<std::uint64_t>(seed)) != 3) {
            Script s;
            Outcome o;
            run_low(static_cast<std::uint64_t>(seed), o, s.label, false);
            line(f, seed, class_name[o.kind], &s, &o, now_seconds() - t0);
            continue;
        }
        const Script s = make_script(static_cast<std::uint64_t>(seed));
        Outcome o;
        run_script(s, o, limits);
        line(f, seed, class_name[o.kind], &s, &o, now_seconds() - t0);
    }
    shared->seed = last + stride;
    std::fclose(f);
}

// The seed after the last one of a results file.
long long resume_seed(const std::string& results, long long first, long long stride) {
    std::ifstream in(results);
    std::string text;
    long long seed = first - stride;
    while (std::getline(in, text)) {
        char* end = nullptr;
        const long long s = std::strtoll(text.c_str(), &end, 10);
        if (end != text.c_str()) seed = std::max(seed, s);
    }
    return seed + stride;
}

int long_run(long long first, long long count, long long stride, const std::string& results, double seconds,
             double timeout) {
    const long long last = first + (count - 1) * stride;
    auto* shared =
        static_cast<Shared*>(mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    shared->seed = resume_seed(results, first, stride);
    const double deadline = now_seconds() + seconds;
    std::printf("mesh_fuzz run: seeds %lld to %lld by %lld, resuming at %lld, results %s\n", first, last, stride,
                static_cast<long long>(shared->seed), results.c_str());
    std::fflush(stdout);
    while (shared->seed <= last && now_seconds() < deadline) {
        shared->beat = now_seconds();
        const pid_t child = fork();
        if (child == 0) {
            prctl(PR_SET_PDEATHSIG, SIGKILL);  // no orphan past a killed job
            const rlimit none{0, 0};
            setrlimit(RLIMIT_CORE, &none);
            Kokkos::initialize();
            work(shared, results, last, stride, deadline);
            Kokkos::finalize();
            if (__gcov_dump) __gcov_dump();  // _exit skips it
            _exit(0);
        }
        int status = 0;
        const char* kind = nullptr;
        for (;;) {
            const pid_t done = waitpid(child, &status, WNOHANG);
            if (done == child) {
                if (WIFSIGNALED(status) || (WIFEXITED(status) && WEXITSTATUS(status) != 0)) kind = "CRASH";
                break;
            }
            if (now_seconds() - shared->beat > timeout) {
                kill(child, SIGKILL);
                waitpid(child, &status, 0);
                kind = "HANG";
                break;
            }
            usleep(20000);
        }
        if (kind == nullptr) break;
        const long long seed = shared->seed;
        Outcome o;
        o.kind = std::strcmp(kind, "CRASH") == 0 ? crash : hang;
        o.signature =
            std::string(kind) + ":" + (WIFSIGNALED(status) ? "signal" + std::to_string(WTERMSIG(status)) : "exit");
        const Script s;  // the driver's memory stays small: the seed says it all
        FILE* f = std::fopen(results.c_str(), "a");
        line(f, seed, kind, &s, &o, now_seconds() - shared->beat);
        std::fclose(f);
        shared->seed = seed + stride;
    }
    return 0;
}

// ------------------------------------------------------------------- one case

std::string run_verbose(const Script& s) {
    Limits limits;
    limits.verbose = true;
    Outcome o;
    std::cout << "label " << s.label << "\nbox " << s.sides[0] << " " << s.sides[1] << " " << s.sides[2] << " variant "
              << s.variant << " n " << s.particles() << " calls " << s.ops.size() << '\n'
              << std::flush;
    run_script(s, o, limits);
    std::cout << "outcome " << o.signature << "\ndetail " << o.detail << "\ncalls " << o.calls << " op " << o.op << '\n'
              << std::flush;
    return o.signature;
}

// ------------------------------------------------------------------- shrinking

// The signature of a script, in a process of its own: a candidate may crash or hang.
std::string signature_in_child(const std::string& program, const Script& s, const std::string& scratch,
                               double timeout) {
    save(s, scratch);
    int pipe_ends[2];
    if (pipe(pipe_ends) != 0) return "error";
    const pid_t child = fork();
    if (child == 0) {
        dup2(pipe_ends[1], 1);
        const int null = open("/dev/null", O_WRONLY);
        dup2(null, 2);
        close(pipe_ends[0]);
        const rlimit none{0, 0};
        setrlimit(RLIMIT_CORE, &none);
        execl(program.c_str(), program.c_str(), "case", scratch.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    close(pipe_ends[1]);
    fcntl(pipe_ends[0], F_SETFL, O_NONBLOCK);
    std::string output;
    const double start = now_seconds();
    int status = 0;
    bool done = false, killed = false;
    char chunk[4096];
    while (!done) {
        ssize_t got;
        while ((got = read(pipe_ends[0], chunk, sizeof chunk)) > 0) output.append(chunk, got);
        if (waitpid(child, &status, WNOHANG) == child)
            done = true;
        else if (now_seconds() - start > timeout) {
            kill(child, SIGKILL);
            waitpid(child, &status, 0);
            killed = done = true;
        } else
            usleep(2000);
    }
    ssize_t got;
    while ((got = read(pipe_ends[0], chunk, sizeof chunk)) > 0) output.append(chunk, got);
    close(pipe_ends[0]);
    if (killed) return "HANG:timeout";
    if (WIFSIGNALED(status)) return "CRASH:signal" + std::to_string(WTERMSIG(status));
    const auto at = output.rfind("outcome ");
    if (at == std::string::npos) return "CRASH:exit";
    return output.substr(at + 8, output.find('\n', at) - at - 8);
}

// Delta debugging: calls, then particles, as long as the signature stays.
Script shrink(const Script& start, const std::string& program, const std::string& scratch, const std::string& signature,
              double timeout) {
    Script s = start;
    int runs = 0;
    auto same = [&](const Script& candidate) {
        ++runs;
        return signature_in_child(program, candidate, scratch, timeout) == signature;
    };
    auto drop_calls = [&]() {
        bool any = false;
        for (std::size_t width = std::max<std::size_t>(s.ops.size() / 2, 1); width >= 1; width /= 2) {
            for (std::size_t first = 0; first + width <= s.ops.size();) {
                Script c = s;
                c.ops.erase(c.ops.begin() + first, c.ops.begin() + first + width);
                if (!c.ops.empty() && same(c)) {
                    s = std::move(c);
                    any = true;
                } else {
                    first += width;
                }
            }
            if (width == 1) break;
        }
        return any;
    };
    auto drop_particles = [&]() {
        bool any = false;
        int n = s.particles();
        for (int width = std::max(n / 2, 1); width >= 1; width /= 2) {
            for (int first = 0; first < n;) {
                std::vector<char> drop(n, 0);
                for (int i = first; i < std::min(n, first + width); ++i) drop[i] = 1;
                Script c = without_particles(s, drop);
                if (c.particles() > 0 && same(c)) {
                    s = std::move(c);
                    n = s.particles();
                    any = true;
                } else {
                    first += width;
                }
            }
            if (width == 1) break;
            width = std::min(width, std::max(n, 2));
        }
        return any;
    };
    for (int pass = 0; pass < 6; ++pass) {
        const bool a = drop_calls(), b = drop_particles();
        std::cout << "pass " << pass << ": n " << s.particles() << " calls " << s.ops.size() << " after " << runs
                  << " runs" << '\n'
                  << std::flush;
        if (!a && !b) break;
    }
    return s;
}

// --------------------------------------------------------------------- ctest

std::string header_of(const std::string& path) {
    std::ifstream in(path);
    std::string text, known;
    while (std::getline(in, text) && text.rfind("ops ", 0) != 0)
        if (text.rfind("known ", 0) == 0) known = text.substr(6);
    return known;
}

constexpr long long fixed_seeds = 16;
constexpr std::uint64_t cycling_2d[] = {6000000000027, 6000000155537, 6000000831756};  // #36: repair never returned

void smoke(Check& c, const std::string& directory, double seconds, bool with_known) {
    std::vector<std::string> names;
    if (DIR* d = opendir(directory.c_str())) {
        while (dirent* e = readdir(d)) {
            const std::string name = e->d_name;
            if (name.size() > 5 && name.substr(name.size() - 5) == ".case") names.push_back(name);
        }
        closedir(d);
    }
    std::sort(names.begin(), names.end());
    for (const std::string& name : names) {
        const std::string path = directory + "/" + name, known = header_of(path);
        if (!known.empty() && !with_known) {
            std::cout << "skipped, known: " << name << " " << known << '\n' << std::flush;
            continue;
        }
        Outcome o;
        run_script(load(path), o, Limits{});
        c.expect(o.kind <= valid || o.kind == rough, "case " + name + " " + o.signature + " " + o.detail);
    }
    for (const std::uint64_t seed : cycling_2d) {
        Outcome o;
        std::string label;
        run_low(seed, o, label, false);
        c.expect(o.kind <= valid, "seed " + std::to_string(seed) + " " + o.signature + " " + o.detail);
    }
    // Fixed seeds. Boxes of a non-dyadic step meet #29, extreme ones #32: these
    // run in the long fuzz, not here.
    long long cases = 0, skipped = 0, count[classes] = {};
    const double start = now_seconds();
    for (long long seed = 1; seed <= fixed_seeds && now_seconds() - start < seconds; ++seed) {
        const Script s = make_script(static_cast<std::uint64_t>(seed));
        bool dyadic = true;
        for (const char* name : {"long ", "ratio ", "ratiolong ", "hcp ", "123 ", "extreme ", "thin "})
            dyadic = dyadic && s.label.rfind(name, 0) != 0;
        if (!dyadic) {
            ++skipped;
            continue;
        }
        Outcome o;
        Limits limits;
        limits.every_call = 2048;
        run_script(s, o, limits);
        ++cases;
        ++count[o.kind];
        c.expect(o.kind != wrong, "seed " + std::to_string(seed) + " " + o.signature + " " + o.detail);
    }
    std::cout << cases << " seeds, " << skipped << " skipped (#29, #32):";
    for (int k = 0; k < classes; ++k) std::cout << " " << class_name[k] << "=" << count[k];
    std::cout << '\n' << std::flush;
}

std::string option(int argc, char** argv, const std::string& name, const std::string& otherwise) {
    const std::string key = "--" + name + "=";
    for (int i = 1; i < argc; ++i)
        if (std::strncmp(argv[i], key.c_str(), key.size()) == 0) return argv[i] + key.size();
    return otherwise;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 && argv[1][0] != '-' ? argv[1] : "smoke";
    const double timeout = std::stod(option(argc, argv, "timeout", "300"));
    if (mode == "run") {
        const std::string results = option(argc, argv, "results", "");
        if (results.empty()) {
            std::cerr << "run needs --results=FILE" << '\n' << std::flush;
            return 2;
        }
        return long_run(std::stoll(option(argc, argv, "first", "1")), std::stoll(option(argc, argv, "count", "1000")),
                        std::stoll(option(argc, argv, "stride", "1")), results,
                        std::stod(option(argc, argv, "seconds", "1e9")), timeout);
    }
    if (mode == "dump") {  // a seed's Script, so that two builds replay the same doubles
        if (argc < 4) return 2;
        save(make_script(std::stoull(argv[2])), std::string(argv[3]));
        return 0;
    }
    if (mode == "min") {  // a process per candidate, no Kokkos here
        const std::string what = argc > 2 ? argv[2] : "";
        const std::string out = option(argc, argv, "out", "minimal.case");
        char self[4096];
        const ssize_t length = readlink("/proc/self/exe", self, sizeof self - 1);
        if (length <= 0) return 2;
        self[length] = 0;
        Script s =
            what.find_first_not_of("0123456789") == std::string::npos ? make_script(std::stoull(what)) : load(what);
        const std::string scratch = out + ".try";
        const std::string signature = signature_in_child(self, s, scratch, timeout);
        std::cout << "seed or file " << what << ": " << signature << '\n' << std::flush;
        if (signature == "exact" || signature.rfind("refused", 0) == 0) return 0;
        s = shrink(s, self, scratch, signature, timeout);
        s.label += " (min of " + what + ", " + signature + ")";
        save(s, out);
        std::remove(scratch.c_str());
        std::cout << "minimal: n " << s.particles() << " calls " << s.ops.size() << " " << signature << " -> " << out
                  << '\n'
                  << std::flush;
        return 0;
    }
    Kokkos::ScopeGuard guard;
    if (mode == "one" || mode == "case") {
        if (argc < 3) return 2;
        if (mode == "one" && dimension_of(std::stoull(argv[2])) != 3) {
            Outcome o;
            std::string label;
            run_low(std::stoull(argv[2]), o, label, true);
            std::cout << "label " << label << "\noutcome " << o.signature << "\ndetail " << o.detail << '\n'
                      << std::flush;
            return 0;
        }
        const Script s = mode == "one" ? make_script(std::stoull(argv[2])) : load(argv[2]);
        const std::string dump = option(argc, argv, "dump", "");
        if (!dump.empty()) save(s, dump);
        run_verbose(s);
        return 0;
    }
    Check c;
    smoke(c, option(argc, argv, "cases", "fuzz_cases"), std::stod(option(argc, argv, "seconds", "60")),
          option(argc, argv, "known", "") == "1");
    std::cout << (c.failed() ? "FAILED\n" : "passed\n");
    return c.failed() != 0;
}
