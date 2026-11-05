// 3D special cases of the mesh, each against the independent oracle
// (oracle.hpp, through mesh_check.hpp), each named:
//   mesh_cases_test list              the names
//   mesh_cases_test <prefix> ...      the cases whose name starts with a
//                                     prefix, in one process (ctest runs one
//                                     per group)
// A case either requires the exact mesh, or accepts a refusal (an exception)
// in place of it: what it never accepts is a wrong mesh without an error.
// Cases of a known open problem carry the issue and run with --known only.

#include "check.hpp"
#include "mesh_check.hpp"

#include <functional>
#include <iomanip>
#include <limits>
#include <random>

namespace {

using namespace support;
namespace mc = mesh_check;

struct Case {
    std::string name;
    std::function<void(Check&)> run;
    int issue = 0;  // an open issue: runs with --known only
};
std::vector<Case>& cases() {
    static std::vector<Case> all;
    return all;
}
void add(const std::string& name, std::function<void(Check&)> run, int issue = 0) {
    cases().push_back({name, std::move(run), issue});
}

enum class Expect { exact, exact_or_refused, refused };

// A fresh build of `points`, judged. Returns whether there is a mesh.
bool build(Check& check, const std::string& name, Mesh& mesh, const Points& points, Expect expect, int cells = 32) {
    try {
        const voronoi::Report r = mesh.update(to_device(points));
        mesh.update_cells();
        const mc::Result c = mc::check(mesh, cells);
        check.expect(expect != Expect::refused && c.ok(), name + ": built (patched " + std::to_string(r.start.patched) +
                                                              ", rounds " + std::to_string(r.start.rounds) + ", faces " +
                                                              std::to_string(r.faces) + "): " + c.text());
        return true;
    } catch (const std::exception& e) {
        check.expect(expect != Expect::exact, name + ": refused: " + e.what());
        return false;
    }
}

// One update to `points`, judged.
voronoi::Report step(Check& check, const std::string& name, Mesh& mesh, const Points& points, int cells = 16) {
    const voronoi::Report r = mesh.update(to_device(points));
    mesh.update_cells();
    const mc::Result c = mc::check(mesh, cells);
    std::ostringstream what;
    what << name << ": flips " << r.flips << " rounds " << r.rounds << " stuck " << r.stuck << " host " << r.host_flips
         << "/" << r.host_rebuilds << " rebuilt " << r.rebuilt << ": " << c.text();
    check.expect(c.ok(), what.str());
    return r;
}

// The same, accepting a refusal.
voronoi::Report step_or_refused(Check& check, const std::string& name, Mesh& mesh, const Points& points, bool& refused,
                                int cells = 16) {
    try {
        return step(check, name, mesh, points, cells);
    } catch (const std::exception& e) {
        check.expect(true, name + ": refused: " + e.what());
        refused = true;
        return {};
    }
}

Point on_sphere(std::mt19937_64& rng) {
    std::normal_distribution<double> g(0, 1);
    const Point p{g(rng), g(rng), g(rng)};
    const double r = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    return {p[0] / r, p[1] / r, p[2] / r};
}

void add_ball(Points& p, std::mt19937_64& rng, Point c, double radius, int n) {
    std::uniform_real_distribution<double> u(0, 1);
    for (int i = 0; i < n; ++i) {
        const Point d = on_sphere(rng);
        const double r = radius * std::cbrt(u(rng));
        p.push_back({wrap(c[0] + r * d[0]), wrap(c[1] + r * d[1]), wrap(c[2] + r * d[2])});
    }
}

void add_shell(Points& p, std::mt19937_64& rng, Point c, double radius, int n) {
    for (int i = 0; i < n; ++i) {
        const Point d = on_sphere(rng);
        p.push_back({wrap(c[0] + radius * d[0]), wrap(c[1] + radius * d[1]), wrap(c[2] + radius * d[2])});
    }
}

void add_uniform(Points& p, std::mt19937_64& rng, int n) {
    std::uniform_real_distribution<double> u(0, 1);
    for (int i = 0; i < n; ++i) p.push_back({u(rng), u(rng), u(rng)});
}

const double grid_step = 1.0 / exact_predicates::scale;

// ---------------------------------------------------------------- coincident

void coincident_cases() {
    for (const bool lattice : {false, true}) {
        const std::string on = lattice ? ".lattice" : "";
        const char* kind = lattice ? "lattice" : "jittered";
        add("coincident.start.pair" + on, [=](Check& c) {
            Points p = make_points(kind, 4096);
            p[7] = p[1000];
            Mesh m(Exec{}, 1.0);
            build(c, "two particles at one position", m, p, Expect::exact_or_refused);
        });
        add("coincident.start.one_step" + on, [=](Check& c) {
            Points p = make_points(kind, 4096);
            p[7] = p[1000];
            p[7][0] += grid_step;
            Mesh m(Exec{}, 1.0);
            build(c, "two particles one grid step apart", m, p, Expect::exact);
        });
        add("coincident.start.collapse" + on, [=](Check& c) {
            Points p = make_points(kind, 4096);
            p[7] = p[1000];
            p[7][0] += 0.25 * grid_step;  // apart in double, one position on the grid
            Mesh m(Exec{}, 1.0);
            build(c, "two doubles on one grid position", m, p, Expect::exact_or_refused);
        });
        add("coincident.start.triple" + on, [=](Check& c) {
            Points p = make_points(kind, 4096);
            p[7] = p[9] = p[1000];
            Mesh m(Exec{}, 1.0);
            build(c, "three particles at one position", m, p, Expect::exact_or_refused);
        });
        add("coincident.start.percent" + on, [=](Check& c) {
            Points p = make_points(kind, 4096);
            for (std::size_t i = 0; i + 1 < p.size(); i += 100) p[i] = p[i + 1];
            Mesh m(Exec{}, 1.0);
            build(c, "1 % of the particles doubled", m, p, Expect::exact_or_refused);
        });
    }
    add("coincident.start.all_doubled", [](Check& c) {
        Points p = make_points("jittered", 2048);
        const std::size_t n = p.size();
        for (std::size_t i = 0; i < n; ++i) p.push_back(p[i]);
        Mesh m(Exec{}, 1.0);
        build(c, "every particle twice", m, p, Expect::exact_or_refused);
    });
    add("coincident.start.two_sites", [](Check& c) {
        Points p(100, Point{0.25, 0.25, 0.25});
        for (int i = 0; i < 50; ++i) p[i] = {0.75, 0.75, 0.75};
        Mesh m(Exec{}, 1.0);
        build(c, "100 particles on two positions", m, p, Expect::exact_or_refused);
    });
    add("coincident.start.all_same", [](Check& c) {
        Points p(64, Point{0.5, 0.5, 0.5});
        Mesh m(Exec{}, 1.0);
        build(c, "64 particles at one position", m, p, Expect::exact_or_refused);
    });

    // During steps. Every particle stays; a few move as written.
    auto walk = [](Check& c, const std::string& name, Points p, const std::vector<std::pair<int, Point>>& targets, int steps,
                   int stay) {
        Mesh m(Exec{}, 1.0);
        if (!build(c, name + ", start", m, p, Expect::exact)) return;
        std::vector<Point> from;
        for (const auto& [i, t] : targets) from.push_back(p[i]);
        bool refused = false;
        for (int s = 1; s <= steps + stay + 2 && !refused; ++s) {
            const double f = std::min(1.0, double(s) / steps);
            for (std::size_t k = 0; k < targets.size(); ++k) {
                const auto& [i, t] = targets[k];
                for (int a = 0; a < 3; ++a) p[i][a] = wrap(from[k][a] + f * nearest_image(t[a] - from[k][a]));
            }
            if (s > steps + stay)  // leave again
                for (const auto& [i, t] : targets) p[i][0] = wrap(p[i][0] + (s - steps - stay) * 1e-3);
            step_or_refused(c, name + ", step " + std::to_string(s), m, p, refused);
        }
    };
    add("coincident.step.onto", [=](Check& c) {
        Points p = make_points("jittered", 4096);
        walk(c, "a particle moves onto another", p, {{7, p[1000]}}, 6, 2);
    });
    add("coincident.step.onto_lattice", [=](Check& c) {
        Points p = make_points("lattice", 4096);
        walk(c, "a lattice particle moves onto a neighbour", p, {{7, p[8]}}, 6, 2);
    });
    add("coincident.step.several", [=](Check& c) {
        Points p = make_points("jittered", 4096);
        std::vector<std::pair<int, Point>> t;
        for (int i = 0; i < 40; ++i) t.push_back({i * 100, p[i * 100 + 1]});
        walk(c, "40 particles move onto neighbours", p, t, 6, 2);
    });
    add("coincident.step.collapse", [](Check& c) {
        // Two particles close to within a grid step in double, on one position.
        Points p = make_points("jittered", 4096);
        Mesh m(Exec{}, 1.0);
        build(c, "start", m, p, Expect::exact);
        const Point centre = p[1000];
        std::vector<double> gap{1e-3, 1e-5, 1e-7, 8 * grid_step, grid_step, 0.3 * grid_step, 1e-3 * grid_step, grid_step};
        bool refused = false;
        for (std::size_t s = 0; s < gap.size() && !refused; ++s) {
            p[7] = centre;
            p[7][1] += gap[s];
            step_or_refused(c, "separation " + sci(gap[s]), m, p, refused);
        }
    });
    add("coincident.step.swap", [](Check& c) {
        // Two neighbours exchange places through one point.
        Points p = make_points("jittered", 4096);
        Mesh m(Exec{}, 1.0);
        build(c, "start", m, p, Expect::exact);
        const Point a = p[1000], b = p[1001];
        bool refused = false;
        for (int s = 1; s <= 8 && !refused; ++s) {
            const double f = s / 8.0;
            for (int k = 0; k < 3; ++k) {
                p[1000][k] = wrap(a[k] + f * nearest_image(b[k] - a[k]));
                p[1001][k] = wrap(b[k] + f * nearest_image(a[k] - b[k]));
            }
            to_grid(p);
            step_or_refused(c, "swap step " + std::to_string(s), m, p, refused);
        }
    });
    add("coincident.step.mass_merge", [](Check& c) {
        // A quarter of the particles land on their neighbour in one step.
        Points p = make_points("jittered", 4096);
        Mesh m(Exec{}, 1.0);
        build(c, "start", m, p, Expect::exact);
        for (std::size_t i = 0; i + 3 < p.size(); i += 4) p[i] = p[i + 1];
        bool refused = false;
        step_or_refused(c, "merge", m, p, refused);
    });
    add("coincident.step.near_pairs", [](Check& c) {
        // Every other particle follows its neighbour at one grid step.
        Points p = make_points("jittered", 4096);
        Mesh m(Exec{}, 1.0);
        build(c, "start", m, p, Expect::exact);
        for (std::size_t i = 0; i + 1 < p.size(); i += 2) {
            p[i] = p[i + 1];
            p[i][0] = wrap(p[i][0] + grid_step);
        }
        step(c, "pairs one step apart", m, p);
    });
}

// ----------------------------------------------------------------- few points

void few_cases() {
    for (const char* kind : {"uniform", "jittered", "lattice"})
        for (const int n : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 27, 32, 40, 48, 64, 100, 125, 216, 512}) {
            add("few." + std::string(kind) + "." + std::to_string(n), [=](Check& c) {
                const Points p = make_points(kind, n);
                Mesh m(Exec{}, 1.0);
                const std::string name = std::string(kind) + " n=" + std::to_string(p.size());
                if (!build(c, name, m, p, Expect::exact_or_refused, 512)) return;
                Points q = p;
                const Flow flow;
                bool refused = false;
                const double spacing = std::cbrt(1.0 / p.size());
                for (int s = 1; s <= 3 && !refused; ++s) {
                    flow.advect(q, 0.3 * spacing);
                    step_or_refused(c, name + " step " + std::to_string(s), m, q, refused, 512);
                }
            });
        }
    add("few.empty", [](Check& c) {
        Mesh m(Exec{}, 1.0);
        build(c, "no particles", m, {}, Expect::exact_or_refused);
    });
}

// ------------------------------------------------------------------- contrast

void contrast_cases() {
    for (const double ratio : {1e2, 1e4, 1e6, 1e8})
        add("contrast.two_density." + sci(ratio), [=](Check& c) {
            // 512 background particles, 3584 in a cube of the given density ratio.
            std::mt19937_64 rng(1);
            Points p;
            add_uniform(p, rng, 512);
            const double side = std::cbrt(7.0 / ratio);
            std::uniform_real_distribution<double> u(0.4, 0.4 + side);
            for (int i = 0; i < 3584; ++i) p.push_back({u(rng), u(rng), u(rng)});
            Mesh m(Exec{}, 1.0);
            build(c, "density ratio " + sci(ratio), m, p, Expect::exact);
        });
    for (const double radius : {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8}) {
        add("contrast.ball." + sci(radius), [=](Check& c) {
            std::mt19937_64 rng(2);
            Points p;
            add_uniform(p, rng, 512);
            add_ball(p, rng, {0.3, 0.6, 0.45}, radius, 3584);
            Mesh m(Exec{}, 1.0);
            build(c, "3584 particles in a ball of radius " + sci(radius), m, p, Expect::exact_or_refused);
        });
        add("contrast.shell." + sci(radius), [=](Check& c) {
            std::mt19937_64 rng(3);
            Points p;
            add_uniform(p, rng, 512);
            add_shell(p, rng, {0.3, 0.6, 0.45}, radius, 2000);
            Mesh m(Exec{}, 1.0);
            build(c, "2000 particles on a sphere of radius " + sci(radius), m, p, Expect::exact_or_refused);
        });
        add("contrast.shell_centre." + sci(radius), [=](Check& c) {
            std::mt19937_64 rng(4);
            Points p;
            add_uniform(p, rng, 512);
            add_shell(p, rng, {0.3, 0.6, 0.45}, radius, 2000);
            p.push_back({0.3, 0.6, 0.45});
            Mesh m(Exec{}, 1.0);
            build(c, "a particle at the centre of a sphere of radius " + sci(radius), m, p, Expect::exact_or_refused);
        });
    }
    for (const double radius : {0.2, 0.1, 0.01, 1e-3, 1e-5})
        for (const int n : {5, 10, 20, 50, 1000})
            add("contrast.alone." + std::to_string(n) + "." + sci(radius), [=](Check& c) {
                std::mt19937_64 rng(5);
                Points p;
                add_ball(p, rng, {0.5, 0.5, 0.5}, radius, n);
                Mesh m(Exec{}, 1.0);
                build(c, std::to_string(n) + " particles of an empty box in a ball of radius " + sci(radius), m, p,
                      Expect::exact_or_refused, 64);
            });
    add("contrast.hub", [](Check& c) {
        std::mt19937_64 rng(6);
        Points p;
        add_uniform(p, rng, 512);
        add_shell(p, rng, {0.5, 0.5, 0.5}, 0.1, 2000);
        p.push_back({0.5, 0.5, 0.5});
        Mesh m(Exec{}, 1.0);
        build(c, "a cell of about 2000 faces", m, p, Expect::exact_or_refused);
    });
    add("contrast.line", [](Check& c) {
        Points p;
        for (int i = 0; i < 200; ++i) p.push_back({(i + 0.5) / 200, 0.5, 0.5});
        Mesh m(Exec{}, 1.0);
        build(c, "200 particles on a line", m, p, Expect::exact_or_refused);
    });
    add("contrast.plane", [](Check& c) {
        std::mt19937_64 rng(7);
        std::uniform_real_distribution<double> u(0, 1);
        Points p;
        for (int i = 0; i < 500; ++i) p.push_back({u(rng), u(rng), 0.5});
        Mesh m(Exec{}, 1.0);
        build(c, "500 particles in one plane", m, p, Expect::exact_or_refused);
    });
    add("contrast.skew_lines", [](Check& c) {
        // Two skew lines: the quadratic worst case of a Delaunay tetrahedralization.
        Points p;
        for (int i = 0; i < 256; ++i) p.push_back({(i + 0.5) / 256, 0.25, 0.25});
        for (int i = 0; i < 256; ++i) p.push_back({0.75, (i + 0.5) / 256, 0.75});
        Mesh m(Exec{}, 1.0);
        build(c, "256 + 256 particles on two skew lines", m, p, Expect::exact_or_refused, 64);
    });
    add("contrast.skew_segments", [](Check& c) {
        // Short segments, far from their periodic images.
        Points p;
        for (int i = 0; i < 256; ++i) p.push_back({0.4 + 0.2 * i / 256, 0.45, 0.45});
        for (int i = 0; i < 256; ++i) p.push_back({0.5, 0.4 + 0.2 * i / 256, 0.55});
        std::mt19937_64 rng(8);
        add_uniform(p, rng, 512);
        Mesh m(Exec{}, 1.0);
        build(c, "two short skew segments of 256 particles", m, p, Expect::exact_or_refused, 64);
    });
    add("contrast.halo_core", [](Check& c) {
        // A cusp: log-uniform in radius down to about 1e-7.
        std::mt19937_64 rng(9);
        std::uniform_real_distribution<double> u(0, 1);
        Points p;
        add_uniform(p, rng, 512);
        for (int i = 0; i < 3584; ++i) {
            const double r = 0.2 * std::pow(10.0, -6.3 * u(rng));
            const Point d = on_sphere(rng);
            p.push_back({wrap(0.5 + r * d[0]), wrap(0.5 + r * d[1]), wrap(0.5 + r * d[2])});
        }
        Mesh m(Exec{}, 1.0);
        build(c, "a log-uniform cusp over 6 decades", m, p, Expect::exact_or_refused);
    });
}

// ---------------------------------------------------------------------- boxes

void box_cases() {
    // Slabs: m cells across a side of 1/8.
    for (const int across : {1, 2, 3, 4, 5, 6, 7, 8})
        for (const char* kind : {"jittered", "lattice", "uniform"})
            add("box.slab." + std::string(kind) + "." + std::to_string(across), [=](Check& c) {
                const Box box(1, 1, 0.125);
                const int side = 8 * across;
                const Points p = make_points(kind, side * side * across, box);
                Mesh m(Exec{}, box);
                build(c, std::string(kind) + " in a 1:1:1/8 slab, about " + std::to_string(across) + " cells across", m, p,
                      Expect::exact_or_refused, 256);
            });
    for (const double thin : {1.0 / 32, 1.0 / 256, 1.0 / 1024})
        add("box.thin." + sci(thin), [=](Check& c) {
            const Box box(1, 1, thin);
            const Points p = make_points("jittered", 4096, box);
            Mesh m(Exec{}, box);
            build(c, "4096 jittered in a 1:1:" + sci(thin) + " slab", m, p, Expect::exact_or_refused, 64);
        });
    for (const double length : {4.0, 16.0, 64.0, 256.0, 1024.0, 65536.0})
        add("box.long." + sci(length), [=](Check& c) {
            const Box box(1, 1, length);
            const Points p = make_points("jittered", 4096, box);
            Mesh m(Exec{}, box);
            build(c, "4096 jittered in a 1:1:" + sci(length) + " box", m, p, Expect::exact_or_refused, 64);
        });
    add("box.long.few_across", [](Check& c) {
        const Box box(1, 1, 64);
        const Points p = make_points("jittered", 8 * 8 * 512, box);
        Mesh m(Exec{}, box);
        build(c, "8 x 8 x 512 jittered in a 1:1:64 box", m, p, Expect::exact_or_refused, 64);
    });
    // One void in a uniform background, its centre empty or with one
    // particle: the cells around it reach further with the radius.
    for (const double radius : {0.1, 0.15, 0.18, 0.2, 0.22, 0.25, 0.3, 0.35, 0.4, 0.45})
        for (const bool centre : {false, true})
            add(std::string("box.void.") + (centre ? "centre." : "empty.") + sci(radius), [=](Check& c) {
                const Points all = make_points("uniform", 32768);
                Points p;
                for (const auto& q : all) {
                    double d2 = 0;
                    for (int a = 0; a < 3; ++a) d2 += (q[a] - 0.5) * (q[a] - 0.5);
                    if (d2 > radius * radius) p.push_back(q);
                }
                if (centre) p.push_back({0.5, 0.5, 0.5});
                Mesh m(Exec{}, 1.0);
                build(c, "a void of radius " + sci(radius), m, p, Expect::exact_or_refused, 64);
            });
    add("box.edges", [](Check& c) {
        // Particles on every wrap: x = 0, side, just below 0, a period off.
        Points p = make_points("jittered", 4096);
        const double below = std::nextafter(1.0, 0.0);
        const std::vector<Point> special{{0, 0, 0},         {1, 1, 1},           {below, 0.5, 0.5}, {-grid_step, 0.5, 0.25},
                                         {1.0, 0.25, 0.75}, {-0.25, 0.25, 0.25}, {1.75, 1.75, 0.1}, {0.1, -3.5, 2.5}};
        for (std::size_t i = 0; i < special.size(); ++i) p[i * 500 + 3] = special[i];
        Mesh m(Exec{}, 1.0);
        build(c, "particles on and across the boundaries", m, p, Expect::exact);
        // Own snapping, independent of the library's: floor(x / step) mod period.
        const auto g = mc::grid_of(m);
        long long wrong = 0;
        for (std::size_t i = 0; i < p.size(); ++i)
            for (int a = 0; a < 3; ++a) {
                const auto s = static_cast<long long>(std::floor(p[i][a] * exact_predicates::scale));
                const long long period = exact_predicates::period;
                wrong += g.p[i][a] != ((s % period) + period) % period;
            }
        c.expect(wrong == 0, "snapping equals floor(x / step) mod period (" + std::to_string(wrong) + " differ)");
    });
    add("box.edges.step", [](Check& c) {
        // The whole set crosses the boundary: x = side exactly, and beyond.
        Points p = make_points("jittered", 4096);
        Mesh m(Exec{}, 1.0);
        build(c, "start", m, p, Expect::exact);
        for (int s = 1; s <= 4; ++s) {
            Points q = p;  // not wrapped: x runs past the side of the box
            for (auto& a : q) a[0] += 0.3 * s;
            step(c, "unwrapped positions, step " + std::to_string(s), m, q);
        }
        Points on = p;
        for (std::size_t i = 0; i < on.size(); i += 50) on[i][2] = 1.0;  // on the far side exactly
        step(c, "particles on x = side", m, on);
    });
    for (const double scale : {1e-9, 1e-3, 3.0, 1e3, 1e9})
        add("box.scale." + sci(scale), [=](Check& c) {
            // The same points in a cube of any side: the same grid positions,
            // the same tetrahedra, volumes scaled.
            const Points p = make_points("clustered", 4096);
            Mesh unit(Exec{}, 1.0), other(Exec{}, scale);
            unit.update(to_device(p));
            Points q = p;
            for (auto& a : q)
                for (double& v : a) v = (v + 0.5 * grid_step) * scale;  // inside its grid cell, whatever the rounding
            other.update(to_device(q));
            const long long differ = oracle::different_tetrahedra(mc::tets_of(unit), mc::tets_of(other));
            c.expect(differ == 0, "box " + sci(scale) + ": the tetrahedra of the unit box (" + std::to_string(differ) + " differ)");
            const auto a = to_host(unit.volume()), b = to_host(other.volume());
            double worst = 0;
            for (std::size_t i = 0; i < p.size(); ++i) worst = std::max(worst, std::abs(b(i) / (scale * scale * scale) / a(i) - 1));
            c.expect(worst < 1e-12, "volumes scale with the box (" + sci(worst) + ")");
            const mc::Result r = mc::check(other);
            c.expect(r.ok(), r.text());
        });
    for (const std::string what : {"nan", "inf", "huge"})
        add("box.nonfinite." + what, [=](Check& c) {
            const Points p = make_points("jittered", 4096);
            const double bad = what == "nan" ? std::nan("") : what == "inf" ? std::numeric_limits<double>::infinity() : 1e300;
            Points q = p;
            q[100] = {bad, 0.5, 0.5};
            Mesh m(Exec{}, 1.0);
            build(c, what + " at the start", m, q, Expect::exact_or_refused);
            Mesh n(Exec{}, 1.0);
            build(c, "start", n, p, Expect::exact);
            bool refused = false;
            step_or_refused(c, what + " during a step", n, q, refused);
        });
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    coincident_cases();
    few_cases();
    contrast_cases();
    box_cases();
    std::vector<std::string> wanted;
    bool known = false, list = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--known")
            known = true;
        else if (a == "list")
            list = true;
        else
            wanted.push_back(a);
    }
    if (list) {
        for (const auto& c : cases()) std::cout << c.name << (c.issue ? " known" : "") << "\n";
        return 0;
    }
    Check check;
    for (const auto& c : cases()) {
        if (!wanted.empty() &&
            std::none_of(wanted.begin(), wanted.end(), [&](const std::string& w) { return c.name.rfind(w, 0) == 0; }))
            continue;
        if (c.issue && !known) {
            std::cout << "skip  " << c.name << " (known, issue #" << c.issue << ")\n";
            continue;
        }
        std::cout << "case " << c.name << "\n" << std::flush;
        Kokkos::Timer timer;
        c.run(check);
        std::cout << "  time " << timer.seconds() << " s\n" << std::flush;
    }
    return check.failed() != 0;
}
