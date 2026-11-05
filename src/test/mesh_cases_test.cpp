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
#include "distortion.hpp"
#include "mesh_check.hpp"

#include <cstring>
#include <fstream>
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
    // Each fallback, a pair and a few pairs landing in one step, then what a
    // duplicate does to the next steps.
    const std::pair<voronoi::Fallback, std::string> fallbacks[] = {{voronoi::Fallback::cheapest, "cheapest"},
                                                                   {voronoi::Fallback::host, "host"},
                                                                   {voronoi::Fallback::rebuild, "rebuild"}};
    for (const auto& [mode, label] : fallbacks)
        for (const int pairs : {1, 4, 64}) {
            const voronoi::Fallback fallback = mode;
            add("coincident.step.fallback." + label + "." + std::to_string(pairs), [=](Check& c) {
                Points p = make_points("jittered", 4096);
                Mesh m(Exec{}, 1.0);
                m.set_fallback(fallback);
                build(c, "start", m, p, Expect::exact);
                for (int i = 0; i < pairs; ++i) p[i * 50] = p[i * 50 + 1];
                bool refused = false;
                step_or_refused(c, std::to_string(pairs) + " pairs land on one position", m, p, refused, 4);
                if (refused) return;
                const Flow flow;
                for (int s = 1; s <= 2 && !refused; ++s) {
                    for (auto& q : p) q[0] = wrap(q[0] + 1e-3);  // rigid: the pairs stay
                    step_or_refused(c, "the pairs move on, step " + std::to_string(s), m, p, refused);
                }
            });
        }
    for (const int n : {8000, 9261, 12167, 32768, 262144})
        add("coincident.step.default." + std::to_string(n), [=](Check& c) {
            // The default fallback goes to the host when stuck * 1000 <= n.
            Points p = make_points("jittered", n);
            Mesh m(Exec{}, 1.0);
            build(c, "start", m, p, Expect::exact, 4);
            p[50] = p[51];
            bool refused = false;
            step_or_refused(c, "a pair lands on one position, default fallback, n=" + std::to_string(n), m, p, refused, 4);
            c.expect(refused, "refused");
        });
    // The refusal says what is wrong: the pair, by id. At the start, at each
    // fallback during a step, and with the pair apart in double.
    auto names_pair = [](Check& c, const std::string& name, Mesh& m, const Points& p, int a, int b) {
        const std::string pa = std::to_string(a), pb = std::to_string(b);
        try {
            m.update(to_device(p));
            c.expect(false, name + ": refused");
        } catch (const std::exception& e) {
            const std::string what = e.what();
            c.expect(what.find("particles " + pa + " and " + pb + " ") != std::string::npos, name + ": names the pair: " + what);
        }
    };
    add("coincident.message.start", [=](Check& c) {
        Points p = make_points("jittered", 4096);
        p[7] = p[1000];
        Mesh m(Exec{}, 1.0);
        names_pair(c, "at the start", m, p, 7, 1000);
    });
    add("coincident.message.snapped", [=](Check& c) {
        Points p = make_points("jittered", 4096);
        p[7] = p[1000];
        p[7][0] += 0.25 * grid_step;
        Mesh m(Exec{}, 1.0);
        names_pair(c, "apart in double, one grid position", m, p, 7, 1000);
    });
    for (const auto& [mode, label] : fallbacks)
        for (const int n : {4096, 12167})
            add("coincident.message.step." + label + "." + std::to_string(n), [=, fallback = mode](Check& c) {
                Points p = make_points("jittered", n);
                Mesh m(Exec{}, 1.0);
                m.set_fallback(fallback);
                build(c, "start", m, p, Expect::exact, 4);
                p[50] = p[51];
                names_pair(c, "during a step", m, p, 50, 51);
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
    add(
        "few.empty",
        [](Check& c) {
            Mesh m(Exec{}, 1.0);
            build(c, "no particles", m, {}, Expect::refused);
        });
    add("few.empty_after_a_mesh", [](Check& c) {
        // Nothing to mesh changes nothing: the mesh stays and steps on.
        Points p = make_points("jittered", 1000);
        Mesh m(Exec{}, 1.0);
        build(c, "start", m, p, Expect::exact);
        build(c, "no particles", m, {}, Expect::refused);
        c.expect(m.particles() == static_cast<int>(p.size()), "the mesh stays");
        const Flow flow;
        flow.advect(p, 0.01);
        step(c, "a step after", m, p);
    });
    add("few.single_message", [](Check& c) {
        Mesh m(Exec{}, 1.0);
        try {
            m.update(to_device(Points{{0.5, 0.5, 0.5}}));
            c.expect(false, "one particle refused");
        } catch (const std::exception& e) {
            const std::string what = e.what();
            c.expect(what.find("too few particles") != std::string::npos, "one particle: too few: " + what);
        }
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
            // Below 1e-6 the sphere has particles on one grid position.
            build(c, "2000 particles on a sphere of radius " + sci(radius), m, p,
                  radius >= 1e-6 ? Expect::exact : Expect::exact_or_refused);
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
    for (const int steps : {512, 1024, 2048, 4096, 16384, 65536, 1 << 20})
        add("contrast.grid_scale." + std::to_string(steps), [=](Check& c) {
            // A dense clump on `steps`^3 grid positions, inside the background.
            std::mt19937_64 rng(10);
            std::uniform_int_distribution<int> u(0, steps - 1);
            Points p;
            add_uniform(p, rng, 1024);
            std::vector<std::array<int, 3>> seen;
            for (int i = 0; i < 3072; ++i) {
                seen.push_back({u(rng), u(rng), u(rng)});
                const auto& q = seen.back();
                p.push_back({0.25 + q[0] * grid_step, 0.5 + q[1] * grid_step, 0.75 + q[2] * grid_step});
            }
            std::sort(seen.begin(), seen.end());
            const long long same = std::distance(std::unique(seen.begin(), seen.end()), seen.end());
            Mesh m(Exec{}, 1.0);
            build(c, "3072 particles on " + std::to_string(steps) + "^3 grid positions, " + std::to_string(same) + " duplicates",
                  m, p, same ? Expect::exact_or_refused : Expect::exact, 16);
        });
    // Why a start fails: statuses of the clipping and the checks of tet_start.
    auto diagnose = [](Check& c, const std::string& name, const Points& p) {
        const Positions x = to_device(p);
        tet_mesh::Mesh<Memory> tets;
        const tet_start::Stats stats = start_mesh(x, tets);
        const auto list = cell_list(x);
        auto cells = cells_with_rings(list);
        const auto status = to_host(cells.status);
        std::array<int, shell_voronoi::status_count> counts{};
        for (std::size_t i = 0; i < status.extent(0); ++i) ++counts[status(i)];
        std::ostringstream o;
        o << name << ": exact " << stats.exact << " rounds " << stats.rounds << " bad " << stats.bad_tets << " patched "
          << stats.patched << " tets " << stats.tets << " statuses";
        for (const int k : counts) o << " " << k;
        Points sorted = p;
        to_grid(sorted);
        std::sort(sorted.begin(), sorted.end());
        o << " duplicates " << std::distance(std::unique(sorted.begin(), sorted.end()), sorted.end());
        c.expect(stats.exact, o.str());
    };
    for (const int k : {800, 1000, 1200, 1600, 2000})
        add("contrast.sphere_status." + std::to_string(k), [=](Check& c) {
            std::mt19937_64 rng(3);
            Points p;
            add_uniform(p, rng, 512);
            add_shell(p, rng, {0.3, 0.6, 0.45}, 0.1, k);
            diagnose(c, std::to_string(k) + " particles on a sphere", p);
        });
    // Half the particles in a ball of radius 0.05: fine at small n, the start of
    // the `expansion` flow at 128^3. Too slow for ctest: not in a group it runs.
    for (const int n : {524288, 1048576, 2097152})
        add("dense_ball." + std::to_string(n), [=](Check& c) {
            Points p = make_points("jittered", n);
            std::mt19937_64 rng(3);
            std::normal_distribution<double> g(0, 1);
            std::uniform_real_distribution<double> u(0, 1);
            for (std::size_t i = 0; i < p.size(); i += 2) {
                const Point d{g(rng), g(rng), g(rng)};
                const double size = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                const double r = 0.05 * std::cbrt(u(rng));
                for (int a = 0; a < 3; ++a) p[i][a] = 0.5 + r * d[a] / size;
            }
            diagnose(c, "half of " + std::to_string(n) + " in a ball", p);
        });
    add("contrast.refusal_messages", [](Check& c) {
        // What each status of the clipping is reported as. Made by hand: a cell
        // of more planes than the exact clipping holds costs minutes to reach.
        const Positions x = to_device(make_points("jittered", 64));
        tet_mesh::Points4<Memory> snapped("snapped", 64);
        Kokkos::parallel_for("snapped", Kokkos::RangePolicy<Exec>(0, 64), KOKKOS_LAMBDA(int i) {
            for (int a = 0; a < 3; ++a) snapped(i, a) = x(i, a);
        });
        auto says = [&](int status, const std::string& word) {
            auto cells = shell_voronoi::Cells<Memory>::with_rings(64, 1024);
            Kokkos::deep_copy(Kokkos::subview(cells.status, 3), status);
            const std::string what = voronoi::detail::not_exact(cells, snapped);
            c.expect(what.find(word) != std::string::npos, "status " + std::to_string(status) + ": " + what);
        };
        says(shell_voronoi::uncertified, "too few particles");
        says(shell_voronoi::plane_overflow, "16000 planes");
        says(shell_voronoi::vertex_overflow, "32000 vertices");
        says(shell_voronoi::face_overflow, "1 out of pool room");
        says(shell_voronoi::degenerate, "1 degenerate");
    });
    add("contrast.rank_one", [=](Check& c) {
        // 512 particles along one line that winds around the torus, a clump.
        Points tiny = make_points("uniform", 4096);
        for (auto& q : tiny)
            for (double& v : q) v = 0.5 + (v - 0.5) * 2048 * grid_step;
        Points line;
        for (int i = 1; i < 512; ++i) line.push_back({wrap(0.1 + std::sqrt(2.0) * i), wrap(0.2 + std::sqrt(3.0) * i), wrap(0.3 + std::sqrt(5.0) * i)});
        diagnose(c, "the line alone", line);
        Points both = tiny;
        both.insert(both.end(), line.begin(), line.end());
        diagnose(c, "line and clump", both);
    });
    add("contrast.hub", [](Check& c) {
        std::mt19937_64 rng(6);
        Points p;
        add_uniform(p, rng, 512);
        add_shell(p, rng, {0.5, 0.5, 0.5}, 0.1, 2000);
        p.push_back({0.5, 0.5, 0.5});
        Mesh m(Exec{}, 1.0);
        build(c, "a cell of about 2000 faces", m, p, Expect::exact);
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
        build(c, "two short skew segments of 256 particles", m, p, Expect::exact, 64);
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

Points exact_lattice(int side, const Points& basis);

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
                      Expect::exact_or_refused, 32);
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
                const Points all = make_points("uniform", 8192);
                Points p;
                for (const auto& q : all) {
                    double d2 = 0;
                    for (int a = 0; a < 3; ++a) d2 += (q[a] - 0.5) * (q[a] - 0.5);
                    if (d2 > radius * radius) p.push_back(q);
                }
                if (centre) p.push_back({0.5, 0.5, 0.5});
                Mesh m(Exec{}, 1.0);
                build(c, "a void of radius " + sci(radius), m, p, Expect::exact_or_refused, 16);
            });
    add("box.edges", [](Check& c) {
        // Particles on every wrap: x = 0, side, just below 0, a period off.
        Points p = make_points("jittered", 4096);
        const double below = std::nextafter(1.0, 0.0);
        const std::vector<Point> special{{0, 0, 0},         {1, 1, 0.5},         {below, 0.5, 0.5}, {-grid_step, 0.5, 0.25},
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
    for (const double side : {1.0, 3.0, 0.7, 0.1, 1e-3, 1.0 / 3, 3.086e24})
        add("box.snap_twice." + sci(side), [=](Check& c) {
            // Snapping a snapped position must keep it: the mesh's positions
            // are k * step and tet_start snaps them again.
            const Box box(side);
            const Points p = scaled(make_points("clustered", 4096), box);
            long long moved = 0;
            for (const auto& q : p)
                for (int a = 0; a < 3; ++a) {
                    const int k = exact_predicates::snap(q[a], box, a);
                    moved += exact_predicates::snap(k * box.step, box, a) != k;
                }
            c.expect(moved == 0, "side " + sci(side) + ": " + std::to_string(moved) + " of " + std::to_string(3 * p.size()) +
                                     " coordinates move when snapped twice");
            // On a lattice a step decides ties: the mesh of the positions the
            // caller gave, by the oracle (no tie-break, every image).
            Points lattice = scaled(exact_lattice(16, {{0.5, 0.5, 0.5}}), box);
            for (auto& q : lattice)
                for (int a = 0; a < 3; ++a) q[a] = (exact_predicates::snap(q[a], box, a) + 0.5) * box.step;
            Mesh m(Exec{}, box);
            m.update(to_device(lattice));
            const auto r = oracle::check(mc::grid_of(m), mc::tets_of(m));
            c.expect(r.ok(), "the lattice in that box: " + r.text());
        }, side == 1e-3 ? 22 : 0);
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

// ------------------------------------------------- degeneracy during motion

// A cubic lattice of side^3 cells with the basis, side a power of two, so
// every position is exact on the grid.
Points exact_lattice(int side, const Points& basis) {
    Points p;
    for (int i = 0; i < side; ++i)
        for (int j = 0; j < side; ++j)
            for (int k = 0; k < side; ++k)
                for (const auto& b : basis) p.push_back({(i + b[0]) / side, (j + b[1]) / side, (k + b[2]) / side});
    return p;
}
const Points sc_basis{{0.5, 0.5, 0.5}}, bcc_basis{{0.25, 0.25, 0.25}, {0.75, 0.75, 0.75}},
    fcc_basis{{0.25, 0.25, 0.25}, {0.75, 0.75, 0.25}, {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75}};

void degenerate_cases() {
    struct Layout {
        std::string name;
        Points points;
    };
    for (const Layout& l : {Layout{"sc", exact_lattice(16, sc_basis)}, Layout{"bcc", exact_lattice(8, bcc_basis)},
                            Layout{"fcc", exact_lattice(8, fcc_basis)}}) {
        const Points start = l.points;
        const std::string name = l.name;
        add("degenerate.rigid.on_grid." + name, [=](Check& c) {
            // Rigid, whole grid steps: every predicate is the same; no flip.
            Points p = start;
            Mesh m(Exec{}, 1.0);
            build(c, name + " start", m, p, Expect::exact);
            const double d = std::floor(0.37 / 16 * exact_predicates::scale) * grid_step;
            for (int s = 1; s <= 6; ++s) {
                for (auto& q : p) {
                    q[0] = wrap(q[0] + d);
                    q[1] = wrap(q[1] + 2 * d);
                    q[2] = wrap(q[2] - 3 * d);
                }
                const auto r = step(c, name + " rigid step " + std::to_string(s), m, p);
                c.expect(r.flips == 0 && !r.rebuilt, "no flip, no rebuild");
            }
        });
        add("degenerate.rigid.off_grid." + name, [=](Check& c) {
            // Rigid in double: snapping rounds each particle by up to a grid
            // step, so the lattice is near-degenerate after every step.
            Points p = start;
            Mesh m(Exec{}, 1.0);
            build(c, name + " start", m, p, Expect::exact);
            for (int s = 1; s <= 6; ++s) {
                for (std::size_t i = 0; i < p.size(); ++i)
                    for (int a = 0; a < 3; ++a) p[i][a] = wrap(start[i][a] + s * (0.0231 + 0.0007 * a) + 1.3e-10 * s);
                step(c, name + " rigid off the grid, step " + std::to_string(s), m, p);
            }
        });
        add("degenerate.shear.layers." + name, [=](Check& c) {
            // Layers shift by whole lattice spacings every 8 steps, so an
            // exact lattice recurs; periodic (triangular profile).
            Points p = start;
            Mesh m(Exec{}, 1.0);
            build(c, name + " start", m, p, Expect::exact);
            const int side = name == "sc" ? 16 : 8;
            for (int s = 1; s <= 16; ++s) {
                for (std::size_t i = 0; i < p.size(); ++i) {
                    const int layer = static_cast<int>(std::floor(start[i][2] * side));
                    const double profile = std::min(layer, side - layer);
                    p[i][0] = wrap(start[i][0] + profile * s / 8.0 / side);
                }
                const auto r = step(c, name + " shear step " + std::to_string(s), m, p);
                if (s % 8 == 0) c.expect(true, "lattice recurs: flips " + std::to_string(r.flips));
            }
        });
        add("degenerate.shear.smooth." + name, [=](Check& c) {
            Points p = start;
            Mesh m(Exec{}, 1.0);
            build(c, name + " start", m, p, Expect::exact);
            for (int s = 1; s <= 12; ++s) {
                for (std::size_t i = 0; i < p.size(); ++i)
                    p[i][0] = wrap(start[i][0] + 0.02 * s * std::sin(6.283185307179586 * start[i][2]));
                to_grid(p);
                step(c, name + " smooth shear step " + std::to_string(s), m, p);
            }
        });
        for (const int noise : {1, 16, 1024})
            add("degenerate.noise." + name + "." + std::to_string(noise), [=](Check& c) {  // every step all near-degenerate
                // The lattice with +-noise grid steps of fresh noise every step.
                Points p = start;
                Mesh m(Exec{}, 1.0);
                build(c, name + " start", m, p, Expect::exact);
                std::mt19937_64 rng(noise);
                std::uniform_int_distribution<int> u(-noise, noise);
                for (int s = 1; s <= 2; ++s) {
                    for (std::size_t i = 0; i < p.size(); ++i)
                        for (int a = 0; a < 3; ++a) p[i][a] = wrap(start[i][a] + u(rng) * grid_step);
                    step(c, name + " noise step " + std::to_string(s), m, p);
                }
            });
    }
    for (const Layout& l : {Layout{"sc", exact_lattice(16, sc_basis)}, Layout{"bcc", exact_lattice(8, bcc_basis)},
                            Layout{"fcc", exact_lattice(8, fcc_basis)}}) {
        const Points start = l.points;
        const std::string name = l.name;
        add("degenerate.zero_area." + name, [=](Check& c) {
            // Issue 13: the tie-break's edges are faces of zero area. Their
            // area and centroid must stay finite, here and in a step.
            Mesh m(Exec{}, 1.0);
            build(c, name, m, start, Expect::exact);
            Points p = start;
            for (auto& q : p) q[0] = wrap(q[0] + 256 * grid_step);
            for (int s = 0; s < 2; ++s) {
                const auto area = face_areas(m);
                const auto centroid = to_host(m.face_centroid());
                long long zero = 0, bad = 0;
                for (int f = 0; f < m.faces(); ++f) {
                    double a2 = 0;
                    for (int a = 0; a < 3; ++a) {
                        a2 += area(f, a) * area(f, a);
                        bad += !std::isfinite(centroid(f, a));
                    }
                    zero += a2 == 0;
                }
                c.expect(bad == 0, name + ": " + std::to_string(zero) + " of " + std::to_string(m.faces()) +
                                       " faces have zero area, " + std::to_string(bad) + " non-finite centroid components");
                if (s == 0) step(c, "rigid step", m, p);
            }
        });
    }
    add("degenerate.two_lattices", [](Check& c) {
        // sc lattice 2 slides through sc lattice 1 along z, shifted in x and
        // y: all planes coplanar every 8 steps, bcc on the way.
        const Points a = exact_lattice(8, sc_basis);
        Points p = a;
        for (auto q : a) p.push_back({wrap(q[0] + 0.0625), wrap(q[1] + 0.03125), q[2]});
        Mesh m(Exec{}, 1.0);
        build(c, "start, coplanar", m, p, Expect::exact);
        for (int s = 1; s <= 16; ++s) {
            for (std::size_t i = a.size(); i < p.size(); ++i) p[i][2] = wrap(a[i - a.size()][2] + s / 64.0);
            step(c, "sliding step " + std::to_string(s), m, p);
        }
    });
    add("degenerate.mover", [](Check& c) {
        // One particle walks along lattice directions: through face centres
        // (coplanar), edge midpoints and cube centres (cospherical with 8).
        Points p = exact_lattice(16, sc_basis);
        const double h = 1.0 / 16;
        p.push_back({0.5625, 0.5625, 0.5625});  // a cube centre: sites are at (i + 1/2) / 16
        const Point from = p.back();
        Mesh m(Exec{}, 1.0);
        build(c, "start", m, p, Expect::exact);
        const std::vector<Point> directions{{1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 1}, {0, 0, -1}};
        Point at = from;
        for (const auto& d : directions)
            for (int s = 1; s <= 4; ++s) {
                for (int a = 0; a < 3; ++a) at[a] = wrap(at[a] + d[a] * h / 4 * (d[a] != 0 ? 1.0 : 0.0));
                p.back() = at;
                bool refused = false;
                step_or_refused(c, "mover at " + sci(at[0]) + "," + sci(at[1]) + "," + sci(at[2]), m, p, refused);
                if (refused) return;
            }
    });
    add("degenerate.octahedra", [](Check& c) {
        // Octahedra pulsing with different phases: their 6 points are exactly
        // cospherical whenever the phases meet, every 8 steps.
        const int side = 8;
        const double h = 1.0 / side, big = h / 4, small = h / 64;
        auto at = [&](int s) {
            Points p;
            const double r[3] = {big + small * (s % 8 - 4), big + small * ((s + 3) % 8 - 4) * (s % 8 ? 1 : 0) +
                                                                (s % 8 ? 0 : small * (s % 8 - 4)),
                                 big + small * (s % 8 - 4)};
            for (int i = 0; i < side; ++i)
                for (int j = 0; j < side; ++j)
                    for (int k = 0; k < side; ++k)
                        for (int a = 0; a < 3; ++a)
                            for (const int sign : {-1, 1}) {
                                Point q{(i + 0.5) * h, (j + 0.5) * h, (k + 0.5) * h};
                                q[a] += sign * r[a];
                                p.push_back(q);
                            }
            return p;
        };
        Mesh m(Exec{}, 1.0);
        build(c, "start", m, at(0), Expect::exact, 8);
        for (int s = 1; s <= 8; ++s) step(c, "pulse " + std::to_string(s), m, at(s), 8);
    });
    add("degenerate.integer_sphere", [](Check& c) {
        // Grid points exactly on one sphere: x^2 + y^2 + z^2 = R^2 in integers
        // (scaled solutions of a^2 + b^2 + c^2 = 81^2 etc.), exact cospherical.
        std::vector<std::array<int, 3>> roots;
        const int R = 169;
        for (int a = 0; a <= R; ++a)
            for (int b = a; b <= R; ++b) {
                const int c2 = R * R - a * a - b * b;
                if (c2 < 0) break;
                const int s = static_cast<int>(std::lround(std::sqrt(double(c2))));
                if (s >= b && s * s == c2) roots.push_back({a, b, s});
            }
        Points p = make_points("uniform", 2048);
        const long long scale = (1LL << 30) / (R * 8);
        int on_sphere = 0;
        for (auto r : roots) {
            std::sort(r.begin(), r.end());
            do {
                for (const int sx : {-1, 1})
                    for (const int sy : {-1, 1})
                        for (const int sz : {-1, 1}) {
                            if ((r[0] == 0 && sx < 0) || (r[1] == 0 && sy < 0) || (r[2] == 0 && sz < 0)) continue;
                            p.push_back({0.5 + (sx * r[0] * scale) * grid_step, 0.5 + (sy * r[1] * scale) * grid_step,
                                         0.5 + (sz * r[2] * scale) * grid_step});
                            ++on_sphere;
                        }
            } while (std::next_permutation(r.begin(), r.end()));
        }
        Mesh m(Exec{}, 1.0);
        build(c, std::to_string(on_sphere) + " grid points on one sphere", m, p, Expect::exact_or_refused, 64);
    });
}

// ----------------------------------------------------------- high valence

// `k` particles on a circle of radius r around the z axis through `at`, and
// two poles at +-h on the axis; particles of a sparse background elsewhere.
Points ring_points(const Point& at, double r, int k, double h, int background) {
    std::mt19937_64 rng(12);
    Points p;
    Points all;
    add_uniform(all, rng, background);
    for (const auto& q : all) {
        double d2 = 0;
        for (int a = 0; a < 3; ++a) d2 += (q[a] - at[a]) * (q[a] - at[a]);
        if (d2 > 4 * r * r) p.push_back(q);
    }
    for (int i = 0; i < k; ++i) {
        const double t = 6.283185307179586 * i / k;
        p.push_back({at[0] + r * std::cos(t), at[1] + r * std::sin(t), at[2]});
    }
    p.push_back({at[0], at[1], at[2] + h});
    p.push_back({at[0], at[1], at[2] - h});
    return p;
}

void valence_cases() {
    // The face between the poles has k corners, and the edge k tetrahedra.
    for (const int k : {16, 32, 48, 60, 64, 65, 66, 80, 128, 129, 256, 300, 1000}) {
        add("valence.ring.start." + std::to_string(k), [=](Check& c) {
            Mesh m(Exec{}, 1.0);
            build(c, "a face of " + std::to_string(k) + " corners at the start", m,
                  ring_points({0.5, 0.5, 0.5}, 0.1, k, 0.02, 2048), Expect::exact, 2);
        });
        add("valence.ring.step." + std::to_string(k), [=](Check& c) {
            // The poles arrive during a step, from far outside the circle: the
            // start has no long edge, the repair creates it.
            Mesh m(Exec{}, 1.0);
            const Points far = ring_points({0.5, 0.5, 0.5}, 0.1, k, 0.19, 2048);
            if (!build(c, "start, poles apart", m, far, Expect::exact, 2)) return;
            Points p = far;
            const int n = static_cast<int>(p.size());
            bool refused = false;
            for (const double h : {0.15, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02}) {
                p[n - 2][2] = 0.5 + h;
                p[n - 1][2] = 0.5 - h;
                step_or_refused(c, "poles at " + sci(h), m, p, refused, 2);
                c.expect(!refused, "not refused: every face has its corners, however many");
                if (refused) return;
            }
            // Directly: the pole cells against brute clipping, whatever the sample.
            const auto g = mc::grid_of(m);
            const auto volume = to_host(m.volume());
            for (const int i : {n - 2, n - 1}) {
                const double brute = double(oracle::voronoi_cell(g, i).volume * g.step * g.step * g.step);
                c.expect(std::abs(volume(i) - brute) < 1e-9 * brute,
                         "pole cell volume " + sci(volume(i)) + " brute " + sci(brute));
            }
        });
    }
    for (const int k : {50, 100, 200, 400, 800, 1600}) {
        add("valence.sphere." + std::to_string(k), [=](Check& c) {
            std::mt19937_64 rng(13);
            Points p;
            add_uniform(p, rng, 512);
            add_shell(p, rng, {0.5, 0.5, 0.5}, 0.1, k);
            Mesh m(Exec{}, 1.0);
            build(c, std::to_string(k) + " particles on a sphere", m, p, Expect::exact, 16);
        });
        for (const double thickness : {1e-3, 1e-1})
            add("valence.thick_sphere." + std::to_string(k) + "." + sci(thickness), [=](Check& c) {
                std::mt19937_64 rng(14);
                std::uniform_real_distribution<double> u(1 - thickness, 1 + thickness);
                Points p;
                add_uniform(p, rng, 512);
                for (int i = 0; i < k; ++i) {
                    const Point d = on_sphere(rng);
                    const double r = 0.1 * u(rng);
                    p.push_back({0.5 + r * d[0], 0.5 + r * d[1], 0.5 + r * d[2]});
                }
                Mesh m(Exec{}, 1.0);
                build(c, "a shell of relative thickness " + sci(thickness), m, p, Expect::exact, 16);
            });
        add("valence.hub.step." + std::to_string(k), [=](Check& c) {
            // A particle moves into the centre of a shell: a cell of k faces
            // that no build ever saw.
            std::mt19937_64 rng(15);
            std::normal_distribution<double> g(0, 3e-3);
            Points p;
            add_uniform(p, rng, 1024);
            for (int i = 0; i < k; ++i) {
                const Point d = on_sphere(rng);
                const double r = 0.1 + g(rng);
                p.push_back({0.5 + r * d[0], 0.5 + r * d[1], 0.5 + r * d[2]});
            }
            p.push_back({0.5, 0.5, 0.7});
            Mesh m(Exec{}, 1.0);
            if (!build(c, "start, outside the shell", m, p, Expect::exact, 2)) return;
            bool refused = false;
            for (const double z : {0.65, 0.61, 0.59, 0.55, 0.52, 0.5}) {
                p.back()[2] = z;
                step_or_refused(c, "hub at z=" + sci(z), m, p, refused, 2);
                c.expect(!refused, "not refused");
                if (refused) return;
            }
        });
    }
}

// --------------------------------------------------------------- large motion

void large_cases() {
    for (const double d : {0.5, 1.0, 2.0, 5.0, 20.0})
        add("large.flow." + sci(d), [=](Check& c) {
            Points p = make_points("jittered", 4096);
            Mesh m(Exec{}, 1.0);
            build(c, "start", m, p, Expect::exact);
            const Flow flow;  // compressive: a step of many spacings can empty a region
            bool refused = false;
            for (int s = 1; s <= 4 && !refused; ++s) {
                flow.advect(p, d / 16);
                step_or_refused(c, "displacement " + sci(d) + ", step " + std::to_string(s), m, p, refused);
            }
        });
    for (const double d : {0.3, 1.0, 3.0, 10.0})
        add("large.random_walk." + sci(d), [=](Check& c) {
            // Independent kicks: neighbours move apart by d spacings.
            Points p = make_points("jittered", 4096);
            Mesh m(Exec{}, 1.0);
            build(c, "start", m, p, Expect::exact);
            std::mt19937_64 rng(21);
            std::normal_distribution<double> g(0, d / 16);
            for (int s = 1; s <= 4; ++s) {
                for (auto& q : p)
                    for (double& v : q) v = wrap(v + g(rng));
                to_grid(p);
                step(c, "random walk " + sci(d) + ", step " + std::to_string(s), m, p);
            }
        });
    // Kicks as a fraction of the box, several seeds: the mesh is exact, or the
    // update refuses.
    for (const int n : {4096, 32768})
        for (const double kick : {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 1.0})
            add("kicks." + std::to_string(n) + "." + sci(kick), [=](Check& c) {
                long long wrong = 0, refused_count = 0, rebuilt = 0, clean = 0;
                std::string first;
                for (const int seed : {1, 2, 3, 4}) {
                    Points p = make_points("jittered", n);
                    Mesh m(Exec{}, 1.0);
                    m.update(to_device(p));
                    std::mt19937_64 rng(seed);
                    std::normal_distribution<double> g(0, kick);
                    for (int s = 1; s <= 4; ++s) {
                        for (auto& q : p)
                            for (double& v : q) v = wrap(v + g(rng));
                        to_grid(p);
                        voronoi::Report r;
                        try {
                            r = m.update(to_device(p));
                        } catch (const std::exception&) {
                            ++refused_count;
                            break;
                        }
                        rebuilt += r.rebuilt;
                        // Cheap on a mesh of this size: volume against the box.
                        const auto v = to_host(m.volume());
                        double total = 0;
                        for (std::size_t i = 0; i < p.size(); ++i) total += v(i);
                        const auto check = oracle::check(mc::grid_of(m), mc::tets_of(m));
                        if (check.ok() && std::abs(total - 1) < 1e-10) {
                            clean += !r.rebuilt;
                        } else {
                            ++wrong;
                            if (first.empty()) {
                                std::ostringstream o;
                                o << " first: seed " << seed << " step " << s << " stuck " << r.stuck << " rounds " << r.rounds
                                  << " rebuilt " << r.rebuilt << " total volume " << total << " " << check.text();
                                first = o.str();
                            }
                        }
                    }
                }
                std::ostringstream what;
                what << "n=" << n << " kicks of " << kick << " boxes, 4 seeds, 4 steps: wrong " << wrong << ", refused " << refused_count
                     << ", rebuilt " << rebuilt << ", repaired exact " << clean << first;
                c.expect(wrong == 0, what.str());
            });
    add("large.shuffle", [](Check& c) {
        std::mt19937_64 rng(22);
        Points p;
        add_uniform(p, rng, 4096);
        Mesh m(Exec{}, 1.0);
        build(c, "start", m, p, Expect::exact);
        for (int s = 1; s <= 3; ++s) {
            p.clear();
            add_uniform(p, rng, 4096);
            step(c, "every particle somewhere else, step " + std::to_string(s), m, p);
        }
    });
    add("large.permute", [](Check& c) {
        // The same points, other ids: every particle jumps, the geometry stays.
        Points p = make_points("jittered", 4096);
        Mesh m(Exec{}, 1.0);
        build(c, "start", m, p, Expect::exact);
        std::mt19937_64 rng(23);
        std::shuffle(p.begin(), p.end(), rng);
        step(c, "permuted", m, p);
    });
    add("large.noop", [](Check& c) {
        Points p = make_points("clustered", 4096);
        Mesh m(Exec{}, 1.0);
        build(c, "start", m, p, Expect::exact);
        const auto v0 = to_host(m.volume());
        std::vector<double> before(v0.data(), v0.data() + p.size());
        for (int s = 1; s <= 3; ++s) {
            const auto r = step(c, "same positions, step " + std::to_string(s), m, p);
            c.expect(r.flips == 0 && !r.rebuilt && r.stuck == 0, "nothing to do");
            const auto v = to_host(m.volume());
            c.expect(std::equal(before.begin(), before.end(), v.data()), "the same volumes, bit for bit");
        }
    });
    add("large.resize", [](Check& c) {
        // Another number of particles: a build, up and down.
        Mesh m(Exec{}, 1.0);
        build(c, "4096", m, make_points("jittered", 4096), Expect::exact);
        build(c, "8000", m, make_points("jittered", 8000), Expect::exact);
        build(c, "1000", m, make_points("jittered", 1000), Expect::exact);
        Points p = make_points("jittered", 1000);
        const Flow flow;
        flow.advect(p, 0.01);
        step(c, "a step after", m, p);
    });
    add("large.after_refusal", [](Check& c) {
        // A refused update leaves a mesh that can build again.
        Points p = make_points("jittered", 4096);
        Mesh m(Exec{}, 1.0);
        build(c, "start", m, p, Expect::exact);
        Points bad = p;
        bad[7] = bad[1000];
        bool refused = false;
        step_or_refused(c, "a coincident pair", m, bad, refused);
        c.expect(refused, "refused");
        bool again = false;
        step_or_refused(c, "the good positions again", m, p, again);
        c.expect(!again, "works again after the refusal");
        Points q = p;
        const Flow flow;
        flow.advect(q, 0.01);
        step_or_refused(c, "and steps on", m, q, again);
    });
    add("large.after_refusal.empty_object", [](Check& c) {
        // What a refused object offers: nothing, and no crash.
        Points p = make_points("jittered", 4096);
        Mesh m(Exec{}, 1.0);
        build(c, "start", m, p, Expect::exact);
        Points bad = p;
        bad[7] = bad[1000];
        bool refused = false;
        step_or_refused(c, "a coincident pair", m, bad, refused);
        c.expect(refused && m.faces() == 0 && m.particles() == 0, "no mesh after a refusal");
        bool threw = false;
        try {
            m.update_cells();
        } catch (const std::logic_error&) {
            threw = true;
        }
        c.expect(threw, "update_cells has no mesh to work on and says so");
        build(c, "another number of particles builds", m, make_points("jittered", 1000), Expect::exact);
        refused = false;
        Points few = make_points("jittered", 1000);
        few[3] = few[4];
        step_or_refused(c, "refused again", m, few, refused);
        step_or_refused(c, "a second refusal in a row", m, few, refused);
        c.expect(refused, "refused");
        build(c, "and it builds again", m, make_points("jittered", 1000), Expect::exact);
    });
    for (const auto& [mode, label] : {std::pair{voronoi::Fallback::cheapest, "cheapest"}, std::pair{voronoi::Fallback::host, "host"}})
        add(std::string("large.after_refusal.step.") + label, [fallback = mode](Check& c) {
            // Refused at the end of a step that went through the repair and the host.
            Points p = make_points("jittered", 12167);
            Mesh m(Exec{}, 1.0);
            m.set_fallback(fallback);
            build(c, "start", m, p, Expect::exact, 4);
            Points bad = p;
            bad[50] = bad[51];
            bool refused = false;
            step_or_refused(c, "a coincident pair", m, bad, refused, 4);
            c.expect(refused, "refused");
            bool again = false;
            step_or_refused(c, "the good positions again", m, p, again, 4);
            c.expect(!again, "works again after the refusal");
            const Flow flow;
            flow.advect(p, 0.01);
            step_or_refused(c, "and steps on", m, p, again, 4);
            c.expect(!again, "steps on");
        });
    add("large.no_mesh_yet", [](Check& c) {
        Mesh m(Exec{}, 1.0);
        bool threw = false;
        try {
            m.update_cells();
        } catch (const std::logic_error&) {
            threw = true;
        }
        c.expect(threw && m.faces() == 0, "update_cells before any update says so");
    });
    add("large.long_run", [](Check& c) {
        // Thousands of small steps of a solenoidal flow (no void ends it): no
        // drift in faces, tetrahedra, slots, ring pools, memory.
        using namespace distortion;
        DeviceBytes::start();
        const int steps = std::getenv("MESH_LONG_RUN") ? std::atoi(std::getenv("MESH_LONG_RUN")) : 1000;
        Points p = make_points("jittered", 4096);
        const int n = static_cast<int>(p.size());
        Mesh m(Exec{}, 1.0);
        build(c, "start", m, p, Expect::exact);
        Positions x = to_device(p);
        const Field field(Kind::turbulence);
        Kokkos::View<int*, Memory> side("side", n);
        const double dt = 0.1 / 16;
        double faces_first = 0, slots_first = 0;
        std::int64_t bytes_at_100 = 0;
        for (int s = 1; s <= steps; ++s) {
            advance(field, x, side, s * dt, dt);
            m.update(x);
            m.update_cells();
            if (s == 100) {
                faces_first = double(m.faces()) / n;
                slots_first = double(m.repair_state().count) / n;
                bytes_at_100 = DeviceBytes::current;
            }
            if (s % 300 == 0) {
                const mc::Result r = mc::check(m, 4);
                const auto& state = m.repair_state();
                c.expect(r.ok(), "step " + std::to_string(s) + ": " + r.text());
                std::cout << "    step " << s << " faces " << double(m.faces()) / n << " slots " << double(state.count) / n
                          << " free " << state.free_count << " capacity " << double(state.alive.extent(0)) / n
                          << " B/cell " << double(DeviceBytes::current) / n << "\n"
                          << std::flush;
            }
        }
        const double faces = double(m.faces()) / n, slots = double(m.repair_state().count) / n;
        c.expect(std::abs(faces / faces_first - 1) < 0.05, "faces per cell drift below 5 %: " + sci(faces_first) + " -> " + sci(faces));
        c.expect(slots < 1.1 * slots_first, "tetrahedra slots do not creep: " + sci(slots_first) + " -> " + sci(slots));
        c.expect(double(DeviceBytes::current) < 1.05 * double(bytes_at_100), "memory does not creep: " +
                 sci(double(bytes_at_100) / n) + " -> " + sci(double(DeviceBytes::current) / n) + " B/cell");
    });
    add("large.cycle", [](Check& c) {
        // There and back: the same positions give the same mesh, at any cycle.
        const Points p0 = make_points("jittered", 4096);
        Mesh m(Exec{}, 1.0);
        build(c, "start", m, p0, Expect::exact);
        const auto first = mc::tets_of(m);
        const auto v0 = to_host(m.volume());
        const std::vector<double> before(v0.data(), v0.data() + p0.size());
        const Flow flow;
        std::vector<Points> path{p0};
        for (int s = 0; s < 5; ++s) {
            path.push_back(path.back());
            flow.advect(path.back(), 0.3 / 16);
        }
        for (int cycle = 1; cycle <= 8; ++cycle) {
            for (int s = 1; s <= 5; ++s) {
                m.update(to_device(path[s]));
                m.update_cells();
            }
            for (int s = 4; s >= 0; --s) {
                m.update(to_device(path[s]));
                m.update_cells();
            }
        }
        const long long differ = oracle::different_tetrahedra(first, mc::tets_of(m));
        const auto v = to_host(m.volume());
        c.expect(differ == 0, "after 8 cycles the tetrahedra of the start (" + std::to_string(differ) + " differ)");
        c.expect(std::equal(before.begin(), before.end(), v.data()), "and its volumes, bit for bit");
        c.expect(mc::check(m, 8).ok(), "and the oracle passes");
    });
}

// ------------------------------------------------------------------ fallbacks

void fallback_cases() {
    for (const int rounds : {1, 2})
        for (const double d : {1e-4, 1e-3, 3e-3, 1e-2})
            add("fallback.out_of_rounds." + std::to_string(rounds) + "." + sci(d), [=](Check& c) {
                // A repair stopped with triangles left never returns as a mesh.
                // Small steps: few flips, often none stuck when the rounds end.
                Points p = make_points("jittered", 4096);
                Mesh m(Exec{}, 1.0);
                build(c, "start", m, p, Expect::exact);
                m.set_repair_rounds(rounds);
                const Flow flow;
                flow.advect(p, d / 16);
                step(c, "at most " + std::to_string(rounds) + " rounds", m, p);
            });
    const std::pair<voronoi::Fallback, std::string> modes[] = {{voronoi::Fallback::cheapest, "cheapest"},
                                                               {voronoi::Fallback::host, "host"},
                                                               {voronoi::Fallback::rebuild, "rebuild"}};
    for (const auto& [mode, name] : modes) {
        const voronoi::Fallback fallback = mode;
        const std::string label = name;
        add("fallback.shells." + label, [=](Check& c) {
            Points p = make_points("shells@1e-9", 4096);
            Mesh m(Exec{}, 1.0);
            m.set_fallback(fallback);
            build(c, "start", m, p, Expect::exact);
            const Flow flow;
            for (int s = 1; s <= 3; ++s) {
                flow.advect(p, 0.3 / 16);
                step(c, label + " shells step " + std::to_string(s), m, p);
            }
        });
        add("fallback.large." + label, [=](Check& c) {
            Points p = make_points("jittered", 4096);
            Mesh m(Exec{}, 1.0);
            m.set_fallback(fallback);
            build(c, "start", m, p, Expect::exact);
            std::mt19937_64 rng(31);
            std::normal_distribution<double> g(0, 2.0 / 16);
            for (int s = 1; s <= 3; ++s) {
                for (auto& q : p)
                    for (double& v : q) v = wrap(v + g(rng));
                to_grid(p);
                step(c, label + " kicks of 2 spacings, step " + std::to_string(s), m, p);
            }
        });
        add("fallback.growth." + label, [=](Check& c) {
            // From a lattice (few tetrahedra) to two skew segments in one
            // step: more tetrahedra, faces and ring entries than the build
            // sized anything for.
            Points p = make_points("lattice", 1728);
            Mesh m(Exec{}, 1.0);
            m.set_fallback(fallback);
            build(c, "start", m, p, Expect::exact);
            const int n = static_cast<int>(p.size());
            const int count = n / 4;
            for (int i = 0; i < count; ++i) {
                p[i] = {0.4 + 0.2 * i / count, 0.45, 0.45};
                p[count + i] = {0.5, 0.4 + 0.2 * i / count, 0.55};
            }
            to_grid(p);
            bool refused = false;
            const auto r = step_or_refused(c, label + " to skew segments", m, p, refused);
            std::cout << "    faces " << double(r.faces) / n << " per cell\n";
        });
    }
    // What one step of Fallback::host costs against the stuck entries it gets.
    for (const int n : {1000, 1728, 4096, 8000})
        for (const double d : {0.3, 0.5, 1.0, 2.0})
            add("hostcost." + std::to_string(n) + "." + sci(d), [=](Check& c) {
                Points p = make_points("jittered", n);
                const int count = static_cast<int>(p.size());
                Mesh m(Exec{}, 1.0);
                m.set_fallback(voronoi::Fallback::host);
                build(c, "start", m, p, Expect::exact, 0);
                std::mt19937_64 rng(31);
                std::normal_distribution<double> g(0, d * std::cbrt(1.0 / count));
                for (auto& q : p)
                    for (double& v : q) v = wrap(v + g(rng));
                to_grid(p);
                Kokkos::Timer timer;
                voronoi::Report r;
                try {
                    r = m.update(to_device(p));
                } catch (const std::exception& e) {
                    c.expect(true, "n=" + std::to_string(count) + " kicks of " + sci(d) + " spacings: " + sci(timer.seconds()) +
                                       " s to a refusal: " + e.what());
                    return;
                }
                const double seconds = timer.seconds();
                const mc::Result o = mc::check(m, 4);
                std::ostringstream what;
                what << "n=" << count << " kicks of " << d << " spacings: " << seconds << " s, stuck " << r.stuck << ", host "
                     << r.host_flips << "/" << r.host_rebuilds << ", rebuilt " << r.rebuilt << ", " << r.stuck / seconds
                     << " stuck entries/s";
                c.expect(o.ok(), what.str());
            });
    add("fallback.grow_by_steps", [](Check& c) {
        // The same growth in small steps, so that repair must grow its pools.
        Points p = make_points("lattice", 1728);
        Mesh m(Exec{}, 1.0);
        build(c, "start", m, p, Expect::exact);
        const int n = static_cast<int>(p.size()), count = n / 4;
        const Points from = p;
        Points to = p;
        for (int i = 0; i < count; ++i) {
            to[i] = {0.4 + 0.2 * i / count, 0.45, 0.45};
            to[count + i] = {0.5, 0.4 + 0.2 * i / count, 0.55};
        }
        bool refused = false;
        for (int s = 1; s <= 20 && !refused; ++s) {
            for (int i = 0; i < n; ++i)
                for (int a = 0; a < 3; ++a) p[i][a] = wrap(from[i][a] + s / 20.0 * nearest_image(to[i][a] - from[i][a]));
            to_grid(p);
            step_or_refused(c, "towards skew segments, step " + std::to_string(s), m, p, refused, 4);
        }
    });
    add("fallback.clip_overflow", [](Check& c) {
        // clip() of a cell of many faces, with its pool of 10 per cell.
        std::mt19937_64 rng(32);
        std::normal_distribution<double> g(0, 3e-3);
        Points p;
        add_uniform(p, rng, 1024);
        for (int i = 0; i < 100; ++i) {
            const Point d = on_sphere(rng);
            const double r = 0.1 + g(rng);
            p.push_back({0.5 + r * d[0], 0.5 + r * d[1], 0.5 + r * d[2]});
        }
        p.push_back({0.5, 0.5, 0.5});
        Mesh m(Exec{}, 1.0);
        if (!build(c, "hub", m, p, Expect::exact_or_refused, 4)) return;
        const int hub = static_cast<int>(p.size()) - 1;
        Kokkos::View<int*, Memory> cells("cells", 1);
        Kokkos::deep_copy(cells, hub);
        const auto& local = m.clip(to_device(p), cells);
        const auto status = to_host(local.status);
        const auto owned = to_host(local.owned_count);
        const auto vol = to_host(local.volume);
        const auto g2 = mc::grid_of(m);
        const double brute = double(oracle::voronoi_cell(g2, hub).volume * g2.step * g2.step * g2.step);
        c.expect(status(hub) == shell_voronoi::ok && std::abs(vol(hub) - brute) < 1e-9 * brute,
                 "clip of a hub: status " + std::to_string(status(hub)) + ", faces " + std::to_string(owned(hub)) +
                     ", volume " + sci(vol(hub)) + " brute " + sci(brute));
    });
}

// -------------------------------------------------- predicates on real sets

void predicate_cases() {
    // Library predicates on the quintuples a mesh really asks about, against
    // the oracle's integers: each tetrahedron with the apex across each face.
    auto compare = [](Check& c, const std::string& name, const Points& p) {
        Mesh m(Exec{}, 1.0);
        try {
            m.update(to_device(p));
        } catch (const std::exception& e) {
            c.expect(false, name + ": no mesh: " + e.what());
            return;
        }
        const auto g = mc::grid_of(m);
        const auto t = mc::tets_of(m);
        exact_predicates::Points<Kokkos::HostSpace> point("point", g.size());
        for (int i = 0; i < g.size(); ++i)
            for (int a = 0; a < 3; ++a) point(i, a) = static_cast<int>(g.p[i][a]);
        long long wrong = 0, checked = 0, ties = 0, near = 0;
        std::mt19937_64 rng(41);
        std::uniform_int_distribution<int> other(0, g.size() - 1);
        for (int k = 0; k < t.count(); ++k) {
            if (!t.alive[k]) continue;
            const auto& v = t.vertex[k];
            const oracle::Offset a = oracle::difference(g, v[0], v[1]), b = oracle::difference(g, v[0], v[2]),
                                 d = oracle::difference(g, v[0], v[3]);
            const int orientation = oracle::det(a, b, d) > 0 ? 1 : oracle::det(a, b, d) < 0 ? -1 : 0;
            wrong += orientation != exact_predicates::orient(point, v.data());
            if (orientation == 0) continue;
            for (int j = 0; j < 5; ++j) {
                const int u = j < 4 ? t.adjacent[k][j] : -1;
                int e = other(rng);
                if (j < 4) {
                    if (u < 0) continue;
                    for (int q = 0; q < 4; ++q)
                        if (std::find(v.begin(), v.end(), t.vertex[u][q]) == v.end()) e = t.vertex[u][q];
                }
                if (std::find(v.begin(), v.end(), e) != v.end()) continue;
                const oracle::Offset o = oracle::difference(g, v[0], e);
                const int s = oracle::in_sphere(a, b, d, o);
                const bool inside = exact_predicates::in_sphere(point, v.data(), e);
                ++checked;
                ties += s == 0;
                wrong += (s > 0 && !inside) || (s < 0 && inside);
                const oracle::i128 size = oracle::det(a, b, o);
                near += s != 0 && size < (oracle::i128(1) << 60) && size > -(oracle::i128(1) << 60);
            }
        }
        c.expect(wrong == 0, name + ": " + std::to_string(checked) + " in-sphere, " + std::to_string(ties) + " exact ties, " +
                                 std::to_string(near) + " with a small determinant: " + std::to_string(wrong) + " wrong");
    };
    add("predicates.sets", [=](Check& c) {
        for (const char* kind : {"jittered", "clustered", "halo", "lattice", "wave", "sc@0", "fcc@1e-6", "pairs@1e-6"})
            compare(c, kind, make_points(kind, 4096));
    });
    add("predicates.noise", [=](Check& c) {
        for (const int noise : {1, 3}) {
            Points p = exact_lattice(16, sc_basis);
            std::mt19937_64 rng(42);
            std::uniform_int_distribution<int> u(-noise, noise);
            for (auto& q : p)
                for (double& v : q) v = wrap(v + u(rng) * grid_step);
            compare(c, "lattice +-" + std::to_string(noise) + " steps", p);
        }
    });
    add("predicates.wide", [=](Check& c) {
        // Coordinates over the whole grid, tetrahedra up to a large part of
        // the box: a mesh of few particles, and the last grid positions.
        Points p = make_points("jittered", 512);
        for (auto& q : p)
            for (double& v : q) v = 1 - grid_step * (1 + std::floor((1 - v) * 3e6));  // the top 3e6 steps wrap into 0
        Points w = make_points("jittered", 512);
        compare(c, "512 jittered", w);
        Points tiny = make_points("uniform", 4096);
        for (auto& q : tiny)
            for (double& v : q) v = 0.5 + (v - 0.5) * 2048 * grid_step;  // a box of 2048 steps inside the big one
        tiny.push_back({0.1, 0.1, 0.1});
        for (int i = 1; i < 512; ++i) tiny.push_back({wrap(0.1 + std::sqrt(2.0) * i), wrap(0.2 + std::sqrt(3.0) * i), wrap(0.3 + std::sqrt(5.0) * i)});
        compare(c, "4096 particles on 2048^3 grid positions (no duplicates assumed)", tiny);
    });
}

// ------------------------------------------------------------- reproducibility

template <class E>
std::vector<std::uint64_t> trace(const Points& start, double displacement, int steps) {
    using Memory = typename E::memory_space;
    voronoi::Mesh<3, E> mesh(E{}, 1.0);
    Points p = start;
    const Flow flow;
    std::vector<std::uint64_t> hashes;
    auto put = [](std::uint64_t& h, std::uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    auto bits = [](double d) {
        std::uint64_t u;
        std::memcpy(&u, &d, sizeof u);
        return u;
    };
    Kokkos::View<double* [3], Memory> x("x", p.size());
    for (int s = 0; s <= steps; ++s) {
        auto h = Kokkos::create_mirror_view(x);
        for (std::size_t i = 0; i < p.size(); ++i)
            for (int a = 0; a < 3; ++a) h(i, a) = p[i][a];
        Kokkos::deep_copy(x, h);
        mesh.update(x);
        mesh.update_cells();
        const auto volume = to_host(mesh.volume());
        const auto centroid = to_host(mesh.centroid());
        const auto start_of = to_host(mesh.cell_face_start());
        const auto owner = to_host(mesh.owner());
        const auto neighbour = to_host(mesh.neighbour());
        const auto face = to_host(mesh.cell_face());
        std::uint64_t topology = 1469598103934665603ull, geometry = topology;
        for (std::size_t i = 0; i < p.size(); ++i) {
            put(geometry, bits(volume(i)));
            for (int a = 0; a < 3; ++a) put(geometry, bits(centroid(i, a)));
            for (int k = start_of(i); k < start_of(i + 1); ++k) {
                const int f = face(k);
                put(topology, static_cast<std::uint64_t>(owner(f) == static_cast<int>(i) ? neighbour(f) : owner(f)));
            }
        }
        hashes.push_back(topology);
        hashes.push_back(geometry);
        flow.advect(p, displacement / 16);
    }
    return hashes;
}

void reproducibility_cases() {
    const std::vector<std::pair<std::string, double>> runs{{"jittered", 0.3}, {"clustered", 0.3}, {"lattice", 0.1}, {"wave", 0.3}};
    for (const auto& [kind, displacement] : runs) {
        const std::string name = kind;
        const double d = displacement;
        add("repro.runs." + name, [=](Check& c) {
            const Points p = make_points(name, 4096);
            const auto a = trace<Exec>(p, d, 20), b = trace<Exec>(p, d, 20);
            c.expect(a == b, name + ": two runs give the same bits, 20 steps");
        });
#ifdef KOKKOS_ENABLE_SERIAL
        add("repro.serial." + name, [=](Check& c) {
            const Points p = make_points(name, 4096);
            const auto a = trace<Exec>(p, d, 20);
            const auto b = trace<Kokkos::Serial>(p, d, 20);
            long long topology = 0, geometry = 0;
            for (std::size_t i = 0; i < a.size(); i += 2) {
                topology += a[i] != b[i];
                geometry += a[i + 1] != b[i + 1];
            }
            c.expect(topology == 0, name + ": faces of " + Exec::name() + " and Serial agree (" + std::to_string(topology) + " of 21 steps differ)");
            c.expect(geometry == 0, name + ": bits of volumes and centroids agree (" + std::to_string(geometry) + " of 21 steps differ)");
        });
#endif
        add("repro.dump." + name, [=](Check& c) {
            // Hashes for a comparison across builds and thread counts:
            // MESH_REPRO_DUMP=<file> appends "<case> <step> <topology> <geometry>".
            const char* path = std::getenv("MESH_REPRO_DUMP");
            const auto a = trace<Exec>(make_points(name, 4096), d, 20);
            if (path) {
                std::ofstream out(path, std::ios::app);
                for (std::size_t i = 0; i < a.size(); i += 2)
                    out << name << " " << i / 2 << " " << a[i] << " " << a[i + 1] << "\n";
            }
            c.expect(true, name + ": hashes written");
        });
    }
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    coincident_cases();
    few_cases();
    contrast_cases();
    box_cases();
    degenerate_cases();
    valence_cases();
    large_cases();
    fallback_cases();
    predicate_cases();
    reproducibility_cases();
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
    // ctest runs a subset within its time budget; MESH_CASES_FULL=1 (the A100
    // job) runs the whole matrix, and a case asked for by its full name runs.
    const bool full = std::getenv("MESH_CASES_FULL") != nullptr;
    auto subset_skips = [&](const std::string& name) {
        static const std::vector<std::string> only_full{
            "box.void.centre.",     "box.void.empty.0.1",  "box.void.empty.0.22", "box.void.empty.0.25",
            "box.void.empty.0.3",   "box.void.empty.0.45", "box.slab.lattice.",   "box.slab.uniform.",
            "box.slab.jittered.7",  "box.slab.jittered.8", "box.thin.",           "contrast.ball.0.0",
            "contrast.ball.1e-05",  "contrast.ball.1e-06", "contrast.shell.",     "contrast.shell_centre.",
            "contrast.two_density.100",                    "contrast.two_density.1e+06",
            "contrast.grid_scale.1024",                    "contrast.grid_scale.4096",
            "contrast.grid_scale.16384",                   "contrast.grid_scale.65536",
            "contrast.sphere_status.1",                    "degenerate.noise.bcc.", "degenerate.noise.fcc.",
            "degenerate.noise.sc.16",                      "degenerate.noise.sc.1024",
            "degenerate.shear.smooth.", "degenerate.rigid.off_grid.", "large.random_walk.1", "large.flow.20",
            "fallback.large.", "fallback.growth.", "few.jittered.1", "few.lattice.1", "repro.dump."};
        if (full || std::find(wanted.begin(), wanted.end(), name) != wanted.end()) return false;
        return std::any_of(only_full.begin(), only_full.end(), [&](const std::string& p) { return name.rfind(p, 0) == 0; });
    };
    Check check;
    for (const auto& c : cases()) {
        if (subset_skips(c.name)) continue;
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
