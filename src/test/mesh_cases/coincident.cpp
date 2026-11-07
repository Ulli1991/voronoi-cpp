// mesh_cases_test group `coincident`: registers its cases with add(); shared helpers in common.hpp.

#include "mesh_cases/common.hpp"

namespace mesh_cases {

// ---------------------------------------------------------------- coincident

// Particles on one grid position (duplicates): refused at the start, handled or refused during steps, messages.
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
            p[7][0] += 0.25 * grid_step;
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
            if (s > steps + stay)
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
    // Each fallback, a pair and a few pairs landing in one step, then what a duplicate does to the next steps.
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
                    for (auto& q : p) q[0] = wrap(q[0] + 1e-3);
                    step_or_refused(c, "the pairs move on, step " + std::to_string(s), m, p, refused);
                }
            });
        }
    for (const int n : {8000, 9261, 12167, 32768, 262144})
        add("coincident.step.default." + std::to_string(n), [=](Check& c) {
            // The default fallback goes to the host only when stuck * 200000 <= n.
            Points p = make_points("jittered", n);
            Mesh m(Exec{}, 1.0);
            build(c, "start", m, p, Expect::exact, 4);
            p[50] = p[51];
            bool refused = false;
            step_or_refused(c, "a pair lands on one position, default fallback, n=" + std::to_string(n), m, p, refused, 4);
            c.expect(refused, "refused");
        });
    // The refusal says what is wrong: the pair, by id.
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

}  // namespace mesh_cases
