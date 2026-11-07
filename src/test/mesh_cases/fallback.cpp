// mesh_cases_test group `fallback`: registers its cases with add(); shared helpers in common.hpp.

#include "mesh_cases/common.hpp"

namespace mesh_cases {

// ------------------------------------------------------------------ fallbacks

// Where repair gives up: stuck triangles, host and rebuild fallbacks, pool growth, their cost.
void fallback_cases() {
    add("fallback.cheapest.small_n", [](Check& c) {
        // The cheapest fallback at small n is a rebuild: the host costs up to 50 ms per stuck entry, a rebuild of 32768
        // cells far less than one.
        int stuck_steps = 0;
        for (const double d : {2.0, 4.0})  // a few particles jump: a few stuck entries
            for (int seed = 0; seed < 2; ++seed) {
                Points p = make_points("jittered", 32768);
                Mesh m(Exec{}, 1.0);
                build(c, "start", m, p, Expect::exact, 2);
                std::mt19937_64 rng(100 + seed);
                std::normal_distribution<double> g(0, d / 32);
                std::uniform_int_distribution<int> any(0, 32767);
                for (int k = 0; k < 2; ++k) {
                    auto& q = p[any(rng)];
                    for (double& v : q) v = wrap(v + g(rng));
                }
                to_grid(p);
                const auto r = step(c, "kicks of " + sci(d) + " spacings", m, p, 2);
                stuck_steps += r.stuck > 0;
                c.expect(r.host_flips == 0 && r.host_rebuilds == 0, "no host work");
            }
        c.expect(stuck_steps > 0, std::to_string(stuck_steps) + " steps with stuck entries");
    });
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
            // From a lattice (few tetrahedra) to two skew segments in one step: more tetrahedra, faces and ring entries
            // than the build sized anything for.
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

}  // namespace mesh_cases
