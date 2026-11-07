// mesh_cases_test group `large`: registers its cases with add(); shared helpers in common.hpp.

#include "mesh_cases/common.hpp"

namespace mesh_cases {

// --------------------------------------------------------------- large motion

// Large steps and kicks, permuted or resized input, refused updates, long runs and cycles.
void large_cases() {
    for (const double d : {0.5, 1.0, 2.0, 5.0, 20.0})
        add("large.flow." + sci(d), [=](Check& c) {
            Points p = make_points("jittered", 4096);
            Mesh m(Exec{}, 1.0);
            build(c, "start", m, p, Expect::exact);
            const Flow flow;
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
    // Kicks as a fraction of the box, several seeds: the mesh is exact, or the update refuses.
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
        // Thousands of small steps of a solenoidal flow (no void ends it): no drift in faces, tetrahedra, slots, ring
        // pools, memory.
        using namespace distortion;
        DeviceBytes::start();
        const char* const long_run = std::getenv("MESH_LONG_RUN");
        const int steps = long_run ? std::atoi(long_run) : 1000;
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

}  // namespace mesh_cases
