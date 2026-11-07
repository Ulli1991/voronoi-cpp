// mesh_cases_test group `few`: registers its cases with add(); shared helpers in common.hpp.

#include "mesh_cases/common.hpp"

namespace mesh_cases {

// ----------------------------------------------------------------- few points

// Zero to a few particles, and a mesh that goes back to none.
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

}  // namespace mesh_cases
