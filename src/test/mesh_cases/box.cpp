// mesh_cases_test group `box`: registers its cases with add(); shared helpers in common.hpp.

#include "mesh_cases/common.hpp"

namespace mesh_cases {

// ---------------------------------------------------------------------- boxes

// Boxes: thin slabs, long boxes, voids, positions on the boundary, snapping twice, non-finite input.
void box_cases() {
    // Slabs: m cells across a side of 1/8.
    for (const int across : {1, 2, 3, 4, 5, 6, 7, 8})
        for (const char* kind : {"jittered", "lattice", "uniform"})
            add("box.slab." + std::string(kind) + "." + std::to_string(across), [=](Check& c) {
                const Box box(1, 1, 0.125);
                const int side = 8 * across;
                const Points p = make_points(kind, side * side * across, box);
                Mesh m(Exec{}, box);
                build(c, std::string(kind) + " in a 1:1:1/8 slab, about " + std::to_string(across) + " cells across", m,
                      p, Expect::exact_or_refused, 32);
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
    // One void in a uniform background, its centre empty or with one particle: the cells around it reach further with
    // the radius.
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
        const std::vector<Point> special{
            {0, 0, 0},         {1, 1, 0.5},         {below, 0.5, 0.5}, {-grid_step, 0.5, 0.25},
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
            Points q = p;
            for (auto& a : q) a[0] += 0.3 * s;
            step(c, "unwrapped positions, step " + std::to_string(s), m, q);
        }
        Points on = p;
        for (std::size_t i = 0; i < on.size(); i += 50) on[i][2] = 1.0;  // on the far side exactly
        step(c, "particles on x = side", m, on);
    });
    for (const double scale : {1e-9, 1e-3, 3.0, 1e3, 1e9})
        add("box.scale." + sci(scale), [=](Check& c) {
            // The same points in a cube of any side: the same grid positions, the same tetrahedra, volumes scaled.
            const Points p = make_points("clustered", 4096);
            Mesh unit(Exec{}, 1.0), other(Exec{}, scale);
            unit.update(to_device(p));
            Points q = p;
            for (auto& a : q)
                for (double& v : a) v = (v + 0.5 * grid_step) * scale;
            other.update(to_device(q));
            const long long differ = oracle::different_tetrahedra(mc::tets_of(unit), mc::tets_of(other));
            c.expect(differ == 0,
                     "box " + sci(scale) + ": the tetrahedra of the unit box (" + std::to_string(differ) + " differ)");
            const auto a = to_host(unit.volume()), b = to_host(other.volume());
            double worst = 0;
            for (std::size_t i = 0; i < p.size(); ++i)
                worst = std::max(worst, std::abs(b(i) / (scale * scale * scale) / a(i) - 1));
            c.expect(worst < 1e-12, "volumes scale with the box (" + sci(worst) + ")");
            const mc::Result r = mc::check(other);
            c.expect(r.ok(), r.text());
        });
    for (const double side : {1.0, 3.0, 0.7, 0.1, 1e-3, 1.0 / 3, 3.086e24})
        add(
            "box.snap_twice." + sci(side),
            [=](Check& c) {
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
                c.expect(moved == 0, "side " + sci(side) + ": " + std::to_string(moved) + " of " +
                                         std::to_string(3 * p.size()) + " coordinates move when snapped twice");
                // On a lattice a step decides ties: the mesh of the positions the caller gave, by the oracle (no tie-break,
                // every image).
                Points lattice = scaled(exact_lattice(16, {{0.5, 0.5, 0.5}}), box);
                for (auto& q : lattice)
                    for (int a = 0; a < 3; ++a) q[a] = (exact_predicates::snap(q[a], box, a) + 0.5) * box.step;
                Mesh m(Exec{}, box);
                m.update(to_device(lattice));
                const auto r = oracle::check(mc::grid_of(m), mc::tets_of(m));
                c.expect(r.ok(), "the lattice in that box: " + r.text());
            },
            side > 1e20 ? on_cuda(26) : 0);
    for (const std::string what : {"nan", "inf", "huge"})
        add("box.nonfinite." + what, [=](Check& c) {
            const Points p = make_points("jittered", 4096);
            const double bad = what == "nan"   ? std::nan("")
                               : what == "inf" ? std::numeric_limits<double>::infinity()
                                               : 1e300;
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

}  // namespace mesh_cases
