// mesh_cases_test group `valence`: registers its cases with add(); shared helpers in common.hpp.

#include "mesh_cases/common.hpp"

namespace mesh_cases {

// ----------------------------------------------------------- high valence

// `k` particles on a circle of radius r around the z axis through `at`, and
// two poles at +-h on the axis; particles of a sparse background elsewhere. Used by: valence_cases.
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

// High-valence edges and faces (k tetrahedra around one edge): starts, steps, spheres, hubs.
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

}  // namespace mesh_cases
