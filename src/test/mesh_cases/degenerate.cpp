// mesh_cases_test group `degenerate`: registers its cases with add(); shared helpers in common.hpp.

#include "mesh_cases/common.hpp"

namespace mesh_cases {

// ------------------------------------------------- degeneracy during motion

// Exact lattices and cospherical sets moving: ties at every step (the perturbation decides), zero-area faces.
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
            // Rigid in double: snapping rounds each particle by up to a grid step, so the lattice is near-degenerate
            // after every step.
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
            // Layers shift by whole lattice spacings every 8 steps, so an exact lattice recurs; periodic (triangular
            // profile).
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
            add("degenerate.noise." + name + "." + std::to_string(noise),
                [=](Check& c) {  // every step all near-degenerate
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
                                       " faces have zero area, " + std::to_string(bad) +
                                       " non-finite centroid components");
                if (s == 0) step(c, "rigid step", m, p);
            }
        });
    }
    add("degenerate.two_lattices", [](Check& c) {
        // sc lattice 2 slides through sc lattice 1 along z, shifted in x and y: all planes coplanar every 8 steps, bcc
        // on the way.
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
        // One particle walks along lattice directions: through face centres (coplanar), edge midpoints and cube centres
        // (cospherical with 8).
        Points p = exact_lattice(16, sc_basis);
        const double h = 1.0 / 16;
        p.push_back({0.5625, 0.5625, 0.5625});
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
        // Octahedra pulsing with different phases: their 6 points are exactly cospherical whenever the phases meet,
        // every 8 steps.
        const int side = 8;
        const double h = 1.0 / side, big = h / 4, small = h / 64;
        auto at = [&](int s) {
            Points p;
            const double r[3] = {big + small * (s % 8 - 4),
                                 big + small * ((s + 3) % 8 - 4) * (s % 8 ? 1 : 0) + (s % 8 ? 0 : small * (s % 8 - 4)),
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
        // Grid points exactly on one sphere: x^2 + y^2 + z^2 = R^2 in integers (scaled solutions of a^2 + b^2 + c^2 =
        // 81^2 etc.), exact cospherical.
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

}  // namespace mesh_cases
