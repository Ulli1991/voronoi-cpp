// mesh_cases_test group `predicate`: registers its cases with add(); shared helpers in common.hpp.

#include "mesh_cases/common.hpp"

namespace mesh_cases {

// -------------------------------------------------- predicates on real sets

// The library predicates on real tetrahedron/apex sets against the oracle's integers.
void predicate_cases() {
    // Library predicates on the quintuples a mesh really asks about, against the oracle's integers: each tetrahedron
    // with the apex across each face.
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
        c.expect(wrong == 0, name + ": " + std::to_string(checked) + " in-sphere, " + std::to_string(ties) +
                                 " exact ties, " + std::to_string(near) +
                                 " with a small determinant: " + std::to_string(wrong) + " wrong");
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
        // Coordinates over the whole grid, tetrahedra up to a large part of the box: a mesh of few particles, and the
        // last grid positions.
        Points p = make_points("jittered", 512);
        for (auto& q : p)
            for (double& v : q) v = 1 - grid_step * (1 + std::floor((1 - v) * 3e6));
        Points w = make_points("jittered", 512);
        compare(c, "512 jittered", w);
        Points tiny = make_points("uniform", 4096);
        for (auto& q : tiny)
            for (double& v : q) v = 0.5 + (v - 0.5) * 2048 * grid_step;
        tiny.push_back({0.1, 0.1, 0.1});
        for (int i = 1; i < 512; ++i)
            tiny.push_back(
                {wrap(0.1 + std::sqrt(2.0) * i), wrap(0.2 + std::sqrt(3.0) * i), wrap(0.3 + std::sqrt(5.0) * i)});
        compare(c, "4096 particles on 2048^3 grid positions (no duplicates assumed)", tiny);
    });
}

}  // namespace mesh_cases
