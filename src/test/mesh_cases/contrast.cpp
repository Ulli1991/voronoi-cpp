// mesh_cases_test group `contrast`: registers its cases with add(); shared helpers in common.hpp.

#include "mesh_cases/common.hpp"

namespace mesh_cases {

// ------------------------------------------------------------------- contrast

// Extreme density contrasts: balls, shells, clumps, lines, planes, skew segments; start statuses and refusals.
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
            build(c,
                  "3072 particles on " + std::to_string(steps) + "^3 grid positions, " + std::to_string(same) +
                      " duplicates",
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
    // Half the particles in a ball of radius 0.05: fine at small n, the start of the `expansion` flow at 128^3. Too
    // slow for ctest: not in a group it runs.
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
        // What each status of the clipping is reported as.
        // Made by hand: a cell of more planes than the exact clipping holds costs minutes to reach.
        const Positions x = to_device(make_points("jittered", 64));
        tet_mesh::Points4<Memory> snapped("snapped", 64);
        Kokkos::parallel_for(
            "snapped", Kokkos::RangePolicy<Exec>(0, 64), KOKKOS_LAMBDA(int i) {
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
        for (int i = 1; i < 512; ++i)
            line.push_back(
                {wrap(0.1 + std::sqrt(2.0) * i), wrap(0.2 + std::sqrt(3.0) * i), wrap(0.3 + std::sqrt(5.0) * i)});
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

}  // namespace mesh_cases
