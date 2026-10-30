// Full rebuild by clipping: on jittered, uniform random, lattice, clustered
// and halo points every cell is certified, volumes are positive and fill the
// box, cells close, faces are consistent between neighbours and lie on the
// bisectors (support::check_cells). The thread, warp and exact kernels give
// the same cells, and so does a cell list with a coarser base grid (other
// slots). A face pool too small grows to what the build needs. Particles
// at NaN or infinity never cut another cell.

#include "check.hpp"
#include "support.hpp"

#include <cmath>
#include <limits>
#include <sstream>

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    using namespace support;
    Check check;
    const Exec exec;
    const int n = 4096;
    for (const std::string kind : {"jittered", "uniform", "lattice", "clustered", "halo"}) {
        std::cout << kind << ":\n";
        const Positions x = to_device(make_points(kind, n));
        const auto list = cell_list(x);
        const auto grid = shell_voronoi::make_grid(list);

        shell_voronoi::Cells<Memory> thread(n, true), warp(n, true);
        shell_voronoi::construct(exec, grid, thread);
        shell_voronoi::construct_warp(exec, grid, warp);
        for (const auto& [name, cells] : {std::pair{"thread", thread}, std::pair{"warp", warp}}) {
            std::ostringstream summary;
            const bool valid = check_cells(cells, x, summary) == 0;
            std::string details = summary.str();
            details.pop_back();  // the newline
            check.expect(valid, std::string(name) + " kernel cells valid\n" + details);
        }
        const Reference reference = by_particle(thread);
        const Difference d = compare(reference, by_particle(warp));
        auto agree = [](const Difference& x) {
            return " (" + std::to_string(x.neighbours) + " neighbour sets, volumes " + sci(x.volume) + ")";
        };
        check.expect(d.neighbours == 0 && d.volume < 1e-12, "warp and thread cells agree" + agree(d));

        // The exact per-thread kernel (the start's patch) gives the same cells.
        shell_voronoi::Cells<Memory> exact(n, false);
        shell_voronoi::construct(exec, grid, exact, {}, true);
        const Difference e = compare(reference, by_particle(exact));
        check.expect(e.neighbours == 0 && e.volume < 1e-12, "exact and thread cells agree" + agree(e));

        // A base grid twice as coarse: other slots, the same cells.
        const auto coarse = cell_list(x, 1.0, 2.0);
        shell_voronoi::Cells<Memory> cells(n, false);
        shell_voronoi::construct_warp(exec, shell_voronoi::make_grid(coarse), cells);
        const Difference c = compare(reference, by_particle(cells));
        check.expect(c.neighbours == 0 && c.volume < 1e-12, "coarser base grid gives the same cells" + agree(c));

        // A face pool for one face of one cell grows to what the build needs.
        shell_voronoi::Cells<Memory> small(n, false, 1, 1);
        shell_voronoi::construct_warp(exec, grid, small);
        const Difference w = compare(reference, by_particle(small));
        check.expect(small.capacity() > n && w.neighbours == 0 && w.volume < 1e-12,
                     "a small face pool grows to " + std::to_string(double(small.capacity()) / n) +
                         " faces per cell, same cells");
    }

    // Non-finite positions: a particle at NaN and one at infinity are never
    // a candidate (their distances never fall below the reach), so every
    // other cell is the cell without them: the same neighbours, the same
    // volume (to the last bit unless the two extra particles change the
    // walk's grid levels nearby, which reorders the cuts). (Test idea from
    // Doug Rennehan, PR #2.)
    {
        Points points = make_points("uniform", n);
        const Positions clean = to_device(points);
        points.push_back({std::nan(""), 0.5, 0.5});
        points.push_back({std::numeric_limits<double>::infinity(), 0.5, 0.5});
        const Positions dirty = to_device(points);
        for (const bool warp : {false, true}) {
            auto build = [&](const Positions& x) {
                shell_voronoi::Cells<Memory> cells(static_cast<int>(x.extent(0)), false);
                const auto list = cell_list(x);
                if (warp)
                    shell_voronoi::construct_warp(exec, shell_voronoi::make_grid(list), cells);
                else
                    shell_voronoi::construct(exec, shell_voronoi::make_grid(list), cells);
                return by_particle(cells);
            };
            const Reference a = build(clean), b = build(dirty);
            long long differ = 0;
            for (int i = 0; i < n; ++i)
                differ += a.status[i] != b.status[i] || a.neighbours[i] != b.neighbours[i] ||
                          std::abs(a.volume[i] - b.volume[i]) > 1e-12 * a.volume[i];
            check.expect(differ == 0, std::string(warp ? "warp" : "thread") +
                                          " kernel: NaN and infinite particles are never candidates (" +
                                          std::to_string(differ) + " cells differ)");
        }
    }
    return check.failed() != 0;
}
