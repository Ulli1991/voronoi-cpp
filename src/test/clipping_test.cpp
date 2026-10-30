// Full rebuild by clipping: on jittered, uniform random, lattice, clustered
// and halo points every cell is certified, volumes are positive and fill the
// box, cells close, faces are consistent between neighbours and lie on the
// bisectors (support::check_cells). The thread and warp kernels give the same
// cells, and so does a cell list with a coarser base grid (other slots). A
// face pool too small grows to what the build needs.

#include "check.hpp"
#include "support.hpp"

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

        // A base grid twice as coarse: other slots, the same cells.
        const auto coarse = cell_list(x, 1.0, 2.0);
        shell_voronoi::Cells<Memory> cells(n, false);
        shell_voronoi::construct_warp(exec, shell_voronoi::make_grid(coarse), cells);
        const Difference c = compare(reference, by_particle(cells));
        check.expect(c.neighbours == 0 && c.volume < 1e-12, "coarser base grid gives the same cells" + agree(c));

        // A face pool for one face per cell grows to what the build needs.
        shell_voronoi::Cells<Memory> small(n, false, 1);
        shell_voronoi::construct_warp(exec, grid, small);
        const Difference w = compare(reference, by_particle(small));
        check.expect(small.capacity() > n && w.neighbours == 0 && w.volume < 1e-12,
                     "a small face pool grows to " + std::to_string(double(small.capacity()) / n) +
                         " faces per cell, same cells");
    }
    return check.failed() != 0;
}
