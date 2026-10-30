// Full rebuild by clipping: on jittered, uniform random, lattice and
// clustered points every cell is certified, volumes are positive and fill
// the box, cells close, faces are consistent between neighbours and lie on
// the bisectors (support::check_cells). The thread and warp kernels agree,
// and Morton-ordered cells give the same cells.

#include "check.hpp"
#include "support.hpp"

#include <sstream>

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    using namespace support;
    Check check;
    const Exec exec;
    const int n = 4096;
    for (const std::string kind : {"jittered", "uniform", "lattice", "clustered"}) {
        std::cout << kind << ":\n";
        const Positions x = to_device(make_points(kind, n));
        const auto list = cell_list(x);
        const auto grid = shell_voronoi::make_grid(list);

        shell_voronoi::Cells<Memory> thread(n, true), warp(n, true);
        shell_voronoi::construct<double>(exec, grid, thread);
        shell_voronoi::construct_warp<double>(exec, grid, warp);
        for (const auto& [name, cells] : {std::pair{"thread", thread}, std::pair{"warp", warp}}) {
            std::ostringstream summary;
            const bool valid = check_cells(cells, list.positions, true, summary) == 0;
            std::string details = summary.str();
            details.pop_back();  // the newline
            check.expect(valid, std::string(name) + " kernel cells valid\n" + details);
        }
        const auto v_thread = to_host(thread.volume);
        const auto v_warp = to_host(warp.volume);
        double worst = 0;
        for (int t = 0; t < n; ++t) worst = std::max(worst, std::abs(v_warp(t) - v_thread(t)) / v_thread(t));
        check.expect(worst < 1e-12, "warp and thread volumes agree (" + sci(worst) + ")");

        // Morton cell order: other slots, the same cells.
        const auto morton = cell_list(x, 1.0, true);
        shell_voronoi::Cells<Memory> cells(n, false);
        shell_voronoi::construct_warp<double>(exec, shell_voronoi::make_grid(morton), cells);
        const auto order = to_host(list.order);
        const auto morton_order = to_host(morton.order);
        const auto v_morton = to_host(cells.volume);
        std::vector<double> by_particle(n);
        for (int t = 0; t < n; ++t) by_particle[order(t)] = v_thread(t);
        worst = 0;
        for (int t = 0; t < n; ++t)
            worst = std::max(worst, std::abs(v_morton(t) - by_particle[morton_order(t)]) / v_morton(t));
        check.expect(worst < 1e-12, "Morton cell order gives the same volumes (" + sci(worst) + ")");
    }
    return check.failed() != 0;
}
