// Full rebuild by clipping on jittered, random, lattice, clustered and halo points and the surface input of #60:
// certified cells that fill the box with consistent faces; thread, warp and exact kernels agree; volumes_and_centroids
// gives the full rebuild's bits. Exercises shell_voronoi::construct / construct_warp / volumes_and_centroids.

#include "check.hpp"
#include "clipping_cases.hpp"
#include "support.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <type_traits>

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

        // Thread (construct) and warp (construct_warp) kernels, each validated on its own by support::check_cells.
        shell_voronoi::Cells<Memory> thread(n, true), warp(n, true);
        shell_voronoi::construct(exec, grid, thread);
        shell_voronoi::construct_warp(exec, grid, warp);
        for (const auto& [name, cells] : {std::pair{"thread", thread}, std::pair{"warp", warp}}) {
            std::ostringstream summary;
            const bool valid = check_cells(cells, x, summary) == 0;
            std::string details = summary.str();
            details.pop_back();
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

    // #60: rounded clipping refused cell 255 of these surface points (OpenMP); every kernel now certifies every cell, and
    // the cells agree with the exact kernel's.
    {
        Points points;
        for (const auto& k : clipping_cases::surface_953)
            points.push_back({std::ldexp(k[0], -30), std::ldexp(k[1], -30), std::ldexp(k[2], -30)});
        const int m = static_cast<int>(points.size());
        const Positions x = to_device(points);
        const auto grid = shell_voronoi::make_grid(cell_list(x));
        shell_voronoi::Cells<Memory> thread(m, true), warp(m, true), exact(m, false);
        shell_voronoi::construct(exec, grid, thread);
        shell_voronoi::construct_warp(exec, grid, warp);
        shell_voronoi::construct(exec, grid, exact, {}, true);
        const Reference reference = by_particle(exact);
        for (const auto& [name, cells] : {std::pair{"thread", thread}, std::pair{"warp", warp}}) {
            std::ostringstream summary;
            const bool valid = check_cells(cells, x, summary) == 0;
            std::string details = summary.str();
            details.pop_back();
            const Difference d = compare(reference, by_particle(cells));
            check.expect(valid && d.neighbours == 0 && d.volume < 1e-12,
                         std::string("#60 surface case: ") + name +
                             " kernel certifies every cell and agrees with the exact (" + std::to_string(d.neighbours) +
                             " neighbour sets, volumes " + sci(d.volume) + ")\n" + details);
        }
    }

    // #60 at full size: the surface input, 64³ on a GPU (32³ on the host, in CI time), every cell certified and checked.
    {
        const bool device = !std::is_same_v<Exec, Kokkos::DefaultHostExecutionSpace>;
        const int m = device ? 262144 : 32768;
        const Positions x = to_device(make_points("surface", m));
        shell_voronoi::Cells<Memory> cells(m, true);
        shell_voronoi::construct_warp(exec, shell_voronoi::make_grid(cell_list(x)), cells);
        std::ostringstream summary;
        const bool valid = check_cells(cells, x, summary) == 0;
        std::string details = summary.str();
        details.pop_back();
        check.expect(valid,
                     "#60 surface input, " + std::to_string(m) + " points: every cell certified and valid\n" + details);
    }

    // Non-finite positions are never a candidate, so every other cell is the cell without them: the same neighbours and
    // volume (to the last bit unless the extra particles change the walk's grid levels nearby).
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
    // Volumes and centroids only (volumes_and_centroids): the bits of a full rebuild's and a rings build's volumes and
    // centroids, so all three route each cell alike (#60).
    for (const std::string kind :
         {"jittered", "uniform", "clustered", "halo", "lattice", "wave", "cosmicweb", "fccsolid", "surface"}) {
        const int m = 32 * 32 * 32;
        const Positions x = to_device(make_points(kind, m));
        neighbor_search::CellList<Exec> list;
        neighbor_search::configure(list, m, Box{}, std::cbrt(1.0 / m));
        neighbor_search::build(exec, list, x);
        shell_voronoi::Cells<Memory> full(m, false);
        shell_voronoi::construct_warp(exec, shell_voronoi::make_grid(list), full);
        Kokkos::View<double*, Memory> volume("volume", m);
        Kokkos::View<double* [3], Memory> centroid("centroid", m);
        shell_voronoi::volumes_and_centroids(exec, Box{}, x, volume, centroid);
        auto ringed = shell_voronoi::Cells<Memory>::with_rings(m, 64LL * m);
        shell_voronoi::construct_warp(exec, shell_voronoi::make_grid(list), ringed);
        const auto v0 = to_host(full.volume), v1 = to_host(volume), v2 = to_host(ringed.volume);
        const auto c0 = to_host(full.centroid), c1 = to_host(centroid), c2 = to_host(ringed.centroid);
        int differ = 0, differ_rings = 0;
        for (int i = 0; i < m; ++i) {
            differ += v0(i) != v1(i) || c0(i, 0) != c1(i, 0) || c0(i, 1) != c1(i, 1) || c0(i, 2) != c1(i, 2);
            differ_rings += v0(i) != v2(i) || c0(i, 0) != c2(i, 0) || c0(i, 1) != c2(i, 1) || c0(i, 2) != c2(i, 2);
        }
        check.expect(differ == 0, kind + ": volumes and centroids only, bits of the full rebuild (" +
                                      std::to_string(differ) + " of 32768 cells differ)");
        check.expect(differ_rings == 0, kind + ": rings build, bits of the full rebuild's volumes and centroids (" +
                                            std::to_string(differ_rings) + " of 32768 cells differ)");
    }
    return check.failed() != 0;
}
