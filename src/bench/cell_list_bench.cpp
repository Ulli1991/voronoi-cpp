// The cell list as a plain neighbour search (MI300A study, not part of the library): times neighbor_search::build and
// a gather over all particles within h_i of each particle i, h_i set by the cell list's own level so that about
// `ngb` particles fall within it (the density loop of an SPH/MFM code). One thread per slot, in slot order.
// args: cell_list_bench <n> <kind> [ngb=64] [repeats=5]
#include "support.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>

using namespace support;

namespace {

struct Result {
    long long candidates = 0, neighbours = 0;
    int fewest = 0, most = 0;
};

// Searches at `drop` levels coarser than each slot's own level; `prune` skips cells whose box is farther than h.
template <class Grid, class Counts, class Rho>
double search(const Exec& exec, const Grid& g, double c, int drop, bool prune, int repeats, const Counts& count,
              const Counts& tested, const Rho& rho) {
    const int n = g.n, depth = g.levels + neighbor_search::fine_levels;
    return median_ms(repeats, [&] {
        Kokkos::parallel_for("cell_list_bench::search", Kokkos::RangePolicy<Exec>(exec, 0, n), KOKKOS_LAMBDA(int q) {
            const double px = g.positions(q, 0), py = g.positions(q, 1), pz = g.positions(q, 2);
            const int own = g.level(q);
            const double h = c / double(1 << own), h2 = h * h;
            const int level = Kokkos::max(2, own - drop);
            const int side = 1 << level;
            const double w = 1.0 / side;
            const int reach = Kokkos::min(static_cast<int>(Kokkos::ceil(h / w)), (side - 1) / 2);
            const int hx = neighbor_search::grid_coordinate(px, 1.0, side);
            const int hy = neighbor_search::grid_coordinate(py, 1.0, side);
            const int hz = neighbor_search::grid_coordinate(pz, 1.0, side);
            int found = 0, seen = 0;
            double sum = 0;
            for (int dx = -reach; dx <= reach; ++dx) {
                const double gx = Kokkos::max(0, Kokkos::abs(dx) - 1) * w;
                for (int dy = -reach; dy <= reach; ++dy) {
                    const double gy = Kokkos::max(0, Kokkos::abs(dy) - 1) * w;
                    for (int dz = -reach; dz <= reach; ++dz) {
                        const double gz = Kokkos::max(0, Kokkos::abs(dz) - 1) * w;
                        if (prune && gx * gx + gy * gy + gz * gz >= h2) continue;
                        const int x = (hx + dx + side) & (side - 1), y = (hy + dy + side) & (side - 1),
                                  z = (hz + dz + side) & (side - 1);
                        int begin, end;
                        g.template range<true>(level, x, y, z, begin, end);
                        seen += end - begin;
                        for (int s = begin; s < end; ++s) {
                            double ax = g.positions(s, 0) - px, ay = g.positions(s, 1) - py,
                                   az = g.positions(s, 2) - pz;
                            ax -= Kokkos::round(ax), ay -= Kokkos::round(ay), az -= Kokkos::round(az);
                            const double d2 = ax * ax + ay * ay + az * az;
                            if (d2 < h2) {
                                const double u = 1 - d2 / h2;
                                sum += u * u * u;
                                ++found;
                            }
                        }
                    }
                }
            }
            (void)depth;
            count(q) = found;
            tested(q) = seen;
            rho(q) = sum;
        });
    });
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    if (argc < 3) {
        std::cerr << "usage: cell_list_bench <n> <kind> [ngb=64] [repeats=5]\n";
        return 2;
    }
    const int requested = std::stoi(argv[1]);
    const std::string kind = argv[2];
    const int ngb = argc > 3 ? std::stoi(argv[3]) : 64;
    const int repeats = argc > 4 ? std::stoi(argv[4]) : 5;
    const Exec exec;
    const Box box;
    const Positions x = to_device(make_points(kind, requested, box));
    const int n = static_cast<int>(x.extent(0));

    neighbor_search::CellList<Exec> list;
    neighbor_search::configure(list, n, box, std::cbrt(1.0 / n));
    const double build_ms = median_ms(repeats, [&] { neighbor_search::build(exec, list, x); });
    const auto g = shell_voronoi::make_grid(list);
    const auto level_h = to_host(list.level);
    int lo = 99, hi = 0;
    for (int q = 0; q < n; ++q) lo = std::min<int>(lo, level_h(q)), hi = std::max<int>(hi, level_h(q));
    std::cout << "# backend=" << Exec::name() << " n=" << n << " kind=" << kind << " ngb=" << ngb
              << " base_levels=" << list.levels << " slot_levels=" << lo << ".." << hi << "\n";
    std::cout << "build: ms=" << build_ms << " ns_per_particle=" << build_ms * 1e6 / n
              << " Mparticles_s=" << n / build_ms / 1e3 << "\n";

    // h = c / 2^level: the level's cells hold about one particle, so a ball of radius c cells holds about ngb.
    const double c = std::cbrt(3.0 * ngb / (4.0 * M_PI));
    Kokkos::View<int*, Memory> count("count", n), tested("tested", n);
    Kokkos::View<double*, Memory> rho("rho", n);
    int failures = 0;
    for (const bool prune : {false, true})
        for (int drop = 0; drop <= 2; ++drop) {
            const double ms = search(exec, g, c, drop, prune, repeats, count, tested, rho);
            const auto count_h = to_host(count);
            const auto tested_h = to_host(tested);
            Result r;
            r.fewest = 1 << 30;
            for (int q = 0; q < n; ++q) {
                r.candidates += tested_h(q);
                r.neighbours += count_h(q);
                r.fewest = std::min(r.fewest, count_h(q));
                r.most = std::max(r.most, count_h(q));
            }
            std::cout << "search: drop=" << drop << " prune=" << prune << " ms=" << ms
                      << " ns_per_particle=" << ms * 1e6 / n << " Mparticles_s=" << n / ms / 1e3
                      << " candidates_per_particle=" << double(r.candidates) / n
                      << " neighbours_per_particle=" << double(r.neighbours) / n << " min=" << r.fewest
                      << " max=" << r.most << " hit_rate=" << double(r.neighbours) / r.candidates << "\n";
            if (drop == 0 && prune) {
                // Brute force on 512 sampled slots: the same count within the same h.
                const auto pos = to_host(g.positions);
                std::mt19937_64 rng(5);
                std::uniform_int_distribution<int> any(0, n - 1);
                int wrong = 0;
                for (int k = 0; k < 512; ++k) {
                    const int q = any(rng);
                    const double h = c / double(1 << level_h(q));
                    // A ball wider than the walk's capped reach is not comparable.
                    const int side = 1 << std::max(2, int(level_h(q)));
                    if (std::ceil(h * side) > (side - 1) / 2) continue;
                    int brute = 0;
                    for (int s = 0; s < n; ++s) {
                        double d2 = 0;
                        for (int a = 0; a < 3; ++a) {
                            double d = pos(s, a) - pos(q, a);
                            d -= std::round(d);
                            d2 += d * d;
                        }
                        brute += d2 < h * h;
                    }
                    wrong += brute != count_h(q);
                }
                std::cout << "verify: sampled=512 wrong=" << wrong << (wrong ? " FAIL" : " PASS") << "\n";
                failures += wrong;
            }
        }
    return failures ? 1 : 0;
}
