// Bench: the 2D mesh timed (a build from nothing and moving steps, on one device, in M cells/s); no hydro.
// Flow: make_points -> Mesh<2>::rebuild + update_cells (build) -> drift + update + update_cells (steps).

#include "device_memory.hpp"
#include "voronoi_mesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Memory = Exec::memory_space;
using Mesh = voronoi::Mesh<2, Exec>;
using Box = exact_predicates::BoxN<2>;
using X = Kokkos::View<double* [2], Memory>;

// Value of `--name=value` among the flags after the two positionals, or fallback. Used by: main.
std::string flag(int argc, char** argv, const char* name, const char* fallback) {
    const std::size_t length = std::strlen(name);
    for (int a = 3; a < argc; ++a)
        if (std::strncmp(argv[a], name, length) == 0 && argv[a][length] == '=') return argv[a] + length + 1;
    return fallback;
}

// Interleaves the zero bits of the low 32 bits of v (a Morton coordinate). Used by: make_points (sorted).
std::uint64_t spread(std::uint64_t v) {
    v = (v | v << 16) & 0x0000ffff0000ffffull;
    v = (v | v << 8) & 0x00ff00ff00ff00ffull;
    v = (v | v << 4) & 0x0f0f0f0f0f0f0f0full;
    v = (v | v << 2) & 0x3333333333333333ull;
    return (v | v << 1) & 0x5555555555555555ull;
}

// n points of the given kind (jittered, random, clustered, lattice) in the box, on the device; n is rounded to a
// lattice. sorted: in Morton order so neighbouring cells sit close in memory. Used by: main.
X make_points(const std::string& kind, int n, const Box& box, bool sorted) {
    X x("x", n);
    auto h = Kokkos::create_mirror_view(x);
    std::mt19937_64 rng(20261105);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double> blob(0.0, 0.04);
    const double lx = box.side[0], ly = box.side[1];
    const int nx = static_cast<int>(std::lround(std::sqrt(n * lx / ly))), ny = n / nx;
    double centres[8][2];
    for (auto& c : centres) {
        c[0] = unit(rng);
        c[1] = unit(rng);
    }
    for (int i = 0; i < n; ++i) {
        double px, py;
        if (kind == "random") {
            px = unit(rng);
            py = unit(rng);
        } else if (kind == "clustered") {
            const bool background = i % 5 == 0;
            px = background ? unit(rng) : centres[i % 8][0] + blob(rng);
            py = background ? unit(rng) : centres[i % 8][1] + blob(rng);
        } else {
            const int ix = i / ny, iy = i % ny;
            const double jitter = kind == "jittered" ? 0.1 : 0.0;
            px = (ix + 0.5 + jitter * (2 * unit(rng) - 1)) / nx;
            py = (iy + 0.5 + jitter * (2 * unit(rng) - 1)) / ny;
        }
        h(i, 0) = (px - std::floor(px)) * lx;
        h(i, 1) = (py - std::floor(py)) * ly;
    }
    if (sorted) {
        std::vector<std::pair<std::uint64_t, int>> keys(n);
        for (int i = 0; i < n; ++i)
            keys[i] = {spread(static_cast<std::uint64_t>(h(i, 0) / lx * 65536)) << 1 | spread(static_cast<std::uint64_t>(h(i, 1) / ly * 65536)), i};
        std::sort(keys.begin(), keys.end());
        auto sorted_points = Kokkos::create_mirror_view(x);
        for (int i = 0; i < n; ++i)
            for (int a = 0; a < 2; ++a) sorted_points(i, a) = h(keys[i].second, a);
        Kokkos::deep_copy(x, sorted_points);
        return x;
    }
    Kokkos::deep_copy(x, h);
    return x;
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    if (argc < 3) {
        std::puts("usage: mesh2d_bench <jittered|random|clustered|lattice> <n> [--steps=k] [--move=s] [--repeats=r] [--box=LxxLy]");
        return 2;
    }
    DeviceMemory<Memory>::start();
    const std::string kind = argv[1];
    const int n_wanted = std::atoi(argv[2]);
    const int steps = std::stoi(flag(argc, argv, "--steps", "20")), repeats = std::stoi(flag(argc, argv, "--repeats", "3"));
    const double move = std::stod(flag(argc, argv, "--move", "0.03"));
    double lx = 1, ly = 1;
    std::sscanf(flag(argc, argv, "--box", "1x1").c_str(), "%lfx%lf", &lx, &ly);
    const Box box(lx, ly);
    Exec exec;
    bool unsorted = false;
    for (int a = 3; a < argc; ++a) unsorted = unsorted || std::strcmp(argv[a], "--unsorted") == 0;
    const X x0 = make_points(kind, n_wanted, box, !unsorted);
    const int n = static_cast<int>(x0.extent(0));
    const double spacing = std::sqrt(box.volume() / n);

    double best_build = 1e300;
    voronoi::Timings build_phases;
    for (int r = 0; r < repeats; ++r) {
        Mesh mesh(exec, box);
        voronoi::Timings t;
        Kokkos::Timer timer;
        mesh.rebuild(x0, &t);
        mesh.update_cells(&t);
        exec.fence();
        const double ms = timer.seconds() * 1e3;
        if (ms < best_build) {
            best_build = ms;
            build_phases = t;
        }
    }
    std::printf("mesh2d build %s n=%d: %.1f M cells/s (%.2f ms: clip+triangles %.2f, faces %.2f, cells %.2f)\n", kind.c_str(), n,
                n / best_build * 1e-3, best_build, build_phases.build, build_phases.geometry, build_phases.cells);

    // The moving steps: the first with an empty mesh aside, then `steps`.
    X x("positions", n);
    Kokkos::deep_copy(x, x0);
    Mesh mesh(exec, box);
    mesh.update(x);
    mesh.update_cells();
    X base("base", n);
    double best = 1e300;
    voronoi::Timings phases;
    long long flips_best = 0, rounds_best = 0, stuck_best = 0, rebuilds_best = 0;
    for (int r = 0; r < repeats; ++r) {
        voronoi::Timings t;
        long long flips = 0, rounds = 0, stuck = 0, rebuilds = 0;
        Kokkos::deep_copy(base, x);
        double total = 0;
        for (int s = 0; s < steps; ++s) {
            const double dt = move * spacing, phase = 0.7 * (s + r * steps);
            const double side[2] = {box.side[0], box.side[1]};
            Kokkos::parallel_for(
                "bench::drift", Kokkos::RangePolicy<Exec>(exec, 0, n), KOKKOS_LAMBDA(int i) {
                    const double u = 2 * M_PI * x(i, 0) / side[0], v = 2 * M_PI * x(i, 1) / side[1];
                    const double dx = dt * (0.7 * Kokkos::sin(3 * v + phase) + 0.3 * Kokkos::sin(17 * v + u));
                    const double dy = dt * (0.7 * Kokkos::sin(4 * u + phase) + 0.3 * Kokkos::sin(23 * u - v));
                    const double nx = x(i, 0) + dx, ny = x(i, 1) + dy;
                    x(i, 0) = nx - side[0] * Kokkos::floor(nx / side[0]);
                    x(i, 1) = ny - side[1] * Kokkos::floor(ny / side[1]);
                });
            exec.fence();
            Kokkos::Timer timer;
            const voronoi::Report report = mesh.update(x, &t);
            mesh.update_cells(&t);
            exec.fence();
            total += timer.seconds();
            flips += report.flips;
            rounds += report.rounds;
            stuck += report.stuck;
            rebuilds += report.rebuilt;
        }
        if (total < best) {
            best = total;
            phases = t;
            flips_best = flips;
            rounds_best = rounds;
            stuck_best = stuck;
            rebuilds_best = rebuilds;
        }
    }
    const double per = 1.0 / steps;
    std::printf("mesh2d step %s n=%d move=%.3g: %.1f M cells/s (%.3f ms/step: repair %.3f, faces %.3f, cells %.3f, rebuild %.3f); "
                "flips/step %.0f (%.2f%% of edges), rounds/step %.1f, stuck %lld, rebuilds %lld\n",
                kind.c_str(), n, move, n * steps / best * 1e-6, best * 1e3 * per, phases.repair * per, phases.geometry * per,
                phases.cells * per, phases.build * per, double(flips_best) * per, 100.0 * flips_best / (3.0 * n * steps),
                double(rounds_best) * per, stuck_best, rebuilds_best);
    std::printf("memory peak %.1f B/cell\n", double(DeviceMemory<Memory>::peak) / n);
    return 0;
}
