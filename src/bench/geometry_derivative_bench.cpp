// Bench: voronoi::GeometryDerivative timed against one build and one moving step (G cell updates/s), its scratch in
// B/cell. Flow: points -> build + update_cells -> steps -> tangent, adjoint. Usage: geometry_derivative_bench <2|3> <n>
// [kind] [repeats]; 3D kinds of support::make_points_off_grid, 2D jittered or uniform.

#include "device_memory.hpp"
#include "geometry_derivative.hpp"
#include "support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using support::Exec;
using support::Memory;

// Median ms of f over `repeats` runs after one warm-up.
template <class F>
double median_ms(int repeats, const F& f) {
    f();
    std::vector<double> times;
    for (int r = 0; r < repeats; ++r) {
        Kokkos::Timer timer;
        f();
        Exec{}.fence();
        times.push_back(timer.seconds() * 1e3);
    }
    std::nth_element(times.begin(), times.begin() + times.size() / 2, times.end());
    return times[times.size() / 2];
}

// n points of `kind` in the unit box (D = 2: jittered by 0.1 spacings on a square lattice, or uniform), on the device.
template <int D>
Kokkos::View<double* [D], Kokkos::LayoutRight, Memory> make_points(const std::string& kind, int n) {
    Kokkos::View<double* [D], Kokkos::LayoutRight, Memory> x("x", n);
    auto h = Kokkos::create_mirror_view(x);
    if constexpr (D == 3) {
        const support::Points p = support::make_points_off_grid(kind, n);
        for (int i = 0; i < n; ++i)
            for (int a = 0; a < 3; ++a) h(i, a) = p[i][a];
    } else {
        std::mt19937_64 rng(20261122);
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        const int side = static_cast<int>(std::lround(std::sqrt(n)));
        for (int i = 0; i < n; ++i) {
            const double u = kind == "uniform" ? unit(rng) : (static_cast<double>(i / side) + 0.5 + 0.1 * (2 * unit(rng) - 1)) / side;
            const double v = kind == "uniform" ? unit(rng) : (static_cast<double>(i % side) + 0.5 + 0.1 * (2 * unit(rng) - 1)) / side;
            h(i, 0) = u - std::floor(u);
            h(i, 1) = v - std::floor(v);
        }
        if (kind == "uniform") {  // in row-major order of a grid of about one point per cell, for locality
            std::vector<std::pair<long long, std::array<double, 2>>> keyed(n);
            for (int i = 0; i < n; ++i)
                keyed[i] = {static_cast<long long>(h(i, 1) * side) * side + static_cast<long long>(h(i, 0) * side),
                            {h(i, 0), h(i, 1)}};
            std::sort(keyed.begin(), keyed.end());
            for (int i = 0; i < n; ++i) {
                h(i, 0) = keyed[i].second[0];
                h(i, 1) = keyed[i].second[1];
            }
        }
    }
    Kokkos::deep_copy(x, h);
    return x;
}

// Moves every point by `spacings` spacings along a smooth periodic shear, v_a = sin(2 pi x_{a+1}), wrapped into the box.
template <int D, class X>
void advect(const X& x, double spacings) {
    const int n = static_cast<int>(x.extent(0));
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
    const double distance = spacings * std::pow(1.0 / n, 1.0 / D), two_pi = 6.283185307179586;
    for (int i = 0; i < n; ++i) {
        double p[D];
        for (int a = 0; a < D; ++a) p[a] = h(i, a) + distance * std::sin(two_pi * h(i, (a + 1) % D));
        for (int a = 0; a < D; ++a) {
            h(i, a) = p[a] - std::floor(p[a]);
            if (h(i, a) >= 1.0) h(i, a) = 0;
        }
    }
    Kokkos::deep_copy(x, h);
}

template <int D>
int run(int n, const std::string& kind, int repeats) {
    using Mesh = voronoi::Mesh<D, Exec>;
    using Rows = Kokkos::View<double* [D], Kokkos::LayoutRight, Memory>;
    auto x = make_points<D>(kind, n);
    n = static_cast<int>(x.extent(0));
    Mesh mesh(Exec{}, 1.0);
    if constexpr (D == 3) {
        voronoi::Layout layout;
        layout.at_build = true;  // cells in curve order from the build
        mesh.set_layout(layout);
    }
    const double build_ms = median_ms(repeats, [&] {
        mesh.rebuild(x);
        mesh.update_cells();
    });
    // Moving steps of 0.03 spacings along a smooth flow (update + update_cells), the median after three; the points
    // follow the build's re-sort (3D, at_build) and any later one.
    double step_ms = 0;
    {
        [[maybe_unused]] const voronoi::Report built = mesh.rebuild(x);
        mesh.update_cells();
        if constexpr (D == 3)
            if (built.resorted) mesh.permute(x);
        std::vector<double> times;
        for (int r = 0; r < repeats + 3; ++r) {
            advect<D>(x, 0.03);
            Exec{}.fence();
            Kokkos::Timer timer;
            [[maybe_unused]] const voronoi::Report report = mesh.update(x);
            mesh.update_cells();
            Exec{}.fence();
            if (r >= 3) times.push_back(timer.seconds() * 1e3);
            if constexpr (D == 3)
                if (report.resorted) mesh.permute(x);
        }
        std::nth_element(times.begin(), times.begin() + times.size() / 2, times.end());
        step_ms = times[times.size() / 2];
    }
    const int faces = mesh.faces();
    Rows dx("dx", n), dc("dc", n), dA("dA", faces), dxf("dxf", faces), cbar("cbar", n), Abar("Abar", faces),
        xfbar("xfbar", faces), xbar("xbar", n);
    Kokkos::View<double*, Memory> dV("dV", n), Vbar("Vbar", n);
    Kokkos::deep_copy(dx, 1e-3);
    Kokkos::deep_copy(Vbar, 1.0);
    Kokkos::deep_copy(cbar, 1.0);
    Kokkos::deep_copy(Abar, 1.0);
    Kokkos::deep_copy(xfbar, 1.0);
    const voronoi::GeometryDerivative<Mesh> derivative(mesh);
    const double tangent_ms = median_ms(repeats, [&] { derivative.tangent(dx, dV, dc, dA, dxf); });
    const double adjoint_ms = median_ms(repeats, [&] { derivative.adjoint(Vbar, cbar, Abar, xfbar, xbar); });
    const double scratch = static_cast<double>(derivative.scratch_bytes()) / n;
    const double io = (static_cast<double>(n) * (2 * D + 1) + static_cast<double>(faces) * 2 * D) * 8 / n;
    auto g = [&](double ms) { return n / (ms * 1e6); };
    std::printf(
        "geomderiv: D=%d n=%d kind=%s faces_per_cell=%.2f build_G=%.4f step_G=%.4f tangent_G=%.4f adjoint_G=%.4f "
        "tangent_per_build=%.3f adjoint_per_build=%.3f tangent_per_step=%.3f adjoint_per_step=%.3f "
        "scratch_B_per_cell=%.1f caller_views_B_per_cell=%.1f build_ms=%.3f step_ms=%.3f tangent_ms=%.3f "
        "adjoint_ms=%.3f\n",
        D, n, kind.c_str(), static_cast<double>(faces) / n, g(build_ms), g(step_ms), g(tangent_ms), g(adjoint_ms),
        tangent_ms / build_ms, adjoint_ms / build_ms, tangent_ms / step_ms, adjoint_ms / step_ms, scratch, io, build_ms,
        step_ms, tangent_ms, adjoint_ms);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    if (argc < 3) {
        std::fprintf(stderr, "usage: geometry_derivative_bench <2|3> <n> [kind] [repeats]\n");
        return 2;
    }
    const int dimension = std::atoi(argv[1]), n = std::atoi(argv[2]);
    const std::string kind = argc > 3 ? argv[3] : "jittered";
    const int repeats = argc > 4 ? std::atoi(argv[4]) : 5;
    return dimension == 2 ? run<2>(n, kind, repeats) : run<3>(n, kind, repeats);
}
