// LUNA's 1D tests on the 1D Voronoi mesh (hydro::FiniteVolume<1>), timed.
//
//   hydro_1d <sod|shu-osher|woodward-colella> <N> [--moving] [--riemann=hllc|hll] [--order=0|1]
//            [--dt=luna|arepo] [--courant=<c>] [--jitter=<spacings>] [--limiter=face|cell|none]
//            [--steps=<max>] [--out=<file>]
//
// Setups are LUNA's (scripts/luna_1d_run.py has the file:line of each):
// Sod (Toro 1: (1, 0, 1) | (0.125, 0, 0.1) at 0.5 on [0, 1], t = 0.25,
// zero-gradient walls), Shu-Osher (LEFT = (3.857143, 2.629369, 10.33333)
// for x < -4, (1 + 0.2 sin 5x, 0, 1) beyond, on [-5, 5], t = 1.8,
// zero-gradient walls), Woodward-Colella (rho = 1, u = 0, p = 1000 | 0.01
// | 100 split at 0.1 and 0.9 on [0, 1], t = 0.038, reflective walls);
// gamma 1.4. Every cell starts from the exact average of the initial state
// over its own interval (LUNA's: cell averages over its uniform cells;
// equal on a uniform line). N points on a uniform lattice, or moved by up to
// --jitter / 2 spacings. --dt: LUNA's rule at --courant (default 0.8) or
// AREPO's (default 0.3); the last step lands on t_end. --steps stops after
// that many steps (throughput). The run is made twice, the second timed
// (as scripts/luna_1d_run.py times LUNA). Prints steps and M cell
// updates/s (N * steps / wall time of the stepping loop, time steps
// included); --out writes x (centroid), V, rho, u, p per cell in x order.

#include "finite_volume_1d.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Memory = Exec::memory_space;
using Mesh = voronoi::Mesh<1, Exec>;
using State = hydro::State<1>;
constexpr double gamma_gas = 1.4;

std::string flag(int argc, char** argv, const char* name, const char* fallback) {
    const std::size_t length = std::strlen(name);
    for (int a = 3; a < argc; ++a)
        if (std::strncmp(argv[a], name, length) == 0 && argv[a][length] == '=') return argv[a] + length + 1;
    return fallback;
}
bool has(int argc, char** argv, const char* name) {
    for (int a = 3; a < argc; ++a)
        if (std::strcmp(argv[a], name) == 0) return true;
    return false;
}

// One of LUNA's problems: the line, its walls, t_end, and the exact average
// of the conserved initial state over [a, b] (in the problem's coordinates).
struct Problem {
    double L, origin, t_end;
    hydro::Walls walls;
    State (*average)(double a, double b);
};

State conserved(double rho, double u, double p) { return {rho, rho * u, p / (gamma_gas - 1) + 0.5 * rho * u * u}; }
double fraction_below(double x, double a, double b) { return std::clamp((x - a) / (b - a), 0.0, 1.0); }

// Sod: exact cell averages of the jump at t -> 0 (LUNA:
// er.cell_average(..., t=1e-12)); at t = 0 that is the volume fraction.
State sod(double a, double b) {
    const double f = fraction_below(0.5, a, b);
    const State l = conserved(1.0, 0.0, 1.0), r = conserved(0.125, 0.0, 0.1);
    return {f * l[0] + (1 - f) * r[0], f * l[1] + (1 - f) * r[1], f * l[2] + (1 - f) * r[2]};
}
// Shu-Osher, LUNA's exact cell averages (shu_osher.py:26-46) on [a, b].
State shu_osher(double a, double b) {
    constexpr double amp = 0.2, k = 5, x0 = -4, rhoL = 3.857143, uL = 2.629369, pL = 10.33333;
    const double h = b - a, f = fraction_below(x0, a, b), lo = std::max(a, x0);
    double rho = f * rhoL + (1 - f);
    if (b > lo) rho += amp * (std::cos(k * lo) - std::cos(k * b)) / k / h;
    return {rho, f * rhoL * uL, f * (pL / (gamma_gas - 1) + 0.5 * rhoL * uL * uL) + (1 - f) / (gamma_gas - 1)};
}
// Woodward-Colella, LUNA's exact cell averages (woodward_colella.py:14-27).
State woodward_colella(double a, double b) {
    const double f1 = fraction_below(0.1, a, b), f2 = fraction_below(0.9, a, b);
    const double p = f1 * 1000 + (f2 - f1) * 0.01 + (1 - f2) * 100;
    return {1.0, 0.0, p / (gamma_gas - 1)};
}

Problem problem(const std::string& name) {
    const hydro::Walls outflow{hydro::Wall::outflow, hydro::Wall::outflow};
    if (name == "sod") return {1.0, 0.0, 0.25, outflow, sod};
    if (name == "shu-osher") return {10.0, -5.0, 1.8, outflow, shu_osher};
    if (name == "woodward-colella")
        return {1.0, 0.0, 0.038, hydro::Walls{hydro::Wall::reflective, hydro::Wall::reflective}, woodward_colella};
    throw std::invalid_argument("hydro_1d: unknown problem " + name);
}

struct Result {
    int steps = 0;
    double wall_s = 0;
    long long sorts = 0;
};

template <class Riemann>
Result simulate(const Problem& pb, int n, double jitter, int order, hydro::Limiter limiter, hydro::Motion motion,
                hydro::TimeStep rule, double courant, long long max_steps, const std::string& out) {
    const Exec exec;
    Mesh mesh(exec, pb.L, voronoi::Ends::walls);
    {
        Kokkos::View<double* [1], Memory> x("x", n);
        auto h = Kokkos::create_mirror_view(x);
        std::mt19937_64 rng(1);
        std::uniform_real_distribution<double> unit(-0.5, 0.5);
        for (int i = 0; i < n; ++i) h(i, 0) = pb.L * (i + 0.5 + jitter * unit(rng)) / n;
        Kokkos::deep_copy(x, h);
        mesh.update(x);
    }
    hydro::FiniteVolume<1, Exec, Riemann> fv(exec, mesh, hydro::Gas{gamma_gas}, order, limiter, motion, pb.walls);
    {
        const auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.positions());
        const auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.centroid());
        const auto V = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.volume());
        auto q = Kokkos::create_mirror_view(fv.state());
        for (int i = 0; i < n; ++i) {
            const double a = x(i, 0) + c(i, 0) - 0.5 * V(i), b = a + V(i);
            const State u = pb.average(a + pb.origin, b + pb.origin);
            for (int v = 0; v < 3; ++v) q(i, v) = u[v] * V(i);
        }
        Kokkos::deep_copy(fv.state(), q);
    }
    exec.fence();
    Kokkos::Timer timer;
    Result r;
    double t = 0;
    while (t < pb.t_end * (1 - 1e-12) && (max_steps == 0 || r.steps < max_steps)) {
        const double dt = std::min(fv.time_step(courant, rule), pb.t_end - t);
        fv.step(dt);
        t += dt;
        ++r.steps;
    }
    exec.fence();
    r.wall_s = timer.seconds();
    r.sorts = mesh.sorts();
    if (!out.empty()) {
        mesh.update_cells();
        const auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.positions());
        const auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.centroid());
        const auto V = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.volume());
        const auto q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fv.state());
        std::vector<std::array<double, 5>> rows;
        for (int i = 0; i < n; ++i) {
            const State W = hydro::primitive(State{q(i, 0) / V(i), q(i, 1) / V(i), q(i, 2) / V(i)}, fv.gas());
            rows.push_back({x(i, 0) + c(i, 0) + pb.origin, V(i), W[0], W[1], W[2]});
        }
        std::sort(rows.begin(), rows.end());
        std::ofstream file(out);
        file << "# x V rho u p; t=" << t << " steps=" << r.steps << "\n";
        char line[160];
        for (const auto& row : rows) {
            std::snprintf(line, sizeof line, "%.17g %.17g %.17g %.17g %.17g\n", row[0], row[1], row[2], row[3], row[4]);
            file << line;
        }
    }
    return r;
}

}  // namespace

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    if (argc < 3) {
        std::cerr << "usage: hydro_1d <sod|shu-osher|woodward-colella> <N> [--moving] [--riemann=hllc|hll] [--order=0|1]"
                     " [--dt=luna|arepo] [--courant=c] [--jitter=s] [--limiter=face|cell|none] [--steps=max] [--out=file]\n";
        return 1;
    }
    const std::string name = argv[1];
    const int n = std::stoi(argv[2]);
    const bool moving = has(argc, argv, "--moving");
    const std::string riemann = flag(argc, argv, "--riemann", "hllc"), rule_name = flag(argc, argv, "--dt", "luna");
    const std::string limit = flag(argc, argv, "--limiter", "face");
    const int order = std::stoi(flag(argc, argv, "--order", "1"));
    const hydro::TimeStep rule = rule_name == "arepo" ? hydro::TimeStep::arepo : hydro::TimeStep::luna;
    const double courant = std::stod(flag(argc, argv, "--courant", rule == hydro::TimeStep::arepo ? "0.3" : "0.8"));
    const double jitter = std::stod(flag(argc, argv, "--jitter", "0"));
    const long long max_steps = std::stoll(flag(argc, argv, "--steps", "0"));
    const std::string out = flag(argc, argv, "--out", "");
    const hydro::Limiter limiter = limit == "cell"   ? hydro::Limiter::cell_centroid
                                   : limit == "none" ? hydro::Limiter::none
                                                     : hydro::Limiter::face_centroid;
    const Problem pb = problem(name);
    const hydro::Motion motion = moving ? hydro::Motion::moving : hydro::Motion::static_mesh;
    Result r;
    for (int run = 0; run < 2; ++run) {  // the second run is timed
        const std::string file = run == 1 ? out : std::string();
        r = riemann == "hll" ? simulate<hydro::HLL>(pb, n, jitter, order, limiter, motion, rule, courant, max_steps, file)
                             : simulate<hydro::HLLC>(pb, n, jitter, order, limiter, motion, rule, courant, max_steps, file);
    }
    std::cout << "ours: problem=" << name << " N=" << n << " mesh=" << (moving ? "moving" : "static") << " riemann=" << riemann
              << " order=" << order << " dt=" << rule_name << " courant=" << courant << " jitter=" << jitter
              << " steps=" << r.steps << " sorts=" << r.sorts << " wall_s=" << r.wall_s
              << " cell_updates_per_s=" << double(n) * r.steps / r.wall_s << " backend=" << Exec::name() << "\n";
    return 0;
}
