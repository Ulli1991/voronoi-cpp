// LUNA's 1D tests on the 1D Voronoi mesh (hydro::FiniteVolume<1>), timed.
//
//   hydro_1d <sod|shu-osher|woodward-colella> <N> [--moving] [--riemann=hllc|hll] [--order=0|1]
//            [--dt=luna|arepo|fixed] [--courant=<c>] [--fixed-dt=<dt>] [--jitter=<spacings>]
//            [--limiter=face|cell|none] [--steps=<max>] [--repeats=<r>] [--phases] [--memory]
//            [--no-regularise] [--unfused] [--out=<file.npy>] [--series=<name>]
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
// --jitter / 2 spacings. --dt: LUNA's rule at --courant (default 0.8),
// AREPO's (default 0.3), or a constant --fixed-dt; the last step lands on
// t_end. --steps stops after that many steps (throughput).
//
// The run is made --repeats times (default 2) in one process, from the same
// initial state; every run is printed, and the fastest gives the summary
// (M cell updates/s = N * steps / wall time of the stepping loop, time steps
// included). The state hash must be the same in every run. LUNA's rule runs
// fused, one launch per step (--unfused: time_step() and step() apart).
// --phases, always unfused, fences the phases of a step and prints their rates. --memory adds the peak of the device allocations.
// --out writes an .npy of N rows (generator, centroid, V, rho, u, p) in x
// order, x in the problem's coordinates; with --series also <file.npy>.json
// (series, problem, N, steps, wall_s, peak_MB, file) for scripts/compare_1d.py.

#include "finite_volume_1d.hpp"
#include "problems_1d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Memory = Exec::memory_space;
using Mesh = voronoi::Mesh<1, Exec>;
using State = hydro::State<1>;
using hydro::problems_1d::gamma_gas;
using hydro::problems_1d::Problem;
using hydro::problems_1d::problem;

// Peak of the device allocations, through the Kokkos Tools callbacks.
struct DeviceMemory {
    static inline std::map<const void*, std::uint64_t> live;
    static inline std::uint64_t current = 0, peak = 0;

    static bool device(const Kokkos_Profiling_SpaceHandle& h) { return std::strcmp(h.name, Memory::name()) == 0; }
    static void allocate(Kokkos_Profiling_SpaceHandle h, const char*, const void* ptr, std::uint64_t size) {
        if (!device(h)) return;
        live[ptr] = size;
        current += size;
        peak = std::max(peak, current);
    }
    static void deallocate(Kokkos_Profiling_SpaceHandle h, const char*, const void* ptr, std::uint64_t) {
        if (!device(h)) return;
        const auto it = live.find(ptr);
        if (it == live.end()) return;
        current -= it->second;
        live.erase(it);
    }
    static void start() {
        Kokkos::Tools::Experimental::set_allocate_data_callback(allocate);
        Kokkos::Tools::Experimental::set_deallocate_data_callback(deallocate);
    }
};

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

struct Options {
    int n = 0, order = 1;
    double jitter = 0, courant = 0.8, fixed_dt = 0;
    hydro::Limiter limiter = hydro::Limiter::face_centroid;
    hydro::Motion motion = hydro::Motion::static_mesh;
    bool fixed = false, phases = false, regularise = true, fused = true;
    hydro::TimeStep rule = hydro::TimeStep::luna;
    long long max_steps = 0;
};

struct Result {
    long long steps = 0, sorts = 0;
    double wall_s = 0, t = 0;
    std::uint64_t hash = 0;
    hydro::Timings timings;
};

// FNV-1a over the bits of the state (and the positions of a moving run).
template <class V>
void hash_into(std::uint64_t& h, const V& host, int columns) {
    for (std::size_t i = 0; i < host.extent(0); ++i)
        for (int c = 0; c < columns; ++c) {
            std::uint64_t bits;
            const double value = host(i, c);
            std::memcpy(&bits, &value, sizeof bits);
            h = (h ^ bits) * 1099511628211ull;
        }
}

void write_npy(const std::string& path, const std::vector<std::array<double, 6>>& rows) {
    char header[128];
    int length = std::snprintf(header, sizeof header, "{'descr': '<f8', 'fortran_order': False, 'shape': (%zu, 6), }", rows.size());
    while ((10 + length + 1) % 64 != 0) header[length++] = ' ';
    header[length++] = '\n';
    std::ofstream file(path, std::ios::binary);
    file.write("\x93NUMPY\x01\x00", 8);
    const unsigned short size = static_cast<unsigned short>(length);
    file.write(reinterpret_cast<const char*>(&size), 2);
    file.write(header, length);
    file.write(reinterpret_cast<const char*>(rows.data()), static_cast<std::streamsize>(rows.size() * sizeof rows[0]));
}

template <class Riemann>
Result simulate(const Problem& pb, const Options& o, const std::string& out) {
    const Exec exec;
    const int n = o.n;
    Mesh mesh(exec, pb.L, voronoi::Ends::walls);
    {
        Kokkos::View<double* [1], Memory> x("x", n);
        auto h = Kokkos::create_mirror_view(x);
        std::mt19937_64 rng(1);
        std::uniform_real_distribution<double> unit(-0.5, 0.5);
        for (int i = 0; i < n; ++i) h(i, 0) = pb.L * (i + 0.5 + o.jitter * unit(rng)) / n;
        Kokkos::deep_copy(x, h);
        mesh.update(x);
    }
    hydro::FiniteVolume<1, Exec, Riemann> fv(exec, mesh, hydro::Gas{gamma_gas}, o.order, o.limiter, o.motion, pb.walls, o.regularise);
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
    Result r;
    hydro::Timings* timings = o.phases ? &r.timings : nullptr;
    Kokkos::Timer timer;
    double t = 0;
    const bool fused = o.fused && !o.fixed && !o.phases && o.rule == hydro::TimeStep::luna;
    double next = fused ? fv.time_step(o.courant, o.rule) : 0;
    while (t < pb.t_end * (1 - 1e-12) && (o.max_steps == 0 || r.steps < o.max_steps)) {
        double dt = fused ? next : o.fixed ? o.fixed_dt : fv.time_step(o.courant, o.rule, timings);
        dt = std::min(dt, pb.t_end - t);
        if (fused)
            next = fv.step_and_time_step(dt, o.courant);
        else
            fv.step(dt, timings);
        t += dt;
        ++r.steps;
    }
    exec.fence();
    r.wall_s = timer.seconds();
    r.t = t;
    r.sorts = mesh.sorts();
    const auto q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fv.state());
    r.hash = 14695981039346656037ull;
    hash_into(r.hash, q, 3);
    if (o.motion == hydro::Motion::moving)
        hash_into(r.hash, Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fv.positions()), 1);
    if (!out.empty()) {
        mesh.update_cells();
        const auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.positions());
        const auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.centroid());
        const auto V = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.volume());
        std::vector<std::array<double, 6>> rows;
        rows.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            const State W = hydro::primitive(State{q(i, 0) / V(i), q(i, 1) / V(i), q(i, 2) / V(i)}, fv.gas());
            rows.push_back({x(i, 0) + pb.origin, x(i, 0) + c(i, 0) + pb.origin, V(i), W[0], W[1], W[2]});
        }
        std::sort(rows.begin(), rows.end());
        write_npy(out, rows);
    }
    return r;
}

}  // namespace

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    if (argc < 3) {
        std::cerr << "usage: hydro_1d <sod|shu-osher|woodward-colella> <N> [--moving] [--riemann=hllc|hll] [--order=0|1]"
                     " [--dt=luna|arepo|fixed] [--courant=c] [--fixed-dt=dt] [--jitter=s] [--limiter=face|cell|none]"
                     " [--steps=max] [--repeats=r] [--phases] [--memory] [--no-regularise] [--unfused] [--out=file.npy] [--series=name]\n";
        return 1;
    }
    const std::string name = argv[1];
    Options o;
    o.n = std::stoi(argv[2]);
    const bool moving = has(argc, argv, "--moving");
    o.motion = moving ? hydro::Motion::moving : hydro::Motion::static_mesh;
    const std::string riemann = flag(argc, argv, "--riemann", "hllc"), rule_name = flag(argc, argv, "--dt", "luna");
    const std::string limit = flag(argc, argv, "--limiter", "face");
    o.order = std::stoi(flag(argc, argv, "--order", "1"));
    o.rule = rule_name == "arepo" ? hydro::TimeStep::arepo : hydro::TimeStep::luna;
    o.fixed = rule_name == "fixed";
    o.fixed_dt = std::stod(flag(argc, argv, "--fixed-dt", "0"));
    if (o.fixed && !(o.fixed_dt > 0)) throw std::invalid_argument("hydro_1d: --dt=fixed needs --fixed-dt > 0");
    o.courant = std::stod(flag(argc, argv, "--courant", o.rule == hydro::TimeStep::arepo ? "0.3" : "0.8"));
    o.jitter = std::stod(flag(argc, argv, "--jitter", "0"));
    o.max_steps = std::stoll(flag(argc, argv, "--steps", "0"));
    o.phases = has(argc, argv, "--phases");
    o.regularise = !has(argc, argv, "--no-regularise");
    o.fused = !has(argc, argv, "--unfused");
    const int repeats = std::stoi(flag(argc, argv, "--repeats", "2"));
    const std::string out = flag(argc, argv, "--out", "");
    o.limiter = limit == "cell"   ? hydro::Limiter::cell_centroid
                : limit == "none" ? hydro::Limiter::none
                                  : hydro::Limiter::face_centroid;
    if (has(argc, argv, "--memory")) DeviceMemory::start();
    const Problem pb = problem(name);
    Result best;
    bool same = true;
    for (int run = 0; run < repeats; ++run) {
        const std::string file = run == repeats - 1 ? out : std::string();
        const Result r = riemann == "hll" ? simulate<hydro::HLL>(pb, o, file) : simulate<hydro::HLLC>(pb, o, file);
        std::cout << "run: " << run << " wall_s=" << r.wall_s << " M_cell_updates_per_s=" << double(o.n) * r.steps / r.wall_s / 1e6
                  << " hash=" << r.hash << "\n";
        same = same && (run == 0 || r.hash == best.hash);
        if (run == 0 || r.wall_s < best.wall_s) best = r;
    }
    const double rate = double(o.n) * best.steps / best.wall_s;
    std::cout << "ours: problem=" << name << " N=" << o.n << " mesh=" << (moving ? "moving" : "static") << " riemann=" << riemann
              << " order=" << o.order << " dt=" << rule_name << " courant=" << o.courant << " jitter=" << o.jitter
              << " steps=" << best.steps << " t=" << best.t << " sorts=" << best.sorts << " wall_s=" << best.wall_s
              << " cell_updates=" << double(o.n) * best.steps << " M_cell_updates_per_s=" << rate / 1e6 << " hash=" << best.hash
              << " same_bits=" << (same ? 1 : 0) << " backend=" << Exec::name() << "\n";
    if (o.phases) {
        const hydro::Timings& t = best.timings;
        auto phase = [&](const char* label, double ms) {
            if (ms > 0) std::cout << "phase: " << label << " M_cell_updates_per_s=" << double(o.n) * best.steps / (ms * 1e-3) / 1e6 << "\n";
        };
        phase("time_step", t.sweep);
        phase("hydro", t.fluxes);
        phase("mesh", t.mesh);
        phase("whole", t.sweep + t.fluxes + t.mesh);
    }
    const std::string series = flag(argc, argv, "--series", "");
    if (!series.empty() && !out.empty()) {
        std::ofstream json(out + ".json");
        json << "{\"series\": \"" << series << "\", \"problem\": \"" << name << "\", \"N\": " << o.n << ", \"steps\": " << best.steps
             << ", \"wall_s\": " << best.wall_s << ", \"peak_MB\": " << DeviceMemory::peak / 1e6 << ", \"sorts\": " << best.sorts
             << ", \"file\": \"" << out.substr(out.find_last_of('/') + 1) << "\"}\n";
    }
    if (DeviceMemory::peak > 0)
        std::cout << "memory: peak_MB=" << DeviceMemory::peak / 1e6 << " peak_bytes_per_cell=" << double(DeviceMemory::peak) / o.n << "\n";
    return 0;
}
