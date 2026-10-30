// The Sedov-Taylor blast of LUNA's scripts/validation/runs/sedov3d_run.py
// on the static Voronoi mesh (hydro::FiniteVolume), with phase timings.
//
//   hydro_sedov <side> <jittered|uniform> [--order=0|1] [--update=cell|face]
//               [--limiter=face|cell|none] [--riemann=hllc|hll] [--out=<dir>] [--snapshots=10]
//               [--steps=<max>] [--input-order] [--memory]
//
// side^3 generating points in a 50 pc periodic box (jittered: a lattice
// moved by up to 0.1 spacings; uniform: uniform random). AREPO units (kpc,
// 1e10 Msun, km/s), gamma 5/3, n = 1 cm^-3, T = 1e4 K, 1e51 erg into the
// cells nearest the centre whose volumes add up closest to LUNA's 8 central
// cells at the same resolution (8 (L / side)^3), as extra pressure
// (gamma - 1) E / V_deposit over their actual volume V_deposit; run to
// 1e4 yr. --out writes snapshots 0..snapshots at equal times (LUNA's
// time_bet_snaps = t_end / n_snaps). --steps stops after that many steps
// (timing runs). Particles are numbered in Morton order of their positions
// (--input-order: as generated, lattice rows or random). --update: the
// update loop, default face at order 1 and cell at order 0. --limiter: the
// gradient limiter (hydro::Limiter: face or cell centroids, or none; default
// face). --riemann: HLLC (default, as LUNA) or HLL. Prints per step phase medians and cell updates per second, and the
// face sides whose extrapolation fell back to the cell state (total, steps
// with any, most in one step); --memory adds device memory (Kokkos
// allocations) per phase.

#include "finite_volume.hpp"
#include "snapshot.hpp"
#include "support.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <numeric>
#include <unordered_map>
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace {

using namespace support;

// Device memory through the Kokkos Tools allocation callbacks: now and at
// the peak of the current phase, with the live allocations at that peak.
struct DeviceMemory {
    static inline bool on = false;
    static inline std::unordered_map<const void*, std::pair<std::string, std::uint64_t>> live;
    static inline std::uint64_t current = 0, peak = 0;
    static inline std::map<std::string, std::uint64_t> at_peak;

    static bool device(const Kokkos_Profiling_SpaceHandle& h) { return std::strcmp(h.name, Memory::name()) == 0; }
    static void allocate(Kokkos_Profiling_SpaceHandle h, const char* label, const void* ptr, std::uint64_t size) {
        if (!device(h)) return;
        live[ptr] = {label, size};
        current += size;
        if (current <= peak) return;
        peak = current;
        at_peak.clear();
        for (const auto& [p, entry] : live) at_peak[entry.first] += entry.second;
    }
    static void deallocate(Kokkos_Profiling_SpaceHandle h, const char*, const void* ptr, std::uint64_t) {
        if (!device(h)) return;
        const auto it = live.find(ptr);
        if (it == live.end()) return;
        current -= it->second.second;
        live.erase(it);
    }
    static void start() {
        on = true;
        Kokkos::Tools::Experimental::set_allocate_data_callback(allocate);
        Kokkos::Tools::Experimental::set_deallocate_data_callback(deallocate);
    }
    static void phase(const std::string& name, long long n) {
        if (!on) return;
        Kokkos::fence();
        std::cout << "memory: phase=" << name << " current_MB=" << current / 1e6 << " peak_MB=" << peak / 1e6
                  << " peak_bytes_per_cell=" << double(peak) / n << " current_bytes_per_cell=" << double(current) / n;
#ifdef KOKKOS_ENABLE_CUDA
        std::size_t free = 0, total = 0;
        cudaMemGetInfo(&free, &total);
        std::cout << " gpu_used_MB=" << (total - free) / 1e6 << " gpu_total_MB=" << total / 1e6;
#endif
        std::cout << "\n";
        for (const auto& [label, bytes] : at_peak)
            std::cout << "  " << label << " " << bytes / 1e6 << " MB " << double(bytes) / n << " B/cell\n";
        peak = current;
        at_peak.clear();
        for (const auto& [p, entry] : live) at_peak[entry.first] += entry.second;
    }
};

std::string flag(int argc, char** argv, const std::string& name, const std::string& fallback) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind(name + "=", 0) == 0) return arg.substr(name.size() + 1);
        if (arg == name) return "1";
    }
    return fallback;
}

// The points in Morton order (21 bits per coordinate): particle ids follow
// space, so neighbouring cells sit close in memory.
Points morton_order(Points points, double box) {
    auto spread = [](std::uint64_t v) {  // the 21 bits of v to every third bit
        v &= 0x1fffff;
        v = (v | v << 32) & 0x1f00000000ffffULL;
        v = (v | v << 16) & 0x1f0000ff0000ffULL;
        v = (v | v << 8) & 0x100f00f00f00f00fULL;
        v = (v | v << 4) & 0x10c30c30c30c30c3ULL;
        v = (v | v << 2) & 0x1249249249249249ULL;
        return v;
    };
    std::vector<std::pair<std::uint64_t, std::size_t>> keys(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        std::uint64_t key = 0;
        for (int a = 0; a < 3; ++a) key |= spread(static_cast<std::uint64_t>(points[i][a] / box * (1 << 21))) << a;
        keys[i] = {key, i};
    }
    std::sort(keys.begin(), keys.end());
    Points sorted(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) sorted[i] = points[keys[i].second];
    return sorted;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// The Sedov set-up of LUNA (constants: scripts/validation/runs/sedov3d_run.py:19-26,
// src/luna/ic_generation/sedov3d.py:9-14).
struct Sedov {
    static constexpr double year_s = 3.15576e7, kpc_cm = 3.08568e21, msun_g = 1.98892e33;
    static constexpr double proton_g = 1.6726219e-24, boltzmann = 1.380649e-16;
    hydro::Units units{kpc_cm, 1e10 * msun_g, 1e5};
    double box = 50.0e-3;  // kpc
    double gamma = 5.0 / 3.0;
    double time_s() const { return units.length_cm / units.velocity_cm_s; }
    double density_cgs() const { return units.mass_g / std::pow(units.length_cm, 3); }
    double pressure_cgs() const { return density_cgs() * units.velocity_cm_s * units.velocity_cm_s; }
    double energy_cgs() const { return units.mass_g * units.velocity_cm_s * units.velocity_cm_s; }
    double t_end() const { return 1.0e4 * year_s / time_s(); }
    double rho() const { return 1.0 * 1.0 * proton_g / density_cgs(); }
    double pressure() const { return 1.0 * boltzmann * 1.0e4 / pressure_cgs(); }
    double energy() const { return 1.0e51 / energy_cgs(); }
};

struct Totals {
    double mass = 0, px = 0, py = 0, pz = 0, energy = 0;
};

template <class FV>
Totals totals(const FV& fv, const Kokkos::View<const double*, Memory>& volume) {
    const auto u = fv.state();
    Totals t;
    Kokkos::parallel_reduce(
        "totals", Kokkos::RangePolicy<Exec>(0, static_cast<int>(u.extent(0))),
        KOKKOS_LAMBDA(int i, double& m, double& x, double& y, double& z, double& e) {
            m += u(i, 0) * volume(i);
            x += u(i, 1) * volume(i);
            y += u(i, 2) * volume(i);
            z += u(i, 3) * volume(i);
            e += u(i, 4) * volume(i);
        },
        t.mass, t.px, t.py, t.pz, t.energy);
    return t;
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]).rfind("--", 0) != 0) args.push_back(argv[i]);
    if (args.size() < 2) {
        std::cerr << "usage: hydro_sedov <side> <jittered|uniform> [--order=0|1] [--update=cell|face] [--limiter=face|cell|none]"
                     " [--riemann=hllc|hll] [--out=dir]"
                     " [--snapshots=10] [--steps=max] [--input-order] [--memory]\n";
        return 2;
    }
    if (flag(argc, argv, "--memory", "") == "1") DeviceMemory::start();
    const int side = std::stoi(args[0]);
    const std::string kind = args[1];
    const int order = std::stoi(flag(argc, argv, "--order", "1"));
    // Default loop: the faster one measured on the A100 (research_notes/hydro_stage1_2026-10-31.md).
    const std::string loop = flag(argc, argv, "--update", order == 1 ? "face" : "cell");
    const hydro::Update update = loop == "face" ? hydro::Update::per_face : hydro::Update::per_cell;
    const std::string limit = flag(argc, argv, "--limiter", "face");
    const hydro::Limiter limiter = limit == "none"   ? hydro::Limiter::none
                                   : limit == "cell" ? hydro::Limiter::cell_centroid
                                                     : hydro::Limiter::face_centroid;
    const std::string riemann = flag(argc, argv, "--riemann", "hllc");
    // CourantFac of AREPO next generation's Sedov examples
    // (examples/sedov_3d/sedov_globaltstp_3d/param.txt:52).
    const double courant = 0.3;
    const std::string out = flag(argc, argv, "--out", "");
    const int snapshots = std::stoi(flag(argc, argv, "--snapshots", "10"));
    const long long max_steps = std::stoll(flag(argc, argv, "--steps", "0"));
    const bool input_order = flag(argc, argv, "--input-order", "") == "1";
    const long long n = static_cast<long long>(side) * side * side;
    const Sedov sedov;
    const hydro::Gas gas{sedov.gamma};
    const Exec exec;

    // Mesh.
    Kokkos::Timer timer;
    voronoi::Mesh<Exec> mesh(exec, sedov.box);
    {
        Points points = scaled(make_points_off_grid(kind, static_cast<int>(n)), sedov.box);
        if (!input_order) points = morton_order(points, sedov.box);
        const Positions x = to_device(points);
        mesh.update(x);
        mesh.update_cells();
        exec.fence();
    }
    std::cout << "mesh: n=" << n << " kind=" << kind << (input_order ? " order=input" : " order=morton") << " faces=" << mesh.faces() << " faces_per_cell=" << double(mesh.faces()) / n
              << " build_s=" << timer.seconds() << "\n";
    DeviceMemory::phase("mesh", n);

    // Initial state on the host: the deposit cells by distance to the centre.
    const auto positions = to_host(mesh.positions());
    const auto volume = to_host(mesh.volume());
    const double centre = 0.5 * sedov.box, spacing = sedov.box / side, target = 8 * spacing * spacing * spacing;
    auto radius = [&](long long i) {
        double r2 = 0;
        for (int a = 0; a < 3; ++a) {
            const double d = tet_mesh::wrap(positions(i, a) - centre, sedov.box);
            r2 += d * d;
        }
        return std::sqrt(r2);
    };
    std::vector<int> by_distance(n);
    std::iota(by_distance.begin(), by_distance.end(), 0);
    std::vector<double> r(n);
    for (long long i = 0; i < n; ++i) r[i] = radius(i);
    std::partial_sort(by_distance.begin(), by_distance.begin() + std::min<long long>(n, 1000), by_distance.end(),
                      [&](int a, int b) { return r[a] < r[b] || (r[a] == r[b] && a < b); });
    int deposit = 0;
    double deposited = 0;
    for (int k = 0; k < std::min<long long>(n, 1000); ++k) {
        const double next = deposited + volume(by_distance[k]);
        if (std::abs(next - target) >= std::abs(deposited - target)) break;
        deposited = next;
        ++deposit;
    }
    const double p_inject = (gas.gamma - 1.0) * sedov.energy() / deposited;
    std::cout << "sedov: rho=" << sedov.rho() << " P=" << sedov.pressure() << " E=" << sedov.energy() << " t_end=" << sedov.t_end()
              << " deposit_cells=" << deposit << " deposit_volume=" << deposited << " luna_volume=" << target
              << " deposit_radius_max=" << r[by_distance[deposit - 1]] << " p_inject=" << p_inject << "\n";

    // The run with Riemann solver R (the type of `solver`).
    auto simulate = [&](auto solver) {
        using R = decltype(solver);
        (void)solver;
        hydro::FiniteVolume<Exec, R> fv(exec, mesh, gas, order, update, limiter);
        {
            auto U = Kokkos::create_mirror_view(fv.state());
            for (long long i = 0; i < n; ++i) {
                const hydro::State W{sedov.rho(), 0, 0, 0, sedov.pressure()};
                const hydro::State u = hydro::conserved(W, gas);
                for (int v = 0; v < 5; ++v) U(i, v) = u[v];
            }
            for (int k = 0; k < deposit; ++k) U(by_distance[k], 4) += p_inject / (gas.gamma - 1.0);
            Kokkos::deep_copy(fv.state(), U);
        }
        DeviceMemory::phase("hydro_setup", n);

        hydro::SnapshotData snap;
        snap.box = sedov.box;
        snap.order = order;
        snap.gas = gas;
        snap.units = sedov.units;
        snap.x.resize(n);
        snap.y.resize(n);
        snap.z.resize(n);
        snap.volume.resize(n);
        for (long long i = 0; i < n; ++i) {
            snap.x[i] = positions(i, 0);
            snap.y[i] = positions(i, 1);
            snap.z[i] = positions(i, 2);
            snap.volume[i] = volume(i);
        }
        // The radius of the disturbed region: the farthest cell whose density or
        // pressure differs from the background by more than 1e-6.
        auto write = [&](int index, double t) {
            const auto U = to_host(fv.state());
            snap.index = index;
            snap.time = t;
            double disturbed = 0;
            for (int v = 0; v < 5; ++v) snap.U[v].resize(n);
            for (long long i = 0; i < n; ++i) {
                for (int v = 0; v < 5; ++v) snap.U[v][i] = U(i, v);
                const hydro::State W = hydro::primitive({U(i, 0), U(i, 1), U(i, 2), U(i, 3), U(i, 4)}, gas);
                if (std::abs(W[0] / sedov.rho() - 1) > 1e-6 || std::abs(W[4] / sedov.pressure() - 1) > 1e-6)
                    disturbed = std::max(disturbed, r[i]);
            }
            std::cout << "snapshot: index=" << index << " time=" << t << " disturbed_radius=" << disturbed
                      << " half_box=" << 0.5 * sedov.box;
            if (!out.empty()) std::cout << " file=" << hydro::write_snapshot(out, snap);
            std::cout << "\n";
        };

        const Totals before = totals(fv, mesh.volume());
        if (!out.empty()) write(0, 0.0);
        std::vector<double> dt_ms, gradient_ms, flux_ms, update_ms;
        double t = 0, gpu_ms = 0;
        long long steps = 0, fallbacks = 0, most_fallbacks = 0, steps_with_fallbacks = 0;
        int next = 1;
        Kokkos::Timer wall;
        while (t < sedov.t_end() && (max_steps == 0 || steps < max_steps)) {
            timer.reset();
            double dt = fv.time_step(courant);
            dt_ms.push_back(timer.seconds() * 1e3);
            const double t_next = sedov.t_end() * next / snapshots;
            bool snapshot = false;
            if (t + dt >= t_next) {
                dt = t_next - t;
                snapshot = true;
            }
            hydro::Timings timings;
            fv.step(dt, &timings);
            gradient_ms.push_back(timings.gradients);
            flux_ms.push_back(timings.fluxes);
            update_ms.push_back(timings.update);
            gpu_ms += dt_ms.back() + timings.gradients + timings.fluxes + timings.update;
            fallbacks = fv.fallbacks() < 0 ? -1 : fallbacks + fv.fallbacks();  // -1: not counted (per-cell loop)
            most_fallbacks = std::max(most_fallbacks, fv.fallbacks());
            steps_with_fallbacks += fv.fallbacks() > 0;
            t = snapshot ? t_next : t + dt;
            ++steps;
            if (snapshot) {
                if (!out.empty() || next == snapshots) write(next, t);
                ++next;
            }
        }
        exec.fence();
        const double wall_s = wall.seconds();
        DeviceMemory::phase("steps", n);
        const Totals after = totals(fv, mesh.volume());
        const double step_ms = median(dt_ms) + median(gradient_ms) + median(flux_ms) + median(update_ms);
        std::cout << "run: order=" << order << " update=" << (update == hydro::Update::per_cell ? "cell" : "face") << " steps=" << steps
                  << " limiter=" << limit << " riemann=" << riemann << " t=" << t << " wall_s=" << wall_s << " s_per_step=" << wall_s / steps
                  << " gpu_s=" << gpu_ms * 1e-3 << "\n";
        std::cout << "fallbacks: total=" << fallbacks << " steps_with=" << steps_with_fallbacks << " most_in_a_step=" << most_fallbacks
                  << " face_sides_per_step=" << 2 * static_cast<long long>(mesh.faces()) << "\n";
        std::cout << "phases_ms_median: time_step=" << median(dt_ms) << " gradients=" << median(gradient_ms)
                  << " fluxes=" << median(flux_ms) << " update=" << median(update_ms) << " step=" << step_ms
                  << " cell_updates_per_s=" << n / (step_ms * 1e-3) << "\n";
        std::cout << "conservation: mass=" << (after.mass - before.mass) / before.mass
                  << " energy=" << (after.energy - before.energy) / before.energy << " momentum=(" << after.px << "," << after.py << ","
                  << after.pz << ") momentum_scale=" << before.mass * std::sqrt(2 * before.energy / before.mass) << "\n";
    };
    if (riemann == "hll")
        simulate(hydro::HLL{});
    else
        simulate(hydro::HLLC{});
    return 0;
}
