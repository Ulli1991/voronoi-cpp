// The Sedov-Taylor blast of LUNA's scripts/validation/runs/sedov3d_run.py
// on the Voronoi mesh (hydro::FiniteVolume), static or moving, with phase
// timings.
//
//   hydro_sedov <side> <jittered|uniform> [--order=0|1] [--update=cell|face]
//               [--limiter=face|cell|none] [--riemann=hllc|hll] [--precision=<name>]
//               [--moving] [--bulk=<km/s>] [--no-blast] [--box=fx,fy,fz] [--out=<dir>]
//               [--snapshots=10] [--steps=<max>] [--input-order] [--memory]
//
// side^3 generating points in a 50 pc periodic box (jittered: a lattice
// moved by up to 0.1 spacings; uniform: uniform random). AREPO units (kpc,
// 1e10 Msun, km/s), gamma 5/3, n = 1 cm^-3, T = 1e4 K, 1e51 erg into the
// cells nearest the centre whose volumes add up closest to LUNA's 8 central
// cells at the same resolution (8 (L / side)^3), as extra pressure
// (gamma - 1) E / V_deposit over their actual volume V_deposit; run to
// 1e4 yr. --moving: the points move with the flow (AREPO-ng's
// regularisation), the mesh is updated every step. --bulk: the whole box
// moves along x at that speed (a moving blast wave). --box: the box sides as
// multiples of 50 pc (default 1,1,1; about side^3 cells). --out writes snapshots
// 0..snapshots at equal times (LUNA's time_bet_snaps = t_end / n_snaps).
// --steps stops after that many steps (timing runs). Particles are numbered
// in Morton order of their positions (--input-order: as generated, lattice
// rows or random). --update: the update loop, default face at order 1 and
// cell at order 0. --limiter: the gradient limiter (hydro::Limiter: face or
// cell centroids, or none; default face). --riemann: HLLC (default, as
// LUNA) or HLL. --precision: the hydro's precision policy (run_precision
// below; default double). --no-blast: no blast, a uniform stream (with
// --bulk), printing its largest deviation from uniform. Prints per step
// phase medians and cell updates per second and a checksum of the final
// state, positions and volumes (bits),
// the face sides whose extrapolation fell back to the cell state (total,
// steps with any, most in one step) and what the mesh updates did;
// --memory adds device memory (Kokkos allocations) per phase.


#include "finite_volume.hpp"
#include "snapshot.hpp"
#include "support.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>
#include <type_traits>
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
Points morton_order(Points points, const Box& box) {
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
        for (int a = 0; a < 3; ++a) key |= spread(static_cast<std::uint64_t>(points[i][a] * box.inverse[a] * (1 << 21))) << a;
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
Totals totals(const FV& fv) {
    const auto q = fv.state();
    Totals t;
    Kokkos::parallel_reduce(
        "totals", Kokkos::RangePolicy<Exec>(0, static_cast<int>(q.extent(0))),
        KOKKOS_LAMBDA(int i, double& m, double& x, double& y, double& z, double& e) {
            m += q(i, 0);
            x += q(i, 1);
            y += q(i, 2);
            z += q(i, 3);
            e += q(i, 4);
        },
        t.mass, t.px, t.py, t.pz, t.energy);
    return t;
}

// FNV-1a over the bits of every value of a host view (2D, rows x columns).
template <class View>
std::uint64_t bits_hash(const View& v, std::uint64_t h = 1469598103934665603ULL) {
    for (std::size_t i = 0; i < v.extent(0); ++i)
        for (std::size_t a = 0; a < v.extent(1); ++a) {
            const double x = v(i, a);
            std::uint64_t u = 0;
            std::memcpy(&u, &x, sizeof(u));
            for (int b = 0; b < 8; ++b) {
                h ^= (u >> (8 * b)) & 0xff;
                h *= 1099511628211ULL;
            }
        }
    return h;
}

struct Options {
    int side = 0, order = 1, snapshots = 10;
    std::string kind, loop, limit, riemann, out, precision;
    hydro::Update update = hydro::Update::per_face;
    hydro::Limiter limiter = hydro::Limiter::face_centroid;
    bool moving = false, input_order = false, blast = true;
    double bulk = 0;
    long long max_steps = 0;
    Box box{1.0};
};

// The whole run with the hydro's precision policy P (the mesh carries float
// face rows if P asks for them) and Riemann solver R.
template <class P, class R>
void run(const Options& o, Points points) {
    using FV = hydro::FiniteVolume<3, Exec, R, P>;
    using Mesh = typename FV::Mesh;  // with the float face rows and gradient weights P asks for
    // CourantFac of AREPO next generation's Sedov examples
    // (examples/sedov_3d/sedov_globaltstp_3d/param.txt:52).
    const double courant = 0.3;
    const Sedov sedov;
    const Box& box = o.box;
    const long long n = static_cast<long long>(points.size());
    const hydro::Gas gas{sedov.gamma};
    const Exec exec;

    // Mesh.
    Kokkos::Timer timer;
    Mesh mesh(exec, box);
    {
        if (!o.input_order) points = morton_order(points, box);
        const Positions x = to_device(points);
        mesh.update(x);
        mesh.update_cells();
        exec.fence();
    }
    std::cout << "mesh: n=" << n << " kind=" << o.kind << (o.input_order ? " order=input" : " order=morton") << " faces=" << mesh.faces()
              << " faces_per_cell=" << double(mesh.faces()) / n << " build_s=" << timer.seconds() << "\n";
    DeviceMemory::phase("mesh", n);

    // Initial state on the host: the deposit cells by distance to the centre.
    const auto positions = to_host(mesh.positions());
    const auto volume = to_host(mesh.volume());
    // LUNA's 8 central cells: 8 mean cell volumes.
    const double target = 8 * box.volume() / static_cast<double>(n);
    const double centre[3] = {0.5 * box.side[0], 0.5 * box.side[1], 0.5 * box.side[2]};
    auto radius = [&](long long i) {
        double r2 = 0;
        for (int a = 0; a < 3; ++a) {
            const double d = tet_mesh::wrap(positions(i, a) - centre[a], box.side[a], box.inverse[a]);
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
    if (!o.blast) deposit = 0;  // a uniform (free) stream
    const double p_inject = (gas.gamma - 1.0) * sedov.energy() / deposited;
    std::cout << "sedov: rho=" << sedov.rho() << " P=" << sedov.pressure() << " E=" << sedov.energy() << " t_end=" << sedov.t_end()
              << " deposit_cells=" << deposit << " deposit_volume=" << deposited << " luna_volume=" << target
              << " deposit_radius_max=" << (deposit > 0 ? r[by_distance[deposit - 1]] : 0.0) << " p_inject=" << p_inject << "\n";

    FV fv(exec, mesh, gas, o.order, o.update, o.limiter, o.moving ? hydro::Motion::moving : hydro::Motion::static_mesh);
    {  // Q = U V: the background moving with the bulk velocity, the blast's pressure on top
        auto Q = Kokkos::create_mirror_view(fv.state());
        for (long long i = 0; i < n; ++i) {
            const hydro::State<3> W{sedov.rho(), o.bulk, 0, 0, sedov.pressure()};
            const hydro::State<3> u = hydro::conserved(W, gas);
            for (int v = 0; v < 5; ++v) Q(i, v) = u[v] * volume(i);
        }
        for (int k = 0; k < deposit; ++k) Q(by_distance[k], 4) += p_inject / (gas.gamma - 1.0) * volume(by_distance[k]);
        Kokkos::deep_copy(fv.state(), Q);
    }
    DeviceMemory::phase("hydro_setup", n);

    hydro::SnapshotData snap;
    for (int a = 0; a < 3; ++a) snap.sides[a] = box.side[a];
    snap.order = o.order;
    snap.gas = gas;
    snap.units = sedov.units;
    // The radius of the disturbed region: the farthest cell, from the
    // blast centre moving with the bulk velocity, whose density or
    // pressure differs from the background by more than 1e-6. Uniform
    // stream: the largest relative deviation of rho, P and v from the
    // background.
    auto write = [&](int index, double t) {
        const auto Q = to_host(fv.state());
        const auto x = to_host(mesh.positions());
        const auto V = to_host(mesh.volume());
        snap.index = index;
        snap.time = t;
        for (auto* c : {&snap.x, &snap.y, &snap.z, &snap.volume}) c->resize(n);
        for (int v = 0; v < 5; ++v) snap.U[v].resize(n);
        const double cx = centre[0] + o.bulk * t;
        double disturbed = 0, drho = 0, dP = 0, dv = 0;
        const double speed = std::max(std::abs(o.bulk), 1e-300);
        for (long long i = 0; i < n; ++i) {
            snap.x[i] = x(i, 0);
            snap.y[i] = x(i, 1);
            snap.z[i] = x(i, 2);
            snap.volume[i] = V(i);
            for (int v = 0; v < 5; ++v) snap.U[v][i] = Q(i, v) / V(i);
            const hydro::State<3> W =
                hydro::primitive(hydro::State<3>{snap.U[0][i], snap.U[1][i], snap.U[2][i], snap.U[3][i], snap.U[4][i]}, gas);
            drho = std::max(drho, std::abs(W[0] / sedov.rho() - 1));
            dP = std::max(dP, std::abs(W[4] / sedov.pressure() - 1));
            dv = std::max({dv, std::abs(W[1] - o.bulk) / speed, std::abs(W[2]) / speed, std::abs(W[3]) / speed});
            if (std::abs(W[0] / sedov.rho() - 1) <= 1e-6 && std::abs(W[4] / sedov.pressure() - 1) <= 1e-6) continue;
            double r2 = 0;
            for (int a = 0; a < 3; ++a) {
                const double d = tet_mesh::wrap(x(i, a) - (a == 0 ? cx : centre[a]), box.side[a], box.inverse[a]);
                r2 += d * d;
            }
            disturbed = std::max(disturbed, std::sqrt(r2));
        }
        std::cout << "snapshot: index=" << index << " time=" << t << " disturbed_radius=" << disturbed
                  << " shortest_half_box=" << 0.5 * box.shortest();
        if (!o.blast) std::cout << " uniform_deviation: rho=" << drho << " P=" << dP << " v=" << dv;
        if (!o.out.empty()) std::cout << " file=" << hydro::write_snapshot(o.out, snap);
        std::cout << "\n";
    };

    const Totals before = totals(fv);
    if (!o.out.empty()) write(0, 0.0);
    std::vector<double> prim_ms, sweep_ms, flux_ms, update_ms, drift_ms, mesh_ms, cells_ms;
    double t = 0, gpu_ms = 0;
    long long steps = 0, fallbacks = 0, most_fallbacks = 0, steps_with_fallbacks = 0;
    int next = 1;
    Kokkos::Timer wall;
    while (t < sedov.t_end() && (o.max_steps == 0 || steps < o.max_steps)) {
        hydro::Timings timings;
        double dt = fv.time_step(courant, &timings);
        const double t_next = sedov.t_end() * next / o.snapshots;
        bool snapshot = false;
        if (t + dt >= t_next) {
            dt = t_next - t;
            snapshot = true;
        }
        fv.step(dt, &timings);
        prim_ms.push_back(timings.primitives);
        sweep_ms.push_back(timings.sweep);
        flux_ms.push_back(timings.fluxes);
        update_ms.push_back(timings.update);
        drift_ms.push_back(timings.drift);
        mesh_ms.push_back(timings.mesh);
        cells_ms.push_back(timings.cells);
        gpu_ms += timings.primitives + timings.sweep + timings.fluxes + timings.update + timings.drift + timings.mesh + timings.cells;
        fallbacks = fv.fallbacks() < 0 ? -1 : fallbacks + fv.fallbacks();  // -1: not counted (per-cell loop)
        most_fallbacks = std::max(most_fallbacks, fv.fallbacks());
        steps_with_fallbacks += fv.fallbacks() > 0;
        t = snapshot ? t_next : t + dt;
        ++steps;
        if (snapshot) {
            if (!o.out.empty() || next == o.snapshots) write(next, t);
            ++next;
        }
    }
    exec.fence();
    const double wall_s = wall.seconds();
    if (t < sedov.t_end()) write(next, t);  // stopped by --steps: the last state
    DeviceMemory::phase("steps", n);
    const Totals after = totals(fv);
    const double hydro_ms = median(prim_ms) + median(sweep_ms) + median(flux_ms) + median(update_ms);
    const double step_ms = hydro_ms + median(drift_ms) + median(mesh_ms) + median(cells_ms);
    std::cout << "run: precision=" << o.precision << " order=" << o.order << " update=" << (o.update == hydro::Update::per_cell ? "cell" : "face")
              << " steps=" << steps << " limiter=" << o.limit << " riemann=" << o.riemann << " mesh=" << (o.moving ? "moving" : "static")
              << " bulk=" << o.bulk << " blast=" << o.blast << " t=" << t << " wall_s=" << wall_s << " s_per_step=" << wall_s / steps
              << " gpu_s=" << gpu_ms * 1e-3 << "\n";
    std::cout << "fallbacks: total=" << fallbacks << " steps_with=" << steps_with_fallbacks << " most_in_a_step=" << most_fallbacks
              << " face_sides_per_step=" << 2 * static_cast<long long>(mesh.faces()) << "\n";
    const hydro::MeshStats& ms = fv.mesh_stats();
    std::cout << "mesh_updates: updates=" << ms.updates << " rebuilds=" << ms.rebuilds << " stuck=" << ms.stuck
              << " host_flips=" << ms.host_flips << " device_flips=" << ms.device_flips << "\n";
    auto rate = [&](double ms) { return ms > 0 ? n / (ms * 1e-3) / 1e6 : 0.0; };
    std::cout << "phases_ms_median: primitives=" << median(prim_ms) << " sweep=" << median(sweep_ms) << " fluxes=" << median(flux_ms)
              << " update=" << median(update_ms) << " drift=" << median(drift_ms) << " mesh=" << median(mesh_ms)
              << " cells=" << median(cells_ms) << " hydro=" << hydro_ms << " step=" << step_ms
              << " cell_updates_per_s=" << n / (step_ms * 1e-3) << "\n";
    std::cout << "throughput_M_per_s: primitives=" << rate(median(prim_ms)) << " sweep=" << rate(median(sweep_ms))
              << " fluxes=" << rate(median(flux_ms)) << " update=" << rate(median(update_ms)) << " mesh=" << rate(median(mesh_ms))
              << " cells=" << rate(median(cells_ms)) << " hydro=" << rate(hydro_ms) << " step=" << rate(step_ms)
              << " hydro_G_face_updates_per_s=" << mesh.faces() / (hydro_ms * 1e-3) / 1e9 << "\n";
    std::cout << "conservation: mass=" << (after.mass - before.mass) / before.mass
              << " energy=" << (after.energy - before.energy) / before.energy << " momentum=(" << after.px - before.px << ","
              << after.py - before.py << "," << after.pz - before.pz << ") momentum_scale=" << before.mass * std::sqrt(2 * before.energy / before.mass)
              << "\n";
    const auto Qh = to_host(fv.state());
    const auto xh = to_host(mesh.positions());
    const auto Vh = to_host(Kokkos::View<const double* [1], Kokkos::LayoutLeft, Memory>(mesh.volume().data(), n));
    std::cout << "checksum: steps=" << steps << " Q=" << std::hex << bits_hash(Qh) << " x=" << bits_hash(xh) << " volume=" << bits_hash(Vh)
              << std::dec << "\n";
}

// The precision policies of --precision (hydro/precision.hpp): <name>
// geometry / gradient weights / gradients / face fluxes / Riemann; default:
// double, the fit solved in the sweep (weights void).
template <class R>
void run_precision(const Options& o, const Points& points) {
    using hydro::Half;
    using hydro::Precision;
    const std::string& p = o.precision;
    if (p == "default") return run<Precision<>, R>(o, points);
    if constexpr (std::is_same_v<R, hydro::HLLC>) {
        if (p == "weight32") return run<Precision<double, float>, R>(o, points);
        if (p == "weight64") return run<Precision<double, double>, R>(o, points);
        if (p == "flux32") return run<Precision<double, void, double, float>, R>(o, points);
        if (p == "grad32") return run<Precision<double, void, float>, R>(o, points);
        if (p == "grad16") return run<Precision<double, void, Half>, R>(o, points);
        if (p == "geo32") return run<Precision<float>, R>(o, points);
        if (p == "riemann32") return run<Precision<double, void, double, double, float>, R>(o, points);
        if (p == "mixed") return run<Precision<float, void, float, float>, R>(o, points);
        if (p == "mixed16") return run<Precision<float, void, Half, float>, R>(o, points);
        if (p == "low") return run<Precision<float, void, Half, float, float>, R>(o, points);
        if (p == "lowweights") return run<Precision<float, float, Half, float, float>, R>(o, points);
    }
    throw std::invalid_argument("hydro_sedov: unknown --precision " + p + " (HLL: double only)");
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]).rfind("--", 0) != 0) args.push_back(argv[i]);
    if (args.size() < 2) {
        std::cerr << "usage: hydro_sedov <side> <jittered|uniform> [--order=0|1] [--update=cell|face] [--limiter=face|cell|none]"
                     " [--riemann=hllc|hll] [--precision=default|weight32|weight64|flux32|grad32|grad16|geo32|riemann32|mixed|mixed16|low|lowweights]"
                     " [--moving] [--bulk=km/s] [--no-blast] [--box=fx,fy,fz] [--out=dir]"
                     " [--snapshots=10] [--steps=max] [--input-order] [--memory]\n";
        return 2;
    }
    if (flag(argc, argv, "--memory", "") == "1") DeviceMemory::start();
    Options o;
    o.side = std::stoi(args[0]);
    o.kind = args[1];
    o.order = std::stoi(flag(argc, argv, "--order", "1"));
    // Default loop: the faster one measured on the A100 (research_notes/hydro_stage1_2026-10-31.md).
    o.loop = flag(argc, argv, "--update", o.order == 1 ? "face" : "cell");
    o.update = o.loop == "face" ? hydro::Update::per_face : hydro::Update::per_cell;
    o.limit = flag(argc, argv, "--limiter", "face");
    o.limiter = o.limit == "none"   ? hydro::Limiter::none
                : o.limit == "cell" ? hydro::Limiter::cell_centroid
                                    : hydro::Limiter::face_centroid;
    o.riemann = flag(argc, argv, "--riemann", "hllc");
    o.precision = flag(argc, argv, "--precision", "default");
    o.moving = flag(argc, argv, "--moving", "") == "1";
    o.blast = flag(argc, argv, "--no-blast", "") != "1";
    o.bulk = std::stod(flag(argc, argv, "--bulk", "0"));  // km/s along x
    o.out = flag(argc, argv, "--out", "");
    o.snapshots = std::stoi(flag(argc, argv, "--snapshots", "10"));
    o.max_steps = std::stoll(flag(argc, argv, "--steps", "0"));
    o.input_order = flag(argc, argv, "--input-order", "") == "1";
    // The periodic box: --box=fx,fy,fz times the 50 pc of LUNA's set-up (a
    // cube by default); the lattice keeps about equal spacings per axis.
    const Sedov sedov;
    const std::string shape_arg = flag(argc, argv, "--box", "1,1,1");
    double f[3] = {1, 1, 1};
    std::sscanf(shape_arg.c_str(), "%lf,%lf,%lf", &f[0], &f[1], &f[2]);
    o.box = Box(sedov.box * f[0], sedov.box * f[1], sedov.box * f[2]);
    const Points points = scaled(make_points_off_grid(o.kind, o.side * o.side * o.side, shape(o.box)), o.box);
    if (o.riemann == "hll")
        run_precision<hydro::HLL>(o, points);
    else
        run_precision<hydro::HLLC>(o, points);
    return 0;
}
