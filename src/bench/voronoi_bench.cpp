// Timing and debugging of the mesh pipeline (not a test).
//
//   voronoi_bench rebuild <n> <distribution> [variants=all] [repeats=5] [--width=<factor>] [--no-check]
//       Full rebuild by clipping. Variants thread, exact and warp, selected
//       by substring. Every variant's cells are checked (support::check_cells,
//       PASS/FAIL; brute force on a sample, seconds at 2.1M) unless --no-check.
//   voronoi_bench step <n> <distribution> <displacement> [steps=10] [--verify] [--debug] [--fallback=rebuild]
//       The persistent mesh (voronoi::Mesh): the start kernels (clipping
//       with rings, from_rings, tet_start::build) and the mesh's start, then
//       `steps` steps of the flow, `displacement` spacings each: device
//       repair (screened), the fallback when stuck (host, or a rebuild),
//       circumcentres, face rings, faces and cells. --verify compares every
//       step with an exact rebuild (support::compare_with_rebuild); --debug
//       adds adjacency and stuck details.
//   --memory (both modes): device memory per phase (Kokkos allocations:
//       current, peak, and the live allocations at the peak by label).
//
// Distributions: uniform, jittered, lattice, wave, clustered, halo, or
// <sc|bcc|fcc|shells|planes|pairs>@<perturbation in spacings>.

#include "support.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace {

using namespace support;

// Device memory accounting through the Kokkos Tools allocation callbacks:
// bytes in the default memory space now and at the peak of the current
// phase, and the live allocations (summed by label) at that peak.
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
    // Report the phase that ends here (per cell for n cells), then start the next.
    static void phase(const std::string& name, int n, bool details = true) {
        if (!on) return;
        Kokkos::fence();
        std::cout << "memory: phase=" << name << " current_MB=" << current / 1e6 << " peak_MB=" << peak / 1e6
                  << " peak_bytes_per_cell=" << double(peak) / n;
#ifdef KOKKOS_ENABLE_CUDA
        // Everything the GPU holds: Kokkos views, CUDA context, kernel local memory.
        std::size_t free = 0, total = 0;
        cudaMemGetInfo(&free, &total);
        std::cout << " gpu_used_MB=" << (total - free) / 1e6;
#endif
        std::cout << "\n";
        if (details)
            for (const auto& [label, bytes] : at_peak)
                std::cout << "  " << label << " " << bytes / 1e6 << " MB " << double(bytes) / n << " B/cell\n";
        peak = current;
        at_peak.clear();
        for (const auto& [p, entry] : live) at_peak[entry.first] += entry.second;
    }
};

bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i)
        if (argv[i] == flag) return true;
    return false;
}

double flag_value(int argc, char** argv, const std::string& flag, double fallback) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind(flag + "=", 0) == 0) return std::atof(arg.c_str() + flag.size() + 1);
    }
    return fallback;
}

// Positional arguments, flags removed.
std::vector<std::string> positional(int argc, char** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]).rfind("--", 0) != 0) args.push_back(argv[i]);
    return args;
}

int rebuild(const std::vector<std::string>& args, double width_factor, bool check) {
    const Exec exec;
    const int requested = std::stoi(args.at(1));
    const std::string& kind = args.at(2);
    const std::string selection = args.size() > 3 ? args[3] : "all";
    const int repeats = args.size() > 4 ? std::max(1, std::stoi(args[4])) : 5;
    const Positions x = to_device(make_points(kind, requested));
    const int n = static_cast<int>(x.extent(0));
    neighbor_search::CellList<Exec> list;
    neighbor_search::configure(list, n, 1.0, width_factor * std::cbrt(1.0 / n));
    const double build_ms = median_ms(repeats, [&] { neighbor_search::build(exec, list, x); });
    const auto grid = shell_voronoi::make_grid(list);
    DeviceMemory::phase("positions_and_cell_list", n);
    std::cout << "# backend=" << Exec::name() << " n=" << n << " distribution=" << kind << " grid="
              << list.cells_per_side() << "^3 repeats=" << repeats << "\n";
    int failures = 0;
    std::vector<double> first_volumes;
    for (const std::string variant : {"thread", "exact", "warp"}) {
        if (selection != "all" && variant.find(selection) == std::string::npos) continue;
        shell_voronoi::Cells<Memory> cells(n, true);
        const double construct_ms = median_ms(repeats, [&] {
            if (variant == "thread") shell_voronoi::construct(exec, grid, cells);
            if (variant == "exact") shell_voronoi::construct(exec, grid, cells, {}, true);
            if (variant == "warp") shell_voronoi::construct_warp(exec, grid, cells);
        });
        std::cout << variant << ": build_ms=" << build_ms << " construct_ms=" << construct_ms
                  << " construct_Mcells_s=" << n / construct_ms / 1e3
                  << " pipeline_Mcells_s=" << n / (build_ms + construct_ms) / 1e3 << "\n";
        if (check) failures += check_cells(cells, x, std::cout);
        const auto volume = to_host(cells.volume);
        if (first_volumes.empty()) {
            first_volumes.assign(volume.data(), volume.data() + n);
        } else {
            double worst = 0;
            for (int t = 0; t < n; ++t) worst = std::max(worst, std::abs(volume(t) - first_volumes[t]) / first_volumes[t]);
            std::cout << "  max_relative_volume_difference_to_first_variant=" << worst << "\n";
        }
        DeviceMemory::phase(variant, n);
    }
    return failures;
}

// Adjacency faults of a host mesh: neighbours that are dead, that do not
// share the face, or that do not point back.
void report_adjacency(const tet_flips::Mesh& m, const char* when) {
    long long dead = 0, wrong_face = 0, no_back = 0;
    for (int t = 0; t < m.count; ++t) {
        if (!m.alive(t)) continue;
        for (int k = 0; k < 4; ++k) {
            const int u = m.adjacent(t, k);
            if (u < 0) continue;
            if (!m.alive(u)) {
                ++dead;
                continue;
            }
            int shared = 0, back = 0;
            for (int a = 0; a < 4; ++a) {
                shared += a != k && tet_mesh::position(m, u, m.vertex(t, a)) >= 0;
                back += m.adjacent(u, a) == t;
            }
            wrong_face += shared != 3;
            no_back += back != 1;
        }
    }
    std::cout << "  adjacency " << when << ": to_dead=" << dead << " wrong_face=" << wrong_face << " no_back=" << no_back
              << "\n";
}

// The stuck entries the last device repair left, by reason.
void report_stuck(const tet_repair::State<Memory>& state) {
    const int stuck = Kokkos::min(state.host_counters(tet_repair::counter::stuck), static_cast<int>(state.stuck.extent(0)));
    const auto list = to_host(state.stuck);
    long long inverted = 0, unflippable = 0, rejected = 0;
    for (int i = 0; i < stuck; ++i) {
        const int k = list(i, 1);
        inverted += k == tet_repair::inverted;
        rejected += k == tet_repair::rejected;
        unflippable += k >= 0;
    }
    std::cout << "  stuck by reason: inverted=" << inverted << " unflippable=" << unflippable
              << " rejected=" << rejected << "\n";
}

int step(const std::vector<std::string>& args, bool verify, bool debug, voronoi::Fallback fallback) {
    const Exec exec;
    const int requested = std::stoi(args.at(1));
    const std::string& kind = args.at(2);
    const double displacement = std::stod(args.at(3));
    const int steps = args.size() > 4 ? std::stoi(args[4]) : 10;
    const int repeats = 3;
    // In slot order: a spatial particle order, as a hydro code keeps it.
    Points points = in_slot_order(make_points(kind, requested));
    const int n = static_cast<int>(points.size());
    const double spacing = std::cbrt(1.0 / n);
    Positions x = to_device(points);
    std::cout << "# backend=" << Exec::name() << " n=" << n << " distribution=" << kind
              << " displacement=" << displacement << " steps=" << steps
              << " fallback=" << (fallback == voronoi::Fallback::host ? "host" : "rebuild") << "\n";

    {
        // The start kernels: clipping with rings, from_rings, and the exact
        // start (tet_start::build: check, patch); build rewrites rings, so it
        // is timed together with a fresh clipping.
        const auto list = cell_list(x);
        const auto grid = shell_voronoi::make_grid(list);
        auto cells = cells_with_rings(list);  // sizes the ring pool
        tet_mesh::Mesh<Memory> mesh;
        tet_start::Stats start;
        const double clip_ms = median_ms(repeats, [&] { shell_voronoi::construct_warp(exec, grid, cells); });
        const double from_rings_ms = median_ms(repeats, [&] { tet_mesh::from_rings(exec, cells, mesh); });
        const double start_ms = median_ms(repeats, [&] {
            shell_voronoi::construct_warp(exec, grid, cells);
            start = tet_start::build(exec, grid, cells, mesh);
        });
        const int tets = static_cast<int>(mesh.vertex.extent(0));
        std::cout << "start: clip_ms=" << clip_ms << " from_rings_ms=" << from_rings_ms << " start_ms=" << start_ms
                  << " extra_vs_clip=" << (start_ms - clip_ms) / clip_ms << " exact=" << start.exact
                  << " rounds=" << start.rounds << " bad_tets=" << start.bad_tets << " patched_cells=" << start.patched
                  << " tets=" << tets << " per_cell=" << double(tets) / n
                  << " ring_entries_per_cell=" << double(cells.needed()) / n << "\n";
    }

    DeviceMemory::phase("positions_and_bench_start_kernels", n, false);
    Mesh mesh(exec, 1.0);
    mesh.set_fallback(fallback);
    {
        voronoi::Timings t;
        const voronoi::Report r = mesh.update(x, {}, &t);
        std::cout << "mesh: build_ms=" << t.build << " geometry_ms=" << t.geometry << " cells_ms=" << t.cells
                  << " total_ms=" << t.build + t.geometry + t.cells << " faces=" << r.faces
                  << " faces_per_cell=" << double(r.faces) / n << "\n";
    }
    DeviceMemory::phase("start_and_prepare", n);
    const Flow flow;
    int failures = 0;
    for (int s = 1; s <= steps; ++s) {
        flow.advect(points, displacement * spacing);
        x = to_device(points);
        exec.fence();
        voronoi::Timings t;
        const voronoi::Report r = mesh.update(x, {}, &t);
        // step_ms: the device step (repair, geometry, cells) without a fallback.
        const double step_ms = t.repair + t.geometry + t.cells;
        std::cout << "gpuflips: step=" << s << " gpu_ms=" << t.repair << " rounds=" << r.rounds << " flips=" << r.flips
                  << " stuck=" << r.stuck << " host_ms=" << t.fallback << " host_flips=" << r.host_flips
                  << " host_rebuilt=" << r.host_rebuilds << " rebuilt=" << r.rebuilt << " rebuild_ms=" << t.build
                  << " tets=" << mesh.repair_state().count << " faces=" << r.faces << " centres_faces_ms=" << t.geometry
                  << " cells_ms=" << t.cells << " step_ms=" << step_ms << " step_Mcells_s=" << n / step_ms / 1e3 << "\n";
        if (debug) {
            report_stuck(mesh.repair_state());
            report_adjacency(tet_flips::to_host<Exec>(mesh.tetrahedra(), mesh.repair_state()), "after step");
        }
        if (!verify) continue;
        const Comparison c = compare_with_rebuild(mesh, x);
        failures += !c.pass();
        std::cout << "verify: step=" << s << " " << c << (c.pass() ? "  PASS" : "  FAIL") << "\n";
    }
    DeviceMemory::phase("steps", n);
    return failures;
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    if (has_flag(argc, argv, "--memory")) DeviceMemory::start();
    const auto args = positional(argc, argv);
    if (args.size() >= 3 && args[0] == "rebuild")
        return rebuild(args, flag_value(argc, argv, "--width", 1.0), !has_flag(argc, argv, "--no-check")) != 0;
    if (args.size() >= 4 && args[0] == "step")
        return step(args, has_flag(argc, argv, "--verify"), has_flag(argc, argv, "--debug"),
                    has_flag(argc, argv, "--fallback=rebuild") ? voronoi::Fallback::rebuild : voronoi::Fallback::host) !=
               0;
    std::cerr << "usage: voronoi_bench rebuild <n> <distribution> [variants] [repeats] [--width=f] [--no-check] [--memory]\n"
                 "       voronoi_bench step <n> <distribution> <displacement> [steps] [--verify] [--debug]\n"
                 "                          [--fallback=rebuild] [--memory]\n";
    return 2;
}
