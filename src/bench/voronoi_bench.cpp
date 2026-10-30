// Timing and debugging of the mesh pipeline (not a test).
//
//   voronoi_bench rebuild <n> <distribution> [variants=all] [repeats=5] [--width=<factor>]
//       Full rebuild by clipping, double. Variants thread and warp, selected
//       by substring. Every variant's cells are checked
//       (support::check_cells, PASS/FAIL).
//   voronoi_bench step <n> <distribution> <displacement> [steps=10] [--verify] [--debug]
//       The persistent mesh: the exact start (clipping with rings,
//       tet_start::build) and the per-step kernels on it, then `steps` steps of the flow, `displacement`
//       spacings each: device repair (screened), host fallback when stuck,
//       circumcentres, face rings and faces. --verify compares every step
//       with an exact rebuild (faces the clipping lacks must have area below
//       1e-9 spacings^2); --debug adds adjacency and stuck details.
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

int rebuild(const std::vector<std::string>& args, double width_factor) {
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
    for (const std::string variant : {"thread-double", "warp-double"}) {
        if (selection != "all" && variant.find(selection) == std::string::npos) continue;
        shell_voronoi::Cells<Memory> cells(n, true);
        const double construct_ms = median_ms(repeats, [&] {
            if (variant == "thread-double") shell_voronoi::construct<double>(exec, grid, cells);
            if (variant == "warp-double") shell_voronoi::construct_warp<double>(exec, grid, cells);
        });
        std::cout << variant << ": build_ms=" << build_ms << " construct_ms=" << construct_ms
                  << " construct_Mcells_s=" << n / construct_ms / 1e3
                  << " pipeline_Mcells_s=" << n / (build_ms + construct_ms) / 1e3 << "\n";
        failures += check_cells(cells, list.positions, true, std::cout);
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

void report_stuck(const tet_repair::State<Memory>& state, int stuck) {
    const auto list = to_host(state.stuck);
    long long inverted = 0, unflippable = 0, rejected = 0;
    for (int i = 0; i < stuck; ++i) {
        const int k = list(i, 1);
        inverted += k == -1;
        rejected += k == -2;
        unflippable += k >= 0;
    }
    std::cout << "  stuck by reason: inverted=" << inverted << " unflippable=" << unflippable
              << " rejected=" << rejected << "\n";
}

int step(const std::vector<std::string>& args, bool verify, bool debug) {
    const Exec exec;
    const int requested = std::stoi(args.at(1));
    const std::string& kind = args.at(2);
    const double displacement = std::stod(args.at(3));
    const int steps = args.size() > 4 ? std::stoi(args[4]) : 10;
    const int repeats = 3;
    Points points = in_slot_order(make_points(kind, requested));
    const int n = static_cast<int>(points.size());
    const double spacing = std::cbrt(1.0 / n);
    Positions x = to_device(points);
    std::cout << "# backend=" << Exec::name() << " n=" << n << " distribution=" << kind
              << " displacement=" << displacement << " steps=" << steps << "\n";

    {
        // The start: clipping with rings, from_rings, and the exact start
        // (tet_start::build: check, patch); build rewrites rings, so it is
        // timed together with a fresh clipping.
        const auto list = cell_list(x);
        const auto grid = shell_voronoi::make_grid(list);
        shell_voronoi::Cells<Memory> cells(n, false);
        cells.rings = Kokkos::View<int**, Memory>("rings", n, 384);
        tet_mesh::Mesh<Memory> mesh;
        tet_start::Stats start;
        const double clip_ms = median_ms(repeats, [&] { shell_voronoi::construct<double>(exec, grid, cells); });
        const double from_rings_ms = median_ms(repeats, [&] { tet_mesh::from_rings(exec, cells, mesh); });
        const double start_ms = median_ms(repeats, [&] {
            shell_voronoi::construct<double>(exec, grid, cells);
            start = tet_start::build(exec, grid, cells, mesh);
        });
        const int tets = static_cast<int>(mesh.vertex.extent(0));
        const int faces = static_cast<int>(mesh.owner.extent(0));
        tet_mesh::Points4<Memory> padded("padded", n);
        tet_mesh::pad(exec, x, padded);
        const tet_mesh::Geometry<Memory> geometry{Kokkos::View<double* [3], Memory>("area", faces),
                                                  Kokkos::View<double* [3], Memory>("centroid", faces),
                                                  Kokkos::View<double*, Memory>("volume", n)};
        Kokkos::View<int*, Memory> invalid("invalid", tets);
        int failed = 0;
        const double centres_ms = median_ms(repeats, [&] { tet_mesh::circumcentres(exec, padded, 1.0, mesh); });
        const double faces_ms = median_ms(repeats, [&] { tet_mesh::faces(exec, padded, 1.0, mesh, geometry); });
        const double check_ms = median_ms(repeats, [&] { failed = tet_mesh::check(exec, padded, 1.0, mesh, invalid); });
        const double per_step_ms = centres_ms + faces_ms + check_ms;
        std::cout << "start: clip_ms=" << clip_ms << " from_rings_ms=" << from_rings_ms << " start_ms=" << start_ms
                  << " extra_vs_clip=" << (start_ms - clip_ms) / clip_ms << " exact=" << start.exact
                  << " rounds=" << start.rounds << " bad_tets=" << start.bad_tets << " patched_cells=" << start.patched
                  << " tets=" << tets << " per_cell=" << double(tets) / n << "\n";
        std::cout << "tets: faces_per_cell=" << double(faces) / n << " centres_ms=" << centres_ms
                  << " faces_ms=" << faces_ms << " check_ms=" << check_ms << " step_Mcells_s=" << n / per_step_ms / 1e3
                  << " invalid=" << failed << "\n";
    }

    DeviceMemory::phase("positions_and_bench_start_kernels", n, false);
    MovingMesh moving(x);
    DeviceMemory::phase("start_and_prepare", n);
    const Flow flow;
    int failures = 0;
    for (int s = 1; s <= steps; ++s) {
        flow.advect(points, displacement * spacing);
        x = to_device(points);
        exec.fence();
        Kokkos::Timer timer;
        const auto device = moving.repair(x);
        exec.fence();
        const double gpu_ms = timer.seconds() * 1e3;
        if (debug && device.stuck > 0) {
            report_stuck(moving.state, device.stuck);
            report_adjacency(tet_flips::to_host<Exec>(moving.mesh, moving.state), "after device repair");
        }
        timer.reset();
        const auto host = device.stuck > 0 ? moving.finish() : tet_flips::Stats{};
        const double host_ms = timer.seconds() * 1e3;
        timer.reset();
        const int faces = moving.update_geometry();
        exec.fence();
        const double faces_ms = timer.seconds() * 1e3;
        std::cout << "gpuflips: step=" << s << " gpu_ms=" << gpu_ms << " gpu_Mcells_s=" << n / gpu_ms / 1e3
                  << " rounds=" << device.rounds << " flips23=" << device.flips23 << " flips32=" << device.flips32
                  << " stuck=" << device.stuck << " host_ms=" << host_ms << " host_flips=" << host.flips23 + host.flips32
                  << " host_rebuilt=" << host.rebuilt << " tets=" << moving.state.count << " faces=" << faces
                  << " centres_faces_ms=" << faces_ms << " step_ms=" << gpu_ms + faces_ms
                  << " step_Mcells_s=" << n / (gpu_ms + faces_ms) / 1e3 << "\n";
        if (!verify) continue;

        const tet_flips::Mesh mesh = tet_flips::to_host<Exec>(moving.mesh, moving.state);
        if (debug) report_adjacency(mesh, "after step");
        const long long failing = tet_flips::failing(mesh);
        const Reference exact = clipped(x);
        const NeighbourSets neighbours = delaunay_neighbours(mesh, n);
        long long mismatched = 0, exact_faces = 0, missing = 0, extra = 0;
        for (int i = 0; i < n; ++i) {
            mismatched += neighbours[i] != exact.neighbours[i];
            exact_faces += static_cast<long long>(exact.neighbours[i].size());
        }
        // Faces the clipping lacks must be rounding-level: clipping drops
        // faces below 64 eps R^2, the mesh keeps every Delaunay edge.
        double extra_area = 0;
        {
            const auto owner = to_host(moving.mesh.owner);
            const auto neighbour = to_host(moving.mesh.neighbour);
            const auto area = to_host(moving.geometry.area);
            for (int f = 0; f < faces; ++f) {
                const int i = owner(f), j = neighbour(f);
                if (std::binary_search(exact.neighbours[i].begin(), exact.neighbours[i].end(), j)) continue;
                ++extra;
                extra_area = std::max(extra_area, std::sqrt(area(f, 0) * area(f, 0) + area(f, 1) * area(f, 1) +
                                                            area(f, 2) * area(f, 2)) / (spacing * spacing));
            }
            for (int i = 0; i < n; ++i)
                for (int j : exact.neighbours[i])
                    missing += i < j && !std::binary_search(neighbours[i].begin(), neighbours[i].end(), j);
        }
        const auto volume = to_host(moving.geometry.volume);
        double worst = 0, total = 0;
        int worst_cell = -1;
        for (int i = 0; i < n; ++i) {
            total += volume(i);
            const double error = std::abs(volume(i) - exact.volume[i]) / exact.volume[i];
            if (exact.volume[i] > 0 && error > worst) {
                worst = error;
                worst_cell = i;
            }
        }
        const bool pass = failing == 0 && missing == 0 && extra_area < 1e-9 && worst < 1e-10;
        failures += !pass;
        std::cout << "verify: step=" << s << " failing=" << failing << " mismatched_cells=" << mismatched
                  << " missing_faces=" << missing << " extra_faces=" << extra << " extra_max_area=" << extra_area
                  << " faces=" << faces << " exact_faces=" << exact_faces / 2 << " total_volume=" << total
                  << " max_relative_volume_difference=" << worst << (pass ? "  PASS" : "  FAIL") << "\n";
        if (debug && worst_cell >= 0)
            std::cout << "  worst cell " << worst_cell << " mesh=" << volume(worst_cell)
                      << " reference=" << exact.volume[worst_cell] << " reference_status=" << exact.status[worst_cell]
                      << " reference_faces=" << exact.neighbours[worst_cell].size()
                      << " mesh_faces=" << neighbours[worst_cell].size() << "\n";
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
        return rebuild(args, flag_value(argc, argv, "--width", 1.0)) != 0;
    if (args.size() >= 4 && args[0] == "step")
        return step(args, has_flag(argc, argv, "--verify"), has_flag(argc, argv, "--debug")) != 0;
    std::cerr << "usage: voronoi_bench rebuild <n> <distribution> [variants] [repeats] [--width=f] [--memory]\n"
                 "       voronoi_bench step <n> <distribution> <displacement> [steps] [--verify] [--debug] [--memory]\n";
    return 2;
}
