// Bench for the mesh pipeline (the voronoi::Mesh stage of a solver step), not a test: `rebuild` and `step`;
// flags in docs/design.md. step mirrors a moving run: advect points -> Mesh::update -> update_cells.

#include "device_memory.hpp"
#include "support.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#if defined(VORONOI_MEASURE)
#include "measure.hpp"
#endif
#if defined(KOKKOS_ENABLE_CUDA)
#include <nvtx3/nvToolsExt.h>
#endif

namespace {

using namespace support;

// Time per kernel label through the Kokkos Tools callbacks. Used by: main (start() with --kernels), step (report).
struct KernelTimes {
    static inline bool on = false;
    static inline std::map<std::string, double> total;  // ms over the sampled steps
    static inline std::string running;
    static inline Kokkos::Timer timer;

    static void begin(const char* label, std::uint32_t, std::uint64_t*) {
        running = label;
        timer.reset();
    }
    static void end(std::uint64_t) { total[running] += timer.seconds() * 1e3; }
    static void begin_copy(Kokkos_Profiling_SpaceHandle, const char*, const void*, Kokkos_Profiling_SpaceHandle, const char*,
                           const void*, std::uint64_t) {
        running = "deep_copy";
        timer.reset();
    }
    static void end_copy() { total[running] += timer.seconds() * 1e3; }
    static void start() {
        on = true;
        namespace E = Kokkos::Tools::Experimental;
        E::set_begin_parallel_for_callback(begin);
        E::set_end_parallel_for_callback(end);
        E::set_begin_parallel_scan_callback(begin);
        E::set_end_parallel_scan_callback(end);
        E::set_begin_parallel_reduce_callback(begin);
        E::set_end_parallel_reduce_callback(end);
        E::set_begin_deep_copy_callback(begin_copy);
        E::set_end_deep_copy_callback(end_copy);
    }
    // Prints the slowest kernels per step over `steps` sampled steps, for n cells. Used by: step.
    static void report(int steps, int n) {
        if (!on) return;
        std::vector<std::pair<double, std::string>> rows;
        for (const auto& [label, ms] : total) rows.push_back({ms / steps, label});
        std::sort(rows.rbegin(), rows.rend());
        for (const auto& [ms, label] : rows)
            std::cout << "kernel: " << label << " ms_per_step=" << ms << " Mcells_s=" << n / ms / 1e3 << "\n";
    }
};

// --nvtx: an NVTX range per kernel label and per phase, so Nsight names and filters kernels by label.
// Used by: main (start()), rebuild and step (Phase).
struct Nvtx {
    static inline bool on = false;
    static void push(const char* label) {
#if defined(KOKKOS_ENABLE_CUDA)
        if (on) nvtxRangePushA(label);
#else
        (void)label;
#endif
    }
    static void pop() {
#if defined(KOKKOS_ENABLE_CUDA)
        if (on) nvtxRangePop();
#endif
    }
    static void begin(const char* label, std::uint32_t, std::uint64_t*) { push(label); }
    static void end(std::uint64_t) { pop(); }
    // Labels need Kokkos Tools callbacks, which fence every kernel; --nvtx=phases leaves them out (nsys counts).
    static void start(bool labels) {
        on = true;
        if (!labels) return;
        namespace E = Kokkos::Tools::Experimental;
        E::set_begin_parallel_for_callback(begin);
        E::set_end_parallel_for_callback(end);
        E::set_begin_parallel_scan_callback(begin);
        E::set_end_parallel_scan_callback(end);
        E::set_begin_parallel_reduce_callback(begin);
        E::set_end_parallel_reduce_callback(end);
    }
};

// A phase range (fenced at both ends) while --nvtx is on. Used by: rebuild, step.
struct Phase {
    explicit Phase(const std::string& name) {
        if (!Nvtx::on) return;
        Kokkos::fence();
        Nvtx::push(name.c_str());
    }
    ~Phase() {
        if (!Nvtx::on) return;
        Kokkos::fence();
        Nvtx::pop();
    }
};

// Whether the bare flag is present. Used by: main.
bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i)
        if (argv[i] == flag) return true;
    return false;
}

// Value of `--flag=value`, or fallback. Used by: main.
double flag_value(int argc, char** argv, const std::string& flag, double fallback) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind(flag + "=", 0) == 0) return std::atof(arg.c_str() + flag.size() + 1);
    }
    return fallback;
}

// --box=Lx,Ly,Lz (or one side, a cube); the unit cube by default. Used by: main.
exact_predicates::Box box_flag(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--box=", 0) != 0) continue;
        double side[3];
        int count = 0;
        for (std::size_t at = 6; at <= arg.size() && count < 3; ++count) {
            const std::size_t end = std::min(arg.find(',', at), arg.size());
            side[count] = std::stod(arg.substr(at, end - at));
            at = end + 1;
        }
        return count == 1 ? exact_predicates::Box(side[0]) : exact_predicates::Box(side[0], side[1], side[2]);
    }
    return {};
}

// Value of --dump=path, or "". Used by: main (rebuild --measure).
std::string dump_flag(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]).rfind("--dump=", 0) == 0) return argv[i] + 7;
    return "";
}

// Positional arguments, flags removed. Used by: main.
std::vector<std::string> positional(int argc, char** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]).rfind("--", 0) != 0) args.push_back(argv[i]);
    return args;
}

// The `rebuild` mode: times the cell list and the warp clipping on one point set and checks the cells.
// args: rebuild <n> <distribution> [repeats]. Returns the failure count.
// Used by: main.
int rebuild(const std::vector<std::string>& args, const Box& box, bool check, int measure_every,
            const std::string& dump) {
    const Exec exec;
    const int requested = std::stoi(args.at(1));
    const std::string& kind = args.at(2);
    const int repeats = args.size() > 3 ? std::max(1, std::stoi(args[3])) : 5;
    const Positions x = to_device(make_points(kind, requested, box));
    const int n = static_cast<int>(x.extent(0));
    neighbor_search::CellList<Exec> list;
    const double width = box.cubic() ? box.side[0] * std::cbrt(1.0 / n) : std::cbrt(box.volume() / n);
    neighbor_search::configure(list, n, box, width);
    const double build_ms = median_ms(repeats, [&] {
        const Phase phase("list");
        neighbor_search::build(exec, list, x);
    });
    const auto grid = shell_voronoi::make_grid(list);
    DeviceMemory<Memory>::phase("positions_and_cell_list", n);
    std::cout << "# backend=" << Exec::name() << " n=" << n << " distribution=" << kind << " box=" << box.side[0] << ","
              << box.side[1] << "," << box.side[2] << " tiles=" << list.tiles[0] << "x" << list.tiles[1] << "x"
              << list.tiles[2] << " grid=" << (1 << list.levels) << "^3_per_tile repeats=" << repeats << "\n";
    int failures = 0;
    shell_voronoi::Cells<Memory> cells(n, true);
    const double construct_ms = median_ms(repeats, [&] {
        const Phase phase("clip");
        shell_voronoi::construct_warp(exec, grid, cells);
    });
    std::cout << "warp: build_ms=" << build_ms << " construct_ms=" << construct_ms
              << " construct_Mcells_s=" << n / construct_ms / 1e3
              << " pipeline_Mcells_s=" << n / (build_ms + construct_ms) / 1e3 << "\n";
    if (check) failures += check_cells(cells, x, std::cout, box);
    DeviceMemory<Memory>::phase("warp", n);
#if defined(VORONOI_MEASURE)
    if (measure_every > 0) measure_bench::clipping(exec, grid, box, measure_every, dump);
#else
    (void)measure_every, (void)dump;
#endif
    return failures;
}

// A checksum of the per-cell face lists and the bits of the cell volumes
// and centroids (FNV-1a), to compare builds of the same mesh. Used by: step (build report).
std::uint64_t cells_checksum(const Mesh& mesh) {
    const auto start = to_host(mesh.cell_face_start());
    const auto face = to_host(mesh.cell_face());
    const auto volume = to_host(mesh.volume());
    const auto centroid = to_host(mesh.centroid());
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](std::uint64_t v) {
        for (int b = 0; b < 8; ++b) h = (h ^ ((v >> (8 * b)) & 255)) * 1099511628211ull;
    };
    auto bits = [](double d) {
        std::uint64_t u;
        std::memcpy(&u, &d, sizeof u);
        return u;
    };
    const int n = mesh.particles();
    for (int i = 0; i < n; ++i) {
        mix(static_cast<std::uint64_t>(start(i + 1)));
        for (int a = start(i); a < start(i + 1); ++a) mix(static_cast<std::uint64_t>(face(a)));
        mix(bits(volume(i)));
        for (int a = 0; a < 3; ++a) mix(bits(centroid(i, a)));
    }
    return h;
}

// A checksum of the bits of the cell volumes and centroids. Used by: step (--checksum).
std::uint64_t geometry_checksum(const Mesh& mesh) {
    const auto volume = to_host(mesh.volume());
    const auto centroid = to_host(mesh.centroid());
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](double d) {
        std::uint64_t u;
        std::memcpy(&u, &d, sizeof u);
        h = (h ^ u) * 1099511628211ull;
    };
    for (int i = 0; i < mesh.particles(); ++i) {
        mix(volume(i));
        for (int a = 0; a < 3; ++a) mix(centroid(i, a));
    }
    return h;
}

// The `step` mode: build the mesh, then `steps` times advect the points, Mesh::update and update_cells, timing the
// phases. args: step <n> <distribution> <displacement in spacings> [steps]. Returns the failure count (--verify).
// Used by: main.
int step(const std::vector<std::string>& args, const Box& box, bool verify, bool checksum, bool measure) {
    const Exec exec;
    const int requested = std::stoi(args.at(1));
    const std::string& kind = args.at(2);
    const double displacement = std::stod(args.at(3));
    const int steps = args.size() > 4 ? std::stoi(args[4]) : 10;
    const int repeats = 3;
    // In slot order: a spatial particle order, as a solver keeps it.
    Points points = in_slot_order(make_points(kind, requested, box), box);
    const int n = static_cast<int>(points.size());
    const double spacing = box.cubic() ? box.side[0] * std::cbrt(1.0 / n) : std::cbrt(box.volume() / n);
    Positions x = to_device(points);
    std::cout << "# backend=" << Exec::name() << " n=" << n << " distribution=" << kind << " box=" << box.side[0]
              << "," << box.side[1] << "," << box.side[2] << " displacement=" << displacement << " steps=" << steps
              << "\n";

    {
        // The start kernels: clipping with rings, from_rings and the exact start; tet_start::build rewrites
        // rings, so it is timed with a fresh clipping.
        const auto list = cell_list(x, box);
        const auto grid = shell_voronoi::make_grid(list);
        auto cells = cells_with_rings(list);
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

    DeviceMemory<Memory>::phase("positions_and_bench_start_kernels", n, false);
    Mesh mesh(exec, box);
    {
        voronoi::Timings t;
        const voronoi::Report r = [&] {
            const Phase phase("start");
            return mesh.update(x, &t);
        }();
        {
            const Phase phase("start_cells");
            mesh.update_cells(&t);
        }
        std::cout << "mesh: build_ms=" << t.build << " geometry_ms=" << t.geometry << " cells_ms=" << t.cells
                  << " total_ms=" << t.build + t.geometry << " faces=" << r.faces
                  << " faces_per_cell=" << double(r.faces) / n << " cells_checksum=" << cells_checksum(mesh) << "\n";
    }
    DeviceMemory<Memory>::phase("start_and_prepare", n);
    const Flow flow;
    int failures = 0;
#if defined(VORONOI_MEASURE)
    std::vector<std::vector<int>> sets_before;
    if (measure) measure_bench::step(mesh, sets_before, 0);
#else
    (void)measure;
#endif
    for (int s = 1; s <= steps; ++s) {
        flow.advect(points, displacement * spacing, box);
        x = to_device(points);
        exec.fence();
        if (s == 2) KernelTimes::total.clear();
        voronoi::Timings t;
        const voronoi::Report r = [&] {
            const Phase phase("s" + std::to_string(s) + "_update");
            return mesh.update(x, &t);
        }();
        {
            const Phase phase("s" + std::to_string(s) + "_cells");
            mesh.update_cells(&t);
        }
        // step_ms: the device step (repair, geometry) without a fallback; cells_ms (face lists, volumes, centroids) is on top.
        const double step_ms = t.repair + t.geometry;
        std::cout << "gpuflips: step=" << s << " gpu_ms=" << t.repair << " rounds=" << r.rounds << " flips=" << r.flips
                  << " stuck=" << r.stuck << " host_ms=" << t.fallback << " host_flips=" << r.host_flips
                  << " host_rebuilt=" << r.host_rebuilds << " rebuilt=" << r.rebuilt << " rebuild_ms=" << t.build
                  << " tets=" << mesh.repair_state().count << " faces=" << r.faces << " centres_faces_ms=" << t.geometry
                  << " cells_ms=" << t.cells << " step_ms=" << step_ms << " step_Mcells_s=" << n / step_ms / 1e3 << "\n";
#if defined(VORONOI_MEASURE)
        if (measure) measure_bench::step(mesh, sets_before, s);
#endif
        if (checksum) std::cout << "geometry_checksum: step=" << s << " " << geometry_checksum(mesh) << "\n";
        if (!verify) continue;
        const Comparison c = compare_with_rebuild(mesh, x);
        failures += !c.pass();
        std::cout << "verify: step=" << s << " " << c << (c.pass() ? "  PASS" : "  FAIL") << "\n";
    }
    DeviceMemory<Memory>::phase("steps", n);
    KernelTimes::report(steps - 1, n);
    return failures;
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    if (has_flag(argc, argv, "--memory")) DeviceMemory<Memory>::start();
    if (has_flag(argc, argv, "--kernels")) KernelTimes::start();
    if (has_flag(argc, argv, "--nvtx") || has_flag(argc, argv, "--nvtx=phases"))
        Nvtx::start(has_flag(argc, argv, "--nvtx"));
    const auto args = positional(argc, argv);
    if (args.size() >= 3 && args[0] == "rebuild")
        return rebuild(args, box_flag(argc, argv), !has_flag(argc, argv, "--no-check"),
                       static_cast<int>(flag_value(argc, argv, "--measure", 0)), dump_flag(argc, argv)) != 0;
    if (args.size() >= 4 && args[0] == "step")
        return step(args, box_flag(argc, argv), has_flag(argc, argv, "--verify"), has_flag(argc, argv, "--checksum"),
                    has_flag(argc, argv, "--measure")) != 0;
    std::cerr << "usage: voronoi_bench rebuild <n> <distribution> [repeats] [--no-check] [--memory]\n"
                 "                          [--measure=every --dump=file.npy] (VORONOI_MEASURE builds)\n"
                 "       voronoi_bench step <n> <distribution> <displacement> [steps] [--verify] [--checksum]\n"
                 "                          [--memory] [--kernels] [--measure]\n"
                 "       both: [--nvtx | --nvtx=phases] (CUDA: NVTX ranges per phase, and per kernel label)\n"
                 "       both: [--box=Lx,Ly,Lz]\n";
    return 2;
}
