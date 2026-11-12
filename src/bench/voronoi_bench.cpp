// Bench for the mesh pipeline (the voronoi::Mesh stage of a solver step), not a test: `rebuild` and `step`;
// flags in docs/design.md. step mirrors a moving run: advect points -> Mesh::update -> update_cells.

#include "device_memory.hpp"
#include "support.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <string>
#include <thread>
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

// Value of `--flag=value` as text, or fallback. Used by: main.
std::string text_flag(int argc, char** argv, const std::string& flag, const std::string& fallback) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind(flag + "=", 0) == 0) return arg.substr(flag.size() + 1);
    }
    return fallback;
}

// The mesh layout of --curve, --every, --far, --owner-faces, --owner-bit. Used by: main.
voronoi::Layout layout_flags(int argc, char** argv) {
    voronoi::Layout layout;
    const std::string curve = text_flag(argc, argv, "--curve", "");
    const std::pair<const char*, voronoi::Curve> names[] = {
        {"morton", voronoi::Curve::morton}, {"hilbert", voronoi::Curve::hilbert}, {"moore", voronoi::Curve::moore},
        {"gray", voronoi::Curve::gray},     {"row", voronoi::Curve::row_major},   {"rcm", voronoi::Curve::rcm}};
    bool known = curve.empty();
    for (const auto& [name, c] : names)
        if (curve == name) layout.curve = c, known = true;
    if (!known) throw std::invalid_argument("--curve: morton, hilbert, moore, gray, row or rcm");
    layout.every = static_cast<int>(flag_value(argc, argv, "--every", 0));
    layout.far = flag_value(argc, argv, "--far", 0);
    for (int i = 1; i < argc; ++i) {
        layout.owner_faces |= std::string(argv[i]) == "--owner-faces";
        layout.owner_bit |= std::string(argv[i]) == "--owner-bit";
    }
    return layout;
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
        for (int a = start(i); a < start(i + 1); ++a) mix(static_cast<std::uint64_t>(Mesh::list_face(face(a))));
        mix(bits(volume(i)));
        for (int a = 0; a < 3; ++a) mix(bits(centroid(i, a)));
    }
    return h;
}

// Flow::advect on `threads` host threads, one block of points each (the same result). Used by: step.
void advect(const Flow& flow, Points& points, double distance, const Box& box, int threads) {
    std::vector<std::thread> pool;
    const std::size_t block = (points.size() + threads - 1) / threads;
    for (std::size_t begin = 0; begin < points.size(); begin += block)
        pool.emplace_back([&, begin] {
            const std::size_t end = std::min(points.size(), begin + block);
            Points part(points.begin() + begin, points.begin() + end);
            flow.advect(part, distance, box);
            std::copy(part.begin(), part.end(), points.begin() + begin);
        });
    for (auto& t : pool) t.join();
}

// The q-quantile of v (reordered). Used by: report_order.
template <class T>
T quantile(std::vector<T>& v, double q) {
    if (v.empty()) return T{};
    const auto at = v.begin() + static_cast<std::ptrdiff_t>(q * double(v.size() - 1));
    std::nth_element(v.begin(), at, v.end());
    return *at;
}

// Memory order of faces, lists, tetrahedra and cells (#34): owner step between consecutive faces, |owner - neighbour|,
// list span (largest minus smallest face of a cell), 128-byte lines of `area` rows per list entry, the step of the
// lowest corner between consecutive live tetrahedra, and each cell's distance from its place on the Morton curve.
// Used by: step (--order).
void report_order(const Mesh& mesh, int step) {
    const int n = mesh.particles(), faces = mesh.faces();
    const auto owner = to_host(mesh.owner(), faces);
    const auto neighbour = to_host(mesh.neighbour(), faces);
    const auto start = to_host(mesh.cell_face_start());
    const auto list = to_host(mesh.cell_face(), 2 * static_cast<std::size_t>(faces));
    std::vector<long long> owner_step(faces > 0 ? faces - 1 : 0), pair(faces), span(n);
    long long near_owner = 0, near_pair = 0, lines = 0;
    for (int f = 0; f + 1 < faces; ++f) owner_step[f] = std::abs(owner(f + 1) - owner(f));
    for (int f = 0; f < faces; ++f) pair[f] = std::abs(neighbour(f) - owner(f));
    for (long long s : owner_step) near_owner += s <= 64;
    for (long long s : pair) near_pair += s <= 64;
    std::vector<long long> line;
    for (int i = 0; i < n; ++i) {
        int low = faces, high = -1;
        line.clear();
        for (int k = start(i); k < start(i + 1); ++k) {
            const int f = Mesh::list_face(list(k));
            low = std::min(low, f);
            high = std::max(high, f);
            line.push_back(24LL * f / 128);
        }
        std::sort(line.begin(), line.end());
        lines += std::unique(line.begin(), line.end()) - line.begin();
        span[i] = high - low;
    }
    const auto& state = mesh.repair_state();
    const auto alive = to_host(state.alive);
    const auto vertex = to_host(mesh.tetrahedra().vertex);
    std::vector<long long> tet_step;
    long long near_tet = 0, previous = -1;
    for (int t = 0; t < state.count; ++t) {
        if (!alive(t)) continue;
        const long long low = std::min({vertex(t, 0), vertex(t, 1), vertex(t, 2), vertex(t, 3)});
        if (previous >= 0) {
            tet_step.push_back(std::abs(low - previous));
            near_tet += tet_step.back() <= 64;
        }
        previous = low;
    }
    const auto x = to_host(mesh.positions());
    const Box box = mesh.sides();
    std::vector<std::pair<std::uint64_t, int>> keyed(n);
    for (int i = 0; i < n; ++i) {
        std::uint64_t c[3];
        for (int a = 0; a < 3; ++a) c[a] = static_cast<std::uint64_t>(std::llround(x(i, a) / box.step)) >> 9;
        keyed[i] = {neighbor_search::morton(c[0], c[1], c[2]), i};
    }
    std::sort(keyed.begin(), keyed.end());
    std::vector<long long> place(n);
    for (int q = 0; q < n; ++q) place[q] = std::abs(static_cast<long long>(keyed[q].second) - q);
    const double entries = 2.0 * faces;
    std::cout << "order: step=" << step << " owner_step_p50=" << quantile(owner_step, 0.5)
              << " owner_step_p90=" << quantile(owner_step, 0.9)
              << " owner_step_within_64=" << near_owner / std::max(1.0, faces - 1.0)
              << " pair_p50=" << quantile(pair, 0.5) << " pair_p90=" << quantile(pair, 0.9)
              << " pair_within_64=" << near_pair / std::max(1.0, double(faces))
              << " list_span_p50=" << quantile(span, 0.5) << " list_span_p90=" << quantile(span, 0.9)
              << " lines_per_entry=" << lines / entries << " tet_step_p50=" << quantile(tet_step, 0.5)
              << " tet_step_within_64=" << near_tet / std::max(1.0, double(tet_step.size()))
              << " morton_place_p50=" << quantile(place, 0.5) << " morton_place_p90=" << quantile(place, 0.9) << "\n";
}

// Per particle as first numbered (label), after the last step: volume, centroid relative to the generator, and an
// FNV-1a hash of its sorted neighbours' labels, as two .npy files, to compare runs whatever their order (--dump-cells).
// Used by: step.
void dump_cells(const Mesh& mesh, const std::vector<int>& label, const std::string& path) {
    const int n = mesh.particles(), faces = mesh.faces();
    const auto start = to_host(mesh.cell_face_start());
    const auto list = to_host(mesh.cell_face(), 2 * static_cast<std::size_t>(faces));
    const auto owner = to_host(mesh.owner(), faces);
    const auto neighbour = to_host(mesh.neighbour(), faces);
    const auto volume = to_host(mesh.volume());
    const auto centroid = to_host(mesh.centroid());
    std::vector<double> geometry(4 * static_cast<std::size_t>(n));
    std::vector<std::uint64_t> hashes(n);
    std::vector<int> others;
    for (int i = 0; i < n; ++i) {
        const std::size_t row = 4 * static_cast<std::size_t>(label[i]);
        geometry[row] = volume(i);
        for (int a = 0; a < 3; ++a) geometry[row + 1 + a] = centroid(i, a);
        others.clear();
        for (int k = start(i); k < start(i + 1); ++k) {
            const int f = Mesh::list_face(list(k));
            others.push_back(label[owner(f) == i ? neighbour(f) : owner(f)]);
        }
        std::sort(others.begin(), others.end());
        std::uint64_t h = 1469598103934665603ull;
        for (int o : others) h = (h ^ static_cast<std::uint64_t>(o)) * 1099511628211ull;
        hashes[label[i]] = h;
    }
    auto write = [&](const std::string& name, const char* type, int columns, const void* data, std::size_t bytes) {
        std::string header = "{'descr': '" + std::string(type) + "', 'fortran_order': False, 'shape': (" +
                             std::to_string(n) + (columns > 1 ? ", " + std::to_string(columns) + ")" : ",)") + ", }";
        header.append(63 - (10 + header.size()) % 64, ' ');
        header += '\n';
        std::ofstream out(name, std::ios::binary);
        const unsigned short length = static_cast<unsigned short>(header.size());
        out.write("\x93NUMPY\x01\x00", 8);
        out.write(reinterpret_cast<const char*>(&length), 2);
        out << header;
        out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    };
    write(path + ".geometry.npy", "<f8", 4, geometry.data(), geometry.size() * sizeof(double));
    write(path + ".neighbours.npy", "<u8", 1, hashes.data(), hashes.size() * sizeof(std::uint64_t));
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
int step(const std::vector<std::string>& args, const Box& box, bool verify, bool checksum, bool measure,
         int order_every, const voronoi::Layout& layout, const std::string& dump) {
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
    mesh.set_layout(layout);
    // label[i]: the particle cell i was first; the points follow every re-sort.
    std::vector<int> label(n);
    std::iota(label.begin(), label.end(), 0);
    auto follow = [&](const voronoi::Report& r) {
        if (!r.resorted) return;
        const auto p = to_host(mesh.permutation());
        Points moved(n);
        std::vector<int> relabel(n);
        for (int i = 0; i < n; ++i) moved[i] = points[p(i)], relabel[i] = label[p(i)];
        points.swap(moved);
        label.swap(relabel);
    };
    std::cout << "layout: curve=" << static_cast<int>(layout.curve) << " every=" << layout.every
              << " far=" << layout.far << " owner_faces=" << layout.owner_faces << " owner_bit=" << layout.owner_bit
              << "\n";
    {
        voronoi::Timings t;
        const voronoi::Report r = [&] {
            const Phase phase("start");
            return mesh.update(x, &t);
        }();
        follow(r);
        {
            const Phase phase("start_cells");
            mesh.update_cells(&t);
        }
        std::cout << "mesh: build_ms=" << t.build << " resort_ms=" << t.resort << " geometry_ms=" << t.geometry
                  << " cells_ms=" << t.cells << " total_ms=" << t.build + t.resort + t.geometry << " faces=" << r.faces
                  << " faces_per_cell=" << double(r.faces) / n << " cells_checksum=" << cells_checksum(mesh) << "\n";
    }
    DeviceMemory<Memory>::phase("start_and_prepare", n);
    const Flow flow;
    int failures = 0;
    if (order_every > 0) report_order(mesh, 0);
#if defined(VORONOI_MEASURE)
    std::vector<std::vector<int>> sets_before;
    if (measure) measure_bench::step(mesh, sets_before, 0);
#else
    (void)measure;
#endif
    for (int s = 1; s <= steps; ++s) {
        Kokkos::Timer moving;
        advect(flow, points, displacement * spacing, box,
               static_cast<int>(std::clamp(std::thread::hardware_concurrency(), 1u, 16u)));
        const double advect_ms = moving.seconds() * 1e3;
        x = to_device(points);
        exec.fence();
        const double copy_ms = moving.seconds() * 1e3 - advect_ms;
        if (s == 2) KernelTimes::total.clear();
        voronoi::Timings t;
        const voronoi::Report r = [&] {
            const Phase phase("s" + std::to_string(s) + "_update");
            return mesh.update(x, &t);
        }();
        follow(r);
        {
            const Phase phase("s" + std::to_string(s) + "_cells");
            mesh.update_cells(&t);
        }
        // step_ms: the device step (repair, geometry) without a fallback; cells_ms (face lists, volumes, centroids) is on
        // top, and all_ms adds everything (fallback, rebuild, re-sort).
        const double step_ms = t.repair + t.geometry;
        std::cout << "gpuflips: step=" << s << " gpu_ms=" << t.repair << " rounds=" << r.rounds << " flips=" << r.flips
                  << " stuck=" << r.stuck << " host_ms=" << t.fallback << " host_flips=" << r.host_flips
                  << " host_rebuilt=" << r.host_rebuilds << " rebuilt=" << r.rebuilt << " rebuild_ms=" << t.build
                  << " tets=" << mesh.repair_state().count << " faces=" << r.faces << " centres_faces_ms=" << t.geometry
                  << " cells_ms=" << t.cells << " resorted=" << r.resorted << " resort_ms=" << t.resort
                  << " step_ms=" << step_ms << " step_Mcells_s=" << n / step_ms / 1e3
                  << " all_ms=" << t.repair + t.fallback + t.build + t.resort + t.geometry + t.cells
                  << " advect_ms=" << advect_ms << " copy_ms=" << copy_ms << "\n";
        if (order_every > 0 && s % order_every == 0) report_order(mesh, s);
#if defined(VORONOI_MEASURE)
        if (measure) measure_bench::step(mesh, sets_before, s);
#endif
        if (checksum) std::cout << "geometry_checksum: step=" << s << " " << geometry_checksum(mesh) << "\n";
        if (!verify) continue;
        const Comparison c = compare_with_rebuild(mesh, to_device(points));
        failures += !c.pass();
        std::cout << "verify: step=" << s << " " << c << (c.pass() ? "  PASS" : "  FAIL") << "\n";
    }
    DeviceMemory<Memory>::phase("steps", n);
    KernelTimes::report(steps - 1, n);
    if (!dump.empty()) dump_cells(mesh, label, dump);
    {  // A rebuild of the last positions in the order the run left them (and its re-sort, if the layout has one).
        voronoi::Timings t;
        mesh.rebuild(to_device(points), &t);
        mesh.update_cells(&t);
        std::cout << "final_rebuild: build_ms=" << t.build << " resort_ms=" << t.resort << " geometry_ms=" << t.geometry
                  << " cells_ms=" << t.cells << " total_ms=" << t.build + t.resort + t.geometry + t.cells
                  << " Mcells_s=" << n / (t.build + t.resort + t.geometry + t.cells) / 1e3 << "\n";
    }
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
                    has_flag(argc, argv, "--measure"), static_cast<int>(flag_value(argc, argv, "--order", 0)),
                    layout_flags(argc, argv), text_flag(argc, argv, "--dump-cells", "")) != 0;
    std::cerr << "usage: voronoi_bench rebuild <n> <distribution> [repeats] [--no-check] [--memory]\n"
                 "                          [--measure=every --dump=file.npy] (VORONOI_MEASURE builds)\n"
                 "       voronoi_bench step <n> <distribution> <displacement> [steps] [--verify] [--checksum]\n"
                 "                          [--memory] [--kernels] [--measure] [--order=every]\n"
                 "                          [--curve=morton|hilbert|moore|gray|row|rcm] [--every=k] [--far=share]\n"
                 "                          [--owner-faces] [--owner-bit] [--dump-cells=path]\n"
                 "       both: [--nvtx | --nvtx=phases] (CUDA: NVTX ranges per phase, and per kernel label)\n"
                 "       both: [--box=Lx,Ly,Lz]\n";
    return 2;
}
