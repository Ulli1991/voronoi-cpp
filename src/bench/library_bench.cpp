// voronoi-cpp in the library benchmark (docs/library_benchmark.md), and the writer of its shared inputs.
// Modes: points, trajectory, build, step, lloyd (blue noise); usage() lists their arguments.
// File formats and the record of each timed run: bench/libraries/libbench.hpp.

#include "../../bench/libraries/libbench.hpp"
#include "device_memory.hpp"
#include "support.hpp"

#include <numeric>

namespace {

using namespace support;
using Mesh = voronoi::Mesh<3>;

#ifdef KOKKOS_ENABLE_CUDA
using Peak = libbench::DevicePeak;
const char* const device_name = "A100-SXM4-40GB";
#else
using Peak = libbench::HostPeak;
const char* const device_name = "2x Xeon Platinum 8360Y";
#endif

Points to_points(const double* xyz, std::uint64_t n) {
    Points p(n);
    for (std::uint64_t i = 0; i < n; ++i) p[i] = {xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]};
    return p;
}

std::vector<double> flat(const Points& points) {
    std::vector<double> xyz;
    xyz.reserve(points.size() * 3);
    for (const auto& p : points) xyz.insert(xyz.end(), p.begin(), p.end());
    return xyz;
}

// The commit, from the environment of the run script (git rev-parse).
std::string version() { return libbench::env("LIBBENCH_COMMIT", "unknown"); }

libbench::Setup setup(const std::string& variant, const std::string& outputs, const std::string& timed) {
    libbench::Setup s;
    s.library = "voronoi-cpp";
    s.variant = variant;
    s.version = version();
    s.device = device_name;
    s.threads = Exec().concurrency();
#ifdef KOKKOS_ENABLE_CUDA
    s.threads = 1;
#endif
    s.outputs = outputs;
    s.periodic = "native";
    s.precision = "double, exact predicates on the 2^-30 grid";
    s.timed = timed;
    return s;
}

// Peak memory of one call of f: process-wide (sampled device use or host resident set) and, as an extra field, the
// Kokkos arrays alone (DeviceMemory, Kokkos Tools callbacks).
template <class F>
double peak_of(const F& f) {
    Peak peak;
    DeviceMemory<Memory>::current = DeviceMemory<Memory>::peak = 0;
    DeviceMemory<Memory>::live.clear();
    peak.start();
    f();
    Exec().fence();
    return peak.stop();
}

#ifdef KOKKOS_ENABLE_CUDA
using PinnedSpace = Kokkos::CudaHostPinnedSpace;
#else
using PinnedSpace = Kokkos::HostSpace;
#endif

// Host buffers of the host-to-host volumes paths: points (rows x, y, z), volumes, centroids (columns x, y, z).
struct HostBuffers {
    Kokkos::View<const double* [3], Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> points;
    Kokkos::View<double*, Kokkos::HostSpace> volume;
    Kokkos::View<double* [3], Kokkos::LayoutLeft, Kokkos::HostSpace> centroid;
    Kokkos::View<double* [3], Kokkos::LayoutRight, PinnedSpace> pinned_points;
    Kokkos::View<double*, PinnedSpace> pinned_volume;
    Kokkos::View<double* [3], Kokkos::LayoutLeft, PinnedSpace> pinned_centroid;
};

// The caller's buffers, made and touched once outside the clock (points stay where the file was read).
HostBuffers host_buffers(const double* xyz, std::uint64_t n) {
    HostBuffers b;
    b.points = decltype(b.points)(xyz, n);
    b.volume = decltype(b.volume)("host_v", n);
    b.centroid = decltype(b.centroid)("host_c", n);
    b.pinned_points = decltype(b.pinned_points)("pinned_points", n);
    b.pinned_volume = decltype(b.pinned_volume)("pinned_v", n);
    b.pinned_centroid = decltype(b.pinned_centroid)("pinned_c", n);
    Kokkos::deep_copy(b.pinned_points, b.points);
    return b;
}

// One host-to-host build of volumes and centroids from the points in pageable or pinned memory. With copy, the
// volumes go to `out`.
void host_volumes(const Exec& exec, const Box& box, const HostBuffers& b, std::uint64_t n, bool pinned,
                  std::vector<double>& out, bool copy) {
    Kokkos::View<double* [3], Kokkos::LayoutRight, Memory> xd(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("points")), n);
    Kokkos::View<double*, Memory> v(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("v")), n);
    Kokkos::View<double* [3], Memory> c(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("c")), n);
    if (pinned)
        Kokkos::deep_copy(exec, xd, b.pinned_points);
    else
        Kokkos::deep_copy(exec, xd, b.points);
    shell_voronoi::volumes_and_centroids(exec, box, xd, v, c);
    if (pinned) {
        Kokkos::deep_copy(exec, b.pinned_volume, v);
        Kokkos::deep_copy(exec, b.pinned_centroid, c);
    } else {
        Kokkos::deep_copy(exec, b.volume, v);
        Kokkos::deep_copy(exec, b.centroid, c);
    }
    exec.fence();
    if (!copy) return;
    for (std::uint64_t i = 0; i < n; ++i) out[i] = pinned ? b.pinned_volume(i) : b.volume(i);
}

int build(int argc, char** argv) {
    const std::string file = argv[2];
    const std::string path = libbench::option(argc, argv, "path", "warp");
    const int repeats = std::stoi(libbench::option(argc, argv, "repeats", "3"));
    const auto frames = libbench::read_frames(file);
    const std::uint64_t n = frames.n;
    const Positions x = to_device(to_points(frames.frame(0), n));
    const Box box{};
    const Exec exec;
    std::vector<double> volume(n);
    const Points host = to_points(frames.frame(0), n);  // the warp-host path's input
    // One build; with copy, its volumes to `volume` (outside the clock: timed runs do not copy).
    std::function<void(bool)> body;
    libbench::Setup s;
    HostBuffers buffers;
    if (path == "warp") {
        s = setup("warp clipping rebuild (cell list + construct_warp)", "volumes faces areas centroids",
                  "points on the device to cells on the device: cell list, grid, cell arrays, clipping");
        body = [&](bool copy) {
            neighbor_search::CellList<Exec> list;
            neighbor_search::configure(list, static_cast<int>(n), box, std::cbrt(1.0 / double(n)));
            neighbor_search::build(exec, list, x);
            shell_voronoi::Cells<Memory> cells(static_cast<int>(n), false);
            shell_voronoi::construct_warp(exec, shell_voronoi::make_grid(list), cells);
            exec.fence();
            if (!copy) return;
            const auto v = to_host(cells.volume);
            for (std::uint64_t i = 0; i < n; ++i) volume[i] = v(i);
        };
    } else if (path == "warp-host" || path == "warp-host-vc") {
        if (path == "warp-host")
            s = setup("warp clipping rebuild, host to host (copy in, cell list + construct_warp, copy out)",
                      "volumes faces areas centroids",
                      "points in host memory to cells in host memory: points copied to the device, cell list, grid, "
                      "cell arrays, clipping, then volumes, centroids, per-cell face starts and counts and the whole "
                      "face pool (neighbours, area vectors, face centroids) copied back");
        else
            s = setup("warp clipping rebuild, host to host, volumes and centroids copied out",
                      "volumes centroids (faces built on the device, not copied back)",
                      "points in host memory to volumes and centroids in host memory: points copied to the device, "
                      "cell list, grid, cell arrays, clipping (faces included), volumes and centroids copied back");
        body = [&](bool copy) {
            const Positions xd = to_device(host);
            neighbor_search::CellList<Exec> list;
            neighbor_search::configure(list, static_cast<int>(n), box, std::cbrt(1.0 / double(n)));
            neighbor_search::build(exec, list, xd);
            shell_voronoi::Cells<Memory> cells(static_cast<int>(n), false);
            shell_voronoi::construct_warp(exec, shell_voronoi::make_grid(list), cells);
            const auto v = to_host(cells.volume);
            const auto c = to_host(cells.centroid);
            if (path == "warp-host-vc") {  // the outputs of the volumes-only codes: volumes and centroids
                if (copy)
                    for (std::uint64_t i = 0; i < n; ++i) volume[i] = v(i);
                return;
            }
            const auto start = to_host(cells.face_start);
            const auto count = to_host(cells.owned_count);
            const auto nb = to_host(cells.neighbour);
            const auto area = to_host(cells.area);
            const auto fc = to_host(cells.face_centroid);
            if (!copy) return;
            for (std::uint64_t i = 0; i < n; ++i) volume[i] = v(i);
        };
    } else if (path == "warp-vo") {
        s = setup("volumes and centroids only (shell_voronoi::volumes_and_centroids)", "volumes centroids",
                  "points on the device to volumes and centroids on the device: snap, cell list, grid, clipping "
                  "without faces");
        body = [&](bool copy) {
            Kokkos::View<double*, Memory> v(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("v")), n);
            Kokkos::View<double* [3], Memory> c(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("c")), n);
            shell_voronoi::volumes_and_centroids(exec, box, x, v, c);
            exec.fence();
            if (!copy) return;
            const auto h = to_host(v);
            for (std::uint64_t i = 0; i < n; ++i) volume[i] = h(i);
        };
    } else if (path == "warp-host-vo" || path == "warp-pinned-vo") {
        const bool pinned = path == "warp-pinned-vo";
        if (pinned)
            s = setup("volumes and centroids only, host to host, pinned host buffers",
                      "volumes centroids",
                      "points in pinned host memory to volumes and centroids in pinned host memory: points copied "
                      "to the device, snap, cell list, grid, clipping without faces, the two arrays copied back");
        else
            s = setup("volumes and centroids only, host to host", "volumes centroids",
                      "points in host memory to volumes and centroids in host memory: points copied to the device, "
                      "snap, cell list, grid, clipping without faces, the two arrays copied back");
        buffers = host_buffers(frames.frame(0), n);
        body = [&, pinned](bool copy) { host_volumes(exec, box, buffers, n, pinned, volume, copy); };
    } else {
        s = setup("exact start (Mesh::update from scratch + update_cells)",
                  "Delaunay tetrahedra, volumes faces areas centroids, per-cell face lists",
                  "points on the device to mesh and cells on the device: Mesh construction, update, update_cells");
        body = [&](bool copy) {
            Mesh mesh(exec, box);
            const voronoi::Report r = mesh.update(x);
            mesh.update_cells();
            exec.fence();
            if (!copy) return;
            if (r.resorted) throw std::runtime_error("unexpected re-sort");
            const auto v = to_host(mesh.volume());
            for (std::uint64_t i = 0; i < n; ++i) volume[i] = v(i);
        };
    }
    const std::function<void()> timed_body = [&] { body(false); };
    body(true);  // warm-up, and the volumes
    libbench::write_volumes(libbench::option(argc, argv, "volumes"), volume);
    double kokkos_peak = 0;
    libbench::run_protocol(
        s, file, n, "build", repeats,
        [&] {
            const double p = peak_of(timed_body);
            kokkos_peak = double(DeviceMemory<Memory>::peak);
            return p;
        },
        [&] {
            const double t0 = libbench::now();
            timed_body();
            return libbench::now() - t0;
        });
    std::cout << "{\"library\":\"voronoi-cpp\",\"variant\":\"" << s.variant
              << "\",\"kokkos_arrays_peak_bytes\":" << kokkos_peak << ",\"n\":" << n << "}" << '\n' << std::flush;
    return 0;
}

// Moving mesh: the first frame builds (untimed), then each later frame is one update + update_cells, timed per step.
int step(int argc, char** argv) {
    const std::string file = argv[2];
    const auto frames = libbench::read_frames(file);
    const std::uint64_t n = frames.n;
    const Exec exec;
    auto s =
        setup("moving-mesh step (Mesh::update + update_cells)", "Delaunay tetrahedra, volumes faces areas centroids",
              "positions on the device to repaired mesh and cells on the device, per step");
    Mesh mesh(exec, Box{});
    std::vector<int> label(n);
    std::iota(label.begin(), label.end(), 0);
    auto follow = [&](const voronoi::Report& r) {
        if (!r.resorted) return;
        const auto p = to_host(mesh.permutation());
        std::vector<int> relabel(n);
        for (std::uint64_t i = 0; i < n; ++i) relabel[i] = label[p(i)];
        label.swap(relabel);
    };
    auto positions = [&](std::uint64_t f) {  // frame f in the mesh's current cell order
        Points p(n);
        const double* xyz = frames.frame(f);
        for (std::uint64_t i = 0; i < n; ++i) p[i] = {xyz[3 * label[i]], xyz[3 * label[i] + 1], xyz[3 * label[i] + 2]};
        return to_device(p);
    };
    // Peak memory from before the first build: the mesh's start and its steps, process-wide and Kokkos arrays.
    Peak peak;
    peak.start();
    follow(mesh.update(positions(0)));
    mesh.update_cells();
    const std::uint64_t built = DeviceMemory<Memory>::current;
    DeviceMemory<Memory>::peak = built;
    for (std::uint64_t f = 1; f < frames.frames; ++f) {
        const Positions x = positions(f);
        exec.fence();
        const double t0 = libbench::now();
        const voronoi::Report r = mesh.update(x);
        mesh.update_cells();
        exec.fence();
        const double seconds = libbench::now() - t0;
        follow(r);
        std::ostringstream extra;
        extra << ",\"flips\":" << r.flips << ",\"stuck\":" << r.stuck << ",\"rebuilt\":" << r.rebuilt;
        libbench::record(s, file, libbench::input_name(file), n, "step", static_cast<int>(f), seconds, -1, extra.str());
    }
    std::cout << "{\"library\":\"voronoi-cpp\",\"variant\":\"" << s.variant
              << "\",\"start_and_steps_peak_bytes\":" << peak.stop() << ",\"kokkos_arrays_after_start_bytes\":" << built
              << ",\"kokkos_arrays_step_peak_bytes\":" << DeviceMemory<Memory>::peak << ",\"n\":" << n << "}"
              << '\n' << std::flush;
    const auto v = to_host(mesh.volume());
    std::vector<double> volume(n);
    for (std::uint64_t i = 0; i < n; ++i) volume[label[i]] = v(i);
    libbench::write_volumes(libbench::option(argc, argv, "volumes"), volume);
    return 0;
}

// Blue noise as in Ray et al. 2018 and Basselin et al. 2021 (centroidal Voronoi tessellation): Lloyd iterations from
// the points of `in` in the periodic unit box, each moving every point to its cell's centroid (the warp build's centroid
// is relative to the generator), wrapped and snapped to the 2^-30 grid. Prints the rms spread of the cell volumes
// (relative to the mean) every 10 iterations.
int lloyd(const std::string& in, int iterations, const std::string& out) {
    const auto frames = libbench::read_frames(in);
    const std::uint64_t n = frames.n;
    Points p = to_points(frames.frame(0), n);
    const Exec exec;
    const Box box{};
    for (int it = 1; it <= iterations; ++it) {
        const Positions x = to_device(p);
        neighbor_search::CellList<Exec> list;
        neighbor_search::configure(list, static_cast<int>(n), box, std::cbrt(1.0 / double(n)));
        neighbor_search::build(exec, list, x);
        shell_voronoi::Cells<Memory> cells(static_cast<int>(n), false);
        shell_voronoi::construct_warp(exec, shell_voronoi::make_grid(list), cells);
        const auto c = to_host(cells.centroid);
        const auto v = to_host(cells.volume);
        double s1 = 0, s2 = 0;
        for (std::uint64_t i = 0; i < n; ++i) {
            for (int a = 0; a < 3; ++a) p[i][a] = wrap(p[i][a] + c(i, a));
            const double r = v(i) * double(n);
            s1 += r, s2 += r * r;
        }
        to_grid(p);
        if (it % 10 == 0 || it == iterations)
            std::cout << "lloyd: iteration " << it << " volume/mean: rms spread "
                      << std::sqrt(std::max(0.0, s2 / double(n) - (s1 / double(n)) * (s1 / double(n)))) << "\n";
    }
    libbench::write_frames(out, n, flat(p));
    std::cout << "wrote " << n << " points to " << out << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    DeviceMemory<Memory>::start();
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "points" && argc >= 5) {
        const Points p = make_points(argv[3], std::stoi(argv[2]));
        libbench::write_frames(argv[4], p.size(), flat(p));
        std::cout << "wrote " << p.size() << " points to " << argv[4] << "\n";
        return 0;
    }
    if (mode == "trajectory" && argc >= 7) {
        Points p = in_slot_order(make_points(argv[3], std::stoi(argv[2])));
        const double spacing = std::cbrt(1.0 / double(p.size()));
        const double distance = std::stod(argv[4]) * spacing;
        std::vector<double> xyz = flat(p);
        const Flow flow;
        for (int s = 0; s < std::stoi(argv[5]); ++s) {
            flow.advect(p, distance);
            const auto f = flat(p);
            xyz.insert(xyz.end(), f.begin(), f.end());
        }
        libbench::write_frames(argv[6], p.size(), xyz);
        std::cout << "wrote " << xyz.size() / (3 * p.size()) << " frames of " << p.size() << " points to " << argv[6]
                  << "\n";
        return 0;
    }
    if (mode == "lloyd" && argc >= 5) return lloyd(argv[2], std::stoi(argv[3]), argv[4]);
    if (mode == "build" && argc >= 3) return build(argc, argv);
    if (mode == "step" && argc >= 3) return step(argc, argv);
    std::cerr << "usage: library_bench points <n> <distribution> <file>\n"
                 "       library_bench trajectory <n> <distribution> <spacings> <steps> <file>\n"
                 "       library_bench build <points file> [--path=warp|warp-host|warp-host-vc|warp-vo|warp-host-vo|warp-pinned-vo|exact] [--repeats=3] "
                 "[--volumes=file]\n"
                 "       library_bench step <trajectory file> [--volumes=file]\n"
                 "       library_bench lloyd <points file> <iterations> <file>\n";
    return 2;
}
