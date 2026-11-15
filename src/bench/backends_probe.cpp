// Probe, not a test (needs Kokkos with OpenMP and CUDA; docs/ideas.md, "CPU and GPU together"): Mesh<3, OpenMP>
// against Mesh<3, Cuda> on one input (bits, times), and the CPU's time on the clipping's tail cells.
//   backends_probe <n> <distribution> <displacement in spacings> <steps> <tail cells>

#include "support.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>

#if !defined(KOKKOS_ENABLE_OPENMP) || !defined(KOKKOS_ENABLE_CUDA)
#error "backends_probe needs a Kokkos with OpenMP and CUDA"
#endif

namespace {

using namespace support;
using Host = Kokkos::OpenMP;
using Device = Kokkos::Cuda;

template <class E>
using X = Kokkos::View<double* [3], typename E::memory_space>;

template <class E>
X<E> positions(const Points& p) {
    X<E> x("x", p.size());
    auto h = Kokkos::create_mirror_view(x);
    for (std::size_t i = 0; i < p.size(); ++i)
        for (int a = 0; a < 3; ++a) h(i, a) = p[i][a];
    Kokkos::deep_copy(x, h);
    return x;
}

std::uint64_t bits(double d) {
    std::uint64_t u;
    std::memcpy(&u, &d, sizeof u);
    return u;
}

// FNV-1a over values given in a fixed order. Used by: summary.
struct Hash {
    std::uint64_t h = 1469598103934665603ull;
    void mix(std::uint64_t u) { h = (h ^ u) * 1099511628211ull; }
};

// What a mesh holds, in an order independent of how it was numbered: per particle its volume and centroid bits and
// its sorted neighbour ids with the bits of each face's area vector (sign by the lower id). Hashes: topology, all.
template <class M>
std::pair<std::uint64_t, std::uint64_t> summary(const M& mesh, const std::vector<int>& label) {
    const int n = mesh.particles();
    const auto owner = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.owner());
    const auto neighbour = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.neighbour());
    const auto area = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.area());
    const auto volume = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.volume());
    const auto centroid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.centroid());
    std::vector<std::vector<std::array<std::uint64_t, 4>>> faces(n);
    for (int f = 0; f < mesh.faces(); ++f) {
        int a = label[owner(f)], b = label[neighbour(f)];
        const double s = a < b ? 1.0 : -1.0;
        if (a > b) std::swap(a, b);
        faces[a].push_back({std::uint64_t(b), bits(s * area(f, 0)), bits(s * area(f, 1)), bits(s * area(f, 2))});
    }
    std::vector<int> cell_of(n);
    for (int i = 0; i < n; ++i) cell_of[label[i]] = i;
    Hash topology, all;
    for (int p = 0; p < n; ++p) {
        std::sort(faces[p].begin(), faces[p].end());
        const int i = cell_of[p];
        all.mix(bits(volume(i)));
        for (int a = 0; a < 3; ++a) all.mix(bits(centroid(i, a)));
        for (const auto& f : faces[p]) {
            topology.mix(f[0]);
            for (auto u : f) all.mix(u);
        }
    }
    return {topology.h, all.h};
}

// A moving run on execution space E: build, then `steps` steps of the bench flow; hashes per step and times.
template <class E>
std::vector<std::pair<std::uint64_t, std::uint64_t>> run(Points points, double distance, int steps, const char* name) {
    E exec;
    voronoi::Mesh<3, E> mesh(exec, 1.0);
    std::vector<int> label(points.size());
    std::iota(label.begin(), label.end(), 0);
    std::vector<std::pair<std::uint64_t, std::uint64_t>> out;
    const Flow flow;
    for (int s = 0; s <= steps; ++s) {
        if (s > 0) flow.advect(points, distance);
        const auto x = positions<E>(points);
        Kokkos::Timer t;
        const voronoi::Report r = mesh.update(x);
        mesh.update_cells();
        exec.fence();
        const double ms = t.seconds() * 1e3;
        if (r.resorted) throw std::logic_error("no re-sort expected");
        out.push_back(summary(mesh, label));
        std::printf(
            "backend=%s step=%d ms=%.2f Mcells_s=%.2f rebuilt=%d stuck=%d host_flips=%lld topology=%016llx "
            "all=%016llx\n",
            name, s, ms, points.size() / ms / 1e3, int(r.rebuilt), r.stuck, r.host_flips,
            (unsigned long long)out.back().first, (unsigned long long)out.back().second);
    }
    return out;
}

// Warp clipping on the device, then the cells it left to the per-thread tail (status not ok after the warp kernel) on
// the host's per-thread kernel, timed; and the whole clipping on the host. Volumes compared bit by bit.
void tail(const Points& points, int tail_count) {
    const int n = static_cast<int>(points.size());
    const double width = std::cbrt(1.0 / n);
    const Box box{};
    // Device: construct_warp (warp kernel and its tail pass), with the work counters.
    neighbor_search::CellList<Device> dlist;
    neighbor_search::configure(dlist, n, box, width);
    const auto xd = positions<Device>(points);
    neighbor_search::build(Device{}, dlist, xd);
    const auto dgrid = shell_voronoi::make_grid(dlist);
    shell_voronoi::Cells<Device::memory_space> dcells(n, true);
    Kokkos::Timer t;
    shell_voronoi::construct_warp(Device{}, dgrid, dcells);
    Device{}.fence();
    const double device_ms = t.seconds() * 1e3;
    // Host: the same clipping with the per-thread kernel, all cells.
    neighbor_search::CellList<Host> hlist;
    neighbor_search::configure(hlist, n, box, width);
    const auto xh = positions<Host>(points);
    neighbor_search::build(Host{}, hlist, xh);
    const auto hgrid = shell_voronoi::make_grid(hlist);
    shell_voronoi::Cells<Host::memory_space> hcells(n, false);
    t.reset();
    shell_voronoi::construct(Host{}, hgrid, hcells);
    const double host_ms = t.seconds() * 1e3;
    const auto dv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dcells.volume);
    long long differ = 0;
    for (int i = 0; i < n; ++i) differ += bits(dv(i)) != bits(hcells.volume(i));
    // The tail: the `tail` cells with the most clips (the warp kernel's overflows are among them; their count is the
    // measurement build's warp_status), clipped alone on the host, timed.
    const auto work = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dcells.work);
    std::vector<int> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    const int count = std::min(n, tail_count);
    std::partial_sort(ids.begin(), ids.begin() + count, ids.end(),
                      [&](int a, int b) { return work(a, 1) > work(b, 1); });
    Kokkos::View<int*, Kokkos::HostSpace> slots("slots", count);
    Kokkos::View<char*, Kokkos::HostSpace> listed("listed", n);
    std::vector<int> slot_of(n);
    for (int q = 0; q < n; ++q) slot_of[hgrid.order(q)] = q;
    long long clips = 0;
    for (int i = 0; i < count; ++i) slots(i) = slot_of[ids[i]], listed(ids[i]) = 1, clips += work(ids[i], 1);
    shell_voronoi::Cells<Host::memory_space> tcells(n, false, 10, count);
    double tail_ms = 1e30;
    for (int r = 0; r < 3; ++r) {
        t.reset();
        shell_voronoi::construct(Host{}, hgrid, tcells, {slots, listed});
        tail_ms = std::min(tail_ms, t.seconds() * 1e3);
    }
    if (count > 0)
        std::printf("tail: cells=%d clips_per_cell=%.0f (all cells: see work) host_ms=%.3f us_per_cell_core=%.1f\n",
                    count, double(clips) / count, tail_ms, tail_ms * 1e3 * Host().concurrency() / count);
    std::printf(
        "clip: device_ms=%.2f host_ms=%.2f (%d threads) device_Mcells_s=%.2f host_Mcells_s=%.3f ratio=%.1f "
        "volume_bits_differ=%lld of %d\n",
        device_ms, host_ms, Host().concurrency(), n / device_ms / 1e3, n / host_ms / 1e3, host_ms / device_ms, differ,
        n);
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    if (argc < 6) {
        std::fprintf(stderr, "usage: backends_probe <n> <distribution> <displacement> <steps> <tail cells>\n");
        return 2;
    }
    const Points points = in_slot_order(make_points(argv[2], std::stoi(argv[1])));
    const double distance = std::stod(argv[3]) * std::cbrt(1.0 / points.size());
    const int steps = std::stoi(argv[4]);
    std::printf("# n=%zu distribution=%s host_threads=%d\n", points.size(), argv[2], Host().concurrency());
    tail(points, std::stoi(argv[5]));
    const auto h = run<Host>(points, distance, steps, "openmp");
    const auto d = run<Device>(points, distance, steps, "cuda");
    int topology = 0, all = 0;
    for (std::size_t s = 0; s < h.size(); ++s) topology += h[s].first == d[s].first, all += h[s] == d[s];
    std::printf("same: topology %d of %zu steps, all bits %d of %zu\n", topology, h.size(), all, h.size());
    return 0;
}
