// Probe, not a test (needs Kokkos with OpenMP and CUDA; docs/ideas.md, "CPU and GPU together"): Mesh<3, OpenMP>
// against Mesh<3, Cuda> on one input (bits, times), and the CPU's time on the clipping's tail cells.
//   backends_probe <n> <distribution> <displacement in spacings> <steps> <tail cells>

#include "support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
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

// Volume of the cell of particle i in long double: a cube of half side 1/2 around it clipped by the bisector planes
// of its mesh neighbours (nearest images), each face a square on its plane clipped by every other plane, the volume
// the sum of face area times distance over 3. The reference for the volume errors of both spaces.
using Real = long double;
using R3 = std::array<Real, 3>;
Real dot(const R3& a, const R3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
R3 cross(const R3& a, const R3& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Real reference_volume(const std::vector<R3>& d) {  // d: neighbour offsets; plane: y.d <= d.d / 2
    std::vector<R3> planes = d;
    for (int a = 0; a < 3; ++a)
        for (Real s : {Real(-1), Real(1)}) {
            R3 c{0, 0, 0};
            c[a] = s;  // y.c <= 1/2: the cube
            planes.push_back(c);
        }
    Real volume = 0;
    for (std::size_t k = 0; k < planes.size(); ++k) {
        const R3& m = planes[k];
        const Real mm = dot(m, m), off = mm / 2;  // plane y.m = off
        const R3 n{m[0] / std::sqrt(mm), m[1] / std::sqrt(mm), m[2] / std::sqrt(mm)};
        R3 u = std::abs(n[0]) < 0.6L ? R3{1, 0, 0} : R3{0, 1, 0};
        u = cross(n, u);
        const Real lu = std::sqrt(dot(u, u));
        u = {u[0] / lu, u[1] / lu, u[2] / lu};
        const R3 v = cross(n, u);
        const Real h = off / std::sqrt(mm);  // distance of the plane from the particle
        std::vector<R3> poly;
        for (auto [a, b] : {std::pair<Real, Real>{-4, -4}, {4, -4}, {4, 4}, {-4, 4}})
            poly.push_back({h * n[0] + a * u[0] + b * v[0], h * n[1] + a * u[1] + b * v[1],
                            h * n[2] + a * u[2] + b * v[2]});
        for (std::size_t j = 0; j < planes.size() && !poly.empty(); ++j) {
            if (j == k) continue;
            const R3& q = planes[j];
            const Real qo = dot(q, q) / 2;
            std::vector<R3> next;
            for (std::size_t p = 0; p < poly.size(); ++p) {
                const R3& A = poly[p];
                const R3& B = poly[(p + 1) % poly.size()];
                const Real fa = dot(A, q) - qo, fb = dot(B, q) - qo;
                if (fa <= 0) next.push_back(A);
                if ((fa < 0) != (fb < 0) && fa != fb) {
                    const Real t = fa / (fa - fb);
                    next.push_back({A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1]), A[2] + t * (B[2] - A[2])});
                }
            }
            poly.swap(next);
        }
        R3 s{0, 0, 0};
        for (std::size_t p = 0; p < poly.size(); ++p) {
            const R3 c = cross(poly[p], poly[(p + 1) % poly.size()]);
            for (int a = 0; a < 3; ++a) s[a] += c[a];
        }
        volume += h * std::abs(dot(s, n)) / 2 / 3;
    }
    return volume;
}

// Relative volume errors of a mesh against reference_volume on every 16th particle: "median max" and the values.
template <class M>
std::vector<double> volume_errors(const M& mesh, const Points& points) {
    const int n = mesh.particles();
    const auto owner = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.owner());
    const auto neighbour = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.neighbour());
    const auto volume = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.volume());
    std::vector<std::vector<int>> adjacent(n);
    for (int f = 0; f < mesh.faces(); ++f)
        if (owner(f) % 16 == 0 || neighbour(f) % 16 == 0)
            adjacent[owner(f)].push_back(neighbour(f)), adjacent[neighbour(f)].push_back(owner(f));
    std::vector<double> error((n + 15) / 16);
    Kokkos::parallel_for(
        "reference_volumes", Kokkos::RangePolicy<Host>(0, int(error.size())), [&](int s) {
            const int i = 16 * s;
            std::vector<R3> d;
            for (int j : adjacent[i]) {
                R3 r;
                for (int a = 0; a < 3; ++a) r[a] = Real(nearest_image(points[j][a] - points[i][a]));
                d.push_back(r);
            }
            const Real ref = reference_volume(d);
            error[s] = double((Real(volume(i)) - ref) / ref);
        });
    Host{}.fence();
    return error;
}

// A moving run on execution space E: build, then `steps` steps of the bench flow; hashes per step and times.
template <class E>
std::vector<std::pair<std::uint64_t, std::uint64_t>> run(Points points, double distance, int steps, const char* name,
                                                         std::vector<std::vector<double>>* errors,
                                                         std::vector<std::vector<double>>* volumes) {
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
        errors->push_back(volume_errors(mesh, points));
        const auto v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.volume());
        volumes->emplace_back(v.data(), v.data() + mesh.particles());
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
    // The kernels apart from the spaces: the per-thread kernel on the device against it on the host (same code, other
    // compiler), and against the warp kernel on the device (same compiler, other code).
    {
        shell_voronoi::Cells<Device::memory_space> tdev(n, false);
        shell_voronoi::construct(Device{}, dgrid, tdev);
        const auto tv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tdev.volume);
        long long thread_spaces = 0, kernels = 0;
        double worst_spaces = 0, worst_kernels = 0;
        for (int i = 0; i < n; ++i) {
            thread_spaces += bits(tv(i)) != bits(hcells.volume(i));
            kernels += bits(tv(i)) != bits(dv(i));
            worst_spaces = std::max(worst_spaces, std::abs(tv(i) - hcells.volume(i)) / hcells.volume(i));
            worst_kernels = std::max(worst_kernels, std::abs(tv(i) - dv(i)) / dv(i));
        }
        // Cells where the kernels disagree beyond rounding: statuses, volumes, the exact kernel's volume on the host.
        const auto ts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tdev.status);
        const auto ds = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dcells.status);
        const auto tf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tdev.face_count);
        std::vector<int> odd;
        for (int i = 0; i < n; ++i) {
            const double h = hcells.volume(i);
            if (std::abs(tv(i) - h) > 1e-9 * h || std::abs(dv(i) - h) > 1e-9 * h || ts(i) != 0 ||
                hcells.status(i) != 0 || ds(i) != 0)
                odd.push_back(i);
        }
        if (!odd.empty()) {
            shell_voronoi::Cells<Host::memory_space> ecells(n, false);
            shell_voronoi::construct(Host{}, hgrid, ecells, {}, true);
            for (std::size_t k = 0; k < odd.size() && k < 20; ++k) {
                const int i = odd[k];
                std::printf(
                    "clip_odd: cell=%d x=(%.17g, %.17g, %.17g) status thread_device=%d thread_host=%d warp_device=%d "
                    "exact_host=%d volume thread_device=%.17g thread_host=%.17g warp_device=%.17g exact_host=%.17g "
                    "faces thread_device=%d thread_host=%d exact_host=%d\n",
                    i, points[i][0], points[i][1], points[i][2], ts(i), hcells.status(i), ds(i), ecells.status(i),
                    tv(i), hcells.volume(i), dv(i), ecells.volume(i), tf(i), hcells.face_count(i),
                    ecells.face_count(i));
            }
            std::printf("clip_odd: %zu cells\n", odd.size());
        }
        std::printf(
            "clip_kernels: thread_device_vs_thread_host_differ=%lld (max rel %.3g) thread_device_vs_warp_device_differ="
            "%lld (max rel %.3g) of %d\n",
            thread_spaces, worst_spaces, kernels, worst_kernels, n);
    }
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
    std::vector<std::vector<double>> he, de, hv, dv;
    const auto h = run<Host>(points, distance, steps, "openmp", &he, &hv);
    const auto d = run<Device>(points, distance, steps, "cuda", &de, &dv);
    int topology = 0, all = 0;
    for (std::size_t s = 0; s < h.size(); ++s) topology += h[s].first == d[s].first, all += h[s] == d[s];
    // Volume errors against the long-double reference (every 16th particle), each space, and the spaces' difference.
    auto stats = [](std::vector<double> v) {
        for (auto& x : v) x = std::abs(x);
        std::sort(v.begin(), v.end());
        double rms = 0;
        for (double x : v) rms += x * x;
        char text[160];
        std::snprintf(text, sizeof text, "median=%.3g p99=%.3g max=%.3g rms=%.3g", v[v.size() / 2],
                      v[std::size_t(0.99 * (v.size() - 1))], v.back(), std::sqrt(rms / v.size()));
        return std::string(text);
    };
    for (std::size_t s = 0; s < h.size(); ++s) {
        std::vector<double> gap(he[s].size());
        long long same = 0;
        for (std::size_t k = 0; k < gap.size(); ++k) gap[k] = de[s][k] - he[s][k], same += he[s][k] == de[s][k];
        std::vector<double> apart(hv[s].size());
        long long equal = 0;
        for (std::size_t i = 0; i < apart.size(); ++i)
            apart[i] = (dv[s][i] - hv[s][i]) / hv[s][i], equal += bits(dv[s][i]) == bits(hv[s][i]);
        std::printf("volume_error step=%zu openmp: %s | cuda: %s | cuda-openmp: %s | same=%lld of %zu\n", s,
                    stats(he[s]).c_str(), stats(de[s]).c_str(), stats(gap).c_str(), same, gap.size());
        std::printf("volume_apart step=%zu (cuda-openmp)/openmp, all cells: %s | bit-equal=%lld of %zu\n", s,
                    stats(apart).c_str(), equal, apart.size());
    }
    std::printf("same: topology %d of %zu steps, all bits %d of %zu\n", topology, h.size(), all, h.size());
    return 0;
}
