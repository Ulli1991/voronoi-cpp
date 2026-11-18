// The modes of voronoi::Engine (cpu, gpu, hybrid; docs/ideas.md, "CPU and GPU together") against cpu: a build and moving
// steps per input, hashed by particle. hybrid gives cpu's bits (Layout::same_bits_as_host); gpu, fused, its faces and
// volumes to 1e-12. Without a device space only cpu runs, and gpu must be refused.

#include "check.hpp"
#include "support.hpp"
#include "voronoi_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

namespace {

using namespace support;

std::uint64_t bits(double d) {
    std::uint64_t u;
    std::memcpy(&u, &d, sizeof u);
    return u;
}

// FNV-1a. Used by: hashes.
struct Hash {
    std::uint64_t h = 1469598103934665603ull;
    void mix(std::uint64_t u) { h = (h ^ u) * 1099511628211ull; }
};

// Hashes of the engine's last update, independent of face numbering: per particle its volume and centroid bits, then
// its faces to higher ids sorted by that id, with the area vector's bits. first: topology only; second: everything.
template <int D>
std::pair<std::uint64_t, std::uint64_t> hashes(voronoi::Engine<D>& engine) {
    engine.update_cells();
    const int n = engine.particles();
    const auto owner = engine.owner();
    const auto neighbour = engine.neighbour();
    const auto area = engine.area();
    const auto volume = engine.volume();
    const auto centroid = engine.centroid();
    std::vector<std::vector<std::vector<std::uint64_t>>> faces(n);
    for (int f = 0; f < engine.faces(); ++f) {
        int a = owner(f), b = neighbour(f);
        const double s = a < b ? 1.0 : -1.0;
        if (a > b) std::swap(a, b);
        std::vector<std::uint64_t> row{std::uint64_t(b)};
        for (int k = 0; k < D; ++k) row.push_back(bits(s * area(f, k)));
        faces[a].push_back(row);
    }
    Hash topology, all;
    for (int i = 0; i < n; ++i) {
        std::sort(faces[i].begin(), faces[i].end());
        all.mix(bits(volume(i)));
        for (int k = 0; k < D; ++k) all.mix(bits(centroid(i, k)));
        for (const auto& row : faces[i]) {
            topology.mix(row[0]);
            for (auto u : row) all.mix(u);
        }
    }
    return {topology.h, all.h};
}

// Host positions of `points` (D columns of each). Used by: run.
template <int D>
Kokkos::View<double* [D], Kokkos::HostSpace> host_rows(const std::vector<std::array<double, 3>>& points) {
    Kokkos::View<double* [D], Kokkos::HostSpace> x("x", points.size());
    for (std::size_t i = 0; i < points.size(); ++i)
        for (int a = 0; a < D; ++a) x(i, a) = points[i][a];
    return x;
}

// One update in one mode: the hashes, and the volumes by particle.
struct Update {
    std::pair<std::uint64_t, std::uint64_t> hash;
    std::vector<double> volume;
    bool local = false;  // the update finished by a local repair (tet_local)
};

// A build and `steps` steps of the bench flow, scaled to `spacings` per step, in one mode; `jumpers` particles (the
// same in every mode) jump 4 spacings each step, which leaves stuck entries for the local repair.
template <int D>
std::vector<Update> run(voronoi::Backend backend, Points points, double spacings, int steps, int jumpers = 0) {
    voronoi::Engine<D> engine(1.0, {backend});
    const double spacing = std::pow(1.0 / double(points.size()), 1.0 / D), distance = spacings * spacing;
    const Flow flow;
    std::mt19937_64 rng(17);
    std::normal_distribution<double> g(0, 1);
    std::uniform_int_distribution<std::size_t> any(0, points.size() - 1);
    std::vector<Update> out;
    for (int s = 0; s <= steps; ++s) {
        if (s > 0) flow.advect(points, distance);
        for (int k = 0; s > 0 && k < jumpers; ++k) {
            auto& p = points[any(rng)];
            const double u[3] = {g(rng), g(rng), g(rng)};
            const double scale = 4 * spacing / std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
            for (int a = 0; a < 3; ++a) p[a] = wrap(p[a] + scale * u[a]);
        }
        if (jumpers > 0) to_grid(points);
        if constexpr (D < 3)
            for (auto& p : points)
                for (int a = D; a < 3; ++a) p[a] = 0;
        const voronoi::Report r = engine.update(host_rows<D>(points));
        Update u{hashes(engine), {}, r.local_done && !r.rebuilt};
        const auto volume = engine.volume();
        u.volume.assign(volume.data(), volume.data() + engine.particles());
        out.push_back(std::move(u));
    }
    return out;
}

// Random points in the unit square or segment, n of them. Used by: main (2D, 1D).
Points flat_points(int n, int d, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0, 1);
    Points p(n);
    for (auto& q : p) q = {u(rng), d > 1 ? u(rng) : 0.0, 0.0};
    return p;
}

std::string sci(double x) {
    char text[32];
    std::snprintf(text, sizeof text, "%.3g", x);
    return text;
}

std::string hex(std::uint64_t h) {
    char text[17];
    std::snprintf(text, sizeof text, "%016llx", static_cast<unsigned long long>(h));
    return text;
}

// Every mode this build has against the first; `what` names the case.
template <int D>
void compare(Check& check, const std::string& what, const Points& points, double spacings, int steps,
             int jumpers = 0) {
    std::vector<voronoi::Backend> modes{voronoi::Backend::cpu};
    if (voronoi::has_device()) modes = {voronoi::Backend::cpu, voronoi::Backend::gpu, voronoi::Backend::hybrid};
    std::vector<std::vector<Update>> runs;
    runs.reserve(modes.size());  // `first` and `other` stay valid
    runs.push_back(run<D>(modes[0], points, spacings, steps, jumpers));
    const auto& first = runs[0];
    const int updates = int(first.size());
    for (std::size_t m = 1; m < modes.size(); ++m) {
        runs.push_back(run<D>(modes[m], points, spacings, steps, jumpers));
        if (jumpers > 0) {
            const auto local = std::count_if(runs.back().begin(), runs.back().end(), [](const Update& u) { return u.local; });
            check.expect(local > 0, what + ": " + voronoi::name(modes[m]) + " repaired " + std::to_string(local) + " of " +
                                        std::to_string(updates) + " updates locally");
        }
        const auto& other = runs.back();
        int topology = 0, all = 0;
        long long cells = 0, cells_same = 0;
        double worst = 0;
        for (int s = 0; s < updates; ++s) {
            topology += first[s].hash.first == other[s].hash.first, all += first[s].hash == other[s].hash;
            for (std::size_t i = 0; i < first[s].volume.size(); ++i) {
                const double a = first[s].volume[i], b = other[s].volume[i];
                ++cells, cells_same += bits(a) == bits(b);
                worst = std::max(worst, std::abs(b - a) / a);
            }
        }
        const std::string counts = std::to_string(all) + " of " + std::to_string(updates) + " updates in every bit, " +
                                   std::to_string(cells_same) + " of " + std::to_string(cells) +
                                   " volumes; largest relative volume difference " + sci(worst) + "; last hash " +
                                   hex(other.back().hash.second);
        const std::string mode = voronoi::name(modes[m]);
        check.expect(topology == updates, what + ": " + mode + " has cpu's faces in " + std::to_string(topology) +
                                              " of " + std::to_string(updates) + " updates");
        if (modes[m] == voronoi::Backend::hybrid)
            check.expect(all == updates, what + ": hybrid equals cpu in every bit (" + counts + ")");
        else
            check.expect(worst < 1e-12, what + ": " + mode + " equals cpu to 1e-12 in volume (" + counts + ")");
    }
    if (modes.size() == 1)
        check.expect(true, what + ": cpu only (no device space), last hash " + hex(first.back().hash.second));
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    Check check;
    std::cout << "host space " << Kokkos::DefaultHostExecutionSpace::name() << ", device space "
              << Kokkos::DefaultExecutionSpace::name() << "\n";

    for (const std::string kind : {"jittered", "uniform", "clustered", "halo", "cosmicweb"})
        compare<3>(check, "3D " + kind + " 32768, 0.3 spacings, 4 steps", in_slot_order(make_points(kind, 32768)), 0.3,
                   4);
    compare<3>(check, "3D uniform 32768, 2 spacings (rebuilds), 2 steps", in_slot_order(make_points("uniform", 32768)),
               2.0, 2);
    compare<3>(check, "3D jittered 32768, 0.1 spacings and 4 jumpers of 4 spacings (local repairs), 4 steps",
               in_slot_order(make_points("jittered", 32768)), 0.1, 4, 4);
    compare<2>(check, "2D uniform 16384, 0.3 spacings, 4 steps", flat_points(16384, 2, 7), 0.3, 4);
    compare<1>(check, "1D uniform 4096, 0.3 spacings, 4 steps", flat_points(4096, 1, 11), 0.3, 4);

    {  // the options: names, refusals
        bool refused = false;
        try {
            voronoi::parse_backend("tpu");
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        check.expect(refused && voronoi::parse_backend("hybrid") == voronoi::Backend::hybrid,
                     "parse_backend knows cpu, gpu, hybrid and refuses anything else");
        bool no_device = false;
        try {
            voronoi::Engine<3> engine(1.0, {voronoi::Backend::gpu});
        } catch (const std::invalid_argument&) {
            no_device = true;
        }
        check.expect(no_device != voronoi::has_device(), "gpu is refused exactly when the build has no device space");
        bool negative = false;
        try {
            voronoi::Engine<3> engine(1.0, {voronoi::Backend::cpu, voronoi::Tail::off, -1});
        } catch (const std::invalid_argument&) {
            negative = true;
        }
        check.expect(negative, "negative host_threads is refused");
    }
    {  // a cpu engine on fewer host threads gives the same bits
        const Points points = in_slot_order(make_points("uniform", 32768));
        voronoi::Engine<3> all(1.0, {voronoi::Backend::cpu}), two(1.0, {voronoi::Backend::cpu, voronoi::Tail::off, 2});
        all.update(host_rows<3>(points));
        two.update(host_rows<3>(points));
        check.expect(hashes(all) == hashes(two), "cpu on 2 host threads equals cpu on all");
    }
    return check.failed() != 0;
}
