// The modes of voronoi::Engine (cpu, gpu, hybrid; docs/ideas.md, "CPU and GPU together") give the same bits: a build
// and moving steps per input in every mode the build has, hashed by particle and compared with cpu. Without a device
// space only cpu runs, and gpu must be refused.

#include "check.hpp"
#include "support.hpp"
#include "voronoi_engine.hpp"

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

// A build and `steps` steps of the bench flow, scaled to `spacings` per step, in one mode: hashes per update.
template <int D>
std::vector<std::pair<std::uint64_t, std::uint64_t>> run(voronoi::Backend backend, Points points, double spacings,
                                                         int steps) {
    voronoi::Engine<D> engine(1.0, {backend});
    const double distance = spacings * std::pow(1.0 / double(points.size()), 1.0 / D);
    const Flow flow;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> out;
    for (int s = 0; s <= steps; ++s) {
        if (s > 0) flow.advect(points, distance);
        if constexpr (D < 3)
            for (auto& p : points)
                for (int a = D; a < 3; ++a) p[a] = 0;
        engine.update(host_rows<D>(points));
        out.push_back(hashes(engine));
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

std::string hex(std::uint64_t h) {
    char text[17];
    std::snprintf(text, sizeof text, "%016llx", static_cast<unsigned long long>(h));
    return text;
}

// Every mode this build has against the first; `what` names the case.
template <int D>
void compare(Check& check, const std::string& what, const Points& points, double spacings, int steps) {
    std::vector<voronoi::Backend> modes{voronoi::Backend::cpu};
    if (voronoi::has_device()) modes = {voronoi::Backend::cpu, voronoi::Backend::gpu, voronoi::Backend::hybrid};
    const auto first = run<D>(modes[0], points, spacings, steps);
    for (std::size_t m = 1; m < modes.size(); ++m) {
        const auto other = run<D>(modes[m], points, spacings, steps);
        int topology = 0, all = 0;
        for (std::size_t s = 0; s < first.size(); ++s)
            topology += first[s].first == other[s].first, all += first[s] == other[s];
        check.expect(all == int(first.size()), what + ": " + voronoi::name(modes[m]) + " equals cpu in every bit, " +
                                                   std::to_string(all) + " of " + std::to_string(first.size()) +
                                                   " updates (topology " + std::to_string(topology) + "; last hash " +
                                                   hex(other.back().second) + ")");
    }
    if (modes.size() == 1)
        check.expect(true, what + ": cpu only (no device space), last hash " + hex(first.back().second));
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
