// The 1D mesh (voronoi::Mesh<1>, sorted_mesh.hpp) against a host oracle:
// the order is the sort of the snapped positions with ties by id, volumes
// and centroids are the exact midpoint cells (in quarter grid steps), the
// volumes sum to the line exactly, faces are stored once with owner < neighbour,
// area +-1 from owner to neighbour and centroid relative to the owner, the
// per-cell face lists name a cell's two faces in increasing id of the other
// particle; on a line with walls the end cells reach the walls and the two
// wall faces point out. Moving: after every step (small steps that keep the
// order, large ones that cross, points wrapping through L) the mesh equals
// a fresh build bit for bit, and the order is sorted again only when points
// crossed.

#include "check.hpp"
#include "voronoi_mesh.hpp"

#include <algorithm>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Memory = Exec::memory_space;
using Mesh = voronoi::Mesh<1, Exec>;
using X = Kokkos::View<double* [1], Memory>;
namespace ep = exact_predicates;

template <class V>
auto host(const V& v) {
    auto h = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h, v);
    return h;
}

X device(const std::vector<double>& x) {
    X v("x", x.size());
    auto h = Kokkos::create_mirror_view(v);
    for (std::size_t i = 0; i < x.size(); ++i) h(i, 0) = x[i];
    Kokkos::deep_copy(v, h);
    return v;
}

// A face as a comparable tuple: owner, neighbour, area, centroid.
using Face = std::tuple<int, int, double, double>;

struct Snapshot {
    std::vector<double> volume, centroid, position;
    std::vector<int> others;  // the other particle of each list entry, 2 per cell
    std::vector<Face> faces;  // sorted
};

Snapshot read(const Mesh& mesh) {
    const int n = mesh.particles(), faces = mesh.faces();
    const auto owner = host(mesh.owner()), neighbour = host(mesh.neighbour()), list = host(mesh.cell_face());
    const auto area = host(mesh.area()), fc = host(mesh.face_centroid());
    const auto volume = host(mesh.volume());
    const auto centroid = host(mesh.centroid()), positions = host(mesh.positions());
    Snapshot s;
    for (int i = 0; i < n; ++i) {
        s.volume.push_back(volume(i));
        s.centroid.push_back(centroid(i, 0));
        s.position.push_back(positions(i, 0));
        for (int e = 2 * i; e < 2 * i + 2; ++e) {
            const int f = list(e);
            s.others.push_back(owner(f) == i ? neighbour(f) : owner(f));
        }
    }
    for (int f = 0; f < faces; ++f) s.faces.emplace_back(owner(f), neighbour(f), area(f, 0), fc(f, 0));
    std::sort(s.faces.begin(), s.faces.end());
    return s;
}

bool same(const Snapshot& a, const Snapshot& b) {
    return a.volume == b.volume && a.centroid == b.centroid && a.position == b.position && a.others == b.others &&
           a.faces == b.faces;
}

// The oracle: sort (key, id) on the host, and every cell and face from it in
// integers (quarter grid steps).
std::string oracle(const Mesh& mesh, const std::vector<double>& x, const ep::BoxN<1>& box, bool periodic) {
    const int n = static_cast<int>(x.size());
    const long long P = box.period[0];
    std::vector<long long> key(n);
    for (int i = 0; i < n; ++i) {
        long long k = static_cast<long long>(std::floor(x[i] / box.step));
        key[i] = periodic ? ((k % P) + P) % P : std::clamp(k, 0LL, P - 1);
    }
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return key[a] != key[b] ? key[a] < key[b] : a < b; });
    auto gap = [&](int s) -> long long {  // slot s to s + 1, s in [-1, n)
        if (!periodic) {
            if (s < 0) return 2 * key[order[0]];
            if (s >= n - 1) return 2 * (P - key[order[n - 1]]);
            return key[order[s + 1]] - key[order[s]];
        }
        const int a = (s + n) % n, b = (a + 1) % n;
        const long long d = key[order[b]] - key[order[a]];
        return d < 0 ? d + P : d;
    };
    const Snapshot m = read(mesh);
    if (mesh.faces() != (periodic ? n : n + 1)) return "face count";
    long long total = 0;  // in half steps
    std::vector<Face> faces;
    for (int s = 0; s < n; ++s) {
        const int i = order[s];
        const long long l = gap(s - 1), r = gap(s);
        total += l + r;
        if (m.volume[i] != static_cast<double>(l + r) * (0.5 * box.step)) return "volume of " + std::to_string(i);
        if (m.centroid[i] != static_cast<double>(r - l) * (0.25 * box.step)) return "centroid of " + std::to_string(i);
        if (m.position[i] != static_cast<double>(key[i]) * box.step) return "position of " + std::to_string(i);
        const bool right_wall = !periodic && s == n - 1, left_wall = !periodic && s == 0;
        const int right = right_wall ? -2 : order[(s + 1) % n], left = left_wall ? -1 : order[(s + n - 1) % n];
        if (m.others[2 * i] != std::min(left, right) || m.others[2 * i + 1] != std::max(left, right))
            return "face list of " + std::to_string(i);
        const double half = static_cast<double>(r) * (0.5 * box.step);
        if (right_wall)
            faces.emplace_back(i, -2, 1.0, half);
        else
            faces.emplace_back(std::min(i, right), std::max(i, right), i < right ? 1.0 : -1.0, i < right ? half : -half);
        if (left_wall) faces.emplace_back(i, -1, -1.0, -static_cast<double>(l) * (0.5 * box.step));
    }
    std::sort(faces.begin(), faces.end());
    if (faces != m.faces) return "faces";
    if (total != 2 * P) return "volumes do not sum to the line";
    double sum = 0;
    for (double v : m.volume) sum += v;
    if (sum != box.side[0]) return "summed volumes differ from L";
    return "";
}

}  // namespace

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    Check check;
    const Exec exec;
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (const bool periodic : {true, false}) {
        const std::string ends = periodic ? "periodic" : "walls";
        for (const double L : {1.0, 3.0}) {
            const ep::BoxN<1> box(L);
            const int n = 1000;
            // Inputs: uniform random; a lattice; clumps of snapped duplicates (ties by id).
            std::vector<std::pair<std::string, std::vector<double>>> inputs;
            std::vector<double> x(n);
            for (double& v : x) v = L * unit(rng);
            inputs.emplace_back("uniform", x);
            for (int i = 0; i < n; ++i) x[i] = L * ((n - 1 - i) + 0.5) / n;  // reversed ids
            inputs.emplace_back("lattice", x);
            for (int i = 0; i < n; ++i) x[i] = L * std::floor(20 * unit(rng)) / 20 + 0.1 * box.step * unit(rng);
            inputs.emplace_back("duplicates", x);
            for (const auto& [name, input] : inputs) {
                Mesh mesh(exec, box, periodic ? voronoi::Ends::periodic : voronoi::Ends::walls);
                mesh.update(device(input));
                const std::string bad = oracle(mesh, input, box, periodic);
                check.expect(bad.empty(), "build: " + ends + " L=" + std::to_string(int(L)) + " " + name +
                                              ": the oracle's order, cells and faces" + (bad.empty() ? "" : " (" + bad + ")"));
            }
            // Moving: steps of 0.01 spacings (order kept), then 0.7 spacings
            // (points cross), with a drift that wraps (periodic) or is
            // clamped at the walls.
            std::vector<double> y = inputs[0].second;
            Mesh mesh(exec, box, periodic ? voronoi::Ends::periodic : voronoi::Ends::walls);
            mesh.update(device(y));
            const double spacing = L / n;
            bool all_same = true, all_oracle = true;
            long long sorts_small = 0;
            std::uniform_real_distribution<double> jitter(-1.0, 1.0);
            for (int stepno = 0; stepno < 12; ++stepno) {
                const bool large = stepno >= 6;
                // Small steps: a monotone map (a smooth compression, plus a
                // shift through L when periodic) that keeps the order.
                for (double& v : y) {
                    v += large ? spacing * 0.7 * jitter(rng)
                               : 0.5 * spacing * std::sin(2 * M_PI * v / L) + (periodic ? 0.03 * L : 0.0);
                    v = periodic ? v - L * std::floor(v / L) : std::clamp(v, 0.0, L);
                }
                mesh.update(device(y));
                if (!large) sorts_small = mesh.sorts();
                Mesh fresh(exec, box, periodic ? voronoi::Ends::periodic : voronoi::Ends::walls);
                fresh.update(device(y));
                all_same = all_same && same(read(mesh), read(fresh));
                all_oracle = all_oracle && oracle(mesh, y, box, periodic).empty();
            }
            check.expect(all_same && all_oracle, "moving " + ends + " L=" + std::to_string(int(L)) +
                                                     ": 12 updates equal a fresh build bit for bit and the oracle");
            check.expect(sorts_small == 0 && mesh.sorts() > 0,
                         "moving " + ends + ": no sort while the order holds" + (periodic ? " (rotating as points wrap)" : "") +
                             ", sorted again when points cross (" + std::to_string(mesh.sorts()) + " of 6)");
        }
    }
    {
        // order() leaves the face arrays stale, update_cells() brings them back.
        const ep::BoxN<1> box(1.0);
        std::vector<double> x(64);
        for (double& v : x) v = unit(rng);
        Mesh mesh(exec, box);
        mesh.update(device(x));
        for (double& v : x) v = std::fmod(v + 0.37, 1.0);
        mesh.order(device(x));
        mesh.update_cells();
        check.expect(oracle(mesh, x, box, true).empty(), "order() then update_cells() equals the oracle");
        bool refused = false;
        try {
            mesh.update(device({0.1, 0.2}));
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        check.expect(refused, "fewer than 3 particles are refused");
    }
    return check.failed() != 0;
}
