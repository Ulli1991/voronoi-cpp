// The 1D mesh (voronoi::Mesh<1>, sorted_mesh.hpp) against a host oracle in integers (quarter grid steps): order, cells,
// faces, wall cases, and after every step (order kept or crossed, wrapping) equal to a fresh build bit for bit.

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
std::string oracle(const Mesh& mesh, const std::vector<double>& x, const sorted_mesh::Grid& grid, bool periodic) {
    const int n = static_cast<int>(x.size());
    const long long P = sorted_mesh::Grid::period;
    std::vector<long long> key(n);
    for (int i = 0; i < n; ++i) {
        long long k = static_cast<long long>(std::floor(x[i] / grid.step));
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
        if (m.volume[i] != static_cast<double>(l + r) * (0.5 * grid.step)) return "volume of " + std::to_string(i);
        if (m.centroid[i] != static_cast<double>(r - l) * (0.25 * grid.step)) return "centroid of " + std::to_string(i);
        if (m.position[i] != static_cast<double>(key[i]) * grid.step) return "position of " + std::to_string(i);
        const bool right_wall = !periodic && s == n - 1, left_wall = !periodic && s == 0;
        const int right = right_wall ? -2 : order[(s + 1) % n], left = left_wall ? -1 : order[(s + n - 1) % n];
        if (m.others[2 * i] != std::min(left, right) || m.others[2 * i + 1] != std::max(left, right))
            return "face list of " + std::to_string(i);
        const double half = static_cast<double>(r) * (0.5 * grid.step);
        if (right_wall)
            faces.emplace_back(i, -2, 1.0, half);
        else
            faces.emplace_back(std::min(i, right), std::max(i, right), i < right ? 1.0 : -1.0, i < right ? half : -half);
        if (left_wall) faces.emplace_back(i, -1, -1.0, -static_cast<double>(l) * (0.5 * grid.step));
    }
    std::sort(faces.begin(), faces.end());
    if (faces != m.faces) return "faces";
    if (total != 2 * P) return "volumes do not sum to the line";
    double sum = 0;
    for (double v : m.volume) sum += v;
    if (std::abs(sum - grid.side) > 1e-12 * grid.side) return "summed volumes differ from L";  // total is the exact check
    return "";
}

// The cells of the snapped points as intervals in half grid steps: a periodic
// cell may run past the end of the line. O(n^2) in overlaps, a test's oracle.
struct Interval {
    long long lo, hi;
};

std::vector<long long> keys_of(const std::vector<double>& x, const sorted_mesh::Grid& grid, bool periodic) {
    std::vector<long long> key;
    for (const double v : x) key.push_back(static_cast<long long>(sorted_mesh::snap(v, grid, periodic)));
    return key;
}

std::vector<Interval> cells_of(const std::vector<long long>& key, bool periodic) {
    const long long P = sorted_mesh::Grid::period;
    const int n = static_cast<int>(key.size());
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return key[a] != key[b] ? key[a] < key[b] : a < b; });
    std::vector<Interval> cell(n);
    for (int s = 0; s < n; ++s) {
        const long long k = key[order[s]];
        const long long left = s > 0 ? key[order[s - 1]] : periodic ? key[order[n - 1]] - P : -k;
        const long long right = s + 1 < n ? key[order[s + 1]] : periodic ? key[order[0]] + P : 2 * P - k;
        cell[order[s]] = {left + k, k + right};
    }
    return cell;
}

long long common(const Interval& a, const Interval& b, bool periodic) {
    long long sum = 0;
    const long long P2 = 2 * sorted_mesh::Grid::period;
    for (int shift = periodic ? -1 : 0; shift <= (periodic ? 1 : 0); ++shift)
        sum += std::max(0LL, std::min(a.hi, b.hi + shift * P2) - std::max(a.lo, b.lo + shift * P2));
    return sum;
}

// The report of one change against the cells before and after it. Insert:
// the new cells' overlaps; remove: what the neighbours gain from the gone
// cell, the same integers with the old and the new cells' roles swapped.
std::string check_report(const Mesh::Changes& c, const std::vector<Interval>& parents, const std::vector<Interval>& children,
                         const std::vector<int>& child_of_request, const std::vector<Interval>& before, double step, bool periodic) {
    const auto id = host(c.id);
    const auto neighbour = host(c.neighbour);
    const auto overlap = host(c.overlap);
    const auto volume = host(c.volume);
    for (int r = 0; r < c.requested; ++r) {
        if (id(r) < 0) continue;
        const Interval& cell = children[child_of_request[r]];
        for (int side = 0; side < 2; ++side) {
            const int other = neighbour(r, side);
            if (other < 0) {
                if (overlap(r, side) != 0.0) return "a wall gives nothing";
                continue;
            }
            const double expected = 0.5 * step * static_cast<double>(common(cell, parents[other], periodic));
            if (overlap(r, side) != expected) return "overlap of request " + std::to_string(r);
            if (volume(r, side) != 0.5 * step * static_cast<double>(before[other].hi - before[other].lo))
                return "volume before, request " + std::to_string(r);
        }
        if (overlap(r, 0) + overlap(r, 1) != 0.5 * step * static_cast<double>(cell.hi - cell.lo)) return "the overlaps miss part of the cell";
    }
    return "";
}

// Random batches of insertions and removals on a line.
void changes_on(Check& check, bool periodic, const Exec& exec, std::mt19937_64& rng) {
    const double L = 3.0;
    const sorted_mesh::Grid grid(L);
    const std::string ends = periodic ? "periodic" : "walls";
    const auto kind = periodic ? voronoi::Ends::periodic : voronoi::Ends::walls;
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::vector<double> x(200);
    for (double& v : x) v = L * unit(rng);
    Mesh mesh(exec, L, kind);
    mesh.update(device(x));
    bool order_good = true, fresh_good = true, report_good = true, same_gap_deferred = false, on_generator_deferred = false;
    std::string why;
    long long inserted = 0, removed = 0;
    int deferred = 0;
    const int capacity = mesh.capacity();
    for (int round = 0; round < 60; ++round) {
        // Rotate a periodic line: the order holds without a sort.
        if (periodic) {
            for (double& v : x) v = std::fmod(v + 0.07 * L, L);
            mesh.update(device(x));
        }
        const std::vector<long long> key_before = keys_of(x, grid, periodic);
        const std::vector<Interval> before = cells_of(key_before, periodic);
        if (round % 3 != 2) {
            std::vector<double> points;
            const int k = 1 + static_cast<int>(rng() % 12);
            for (int r = 0; r < k; ++r) points.push_back(L * unit(rng));
            points.push_back(x[rng() % x.size()]);
            points.push_back(std::nextafter(points[0], L) + grid.step);
            if (!periodic) points.push_back(0.1 * L / static_cast<double>(x.size()) * unit(rng));
            if (!periodic) points.push_back(L * (1 - 0.1 * unit(rng) / static_cast<double>(x.size())));
            const X requests = device(points);
            const std::size_t old_n = x.size();
            const Mesh::Changes& c = mesh.insert(requests);
            const auto id = host(c.id);
            std::vector<int> child_of_request(points.size(), -1);
            x.resize(old_n + static_cast<std::size_t>(c.done));
            for (int r = 0; r < c.requested; ++r) {
                if (id(r) < 0) {
                    ++deferred;
                    const long long key = static_cast<long long>(sorted_mesh::snap(points[r], grid, periodic));
                    if (std::find(key_before.begin(), key_before.end(), key) != key_before.end())
                        on_generator_deferred = true;
                    else
                        same_gap_deferred = true;
                    continue;
                }
                if (id(r) < c.first || id(r) >= c.first + c.done) report_good = false;
                x[id(r)] = points[r];
                child_of_request[r] = id(r);
                ++inserted;
            }
            const std::vector<Interval> after = cells_of(keys_of(x, grid, periodic), periodic);
            const std::string bad = check_report(c, before, after, child_of_request, before, grid.step, periodic);
            if (!bad.empty()) why = "insert: " + bad;
        } else {
            // Up to 8 ids, some of them neighbours on the line.
            std::vector<int> ids;
            const int k = 1 + static_cast<int>(rng() % 8);
            for (int r = 0; r < k; ++r) ids.push_back(static_cast<int>(rng() % x.size()));
            Kokkos::View<int*, Memory> requests("ids", ids.size());
            const auto requests_host = Kokkos::create_mirror_view(requests);
            for (std::size_t r = 0; r < ids.size(); ++r) requests_host(r) = ids[r];
            Kokkos::deep_copy(requests, requests_host);
            const Mesh::Changes& c = mesh.remove(requests);
            const auto id = host(c.id);
            const auto moved = host(c.moved);
            std::vector<int> cell_of_request(ids.size(), -1);
            for (int r = 0; r < c.requested; ++r) {
                if (id(r) >= 0)
                    cell_of_request[r] = id(r);
                else
                    ++deferred;
            }
            // The cells after, under the ids before: what the neighbours became.
            std::vector<char> gone(x.size(), 0);
            for (int r = 0; r < c.requested; ++r)
                if (id(r) >= 0) gone[id(r)] = 1;
            std::vector<long long> kept_keys;
            std::vector<int> kept_ids;
            for (std::size_t i = 0; i < x.size(); ++i)
                if (!gone[i]) {
                    kept_keys.push_back(key_before[i]);
                    kept_ids.push_back(static_cast<int>(i));
                }
            const std::vector<Interval> kept = cells_of(kept_keys, periodic);
            std::vector<Interval> after(x.size(), Interval{0, 0});
            for (std::size_t j = 0; j < kept.size(); ++j) after[kept_ids[j]] = kept[j];
            const std::string bad = check_report(c, after, before, cell_of_request, before, grid.step, periodic);
            if (!bad.empty()) why = "remove: " + bad;
            for (int m = 0; m < c.moves; ++m) x[moved(m, 1)] = x[moved(m, 0)];
            const std::size_t now = x.size() - static_cast<std::size_t>(c.done);
            x.resize(now);
            removed += c.done;
        }
        mesh.update_cells();
        if (mesh.particles() != static_cast<int>(x.size())) why = "the count";
        // Renaming can leave ties out of id order, and the next update sorts them.
        mesh.update(device(x));
        const std::string bad = oracle(mesh, x, grid, periodic);
        if (!bad.empty()) {
            order_good = false;
            why = bad;
        }
        Mesh fresh(exec, L, kind);
        fresh.update(device(x));
        fresh_good = fresh_good && same(read(mesh), read(fresh));
    }
    check.expect(order_good && fresh_good && report_good && why.empty(),
                 ends + ": 60 batches, " + std::to_string(inserted) + " inserted, " + std::to_string(removed) + " removed, " + std::to_string(deferred) +
                     " deferred: the oracle, a fresh build, and overlaps against the host's" + (why.empty() ? "" : " (" + why + ")"));
    check.expect(same_gap_deferred && on_generator_deferred, ends + ": a second request in a gap and one on a generator are deferred");
    check.expect(mesh.capacity() > capacity, ends + ": the arrays grew, from " + std::to_string(capacity) + " to " + std::to_string(mesh.capacity()));
}

// Insertion at the two ends of a line with walls: a transfer by the report
// conserves to rounding, and no sort runs.
void transfer_at_a_wall(Check& check, const Exec& exec) {
    std::vector<double> x;
    const double L = 2.0;
    for (int i = 0; i < 64; ++i) x.push_back(L * (i + 0.5) / 64);
    Mesh mesh(exec, L, voronoi::Ends::walls);
    mesh.update(device(x));
    std::vector<double> mass(64);
    double total = 0;
    for (int i = 0; i < 64; ++i) total += mass[i] = 1.0 + 0.3 * std::sin(i * 0.37);
    const Mesh::Changes& c = mesh.insert(device({0.01, 1.99}));
    const auto id = host(c.id);
    const auto neighbour = host(c.neighbour);
    const auto overlap = host(c.overlap);
    const auto volume = host(c.volume);
    check.expect(neighbour(0, 0) == -1 && neighbour(1, 1) == -2 && c.done == 2, "insertion next to the walls: the wall is reported");
    mass.resize(66);
    for (int r = 0; r < 2; ++r) {
        double taken = 0;
        for (int side = 0; side < 2; ++side) {
            const int other = neighbour(r, side);
            if (other < 0) continue;
            const double part = overlap(r, side) / volume(r, side) * mass[other];
            mass[other] -= part;
            taken += part;
        }
        mass[id(r)] = taken;
    }
    double after = 0;
    for (const double m : mass) after += m;
    check.expect(std::abs(after - total) < 1e-14 * total, "a transfer by the report conserves");
    check.expect(mesh.sorts() == 0, "insertion does not sort");
    x.push_back(0.01);
    x.push_back(1.99);
    mesh.update(device(x));
    check.expect(mesh.sorts() == 0, "the next update finds the order good");
    check.expect(host(mesh.line().order)(0) == id(0),
                 "the new point is the first of the line");
    check.expect(host(mesh.line().order)(65) == id(1),
                 "and the other the last");
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
            const sorted_mesh::Grid grid(L);
            const int n = 1000;
            // Inputs: uniform random; a lattice; clumps of snapped duplicates (ties by id).
            std::vector<std::pair<std::string, std::vector<double>>> inputs;
            std::vector<double> x(n);
            for (double& v : x) v = L * unit(rng);
            inputs.emplace_back("uniform", x);
            for (int i = 0; i < n; ++i) x[i] = L * ((n - 1 - i) + 0.5) / n;
            inputs.emplace_back("lattice", x);
            for (int i = 0; i < n; ++i) x[i] = L * std::floor(20 * unit(rng)) / 20 + 0.1 * grid.step * unit(rng);
            inputs.emplace_back("duplicates", x);
            for (const auto& [name, input] : inputs) {
                Mesh mesh(exec, box, periodic ? voronoi::Ends::periodic : voronoi::Ends::walls);
                mesh.update(device(input));
                const std::string bad = oracle(mesh, input, grid, periodic);
                check.expect(bad.empty(), "build: " + ends + " L=" + std::to_string(int(L)) + " " + name +
                                              ": the oracle's order, cells and faces" + (bad.empty() ? "" : " (" + bad + ")"));
            }
            // Moving: steps of 0.01 spacings (order kept), then 0.7 spacings (points cross), with a drift that wraps
            // (periodic) or is clamped at the walls.
            std::vector<double> y = inputs[0].second;
            Mesh mesh(exec, box, periodic ? voronoi::Ends::periodic : voronoi::Ends::walls);
            mesh.update(device(y));
            const double spacing = L / n;
            bool all_same = true, all_oracle = true;
            long long sorts_small = 0;
            std::uniform_real_distribution<double> jitter(-1.0, 1.0);
            for (int stepno = 0; stepno < 12; ++stepno) {
                const bool large = stepno >= 6;
                // Small steps: a monotone map (a smooth compression, plus a shift through L when periodic) that keeps
                // the order.
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
                all_oracle = all_oracle && oracle(mesh, y, grid, periodic).empty();
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
        check.expect(oracle(mesh, x, sorted_mesh::Grid(1.0), true).empty(), "order() then update_cells() equals the oracle");
        bool refused = false;
        try {
            mesh.update(device({0.1, 0.2}));
        } catch (const std::invalid_argument&) {
            refused = true;
        }
        check.expect(refused, "fewer than 3 particles are refused");
    }
    changes_on(check, true, exec, rng);
    changes_on(check, false, exec, rng);
    transfer_at_a_wall(check, exec);
    return check.failed() != 0;
}
