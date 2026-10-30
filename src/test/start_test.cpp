// The starting mesh is the Delaunay tetrahedralization of the snapped
// positions with ties broken by symbolic perturbation, for random,
// clustered, lattice and degenerate inputs. Independent host oracle, brute
// force (test only, O(tetrahedra x particles)):
//   1. a triangulation of the box: no flat tetrahedron, every triangle in
//      exactly two tetrahedra whose apexes lie on opposite sides, adjacency
//      equal to that pairing, every particle used, and the exact volumes
//      summing to the box;
//   2. every circumsphere empty of every other particle (exact, perturbed).
// Both hold only for the Delaunay tetrahedralization, which the perturbation
// makes unique. Also every edge of the tetrahedra has exactly one
// representative (a Voronoi face), also where the clipping dropped a
// rounding-level face (wave, 32768 particles: 59 edges had none).

#include "check.hpp"
#include "support.hpp"

#include <map>
#include <sstream>

namespace {

using namespace support;
namespace ep = exact_predicates;

struct Verdict {
    long long flat = 0, unpaired = 0, folded = 0, wrong_adjacency = 0, unused = 0, not_empty = 0;
    bool box_volume = false;
};

Verdict oracle(const Positions& x, const tet_mesh::Mesh<Memory>& mesh) {
    const auto vertex = to_host(mesh.vertex);
    const auto adjacent = to_host(mesh.adjacent);
    const auto hx = to_host(x);
    const int n = static_cast<int>(hx.extent(0)), count = static_cast<int>(vertex.extent(0));
    ep::Points<Kokkos::HostSpace> point("point", n);
    for (int i = 0; i < n; ++i)
        for (int a = 0; a < 3; ++a) point(i, a) = static_cast<int>(ep::snap(hx(i, a)));
    Verdict verdict;
    std::map<std::array<int, 3>, std::vector<std::pair<int, int>>> triangles;  // -> (tetrahedron, apex corner)
    std::vector<char> used(n, 0);
    ep::i128 volume6 = 0;
    for (int t = 0; t < count; ++t) {
        const int v[4] = {vertex(t, 0), vertex(t, 1), vertex(t, 2), vertex(t, 3)};
        const ep::i128 det = ep::det3(ep::offset(point, v[0], v[1]), ep::offset(point, v[0], v[2]),
                                      ep::offset(point, v[0], v[3]));
        verdict.flat += det == 0;
        volume6 += det < 0 ? -det : det;
        for (int k = 0; k < 4; ++k) {
            used[v[k]] = 1;
            std::array<int, 3> f;
            tet_mesh::sorted_face(v, k, f.data());
            triangles[f].push_back({t, k});
        }
    }
    for (const auto& [f, pair] : triangles) {
        if (pair.size() != 2) {
            ++verdict.unpaired;
            continue;
        }
        const auto [t, k] = pair[0];
        const auto [u, j] = pair[1];
        verdict.wrong_adjacency += adjacent(t, k) != u || adjacent(u, j) != t;
        // Apexes on opposite sides of the triangle.
        const int a[4] = {f[0], f[1], f[2], vertex(t, k)}, b[4] = {f[0], f[1], f[2], vertex(u, j)};
        verdict.folded += ep::orient(point, a) * ep::orient(point, b) >= 0;
    }
    verdict.unused = n - std::count(used.begin(), used.end(), 1);
    verdict.box_volume = volume6 == 6 * (ep::i128(1) << 90);
    // Emptiness: the circumcentre from exact integers (rounded once), so a
    // particle clearly outside needs no exact test.
    auto cross = [](const ep::Offset& p, const ep::Offset& q, ep::i128* r) {
        r[0] = ep::i128(p[1]) * q[2] - ep::i128(p[2]) * q[1];
        r[1] = ep::i128(p[2]) * q[0] - ep::i128(p[0]) * q[2];
        r[2] = ep::i128(p[0]) * q[1] - ep::i128(p[1]) * q[0];
    };
    for (int t = 0; t < count; ++t) {
        const int v[4] = {vertex(t, 0), vertex(t, 1), vertex(t, 2), vertex(t, 3)};
        const ep::Offset a = ep::offset(point, v[0], v[1]), b = ep::offset(point, v[0], v[2]),
                         c = ep::offset(point, v[0], v[3]);
        ep::i128 bc[3], ca[3], ab[3];
        cross(b, c, bc);
        cross(c, a, ca);
        cross(a, b, ab);
        const double det = double(ep::det3(a, b, c));
        double centre[3], r2 = 0;
        for (int k = 0; k < 3; ++k) {
            centre[k] = double(ep::lift(a) * bc[k] + ep::lift(b) * ca[k] + ep::lift(c) * ab[k]) / (2 * det);
            r2 += centre[k] * centre[k];
        }
        bool empty = true;
        for (int e = 0; e < n && empty; ++e) {
            if (e == v[0] || e == v[1] || e == v[2] || e == v[3]) continue;
            const ep::Offset d = ep::offset(point, v[0], e);
            double d2 = 0;
            for (int k = 0; k < 3; ++k) d2 += (double(d[k]) - centre[k]) * (double(d[k]) - centre[k]);
            if (d2 <= r2 * (1 + 1e-9) + 1) empty = !ep::in_sphere(point, v, e);
        }
        verdict.not_empty += !empty;
    }
    return verdict;
}

// Edges of the tetrahedra with no representative, and with more than one.
std::pair<long long, long long> representatives(const tet_mesh::Mesh<Memory>& mesh) {
    const auto vertex = to_host(mesh.vertex);
    const auto edges = to_host(mesh.edges);
    std::map<std::pair<int, int>, int> count;
    for (int t = 0; t < static_cast<int>(vertex.extent(0)); ++t)
        for (int k = 0; k < 6; ++k) {
            int a, b;
            tet_mesh::edge_ends(k, a, b);
            count[std::minmax(vertex(t, a), vertex(t, b))] += edges(t) >> k & 1;
        }
    long long none = 0, many = 0;
    for (const auto& [edge, reps] : count) {
        none += reps == 0;
        many += reps > 1;
    }
    return {none, many};
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    Check check;
    const int n = 4096;
    for (const std::string kind : {"jittered", "uniform", "clustered", "lattice", "wave", "sc@1e-6", "bcc@0",
                                   "fcc@0", "fcc@1e-6", "shells@0", "shells@1e-9", "planes@0", "pairs@1e-6"}) {
        const Positions x = to_device(in_slot_order(make_points(kind, n)));
        tet_mesh::Mesh<Memory> mesh;
        const tet_start::Stats stats = start_mesh(x, mesh);
        const Verdict v = oracle(x, mesh);
        std::ostringstream what;
        what << kind << " (" << x.extent(0) << "): tetrahedra " << stats.tets << ", first check " << stats.bad_tets
             << " bad, " << stats.patched << " cells patched in " << stats.rounds << " rounds; flat " << v.flat
             << " unpaired " << v.unpaired << " folded " << v.folded << " adjacency " << v.wrong_adjacency
             << " unused " << v.unused << " box volume " << (v.box_volume ? "yes" : "NO") << " not empty "
             << v.not_empty;
        check.expect(stats.exact && v.flat + v.unpaired + v.folded + v.wrong_adjacency + v.unused + v.not_empty == 0 &&
                         v.box_volume,
                     what.str());
        const auto [none, many] = representatives(mesh);
        check.expect(none + many == 0, kind + ": every edge has one representative (none " + std::to_string(none) +
                                            ", several " + std::to_string(many) + ")");
        // Random input needs no patch: the clipping's rings are already right.
        if (kind == "jittered") check.expect(stats.patched == 0, "jittered: nothing patched");
        // The oracle is not blind: it rejects the clipping's guess on a lattice.
        if (kind == "lattice") {
            tet_mesh::Mesh<Memory> guess;
            tet_mesh::from_rings(Exec{}, cells_with_rings(x), guess);
            const Verdict g = oracle(x, guess);
            check.expect(!g.box_volume && g.unpaired > 0, "lattice: the oracle rejects tetrahedra from the clipping's rings");
        }
    }
    {  // larger, representatives only (the oracle above is quadratic)
        const Positions x = to_device(in_slot_order(make_points("wave", 32768)));
        tet_mesh::Mesh<Memory> mesh;
        const tet_start::Stats stats = start_mesh(x, mesh);
        const auto [none, many] = representatives(mesh);
        check.expect(stats.exact && none + many == 0, "wave (32768): every edge has one representative (none " +
                                                          std::to_string(none) + ", several " + std::to_string(many) + ")");
    }
    return check.failed() != 0;
}
