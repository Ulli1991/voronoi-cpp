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
// rounding-level face (wave, 32768 particles: 59 edges had none). Boxes:
// cubes of side 1, 2, 3, a 1 x 1 x 0.125 slab (8 cells across) and a
// 1 x 2 x 3 box. The emptiness test scans only the particles whose x lies
// within the circumradius (sorted by x; brute force otherwise).

#include "check.hpp"
#include "support.hpp"

#include <algorithm>
#include <map>
#include <sstream>
#include <utility>
#include <vector>

namespace {

using namespace support;
namespace ep = exact_predicates;

struct Verdict {
    long long flat = 0, unpaired = 0, folded = 0, wrong_adjacency = 0, unused = 0, not_empty = 0;
    bool box_volume = false;
};

Verdict oracle(const Positions& x, const Box& box, const tet_mesh::Mesh<Memory>& mesh) {
    const auto vertex = to_host(mesh.vertex);
    const auto adjacent = to_host(mesh.adjacent);
    const auto hx = to_host(x);
    const int n = static_cast<int>(hx.extent(0)), count = static_cast<int>(vertex.extent(0));
    ep::Points<Kokkos::HostSpace> point("point", n, box);
    for (int i = 0; i < n; ++i)
        for (int a = 0; a < 3; ++a) point(i, a) = ep::snap(hx(i, a), box, a);
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
    verdict.box_volume = volume6 == 6 * ep::i128(box.period[0]) * box.period[1] * box.period[2];
    std::vector<std::pair<ep::i64, int>> by_x(n);  // particles sorted by snapped x
    for (int i = 0; i < n; ++i) by_x[i] = {point(i, 0), i};
    std::sort(by_x.begin(), by_x.end());
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
        auto test = [&](int e) {
            if (e == v[0] || e == v[1] || e == v[2] || e == v[3]) return;
            const ep::Offset d = ep::offset(point, v[0], e);
            double d2 = 0;
            for (int k = 0; k < 3; ++k) d2 += (double(d[k]) - centre[k]) * (double(d[k]) - centre[k]);
            if (d2 <= r2 * (1 + 1e-9) + 1) empty = !ep::in_sphere(point, v, e);
        };
        // Particles with x within the radius (and a margin) of the centre,
        // in every periodic image; all of them if that spans the box.
        const double period = double(box.period[0]), reach = std::sqrt(r2 * (1 + 1e-9) + 1) + 2;
        const double lo = point(v[0], 0) + centre[0] - reach, hi = point(v[0], 0) + centre[0] + reach;
        if (hi - lo >= period) {
            for (int e = 0; e < n && empty; ++e) test(e);
        } else {
            for (const double shift : {-period, 0.0, period}) {
                auto it = std::lower_bound(by_x.begin(), by_x.end(), std::make_pair(ep::i64(std::ceil(lo + shift)), -1));
                for (; it != by_x.end() && double(it->first) <= hi + shift && empty; ++it) test(it->second);
            }
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
    struct Case {
        std::string kind;
        Box box;
        int n = 4096;
        std::string name = "";  // of the box
    };
    // Boxes of side 2 and 3 as well: snapping divides by the box (box 3
    // positions are not on the grid; the start is exact on their snapped
    // positions all the same). Non-cubic boxes: a slab of aspect 1:1:0.125
    // with 8 lattice cells across (64 x 64 x 8; fewer would leave cells
    // uncertified, see voronoi_mesh_test), with a degenerate lattice, and a
    // 1:2:3 box (sides not whole numbers of grid steps: rounded).
    const Box slab(1, 1, 0.125), box123(1, 2, 3);
    for (const Case& c : {Case{"jittered", 1}, Case{"uniform", 1}, Case{"clustered", 1}, Case{"lattice", 1},
                         Case{"wave", 1}, Case{"sc@1e-6", 1}, Case{"bcc@0", 1}, Case{"fcc@0", 1}, Case{"fcc@1e-6", 1},
                         Case{"shells@0", 1}, Case{"shells@1e-9", 1}, Case{"planes@0", 1}, Case{"pairs@1e-6", 1},
                         Case{"jittered", 3, 4096, "a box of side 3"}, Case{"lattice", 2, 4096, "a box of side 2"},
                         Case{"jittered", slab, 32768, "a 1:1:0.125 slab"}, Case{"sc@0", slab, 32768, "a 1:1:0.125 slab"},
                         Case{"uniform", box123, 4096, "a 1:2:3 box"}, Case{"bcc@0", box123, 4096, "a 1:2:3 box"}}) {
        const std::string kind = c.name.empty() ? c.kind : c.kind + " in " + c.name;
        const Positions x = to_device(c.box.cubic() ? scaled(in_slot_order(make_points(c.kind, c.n)), c.box)
                                                    : in_slot_order(make_points(c.kind, c.n, c.box), c.box));
        tet_mesh::Mesh<Memory> mesh;
        const tet_start::Stats stats = start_mesh(x, mesh, c.box);
        const Verdict v = oracle(x, c.box, mesh);
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
        // The oracle is not blind: it rejects the mesh with one corner moved
        // to another particle.
        if (kind == "jittered") {
            tet_mesh::Mesh<Memory> broken = mesh;
            broken.vertex = tet_mesh::Quads<Memory>("broken", mesh.vertex.extent(0));
            Kokkos::deep_copy(broken.vertex, mesh.vertex);
            auto corner = Kokkos::subview(broken.vertex, 0, 0);
            Kokkos::deep_copy(corner, (to_host(Kokkos::subview(mesh.vertex, 0, 0))() + 1000) % x.extent(0));
            const Verdict g = oracle(x, 1.0, broken);
            check.expect(g.unpaired > 0 && g.not_empty > 0, "jittered: the oracle rejects a mesh with one corner moved");
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
