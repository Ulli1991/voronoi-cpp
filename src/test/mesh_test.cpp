// Persistent tetrahedra from a clipping build: from_rings yields the same
// tetrahedra as a host construction from the rings, with consistent
// adjacency and one representative per edge; the Voronoi geometry from
// their circumcentres (on the stored rings, and on rings walked from the
// edge representatives) matches the clipped cells, and the local Delaunay
// check passes. The circumcentre of a sliver is exact.

#include "check.hpp"
#include "support.hpp"

#include <map>
#include <set>

namespace {

using namespace support;

// Host reference: tetrahedron (i, j, ring[m], ring[m + 1]) for every owned
// face i-j and ring slot m, as sorted particle quadruples.
std::set<std::array<int, 4>> tets_from_rings(const shell_voronoi::Cells<Memory>& cells) {
    const auto owned = to_host(cells.owned_count);
    const auto start = to_host(cells.ring_start);
    const auto rings = to_host(cells.rings);
    std::set<std::array<int, 4>> tets;
    for (int i = 0; i < static_cast<int>(owned.extent(0)); ++i)
        for (long long f = 0, at = start(i); f < owned(i); ++f) {
            const int j = rings(at), k = rings(at + 1);
            for (int m = 0; m < k; ++m) {
                std::array<int, 4> key{i, j, rings(at + 2 + m), rings(at + 2 + (m + 1) % k)};
                std::sort(key.begin(), key.end());
                tets.insert(key);
            }
            at += 2 + k;
        }
    return tets;
}

double worst_volume_difference(const tet_mesh::Geometry<Memory>& geometry, const shell_voronoi::Cells<Memory>& cells) {
    const auto mesh_volume = to_host(geometry.volume);
    const auto clipped_volume = to_host(cells.volume);
    double worst = 0;
    for (std::size_t i = 0; i < mesh_volume.extent(0); ++i)
        worst = std::max(worst, std::abs(mesh_volume(i) - clipped_volume(i)) / clipped_volume(i));
    return worst;
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    Check check;
    const Exec exec;
    const int n = 4096;
    for (const std::string kind : {"jittered", "uniform"}) {
        std::cout << kind << ":\n";
        const Positions x = to_device(in_slot_order(make_points(kind, n)));
        const auto cells = cells_with_rings(x);
        const auto status = to_host(cells.status);
        check.expect(std::count(status.data(), status.data() + n, int(shell_voronoi::ok)) == n, "clipping build ok");
        {  // a ring pool too small grows to what the build needs
            auto small = shell_voronoi::Cells<Memory>::with_rings(n, 1000);
            shell_voronoi::construct_warp<double>(exec, shell_voronoi::make_grid(cell_list(x)), small);
            const long long needed = small.needed(), entries = cells.needed();
            const auto small_status = to_host(small.status);
            check.expect(needed == entries && small.capacity() >= needed &&
                             std::count(small_status.data(), small_status.data() + n, int(shell_voronoi::ok)) == n,
                         "a small ring pool grows to the entries needed (" + std::to_string(double(entries) / n) +
                             " per cell)");
        }

        tet_mesh::Mesh<Memory> mesh;
        check.expect(tet_mesh::from_rings(exec, cells, mesh) == 0, "from_rings resolves every ring slot and neighbour");
        const int count = static_cast<int>(mesh.vertex.extent(0));
        const auto vertex = to_host(mesh.vertex);
        const auto adjacent = to_host(mesh.adjacent);
        const auto edges = to_host(mesh.edges);
        std::set<std::array<int, 4>> built;
        for (int t = 0; t < count; ++t) {
            std::array<int, 4> key{vertex(t, 0), vertex(t, 1), vertex(t, 2), vertex(t, 3)};
            std::sort(key.begin(), key.end());
            built.insert(key);
        }
        check.expect(static_cast<int>(built.size()) == count && built == tets_from_rings(cells),
                     "from_rings gives the host's tetrahedra, each once (" + std::to_string(count) + ")");

        // Across face k lies a tetrahedron sharing that face and pointing back.
        int inconsistent = 0;
        for (int t = 0; t < count; ++t)
            for (int k = 0; k < 4; ++k) {
                const int u = adjacent(t, k);
                int shared = 0, back = 0;
                for (int a = 0; a < 4 && u >= 0; ++a) {
                    for (int b = 0; b < 4; ++b) shared += a != k && vertex(t, a) == vertex(u, b);
                    back += adjacent(u, a) == t;
                }
                inconsistent += u < 0 || shared != 3 || back != 1;
            }
        check.expect(inconsistent == 0, "adjacency is mutual across shared faces");

        // Every owned face (Delaunay edge) has exactly one representative.
        std::vector<std::pair<int, int>> represented, owned_faces;
        for (int t = 0; t < count; ++t)
            for (int k = 0; k < 6; ++k)
                if (edges(t) >> k & 1) {
                    int a, b;
                    tet_mesh::edge_ends(k, a, b);
                    represented.push_back(std::minmax(vertex(t, a), vertex(t, b)));
                }
        const auto owned = to_host(cells.owned_count);
        const auto start = to_host(cells.ring_start);
        const auto rings = to_host(cells.rings);
        for (int i = 0; i < n; ++i)
            for (long long f = 0, at = start(i); f < owned(i); ++f, at += 2 + rings(at + 1))
                owned_faces.push_back({i, rings(at)});
        std::sort(represented.begin(), represented.end());
        std::sort(owned_faces.begin(), owned_faces.end());
        check.expect(represented == owned_faces, "one edge representative per face");

        // Geometry on the stored rings, then on rings walked around the edges.
        tet_mesh::Points4<Memory> padded("padded", n);
        tet_mesh::pad(exec, x, padded);
        tet_mesh::circumcentres(exec, padded, 1.0, mesh);
        const int faces = static_cast<int>(owned_faces.size());
        tet_mesh::Geometry<Memory> geometry{Kokkos::View<double* [3], Memory>("area", faces),
                                            Kokkos::View<double* [3], Memory>("centroid", faces),
                                            Kokkos::View<double*, Memory>("volume", n)};
        tet_mesh::faces(exec, padded, 1.0, mesh, geometry);
        double worst = worst_volume_difference(geometry, cells);
        check.expect(worst < 1e-12, "volumes from stored rings match clipping (" + sci(worst) + ")");
        Kokkos::View<int*, Memory> invalid("invalid", count);
        check.expect(tet_mesh::check(exec, padded, 1.0, mesh, invalid) == 0, "local Delaunay check passes");

        auto walked = tet_mesh::make_face_rings<Memory>(count, faces, 8 * faces);  // no rings yet: walk all
        Kokkos::View<char*, Memory> touched("touched", count);
        check.expect(tet_mesh::update_face_rings(exec, mesh, count, touched, {}, walked) == faces,
                     "walking the represented edges finds every face");
        tet_mesh::faces(exec, padded, 1.0, mesh, geometry);
        worst = worst_volume_difference(geometry, cells);
        check.expect(worst < 1e-12, "volumes from walked rings match clipping (" + sci(worst) + ")");
    }

    // A sliver: three corners on a circle in a grid plane, the fourth one grid
    // step above a point near that circle. Its circumcentre in grid units is
    // (A / 2, B / 2, (C^2 + D^2 + 1 - A C - B D) / 2) from corner 0.
    {
        using ep_i128 = exact_predicates::i128;
        const long long A = 3000017, B = 2000003;
        const double radius = 0.5 * std::sqrt(double(A) * A + double(B) * B);
        const long long C = std::llround(0.5 * A + radius * std::cos(1.0)), D = std::llround(0.5 * B + radius * std::sin(1.0));
        const long long g[4][3] = {{0, 0, 0}, {A, 0, 0}, {0, B, 0}, {C, D, 1}};
        const double step = 1.0 / exact_predicates::scale, origin = 0.25;
        Positions x("sliver", 4);
        tet_mesh::Mesh<Memory> mesh;
        mesh.vertex = tet_mesh::Quads<Memory>("vertex", 1);
        mesh.centre = tet_mesh::Points4<Memory>("centre", 1);
        auto hx = Kokkos::create_mirror_view(x);
        auto hv = Kokkos::create_mirror_view(mesh.vertex);
        for (int k = 0; k < 4; ++k) {
            hv(0, k) = k;
            for (int a = 0; a < 3; ++a) hx(k, a) = origin + g[k][a] * step;  // on the snapping grid
        }
        Kokkos::deep_copy(x, hx);
        Kokkos::deep_copy(mesh.vertex, hv);
        tet_mesh::circumcentres(exec, x, 1.0, mesh);
        const auto centre = to_host(mesh.centre);
        const ep_i128 twice_z = ep_i128(C) * C + ep_i128(D) * D + 1 - ep_i128(A) * C - ep_i128(B) * D;
        const double expected[3] = {origin + 0.5 * A * step, origin + 0.5 * B * step, origin + 0.5 * double(twice_z) * step};
        double error = 0, r2 = 0;
        for (int a = 0; a < 3; ++a) {
            error = std::max(error, std::abs(centre(0, a) - expected[a]));
            r2 += (expected[a] - origin) * (expected[a] - origin);
        }
        check.expect(error < 1e-12 * std::sqrt(r2), "sliver circumcentre is exact (error / radius " +
                                                       sci(error / std::sqrt(r2)) + ")");
    }
    return check.failed() != 0;
}
