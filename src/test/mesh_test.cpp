// Persistent tetrahedra from a clipping build: from_rings equals a host construction from the rings, adjacency and edge
// representatives are consistent, and the Voronoi geometry matches the clipped cells (tet_mesh, 2D cell list).

#include "check.hpp"
#include "support.hpp"
#include "tet_cells.hpp"

#include <map>
#include <random>
#include <set>

namespace {

using namespace support;

// Host reference: tetrahedron (i, j, ring[m], ring[m + 1]) for every owned
// face i-j and ring slot m, as sorted particle quadruples.
// Used by: the from_rings check in main (tet_mesh::from_rings).
std::set<std::array<int, 4>> tets_from_rings(const shell_voronoi::Cells<Memory>& cells) {
    const auto owned = to_host(cells.owned_count);
    std::vector<long long> start;
    const auto rings = ring_entries(cells, start);
    std::set<std::array<int, 4>> tets;
    for (int i = 0; i < static_cast<int>(owned.extent(0)); ++i)
        for (long long f = 0, at = start[i]; f < owned(i); ++f) {
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

// Largest relative difference between tet_mesh geometry volumes and the clipped cells' volumes.
// Used by: the stored-ring and walked-ring volume checks in main (tet_mesh::faces).
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
        {
            auto small = shell_voronoi::Cells<Memory>::with_rings(n, 1000);
            shell_voronoi::construct_warp(exec, shell_voronoi::make_grid(cell_list(x)), small);
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
        std::vector<long long> start;
        const auto rings = ring_entries(cells, start);
        for (int i = 0; i < n; ++i)
            for (long long f = 0, at = start[i]; f < owned(i); ++f, at += 2 + rings(at + 1))
                owned_faces.push_back({i, rings(at)});
        std::sort(represented.begin(), represented.end());
        std::sort(owned_faces.begin(), owned_faces.end());
        check.expect(represented == owned_faces, "one edge representative per face");

        // Geometry on the stored rings, then on rings walked around the edges.
        tet_mesh::Points4<Memory> padded("padded", n);
        voronoi::detail::snap_positions(exec, x, 1.0, padded);
        tet_mesh::circumcentres(exec, padded, 1.0, mesh);
        const int faces = static_cast<int>(owned_faces.size());
        tet_mesh::Geometry<Memory> geometry{Kokkos::View<double* [3], Kokkos::LayoutRight, Memory>("area", faces),
                                            Kokkos::View<double* [3], Kokkos::LayoutRight, Memory>("centroid", faces),
                                            Kokkos::View<double*, Memory>("volume", n), {}};
        tet_mesh::faces(exec, padded, 1.0, mesh, geometry);
        double worst = worst_volume_difference(geometry, cells);
        check.expect(worst < 1e-12, "volumes from stored rings match clipping (" + sci(worst) + ")");
        exact_predicates::Points<Memory> point("point", n);
        exact_predicates::snap(exec, x, 1.0, point);
        Kokkos::View<int*, Memory> marked("marked", n);
        check.expect(tet_start::check(exec, mesh, point, count, marked).ok(), "the exact check passes");

        auto walked = tet_mesh::make_face_rings<Memory>(count, faces, 8 * faces);
        Kokkos::View<char*, Memory> touched("touched", count);
        check.expect(tet_mesh::update_face_rings(exec, mesh, count, touched, {}, walked) == faces,
                     "walking the represented edges finds every face");
        tet_mesh::faces(exec, padded, 1.0, mesh, geometry);
        worst = worst_volume_difference(geometry, cells);
        check.expect(worst < 1e-12, "volumes from walked rings match clipping (" + sci(worst) + ")");
    }

    // A sliver: three corners on a circle in a grid plane, the fourth one grid step above a point near that circle.
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
        // Stored in grid steps, wrapped into [0, 2^30).
        const double from_origin[3] = {0.5 * A, 0.5 * B, 0.5 * double(twice_z)};
        double error = 0, r2 = 0;
        for (int a = 0; a < 3; ++a) {
            double expected = std::fmod(origin / step + from_origin[a], exact_predicates::scale);
            if (expected < 0) expected += exact_predicates::scale;
            error = std::max(error, std::abs(centre(0, a) - expected));
            r2 += from_origin[a] * from_origin[a];
        }
        check.expect(error < 1e-12 * std::sqrt(r2), "sliver circumcentre is exact (error / radius " +
                                                       sci(error / std::sqrt(r2)) + ")");
    }
    {
        // The 2D pieces of stage 0: the 2D cell list sorts by tile and Morton key (against a host sort),
        // and tet_mesh::cells in 2D sums a square lattice's faces to cells of area h^2 and centroid 0
        // (pyramid 1/4, centroid 2/3).
        constexpr int m = 32, n = m * m;  // a power of two: on the snapping grid, as Mesh gives cells() its positions
        const exact_predicates::BoxN<2> box(2.0, 1.0);
        Kokkos::View<double* [2], Memory> x("x2", n);
        auto hx = Kokkos::create_mirror_view(x);
        std::mt19937_64 rng(3);
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        for (int i = 0; i < n; ++i) {
            hx(i, 0) = 2.0 * unit(rng);
            hx(i, 1) = unit(rng);
        }
        Kokkos::deep_copy(x, hx);
        neighbor_search::CellList<Exec, 2> list;
        neighbor_search::configure(list, n, box, std::sqrt(box.volume() / n));
        neighbor_search::build(exec, list, x);
        const auto order = to_host(list.order);
        const int depth = list.levels + neighbor_search::fine_levels;
        std::vector<std::pair<std::uint64_t, int>> keys(n);
        for (int i = 0; i < n; ++i) {
            int c[2], tile[2];
            for (int a = 0; a < 2; ++a) {
                const int g = neighbor_search::grid_coordinate(hx(i, a), box.side[a], list.tiles[a] << depth);
                tile[a] = g >> depth;
                c[a] = g & ((1 << depth) - 1);
            }
            std::uint64_t morton = 0;
            for (int b = 0; b < depth; ++b)
                morton |= (std::uint64_t(c[0] >> b & 1) << (2 * b + 1)) | (std::uint64_t(c[1] >> b & 1) << (2 * b));
            keys[i] = {(std::uint64_t(tile[0]) * list.tiles[1] + tile[1]) << (2 * depth) | morton, i};
        }
        std::sort(keys.begin(), keys.end());
        bool sorted = list.tiles[0] == 2 && list.tiles[1] == 1;
        for (int q = 0; q < n; ++q) sorted = sorted && order(q) == keys[q].second;
        check.expect(sorted, "2D cell list: 2 x 1 tiles, slots in (tile, Morton key, id) order");

        // Square lattice m x m in a unit square, spacing h; faces to the right and upper neighbours, owned by the lower
        // id.
        const double h = 1.0 / m;
        const exact_predicates::BoxN<2> square(1.0);
        Kokkos::View<double* [2], Memory> p("p2", n);
        Kokkos::View<int*, Memory> owner("owner2", 2 * n), neighbour("neighbour2", 2 * n);
        tet_mesh::Geometry<Memory, 2> geometry{Kokkos::View<double* [2], Kokkos::LayoutRight, Memory>("area2", 2 * n),
                                               Kokkos::View<double* [2], Kokkos::LayoutRight, Memory>("centroid2", 2 * n),
                                               Kokkos::View<double*, Memory>("volume2", n), {}};
        auto hp = Kokkos::create_mirror_view(p);
        auto ho = Kokkos::create_mirror_view(owner);
        auto hn = Kokkos::create_mirror_view(neighbour);
        auto ha = Kokkos::create_mirror_view(geometry.area);
        auto hc = Kokkos::create_mirror_view(geometry.centroid);
        for (int i = 0; i < n; ++i) {
            const int ix = i % m, iy = i / m;
            hp(i, 0) = (ix + 0.5) * h;
            hp(i, 1) = (iy + 0.5) * h;
            const int across[2] = {(ix + 1) % m + iy * m, ix + ((iy + 1) % m) * m};
            for (int a = 0; a < 2; ++a) {
                const int f = 2 * i + a, j = across[a];
                const double sign = i < j ? 1.0 : -1.0;
                ho(f) = std::min(i, j);
                hn(f) = std::max(i, j);
                ha(f, a) = sign * h;
                ha(f, 1 - a) = 0;
                hc(f, a) = sign * 0.5 * h;
                hc(f, 1 - a) = 0;
            }
        }
        Kokkos::deep_copy(p, hp);
        Kokkos::deep_copy(owner, ho);
        Kokkos::deep_copy(neighbour, hn);
        Kokkos::deep_copy(geometry.area, ha);
        Kokkos::deep_copy(geometry.centroid, hc);
        tet_mesh::CellFaces<Memory, 2> cells{Kokkos::View<int*, Memory>("start2", n + 1), Kokkos::View<int*, Memory>("face2", 0),
                                             Kokkos::View<int*, Memory>("cursor2", n), Kokkos::View<double*, Memory>("v2", n),
                                             Kokkos::View<double* [2], Kokkos::LayoutRight, Memory>("c2", n)};
        Kokkos::View<char*, Memory> scratch("scratch2", tet_mesh::cells_scratch_bytes(2 * n));
        tet_mesh::cells(exec, p, square, Kokkos::View<const int*, Memory>(owner), Kokkos::View<const int*, Memory>(neighbour),
                        geometry, cells, scratch.data());
        const auto volume = to_host(cells.volume);
        const auto centroid = to_host(cells.centroid);
        const auto start = to_host(cells.start);
        double worst_volume = 0, worst_centroid = 0;
        bool four = true;
        for (int i = 0; i < n; ++i) {
            worst_volume = std::max(worst_volume, std::abs(volume(i) - h * h) / (h * h));
            worst_centroid = std::max({worst_centroid, std::abs(centroid(i, 0)) / h, std::abs(centroid(i, 1)) / h});
            four = four && start(i + 1) - start(i) == 4;
        }
        check.expect(four && worst_volume < 1e-14 && worst_centroid < 1e-14,
                     "2D tet_mesh::cells on a square lattice: 4 faces per cell, area h^2 (" + sci(worst_volume) +
                         "), centroid at the generator (" + sci(worst_centroid) + ")");
    }
    return check.failed() != 0;
}
