// The memory layout of voronoi::Mesh<3> (#34): the curves of mesh_order, the permutation contract of a re-sort (cells
// renumbered, the mesh as a set unchanged), faces numbered by owner and the owner bit (bits unchanged).

#include "check.hpp"
#include "support.hpp"

#include <map>
#include <numeric>
#include <set>

namespace {

using namespace support;
using voronoi::Curve;

// Grid cells of a 2^b cube in the order of `curve`, from keys on the top b of the axis_bits bits.
std::vector<std::array<std::uint32_t, 3>> walk(Curve curve, int b) {
    const auto moore = mesh_order::MooreTable::build(mesh_order::axis_bits - 1);
    std::vector<std::pair<std::uint64_t, std::array<std::uint32_t, 3>>> cells;
    const std::uint32_t side = 1u << b, shift = mesh_order::axis_bits - b;
    for (std::uint32_t x = 0; x < side; ++x)
        for (std::uint32_t y = 0; y < side; ++y)
            for (std::uint32_t z = 0; z < side; ++z) {
                const std::uint32_t c[3] = {x << shift, y << shift, z << shift};
                cells.push_back({mesh_order::key(curve, c, moore), {x, y, z}});
            }
    std::sort(cells.begin(), cells.end());
    std::vector<std::array<std::uint32_t, 3>> out;
    for (const auto& c : cells) out.push_back(c.second);
    return out;
}

// Whether two cells share a face.
bool adjacent(const std::array<std::uint32_t, 3>& a, const std::array<std::uint32_t, 3>& b) {
    int distance = 0;
    for (int k = 0; k < 3; ++k) distance += std::abs(static_cast<int>(a[k]) - static_cast<int>(b[k]));
    return distance == 1;
}

// One cell as the label it started with sees it: neighbours by label, volume, centroid.
struct Cell {
    std::vector<int> neighbours;
    double volume;
    Point centroid;
};

// The cells of a mesh after update_cells, by label (label[i]: the particle cell i was first). Used by: the re-sort section.
std::vector<Cell> cells_by_label(const Mesh& mesh, const std::vector<int>& label) {
    const int n = mesh.particles();
    const auto start = to_host(mesh.cell_face_start());
    const auto list = to_host(mesh.cell_face(), 2 * mesh.faces());
    const auto owner = to_host(mesh.owner(), mesh.faces());
    const auto neighbour = to_host(mesh.neighbour(), mesh.faces());
    const auto volume = to_host(mesh.volume());
    const auto centroid = to_host(mesh.centroid());
    std::vector<Cell> cells(n);
    for (int i = 0; i < n; ++i) {
        Cell& c = cells[label[i]];
        for (int k = start(i); k < start(i + 1); ++k) {
            const int f = Mesh::list_face(list(k));
            c.neighbours.push_back(label[owner(f) == i ? neighbour(f) : owner(f)]);
        }
        std::sort(c.neighbours.begin(), c.neighbours.end());
        c.volume = volume(i);
        c.centroid = {centroid(i, 0), centroid(i, 1), centroid(i, 2)};
    }
    return cells;
}

// Face areas by (owner, neighbour), bits. Used by: the owner-faces section.
std::map<std::pair<int, int>, Point> areas_by_pair(const Mesh& mesh) {
    const auto owner = to_host(mesh.owner(), mesh.faces());
    const auto neighbour = to_host(mesh.neighbour(), mesh.faces());
    const auto area = face_areas(mesh);
    std::map<std::pair<int, int>, Point> out;
    for (int f = 0; f < mesh.faces(); ++f) out[{owner(f), neighbour(f)}] = {area(f, 0), area(f, 1), area(f, 2)};
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    Check check;

    {  // the curves on an 8^3 grid: continuous where they should be, each a permutation
        const auto hilbert = walk(Curve::hilbert, 3), moore = walk(Curve::moore, 3), gray = walk(Curve::gray, 3);
        const auto row = walk(Curve::row_major, 3), morton = walk(Curve::morton, 3);
        long long hilbert_jumps = 0, moore_jumps = 0, gray_bits = 0;
        for (std::size_t q = 1; q < hilbert.size(); ++q) {
            hilbert_jumps += !adjacent(hilbert[q - 1], hilbert[q]);
            moore_jumps += !adjacent(moore[q - 1], moore[q]);
            const auto code = [](const std::array<std::uint32_t, 3>& c) {
                return neighbor_search::morton(c[0], c[1], c[2]);
            };
            gray_bits += __builtin_popcountll(code(gray[q - 1]) ^ code(gray[q])) != 1;
        }
        check.expect(hilbert_jumps == 0, "Hilbert: each cell next to the one before");
        check.expect(moore_jumps == 0 && adjacent(moore.back(), moore.front()), "Moore: a closed loop of neighbours");
        check.expect(gray_bits == 0, "Gray-coded Morton: Morton codes one bit apart");
        bool rows = true;
        for (std::size_t q = 0; q < row.size(); ++q)
            rows &= row[q][0] == q % 8 && row[q][1] == q / 8 % 8 && row[q][2] == q / 64;
        check.expect(rows, "row-major: x fastest");
        std::set<std::array<std::uint32_t, 3>> all;
        for (const auto* w : {&hilbert, &moore, &gray, &row, &morton}) all.insert(w->begin(), w->end());
        check.expect(all.size() == 512 && morton[1] == std::array<std::uint32_t, 3>{0, 0, 1},
                     "every curve visits every cell");
    }

    const double box = 2.0;
    const Points start_points = make_points_off_grid("jittered", 4096);
    const int n = static_cast<int>(start_points.size());
    const double spacing = std::cbrt(1.0 / n);
    const Flow flow;

    for (const Curve curve : {Curve::morton, Curve::hilbert, Curve::moore, Curve::gray, Curve::row_major, Curve::rcm}) {
        // A re-sort every second update: the caller follows the permutation; the mesh as a set equals an unsorted one.
        const std::string name = "curve " + std::to_string(static_cast<int>(curve)) + ": ";
        Points points = start_points, sorted_points = start_points;
        std::vector<int> label(n);
        std::iota(label.begin(), label.end(), 0);
        Mesh plain(Exec{}, box), sorted(Exec{}, box);
        voronoi::Layout layout;
        layout.curve = curve;
        layout.every = 2;
        layout.owner_faces = curve == Curve::hilbert || curve == Curve::rcm;
        layout.owner_bit = curve != Curve::morton;
        sorted.set_layout(layout);
        long long wrong_permutation = 0, wrong_follow = 0, wrong_sets = 0, resorts = 0;
        double worst_volume = 0, worst_centroid = 0;
        for (int s = 0; s <= 6; ++s) {
            if (s > 0) {
                flow.advect(points, 0.3 * spacing);
                flow.advect(sorted_points, 0.3 * spacing);
            }
            plain.update(to_device(scaled(points, box)));
            Positions x = to_device(scaled(sorted_points, box));
            const voronoi::Report r = sorted.update(x);
            plain.update_cells();
            sorted.update_cells();
            if (r.resorted) {
                ++resorts;
                const auto p = to_host(sorted.permutation());
                std::vector<int> seen(n, 0), relabel(n);
                Points moved(n);
                for (int i = 0; i < n; ++i) {
                    wrong_permutation += p(i) < 0 || p(i) >= n || seen[p(i)]++;
                    moved[i] = sorted_points[p(i)];
                    relabel[i] = label[p(i)];
                }
                sorted_points.swap(moved);
                label.swap(relabel);
                sorted.permute(x);
                const auto followed = to_host(x), expected = to_host(to_device(scaled(sorted_points, box)));
                const auto snapped = to_host(sorted.positions());
                for (int i = 0; i < n; ++i)
                    for (int a = 0; a < 3; ++a) {
                        wrong_follow += followed(i, a) != expected(i, a);
                        wrong_follow += snapped(i, a) !=
                                        exact_predicates::snap(expected(i, a), sorted.sides(), a) * sorted.sides().step;
                    }
            }
            std::vector<int> identity(n);
            std::iota(identity.begin(), identity.end(), 0);
            const auto a = cells_by_label(plain, identity), b = cells_by_label(sorted, label);
            for (int i = 0; i < n; ++i) {
                wrong_sets += a[i].neighbours != b[i].neighbours;
                worst_volume = std::max(worst_volume, std::abs(a[i].volume - b[i].volume) / a[i].volume);
                for (int k = 0; k < 3; ++k)
                    worst_centroid =
                        std::max(worst_centroid, std::abs(a[i].centroid[k] - b[i].centroid[k]) / (spacing * box));
            }
        }
        check.expect(resorts == 4 && wrong_permutation == 0,
                     name + "re-sorted at the build and every second update, a permutation each time");
        check.expect(wrong_follow == 0,
                     name + "permute() carries the caller's rows, and the mesh's positions are those rows");
        check.expect(wrong_sets == 0, name + "every particle keeps its neighbours, by its first number");
        check.expect(
            worst_volume < 1e-12 && worst_centroid < 1e-12,
            name + "volumes and centroids equal to rounding (" + sci(worst_volume) + ", " + sci(worst_centroid) + ")");
    }

    {  // faces by owner and the owner bit, without a re-sort: the same ids, the same bits
        Points points = start_points;
        Mesh plain(Exec{}, box), owned(Exec{}, box);
        voronoi::Layout layout;
        layout.owner_faces = layout.owner_bit = true;
        owned.set_layout(layout);
        long long unsorted = 0, wrong_start = 0, wrong_bits = 0, differing = 0, wrong_entries = 0;
        for (int s = 0; s <= 6; ++s) {
            if (s > 0) flow.advect(points, 0.03 * spacing);
            const Positions x = to_device(scaled(points, box));
            plain.update(x);
            const voronoi::Report r = owned.update(x);
            unsorted += r.resorted;
            plain.update_cells();
            owned.update_cells();
            const auto owner = to_host(owned.owner(), owned.faces());
            const auto first = to_host(owned.owned_start());
            for (int f = 1; f < owned.faces(); ++f) unsorted += owner(f - 1) > owner(f);
            for (int i = 0; i < n; ++i)
                for (int f = first(i); f < first(i + 1); ++f) wrong_start += owner(f) != i;
            wrong_start += first(n) != owned.faces();
            const auto pa = to_host(plain.volume()), pb = to_host(owned.volume());
            const auto ca = to_host(plain.centroid()), cb = to_host(owned.centroid());
            for (int i = 0; i < n; ++i) {
                wrong_bits += pa(i) != pb(i);
                for (int a = 0; a < 3; ++a) wrong_bits += ca(i, a) != cb(i, a);
            }
            differing += areas_by_pair(plain) != areas_by_pair(owned);
            const auto start = to_host(owned.cell_face_start());
            const auto list = to_host(owned.cell_face(), 2 * owned.faces());
            for (int i = 0; i < n; ++i)
                for (int k = start(i); k < start(i + 1); ++k) {
                    const int f = Mesh::list_face(list(k));
                    wrong_entries += Mesh::list_owned(list(k)) != (owner(f) == i);
                }
        }
        check.expect(unsorted == 0 && wrong_start == 0, "faces by owner; owned_start names each cell's faces");
        check.expect(differing == 0 && wrong_bits == 0, "the same face areas, volumes and centroids, bit for bit");
        check.expect(wrong_entries == 0, "list entries carry the owner bit");
    }
    return check.failed() != 0;
}
