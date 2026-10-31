// The contract of voronoi::Mesh, the hydro code's interface, on positions
// off the snapping grid in a box of side 2:
//   - the first update builds the mesh; later ones repair it, and after
//     each the faces are those of an exact rebuild (neighbours, volumes);
//     on near-cospherical shells, where the device repair leaves stuck
//     triangles, Fallback::host (the host finishes them) and
//     Fallback::rebuild (the step rebuilds) are exact too;
//   - faces are stored once, owner < neighbour, and close every cell;
//     volumes fill the box; cell centroids equal those of clipping;
//   - the per-cell face lists (update_cells) name each cell's faces once,
//     in increasing id of the other particle;
//   - particle ids are the caller's: a permuted input gives the permuted
//     mesh; the same input gives the same bits;
//   - a local rebuild by clipping returns each face of the listed cells
//     once, equal to the mesh's;
//   - the scratch memory can be used between updates.

#include "check.hpp"
#include "support.hpp"

#include <map>
#include <sstream>

namespace {

using namespace support;

using Face = std::pair<int, int>;

// The faces of a mesh by particle pair: area vector from the lower id.
std::map<Face, Point> faces_of(const Mesh& mesh, const std::vector<int>& rename = {}) {
    const auto owner = to_host(mesh.owner());
    const auto neighbour = to_host(mesh.neighbour());
    const auto area = face_areas(mesh);
    std::map<Face, Point> faces;
    for (int f = 0; f < mesh.faces(); ++f) {
        int i = owner(f), j = neighbour(f);
        double sign = 1;
        if (!rename.empty()) {
            i = rename[i];
            j = rename[j];
        }
        if (i > j) {
            std::swap(i, j);
            sign = -1;
        }
        faces[{i, j}] = {sign * area(f, 0), sign * area(f, 1), sign * area(f, 2)};
    }
    return faces;
}

double difference(const Point& a, const Point& b) {
    return std::max({std::abs(a[0] - b[0]), std::abs(a[1] - b[1]), std::abs(a[2] - b[2])});
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    Check check;
    const double box = 2.0;
    Points points = make_points_off_grid("jittered", 4096);
    const int n = static_cast<int>(points.size());
    const double spacing = box * std::cbrt(1.0 / n), area_scale = spacing * spacing;
    Positions x = to_device(scaled(points, box));

    Mesh mesh(Exec{}, box);
    const voronoi::Report first = mesh.update(x);
    mesh.update_cells();
    check.expect(first.rebuilt && first.start.exact, "the first update builds an exact mesh");

    {  // faces once, owner < neighbour, closing every cell; volumes fill the box
        const auto owner = to_host(mesh.owner());
        const auto neighbour = to_host(mesh.neighbour());
        const auto area = face_areas(mesh);
        const auto volume = to_host(mesh.volume());
        std::vector<Point> closure(n, Point{0, 0, 0});
        std::vector<double> magnitude(n, 0);
        long long ordered = 0;
        for (int f = 0; f < mesh.faces(); ++f) {
            ordered += owner(f) < neighbour(f);
            for (int a = 0; a < 3; ++a) {
                closure[owner(f)][a] += area(f, a);
                closure[neighbour(f)][a] -= area(f, a);
            }
            const double m = std::sqrt(area(f, 0) * area(f, 0) + area(f, 1) * area(f, 1) + area(f, 2) * area(f, 2));
            magnitude[owner(f)] += m;
            magnitude[neighbour(f)] += m;
        }
        double worst = 0, total = 0;
        for (int i = 0; i < n; ++i) {
            worst = std::max(worst, std::sqrt(closure[i][0] * closure[i][0] + closure[i][1] * closure[i][1] +
                                              closure[i][2] * closure[i][2]) / magnitude[i]);
            total += volume(i);
        }
        check.expect(ordered == mesh.faces() && static_cast<int>(faces_of(mesh).size()) == mesh.faces(),
                     "faces are stored once, owner < neighbour");
        check.expect(worst < 1e-12, "the faces close every cell (" + sci(worst) + ")");
        check.expect(std::abs(total / (box * box * box) - 1) < 1e-12, "volumes fill the box");
    }
    {  // per-cell face lists
        const auto start = to_host(mesh.cell_face_start());
        const auto face = to_host(mesh.cell_face());
        const auto owner = to_host(mesh.owner());
        const auto neighbour = to_host(mesh.neighbour());
        std::vector<int> seen(mesh.faces(), 0);
        long long wrong = 0;
        for (int i = 0; i < n; ++i) {
            int last = -1;
            for (int a = start(i); a < start(i + 1); ++a) {
                const int f = face(a), other = owner(f) == i ? neighbour(f) : owner(f);
                wrong += (owner(f) != i && neighbour(f) != i) || other <= last;
                last = other;
                ++seen[f];
            }
        }
        wrong += std::count_if(seen.begin(), seen.end(), [](int s) { return s != 2; });
        check.expect(wrong == 0 && start(n) == 2 * mesh.faces(),
                     "each cell lists its faces once, by increasing id of the other particle");
    }
    {  // cell centroids equal those of clipping (both relative to the snapped generator)
        const auto list = cell_list(mesh.positions(), box);
        shell_voronoi::Cells<Memory> cells(n, false);
        shell_voronoi::construct_warp(Exec{}, shell_voronoi::make_grid(list), cells);
        const auto clipped_centroid = to_host(cells.centroid);
        const auto centroid = to_host(mesh.centroid());
        double worst = 0;
        for (int i = 0; i < n; ++i)
            for (int a = 0; a < 3; ++a) worst = std::max(worst, std::abs(centroid(i, a) - clipped_centroid(i, a)));
        check.expect(worst < 1e-10 * spacing, "cell centroids equal those of clipping (" + sci(worst / spacing) +
                                                  " spacings)");
    }
    {  // the same input gives the same bits; a permuted input the permuted mesh
        Mesh again(Exec{}, box);
        again.update(x);
        again.update_cells();
        const auto v1 = to_host(mesh.volume()), v2 = to_host(again.volume());
        const auto c1 = to_host(mesh.centroid()), c2 = to_host(again.centroid());
        bool same = true;
        for (int i = 0; i < n; ++i)
            same = same && v1(i) == v2(i) && c1(i, 0) == c2(i, 0) && c1(i, 1) == c2(i, 1) && c1(i, 2) == c2(i, 2);
        check.expect(same, "the same input gives the same volumes and centroids, bit for bit");

        std::vector<int> permutation(n);  // particle k of the permuted input is particle permutation[k]
        for (int k = 0; k < n; ++k) permutation[k] = static_cast<int>((7919LL * k + 13) % n);
        Points permuted(n);
        for (int k = 0; k < n; ++k) permuted[k] = points[permutation[k]];
        Mesh other(Exec{}, box);
        other.update(to_device(scaled(permuted, box)));
        const auto faces = faces_of(mesh), renamed = faces_of(other, permutation);
        double worst = 0;
        bool same_faces = faces.size() == renamed.size();
        for (const auto& [pair, a] : faces) {
            const auto it = renamed.find(pair);
            if (it == renamed.end()) {
                same_faces = false;
                break;
            }
            worst = std::max(worst, difference(a, it->second) / area_scale);
        }
        const auto v3 = to_host(other.volume());
        double volume = 0;
        for (int k = 0; k < n; ++k)
            volume = std::max(volume, std::abs(v3(k) - v1(permutation[k])) / v1(permutation[k]));
        check.expect(same_faces && worst < 1e-12 && volume < 1e-12,
                     "a permuted input gives the permuted faces (areas " + sci(worst) + ", volumes " + sci(volume) +
                         ")");
    }
    {  // a local rebuild returns each face of the listed cells once, equal to the mesh's
        std::vector<int> listed;
        for (int i = 0; i < n; i += 7) listed.push_back(i);
        Kokkos::View<int*, Memory> cells("cells", listed.size());
        Kokkos::deep_copy(cells, Kokkos::View<int*, Kokkos::HostSpace>(listed.data(), listed.size()));
        const auto& local = mesh.clip(x, cells);
        const auto status = to_host(local.status);
        const auto owned = to_host(local.owned_count);
        const auto start = to_host(local.face_start);
        const auto neighbour = to_host(local.neighbour);
        const auto area = to_host(local.area);
        const auto faces = faces_of(mesh);
        std::vector<char> is_listed(n, 0);
        for (int i : listed) is_listed[i] = 1;
        std::map<Face, int> returned;
        double worst = 0;
        long long bad = 0;
        for (int i : listed) {
            bad += status(i) != shell_voronoi::ok;
            for (int f = start(i); f < start(i) + owned(i); ++f) {
                const int j = neighbour(f);
                const double sign = i < j ? 1 : -1;
                const Face pair = std::minmax(i, j);
                ++returned[pair];
                const auto it = faces.find(pair);
                if (it == faces.end()) {
                    ++bad;
                    continue;
                }
                worst = std::max(worst, difference({sign * area(f, 0), sign * area(f, 1), sign * area(f, 2)}, it->second) /
                                            area_scale);
            }
        }
        long long expected = 0;
        for (const auto& [pair, a] : faces) {
            if (!is_listed[pair.first] && !is_listed[pair.second]) continue;
            ++expected;
            bad += returned[pair] != 1;
        }
        check.expect(bad == 0 && static_cast<long long>(returned.size()) == expected && worst < 1e-9,
                     "a local rebuild returns each face of the listed cells once (" + std::to_string(expected) +
                         " faces, areas within " + sci(worst) + " spacings^2)");
    }
    {  // the scratch memory between updates, then steps, then both fallbacks
        auto scratch = mesh.scratch(64 * static_cast<std::size_t>(n));
        mesh.set_fallback(voronoi::Fallback::host);  // nothing sticks here: never a rebuild
        Kokkos::deep_copy(scratch, char(7));
        const Flow flow;
        bool exact = true, repaired = true;
        std::ostringstream what;
        for (int step = 1; step <= 3; ++step) {
            flow.advect(points, 0.3 * spacing / box);
            x = to_device(scaled(points, box));
            const voronoi::Report r = mesh.update(x);
            const Comparison c = compare_with_rebuild(mesh, x);
            exact = exact && c.pass();
            repaired = repaired && !r.rebuilt;
            what << " step " << step << ": " << (c.pass() ? "exact" : "NOT exact") << ", flips " << r.flips;
        }
        check.expect(exact && repaired, "updates repair the mesh, exact after each:" + what.str());

        for (const auto fallback : {voronoi::Fallback::host, voronoi::Fallback::rebuild}) {
            const bool host = fallback == voronoi::Fallback::host;
            Points shells = make_points("shells@1e-9", 4096);
            Positions xs = to_device(scaled(shells, box));
            Mesh stuck(Exec{}, box);
            stuck.set_fallback(fallback);
            stuck.update(xs);
            flow.advect(shells, 0.3 * spacing / box);
            xs = to_device(scaled(shells, box));
            const voronoi::Report r = stuck.update(xs);
            const Comparison c = compare_with_rebuild(stuck, xs);
            check.expect(r.stuck > 0 && (host || r.rebuilt) && c.pass(),
                         std::string(host ? "Fallback::host" : "Fallback::rebuild") +
                             ": a step the device repair leaves stuck (" + std::to_string(r.stuck) + "; host flips " +
                             std::to_string(r.host_flips) + ", local rebuilds " + std::to_string(r.host_rebuilds) +
                             ", rebuilt " + std::to_string(r.rebuilt) + ") is exact");
        }
    }
    return check.failed() != 0;
}
