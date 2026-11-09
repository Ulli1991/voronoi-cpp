// Repair of the persistent mesh as particles move under a smooth flow: after each step it is Delaunay and matches an exact
// clipped rebuild in neighbours and face volumes. Exercises voronoi::Mesh::update (flip repair) and update_cells.

#include "check.hpp"
#include "support.hpp"

#include <sys/mman.h>
#include <unistd.h>

#include <map>
#include <random>
#include <sstream>

namespace {

// Local rebuilds in a small region with open faces reach a bounding corner, which used to become the frame's origin,
// particle -1 (#35). The positions start a page after a guard page, so such a read ends the program in any build.
void test_local_rebuild_near_a_corner(Check& check) {
    using Host = Kokkos::HostSpace;
    namespace ep = exact_predicates;
    constexpr int n = 12;
    const long page = sysconf(_SC_PAGESIZE);
    void* pages = mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check.expect(pages != MAP_FAILED && mprotect(pages, page, PROT_NONE) == 0, "a guard page before the positions");
    int* rows = reinterpret_cast<int*>(static_cast<char*>(pages) + page);
    std::mt19937 rng(5);
    std::vector<ep::Offset> local;
    std::vector<int> id;
    for (int i = 0; i < n; ++i) {
        for (int a = 0; a < 3; ++a) rows[4 * i + a] = 1000 + static_cast<int>(rng() % 1000);
        local.push_back({rows[4 * i], rows[4 * i + 1], rows[4 * i + 2]});
        id.push_back(i);
    }
    std::vector<tet_flips::Tet> tets;
    for (const auto& t : tet_flips::detail::delaunay(local, id))
        if (*std::max_element(t.begin(), t.end()) < n) tets.push_back(t);
    const int count = static_cast<int>(tets.size()), capacity = 4 * count + 256;
    const ep::Box box(1.0);
    long long rebuilt = 0;
    for (int seed = 0; seed < count; ++seed) {
        tet_flips::Mesh m{tet_mesh::Quads<Host>("vertex", capacity),
                          tet_mesh::Quads<Host>("adjacent", capacity),
                          Kokkos::View<char*, Host>("alive", capacity),
                          {ep::Points<Host>::View(rows, n), box.period},
                          count};
        std::map<std::array<int, 3>, std::pair<int, int>> open;
        for (int t = 0; t < count; ++t) {
            m.alive(t) = 1;
            for (int k = 0; k < 4; ++k) {
                m.vertex(t, k) = tets[t][k];
                m.adjacent(t, k) = -1;
            }
            for (int k = 0; k < 4; ++k) {
                const auto key = tet_flips::detail::face_key(tets[t], k);
                const auto it = open.find(key);
                if (it == open.end()) {
                    open[key] = {t, k};
                    continue;
                }
                m.adjacent(t, k) = it->second.first;
                m.adjacent(it->second.first, it->second.second) = t;
            }
        }
        rebuilt += !tet_flips::detail::rebuild(m, seed, -1).empty();
    }
    check.expect(count > 0, "local rebuilds near a bounding corner read no particle -1 (" + std::to_string(count) +
                                " seeds, " + std::to_string(rebuilt) + " replaced)");
    munmap(pages, 2 * page);
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    using namespace support;
    Check check;
    test_local_rebuild_near_a_corner(check);
    const Flow flow;
    // One flow run: point kind, spacings moved per step, step count, box, particle count, label for the box.
    struct Case {
        const char* kind;
        double displacement;
        int steps;
        Box box = 1.0;
        int n = 4096;
        const char* name = "";
    };
    for (const Case& c : {Case{"jittered", 0.03, 10}, Case{"jittered", 0.3, 10}, Case{"lattice", 0.03, 4},
                         Case{"lattice", 0.3, 4}, Case{"wave", 0.03, 4}, Case{"wave", 0.3, 4},
                         Case{"shells@1e-9", 0.03, 2}, Case{"jittered", 0.3, 3, 2.0, 4096, "2"},
                         Case{"uniform", 0.3, 3, Box(1, 2, 3), 4096, "1:2:3"},
                         Case{"jittered", 0.3, 2, Box(1, 1, 0.125), 32768, "1:1:0.125"}}) {
        const double displacement = c.displacement;
        const int steps = c.steps;
        const bool cube = c.box.cubic();
        std::cout << c.kind << ", displacement " << displacement << " spacings per step, box "
                  << (*c.name ? c.name : "1") << ":\n";
        // Cubes: points in the unit box, scaled; other boxes: in the box.
        Points points = cube ? make_points(c.kind, c.n) : make_points(c.kind, c.n, c.box);
        const int n = static_cast<int>(points.size());
        const double spacing = cube ? std::cbrt(1.0 / n) : std::cbrt(c.box.volume() / n);
        Positions x = to_device(cube ? scaled(points, c.box) : points);
        Mesh mesh(Exec{}, c.box);
        check.expect(mesh.update(x).start.exact, "exact starting mesh");
        mesh.update_cells();
        for (int step = 1; step <= steps; ++step) {
            if (cube)
                flow.advect(points, displacement * spacing);
            else
                flow.advect(points, displacement * spacing, c.box);
            x = to_device(cube ? scaled(points, c.box) : points);
            const voronoi::Report report = mesh.update(x);
            mesh.update_cells();
            // The mesh against an exact clipping rebuild of the same positions (support::compare_with_rebuild).
        const Comparison d = compare_with_rebuild(mesh, x);
            std::ostringstream what;
            what << "step " << step << ": flips " << report.flips << " stuck " << report.stuck << " host flips "
                 << report.host_flips << " rebuilt " << report.host_rebuilds << "; " << d;
            check.expect(d.failing == 0 && d.mismatched == 0 && d.faces == d.clipped_faces && d.volume < 1e-10 && d.wrong_lists == 0 &&
                             std::abs(d.total - 1.0) < 1e-12,
                         what.str());
        }
    }
    return check.failed() != 0;
}
