#pragma once

#include "exact_predicates.hpp"
#include "shell_voronoi.hpp"
#include "tet_mesh.hpp"

#include <cstdint>

// The starting tetrahedralization: exactly the Delaunay tetrahedralization
// of the snapped positions, ties broken by the symbolic perturbation of
// exact_predicates, for any input (lattices and other cospherical layouts
// included).
//
// The face rings of a clipping build give a fast guess (tet_mesh::from_rings),
// but rounding decides its near-ties, so neighbouring cells can disagree:
// holes and overlaps on near-degenerate input, now and then on random input.
// build() checks the guess exactly and rebuilds the cells around whatever is
// wrong by the exact per-thread clipping (shell_voronoi::construct: every
// vertex is named by the three particles whose bisectors meet there, and
// every cut is the exact perturbed in-sphere sign, so cells agree with each
// other by construction).
namespace tet_start {

namespace ep = exact_predicates;

struct Stats {
    int rounds = 0;              // checks of a guess (1: the clipping's rings were right)
    long long bad_tets = 0;      // tetrahedra failing the first check
    long long patched = 0;       // cells rebuilt by exact clipping, all rounds
    long long tets = 0;          // tetrahedra of the result
    bool exact = false;          // the result passed the check
};

// Result of check(): tetrahedra that are flat, lack a neighbour, disagree
// with one, fold over one or fail the local Delaunay test; particles in no
// tetrahedron; and the total volume (box units).
struct Report {
    long long bad_tets = 0, unused = 0;
    double volume = 0;
    bool ok() const { return bad_tets == 0 && unused == 0 && volume > 0.5 && volume < 1.5; }
};

namespace detail {

// Particles in tetrahedra holding a marked particle: marked -> grown.
template <class ExecSpace>
void grow(const ExecSpace& exec, const tet_mesh::Mesh<typename ExecSpace::memory_space>& mesh,
          const Kokkos::View<int*, typename ExecSpace::memory_space>& marked,
          const Kokkos::View<int*, typename ExecSpace::memory_space>& grown) {
    Kokkos::deep_copy(exec, grown, marked);
    Kokkos::parallel_for(
        "tet_start::grow", Kokkos::RangePolicy<ExecSpace>(exec, 0, mesh.vertex.extent(0)), KOKKOS_LAMBDA(int t) {
            bool any = false;
            for (int k = 0; k < 4; ++k) any |= marked(mesh.vertex(t, k)) != 0;
            if (any)
                for (int k = 0; k < 4; ++k) grown(mesh.vertex(t, k)) = 1;
        });
}

}  // namespace detail

// Exact check of a tetrahedralization of all n particles, one thread per
// tetrahedron: it is not flat; across each face lies a tetrahedron sharing
// that face and pointing back, with its far vertex on the other side and
// outside this circumsphere (local Delaunay, perturbed); every particle is
// used; the volumes sum to the box. Together these make it the Delaunay
// tetrahedralization: a closed, consistently oriented complex covers the box
// a whole number of times (the volume says once), and a locally Delaunay
// triangulation is Delaunay. Marks the particles of bad tetrahedra and the
// unused ones in `marked` (1), leaving other entries as they are.
template <class ExecSpace>
Report check(const ExecSpace& exec, const tet_mesh::Mesh<typename ExecSpace::memory_space>& mesh,
             const ep::Points<typename ExecSpace::memory_space>& point, int count,
             const Kokkos::View<int*, typename ExecSpace::memory_space>& marked) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const int n = static_cast<int>(point.extent(0));
    Kokkos::View<int*, Memory> used("tet_start_used", n);
    const auto vertex = mesh.vertex;
    const auto adjacent = mesh.adjacent;
    constexpr double unit3 = 1.0 / (ep::scale * ep::scale * ep::scale);
    Report report;
    Kokkos::parallel_reduce(
        "tet_start::check", Range(exec, 0, count),
        KOKKOS_LAMBDA(int t, long long& bad, double& volume) {
            int v[4] = {vertex(t, 0), vertex(t, 1), vertex(t, 2), vertex(t, 3)};
            int adj[4] = {adjacent(t, 0), adjacent(t, 1), adjacent(t, 2), adjacent(t, 3)};
            for (int k = 0; k < 4; ++k) used(v[k]) = 1;
            const ep::Offset o[4] = {{0, 0, 0}, ep::offset(point, v[0], v[1]), ep::offset(point, v[0], v[2]),
                                     ep::offset(point, v[0], v[3])};
            const int orientation = ep::orient(o[1], o[2], o[3]);
            volume += Kokkos::abs(double(ep::det3(o[1], o[2], o[3]))) * unit3 / 6;
            bool fine = orientation != 0;
            if (orientation < 0) {  // as tet_repair::prepare orients it
                Kokkos::kokkos_swap(v[2], v[3]);
                Kokkos::kokkos_swap(adj[2], adj[3]);
            }
            for (int k = 0; k < 4 && fine; ++k) {
                const int u = adj[k];
                if (u < 0) {
                    fine = false;
                    break;
                }
                int far = -1, shared = 0;
                for (int c = 0; c < 4; ++c) {
                    const int p = vertex(u, c);
                    int at = -1;
                    for (int d = 0; d < 4; ++d)
                        if (v[d] == p) at = d;
                    if (at < 0) {
                        far = c;
                    } else {
                        shared += at != k;
                    }
                }
                if (far < 0 || shared != 3 || adjacent(u, far) != t) {
                    fine = false;
                    break;
                }
                // The geometric tests are symmetric: once per triangle.
                if (u < t) continue;
                int s[4] = {v[0], v[1], v[2], v[3]};
                s[k] = vertex(u, far);
                fine = ep::orient(point, s) < 0 && !ep::in_sphere(point, v, o, s[k]);
                if (!fine) marked(s[k]) = 1;
            }
            if (!fine) {
                ++bad;
                for (int k = 0; k < 4; ++k) marked(v[k]) = 1;
            }
        },
        report.bad_tets, report.volume);
    Kokkos::parallel_reduce(
        "tet_start::unused", Range(exec, 0, n),
        KOKKOS_LAMBDA(int i, long long& missing) {
            if (used(i)) return;
            ++missing;
            marked(i) = 1;
        },
        report.unused);
    return report;
}

// The Delaunay tetrahedralization from a clipping build with face rings
// (shell_voronoi::construct of `grid`, every cell ok or not): from_rings,
// check, and where it fails, rebuild the cells of the bad tetrahedra and
// their neighbours by exact clipping, then assemble and check again. When
// many cells are bad, or the second try still fails, every cell is rebuilt.
// Rebuilt cells claim new blocks in the ring pool of `cells`
// (Cells::with_rings); a cell that does not fit is face_overflow, and the
// result is not exact: build again with a pool of Cells::needed().
template <class ExecSpace>
Stats build(const ExecSpace& exec, const shell_voronoi::Grid<typename ExecSpace::memory_space>& grid,
            const shell_voronoi::Cells<typename ExecSpace::memory_space>& cells,
            tet_mesh::Mesh<typename ExecSpace::memory_space>& mesh) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const int n = grid.n;
    Stats stats;
    // Snapped positions by particle.
    ep::Points<Memory> point("tet_start_point", n);
    const auto positions = grid.positions;
    const auto order = grid.order;
    const double box = grid.box;
    Kokkos::parallel_for(
        "tet_start::snap", Range(exec, 0, n), KOKKOS_LAMBDA(int q) {
            for (int a = 0; a < 3; ++a) point(order(q), a) = ep::snap(positions(q, a), box);
        });
    const auto stencil = shell_voronoi::make_stencil(exec);
    Kokkos::View<int*, Memory> marked("tet_start_marked", n), grown("tet_start_grown", n), todo("tet_start_todo", n);
    // Cells the clipping did not finish count as bad from the start.
    const auto status = cells.status;
    Kokkos::parallel_for(
        "tet_start::unfinished", Range(exec, 0, n), KOKKOS_LAMBDA(int i) { marked(i) = status(i) != 0; });
    bool all = false;
    for (int round = 0; round < 3; ++round) {
        tet_mesh::from_rings(exec, cells, mesh);
        const int count = static_cast<int>(mesh.vertex.extent(0));
        const Report report = check(exec, mesh, point, count, marked);
        ++stats.rounds;
        if (round == 0) stats.bad_tets = report.bad_tets;
        stats.tets = count;
        if (report.ok()) {
            stats.exact = true;
            break;
        }
        if (all || round == 2) break;
        detail::grow(exec, mesh, marked, grown);
        int listed = 0;
        Kokkos::parallel_scan(  // the slots of the grown particles
            "tet_start::list", Range(exec, 0, n),
            KOKKOS_LAMBDA(int q, int& at, bool last) {
                if (!grown(order(q))) return;
                if (last) todo(at) = q;
                ++at;
            },
            listed);
        // Many bad cells (a lattice), or a second try: rebuild them all.
        all = round > 0 || listed > n / 8;
        if (all) {
            Kokkos::parallel_for(
                "tet_start::all", Range(exec, 0, n), KOKKOS_LAMBDA(int i) { todo(i) = i; });
            listed = n;
            Kokkos::deep_copy(exec, cells.ring_used, 0u);  // every block is written again
        }
        shell_voronoi::detail::construct_cells(exec, grid, stencil, cells, shell_voronoi::detail::Pass::listed, todo,
                                               listed);
        stats.patched += listed;
        Kokkos::deep_copy(exec, marked, 0);
    }
    return stats;
}

}  // namespace tet_start
