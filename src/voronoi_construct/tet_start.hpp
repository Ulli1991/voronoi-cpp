#pragma once

#include "exact_predicates.hpp"
#include "shell_voronoi.hpp"
#include "tet_mesh.hpp"

#include <cstdint>

// Stage 3 (start): called by voronoi::Mesh<3>::rebuild after shell_voronoi clipping; feeds tet_repair::prepare.
// Exactly the Delaunay tetrahedralization of the snapped positions (ties: symbolic perturbation of exact_predicates);
// a clipping guess is checked exactly and wrong cells rebuilt by exact clipping.
namespace tet_start {

namespace ep = exact_predicates;

// What build() did. Used by: Mesh::rebuild (Report::start; `exact` false -> refusal via not_exact).
struct Stats {
    int rounds = 0;              // checks of a guess (1: the clipping's rings were right)
    long long bad_tets = 0;      // tetrahedra failing the first check
    long long patched = 0;       // cells rebuilt by exact clipping, all rounds
    long long tets = 0;          // tetrahedra of the result
    bool exact = false;          // the result passed the check
};

// Result of check(): tetrahedra that are flat, lack a neighbour, disagree with one, fold over one or fail the local
// Delaunay test; particles in no tetrahedron; and the total volume (box volumes).
// Used by: build.
struct Report {
    long long bad_tets = 0, unused = 0;
    double volume = 0;
    bool ok() const { return bad_tets == 0 && unused == 0 && volume > 0.5 && volume < 1.5; }
};

namespace detail {

// Particles in tetrahedra holding a marked particle: marked -> grown (the cells to re-clip around a bad one).
// Used by: build.
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

// The volume of a tetrahedron in box volumes (unit3 = 1 / box volume in grid steps), and its corner offsets from v[0].
template <class Points>
KOKKOS_INLINE_FUNCTION double volume(const Points& point, const int* v, double unit3, ep::Offset* o) {
    o[0] = {0, 0, 0};
    for (int k = 1; k < 4; ++k) o[k] = ep::offset(point, v[0], v[k]);
    return Kokkos::abs(double(ep::det3(o[1], o[2], o[3]))) * unit3 / 6;
}

// The sign of ep::orient(a, b, c) where its double filter decides it, else 0.
KOKKOS_INLINE_FUNCTION int orient_filter(const ep::Offset& a, const ep::Offset& b, const ep::Offset& c) {
    double permanent;
    const double d = ep::det3_float(a, b, c, permanent), bound = 8 * ep::epsilon * permanent;
    return d > bound ? 1 : d < -bound ? -1 : 0;
}

// ep::in_sphere(point, v, o, e) where the double filters decide it: 1 inside, -1 not, 0 undecided.
template <class Points>
KOKKOS_INLINE_FUNCTION int in_sphere_filter(const Points& point, const int* v, const ep::Offset* o, int e) {
    const ep::Offset x = ep::offset(point, v[0], e);
    const int orientation = orient_filter(o[1], o[2], o[3]);
    auto lift = [](const ep::Offset& y) {
        return double(y[0]) * double(y[0]) + double(y[1]) * double(y[1]) + double(y[2]) * double(y[2]);
    };
    double pa, pb, pc, pe;
    const double ta = ep::det3_float(o[2], o[3], x, pa), tb = ep::det3_float(o[1], o[3], x, pb),
                 tc = ep::det3_float(o[1], o[2], x, pc), te = ep::det3_float(o[1], o[2], o[3], pe);
    const double la = lift(o[1]), lb = lift(o[2]), lc = lift(o[3]), le = lift(x);
    const double value = -la * ta + lb * tb - lc * tc + le * te;
    const double bound = 20 * ep::epsilon * (la * pa + lb * pb + lc * pc + le * pe);
    const int d = value > bound ? 1 : value < -bound ? -1 : 0;
    if (orientation == 0 || d == 0) return 0;
    return d * orientation < 0 ? 1 : -1;
}

// The check of tetrahedron t (see check), corners v (offsets o from v[0]) and neighbours adj: 1 fine; 0 bad, marking its
// particles and a failing far vertex; -1 undecided by the double filters (Filtered only; nothing marked).
template <bool Filtered, class Quads, class Points, class Marked>
KOKKOS_INLINE_FUNCTION int check_one(const Quads& vertex, const Quads& adjacent, const Points& point,
                                     const Marked& marked, int t, int* v, int* adj, const ep::Offset* o) {
    const int orientation = Filtered ? orient_filter(o[1], o[2], o[3]) : ep::orient(o[1], o[2], o[3]);
    if (Filtered && orientation == 0) return -1;
    bool fine = orientation != 0;
    if (orientation < 0) {
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
        if constexpr (Filtered) {
            const int side = orient_filter(ep::offset(point, s[0], s[1]), ep::offset(point, s[0], s[2]),
                                           ep::offset(point, s[0], s[3]));
            if (side == 0) return -1;
            if (side < 0) {
                const int inside = in_sphere_filter(point, v, o, s[k]);
                if (inside == 0) return -1;
                fine = inside < 0;
            } else {
                fine = false;
            }
        } else {
            fine = ep::orient(point, s) < 0 && !ep::in_sphere(point, v, o, s[k]);
        }
        if (!fine) marked(s[k]) = 1;
    }
    if (!fine)
        for (int k = 0; k < 4; ++k) marked(v[k]) = 1;
    return fine;
}

}  // namespace detail

// A check after an incremental assemble: only tetrahedra redone or next to one are tested, as the others passed the
// round before with the same neighbours; `volume` is that round's less its tetrahedra of changed owners, to which the
// new ones are added. Empty: a full check. Used by: check, build.
template <class MemorySpace>
struct Partial {
    Kokkos::View<const char*, MemorySpace> redone;
    Kokkos::View<const int*, MemorySpace> changed;
    double volume = 0;
};

// Storage of check kept from round to round (see tet_mesh::Assembly::Store). Used by: check, build.
template <class MemorySpace>
struct CheckStore {
    Kokkos::View<int*, MemorySpace> used, listed, undecided;
};

// Exact check of a tetrahedralization of all n particles, one thread per tetrahedron: it is not flat; across each face
// lies a tetrahedron sharing it and pointing back, its far vertex on the other side and outside this circumsphere
// (local Delaunay, perturbed); every particle is used; the volumes sum to the box. Marks the particles of bad
// tetrahedra and the unused ones in `marked` (1), leaving other entries as they are. `count` = tetrahedra to check.
// Used by: build.
template <class ExecSpace>
Report check(const ExecSpace& exec, const tet_mesh::Mesh<typename ExecSpace::memory_space>& mesh,
             const ep::Points<typename ExecSpace::memory_space>& point, int count,
             const Kokkos::View<int*, typename ExecSpace::memory_space>& marked,
             const Partial<typename ExecSpace::memory_space>& partial = {},
             CheckStore<typename ExecSpace::memory_space>* kept = nullptr) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const int n = static_cast<int>(point.extent(0));
    using tet_mesh::detail::take;
    CheckStore<Memory> local;
    auto& store = kept ? *kept : local;
    const Kokkos::View<int*, Memory> used = take(store.used, n, "tet_start_used");
    Kokkos::deep_copy(exec, used, 0);
    const auto vertex = mesh.vertex;
    const auto adjacent = mesh.adjacent;
    const double unit3 = 1.0 / (double(point.period[0]) * double(point.period[1]) * double(point.period[2]));
    const bool some = partial.redone.extent(0) > 0;
    const auto redone = partial.redone;
    const auto changed = partial.changed;
    // Incremental: every tetrahedron marks its particles used, and those redone or next to one are listed.
    const Kokkos::View<int*, Memory> listed = take(store.listed, some ? count : 0, "tet_start_listed");
    int checked = count;
    if (some) {
        Kokkos::parallel_scan(
            "tet_start::near_redone", Range(exec, 0, count),
            KOKKOS_LAMBDA(int t, int& at, bool last) {
                bool look = redone(t);
                for (int k = 0; k < 4; ++k) {
                    if (last) used(vertex(t, k)) = 1;
                    const int u = adjacent(t, k);
                    look |= u >= 0 && redone(u);
                }
                if (!look) return;
                if (last) listed(at) = t;
                ++at;
            },
            checked);
    }
    // Most tetrahedra are decided by the double filters; the rest go to `undecided` and the exact predicates.
    const Kokkos::View<int*, Memory> undecided = take(store.undecided, checked, "tet_start_undecided");
    Kokkos::View<int, Memory> undecided_count("tet_start_undecided_count");
    Report report;
    Kokkos::parallel_reduce(
        "tet_start::check", Range(exec, 0, checked),
        KOKKOS_LAMBDA(int i, long long& bad, double& volume) {
            const int t = some ? listed(i) : i;
            int v[4] = {vertex(t, 0), vertex(t, 1), vertex(t, 2), vertex(t, 3)};
            int adj[4] = {adjacent(t, 0), adjacent(t, 1), adjacent(t, 2), adjacent(t, 3)};
            if (!some)
                for (int k = 0; k < 4; ++k) used(v[k]) = 1;
            ep::Offset o[4];
            const double size = detail::volume(point, v, unit3, o);
            if (!some || changed(v[0])) volume += size;
            const int state = detail::check_one<true>(vertex, adjacent, point, marked, t, v, adj, o);
            if (state < 0) undecided(Kokkos::atomic_fetch_add(&undecided_count(), 1)) = t;
            bad += state == 0;
        },
        report.bad_tets, report.volume);
    int hard = 0;
    Kokkos::deep_copy(exec, hard, undecided_count);
    exec.fence();
    if (hard > 0) {
        long long bad = 0;
        Kokkos::parallel_reduce(
            "tet_start::check_exact", Range(exec, 0, hard),
            KOKKOS_LAMBDA(int i, long long& sum) {
                const int t = undecided(i);
                int v[4] = {vertex(t, 0), vertex(t, 1), vertex(t, 2), vertex(t, 3)};
                int adj[4] = {adjacent(t, 0), adjacent(t, 1), adjacent(t, 2), adjacent(t, 3)};
                ep::Offset o[4];
                detail::volume(point, v, unit3, o);
                sum += detail::check_one<false>(vertex, adjacent, point, marked, t, v, adj, o) == 0;
            },
            bad);
        report.bad_tets += bad;
    }
    report.volume += partial.volume;
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

// The Delaunay tetrahedralization from a clipping build with face rings: assemble, check, and where it fails rebuild
// the bad cells by exact clipping and check again (3 rounds at most). After a partial rebuild the next round assembles
// and checks incrementally (only around the rebuilt cells, the same arrays as a full round). `cells` are the clipping's
// output, `mesh` the result. face_overflow means build again with a pool of Cells::needed().
// Used by: voronoi::Mesh<3>::rebuild.
template <class ExecSpace>
Stats build(const ExecSpace& exec, const shell_voronoi::Grid<typename ExecSpace::memory_space>& grid,
            const shell_voronoi::Cells<typename ExecSpace::memory_space>& cells,
            tet_mesh::Mesh<typename ExecSpace::memory_space>& mesh) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const int n = grid.n;
    Stats stats;
    // Grid positions indexed by particle id (grid.positions is in the cell list's sorted order).
    ep::Points<Memory> point("tet_start_point", n, grid.box);
    const auto positions = grid.positions;
    const auto order = grid.order;
    const ep::Box box = grid.box;
    Kokkos::parallel_for(
        "tet_start::snap", Range(exec, 0, n), KOKKOS_LAMBDA(int q) {
            for (int a = 0; a < 3; ++a) point(order(q), a) = ep::snap(positions(q, a), box, a);
        });
    const auto stencil = shell_voronoi::make_stencil(exec);
    Kokkos::View<int*, Memory> marked("tet_start_marked", n), grown("tet_start_grown", n), todo("tet_start_todo", n);
    // Cells the clipping did not finish count as bad from the start.
    const auto status = cells.status;
    Kokkos::parallel_for(
        "tet_start::unfinished", Range(exec, 0, n), KOKKOS_LAMBDA(int i) { marked(i) = status(i) != 0; });
    const double unit3 = 1.0 / (double(point.period[0]) * double(point.period[1]) * double(point.period[2]));
    bool all = false;
    tet_mesh::Assembly<Memory> assembly;
    tet_mesh::Previous<Memory> previous;
    CheckStore<Memory> check_store;
    double carried = 0;  // volume of the last round's tetrahedra of unchanged owners
    for (int round = 0; round < 3; ++round) {
        tet_mesh::assemble(exec, cells, mesh, assembly, previous);
        const Partial<Memory> partial{assembly.redone, previous.changed, carried};
        previous = {};
        const int count = assembly.counts.tets;
        const Report report = check(exec, mesh, point, count, marked, partial, &check_store);
        ++stats.rounds;
        if (round == 0) stats.bad_tets = report.bad_tets;
        stats.tets = count;
        if (report.ok()) {
            tet_mesh::allocate_centres(mesh);
            tet_mesh::represent(exec, mesh);
            stats.exact = true;
            break;
        }
        if (all || round == 2) break;
        detail::grow(exec, mesh, marked, grown);
        int listed = 0;
        Kokkos::parallel_scan(
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
            Kokkos::deep_copy(exec, cells.ring_used, 0u);
        } else {
            // The next round keeps what does not depend on the re-clipped cells (grown), less their volume.
            const auto vertex = mesh.vertex;
            double removed = 0;
            Kokkos::parallel_reduce(
                "tet_start::changed_volume", Range(exec, 0, count),
                KOKKOS_LAMBDA(int t, double& sum) {
                    if (!grown(vertex(t, 0))) return;
                    const int v[4] = {vertex(t, 0), vertex(t, 1), vertex(t, 2), vertex(t, 3)};
                    ep::Offset o[4];
                    sum += detail::volume(point, v, unit3, o);
                },
                removed);
            carried = report.volume - removed;
            previous = {mesh.vertex, mesh.adjacent, mesh.ring, assembly.slot_start, assembly.tet_start, grown};
        }
        namespace sv = shell_voronoi::detail;
        sv::construct_cells<sv::ExactCell>(exec, grid, stencil, cells, sv::Pass::all,
                                           {Kokkos::subview(todo, std::make_pair(0, listed)), {}});
        stats.patched += listed;
        Kokkos::deep_copy(exec, marked, 0);
    }
    if (!stats.exact) tet_mesh::allocate_centres(mesh);
    return stats;
}

}  // namespace tet_start
