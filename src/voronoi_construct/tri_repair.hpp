#pragma once

#include "tri_mesh.hpp"

#include <Kokkos_Core.hpp>

#include <string>
#include <utility>

// 2D repair stage, device: called by Mesh<2>::update after snap, then tri_finish on stuck triangles; feeds tri_mesh::faces.
// Keeps the triangulation Delaunay as particles move: screen() lists failing edges, repair() flips them in rounds (Lawson 2-2).
// Conflicting flips are settled by votes in per-triangle locks whose keys grow with the round, so winners do not depend on list order.
// Inversions no flip removes stay stuck, for tri_finish::finish() or a rebuild in voronoi::Mesh.
namespace tri_repair {

namespace ep = exact_predicates;
using tri_mesh::Triangles;

// Used by: repair, Mesh<2>::update.
struct Stats {
    long long flips = 0;
    int rounds = 0, stuck = 0;
};

// The repair's work lists and counters (device), kept across updates.
// Used by: screen, repair, tri_finish::finish, Mesh<2>.
template <class MemorySpace>
struct State {
    Kokkos::View<unsigned long long*, MemorySpace> lock;  // of a triangle
    Kokkos::View<int*, MemorySpace> list, next;           // edges to test: this round, the next
    Kokkos::View<int*, MemorySpace> touched;              // triangles that flipped, for their centres
    Kokkos::View<int*, MemorySpace> flipped;              // faces that flipped; one for two touched entries
    Kokkos::View<int[2], MemorySpace> used;               // device counters: next, touched
    unsigned long long round = 0;
    int entries = 0, touched_count = 0;
    bool inverted = false;  // the screen or the repair saw an inverted or flat triangle
    double area = 0;        // the screen's sum of twice the signed areas, in grid steps^2
    bool area_stale = false;  // the screen saw an inversion: total_area() decides, not `area`

    State() = default;
    State(int triangles, int capacity)
        : lock("tri_lock", triangles),
          list(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("tri_list")), capacity),
          next(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("tri_next")), capacity),
          touched(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("tri_touched")), capacity),
          flipped(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("tri_flipped")), capacity),
          used("tri_used") {}
};

namespace detail {
// Used by: screen, repair, tri_finish::finish.
template <class V>
void reserve(V& v, std::size_t size, bool keep) {
    if (v.extent(0) >= size) return;
    if (keep)
        Kokkos::resize(v, size + size / 2);
    else
        Kokkos::realloc(Kokkos::WithoutInitializing, v, size + size / 2);
}
KOKKOS_INLINE_FUNCTION unsigned hash(unsigned x) {
    x *= 0x9e3779b1u;
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    return x;
}
}  // namespace detail

// See the top of this file.
// Used by: screen.
template <class Memory>
struct Screen {
    using Points = ep::Points<Memory, 2>;
    Kokkos::View<const int* [6], Kokkos::LayoutRight, Memory> corners;
    tri_mesh::Rows2<Memory> centre;
    Points point;
    Kokkos::View<int*, Memory> list;
    Kokkos::View<int[2], Memory> used;
    double step;
    int px, py, capacity;

    KOKKOS_INLINE_FUNCTION void push(int edge) const {
        const int at = Kokkos::atomic_fetch_inc(&used(0));
        if (at < capacity) list(at) = edge;
    }
    KOKKOS_INLINE_FUNCTION void operator()(int t, long long& failing, long long& bad, double& area) const {
        const int v0 = corners(t, 0), v1 = corners(t, 1), v2 = corners(t, 2);
        const int x0 = point(v0, 0), y0 = point(v0, 1);
        const double ax = tri_mesh::steps(x0, point(v1, 0), px), ay = tri_mesh::steps(y0, point(v1, 1), py);
        const double bx = tri_mesh::steps(x0, point(v2, 0), px), by = tri_mesh::steps(y0, point(v2, 1), py);
        const tri_mesh::Circle circle(ax, ay, bx, by);
        const double la = circle.la, lb = circle.lb, mx = circle.mx, my = circle.my, det = circle.det;
        area += det;
        tet_mesh::store(centre, t, circle.centre(x0, y0, step));
        const double spread = Kokkos::abs(ax * by) + Kokkos::abs(ay * bx);
        if (!(det > 8 * ep::epsilon * spread)) {
            const ep::i64 exact = static_cast<ep::i64>(ax) * static_cast<ep::i64>(by) - static_cast<ep::i64>(ay) * static_cast<ep::i64>(bx);
            if (exact <= 0) {
                ++bad;
                failing += 3;
                for (int k = 0; k < 3; ++k) push(t << 2 | k);
                return;
            }
        }
        const double pmx = la * Kokkos::abs(bx) + lb * Kokkos::abs(ax), pmy = la * Kokkos::abs(by) + lb * Kokkos::abs(ay);
        for (int k = 0; k < 3; ++k) {
            const int d = corners(t, 3 + k);
            if (corners(t, k) > d) continue;
            const double ex = tri_mesh::steps(x0, point(d, 0), px), ey = tri_mesh::steps(y0, point(d, 1), py);
            const double le = ex * ex + ey * ey;
            const double D = ey * mx - ex * my + le * det;
            if (D > 16 * ep::epsilon * (Kokkos::abs(ey) * pmx + Kokkos::abs(ex) * pmy + le * spread)) continue;
            // Inside, or within rounding of the circle: the repair decides exactly.
            ++failing;
            push(t << 2 | k);
        }
    }
};

// Fills the list; every circumcentre is written.
// Used by: Mesh<2>::update and rebuild (first step of the repair).
template <class ExecSpace>
void screen(const ExecSpace& exec, const ep::Points<typename ExecSpace::memory_space, 2>& point, const ep::BoxN<2>& box,
            const Triangles<typename ExecSpace::memory_space>& mesh, State<typename ExecSpace::memory_space>& state) {
    using Memory = typename ExecSpace::memory_space;
    for (;;) {
        Kokkos::deep_copy(exec, state.used, 0);
        const Screen<Memory> kernel{mesh.corners, mesh.centre, point, state.list, state.used, box.step,
                                    static_cast<int>(box.period[0]), static_cast<int>(box.period[1]),
                                    static_cast<int>(state.list.extent(0))};
        long long failing = 0, bad = 0;
        double area = 0;
        Kokkos::parallel_reduce("tri_repair::screen", Kokkos::RangePolicy<ExecSpace>(exec, 0, mesh.count()), kernel, failing, bad, area);
        state.entries = static_cast<int>(failing);
        state.inverted = bad > 0;
        state.area = area;
        state.area_stale = bad > 0;
        state.touched_count = 0;
        if (failing <= static_cast<long long>(state.list.extent(0))) return;
        detail::reserve(state.list, static_cast<std::size_t>(failing), false);
        detail::reserve(state.next, static_cast<std::size_t>(failing), false);
    }
}

// Whether the edge between (c, a, b) and (d, b, a) flips; `bad` set if one of
// them is not positive.
// Used by: Plan.
template <class P>
KOKKOS_INLINE_FUNCTION bool wants_flip(const P& point, int c, int a, int b, int d, bool& bad) {
    const ep::Offset2 A = ep::offset(point, c, a), B = ep::offset(point, c, b), D = ep::offset(point, c, d);
    const ep::i64 t = ep::cross(A, B), u = ep::cross(ep::minus(B, D), ep::minus(A, D));
    bool flip = true;
    if (t > 0 && u > 0) {
        const ep::Offset2 p[4] = {{0, 0}, A, B, D};
        const int id[4] = {c, a, b, d};
        flip = ep::in_circle(p, id);
    } else {
        bad = true;
    }
    return flip && ep::cross(A, D) > 0 && ep::cross(D, B) > 0;
}

// One triangle's rows, to rewrite it with the lowest corner first.
// Used by: Apply.
struct Row {
    int v[3], apex[3], adjacent[3], face[3];
};

// Used by: repair (first pass of a round).
template <class Memory>
struct Plan {
    Kokkos::View<const int* [6], Kokkos::LayoutRight, Memory> corners, links;
    ep::Points<Memory, 2> point;
    Kokkos::View<unsigned long long*, Memory> lock;
    Kokkos::View<int*, Memory> list;
    unsigned long long token;

    KOKKOS_INLINE_FUNCTION void operator()(int s, long long& wanted, long long& bad) const {
        const int e = list(s), t = e >> 2, k = e & 3;
        const int u = links(t, k);
        const int k1 = (k + 1) % 3, k2 = (k + 2) % 3;
        bool inverted = false;
        const bool flip = wants_flip(point, corners(t, k), corners(t, k1), corners(t, k2), corners(t, 3 + k), inverted);
        bad += inverted;
        if (!flip) {
            list(s) = -1;
            return;
        }
        const int f = links(t, 3 + k);
        int j = 0;
        while (links(u, 3 + j) != f) ++j;
        const int j1 = (j + 1) % 3, j2 = (j + 2) % 3;
        const int outer[4] = {links(t, k1), links(t, k2), links(u, j1), links(u, j2)};
        for (int o = 0; o < 4; ++o)
            if (outer[o] == t || outer[o] == u) {
                list(s) = -1;
                return;
            }
        ++wanted;
        const unsigned long long key = token | detail::hash(static_cast<unsigned>(e) ^ static_cast<unsigned>(token >> 29));
        Kokkos::atomic_max(&lock(t), key);
        Kokkos::atomic_max(&lock(u), key);
        for (int o = 0; o < 4; ++o) Kokkos::atomic_max(&lock(outer[o]), key);
    }
};

// Used by: repair (second pass of a round).
template <class Memory>
struct Apply {
    Kokkos::View<int* [6], Kokkos::LayoutRight, Memory> corners, links;
    Kokkos::View<int*, Memory> owner, neighbour;
    Kokkos::View<unsigned long long*, Memory> lock;
    Kokkos::View<const int*, Memory> list;
    Kokkos::View<int*, Memory> next, touched, flipped;
    Kokkos::View<int[2], Memory> used;
    unsigned long long token;

    KOKKOS_INLINE_FUNCTION void push(int edge) const { next(Kokkos::atomic_fetch_inc(&used(0))) = edge; }

    // Writes row r into slot t, lowest corner first; `first`, `second`: the
    // corners whose edges are listed.
    KOKKOS_INLINE_FUNCTION void store(int t, const Row& r, int first, int second) const {
        const int shift = r.v[0] < r.v[1] && r.v[0] < r.v[2] ? 0 : r.v[1] < r.v[2] ? 1 : 2;
        for (int k = 0; k < 3; ++k) {
            const int from = (k + shift) % 3;
            corners(t, k) = r.v[from];
            corners(t, 3 + k) = r.apex[from];
            links(t, k) = r.adjacent[from];
            links(t, 3 + k) = r.face[from];
        }
        push(t << 2 | (first + 3 - shift) % 3);
        push(t << 2 | (second + 3 - shift) % 3);
    }
    // Outer triangle x: across its face f now lies `across`, with apex `apex`.
    KOKKOS_INLINE_FUNCTION void patch(int x, int f, int across, int apex) const {
        for (int j = 0; j < 3; ++j)
            if (links(x, 3 + j) == f) {
                links(x, j) = across;
                corners(x, 3 + j) = apex;
            }
    }
    KOKKOS_INLINE_FUNCTION void operator()(int s, long long& flips, long long& entries) const {
        const int e = list(s);
        if (e < 0) return;
        const int t = e >> 2, k = e & 3;
        const unsigned long long key = token | detail::hash(static_cast<unsigned>(e) ^ static_cast<unsigned>(token >> 29));
        // A flip writes rows only with its lock, so while the locks read so far are ours the planned rows are current.
        bool won = Kokkos::atomic_compare_exchange(&lock(t), key, token) == key;
        int u = -1, j = 0;
        if (won) {
            u = links(t, k);
            won = lock(u) == key;
        }
        if (won) {
            const int f = links(t, 3 + k);
            while (links(u, 3 + j) != f) ++j;
            won = lock(links(t, (k + 1) % 3)) == key && lock(links(t, (k + 2) % 3)) == key &&
                  lock(links(u, (j + 1) % 3)) == key && lock(links(u, (j + 2) % 3)) == key;
        }
        if (!won) {
            ++entries;
            push(e);
            return;
        }
        const int k1 = (k + 1) % 3, k2 = (k + 2) % 3, j1 = (j + 1) % 3, j2 = (j + 2) % 3;
        const int c = corners(t, k), a = corners(t, k1), b = corners(t, k2), d = corners(t, 3 + k), F = links(t, 3 + k);
        const int outer_bc = links(t, k1), face_bc = links(t, 3 + k1), apex_bc = corners(t, 3 + k1);
        const int outer_ca = links(t, k2), face_ca = links(t, 3 + k2), apex_ca = corners(t, 3 + k2);
        const int outer_ad = links(u, j1), face_ad = links(u, 3 + j1), apex_ad = corners(u, 3 + j1);
        const int outer_db = links(u, j2), face_db = links(u, 3 + j2), apex_db = corners(u, 3 + j2);
        store(t, Row{{c, a, d}, {apex_ad, b, apex_ca}, {outer_ad, u, outer_ca}, {face_ad, F, face_ca}}, 0, 2);
        store(u, Row{{c, d, b}, {apex_db, apex_bc, a}, {outer_db, outer_bc, t}, {face_db, face_bc, F}}, 0, 1);
        patch(outer_ad, face_ad, t, c);
        patch(outer_bc, face_bc, u, d);
        patch(outer_ca, face_ca, t, d);
        patch(outer_db, face_db, u, c);
        owner(F) = c < d ? c : d;
        neighbour(F) = c < d ? d : c;
        ++flips;
        entries += 4;
        const int at = Kokkos::atomic_fetch_add(&used(1), 2);
        touched(at) = t;
        touched(at + 1) = u;
        flipped(at / 2) = F;
    }
};

// Inverted or flat triangles left.
// Used by: repair.
template <class ExecSpace>
int count_inverted(const ExecSpace& exec, const ep::Points<typename ExecSpace::memory_space, 2>& point,
                   const Triangles<typename ExecSpace::memory_space>& mesh) {
    const auto corners = mesh.corners;
    int inverted = 0;
    Kokkos::parallel_reduce(
        "tri_repair::inverted", Kokkos::RangePolicy<ExecSpace>(exec, 0, mesh.count()),
        KOKKOS_LAMBDA(int t, int& count) {
            const int o = corners(t, 0);
            count += ep::cross(ep::offset(point, o, corners(t, 1)), ep::offset(point, o, corners(t, 2))) <= 0;
        },
        inverted);
    return inverted;
}

// Twice the sum of the signed areas, in grid steps^2: a step of order the box can leave a covering of the torus several times.
// Flips of positive triangles keep it, so this runs only after an inversion.
// Used by: Mesh<2>::fills_box.
template <class ExecSpace>
double total_area(const ExecSpace& exec, const ep::Points<typename ExecSpace::memory_space, 2>& point, const ep::BoxN<2>& box,
                  const Triangles<typename ExecSpace::memory_space>& mesh) {
    const auto corners = mesh.corners;
    const int px = static_cast<int>(box.period[0]), py = static_cast<int>(box.period[1]);
    double sum = 0;
    Kokkos::parallel_reduce(
        "tri_repair::total_area", Kokkos::RangePolicy<ExecSpace>(exec, 0, mesh.count()),
        KOKKOS_LAMBDA(int t, double& area) {
            const int v0 = corners(t, 0), v1 = corners(t, 1), v2 = corners(t, 2);
            const int x0 = point(v0, 0), y0 = point(v0, 1);
            area += tri_mesh::Circle(tri_mesh::steps(x0, point(v1, 0), px), tri_mesh::steps(y0, point(v1, 1), py),
                                     tri_mesh::steps(x0, point(v2, 0), px), tri_mesh::steps(y0, point(v2, 1), py)).det;
        },
        sum);
    return sum;
}

// Works through the list. stuck: inverted or flat triangles left.
// Used by: Mesh<2>::update and rebuild.
template <class ExecSpace>
Stats repair(const ExecSpace& exec, const ep::Points<typename ExecSpace::memory_space, 2>& point,
             const Triangles<typename ExecSpace::memory_space>& mesh, State<typename ExecSpace::memory_space>& state) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    Stats stats;
    Kokkos::deep_copy(exec, Kokkos::subview(state.used, 1), state.touched_count);
    while (state.entries > 0) {
        ++state.round;
        const unsigned long long token = state.round << 32;
        long long wanted = 0, bad = 0;
        Kokkos::parallel_reduce("tri_repair::plan", Range(exec, 0, state.entries),
                                Plan<Memory>{mesh.corners, mesh.links, point, state.lock, state.list, token}, wanted, bad);
        state.inverted = state.inverted || bad > 0;
        ++stats.rounds;
        if (wanted == 0) break;
        detail::reserve(state.next, static_cast<std::size_t>(4 * wanted), false);
        detail::reserve(state.touched, static_cast<std::size_t>(state.touched_count + 2 * wanted), true);
        detail::reserve(state.flipped, state.touched.extent(0), true);
        Kokkos::deep_copy(exec, Kokkos::subview(state.used, 0), 0);
        long long flips = 0, entries = 0;
        Kokkos::parallel_reduce("tri_repair::apply", Range(exec, 0, state.entries),
                                Apply<Memory>{mesh.corners, mesh.links, mesh.owner, mesh.neighbour, state.lock, state.list,
                                              state.next, state.touched, state.flipped, state.used, token},
                                flips, entries);
        stats.flips += flips;
        state.touched_count += static_cast<int>(2 * flips);
        state.entries = static_cast<int>(entries);
        std::swap(state.list, state.next);
    }
    state.entries = 0;
    if (state.inverted) {
        stats.stuck = count_inverted(exec, point, mesh);
        state.inverted = stats.stuck > 0;
    }
    return stats;
}

// The circumcentres of the triangles that flipped.
// Used by: Mesh<2>::geometry.
template <class ExecSpace>
void centres(const ExecSpace& exec, const ep::Points<typename ExecSpace::memory_space, 2>& point, const ep::BoxN<2>& box,
             const Triangles<typename ExecSpace::memory_space>& mesh, const State<typename ExecSpace::memory_space>& state) {
    const auto corners = mesh.corners;
    const auto centre = mesh.centre;
    const auto touched = state.touched;
    const int px = static_cast<int>(box.period[0]), py = static_cast<int>(box.period[1]);
    const double step = box.step;
    Kokkos::parallel_for(
        "tri_repair::centres", Kokkos::RangePolicy<ExecSpace>(exec, 0, state.touched_count), KOKKOS_LAMBDA(int s) {
            const int t = touched(s), v0 = corners(t, 0), v1 = corners(t, 1), v2 = corners(t, 2);
            const int x0 = point(v0, 0), y0 = point(v0, 1);
            const tri_mesh::Circle circle(tri_mesh::steps(x0, point(v1, 0), px), tri_mesh::steps(y0, point(v1, 1), py),
                                          tri_mesh::steps(x0, point(v2, 0), px), tri_mesh::steps(y0, point(v2, 1), py));
            tet_mesh::store(centre, t, circle.centre(x0, y0, step));
        });
}

}  // namespace tri_repair
