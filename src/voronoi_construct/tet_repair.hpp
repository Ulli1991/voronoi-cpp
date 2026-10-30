#pragma once

#include "exact_predicates.hpp"
#include "tet_mesh.hpp"

// Delaunay repair by bistellar flips on the device, after the particles of a
// persistent tetrahedralization (tet_mesh::Mesh, positively oriented) moved.
//
// Each round
//   1. finds the shared triangles that fail the local Delaunay test (exact,
//      ties perturbed) among the tetrahedra marked dirty,
//   2. plans a 2->3 flip (convex pair) or a 3->2 flip (an edge of the
//      triangle with three tetrahedra around it) for each,
//   3. lets every plan vote, with its index, on the tetrahedra it would
//      rewrite and their outer neighbours (largest index wins), and
//   4. applies the plans that won all their votes; their new tetrahedra
//      are dirty for the next round, losing plans retry.
// Triangles no flip repairs, flips that would not close their cavity, and
// tetrahedra seen inverted are listed as stuck, for the host fallback
// (tet_flips::finish).
namespace tet_repair {

namespace ep = exact_predicates;
using tet_mesh::Vec3;

namespace detail {
struct Plan;
}

template <class MemorySpace>
struct State {
    ep::Points<MemorySpace> point;  // snapped positions
    Kokkos::View<char*, MemorySpace> alive;
    Kokkos::View<char*, MemorySpace> touched;  // rewritten by a flip since last cleared
    Kokkos::View<int*, MemorySpace> dirty;   // 1 while listed in `listed`
    Kokkos::View<int*, MemorySpace> listed;  // the dirty tetrahedra
    // Votes: round in the high 32 bits, plan index + 1 below, so a vote of
    // an earlier round always loses and locks never need resetting.
    Kokkos::View<unsigned long long*, MemorySpace> lock;
    Kokkos::View<int*, MemorySpace> free_slots;  // popped from the top
    Kokkos::View<int* [2], MemorySpace> work;   // failing triangles of a round
    Kokkos::View<detail::Plan*, MemorySpace> plans;
    Kokkos::View<int*, MemorySpace> freed;
    // What the last repair left: (t, k) a triangle no flip repairs, (t, -1)
    // an inverted tetrahedron, (t, -2) a flip that would not close its cavity.
    Kokkos::View<int* [2], MemorySpace> stuck;
    // Device counters: 0 work, 1 stuck, 2 freed, 3 listed, 4 pops, 5 growth, 6 2->3, 7 3->2,
    // and their pinned host copy, read once or twice per round.
    Kokkos::View<int*, MemorySpace> counters;
    Kokkos::View<int*, Kokkos::SharedHostPinnedSpace> host_counters;
    int count = 0;  // tetrahedra slots in use
    int free_count = 0;
    unsigned long long round = 0;
};

struct Stats {
    long long rounds = 0, checked = 0, flips23 = 0, flips32 = 0;
    int stuck = 0;  // entries listed in State::stuck
};

// Whether new tetrahedra `fresh` close the cavity of the old ones `old` of
// mesh m exactly: six outer faces, each face of a new tetrahedron matched
// once (outside or by another new one), each outer face used once, and no
// new tetrahedron duplicating the outer one it attaches to. Near-degenerate
// configurations can break this.
template <class M>
KOKKOS_INLINE_FUNCTION bool closes(const M& m, const int* old, int olds, const int (*fresh)[4], int freshes) {
    int outer_key[8][3], outer_tet[8], outers = 0;
    for (int r = 0; r < olds; ++r) {
        const int o = old[r];
        const int v[4] = {m.vertex(o, 0), m.vertex(o, 1), m.vertex(o, 2), m.vertex(o, 3)};
        for (int j = 0; j < 4; ++j) {
            const int w = m.adjacent(o, j);
            bool inner = false;
            for (int q = 0; q < olds; ++q) inner |= w == old[q];
            if (inner) continue;
            if (outers == 8) return false;
            tet_mesh::sorted_face(v, j, outer_key[outers]);
            outer_tet[outers++] = w;
        }
    }
    if (outers != 6) return false;
    int used[6] = {0, 0, 0, 0, 0, 0};
    for (int r = 0; r < freshes; ++r)
        for (int c = 0; c < 4; ++c) {
            int key[3];
            tet_mesh::sorted_face(fresh[r], c, key);
            int matches = 0;
            for (int q = 0; q < 6; ++q)
                if (key[0] == outer_key[q][0] && key[1] == outer_key[q][1] && key[2] == outer_key[q][2]) {
                    ++matches;
                    ++used[q];
                    if (outer_tet[q] >= 0 && tet_mesh::position(m, outer_tet[q], fresh[r][c]) >= 0) return false;
                }
            for (int s2 = 0; s2 < freshes; ++s2) {
                if (s2 == r) continue;
                for (int c2 = 0; c2 < 4; ++c2) {
                    int other[3];
                    tet_mesh::sorted_face(fresh[s2], c2, other);
                    matches += key[0] == other[0] && key[1] == other[1] && key[2] == other[2];
                }
            }
            if (matches != 1) return false;
        }
    for (int q = 0; q < 6; ++q)
        if (used[q] != 1) return false;
    return true;
}

namespace detail {

// A planned flip: the old tetrahedra and the new ones (vertex lists).
struct Plan {
    int kind = 0;  // 0 none, 2 = 2->3, 3 = 3->2, -1 stuck
    int old[3] = {-1, -1, -1};
    int fresh[3][4];
    int olds = 0, freshes = 0;
};

// Plan the repair of triangle t|u (t's face opposite corner k): the first
// valid 2->3 or 3->2 flip, or stuck. Unlike the host, it does not try
// another flip when this one would not close its cavity.
template <class Mesh, class Point>
KOKKOS_INLINE_FUNCTION Plan plan(const Mesh& m, const Point& point, int t, int k) {
    Plan p;
    const int u = m.adjacent(t, k);
    const int T[4] = {m.vertex(t, 0), m.vertex(t, 1), m.vertex(t, 2), m.vertex(t, 3)};
    const int far = tet_mesh::far_vertex(m, t, u);
    if (far < 0 || !ep::in_sphere(point, T, far)) return p;
    const int a = T[k];
    int f[3];
    tet_mesh::sorted_face(T, k, f);
    // 2->3: segment a-far crosses the shared face.
    int positive = 0, negative = 0;
    for (int e = 0; e < 3; ++e) {
        const int n[4] = {a, far, f[e], f[(e + 1) % 3]};
        const int o = ep::orient(point, n);
        positive += o > 0;
        negative += o < 0;
        for (int c = 0; c < 4; ++c) p.fresh[e][c] = n[c];
    }
    if (positive == 3 || negative == 3) {
        p.kind = 2;
        p.old[0] = t;
        p.old[1] = u;
        p.olds = 2;
        p.freshes = 3;
        return p;
    }
    // 3->2: an edge p-q of the face with exactly three tetrahedra around it.
    for (int e = 0; e < 3; ++e) {
        const int pv = f[e], qv = f[(e + 1) % 3];
        int ring[3] = {t, -1, -1}, size = 1;
        bool closed = false;
        for (int current = t, previous = -1; size <= 3;) {
            int next = -1;
            for (int j = 0; j < 4 && next < 0; ++j) {
                const int v = m.vertex(current, j);
                if (v == pv || v == qv) continue;
                const int w = m.adjacent(current, j);
                if (w >= 0 && w != previous) next = w;
            }
            if (next < 0) break;
            if (next == t) {
                closed = true;
                break;
            }
            if (size == 3) break;
            ring[size++] = next;
            previous = current;
            current = next;
        }
        if (!closed || size != 3 || (ring[1] != u && ring[2] != u)) continue;
        int others[3], count = 0;
        for (int r = 0; r < 3; ++r)
            for (int j = 0; j < 4; ++j) {
                const int v = m.vertex(ring[r], j);
                bool seen = v == pv || v == qv;
                for (int c = 0; c < count; ++c) seen |= others[c] == v;
                if (!seen && count < 3) others[count++] = v;
            }
        if (count != 3) continue;
        const int n1[4] = {pv, others[0], others[1], others[2]}, n2[4] = {qv, others[0], others[1], others[2]};
        if (ep::orient(point, n1) == 0 || ep::orient(point, n2) == 0) continue;
        const int side_p[4] = {others[0], others[1], others[2], pv}, side_q[4] = {others[0], others[1], others[2], qv};
        if (ep::orient(point, side_p) * ep::orient(point, side_q) >= 0) continue;
        p.kind = 3;
        for (int r = 0; r < 3; ++r) p.old[r] = ring[r];
        p.olds = 3;
        p.freshes = 2;
        for (int c = 0; c < 4; ++c) {
            p.fresh[0][c] = n1[c];
            p.fresh[1][c] = n2[c];
        }
        return p;
    }
    p.kind = -1;
    return p;
}

template <class ExecSpace, class X>
void snap(const ExecSpace& exec, const X& x, const ep::Points<typename ExecSpace::memory_space>& point) {
    Kokkos::parallel_for(
        "tet_repair::snap", Kokkos::RangePolicy<ExecSpace>(exec, 0, x.extent(0)), KOKKOS_LAMBDA(int i) {
            for (int a = 0; a < 3; ++a) point(i, a) = ep::snap(x(i, a));
        });
}

// List t as dirty unless it already is.
template <class State>
KOKKOS_INLINE_FUNCTION void mark(const State& s, int t) {
    if (Kokkos::atomic_exchange(&s.dirty(t), 1) == 0) s.listed(Kokkos::atomic_fetch_add(&s.counters(3), 1)) = t;
}

}  // namespace detail

// Snapped positions and positive orientation of every tetrahedron; allocates
// the bookkeeping for up to `capacity` tetrahedra (the mesh views are grown to it).
template <class ExecSpace, class X>
State<typename ExecSpace::memory_space> prepare(const ExecSpace& exec,
                                                const X& x,
                                                tet_mesh::Mesh<typename ExecSpace::memory_space>& m, int capacity) {
    using Memory = typename ExecSpace::memory_space;
    State<Memory> s;
    const int n = static_cast<int>(x.extent(0));
    s.count = static_cast<int>(m.vertex.extent(0));
    Kokkos::resize(m.vertex, capacity);
    Kokkos::resize(m.adjacent, capacity);
    Kokkos::resize(m.centre, capacity);
    Kokkos::resize(m.edges, capacity);
    s.point = ep::Points<Memory>("tet_repair_point", n);
    s.alive = Kokkos::View<char*, Memory>("tet_repair_alive", capacity);
    s.touched = Kokkos::View<char*, Memory>("tet_repair_touched", capacity);
    s.dirty = Kokkos::View<int*, Memory>("tet_repair_dirty", capacity);
    s.listed = Kokkos::View<int*, Memory>("tet_repair_listed", capacity);
    s.lock = Kokkos::View<unsigned long long*, Memory>("tet_repair_lock", capacity);
    s.free_slots = Kokkos::View<int*, Memory>("tet_repair_free", capacity);
    const int work_capacity = capacity / 4 + 1024;
    s.work = Kokkos::View<int* [2], Memory>("tet_repair_work", work_capacity);
    s.plans = Kokkos::View<detail::Plan*, Memory>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("tet_repair_plans")), work_capacity);
    s.freed = Kokkos::View<int*, Memory>("tet_repair_freed", work_capacity);
    s.stuck = Kokkos::View<int* [2], Memory>("tet_repair_stuck", work_capacity);
    s.counters = Kokkos::View<int*, Memory>("tet_repair_counters", 8);
    s.host_counters = Kokkos::View<int*, Kokkos::SharedHostPinnedSpace>("tet_repair_host_counters", 8);
    detail::snap(exec, x, s.point);
    const auto point = s.point;
    const auto alive = s.alive;
    const auto vertex = m.vertex;
    const auto adjacent = m.adjacent;
    const auto edges = m.edges;
    Kokkos::parallel_for(
        "tet_repair::orient", Kokkos::RangePolicy<ExecSpace>(exec, 0, s.count), KOKKOS_LAMBDA(int t) {
            alive(t) = 1;
            const int v[4] = {vertex(t, 0), vertex(t, 1), vertex(t, 2), vertex(t, 3)};
            if (ep::orient(point, v) < 0) {
                Kokkos::kokkos_swap(vertex(t, 2), vertex(t, 3));
                Kokkos::kokkos_swap(adjacent(t, 2), adjacent(t, 3));
                // Edges (0,2)<->(0,3) and (1,2)<->(1,3) trade places.
                const int b = edges(t);
                edges(t) = (b & ~0x1e) | (b & 0x2) << 1 | (b & 0x4) >> 1 | (b & 0x8) << 1 | (b & 0x10) >> 1;
            }
        });
    return s;
}

// New positions x: snaps them and lists the tetrahedra that may need repair,
// screened in double precision with their circumcentres (m.centre, computed
// for x): those with a neighbour's far vertex inside or near the
// circumsphere, and the badly shaped ones (possibly flat or inverted). The
// margin covers the snapping of positions to the exact grid; what it lists
// is decided exactly by repair.
template <class ExecSpace, class X>
void move_screened(const ExecSpace& exec, const X& x, double box,
                   const tet_mesh::Mesh<typename ExecSpace::memory_space>& m,
                   State<typename ExecSpace::memory_space>& s) {
    detail::snap(exec, x, s.point);
    Kokkos::deep_copy(exec, s.counters, 0);
    const auto& st = s;
    const auto vertex = m.vertex;
    const auto adjacent = m.adjacent;
    const auto centre = m.centre;
    const double snap = box / ep::scale * 1.7320508075688772;  // largest move by snapping
    Kokkos::parallel_for(
        "tet_repair::screen", Kokkos::RangePolicy<ExecSpace>(exec, 0, s.count), KOKKOS_LAMBDA(int t) {
            st.dirty(t) = 0;
            if (!st.alive(t)) return;
            auto from = [&](double cx, double cy, double cz, int v) {
                return Vec3<double>{tet_mesh::wrap(x(v, 0) - cx, box), tet_mesh::wrap(x(v, 1) - cy, box),
                                    tet_mesh::wrap(x(v, 2) - cz, box)};
            };
            const int v0 = vertex(t, 0);
            const Vec3<double> d1 = from(x(v0, 0), x(v0, 1), x(v0, 2), vertex(t, 1)),
                               d2 = from(x(v0, 0), x(v0, 1), x(v0, 2), vertex(t, 2)),
                               d3 = from(x(v0, 0), x(v0, 1), x(v0, 2), vertex(t, 3));
            const double l1 = dot(d1, d1), l2 = dot(d2, d2), l3 = dot(d3, d3);
            const double volume6 = dot(d1, cross(d2, d3));
            // Well shaped and clearly positive, else decide exactly.
            bool suspect = !(volume6 > 1e-3 * Kokkos::sqrt(l1 * l2 * l3));
            const double r2 = centre(t, 3);
            const double margin = 1e-6 * r2 + 8 * snap * (Kokkos::sqrt(r2) + snap);
            for (int k = 0; k < 4 && !suspect; ++k) {
                const int u = adjacent(t, k);
                if (u < 0) continue;
                const int far = tet_mesh::far_vertex(m, t, u);
                if (far < 0) continue;
                const Vec3<double> e = from(centre(t, 0), centre(t, 1), centre(t, 2), far);
                suspect = dot(e, e) < r2 + margin;
            }
            if (suspect) detail::mark(st, t);
        });
}

// Flip until no dirty tetrahedron fails or no plan can proceed; what is left
// is listed in s.stuck (stats.stuck entries).
template <class ExecSpace>
Stats repair(const ExecSpace& exec, const tet_mesh::Mesh<typename ExecSpace::memory_space>& m,
             State<typename ExecSpace::memory_space>& s) {
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const int capacity = static_cast<int>(s.alive.extent(0));
    Stats stats;
    const auto work = s.work;
    const auto plans = s.plans;
    const auto counters = s.counters;
    const auto freed = s.freed;
    const auto listed = s.listed;
    const auto& st = s;
    const int work_capacity = static_cast<int>(work.extent(0));
    const auto stuck = s.stuck;
    const auto vertex = m.vertex;
    const auto adjacent = m.adjacent;
    const auto point = s.point;
    const auto alive = s.alive;
    const auto dirty = s.dirty;
    const auto lock = s.lock;
    const auto free_slots = s.free_slots;
    const auto edges = m.edges;
    const auto host = s.host_counters;
    auto read = [&] {  // one asynchronous copy into pinned memory, then wait
        Kokkos::deep_copy(exec, host, counters);
        exec.fence();
    };
    auto clear = [&](int k) { Kokkos::deep_copy(exec, Kokkos::subview(counters, k), 0); };
    // The stuck list fills through counters(1) for the whole repair.
    clear(1);
    read();
    int dirty_count = host(3);
    long long flips_at_last_retry = -1;

    for (int round = 0; round < 1000; ++round) {
        // 1. Failing triangles among the listed dirty tetrahedra (each triangle
        // once: from the dirty side, or from the lower index when both are).
        clear(0);
        clear(2);
        Kokkos::parallel_for(
            "tet_repair::find", Range(exec, 0, dirty_count), KOKKOS_LAMBDA(int i) {
                const int t = listed(i);
                if (!alive(t)) return;
                const int v[4] = {vertex(t, 0), vertex(t, 1), vertex(t, 2), vertex(t, 3)};
                const ep::Offset o[4] = {{0, 0, 0}, ep::offset(point, v[0], v[1]), ep::offset(point, v[0], v[2]),
                                         ep::offset(point, v[0], v[3])};
                // Turned inside out: listed for the host, but flips usually untangle it.
                if (ep::orient(o[1], o[2], o[3]) <= 0) {
                    const int at = Kokkos::atomic_fetch_add(&counters(1), 1);
                    if (at < static_cast<int>(stuck.extent(0))) {
                        stuck(at, 0) = t;
                        stuck(at, 1) = -1;
                    }
                }
                for (int k = 0; k < 4; ++k) {
                    const int u = adjacent(t, k);
                    if (u < 0 || (dirty(u) && u < t)) continue;
                    const int far = tet_mesh::far_vertex(m, t, u);
                    if (far < 0 || !ep::in_sphere(point, v, o, far)) continue;
                    const int at = Kokkos::atomic_fetch_add(&counters(0), 1);
                    if (at < work_capacity) {
                        work(at, 0) = t;
                        work(at, 1) = k;
                    }
                }
            });
        // Unlist them; the apply step lists the next round's.
        Kokkos::parallel_for(
            "tet_repair::unlist", Range(exec, 0, dirty_count), KOKKOS_LAMBDA(int i) { dirty(listed(i)) = 0; });
        Kokkos::deep_copy(exec, Kokkos::subview(counters, std::make_pair(3, 8)), 0);
        read();
        const int items = Kokkos::min(host(0), work_capacity);
        stats.checked += items;
        if (items == 0) {
            // A triangle stuck earlier may have become flippable since: look
            // at the stuck ones again while that still makes progress.
            const long long flips = stats.flips23 + stats.flips32;
            if (host(1) == 0 || flips == flips_at_last_retry) break;
            flips_at_last_retry = flips;
            const int listed_stuck = Kokkos::min(host(1), static_cast<int>(stuck.extent(0)));
            Kokkos::parallel_for(
                "tet_repair::retry_stuck", Range(exec, 0, listed_stuck), KOKKOS_LAMBDA(int i) {
                    const int t = stuck(i, 0);
                    if (t >= 0 && alive(t)) detail::mark(st, t);
                });
            clear(1);
            read();
            dirty_count = host(3);
            continue;
        }
        ++stats.rounds;
        const unsigned long long token = ++s.round << 32;

        // 2-4. Plan, vote, apply; losers stay dirty for the next round.
        Kokkos::parallel_for(
            "tet_repair::plan", Range(exec, 0, items), KOKKOS_LAMBDA(int i) {
                const int t = work(i, 0), k = work(i, 1);
                detail::Plan p;
                const int u = adjacent(t, k);
                if (alive(t) && u >= 0 && alive(u)) p = detail::plan(m, point, t, k);
                if (p.kind > 0 && !closes(m, p.old, p.olds, p.fresh, p.freshes)) {
                    p.kind = 0;  // never votes, so it cannot block other flips
                    const int at = Kokkos::atomic_fetch_add(&counters(1), 1);
                    if (at < static_cast<int>(stuck.extent(0))) {
                        stuck(at, 0) = t;
                        stuck(at, 1) = -2;  // flip rejected: does not close its cavity
                    }
                }
                plans(i) = p;
                if (p.kind == -1) {
                    const int at = Kokkos::atomic_fetch_add(&counters(1), 1);
                    if (at < static_cast<int>(stuck.extent(0))) {
                        stuck(at, 0) = t;
                        stuck(at, 1) = k;
                    }
                }
                if (p.kind <= 0) return;
                for (int r = 0; r < p.olds; ++r) {
                    const int o = p.old[r];
                    Kokkos::atomic_max(&lock(o), token + i + 1);
                    for (int j = 0; j < 4; ++j)
                        if (adjacent(o, j) >= 0) Kokkos::atomic_max(&lock(adjacent(o, j)), token + i + 1);
                }
            });

        const int free_top = s.free_count;
        const int count = s.count;
        Kokkos::parallel_for(
            "tet_repair::apply", Range(exec, 0, items), KOKKOS_LAMBDA(int i) {
                const detail::Plan p = plans(i);
                if (p.kind <= 0) return;
                bool won = true;
                for (int r = 0; r < p.olds && won; ++r) {
                    const int o = p.old[r];
                    won = lock(o) == token + i + 1;
                    for (int j = 0; j < 4 && won; ++j)
                        if (adjacent(o, j) >= 0) won = lock(adjacent(o, j)) == token + i + 1;
                }
                if (!won) {
                    detail::mark(st, p.old[0]);  // retry next round
                    return;
                }
                // Outer faces of the old tetrahedra: sorted triple, outer tet, its index back.
                int outer_key[8][3], outer_tet[8], outer_back[8], outers = 0;
                for (int r = 0; r < p.olds; ++r) {
                    const int o = p.old[r];
                    const int v[4] = {vertex(o, 0), vertex(o, 1), vertex(o, 2), vertex(o, 3)};
                    for (int j = 0; j < 4; ++j) {
                        const int w = adjacent(o, j);
                        bool inner = false;
                        for (int q = 0; q < p.olds; ++q) inner |= w == p.old[q];
                        if (inner || outers == 8) continue;
                        tet_mesh::sorted_face(v, j, outer_key[outers]);
                        outer_tet[outers] = w;
                        outer_back[outers] = -1;
                        if (w >= 0)
                            for (int b = 0; b < 4; ++b)
                                if (adjacent(w, b) == o) outer_back[outers] = b;
                        ++outers;
                    }
                }
                Kokkos::atomic_inc(&counters(p.kind == 2 ? 6 : 7));
                // Edges the old tetrahedra represented; they get new representatives below.
                int kept_edge[19][2], kept_edges = 0;
                for (int r = 0; r < p.olds; ++r) {
                    const int o = p.old[r];
                    for (int bits = edges(o); bits; bits &= bits - 1) {
                        int ka, kb;
                        tet_mesh::edge_ends(Kokkos::Experimental::countr_zero_builtin(static_cast<unsigned>(bits)), ka,
                                            kb);
                        kept_edge[kept_edges][0] = vertex(o, ka);
                        kept_edge[kept_edges][1] = vertex(o, kb);
                        ++kept_edges;
                    }
                    edges(o) = 0;
                }
                if (p.kind == 2) {  // the new edge a-far
                    kept_edge[kept_edges][0] = p.fresh[0][0];
                    kept_edge[kept_edges][1] = p.fresh[0][1];
                    ++kept_edges;
                }
                // Slots: reuse the old ones, then pop free slots or grow.
                int slot[3];
                for (int r = 0; r < p.freshes; ++r) {
                    if (r < p.olds) {
                        slot[r] = p.old[r];
                    } else {
                        const int popped = Kokkos::atomic_fetch_add(&counters(4), 1);
                        slot[r] = popped < free_top ? free_slots(free_top - 1 - popped)
                                                    : count + Kokkos::atomic_fetch_add(&counters(5), 1);
                    }
                }
                for (int r = p.freshes; r < p.olds; ++r) {
                    alive(p.old[r]) = 0;
                    freed(Kokkos::atomic_fetch_add(&counters(2), 1)) = p.old[r];
                }
                for (int r = 0; r < p.freshes; ++r) {
                    int v[4] = {p.fresh[r][0], p.fresh[r][1], p.fresh[r][2], p.fresh[r][3]};
                    if (ep::orient(point, v) < 0) Kokkos::kokkos_swap(v[2], v[3]);
                    for (int c = 0; c < 4; ++c) {
                        vertex(slot[r], c) = v[c];
                        adjacent(slot[r], c) = -1;
                    }
                    alive(slot[r]) = 1;
                }
                // Connect faces: to the outside through the old outer faces, else to each other.
                for (int r = 0; r < p.freshes; ++r) {
                    const int v[4] = {vertex(slot[r], 0), vertex(slot[r], 1), vertex(slot[r], 2), vertex(slot[r], 3)};
                    for (int c = 0; c < 4; ++c) {
                        int key[3];
                        tet_mesh::sorted_face(v, c, key);
                        for (int q = 0; q < outers; ++q)
                            if (key[0] == outer_key[q][0] && key[1] == outer_key[q][1] && key[2] == outer_key[q][2]) {
                                adjacent(slot[r], c) = outer_tet[q];
                                if (outer_tet[q] >= 0 && outer_back[q] >= 0) adjacent(outer_tet[q], outer_back[q]) = slot[r];
                            }
                        for (int s2 = 0; s2 < p.freshes; ++s2) {
                            if (s2 == r) continue;
                            const int w[4] = {p.fresh[s2][0], p.fresh[s2][1], p.fresh[s2][2], p.fresh[s2][3]};
                            for (int c2 = 0; c2 < 4; ++c2) {
                                int other[3];
                                tet_mesh::sorted_face(w, c2, other);
                                if (key[0] == other[0] && key[1] == other[1] && key[2] == other[2]) adjacent(slot[r], c) = slot[s2];
                            }
                        }
                    }
                }
                // Each kept edge (except the one a 3->2 flip removes) to the first new tetrahedron holding it.
                for (int q = 0; q < kept_edges; ++q) {
                    const int a = kept_edge[q][0], b = kept_edge[q][1];
                    for (int r = 0; r < p.freshes; ++r) {
                        int pa = -1, pb = -1;
                        for (int c = 0; c < 4; ++c) {
                            if (vertex(slot[r], c) == a) pa = c;
                            if (vertex(slot[r], c) == b) pb = c;
                        }
                        if (pa >= 0 && pb >= 0) {
                            edges(slot[r]) |= 1 << tet_mesh::edge_index(pa, pb);
                            break;
                        }
                    }
                }
                for (int r = 0; r < p.olds; ++r) st.touched(p.old[r]) = 1;
                for (int r = 0; r < p.freshes; ++r) {
                    st.touched(slot[r]) = 1;
                    detail::mark(st, slot[r]);
                }
            });
        read();
        dirty_count = host(3);
        // Pops used, growth, and the slots freed this round onto the free list.
        const int popped = Kokkos::min(host(4), free_top);
        s.free_count = free_top - popped;
        s.count = count + host(5);
        if (s.count > capacity) Kokkos::abort("tet_repair: capacity exceeded");
        const int freed_count = host(2);
        const int base = s.free_count;
        Kokkos::parallel_for(
            "tet_repair::release", Range(exec, 0, freed_count),
            KOKKOS_LAMBDA(int i) { free_slots(base + i) = freed(i); });
        s.free_count += freed_count;
        stats.flips23 += host(6);
        stats.flips32 += host(7);
    }
    stats.stuck = Kokkos::min(host(1), static_cast<int>(stuck.extent(0)));
    return stats;
}

}  // namespace tet_repair
