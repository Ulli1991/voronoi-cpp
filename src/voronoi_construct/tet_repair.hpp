#pragma once

#include "exact_predicates.hpp"
#include "tet_mesh.hpp"

// Stage 4a (repair, device): called by voronoi::Mesh<3>::update (prepare: rebuild) after snap; feeds geometry (tet_mesh).
// Delaunay repair by bistellar flips (2->3, 3->2) after particles moved: plans vote on the tetrahedra they rewrite and
// the winners apply. Entries no flip repairs are listed as stuck for tet_flips::finish.
namespace tet_repair {

namespace ep = exact_predicates;
using tet_mesh::Vec3;

namespace detail {
struct Packed;
}

// Entries of State::counters: failing triangles this round, stuck entries, slots freed (in move_screened: slivers),
// dirty tetrahedra listed, free slots popped, new slots, flips of each kind, tetrahedra created this step.
namespace counter {
constexpr int work = 0, stuck = 1, freed = 2, listed = 3, pops = 4, growth = 5, flips23 = 6, flips32 = 7, created = 8,
              count = 9;
}

// A stuck entry (t, k) of State::stuck: the triangle of t opposite corner
// k >= 0 that no flip repairs, or one of these.
constexpr int inverted = -1;  // t is flat or inside out
constexpr int rejected = -2;  // the planned flip would not close its cavity

// Device bookkeeping of the repair, persistent between updates (see the member comments).
// Used by: prepare (creates), move_screened, repair, tet_flips::finish; Mesh::update_cells (touched_list).
template <class MemorySpace>
struct State {
    ep::Points<MemorySpace> point;  // snapped positions
    Kokkos::View<char*, MemorySpace> alive;
    Kokkos::View<char*, MemorySpace> touched;  // rewritten by a flip since last cleared
    // The tetrahedra flips created since the last move: touched_entries of them, -1 if more than fit.
    Kokkos::View<int*, MemorySpace> touched_list;
    Kokkos::View<int*, MemorySpace> dirty;   // 1 while listed in `listed`
    Kokkos::View<int*, MemorySpace> listed;  // the dirty tetrahedra
    // Votes: round in the high 32 bits, a bijective hash of the plan index below, so earlier rounds always lose.
    Kokkos::View<unsigned long long*, MemorySpace> lock;
    Kokkos::View<int*, MemorySpace> free_slots;  // popped from the top
    Kokkos::View<int* [2], MemorySpace> work;    // failing triangles of a round
    Kokkos::View<detail::Packed*, MemorySpace> plans;
    Kokkos::View<int*, MemorySpace> freed;
    // What the last repair left (see inverted, rejected).
    Kokkos::View<int* [2], MemorySpace> stuck;
    // Device counters (see counter), and their pinned host copy, read once or twice per round.
    Kokkos::View<int*, MemorySpace> counters;
    Kokkos::View<int*, Kokkos::SharedHostPinnedSpace> host_counters;
    // What the small rounds left (detail::small), and its pinned host copy; allocated by the first repair.
    Kokkos::View<long long*, MemorySpace> small;
    Kokkos::View<long long*, Kokkos::SharedHostPinnedSpace> host_small;
    int count = 0;  // tetrahedra slots in use
    int free_count = 0;
    int touched_entries = 0;
    unsigned long long round = 0;
    bool screen_counts = false;  // host_counters hold move_screened's counts, which repair starts from
#if defined(VORONOI_MEASURE)
    int screened = 0;       // measurement build only: tetrahedra the screen listed, last repair
    long long checked = 0;  // failing triangles the rounds handled, last repair
#endif
};

// What repair() did. Used by: Mesh::update, tet_flips::finish.
struct Stats {
    long long rounds = 0, checked = 0, flips23 = 0, flips32 = 0;
    int stuck = 0;            // entries listed in State::stuck
    bool unfinished = false;  // out of rounds with triangles left: the mesh may fail anywhere
};

namespace detail {

// A planned flip as kept between planning and applying: the kind, the old tetrahedra and five particles, {a, far, f0,
// f1, f2} for 2->3 (old t, u; new: a, far, f[e], f[e + 1]) or {p, q, o0, o1, o2} for 3->2 (old: the three around edge
// p-q; new: p or q, o0, o1, o2). Used by: State::plans; plan, repair (plan and apply steps).
struct Packed {
    enum Kind : int { none = 0, flip23 = 2, flip32 = 3, unflippable = -1, rejected = -2 };
    int kind;
    int old[3];
    int v[5];
};

// The entry of `a` at the corner of `v` holding particle x (x must be one of them): a select, not an index.
KOKKOS_INLINE_FUNCTION int at_particle(const int (&v)[4], const int (&a)[4], int x) {
    return v[0] == x ? a[0] : v[1] == x ? a[1] : v[2] == x ? a[2] : a[3];
}

// Whether candidate c of triangle t|u (t = (a, f), far the particle of u that t lacks) is a valid flip, and if so the
// flip packed in p: c = 0 the 2->3 flip, valid when segment a-far crosses the triangle (the three new tetrahedra share
// one orientation); c = 1, 2, 3 the 3->2 flip around edge p-q = (f[c - 1], f[c % 3]), valid when exactly three
// tetrahedra, u among them, share the edge and p and q lie on opposite sides of the triangle of their other particles.
// Used by: plan.
template <class Mesh, class Point>
KOKKOS_INLINE_FUNCTION bool candidate(const Mesh& m, const Point& point, int t, int u, int a, int far,
                                      const int (&f)[3], int c, Packed& s) {
    if (c == 0) {
        int positive = 0, negative = 0;
        for (int e = 0; e < 3; ++e) {
            const int n[4] = {a, far, f[e], f[(e + 1) % 3]};
            const int o = ep::orient(point, n);
            positive += o > 0;
            negative += o < 0;
        }
        s = {Packed::flip23, {t, u, -1}, {a, far, f[0], f[1], f[2]}};
        return positive == 3 || negative == 3;
    }
    const int p = c == 1 ? f[0] : c == 2 ? f[1] : f[2], q = c == 1 ? f[1] : c == 2 ? f[2] : f[0];
    // The ring around p-q from t closes after exactly three tetrahedra.
    const int r1 = tet_mesh::next_around_edge(m, t, -1, p, q);
    if (r1 < 0 || r1 == t) return false;
    const int r2 = tet_mesh::next_around_edge(m, r1, t, p, q);
    if (r2 < 0 || r2 == t || tet_mesh::next_around_edge(m, r2, r1, p, q) != t) return false;
    if (r1 != u && r2 != u) return false;
    // Their particles other than p and q, in ring and corner order.
    int o[3] = {-1, -1, -1}, count = 0;
    const int ring[3] = {t, r1, r2};
    for (int r = 0; r < 3; ++r)
        for (int j = 0; j < 4; ++j) {
            const int v = m.vertex(ring[r], j);
            if (v == p || v == q || v == o[0] || v == o[1] || v == o[2] || count == 3) continue;
            o[0] = count == 0 ? v : o[0];
            o[1] = count == 1 ? v : o[1];
            o[2] = count == 2 ? v : o[2];
            ++count;
        }
    if (count != 3) return false;
    // The new tetrahedra (p, o) and (q, o) are flat exactly when p or q lies in the plane of o, and their orientations
    // are minus those of (o, p) and (o, q): two tests decide all four.
    const int side_p[4] = {o[0], o[1], o[2], p}, side_q[4] = {o[0], o[1], o[2], q};
    const int sp = ep::orient(point, side_p), sq = ep::orient(point, side_q);
    s = {Packed::flip32, {t, r1, r2}, {p, q, o[0], o[1], o[2]}};
    return sp != 0 && sq != 0 && sp * sq < 0;
}

// Whether the 2->3 flip of t|u (t = (a, f), u = (far, f), u across t's corner holding a) closes its cavity exactly (six
// outer faces, each new face matched once, each outer face used once, no new tetrahedron duplicating the outer one it
// attaches to), read off the rows of t and u: each holds the triangle f and the other exactly once as a neighbour,
// across the triangle; and no new tetrahedron (a, far, f[e], f[e + 1]) duplicates the outer one it attaches to (the
// tetrahedron across t's face opposite f[e + 2] must lack far, the one across u's must lack a). Used by: plan.
template <class M>
KOKKOS_INLINE_FUNCTION bool closes23(const M& m, int t, int u, int a, int far, const int (&f)[3]) {
    const int T[4] = {m.vertex(t, 0), m.vertex(t, 1), m.vertex(t, 2), m.vertex(t, 3)};
    const int U[4] = {m.vertex(u, 0), m.vertex(u, 1), m.vertex(u, 2), m.vertex(u, 3)};
    const int AT[4] = {m.adjacent(t, 0), m.adjacent(t, 1), m.adjacent(t, 2), m.adjacent(t, 3)};
    const int AU[4] = {m.adjacent(u, 0), m.adjacent(u, 1), m.adjacent(u, 2), m.adjacent(u, 3)};
    auto holds = [](const int(&v)[4], int x) { return v[0] == x || v[1] == x || v[2] == x || v[3] == x; };
    for (int e = 0; e < 3; ++e)
        if (!holds(U, f[e])) return false;
    int inner = 0;
    for (int j = 0; j < 4; ++j) {
        inner += AT[j] == t || AT[j] == u;
        inner += AU[j] == t || AU[j] == u;
    }
    if (inner != 2 || at_particle(T, AT, a) != u || at_particle(U, AU, far) != t) return false;
    for (int e = 0; e < 3; ++e) {
        const int g = f[(e + 2) % 3];
        const int wt = at_particle(T, AT, g), wu = at_particle(U, AU, g);
        if (wt >= 0 && tet_mesh::position(m, wt, far) >= 0) return false;
        if (wu >= 0 && tet_mesh::position(m, wu, a) >= 0) return false;
    }
    return true;
}

// Whether the 3->2 flip around edge p-q of the ring `old` (o: the ring's other particles) closes its cavity exactly (as
// closes23): each old tetrahedron holds p, q and two of o, the three lack different ones, exactly six of their
// faces (those opposite an o) are shared among them and those opposite p and q are not; and no new tetrahedron
// (p, o) or (q, o) duplicates the outer one it attaches to (across the face opposite q, resp. p, of the old one lacking
// o[i], which must lack o[i]). Used by: plan.
template <class M>
KOKKOS_INLINE_FUNCTION bool closes32(const M& m, const int (&old)[3], int p, int q, const int (&o)[3]) {
    int inner = 0, lacked = 0;
    for (int r = 0; r < 3; ++r) {
        const int V[4] = {m.vertex(old[r], 0), m.vertex(old[r], 1), m.vertex(old[r], 2), m.vertex(old[r], 3)};
        const int A[4] = {m.adjacent(old[r], 0), m.adjacent(old[r], 1), m.adjacent(old[r], 2), m.adjacent(old[r], 3)};
        auto holds = [&](int x) { return V[0] == x || V[1] == x || V[2] == x || V[3] == x; };
        auto is_old = [&](int w) { return w == old[0] || w == old[1] || w == old[2]; };
        if (!holds(p) || !holds(q) || holds(o[0]) + holds(o[1]) + holds(o[2]) != 2) return false;
        const int lacks = !holds(o[0]) ? 0 : !holds(o[1]) ? 1 : 2;
        lacked |= 1 << lacks;
        for (int j = 0; j < 4; ++j) inner += is_old(A[j]);
        const int ap = at_particle(V, A, p), aq = at_particle(V, A, q);
        if (is_old(ap) || is_old(aq)) return false;
        const int missing = lacks == 0 ? o[0] : lacks == 1 ? o[1] : o[2];
        if (aq >= 0 && tet_mesh::position(m, aq, missing) >= 0) return false;
        if (ap >= 0 && tet_mesh::position(m, ap, missing) >= 0) return false;
    }
    return inner == 6 && lacked == 7;
}

// Plan the repair of triangle t|u, t's face opposite corner k, which fails the local Delaunay test (the find step
// decided it on the same positions): the first candidate (2->3, then the 3->2 flips around the triangle's three edges)
// that is valid and closes its cavity; else unflippable (none valid) or rejected (none of the valid ones closes). The
// search and the check are loops a warp runs together, not one copy per candidate. Used by: repair (plan step).
template <class Mesh, class Point>
KOKKOS_INLINE_FUNCTION Packed plan(const Mesh& m, const Point& point, int t, int k, int u) {
    Packed s{Packed::none, {-1, -1, -1}, {0, 0, 0, 0, 0}};
    const int T[4] = {m.vertex(t, 0), m.vertex(t, 1), m.vertex(t, 2), m.vertex(t, 3)};
    const int far = tet_mesh::far_vertex(m, t, u);
    if (far < 0) return s;
    int f[3];
    tet_mesh::sorted_face(T, k, f);
    const int a = k == 0 ? T[0] : k == 1 ? T[1] : k == 2 ? T[2] : T[3];
    bool rejected = false;
    for (int from = 0; from < 4;) {
        int c = from;
        Packed p{};
#if defined(__CUDA_ARCH__)
#pragma unroll 1
#endif
        for (; c < 4; ++c)
            if (candidate(m, point, t, u, a, far, f, c, p)) break;
        if (c == 4) break;
        const bool two = p.kind == Packed::flip23;
        const int old[3] = {p.old[0], p.old[1], p.old[2]}, o[3] = {p.v[2], p.v[3], p.v[4]};
        if (two ? closes23(m, t, u, a, far, f) : closes32(m, old, p.v[0], p.v[1], o)) return p;
        rejected = true;
        from = c + 1;
    }
    s.kind = rejected ? Packed::rejected : Packed::unflippable;
    return s;
}

// Mark edge a-b on the first new tetrahedron of `fresh` that holds both ends, its bit in `bits` (none holds the edge a
// 3->2 flip removes). Used by: represent_edges, apply_flip.
template <int Freshes>
KOKKOS_INLINE_FUNCTION void represent(const int (&fresh)[Freshes][4], int a, int b, int (&bits)[Freshes]) {
    bool done = false;
    for (int r = 0; r < Freshes; ++r) {
        int pa = -1, pb = -1;
        for (int c = 0; c < 4; ++c) {
            pa = fresh[r][c] == a ? c : pa;
            pb = fresh[r][c] == b ? c : pb;
        }
        if (!done && pa >= 0 && pb >= 0) {
            bits[r] |= 1 << tet_mesh::edge_index(pa, pb);
            done = true;
        }
    }
}
// The same for every edge an old tetrahedron (corners v) represents (bits `edges`). Used by: apply_flip.
template <int Freshes>
KOKKOS_INLINE_FUNCTION void represent_edges(const int (&v)[4], int edges, const int (&fresh)[Freshes][4],
                                            int (&bits)[Freshes]) {
    for (int e = 0; e < 6; ++e) {
        int ka, kb;
        tet_mesh::edge_ends(e, ka, kb);
        if (edges >> e & 1) represent(fresh, v[ka], v[kb], bits);
    }
}

// Write the first `freshes` new tetrahedra of a flip into `slot`: vertices, adjacency, edge bits. Each new tetrahedron's
// corners 2 and 3 are swapped when `swap` (its orientation was negative); slots from `popped_from` on were popped and
// keep their bits (cleared when freed). Used by: apply_flip.
template <class M>
KOKKOS_INLINE_FUNCTION void write_fresh(const M& m, const int (&slot)[3], int freshes, const int (&fresh)[3][4],
                                        const int (&adjacent)[3][4], const bool (&swap)[3], const int (&bits)[3],
                                        int popped_from) {
    for (int r = 0; r < 3; ++r) {
        if (r >= freshes) continue;
        const int s = slot[r];
        for (int c = 0; c < 4; ++c) {
            const int cs = !swap[r] || c < 2 ? c : 5 - c;  // where corner c goes
            m.vertex(s, cs) = fresh[r][c];
            m.adjacent(s, cs) = adjacent[r][c];
        }
        int b = 0;  // edge bits in the stored corner order
        for (int e = 0; e < 6; ++e) {
            int ka, kb;
            tet_mesh::edge_ends(e, ka, kb);
            const int sa = !swap[r] || ka < 2 ? ka : 5 - ka, sb = !swap[r] || kb < 2 ? kb : 5 - kb;
            if (bits[r] >> e & 1) b |= 1 << tet_mesh::edge_index(sa, sb);
        }
        m.edges(s) = static_cast<std::uint8_t>(r >= popped_from ? m.edges(s) | b : b);
    }
}

// Each outer face c of new tetrahedron r: the outer tetrahedron w = adjacent[r][c] links back to slot[r] across it.
// Used by: apply_flip.
template <class M>
KOKKOS_INLINE_FUNCTION void link_back(const M& m, const int (&slot)[3], const int (&fresh)[3][4],
                                      const int (&adjacent)[3][4], const bool (&outer)[3][4]) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) {
            const int w = adjacent[r][c];
            if (!outer[r][c] || w < 0) continue;
            const int f0 = c == 0 ? fresh[r][1] : fresh[r][0], f1 = c <= 1 ? fresh[r][2] : fresh[r][1],
                      f2 = c <= 2 ? fresh[r][3] : fresh[r][2];
            m.adjacent(w, tet_mesh::opposite(m, w, f0, f1, f2)) = slot[r];
        }
}

// Apply flip p into slot, connected by formula, one code path for both kinds (a warp's threads stay together):
// 2->3 (old t, u reused, slot[2] popped or grown): new (a, far, f[e], f[e + 1]), outside through the faces of u and t
// opposite f[e + 2]; 3->2 (the first two old tetrahedra reused, the third freed and listed in `freed`): new (p, o) and
// (q, o), outside through the faces opposite q and p of the old tetrahedron lacking each o[i]. Reads every old row
// before the first write. Used by: repair (apply step).
template <class M, class Point, class Alive, class Freed, class Counters>
KOKKOS_INLINE_FUNCTION void apply_flip(const M& m, const Point& point, const Packed& p, const int (&slot)[3],
                                       const Alive& alive, const Freed& freed, const Counters& counters) {
    const bool two = p.kind == Packed::flip23;
    const int olds = two ? 2 : 3, freshes = two ? 3 : 2;
    int V[3][4], A[3][4], E[3];
    for (int r = 0; r < 3; ++r) {
        const int t = r < olds ? p.old[r] : p.old[0];
        for (int c = 0; c < 4; ++c) {
            V[r][c] = m.vertex(t, c);
            A[r][c] = m.adjacent(t, c);
        }
        E[r] = r < olds ? m.edges(t) : 0;
    }
    int fresh[3][4], adjacent[3][4], bits[3] = {0, 0, 0};
    bool outer[3][4], swap[3];
    // 3->2: across the faces of p (resp. q) and the two o other than o[i], from the old tetrahedron lacking o[i].
    int across_p[3] = {-1, -1, -1}, across_q[3] = {-1, -1, -1};
    for (int r = 0; r < 3; ++r) {
        auto holds = [&](int x) { return V[r][0] == x || V[r][1] == x || V[r][2] == x || V[r][3] == x; };
        const int lacks = !holds(p.v[2]) ? 0 : !holds(p.v[3]) ? 1 : 2;
        const int aq = at_particle(V[r], A[r], p.v[1]), ap = at_particle(V[r], A[r], p.v[0]);
        for (int i = 0; i < 3; ++i) {
            across_p[i] = lacks == i ? aq : across_p[i];
            across_q[i] = lacks == i ? ap : across_q[i];
        }
    }
    for (int e = 0; e < 3; ++e) {
        fresh[e][0] = two || e == 0 ? p.v[0] : p.v[1];
        fresh[e][1] = two ? p.v[1] : p.v[2];
        fresh[e][2] = two ? p.v[2 + e] : p.v[3];
        fresh[e][3] = two ? p.v[2 + (e + 1) % 3] : p.v[4];
        const int g = p.v[2 + (e + 2) % 3];  // 2->3: the triangle's particle new tetrahedron e lacks
        adjacent[e][0] = two ? at_particle(V[1], A[1], g) : e == 0 ? slot[1] : slot[0];
        for (int c = 1; c < 4; ++c) adjacent[e][c] = e == 0 ? across_p[c - 1] : across_q[c - 1];
        if (two) {
            adjacent[e][1] = at_particle(V[0], A[0], g);
            adjacent[e][2] = slot[(e + 1) % 3];
            adjacent[e][3] = slot[(e + 2) % 3];
        }
        for (int c = 0; c < 4; ++c) outer[e][c] = two ? c < 2 : e < 2 && c > 0;
    }
    // A 2->3 flip's three new tetrahedra share one orientation (the plan's test); a 3->2 flip's two have opposite ones.
    const bool negative = ep::orient(point, fresh[0]) < 0;
    for (int e = 0; e < 3; ++e) swap[e] = two || e == 0 ? negative : !negative;
    for (int r = 0; r < 3; ++r) represent_edges(V[r], E[r], fresh, bits);
    represent(fresh, two ? p.v[0] : -1, two ? p.v[1] : -1, bits);  // 2->3: the new edge a-far
    if (!two) {
        alive(p.old[2]) = 0;
        m.edges(p.old[2]) = 0;
        freed(tet_mesh::claim(&counters(counter::freed), 1)) = p.old[2];
    }
    write_fresh(m, slot, freshes, fresh, adjacent, swap, bits, 2);
    link_back(m, slot, fresh, adjacent, outer);
}

// The vote of plan i: a bijective hash of i, so winners of overlapping plans do not follow the work list's order.
// Used by: repair (vote and apply steps).
KOKKOS_INLINE_FUNCTION unsigned long long priority(int i) {
    unsigned x = static_cast<unsigned>(i) * 0x9e3779b1u;  // odd multiplier: a bijection
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    return x;
}

// Add stuck entry (t, k) to State::stuck (k: corner, or inverted/rejected). Used by: repair.
template <class Counters, class Stuck>
KOKKOS_INLINE_FUNCTION void list_stuck(const Counters& counters, const Stuck& stuck, int t, int k) {
    const int at = tet_mesh::claim(&counters(counter::stuck), 1);
    if (at < static_cast<int>(stuck.extent(0))) {
        stuck(at, 0) = t;
        stuck(at, 1) = k;
    }
}

// List t as dirty unless it already is. Used by: move_screened, repair, tet_flips::detail::mark_stuck/finish.
template <class State>
KOKKOS_INLINE_FUNCTION void mark(const State& s, int t) {
    if (Kokkos::atomic_exchange(&s.dirty(t), 1) == 0) s.listed(tet_mesh::claim(&s.counters(counter::listed), 1)) = t;
}

// One flip round's steps, one item each: for the per-step kernels of large rounds and for the one team that runs small
// rounds back to back (repair). Holds the round's views and the host's sizes at its start.
template <class MemorySpace>
struct Round {
    tet_mesh::Mesh<MemorySpace> m;
    State<MemorySpace> st;
    Kokkos::View<int* [2], MemorySpace> work, stuck;
    Kokkos::View<Packed*, MemorySpace> plans;
    Kokkos::View<int*, MemorySpace> freed;
    int capacity = 0, work_capacity = 0;

    // 1. Listed tetrahedron i: its failing triangles onto the work list, each triangle once (from the dirty side, or
    // from the lower index when both are). Turned inside out: listed stuck for the host, but flips usually untangle it.
    KOKKOS_INLINE_FUNCTION void find(int i) const {
        const int t = st.listed(i);
        if (!st.alive(t)) return;
        const auto& point = st.point;
        const int v[4] = {m.vertex(t, 0), m.vertex(t, 1), m.vertex(t, 2), m.vertex(t, 3)};
        const ep::Offset o[4] = {
            {0, 0, 0}, ep::offset(point, v[0], v[1]), ep::offset(point, v[0], v[2]), ep::offset(point, v[0], v[3])};
        // Turned inside out: listed for the host, but flips usually untangle it.
        if (ep::orient(o[1], o[2], o[3]) <= 0) list_stuck(st.counters, stuck, t, inverted);
        for (int k = 0; k < 4; ++k) {
            const int u = m.adjacent(t, k);
            if (u < 0 || (st.dirty(u) && u < t)) continue;
            // The particle of u that t lacks (t's corners are in registers).
            int far = -1;
            for (int j = 0; j < 4; ++j) {
                const int w = m.vertex(u, j);
                if (far < 0 && w != v[0] && w != v[1] && w != v[2] && w != v[3]) far = w;
            }
            if (far < 0 || !ep::in_sphere(point, v, o, far)) continue;
            const int at = tet_mesh::claim(&st.counters(counter::work), 1);
            if (at < work_capacity) {
                work(at, 0) = t;
                work(at, 1) = k;
            }
        }
    }

    // 2-3. Plan work item i and vote with `vote` on the tetrahedra it rewrites and their neighbours.
    KOKKOS_INLINE_FUNCTION void plan(int i, unsigned long long vote) const {
        const int t = work(i, 0), k = work(i, 1);
        Packed p{Packed::none, {-1, -1, -1}, {0, 0, 0, 0, 0}};
        const int u = m.adjacent(t, k);
        if (st.alive(t) && u >= 0 && st.alive(u)) p = detail::plan(m, st.point, t, k, u);
        plans(i) = p;
        if (p.kind < 0) list_stuck(st.counters, stuck, t, p.kind == Packed::rejected ? rejected : k);
        if (p.kind <= 0) return;  // never votes, so it cannot block other flips
        for (int r = 0; r < 3; ++r) {
            if (r == 2 && p.kind == Packed::flip23) break;
            const int o = p.old[r];
            Kokkos::atomic_max(&st.lock(o), vote);
            for (int j = 0; j < 4; ++j) {
                const int w = m.adjacent(o, j);
                if (w >= 0) Kokkos::atomic_max(&st.lock(w), vote);
            }
        }
    }

    // 4. Apply plan i if it won every vote; losers stay dirty for the next round. Slots: the old ones, then a free one
    // (of the `free_top` at the round's start) or a new one after `count`; a flip with no room retries next round.
    KOKKOS_INLINE_FUNCTION void apply(int i, unsigned long long vote, int free_top, int count) const {
        const Packed p = plans(i);
        if (p.kind <= 0) return;
        const bool two = p.kind == Packed::flip23;  // 2->3: two old tetrahedra, three new
        bool won = true;
        for (int r = 0; r < 3 && won; ++r) {
            if (r == 2 && two) break;
            const int o = p.old[r];
            won = st.lock(o) == vote;
            for (int j = 0; j < 4 && won; ++j) {
                const int w = m.adjacent(o, j);
                if (w >= 0) won = st.lock(w) == vote;
            }
        }
        int slot[3] = {p.old[0], p.old[1], p.old[2]};
        if (won && two) {
            const int popped = Kokkos::atomic_fetch_add(&st.counters(counter::pops), 1);
            slot[2] = popped < free_top ? st.free_slots(free_top - 1 - popped)
                                        : count + Kokkos::atomic_fetch_add(&st.counters(counter::growth), 1);
            won = slot[2] < capacity;
        }
        if (!won) {
            mark(st, p.old[0]);
            return;
        }
        Kokkos::atomic_inc(&st.counters(two ? counter::flips23 : counter::flips32));
        apply_flip(m, st.point, p, slot, st.alive, freed, st.counters);
        // List the new tetrahedra: free-list slots always, reused ones unless already touched. A round's flips share
        // none.
        const int freshes = two ? 3 : 2;
        bool listing[3];
        int listings = 0;
        for (int r = 0; r < 3; ++r) {
            if (r < freshes) st.alive(slot[r]) = 1;
            listing[r] = r < freshes && (r == 2 || !st.touched(slot[r]));
            listings += listing[r];
        }
        int at = tet_mesh::claim(&st.counters(counter::created), listings);
        for (int r = 0; r < 3; ++r)
            if (listing[r]) {
                if (at < static_cast<int>(st.touched_list.extent(0))) st.touched_list(at) = slot[r];
                ++at;
            }
        for (int r = 0; r < 3; ++r)
            if (r < 2 || !two) st.touched(p.old[r]) = 1;
        for (int r = 0; r < 3; ++r)
            if (r < freshes) {
                st.touched(slot[r]) = 1;
                mark(st, slot[r]);
            }
    }
};

// What the small rounds left for the host (repair), in State::small: entries of `small`.
namespace small {
constexpr int rounds = 0, checked = 1, flips23 = 2, flips32 = 3, free_count = 4, count = 5, dirty = 6, unreleased = 7,
              no_work = 8, needed = 9, size = 10;
}

// Rounds of the repair while the dirty list is short (at most `limit` tetrahedra), back to back on one team: the steps
// of a round are separated by team barriers instead of launches, and the sizes the host reads after each round stay on
// the device. Stops when a round finds no failing triangle, the list outgrows `limit` or the work and stuck lists, a
// flip found no slot, or the free list has no room for the slots a round freed (left unreleased for the host).
// Starts from the host's sizes; results in `out` (see small::). Used by: repair.
template <class ExecSpace>
void small_rounds(const ExecSpace& exec, const Round<typename ExecSpace::memory_space>& round,
                  const Kokkos::View<long long*, typename ExecSpace::memory_space>& out, int dirty_count, int free_top,
                  int count, unsigned long long first_round, int max_rounds, int limit) {
    using Policy = Kokkos::TeamPolicy<ExecSpace>;
    using Member = typename Policy::member_type;
    const auto& st = round.st;
    const int stuck_capacity = static_cast<int>(round.stuck.extent(0));
    const int free_capacity = static_cast<int>(st.free_slots.extent(0));
    auto body = KOKKOS_LAMBDA(const Member& team) {
        const auto& c = st.counters;
        int dirty = dirty_count, free_count = free_top, used = count, done = 0;
        long long checked = 0, flips23 = 0, flips32 = 0;
        int unreleased = 0, no_work = 0, needed = count;
        for (; done < max_rounds; ++done) {
            const int stuck_before = c(counter::stuck);
            if (dirty > limit || 4LL * dirty > round.work_capacity || stuck_before + 5LL * dirty > stuck_capacity)
                break;
            team.team_barrier();  // every thread has read the counters
            Kokkos::single(Kokkos::PerTeam(team), [&] { c(counter::freed) = c(counter::work) = 0; });
            team.team_barrier();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, dirty), [&](int i) { round.find(i); });
            team.team_barrier();
            const int work = c(counter::work);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, dirty), [&](int i) { st.dirty(st.listed(i)) = 0; });
            team.team_barrier();
            Kokkos::single(Kokkos::PerTeam(team), [&] {
                for (int k = counter::listed; k < counter::created; ++k) c(k) = 0;
            });
            team.team_barrier();
            if (work == 0) {
                no_work = 1;
                dirty = 0;  // unlisted above
                break;
            }
            const unsigned long long token = (first_round + done + 1) << 32;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, work),
                                 [&](int i) { round.plan(i, token | priority(i)); });
            team.team_barrier();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, work),
                                 [&](int i) { round.apply(i, token | priority(i), free_count, used); });
            team.team_barrier();
            checked += work;
            flips23 += c(counter::flips23);
            flips32 += c(counter::flips32);
            const int popped = Kokkos::min(c(counter::pops), free_count), growth = c(counter::growth);
            const int freed = c(counter::freed);
            free_count -= popped;
            dirty = c(counter::listed);
            needed = used + growth;
            const bool full = needed > round.capacity;
            used = Kokkos::min(used + growth, round.capacity);
            if (free_count + freed > free_capacity) {
                unreleased = freed;
                ++done;
                break;
            }
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, freed),
                                 [&](int i) { st.free_slots(free_count + i) = round.freed(i); });
            free_count += freed;
            team.team_barrier();
            if (full) {
                ++done;
                break;
            }
        }
        Kokkos::single(Kokkos::PerTeam(team), [&] {
            out(small::rounds) = done;
            out(small::checked) = checked;
            out(small::flips23) = flips23;
            out(small::flips32) = flips32;
            out(small::free_count) = free_count;
            out(small::count) = used;
            out(small::dirty) = dirty;
            out(small::unreleased) = unreleased;
            out(small::no_work) = no_work;
            out(small::needed) = needed;
        });
    };
    const int team_size = Kokkos::min(1024, Policy(exec, 1, 1).team_size_max(body, Kokkos::ParallelForTag()));
    Kokkos::parallel_for("tet_repair::small_rounds", Policy(exec, 1, team_size), body);
}

}  // namespace detail

// Room for `capacity` tetrahedra in the mesh and the per-tetrahedron state; contents are kept, new slots are zero
// (not alive, not dirty). Used by: prepare, repair (growth), tet_flips::finish.
template <class MemorySpace>
void reserve(tet_mesh::Mesh<MemorySpace>& m, State<MemorySpace>& s, int capacity) {
    Kokkos::resize(m.vertex, capacity);
    Kokkos::resize(m.adjacent, capacity);
    Kokkos::resize(m.centre, capacity);
    Kokkos::resize(m.edges, capacity);
    Kokkos::resize(s.alive, capacity);
    Kokkos::resize(s.touched, capacity);
    Kokkos::resize(s.dirty, capacity);
    Kokkos::resize(s.listed, capacity);
    Kokkos::resize(s.lock, capacity);
}

// Snapped positions and positive orientation of every tetrahedron; allocates the bookkeeping for `capacity`
// tetrahedra and `work_capacity` failing triangles per round and stuck entries (repair grows all when needed).
// Used by: voronoi::Mesh<3>::rebuild, once after tet_start::build.
template <class ExecSpace, class X>
State<typename ExecSpace::memory_space> prepare(const ExecSpace& exec, const X& x, const ep::Box& box,
                                                tet_mesh::Mesh<typename ExecSpace::memory_space>& m, int capacity,
                                                int work_capacity) {
    using Memory = typename ExecSpace::memory_space;
    State<Memory> s;
    const int n = static_cast<int>(x.extent(0));
    s.count = static_cast<int>(m.vertex.extent(0));
    s.point = ep::Points<Memory>("tet_repair_point", n, box);
    s.alive = Kokkos::View<char*, Memory>("tet_repair_alive", 0);  // sized by reserve
    s.touched = Kokkos::View<char*, Memory>("tet_repair_touched", 0);
    s.dirty = Kokkos::View<int*, Memory>("tet_repair_dirty", 0);
    s.listed = Kokkos::View<int*, Memory>("tet_repair_listed", 0);
    s.lock = Kokkos::View<unsigned long long*, Memory>("tet_repair_lock", 0);
    s.free_slots = Kokkos::View<int*, Memory>("tet_repair_free", capacity / 64 + 1024);  // grows in repair
    reserve(m, s, capacity);
    s.work = Kokkos::View<int* [2], Memory>("tet_repair_work", work_capacity);
    s.plans = Kokkos::View<detail::Packed*, Memory>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("tet_repair_plans")), work_capacity);
    s.freed = Kokkos::View<int*, Memory>("tet_repair_freed", work_capacity);
    s.stuck = Kokkos::View<int* [2], Memory>("tet_repair_stuck", work_capacity);
    s.touched_list = Kokkos::View<int*, Memory>("tet_repair_touched_list", capacity / 4 + 1024);
    s.counters = Kokkos::View<int*, Memory>("tet_repair_counters", counter::count);
    s.host_counters = Kokkos::View<int*, Kokkos::SharedHostPinnedSpace>("tet_repair_host_counters", counter::count);
    ep::snap(exec, x, box, s.point);
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
                edges(t) = static_cast<std::uint8_t>((b & ~0x1e) | (b & 0x2) << 1 | (b & 0x4) >> 1 | (b & 0x8) << 1 |
                                                     (b & 0x10) >> 1);
            }
        });
    return s;
}

// New positions x: snaps them, computes the circumcentres (m.centre) of the live tetrahedra (slivers exactly in a
// second pass) and lists those that may need repair: a neighbour's far vertex inside or near the circumsphere, or
// badly shaped. The margin covers the snapping; repair decides exactly.
// Used by: voronoi::Mesh<3>::update, before repair.
template <class ExecSpace, class X>
void move_screened(const ExecSpace& exec, const X& x, const ep::Box& box,
                   const tet_mesh::Mesh<typename ExecSpace::memory_space>& m,
                   State<typename ExecSpace::memory_space>& s) {
    ep::snap(exec, x, box, s.point);
    Kokkos::deep_copy(exec, s.counters, 0);
    const auto& st = s;
    const auto vertex = m.vertex;
    const auto adjacent = m.adjacent;
    const auto centre = m.centre;
    const auto counters = s.counters;
    const auto slivers = s.freed;                       // free until the repair starts; counter::freed counts them
    const double snap = box.step * 1.7320508075688772;  // largest move by snapping
    // At most 64 registers: 8 blocks of 128 per SM. The kernel spills at this cap (985 B/cell of local traffic at 128³,
    // 0.1 spacings) and is still 4 to 7 % faster than without it (76 registers, no spills): occupancy pays more here.
    Kokkos::parallel_for(
        "tet_repair::screen", Kokkos::RangePolicy<ExecSpace, Kokkos::LaunchBounds<128, 8>>(exec, 0, s.count),
        KOKKOS_LAMBDA(int t) {
            st.dirty(t) = 0;
            if (!st.alive(t)) return;
            // Corners sorted by particle id and the centre in rounded arithmetic, as tet_mesh::stored_centre for
            // created tetrahedra: the same bits whichever kernel and corner order computed it.
            int v[4] = {vertex(t, 0), vertex(t, 1), vertex(t, 2), vertex(t, 3)};
            const int parity = tet_mesh::sort_corners(v);
            Vec3 d[3];
            tet_mesh::rounded_offsets(x, box, v, d);
            double volume6, shape;
            // Slivers: suspect; their centres are computed exactly below.
            if (tet_mesh::rounded_sliver(d, volume6, shape)) {
                const int at = tet_mesh::claim(&counters(counter::freed), 1);
                if (at < static_cast<int>(slivers.extent(0))) slivers(at) = t;
                detail::mark(st, t);
                return;
            }
            // Well shaped and clearly positive (in the stored corner order), else decide exactly.
            bool suspect = !(parity * volume6 > 1e-3 * shape);
            const Vec3 c = tet_mesh::rounded_circumcentre(d, volume6);
            // The row as stored, kept in registers: the tests below read no centre back.
            const tet_mesh::CentreRow row = tet_mesh::store_centre(x, box, v[0], c, centre, t);
            const double r2 = row.c[3];
            const double margin = 1e-6 * r2 + 8 * snap * (Kokkos::sqrt(r2) + snap);
            for (int k = 0; k < 4 && !suspect; ++k) {
                const int u = adjacent(t, k);
                // A triangle fails from both sides or from neither: the side with the lower index decides.
                if (u < t) continue;
                // The particle of u that t lacks (t's corners are in registers).
                int far = -1;
                for (int j = 0; j < 4; ++j) {
                    const int w = vertex(u, j);
                    if (far < 0 && w != v[0] && w != v[1] && w != v[2] && w != v[3]) far = w;
                }
                if (far < 0) continue;
                const Vec3 e = tet_mesh::from_point(row, tet_mesh::GridPoint(x, far, box), box.step);
                suspect = dot(e, e) < r2 + margin;
            }
            if (suspect) detail::mark(st, t);
        });
    Kokkos::deep_copy(exec, s.host_counters, counters);
    exec.fence();
    s.screen_counts = true;
    const int sliver_count = s.host_counters(counter::freed);
    if (sliver_count > static_cast<int>(slivers.extent(0)))
        tet_mesh::circumcentres(exec, x, box, m, s.count);
    else if (sliver_count > 0)
        tet_mesh::circumcentres(exec, x, box, m, -1,
                                Kokkos::View<const int*, typename ExecSpace::memory_space>(
                                    Kokkos::subview(slivers, std::make_pair(0, sliver_count))));
}

// Dirty lists up to this length run their rounds on one team (detail::small_rounds) instead of five launches and a
// host wait per round.
constexpr int small_limit = 2048;

// Flip until no dirty tetrahedron fails or no plan can proceed; what is left
// is listed in s.stuck (stats.stuck entries). At most `rounds` rounds.
// Used by: voronoi::Mesh<3>::update; tet_flips::finish (re-runs it after the host rebuilds).
template <class ExecSpace>
Stats repair(const ExecSpace& exec, tet_mesh::Mesh<typename ExecSpace::memory_space>& m,
             State<typename ExecSpace::memory_space>& s, int rounds = 1000) {
    using Range = Kokkos::RangePolicy<ExecSpace>;
    using Memory = typename ExecSpace::memory_space;
    Stats stats;
    auto work = s.work;
    auto plans = s.plans;
    const auto counters = s.counters;
    auto freed = s.freed;
    const auto& st = s;
    int work_capacity = static_cast<int>(work.extent(0));
    auto stuck = s.stuck;
    const auto point = s.point;
    const auto host = s.host_counters;
    auto read = [&] {  // one asynchronous copy into pinned memory, then wait
        Kokkos::deep_copy(exec, host, counters);
        exec.fence();
    };
    auto clear = [&](int k) { Kokkos::deep_copy(exec, Kokkos::subview(counters, k), 0); };
    // The work and stuck lists never drop entries: before a kernel could overrun them they grow (the stuck list keeps
    // its entries).
    auto grow = [&](int work_needed, int stuck_needed) {
        if (work_needed > work_capacity) {
            work_capacity = work_needed + work_needed / 2;
            Kokkos::resize(Kokkos::WithoutInitializing, s.work, work_capacity);
            // Plans and freed slots are written afresh after the listing: nothing to copy (#47).
            Kokkos::realloc(Kokkos::WithoutInitializing, s.plans, work_capacity);
            Kokkos::realloc(Kokkos::WithoutInitializing, s.freed, work_capacity);
            work = s.work;
            plans = s.plans;
            freed = s.freed;
        }
        if (stuck_needed > static_cast<int>(stuck.extent(0))) {
            Kokkos::resize(s.stuck, stuck_needed + stuck_needed / 2);
            stuck = s.stuck;
        }
    };
    // The stuck list fills for the whole repair.
    clear(counter::stuck);
    if (s.screen_counts)
        host(counter::stuck) = 0;
    else
        read();
    s.screen_counts = false;
    int dirty_count = host(counter::listed);
#if defined(VORONOI_MEASURE)
    s.screened = dirty_count;
#endif
    long long flips_at_last_retry = -1;
    bool finished = false;

    // A triangle stuck earlier may have become flippable since: look at the stuck ones again while that still makes
    // progress. Returns whether the repair is finished.
    auto retry = [&] {
        const long long flips = stats.flips23 + stats.flips32;
        finished = host(counter::stuck) == 0 || flips == flips_at_last_retry;
        if (finished) return true;
        flips_at_last_retry = flips;
        const int listed_stuck = Kokkos::min(host(counter::stuck), static_cast<int>(stuck.extent(0)));
        const auto alive = s.alive;
        Kokkos::parallel_for(
            "tet_repair::retry_stuck", Range(exec, 0, listed_stuck), KOKKOS_LAMBDA(int i) {
                const int t = stuck(i, 0);
                if (t >= 0 && alive(t)) detail::mark(st, t);
            });
        clear(counter::stuck);
        read();
        dirty_count = host(counter::listed);
        return false;
    };
    bool small_ok = true;  // see the small rounds below
    // The slots `freed_count` flips freed (s.freed) onto the free list after its first `base` entries.
    auto release = [&](int base, int freed_count) {
        // The free list holds only the slots 3->2 flips freed: sized to its use, not to the tetrahedra.
        if (base + freed_count > static_cast<int>(s.free_slots.extent(0)))
            Kokkos::resize(s.free_slots, base + freed_count + (base + freed_count) / 4);
        const auto to_free = s.free_slots;
        Kokkos::parallel_for(
            "tet_repair::release", Range(exec, 0, freed_count), KOKKOS_LAMBDA(int i) { to_free(base + i) = freed(i); });
        s.free_count = base + freed_count;
    };
    if (!s.small.data()) {
        s.small = Kokkos::View<long long*, Memory>("tet_repair_small", detail::small::size);
        s.host_small =
            Kokkos::View<long long*, Kokkos::SharedHostPinnedSpace>("tet_repair_host_small", detail::small::size);
    }

    for (int round = 0; round < rounds; ++round) {
        // The tetrahedra may have grown in the last round (reserve below).
        const int capacity = static_cast<int>(s.alive.extent(0));
        const auto listed = s.listed;
        const auto alive = s.alive;
        const auto dirty = s.dirty;
        clear(counter::freed);
        const int stuck_before = host(counter::stuck);
        grow(0, stuck_before + dirty_count);
        // At most 4 failing triangles per listed tetrahedron and one stuck entry each: when the lists hold that, the work
        // count is read with the apply step's counters, one wait per round instead of two.
        const bool ahead = 4LL * dirty_count <= work_capacity &&
                           stuck_before + 5LL * dirty_count <= static_cast<long long>(stuck.extent(0));
        const detail::Round<Memory> step{m, s, work, stuck, plans, freed, capacity, work_capacity};

        // A short dirty list: rounds back to back on one team, one wait for all of them.
        if (ahead && small_ok && dirty_count > 0 && dirty_count <= small_limit) {
            detail::small_rounds(exec, step, s.small, dirty_count, s.free_count, s.count, s.round, rounds - round,
                                 small_limit);
            Kokkos::deep_copy(exec, s.host_small, s.small);
            read();
            const auto& o = s.host_small;
            const int done = static_cast<int>(o(detail::small::rounds));
            s.round += done;
            stats.rounds += done;
            round += Kokkos::max(done, 1) - 1;  // the loop adds one
            small_ok = done > 0 || o(detail::small::no_work);
            stats.checked += o(detail::small::checked);
            stats.flips23 += o(detail::small::flips23);
            stats.flips32 += o(detail::small::flips32);
            s.count = static_cast<int>(o(detail::small::count));
            dirty_count = static_cast<int>(o(detail::small::dirty));
            s.free_count = static_cast<int>(o(detail::small::free_count));
            if (o(detail::small::unreleased) > 0) release(s.free_count, static_cast<int>(o(detail::small::unreleased)));
            const int needed = static_cast<int>(o(detail::small::needed));
            if (needed > capacity) reserve(m, s, capacity + Kokkos::max(capacity / 8, needed - capacity));
            if (o(detail::small::no_work)) {
                ++round;  // the round that found nothing
                if (retry()) break;
            }
            continue;
        }
        // 1. Failing triangles among the listed dirty tetrahedra.
        for (bool again = true; again;) {
            clear(counter::work);
            const detail::Round<Memory> finding{m, s, work, stuck, plans, freed, capacity, work_capacity};
            Kokkos::parallel_for(
                "tet_repair::find", Range(exec, 0, dirty_count), KOKKOS_LAMBDA(int i) { finding.find(i); });
            if (ahead) break;
            read();
            again = host(counter::work) > work_capacity;
            if (again) {
                grow(host(counter::work), 0);
                Kokkos::deep_copy(exec, Kokkos::subview(counters, counter::stuck), stuck_before);
            }
        }
        // Unlist them; the apply step lists the next round's. Ahead, the plan step does it (a thread for each), and
        // clears the apply step's counters, two launches fewer.
        const bool unlist_in_plan = ahead && dirty_count > 0;
        if (!unlist_in_plan) {
            Kokkos::parallel_for(
                "tet_repair::unlist", Range(exec, 0, dirty_count), KOKKOS_LAMBDA(int i) { dirty(listed(i)) = 0; });
            Kokkos::deep_copy(exec, Kokkos::subview(counters, std::make_pair(counter::listed, counter::created)), 0);
        }
        // Ahead: plan and apply over the bound; their threads past the count return.
        const int bound = ahead ? 4 * dirty_count : host(counter::work);
        if (bound == 0) {
            if (retry()) break;
            continue;
        }
        const unsigned long long token = (s.round + 1) << 32;

        // 2-4. Plan, vote, apply; losers stay dirty for the next round.
        if (!ahead) grow(0, host(counter::stuck) + bound);  // at most one stuck entry per plan
        const detail::Round<Memory> voting{m, s, work, stuck, plans, freed, capacity, work_capacity};
        Kokkos::parallel_for(
            "tet_repair::plan", Range(exec, 0, bound), KOKKOS_LAMBDA(int i) {
                if (unlist_in_plan && i < dirty_count) dirty(listed(i)) = 0;
                if (unlist_in_plan && i == 0)
                    for (int k = counter::listed; k < counter::created; ++k) counters(k) = 0;
                if (i < counters(counter::work)) voting.plan(i, token | detail::priority(i));
            });
        const int free_top = s.free_count;
        const int count = s.count;
        Kokkos::parallel_for(
            "tet_repair::apply", Range(exec, 0, bound), KOKKOS_LAMBDA(int i) {
                if (i < counters(counter::work)) voting.apply(i, token | detail::priority(i), free_top, count);
            });
        read();
        const int items = host(counter::work);
        stats.checked += items;
        if (items == 0) {  // ahead, and nothing failed: plan and apply did nothing
            if (retry()) break;
            continue;
        }
        ++s.round;
        ++stats.rounds;
        dirty_count = host(counter::listed);
        // Pops used, growth, and the slots freed this round onto the free list.
        const int popped = Kokkos::min(host(counter::pops), free_top);
        s.count = Kokkos::min(count + host(counter::growth), capacity);
        release(free_top - popped, host(counter::freed));
        stats.flips23 += host(counter::flips23);
        stats.flips32 += host(counter::flips32);
        // Flips that found no slot retry next round with 1/8 more room.
        if (count + host(counter::growth) > capacity)
            reserve(m, s, capacity + Kokkos::max(capacity / 8, count + host(counter::growth) - capacity));
    }
    stats.unfinished = !finished;
#if defined(VORONOI_MEASURE)
    s.checked = stats.checked;
#endif
    stats.stuck = Kokkos::min(host(counter::stuck), static_cast<int>(s.stuck.extent(0)));
    s.touched_entries =
        host(counter::created) <= static_cast<int>(s.touched_list.extent(0)) ? host(counter::created) : -1;
    // The created tetrahedra did not fit their list (this step updates all faces): room for 1/4 more next time.
    if (s.touched_entries < 0) Kokkos::resize(s.touched_list, host(counter::created) + host(counter::created) / 4);
    // Room for the tetrahedra a host fallback may add.
    const int spare = s.count / 64 + 1024;
    if (stats.stuck > 0 && s.count + spare > static_cast<int>(s.alive.extent(0))) reserve(m, s, s.count + spare);
    return stats;
}

}  // namespace tet_repair
