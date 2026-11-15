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
              motion = 9, skipped = 10, count = 11;
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
    // Quiet tetrahedra (V1): 0x80 | the last step (mod 128) whose screening a certificate covers, 0 if none.
    Kokkos::View<std::uint8_t*, MemorySpace> quiet;
    Kokkos::View<int*, MemorySpace> listed;  // the dirty tetrahedra
    // Votes: round in the high 32 bits, a bijective hash of the plan index below, so earlier rounds always lose.
    Kokkos::View<unsigned long long*, MemorySpace> lock;
    Kokkos::View<int*, MemorySpace> free_slots;  // popped from the top
    Kokkos::View<int* [2], MemorySpace> work;   // failing triangles of a round
    Kokkos::View<detail::Packed*, MemorySpace> plans;
    Kokkos::View<int*, MemorySpace> freed;
    // What the last repair left (see inverted, rejected).
    Kokkos::View<int* [2], MemorySpace> stuck;
    // Device counters (see counter), and their pinned host copy, read once or twice per round.
    Kokkos::View<int*, MemorySpace> counters;
    Kokkos::View<int*, Kokkos::SharedHostPinnedSpace> host_counters;
    int count = 0;  // tetrahedra slots in use
    int free_count = 0;
    int touched_entries = 0;
    unsigned long long round = 0;
    bool screen_counts = false;  // host_counters hold move_screened's counts, which repair starts from
    // V1: screens so far, and the most an edge may change per step (grid steps) that the certificates assume (0: none).
    int step = 0;
    double speed = 0;
    bool quiet_screen = true;  // skip the neighbour tests of certified tetrahedra
    int skipped = 0;           // tetrahedra the last screen skipped
    int motion = 0;            // the most an edge changed in the last step, grid steps
#if defined(VORONOI_MEASURE)
    int screened = 0;       // measurement build only: tetrahedra the screen listed, last repair
    long long checked = 0;  // failing triangles the rounds handled, last repair
#endif
};

// What repair() did. Used by: Mesh::update, tet_flips::finish.
struct Stats {
    long long rounds = 0, checked = 0, flips23 = 0, flips32 = 0;
    int stuck = 0;  // entries listed in State::stuck
    bool unfinished = false;  // out of rounds with triangles left: the mesh may fail anywhere
};

// The outer faces of the cavity of tetrahedra old[0, olds): the sorted particles of each (key) and the tetrahedron
// across it (across, -1 if none). Returns their number, or -1 if there are more than 8.
// Used by: closes, repair (apply step).
template <class M>
KOKKOS_INLINE_FUNCTION int outer_faces(const M& m, const int* old, int olds, int (*key)[3], int* across) {
    int outers = 0;
    for (int r = 0; r < olds; ++r) {
        const int o = old[r];
        const int v[4] = {m.vertex(o, 0), m.vertex(o, 1), m.vertex(o, 2), m.vertex(o, 3)};
        for (int j = 0; j < 4; ++j) {
            const int w = m.adjacent(o, j);
            bool inner = false;
            for (int q = 0; q < olds; ++q) inner |= w == old[q];
            if (inner) continue;
            if (outers == 8) return -1;
            tet_mesh::sorted_face(v, j, key[outers]);
            across[outers++] = w;
        }
    }
    return outers;
}

// Whether new tetrahedra `fresh` close the cavity of the old ones `old` of mesh m exactly: six outer faces, each
// new face matched once, each outer face used once, no new tetrahedron duplicating the outer one it attaches to.
// Used by: plan, repair (plan step).
template <class M>
KOKKOS_INLINE_FUNCTION bool closes(const M& m, const int* old, int olds, const int (*fresh)[4], int freshes) {
    int outer_key[8][3], outer_tet[8];
    if (outer_faces(m, old, olds, outer_key, outer_tet) != 6) return false;
    int used[6] = {0, 0, 0, 0, 0, 0};
    for (int r = 0; r < freshes; ++r)
        for (int c = 0; c < 4; ++c) {
            int key[3];
            tet_mesh::sorted_face(fresh[r], c, key);
            int matches = 0;
            for (int q = 0; q < 6; ++q)
                if (tet_mesh::same_face(key, outer_key[q])) {
                    ++matches;
                    ++used[q];
                    if (outer_tet[q] >= 0 && tet_mesh::position(m, outer_tet[q], fresh[r][c]) >= 0) return false;
                }
            for (int s2 = 0; s2 < freshes; ++s2) {
                if (s2 == r) continue;
                for (int c2 = 0; c2 < 4; ++c2) {
                    int other[3];
                    tet_mesh::sorted_face(fresh[s2], c2, other);
                    matches += tet_mesh::same_face(key, other);
                }
            }
            if (matches != 1) return false;
        }
    for (int q = 0; q < 6; ++q)
        if (used[q] != 1) return false;
    return true;
}

namespace detail {

// A planned flip: the old tetrahedra and the new ones (vertex lists). Used by: plan, pack, unpack.
struct Plan {
    enum Kind : int { none = 0, flip23 = 2, flip32 = 3, unflippable = -1, rejected = -2 };
    int kind = none;
    int old[3] = {-1, -1, -1};
    int fresh[3][4];
    int olds = 0, freshes = 0;
};

// A plan as kept between planning and applying: the kind, the old tetrahedra and five particles, {a, far, f0, f1, f2}
// for 2->3 (new: a, far, f[e], f[e + 1]) or {p, q, o0, o1, o2} for 3->2 (new: p or q, o0, o1, o2).
// Used by: State::plans; pack, unpack (repair's plan and apply steps).
struct Packed {
    int kind;
    int old[3];
    int v[5];
};

// Plan -> Packed (repair, plan step).
KOKKOS_INLINE_FUNCTION Packed pack(const Plan& p) {
    Packed s{p.kind, {p.old[0], p.old[1], p.old[2]}, {}};
    if (p.kind == Plan::flip23) {
        for (int e = 0; e < 3; ++e) s.v[2 + e] = p.fresh[e][2];
        s.v[0] = p.fresh[0][0];
        s.v[1] = p.fresh[0][1];
    } else if (p.kind == Plan::flip32) {
        s.v[0] = p.fresh[0][0];
        s.v[1] = p.fresh[1][0];
        for (int c = 0; c < 3; ++c) s.v[2 + c] = p.fresh[0][1 + c];
    }
    return s;
}

// Packed -> Plan (repair, apply step).
KOKKOS_INLINE_FUNCTION Plan unpack(const Packed& s) {
    Plan p;
    p.kind = s.kind;
    for (int r = 0; r < 3; ++r) p.old[r] = s.old[r];
    if (s.kind == Plan::flip23) {
        p.olds = 2;
        p.freshes = 3;
        for (int e = 0; e < 3; ++e) {
            const int n[4] = {s.v[0], s.v[1], s.v[2 + e], s.v[2 + (e + 1) % 3]};
            for (int c = 0; c < 4; ++c) p.fresh[e][c] = n[c];
        }
    } else if (s.kind == Plan::flip32) {
        p.olds = 3;
        p.freshes = 2;
        for (int r = 0; r < 2; ++r) {
            p.fresh[r][0] = s.v[r];
            for (int c = 0; c < 3; ++c) p.fresh[r][1 + c] = s.v[2 + c];
        }
    }
    return p;
}

// The 2->3 flip of triangle t|u, apex a of t and far vertex far of u, f the triangle: valid when segment a-far
// crosses the triangle, i.e. the three new tetrahedra share one orientation. Used by: plan.
template <class Point>
KOKKOS_INLINE_FUNCTION bool flip23(const Point& point, int t, int u, int a, int far, const int* f, Plan& p) {
    p = {Plan::flip23, {t, u, -1}, {}, 2, 3};
    int positive = 0, negative = 0;
    for (int e = 0; e < 3; ++e) {
        const int n[4] = {a, far, f[e], f[(e + 1) % 3]};
        const int o = ep::orient(point, n);
        positive += o > 0;
        negative += o < 0;
        for (int c = 0; c < 4; ++c) p.fresh[e][c] = n[c];
    }
    return positive == 3 || negative == 3;
}

// The 3->2 flip around edge pv-qv of triangle t|u: valid when exactly three tetrahedra, u among them, share the edge
// and pv and qv lie on opposite sides of the triangle of their other particles. Used by: plan.
template <class Mesh, class Point>
KOKKOS_INLINE_FUNCTION bool flip32(const Mesh& m, const Point& point, int t, int u, int pv, int qv, Plan& p) {
    int ring[3] = {t, -1, -1}, size = 1;
    bool closed = false;
    for (int current = t, previous = -1; size <= 3;) {
        const int next = tet_mesh::next_around_edge(m, current, previous, pv, qv);
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
    if (!closed || size != 3 || (ring[1] != u && ring[2] != u)) return false;
    int others[3], count = 0;
    for (int r = 0; r < 3; ++r)
        for (int j = 0; j < 4; ++j) {
            const int v = m.vertex(ring[r], j);
            bool seen = v == pv || v == qv;
            for (int c = 0; c < count; ++c) seen |= others[c] == v;
            if (!seen && count < 3) others[count++] = v;
        }
    if (count != 3) return false;
    const int n1[4] = {pv, others[0], others[1], others[2]}, n2[4] = {qv, others[0], others[1], others[2]};
    if (ep::orient(point, n1) == 0 || ep::orient(point, n2) == 0) return false;
    const int side_p[4] = {others[0], others[1], others[2], pv}, side_q[4] = {others[0], others[1], others[2], qv};
    if (ep::orient(point, side_p) * ep::orient(point, side_q) >= 0) return false;
    p = {Plan::flip32, {ring[0], ring[1], ring[2]}, {}, 3, 2};
    for (int c = 0; c < 4; ++c) {
        p.fresh[0][c] = n1[c];
        p.fresh[1][c] = n2[c];
    }
    return true;
}

// Plan the repair of triangle t|u (t's face opposite corner k) if it fails the local Delaunay test: the first valid
// of the 2->3 flip and the 3->2 flips around its three edges (that closes its cavity, if `closing`); else
// unflippable (none valid) or rejected (none of the valid ones closes). Used by: repair (plan step).
template <class Mesh, class Point>
KOKKOS_INLINE_FUNCTION Plan plan(const Mesh& m, const Point& point, int t, int k, bool closing) {
    Plan p;
    const int u = m.adjacent(t, k);
    const int T[4] = {m.vertex(t, 0), m.vertex(t, 1), m.vertex(t, 2), m.vertex(t, 3)};
    const int far = tet_mesh::far_vertex(m, t, u);
    if (far < 0 || !ep::in_sphere(point, T, far)) return p;
    int f[3];
    tet_mesh::sorted_face(T, k, f);
    bool rejected = false;
    for (int candidate = 0; candidate < 4; ++candidate) {
        const bool valid = candidate == 0 ? flip23(point, t, u, T[k], far, f, p)
                                          : flip32(m, point, t, u, f[candidate - 1], f[candidate % 3], p);
        if (!valid) continue;
        if (!closing || closes(m, p.old, p.olds, p.fresh, p.freshes)) return p;
        rejected = true;
    }
    p.kind = rejected ? Plan::rejected : Plan::unflippable;
    return p;
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

// The largest motion `eps` (grid steps) of the points a[0..3], offsets from a fifth point that stays, under which the
// sign of the in-sphere determinant P = det[a_k, |a_k|^2] provably stays; 0 if none. P's first-order change from the
// cofactors, the rest bounded by Hadamard's inequality with the last column over L = max |a_k|; rounding of 1e-12 |P|.
KOKKOS_INLINE_FUNCTION double certified_motion(const double (*a)[3]) {
    double m[4][4], length[4], L = 0;
    for (int k = 0; k < 4; ++k) {
        const double q = a[k][0] * a[k][0] + a[k][1] * a[k][1] + a[k][2] * a[k][2];
        m[k][0] = a[k][0], m[k][1] = a[k][1], m[k][2] = a[k][2], m[k][3] = q;
        length[k] = Kokkos::sqrt(q);
        L = Kokkos::fmax(L, length[k]);
    }
    // The adjugate by 2x2 minors of rows 0, 1 (s) and 2, 3 (c) (Eberly, "The Laplace expansion theorem").
    const double s0 = m[0][0] * m[1][1] - m[1][0] * m[0][1], s1 = m[0][0] * m[1][2] - m[1][0] * m[0][2];
    const double s2 = m[0][0] * m[1][3] - m[1][0] * m[0][3], s3 = m[0][1] * m[1][2] - m[1][1] * m[0][2];
    const double s4 = m[0][1] * m[1][3] - m[1][1] * m[0][3], s5 = m[0][2] * m[1][3] - m[1][2] * m[0][3];
    const double c5 = m[2][2] * m[3][3] - m[3][2] * m[2][3], c4 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
    const double c3 = m[2][1] * m[3][2] - m[3][1] * m[2][2], c2 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
    const double c1 = m[2][0] * m[3][2] - m[3][0] * m[2][2], c0 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
    const double P = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
    // cofactor[k][j] of entry (k, j).
    const double cofactor[4][4] = {
        {m[1][1] * c5 - m[1][2] * c4 + m[1][3] * c3, -m[1][0] * c5 + m[1][2] * c2 - m[1][3] * c1,
         m[1][0] * c4 - m[1][1] * c2 + m[1][3] * c0, -m[1][0] * c3 + m[1][1] * c1 - m[1][2] * c0},
        {-m[0][1] * c5 + m[0][2] * c4 - m[0][3] * c3, m[0][0] * c5 - m[0][2] * c2 + m[0][3] * c1,
         -m[0][0] * c4 + m[0][1] * c2 - m[0][3] * c0, m[0][0] * c3 - m[0][1] * c1 + m[0][2] * c0},
        {m[3][1] * s5 - m[3][2] * s4 + m[3][3] * s3, -m[3][0] * s5 + m[3][2] * s2 - m[3][3] * s1,
         m[3][0] * s4 - m[3][1] * s2 + m[3][3] * s0, -m[3][0] * s3 + m[3][1] * s1 - m[3][2] * s0},
        {-m[2][1] * s5 + m[2][2] * s4 - m[2][3] * s3, m[2][0] * s5 - m[2][2] * s2 + m[2][3] * s1,
         -m[2][0] * s4 + m[2][1] * s2 - m[2][3] * s0, m[2][0] * s3 - m[2][1] * s1 + m[2][2] * s0}};
    double space[4], lift[4], g1 = 0, size = 0;
    for (int k = 0; k < 4; ++k) {
        space[k] = Kokkos::sqrt(cofactor[k][0] * cofactor[k][0] + cofactor[k][1] * cofactor[k][1] +
                                cofactor[k][2] * cofactor[k][2]);
        lift[k] = Kokkos::abs(cofactor[k][3]);
        g1 += space[k] + 2 * length[k] * lift[k];
        size += length[k] * space[k] + m[k][3] * lift[k];
    }
    const double room = Kokkos::abs(P) - 1e-12 * size;
    if (!(room > 0) || !(g1 > 0)) return 0;
    const double A = 1.4142135623730951 * L, B = 2.47;  // row norms of the scaled matrix and its change, eps <= L / 4
    double eps = Kokkos::fmin(0.25 * L, room / (2 * g1 * (1 + 1e-12)));
    for (int tries = 0; tries < 4; ++tries, eps *= 0.5) {
        double first = 0;
        for (int k = 0; k < 4; ++k) first += eps * (space[k] + lift[k] * (2 * length[k] + eps));
        const double b = B * eps;
        const double rest = L * (6 * A * A * b * b + 4 * A * b * b * b + b * b * b * b);
        if ((first + rest) * (1 + 1e-12) < room) return eps;
    }
    return 0;
}

// The largest `value` of the calling threads to the counter (atomically, one atomic per group on CUDA).
KOKKOS_INLINE_FUNCTION void largest(int* counter, int value) {
#if defined(__CUDA_ARCH__)
    namespace cg = cooperative_groups;
    const cg::coalesced_group g = cg::coalesced_threads();
    for (int d = 1; d < static_cast<int>(g.size()); d <<= 1) {  // prefix maxima, as tet_mesh::claim's sums
        const int before = g.shfl_up(value, d);
        if (static_cast<int>(g.thread_rank()) >= d) value = Kokkos::max(value, before);
    }
    if (g.thread_rank() == g.size() - 1) atomicMax(counter, value);
#else
    Kokkos::atomic_max(counter, value);
#endif
}

// List t as dirty unless it already is. Used by: move_screened, repair, tet_flips::detail::mark_stuck/finish.
template <class State>
KOKKOS_INLINE_FUNCTION void mark(const State& s, int t) {
    if (Kokkos::atomic_exchange(&s.dirty(t), 1) == 0) s.listed(tet_mesh::claim(&s.counters(counter::listed), 1)) = t;
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
    Kokkos::resize(s.quiet, capacity);
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
    s.quiet = Kokkos::View<std::uint8_t*, Memory>("tet_repair_quiet", 0);
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

// New positions x: computes the circumcentres (m.centre) of the live tetrahedra (slivers exactly in a second pass)
// and lists those that may need repair: a neighbour's far vertex inside or near the circumsphere, or badly shaped.
// The margin covers the snapping; repair decides exactly. Then snaps x into s.point.
// Quiet tetrahedra (V1): a tetrahedron whose neighbour tests passed keeps a certificate for the next k steps: each of its
// tested pairs keeps the sign of its in-sphere test while every point moves at most k * s.speed against a corner of
// their shared triangle (certified_motion). Each step measures the most any edge changed (an edge of t or of u joins
// that corner to every other point of the pair) and screens again without skipping if that exceeds s.speed.
// Used by: voronoi::Mesh<3>::update, before repair.
template <class ExecSpace, class X>
void move_screened(const ExecSpace& exec, const X& x, const ep::Box& box,
                   const tet_mesh::Mesh<typename ExecSpace::memory_space>& m,
                   State<typename ExecSpace::memory_space>& s) {
    const auto& st = s;
    const auto vertex = m.vertex;
    const auto adjacent = m.adjacent;
    const auto centre = m.centre;
    const auto counters = s.counters;
    const auto point = s.point;  // the last step's
    const auto quiet = s.quiet;
    const auto slivers = s.freed;  // free until the repair starts; counter::freed counts them
    const double snap = box.step * 1.7320508075688772;  // largest move by snapping
    const int step = ++s.step;
    auto screen = [&](bool skip, double speed) {
        Kokkos::deep_copy(exec, s.counters, 0);
        // At most 64 registers (72 unbounded): 8 blocks of 128 per SM, 6 % faster on the A100.
        Kokkos::parallel_for(
            "tet_repair::screen", Kokkos::RangePolicy<ExecSpace, Kokkos::LaunchBounds<128, 8>>(exec, 0, st.count),
            KOKKOS_LAMBDA(int t) {
                st.dirty(t) = 0;
                if (!st.alive(t)) return;
                auto from = [&](int v) {
                    return tet_mesh::from_point(centre, t, tet_mesh::GridPoint(x, v, box), box.step);
                };
                // Corners sorted by particle id and the centre in rounded arithmetic, as tet_mesh::stored_centre for
                // created tetrahedra: the same bits whichever kernel and corner order computed it.
                const int stored[4] = {vertex(t, 0), vertex(t, 1), vertex(t, 2), vertex(t, 3)};
                int v[4] = {stored[0], stored[1], stored[2], stored[3]};
                const int parity = tet_mesh::sort_corners(v);
                // The most an edge changed since the last step, in grid steps.
                long long now[4][3], moved[4][3];
                for (int k = 0; k < 4; ++k)
                    for (int a = 0; a < 3; ++a) {
                        now[k][a] = ep::snap(x(v[k], a), box, a);
                        moved[k][a] = ep::delta(point(v[k], a), now[k][a], box.period[a]);
                    }
                double edge2 = 0;
                for (int k = 0; k < 4; ++k)
                    for (int l = k + 1; l < 4; ++l) {
                        double e2 = 0;
                        for (int a = 0; a < 3; ++a) {
                            const double e = static_cast<double>(moved[k][a] - moved[l][a]);
                            e2 += e * e;
                        }
                        edge2 = Kokkos::fmax(edge2, e2);
                    }
                detail::largest(&counters(counter::motion), static_cast<int>(Kokkos::ceil(Kokkos::sqrt(edge2))) + 1);
                Vec3 d[3];
                tet_mesh::rounded_offsets(x, box, v, d);
                double volume6, shape;
                // Slivers: suspect; their centres are computed exactly below.
                if (tet_mesh::rounded_sliver(d, volume6, shape)) {
                    const int at = tet_mesh::claim(&counters(counter::freed), 1);
                    if (at < static_cast<int>(slivers.extent(0))) slivers(at) = t;
                    quiet(t) = 0;
                    detail::mark(st, t);
                    return;
                }
                // Well shaped and clearly positive (in the stored corner order), else decide exactly.
                bool suspect = !(parity * volume6 > 1e-3 * shape);
                const Vec3 c = tet_mesh::rounded_circumcentre(d, volume6);
                tet_mesh::store_centre(x, box, v[0], c, centre, t);
                const int q = quiet(t);
                if (!suspect && skip && (q & 0x80) && ((q - step) & 127) < 64) {
                    tet_mesh::claim(&counters(counter::skipped), 1);
                    return;
                }
                const double r2 = centre(t, 3);
                const double margin = 1e-6 * r2 + 8 * snap * (Kokkos::sqrt(r2) + snap);
                double certified = 1e300;
                for (int k = 0; k < 4 && !suspect; ++k) {
                    const int u = adjacent(t, k);
                    // A triangle fails from both sides or from neither: the side with the lower index decides.
                    if (u < t) continue;
                    const int far = tet_mesh::far_vertex(m, t, u);
                    if (far < 0) continue;
                    const Vec3 e = from(far);
                    suspect = dot(e, e) < r2 + margin;
                    if (suspect || speed <= 0) continue;
                    // The pair's points from w, a corner of the shared triangle (opposite stored corner k).
                    const int w = stored[(k + 1) & 3];
                    long long at[3];
                    for (int a = 0; a < 3; ++a) at[a] = ep::snap(x(w, a), box, a);
                    double offsets[4][3];
                    for (int j = 0, r = 0; j < 4; ++j) {
                        if (v[j] == w) continue;
                        for (int a = 0; a < 3; ++a)
                            offsets[r][a] = static_cast<double>(ep::delta(at[a], now[j][a], box.period[a]));
                        ++r;
                    }
                    for (int a = 0; a < 3; ++a)
                        offsets[3][a] =
                            static_cast<double>(ep::delta(at[a], ep::snap(x(far, a), box, a), box.period[a]));
                    certified = Kokkos::fmin(certified, detail::certified_motion(offsets));
                }
                if (suspect) {
                    quiet(t) = 0;
                    detail::mark(st, t);
                    return;
                }
                const double steps = speed > 0 ? Kokkos::fmin(50.0, Kokkos::floor(certified / speed)) : 0;
                quiet(t) = steps >= 1 ? static_cast<std::uint8_t>(0x80 | ((step + static_cast<int>(steps)) & 127)) : 0;
            });
        Kokkos::deep_copy(exec, s.host_counters, counters);
        exec.fence();
    };
    const bool skip = s.quiet_screen && s.speed > 0;
    screen(skip, s.quiet_screen ? s.speed : 0);
    s.motion = s.host_counters(counter::motion);
    // An edge changed more than the certificates allow: screen again, every tetrahedron, at the new speed.
    if (s.quiet_screen && s.motion > s.speed) {
        s.speed = 1.5 * s.motion;
        if (skip) screen(false, s.speed);
    } else if (s.motion < 0.5 * s.speed) {
        s.speed *= 0.75;  // lower is always safe: the certificates assumed more
    }
    s.skipped = s.host_counters(counter::skipped);
    ep::snap(exec, x, box, s.point);
    s.screen_counts = true;
    const int sliver_count = s.host_counters(counter::freed);
    if (sliver_count > static_cast<int>(slivers.extent(0)))
        tet_mesh::circumcentres(exec, x, box, m, s.count);
    else if (sliver_count > 0)
        tet_mesh::circumcentres(exec, x, box, m, -1,
                                Kokkos::View<const int*, typename ExecSpace::memory_space>(
                                    Kokkos::subview(slivers, std::make_pair(0, sliver_count))));
}

// Certificates of quiet tetrahedra no longer hold (the host repaired or the tetrahedra were reordered).
// Used by: voronoi::Mesh<3>::update and resort.
template <class ExecSpace, class MemorySpace>
void forget_quiet(const ExecSpace& exec, State<MemorySpace>& s) {
    Kokkos::deep_copy(exec, s.quiet, std::uint8_t(0));
}

// Flip until no dirty tetrahedron fails or no plan can proceed; what is left
// is listed in s.stuck (stats.stuck entries). At most `rounds` rounds.
// Used by: voronoi::Mesh<3>::update; tet_flips::finish (re-runs it after the host rebuilds).
template <class ExecSpace>
Stats repair(const ExecSpace& exec, tet_mesh::Mesh<typename ExecSpace::memory_space>& m,
             State<typename ExecSpace::memory_space>& s, int rounds = 1000) {
    using Range = Kokkos::RangePolicy<ExecSpace>;
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

    for (int round = 0; round < rounds; ++round) {
        // The tetrahedra may have grown in the last round (reserve below).
        const int capacity = static_cast<int>(s.alive.extent(0));
        const auto listed = s.listed;
        const auto vertex = m.vertex;
        const auto adjacent = m.adjacent;
        const auto alive = s.alive;
        const auto dirty = s.dirty;
        const auto lock = s.lock;
        const auto free_slots = s.free_slots;
        const auto edges = m.edges;
        // 1. Failing triangles among the listed dirty tetrahedra (each triangle once: from the dirty side, or from the
        // lower index when both are).
        clear(counter::freed);
        const int stuck_before = host(counter::stuck);
        grow(0, stuck_before + dirty_count);
        // At most 4 failing triangles per listed tetrahedron and one stuck entry each: when the lists hold that, the work
        // count is read with the apply step's counters, one wait per round instead of two.
        const bool ahead = 4LL * dirty_count <= work_capacity &&
                           stuck_before + 5LL * dirty_count <= static_cast<long long>(stuck.extent(0));
        for (bool again = true; again;) {
            clear(counter::work);
            Kokkos::parallel_for(
                "tet_repair::find", Range(exec, 0, dirty_count), KOKKOS_LAMBDA(int i) {
                    const int t = listed(i);
                    if (!alive(t)) return;
                    const int v[4] = {vertex(t, 0), vertex(t, 1), vertex(t, 2), vertex(t, 3)};
                    const ep::Offset o[4] = {{0, 0, 0}, ep::offset(point, v[0], v[1]), ep::offset(point, v[0], v[2]),
                                             ep::offset(point, v[0], v[3])};
                    // Turned inside out: listed for the host, but flips usually untangle it.
                    if (ep::orient(o[1], o[2], o[3]) <= 0) detail::list_stuck(counters, stuck, t, inverted);
                    for (int k = 0; k < 4; ++k) {
                        const int u = adjacent(t, k);
                        if (u < 0 || (dirty(u) && u < t)) continue;
                        const int far = tet_mesh::far_vertex(m, t, u);
                        if (far < 0 || !ep::in_sphere(point, v, o, far)) continue;
                        const int at = tet_mesh::claim(&counters(counter::work), 1);
                        if (at < work_capacity) {
                            work(at, 0) = t;
                            work(at, 1) = k;
                        }
                    }
                });
            if (ahead) break;
            read();
            again = host(counter::work) > work_capacity;
            if (again) {
                grow(host(counter::work), 0);
                Kokkos::deep_copy(exec, Kokkos::subview(counters, counter::stuck), stuck_before);
            }
        }
        // Unlist them; the apply step lists the next round's.
        Kokkos::parallel_for(
            "tet_repair::unlist", Range(exec, 0, dirty_count), KOKKOS_LAMBDA(int i) { dirty(listed(i)) = 0; });
        Kokkos::deep_copy(exec, Kokkos::subview(counters, std::make_pair(counter::listed, counter::created)), 0);
        // A triangle stuck earlier may have become flippable since: look at the stuck ones again while that still makes
        // progress. Returns whether the repair is finished.
        auto retry = [&] {
            const long long flips = stats.flips23 + stats.flips32;
            finished = host(counter::stuck) == 0 || flips == flips_at_last_retry;
            if (finished) return true;
            flips_at_last_retry = flips;
            const int listed_stuck = Kokkos::min(host(counter::stuck), static_cast<int>(stuck.extent(0)));
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
        // Ahead: plan and apply over the bound; their threads past the count return.
        const int bound = ahead ? 4 * dirty_count : host(counter::work);
        if (bound == 0) {
            if (retry()) break;
            continue;
        }
        const unsigned long long token = (s.round + 1) << 32;

        // 2-4. Plan, vote, apply; losers stay dirty for the next round.
        if (!ahead) grow(0, host(counter::stuck) + bound);  // at most one stuck entry per plan
        Kokkos::parallel_for(
            "tet_repair::plan", Range(exec, 0, bound), KOKKOS_LAMBDA(int i) {
                if (i >= counters(counter::work)) return;
                const unsigned long long vote = token | detail::priority(i);
                const int t = work(i, 0), k = work(i, 1);
                detail::Plan p;
                const int u = adjacent(t, k);
                if (alive(t) && u >= 0 && alive(u)) p = detail::plan(m, point, t, k, false);
                // The first valid flip nearly always closes its cavity; else the first of the others that does
                // (checking each inside one plan() made steps 2 % slower).
                if (p.kind > 0 && !closes(m, p.old, p.olds, p.fresh, p.freshes)) p = detail::plan(m, point, t, k, true);
                plans(i) = detail::pack(p);
                if (p.kind < 0) detail::list_stuck(counters, stuck, t, p.kind == detail::Plan::rejected ? rejected : k);
                if (p.kind <= 0) return;  // never votes, so it cannot block other flips
                for (int r = 0; r < p.olds; ++r) {
                    const int o = p.old[r];
                    Kokkos::atomic_max(&lock(o), vote);
                    for (int j = 0; j < 4; ++j)
                        if (adjacent(o, j) >= 0) Kokkos::atomic_max(&lock(adjacent(o, j)), vote);
                }
            });

        const int free_top = s.free_count;
        const int count = s.count;
        Kokkos::parallel_for(
            "tet_repair::apply", Range(exec, 0, bound), KOKKOS_LAMBDA(int i) {
                if (i >= counters(counter::work)) return;
                const unsigned long long vote = token | detail::priority(i);
                const detail::Plan p = detail::unpack(plans(i));
                if (p.kind <= 0) return;
                bool won = true;
                for (int r = 0; r < p.olds && won; ++r) {
                    const int o = p.old[r];
                    won = lock(o) == vote;
                    for (int j = 0; j < 4 && won; ++j)
                        if (adjacent(o, j) >= 0) won = lock(adjacent(o, j)) == vote;
                }
                // Slots: reuse the old ones, then pop free slots or grow; a flip with no room retries next round.
                int slot[3];
                for (int r = 0; r < p.freshes && won; ++r) {
                    if (r < p.olds) {
                        slot[r] = p.old[r];
                    } else {
                        const int popped = Kokkos::atomic_fetch_add(&counters(counter::pops), 1);
                        slot[r] = popped < free_top ? free_slots(free_top - 1 - popped)
                                                    : count + Kokkos::atomic_fetch_add(&counters(counter::growth), 1);
                        won = slot[r] < capacity;
                    }
                }
                if (!won) {
                    detail::mark(st, p.old[0]);
                    return;
                }
                // Outer faces of the old tetrahedra (six: the plan closes).
                int outer_key[8][3], outer_tet[8];
                const int outers = outer_faces(m, p.old, p.olds, outer_key, outer_tet);
                Kokkos::atomic_inc(&counters(p.kind == detail::Plan::flip23 ? counter::flips23 : counter::flips32));
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
                if (p.kind == detail::Plan::flip23) {
                    kept_edge[kept_edges][0] = p.fresh[0][0];
                    kept_edge[kept_edges][1] = p.fresh[0][1];
                    ++kept_edges;
                }
                for (int r = p.freshes; r < p.olds; ++r) {
                    alive(p.old[r]) = 0;
                    freed(tet_mesh::claim(&counters(counter::freed), 1)) = p.old[r];
                }
                for (int r = 0; r < p.freshes; ++r) {
                    int v[4] = {p.fresh[r][0], p.fresh[r][1], p.fresh[r][2], p.fresh[r][3]};
                    if (ep::orient(point, v) < 0) Kokkos::kokkos_swap(v[2], v[3]);
                    for (int c = 0; c < 4; ++c) {
                        vertex(slot[r], c) = v[c];
                        adjacent(slot[r], c) = -1;
                    }
                    alive(slot[r]) = 1;
                    st.quiet(slot[r]) = 0;
                }
                // Connect faces: to the outside through the old outer faces, else to each other.
                for (int r = 0; r < p.freshes; ++r) {
                    const int v[4] = {vertex(slot[r], 0), vertex(slot[r], 1), vertex(slot[r], 2), vertex(slot[r], 3)};
                    for (int c = 0; c < 4; ++c) {
                        int key[3];
                        tet_mesh::sorted_face(v, c, key);
                        for (int q = 0; q < outers; ++q)
                            if (tet_mesh::same_face(key, outer_key[q])) {
                                const int w = outer_tet[q];
                                adjacent(slot[r], c) = w;
                                if (w >= 0) {
                                    adjacent(w, tet_mesh::opposite(m, w, key)) = slot[r];
                                    st.quiet(w) = 0;  // its neighbour changed
                                }
                            }
                        for (int s2 = 0; s2 < p.freshes; ++s2) {
                            if (s2 == r) continue;
                            for (int c2 = 0; c2 < 4; ++c2) {
                                int other[3];
                                tet_mesh::sorted_face(p.fresh[s2], c2, other);
                                if (tet_mesh::same_face(key, other)) adjacent(slot[r], c) = slot[s2];
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
                            edges(slot[r]) |= static_cast<std::uint8_t>(1 << tet_mesh::edge_index(pa, pb));
                            break;
                        }
                    }
                }
                // List the new tetrahedra: free-list slots always, reused ones unless already touched. A round's flips share none.
                int listing[3], listings = 0;
                for (int r = 0; r < p.freshes; ++r)
                    if (r >= p.olds || !st.touched(slot[r])) listing[listings++] = slot[r];
                const int listed_at = tet_mesh::claim(&counters(counter::created), listings);
                for (int q = 0; q < listings; ++q)
                    if (listed_at + q < static_cast<int>(st.touched_list.extent(0)))
                        st.touched_list(listed_at + q) = listing[q];
                for (int r = 0; r < p.olds; ++r) st.touched(p.old[r]) = 1;
                for (int r = 0; r < p.freshes; ++r) {
                    st.touched(slot[r]) = 1;
                    detail::mark(st, slot[r]);
                }
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
        s.free_count = free_top - popped;
        s.count = Kokkos::min(count + host(counter::growth), capacity);
        const int freed_count = host(counter::freed);
        const int base = s.free_count;
        // The free list holds only the slots 3->2 flips freed: sized to its use, not to the tetrahedra.
        if (base + freed_count > static_cast<int>(s.free_slots.extent(0)))
            Kokkos::resize(s.free_slots, base + freed_count + (base + freed_count) / 4);
        const auto to_free = s.free_slots;
        Kokkos::parallel_for(
            "tet_repair::release", Range(exec, 0, freed_count), KOKKOS_LAMBDA(int i) { to_free(base + i) = freed(i); });
        s.free_count += freed_count;
        stats.flips23 += host(counter::flips23);
        stats.flips32 += host(counter::flips32);
        // Flips that found no slot retry next round with 1/8 more room.
        if (count + host(counter::growth) > capacity) reserve(m, s, capacity + Kokkos::max(capacity / 8, count + host(counter::growth) - capacity));
    }
    stats.unfinished = !finished;
#if defined(VORONOI_MEASURE)
    s.checked = stats.checked;
#endif
    stats.stuck = Kokkos::min(host(counter::stuck), static_cast<int>(s.stuck.extent(0)));
    s.touched_entries = host(counter::created) <= static_cast<int>(s.touched_list.extent(0)) ? host(counter::created) : -1;
    // The created tetrahedra did not fit their list (this step updates all faces): room for 1/4 more next time.
    if (s.touched_entries < 0) Kokkos::resize(s.touched_list, host(counter::created) + host(counter::created) / 4);
    // Room for the tetrahedra a host fallback may add.
    const int spare = s.count / 64 + 1024;
    if (stats.stuck > 0 && s.count + spare > static_cast<int>(s.alive.extent(0))) reserve(m, s, s.count + spare);
    return stats;
}

}  // namespace tet_repair
