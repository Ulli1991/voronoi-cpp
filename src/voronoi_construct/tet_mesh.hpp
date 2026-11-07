#pragma once

#include "exact_predicates.hpp"
#include "shell_voronoi.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// Stage 5 (geometry) plus the mesh data structure of stages 3-4: built by tet_start, edited by tet_repair/tet_flips.
// Mesh::update/update_cells call circumcentres -> update_face_rings -> faces -> cells (or cells_incremental) -> gradient_weights.
// Per step: one circumcentre per new tetrahedron, face geometry, per-cell lists; repair only where Delaunay fails.
namespace tet_mesh {

using shell_voronoi::Vec2;
using shell_voronoi::Vec3;
using shell_voronoi::VecN;

// One tetrahedron's entries are contiguous (row-major on every backend): a random access touches one memory sector.
template <class MemorySpace>
using Quads = Kokkos::View<int* [4], Kokkos::LayoutRight, MemorySpace>;
// Positions or centres padded to 32 bytes (one memory sector per point).
template <class MemorySpace>
using Points4 = Kokkos::View<double* [4], Kokkos::LayoutRight, MemorySpace>;

template <class MemorySpace>
struct Mesh {
    Quads<MemorySpace> vertex;     // particles
    Quads<MemorySpace> adjacent;   // tetrahedron across the face opposite vertex k, -1 if none
    Points4<MemorySpace> centre;   // circumcentre, unwrapped near vertex 0, and squared radius
    // Owned faces (owner < neighbour); face f's ring is ring[ring_start(f), ring_start(f) + ring_length(f)).
    Kokkos::View<int*, MemorySpace> owner, neighbour, ring_start, ring_length, ring;
    // Bit k: this tetrahedron represents its edge k (see edge_ends), the Voronoi face of that pair.
    // Every edge has exactly one representative, so a scan over the bits numbers the faces.
    Kokkos::View<std::uint8_t*, MemorySpace> edges;
};

// One float row per face (32 bytes): area vector (owner to neighbour) in [0, D), centroid (relative to the owner)
// in [3, 3 + D), owner and neighbour (int bits) in 6 and 7; written only when allocated.
template <class MemorySpace>
using FaceRows = Kokkos::View<float* [8], Kokkos::LayoutRight, MemorySpace>;

// Face area vectors (owner to neighbour), centroids (relative to the owner) and one cell volume per particle,
// in D = 2 or 3 dimensions; rows are contiguous (LayoutRight on every backend).
template <class MemorySpace, int D = 3>
struct Geometry {
    using Areas = Kokkos::View<double* [D], Kokkos::LayoutRight, MemorySpace>;
    Areas area, centroid;
    Kokkos::View<double*, MemorySpace> volume;
    FaceRows<MemorySpace> rows;  // optional float copy of area and centroid
};

// The corners of local edge k: (0,1) (0,2) (0,3) (1,2) (1,3) (2,3).
KOKKOS_INLINE_FUNCTION void edge_ends(int k, int& a, int& b) {
    a = k < 3 ? 0 : k < 5 ? 1 : 2;
    b = k < 3 ? k + 1 : k < 5 ? k - 1 : 3;
}
KOKKOS_INLINE_FUNCTION int edge_index(int a, int b) {  // corners a != b
    if (a > b) Kokkos::kokkos_swap(a, b);
    return a == 0 ? b - 1 : a == 1 ? b + 1 : 5;
}

// Topology queries shared by the device repair and the host fallback; m has vertex(t, k) and adjacent(t, k).

// Corner of tetrahedron t holding particle v, or -1.
template <class M>
KOKKOS_INLINE_FUNCTION int position(const M& m, int t, int v) {
    for (int k = 0; k < 4; ++k)
        if (m.vertex(t, k) == v) return k;
    return -1;
}

// The particle of u that t lacks, or -1. Used by: tet_repair::plan/repair, tet_flips::detail::fails.
template <class M>
KOKKOS_INLINE_FUNCTION int far_vertex(const M& m, int t, int u) {
    for (int k = 0; k < 4; ++k)
        if (position(m, t, m.vertex(u, k)) < 0) return m.vertex(u, k);
    return -1;
}

// The face of tetrahedron v opposite corner k, its particles sorted. Used by: tet_repair (closes), tet_flips.
KOKKOS_INLINE_FUNCTION void sorted_face(const int* v, int k, int* f) {
    for (int a = 0, b = 0; a < 4; ++a)
        if (a != k) f[b++] = v[a];
    if (f[0] > f[1]) Kokkos::kokkos_swap(f[0], f[1]);
    if (f[1] > f[2]) Kokkos::kokkos_swap(f[1], f[2]);
    if (f[0] > f[1]) Kokkos::kokkos_swap(f[0], f[1]);
}

KOKKOS_INLINE_FUNCTION bool same_face(const int* a, const int* b) { return a[0] == b[0] && a[1] == b[1] && a[2] == b[2]; }

// The corner of tetrahedron t opposite its face f (sorted particles). Used by: tet_flips::finish.
template <class M>
KOKKOS_INLINE_FUNCTION int opposite(const M& m, int t, const int* f) {
    for (int k = 0; k < 3; ++k) {
        const int v = m.vertex(t, k);
        if (v != f[0] && v != f[1] && v != f[2]) return k;
    }
    return 3;
}

// Bound on walks around an edge, not a limit on faces: on a checked mesh a ring closes within n - 2 tetrahedra.
// It only stops a walk on broken adjacency, whose ring then fails to fit the pool (Kokkos::abort).
constexpr int max_ring_length = 1 << 30;

// The tetrahedron after `current` around edge i-j (particles of current),
// away from `before`; -1 at an open face.
template <class M>
KOKKOS_INLINE_FUNCTION int next_around_edge(const M& m, int current, int before, int i, int j) {
    for (int c = 0; c < 4; ++c) {
        const int v = m.vertex(current, c);
        if (v == i || v == j) continue;
        const int w = m.adjacent(current, c);
        if (w >= 0 && w != before) return w;
    }
    return -1;
}

// Tetrahedra around edge i-j from t, in order and t first, written to ring[]
// unless it is null. Returns the number of steps (the ring length when it
// closes).
template <class M>
KOKKOS_INLINE_FUNCTION int walk_edge(const M& m, int t, int i, int j, int* ring = nullptr) {
    int current = t, before = -1, steps = 0;
    do {
        if (ring) ring[steps] = current;
        const int next = next_around_edge(m, current, before, i, j);
        if (next < 0 || ++steps > max_ring_length) break;
        before = current;
        current = next;
    } while (current != t);
    return steps;
}

// The lowest-numbered tetrahedron around edge i-j of t.
template <class M>
KOKKOS_INLINE_FUNCTION int lowest_around_edge(const M& m, int t, int i, int j) {
    int lowest = t, current = t, before = -1;
    for (int steps = 0; steps <= max_ring_length; ++steps) {
        lowest = Kokkos::min(lowest, current);
        const int next = next_around_edge(m, current, before, i, j);
        if (next < 0 || next == t) break;
        before = current;
        current = next;
    }
    return lowest;
}

// The ring of edge i-j from t written to ring[] in canonical order: from the tetrahedron whose other two particle ids
// (smaller first) are least, towards the smaller of its two ring neighbours, so face sums do not depend on which
// tetrahedron represents the face. Open rings: walk_edge from t. Ring is an int* or gives room via ring(steps).
// Returns the number of steps, as walk_edge.
template <class M, class Ring>
KOKKOS_INLINE_FUNCTION int walk_edge_canonical(const M& m, int t, int i, int j, Ring ring) {
    auto key = [&](int u) {  // the other two particles of u, packed
        int lo = -1, hi = -1;
        for (int c = 0; c < 4; ++c) {
            const int v = m.vertex(u, c);
            if (v == i || v == j) continue;
            if (lo < 0) lo = v; else hi = v;
        }
        if (lo > hi) Kokkos::kokkos_swap(lo, hi);
        return (static_cast<long long>(lo) << 32) | static_cast<unsigned>(hi);
    };
    // First walk: the least key, its predecessor and successor on the ring.
    int current = t, before = -1, steps = 0, best = -1, best_before = -1, best_after = -1, last = -1;
    long long best_key = 0, before_key = 0, after_key = 0, last_key = 0, first_key = 0;
    bool closed = false;
    for (;;) {
        const long long k = key(current);
        if (steps == 0) first_key = k;
        if (best >= 0 && best_after < 0) {
            best_after = current;
            after_key = k;
        }
        if (best < 0 || k < best_key) {
            best = current;
            best_key = k;
            best_before = before;
            before_key = last_key;
            best_after = -1;
        }
        const int next = next_around_edge(m, current, before, i, j);
        ++steps;
        if (next < 0 || steps > max_ring_length) break;
        last = current;
        last_key = k;
        before = current;
        current = next;
        if (current == t) {
            closed = true;
            break;
        }
    }
    if constexpr (!std::is_pointer_v<Ring>) {  // Ring: place(steps) gives the room
        if (!closed || steps < 3) {
            const int length = walk_edge(m, t, i, j);
            int* room = ring(length + 1);
            if (room) walk_edge(m, t, i, j, room);
            return length;
        }
    } else {
        if (!closed || steps < 3) return walk_edge(m, t, i, j, ring);
    }
    if (best_before < 0) {  // best is t: its predecessor is the last
        best_before = last;
        before_key = last_key;
    }
    if (best_after < 0) {  // best is the last: its successor is t
        best_after = t;
        after_key = first_key;
    }
    // Second walk from best, away from the larger neighbour.
    current = best;
    before = after_key < before_key ? best_before : best_after;
    int* out;
    if constexpr (std::is_pointer_v<Ring>)
        out = ring;
    else
        out = ring(steps);
    if (!out) return steps;
    for (int s = 0; s < steps; ++s) {
        out[s] = current;
        const int next = next_around_edge(m, current, before, i, j);
        before = current;
        current = next;
    }
    return steps;
}

// Sorts the four corners v by particle id; returns the permutation parity (+1 even, -1 odd). Used by: tet_repair::move_screened.
// Circumcentres from sorted corners are the same bits whatever order the flips built the tetrahedron in.
KOKKOS_INLINE_FUNCTION int sort_corners(int* v) {
    int parity = 1;
    for (int a = 1; a < 4; ++a)
        for (int b = a; b > 0 && v[b - 1] > v[b]; --b) {
            const int t = v[b - 1];
            v[b - 1] = v[b];
            v[b] = t;
            parity = -parity;
        }
    return parity;
}

using Box = exact_predicates::Box;

// A coordinate difference in a periodic box, to the nearest image; multiplies by 1 / side, so it differs from a
// division only if d is within an ulp of half a side.
KOKKOS_INLINE_FUNCTION double wrap(double d, double side, double inverse) {
    return d - side * Kokkos::nearbyint(d * inverse);
}
KOKKOS_INLINE_FUNCTION Vec3 wrap(Vec3 d, const Box& box) {
    return {wrap(d.x, box.side[0], box.inverse[0]), wrap(d.y, box.side[1], box.inverse[1]),
            wrap(d.z, box.side[2], box.inverse[2])};
}
KOKKOS_INLINE_FUNCTION Vec2 wrap(Vec2 d, const exact_predicates::BoxN<2>& box) {
    return {wrap(d.x, box.side[0], box.inverse[0]), wrap(d.y, box.side[1], box.inverse[1])};
}

// Row i of positions or centres x, in D dimensions.
template <int D = 3, class X>
KOKKOS_INLINE_FUNCTION VecN<D> row(const X& x, int i) {
    if constexpr (D == 3)
        return {x(i, 0), x(i, 1), x(i, 2)};
    else
        return {x(i, 0), x(i, 1)};
}
// Stores v as row i of x.
template <class X>
KOKKOS_INLINE_FUNCTION void store(const X& x, int i, Vec3 v) {
    x(i, 0) = v.x;
    x(i, 1) = v.y;
    x(i, 2) = v.z;
}
template <class X>
KOKKOS_INLINE_FUNCTION void store(const X& x, int i, Vec2 v) {
    x(i, 0) = v.x;
    x(i, 1) = v.y;
}

namespace detail {
// The point equidistant from the origin and a, b, c.
KOKKOS_INLINE_FUNCTION Vec3 circumcentre(Vec3 a, Vec3 b, Vec3 c) {
    const Vec3 bc = cross(b, c), ca = cross(c, a), ab = cross(a, b);
    const double scale = 0.5 / dot(a, bc);
    return (bc * dot(a, a) + ca * dot(b, b) + ab * dot(c, c)) * scale;
}

// Whether a tetrahedron with edge vectors d[0..2] from one corner is too flat for that formula: sets volume6 (six times
// its volume) and shape (product of the edge lengths) and returns true if volume6 is tiny against shape.
KOKKOS_INLINE_FUNCTION bool sliver(const Vec3* d, double& volume6, double& shape) {
    volume6 = dot(d[0], cross(d[1], d[2]));
    shape = Kokkos::sqrt(dot(d[0], d[0]) * dot(d[1], d[1]) * dot(d[2], d[2]));
    return !(Kokkos::abs(volume6) > 1e-4 * shape);
}
}  // namespace detail

// Stage 3: The tetrahedra and their adjacency from the face rings of a clipping build (Cells::with_rings, cells in slot order,
// every cell ok), without hashing: tetrahedron (i, j, q_m, q_m+1) is emitted only by the face of its two smallest
// particles, and each triangle's two tetrahedra are adjacent in the ring of its two smallest particles' face.
// Faces in CSR come back in face_start (cell t owns [face_start(t), face_start(t + 1))). Returns the number of ring
// slots or adjacencies not resolved (0 for a consistent build); edges without a face are left to represent_dropped.
// Used by: tet_start::build (every round).
template <class ExecSpace>
int assemble(const ExecSpace& exec, const shell_voronoi::Cells<typename ExecSpace::memory_space>& cells,
             Mesh<typename ExecSpace::memory_space>& mesh,
             Kokkos::View<int*, typename ExecSpace::memory_space>& face_start) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const int n = static_cast<int>(cells.owned_count.extent(0));

    face_start = Kokkos::View<int*, Memory>("face_start", n + 1);
    int faces = 0;
    const auto starts = face_start;
    Kokkos::parallel_scan(
        "tet_mesh::face_start", Range(exec, 0, n),
        KOKKOS_LAMBDA(int t, int& sum, bool last) {
            if (last) starts(t) = sum;
            sum += cells.owned_count(t);
            if (last && t == n - 1) starts(n) = sum;
        },
        faces);
    Kokkos::View<int*, Memory> owner("face_owner", faces), neighbour("face_neighbour", faces),
        length("face_length", faces);
    const auto rings = cells.rings;
    Kokkos::parallel_for(
        "tet_mesh::face_lengths", Range(exec, 0, n), KOKKOS_LAMBDA(int t) {
            long long at = cells.ring_start(t);
            for (int f = 0; f < cells.owned_count(t); ++f) {
                const int g = starts(t) + f, k = rings(at + 1);
                owner(g) = t;
                neighbour(g) = rings(at);
                length(g) = k;
                at += 2 + k;
            }
        });
    auto face_of = KOKKOS_LAMBDA(int a, int b) {  // a < b
        for (int g = starts(a); g < starts(a + 1); ++g)
            if (neighbour(g) == b) return g;
        return -1;
    };

    Kokkos::View<int*, Memory> ring_start("face_ring_start", faces + 1), first_tet("face_first_tet", faces + 1);
    int slots = 0, tets = 0;
    Kokkos::parallel_scan(
        "tet_mesh::ring_start", Range(exec, 0, faces),
        KOKKOS_LAMBDA(int g, int& sum, bool last) {
            if (last) ring_start(g) = sum;
            sum += length(g);
            if (last && g == faces - 1) ring_start(faces) = sum;
        },
        slots);
    // The rings' particles, contiguous per face in face order.
    Kokkos::View<int*, Memory> particles(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("face_particles")),
                                         slots);
    Kokkos::parallel_for(
        "tet_mesh::face_particles", Range(exec, 0, n), KOKKOS_LAMBDA(int t) {
            long long at = cells.ring_start(t);
            for (int f = 0; f < cells.owned_count(t); ++f) {
                const int g = starts(t) + f, k = length(g);
                for (int m = 0; m < k; ++m) particles(ring_start(g) + m) = rings(at + 2 + m);
                at += 2 + k;
            }
        });
    auto ring_of = KOKKOS_LAMBDA(int g, int m) { return particles(ring_start(g) + m % length(g)); };
    // A face emits the slots whose tetrahedron has the face's particles as its two smallest.
    auto emits = KOKKOS_LAMBDA(int g, int m) {
        const int j = neighbour(g);
        return j < ring_of(g, m) && j < ring_of(g, m + 1);
    };
    Kokkos::parallel_scan(
        "tet_mesh::first_tet", Range(exec, 0, faces),
        KOKKOS_LAMBDA(int g, int& sum, bool last) {
            if (last) first_tet(g) = sum;
            for (int m = 0; m < length(g); ++m) sum += emits(g, m);
            if (last && g == faces - 1) first_tet(faces) = sum;
        },
        tets);
    mesh.vertex = Quads<Memory>("tet_vertex", tets);
    mesh.adjacent = Quads<Memory>("tet_adjacent", tets);
    mesh.centre = Points4<Memory>("tet_centre", tets);
    mesh.edges = Kokkos::View<std::uint8_t*, Memory>("tet_edges", tets);
    mesh.owner = owner;
    mesh.neighbour = neighbour;
    mesh.ring_start = Kokkos::subview(ring_start, std::make_pair(0, faces));
    mesh.ring_length = length;
    mesh.ring = Kokkos::View<int*, Memory>("face_ring", slots);
    const auto vertex = mesh.vertex;
    const auto adjacent = mesh.adjacent;
    const auto ring = mesh.ring;

    // Emit, and name every ring slot by its tetrahedron.
    int unresolved = 0;
    Kokkos::parallel_reduce(
        "tet_mesh::emit", Range(exec, 0, faces),
        KOKKOS_LAMBDA(int g, int& missing) {
            const int i = owner(g), j = neighbour(g);
            for (int m = 0, e = first_tet(g); m < length(g); ++m) {
                const int p = ring_of(g, m), q = ring_of(g, m + 1);
                if (emits(g, m)) {
                    vertex(e, 0) = i;
                    vertex(e, 1) = j;
                    vertex(e, 2) = p;
                    vertex(e, 3) = q;
                    ring(ring_start(g) + m) = e++;
                    continue;
                }
                // The two smallest of {i, j, p, q} own the face that emits it.
                int s[4] = {i, j, p, q};
                for (int a = 1; a < 4; ++a)
                    for (int b = a; b > 0 && s[b - 1] > s[b]; --b) Kokkos::kokkos_swap(s[b - 1], s[b]);
                const int h = face_of(s[0], s[1]);
                int id = -1;
                for (int r = 0, e2 = h < 0 ? 0 : first_tet(h); h >= 0 && r < length(h); ++r) {
                    const bool emitted = emits(h, r);
                    const int u = ring_of(h, r), w = ring_of(h, r + 1);
                    if (emitted && ((u == s[2] && w == s[3]) || (u == s[3] && w == s[2]))) {
                        id = e2;
                        break;
                    }
                    e2 += emitted;
                }
                ring(ring_start(g) + m) = id;
                missing += id < 0;
            }
        },
        unresolved);

    // Neighbours: across a triangle through edge i-j, the next tetrahedron in that ring; across the other two
    // triangles, look up their own rings.
    int open = 0;
    Kokkos::parallel_reduce(
        "tet_mesh::adjacency", Range(exec, 0, faces),
        KOKKOS_LAMBDA(int g, int& missing) {
            const int k = length(g), base = ring_start(g);
            for (int m = 0; m < k; ++m) {
                if (!emits(g, m)) continue;
                const int e = ring(base + m);
                adjacent(e, 2) = ring(base + (m + 1) % k);
                adjacent(e, 3) = ring(base + (m + k - 1) % k);
                const int v[4] = {vertex(e, 0), vertex(e, 1), vertex(e, 2), vertex(e, 3)};
                for (int drop = 0; drop < 2; ++drop) {
                    int tri[3];
                    sorted_face(v, drop, tri);
                    const int h = face_of(tri[0], tri[1]);
                    int other = -1;
                    for (int r = 0; h >= 0 && r < length(h); ++r)
                        if (ring_of(h, r) == tri[2]) {
                            // Tetrahedra at slots r - 1 and r share this triangle.
                            const int before = ring(ring_start(h) + (r + length(h) - 1) % length(h));
                            const int after = ring(ring_start(h) + r);
                            other = before == e ? after : before;
                            break;
                        }
                    adjacent(e, drop) = other;
                    missing += other < 0;
                }
                missing += adjacent(e, 2) < 0 || adjacent(e, 3) < 0;
            }
        },
        open);
    // Each face's first ring tetrahedron represents its edge (bits gathered
    // with 32-bit atomics, then narrowed to bytes).
    Kokkos::View<int*, Memory> bits("tet_edge_bits", tets);
    Kokkos::parallel_for(
        "tet_mesh::represent", Range(exec, 0, faces), KOKKOS_LAMBDA(int g) {
            const int e = ring(ring_start(g));
            if (e < 0) return;
            int a = -1, b = -1;
            for (int k = 0; k < 4; ++k) {
                if (vertex(e, k) == owner(g)) a = k;
                if (vertex(e, k) == neighbour(g)) b = k;
            }
            if (a >= 0 && b >= 0) Kokkos::atomic_or(&bits(e), 1 << edge_index(a, b));
        });
    const auto edges = mesh.edges;
    Kokkos::parallel_for(
        "tet_mesh::edge_bytes", Range(exec, 0, tets),
        KOKKOS_LAMBDA(int e) { edges(e) = static_cast<std::uint8_t>(bits(e)); });
    return unresolved + open;
}

// After assemble on a consistent mesh: an edge without a face (the owner's cell dropped a rounding-level face its
// neighbour kept) is represented by the lowest-numbered tetrahedron around it, so every edge has one representative.
// Used by: tet_start::build, once the start passes check.
template <class ExecSpace>
void represent_dropped(const ExecSpace& exec, const Mesh<typename ExecSpace::memory_space>& mesh,
                       const Kokkos::View<int*, typename ExecSpace::memory_space>& face_start) {
    const auto vertex = mesh.vertex;
    const auto neighbour = mesh.neighbour;
    const auto ring_start = mesh.ring_start;
    const auto ring = mesh.ring;
    const auto edges = mesh.edges;
    Kokkos::parallel_for(
        "tet_mesh::represent_dropped",
        Kokkos::RangePolicy<ExecSpace>(exec, 0, static_cast<int>(vertex.extent(0))), KOKKOS_LAMBDA(int e) {
            for (int k = 0; k < 6; ++k) {
                int ka, kb;
                edge_ends(k, ka, kb);
                const int a = vertex(e, ka), b = vertex(e, kb), low = Kokkos::min(a, b), high = Kokkos::max(a, b);
                int first = -1;
                for (int g = face_start(low); g < face_start(low + 1); ++g)
                    if (neighbour(g) == high) first = ring(ring_start(g));
                if (first >= 0 && position(mesh, first, a) >= 0 && position(mesh, first, b) >= 0) continue;
                if (lowest_around_edge(mesh, e, a, b) == e) edges(e) |= static_cast<std::uint8_t>(1 << k);
            }
        });
    exec.fence();  // the caller drops these views next
}

// assemble, then represent_dropped if it resolved everything; returns assemble's unresolved count.
template <class ExecSpace>
int from_rings(const ExecSpace& exec, const shell_voronoi::Cells<typename ExecSpace::memory_space>& cells,
               Mesh<typename ExecSpace::memory_space>& mesh) {
    Kokkos::View<int*, typename ExecSpace::memory_space> face_start;
    const int unresolved = assemble(exec, cells, mesh, face_start);
    if (unresolved == 0) represent_dropped(exec, mesh, face_start);
    return unresolved;
}

// Face rings kept from step to step. Face start(t) + rank is the rank-th edge (bit order) tetrahedron t represents;
// its ring is pool[ring_start(f), ring_start(f) + ring_length(f)). Two pools: one in use, one spare to compact into;
// the spare is the mesh's scratch memory between updates and holds the temporaries of update_face_rings/update_cells.
// Used by: Mesh<3>::rebuild (make_face_rings), Mesh::geometry (update_face_rings), Mesh::scratch.
template <class MemorySpace>
struct FaceRings {
    using Ints = Kokkos::View<int*, MemorySpace>;
    using Borrowed = Kokkos::View<int*, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Ints start, ring_start, ring_length, pool[2];
    Ints owner, neighbour;
    Borrowed dirty;                         // faces to walk this step, in the spare pool
    Kokkos::View<int, MemorySpace> listed;  // their number
    int faces = -1;       // -1: none yet, every face is walked
    int in_use = 0;       // pool[in_use] holds the rings
    int used = 0;         // its entries in use, superseded rings included
    int tets = 0;         // tetrahedra numbered by the last update
    int dirty_count = 0;  // entries of `dirty`
    // Whether the spare pool holds, from its start, the new number of every face of the last update but one (-1: gone),
    // `renumbered` faces.
    bool renumber_valid = false;
    int renumbered = 0;
};

template <class MemorySpace>
FaceRings<MemorySpace> make_face_rings(int tets, int faces, int entries) {
    using Ints = Kokkos::View<int*, MemorySpace>;
    FaceRings<MemorySpace> r;
    r.start = Ints("face_rings_start", tets + 1);
    r.ring_start = Ints("face_rings_ring_start", faces);
    r.ring_length = Ints("face_rings_ring_length", faces);
    for (int b = 0; b < 2; ++b)
        r.pool[b] = Ints(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("face_rings_pool")), entries);
    r.owner = Ints("face_rings_owner", faces);
    r.neighbour = Ints("face_rings_neighbour", faces);
    r.listed = Kokkos::View<int, MemorySpace>("face_rings_listed");
    return r;
}

// The face of edge i-j (particles of tetrahedron t) in the numbering `start`, and its representative, by walking around
// the edge. Returns -1 if the walk does not reach it, or first meets a touched tetrahedron numbered below t
// (touched(u) != 0), which finds the face itself. Used by: number_faces, update_face_rings.
template <class M, class Start, class Touched>
KOKKOS_INLINE_FUNCTION int edge_face(const M& m, const Start& start, const Touched& touched, int t, int i, int j,
                                     int& representative) {
    for (int current = t, before = -1, steps = 0; steps < max_ring_length; ++steps) {
        const int bits = m.edges(current);
        const int a = position(m, current, i), b = position(m, current, j);
        if (a < 0 || b < 0) return -1;
        const int bit = 1 << edge_index(a, b);
        if (bits & bit) {
            representative = current;
            return start(current) + Kokkos::Experimental::popcount_builtin(static_cast<unsigned>(bits & (bit - 1)));
        }
        if (current < t && touched(current)) return -1;
        const int next = next_around_edge(m, current, before, i, j);
        if (next < 0 || next == t) return -1;
        before = current;
        current = next;
    }
    return -1;
}

namespace detail {
// The last step's numbers (start of each tetrahedron, ring of each face) and the renumbering table to fill: copies in the spare pool.
template <class MemorySpace>
struct OldFaces {
    using Borrowed = typename FaceRings<MemorySpace>::Borrowed;
    Borrowed renumber, start, ring_start, ring_length;
};

// Number the faces of update_face_rings, with owner and neighbour. A face represented by an untouched tetrahedron takes
// its old ring, one represented by a touched tetrahedron gets ring_start 0 (marked later); the first call walks every face.
// Writes at most the face capacity; returns the number of faces. Used by: update_face_rings.
template <class ExecSpace>
int number_faces(const ExecSpace& exec, const Mesh<typename ExecSpace::memory_space>& mesh, int count,
                 const Kokkos::View<char*, typename ExecSpace::memory_space>& touched,
                 const FaceRings<typename ExecSpace::memory_space>& rings,
                 const OldFaces<typename ExecSpace::memory_space>& old) {
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const bool all = rings.faces < 0;
    const int old_tets = rings.tets;
    const auto edges = mesh.edges;
    const auto start = rings.start;
    const auto all_ring_start = rings.ring_start, all_ring_length = rings.ring_length;
    const auto owner = rings.owner, neighbour = rings.neighbour;
    const int face_capacity = static_cast<int>(owner.extent(0));
    int faces = 0;
    Kokkos::parallel_scan(
        "tet_mesh::face_numbers", Range(exec, 0, count),
        KOKKOS_LAMBDA(int t, int& sum, bool final) {
            int bits = edges(t);
            const int first = sum;
            sum += Kokkos::Experimental::popcount_builtin(static_cast<unsigned>(bits));
            if (!final) return;
            start(t) = first;
            if (t == count - 1) start(count) = sum;
            const bool kept = !all && !touched(t);
            // Faces of this tetrahedron's last contents with no number of their own.
            if (!all && !kept && t < old_tets)
                for (int g = old.start(t); g < old.start(t + 1); ++g) old.renumber(g) = -1;
            for (int f = first, rank = 0; bits && f < face_capacity; ++f, ++rank, bits &= bits - 1) {
                int ka, kb;
                edge_ends(Kokkos::Experimental::countr_zero_builtin(static_cast<unsigned>(bits)), ka, kb);
                int i = mesh.vertex(t, ka), j = mesh.vertex(t, kb);
                if (i > j) Kokkos::kokkos_swap(i, j);
                owner(f) = i;
                neighbour(f) = j;
                if (all) {
                    all_ring_start(f) = ~t;
                    all_ring_length(f) = walk_edge(mesh, t, i, j);
                } else if (kept) {
                    const int was = old.start(t) + rank;
                    all_ring_start(f) = old.ring_start(was);
                    all_ring_length(f) = old.ring_length(was);
                    old.renumber(was) = f;
                } else {
                    all_ring_start(f) = 0;
                }
            }
        },
        faces);
    return faces;
}
}  // namespace detail

// Renumber the faces after a repair and update their rings in place: a face keeps its ring unless a touched
// tetrahedron lies on it (`touched_list`; empty handle: all touched). Sets mesh.owner/neighbour/ring_*; returns the face count.
// Used by: Mesh<3>::geometry, after circumcentres.
template <class ExecSpace>
int update_face_rings(const ExecSpace& exec, Mesh<typename ExecSpace::memory_space>& mesh, int count,
                      const Kokkos::View<char*, typename ExecSpace::memory_space>& touched,
                      const Kokkos::View<const int*, typename ExecSpace::memory_space>& touched_list,
                      FaceRings<typename ExecSpace::memory_space>& rings) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    using Borrowed = typename FaceRings<Memory>::Borrowed;
    const bool all = rings.faces < 0;
    // These references would hold on to what is reallocated below.
    mesh.owner = mesh.neighbour = mesh.ring_start = mesh.ring_length = mesh.ring = {};
    // The spare pool: the renumbering table, then the last step's numbers.
    detail::OldFaces<Memory> old;
    if (!all) {
        const std::size_t table = static_cast<std::size_t>(rings.faces), tets = static_cast<std::size_t>(rings.tets) + 1;
        auto& spare = rings.pool[1 - rings.in_use];
        if (spare.extent(0) < 3 * table + tets)
            Kokkos::realloc(Kokkos::WithoutInitializing, spare, 3 * table + tets + table / 32);
        int* at = spare.data();
        old.renumber = Borrowed(at, table);
        old.start = Borrowed(at + table, tets);
        old.ring_start = Borrowed(at + table + tets, table);
        old.ring_length = Borrowed(at + 2 * table + tets, table);
        Kokkos::deep_copy(exec, old.start, Kokkos::subview(rings.start, std::make_pair(std::size_t(0), tets)));
        Kokkos::deep_copy(exec, old.ring_start, Kokkos::subview(rings.ring_start, std::make_pair(std::size_t(0), table)));
        Kokkos::deep_copy(exec, old.ring_length, Kokkos::subview(rings.ring_length, std::make_pair(std::size_t(0), table)));
    }
    if (static_cast<int>(rings.start.extent(0)) < count + 1)
        Kokkos::realloc(Kokkos::WithoutInitializing, rings.start, count + 1 + count / 32);
    rings.renumber_valid = !all;
    rings.renumbered = rings.faces;
    int faces = detail::number_faces(exec, mesh, count, touched, rings, old);
    if (faces > static_cast<int>(rings.owner.extent(0))) {
        const int capacity = faces + faces / 32;
        for (auto* v : {&rings.owner, &rings.neighbour, &rings.ring_start, &rings.ring_length})
            Kokkos::realloc(Kokkos::WithoutInitializing, *v, capacity);
        faces = detail::number_faces(exec, mesh, count, touched, rings, old);
    }
    const auto start = rings.start;
    const auto all_ring_start = rings.ring_start, all_ring_length = rings.ring_length;
    const auto owner = rings.owner, neighbour = rings.neighbour;
    const auto ring_start = Kokkos::subview(all_ring_start, std::make_pair(0, faces));
    const auto ring_length = Kokkos::subview(all_ring_length, std::make_pair(0, faces));
    const auto listed = rings.listed;

    // The dirty list follows the table, over the last numbers, done with.
    rings.dirty = all ? Borrowed() : Borrowed(rings.pool[1 - rings.in_use].data() + rings.faces, faces);
    const auto dirty = rings.dirty;

    // Dirty faces (a touched tetrahedron on their ring): ring_start ~representative, listed once in `dirty`.
    Kokkos::deep_copy(exec, listed, 0);
    int dirty_faces = 0;
    if (!all) {
        const bool from_list = touched_list.data() != nullptr;
        Kokkos::parallel_reduce(
            "tet_mesh::dirty_faces",
            Range(exec, 0, from_list ? static_cast<int>(touched_list.extent(0)) : count),
            KOKKOS_LAMBDA(int s, int& sum) {
                const int t = from_list ? touched_list(s) : s;
                if (!from_list && !touched(t)) return;
                int found[6], n = 0;
                for (int k = 0; k < 6; ++k) {
                    int ka, kb, representative = -1;
                    edge_ends(k, ka, kb);
                    const int f =
                        edge_face(mesh, start, touched, t, mesh.vertex(t, ka), mesh.vertex(t, kb), representative);
                    if (f >= 0 && Kokkos::atomic_exchange(&ring_start(f), ~representative) >= 0) found[n++] = f;
                }
                if (n == 0) return;
                const int at = Kokkos::atomic_fetch_add(&listed(), n);
                for (int q = 0; q < n; ++q) dirty(at + q) = found[q];
                sum += n;
            },
            dirty_faces);
    }
    // The dirty faces' rings, one kernel: walked, room claimed in the pool, walked again canonically; a ring without
    // room stays dirty for the compaction.
    int dirty_entries = 0;
    if (!all) {
        const auto pool = rings.pool[rings.in_use];
        const int base = rings.used, room = static_cast<int>(pool.extent(0));
        Kokkos::deep_copy(exec, listed, 0);
        Kokkos::parallel_reduce(
            "tet_mesh::append_rings", Range(exec, 0, dirty_faces),
            KOKKOS_LAMBDA(int s, int& sum) {
                const int f = dirty(s), representative = ~ring_start(f);
                int at = -1;
                auto place = [&](int steps) -> int* {
                    at = base + Kokkos::atomic_fetch_add(&listed(), steps);
                    return at + steps <= room ? &pool(at) : nullptr;
                };
                const int length = walk_edge_canonical(mesh, representative, owner(f), neighbour(f), place);
                ring_length(f) = length;
                sum += length;
                if (at + length <= room) ring_start(f) = at;
                // Faces without an old number: ~f in the list.
                if (touched(representative)) dirty(s) = ~f;
            },
            dirty_entries);
    }
    if (!all && rings.used + dirty_entries <= static_cast<int>(rings.pool[rings.in_use].extent(0))) {
        rings.used += dirty_entries;
    } else {
        // Rewrite every ring into the spare pool: copy the clean ones, walk the dirty ones.
        rings.renumber_valid = false;
        int entries = 0;
        Kokkos::parallel_reduce(
            "tet_mesh::ring_entries", Range(exec, 0, faces),
            KOKKOS_LAMBDA(int f, int& sum) { sum += ring_length(f); }, entries);
        {
            auto& spare = rings.pool[1 - rings.in_use];
            if (entries > static_cast<int>(spare.extent(0)))
                Kokkos::realloc(Kokkos::WithoutInitializing, spare, entries + entries / 4);
            const auto from = rings.pool[rings.in_use];
            const auto pool = spare;
            const int capacity = static_cast<int>(pool.extent(0));
            rings.in_use = 1 - rings.in_use;
            Kokkos::parallel_scan(
                "tet_mesh::compact_rings", Range(exec, 0, faces),
                KOKKOS_LAMBDA(int f, int& sum, bool final) {
                    const int begin = ring_start(f), length = ring_length(f);
                    if (final && sum + length > capacity) Kokkos::abort("update_face_rings: ring capacity exceeded");
                    if (final && begin < 0) {
                        walk_edge_canonical(mesh, ~begin, owner(f), neighbour(f), &pool(sum));
                        ring_start(f) = sum;
                    }
                    if (final && begin >= 0) {
                        for (int r = 0; r < length; ++r) pool(sum + r) = from(begin + r);
                        ring_start(f) = sum;
                    }
                    sum += length;
                },
                entries);
            rings.used = entries;
        }
        // The emptied pool is the next spare.
        auto& spare = rings.pool[1 - rings.in_use];
        const auto& pool = rings.pool[rings.in_use];
        if (spare.extent(0) < pool.extent(0)) {
            if (static_cast<std::size_t>(entries) + entries / 4 > spare.extent(0))
                Kokkos::realloc(Kokkos::WithoutInitializing, spare, std::min<std::size_t>(entries + entries / 2, pool.extent(0)));
            Kokkos::deep_copy(exec, Kokkos::subview(spare, std::make_pair(0, entries)),
                              Kokkos::subview(pool, std::make_pair(0, entries)));
            rings.in_use = 1 - rings.in_use;
        }
    }
    rings.faces = faces;
    rings.tets = count;
    rings.dirty_count = dirty_faces;
    mesh.owner = Kokkos::subview(owner, std::make_pair(0, faces));
    mesh.neighbour = Kokkos::subview(neighbour, std::make_pair(0, faces));
    mesh.ring_start = ring_start;
    mesh.ring_length = ring_length;
    mesh.ring = rings.pool[rings.in_use];
    return faces;
}

// The circumcentre of particles v[0..3] relative to x(v[0]), given the
// offsets d[k] = x(v[k + 1]) - x(v[0]) (nearest images).
template <class X>
KOKKOS_INLINE_FUNCTION Vec3 circumcentre(const X& x, const Box& box, const int* v, const Vec3* d) {
    double volume6, shape;
    const Vec3 c = detail::circumcentre(d[0], d[1], d[2]);
    if (!detail::sliver(d, volume6, shape)) return c;
    // Redo it on the snapped grid with exact integers, dividing once at the end.
    namespace ep = exact_predicates;
    ep::Offset g[3];
    for (int k = 0; k < 3; ++k)
        for (int a = 0; a < 3; ++a)
            g[k][a] = ep::delta(ep::snap(x(v[0], a), box, a), ep::snap(x(v[k + 1], a), box, a), box.period[a]);
    ep::i128 n[3];
    const ep::i128 det = ep::circumcentre(g[0], g[1], g[2], n);
    if (det == 0) return c;
    const double unit = box.step / (2 * double(det));
    return {double(n[0]) * unit, double(n[1]) * unit, double(n[2]) * unit};
}

// Operations rounded one by one (no fused multiply-add), so inlined into different kernels they give the same bits.
namespace rounded {
KOKKOS_INLINE_FUNCTION double mul(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dmul_rn(a, b);
#else
    const volatile double r = a * b;  // stored, so it cannot be fused into a later sum
    return r;
#endif
}
KOKKOS_INLINE_FUNCTION double add(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dadd_rn(a, b);
#else
    return a + b;
#endif
}
KOKKOS_INLINE_FUNCTION double sub(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dsub_rn(a, b);
#else
    return a - b;
#endif
}
KOKKOS_INLINE_FUNCTION Vec3 cross(Vec3 a, Vec3 b) {
    return {sub(mul(a.y, b.z), mul(a.z, b.y)), sub(mul(a.z, b.x), mul(a.x, b.z)), sub(mul(a.x, b.y), mul(a.y, b.x))};
}
KOKKOS_INLINE_FUNCTION double dot(Vec3 a, Vec3 b) { return add(add(mul(a.x, b.x), mul(a.y, b.y)), mul(a.z, b.z)); }
}  // namespace rounded

// The pieces of a stored circumcentre (Mesh::centre), each operation rounded on its own (`rounded`).

namespace R = rounded;

// Offsets d[k] = x(v[k + 1]) - x(v[0]) (nearest images) of corners v.
template <class X>
KOKKOS_INLINE_FUNCTION void rounded_offsets(const X& x, const Box& box, const int* v, Vec3* d) {
    double e[3][3];
    for (int k = 0; k < 3; ++k)
        for (int a = 0; a < 3; ++a) {
            const double delta = R::sub(x(v[k + 1], a), x(v[0], a));
            e[k][a] = R::sub(delta, R::mul(box.side[a], Kokkos::nearbyint(R::mul(delta, box.inverse[a]))));
        }
    for (int k = 0; k < 3; ++k) d[k] = {e[k][0], e[k][1], e[k][2]};
}

// Six times the signed volume and the shape scale of detail::sliver; true
// for a sliver (centre computed exactly).
KOKKOS_INLINE_FUNCTION bool rounded_sliver(const Vec3* d, double& volume6, double& shape) {
    volume6 = R::dot(d[0], R::cross(d[1], d[2]));
    shape = Kokkos::sqrt(R::mul(R::mul(R::dot(d[0], d[0]), R::dot(d[1], d[1])), R::dot(d[2], d[2])));
    return !(Kokkos::abs(volume6) > R::mul(1e-4, shape));
}

// The circumcentre relative to corner 0 (detail::circumcentre, rounded).
KOKKOS_INLINE_FUNCTION Vec3 rounded_circumcentre(const Vec3* d, double volume6) {
    const Vec3 bc = R::cross(d[1], d[2]), ca = R::cross(d[2], d[0]), ab = R::cross(d[0], d[1]);
    const double scale = 0.5 / volume6, aa = R::dot(d[0], d[0]), bb = R::dot(d[1], d[1]), cc = R::dot(d[2], d[2]);
    return {R::mul(R::add(R::add(R::mul(bc.x, aa), R::mul(ca.x, bb)), R::mul(ab.x, cc)), scale),
            R::mul(R::add(R::add(R::mul(bc.y, aa), R::mul(ca.y, bb)), R::mul(ab.y, cc)), scale),
            R::mul(R::add(R::add(R::mul(bc.z, aa), R::mul(ca.z, bb)), R::mul(ab.z, cc)), scale)};
}

// Stores centre c relative to corner v0 as Mesh::centre row: absolute, and
// the squared radius.
template <class X, class Centre>
KOKKOS_INLINE_FUNCTION void store_centre(const X& x, int v0, Vec3 c, const Centre& centre, int t) {
    centre(t, 0) = R::add(x(v0, 0), c.x);
    centre(t, 1) = R::add(x(v0, 1), c.y);
    centre(t, 2) = R::add(x(v0, 2), c.z);
    centre(t, 3) = R::dot(c, c);
}

// The stored circumcentre of tetrahedron t with corners `corners` (any
// order); slivers exactly on the snapped grid, as circumcentre().
template <class X, class Centre>
KOKKOS_INLINE_FUNCTION void stored_centre(const X& x, const Box& box, const int* corners, const Centre& centre, int t) {
    int v[4] = {corners[0], corners[1], corners[2], corners[3]};
    sort_corners(v);
    Vec3 d[3];
    rounded_offsets(x, box, v, d);
    double volume6, shape;
    Vec3 c;
    if (!rounded_sliver(d, volume6, shape)) {
        c = rounded_circumcentre(d, volume6);
    } else {
        namespace ep = exact_predicates;
        ep::Offset g[3];
        for (int k = 0; k < 3; ++k)
            for (int a = 0; a < 3; ++a)
                g[k][a] = ep::delta(ep::snap(x(v[0], a), box, a), ep::snap(x(v[k + 1], a), box, a), box.period[a]);
        ep::i128 n[3];
        const ep::i128 det = ep::circumcentre(g[0], g[1], g[2], n);
        if (det != 0) {
            const double unit = box.step / (2 * double(det));
            c = {double(n[0]) * unit, double(n[1]) * unit, double(n[2]) * unit};
        } else {
            c = rounded_circumcentre(d, volume6);
        }
    }
    store_centre(x, v[0], c, centre, t);
}

// 1. Circumcentres, one thread per tetrahedron: of the first `count` (all if negative) or of those listed in `only`.
// Used by: Mesh<3>::geometry; tet_repair::repair (only for the first `count`, all dirty ones).
template <class ExecSpace, class X>
void circumcentres(const ExecSpace& exec, const X& x, const Box& box,
                   const Mesh<typename ExecSpace::memory_space>& mesh, int count = -1,
                   const Kokkos::View<const int*, typename ExecSpace::memory_space>& only = {}) {
    const bool listed = only.data() != nullptr;
    if (listed) count = static_cast<int>(only.extent(0));
    Kokkos::parallel_for(
        "tet_mesh::circumcentres",
        Kokkos::RangePolicy<ExecSpace>(exec, 0, count < 0 ? static_cast<int>(mesh.vertex.extent(0)) : count),
        KOKKOS_LAMBDA(int s) {
            const int t = listed ? only(s) : s;
            const int v[4] = {mesh.vertex(t, 0), mesh.vertex(t, 1), mesh.vertex(t, 2), mesh.vertex(t, 3)};
            stored_centre(x, box, v, mesh.centre, t);
        });
}

// 2. Face area vectors (owner to neighbour), centroids (relative to the owner) and, if out.volume is allocated,
// cell volumes (summed by atomics), one thread per owned face. Used by: Mesh<3>::geometry, after update_face_rings.
template <class ExecSpace, class X>
void faces(const ExecSpace& exec, const X& x, const Box& box,
           const Mesh<typename ExecSpace::memory_space>& mesh, const Geometry<typename ExecSpace::memory_space>& out) {
    const bool with_volume = out.volume.extent(0) > 0;
    const bool with_rows = out.rows.extent(0) > 0;
    if (with_volume) Kokkos::deep_copy(exec, out.volume, 0.0);
    Kokkos::parallel_for(
        "tet_mesh::faces", Kokkos::RangePolicy<ExecSpace>(exec, 0, mesh.owner.extent(0)), KOKKOS_LAMBDA(int f) {
            const int i = mesh.owner(f), j = mesh.neighbour(f);
            const int begin = mesh.ring_start(f), end = begin + mesh.ring_length(f);
            const Vec3 p = row(x, i);
            auto vertex = [&](int m) { return wrap(row(mesh.centre, mesh.ring(m)) - p, box); };
            const Vec3 dj = wrap(row(x, j) - p, box);
            // Fan triangles (first, previous, v): twice the area vector and (previous + v) weighted by the triangle's
            // area projected on dj, divided once per face (idea from Doug Rennehan's plume, commit 6cbb429).
            const Vec3 first = vertex(begin);
            Vec3 previous = vertex(begin + 1), twice_area{0, 0, 0}, weighted{0, 0, 0};
            double weight = 0;
            for (int m = begin + 2; m < end; ++m) {
                const Vec3 v = vertex(m);
                const Vec3 twice = cross(previous - first, v - first);
                twice_area = twice_area + twice;
                const double w = Kokkos::abs(dot(twice, dj));
                weight += w;
                weighted = weighted + (previous + v) * w;
                previous = v;
            }
            Vec3 area = twice_area * 0.5;
            double projected = dot(area, dj);
            if (projected < 0) {
                area = area * -1.0;
                projected = -projected;
            }
            const Vec3 centroid = weight > 0 ? (first + weighted * (1.0 / weight)) * (1.0 / 3.0) : first;
            out.area(f, 0) = area.x;
            out.area(f, 1) = area.y;
            out.area(f, 2) = area.z;
            out.centroid(f, 0) = centroid.x;
            out.centroid(f, 1) = centroid.y;
            out.centroid(f, 2) = centroid.z;
            if (with_rows) {
                const float row[8] = {float(area.x),     float(area.y),     float(area.z),
                                      float(centroid.x), float(centroid.y), float(centroid.z),
                                      Kokkos::bit_cast<float>(i), Kokkos::bit_cast<float>(j)};
                for (int c = 0; c < 8; ++c) out.rows(f, c) = row[c];
            }
            if (!with_volume) return;
            const double pyramid = projected * (1.0 / 6.0);
            Kokkos::atomic_add(&out.volume(i), pyramid);
            Kokkos::atomic_add(&out.volume(j), pyramid);
        });
}

// Per-cell face lists (CSR) and cell geometry: cell i's faces are face(start(i)) .. face(start(i + 1) - 1) in increasing
// id of the other particle; volume and centroid (relative to the generator) are summed in that order, so run to run the same.
template <class MemorySpace, int D = 3>
struct CellFaces {
    Kokkos::View<int*, MemorySpace> start, face;
    Kokkos::View<int*, MemorySpace> cursor;  // fill positions while building
    Kokkos::View<double*, MemorySpace> volume;
    Kokkos::View<double* [D], Kokkos::LayoutRight, MemorySpace> centroid;
};

// Bytes of scratch cells() needs for `faces` faces: the other particle of
// each list entry.
inline std::size_t cells_scratch_bytes(int faces) { return 2 * static_cast<std::size_t>(faces) * sizeof(int); }

namespace detail {

// In D dimensions a cell's pyramid over a face has volume |A.d| / (2D)
// and its centroid at D / (D + 1) of the way from the generator to the face
// centroid (3D: 1/6 and 3/4).
template <int D>
constexpr double pyramid_volume = 1.0 / (2 * D);
template <int D>
constexpr double pyramid_centroid = static_cast<double>(D) / (D + 1);

// Cell i's pyramid over face f (other particle j): 2D times its volume, |A.d| (A parallel to d up to rounding),
// and the face centroid c relative to i (given relative to the owner, the lower id).
template <int D, class X, class Area, class Centroid>
KOKKOS_INLINE_FUNCTION double pyramid(const X& x, const exact_predicates::BoxN<D>& box, const Area& area,
                                      const Centroid& centroid, int i, int j, int f, VecN<D>& c) {
    const VecN<D> d = wrap(row<D>(x, j) - row<D>(x, i), box);
    const VecN<D> A = row<D>(area, f);
    c = row<D>(centroid, f);
    if (j < i) c = c + d;
    return Kokkos::abs(dot(A, d));
}

// Sort cell i's list entries [begin, end) by the other particle, the XOR of
// the face's two particles and i. `other`: room for the keys.
template <class Ints, class Others, class Const>
KOKKOS_INLINE_FUNCTION void sort_list(const Ints& face, const Others& other, const Const& owner, const Const& neighbour,
                                      int i, int begin, int end) {
    for (int a = begin; a < end; ++a) other(a) = owner(face(a)) ^ neighbour(face(a)) ^ i;
    for (int a = begin + 1; a < end; ++a) {
        const int k = other(a), f = face(a);
        int b = a;
        for (; b > begin && other(b - 1) > k; --b) {
            other(b) = other(b - 1);
            face(b) = face(b - 1);
        }
        other(b) = k;
        face(b) = f;
    }
}

// The sums of cells(), viewed through this: the lists of the `marked` cells
// (all if `all`) are sorted and in `face`; an unmarked cell's list is its
// old list (old_start, old_face) under new numbers (renumber), still sorted.
template <int D, class X, class Areas, class Ints, class Const, class Marks, class Volume, class Centroids>
struct CellLists {
    X x;
    exact_predicates::BoxN<D> box;
    Areas area, centroid;
    Ints start, face;
    Const old_start, old_face, renumber, owner, neighbour;
    Marks marked;
    Volume volume;
    Centroids cell_centroid;
    int n;
    bool all;

    KOKKOS_INLINE_FUNCTION bool kept(int i) const { return !all && !marked(i); }
    // List entry `at` of a cell, moved into `face` if it is kept.
    KOKKOS_INLINE_FUNCTION int entry(int begin, int from, int at, bool is_kept) const {
        if (!is_kept) return face(begin + at);
        const int f = renumber(old_face(from + at));
        face(begin + at) = f;
        return f;
    }
    KOKKOS_INLINE_FUNCTION double side(int i, int f, VecN<D>& c) const {
        return pyramid<D>(x, box, area, centroid, i, owner(f) ^ neighbour(f) ^ i, f, c);
    }
    KOKKOS_INLINE_FUNCTION void finish(int i, double v, VecN<D> moment) const {
        volume(i) = v * pyramid_volume<D>;
        store(cell_centroid, i, moment * (pyramid_centroid<D> / v));
    }
    // The sequential sum of one cell.
    KOKKOS_INLINE_FUNCTION void sum(int i) const {
        const int begin = start(i), length = start(i + 1) - begin, from = all ? 0 : old_start(i);
        const bool is_kept = kept(i);
        double v = 0;
        VecN<D> moment{};
        for (int at = 0; at < length; ++at) {
            VecN<D> c;
            const double p = side(i, entry(begin, from, at, is_kept), c);
            v += p;
            moment = moment + c * p;
        }
        finish(i, v, moment);
    }
};

#if defined(KOKKOS_ENABLE_CUDA)
// Sorting on CUDA, a warp per cell: lane l holds entry l, a warp bitonic sort.
template <class Ints, class Others, class Const, class Marks>
struct CellSortWarp {
    static constexpr int warps = 4;  // cells per block
    Ints start, face;
    Others other;
    Const owner, neighbour;
    Marks marked;
    int n;
    bool all;

    KOKKOS_FUNCTION void operator()(const typename Kokkos::TeamPolicy<Kokkos::Cuda>::member_type& member) const {
        const int i = member.league_rank() * warps + member.team_rank();
        if (i >= n || !(all || marked(i) == 2)) return;
#if defined(__CUDA_ARCH__)
        const int begin = start(i), end = start(i + 1);
        constexpr unsigned everyone = 0xffffffffu;
        const int l = threadIdx.x;
        if (end - begin > 32) {
            if (l == 0) sort_list(face, other, owner, neighbour, i, begin, end);
            return;
        }
        int key = 0x7fffffff, f = -1;
        if (begin + l < end) {
            f = face(begin + l);
            key = owner(f) ^ neighbour(f) ^ i;
        }
        for (int size = 2; size <= 32; size <<= 1)
            for (int stride = size / 2; stride > 0; stride >>= 1) {
                const int other_key = __shfl_xor_sync(everyone, key, stride);
                const int other_f = __shfl_xor_sync(everyone, f, stride);
                const bool ascending = (l & size) == 0, lower = (l & stride) == 0;
                if ((other_key < key) == (lower == ascending) && other_key != key) {
                    key = other_key;
                    f = other_f;
                }
            }
        if (begin + l < end) face(begin + l) = f;
#endif
    }
};

// The sums on CUDA: a warp holds 32 / lanes cells, the lanes of a cell gather its list a batch at a time, in list order,
// so the bits equal the sequential loop of CellLists::sum (a pyramid of zero past the list's end changes nothing).
template <class Lists, int D>
struct CellSumsWarp {
    static constexpr int warps = 4, lanes = 4, cells = 32 / lanes;
    static constexpr int block = warps * cells;  // cells per block
    Lists lists;

    KOKKOS_FUNCTION void operator()(const typename Kokkos::TeamPolicy<Kokkos::Cuda>::member_type& member) const {
#if defined(__CUDA_ARCH__)
        constexpr unsigned everyone = 0xffffffffu;
        const int l = threadIdx.x, lane = l % lanes, first = l - lane;
        const int i = (member.league_rank() * warps + member.team_rank()) * cells + l / lanes;
        const bool real = i < lists.n;
        const int begin = real ? lists.start(i) : 0, length = real ? lists.start(i + 1) - begin : 0;
        const int from = real && !lists.all ? lists.old_start(i) : 0;
        const bool is_kept = real && lists.kept(i);
        int batches = (length + lanes - 1) / lanes;
        for (int stride = lanes; stride < 32; stride <<= 1)
            batches = Kokkos::max(batches, __shfl_xor_sync(everyone, batches, stride));
        double v = 0;
        VecN<D> moment{};
        for (int b = 0; b < batches; ++b) {
            const int at = lane + b * lanes;
            double p = 0;
            VecN<D> c{};
            if (at < length) p = lists.side(i, lists.entry(begin, from, at, is_kept), c);
            for (int r = 0; r < lanes; ++r) {
                const double pr = __shfl_sync(everyone, p, first + r);
                VecN<D> cr;
                cr.x = __shfl_sync(everyone, c.x, first + r);
                cr.y = __shfl_sync(everyone, c.y, first + r);
                if constexpr (D == 3) cr.z = __shfl_sync(everyone, c.z, first + r);
                v += pr;
                moment = moment + cr * pr;
            }
        }
        if (real && lane == 0) lists.finish(i, v, moment);
#endif
    }
};
#endif

// Sort the `marked` lists (all if `all`), then sum every cell.
template <int D, class ExecSpace, class X, class Others, class Marks>
void sort_and_sum(const ExecSpace& exec, const X& x, const exact_predicates::BoxN<D>& box,
                  const Geometry<typename ExecSpace::memory_space, D>& geometry,
                  const CellFaces<typename ExecSpace::memory_space, D>& out,
                  const Kokkos::View<const int*, typename ExecSpace::memory_space>& old_start,
                  const Kokkos::View<const int*, typename ExecSpace::memory_space>& old_face,
                  const Kokkos::View<const int*, typename ExecSpace::memory_space>& renumber,
                  const Kokkos::View<const int*, typename ExecSpace::memory_space>& owner,
                  const Kokkos::View<const int*, typename ExecSpace::memory_space>& neighbour, const Others& other,
                  const Marks& marked, bool all, bool sort = true) {
    using Memory = typename ExecSpace::memory_space;
    using Const = Kokkos::View<const int*, Memory>;
    using Ints = Kokkos::View<int*, Memory>;
    using Lists = CellLists<D, X, typename Geometry<Memory, D>::Areas, Ints, Const, Marks, Kokkos::View<double*, Memory>,
                            Kokkos::View<double* [D], Kokkos::LayoutRight, Memory>>;
    const int n = static_cast<int>(out.volume.extent(0));
    const Lists lists{x,        box,   geometry.area, geometry.centroid, out.start, out.face, old_start,
                      old_face, renumber, owner,      neighbour,         marked,    out.volume, out.centroid,
                      n,        all};
#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
        using Sort = CellSortWarp<Ints, Others, Const, Marks>;
        using Sums = CellSumsWarp<Lists, D>;
        if (sort)
            Kokkos::parallel_for("tet_mesh::cell_sort",
                                 Kokkos::TeamPolicy<ExecSpace>(exec, (n + Sort::warps - 1) / Sort::warps, Sort::warps, 32),
                                 Sort{out.start, out.face, other, owner, neighbour, marked, n, all});
        Kokkos::parallel_for("tet_mesh::cell_sums",
                             Kokkos::TeamPolicy<ExecSpace>(exec, (n + Sums::block - 1) / Sums::block, Sums::warps, 32),
                             Sums{lists});
        return;
    }
#endif
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const auto start = out.start;
    const auto face = out.face;
    if (sort)
        Kokkos::parallel_for(
            "tet_mesh::cell_sort", Range(exec, 0, n), KOKKOS_LAMBDA(int i) {
                if (all || marked(i) == 2) sort_list(face, other, owner, neighbour, i, start(i), start(i + 1));
            });
    Kokkos::parallel_for("tet_mesh::cell_sums", Range(exec, 0, n), KOKKOS_LAMBDA(int i) { lists.sum(i); });
}

}  // namespace detail

// 3. The cell face lists, volumes and centroids from the faces (owner, neighbour) and their geometry (area, centroid),
// D = 2 or 3, x the positions the geometry came from; `scratch` holds cells_scratch_bytes(faces) bytes. Grows out.face when needed.
// One pass appends each face to both cells' lists; each cell then sorts its list by the other particle's id and sums its
// pyramids in that order (detail::sort_and_sum): the same sums as faces(), in fixed order.
// Used by: Mesh::update_cells (full path).
template <class ExecSpace, class X, int D>
void cells(const ExecSpace& exec, const X& x, const exact_predicates::BoxN<D>& box,
           const Kokkos::View<const int*, typename ExecSpace::memory_space>& owner,
           const Kokkos::View<const int*, typename ExecSpace::memory_space>& neighbour,
           const Geometry<typename ExecSpace::memory_space, D>& geometry,
           CellFaces<typename ExecSpace::memory_space, D>& out, char* scratch) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const int n = static_cast<int>(out.volume.extent(0)), faces = static_cast<int>(owner.extent(0));
    if (static_cast<int>(out.face.extent(0)) < 2 * faces)
        Kokkos::realloc(Kokkos::WithoutInitializing, out.face, 2 * faces + faces / 16);
    const Kokkos::View<int*, Memory, Kokkos::MemoryTraits<Kokkos::Unmanaged>> other(reinterpret_cast<int*>(scratch),
                                                                                    2 * faces);
    const auto start = out.start, cursor = out.cursor, face = out.face;
    Kokkos::deep_copy(exec, start, 0);
    Kokkos::parallel_for(
        "tet_mesh::cell_face_counts", Range(exec, 0, faces), KOKKOS_LAMBDA(int f) {
            Kokkos::atomic_inc(&start(owner(f)));
            Kokkos::atomic_inc(&start(neighbour(f)));
        });
    Kokkos::parallel_scan(
        "tet_mesh::cell_face_start", Range(exec, 0, n + 1), KOKKOS_LAMBDA(int i, int& sum, bool final) {
            const int count = i < n ? start(i) : 0;
            if (final) {
                start(i) = sum;
                if (i < n) cursor(i) = sum;
            }
            sum += count;
        });
    Kokkos::parallel_for(
        "tet_mesh::cell_face_fill", Range(exec, 0, faces), KOKKOS_LAMBDA(int f) {
            const int o = owner(f), j = neighbour(f);
            const int at_owner = Kokkos::atomic_fetch_inc(&cursor(o)), at_neighbour = Kokkos::atomic_fetch_inc(&cursor(j));
            face(at_owner) = f;
            face(at_neighbour) = f;
        });
    detail::sort_and_sum<D>(exec, x, box, geometry, out, {}, {}, {}, owner, neighbour, other,
                            Kokkos::View<const char*, Memory, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(), true);
}

// Bytes of scratch cells_incremental needs after the renumbering table.
inline std::size_t incremental_scratch_bytes(int n, int old_faces, int faces) {
    return sizeof(int) * (static_cast<std::size_t>(n) + 1 + 2 * static_cast<std::size_t>(old_faces) + 2 * static_cast<std::size_t>(faces)) + n;
}

namespace detail {

// The scratch of incremental lists: the old lists, the sort keys, the marks.
template <class Memory>
struct IncrementalLists {
    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using Ints = Kokkos::View<int*, Memory, Unmanaged>;
    Ints old_start, old_face, other;
    Kokkos::View<char*, Memory, Unmanaged> marked;
};

// Sets up in `at` (incremental_scratch_bytes(n, old_faces, faces) bytes): copies
// the old lists, clears the marks, makes room for the new lists.
template <class ExecSpace, int D>
IncrementalLists<typename ExecSpace::memory_space> begin_incremental(const ExecSpace& exec, int* at, int n, int old_faces, int faces,
                                                                     CellFaces<typename ExecSpace::memory_space, D>& out) {
    using Memory = typename ExecSpace::memory_space;
    using Lists = IncrementalLists<Memory>;
    IncrementalLists<Memory> l;
    l.old_start = typename Lists::Ints(at, n + 1);
    at += n + 1;
    l.old_face = typename Lists::Ints(at, 2 * old_faces);
    at += 2 * old_faces;
    l.other = typename Lists::Ints(at, 2 * faces);
    at += 2 * faces;
    l.marked = Kokkos::View<char*, Memory, typename Lists::Unmanaged>(reinterpret_cast<char*>(at), n);
    Kokkos::deep_copy(exec, l.old_start, Kokkos::View<const int*, Memory>(out.start));
    Kokkos::deep_copy(exec, l.old_face, Kokkos::subview(out.face, std::make_pair(0, 2 * old_faces)));
    Kokkos::deep_copy(exec, l.marked, char(0));
    if (static_cast<int>(out.face.extent(0)) < 2 * faces)
        Kokkos::realloc(Kokkos::WithoutInitializing, out.face, 2 * faces + faces / 16);
    return l;
}

// The lists and sums from the old lists, D = 2 or 3; l.marked is set for every cell that may have lost or gained a face;
// renumber(f) is the new number of old face f (-1: none), and ~f in dirty[0, dirty_count) names each new face without one once.
template <int D, class ExecSpace, class X, class Dirty>
void finish_incremental(const ExecSpace& exec, const X& x, const exact_predicates::BoxN<D>& box,
                        const Kokkos::View<const int*, typename ExecSpace::memory_space>& owner,
                        const Kokkos::View<const int*, typename ExecSpace::memory_space>& neighbour,
                        const Geometry<typename ExecSpace::memory_space, D>& geometry,
                        CellFaces<typename ExecSpace::memory_space, D>& out,
                        const Kokkos::View<const int*, typename ExecSpace::memory_space>& renumber, const Dirty& dirty,
                        int dirty_count, const IncrementalLists<typename ExecSpace::memory_space>& lists) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    const int n = static_cast<int>(out.volume.extent(0));
    const auto old_start = lists.old_start, old_face = lists.old_face, other = lists.other;
    const auto marked = lists.marked;
    const auto start = out.start, cursor = out.cursor, face = out.face;
    // The marked cells' counts in `start`, the others' from the old list.
    Kokkos::parallel_for(
        "tet_mesh::marked_counts", Range(exec, 0, n), KOKKOS_LAMBDA(int i) {
            if (!marked(i)) return;
            int kept = 0;
            for (int a = old_start(i); a < old_start(i + 1); ++a) kept += renumber(old_face(a)) >= 0;
            start(i) = kept;
        });
    Kokkos::parallel_for(
        "tet_mesh::fresh_counts", Range(exec, 0, dirty_count), KOKKOS_LAMBDA(int s) {
            if (dirty(s) >= 0) return;
            const int f = ~dirty(s);
            Kokkos::atomic_inc(&start(owner(f)));
            Kokkos::atomic_inc(&start(neighbour(f)));
        });
    Kokkos::parallel_scan(
        "tet_mesh::cell_face_start", Range(exec, 0, n + 1), KOKKOS_LAMBDA(int i, int& sum, bool final) {
            const int count = i == n ? 0 : marked(i) ? start(i) : old_start(i + 1) - old_start(i);
            if (final) start(i) = sum;
            sum += count;
        });
    Kokkos::parallel_for(
        "tet_mesh::marked_fill", Range(exec, 0, n), KOKKOS_LAMBDA(int i) {
            if (!marked(i)) return;
            int to = start(i);
            for (int a = old_start(i); a < old_start(i + 1); ++a) {
                const int f = renumber(old_face(a));
                if (f >= 0) face(to++) = f;
            }
            cursor(i) = to;
        });
    Kokkos::parallel_for(
        "tet_mesh::fresh_fill", Range(exec, 0, dirty_count), KOKKOS_LAMBDA(int s) {
            if (dirty(s) >= 0) return;
            const int f = ~dirty(s);
            const int o = owner(f), j = neighbour(f);
            face(Kokkos::atomic_fetch_inc(&cursor(o))) = f;
            face(Kokkos::atomic_fetch_inc(&cursor(j))) = f;
            marked(o) = 2;
            marked(j) = 2;
        });
    sort_and_sum<D>(exec, x, box, geometry, out, old_start, old_face, renumber, owner, neighbour, other,
                    Kokkos::View<const char*, Memory, Unmanaged>(marked), false);
}

}  // namespace detail

// 3'. cells() after a repair, from the last step's lists, bit for bit the same: only cells with a corner of a
// tetrahedron created by a flip (list `created`) can change faces; the others renumber (`renumber`) and sum again.
// Used by: Mesh<3>::update_cells (when few entries were touched).
template <class ExecSpace, class X>
void cells_incremental(const ExecSpace& exec, const X& x, const Box& box,
                       const Kokkos::View<const int*, typename ExecSpace::memory_space>& owner,
                       const Kokkos::View<const int*, typename ExecSpace::memory_space>& neighbour,
                       const Mesh<typename ExecSpace::memory_space>& mesh,
                       const Kokkos::View<const char*, typename ExecSpace::memory_space>& alive,
                       const Kokkos::View<const int*, typename ExecSpace::memory_space>& created,
                       const FaceRings<typename ExecSpace::memory_space>& rings,
                       const Geometry<typename ExecSpace::memory_space, 3>& geometry,
                       CellFaces<typename ExecSpace::memory_space, 3>& out, char* scratch) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using Ints = Kokkos::View<int*, Memory, Unmanaged>;
    const int n = static_cast<int>(out.volume.extent(0)), faces = static_cast<int>(owner.extent(0));
    const int old_faces = rings.renumbered;
    // After the table and the dirty list (in the spare pool, update_face_rings).
    const Ints renumber(reinterpret_cast<int*>(scratch), old_faces);
    const auto lists = detail::begin_incremental<ExecSpace, 3>(exec, renumber.data() + old_faces + rings.dirty_count, n, old_faces, faces, out);
    const auto marked = lists.marked;
    const auto vertex = mesh.vertex;
    Kokkos::parallel_for(
        "tet_mesh::mark_cells", Range(exec, 0, created.extent(0)), KOKKOS_LAMBDA(int s) {
            const int t = created(s);
            if (!alive(t)) return;
            for (int c = 0; c < 4; ++c) marked(vertex(t, c)) = 1;
        });
    detail::finish_incremental<3>(exec, x, box, owner, neighbour, geometry, out, Kokkos::View<const int*, Memory>(renumber), rings.dirty,
                                  rings.dirty_count, lists);
}

// 4. Least-squares gradient weights of every per-cell list entry, as Arepo's prepare_gradient_operator: for cell i with
// entries k (face f, other cell j), X_k = (1, d_k), d_k from the centroid of i to that of j (nearest image), w_k = |A_f|,
// M = sum_k w_k X_k X_k^T; entry k gets weight(k) = w_k (M^-1 X_k)_{1..D}, so sum_k weight(k) (u_j - u_i) is the gradient
// of the area-weighted fit u_j - u_i = a + g.d_k. In double (Cholesky of M), stored in the type of `out` (rows of D);
// one thread per cell, after cells(). Used by: Mesh::update_cells (Weight != void).
template <class ExecSpace, class Xs, int D, class Weights>
void gradient_weights(const ExecSpace& exec, const Xs& x, const exact_predicates::BoxN<D>& box,
                      const Kokkos::View<const int*, typename ExecSpace::memory_space>& owner,
                      const Kokkos::View<const int*, typename ExecSpace::memory_space>& neighbour,
                      const Geometry<typename ExecSpace::memory_space, D>& geometry,
                      const CellFaces<typename ExecSpace::memory_space, D>& cells, const Weights& out) {
    using Weight = typename Weights::value_type;
    const int n = static_cast<int>(cells.volume.extent(0));
    const auto start = cells.start, face = cells.face;
    const auto centroid = cells.centroid;
    const auto area = geometry.area;
    Kokkos::parallel_for(
        "tet_mesh::gradient_weights", Kokkos::RangePolicy<ExecSpace>(exec, 0, n), KOKKOS_LAMBDA(int i) {
            constexpr int U = D + 1;
            // Regressors (1, d) and weight of entry k.
            auto regressors = [&](int k, double* X) {
                const int f = face(k), o = owner(f), m = neighbour(f), j = o == i ? m : o;
                X[0] = 1;
                for (int a = 0; a < D; ++a)
                    X[1 + a] = wrap(x(j, a) - x(i, a), box.side[a], box.inverse[a]) + centroid(j, a) - centroid(i, a);
                double s = 0;
                for (int a = 0; a < D; ++a) s += area(f, a) * area(f, a);
                return Kokkos::sqrt(s);
            };
            double M[U][U] = {};
            for (int k = start(i); k < start(i + 1); ++k) {
                double X[U];
                const double w = regressors(k, X);
                for (int p = 0; p < U; ++p)
                    for (int s = p; s < U; ++s) M[p][s] += w * X[p] * X[s];
            }
            double L[U][U] = {};
            for (int p = 0; p < U; ++p)
                for (int s = 0; s <= p; ++s) {
                    double sum = M[s][p];
                    for (int m = 0; m < s; ++m) sum -= L[p][m] * L[s][m];
                    L[p][s] = p == s ? Kokkos::sqrt(sum) : sum / L[s][s];
                }
            for (int k = start(i); k < start(i + 1); ++k) {
                double X[U], z[U], y[U];
                const double w = regressors(k, X);
                for (int p = 0; p < U; ++p) {
                    double sum = X[p];
                    for (int m = 0; m < p; ++m) sum -= L[p][m] * z[m];
                    z[p] = sum / L[p][p];
                }
                for (int p = U - 1; p >= 0; --p) {
                    double sum = z[p];
                    for (int m = p + 1; m < U; ++m) sum -= L[m][p] * y[m];
                    y[p] = sum / L[p][p];
                }
                for (int a = 0; a < D; ++a) out(k, a) = static_cast<Weight>(w * y[1 + a]);
            }
        });
}

}  // namespace tet_mesh
