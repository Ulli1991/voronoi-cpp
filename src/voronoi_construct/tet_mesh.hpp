#pragma once

#include "exact_predicates.hpp"
#include "shell_voronoi.hpp"

// A persistent Delaunay tetrahedralization and the Voronoi geometry it implies.
//
// Each tetrahedron is four particles whose circumsphere holds no other
// particle; its circumcentre is a Voronoi vertex shared by the four cells.
// A Voronoi face between particles i and j is the ring of circumcentres of
// the tetrahedra around the edge i-j. Kept from step to step, the mesh needs
// per step only
//   1. one circumcentre per tetrahedron,
//   2. face geometry from those circumcentres,
//   3. a check that each tetrahedron's circumsphere is still empty of the
//      far vertices of its neighbours (the local Delaunay condition),
// and repair only where the check fails.
namespace tet_mesh {

using shell_voronoi::Vec3;

// One tetrahedron's entries are contiguous (row-major on every backend), so
// a random access touches one memory sector instead of one per entry.
template <class MemorySpace>
using Quads = Kokkos::View<int* [4], Kokkos::LayoutRight, MemorySpace>;
template <class MemorySpace>
using Points = Kokkos::View<double* [3], Kokkos::LayoutRight, MemorySpace>;
// Positions or centres padded to 32 bytes: one memory sector per point.
template <class MemorySpace>
using Points4 = Kokkos::View<double* [4], Kokkos::LayoutRight, MemorySpace>;

template <class MemorySpace>
struct Mesh {
    Quads<MemorySpace> vertex;     // particles
    Quads<MemorySpace> adjacent;   // tetrahedron across the face opposite vertex k, -1 if none
    Points4<MemorySpace> centre;   // circumcentre, unwrapped near vertex 0, and squared radius
    // Owned faces (owner < neighbour), each a ring of tetrahedra in order:
    // face f's is ring[ring_start(f), ring_start(f) + ring_length(f)).
    Kokkos::View<int*, MemorySpace> owner, neighbour, ring_start, ring_length, ring;
    // Bit k: this tetrahedron represents its edge k (see edge_ends), i.e. the
    // Voronoi face of that pair of particles. Every edge has exactly one
    // representative, so a scan over the bits numbers the faces.
    Kokkos::View<int*, MemorySpace> edges;
};

// Face area vectors (owner to neighbour), centroids (relative to the owner),
// and cell volumes, one per particle.
template <class MemorySpace>
struct Geometry {
    Kokkos::View<double* [3], MemorySpace> area, centroid;
    Kokkos::View<double*, MemorySpace> volume;
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

// Topology queries shared by the device repair and the host fallback; m is
// anything with vertex(t, k) and adjacent(t, k).

// Corner of tetrahedron t holding particle v, or -1.
template <class M>
KOKKOS_INLINE_FUNCTION int position(const M& m, int t, int v) {
    for (int k = 0; k < 4; ++k)
        if (m.vertex(t, k) == v) return k;
    return -1;
}

// The particle of u that t lacks, or -1.
template <class M>
KOKKOS_INLINE_FUNCTION int far_vertex(const M& m, int t, int u) {
    for (int k = 0; k < 4; ++k)
        if (position(m, t, m.vertex(u, k)) < 0) return m.vertex(u, k);
    return -1;
}

// The face of tetrahedron v opposite corner k, its particles sorted.
KOKKOS_INLINE_FUNCTION void sorted_face(const int* v, int k, int* f) {
    for (int a = 0, b = 0; a < 4; ++a)
        if (a != k) f[b++] = v[a];
    if (f[0] > f[1]) Kokkos::kokkos_swap(f[0], f[1]);
    if (f[1] > f[2]) Kokkos::kokkos_swap(f[1], f[2]);
    if (f[0] > f[1]) Kokkos::kokkos_swap(f[0], f[1]);
}

// Tetrahedra around edge i-j from t, in order and t first, written to
// ring[0, 65) unless it is null. Returns the number of steps (the ring
// length when it closes).
template <class M>
KOKKOS_INLINE_FUNCTION int walk_edge(const M& m, int t, int i, int j, int* ring = nullptr) {
    int current = t, before = -1, steps = 0;
    do {
        if (ring) ring[steps] = current;
        int next = -1;
        for (int c = 0; c < 4 && next < 0; ++c) {
            const int v = m.vertex(current, c);
            if (v == i || v == j) continue;
            const int w = m.adjacent(current, c);
            if (w != before) next = w;
        }
        if (next < 0 || ++steps > 64) break;
        before = current;
        current = next;
    } while (current != t);
    return steps;
}

// A coordinate difference in a periodic box, to the nearest image.
KOKKOS_INLINE_FUNCTION double wrap(double d, double box) { return d - box * Kokkos::nearbyint(d / box); }

namespace detail {
// The point equidistant from the origin and a, b, c.
KOKKOS_INLINE_FUNCTION Vec3<double> circumcentre(Vec3<double> a, Vec3<double> b, Vec3<double> c) {
    const Vec3<double> bc = cross(b, c), ca = cross(c, a), ab = cross(a, b);
    const double scale = 0.5 / dot(a, bc);
    return (bc * dot(a, a) + ca * dot(b, b) + ab * dot(c, c)) * scale;
}
}  // namespace detail

// Positions padded to 32 bytes, the preferred input of the kernels below.
template <class ExecSpace, class X>
void pad(const ExecSpace& exec, const X& x, const Points4<typename ExecSpace::memory_space>& padded) {
    Kokkos::parallel_for(
        "tet_mesh::pad", Kokkos::RangePolicy<ExecSpace>(exec, 0, x.extent(0)), KOKKOS_LAMBDA(int i) {
            for (int a = 0; a < 3; ++a) padded(i, a) = x(i, a);
        });
}

// The tetrahedra and their adjacency from the face rings of a clipping build
// (cells in slot order, every cell ok), without hashing: tetrahedron
// (i, j, q_m, q_m+1) is emitted only by the face of its two smallest
// particles, and each triangle's two tetrahedra sit next to each other in
// the ring of its two smallest particles' face. Returns the number of ring
// slots or adjacencies that could not be resolved (0 for a consistent build).
template <class ExecSpace>
int from_rings(const ExecSpace& exec, const shell_voronoi::Cells<typename ExecSpace::memory_space>& cells,
               Mesh<typename ExecSpace::memory_space>& mesh) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const int n = static_cast<int>(cells.owned_count.extent(0));

    // Faces in CSR: cell t owns faces [face_start(t), face_start(t + 1)).
    Kokkos::View<int*, Memory> face_start("face_start", n + 1);
    int faces = 0;
    Kokkos::parallel_scan(
        "tet_mesh::face_start", Range(exec, 0, n),
        KOKKOS_LAMBDA(int t, int& sum, bool last) {
            if (last) face_start(t) = sum;
            sum += cells.owned_count(t);
            if (last && t == n - 1) face_start(n) = sum;
        },
        faces);
    Kokkos::View<int*, Memory> owner("face_owner", faces), neighbour("face_neighbour", faces),
        length("face_length", faces);
    Kokkos::parallel_for(
        "tet_mesh::face_lengths", Range(exec, 0, n), KOKKOS_LAMBDA(int t) {
            for (int f = 0, at = 0; f < cells.owned_count(t); ++f) {
                const int g = face_start(t) + f, k = cells.rings(t, at);
                owner(g) = t;
                neighbour(g) = cells.neighbour(t, f);
                length(g) = k;
                at += 1 + k;
            }
        });
    auto face_of = KOKKOS_LAMBDA(int a, int b) {  // a < b
        for (int g = face_start(a); g < face_start(a + 1); ++g)
            if (neighbour(g) == b) return g;
        return -1;
    };

    Kokkos::View<int*, Memory> ring_start("face_ring_start", faces + 1), tet_start("face_tet_start", faces + 1);
    int slots = 0, tets = 0;
    Kokkos::parallel_scan(
        "tet_mesh::ring_start", Range(exec, 0, faces),
        KOKKOS_LAMBDA(int g, int& sum, bool last) {
            if (last) ring_start(g) = sum;
            sum += length(g);
            if (last && g == faces - 1) ring_start(faces) = sum;
        },
        slots);
    // The rings' particles, contiguous per face (the cells' rows are strided).
    Kokkos::View<int*, Memory> particles(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("face_particles")),
                                         slots);
    Kokkos::parallel_for(
        "tet_mesh::face_particles", Range(exec, 0, n), KOKKOS_LAMBDA(int t) {
            for (int f = 0, at = 0; f < cells.owned_count(t); ++f) {
                const int g = face_start(t) + f, k = length(g);
                for (int m = 0; m < k; ++m) particles(ring_start(g) + m) = cells.rings(t, at + 1 + m);
                at += 1 + k;
            }
        });
    auto ring_of = KOKKOS_LAMBDA(int g, int m) { return particles(ring_start(g) + m % length(g)); };
    // A face emits the slots whose tetrahedron has the face's particles as its two smallest.
    auto emits = KOKKOS_LAMBDA(int g, int m) {
        const int j = neighbour(g);
        return j < ring_of(g, m) && j < ring_of(g, m + 1);
    };
    Kokkos::parallel_scan(
        "tet_mesh::tet_start", Range(exec, 0, faces),
        KOKKOS_LAMBDA(int g, int& sum, bool last) {
            if (last) tet_start(g) = sum;
            for (int m = 0; m < length(g); ++m) sum += emits(g, m);
            if (last && g == faces - 1) tet_start(faces) = sum;
        },
        tets);
    mesh.vertex = Quads<Memory>("tet_vertex", tets);
    mesh.adjacent = Quads<Memory>("tet_adjacent", tets);
    mesh.centre = Points4<Memory>("tet_centre", tets);
    mesh.edges = Kokkos::View<int*, Memory>("tet_edges", tets);
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
            for (int m = 0, e = tet_start(g); m < length(g); ++m) {
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
                for (int r = 0, e2 = h < 0 ? 0 : tet_start(h); h >= 0 && r < length(h); ++r) {
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

    // Neighbours: across a triangle through edge i-j, the next tetrahedron in
    // that ring; across the other two triangles, look up their own rings.
    int open = 0;
    Kokkos::parallel_reduce(
        "tet_mesh::adjacency", Range(exec, 0, faces),
        KOKKOS_LAMBDA(int g, int& missing) {
            const int k = length(g), base = ring_start(g);
            for (int m = 0; m < k; ++m) {
                if (!emits(g, m)) continue;
                const int e = ring(base + m);
                adjacent(e, 2) = ring(base + (m + 1) % k);      // opposite p: triangle (i, j, q)
                adjacent(e, 3) = ring(base + (m + k - 1) % k);  // opposite q: triangle (i, j, p)
                for (int drop = 0; drop < 2; ++drop) {          // opposite i, then j
                    int tri[3] = {vertex(e, drop == 0 ? 1 : 0), vertex(e, 2), vertex(e, 3)};
                    for (int a = 1; a < 3; ++a)
                        for (int b = a; b > 0 && tri[b - 1] > tri[b]; --b) Kokkos::kokkos_swap(tri[b - 1], tri[b]);
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
    // Each face's first ring tetrahedron represents its edge.
    const auto edges = mesh.edges;
    Kokkos::parallel_for(
        "tet_mesh::represent", Range(exec, 0, faces), KOKKOS_LAMBDA(int g) {
            const int e = ring(ring_start(g));
            if (e < 0) return;
            int a = -1, b = -1;
            for (int k = 0; k < 4; ++k) {
                if (vertex(e, k) == owner(g)) a = k;
                if (vertex(e, k) == neighbour(g)) b = k;
            }
            if (a >= 0 && b >= 0) Kokkos::atomic_or(&edges(e), 1 << edge_index(a, b));
        });
    return unresolved + open;
}

// Face rings kept from step to step. Face start(t) + rank is the rank-th
// edge (in bit order) that tetrahedron t represents; its ring of tetrahedra
// is pool[ring_start(f), ring_start(f) + ring_length(f)). Numbers and ring
// positions come in pairs, this step's and the last one's; the pool comes
// twice, one in use and a spare to compact it into.
template <class MemorySpace>
struct FaceRings {
    using Ints = Kokkos::View<int*, MemorySpace>;
    Ints start[2], ring_start[2], ring_length[2], pool[2];
    Ints owner, neighbour;
    Ints dirty;                                // faces to walk this step
    Kokkos::View<int, MemorySpace> listed;     // their number
    int faces = -1;   // -1: none yet, every face is walked
    int current = 0;  // start, ring_start, ring_length [current] are the last step's
    int in_use = 0;   // pool[in_use] holds the rings
    int used = 0;     // its entries in use, superseded rings included
};

template <class MemorySpace>
FaceRings<MemorySpace> make_face_rings(int tets, int faces, int entries) {
    using Ints = Kokkos::View<int*, MemorySpace>;
    FaceRings<MemorySpace> r;
    for (int b = 0; b < 2; ++b) {
        r.start[b] = Ints("face_rings_start", tets + 1);
        r.ring_start[b] = Ints("face_rings_ring_start", faces);
        r.ring_length[b] = Ints("face_rings_ring_length", faces);
        r.pool[b] = Ints(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("face_rings_pool")), entries);
    }
    r.owner = Ints("face_rings_owner", faces);
    r.neighbour = Ints("face_rings_neighbour", faces);
    r.dirty = Ints(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("face_rings_dirty")), faces);
    r.listed = Kokkos::View<int, MemorySpace>("face_rings_listed");
    return r;
}

// The face of edge i-j (particles of tetrahedron t) in the numbering
// `start`, and its representative: found by walking around the edge.
// Returns -1 if the walk does not reach it.
template <class M, class Start>
KOKKOS_INLINE_FUNCTION int edge_face(const M& m, const Start& start, int t, int i, int j, int& representative) {
    for (int current = t, before = -1, steps = 0; steps < 64; ++steps) {
        const int bits = m.edges(current);
        const int a = position(m, current, i), b = position(m, current, j);
        if (a < 0 || b < 0) return -1;
        const int bit = 1 << edge_index(a, b);
        if (bits & bit) {
            representative = current;
            return start(current) + Kokkos::Experimental::popcount_builtin(static_cast<unsigned>(bits & (bit - 1)));
        }
        int next = -1;
        for (int c = 0; c < 4 && next < 0; ++c) {
            const int v = m.vertex(current, c);
            if (v == i || v == j) continue;
            const int w = m.adjacent(current, c);
            if (w != before) next = w;
        }
        if (next < 0 || next == t) return -1;
        before = current;
        current = next;
    }
    return -1;
}

// Renumber the faces after a repair and update their rings in place: a face
// keeps its ring unless a tetrahedron touched by a flip lies on it; then
// its edge is walked and the new ring appended to the pool. Those faces are
// found by walking the edges of the touched tetrahedra, `touched_list` (any
// order, repeats allowed; an empty handle: all t with touched(t) != 0), so
// clean rings are never read and only listed faces are walked. Requires
// that untouched tetrahedra keep their edge bits. When the pool is full,
// the rings are compacted into the spare one. Sets
// mesh.owner/neighbour/ring_start/ring_length/ring; returns the number of faces.
template <class ExecSpace>
int update_face_rings(const ExecSpace& exec, Mesh<typename ExecSpace::memory_space>& mesh, int count,
                      const Kokkos::View<char*, typename ExecSpace::memory_space>& touched,
                      const Kokkos::View<const int*, typename ExecSpace::memory_space>& touched_list,
                      FaceRings<typename ExecSpace::memory_space>& rings) {
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const bool all = rings.faces < 0;
    const int now = 1 - rings.current;
    const auto edges = mesh.edges;
    const auto start = rings.start[now], old_start = rings.start[rings.current];
    const auto all_ring_start = rings.ring_start[now], all_ring_length = rings.ring_length[now];
    const auto old_ring_start = rings.ring_start[rings.current];
    const auto old_ring_length = rings.ring_length[rings.current];
    const auto owner = rings.owner, neighbour = rings.neighbour;
    const int face_capacity = static_cast<int>(owner.extent(0));
    // Number the faces, with owner and neighbour. A face represented by an
    // untouched tetrahedron takes its old ring (unless marked below), one
    // represented by a touched tetrahedron gets ring_start 0 (always marked
    // below), and in the first call every face is walked.
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
                    const int old = old_start(t) + rank;
                    all_ring_start(f) = old_ring_start(old);
                    all_ring_length(f) = old_ring_length(old);
                } else {
                    all_ring_start(f) = 0;
                }
            }
        },
        faces);
    if (faces > face_capacity) Kokkos::abort("update_face_rings: face capacity exceeded");
    const auto ring_start = Kokkos::subview(all_ring_start, std::make_pair(0, faces));
    const auto ring_length = Kokkos::subview(all_ring_length, std::make_pair(0, faces));
    const auto dirty = rings.dirty;
    const auto listed = rings.listed;

    // Dirty faces (a touched tetrahedron on their ring): ring_start
    // ~representative, listed once in `dirty`.
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
                    const int f = edge_face(mesh, start, t, mesh.vertex(t, ka), mesh.vertex(t, kb), representative);
                    if (f >= 0 && Kokkos::atomic_exchange(&ring_start(f), ~representative) >= 0) found[n++] = f;
                }
                if (n == 0) return;
                const int at = Kokkos::atomic_fetch_add(&listed(), n);
                for (int q = 0; q < n; ++q) dirty(at + q) = found[q];
                sum += n;
            },
            dirty_faces);
    }
    // The dirty faces' ring lengths.
    int dirty_entries = 0;
    Kokkos::parallel_reduce(
        "tet_mesh::dirty_lengths", Range(exec, 0, dirty_faces),
        KOKKOS_LAMBDA(int s, int& sum) {
            const int f = dirty(s);
            ring_length(f) = walk_edge(mesh, ~ring_start(f), owner(f), neighbour(f));
            sum += ring_length(f);
        },
        dirty_entries);

    const int capacity = static_cast<int>(rings.pool[0].extent(0));
    auto store = KOKKOS_LAMBDA(int f, int at, const Kokkos::View<int*, typename ExecSpace::memory_space>& pool) {
        if (at + ring_length(f) > capacity) Kokkos::abort("update_face_rings: ring capacity exceeded");
        walk_edge(mesh, ~ring_start(f), owner(f), neighbour(f), &pool(at));
        ring_start(f) = at;
    };
    if (!all && rings.used + dirty_entries <= capacity) {
        // Append the dirty rings.
        const auto pool = rings.pool[rings.in_use];
        const int base = rings.used;
        Kokkos::parallel_scan(
            "tet_mesh::append_rings", Range(exec, 0, dirty_faces),
            KOKKOS_LAMBDA(int s, int& sum, bool final) {
                const int f = dirty(s), length = ring_length(f);
                if (final) store(f, base + sum, pool);
                sum += length;
            });
        rings.used += dirty_entries;
    } else {
        // Rewrite every ring into the spare pool: copy the clean ones, walk the dirty ones.
        const auto from = rings.pool[rings.in_use];
        rings.in_use = 1 - rings.in_use;
        const auto pool = rings.pool[rings.in_use];
        int entries = 0;
        Kokkos::parallel_scan(
            "tet_mesh::compact_rings", Range(exec, 0, faces),
            KOKKOS_LAMBDA(int f, int& sum, bool final) {
                const int begin = ring_start(f), length = ring_length(f);
                if (final && begin < 0) store(f, sum, pool);
                if (final && begin >= 0) {
                    if (sum + length > capacity) Kokkos::abort("update_face_rings: ring capacity exceeded");
                    for (int r = 0; r < length; ++r) pool(sum + r) = from(begin + r);
                    ring_start(f) = sum;
                }
                sum += length;
            },
            entries);
        rings.used = entries;
    }
    rings.faces = faces;
    rings.current = now;
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
KOKKOS_INLINE_FUNCTION Vec3<double> circumcentre(const X& x, double box, const int* v, const Vec3<double>* d) {
    Vec3<double> c = detail::circumcentre(d[0], d[1], d[2]);
    // Nearly flat (sliver) tetrahedra: the double formula divides by a
    // tiny determinant. Redo it on the snapped grid, where numerator
    // and denominator are exact integers, and divide once at the end.
    const double volume6 = dot(d[0], cross(d[1], d[2]));
    if (!(Kokkos::abs(volume6) > 1e-4 * Kokkos::sqrt(dot(d[0], d[0]) * dot(d[1], d[1]) * dot(d[2], d[2])))) {
        namespace ep = exact_predicates;
        ep::Offset g[3];
        const ep::i64 g0[3] = {ep::snap(x(v[0], 0) / box), ep::snap(x(v[0], 1) / box), ep::snap(x(v[0], 2) / box)};
        for (int k = 0; k < 3; ++k)
            for (int a = 0; a < 3; ++a) g[k][a] = ep::delta(g0[a], ep::snap(x(v[k + 1], a) / box));
        auto cross128 = [](const ep::Offset& p, const ep::Offset& q, ep::i128* r) {
            r[0] = ep::i128(p[1]) * q[2] - ep::i128(p[2]) * q[1];
            r[1] = ep::i128(p[2]) * q[0] - ep::i128(p[0]) * q[2];
            r[2] = ep::i128(p[0]) * q[1] - ep::i128(p[1]) * q[0];
        };
        ep::i128 bc[3], ca[3], ab[3];
        cross128(g[1], g[2], bc);
        cross128(g[2], g[0], ca);
        cross128(g[0], g[1], ab);
        const ep::i128 la = ep::lift(g[0]), lb = ep::lift(g[1]), lc = ep::lift(g[2]);
        const ep::i128 det = ep::det3(g[0], g[1], g[2]);
        if (det != 0) {
            const double unit = box / ep::scale / (2 * double(det));
            c = {double(la * bc[0] + lb * ca[0] + lc * ab[0]) * unit,
                 double(la * bc[1] + lb * ca[1] + lc * ab[1]) * unit,
                 double(la * bc[2] + lb * ca[2] + lc * ab[2]) * unit};
        }
    }
    return c;
}

// 1. Circumcentres, one thread per tetrahedron: of the first `count` (all
// if negative), or of those listed in `only` (e.g. the ones a repair
// touched, at unchanged positions).
template <class ExecSpace, class X>
void circumcentres(const ExecSpace& exec, const X& x, double box,
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
            Vec3<double> d[3];
            for (int k = 0; k < 3; ++k)
                d[k] = {wrap(x(v[k + 1], 0) - x(v[0], 0), box), wrap(x(v[k + 1], 1) - x(v[0], 1), box),
                        wrap(x(v[k + 1], 2) - x(v[0], 2), box)};
            const Vec3<double> c = circumcentre(x, box, v, d);
            mesh.centre(t, 0) = x(v[0], 0) + c.x;
            mesh.centre(t, 1) = x(v[0], 1) + c.y;
            mesh.centre(t, 2) = x(v[0], 2) + c.z;
            mesh.centre(t, 3) = dot(c, c);
        });
}

// 2. Face area vectors (owner to neighbour), centroids (relative to the
// owner) and cell volumes, one thread per owned face.
template <class ExecSpace, class X>
void faces(const ExecSpace& exec, const X& x, double box,
           const Mesh<typename ExecSpace::memory_space>& mesh, const Geometry<typename ExecSpace::memory_space>& out) {
    Kokkos::deep_copy(exec, out.volume, 0.0);
    Kokkos::parallel_for(
        "tet_mesh::faces", Kokkos::RangePolicy<ExecSpace>(exec, 0, mesh.owner.extent(0)), KOKKOS_LAMBDA(int f) {
            const int i = mesh.owner(f), j = mesh.neighbour(f);
            const int begin = mesh.ring_start(f), end = begin + mesh.ring_length(f);
            auto vertex = [&](int m) {
                const int t = mesh.ring(m);
                return Vec3<double>{wrap(mesh.centre(t, 0) - x(i, 0), box),
                                    wrap(mesh.centre(t, 1) - x(i, 1), box),
                                    wrap(mesh.centre(t, 2) - x(i, 2), box)};
            };
            const Vec3<double> dj{wrap(x(j, 0) - x(i, 0), box), wrap(x(j, 1) - x(i, 1), box),
                                  wrap(x(j, 2) - x(i, 2), box)};
            const Vec3<double> first = vertex(begin);
            Vec3<double> previous = vertex(begin + 1), area{0, 0, 0}, moment{0, 0, 0};
            double weight = 0;
            for (int m = begin + 2; m < end; ++m) {
                const Vec3<double> v = vertex(m);
                const Vec3<double> twice = cross(previous - first, v - first);
                area = area + twice * 0.5;
                const double w = Kokkos::sqrt(dot(twice, twice));
                weight += w;
                moment = moment + (first + previous + v) * (w / 3.0);
                previous = v;
            }
            if (dot(area, dj) < 0) area = area * -1.0;
            const Vec3<double> centroid = weight > 0 ? moment * (1.0 / weight) : first;
            out.area(f, 0) = area.x;
            out.area(f, 1) = area.y;
            out.area(f, 2) = area.z;
            out.centroid(f, 0) = centroid.x;
            out.centroid(f, 1) = centroid.y;
            out.centroid(f, 2) = centroid.z;
            const double pyramid = Kokkos::sqrt(dot(area, area) * dot(dj, dj)) / 6.0;
            Kokkos::atomic_add(&out.volume(i), pyramid);
            Kokkos::atomic_add(&out.volume(j), pyramid);
        });
}

// 3. Local Delaunay check, one thread per tetrahedron: the far vertex of each
// neighbour must not lie inside this circumsphere. Writes 1 per failing
// tetrahedron into `invalid` and returns how many failed. Rounding-level
// cases count as valid here; deciding them exactly is the repair's job.
template <class ExecSpace, class X>
int check(const ExecSpace& exec, const X& x, double box,
          const Mesh<typename ExecSpace::memory_space>& mesh,
          const Kokkos::View<int*, typename ExecSpace::memory_space>& invalid) {
    int failed = 0;
    Kokkos::parallel_reduce(
        "tet_mesh::check", Kokkos::RangePolicy<ExecSpace>(exec, 0, mesh.vertex.extent(0)),
        KOKKOS_LAMBDA(int t, int& count) {
            const Vec3<double> c{mesh.centre(t, 0), mesh.centre(t, 1), mesh.centre(t, 2)};
            auto from_centre = [&](int v) {
                return Vec3<double>{wrap(x(v, 0) - c.x, box), wrap(x(v, 1) - c.y, box),
                                    wrap(x(v, 2) - c.z, box)};
            };
            const double r2 = mesh.centre(t, 3);
            int bad = 0;
            for (int k = 0; k < 4 && !bad; ++k) {
                const int other = mesh.adjacent(t, k);
                if (other < 0) continue;
                const int far = far_vertex(mesh, t, other);
                if (far < 0) continue;
                const Vec3<double> d = from_centre(far);
                bad = dot(d, d) < r2 * (1.0 - 1e-10);
            }
            invalid(t) = bad;
            count += bad;
        },
        failed);
    return failed;
}

}  // namespace tet_mesh
