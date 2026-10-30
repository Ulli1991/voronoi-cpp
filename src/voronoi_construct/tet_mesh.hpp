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
    // Owned faces (owner < neighbour), each a ring of tetrahedra in order.
    Kokkos::View<int*, MemorySpace> owner, neighbour, ring_start, ring;
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
template <class V>
KOKKOS_INLINE_FUNCTION void sorted_face(const V& v, int k, int* f) {
    for (int a = 0, b = 0; a < 4; ++a)
        if (a != k) f[b++] = v[a];
    if (f[0] > f[1]) Kokkos::kokkos_swap(f[0], f[1]);
    if (f[1] > f[2]) Kokkos::kokkos_swap(f[1], f[2]);
    if (f[0] > f[1]) Kokkos::kokkos_swap(f[0], f[1]);
}

// Tetrahedra around edge i-j from t, in order; calls take(u) for each, t
// first. Returns the number of steps (the ring length when it closes).
template <class M, class Take>
KOKKOS_INLINE_FUNCTION int walk_edge(const M& m, int t, int i, int j, Take&& take) {
    int current = t, before = -1, steps = 0;
    do {
        take(current);
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
    mesh.ring_start = ring_start;
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

// Face rings kept from step to step: face f of tetrahedron t's represented
// edges (in bit order) is face start(t) + rank; its ring of tetrahedra is
// ring[ring_start(f), ring_start(f + 1)).
template <class MemorySpace>
struct FaceRings {
    Kokkos::View<int*, MemorySpace> start, owner, neighbour, ring_start, ring;
    int faces = -1;  // -1: none yet, every face is walked
};

template <class MemorySpace>
FaceRings<MemorySpace> make_face_rings(int tets, int faces, int entries) {
    return {Kokkos::View<int*, MemorySpace>("face_rings_start", tets + 1),
            Kokkos::View<int*, MemorySpace>("face_rings_owner", faces),
            Kokkos::View<int*, MemorySpace>("face_rings_neighbour", faces),
            Kokkos::View<int*, MemorySpace>("face_rings_ring_start", faces + 1),
            Kokkos::View<int*, MemorySpace>("face_rings_ring", entries), -1};
}

// Rebuild the face rings in `out` from those of the previous step (`in`):
// a face keeps its ring when neither its representative nor any tetrahedron
// of its old ring was touched by a flip (touched(t) != 0), else its edge is
// walked. Sets mesh.owner/neighbour/ring_start/ring to `out`; returns the
// number of faces.
template <class ExecSpace>
int update_face_rings(const ExecSpace& exec, Mesh<typename ExecSpace::memory_space>& mesh, int count,
                      const Kokkos::View<char*, typename ExecSpace::memory_space>& touched,
                      const FaceRings<typename ExecSpace::memory_space>& in,
                      FaceRings<typename ExecSpace::memory_space>& out) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const auto edges = mesh.edges;
    const auto start = out.start;
    int faces = 0;
    Kokkos::parallel_scan(
        "tet_mesh::face_numbers", Range(exec, 0, count),
        KOKKOS_LAMBDA(int t, int& sum, bool last) {
            if (last) start(t) = sum;
            sum += Kokkos::Experimental::popcount_builtin(static_cast<unsigned>(edges(t)));
            if (last && t == count - 1) start(count) = sum;
        },
        faces);
    if (faces > static_cast<int>(out.owner.extent(0))) Kokkos::abort("update_face_rings: face capacity exceeded");
    const bool all = in.faces < 0;
    const int old_tets = all ? 0 : static_cast<int>(in.start.extent(0)) - 1;
    // Lengths (and owner, neighbour); ~length (negative) marks a face to walk.
    Kokkos::View<int*, Memory> length(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("face_length")),
                                      faces);
    const auto owner = out.owner, neighbour = out.neighbour;
    Kokkos::parallel_for(
        "tet_mesh::face_lengths", Range(exec, 0, count), KOKKOS_LAMBDA(int t) {
            int bits = edges(t);
            for (int f = start(t), rank = 0; bits; ++f, ++rank, bits &= bits - 1) {
                int ka, kb;
                edge_ends(Kokkos::Experimental::countr_zero_builtin(static_cast<unsigned>(bits)), ka, kb);
                int i = mesh.vertex(t, ka), j = mesh.vertex(t, kb);
                if (i > j) Kokkos::kokkos_swap(i, j);
                owner(f) = i;
                neighbour(f) = j;
                bool clean = !all && !touched(t) && t < old_tets;
                int old = -1;
                if (clean) {
                    old = in.start(t) + rank;
                    clean = old < in.start(t + 1) && in.owner(old) == i && in.neighbour(old) == j;
                }
                for (int r = clean ? in.ring_start(old) : 0; clean && r < in.ring_start(old + 1); ++r)
                    clean = !touched(in.ring(r));
                length(f) = clean ? in.ring_start(old + 1) - in.ring_start(old)
                                  : ~walk_edge(mesh, t, i, j, [](int) {});
            }
        });
    const auto ring_start = out.ring_start;
    int entries = 0;
    Kokkos::parallel_scan(
        "tet_mesh::face_ring_start", Range(exec, 0, faces),
        KOKKOS_LAMBDA(int f, int& sum, bool last) {
            if (last) ring_start(f) = sum;
            sum += length(f) >= 0 ? length(f) : ~length(f);
            if (last && f == faces - 1) ring_start(faces) = sum;
        },
        entries);
    if (entries > static_cast<int>(out.ring.extent(0))) Kokkos::abort("update_face_rings: ring capacity exceeded");
    const auto ring = out.ring;
    Kokkos::parallel_for(
        "tet_mesh::face_rings", Range(exec, 0, count), KOKKOS_LAMBDA(int t) {
            int bits = edges(t);
            for (int f = start(t), rank = 0; bits; ++f, ++rank, bits &= bits - 1) {
                int at = ring_start(f);
                if (length(f) >= 0) {  // copy the kept ring
                    const int old = in.start(t) + rank;
                    for (int r = in.ring_start(old); r < in.ring_start(old + 1); ++r) ring(at++) = in.ring(r);
                } else {
                    walk_edge(mesh, t, owner(f), neighbour(f), [&](int u) { ring(at++) = u; });
                }
            }
        });
    out.faces = faces;
    mesh.owner = Kokkos::subview(owner, std::make_pair(0, faces));
    mesh.neighbour = Kokkos::subview(neighbour, std::make_pair(0, faces));
    mesh.ring_start = Kokkos::subview(ring_start, std::make_pair(0, faces + 1));
    mesh.ring = ring;
    return faces;
}

// 1. Circumcentres, one thread per tetrahedron.
template <class ExecSpace, class X>
void circumcentres(const ExecSpace& exec, const X& x,
                   double box, const Mesh<typename ExecSpace::memory_space>& mesh, int count = -1) {
    Kokkos::parallel_for(
        "tet_mesh::circumcentres",
        Kokkos::RangePolicy<ExecSpace>(exec, 0, count < 0 ? static_cast<int>(mesh.vertex.extent(0)) : count),
        KOKKOS_LAMBDA(int t) {
            const int v0 = mesh.vertex(t, 0);
            Vec3<double> d[3];
            for (int k = 0; k < 3; ++k) {
                const int v = mesh.vertex(t, k + 1);
                d[k] = {wrap(x(v, 0) - x(v0, 0), box), wrap(x(v, 1) - x(v0, 1), box),
                        wrap(x(v, 2) - x(v0, 2), box)};
            }
            Vec3<double> c = detail::circumcentre(d[0], d[1], d[2]);
            // Nearly flat (sliver) tetrahedra: the double formula divides by a
            // tiny determinant. Redo it on the snapped grid, where numerator
            // and denominator are exact integers, and divide once at the end.
            const double volume6 = dot(d[0], cross(d[1], d[2]));
            if (!(Kokkos::abs(volume6) > 1e-4 * Kokkos::sqrt(dot(d[0], d[0]) * dot(d[1], d[1]) * dot(d[2], d[2])))) {
                namespace ep = exact_predicates;
                ep::Offset g[3];
                const ep::i64 g0[3] = {ep::snap(x(v0, 0) / box), ep::snap(x(v0, 1) / box), ep::snap(x(v0, 2) / box)};
                for (int k = 0; k < 3; ++k) {
                    const int v = mesh.vertex(t, k + 1);
                    for (int a = 0; a < 3; ++a) g[k][a] = ep::delta(g0[a], ep::snap(x(v, a) / box));
                }
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
            mesh.centre(t, 0) = x(v0, 0) + c.x;
            mesh.centre(t, 1) = x(v0, 1) + c.y;
            mesh.centre(t, 2) = x(v0, 2) + c.z;
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
            const int begin = mesh.ring_start(f), end = mesh.ring_start(f + 1);
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
