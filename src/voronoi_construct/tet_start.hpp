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
// wrong by exact clipping: the same clipping, but every vertex is named by
// the three particles whose bisectors meet there, and every cut is the exact
// (perturbed) in-sphere sign, so cells agree with each other by construction.
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

constexpr int max_planes = 48, max_vertices = 64;    // as the clipping kernel
constexpr int big_planes = 128, big_vertices = 256;  // for the rare cells that overflow those
constexpr int box_planes = 6;
constexpr int ring_buffer = 32;

// One cell under exact construction, in grid steps relative to its
// generator. Vertex v is the circumcentre of the generator and the
// particles of its three planes; a particle at e lies inside that sphere iff
// e.w + |e|^2 o has the opposite sign of o (w and o are exact integers,
// stored rounded once).
template <int Planes, int Vertices>
struct Cell {
    static_assert(Planes <= 255, "plane indices are bytes");
    static constexpr int planes_capacity = Planes, vertices_capacity = Vertices;
    ep::Offset offset[Planes];  // generator to the plane's particle
    int id[Planes];             // that particle; box planes are virtual particles beyond n
    std::uint8_t corner[Vertices][3];
    double w[Vertices][3], o[Vertices];
    int planes = 0, vertices = 0;
    double radius2 = 0;  // bound on the squared distance of every vertex
};

// Vertex v at the meeting point of planes a, b, c (in walking order);
// false if they do not meet in one point.
template <class C>
KOKKOS_INLINE_FUNCTION bool set_vertex(C& cell, int v, int a, int b, int c) {
    const ep::Offset &A = cell.offset[a], &B = cell.offset[b], &D = cell.offset[c];
    auto cross = [](const ep::Offset& p, const ep::Offset& q, ep::i128* r) {
        r[0] = ep::i128(p[1]) * q[2] - ep::i128(p[2]) * q[1];
        r[1] = ep::i128(p[2]) * q[0] - ep::i128(p[0]) * q[2];
        r[2] = ep::i128(p[0]) * q[1] - ep::i128(p[1]) * q[0];
    };
    ep::i128 bd[3], ad[3], ab[3];
    cross(B, D, bd);
    cross(A, D, ad);
    cross(A, B, ab);
    const ep::i128 o = A[0] * bd[0] + A[1] * bd[1] + A[2] * bd[2];
    if (o == 0) return false;
    const ep::i128 la = ep::lift(A), lb = ep::lift(B), ld = ep::lift(D);
    for (int k = 0; k < 3; ++k) cell.w[v][k] = double(-la * bd[k] + lb * ad[k] - ld * ab[k]);
    cell.o[v] = double(o);
    cell.corner[v][0] = static_cast<std::uint8_t>(a);
    cell.corner[v][1] = static_cast<std::uint8_t>(b);
    cell.corner[v][2] = static_cast<std::uint8_t>(c);
    return true;
}

// Bound on the squared distance of vertex v from the generator: the
// circumcentre is -w / (2 o).
template <class C>
KOKKOS_INLINE_FUNCTION double radius2(const C& cell, int v) {
    const double* w = cell.w[v];
    return (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]) / (4 * cell.o[v] * cell.o[v]) * (1 + 1e-12);
}

// Whether particle `id` at e lies inside the circumsphere of vertex v; the
// filtered value decides unless it is within its rounding bound.
template <class C>
KOKKOS_INLINE_FUNCTION bool inside(const C& cell, int v, const ep::Offset& e, int id, int self) {
    const double ex = double(e[0]), ey = double(e[1]), ez = double(e[2]);
    const double le = ex * ex + ey * ey + ez * ez;
    const double t0 = ex * cell.w[v][0], t1 = ey * cell.w[v][1], t2 = ez * cell.w[v][2], t3 = le * cell.o[v];
    const double value = t0 + t1 + t2 + t3;
    const double bound =
        16 * ep::epsilon * (Kokkos::abs(t0) + Kokkos::abs(t1) + Kokkos::abs(t2) + Kokkos::abs(t3));
    if (value > bound) return cell.o[v] < 0;
    if (value < -bound) return cell.o[v] > 0;
    const int a = cell.corner[v][0], b = cell.corner[v][1], c = cell.corner[v][2];
    const ep::Offset p[5] = {{0, 0, 0}, cell.offset[a], cell.offset[b], cell.offset[c], e};
    const int ids[5] = {self, cell.id[a], cell.id[b], cell.id[c], id};
    return ep::in_sphere(p, ids);
}

// The cube of half-width h around the generator, bounded by six virtual
// particles at distance 2h (ids n..n+5); corner order as the clipping kernel.
template <class C>
KOKKOS_INLINE_FUNCTION void start_cube(C& cell, ep::i64 h, int n) {
    for (int a = 0; a < 3; ++a)
        for (int s = 0; s < 2; ++s) {
            ep::Offset& d = cell.offset[2 * a + s];
            d = {0, 0, 0};
            d[a] = s ? -2 * h : 2 * h;
            cell.id[2 * a + s] = n + 2 * a + s;
        }
    cell.planes = box_planes;
    cell.radius2 = 0;
    for (int c = 0; c < 8; ++c) {
        const int sx = c & 1, sy = (c >> 1) & 1, sz = (c >> 2) & 1;
        const bool odd = (sx + sy + sz) % 2 == 1;
        set_vertex(cell, c, sx, odd ? 4 + sz : 2 + sy, odd ? 2 + sy : 4 + sz);
        cell.radius2 = Kokkos::max(cell.radius2, radius2(cell, c));
    }
    cell.vertices = 8;
}

// Drop planes no vertex uses any more and renumber the rest (box planes keep
// their indices). Returns the freed count.
template <class C>
KOKKOS_INLINE_FUNCTION int compact_planes(C& cell) {
    constexpr int capacity = C::planes_capacity;
    bool used[capacity];
    for (int p = 0; p < cell.planes; ++p) used[p] = p < box_planes;
    for (int v = 0; v < cell.vertices; ++v)
        for (int k = 0; k < 3; ++k) used[cell.corner[v][k]] = true;
    std::uint8_t renumber[capacity];
    int planes = 0;
    for (int p = 0; p < cell.planes; ++p) {
        if (!used[p]) continue;
        renumber[p] = static_cast<std::uint8_t>(planes);
        cell.offset[planes] = cell.offset[p];
        cell.id[planes] = cell.id[p];
        ++planes;
    }
    for (int v = 0; v < cell.vertices; ++v)
        for (int k = 0; k < 3; ++k) cell.corner[v][k] = renumber[cell.corner[v][k]];
    const int freed = cell.planes - planes;
    cell.planes = planes;
    return freed;
}

// Clip by the bisector of the generator `self` and particle `id` at offset e:
// remove the vertices whose circumsphere holds it and close the hole with a
// new face (as shell_voronoi::detail::clip, with exact decisions).
template <class C>
KOKKOS_INLINE_FUNCTION int clip(C& cell, const ep::Offset& e, int id, int self) {
    using shell_voronoi::ok;
    constexpr int max_holes = C::vertices_capacity / 2;
    bool cut[C::vertices_capacity];
    bool any = false;
    for (int v = 0; v < cell.vertices; ++v) {
        cut[v] = inside(cell, v, e, id, self);
        any |= cut[v];
    }
    if (!any) return ok;
    if (cell.planes == C::planes_capacity && compact_planes(cell) == 0) return shell_voronoi::plane_overflow;

    // Directed edges of the removed vertices not cancelled by a reversed twin
    // bound the hole.
    std::uint8_t hole[max_holes][2];
    int holes = 0;
    for (int v = 0; v < cell.vertices; ++v) {
        if (!cut[v]) continue;
        for (int k = 0; k < 3; ++k) {
            const std::uint8_t a = cell.corner[v][k], b = cell.corner[v][(k + 1) % 3];
            int twin = -1;
            for (int h = 0; h < holes; ++h)
                if (hole[h][0] == b && hole[h][1] == a) twin = h;
            if (twin >= 0) {
                hole[twin][0] = hole[holes - 1][0];
                hole[twin][1] = hole[holes - 1][1];
                --holes;
            } else {
                if (holes == max_holes) return shell_voronoi::vertex_overflow;
                hole[holes][0] = a;
                hole[holes][1] = b;
                ++holes;
            }
        }
    }
    const int plane = cell.planes++;
    cell.offset[plane] = e;
    cell.id[plane] = id;
    int kept = 0;
    for (int v = 0; v < cell.vertices; ++v) {
        if (cut[v]) continue;
        for (int k = 0; k < 3; ++k) {
            cell.corner[kept][k] = cell.corner[v][k];
            cell.w[kept][k] = cell.w[v][k];
        }
        cell.o[kept] = cell.o[v];
        ++kept;
    }
    if (kept + holes > C::vertices_capacity) return shell_voronoi::vertex_overflow;
    double r2 = 0;
    for (int v = 0; v < kept; ++v) r2 = Kokkos::max(r2, radius2(cell, v));
    for (int h = 0; h < holes; ++h) {
        if (!set_vertex(cell, kept, hole[h][0], hole[h][1], plane)) return shell_voronoi::degenerate;
        r2 = Kokkos::max(r2, radius2(cell, kept));
        ++kept;
    }
    cell.vertices = kept;
    cell.radius2 = r2;
    return ok;
}

struct Candidate {
    double distance2;  // order and reach, box units
    int slot;
    std::int8_t image[3];
};

// Exact cells for the slots todo[0, count): status, faces and rings (volumes
// and face geometry are left as they are). Same walk as the clipping
// kernels (shell_voronoi::Stencil), with the reach widened by the snapping.
template <int Planes, int Vertices, class ExecSpace>
void exact_cells(const ExecSpace& exec, const shell_voronoi::Grid<typename ExecSpace::memory_space>& grid,
                 const ep::Points<typename ExecSpace::memory_space>& point,
                 const Kokkos::View<int*, typename ExecSpace::memory_space>& todo, int count,
                 const shell_voronoi::Cells<typename ExecSpace::memory_space>& out) {
    const int n = grid.n;
    const double box = grid.box, unit = grid.box / ep::scale;
    const auto positions = grid.positions;
    const auto half = static_cast<ep::i64>(shell_voronoi::detail::start_half_width<double>(box) / unit);
    constexpr ep::i64 period = ep::i64(1) << 30;
    const shell_voronoi::Stencil<typename ExecSpace::memory_space> stencil = shell_voronoi::make_stencil(exec);
    Kokkos::parallel_for(
        "tet_start::exact_cells", Kokkos::RangePolicy<ExecSpace>(exec, 0, count), KOKKOS_LAMBDA(int i) {
            using namespace shell_voronoi;
            const int t = todo(i);
            const Vec3<double> p{positions(t, 0), positions(t, 1), positions(t, 2)};
            Cell<Planes, Vertices> cell;
            start_cube(cell, half, n);
            // A particle cuts only within twice the largest vertex distance;
            // 8 grid steps cover the snapping of both ends.
            auto reach2 = [&] {
                const double r = (2 * Kokkos::sqrt(cell.radius2) + 8) * unit;
                return r * r;
            };
            int status = ok;
            bool certified = false;
            Candidate buffer[ring_buffer];
            int buffered = 0;
            auto flush = [&]() {
                for (int a = 1; a < buffered; ++a) {
                    const Candidate c = buffer[a];
                    int b = a;
                    for (; b > 0 && buffer[b - 1].distance2 > c.distance2; --b) buffer[b] = buffer[b - 1];
                    buffer[b] = c;
                }
                for (int a = 0; a < buffered && status == ok; ++a) {
                    const Candidate& c = buffer[a];
                    if (c.distance2 >= reach2()) break;
                    ep::Offset e;
                    for (int k = 0; k < 3; ++k) e[k] = point(c.slot, k) + c.image[k] * period - point(t, k);
                    status = clip(cell, e, c.slot, t);
                }
                buffered = 0;
            };
            int level = grid.level(t), home[3], finer = -1;
            home_cell(p, box, level, home);
            while (status == ok) {
                const double width = box / (1 << level);
                const int reach = level_reach(level), entries = stencil.ends(reach);
                for (int first = 0; first < entries && status == ok && !certified; first += ring_buffer) {
                    for (int e = first; e < Kokkos::min(first + ring_buffer, entries) && status == ok; ++e) {
                        if (double(stencil.bound2(e)) * width * width >= reach2()) {
                            certified = true;
                            break;
                        }
                        int begin = 0, end = 0, packed = 0;
                        if (!stencil_cell(grid, level, home, finer, stencil.offsets(e), p, reach2(), begin, end, packed))
                            continue;
                        const int shift[3] = {image(packed, 0), image(packed, 1), image(packed, 2)};
                        for (int q = begin; q < end; ++q) {
                            if (q == t) continue;
                            const double ddx = positions(q, 0) + shift[0] * box - p.x;
                            const double ddy = positions(q, 1) + shift[1] * box - p.y;
                            const double ddz = positions(q, 2) + shift[2] * box - p.z;
                            const double d2 = ddx * ddx + ddy * ddy + ddz * ddz;
                            if (d2 >= reach2()) continue;
                            if (buffered == ring_buffer) flush();
                            using Image = std::int8_t;
                            buffer[buffered++] = {d2, q, {Image(shift[0]), Image(shift[1]), Image(shift[2])}};
                        }
                    }
                    flush();
                }
                if (status != ok || certified) break;
                // Cells outside the stencil are at least `reach` widths away.
                if (reach * width * reach * width >= reach2()) {
                    certified = true;
                    break;
                }
                if (level == 2) break;  // no coarser grid fits the box
                finer = (home[0] & 1) | (home[1] & 1) << 1 | (home[2] & 1) << 2;
                for (int k = 0; k < 3; ++k) home[k] >>= 1;
                --level;
            }
            if (status == ok && !certified) status = uncertified;
            for (int v = 0; v < cell.vertices && status == ok; ++v)
                for (int k = 0; k < 3; ++k)
                    if (cell.corner[v][k] < box_planes) status = uncertified;

            // Rings: a new block in the pool, at most a neighbour, a length
            // and one entry per vertex for each owned face.
            long long ring_at = 0;
            if (status == ok) {
                int need = 0;
                for (int plane = box_planes; plane < cell.planes; ++plane) need += 2 * (cell.id[plane] > t);
                for (int v = 0; v < cell.vertices; ++v)
                    for (int k = 0; k < 3; ++k) need += cell.id[cell.corner[v][k]] > t;
                if (!claim_rings(out, t, need)) status = face_overflow;
                ring_at = out.ring_start(t);
            }

            // Faces: every plane with a vertex left, its ring walked in order.
            int faces = 0, owned = 0, used = 0;
            for (int plane = box_planes; plane < cell.planes && status == ok; ++plane) {
                int start = -1;
                for (int v = 0; v < cell.vertices && start < 0; ++v)
                    for (int k = 0; k < 3; ++k)
                        if (cell.corner[v][k] == plane) start = v;
                if (start < 0) continue;
                ++faces;
                const int neighbour = cell.id[plane];
                const bool own = neighbour > t;
                int current = start, length = 0;
                do {
                    int at = 0;
                    while (cell.corner[current][at] != plane) ++at;
                    const std::uint8_t last = cell.corner[current][(at + 2) % 3];
                    if (own) out.rings(ring_at + used + 2 + length) = cell.id[last];
                    ++length;
                    int next = -1;
                    for (int w = 0; w < cell.vertices && next < 0; ++w)
                        for (int k = 0; k < 3; ++k)
                            if (cell.corner[w][k] == plane && cell.corner[w][(k + 1) % 3] == last) next = w;
                    if (next < 0 || length > Vertices) {
                        status = degenerate;
                        break;
                    }
                    current = next;
                } while (current != start);
                if (status != ok || !own) continue;
                out.rings(ring_at + used) = neighbour;
                out.rings(ring_at + used + 1) = length;
                used += 2 + length;
                ++owned;
            }
            out.status(t) = status;
            out.face_count(t) = status == ok ? faces : 0;
            out.owned_count(t) = status == ok ? owned : 0;
        });
}

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

// Snapped positions for the exact predicates (as tet_repair uses them).
template <class ExecSpace, class X>
void snap(const ExecSpace& exec, const X& x, double box, const ep::Points<typename ExecSpace::memory_space>& point) {
    Kokkos::parallel_for(
        "tet_start::snap", Kokkos::RangePolicy<ExecSpace>(exec, 0, x.extent(0)), KOKKOS_LAMBDA(int i) {
            for (int a = 0; a < 3; ++a) point(i, a) = ep::snap(x(i, a) / box);
        });
}

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
// result is not exact: build again with a pool of ring_entries_needed().
template <class ExecSpace>
Stats build(const ExecSpace& exec, const shell_voronoi::Grid<typename ExecSpace::memory_space>& grid,
            const shell_voronoi::Cells<typename ExecSpace::memory_space>& cells,
            tet_mesh::Mesh<typename ExecSpace::memory_space>& mesh) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const int n = grid.n;
    Stats stats;
    ep::Points<Memory> point("tet_start_point", n);
    snap(exec, grid.positions, grid.box, point);
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
        Kokkos::parallel_scan(
            "tet_start::list", Range(exec, 0, n),
            KOKKOS_LAMBDA(int i, int& at, bool last) {
                if (!grown(i)) return;
                if (last) todo(at) = i;
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
        detail::exact_cells<detail::max_planes, detail::max_vertices>(exec, grid, point, todo, listed, cells);
        // Cells that overflowed the usual capacities, once more with big ones.
        int overflowed = 0;
        Kokkos::parallel_scan(
            "tet_start::overflowed", Range(exec, 0, listed),
            KOKKOS_LAMBDA(int i, int& at, bool last) {
                const int s = status(todo(i));
                if (s != shell_voronoi::vertex_overflow && s != shell_voronoi::plane_overflow) return;
                if (last) grown(at) = todo(i);
                ++at;
            },
            overflowed);
        if (overflowed > 0)
            detail::exact_cells<detail::big_planes, detail::big_vertices>(exec, grid, point, grown, overflowed, cells);
        stats.patched += listed;
        Kokkos::deep_copy(exec, marked, 0);
    }
    return stats;
}

}  // namespace tet_start
