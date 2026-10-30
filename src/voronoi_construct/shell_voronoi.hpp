#pragma once

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <stdexcept>

// Fused natural-neighbour Voronoi construction in a periodic box.
//
// Each cell is built by walking the cell grid outward in rings of grid cells
// around its generator and clipping with every particle that can still cut
// it. A particle at distance d cuts the cell only if d < 2R, where R is the
// largest distance from the generator to a current vertex; the walk stops at
// the first ring lying entirely beyond 2R, which certifies the cell. There is
// no candidate table and no global sort: each ring's particles are clipped
// nearest-first from a small private buffer.
//
// Output is lean: every cell stores its volume and centroid, and each face is
// stored once, by the cell with the lower slot index (its owner). A flux
// computed on that single face and applied with opposite signs conserves
// exactly. Slots, positions and cell offsets come from a built
// neighbor_search::DenseCellSearch.
namespace shell_voronoi {

enum Status : int {
    ok = 0,
    uncertified = 1,      // the walk reached the largest ring the box allows
    plane_overflow = 2,
    vertex_overflow = 3,
    face_overflow = 4,    // more owned faces than the output can hold
    degenerate = 5,
};
constexpr int status_count = 6;

constexpr int max_planes = 48;    // cutting planes, including the 6 box planes
constexpr int max_vertices = 64;  // one 64-bit mask marks vertices to remove
constexpr int big_planes = 128, big_vertices = 256;  // for the rare cells that overflow those
constexpr int ring_buffer = 32;
constexpr int max_owned_faces = 32;
constexpr int box_planes = 6;

template <class Real>
struct Vec3 {
    Real x, y, z;
};

template <class Real>
KOKKOS_INLINE_FUNCTION Vec3<Real> operator+(Vec3<Real> a, Vec3<Real> b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
template <class Real>
KOKKOS_INLINE_FUNCTION Vec3<Real> operator-(Vec3<Real> a, Vec3<Real> b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
template <class Real>
KOKKOS_INLINE_FUNCTION Vec3<Real> operator*(Vec3<Real> a, Real s) {
    return {a.x * s, a.y * s, a.z * s};
}
template <class Real>
KOKKOS_INLINE_FUNCTION Real dot(Vec3<Real> a, Vec3<Real> b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
template <class Real>
KOKKOS_INLINE_FUNCTION Vec3<Real> cross(Vec3<Real> a, Vec3<Real> b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
template <class To, class From>
KOKKOS_INLINE_FUNCTION Vec3<To> convert(Vec3<From> a) {
    return {static_cast<To>(a.x), static_cast<To>(a.y), static_cast<To>(a.z)};
}

// Read-only inputs, all in slot order. Grid cells must be cubes.
template <class MemorySpace>
struct Grid {
    Kokkos::View<double* [3], MemorySpace> positions;
    Kokkos::View<int*, MemorySpace> cell_offsets;
    int n = 0;
    int cells_per_side = 1;
    double cell_width = 1.0;
    double box = 1.0;
    // Cartesian cell key -> its place in cell_offsets when cells are stored
    // in another order (Morton); size 0 = Cartesian order.
    Kokkos::View<int*, MemorySpace> cell_rank;

    KOKKOS_INLINE_FUNCTION int rank(int key) const { return cell_rank.extent(0) ? cell_rank(key) : key; }
};

// Per-cell results. Face arrays are [cell, owned face]; area vectors point
// out of the owning cell, centroids are relative to the owner's generator.
template <class MemorySpace>
struct Cells {
    Kokkos::View<int*, MemorySpace> status;
    Kokkos::View<int*, MemorySpace> face_count;   // all faces of the cell
    Kokkos::View<int*, MemorySpace> owned_count;  // faces stored below
    Kokkos::View<int**, MemorySpace> neighbour;
    Kokkos::View<double** [3], MemorySpace> area;
    Kokkos::View<double** [3], MemorySpace> face_centroid;
    Kokkos::View<double*, MemorySpace> volume;
    Kokkos::View<double* [3], MemorySpace> centroid;
    // Optional: every natural neighbour of a cell (slots, face_count of them),
    // the state a warm rebuild starts from. Size 0 = off.
    Kokkos::View<int**, MemorySpace> all_neighbours;
    // Optional: for each owned face, in order, its ring length k followed by
    // the k slots whose bisectors bound it, so that vertex m of the face is
    // equidistant from the cell, the face neighbour, ring[m] and ring[m + 1].
    // Size 0 = off.
    Kokkos::View<int**, MemorySpace> rings;
    // Optional diagnostics, size 0 = off: |sum of area vectors| / sum of
    // |area|, and work per cell (rings walked, candidates clipped, cuts).
    Kokkos::View<double*, MemorySpace> closure;
    Kokkos::View<int* [3], MemorySpace> work;
    // Optional, size 0 = off: clock cycles per phase of the warp kernel when
    // built with SHELL_WARP_PHASES (walk, fetch, sort, clips that do not cut,
    // clips that cut, faces).
    Kokkos::View<long long* [6], MemorySpace> phase_cycles;

    Cells() = default;
    Cells(int n, bool with_diagnostics)
        : status("cell_status", n),
          face_count("cell_face_count", n),
          owned_count("cell_owned_count", n),
          neighbour("face_neighbour", n, max_owned_faces),
          area("face_area", n, max_owned_faces),
          face_centroid("face_centroid", n, max_owned_faces),
          volume("cell_volume", n),
          centroid("cell_centroid", n),
          closure("cell_closure", with_diagnostics ? n : 0),
          work("cell_work", with_diagnostics ? n : 0) {}
};

namespace detail {

// Private state of one cell under construction. A vertex is the meeting point
// of three planes, stored in an order that walks each face consistently.
// Planes and Vertices are capacities: the defaults fit almost every cell;
// the rare larger ones are rebuilt with big ones (big_planes, big_vertices).
template <class Real, int Planes = max_planes, int Vertices = max_vertices>
struct Polyhedron {
    static_assert(Planes <= 255, "plane indices are bytes");
    static constexpr int planes_capacity = Planes, vertices_capacity = Vertices;
    Vec3<Real> normal[Planes];  // the cell is {x : normal . x <= offset}
    Real offset[Planes];
    int neighbour[Planes];
    Vec3<Real> vertex[Vertices];
    std::uint8_t corner[Vertices][3];
    int planes = 0;
    int vertices = 0;
    Real radius2 = 0;  // largest squared vertex distance from the generator
};

template <class Cell, class Real>
KOKKOS_INLINE_FUNCTION bool intersect(const Cell& cell, int a, int b, int c,
                                      Vec3<Real>& x) {
    const Vec3<Real> bc = cross(cell.normal[b], cell.normal[c]);
    const Vec3<Real> ca = cross(cell.normal[c], cell.normal[a]);
    const Vec3<Real> ab = cross(cell.normal[a], cell.normal[b]);
    const Real det = dot(cell.normal[a], bc);
    if (!(Kokkos::abs(det) > Real(0))) return false;
    x = (bc * cell.offset[a] + ca * cell.offset[b] + ab * cell.offset[c]) * (Real(1) / det);
    return true;
}

// Cube of half-width h around the generator; corners keep the orientation
// convention of local_voronoi.hpp.
template <class Cell, class Real>
KOKKOS_INLINE_FUNCTION void start_cube(Cell& cell, Real h) {
    const Vec3<Real> axes[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int a = 0; a < 3; ++a) {
        cell.normal[2 * a] = axes[a];
        cell.normal[2 * a + 1] = axes[a] * Real(-1);
        cell.offset[2 * a] = cell.offset[2 * a + 1] = h;
        cell.neighbour[2 * a] = cell.neighbour[2 * a + 1] = -1;
    }
    cell.planes = box_planes;
    for (int c = 0; c < 8; ++c) {
        const int sx = c & 1, sy = (c >> 1) & 1, sz = (c >> 2) & 1;
        const bool odd = (sx + sy + sz) % 2 == 1;
        const std::uint8_t px = sx, py = 2 + sy, pz = 4 + sz;
        cell.corner[c][0] = px;
        cell.corner[c][1] = odd ? pz : py;
        cell.corner[c][2] = odd ? py : pz;
        cell.vertex[c] = {sx ? -h : h, sy ? -h : h, sz ? -h : h};
    }
    cell.vertices = 8;
    cell.radius2 = 3 * h * h;
}

// Drop planes no vertex uses any more (cut away by later planes) and renumber
// the rest. The six box planes keep their indices. Returns the freed count.
template <class Cell>
KOKKOS_INLINE_FUNCTION int compact_planes(Cell& cell) {
    constexpr int capacity = Cell::planes_capacity;
    bool used[capacity];
    for (int p = 0; p < cell.planes; ++p) used[p] = p < box_planes;
    for (int v = 0; v < cell.vertices; ++v)
        for (int k = 0; k < 3; ++k) used[cell.corner[v][k]] = true;
    std::uint8_t renumber[capacity];
    int planes = 0;
    for (int p = 0; p < cell.planes; ++p) {
        if (!used[p]) continue;
        renumber[p] = static_cast<std::uint8_t>(planes);
        cell.normal[planes] = cell.normal[p];
        cell.offset[planes] = cell.offset[p];
        cell.neighbour[planes] = cell.neighbour[p];
        ++planes;
    }
    for (int v = 0; v < cell.vertices; ++v)
        for (int k = 0; k < 3; ++k) cell.corner[v][k] = renumber[cell.corner[v][k]];
    const int freed = cell.planes - planes;
    cell.planes = planes;
    return freed;
}

// Clip by the bisector of the generator and a particle at offset d. Returns
// ok when the plane does not cut, or after the cut has been applied.
template <class Cell, class Real>
KOKKOS_INLINE_FUNCTION int clip(Cell& cell, Vec3<Real> d, int neighbour) {
    constexpr int max_holes = Cell::vertices_capacity / 2;
    const Real offset = Real(0.5) * dot(d, d);
    const Real tolerance =
        Real(64) * Kokkos::Experimental::epsilon_v<Real> * Kokkos::sqrt(dot(d, d) * cell.radius2);
    bool outside[Cell::vertices_capacity];
    bool any = false;
    for (int v = 0; v < cell.vertices; ++v) {
        outside[v] = dot(d, cell.vertex[v]) - offset > tolerance;
        any |= outside[v];
    }
    if (!any) return ok;
    if (cell.planes == Cell::planes_capacity && compact_planes(cell) == 0) return plane_overflow;

    // The removed vertices' directed edges that are not cancelled by a
    // reversed twin form the boundary of the hole the new face closes.
    std::uint8_t hole[max_holes][2];
    int holes = 0;
    for (int v = 0; v < cell.vertices; ++v) {
        if (!outside[v]) continue;
        for (int e = 0; e < 3; ++e) {
            const std::uint8_t a = cell.corner[v][e], b = cell.corner[v][(e + 1) % 3];
            int twin = -1;
            for (int h = 0; h < holes; ++h)
                if (hole[h][0] == b && hole[h][1] == a) twin = h;
            if (twin >= 0) {
                hole[twin][0] = hole[holes - 1][0];
                hole[twin][1] = hole[holes - 1][1];
                --holes;
            } else {
                if (holes == max_holes) return vertex_overflow;
                hole[holes][0] = a;
                hole[holes][1] = b;
                ++holes;
            }
        }
    }

    const int plane = cell.planes++;
    cell.normal[plane] = d;
    cell.offset[plane] = offset;
    cell.neighbour[plane] = neighbour;

    int kept = 0;
    for (int v = 0; v < cell.vertices; ++v) {
        if (outside[v]) continue;
        cell.vertex[kept] = cell.vertex[v];
        for (int k = 0; k < 3; ++k) cell.corner[kept][k] = cell.corner[v][k];
        ++kept;
    }
    if (kept + holes > Cell::vertices_capacity) return vertex_overflow;
    if (kept + holes < 4) return degenerate;
    Real radius2 = 0;
    for (int v = 0; v < kept; ++v) radius2 = Kokkos::max(radius2, dot(cell.vertex[v], cell.vertex[v]));
    for (int h = 0; h < holes; ++h) {
        Vec3<Real> x;
        if (!intersect(cell, hole[h][0], hole[h][1], plane, x)) return degenerate;
        cell.vertex[kept] = x;
        cell.corner[kept][0] = hole[h][0];
        cell.corner[kept][1] = hole[h][1];
        cell.corner[kept][2] = static_cast<std::uint8_t>(plane);
        radius2 = Kokkos::max(radius2, dot(x, x));
        ++kept;
    }
    cell.vertices = kept;
    cell.radius2 = radius2;
    return ok;
}

struct Candidate {
    float distance2;  // sort key only; clipping recomputes the offset exactly
    int slot;
    std::int8_t image[3];  // periodic image of the slot, in box lengths
};

}  // namespace detail

namespace detail {

// Which cells a pass builds.
enum class Pass { all, failed, overflowed };

template <class Real, int Planes, int Vertices, class ExecSpace>
void construct_pass(const ExecSpace& exec, const Grid<typename ExecSpace::memory_space>& grid,
                    const Cells<typename ExecSpace::memory_space>& out, Pass pass) {
    const int n = grid.n, side = grid.cells_per_side;
    const double width = grid.cell_width, box = grid.box;
    if (n <= 0 || side < 3 || out.status.extent(0) != static_cast<std::size_t>(n))
        throw std::invalid_argument("shell_voronoi::construct: invalid grid or output size");
    const auto positions = grid.positions;
    const auto offsets = grid.cell_offsets;
    const bool with_closure = out.closure.extent(0) > 0;
    // Rings beyond this would reach a cell from both sides of the box.
    const int max_ring = (side - 1) / 2;
    // The starting cube must lie well inside half the box so that every
    // candidate that can cut the cell is unique in the periodic image.
    const Real half_width = static_cast<Real>(Kokkos::min(0.25 * box, (max_ring + 1) * width));

    Kokkos::parallel_for(
        "shell_voronoi::construct", Kokkos::RangePolicy<ExecSpace>(exec, 0, n),
        KOKKOS_LAMBDA(int t) {
            const int before = out.status(t);
            if ((pass == Pass::failed && before == ok) ||
                (pass == Pass::overflowed && before != vertex_overflow && before != plane_overflow))
                return;
            const Vec3<double> p{positions(t, 0), positions(t, 1), positions(t, 2)};
            int home[3];
            double inside = width;  // distance to the nearest face of the home cell
            for (int a = 0; a < 3; ++a) {
                const double x = a == 0 ? p.x : a == 1 ? p.y : p.z;
                home[a] = Kokkos::clamp(static_cast<int>(x / width), 0, side - 1);
                const double frac = x - home[a] * width;
                inside = Kokkos::min(inside, Kokkos::min(frac, width - frac));
            }

            Polyhedron<Real, Planes, Vertices> cell;
            detail::start_cube(cell, half_width);
            int status = ok;
            bool certified = false;
            Candidate buffer[ring_buffer];
            int buffered = 0;
            int rings = 0, clipped = 0, cuts = 0;

            // Clip with the buffered particles nearest-first; stop as soon as
            // the remaining ones are too far to cut.
            auto flush = [&]() {
                for (int i = 1; i < buffered; ++i) {
                    const Candidate c = buffer[i];
                    int j = i;
                    for (; j > 0 && buffer[j - 1].distance2 > c.distance2; --j) buffer[j] = buffer[j - 1];
                    buffer[j] = c;
                }
                for (int i = 0; i < buffered && status == ok; ++i) {
                    const Candidate& c = buffer[i];
                    if (c.distance2 >= 4 * static_cast<float>(cell.radius2) * 1.0001f) break;
                    const Vec3<double> q{positions(c.slot, 0) + c.image[0] * box,
                                         positions(c.slot, 1) + c.image[1] * box,
                                         positions(c.slot, 2) + c.image[2] * box};
                    const int planes_before = cell.planes;
                    status = detail::clip(cell, convert<Real>(q - p), c.slot);
                    ++clipped;
                    cuts += cell.planes > planes_before;
                }
                buffered = 0;
            };

            for (int ring = 0; ring <= max_ring && status == ok; ++ring) {
                const double gap = ring == 0 ? 0.0 : (ring - 1) * width + inside;
                if (gap * gap > 4.0 * static_cast<double>(cell.radius2)) {
                    certified = true;
                    break;
                }
                ++rings;
                for (int dx = -ring; dx <= ring && status == ok; ++dx) {
                    for (int dy = -ring; dy <= ring && status == ok; ++dy) {
                        const int ux = home[0] + dx, uy = home[1] + dy;
                        const int x = (ux + side) % side, y = (uy + side) % side;
                        const std::int8_t image_x = ux < 0 ? -1 : ux >= side ? 1 : 0;
                        const std::int8_t image_y = uy < 0 ? -1 : uy >= side ? 1 : 0;
                        // Skip columns that lie entirely beyond 2R in x and y.
                        const double gx = Kokkos::max(0.0, Kokkos::max(ux * width - p.x, p.x - (ux + 1) * width));
                        const double gy = Kokkos::max(0.0, Kokkos::max(uy * width - p.y, p.y - (uy + 1) * width));
                        if (gx * gx + gy * gy >= 4.0 * static_cast<double>(cell.radius2)) continue;
                        // On the ring's x/y rim the whole z column belongs to
                        // the ring; inside it only the two end cells do.
                        const bool rim = dx == -ring || dx == ring || dy == -ring || dy == ring;
                        const int dz_step = rim ? 1 : 2 * ring;
                        for (int dz = -ring; dz <= ring && status == ok; dz += Kokkos::max(dz_step, 1)) {
                            const int uz = home[2] + dz;
                            const int z = (uz + side) % side;
                            const std::int8_t image_z = uz < 0 ? -1 : uz >= side ? 1 : 0;
                            const int key = grid.rank((x * side + y) * side + z);
                            for (int q = offsets(key); q < offsets(key + 1); ++q) {
                                if (q == t) continue;
                                const double ddx = positions(q, 0) + image_x * box - p.x;
                                const double ddy = positions(q, 1) + image_y * box - p.y;
                                const double ddz = positions(q, 2) + image_z * box - p.z;
                                const double d2 = ddx * ddx + ddy * ddy + ddz * ddz;
                                if (d2 >= 4.0 * static_cast<double>(cell.radius2)) continue;
                                if (buffered == ring_buffer) flush();
                                buffer[buffered++] = {static_cast<float>(d2), q, {image_x, image_y, image_z}};
                            }
                        }
                    }
                }
                flush();
            }
            if (status == ok && !certified) status = uncertified;
            for (int v = 0; v < cell.vertices && status == ok; ++v)
                for (int k = 0; k < 3; ++k)
                    if (cell.corner[v][k] < box_planes) status = uncertified;

            // Faces: walk each plane's vertices in order, integrating the face
            // area vector and centroid and the cell volume and centroid.
            int faces = 0, owned = 0, ring_used = 0;
            const bool with_rings = out.rings.extent(0) > 0;
            const int ring_capacity = with_rings ? static_cast<int>(out.rings.extent(1)) : 0;
            double volume = 0;
            Vec3<double> moment{0, 0, 0}, area_sum{0, 0, 0};
            double area_magnitude = 0;
            for (int plane = box_planes; plane < cell.planes && status == ok; ++plane) {
                int start = -1;
                for (int v = 0; v < cell.vertices && start < 0; ++v)
                    for (int k = 0; k < 3; ++k)
                        if (cell.corner[v][k] == plane) start = v;
                if (start < 0) continue;  // plane later cut away completely

                const Vec3<double> normal = convert<double>(cell.normal[plane]);
                const Vec3<double> first = convert<double>(cell.vertex[start]);
                Vec3<double> previous = first, area{0, 0, 0}, face_moment{0, 0, 0};
                double weight = 0;
                int current = start, steps = 0, ring_length = 0;
                do {
                    int at = 0;
                    while (cell.corner[current][at] != plane) ++at;
                    const std::uint8_t last = cell.corner[current][(at + 2) % 3];
                    // The plane shared with the next vertex bounds the face there.
                    if (with_rings && ring_used + 1 + ring_length < ring_capacity)
                        out.rings(t, ring_used + 1 + ring_length) = cell.neighbour[last];
                    ++ring_length;
                    int next = -1;
                    for (int w = 0; w < cell.vertices && next < 0; ++w)
                        for (int k = 0; k < 3; ++k)
                            if (cell.corner[w][k] == plane && cell.corner[w][(k + 1) % 3] == last) next = w;
                    if (next < 0 || ++steps > Vertices) {
                        status = degenerate;
                        break;
                    }
                    const Vec3<double> x = convert<double>(cell.vertex[next]);
                    const Vec3<double> twice = cross(previous - first, x - first);
                    area = area + twice * 0.5;
                    const double w = dot(twice, normal);
                    weight += w;
                    face_moment = face_moment + (first + previous + x) * (w / 3.0);
                    const double tetra = dot(first, cross(previous, x)) / 6.0;
                    volume += tetra;
                    moment = moment + (first + previous + x) * (tetra / 4.0);
                    previous = x;
                    current = next;
                } while (current != start);
                if (status != ok) break;

                // Faces below rounding level (degenerate, e.g. cospherical
                // generators) carry no flux and are not reported; their
                // volume contribution is kept.
                area_sum = area_sum + area;
                const double sliver = 64 * Kokkos::Experimental::epsilon_v<Real> *
                                      static_cast<double>(cell.radius2);
                if (Kokkos::sqrt(dot(area, area)) <= sliver) continue;
                if (out.all_neighbours.extent(0) > 0) {
                    if (faces == static_cast<int>(out.all_neighbours.extent(1))) {
                        status = face_overflow;
                        break;
                    }
                    out.all_neighbours(t, faces) = cell.neighbour[plane];
                }
                ++faces;
                area_magnitude += Kokkos::sqrt(dot(area, area));
                const int neighbour = cell.neighbour[plane];
                if (neighbour > t) {
                    if (owned == max_owned_faces) {
                        status = face_overflow;
                        break;
                    }
                    if (with_rings) {
                        if (ring_used + 1 + ring_length > ring_capacity) {
                            status = face_overflow;
                            break;
                        }
                        out.rings(t, ring_used) = ring_length;
                        ring_used += 1 + ring_length;
                    }
                    // A sliver face has no meaningful centroid; use a vertex.
                    const Vec3<double> c = weight > 0 ? face_moment * (1.0 / weight) : first;
                    out.neighbour(t, owned) = neighbour;
                    out.area(t, owned, 0) = area.x;
                    out.area(t, owned, 1) = area.y;
                    out.area(t, owned, 2) = area.z;
                    out.face_centroid(t, owned, 0) = c.x;
                    out.face_centroid(t, owned, 1) = c.y;
                    out.face_centroid(t, owned, 2) = c.z;
                    ++owned;
                }
            }
            if (status == ok && !(volume > 0)) status = degenerate;

            out.status(t) = status;
            out.face_count(t) = status == ok ? faces : 0;
            out.owned_count(t) = status == ok ? owned : 0;
            out.volume(t) = status == ok ? volume : 0.0;
            const Vec3<double> c = status == ok ? moment * (1.0 / volume) : Vec3<double>{0, 0, 0};
            out.centroid(t, 0) = c.x;
            out.centroid(t, 1) = c.y;
            out.centroid(t, 2) = c.z;
            if (with_closure) {
                out.closure(t) = status == ok ? Kokkos::sqrt(dot(area_sum, area_sum)) / area_magnitude : 0.0;
                out.work(t, 0) = rings;
                out.work(t, 1) = clipped;
                out.work(t, 2) = cuts;
            }
        });
}

}  // namespace detail

// Build every cell, or with only_failed just those whose status is not ok
// (the fallback after a faster kernel). Real is the precision of the private
// polyhedron; inputs and outputs are double. Cells that overflow the usual
// capacities are then rebuilt once more, in double with big ones.
template <class Real, class ExecSpace>
void construct(const ExecSpace& exec, const Grid<typename ExecSpace::memory_space>& grid,
               const Cells<typename ExecSpace::memory_space>& out, bool only_failed = false) {
    using detail::Pass;
    detail::construct_pass<Real, max_planes, max_vertices>(exec, grid, out, only_failed ? Pass::failed : Pass::all);
    int overflowed = 0;
    const auto status = out.status;
    Kokkos::parallel_reduce(
        "shell_voronoi::count_overflows", Kokkos::RangePolicy<ExecSpace>(exec, 0, grid.n),
        KOKKOS_LAMBDA(int t, int& count) { count += status(t) == vertex_overflow || status(t) == plane_overflow; },
        overflowed);
    if (overflowed > 0) detail::construct_pass<double, big_planes, big_vertices>(exec, grid, out, Pass::overflowed);
}

}  // namespace shell_voronoi
