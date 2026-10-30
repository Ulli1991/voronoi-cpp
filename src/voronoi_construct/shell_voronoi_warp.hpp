#pragma once

#include "shell_voronoi.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <type_traits>
#include <utility>
#include <vector>

// Warp-per-cell variant of shell_voronoi::construct for CUDA.
//
// One warp builds one cell. The polyhedron lives in the warp's slice of shared
// memory; lane l owns vertices l and l + 32 and planes l and l + 32. Every
// vertex stores its three planes and the three vertices across its edges, so
// a cut, the creation of new vertices and the walk around a face are each
// constant work per vertex, done by all lanes at once. Particles of a grid
// ring are tested 32 at a time and clipped nearest-first after a warp sort.
// Teams of 16 or 8 lanes per cell were 2x and 3.6x slower on an A100.
//
// Cells that overflow the warp capacities are rebuilt by the per-thread
// kernel. On other backends construct_warp simply calls that kernel.
namespace shell_voronoi {
namespace warp {

constexpr int lanes = 32;
constexpr int max_planes = 64;
constexpr int max_vertices = 64;
constexpr int per_lane = max_vertices / lanes;  // vertices (and planes) each lane owns
// Four cells (warps) per block: their polyhedra fill about half of the 48 KB
// block scratch.
constexpr int cells_per_block = 4;
// Caps the registers so that 5 blocks fit per SM: about 4 % faster on an A100.
constexpr int min_blocks_per_sm = 5;
constexpr std::uint8_t none = 255;

// One warp's polyhedron. Corners and adjacency hold three 8-bit indices each:
// corner k is a plane, adjacency k is the vertex across the edge from corner
// k to corner k + 1.
template <class Real>
struct alignas(16) Polyhedron {
    Real nx[max_planes], ny[max_planes], nz[max_planes], offset[max_planes];
    int neighbour[max_planes];
    // Vertices in two copies: a cut reads one and writes the next, so new and
    // kept vertices go straight to their final slots, without staging.
    Real x[2][max_vertices], y[2][max_vertices], z[2][max_vertices];
    Real side[max_vertices];                 // signed distance to the cutting plane
    std::uint32_t corner[2][max_vertices];
    std::uint32_t adjacent[2][max_vertices];
    std::uint32_t replacement[max_vertices];  // new vertex on each cut edge
    std::uint8_t starts[max_planes];          // new vertex whose hole edge starts at a plane
    std::uint8_t ends[max_planes];            // ... and ends at a plane
    std::uint8_t first[max_planes];           // a vertex on each face, for the face walk
    float candidate_distance2[lanes];
    int candidate_slot[lanes];
    int candidate_image[lanes];
};

// One copy of the vertices.
template <class Real>
struct Vertices {
    Real* x;
    Real* y;
    Real* z;
    std::uint32_t* corner;
    std::uint32_t* adjacent;
};
template <class Real>
KOKKOS_INLINE_FUNCTION Vertices<Real> copy_of(Polyhedron<Real>& c, int b) {
    return {c.x[b], c.y[b], c.z[b], c.corner[b], c.adjacent[b]};
}

KOKKOS_INLINE_FUNCTION int byte(std::uint32_t packed, int k) { return (packed >> (8 * k)) & 255; }
KOKKOS_INLINE_FUNCTION std::uint32_t pack(int a, int b, int c) {
    return std::uint32_t(a) | std::uint32_t(b) << 8 | std::uint32_t(c) << 16;
}
KOKKOS_INLINE_FUNCTION std::uint32_t with_byte(std::uint32_t packed, int k, int value) {
    return (packed & ~(std::uint32_t(255) << (8 * k))) | std::uint32_t(value) << (8 * k);
}

#if defined(__CUDA_ARCH__)
// The warp building a cell is the team's vector range: lane l is threadIdx.x.
__device__ inline int lane() { return threadIdx.x; }
constexpr unsigned all_lanes = 0xffffffffu;

// One bit per vertex: vertex l + k * lanes is bit l + k * lanes.
__device__ inline std::uint64_t vertex_ballot(const bool (&out)[per_lane]) {
    std::uint64_t bits = 0;
    for (int k = 0; k < per_lane; ++k) bits |= std::uint64_t(__ballot_sync(all_lanes, out[k])) << (k * lanes);
    return bits;
}

template <class T>
__device__ T warp_max(T v) {
    for (int o = lanes / 2; o > 0; o >>= 1) v = Kokkos::max(v, __shfl_xor_sync(all_lanes, v, o));
    return v;
}

template <class T>
__device__ T warp_sum(T v) {
    for (int o = lanes / 2; o > 0; o >>= 1) v += __shfl_xor_sync(all_lanes, v, o);
    return v;
}

// Exclusive prefix sum over lanes; total receives the warp sum.
__device__ inline int exclusive_scan(int v, int& total) {
    int inclusive = v;
    for (int o = 1; o < lanes; o <<= 1) {
        const int up = __shfl_up_sync(all_lanes, inclusive, o);
        if (lane() >= o) inclusive += up;
    }
    total = __shfl_sync(all_lanes, inclusive, lanes - 1);
    return inclusive - v;
}

// Index a kept vertex has after the removed ones are squeezed out.
__device__ inline int kept_index(int v, std::uint64_t removed) {
    return v - __popcll(removed & ((std::uint64_t(1) << v) - 1));
}

template <class Real>
__device__ void start_cube(Polyhedron<Real>& c, Real h, int& planes, int& vertices) {
    const int l = lane();
    const Vertices<Real> v = copy_of(c, 0);
    if (l < box_planes) {
        const int axis = l / 2;
        const Real sign = l % 2 == 0 ? Real(1) : Real(-1);
        c.nx[l] = axis == 0 ? sign : Real(0);
        c.ny[l] = axis == 1 ? sign : Real(0);
        c.nz[l] = axis == 2 ? sign : Real(0);
        c.offset[l] = h;
        c.neighbour[l] = -1;
    }
    if (l < 8) {
        const int sx = l & 1, sy = (l >> 1) & 1, sz = (l >> 2) & 1;
        const bool odd = (sx + sy + sz) % 2 == 1;
        const int px = sx, py = 2 + sy, pz = 4 + sz;
        v.corner[l] = odd ? pack(px, pz, py) : pack(px, py, pz);
        v.x[l] = sx ? -h : h;
        v.y[l] = sy ? -h : h;
        v.z[l] = sz ? -h : h;
    }
    __syncwarp(all_lanes);
    if (l < 8) {
        // Across edge (a, b) lies the corner that has the edge (b, a).
        std::uint32_t adjacent = 0;
        for (int e = 0; e < 3; ++e) {
            const int a = byte(v.corner[l], e), b = byte(v.corner[l], (e + 1) % 3);
            for (int w = 0; w < 8; ++w)
                for (int k = 0; k < 3; ++k)
                    if (byte(v.corner[w], k) == b && byte(v.corner[w], (k + 1) % 3) == a)
                        adjacent = with_byte(adjacent, e, w);
        }
        v.adjacent[l] = adjacent;
    }
    __syncwarp(all_lanes);
    planes = box_planes;
    vertices = 8;
}

// Clip by the bisector with a particle at offset d (all lanes pass the same
// d). Returns ok when the plane does not cut or after the cut is applied.
template <class Real>
__device__ int clip(Polyhedron<Real>& c, Vertices<Real> v, Vertices<Real> next, int& planes, int& vertices,
                    Real& radius2, Vec3<Real> d, int neighbour) {
    const int l = lane();
    const Real offset = Real(0.5) * dot(d, d);
    const Real tolerance =
        Real(64) * Kokkos::Experimental::epsilon_v<Real> * Kokkos::sqrt(dot(d, d) * radius2);
    bool out[per_lane] = {};
    for (int k = 0; k < per_lane; ++k) {
        const int w = l + k * lanes;
        if (w >= vertices) continue;
        const Real s = d.x * v.x[w] + d.y * v.y[w] + d.z * v.z[w] - offset;
        c.side[w] = s;
        out[k] = s > tolerance;
    }
    const std::uint64_t removed = vertex_ballot(out);
    if (removed == 0) return ok;
    if (planes == max_planes) return plane_overflow;
    __syncwarp(all_lanes);

    // Each edge from a removed vertex to a kept one crosses the plane and
    // becomes a new vertex. Count them per lane, then number them.
    int crossings = 0;
    for (int k = 0; k < per_lane; ++k) {
        const int w = l + k * lanes;
        if (!out[k]) continue;
        for (int e = 0; e < 3; ++e)
            if (!(removed >> byte(v.adjacent[w], e) & 1)) ++crossings;
    }
    int created = 0;
    const int first_new = exclusive_scan(crossings, created);
    const int kept = vertices - __popcll(removed);
    if (kept + created > max_vertices) return vertex_overflow;
    if (kept + created < 4) return degenerate;
    const int plane = planes;

    // New vertices, straight into the next copy: positions, planes, and
    // the tables that link them.
    Real radius = 0;
    int made = 0;
    for (int k = 0; k < per_lane; ++k) {
        const int w = l + k * lanes;
        if (!out[k]) continue;
        std::uint32_t replacement = 0;
        for (int e = 0; e < 3; ++e) {
            const int u = byte(v.adjacent[w], e);
            if (removed >> u & 1) continue;
            const int index = kept + first_new + made;
            const Real t = c.side[w] / (c.side[w] - c.side[u]);
            const Real px = v.x[w] + (v.x[u] - v.x[w]) * t, py = v.y[w] + (v.y[u] - v.y[w]) * t,
                       pz = v.z[w] + (v.z[u] - v.z[w]) * t;
            next.x[index] = px;
            next.y[index] = py;
            next.z[index] = pz;
            radius = Kokkos::max(radius, px * px + py * py + pz * pz);
            const int a = byte(v.corner[w], e), b = byte(v.corner[w], (e + 1) % 3);
            next.corner[index] = pack(a, b, plane);
            next.adjacent[index] = kept_index(u, removed);  // across (a, b)
            c.starts[a] = index;
            c.ends[b] = index;
            replacement = with_byte(replacement, e, index);
            ++made;
        }
        c.replacement[w] = replacement;
    }
    __syncwarp(all_lanes);
    // Across (b, plane) is the new vertex whose hole edge starts at b, across
    // (plane, a) the one whose hole edge ends at a.
    for (int m = 0; m < made; ++m) {
        const int index = kept + first_new + m;
        const int a = byte(next.corner[index], 0), b = byte(next.corner[index], 1);
        next.adjacent[index] |= std::uint32_t(c.starts[b]) << 8 | std::uint32_t(c.ends[a]) << 16;
    }
    // Kept vertices, renumbered (a removed neighbour is replaced by the new
    // vertex on that edge), into the next copy.
    for (int k = 0; k < per_lane; ++k) {
        const int u = l + k * lanes;
        if (u >= vertices || out[k]) continue;
        const int i = kept_index(u, removed);
        next.x[i] = v.x[u];
        next.y[i] = v.y[u];
        next.z[i] = v.z[u];
        next.corner[i] = v.corner[u];
        radius = Kokkos::max(radius, v.x[u] * v.x[u] + v.y[u] * v.y[u] + v.z[u] * v.z[u]);
        std::uint32_t adjacent = 0;
        for (int e = 0; e < 3; ++e) {
            const int w = byte(v.adjacent[u], e);
            int index;
            if (removed >> w & 1) {
                int back = 0;
                while (back < 2 && byte(v.adjacent[w], back) != u) ++back;
                index = byte(c.replacement[w], back);
            } else {
                index = kept_index(w, removed);
            }
            adjacent = with_byte(adjacent, e, index);
        }
        next.adjacent[i] = adjacent;
    }
    if (l == 0) {
        c.nx[plane] = d.x;
        c.ny[plane] = d.y;
        c.nz[plane] = d.z;
        c.offset[plane] = offset;
        c.neighbour[plane] = neighbour;
    }
    planes = plane + 1;
    vertices = kept + created;
    radius2 = warp_max(radius);
    __syncwarp(all_lanes);
    return ok;
}

// Warp bitonic sort of one (key, slot, image) entry per lane, ascending by
// key and then slot so the order is deterministic.
__device__ inline void sort_candidates(float& key, int& slot, int& packed_image) {
    const int l = lane();
    for (int size = 2; size <= lanes; size <<= 1) {
        for (int stride = size / 2; stride > 0; stride >>= 1) {
            const float other_key = __shfl_xor_sync(all_lanes, key, stride);
            const int other_slot = __shfl_xor_sync(all_lanes, slot, stride);
            const int other_image = __shfl_xor_sync(all_lanes, packed_image, stride);
            const bool ascending = (l & size) == 0;
            const bool lower = (l & stride) == 0;
            const bool other_smaller = other_key < key || (other_key == key && other_slot < slot);
            const bool take = lower == ascending ? other_smaller : !other_smaller && (other_key != key || other_slot != slot);
            if (take) {
                key = other_key;
                slot = other_slot;
                packed_image = other_image;
            }
        }
    }
}
#endif  // __CUDA_ARCH__

#if defined(KOKKOS_ENABLE_CUDA)
// Adaptive: walk each cell at its level (CellList::level); otherwise at the
// base level only, leaving cells it cannot certify to the thread kernel.
template <class Real, class MemorySpace, bool adaptive>
struct Kernel {
    using Member = typename Kokkos::TeamPolicy<Kokkos::Cuda>::member_type;
    Grid<MemorySpace> grid;
    Cells<MemorySpace> out;
    Real half_width;
    Stencil<MemorySpace> stencil;
    // Persistent warps (about two waves of resident blocks) take cells from
    // this counter until none are left, so a slow cell holds up only its own
    // warp: 1.06x on uniform random points over one block per four cells.
    Kokkos::View<int, MemorySpace> next_cell;

    KOKKOS_FUNCTION void operator()(const Member& member) const {
#if defined(__CUDA_ARCH__)
        auto* base = static_cast<Polyhedron<Real>*>(
            member.team_shmem().get_shmem(cells_per_block * sizeof(Polyhedron<Real>)));
        Polyhedron<Real>& c = base[member.team_rank()];
        while (true) {
            int taken = 0;
            if (lane() == 0) taken = Kokkos::atomic_fetch_add(&next_cell(), 1);
            const int t = __shfl_sync(all_lanes, taken, 0);
            if (t >= grid.n) return;  // the whole warp
            build(c, t);
        }
#else
        (void)member;
#endif
    }

#if defined(__CUDA_ARCH__)
    // Build cell t in polyhedron c, with all lanes of the warp.
    __device__ __forceinline__ void build(Polyhedron<Real>& c, int t) const {
        const int l = lane();
        const auto& positions = grid.positions;
        const double box = grid.box;
        const Vec3<double> p{positions(t, 0), positions(t, 1), positions(t, 2)};

        int planes = 0, vertices = 0;
        Real radius2 = 3 * half_width * half_width;
        start_cube(c, half_width, planes, vertices);
        int current = 0;  // which copy of the vertices is the polyhedron
        int status = ok, rings = 0, clipped = 0, cuts = 0, buffered = 0;
        bool certified = false;

        // Sort the buffered candidates, then clip nearest-first while they
        // can still cut the cell.
        auto flush = [&]() {
            float key = l < buffered ? c.candidate_distance2[l] : 3.0e38f;
            int slot = l < buffered ? c.candidate_slot[l] : 0x7fffffff;
            int packed = l < buffered ? c.candidate_image[l] : 0;
            __syncwarp(all_lanes);
            sort_candidates(key, slot, packed);
            for (int i = 0; i < buffered && status == ok; ++i) {
                const float k = __shfl_sync(all_lanes, key, i);
                if (k >= 4.0f * static_cast<float>(radius2) * 1.0001f) break;
                const int q = __shfl_sync(all_lanes, slot, i);
                const int im = __shfl_sync(all_lanes, packed, i);
                const Vec3<double> d{positions(q, 0) + image(im, 0) * box - p.x,
                                     positions(q, 1) + image(im, 1) * box - p.y,
                                     positions(q, 2) + image(im, 2) * box - p.z};
                const int before = planes;
                status = clip(c, copy_of(c, current), copy_of(c, 1 - current), planes, vertices, radius2,
                              convert<Real>(d), q);
                if (planes > before) current = 1 - current;
                ++clipped;
                cuts += planes > before;
            }
            buffered = 0;
        };

        // Visit the grid cells of the walk (see shell_voronoi.hpp) in order of
        // their smallest possible distance, 32 at a time: each lane checks one
        // cell against 2R and reads its particles, so the memory accesses
        // overlap. A level is certified once the next cell's lower bound
        // exceeds 2R.
        int level = adaptive ? int(grid.level(t)) : grid.levels, home[3], finer = -1;
        home_cell(p, box, level, home);
        while (status == ok) {
            const double width = box / (1 << level);
            const int reach = level_reach(level), entries = stencil.ends(reach);
            for (int first = 0; first < entries && status == ok; first += lanes) {
                if (double(stencil.bound2(first)) * width * width >= 4.0 * static_cast<double>(radius2)) {
                    certified = true;
                    break;
                }
                ++rings;
                const int e = first + l;
                int begin = 0, end = 0, packed = 0;
                const double reach2 = 4.0 * static_cast<double>(radius2);
                if (e < entries)
                    stencil_cell<adaptive>(grid, level, home, finer, stencil.offsets(e), p, reach2, begin, end, packed);
                const int most = warp_max(end - begin);
                for (int k = 0; k < most && status == ok; ++k) {
                    const int q = begin + k;
                    double d2 = 0;
                    bool take = false;
                    if (q < end && q != t) {
                        const double ddx = positions(q, 0) + image(packed, 0) * box - p.x;
                        const double ddy = positions(q, 1) + image(packed, 1) * box - p.y;
                        const double ddz = positions(q, 2) + image(packed, 2) * box - p.z;
                        d2 = ddx * ddx + ddy * ddy + ddz * ddz;
                        take = d2 < 4.0 * static_cast<double>(radius2);
                    }
                    const unsigned taking = __ballot_sync(all_lanes, take);
                    if (taking == 0) continue;
                    if (buffered + __popc(taking) > lanes) flush();
                    if (take) {
                        const int index = buffered + __popc(taking & ((1u << l) - 1));
                        c.candidate_distance2[index] = static_cast<float>(d2);
                        c.candidate_slot[index] = q;
                        c.candidate_image[index] = packed;
                    }
                    buffered += __popc(taking);
                    __syncwarp(all_lanes);
                }
                if (buffered > 0) flush();
            }
            if (status != ok || certified) break;
            // Cells outside the stencil are at least `reach` widths away.
            if (reach * width * reach * width >= 4.0 * static_cast<double>(radius2)) {
                certified = true;
                break;
            }
            if (!adaptive || level == 2) break;  // no coarser grid fits the box
            finer = (home[0] & 1) | (home[1] & 1) << 1 | (home[2] & 1) << 2;
            for (int a = 0; a < 3; ++a) home[a] >>= 1;
            --level;
        }
        if (status == ok && !certified) status = uncertified;

        const Vertices<Real> cv = copy_of(c, current);
        // Certification: no vertex may still touch the starting cube.
        if (status == ok) {
            bool touches = false;
            for (int k = 0; k < per_lane; ++k) {
                const int v = l + k * lanes;
                if (v < vertices)
                    for (int e = 0; e < 3; ++e) touches |= byte(cv.corner[v], e) < box_planes;
            }
            if (__any_sync(all_lanes, touches)) status = uncertified;
        }

        // Faces: lane l walks planes l and l + 32 around their vertices.
        for (int k = 0; k < per_lane; ++k)
            if (l + k * lanes < planes) c.first[l + k * lanes] = none;
        __syncwarp(all_lanes);
        for (int k = 0; k < per_lane; ++k) {
            const int v = l + k * lanes;
            if (v < vertices)
                for (int e = 0; e < 3; ++e) c.first[byte(cv.corner[v], e)] = v;
        }
        __syncwarp(all_lanes);

        double volume = 0;
        Vec3<double> moment{0, 0, 0}, area_sum{0, 0, 0};
        double area_magnitude = 0;
        int faces = 0;
        bool broken = false;
        int owned_here = 0;
        int owned_plane[per_lane];
        for (int k = 0; k < per_lane; ++k) owned_plane[k] = -1;
        Vec3<double> owned_area[per_lane], owned_centroid[per_lane];
        const double sliver = 64 * Kokkos::Experimental::epsilon_v<Real> * static_cast<double>(radius2);
        for (int k = 0; k < per_lane && status == ok; ++k) {
            const int plane = l + k * lanes;
            if (plane < box_planes || plane >= planes || c.first[plane] == none) continue;
            const int start = c.first[plane];
            const Vec3<double> normal{double(c.nx[plane]), double(c.ny[plane]), double(c.nz[plane])};
            const Vec3<double> first{double(cv.x[start]), double(cv.y[start]), double(cv.z[start])};
            Vec3<double> previous = first, area{0, 0, 0}, face_moment{0, 0, 0};
            double weight = 0;
            int vertex = start, steps = 0;
            do {
                int at = 0;
                while (at < 3 && byte(cv.corner[vertex], at) != plane) ++at;
                if (at == 3 || ++steps > max_vertices) {
                    broken = true;
                    break;
                }
                const int next = byte(cv.adjacent[vertex], (at + 2) % 3);
                const Vec3<double> x{double(cv.x[next]), double(cv.y[next]), double(cv.z[next])};
                const Vec3<double> twice = cross(previous - first, x - first);
                area = area + twice * 0.5;
                const double w = dot(twice, normal);
                weight += w;
                face_moment = face_moment + (first + previous + x) * (w / 3.0);
                const double tetra = dot(first, cross(previous, x)) / 6.0;
                volume += tetra;
                moment = moment + (first + previous + x) * (tetra / 4.0);
                previous = x;
                vertex = next;
            } while (vertex != start);
            if (broken) break;
            area_sum = area_sum + area;
            const double magnitude = Kokkos::sqrt(dot(area, area));
            if (magnitude <= sliver) continue;
            ++faces;
            area_magnitude += magnitude;
            if (c.neighbour[plane] > t) {
                owned_plane[owned_here] = plane;
                owned_area[owned_here] = area;
                owned_centroid[owned_here] = weight > 0 ? face_moment * (1.0 / weight) : first;
                ++owned_here;
            }
        }
        if (status == ok && __any_sync(all_lanes, broken)) status = degenerate;

        volume = warp_sum(volume);
        moment = {warp_sum(moment.x), warp_sum(moment.y), warp_sum(moment.z)};
        area_sum = {warp_sum(area_sum.x), warp_sum(area_sum.y), warp_sum(area_sum.z)};
        area_magnitude = warp_sum(area_magnitude);
        faces = warp_sum(faces);
        int owned = 0;
        const int first_owned = exclusive_scan(owned_here, owned);
        if (status == ok && owned > out.owned_capacity()) status = face_overflow;
        if (status == ok && !(volume > 0)) status = degenerate;

        if (status == ok) {
            for (int m = 0; m < owned_here; ++m) {
                const int f = first_owned + m;
                out.neighbour(t, f) = c.neighbour[owned_plane[m]];
                out.area(t, f, 0) = owned_area[m].x;
                out.area(t, f, 1) = owned_area[m].y;
                out.area(t, f, 2) = owned_area[m].z;
                out.face_centroid(t, f, 0) = owned_centroid[m].x;
                out.face_centroid(t, f, 1) = owned_centroid[m].y;
                out.face_centroid(t, f, 2) = owned_centroid[m].z;
            }
        }
        if (l == 0) {
            const bool good = status == ok;
            out.status(t) = status;
            out.face_count(t) = good ? faces : 0;
            out.owned_count(t) = good || status == face_overflow ? owned : 0;
            out.volume(t) = good ? volume : 0.0;
            out.centroid(t, 0) = good ? moment.x / volume : 0.0;
            out.centroid(t, 1) = good ? moment.y / volume : 0.0;
            out.centroid(t, 2) = good ? moment.z / volume : 0.0;
            if (out.closure.extent(0) > 0) {
                out.closure(t) = good ? Kokkos::sqrt(dot(area_sum, area_sum)) / area_magnitude : 0.0;
                out.work(t, 0) = rings;
                out.work(t, 1) = clipped;
                out.work(t, 2) = cuts;
            }
        }
    }
#endif
};

#endif  // KOKKOS_ENABLE_CUDA

}  // namespace warp

// Build every cell with one warp per cell, then rebuild cells that overflowed
// the warp capacities with the per-thread kernel. Other backends, and
// requests for face rings, use the per-thread kernel throughout.
template <class Real, class ExecSpace>
void construct_warp(const ExecSpace& exec, const Grid<typename ExecSpace::memory_space>& grid,
                    Cells<typename ExecSpace::memory_space>& out) {
#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
        if (out.rings.extent(0) == 0) {
            if (grid.n <= 0 || out.status.extent(0) != static_cast<std::size_t>(grid.n))
                throw std::invalid_argument("shell_voronoi::construct_warp: invalid grid or output size");
            const Real half_width = detail::start_half_width<Real>(grid.box);
            using Memory = typename ExecSpace::memory_space;
            const Stencil<Memory> stencil = make_stencil(exec);
            Kokkos::View<int, Memory> next_cell("shell_warp_next_cell");
            const int blocks = Kokkos::min((grid.n + warp::cells_per_block - 1) / warp::cells_per_block,
                                           2 * warp::min_blocks_per_sm * exec.cuda_device_prop().multiProcessorCount);
            // A block holds cells_per_block warps; each warp is a team's vector range.
            Kokkos::TeamPolicy<Kokkos::Cuda,
                               Kokkos::LaunchBounds<warp::cells_per_block * warp::lanes, warp::min_blocks_per_sm>>
                policy(exec, blocks, warp::cells_per_block, warp::lanes);
            policy.set_scratch_size(0, Kokkos::PerTeam(warp::cells_per_block * sizeof(warp::Polyhedron<Real>)));
            // Near-uniform input (every particle at the base level) takes the
            // kernel without level changes and fine-cell searches: its smaller
            // code is 4 % faster on uniform random points.
            int off_base = 0;
            const auto level = grid.level;
            const int base = grid.levels;
            Kokkos::parallel_reduce(
                "shell_voronoi::off_base_levels", Kokkos::RangePolicy<ExecSpace>(exec, 0, grid.n),
                KOKKOS_LAMBDA(int t, int& count) { count += level(t) != base; }, off_base);
            if (off_base > 0)
                Kokkos::parallel_for("shell_voronoi::construct_warp", policy,
                                     warp::Kernel<Real, Memory, true>{grid, out, half_width, stencil, next_cell});
            else
                Kokkos::parallel_for("shell_voronoi::construct_warp", policy,
                                     warp::Kernel<Real, Memory, false>{grid, out, half_width, stencil, next_cell});
            detail::construct_with_overflow<Real>(exec, grid, stencil, out, detail::Pass::failed);
            return;
        }
    }
#endif
    construct<Real>(exec, grid, out);
}

}  // namespace shell_voronoi
