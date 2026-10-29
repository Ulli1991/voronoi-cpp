#pragma once

#include "shell_voronoi.hpp"

#include <algorithm>
#include <cmath>
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
//
// Cells that overflow the warp capacities are rebuilt by the per-thread
// kernel. On other backends construct_warp simply calls that kernel.
namespace shell_voronoi {
namespace warp {

constexpr int lanes = 32;
constexpr int max_planes = 64;
constexpr int max_vertices = 64;
constexpr int warps_per_block = 4;
#ifndef SHELL_WARP_MIN_BLOCKS
#define SHELL_WARP_MIN_BLOCKS 5  // caps registers so 5 blocks fit per SM: ~4% faster on A100
#endif
constexpr std::uint8_t none = 255;

// One warp's polyhedron. Corners and adjacency hold three 8-bit indices each:
// corner k is a plane, adjacency k is the vertex across the edge from corner
// k to corner k + 1.
template <class Real>
struct alignas(16) Polyhedron {
    Real nx[max_planes], ny[max_planes], nz[max_planes], offset[max_planes];
    int neighbour[max_planes];
    Real x[max_vertices], y[max_vertices], z[max_vertices];
    Real side[max_vertices];                 // signed distance to the cutting plane
    std::uint32_t corner[max_vertices];
    std::uint32_t adjacent[max_vertices];
    std::uint32_t replacement[max_vertices];  // new vertex on each cut edge
    std::uint8_t starts[max_planes];          // new vertex whose hole edge starts at a plane
    std::uint8_t ends[max_planes];            // ... and ends at a plane
    std::uint8_t first[max_planes];           // a vertex on each face, for the face walk
    float candidate_distance2[lanes];
    int candidate_slot[lanes];
    int candidate_image[lanes];
};

KOKKOS_INLINE_FUNCTION int byte(std::uint32_t packed, int k) { return (packed >> (8 * k)) & 255; }
KOKKOS_INLINE_FUNCTION std::uint32_t pack(int a, int b, int c) {
    return std::uint32_t(a) | std::uint32_t(b) << 8 | std::uint32_t(c) << 16;
}
KOKKOS_INLINE_FUNCTION std::uint32_t with_byte(std::uint32_t packed, int k, int value) {
    return (packed & ~(std::uint32_t(255) << (8 * k))) | std::uint32_t(value) << (8 * k);
}
KOKKOS_INLINE_FUNCTION int pack_image(int x, int y, int z) { return (x + 1) | (y + 1) << 2 | (z + 1) << 4; }
KOKKOS_INLINE_FUNCTION int image(int packed, int axis) { return ((packed >> (2 * axis)) & 3) - 1; }

#if defined(__CUDA_ARCH__)
constexpr unsigned full_mask = 0xffffffffu;

__device__ inline int lane() { return threadIdx.x; }

__device__ inline std::uint64_t ballot64(bool low, bool high) {
    return std::uint64_t(__ballot_sync(full_mask, low)) | std::uint64_t(__ballot_sync(full_mask, high)) << 32;
}

template <class T>
__device__ T warp_max(T v) {
    for (int o = 16; o > 0; o >>= 1) v = Kokkos::max(v, __shfl_xor_sync(full_mask, v, o));
    return v;
}

template <class T>
__device__ T warp_sum(T v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(full_mask, v, o);
    return v;
}

// Exclusive prefix sum over lanes; total receives the warp sum.
__device__ inline int exclusive_scan(int v, int& total) {
    int inclusive = v;
    for (int o = 1; o < lanes; o <<= 1) {
        const int up = __shfl_up_sync(full_mask, inclusive, o);
        if (lane() >= o) inclusive += up;
    }
    total = __shfl_sync(full_mask, inclusive, lanes - 1);
    return inclusive - v;
}

// Index a kept vertex has after the removed ones are squeezed out.
__device__ inline int kept_index(int v, std::uint64_t removed) {
    return v - __popcll(removed & ((std::uint64_t(1) << v) - 1));
}

template <class Real>
__device__ void start_cube(Polyhedron<Real>& c, Real h, int& planes, int& vertices) {
    const int l = lane();
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
        c.corner[l] = odd ? pack(px, pz, py) : pack(px, py, pz);
        c.x[l] = sx ? -h : h;
        c.y[l] = sy ? -h : h;
        c.z[l] = sz ? -h : h;
    }
    __syncwarp();
    if (l < 8) {
        // Across edge (a, b) lies the corner that has the edge (b, a).
        std::uint32_t adjacent = 0;
        for (int e = 0; e < 3; ++e) {
            const int a = byte(c.corner[l], e), b = byte(c.corner[l], (e + 1) % 3);
            for (int w = 0; w < 8; ++w)
                for (int k = 0; k < 3; ++k)
                    if (byte(c.corner[w], k) == b && byte(c.corner[w], (k + 1) % 3) == a)
                        adjacent = with_byte(adjacent, e, w);
        }
        c.adjacent[l] = adjacent;
    }
    __syncwarp();
    planes = box_planes;
    vertices = 8;
}

// Clip by the bisector with a particle at offset d (all lanes pass the same
// d). Returns ok when the plane does not cut or after the cut is applied.
template <class Real>
__device__ int clip(Polyhedron<Real>& c, int& planes, int& vertices, Real& radius2, Vec3<Real> d,
                    int neighbour) {
    const int l = lane();
    const Real offset = Real(0.5) * dot(d, d);
    const Real tolerance =
        Real(64) * Kokkos::Experimental::epsilon_v<Real> * Kokkos::sqrt(dot(d, d) * radius2);
    bool out[2] = {false, false};
    for (int k = 0; k < 2; ++k) {
        const int v = l + k * lanes;
        if (v >= vertices) continue;
        const Real s = d.x * c.x[v] + d.y * c.y[v] + d.z * c.z[v] - offset;
        c.side[v] = s;
        out[k] = s > tolerance;
    }
    const std::uint64_t removed = ballot64(out[0], out[1]);
    if (removed == 0) return ok;
    if (planes == max_planes) return plane_overflow;
    __syncwarp();

    // Each edge from a removed vertex to a kept one crosses the plane and
    // becomes a new vertex. Count them per lane, then number them.
    int crossings = 0;
    for (int k = 0; k < 2; ++k) {
        const int v = l + k * lanes;
        if (!out[k]) continue;
        for (int e = 0; e < 3; ++e)
            if (!(removed >> byte(c.adjacent[v], e) & 1)) ++crossings;
    }
    int created = 0;
    const int first_new = exclusive_scan(crossings, created);
    const int kept = vertices - __popcll(removed);
    if (kept + created > max_vertices) return vertex_overflow;
    if (kept + created < 4) return degenerate;
    const int plane = planes;

    // Pass 1: new vertex positions and planes, and the tables that link them.
    Real new_x[6], new_y[6], new_z[6];
    std::uint32_t new_corner[6], new_adjacent[6];
    int made = 0;
    for (int k = 0; k < 2; ++k) {
        const int v = l + k * lanes;
        if (!out[k]) continue;
        std::uint32_t replacement = 0;
        for (int e = 0; e < 3; ++e) {
            const int u = byte(c.adjacent[v], e);
            if (removed >> u & 1) continue;
            const int index = kept + first_new + made;
            const Real t = c.side[v] / (c.side[v] - c.side[u]);
            new_x[made] = c.x[v] + (c.x[u] - c.x[v]) * t;
            new_y[made] = c.y[v] + (c.y[u] - c.y[v]) * t;
            new_z[made] = c.z[v] + (c.z[u] - c.z[v]) * t;
            const int a = byte(c.corner[v], e), b = byte(c.corner[v], (e + 1) % 3);
            new_corner[made] = pack(a, b, plane);
            new_adjacent[made] = kept_index(u, removed);  // across (a, b)
            c.starts[a] = index;
            c.ends[b] = index;
            replacement = with_byte(replacement, e, index);
            ++made;
        }
        c.replacement[v] = replacement;
    }
    __syncwarp();
    // Across (b, plane) is the new vertex whose hole edge starts at b, across
    // (plane, a) the one whose hole edge ends at a.
    for (int m = 0; m < made; ++m) {
        const int a = byte(new_corner[m], 0), b = byte(new_corner[m], 1);
        new_adjacent[m] |= std::uint32_t(c.starts[b]) << 8 | std::uint32_t(c.ends[a]) << 16;
    }

    // Kept vertices: read, renumber neighbours (a removed neighbour is
    // replaced by the new vertex on that edge).
    Real keep_x[2], keep_y[2], keep_z[2];
    std::uint32_t keep_corner[2], keep_adjacent[2];
    int keep_index[2] = {-1, -1};
    for (int k = 0; k < 2; ++k) {
        const int u = l + k * lanes;
        if (u >= vertices || out[k]) continue;
        keep_index[k] = kept_index(u, removed);
        keep_x[k] = c.x[u];
        keep_y[k] = c.y[u];
        keep_z[k] = c.z[u];
        keep_corner[k] = c.corner[u];
        std::uint32_t adjacent = 0;
        for (int e = 0; e < 3; ++e) {
            const int w = byte(c.adjacent[u], e);
            int index;
            if (removed >> w & 1) {
                int back = 0;
                while (back < 2 && byte(c.adjacent[w], back) != u) ++back;
                index = byte(c.replacement[w], back);
            } else {
                index = kept_index(w, removed);
            }
            adjacent = with_byte(adjacent, e, index);
        }
        keep_adjacent[k] = adjacent;
    }
    __syncwarp();

    // Pass 2: write the compacted polyhedron.
    Real radius = 0;
    for (int k = 0; k < 2; ++k) {
        const int i = keep_index[k];
        if (i < 0) continue;
        c.x[i] = keep_x[k];
        c.y[i] = keep_y[k];
        c.z[i] = keep_z[k];
        c.corner[i] = keep_corner[k];
        c.adjacent[i] = keep_adjacent[k];
        radius = Kokkos::max(radius, keep_x[k] * keep_x[k] + keep_y[k] * keep_y[k] + keep_z[k] * keep_z[k]);
    }
    for (int m = 0; m < made; ++m) {
        const int i = kept + first_new + m;
        c.x[i] = new_x[m];
        c.y[i] = new_y[m];
        c.z[i] = new_z[m];
        c.corner[i] = new_corner[m];
        c.adjacent[i] = new_adjacent[m];
        radius = Kokkos::max(radius, new_x[m] * new_x[m] + new_y[m] * new_y[m] + new_z[m] * new_z[m]);
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
    __syncwarp();
    return ok;
}

// Warp bitonic sort of one (key, slot, image) entry per lane, ascending by
// key and then slot so the order is deterministic.
__device__ inline void sort_candidates(float& key, int& slot, int& packed_image) {
    const int l = lane();
    for (int size = 2; size <= lanes; size <<= 1) {
        for (int stride = size / 2; stride > 0; stride >>= 1) {
            const float other_key = __shfl_xor_sync(full_mask, key, stride);
            const int other_slot = __shfl_xor_sync(full_mask, slot, stride);
            const int other_image = __shfl_xor_sync(full_mask, packed_image, stride);
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
template <class Real, class MemorySpace>
struct Kernel {
    using Member = typename Kokkos::TeamPolicy<Kokkos::Cuda>::member_type;
    Grid<MemorySpace> grid;
    Cells<MemorySpace> out;
    Real half_width;
    // Grid-cell offsets (4 bits per axis, biased by 8) sorted by the squared
    // smallest distance any point of the home cell can have to them.
    Kokkos::View<const int*, MemorySpace> stencil;
    Kokkos::View<const float*, MemorySpace> stencil_bound2;
    int stencil_size;
    double stencil_reach;

    KOKKOS_FUNCTION void operator()(const Member& member) const {
#if defined(__CUDA_ARCH__)
        const int t = member.league_rank() * warps_per_block + member.team_rank();
        if (t >= grid.n) return;  // whole warp
        auto* base = static_cast<Polyhedron<Real>*>(
            member.team_shmem().get_shmem(warps_per_block * sizeof(Polyhedron<Real>)));
        Polyhedron<Real>& c = base[member.team_rank()];
        const int l = lane();
        const auto& positions = grid.positions;
        const auto& offsets = grid.cell_offsets;
        const int side = grid.cells_per_side;
        const double width = grid.cell_width, box = grid.box;

        const Vec3<double> p{positions(t, 0), positions(t, 1), positions(t, 2)};
        int home[3];
        double inside = width;
        for (int a = 0; a < 3; ++a) {
            const double x = a == 0 ? p.x : a == 1 ? p.y : p.z;
            home[a] = Kokkos::clamp(static_cast<int>(x / width), 0, side - 1);
            const double frac = x - home[a] * width;
            inside = Kokkos::min(inside, Kokkos::min(frac, width - frac));
        }

        int planes = 0, vertices = 0;
        Real radius2 = 3 * half_width * half_width;
        start_cube(c, half_width, planes, vertices);
        int status = ok, rings = 0, clipped = 0, cuts = 0, buffered = 0;
        bool certified = false;

        // Sort the buffered candidates, then clip nearest-first while they
        // can still cut the cell.
        auto flush = [&]() {
            float key = l < buffered ? c.candidate_distance2[l] : 3.0e38f;
            int slot = l < buffered ? c.candidate_slot[l] : 0x7fffffff;
            int packed = l < buffered ? c.candidate_image[l] : 0;
            __syncwarp();
            sort_candidates(key, slot, packed);
            for (int i = 0; i < buffered && status == ok; ++i) {
                const float k = __shfl_sync(full_mask, key, i);
                if (k >= 4.0f * static_cast<float>(radius2) * 1.0001f) break;
                const int q = __shfl_sync(full_mask, slot, i);
                const int im = __shfl_sync(full_mask, packed, i);
                const Vec3<double> d{positions(q, 0) + image(im, 0) * box - p.x,
                                     positions(q, 1) + image(im, 1) * box - p.y,
                                     positions(q, 2) + image(im, 2) * box - p.z};
                const int before = planes;
                status = clip(c, planes, vertices, radius2, convert<Real>(d), q);
                ++clipped;
                cuts += planes > before;
            }
            buffered = 0;
        };

        // Visit neighbouring grid cells in order of their smallest possible
        // distance, 32 at a time: each lane checks one cell against 2R and
        // reads its particles, so the memory accesses overlap. The walk is
        // certified once the next cell's lower bound exceeds 2R.
        for (int first = 0; first < stencil_size && status == ok; first += lanes) {
            if (double(stencil_bound2(first)) >= 4.0 * static_cast<double>(radius2)) {
                certified = true;
                break;
            }
            ++rings;
            const int e = first + l;
            int begin = 0, end = 0, packed = 0;
            if (e < stencil_size) {
                const int code = stencil(e);
                const int u[3] = {home[0] + (code & 15) - 8, home[1] + ((code >> 4) & 15) - 8,
                                  home[2] + ((code >> 8) & 15) - 8};
                const double q[3] = {p.x, p.y, p.z};
                double gap2 = 0;
                int wrapped[3], shift[3];
                for (int a = 0; a < 3; ++a) {
                    const double g = Kokkos::max(0.0, Kokkos::max(u[a] * width - q[a], q[a] - (u[a] + 1) * width));
                    gap2 += g * g;
                    wrapped[a] = (u[a] + side) % side;
                    shift[a] = u[a] < 0 ? -1 : u[a] >= side ? 1 : 0;
                }
                if (gap2 < 4.0 * static_cast<double>(radius2)) {
                    const int key = grid.rank((wrapped[0] * side + wrapped[1]) * side + wrapped[2]);
                    begin = offsets(key);
                    end = offsets(key + 1);
                    packed = pack_image(shift[0], shift[1], shift[2]);
                }
            }
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
                const unsigned taking = __ballot_sync(full_mask, take);
                if (taking == 0) continue;
                if (buffered + __popc(taking) > lanes) flush();
                if (take) {
                    const int at = buffered + __popc(taking & ((1u << l) - 1));
                    c.candidate_distance2[at] = static_cast<float>(d2);
                    c.candidate_slot[at] = q;
                    c.candidate_image[at] = packed;
                }
                buffered += __popc(taking);
                __syncwarp();
            }
            if (buffered > 0) flush();
        }
        // Cells outside the stencil are at least stencil_reach away.
        if (status == ok && !certified && stencil_reach * stencil_reach >= 4.0 * static_cast<double>(radius2))
            certified = true;
        if (status == ok && !certified) status = uncertified;

        // Certification: no vertex may still touch the starting cube.
        if (status == ok) {
            bool touches = false;
            for (int k = 0; k < 2; ++k) {
                const int v = l + k * lanes;
                if (v < vertices)
                    for (int e = 0; e < 3; ++e) touches |= byte(c.corner[v], e) < box_planes;
            }
            if (__any_sync(full_mask, touches)) status = uncertified;
        }

        // Faces: lane l walks planes l and l + 32 around their vertices.
        for (int k = 0; k < 2; ++k)
            if (l + k * lanes < planes) c.first[l + k * lanes] = none;
        __syncwarp();
        for (int k = 0; k < 2; ++k) {
            const int v = l + k * lanes;
            if (v < vertices)
                for (int e = 0; e < 3; ++e) c.first[byte(c.corner[v], e)] = v;
        }
        __syncwarp();

        double volume = 0;
        Vec3<double> moment{0, 0, 0}, area_sum{0, 0, 0};
        double area_magnitude = 0;
        int faces = 0;
        bool broken = false;
        int owned_here = 0;
        int owned_plane[2] = {-1, -1};
        Vec3<double> owned_area[2], owned_centroid[2];
        const double sliver = 64 * Kokkos::Experimental::epsilon_v<Real> * static_cast<double>(radius2);
        for (int k = 0; k < 2 && status == ok; ++k) {
            const int plane = l + k * lanes;
            if (plane < box_planes || plane >= planes || c.first[plane] == none) continue;
            const int start = c.first[plane];
            const Vec3<double> normal{double(c.nx[plane]), double(c.ny[plane]), double(c.nz[plane])};
            const Vec3<double> first{double(c.x[start]), double(c.y[start]), double(c.z[start])};
            Vec3<double> previous = first, area{0, 0, 0}, face_moment{0, 0, 0};
            double weight = 0;
            int current = start, steps = 0;
            do {
                int at = 0;
                while (at < 3 && byte(c.corner[current], at) != plane) ++at;
                if (at == 3 || ++steps > max_vertices) {
                    broken = true;
                    break;
                }
                const int next = byte(c.adjacent[current], (at + 2) % 3);
                const Vec3<double> x{double(c.x[next]), double(c.y[next]), double(c.z[next])};
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
        if (status == ok && __any_sync(full_mask, broken)) status = degenerate;

        volume = warp_sum(volume);
        moment = {warp_sum(moment.x), warp_sum(moment.y), warp_sum(moment.z)};
        area_sum = {warp_sum(area_sum.x), warp_sum(area_sum.y), warp_sum(area_sum.z)};
        area_magnitude = warp_sum(area_magnitude);
        faces = warp_sum(faces);
        int owned = 0;
        const int first_owned = exclusive_scan(owned_here, owned);
        if (status == ok && owned > max_owned_faces) status = face_overflow;
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
            out.owned_count(t) = good ? owned : 0;
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
#else
        (void)member;
#endif
    }
};

#endif  // KOKKOS_ENABLE_CUDA

}  // namespace warp

// Build every cell with one warp per cell, then rebuild cells that overflowed
// the warp capacities with the per-thread kernel. Non-CUDA backends use the
// per-thread kernel directly.
template <class Real, class ExecSpace>
void construct_warp(const ExecSpace& exec, const Grid<typename ExecSpace::memory_space>& grid,
                    const Cells<typename ExecSpace::memory_space>& out) {
#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
        const int side = grid.cells_per_side;
        if (grid.n <= 0 || side < 3 || out.status.extent(0) != static_cast<std::size_t>(grid.n))
            throw std::invalid_argument("shell_voronoi::construct_warp: invalid grid or output size");
        const int max_ring = (side - 1) / 2;
        const Real half_width = static_cast<Real>(Kokkos::min(0.25 * grid.box, (max_ring + 1) * grid.cell_width));

        // Stencil of grid cells up to `reach` cells away; beyond it a cell is
        // left uncertified and rebuilt by the per-thread kernel.
        const int reach = Kokkos::min(max_ring, 4);
        std::vector<std::pair<float, int>> entries;
        for (int dx = -reach; dx <= reach; ++dx)
            for (int dy = -reach; dy <= reach; ++dy)
                for (int dz = -reach; dz <= reach; ++dz) {
                    auto beyond = [](int d) { return Kokkos::max(Kokkos::abs(d) - 1, 0); };
                    // Only cells up to `reach` widths away: beyond that the cube's
                    // corners alone would claim distances the stencil cannot vouch for.
                    if (beyond(dx) * beyond(dx) + beyond(dy) * beyond(dy) + beyond(dz) * beyond(dz) > reach * reach)
                        continue;
                    const double bound = grid.cell_width * std::sqrt(double(beyond(dx) * beyond(dx) +
                                                                            beyond(dy) * beyond(dy) +
                                                                            beyond(dz) * beyond(dz)));
                    entries.push_back({float(bound * bound), (dx + 8) | (dy + 8) << 4 | (dz + 8) << 8});
                }
        std::sort(entries.begin(), entries.end());
        // Round the bounds down so float storage never overstates a distance.
        Kokkos::View<int*, typename ExecSpace::memory_space> stencil("shell_stencil", entries.size());
        Kokkos::View<float*, typename ExecSpace::memory_space> bound2("shell_stencil_bound2", entries.size());
        {
            auto hs = Kokkos::create_mirror_view(stencil);
            auto hb = Kokkos::create_mirror_view(bound2);
            for (std::size_t i = 0; i < entries.size(); ++i) {
                hs(i) = entries[i].second;
                hb(i) = std::nextafter(entries[i].first, 0.0f);
            }
            Kokkos::deep_copy(exec, stencil, hs);
            Kokkos::deep_copy(exec, bound2, hb);
        }
        const int blocks = (grid.n + warp::warps_per_block - 1) / warp::warps_per_block;
        Kokkos::TeamPolicy<Kokkos::Cuda, Kokkos::LaunchBounds<warp::warps_per_block * warp::lanes, SHELL_WARP_MIN_BLOCKS>>
            policy(exec, blocks, warp::warps_per_block, warp::lanes);
        policy.set_scratch_size(0, Kokkos::PerTeam(warp::warps_per_block * sizeof(warp::Polyhedron<Real>)));
        Kokkos::parallel_for("shell_voronoi::construct_warp", policy,
                             warp::Kernel<Real, typename ExecSpace::memory_space>{
                                 grid, out, half_width, stencil, bound2, static_cast<int>(entries.size()),
                                 reach * grid.cell_width});
        construct<Real>(exec, grid, out, /*only_failed=*/true);
        return;
    }
#endif
    construct<Real>(exec, grid, out);
}

}  // namespace shell_voronoi
