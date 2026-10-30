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
// k to corner k + 1. Vertices stay in their slots: a cut puts its new
// vertices into free slots and rewires only the edges it crosses. A 64-bit
// mask (in registers) marks the slots in use.
template <class Real>
struct alignas(16) Polyhedron {
    Real nx[max_planes], ny[max_planes], nz[max_planes], offset[max_planes];
    int neighbour[max_planes];
    Real x[max_vertices], y[max_vertices], z[max_vertices];
    Real side[max_vertices];  // signed distance to the cutting plane
    std::uint32_t corner[max_vertices];
    std::uint32_t adjacent[max_vertices];
    std::uint8_t free_slots[max_vertices];  // stack of the slots not in use
    std::uint8_t crossing[max_vertices];    // edges a cut crosses: removed vertex | edge << 6
    std::uint8_t starts[max_planes];        // new vertex whose hole edge starts at a plane
    std::uint8_t ends[max_planes];          // ... and ends at a plane
    std::uint8_t first[max_planes];         // a vertex on each face, for the face walk
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

// Grid-cell offsets (4 bits per axis, biased by 8) sorted by the squared
// smallest distance any point of the home cell can have to them.
template <class MemorySpace>
struct Stencil {
    Kokkos::View<int*, MemorySpace> offsets;
    Kokkos::View<float*, MemorySpace> bound2;
    double reach = 0;  // cells outside the stencil are at least this far
};

// Every grid cell up to `reach` = min(largest ring, 4) cell widths away;
// a cell that needs more is left uncertified for the per-thread kernel.
template <class ExecSpace>
Stencil<typename ExecSpace::memory_space> make_stencil(const ExecSpace& exec, int cells_per_side, double cell_width) {
    const int reach = std::min((cells_per_side - 1) / 2, 4);
    std::vector<std::pair<float, int>> entries;
    auto beyond = [](int d) { return std::max(std::abs(d) - 1, 0); };
    for (int dx = -reach; dx <= reach; ++dx)
        for (int dy = -reach; dy <= reach; ++dy)
            for (int dz = -reach; dz <= reach; ++dz) {
                // Only cells up to `reach` widths away: beyond that the cube's
                // corners alone would claim distances the stencil cannot vouch for.
                const int gaps = beyond(dx) * beyond(dx) + beyond(dy) * beyond(dy) + beyond(dz) * beyond(dz);
                if (gaps > reach * reach) continue;
                const double bound = cell_width * std::sqrt(double(gaps));
                // Round down so float storage never overstates a distance.
                entries.push_back({std::nextafter(float(bound * bound), 0.0f), (dx + 8) | (dy + 8) << 4 | (dz + 8) << 8});
            }
    std::sort(entries.begin(), entries.end());
    using Memory = typename ExecSpace::memory_space;
    Stencil<Memory> s{Kokkos::View<int*, Memory>("shell_stencil", entries.size()),
                      Kokkos::View<float*, Memory>("shell_stencil_bound2", entries.size()), reach * cell_width};
    auto offsets = Kokkos::create_mirror_view(s.offsets);
    auto bound2 = Kokkos::create_mirror_view(s.bound2);
    for (std::size_t i = 0; i < entries.size(); ++i) {
        offsets(i) = entries[i].second;
        bound2(i) = entries[i].first;
    }
    Kokkos::deep_copy(exec, s.offsets, offsets);
    Kokkos::deep_copy(exec, s.bound2, bound2);
    return s;
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

// Stores byte k of a packed word in shared memory; other lanes may store its
// other bytes at the same time.
__device__ inline void store_byte(std::uint32_t& packed, int k, int value) {
    reinterpret_cast<std::uint8_t*>(&packed)[k] = static_cast<std::uint8_t>(value);
}

template <class Real>
__device__ void start_cube(Polyhedron<Real>& c, Real h, int& planes, std::uint64_t& live) {
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
    for (int k = 0; k < per_lane; ++k) {
        const int slot = l + k * lanes;
        // Slots 8.. start free; their corners name no plane, so no slot can
        // pass for a new vertex (see clip).
        if (slot >= 8) c.corner[slot] = ~std::uint32_t(0);
        if (slot < max_vertices - 8) c.free_slots[slot] = max_vertices - 1 - slot;
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
    __syncwarp(all_lanes);
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
    __syncwarp(all_lanes);
    planes = box_planes;
    live = 0xff;
}

// Clip by the bisector with a particle at offset d (all lanes pass the same
// d). Returns ok when the plane does not cut or after the cut is applied.
template <class Real>
__device__ int clip(Polyhedron<Real>& c, int& planes, std::uint64_t& live, Real& radius2, Vec3<Real> d,
                    int neighbour) {
    const int l = lane();
    const Real offset = Real(0.5) * dot(d, d);
    // A vertex is cut off when side > 64 eps |d| R, compared in squares.
    const Real scale = Real(64) * Kokkos::Experimental::epsilon_v<Real>;
    const Real tolerance2 = scale * scale * dot(d, d) * radius2;
    bool out[per_lane] = {};
    Real side[per_lane] = {};
    for (int k = 0; k < per_lane; ++k) {
        const int w = l + k * lanes;
        if (!(live >> w & 1)) continue;
        side[k] = d.x * c.x[w] + d.y * c.y[w] + d.z * c.z[w] - offset;
        c.side[w] = side[k];
        out[k] = side[k] > 0 && side[k] * side[k] > tolerance2;
    }
    const std::uint64_t removed = vertex_ballot(out);
    if (removed == 0) return ok;
    if (planes == max_planes) return plane_overflow;
    __syncwarp(all_lanes);

    // Each edge from a removed vertex to a kept one crosses the plane and
    // becomes a new vertex. One ballot per edge numbers them; lane j builds
    // new vertices j and j + 32.
    const unsigned below = (1u << l) - 1;
    int created = 0;
    for (int k = 0; k < per_lane; ++k) {
        const int w = l + k * lanes;
        const std::uint32_t adjacent = out[k] ? c.adjacent[w] : 0;
        for (int e = 0; e < 3; ++e) {
            const bool crossing = out[k] && !(removed >> byte(adjacent, e) & 1);
            const unsigned crossings = __ballot_sync(all_lanes, crossing);
            const int number = created + __popc(crossings & below);
            if (crossing && number < max_vertices) c.crossing[number] = w | e << 6;
            created += __popc(crossings);
        }
    }
    const int in_use = __popcll(live);
    const int kept = in_use - __popcll(removed);
    // New vertices take slots that were free before the cut.
    if (in_use + created > max_vertices) return vertex_overflow;
    if (kept + created < 4) return degenerate;
    const int plane = planes;
    const int free = max_vertices - in_use;
    __syncwarp(all_lanes);

    // New vertex j lies on the edge from removed w to kept u, between planes
    // a and b. It takes a free slot; u's edge back now ends at it.
    Real radius = 0;
    int slot[per_lane], kept_end[per_lane], a[per_lane], b[per_lane];
    for (int k = 0; k < per_lane; ++k) {
        const int j = l + k * lanes;
        if (j >= created) continue;
        const int code = c.crossing[j], w = code & 63, e = code >> 6;
        const std::uint32_t corner = c.corner[w];
        const int u = byte(c.adjacent[w], e);
        const int s = c.free_slots[free - 1 - j];
        const Real t = c.side[w] / (c.side[w] - c.side[u]);
        const Real px = c.x[w] + (c.x[u] - c.x[w]) * t, py = c.y[w] + (c.y[u] - c.y[w]) * t,
                   pz = c.z[w] + (c.z[u] - c.z[w]) * t;
        c.x[s] = px;
        c.y[s] = py;
        c.z[s] = pz;
        radius = Kokkos::max(radius, px * px + py * py + pz * pz);
        a[k] = byte(corner, e);
        b[k] = byte(corner, (e + 1) % 3);
        c.corner[s] = pack(a[k], b[k], plane);
        const std::uint32_t back = c.adjacent[u];
        store_byte(c.adjacent[u], byte(back, 0) == w ? 0 : byte(back, 1) == w ? 1 : 2, s);
        c.starts[a[k]] = s;
        c.ends[b[k]] = s;
        slot[k] = s;
        kept_end[k] = u;
    }
    __syncwarp(all_lanes);
    // Across (a, b) is u, across (b, plane) the new vertex whose hole edge
    // starts at b, across (plane, a) the one whose hole edge ends at a.
    for (int k = 0; k < per_lane; ++k)
        if (l + k * lanes < created) c.adjacent[slot[k]] = pack(kept_end[k], c.starts[b[k]], c.ends[a[k]]);
    // Removed slots go back on the stack; slots that took new vertices (the
    // only ones on the new plane) join the polyhedron.
    bool in[per_lane];
    for (int k = 0; k < per_lane; ++k) {
        const int w = l + k * lanes;
        if (out[k]) c.free_slots[free - created + __popcll(removed & ((std::uint64_t(1) << w) - 1))] = w;
        const bool was = live >> w & 1;
        const bool fresh = !was && byte(c.corner[w], 2) == plane;
        in[k] = fresh || (was && !out[k]);
        if (was && !out[k]) radius = Kokkos::max(radius, c.x[w] * c.x[w] + c.y[w] * c.y[w] + c.z[w] * c.z[w]);
    }
    live = vertex_ballot(in);
    if (l == 0) {
        c.nx[plane] = d.x;
        c.ny[plane] = d.y;
        c.nz[plane] = d.z;
        c.offset[plane] = offset;
        c.neighbour[plane] = neighbour;
    }
    planes = plane + 1;
    radius2 = warp_max(radius);
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
template <class Real, class MemorySpace>
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
        const int stencil_size = static_cast<int>(stencil.offsets.extent(0));
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

        int planes = 0;
        std::uint64_t live = 0;  // slots holding vertices
        Real radius2 = 3 * half_width * half_width;
        start_cube(c, half_width, planes, live);
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
                status = clip(c, planes, live, radius2, convert<Real>(d), q);
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
            if (double(stencil.bound2(first)) >= 4.0 * static_cast<double>(radius2)) {
                certified = true;
                break;
            }
            ++rings;
            const int e = first + l;
            int begin = 0, end = 0, packed = 0;
            if (e < stencil_size) {
                const int code = stencil.offsets(e);
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
                const unsigned taking = __ballot_sync(all_lanes, take);
                if (taking == 0) continue;
                if (buffered + __popc(taking) > lanes) flush();
                if (take) {
                    const int at = buffered + __popc(taking & ((1u << l) - 1));
                    c.candidate_distance2[at] = static_cast<float>(d2);
                    c.candidate_slot[at] = q;
                    c.candidate_image[at] = packed;
                }
                buffered += __popc(taking);
                __syncwarp(all_lanes);
            }
            if (buffered > 0) flush();
        }
        // Cells outside the stencil are at least stencil_reach away.
        if (status == ok && !certified && stencil.reach * stencil.reach >= 4.0 * static_cast<double>(radius2))
            certified = true;
        if (status == ok && !certified) status = uncertified;

        // Certification: no vertex may still touch the starting cube.
        if (status == ok) {
            bool touches = false;
            for (int k = 0; k < per_lane; ++k) {
                const int v = l + k * lanes;
                if (live >> v & 1)
                    for (int e = 0; e < 3; ++e) touches |= byte(c.corner[v], e) < box_planes;
            }
            if (__any_sync(all_lanes, touches)) status = uncertified;
        }

        // Faces: lane l walks planes l and l + 32 around their vertices.
        for (int k = 0; k < per_lane; ++k)
            if (l + k * lanes < planes) c.first[l + k * lanes] = none;
        __syncwarp(all_lanes);
        for (int k = 0; k < per_lane; ++k) {
            const int v = l + k * lanes;
            if (live >> v & 1)
                for (int e = 0; e < 3; ++e) c.first[byte(c.corner[v], e)] = v;
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
            const Vec3<double> first{double(c.x[start]), double(c.y[start]), double(c.z[start])};
            Vec3<double> previous = first, area{0, 0, 0}, face_moment{0, 0, 0};
            double weight = 0;
            int vertex = start, steps = 0;
            do {
                int at = 0;
                while (at < 3 && byte(c.corner[vertex], at) != plane) ++at;
                if (at == 3 || ++steps > max_vertices) {
                    broken = true;
                    break;
                }
                const int next = byte(c.adjacent[vertex], (at + 2) % 3);
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
                    const Cells<typename ExecSpace::memory_space>& out) {
#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
        if (out.rings.extent(0) == 0) {
            const int side = grid.cells_per_side;
            if (grid.n <= 0 || side < 3 || out.status.extent(0) != static_cast<std::size_t>(grid.n))
                throw std::invalid_argument("shell_voronoi::construct_warp: invalid grid or output size");
            const int max_ring = (side - 1) / 2;
            const Real half_width =
                static_cast<Real>(Kokkos::min(0.25 * grid.box, (max_ring + 1) * grid.cell_width));
            using Memory = typename ExecSpace::memory_space;
            const warp::Stencil<Memory> stencil = warp::make_stencil(exec, side, grid.cell_width);
            Kokkos::View<int, Memory> next_cell("shell_warp_next_cell");
            const int blocks = Kokkos::min((grid.n + warp::cells_per_block - 1) / warp::cells_per_block,
                                           2 * warp::min_blocks_per_sm * exec.cuda_device_prop().multiProcessorCount);
            // A block holds cells_per_block warps; each warp is a team's vector range.
            Kokkos::TeamPolicy<Kokkos::Cuda,
                               Kokkos::LaunchBounds<warp::cells_per_block * warp::lanes, warp::min_blocks_per_sm>>
                policy(exec, blocks, warp::cells_per_block, warp::lanes);
            policy.set_scratch_size(0, Kokkos::PerTeam(warp::cells_per_block * sizeof(warp::Polyhedron<Real>)));
            Kokkos::parallel_for("shell_voronoi::construct_warp", policy,
                                 warp::Kernel<Real, Memory>{grid, out, half_width, stencil, next_cell});
            detail::construct_with_overflow<Real>(exec, grid, out, detail::Pass::failed);
            return;
        }
    }
#endif
    construct<Real>(exec, grid, out);
}

}  // namespace shell_voronoi
