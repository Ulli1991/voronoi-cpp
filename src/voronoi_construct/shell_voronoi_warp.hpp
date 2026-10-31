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
// Cells that overflow the warp capacities, or that its walk cannot certify,
// are rebuilt by the per-thread kernel. On other backends construct_warp
// simply calls that kernel.
namespace shell_voronoi {
namespace warp {

constexpr int lanes = 32;
constexpr int max_planes = 64;
constexpr int max_vertices = 64;
constexpr int per_lane = max_vertices / lanes;  // vertices (and planes) each lane owns
// Four cells (warps) per block: their polyhedra (4.4 KB each) take
// 18 KB of block scratch.
constexpr int cells_per_block = 4;
// Caps the registers so that 5 blocks fit per SM: about 4 % faster on an A100.
constexpr int min_blocks_per_sm = 5;
constexpr std::uint8_t none = 255;

// One warp's polyhedron. Corners and adjacency hold three 8-bit indices each:
// corner k is a plane, adjacency k is the vertex across the edge from corner
// k to corner k + 1. Vertices stay in their slots: a cut puts its new
// vertices into free slots and rewires only the edges it crosses. A 64-bit
// mask (in registers) marks the slots in use.
struct alignas(16) Polyhedron {
    // Plane normals only weight the face centroids: a rounded normal scales
    // every triangle of a face alike, so float is enough.
    float nx[max_planes], ny[max_planes], nz[max_planes];
    int neighbour[max_planes];
    double x[max_vertices], y[max_vertices], z[max_vertices];
    double side[max_vertices];  // signed distance to the cutting plane
    std::uint32_t corner[max_vertices];
    std::uint32_t adjacent[max_vertices];
    std::uint8_t free_slots[max_vertices];  // stack of the slots not in use
    std::uint32_t crossing[max_vertices];   // edges a cut crosses: removed, kept vertex, planes a, b
    std::uint8_t starts[max_planes];        // new vertex whose hole edge starts at a plane
    std::uint8_t ends[max_planes];          // ... and ends at a plane
    std::uint8_t first[max_planes];         // a vertex on each face, for the face walk
    float candidate_distance2[lanes];
    int candidate_slot[lanes];
    int candidate_image[lanes];
    // Owned faces without rings: the warp claims face_block entries of the
    // pool at a time, so the pool's counter sees one atomic per block
    // instead of one per cell; this warp's block is [face_at, face_end).
    long long face_at, face_end;
};

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

// Largest of the lanes' non-negative values. Their bit patterns order like
// unsigned integers, so 32-bit warp reductions find it exactly.
__device__ inline float warp_max_nonnegative(float v) {
    return __uint_as_float(__reduce_max_sync(all_lanes, __float_as_uint(v)));
}
__device__ inline double warp_max_nonnegative(double v) {
    const auto bits = static_cast<unsigned long long>(__double_as_longlong(v));
    const auto high_word = static_cast<unsigned>(bits >> 32), low_word = static_cast<unsigned>(bits);
    const unsigned high = __reduce_max_sync(all_lanes, high_word);
    const unsigned low = __reduce_max_sync(all_lanes, high_word == high ? low_word : 0u);
    return __longlong_as_double(static_cast<long long>(static_cast<unsigned long long>(high) << 32 | low));
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

__device__ inline void start_cube(Polyhedron& c, double h, int& planes, std::uint64_t& live) {
    const int l = lane();
    if (l < box_planes) {
        const int axis = l / 2;
        const float sign = l % 2 == 0 ? 1.0f : -1.0f;
        c.nx[l] = axis == 0 ? sign : 0.0f;
        c.ny[l] = axis == 1 ? sign : 0.0f;
        c.nz[l] = axis == 2 ? sign : 0.0f;
        c.neighbour[l] = -1;
    }
    for (int k = 0; k < per_lane; ++k) {
        const int slot = l + k * lanes;
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
__device__ inline int clip(Polyhedron& c, int& planes, std::uint64_t& live, double& radius2, Vec3 d,
                    int neighbour) {
    const int l = lane();
    const double offset = 0.5 * dot(d, d);
    // A vertex is cut off when side > 64 eps |d| R, compared in squares.
    const double scale = 64 * Kokkos::Experimental::epsilon_v<double>;
    const double tolerance2 = scale * scale * dot(d, d) * radius2;
    bool out[per_lane] = {};
    double radius = 0;  // largest squared distance of a kept vertex, then of any
    for (int k = 0; k < per_lane; ++k) {
        const int w = l + k * lanes;
        if (!(live >> w & 1)) continue;
        const double x = c.x[w], y = c.y[w], z = c.z[w];
        const double side = d.x * x + d.y * y + d.z * z - offset;
        c.side[w] = side;
        out[k] = side > 0 && side * side > tolerance2;
        if (!out[k]) radius = Kokkos::max(radius, x * x + y * y + z * z);
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
        if (!(removed >> (k * lanes))) continue;  // the whole warp: no removed vertex here
        const int w = l + k * lanes;
        std::uint32_t corner = 0, adjacent = 0;
        if (out[k]) {
            corner = c.corner[w];
            adjacent = c.adjacent[w];
        }
        for (int e = 0; e < 3; ++e) {
            const int u = byte(adjacent, e);
            const bool crossing = out[k] && !(removed >> u & 1);
            const unsigned crossings = __ballot_sync(all_lanes, crossing);
            const int number = created + __popc(crossings & below);
            if (crossing && number < max_vertices)
                c.crossing[number] = pack(w, u, byte(corner, e)) | std::uint32_t(byte(corner, (e + 1) % 3)) << 24;
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
    int slot[per_lane], kept_end[per_lane], a[per_lane], b[per_lane];
    std::uint64_t made = 0;  // slots this lane filled
    for (int k = 0; k < per_lane; ++k) {
        const int j = l + k * lanes;
        if (j >= created) continue;
        const std::uint32_t edge = c.crossing[j];
        const int w = byte(edge, 0), u = byte(edge, 1);
        const int s = c.free_slots[free - 1 - j];
        const double t = c.side[w] / (c.side[w] - c.side[u]);
        const double px = c.x[w] + (c.x[u] - c.x[w]) * t, py = c.y[w] + (c.y[u] - c.y[w]) * t,
                   pz = c.z[w] + (c.z[u] - c.z[w]) * t;
        c.x[s] = px;
        c.y[s] = py;
        c.z[s] = pz;
        radius = Kokkos::max(radius, px * px + py * py + pz * pz);
        a[k] = byte(edge, 2);
        b[k] = byte(edge, 3);
        c.corner[s] = pack(a[k], b[k], plane);
        const std::uint32_t back = c.adjacent[u];
        store_byte(c.adjacent[u], byte(back, 0) == w ? 0 : byte(back, 1) == w ? 1 : 2, s);
        c.starts[a[k]] = s;
        c.ends[b[k]] = s;
        slot[k] = s;
        kept_end[k] = u;
        made |= std::uint64_t(1) << s;
    }
    const unsigned made_low = __reduce_or_sync(all_lanes, static_cast<unsigned>(made));
    const unsigned made_high = __reduce_or_sync(all_lanes, static_cast<unsigned>(made >> 32));
    __syncwarp(all_lanes);
    // Across (a, b) is u, across (b, plane) the new vertex whose hole edge
    // starts at b, across (plane, a) the one whose hole edge ends at a.
    for (int k = 0; k < per_lane; ++k)
        if (l + k * lanes < created) c.adjacent[slot[k]] = pack(kept_end[k], c.starts[b[k]], c.ends[a[k]]);
    // Removed slots go back on the stack.
    for (int k = 0; k < per_lane; ++k) {
        const int w = l + k * lanes;
        if (out[k]) c.free_slots[free - created + __popcll(removed & ((std::uint64_t(1) << w) - 1))] = w;
    }
    live = (live & ~removed) | made_low | std::uint64_t(made_high) << 32;
    if (l == 0) {
        c.nx[plane] = static_cast<float>(d.x);
        c.ny[plane] = static_cast<float>(d.y);
        c.nz[plane] = static_cast<float>(d.z);
        c.neighbour[plane] = neighbour;
    }
    planes = plane + 1;
    radius2 = warp_max_nonnegative(radius);
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
// with_rings: record face rings (Cells::with_rings) instead of face geometry.
// local: build only the cells of `selection` (otherwise it is unused).
// cubic: the grid is one cube (Grid::cube()): the walk's shorter arithmetic.
template <class MemorySpace, bool adaptive, bool with_rings, bool local = false, bool cubic = false>
struct Kernel {
    using Member = typename Kokkos::TeamPolicy<Kokkos::Cuda>::member_type;
    Grid<MemorySpace> grid;
    Cells<MemorySpace> out;
    double half_width;
    Stencil<MemorySpace> stencil;
    // Persistent warps (about two waves of resident blocks) take cells from
    // this counter until none are left, so a slow cell holds up only its own
    // warp: 1.06x on uniform random points over one block per four cells.
    Kokkos::View<int, MemorySpace> next_cell;
    Selection<MemorySpace> selection;

    KOKKOS_FUNCTION void operator()(const Member& member) const {
#if defined(__CUDA_ARCH__)
        auto* base = static_cast<Polyhedron*>(
            member.team_shmem().get_shmem(cells_per_block * sizeof(Polyhedron)));
        Polyhedron& c = base[member.team_rank()];
        if (lane() == 0) c.face_at = c.face_end = 0;
        while (true) {
            int taken = 0;
            if (lane() == 0) taken = Kokkos::atomic_fetch_add(&next_cell(), 1);
            const int t = __shfl_sync(all_lanes, taken, 0);
            if (t >= (local ? selection.count(grid.n) : grid.n)) return;  // the whole warp
            build(c, local ? selection.slot(t) : t);
        }
#else
        (void)member;
#endif
    }

#if defined(__CUDA_ARCH__)
    // Build cell t (a slot) in polyhedron c, with all lanes of the warp.
    __device__ __forceinline__ void build(Polyhedron& c, int t) const {
        const int l = lane();
        const auto& positions = grid.positions;
        const double* box = grid.box.side;
        const double side[3] = {box[0], box[cubic ? 0 : 1], box[cubic ? 0 : 2]};
        const Vec3 p{positions(t, 0), positions(t, 1), positions(t, 2)};

        int planes = 0;
        std::uint64_t live = 0;  // slots holding vertices
        double radius2 = 3 * half_width * half_width;
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
            // Lane i loads candidate i at once; the loop broadcasts it.
            Vec3 mine{0, 0, 0};
            if (l < buffered)
                mine = {positions(slot, 0) + image(packed, 0) * side[0] - p.x,
                        positions(slot, 1) + image(packed, 1) * side[1] - p.y,
                        positions(slot, 2) + image(packed, 2) * side[2] - p.z};
            for (int i = 0; i < buffered && status == ok; ++i) {
                const float k = __shfl_sync(all_lanes, key, i);
                if (k >= 4.0f * static_cast<float>(radius2) * 1.0001f) break;
                const int q = __shfl_sync(all_lanes, slot, i);
                const Vec3 d{__shfl_sync(all_lanes, mine.x, i), __shfl_sync(all_lanes, mine.y, i),
                             __shfl_sync(all_lanes, mine.z, i)};
                const int before = planes;
                status = clip(c, planes, live, radius2, d, q);
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
        home_cell<cubic>(p, grid, level, home);
        while (status == ok) {
            const double width = grid.narrowest / (1 << level);
            const int reach = level_reach(level, cubic ? 1 : grid.fewest), entries = stencil.ends(reach);
            for (int first = 0; first < entries && status == ok; first += lanes) {
                if (double(stencil.bound2(first)) * width * width >= 4.0 * radius2) {
                    certified = true;
                    break;
                }
                ++rings;
                const int e = first + l;
                int begin = 0, end = 0, packed = 0;
                const double reach2 = 4.0 * radius2;
                if (e < entries)
                    stencil_cell<adaptive, cubic>(grid, level, home, finer, stencil.offsets(e), p, reach2, begin, end,
                                                  packed);
                const int most = warp_max(end - begin);
                for (int k = 0; k < most && status == ok; ++k) {
                    const int q = begin + k;
                    double d2 = 0;
                    bool take = false;
                    if (q < end && q != t) {
                        const double ddx = positions(q, 0) + image(packed, 0) * side[0] - p.x;
                        const double ddy = positions(q, 1) + image(packed, 1) * side[1] - p.y;
                        const double ddz = positions(q, 2) + image(packed, 2) * side[2] - p.z;
                        d2 = ddx * ddx + ddy * ddy + ddz * ddz;
                        take = d2 < 4.0 * radius2;
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
            if (reach * width * reach * width >= 4.0 * radius2) {
                certified = true;
                break;
            }
            if (!adaptive || level == 2) break;  // no coarser grid fits the box
            finer = (home[0] & 1) | (home[1] & 1) << 1 | (home[2] & 1) << 2;
            for (int a = 0; a < 3; ++a) home[a] >>= 1;
            --level;
        }
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

        // Outputs and the owner rule use particle ids: the planes' slots
        // become the particles there.
        const int id = grid.order(t);
        for (int k = 0; k < per_lane; ++k) {
            const int plane = l + k * lanes;
            if (plane >= box_planes && plane < planes) c.neighbour[plane] = grid.order(c.neighbour[plane]);
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
        Vec3 moment{0, 0, 0}, area_sum{0, 0, 0};
        double area_magnitude = 0;
        int faces = 0;
        bool broken = false;
        int owned_here = 0;
        int owned_plane[per_lane], owned_length[per_lane];
        for (int k = 0; k < per_lane; ++k) owned_plane[k] = -1;
        Vec3 owned_area[per_lane], owned_centroid[per_lane];
        const double sliver = 64 * Kokkos::Experimental::epsilon_v<double> * radius2;
        for (int k = 0; k < per_lane && status == ok; ++k) {
            const int plane = l + k * lanes;
            if (plane < box_planes || plane >= planes || c.first[plane] == none) continue;
            const int start = c.first[plane];
            const Vec3 normal{double(c.nx[plane]), double(c.ny[plane]), double(c.nz[plane])};
            const Vec3 first{c.x[start], c.y[start], c.z[start]};
            // Fan triangles (first, previous, x): twice the area vector, and
            // the sum of (previous + x) weighted by the triangles' areas
            // projected on the normal; divided once per face.
            Vec3 previous = first, twice_area{0, 0, 0}, weighted{0, 0, 0};
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
                const Vec3 x{c.x[next], c.y[next], c.z[next]};
                const Vec3 twice = cross(previous - first, x - first);
                twice_area = twice_area + twice;
                const double w = dot(twice, normal);
                weight += w;
                weighted = weighted + (previous + x) * w;
                previous = x;
                vertex = next;
            } while (vertex != start);
            if (broken) break;
            const Vec3 area = twice_area * 0.5;
            // A sliver face has no meaningful centroid; use a vertex.
            const Vec3 centroid = weight > 0 ? (first + weighted * (1.0 / weight)) * (1.0 / 3.0) : first;
            // The pyramid over the face from the generator: volume A.first / 3,
            // centroid at 3/4 of the face centroid (idea from Doug Rennehan's
            // plume, commit 6cbb429).
            const double pyramid = dot(area, first) * (1.0 / 3.0);
            volume += pyramid;
            moment = moment + centroid * (0.75 * pyramid);
            area_sum = area_sum + area;
            const double magnitude = Kokkos::sqrt(dot(area, area));
            if (magnitude <= sliver) continue;
            ++faces;
            area_magnitude += magnitude;
            if (local ? selection.owns(id, c.neighbour[plane]) : c.neighbour[plane] > id) {
                owned_plane[owned_here] = plane;
                owned_length[owned_here] = steps;
                owned_area[owned_here] = area;
                owned_centroid[owned_here] = centroid;
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
        int offset = exclusive_scan(owned_here, owned);
        if (status == ok && !(volume > 0)) status = degenerate;

        // The owned faces go to one block per cell that lane 0 claims: with
        // rings each face as its neighbour, ring length and ring (see
        // Cells::rings), otherwise its neighbour, area and centroid.
        int fits = 1;
        long long block = 0;
        if constexpr (with_rings) {
            int need = 0;
            for (int m = 0; m < owned_here; ++m) need += 2 + owned_length[m];
            int total = 0;
            offset = exclusive_scan(need, total);
            if (l == 0 && status == ok) {
                fits = claim_rings(out, id, total);
                block = out.ring_start(id);
            }
        } else if (l == 0 && status == ok) {
            if (c.face_at + owned > c.face_end) {  // a new block for this warp
                const int size = Kokkos::max(owned, int{face_block});
                long long at = 0;
                fits = claim(out.face_used, static_cast<long long>(out.neighbour.extent(0)), 0, size, at);
                c.face_at = at;
                c.face_end = fits ? at + size : at;
            }
            if (fits) {
                block = c.face_at;
                out.face_start(id) = static_cast<int>(block);
                c.face_at += owned;
            }
        }
        if (__shfl_sync(all_lanes, fits, 0) == 0) status = face_overflow;
        if (status == ok) {
            long long at = __shfl_sync(all_lanes, block, 0) + offset;
            for (int m = 0; m < owned_here; ++m) {
                const int plane = owned_plane[m];
                if constexpr (with_rings) {
                    out.rings(at) = c.neighbour[plane];
                    out.rings(at + 1) = owned_length[m];
                    at += 2;
                    // Walk the face again; the plane shared with the next
                    // vertex bounds the face there.
                    for (int vertex = c.first[plane], k = 0; k < owned_length[m]; ++k) {
                        int corner = 0;
                        while (byte(c.corner[vertex], corner) != plane) ++corner;
                        out.rings(at++) = c.neighbour[byte(c.corner[vertex], (corner + 2) % 3)];
                        vertex = byte(c.adjacent[vertex], (corner + 2) % 3);
                    }
                } else {
                    const int f = static_cast<int>(at++);
                    out.neighbour(f) = c.neighbour[plane];
                    out.area(f, 0) = owned_area[m].x;
                    out.area(f, 1) = owned_area[m].y;
                    out.area(f, 2) = owned_area[m].z;
                    out.face_centroid(f, 0) = owned_centroid[m].x;
                    out.face_centroid(f, 1) = owned_centroid[m].y;
                    out.face_centroid(f, 2) = owned_centroid[m].z;
                }
            }
        }
        if (l == 0) {
            const bool good = status == ok;
            out.status(id) = status;
            out.face_count(id) = good ? faces : 0;
            out.owned_count(id) = good ? owned : 0;
            out.volume(id) = good ? volume : 0.0;
            out.centroid(id, 0) = good ? moment.x / volume : 0.0;
            out.centroid(id, 1) = good ? moment.y / volume : 0.0;
            out.centroid(id, 2) = good ? moment.z / volume : 0.0;
            if (out.closure.extent(0) > 0) {
                out.closure(id) = good ? Kokkos::sqrt(dot(area_sum, area_sum)) / area_magnitude : 0.0;
                out.work(id, 0) = rings;
                out.work(id, 1) = clipped;
                out.work(id, 2) = cuts;
            }
        }
    }
#endif
};

#endif  // KOKKOS_ENABLE_CUDA

}  // namespace warp

// Build the selected cells (default: all) with one warp per cell, then
// rebuild cells that overflowed the warp capacities with the per-thread
// kernel; with face rings if `out` records them (Cells::with_rings). Other
// backends use the per-thread kernel throughout.
template <class ExecSpace>
void construct_warp(const ExecSpace& exec, const Grid<typename ExecSpace::memory_space>& grid,
                    Cells<typename ExecSpace::memory_space>& out,
                    const Selection<typename ExecSpace::memory_space>& selection = {}) {
#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
        if (!selection.all() && out.with_rings()) {  // not needed: the per-thread kernel
            construct(exec, grid, out, selection);
            return;
        }
        if (grid.n <= 0 || out.status.extent(0) != static_cast<std::size_t>(grid.n))
            throw std::invalid_argument("shell_voronoi::construct_warp: invalid grid or output size");
        const bool rings = out.with_rings(), cube = grid.cube();  // cube: the plain kernels' shorter walk
        const double half_width = detail::start_half_width(grid.box);
        using Memory = typename ExecSpace::memory_space;
        const Stencil<Memory> stencil = make_stencil(exec);
        Kokkos::View<int, Memory> next_cell("shell_warp_next_cell");
        const int blocks = Kokkos::min((selection.count(grid.n) + warp::cells_per_block - 1) / warp::cells_per_block,
                                       2 * warp::min_blocks_per_sm * exec.cuda_device_prop().multiProcessorCount);
        // A block holds cells_per_block warps; each warp is a team's vector range.
        Kokkos::TeamPolicy<Kokkos::Cuda,
                           Kokkos::LaunchBounds<warp::cells_per_block * warp::lanes, warp::min_blocks_per_sm>>
            policy(exec, blocks, warp::cells_per_block, warp::lanes);
        policy.set_scratch_size(0, Kokkos::PerTeam(warp::cells_per_block * sizeof(warp::Polyhedron)));
        // Near-uniform input (every particle at the base level) takes the
        // kernel without level changes and fine-cell searches: its smaller
        // code is 4 % faster on uniform random points.
        int off_base = 0;
        const auto level = grid.level;
        const int base = grid.levels;
        Kokkos::parallel_reduce(
            "shell_voronoi::off_base_levels", Kokkos::RangePolicy<ExecSpace>(exec, 0, grid.n),
            KOKKOS_LAMBDA(int t, int& count) { count += level(t) != base; }, off_base);
        detail::until_pool_fits(exec, out, [&] {
            Kokkos::deep_copy(exec, next_cell, 0);
            if (!selection.all())
                Kokkos::parallel_for("shell_voronoi::construct_warp_local", policy,
                                     warp::Kernel<Memory, true, false, true>{grid, out, half_width, stencil, next_cell,
                                                                             selection});
            else if (rings)
                Kokkos::parallel_for("shell_voronoi::construct_warp_rings", policy,
                                     warp::Kernel<Memory, true, true>{grid, out, half_width, stencil, next_cell, selection});
            else if (off_base > 0 && cube)
                Kokkos::parallel_for("shell_voronoi::construct_warp", policy,
                                     warp::Kernel<Memory, true, false, false, true>{grid, out, half_width, stencil,
                                                                                    next_cell, selection});
            else if (off_base > 0)
                Kokkos::parallel_for("shell_voronoi::construct_warp", policy,
                                     warp::Kernel<Memory, true, false>{grid, out, half_width, stencil, next_cell, selection});
            else if (cube)
                Kokkos::parallel_for("shell_voronoi::construct_warp", policy,
                                     warp::Kernel<Memory, false, false, false, true>{grid, out, half_width, stencil,
                                                                                     next_cell, selection});
            else
                Kokkos::parallel_for("shell_voronoi::construct_warp", policy,
                                     warp::Kernel<Memory, false, false>{grid, out, half_width, stencil, next_cell, selection});
            detail::construct_cells<detail::RoundedCell>(exec, grid, stencil, out, detail::Pass::failed, selection);
        });
        return;
    }
#endif
    construct(exec, grid, out, selection);
}

}  // namespace shell_voronoi
