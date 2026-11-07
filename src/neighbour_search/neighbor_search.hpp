#pragma once

#include "exact_predicates.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cub/device/device_radix_sort.cuh>
#endif

// Stage 2, index (3D and 2D): called by Mesh::index after snap (configure when n changes, then build); the sorted cell
// list feeds the clipping start (shell_voronoi make_grid, polygon_clip make_grid). 1D needs none.
// Particles of a periodic box sorted along a Morton curve: an implicit octree whose nodes each own
// a contiguous range of slots. Buffers persist across builds; a build allocates nothing.
namespace neighbor_search {

constexpr int fine_levels = 10;
constexpr int max_levels = 10;  // base levels: Morton cell numbers fit in an int

// Cell list in D = 2 or 3 dimensions (a quadtree in 2D); 1D needs none. The box is cut into tiles (near-cubes of
// about its shortest side, one for a cube), each an octree root. Slot q holds particle order(q) at positions(q).
// Base cell c owns slots [cell_offsets(c), cell_offsets(c + 1)); within it slots are sorted by the fine Morton key.
// level(q) is the grid level (2^level cells per tile side) whose cells hold about one particle near q.
// Used by: configure, build, Mesh::index; read through shell_voronoi::make_grid and polygon_clip::make_grid.
template <class ExecSpace, int D = 3>
struct CellList {
    static_assert(D == 2 || D == 3, "neighbor_search::CellList: 2 or 3 dimensions");
    using memory_space = typename ExecSpace::memory_space;
    static constexpr int dimensions = D;

    int n = 0;
    int levels = 2;  // the base grid has 2^levels cells per tile side
    exact_predicates::BoxN<D> box;
    int tiles[D];

    Kokkos::View<int*, memory_space> order, cell_offsets;
    Kokkos::View<double* [D], memory_space> positions;
    Kokkos::View<std::uint32_t*, memory_space> fine;
    Kokkos::View<std::uint8_t*, memory_space> level;
    Kokkos::View<char*, memory_space> sort_scratch;  // radix sort temporary storage (CUDA)
    // The sort's buffers (24 bytes per particle) when positions are smaller (2D).
    Kokkos::View<double*, memory_space> sort_storage;

    CellList() {
        for (int a = 0; a < D; ++a) tiles[a] = 1;
    }
    int tile_count() const {
        int t = 1;
        for (int a = 0; a < D; ++a) t *= tiles[a];
        return t;
    }
    int cell_count() const { return tile_count() << (D * levels); }
};

// Morton interleaving: bit k of x goes to bit 3k + 2, of y to 3k + 1, of z to 3k.
KOKKOS_INLINE_FUNCTION std::uint64_t spread3(std::uint64_t v) {
    v &= 0x1fffff;
    v = (v | v << 32) & 0x1f00000000ffffull;
    v = (v | v << 16) & 0x1f0000ff0000ffull;
    v = (v | v << 8) & 0x100f00f00f00f00full;
    v = (v | v << 4) & 0x10c30c30c30c30c3ull;
    v = (v | v << 2) & 0x1249249249249249ull;
    return v;
}
// Used by: build (the keys), shell_voronoi::Grid cell lookups.
KOKKOS_INLINE_FUNCTION std::uint64_t morton(std::uint64_t x, std::uint64_t y, std::uint64_t z) {
    return spread3(x) << 2 | spread3(y) << 1 | spread3(z);
}
// The same for coordinates of at most 10 bits, in 32-bit arithmetic.
KOKKOS_INLINE_FUNCTION std::uint32_t spread3_short(std::uint32_t v) {
    v &= 0x3ff;
    v = (v | v << 16) & 0x30000ffu;
    v = (v | v << 8) & 0x300f00fu;
    v = (v | v << 4) & 0x30c30c3u;
    v = (v | v << 2) & 0x9249249u;
    return v;
}
KOKKOS_INLINE_FUNCTION std::uint32_t morton_short(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    return spread3_short(x) << 2 | spread3_short(y) << 1 | spread3_short(z);
}
// In 2D: bit k of x goes to bit 2k + 1, of y to 2k.
KOKKOS_INLINE_FUNCTION std::uint64_t spread2(std::uint64_t v) {
    v &= 0xffffffffull;
    v = (v | v << 16) & 0x0000ffff0000ffffull;
    v = (v | v << 8) & 0x00ff00ff00ff00ffull;
    v = (v | v << 4) & 0x0f0f0f0f0f0f0f0full;
    v = (v | v << 2) & 0x3333333333333333ull;
    v = (v | v << 1) & 0x5555555555555555ull;
    return v;
}
// Used by: build in 2D.
KOKKOS_INLINE_FUNCTION std::uint64_t morton(std::uint64_t x, std::uint64_t y) { return spread2(x) << 1 | spread2(y); }

// Integer coordinate of x on a grid of `side` cells over [0, box) (one axis); every user computes it this way.
// Used by: build, shell_voronoi and polygon_clip home-cell lookups.
KOKKOS_INLINE_FUNCTION int grid_coordinate(double x, double box, int side) {
    return Kokkos::clamp(static_cast<int>(x / box * side), 0, side - 1);
}

// First slot in [begin, end) whose fine key is not below `key`.
// Used by: build (per-slot level), shell_voronoi::Grid and polygon_clip::Grid cell ranges.
template <class Fine>
KOKKOS_INLINE_FUNCTION int lower_bound(const Fine& fine, int begin, int end, std::uint32_t key) {
    while (begin < end) {
        const int mid = begin + (end - begin) / 2;
        if (fine(mid) < key)
            begin = mid + 1;
        else
            end = mid;
    }
    return begin;
}

namespace detail {

// Used by: configure.
template <class View, class... Extents>
void ensure(View& v, const char* label, Extents... extents) {
    const std::size_t want[] = {static_cast<std::size_t>(extents)...};
    bool same = v.is_allocated();
    for (std::size_t r = 0; r < sizeof...(Extents); ++r) same = same && v.extent(r) == want[r];
    if (!same) v = View(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)), extents...);
}

// The sort works in the storage of `positions` (24 bytes per particle): two buffers of 64-bit keys, two of ids.
// Used by: sort_buffers, build.
struct SortBuffers {
    std::uint64_t* keys[2];
    int* ids[2];
};
inline SortBuffers sort_buffers(double* storage, int n) {
    auto* keys = reinterpret_cast<std::uint64_t*>(storage);
    auto* ids = reinterpret_cast<int*>(keys + 2 * static_cast<std::size_t>(n));
    return {{keys, keys + n}, {ids, ids + n}};
}

}  // namespace detail

// Bits of the numbers 0 .. count - 1.
inline int bits(long long count) {
    int b = 0;
    while ((1LL << b) < count) ++b;
    return b;
}

// Bits of a particle's key: its tile, then its Morton key inside the tile.
// Used by: configure, build (sort key width).
template <class ExecSpace, int D>
int key_bits(const CellList<ExecSpace, D>& s) {
    return bits(s.tile_count()) + D * (s.levels + fine_levels);
}

// Size the tiles, the base grid (2^levels cells per tile side, the power of two nearest to tile / cell_width)
// and the buffers; call again when n, the box or the width changes. Levels are capped so keys fit 64 bits.
// Used by: Mesh::index when the particle count changes.
template <class ExecSpace, int D>
void configure(CellList<ExecSpace, D>& s, int n, const exact_predicates::BoxN<D>& box, double cell_width) {
    s.n = n;
    s.box = box;
    const double shortest = box.shortest();
    long long tiles = 1;
    int most = max_levels;
    for (int a = 0; a < D; ++a) {
        s.tiles[a] = static_cast<int>(std::max(1LL, std::llround(box.side[a] / shortest)));
        tiles *= s.tiles[a];
        most = std::min(most, 30 - bits(s.tiles[a]) - fine_levels);
    }
    most = std::min({most, (64 - bits(tiles)) / D - fine_levels, (30 - bits(tiles)) / D});
    if (most < 2)
        throw std::invalid_argument("neighbor_search::configure: box too elongated (" + std::to_string(tiles) +
                                    " tiles of its shortest side)");
    s.levels = std::clamp(static_cast<int>(std::lround(std::log2(shortest / cell_width))), 2, most);
    detail::ensure(s.order, "particle_order", n);
    detail::ensure(s.cell_offsets, "cell_offsets", s.cell_count() + 1);
    detail::ensure(s.positions, "slot_positions", n);
    detail::ensure(s.fine, "slot_fine_keys", n);
    detail::ensure(s.level, "slot_levels", n);
    if constexpr (D < 3) detail::ensure(s.sort_storage, "cell_list_sort_storage", 3 * n);
#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
        std::size_t bytes = 0;
        const auto b = detail::sort_buffers(nullptr, n);
        cub::DoubleBuffer<std::uint64_t> keys(b.keys[0], b.keys[1]);
        cub::DoubleBuffer<int> ids(b.ids[0], b.ids[1]);
        cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys, ids, n, 0, key_bits(s));
        detail::ensure(s.sort_scratch, "cell_list_sort_scratch", bytes);
    }
#endif
}

// Sort the particles by Morton key, then fill the slots, the base-cell
// offsets and the per-slot levels.
// Used by: Mesh::index, every update.
template <class ExecSpace, int D, class InPositions>
void build(const ExecSpace& exec, const CellList<ExecSpace, D>& s, const InPositions& in_positions) {
    const int n = s.n, cells = s.cell_count(), base = s.levels, depth = s.levels + fine_levels;
    const exact_predicates::BoxN<D> box = s.box;
    const int tx = s.tiles[0], ty = s.tiles[1], tz = s.tiles[D - 1];
    const auto order = s.order;
    const auto cell_offsets = s.cell_offsets;
    const auto positions = s.positions;
    const auto fine = s.fine;
    const auto level = s.level;
    const detail::SortBuffers b = detail::sort_buffers(D == 3 ? positions.data() : s.sort_storage.data(), n);
    using Range = Kokkos::RangePolicy<ExecSpace>;

    Kokkos::parallel_for("cell_list::keys", Range(exec, 0, n), KOKKOS_LAMBDA(int i) {
        const int tiles[3] = {tx, ty, tz};
        int c[D], tile[D];
        for (int a = 0; a < D; ++a) {
            const int g = grid_coordinate(in_positions(i, a), box.side[a], tiles[a] << depth);
            tile[a] = g >> depth;
            c[a] = g & ((1 << depth) - 1);
        }
        std::uint64_t key;
        if constexpr (D == 3) {
            const std::uint64_t t = (static_cast<std::uint64_t>(tile[0]) * tiles[1] + tile[1]) * tiles[2] + tile[2];
            key = t << (3 * depth) | morton(c[0], c[1], c[2]);
        } else {
            const std::uint64_t t = static_cast<std::uint64_t>(tile[0]) * tiles[1] + tile[1];
            key = t << (2 * depth) | morton(c[0], c[1]);
        }
        b.keys[0][i] = key;
        b.ids[0][i] = i;
    });
    // Stable by key, so equal keys keep the particle-ID order.
    int sorted = 0;  // which buffers hold the result
#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
        cub::DoubleBuffer<std::uint64_t> keys(b.keys[0], b.keys[1]);
        cub::DoubleBuffer<int> ids(b.ids[0], b.ids[1]);
        std::size_t bytes = s.sort_scratch.extent(0);
        cub::DeviceRadixSort::SortPairs(s.sort_scratch.data(), bytes, keys, ids, n, 0, key_bits(s), exec.cuda_stream());
        sorted = keys.selector;
    } else
#endif
    {
        if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename ExecSpace::memory_space>::accessible) {
            exec.fence();
            std::sort(b.ids[0], b.ids[0] + n, [&](int i, int j) {
                return b.keys[0][i] != b.keys[0][j] ? b.keys[0][i] < b.keys[0][j] : i < j;
            });
            for (int q = 0; q < n; ++q) b.keys[1][q] = b.keys[0][b.ids[0][q]];
            std::copy(b.ids[0], b.ids[0] + n, b.ids[1]);
            sorted = 1;
        } else {
            throw std::runtime_error("neighbor_search::build: no sort for this backend");
        }
    }
    const std::uint64_t* keys = b.keys[sorted];
    const int* ids = b.ids[sorted];
    constexpr std::uint64_t fine_mask = (std::uint64_t(1) << (D * fine_levels)) - 1;

    // Slot q starts every base cell after the one before it, up to its own.
    Kokkos::parallel_for("cell_list::slots", Range(exec, 0, n + 1), KOKKOS_LAMBDA(int q) {
        const int previous = q == 0 ? -1 : static_cast<int>(keys[q - 1] >> (D * fine_levels));
        const int cell = q == n ? cells : static_cast<int>(keys[q] >> (D * fine_levels));
        for (int c = previous + 1; c <= cell; ++c) cell_offsets(c) = q;
        if (q < n) {
            order(q) = ids[q];
            fine(q) = static_cast<std::uint32_t>(keys[q] & fine_mask);
        }
    });
    // Per-slot level: the deepest node around q holding at least `enough` particles, refined by the (2^D)-th root of
    // its count; coarser nodes are blocks of base cells, finer ones come from the fine keys.
    Kokkos::parallel_for("cell_list::levels", Range(exec, 0, n), KOKKOS_LAMBDA(int q) {
        constexpr int enough = 64;
        const int cell = static_cast<int>(keys[q] >> (D * fine_levels));
        const int begin = cell_offsets(cell), end = cell_offsets(cell + 1);
        int node = base, count = end - begin;
        if (count >= enough) {
            while (node < depth) {
                const int drop = D * (depth - node - 1);
                const std::uint32_t prefix = fine(q) >> drop;
                const int lo = lower_bound(fine, begin, q, prefix << drop);
                const int hi = lower_bound(fine, q, end, (prefix + 1) << drop);
                if (hi - lo < enough) break;
                ++node;
                count = hi - lo;
            }
        } else {
            while (node > 0 && count < enough) {
                --node;
                const int span = D * (base - node);
                const int first = (cell >> span) << span;
                count = cell_offsets(first + (1 << span)) - cell_offsets(first);
            }
        }
        // One particle per cell: count / 2^(D k) ~ 1, k rounded.
        const int finer = static_cast<int>(Kokkos::log2(static_cast<double>(Kokkos::max(count, 1))) / double(D) + 0.5);
        level(q) = static_cast<std::uint8_t>(Kokkos::clamp(node + finer, 2, depth));
    });
    Kokkos::parallel_for("cell_list::positions", Range(exec, 0, n), KOKKOS_LAMBDA(int q) {
        for (int a = 0; a < D; ++a) positions(q, a) = in_positions(order(q), a);
    });
}

}  // namespace neighbor_search
