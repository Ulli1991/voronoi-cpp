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

// Particles of a periodic box sorted along a Morton curve: an implicit
// octree in which every node, from a tile down to cells 2^-(levels +
// fine_levels) tiles wide, owns a contiguous range of slots.
//
// Tiles: the box is cut into tiles[0] x tiles[1] x tiles[2] near-cubes of
// about its shortest side (one tile, the box itself, for a cube), numbered
// in row-major order; each is the root of an octree, and a cell number is
// the tile's number above the Morton number inside it. So unequal sides
// need no padding to a cube (a 2 x 2 x 1/16 slab is 32 x 32 x 1 tiles).
//
// Slot q holds particle order(q) at positions(q). The base grid (2^levels
// cells per tile side, cells numbered as above) has a dense offset table:
// base cell c owns the slots [cell_offsets(c), cell_offsets(c + 1)), and so a
// block of 8^k base cells at a coarser level owns a contiguous range too.
// Below the base grid, the slots of a base cell are sorted by fine(q), the
// Morton key of the particle inside its base cell (fine_levels levels), so a
// finer node is found by binary search. Ties keep particle-ID order, which
// makes the slot order deterministic.
//
// level(q) is the grid level whose cells hold about one particle near slot q
// (2^level cells per side): the deepest node around q with at least 64
// particles gives the local density. Clipping walks the cells of that level
// around each particle, so clusters and voids each get a fitting cell size.
//
// Buffers persist across builds; a build allocates nothing.
namespace neighbor_search {

constexpr int fine_levels = 10;
constexpr int max_levels = 10;  // base levels: Morton cell numbers fit in an int

template <class ExecSpace>
struct CellList {
    using memory_space = typename ExecSpace::memory_space;

    int n = 0;
    int levels = 2;  // the base grid has 2^levels cells per tile side
    exact_predicates::Box box;
    int tiles[3] = {1, 1, 1};

    Kokkos::View<int*, memory_space> order, cell_offsets;
    Kokkos::View<double* [3], memory_space> positions;
    Kokkos::View<std::uint32_t*, memory_space> fine;
    Kokkos::View<std::uint8_t*, memory_space> level;
    Kokkos::View<char*, memory_space> sort_scratch;  // radix sort temporary storage (CUDA)

    int cell_count() const { return tiles[0] * tiles[1] * tiles[2] << (3 * levels); }
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

// Integer coordinate of x on a grid of `side` cells over [0, box) (one
// axis). Every user computes it this way, so a particle's node never
// depends on who asks.
KOKKOS_INLINE_FUNCTION int grid_coordinate(double x, double box, int side) {
    return Kokkos::clamp(static_cast<int>(x / box * side), 0, side - 1);
}

// First slot in [begin, end) whose fine key is not below `key`.
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

template <class View, class... Extents>
void ensure(View& v, const char* label, Extents... extents) {
    const std::size_t want[] = {static_cast<std::size_t>(extents)...};
    bool same = v.is_allocated();
    for (std::size_t r = 0; r < sizeof...(Extents); ++r) same = same && v.extent(r) == want[r];
    if (!same) v = View(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)), extents...);
}

// The sort works in the storage of `positions` (24 bytes per particle), which
// is only written after it: two buffers of 64-bit keys, two of particle IDs.
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
template <class ExecSpace>
int key_bits(const CellList<ExecSpace>& s) {
    return bits(static_cast<long long>(s.tiles[0]) * s.tiles[1] * s.tiles[2]) + 3 * (s.levels + fine_levels);
}

// Size the tiles, the base grid (2^levels cells per tile side, the power of
// two nearest to tile / cell_width) and the buffers. Call again whenever n,
// the box or the width changes; unchanged buffers are kept. Levels are
// capped so that keys fit 64 bits and cell numbers an int.
template <class ExecSpace>
void configure(CellList<ExecSpace>& s, int n, const exact_predicates::Box& box, double cell_width) {
    s.n = n;
    s.box = box;
    const double shortest = box.shortest();
    long long tiles = 1;
    int most = max_levels;
    for (int a = 0; a < 3; ++a) {
        s.tiles[a] = static_cast<int>(std::max(1LL, std::llround(box.side[a] / shortest)));
        tiles *= s.tiles[a];
        most = std::min(most, 30 - bits(s.tiles[a]) - fine_levels);  // grid coordinates fit an int
    }
    most = std::min({most, (64 - bits(tiles)) / 3 - fine_levels, (30 - bits(tiles)) / 3});
    if (most < 2)
        throw std::invalid_argument("neighbor_search::configure: box too elongated (" + std::to_string(tiles) +
                                    " tiles of its shortest side)");
    s.levels = std::clamp(static_cast<int>(std::lround(std::log2(shortest / cell_width))), 2, most);
    detail::ensure(s.order, "particle_order", n);
    detail::ensure(s.cell_offsets, "cell_offsets", s.cell_count() + 1);
    detail::ensure(s.positions, "slot_positions", n);
    detail::ensure(s.fine, "slot_fine_keys", n);
    detail::ensure(s.level, "slot_levels", n);
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
template <class ExecSpace, class InPositions>
void build(const ExecSpace& exec, const CellList<ExecSpace>& s, const InPositions& in_positions) {
    const int n = s.n, cells = s.cell_count(), base = s.levels, depth = s.levels + fine_levels;
    const exact_predicates::Box box = s.box;
    const int tx = s.tiles[0], ty = s.tiles[1], tz = s.tiles[2];
    const auto order = s.order;
    const auto cell_offsets = s.cell_offsets;
    const auto positions = s.positions;
    const auto fine = s.fine;
    const auto level = s.level;
    const detail::SortBuffers b = detail::sort_buffers(positions.data(), n);
    using Range = Kokkos::RangePolicy<ExecSpace>;

    Kokkos::parallel_for("cell_list::keys", Range(exec, 0, n), KOKKOS_LAMBDA(int i) {
        const int tiles[3] = {tx, ty, tz};
        int c[3], tile[3];
        for (int a = 0; a < 3; ++a) {
            const int g = grid_coordinate(in_positions(i, a), box.side[a], tiles[a] << depth);
            tile[a] = g >> depth;
            c[a] = g & ((1 << depth) - 1);
        }
        const std::uint64_t t = (static_cast<std::uint64_t>(tile[0]) * ty + tile[1]) * tz + tile[2];
        b.keys[0][i] = t << (3 * depth) | morton(c[0], c[1], c[2]);
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
    constexpr std::uint64_t fine_mask = (std::uint64_t(1) << (3 * fine_levels)) - 1;

    // Slot q starts every base cell after the one before it, up to its own.
    Kokkos::parallel_for("cell_list::slots", Range(exec, 0, n + 1), KOKKOS_LAMBDA(int q) {
        const int previous = q == 0 ? -1 : static_cast<int>(keys[q - 1] >> (3 * fine_levels));
        const int cell = q == n ? cells : static_cast<int>(keys[q] >> (3 * fine_levels));
        for (int c = previous + 1; c <= cell; ++c) cell_offsets(c) = q;
        if (q < n) {
            order(q) = ids[q];
            fine(q) = static_cast<std::uint32_t>(keys[q] & fine_mask);
        }
    });
    // Per-slot level: the deepest node around q holding at least `enough`
    // particles, refined by the eighth root of its count. Coarser nodes are
    // blocks of base cells; finer ones are found in the base cell's fine keys.
    Kokkos::parallel_for("cell_list::levels", Range(exec, 0, n), KOKKOS_LAMBDA(int q) {
        constexpr int enough = 64;
        const int cell = static_cast<int>(keys[q] >> (3 * fine_levels));
        const int begin = cell_offsets(cell), end = cell_offsets(cell + 1);
        int node = base, count = end - begin;
        if (count >= enough) {
            while (node < depth) {
                const int drop = 3 * (depth - node - 1);
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
                const int span = 3 * (base - node);
                const int first = (cell >> span) << span;
                count = cell_offsets(first + (1 << span)) - cell_offsets(first);
            }
        }
        // One particle per cell: count / 8^k ~ 1, k rounded.
        const int finer = static_cast<int>(Kokkos::log2(static_cast<double>(Kokkos::max(count, 1))) / 3.0 + 0.5);
        level(q) = static_cast<std::uint8_t>(Kokkos::clamp(node + finer, 2, depth));
    });
    Kokkos::parallel_for("cell_list::positions", Range(exec, 0, n), KOKKOS_LAMBDA(int q) {
        for (int a = 0; a < 3; ++a) positions(q, a) = in_positions(order(q), a);
    });
}

}  // namespace neighbor_search
