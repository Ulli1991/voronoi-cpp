#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Particles of the periodic cube [0, box)^3 sorted into a grid of cubic cells.
//
// A counting sort puts the particles into cell order: slot q holds particle
// order(q) at positions(q), and grid cell c owns the slots
// [cell_offsets(c), cell_offsets(c + 1)). Within a cell the slots follow the
// particle IDs, so the order is deterministic although the scatter uses
// atomics. Buffers persist across builds; a build allocates nothing.
namespace neighbor_search {

template <class ExecSpace>
struct CellList {
    using memory_space = typename ExecSpace::memory_space;

    int n = 0;
    int cells_per_side = 1;
    double box = 1.0;
    double cell_width = 1.0;
    // Cells stored in Morton order instead of Cartesian order. Set before
    // configure; cell_rank then maps a Cartesian cell key to its place.
    bool morton = false;

    Kokkos::View<int*, memory_space> order, cell_offsets;
    Kokkos::View<double* [3], memory_space> positions;
    Kokkos::View<int*, memory_space> cell_rank;  // empty unless morton
    Kokkos::View<int*, memory_space> keys, cell_counts;  // build scratch

    int cell_count() const { return cells_per_side * cells_per_side * cells_per_side; }
};

namespace detail {

// 21 coordinate bits per axis fit in a 64-bit Morton key.
inline std::uint64_t morton_key(unsigned x, unsigned y, unsigned z) {
    std::uint64_t result = 0;
    for (unsigned bit = 0; bit < 21; ++bit) {
        result |= std::uint64_t((x >> bit) & 1) << (3 * bit + 2);
        result |= std::uint64_t((y >> bit) & 1) << (3 * bit + 1);
        result |= std::uint64_t((z >> bit) & 1) << (3 * bit);
    }
    return result;
}

template <class View, class... Extents>
void ensure(View& v, const char* label, Extents... extents) {
    const std::size_t want[] = {static_cast<std::size_t>(extents)...};
    bool same = v.is_allocated();
    for (std::size_t r = 0; r < sizeof...(Extents); ++r) same = same && v.extent(r) == want[r];
    if (!same) v = View(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)), extents...);
}

}  // namespace detail

// Size the grid (cells at most `cell_width` wide) and the buffers. Call again
// whenever n, the box or the width changes; unchanged buffers are kept.
template <class ExecSpace>
void configure(CellList<ExecSpace>& s, int n, double box, double cell_width) {
    s.n = n;
    s.box = box;
    s.cells_per_side = std::max(1, static_cast<int>(std::ceil(box / cell_width)));
    s.cell_width = box / s.cells_per_side;
    const int side = s.cells_per_side, cells = s.cell_count();
    if (!s.morton) {
        s.cell_rank = {};
    } else if (s.cell_rank.extent(0) != static_cast<std::size_t>(cells)) {
        if (side > (1 << 21)) throw std::invalid_argument("Morton cell coordinates exceed 21 bits");
        std::vector<std::pair<std::uint64_t, int>> cells_by_key;
        cells_by_key.reserve(cells);
        for (int x = 0; x < side; ++x)
            for (int y = 0; y < side; ++y)
                for (int z = 0; z < side; ++z)
                    cells_by_key.emplace_back(detail::morton_key(x, y, z), (x * side + y) * side + z);
        std::sort(cells_by_key.begin(), cells_by_key.end());
        detail::ensure(s.cell_rank, "morton_cell_rank", cells);
        auto rank = Kokkos::create_mirror_view(s.cell_rank);
        for (int i = 0; i < cells; ++i) rank(cells_by_key[i].second) = i;
        Kokkos::deep_copy(s.cell_rank, rank);
    }
    const bool fresh_counts = s.cell_counts.extent(0) != static_cast<std::size_t>(cells);
    detail::ensure(s.keys, "cell_keys", n);
    detail::ensure(s.order, "particle_order", n);
    detail::ensure(s.cell_counts, "cell_counts", cells);
    detail::ensure(s.cell_offsets, "cell_offsets", cells + 1);
    detail::ensure(s.positions, "slot_positions", n);
    // Builds leave cell_counts at zero, so it is cleared only on allocation.
    if (fresh_counts) Kokkos::deep_copy(s.cell_counts, 0);
}

// Counting sort of the particles into cells, then a copy into slot order.
template <class ExecSpace, class InPositions>
void build(const ExecSpace& exec, const CellList<ExecSpace>& s, const InPositions& in_positions) {
    const int n = s.n, cells = s.cell_count(), side = s.cells_per_side;
    const double width = s.cell_width;
    const auto keys = s.keys;
    const auto cell_rank = s.cell_rank;
    const bool morton = s.morton;
    const auto order = s.order;
    const auto cell_counts = s.cell_counts;
    const auto cell_offsets = s.cell_offsets;
    const auto positions = s.positions;
    using Range = Kokkos::RangePolicy<ExecSpace>;

    Kokkos::parallel_for("cell_list::count", Range(exec, 0, n), KOKKOS_LAMBDA(int i) {
        int key = 0;
        for (int a = 0; a < 3; ++a)
            key = key * side + Kokkos::clamp(static_cast<int>(in_positions(i, a) / width), 0, side - 1);
        if (morton) key = cell_rank(key);
        keys(i) = key;
        Kokkos::atomic_inc(&cell_counts(key));
    });
    Kokkos::parallel_scan(
        "cell_list::scan", Range(exec, 0, cells), KOKKOS_LAMBDA(int c, int& update, bool final) {
            const int count = cell_counts(c);
            if (final) {
                cell_offsets(c) = update;
                cell_counts(c) = 0;  // reused as the scatter cursor
            }
            update += count;
            if (final && c == cells - 1) cell_offsets(cells) = update;
        });
    Kokkos::parallel_for("cell_list::scatter", Range(exec, 0, n), KOKKOS_LAMBDA(int i) {
        const int key = keys(i);
        order(cell_offsets(key) + Kokkos::atomic_fetch_add(&cell_counts(key), 1)) = i;
    });
    Kokkos::parallel_for("cell_list::order", Range(exec, 0, cells), KOKKOS_LAMBDA(int c) {
        const int begin = cell_offsets(c), end = cell_offsets(c + 1);
        for (int q = begin + 1; q < end; ++q) {
            const int id = order(q);
            int k = q;
            while (k > begin && order(k - 1) > id) {
                order(k) = order(k - 1);
                --k;
            }
            order(k) = id;
        }
        for (int q = begin; q < end; ++q)
            for (int a = 0; a < 3; ++a) positions(q, a) = in_positions(order(q), a);
        cell_counts(c) = 0;
    });
}

}  // namespace neighbor_search
