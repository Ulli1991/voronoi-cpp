#pragma once

#include "shell_grid.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

// Clipping, the k nearest particles (3D): a pass over the cell list (a linear octree: Morton order, a level per
// particle) that lists each cell's k nearest particles, nearest first, for the warp kernel's listed clipping
// (shell_voronoi_warp.hpp). Used by: construct_warp when a list is asked for (Listing).
namespace shell_voronoi {

// An empty list entry; a list whose search could not be completed holds only these.
constexpr unsigned no_entry = ~0u;
// Slots go in the bits above the packed image (6 bits).
constexpr int largest_listed_n = 1 << 26;

// Lists of k entries per listed slot (in the order of the slots listed), slot << 6 | packed image, by increasing distance (ties: walk order). Every
// particle not listed lies at least as far as the k-th. reads: particles whose distance the search computed (or empty).
template <class MemorySpace>
struct Neighbours {
    Kokkos::View<unsigned*, MemorySpace> list;
    Kokkos::View<int*, MemorySpace> reads;
    int k = 0;
};

namespace detail {

// The search of one slot: keys are squared distances in float (in units of `unit2`), kept sorted in registers by an
// unrolled insertion. Returns the particles read; an incomplete search leaves no entries.
template <int K, bool cubic, class MemorySpace>
KOKKOS_INLINE_FUNCTION int nearest(const Grid<MemorySpace>& grid, const Stencil<MemorySpace>& stencil, int t,
                                   double unit2, float (&key)[K], unsigned (&entry)[K]) {
    const auto& positions = grid.positions;
    const double* side = grid.box.side;
    const Vec3 p{positions(t, 0), positions(t, 1), positions(t, 2)};
    int reads = 0;
    int level = grid.level(t), home[3], finer = -1;
    home_cell<cubic>(p, grid, level, home);
    // The k-th key, widened so that its double distance is not understated.
    auto kth2 = [&] { return key[K - 1] >= 3.0e38f ? 1e300 : double(key[K - 1]) * (1 + 0x1p-20) / unit2; };
    while (true) {
        const double width = grid.narrowest / (1 << level);
        const int reach = level_reach(level, cubic ? 1 : grid.fewest), entries = stencil.ends(reach);
        bool done = false;
        for (int e = 0; e < entries; ++e) {
            const double limit2 = kth2();
            if (double(stencil.bound2(e)) * width * width >= limit2) {
                done = true;
                break;
            }
            int begin = 0, end = 0, packed = 0;
            if (!stencil_cell<true, cubic>(grid, level, home, finer, stencil.offsets(e), p, limit2, begin, end, packed))
                continue;
            const double shift[3] = {image(packed, 0) * side[0], image(packed, 1) * side[cubic ? 0 : 1],
                                     image(packed, 2) * side[cubic ? 0 : 2]};
            for (int q = begin; q < end; ++q) {
                if (q == t) continue;
                ++reads;
                const double dx = positions(q, 0) + shift[0] - p.x;
                const double dy = positions(q, 1) + shift[1] - p.y;
                const double dz = positions(q, 2) + shift[2] - p.z;
                const float d = static_cast<float>((dx * dx + dy * dy + dz * dz) * unit2);
                if (!(d < key[K - 1])) continue;
                const unsigned u = static_cast<unsigned>(q) << 6 | static_cast<unsigned>(packed);
#pragma unroll
                for (int j = K - 1; j > 0; --j) {
                    const bool up = key[j - 1] > d, here = !up && key[j] > d;
                    entry[j] = up ? entry[j - 1] : here ? u : entry[j];
                    key[j] = up ? key[j - 1] : here ? d : key[j];
                }
                if (key[0] > d) {
                    entry[0] = u;
                    key[0] = d;
                }
            }
        }
        // Cells outside the stencil are at least `reach` widths away.
        if (done || reach * width * reach * width >= kth2()) return reads;
        if (level == 2) break;
        finer = (home[0] & 1) | (home[1] & 1) << 1 | (home[2] & 1) << 2;
        for (int a = 0; a < 3; ++a) home[a] >>= 1;
        --level;
    }
#pragma unroll
    for (int i = 0; i < K; ++i) entry[i] = no_entry;  // not completed: the walk builds the cell
    return reads;
}

// The lists of the slots `todo` (all if empty), one thread a slot; with `level` >= 0 only slots at that grid level
// (the even-density cells), others get an empty list.
template <int K, bool cubic, class ExecSpace>
void list_nearest(const ExecSpace& exec, const Grid<typename ExecSpace::memory_space>& grid,
                  const Stencil<typename ExecSpace::memory_space>& stencil,
                  const Neighbours<typename ExecSpace::memory_space>& out,
                  const Kokkos::View<const int*, typename ExecSpace::memory_space>& todo, int only_level) {
    const int count = todo.extent(0) > 0 ? static_cast<int>(todo.extent(0)) : grid.n;
    const double unit = std::ldexp(1.0, -std::ilogb(grid.narrowest)), unit2 = unit * unit;
    const bool with_reads = out.reads.extent(0) > 0;
    Kokkos::parallel_for(
        "shell_voronoi::list_nearest", Kokkos::RangePolicy<ExecSpace>(exec, 0, count), KOKKOS_LAMBDA(int i) {
            const int t = todo.extent(0) > 0 ? todo(i) : i;
            float key[K];
            unsigned entry[K];
#pragma unroll
            for (int j = 0; j < K; ++j) {
                key[j] = 3.0e38f;
                entry[j] = no_entry;
            }
            int reads = 0;
            if (only_level < 0 || grid.level(t) == only_level)
                reads = nearest<K, cubic>(grid, stencil, t, unit2, key, entry);
            const long long base = static_cast<long long>(i) * K;
#pragma unroll
            for (int j = 0; j < K; ++j) out.list(base + j) = entry[j];
            if (with_reads) out.reads(i) = reads;
        });
}

}  // namespace detail

// The k nearest particles of the slots `todo` (all if empty), for k = 16, 24, 32, 48 or 64.
template <class ExecSpace>
void list_nearest(const ExecSpace& exec, const Grid<typename ExecSpace::memory_space>& grid,
                  const Stencil<typename ExecSpace::memory_space>& stencil,
                  Neighbours<typename ExecSpace::memory_space>& out,
                  const Kokkos::View<const int*, typename ExecSpace::memory_space>& todo = {}, int only_level = -1) {
    if (grid.n >= largest_listed_n) throw std::invalid_argument("shell_voronoi::list_nearest: too many particles");
    const int count = todo.extent(0) > 0 ? static_cast<int>(todo.extent(0)) : grid.n;
    const std::size_t size = static_cast<std::size_t>(count) * out.k;
    if (out.list.extent(0) < size)
        out.list = Kokkos::View<unsigned*, typename ExecSpace::memory_space>(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("shell_knn_list")), size);
    auto run = [&](auto k, auto cubic) {
        detail::list_nearest<decltype(k)::value, decltype(cubic)::value>(exec, grid, stencil, out, todo, only_level);
    };
    auto pick = [&](auto cubic) {
        switch (out.k) {
            case 16: return run(std::integral_constant<int, 16>{}, cubic);
            case 24: return run(std::integral_constant<int, 24>{}, cubic);
            case 32: return run(std::integral_constant<int, 32>{}, cubic);
            case 48: return run(std::integral_constant<int, 48>{}, cubic);
            case 64: return run(std::integral_constant<int, 64>{}, cubic);
            default: throw std::invalid_argument("shell_voronoi::list_nearest: k must be 16, 24, 32, 48 or 64");
        }
    };
    if (grid.cube())
        pick(std::true_type{});
    else
        pick(std::false_type{});
}

}  // namespace shell_voronoi
