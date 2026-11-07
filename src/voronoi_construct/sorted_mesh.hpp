#pragma once

#include "exact_predicates.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cub/device/device_radix_sort.cuh>
#endif

// The 1D path of voronoi::Mesh<1> (included by voronoi_mesh.hpp): snap -> sort -> geometry, no clipping or repair.
// Called by the hydro code (finite_volume_1d.hpp); feeds it faces, volumes and centroids.
// The 1D Voronoi mesh: generators sorted by snapped position (ties by particle id), faces at the midpoints,
// exact because distances are whole grid steps; periodic, or with walls via mirrored end generators.
namespace sorted_mesh {

namespace ep = exact_predicates;
using i64 = ep::i64;

using Key = std::uint64_t;

// The grid of a line of length `side`.
// Used by: Mesh<1>, snap, hydro/finite_volume_1d.hpp.
struct Grid {
#ifndef VORONOI_1D_BITS
#define VORONOI_1D_BITS 52
#endif
    static constexpr int bits = VORONOI_1D_BITS;
    static constexpr i64 period = i64(1) << bits;
    double side = 1, step = 1.0 / static_cast<double>(period);

    Grid() = default;
    explicit Grid(double length) : side(length), step(length / static_cast<double>(period)) {}
};

// The sorted line as kernels read it: slot s holds particle order(s) at
// grid step key(s).
// Used by: Sorter::sort (fills it), Mesh<1>::geometry, insert/remove.
template <class MemorySpace>
struct Line {
    Kokkos::View<int*, MemorySpace> order;
    Kokkos::View<Key*, MemorySpace> key;
    int n = 0;
    i64 period = Grid::period;  // grid steps across the line
    double step = 1.0 / static_cast<double>(Grid::period);
    bool periodic = true;

    KOKKOS_INLINE_FUNCTION double side() const { return static_cast<double>(period) * step; }
    // Slot s + k of a periodic line, or -1 past a wall.
    KOKKOS_INLINE_FUNCTION int slot(int s, int k) const {
        const int t = s + k;
        if (periodic) return t < 0 ? t + n : t >= n ? t - n : t;
        return t < 0 || t >= n ? -1 : t;
    }
    // Grid steps from slot s to slot s + 1, s in [-1, n): across the seam
    // of a periodic line, or to the mirror of the end generator behind a wall.
    KOKKOS_INLINE_FUNCTION i64 gap(int s) const {
        if (!periodic) {
            if (s < 0) return 2 * i64(key(0));
            if (s >= n - 1) return 2 * (period - i64(key(n - 1)));
            return i64(key(s + 1)) - i64(key(s));
        }
        const int a = s < 0 ? n - 1 : s, b = a + 1 == n ? 0 : a + 1;
        const i64 d = i64(key(b)) - i64(key(a));
        return d < 0 ? d + period : d;
    }
    // Volume and centroid (relative to the generator) of slot s.
    KOKKOS_INLINE_FUNCTION double volume(int s) const { return static_cast<double>(gap(s - 1) + gap(s)) * (0.5 * step); }
    KOKKOS_INLINE_FUNCTION double centroid(int s) const {
        return static_cast<double>(gap(s) - gap(s - 1)) * (0.25 * step);
    }
};

// (key, id) of a < (key, id) of b: the order of the line.
// Used by: Sorter::sort, Mesh<1>::still_sorted.
KOKKOS_INLINE_FUNCTION bool before(Key ka, int a, Key kb, int b) {
    return ka < kb || (ka == kb && a < b);
}

// The grid step of x on a line of `period` steps: wrapped (periodic) or
// clamped to [0, period) (walls).
// Step 1, snap. Used by: Mesh<1>::snap_all, still_sorted, finite_volume_1d.
KOKKOS_INLINE_FUNCTION Key snap(double x, const Grid& grid, bool periodic) {
    const double y = periodic ? x - grid.side * Kokkos::floor(x / grid.side) : x;
    const i64 k = static_cast<i64>(Kokkos::floor(y / grid.step));
    return static_cast<Key>(k < 0 ? 0 : k >= Grid::period ? (periodic ? 0 : Grid::period - 1) : k);
}

// Sorts the particles by (key, id): keys(i) of particle i in `keys`, the
// result in line.order and line.key. On CUDA a stable radix sort over the
// key bits (ids in increasing order keep ties by id); elsewhere std::sort.
// Used by: Mesh<1>::order.
template <class ExecSpace>
class Sorter {
public:
    using Memory = typename ExecSpace::memory_space;

    void sort(const ExecSpace& exec, const Kokkos::View<Key*, Memory>& keys, const Kokkos::View<int*, Memory>& ids,
              const Line<Memory>& line) {
        const int n = line.n;
        const int bits = [&] {
            int b = 0;
            while ((i64(1) << b) < line.period) ++b;
            return b;
        }();
        Kokkos::parallel_for(
            "sorted_mesh::ids", Kokkos::RangePolicy<ExecSpace>(exec, 0, n), KOKKOS_LAMBDA(int i) { ids(i) = i; });
#if defined(KOKKOS_ENABLE_CUDA)
        if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
            std::size_t bytes = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys.data(), line.key.data(), ids.data(), line.order.data(), n, 0,
                                            bits, exec.cuda_stream());
            if (scratch_.extent(0) < bytes)
                scratch_ = Kokkos::View<char*, Memory>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("sorted_mesh_sort")), bytes);
            bytes = scratch_.extent(0);
            cub::DeviceRadixSort::SortPairs(scratch_.data(), bytes, keys.data(), line.key.data(), ids.data(), line.order.data(),
                                            n, 0, bits, exec.cuda_stream());
            return;
        }
#endif
        (void)bits;
        if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, Memory>::accessible) {
            exec.fence();
            Key* k = keys.data();
            int* order = line.order.data();
            std::copy(ids.data(), ids.data() + n, order);
            std::sort(order, order + n, [&](int a, int b) { return before(k[a], a, k[b], b); });
            for (int s = 0; s < n; ++s) line.key(s) = k[order[s]];
        } else {
            throw std::runtime_error("sorted_mesh::Sorter: no sort for this backend");
        }
    }
    std::size_t bytes() const { return scratch_.extent(0); }

private:
    Kokkos::View<char*, Memory> scratch_;
};

}  // namespace sorted_mesh

namespace voronoi {

// The ends of a 1D line: periodic, or walls at 0 and L.
enum class Ends { periodic, walls };

// What Mesh<1>::insert() or remove() did, per request r: id (-1: deferred), the two neighbours (-1, -2: walls), the
// lengths they give to or take from the cell, and their old volumes. Ids of before the change; the views are the mesh's
// and hold until the next change. Using it to move conserved quantities: docs/design.md.
// Used by: Mesh<1>::insert, remove.
template <class Memory>
struct Changes {
    int requested = 0, done = 0, first = 0, moves = 0;
    Kokkos::View<int*, Memory> id;
    Kokkos::View<int* [2], Memory> neighbour, moved;
    Kokkos::View<double* [2], Memory> overlap, volume;
};

// The 1D mesh: the sorted order of the generators, with the contract of Mesh<3> at D = 1 (faces once, owner the lower
// id, area +-1 owner to neighbour, centroids relative to the owner). Wall faces have neighbour -1 (at 0) and -2 (at
// L). Face s lies between slots s and s + 1; cell i's faces are 2i, 2i + 1 in increasing id of the other particle.
// update(x) snaps, rechecks the order and sorts again only if particles crossed; order(x) does only that.
// Used by: voronoi::Mesh<1> users (FiniteVolume 1D).
template <class ExecSpace>
class Mesh<1, ExecSpace, double, void> {
public:
    static constexpr int dimensions = 1;
    using Memory = typename ExecSpace::memory_space;
    template <class T>
    using View = Kokkos::View<T, Memory>;
    using Positions = View<double* [1]>;
    using Line = sorted_mesh::Line<Memory>;
    using Changes = voronoi::Changes<Memory>;

    Mesh(const ExecSpace& exec, const ep::BoxN<1>& box, Ends ends = Ends::periodic)
        : exec_(exec), box_(box), grid_(box.side[0]), periodic_(ends == Ends::periodic) {}
    Mesh(const ExecSpace& exec, double side, Ends ends = Ends::periodic) : Mesh(exec, ep::BoxN<1>(side), ends) {}

    // New positions x(i, 0) in [0, L) (walls: outside ones go to the nearest end step). Builds on the first call or when
    // the particle count changed. `active` and `timings` as Mesh<3> (unused).
    // Used by: callers, every step (snap -> order -> geometry).
    template <class X>
    Report update(const X& x, const View<const int*>& active = {}, Timings* timings = nullptr) {
        (void)active;
        (void)timings;
        Report report = order(x);
        geometry();
        report.faces = faces();
        return report;
    }
    // Used by: callers, to force a fresh sort.
    template <class X>
    Report rebuild(const X& x, Timings* timings = nullptr) {
        (void)timings;
        n_ = -1;
        return update(x);
    }
    // The order alone (face and cell arrays stale until update_cells()).
    // Steps snap + sort: only sorts again if particles crossed.
    // Used by: update, callers that compute keys themselves.
    template <class X>
    Report order(const X& x) {
        Report report;
        const int n = static_cast<int>(x.extent(0));
        if (n < 3) throw std::invalid_argument("voronoi::Mesh<1>: at least 3 particles");
        if (n != n_) {
            allocate(n);
            snap_all(x);
            sorter_.sort(exec_, keys_, ids_, line_);
            report.rebuilt = true;
        } else if (!still_sorted(x)) {
            snap_all(x);
            sorter_.sort(exec_, keys_, ids_, line_);
            report.rebuilt = true;
            ++sorts_;
        }
        current_ = false;
        return report;
    }
    // Used by: callers after order(), insert(), remove().
    void update_cells(Timings* timings = nullptr) {
        (void)timings;
        geometry();
    }
    void set_fallback(Fallback) {}

    // Scratch memory the caller may use between updates, grown to `bytes` (valid until the next update).
    using Scratch = Kokkos::View<char*, Memory, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Scratch scratch(std::size_t bytes) {
        if (scratch_.extent(0) < bytes) scratch_ = View<char*>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("mesh1_scratch")), bytes);
        return Scratch(scratch_.data(), bytes);
    }

    // Adds generators at points(r, 0), in the mesh's memory, to the line of the last update()/order(), without a sort. Views
    // may then be longer than particles(), but x given to update() or order() must have exactly particles() rows. A request
    // in a gap already taken or on a generator is deferred. Face and cell arrays stale until update_cells().
    // Used by: callers; geometry() via update_cells().
    template <class Points>
    const Changes& insert(const Points& points) {
        const int requested = static_cast<int>(points.extent(0));
        reserve(n_ + requested);
        changes_for(requested);
        const Line line = line_;
        const auto rank = rank_;
        const auto keys = request_keys_;
        const sorted_mesh::Grid grid = grid_;
        const bool periodic = periodic_;
        const int n = n_, start = first_slot();
        Kokkos::parallel_for(
            "mesh1::locate", Kokkos::RangePolicy<ExecSpace>(exec_, 0, requested), KOKKOS_LAMBDA(int r) {
                const sorted_mesh::Key key = sorted_mesh::snap(points(r, 0), grid, periodic);
                int lo = 0, hi = n;
                while (lo < hi) {
                    const int mid = lo + (hi - lo) / 2, slot = mid + start >= n ? mid + start - n : mid + start;
                    if (line.key(slot) <= key)
                        lo = mid + 1;
                    else
                        hi = mid;
                }
                keys(r) = key;
                const int before = lo == 0 ? 0 : (lo - 1 + start >= n ? lo - 1 + start - n : lo - 1 + start);
                rank(r) = lo == 0 || line.key(before) != key ? lo : -1;
            });
        // Which ones go ahead, decided on the host: a request per gap, the earliest in order of position, then of
        // request.
        const auto rank_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rank);
        const auto key_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, keys);
        std::vector<int> by_place(static_cast<std::size_t>(requested));
        std::iota(by_place.begin(), by_place.end(), 0);
        const auto gap_of = [&](int r) { return periodic && rank_host(r) == n ? 0 : rank_host(r); };
        std::stable_sort(by_place.begin(), by_place.end(), [&](int a, int b) { return gap_of(a) < gap_of(b); });
        std::vector<char> go(static_cast<std::size_t>(requested), 0);
        int last_gap = -1;
        for (const int r : by_place) {
            if (rank_host(r) < 0 || gap_of(r) == last_gap) continue;
            go[static_cast<std::size_t>(r)] = 1;
            last_gap = gap_of(r);
        }
        const auto id_host = Kokkos::create_mirror_view(changes_.id);
        int done = 0;
        for (int r = 0; r < requested; ++r) id_host(r) = go[static_cast<std::size_t>(r)] ? n + done++ : -1;
        Kokkos::deep_copy(exec_, changes_.id, id_host);
        changes_.requested = requested;
        changes_.done = done;
        changes_.first = n;
        changes_.moves = 0;
        // The accepted ones by position, for the pass over the line.
        by_place.erase(std::remove_if(by_place.begin(), by_place.end(), [&](int r) { return !go[static_cast<std::size_t>(r)]; }),
                       by_place.end());
        std::stable_sort(by_place.begin(), by_place.end(), [&](int a, int b) { return rank_host(a) < rank_host(b); });
        const auto ranks = sorted_ranks_;
        const auto place_host = Kokkos::create_mirror_view(ranks);
        const auto sorted_keys = sorted_keys_;
        const auto sorted_ids = sorted_ids_;
        const auto sorted_key_host = Kokkos::create_mirror_view(sorted_keys);
        const auto sorted_id_host = Kokkos::create_mirror_view(sorted_ids);
        for (int j = 0; j < done; ++j) {
            const int r = by_place[static_cast<std::size_t>(j)];
            place_host(j) = rank_host(r);
            sorted_key_host(j) = key_host(r);
            sorted_id_host(j) = id_host(r);
        }
        Kokkos::deep_copy(exec_, ranks, place_host);
        Kokkos::deep_copy(exec_, sorted_keys, sorted_key_host);
        Kokkos::deep_copy(exec_, sorted_ids, sorted_id_host);
        const auto id = changes_.id;
        const auto neighbour = changes_.neighbour;
        const auto overlap = changes_.overlap;
        const auto volume = changes_.volume;
        Kokkos::parallel_for(
            "mesh1::overlaps", Kokkos::RangePolicy<ExecSpace>(exec_, 0, requested), KOKKOS_LAMBDA(int r) {
                for (int side = 0; side < 2; ++side) {
                    neighbour(r, side) = side == 0 ? -1 : -2;
                    overlap(r, side) = 0;
                    volume(r, side) = 0;
                }
                if (id(r) < 0) return;
                const int at = rank(r);
                const sorted_mesh::i64 p = static_cast<sorted_mesh::i64>(keys(r));
                const bool wall_left = !periodic && at == 0, wall_right = !periodic && at == n;
                const int a = (at - 1 + start + n) % n, b = (at + start) % n;
                const sorted_mesh::i64 ka = static_cast<sorted_mesh::i64>(line.key(a)), kb = static_cast<sorted_mesh::i64>(line.key(b));
                const double half = 0.5 * line.step;
                if (!wall_left) {
                    neighbour(r, 0) = line.order(a);
                    volume(r, 0) = line.volume(a);
                }
                if (!wall_right) {
                    neighbour(r, 1) = line.order(b);
                    volume(r, 1) = line.volume(b);
                }
                if (wall_left) {
                    overlap(r, 1) = static_cast<double>(p + kb) * half;
                } else if (wall_right) {
                    overlap(r, 0) = static_cast<double>(2 * line.period - ka - p) * half;
                } else {
                    const sorted_mesh::i64 from_a = kb - p, from_b = p - ka;
                    overlap(r, 0) = static_cast<double>(from_a < 0 ? from_a + line.period : from_a) * half;
                    overlap(r, 1) = static_cast<double>(from_b < 0 ? from_b + line.period : from_b) * half;
                }
            });
        // The old slots move up by the new points before them; ids_ and keys_ are spare, and the line starts at its
        // smallest key from here on.
        const auto next_order = ids_;
        const auto next_key = keys_;
        const int accepted = done;
        Kokkos::parallel_for(
            "mesh1::insert", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n), KOKKOS_LAMBDA(int s) {
                const int u = s - start < 0 ? s - start + n : s - start;
                const int up = count_up_to(ranks, accepted, u) + u;
                next_order(up) = line.order(s);
                next_key(up) = line.key(s);
            });
        Kokkos::parallel_for(
            "mesh1::insert_new", Kokkos::RangePolicy<ExecSpace>(exec_, 0, accepted), KOKKOS_LAMBDA(int j) {
                next_order(ranks(j) + j) = sorted_ids(j);
                next_key(ranks(j) + j) = sorted_keys(j);
            });
        std::swap(line_.order, ids_);
        std::swap(line_.key, keys_);
        n_ += done;
        line_.n = n_;
        current_ = false;
        return changes_;
    }

    // Takes particles ids(r) off the line. A request is deferred if it would leave fewer than 3 particles, if a neighbour
    // on the line is going as well, or if the id is not on the line. Stale arrays, as insert().
    // Used by: callers; geometry() via update_cells().
    template <class Ids>
    const Changes& remove(const Ids& ids) {
        const int requested = static_cast<int>(ids.extent(0));
        changes_for(requested);
        const Line line = line_;
        const int n = n_;
        const bool periodic = periodic_;
        const auto slot_of = ids_;
        Kokkos::parallel_for(
            "mesh1::slot_of", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n), KOKKOS_LAMBDA(int s) { slot_of(line.order(s)) = s; });
        const auto rank = rank_;
        Kokkos::parallel_for(
            "mesh1::find", Kokkos::RangePolicy<ExecSpace>(exec_, 0, requested), KOKKOS_LAMBDA(int r) {
                const int i = ids(r);
                rank(r) = i >= 0 && i < n ? slot_of(i) : -1;
            });
        const auto slot_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rank);
        std::vector<int> by_slot(static_cast<std::size_t>(requested));
        std::iota(by_slot.begin(), by_slot.end(), 0);
        std::stable_sort(by_slot.begin(), by_slot.end(), [&](int a, int b) { return slot_host(a) < slot_host(b); });
        std::vector<char> go(static_cast<std::size_t>(requested), 0);
        int done = 0, last = -2, first_slot_gone = -1;
        for (const int r : by_slot) {
            const int s = slot_host(r);
            if (s < 0 || s == last || s == last + 1 || n - done <= 3) continue;
            if (periodic && s == n - 1 && first_slot_gone == 0) continue;
            go[static_cast<std::size_t>(r)] = 1;
            last = s;
            if (done++ == 0) first_slot_gone = s;
        }
        const auto id_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ids);
        const auto reported_host = Kokkos::create_mirror_view(changes_.id);
        by_slot.erase(std::remove_if(by_slot.begin(), by_slot.end(), [&](int r) { return !go[static_cast<std::size_t>(r)]; }),
                      by_slot.end());
        const auto gone = sorted_ranks_;
        const auto gone_host = Kokkos::create_mirror_view(gone);
        for (int r = 0; r < requested; ++r) reported_host(r) = go[static_cast<std::size_t>(r)] ? id_host(r) : -1;
        for (int j = 0; j < done; ++j) gone_host(j) = slot_host(by_slot[static_cast<std::size_t>(j)]);
        Kokkos::deep_copy(exec_, changes_.id, reported_host);
        Kokkos::deep_copy(exec_, gone, gone_host);
        // Holes below the new count take the last ids that stay, both in order.
        const int kept = n - done;
        std::vector<int> holes;
        std::vector<char> leaving(static_cast<std::size_t>(n - kept), 0);
        for (int j = 0; j < done; ++j) {
            const int i = id_host(by_slot[static_cast<std::size_t>(j)]);
            if (i < kept)
                holes.push_back(i);
            else
                leaving[static_cast<std::size_t>(i - kept)] = 1;
        }
        std::sort(holes.begin(), holes.end());
        const auto moved_host = Kokkos::create_mirror_view(changes_.moved);
        const auto new_id = sorted_ids_;
        const auto new_id_host = Kokkos::create_mirror_view(new_id);
        int moves = 0;
        for (int i = kept; i < n; ++i) {
            new_id_host(i - kept) = -1;
            if (leaving[static_cast<std::size_t>(i - kept)]) continue;
            const int hole = holes[static_cast<std::size_t>(moves)];
            moved_host(moves, 0) = i;
            moved_host(moves, 1) = hole;
            new_id_host(i - kept) = hole;
            ++moves;
        }
        Kokkos::deep_copy(exec_, changes_.moved, moved_host);
        Kokkos::deep_copy(exec_, new_id, new_id_host);
        changes_.requested = requested;
        changes_.done = done;
        changes_.first = 0;
        changes_.moves = moves;
        const auto id = changes_.id;
        const auto neighbour = changes_.neighbour;
        const auto overlap = changes_.overlap;
        const auto volume = changes_.volume;
        Kokkos::parallel_for(
            "mesh1::overlaps", Kokkos::RangePolicy<ExecSpace>(exec_, 0, requested), KOKKOS_LAMBDA(int r) {
                for (int side = 0; side < 2; ++side) {
                    neighbour(r, side) = side == 0 ? -1 : -2;
                    overlap(r, side) = 0;
                    volume(r, side) = 0;
                }
                if (id(r) < 0) return;
                const int s = rank(r), a = line.slot(s, -1), b = line.slot(s, 1);
                const double half = 0.5 * line.step;
                if (a >= 0) {
                    neighbour(r, 0) = line.order(a);
                    volume(r, 0) = line.volume(a);
                }
                if (b >= 0) {
                    neighbour(r, 1) = line.order(b);
                    volume(r, 1) = line.volume(b);
                }
                if (a < 0)
                    overlap(r, 1) = line.volume(s);
                else if (b < 0)
                    overlap(r, 0) = line.volume(s);
                else {
                    overlap(r, 0) = static_cast<double>(line.gap(s)) * half;
                    overlap(r, 1) = static_cast<double>(line.gap(s - 1)) * half;
                }
            });
        // Down by the removed slots before each one, ids renamed.
        const auto next_order = ids_;
        const auto next_key = keys_;
        const int count = done;
        Kokkos::parallel_for(
            "mesh1::remove", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n), KOKKOS_LAMBDA(int s) {
                const int below = count_up_to(gone, count, s);
                if (below > 0 && gone(below - 1) == s) return;
                const int i = line.order(s);
                next_order(s - below) = i < kept ? i : new_id(i - kept);
                next_key(s - below) = line.key(s);
            });
        std::swap(line_.order, ids_);
        std::swap(line_.key, keys_);
        n_ = kept;
        line_.n = n_;
        current_ = false;
        return changes_;
    }

    // Room for `count` particles without another allocation.
    void reserve(int count) {
        if (count <= capacity_) return;
        const int capacity = std::max(count, capacity_ + capacity_ / 4);
        const auto order = line_.order;
        const auto key = line_.key;
        const auto used = std::make_pair(std::size_t(0), static_cast<std::size_t>(n_));
        make_arrays(capacity);
        Kokkos::deep_copy(exec_, Kokkos::subview(line_.order, used), Kokkos::subview(order, used));
        Kokkos::deep_copy(exec_, Kokkos::subview(line_.key, used), Kokkos::subview(key, used));
        current_ = false;
    }

    int particles() const { return n_; }
    int capacity() const { return capacity_; }
    bool periodic() const { return periodic_; }
    const ep::BoxN<1>& sides() const { return box_; }
    double box() const { return box_.side[0]; }
    int faces() const { return periodic_ ? n_ : n_ + 1; }
    View<const int*> owner() const { return owner_; }
    View<const int*> neighbour() const { return neighbour_; }
    View<const double* [1]> area() const { return area_; }
    View<const double* [1]> face_centroid() const { return face_centroid_; }
    View<const double*> volume() const { return volume_; }
    View<const double* [1]> centroid() const { return centroid_; }
    View<const int*> cell_face_start() const { return cell_face_start_; }
    View<const int*> cell_face() const { return cell_face_; }
    // Snapped positions in id order (after update(); stale after order()).
    Positions positions() const { return positions_; }
    // The sorted line (current after update() and order()).
    const Line& line() const { return line_; }
    // For a caller computing the next keys itself in slot order: write spare_keys(), then swap_keys() (order() only if
    // the order broke).
    View<sorted_mesh::Key*> spare_keys() const { return keys_; }
    void swap_keys() {
        std::swap(line_.key, keys_);
        current_ = false;
    }
    // How often the order had to be sorted again since the first build.
    long long sorts() const { return sorts_; }

    // Internals, public only because CUDA device lambdas may not be defined in private member functions.
    void allocate(int n) {
        n_ = n;
        sorts_ = 0;
        line_.n = n;
        line_.period = sorted_mesh::Grid::period;
        line_.step = grid_.step;
        line_.periodic = periodic_;
        make_arrays(n);
    }

    // Every array for n particles; none keeps its contents.
    void make_arrays(int n) {
        capacity_ = n;
        auto make = [](auto& v, const char* label, auto... extents) {
            using V = std::decay_t<decltype(v)>;
            v = V(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)), extents...);
        };
        make(line_.order, "mesh1_order", n);
        make(line_.key, "mesh1_key", n);
        make(keys_, "mesh1_keys_by_id", n);
        make(ids_, "mesh1_ids", n);
        const int faces = periodic_ ? n : n + 1;
        make(owner_, "mesh1_owner", faces);
        make(neighbour_, "mesh1_neighbour", faces);
        make(area_, "mesh1_area", faces);
        make(face_centroid_, "mesh1_face_centroid", faces);
        make(volume_, "mesh1_volume", n);
        make(centroid_, "mesh1_centroid", n);
        make(positions_, "mesh1_positions", n);
        make(cell_face_, "mesh1_cell_face", 2 * n);
        make(cell_face_start_, "mesh1_cell_face_start", n + 1);
        const auto start = cell_face_start_;
        Kokkos::parallel_for(
            "mesh1::cell_face_start", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n + 1), KOKKOS_LAMBDA(int i) { start(i) = 2 * i; });
    }

    // Used by: order.
    template <class X>
    void snap_all(const X& x) {
        const auto keys = keys_;
        const sorted_mesh::Grid grid = grid_;
        const bool periodic = periodic_;
        Kokkos::parallel_for(
            "mesh1::snap", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_),
            KOKKOS_LAMBDA(int i) { keys(i) = sorted_mesh::snap(x(i, 0), grid, periodic); });
    }

    // Snaps x into the line's slots and counts the slots out of order: a
    // periodic line is still sorted (as a cycle) with exactly one descent,
    // one with walls with none.
    // Used by: order.
    template <class X>
    bool still_sorted(const X& x) {
        const Line line = line_;
        const sorted_mesh::Grid grid = grid_;
        const bool periodic = periodic_;
        int descents = 0;
        Kokkos::parallel_reduce(
            "mesh1::check_order", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_),
            KOKKOS_LAMBDA(int s, int& count) {
                const int i = line.order(s);
                const sorted_mesh::Key k = sorted_mesh::snap(x(i, 0), grid, periodic);
                line.key(s) = k;
                if (!periodic && s == line.n - 1) return;
                const int t = s + 1 == line.n ? 0 : s + 1, j = line.order(t);
                count += !sorted_mesh::before(k, i, sorted_mesh::snap(x(j, 0), grid, periodic), j);
            },
            descents);
        return descents == (periodic ? 1 : 0);
    }

    // The entries of `sorted` (of length count) that are at most `value`.
    KOKKOS_INLINE_FUNCTION static int count_up_to(const View<int*>& sorted, int count, int value) {
        int lo = 0, hi = count;
        while (lo < hi) {
            const int mid = lo + (hi - lo) / 2;
            if (sorted(mid) <= value)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }

    // The slot of the smallest key: a periodic line may have rotated.
    int first_slot() {
        if (!periodic_) return 0;
        const Line line = line_;
        int found = 0;
        Kokkos::parallel_reduce(
            "mesh1::first_slot", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_),
            KOKKOS_LAMBDA(int s, int& at) {
                const int t = s + 1 == line.n ? 0 : s + 1;
                if (line.key(t) < line.key(s)) at = t;
            },
            Kokkos::Max<int>(found));
        return std::max(found, 0);
    }

    // Room for a batch of `requested`, reused from batch to batch.
    void changes_for(int requested) {
        if (requested <= static_cast<int>(rank_.extent(0))) return;
        auto make = [](auto& v, const char* label, auto... extents) {
            using V = std::decay_t<decltype(v)>;
            v = V(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)), extents...);
        };
        make(changes_.id, "mesh1_change_id", requested);
        make(changes_.neighbour, "mesh1_change_neighbour", requested);
        make(changes_.moved, "mesh1_change_moved", requested);
        make(changes_.overlap, "mesh1_change_overlap", requested);
        make(changes_.volume, "mesh1_change_volume", requested);
        make(rank_, "mesh1_change_rank", requested);
        make(sorted_ranks_, "mesh1_change_sorted", requested);
        make(sorted_ids_, "mesh1_change_ids", requested);
        make(request_keys_, "mesh1_change_keys", requested);
        make(sorted_keys_, "mesh1_change_sorted_keys", requested);
    }

    // Faces, volumes, centroids and per-cell face lists from the line, one
    // thread per slot.
    // Used by: update, update_cells.
    void geometry() {
        if (current_) return;
        const Line line = line_;
        const auto owner = owner_, neighbour = neighbour_, cell_face = cell_face_;
        const auto area = area_, face_centroid = face_centroid_, centroid = centroid_, positions = positions_;
        const auto volume = volume_;
        const bool periodic = periodic_;
        const int n = n_;
        Kokkos::parallel_for(
            "mesh1::geometry", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n), KOKKOS_LAMBDA(int s) {
                const int i = line.order(s), r = line.slot(s, 1), l = line.slot(s, -1);
                const int right = r < 0 ? -2 : line.order(r), left = l < 0 ? -1 : line.order(l);
                const double half = static_cast<double>(line.gap(s)) * (0.5 * line.step);
                positions(i, 0) = static_cast<double>(line.key(s)) * line.step;
                volume(i) = line.volume(s);
                centroid(i, 0) = line.centroid(s);
                // The face to the right (slot s), and the wall faces.
                const int f_right = r < 0 ? n : s;
                const int f_left = l < 0 ? n - 1 : (periodic ? (s == 0 ? n - 1 : s - 1) : s - 1);
                if (r >= 0) {
                    owner(s) = i < right ? i : right;
                    neighbour(s) = i < right ? right : i;
                    area(s, 0) = i < right ? 1.0 : -1.0;
                    face_centroid(s, 0) = i < right ? half : -half;
                } else {
                    owner(n) = i;
                    neighbour(n) = -2;
                    area(n, 0) = 1.0;
                    face_centroid(n, 0) = half;
                }
                if (l < 0) {
                    owner(n - 1) = i;
                    neighbour(n - 1) = -1;
                    area(n - 1, 0) = -1.0;
                    face_centroid(n - 1, 0) = -static_cast<double>(line.gap(-1)) * (0.5 * line.step);
                }
                const bool left_first = left < right;
                cell_face(2 * i) = left_first ? f_left : f_right;
                cell_face(2 * i + 1) = left_first ? f_right : f_left;
            });
        current_ = true;
    }

private:
    ExecSpace exec_;
    ep::BoxN<1> box_;
    sorted_mesh::Grid grid_;
    bool periodic_;
    int n_ = 0, capacity_ = 0;
    long long sorts_ = 0;
    bool current_ = false;
    Line line_;
    Changes changes_;
    View<int*> rank_, sorted_ranks_, sorted_ids_;
    View<sorted_mesh::Key*> request_keys_, sorted_keys_;
    View<sorted_mesh::Key*> keys_;
    View<int*> ids_;
    sorted_mesh::Sorter<ExecSpace> sorter_;
    View<int*> owner_, neighbour_, cell_face_, cell_face_start_;
    View<double* [1]> area_, face_centroid_, centroid_;
    Positions positions_;
    View<double*> volume_;
    View<char*> scratch_;
};

}  // namespace voronoi
