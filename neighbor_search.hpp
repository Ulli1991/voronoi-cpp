#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace neighbor_search {

// Gather-style radius search on a dense cell grid over [0,box)^3. Particles
// are copied into cell order; "slot" q holds particle order(q), and cell c
// owns slots [cell_offsets(c), cell_offsets(c+1)). Row t of `table` lists the
// slots of every other particle within radii(t) of slot t, `counts(t)` of
// them. All buffers persist across rebuilds and are reused without zeroing, so
// a rebuild allocates nothing and needs one host read: the overflow check.
//
// Periodic searches require every radius to be below half the box length.
template<class ExecSpace>
struct DenseCellSearch {
    using exec_space = ExecSpace;
    using memory_space = typename ExecSpace::memory_space;
    using IntView = Kokkos::View<int*, memory_space>;
    using Positions = Kokkos::View<double*[3], memory_space>;
    using ScalarView = Kokkos::View<double*, memory_space>;
    using Table = Kokkos::View<int**, memory_space>;
    using DistanceTable = Kokkos::View<double**, memory_space>;

    int n = 0, capacity = 0;
    double cell_width = 1.0;
    int nx = 1, ny = 1, nz = 1;
    double hx = 1.0, hy = 1.0, hz = 1.0;
    Kokkos::Array<double, 3> box{1.0, 1.0, 1.0};
    bool periodic = false;
    bool nearest_first = false;
    // Lanes cooperating on one particle's search; 1 = one thread per particle.
    int vector_length = 1;

    IntView keys, order, cell_counts, cell_offsets, counts;
    Positions positions;
    ScalarView radii;
    Table table;
    DistanceTable distance2;
    Kokkos::View<int, memory_space> max_count;

    int cell_count() const { return nx * ny * nz; }
};

namespace detail {

template<class View, class... Extents>
void ensure(View& v, const char* label, Extents... extents) {
    const std::size_t want[] = {static_cast<std::size_t>(extents)...};
    bool same = v.is_allocated();
    for (std::size_t r = 0; r < sizeof...(Extents); ++r) same = same && v.extent(r) == want[r];
    if (!same) v = View(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)), extents...);
}

}  // namespace detail

// With one thread per particle a small search cannot fill a GPU, and each
// thread walks its candidates serially; there, lanes of a warp share each
// particle's candidates. On an A100 that paid off up to about 4k particles
// (~65k lanes in total) and lost from 32k on, where the lanes' redundant cell
// walk costs more than the shared candidate tests save. Host backends use one
// lane.
template<class ExecSpace>
int default_vector_length(int n) {
    if (Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                   typename ExecSpace::memory_space>::accessible) return 1;
    if (n >= 16384) return 1;
    int lanes = 1;
    while (lanes < 32 && static_cast<long long>(n) * lanes * 2 <= 65536) lanes *= 2;
    return lanes;
}

// Size the grid and buffers. Cells are at most `cell_width` wide; widths near
// 0.75 of the typical radius were fastest on an A100. Call again whenever n,
// the box, the width, or the row capacity changes; unchanged buffers are kept.
template<class ExecSpace>
void configure(DenseCellSearch<ExecSpace>& s, int n, Kokkos::Array<double, 3> box,
               double cell_width, bool periodic, int capacity, bool nearest_first,
               int vector_length = 0) {
    s.n = n;
    s.cell_width = cell_width;
    s.box = box;
    s.periodic = periodic;
    s.capacity = capacity;
    s.nearest_first = nearest_first;
    s.vector_length = vector_length > 0 ? vector_length : default_vector_length<ExecSpace>(n);
    s.nx = std::max(1, static_cast<int>(std::ceil(box[0] / cell_width)));
    s.ny = std::max(1, static_cast<int>(std::ceil(box[1] / cell_width)));
    s.nz = std::max(1, static_cast<int>(std::ceil(box[2] / cell_width)));
    s.hx = box[0] / s.nx;
    s.hy = box[1] / s.ny;
    s.hz = box[2] / s.nz;
    const int cells = s.cell_count();
    const bool fresh_counts = !s.cell_counts.is_allocated() ||
                              s.cell_counts.extent(0) != static_cast<std::size_t>(cells);
    detail::ensure(s.keys, "cell_keys", n);
    detail::ensure(s.order, "particle_order", n);
    detail::ensure(s.cell_counts, "cell_counts", cells);
    detail::ensure(s.cell_offsets, "cell_offsets", cells + 1);
    detail::ensure(s.counts, "neighbor_counts", n);
    detail::ensure(s.positions, "slot_positions", n);
    detail::ensure(s.radii, "slot_radii", n);
    detail::ensure(s.table, "neighbor_table", n, capacity);
    if (nearest_first) detail::ensure(s.distance2, "neighbor_distance2", n, capacity);
    if (!s.max_count.is_allocated())
        s.max_count = Kokkos::View<int, typename ExecSpace::memory_space>("max_neighbor_count");
    // Builds leave cell_counts at zero, so it is cleared only on allocation.
    if (fresh_counts) Kokkos::deep_copy(s.cell_counts, 0);
}

// Counting sort of particles into cells, then a copy into slot order. The
// scatter uses atomics, so each cell's slots are re-sorted by particle id to
// keep the slot order deterministic.
template<class ExecSpace, class InPositions, class InRadii>
void build(const ExecSpace& exec, const DenseCellSearch<ExecSpace>& s,
           const InPositions& in_positions, const InRadii& in_radii) {
    const int n = s.n, cells = s.cell_count();
    const int nx = s.nx, ny = s.ny, nz = s.nz;
    const double hx = s.hx, hy = s.hy, hz = s.hz;
    const auto keys = s.keys;
    const auto order = s.order;
    const auto cell_counts = s.cell_counts;
    const auto cell_offsets = s.cell_offsets;
    const auto positions = s.positions;
    const auto radii = s.radii;
    using Range = Kokkos::RangePolicy<ExecSpace>;

    Kokkos::parallel_for("dense_cell_key_count", Range(exec, 0, n), KOKKOS_LAMBDA(int i) {
        const int ix = Kokkos::clamp(static_cast<int>(in_positions(i, 0) / hx), 0, nx - 1);
        const int iy = Kokkos::clamp(static_cast<int>(in_positions(i, 1) / hy), 0, ny - 1);
        const int iz = Kokkos::clamp(static_cast<int>(in_positions(i, 2) / hz), 0, nz - 1);
        const int key = (ix * ny + iy) * nz + iz;
        keys(i) = key;
        Kokkos::atomic_inc(&cell_counts(key));
    });
    Kokkos::parallel_scan("dense_cell_scan", Range(exec, 0, cells),
        KOKKOS_LAMBDA(int c, int& update, bool final) {
            const int count = cell_counts(c);
            if (final) {
                cell_offsets(c) = update;
                cell_counts(c) = 0;  // reused as the scatter cursor
            }
            update += count;
            if (final && c == cells - 1) cell_offsets(cells) = update;
        });
    Kokkos::parallel_for("dense_cell_scatter", Range(exec, 0, n), KOKKOS_LAMBDA(int i) {
        const int key = keys(i);
        order(cell_offsets(key) + Kokkos::atomic_fetch_add(&cell_counts(key), 1)) = i;
    });
    Kokkos::parallel_for("dense_cell_order", Range(exec, 0, cells), KOKKOS_LAMBDA(int c) {
        const int begin = cell_offsets(c), end = cell_offsets(c + 1);
        for (int q = begin + 1; q < end; ++q) {
            const int id = order(q);
            int k = q;
            while (k > begin && order(k - 1) > id) { order(k) = order(k - 1); --k; }
            order(k) = id;
        }
        for (int q = begin; q < end; ++q) {
            const int i = order(q);
            for (int a = 0; a < 3; ++a) positions(q, a) = in_positions(i, a);
            radii(q) = in_radii(i);
        }
        cell_counts(c) = 0;
    });
}

namespace detail {

// Visit the slot ranges of cells within reach of a point along one axis:
// calls f(unwrapped_cell, wrapped_cell) once per distinct cell.
template<class F>
KOKKOS_INLINE_FUNCTION void axis_cells(int i0, int reach, int n, bool periodic, const F& f) {
    if (periodic && 2 * reach + 1 >= n) {
        for (int c = 0; c < n; ++c) f(c, c);
        return;
    }
    const int lo = periodic ? i0 - reach : Kokkos::max(i0 - reach, 0);
    const int hi = periodic ? i0 + reach : Kokkos::min(i0 + reach, n - 1);
    for (int c = lo; c <= hi; ++c) f(c, c < 0 ? c + n : (c >= n ? c - n : c));
}

KOKKOS_INLINE_FUNCTION double gap(double x, double lo, double width) {
    return x < lo ? lo - x : (x > lo + width ? x - lo - width : 0.0);
}

KOKKOS_INLINE_FUNCTION double minimum_image(double d, double length) {
    return d - length * Kokkos::nearbyint(d / length);
}

}  // namespace detail

// Device-side view of one rebuild's search state. `for_each_segment` walks the
// cells within reach of slot t and hands each contiguous slot range to `scan`
// together with the query point shifted to that range's periodic image.
template<class ExecSpace>
struct RowSearch {
    using memory_space = typename ExecSpace::memory_space;
    Kokkos::View<double*[3], memory_space> p;
    Kokkos::View<double*, memory_space> radii;
    Kokkos::View<int*, memory_space> offsets;
    int nx, ny, nz;
    double hx, hy, hz, lx, ly, lz;
    bool periodic;

    struct Query {
        double r2;
        bool wide_x, wide_y, wide_z;
    };

    // Each visited cell has a known periodic image, so the query point is
    // shifted once per cell instead of rounding every pair. Only an axis whose
    // search spans the whole box needs the minimum image.
    KOKKOS_INLINE_FUNCTION double distance2(int q, double qx, double qy, double qz,
                                            const Query& query) const {
        double dx = p(q, 0) - qx, dy = p(q, 1) - qy, dz = p(q, 2) - qz;
        if (query.wide_x) dx = detail::minimum_image(dx, lx);
        if (query.wide_y) dy = detail::minimum_image(dy, ly);
        if (query.wide_z) dz = detail::minimum_image(dz, lz);
        return dx * dx + dy * dy + dz * dz;
    }

    template<class Scan>
    KOKKOS_INLINE_FUNCTION Query for_each_segment(int t, const Scan& scan) const {
        const double px = p(t, 0), py = p(t, 1), pz = p(t, 2);
        const double r = radii(t);
        const int ix0 = Kokkos::clamp(static_cast<int>(px / hx), 0, nx - 1);
        const int iy0 = Kokkos::clamp(static_cast<int>(py / hy), 0, ny - 1);
        const int iz0 = Kokkos::clamp(static_cast<int>(pz / hz), 0, nz - 1);
        const int rx = static_cast<int>(Kokkos::ceil(r / hx));
        const int ry = static_cast<int>(Kokkos::ceil(r / hy));
        const int rz = static_cast<int>(Kokkos::ceil(r / hz));
        const Query query{r * r, periodic && 2 * rx + 1 >= nx, periodic && 2 * ry + 1 >= ny,
                          periodic && 2 * rz + 1 >= nz};
        detail::axis_cells(ix0, rx, nx, periodic, [&](int ux, int x) {
            const double gx = query.wide_x ? 0.0 : detail::gap(px, ux * hx, hx);
            const double qx = px - (ux - x) * hx;
            detail::axis_cells(iy0, ry, ny, periodic, [&](int uy, int y) {
                const double gy = query.wide_y ? 0.0 : detail::gap(py, uy * hy, hy);
                if (gx * gx + gy * gy > query.r2) return;
                const double qy = py - (uy - y) * hy;
                const int column = (x * ny + y) * nz;
                // Along z, the cells of one (x, y) column are consecutive keys,
                // so each z segment is one contiguous slot range.
                if (query.wide_z) {
                    scan(offsets(column), offsets(column + nz), qx, qy, pz, query);
                    return;
                }
                const int z_lo = iz0 - rz, z_hi = iz0 + rz;
                const int a = Kokkos::max(z_lo, 0), b = Kokkos::min(z_hi, nz - 1);
                scan(offsets(column + a), offsets(column + b + 1), qx, qy, pz, query);
                if (periodic && z_lo < 0)
                    scan(offsets(column + z_lo + nz), offsets(column + nz), qx, qy, pz + lz, query);
                if (periodic && z_hi >= nz)
                    scan(offsets(column), offsets(column + z_hi - nz + 1), qx, qy, pz - lz, query);
            });
        });
        return query;
    }
};

// One search pass into the table. Rows longer than the capacity store only
// the first `capacity` entries but record their true count; the maximum count
// goes to `max_count` so the caller can grow the table and repeat. With
// several lanes per particle, a lane-level scan assigns each hit the position
// it would have in the one-lane search, so both produce identical rows. With
// `nearest_first`, each row is then ordered by distance, ties by slot.
template<class ExecSpace>
void fill(const ExecSpace& exec, const DenseCellSearch<ExecSpace>& s) {
    const int n = s.n, capacity = s.capacity;
    const bool nearest_first = s.nearest_first;
    const RowSearch<ExecSpace> search{s.positions, s.radii, s.cell_offsets, s.nx, s.ny, s.nz,
                                      s.hx, s.hy, s.hz, s.box[0], s.box[1], s.box[2], s.periodic};
    const auto table = s.table;
    const auto distance2 = s.distance2;
    const auto counts = s.counts;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    using Query = typename RowSearch<ExecSpace>::Query;
    using memory_space = typename ExecSpace::memory_space;

    if (s.vector_length == 1) {
        Kokkos::parallel_reduce("dense_neighbor_fill", Range(exec, 0, n),
            KOKKOS_LAMBDA(int t, int& longest) {
                int count = 0;
                search.for_each_segment(t, [&](int first, int last, double qx, double qy,
                                               double qz, const Query& query) {
                    for (int q = first; q < last; ++q) {
                        if (q == t) continue;
                        const double d2 = search.distance2(q, qx, qy, qz, query);
                        if (d2 > query.r2) continue;
                        if (count < capacity) {
                            table(t, count) = q;
                            if (nearest_first) distance2(t, count) = d2;
                        }
                        ++count;
                    }
                });
                counts(t) = count;
                longest = Kokkos::max(longest, count);
            }, Kokkos::Max<int, memory_space>(s.max_count));
    } else {
        using Policy = Kokkos::TeamPolicy<ExecSpace>;
        using Member = typename Policy::member_type;
        const int lanes = s.vector_length;
        // Host backends cap the team size at the thread count; there the lane
        // path only exists to be testable.
        const bool host = Kokkos::SpaceAccessibility<Kokkos::HostSpace, memory_space>::accessible;
        const int team_size = host ? 1 : Kokkos::max(1, 128 / lanes);
        const int league = (n + team_size - 1) / team_size;
        Kokkos::parallel_reduce("dense_neighbor_fill_lanes", Policy(exec, league, team_size, lanes),
            KOKKOS_LAMBDA(const Member& member, int& longest) {
                const int t = member.league_rank() * member.team_size() + member.team_rank();
                if (t >= n) return;
                int count = 0;
                search.for_each_segment(t, [&](int first, int last, double qx, double qy,
                                               double qz, const Query& query) {
                    int found = 0;
                    Kokkos::parallel_scan(Kokkos::ThreadVectorRange(member, last - first),
                        [&](int k, int& position, bool final) {
                            const int q = first + k;
                            const double d2 = q == t ? query.r2 + 1.0
                                                     : search.distance2(q, qx, qy, qz, query);
                            if (d2 > query.r2) return;
                            const int c = count + position;
                            if (final && c < capacity) {
                                table(t, c) = q;
                                if (nearest_first) distance2(t, c) = d2;
                            }
                            ++position;
                        }, found);
                    count += found;
                });
                Kokkos::single(Kokkos::PerThread(member), [&] { counts(t) = count; });
                // Max is idempotent, so every lane may contribute the same count.
                longest = Kokkos::max(longest, count);
            }, Kokkos::Max<int, memory_space>(s.max_count));
    }

    if (!nearest_first) return;
    Kokkos::parallel_for("dense_neighbor_nearest_first", Range(exec, 0, n), KOKKOS_LAMBDA(int t) {
        const int count = Kokkos::min(counts(t), capacity);
        for (int c = 1; c < count; ++c) {
            const double d2 = distance2(t, c);
            const int q = table(t, c);
            int k = c;
            while (k > 0 && (distance2(t, k - 1) > d2 ||
                             (distance2(t, k - 1) == d2 && table(t, k - 1) > q))) {
                distance2(t, k) = distance2(t, k - 1);
                table(t, k) = table(t, k - 1);
                --k;
            }
            distance2(t, k) = d2;
            table(t, k) = q;
        }
    });
}

// Full rebuild with growth on overflow. Returns the number of capacity
// retries; on return every row fits in the table.
template<class ExecSpace, class InPositions, class InRadii>
int rebuild(const ExecSpace& exec, DenseCellSearch<ExecSpace>& s,
            const InPositions& in_positions, const InRadii& in_radii) {
    int retries = 0;
    while (true) {
        build(exec, s, in_positions, in_radii);
        fill(exec, s);
        int longest = 0;
        Kokkos::deep_copy(exec, longest, s.max_count);
        exec.fence("neighbor_search::rebuild overflow check");
        if (longest <= s.capacity) return retries;
        configure(s, s.n, s.box, s.cell_width, s.periodic, longest + longest / 4, s.nearest_first,
                  s.vector_length);
        ++retries;
    }
}

}  // namespace neighbor_search
