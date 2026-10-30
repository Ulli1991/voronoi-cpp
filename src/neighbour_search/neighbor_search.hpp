#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <utility>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace neighbor_search {

enum class RowSort { insertion, team_bitonic, team_bitonic_recompute };

// Opt-in, synchronized stage measurements. Production calls omit this pointer
// and retain their original asynchronous launches and single overflow read.
// Times accumulate across capacity retries and include launch/fence overhead.
struct StageProfile {
    double key_count_ms=0, scan_ms=0, scatter_ms=0, cell_order_gather_ms=0;
    double candidate_fill_ms=0, nearest_order_ms=0, overflow_check_ms=0;
};

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
    // Opt-in Morton order of cell blocks; IDs within a cell remain deterministic.
    // Select before configure. The spatial query still visits every required cell.
    bool morton_cells = false;
    int morton_nx = 0, morton_ny = 0, morton_nz = 0;
    // Explicit experimental option; configure/rebuild preserve this selection.
    // The insertion path remains the default until GPU comparison is complete.
    RowSort row_sort = RowSort::insertion;
    // Lanes cooperating on one particle's search; 1 = one thread per particle.
    int vector_length = 1;

    IntView keys, order, cell_counts, cell_offsets, counts;
    IntView cell_rank; // Cartesian cell key -> rank in Morton cell order
    Positions positions;
    ScalarView radii;
    Table table;
    DistanceTable distance2;
    Kokkos::View<int, memory_space> max_count;

    int cell_count() const { return nx * ny * nz; }
};

namespace detail {

template<class ExecSpace>
void record_stage(const ExecSpace& exec, std::optional<Kokkos::Timer>& timer,
                  double* elapsed) {
    if (!elapsed) return;
    exec.fence("neighbor search stage profile");
    *elapsed+=timer->seconds()*1e3;
    timer->reset();
}

// 21 coordinate bits per axis fit in a 64-bit Morton key.
inline std::uint64_t morton_key(unsigned x, unsigned y, unsigned z) {
    std::uint64_t result=0;
    for (unsigned bit=0;bit<21;++bit) {
        result|=std::uint64_t((x>>bit)&1)<<(3*bit+2);
        result|=std::uint64_t((y>>bit)&1)<<(3*bit+1);
        result|=std::uint64_t((z>>bit)&1)<<(3*bit);
    }
    return result;
}

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
    if (s.morton_cells && (s.morton_nx!=s.nx || s.morton_ny!=s.ny || s.morton_nz!=s.nz)) {
        if (s.nx>(1<<21) || s.ny>(1<<21) || s.nz>(1<<21))
            throw std::invalid_argument("Morton cell coordinates exceed 21 bits");
        // Geometry-dependent map, cached across rebuilds and capacity growth.
        std::vector<std::pair<std::uint64_t,int>> cells_by_key;
        cells_by_key.reserve(cells);
        for (int x=0;x<s.nx;++x)
            for (int y=0;y<s.ny;++y)
                for (int z=0;z<s.nz;++z)
                    cells_by_key.emplace_back(detail::morton_key(x,y,z),(x*s.ny+y)*s.nz+z);
        std::sort(cells_by_key.begin(),cells_by_key.end());
        detail::ensure(s.cell_rank,"morton_cell_rank",cells);
        auto rank=Kokkos::create_mirror(s.cell_rank);
        for (int i=0;i<cells;++i) rank(cells_by_key[i].second)=i;
        Kokkos::deep_copy(s.cell_rank,rank);
        s.morton_nx=s.nx; s.morton_ny=s.ny; s.morton_nz=s.nz;
    }
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
    // The opt-in scratch sort keeps only candidate IDs globally. Oversized
    // rows retain the insertion fallback and its persistent distance storage.
    if (nearest_first && (s.row_sort != RowSort::team_bitonic_recompute || capacity > 2048))
        detail::ensure(s.distance2, "neighbor_distance2", n, capacity);
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
           const InPositions& in_positions, const InRadii& in_radii,
           StageProfile* profile = nullptr) {
    std::optional<Kokkos::Timer> stage_timer;
    if (profile) {
        exec.fence("neighbor build profiling start");
        stage_timer.emplace();
    }
    const int n = s.n, cells = s.cell_count();
    const int nx = s.nx, ny = s.ny, nz = s.nz;
    const double hx = s.hx, hy = s.hy, hz = s.hz;
    const auto keys = s.keys;
    const auto cell_rank = s.cell_rank;
    const bool morton = s.morton_cells;
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
        const int cartesian = (ix * ny + iy) * nz + iz;
        const int key = morton ? cell_rank(cartesian) : cartesian;
        keys(i) = key;
        Kokkos::atomic_inc(&cell_counts(key));
    });
    detail::record_stage(exec,stage_timer,profile ? &profile->key_count_ms : nullptr);
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
    detail::record_stage(exec,stage_timer,profile ? &profile->scan_ms : nullptr);
    Kokkos::parallel_for("dense_cell_scatter", Range(exec, 0, n), KOKKOS_LAMBDA(int i) {
        const int key = keys(i);
        order(cell_offsets(key) + Kokkos::atomic_fetch_add(&cell_counts(key), 1)) = i;
    });
    detail::record_stage(exec,stage_timer,profile ? &profile->scatter_ms : nullptr);
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
    detail::record_stage(exec,stage_timer,profile ? &profile->cell_order_gather_ms : nullptr);
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
    Kokkos::View<int*, memory_space> cell_rank;
    bool morton=false;

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
                // Cartesian blocks can merge a whole z segment. Morton blocks
                // use per-cell ranges: the same spatial coverage, different layout.
                auto segment=[&](int first_z,int last_z,double shifted_z) {
                    if (first_z>=last_z) return;
                    if (!morton) {
                        scan(offsets(column+first_z),offsets(column+last_z),qx,qy,shifted_z,query);
                    } else {
                        for (int z=first_z;z<last_z;++z) {
                            const int rank=cell_rank(column+z);
                            scan(offsets(rank),offsets(rank+1),qx,qy,shifted_z,query);
                        }
                    }
                };
                if (query.wide_z) {
                    segment(0,nz,pz);
                    return;
                }
                const int z_lo = iz0 - rz, z_hi = iz0 + rz;
                const int a = Kokkos::max(z_lo, 0), b = Kokkos::min(z_hi, nz - 1);
                segment(a,b+1,pz);
                if (periodic && z_lo < 0)
                    segment(z_lo+nz,nz,pz+lz);
                if (periodic && z_hi >= nz)
                    segment(0,z_hi-nz+1,pz-lz);
            });
        });
        return query;
    }
};

struct AllRows {
    KOKKOS_INLINE_FUNCTION bool operator()(int) const { return true; }
};

// Parallel bitonic network in team scratch: each candidate is loaded/stored
// once, instead of insertion sort's repeated strided global-memory shifts.
// The network compares (distance squared, slot), matching insertion exactly.
// Return false for oversized tables so callers can use the preserved fallback.
// Scratch is bounded at about 24 KiB/team, with no persistent or host allocations.
template<class ExecSpace, class RowFilter=AllRows>
bool sort_rows_team(const ExecSpace& exec, const DenseCellSearch<ExecSpace>& s, RowFilter selected={}) {
    constexpr int max_team_capacity = 2048;
    if (s.capacity > max_team_capacity || s.capacity < 0) return false;
    if (s.n == 0) return true;
    int padded_capacity = 1;
    while (padded_capacity < s.capacity) padded_capacity *= 2;
    using Policy = Kokkos::TeamPolicy<ExecSpace>;
    using Member = typename Policy::member_type;
    using Scratch = typename ExecSpace::scratch_memory_space;
    using Distances = Kokkos::View<double*, Scratch, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using Ids = Kokkos::View<int*, Scratch, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    const bool host = Kokkos::SpaceAccessibility<Kokkos::HostSpace,
        typename ExecSpace::memory_space>::accessible;
    const int team_size = host ? 1 : 128;
    const auto table = s.table;
    const auto distance2 = s.distance2;
    const auto counts = s.counts;
    const auto positions = s.positions;
    const auto box = s.box;
    const bool periodic = s.periodic;
    const bool recompute = s.row_sort == RowSort::team_bitonic_recompute;
    const int capacity = s.capacity;
    const double padding_distance = std::numeric_limits<double>::infinity();
    const int padding_id = std::numeric_limits<int>::max();
    Policy policy(exec,s.n,team_size);
    policy.set_scratch_size(0,Kokkos::PerTeam(
        Distances::shmem_size(padded_capacity)+Ids::shmem_size(padded_capacity)));
    Kokkos::parallel_for("dense_neighbor_team_bitonic",policy,KOKKOS_LAMBDA(const Member& member) {
        const int row = member.league_rank();
        if (!selected(row)) return;
        const int count = Kokkos::clamp(counts(row),0,capacity);
        int padded = 1;
        while (padded < count) padded *= 2;
        Distances distances(member.team_scratch(0),padded_capacity);
        Ids ids(member.team_scratch(0),padded_capacity);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member,padded),[&](int i) {
            ids(i) = i < count ? table(row,i) : padding_id;
            double d2 = padding_distance;
            if (i < count) {
                if (recompute) {
                    const int other = ids(i);
                    double dx = positions(other,0)-positions(row,0);
                    double dy = positions(other,1)-positions(row,1);
                    double dz = positions(other,2)-positions(row,2);
                    if (periodic) {
                        dx = detail::minimum_image(dx,box[0]);
                        dy = detail::minimum_image(dy,box[1]);
                        dz = detail::minimum_image(dz,box[2]);
                    }
                    d2 = dx*dx+dy*dy+dz*dz;
                } else d2 = distance2(row,i);
            }
            distances(i) = d2;
        });
        member.team_barrier();
        for (int width=2;width<=padded;width*=2) {
            for (int stride=width/2;stride>0;stride/=2) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(member,padded),[&](int i) {
                    const int j = i ^ stride;
                    if (j <= i) return;
                    const double a = distances(i), b = distances(j);
                    const int ia = ids(i), ib = ids(j);
                    const bool a_less = a < b || (a == b && ia < ib);
                    const bool b_less = b < a || (b == a && ib < ia);
                    if ((i & width) == 0 ? b_less : a_less) {
                        distances(i)=b; distances(j)=a;
                        ids(i)=ib; ids(j)=ia;
                    }
                });
                member.team_barrier();
            }
        }
        Kokkos::parallel_for(Kokkos::TeamThreadRange(member,count),[&](int i) {
            if (!recompute) distance2(row,i)=distances(i);
            table(row,i)=ids(i);
        });
    });
    return true;
}

// One search pass into the table. Rows longer than the capacity store only
// the first `capacity` entries but record their true count; the maximum count
// goes to `max_count` so the caller can grow the table and repeat. With
// several lanes per particle, a lane-level scan assigns each hit the position
// it would have in the one-lane search, so both produce identical rows. With
// `nearest_first`, each row is then ordered by distance, ties by slot.
template<class ExecSpace>
void fill(const ExecSpace& exec, const DenseCellSearch<ExecSpace>& s,
          StageProfile* profile = nullptr) {
    std::optional<Kokkos::Timer> stage_timer;
    if (profile) {
        exec.fence("neighbor fill profiling start");
        stage_timer.emplace();
    }
    const int n = s.n, capacity = s.capacity;
    const bool nearest_first = s.nearest_first;
    const bool store_distances = nearest_first &&
        (s.row_sort != RowSort::team_bitonic_recompute || capacity > 2048);
    const RowSearch<ExecSpace> search{s.positions, s.radii, s.cell_offsets, s.nx, s.ny, s.nz,
                                      s.hx, s.hy, s.hz, s.box[0], s.box[1], s.box[2], s.periodic, s.cell_rank, s.morton_cells};
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
                            if (store_distances) distance2(t, count) = d2;
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
                                if (store_distances) distance2(t, c) = d2;
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

    detail::record_stage(exec,stage_timer,profile ? &profile->candidate_fill_ms : nullptr);
    if (!nearest_first) return;
    if (s.row_sort != RowSort::insertion && sort_rows_team(exec,s)) {
        detail::record_stage(exec,stage_timer,profile ? &profile->nearest_order_ms : nullptr);
        return;
    }
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
    detail::record_stage(exec,stage_timer,profile ? &profile->nearest_order_ms : nullptr);
}

// Full rebuild with growth on overflow. Returns the number of capacity
// retries; on return every row fits in the table.
template<class ExecSpace, class InPositions, class InRadii>
int rebuild(const ExecSpace& exec, DenseCellSearch<ExecSpace>& s,
            const InPositions& in_positions, const InRadii& in_radii,
            StageProfile* profile = nullptr) {
    if (profile) *profile=StageProfile{};
    int retries = 0;
    while (true) {
        build(exec, s, in_positions, in_radii, profile);
        fill(exec, s, profile);
        std::optional<Kokkos::Timer> overflow_timer;
        if (profile) overflow_timer.emplace();
        int longest = 0;
        Kokkos::deep_copy(exec, longest, s.max_count);
        exec.fence("neighbor_search::rebuild overflow check");
        if (profile) profile->overflow_check_ms+=overflow_timer->seconds()*1e3;
        if (longest <= s.capacity) return retries;
        configure(s, s.n, s.box, s.cell_width, s.periodic, longest + longest / 4, s.nearest_first,
                  s.vector_length);
        ++retries;
    }
}

}  // namespace neighbor_search
