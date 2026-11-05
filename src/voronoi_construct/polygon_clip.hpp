#pragma once

#include "exact_predicates_2d.hpp"
#include "neighbor_search.hpp"
#include "shell_voronoi.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// The exact Voronoi cells of particles in a periodic 2D box, each built alone
// by clipping a polygon with its neighbours (shell_voronoi's walk in 2D: grid
// cells around the generator nearest first, on the level fitted to the local
// density, until nothing within twice the farthest vertex is left).
//
// A cell is a convex polygon: a cycle of edges, each the bisector with one
// neighbour, and vertices between consecutive edges. A bisector cuts off one
// contiguous arc of vertices, which two new vertices replace. Every decision
// is exact: whether a vertex lies beyond a bisector is the perturbed in-circle
// test of the generator, the vertex's two neighbours and the candidate
// (exact_predicates_2d.hpp), which a double comparison with a rounding bound
// settles nearly always. So the cells agree with each other: q is in the ring
// of p exactly if p is in the ring of q. There is one kind of cell, no rounded
// pass and no patching.
//
// All lengths are in grid steps, relative to the generator. The output is the
// ring of each cell: its neighbours in counter-clockwise order, the Delaunay
// star of the generator (Rings).
namespace polygon_clip {

namespace ep = exact_predicates;
namespace sv = shell_voronoi;

enum Status : int {
    ok = 0,
    uncertified = 1,  // the walk reached the coarsest grid the box allows
    overflow = 2,     // more edges than the polygon holds
    pool_full = 3,    // the cell's chunk of the ring pool is full
};

constexpr int usual_edges = 32, big_edges = 256;  // polygon capacities
constexpr int candidates = 16;                    // clipped nearest-first from a small buffer

// Read-only inputs in slot order, from a built neighbor_search::CellList.
template <class MemorySpace>
struct Grid {
    Kokkos::View<double* [2], MemorySpace> positions;
    Kokkos::View<int*, MemorySpace> order, cell_offsets;
    Kokkos::View<std::uint32_t*, MemorySpace> fine;
    Kokkos::View<std::uint8_t*, MemorySpace> level;
    int n = 0, levels = 2;
    ep::BoxN<2> box;
    int tiles[2] = {1, 1};
    double width[2] = {1, 1};  // tile widths in grid steps
    double narrowest = 1;
    int fewest = 1;

    // Morton number of (x, y), at most 10 bits each.
    KOKKOS_INLINE_FUNCTION static std::uint32_t spread(std::uint32_t v) {
        v &= 0x3ff;
        v = (v | v << 8) & 0x00ff00ffu;
        v = (v | v << 4) & 0x0f0f0f0fu;
        v = (v | v << 2) & 0x33333333u;
        v = (v | v << 1) & 0x55555555u;
        return v;
    }
    KOKKOS_INLINE_FUNCTION static std::uint32_t morton(std::uint32_t x, std::uint32_t y) { return spread(x) << 1 | spread(y); }

    // The first base cell of grid cell (x, y) at `at` <= levels.
    KOKKOS_INLINE_FUNCTION int cell(int at, int x, int y) const {
        const int mask = (1 << at) - 1;
        const int tile = (x >> at) * tiles[1] + (y >> at);
        return tile << (2 * levels) | static_cast<int>(morton(x & mask, y & mask)) << (2 * (levels - at));
    }
    // The slots [begin, end) of grid cell (x, y) at level `at`.
    KOKKOS_INLINE_FUNCTION void range(int at, int x, int y, int& begin, int& end) const {
        const int below = at - levels;
        if (below <= 0) {
            const int first = cell(at, x, y);
            begin = cell_offsets(first);
            end = cell_offsets(first + (1 << (-2 * below)));
            return;
        }
        const int base = cell(levels, x >> below, y >> below);
        begin = cell_offsets(base);
        end = cell_offsets(base + 1);
        if (begin == end) return;
        const int mask = (1 << below) - 1, drop = 2 * (neighbor_search::fine_levels - below);
        const std::uint32_t key = morton(x & mask, y & mask) << drop;
        begin = neighbor_search::lower_bound(fine, begin, end, key);
        end = neighbor_search::lower_bound(fine, begin, end, key + (std::uint32_t(1) << drop));
    }
};

template <class ExecSpace>
Grid<typename ExecSpace::memory_space> make_grid(const neighbor_search::CellList<ExecSpace, 2>& list) {
    Grid<typename ExecSpace::memory_space> g{list.positions, list.order, list.cell_offsets, list.fine, list.level,
                                             list.n, list.levels, list.box};
    g.narrowest = static_cast<double>(list.box.period[0]);
    g.fewest = list.tiles[0];
    for (int a = 0; a < 2; ++a) {
        g.tiles[a] = list.tiles[a];
        g.width[a] = static_cast<double>(list.box.period[a]) / list.tiles[a];
        g.narrowest = std::min(g.narrowest, g.width[a]);
        g.fewest = std::min(g.fewest, g.tiles[a]);
    }
    return g;
}

// Grid-cell offsets (5 bits per axis, biased by 8) sorted by the squared
// smallest distance, in cell widths, any point of the home cell can have to
// them (shell_voronoi::Stencil in 2D).
template <class ExecSpace>
sv::Stencil<typename ExecSpace::memory_space> make_stencil(const ExecSpace& exec) {
    using Memory = typename ExecSpace::memory_space;
    constexpr int reach = sv::max_reach;
    std::vector<std::pair<int, int>> entries;
    auto beyond = [](int d) { return std::max(std::abs(d) - 1, 0); };
    for (int dx = -reach; dx <= reach; ++dx)
        for (int dy = -reach; dy <= reach; ++dy) {
            const int gaps = beyond(dx) * beyond(dx) + beyond(dy) * beyond(dy);
            if (gaps < reach * reach) entries.push_back({gaps, (dx + 8) | (dy + 8) << 5});
        }
    std::sort(entries.begin(), entries.end());
    sv::Stencil<Memory> s{Kokkos::View<int*, Memory>("polygon_stencil", entries.size()),
                          Kokkos::View<float*, Memory>("polygon_stencil_bound2", entries.size()),
                          Kokkos::View<int*, Memory>("polygon_stencil_ends", reach + 1)};
    auto offsets = Kokkos::create_mirror_view(s.offsets);
    auto bound2 = Kokkos::create_mirror_view(s.bound2);
    auto ends = Kokkos::create_mirror_view(s.ends);
    Kokkos::deep_copy(ends, 0);
    for (std::size_t i = 0; i < entries.size(); ++i) {
        offsets(i) = entries[i].second;
        bound2(i) = std::nextafter(float(entries[i].first), 0.0f);  // never overstates a distance
        for (int r = 0; r <= reach; ++r)
            if (entries[i].first < r * r) ends(r) = static_cast<int>(i) + 1;
    }
    Kokkos::deep_copy(exec, s.offsets, offsets);
    Kokkos::deep_copy(exec, s.bound2, bound2);
    Kokkos::deep_copy(exec, s.ends, ends);
    return s;
}

// The rings: cell i's neighbours, counter-clockwise, are
// ring(start(i)) .. ring(start(i) + degree(i) - 1). The degrees sum to 6n on
// a torus; the pool holds them in chunks claimed by hashed counters.
template <class MemorySpace>
struct Rings {
    Kokkos::View<int*, MemorySpace> status, degree, start, ring;
    Kokkos::View<unsigned*, MemorySpace> used;

    Rings() = default;
    Rings(int n, long long entries)
        : status(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("ring_status")), n),
          degree(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("ring_degree")), n),
          start(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("ring_start")), n),
          ring(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("rings")), entries),
          used("ring_used", std::clamp(n / 4096, 1, 4096)) {}
    long long capacity() const { return static_cast<long long>(ring.extent(0)); }
    // Pool entries for n cells: 6n and a margin for the hashing of the chunks.
    static long long entries_for(int n) { return 6LL * n + n / 8 + 256; }
};

// A corner of a polygon: edge m, the bisector with the particle at offset o
// in `slot` (the virtual particles of the starting square are negative), and
// the vertex between edges m and m + 1 with a bound on its rounding error per
// axis.
struct Corner {
    double ox, oy, vx, vy, tol;
    int slot;
};

// The vertex where the bisectors of a and b meet (b counter-clockwise of a).
KOKKOS_INLINE_FUNCTION void set_vertex(Corner& out, double ax, double ay, double bx, double by) {
    const double la = ax * ax + ay * ay, lb = bx * bx + by * by;
    const double det = ax * by - ay * bx, scale = 0.5 / det;
    out.vx = (la * by - lb * ay) * scale;
    out.vy = (lb * ax - la * bx) * scale;
    const double spread = Kokkos::abs(ax * by) + Kokkos::abs(ay * bx);
    const double numerator = la * (Kokkos::abs(bx) + Kokkos::abs(by)) + lb * (Kokkos::abs(ax) + Kokkos::abs(ay));
    out.tol = 8 * ep::epsilon * ((Kokkos::abs(out.vx) + Kokkos::abs(out.vy)) * spread * 2 + numerator) * Kokkos::abs(scale);
    if (!(spread * 1e-14 < Kokkos::abs(det))) out.tol = Kokkos::Experimental::infinity_v<double>;
}

template <int Capacity>
struct Polygon {
    Corner corner[Capacity];
    int count = 4;
    double reach2 = 0;  // above 4 R^2

    // The starting square of half-width `half`, bounded by 4 virtual particles.
    KOKKOS_INLINE_FUNCTION void start(double half) {
        const double x[4] = {2 * half, 0, -2 * half, 0}, y[4] = {0, 2 * half, 0, -2 * half};
        for (int m = 0; m < 4; ++m) {
            corner[m].ox = x[m];
            corner[m].oy = y[m];
            corner[m].slot = m - 4;
            set_vertex(corner[m], x[m], y[m], x[(m + 1) % 4], y[(m + 1) % 4]);
        }
        measure();
    }
    // 4 R^2 and a margin: particles exactly at 2R (cocircular) are tested.
    KOKKOS_INLINE_FUNCTION void measure() {
        double r2 = 0;
        for (int m = 0; m < count; ++m) {
            const double v = corner[m].vx * corner[m].vx + corner[m].vy * corner[m].vy;
            if (v > r2) r2 = v;
        }
        reach2 = 4 * r2 * (1 + 1e-9);
    }
};

// Whether the bisector with e (offset ex, ey, lift le) cuts vertex m of the
// polygon: e lies inside the circle of the generator and the vertex's two
// neighbours. The double comparison decides unless within rounding of a tie.
template <class Exact, int Capacity>
KOKKOS_INLINE_FUNCTION bool beyond(const Polygon<Capacity>& p, int m, double ex, double ey, double le, const Exact& exact) {
    const Corner& c = p.corner[m];
    const double s = c.vx * ex + c.vy * ey - 0.5 * le;
    const double bound = c.tol * (Kokkos::abs(ex) + Kokkos::abs(ey)) +
                         4 * ep::epsilon * (Kokkos::abs(c.vx * ex) + Kokkos::abs(c.vy * ey) + 0.5 * le);
    if (s > bound) return true;
    if (s < -bound) return false;
    return exact(m);
}

// Clips polygon p with the particle at offset (ex, ey): false if it overflows.
template <class Exact, int Capacity>
KOKKOS_INLINE_FUNCTION bool clip(Polygon<Capacity>& p, double ex, double ey, int slot, const Exact& exact) {
    const double le = ex * ex + ey * ey;
    // The cut arc of vertices [first, last] (cyclic), found in one pass.
    int cuts = 0, first = -1, last = -1;
    const bool at_end = beyond(p, p.count - 1, ex, ey, le, exact);
    bool previous = at_end;
    for (int m = 0; m < p.count; ++m) {
        const bool cut = m + 1 == p.count ? at_end : beyond(p, m, ex, ey, le, exact);
        cuts += cut;
        if (cut && !previous) first = m;
        if (!cut && previous) last = m == 0 ? p.count - 1 : m - 1;
        previous = cut;
    }
    if (cuts == 0) return true;
    if (cuts == p.count || p.count + 2 - cuts > Capacity) return false;
    const Corner before = p.corner[first], after = p.corner[last + 1 == p.count ? 0 : last + 1];
    int f = first;
    if (last < first) {  // the arc wraps: the edges last + 1 .. first stay
        const int shift = last + 1;
        for (int m = shift; m <= first; ++m) p.corner[m - shift] = p.corner[m];
        f = first - shift;
        p.count = f + 1;
    } else {  // the edges 0 .. first and last + 1 .. stay; room for one edge
        const int tail = p.count - (last + 1), move = first + 2 - (last + 1);
        if (move > 0)
            for (int m = tail - 1; m >= 0; --m) p.corner[last + 1 + m + move] = p.corner[last + 1 + m];
        else if (move < 0)
            for (int m = 0; m < tail; ++m) p.corner[last + 1 + m + move] = p.corner[last + 1 + m];
        p.count += move - 1;
    }
    set_vertex(p.corner[f], before.ox, before.oy, ex, ey);
    Corner& added = p.corner[f + 1];
    added.ox = ex;
    added.oy = ey;
    added.slot = slot;
    set_vertex(added, ex, ey, after.ox, after.oy);
    ++p.count;
    p.measure();
    return true;
}

struct Candidate {
    double distance2, ox, oy;
    int slot;
};

// Builds the polygons of the selected slots (all if none) with capacity
// Capacity; `failed`: only those of cells whose status is overflow. Writes
// status, degree, and ring blocks claimed from the pool.
template <int Capacity, class ExecSpace>
void construct_pass(const ExecSpace& exec, const Grid<typename ExecSpace::memory_space>& grid,
                    const sv::Stencil<typename ExecSpace::memory_space>& stencil,
                    const Rings<typename ExecSpace::memory_space>& out, bool failed,
                    const Kokkos::View<const int*, typename ExecSpace::memory_space>& slots) {
    const bool selected = slots.data() != nullptr;
    const int count = selected ? static_cast<int>(slots.extent(0)) : grid.n;
    const ep::BoxN<2> box = grid.box;
    const double to_steps = 1.0 / box.step;
    const double half = 0.25 * static_cast<double>(std::min(box.period[0], box.period[1]));
    const long long capacity = out.capacity();
    const auto positions = grid.positions;

    Kokkos::parallel_for(
        "polygon_clip::construct", Kokkos::RangePolicy<ExecSpace>(exec, 0, count), KOKKOS_LAMBDA(int c) {
            const int t = selected ? slots(c) : c, id = grid.order(t);
            if (failed && out.status(id) != overflow) return;
            const double p[2] = {positions(t, 0), positions(t, 1)};
            const double self[2] = {Kokkos::nearbyint(p[0] * to_steps), Kokkos::nearbyint(p[1] * to_steps)};
            Polygon<Capacity> polygon;
            polygon.start(half);
            int status = ok;

            auto particle = [&](int slot) { return slot < 0 ? slot : grid.order(slot); };
            Candidate buffer[candidates];
            int buffered = 0;
            auto flush = [&]() {  // nearest first; the rest may be out of reach
                for (int a = 1; a < buffered; ++a) {
                    const Candidate k = buffer[a];
                    int b = a;
                    for (; b > 0 && buffer[b - 1].distance2 > k.distance2; --b) buffer[b] = buffer[b - 1];
                    buffer[b] = k;
                }
                for (int a = 0; a < buffered && status == ok; ++a) {
                    const Candidate k = buffer[a];
                    if (k.distance2 > polygon.reach2) break;
                    auto exact = [&](int m) {  // the perturbed in-circle test of vertex m
                        const Corner& e0 = polygon.corner[m];
                        const Corner& e1 = polygon.corner[m + 1 == polygon.count ? 0 : m + 1];
                        const ep::Offset2 points[4] = {{0, 0},
                                                       {static_cast<ep::i64>(e0.ox), static_cast<ep::i64>(e0.oy)},
                                                       {static_cast<ep::i64>(e1.ox), static_cast<ep::i64>(e1.oy)},
                                                       {static_cast<ep::i64>(k.ox), static_cast<ep::i64>(k.oy)}};
                        const int ids[4] = {id, particle(e0.slot), particle(e1.slot), particle(k.slot)};
                        return ep::in_circle(points, ids);
                    };
                    if (!clip(polygon, k.ox, k.oy, k.slot, exact)) status = overflow;
                }
                buffered = 0;
            };

            // The walk: the stencil at the slot's level, then coarser.
            int level = grid.level(t), finer = -1;
            int home[2];
            for (int a = 0; a < 2; ++a)
                home[a] = neighbor_search::grid_coordinate(p[a], box.side[a], grid.tiles[a] << level);
            bool certified = false;
            while (status == ok && !certified) {
                const double width = grid.narrowest / (1 << level);
                const int reach = sv::level_reach(level, grid.fewest), entries = stencil.ends(reach);
                for (int e = 0; e < entries && status == ok; ++e) {
                    if (double(stencil.bound2(e)) * width * width >= polygon.reach2) {
                        certified = true;
                        break;
                    }
                    const int code = stencil.offsets(e);
                    const int u[2] = {home[0] + (code & 31) - 8, home[1] + ((code >> 5) & 31) - 8};
                    double gap2 = 0;
                    for (int a = 0; a < 2; ++a) {
                        const double w = grid.width[a] / (1 << level);
                        const double g = Kokkos::max(0.0, Kokkos::max(u[a] * w - self[a], self[a] - (u[a] + 1) * w));
                        gap2 += g * g;
                    }
                    bool skip = gap2 > polygon.reach2;
                    if (!skip && finer >= 0) {  // inside the finished walk one level finer?
                        int gaps = 0;
                        for (int a = 0; a < 2; ++a) {
                            const int child = 2 * (u[a] - home[a]) - ((finer >> a) & 1);
                            gaps += Kokkos::max(sv::beyond2(child), sv::beyond2(child + 1));
                        }
                        const int covered = sv::level_reach(level + 1, grid.fewest);
                        skip = gaps < covered * covered;
                    }
                    if (!skip) {
                        int w[2];
                        double image[2];  // of the slots' coordinates, in steps
                        for (int a = 0; a < 2; ++a) {
                            const int cells = grid.tiles[a] << level;
                            const int shift = u[a] < 0 ? -1 : u[a] >= cells ? 1 : 0;
                            w[a] = u[a] - shift * cells;
                            image[a] = static_cast<double>(shift * box.period[a]);
                        }
                        int begin = 0, end = 0;
                        grid.range(level, w[0], w[1], begin, end);
                        for (int s = begin; s < end; ++s) {
                            if (s == t) continue;
                            const double ox = Kokkos::nearbyint(positions(s, 0) * to_steps) + image[0] - self[0];
                            const double oy = Kokkos::nearbyint(positions(s, 1) * to_steps) + image[1] - self[1];
                            const double d2 = ox * ox + oy * oy;
                            if (!(d2 <= polygon.reach2)) continue;
                            if (buffered == candidates) flush();
                            buffer[buffered++] = {d2, ox, oy, s};
                        }
                    }
                    if (e + 1 == entries || stencil.bound2(e + 1) > stencil.bound2(e)) flush();
                }
                if (status != ok || certified) break;
                flush();
                // Cells outside the stencil are at least `reach` widths away.
                if (reach * width * reach * width >= polygon.reach2) {
                    certified = true;
                    break;
                }
                if (level == 2) break;  // no coarser grid fits the box
                finer = (home[0] & 1) | (home[1] & 1) << 1;
                home[0] >>= 1;
                home[1] >>= 1;
                --level;
            }
            flush();
            if (status == ok && !certified) status = uncertified;
            for (int m = 0; m < polygon.count && status == ok; ++m)
                if (polygon.corner[m].slot < 0) status = uncertified;
            if (status == ok) {
                int at = 0;
                if (sv::claim(out.used, capacity, id, polygon.count, at)) {
                    for (int m = 0; m < polygon.count; ++m) out.ring(at + m) = grid.order(polygon.corner[m].slot);
                    out.start(id) = at;
                } else {
                    status = pool_full;
                }
            }
            out.status(id) = status;
            out.degree(id) = status == ok ? polygon.count : 0;
        });
}

// The cells of status `which`.
template <class ExecSpace>
long long count_status(const ExecSpace& exec, const Rings<typename ExecSpace::memory_space>& rings, int which) {
    const auto status = rings.status;
    long long count = 0;
    Kokkos::parallel_reduce(
        "polygon_clip::status", Kokkos::RangePolicy<ExecSpace>(exec, 0, status.extent(0)),
        KOKKOS_LAMBDA(int i, long long& sum) { sum += status(i) == which; }, count);
    return count;
}

// The rings of all particles (or of the listed slots): the usual polygon,
// the rare larger cells again with a big one, a larger pool if it was full.
// Uncertified cells stay in the status (voronoi::Mesh refuses them).
template <class ExecSpace>
Rings<typename ExecSpace::memory_space> construct(const ExecSpace& exec, const Grid<typename ExecSpace::memory_space>& grid,
                                                  const sv::Stencil<typename ExecSpace::memory_space>& stencil,
                                                  const Kokkos::View<const int*, typename ExecSpace::memory_space>& slots = {}) {
    using Memory = typename ExecSpace::memory_space;
    long long entries = Rings<Memory>::entries_for(grid.n);
    for (;;) {
        Rings<Memory> rings(grid.n, entries);
        if (slots.data() != nullptr) Kokkos::deep_copy(exec, rings.status, static_cast<int>(uncertified));
        construct_pass<usual_edges>(exec, grid, stencil, rings, false, slots);
        if (count_status(exec, rings, overflow) > 0) {
            construct_pass<big_edges>(exec, grid, stencil, rings, true, slots);
            if (count_status(exec, rings, overflow) > 0)
                throw std::runtime_error("polygon_clip::construct: a cell with more than 256 edges");
        }
        if (count_status(exec, rings, pool_full) == 0) return rings;
        entries += entries / 2;
    }
}

}  // namespace polygon_clip
