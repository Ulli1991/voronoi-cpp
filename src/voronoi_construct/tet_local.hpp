#pragma once

#include "exact_predicates.hpp"
#include "shell_voronoi.hpp"
#include "tet_flips.hpp"
#include "tet_mesh.hpp"
#include "tet_repair.hpp"

#include <algorithm>
#include <string>

// Stage 4c (local repair, device; ideas.md S2): called by voronoi::Mesh<3>::update when tet_repair::repair leaves stuck
// entries. Clips the cells around them exactly, swaps their tetrahedra for those the cells imply and lets the device
// repair check the result; a stitch that does not close widens once, else the caller rebuilds.
namespace tet_local {

namespace ep = exact_predicates;

// What repair() did (counts of its last attempt). Used by: Mesh::update (Report::local_*), the bench.
struct Stats {
    int seeds = 0;                 // particles of the stuck entries
    int cells = 0;                 // cells clipped exactly (the affected set)
    int removed = 0, created = 0;  // tetrahedra replaced
    int attempts = 0;
    long long flips = 0;  // device flips of the checking repairs
    bool done = false;    // the mesh passed the device repair's exact check
};

// Affected-set flags per particle: none, a seed (particle of a stuck entry), affected, or added by the pass running.
namespace flag {
constexpr char none = 0, seed = 1, affected = 2, added = 3;
}

namespace detail {

// Whether cell i emits tetrahedron (i, j, p, q) of its face to j: i is the lowest affected particle of the four and j
// the lowest of the other three, so each tetrahedron with an affected corner comes once. Used by: count, emit.
template <class Flags>
KOKKOS_INLINE_FUNCTION bool emits(const Flags& f, int i, int j, int p, int q) {
    return j < p && j < q && !(f(j) && j < i) && !(f(p) && p < i) && !(f(q) && q < i);
}

// A hash of a sorted triangle into a table of mask + 1 entries.
KOKKOS_INLINE_FUNCTION unsigned hash(const int* key, unsigned mask) {
    unsigned long long h = static_cast<unsigned>(key[0]) * 0x9e3779b97f4a7c15ULL;
    h ^= (h >> 29) + static_cast<unsigned>(key[1]) * 0xbf58476d1ce4e5b9ULL;
    h ^= (h >> 31) + static_cast<unsigned>(key[2]) * 0x94d049bb133111ebULL;
    return static_cast<unsigned>(h ^ (h >> 32)) & mask;
}

// Triangles of the stitch: new tetrahedron faces (value 4e + corner) and the kept side of the old boundary
// (value -1 - tetrahedron); each key must occur exactly twice. Used by: repair.
template <class MemorySpace>
struct Table {
    Kokkos::View<int*, MemorySpace> used, value;
    Kokkos::View<int* [3], Kokkos::LayoutRight, MemorySpace> key;
    unsigned mask = 0;

    KOKKOS_INLINE_FUNCTION void insert(const int* k, int v) const {
        for (unsigned h = hash(k, mask);; h = (h + 1) & mask)
            if (Kokkos::atomic_compare_exchange(&used(h), 0, 1) == 0) {
                for (int a = 0; a < 3; ++a) key(h, a) = k[a];
                value(h) = v;
                return;
            }
    }
    // The values stored under k other than `self`: how many, and the last one in `other`.
    KOKKOS_INLINE_FUNCTION int others(const int* k, int self, int& other) const {
        int count = 0;
        for (unsigned h = hash(k, mask); used(h); h = (h + 1) & mask)
            if (key(h, 0) == k[0] && key(h, 1) == k[1] && key(h, 2) == k[2] && value(h) != self) {
                other = value(h);
                ++count;
            }
        return count;
    }
};

}  // namespace detail

// Replace the tetrahedra around the stuck entries of the last tet_repair::repair by exact local Delaunay cells: seeds
// (their particles), first ring, the seeds' new neighbours (a jumped particle's), clipped on `grid` (the cell list of
// the positions); widened by a ring and all new neighbours on a second attempt. Used by: voronoi::Mesh<3>::update.
template <class ExecSpace>
Stats repair(const ExecSpace& exec, const shell_voronoi::Grid<typename ExecSpace::memory_space>& grid,
             tet_mesh::Mesh<typename ExecSpace::memory_space>& m, tet_repair::State<typename ExecSpace::memory_space>& s) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    using Ints = Kokkos::View<int*, Memory>;
    auto uninitialized = [](const char* label) {
        return Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label));
    };
    const int n = grid.n;
    Stats stats;
    const Kokkos::View<char*, Memory> flags("local_flags", n);
    const auto order = grid.order;
    const auto point = s.point;
    auto cells = shell_voronoi::Cells<Memory>::with_rings(n, 4096);
    cells.ring_used = Kokkos::View<unsigned*, Memory>("local_ring_used", 1);  // one pool chunk: few cells
    const Kokkos::View<int, Memory> failures("local_failures");
    auto read = [&](const auto& scalar) {
        int value = 0;
        Kokkos::deep_copy(exec, value, scalar);
        exec.fence();
        return value;
    };
    // Flags `added` -> `affected`.
    auto settle = [&] {
        Kokkos::parallel_for(
            "tet_local::settle", Range(exec, 0, n), KOKKOS_LAMBDA(int i) {
                if (flags(i) == flag::added) flags(i) = flag::affected;
            });
    };
    // Particles of live tetrahedra holding a particle flagged 1..level are affected.
    auto ring = [&](char level) {
        const auto vertex = m.vertex;
        const auto alive = s.alive;
        Kokkos::parallel_for(
            "tet_local::ring", Range(exec, 0, s.count), KOKKOS_LAMBDA(int t) {
                if (!alive(t)) return;
                bool any = false;
                for (int k = 0; k < 4; ++k) {
                    const char f = flags(vertex(t, k));
                    any |= f != flag::none && f <= level;
                }
                if (any)
                    for (int k = 0; k < 4; ++k)
                        if (flags(vertex(t, k)) == flag::none) flags(vertex(t, k)) = flag::added;
            });
        settle();
    };
    // The slots of the cells flagged 1..level, in slot order.
    Ints slots;
    auto select = [&](char level) {
        int count = 0;
        Kokkos::parallel_reduce(
            "tet_local::count", Range(exec, 0, n),
            KOKKOS_LAMBDA(int q, int& sum) {
                const char f = flags(order(q));
                sum += f != flag::none && f <= level;
            },
            count);
        slots = Ints(uninitialized("local_slots"), count);
        const auto out = slots;
        Kokkos::parallel_scan(
            "tet_local::select", Range(exec, 0, n), KOKKOS_LAMBDA(int q, int& at, bool last) {
                const char f = flags(order(q));
                if (f == flag::none || f > level) return;
                if (last) out(at) = q;
                ++at;
            });
        return count;
    };
    // Exact cells of the selected slots, listed = every flagged particle (so faces to unflagged ones are kept).
    auto clip = [&](int count) {
        const long long entries = 128LL * count + 4096;
        if (cells.capacity() < entries) cells.reallocate(entries);
        shell_voronoi::construct(exec, grid, cells,
                                 {Kokkos::View<const int*, Memory>(slots), Kokkos::View<const char*, Memory>(flags)},
                                 true);
    };
    // The unflagged neighbours of the selected cells become affected.
    auto neighbours = [&] {
        const auto sel = slots;
        const auto rings = cells.rings;
        const auto start = cells.ring_start;
        const auto owned = cells.owned_count;
        Kokkos::parallel_for(
            "tet_local::neighbours", Range(exec, 0, static_cast<int>(sel.extent(0))), KOKKOS_LAMBDA(int c) {
                const int i = order(sel(c));
                long long at = start(i);
                for (int f = 0; f < owned(i); ++f) {
                    const int j = rings(at);
                    if (flags(j) == flag::none) flags(j) = flag::added;
                    at += 2 + rings(at + 1);
                }
            });
        settle();
    };

    for (int attempt = 0; attempt < 2 && !stats.done; ++attempt) {
        stats = Stats{0, 0, 0, 0, attempt + 1, stats.flips, false};
        const int stuck = Kokkos::min(s.host_counters(tet_repair::counter::stuck), static_cast<int>(s.stuck.extent(0)));
        // Seeds: the particles of the stuck tetrahedra and of the ones across their stuck triangles.
        Kokkos::deep_copy(exec, flags, flag::none);
        {
            const auto vertex = m.vertex, adjacent = m.adjacent;
            const auto alive = s.alive;
            const auto entries = s.stuck;
            Kokkos::parallel_for(
                "tet_local::seeds", Range(exec, 0, stuck), KOKKOS_LAMBDA(int i) {
                    const int t = entries(i, 0), k = entries(i, 1);
                    if (t < 0 || !alive(t)) return;
                    const int u = k >= 0 ? adjacent(t, k) : -1;
                    for (int c = 0; c < 4; ++c) {
                        flags(vertex(t, c)) = flag::seed;
                        if (u >= 0 && alive(u)) flags(vertex(u, c)) = flag::seed;
                    }
                });
        }
        ring(flag::seed);
        stats.seeds = select(flag::seed);
        if (stats.seeds == 0) break;
        clip(stats.seeds);
        neighbours();  // a jumped particle's new neighbourhood
        if (attempt > 0) {  // wider: one more ring of old and of new neighbours
            ring(flag::affected);
            clip(select(flag::affected));
            neighbours();
        }
        const int count = select(flag::affected);
        stats.cells = count;
        clip(count);
        // Every affected cell must be exact and certified.
        {
            const auto status = cells.status;
            const auto sel = slots;
            int bad = 0;
            Kokkos::parallel_reduce(
                "tet_local::status", Range(exec, 0, count),
                KOKKOS_LAMBDA(int c, int& sum) { sum += status(order(sel(c))) != shell_voronoi::ok; }, bad);
            if (bad > 0) break;
        }

        // The tetrahedra the cells imply, numbered by cell, positively oriented.
        const auto rings = cells.rings;
        const auto ring_start = cells.ring_start;
        const auto owned = cells.owned_count;
        const auto sel = slots;
        const Ints first(uninitialized("local_first"), count + 1);
        Kokkos::parallel_scan(
            "tet_local::count_tets", Range(exec, 0, count), KOKKOS_LAMBDA(int c, int& at, bool last) {
                const int i = order(sel(c));
                long long r = ring_start(i);
                int here = 0;
                for (int f = 0; f < owned(i); ++f) {
                    const int j = rings(r), k = rings(r + 1);
                    for (int a = 0; a < k; ++a)
                        here += detail::emits(flags, i, j, rings(r + 2 + a), rings(r + 2 + (a + 1 == k ? 0 : a + 1)));
                    r += 2 + k;
                }
                if (last) first(c) = at;
                at += here;
                if (last && c == count - 1) first(count) = at;
            });
        Kokkos::deep_copy(exec, failures, 0);
        const int created = read(Kokkos::subview(first, count));
        const tet_mesh::Quads<Memory> fresh(uninitialized("local_fresh"), created);
        Kokkos::parallel_for(
            "tet_local::emit", Range(exec, 0, count), KOKKOS_LAMBDA(int c) {
                const int i = order(sel(c));
                long long r = ring_start(i);
                int e = first(c);
                for (int f = 0; f < owned(i); ++f) {
                    const int j = rings(r), k = rings(r + 1);
                    for (int a = 0; a < k; ++a) {
                        const int p = rings(r + 2 + a), q = rings(r + 2 + (a + 1 == k ? 0 : a + 1));
                        if (!detail::emits(flags, i, j, p, q)) continue;
                        int v[4] = {i, j, p, q};
                        const int o = ep::orient(point, v);
                        if (o == 0) Kokkos::atomic_inc(&failures());
                        if (o < 0) Kokkos::kokkos_swap(v[2], v[3]);
                        for (int b = 0; b < 4; ++b) fresh(e, b) = v[b];
                        ++e;
                    }
                    r += 2 + k;
                }
            });

        // The old tetrahedra with an affected corner, and their faces to kept ones (the boundary).
        const auto vertex = m.vertex, adjacent = m.adjacent;
        const auto alive = s.alive;
        auto touches = KOKKOS_LAMBDA(int t) {
            return flags(vertex(t, 0)) || flags(vertex(t, 1)) || flags(vertex(t, 2)) || flags(vertex(t, 3));
        };
        int removed = 0;
        Kokkos::parallel_reduce(
            "tet_local::count_old", Range(exec, 0, s.count),
            KOKKOS_LAMBDA(int t, int& sum) { sum += alive(t) && touches(t); }, removed);
        const Ints old(uninitialized("local_old"), removed), border(uninitialized("local_border"), 4 * removed);
        const Kokkos::View<int, Memory> borders("local_borders");
        Kokkos::parallel_scan(
            "tet_local::old", Range(exec, 0, s.count), KOKKOS_LAMBDA(int t, int& at, bool last) {
                if (!alive(t) || !touches(t)) return;
                if (last) {
                    old(at) = t;
                    for (int k = 0; k < 4; ++k) {
                        const int o = adjacent(t, k);
                        if (o >= 0 && !touches(o)) border(Kokkos::atomic_fetch_add(&borders(), 1)) = 4 * t + k;
                    }
                }
                ++at;
            });
        const int boundary = read(borders);
        stats.removed = removed;
        stats.created = created;

        // Stitch: every face of a new tetrahedron meets exactly one other new face or one kept boundary face.
        detail::Table<Memory> table;
        unsigned size = 1024;
        while (size < 2u * (4u * static_cast<unsigned>(created) + static_cast<unsigned>(boundary))) size *= 2;
        table.mask = size - 1;
        table.used = Ints("local_used", size);
        table.value = Ints(uninitialized("local_value"), size);
        table.key = Kokkos::View<int* [3], Kokkos::LayoutRight, Memory>(uninitialized("local_key"), size);
        const tet_mesh::Quads<Memory> across(uninitialized("local_across"), created);
        Kokkos::parallel_for(
            "tet_local::insert", Range(exec, 0, 4 * created + boundary), KOKKOS_LAMBDA(int i) {
                int key[3];
                if (i < 4 * created) {
                    const int v[4] = {fresh(i / 4, 0), fresh(i / 4, 1), fresh(i / 4, 2), fresh(i / 4, 3)};
                    tet_mesh::sorted_face(v, i % 4, key);
                    table.insert(key, i);
                } else {
                    const int b = border(i - 4 * created), t = b / 4;
                    const int v[4] = {vertex(t, 0), vertex(t, 1), vertex(t, 2), vertex(t, 3)};
                    tet_mesh::sorted_face(v, b % 4, key);
                    table.insert(key, -1 - adjacent(t, b % 4));
                }
            });
        Kokkos::parallel_for(
            "tet_local::match", Range(exec, 0, 4 * created + boundary), KOKKOS_LAMBDA(int i) {
                int key[3], self, other = -1;
                if (i < 4 * created) {
                    const int v[4] = {fresh(i / 4, 0), fresh(i / 4, 1), fresh(i / 4, 2), fresh(i / 4, 3)};
                    tet_mesh::sorted_face(v, i % 4, key);
                    self = i;
                } else {
                    const int b = border(i - 4 * created), t = b / 4;
                    const int v[4] = {vertex(t, 0), vertex(t, 1), vertex(t, 2), vertex(t, 3)};
                    tet_mesh::sorted_face(v, b % 4, key);
                    self = -1 - adjacent(t, b % 4);
                }
                const int found = table.others(key, self, other);
                if (found != 1 || (i >= 4 * created && other < 0)) Kokkos::atomic_inc(&failures());
                if (i < 4 * created) across(i / 4, i % 4) = other;
            });
        if (read(failures) > 0) continue;  // the region does not close: widen

        // Slots: the old ones first, then free ones, then new; old ones left over are freed.
        const int extra = Kokkos::max(0, created - removed), dead = Kokkos::max(0, removed - created);
        const int free_top = s.free_count, pops = Kokkos::min(extra, free_top), growth = extra - pops;
        if (s.count + growth > static_cast<int>(s.alive.extent(0)))
            tet_repair::reserve(m, s, s.count + growth + (s.count + growth) / 64 + 1024);
        const int base = s.count;
        const auto free_slots = s.free_slots;
        auto slot = KOKKOS_LAMBDA(int e) {
            if (e < removed) return old(e);
            const int x = e - removed;
            return x < pops ? free_slots(free_top - 1 - x) : base + x - pops;
        };
        Kokkos::View<int* [11], Kokkos::LayoutRight, Memory> rows(uninitialized("local_rows"), created + dead);
        const auto old_vertex = m.vertex;  // after reserve
        Kokkos::parallel_for(
            "tet_local::rows", Range(exec, 0, created + dead), KOKKOS_LAMBDA(int e) {
                if (e >= created) {  // freed
                    const int t = old(e);
                    rows(e, 0) = t;
                    rows(e, 1) = 0;
                    for (int k = 0; k < 4; ++k) {
                        rows(e, 2 + k) = old_vertex(t, k);
                        rows(e, 6 + k) = -1;
                    }
                    rows(e, 10) = 0;
                    return;
                }
                int open = 0;
                rows(e, 0) = slot(e);
                rows(e, 1) = 1;
                for (int k = 0; k < 4; ++k) {
                    const int a = across(e, k);
                    rows(e, 2 + k) = fresh(e, k);
                    rows(e, 6 + k) = a >= 0 ? slot(a / 4) : -1 - a;
                    open |= (a < 0) << k;
                }
                rows(e, 10) = open;
            });
        s.count += growth;
        s.free_count = free_top - pops;
        if (dead > 0) {
            if (s.free_count + dead > static_cast<int>(s.free_slots.extent(0)))
                Kokkos::resize(s.free_slots, s.free_count + dead + (s.free_count + dead) / 4);
            const auto to_free = s.free_slots;
            const int at = s.free_count;
            Kokkos::parallel_for(
                "tet_local::free", Range(exec, 0, dead), KOKKOS_LAMBDA(int d) { to_free(at + d) = old(created + d); });
            s.free_count += dead;
        }
        tet_flips::write_region(exec, m, s, rows, stuck);
        const tet_repair::Stats device = tet_repair::repair(exec, m, s);
        stats.flips += device.flips23 + device.flips32;
        stats.done = device.stuck == 0 && !device.unfinished;
        if (device.unfinished) break;
    }
    return stats;
}

}  // namespace tet_local
