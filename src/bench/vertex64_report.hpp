#pragma once

// Diagnostic build only (-DVORONOI_VERTEX64): the report of issue #64 for `voronoi_bench rebuild ... --vertex64`.
// After a warp rebuild with the sinks installed: the global counters, the cells by bit, then the flagged cells (and
// a sample of unflagged ones, and the cells whose faces do not pair) built again exactly (ExactCell) and compared:
// volume, centroid, radius, and the worst cut's vertex against the exact meeting point of its three planes.

#include "support.hpp"
#include "vertex64.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace vertex64_report {

using Memory = support::Memory;
using Exec = Kokkos::DefaultExecutionSpace;

struct Views {
    Kokkos::View<unsigned long long*, Memory> counter{"v64_counter", vertex64::counters};
    Kokkos::View<unsigned*, Memory> flag;
    Kokkos::View<float*, Memory> worst;
    Kokkos::View<double*, Memory> radius, exact_radius;
    Kokkos::View<int*, Memory> worst_slot;
    Kokkos::View<double*, Memory> worst_vertex;
    explicit Views(int n)
        : flag("v64_flag", n),
          worst("v64_worst", n),
          radius("v64_radius", n),
          exact_radius("v64_exact_radius", n),
          worst_slot("v64_worst_slot", 4 * n),
          worst_vertex("v64_worst_vertex", 3 * n) {}
    vertex64::Sinks sinks(int n) const {
        vertex64::Sinks s;
        s.counter = counter.data();
        s.cap = n;
        s.flag = flag.data();
        s.worst = worst.data();
        s.radius = radius.data();
        s.exact_radius = exact_radius.data();
        s.worst_slot = worst_slot.data();
        s.worst_vertex = worst_vertex.data();
        return s;
    }
    void install(int n) const {
        vertex64::install(sinks(n), !std::is_same_v<Memory, Kokkos::HostSpace>);
    }
    void clear() const {
        Kokkos::deep_copy(counter, 0ull);
        Kokkos::deep_copy(flag, 0u);
        Kokkos::deep_copy(worst, -1.0f);
    }
};

inline void print_counters(const Views& v, const char* label) {
    const auto c = support::to_host(v.counter);
    std::printf("v64 %s: new_vertices=%llu strict=%llu kept_outside=%llu beyond_bound=%llu err>1e-9R=%llu "
                "err>1e-6R=%llu err>1e-3R=%llu\n",
                label, c(0), c(1), c(2), c(3), c(4), c(5), c(6));
}

// The report; `dump`: a CSV of the compared cells, or "".
template <class Grid, class Cells>
void report(const Exec& exec, const Grid& grid, const Views& v, const Cells& cells, const std::string& dump) {
    const int n = grid.n;
    print_counters(v, "warp rebuild");
    const auto flag = support::to_host(v.flag);
    const auto worst = support::to_host(v.worst);
    const auto radius = support::to_host(v.radius);
    const auto wslot = support::to_host(v.worst_slot);
    const auto wvertex = support::to_host(v.worst_vertex);
    const auto status = support::to_host(cells.status);
    const auto volume = support::to_host(cells.volume);
    const auto centroid = support::to_host(cells.centroid);
    const auto order = support::to_host(grid.order);
    const auto positions = support::to_host(grid.positions);
    const auto& box = grid.box;

    // Cells by bit (the last rounded build of each cell; built_exactly: finished by exact_where_failed).
    const unsigned bits[] = {vertex64::strict, vertex64::kept_outside, vertex64::beyond_bound, vertex64::err9,
                             vertex64::err6, vertex64::err3, vertex64::built_exactly};
    const char* names[] = {"strict", "kept_outside", "beyond_bound", "err9", "err6", "err3", "built_exactly"};
    long long by_bit[7] = {};
    for (int i = 0; i < n; ++i)
        for (int b = 0; b < 7; ++b) by_bit[b] += (flag(i) & bits[b]) != 0;
    std::printf("v64 cells: n=%d", n);
    for (int b = 0; b < 7; ++b) std::printf(" %s=%lld", names[b], by_bit[b]);
    std::printf("\n");

    // Face pairing (#58): a cell's face_count against its owned faces plus the faces others own toward it.
    std::vector<int> toward(n, 0);
    const auto face_start = support::to_host(cells.face_start);
    const auto owned_count = support::to_host(cells.owned_count);
    const auto face_count = support::to_host(cells.face_count);
    const auto neighbour = support::to_host(cells.neighbour);
    long long faces = 0, owned = 0;
    for (int i = 0; i < n; ++i) {
        faces += face_count(i);
        owned += owned_count(i);
        for (int k = 0; k < owned_count(i); ++k) ++toward[neighbour(face_start(i) + k)];
    }
    std::vector<int> unpaired;
    for (int i = 0; i < n; ++i)
        if (status(i) == shell_voronoi::ok && face_count(i) != owned_count(i) + toward[i]) unpaired.push_back(i);
    long long unpaired_flagged = 0, unpaired_err9 = 0;
    for (const int i : unpaired) {
        unpaired_flagged += (flag(i) & (vertex64::kept_outside | vertex64::err6 | vertex64::strict)) != 0;
        unpaired_err9 += (flag(i) & vertex64::err9) != 0;
    }
    std::printf("v64 pairing: faces=%lld owned=%lld (2*owned %s) unpaired_cells=%zu of_them_kept_outside_err6_strict=%lld "
                "of_them_err9=%lld\n",
                faces, owned, faces == 2 * owned ? "yes" : "NO", unpaired.size(), unpaired_flagged, unpaired_err9);

    // The cells to compare: flagged (err9 or worse, kept_outside, strict), the unpaired, and 4096 unflagged.
    const unsigned flagged_bits = vertex64::strict | vertex64::kept_outside | vertex64::err9;
    std::vector<int> ids, kind;  // kind: 0 flagged, 1 unpaired, 2 sample
    std::vector<char> taken(n, 0);
    for (int i = 0; i < n; ++i)
        if ((flag(i) & flagged_bits) && !(flag(i) & vertex64::built_exactly) && status(i) == shell_voronoi::ok) {
            ids.push_back(i), kind.push_back(0), taken[i] = 1;
        }
    if (ids.size() > 400000) {  // keep the worst 400k
        std::sort(ids.begin(), ids.end(), [&](int a, int b) { return worst(a) > worst(b); });
        for (std::size_t k = 400000; k < ids.size(); ++k) taken[ids[k]] = 0;
        ids.resize(400000), kind.resize(400000);
    }
    for (const int i : unpaired)
        if (!taken[i]) ids.push_back(i), kind.push_back(1), taken[i] = 1;
    std::mt19937_64 rng(64);
    for (int k = 0, tries = 0; k < 4096 && tries < 100000; ++tries) {
        const int i = static_cast<int>(rng() % static_cast<unsigned long long>(n));
        if (taken[i] || status(i) != shell_voronoi::ok || (flag(i) & vertex64::built_exactly)) continue;
        ids.push_back(i), kind.push_back(2), taken[i] = 1, ++k;
    }
    // Slots of those cells (particle id -> slot).
    std::vector<int> slot_of(n);
    for (int s = 0; s < n; ++s) slot_of[order(s)] = s;
    Kokkos::View<int*, Memory> slots("v64_slots", ids.size());
    {
        auto h = Kokkos::create_mirror_view(slots);
        for (std::size_t k = 0; k < ids.size(); ++k) h(k) = slot_of[ids[k]];
        Kokkos::deep_copy(slots, h);
    }
    shell_voronoi::Cells<Memory> exact(n, false, 10, static_cast<int>(ids.size()));
    Kokkos::Timer timer;
    shell_voronoi::construct(exec, grid, exact, {slots, {}}, true);
    exec.fence();
    const double exact_ms = timer.seconds() * 1e3;
    const auto estatus = support::to_host(exact.status);
    const auto evolume = support::to_host(exact.volume);
    const auto ecentroid = support::to_host(exact.centroid);
    const auto eradius = support::to_host(v.exact_radius);

    // The exact meeting point of the worst cut's three planes, in box units relative to the generator.
    auto snapped = [&](int s, int a) { return static_cast<exact_predicates::i64>(exact_predicates::snap(positions(s, a), box, a)); };
    auto exact_vertex = [&](int t, const int* planes, double* x) {
        exact_predicates::Offset off[3];
        for (int k = 0; k < 3; ++k) {
            if (planes[k] < 0) return false;  // a box plane
            for (int a = 0; a < 3; ++a)
                off[k][a] = exact_predicates::delta(snapped(t, a), snapped(planes[k], a), box.period[a]);
        }
        exact_predicates::i128 num[3];
        const exact_predicates::i128 o = exact_predicates::circumcentre(off[0], off[1], off[2], num);
        if (o == 0) return false;
        for (int a = 0; a < 3; ++a)
            x[a] = static_cast<double>(static_cast<long double>(num[a]) / (2.0L * static_cast<long double>(o)) *
                                       static_cast<long double>(box.step));
        return true;
    };

    struct Worst {
        double volume = 0, centroid = 0, radius = 0, vertex = 0;
        long long count = 0, exact_failed = 0, box_plane = 0, past_1e9 = 0, past_1e6 = 0;
    } w[3];
    std::FILE* out = dump.empty() ? nullptr : std::fopen(dump.c_str(), "w");
    if (out)
        std::fprintf(out, "id,kind,flag,worst_bound,volume,exact_volume,dvolume_rel,dcentroid_rel,radius,exact_radius,"
                          "dradius_rel,dvertex_rel\n");
    for (std::size_t k = 0; k < ids.size(); ++k) {
        const int i = ids[k];
        Worst& r = w[kind[k]];
        ++r.count;
        if (estatus(i) != shell_voronoi::ok) {
            ++r.exact_failed;
            continue;
        }
        const double size = std::cbrt(evolume(i));
        const double dv = std::abs(volume(i) - evolume(i)) / evolume(i);
        double dc = 0;
        for (int a = 0; a < 3; ++a) dc = std::max(dc, std::abs(centroid(i, a) - ecentroid(i, a)) / size);
        const double dr = std::abs(radius(i) - eradius(i)) / eradius(i);
        double dx = -1;
        if (kind[k] == 0) {
            const int planes[3] = {wslot(4 * i + 1), wslot(4 * i + 2), wslot(4 * i + 3)};
            double x[3];
            if (exact_vertex(wslot(4 * i), planes, x)) {
                dx = 0;
                for (int a = 0; a < 3; ++a) dx = std::max(dx, std::abs(wvertex(3 * i + a) - x[a]) / eradius(i));
            } else {
                ++r.box_plane;
            }
        }
        r.volume = std::max(r.volume, dv);
        r.centroid = std::max(r.centroid, dc);
        r.radius = std::max(r.radius, dr);
        r.vertex = std::max(r.vertex, dx);
        const double most = std::max(std::max(dv, dc), std::max(dr, dx));
        r.past_1e9 += most > 1e-9;
        r.past_1e6 += most > 1e-6;
        if (out)
            std::fprintf(out, "%d,%d,%u,%.3e,%.17g,%.17g,%.3e,%.3e,%.17g,%.17g,%.3e,%.3e\n", i, kind[k], flag(i),
                         double(worst(i)), volume(i), evolume(i), dv, dc, radius(i), eradius(i), dr, dx);
    }
    if (out) std::fclose(out);
    const char* kinds[] = {"flagged", "unpaired", "sample"};
    for (int c = 0; c < 3; ++c)
        std::printf("v64 exact %s: cells=%lld exact_not_ok=%lld max_dvolume_rel=%.3e max_dcentroid_per_size=%.3e "
                    "max_dradius_rel=%.3e max_dvertex_per_R=%.3e (box-plane vertices skipped %lld) "
                    "cells_past_1e-9=%lld past_1e-6=%lld\n",
                    kinds[c], w[c].count, w[c].exact_failed, w[c].volume, w[c].centroid, w[c].radius, w[c].vertex,
                    w[c].box_plane, w[c].past_1e9, w[c].past_1e6);
    std::printf("v64 exact rebuild of %zu cells: %.1f ms\n", ids.size(), exact_ms);
}

}  // namespace vertex64_report
