#pragma once
// Measurement build only (-DVORONOI_MEASURE): cost per cell of the warp clipping, the shuffle test of clipping order,
// and what a moving step changes. Used by voronoi_bench (--measure); results in docs/measure_2026-11-10.md.

#include "support.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace measure_bench {

using namespace support;
namespace sm = shell_voronoi::measure;

// Name of each column of Cells::measure, in shell_voronoi::measure order.
inline const char* column_name(int k) {
    static const char* names[] = {"tested",        "taken",           "clipped",     "cuts",           "vertices_end",
                                  "vertices_peak", "walk_cells",      "walk_open",   "tested_outside", "taken_outside",
                                  "open_outside",  "in_open_outside", "warp_status", "cycles",         "batches"};
    return names[k];
}

// Mean, median, p90, p99 and max of v, printed on one line under `name`.
inline void distribution(const std::string& name, std::vector<long long> v) {
    if (v.empty()) return;
    std::sort(v.begin(), v.end());
    const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    auto at = [&](double q) { return v[std::min(v.size() - 1, static_cast<std::size_t>(q * v.size()))]; };
    std::printf("measure: %-16s mean=%.3f p50=%lld p90=%lld p99=%lld p999=%lld max=%lld\n", name.c_str(), mean, at(0.5),
                at(0.9), at(0.99), at(0.999), v.back());
}

// A .npy file of int32 rows x columns. Used by: clipping (--dump).
inline void write_npy(const std::string& path, const std::vector<int>& data, std::size_t rows, std::size_t columns) {
    std::string header = "{'descr': '<i4', 'fortran_order': False, 'shape': (" + std::to_string(rows) + ", " +
                         std::to_string(columns) + "), }";
    while ((10 + header.size() + 1) % 64 != 0) header += ' ';
    header += '\n';
    std::ofstream out(path, std::ios::binary);
    const unsigned short length = static_cast<unsigned short>(header.size());
    out.write("\x93NUMPY\x01\x00", 8);
    out.write(reinterpret_cast<const char*>(&length), 2);
    out << header;
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(int)));
}

// Replays the sampled candidate lists on the host: kernel order, sorted nearest first, and `orders` random orders.
// Per order: clips (candidates within 2R at their turn), cuts, peak vertices, vertex tests (live vertices per clip).
inline void shuffle(const shell_voronoi::Cells<Memory>& cells, const Box& box, int orders) {
    const auto count = to_host(cells.sample_count);
    const auto sample = to_host(cells.sample);
    const int cap = cells.sample_cap;
    using Cell = shell_voronoi::detail::RoundedCell<shell_voronoi::big_planes, shell_voronoi::big_vertices>;
    const double h = shell_voronoi::detail::start_half_width(box);
    struct Work {
        long long clips = 0, cuts = 0, peak = 0, tests = 0;
    };
    auto replay = [&](const std::vector<shell_voronoi::Vec3>& d, const std::vector<int>& order, Cell& cell) {
        shell_voronoi::detail::start_cube(cell, h, 1.0, 1 << 30);
        Work w;
        for (int i : order) {
            if (!(dot(d[i], d[i]) < 4 * cell.radius2)) continue;
            const int planes = cell.planes, live = cell.vertices - cell.free_count;
            w.tests += live;
            if (shell_voronoi::detail::clip(cell, {}, d[i], i, -1) != shell_voronoi::ok) return Work{-1, 0, 0, 0};
            ++w.clips;
            w.cuts += cell.planes > planes;
            w.peak = std::max<long long>(w.peak, cell.vertices - cell.free_count);
        }
        return w;
    };
    auto cell = std::make_unique<Cell>();
    std::mt19937_64 random(12345);
    std::vector<double> spread_clips, spread_cuts, spread_peak, spread_tests;
    double kernel[4] = {}, sorted[4] = {}, mean[4] = {}, best[4] = {}, worst[4] = {};
    int used = 0, skipped = 0;
    for (std::size_t s = 0; s < count.extent(0); ++s) {
        const int k = count(s);
        if (k <= 0 || k > cap) {
            skipped += k > cap;
            continue;
        }
        std::vector<shell_voronoi::Vec3> d(k);
        for (int i = 0; i < k; ++i) {
            const std::size_t at = s * cap + i;
            d[i] = {sample(at, 0), sample(at, 1), sample(at, 2)};
        }
        std::vector<int> order(k);
        std::iota(order.begin(), order.end(), 0);
        const Work a = replay(d, order, *cell);
        std::sort(order.begin(), order.end(), [&](int x, int y) { return dot(d[x], d[x]) < dot(d[y], d[y]); });
        const Work b = replay(d, order, *cell);
        if (a.clips < 0 || b.clips < 0) continue;
        double lo[4] = {1e300, 1e300, 1e300, 1e300}, hi[4] = {}, sum[4] = {};
        bool fine = true;
        for (int r = 0; r < orders && fine; ++r) {
            std::shuffle(order.begin(), order.end(), random);
            const Work w = replay(d, order, *cell);
            if (w.clips < 0) fine = false;
            const double v[4] = {double(w.clips), double(w.cuts), double(w.peak), double(w.tests)};
            for (int q = 0; q < 4; ++q) {
                lo[q] = std::min(lo[q], v[q]);
                hi[q] = std::max(hi[q], v[q]);
                sum[q] += v[q];
            }
        }
        if (!fine) continue;
        ++used;
        const double va[4] = {double(a.clips), double(a.cuts), double(a.peak), double(a.tests)};
        const double vb[4] = {double(b.clips), double(b.cuts), double(b.peak), double(b.tests)};
        for (int q = 0; q < 4; ++q) {
            kernel[q] += va[q];
            sorted[q] += vb[q];
            mean[q] += sum[q] / orders;
            best[q] += std::min(lo[q], std::min(va[q], vb[q]));
            worst[q] += hi[q];
        }
        spread_clips.push_back((hi[0] - lo[0]) / (sum[0] / orders));
        spread_cuts.push_back((hi[1] - lo[1]) / (sum[1] / orders));
        spread_peak.push_back((hi[2] - lo[2]) / (sum[2] / orders));
        spread_tests.push_back((hi[3] - lo[3]) / (sum[3] / orders));
    }
    const char* names[4] = {"clips", "cuts", "peak_vertices", "vertex_tests"};
    std::printf("shuffle: cells=%d skipped_over_cap=%d orders=%d\n", used, skipped, orders);
    std::vector<double>* spreads[4] = {&spread_clips, &spread_cuts, &spread_peak, &spread_tests};
    for (int q = 0; q < 4; ++q) {
        auto& v = *spreads[q];
        std::sort(v.begin(), v.end());
        const auto at = [&](double f) {
            return v.empty() ? 0.0 : v[std::min(v.size() - 1, std::size_t(f * v.size()))];
        };
        std::printf(
            "shuffle: %-13s per_cell kernel=%.2f sorted=%.2f random_mean=%.2f best=%.2f worst=%.2f "
            "spread_p50=%.3f spread_p90=%.3f\n",
            names[q], kernel[q] / used, sorted[q] / used, mean[q] / used, best[q] / used, worst[q] / used, at(0.5),
            at(0.9));
    }
}

// Runs the warp clipping with counters (and once more with the vertex-ball counts), prints distributions, the share
// of cycles in the tail, the cells sent to the thread kernel, and the shuffle test on every `every`-th slot.
inline void clipping(const Exec& exec, const shell_voronoi::Grid<Memory>& grid, const Box& box, int every,
                     const std::string& dump) {
    const int n = grid.n;
    shell_voronoi::Cells<Memory> cells(n, true);
    cells.measure = Kokkos::View<long long**, Memory>("measure", n, sm::count);
    cells.sample_every = every;
    cells.sample_cap = 1024;
    cells.sample = Kokkos::View<double* [3], Memory>("measure_sample", std::size_t(n / every + 1) * cells.sample_cap);
    cells.sample_count = Kokkos::View<int*, Memory>("measure_sample_count", n / every + 1);
    shell_voronoi::construct_warp(exec, grid, cells);
    const auto plain = to_host(cells.measure);
    cells.balls = true;
    shell_voronoi::construct_warp(exec, grid, cells);
    const auto balls = to_host(cells.measure);
    std::vector<long long> column(n);
    for (int k = 0; k < sm::count; ++k) {
        const bool ball_column = k >= sm::tested_outside && k <= sm::in_open_outside;
        for (int i = 0; i < n; ++i) column[i] = (ball_column ? balls : plain)(i, k);
        distribution(column_name(k), column);
    }
    long long sent[shell_voronoi::status_count] = {};
    for (int i = 0; i < n; ++i) ++sent[plain(i, sm::warp_status)];
    std::printf(
        "measure: warp_status ok=%lld uncertified=%lld plane_overflow=%lld vertex_overflow=%lld "
        "face_overflow=%lld degenerate=%lld\n",
        sent[0], sent[1], sent[2], sent[3], sent[4], sent[5]);
    // The share of all warp cycles spent in the slowest cells, and in the cells with the most clips.
    std::vector<long long> cycles(n);
    for (int i = 0; i < n; ++i) cycles[i] = plain(i, sm::cycles);
    std::vector<long long> sorted = cycles;
    std::sort(sorted.rbegin(), sorted.rend());
    const double total = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    auto top = [&](double f) {
        return std::accumulate(sorted.begin(), sorted.begin() + std::max(1, int(f * n)), 0.0) / total;
    };
    std::printf("measure: cycle_share slowest_1%%=%.4f slowest_10%%=%.4f slowest_0.1%%=%.4f\n", top(0.01), top(0.1),
                top(0.001));
    std::vector<int> by_clips(n);
    std::iota(by_clips.begin(), by_clips.end(), 0);
    std::sort(by_clips.begin(), by_clips.end(),
              [&](int a, int b) { return plain(a, sm::clipped) > plain(b, sm::clipped); });
    double most = 0;
    for (int i = 0; i < std::max(1, n / 100); ++i) most += cycles[by_clips[i]];
    std::printf("measure: cycle_share most_clips_1%%=%.4f cycles_per_clip=%.1f cycles_per_test=%.1f\n", most / total,
                total / std::accumulate(by_clips.begin(), by_clips.end(), 0.0,
                                        [&](double s, int i) { return s + plain(i, sm::clipped); }),
                total / std::accumulate(by_clips.begin(), by_clips.end(), 0.0,
                                        [&](double s, int i) { return s + plain(i, sm::tested); }));
    double sums[sm::count] = {};
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < sm::count; ++k)
            sums[k] += (k >= sm::tested_outside && k <= sm::in_open_outside ? balls : plain)(i, k);
    std::printf(
        "measure: ratios taken/tested=%.4f clipped/taken=%.4f cuts/clipped=%.4f tested_outside/tested=%.4f "
        "taken_outside/taken=%.4f open/walk=%.4f open_outside/open=%.4f in_open_outside/tested=%.4f\n",
        sums[sm::taken] / sums[sm::tested], sums[sm::clipped] / sums[sm::taken], sums[sm::cuts] / sums[sm::clipped],
        sums[sm::tested_outside] / sums[sm::tested], sums[sm::taken_outside] / sums[sm::taken],
        sums[sm::walk_open] / sums[sm::walk_cells], sums[sm::open_outside] / sums[sm::walk_open],
        sums[sm::in_open_outside] / sums[sm::tested]);
    if (!dump.empty()) {
        // Columns 0-14 as in shell_voronoi::measure, the vertex-ball columns from the second run.
        std::vector<int> data(std::size_t(n) * sm::count);
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < sm::count; ++k) {
                const bool ball_column = k >= sm::tested_outside && k <= sm::in_open_outside;
                data[std::size_t(i) * sm::count + k] = static_cast<int>((ball_column ? balls : plain)(i, k));
            }
        write_npy(dump, data, n, sm::count);
    }
    shuffle(cells, box, 64);
}

// Neighbour sets of every cell from the per-cell face lists (after update_cells).
inline std::vector<std::vector<int>> neighbour_sets(const Mesh& mesh) {
    const int n = mesh.particles();
    const auto start = to_host(mesh.cell_face_start());
    const auto face = to_host(mesh.cell_face());
    const auto owner = to_host(mesh.owner());
    const auto neighbour = to_host(mesh.neighbour());
    std::vector<std::vector<int>> sets(n);
    for (int i = 0; i < n; ++i) {
        for (int a = start(i); a < start(i + 1); ++a)
            sets[i].push_back(owner(face(a)) == i ? neighbour(face(a)) : owner(face(a)));
        std::sort(sets[i].begin(), sets[i].end());
    }
    return sets;
}

// One step's changes (idea E) and new neighbours by ring distance in the last step's graph (idea S2).
inline void step(const Mesh& mesh, std::vector<std::vector<int>>& before, int s) {
    const auto& state = mesh.repair_state();
    const auto alive = to_host(state.alive);
    long long live = 0;
    for (int t = 0; t < state.count; ++t) live += alive(t) != 0;
    auto now = neighbour_sets(mesh);
    const int n = mesh.particles();
    auto has = [](const std::vector<int>& v, int x) { return std::binary_search(v.begin(), v.end(), x); };
    long long added = 0, lost = 0, ring2 = 0, ring3 = 0, cells_changed = 0;
    if (!before.empty())
        for (int i = 0; i < n; ++i) {
            bool changed = false;
            for (int j : now[i]) {
                if (has(before[i], j)) continue;
                changed = true;
                ++added;
                bool two = false, three = false;
                for (int k : before[i]) two = two || has(before[k], j);
                for (int k : before[i])
                    for (int m : before[k]) three = three || two || has(before[j], m);
                ring2 += two;
                ring3 += three;
            }
            for (int j : before[i]) {
                if (has(now[i], j)) continue;
                changed = true;
                ++lost;
            }
            cells_changed += changed;
        }
    std::printf(
        "step_measure: step=%d live_tets=%lld screened=%d failing_checked=%lld created=%d faces=%d "
        "faces_rewalked=%d cells_changed=%lld new_neighbours=%lld lost=%lld in_2ring=%lld in_3ring=%lld\n",
        s, live, state.screened, state.checked, state.touched_entries, mesh.faces(), mesh.face_rings().dirty_count,
        cells_changed, added, lost, ring2, ring3);
    before = std::move(now);
}

}  // namespace measure_bench
