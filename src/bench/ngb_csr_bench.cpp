// The adaptive Morton cell list (neighbor_search::CellList) as GIZMO's neighbour search (MI300A study, not part of the
// library): per-particle radii h_i and a CSR neighbour list per particle, one-way (r_ij < h_i) and symmetric
// (r_ij < max(h_i, h_j)), self excluded; periodic unit box, slots in Morton order. Arguments and timings: main().
#include "support.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace support;

namespace {

using Grid = shell_voronoi::Grid<Memory>;
using Ints = Kokkos::View<int*, Memory>;
using Doubles = Kokkos::View<double*, Memory>;

std::string text_arg(int argc, char** argv, const std::string& key, const std::string& fallback) {
    for (int a = 1; a < argc; ++a) {
        const std::string s = argv[a];
        if (s.rfind(key + "=", 0) == 0) return s.substr(key.size() + 1);
    }
    return fallback;
}

// Calls f(slot j, d2) for every slot within h of slot q's position (q itself included), walking the grid cells at the
// level where cells are about h / per_h wide, skipping cells whose box is farther than h.
template <class F>
KOKKOS_INLINE_FUNCTION void ball(const Grid& g, int q, double h, int per_h, int depth, const F& f,
                                 int* tested = nullptr) {
    const double px = g.positions(q, 0), py = g.positions(q, 1), pz = g.positions(q, 2);
    const double h2 = h * h;
    // The finest level whose cells are at least h / per_h wide.
    int level = static_cast<int>(Kokkos::floor(Kokkos::log2(per_h / h)));
    level = Kokkos::min(Kokkos::max(level, 2), depth);
    const int side = 1 << level;
    const double w = 1.0 / side;
    const int reach = static_cast<int>(Kokkos::ceil(h / w));
    const int hx = neighbor_search::grid_coordinate(px, 1.0, side);
    const int hy = neighbor_search::grid_coordinate(py, 1.0, side);
    const int hz = neighbor_search::grid_coordinate(pz, 1.0, side);
    // Offsets within the home cell, for the distance from q to a cell's box.
    const double fx = px - hx * w, fy = py - hy * w, fz = pz - hz * w;
    for (int dx = -reach; dx <= reach; ++dx) {
        const double gx = dx > 0 ? dx * w - fx : dx < 0 ? fx - (dx + 1) * w : 0;
        for (int dy = -reach; dy <= reach; ++dy) {
            const double gy = dy > 0 ? dy * w - fy : dy < 0 ? fy - (dy + 1) * w : 0;
            for (int dz = -reach; dz <= reach; ++dz) {
                const double gz = dz > 0 ? dz * w - fz : dz < 0 ? fz - (dz + 1) * w : 0;
                if (gx * gx + gy * gy + gz * gz >= h2) continue;
                const int x = (hx + dx + side) & (side - 1), y = (hy + dy + side) & (side - 1),
                          z = (hz + dz + side) & (side - 1);
                int begin, end;
                g.template range<true>(level, x, y, z, begin, end);
                if (tested) *tested += end - begin;
                for (int s = begin; s < end; ++s) {
                    double ax = g.positions(s, 0) - px, ay = g.positions(s, 1) - py, az = g.positions(s, 2) - pz;
                    ax -= Kokkos::round(ax), ay -= Kokkos::round(ay), az -= Kokkos::round(az);
                    const double d2 = ax * ax + ay * ay + az * az;
                    if (d2 < h2) f(s, d2);
                }
            }
        }
    }
}

template <class F>
double timed(int repeats, const F& f) {
    return median_ms(repeats, f);
}

double percentile(std::vector<int> v, double p) {
    std::sort(v.begin(), v.end());
    return v[static_cast<std::size_t>(p * (v.size() - 1))];
}

}  // namespace

// Timed separately: the cell list (Morton sort + offsets + levels), and the list: count pass, scan, fill pass; for the
// symmetric list also the transpose of the one-way list and its merge (j joins i's row when r_ij < h_j and r_ij >= h_i,
// so nothing is listed twice).
// args: ngb_csr_bench <n> <kind|file.pts> [--ngb=32] [--h=file.h] [--repeats=5] [--verify=256] [--per-h=1|2]
//   file.pts: uint64 n, uint64 frames, n x 3 doubles in [0, 1); file.h: n doubles (radii in the same units).
//   --per-h: grid cells per h along an axis at the walk level (1: cells >= h wide, reach 1; 2: >= h/2, reach 2).
int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    if (argc < 3) {
        std::cerr << "usage: ngb_csr_bench <n> <kind|file.pts> [--ngb=32] [--h=file] [--repeats=5] [--verify=256] "
                     "[--per-h=1|2]\n";
        return 2;
    }
    const int requested = std::stoi(argv[1]);
    const std::string kind = argv[2];
    const int ngb = std::stoi(text_arg(argc, argv, "--ngb", "32"));
    const int repeats = std::stoi(text_arg(argc, argv, "--repeats", "5"));
    const int sampled = std::stoi(text_arg(argc, argv, "--verify", "256"));
    const int per_h = std::stoi(text_arg(argc, argv, "--per-h", "1"));
    const std::string h_file = text_arg(argc, argv, "--h", "");
    const Exec exec;
    const Box box;

    Points points;
    if (kind.size() > 4 && kind.compare(kind.size() - 4, 4, ".pts") == 0) {
        std::ifstream in(kind, std::ios::binary);
        std::uint64_t head[2] = {0, 0};
        in.read(reinterpret_cast<char*>(head), sizeof head);
        std::vector<double> xyz(3 * head[0]);
        in.read(reinterpret_cast<char*>(xyz.data()), static_cast<std::streamsize>(xyz.size() * sizeof(double)));
        if (!in) throw std::runtime_error("cannot read " + kind);
        points.resize(head[0]);
        for (std::size_t i = 0; i < points.size(); ++i) points[i] = {xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]};
    } else {
        points = make_points(kind, requested, box);
    }
    const Positions x = to_device(points);
    const int n = static_cast<int>(x.extent(0));

    // The cell list: Morton sort, base-cell offsets, per-slot levels, slot positions.
    neighbor_search::CellList<Exec> list;
    neighbor_search::configure(list, n, box, std::cbrt(1.0 / n));
    const double build_ms = timed(repeats, [&] { neighbor_search::build(exec, list, x); });
    const Grid g = shell_voronoi::make_grid(list);
    const int depth = g.levels + neighbor_search::fine_levels;
    std::cout << "# backend=" << Exec::name() << " concurrency=" << exec.concurrency() << " n=" << n
              << " input=" << kind << " ngb=" << ngb << " per_h=" << per_h << " base_levels=" << list.levels << "\n";
    std::cout << "cell_list: build_ms=" << build_ms << " ns_per_particle=" << build_ms * 1e6 / n << "\n";

    // Radii per slot: given, or from the density, GIZMO-like: start from the slot's cell-list level, then three
    // updates h <- h (ngb / count)^(1/3) (untimed setup).
    Doubles h("h", n);
    Ints count("count", n), offsets("offsets", n + 1);
    const auto order = list.order;
    if (!h_file.empty()) {
        std::vector<double> given(n);
        std::ifstream in(h_file, std::ios::binary);
        in.read(reinterpret_cast<char*>(given.data()), static_cast<std::streamsize>(n * sizeof(double)));
        if (!in) throw std::runtime_error("cannot read " + h_file);
        const auto order_h = to_host(order);
        auto hh = Kokkos::create_mirror_view(h);
        for (int q = 0; q < n; ++q) hh(q) = given[order_h(q)];
        Kokkos::deep_copy(h, hh);
    } else {
        const double c = std::cbrt(3.0 * ngb / (4.0 * M_PI));
        const auto level = g.level;
        Kokkos::parallel_for(
            "ngb::h0", Kokkos::RangePolicy<Exec>(exec, 0, n),
            KOKKOS_LAMBDA(int q) { h(q) = c / double(1 << level(q)); });
        for (int it = 0; it < 4; ++it)
            Kokkos::parallel_for(
                "ngb::h_update", Kokkos::RangePolicy<Exec>(exec, 0, n), KOKKOS_LAMBDA(int q) {
                    int k = 0;
                    ball(g, q, h(q), per_h, depth, [&](int, double) { ++k; });
                    h(q) *= Kokkos::cbrt(double(ngb + 1) / Kokkos::max(k, 1));  // +1: the particle itself
                });
    }
    exec.fence();

    using Range = Kokkos::RangePolicy<Exec>;
    // One-way: count, scan, fill.
    auto count_pass = [&] {
        Kokkos::parallel_for(
            "ngb::count", Range(exec, 0, n), KOKKOS_LAMBDA(int q) {
                int k = 0;
                ball(g, q, h(q), per_h, depth, [&](int s, double) { k += s != q; });
                count(q) = k;
            });
    };
    int total = 0;
    auto scan_pass = [&](const Ints& c, const Ints& o, int& sum) {
        Kokkos::parallel_scan(
            "ngb::scan", Range(exec, 0, n + 1),
            KOKKOS_LAMBDA(int q, int& at, bool last) {
                const int k = q < n ? c(q) : 0;
                if (last) o(q) = at;
                at += k;
            },
            sum);
    };
    Ints neighbours;
    auto fill_pass = [&] {
        Kokkos::parallel_for(
            "ngb::fill", Range(exec, 0, n), KOKKOS_LAMBDA(int q) {
                int at = offsets(q);
                ball(g, q, h(q), per_h, depth, [&](int s, double) {
                    if (s != q) neighbours(at++) = s;
                });
            });
    };
    const double count_ms = timed(repeats, count_pass);
    const double scan_ms = timed(repeats, [&] { scan_pass(count, offsets, total); });
    neighbours = Ints(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("neighbours")), total);
    const double fill_ms = timed(repeats, fill_pass);
    {
        const auto ch = to_host(count);
        std::vector<int> v(ch.data(), ch.data() + n);
        long long sum = 0;
        for (int k : v) sum += k;
        std::cout << "oneway: count_ms=" << count_ms << " scan_ms=" << scan_ms << " fill_ms=" << fill_ms
                  << " list_ms=" << count_ms + scan_ms + fill_ms
                  << " ns_per_particle=" << (count_ms + scan_ms + fill_ms) * 1e6 / n
                  << " with_cell_list_ns_per_particle=" << (build_ms + count_ms + scan_ms + fill_ms) * 1e6 / n
                  << " entries=" << total << " mean=" << double(sum) / n << " p1=" << percentile(v, 0.01)
                  << " p50=" << percentile(v, 0.5) << " p99=" << percentile(v, 0.99) << " max=" << percentile(v, 1.0)
                  << "\n";
    }

    {  // Candidates (particles whose distance is computed) per particle, untimed.
        long long tested = 0;
        Kokkos::parallel_reduce(
            "ngb::candidates", Range(exec, 0, n),
            KOKKOS_LAMBDA(int q, long long& sum) {
                int t = 0;
                ball(g, q, h(q), per_h, depth, [&](int, double) {}, &t);
                sum += t;
            },
            tested);
        std::cout << "candidates_per_particle=" << double(tested) / n << "\n";
    }

    // Symmetric: the transpose of the one-way list (atomics), then row i = one-way row i plus the j of transpose
    // row i with r_ij >= h_i (those with r_ij < h_i are in the one-way row already).
    Ints tcount("tcount", n), toffsets("toffsets", n + 1), tcursor("tcursor", n), transposed;
    Ints scount("scount", n), soffsets("soffsets", n + 1), sym;
    int ttotal = 0, stotal = 0;
    auto transpose = [&] {
        Kokkos::deep_copy(exec, tcount, 0);
        Kokkos::parallel_for(
            "ngb::t_count", Range(exec, 0, n), KOKKOS_LAMBDA(int q) {
                for (int a = offsets(q); a < offsets(q + 1); ++a) Kokkos::atomic_inc(&tcount(neighbours(a)));
            });
        scan_pass(tcount, toffsets, ttotal);
        if (transposed.extent(0) != static_cast<std::size_t>(ttotal))
            transposed = Ints(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("transposed")), ttotal);
        Kokkos::deep_copy(exec, tcursor, 0);
        Kokkos::parallel_for(
            "ngb::t_fill", Range(exec, 0, n), KOKKOS_LAMBDA(int q) {
                for (int a = offsets(q); a < offsets(q + 1); ++a) {
                    const int j = neighbours(a);
                    transposed(toffsets(j) + Kokkos::atomic_fetch_inc(&tcursor(j))) = q;
                }
            });
    };
    auto extra = KOKKOS_LAMBDA(int q, int j) {
        double ax = g.positions(j, 0) - g.positions(q, 0), ay = g.positions(j, 1) - g.positions(q, 1),
               az = g.positions(j, 2) - g.positions(q, 2);
        ax -= Kokkos::round(ax), ay -= Kokkos::round(ay), az -= Kokkos::round(az);
        return ax * ax + ay * ay + az * az >= h(q) * h(q);
    };
    auto merge = [&] {
        Kokkos::parallel_for(
            "ngb::s_count", Range(exec, 0, n), KOKKOS_LAMBDA(int q) {
                int k = offsets(q + 1) - offsets(q);
                for (int a = toffsets(q); a < toffsets(q + 1); ++a) k += extra(q, transposed(a));
                scount(q) = k;
            });
        scan_pass(scount, soffsets, stotal);
        if (sym.extent(0) != static_cast<std::size_t>(stotal))
            sym = Ints(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("symmetric")), stotal);
        Kokkos::parallel_for(
            "ngb::s_fill", Range(exec, 0, n), KOKKOS_LAMBDA(int q) {
                int at = soffsets(q);
                for (int a = offsets(q); a < offsets(q + 1); ++a) sym(at++) = neighbours(a);
                for (int a = toffsets(q); a < toffsets(q + 1); ++a)
                    if (extra(q, transposed(a))) sym(at++) = transposed(a);
            });
    };
    const double transpose_ms = timed(repeats, transpose);
    const double merge_ms = timed(repeats, merge);
    {
        const auto sh = to_host(scount);
        std::vector<int> v(sh.data(), sh.data() + n);
        const double all = count_ms + scan_ms + fill_ms + transpose_ms + merge_ms;
        std::cout << "symmetric: transpose_ms=" << transpose_ms << " merge_ms=" << merge_ms
                  << " list_ms(incl_oneway)=" << all << " ns_per_particle=" << all * 1e6 / n
                  << " with_cell_list_ns_per_particle=" << (build_ms + all) * 1e6 / n << " entries=" << stotal
                  << " mean=" << double(stotal) / n << " p1=" << percentile(v, 0.01) << " p50=" << percentile(v, 0.5)
                  << " p99=" << percentile(v, 0.99) << " max=" << percentile(v, 1.0) << "\n";
    }

    // Verification on sampled slots: brute force over all particles (periodic nearest image), in long double.
    // A pair within 1e-12 relative of the radius is borderline (the device may round d2 differently) and not counted.
    int wrong = 0, borderline = 0;
    {
        const auto pos = to_host(g.positions);
        const auto hh = to_host(h);
        const auto off = to_host(offsets), nb = to_host(neighbours);
        const auto soff = to_host(soffsets), sn = to_host(sym);
        std::mt19937_64 rng(5);
        std::uniform_int_distribution<int> any(0, n - 1);
        for (int k = 0; k < sampled; ++k) {
            const int q = any(rng);
            std::vector<int> one(nb.data() + off(q), nb.data() + off(q + 1));
            std::vector<int> two(sn.data() + soff(q), sn.data() + soff(q + 1));
            std::sort(one.begin(), one.end());
            std::sort(two.begin(), two.end());
            const bool dup = std::adjacent_find(two.begin(), two.end()) != two.end();
            std::vector<int> want_one, want_two, near;
            for (int s = 0; s < n; ++s) {
                if (s == q) continue;
                long double d2 = 0;
                for (int a = 0; a < 3; ++a) {
                    long double d = static_cast<long double>(pos(s, a)) - pos(q, a);
                    d -= std::round(static_cast<double>(d));
                    d2 += d * d;
                }
                const long double hi2 = static_cast<long double>(hh(q)) * hh(q);
                const long double hm = std::max(hh(q), hh(s));
                if (d2 < hi2) want_one.push_back(s);
                if (d2 < hm * hm) want_two.push_back(s);
                if (std::fabs(static_cast<double>(d2 / hi2 - 1)) < 1e-12 ||
                    std::fabs(static_cast<double>(d2 / (hm * hm) - 1)) < 1e-12)
                    near.push_back(s);
            }
            auto differ = [&](const std::vector<int>& got, const std::vector<int>& want) {
                std::vector<int> d;
                std::set_symmetric_difference(got.begin(), got.end(), want.begin(), want.end(), std::back_inserter(d));
                int bad = 0;
                for (int s : d) bad += std::find(near.begin(), near.end(), s) == near.end();
                borderline += static_cast<int>(d.size()) - bad;
                return bad;
            };
            wrong += differ(one, want_one) + differ(two, want_two) + dup;
        }
    }
    std::cout << "verify: sampled=" << sampled << " wrong=" << wrong << " borderline=" << borderline
              << (wrong ? " FAIL" : " PASS") << "\n";
    return wrong ? 1 : 0;
}
