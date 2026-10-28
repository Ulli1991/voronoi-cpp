#include <Kokkos_Core.hpp>
#include <Kokkos_Graph.hpp>
#include <Kokkos_Sort.hpp>

#include "neighbor_search.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Positions = Kokkos::View<double*[3]>;
using ScalarView = Kokkos::View<double*>;
using IntView = Kokkos::View<int*>;

struct BvhNode {
    double lo[3];
    double hi[3];
    int left;
    int right;
    int begin;
    int end;
};

struct PointSet {
    std::string name;
    std::vector<std::array<double, 3>> points;
    std::vector<double> radii;
};

PointSet make_uniform(int n, std::uint64_t seed) {
    PointSet s{"uniform", {}, {}};
    s.points.resize(n);
    s.radii.resize(n, 0.075);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(1.0e-9, 1.0 - 1.0e-9);
    for (auto& p : s.points) for (double& x : p) x = u(rng);
    return s;
}

PointSet make_jittered(int n, std::uint64_t seed) {
    const int side = static_cast<int>(std::llround(std::cbrt(static_cast<double>(n))));
    n = side * side * side;
    PointSet s{"jittered_lattice", {}, {}};
    s.points.reserve(n);
    s.radii.resize(n, 0.90 / side);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> jitter(-0.18, 0.18);
    for (int i = 0; i < side; ++i)
        for (int j = 0; j < side; ++j)
            for (int k = 0; k < side; ++k) {
                const double q[3] = {(i + 0.5) / side, (j + 0.5) / side, (k + 0.5) / side};
                std::array<double, 3> p;
                for (int a = 0; a < 3; ++a)
                    p[a] = std::max(1.0e-9, std::min(1.0 - 1.0e-9, q[a] + jitter(rng) / side));
                s.points.push_back(p);
            }
    return s;
}

PointSet make_clustered(int n, std::uint64_t seed) {
    PointSet s{"eight_clusters", {}, {}};
    s.points.resize(n);
    s.radii.resize(n, 0.04);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.12, 0.88);
    std::normal_distribution<double> g(0.0, 0.055);
    std::array<std::array<double, 3>, 8> centers;
    for (auto& c : centers) for (double& x : c) x = u(rng);
    for (int i = 0; i < n; ++i) {
        const auto& c = centers[i % 8];
        for (int a = 0; a < 3; ++a)
            s.points[i][a] = std::max(1.0e-9, std::min(1.0 - 1.0e-9, c[a] + g(rng)));
    }
    return s;
}

PointSet make_gradient(int n, std::uint64_t seed) {
    PointSet s{"density_gradient", {}, {}};
    s.points.resize(n);
    s.radii.resize(n);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        // Sample p(x) proportional to 0.2 + 1.8 x by inverting its CDF.
        const double q = u(rng);
        const double x = (-0.2 + std::sqrt(0.04 + 3.96 * q)) / 1.8;
        s.points[i] = {x, u(rng), u(rng)};
        const double density = 0.2 + 1.8 * x;
        s.radii[i] = 0.075 * std::cbrt(1.1 / density);
    }
    return s;
}

Positions copy_positions(const PointSet& s) {
    Positions p("positions", s.points.size());
    auto h = Kokkos::create_mirror_view(p);
    for (std::size_t i = 0; i < s.points.size(); ++i)
        for (int a = 0; a < 3; ++a) h(i, a) = s.points[i][a];
    Kokkos::deep_copy(p, h);
    return p;
}

ScalarView copy_radii(const PointSet& s) {
    ScalarView r("radii", s.radii.size());
    auto h = Kokkos::create_mirror_view(r);
    for (std::size_t i = 0; i < s.radii.size(); ++i) h(i) = s.radii[i];
    Kokkos::deep_copy(r, h);
    return r;
}

struct CellIndex {
    IntView occupied_keys;
    IntView sorted_ids;
    IntView offsets;
    int occupied_count;
    int nx, ny, nz;
    double hx, hy, hz;
};

CellIndex build_cell_index(const Exec& exec, Positions positions, double target_width) {
    const int n = static_cast<int>(positions.extent(0));
    const int nx = std::max(1, static_cast<int>(std::ceil(1.0 / target_width)));
    const int ny = nx, nz = nx;
    const double hx = 1.0 / nx, hy = hx, hz = hx;
    IntView keys("cell_keys", n), ids("particle_ids", n);
    Kokkos::parallel_for("cell_key", Kokkos::RangePolicy<Exec>(exec, 0, n), KOKKOS_LAMBDA(int i) {
        int ix = static_cast<int>(positions(i, 0) / hx);
        int iy = static_cast<int>(positions(i, 1) / hy);
        int iz = static_cast<int>(positions(i, 2) / hz);
        ix = ix < nx ? ix : nx - 1;
        iy = iy < ny ? iy : ny - 1;
        iz = iz < nz ? iz : nz - 1;
        keys(i) = (ix * ny + iy) * nz + iz;
        ids(i) = i;
    });
    Kokkos::Experimental::sort_by_key(exec, keys, ids);

    IntView is_start("occupied_cell_start", n), compact_id("occupied_cell_id", n);
    Kokkos::parallel_for("mark_occupied_cells", Kokkos::RangePolicy<Exec>(exec, 0, n),
        KOKKOS_LAMBDA(int i) { is_start(i) = (i == 0 || keys(i) != keys(i-1)) ? 1 : 0; });
    int occupied_count=0;
    Kokkos::parallel_scan("compact_cell_scan", Kokkos::RangePolicy<Exec>(exec, 0, n),
        KOKKOS_LAMBDA(int i, int& update, bool final) {
            if (final) compact_id(i) = update;
            update += is_start(i);
        }, occupied_count);
    IntView occupied_keys("occupied_cell_keys", n), offsets("occupied_cell_offsets", n+1);
    Kokkos::parallel_for("write_occupied_cells", Kokkos::RangePolicy<Exec>(exec, 0, n),
        KOKKOS_LAMBDA(int i) {
            if (is_start(i)) {
                const int u=compact_id(i);
                occupied_keys(u)=keys(i);
                offsets(u)=i;
            }
        });
    Kokkos::parallel_for("occupied_cell_offset_end", Kokkos::RangePolicy<Exec>(exec, 0, 1),
        KOKKOS_LAMBDA(int) { offsets(occupied_count)=n; });
    exec.fence("cell index build");
    return {occupied_keys, ids, offsets, occupied_count, nx, ny, nz, hx, hy, hz};
}

struct HostBvh {
    std::vector<BvhNode> nodes;
    std::vector<int> ids;
};

int build_bvh_node(HostBvh& b, const std::vector<std::array<double, 3>>& p,
                   int begin, int end, int leaf_size) {
    const int node_id = static_cast<int>(b.nodes.size());
    b.nodes.emplace_back();
    BvhNode node{};
    for (int a = 0; a < 3; ++a) {
        node.lo[a] = std::numeric_limits<double>::infinity();
        node.hi[a] = -std::numeric_limits<double>::infinity();
    }
    for (int k = begin; k < end; ++k) {
        const auto& x = p[b.ids[k]];
        for (int a = 0; a < 3; ++a) {
            node.lo[a] = std::min(node.lo[a], x[a]);
            node.hi[a] = std::max(node.hi[a], x[a]);
        }
    }
    node.begin = begin; node.end = end; node.left = node.right = -1;
    if (end - begin > leaf_size) {
        int axis = 0;
        for (int a = 1; a < 3; ++a)
            if (node.hi[a] - node.lo[a] > node.hi[axis] - node.lo[axis]) axis = a;
        const int mid = begin + (end - begin) / 2;
        std::nth_element(b.ids.begin() + begin, b.ids.begin() + mid, b.ids.begin() + end,
            [&](int x, int y) { return p[x][axis] < p[y][axis]; });
        node.left = build_bvh_node(b, p, begin, mid, leaf_size);
        node.right = build_bvh_node(b, p, mid, end, leaf_size);
    }
    b.nodes[node_id] = node;
    return node_id;
}

HostBvh make_host_bvh(const std::vector<std::array<double, 3>>& points, int leaf_size = 8) {
    HostBvh b;
    b.ids.resize(points.size());
    std::iota(b.ids.begin(), b.ids.end(), 0);
    build_bvh_node(b, points, 0, static_cast<int>(points.size()), leaf_size);
    return b;
}

struct DeviceBvh {
    Kokkos::View<BvhNode*> nodes;
    IntView ids;
};

DeviceBvh copy_bvh(const HostBvh& b) {
    Kokkos::View<BvhNode*> nodes("bvh_nodes", b.nodes.size());
    IntView ids("bvh_ids", b.ids.size());
    auto hn = Kokkos::create_mirror_view(nodes);
    auto hi = Kokkos::create_mirror_view(ids);
    for (std::size_t i = 0; i < b.nodes.size(); ++i) hn(i) = b.nodes[i];
    for (std::size_t i = 0; i < b.ids.size(); ++i) hi(i) = b.ids[i];
    Kokkos::deep_copy(nodes, hn);
    Kokkos::deep_copy(ids, hi);
    return {nodes, ids};
}

template<class Search>
void search_count(const Exec& exec, Positions p, ScalarView radii, IntView query_order,
                  Search search, IntView counts, IntView checks, IntView index_tests) {
    const int n = static_cast<int>(p.extent(0));
    Kokkos::parallel_for("neighbor_count", Kokkos::RangePolicy<Exec>(exec, 0, n),
        KOKKOS_LAMBDA(int t) {
            const int i = query_order(t);
            int hits = 0, tested = 0, broad_tests = 0;
            search.count(i, p, radii(i), hits, tested, broad_tests);
            counts(i) = hits;
            checks(i) = tested;
            index_tests(i) = broad_tests;
        });
}

template<class Search>
void search_fill(const Exec& exec, Positions p, ScalarView radii, IntView query_order,
                 Search search, IntView offsets, IntView ids) {
    const int n = static_cast<int>(p.extent(0));
    Kokkos::parallel_for("neighbor_fill", Kokkos::RangePolicy<Exec>(exec, 0, n),
        KOKKOS_LAMBDA(int t) {
            const int i = query_order(t);
            int ignored_hits = 0, ignored_tests = 0, ignored_index_tests = 0;
            int cursor = offsets(i);
            search.fill(i, p, radii(i), ids, cursor, ignored_hits, ignored_tests,
                        ignored_index_tests);
        });
}

struct CellSearch {
    IntView order;
    IntView occupied_keys;
    IntView offsets;
    int occupied_count;
    int nx, ny, nz;
    double hx, hy, hz;

    KOKKOS_INLINE_FUNCTION int cell(int x, int y, int z) const { return (x * ny + y) * nz + z; }
    KOKKOS_INLINE_FUNCTION int find_bin(int key) const {
        int lo=0, hi=occupied_count;
        while(lo<hi) {
            const int mid=lo+(hi-lo)/2;
            if(occupied_keys(mid)<key) lo=mid+1; else hi=mid;
        }
        return (lo<occupied_count && occupied_keys(lo)==key) ? lo : -1;
    }
    KOKKOS_INLINE_FUNCTION void count(int i, Positions p, double r, int& hits, int& checks,
                                     int& index_tests) const {
        const int ix0 = static_cast<int>(p(i, 0) / hx);
        const int iy0 = static_cast<int>(p(i, 1) / hy);
        const int iz0 = static_cast<int>(p(i, 2) / hz);
        const int rx = static_cast<int>(Kokkos::ceil(r / hx));
        const int ry = static_cast<int>(Kokkos::ceil(r / hy));
        const int rz = static_cast<int>(Kokkos::ceil(r / hz));
        const double r2 = r * r;
        for (int x = ix0-rx; x <= ix0+rx; ++x) if (x >= 0 && x < nx)
            for (int y = iy0-ry; y <= iy0+ry; ++y) if (y >= 0 && y < ny)
                for (int z = iz0-rz; z <= iz0+rz; ++z) if (z >= 0 && z < nz) {
                    ++index_tests;
                    const double lx=x*hx, ly=y*hy, lz=z*hz;
                    const double qx=p(i,0)<lx ? lx-p(i,0) : (p(i,0)>lx+hx ? p(i,0)-lx-hx : 0.0);
                    const double qy=p(i,1)<ly ? ly-p(i,1) : (p(i,1)>ly+hy ? p(i,1)-ly-hy : 0.0);
                    const double qz=p(i,2)<lz ? lz-p(i,2) : (p(i,2)>lz+hz ? p(i,2)-lz-hz : 0.0);
                    if(qx*qx+qy*qy+qz*qz>r2) continue;
                    const int c = find_bin(cell(x,y,z));
                    if(c<0) continue;
                    for (int q = offsets(c); q < offsets(c+1); ++q) {
                        const int j = order(q);
                        if (j == i) continue;
                        ++checks;
                        const double dx=p(j,0)-p(i,0), dy=p(j,1)-p(i,1), dz=p(j,2)-p(i,2);
                        if (dx*dx+dy*dy+dz*dz <= r2) ++hits;
                    }
                }
    }
    KOKKOS_INLINE_FUNCTION void fill(int i, Positions p, double r, IntView ids, int& cursor,
                                    int& hits, int& checks, int& index_tests) const {
        const int ix0 = static_cast<int>(p(i, 0) / hx);
        const int iy0 = static_cast<int>(p(i, 1) / hy);
        const int iz0 = static_cast<int>(p(i, 2) / hz);
        const int rx = static_cast<int>(Kokkos::ceil(r / hx));
        const int ry = static_cast<int>(Kokkos::ceil(r / hy));
        const int rz = static_cast<int>(Kokkos::ceil(r / hz));
        const double r2 = r * r;
        for (int x = ix0-rx; x <= ix0+rx; ++x) if (x >= 0 && x < nx)
            for (int y = iy0-ry; y <= iy0+ry; ++y) if (y >= 0 && y < ny)
                for (int z = iz0-rz; z <= iz0+rz; ++z) if (z >= 0 && z < nz) {
                    const double lx=x*hx, ly=y*hy, lz=z*hz;
                    const double qx=p(i,0)<lx ? lx-p(i,0) : (p(i,0)>lx+hx ? p(i,0)-lx-hx : 0.0);
                    const double qy=p(i,1)<ly ? ly-p(i,1) : (p(i,1)>ly+hy ? p(i,1)-ly-hy : 0.0);
                    const double qz=p(i,2)<lz ? lz-p(i,2) : (p(i,2)>lz+hz ? p(i,2)-lz-hz : 0.0);
                    if(qx*qx+qy*qy+qz*qz>r2) continue;
                    const int c = find_bin(cell(x,y,z));
                    if(c<0) continue;
                    for (int q = offsets(c); q < offsets(c+1); ++q) {
                        const int j = order(q);
                        if (j == i) continue;
                        const double dx=p(j,0)-p(i,0), dy=p(j,1)-p(i,1), dz=p(j,2)-p(i,2);
                        if (dx*dx+dy*dy+dz*dz <= r2) { ids(cursor++)=j; ++hits; }
                    }
                }
        (void)checks;
        (void)index_tests;
    }
};

struct BvhSearch {
    Kokkos::View<BvhNode*> nodes;
    IntView order;

    KOKKOS_INLINE_FUNCTION bool intersects(const BvhNode& node, Positions p, int i, double r2) const {
        double d2 = 0.0;
        for (int a=0; a<3; ++a) {
            const double x=p(i,a);
            const double d=x<node.lo[a] ? node.lo[a]-x : (x>node.hi[a] ? x-node.hi[a] : 0.0);
            d2 += d*d;
        }
        return d2 <= r2;
    }
    KOKKOS_INLINE_FUNCTION void count(int i, Positions p, double r, int& hits, int& checks,
                                     int& index_tests) const {
        int stack[64]; int top=0; stack[top++]=0;
        const double r2=r*r;
        while(top>0) {
            const BvhNode current_node=nodes(stack[--top]);
            ++index_tests;
            if(!intersects(current_node,p,i,r2)) continue;
            if(current_node.left>=0) { stack[top++]=current_node.right; stack[top++]=current_node.left; }
            else for(int q=current_node.begin;q<current_node.end;++q) {
                const int j=order(q); if(j==i) continue; ++checks;
                const double dx=p(j,0)-p(i,0),dy=p(j,1)-p(i,1),dz=p(j,2)-p(i,2);
                if(dx*dx+dy*dy+dz*dz<=r2) ++hits;
            }
        }
    }
    KOKKOS_INLINE_FUNCTION void fill(int i, Positions p, double r, IntView ids, int& cursor,
                                    int& hits, int& checks, int& index_tests) const {
        int stack[64]; int top=0; stack[top++]=0;
        const double r2=r*r;
        while(top>0) {
            const BvhNode current_node=nodes(stack[--top]);
            if(!intersects(current_node,p,i,r2)) continue;
            if(current_node.left>=0) { stack[top++]=current_node.right; stack[top++]=current_node.left; }
            else for(int q=current_node.begin;q<current_node.end;++q) {
                const int j=order(q); if(j==i) continue;
                const double dx=p(j,0)-p(i,0),dy=p(j,1)-p(i,1),dz=p(j,2)-p(i,2);
                if(dx*dx+dy*dy+dz*dz<=r2) { ids(cursor++)=j; ++hits; }
            }
        }
        (void)checks;
        (void)index_tests;
    }
};


// Compact the table into CSR without reading the total back to the host.
void dense_compact(const Exec& exec, Kokkos::View<int**> table, IntView counts,
                   IntView offsets, IntView ids) {
    const int n = static_cast<int>(counts.extent(0));
    Kokkos::parallel_scan("dense_neighbor_scan", Kokkos::RangePolicy<Exec>(exec, 0, n),
        KOKKOS_LAMBDA(int i, int& update, bool final) {
            if (final) offsets(i) = update;
            update += counts(i);
            if (final && i == n - 1) offsets(n) = update;
        });
    Kokkos::parallel_for("dense_neighbor_compact", Kokkos::RangePolicy<Exec>(exec, 0, n),
        KOKKOS_LAMBDA(int t) {
            const int begin = offsets(t);
            for (int c = 0; c < counts(t); ++c) ids(begin + c) = table(t, c);
        });
}

struct Csr {
    IntView counts, checks, index_tests, offsets, ids;
};

// Two-pass CSR construction: count, scan, fill. Reading the scan total back
// to size `ids` synchronizes with the host once per rebuild.
template<class Search>
Csr build_csr(const Exec& exec, Positions p, ScalarView radii, IntView query_order,
              Search search) {
    const int n = static_cast<int>(p.extent(0));
    IntView counts("neighbor_counts", n), checks("distance_checks", n);
    IntView index_tests("index_tests", n), offsets("neighbor_offsets", n+1);
    search_count(exec, p, radii, query_order, search, counts, checks, index_tests);
    int total=0;
    Kokkos::parallel_scan("neighbor_scan", Kokkos::RangePolicy<Exec>(exec, 0, n),
        KOKKOS_LAMBDA(int i, int& update, bool final) {
            if(final) offsets(i)=update;
            update += counts(i);
        }, total);
    Kokkos::parallel_for("neighbor_scan_end", Kokkos::RangePolicy<Exec>(exec, 0, 1),
        KOKKOS_LAMBDA(int) { offsets(n)=total; });
    IntView ids("neighbor_ids", total);
    search_fill(exec, p, radii, query_order, search, offsets, ids);
    exec.fence("neighbor CSR construction");
    return {counts, checks, index_tests, offsets, ids};
}

double median(std::vector<double> v) {
    std::nth_element(v.begin(),v.begin()+v.size()/2,v.end());
    return v[v.size()/2];
}

// One untimed warm-up call absorbs first-launch, module-load, and allocator
// costs; the median of the timed calls is reported.
template<class F>
double median_ms(int repeats, const F& f) {
    f();
    std::vector<double> times(repeats);
    for(double& t : times) { Kokkos::Timer timer; f(); t=timer.seconds()*1.0e3; }
    return median(times);
}

struct Result {
    double build_ms;
    double query_ms;
    double csr_copy_ms;
    long long distance_checks;
    long long index_tests;
    long long edges;
    int max_neighbors;
    std::vector<std::vector<int>> adjacency;
};

template<class Search>
Result run_query(const Exec& exec, Positions p, ScalarView radii, IntView query_order,
                 Search search, double build_ms, int repeats) {
    const int n = static_cast<int>(p.extent(0));
    Csr csr;
    const double query_ms=median_ms(repeats, [&] {
        csr=build_csr(exec, p, radii, query_order, search);
    });
    // Moving the graph to the host is what a host consumer of a device search
    // (or the reverse) pays; on host backends the mirror aliases and costs ~0.
    const double csr_copy_ms=median_ms(repeats, [&] {
        auto o=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},csr.offsets);
        auto q=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},csr.ids);
    });

    auto hc=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},csr.counts);
    auto hcheck=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},csr.checks);
    auto hindex=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},csr.index_tests);
    auto hoff=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},csr.offsets);
    auto hids=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},csr.ids);
    Result r{build_ms,query_ms,csr_copy_ms,0,0,static_cast<long long>(csr.ids.extent(0)),0,{}};
    r.adjacency.resize(n);
    for(int i=0;i<n;++i) {
        r.max_neighbors=std::max(r.max_neighbors,hc(i));
        r.distance_checks+=hcheck(i);
        r.index_tests+=hindex(i);
        auto& row=r.adjacency[i]; row.reserve(hc(i));
        for(int q=hoff(i);q<hoff(i+1);++q) row.push_back(hids(q));
        std::sort(row.begin(),row.end());
    }
    return r;
}

enum class DenseMode { table, nearest_first, csr, graph };

const char* mode_name(DenseMode m) {
    switch(m) {
        case DenseMode::table: return "dense_table";
        case DenseMode::nearest_first: return "dense_nearest_first";
        case DenseMode::csr: return "dense_csr";
        case DenseMode::graph: return "dense_graph";
    }
    return "";
}

// Persistent-workspace search: sized by one untimed rebuild (which grows the
// table on overflow, counted in `retries`), then the cell build and the fill
// are timed. Each timed fill reads the overflow flag back, as a solver must.
// In graph mode the whole rebuild is one captured graph, so only its total
// time is reported.
Result run_dense(const Exec& exec, Positions p, ScalarView radii, double width, bool periodic,
                 int capacity, DenseMode mode, int repeats, bool validate, int& retries,
                 int& lanes) {
    const int n = static_cast<int>(p.extent(0));
    neighbor_search::DenseCellSearch<Exec> s;
    // VORONOI_LANES overrides the lanes-per-particle heuristic for tuning.
    const char* lanes_env=std::getenv("VORONOI_LANES");
    neighbor_search::configure(s, n, Kokkos::Array<double,3>{1.0,1.0,1.0}, width, periodic,
                               capacity, mode==DenseMode::nearest_first,
                               lanes_env ? std::atoi(lanes_env) : 0);
    lanes=s.vector_length;
    retries=neighbor_search::rebuild(exec,s,p,radii);
    IntView offsets("dense_neighbor_offsets", n+1);
    IntView ids(mode==DenseMode::csr ? "dense_neighbor_ids" : "unused_ids",
                mode==DenseMode::csr ? static_cast<std::size_t>(n)*s.capacity : 0);
    int longest=0;
    auto check_overflow=[&] {
        Kokkos::deep_copy(exec,longest,s.max_count);
        exec.fence("overflow check");
        if(longest>s.capacity) throw std::runtime_error("neighbor table overflow after sizing");
    };
    double build_ms=0.0, query_ms=0.0;
    if(mode==DenseMode::graph) {
#if defined(KOKKOS_ENABLE_CUDA)
        if constexpr (std::is_same_v<Exec, Kokkos::Cuda>) {
            auto graph=Kokkos::Experimental::create_graph<Exec>([&](const auto& root) {
                root.cuda_capture(exec, [=](const Exec& e) {
                    neighbor_search::build(e,s,p,radii);
                    neighbor_search::fill(e,s);
                });
            });
            graph.instantiate();
            query_ms=median_ms(repeats, [&] { graph.submit(exec); check_overflow(); });
        } else
#endif
        {
            throw std::runtime_error("graph mode needs the CUDA backend");
        }
    } else {
        build_ms=median_ms(repeats, [&] {
            neighbor_search::build(exec,s,p,radii);
            exec.fence("dense build");
        });
        query_ms=median_ms(repeats, [&] {
            neighbor_search::fill(exec,s);
            if(mode==DenseMode::csr) dense_compact(exec,s.table,s.counts,offsets,ids);
            check_overflow();
        });
    }

    auto hc=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},s.counts);
    long long total=0;
    for(int t=0;t<n;++t) total+=hc(t);
    Result r{build_ms,query_ms,0.0,0,0,total,longest,{}};
    if(!validate) return r;
    auto horder=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},s.order);
    auto ht=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},s.table);
    auto hp=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},s.positions);
    r.adjacency.resize(n);
    for(int t=0;t<n;++t) {
        auto& row=r.adjacency[horder(t)];
        double previous=-1.0;
        for(int c=0;c<hc(t);++c) {
            const int q=ht(t,c);
            row.push_back(horder(q));
            if(mode!=DenseMode::nearest_first) continue;
            double d2=0.0;
            for(int a=0;a<3;++a) {
                double d=hp(q,a)-hp(t,a);
                if(periodic) d-=std::nearbyint(d);
                d2+=d*d;
            }
            // An out-of-order row is reported as a mismatch via a sentinel id.
            if(d2<previous) row.push_back(-1);
            previous=d2;
        }
    }
    for(auto& row : r.adjacency) std::sort(row.begin(),row.end());
    return r;
}

std::vector<std::vector<int>> brute_force(const PointSet& s, bool periodic) {
    const int n=static_cast<int>(s.points.size());
    std::vector<std::vector<int>> out(n);
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) if(i!=j) {
        double d2=0.;
        for(int a=0;a<3;++a) {
            double d=s.points[i][a]-s.points[j][a];
            if(periodic) d-=std::nearbyint(d);
            d2+=d*d;
        }
        if(d2<=s.radii[i]*s.radii[i]) out[i].push_back(j);
    }
    return out;
}

// Keep the expected neighbor count independent of n; at n=4096 and scale 1
// the radii are unchanged. The jittered lattice already scales with spacing.
void scale_radii(PointSet& s, double factor) {
    for(double& r : s.radii) r*=factor;
}

void report(const PointSet& s, const std::string& method, const std::string& order,
            const Result& r, const std::vector<std::vector<int>>& reference,
            const std::string& reference_name, double cell_width=0.0) {
    // -1 marks an unvalidated row (timing mode, or no reference available).
    int bad_rows=reference.empty() || r.adjacency.empty() ? -1 : 0;
    if(bad_rows==0)
        for(std::size_t i=0;i<reference.size();++i) if(r.adjacency[i]!=reference[i]) ++bad_rows;
    const double n=s.points.size();
    std::cout << s.name << ',' << method << ',' << order << ',' << std::fixed
              << std::setprecision(4) << n << ',' << cell_width << ',' << r.build_ms << ','
              << r.query_ms << ',' << r.build_ms+r.query_ms << ',' << r.csr_copy_ms << ','
              << r.index_tests/n << ',' << r.distance_checks/n << ','
              << static_cast<double>(r.edges)/n << ','
              << r.max_neighbors << ',' << reference_name << ',' << bad_rows << std::endl;
}

} // namespace

// Usage: neighbor_benchmark [n=4096] [radius_scale=1] [repeats=1]
//                           [methods=all|dense|timing] [periodic=0|1]
// "dense" runs the dense-cell variants plus the BVH used as their reference;
// "timing" skips every reference and check. Periodic runs use the minimum
// image and are validated against all pairs only up to the oracle limit.
int main(int argc, char** argv) {
    Kokkos::initialize(argc,argv);
    {
        int n=argc>1 ? std::max(64,std::atoi(argv[1])) : 4096;
        const double radius_scale=argc>2 ? std::atof(argv[2]) : 1.0;
        const int repeats=argc>3 ? std::max(1,std::atoi(argv[3])) : 1;
        const std::string methods=argc>4 ? argv[4] : "all";
        const bool periodic=argc>5 && std::atoi(argv[5])!=0;
        const bool validate=methods!="timing";
        const bool legacy=methods=="all" && !periodic;
        const int oracle_limit=32768;
        const Exec exec{};
        std::cout << "# backend=" << Exec::name() << " concurrency=" << exec.concurrency()
                  << " radius_scale=" << radius_scale << " repeats=" << repeats
                  << " methods=" << methods << " periodic=" << periodic << '\n';
        std::cout << "distribution,method,query_order,n,cell_width,build_ms,query_ms,total_ms,csr_copy_ms,index_tests_per_particle,point_distance_checks_per_particle,mean_neighbors,max_neighbors,reference,mismatched_rows\n";
        const std::uint64_t seed=20261027;
        const double density_scale=std::cbrt(4096.0/n);
        std::vector<PointSet> sets;
        sets.push_back(make_uniform(n,seed));
        sets.push_back(make_jittered(n,seed+1));
        sets.push_back(make_clustered(n,seed+2));
        sets.push_back(make_gradient(n,seed+3));
        scale_radii(sets[0],radius_scale*density_scale);
        scale_radii(sets[1],radius_scale);
        scale_radii(sets[2],radius_scale*density_scale);
        scale_radii(sets[3],radius_scale*density_scale);
        std::vector<DenseMode> modes{DenseMode::table, DenseMode::nearest_first, DenseMode::csr};
#if defined(KOKKOS_ENABLE_CUDA)
        if(std::is_same_v<Exec, Kokkos::Cuda>) modes.push_back(DenseMode::graph);
#endif
        for(const auto& s:sets) {
            const int m=static_cast<int>(s.points.size());
            Positions p=copy_positions(s); ScalarView radii=copy_radii(s);
            IntView input_order("input_order", m);
            Kokkos::parallel_for("input_order", Kokkos::RangePolicy<Exec>(exec, 0, m),
                KOKKOS_LAMBDA(int i) { input_order(i)=i; });

            // Above the oracle limit the all-pairs check is too slow; the
            // (non-periodic) BVH, validated against it at smaller n, serves as
            // the reference there.
            std::vector<std::vector<int>> reference;
            std::string reference_name="none";
            int capacity=16;
            DeviceBvh device_bvh;
            double bvh_build_ms=0.0;
            if(validate && !periodic) {
                bvh_build_ms=median_ms(repeats, [&] {
                    HostBvh host_bvh=make_host_bvh(s.points);
                    device_bvh=copy_bvh(host_bvh);
                    exec.fence("BVH setup");
                });
                BvhSearch tree{device_bvh.nodes,device_bvh.ids};
                Result tree_result=run_query(exec,p,radii,input_order,tree,bvh_build_ms,repeats);
                const bool exact=m<=oracle_limit;
                reference_name=exact ? "all_pairs" : "median_bvh";
                reference=exact ? brute_force(s,false) : tree_result.adjacency;
                report(s,"median_bvh","input",tree_result,reference,reference_name);
                // 25% headroom over the known maximum, as a solver would size
                // the table from its previous rebuild.
                capacity=std::max(8,tree_result.max_neighbors+tree_result.max_neighbors/4);
                if(legacy) {
                    tree_result=run_query(exec,p,radii,device_bvh.ids,tree,bvh_build_ms,repeats);
                    report(s,"median_bvh","spatial",tree_result,reference,reference_name);
                }
            } else if(validate && m<=oracle_limit) {
                reference=brute_force(s,true);
                reference_name="all_pairs_periodic";
            }

            const double med=median(s.radii);
            for(double factor : {0.5,0.75,1.0,1.5}) {
                const double width=med*factor;
                for(DenseMode mode : modes) {
                    int retries=0, lanes=1;
                    Result dense_result=run_dense(exec,p,radii,width,periodic,capacity,mode,
                                                  repeats,validate && !reference.empty(),retries,
                                                  lanes);
                    report(s,mode_name(mode),"lanes="+std::to_string(lanes),dense_result,
                           reference,reference_name,width);
                    if(retries>0) std::cout << "# capacity retries=" << retries << '\n';
                }
                if(!legacy) continue;
                CellIndex cells;
                const double cell_build_ms=median_ms(repeats, [&] {
                    cells=build_cell_index(exec,p,width);
                });
                CellSearch cell{cells.sorted_ids,cells.occupied_keys,cells.offsets,
                                cells.occupied_count,cells.nx,cells.ny,cells.nz,
                                cells.hx,cells.hy,cells.hz};
                Result cell_result=run_query(exec,p,radii,input_order,cell,cell_build_ms,repeats);
                report(s,"sorted_cells","input",cell_result,reference,reference_name,width);
                cell_result=run_query(exec,p,radii,cells.sorted_ids,cell,cell_build_ms,repeats);
                report(s,"sorted_cells","spatial",cell_result,reference,reference_name,width);
            }
        }
    }
    Kokkos::finalize();
    return 0;
}
