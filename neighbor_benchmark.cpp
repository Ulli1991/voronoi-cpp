#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

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
#include <string>
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
void search_count(const Exec& exec, Positions p, ScalarView radii, Search search,
                  IntView counts, IntView checks, IntView index_tests) {
    const int n = static_cast<int>(p.extent(0));
    Kokkos::parallel_for("neighbor_count", Kokkos::RangePolicy<Exec>(exec, 0, n),
        KOKKOS_LAMBDA(int i) {
            int hits = 0, tested = 0, broad_tests = 0;
            search.count(i, p, radii(i), hits, tested, broad_tests);
            counts(i) = hits;
            checks(i) = tested;
            index_tests(i) = broad_tests;
        });
}

template<class Search>
void search_fill(const Exec& exec, Positions p, ScalarView radii, Search search,
                 IntView offsets, IntView ids) {
    const int n = static_cast<int>(p.extent(0));
    Kokkos::parallel_for("neighbor_fill", Kokkos::RangePolicy<Exec>(exec, 0, n),
        KOKKOS_LAMBDA(int i) {
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

struct Result {
    double build_ms;
    double query_ms;
    long long distance_checks;
    long long index_tests;
    long long edges;
    int max_neighbors;
    std::vector<std::vector<int>> adjacency;
};

template<class Search>
Result run_query(const Exec& exec, Positions p, ScalarView radii, Search search,
                 double build_ms) {
    const int n = static_cast<int>(p.extent(0));
    IntView counts("neighbor_counts", n), checks("distance_checks", n);
    IntView index_tests("index_tests", n), offsets("neighbor_offsets", n+1);
    Kokkos::Timer timer;
    search_count(exec, p, radii, search, counts, checks, index_tests);
    int total=0;
    Kokkos::parallel_scan("neighbor_scan", Kokkos::RangePolicy<Exec>(exec, 0, n),
        KOKKOS_LAMBDA(int i, int& update, bool final) {
            if(final) offsets(i)=update;
            update += counts(i);
        }, total);
    Kokkos::parallel_for("neighbor_scan_end", Kokkos::RangePolicy<Exec>(exec, 0, 1),
        KOKKOS_LAMBDA(int) { offsets(n)=total; });
    IntView ids("neighbor_ids", total);
    search_fill(exec, p, radii, search, offsets, ids);
    exec.fence("neighbor CSR construction");
    const double query_ms=timer.seconds()*1.0e3;

    auto hc=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},counts);
    auto hcheck=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},checks);
    auto hindex=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},index_tests);
    auto hoff=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},offsets);
    auto hids=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},ids);
    Result r{build_ms,query_ms,0,0,total,0,{}};
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

std::vector<std::vector<int>> brute_force(const PointSet& s) {
    const int n=static_cast<int>(s.points.size());
    std::vector<std::vector<int>> out(n);
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) if(i!=j) {
        double d2=0.; for(int a=0;a<3;++a) { const double d=s.points[i][a]-s.points[j][a]; d2+=d*d; }
        if(d2<=s.radii[i]*s.radii[i]) out[i].push_back(j);
    }
    return out;
}

double median(std::vector<double> v) {
    std::nth_element(v.begin(),v.begin()+v.size()/2,v.end());
    return v[v.size()/2];
}

void report(const PointSet& s, const std::string& method, const Result& r,
            const std::vector<std::vector<int>>& reference, double cell_width=0.0) {
    int bad_rows=0;
    for(std::size_t i=0;i<reference.size();++i) if(r.adjacency[i]!=reference[i]) ++bad_rows;
    const double n=s.points.size();
    std::cout << s.name << ',' << method << ',' << std::fixed << std::setprecision(4)
              << n << ',' << cell_width << ',' << r.build_ms << ',' << r.query_ms << ','
              << r.index_tests/n << ',' << r.distance_checks/n << ','
              << static_cast<double>(r.edges)/n << ','
              << r.max_neighbors << ',' << bad_rows << '\n';
}

} // namespace

int main(int argc, char** argv) {
    int n=4096;
    if(argc>1) n=std::max(64,std::atoi(argv[1]));
    Kokkos::initialize(argc,argv);
    {
        const Exec exec{};
        std::cout << "distribution,method,n,cell_width,build_ms,query_ms,index_tests_per_particle,point_distance_checks_per_particle,mean_neighbors,max_neighbors,mismatched_rows\n";
        const std::uint64_t seed=20261027;
        std::vector<PointSet> sets;
        sets.push_back(make_uniform(n,seed));
        sets.push_back(make_jittered(n,seed+1));
        sets.push_back(make_clustered(n,seed+2));
        sets.push_back(make_gradient(n,seed+3));
        for(const auto& s:sets) {
            const int m=static_cast<int>(s.points.size());
            Positions p=copy_positions(s); ScalarView radii=copy_radii(s);
            const auto reference=brute_force(s);

            Kokkos::Timer bt;
            HostBvh host_bvh=make_host_bvh(s.points);
            DeviceBvh device_bvh=copy_bvh(host_bvh);
            exec.fence("BVH setup");
            const double bvh_build_ms=bt.seconds()*1.0e3;
            BvhSearch tree{device_bvh.nodes,device_bvh.ids};
            Result tree_result=run_query(exec,p,radii,tree,bvh_build_ms);
            report(s,"median_bvh",tree_result,reference);

            const double med=median(s.radii);
            for(double factor : {0.5,1.0,2.0}) {
                const double width=med*factor;
                Kokkos::Timer ct;
                CellIndex cells=build_cell_index(exec,p,width);
                exec.fence("cell setup");
                const double cell_build_ms=ct.seconds()*1.0e3;
                CellSearch cell{cells.sorted_ids,cells.occupied_keys,cells.offsets,
                                cells.occupied_count,cells.nx,cells.ny,cells.nz,
                                cells.hx,cells.hy,cells.hz};
                Result cell_result=run_query(exec,p,radii,cell,cell_build_ms);
                report(s,"sorted_cells",cell_result,reference,width);
            }
            (void)m;
        }
    }
    Kokkos::finalize();
    return 0;
}
