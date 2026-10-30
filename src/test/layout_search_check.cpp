#include "neighbor_search.hpp"

#include <iostream>
#include <numeric>

namespace {
using Exec=Kokkos::DefaultExecutionSpace;
using Memory=Exec::memory_space;
template<class View> auto snapshot(View v) {
    auto h=Kokkos::create_mirror(v);
    Kokkos::deep_copy(h,v);
    return h;
}
int check(bool periodic, int lanes, bool nearest, bool morton) {
    constexpr int n=64;
    Kokkos::View<double*[3],Memory> positions("positions",n);
    Kokkos::View<double*,Memory> radii("radii",n);
    auto input=Kokkos::create_mirror(positions);
    for (int i=0;i<n;++i) {
        const int grid=n-1-i;
        input(i,0)=(grid/16+0.5)/4;
        input(i,1)=((grid/4)%4+0.5)/4;
        input(i,2)=(grid%4+0.5)/4;
    }
    Kokkos::deep_copy(positions,input);
    Kokkos::deep_copy(radii,0.49);
    neighbor_search::DenseCellSearch<Exec> search;
    search.row_sort=neighbor_search::RowSort::team_bitonic_recompute;
    search.morton_cells=morton;
    int errors=0;
    // Reconfigure the SAME search, including anisotropic/non-power-of-two grids.
    for (const double width : {0.125,0.3,0.5}) {
        const Kokkos::Array<double,3> box{1,1.25,1.5};
        neighbor_search::configure(search,n,box,width,periodic,1,nearest,lanes);
        if (neighbor_search::rebuild(Exec{},search,positions,radii)==0) ++errors;
        if (search.distance2.is_allocated()) ++errors;
        auto order=snapshot(search.order), counts=snapshot(search.counts);
        auto table=snapshot(search.table); auto p=snapshot(search.positions);
        std::vector<bool> seen(n,false);
        std::uint64_t previous_key=0;
        for (int i=0;i<n;++i) {
            const int id=order(i);
            if (id<0 || id>=n || seen[id]) { ++errors; continue; }
            seen[id]=true;
            for (int a=0;a<3;++a) if (p(i,a)!=input(id,a)) ++errors;
            if (morton) {
                const unsigned x=static_cast<unsigned>(p(i,0)/search.hx);
                const unsigned y=static_cast<unsigned>(p(i,1)/search.hy);
                const unsigned z=static_cast<unsigned>(p(i,2)/search.hz);
                const auto key=neighbor_search::detail::morton_key(x,y,z);
                if (i && key<previous_key) ++errors;
                previous_key=key;
            }
            std::vector<std::pair<double,int>> expected;
            for (int j=0;j<n;++j) {
                if (j==i) continue;
                double d2=0;
                for (int a=0;a<3;++a) {
                    double d=p(j,a)-p(i,a);
                    if (periodic) d-=box[a]*std::nearbyint(d/box[a]);
                    d2+=d*d;
                }
                if (d2<=0.49*0.49) expected.emplace_back(d2,j);
            }
            std::sort(expected.begin(),expected.end());
            std::vector<int> want,actual;
            for (const auto& entry:expected) want.push_back(entry.second);
            for (int q=0;q<counts(i);++q) actual.push_back(table(i,q));
            if (!nearest) {
                std::sort(want.begin(),want.end());
                std::sort(actual.begin(),actual.end());
            }
            if (actual!=want) ++errors;
        }
        neighbor_search::rebuild(Exec{},search,positions,radii);
        auto again=snapshot(search.table);
        for (int i=0;i<n;++i)
            for (int q=0;q<counts(i);++q)
                if (again(i,q)!=table(i,q)) ++errors;
    }
    std::cout << "periodic=" << periodic << " lanes=" << lanes << " nearest=" << nearest
              << " morton=" << morton << " errors=" << errors << '\n';
    return errors;
}
}
int main(int argc,char** argv) {
    Kokkos::initialize(argc,argv);
    int errors=0;
    for (bool periodic:{false,true})
        for (int lanes:{1,4})
            for (bool nearest:{false,true})
                for (bool morton:{false,true}) errors+=check(periodic,lanes,nearest,morton);
    Kokkos::finalize();
    return errors ? 2 : 0;
}
