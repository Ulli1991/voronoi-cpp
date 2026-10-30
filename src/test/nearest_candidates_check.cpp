#include "local_voronoi.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace {
using Exec=Kokkos::DefaultExecutionSpace;
using Memory=Exec::memory_space;

// Distances come from a few integers so that equal keys straddle batch
// boundaries; ids are shuffled so that row order says nothing about the key.
// Every finite or infinite candidate must come back exactly once in
// (distance, id) order, and a NaN candidate never.
template<int Batch>
int check(int size,std::mt19937_64& rng) {
    const double nan=std::numeric_limits<double>::quiet_NaN();
    const double inf=std::numeric_limits<double>::infinity();
    std::vector<int> row(size);
    std::iota(row.begin(),row.end(),0);
    std::shuffle(row.begin(),row.end(),rng);
    std::vector<local_voronoi::Vec3> offset(size);
    std::uniform_int_distribution<int> radius(1,3);
    for (int id=0;id<size;++id) {
        const int kind=id%11;
        const double x=kind==7 ? nan : (kind==9 ? inf : radius(rng));
        offset[id]={x,0,0};
    }
    std::vector<std::pair<double,int>> expected;
    for (int id=0;id<size;++id) {
        const double d2=local_voronoi::dot(offset[id],offset[id]);
        if (!std::isnan(d2)) expected.push_back({d2,id});
    }
    std::sort(expected.begin(),expected.end());
    // Select on the execution space reconstruction runs on, so a GPU build
    // checks the device code. At most size+1 ids are kept, enough to see a
    // candidate returned twice.
    Kokkos::View<int*,Memory> row_view("row",size),visited_view("visited",size+1),count("count",1);
    Kokkos::View<double*,Memory> x_view("x",size);
    auto row_host=Kokkos::create_mirror_view(row_view);
    auto x_host=Kokkos::create_mirror_view(x_view);
    for (int q=0;q<size;++q) { row_host(q)=row[q]; x_host(q)=offset[q].x; }
    Kokkos::deep_copy(row_view,row_host); Kokkos::deep_copy(x_view,x_host);
    Kokkos::parallel_for("nearest_candidates_check",Kokkos::RangePolicy<Exec>(0,1),KOKKOS_LAMBDA(int) {
        auto ids=[&](int q) { return row_view(q); };
        auto separation=[&](int id) { return local_voronoi::Vec3{x_view(id),0,0}; };
        local_voronoi::NearestCandidates<Batch> nearest;
        int visited=0,id=-1;
        while (visited<=size && nearest.next(0,size,ids,separation,id)) visited_view(visited++)=id;
        count(0)=visited;
    });
    auto visited=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},visited_view);
    auto visited_count=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},count);
    const std::size_t found=visited_count(0);
    int errors=found!=expected.size();
    for (std::size_t k=0;!errors && k<found;++k) errors=visited(k)!=expected[k].second;
    if (errors) std::cout << "batch=" << Batch << " size=" << size << " visited=" << found
                          << " expected=" << expected.size() << " order_error\n";
    return errors;
}

template<int Batch>
int check_sizes(std::mt19937_64& rng) {
    int errors=0;
    for (int size:{0,1,Batch-1,Batch,Batch+1,2*Batch,3*Batch,3*Batch+1,40})
        for (int repeat=0;repeat<20;++repeat) errors+=check<Batch>(std::max(size,0),rng);
    return errors;
}
} // namespace

int main(int argc,char** argv) {
    Kokkos::initialize(argc,argv);
    int errors=0;
    {
        std::mt19937_64 rng(20261028);
        errors+=check_sizes<1>(rng);
        errors+=check_sizes<3>(rng);
        errors+=check_sizes<4>(rng);
        errors+=check_sizes<8>(rng);
        errors+=check_sizes<32>(rng);
    }
    Kokkos::finalize();
    std::cout << "nearest_candidates_errors=" << errors << '\n';
    return errors ? 1 : 0;
}
