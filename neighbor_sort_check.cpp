#include "neighbor_search.hpp"

#include <iostream>
#include <random>
#include <utility>
#include <vector>

namespace {
using Exec=Kokkos::DefaultExecutionSpace;
template<class View> auto snapshot(View view) {
    // Always copy: create_mirror_view may alias on host execution spaces.
    auto copy=Kokkos::create_mirror(view);
    Kokkos::deep_copy(copy,view);
    return copy;
}

int check_capacity(int capacity) {
    constexpr int rows=12;
    neighbor_search::DenseCellSearch<Exec> search;
    neighbor_search::configure(search,rows,{1,1,1},0.25,false,capacity,true);
    auto ids=Kokkos::create_mirror(search.table);
    auto distances=Kokkos::create_mirror(search.distance2);
    auto counts=Kokkos::create_mirror(search.counts);
    const int lengths[rows]={0,1,2,3,7,31,32,33,capacity/2,capacity-1,capacity,capacity+7};
    std::mt19937 rng(20261027);
    std::vector<std::vector<std::pair<double,int>>> expected(rows);
    for (int row=0;row<rows;++row) {
        counts(row)=std::max(0,lengths[row]);
        const int stored=std::min(counts(row),capacity);
        for (int q=0;q<capacity;++q) {
            // Repeated exact distances exercise deterministic ID tie-breaking.
            const double distance=row%3==0 ? 1.0 :
                (row%3==1 ? (capacity-q)*0.125 : (rng()%17)*0.125);
            ids(row,q)=q<stored ? row*std::max(1,capacity)+capacity-q : -17;
            distances(row,q)=q<stored ? distance : -19.0;
            if (q<stored) expected[row].emplace_back(distance,ids(row,q));
        }
        std::sort(expected[row].begin(),expected[row].end());
    }
    Kokkos::deep_copy(search.table,ids);
    Kokkos::deep_copy(search.distance2,distances);
    Kokkos::deep_copy(search.counts,counts);
    const bool supported=neighbor_search::sort_rows_team(Exec{},search);
    Exec{}.fence("sort check completion");
    auto result_ids=snapshot(search.table);
    auto result_distances=snapshot(search.distance2);
    auto result_counts=snapshot(search.counts);
    int errors=supported!=(capacity<=2048);
    for (int row=0;row<rows;++row) {
        if (result_counts(row)!=counts(row)) ++errors;
        const int stored=std::min(counts(row),capacity);
        for (int q=0;q<capacity;++q) {
            const bool sorted=supported && q<stored;
            const int want_id=sorted ? expected[row][q].second : ids(row,q);
            const double want_distance=sorted ? expected[row][q].first : distances(row,q);
            if (result_ids(row,q)!=want_id || result_distances(row,q)!=want_distance) ++errors;
        }
    }
    std::cout << "backend=" << Exec::name() << " sort_capacity=" << capacity
              << " supported=" << supported << " errors=" << errors << '\n';
    return errors;
}

int check_fallback() {
    // Capacity above the scratch limit must dispatch to the retained insertion
    // sort, rather than leave the row unsorted or attempt excessive scratch.
    using Memory=Exec::memory_space;
    Kokkos::View<double*[3],Memory> positions("positions",5);
    Kokkos::View<double*,Memory> radii("radii",5);
    auto p=Kokkos::create_mirror(positions);
    for (int i=0;i<5;++i) {
        p(i,0)=0.25+0.0625*i;
        p(i,1)=p(i,2)=0.5;
    }
    Kokkos::deep_copy(positions,p);
    Kokkos::deep_copy(radii,0.4);
    neighbor_search::DenseCellSearch<Exec> search;
    neighbor_search::configure(search,5,{1,1,1},0.25,false,2049,true);
    neighbor_search::rebuild(Exec{},search,positions,radii);
    auto expected_ids=snapshot(search.table);
    auto expected_distances=snapshot(search.distance2);
    search.row_sort=neighbor_search::RowSort::team_bitonic;
    neighbor_search::rebuild(Exec{},search,positions,radii);
    auto ids=snapshot(search.table);
    auto counts=snapshot(search.counts);
    auto distances=snapshot(search.distance2);
    int errors=0;
    for (int row=0;row<5;++row)
        for (int q=0;q<counts(row);++q)
            if (ids(row,q)!=expected_ids(row,q) ||
                distances(row,q)!=expected_distances(row,q)) ++errors;
    std::cout << "backend=" << Exec::name() << " oversized_fallback_errors=" << errors << '\n';
    return errors;
}
} // namespace

int main(int argc,char** argv) {
    Kokkos::initialize(argc,argv);
    int errors=0;
    {
        for (int capacity:{0,1,7,33,128,257,513,1025,2048,2049}) errors+=check_capacity(capacity);
        errors+=check_fallback();
    }
    Kokkos::finalize();
    return errors ? 2 : 0;
}
