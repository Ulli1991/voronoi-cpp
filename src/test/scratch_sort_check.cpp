#include "neighbor_search.hpp"

#include <iostream>

namespace {
using Exec = Kokkos::DefaultExecutionSpace;
using Memory = Exec::memory_space;

template<class View> auto snapshot(View v) {
    auto h = Kokkos::create_mirror(v);
    Kokkos::deep_copy(h,v);
    return h;
}

int check(bool periodic, int lanes, int capacity, double width) {
    constexpr int n=64;
    Kokkos::View<double*[3],Memory> positions("positions",n);
    Kokkos::View<double*,Memory> radii("radii",n);
    auto p=Kokkos::create_mirror(positions);
    for (int i=0;i<n;++i) {
        // Binary-exact coordinates give repeated distances and boundary pairs.
        p(i,0)=(i/16+0.5)/4;
        p(i,1)=((i/4)%4+0.5)/4;
        p(i,2)=(i%4+0.5)/4;
    }
    Kokkos::deep_copy(positions,p);
    Kokkos::deep_copy(radii,0.49);
    neighbor_search::DenseCellSearch<Exec> reference, scratch;
    scratch.row_sort=neighbor_search::RowSort::team_bitonic_recompute;
    neighbor_search::configure(reference,n,{1,1,1},width,periodic,capacity,true,lanes);
    neighbor_search::configure(scratch,n,{1,1,1},width,periodic,capacity,true,lanes);
    const int reference_retries=neighbor_search::rebuild(Exec{},reference,positions,radii);
    const int scratch_retries=neighbor_search::rebuild(Exec{},scratch,positions,radii);
    const auto ids=snapshot(scratch.table), expected=snapshot(reference.table);
    const auto counts=snapshot(scratch.counts), expected_counts=snapshot(reference.counts);
    const auto order=snapshot(scratch.order), expected_order=snapshot(reference.order);
    int errors=reference_retries!=scratch_retries;
    if (scratch.capacity<=2048 && scratch.distance2.is_allocated()) ++errors;
    if (scratch.capacity>2048 && !scratch.distance2.is_allocated()) ++errors;
    for (int row=0;row<n;++row) {
        if (order(row)!=expected_order(row) || counts(row)!=expected_counts(row)) ++errors;
        for (int q=0;q<counts(row);++q)
            if (ids(row,q)!=expected(row,q)) ++errors;
    }
    // The same allocation must also support a repeated rebuild.
    neighbor_search::rebuild(Exec{},scratch,positions,radii);
    const auto repeated=snapshot(scratch.table);
    for (int row=0;row<n;++row)
        for (int q=0;q<counts(row);++q)
            if (repeated(row,q)!=expected(row,q)) ++errors;
    std::cout << "periodic=" << periodic << " lanes=" << lanes
              << " capacity=" << capacity << " width=" << width
              << " retries=" << scratch_retries << " errors=" << errors << '\n';
    return errors;
}
}

int main(int argc,char** argv) {
    Kokkos::initialize(argc,argv);
    int errors=0;
    for (bool periodic : {false,true})
        for (int lanes : {1,4})
            for (int capacity : {1,250,2049})
                for (double width : {0.125,0.5})
                    errors+=check(periodic,lanes,capacity,width);
    Kokkos::finalize();
    return errors ? 2 : 0;
}
