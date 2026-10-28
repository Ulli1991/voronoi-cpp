#pragma once
#include "local_voronoi.hpp"
#include <stdexcept>
#include <type_traits>

namespace local_voronoi {
// Cooperative clipping primitive: a team owns one cell; lanes classify and
// compact vertices. This primitive is not yet wired through the sequential
// topology updates in reconstruct_row.
template<class Exec,class Coordinates,class Counts,class Planes,class OutCoordinates,
         class OutCounts,class Flags,class Radii>
void classify_and_compact(Exec exec,Coordinates x,Counts counts,Planes planes,
                         OutCoordinates kept,OutCounts kept_counts,Flags outside,
                         Radii radius2) {
    const auto n=counts.extent(0);
    if(x.extent(0)!=n || x.extent(1)<max_vertices || x.extent(2)!=3 ||
       kept.extent(0)!=n || kept.extent(1)<max_vertices || kept.extent(2)!=3 ||
       planes.extent(0)!=n || kept_counts.extent(0)!=n || outside.extent(0)!=n ||
       outside.extent(1)<max_vertices || radius2.extent(0)!=n)
        throw std::invalid_argument("cooperative clipping extent mismatch");
    using Policy=Kokkos::TeamPolicy<Exec>;
    using Member=typename Policy::member_type;
    int lanes=1;
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Exec,Kokkos::Cuda>) lanes=32;
#endif
    Kokkos::parallel_for("local_voronoi::classify_and_compact",Policy(exec,n,lanes),
        KOKKOS_LAMBDA(const Member& team){
            const int row=team.league_rank();
            const int count=counts(row);
            // Invalid counts yield no usable compacted output.
            if(count<0 || count>max_vertices){
                Kokkos::single(Kokkos::PerTeam(team),[&](){kept_counts(row)=-1;radius2(row)=0;});
                return;
            }
            const auto plane=planes(row);
            double maximum=0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,count),
                [&](int v,double& r){
                    const Vec3 point{x(row,v,0),x(row,v,1),x(row,v,2)};
                    outside(row,v)=RelativeClippingTolerance::outside(plane.normal,point,plane.offset,0);
                    r=Kokkos::max(r,dot(point,point));
                },Kokkos::Max<double>(maximum));
            team.team_barrier();
            int total=0;
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(team,count),
                [&](int v,int& prefix,bool final){
                    const bool retain=!outside(row,v);
                    if(final && retain) for(int a=0;a<3;++a) kept(row,prefix,a)=x(row,v,a);
                    prefix+=retain;
                },total);
            Kokkos::single(Kokkos::PerTeam(team),[&](){kept_counts(row)=total;radius2(row)=count ? maximum : 0;});
        });
    exec.fence("cooperative classification complete");
}
}
