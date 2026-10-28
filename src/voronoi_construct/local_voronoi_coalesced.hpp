#pragma once
#include "local_voronoi_retry.hpp"
namespace local_voronoi {
// Explicitly coalesced coordinate cache. First index is contiguous across rows.
// Allocate once outside timed calls; no hidden allocation or host status reads.
template<class Memory>
struct CoalescedVertexPositions {
    static constexpr bool caches_coordinates=true;
    Kokkos::View<double***,Kokkos::LayoutLeft,Memory> coordinates;
    int row=0;
    KOKKOS_INLINE_FUNCTION void set(int i,Vec3 x) {
        coordinates(row,i,0)=x.x; coordinates(row,i,1)=x.y; coordinates(row,i,2)=x.z;
    }
    KOKKOS_INLINE_FUNCTION void initialize(int i,double reach) {
        set(i,{(i&1)?-reach:reach,(i&2)?-reach:reach,(i&4)?-reach:reach});
    }
    KOKKOS_INLINE_FUNCTION void keep(int dst,int src) {
        for (int a=0;a<3;++a) coordinates(row,dst,a)=coordinates(row,src,a);
    }
    template<class Tolerance,class Planes>
    KOKKOS_INLINE_FUNCTION bool position(int i,const VertexPlanes&,const Planes&,Vec3& x) const {
        x={coordinates(row,i,0),coordinates(row,i,1),coordinates(row,i,2)}; return true;
    }
};
// First experiment relocates only cached coordinates; planes/topology retain
// their reference storage. Costs 1536 additional persistent bytes per row.
template<class Exec,class Positions,class Counts,class Table,class Reach,
         class FaceCounts,class Ids,class Areas,class FaceCenters,class Volumes,
         class Centers,class Status>
void reconstruct_hydro_coalesced(Exec exec,Positions p,Counts counts,Table table,
    Reach reach,Kokkos::Array<double,3> box,FaceCounts faces,Ids ids,Areas areas,
    FaceCenters face_centers,Volumes volumes,Centers centers,Status status,
    CoalescedVertexPositions<typename Exec::memory_space> cache) {
    if(cache.coordinates.extent(0)==0 || cache.coordinates.extent(1)!=max_vertices || cache.coordinates.extent(2)!=3)
        throw std::invalid_argument("coalesced cache extent mismatch");
    const auto capacity=table.extent(1);
    const int batch_size=static_cast<int>(cache.coordinates.extent(0));
    for(int begin=0;begin<static_cast<int>(p.extent(0));begin+=batch_size){
    const int end=Kokkos::min(begin+batch_size,static_cast<int>(p.extent(0)));
    Kokkos::parallel_for("local_voronoi::coalesced_coordinates",Kokkos::RangePolicy<Exec>(exec,begin,end),KOKKOS_LAMBDA(int row){
        detail::HydroGeometry<Volumes,Centers,FaceCenters> geometry{volumes,centers,face_centers,row};
        if(counts(row)<0 || static_cast<std::size_t>(counts(row))>capacity){
            faces(row)=0; status(row)=geometry.finish(invalid_candidate_count); return;
        }
        auto row_cache=cache; row_cache.row=row-begin;
        reconstruct_row<decltype(cache),PlaneWorkspace<64,RelativeClippingTolerance,true>>(
            row,p,detail::TableRowOffsets<Counts>{counts,row},detail::TableRowIds<Table>{table,row},reach(row),box,
            faces,ids,areas,detail::NoDiagnosticVertices{},detail::NoDiagnosticVertices{},status,geometry,NoWorkspaceTrace{},row_cache);
        if(status(row)!=success) faces(row)=0;
    });
    } // Ordered launches on the same execution instance safely reuse the cache.
    Kokkos::parallel_for("local_voronoi::coalesced_coordinate_retry",Kokkos::RangePolicy<Exec>(exec,0,p.extent(0)),KOKKOS_LAMBDA(int row){
        if(status(row)==plane_capacity){
            detail::HydroGeometry<Volumes,Centers,FaceCenters> retry_geometry{volumes,centers,face_centers,row};
            reconstruct_row<CachedVertexPositions,PlaneWorkspace<max_planes,RelativeClippingTolerance,true>>(
                row,p,detail::TableRowOffsets<Counts>{counts,row},detail::TableRowIds<Table>{table,row},reach(row),box,
                faces,ids,areas,detail::NoDiagnosticVertices{},detail::NoDiagnosticVertices{},status,retry_geometry);
        }
        if(status(row)!=success) faces(row)=0;
    });
    exec.fence("coalesced coordinate reconstruction");
}
}
