#pragma once

#include "local_voronoi_retry.hpp"
#include "neighbor_search.hpp"

namespace local_voronoi {
namespace detail {
template<class StatusView> struct FailedOrderRows {
    StatusView status;
    KOKKOS_INLINE_FUNCTION bool operator()(int row) const {
        return status(row)!=success && status(row)!=invalid_candidate_count &&
               status(row)!=coincident_particles;
    }
};
}

// Experimental recovery after unsorted reconstruction. Sorts FAILED candidate
// rows IN PLACE, then rebuilds only those cells with the full reference workspace.
// No new persistent buffers or relaxed tolerances. A subsequent fill restores
// raw candidate order. Unsupported sort capacities leave failures explicit.
template<class VertexStorage=CachedVertexPositions, class Workspace=PlaneWorkspace<64,RelativeClippingTolerance,true>,
         class ExecSpace, class FaceCounts, class FaceIds, class FaceAreas,
         class FaceVertexCounts, class FaceVertices, class RowStatus>
void retry_table_order(ExecSpace exec, const neighbor_search::DenseCellSearch<ExecSpace>& search,
                       FaceCounts face_counts, FaceIds face_ids, FaceAreas face_areas,
                       FaceVertexCounts face_vertex_counts, FaceVertices face_vertices,
                       RowStatus row_status) {
    auto sorting=search;
    sorting.row_sort=neighbor_search::RowSort::team_bitonic_recompute;
    const detail::FailedOrderRows<RowStatus> selected{row_status};
    if (!neighbor_search::sort_rows_team(exec,sorting,selected)) return;
    const auto positions=search.positions;
    const auto counts=search.counts;
    const auto table=search.table;
    const auto reach=search.radii;
    const auto box=search.box;
    Kokkos::parallel_for("local_voronoi::retry_table_candidate_order",
        Kokkos::RangePolicy<ExecSpace>(exec,0,positions.extent(0)), KOKKOS_LAMBDA(int row) {
            if (!selected(row)) return;
            reconstruct_row<VertexStorage,PlaneWorkspace<max_planes,typename Workspace::tolerance,Workspace::early_stop>>(
                row,positions,detail::TableRowOffsets<decltype(counts)>{counts,row},
                detail::TableRowIds<decltype(table)>{table,row},reach(row),box,face_counts,
                face_ids,face_areas,face_vertex_counts,face_vertices,row_status);
            if (row_status(row)!=success) face_counts(row)=0;
        });
    exec.fence("local Voronoi table workspace retry");
}

template<class VertexStorage=CachedVertexPositions, class Workspace=PlaneWorkspace<64,RelativeClippingTolerance,true>,
         class ExecSpace, class FaceCounts, class FaceIds, class FaceAreas,
         class FaceCentroids, class Volumes, class CellCentroids, class RowStatus>
void retry_hydro_order(ExecSpace exec, const neighbor_search::DenseCellSearch<ExecSpace>& search,
                       FaceCounts face_counts, FaceIds face_ids, FaceAreas face_areas,
                       FaceCentroids face_centroids, Volumes volumes,
                       CellCentroids cell_centroids, RowStatus row_status) {
    // Unsorted clipping can retain numerically tiny coplanar faces even when
    // volume/moments pass. Retry their cells in reference order; never erase a
    // face or relax a predicate. Legitimate small faces remain in sorted output.
    Kokkos::parallel_for("local_voronoi::flag_order_sensitive_faces",
        Kokkos::RangePolicy<ExecSpace>(exec,0,search.n), KOKKOS_LAMBDA(int row) {
            if (row_status(row)!=success) return;
            double largest=0;
            for (int f=0;f<face_counts(row);++f) {
                double area2=0;
                for (int a=0;a<3;++a) area2+=face_areas(row,f,a)*face_areas(row,f,a);
                largest=Kokkos::fmax(largest,area2);
            }
            for (int f=0;f<face_counts(row);++f) {
                double area2=0;
                for (int a=0;a<3;++a) area2+=face_areas(row,f,a)*face_areas(row,f,a);
                if (area2<=largest*1.0e-24) row_status(row)=degenerate_geometry;
            }
        });
    auto sorting=search;
    sorting.row_sort=neighbor_search::RowSort::team_bitonic_recompute;
    const detail::FailedOrderRows<RowStatus> selected{row_status};
    if (!neighbor_search::sort_rows_team(exec,sorting,selected)) return;
    const auto positions=search.positions;
    const auto counts=search.counts;
    const auto table=search.table;
    const auto reach=search.radii;
    const auto box=search.box;
    Kokkos::parallel_for("local_voronoi::retry_compact_candidate_order",
        Kokkos::RangePolicy<ExecSpace>(exec,0,positions.extent(0)), KOKKOS_LAMBDA(int row) {
            if (!selected(row)) return;
            detail::HydroGeometry<Volumes,CellCentroids,FaceCentroids> geometry{
                volumes,cell_centroids,face_centroids,row};
            reconstruct_row<VertexStorage,PlaneWorkspace<max_planes,typename Workspace::tolerance,Workspace::early_stop>>(
                row,positions,detail::TableRowOffsets<decltype(counts)>{counts,row},
                detail::TableRowIds<decltype(table)>{table,row},reach(row),box,face_counts,
                face_ids,face_areas,detail::NoDiagnosticVertices{},detail::NoDiagnosticVertices{},
                row_status,geometry);
            if (row_status(row)!=success) face_counts(row)=0;
        });
    exec.fence("local Voronoi compact workspace retry");
}

} // namespace local_voronoi
