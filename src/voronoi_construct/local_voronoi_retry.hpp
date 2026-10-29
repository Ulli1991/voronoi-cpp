#pragma once

#include "local_voronoi_hydro.hpp"

namespace local_voronoi {

// Smaller per-cell plane storage, followed by a device-side retry only for
// rows that exhausted it. No extra buffers, allocation or host status read.
// Other geometry failures are preserved. The fallback is the existing full
// workspace, not relaxed tolerances or truncated geometry. Both launches finish
// before return; timings should include both, even if no row needs a retry.
template<class VertexStorage = CachedVertexPositions, class SmallWorkspace = PlaneWorkspace<64>,
         class ExecSpace, class Positions, class Counts, class Table,
         class ReachView, class FaceCounts, class FaceIds, class FaceAreas,
         class FaceVertexCounts, class FaceVertices, class RowStatus>
void reconstruct_table_retry(ExecSpace exec, Positions positions, Counts counts,
                       Table table, ReachView reach, Kokkos::Array<double,3> box,
                       FaceCounts face_counts, FaceIds face_ids, FaceAreas face_areas,
                       FaceVertexCounts face_vertex_counts, FaceVertices face_vertices,
                       RowStatus row_status) {
    static_assert(SmallWorkspace::plane_capacity < max_planes,
                  "retry path requires a smaller initial plane workspace");
    reconstruct_table<VertexStorage,SmallWorkspace>(exec,positions,counts,table,reach,
        box,face_counts,face_ids,face_areas,face_vertex_counts,face_vertices,row_status);
    Kokkos::parallel_for("local_voronoi::retry_table_plane_capacity",
        Kokkos::RangePolicy<ExecSpace>(exec,0,positions.extent(0)), KOKKOS_LAMBDA(int row) {
            if (row_status(row)!=plane_capacity) return;
            reconstruct_row<VertexStorage,PlaneWorkspace<max_planes,typename SmallWorkspace::tolerance,SmallWorkspace::early_stop,SmallWorkspace::direct_face_links>>(
                row,positions,detail::TableRowOffsets<Counts>{counts,row},
                detail::TableRowIds<Table>{table,row},reach(row),box,face_counts,
                face_ids,face_areas,face_vertex_counts,face_vertices,row_status);
            if (row_status(row)!=success) face_counts(row)=0;
        });
    exec.fence("local Voronoi table workspace retry");
}

template<class VertexStorage = CachedVertexPositions, class SmallWorkspace = PlaneWorkspace<64>,
         class ExecSpace, class Positions, class Counts, class Table,
         class ReachView, class FaceCounts, class FaceIds, class FaceAreas,
         class FaceCentroids, class Volumes, class CellCentroids, class RowStatus>
void reconstruct_hydro_retry(ExecSpace exec, Positions positions, Counts counts,
                       Table table, ReachView reach, Kokkos::Array<double,3> box,
                       FaceCounts face_counts, FaceIds face_ids, FaceAreas face_areas,
                       FaceCentroids face_centroids, Volumes volumes,
                       CellCentroids cell_centroids, RowStatus row_status) {
    static_assert(SmallWorkspace::plane_capacity < max_planes,
                  "retry path requires a smaller initial plane workspace");
    reconstruct_hydro<VertexStorage,SmallWorkspace>(exec,positions,counts,table,reach,
        box,face_counts,face_ids,face_areas,face_centroids,volumes,cell_centroids,row_status);
    Kokkos::parallel_for("local_voronoi::retry_compact_plane_capacity",
        Kokkos::RangePolicy<ExecSpace>(exec,0,positions.extent(0)), KOKKOS_LAMBDA(int row) {
            if (row_status(row)!=plane_capacity) return;
            detail::HydroGeometry<Volumes,CellCentroids,FaceCentroids> geometry{
                volumes,cell_centroids,face_centroids,row};
            reconstruct_row<VertexStorage,PlaneWorkspace<max_planes,typename SmallWorkspace::tolerance,SmallWorkspace::early_stop,SmallWorkspace::direct_face_links>>(
                row,positions,detail::TableRowOffsets<Counts>{counts,row},
                detail::TableRowIds<Table>{table,row},reach(row),box,face_counts,
                face_ids,face_areas,detail::NoDiagnosticVertices{},detail::NoDiagnosticVertices{},
                row_status,geometry);
            if (row_status(row)!=success) face_counts(row)=0;
        });
    exec.fence("local Voronoi compact workspace retry");
}
} // namespace local_voronoi
