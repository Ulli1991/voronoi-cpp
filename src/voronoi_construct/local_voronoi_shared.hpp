#pragma once
#include "local_voronoi_hydro.hpp"
#include "team_cell_storage.hpp"
namespace local_voronoi {
// Serial-in-team baseline for validating scratch storage. Cooperative clipping
// must replace the sequential row routine before this is a warp implementation.
template<class VertexStorage = RecomputedVertexPositions,
         class Workspace = PlaneWorkspace<>,
         class ExecSpace, class Positions, class Counts, class Table,
         class ReachView, class FaceCounts, class FaceIds, class FaceAreas,
         class FaceCentroids, class Volumes, class CellCentroids, class RowStatus>
void reconstruct_hydro_shared(ExecSpace exec, Positions positions, Counts counts,
                       Table table, ReachView reach, Kokkos::Array<double,3> box,
                       FaceCounts face_counts, FaceIds face_ids, FaceAreas face_areas,
                       FaceCentroids face_centroids, Volumes volumes,
                       CellCentroids cell_centroids, RowStatus row_status) {
    const auto n=positions.extent(0);
    if (positions.extent(1)!=3 || counts.extent(0)!=n || table.extent(0)!=n ||
        reach.extent(0)!=n || face_counts.extent(0)!=n || row_status.extent(0)!=n ||
        face_ids.extent(0)!=n || face_ids.extent(1)<max_faces ||
        face_areas.extent(0)!=n || face_areas.extent(1)<max_faces || face_areas.extent(2)!=3 ||
        face_centroids.extent(0)!=n || face_centroids.extent(1)<max_faces ||
        face_centroids.extent(2)!=3 || volumes.extent(0)!=n ||
        cell_centroids.extent(0)!=n || cell_centroids.extent(1)!=3)
        throw std::invalid_argument("local_voronoi::reconstruct_hydro: incompatible view extents");
    const auto capacity=table.extent(1);
    using Policy=Kokkos::TeamPolicy<ExecSpace>;
    using Member=typename Policy::member_type;
    using Scratch=typename Member::scratch_memory_space;
    using State=TeamCellStorage<Workspace::plane_capacity,Scratch>;
    using Cache=TeamVertexPositions<Scratch>;
    const std::size_t bytes=State::bytes()+Cache::bytes();
    Kokkos::parallel_for("local_voronoi::shared_cell_baseline",
        Policy(exec,n,1).set_scratch_size(0,Kokkos::PerTeam(bytes)),
        KOKKOS_LAMBDA(const Member& team) {
            const int row=team.league_rank();
            State state(team.team_scratch(0));
            Cache cache(team.team_scratch(0));
            detail::HydroGeometry<Volumes,CellCentroids,FaceCentroids> geometry{
                volumes,cell_centroids,face_centroids,row};
            if (counts(row)<0 || static_cast<std::size_t>(counts(row))>capacity) {
                face_counts(row)=0;
                row_status(row)=geometry.finish(invalid_candidate_count);
                return;
            }
            reconstruct_row<Cache,Workspace>(row,positions,detail::TableRowOffsets<Counts>{counts,row},
                detail::TableRowIds<Table>{table,row},reach(row),box,face_counts,
                face_ids,face_areas,detail::NoDiagnosticVertices{},
                detail::NoDiagnosticVertices{},row_status,geometry,NoWorkspaceTrace{},cache,state);
            if (row_status(row)!=success) face_counts(row)=0;
        });
    exec.fence("local Voronoi hydro reconstruction");
}
}
