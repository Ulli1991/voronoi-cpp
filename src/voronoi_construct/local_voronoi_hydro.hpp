#pragma once

#include "local_voronoi_table.hpp"

namespace local_voronoi {
namespace detail {

// Fan triangles integrate each face and its tetrahedra with the generator.
// Only streaming accumulators are needed; no polygon or vertex output buffer.
template<class Volumes, class CellCentroids, class FaceCentroids>
struct HydroGeometry {
    static constexpr bool store_vertices = false;
    Volumes volumes;
    CellCentroids cell_centroids;
    FaceCentroids face_centroids;
    int row;
    Vec3 first{0,0,0}, previous{0,0,0}, normal{0,0,0};
    Vec3 face_moment{0,0,0}, cell_moment{0,0,0};
    double face_weight=0, volume=0;
    int count=0;

    KOKKOS_INLINE_FUNCTION void begin_face(Vec3 outward) {
        // Normal magnitude cancels when dividing the face first moment.
        normal=outward;
        face_moment={0,0,0};
        face_weight=0;
        count=0;
    }

    KOKKOS_INLINE_FUNCTION void vertex(Vec3 x) {
        if (count==0) first=x;
        if (count>=2) {
            const Vec3 sum=add(add(first,previous),x);
            const double weight=dot(cross(sub(previous,first),sub(x,first)),normal);
            face_weight+=weight;
            face_moment=add(face_moment,mul(sum,weight/3.0));
            const double tetra_volume=dot(first,cross(previous,x))/6.0;
            volume+=tetra_volume;
            cell_moment=add(cell_moment,mul(sum,tetra_volume/4.0));
        }
        previous=x;
        ++count;
    }

    KOKKOS_INLINE_FUNCTION int end_face(int face) {
        if (count<3 || !(face_weight>0) || !Kokkos::isfinite(face_weight))
            return degenerate_geometry;
        const Vec3 centroid=mul(face_moment,1.0/face_weight);
        if (!Kokkos::isfinite(centroid.x) || !Kokkos::isfinite(centroid.y) ||
            !Kokkos::isfinite(centroid.z)) return degenerate_geometry;
        face_centroids(row,face,0)=centroid.x;
        face_centroids(row,face,1)=centroid.y;
        face_centroids(row,face,2)=centroid.z;
        return success;
    }

    KOKKOS_INLINE_FUNCTION int finish(int status) {
        if (status==success && (!(volume>0) || !Kokkos::isfinite(volume)))
            status=degenerate_geometry;
        Vec3 centroid{0,0,0};
        if (status==success) {
            centroid=mul(cell_moment,1.0/volume);
            if (!Kokkos::isfinite(centroid.x) || !Kokkos::isfinite(centroid.y) ||
                !Kokkos::isfinite(centroid.z)) status=degenerate_geometry;
        }
        volumes(row)=status==success ? volume : 0.0;
        cell_centroids(row,0)=status==success ? centroid.x : 0.0;
        cell_centroids(row,1)=status==success ? centroid.y : 0.0;
        cell_centroids(row,2)=status==success ? centroid.z : 0.0;
        return status;
    }
};

struct NoDiagnosticVertices {};
} // namespace detail

// Compact hydro geometry from a completed nearest-first candidate table.
// Output shapes: counts/status/volumes [N], IDs [N,max_faces], area vectors
// and face centroids [N,max_faces,3], cell centroids [N,3]. Centroids are
// offsets from the generator, in its local periodic image. IDs are in the
// input index space (DenseCellSearch uses slots). Only successful rows and
// faces below face_counts(row) are usable. Failure zeros volume and centroid.
// No allocation, CSR conversion, or diagnostic vertex storage is performed.
template<class VertexStorage = RecomputedVertexPositions,
         class Workspace = PlaneWorkspace<>,
         class ExecSpace, class Positions, class Counts, class Table,
         class ReachView, class FaceCounts, class FaceIds, class FaceAreas,
         class FaceCentroids, class Volumes, class CellCentroids, class RowStatus>
void reconstruct_hydro(ExecSpace exec, Positions positions, Counts counts,
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
    Kokkos::parallel_for("local_voronoi::reconstruct_hydro",
        Kokkos::RangePolicy<ExecSpace>(exec,0,n), KOKKOS_LAMBDA(int row) {
            detail::HydroGeometry<Volumes,CellCentroids,FaceCentroids> geometry{
                volumes,cell_centroids,face_centroids,row};
            if (counts(row)<0 || static_cast<std::size_t>(counts(row))>capacity) {
                face_counts(row)=0;
                row_status(row)=geometry.finish(invalid_candidate_count);
                return;
            }
            reconstruct_row<VertexStorage,Workspace>(row,positions,detail::TableRowOffsets<Counts>{counts,row},
                detail::TableRowIds<Table>{table,row},reach(row),box,face_counts,
                face_ids,face_areas,detail::NoDiagnosticVertices{},
                detail::NoDiagnosticVertices{},row_status,geometry);
            if (row_status(row)!=success) face_counts(row)=0;
        });
    exec.fence("local Voronoi hydro reconstruction");
}
} // namespace local_voronoi
