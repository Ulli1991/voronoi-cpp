#pragma once

#include "local_voronoi.hpp"

#include <stdexcept>

namespace local_voronoi {

// Additional status for a table row whose count exceeds its storage. No
// geometry is returned for that row; rebuild/grow the neighbor table first.
constexpr int invalid_candidate_count = 8;

namespace detail {
template<class Counts>
struct TableRowOffsets {
    Counts counts;
    int row;
    KOKKOS_INLINE_FUNCTION int operator()(int index) const {
        return index == row ? 0 : counts(row);
    }
};

template<class Table>
struct TableRowIds {
    Table table;
    int row;
    KOKKOS_INLINE_FUNCTION int operator()(int index) const {
        return table(row, index);
    }
};
} // namespace detail

// Direct fixed-capacity-table input, without allocating or copying a CSR
// graph. Positions, reach, counts and table IDs must all use the same index
// space. For DenseCellSearch these are search.positions, search.radii,
// search.counts and search.table: outputs are in SLOT order, and face IDs
// are slots too. search.order(slot) maps either back to the input particle ID.
// Use a completed rebuild with nearest_first=true to limit temporary geometry
// growth. Do not mutate the search buffers until this call completes.
// Output shapes are the same as reconstruct(). Geometry vertices are relative
// to the generating particle, including across periodic boundaries.
template<class VertexStorage = RecomputedVertexPositions,
         class Workspace = PlaneWorkspace<>,
         class ExecSpace, class Positions, class Counts, class Table,
         class ReachView, class FaceCounts, class FaceIds, class FaceAreas,
         class FaceVertexCounts, class FaceVertices, class RowStatus>
void reconstruct_table(ExecSpace exec, Positions positions, Counts counts,
                       Table table, ReachView reach, Kokkos::Array<double,3> box,
                       FaceCounts face_counts, FaceIds face_ids, FaceAreas face_areas,
                       FaceVertexCounts face_vertex_counts, FaceVertices face_vertices,
                       RowStatus row_status) {
    const auto n = positions.extent(0);
    if (positions.extent(1) != 3 || counts.extent(0) != n ||
        table.extent(0) != n || reach.extent(0) != n ||
        face_counts.extent(0) != n || row_status.extent(0) != n ||
        face_ids.extent(0) != n || face_ids.extent(1) < max_faces ||
        face_areas.extent(0) != n || face_areas.extent(1) < max_faces ||
        face_areas.extent(2) != 3 || face_vertex_counts.extent(0) != n ||
        face_vertex_counts.extent(1) < max_faces || face_vertices.extent(0) != n ||
        face_vertices.extent(1) < max_faces ||
        face_vertices.extent(2) < max_face_vertices || face_vertices.extent(3) != 3)
        throw std::invalid_argument("local_voronoi::reconstruct_table: incompatible view extents");
    const auto capacity = table.extent(1);
    Kokkos::parallel_for("local_voronoi::reconstruct_table",
        Kokkos::RangePolicy<ExecSpace>(exec,0,n), KOKKOS_LAMBDA(int row) {
            if (counts(row) < 0 || static_cast<std::size_t>(counts(row)) > capacity) {
                face_counts(row) = 0;
                row_status(row) = invalid_candidate_count;
                return;
            }
            reconstruct_row<VertexStorage,Workspace>(row,positions,detail::TableRowOffsets<Counts>{counts,row},
                detail::TableRowIds<Table>{table,row},reach(row),box,face_counts,
                face_ids,face_areas,face_vertex_counts,face_vertices,row_status);
            // Failed rows must not expose a partially written face list.
            if (row_status(row) != success) face_counts(row) = 0;
        });
    exec.fence("local Voronoi table reconstruction");
}

} // namespace local_voronoi
