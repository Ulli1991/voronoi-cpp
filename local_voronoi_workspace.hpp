#pragma once

#include "local_voronoi_table.hpp"

namespace local_voronoi {

// Profiling-only storage. Peaks include the next required entry on a capacity
// failure, not just successfully stored entries. Planes include the initial six.
struct WorkspaceUsage {
    int planes=0, vertices=0, hole_edges=0, faces=0, face_vertices=0;
    int candidates=0, vertex_tests=0, coordinate_reads=0, created_vertices=0;
    int intersection_solves=0, status=success;
};

namespace detail {
template<class UsageView>
struct WorkspaceTrace {
    UsageView output;
    WorkspaceUsage usage;
    KOKKOS_INLINE_FUNCTION void planes(int n) { usage.planes=Kokkos::max(usage.planes,n); }
    KOKKOS_INLINE_FUNCTION void vertices(int n) { usage.vertices=Kokkos::max(usage.vertices,n); }
    KOKKOS_INLINE_FUNCTION void hole(int n) { usage.hole_edges=Kokkos::max(usage.hole_edges,n); }
    KOKKOS_INLINE_FUNCTION void faces(int n) { usage.faces=Kokkos::max(usage.faces,n); }
    KOKKOS_INLINE_FUNCTION void face_vertices(int n) { usage.face_vertices=Kokkos::max(usage.face_vertices,n); }
    KOKKOS_INLINE_FUNCTION void candidate() { ++usage.candidates; }
    KOKKOS_INLINE_FUNCTION void vertex_test() { ++usage.vertex_tests; }
    KOKKOS_INLINE_FUNCTION void coordinate_read() { ++usage.coordinate_reads; }
    KOKKOS_INLINE_FUNCTION void created_vertex() { ++usage.created_vertices; }
    KOKKOS_INLINE_FUNCTION void finish(int row, int status, bool cached) {
        usage.status=status;
        usage.intersection_solves=usage.created_vertices+(cached ? 0 : usage.coordinate_reads);
        output(row)=usage;
    }
};

struct DiscardOutput {
    struct Value {
        template<class T> KOKKOS_INLINE_FUNCTION void operator=(T) const {}
    };
    template<class... Indices>
    KOKKOS_INLINE_FUNCTION Value operator()(Indices...) const { return {}; }
};

struct WorkspaceGeometry : DiagnosticGeometry {
    static constexpr bool store_vertices=false;
};
} // namespace detail

// Separate instrumented launch: no output geometry buffers are needed. It uses
// the identical clipping/face walk, but counts geometry work rather than timing
// it. Do not interpret this launch's runtime as production reconstruction time.
template<class VertexStorage = CachedVertexPositions, class Workspace = PlaneWorkspace<>,
         class ExecSpace, class Positions, class Counts, class Table,
         class ReachView, class UsageView>
void profile_workspace(ExecSpace exec, Positions positions, Counts counts,
                       Table table, ReachView reach, Kokkos::Array<double,3> box,
                       UsageView usage) {
    const auto n=positions.extent(0);
    if (positions.extent(1)!=3 || counts.extent(0)!=n || table.extent(0)!=n ||
        reach.extent(0)!=n || usage.extent(0)!=n)
        throw std::invalid_argument("local_voronoi::profile_workspace: incompatible view extents");
    const auto capacity=table.extent(1);
    Kokkos::parallel_for("local_voronoi::profile_workspace",
        Kokkos::RangePolicy<ExecSpace>(exec,0,n), KOKKOS_LAMBDA(int row) {
            detail::WorkspaceTrace<UsageView> trace{usage,{}};
            if (counts(row)<0 || static_cast<std::size_t>(counts(row))>capacity) {
                trace.finish(row,invalid_candidate_count,VertexStorage::caches_coordinates);
                return;
            }
            const detail::DiscardOutput discard{};
            reconstruct_row<VertexStorage,Workspace>(row,positions,
                detail::TableRowOffsets<Counts>{counts,row},detail::TableRowIds<Table>{table,row},
                reach(row),box,discard,discard,discard,discard,discard,discard,
                detail::WorkspaceGeometry{},trace);
        });
    exec.fence("local Voronoi workspace profile");
}
} // namespace local_voronoi
