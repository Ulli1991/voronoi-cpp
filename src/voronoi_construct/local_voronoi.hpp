#pragma once

#include <Kokkos_Core.hpp>
#include "fixed_point_positions.hpp"

#include <cstdint>
#include <type_traits>

namespace local_voronoi {

// Fixed-size, allocation-free workspace for one cell. Capacities are explicit
// so a device kernel can report overflow instead of truncating a polyhedron.
constexpr int max_planes = 160;
constexpr int max_vertices = 64;
constexpr int max_hole_edges = 64;
constexpr int max_faces = 32;
constexpr int max_face_vertices = 32;
constexpr int n_box_planes = 6;

enum Status : int {
    success = 0,
    uncertified = 1,
    coincident_particles = 2,
    plane_capacity = 3,
    vertex_capacity = 4,
    hole_capacity = 5,
    face_capacity = 6,
    degenerate_geometry = 7,
};

struct Vec3 { double x, y, z; };
struct Plane { Vec3 normal; double offset; int neighbour; };
struct VertexPlanes { std::uint16_t p[3]; };
struct Edge { std::uint16_t a, b; };

// Preserve the original numerical path for comparisons.
struct LegacyClippingTolerance {
    static constexpr bool stable_intersections=false;
    KOKKOS_INLINE_FUNCTION static bool coincident(double d2) { return d2<=1.0e-28; }
    KOKKOS_INLINE_FUNCTION static double plane_tolerance(double d2, double reach) {
        return 1.0e-12*(1.0+d2+reach*Kokkos::sqrt(d2));
    }
    KOKKOS_INLINE_FUNCTION static bool outside(Vec3 d, Vec3 x, double offset, double tolerance) {
        return d.x*x.x+d.y*x.y+d.z*x.z-offset>tolerance;
    }
};

// Experimental arithmetic-scaled classification: no absolute length floor.
// This bounds rounding in the plane dot product; it is not an exact predicate
// and does not remove the separate nearly-singular intersection failure.
struct RelativeClippingTolerance {
    static constexpr bool stable_intersections=true;
    KOKKOS_INLINE_FUNCTION static bool coincident(double d2) { return d2==0.0; }
    KOKKOS_INLINE_FUNCTION static double plane_tolerance(double, double) { return 0.0; }
    KOKKOS_INLINE_FUNCTION static bool outside(Vec3 d, Vec3 x, double offset, double) {
        constexpr double roundoff_margin=1.4210854715202004e-14; // 64 double epsilons
        const double scale=Kokkos::abs(d.x*x.x)+Kokkos::abs(d.y*x.y)+
            Kokkos::abs(d.z*x.z)+Kokkos::abs(offset);
        return d.x*x.x+d.y*x.y+d.z*x.z-offset>roundoff_margin*scale;
    }
};

// Plane storage can be tuned independently of the public output capacities.
// The original 160-plane workspace remains the reference/default.
// A positive NearestBatch visits an unsorted row in (distance squared, slot)
// order, the order insertion sorting gives, selecting that many candidates per
// scan of the row; the search then needs neither nearest_first nor its
// distance table. Zero keeps the row's own order.
template<int Planes = max_planes, class Tolerance = LegacyClippingTolerance, bool EarlyStop = false,
         bool DirectFaceLinks = false, int NearestBatch = 0>
struct PlaneWorkspace {
    static_assert(Planes >= n_box_planes && Planes <= 65535,
                  "plane capacity must fit the vertex plane indices");
    static_assert(NearestBatch >= 0, "nearest batch size must be zero (off) or positive");
    static_assert(NearestBatch == 0 || EarlyStop,
                  "without early stop a nearest batch only rescans the row; keep the row order");
    static constexpr int plane_capacity = Planes;
    static constexpr bool early_stop = EarlyStop;
    static constexpr bool direct_face_links = DirectFaceLinks;
    static constexpr bool nearest_batch = NearestBatch > 0;
    static constexpr int nearest_batch_size = NearestBatch;
    using tolerance = Tolerance;
    // The same options at another plane capacity, as used by the full-workspace retry.
    template<int OtherPlanes>
    using with_planes = PlaneWorkspace<OtherPlanes,Tolerance,EarlyStop,DirectFaceLinks,NearestBatch>;
};

// The clipping state can be owned privately or supplied by a team scratch
// allocation. Keeping the state together removes storage ownership from the
// geometry algorithm without changing its arithmetic or public defaults.
template<int Planes>
struct CellClippingState {
    Kokkos::Array<Plane,Planes> planes;
    Kokkos::Array<VertexPlanes,max_vertices> vertices;
    Kokkos::Array<Edge,max_hole_edges> hole;
};

// Compile-time disabled on normal calls: no counters, output or atomics.
struct NoWorkspaceTrace {
    KOKKOS_INLINE_FUNCTION void planes(int) {}
    KOKKOS_INLINE_FUNCTION void vertices(int) {}
    KOKKOS_INLINE_FUNCTION void hole(int) {}
    KOKKOS_INLINE_FUNCTION void faces(int) {}
    KOKKOS_INLINE_FUNCTION void face_vertices(int) {}
    KOKKOS_INLINE_FUNCTION void candidate() {}
    KOKKOS_INLINE_FUNCTION void suffix_test() {}
    KOKKOS_INLINE_FUNCTION void edge_test() {}
    KOKKOS_INLINE_FUNCTION void face_seed_test() {}
    KOKKOS_INLINE_FUNCTION void face_link_test() {}
    KOKKOS_INLINE_FUNCTION void vertex_test() {}
    KOKKOS_INLINE_FUNCTION void coordinate_read() {}
    KOKKOS_INLINE_FUNCTION void created_vertex() {}
    KOKKOS_INLINE_FUNCTION void finish(int, int, bool) {}
};

KOKKOS_INLINE_FUNCTION Vec3 add(Vec3 a, Vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
KOKKOS_INLINE_FUNCTION Vec3 sub(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
KOKKOS_INLINE_FUNCTION Vec3 mul(Vec3 a, double s) { return {a.x*s, a.y*s, a.z*s}; }
KOKKOS_INLINE_FUNCTION Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
KOKKOS_INLINE_FUNCTION double dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
KOKKOS_INLINE_FUNCTION double norm(Vec3 a) { return Kokkos::sqrt(dot(a,a)); }

// The next candidates of one row in (distance squared, slot) order. Each
// refill scans the row once and keeps the `Batch` smallest keys after the last
// one returned, so a row is never sorted and no per-row distance is stored.
template<int Batch>
struct NearestCandidates {
    double distance2[Batch];
    int ids[Batch];
    int size = 0, taken = 0;
    bool exhausted = false;
    double last_distance2 = -1.0;
    int last_id = -1;

    template<class Ids, class Separation>
    KOKKOS_INLINE_FUNCTION void refill(int first, int last, const Ids& neighbor_ids,
                                       const Separation& separation) {
        size = 0;
        taken = 0;
        for (int q=first;q<last;++q) {
            const int id=neighbor_ids(q);
            const Vec3 d=separation(id);
            const double d2=dot(d,d);
            // Written so that a NaN distance is never selected: it cuts no
            // vertex on the reference path, and returning it would make every
            // later key compare false and the row repeat forever.
            if (!(d2>last_distance2 || (d2==last_distance2 && id>last_id))) continue;
            int k=size<Batch ? size : Batch-1;
            if (size==Batch && (d2>distance2[k] || (d2==distance2[k] && id>ids[k]))) continue;
            while (k>0 && (distance2[k-1]>d2 || (distance2[k-1]==d2 && ids[k-1]>id))) {
                distance2[k]=distance2[k-1];
                ids[k]=ids[k-1];
                --k;
            }
            distance2[k]=d2;
            ids[k]=id;
            if (size<Batch) ++size;
        }
        exhausted = size<Batch;
    }

    // False once every candidate of the row has been returned.
    template<class Ids, class Separation>
    KOKKOS_INLINE_FUNCTION bool next(int first, int last, const Ids& neighbor_ids,
                                     const Separation& separation, int& id) {
        if (taken==size) {
            if (exhausted && size>0) return false;
            refill(first,last,neighbor_ids,separation);
            if (size==0) return false;
        }
        last_distance2=distance2[taken];
        last_id=ids[taken];
        id=ids[taken++];
        return true;
    }
};

// GPU threads keep the batch in local memory; 8 was fastest on a jittered
// lattice from 32 to 80 cubed (Quadro RTX 6000, fixed clock). Host threads
// only pay for rescans of the row, so they get a larger batch; 32 is not yet
// measured at a fixed clock.
template<class ExecSpace>
constexpr int default_nearest_batch() {
    return Kokkos::SpaceAccessibility<Kokkos::HostSpace,
        typename ExecSpace::memory_space>::accessible ? 32 : 8;
}

// Default geometry observer is compiled away. Hydro can instead integrate
// moments during the existing face walk, without storing diagnostic vertices.
struct DiagnosticGeometry {
    static constexpr bool store_vertices = true;
    KOKKOS_INLINE_FUNCTION void begin_face(Vec3) {}
    KOKKOS_INLINE_FUNCTION void vertex(Vec3) {}
    KOKKOS_INLINE_FUNCTION int end_face(int) { return success; }
    KOKKOS_INLINE_FUNCTION int finish(int status) { return status; }
};

KOKKOS_INLINE_FUNCTION bool intersect(const Plane& a, const Plane& b, const Plane& c,
                                      Vec3& x) {
    const Vec3 bc = cross(b.normal, c.normal);
    const Vec3 ca = cross(c.normal, a.normal);
    const Vec3 ab = cross(a.normal, b.normal);
    const double det = dot(a.normal, bc);
    const double scale = norm(a.normal)*norm(b.normal)*norm(c.normal);
    if (Kokkos::abs(det) <= 1.0e-13*scale) return false;
    x = mul(add(add(mul(bc,a.offset),mul(ca,b.offset)),mul(ab,c.offset)),1.0/det);
    return true;
}

// Equilibrated, partially pivoted elimination avoids cancellation in Cramer's
// determinant ratios when neighboring generators form almost parallel planes.
// Used only by the experimental relative policy; the original solver remains.
KOKKOS_INLINE_FUNCTION bool intersect_stable(const Plane& a, const Plane& b,
                                            const Plane& c, Vec3& x) {
    double matrix[3][4];
    const Plane input[3]={a,b,c};
    for (int row=0;row<3;++row) {
        const double length=norm(input[row].normal);
        if (!(length>0) || !Kokkos::isfinite(length)) return false;
        matrix[row][0]=input[row].normal.x/length;
        matrix[row][1]=input[row].normal.y/length;
        matrix[row][2]=input[row].normal.z/length;
        matrix[row][3]=input[row].offset/length;
    }
    for (int column=0;column<3;++column) {
        int pivot=column;
        for (int row=column+1;row<3;++row)
            if (Kokkos::abs(matrix[row][column])>Kokkos::abs(matrix[pivot][column])) pivot=row;
        if (Kokkos::abs(matrix[pivot][column])<=1e-13) return false;
        if (pivot!=column)
            for (int j=column;j<4;++j) {
                const double tmp=matrix[column][j];
                matrix[column][j]=matrix[pivot][j]; matrix[pivot][j]=tmp;
            }
        for (int row=column+1;row<3;++row) {
            const double factor=matrix[row][column]/matrix[column][column];
            for (int j=column+1;j<4;++j)
                matrix[row][j]=Kokkos::fma(-factor,matrix[column][j],matrix[row][j]);
        }
    }
    double result[3];
    for (int row=2;row>=0;--row) {
        double rhs=matrix[row][3];
        for (int j=row+1;j<3;++j) rhs=Kokkos::fma(-matrix[row][j],result[j],rhs);
        result[row]=rhs/matrix[row][row];
        if (!Kokkos::isfinite(result[row])) return false;
    }
    x={result[0],result[1],result[2]};
    return true;
}

template<class Tolerance = LegacyClippingTolerance, class Planes>
KOKKOS_INLINE_FUNCTION bool vertex_position(const VertexPlanes& v,
                                             const Planes& planes,
                                             Vec3& x) {
    if constexpr (Tolerance::stable_intersections)
        return intersect_stable(planes[v.p[0]],planes[v.p[1]],planes[v.p[2]],x);
    return intersect(planes[v.p[0]],planes[v.p[1]],planes[v.p[2]],x);
}

// Reference path stores only plane triples and reconstructs coordinates on use.
struct RecomputedVertexPositions {
    static constexpr bool caches_coordinates = false;
    KOKKOS_INLINE_FUNCTION void initialize(int, double) {}
    KOKKOS_INLINE_FUNCTION void keep(int, int) {}
    KOKKOS_INLINE_FUNCTION void set(int, Vec3) {}
    template<class Tolerance = LegacyClippingTolerance, class Planes>
    KOKKOS_INLINE_FUNCTION bool position(int, const VertexPlanes& vertex,
        const Planes& planes, Vec3& x) const {
        return vertex_position<Tolerance>(vertex,planes,x);
    }
};

// Experimental path: compute each new vertex once, carry surviving positions
// through compaction, and reuse them during clipping, certification and output.
// Adds 64*3 doubles (1,536 bytes) to each cell's private temporary workspace;
// no persistent output or neighbor storage is added. Benchmark occupancy costs.
struct CachedVertexPositions {
    static constexpr bool caches_coordinates = true;
    Kokkos::Array<Vec3,max_vertices> positions;
    KOKKOS_INLINE_FUNCTION void initialize(int c, double reach) {
        positions[c]={(c&1) ? -reach : reach,
                      (c&2) ? -reach : reach,
                      (c&4) ? -reach : reach};
    }
    KOKKOS_INLINE_FUNCTION void keep(int destination, int source) {
        positions[destination]=positions[source];
    }
    KOKKOS_INLINE_FUNCTION void set(int index, Vec3 x) { positions[index]=x; }
    template<class Tolerance = LegacyClippingTolerance, class Planes>
    KOKKOS_INLINE_FUNCTION bool position(int index, const VertexPlanes&,
        const Planes&, Vec3& x) const {
        x=positions[index];
        return true;
    }
};

// Opt-in mixed cache: only clipping uses rounded coordinates. Boundary tests
// fall back to the double plane intersection; certification/output always do.
struct FloatCachedVertexPositions {
    // position() reconstructs in double; approximate() is the separate FP32 cache.
    static constexpr bool caches_coordinates = false;
    struct Position { float x,y,z; };
    Kokkos::Array<Position,max_vertices> positions;
    KOKKOS_INLINE_FUNCTION void set(int i, Vec3 x) {
        positions[i]={static_cast<float>(x.x),static_cast<float>(x.y),static_cast<float>(x.z)};
    }
    KOKKOS_INLINE_FUNCTION void initialize(int c,double reach) {
        set(c,{(c&1)?-reach:reach,(c&2)?-reach:reach,(c&4)?-reach:reach});
    }
    KOKKOS_INLINE_FUNCTION void keep(int destination,int source) {
        positions[destination]=positions[source];
    }
    KOKKOS_INLINE_FUNCTION Vec3 approximate(int i) const {
        return {positions[i].x,positions[i].y,positions[i].z};
    }
    template<class Tolerance=LegacyClippingTolerance,class Planes>
    KOKKOS_INLINE_FUNCTION bool position(int,const VertexPlanes& vertex,
                                        const Planes& planes,Vec3& x) const {
        return vertex_position<Tolerance>(vertex,planes,x);
    }
};
static_assert(sizeof(FloatCachedVertexPositions)*2==sizeof(CachedVertexPositions));

template<class Edges, class Trace=NoWorkspaceTrace>
KOKKOS_INLINE_FUNCTION int edge_add_or_cancel(Edges& edges,
                                               int& count, std::uint16_t a,
                                               std::uint16_t b, Trace&& trace={}) {
    for (int i=0;i<count;++i) {
        trace.edge_test();
        if (edges[i].a==b && edges[i].b==a) {
            edges[i]=edges[count-1];
            --count;
            return success;
        }
    }
    if (count==max_hole_edges) return hole_capacity;
    edges[count++]={a,b};
    return success;
}

// Build one cell from its candidate particle IDs. The list must contain every
// particle within `reach` of this row. A successful certificate proves that
// particles outside that radius cannot cut the returned cell. Sort candidates
// nearest-first to reduce transient workspace growth; an unfavorable order can
// trigger a capacity status even though the final cell would fit.
template<class VertexStorage = RecomputedVertexPositions,
         class Workspace = PlaneWorkspace<>,
         class Positions, class RowOffsets, class NeighborIds,
         class FaceCounts, class FaceIds, class FaceAreas, class FaceVertexCounts,
         class FaceVertices, class RowStatus, class Geometry = DiagnosticGeometry,
         class Trace = NoWorkspaceTrace,
         class CellState = CellClippingState<Workspace::plane_capacity>>
KOKKOS_INLINE_FUNCTION void reconstruct_row(
    int row, const Positions positions, const RowOffsets neighbor_offsets,
    const NeighborIds neighbor_ids, double reach, const Kokkos::Array<double,3>& box,
    const FaceCounts face_counts, const FaceIds face_ids,
    const FaceAreas face_areas, const FaceVertexCounts face_vertex_counts,
    const FaceVertices face_vertices, const RowStatus row_status,
    Geometry geometry = {}, Trace trace = {}, VertexStorage vertex_storage = {},
    CellState state = {}) {
    auto& planes=state.planes;
    auto& vertices=state.vertices;
    auto& hole=state.hole;
    int plane_count=n_box_planes;
    int vertex_count=8;
    int status=success;
    using Tolerance = typename Workspace::tolerance;
    trace.planes(plane_count);
    trace.vertices(vertex_count);
    auto position = [&](int index, Vec3& x) {
        trace.coordinate_read();
        return vertex_storage.template position<Tolerance>(index,vertices[index],planes,x);
    };

    // Six half-spaces make a temporary cube [-reach,reach]^3. Plane indices
    // are +x,-x,+y,-y,+z,-z; each initial vertex stores its three planes.
    planes[0]={{ 1,0,0},reach,-1}; planes[1]={{-1,0,0},reach,-1};
    planes[2]={{0, 1,0},reach,-1}; planes[3]={{0,-1,0},reach,-1};
    planes[4]={{0,0, 1},reach,-1}; planes[5]={{0,0,-1},reach,-1};
    for (int c=0;c<8;++c) {
        const std::uint16_t x=static_cast<std::uint16_t>(c&1);
        const std::uint16_t y=static_cast<std::uint16_t>(2+((c>>1)&1));
        const std::uint16_t z=static_cast<std::uint16_t>(4+((c>>2)&1));
        const bool odd=((c&1)+((c>>1)&1)+((c>>2)&1))%2==1;
        vertices[c].p[0]=x;
        vertices[c].p[1]=odd?z:y;
        vertices[c].p[2]=odd?y:z;
        vertex_storage.initialize(c,reach);
    }

    // A possibly stale enclosing radius is conservative: clipping only shrinks
    // the cell. Refresh it during the existing classification traversal.
    double enclosing_radius2=3.0*reach*reach;
    auto separation = [&](int other) {
        return Vec3{coordinate_displacement(positions,other,row,0,box[0]),
                    coordinate_displacement(positions,other,row,1,box[1]),
                    coordinate_displacement(positions,other,row,2,box[2])};
    };
    const int first_candidate=neighbor_offsets(row), end_candidate=neighbor_offsets(row+1);
    int q=first_candidate;
    NearestCandidates<Workspace::nearest_batch ? Workspace::nearest_batch_size : 1> nearest;
    while (true) {
        int other;
        if constexpr (Workspace::nearest_batch) {
            if (!nearest.next(first_candidate,end_candidate,neighbor_ids,separation,other)) break;
        } else {
            if (q==end_candidate) break;
            other=neighbor_ids(q++);
        }
        trace.candidate();
        const Vec3 d=separation(other);
        const double d2=dot(d,d);
        if (Tolerance::coincident(d2)) { status=coincident_particles; break; }
        if constexpr (Workspace::early_stop) {
            // A bisector farther than the enclosing sphere cannot cut it.
            const double cutoff=4.0*enclosing_radius2*(1.0+1.0e-10);
            if (Kokkos::isfinite(cutoff) && d2>cutoff) {
                // Nearest-batch order is exact, so every later candidate is
                // farther still. Otherwise check the entire suffix rather than
                // trusting input ordering; unsorted rows and late coincidences
                // retain the reference path.
                if constexpr (Workspace::nearest_batch) break;
                bool safe=true;
                // q already points past the current candidate.
                for (int k=q;k<end_candidate;++k) {
                    trace.suffix_test();
                    const Vec3 delta=separation(neighbor_ids(k));
                    const double distance2=dot(delta,delta);
                    if (!(distance2>cutoff) || Tolerance::coincident(distance2)) { safe=false; break; }
                }
                if (safe) break;
            }
        }

        // The Voronoi half-space is d.x <= |d|^2/2, with outward normal d.
        static_assert(max_vertices<=64, "classification mask needs at most 64 vertices");
        std::uint64_t outside_mask=0;
        bool any_outside=false;
        const double plane_offset=0.5*d2;
        const double tol=Tolerance::plane_tolerance(d2,reach);
        double visited_radius2=0.0;
        for (int v=0;v<vertex_count;++v) {
            Vec3 x;
            trace.vertex_test();
            if constexpr (std::is_same_v<VertexStorage,FloatCachedVertexPositions>) {
                x=vertex_storage.approximate(v);
                // Bound cache rounding (including float subnormals) separately
                // from the existing double arithmetic classification tolerance.
                const double magnitude=Kokkos::abs(d.x*x.x)+Kokkos::abs(d.y*x.y)+Kokkos::abs(d.z*x.z);
                const double uncertainty=2.0e-7*magnitude+1.5e-45*
                    (Kokkos::abs(d.x)+Kokkos::abs(d.y)+Kokkos::abs(d.z));
                const double residual=dot(d,x)-plane_offset;
                if (!Kokkos::isfinite(magnitude) ||
                    Kokkos::abs(residual)<=uncertainty+Kokkos::abs(tol)+
                        1.4210854715202004e-14*(magnitude+Kokkos::abs(plane_offset))) {
                    if (!position(v,x)) { status=degenerate_geometry; break; }
                }
                if constexpr (Workspace::early_stop) {
                    const double pad=2.0e-7*(Kokkos::abs(x.x)+Kokkos::abs(x.y)+Kokkos::abs(x.z))+4.5e-45;
                    const double radius=Kokkos::sqrt(dot(x,x))+pad;
                    visited_radius2=Kokkos::fmax(visited_radius2,radius*radius);
                }
            } else {
                if (!position(v,x)) { status=degenerate_geometry; break; }
                if constexpr (Workspace::early_stop)
                    visited_radius2=Kokkos::fmax(visited_radius2,dot(x,x));
            }
            const bool is_outside=Tolerance::outside(d,x,plane_offset,tol);
            if (is_outside) outside_mask|=std::uint64_t(1)<<v;
            any_outside=any_outside||is_outside;
        }
        if (status!=success) break;
        if constexpr (Workspace::early_stop) enclosing_radius2=visited_radius2;
        if (!any_outside) continue;
        trace.planes(plane_count+1);
        if (plane_count==Workspace::plane_capacity) { status=plane_capacity; break; }

        int hole_count=0;
        for (int v=0;v<vertex_count;++v) {
            if (!(outside_mask & (std::uint64_t(1)<<v))) continue;
            const auto p=vertices[v].p;
            for (int e=0;e<3;++e) {
                status=edge_add_or_cancel(hole,hole_count,p[e],p[(e+1)%3],trace);
                trace.hole(hole_count+(status==hole_capacity ? 1 : 0));
                if (status!=success) break;
            }
            if (status!=success) break;
        }
        if (status!=success) break;
        const int new_plane=plane_count++;
        planes[new_plane]={d,plane_offset,other};
        int kept=0;
        for (int v=0;v<vertex_count;++v)
            if (!(outside_mask & (std::uint64_t(1)<<v))) {
                vertex_storage.keep(kept,v);
                vertices[kept++]=vertices[v];
            }
        trace.vertices(kept+hole_count);
        if (kept+hole_count<4) { status=degenerate_geometry; break; }
        if (kept+hole_count>max_vertices) { status=vertex_capacity; break; }
        for (int e=0;e<hole_count;++e) {
            if (kept==max_vertices) { status=vertex_capacity; break; }
            vertices[kept]={{hole[e].a,hole[e].b,static_cast<std::uint16_t>(new_plane)}};
            Vec3 x;
            trace.created_vertex();
            if (!vertex_position<Tolerance>(vertices[kept],planes,x)) { status=degenerate_geometry; break; }
            vertex_storage.set(kept,x);
            if constexpr (Workspace::early_stop)
                enclosing_radius2=Kokkos::fmax(enclosing_radius2,dot(x,x));
            ++kept;
        }
        if (status!=success) break;
        vertex_count=kept;
    }

    face_counts(row)=0;
    if (status==success) {
        double radius2=0.0;
        for (int v=0;v<vertex_count;++v) {
            const auto p=vertices[v].p;
            if (p[0]<n_box_planes || p[1]<n_box_planes || p[2]<n_box_planes) {
                status=uncertified;
                break;
            }
            Vec3 x;
            if (!position(v,x)) { status=degenerate_geometry; break; }
            radius2=Kokkos::fmax(radius2,dot(x,x));
        }
        if (status==success && 4.0*radius2>reach*reach*(1.0+1.0e-10)) status=uncertified;
    }

    if (status==success) {
        int output_count=0;
        for (int p=n_box_planes;p<plane_count;++p) {
            Vec3 twice_area{0,0,0};
            // A vertex triple containing p lies on that face. Adjacent vertices
            // are linked by the directed pair of planes around that face.
            // One byte per plane, reused for each face. A directed (p,q)
            // plane pair identifies its vertex; 255 is an invalid vertex.
            Kokkos::Array<std::uint8_t,Workspace::direct_face_links ? Workspace::plane_capacity : 1> links;
            if constexpr (Workspace::direct_face_links)
                for (int q=0;q<plane_count;++q) links[q]=255;
            int start=-1;
            for (int v=0;v<vertex_count;++v) {
                trace.face_seed_test();
                for (int k=0;k<3;++k)
                    if (vertices[v].p[k]==p) {
                        start=v;
                        if constexpr (Workspace::direct_face_links) {
                            const int q=vertices[v].p[(k+1)%3];
                            // Preserve the reference's first matching vertex.
                            if (links[q]==255) links[q]=static_cast<std::uint8_t>(v);
                        }
                    }
            }
            if (start<0) continue;
            trace.faces(output_count+1);
            if (output_count==max_faces) { status=face_capacity; break; }
            geometry.begin_face(planes[p].normal);
            int current=start;
            int polygon_count=0;
            bool closed=false;
            for (int edge=0;edge<max_face_vertices;++edge) {
                if (polygon_count==max_face_vertices) { status=face_capacity; break; }
                Vec3 x;
                if (!position(current,x)) { status=degenerate_geometry; break; }
                if constexpr (Geometry::store_vertices) {
                    face_vertices(row,output_count,polygon_count,0)=x.x;
                    face_vertices(row,output_count,polygon_count,1)=x.y;
                    face_vertices(row,output_count,polygon_count,2)=x.z;
                }
                geometry.vertex(x);
                ++polygon_count;
                trace.face_vertices(polygon_count);
                const auto ids=vertices[current].p;
                int at=-1;
                for (int k=0;k<3;++k) if (ids[k]==p) at=k;
                if (at<0) { status=degenerate_geometry; break; }
                const std::uint16_t last=ids[(at+2)%3];
                int next=-1;
                if constexpr (Workspace::direct_face_links) {
                    trace.face_link_test();
                    if (links[last]!=255) next=links[last];
                } else {
                for (int w=0;w<vertex_count;++w) {
                    trace.face_link_test();
                    const auto other=vertices[w].p;
                    int pos=-1;
                    for (int k=0;k<3;++k) if (other[k]==p) pos=k;
                    if (pos>=0 && other[(pos+1)%3]==last) { next=w; break; }
                }
                }
                if (next<0) { status=degenerate_geometry; break; }
                Vec3 y;
                if (!position(next,y)) {
                    status=degenerate_geometry;
                    break;
                }
                twice_area=add(twice_area,cross(x,y));
                if (next==start) { closed=true; break; }
                current=next;
            }
            if (status==success && !closed) status=degenerate_geometry;
            if (status!=success) break;
            face_ids(row,output_count)=planes[p].neighbour;
            face_areas(row,output_count,0)=0.5*twice_area.x;
            face_areas(row,output_count,1)=0.5*twice_area.y;
            face_areas(row,output_count,2)=0.5*twice_area.z;
            if constexpr (Geometry::store_vertices)
                face_vertex_counts(row,output_count)=polygon_count;
            status=geometry.end_face(output_count);
            if (status!=success) break;
            ++output_count;
        }
        face_counts(row)=output_count;
    }
    const int final_status=geometry.finish(status);
    row_status(row)=final_status;
    trace.finish(row,final_status,VertexStorage::caches_coordinates);
}

// Kokkos batch wrapper. `neighbor_offsets` and `neighbor_ids` are a CSR
// candidate graph; Cabana can own and populate these arrays independently of
// the gravity tree. Output views have shapes [N], [N,max_faces],
// [N,max_faces,3], and [N], respectively.
template<class ExecSpace, class Positions, class RowOffsets, class NeighborIds,
         class ReachView, class FaceCounts, class FaceIds, class FaceAreas,
         class FaceVertexCounts, class FaceVertices, class RowStatus>
void reconstruct(ExecSpace exec, Positions positions, RowOffsets neighbor_offsets,
                 NeighborIds neighbor_ids, ReachView reach, Kokkos::Array<double,3> box,
                 FaceCounts face_counts, FaceIds face_ids, FaceAreas face_areas,
                 FaceVertexCounts face_vertex_counts, FaceVertices face_vertices,
                 RowStatus row_status) {
    const int n=static_cast<int>(positions.extent(0));
    Kokkos::parallel_for("local_voronoi::reconstruct",Kokkos::RangePolicy<ExecSpace>(exec,0,n),
        KOKKOS_LAMBDA(const int row) {
            reconstruct_row(row,positions,neighbor_offsets,neighbor_ids,reach(row),box,
                            face_counts,face_ids,face_areas,face_vertex_counts,face_vertices,
                            row_status);
        });
    exec.fence("local Voronoi cell reconstruction");
}

} // namespace local_voronoi
