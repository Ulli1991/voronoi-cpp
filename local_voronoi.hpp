#pragma once

#include <Kokkos_Core.hpp>

#include <cstdint>

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

KOKKOS_INLINE_FUNCTION Vec3 add(Vec3 a, Vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
KOKKOS_INLINE_FUNCTION Vec3 sub(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
KOKKOS_INLINE_FUNCTION Vec3 mul(Vec3 a, double s) { return {a.x*s, a.y*s, a.z*s}; }
KOKKOS_INLINE_FUNCTION Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
KOKKOS_INLINE_FUNCTION double dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
KOKKOS_INLINE_FUNCTION double norm(Vec3 a) { return Kokkos::sqrt(dot(a,a)); }

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

KOKKOS_INLINE_FUNCTION bool vertex_position(const VertexPlanes& v,
                                             const Kokkos::Array<Plane,max_planes>& planes,
                                             Vec3& x) {
    return intersect(planes[v.p[0]],planes[v.p[1]],planes[v.p[2]],x);
}

KOKKOS_INLINE_FUNCTION int edge_add_or_cancel(Kokkos::Array<Edge,max_hole_edges>& edges,
                                               int& count, std::uint16_t a,
                                               std::uint16_t b) {
    for (int i=0;i<count;++i) {
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
// particles outside that radius cannot cut the returned cell.
template<class Positions, class RowOffsets, class NeighborIds,
         class FaceCounts, class FaceIds, class FaceAreas, class FaceVertexCounts,
         class FaceVertices, class RowStatus>
KOKKOS_INLINE_FUNCTION void reconstruct_row(
    int row, const Positions positions, const RowOffsets neighbor_offsets,
    const NeighborIds neighbor_ids, double reach, const Kokkos::Array<double,3>& box,
    const FaceCounts face_counts, const FaceIds face_ids,
    const FaceAreas face_areas, const FaceVertexCounts face_vertex_counts,
    const FaceVertices face_vertices, const RowStatus row_status) {
    Kokkos::Array<Plane,max_planes> planes;
    Kokkos::Array<VertexPlanes,max_vertices> vertices;
    Kokkos::Array<Edge,max_hole_edges> hole;
    int plane_count=n_box_planes;
    int vertex_count=8;
    int status=success;

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
    }

    for (int q=neighbor_offsets(row); q<neighbor_offsets(row+1); ++q) {
        const int other=neighbor_ids(q);
        Vec3 d{positions(other,0)-positions(row,0),
               positions(other,1)-positions(row,1),
               positions(other,2)-positions(row,2)};
        if (box[0]>0) d.x-=box[0]*Kokkos::floor(d.x/box[0]+0.5);
        if (box[1]>0) d.y-=box[1]*Kokkos::floor(d.y/box[1]+0.5);
        if (box[2]>0) d.z-=box[2]*Kokkos::floor(d.z/box[2]+0.5);
        const double d2=dot(d,d);
        if (d2<=1.0e-28) { status=coincident_particles; break; }

        // The Voronoi half-space is d.x <= |d|^2/2, with outward normal d.
        bool outside[max_vertices];
        bool any_outside=false;
        const double plane_offset=0.5*d2;
        const double tol=1.0e-12*(1.0+d2+reach*Kokkos::sqrt(d2));
        for (int v=0;v<vertex_count;++v) {
            Vec3 x;
            if (!vertex_position(vertices[v],planes,x)) { status=degenerate_geometry; break; }
            outside[v]=dot(d,x)-plane_offset>tol;
            any_outside=any_outside||outside[v];
        }
        if (status!=success) break;
        if (!any_outside) continue;
        if (plane_count==max_planes) { status=plane_capacity; break; }

        int hole_count=0;
        for (int v=0;v<vertex_count;++v) {
            if (!outside[v]) continue;
            const auto p=vertices[v].p;
            for (int e=0;e<3;++e) {
                status=edge_add_or_cancel(hole,hole_count,p[e],p[(e+1)%3]);
                if (status!=success) break;
            }
            if (status!=success) break;
        }
        if (status!=success) break;
        const int new_plane=plane_count++;
        planes[new_plane]={d,plane_offset,other};
        int kept=0;
        for (int v=0;v<vertex_count;++v)
            if (!outside[v]) vertices[kept++]=vertices[v];
        if (kept+hole_count<4) { status=degenerate_geometry; break; }
        if (kept+hole_count>max_vertices) { status=vertex_capacity; break; }
        for (int e=0;e<hole_count;++e) {
            if (kept==max_vertices) { status=vertex_capacity; break; }
            vertices[kept]={{hole[e].a,hole[e].b,static_cast<std::uint16_t>(new_plane)}};
            Vec3 x;
            if (!vertex_position(vertices[kept],planes,x)) { status=degenerate_geometry; break; }
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
            if (!vertex_position(vertices[v],planes,x)) { status=degenerate_geometry; break; }
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
            int start=-1;
            for (int v=0;v<vertex_count;++v)
                for (int k=0;k<3;++k)
                    if (vertices[v].p[k]==p) start=v;
            if (start<0) continue;
            if (output_count==max_faces) { status=face_capacity; break; }
            int current=start;
            int polygon_count=0;
            bool closed=false;
            for (int edge=0;edge<max_face_vertices;++edge) {
                if (polygon_count==max_face_vertices) { status=face_capacity; break; }
                Vec3 x;
                if (!vertex_position(vertices[current],planes,x)) { status=degenerate_geometry; break; }
                face_vertices(row,output_count,polygon_count,0)=x.x;
                face_vertices(row,output_count,polygon_count,1)=x.y;
                face_vertices(row,output_count,polygon_count,2)=x.z;
                ++polygon_count;
                const auto ids=vertices[current].p;
                int at=-1;
                for (int k=0;k<3;++k) if (ids[k]==p) at=k;
                if (at<0) { status=degenerate_geometry; break; }
                const std::uint16_t last=ids[(at+2)%3];
                int next=-1;
                for (int w=0;w<vertex_count;++w) {
                    const auto other=vertices[w].p;
                    int pos=-1;
                    for (int k=0;k<3;++k) if (other[k]==p) pos=k;
                    if (pos>=0 && other[(pos+1)%3]==last) { next=w; break; }
                }
                if (next<0) { status=degenerate_geometry; break; }
                Vec3 y;
                if (!vertex_position(vertices[next],planes,y)) {
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
            face_vertex_counts(row,output_count)=polygon_count;
            ++output_count;
        }
        face_counts(row)=output_count;
    }
    row_status(row)=status;
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
