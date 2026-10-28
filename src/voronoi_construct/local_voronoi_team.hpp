#pragma once
#include "local_voronoi_hydro.hpp"
#include "local_voronoi_workspace.hpp"
#include <type_traits>
namespace local_voronoi {
struct CooperativeCell {
    Kokkos::Array<Plane,64> planes;
    Kokkos::Array<VertexPlanes,max_vertices> vertices;
    Kokkos::Array<Edge,max_hole_edges> hole;
    Kokkos::Array<Vec3,max_vertices> coordinates;
    int plane_count,vertex_count,status,hole_count,kept,new_plane;
    Vec3 delta;
    double distance2;
};

template<class Exec,class Positions,class Counts,class Table,class Reach,
         class FaceCounts,class FaceIds,class FaceAreas,class FaceCenters,
         class Volumes,class Centers,class StatusView>
void reconstruct_hydro_team(Exec exec,Positions positions,Counts counts,Table table,
    Reach reaches,Kokkos::Array<double,3> box,FaceCounts face_counts,FaceIds face_ids,
    FaceAreas face_areas,FaceCenters face_centers,Volumes volumes,Centers centers,
    StatusView row_status){
    using Policy=Kokkos::TeamPolicy<Exec>;
    using Member=typename Policy::member_type;
    using Scratch=Kokkos::View<CooperativeCell,typename Member::scratch_memory_space,Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    int lanes=1;
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Exec,Kokkos::Cuda>) lanes=32;
#endif
    const auto capacity=table.extent(1);
    Kokkos::parallel_for("local_voronoi::cooperative_cell",Policy(exec,positions.extent(0),lanes).set_scratch_size(0,Kokkos::PerTeam(Scratch::shmem_size())),KOKKOS_LAMBDA(const Member& team){
        const int row=team.league_rank(); const double reach=reaches(row);
        Scratch memory(team.team_scratch(0)); auto& cell=memory();
        auto& planes=cell.planes; auto& vertices=cell.vertices; auto& hole=cell.hole;
        auto& plane_count=cell.plane_count; auto& vertex_count=cell.vertex_count; auto& status=cell.status;
        Kokkos::single(Kokkos::PerTeam(team),[&](){
            plane_count=6; vertex_count=8; status=success;
            if(counts(row)<0 || static_cast<std::size_t>(counts(row))>capacity) status=invalid_candidate_count;
            planes[0]={{1,0,0},reach,-1};planes[1]={{-1,0,0},reach,-1};
            planes[2]={{0,1,0},reach,-1};planes[3]={{0,-1,0},reach,-1};
            planes[4]={{0,0,1},reach,-1};planes[5]={{0,0,-1},reach,-1};
            for(int c=0;c<8;++c){
                const auto x=std::uint16_t(c&1),y=std::uint16_t(2+((c>>1)&1)),z=std::uint16_t(4+((c>>2)&1));
                const bool odd=((c&1)+((c>>1)&1)+((c>>2)&1))%2;
                vertices[c]={{x,odd?z:y,odd?y:z}};
                cell.coordinates[c]={(c&1)?-reach:reach,(c&2)?-reach:reach,(c&4)?-reach:reach};
            }
        }); team.team_barrier();
        for(int q=0;q<counts(row);++q){
            if(status!=success) break;
            Kokkos::single(Kokkos::PerTeam(team),[&](){
                const int other=table(row,q);
                Vec3 d{positions(other,0)-positions(row,0),positions(other,1)-positions(row,1),positions(other,2)-positions(row,2)};
                if(box[0]>0)d.x-=box[0]*Kokkos::floor(d.x/box[0]+0.5);
                if(box[1]>0)d.y-=box[1]*Kokkos::floor(d.y/box[1]+0.5);
                if(box[2]>0)d.z-=box[2]*Kokkos::floor(d.z/box[2]+0.5);
                cell.delta=d;cell.distance2=dot(d,d);
                if(RelativeClippingTolerance::coincident(cell.distance2))status=coincident_particles;
            }); team.team_barrier();
            if(status!=success)break;
            std::uint64_t mask=0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,vertex_count),[&](int v,std::uint64_t& bits){
                if(RelativeClippingTolerance::outside(cell.delta,cell.coordinates[v],0.5*cell.distance2,0))bits|=std::uint64_t(1)<<v;
            },Kokkos::BOr<std::uint64_t>(mask));
            if(mask==0)continue;
            Kokkos::single(Kokkos::PerTeam(team),[&](){
                cell.hole_count=0;
                if(plane_count==64){status=plane_capacity;return;}
                for(int v=0;v<vertex_count && status==success;++v)if(mask&(std::uint64_t(1)<<v)){
                    const auto p=vertices[v].p;
                    for(int e=0;e<3 && status==success;++e)status=edge_add_or_cancel(hole,cell.hole_count,p[e],p[(e+1)%3]);
                }
                if(status!=success)return;
                cell.new_plane=plane_count++;
                planes[cell.new_plane]={cell.delta,0.5*cell.distance2,table(row,q)};
                cell.kept=0;
                for(int v=0;v<vertex_count;++v)if(!(mask&(std::uint64_t(1)<<v))){
                    cell.coordinates[cell.kept]=cell.coordinates[v];vertices[cell.kept++]=vertices[v];
                }
                if(cell.kept+cell.hole_count<4)status=degenerate_geometry;
                if(cell.kept+cell.hole_count>max_vertices)status=vertex_capacity;
            });team.team_barrier();
            if(status!=success)break;
            int failures=0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,cell.hole_count),[&](int e,int& bad){
                const int v=cell.kept+e;
                vertices[v]={{hole[e].a,hole[e].b,static_cast<std::uint16_t>(cell.new_plane)}};
                Vec3 point;
                if(!vertex_position<RelativeClippingTolerance>(vertices[v],planes,point))++bad;
                else cell.coordinates[v]=point;
            },failures);
            Kokkos::single(Kokkos::PerTeam(team),[&](){vertex_count=cell.kept+cell.hole_count;if(failures)status=degenerate_geometry;});
            team.team_barrier();
        }
        Kokkos::single(Kokkos::PerTeam(team),[&](){
            detail::HydroGeometry<Volumes,Centers,FaceCenters> geometry{volumes,centers,face_centers,row};
            NoWorkspaceTrace trace;
            using VertexStorage=CachedVertexPositions;
            auto position=[&](int i,Vec3& x){x=cell.coordinates[i];return true;};
            detail::DiscardOutput face_vertices,face_vertex_counts;
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
            int start=-1;
            for (int v=0;v<vertex_count;++v)
                for (int k=0;k<3;++k)
                    if (vertices[v].p[k]==p) start=v;
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
                if constexpr (false) {
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
                for (int w=0;w<vertex_count;++w) {
                    const auto other=vertices[w].p;
                    int pos=-1;
                    for (int k=0;k<3;++k) if (other[k]==p) pos=k;
                    if (pos>=0 && other[(pos+1)%3]==last) { next=w; break; }
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
            if constexpr (false)
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
            if(row_status(row)!=success)face_counts(row)=0;
        });
    });
    exec.fence("cooperative full reconstruction");
}
}
