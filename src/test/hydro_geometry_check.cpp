#include "local_voronoi_hydro.hpp"
#include "local_voronoi_workspace.hpp"
#include "local_voronoi_retry.hpp"

#include <cmath>
#include <array>
#include <iostream>
#include <type_traits>

namespace {
using Exec=Kokkos::DefaultExecutionSpace;
using Memory=Exec::memory_space;
template<class View> auto host(View view) {
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},view);
}

template<class VertexStorage = local_voronoi::RecomputedVertexPositions,
         class Workspace = local_voronoi::PlaneWorkspace<>>
int check_box(double scale, bool asymmetric, bool rotated) {
    constexpr int n=7;
    Kokkos::View<double*[3],Memory> positions("positions",n);
    Kokkos::View<double*,Memory> reach("reach",n), volume("volume",n);
    Kokkos::View<int*,Memory> counts("counts",n), faces("faces",n), status("status",n);
    Kokkos::View<int**,Memory> table("table",n,6), ids("ids",n,local_voronoi::max_faces);
    Kokkos::View<double**,Memory> centroid("centroid",n,3);
    Kokkos::View<double***,Memory> area("area",n,local_voronoi::max_faces,3);
    Kokkos::View<double***,Memory> face_centroid("face_centroid",n,local_voronoi::max_faces,3);
    auto p=Kokkos::create_mirror_view(positions);
    auto c=Kokkos::create_mirror_view(counts);
    auto t=Kokkos::create_mirror_view(table);
    const double lower[3]={-0.2*scale,-0.3*scale,-0.4*scale};
    const double upper[3]={(asymmetric ? 0.4 : 0.2)*scale,
                          (asymmetric ? 0.5 : 0.3)*scale,
                          (asymmetric ? 0.6 : 0.4)*scale};
    const double translation[3]={1.2*scale,-2.3*scale,4.1*scale};
    auto rotate=[&](std::array<double,3> x) -> std::array<double,3> {
        return rotated ? std::array<double,3>{0.6*x[0]-0.8*x[1],0.8*x[0]+0.6*x[1],x[2]} : x;
    };
    for (int i=0;i<n;++i) {
        c(i)=i==0 ? 6 : 0;
        for (int a=0;a<3;++a) p(i,a)=translation[a];
    }
    for (int q=0;q<6;++q) {
        const int a=q/2;
        p(q+1,a)+=2*(q%2 ? upper[a] : lower[a]);
        t(0,q)=q+1;
    }
    for (int i=0;i<n;++i) {
        const auto x=rotate({p(i,0),p(i,1),p(i,2)});
        for (int a=0;a<3;++a) p(i,a)=x[a];
    }
    Kokkos::deep_copy(positions,p);
    Kokkos::deep_copy(counts,c);
    Kokkos::deep_copy(table,t);
    Kokkos::deep_copy(reach,2*scale);
    auto reconstruct=[&] {
        local_voronoi::reconstruct_hydro<VertexStorage,Workspace>(Exec{},positions,counts,table,reach,
            Kokkos::Array<double,3>{0,0,0},faces,ids,area,face_centroid,volume,centroid,status);
    };
    reconstruct();
    auto hs=host(status), hf=host(faces);
    auto hi=host(ids);
    auto hv=host(volume);
    auto hc=host(centroid);
    auto ha=host(area), hfc=host(face_centroid);
    int errors=0;
    Kokkos::View<local_voronoi::WorkspaceUsage*,Memory> workspace_usage("workspace_usage",n);
    local_voronoi::profile_workspace<VertexStorage,Workspace>(Exec{},positions,counts,table,reach,
        Kokkos::Array<double,3>{0,0,0},workspace_usage);
    auto usage=host(workspace_usage);
    if (usage(0).status!=hs(0) || usage(0).faces!=hf(0)) ++errors;
    if (!rotated) {
        if (usage(0).planes!=12 || usage(0).vertices!=8 || usage(0).faces!=6 ||
            usage(0).face_vertices!=4 || usage(0).candidates!=6 ||
            usage(0).vertex_tests!=48 || usage(0).coordinate_reads!=104 ||
            usage(0).created_vertices!=24 || usage(0).intersection_solves!=
                (VertexStorage::caches_coordinates ? 24 : 128)) ++errors;
    }
    using Small=local_voronoi::PlaneWorkspace<8,typename Workspace::tolerance>;
    local_voronoi::profile_workspace<VertexStorage,Small>(
        Exec{},positions,counts,table,reach,Kokkos::Array<double,3>{0,0,0},workspace_usage);
    usage=host(workspace_usage);
    if (usage(0).status!=local_voronoi::plane_capacity || usage(0).planes!=9) ++errors;
    const double expected_volume=(upper[0]-lower[0])*(upper[1]-lower[1])*(upper[2]-lower[2]);
    if (hs(0)!=local_voronoi::success || hf(0)!=6 || !std::isfinite(hv(0)) ||
        std::abs(hv(0)-expected_volume)>1e-11*expected_volume) ++errors;
    const auto expected_centroid=rotate({(upper[0]+lower[0])/2,
        (upper[1]+lower[1])/2,(upper[2]+lower[2])/2});
    for (int a=0;a<3;++a)
        if (!std::isfinite(hc(0,a)) || std::abs(hc(0,a)-expected_centroid[a])>1e-11*scale) ++errors;
    if (hs(0)==local_voronoi::success) {
        bool seen[6]={false,false,false,false,false,false};
        for (int f=0;f<hf(0);++f) {
            const int q=hi(0,f)-1;
            if (q<0 || q>=6 || seen[q]) { ++errors; continue; }
            seen[q]=true;
            const int axis=q/2;
            double expected_area=1;
            for (int a=0;a<3;++a) if (a!=axis) expected_area*=upper[a]-lower[a];
            std::array<double,3> expected_vector{}, expected_face{};
            expected_vector[axis]=q%2 ? expected_area : -expected_area;
            for (int a=0;a<3;++a)
                expected_face[a]=a==axis ? (q%2 ? upper[a] : lower[a]) : (upper[a]+lower[a])/2;
            expected_vector=rotate(expected_vector);
            expected_face=rotate(expected_face);
            for (int a=0;a<3;++a) {
                const double component=expected_vector[a];
                const double center=expected_face[a];
                if (!std::isfinite(ha(0,f,a)) || std::abs(ha(0,f,a)-component)>1e-11*scale*scale ||
                    !std::isfinite(hfc(0,f,a)) || std::abs(hfc(0,f,a)-center)>1e-11*scale) ++errors;
            }
        }
    }
    for (int i=1;i<n;++i)
        if (hs(i)!=local_voronoi::uncertified || hf(i)!=0 || hv(i)!=0) ++errors;
    // Force the small workspace to fail; both retry interfaces must reproduce
    // the known analytic cell using the full reference capacity.
    int expected_ids[6]{};
    double expected_areas[6][3]{};
    if (hs(0)==local_voronoi::success && hf(0)==6)
        for (int f=0;f<6;++f) {
            expected_ids[f]=hi(0,f);
            for (int a=0;a<3;++a) expected_areas[f][a]=ha(0,f,a);
        }
    local_voronoi::reconstruct_hydro_retry<VertexStorage,Small>(
        Exec{},positions,counts,table,reach,Kokkos::Array<double,3>{0,0,0},
        faces,ids,area,face_centroid,volume,centroid,status);
    hs=host(status); hf=host(faces); hv=host(volume); hc=host(centroid);
    if (hs(0)!=local_voronoi::success || hf(0)!=6 ||
        std::abs(hv(0)-expected_volume)>1e-11*expected_volume) ++errors;
    for (int a=0;a<3;++a)
        if (std::abs(hc(0,a)-expected_centroid[a])>1e-11*scale) ++errors;
    Kokkos::View<int**,Memory> polygon_counts("polygon_counts",n,local_voronoi::max_faces);
    Kokkos::View<double****,Memory> polygon_vertices("polygon_vertices",n,
        local_voronoi::max_faces,local_voronoi::max_face_vertices,3);
    local_voronoi::reconstruct_table_retry<VertexStorage,Small>(
        Exec{},positions,counts,table,reach,Kokkos::Array<double,3>{0,0,0},
        faces,ids,area,polygon_counts,polygon_vertices,status);
    hs=host(status); hf=host(faces); hi=host(ids); ha=host(area);
    if (hs(0)!=local_voronoi::success || hf(0)!=6) ++errors;
    if (hf(0)==6)
        for (int f=0;f<6;++f) {
            if (hi(0,f)!=expected_ids[f]) ++errors;
            for (int a=0;a<3;++a) if (ha(0,f,a)!=expected_areas[f][a]) ++errors;
        }
    // A failed rebuild must clear valid moments left by the preceding call.
    for (int a=0;a<3;++a) p(1,a)=p(0,a);
    Kokkos::deep_copy(positions,p);
    reconstruct();
    hs=host(status); hf=host(faces); hv=host(volume); hc=host(centroid);
    if (hs(0)!=local_voronoi::coincident_particles || hf(0)!=0 || hv(0)!=0 ||
        hc(0,0)!=0 || hc(0,1)!=0 || hc(0,2)!=0) ++errors;
    bool rejected=false;
    try {
        Kokkos::View<double**,Memory> wrong_centroid("wrong_centroid",n,2);
        local_voronoi::reconstruct_hydro(Exec{},positions,counts,table,reach,
            Kokkos::Array<double,3>{0,0,0},faces,ids,area,face_centroid,volume,wrong_centroid,status);
    } catch (const std::invalid_argument&) { rejected=true; }
    if (!rejected) ++errors;
    std::cout << "backend=" << Exec::name() << " analytic_box_scale=" << scale
              << " asymmetric=" << asymmetric << " rotated=" << rotated
              << " vertices=" << (std::is_same_v<VertexStorage,local_voronoi::CachedVertexPositions> ? "cached" : "recomputed")
              << " clipping_policy=" << (std::is_same_v<typename Workspace::tolerance,
                  local_voronoi::RelativeClippingTolerance> ? "relative_pivoted" : "legacy")
              << " errors=" << errors << '\n';
    return errors;
}
} // namespace

int main(int argc,char** argv) {
    Kokkos::initialize(argc,argv);
    int errors=0;
    {
        for (double scale:{0.001,0.1,1.0,10.0,1000.0})
            for (bool asymmetric:{false,true})
                for (bool rotated:{false,true}) {
                    errors+=check_box(scale,asymmetric,rotated);
                    errors+=check_box<local_voronoi::CachedVertexPositions>(scale,asymmetric,rotated);
                    errors+=check_box<local_voronoi::CachedVertexPositions,
                        local_voronoi::PlaneWorkspace<64,local_voronoi::RelativeClippingTolerance>>(
                            scale,asymmetric,rotated);
                }
    }
    Kokkos::finalize();
    return errors ? 2 : 0;
}
