#include "local_voronoi.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    int exit_code=0;
    {
        // A reproducible arbitrary periodic particle distribution. Replace
        // this all-to-all candidate graph with Cabana's neighbor list.
        constexpr int n=128;
        Kokkos::View<double*[3]> positions("positions",n);
        Kokkos::View<int*> offsets("neighbor_offsets",n+1);
        Kokkos::View<int*> ids("neighbor_ids",n*(n-1));
        auto h_positions=Kokkos::create_mirror_view(positions);
        auto h_offsets=Kokkos::create_mirror_view(offsets);
        auto h_ids=Kokkos::create_mirror_view(ids);

        std::mt19937_64 generator(20261027);
        std::uniform_real_distribution<double> uniform(0.0,1.0);
        for (int i=0;i<n;++i)
            for (int axis=0;axis<3;++axis)
                h_positions(i,axis)=uniform(generator);
        int cursor=0;
        for (int i=0;i<n;++i) {
            h_offsets(i)=cursor;
            for (int j=0;j<n;++j) if (j!=i) h_ids(cursor++)=j;
        }
        h_offsets(n)=cursor;
        Kokkos::deep_copy(positions,h_positions);
        Kokkos::deep_copy(offsets,h_offsets);
        Kokkos::deep_copy(ids,h_ids);

        Kokkos::View<double*> reach("reach",n);
        Kokkos::View<int*> counts("face_counts",n);
        Kokkos::View<int**> face_ids("face_ids",n,local_voronoi::max_faces);
        Kokkos::View<double***> areas("face_areas",n,local_voronoi::max_faces,3);
        Kokkos::View<int**> face_vertex_counts("face_vertex_counts",n,local_voronoi::max_faces);
        Kokkos::View<double****> face_vertices("face_vertices",n,local_voronoi::max_faces,
                                               local_voronoi::max_face_vertices,3);
        Kokkos::View<int*> status("row_status",n);
        Kokkos::deep_copy(reach,1.0);
        local_voronoi::reconstruct(Kokkos::DefaultExecutionSpace{},positions,offsets,ids,
            reach,Kokkos::Array<double,3>{1.0,1.0,1.0},counts,face_ids,areas,
            face_vertex_counts,face_vertices,status);

        auto h_counts=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},counts);
        auto h_status=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},status);
        auto h_face_ids=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},face_ids);
        auto h_areas=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},areas);
        auto h_vertex_counts=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},face_vertex_counts);
        auto h_vertices=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},face_vertices);
        std::ofstream particle_file("particles_128.csv");
        particle_file << "id,x,y,z\n";
        for (int i=0;i<n;++i)
            particle_file << i << ',' << h_positions(i,0) << ',' << h_positions(i,1) << ','
                          << h_positions(i,2) << '\n';
        std::ofstream face_file("faces_128.csv");
        face_file << "row,neighbor,vertex,x,y,z\n";
        for (int i=0;i<n;++i)
            for (int f=0;f<h_counts(i);++f)
                for (int v=0;v<h_vertex_counts(i,f);++v)
                    face_file << i << ',' << h_face_ids(i,f) << ',' << v << ','
                              << h_positions(i,0)+h_vertices(i,f,v,0) << ','
                              << h_positions(i,1)+h_vertices(i,f,v,1) << ','
                              << h_positions(i,2)+h_vertices(i,f,v,2) << '\n';
        int bad=0;
        double worst_closure=0.0;
        double worst_pair=0.0;
        for (int i=0;i<n;++i) {
            if (h_status(i)!=local_voronoi::success) {
                std::cerr << "row " << i << " failed certification/status " << h_status(i) << '\n';
                ++bad;
                continue;
            }
            double closure[3]={0.0,0.0,0.0};
            for (int f=0;f<h_counts(i);++f)
                for (int axis=0;axis<3;++axis) closure[axis]+=h_areas(i,f,axis);
            const double closure_norm=std::sqrt(closure[0]*closure[0]+closure[1]*closure[1]+closure[2]*closure[2]);
            worst_closure=std::max(worst_closure,closure_norm);
            for (int f=0;f<h_counts(i);++f) {
                const int j=h_face_ids(i,f);
                int reverse=-1;
                for (int g=0;g<h_counts(j);++g)
                    if (h_face_ids(j,g)==i) reverse=g;
                if (reverse<0) { ++bad; continue; }
                double mismatch2=0.0;
                for (int axis=0;axis<3;++axis) {
                    const double mismatch=h_areas(i,f,axis)+h_areas(j,reverse,axis);
                    mismatch2+=mismatch*mismatch;
                }
                worst_pair=std::max(worst_pair,std::sqrt(mismatch2));
            }
        }
        std::cout << "built " << n-bad << '/' << n << " cells; row 0 has "
                  << h_counts(0) << " faces; max |sum(area)|=" << worst_closure
                  << "; max shared-face mismatch=" << worst_pair << '\n';
        if (worst_closure>1.0e-9 || worst_pair>1.0e-9) bad=1;
        if (bad) exit_code=2;
    }
    Kokkos::finalize();
    return exit_code;
}
