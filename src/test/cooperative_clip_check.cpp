#include "cooperative_clip.hpp"
#include <iostream>
int main(int argc,char**argv){
    Kokkos::initialize(argc,argv);
    int errors=0;
    {
        using Exec=Kokkos::DefaultExecutionSpace;
        using M=Exec::memory_space;
        constexpr int rows=66;
        Kokkos::View<double***,M> x("vertices",rows,64,3),out("kept",rows,64,3);
        Kokkos::View<int*,M> counts("counts",rows),kept("kept_counts",rows);
        Kokkos::View<int**,M> outside("outside",rows,64);
        Kokkos::View<double*,M> radii("radii",rows);
        Kokkos::View<local_voronoi::Plane*,M> planes("planes",rows);
        auto hx=Kokkos::create_mirror_view(x); auto hc=Kokkos::create_mirror_view(counts);
        auto hp=Kokkos::create_mirror_view(planes);
        for(int row=0;row<rows;++row){
            hc(row)=row<=64 ? row : -1;
            hp(row)={{1,0,0},row%3==0 ? -100.0 : (row%3==1 ? 0.0 : 100.0),0};
            for(int v=0;v<64;++v){ hx(row,v,0)=v-32; hx(row,v,1)=1; hx(row,v,2)=0; }
        }
        Kokkos::deep_copy(x,hx); Kokkos::deep_copy(counts,hc); Kokkos::deep_copy(planes,hp);
        local_voronoi::classify_and_compact(Exec{},x,counts,planes,out,kept,outside,radii);
        auto ho=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},out);
        auto hk=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},kept);
        auto hr=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},radii);
        for(int row=0;row<rows;++row){
            if(hc(row)<0){ if(hk(row)!=-1)++errors; continue; }
            int expected=0; double maxr=0;
            for(int v=0;v<hc(row);++v){
                double xx=hx(row,v,0); maxr=std::max(maxr,xx*xx+1);
                if(xx<=hp(row).offset){
                    for(int a=0;a<3;++a) if(ho(row,expected,a)!=hx(row,v,a)) ++errors;
                    ++expected;
                }
            }
            if(hk(row)!=expected || hr(row)!=maxr)++errors;
        }
    }
    Kokkos::finalize(); std::cout<<"cooperative_clip_errors="<<errors<<'\n'; return errors?1:0;
}
