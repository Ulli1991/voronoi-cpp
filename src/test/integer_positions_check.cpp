#include "local_voronoi_retry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {
using Exec=Kokkos::DefaultExecutionSpace;
using Memory=Exec::memory_space;
template<class View> auto host(View v) {
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},v);
}
struct Geometry {
    Kokkos::View<int*,Memory> faces,status;
    Kokkos::View<int**,Memory> ids;
    Kokkos::View<double***,Memory> areas,face_centers;
    Kokkos::View<double*,Memory> volumes;
    Kokkos::View<double**,Memory> centers;
    explicit Geometry(int n):faces("faces",n),status("status",n),ids("ids",n,32),
        areas("areas",n,32,3),face_centers("face_centers",n,32,3),
        volumes("volumes",n),centers("centers",n,3) {}
};
int displacement_check() {
    local_voronoi::FixedPointPositions32<Memory> p;
    p.values=decltype(p.values)("boundary_positions",8);
    auto h=Kokkos::create_mirror_view(p.values);
    const std::uint32_t values[8]={0,1,0x7fffffffu,0x80000000u,0xfffffffeu,0xffffffffu,12345,4000000000u};
    for (int i=0;i<8;++i) for (int a=0;a<3;++a) h(i,a)=values[i];
    Kokkos::deep_copy(p.values,h);
    int errors=0;
    Kokkos::parallel_reduce("integer_displacement_boundaries",Kokkos::RangePolicy<Exec>(0,64),
        KOKKOS_LAMBDA(int index,int& bad) {
            const int i=index/8,j=index%8;
            for (int a=0;a<3;++a) {
                const double raw=p(i,a)-p(j,a);
                const double expected=raw-Kokkos::floor(raw+0.5);
                if (local_voronoi::coordinate_displacement(p,i,j,a,1)!=expected) ++bad;
                if (local_voronoi::coordinate_displacement(p,i,j,a,0)!=raw) ++bad;
            }
        },errors);
    return errors;
}
int geometry_check(bool jitter,bool translate) {
    constexpr int side=5,n=side*side*side;
    local_voronoi::FixedPointPositions32<Memory> integer;
    integer.values=decltype(integer.values)("integer_positions",n);
    Kokkos::View<double*[3],Memory> decoded("decoded",n);
    Kokkos::View<int*,Memory> counts("counts",n);
    Kokkos::View<int**,Memory> table("table",n,n-1);
    Kokkos::View<double*,Memory> reach("reach",n);
    auto hp=Kokkos::create_mirror_view(integer.values);
    auto hd=Kokkos::create_mirror_view(decoded);
    auto ht=Kokkos::create_mirror_view(table);
    std::mt19937 rng(4261);
    std::uniform_real_distribution<double> noise(-0.02,0.02);
    int errors=0;
    for (int i=0;i<n;++i) {
        const int c[3]={i/(side*side),(i/side)%side,i%side};
        for (int a=0;a<3;++a) {
            double x=(c[a]+0.5)/side+(jitter?noise(rng):0)+(translate?0.499999:0);
            x-=std::floor(x);
            hp(i,a)=static_cast<std::uint32_t>(x*4294967296.0);
            hd(i,a)=double(hp(i,a))*0x1.0p-32;
            if (x-hd(i,a)<0 || x-hd(i,a)>=0x1.0p-32) ++errors;
        }
    }
    for (int i=0;i<n;++i) {
        std::vector<std::pair<double,int>> neighbors;
        for (int j=0;j<n;++j) if (i!=j) {
            double d2=0;
            for (int a=0;a<3;++a) {
                double d=hd(j,a)-hd(i,a); d-=std::floor(d+0.5); d2+=d*d;
            }
            neighbors.emplace_back(d2,j);
        }
        std::sort(neighbors.begin(),neighbors.end());
        for (int q=0;q<n-1;++q) ht(i,q)=neighbors[q].second;
    }
    Kokkos::deep_copy(integer.values,hp); Kokkos::deep_copy(decoded,hd);
    Kokkos::deep_copy(table,ht); Kokkos::deep_copy(counts,n-1); Kokkos::deep_copy(reach,0.49);
    Geometry reference(n),actual(n);
    auto run=[&](auto positions,Geometry& g) {
        local_voronoi::reconstruct_hydro_retry<local_voronoi::CachedVertexPositions,
            local_voronoi::PlaneWorkspace<64,local_voronoi::RelativeClippingTolerance,true>>(
                Exec{},positions,counts,table,reach,Kokkos::Array<double,3>{1,1,1},
                g.faces,g.ids,g.areas,g.face_centers,g.volumes,g.centers,g.status);
    };
    run(decoded,reference); run(integer,actual);
    auto rs=host(reference.status),as=host(actual.status),rf=host(reference.faces),af=host(actual.faces);
    auto ri=host(reference.ids),ai=host(actual.ids);
    auto rv=host(reference.volumes),av=host(actual.volumes);
    auto rc=host(reference.centers),ac=host(actual.centers);
    auto ra=host(reference.areas),aa=host(actual.areas);
    auto rfc=host(reference.face_centers),afc=host(actual.face_centers);
    for (int i=0;i<n;++i) {
        if (rs(i)!=local_voronoi::success || as(i)!=rs(i) || af(i)!=rf(i)) { ++errors; continue; }
        if (av(i)!=rv(i)) ++errors;
        for (int a=0;a<3;++a) if (ac(i,a)!=rc(i,a)) ++errors;
        for (int f=0;f<rf(i);++f) {
            if (ai(i,f)!=ri(i,f)) ++errors;
            for (int a=0;a<3;++a)
                if (aa(i,f,a)!=ra(i,f,a) || afc(i,f,a)!=rfc(i,f,a)) ++errors;
        }
    }
    std::cout << "integer_geometry jitter=" << jitter << " translated=" << translate
              << " exact_comparison_errors=" << errors << '\n';
    return errors;
}
}
int main(int argc,char** argv) {
    Kokkos::initialize(argc,argv);
    int errors=displacement_check();
    for (bool jitter:{false,true}) for (bool translate:{false,true})
        errors+=geometry_check(jitter,translate);
    Kokkos::finalize();
    return errors?1:0;
}
