#ifndef VORONOI_TEST_FACE_LINKS
#define VORONOI_TEST_FACE_LINKS false
#endif
#ifndef VORONOI_TEST_TEAM
#define VORONOI_TEST_TEAM false
#endif
#include "local_voronoi_team.hpp"
#ifndef VORONOI_TEST_EARLY_STOP
#define VORONOI_TEST_EARLY_STOP false
#endif
#include "local_voronoi_retry.hpp"
#include "local_voronoi_workspace.hpp"
#include "neighbor_search.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {
using Exec=Kokkos::DefaultExecutionSpace;
using Memory=Exec::memory_space;
using Point=std::array<double,3>;
using Wide=std::array<long double,3>;
Wide add(Wide a,Wide b) { return {a[0]+b[0],a[1]+b[1],a[2]+b[2]}; }
Wide sub(Wide a,Wide b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
Wide mul(Wide a,long double s) { return {a[0]*s,a[1]*s,a[2]*s}; }
long double dot(Wide a,Wide b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
Wide cross(Wide a,Wide b) { return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; }
long double norm(Wide a) { return std::sqrt(dot(a,a)); }
template<class View> auto host(View v) {
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},v);
}
struct Face { int id; Wide area,center; };
struct Reference { std::vector<Face> faces; long double volume=0; Wide center{}; };

// Deliberately independent of incremental clipping: enumerate all plane triples
// in long double, retain feasible intersections, group by plane and angularly
// order their vertices. This is a slow CPU oracle for small bounded cells.
Reference reference(const std::vector<Point>& offsets) {
    struct Halfspace { Wide normal; long double offset; };
    std::vector<Halfspace> planes;
    long double length=0;
    for (const auto& p:offsets) {
        Wide d{p[0],p[1],p[2]};
        const long double r=norm(d);
        if (!(r>0)) throw std::runtime_error("reference requires distinct sites");
        planes.push_back({mul(d,1/r),r/2});
        length=std::max(length,r);
    }
    const long double tolerance=256*std::numeric_limits<long double>::epsilon()*length;
    std::vector<Wide> vertices;
    for (std::size_t a=0;a<planes.size();++a)
        for (std::size_t b=a+1;b<planes.size();++b)
            for (std::size_t c=b+1;c<planes.size();++c) {
                const auto& p=planes[a]; const auto& q=planes[b]; const auto& r=planes[c];
                const Wide qr=cross(q.normal,r.normal);
                const long double det=dot(p.normal,qr);
                if (std::abs(det)<1e-18L) continue;
                const Wide x=mul(add(add(mul(qr,p.offset),mul(cross(r.normal,p.normal),q.offset)),
                    mul(cross(p.normal,q.normal),r.offset)),1/det);
                bool feasible=true;
                for (const auto& plane:planes)
                    if (dot(plane.normal,x)>plane.offset+tolerance) { feasible=false; break; }
                if (!feasible) continue;
                bool duplicate=false;
                for (const auto& v:vertices)
                    if (norm(sub(x,v))<32*tolerance) { duplicate=true; break; }
                if (!duplicate) vertices.push_back(x);
            }
    Reference result;
    Wide cell_moment{};
    for (std::size_t i=0;i<planes.size();++i) {
        const auto& plane=planes[i];
        std::vector<Wide> polygon;
        Wide middle{};
        for (auto x:vertices)
            if (std::abs(dot(plane.normal,x)-plane.offset)<=32*tolerance) {
                polygon.push_back(x); middle=add(middle,x);
            }
        if (polygon.size()<3) continue;
        middle=mul(middle,1.0L/polygon.size());
        const Wide axis=std::abs(plane.normal[0])<0.8L ? Wide{1,0,0} : Wide{0,1,0};
        Wide u=cross(plane.normal,axis); u=mul(u,1/norm(u));
        const Wide v=cross(plane.normal,u);
        std::sort(polygon.begin(),polygon.end(),[&](Wide a,Wide b) {
            a=sub(a,middle); b=sub(b,middle);
            return std::atan2(dot(a,v),dot(a,u))<std::atan2(dot(b,v),dot(b,u));
        });
        long double area=0;
        Wide face_moment{};
        for (std::size_t j=1;j+1<polygon.size();++j) {
            const Wide a=polygon[0],b=polygon[j],c=polygon[j+1];
            const Wide sum=add(add(a,b),c);
            const long double weight=dot(cross(sub(b,a),sub(c,a)),plane.normal)/2;
            area+=weight;
            face_moment=add(face_moment,mul(sum,weight/3));
            const long double tetra=dot(a,cross(b,c))/6;
            result.volume+=tetra;
            cell_moment=add(cell_moment,mul(sum,tetra/4));
        }
        if (!(area>0)) throw std::runtime_error("reference face orientation failure");
        result.faces.push_back({static_cast<int>(i)+1,mul(plane.normal,area),mul(face_moment,1/area)});
    }
    if (!(result.volume>0)) throw std::runtime_error("reference cell is not bounded");
    result.center=mul(cell_moment,1/result.volume);
    return result;
}

struct Case { std::string name; std::vector<Point> offsets; double reach,condition=1; bool required=true; bool sorted=true; };
Case box(std::string name,double scale,double aspect=1) {
    Case result{std::move(name),{},2*scale,aspect};
    for (int a=0;a<3;++a)
        for (int sign:{-1,1}) {
            Point d{}; d[a]=sign*(a==0 ? 0.4/aspect : (a==1 ? 0.6 : 0.8))*scale;
            // Rotate to avoid axis-aligned special cases.
            result.offsets.push_back({0.6*d[0]-0.8*d[1],0.8*d[0]+0.6*d[1],d[2]});
        }
    return result;
}

int check(const Case& input) {
    auto offsets=input.offsets;
    if (input.sorted) std::sort(offsets.begin(),offsets.end(),[](Point a,Point b) {
        auto square=[](Point p) { return p[0]*p[0]+p[1]*p[1]+p[2]*p[2]; };
        const double da=square(a),db=square(b);
        return da==db ? a<b : da<db;
    });
    const bool coincident=std::any_of(offsets.begin(),offsets.end(),[](Point p) {
        return p[0]==0 && p[1]==0 && p[2]==0;
    });
    const auto expected=coincident ? Reference{} : reference(offsets);
    const int n=static_cast<int>(offsets.size())+1;
    Kokkos::View<double*[3],Memory> positions("positions",n);
    Kokkos::View<double*,Memory> reach("reach",n),volumes("volumes",n);
    Kokkos::View<int*,Memory> counts("counts",n),faces("faces",n),status("status",n);
    Kokkos::View<int**,Memory> table("table",n,n-1),ids("ids",n,local_voronoi::max_faces);
    Kokkos::View<double**,Memory> centers("centers",n,3);
    Kokkos::View<double***,Memory> areas("areas",n,local_voronoi::max_faces,3);
    Kokkos::View<double***,Memory> face_centers("face_centers",n,local_voronoi::max_faces,3);
    auto p=Kokkos::create_mirror_view(positions);
    auto t=Kokkos::create_mirror_view(table);
    for (int q=0;q<n-1;++q) {
        for (int a=0;a<3;++a) p(q+1,a)=offsets[q][a];
        t(0,q)=q+1;
    }
    Kokkos::deep_copy(positions,p); Kokkos::deep_copy(table,t);
    Kokkos::deep_copy(reach,input.reach);
    auto count=Kokkos::create_mirror_view(counts);
    count(0)=n-1; Kokkos::deep_copy(counts,count);
    const Kokkos::Array<double,3> no_box{0,0,0};
    local_voronoi::reconstruct_hydro<local_voronoi::CachedVertexPositions>(Exec{},positions,counts,table,
        reach,no_box,faces,ids,areas,face_centers,volumes,centers,status);
    const int legacy_status=host(status)(0);
    using Relative=local_voronoi::PlaneWorkspace<64,local_voronoi::RelativeClippingTolerance,VORONOI_TEST_EARLY_STOP,VORONOI_TEST_FACE_LINKS>;
    if constexpr (VORONOI_TEST_TEAM)
        local_voronoi::reconstruct_hydro_team(
            Exec{},positions,counts,table,reach,no_box,faces,ids,areas,face_centers,volumes,centers,status);
    else
    local_voronoi::reconstruct_hydro_retry<local_voronoi::CachedVertexPositions,Relative>(
        Exec{},positions,counts,table,reach,no_box,faces,ids,areas,face_centers,volumes,centers,status);
    auto hs=host(status),hf=host(faces); auto hi=host(ids);
    auto hv=host(volumes); auto hc=host(centers); auto ha=host(areas); auto hfc=host(face_centers);
    int errors=0;
    if (coincident) {
        errors=hs(0)!=local_voronoi::coincident_particles || hf(0)!=0 || hv(0)!=0;
        std::cout << "backend=" << Exec::name() << " stress=" << input.name
                  << " relative_status=" << hs(0) << " errors=" << errors << '\n';
        return errors;
    }
    long double volume_error=0,area_error=0;
    const long double bound=std::max(5e-12L,128.0L*std::numeric_limits<double>::epsilon()*input.condition);
    if (hs(0)!=local_voronoi::success) {
        if (input.required || hf(0)!=0 || hv(0)!=0) ++errors;
    } else {
        volume_error=std::abs(hv(0)-expected.volume)/expected.volume;
        if (!std::isfinite(hv(0)) || volume_error>bound || hf(0)!=static_cast<int>(expected.faces.size())) ++errors;
        Wide closure{};
        std::vector<bool> seen(n,false);
        for (int f=0;f<hf(0);++f) {
            if (hi(0,f)<1 || hi(0,f)>=n || seen[hi(0,f)]) { ++errors; continue; }
            seen[hi(0,f)]=true;
            const auto it=std::find_if(expected.faces.begin(),expected.faces.end(),
                [&](const Face& face) { return face.id==hi(0,f); });
            if (it==expected.faces.end()) { ++errors; continue; }
            const long double face_area=norm(it->area);
            for (int a=0;a<3;++a) {
                if (!std::isfinite(ha(0,f,a))) ++errors;
                area_error=std::max(area_error,std::abs(ha(0,f,a)-it->area[a])/face_area);
                closure[a]+=ha(0,f,a);
                if (!std::isfinite(hfc(0,f,a)) || std::abs(hfc(0,f,a)-it->center[a])>bound*input.reach) ++errors;
            }
        }
        if (area_error>bound || norm(closure)>bound*input.reach*input.reach) ++errors;
        for (int a=0;a<3;++a)
            if (!std::isfinite(hc(0,a)) || std::abs(hc(0,a)-expected.center[a])>bound*input.reach) ++errors;
    }
    std::cout << "backend=" << Exec::name() << " stress=" << input.name
              << " neighbors=" << n-1 << " legacy_status=" << legacy_status
              << " relative_status=" << hs(0) << " relative_volume_error=" << static_cast<double>(volume_error)
              << " relative_area_error=" << static_cast<double>(area_error)
              << " tolerance=" << static_cast<double>(bound) << " errors=" << errors << '\n';
    return errors;
}

int check_cloud() {
    constexpr int n=128, dense=64;
    Kokkos::View<double*[3],Memory> positions("cloud_positions",n);
    Kokkos::View<double*,Memory> radii("cloud_radii",n);
    auto p=Kokkos::create_mirror_view(positions);
    auto r=Kokkos::create_mirror_view(radii);
    std::mt19937_64 rng(20261027);
    std::uniform_real_distribution<double> uniform(0.0,1.0);
    for (int i=0;i<n;++i) {
        r(i)=i<dense ? 1e-4 : 0.49;
        for (int a=0;a<3;++a) {
            const int grid=a==0 ? (i-dense)/16 : (a==1 ? ((i-dense)/4)%4 : (i-dense)%4);
            p(i,a)=i<dense ? 0.25+1e-5*(uniform(rng)-0.5) : (grid+0.5)/4;
        }
    }
    Kokkos::deep_copy(positions,p); Kokkos::deep_copy(radii,r);
    neighbor_search::DenseCellSearch<Exec> search;
    // Grid width is independent of the smallest radius: avoid a dense grid
    // with a prohibitive number of empty cells in a mixed-density problem.
    neighbor_search::configure(search,n,{1,1,1},0.15,true,8,true);
    if (!Kokkos::SpaceAccessibility<Kokkos::HostSpace,Memory>::accessible)
        search.row_sort=neighbor_search::RowSort::team_bitonic;
    neighbor_search::rebuild(Exec{},search,positions,radii);
    auto sp=host(search.positions); auto sr=host(search.radii);
    auto counts=host(search.counts); auto table=host(search.table);
    int errors=0,graph_errors=0,nearest_errors=0,bisector_errors=0,closure_errors=0;
    double worst_nearest=0,worst_bisector=0,worst_closure=0;
    for (int i=0;i<n;++i) {
        std::vector<int> actual,expected;
        for (int q=0;q<counts(i);++q) actual.push_back(table(i,q));
        for (int j=0;j<n;++j) {
            if (i==j) continue;
            double d2=0;
            for (int a=0;a<3;++a) { double d=sp(j,a)-sp(i,a); d-=std::floor(d+0.5); d2+=d*d; }
            if (d2<=sr(i)*sr(i)) expected.push_back(j);
        }
        std::sort(actual.begin(),actual.end());
        if (actual!=expected) { ++errors; ++graph_errors; }
    }
    Kokkos::View<int*,Memory> faces("cloud_faces",n),status("cloud_status",n);
    Kokkos::View<int**,Memory> ids("cloud_ids",n,local_voronoi::max_faces);
    Kokkos::View<int**,Memory> vertices_per_face("cloud_vertices_per_face",n,local_voronoi::max_faces);
    Kokkos::View<double***,Memory> areas("cloud_areas",n,local_voronoi::max_faces,3);
    Kokkos::View<double****,Memory> vertices("cloud_vertices",n,local_voronoi::max_faces,
        local_voronoi::max_face_vertices,3);
    using Relative=local_voronoi::PlaneWorkspace<64,local_voronoi::RelativeClippingTolerance,VORONOI_TEST_EARLY_STOP,VORONOI_TEST_FACE_LINKS>;
    local_voronoi::reconstruct_table_retry<local_voronoi::CachedVertexPositions,Relative>(
        Exec{},search.positions,search.counts,search.table,search.radii,{1,1,1},
        faces,ids,areas,vertices_per_face,vertices,status);
    auto hs=host(status), hf=host(faces); auto hi=host(ids); auto hn=host(vertices_per_face);
    auto ha=host(areas); auto hv=host(vertices);
    int certified=0,uncertified=0,other_failures=0;
    for (int i=0;i<n;++i) {
        if (hs(i)!=local_voronoi::success) {
            if (hf(i)!=0) ++errors;
            if (hs(i)==local_voronoi::uncertified) ++uncertified;
            else ++other_failures;
            continue;
        }
        ++certified;
        double volume=0,closure[3]{};
        for (int f=0;f<hf(i);++f) {
            if (hi(i,f)<0 || hi(i,f)>=n) { ++errors; continue; }
            for (int a=0;a<3;++a) {
                if (!std::isfinite(ha(i,f,a))) ++errors;
                closure[a]+=ha(i,f,a);
                volume+=ha(i,f,a)*hv(i,f,0,a)/3;
            }
            for (int v=0;v<hn(i,f);++v) {
                double own2=0,partner2=0;
                for (int a=0;a<3;++a) own2+=hv(i,f,v,a)*hv(i,f,v,a);
                for (int j=0;j<n;++j) {
                    double d2=0;
                    for (int a=0;a<3;++a) {
                        double d=sp(j,a)-sp(i,a)-hv(i,f,v,a); d-=std::floor(d+0.5); d2+=d*d;
                    }
                    worst_nearest=std::max(worst_nearest,(own2-d2)/std::max(own2,1e-30));
                    if (d2+1e-9*std::max(own2,1e-30)<own2) { ++errors; ++nearest_errors; }
                    if (j==hi(i,f)) partner2=d2;
                }
                worst_bisector=std::max(worst_bisector,std::abs(partner2-own2)/std::max(own2,1e-30));
                if (std::abs(partner2-own2)>1e-9*std::max(own2,1e-30)) { ++errors; ++bisector_errors; }
            }
        }
        if (!(volume>0) || !std::isfinite(volume)) ++errors;
        for (double component:closure) {
            worst_closure=std::max(worst_closure,std::abs(component)/std::pow(volume,2.0/3));
            if (std::abs(component)>1e-9*std::pow(volume,2.0/3)) { ++errors; ++closure_errors; }
        }
    }
    if (certified==0 || uncertified==0 || other_failures!=0) ++errors;
    std::cout << "backend=" << Exec::name() << " stress=cluster_void_cloud particles=" << n
              << " radius_ratio=4900 certified=" << certified << " uncertified=" << uncertified
              << " other_failures=" << other_failures << " graph_errors=" << graph_errors
              << " nearest_errors=" << nearest_errors << " bisector_errors=" << bisector_errors
              << " closure_errors=" << closure_errors << " worst_nearest=" << worst_nearest
              << " worst_bisector=" << worst_bisector << " worst_closure=" << worst_closure
              << " errors=" << errors << '\n';
    return errors;
}
} // namespace

int main(int argc,char** argv) {
    Kokkos::initialize(argc,argv);
    int errors=0;
    {
        for (int exponent:{-8,-4,0,4,8}) {
            const double scale=std::pow(10.0,exponent);
            auto input=box("scale_1e"+std::to_string(exponent),scale);
            std::mt19937_64 rng(20261027);
            std::uniform_real_distribution<double> uniform(-1.0,1.0);
            for (int j=0;j<18;++j) {
                Point d{uniform(rng),uniform(rng),uniform(rng)};
                const double norm=std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
                for (auto& x:d) x*=0.45*scale/norm;
                input.offsets.push_back(d);
            }
            errors+=check(input);
        }
        for (double aspect:{1e2,1e4,1e6,1e8})
            errors+=check(box("sliver_"+std::to_string(aspect),1,aspect));
        auto cluster=box("dense_cluster_beside_void",1);
        cluster.offsets.push_back({1e-8,2e-9,-1e-9});
        cluster.condition=1e4;
        errors+=check(cluster);
        auto close=box("nearly_coincident_distinct_site",1);
        close.offsets.push_back({1e-15,2e-16,-1e-16});
        errors+=check(close);
        auto coincident=box("coincident_rejected",1);
        coincident.offsets.push_back({0,0,0});
        errors+=check(coincident);
        // Deliberately put irrelevant distant planes before a late cutter or
        // coincidence; a sorted-only break would silently miss these cases.
        auto unsorted=box("early_stop_unsorted_late_cutter",1);
        unsorted.sorted=false;
        unsorted.offsets.push_back({1.8,0,0});
        unsorted.offsets.push_back({0,1.8,0});
        unsorted.offsets.push_back({0.1,0.1,0.1});
        errors+=check(unsorted);
        unsorted.name="early_stop_unsorted_late_coincidence";
        unsorted.offsets.back()={0,0,0};
        errors+=check(unsorted);
        auto boundary=box("early_stop_radius_boundary",1);
        boundary.sorted=false;
        boundary.offsets.push_back({1.08,0,0});
        boundary.offsets.push_back({1.07,0,0});
        errors+=check(boundary);
        errors+=check_cloud();
    }
    Kokkos::finalize();
    return errors ? 2 : 0;
}
