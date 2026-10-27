#include "local_voronoi_retry.hpp"
#include "neighbor_search.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace {
using Exec=Kokkos::DefaultExecutionSpace;
using Memory=Exec::memory_space;
using Range=Kokkos::RangePolicy<Exec>;
constexpr double radius_scale=3.5, jitter_amplitude=0.1;
constexpr int capacity=250; // Conservative lattice bound, not a measured mean.
struct DeviceMemory { std::uint64_t free=0,total=0; };

DeviceMemory device_memory() {
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr (std::is_same_v<Exec,Kokkos::Cuda>) {
        std::size_t free=0,total=0;
        const auto error=cudaMemGetInfo(&free,&total);
        if (error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        return {free,total};
    }
#endif
    return {};
}

KOKKOS_INLINE_FUNCTION std::uint64_t mix(std::uint64_t x) {
    x+=0x9e3779b97f4a7c15ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}

// Total simultaneously live array payload. Includes input and sorted positions,
// radii, dense grid, fixed neighbor/distance tables, and compact cell geometry.
std::uint64_t array_bytes(int side) {
    const std::uint64_t n=std::uint64_t(side)*side*side;
    const double width=0.75*(radius_scale/side);
    const std::uint64_t nx=static_cast<std::uint64_t>(std::ceil(1.0/width));
    const std::uint64_t geometry=sizeof(int)*(2+local_voronoi::max_faces)+
        sizeof(double)*(4+6*local_voronoi::max_faces);
    return n*(32+44+12*capacity+geometry)+8*nx*nx*nx+8;
}

double median(std::vector<double> values) {
    std::sort(values.begin(),values.end());
    return values[values.size()/2];
}

struct Result {
    std::string status="PASS",policy;
    int side=0,repeats=0;
    std::uint64_t n=0,estimated_arrays=0,total_memory=0,initial_free=0,reserve=0,max_observed_used=0;
    double search_ms=0,build_ms=0,pipeline_ms=0,mean_neighbors=0,volume_error=0;
    double closure_error=0,tensor_error=0;
    long long validation_errors=0;
};

void write_result(const Result& r,const std::string& csv_path) {
    const auto rate=[&](double ms) { return ms>0 ? r.n/(ms*1e3) : 0.0; };
    std::cout << std::setprecision(9) << "status=" << r.status << " backend=" << Exec::name()
        << " side=" << r.side << " particles=" << r.n << " policy=" << r.policy
        << " repeats=" << r.repeats << " neighbor_capacity=" << capacity
        << " estimated_array_bytes=" << r.estimated_arrays
        << " initial_free_bytes=" << r.initial_free << " reserve_bytes=" << r.reserve
        << " max_observed_device_used_bytes=" << r.max_observed_used
        << " search_ms=" << r.search_ms << " reconstruction_ms=" << r.build_ms
        << " pipeline_ms=" << r.pipeline_ms << " search_Mparticles_s=" << rate(r.search_ms)
        << " reconstruction_Mcells_s=" << rate(r.build_ms)
        << " pipeline_Mparticles_s=" << rate(r.pipeline_ms)
        << " mean_neighbors=" << r.mean_neighbors << " volume_error=" << r.volume_error
        << " relative_closure=" << r.closure_error << " relative_centroid_tensor=" << r.tensor_error
        << " validation_errors=" << r.validation_errors << std::endl;
    if (csv_path.empty()) return;
    std::ifstream existing(csv_path);
    const bool header=!existing.good() || existing.peek()==std::ifstream::traits_type::eof();
    existing.close();
    std::ofstream out(csv_path,std::ios::app);
    if (!out) throw std::runtime_error("cannot append scaling CSV");
    if (header) out << "status,backend,side,particles,policy,repeats,neighbor_capacity,estimated_array_bytes,"
        "total_device_bytes,initial_free_bytes,reserve_bytes,max_observed_device_used_bytes,"
        "search_ms,reconstruction_ms,pipeline_ms,search_Mparticles_s,reconstruction_Mcells_s,"
        "pipeline_Mparticles_s,mean_neighbors,volume_error,relative_closure,relative_centroid_tensor,validation_errors\n";
    out << std::setprecision(17) << r.status << ',' << Exec::name() << ',' << r.side << ',' << r.n
        << ',' << r.policy << ',' << r.repeats << ',' << capacity << ',' << r.estimated_arrays
        << ',' << r.total_memory << ',' << r.initial_free << ',' << r.reserve << ',' << r.max_observed_used
        << ',' << r.search_ms << ',' << r.build_ms << ',' << r.pipeline_ms
        << ',' << rate(r.search_ms) << ',' << rate(r.build_ms) << ',' << rate(r.pipeline_ms)
        << ',' << r.mean_neighbors << ',' << r.volume_error << ',' << r.closure_error
        << ',' << r.tensor_error << ',' << r.validation_errors << '\n';
}

template<class Tolerance>
int run(int side,int repeats,const std::string& policy,const std::string& csv_path) {
    Result result;
    result.side=side; result.repeats=repeats; result.policy=policy;
    result.n=std::uint64_t(side)*side*side;
    result.estimated_arrays=array_bytes(side);
    const auto initial=device_memory();
    if (!initial.total && result.n>32768)
        throw std::runtime_error("large scaling requires the CUDA build; CPU smoke checks are limited to 32^3");
    result.total_memory=initial.total; result.initial_free=initial.free;
    if (initial.total) {
        result.reserve=std::max<std::uint64_t>(2ULL<<30,initial.total*15/100);
        if (initial.free<=result.reserve || result.estimated_arrays>initial.free-result.reserve) {
            result.status="SKIP_MEMORY";
            write_result(result,csv_path);
            return 77;
        }
    }
    const int n=static_cast<int>(result.n);
    Exec exec;
    const double radius=radius_scale/side;
    const Kokkos::Array<double,3> box{1,1,1};
    Kokkos::View<double*[3],Memory> positions(Kokkos::view_alloc(Kokkos::WithoutInitializing,"input_positions"),n);
    Kokkos::View<double*,Memory> radii(Kokkos::view_alloc(Kokkos::WithoutInitializing,"input_radii"),n);
    Kokkos::parallel_for("scaling_device_input",Range(exec,0,n),KOKKOS_LAMBDA(int i) {
        const int grid=n-1-i;
        const int coordinate[3]={grid/(side*side),(grid/side)%side,grid%side};
        for (int a=0;a<3;++a) {
            const auto hash=mix(std::uint64_t(grid)*3+a+20261027ULL);
            const double uniform=static_cast<double>(hash>>11)*0x1.0p-53;
            positions(i,a)=(coordinate[a]+0.5+jitter_amplitude*(2*uniform-1))/side;
        }
        radii(i)=radius;
    });
    exec.fence("scaling input ready");
    neighbor_search::DenseCellSearch<Exec> search;
    search.row_sort=neighbor_search::RowSort::team_bitonic;
    neighbor_search::configure(search,n,box,0.75*radius,true,capacity,true,1);
    auto allocate=[](auto& view,const char* name,auto... sizes) {
        using View=std::decay_t<decltype(view)>;
        view=View(Kokkos::view_alloc(Kokkos::WithoutInitializing,std::string(name)),sizes...);
    };
    Kokkos::View<int*,Memory> faces,status;
    Kokkos::View<int**,Memory> ids;
    Kokkos::View<double***,Memory> areas,face_centers;
    Kokkos::View<double*,Memory> volumes;
    Kokkos::View<double**,Memory> centers;
    allocate(faces,"face_counts",n); allocate(status,"status",n);
    allocate(ids,"face_ids",n,local_voronoi::max_faces);
    allocate(areas,"face_areas",n,local_voronoi::max_faces,3);
    allocate(face_centers,"face_centroids",n,local_voronoi::max_faces,3);
    allocate(volumes,"volumes",n); allocate(centers,"cell_centroids",n,3);
    auto observe_memory=[&] {
        const auto memory=device_memory();
        if (memory.total) result.max_observed_used=std::max(result.max_observed_used,memory.total-memory.free);
    };
    observe_memory();
    auto rebuild=[&] {
        neighbor_search::build(exec,search,positions,radii);
        neighbor_search::fill(exec,search);
        int longest=0;
        Kokkos::deep_copy(exec,longest,search.max_count);
        exec.fence("scaling neighbor completeness");
        if (longest>capacity) throw std::runtime_error("lattice neighbor capacity bound exceeded");
    };
    auto reconstruct=[&] {
        local_voronoi::reconstruct_hydro_retry<local_voronoi::CachedVertexPositions,
            local_voronoi::PlaneWorkspace<64,Tolerance>>(exec,search.positions,search.counts,
                search.table,search.radii,box,faces,ids,areas,face_centers,volumes,centers,status);
    };
    for (int warm=0;warm<2;++warm) { rebuild(); reconstruct(); }
    observe_memory();
    std::vector<double> search_times,build_times,pipeline_times;
    for (int repeat=0;repeat<repeats;++repeat) {
        Kokkos::Timer timer;
        rebuild(); search_times.push_back(timer.seconds()*1e3);
        timer.reset(); reconstruct(); build_times.push_back(timer.seconds()*1e3);
        timer.reset(); rebuild(); reconstruct(); pipeline_times.push_back(timer.seconds()*1e3);
        observe_memory();
    }
    result.search_ms=median(search_times); result.build_ms=median(build_times);
    result.pipeline_ms=median(pipeline_times);
    const auto counts=search.counts; const auto p=search.positions; const auto table=search.table;
    long long bad=0,edges=0;
    double volume_sum=0,max_closure=0,max_tensor=0;
    Kokkos::parallel_reduce("scaling_full_geometry_validation",Range(exec,0,n),
        KOKKOS_LAMBDA(int i,long long& invalid,long long& neighbor_edges,double& sum,
                      double& worst_closure,double& worst_tensor) {
            neighbor_edges+=counts(i);
            const double volume=volumes(i);
            if (status(i)!=local_voronoi::success || faces(i)<4 || faces(i)>local_voronoi::max_faces ||
                !(volume>0) || !Kokkos::isfinite(volume)) { ++invalid; return; }
            sum+=volume;
            double closure[3]{},tensor[3][3]{};
            for (int f=0;f<faces(i);++f) {
                if (ids(i,f)<0 || ids(i,f)>=n) ++invalid;
                for (int a=0;a<3;++a) {
                    if (!Kokkos::isfinite(areas(i,f,a)) || !Kokkos::isfinite(face_centers(i,f,a))) ++invalid;
                    closure[a]+=areas(i,f,a);
                    for (int b=0;b<3;++b) tensor[a][b]+=face_centers(i,f,a)*areas(i,f,b);
                }
            }
            const double closure_norm=Kokkos::sqrt(closure[0]*closure[0]+closure[1]*closure[1]+closure[2]*closure[2])/
                Kokkos::pow(volume,2.0/3.0);
            worst_closure=Kokkos::max(worst_closure,closure_norm);
            if (!Kokkos::isfinite(closure_norm) || closure_norm>1e-9) ++invalid;
            for (int a=0;a<3;++a) {
                if (!Kokkos::isfinite(centers(i,a))) ++invalid;
                for (int b=0;b<3;++b) {
                    const double error=Kokkos::abs(tensor[a][b]-(a==b ? volume : 0.0))/volume;
                    worst_tensor=Kokkos::max(worst_tensor,error);
                    if (!Kokkos::isfinite(error) || error>1e-9) ++invalid;
                }
            }
        },Kokkos::Sum<long long>(bad),Kokkos::Sum<long long>(edges),Kokkos::Sum<double>(volume_sum),
          Kokkos::Max<double>(max_closure),Kokkos::Max<double>(max_tensor));
    exec.fence("scaling full validation complete");
    result.volume_error=std::abs(volume_sum-1);
    result.mean_neighbors=static_cast<double>(edges)/n;
    result.closure_error=max_closure; result.tensor_error=max_tensor;
    result.validation_errors=bad+(result.volume_error>1e-11 || !std::isfinite(volume_sum));
    // Sampled all-pairs graph oracle on device; avoids an N^2 host reference.
    constexpr int samples=16;
    using WideRange=Kokkos::RangePolicy<Exec,Kokkos::IndexType<long long>>;
    long long missing=0;
    Kokkos::parallel_reduce("scaling_sampled_neighbor_oracle",WideRange(exec,0,1LL*samples*n),
        KOKKOS_LAMBDA(long long q,long long& invalid) {
            const int sample=static_cast<int>(q/n),other=static_cast<int>(q%n);
            const int row=static_cast<int>(1LL*sample*(n-1)/(samples-1));
            double d2=0;
            for (int a=0;a<3;++a) { double d=p(other,a)-p(row,a); d-=Kokkos::floor(d+0.5); d2+=d*d; }
            if (other!=row && d2<=radius*radius) {
                bool found=false;
                for (int c=0;c<counts(row);++c) found=found || table(row,c)==other;
                if (!found) ++invalid;
            }
            if (other==0)
                for (int c=0;c<counts(row);++c) {
                    const int id=table(row,c);
                    if (id<0 || id>=n || id==row) { ++invalid; continue; }
                    for (int k=0;k<c;++k) if (table(row,k)==id) ++invalid;
                    double distance2=0;
                    for (int a=0;a<3;++a) { double d=p(id,a)-p(row,a); d-=Kokkos::floor(d+0.5); distance2+=d*d; }
                    if (distance2>radius*radius) ++invalid;
                }
        },missing);
    exec.fence("scaling neighbor oracle complete");
    result.validation_errors+=missing;
    observe_memory();
    if (result.validation_errors) result.status="FAIL_VALIDATION";
    write_result(result,csv_path);
    return result.validation_errors ? 2 : 0;
}
} // namespace

int main(int argc,char** argv) {
    Kokkos::initialize(argc,argv);
    int result=1;
    try {
        const int side=argc>1 ? std::stoi(argv[1]) : 32;
        const int repeats=argc>2 ? std::stoi(argv[2]) : 5;
        const std::string policy=argc>3 ? argv[3] : "relative";
        const std::string csv=argc>4 ? argv[4] : "";
        if (side<12 || side>1024 || repeats<1 || repeats>101)
            throw std::invalid_argument("usage: construction_scaling [side 12..1024] [repeats 1..101] [relative|legacy] [CSV]");
        const std::uint64_t n=std::uint64_t(side)*side*side;
        if (n>std::numeric_limits<int>::max()) throw std::invalid_argument("particle IDs exceed int range");
        if (policy=="relative") result=run<local_voronoi::RelativeClippingTolerance>(side,repeats,policy,csv);
        else if (policy=="legacy") result=run<local_voronoi::LegacyClippingTolerance>(side,repeats,policy,csv);
        else throw std::invalid_argument("policy must be relative or legacy");
    } catch (const std::exception& error) {
        std::cerr << "scaling_error=" << error.what() << '\n';
        result=1;
    }
    Kokkos::finalize();
    return result;
}
