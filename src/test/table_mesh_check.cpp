#include "local_voronoi_table.hpp"
#include "local_voronoi_hydro.hpp"
#include "local_voronoi_team.hpp"
#include "local_voronoi_workspace.hpp"
#include "local_voronoi_retry.hpp"
#include "local_voronoi_order_retry.hpp"
#include "neighbor_search.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using Exec = Kokkos::DefaultExecutionSpace;
using Memory = Exec::memory_space;
template<class View> auto host(View view) {
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
}

template<class VertexStorage = local_voronoi::RecomputedVertexPositions,
         class Workspace = local_voronoi::PlaneWorkspace<>, bool Cooperative = false>
int check(const char* output_path, int side, neighbor_search::RowSort row_sort,
          const std::string& distribution, bool unsorted=false, bool morton=false) {
    if (side<12 || side>32) throw std::invalid_argument("side must be between 12 and 32");
    const int n = side*side*side;
    const double radius = 3.5/side;
    Kokkos::View<double*[3], Memory> positions("input_positions", n);
    Kokkos::View<double*, Memory> radii("input_radii", n);
    auto hp = Kokkos::create_mirror_view(positions);
    std::mt19937_64 rng(20261027);
    const double jitter_amplitude=distribution=="disordered" ? 0.45 :
        (distribution=="regular" ? 0.0 : 0.1);
    std::uniform_real_distribution<double> jitter(-jitter_amplitude,jitter_amplitude);
    for (int i=0;i<n;++i) {
        // Reverse input order so slot and particle IDs cannot be confused.
        const int grid = n-1-i;
        hp(i,0) = (grid/(side*side)+0.5+jitter(rng))/side;
        hp(i,1) = ((grid/side)%side+0.5+jitter(rng))/side;
        hp(i,2) = (grid%side+0.5+jitter(rng))/side;
    }
    if (distribution=="random") {
        std::uniform_real_distribution<double> uniform(0.0,1.0);
        for (int i=0;i<n;++i)
            for (int a=0;a<3;++a) hp(i,a)=uniform(rng);
    }
    if (distribution=="gradient") {
        constexpr double tau=6.2831853071795864769;
        // Smooth periodic compression: a 4:1 contrast in local x spacing.
        for (int i=0;i<n;++i) hp(i,0)-=0.6/tau*std::sin(tau*hp(i,0));
    }
    Kokkos::deep_copy(positions,hp);
    Kokkos::deep_copy(radii,radius);
    neighbor_search::DenseCellSearch<Exec> search;
    search.row_sort=row_sort;
    search.morton_cells=morton;
    const Kokkos::Array<double,3> box{1,1,1};
    // Deliberately force capacity growth, then feed the completed table directly.
    neighbor_search::configure(search,n,box,0.75*radius,true,8,!unsorted);
    const int retries = neighbor_search::rebuild(Exec{},search,positions,radii);
    int sort_errors=0;
    if (row_sort==neighbor_search::RowSort::team_bitonic) {
        // Preserve an independent copy even on CPU, where mirror_view aliases.
        auto team_ids=Kokkos::create_mirror(search.table);
        auto team_distances=Kokkos::create_mirror(search.distance2);
        Kokkos::deep_copy(team_ids,search.table);
        Kokkos::deep_copy(team_distances,search.distance2);
        search.row_sort=neighbor_search::RowSort::insertion;
        neighbor_search::rebuild(Exec{},search,positions,radii);
        auto reference_ids=host(search.table);
        auto reference_distances=host(search.distance2);
        auto reference_counts=host(search.counts);
        for (int row=0;row<n;++row)
            for (int q=0;q<reference_counts(row);++q)
                if (team_ids(row,q)!=reference_ids(row,q) ||
                    team_distances(row,q)!=reference_distances(row,q)) ++sort_errors;
        search.row_sort=row_sort;
        neighbor_search::rebuild(Exec{},search,positions,radii);
    }
    std::vector<double> search_timings;
    for (int repeat=0;repeat<5;++repeat) {
        Kokkos::Timer timer;
        neighbor_search::rebuild(Exec{},search,positions,radii);
        search_timings.push_back(timer.seconds()*1e3);
    }
    std::sort(search_timings.begin(),search_timings.end());
    neighbor_search::StageProfile profile;
    std::vector<neighbor_search::StageProfile> profiles(5);
    for (auto& sample:profiles)
        neighbor_search::rebuild(Exec{},search,positions,radii,&sample);
    auto stage_median=[&](double neighbor_search::StageProfile::*field) {
        std::vector<double> samples;
        for (const auto& sample:profiles) samples.push_back(sample.*field);
        std::sort(samples.begin(),samples.end());
        return samples[2];
    };
    profile.key_count_ms=stage_median(&neighbor_search::StageProfile::key_count_ms);
    profile.scan_ms=stage_median(&neighbor_search::StageProfile::scan_ms);
    profile.scatter_ms=stage_median(&neighbor_search::StageProfile::scatter_ms);
    profile.cell_order_gather_ms=stage_median(&neighbor_search::StageProfile::cell_order_gather_ms);
    profile.candidate_fill_ms=stage_median(&neighbor_search::StageProfile::candidate_fill_ms);
    profile.nearest_order_ms=stage_median(&neighbor_search::StageProfile::nearest_order_ms);
    profile.overflow_check_ms=stage_median(&neighbor_search::StageProfile::overflow_check_ms);
    auto p = host(search.positions);
    auto order = host(search.order);
    auto counts = host(search.counts);
    auto table = host(search.table);
    int errors = (retries == 0)+sort_errors;
    std::vector<bool> seen(n,false);
    for (int i=0;i<n;++i) {
        if (order(i)<0 || order(i)>=n || seen[order(i)]) { ++errors; continue; }
        seen[order(i)] = true;
        for (int a=0;a<3;++a) if (p(i,a)!=hp(order(i),a)) ++errors;
        std::vector<int> actual, expected;
        for (int q=0;q<counts(i);++q) actual.push_back(table(i,q));
        for (int j=0;j<n;++j) {
            if (i==j) continue;
            double d2=0;
            for (int a=0;a<3;++a) {
                double d=p(j,a)-p(i,a);
                d-=std::floor(d+0.5);
                d2+=d*d;
            }
            if (d2<=radius*radius) expected.push_back(j);
        }
        std::sort(actual.begin(),actual.end());
        if (actual!=expected) ++errors;
    }
    Kokkos::View<int*,Memory> faces("faces",n), status("status",n);
    Kokkos::View<int**,Memory> ids("ids",n,local_voronoi::max_faces);
    Kokkos::View<int**,Memory> vertices_per_face("vertices_per_face",n,local_voronoi::max_faces);
    Kokkos::View<double***,Memory> areas("areas",n,local_voronoi::max_faces,3);
    Kokkos::View<double****,Memory> vertices("vertices",n,local_voronoi::max_faces,
        local_voronoi::max_face_vertices,3);
    auto reconstruct = [&] {
        if constexpr (Workspace::plane_capacity<local_voronoi::max_planes)
            local_voronoi::reconstruct_table_retry<VertexStorage,Workspace>(Exec{},search.positions,search.counts,search.table,
                search.radii,box,faces,ids,areas,vertices_per_face,vertices,status);
        else
            local_voronoi::reconstruct_table<VertexStorage,Workspace>(Exec{},search.positions,search.counts,search.table,
                search.radii,box,faces,ids,areas,vertices_per_face,vertices,status);
        if (unsorted) local_voronoi::retry_table_order<VertexStorage,Workspace>(
            Exec{},search,faces,ids,areas,vertices_per_face,vertices,status);
    };
    reconstruct(); // Warm up before timing repeated reconstruction of identical rows.
    std::vector<double> timings;
    for (int repeat=0;repeat<5;++repeat) {
        Kokkos::Timer timer;
        reconstruct();
        timings.push_back(timer.seconds()*1e3);
    }
    std::sort(timings.begin(),timings.end());
    std::vector<double> pipeline_timings;
    for (int repeat=0;repeat<5;++repeat) {
        Kokkos::Timer timer;
        neighbor_search::rebuild(Exec{},search,positions,radii);
        reconstruct();
        pipeline_timings.push_back(timer.seconds()*1e3);
    }
    std::sort(pipeline_timings.begin(),pipeline_timings.end());
    auto hs=host(status), hf=host(faces);
    auto hi=host(ids);
    auto ha=host(areas);
    auto hv=host(vertices);
    Kokkos::View<int*,Memory> hydro_faces("hydro_faces",n), hydro_status("hydro_status",n);
    Kokkos::View<int**,Memory> hydro_ids("hydro_ids",n,local_voronoi::max_faces);
    Kokkos::View<double***,Memory> hydro_areas("hydro_areas",n,local_voronoi::max_faces,3);
    Kokkos::View<double***,Memory> face_centroids("face_centroids",n,local_voronoi::max_faces,3);
    Kokkos::View<double*,Memory> volumes("volumes",n);
    Kokkos::View<double**,Memory> cell_centroids("cell_centroids",n,3);
    auto reconstruct_hydro=[&] {
        if constexpr (Cooperative)
            local_voronoi::reconstruct_hydro_team(Exec{},search.positions,search.counts,search.table,
                search.radii,box,hydro_faces,hydro_ids,hydro_areas,face_centroids,
                volumes,cell_centroids,hydro_status);
        else if constexpr (Workspace::plane_capacity<local_voronoi::max_planes)
            local_voronoi::reconstruct_hydro_retry<VertexStorage,Workspace>(Exec{},search.positions,search.counts,search.table,
                search.radii,box,hydro_faces,hydro_ids,hydro_areas,face_centroids,
                volumes,cell_centroids,hydro_status);
        else
            local_voronoi::reconstruct_hydro<VertexStorage,Workspace>(Exec{},search.positions,search.counts,search.table,
                search.radii,box,hydro_faces,hydro_ids,hydro_areas,face_centroids,
                volumes,cell_centroids,hydro_status);
        if (unsorted) local_voronoi::retry_hydro_order<VertexStorage,Workspace>(
            Exec{},search,hydro_faces,hydro_ids,hydro_areas,face_centroids,
            volumes,cell_centroids,hydro_status);
    };
    reconstruct_hydro();
    std::vector<double> hydro_timings, hydro_pipeline_timings;
    for (int repeat=0;repeat<5;++repeat) {
        Kokkos::Timer timer;
        reconstruct_hydro();
        hydro_timings.push_back(timer.seconds()*1e3);
        timer.reset();
        neighbor_search::rebuild(Exec{},search,positions,radii);
        reconstruct_hydro();
        hydro_pipeline_timings.push_back(timer.seconds()*1e3);
    }
    std::sort(hydro_timings.begin(),hydro_timings.end());
    std::sort(hydro_pipeline_timings.begin(),hydro_pipeline_timings.end());
    // Compact geometry rejects degenerate faces/moments that the diagnostic
    // vertex writer does not validate. Its order retry can therefore sort more
    // rows. Refresh the diagnostic oracle against that final candidate table.
    if (unsorted) {
        reconstruct();
        hs=host(status); hf=host(faces); hi=host(ids); ha=host(areas); hv=host(vertices);
    }
    auto hhs=host(hydro_status), hhf=host(hydro_faces);
    auto hhi=host(hydro_ids);
    auto hha=host(hydro_areas);
    auto hfc=host(face_centroids);
    auto hvol=host(volumes);
    auto hcc=host(cell_centroids);
    auto hnverts=host(vertices_per_face);
    Kokkos::View<local_voronoi::WorkspaceUsage*,Memory> workspace_usage("workspace_usage",n);
    local_voronoi::profile_workspace<VertexStorage,Workspace>(Exec{},search.positions,
        search.counts,search.table,search.radii,box,workspace_usage);
    auto usage=host(workspace_usage);
    int fallback_rows=0;
    Kokkos::View<local_voronoi::WorkspaceUsage*,Memory> fallback_usage;
    if constexpr (Workspace::plane_capacity<local_voronoi::max_planes) {
        for (int i=0;i<n;++i) if (usage(i).status==local_voronoi::plane_capacity) ++fallback_rows;
        fallback_usage=decltype(fallback_usage)("fallback_usage",n);
        local_voronoi::profile_workspace<VertexStorage,local_voronoi::PlaneWorkspace<
            local_voronoi::max_planes,typename Workspace::tolerance,Workspace::early_stop>>(Exec{},search.positions,
            search.counts,search.table,search.radii,box,fallback_usage);
        usage=host(fallback_usage);
    }
    auto peak=[&](int local_voronoi::WorkspaceUsage::*field) {
        int value=0;
        for (int i=0;i<n;++i) value=std::max(value,usage(i).*field);
        return value;
    };
    long long tested_vertices=0, read_coordinates=0, created_vertices=0, intersection_solves=0;
    for (int i=0;i<n;++i) {
        if (usage(i).status!=hs(i) || usage(i).faces!=hf(i)) ++errors;
        tested_vertices+=usage(i).vertex_tests;
        read_coordinates+=usage(i).coordinate_reads;
        created_vertices+=usage(i).created_vertices;
        intersection_solves+=usage(i).intersection_solves;
    }
    std::ofstream output;
    std::ofstream hydro_output;
    if (output_path) {
        output.open(output_path);
        if (!output) throw std::runtime_error("cannot open comparison output");
        output << std::setprecision(17);
        hydro_output.open(std::string(output_path)+".hydro.csv");
        if (!hydro_output) throw std::runtime_error("cannot open hydro comparison output");
        hydro_output << std::setprecision(17);
    }
    double total_volume=0, max_closure=0, max_pair_error=0;
    for (int i=0;i<n;++i) {
        if (hs(i)!=local_voronoi::success) { ++errors; continue; }
        if (hhs(i)!=hs(i) || hhf(i)!=hf(i)) { ++errors; continue; }
        double volume=0, closure[3]={0,0,0};
        double centroid_area_tensor[3][3]{};
        local_voronoi::Vec3 reference_moment{0,0,0};
        for (int f=0;f<hf(i);++f) {
            const int j=hi(i,f);
            if (j<0 || j>=n) { ++errors; continue; }
            int reverse=-1;
            for (int g=0;g<hf(j);++g) if (hi(j,g)==i) reverse=g;
            if (reverse<0) ++errors;
            if (hhi(i,f)!=j) ++errors;
            auto vertex=[&](int v) -> local_voronoi::Vec3 {
                return {hv(i,f,v,0),hv(i,f,v,1),hv(i,f,v,2)};
            };
            double face_weight=0;
            local_voronoi::Vec3 reference_face_moment{0,0,0};
            using namespace local_voronoi;
            const Vec3 anchor=vertex(0);
            for (int v=1;v+1<hnverts(i,f);++v) {
                const Vec3 b=vertex(v), c=vertex(v+1), sum=add(add(anchor,b),c);
                const double weight=norm(cross(sub(b,anchor),sub(c,anchor)));
                face_weight+=weight;
                reference_face_moment=add(reference_face_moment,mul(sum,weight/3));
                reference_moment=add(reference_moment,mul(sum,dot(anchor,cross(b,c))/24));
            }
            const Vec3 reference_face=mul(reference_face_moment,1/face_weight);
            const double reference_components[3]={reference_face.x,reference_face.y,reference_face.z};
            for (int a=0;a<3;++a) {
                for (int b=0;b<3;++b)
                    centroid_area_tensor[a][b]+=hfc(i,f,a)*hha(i,f,b);
                closure[a]+=ha(i,f,a);
                volume+=ha(i,f,a)*hv(i,f,0,a)/3;
                if (reverse>=0) max_pair_error=std::max(max_pair_error,
                    std::abs(ha(i,f,a)+ha(j,reverse,a)));
                if (std::abs(hha(i,f,a)-ha(i,f,a))>1e-12 ||
                    !std::isfinite(hfc(i,f,a)) ||
                    std::abs(hfc(i,f,a)-reference_components[a])>1e-11) ++errors;
                if (reverse>=0) {
                    double delta=p(i,a)+hfc(i,f,a)-p(j,a)-hfc(j,reverse,a);
                    delta-=std::floor(delta+0.5);
                    if (std::abs(delta)>1e-11) ++errors;
                }
            }
            if (output_path) output << order(i) << ',' << order(j) << ','
                << ha(i,f,0) << ',' << ha(i,f,1) << ',' << ha(i,f,2) << '\n';
            if (output_path) hydro_output << order(i) << ',' << order(j) << ','
                << hha(i,f,0) << ',' << hha(i,f,1) << ',' << hha(i,f,2) << ','
                << hfc(i,f,0) << ',' << hfc(i,f,1) << ',' << hfc(i,f,2) << ','
                << hvol(i) << ',' << hcc(i,0) << ',' << hcc(i,1) << ',' << hcc(i,2) << '\n';
            // Independent nearest-site oracle for vertices of 32 sampled cells.
            // This checks all particles, including those outside the candidate
            // radius, and uses the nearest periodic image of the vertex itself.
            if (i%std::max(1,n/32)==0) {
                for (int v=0;v<hnverts(i,f);++v) {
                    double own_distance2=0;
                    for (int a=0;a<3;++a) own_distance2+=hv(i,f,v,a)*hv(i,f,v,a);
                    for (int other=0;other<n;++other) {
                        double other_distance2=0;
                        for (int a=0;a<3;++a) {
                            double d=p(other,a)-p(i,a)-hv(i,f,v,a);
                            d-=std::floor(d+0.5);
                            other_distance2+=d*d;
                        }
                        if (!std::isfinite(other_distance2) || other_distance2+
                            1e-10*std::max(own_distance2,1.0/(side*side))<own_distance2) ++errors;
                    }
                }
            }
        }
        if (!(volume>0)) ++errors;
        if (!std::isfinite(hvol(i)) || std::abs(hvol(i)-volume)>1e-11*volume) ++errors;
        // Divergence theorem: sum_f centroid_f (area_f)^T = V I. It checks
        // face centroids without repeating the triangle integration formula.
        for (int a=0;a<3;++a)
            for (int b=0;b<3;++b)
                if (!std::isfinite(centroid_area_tensor[a][b]) ||
                    std::abs(centroid_area_tensor[a][b]-(a==b ? volume : 0.0))>1e-10*volume) ++errors;
        const double reference_components[3]={reference_moment.x,reference_moment.y,reference_moment.z};
        for (int a=0;a<3;++a)
            if (!std::isfinite(hcc(i,a)) ||
                std::abs(hcc(i,a)-reference_components[a]/volume)>1e-11) ++errors;
        total_volume+=volume;
        for (double c:closure) max_closure=std::max(max_closure,std::abs(c));
    }
    if (std::abs(total_volume-1)>1e-12 || max_closure>1e-9 || max_pair_error>1e-9) ++errors;

    // A truncated or invalid row must report failure without reading padding.
    counts(0)=search.capacity+1;
    counts(1)=-1;
    Kokkos::deep_copy(search.counts,counts);
    reconstruct();
    reconstruct_hydro();
    hs=host(status); hf=host(faces);
    hhs=host(hydro_status); hhf=host(hydro_faces); hvol=host(volumes); hcc=host(cell_centroids);
    for (int i=0;i<2;++i)
        if (hs(i)!=local_voronoi::invalid_candidate_count || hf(i)!=0 ||
            hhs(i)!=local_voronoi::invalid_candidate_count || hhf(i)!=0 || hvol(i)!=0 ||
            hcc(i,0)!=0 || hcc(i,1)!=0 || hcc(i,2)!=0) ++errors;
    // An incomplete candidate set must fail certification, with no usable faces.
    counts(0)=0;
    Kokkos::deep_copy(search.counts,counts);
    reconstruct();
    reconstruct_hydro();
    hs=host(status); hf=host(faces);
    hhs=host(hydro_status); hhf=host(hydro_faces); hvol=host(volumes);
    if (hs(0)!=local_voronoi::uncertified || hf(0)!=0) ++errors;
    if (hhs(0)!=local_voronoi::uncertified || hhf(0)!=0 || hvol(0)!=0) ++errors;
    bool rejected=false;
    try {
        Kokkos::View<int*,Memory> short_counts("short_counts",n-1);
        local_voronoi::reconstruct_table(Exec{},search.positions,short_counts,search.table,
            search.radii,box,faces,ids,areas,vertices_per_face,vertices,status);
    } catch (const std::invalid_argument&) { rejected=true; }
    if (!rejected) ++errors;
    const std::size_t diagnostic_bytes=faces.size()*sizeof(int)+status.size()*sizeof(int)+
        ids.size()*sizeof(int)+vertices_per_face.size()*sizeof(int)+
        areas.size()*sizeof(double)+vertices.size()*sizeof(double);
    const std::size_t hydro_bytes=hydro_faces.size()*sizeof(int)+hydro_status.size()*sizeof(int)+
        hydro_ids.size()*sizeof(int)+hydro_areas.size()*sizeof(double)+
        face_centroids.size()*sizeof(double)+volumes.size()*sizeof(double)+
        cell_centroids.size()*sizeof(double);
    std::cout << "backend=" << Exec::name() << " particles=" << n
              << " vertices=" << (std::is_same_v<VertexStorage,local_voronoi::CachedVertexPositions> ? "cached" : "recomputed")
              << " distribution=" << distribution
              << " plane_capacity=" << Workspace::plane_capacity
              << " clipping_policy=" << (std::is_same_v<typename Workspace::tolerance,
                  local_voronoi::RelativeClippingTolerance> ? "relative_pivoted" : "legacy")
              << " fallback_rows=" << fallback_rows
              << " position_cache_bytes_per_cell=" <<
                  (std::is_same_v<VertexStorage,local_voronoi::CachedVertexPositions> ?
                      sizeof(local_voronoi::CachedVertexPositions) : 0)
              << " morton_cells=" << morton << " nearest_first=" << !unsorted
              << " row_sort=" << (row_sort==neighbor_search::RowSort::team_bitonic_recompute ? "team_bitonic_recompute" : row_sort==neighbor_search::RowSort::team_bitonic ? "team_bitonic" : "insertion")
              << " capacity_retries=" << retries << " volume_error=" << std::abs(total_volume-1)
              << " reconstruction_median_ms=" << timings[2]
              << " search_rebuild_median_ms=" << search_timings[2]
              << " pipeline_median_ms=" << pipeline_timings[2]
              << " hydro_reconstruction_median_ms=" << hydro_timings[2]
              << " hydro_pipeline_median_ms=" << hydro_pipeline_timings[2]
              << " diagnostic_output_bytes=" << diagnostic_bytes
              << " hydro_output_bytes=" << hydro_bytes
              << " max_closure=" << max_closure << " max_pair_error=" << max_pair_error
              << " errors=" << errors << '\n';
    std::cout << "profile_backend=" << Exec::name() << " particles=" << n
              << " morton_cells=" << morton << " nearest_first=" << !unsorted
              << " row_sort=" << (row_sort==neighbor_search::RowSort::team_bitonic_recompute ? "team_bitonic_recompute" : row_sort==neighbor_search::RowSort::team_bitonic ? "team_bitonic" : "insertion")
              << " key_count_ms=" << profile.key_count_ms << " scan_ms=" << profile.scan_ms
              << " scatter_ms=" << profile.scatter_ms
              << " cell_order_gather_ms=" << profile.cell_order_gather_ms
              << " candidate_fill_ms=" << profile.candidate_fill_ms
              << " nearest_order_ms=" << profile.nearest_order_ms
              << " overflow_check_ms=" << profile.overflow_check_ms << '\n';
    std::cout << "workspace_backend=" << Exec::name() << " particles=" << n
              << " distribution=" << distribution
              << " plane_capacity=" << Workspace::plane_capacity
              << " peak_planes=" << peak(&local_voronoi::WorkspaceUsage::planes)
              << " peak_vertices=" << peak(&local_voronoi::WorkspaceUsage::vertices)
              << " peak_hole_edges=" << peak(&local_voronoi::WorkspaceUsage::hole_edges)
              << " peak_faces=" << peak(&local_voronoi::WorkspaceUsage::faces)
              << " peak_face_vertices=" << peak(&local_voronoi::WorkspaceUsage::face_vertices)
              << " vertex_tests=" << tested_vertices << " coordinate_reads=" << read_coordinates
              << " created_vertices=" << created_vertices << " intersection_solves=" << intersection_solves << '\n';
    return errors ? 2 : 0;
}
} // namespace

int main(int argc,char** argv) {
    Kokkos::initialize(argc,argv);
    int result;
    {
        const std::string mode=argc>3 ? argv[3] : "insertion";
        if (mode!="insertion" && mode!="team_bitonic" && mode!="team_bitonic_recompute" && mode!="unsorted" && mode!="morton" && mode!="morton_unsorted")
            throw std::invalid_argument("sort must be insertion, team_bitonic, team_bitonic_recompute, unsorted, morton or morton_unsorted");
        const std::string vertices=argc>4 ? argv[4] : "recomputed";
        const std::string distribution=argc>5 ? argv[5] : "lattice";
        const int planes=argc>6 ? std::stoi(argv[6]) : local_voronoi::max_planes;
        const std::string tolerance=argc>7 ? argv[7] : "legacy";
        if (tolerance!="legacy" && tolerance!="relative" && tolerance!="relative_early" && tolerance!="cooperative")
            throw std::invalid_argument("clipping policy must be legacy, relative, relative_early or cooperative");
        if (tolerance!="legacy" && vertices!="cached")
            throw std::invalid_argument("relative benchmark uses cached vertices");
        if (planes!=64 && planes!=local_voronoi::max_planes)
            throw std::invalid_argument("plane capacity must be 64 or 160");
        if (vertices!="recomputed" && vertices!="cached")
            throw std::invalid_argument("vertices must be recomputed or cached");
        if (distribution!="lattice" && distribution!="disordered" && distribution!="regular" &&
            distribution!="random" && distribution!="gradient")
            throw std::invalid_argument("distribution must be lattice, disordered, regular, random or gradient");
        const auto sort=mode=="team_bitonic_recompute" ? neighbor_search::RowSort::team_bitonic_recompute : mode=="team_bitonic" ? neighbor_search::RowSort::team_bitonic : neighbor_search::RowSort::insertion;
        if (mode=="unsorted" || mode=="morton" || mode=="morton_unsorted") {
            if (tolerance!="relative_early" || planes!=64 || vertices!="cached")
                throw std::invalid_argument("layout experiment requires cached,64,relative_early");
            result=check<local_voronoi::CachedVertexPositions,local_voronoi::PlaneWorkspace<64,local_voronoi::RelativeClippingTolerance,true>>(
                argc>1 ? argv[1] : nullptr,argc>2 ? std::stoi(argv[2]) : 12,
                neighbor_search::RowSort::team_bitonic_recompute,distribution,mode!="morton",mode!="unsorted");
        }
        else if (tolerance=="cooperative")
            result=check<local_voronoi::CachedVertexPositions,local_voronoi::PlaneWorkspace<160,local_voronoi::RelativeClippingTolerance>,true>(argc>1 ? argv[1] : nullptr,argc>2 ? std::stoi(argv[2]) : 12,sort,distribution);
        else if (tolerance=="relative_early" && planes==64)
            result=check<local_voronoi::CachedVertexPositions,local_voronoi::PlaneWorkspace<64,local_voronoi::RelativeClippingTolerance,true>>(argc>1 ? argv[1] : nullptr,argc>2 ? std::stoi(argv[2]) : 12,sort,distribution);
        else if (tolerance=="relative_early")
            result=check<local_voronoi::CachedVertexPositions,local_voronoi::PlaneWorkspace<160,local_voronoi::RelativeClippingTolerance,true>>(argc>1 ? argv[1] : nullptr,argc>2 ? std::stoi(argv[2]) : 12,sort,distribution);
        else if (tolerance=="relative" && planes==64)
            result=check<local_voronoi::CachedVertexPositions,local_voronoi::PlaneWorkspace<
                64,local_voronoi::RelativeClippingTolerance>>(argc>1 ? argv[1] : nullptr,
                argc>2 ? std::stoi(argv[2]) : 12,sort,distribution);
        else if (tolerance=="relative")
            result=check<local_voronoi::CachedVertexPositions,local_voronoi::PlaneWorkspace<
                160,local_voronoi::RelativeClippingTolerance>>(argc>1 ? argv[1] : nullptr,
                argc>2 ? std::stoi(argv[2]) : 12,sort,distribution);
        else if (vertices=="cached" && planes==64)
            result=check<local_voronoi::CachedVertexPositions,local_voronoi::PlaneWorkspace<64>>(
                argc>1 ? argv[1] : nullptr,argc>2 ? std::stoi(argv[2]) : 12,sort,distribution);
        else if (vertices=="recomputed" && planes==64)
            result=check<local_voronoi::RecomputedVertexPositions,local_voronoi::PlaneWorkspace<64>>(
                argc>1 ? argv[1] : nullptr,argc>2 ? std::stoi(argv[2]) : 12,sort,distribution);
        else if (vertices=="cached")
            result=check<local_voronoi::CachedVertexPositions>(argc>1 ? argv[1] : nullptr,
                argc>2 ? std::stoi(argv[2]) : 12,sort,distribution);
        else
            result=check(argc>1 ? argv[1] : nullptr,argc>2 ? std::stoi(argv[2]) : 12,sort,distribution);
    }
    Kokkos::finalize();
    return result;
}
