// Liu, Ma, Guo, Yan, "Parallel Computation of 3D Clipped Voronoi Diagrams", IEEE TVCG 28(2) 2022, in the library
// benchmark on one A100. Their code (github.com/xh-liu-tech/3D-Voronoi-GPU) computes the Voronoi diagram of sites
// clipped by a tetrahedral mesh, in float, VoroGPU-style (kNN + clipping by bisector planes with the security radius),
// with two strategies that this driver times as two variants:
//   "cell-centric": voro_cell_test_GPU_param (each site's cell in the box [-0.1, 1000.1]^3, stored as a VoronoiCell)
//                   then clipped_voro_cell_test_GPU_param (each (tetrahedron, candidate site) pair clips the stored
//                   cell by the tetrahedron and adds its volume),
//   "tet-centric":  clipped_voro_cell_test_GPU_param_tet (each pair starts from the tetrahedron and clips it by the
//                   site's K nearest neighbours).
// Here the domain is the padded periodic box (gpu_images.hpp) and the mesh is ONE tetrahedron enclosing [0, 1000]^3,
// so the clipping by the mesh changes no original cell, and its candidate list is every site (tet_k = number of
// sites). compute_clipped_voro_diagram_GPU finds the candidates with a brute-force kNN between tetrahedron centroids
// and sites (kNN-CUDA: an n_tet x n_site distance matrix and an insertion sort with k = tet_k), which with k = all
// sites is O(n^2); the driver passes the trivial list instead and launches the same kernels in the same order (it
// compiles the upstream voronoi.cu into this unit). Volumes are those of the cells whose status is success (security
// radius reached, no overflow); others are NaN and counted.
//   voronoi_gpu_liu <points file> [--repeats=3] [--width=<shell width in mean spacings>] [--volumes=file]
// K from params.h (PRESET 0, upstream default: K 90).

#include "libbench.hpp"
#include "gpu_images.hpp"

#include "voronoi.cu"  // upstream (with voronoi-gpu-liu.patch): kernels, ConvexCell, VoronoiCell

namespace {

__global__ void iota(int m, int* a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) a[i] = i;
}

// Volumes of the originals in input order: success -> volume / scale^3, else NaN. Status per site: the stored cell's
// (cell-centric) or the pair's (tet-centric; with one tetrahedron, pair = site).
__global__ void gather(int m, int n, const unsigned* perm, const float* vol, const Status* pair_stat,
                       const VoronoiCell* cells, double inv_scale3, double* volume, Status* site_stat) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= m) return;
    const Status st = cells ? (cells[pos].status == success ? pair_stat[pos] : cells[pos].status) : pair_stat[pos];
    site_stat[pos] = st;
    const unsigned i = perm[pos];
    if (i >= unsigned(n)) return;
    volume[i] = st == success ? double(vol[pos]) * inv_scale3 : __longlong_as_double(0x7ff8000000000000LL);
}

struct Result {
    std::vector<double> volume;
    std::vector<int> failed;
    std::uint64_t padded = 0;
};

void run(const double* x, std::uint64_t n, double w, bool cell_centric, Result& r, bool stats) {
    gpu_images::Padded p;
    gpu_images::pad(x, n, w, p);
    std::vector<float> pts;
    const double scale = gpu_images::to_box1000(p, pts, 3);
    const int m = int(p.size());
    r.padded = p.size();
    // the enclosing tetrahedron, transposed as copy_tet_data stores it (vert pitch 4, idx pitch 1)
    const float a = -10.f, L = 3100.f;
    const float vert[12] = {a, a + L, a, a, a, a, a + L, a, a, a, a, a + L};  // x of 4 vertices, y, z
    const int idx[4] = {0, 1, 2, 3};
    float* vert_dev = nullptr;
    int* idx_dev = nullptr;
    cuda_check(cudaMalloc(&vert_dev, sizeof vert));
    cuda_check(cudaMalloc(&idx_dev, sizeof idx));
    cuda_check(cudaMemcpy(vert_dev, vert, sizeof vert, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(idx_dev, idx, sizeof idx, cudaMemcpyHostToDevice));

    kn_problem* kn = kn_prepare(reinterpret_cast<float3*>(pts.data()), m);  // site_is_transposed = false path
    kn_solve(kn);
    const float* site = reinterpret_cast<const float*>(kn->d_stored_points);
    const int* site_knn = reinterpret_cast<const int*>(kn->d_knearests);
    int* tet_knn = nullptr;
    Status* pair_stat = nullptr;
    float *bary = nullptr, *vol = nullptr;
    VoronoiCell* cells = nullptr;
    double* volume = nullptr;
    Status* site_stat = nullptr;
    cuda_check(cudaMalloc(&tet_knn, sizeof(int) * std::size_t(m)));
    cuda_check(cudaMalloc(&pair_stat, sizeof(Status) * std::size_t(m)));
    cuda_check(cudaMalloc(&bary, sizeof(float) * 3 * std::size_t(m)));
    cuda_check(cudaMalloc(&vol, sizeof(float) * std::size_t(m)));
    cuda_check(cudaMalloc(&volume, sizeof(double) * n));
    cuda_check(cudaMalloc(&site_stat, sizeof(Status) * std::size_t(m)));
    iota<<<(m + 255) / 256, 256>>>(m, tet_knn);
    cuda_check(cudaMemset(bary, 0, sizeof(float) * 3 * std::size_t(m)));
    cuda_check(cudaMemset(vol, 0, sizeof(float) * std::size_t(m)));
    const int n_tet = 1, tet_k = m;
    if (cell_centric) {
        cuda_check(cudaMalloc(&cells, sizeof(VoronoiCell) * std::size_t(m)));
        voro_cell_test_GPU_param<<<m / VORO_BLOCK_SIZE + 1, VORO_BLOCK_SIZE>>>(site, m, 0, site_knn, 0, cells);
        cuda_check_error();
        clipped_voro_cell_test_GPU_param<<<n_tet * tet_k / VORO_BLOCK_SIZE + 1, VORO_BLOCK_SIZE>>>(
            vert_dev, 4, 4, idx_dev, n_tet, 1, tet_knn, 1, tet_k, pair_stat, cells, bary, 0, vol);
        cuda_check_error();
    } else {
        clipped_voro_cell_test_GPU_param_tet<<<n_tet * tet_k / VORO_BLOCK_SIZE + 1, VORO_BLOCK_SIZE>>>(
            site, m, 0, site_knn, 0, vert_dev, 4, 4, idx_dev, n_tet, 1, tet_knn, 1, tet_k, pair_stat, cells, bary, 0,
            vol);
        cuda_check_error();
    }
    gather<<<(m + 255) / 256, 256>>>(m, int(n), kn->d_permutation, vol, pair_stat, cells, 1.0 / (scale * scale * scale),
                                     volume, site_stat);
    cuda_check_error();
    r.volume.resize(n);
    cuda_check(cudaMemcpy(r.volume.data(), volume, sizeof(double) * n, cudaMemcpyDeviceToHost));
    if (stats) {
        std::vector<Status> s(m);
        std::vector<unsigned> perm(m);
        cuda_check(cudaMemcpy(s.data(), site_stat, sizeof(Status) * m, cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(perm.data(), kn->d_permutation, sizeof(unsigned) * m, cudaMemcpyDeviceToHost));
        r.failed.assign(8, 0);
        for (int pos = 0; pos < m; ++pos)
            if (perm[pos] < n && s[pos] != success) ++r.failed[int(s[pos])];
    }
    cudaFree(vert_dev);
    cudaFree(idx_dev);
    cudaFree(tet_knn);
    cudaFree(pair_stat);
    cudaFree(bary);
    cudaFree(vol);
    cudaFree(cells);
    cudaFree(volume);
    cudaFree(site_stat);
    kn_free(&kn);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: voronoi_gpu_liu <points file> [--repeats=3] [--width=spacings] [--volumes=file]\n";
        return 1;
    }
    const std::string file = argv[1];
    const int repeats = std::stoi(libbench::option(argc, argv, "repeats", "3"));
    const auto frames = libbench::read_frames(file);
    const std::uint64_t n = frames.n;
    const std::string input = libbench::input_name(file);
    const double spacings = std::stod(libbench::option(argc, argv, "width", "0"));
    const double ws = spacings > 0 ? spacings : gpu_images::default_width_spacings(input);
    const double w = ws * std::cbrt(1.0 / double(n));
    const double* x = frames.frame(0);
    const std::string only = libbench::option(argc, argv, "variant", "");
    cudaSetDevice(0);
    cudaFree(nullptr);

    libbench::Setup s;
    s.library = "Liu et al. 3D-Voronoi-GPU";
    s.version = libbench::env("LIBBENCH_LIB_VERSION", "b2c64f5 + voronoi-gpu-liu.patch");
    s.device = "A100-SXM4-40GB";
    s.threads = 1;
    s.outputs = "volumes (and barycentres)";
    std::ostringstream per;
    per << "images: shell of width " << ws << " mean spacings (" << w << ") around the unit box; clipping mesh: one "
        << "tetrahedron enclosing it";
    s.periodic = per.str();
    s.precision = "float (points scaled to [0, 1000]^3), certified by the security radius; no exact predicates";
    s.note = "K " + std::to_string(_K_) + "; failed cells NaN; candidate list of the one tetrahedron = all sites "
             "(replaces the brute-force tet-site kNN of compute_clipped_voro_diagram_GPU); padding on " +
             std::to_string(gpu_images::threads()) + " OpenMP threads";

    for (int variant = 0; variant < 2; ++variant) {
        const bool cell_centric = variant == 0;
        if (!only.empty() && only != (cell_centric ? "cell" : "tet")) continue;
        s.variant = cell_centric ? "cell-centric: cells, then clipped by the tetrahedra"
                                 : "tet-centric: tetrahedra clipped by the cells";
        s.timed = std::string("points in host memory (double) to volumes in host memory: padding, float conversion, "
                              "kn_prepare (H2D, grid sort), kn_solve, ") +
                  (cell_centric ? "voro_cell_test_GPU_param + clipped_voro_cell_test_GPU_param"
                                : "clipped_voro_cell_test_GPU_param_tet") +
                  ", gather by the permutation, D2H";
        Result r;
        double peak = 0;
        const double start = libbench::now();  // as run_protocol: warm-up runs until 2 s after this
        {
            libbench::DevicePeak dp;
            dp.start();
            run(x, n, w, cell_centric, r, true);
            peak = dp.stop();
        }
        const std::string vfile = libbench::option(argc, argv, "volumes");
        if (!vfile.empty() && cell_centric) libbench::write_volumes(vfile, r.volume);
        if (!vfile.empty() && !cell_centric) {
            std::string f = vfile;
            const auto at = f.find("voronoi-gpu-liu");
            if (at != std::string::npos) f.replace(at, 15, "voronoi-gpu-liu-tet");
            libbench::write_volumes(f, r.volume);
        }
        long long nan = 0;
        double sum = 0;
        for (double v : r.volume) (std::isnan(v) ? (void)++nan : (void)(sum += v));
        std::ostringstream o;
        o << ",\"K\":" << _K_ << ",\"width\":" << w << ",\"padded_points\":" << r.padded << ",\"failed_cells\":" << nan
          << ",\"failed_by_status\":{\"triangle_overflow\":" << r.failed[0] << ",\"vertex_overflow\":" << r.failed[1]
          << ",\"inconsistent_boundary\":" << r.failed[2] << ",\"security_radius_not_reached\":" << r.failed[3]
          << ",\"needs_exact_predicates\":" << r.failed[5] << ",\"no_intersection\":" << r.failed[6] << "}";
        std::cerr << "# liu " << input << " " << n << " " << s.variant << ": sum of volumes - 1 = " << sum - 1.0
                  << o.str() << std::endl;
        while (libbench::now() - start < 2.0) {
            Result t;
            run(x, n, w, cell_centric, t, false);
        }
        for (int rep = 1; rep <= repeats; ++rep) {
            Result t;
            const double t0 = libbench::now();
            run(x, n, w, cell_centric, t, false);
            libbench::record(s, file, input, n, "build", rep, libbench::now() - t0, peak, o.str());
        }
    }
    return 0;
}
