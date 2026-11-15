// VoroGPU (Ray, Sokolov, Lefebvre, Levy, "Meshless Voronoi on the GPU", ACM TOG 37(6) 2018) in the library
// benchmark on one A100: k nearest neighbours on a grid (knearests.cu: kn_prepare, kn_solve), then one thread per
// cell clips the box [-0.1, 1000.1]^3 by the bisector planes of its K nearest neighbours in order, in float, until the
// security radius is reached (the cell is certified: no farther point can cut it); a cell whose K neighbours do not
// reach the radius, or that overflows its local arrays, gets a failure status (the paper's CPU fallback is not in the
// public code; failed cells are NaN here and counted). This compiles the upstream voronoi.cu into this unit to launch
// its kernel (voro_cell_test_GPU_param) directly: compute_voro_diagram_GPU copies points and results through host
// vectors in the sorted order and loses the permutation, so the driver does the same steps and maps the volumes back.
//   vorogpu <points file> [--repeats=3] [--width=<shell width in mean spacings>] [--volumes=file]
// K is fixed at compile time (params.h PRESET: 0 = K 190 "white noise", 1 = K 90 "perturbed grid", 2 = K 35 "blue
// noise"); the build makes one binary per preset.
// Not periodic: periodic images in a shell (gpu_images.hpp), scaled to [0, 1000]^3 in float, are the input.

#include "libbench.hpp"
#include "gpu_images.hpp"

#include "voronoi.cu"  // upstream (with vorogpu.patch): kernels and ConvexCell

namespace {

// Volumes of the originals in input order from the sorted cells: status success -> w / scale^3, otherwise NaN.
__global__ void gather(int m, int n, const unsigned* perm, const float* out, const Status* stat, double inv_scale3,
                       double* volume) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= m) return;
    const unsigned i = perm[pos];
    if (i >= unsigned(n)) return;
    volume[i] = stat[pos] == success ? double(out[4 * pos + 3]) * inv_scale3 : __longlong_as_double(0x7ff8000000000000LL);
}

struct Result {
    std::vector<double> volume;
    std::vector<int> failed;  // per status, originals only
    std::uint64_t padded = 0;
};

void run(const double* x, std::uint64_t n, double w, Result& r, bool stats) {
    gpu_images::Padded p;
    gpu_images::pad(x, n, w, p);
    std::vector<float> pts;
    const double scale = gpu_images::to_box1000(p, pts, 3);
    const int m = int(p.size());
    r.padded = p.size();
    kn_problem* kn = kn_prepare(reinterpret_cast<float3*>(pts.data()), m);
    kn_solve(kn);
    float* out = nullptr;
    Status* stat = nullptr;
    double* vol = nullptr;
    cuda_check(cudaMalloc(&out, sizeof(float) * 4 * std::size_t(m)));
    cuda_check(cudaMalloc(&stat, sizeof(Status) * std::size_t(m)));
    cuda_check(cudaMalloc(&vol, sizeof(double) * n));
    cuda_check(cudaMemset(out, 0, sizeof(float) * 4 * std::size_t(m)));
    // as compute_voro_diagram_GPU: <<<nbpts / 16 + 1, 16>>> on the sorted points; no tetrahedra (tet < 0: the box)
    voro_cell_test_GPU_param<<<m / 16 + 1, 16>>>(reinterpret_cast<float*>(kn->d_stored_points), m, 0,
                                                 kn->d_knearests, stat, out, nullptr, nullptr);
    cuda_check_error();
    gather<<<(m + 255) / 256, 256>>>(m, int(n), kn->d_permutation, out, stat, 1.0 / (scale * scale * scale), vol);
    cuda_check_error();
    r.volume.resize(n);
    cuda_check(cudaMemcpy(r.volume.data(), vol, sizeof(double) * n, cudaMemcpyDeviceToHost));
    if (stats) {
        std::vector<Status> s(m);
        std::vector<unsigned> perm(m);
        cuda_check(cudaMemcpy(s.data(), stat, sizeof(Status) * m, cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(perm.data(), kn->d_permutation, sizeof(unsigned) * m, cudaMemcpyDeviceToHost));
        r.failed.assign(8, 0);
        for (int pos = 0; pos < m; ++pos)
            if (perm[pos] < n && s[pos] != success) ++r.failed[int(s[pos])];
    }
    cudaFree(out);
    cudaFree(stat);
    cudaFree(vol);
    kn_free(&kn);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: vorogpu <points file> [--repeats=3] [--width=spacings] [--volumes=file]\n";
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
    cudaSetDevice(0);
    cudaFree(nullptr);

    libbench::Setup s;
    s.library = "VoroGPU";
    s.variant = "kNN + cell clipping, K " + std::to_string(_K_) + " (PRESET " + std::to_string(PRESET) + ")";
    s.version = libbench::env("LIBBENCH_LIB_VERSION", "f4d7e7e + vorogpu.patch");
    s.device = "A100-SXM4-40GB";
    s.threads = 1;
    s.outputs = "volumes (and barycentres)";
    std::ostringstream per;
    per << "images: shell of width " << ws << " mean spacings (" << w << ") around the unit box, in the box "
        << "[-0.1, 1000.1]^3 of the code";
    s.periodic = per.str();
    s.precision = "float (points scaled to [0, 1000]^3), certified by the security radius; no exact predicates";
    s.timed = "points in host memory (double) to volumes in host memory: padding, float conversion, kn_prepare "
              "(H2D, grid sort), kn_solve, cell kernel, gather by the permutation, D2H";

    Result r;
    double peak = 0;
    const double start = libbench::now();  // as run_protocol: warm-up runs until 2 s after this
    {
        libbench::DevicePeak dp;
        dp.start();
        run(x, n, w, r, true);
        peak = dp.stop();
    }
    libbench::write_volumes(libbench::option(argc, argv, "volumes"), r.volume);
    long long nan = 0;
    double sum = 0;
    for (double v : r.volume) (std::isnan(v) ? (void)++nan : (void)(sum += v));
    std::ostringstream o;
    o << ",\"K\":" << _K_ << ",\"width\":" << w << ",\"padded_points\":" << r.padded << ",\"failed_cells\":" << nan
      << ",\"failed_by_status\":{\"triangle_overflow\":" << r.failed[0] << ",\"vertex_overflow\":" << r.failed[1]
      << ",\"inconsistent_boundary\":" << r.failed[2] << ",\"security_radius_not_reached\":" << r.failed[3]
      << ",\"needs_exact_predicates\":" << r.failed[5] << ",\"empty_cell\":" << r.failed[6] << "}";
    std::cerr << "# vorogpu " << input << " " << n << " K " << _K_ << ": sum of volumes - 1 = " << sum - 1.0 << o.str()
              << std::endl;
    s.note = "failed cells are NaN (no CPU fallback in the public code); padding on " +
             std::to_string(gpu_images::threads()) + " OpenMP threads";
    while (libbench::now() - start < 2.0) {
        Result t;
        run(x, n, w, t, false);
    }
    for (int rep = 1; rep <= repeats; ++rep) {
        Result t;
        const double t0 = libbench::now();
        run(x, n, w, t, false);
        libbench::record(s, file, input, n, "build", rep, libbench::now() - t0, peak, o.str());
    }
    return 0;
}
