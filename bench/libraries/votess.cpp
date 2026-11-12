// votess (S. D. Singh, C. Byrohl, D. Nelson, arXiv:2412.04514) in the library benchmark: votess::tesellate()
// called from C++, on the CPU path (device::cpu, std::thread workers) or the SYCL device path (device::gpu).
//   votess <points file> [--device=cpu|gpu] [--threads=0] [--shell=4] [--k=64] [--grid=<cells per side>]
//          [--repeats=3] [--volumes=file]
// File formats and the record of each timed run: libbench.hpp.
//
// votess returns only the direct Voronoi neighbours of each point (class dnn, a CSR of neighbour indices): no volumes,
// faces or areas. It is not periodic and takes points strictly inside (0, 1)^3, reordering them in place (a grid
// sort). So, inside the clock: the n points plus their periodic images in a shell of width w, mapped affinely into
// (0, 1)^3, as a host std::vector<std::array<double, 3>> (votess's input type), then tesellate(). Variants:
//  - "dnn": the library's output, points in host memory to neighbour lists in host memory;
//  - "dnn + faces (ours)": plus the full cells of the n originals from those lists in double (clip_cells.hpp: a cube
//    cut by the bisector of each listed neighbour; volumes, centroids, faces with neighbours and areas; OpenMP over
//    cells); NaN volume for a cell left unbounded by its list. The shell is checked as in paragram.py: w >= 2 max R.
// use_recompute = true (the documented CPU fallback that "ensures all points are valid voronoi cells").

#include "clip_cells.hpp"
#include "libbench.hpp"

#ifdef __ADAPTIVECPP__
// votess includes <CL/sycl.hpp> and then names sycl::; DPC++'s CL/sycl.hpp declares ::sycl, AdaptiveCpp's only
// cl::sycl. AdaptiveCpp's <sycl/sycl.hpp> declares ::sycl (the library itself is unchanged).
#include <sycl/sycl.hpp>
#endif
#include <votess.hpp>

#include <omp.h>

#include <array>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace {

#ifdef VOTESS_DEVICE_PEAK
// Device peak memory as libbench::DevicePeak (cudaMemGetInfo, total minus free, sampled every 200 us). Not that class:
// libbench.hpp with LIBBENCH_CUDA includes cuda_runtime_api.h, whose dim3 clashes with AdaptiveCpp's pcuda dim3, so
// the three CUDA runtime calls are declared here (C linkage, cudaError_t returned as int) and libcudart is linked.
extern "C" int cudaMemGetInfo(std::size_t* free, std::size_t* total);
extern "C" int cudaDeviceSynchronize();
struct DevicePeak {
    std::atomic<bool> running{false};
    std::atomic<long long> peak{0};
    long long base = 0;
    std::thread sampler;
    static long long used() {
        std::size_t free = 0, total = 0;
        cudaMemGetInfo(&free, &total);
        return static_cast<long long>(total - free);
    }
    void start() {
        cudaDeviceSynchronize();
        base = used();
        peak = base;
        running = true;
        sampler = std::thread([this] {
            while (running) {
                const long long u = used();
                if (u > peak) peak = u;
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        });
    }
    double stop() {
        cudaDeviceSynchronize();
        const long long u = used();
        if (u > peak) peak = u;
        running = false;
        sampler.join();
        return double(peak - base);
    }
};
#endif


}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: votess <points file> [--device=cpu|gpu] [--threads=0] [--shell=4] [--k=64] [--grid=g]"
                     " [--repeats=3] [--volumes=file]\n";
        return 2;
    }
    const std::string file = argv[1];
    const bool gpu = libbench::option(argc, argv, "device", "cpu") == "gpu";
    const int threads = std::stoi(libbench::option(argc, argv, "threads", "0"));
    const double shell = std::stod(libbench::option(argc, argv, "shell", "4"));
    const int k = std::stoi(libbench::option(argc, argv, "k", "64"));
    const int repeats = std::stoi(libbench::option(argc, argv, "repeats", "3"));
    const auto frames = libbench::read_frames(file);
    const std::uint64_t n = frames.n;
    const double* x = frames.frame(0);
    const double w = shell * std::cbrt(1.0 / double(n)), scale = 1.0 / (1.0 + 2.0 * w);

    // The padded set in votess's input type, mapped into (0, 1)^3; index i < n is point i.
    std::vector<std::uint32_t> source;  // original of each padded point (built with the first call)
    auto padded = [&] {
        const bool record_source = source.empty();
        std::vector<std::array<double, 3>> p;
        p.reserve(static_cast<std::size_t>(n * (1.0 + 7.0 * w)));
        for (int sx = 0; sx <= 2; ++sx)
            for (int sy = 0; sy <= 2; ++sy)
                for (int sz = 0; sz <= 2; ++sz) {
                    const int s[3] = {sx == 2 ? -1 : sx, sy == 2 ? -1 : sy, sz == 2 ? -1 : sz};  // (0,0,0) first
                    for (std::uint64_t i = 0; i < n; ++i) {
                        std::array<double, 3> q;
                        bool in = true;
                        for (int a = 0; a < 3; ++a) {
                            const double y = x[3 * i + a] + s[a];
                            in &= y > -w && y < 1.0 + w;
                            q[a] = (y + w) * scale;
                        }
                        if (in) {
                            p.push_back(q);
                            if (record_source) source.push_back(std::uint32_t(i));
                        }
                    }
                }
        return p;
    };
    const std::size_t m = padded().size();
    const int grid = std::stoi(libbench::option(argc, argv, "grid", std::to_string(std::max(
                                                                        16, int(std::lround(std::cbrt(m / 8.0)))))));
    votess::vtargs args;
    args["k"] = k;
    args["knn_grid_resolution"] = grid;
    args["use_recompute"] = true;
    args["cpu_nthreads"] = threads;
    args["dev_suppress_stdout"] = true;
    if (gpu) args["gpu_ndsize"] = 16;
    // Both paths size their buffers with the index type (int): subsize * cc_p_maxsize (32) * 4 overflows at more than
    // 2^24 points (16.8M padded: SYCL buffer of non-positive size, assertion in AdaptiveCpp, job 32231100). The library's
    // own chunking (use_chunking, chunksize points per pass) avoids it; on by default only where needed.
    const long long chunk = std::stoll(libbench::option(argc, argv, "chunk",
                                                        m * 32LL * 4 > 2147483647LL ? "4194304" : "0"));
    if (chunk > 0) {
        args["use_chunking"] = true;
        args["chunksize"] = int(chunk);
    }
    const int nthreads = threads > 0 ? threads : int(std::thread::hardware_concurrency());

    std::vector<std::array<double, 3>> p;
    votess::dnn<int> dnn;
    std::vector<double> volume(n), rmax(n);
    auto tess = [&] {
        p = padded();
        dnn = votess::tesellate<int, double>(p, args, gpu ? votess::device::gpu : votess::device::cpu);
    };
    // Our cells. tesellate() reorders p; `sorted_of[i]` is the sorted index of input point i and `label[k]` the original
    // of sorted point k (both found outside the clock below, by coordinates: votess returns no permutation).
    std::vector<int> sorted_of(n);
    std::vector<std::uint32_t> label;
    std::vector<double> centroid(3 * n);
    std::vector<std::uint64_t> face_start(n + 1);
    std::vector<std::vector<std::int64_t>> part_neighbour(nthreads);
    std::vector<std::vector<double>> part_area(nthreads);
    std::vector<std::int64_t> neighbour;
    std::vector<double> area;
    auto volumes = [&] {
        const std::uint64_t chunk = (n + nthreads - 1) / nthreads;
#pragma omp parallel num_threads(nthreads)
        {
            const int me = omp_get_thread_num();
            auto& nbr = part_neighbour[me];
            auto& ar = part_area[me];
            nbr.clear(), ar.clear();
            const std::uint64_t lo = std::min<std::uint64_t>(n, me * chunk), hi = std::min<std::uint64_t>(n, lo + chunk);
            for (std::uint64_t i = lo; i < hi; ++i) {
                const int si = sorted_of[i];
                const int b = dnn.offs[si], m = dnn.offs[si + 1] - b;
                const double* seed = p[si].data();
                const auto r = clip::cell(
                    m,
                    [&](int j, double* d) {
                        const double* q = p[dnn.list[b + j]].data();
                        for (int a = 0; a < 3; ++a) d[a] = q[a] - seed[a];
                    },
                    [&](int j, const double* a) {
                        nbr.push_back(label[dnn.list[b + j]]);
                        ar.insert(ar.end(), {a[0] / (scale * scale), a[1] / (scale * scale), a[2] / (scale * scale)});
                    });
                volume[i] = r.ok ? r.volume / (scale * scale * scale) : NAN;
                rmax[i] = r.rmax / scale;
                for (int a = 0; a < 3; ++a) centroid[3 * i + a] = x[3 * i + a] + r.centroid[a] / scale;
                face_start[i + 1] = std::uint64_t(r.faces);
            }
        }
        for (std::uint64_t i = 0; i < n; ++i) face_start[i + 1] += face_start[i];
        neighbour.clear(), area.clear();
        for (int t = 0; t < nthreads; ++t) {
            neighbour.insert(neighbour.end(), part_neighbour[t].begin(), part_neighbour[t].end());
            area.insert(area.end(), part_area[t].begin(), part_area[t].end());
        }
    };

    // Warm-up, the index map, the volumes and the checks.
    tess();
    {
        struct H {
            std::size_t operator()(const std::array<double, 3>& a) const {
                std::size_t h = 0;
                for (double v : a) h = h * 1000003u ^ std::hash<double>()(v);
                return h;
            }
        };
        std::unordered_map<std::array<double, 3>, int, H> where;
        where.reserve(p.size());
        for (std::size_t i = 0; i < p.size(); ++i) where[p[i]] = int(i);
        const auto q = padded();
        for (std::uint64_t i = 0; i < n; ++i) sorted_of[i] = where.at(q[i]);
        label.assign(q.size(), 0);
        for (std::size_t k = 0; k < q.size(); ++k) label[where.at(q[k])] = source[k];
    }
    volumes();
    double rmax_all = 0, sum = 0, degree = 0;
    long nan = 0;
    for (std::uint64_t i = 0; i < n; ++i) {
        if (std::isnan(volume[i])) ++nan;
        else sum += volume[i], rmax_all = std::max(rmax_all, rmax[i]);
        degree += dnn.offs[sorted_of[i] + 1] - dnn.offs[sorted_of[i]];
    }
    const bool shell_ok = 2 * rmax_all <= w;
    std::cerr << "# " << file << ": n " << n << ", images " << m - n << ", w " << w << ", grid " << grid
              << ", NaN volumes " << nan << ", max R " << rmax_all << " (2R <= w: " << shell_ok << "), sum " << sum
              << ", mean neighbours " << degree / n << "\n";
    libbench::write_volumes(libbench::option(argc, argv, "volumes"), volume);

    libbench::Setup s;
    s.library = "votess";
    s.version = libbench::env("VOTESS_VERSION", "f0f805f (main 2025-07-15)");
    s.device = gpu ? "A100-SXM4-40GB" : "2x Xeon Platinum 8360Y";
    s.threads = gpu ? 1 : nthreads;
    std::ostringstream per;
    per << "images: shell of width " << w << " (" << shell << " mean spacings), " << m - n << " extra points; 2 max R <= w: "
        << (shell_ok ? "yes" : "NO");
    s.periodic = per.str();
    s.precision = "double (Tf = double)";
    std::ostringstream note;
    note << "k " << k << ", knn grid " << grid << "^3, use_recompute on";
    if (chunk > 0) note << ", use_chunking with chunksize " << chunk << " (int index overflow otherwise)";
    s.note = note.str();
    std::ostringstream extra;
    extra << ",\"images\":" << m - n << ",\"nan_volumes\":" << nan;
    struct Variant {
        std::string variant, outputs, timed;
        bool vol;
    };
    for (const Variant& v :
         {Variant{"tesellate (dnn)", "Voronoi neighbour lists",
                  "points in host memory to padded set and neighbour lists in host memory", false},
          Variant{"tesellate + faces from its neighbour lists (ours)",
                  "Voronoi neighbour lists; cells (ours): volumes centroids faces areas neighbours",
                  "points in host memory to padded set, neighbour lists and the full cells of the n originals in host "
                  "memory (our clip: clip_cells.hpp, OpenMP; the index maps of votess's reordering outside the clock)",
                  true}}) {
        s.variant = v.variant;
        s.outputs = v.outputs;
        s.timed = v.timed;
        if (gpu) s.timed += "; the SYCL device path copies to and from the device inside tesellate()";
        p = {};
        dnn = {};
        const double start = libbench::now();
        // Peak memory: host VmHWM on the CPU path; on the SYCL GPU path the device (cudaMemGetInfo sampled), the
        // host peak (padded copy, dnn) on stderr.
        libbench::HostPeak peak;
        peak.start();
#ifdef VOTESS_DEVICE_PEAK
        DevicePeak dpeak;
        if (gpu) dpeak.start();
#endif
        tess();
        if (v.vol) volumes();
        double bytes = peak.stop();
#ifdef VOTESS_DEVICE_PEAK
        if (gpu) {
            std::cerr << "# " << v.variant << ": host peak " << bytes << " B\n";
            bytes = dpeak.stop();
        }
#endif
        while (libbench::now() - start < 2.0) {  // warm-up as libbench::run_protocol
            tess();
            if (v.vol) volumes();
        }
        for (int r = 1; r <= repeats; ++r) {
            const double t0 = libbench::now();
            tess();
            if (v.vol) volumes();
            libbench::record(s, file, libbench::input_name(file), n, "build", r, libbench::now() - t0, bytes,
                             extra.str());
        }
    }
    return 0;
}
