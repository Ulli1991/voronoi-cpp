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
//  - "dnn + volumes (ours)": plus the volumes of the n original cells from those lists in double (our clip: a cube
//    cut by the bisector of each listed neighbour, OpenMP over cells), the volumes file's source; NaN for a cell
//    left unbounded by its list. The shell is checked as in paragram.py: w >= 2 max R.
// use_recompute = true (the documented CPU fallback that "ensures all points are valid voronoi cells").

#include "libbench.hpp"

#ifdef __ADAPTIVECPP__
// votess includes <CL/sycl.hpp> and then names sycl::; DPC++'s CL/sycl.hpp declares ::sycl, AdaptiveCpp's only
// cl::sycl. AdaptiveCpp's <sycl/sycl.hpp> declares ::sycl (the library itself is unchanged).
#include <sycl/sycl.hpp>
#endif
#include <votess.hpp>

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

constexpr int MAXP = 128, MAXT = 256;

void vertex3(const double (*n)[3], const double* nn, int a, int b, int c, double* v) {
    const double *A = n[a], *B = n[b], *C = n[c];
    const double bc[3] = {B[1] * C[2] - B[2] * C[1], B[2] * C[0] - B[0] * C[2], B[0] * C[1] - B[1] * C[0]};
    const double ca[3] = {C[1] * A[2] - C[2] * A[1], C[2] * A[0] - C[0] * A[2], C[0] * A[1] - C[1] * A[0]};
    const double ab[3] = {A[1] * B[2] - A[2] * B[1], A[2] * B[0] - A[0] * B[2], A[0] * B[1] - A[1] * B[0]};
    const double det = A[0] * bc[0] + A[1] * bc[1] + A[2] * bc[2];
    for (int k = 0; k < 3; ++k) v[k] = (nn[a] * bc[k] + nn[b] * ca[k] + nn[c] * ab[k]) / det;
}

int across(const unsigned char* ta, const unsigned char* tb, const unsigned char* tc, int nt, int y, int x) {
    for (int t = 0; t < nt; ++t)
        if ((ta[t] == y && tb[t] == x) || (tb[t] == y && tc[t] == x) || (tc[t] == y && ta[t] == x)) return t;
    return -1;
}

// Volume of the cell of seed s cut by the bisectors with the points nb[0..m) (absolute positions), and its largest
// vertex distance; NaN if the list leaves it unbounded or overflows the caps. Same algorithm as paragram.py's kernel.
double cell_volume(const double* s, const std::vector<const double*>& nb, double& rmax) {
    double pn[MAXP][3], nn[MAXP];
    unsigned char ta[MAXT], tb[MAXT], tc[MAXT];
    double vx[MAXT], vy[MAXT], vz[MAXT];
    const double R0 = 0.5;
    for (int k = 0; k < 6; ++k) {
        pn[k][0] = pn[k][1] = pn[k][2] = 0;
        pn[k][k / 2] = (k % 2 == 0) ? R0 : -R0;
        nn[k] = R0 * R0;
    }
    int np = 6, nt = 0;
    for (int c = 0; c < 8; ++c) {
        const int sx = c & 1, sy = (c >> 1) & 1, sz = (c >> 2) & 1;
        const bool even = ((sx + sy + sz) % 2) == 0;
        ta[nt] = sx;
        tb[nt] = even ? 2 + sy : 4 + sz;
        tc[nt] = even ? 4 + sz : 2 + sy;
        double v[3];
        vertex3(pn, nn, ta[nt], tb[nt], tc[nt], v);
        vx[nt] = v[0], vy[nt] = v[1], vz[nt] = v[2];
        ++nt;
    }
    bool bad = false;
    for (const double* q : nb) {
        if (np >= MAXP) {
            bad = true;
            break;
        }
        const double fx = 0.5 * (q[0] - s[0]), fy = 0.5 * (q[1] - s[1]), fz = 0.5 * (q[2] - s[2]);
        const double f2 = fx * fx + fy * fy + fz * fz;
        if (f2 == 0) continue;
        bool kill[MAXT], any = false;
        for (int t = 0; t < nt; ++t) any |= (kill[t] = fx * vx[t] + fy * vy[t] + fz * vz[t] > f2);
        if (!any) continue;
        pn[np][0] = fx, pn[np][1] = fy, pn[np][2] = fz, nn[np] = f2;
        unsigned char na[MAXT], nb2[MAXT];
        int nnew = 0;
        for (int t = 0; t < nt && !bad; ++t) {
            if (!kill[t]) continue;
            const int vtx[3] = {ta[t], tb[t], tc[t]};
            for (int k = 0; k < 3 && !bad; ++k) {
                const int x = vtx[k], y = vtx[(k + 1) % 3];
                const int o = across(ta, tb, tc, nt, y, x);
                if (o >= 0 && !kill[o]) {
                    if (nnew >= MAXT) bad = true;
                    else na[nnew] = x, nb2[nnew] = y, ++nnew;
                }
            }
        }
        int w = 0;
        for (int t = 0; t < nt; ++t)
            if (!kill[t]) ta[w] = ta[t], tb[w] = tb[t], tc[w] = tc[t], vx[w] = vx[t], vy[w] = vy[t], vz[w] = vz[t], ++w;
        if (bad || w + nnew > MAXT) {
            bad = true;
            break;
        }
        nt = w;
        for (int k = 0; k < nnew; ++k) {
            ta[nt] = na[k], tb[nt] = nb2[k], tc[nt] = static_cast<unsigned char>(np);
            double v[3];
            vertex3(pn, nn, na[k], nb2[k], np, v);
            vx[nt] = v[0], vy[nt] = v[1], vz[nt] = v[2];
            ++nt;
        }
        ++np;
    }
    double V = 0, r2 = 0;
    for (int t = 0; t < nt && !bad; ++t) {
        if (ta[t] < 6 || tb[t] < 6 || tc[t] < 6) bad = true;
        r2 = std::max(r2, vx[t] * vx[t] + vy[t] * vy[t] + vz[t] * vz[t]);
        const int vtx[3] = {ta[t], tb[t], tc[t]};
        for (int k = 0; k < 3; ++k) {
            const int x = vtx[k], y = vtx[(k + 1) % 3];
            const int o = across(ta, tb, tc, nt, y, x);
            if (o < 0) {
                bad = true;
                break;
            }
            const double cx = vy[t] * vz[o] - vz[t] * vy[o], cy = vz[t] * vx[o] - vx[t] * vz[o],
                         cz = vx[t] * vy[o] - vy[t] * vx[o];
            V += cx * pn[y][0] + cy * pn[y][1] + cz * pn[y][2];
        }
    }
    rmax = std::sqrt(r2);
    return bad ? NAN : std::fabs(V) / 6.0;
}

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
    auto padded = [&] {
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
                        if (in) p.push_back(q);
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
    // Our volumes. tesellate() reorders p; `sorted_of[i]` is the sorted index of input point i (found outside the
    // clock below, by coordinates).
    std::vector<int> sorted_of(n);
    auto volumes = [&] {
#pragma omp parallel for schedule(dynamic, 1024) num_threads(nthreads)
        for (std::int64_t i = 0; i < std::int64_t(n); ++i) {
            const int si = sorted_of[i];
            std::vector<const double*> nb;
            for (int e = dnn.offs[si]; e < dnn.offs[si + 1]; ++e) nb.push_back(p[dnn.list[e]].data());
            volume[i] = cell_volume(p[si].data(), nb, rmax[i]);
        }
        for (std::uint64_t i = 0; i < n; ++i) volume[i] /= scale * scale * scale, rmax[i] /= scale;
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
          Variant{"tesellate + volumes from its neighbour lists (ours)", "Voronoi neighbour lists, volumes (ours)",
                  "points in host memory to padded set, neighbour lists and volumes of the n cells in host memory",
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
