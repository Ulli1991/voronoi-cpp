// Basselin, Alonso, Ray, Sokolov, Lefebvre, Levy, "Restricted Power Diagrams on the GPU", Computer Graphics Forum
// 40(2) (Eurographics 2021), in the library benchmark on one A100 through OpenCL (NVIDIA's OpenCL 3.0 driver).
// VoroAlgo (voronoi.h): setInputPoints builds the kNN grid on the device, launch computes the convex cells by
// clipping with the K nearest neighbours until the security radius is reached, then relaunches the failed cells
// (security radius not reached, plane/vertex/triangle overflow) with K, P, V, T times 1.5, up to 5 times; built with
// USE_DOUBLE (the code's double mode). Without a grid (volume mesh) the cells are those of the box [-0.1, 1000.1]^3.
// The call sequence is that of the code's test_voronoi.cpp (computeVoroDiagram): Options -> initVoroAlgoParameters,
// m_keepDistance = keepDistanceLast, setInputPoints(host xyzw), launch(saveData: getBarycenters,
// getPermutationToUser). Not periodic: images in a shell (gpu_images.hpp) scaled to [0, 1000]^3.
//   basselin <points file> [--repeats=3] [--width=<spacings>] [--volumes=file] [--K=70 --P=30 --V=40 ...]
// --K/--P/--V/... go to the code's own Options (defaults: the README's white-noise settings K 70, P 30, V 40).

#include "libbench.hpp"
#include "gpu_images.hpp"

#include "cvoro_config.h"  // defines `real` (double with USE_DOUBLE); Options.h uses it without including it
#include "openCL.h"
#include "voronoi.h"
#include "Options.h"

namespace {

struct Result {
    std::vector<double> volume;
    std::uint64_t padded = 0;
    std::string stats;
};

void run(const std::shared_ptr<OpenCLContext>& context, const double* x, std::uint64_t n, double w, Result& r,
         bool keep_stats) {
    gpu_images::Padded p;
    gpu_images::pad(x, n, w, p);
    std::vector<real> seeds;
    const double scale = gpu_images::to_box1000(p, seeds, 4);
    r.padded = p.size();
    const Options& options = getOptions();
    VoroAlgo algo(context, options.m_debug);
    options.initVoroAlgoParameters(algo);
    algo.m_keepDistance = options.m_keepDistanceLast;
    algo.setInputPoints(seeds);
    std::vector<real> bary;
    std::vector<int> permutation;
    std::vector<VoroAlgoStat> stats;
    const bool ok = algo.launch(
        [&](VoroAlgo const& a) {
            bary = a.getBarycenters();
            permutation = a.getPermutationToUser();
            return true;
        },
        stats);
    if (!ok) throw std::runtime_error("VoroAlgo::launch failed");
    const double inv = 1.0 / (scale * scale * scale);
    r.volume.assign(n, std::nan(""));
    const std::int64_t m = std::int64_t(bary.size() / 4);
    const bool identity = permutation.empty();
#pragma omp parallel for schedule(static)
    for (std::int64_t pos = 0; pos < m; ++pos) {
        const std::int64_t i = identity ? pos : permutation[pos];
        const double v = double(bary[4 * pos + 3]);
        if (i < std::int64_t(n) && v > 0) r.volume[i] = v * inv;  // w <= 0: no cell (empty / NO_CELL_LIMITS)
    }
    if (keep_stats) {
        std::ostringstream o;
        o << "[";
        for (std::size_t i = 0; i < stats.size(); ++i) {
            stats[i].pythonPrint(o);
            if (i + 1 < stats.size()) o << ",";
        }
        o << "]";
        r.stats = o.str();
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: basselin <points file> [--repeats=3] [--width=spacings] [--volumes=file] [--K= ...]\n";
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

    // The code's options: README white-noise defaults, then any --K= --P= --V= --T= ... of the command line.
    std::vector<std::string> args = {"basselin", "--K=70", "--P=30", "--V=40"};
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--repeats=", 0) && a.rfind("--width=", 0) && a.rfind("--volumes=", 0)) args.push_back(a);
    }
    std::vector<const char*> cargs;
    for (auto& a : args) cargs.push_back(a.c_str());
    Options& options = getOptions();
    if (options.parseOptions(int(cargs.size()), cargs.data()) != int(cargs.size())) {
        std::cerr << "basselin: bad option\n";
        return 1;
    }
    std::ostringstream params;
    for (std::size_t i = 1; i < args.size(); ++i) params << (i > 1 ? " " : "") << args[i];

#ifdef LIBBENCH_HAS_CUDA
    cudaSetDevice(0);
    cudaFree(nullptr);  // CUDA context of the memory sampler, before the clock
#endif
    auto context = std::make_shared<OpenCLContext>(false);
    if (!context->createContext(true) || !context->hasContext()) {
        std::cerr << "basselin: no OpenCL GPU context\n";
        return 1;
    }

    libbench::Setup s;
    s.library = "Basselin et al. GPU-Restricted-Power-Diagrams";
    s.variant = "VoroAlgo (OpenCL), no grid, " + params.str();
    s.version = libbench::env("LIBBENCH_LIB_VERSION", "196e7d7 + basselin.patch");
    s.device = "A100-SXM4-40GB";
    s.threads = 1;
    s.outputs = "volumes (and barycentres)";
    std::ostringstream per;
    per << "images: shell of width " << ws << " mean spacings (" << w << ") around the unit box, in the box "
        << "[-0.1, 1000.1]^3 of the code";
    s.periodic = per.str();
    s.precision = std::string(sizeof(real) == 8 ? "double" : "float") +
                  " (points scaled to [0, 1000]^3), certified by the security radius, failed cells relaunched with "
                  "larger K/P/V/T; no exact predicates";
    s.timed = "points in host memory (double) to volumes in host memory: padding, conversion, setInputPoints (H2D, "
              "kNN grid), launch (cells and relaunches), getBarycenters/getPermutationToUser (D2H), volumes by the "
              "permutation";

    Result r;
    double peak = 0;
    const double start = libbench::now();  // as run_protocol: warm-up runs until 2 s after this
    {
        run(context, x, n, w, r, false);  // warm-up: builds the OpenCL programs of every pass (cached in the context)
#ifdef LIBBENCH_HAS_CUDA
        libbench::DevicePeak dp;
        dp.start();
        run(context, x, n, w, r, true);
        peak = dp.stop();
#else
        run(context, x, n, w, r, true);
        peak = -1;
#endif
    }
    libbench::write_volumes(libbench::option(argc, argv, "volumes"), r.volume);
    long long nan = 0;
    double sum = 0;
    for (double v : r.volume) (std::isnan(v) ? (void)++nan : (void)(sum += v));
    std::ostringstream o;
    o << ",\"width\":" << w << ",\"padded_points\":" << r.padded << ",\"failed_cells\":" << nan
      << ",\"passes\":" << r.stats;
    std::cerr << "# basselin " << input << " " << n << ": sum of volumes - 1 = " << sum - 1.0 << o.str() << std::endl;
    s.note = "failed cells (none expected after the relaunches) NaN; OpenCL programs built in a warm-up run outside "
             "the clock; device peak = cudaMemGetInfo (device-wide) sampled; padding on " +
             std::to_string(gpu_images::threads()) + " OpenMP threads";
    while (libbench::now() - start < 2.0) {
        Result t;
        run(context, x, n, w, t, false);
    }
    for (int rep = 1; rep <= repeats; ++rep) {
        Result t;
        const double t0 = libbench::now();
        run(context, x, n, w, t, false);
        libbench::record(s, file, input, n, "build", rep, libbench::now() - t0, peak, o.str());
    }
    return 0;
}
