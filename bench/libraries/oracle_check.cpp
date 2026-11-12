// Accuracy of each library's cell volumes against the brute-force oracle of src/test/oracle.hpp (exact 128-bit
// predicates, long double clipping over all periodic images; it shares no code with voronoi-cpp), on a random sample
// of cells, and on all cells the checks that need no oracle (missing or non-positive volumes, the sum against the box).
//   oracle_check <points file>[@frame] <sample size> [volumes file...]   (frame: of a trajectory file, default 0)
// The sample's oracle volumes are cached next to the point file (<points>.oracle<sample size>). One JSON line per
// volumes file. Brute force: O(n log n) per sampled cell, run on a full CPU node (OpenMP).
// Build: g++ -O2 -std=c++17 -fopenmp -I../../src/test oracle_check.cpp -o oracle_check

#include "libbench.hpp"
#include "oracle.hpp"

#include <cstdio>
#include <random>

namespace {

std::vector<double> read_doubles(const std::string& path, std::uint64_t n) {
    std::vector<double> v(n);
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 8));
    if (!in) throw std::runtime_error("short or missing volumes file " + path);
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: oracle_check <points file>[@frame] <sample size> [volumes file...]\n";
        return 2;
    }
    const std::string points = argv[1];
    const auto at = points.find('@');
    const std::uint64_t frame = at == std::string::npos ? 0 : std::stoull(points.substr(at + 1));
    const int sample = std::stoi(argv[2]);
    const auto frames = libbench::read_frames(points.substr(0, at));
    const std::uint64_t n = frames.n;
    constexpr double scale = 1073741824.0;  // 2^30, the grid of the point files
    oracle::Grid g;
    g.period = {1LL << 30, 1LL << 30, 1LL << 30};
    g.step = 1.0L / scale;
    g.p.resize(n);
    for (std::uint64_t i = 0; i < n; ++i)
        for (int a = 0; a < 3; ++a) {
            const double x = frames.frame(frame)[3 * i + a] * scale;
            if (x != std::floor(x)) throw std::runtime_error("point off the 2^-30 grid");
            g.p[i][a] = static_cast<oracle::i64>(x);
        }
    // The sample: distinct cells drawn with a fixed seed.
    std::vector<int> index;
    {
        std::mt19937_64 rng(20261112);
        std::uniform_int_distribution<std::uint64_t> pick(0, n - 1);
        std::vector<char> taken(n, 0);
        while (static_cast<int>(index.size()) < sample) {
            const auto i = pick(rng);
            if (!taken[i]) taken[i] = 1, index.push_back(static_cast<int>(i));
        }
    }
    const std::string cache = points + ".oracle" + std::to_string(sample);
    std::vector<double> exact(sample);
    std::ifstream cached(cache, std::ios::binary);
    if (cached) {
        cached.read(reinterpret_cast<char*>(exact.data()), sample * 8);
    } else {
        const double t0 = libbench::now();
        const long double cell = std::pow(1.0L / scale, 3);
#pragma omp parallel for schedule(dynamic, 1)
        for (int k = 0; k < sample; ++k) exact[k] = static_cast<double>(oracle::voronoi_cell(g, index[k]).volume * cell);
        std::ofstream(cache, std::ios::binary)
            .write(reinterpret_cast<const char*>(exact.data()), static_cast<std::streamsize>(sample * 8));
        std::cerr << "oracle: " << sample << " cells in " << libbench::now() - t0 << " s\n";
    }
    for (int f = 3; f < argc; ++f) {
        const auto v = read_doubles(argv[f], n);
        long long missing = 0;
        long double sum = 0;
        for (const double x : v) {
            if (!(x > 0) || !std::isfinite(x)) ++missing;
            else sum += x;
        }
        std::vector<double> err(sample);
        int above12 = 0, above6 = 0, above2 = 0;
        for (int k = 0; k < sample; ++k) {
            const double x = v[index[k]];
            err[k] = std::isfinite(x) ? std::abs(x - exact[k]) / exact[k] : INFINITY;
            above12 += err[k] > 1e-12;
            above6 += err[k] > 1e-6;
            above2 += err[k] > 1e-2;
        }
        std::vector<double> sorted = err;
        std::sort(sorted.begin(), sorted.end());
        // A missing sampled cell makes the error infinite: written as null (JSON has no inf), counted in the columns.
        auto number = [](double x) {
            char b[32];
            if (!std::isfinite(x)) return std::string("null");
            std::snprintf(b, sizeof b, "%.3e", x);
            return std::string(b);
        };
        std::printf(
            "{\"volumes\":\"%s\",\"points\":\"%s\",\"n\":%llu,\"sample\":%d,\"missing_or_nonpositive\":%lld,"
            "\"sum_minus_box\":%s,\"median_rel_err\":%s,\"max_rel_err\":%s,\"cells_rel_err_above_1e-12\":%d,"
            "\"above_1e-6\":%d,\"above_1e-2\":%d}\n",
            argv[f], points.c_str(), static_cast<unsigned long long>(n), sample, missing,
            number(double(sum - 1.0L)).c_str(), number(sorted[sample / 2]).c_str(), number(sorted.back()).c_str(),
            above12, above6, above2);
    }
    return 0;
}
