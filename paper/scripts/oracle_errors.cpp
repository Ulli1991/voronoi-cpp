// Per-cell relative volume errors against the cached oracle volumes of bench/libraries/oracle_check.cpp: the same
// 1000-cell sample (same seed and generator), one CSV row per volumes file with all its errors (inf: no volume).
//   g++ -O2 -std=c++17 oracle_errors.cpp -o oracle_errors; oracle_errors <points file>[@frame] <volumes file>...

#include "../../bench/libraries/libbench.hpp"

#include <cstdio>
#include <random>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: oracle_errors <points file>[@frame] <volumes file>...\n";
        return 2;
    }
    const std::string points = argv[1];
    std::uint64_t n = 0;
    std::ifstream(points.substr(0, points.find('@')), std::ios::binary).read(reinterpret_cast<char*>(&n), 8);
    const int sample = 1000;
    std::vector<std::uint64_t> index;
    std::mt19937_64 rng(20261112);
    std::uniform_int_distribution<std::uint64_t> pick(0, n - 1);
    std::vector<char> taken(n, 0);
    while (static_cast<int>(index.size()) < sample) {
        const auto i = pick(rng);
        if (!taken[i]) taken[i] = 1, index.push_back(i);
    }
    std::vector<double> exact(sample);
    std::ifstream cache(points + ".oracle1000", std::ios::binary);
    if (!cache.read(reinterpret_cast<char*>(exact.data()), sample * 8)) throw std::runtime_error("no oracle cache");
    for (int f = 2; f < argc; ++f) {
        std::ifstream in(argv[f], std::ios::binary);
        std::vector<double> v(n);
        if (!in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 8))) continue;
        long long missing = 0;
        for (const double x : v) missing += !(x > 0) || !std::isfinite(x);
        std::printf("%s,%llu,%lld,", argv[f], static_cast<unsigned long long>(n), missing);
        for (int k = 0; k < sample; ++k) {
            const double x = v[index[k]];
            const double e = (x > 0 && std::isfinite(x)) ? std::abs(x - exact[k]) / exact[k] : INFINITY;
            std::printf(k ? " %.3e" : "%.3e", e);
        }
        std::printf("\n");
    }
    return 0;
}
