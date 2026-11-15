#pragma once

// Shared by every library driver of the library benchmark (docs/library_benchmark.md): the point and trajectory
// files, the clock, peak memory (host resident set, or device memory sampled from a second thread) and the result
// records, one JSON line per timed run. Host-only C++17; define LIBBENCH_CUDA (or compile with nvcc) for the device
// memory sampler.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#if defined(__CUDACC__) || defined(LIBBENCH_CUDA)
#include <cuda_runtime_api.h>
#define LIBBENCH_HAS_CUDA 1
#endif

namespace libbench {

// A point or trajectory file: uint64 n, uint64 frames, then frames x n x 3 doubles (x, y, z per point), little
// endian, positions in the periodic unit box [0, 1)^3 on the grid of 2^-30 that voronoi-cpp snaps to. A point file
// has one frame. Written by library_bench (src/bench/library_bench.cpp) from the generators of src/test/support.hpp.
struct Frames {
    std::uint64_t n = 0, frames = 0;
    std::vector<double> xyz;  // frames * n * 3
    const double* frame(std::uint64_t f) const { return xyz.data() + f * n * 3; }
};

inline Frames read_frames(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    Frames f;
    in.read(reinterpret_cast<char*>(&f.n), 8);
    in.read(reinterpret_cast<char*>(&f.frames), 8);
    f.xyz.resize(f.n * f.frames * 3);
    in.read(reinterpret_cast<char*>(f.xyz.data()), static_cast<std::streamsize>(f.xyz.size() * 8));
    if (!in) throw std::runtime_error("short file " + path);
    return f;
}

inline void write_frames(const std::string& path, std::uint64_t n, const std::vector<double>& xyz) {
    std::ofstream out(path, std::ios::binary);
    const std::uint64_t frames = xyz.size() / (3 * n);
    out.write(reinterpret_cast<const char*>(&n), 8);
    out.write(reinterpret_cast<const char*>(&frames), 8);
    out.write(reinterpret_cast<const char*>(xyz.data()), static_cast<std::streamsize>(xyz.size() * 8));
    if (!out) throw std::runtime_error("cannot write " + path);
}

// Cell volumes in input order, n doubles (NaN for a cell the library did not produce), for oracle_check.
inline void write_volumes(const std::string& path, const std::vector<double>& volume) {
    if (path.empty()) return;
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(volume.data()), static_cast<std::streamsize>(volume.size() * 8));
    if (!out) throw std::runtime_error("cannot write " + path);
}

// Wall clock (steady_clock), seconds.
inline double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Host memory: the resident set's high-water mark (VmHWM), which writing 5 to /proc/self/clear_refs resets to the
// current resident set (Linux 4.0+).
inline long long status_kb(const char* key) {
    std::ifstream in("/proc/self/status");
    std::string line;
    const std::string k(key);
    while (std::getline(in, line))
        if (line.rfind(k, 0) == 0) return std::atoll(line.c_str() + k.size() + 1);
    return -1;
}
struct HostPeak {
    long long base_kb = 0;
    void start() {
        std::ofstream("/proc/self/clear_refs") << "5";
        base_kb = status_kb("VmRSS:");
    }
    // Peak bytes above the resident set at start().
    double stop() const { return 1024.0 * double(status_kb("VmHWM:") - base_kb); }
};

#ifdef LIBBENCH_HAS_CUDA
// Device memory: total minus free (cudaMemGetInfo) sampled every 200 us on a second host thread, peak above the use at
// start(). Counts every allocation of the process (library pools, CUDA runtime, kernel stacks), not only arrays.
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
            int device = 0;
            cudaGetDevice(&device);
            cudaSetDevice(device);
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

// What one record says about the run besides its time. Strings are written as given (no escaping: keep them plain).
struct Setup {
    std::string library;   // "voro++", "CGAL Periodic_3_Delaunay_triangulation_3", ...
    std::string variant;   // what was called, e.g. "container_periodic, compute_cell" or "Delaunay + dual volumes"
    std::string version;   // release or commit
    std::string device;    // "A100-SXM4-40GB" or "2x Xeon Platinum 8360Y"
    int threads = 1;       // host threads used (GPU codes: 1)
    std::string outputs;   // "volumes", "volumes faces areas", "tetrahedra", ...
    std::string periodic;  // "native" or how it was emulated ("images: shell of width w" ...)
    std::string precision; // "double", "float", "double, exact predicates", ...
    std::string timed;     // what the clock covers, e.g. "points in host memory to volumes in host memory"
    std::string note;
};

inline std::string env(const char* key, const char* fallback = "") {
    const char* v = std::getenv(key);
    return v ? v : fallback;
}

// One JSON line on stdout: setup, input, phase ("build" or "step"), repeat number, seconds and peak bytes.
inline void record(const Setup& s, const std::string& points_file, const std::string& input, std::uint64_t n,
                   const std::string& phase, int repeat, double seconds, double peak_bytes,
                   const std::string& extra = "") {
    char host[256] = {0};
    gethostname(host, sizeof host - 1);
    std::ostringstream o;
    o.precision(9);
    o << "{\"library\":\"" << s.library << "\",\"variant\":\"" << s.variant << "\",\"version\":\"" << s.version
      << "\",\"device\":\"" << s.device << "\",\"threads\":" << s.threads << ",\"outputs\":\"" << s.outputs
      << "\",\"periodic\":\"" << s.periodic << "\",\"precision\":\"" << s.precision << "\",\"timed\":\"" << s.timed
      << "\",\"note\":\"" << s.note << "\",\"points\":\"" << points_file << "\",\"input\":\"" << input
      << "\",\"n\":" << n << ",\"phase\":\"" << phase << "\",\"repeat\":" << repeat << ",\"seconds\":" << seconds
      << ",\"Mcells_s\":" << n / seconds / 1e6 << ",\"peak_bytes\":" << peak_bytes
      << ",\"bytes_per_cell\":" << peak_bytes / double(n) << ",\"job\":\"" << env("SLURM_JOB_ID") << "\",\"host\":\""
      << host << "\"" << extra << "}";
    std::cout << o.str() << std::endl;
}

// The input's name from a point file's name: ".../jittered_2097152.pts" -> "jittered".
inline std::string input_name(const std::string& path) {
    std::string base = path.substr(path.find_last_of('/') + 1);
    return base.substr(0, base.find('_'));
}

// Command-line helpers: positional arguments, --key=value and --flag.
inline std::string option(int argc, char** argv, const std::string& key, const std::string& fallback = "") {
    const std::string prefix = "--" + key + "=";
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]).rfind(prefix, 0) == 0) return argv[i] + prefix.size();
    return fallback;
}
inline bool flag(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--" + key) return true;
    return false;
}

// The protocol of every driver: one untimed run that measures peak memory (memory_run, which returns peak bytes),
// untimed warm-up runs until 2 s have passed since its start (GPU clocks ramp up, caches and pools fill; a run of
// tens of ms would otherwise be timed cold), then `repeats` timed runs (timed_run returns seconds), each recorded.
// `n` cells.
inline void run_protocol(const Setup& s, const std::string& points_file, std::uint64_t n, const std::string& phase,
                         int repeats, const std::function<double()>& memory_run,
                         const std::function<double()>& timed_run) {
    const double start = now();
    const double peak = memory_run();
    while (now() - start < 2.0) timed_run();
    for (int r = 1; r <= repeats; ++r) record(s, points_file, input_name(points_file), n, phase, r, timed_run(), peak);
}

}  // namespace libbench
