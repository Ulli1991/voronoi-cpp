#pragma once

// Periodic images for the GPU Delaunay/Voronoi libraries of the library benchmark that only know a bounding box
// (gDel3D, VoroGPU, Liu et al., Basselin et al.): the n points of the periodic unit box, then every periodic image
// x + s (s in {-1, 0, 1}^3, s != 0) that lies in the padded box [-w, 1 + w)^3. The cell of an original point is then
// the periodic cell whenever every Delaunay circumball incident to it lies inside the padded box (all images inside
// such a ball are in the set); gdel3d.cu checks this on its tetrahedra and prints the width it needed.
// Host code, OpenMP over points (count, scan, fill), part of every timed run. Used by gdel3d.cu, vorogpu.cu,
// voronoi_gpu_liu.cu, basselin.cpp.

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gpu_images {

// The padded set: xyz of the n originals first (same order), then the images; origin[i] = the original of point i.
struct Padded {
    std::uint64_t n = 0;     // originals
    double w = 0;            // shell width (unit box lengths)
    std::vector<double> xyz; // (n + images) * 3
    std::vector<std::uint32_t> source;  // the original of each image (index - n)
    std::uint64_t size() const { return xyz.size() / 3; }
};

// The shifts of one coordinate that keep its image inside [-w, 1 + w): 0, and +1 near 0 or -1 near 1 (w < 1/2).
inline int shifts(double x, double w, int* out) {
    int k = 0;
    out[k++] = 0;
    if (x < w) out[k++] = 1;
    if (x >= 1.0 - w) out[k++] = -1;
    return k;
}

inline void pad(const double* x, std::uint64_t n, double w, Padded& p) {
    p.n = n;
    p.w = w;
    std::vector<std::uint32_t> count(n + 1, 0);
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < std::int64_t(n); ++i) {
        int s[3][3], k[3];
        for (int a = 0; a < 3; ++a) k[a] = shifts(x[3 * i + a], w, s[a]);
        count[i + 1] = std::uint32_t(k[0] * k[1] * k[2] - 1);
    }
    for (std::uint64_t i = 0; i < n; ++i) count[i + 1] += count[i];  // inclusive scan: images before point i+1
    const std::uint64_t total = n + count[n];
    p.xyz.resize(total * 3);
    p.source.resize(total - n);
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < std::int64_t(n); ++i) {
        const double* q = x + 3 * i;
        p.xyz[3 * i] = q[0];
        p.xyz[3 * i + 1] = q[1];
        p.xyz[3 * i + 2] = q[2];
        int s[3][3], k[3];
        for (int a = 0; a < 3; ++a) k[a] = shifts(q[a], w, s[a]);
        std::uint64_t o = n + count[i];
        for (int a = 0; a < k[0]; ++a)
            for (int b = 0; b < k[1]; ++b)
                for (int c = 0; c < k[2]; ++c) {
                    if (a == 0 && b == 0 && c == 0) continue;
                    p.xyz[3 * o] = q[0] + s[0][a];
                    p.xyz[3 * o + 1] = q[1] + s[1][b];
                    p.xyz[3 * o + 2] = q[2] + s[2][c];
                    p.source[o - n] = std::uint32_t(i);
                    ++o;
                }
    }
}

// The padded set as float in [0, 1000]^3 (the box of the VoroGPU-derived codes): x -> (x + w) * scale, scale =
// 1000 / (1 + 2w); stride 3 (VoroGPU, Liu) or 4 (Basselin, w = 0). Volumes scale back by 1 / scale^3.
template <class Real>
inline double to_box1000(const Padded& p, std::vector<Real>& out, int stride) {
    const double scale = 1000.0 / (1.0 + 2.0 * p.w);
    const std::int64_t m = std::int64_t(p.size());
    out.resize(std::size_t(m) * stride);
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < m; ++i) {
        for (int a = 0; a < 3; ++a) out[stride * i + a] = Real((p.xyz[3 * i + a] + p.w) * scale);
        if (stride == 4) out[stride * i + 3] = Real(0);
    }
    return scale;
}

// The shell width for an input: --width=<w in mean spacings n^(-1/3)> if given, else the default of the input
// (measured with gdel3d.cu: the width every Delaunay circumball incident to an original point needs, times a margin;
// see default_width_spacings).
inline double default_width_spacings(const std::string& input) {
    if (input == "jittered" || input == "bluenoise") return 3.0;
    if (input == "uniform") return 5.0;
    if (input == "clustered") return 8.0;
    return 8.0;
}

inline int threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

}  // namespace gpu_images
