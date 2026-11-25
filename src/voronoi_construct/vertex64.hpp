#pragma once

// Diagnostic build only (-DVORONOI_VERTEX64, off by default): issue #64, the rounded clipping's new vertex on a cut
// edge. Each new vertex of a rounded cut is classified by its two ends' signed distances to the cutting plane: global
// counters, flags and the worst vertex per cell. Bound and results: research_notes/vertex64_2026-11-23.md.
#if defined(VORONOI_VERTEX64)

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdint>

namespace vertex64 {

// Bits of a cut (and, or-ed, of a cell).
enum Bit : unsigned {
    strict = 1,        // |side[w]| <= B_w and |side[u]| <= B_u: both ends within the rounding bound of the plane
    kept_outside = 2,  // side[u] > 0: the kept end lies outside, t > 1, the vertex is placed beyond u
    beyond_bound = 4,  // side[u] > B_u: the kept end lies outside for certain
    err9 = 8,          // first-order bound of the vertex's displacement along the edge > 1e-9 R
    err6 = 16,         // ... > 1e-6 R
    err3 = 32,         // ... > 1e-3 R
    built_exactly = 128,  // the cell's last build was exact (written by exact passes)
};
// Global counters.
enum Counter : int { new_vertices, c_strict, c_kept_outside, c_beyond_bound, c_err9, c_err6, c_err3, counters };

// Where the kernels write; null pointers write nothing. Per-cell arrays are indexed by particle id, below cap.
struct Sinks {
    unsigned long long* counter = nullptr;  // [counters]
    int cap = 0;
    unsigned* flag = nullptr;      // or of the cut bits of the cell's last rounded build
    float* worst = nullptr;        // largest displacement bound / R of its cuts
    double* radius = nullptr;      // final radius of the last rounded build (box units)
    double* exact_radius = nullptr;  // final radius of the last exact build (box units)
    int* worst_slot = nullptr;     // [4 cap]: generator slot, the slots of planes a, b and the cutting plane
                                   //   (a box plane k as -1 - k)
    double* worst_vertex = nullptr;  // [3 cap]: the vertex placed by that cut, relative to the generator
};

#if defined(__CUDACC__)
static __device__ Sinks device_sinks;
#endif
// The sinks host kernels write to.
inline Sinks& host_sinks() {
    static Sinks s;
    return s;
}

// Installs the sinks for device kernels (CUDA) and host kernels alike.
inline void install(const Sinks& s, bool on_device) {
    if (on_device) {
#if defined(__CUDACC__)
        cudaMemcpyToSymbol(device_sinks, &s, sizeof(Sinks));
#endif
    } else {
        host_sinks() = s;
    }
}

// The installed sinks of the side the caller runs on (device or host).
KOKKOS_INLINE_FUNCTION const Sinks& sinks() {
    KOKKOS_IF_ON_DEVICE((return device_sinks;))
    KOKKOS_IF_ON_HOST((return host_sinks();))
}

// The bound on the rounding of side = d.x - |d|^2 / 2 for the stored d and x (research_notes/vertex64_2026-11-23.md):
// gamma_8 (S + o) with
// S = sum |d_k x_k|, o = |d|^2 / 2, taken as 9 u (S + o), u = 2^-53, so that rounding the bound itself is covered.
KOKKOS_INLINE_FUNCTION double side_bound(double dx, double dy, double dz, double x, double y, double z, double offset) {
    return 9 * 0x1p-53 * (Kokkos::abs(dx * x) + Kokkos::abs(dy * y) + Kokkos::abs(dz * z) + Kokkos::abs(offset));
}

// Bits of one new vertex on the edge from removed w (side sw, bound bw) to kept u (su, bu), the edge `edge` long, in a
// cell of radius r; err: the first-order bound of the vertex's displacement along the edge, over r.
KOKKOS_INLINE_FUNCTION unsigned classify(double sw, double su, double bw, double bu, double edge, double r,
                                         float& err) {
    unsigned bits = 0;
    if (Kokkos::abs(sw) <= bw && Kokkos::abs(su) <= bu) bits |= strict;
    if (su > 0) bits |= kept_outside;
    if (su > bu) bits |= beyond_bound;
    const double delta = sw - su;
    // t = sw / delta: dt/dsw = -su / delta^2, dt/dsu = sw / delta^2.
    const double e = delta > 0 ? edge * (Kokkos::abs(su) * bw + Kokkos::abs(sw) * bu) / (delta * delta) / r
                               : static_cast<double>(INFINITY);
    err = static_cast<float>(Kokkos::fmin(e, 3.0e38));
    if (e > 1e-9) bits |= err9;
    if (e > 1e-6) bits |= err6;
    if (e > 1e-3) bits |= err3;
    return bits;
}

// The counter of each bit, in Counter order after new_vertices.
KOKKOS_INLINE_FUNCTION unsigned counted_bit(int k) {
    return k == c_strict ? strict : k == c_kept_outside ? kept_outside : k == c_beyond_bound ? beyond_bound
                                    : k == c_err9       ? err9
                                    : k == c_err6       ? err6
                                                        : err3;
}

// The per-cell record of a rounded build: bits, counts of its cuts, the worst cut and the final radius.
struct Record {
    unsigned flag = 0;
    float worst = -1;
    int slot[4] = {0, 0, 0, 0};
    double vertex[3] = {0, 0, 0};
    unsigned count[counters] = {};
};

// Adds a cut's bits to the cell's counts (one call per new vertex); write() adds them to the global counters, one
// atomic per counter and cell (one per vertex contended too much on the A100).
KOKKOS_INLINE_FUNCTION void count(Record& r, unsigned bits) {
    ++r.count[new_vertices];
    for (int k = 1; k < counters; ++k) r.count[k] += (bits & counted_bit(k)) != 0;
}

// Adds a rounded build's counts to the global counters and stores its flag, worst cut and radius for cell id.
KOKKOS_INLINE_FUNCTION void write(int id, const Record& r, double radius) {
    const Sinks& s = sinks();
    if (s.counter != nullptr)
        for (int k = 0; k < counters; ++k)
            if (r.count[k] != 0) {
                KOKKOS_IF_ON_DEVICE((atomicAdd(&s.counter[k], static_cast<unsigned long long>(r.count[k]));))
                KOKKOS_IF_ON_HOST((Kokkos::atomic_add(&s.counter[k], static_cast<unsigned long long>(r.count[k]));))
            }
    if (id >= s.cap || s.flag == nullptr) return;
    s.flag[id] = r.flag;
    s.worst[id] = r.worst;
    s.radius[id] = radius;
    for (int k = 0; k < 4; ++k) s.worst_slot[4 * id + k] = r.slot[k];
    for (int k = 0; k < 3; ++k) s.worst_vertex[3 * id + k] = r.vertex[k];
}

// Marks cell id as built exactly and stores that build's radius.
KOKKOS_INLINE_FUNCTION void write_exact(int id, double radius) {
    const Sinks& s = sinks();
    if (id >= s.cap || s.flag == nullptr) return;
    s.flag[id] |= built_exactly;
    s.exact_radius[id] = radius;
}

}  // namespace vertex64

#endif  // VORONOI_VERTEX64
