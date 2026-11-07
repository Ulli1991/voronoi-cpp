#pragma once

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <type_traits>
#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_fp16.h>
#endif

// Compile-time precision policy of FiniteVolume (all double by default); fed to it as a template argument.
// Settings: Geometry (face rows), Weight (stored least-squares weights, or void), Gradient (gradients and
// face-state change), Flux (stored face fluxes), Riemann (the Riemann solve).
namespace hydro {

// FP16 storage of float gradients (a Gradient setting).
// Used by: Precision, FiniteVolume's gradient store and face kernel; hydro_sedov picks it.
struct Half {};

// The precision choices of one run; each alias below is read by FiniteVolume (geometry rows from Mesh,
// gradients in the sweep, fluxes and Riemann solve in the update).
template <class Geometry = double, class Weight = void, class Gradient = double, class Flux = double,
          class Riemann = double>
struct Precision {
    static_assert(std::is_same_v<Geometry, double> || std::is_same_v<Geometry, float>, "geometry: double or float");
    static_assert(std::is_void_v<Weight> || std::is_same_v<Weight, double> || std::is_same_v<Weight, float>,
                  "weight: void, double or float");
    using weight = Weight;
    static_assert(std::is_same_v<Gradient, double> || std::is_same_v<Gradient, float> || std::is_same_v<Gradient, Half>,
                  "gradient: double, float or Half");
    static_assert(std::is_same_v<Flux, double> || std::is_same_v<Flux, float>, "flux: double or float");
    static_assert(std::is_same_v<Riemann, double> || std::is_same_v<Riemann, float>, "riemann: double or float");
    using geometry = Geometry;
    using gradient = std::conditional_t<std::is_same_v<Gradient, Half>, float, Gradient>;  // arithmetic
    using gradient_store = std::conditional_t<std::is_same_v<Gradient, Half>, std::uint16_t, gradient>;
    static constexpr bool half = std::is_same_v<Gradient, Half>;
    using flux = Flux;
    using riemann = Riemann;
};

namespace half {

// float -> IEEE binary16 bits, round to nearest even; and back.
// Used by: FiniteVolume's gradient store (from_float) and face kernel (to_float).
KOKKOS_INLINE_FUNCTION std::uint16_t from_float(float f) {
#if defined(__CUDA_ARCH__)
    return __half_as_ushort(__float2half_rn(f));
#else
    const std::uint32_t x = Kokkos::bit_cast<std::uint32_t>(f);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    const int exponent = static_cast<int>((x >> 23) & 0xff) - 127 + 15;
    std::uint32_t mantissa = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) return static_cast<std::uint16_t>(sign | 0x7c00u | (mantissa ? 0x200u : 0u));
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<std::uint16_t>(sign);
        mantissa |= 0x800000u;
        const int shift = 14 - exponent;
        std::uint32_t h = mantissa >> shift;
        const std::uint32_t rest = mantissa & ((1u << shift) - 1), halfway = 1u << (shift - 1);
        if (rest > halfway || (rest == halfway && (h & 1))) ++h;
        return static_cast<std::uint16_t>(sign | h);
    }
    std::uint32_t h = static_cast<std::uint32_t>(exponent) << 10 | mantissa >> 13;
    const std::uint32_t rest = mantissa & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1))) ++h;
    return static_cast<std::uint16_t>(sign | h);
#endif
}

KOKKOS_INLINE_FUNCTION float to_float(std::uint16_t h) {
#if defined(__CUDA_ARCH__)
    return __half2float(__ushort_as_half(h));
#else
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const int exponent = (h >> 10) & 0x1f;
    std::uint32_t mantissa = h & 0x3ffu;
    if (exponent == 0x1f) return Kokkos::bit_cast<float>(sign | 0x7f800000u | mantissa << 13);
    if (exponent == 0) {
        if (mantissa == 0) return Kokkos::bit_cast<float>(sign);
        int e = -1;
        do {
            mantissa <<= 1;
            ++e;
        } while (!(mantissa & 0x400u));
        return Kokkos::bit_cast<float>(sign | static_cast<std::uint32_t>(127 - 15 - e) << 23 | (mantissa & 0x3ffu) << 13);
    }
    return Kokkos::bit_cast<float>(sign | static_cast<std::uint32_t>(exponent - 15 + 127) << 23 | mantissa << 13);
#endif
}

// 2^k for k in [-126, 127]; used by: FiniteVolume's FP16 gradient scaling.
KOKKOS_INLINE_FUNCTION float power_of_two(int k) { return Kokkos::bit_cast<float>(static_cast<std::uint32_t>(k + 127) << 23); }

// The exponent k of a group whose largest magnitude is m: m 2^k in
// [2^14, 2^15) (k = 0 for m = 0), within [-114, 126]: no overflow, 11 bits
// relative to the group's largest component. Used by: FiniteVolume's gradient store.
KOKKOS_INLINE_FUNCTION int group_exponent(float m) {
    if (!(m > 0)) return 0;
    const int e = static_cast<int>((Kokkos::bit_cast<std::uint32_t>(m) >> 23) & 0xff) - 127;
    return Kokkos::clamp(14 - e, -114, 126);
}

// The group of gradient component c = D v + a: rho (v = 0), velocity, pressure; each group shares one exponent.
// Used by: FiniteVolume's gradient store and face kernel.
template <int D>
KOKKOS_INLINE_FUNCTION int group(int c) {
    return c < D ? 0 : c < D + D * D ? 1 : 2;
}

}  // namespace half

// Entries of a row of `count` values of type T, padded to whole 32-byte sectors.
// Used by: FiniteVolume::gradient_width.
template <class T>
constexpr int row_width(int count) {
    constexpr int per_sector = 32 / static_cast<int>(sizeof(T));
    return (count + per_sector - 1) / per_sector * per_sector;
}

}  // namespace hydro
