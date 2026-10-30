#pragma once
#include <Kokkos_Core.hpp>
#include <cstdint>

namespace local_voronoi {
// Experimental unit-periodic-box input adapter. Decodes on access; the current
// DenseCellSearch still gathers into double positions internally. Reconstruction
// may instead receive a slot-ordered copy of values and subtract integers first.
template<class MemorySpace>
struct FixedPointPositions32 {
    Kokkos::View<std::uint32_t*[3],MemorySpace> values;
    static constexpr double scale=4294967296.0;
    KOKKOS_INLINE_FUNCTION double operator()(int i,int axis) const {
        return double(values(i,axis))/scale;
    }
    KOKKOS_INLINE_FUNCTION void set(int i,int axis,double x) const {
        const double wrapped=x-Kokkos::floor(x);
        values(i,axis)=static_cast<std::uint32_t>(wrapped*scale);
    }
    KOKKOS_INLINE_FUNCTION auto extent(int axis) const { return values.extent(axis); }
    // Signed shortest displacement in [-2^31,2^31), avoiding signed overflow.
    KOKKOS_INLINE_FUNCTION double displacement(int i,int j,int axis) const {
        std::int64_t d=std::int64_t(values(i,axis))-std::int64_t(values(j,axis));
        if (d>=2147483648LL) d-=4294967296LL;
        if (d< -2147483648LL) d+=4294967296LL;
        return double(d)/scale;
    }
};
// Reference displacement remains unchanged for floating-point position views.
template<class Positions>
KOKKOS_INLINE_FUNCTION double coordinate_displacement(
    const Positions& positions,int i,int j,int axis,double period) {
    double d=positions(i,axis)-positions(j,axis);
    if (period>0) d-=period*Kokkos::floor(d/period+0.5);
    return d;
}

// Unit-periodic coordinates: unsigned subtraction wraps modulo 2^32; interpret
// the high half as negative without implementation-defined signed casts.
// Other periods retain the reference wrapping semantics on the exact difference.
template<class MemorySpace>
KOKKOS_INLINE_FUNCTION double coordinate_displacement(
    const FixedPointPositions32<MemorySpace>& positions,int i,int j,int axis,double period) {
    if (period==1.0) {
        const std::uint32_t delta=positions.values(i,axis)-positions.values(j,axis);
        const std::int64_t signed_delta=delta<0x80000000u ? std::int64_t(delta)
            : std::int64_t(delta)-4294967296LL;
        return double(signed_delta)*0x1.0p-32;
    }
    double d=double(std::int64_t(positions.values(i,axis))-
                    std::int64_t(positions.values(j,axis)))*0x1.0p-32;
    if (period>0) d-=period*Kokkos::floor(d/period+0.5);
    return d;
}
} // namespace local_voronoi
