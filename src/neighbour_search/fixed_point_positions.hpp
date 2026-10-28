#pragma once
#include <Kokkos_Core.hpp>
#include <cstdint>

namespace local_voronoi {
// Experimental unit-periodic-box input adapter. Decodes on access; the current
// DenseCellSearch still gathers into double positions internally.
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
} // namespace local_voronoi
