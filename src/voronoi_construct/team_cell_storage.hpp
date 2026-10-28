#pragma once
#include "local_voronoi.hpp"

namespace local_voronoi {
// Non-owning references into one team's scratch allocation. Copying this
// descriptor never copies the cell state or introduces per-thread arrays.
template<class T>
struct ScratchArray {
    T* data;
    KOKKOS_INLINE_FUNCTION T& operator[](int i) const { return data[i]; }
};

template<int Planes,class ScratchSpace>
struct TeamCellStorage {
    using Owner=Kokkos::View<CellClippingState<Planes>,ScratchSpace,Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Owner owner;
    ScratchArray<Plane> planes;
    ScratchArray<VertexPlanes> vertices;
    ScratchArray<Edge> hole;
    KOKKOS_INLINE_FUNCTION explicit TeamCellStorage(const ScratchSpace& scratch)
      : owner(scratch), planes{owner().planes.data()},
        vertices{owner().vertices.data()},hole{owner().hole.data()} {}
    static std::size_t bytes(){return Owner::shmem_size();}
};

template<class ScratchSpace>
struct TeamVertexPositions {
    static constexpr bool caches_coordinates=true;
    using View=Kokkos::View<Vec3*,ScratchSpace,Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    View coordinates;
    KOKKOS_INLINE_FUNCTION explicit TeamVertexPositions(const ScratchSpace& scratch)
      : coordinates(scratch,max_vertices) {}
    KOKKOS_INLINE_FUNCTION void initialize(int i,double r){set(i,{(i&1)?-r:r,(i&2)?-r:r,(i&4)?-r:r});}
    KOKKOS_INLINE_FUNCTION void keep(int dst,int src){coordinates(dst)=coordinates(src);}
    KOKKOS_INLINE_FUNCTION void set(int i,Vec3 x){coordinates(i)=x;}
    template<class Tolerance,class Planes>
    KOKKOS_INLINE_FUNCTION bool position(int i,const VertexPlanes&,const Planes&,Vec3& x) const {
        x=coordinates(i);return true;
    }
    static std::size_t bytes(){return View::shmem_size(max_vertices);}
};
} // namespace local_voronoi
