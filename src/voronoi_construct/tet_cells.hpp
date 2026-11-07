#pragma once

#include "tet_mesh.hpp"

// Stage 5b (geometry): per-cell face lists, volumes and centroids, and the gradient weights, on top of tet_mesh.hpp's faces.
// Called by Mesh::update_cells (3D: voronoi_mesh.hpp; 2D: voronoi_mesh_2d.hpp): cells or cells_incremental, then
// gradient_weights. Same namespace as tet_mesh.hpp, which it needs for Box, Vec3 and the face arrays.
namespace tet_mesh {

// Per-cell face lists (CSR) and cell geometry: cell i's faces are face(start(i)) .. face(start(i + 1) - 1) in increasing
// id of the other particle; volume and centroid (relative to the generator) are summed in that order, so run to run the same.
template <class MemorySpace, int D = 3>
struct CellFaces {
    Kokkos::View<int*, MemorySpace> start, face;
    Kokkos::View<int*, MemorySpace> cursor;  // fill positions while building
    Kokkos::View<double*, MemorySpace> volume;
    Kokkos::View<double* [D], Kokkos::LayoutRight, MemorySpace> centroid;
};

// Bytes of scratch cells() needs for `faces` faces: the other particle of
// each list entry.
inline std::size_t cells_scratch_bytes(int faces) { return 2 * static_cast<std::size_t>(faces) * sizeof(int); }

namespace detail {

// In D dimensions a cell's pyramid over a face has volume |A.d| / (2D)
// and its centroid at D / (D + 1) of the way from the generator to the face
// centroid (3D: 1/6 and 3/4).
template <int D>
constexpr double pyramid_volume = 1.0 / (2 * D);
// Fraction of the generator-to-face-centroid distance at which a pyramid's centroid lies, D / (D + 1).
// Used by: CellLists::finish.
template <int D>
constexpr double pyramid_centroid = static_cast<double>(D) / (D + 1);

// Cell i's pyramid over face f (other particle j): 2D times its volume, |A.d| (A parallel to d up to rounding),
// and the face centroid c relative to i (given relative to the owner, the lower id).
template <int D, class X, class Area, class Centroid>
KOKKOS_INLINE_FUNCTION double pyramid(const X& x, const exact_predicates::BoxN<D>& box, const Area& area,
                                      const Centroid& centroid, int i, int j, int f, VecN<D>& c) {
    const VecN<D> d = wrap(row<D>(x, j) - row<D>(x, i), box);
    const VecN<D> A = row<D>(area, f);
    c = row<D>(centroid, f);
    if (j < i) c = c + d;
    return Kokkos::abs(dot(A, d));
}

// Sort cell i's list entries [begin, end) by the other particle, the XOR of
// the face's two particles and i. `other`: room for the keys.
template <class Ints, class Others, class Const>
KOKKOS_INLINE_FUNCTION void sort_list(const Ints& face, const Others& other, const Const& owner, const Const& neighbour,
                                      int i, int begin, int end) {
    for (int a = begin; a < end; ++a) other(a) = owner(face(a)) ^ neighbour(face(a)) ^ i;
    for (int a = begin + 1; a < end; ++a) {
        const int k = other(a), f = face(a);
        int b = a;
        for (; b > begin && other(b - 1) > k; --b) {
            other(b) = other(b - 1);
            face(b) = face(b - 1);
        }
        other(b) = k;
        face(b) = f;
    }
}

// The sums of cells(), viewed through this: the lists of the `marked` cells
// (all if `all`) are sorted and in `face`; an unmarked cell's list is its
// old list (old_start, old_face) under new numbers (renumber), still sorted.
template <int D, class X, class Areas, class Ints, class Const, class Marks, class Volume, class Centroids>
struct CellLists {
    X x;
    exact_predicates::BoxN<D> box;
    Areas area, centroid;
    Ints start, face;
    Const old_start, old_face, renumber, owner, neighbour;
    Marks marked;
    Volume volume;
    Centroids cell_centroid;
    int n;
    bool all;

    KOKKOS_INLINE_FUNCTION bool kept(int i) const { return !all && !marked(i); }
    // List entry `at` of a cell, moved into `face` if it is kept.
    KOKKOS_INLINE_FUNCTION int entry(int begin, int from, int at, bool is_kept) const {
        if (!is_kept) return face(begin + at);
        const int f = renumber(old_face(from + at));
        face(begin + at) = f;
        return f;
    }
    KOKKOS_INLINE_FUNCTION double side(int i, int f, VecN<D>& c) const {
        return pyramid<D>(x, box, area, centroid, i, owner(f) ^ neighbour(f) ^ i, f, c);
    }
    KOKKOS_INLINE_FUNCTION void finish(int i, double v, VecN<D> moment) const {
        volume(i) = v * pyramid_volume<D>;
        store(cell_centroid, i, moment * (pyramid_centroid<D> / v));
    }
    // The sequential sum of one cell.
    KOKKOS_INLINE_FUNCTION void sum(int i) const {
        const int begin = start(i), length = start(i + 1) - begin, from = all ? 0 : old_start(i);
        const bool is_kept = kept(i);
        double v = 0;
        VecN<D> moment{};
        for (int at = 0; at < length; ++at) {
            VecN<D> c;
            const double p = side(i, entry(begin, from, at, is_kept), c);
            v += p;
            moment = moment + c * p;
        }
        finish(i, v, moment);
    }
};

#if defined(KOKKOS_ENABLE_CUDA)
// Sorting on CUDA, a warp per cell: lane l holds entry l, a warp bitonic sort.
template <class Ints, class Others, class Const, class Marks>
struct CellSortWarp {
    static constexpr int warps = 4;  // cells per block
    Ints start, face;
    Others other;
    Const owner, neighbour;
    Marks marked;
    int n;
    bool all;

    KOKKOS_FUNCTION void operator()(const typename Kokkos::TeamPolicy<Kokkos::Cuda>::member_type& member) const {
        const int i = member.league_rank() * warps + member.team_rank();
        if (i >= n || !(all || marked(i) == 2)) return;
#if defined(__CUDA_ARCH__)
        const int begin = start(i), end = start(i + 1);
        constexpr unsigned everyone = 0xffffffffu;
        const int l = threadIdx.x;
        if (end - begin > 32) {
            if (l == 0) sort_list(face, other, owner, neighbour, i, begin, end);
            return;
        }
        int key = 0x7fffffff, f = -1;
        if (begin + l < end) {
            f = face(begin + l);
            key = owner(f) ^ neighbour(f) ^ i;
        }
        for (int size = 2; size <= 32; size <<= 1)
            for (int stride = size / 2; stride > 0; stride >>= 1) {
                const int other_key = __shfl_xor_sync(everyone, key, stride);
                const int other_f = __shfl_xor_sync(everyone, f, stride);
                const bool ascending = (l & size) == 0, lower = (l & stride) == 0;
                if ((other_key < key) == (lower == ascending) && other_key != key) {
                    key = other_key;
                    f = other_f;
                }
            }
        if (begin + l < end) face(begin + l) = f;
#endif
    }
};

// The sums on CUDA: a warp holds 32 / lanes cells, the lanes of a cell gather its list a batch at a time, in list order,
// so the bits equal the sequential loop of CellLists::sum (a pyramid of zero past the list's end changes nothing).
template <class Lists, int D>
struct CellSumsWarp {
    static constexpr int warps = 4, lanes = 4, cells = 32 / lanes;
    static constexpr int block = warps * cells;  // cells per block
    Lists lists;

    KOKKOS_FUNCTION void operator()(const typename Kokkos::TeamPolicy<Kokkos::Cuda>::member_type& member) const {
#if defined(__CUDA_ARCH__)
        constexpr unsigned everyone = 0xffffffffu;
        const int l = threadIdx.x, lane = l % lanes, first = l - lane;
        const int i = (member.league_rank() * warps + member.team_rank()) * cells + l / lanes;
        const bool real = i < lists.n;
        const int begin = real ? lists.start(i) : 0, length = real ? lists.start(i + 1) - begin : 0;
        const int from = real && !lists.all ? lists.old_start(i) : 0;
        const bool is_kept = real && lists.kept(i);
        int batches = (length + lanes - 1) / lanes;
        for (int stride = lanes; stride < 32; stride <<= 1)
            batches = Kokkos::max(batches, __shfl_xor_sync(everyone, batches, stride));
        double v = 0;
        VecN<D> moment{};
        for (int b = 0; b < batches; ++b) {
            const int at = lane + b * lanes;
            double p = 0;
            VecN<D> c{};
            if (at < length) p = lists.side(i, lists.entry(begin, from, at, is_kept), c);
            for (int r = 0; r < lanes; ++r) {
                const double pr = __shfl_sync(everyone, p, first + r);
                VecN<D> cr;
                cr.x = __shfl_sync(everyone, c.x, first + r);
                cr.y = __shfl_sync(everyone, c.y, first + r);
                if constexpr (D == 3) cr.z = __shfl_sync(everyone, c.z, first + r);
                v += pr;
                moment = moment + cr * pr;
            }
        }
        if (real && lane == 0) lists.finish(i, v, moment);
#endif
    }
};
#endif

// Sort the `marked` lists (all if `all`), then sum every cell.
template <int D, class ExecSpace, class X, class Others, class Marks>
void sort_and_sum(const ExecSpace& exec, const X& x, const exact_predicates::BoxN<D>& box,
                  const Geometry<typename ExecSpace::memory_space, D>& geometry,
                  const CellFaces<typename ExecSpace::memory_space, D>& out,
                  const Kokkos::View<const int*, typename ExecSpace::memory_space>& old_start,
                  const Kokkos::View<const int*, typename ExecSpace::memory_space>& old_face,
                  const Kokkos::View<const int*, typename ExecSpace::memory_space>& renumber,
                  const Kokkos::View<const int*, typename ExecSpace::memory_space>& owner,
                  const Kokkos::View<const int*, typename ExecSpace::memory_space>& neighbour, const Others& other,
                  const Marks& marked, bool all, bool sort = true) {
    using Memory = typename ExecSpace::memory_space;
    using Const = Kokkos::View<const int*, Memory>;
    using Ints = Kokkos::View<int*, Memory>;
    using Lists = CellLists<D, X, typename Geometry<Memory, D>::Areas, Ints, Const, Marks,
                            Kokkos::View<double*, Memory>, Kokkos::View<double* [D], Kokkos::LayoutRight, Memory>>;
    const int n = static_cast<int>(out.volume.extent(0));
    const Lists lists{x,        box,      geometry.area, geometry.centroid, out.start, out.face,   old_start,
                      old_face, renumber, owner,         neighbour,         marked,    out.volume, out.centroid,
                      n,        all};
#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
        using Sort = CellSortWarp<Ints, Others, Const, Marks>;
        using Sums = CellSumsWarp<Lists, D>;
        if (sort)
            Kokkos::parallel_for(
                "tet_mesh::cell_sort",
                Kokkos::TeamPolicy<ExecSpace>(exec, (n + Sort::warps - 1) / Sort::warps, Sort::warps, 32),
                Sort{out.start, out.face, other, owner, neighbour, marked, n, all});
        Kokkos::parallel_for("tet_mesh::cell_sums",
                             Kokkos::TeamPolicy<ExecSpace>(exec, (n + Sums::block - 1) / Sums::block, Sums::warps, 32),
                             Sums{lists});
        return;
    }
#endif
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const auto start = out.start;
    const auto face = out.face;
    if (sort)
        Kokkos::parallel_for(
            "tet_mesh::cell_sort", Range(exec, 0, n), KOKKOS_LAMBDA(int i) {
                if (all || marked(i) == 2) sort_list(face, other, owner, neighbour, i, start(i), start(i + 1));
            });
    Kokkos::parallel_for("tet_mesh::cell_sums", Range(exec, 0, n), KOKKOS_LAMBDA(int i) { lists.sum(i); });
}

}  // namespace detail

// 3. The cell face lists, volumes and centroids from the faces (owner, neighbour) and their geometry (area, centroid),
// D = 2 or 3, x the positions the geometry came from; `scratch` holds cells_scratch_bytes(faces) bytes. Grows out.face when needed.
// One pass appends each face to both cells' lists; each cell then sorts its list by the other particle's id and sums its
// pyramids in that order (detail::sort_and_sum): the same sums as faces(), in fixed order.
// Used by: Mesh::update_cells (full path).
template <class ExecSpace, class X, int D>
void cells(const ExecSpace& exec, const X& x, const exact_predicates::BoxN<D>& box,
           const Kokkos::View<const int*, typename ExecSpace::memory_space>& owner,
           const Kokkos::View<const int*, typename ExecSpace::memory_space>& neighbour,
           const Geometry<typename ExecSpace::memory_space, D>& geometry,
           CellFaces<typename ExecSpace::memory_space, D>& out, char* scratch) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const int n = static_cast<int>(out.volume.extent(0)), faces = static_cast<int>(owner.extent(0));
    if (static_cast<int>(out.face.extent(0)) < 2 * faces)
        Kokkos::realloc(Kokkos::WithoutInitializing, out.face, 2 * faces + faces / 16);
    const Kokkos::View<int*, Memory, Kokkos::MemoryTraits<Kokkos::Unmanaged>> other(reinterpret_cast<int*>(scratch),
                                                                                    2 * faces);
    const auto start = out.start, cursor = out.cursor, face = out.face;
    Kokkos::deep_copy(exec, start, 0);
    Kokkos::parallel_for(
        "tet_mesh::cell_face_counts", Range(exec, 0, faces), KOKKOS_LAMBDA(int f) {
            Kokkos::atomic_inc(&start(owner(f)));
            Kokkos::atomic_inc(&start(neighbour(f)));
        });
    Kokkos::parallel_scan(
        "tet_mesh::cell_face_start", Range(exec, 0, n + 1), KOKKOS_LAMBDA(int i, int& sum, bool final) {
            const int count = i < n ? start(i) : 0;
            if (final) {
                start(i) = sum;
                if (i < n) cursor(i) = sum;
            }
            sum += count;
        });
    Kokkos::parallel_for(
        "tet_mesh::cell_face_fill", Range(exec, 0, faces), KOKKOS_LAMBDA(int f) {
            const int o = owner(f), j = neighbour(f);
            const int at_owner = Kokkos::atomic_fetch_inc(&cursor(o)),
                      at_neighbour = Kokkos::atomic_fetch_inc(&cursor(j));
            face(at_owner) = f;
            face(at_neighbour) = f;
        });
    detail::sort_and_sum<D>(exec, x, box, geometry, out, {}, {}, {}, owner, neighbour, other,
                            Kokkos::View<const char*, Memory, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(), true);
}

// Bytes of scratch cells_incremental needs after the renumbering table.
inline std::size_t incremental_scratch_bytes(int n, int old_faces, int faces) {
    return sizeof(int) * (static_cast<std::size_t>(n) + 1 + 2 * static_cast<std::size_t>(old_faces) +
                          2 * static_cast<std::size_t>(faces)) +
           n;
}

namespace detail {

// The scratch of incremental lists: the old lists, the sort keys, the marks.
template <class Memory>
struct IncrementalLists {
    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using Ints = Kokkos::View<int*, Memory, Unmanaged>;
    Ints old_start, old_face, other;
    Kokkos::View<char*, Memory, Unmanaged> marked;
};

// Sets up in `at` (incremental_scratch_bytes(n, old_faces, faces) bytes): copies
// the old lists, clears the marks, makes room for the new lists.
template <class ExecSpace, int D>
IncrementalLists<typename ExecSpace::memory_space> begin_incremental(
    const ExecSpace& exec, int* at, int n, int old_faces, int faces,
    CellFaces<typename ExecSpace::memory_space, D>& out) {
    using Memory = typename ExecSpace::memory_space;
    using Lists = IncrementalLists<Memory>;
    IncrementalLists<Memory> l;
    l.old_start = typename Lists::Ints(at, n + 1);
    at += n + 1;
    l.old_face = typename Lists::Ints(at, 2 * old_faces);
    at += 2 * old_faces;
    l.other = typename Lists::Ints(at, 2 * faces);
    at += 2 * faces;
    l.marked = Kokkos::View<char*, Memory, typename Lists::Unmanaged>(reinterpret_cast<char*>(at), n);
    Kokkos::deep_copy(exec, l.old_start, Kokkos::View<const int*, Memory>(out.start));
    Kokkos::deep_copy(exec, l.old_face, Kokkos::subview(out.face, std::make_pair(0, 2 * old_faces)));
    Kokkos::deep_copy(exec, l.marked, char(0));
    if (static_cast<int>(out.face.extent(0)) < 2 * faces)
        Kokkos::realloc(Kokkos::WithoutInitializing, out.face, 2 * faces + faces / 16);
    return l;
}

// The lists and sums from the old lists, D = 2 or 3; l.marked is set for every cell that may have lost or gained a face;
// renumber(f) is the new number of old face f (-1: none), and ~f in dirty[0, dirty_count) names each new face without one once.
template <int D, class ExecSpace, class X, class Dirty>
void finish_incremental(const ExecSpace& exec, const X& x, const exact_predicates::BoxN<D>& box,
                        const Kokkos::View<const int*, typename ExecSpace::memory_space>& owner,
                        const Kokkos::View<const int*, typename ExecSpace::memory_space>& neighbour,
                        const Geometry<typename ExecSpace::memory_space, D>& geometry,
                        CellFaces<typename ExecSpace::memory_space, D>& out,
                        const Kokkos::View<const int*, typename ExecSpace::memory_space>& renumber, const Dirty& dirty,
                        int dirty_count, const IncrementalLists<typename ExecSpace::memory_space>& lists) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    const int n = static_cast<int>(out.volume.extent(0));
    const auto old_start = lists.old_start, old_face = lists.old_face, other = lists.other;
    const auto marked = lists.marked;
    const auto start = out.start, cursor = out.cursor, face = out.face;
    // The marked cells' counts in `start`, the others' from the old list.
    Kokkos::parallel_for(
        "tet_mesh::marked_counts", Range(exec, 0, n), KOKKOS_LAMBDA(int i) {
            if (!marked(i)) return;
            int kept = 0;
            for (int a = old_start(i); a < old_start(i + 1); ++a) kept += renumber(old_face(a)) >= 0;
            start(i) = kept;
        });
    Kokkos::parallel_for(
        "tet_mesh::fresh_counts", Range(exec, 0, dirty_count), KOKKOS_LAMBDA(int s) {
            if (dirty(s) >= 0) return;
            const int f = ~dirty(s);
            Kokkos::atomic_inc(&start(owner(f)));
            Kokkos::atomic_inc(&start(neighbour(f)));
        });
    Kokkos::parallel_scan(
        "tet_mesh::cell_face_start", Range(exec, 0, n + 1), KOKKOS_LAMBDA(int i, int& sum, bool final) {
            const int count = i == n ? 0 : marked(i) ? start(i) : old_start(i + 1) - old_start(i);
            if (final) start(i) = sum;
            sum += count;
        });
    Kokkos::parallel_for(
        "tet_mesh::marked_fill", Range(exec, 0, n), KOKKOS_LAMBDA(int i) {
            if (!marked(i)) return;
            int to = start(i);
            for (int a = old_start(i); a < old_start(i + 1); ++a) {
                const int f = renumber(old_face(a));
                if (f >= 0) face(to++) = f;
            }
            cursor(i) = to;
        });
    Kokkos::parallel_for(
        "tet_mesh::fresh_fill", Range(exec, 0, dirty_count), KOKKOS_LAMBDA(int s) {
            if (dirty(s) >= 0) return;
            const int f = ~dirty(s);
            const int o = owner(f), j = neighbour(f);
            face(Kokkos::atomic_fetch_inc(&cursor(o))) = f;
            face(Kokkos::atomic_fetch_inc(&cursor(j))) = f;
            marked(o) = 2;
            marked(j) = 2;
        });
    sort_and_sum<D>(exec, x, box, geometry, out, old_start, old_face, renumber, owner, neighbour, other,
                    Kokkos::View<const char*, Memory, Unmanaged>(marked), false);
}

}  // namespace detail

// 3'. cells() after a repair, from the last step's lists, bit for bit the same: only cells with a corner of a
// tetrahedron created by a flip (list `created`) can change faces; the others renumber (`renumber`) and sum again.
// Used by: Mesh<3>::update_cells (when few entries were touched).
template <class ExecSpace, class X>
void cells_incremental(const ExecSpace& exec, const X& x, const Box& box,
                       const Kokkos::View<const int*, typename ExecSpace::memory_space>& owner,
                       const Kokkos::View<const int*, typename ExecSpace::memory_space>& neighbour,
                       const Mesh<typename ExecSpace::memory_space>& mesh,
                       const Kokkos::View<const char*, typename ExecSpace::memory_space>& alive,
                       const Kokkos::View<const int*, typename ExecSpace::memory_space>& created,
                       const FaceRings<typename ExecSpace::memory_space>& rings,
                       const Geometry<typename ExecSpace::memory_space, 3>& geometry,
                       CellFaces<typename ExecSpace::memory_space, 3>& out, char* scratch) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using Ints = Kokkos::View<int*, Memory, Unmanaged>;
    const int n = static_cast<int>(out.volume.extent(0)), faces = static_cast<int>(owner.extent(0));
    const int old_faces = rings.renumbered;
    // After the table and the dirty list (in the spare pool, update_face_rings).
    const Ints renumber(reinterpret_cast<int*>(scratch), old_faces);
    const auto lists = detail::begin_incremental<ExecSpace, 3>(exec, renumber.data() + old_faces + rings.dirty_count, n,
                                                               old_faces, faces, out);
    const auto marked = lists.marked;
    const auto vertex = mesh.vertex;
    Kokkos::parallel_for(
        "tet_mesh::mark_cells", Range(exec, 0, created.extent(0)), KOKKOS_LAMBDA(int s) {
            const int t = created(s);
            if (!alive(t)) return;
            for (int c = 0; c < 4; ++c) marked(vertex(t, c)) = 1;
        });
    detail::finish_incremental<3>(exec, x, box, owner, neighbour, geometry, out,
                                  Kokkos::View<const int*, Memory>(renumber), rings.dirty, rings.dirty_count, lists);
}

// 4. Least-squares gradient weights of every per-cell list entry, as Arepo's prepare_gradient_operator: for cell i with
// entries k (face f, other cell j), X_k = (1, d_k), d_k from the centroid of i to that of j (nearest image), w_k = |A_f|,
// M = sum_k w_k X_k X_k^T; entry k gets weight(k) = w_k (M^-1 X_k)_{1..D}, so sum_k weight(k) (u_j - u_i) is the gradient
// of the area-weighted fit u_j - u_i = a + g.d_k. In double (Cholesky of M), stored in the type of `out` (rows of D);
// one thread per cell, after cells(). Used by: Mesh::update_cells (Weight != void).
template <class ExecSpace, class Xs, int D, class Weights>
void gradient_weights(const ExecSpace& exec, const Xs& x, const exact_predicates::BoxN<D>& box,
                      const Kokkos::View<const int*, typename ExecSpace::memory_space>& owner,
                      const Kokkos::View<const int*, typename ExecSpace::memory_space>& neighbour,
                      const Geometry<typename ExecSpace::memory_space, D>& geometry,
                      const CellFaces<typename ExecSpace::memory_space, D>& cells, const Weights& out) {
    using Weight = typename Weights::value_type;
    const int n = static_cast<int>(cells.volume.extent(0));
    const auto start = cells.start, face = cells.face;
    const auto centroid = cells.centroid;
    const auto area = geometry.area;
    Kokkos::parallel_for(
        "tet_mesh::gradient_weights", Kokkos::RangePolicy<ExecSpace>(exec, 0, n), KOKKOS_LAMBDA(int i) {
            constexpr int U = D + 1;
            // Regressors (1, d) and weight of entry k.
            auto regressors = [&](int k, double* X) {
                const int f = face(k), o = owner(f), m = neighbour(f), j = o == i ? m : o;
                X[0] = 1;
                for (int a = 0; a < D; ++a)
                    X[1 + a] = wrap(x(j, a) - x(i, a), box.side[a], box.inverse[a]) + centroid(j, a) - centroid(i, a);
                double s = 0;
                for (int a = 0; a < D; ++a) s += area(f, a) * area(f, a);
                return Kokkos::sqrt(s);
            };
            double M[U][U] = {};
            for (int k = start(i); k < start(i + 1); ++k) {
                double X[U];
                const double w = regressors(k, X);
                for (int p = 0; p < U; ++p)
                    for (int s = p; s < U; ++s) M[p][s] += w * X[p] * X[s];
            }
            double L[U][U] = {};
            for (int p = 0; p < U; ++p)
                for (int s = 0; s <= p; ++s) {
                    double sum = M[s][p];
                    for (int m = 0; m < s; ++m) sum -= L[p][m] * L[s][m];
                    L[p][s] = p == s ? Kokkos::sqrt(sum) : sum / L[s][s];
                }
            for (int k = start(i); k < start(i + 1); ++k) {
                double X[U], z[U], y[U];
                const double w = regressors(k, X);
                for (int p = 0; p < U; ++p) {
                    double sum = X[p];
                    for (int m = 0; m < p; ++m) sum -= L[p][m] * z[m];
                    z[p] = sum / L[p][p];
                }
                for (int p = U - 1; p >= 0; --p) {
                    double sum = z[p];
                    for (int m = p + 1; m < U; ++m) sum -= L[m][p] * y[m];
                    y[p] = sum / L[p][p];
                }
                for (int a = 0; a < D; ++a) out(k, a) = static_cast<Weight>(w * y[1 + a]);
            }
        });
}

}  // namespace tet_mesh
