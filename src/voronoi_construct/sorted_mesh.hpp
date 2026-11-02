#pragma once

#include "exact_predicates.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cub/device/device_radix_sort.cuh>
#endif

// The 1D Voronoi mesh: generators sorted by their snapped position, faces
// at the midpoints between neighbours. This is AREPO's 1D mesh
// (/u/uli/codes/arepo/src/voronoi_1d.c: faces at the midpoints, :194; volume
// the distance between the faces and centre their midpoint, :204-205), with
// the order decided on the snapped integers and ties by particle id (the 1D
// symbolic perturbation), so it is exact.
//
// A periodic line [0, L) closes the order into a cycle. A line with walls
// [0, L] puts a mirror of the end generators behind each wall (AREPO's
// ghosts at -x_0 and 2L - x_{n-1}, voronoi_1d.c:130-133): the end cells
// reach the walls exactly.
//
// Distances are whole grid steps: gap(s) is the distance from slot s to
// slot s + 1 (to the mirror behind a wall: twice the distance to the wall),
// so volumes, (gap(s - 1) + gap(s)) / 2, and centroids relative to the
// generator, (gap(s) - gap(s - 1)) / 4, are exact multiples of a quarter
// step, the same bits wherever they are computed.
namespace sorted_mesh {

namespace ep = exact_predicates;
using i64 = ep::i64;

// The sorted line as kernels read it: slot s holds particle order(s) at
// grid step key(s).
template <class MemorySpace>
struct Line {
    Kokkos::View<int*, MemorySpace> order;
    Kokkos::View<std::uint32_t*, MemorySpace> key;
    int n = 0;
    i64 period = ep::period;  // grid steps across the line
    double step = 1 / ep::scale;
    bool periodic = true;

    // Slot s + k of a periodic line, or -1 past a wall.
    KOKKOS_INLINE_FUNCTION int slot(int s, int k) const {
        const int t = s + k;
        if (periodic) return t < 0 ? t + n : t >= n ? t - n : t;
        return t < 0 || t >= n ? -1 : t;
    }
    // Grid steps from slot s to slot s + 1, s in [-1, n): across the seam
    // of a periodic line, or to the mirror of the end generator behind a wall.
    KOKKOS_INLINE_FUNCTION i64 gap(int s) const {
        if (!periodic) {
            if (s < 0) return 2 * i64(key(0));
            if (s >= n - 1) return 2 * (period - i64(key(n - 1)));
            return i64(key(s + 1)) - i64(key(s));
        }
        const int a = s < 0 ? n - 1 : s, b = a + 1 == n ? 0 : a + 1;
        const i64 d = i64(key(b)) - i64(key(a));
        return d < 0 ? d + period : d;
    }
    // Volume and centroid (relative to the generator) of slot s.
    KOKKOS_INLINE_FUNCTION double volume(int s) const { return static_cast<double>(gap(s - 1) + gap(s)) * (0.5 * step); }
    KOKKOS_INLINE_FUNCTION double centroid(int s) const {
        return static_cast<double>(gap(s) - gap(s - 1)) * (0.25 * step);
    }
};

// (key, id) of a < (key, id) of b: the order of the line.
KOKKOS_INLINE_FUNCTION bool before(std::uint32_t ka, int a, std::uint32_t kb, int b) {
    return ka < kb || (ka == kb && a < b);
}

// The grid step of x on a line of `period` steps: wrapped (periodic) or
// clamped to [0, period) (walls).
KOKKOS_INLINE_FUNCTION std::uint32_t snap(double x, const ep::BoxN<1>& box, bool periodic) {
    if (periodic) return static_cast<std::uint32_t>(ep::snap(x, box, 0));
    const i64 s = static_cast<i64>(Kokkos::floor(x / box.step));
    return static_cast<std::uint32_t>(s < 0 ? 0 : s >= box.period[0] ? box.period[0] - 1 : s);
}

// Sorts the particles by (key, id): keys(i) of particle i in `keys`, the
// result in line.order and line.key. On CUDA a stable radix sort over the
// key bits (ids in increasing order keep ties by id); elsewhere std::sort.
template <class ExecSpace>
class Sorter {
public:
    using Memory = typename ExecSpace::memory_space;

    void sort(const ExecSpace& exec, const Kokkos::View<std::uint32_t*, Memory>& keys, const Kokkos::View<int*, Memory>& ids,
              const Line<Memory>& line) {
        const int n = line.n;
        const int bits = [&] {
            int b = 0;
            while ((i64(1) << b) < line.period) ++b;
            return b;
        }();
        Kokkos::parallel_for(
            "sorted_mesh::ids", Kokkos::RangePolicy<ExecSpace>(exec, 0, n), KOKKOS_LAMBDA(int i) { ids(i) = i; });
#if defined(KOKKOS_ENABLE_CUDA)
        if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
            std::size_t bytes = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys.data(), line.key.data(), ids.data(), line.order.data(), n, 0,
                                            bits, exec.cuda_stream());
            if (scratch_.extent(0) < bytes)
                scratch_ = Kokkos::View<char*, Memory>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("sorted_mesh_sort")), bytes);
            bytes = scratch_.extent(0);
            cub::DeviceRadixSort::SortPairs(scratch_.data(), bytes, keys.data(), line.key.data(), ids.data(), line.order.data(),
                                            n, 0, bits, exec.cuda_stream());
            return;
        }
#endif
        (void)bits;  // the host sort compares whole keys
        if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, Memory>::accessible) {
            exec.fence();
            std::uint32_t* k = keys.data();
            int* order = line.order.data();
            std::copy(ids.data(), ids.data() + n, order);
            std::sort(order, order + n, [&](int a, int b) { return before(k[a], a, k[b], b); });
            for (int s = 0; s < n; ++s) line.key(s) = k[order[s]];
        } else {
            throw std::runtime_error("sorted_mesh::Sorter: no sort for this backend");
        }
    }
    std::size_t bytes() const { return scratch_.extent(0); }

private:
    Kokkos::View<char*, Memory> scratch_;
};

}  // namespace sorted_mesh

namespace voronoi {

// The ends of a 1D line: periodic, or walls at 0 and L (whose boundary
// conditions belong to the hydro).
enum class Ends { periodic, walls };

// The 1D mesh: the sorted order of the generators (sorted_mesh).
//
// The contract of Mesh<3> holds with D = 1: faces once, owner the lower id,
// area +-1 from owner to neighbour, centroids relative to the owner; on a
// line with walls the two wall faces have neighbour -1 (at 0) and -2 (at
// L), area pointing out. Faces are numbered by slot: face s lies between
// slots s and s + 1 (s < n on a periodic line; n - 1 inner faces on a line
// with walls, then the wall faces n - 1 at 0 and n at L). The per-cell
// face lists are cell i's two faces at 2i, 2i + 1, in increasing id of the
// other particle (a wall's negative id first). update() computes all of it
// at once, without atomics; update_cells() has nothing left to do.
//
// update(x) snaps, then checks whether the order of the last update still
// holds (one pass; a periodic line may have rotated as particles wrapped
// around) and sorts again only if particles crossed. order(x) does only
// that: the 1D hydro reads the line and needs none of the face arrays.
template <class ExecSpace>
class Mesh<1, ExecSpace> {
public:
    static constexpr int dimensions = 1;
    using Memory = typename ExecSpace::memory_space;
    template <class T>
    using View = Kokkos::View<T, Memory>;
    using Positions = View<double* [1]>;
    using Line = sorted_mesh::Line<Memory>;

    Mesh(const ExecSpace& exec, const ep::BoxN<1>& box, Ends ends = Ends::periodic)
        : exec_(exec), box_(box), periodic_(ends == Ends::periodic) {}
    Mesh(const ExecSpace& exec, double side, Ends ends = Ends::periodic) : Mesh(exec, ep::BoxN<1>(side), ends) {}

    // New positions x(i, 0) in [0, L) (walls: positions outside are put
    // on the nearest end step). Builds on the first call or when the number
    // of particles changed. `active` and `timings` as Mesh<3> (unused).
    template <class X>
    Report update(const X& x, const View<const int*>& active = {}, Timings* timings = nullptr) {
        (void)active;
        (void)timings;
        Report report = order(x);
        geometry();
        report.faces = faces();
        return report;
    }
    template <class X>
    Report rebuild(const X& x, Timings* timings = nullptr) {
        (void)timings;
        n_ = -1;
        return update(x);
    }
    // The order alone (face and cell arrays stale until update_cells()).
    template <class X>
    Report order(const X& x) {
        Report report;
        const int n = static_cast<int>(x.extent(0));
        if (n < 3) throw std::invalid_argument("voronoi::Mesh<1>: at least 3 particles");
        if (n != n_) {
            allocate(n);
            snap_all(x);
            sorter_.sort(exec_, keys_, ids_, line_);
            report.rebuilt = true;
        } else if (!still_sorted(x)) {
            snap_all(x);
            sorter_.sort(exec_, keys_, ids_, line_);
            report.rebuilt = true;  // sorted again
            ++sorts_;
        }
        current_ = false;
        return report;
    }
    void update_cells(Timings* timings = nullptr) {
        (void)timings;
        geometry();
    }
    void set_fallback(Fallback) {}

    // Scratch memory the caller may use between updates, grown to `bytes`
    // (valid until the next update; the 1D mesh itself needs none).
    using Scratch = Kokkos::View<char*, Memory, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Scratch scratch(std::size_t bytes) {
        if (scratch_.extent(0) < bytes) scratch_ = View<char*>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("mesh1_scratch")), bytes);
        return Scratch(scratch_.data(), bytes);
    }

    int particles() const { return n_; }
    bool periodic() const { return periodic_; }
    const ep::BoxN<1>& sides() const { return box_; }
    double box() const { return box_.side[0]; }
    int faces() const { return periodic_ ? n_ : n_ + 1; }
    View<const int*> owner() const { return owner_; }
    View<const int*> neighbour() const { return neighbour_; }
    View<const double* [1]> area() const { return area_; }
    View<const double* [1]> face_centroid() const { return face_centroid_; }
    View<const double*> volume() const { return volume_; }
    View<const double* [1]> centroid() const { return centroid_; }
    View<const int*> cell_face_start() const { return cell_face_start_; }
    View<const int*> cell_face() const { return cell_face_; }
    // Snapped positions in id order (after update(); stale after order()).
    Positions positions() const { return positions_; }
    // The sorted line (current after update() and order()).
    const Line& line() const { return line_; }
    // How often the order had to be sorted again since the first build.
    long long sorts() const { return sorts_; }

    // Internals, public only because CUDA device lambdas may not be
    // defined in private member functions.
    void allocate(int n) {
        n_ = n;
        sorts_ = 0;
        auto make = [](auto& v, const char* label, auto... extents) {
            using V = std::decay_t<decltype(v)>;
            v = V(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)), extents...);
        };
        line_.n = n;
        line_.period = box_.period[0];
        line_.step = box_.step;
        line_.periodic = periodic_;
        make(line_.order, "mesh1_order", n);
        make(line_.key, "mesh1_key", n);
        make(keys_, "mesh1_keys_by_id", n);
        make(ids_, "mesh1_ids", n);
        const int faces = periodic_ ? n : n + 1;
        make(owner_, "mesh1_owner", faces);
        make(neighbour_, "mesh1_neighbour", faces);
        make(area_, "mesh1_area", faces);
        make(face_centroid_, "mesh1_face_centroid", faces);
        make(volume_, "mesh1_volume", n);
        make(centroid_, "mesh1_centroid", n);
        make(positions_, "mesh1_positions", n);
        make(cell_face_, "mesh1_cell_face", 2 * n);
        make(cell_face_start_, "mesh1_cell_face_start", n + 1);
        const auto start = cell_face_start_;
        Kokkos::parallel_for(
            "mesh1::cell_face_start", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n + 1), KOKKOS_LAMBDA(int i) { start(i) = 2 * i; });
    }

    template <class X>
    void snap_all(const X& x) {
        const auto keys = keys_;
        const ep::BoxN<1> box = box_;
        const bool periodic = periodic_;
        Kokkos::parallel_for(
            "mesh1::snap", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_),
            KOKKOS_LAMBDA(int i) { keys(i) = sorted_mesh::snap(x(i, 0), box, periodic); });
    }

    // Snaps x into the line's slots and counts the slots out of order: a
    // periodic line is still sorted (as a cycle) with exactly one descent,
    // one with walls with none.
    template <class X>
    bool still_sorted(const X& x) {
        const Line line = line_;
        const ep::BoxN<1> box = box_;
        const bool periodic = periodic_;
        int descents = 0;
        Kokkos::parallel_reduce(
            "mesh1::check_order", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_),
            KOKKOS_LAMBDA(int s, int& count) {
                const int i = line.order(s);
                const std::uint32_t k = sorted_mesh::snap(x(i, 0), box, periodic);
                line.key(s) = k;
                if (!periodic && s == line.n - 1) return;
                const int t = s + 1 == line.n ? 0 : s + 1, j = line.order(t);
                count += !sorted_mesh::before(k, i, sorted_mesh::snap(x(j, 0), box, periodic), j);
            },
            descents);
        return descents == (periodic ? 1 : 0);
    }

    // Faces, volumes, centroids and per-cell face lists from the line, one
    // thread per slot.
    void geometry() {
        if (current_) return;
        const Line line = line_;
        const auto owner = owner_, neighbour = neighbour_, cell_face = cell_face_;
        const auto area = area_, face_centroid = face_centroid_, centroid = centroid_, positions = positions_;
        const auto volume = volume_;
        const bool periodic = periodic_;
        const int n = n_;
        Kokkos::parallel_for(
            "mesh1::geometry", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n), KOKKOS_LAMBDA(int s) {
                const int i = line.order(s), r = line.slot(s, 1), l = line.slot(s, -1);
                const int right = r < 0 ? -2 : line.order(r), left = l < 0 ? -1 : line.order(l);
                const double half = static_cast<double>(line.gap(s)) * (0.5 * line.step);
                positions(i, 0) = static_cast<double>(line.key(s)) * line.step;
                volume(i) = line.volume(s);
                centroid(i, 0) = line.centroid(s);
                // The face to the right (slot s), and the wall faces.
                const int f_right = r < 0 ? n : s;
                const int f_left = l < 0 ? n - 1 : (periodic ? (s == 0 ? n - 1 : s - 1) : s - 1);
                if (r >= 0) {
                    owner(s) = i < right ? i : right;
                    neighbour(s) = i < right ? right : i;
                    area(s, 0) = i < right ? 1.0 : -1.0;
                    face_centroid(s, 0) = i < right ? half : -half;
                } else {
                    owner(n) = i;
                    neighbour(n) = -2;
                    area(n, 0) = 1.0;
                    face_centroid(n, 0) = half;  // half the gap to the mirror: the wall
                }
                if (l < 0) {
                    owner(n - 1) = i;
                    neighbour(n - 1) = -1;
                    area(n - 1, 0) = -1.0;
                    face_centroid(n - 1, 0) = -static_cast<double>(line.gap(-1)) * (0.5 * line.step);
                }
                const bool left_first = left < right;
                cell_face(2 * i) = left_first ? f_left : f_right;
                cell_face(2 * i + 1) = left_first ? f_right : f_left;
            });
        current_ = true;
    }

private:
    ExecSpace exec_;
    ep::BoxN<1> box_;
    bool periodic_;
    int n_ = 0;
    long long sorts_ = 0;
    bool current_ = false;
    Line line_;
    View<std::uint32_t*> keys_;
    View<int*> ids_;
    sorted_mesh::Sorter<ExecSpace> sorter_;
    View<int*> owner_, neighbour_, cell_face_, cell_face_start_;
    View<double* [1]> area_, face_centroid_, centroid_;
    Positions positions_;
    View<double*> volume_;
    View<char*> scratch_;
};

}  // namespace voronoi
