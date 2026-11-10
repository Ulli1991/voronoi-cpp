#pragma once

#include "exact_predicates.hpp"
#include "neighbor_search.hpp"
#include "shell_voronoi_warp.hpp"
#include "tet_cells.hpp"
#include "tet_flips.hpp"
#include "tet_mesh.hpp"
#include "tet_repair.hpp"
#include "tet_start.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Top level (the interface luna-cpp's finite-volume code calls): Mesh<3>::update drives snap -> index -> start -> repair -> geometry (tet_*.hpp).
// Particles are named by id (index); cospherical ties break by id; topology and geometry both come from positions
// snapped to the 2³⁰ grid. Faces are stored once, by the lower id.
namespace voronoi {

namespace ep = exact_predicates;

namespace detail {
// Stage 1, snap: positions = x snapped to the grid (whole grid steps), so all later tests are exact.
// Used by: Mesh::snap (from update, rebuild, clip).
template <class ExecSpace, class X>
void snap_positions(const ExecSpace& exec, const X& x, const ep::Box& box,
                    const tet_mesh::Points4<typename ExecSpace::memory_space>& positions) {
    const double unit = box.step;
    Kokkos::parallel_for(
        "voronoi::snap", Kokkos::RangePolicy<ExecSpace>(exec, 0, positions.extent(0)), KOKKOS_LAMBDA(int i) {
            for (int a = 0; a < 3; ++a) positions(i, a) = ep::snap(x(i, a), box, a) * unit;
        });
}

// The sum of the cell volumes.
// Used by: Mesh::geometry, for update()'s check that the repaired mesh fills the box exactly once.
template <class ExecSpace, class Volume>
double total_volume(const ExecSpace& exec, const Volume& volume) {
    double total = 0;
    Kokkos::parallel_reduce(
        "voronoi::total_volume", Kokkos::RangePolicy<ExecSpace>(exec, 0, volume.extent(0)),
        KOKKOS_LAMBDA(int i, double& sum) { sum += volume(i); }, total);
    return total;
}

// Particles on one position of the 2³⁰ grid, found by sorting on the host (a refusal only): "" if none, else the
// error naming a pair. Such particles have no bisector, so no cell, and voronoi::Mesh refuses them.
// Used by: not_exact.
template <class Positions>
std::string coincident(const Positions& positions) {
    const auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, positions);
    std::vector<int> id(x.extent(0));
    std::iota(id.begin(), id.end(), 0);
    const bool third = x.extent(1) > 2;  // D = 2 or 3
    auto key = [&](int i) { return std::array<double, 3>{x(i, 0), x(i, 1), third ? x(i, 2) : 0.0}; };
    std::sort(id.begin(), id.end(), [&](int a, int b) { return std::make_pair(key(a), a) < std::make_pair(key(b), b); });
    long long extra = 0;
    int first = -1, second = -1;
    for (std::size_t i = 1; i < id.size(); ++i) {
        if (key(id[i]) != key(id[i - 1])) continue;
        if (extra++ == 0) {
            first = id[i - 1];
            second = id[i];
        }
    }
    if (extra == 0) return "";
    return "voronoi::Mesh: particles " + std::to_string(first) + " and " + std::to_string(second) +
           " are on one position of the snapping grid (" + std::to_string(extra) + " coincident in all" +
           "): a doubled particle has no Voronoi cell; move or merge them";
}

// Why a start failed: coincident particles, cells the clipping could not certify (in one periodic image, see
// shell_voronoi::level_reach), or else not exact. `cells` are the clipping's per-cell results (status).
// Used by: Mesh::rebuild, when tet_start::build is not exact (the message goes into the thrown runtime_error).
template <class Cells, class Positions>
std::string not_exact(const Cells& cells, const Positions& positions) {
    const std::string pair = coincident(positions);
    if (!pair.empty()) return pair;
    const auto status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.status);
    long long count[shell_voronoi::status_count] = {};
    for (std::size_t i = 0; i < status.extent(0); ++i) ++count[status(i)];
    if (count[shell_voronoi::uncertified] > 0)
        return "voronoi::Mesh: " + std::to_string(count[shell_voronoi::uncertified]) +
               " cells reach too far to be certified in one periodic image (a cell's radius must stay below about 3/16 "
               "of the shortest box side): too few particles across the box";
    const long long overflow = count[shell_voronoi::plane_overflow] + count[shell_voronoi::vertex_overflow];
    if (overflow > 0)
        return "voronoi::Mesh: " + std::to_string(overflow) + " cells need more than the clipping's " +
               std::to_string(shell_voronoi::huge_planes) + " planes or " + std::to_string(shell_voronoi::huge_vertices) +
               " vertices (too many particles around one cell)";
    return "voronoi::Mesh: the start is not exact (clipping statuses: " + std::to_string(count[0]) + " ok, " +
           std::to_string(count[shell_voronoi::face_overflow]) + " out of pool room, " +
           std::to_string(count[shell_voronoi::degenerate]) + " degenerate)";
}
}  // namespace detail

// What an update did: filled by update() and rebuild(), returned to the caller.
struct Report {
    bool rebuilt = false;          // built from scratch (first update, rebuild(), or the guard)
    tet_start::Stats start;        // of that build
    long long flips = 0;           // device flips
    int rounds = 0;                // device repair rounds
    int stuck = 0;                 // triangles the device repair could not repair
    bool overlapped = false;       // the repair's mesh did not fill the box once: rebuilt
    long long host_flips = 0;      // flips and local rebuilds of the host fallback
    long long host_rebuilds = 0;
    int faces = 0;
};

// Phase times of an update, in ms, for the benchmark: given to update(), the phases are fenced and timed.
// Used by: Mesh::lap, from update, rebuild, geometry, update_cells.
struct Timings {
    double repair = 0, fallback = 0, geometry = 0, build = 0, cells = 0;
};

// What update() does when the device repair leaves triangles it cannot repair: repair around them on the host
// (tet_flips; a rebuild if that does not finish), rebuild from scratch on the device, or (cheapest) choose by the
// number of stuck entries (host only up to n / 200000, none below 200000 cells). No clock decides: the same input
// takes the same path.
enum class Fallback { cheapest, host, rebuild };

// The mesh in D dimensions: one interface, an engine per D (3D tetrahedra, 2D voronoi_mesh_2d.hpp, 1D sorted_mesh.hpp).
//
// Workflow of the 3D mesh (public methods in call order):
//   update(x)        every step. First call, changed count, or failure -> rebuild(x); else
//                    snap -> tet_repair::move_screened -> tet_repair::repair (device) -> tet_flips::finish (host
//                    fallback, if stuck) -> geometry() (tet_mesh); a mesh not filling the box once -> rebuild(x).
//   rebuild(x)       from scratch: snap -> index -> shell_voronoi clipping -> tet_start::build -> repair::prepare
//                    -> geometry(). Throws std::runtime_error if the start is not exact.
//   update_cells()   optional, after update/rebuild: per-cell face lists, volumes, centroids (and gradient weights).
//   accessors        faces(), owner(), neighbour(), area(), volume(), ... read the last update's result.
//   clip(x, cells)   independent of the above: snap -> index -> clipping of the listed cells only.
//
// FaceReal = float adds one float row per face (face_rows()) for a reduced-precision solver; Weight = float or double
// adds the least-squares gradient weights of every per-cell list entry (computed by update_cells()); void: none.
template <int D, class ExecSpace = Kokkos::DefaultExecutionSpace, class FaceReal = double, class Weight = void>
class Mesh;

// Mesh<3>: the 3D Voronoi mesh (tets + shell clipping); the contract is described above.
// Used by: luna-cpp and the tests, through Mesh<3>; Mesh<2> is in voronoi_mesh_2d.hpp.
template <class ExecSpace, class FaceReal, class Weight>
class Mesh<3, ExecSpace, FaceReal, Weight> {
public:
    static_assert(std::is_same_v<FaceReal, double> || std::is_same_v<FaceReal, float>, "FaceReal: double or float");
    static_assert(std::is_void_v<Weight> || std::is_same_v<Weight, double> || std::is_same_v<Weight, float>,
                  "Weight: void, double or float");
    static constexpr int dimensions = 3;
    static constexpr bool float_rows = std::is_same_v<FaceReal, float>;
    static constexpr bool with_weights = !std::is_void_v<Weight>;
    using WeightReal = std::conditional_t<with_weights, Weight, float>;
    using Memory = typename ExecSpace::memory_space;
    template <class T>
    using View = Kokkos::View<T, Memory>;
    using Positions = tet_mesh::Points4<Memory>;
    using Rows = Kokkos::View<const double* [3], Kokkos::LayoutRight, Memory>;  // area, centroids
    using FaceRows = Kokkos::View<const float* [8], Kokkos::LayoutRight, Memory>;
    // Row k: the weight vector of list entry k (cell_face(k)).
    using Weights = Kokkos::View<const WeightReal* [3], Kokkos::LayoutRight, Memory>;

    // A periodic box of sides box.side[a] (see exact_predicates::Box: sides are rounded to whole grid steps, side[]
    // holds them), or a cube.
    Mesh(const ExecSpace& exec, const ep::Box& box) : exec_(exec), box_(box) {}
    Mesh(const ExecSpace& exec, double box) : exec_(exec), box_(box) {}

    // New positions of all particles, x(i, a) in [0, side[a]) (any device view). Repairs the mesh, or builds it on the
    // first call or when the particle count changed; exact on return, or a std::runtime_error saying what is wrong.
    // After a refusal there is no mesh and the next update() builds from scratch; no particles is a std::invalid_argument
    // and leaves the object as it was. `active` (particle ids) is accepted for a later regional repair; unused today.
    template <class X>
    Report update(const X& x, const View<const int*>& active = {}, Timings* timings = nullptr) {
        (void)active;
        if (!built() || static_cast<int>(x.extent(0)) != n_) return rebuild(x, timings);
        Report report;
        Kokkos::Timer timer;
        snap(x);
        tet_repair::move_screened(exec_, positions_, box_, tets_, state_);
        const tet_repair::Stats device = tet_repair::repair(exec_, tets_, state_, repair_rounds_);
        report.flips = device.flips23 + device.flips32;
        report.rounds = static_cast<int>(device.rounds);
        report.stuck = device.stuck;
        lap(timer, timings, &Timings::repair);
        bool stuck = device.stuck > 0 || device.unfinished;
        if (stuck && !device.unfinished && fallback_ != Fallback::rebuild &&
            (fallback_ == Fallback::host || static_cast<long long>(device.stuck) * 200000 <= n_)) {
            tet_flips::Stats host;
            try {
                host = tet_flips::finish(exec_, tets_, state_);
            } catch (const std::exception&) {
                host.stuck = 1;
            }
            report.host_flips = host.flips23 + host.flips32;
            report.host_rebuilds = host.rebuilt;
            stuck = host.stuck > 0;
            lap(timer, timings, &Timings::fallback);
        }
        if (!stuck) {
            report.faces = geometry(timings);
            // A step of order the box leaves tetrahedra whose corners see each other through different images: every
            // local test passes and the mesh covers the torus many times.
            report.overlapped = !(Kokkos::abs(total_volume_ / box_.volume() - 1) < 1e-6);
            if (!report.overlapped) return report;
        }
        // A rebuild, also for what the host left.
        Report rebuilt = rebuild(x, timings);
        rebuilt.flips = report.flips;
        rebuilt.rounds = report.rounds;
        rebuilt.stuck = report.stuck;
        rebuilt.host_flips = report.host_flips;
        rebuilt.host_rebuilds = report.host_rebuilds;
        rebuilt.overlapped = report.overlapped;
        return rebuilt;
    }

    // The exact mesh of positions x from scratch: clipping with face rings, the exact start (tet_start), then geometry.
    // Used by: update (first call, changed count, stuck or overlapping repair); also callable directly.
    template <class X>
    Report rebuild(const X& x, Timings* timings = nullptr) {
        if (x.extent(0) == 0) throw std::invalid_argument("voronoi::Mesh: no particles");
        Kokkos::Timer timer;
        Report report;
        report.rebuilt = true;
        const int particles = static_cast<int>(x.extent(0));
        forget();
        n_ = particles;
        snap(x);
        {
            index(positions_);
            auto cells = shell_voronoi::Cells<Memory>::with_rings(n_, ring_entries_ > 0 ? ring_entries_ : 64LL * n_);
            const auto grid = shell_voronoi::make_grid(list_);
            shell_voronoi::construct_warp(exec_, grid, cells);
            report.start = tet_start::build(exec_, grid, cells, tets_);
            if (!report.start.exact && cells.needed() > cells.capacity()) {
                cells = shell_voronoi::Cells<Memory>::with_rings(n_, cells.needed() + cells.needed() / 16);
                shell_voronoi::construct_warp(exec_, grid, cells);
                report.start = tet_start::build(exec_, grid, cells, tets_);
            }
            ring_entries_ = cells.capacity();
            if (!report.start.exact) {
                const std::string why = detail::not_exact(cells, positions_);
                forget();
                throw std::runtime_error(why);
            }
        }
        // Capacities: 1/8 more tetrahedra and 1/32 more faces than the starting mesh (ring pools 1/2 more); repair grows them.
        const int tets = static_cast<int>(tets_.vertex.extent(0));
        const int faces = static_cast<int>(tets_.owner.extent(0)) + static_cast<int>(tets_.owner.extent(0)) / 32;
        const int entries = static_cast<int>(tets_.ring.extent(0));
        tets_.owner = tets_.neighbour = tets_.ring_start = tets_.ring_length = tets_.ring = {};
        state_ = tet_repair::prepare(exec_, positions_, box_, tets_, tets + tets / 8, n_ / 4 + 1024);
        rings_ = tet_mesh::make_face_rings<Memory>(tets + tets / 8, faces, entries + entries / 2);
        using Areas = typename tet_mesh::Geometry<Memory>::Areas;
        geometry_ = {Areas("face_area", faces), Areas("face_centroid", faces), View<double*>("cell_volume", n_),
                     tet_mesh::FaceRows<Memory>("face_rows", float_rows ? faces : 0)};
        cells_ = {View<int*>("cell_face_start", n_ + 1), View<int*>("cell_face", 0), View<int*>("cell_face_cursor", n_),
                  geometry_.volume, Areas("cell_centroid", n_)};
        lap(timer, timings, &Timings::build);
        report.faces = geometry(timings);
        return report;
    }

    // Local rebuild by clipping: the cells of the particles `cells` at positions x, independent of the persistent mesh.
    // Each face of a listed cell is returned once: by the lower id if both are listed, else by the listed one (Cells:
    // owned faces of each listed particle, neighbour ids, area vectors, centroids relative to it).
    // Used by: callers such as luna-cpp; not part of update().
    template <class X>
    const shell_voronoi::Cells<Memory>& clip(const X& x, const View<const int*>& cells) {
        if (x.extent(0) == 0) throw std::invalid_argument("voronoi::Mesh::clip: no particles");
        n_ = n_ > 0 ? n_ : static_cast<int>(x.extent(0));
        if (static_cast<int>(x.extent(0)) != n_) throw std::invalid_argument("voronoi::Mesh::clip: other particles");
        snap(x);
        index(positions_);
        const int count = static_cast<int>(cells.extent(0));
        View<int*> slots("clip_slots", count);
        View<char*> listed("clip_listed", n_);
        View<int*> slot_of(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("clip_slot_of")), n_);
        const auto order = list_.order;
        Kokkos::parallel_for(
            "voronoi::slot_of", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int q) { slot_of(order(q)) = q; });
        Kokkos::parallel_for(
            "voronoi::listed", Kokkos::RangePolicy<ExecSpace>(exec_, 0, count), KOKKOS_LAMBDA(int i) {
                slots(i) = slot_of(cells(i));
                listed(cells(i)) = 1;
            });
        local_ = shell_voronoi::Cells<Memory>(n_, false, 10, count);
        // Exact cells: nothing checks a rounded one here (#30).
        shell_voronoi::construct(exec_, shell_voronoi::make_grid(list_), local_, {slots, listed}, true);
        return local_;
    }

    // Scratch memory the caller may use between updates: the spare ring pool, grown to `bytes` if smaller.
    using Scratch = Kokkos::View<char*, Memory, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Scratch scratch(std::size_t bytes) {
        auto& spare = rings_.pool[1 - rings_.in_use];
        if (spare.extent(0) * sizeof(int) < bytes)
            Kokkos::realloc(Kokkos::WithoutInitializing, spare, (bytes + bytes / 64) / sizeof(int) + 1);
        return Scratch(reinterpret_cast<char*>(spare.data()), bytes);
    }

    // The per-cell face lists, and volumes and centroids summed over them in fixed order (bit-reproducible), for the
    // faces of the last update; does nothing if they are current. Incremental when few entries were touched.
    // Used by: callers after update(); runs tet_mesh::cells_incremental, cells, gradient_weights.
    void update_cells(Timings* timings = nullptr) {
        if (cells_current_) return;
        if (!built()) throw std::logic_error("voronoi::Mesh::update_cells: no mesh (no update yet, or it was refused)");
        Kokkos::Timer timer;
        // Its temporaries live in the spare ring pool (see scratch).
        const auto& spare = rings_.pool[1 - rings_.in_use];
        const std::size_t table = sizeof(int) * (static_cast<std::size_t>(rings_.renumbered) + rings_.dirty_count);
        if (lists_of_last_update_ && rings_.renumber_valid && state_.touched_entries >= 0 &&
            state_.touched_entries <= n_ / 5 &&
            table + tet_mesh::incremental_scratch_bytes(n_, rings_.renumbered, faces()) <= spare.extent(0) * sizeof(int)) {
            tet_mesh::cells_incremental(exec_, positions_, box_, owner(), neighbour(), tets_, state_.alive,
                                        Kokkos::subview(state_.touched_list, std::make_pair(0, state_.touched_entries)),
                                        rings_, geometry_, cells_, reinterpret_cast<char*>(spare.data()));
        } else {
            tet_mesh::cells(exec_, positions_, box_, owner(), neighbour(), geometry_, cells_,
                            scratch(tet_mesh::cells_scratch_bytes(faces())).data());
        }
        updates_since_lists_ = 0;
        if constexpr (with_weights) {
            // As many rows as list entries (2 per face), with the list's 1/32 spare.
            const std::size_t entries = 2 * static_cast<std::size_t>(faces());
            if (weights_.extent(0) < entries)
                weights_ = Kokkos::View<WeightReal* [3], Kokkos::LayoutRight, Memory>(
                    Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("cell_face_weights")), entries + entries / 32);
            tet_mesh::gradient_weights(exec_, positions_, box_, owner(), neighbour(), geometry_, cells_, weights_);
        }
        cells_current_ = true;
        lap(timer, timings, &Timings::cells);
    }

    bool cells_current() const { return cells_current_; }
    void set_fallback(Fallback f) { fallback_ = f; }
    // The most rounds of the device repair in one update; for tests.
    void set_repair_rounds(int rounds) { repair_rounds_ = rounds; }

    int particles() const { return n_; }
    // The box; box() is the side of a cubic one.
    const ep::Box& sides() const { return box_; }
    double box() const {
        if (!box_.cubic()) throw std::logic_error("voronoi::Mesh::box: the box is not a cube; use sides()");
        return box_.side[0];
    }
    // Faces f < faces(). owner() and neighbour() have exactly faces() rows;
    // area() and face_centroid() may hold more (spare capacity): loop to
    // faces(), never to their extent.
    int faces() const { return static_cast<int>(tets_.owner.extent(0)); }
    View<const int*> owner() const { return tets_.owner; }
    View<const int*> neighbour() const { return tets_.neighbour; }
    Rows area() const { return geometry_.area; }
    Rows face_centroid() const { return geometry_.centroid; }
    // FaceReal = float: the float rows of the faces (tet_mesh::FaceRows); empty otherwise.
    FaceRows face_rows() const { return geometry_.rows; }
    // Cell volumes (see update_cells); after update_cells() also the
    // centroids (relative to the generator) and the faces of cell i,
    // cell_face(cell_face_start(i)) .. cell_face(cell_face_start(i + 1) - 1).
    View<const double*> volume() const { return geometry_.volume; }
    Rows centroid() const { return cells_.centroid; }
    View<const int*> cell_face_start() const { return cells_.start; }
    View<const int*> cell_face() const { return cells_.face; }
    // Weight != void, after update_cells(): the gradient weights of list
    // entry k (tet_mesh::gradient_weights), row k for cell_face(k).
    Weights gradient_weights() const { return weights_; }
    // The snapped positions the geometry is computed from (padded to 32 bytes).
    tet_mesh::Points4<Memory> positions() const { return positions_; }

    // Internals, for tests and the benchmark.
    const tet_mesh::Mesh<Memory>& tetrahedra() const { return tets_; }
    const tet_repair::State<Memory>& repair_state() const { return state_; }

private:
    bool built() const { return state_.alive.extent(0) > 0; }

    // No mesh any more: the next update() builds from scratch.
    // Used by: rebuild (start and on refusal).
    void forget() {
        exec_.fence();
        tets_ = tet_mesh::Mesh<Memory>{};
        state_ = tet_repair::State<Memory>{};
        rings_ = tet_mesh::FaceRings<Memory>{};
        geometry_ = tet_mesh::Geometry<Memory>{};
        cells_ = tet_mesh::CellFaces<Memory>{};
        weights_ = {};
        n_ = 0;
        cells_current_ = false;
        updates_since_lists_ = 1;
        lists_of_last_update_ = false;
    }

    // With timings: fence, add the time since `timer` started to the phase `field`, restart the timer.
    // Used by: update, rebuild, update_cells, geometry.
    void lap(Kokkos::Timer& timer, Timings* timings, double Timings::*field) const {
        if (!timings) return;
        exec_.fence();
        timings->*field += timer.seconds() * 1e3;
        timer.reset();
    }

    // Stage 1: positions_ = x snapped to the grid (reallocated if the count changed).
    // Used by: update, rebuild, clip.
    template <class X>
    void snap(const X& x) {
        if (static_cast<int>(positions_.extent(0)) != n_)
            positions_ = tet_mesh::Points4<Memory>("positions", n_);  // zeroed once: the 4th column is padding (#47)
        detail::snap_positions(exec_, x, box_, positions_);
    }

    // Stage 2: the cell list (spatial grid) of positions p, for the neighbour search of the clipping.
    // Used by: rebuild, clip.
    template <class P>
    void index(const P& p) {
        if (list_.n != n_) neighbor_search::configure(list_, n_, box_, box_.cubic() ? box_.side[0] * std::cbrt(1.0 / n_)
                                                                                   : std::cbrt(box_.volume() / n_));
        neighbor_search::build(exec_, list_, p);
    }

    // Stage 5: circumcentres of the created tetrahedra, face rings, face areas/centroids and cell volumes (tet_mesh).
    // Returns the face count. `created` = entries touched by the repair, or empty for all (first build).
    // Used by: update, rebuild.
    int geometry(Timings* timings) {
        Kokkos::Timer timer;
        lists_of_last_update_ = updates_since_lists_ == 0;
        ++updates_since_lists_;
        const Kokkos::View<const int*, Memory> created =
            state_.touched_entries < 0 ? Kokkos::View<const int*, Memory>()
                                       : Kokkos::subview(state_.touched_list, std::make_pair(0, state_.touched_entries));
        if (rings_.faces < 0)
            tet_mesh::circumcentres(exec_, positions_, box_, tets_, state_.count);
        else
            tet_mesh::circumcentres(exec_, positions_, box_, tets_, state_.count, created);
        const int faces = tet_mesh::update_face_rings(exec_, tets_, state_.count, state_.touched, created, rings_);
        if (faces > static_cast<int>(geometry_.area.extent(0))) {
            Kokkos::realloc(Kokkos::WithoutInitializing, geometry_.area, faces + faces / 32);
            Kokkos::realloc(Kokkos::WithoutInitializing, geometry_.centroid, faces + faces / 32);
            if (float_rows) Kokkos::realloc(Kokkos::WithoutInitializing, geometry_.rows, faces + faces / 32);
        }
        tet_mesh::faces(exec_, positions_, box_, tets_, geometry_);
        total_volume_ = detail::total_volume(exec_, geometry_.volume);
        Kokkos::deep_copy(exec_, state_.touched, 0);
        cells_current_ = false;
        lap(timer, timings, &Timings::geometry);
        return faces;
    }

    ExecSpace exec_;
    ep::Box box_;
    int n_ = 0;
    long long ring_entries_ = 0;  // ring pool size of the last start
    Fallback fallback_ = Fallback::cheapest;
    int repair_rounds_ = 1000;
    double total_volume_ = 0;  // of the cells, from the last geometry()
    bool cells_current_ = false;
    int updates_since_lists_ = 1;       // updates since the last update_cells()
    bool lists_of_last_update_ = false;  // the cell lists have the numbers of the update before the last
    tet_mesh::Points4<Memory> positions_;
    neighbor_search::CellList<ExecSpace> list_;
    tet_mesh::Mesh<Memory> tets_;
    tet_repair::State<Memory> state_;
    tet_mesh::FaceRings<Memory> rings_;
    tet_mesh::Geometry<Memory> geometry_;
    tet_mesh::CellFaces<Memory> cells_;
    shell_voronoi::Cells<Memory> local_;
    Kokkos::View<WeightReal* [3], Kokkos::LayoutRight, Memory> weights_;
};

}  // namespace voronoi
#include "sorted_mesh.hpp"  // Mesh<1>
#include "voronoi_mesh_2d.hpp"  // Mesh<2>
