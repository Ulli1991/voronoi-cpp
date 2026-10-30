#pragma once

#include "exact_predicates.hpp"
#include "neighbor_search.hpp"
#include "shell_voronoi_warp.hpp"
#include "tet_flips.hpp"
#include "tet_mesh.hpp"
#include "tet_repair.hpp"
#include "tet_start.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

// The Voronoi mesh of particles in a periodic box, kept exact as they move:
// the interface of the hydro code to this library.
//
// Particles are named by their index in the caller's position array (the
// particle id); every output uses these ids, and every tie between
// cospherical particles is broken by them (symbolic perturbation), at the
// start and in the repair alike. Positions are snapped to the 2^30 grid of
// the exact predicates inside; topology and geometry both come from the
// snapped positions, so they always agree.
//
// Faces are stored once, by the particle with the lower id (the owner),
// with the area vector pointing from owner to neighbour and the centroid
// relative to the owner. update() gives faces and cell volumes (summed with
// atomics: the last bits may differ run to run). update_cells() adds the
// per-cell face lists, each cell's faces in a fixed order (increasing id of
// the other particle), and sums volumes and centroids in that order, so
// they are reproducible bit for bit.
// Faces are numbered by the tetrahedra that represent them, which follow
// the particles' order at the start: faces then come nearly sorted by owner
// when the caller's particle order is spatial (Morton, say).
//
// Hidden: the cell list, the clipping, the tetrahedra and their repair, the
// face rings. Room is left for later work: ghost (not owned) particles of a
// multi-GPU run would be particles like the others, with ids above the
// owned ones; repair limited to an active region would take the `active`
// list of update(); neither is implemented.
namespace voronoi {

namespace ep = exact_predicates;

namespace detail {
// positions = x snapped to the grid, in box units.
template <class ExecSpace, class X>
void snap_positions(const ExecSpace& exec, const X& x, double box,
                    const tet_mesh::Points4<typename ExecSpace::memory_space>& positions) {
    const double unit = box / ep::scale;
    Kokkos::parallel_for(
        "voronoi::snap", Kokkos::RangePolicy<ExecSpace>(exec, 0, positions.extent(0)), KOKKOS_LAMBDA(int i) {
            for (int a = 0; a < 3; ++a) positions(i, a) = ep::snap(x(i, a), box) * unit;
        });
}
}  // namespace detail

// What an update did.
struct Report {
    bool rebuilt = false;          // built from scratch (first update, rebuild(), or the guard)
    tet_start::Stats start;        // of that build
    long long flips = 0;           // device flips
    int rounds = 0;                // device repair rounds
    int stuck = 0;                 // triangles the device repair could not repair
    long long host_flips = 0;      // flips and local rebuilds of the host fallback
    long long host_rebuilds = 0;
    int faces = 0;
};

// Phase times of an update, in ms, for the benchmark: given to update(),
// the phases are fenced and timed.
struct Timings {
    double repair = 0, fallback = 0, geometry = 0, build = 0, cells = 0;
};

// What update() does when the device repair leaves triangles it cannot
// repair: repair the region around them on the host (tet_flips), build the
// mesh again from scratch on the device, or (cheapest) choose by the
// number of stuck entries. Measured on an A100 at 2.1M cells: the host
// takes about 2-5 ms for a handful of stuck triangles and 0.12 ms per
// stuck entry beyond (20.8 s for the 178k of a wave lattice's first step),
// a rebuild 0.17-0.25 s; so rebuild when more than n / 1000 are stuck.
enum class Fallback { cheapest, host, rebuild };

template <class ExecSpace = Kokkos::DefaultExecutionSpace>
class Mesh {
public:
    using Memory = typename ExecSpace::memory_space;
    template <class T>
    using View = Kokkos::View<T, Memory>;

    Mesh(const ExecSpace& exec, double box) : exec_(exec), box_(box) {}

    // New positions of all particles, x(i, a) in [0, box) for particle i
    // (any view on the device). Repairs the mesh, or builds it on the first
    // call or when the number of particles changed; exact on return either
    // way. `active` (particle ids) is accepted for a later repair limited
    // to an active region; today every particle is repaired.
    template <class X>
    Report update(const X& x, const View<const int*>& active = {}, Timings* timings = nullptr) {
        (void)active;
        if (!built() || static_cast<int>(x.extent(0)) != n_) return rebuild(x, timings);
        Report report;
        Kokkos::Timer timer;
        auto phase = [&](double& ms) {
            if (!timings) return;
            exec_.fence();
            ms += timer.seconds() * 1e3;
            timer.reset();
        };
        snap(x);
        tet_repair::move_screened(exec_, positions_, box_, tets_, state_);
        const tet_repair::Stats device = tet_repair::repair(exec_, tets_, state_);
        report.flips = device.flips23 + device.flips32;
        report.rounds = static_cast<int>(device.rounds);
        report.stuck = device.stuck;
        phase(timings ? timings->repair : unused_);
        if (device.stuck > 0) {
            if (fallback_ == Fallback::rebuild ||
                (fallback_ == Fallback::cheapest && static_cast<long long>(device.stuck) * 1000 > n_)) {
                Report rebuilt = rebuild(x, timings);
                rebuilt.flips = report.flips;
                rebuilt.rounds = report.rounds;
                rebuilt.stuck = report.stuck;
                return rebuilt;
            }
            const tet_flips::Stats host = tet_flips::finish(exec_, tets_, state_);
            report.host_flips = host.flips23 + host.flips32;
            report.host_rebuilds = host.rebuilt;
            phase(timings ? timings->fallback : unused_);
        }
        report.faces = geometry(timings);
        return report;
    }

    // The exact mesh of positions x from scratch: clipping with face rings,
    // the exact start (tet_start), then faces and cells.
    template <class X>
    Report rebuild(const X& x, Timings* timings = nullptr) {
        Kokkos::Timer timer;
        Report report;
        report.rebuilt = true;
        n_ = static_cast<int>(x.extent(0));
        snap(x);
        {
            index(positions_);
            auto cells = shell_voronoi::Cells<Memory>::with_rings(n_, ring_entries_ > 0 ? ring_entries_ : 64LL * n_);
            const auto grid = shell_voronoi::make_grid(list_);
            shell_voronoi::construct_warp(exec_, grid, cells);
            report.start = tet_start::build(exec_, grid, cells, tets_);
            if (!report.start.exact && cells.needed() > cells.capacity()) {  // the patches did not fit
                cells = shell_voronoi::Cells<Memory>::with_rings(n_, cells.needed() + cells.needed() / 16);
                shell_voronoi::construct_warp(exec_, grid, cells);
                report.start = tet_start::build(exec_, grid, cells, tets_);
            }
            ring_entries_ = cells.capacity();
            if (!report.start.exact) throw std::runtime_error("voronoi::Mesh: the start is not exact");
        }
        // Capacities: 1/8 more than the starting mesh needs (ring pools
        // 1/2 more: room to append between compactions); the repair and
        // the face rings grow them when a step needs more.
        const int tets = static_cast<int>(tets_.vertex.extent(0));
        const int faces = static_cast<int>(tets_.owner.extent(0)) + static_cast<int>(tets_.owner.extent(0)) / 8;
        const int entries = static_cast<int>(tets_.ring.extent(0));
        tets_.owner = tets_.neighbour = tets_.ring_start = tets_.ring_length = tets_.ring = {};  // walked again below
        state_ = tet_repair::prepare(exec_, positions_, box_, tets_, tets + tets / 8, n_ / 4 + 1024);
        rings_ = tet_mesh::make_face_rings<Memory>(tets + tets / 8, faces, entries + entries / 2);
        geometry_ = {View<double* [3]>("face_area", faces), View<double* [3]>("face_centroid", faces),
                     View<double*>("cell_volume", n_)};
        cells_ = {View<int*>("cell_face_start", n_ + 1), View<int*>("cell_face", 0), View<int*>("cell_face_cursor", n_),
                  geometry_.volume, View<double* [3]>("cell_centroid", n_)};
        if (timings) {
            exec_.fence();
            timings->build += timer.seconds() * 1e3;
        }
        report.faces = geometry(timings);
        return report;
    }

    // Local rebuild by clipping: the cells of the particles `cells` at
    // positions x, independent of the persistent mesh. Each face of a
    // listed cell is returned once: by the lower id when both cells are
    // listed, else by the listed one (Cells: owned faces of each listed
    // particle, neighbour ids, area vectors, centroids relative to it).
    template <class X>
    const shell_voronoi::Cells<Memory>& clip(const X& x, const View<const int*>& cells) {
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
        shell_voronoi::construct_warp(exec_, shell_voronoi::make_grid(list_), local_, {slots, listed});
        return local_;
    }

    // Scratch memory the caller may use between updates: the spare face-ring
    // pool, grown to `bytes` if smaller (the growth is kept). Valid only until
    // the next update(), rebuild() or update_cells(), which reuse the pool and
    // may move it: ask again after each of them, and keep nothing in it
    // across them (clip() does not touch it).
    using Scratch = Kokkos::View<char*, Memory, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Scratch scratch(std::size_t bytes) {
        auto& spare = rings_.pool[1 - rings_.in_use];
        if (spare.extent(0) * sizeof(int) < bytes)
            Kokkos::realloc(Kokkos::WithoutInitializing, spare, (bytes + sizeof(int) - 1) / sizeof(int));
        return Scratch(reinterpret_cast<char*>(spare.data()), bytes);
    }

    // The per-cell face lists, and volumes and centroids summed over them in
    // fixed order (bit-reproducible), for the faces of the last update; does
    // nothing if they are current.
    void update_cells(Timings* timings = nullptr) {
        if (cells_current_) return;
        Kokkos::Timer timer;
        // Its temporaries live in the spare ring pool (see scratch).
        tet_mesh::cells(exec_, positions_, box_, tets_, geometry_, cells_,
                        scratch(tet_mesh::cells_scratch_bytes(faces())).data());
        cells_current_ = true;
        if (timings) {
            exec_.fence();
            timings->cells += timer.seconds() * 1e3;
        }
    }

    void set_fallback(Fallback f) { fallback_ = f; }

    int particles() const { return n_; }
    double box() const { return box_; }
    // Faces f < faces(). owner() and neighbour() have exactly faces() rows;
    // area() and face_centroid() may hold more (spare capacity): loop to
    // faces(), never to their extent.
    int faces() const { return static_cast<int>(tets_.owner.extent(0)); }
    View<const int*> owner() const { return tets_.owner; }
    View<const int*> neighbour() const { return tets_.neighbour; }
    View<const double* [3]> area() const { return geometry_.area; }
    View<const double* [3]> face_centroid() const { return geometry_.centroid; }
    // Cell volumes (see update_cells); after update_cells() also the
    // centroids (relative to the generator) and the faces of cell i,
    // cell_face(cell_face_start(i)) .. cell_face(cell_face_start(i + 1) - 1).
    View<const double*> volume() const { return geometry_.volume; }
    View<const double* [3]> centroid() const { return cells_.centroid; }
    View<const int*> cell_face_start() const { return cells_.start; }
    View<const int*> cell_face() const { return cells_.face; }
    // The snapped positions the geometry is computed from (padded to 32 bytes).
    tet_mesh::Points4<Memory> positions() const { return positions_; }

    // Internals, for tests and the benchmark.
    const tet_mesh::Mesh<Memory>& tetrahedra() const { return tets_; }
    const tet_repair::State<Memory>& repair_state() const { return state_; }

private:
    bool built() const { return state_.alive.extent(0) > 0; }

    // positions_ = x snapped to the grid, in box units.
    template <class X>
    void snap(const X& x) {
        if (static_cast<int>(positions_.extent(0)) != n_)
            positions_ = tet_mesh::Points4<Memory>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("positions")), n_);
        detail::snap_positions(exec_, x, box_, positions_);
    }

    // The cell list of positions p.
    template <class P>
    void index(const P& p) {
        if (list_.n != n_) neighbor_search::configure(list_, n_, box_, box_ * std::cbrt(1.0 / n_));
        neighbor_search::build(exec_, list_, p);
    }

    // Circumcentres of the created tetrahedra, face rings, faces and cells.
    int geometry(Timings* timings) {
        Kokkos::Timer timer;
        const Kokkos::View<const int*, Memory> created =
            state_.touched_entries < 0 ? Kokkos::View<const int*, Memory>()
                                       : Kokkos::subview(state_.touched_list, std::make_pair(0, state_.touched_entries));
        if (rings_.faces < 0)
            tet_mesh::circumcentres(exec_, positions_, box_, tets_, state_.count);  // a new mesh: every centre
        else
            tet_mesh::circumcentres(exec_, positions_, box_, tets_, state_.count, created);
        const int faces = tet_mesh::update_face_rings(exec_, tets_, state_.count, state_.touched, created, rings_);
        if (faces > static_cast<int>(geometry_.area.extent(0))) {
            Kokkos::realloc(Kokkos::WithoutInitializing, geometry_.area, faces + faces / 8);
            Kokkos::realloc(Kokkos::WithoutInitializing, geometry_.centroid, faces + faces / 8);
        }
        tet_mesh::faces(exec_, positions_, box_, tets_, geometry_);
        Kokkos::deep_copy(exec_, state_.touched, 0);
        cells_current_ = false;
        if (timings) {
            exec_.fence();
            timings->geometry += timer.seconds() * 1e3;
        }
        return faces;
    }

    ExecSpace exec_;
    double box_;
    int n_ = 0;
    long long ring_entries_ = 0;  // ring pool size of the last start
    Fallback fallback_ = Fallback::cheapest;
    bool cells_current_ = false;
    double unused_ = 0;
    tet_mesh::Points4<Memory> positions_;
    neighbor_search::CellList<ExecSpace> list_;
    tet_mesh::Mesh<Memory> tets_;
    tet_repair::State<Memory> state_;
    tet_mesh::FaceRings<Memory> rings_;
    tet_mesh::Geometry<Memory> geometry_;
    tet_mesh::CellFaces<Memory> cells_;
    shell_voronoi::Cells<Memory> local_;
};

}  // namespace voronoi
