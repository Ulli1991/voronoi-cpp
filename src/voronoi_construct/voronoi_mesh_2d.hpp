#pragma once

#include "tri_finish.hpp"
#include "tri_repair.hpp"

// The 2D Voronoi mesh: Mesh<3>'s contract on the engine of tri_mesh.hpp and
// tri_repair.hpp. Faces are numbered for good at the start: there are always
// exactly 3n, and a flip gives its face new particles. volume() is current
// after update_cells() only (no atomic sums in update()). As Mesh<3>: particles
// on one grid position are refused by name, a refused update leaves no mesh and
// the next builds, no particles are an invalid_argument that leaves the object
// as it was, and a repair whose areas do not fill the box once is rebuilt.
// Hidden: the cell list, the rings, the triangles.
namespace voronoi {

template <class ExecSpace, class FaceReal, class Weight>
class Mesh<2, ExecSpace, FaceReal, Weight> {
public:
    static_assert(std::is_same_v<FaceReal, double> || std::is_same_v<FaceReal, float>, "FaceReal: double or float");
    static_assert(std::is_void_v<Weight> || std::is_same_v<Weight, double> || std::is_same_v<Weight, float>,
                  "Weight: void, double or float");
    static constexpr int dimensions = 2;
    static constexpr bool float_rows = std::is_same_v<FaceReal, float>;
    static constexpr bool with_weights = !std::is_void_v<Weight>;
    using WeightReal = std::conditional_t<with_weights, Weight, float>;
    using Memory = typename ExecSpace::memory_space;
    template <class T>
    using View = Kokkos::View<T, Memory>;
    using Box = ep::BoxN<2>;
    using Positions = tri_mesh::Rows2<Memory>;
    using Rows = Kokkos::View<const double* [2], Kokkos::LayoutRight, Memory>;
    using FaceRows = Kokkos::View<const float* [8], Kokkos::LayoutRight, Memory>;
    using Weights = Kokkos::View<const WeightReal* [2], Kokkos::LayoutRight, Memory>;

    Mesh(const ExecSpace& exec, const Box& box) : exec_(exec), box_(box), point_("point2", 0, box) {}
    Mesh(const ExecSpace& exec, double side) : Mesh(exec, Box(side)) {}

    // As Mesh<3>::update: new positions x(i, a) in [0, side[a]), repaired,
    // or built on the first call or when the number of particles changed;
    // exact on return, or a std::runtime_error saying what is wrong with the
    // positions, and then there is no mesh and the next update() builds.
    template <class X>
    Report update(const X& x, const View<const int*>& active = {}, Timings* timings = nullptr) {
        (void)active;
        if (!built() || static_cast<int>(x.extent(0)) != n_) return rebuild(x, timings);
        Report report;
        Kokkos::Timer timer;
        snap(x);
        tri_repair::screen(exec_, point_, box_, mesh_, state_);
        const tri_repair::Stats device = tri_repair::repair(exec_, point_, mesh_, state_);
        report.flips = device.flips;
        report.rounds = device.rounds;
        report.stuck = device.stuck;
        lap(timer, timings, &Timings::repair);
        int stuck = device.stuck;
        if (stuck > 0 && fallback_ != Fallback::rebuild &&
            (fallback_ == Fallback::host || static_cast<long long>(stuck) * host_limit <= n_)) {
            for (int pass = 0; pass < 4 && stuck > 0; ++pass) {
                tri_finish::Stats host;
                try {
                    host = tri_finish::finish(exec_, point_, mesh_, state_);
                } catch (const std::exception&) {  // out of local room: the rebuild decides
                    break;
                }
                report.host_flips += host.triangles;
                report.host_rebuilds += host.patches;
                patched_ = true;
                if (!host.done && host.patches == 0) break;
                const tri_repair::Stats again = tri_repair::repair(exec_, point_, mesh_, state_);
                report.flips += again.flips;
                report.rounds += again.rounds;
                stuck = again.stuck;
            }
            lap(timer, timings, &Timings::fallback);
        }
        if (stuck == 0) {
            report.overlapped = !fills_box();
            if (!report.overlapped) {
                report.faces = geometry(timings);
                return report;
            }
        }
        Report rebuilt = rebuild(x, timings);
        rebuilt.flips = report.flips;
        rebuilt.rounds = report.rounds;
        rebuilt.stuck = report.stuck;
        rebuilt.host_flips = report.host_flips;
        rebuilt.host_rebuilds = report.host_rebuilds;
        rebuilt.overlapped = report.overlapped;
        return rebuilt;
    }

    // The exact mesh of positions x from scratch: clipping, then the
    // triangles of the rings (screened and repaired: nothing to do when the
    // clipping is right; both are counted in the report).
    template <class X>
    Report rebuild(const X& x, Timings* timings = nullptr) {
        if (x.extent(0) == 0) throw std::invalid_argument("voronoi::Mesh<2>: no particles");
        Kokkos::Timer timer;
        Report report;
        report.rebuilt = true;
        const int particles = static_cast<int>(x.extent(0));
        forget();
        n_ = particles;
        snap(x);
        {
            index();
            if (stencil_.offsets.extent(0) == 0) stencil_ = polygon_clip::make_stencil(exec_);
            polygon_clip::Rings<Memory> rings;
            try {
                rings = polygon_clip::construct(exec_, polygon_clip::make_grid(list_), stencil_);
            } catch (const std::runtime_error& e) {  // also two particles on one position: no bisector
                refuse(e.what());
            }
            const long long uncertified = polygon_clip::count_status(exec_, rings, polygon_clip::uncertified);
            if (uncertified > 0)
                refuse("voronoi::Mesh<2>: " + std::to_string(uncertified) +
                       " cells reach too far to be certified in one periodic image (a cell's radius must "
                       "stay below about 3/16 of the shortest box side): too few particles across the box");
            try {
                mesh_ = tri_mesh::from_rings(exec_, rings);
            } catch (const std::runtime_error& e) {
                refuse(e.what());
            }
        }
        const int triangles = mesh_.count(), faces = 3 * n_;
        state_ = tri_repair::State<Memory>(triangles, n_ / 4 + 1024);
        lists_valid_ = false;
        using Areas = typename tet_mesh::Geometry<Memory, 2>::Areas;
        geometry_ = {Areas(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("face_area")), faces),
                     Areas(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("face_centroid")), faces),
                     View<double*>("cell_volume", n_),
                     tet_mesh::FaceRows<Memory>("face_rows", float_rows ? faces : 0)};
        cells_ = {View<int*>("cell_face_start", n_ + 1), View<int*>("cell_face", 0), View<int*>("cell_face_cursor", n_),
                  geometry_.volume, Areas("cell_centroid", n_)};
        lap(timer, timings, &Timings::build);
        tri_repair::screen(exec_, point_, box_, mesh_, state_);
        const tri_repair::Stats device = tri_repair::repair(exec_, point_, mesh_, state_);
        report.flips = device.flips;
        report.rounds = device.rounds;
        report.stuck = device.stuck;
        if (device.stuck > 0) refuse("voronoi::Mesh<2>: inverted triangles after a build");
        if (!fills_box()) refuse("voronoi::Mesh<2>: the triangles of a build do not fill the box once");
        report.faces = geometry(timings);
        return report;
    }

    // The clipped rings of the particles `cells` at positions x, independent
    // of the persistent mesh; ring-status uncertified for the others.
    template <class X>
    polygon_clip::Rings<Memory> clip(const X& x, const View<const int*>& cells) {
        if (x.extent(0) == 0) throw std::invalid_argument("voronoi::Mesh::clip: no particles");
        n_ = n_ > 0 ? n_ : static_cast<int>(x.extent(0));
        if (static_cast<int>(x.extent(0)) != n_) throw std::invalid_argument("voronoi::Mesh::clip: other particles");
        snap(x);
        index();
        if (stencil_.offsets.extent(0) == 0) stencil_ = polygon_clip::make_stencil(exec_);
        return polygon_clip::construct(exec_, polygon_clip::make_grid(list_), stencil_, slots_of(cells));
    }

    // Scratch memory the caller may use between updates, grown to `bytes`:
    // valid until the next update_cells(), which uses it too.
    using Scratch = Kokkos::View<char*, Memory, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Scratch scratch(std::size_t bytes) {
        if (scratch_.extent(0) < bytes)
            scratch_ = View<char*>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("mesh2_scratch")), bytes + bytes / 64);
        return Scratch(scratch_.data(), bytes);
    }

    // The per-cell face lists, and volumes and centroids summed over them in
    // fixed order (bit-reproducible); does nothing if they are current.
    void update_cells(Timings* timings = nullptr) {
        if (cells_current_) return;
        if (!built()) throw std::logic_error("voronoi::Mesh<2>::update_cells: no mesh (no update yet, or it was refused)");
        Kokkos::Timer timer;
        // As Mesh<3>: from the last step's lists up to one triangle created per
        // five cells (tet_mesh::finish_incremental). Face numbers stay, so the
        // table is the identity, but for the faces that flipped: they have
        // no old number, and each is a new face.
        const bool same_lists = lists_valid_ && updates_since_lists_ == 1 && !patched_;
        if (same_lists && state_.touched_count == 0) {
            tet_mesh::detail::sort_and_sum<2>(exec_, positions_, box_, geometry_, cells_, {}, {}, {}, owner(), neighbour(),
                                              View<int*>(), Kokkos::View<const char*, Memory, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(),
                                              true, false);
        } else if (same_lists && state_.touched_count <= n_ / 5) {
            incremental_lists();
        } else {
            tet_mesh::cells(exec_, positions_, box_, owner(), neighbour(), geometry_, cells_,
                            scratch(tet_mesh::cells_scratch_bytes(faces())).data());
        }
        lists_valid_ = true;
        patched_ = false;
        updates_since_lists_ = 0;
        if constexpr (with_weights) {
            const std::size_t entries = 2 * static_cast<std::size_t>(faces());
            if (weights_.extent(0) < entries)
                weights_ = Kokkos::View<WeightReal* [2], Kokkos::LayoutRight, Memory>(
                    Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("cell_face_weights")), entries);
            tet_mesh::gradient_weights(exec_, positions_, box_, owner(), neighbour(), geometry_, cells_, weights_);
        }
        cells_current_ = true;
        lap(timer, timings, &Timings::cells);
    }

    bool cells_current() const { return cells_current_; }
    void set_fallback(Fallback f) { fallback_ = f; }

    int particles() const { return n_; }
    const Box& sides() const { return box_; }
    double box() const {
        if (!box_.cubic()) throw std::logic_error("voronoi::Mesh::box: the box is not a square; use sides()");
        return box_.side[0];
    }
    int faces() const { return static_cast<int>(mesh_.owner.extent(0)); }
    View<const int*> owner() const { return mesh_.owner; }
    View<const int*> neighbour() const { return mesh_.neighbour; }
    Rows area() const { return geometry_.area; }
    Rows face_centroid() const { return geometry_.centroid; }
    FaceRows face_rows() const { return geometry_.rows; }
    View<const double*> volume() const { return geometry_.volume; }
    Rows centroid() const { return cells_.centroid; }
    View<const int*> cell_face_start() const { return cells_.start; }
    View<const int*> cell_face() const { return cells_.face; }
    Weights gradient_weights() const { return weights_; }
    Positions positions() const { return positions_; }

    // Internals, for tests and the benchmark.
    const tri_mesh::Triangles<Memory>& triangles() const { return mesh_; }
    const ep::Points<Memory, 2>& points() const { return point_; }

    // Public only because CUDA device lambdas may not be defined in private
    // member functions.
    template <class X>
    void snap(const X& x) {
        if (static_cast<int>(positions_.extent(0)) != n_) {
            positions_ = Positions(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("positions")), n_);
            point_ = ep::Points<Memory, 2>("point2", n_, box_);
        }
        const auto positions = positions_;
        const auto point = point_;
        const Box box = box_;
        Kokkos::parallel_for(
            "voronoi2::snap", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                for (int a = 0; a < 2; ++a) {
                    const int k = ep::snap(x(i, a), box, a);
                    point(i, a) = k;
                    positions(i, a) = k * box.step;
                }
            });
    }
    void incremental_lists() {
        using Range = Kokkos::RangePolicy<ExecSpace>;
        if (renumber_.extent(0) == 0) {
            renumber_ = View<int*>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("face_renumber")), faces());
            const auto table = renumber_;
            Kokkos::parallel_for("voronoi2::identity", Range(exec_, 0, faces()), KOKKOS_LAMBDA(int f) { table(f) = f; });
        }
        const int count = state_.touched_count / 2, all = faces();
        const auto lists = tet_mesh::detail::begin_incremental<ExecSpace, 2>(
            exec_, reinterpret_cast<int*>(scratch(tet_mesh::incremental_scratch_bytes(n_, all, all)).data()), n_, all, all, cells_);
        const auto table = renumber_;
        const auto dirty = state_.flipped;  // ~f for the first entry of a face, f for the others
        const auto touched = state_.touched;
        const auto corners = mesh_.corners;
        const auto marked = lists.marked;
        Kokkos::parallel_for(
            "voronoi2::flipped", Range(exec_, 0, count), KOKKOS_LAMBDA(int s) {
                const int f = dirty(s);
                dirty(s) = Kokkos::atomic_exchange(&table(f), -1) == f ? ~f : f;
                for (int side = 0; side < 2; ++side) {
                    const int t = touched(2 * s + side);
                    for (int c = 0; c < 3; ++c) marked(corners(t, c)) = 1;
                }
            });
        tet_mesh::detail::finish_incremental<2>(exec_, positions_, box_, owner(), neighbour(), geometry_, cells_, View<const int*>(table),
                                                dirty, count, lists);
        Kokkos::parallel_for(
            "voronoi2::identity_again", Range(exec_, 0, count), KOKKOS_LAMBDA(int s) {
                const int f = dirty(s) < 0 ? ~dirty(s) : dirty(s);
                table(f) = f;
            });
    }
    View<const int*> slots_of(const View<const int*>& cells) {
        const int count = static_cast<int>(cells.extent(0));
        View<int*> slots("clip_slots", count), slot_of("clip_slot_of", n_);
        const auto order = list_.order;
        Kokkos::parallel_for(
            "voronoi2::slot_of", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int q) { slot_of(order(q)) = q; });
        Kokkos::parallel_for(
            "voronoi2::slots", Kokkos::RangePolicy<ExecSpace>(exec_, 0, count), KOKKOS_LAMBDA(int i) { slots(i) = slot_of(cells(i)); });
        return slots;
    }

private:
    bool built() const { return mesh_.corners.extent(0) > 0; }

    // No mesh any more: the next update() builds from scratch.
    void forget() {
        exec_.fence();  // kernels in flight may still read what goes here
        mesh_ = tri_mesh::Triangles<Memory>{};
        state_ = tri_repair::State<Memory>{};
        geometry_ = tet_mesh::Geometry<Memory, 2>{};
        cells_ = tet_mesh::CellFaces<Memory, 2>{};
        weights_ = {};
        renumber_ = {};
        n_ = 0;
        cells_current_ = false;
        lists_valid_ = false;
        patched_ = false;
        updates_since_lists_ = 0;
    }
    // The positions were refused: coincident particles by name, else `why`.
    [[noreturn]] void refuse(const std::string& why) {
        const std::string pair = detail::coincident(positions_);
        forget();
        throw std::runtime_error(pair.empty() ? why : pair);
    }
    // Whether the signed areas add up to the box once (tri_repair::total_area).
    bool fills_box() const {
        const double whole = 2.0 * static_cast<double>(box_.period[0]) * static_cast<double>(box_.period[1]);
        const double area = state_.area_stale || patched_ ? tri_repair::total_area(exec_, point_, box_, mesh_) : state_.area;
        return Kokkos::abs(area / whole - 1) < 1e-6;
    }

    void lap(Kokkos::Timer& timer, Timings* timings, double Timings::*field) const {
        if (!timings) return;
        exec_.fence();
        timings->*field += timer.seconds() * 1e3;
        timer.reset();
    }
    void index() {
        if (list_.n != n_) neighbor_search::configure(list_, n_, box_, std::sqrt(box_.volume() / n_));
        neighbor_search::build(exec_, list_, positions_);
    }
    // The centres of flipped triangles, then the faces.
    int geometry(Timings* timings) {
        Kokkos::Timer timer;
        tri_repair::centres(exec_, point_, box_, mesh_, state_);
        tri_mesh::faces(exec_, positions_, box_, mesh_, geometry_);
        cells_current_ = false;
        ++updates_since_lists_;
        lap(timer, timings, &Timings::geometry);
        return faces();
    }

    static constexpr long long host_limit = 4000;  // cheapest: on the host up to n / host_limit stuck triangles

    ExecSpace exec_;
    Box box_;
    int n_ = 0;
    Fallback fallback_ = Fallback::cheapest;
    bool cells_current_ = false;
    bool lists_valid_ = false;     // cells_ holds the lists of the last update_cells()
    bool patched_ = false;         // the host fallback changed faces the flip list does not know
    int updates_since_lists_ = 0;  // more than one: the flip list covers only the last
    View<int*> renumber_;          // the identity between steps (incremental_lists)
    Positions positions_;
    ep::Points<Memory, 2> point_;
    neighbor_search::CellList<ExecSpace, 2> list_;
    shell_voronoi::Stencil<Memory> stencil_;
    tri_mesh::Triangles<Memory> mesh_;
    tri_repair::State<Memory> state_;
    tet_mesh::Geometry<Memory, 2> geometry_;
    tet_mesh::CellFaces<Memory, 2> cells_;
    View<char*> scratch_;
    Kokkos::View<WeightReal* [2], Kokkos::LayoutRight, Memory> weights_;
};

}  // namespace voronoi
