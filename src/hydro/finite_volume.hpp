#pragma once

#include "euler.hpp"
#include "precision.hpp"
#include "voronoi_mesh.hpp"

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

// Finite volumes on a voronoi::Mesh<D> (D = 2, 3), static or moving: MUSCL-Hancock (order 1) or Godunov (order 0),
// HLLC or HLL on every face, one global time step; the state is the extensive Q = (mass, momentum, energy) per cell.
// Step order: time_step() (primitives, gradients, dt) -> step(dt) (face fluxes via euler.hpp, update Q, then mesh
// update if moving). Driven by bench/hydro_sedov.cpp and the hydro tests; Q is read out through snapshot.hpp.
namespace hydro {

// How the fluxes reach Q: one fused loop per cell (each face computed twice), or fluxes stored per face then summed.
// Used by: FiniteVolume::advance.
enum class Update { per_cell, per_face };

// The gradient limiter: at the face centroids, at the neighbours' centroids, or none.
// Used by: Sweep::operator().
enum class Limiter { face_centroid, cell_centroid, none };

// A static mesh, or one whose points move with the flow.
// Used by: FiniteVolume::step (moving: drift points, then Mesh::update).
enum class Motion { static_mesh, moving };

// Times of the phases of a step, in ms, summed over steps (fenced).
// primitives and sweep (point velocities, time step and gradients): in time_step(); fluxes: with the per-cell loop
// also the update. Used by: FiniteVolume::phase_end; printed by hydro_sedov.
struct Timings {
    double primitives = 0, sweep = 0, fluxes = 0, update = 0, drift = 0, mesh = 0, cells = 0;
};

// What the mesh updates of a moving run did, summed over steps.
// Used by: FiniteVolume::move; printed by hydro_sedov.
struct MeshStats {
    long long updates = 0, rebuilds = 0, stuck = 0, host_flips = 0, device_flips = 0;
};

// The solver on a mesh in D dimensions. Riemann: HLL or HLLC (euler.hpp); Precision: see precision.hpp.
// Used by: hydro_sedov (run), the hydro tests.
template <int D, class ExecSpace = Kokkos::DefaultExecutionSpace, class Riemann = HLLC,
          class Precision = hydro::Precision<>>
class FiniteVolume {
public:
    static_assert(D == 2 || D == 3, "hydro::FiniteVolume: D = 1 is finite_volume_1d.hpp");
    static constexpr int V = D + 2;  // variables
    using P = Precision;
    using Memory = typename ExecSpace::memory_space;
    using Mesh = voronoi::Mesh<D, ExecSpace, typename P::geometry, typename P::weight>;
    using State = hydro::State<D>;
    using Vec = hydro::Vec<D>;
    using G = typename P::gradient;            // gradient arithmetic
    using GStore = typename P::gradient_store;  // gradient storage
    using FluxStore = typename P::flux;
    static constexpr int gradient_width = row_width<GStore>(V * D);
    using Cells = Kokkos::View<double* [V], Kokkos::LayoutRight, Memory>;
    using Points = Kokkos::View<double* [D], Kokkos::LayoutRight, Memory>;
    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    // Per cell and step: W = primitive(Q / V), the sound speed, the radius.
    using Prims = Kokkos::View<double* [V + 2], Kokkos::LayoutRight, Memory, Unmanaged>;
    using Gradients = Kokkos::View<GStore* [gradient_width], Kokkos::LayoutRight, Memory, Unmanaged>;
    using Exponents = Kokkos::View<std::uint32_t*, Memory, Unmanaged>;  // FP16 gradients: one byte per group
    using FaceFluxes = Kokkos::View<FluxStore* [V], Kokkos::LayoutRight, Memory, Unmanaged>;

    // AREPO next generation's mesh regularisation as its Sedov examples configure it
    // (examples/sedov_3d/sedov_globaltstp_3d/Config.sh:11-13, param.txt:93-94). Used by: Sweep (moving).
    static constexpr double cell_shaping_speed = 0.5, cell_max_angle_factor = 2.25;

    // Allocates Q and (moving) the point arrays. mesh: built, with update_cells() done; for a static run its
    // geometry must not change while this object is used. Used by: hydro_sedov before the first time_step().
    FiniteVolume(const ExecSpace& exec, Mesh& mesh, const Gas& gas, int order, Update update,
                 Limiter limiter = Limiter::face_centroid, Motion motion = Motion::static_mesh)
        : exec_(exec), mesh_(&mesh), gas_(gas), order_(order), update_(update), limiter_(limiter), motion_(motion),
          n_(mesh.particles()) {
        if (order != 0 && order != 1) throw std::invalid_argument("hydro::FiniteVolume: order 0 or 1");
        q_ = Cells("hydro_Q", n_);
        if (order == 0 && update == Update::per_cell)
            next_ = Cells(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_Q_next")), n_);
        if (motion == Motion::moving) {
            x_ = Points(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_x")), n_);
            w_ = Points("hydro_w", n_);
            copy_positions();
        }
        refresh();
    }

    // The extensive conserved state of each cell, Q(i, v) = (mass, momentum, energy); the caller sets the
    // initial condition here before stepping and reads it for snapshots.
    Cells state() const { return q_; }
    // Moving: the positions of the mesh-generating points and their velocities.
    Points positions() const { return x_; }
    Points point_velocities() const { return w_; }
    int order() const { return order_; }
    Update update() const { return update_; }
    Motion motion() const { return motion_; }
    const Gas& gas() const { return gas_; }
    const MeshStats& mesh_stats() const { return mesh_stats_; }

    // The global time step: the smallest of AREPO-ng's cell time steps, without the power-of-two bins; the sweep before
    // it sets the point velocities (moving) and the gradients that step() uses.
    // Used by: the bench main loop, before each step().
    double time_step(double courant, Timings* timings = nullptr) {
        timer_.reset();
        refresh();
        borrow_scratch();
        primitives();
        phase_end(timings, &Timings::primitives);
        const double dt = sweep(courant);
        swept_ = true;
        phase_end(timings, &Timings::sweep);
        return dt;
    }

    // One step of length dt with the gradients and point velocities of the last time_step() (swept again if there
    // was none since the last step; the state must not change in between). Used by: the bench main loop.
    void step(double dt, Timings* timings = nullptr) {
        timer_.reset();
        refresh();
        if (!swept_) {
            borrow_scratch();
            primitives();
            phase_end(timings, &Timings::primitives);
            sweep(1.0);
            phase_end(timings, &Timings::sweep);
        }
        swept_ = false;
        if (order_ == 1)
            advance<1>(dt, timings);
        else
            advance<0>(dt, timings);
        if (motion_ == Motion::moving) move(dt, timings);
    }

    // Face sides of the last step whose extrapolated state was not physical and which took the cell's state instead.
    long long fallbacks() const { return fallbacks_; }

    // The radius of a sphere (D = 3) or circle (D = 2) of volume v (AREPO's get_cell_radius, voronoi.c:1053-1071).
    // Used by: primitives() (stored per cell for the time step).
    KOKKOS_INLINE_FUNCTION static double radius(double v) {
        if constexpr (D == 3)
            return Kokkos::cbrt(v * (3.0 / (4.0 * M_PI)));
        else
            return Kokkos::sqrt(v * (1.0 / M_PI));
    }
    // The size of a face of area (2D: length) |A| in AREPO's face angle. Used by: Sweep (moving regularisation).
    KOKKOS_INLINE_FUNCTION static double face_angle(double area) {
        if constexpr (D == 3)
            return Kokkos::sqrt(area / M_PI);
        else
            return 0.5 * area;
    }

    // Mesh arrays and the per-face and per-cell operations on them (float face rows are read back as double).
    // Copied by value into the kernels. Used by: Sweep, FaceStep, FaceFlux, advance.
    struct Geometry {
        static constexpr bool float_rows = Mesh::float_rows;
        Kokkos::View<const int*, Memory> owner, neighbour, start, face;
        typename Mesh::Rows area, face_centroid, centroid;
        typename Mesh::FaceRows rows;
        typename Mesh::Weights weights;
        Kokkos::View<const double*, Memory> volume;
        typename Mesh::Positions positions;
        exact_predicates::BoxN<D> box;

        template <class Rows>
        KOKKOS_INLINE_FUNCTION Vec row(const Rows& v, int i) const {
            return make_vec<D>([&](int a) { return v(i, a); });
        }
        // The conserved state of cell i per unit volume.
        KOKKOS_INLINE_FUNCTION State per_volume(const Cells& q, int i) const {
            const double inv = 1.0 / volume(i);
            return make_state<D>([&](int v) { return q(i, v) * inv; });
        }
        // Nearest image of generator j seen from generator i (periodic box).
        KOKKOS_INLINE_FUNCTION Vec separation(int i, int j) const {
            return make_vec<D>(
                [&](int a) { return tet_mesh::wrap(positions(j, a) - positions(i, a), box.side[a], box.inverse[a]); });
        }
        KOKKOS_INLINE_FUNCTION int owner_of(int f) const {
            if constexpr (float_rows)
                return Kokkos::bit_cast<int>(rows(f, 6));
            else
                return owner(f);
        }
        KOKKOS_INLINE_FUNCTION int neighbour_of(int f) const {
            if constexpr (float_rows)
                return Kokkos::bit_cast<int>(rows(f, 7));
            else
                return neighbour(f);
        }
        KOKKOS_INLINE_FUNCTION Vec area_of(int f) const {
            if constexpr (float_rows)
                return make_vec<D>([&](int a) { return static_cast<double>(rows(f, a)); });
            else
                return row(area, f);
        }
        KOKKOS_INLINE_FUNCTION Vec face_centroid_of(int f) const {
            if constexpr (float_rows)
                return make_vec<D>([&](int a) { return static_cast<double>(rows(f, 3 + a)); });
            else
                return row(face_centroid, f);
        }
        // The cell on the other side of face f from cell i; both loads issued at once (no dependent second load).
        KOKKOS_INLINE_FUNCTION int other(int f, int i) const {
            const int o = owner_of(f), n = neighbour_of(f);
            return o == i ? n : o;
        }
        // Face centroid of face f relative to the centroid of cell i (a side of f).
        KOKKOS_INLINE_FUNCTION Vec face_offset(int f, int i) const {
            const int o = owner_of(f);
            const Vec relative = o == i ? face_centroid_of(f) : face_centroid_of(f) + separation(i, o);
            return relative - row(centroid, i);
        }
    };

    // The phases, public only because CUDA device lambdas may not be defined in private member functions.

    // Ends a timed phase (fenced) when timings are wanted. Used by: time_step, step, move, advance.
    void phase_end(Timings* timings, double Timings::*part) {
        if (!timings) return;
        exec_.fence();
        timings->*part += timer_.seconds() * 1e3;
        timer_.reset();
    }

    // The positions start from the mesh's snapped ones. Used by: the constructor (moving).
    void copy_positions() {
        const auto from = mesh_->positions();
        const auto x = x_;
        Kokkos::parallel_for(
            "hydro::positions", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                for (int a = 0; a < D; ++a) x(i, a) = from(i, a);
            });
    }

    // The mesh's current arrays (an update may reallocate them). Used by: the constructor, time_step, step, move.
    void refresh() {
        const Mesh& mesh = *mesh_;
        faces_ = mesh.faces();
        geometry_ = {mesh.owner(),     mesh.neighbour(), mesh.cell_face_start(), mesh.cell_face(),
                     mesh.area(),      mesh.face_centroid(), mesh.centroid(),    mesh.face_rows(),
                     mesh.gradient_weights(), mesh.volume(), mesh.positions(),   mesh.sides()};
    }

    // Primitive rows, gradient rows (and FP16 exponents), then face flux rows (per-face loop), in the mesh's scratch
    // memory, each part starting on a 32-byte boundary. Used by: time_step, step (before primitives()).
    void borrow_scratch() {
        auto sectors = [](std::size_t bytes) { return (bytes + 31) / 32 * 32; };
        const std::size_t cells = static_cast<std::size_t>(n_);
        const std::size_t prim_bytes = sectors(sizeof(double) * (V + 2) * cells);
        const std::size_t gradient_bytes = order_ == 1 ? sizeof(GStore) * gradient_width * cells : 0;
        const std::size_t exponent_bytes = order_ == 1 && P::half ? sectors(sizeof(std::uint32_t) * cells) : 0;
        const std::size_t flux_bytes =
            update_ == Update::per_face ? sizeof(FluxStore) * V * static_cast<std::size_t>(faces_) : 0;
        char* base = mesh_->scratch(prim_bytes + gradient_bytes + exponent_bytes + flux_bytes).data();
        prims_ = Prims(reinterpret_cast<double*>(base), n_);
        base += prim_bytes;
        if (order_ == 1) gradients_ = Gradients(reinterpret_cast<GStore*>(base), n_);
        if (exponent_bytes > 0) exponents_ = Exponents(reinterpret_cast<std::uint32_t*>(base + gradient_bytes), n_);
        if (update_ == Update::per_face)
            flux_ = FaceFluxes(reinterpret_cast<FluxStore*>(base + gradient_bytes + exponent_bytes), faces_);
    }

    // W = primitive(Q / V), the sound speed and the radius of every cell, once per step.
    // Used by: time_step, step (first phase).
    void primitives() {
        const auto q = q_;
        const auto prim = prims_;
        const Geometry geo = geometry_;
        const Gas gas = gas_;
        Kokkos::parallel_for(
            "hydro::primitives", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                const State W = primitive(geo.per_volume(q, i), gas);
                for (int v = 0; v < V; ++v) prim(i, v) = W[v];
                prim(i, V) = sound_speed(W, gas);
                prim(i, V + 1) = radius(geo.volume(i));
            });
    }
    // Row i of the primitives (W only). Used by: Sweep, neighbour_range, FaceFlux::side.
    KOKKOS_INLINE_FUNCTION static State primitive_row(const Prims& prim, int i) {
        return make_state<D>([&](int v) { return prim(i, v); });
    }

    // The sweep of time_step() over cell i's face list: (Moving) the point velocity, AREPO-ng's
    // regularisation; (Grad) the limited least-squares gradients of the primitives, from the weights if the
    // precision has them. Used by: sweep(order, motion) (the reconstruction stage).
    template <bool Moving, bool Grad>
    struct Sweep {
        Prims prim;
        Points w;
        Gradients g;
        Exponents exponents;
        Geometry geo;
        Gas gas;
        Limiter limiter;

        KOKKOS_INLINE_FUNCTION void operator()(int i) const {
            constexpr bool exact = std::is_same_v<G, double>;
            const State Wi = primitive_row(prim, i);
            double surface = 0, smallest_face = Kokkos::Experimental::infinity_v<double>, max_angle = 0;
            Gradient<D, G> grad;
            // Neighbour range for the limiter: values W_j (double), or differences W_j - W_i (reduced precision).
            G lo[V], hi[V];
            for (int v = 0; v < V; ++v) {
                grad[v] = hydro::Vec<D, G>{};
                lo[v] = Kokkos::Experimental::finite_max_v<G>;
                hi[v] = -Kokkos::Experimental::finite_max_v<G>;
            }
            // Without stored weights: the fit's normal equations M x = b, summed here, solved after the loop.
            constexpr int U = D + 1;
            G M[U][U] = {};
            G b[V][U] = {};
            const Vec centroid = geo.row(geo.centroid, i);
            for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                const int f = geo.face(k), j = geo.other(f, i);
                const State Wj = primitive_row(prim, j);
                const Vec d = geo.separation(i, j);
                const Vec a = geo.area_of(f);
                const double magnitude = Kokkos::sqrt(dot(a, a));
                surface += magnitude;
                smallest_face = Kokkos::fmin(smallest_face, magnitude);
                if constexpr (Moving) max_angle = Kokkos::fmax(max_angle, face_angle(magnitude) / (0.5 * Kokkos::sqrt(dot(d, d))));
                if constexpr (Grad) {
                    G weight[D], X[U];
                    if constexpr (Mesh::with_weights) {
                        for (int c = 0; c < D; ++c) weight[c] = static_cast<G>(geo.weights(k, c));
                    } else {
                        const Vec between = d + geo.row(geo.centroid, j) - centroid;
                        X[0] = 1;
                        for (int p = 1; p < U; ++p) X[p] = static_cast<G>(between[p - 1]);
                        const G w = static_cast<G>(magnitude);
                        for (int p = 0; p < U; ++p)
                            for (int s = p; s < U; ++s) M[p][s] += w * X[p] * X[s];
                    }
                    for (int v = 0; v < V; ++v) {
                        const double dW = Wj[v] - Wi[v];
                        if constexpr (Mesh::with_weights) {
                            for (int c = 0; c < D; ++c) grad[v][c] += weight[c] * static_cast<G>(dW);
                        } else {
                            const G w = static_cast<G>(magnitude);
                            for (int p = 0; p < U; ++p) b[v][p] += w * X[p] * static_cast<G>(dW);
                        }
                        const G value = exact ? static_cast<G>(Wj[v]) : static_cast<G>(dW);
                        lo[v] = Kokkos::fmin(lo[v], value);
                        hi[v] = Kokkos::fmax(hi[v], value);
                    }
                }
            }
            // Rare: a face below 1e-6 of the surface does not count for the range or the limiter.
            const double tiny = 1e-6 * surface;
            const bool any_tiny = smallest_face < tiny;
            if constexpr (Grad) {
                if (any_tiny) neighbour_range(i, Wi, tiny, lo, hi);
            }
            if constexpr (Grad && !Mesh::with_weights) solve(M, b, grad);
            if constexpr (Moving) {
                double fraction = 0;
                if (max_angle > 0.75 * cell_max_angle_factor)
                    fraction = max_angle > cell_max_angle_factor
                                   ? cell_shaping_speed
                                   : cell_shaping_speed * (max_angle - 0.75 * cell_max_angle_factor) / (0.25 * cell_max_angle_factor);
                Vec wi = velocity(Wi);
                const Vec c = centroid;
                const double cd = Kokkos::sqrt(dot(c, c));
                if (cd > 0 && fraction > 0) wi = wi + c * (fraction * prim(i, V) / cd);
                for (int a = 0; a < D; ++a) w(i, a) = wi[a];
            }
            if constexpr (Grad) {
                // limit_gradient (mesh_gradients.cc:120-148), face by face.
                for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                    if (limiter == Limiter::none) break;
                    const int f = geo.face(k);
                    if (any_tiny) {
                        const Vec a = geo.area_of(f);
                        if (Kokkos::sqrt(dot(a, a)) < tiny) continue;
                    }
                    const int j = geo.other(f, i);
                    const Vec offset = limiter == Limiter::face_centroid
                                           ? geo.face_offset(f, i)
                                           : geo.separation(i, j) + geo.row(geo.centroid, j) - centroid;
                    const hydro::Vec<D, G> rg = cast<G>(offset);
                    for (int v = 0; v < V; ++v) {
                        const G dp = dot(grad[v], rg);
                        if constexpr (exact) {
                            if (dp > 0 && Wi[v] + dp > hi[v])
                                grad[v] = grad[v] * (hi[v] > Wi[v] ? (hi[v] - Wi[v]) / dp : 0.0);
                            else if (dp < 0 && Wi[v] + dp < lo[v])
                                grad[v] = grad[v] * (lo[v] < Wi[v] ? (lo[v] - Wi[v]) / dp : 0.0);
                        } else {
                            if (dp > 0 && dp > hi[v])
                                grad[v] = grad[v] * (hi[v] > 0 ? hi[v] / dp : G(0));
                            else if (dp < 0 && dp < lo[v])
                                grad[v] = grad[v] * (lo[v] < 0 ? lo[v] / dp : G(0));
                        }
                    }
                }
                store(i, grad);
            }
        }

        // The neighbours' range without the faces below tiny. Used by: operator() when a face is tiny.
        KOKKOS_FUNCTION void neighbour_range(int i, const State& Wi, double tiny, G (&lo)[V], G (&hi)[V]) const {
            constexpr bool exact = std::is_same_v<G, double>;
            for (int v = 0; v < V; ++v) {
                lo[v] = Kokkos::Experimental::finite_max_v<G>;
                hi[v] = -Kokkos::Experimental::finite_max_v<G>;
            }
            for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                const int f = geo.face(k), j = geo.other(f, i);
                const Vec a = geo.area_of(f);
                if (Kokkos::sqrt(dot(a, a)) < tiny) continue;
                const State Wj = primitive_row(prim, j);
                for (int v = 0; v < V; ++v) {
                    const G value = exact ? static_cast<G>(Wj[v]) : static_cast<G>(Wj[v] - Wi[v]);
                    lo[v] = Kokkos::fmin(lo[v], value);
                    hi[v] = Kokkos::fmax(hi[v], value);
                }
            }
        }

        // The gradient part of the fit's solution, M x_v = b_v for every variable: M is symmetric positive
        // definite, Cholesky M = L L^T once. Used by: operator() (no stored weights).
        template <int U>
        KOKKOS_INLINE_FUNCTION static void solve(const G (&M)[U][U], const G (&b)[V][U], Gradient<D, G>& grad) {
            G L[U][U] = {};
            for (int p = 0; p < U; ++p) {
                for (int s = 0; s <= p; ++s) {
                    G sum = M[s][p];
                    for (int m = 0; m < s; ++m) sum -= L[p][m] * L[s][m];
                    L[p][s] = p == s ? Kokkos::sqrt(sum) : sum / L[s][s];
                }
            }
            for (int v = 0; v < V; ++v) {
                G z[U], x[U];
                for (int p = 0; p < U; ++p) {
                    G sum = b[v][p];
                    for (int m = 0; m < p; ++m) sum -= L[p][m] * z[m];
                    z[p] = sum / L[p][p];
                }
                for (int p = U - 1; p >= 0; --p) {
                    G sum = z[p];
                    for (int m = p + 1; m < U; ++m) sum -= L[m][p] * x[m];
                    x[p] = sum / L[p][p];
                }
                for (int a = 0; a < D; ++a) grad[v][a] = x[1 + a];
            }
        }

        // Writes the gradient row of cell i; FP16: each variable group (rho, v, P) scaled by 2^k so that its
        // largest component lies in [2^14, 2^15), k in exponents(i) (one byte per group). Used by: operator().
        KOKKOS_INLINE_FUNCTION void store(int i, const Gradient<D, G>& grad) const {
            if constexpr (P::half) {
                float largest[3] = {0, 0, 0};
                for (int v = 0; v < V; ++v)
                    for (int a = 0; a < D; ++a) {
                        const int group = half::group<D>(D * v + a);
                        largest[group] = Kokkos::fmax(largest[group], Kokkos::fabs(grad[v][a]));
                    }
                int k[3];
                float scale[3];
                for (int s = 0; s < 3; ++s) {
                    k[s] = half::group_exponent(largest[s]);
                    scale[s] = half::power_of_two(k[s]);
                }
                for (int v = 0; v < V; ++v)
                    for (int a = 0; a < D; ++a) g(i, D * v + a) = half::from_float(grad[v][a] * scale[half::group<D>(D * v + a)]);
                exponents(i) = static_cast<std::uint32_t>(static_cast<std::uint8_t>(static_cast<std::int8_t>(k[0]))) |
                               static_cast<std::uint32_t>(static_cast<std::uint8_t>(static_cast<std::int8_t>(k[1]))) << 8 |
                               static_cast<std::uint32_t>(static_cast<std::uint8_t>(static_cast<std::int8_t>(k[2]))) << 16;
            } else {
                for (int v = 0; v < V; ++v)
                    for (int a = 0; a < D; ++a) g(i, D * v + a) = grad[v][a];
            }
        }
    };

    // The cell terms of the time step (see time_step()): reduces to the smallest. Used by: sweep(courant).
    template <bool Moving>
    struct CellStep {
        Prims prim;
        Points w;
        double courant;

        KOKKOS_INLINE_FUNCTION void operator()(int i, double& smallest) const {
            Vec relative = make_vec<D>([&](int a) { return prim(i, 1 + a); });
            if constexpr (Moving) relative = relative - make_vec<D>([&](int a) { return w(i, a); });
            const double step = prim(i, V + 1) / (prim(i, V) + Kokkos::sqrt(dot(relative, relative)));
            smallest = Kokkos::fmin(smallest, courant * step);
        }
    };
    // The pair terms: vsig_ij and d_ij are the same from both sides, so each face gives both cells' terms at once;
    // only their minimum is wanted. Used by: sweep(courant).
    struct FaceStep {
        Prims prim;
        Geometry geo;
        double courant;

        KOKKOS_INLINE_FUNCTION void operator()(int f, double& smallest) const {
            const int o = geo.owner_of(f), j = geo.neighbour_of(f);
            const Vec d = geo.separation(o, j);
            const double dist = Kokkos::sqrt(dot(d, d));
            const Vec dv = make_vec<D>([&](int a) { return prim(j, 1 + a) - prim(o, 1 + a); });
            const double vsig = prim(o, V) + prim(j, V) - dot(dv, d) / dist;
            if (vsig > 0) {
                smallest = Kokkos::fmin(smallest, courant * ((dist + prim(o, V + 1)) / vsig));
                smallest = Kokkos::fmin(smallest, courant * ((dist + prim(j, V + 1)) / vsig));
            }
        }
    };

    // Runs the sweep (gradients, point velocities) and returns the global time step. Used by: time_step, step.
    double sweep(double courant) {
        const bool moving = motion_ == Motion::moving;
        if (order_ == 1)
            return moving ? sweep<true, 1>(courant) : sweep<false, 1>(courant);
        return moving ? sweep<true, 0>(courant) : sweep<false, 0>(courant);
    }
    template <bool Moving, int Order>
    double sweep(double courant) {
        const Kokkos::RangePolicy<ExecSpace> cells(exec_, 0, n_);
        if constexpr (Moving || Order == 1)
            Kokkos::parallel_for("hydro::sweep", cells,
                                 Sweep<Moving, Order == 1>{prims_, w_, gradients_, exponents_, geometry_, gas_, limiter_});
        double by_cell = 0, by_face = 0;
        Kokkos::parallel_reduce("hydro::cell_step", cells, CellStep<Moving>{prims_, w_, courant}, Kokkos::Min<double>(by_cell));
        Kokkos::parallel_reduce("hydro::face_step", Kokkos::RangePolicy<ExecSpace>(exec_, 0, faces_),
                                FaceStep{prims_, geometry_, courant}, Kokkos::Min<double>(by_face));
        return Kokkos::min(by_cell, by_face);
    }

    // The points drift by dt * w and stay in the box; the mesh is updated (Mesh::update, then update_cells).
    // Used by: step (moving), after advance().
    void move(double dt, Timings* timings) {
        const auto x = x_;
        const auto w = w_;
        const exact_predicates::BoxN<D> box = mesh_->sides();
        Kokkos::parallel_for(
            "hydro::drift", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                for (int a = 0; a < D; ++a) {
                    const double moved = x(i, a) + dt * w(i, a);
                    x(i, a) = moved - box.side[a] * Kokkos::floor(moved / box.side[a]);
                }
            });
        phase_end(timings, &Timings::drift);
        geometry_ = {};
        const voronoi::Report report = mesh_->update(x_);
        mesh_stats_.updates += 1;
        mesh_stats_.rebuilds += report.rebuilt;
        mesh_stats_.stuck += report.stuck;
        mesh_stats_.host_flips += report.host_flips;
        mesh_stats_.device_flips += report.flips;
        phase_end(timings, &Timings::mesh);
        mesh_->update_cells();
        phase_end(timings, &Timings::cells);
        refresh();
    }

    // The flux through face f (times its area, from owner to neighbour): reconstruct both sides, Riemann solve.
    // Order: 0 or 1; Moving: faces have velocity. Used by: advance (per-cell or per-face loop).
    template <int Order, bool Moving>
    struct FaceFlux {
        Cells q;
        Prims prim;
        Gradients g;
        Exponents exponents;
        Points w;
        Geometry geo;
        Gas gas;
        double dt;

        // The stored gradient of cell i (FP16: decoded with the cell's group exponents).
        KOKKOS_INLINE_FUNCTION Gradient<D, G> gradient(int i) const {
            Gradient<D, G> grad;
            if constexpr (P::half) {
                const std::uint32_t k = exponents(i);
                float scale[3];
                for (int s = 0; s < 3; ++s) scale[s] = half::power_of_two(-static_cast<int>(static_cast<std::int8_t>(k >> (8 * s))));
                for (int v = 0; v < V; ++v)
                    grad[v] = make_vec<D, G>(
                        [&](int a) { return half::to_float(g(i, D * v + a)) * scale[half::group<D>(D * v + a)]; });
            } else {
                for (int v = 0; v < V; ++v) grad[v] = make_vec<D, G>([&](int a) { return g(i, D * v + a); });
            }
            return grad;
        }
        // As AREPO's prepare_state: time and space extrapolation to the face centroid at the half step,
        // only if rho and P stay positive, else the cell's state. Order 0: W computed here from Q.
        // r: offset of the face centroid from cell i's centroid. Used by: operator().
        KOKKOS_INLINE_FUNCTION State side(int i, Vec r, int& fallbacks) const {
            if constexpr (Order == 0) {
                (void)r;
                (void)fallbacks;
                return primitive(geo.per_volume(q, i), gas);
            } else {
                const State W = primitive_row(prim, i);
                const Gradient<D, G> grad = gradient(i);
                State Wf;
                if constexpr (std::is_same_v<G, double>) {
                    Wf = half_step(W, grad, dt, gas);
                    for (int v = 0; v < V; ++v) Wf[v] += dot(grad[v], r);
                } else {
                    const StateT<D, G> Wg = make_state<D, G>([&](int v) { return static_cast<G>(W[v]); });
                    const StateT<D, G> rate = half_step_rate(Wg, grad, gas);
                    const hydro::Vec<D, G> rg = cast<G>(r);
                    const G h = G(0.5) * static_cast<G>(dt);
                    for (int v = 0; v < V; ++v) Wf[v] = W[v] + static_cast<double>(dot(grad[v], rg) - h * rate[v]);
                }
                if (Wf[0] > 0 && Wf[D + 1] > 0) return Wf;
                ++fallbacks;
                return W;
            }
        }
        // The face velocity of AREPO-ng's prepare_face_velocity: the mean of the two point velocities
        // plus a term along the generator direction for the face centroid's offset. Used by: operator() (moving).
        KOKKOS_INLINE_FUNCTION Vec face_velocity(int f, int o, int j) const {
            const Vec wL = geo.row(w, o), wR = geo.row(w, j);
            const Vec d = geo.separation(o, j);
            const double dist = Kokkos::sqrt(dot(d, d));
            const Vec c = geo.face_centroid_of(f) - d * 0.5;
            double facv = dot(c, wL - wR) / dist;
            const double cc = Kokkos::sqrt(dot(c, c));
            if (cc > 0.9 * dist) facv *= 0.9 * dist / cc;
            return (wL + wR) * 0.5 + d * (facv / dist);
        }
        // The Riemann solve along the unit normal n; float: in the face's frame (velocities relative to wf in
        // double, then rounded), the flux back to the lab frame in double. Used by: operator().
        KOKKOS_INLINE_FUNCTION static State solve(const State& L, const State& R, Vec n, Vec wf, const Gas& gas) {
            if constexpr (std::is_same_v<typename P::riemann, double>) {
                return Riemann::flux(L, R, n, wf, gas);
            } else {
                using T = typename P::riemann;
                auto relative = [&](const State& S) {
                    return make_state<D, T>([&](int v) {
                        return static_cast<T>(v >= 1 && v <= D ? S[v] - wf[v - 1] : S[v]);
                    });
                };
                const StateT<D, T> F = Riemann::flux(relative(L), relative(R), cast<T>(n), hydro::Vec<D, T>{}, gas);
                return detail::to_lab(make_state<D>([&](int v) { return static_cast<double>(F[v]); }), wf);
            }
        }
        // fallbacks_o, fallbacks_j: counts of the owner's and neighbour's sides that took the cell's state.
        KOKKOS_INLINE_FUNCTION State operator()(int f, int& fallbacks_o, int& fallbacks_j) const {
            const int o = geo.owner_of(f), j = geo.neighbour_of(f);
            const Vec a = geo.area_of(f);
            const double magnitude = Kokkos::sqrt(dot(a, a));
            if (magnitude == 0) return State{};  // cospherical corners (an exact lattice): no normal, no flux
            Vec wf{};
            if constexpr (Moving) wf = face_velocity(f, o, j);
            State L, R;
            if constexpr (Order == 0) {
                L = side(o, Vec{}, fallbacks_o);
                R = side(j, Vec{}, fallbacks_j);
            } else {
                const Vec shift = wf * (0.5 * dt);
                L = side(o, geo.face_offset(f, o) + shift, fallbacks_o);
                R = side(j, geo.face_offset(f, j) + shift, fallbacks_j);
            }
            State F = solve(L, R, a * (1.0 / magnitude), wf, gas);
            for (int v = 0; v < V; ++v) F[v] *= magnitude;
            return F;
        }
    };

    // Dispatches the update on the runtime motion. Used by: step.
    template <int Order>
    void advance(double dt, Timings* timings) {
        if (motion_ == Motion::moving)
            advance<Order, true>(dt, timings);
        else
            advance<Order, false>(dt, timings);
    }

    // Fluxes on every face, then Q -= dt * sum of signed fluxes (per_cell: one loop; per_face: two).
    // Used by: step, after the sweep.
    template <int Order, bool Moving>
    void advance(double dt, Timings* timings) {
        const FaceFlux<Order, Moving> flux{q_, prims_, gradients_, exponents_, w_, geometry_, gas_, dt};
        const Geometry geo = geometry_;
        const auto q = q_;
        if (update_ == Update::per_cell) {
            // In place at order 1: the fluxes read only the primitive and gradient rows; order 0 reads Q
            // (FaceFlux::side), so into next_.
            const auto next = Order == 0 ? next_ : q_;
            Kokkos::parallel_reduce(
                "hydro::fluxes_and_update", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_),
                KOKKOS_LAMBDA(int i, long long& count) {
                    State sum{};
                    for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                        const int f = geo.face(k);
                        int fallbacks_o = 0, fallbacks_j = 0;
                        const State F = flux(f, fallbacks_o, fallbacks_j);
                        const bool owned = geo.owner_of(f) == i;
                        count += owned ? fallbacks_o : fallbacks_j;
                        const double sign = owned ? 1.0 : -1.0;
                        for (int v = 0; v < V; ++v) sum[v] += sign * F[v];
                    }
                    for (int v = 0; v < V; ++v) next(i, v) = q(i, v) - dt * sum[v];
                },
                fallbacks_);
            phase_end(timings, &Timings::fluxes);
            if constexpr (Order == 0) std::swap(q_, next_);
            return;
        }
        const auto F = flux_;
        Kokkos::parallel_reduce(
            "hydro::fluxes", Kokkos::RangePolicy<ExecSpace>(exec_, 0, faces_),
            KOKKOS_LAMBDA(int f, long long& count) {
                int fallbacks = 0;
                const State value = flux(f, fallbacks, fallbacks);
                count += fallbacks;
                for (int v = 0; v < V; ++v) F(f, v) = static_cast<FluxStore>(value[v]);
            },
            fallbacks_);
        phase_end(timings, &Timings::fluxes);
        Kokkos::parallel_for(
            "hydro::update", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                State sum{};
                for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                    const int f = geo.face(k);
                    const double sign = geo.owner_of(f) == i ? 1.0 : -1.0;
                    for (int v = 0; v < V; ++v) sum[v] += sign * static_cast<double>(F(f, v));
                }
                for (int v = 0; v < V; ++v) q(i, v) = q(i, v) - dt * sum[v];
            });
        phase_end(timings, &Timings::update);
    }

private:
    ExecSpace exec_;
    Mesh* mesh_;
    Gas gas_;
    int order_;
    Update update_;
    Limiter limiter_;
    Motion motion_;
    int n_, faces_ = 0;
    bool swept_ = false;
    long long fallbacks_ = 0;
    Kokkos::Timer timer_;
    Cells q_, next_;
    Prims prims_;
    Points x_, w_;
    Gradients gradients_;
    Exponents exponents_;
    FaceFluxes flux_;
    Geometry geometry_;
    MeshStats mesh_stats_;
};

}  // namespace hydro
