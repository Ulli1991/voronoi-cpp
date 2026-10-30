#pragma once

#include "euler.hpp"
#include "voronoi_mesh.hpp"

#include <Kokkos_Core.hpp>

#include <stdexcept>
#include <string>
#include <utility>

// Finite volumes on a static voronoi::Mesh: MUSCL-Hancock (order 1) or
// first-order Godunov (order 0), a Riemann solver on every face along its
// normal (HLLC by default, as LUNA's riemann_solver; HLL; any type with
// their static flux() fits), one global time
// step. The state between steps is the conserved U per unit volume only;
// primitives are computed where they are needed.
//
// A step, after LUNA (time_integration/integrators.py:257-263 and
// hydro/muscl_hancock.py:90-160) on the faces of the Voronoi mesh:
//   1. W = primitive(U) per cell (floored);
//   2. order 1: least-squares gradients of W over the face neighbours,
//      limited at the face centroids (AREPO next generation);
//   3. per face: each side's W advanced half a step with the primitive
//      Jacobian and extrapolated from the cell centroid to the face
//      centroid, or, if that leaves rho or P not positive, the cell's W
//      (AREPO next generation); the flux along n = A / |A|, times |A|;
//   4. U += -dt / V sum over the cell's faces of the outgoing flux.
// Everything a cell sums (gradients, fluxes) is summed over the mesh's
// per-cell face list in its fixed order, without atomics: the same bits
// every run.
//
// Two update loops, measured against each other (research notes):
//   per_cell: one thread per cell computes the flux of each of its faces
//     (each face twice, once from each side, with the same arguments, so
//     both get the same bits) and writes the new state into a second array;
//   per_face: one thread per face writes its flux into a face array, then
//     one thread per cell gathers them and updates in place.
// Gradients (15 doubles per cell) and the face fluxes live in the mesh's
// scratch memory, asked for at every step (scratch is valid until the
// mesh's next update(), rebuild() or update_cells()).
namespace hydro {

enum class Update { per_cell, per_face };

// Where the gradient limiter keeps the extrapolated values within the
// neighbours' range (AREPO next generation, mesh_gradients.cc:84-94): at
// the face centroids (AREPO's default), at the neighbours' centroids
// (AREPO's USE_MINMOD_SLOPE_LIMITING), or nowhere (no limiter).
enum class Limiter { face_centroid, cell_centroid, none };

// Times of the phases of a step, in ms, summed over steps (fenced).
struct Timings {
    double gradients = 0, fluxes = 0, update = 0;
};

template <class ExecSpace = Kokkos::DefaultExecutionSpace, class Riemann = HLLC>
class FiniteVolume {
public:
    using Memory = typename ExecSpace::memory_space;
    using Mesh = voronoi::Mesh<ExecSpace>;
    using Cells = Kokkos::View<double* [5], Memory>;
    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using Gradients = Kokkos::View<double* [15], typename Cells::array_layout, Memory, Unmanaged>;
    using FaceFluxes = Kokkos::View<double* [5], typename Cells::array_layout, Memory, Unmanaged>;

    // mesh: built, with update_cells() done; its geometry must not change
    // while this object is used.
    FiniteVolume(const ExecSpace& exec, Mesh& mesh, const Gas& gas, int order, Update update,
                 Limiter limiter = Limiter::face_centroid)
        : exec_(exec), mesh_(&mesh), gas_(gas), order_(order), update_(update), limiter_(limiter), n_(mesh.particles()),
          faces_(mesh.faces()) {
        if (order != 0 && order != 1) throw std::invalid_argument("hydro::FiniteVolume: order 0 or 1");
        u_ = Cells("hydro_U", n_);
        if (update == Update::per_cell) next_ = Cells(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_U_next")), n_);
        geometry_ = {mesh.owner(),     mesh.neighbour(), mesh.cell_face_start(), mesh.cell_face(), mesh.area(),
                     mesh.face_centroid(), mesh.centroid(), mesh.volume(), mesh.positions(), mesh.box()};
    }

    // The conserved state per unit volume, U(i, v); set it before stepping.
    Cells state() const { return u_; }
    int order() const { return order_; }
    Update update() const { return update_; }
    const Gas& gas() const { return gas_; }

    // The global time step: the smallest of AREPO next generation's cell
    // time steps (FORCE_EQUAL_TIMESTEPS, time_integration/timestep.cc:39-63)
    // without the rounding to a power-of-two time bin. Cell i:
    // courant * min(r_i / (c_i + |v_i|), min_j (d_ij + r_i) / vsig_ij), r_i
    // the radius of a sphere of the cell's volume (sph_particle_data.h:280-287,
    // 305-334), and for every face neighbour j with
    // vsig_ij = c_i + c_j - (v_j - v_i).d_ij / d_ij > 0, d_ij between the
    // generators (timestep_treebased.cc:72-91; AREPO walks every particle a
    // signal can reach, here only the face neighbours).
    double time_step(double courant) const {
        double dt = 0;
        const auto u = u_;
        const Geometry geo = geometry_;
        const Gas gas = gas_;
        Kokkos::parallel_reduce(
            "hydro::time_step", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_),
            KOKKOS_LAMBDA(int i, double& smallest) {
                const State Wi = primitive(load(u, i), gas);
                const double ci = sound_speed(Wi, gas);
                const double r = Kokkos::cbrt(geo.volume(i) * (3.0 / (4.0 * M_PI)));
                double step = r / (ci + Kokkos::sqrt(Wi[1] * Wi[1] + Wi[2] * Wi[2] + Wi[3] * Wi[3]));
                for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                    const int j = geo.other(geo.face(k), i);
                    const State Wj = primitive(load(u, j), gas);
                    const Vec3 d = geo.separation(i, j);
                    const double dist = Kokkos::sqrt(dot(d, d));
                    const Vec3 dv{Wj[1] - Wi[1], Wj[2] - Wi[2], Wj[3] - Wi[3]};
                    const double vsig = ci + sound_speed(Wj, gas) - dot(dv, d) / dist;
                    if (vsig > 0) step = Kokkos::fmin(step, (dist + r) / vsig);
                }
                smallest = Kokkos::fmin(smallest, courant * step);
            },
            Kokkos::Min<double>(dt));
        return dt;
    }

    // One step of length dt.
    void step(double dt, Timings* timings = nullptr) {
        timer_.reset();
        borrow_scratch();
        if (order_ == 1) gradients();
        phase_end(timings, &Timings::gradients);
        if (order_ == 1)
            advance<1>(dt, timings);
        else
            advance<0>(dt, timings);
    }

    // Face sides of the last step whose extrapolated state was not
    // physical and which took the cell's state instead (per-face loop;
    // -1 with the per-cell loop, which does not count them).
    long long fallbacks() const { return fallbacks_; }

    // Mesh arrays and the per-face and per-cell operations on them.
    struct Geometry {
        Kokkos::View<const int*, Memory> owner, neighbour, start, face;
        Kokkos::View<const double* [3], Memory> area, face_centroid, centroid;
        Kokkos::View<const double*, Memory> volume;
        tet_mesh::Points4<Memory> positions;
        double box;

        KOKKOS_INLINE_FUNCTION Vec3 row(const Kokkos::View<const double* [3], Memory>& v, int i) const {
            return {v(i, 0), v(i, 1), v(i, 2)};
        }
        // Nearest image of generator j seen from generator i.
        KOKKOS_INLINE_FUNCTION Vec3 separation(int i, int j) const {
            return {tet_mesh::wrap(positions(j, 0) - positions(i, 0), box),
                    tet_mesh::wrap(positions(j, 1) - positions(i, 1), box),
                    tet_mesh::wrap(positions(j, 2) - positions(i, 2), box)};
        }
        KOKKOS_INLINE_FUNCTION int other(int f, int i) const { return owner(f) == i ? neighbour(f) : owner(f); }
        // Face centroid of face f relative to the centroid of cell i (a side of f).
        KOKKOS_INLINE_FUNCTION Vec3 face_offset(int f, int i) const {
            const int o = owner(f);
            const Vec3 relative = o == i ? row(face_centroid, f) : row(face_centroid, f) + separation(i, o);
            return relative - row(centroid, i);
        }
        KOKKOS_INLINE_FUNCTION double surface(int i) const {
            double sum = 0;
            for (int k = start(i); k < start(i + 1); ++k) {
                const Vec3 a = row(area, face(k));
                sum += Kokkos::sqrt(dot(a, a));
            }
            return sum;
        }
    };

    // The phases of step(), public only because CUDA device lambdas may not
    // be defined in private member functions.

    // Ends a timed phase (fenced) when timings are wanted.
    void phase_end(Timings* timings, double Timings::*part) {
        if (!timings) return;
        exec_.fence();
        timings->*part += timer_.seconds() * 1e3;
        timer_.reset();
    }

    // Gradients, then face fluxes, in the mesh's scratch memory.
    void borrow_scratch() {
        const std::size_t gradient_bytes = order_ == 1 ? sizeof(double) * 15 * std::size_t(n_) : 0;
        const std::size_t flux_bytes = update_ == Update::per_face ? sizeof(double) * 5 * std::size_t(faces_) : 0;
        char* base = mesh_->scratch(gradient_bytes + flux_bytes).data();
        if (order_ == 1) gradients_ = Gradients(reinterpret_cast<double*>(base), n_);
        if (update_ == Update::per_face) flux_ = FaceFluxes(reinterpret_cast<double*>(base + gradient_bytes), faces_);
    }

    KOKKOS_INLINE_FUNCTION static State load(const Cells& u, int i) { return {u(i, 0), u(i, 1), u(i, 2), u(i, 3), u(i, 4)}; }

    // Gradients of the primitives as AREPO's next generation computes them
    // (Arepo2-nextgeneration-SoA, src/mesh/mesh_gradients.cc; lines in the
    // notes): a least-squares fit W_j - W_i = a + g.d_ij over all face
    // neighbours j, weighted by the face area, d_ij between the cell
    // centroids (an intercept a and the gradient g: four unknowns); then,
    // for each variable, the gradient is scaled down face by face (faces
    // above 1e-6 of the cell's surface, in the cell's face order) until the
    // value extrapolated to every face centroid stays within the range of
    // the neighbours' values.
    void gradients() {
        const auto u = u_;
        const auto g = gradients_;
        const Geometry geo = geometry_;
        const Gas gas = gas_;
        const Limiter limiter = limiter_;
        Kokkos::parallel_for(
            "hydro::gradients", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                const State Wi = primitive(load(u, i), gas);
                const double tiny = 1e-6 * geo.surface(i);
                const Vec3 ci = geo.row(geo.centroid, i);
                double A[4][4] = {};
                double b[5][4] = {};
                State lo, hi;
                for (int v = 0; v < 5; ++v) {
                    lo[v] = Kokkos::Experimental::finite_max_v<double>;
                    hi[v] = -Kokkos::Experimental::finite_max_v<double>;
                }
                for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                    const int f = geo.face(k), j = geo.other(f, i);
                    const Vec3 a = geo.row(geo.area, f);
                    const double w = Kokkos::sqrt(dot(a, a));
                    const Vec3 d = geo.separation(i, j) + geo.row(geo.centroid, j) - ci;
                    const double X[4] = {1, d.x, d.y, d.z};
                    for (int p = 0; p < 4; ++p)
                        for (int q = p; q < 4; ++q) A[p][q] += w * X[p] * X[q];
                    const State Wj = primitive(load(u, j), gas);
                    for (int v = 0; v < 5; ++v) {
                        const double dW = Wj[v] - Wi[v];
                        for (int p = 0; p < 4; ++p) b[v][p] += w * X[p] * dW;
                        if (w < tiny) continue;
                        lo[v] = Kokkos::fmin(lo[v], Wj[v]);
                        hi[v] = Kokkos::fmax(hi[v], Wj[v]);
                    }
                }
                // A is symmetric positive definite: Cholesky A = L L^T, once for all five.
                double L[4][4] = {};
                for (int p = 0; p < 4; ++p) {
                    for (int q = 0; q <= p; ++q) {
                        double sum = A[q][p];
                        for (int m = 0; m < q; ++m) sum -= L[p][m] * L[q][m];
                        L[p][q] = p == q ? Kokkos::sqrt(sum) : sum / L[q][q];
                    }
                }
                Gradient grad;
                for (int v = 0; v < 5; ++v) {
                    double z[4], x[4];
                    for (int p = 0; p < 4; ++p) {
                        double sum = b[v][p];
                        for (int m = 0; m < p; ++m) sum -= L[p][m] * z[m];
                        z[p] = sum / L[p][p];
                    }
                    for (int p = 3; p >= 0; --p) {
                        double sum = z[p];
                        for (int m = p + 1; m < 4; ++m) sum -= L[m][p] * x[m];
                        x[p] = sum / L[p][p];
                    }
                    grad[v] = Vec3{x[1], x[2], x[3]};
                }
                // limit_gradient (mesh_gradients.cc:120-148), face by face.
                for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                    if (limiter == Limiter::none) break;
                    const int f = geo.face(k);
                    const Vec3 a = geo.row(geo.area, f);
                    if (Kokkos::sqrt(dot(a, a)) < tiny) continue;
                    const int j = geo.other(f, i);
                    const Vec3 r = limiter == Limiter::face_centroid
                                       ? geo.face_offset(f, i)
                                       : geo.separation(i, j) + geo.row(geo.centroid, j) - ci;
                    for (int v = 0; v < 5; ++v) {
                        const double dp = dot(grad[v], r);
                        if (dp > 0 && Wi[v] + dp > hi[v])
                            grad[v] = grad[v] * (hi[v] > Wi[v] ? (hi[v] - Wi[v]) / dp : 0.0);
                        else if (dp < 0 && Wi[v] + dp < lo[v])
                            grad[v] = grad[v] * (lo[v] < Wi[v] ? (lo[v] - Wi[v]) / dp : 0.0);
                    }
                }
                for (int v = 0; v < 5; ++v) {
                    g(i, 3 * v + 0) = grad[v].x;
                    g(i, 3 * v + 1) = grad[v].y;
                    g(i, 3 * v + 2) = grad[v].z;
                }
            });
    }

    // The flux through face f (times its area), from owner to neighbour;
    // `fallbacks` counts the sides that took the cell's state.
    template <int Order>
    struct FaceFlux {
        Cells u;
        Gradients g;
        Geometry geo;
        Gas gas;
        double dt;

        // As AREPO's prepare_state (mesh_fluxes.cc:451-479): the time and
        // space extrapolations are added only if rho and P stay positive;
        // otherwise the cell's own state is used.
        KOKKOS_INLINE_FUNCTION State side(int i, const Vec3& r, int& fallbacks) const {
            const State W = primitive(load(u, i), gas);
            if constexpr (Order == 0) {
                (void)r;
                (void)fallbacks;
                return W;
            } else {
                Gradient grad;
                for (int v = 0; v < 5; ++v) grad[v] = Vec3{g(i, 3 * v), g(i, 3 * v + 1), g(i, 3 * v + 2)};
                State Wf = half_step(W, grad, dt, gas);
                for (int v = 0; v < 5; ++v) Wf[v] += dot(grad[v], r);
                if (Wf[0] > 0 && Wf[4] > 0) return Wf;
                ++fallbacks;
                return W;
            }
        }
        KOKKOS_INLINE_FUNCTION State operator()(int f, int& fallbacks) const {
            const int o = geo.owner(f), j = geo.neighbour(f);
            const Vec3 a = geo.row(geo.area, f);
            const double magnitude = Kokkos::sqrt(dot(a, a));
            State L, R;
            if constexpr (Order == 0) {
                L = side(o, {0, 0, 0}, fallbacks);
                R = side(j, {0, 0, 0}, fallbacks);
            } else {
                L = side(o, geo.face_offset(f, o), fallbacks);
                R = side(j, geo.face_offset(f, j), fallbacks);
            }
            State F = Riemann::flux(L, R, a * (1.0 / magnitude), Vec3{0, 0, 0}, gas);
            for (int v = 0; v < 5; ++v) F[v] *= magnitude;
            return F;
        }
    };

    template <int Order>
    void advance(double dt, Timings* timings) {
        const FaceFlux<Order> flux{u_, gradients_, geometry_, gas_, dt};
        const Geometry geo = geometry_;
        if (update_ == Update::per_cell) {
            const auto u = u_, next = next_;
            // Not counted here: a reduction makes this kernel 2x slower on an
            // A100 (job 31699166); fallbacks() is -1.
            Kokkos::parallel_for(
                "hydro::fluxes_and_update", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                    State sum{0, 0, 0, 0, 0};
                    for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                        const int f = geo.face(k);
                        int fallbacks = 0;
                        const State F = flux(f, fallbacks);
                        const double sign = geo.owner(f) == i ? 1.0 : -1.0;
                        for (int v = 0; v < 5; ++v) sum[v] += sign * F[v];
                    }
                    const double factor = -dt / geo.volume(i);
                    for (int v = 0; v < 5; ++v) next(i, v) = u(i, v) + factor * sum[v];
                });
            fallbacks_ = -1;
            phase_end(timings, &Timings::fluxes);
            std::swap(u_, next_);
            return;
        }
        const auto F = flux_;
        Kokkos::parallel_reduce(
            "hydro::fluxes", Kokkos::RangePolicy<ExecSpace>(exec_, 0, faces_),
            KOKKOS_LAMBDA(int f, long long& count) {
                int fallbacks = 0;
                const State value = flux(f, fallbacks);
                count += fallbacks;
                for (int v = 0; v < 5; ++v) F(f, v) = value[v];
            },
            fallbacks_);
        phase_end(timings, &Timings::fluxes);
        const auto u = u_;
        Kokkos::parallel_for(
            "hydro::update", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                State sum{0, 0, 0, 0, 0};
                for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                    const int f = geo.face(k);
                    const double sign = geo.owner(f) == i ? 1.0 : -1.0;
                    for (int v = 0; v < 5; ++v) sum[v] += sign * F(f, v);
                }
                const double factor = -dt / geo.volume(i);
                for (int v = 0; v < 5; ++v) u(i, v) = u(i, v) + factor * sum[v];
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
    int n_, faces_;
    long long fallbacks_ = 0;
    Kokkos::Timer timer_;
    Cells u_, next_;
    Gradients gradients_;
    FaceFluxes flux_;
    Geometry geometry_;
};

}  // namespace hydro
