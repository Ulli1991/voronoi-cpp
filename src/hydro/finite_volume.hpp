#pragma once

#include "euler.hpp"
#include "voronoi_mesh.hpp"

#include <Kokkos_Core.hpp>

#include <stdexcept>
#include <string>
#include <utility>

// Finite volumes on a voronoi::Mesh, static or moving with the flow:
// MUSCL-Hancock (order 1) or first-order Godunov (order 0), a Riemann
// solver on every face along its normal in the frame of the moving face
// (HLLC by default, as LUNA's riemann_solver; HLL; any type with their
// static flux() fits), one global time step. The state between steps is
// the extensive conserved Q = (mass, momentum, energy) of each cell, as
// AREPO next generation keeps it (mesh_fluxes.cc:162-164, density =
// mass / volume in mesh_update_primitives.cc:129-131); primitives are
// computed from Q / V where they are needed. Line references are in
// research_notes/hydro_stage1_2026-10-31.md and hydro_stage3_2026-10-31.md.
//
// A step:
//   time_step(): (moving) the mesh-generating point velocities, fluid
//     velocity plus AREPO-ng's regularisation; the global time step;
//   step(dt):
//   1. W = primitive(Q / V) per cell (floored);
//   2. order 1: least-squares gradients of W over the face neighbours,
//      limited at the face centroids (AREPO next generation);
//   3. per face: the face velocity from the two point velocities
//      (AREPO-ng); each side's W advanced half a step with the primitive
//      Jacobian and extrapolated to where the face centroid is at the half
//      step, or, if that leaves rho or P not positive, the cell's W
//      (AREPO next generation); the flux through the moving face along
//      n = A / |A|, times |A|, on the geometry at the start of the step;
//   4. Q -= dt * sum over the cell's faces of the outgoing flux;
//   5. (moving) the points drift by dt * w, the mesh is updated
//      (voronoi::Mesh::update, update_cells).
// Everything a cell sums is summed over the mesh's per-cell face list in
// its fixed order, without atomics: the same bits every run.
//
// Two update loops, measured against each other (research notes):
//   per_cell: one thread per cell computes the flux of each of its faces
//     (each face twice, once from each side, with the same arguments, so
//     both get the same bits) and writes the new state into a second array;
//   per_face: one thread per face writes its flux into a face array, then
//     one thread per cell gathers them and updates in place.
// Gradients (15 doubles per cell) and the face fluxes live in the mesh's
// scratch memory, asked for at every step after the mesh update (scratch
// is valid until the mesh's next update(), rebuild() or update_cells()).
namespace hydro {

enum class Update { per_cell, per_face };

// Where the gradient limiter keeps the extrapolated values within the
// neighbours' range (AREPO next generation, mesh_gradients.cc:84-94): at
// the face centroids (AREPO's default), at the neighbours' centroids
// (AREPO's USE_MINMOD_SLOPE_LIMITING), or nowhere (no limiter).
enum class Limiter { face_centroid, cell_centroid, none };

// A static mesh, or one whose points move with the flow.
enum class Motion { static_mesh, moving };

// Times of the phases of a step, in ms, summed over steps (fenced).
struct Timings {
    double velocities = 0, gradients = 0, fluxes = 0, update = 0, drift = 0, mesh = 0, cells = 0;
};

// What the mesh updates of a moving run did, summed over steps.
struct MeshStats {
    long long updates = 0, rebuilds = 0, stuck = 0, host_flips = 0, device_flips = 0;
};

template <class ExecSpace = Kokkos::DefaultExecutionSpace, class Riemann = HLLC>
class FiniteVolume {
public:
    using Memory = typename ExecSpace::memory_space;
    using Mesh = voronoi::Mesh<ExecSpace>;
    using Cells = Kokkos::View<double* [5], Memory>;
    using Points = Kokkos::View<double* [3], Memory>;
    using Unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
    using Gradients = Kokkos::View<double* [15], typename Cells::array_layout, Memory, Unmanaged>;
    using FaceFluxes = Kokkos::View<double* [5], typename Cells::array_layout, Memory, Unmanaged>;

    // AREPO next generation's mesh regularisation as its Sedov examples
    // configure it (examples/sedov_3d/sedov_globaltstp_3d/Config.sh:11-13:
    // REGULARIZE_MESH_CM_DRIFT, _USE_SOUNDSPEED, _FACE_ANGLE; param.txt:93-94).
    static constexpr double cell_shaping_speed = 0.5, cell_max_angle_factor = 2.25;

    // mesh: built, with update_cells() done. Static: its geometry must not
    // change while this object is used. Moving: this object owns the
    // positions (starting from the mesh's snapped ones) and updates the mesh.
    FiniteVolume(const ExecSpace& exec, Mesh& mesh, const Gas& gas, int order, Update update,
                 Limiter limiter = Limiter::face_centroid, Motion motion = Motion::static_mesh)
        : exec_(exec), mesh_(&mesh), gas_(gas), order_(order), update_(update), limiter_(limiter), motion_(motion),
          n_(mesh.particles()) {
        if (order != 0 && order != 1) throw std::invalid_argument("hydro::FiniteVolume: order 0 or 1");
        q_ = Cells("hydro_Q", n_);
        if (update == Update::per_cell) next_ = Cells(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_Q_next")), n_);
        if (motion == Motion::moving) {
            x_ = Points(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_x")), n_);
            w_ = Points("hydro_w", n_);
            copy_positions();
        }
        refresh();
    }

    // The extensive conserved state of each cell, Q(i, v) = (mass, momentum,
    // energy); set it before stepping.
    Cells state() const { return q_; }
    // Moving: the positions of the mesh-generating points and their velocities.
    Points positions() const { return x_; }
    Points point_velocities() const { return w_; }
    int order() const { return order_; }
    Update update() const { return update_; }
    Motion motion() const { return motion_; }
    const Gas& gas() const { return gas_; }
    const MeshStats& mesh_stats() const { return mesh_stats_; }

    // The global time step: the smallest of AREPO next generation's cell
    // time steps (FORCE_EQUAL_TIMESTEPS, time_integration/timestep.cc:39-63)
    // without the rounding to a power-of-two time bin. Cell i:
    // courant * min(r_i / (c_i + |v_i - w_i|), min_j (d_ij + r_i) / vsig_ij),
    // w_i the point velocity (0 on a static mesh; timestep.cc:156-165,
    // sph_particle_data.h:305-334), r_i the radius of a sphere of the
    // cell's volume (sph_particle_data.h:280-287), and for every face
    // neighbour j with vsig_ij = c_i + c_j - (v_j - v_i).d_ij / d_ij > 0,
    // d_ij between the generators (timestep_treebased.cc:72-91; AREPO walks
    // every particle a signal can reach, here only the face neighbours).
    // Moving: sets the point velocities first (they do not depend on dt).
    double time_step(double courant, Timings* timings = nullptr) {
        timer_.reset();
        refresh();
        if (motion_ == Motion::moving) point_velocities_kernel();
        phase_end(timings, &Timings::velocities);
        double dt = 0;
        const auto q = q_;
        const auto w = w_;
        const bool moving = motion_ == Motion::moving;
        const Geometry geo = geometry_;
        const Gas gas = gas_;
        Kokkos::parallel_reduce(
            "hydro::time_step", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_),
            KOKKOS_LAMBDA(int i, double& smallest) {
                const State Wi = primitive(geo.per_volume(q, i), gas);
                const double ci = sound_speed(Wi, gas);
                const double r = Kokkos::cbrt(geo.volume(i) * (3.0 / (4.0 * M_PI)));
                Vec3 relative{Wi[1], Wi[2], Wi[3]};
                if (moving) relative = relative - Vec3{w(i, 0), w(i, 1), w(i, 2)};
                double step = r / (ci + Kokkos::sqrt(dot(relative, relative)));
                for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                    const int j = geo.other(geo.face(k), i);
                    const State Wj = primitive(geo.per_volume(q, j), gas);
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

    // One step of length dt (moving: after time_step(), which set the point
    // velocities).
    void step(double dt, Timings* timings = nullptr) {
        timer_.reset();
        refresh();
        borrow_scratch();
        if (order_ == 1) gradients();
        phase_end(timings, &Timings::gradients);
        if (order_ == 1)
            advance<1>(dt, timings);
        else
            advance<0>(dt, timings);
        if (motion_ == Motion::moving) move(dt, timings);
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
        // The conserved state of cell i per unit volume.
        KOKKOS_INLINE_FUNCTION State per_volume(const Cells& q, int i) const {
            const double inv = 1.0 / volume(i);
            return {q(i, 0) * inv, q(i, 1) * inv, q(i, 2) * inv, q(i, 3) * inv, q(i, 4) * inv};
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

    // The positions start from the mesh's snapped ones.
    void copy_positions() {
        const auto from = mesh_->positions();
        const auto x = x_;
        Kokkos::parallel_for(
            "hydro::positions", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                for (int a = 0; a < 3; ++a) x(i, a) = from(i, a);
            });
    }

    // The mesh's current arrays (an update may reallocate them).
    void refresh() {
        const Mesh& mesh = *mesh_;
        faces_ = mesh.faces();
        geometry_ = {mesh.owner(),         mesh.neighbour(), mesh.cell_face_start(), mesh.cell_face(), mesh.area(),
                     mesh.face_centroid(), mesh.centroid(),  mesh.volume(),          mesh.positions(), mesh.box()};
    }

    // Gradients, then face fluxes, in the mesh's scratch memory.
    void borrow_scratch() {
        const std::size_t gradient_bytes = order_ == 1 ? sizeof(double) * 15 * std::size_t(n_) : 0;
        const std::size_t flux_bytes = update_ == Update::per_face ? sizeof(double) * 5 * std::size_t(faces_) : 0;
        char* base = mesh_->scratch(gradient_bytes + flux_bytes).data();
        if (order_ == 1) gradients_ = Gradients(reinterpret_cast<double*>(base), n_);
        if (update_ == Update::per_face) flux_ = FaceFluxes(reinterpret_cast<double*>(base + gradient_bytes), faces_);
    }

    // The point velocities, AREPO-ng's set_vertex_velocities
    // (src/voronoi/voronoi_vertex_velocities.cc:26-139 with
    // REGULARIZE_MESH_CM_DRIFT, _USE_SOUNDSPEED, _FACE_ANGLE): the fluid
    // velocity (:31), plus, for a cell whose largest face angle
    // sqrt(A / pi) / (d_ij / 2) (src/voronoi/cell.cc:563, 576-578) exceeds
    // 0.75 cell_max_angle_factor, a drift towards the cell's centroid at
    // fraction * c, fraction rising linearly from 0 to cell_shaping_speed
    // at cell_max_angle_factor (:100-106, :109-137; no gravity).
    void point_velocities_kernel() {
        const auto q = q_;
        const auto w = w_;
        const Geometry geo = geometry_;
        const Gas gas = gas_;
        Kokkos::parallel_for(
            "hydro::point_velocities", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                const State W = primitive(geo.per_volume(q, i), gas);
                double max_angle = 0;
                for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                    const int f = geo.face(k);
                    const Vec3 a = geo.row(geo.area, f), d = geo.separation(i, geo.other(f, i));
                    const double angle = Kokkos::sqrt(Kokkos::sqrt(dot(a, a)) / M_PI) / (0.5 * Kokkos::sqrt(dot(d, d)));
                    max_angle = Kokkos::fmax(max_angle, angle);
                }
                double fraction = 0;
                if (max_angle > 0.75 * cell_max_angle_factor)
                    fraction = max_angle > cell_max_angle_factor
                                   ? cell_shaping_speed
                                   : cell_shaping_speed * (max_angle - 0.75 * cell_max_angle_factor) / (0.25 * cell_max_angle_factor);
                Vec3 v{W[1], W[2], W[3]};
                const Vec3 c = geo.row(geo.centroid, i);  // centroid - generator = -dx of AREPO
                const double d = Kokkos::sqrt(dot(c, c));
                if (d > 0 && fraction > 0) v = v + c * (fraction * sound_speed(W, gas) / d);
                w(i, 0) = v.x;
                w(i, 1) = v.y;
                w(i, 2) = v.z;
            });
    }

    // The points drift by dt * w and stay in the box; the mesh is updated.
    void move(double dt, Timings* timings) {
        const auto x = x_;
        const auto w = w_;
        const double box = mesh_->box();
        Kokkos::parallel_for(
            "hydro::drift", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                for (int a = 0; a < 3; ++a) {
                    const double moved = x(i, a) + dt * w(i, a);
                    x(i, a) = moved - box * Kokkos::floor(moved / box);
                }
            });
        phase_end(timings, &Timings::drift);
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
        const auto q = q_;
        const auto g = gradients_;
        const Geometry geo = geometry_;
        const Gas gas = gas_;
        const Limiter limiter = limiter_;
        Kokkos::parallel_for(
            "hydro::gradients", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                const State Wi = primitive(geo.per_volume(q, i), gas);
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
                        for (int r = p; r < 4; ++r) A[p][r] += w * X[p] * X[r];
                    const State Wj = primitive(geo.per_volume(q, j), gas);
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
                    for (int r = 0; r <= p; ++r) {
                        double sum = A[r][p];
                        for (int m = 0; m < r; ++m) sum -= L[p][m] * L[r][m];
                        L[p][r] = p == r ? Kokkos::sqrt(sum) : sum / L[r][r];
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

    // The flux through face f (times its area, from owner to neighbour);
    // `fallbacks` counts the sides that took the cell's state.
    template <int Order, bool Moving>
    struct FaceFlux {
        Cells q;
        Gradients g;
        Points w;
        Geometry geo;
        Gas gas;
        double dt;

        // As AREPO's prepare_state (mesh_fluxes.cc:451-479): the time and
        // space extrapolations are added only if rho and P stay positive;
        // otherwise the cell's own state is used. The space extrapolation
        // goes to where the face centroid is at the half step,
        // r + dt/2 w_face: the half step in the frame of the moving cell
        // (see the notes; AREPO-ng shifts its extrapolation the same way,
        // mesh_fluxes.cc:441).
        KOKKOS_INLINE_FUNCTION State side(int i, Vec3 r, int& fallbacks) const {
            const State W = primitive(geo.per_volume(q, i), gas);
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
        // The face velocity of AREPO-ng's prepare_face_velocity
        // (mesh_fluxes.cc:294-331): the mean of the two point velocities
        // plus, along the generator direction n, (c.(w_L - w_R)) / |d| with
        // c the face centroid relative to the generators' midpoint, |c|
        // limited to 0.9 |d|.
        KOKKOS_INLINE_FUNCTION Vec3 face_velocity(int f, int o, int j) const {
            const Vec3 wL{w(o, 0), w(o, 1), w(o, 2)}, wR{w(j, 0), w(j, 1), w(j, 2)};
            const Vec3 d = geo.separation(o, j);
            const double dist = Kokkos::sqrt(dot(d, d));
            const Vec3 c = geo.row(geo.face_centroid, f) - d * 0.5;
            double facv = dot(c, wL - wR) / dist;
            const double cc = Kokkos::sqrt(dot(c, c));
            if (cc > 0.9 * dist) facv *= 0.9 * dist / cc;
            return (wL + wR) * 0.5 + d * (facv / dist);
        }
        KOKKOS_INLINE_FUNCTION State operator()(int f, int& fallbacks) const {
            const int o = geo.owner(f), j = geo.neighbour(f);
            const Vec3 a = geo.row(geo.area, f);
            const double magnitude = Kokkos::sqrt(dot(a, a));
            Vec3 wf{0, 0, 0};
            if constexpr (Moving) wf = face_velocity(f, o, j);
            State L, R;
            if constexpr (Order == 0) {
                L = side(o, {0, 0, 0}, fallbacks);
                R = side(j, {0, 0, 0}, fallbacks);
            } else {
                const Vec3 shift = wf * (0.5 * dt);
                L = side(o, geo.face_offset(f, o) + shift, fallbacks);
                R = side(j, geo.face_offset(f, j) + shift, fallbacks);
            }
            State F = Riemann::flux(L, R, a * (1.0 / magnitude), wf, gas);
            for (int v = 0; v < 5; ++v) F[v] *= magnitude;
            return F;
        }
    };

    template <int Order>
    void advance(double dt, Timings* timings) {
        if (motion_ == Motion::moving)
            advance<Order, true>(dt, timings);
        else
            advance<Order, false>(dt, timings);
    }

    template <int Order, bool Moving>
    void advance(double dt, Timings* timings) {
        const FaceFlux<Order, Moving> flux{q_, gradients_, w_, geometry_, gas_, dt};
        const Geometry geo = geometry_;
        if (update_ == Update::per_cell) {
            const auto q = q_, next = next_;
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
                    for (int v = 0; v < 5; ++v) next(i, v) = q(i, v) - dt * sum[v];
                });
            fallbacks_ = -1;
            phase_end(timings, &Timings::fluxes);
            std::swap(q_, next_);
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
        const auto q = q_;
        Kokkos::parallel_for(
            "hydro::update", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), KOKKOS_LAMBDA(int i) {
                State sum{0, 0, 0, 0, 0};
                for (int k = geo.start(i); k < geo.start(i + 1); ++k) {
                    const int f = geo.face(k);
                    const double sign = geo.owner(f) == i ? 1.0 : -1.0;
                    for (int v = 0; v < 5; ++v) sum[v] += sign * F(f, v);
                }
                for (int v = 0; v < 5; ++v) q(i, v) = q(i, v) - dt * sum[v];
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
    long long fallbacks_ = 0;
    Kokkos::Timer timer_;
    Cells q_, next_;
    Points x_, w_;
    Gradients gradients_;
    FaceFluxes flux_;
    Geometry geometry_;
    MeshStats mesh_stats_;
};

}  // namespace hydro
