#pragma once

#include "finite_volume.hpp"

// Finite volumes on the 1D mesh (voronoi::Mesh<1>), static or moving: the
// scheme of FiniteVolume<D> (finite_volume.hpp) in one dimension, with
// walls, in one fused kernel per step (owner's decision 2,
// research_notes/dimension_design_2026-11-02.md).
//
// On the sorted line a cell's neighbours are the slots s - 1 and s + 1, so
// one thread per slot reads the five cells s - 2 .. s + 2 and computes, in
// registers, the limited gradients of s - 1, s, s + 1, the four face states
// and the fluxes through its two faces, and writes its new state (moving:
// and drifts its point). Each face is computed by both of its cells from the
// same data in the same frame (every distance is an integer of the line,
// sorted_mesh::Line), so both get the same bits and the update conserves to
// rounding. A static step is two launches (time step, step), a moving one
// four (time step, step, the mesh's order check, and a sort only when
// points crossed).
//
// Scheme, as finite_volume.hpp unless said:
// - gradients: AREPO-ng's least-squares fit with an intercept over the face
//   neighbours, weighted by the face area (mesh_gradients.cc; lines in the
//   stage 1 notes). In 1D the two neighbours with weight 1 determine the
//   fit exactly: g = (dW_R - dW_L) / (d_R - d_L), d the separations of the
//   cell centroids. The limiter is AREPO-ng's (limit_gradient,
//   mesh_gradients.cc:120-148): scaled face by face (left, then right)
//   until the values extrapolated to the face centroids (or the neighbours'
//   centroids) stay within the neighbours' range. On a uniform line this is
//   the MC limiter of LUNA (slopes.py:22-24) on the central difference.
// - the MUSCL-Hancock half step and HLLC/HLL in 1D (euler.hpp), each side's
//   face state the cell's if rho or P would not stay positive.
// - moving: the point velocity is the fluid velocity. AREPO's 1D mesh has
//   no face angle (voronoi_1d.c computes none), so the regularisation of
//   finite_volume.hpp (a drift when the largest face angle exceeds its
//   threshold) never acts in 1D. The face velocity is the mean of the two
//   point velocities: AREPO-ng's prepare_face_velocity (mesh_fluxes.cc:
//   294-331) with the face centroid at the generators' midpoint.
// - walls (a mesh with Ends::walls): each wall face takes a ghost state for
//   the outside: reflective, the mirror of the cell's face state (normal
//   velocity negated); outflow (zero gradient), the cell's face state;
//   inflow, a fixed state. The ghost is also the cell's outer neighbour in
//   its gradient, at the mirrored centroid (AREPO's mirror ghosts,
//   voronoi_1d.c:130-133, with the velocity flipped for reflection). On a
//   uniform line these are LUNA's ghost cells of BOUNDARY 2 and 1
//   (halo_local.py:16-31): its ghost's slope is the mirror of (reflective)
//   or zero like (zero gradient) the edge cell's.
namespace hydro {

// The boundary of a wall of the 1D line.
enum class Wall { reflective, outflow, inflow };
struct Walls {
    Wall left = Wall::outflow, right = Wall::outflow;
    State<1> left_inflow{}, right_inflow{};  // primitives (rho, v, P) of an inflow
};

// The global time step's rule: AREPO next generation's (finite_volume.hpp,
// time_step()), or LUNA's, courant * min_i V_i / (|v_i - w_i| + c_i) (the
// RAMSES rule of LUNA's integrators.py:11-13 with dx the cell's length, in
// the frame of the moving point).
enum class TimeStep { arepo, luna };

template <class ExecSpace, class Riemann>
class FiniteVolume<1, ExecSpace, Riemann> {
public:
    static constexpr int D = 1, V = 3;
    using Memory = typename ExecSpace::memory_space;
    using Mesh = voronoi::Mesh<1, ExecSpace>;
    using State = hydro::State<1>;
    using Cells = Kokkos::View<double* [V], Memory>;
    using Points = Kokkos::View<double* [1], Memory>;
    using Line = typename Mesh::Line;

    // mesh: built (update()). Walls must be given exactly when the mesh has
    // them. Moving: this object owns the positions (starting from the
    // mesh's snapped ones) and updates the mesh's order.
    FiniteVolume(const ExecSpace& exec, Mesh& mesh, const Gas& gas, int order, Limiter limiter = Limiter::face_centroid,
                 Motion motion = Motion::static_mesh, const Walls& walls = {})
        : exec_(exec), mesh_(&mesh), gas_(gas), order_(order), limiter_(limiter), motion_(motion), walls_(walls),
          n_(mesh.particles()) {
        if (order != 0 && order != 1) throw std::invalid_argument("hydro::FiniteVolume<1>: order 0 or 1");
        q_ = Cells("hydro_Q", n_);
        next_ = Cells(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_Q_next")), n_);
        if (motion == Motion::moving) {
            x_ = Points(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_x")), n_);
            Kokkos::deep_copy(exec_, x_, mesh.positions());
        }
    }

    Cells state() const { return q_; }
    Points positions() const { return x_; }
    int order() const { return order_; }
    Motion motion() const { return motion_; }
    const Gas& gas() const { return gas_; }
    const MeshStats& mesh_stats() const { return mesh_stats_; }
    long long fallbacks() const { return -1; }  // not counted (as the per-cell loop of finite_volume.hpp)

    // The cells and their neighbours along the line, for the kernels.
    struct Stencil {
        Line line;
        Cells q;
        Gas gas;
        Walls walls;
        Limiter limiter;

        KOKKOS_INLINE_FUNCTION double length(int s) const { return static_cast<double>(line.gap(s)) * line.step; }
        KOKKOS_INLINE_FUNCTION State primitive_of(int s) const {
            const int i = line.order(s);
            const double inv = 1.0 / line.volume(s);
            return primitive(State{q(i, 0) * inv, q(i, 1) * inv, q(i, 2) * inv}, gas);
        }
        KOKKOS_INLINE_FUNCTION static State mirror(State W) {
            W[1] = -W[1];
            return W;
        }
        // The ghost behind wall `left` of the end slot whose state is W.
        KOKKOS_INLINE_FUNCTION State ghost(const State& W, bool left) const {
            const Wall wall = left ? walls.left : walls.right;
            if (wall == Wall::reflective) return mirror(W);
            if (wall == Wall::inflow) return left ? walls.left_inflow : walls.right_inflow;
            return W;
        }
    };

    // The global time step (see TimeStep). Moving: point velocities are the
    // fluid's, so |v - w| = 0.
    double time_step(double courant, TimeStep rule = TimeStep::arepo) {
        const Stencil st = stencil();
        const bool moving = motion_ == Motion::moving, luna = rule == TimeStep::luna;
        double dt = 0;
        Kokkos::parallel_reduce(
            "hydro1::time_step", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_),
            KOKKOS_LAMBDA(int s, double& smallest) {
                const State Wi = st.primitive_of(s);
                const double ci = sound_speed(Wi, st.gas), vi = moving ? 0.0 : Kokkos::fabs(Wi[1]);
                const double volume = st.line.volume(s);
                if (luna) {
                    smallest = Kokkos::fmin(smallest, courant * volume / (vi + ci));
                    return;
                }
                const double r = 0.5 * volume;  // AREPO's get_cell_radius in 1D (voronoi.c:1061-1063)
                double step = r / (ci + vi);
                for (int side = -1; side <= 1; side += 2) {
                    const int t = st.line.slot(s, side);
                    if (t < 0) continue;
                    const State Wj = st.primitive_of(t);
                    const double dist = st.length(side < 0 ? s - 1 : s);
                    const double vsig = ci + sound_speed(Wj, st.gas) - (Wj[1] - Wi[1]) * side;
                    if (vsig > 0) step = Kokkos::fmin(step, (dist + r) / vsig);
                }
                smallest = Kokkos::fmin(smallest, courant * step);
            },
            Kokkos::Min<double>(dt));
        return dt;
    }

    // One step of length dt (moving: the points then drift and the mesh's
    // order is updated).
    void step(double dt) {
        if (order_ == 1)
            advance<1>(dt);
        else
            advance<0>(dt);
        std::swap(q_, next_);
        if (motion_ == Motion::moving) {
            const voronoi::Report report = mesh_->order(x_);
            mesh_stats_.updates += 1;
            mesh_stats_.rebuilds += report.rebuilt;
        }
    }

    // The fused step, one thread per slot: public only because CUDA device
    // lambdas may not be defined in private member functions.
    template <int Order, bool Moving>
    struct Kernel {
        Stencil st;
        Cells next;
        Points x;
        double dt;

        // The state of slot t (s + k for k in -2 .. 2, t may be a ghost: -1
        // or n for walls; ghost(t) of the slot next to it).
        struct Cell {
            State W;
            double c = 0;    // centroid relative to the generator
            bool real = true;
        };

        // A cell's state at a face at offset r from its centroid (moving:
        // where the face is at the half step).
        KOKKOS_INLINE_FUNCTION State face_state(const State& W, const State& g, double r) const {
            if constexpr (Order == 0) {
                (void)g;
                (void)r;
                return W;
            } else {
                Gradient<1> grad;
                for (int v = 0; v < V; ++v) grad[v][0] = g[v];
                State Wf = half_step(W, grad, dt, st.gas);
                for (int v = 0; v < V; ++v) Wf[v] += g[v] * r;
                if (Wf[0] > 0 && Wf[2] > 0) return Wf;
                return W;
            }
        }

        KOKKOS_INLINE_FUNCTION void operator()(int s) const {
            const Line& line = st.line;
            const bool walls = !line.periodic;
            // Slots s - 2 .. s + 2; outside a wall the ghost of the end slot.
            int slot[5];
            Cell cell[5];
            for (int k = 0; k < 5; ++k) slot[k] = line.slot(s, k - 2);
            for (int k = 0; k < 5; ++k) {
                if (slot[k] >= 0) {
                    cell[k].W = st.primitive_of(slot[k]);
                    cell[k].c = line.centroid(slot[k]);
                }
            }
            for (int k = 0; k < 5; ++k) {
                if (slot[k] >= 0) continue;
                cell[k].real = false;
                const int end = k < 2 ? k + 1 : k - 1;  // the slot towards s
                if (slot[end] < 0) continue;            // two past the wall: never used
                cell[k].W = st.ghost(cell[end].W, k < 2);
                cell[k].c = -cell[end].c;  // the mirror's centroid
            }
            // Generator-to-generator lengths between k and k + 1 (k = 0 .. 3),
            // walls: to the mirror, twice the distance to the wall.
            double gap[4];
            for (int k = 0; k < 4; ++k) {
                const int a = slot[k], b = slot[k + 1];
                gap[k] = a >= 0 ? st.length(a) : b >= 0 ? st.length(-1) : 0.0;
            }
            // Gradients of slots s - 1, s, s + 1 (k = 1, 2, 3).
            State grad[5];
            for (int k = 1; k <= 3; ++k) {
                for (int v = 0; v < V; ++v) grad[k][v] = 0;
                if constexpr (Order == 1) {
                    if (!cell[k].real) continue;
                    const double dL = -gap[k - 1] + cell[k - 1].c - cell[k].c, dR = gap[k] + cell[k + 1].c - cell[k].c;
                    const double rL = -0.5 * gap[k - 1] - cell[k].c, rR = 0.5 * gap[k] - cell[k].c;  // face offsets
                    const double r[2] = {st.limiter == Limiter::face_centroid ? rL : dL,
                                         st.limiter == Limiter::face_centroid ? rR : dR};
                    for (int v = 0; v < V; ++v) {
                        const double Wi = cell[k].W[v], left = cell[k - 1].W[v], right = cell[k + 1].W[v];
                        double g = ((right - Wi) - (left - Wi)) / (dR - dL);
                        if (st.limiter != Limiter::none) {
                            const double lo = Kokkos::fmin(left, right), hi = Kokkos::fmax(left, right);
                            for (int f = 0; f < 2; ++f) {
                                const double dp = g * r[f];
                                if (dp > 0 && Wi + dp > hi)
                                    g = g * (hi > Wi ? (hi - Wi) / dp : 0.0);
                                else if (dp < 0 && Wi + dp < lo)
                                    g = g * (lo < Wi ? (lo - Wi) / dp : 0.0);
                            }
                        }
                        grad[k][v] = g;
                    }
                }
            }
            // The fluxes through the faces left (k = 1 | 2) and right (2 | 3)
            // of s, positive along +x.
            State F[2];
            for (int f = 0; f < 2; ++f) {
                const int a = 1 + f, b = 2 + f;  // the cells left and right of the face
                // A wall face is static; a face between cells moves with
                // the mean of their point velocities.
                double w = 0;
                if constexpr (Moving)
                    if (cell[a].real && cell[b].real) w = 0.5 * (cell[a].W[1] + cell[b].W[1]);
                const double shift = 0.5 * dt * w;
                State L, R;
                if (cell[a].real) L = face_state(cell[a].W, grad[a], 0.5 * gap[a] - cell[a].c + shift);
                if (cell[b].real) R = face_state(cell[b].W, grad[b], -0.5 * gap[a] - cell[b].c + shift);
                if (!cell[a].real) L = st.ghost(R, true);
                if (!cell[b].real) R = st.ghost(L, false);
                F[f] = Riemann::flux(L, R, Vec<1>{1.0}, Vec<1>{w}, st.gas);
            }
            const int i = line.order(s);
            for (int v = 0; v < V; ++v) next(i, v) = st.q(i, v) - dt * (F[1][v] - F[0][v]);
            if constexpr (Moving) {
                const double side = static_cast<double>(line.period) * line.step;
                const double moved = x(i, 0) + dt * cell[2].W[1];
                x(i, 0) = walls ? Kokkos::fmin(Kokkos::fmax(moved, 0.0), side)
                                : moved - side * Kokkos::floor(moved / side);
            }
        }
    };

    template <int Order>
    void advance(double dt) {
        if (motion_ == Motion::moving)
            run<Order, true>(dt);
        else
            run<Order, false>(dt);
    }
    template <int Order, bool Moving>
    void run(double dt) {
        const Kernel<Order, Moving> kernel{stencil(), next_, x_, dt};
        Kokkos::parallel_for("hydro1::step", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), kernel);
    }

private:
    Stencil stencil() const { return {mesh_->line(), q_, gas_, walls_, limiter_}; }

    ExecSpace exec_;
    Mesh* mesh_;
    Gas gas_;
    int order_;
    Limiter limiter_;
    Motion motion_;
    Walls walls_;
    int n_;
    Cells q_, next_;
    Points x_;
    MeshStats mesh_stats_;
};

}  // namespace hydro
