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
// rounding. A static step is two launches, time_step() and step(), a moving one
// three with the mesh's order check; a sort runs only when points crossed.
// step_and_time_step() fuses them into one for LUNA's rule, the static default.
// AREPO's rule, the moving default, cannot be fused: its pair term reads the
// neighbours' next state. Fusing only the order check into step() was
// measured and is 2-6 % slower (research_notes/stage1_1d_2026-11-04.md).
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
// - moving: the point velocity is the fluid velocity plus AREPO-ng's
//   regularisation. A face in 1D has no extent, so the face angle of
//   finite_volume.hpp does not exist; the criterion is AREPO-ng's other
//   branch (Arepo2-nextgeneration, set_vertex_velocities, !REGULARIZE_MESH_FACE_ANGLE,
//   with CellShapingFactor 1 and CellShapingSpeed 0.5 of its examples): a
//   drift at fraction * c towards the cell's centroid, displaced towards
//   higher density, fraction rising from 0 at an offset of 0.75 r to 0.5 at
//   r, r = V / 2. It keeps generators from crossing, which would swap the two
//   cells' volumes under their masses (research_notes/stage1_1d_2026-11-04.md).
//   The face velocity is the mean of the two point velocities: AREPO-ng's
//   prepare_face_velocity (mesh_fluxes.cc:294-331) with the face centroid at
//   the generators' midpoint.
// - walls (a mesh with Ends::walls): a wall face is static and takes a ghost
//   state for the outside, LUNA's ghost cells on a uniform line
//   (halo_local.py:16-31, the wide step of muscl_hancock.py:246-247):
//   reflective, the mirror of the cell's face state (the ghost's stencil is
//   the mirror of the cell's); zero gradient, the cell's W (two equal ghosts,
//   a flat stencil, no half step); inflow, a fixed state, which LUNA does not
//   have. The ghost is also the cell's outer neighbour in its gradient, at
//   the mirrored centroid.
namespace hydro {

// The boundary of a wall of the 1D line.
enum class Wall { reflective, outflow, inflow };
struct Walls {
    Wall left = Wall::outflow, right = Wall::outflow;
    State<1> left_inflow{}, right_inflow{};  // primitives (rho, v, P) of an inflow
};

// The global time step's rule.
// arepo: AREPO next generation's, as finite_volume.hpp, time_step() (lines of
// Arepo2-nextgeneration-SoA): courant * min_i min(r_i / (c_i + |v_i - w_i|),
// min_j (d_ij + r_i) / vsig_ij), with |v_i| on a static mesh (timestep.cc:156-165,
// sph_particle_data.h:305-334), vsig_ij = c_i + c_j - (v_j - v_i).d_ij / d_ij > 0
// in the fluid velocities (timestep_treebased.cc:81-88). AREPO-ng has no 1D
// mesh, so three points are this code's reading in 1D: r_i = V_i / 2, the radius
// of a 1D ball of the cell's volume, as its 2D and 3D branches are of theirs
// (sph_particle_data.h:280-287 falls through to the cube root for ONEDIMS);
// j runs over the two neighbours, not every point a signal reaches (as the
// 3D code, finite_volume.hpp); and a wall gives no pair term, there being no
// point behind it. research_notes/stage1_1d_2026-11-04.md has what each costs.
// luna: courant * min_i V_i / (|v_i - w_i| + c_i), the RAMSES rule of LUNA's
// integrators.py:11-13 with dx the cell's length, in the frame of the point.
enum class TimeStep { arepo, luna };

// The defaults of a 1D run (owner, 2026-11-05): static, LUNA's rule at 0.8,
// which reproduces LUNA; moving, AREPO's rule at 0.4. Not 3D's 0.3.
inline constexpr double courant_static_1d = 0.8, courant_moving_1d = 0.4;

template <class ExecSpace, class Riemann>
class FiniteVolume<1, ExecSpace, Riemann, Precision<>> {
public:
    static constexpr int D = 1, V = 3;
    static constexpr double cell_shaping_speed = 0.5;
    using Memory = typename ExecSpace::memory_space;
    using Mesh = voronoi::Mesh<1, ExecSpace>;
    using State = hydro::State<1>;
    using Cells = Kokkos::View<double* [V], Kokkos::LayoutRight, Memory>;
    using Points = Kokkos::View<double* [1], Kokkos::LayoutRight, Memory>;
    using Line = typename Mesh::Line;
    using i64 = exact_predicates::i64;
    using Keys = Kokkos::View<sorted_mesh::Key*, Memory>;

    // mesh: built (update()). Walls must be given exactly when the mesh has
    // them. Moving: this object owns the positions (starting from the
    // mesh's snapped ones) and updates the mesh's order.
    FiniteVolume(const ExecSpace& exec, Mesh& mesh, const Gas& gas, int order, Limiter limiter = Limiter::face_centroid,
                 Motion motion = Motion::static_mesh, const Walls& walls = {}, bool regularise = true)
        : exec_(exec), mesh_(&mesh), gas_(gas), order_(order), limiter_(limiter), motion_(motion), walls_(walls),
          regularise_(regularise), n_(mesh.particles()) {
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
        bool regularise;

        // A cell of a stencil: its primitives, its centroid relative to the
        // generator; real = false for a ghost behind a wall.
        struct Cell {
            State W{};
            double c = 0, V = 0;
            bool real = true;
        };

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
        // The ghost cell behind wall `left` of the end slot whose state is W.
        KOKKOS_INLINE_FUNCTION State ghost(const State& W, bool left) const {
            const Wall wall = left ? walls.left : walls.right;
            if (wall == Wall::reflective) return mirror(W);
            if (wall == Wall::inflow) return left ? walls.left_inflow : walls.right_inflow;
            return W;
        }
        // The ghost's face state at the wall, for an end cell of state W
        // and face state Wf. Reflective: the mirror of Wf, the ghost's stencil
        // being the mirror of the cell's. Zero gradient and inflow: the
        // ghost's own W, its two-cell stencil being flat, so its slope and
        // half step vanish (LUNA's wide step, muscl_hancock.py:246-247 on
        // halo_local.py:16-17, BOUNDARY 1: mode "edge").
        KOKKOS_INLINE_FUNCTION State ghost_face(const State& Wf, const State& W, bool left) const {
            const Wall wall = left ? walls.left : walls.right;
            if (wall == Wall::reflective) return mirror(Wf);
            return ghost(W, left);
        }

        // Cells s - R .. s + R into cell[], outside a wall the ghost of the
        // end slot (two past the wall: never used), and the distances between
        // their generators into gap[] (to a mirror: twice the wall's).
        template <int R>
        KOKKOS_INLINE_FUNCTION void load(int s, Cell (&cell)[2 * R + 1], double (&gap)[2 * R]) const {
            constexpr int M = 2 * R + 1;
            int slot[M];
            for (int k = 0; k < M; ++k) slot[k] = line.slot(s, k - R);
            for (int k = 0; k < M; ++k) {
                if (slot[k] >= 0) {
                    cell[k].W = primitive_of(slot[k]);
                    cell[k].c = line.centroid(slot[k]);
                    cell[k].V = line.volume(slot[k]);
                }
            }
            for (int k = 0; k < M; ++k) {
                if (slot[k] >= 0) continue;
                cell[k].real = false;
                const int end = k < R ? k + 1 : k - 1;  // the slot towards s
                if (slot[end] < 0) continue;
                cell[k].W = ghost(cell[end].W, k < R);
                cell[k].c = -cell[end].c;  // the mirror's centroid
            }
            for (int k = 0; k < M - 1; ++k) {
                const int a = slot[k], b = slot[k + 1];
                gap[k] = a >= 0 ? length(a) : b >= 0 ? length(-1) : 0.0;
            }
        }

        // load<R> far from the ends of the arrays (R + 1 <= s <= n - R - 2):
        // the same bits from whole-step gaps read once, each key once.
        template <int R>
        KOKKOS_INLINE_FUNCTION void load_inner(int s, Cell (&cell)[2 * R + 1], double (&gap)[2 * R]) const {
            constexpr int M = 2 * R + 1;
            double steps[M + 1];  // steps[k]: grid steps from slot s - R - 1 + k to the next
            i64 previous = static_cast<i64>(line.key(s - R - 1));
            for (int k = 0; k <= M; ++k) {
                const i64 here = static_cast<i64>(line.key(s - R + k));
                const i64 d = here - previous;
                steps[k] = static_cast<double>(d < 0 ? d + line.period : d);  // a periodic line's seam
                previous = here;
            }
            for (int k = 0; k < M; ++k) {
                const int i = line.order(s - R + k);
                cell[k].V = (steps[k] + steps[k + 1]) * (0.5 * line.step);
                const double inv = 1.0 / cell[k].V;
                cell[k].W = primitive(State{q(i, 0) * inv, q(i, 1) * inv, q(i, 2) * inv}, gas);
                cell[k].c = (steps[k + 1] - steps[k]) * (0.25 * line.step);
            }
            for (int k = 0; k < M - 1; ++k) gap[k] = steps[k + 1] * line.step;
        }

        // The limited gradient of variable v of the cell at *cell (see the
        // head of this file), its neighbours at distances gapL and gapR.
        KOKKOS_INLINE_FUNCTION double gradient(const Cell* cell, double gapL, double gapR, int v) const {
            const Cell& me = *cell;
            const double dL = -gapL + cell[-1].c - me.c, dR = gapR + cell[1].c - me.c;
            const double Wi = me.W[v], left = cell[-1].W[v], right = cell[1].W[v];
            double g = ((right - Wi) - (left - Wi)) / (dR - dL);
            if (limiter == Limiter::none) return g;
            const bool faces = limiter == Limiter::face_centroid;
            const double r[2] = {faces ? -0.5 * gapL - me.c : dL, faces ? 0.5 * gapR - me.c : dR};
            const double lo = Kokkos::fmin(left, right), hi = Kokkos::fmax(left, right);
            for (int f = 0; f < 2; ++f) {
                const double dp = g * r[f];
                if (dp > 0 && Wi + dp > hi)
                    g = g * (hi > Wi ? (hi - Wi) / dp : 0.0);
                else if (dp < 0 && Wi + dp < lo)
                    g = g * (lo < Wi ? (lo - Wi) / dp : 0.0);
            }
            return g;
        }

        // The point velocity's difference from the fluid's, AREPO-ng's
        // regularisation (set_vertex_velocities, voronoi_vertex_velocities.cc), for a cell of
        // primitives W, limited density gradient g, centroid c relative to the
        // generator and volume `volume`. Below 0.5 r the displaced centre
        // stays below the threshold of 0.75 r.
        KOKKOS_INLINE_FUNCTION double drift(const State& W, double g, double c, double volume) const {
            const double r = 0.5 * volume;
            if (!regularise || Kokkos::fabs(c) <= 0.5 * r) return 0.0;
            double offset = c;
            const double dgrad = Kokkos::fabs(g);
            if (dgrad > 0) {
                const double tmp = 3 * r + W[0] / dgrad;
                const double x = (tmp - Kokkos::sqrt(tmp * tmp - 8 * r * r)) / 4;
                if (x < 0.25 * r) offset += x / dgrad * g;
            }
            const double d = Kokkos::fabs(offset);
            if (d <= 0.75 * r) return 0.0;
            const double fraction = d > r ? cell_shaping_speed : cell_shaping_speed * (d - 0.75 * r) / (0.25 * r);
            return offset * (fraction * sound_speed(W, gas) / d);
        }
    };

    // The global time step (see TimeStep). |v - w| is the regularisation's.
    double time_step(double courant, TimeStep rule = TimeStep::arepo, Timings* timings = nullptr) {
        timer_.reset();
        const Stencil st = stencil();
        const bool moving = motion_ == Motion::moving, luna = rule == TimeStep::luna;
        const bool second_order = order_ == 1;
        double dt = 0;
        Kokkos::parallel_reduce(
            "hydro1::time_step", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_),
            KOKKOS_LAMBDA(int s, double& smallest) {
                typename Stencil::Cell cell[3];
                double gap[2] = {0, 0};
                if (moving && s >= 2 && s <= st.line.n - 3)
                    st.template load_inner<1>(s, cell, gap);
                else if (moving)
                    st.template load<1>(s, cell, gap);
                else
                    cell[1].W = st.primitive_of(s);
                const State& Wi = cell[1].W;
                const double ci = sound_speed(Wi, st.gas);
                const double volume = st.line.volume(s);
                double vi = Kokkos::fabs(Wi[1]);
                if (moving) {
                    const double g = second_order ? st.gradient(cell + 1, gap[0], gap[1], 0) : 0.0;
                    vi = Kokkos::fabs(st.drift(Wi, g, cell[1].c, volume));
                }
                if (luna) {
                    smallest = Kokkos::fmin(smallest, courant * volume / (vi + ci));
                    return;
                }
                const double r = 0.5 * volume;  // the half length of the cell
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
        phase_end(timings, &Timings::sweep);
        return dt;
    }

    // One step of length dt (moving: the points then drift and the mesh's
    // order is updated).
    void step(double dt, Timings* timings = nullptr) {
        timer_.reset();
        if (order_ == 1)
            advance<1>(dt);
        else
            advance<0>(dt);
        phase_end(timings, &Timings::fluxes);
        std::swap(q_, next_);
        if (motion_ == Motion::moving) {
            const voronoi::Report report = mesh_->order(x_);
            mesh_stats_.updates += 1;
            mesh_stats_.rebuilds += report.rebuilt;
            phase_end(timings, &Timings::mesh);
        }
    }

    // One step of length dt in one launch (see the head of this file);
    // returns the time step of the next one by LUNA's rule at this Courant
    // number. Static: time_step(courant, luna) after step(dt), to rounding.
    double step_and_time_step(double dt, double courant) {
        if (x_.extent(0) > 0 && x_next_.extent(0) == 0)
            x_next_ = Points(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("hydro_x_next")), n_);
        const bool moving = motion_ == Motion::moving;
        if (order_ == 1) return moving ? run_fused<1, true>(dt, courant) : run_fused<1, false>(dt, courant);
        return moving ? run_fused<0, true>(dt, courant) : run_fused<0, false>(dt, courant);
    }

    // Ends a timed phase (fenced) when timings are wanted.
    void phase_end(Timings* timings, double Timings::*part) {
        if (!timings) return;
        exec_.fence();
        timings->*part += timer_.seconds() * 1e3;
        timer_.reset();
    }

    // The fused step, one thread per slot: public only because CUDA device
    // lambdas may not be defined in private member functions.
    // Fused: also the next time step (LUNA's rule) from the new state, on the
    // moved mesh, with the regularisation's |v - w| of this step, and the
    // next keys and a count of the slots out of order (moving).
    template <int Order, bool Moving, bool Fused>
    struct Kernel {
        Stencil st;
        Cells next;
        Points x, x_next;
        Keys keys;
        sorted_mesh::Grid grid;
        double dt, courant;

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
            double smallest = 0;
            int descents = 0;
            (*this)(s, smallest, descents);
        }
        KOKKOS_INLINE_FUNCTION void operator()(int s, double& smallest, int& descents) const {
            if (s >= 3 && s <= st.line.n - 4)
                body<true>(s, smallest, descents);
            else
                body<false>(s, smallest, descents);
        }

        // The point at x0 after this step.
        KOKKOS_INLINE_FUNCTION double moved(double x0, double velocity) const {
            const double side = st.line.side();
            const double y = x0 + dt * velocity;
            return st.line.periodic ? y - side * Kokkos::floor(y / side) : Kokkos::fmin(Kokkos::fmax(y, 0.0), side);
        }

        // Inner: no ghost in reach, nothing to wrap.
        template <bool Inner>
        KOKKOS_INLINE_FUNCTION void body(int s, double& smallest, int& descents) const {
            const Line& line = st.line;
            typename Stencil::Cell cell[5];
            double gap[4];
            if constexpr (Inner)
                st.template load_inner<2>(s, cell, gap);
            else
                st.template load<2>(s, cell, gap);
            // Gradients of slots s - 1, s, s + 1 (k = 1, 2, 3) and their point velocities.
            State grad[5];
            double point[5] = {0, 0, 0, 0, 0};
            for (int k = 1; k <= 3; ++k) {
                for (int v = 0; v < V; ++v) grad[k][v] = 0;
                if (!Inner && !cell[k].real) continue;
                if constexpr (Order == 1)
                    for (int v = 0; v < V; ++v) grad[k][v] = st.gradient(cell + k, gap[k - 1], gap[k], v);
                if constexpr (Moving)
                    point[k] = cell[k].W[1] + st.drift(cell[k].W, grad[k][0], cell[k].c, 0.5 * (gap[k - 1] + gap[k]));
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
                    if (Inner || (cell[a].real && cell[b].real)) w = 0.5 * (point[a] + point[b]);
                const double shift = 0.5 * dt * w;
                State L, R;
                if (Inner || cell[a].real) L = face_state(cell[a].W, grad[a], 0.5 * gap[a] - cell[a].c + shift);
                if (Inner || cell[b].real) R = face_state(cell[b].W, grad[b], -0.5 * gap[a] - cell[b].c + shift);
                if constexpr (!Inner) {
                    if (!cell[a].real) L = st.ghost_face(R, cell[b].W, true);
                    if (!cell[b].real) R = st.ghost_face(L, cell[a].W, false);
                }
                F[f] = Riemann::flux(L, R, Vec<1>{1.0}, Vec<1>{w}, st.gas);
            }
            const int i = line.order(s);
            State Unew;
            for (int v = 0; v < V; ++v) {
                Unew[v] = st.q(i, v) - dt * (F[1][v] - F[0][v]);
                next(i, v) = Unew[v];
            }
            if constexpr (Moving && !Fused) x(i, 0) = moved(x(i, 0), point[2]);
            if constexpr (Fused) {
                double volume = cell[2].V, relative = 0;
                if constexpr (Moving) {
                    using Key = sorted_mesh::Key;
                    const double here = moved(x(i, 0), point[2]);
                    x_next(i, 0) = here;
                    const Key key = sorted_mesh::snap(here, grid, line.periodic);
                    keys(s) = key;
                    const auto next_key = [&](int slot, int k) {
                        return sorted_mesh::snap(moved(x(line.order(slot), 0), point[k]), grid, line.periodic);
                    };
                    double left, right;
                    if (Inner || cell[1].real) {
                        const Key other = next_key(Inner ? s - 1 : line.slot(s, -1), 1);
                        const i64 d = static_cast<i64>(key) - static_cast<i64>(other);
                        left = static_cast<double>(d < 0 && line.periodic ? d + line.period : d);
                    } else {
                        left = static_cast<double>(2 * static_cast<i64>(key));
                    }
                    if (Inner || cell[3].real) {
                        const int t = Inner ? s + 1 : line.slot(s, 1);
                        const Key other = next_key(t, 3);
                        const i64 d = static_cast<i64>(other) - static_cast<i64>(key);
                        right = static_cast<double>(d < 0 && line.periodic ? d + line.period : d);
                        descents += !sorted_mesh::before(key, i, other, line.order(t));
                    } else {
                        right = static_cast<double>(2 * (line.period - static_cast<i64>(key)));
                    }
                    volume = (left + right) * (0.5 * line.step);
                    relative = Kokkos::fabs(point[2] - cell[2].W[1]);
                }
                const double inv = 1.0 / volume;
                const State Wnew = primitive(State{Unew[0] * inv, Unew[1] * inv, Unew[2] * inv}, st.gas);
                if constexpr (!Moving) relative = Kokkos::fabs(Wnew[1]);
                smallest = Kokkos::fmin(smallest, courant * volume / (relative + sound_speed(Wnew, st.gas)));
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
        const Kernel<Order, Moving, false> kernel{stencil(), next_, x_, x_next_, {}, {}, dt, 0.0};
        Kokkos::parallel_for("hydro1::step", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), kernel);
    }
    template <int Order, bool Moving>
    double run_fused(double dt, double courant) {
        const Line& line = mesh_->line();
        sorted_mesh::Grid grid;
        grid.side = line.side();
        grid.step = line.step;
        const Kernel<Order, Moving, true> kernel{stencil(), next_, x_, x_next_, mesh_->spare_keys(), grid, dt, courant};
        double smallest = 0;
        int descents = 0;
        Kokkos::parallel_reduce("hydro1::step_fused", Kokkos::RangePolicy<ExecSpace>(exec_, 0, n_), kernel,
                                Kokkos::Min<double>(smallest), Kokkos::Sum<int>(descents));
        std::swap(q_, next_);
        if constexpr (Moving) {
            std::swap(x_, x_next_);
            mesh_->swap_keys();
            mesh_stats_.updates += 1;
            if (descents != (line.periodic ? 1 : 0)) {
                mesh_stats_.rebuilds += mesh_->order(x_).rebuilt;
                return time_step(courant, TimeStep::luna);
            }
        }
        return smallest;
    }

private:
    Stencil stencil() const { return {mesh_->line(), q_, gas_, walls_, limiter_, regularise_}; }

    ExecSpace exec_;
    Mesh* mesh_;
    Gas gas_;
    int order_;
    Limiter limiter_;
    Motion motion_;
    Walls walls_;
    bool regularise_;
    int n_;
    Cells q_, next_;
    Points x_, x_next_;
    MeshStats mesh_stats_;
    Kokkos::Timer timer_;
};

}  // namespace hydro
