// The 1D hydro (hydro::FiniteVolume<1>, finite_volume_1d.hpp) on the 1D
// mesh: a uniform state (with a bulk velocity on a periodic line; at rest
// between reflective walls) stays uniform, static and moving, to rounding;
// mass, momentum and energy of a rough state are conserved on a periodic
// line, and mass and energy between reflective walls (to rounding); two
// identical moving runs give the same bits; inflow keeps its state; Sod's
// tube (Toro test 1) against the exact solution's cell averages within
// LUNA's bounds at 128 and 256 cells (test_hydro_accuracy_shocktube.py:14),
// on static and moving meshes, uniform and jittered; LUNA's own final states
// of Sod, Shu-Osher and Woodward-Colella at N = 16 on a static uniform line
// (values printed by LUNA, walls included), to 1e-12; Shu-Osher on a moving
// mesh: the regularised points never cross, the unregularised ones do.

#include "check.hpp"
#include "exact_riemann.hpp"
#include "finite_volume_1d.hpp"
#include "luna_1d_values.hpp"
#include "problems_1d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Memory = Exec::memory_space;
using Mesh = voronoi::Mesh<1, Exec>;
using State = hydro::State<1>;
template <class Riemann = hydro::HLLC>
using FV = hydro::FiniteVolume<1, Exec, Riemann>;
using Primitives = std::function<State(double)>;  // W(x)

std::vector<double> points(int n, double L, double jitter, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> unit(-0.5, 0.5);
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i) x[i] = L * (i + 0.5 + jitter * unit(rng)) / n;
    return x;
}

void build(Mesh& mesh, const std::vector<double>& x) {
    Kokkos::View<double* [1], Memory> v("x", x.size());
    auto h = Kokkos::create_mirror_view(v);
    for (std::size_t i = 0; i < x.size(); ++i) h(i, 0) = x[i];
    Kokkos::deep_copy(v, h);
    mesh.update(v);
}

// Q of each cell from W at its centroid (smooth W) or cell averages.
template <class F>
void set(F& fv, Mesh& mesh, const std::function<State(double, double)>& extensive) {
    mesh.update_cells();
    const auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.positions());
    const auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.centroid());
    const auto V = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.volume());
    auto q = Kokkos::create_mirror_view(fv.state());
    for (int i = 0; i < mesh.particles(); ++i) {
        const State Q = extensive(x(i, 0) + c(i, 0) - 0.5 * V(i), x(i, 0) + c(i, 0) + 0.5 * V(i));
        for (int v = 0; v < 3; ++v) q(i, v) = Q[v];
    }
    Kokkos::deep_copy(fv.state(), q);
}
// From primitives at the centroid.
template <class F>
void set(F& fv, Mesh& mesh, const Primitives& W, const hydro::Gas& gas) {
    set(fv, mesh, [&](double a, double b) {
        const State U = hydro::conserved(W(0.5 * (a + b)), gas);
        return State{U[0] * (b - a), U[1] * (b - a), U[2] * (b - a)};
    });
}

struct Cell {
    double x, V;
    State W;
};
template <class F>
std::vector<Cell> cells(const F& fv, Mesh& mesh) {
    mesh.update_cells();
    const auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.positions());
    const auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.centroid());
    const auto V = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.volume());
    const auto q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fv.state());
    std::vector<Cell> out;
    for (int i = 0; i < mesh.particles(); ++i)
        out.push_back({x(i, 0) + c(i, 0), V(i), hydro::primitive(State{q(i, 0) / V(i), q(i, 1) / V(i), q(i, 2) / V(i)}, fv.gas())});
    return out;
}
template <class F>
std::array<long double, 3> totals(const F& fv) {
    const auto q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fv.state());
    std::array<long double, 3> t{0, 0, 0};
    for (std::size_t i = 0; i < q.extent(0); ++i)
        for (int v = 0; v < 3; ++v) t[v] += q(i, v);
    return t;
}

// Runs to time t_end (the last step shortened to land on it).
template <class F>
int run(F& fv, double t_end, double courant, hydro::TimeStep rule) {
    double t = 0;
    int steps = 0;
    while (t < t_end * (1 - 1e-12)) {
        const double dt = std::min(fv.time_step(courant, rule), t_end - t);
        fv.step(dt);
        t += dt;
        ++steps;
    }
    return steps;
}

}  // namespace

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    Check check;
    const Exec exec;
    const hydro::Gas gas{1.4};
    using hydro::Motion;
    const Motion motions[] = {Motion::static_mesh, Motion::moving};

    // Uniform states stay uniform.
    for (const Motion motion : motions)
        for (int order : {0, 1}) {
            const std::string name = std::string(motion == Motion::moving ? "moving" : "static") + " order " + std::to_string(order);
            {
                Mesh mesh(exec, 1.0);
                build(mesh, points(200, 1.0, 0.8, 1));
                FV<> fv(exec, mesh, gas, order, hydro::Limiter::face_centroid, motion);
                const State W{1.3, 0.7, 0.4};
                set(fv, mesh, [&](double) { return W; }, gas);
                for (int s = 0; s < 20; ++s) fv.step(fv.time_step(0.3));
                double worst = 0;
                for (const Cell& c : cells(fv, mesh))
                    for (int v = 0; v < 3; ++v) worst = std::max(worst, std::abs(c.W[v] - W[v]) / std::abs(W[v]));
                check.expect(worst < 1e-13, "periodic, jittered, " + name + ": a uniform state with bulk velocity stays uniform (" +
                                                sci(worst) + ")");
            }
            {
                Mesh mesh(exec, 2.0, voronoi::Ends::walls);
                build(mesh, points(200, 2.0, 0.8, 2));
                const hydro::Walls walls{hydro::Wall::reflective, hydro::Wall::reflective};
                FV<hydro::HLL> fv(exec, mesh, gas, order, hydro::Limiter::face_centroid, motion, walls);
                const State W{0.8, 0.0, 2.0};
                set(fv, mesh, [&](double) { return W; }, gas);
                for (int s = 0; s < 20; ++s) fv.step(fv.time_step(0.3));
                double worst = 0;
                for (const Cell& c : cells(fv, mesh)) {
                    worst = std::max({worst, std::abs(c.W[0] - W[0]) / W[0], std::abs(c.W[2] - W[2]) / W[2]});
                    worst = std::max(worst, std::abs(c.W[1]) / std::sqrt(gas.gamma * W[2] / W[0]));
                }
                check.expect(worst < 1e-13, "reflective walls, HLL, " + name + ": a uniform state at rest stays at rest (" +
                                                sci(worst) + ")");
            }
        }

    // Conservation of a rough state, and the same bits twice.
    for (const Motion motion : motions)
        for (const bool periodic : {true, false}) {
            const std::string name = std::string(motion == Motion::moving ? "moving" : "static") +
                                     (periodic ? ", periodic" : ", reflective walls");
            auto make = [&](Mesh& mesh, FV<>& fv) {
                std::mt19937_64 rng(5);
                std::uniform_real_distribution<double> unit(-1.0, 1.0);
                std::vector<State> W(300);
                for (State& w : W) w = State{1 + 0.3 * unit(rng), 0.5 * unit(rng), 1 + 0.3 * unit(rng)};
                set(fv, mesh, [&](double a, double b) {
                    const int k = std::min(299, static_cast<int>(150 * (a + b)));  // a blast in the middle
                    State w = W[k];
                    if (std::abs(0.5 * (a + b) - 1.0) < 0.1) w[2] *= 100;
                    const State U = hydro::conserved(w, gas);
                    return State{U[0] * (b - a), U[1] * (b - a), U[2] * (b - a)};
                });
                (void)mesh;
            };
            const hydro::Walls walls{hydro::Wall::reflective, hydro::Wall::reflective};
            const auto ends = periodic ? voronoi::Ends::periodic : voronoi::Ends::walls;
            Mesh mesh(exec, 2.0, ends), again(exec, 2.0, ends);
            build(mesh, points(300, 2.0, 0.8, 3));
            build(again, points(300, 2.0, 0.8, 3));
            FV<> fv(exec, mesh, gas, 1, hydro::Limiter::face_centroid, motion, walls);
            FV<> fv2(exec, again, gas, 1, hydro::Limiter::face_centroid, motion, walls);
            make(mesh, fv);
            make(again, fv2);
            const auto before = totals(fv);
            for (int s = 0; s < 40; ++s) {
                fv.step(fv.time_step(0.3));
                fv2.step(fv2.time_step(0.3));
            }
            const auto after = totals(fv);
            const double mass = double(std::abs(after[0] - before[0]) / before[0]);
            const double energy = double(std::abs(after[2] - before[2]) / before[2]);
            const double momentum = double(std::abs(after[1] - before[1]) / std::sqrt(2 * before[0] * before[2]));
            const bool ok = mass < 1e-14 && energy < 1e-14 && (!periodic || momentum < 1e-14);
            check.expect(ok, name + ": mass " + sci(mass) + ", energy " + sci(energy) +
                                 (periodic ? ", momentum " + sci(momentum) : std::string()) + " conserved over 40 steps");
            const auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fv.state());
            const auto b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fv2.state());
            bool same = true;
            for (std::size_t i = 0; i < a.extent(0); ++i)
                for (int v = 0; v < 3; ++v) same = same && a(i, v) == b(i, v);
            if (motion == Motion::moving) {
                const auto xa = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fv.positions());
                const auto xb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fv2.positions());
                for (std::size_t i = 0; i < xa.extent(0); ++i) same = same && xa(i, 0) == xb(i, 0);
            }
            check.expect(same, name + ": two identical runs give the same bits" +
                                   (motion == Motion::moving ? " (states and positions, " + std::to_string(fv.mesh_stats().rebuilds) +
                                                                   " re-sorts)"
                                                             : std::string()));
        }

    // Inflow: a supersonic state entering from the left stays itself.
    {
        Mesh mesh(exec, 1.0, voronoi::Ends::walls);
        build(mesh, points(100, 1.0, 0.0, 4));
        const State in{2.0, 3.0, 1.0};
        hydro::Walls walls{hydro::Wall::inflow, hydro::Wall::outflow};
        walls.left_inflow = in;
        FV<> fv(exec, mesh, gas, 1, hydro::Limiter::face_centroid, Motion::static_mesh, walls);
        set(fv, mesh, [&](double x) { return x < 0.5 ? in : State{0.5, 0.0, 0.2}; }, gas);
        for (int s = 0; s < 30; ++s) fv.step(fv.time_step(0.3));
        double worst = 0;
        const auto c = cells(fv, mesh);
        for (const Cell& cell : c)
            if (cell.x < 0.3)
                for (int v = 0; v < 3; ++v) worst = std::max(worst, std::abs(cell.W[v] - in[v]) / in[v]);
        check.expect(worst < 1e-12, "inflow wall: the state entering stays the inflow state (" + sci(worst) + ")");
    }

    // Sod (Toro 1) against the exact cell averages at t = 0.25.
    const exact_riemann::Solver sod({1.0, 0.0, 1.0}, {0.125, 0.0, 0.1}, 1.4);
    const double bounds[2][3] = {{0.0063, 0.0097, 0.0047}, {0.0034, 0.0046, 0.0023}};  // LUNA, N = 128, 256
    for (const Motion motion : motions)
        for (const double jitter : {0.0, 0.5})
            for (int k = 0; k < 2; ++k) {
                const int n = 128 << k;
                Mesh mesh(exec, 1.0, voronoi::Ends::walls);
                build(mesh, points(n, 1.0, jitter, 6));
                FV<> fv(exec, mesh, gas, 1, hydro::Limiter::face_centroid, motion, hydro::Walls{});
                set(fv, mesh, [&](double a, double b) {  // exact averages of the initial jump
                    const exact_riemann::Primitive w = sod.average(a, b, 0.5, 1e-12);
                    const State U = hydro::conserved(State{w.rho, w.u, w.p}, gas);
                    return State{U[0] * (b - a), U[1] * (b - a), U[2] * (b - a)};
                });
                const bool moving = motion == Motion::moving;
                const int steps = run(fv, 0.25, moving ? hydro::courant_moving_1d : hydro::courant_static_1d,
                                      moving ? hydro::TimeStep::arepo : hydro::TimeStep::luna);
                double l1[3] = {0, 0, 0};
                const auto c = cells(fv, mesh);
                for (const Cell& cell : c) {
                    const exact_riemann::Primitive e = sod.average(cell.x - 0.5 * cell.V, cell.x + 0.5 * cell.V, 0.5, 0.25);
                    l1[0] += cell.V * std::abs(cell.W[0] - e.rho);
                    l1[1] += cell.V * std::abs(cell.W[1] - e.u);
                    l1[2] += cell.V * std::abs(cell.W[2] - e.p);
                }
                bool ok = true;
                for (int v = 0; v < 3; ++v) ok = ok && l1[v] <= bounds[k][v];
                check.expect(ok, std::string("Sod ") + (moving ? "moving (AREPO dt, 0.4)" : "static (LUNA dt, 0.8)") +
                                     (jitter > 0 ? ", jittered" : ", uniform") + ", N=" + std::to_string(n) + ", " +
                                     std::to_string(steps) + " steps: L1 rho " + sci(l1[0]) + ", u " + sci(l1[1]) + ", p " +
                                     sci(l1[2]) + " within LUNA's bounds");
            }

    // LUNA's final states, MC limiter (our face limiter on a uniform line), HLLC, CFL 0.8.
    {
        struct Case {
            const char* name;
            const double (*values)[3];
            int steps;
        };
        const Case cases[] = {{"sod", luna_sod, luna_steps_sod},
                              {"shu-osher", luna_shu_osher, luna_steps_shu_osher},
                              {"woodward-colella", luna_woodward_colella, luna_steps_woodward_colella}};
        namespace pr = hydro::problems_1d;
        for (const Case& c : cases) {
            const pr::Problem problem = pr::problem(c.name);
            const hydro::Gas gamma14{pr::gamma_gas};
            Mesh mesh(exec, problem.L, voronoi::Ends::walls);
            build(mesh, points(16, problem.L, 0.0, 1));
            FV<> fv(exec, mesh, gamma14, 1, hydro::Limiter::face_centroid, Motion::static_mesh, problem.walls);
            set(fv, mesh, [&](double a, double b) {
                const State U = problem.average(a + problem.origin, b + problem.origin);
                return State{U[0] * (b - a), U[1] * (b - a), U[2] * (b - a)};
            });
            const int steps = run(fv, problem.t_end, 0.8, hydro::TimeStep::luna);
            std::vector<Cell> line = cells(fv, mesh);
            std::sort(line.begin(), line.end(), [](const Cell& a, const Cell& b) { return a.x < b.x; });
            double worst = 0;
            for (int i = 0; i < 16; ++i)
                for (int v = 0; v < 3; ++v) worst = std::max(worst, std::abs(line[i].W[v] - c.values[i][v]) / (1e-3 + std::abs(c.values[i][v])));
            check.expect(steps == c.steps && worst < 1e-12, std::string(c.name) + " at N = 16 equals LUNA's final state over " +
                                                                std::to_string(steps) + " steps (" + sci(worst) + ")");
        }
    }

    // The fused step: static, time_step() + step() to rounding; moving, a
    // rough state conserved, the same bits twice, close to the separate
    // kernels (whose time step knows the new drift).
    for (const Motion motion : motions)
        for (const bool periodic : {true, false}) {
            const std::string name = std::string(motion == Motion::moving ? "moving" : "static") + (periodic ? ", periodic" : ", walls");
            const hydro::Walls walls{hydro::Wall::reflective, hydro::Wall::reflective};
            const auto ends = periodic ? voronoi::Ends::periodic : voronoi::Ends::walls;
            auto rough = [&](FV<>& fv, Mesh& mesh) {
                set(fv, mesh, [&](double a, double b) {
                    const double m = 0.5 * (a + b);
                    const State U = hydro::conserved(State{1.0 + 0.3 * std::sin(9 * m), 0.4 * std::cos(5 * m), std::abs(m - 1) < 0.1 ? 20.0 : 1.0}, gas);
                    return State{U[0] * (b - a), U[1] * (b - a), U[2] * (b - a)};
                });
            };
            Mesh m1(exec, 2.0, ends), m2(exec, 2.0, ends), m3(exec, 2.0, ends);
            build(m1, points(400, 2.0, 0.5, 8));
            build(m2, points(400, 2.0, 0.5, 8));
            build(m3, points(400, 2.0, 0.5, 8));
            FV<> split(exec, m1, gas, 1, hydro::Limiter::face_centroid, motion, walls);
            FV<> fused(exec, m2, gas, 1, hydro::Limiter::face_centroid, motion, walls);
            FV<> again(exec, m3, gas, 1, hydro::Limiter::face_centroid, motion, walls);
            rough(split, m1);
            rough(fused, m2);
            rough(again, m3);
            const auto before = totals(fused);
            for (int k = 0; k < 60; ++k) split.step(split.time_step(0.8, hydro::TimeStep::luna));
            for (FV<>* fv : {&fused, &again}) {
                double dt = fv->time_step(0.8, hydro::TimeStep::luna);
                for (int k = 0; k < 60; ++k) dt = fv->step_and_time_step(dt, 0.8);
            }
            const auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, split.state());
            const auto b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, fused.state());
            const auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, again.state());
            double difference = 0;
            bool same = true;
            for (std::size_t i = 0; i < a.extent(0); ++i)
                for (int v = 0; v < 3; ++v) {
                    difference = std::max(difference, std::abs(a(i, v) - b(i, v)));
                    same = same && b(i, v) == c(i, v);
                }
            const auto after = totals(fused);
            const double mass = double(std::abs(after[0] - before[0]) / before[0]), energy = double(std::abs(after[2] - before[2]) / before[2]);
            check.expect(same && mass < 1e-14 && energy < 1e-14 && difference < 1e-9,
                         "fused step, " + name + ": differs from time_step() + step() by " + sci(difference) + ", mass " + sci(mass) +
                             ", energy " + sci(energy) + ", the same bits twice");
        }

    // A moving step with the order check in its kernel gives the bits of
    // the separate check.
    for (const bool periodic : {true, false}) {
        const hydro::Walls walls{hydro::Wall::reflective, hydro::Wall::reflective};
        const auto ends = periodic ? voronoi::Ends::periodic : voronoi::Ends::walls;
        Mesh m1(exec, 2.0, ends), m2(exec, 2.0, ends);
        build(m1, points(400, 2.0, 0.5, 9));
        build(m2, points(400, 2.0, 0.5, 9));
        FV<> together(exec, m1, gas, 1, hydro::Limiter::face_centroid, Motion::moving, walls);
        FV<> apart(exec, m2, gas, 1, hydro::Limiter::face_centroid, Motion::moving, walls);
        for (auto* pair : {&together, &apart}) {
            Mesh& mesh = pair == &together ? m1 : m2;
            set(*pair, mesh, [&](double a, double b) {
                const double m = 0.5 * (a + b);
                const State U = hydro::conserved(State{1.0 + 0.3 * std::sin(9 * m), 0.4 * std::cos(5 * m), std::abs(m - 1) < 0.1 ? 20.0 : 1.0}, gas);
                return State{U[0] * (b - a), U[1] * (b - a), U[2] * (b - a)};
            });
        }
        hydro::Timings timed;
        for (int k = 0; k < 60; ++k) {
            together.step(together.time_step(hydro::courant_moving_1d));
            apart.step(apart.time_step(hydro::courant_moving_1d), &timed);
        }
        const auto a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, together.state());
        const auto b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, apart.state());
        const auto xa = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, together.positions());
        const auto xb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, apart.positions());
        bool same = true;
        for (std::size_t i = 0; i < a.extent(0); ++i) {
            for (int v = 0; v < 3; ++v) same = same && a(i, v) == b(i, v);
            same = same && xa(i, 0) == xb(i, 0);
        }
        check.expect(same, std::string("moving step, order check in the kernel or apart, ") + (periodic ? "periodic" : "walls") + ": the same bits");
    }

    // Shu-Osher on a moving mesh: without the regularisation the end cell,
    // grown by the inflow, runs into its neighbour; with it no points cross.
    {
        namespace pr = hydro::problems_1d;
        const pr::Problem problem = pr::problem("shu-osher");
        const hydro::Gas gamma14{pr::gamma_gas};
        long long sorts[2];
        for (int regularise = 0; regularise < 2; ++regularise) {
            Mesh mesh(exec, problem.L, voronoi::Ends::walls);
            build(mesh, points(1024, problem.L, 0.0, 1));
            FV<> fv(exec, mesh, gamma14, 1, hydro::Limiter::face_centroid, Motion::moving, problem.walls, regularise == 1);
            set(fv, mesh, [&](double a, double b) {
                const State U = problem.average(a + problem.origin, b + problem.origin);
                return State{U[0] * (b - a), U[1] * (b - a), U[2] * (b - a)};
            });
            run(fv, problem.t_end, 0.8, hydro::TimeStep::luna);
            sorts[regularise] = mesh.sorts();
        }
        check.expect(sorts[1] == 0 && sorts[0] > 0, "Shu-Osher, moving, N = 1024: points crossed " + std::to_string(sorts[0]) +
                                                        " times unregularised, " + std::to_string(sorts[1]) + " regularised");
    }
    return check.failed() != 0;
}
