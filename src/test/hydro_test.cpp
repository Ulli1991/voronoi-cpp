// The finite-volume hydro on the static Voronoi mesh (src/hydro/):
//   - HLL and HLLC: on sample states the flux equals LUNA's
//     hll_flux_primitive / hllc_flux_primitive (src/luna/hydro/fluxes.py;
//     values printed by LUNA in float64); equal states give the exact flux,
//     also through a moving face (F(U) - (w.n) U); swapping the states and
//     the normal flips the flux; rotating states and normal rotates the
//     flux (solving along n in box axes equals LUNA's rotated frame);
//   - a uniform state stays uniform to rounding on jittered and uniform
//     random meshes, first and second order, both update loops;
//   - mass, momentum and energy are conserved to rounding in the periodic
//     box (a blast plus random perturbations, second order);
//   - two runs give the same bits; the two update loops agree to rounding;
//   - moving mesh: a uniform state with a bulk velocity stays uniform;
//     mass, momentum and energy are conserved to rounding while the cells
//     move; a blast moving with the box equals the blast at rest
//     (Galilean invariance), while on a static mesh it smears.

#include "check.hpp"
#include "finite_volume.hpp"
#include "support.hpp"

#include <array>
#include <cstring>
#include <random>

namespace {

using namespace support;
using hydro::State;
using hydro::Vec3;

double relative(const State& a, const State& b) {
    double diff = 0, scale = 0;
    for (int v = 0; v < 5; ++v) {
        diff = std::max(diff, std::abs(a[v] - b[v]));
        scale = std::max(scale, std::abs(b[v]));
    }
    return diff / scale;
}

// The exact Euler flux of W along the unit normal n.
State exact_flux(const State& W, Vec3 n, const hydro::Gas& gas) {
    const State U = hydro::conserved(W, gas);
    const double vn = W[1] * n.x + W[2] * n.y + W[3] * n.z;
    return {U[0] * vn, U[1] * vn + W[4] * n.x, U[2] * vn + W[4] * n.y, U[3] * vn + W[4] * n.z, (U[4] + W[4]) * vn};
}

struct LunaCase {
    int axis;
    State L, R, F;
};

// From LUNA's hll_flux_primitive(W_L, W_R, axis, gamma = 5/3), jax x64.
const LunaCase luna_hll[] = {
    {0, {1.0, 0.75, -0.2, 0.1, 1.0}, {0.125, 0.0, 0.3, -0.4, 0.1}, {1.1242909809054182, 1.5871592695880872, -0.2709502316828797, 0.15852113359233783, 2.5541166523111647}},
    {1, {1.0, 0.75, -0.2, 0.1, 1.0}, {0.125, 0.0, 0.3, -0.4, 0.1}, {0.5644860772447619, 0.47815803940207324, 0.3950293465356404, 0.09297759537014413, 0.9711790427267954}},
    {2, {1.0, 0.75, -0.2, 0.1, 1.0}, {0.125, 0.0, 0.3, -0.4, 0.1}, {0.6632125075884188, 0.5860279000219649, -0.19172151440478433, 0.6503916525501618, 1.2809009133427387}},
    {0, {2.5, -1.3, 0.4, 2.0, 7.0}, {0.9, -2.1, -0.6, 0.5, 0.3}, {-1.0584443837713475, 4.717006838134387, 1.7103752545994175, 1.083740744619239, -1.07747132635772}},
    {1, {2.5, -1.3, 0.4, 2.0, 7.0}, {0.9, -2.1, -0.6, 0.5, 0.3}, {2.0415213285549596, -1.726947890496127, 6.245695294051243, 5.8212236007823925, 20.900143577827794}},
    {2, {2.5, -1.3, 0.4, 2.0, 7.0}, {0.9, -2.1, -0.6, 0.5, 0.3}, {5.117308949104302, -6.505732679928103, 2.2303847720260843, 17.13667456996695, 50.44443752157727}},
    {0, {0.7, 3.0, 1.0, -1.0, 0.2}, {0.6, 2.5, 0.2, 0.3, 0.25}, {2.0999999999999996, 6.499999999999999, 2.0999999999999996, -2.0999999999999996, 13.049999999999999}},
    {1, {0.7, 3.0, 1.0, -1.0, 0.2}, {0.6, 2.5, 0.2, 0.3, 0.25}, {0.5879676457069523, 1.8857556491831435, 0.9965614425941541, -0.9047846970688987, 4.1600086339176}},
    {2, {0.7, 3.0, 1.0, -1.0, 0.2}, {0.6, 2.5, 0.2, 0.3, 0.25}, {-0.10539806119197127, -0.1665460446141617, 0.13403896114683223, -0.05375747231623394, -0.027662919654724116}},
    {0, {1.1, -4.0, 0.0, 0.5, 0.5}, {1.3, -3.5, 0.1, 0.2, 0.4}, {-4.55, 16.325, -0.455, -0.91, -31.482499999999998}},
    {1, {1.1, -4.0, 0.0, 0.5, 0.5}, {1.3, -3.5, 0.1, 0.2, 0.4}, {-0.022038827977848896, -0.1622208790166134, 0.39992476181439823, 0.13920630056788094, 0.9251995978290001}},
    {2, {1.1, -4.0, 0.0, 0.5, 0.5}, {1.3, -3.5, 0.1, 0.2, 0.4}, {0.3956780378220708, -1.7908409418191646, -0.04162575810617627, 0.7953580101016026, 4.6019572824376}},
};

// From LUNA's hllc_flux_primitive(W_L, W_R, axis, gamma = 5/3), jax x64: the
// same states and a pair with a contact at rest.
const LunaCase luna_hllc[] = {
    {0, {1.0, 0.75, -0.2, 0.1, 1.0}, {0.125, 0.0, 0.3, -0.4, 0.1}, {0.9196817777792649, 1.3665683598451346, -0.183936355555853, 0.0919681777779265, 2.2324199149234163}},
    {1, {1.0, 0.75, -0.2, 0.1, 1.0}, {0.125, 0.0, 0.3, -0.4, 0.1}, {0.3130650464627618, 0.2347987848470714, 0.27502286388364394, 0.03130650464627618, 0.5623838935957439}},
    {2, {1.0, 0.75, -0.2, 0.1, 1.0}, {0.125, 0.0, 0.3, -0.4, 0.1}, {0.4144331156680667, 0.31082483675105, -0.08288662313361335, 0.5211506657865912, 0.8927107075712877}},
    {0, {2.5, -1.3, 0.4, 2.0, 7.0}, {0.9, -2.1, -0.6, 0.5, 0.3}, {-1.2261837085084781, 4.840045906572782, 0.735710225105087, -0.6130918542542391, -5.540939099342062}},
    {1, {2.5, -1.3, 0.4, 2.0, 7.0}, {0.9, -2.1, -0.6, 0.5, 0.3}, {1.7831293346997084, -2.3180681351096206, 6.0214990167113935, 3.5662586693994167, 16.22969327030239}},
    {2, {2.5, -1.3, 0.4, 2.0, 7.0}, {0.9, -2.1, -0.6, 0.5, 0.3}, {5.049576145574094, -6.564448989246323, 2.0198304582296376, 16.987836195598348, 49.87243345696272}},
    {0, {0.7, 3.0, 1.0, -1.0, 0.2}, {0.6, 2.5, 0.2, 0.3, 0.25}, {2.0999999999999996, 6.499999999999999, 2.0999999999999996, -2.0999999999999996, 13.049999999999999}},
    {1, {0.7, 3.0, 1.0, -1.0, 0.2}, {0.6, 2.5, 0.2, 0.3, 0.25}, {0.5676075578342731, 1.702822673502819, 0.9838485467049601, -0.5676075578342731, 3.6622970170364457}},
    {2, {0.7, 3.0, 1.0, -1.0, 0.2}, {0.6, 2.5, 0.2, 0.3, 0.25}, {-0.12792377657362372, -0.3198094414340594, -0.02558475531472474, -0.044980280116773574, -0.44134482722254403}},
    {0, {1.1, -4.0, 0.0, 0.5, 0.5}, {1.3, -3.5, 0.1, 0.2, 0.4}, {-4.55, 16.325, -0.455, -0.91, -31.482499999999998}},
    {1, {1.1, -4.0, 0.0, 0.5, 0.5}, {1.3, -3.5, 0.1, 0.2, 0.4}, {0.100583309099083, -0.402333236396332, 0.4124534666188211, 0.0502916545495415, 0.9225936856620478}},
    {2, {1.1, -4.0, 0.0, 0.5, 0.5}, {1.3, -3.5, 0.1, 0.2, 0.4}, {0.46297210995569943, -1.8518884398227977, 0.0, 0.8199163885081319, 4.3200249344746355}},
    {0, {1.0, 0.1, 0.5, -0.3, 1.0}, {1.0, -0.1, -0.4, 0.2, 1.0}, {4.4019662670043687e-17, 1.1490994448735805, 2.2009831335021843e-17, -1.1395537529520383e-17, -5.914963034015329e-17}},
    {1, {1.0, 0.1, 0.5, -0.3, 1.0}, {1.0, -0.1, -0.4, 0.2, 1.0}, {0.06292364833003228, 0.006292364833003228, 1.9890936843476141, -0.018877094499009674, 0.21620235229196627}},
    {2, {1.0, 0.1, 0.5, -0.3, 1.0}, {1.0, -0.1, -0.4, 0.2, 1.0}, {-0.04188835494491575, 0.004188835494491576, 0.016755341977966305, 0.6793458055632945, -0.09350533845911954}},
};

// v rotated by the angle |a| about the axis a (Rodrigues).
Vec3 rotate(Vec3 v, Vec3 a) {
    const double angle = std::sqrt(hydro::dot(a, a));
    const Vec3 k = a * (1.0 / angle);
    const Vec3 kxv{k.y * v.z - k.z * v.y, k.z * v.x - k.x * v.z, k.x * v.y - k.y * v.x};
    return v * std::cos(angle) + kxv * std::sin(angle) + k * (hydro::dot(k, v) * (1 - std::cos(angle)));
}
State rotate(const State& s, Vec3 a) {
    const Vec3 v = rotate(Vec3{s[1], s[2], s[3]}, a);
    return {s[0], v.x, v.y, v.z, s[4]};
}

template <class Riemann, std::size_t N>
void flux_tests(Check& check, const std::string& name, const LunaCase (&cases)[N]) {
    const hydro::Gas gas{5.0 / 3.0};
    double luna = 0;
    for (const auto& c : cases) {
        const Vec3 n{c.axis == 0 ? 1.0 : 0.0, c.axis == 1 ? 1.0 : 0.0, c.axis == 2 ? 1.0 : 0.0};
        luna = std::max(luna, relative(Riemann::flux(c.L, c.R, n, {0, 0, 0}, gas), c.F));
    }
    check.expect(luna < 1e-14, name + " equals LUNA's on " + std::to_string(N) + " sample faces (largest relative difference " +
                                   sci(luna) + ")");

    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    double consistent = 0, moving = 0, symmetric = 0, rotated = 0;
    for (int k = 0; k < 1000; ++k) {
        Vec3 n{unit(rng), unit(rng), unit(rng)};
        n = n * (1.0 / std::sqrt(hydro::dot(n, n)));
        const Vec3 w{unit(rng), unit(rng), unit(rng)};
        const Vec3 axis{3 * unit(rng), 3 * unit(rng), 3 * unit(rng)};
        const State L{1.5 + unit(rng), 2 * unit(rng), 2 * unit(rng), 2 * unit(rng), 1.5 + unit(rng)};
        const State R{1.5 + unit(rng), 2 * unit(rng), 2 * unit(rng), 2 * unit(rng), 1.5 + unit(rng)};
        consistent = std::max(consistent, relative(Riemann::flux(L, L, n, {0, 0, 0}, gas), exact_flux(L, n, gas)));
        State expected = exact_flux(L, n, gas);
        const State U = hydro::conserved(L, gas);
        for (int v = 0; v < 5; ++v) expected[v] -= hydro::dot(w, n) * U[v];
        moving = std::max(moving, relative(Riemann::flux(L, L, n, w, gas), expected));
        const State F = Riemann::flux(L, R, n, w, gas);
        State back = Riemann::flux(R, L, n * -1.0, w, gas);
        for (double& f : back) f = -f;
        symmetric = std::max(symmetric, relative(F, back));
        const State turned = Riemann::flux(rotate(L, axis), rotate(R, axis), rotate(n, axis), rotate(w, axis), gas);
        rotated = std::max(rotated, relative(turned, rotate(F, axis)));
    }
    check.expect(consistent < 1e-14, name + " of equal states is the exact flux (1000 normals, " + sci(consistent) + ")");
    check.expect(moving < 1e-13, name + " through a moving face, equal states give F(U) - (w.n) U (" + sci(moving) + ")");
    check.expect(symmetric < 1e-14, name + ": swapping states and normal flips the flux (" + sci(symmetric) + ")");
    check.expect(rotated < 1e-13, name + ": rotating states, normal and face velocity rotates the flux (" + sci(rotated) + ")");
}

using HostCells = Kokkos::View<double* [5], Kokkos::LayoutRight, Kokkos::HostSpace>;

// The extensive state Q of each cell (mass, momentum, energy).
template <class FV>
HostCells host_extensive(const FV& fv) {
    HostCells h("extensive", fv.state().extent(0));
    auto m = to_host(fv.state());
    for (std::size_t i = 0; i < h.extent(0); ++i)
        for (int v = 0; v < 5; ++v) h(i, v) = m(i, v);
    return h;
}

// The state per unit volume, Q / V with the mesh's current volumes.
template <class FV>
HostCells host_state(const FV& fv, const voronoi::Mesh<Exec>& mesh) {
    HostCells h = host_extensive(fv);
    const auto volume = to_host(mesh.volume());
    for (std::size_t i = 0; i < h.extent(0); ++i)
        for (int v = 0; v < 5; ++v) h(i, v) /= volume(i);
    return h;
}

// Sets Q = U V from the state per unit volume.
template <class FV>
void set_state(FV& fv, const voronoi::Mesh<Exec>& mesh, const HostCells& h) {
    auto m = Kokkos::create_mirror_view(fv.state());
    const auto volume = to_host(mesh.volume());
    for (std::size_t i = 0; i < h.extent(0); ++i)
        for (int v = 0; v < 5; ++v) m(i, v) = h(i, v) * volume(i);
    Kokkos::deep_copy(fv.state(), m);
}

// A blast in the corner plus random perturbations of every variable.
HostCells rough_state(int n, const hydro::Gas& gas) {
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    HostCells h("rough", n);
    for (int i = 0; i < n; ++i) {
        const hydro::State W{1 + 0.3 * unit(rng), 0.5 * unit(rng), 0.5 * unit(rng), 0.5 * unit(rng),
                             (i < 8 ? 100.0 : 1.0) * (1 + 0.3 * unit(rng))};
        const State U = hydro::conserved(W, gas);
        for (int v = 0; v < 5; ++v) h(i, v) = U[v];
    }
    return h;
}

const char* name(hydro::Update u) { return u == hydro::Update::per_cell ? "per cell" : "per face"; }

template <class Riemann>
void mesh_tests(Check& check, const std::string& kind, const std::string& solver) {
    using FV = hydro::FiniteVolume<Exec, Riemann>;
    const hydro::Gas gas{5.0 / 3.0};
    const int n = 4096;
    voronoi::Mesh<Exec> mesh(Exec{}, 1.0);
    mesh.update(to_device(make_points_off_grid(kind, n)));
    mesh.update_cells();
    const auto volume = to_host(mesh.volume());

    for (int order = 0; order <= 1; ++order)
        for (const auto update : {hydro::Update::per_cell, hydro::Update::per_face}) {
            const std::string what = solver + ", " + kind + ", order " + std::to_string(order) + ", " + name(update);
            FV fv(Exec{}, mesh, gas, order, update);
            {  // uniform state
                const State W{1.3, 0.3, -0.2, 0.1, 0.7};
                const State U = hydro::conserved(W, gas);
                HostCells h("uniform", n);
                for (int i = 0; i < n; ++i)
                    for (int v = 0; v < 5; ++v) h(i, v) = U[v];
                set_state(fv, mesh, h);
                for (int s = 0; s < 3; ++s) fv.step(fv.time_step(0.3));
                const HostCells after = host_state(fv, mesh);
                double worst = 0;
                for (int i = 0; i < n; ++i)
                    worst = std::max(worst, relative({after(i, 0), after(i, 1), after(i, 2), after(i, 3), after(i, 4)}, U));
                check.expect(worst < 1e-13, what + ": a uniform state stays uniform over 3 steps (" + sci(worst) + ")");
            }
            {  // conservation, a rough state
                set_state(fv, mesh, rough_state(n, gas));
                auto sums = [&](const HostCells& h) {
                    std::array<long double, 5> s{};
                    for (int i = 0; i < n; ++i)
                        for (int v = 0; v < 5; ++v) s[v] += static_cast<long double>(h(i, v)) * volume(i);
                    return s;
                };
                const auto before = sums(host_state(fv, mesh));
                for (int s = 0; s < 10; ++s) fv.step(fv.time_step(0.3));
                const auto after = sums(host_state(fv, mesh));
                const double momentum_scale = static_cast<double>(before[0]) * std::sqrt(static_cast<double>(before[4] / before[0]));
                const double mass = std::abs(static_cast<double>((after[0] - before[0]) / before[0]));
                const double energy = std::abs(static_cast<double>((after[4] - before[4]) / before[4]));
                double momentum = 0;
                for (int v = 1; v <= 3; ++v)
                    momentum = std::max(momentum, std::abs(static_cast<double>(after[v] - before[v])) / momentum_scale);
                check.expect(mass < 1e-14 && energy < 1e-14 && momentum < 1e-14,
                             what + ": mass, momentum, energy conserved over 10 steps (" + sci(mass) + ", " + sci(momentum) + ", " +
                                 sci(energy) + ")");
            }
        }

    // Bits: two runs of each loop, and the two loops against each other.
    HostCells result[2][2];
    for (const auto update : {hydro::Update::per_cell, hydro::Update::per_face})
        for (int run = 0; run < 2; ++run) {
            FV fv(Exec{}, mesh, gas, 1, update);
            set_state(fv, mesh, rough_state(n, gas));
            for (int s = 0; s < 10; ++s) fv.step(fv.time_step(0.3));
            result[update == hydro::Update::per_face][run] = host_state(fv, mesh);
        }
    auto same = [&](const HostCells& a, const HostCells& b) { return std::memcmp(a.data(), b.data(), sizeof(double) * 5 * n) == 0; };
    check.expect(same(result[0][0], result[0][1]) && same(result[1][0], result[1][1]),
                 solver + ", " + kind + ": two runs give the same bits (both loops, order 1, 10 steps)");
    // The loops compile the same face flux into different kernels, where the
    // compiler may contract multiply-adds differently: equal to rounding,
    // not necessarily bit for bit.
    double loops = 0;
    for (int i = 0; i < n; ++i)
        loops = std::max(loops, relative({result[0][0](i, 0), result[0][0](i, 1), result[0][0](i, 2), result[0][0](i, 3), result[0][0](i, 4)},
                                         {result[1][0](i, 0), result[1][0](i, 1), result[1][0](i, 2), result[1][0](i, 3), result[1][0](i, 4)}));
    check.expect(loops < 1e-12, solver + ", " + kind + ": the per-cell and the per-face loop agree to rounding (" + sci(loops) +
                                    (same(result[0][0], result[1][0]) ? ", same bits)" : ")"));
}


// Moving-mesh tests. Points on the snapping grid (make_points) and steps
// whose displacement v dt is a multiple of the grid step (2^-30 of the
// box), so a uniform translation of the points translates the snapped mesh
// exactly.
using Moving = hydro::FiniteVolume<Exec>;

// A mesh and a FiniteVolume on it (members in this order: the mesh is built first).
struct MovingRun {
    voronoi::Mesh<Exec> mesh;
    Moving fv;
    MovingRun(const Points& points, const hydro::Gas& gas, hydro::Motion motion)
        : mesh(Exec{}, 1.0), fv(Exec{}, build(mesh, points), gas, 1, hydro::Update::per_face, hydro::Limiter::face_centroid, motion) {}
    static voronoi::Mesh<Exec>& build(voronoi::Mesh<Exec>& mesh, const Points& points) {
        mesh.update(to_device(points));
        mesh.update_cells();
        return mesh;
    }
};

// A blast at the box centre (pressure 100 within r < 0.15, else 1), moving
// with velocity (bulk, 0, 0).
HostCells blast(const voronoi::Mesh<Exec>& mesh, double bulk, const hydro::Gas& gas) {
    const auto x = to_host(mesh.positions());
    HostCells h("blast", x.extent(0));
    for (std::size_t i = 0; i < x.extent(0); ++i) {
        double r2 = 0;
        for (int a = 0; a < 3; ++a) r2 += (x(i, a) - 0.5) * (x(i, a) - 0.5);
        const State U = hydro::conserved({1.0, bulk, 0, 0, r2 < 0.15 * 0.15 ? 100.0 : 1.0}, gas);
        for (int v = 0; v < 5; ++v) h(i, v) = U[v];
    }
    return h;
}

// The density in 8 radial bins (r < 0.4) around centre (cx, 0.5, 0.5), by generator.
std::array<double, 8> profile(const voronoi::Mesh<Exec>& mesh, const HostCells& u, double cx) {
    const auto x = to_host(mesh.positions());
    std::array<double, 8> sum{}, count{};
    for (std::size_t i = 0; i < x.extent(0); ++i) {
        double r2 = 0;
        for (int a = 0; a < 3; ++a) {
            const double d = nearest_image(x(i, a) - (a == 0 ? cx : 0.5));
            r2 += d * d;
        }
        const int bin = static_cast<int>(std::sqrt(r2) / 0.05);
        if (bin >= 8) continue;
        sum[bin] += u(i, 0);
        count[bin] += 1;
    }
    for (int b = 0; b < 8; ++b) sum[b] /= std::max(count[b], 1.0);
    return sum;
}

void moving_tests(Check& check) {
    const hydro::Gas gas{5.0 / 3.0};
    const int n = 4096;
    const Points points = make_points("jittered", n);
    {  // a uniform state with a bulk velocity
        MovingRun run(points, gas, hydro::Motion::moving);
        const State U = hydro::conserved({1.3, 1.0, 0.5, -0.25, 0.7}, gas);
        HostCells h("uniform", n);
        for (int i = 0; i < n; ++i)
            for (int v = 0; v < 5; ++v) h(i, v) = U[v];
        set_state(run.fv, run.mesh, h);
        const double dt = std::ldexp(1.0, -12);  // v dt: 2^18, 2^17, 2^16 grid steps
        double cfl = 1;
        for (int s = 0; s < 5; ++s) {
            cfl = std::min(cfl, run.fv.time_step(0.3));
            run.fv.step(dt);
        }
        const HostCells after = host_state(run.fv, run.mesh);
        double worst = 0;
        for (int i = 0; i < n; ++i) worst = std::max(worst, relative({after(i, 0), after(i, 1), after(i, 2), after(i, 3), after(i, 4)}, U));
        check.expect(dt <= cfl && worst < 1e-13,
                     "moving mesh: a uniform state with a bulk velocity stays uniform over 5 steps (" + sci(worst) + ")");
    }
    {  // conservation with moving cells
        MovingRun run(points, gas, hydro::Motion::moving);
        set_state(run.fv, run.mesh, rough_state(n, gas));
        auto sums = [&](const HostCells& q) {
            std::array<long double, 5> s{};
            for (int i = 0; i < n; ++i)
                for (int v = 0; v < 5; ++v) s[v] += q(i, v);
            return s;
        };
        const auto before = sums(host_extensive(run.fv));
        for (int s = 0; s < 10; ++s) run.fv.step(run.fv.time_step(0.3));
        const auto after = sums(host_extensive(run.fv));
        const double scale = static_cast<double>(before[0]) * std::sqrt(static_cast<double>(before[4] / before[0]));
        double worst = std::max(std::abs(static_cast<double>((after[0] - before[0]) / before[0])),
                                std::abs(static_cast<double>((after[4] - before[4]) / before[4])));
        for (int v = 1; v <= 3; ++v) worst = std::max(worst, std::abs(static_cast<double>(after[v] - before[v])) / scale);
        const auto& stats = run.fv.mesh_stats();
        check.expect(worst < 1e-14 && stats.updates == 10, "moving mesh: mass, momentum, energy conserved over 10 steps (" + sci(worst) +
                                                               ", " + std::to_string(stats.device_flips) + " flips, " +
                                                               std::to_string(stats.rebuilds) + " rebuilds)");
    }
    {  // Galilean invariance: a blast at rest and the same blast moving with the box
        const double bulk = 16, dt = std::ldexp(1.0, -12);  // bulk dt = 2^-8 of the box
        const int steps = 40;
        double moving_error = 0, cfl = 1;
        double profile_error[2] = {};
        for (const auto motion : {hydro::Motion::moving, hydro::Motion::static_mesh}) {
            MovingRun rest(points, gas, motion), flow(points, gas, motion);
            set_state(rest.fv, rest.mesh, blast(rest.mesh, 0, gas));
            set_state(flow.fv, flow.mesh, blast(flow.mesh, bulk, gas));
            for (int s = 0; s < steps; ++s) {
                cfl = std::min({cfl, rest.fv.time_step(0.3), flow.fv.time_step(0.3)});
                rest.fv.step(dt);
                flow.fv.step(dt);
            }
            const HostCells a = host_state(rest.fv, rest.mesh), b = host_state(flow.fv, flow.mesh);
            if (motion == hydro::Motion::moving)
                for (int i = 0; i < n; ++i) {  // cell by cell: the same cells, moved by bulk t
                    const State Wa = hydro::primitive({a(i, 0), a(i, 1), a(i, 2), a(i, 3), a(i, 4)}, gas);
                    State Wb = hydro::primitive({b(i, 0), b(i, 1), b(i, 2), b(i, 3), b(i, 4)}, gas);
                    Wb[1] -= bulk;
                    double d = std::abs(Wb[0] - Wa[0]) / Wa[0] + std::abs(Wb[4] - Wa[4]) / Wa[4];
                    for (int v = 1; v <= 3; ++v) d += std::abs(Wb[v] - Wa[v]) / bulk;
                    moving_error = std::max(moving_error, d);
                }
            const auto pa = profile(rest.mesh, a, 0.5), pb = profile(flow.mesh, b, 0.5 + bulk * dt * steps);
            double diff = 0, total = 0;
            for (int k = 0; k < 8; ++k) {
                diff += std::abs(pb[k] - pa[k]);
                total += pa[k];
            }
            profile_error[motion == hydro::Motion::static_mesh] = diff / total;
        }
        check.expect(dt <= cfl, "Galilean test: the fixed step " + sci(dt) + " is within the time-step criterion (" + sci(cfl) + ")");
        // Rounding (v - bulk is not exact) grows through the nonlinear blast
        // and the limiter; measured 7e-8 on the CPU.
        check.expect(moving_error < 1e-6, "moving mesh: a blast moving with the box equals the blast at rest, cell by cell, after " +
                                               std::to_string(steps) + " steps (" + sci(moving_error) + ")");
        check.expect(profile_error[0] < 1e-10 && profile_error[1] > 1e3 * profile_error[0],
                     "radial density profiles, moving blast against blast at rest: moving mesh " + sci(profile_error[0]) +
                         ", static mesh " + sci(profile_error[1]));
    }
}
}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    Check check;
    flux_tests<hydro::HLL>(check, "HLL", luna_hll);
    flux_tests<hydro::HLLC>(check, "HLLC", luna_hllc);
    mesh_tests<hydro::HLLC>(check, "jittered", "HLLC");
    mesh_tests<hydro::HLLC>(check, "uniform", "HLLC");
    mesh_tests<hydro::HLL>(check, "jittered", "HLL");
    moving_tests(check);
    return check.failed() != 0;
}
