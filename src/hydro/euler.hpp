#pragma once

#include <Kokkos_Core.hpp>

// The Euler equations of an ideal gas in D = 1, 2 or 3 dimensions, per cell
// and per face: conversions, the MUSCL-Hancock half step and the HLL and
// HLLC fluxes along a face normal.
//
// Conventions are LUNA's (src/luna/hydro/conversions.py:1): primitives
// W = (rho, v_1 .. v_D, P), conserved U = (rho, rho v_1 .. rho v_D, E) per
// unit volume, D + 2 entries (LUNA's layout, ic.py:141). Line references
// (LUNA file:line, AREPO file:line) are in
// research_notes/hydro_stage1_2026-10-31.md.
//
// Every expression is written so that D = 3 is the arithmetic, in the same
// order, of the 3D-only code before (dot products left to right, the loops
// over d < D the former loops over d < 3): the same bits.
namespace hydro {

template <int D>
using State = Kokkos::Array<double, D + 2>;
// The dimension of a state of N entries (State<D> does not deduce D).
template <std::size_t N>
constexpr int dimension_of = static_cast<int>(N) - 2;

template <int D>
struct Vec {
    double v[D];
    KOKKOS_INLINE_FUNCTION double& operator[](int a) { return v[a]; }
    KOKKOS_INLINE_FUNCTION const double& operator[](int a) const { return v[a]; }
};
using Vec3 = Vec<3>;
template <int D>
KOKKOS_INLINE_FUNCTION Vec<D> operator+(Vec<D> a, Vec<D> b) {
    for (int k = 0; k < D; ++k) a[k] = a[k] + b[k];
    return a;
}
template <int D>
KOKKOS_INLINE_FUNCTION Vec<D> operator-(Vec<D> a, Vec<D> b) {
    for (int k = 0; k < D; ++k) a[k] = a[k] - b[k];
    return a;
}
template <int D>
KOKKOS_INLINE_FUNCTION Vec<D> operator*(Vec<D> a, double s) {
    for (int k = 0; k < D; ++k) a[k] = a[k] * s;
    return a;
}
// Left to right, written out per D (a loop could be contracted otherwise).
template <int D>
KOKKOS_INLINE_FUNCTION double dot(Vec<D> a, Vec<D> b) {
    if constexpr (D == 1) return a[0] * b[0];
    if constexpr (D == 2) return a[0] * b[0] + a[1] * b[1];
    if constexpr (D == 3) return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
// The velocity (or momentum) entries 1 .. D of a state.
template <std::size_t N, int D = dimension_of<N>>
KOKKOS_INLINE_FUNCTION Vec<D> velocity(const Kokkos::Array<double, N>& W) {
    Vec<D> v;
    for (int k = 0; k < D; ++k) v[k] = W[1 + k];
    return v;
}

// Ideal gas with LUNA's RAMSES floors (template_config.py:41-42): density
// at least smallr, specific internal energy at least smallc^2 / (gamma
// (gamma - 1)); applied to primitives, never to the conserved state.
struct Gas {
    double gamma = 5.0 / 3.0;
    double smallr = 1e-10, smallc = 1e-10;
};

// U -> W with the floors (conversions.py:28-36, cons_to_prim_floored).
template <std::size_t N, int D = dimension_of<N>>
KOKKOS_INLINE_FUNCTION State<D> primitive(const Kokkos::Array<double, N>& U, const Gas& gas) {
    const double rho = Kokkos::fmax(U[0], gas.smallr);
    const double inv_rho = 1.0 / rho;
    const Vec<D> v = velocity(U) * inv_rho;
    const double smalle = gas.smallc * gas.smallc / gas.gamma / (gas.gamma - 1.0);
    const double kinetic = 0.5 * rho * dot(v, v);
    const double e = Kokkos::fmax(U[D + 1] - kinetic, smalle * rho);
    State<D> W;
    W[0] = rho;
    for (int k = 0; k < D; ++k) W[1 + k] = v[k];
    W[D + 1] = (gas.gamma - 1.0) * e;
    return W;
}

// W -> U (conversions.py:39-48).
template <std::size_t N, int D = dimension_of<N>>
KOKKOS_INLINE_FUNCTION State<D> conserved(const Kokkos::Array<double, N>& W, const Gas& gas) {
    const double rho = W[0];
    const Vec<D> v = velocity(W);
    const double kinetic = 0.5 * rho * dot(v, v);
    State<D> U;
    U[0] = rho;
    for (int k = 0; k < D; ++k) U[1 + k] = rho * W[1 + k];
    U[D + 1] = W[D + 1] / (gas.gamma - 1.0) + kinetic;
    return U;
}

template <std::size_t N, int D = dimension_of<N>>
KOKKOS_INLINE_FUNCTION double sound_speed(const Kokkos::Array<double, N>& W, const Gas& gas) {
    return Kokkos::sqrt(gas.gamma * W[D + 1] / W[0]);
}

// Gradients of the primitives: g[v] = grad W[v].
template <int D>
using Gradient = Kokkos::Array<Vec<D>, D + 2>;

// MUSCL-Hancock half step: W - dt/2 sum_d A_d(W) dW/dx_d, with the
// primitive-variable Jacobians of jacobian.py:84-97 summed over the axes in
// order as muscl_hancock.py:105-116 does.
template <std::size_t N, int D = dimension_of<N>>
KOKKOS_INLINE_FUNCTION State<D> half_step(const Kokkos::Array<double, N>& W, const Gradient<D>& g, double dt,
                                          const Gas& gas) {
    constexpr int P = D + 1;
    const double rho = W[0], inv_rho = 1.0 / W[0], rho_c2 = gas.gamma * W[P];
    State<D> A;
    for (int v = 0; v < D + 2; ++v) A[v] = 0;
    for (int d = 0; d < D; ++d) {
        const double vn = W[1 + d];
        A[0] += vn * g[0][d] + rho * g[1 + d][d];
        for (int k = 0; k < D; ++k) {
            double row = vn * g[1 + k][d];
            if (k == d) row = row + inv_rho * g[P][d];
            A[1 + k] += row;
        }
        A[P] += rho_c2 * g[1 + d][d] + vn * g[P][d];
    }
    State<D> half;
    for (int v = 0; v < D + 2; ++v) half[v] = W[v] - 0.5 * dt * A[v];
    return half;
}

// Riemann solvers along the unit normal n of a face moving with velocity w
// (w = 0 on a static mesh): the states' velocities are taken relative to
// w, the Riemann problem is solved along n, and the flux returned is the
// lab-frame flux through the moving face (F(U) - (w.n) U for equal
// states). Flux per unit area, from L to R. Wave speeds after Davis as
// LUNA (fluxes.py:146-147, 65-66).
//
// LUNA rotates the states into (normal, two tangents), solves, and rotates
// the flux back (fluxes.py:9-48). Here the vectors stay in box axes: every
// branch of HLL and HLLC is chosen by scalars (S_L, S_M, S_R) that do not
// depend on the frame, and within a branch each flux component is the same
// linear combination of the corresponding components of F, U and U*, so
// rotating is the identity up to rounding; for HLLC the star momentum is
// rho* (v_t + S_M n), the tangential velocity v_t = v - (v.n) n carried
// through, which is LUNA's rho* (S_M, vt1, vt2) in box axes (in 1D, n = +-1
// and v_t = 0). CTest `hydro` checks both against LUNA's values and under
// random rotations.
namespace detail {

template <int D>
struct Side {
    Vec<D> v;            // velocity relative to the face
    double vn, c, E;     // normal velocity, sound speed (0 if c^2 <= 0), energy
    State<D> F, U;       // flux along n and conserved state
};

template <int D>
KOKKOS_INLINE_FUNCTION Side<D> side(const State<D>& W, Vec<D> n, Vec<D> w, const Gas& gas) {
    constexpr int P = D + 1;
    Side<D> s;
    s.v = velocity(W) - w;
    s.vn = dot(s.v, n);
    const double c2 = gas.gamma * W[P] / W[0];
    s.c = c2 > 0 ? Kokkos::sqrt(c2) : 0.0;
    s.E = W[P] / (gas.gamma - 1.0) + W[0] * (0.5 * dot(s.v, s.v));
    const double m = W[0] * s.vn;
    s.F[0] = m;
    for (int k = 0; k < D; ++k) s.F[1 + k] = m * s.v[k] + W[P] * n[k];
    s.F[P] = (s.E + W[P]) * s.vn;
    s.U[0] = W[0];
    for (int k = 0; k < D; ++k) s.U[1 + k] = W[0] * s.v[k];
    s.U[P] = s.E;
    return s;
}

// From the face frame to the lab frame, as AREPO next generation's
// flux_convert_to_lab_frame (mesh_fluxes.cc:492-498): energy + w.F_momentum
// + |w|^2 / 2 F_rho, momentum + w F_rho.
template <int D>
KOKKOS_INLINE_FUNCTION State<D> to_lab(State<D> F, Vec<D> w) {
    F[D + 1] += dot(w, velocity(F)) + 0.5 * dot(w, w) * F[0];
    for (int k = 0; k < D; ++k) F[1 + k] += w[k] * F[0];
    return F;
}

}  // namespace detail

// HLL, LUNA's _hll_core (fluxes.py:135-167).
struct HLL {
    template <int D>
    KOKKOS_INLINE_FUNCTION static State<D> flux(const State<D>& L, const State<D>& R, Vec<D> n, Vec<D> w,
                                                const Gas& gas) {
        const detail::Side<D> l = detail::side(L, n, w, gas), r = detail::side(R, n, w, gas);
        const double SL = Kokkos::fmin(l.vn - l.c, r.vn - r.c);
        const double SR = Kokkos::fmax(l.vn + l.c, r.vn + r.c);
        State<D> F;
        if (SL >= 0.0) {
            F = l.F;
        } else if (SR <= 0.0) {
            F = r.F;
        } else {
            const double inv_dS = 1.0 / (SR - SL);
            for (int v = 0; v < D + 2; ++v) F[v] = (SR * l.F[v] - SL * r.F[v] + SL * SR * (r.U[v] - l.U[v])) * inv_dS;
        }
        return detail::to_lab(F, w);
    }
};

// HLLC, LUNA's _hllc_core (fluxes.py:51-106; Toro 2009, ch. 10), with its
// guards: c^2 < 0 on either side takes F_R (fluxes.py:62, 88-94),
// divisions are kept off zero by 1e-20 (fluxes.py:71-72).
struct HLLC {
    KOKKOS_INLINE_FUNCTION static double nonzero(double d) {
        constexpr double eps = 1e-20;
        return Kokkos::fabs(d) < eps ? (d < 0 ? -eps : eps) : d;
    }
    template <int D>
    KOKKOS_INLINE_FUNCTION static State<D> flux(const State<D>& L, const State<D>& R, Vec<D> n, Vec<D> w,
                                                const Gas& gas) {
        constexpr int P = D + 1;
        // Wave speeds and the contact speed from the normal velocities and
        // sound speeds alone; the flux is then built for one side only.
        const double vnL = dot(velocity(L) - w, n), vnR = dot(velocity(R) - w, n);
        const double c2L = gas.gamma * L[P] / L[0], c2R = gas.gamma * R[P] / R[0];
        const bool ok = c2L >= 0 && c2R >= 0;
        const double cL = c2L > 0 ? Kokkos::sqrt(c2L) : 0.0, cR = c2R > 0 ? Kokkos::sqrt(c2R) : 0.0;
        const double SL = Kokkos::fmin(vnL - cL, vnR - cR);
        const double SR = Kokkos::fmax(vnL + cL, vnR + cR);
        const double denomL = L[0] * (SL - vnL), denomR = R[0] * (SR - vnR);
        const double SM = (R[P] - L[P] + denomL * vnL - denomR * vnR) / nonzero(denomL - denomR);
        // LUNA's branch order (fluxes.py:88-94): F_L, F*_L, F*_R, F_R.
        const bool left = ok && (SL >= 0 || SM >= 0);
        const bool star = ok && SL < 0 && (SM >= 0 || SR > 0);
        const State<D>& W = left ? L : R;
        const detail::Side<D> s = detail::side(W, n, w, gas);
        if (!star) return detail::to_lab(s.F, w);
        // The star state: rho*, rho* (v_t + S_M n), rho* E*.
        const double S = left ? SL : SR, denom = left ? denomL : denomR;
        const double rho_star = denom / nonzero(S - SM);
        const double E_star = s.E * (1.0 / W[0]) + (SM - s.vn) * (SM + W[P] * (1.0 / nonzero(denom)));
        const Vec<D> v_star = s.v + n * (SM - s.vn);
        State<D> U_star;
        U_star[0] = rho_star;
        for (int k = 0; k < D; ++k) U_star[1 + k] = rho_star * v_star[k];
        U_star[P] = rho_star * E_star;
        State<D> F;
        for (int v = 0; v < D + 2; ++v) F[v] = s.F[v] + S * (U_star[v] - s.U[v]);
        return detail::to_lab(F, w);
    }
};

}  // namespace hydro
