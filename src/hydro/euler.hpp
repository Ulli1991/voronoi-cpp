#pragma once

#include <Kokkos_Core.hpp>

// The Euler equations of an ideal gas, per cell and per face: conversions,
// the MUSCL-Hancock half step and the HLL flux along a face normal.
//
// Conventions are LUNA's (src/luna/hydro/conversions.py:1): primitives
// W = (rho, vx, vy, vz, P), conserved U = (rho, rho vx, rho vy, rho vz, E)
// per unit volume. Line references (LUNA file:line, AREPO file:line) are in
// research_notes/hydro_stage1_2026-10-31.md.
namespace hydro {

using State = Kokkos::Array<double, 5>;

struct Vec3 {
    double x, y, z;
};
KOKKOS_INLINE_FUNCTION Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
KOKKOS_INLINE_FUNCTION Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
KOKKOS_INLINE_FUNCTION Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
KOKKOS_INLINE_FUNCTION double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// Ideal gas with LUNA's RAMSES floors (template_config.py:41-42): density
// at least smallr, specific internal energy at least smallc^2 / (gamma
// (gamma - 1)); applied to primitives, never to the conserved state.
struct Gas {
    double gamma = 5.0 / 3.0;
    double smallr = 1e-10, smallc = 1e-10;
};

// U -> W with the floors (conversions.py:28-36, cons_to_prim_floored).
KOKKOS_INLINE_FUNCTION State primitive(const State& U, const Gas& gas) {
    const double rho = Kokkos::fmax(U[0], gas.smallr);
    const double inv_rho = 1.0 / rho;
    const double vx = U[1] * inv_rho, vy = U[2] * inv_rho, vz = U[3] * inv_rho;
    const double smalle = gas.smallc * gas.smallc / gas.gamma / (gas.gamma - 1.0);
    const double kinetic = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
    const double e = Kokkos::fmax(U[4] - kinetic, smalle * rho);
    return {rho, vx, vy, vz, (gas.gamma - 1.0) * e};
}

// W -> U (conversions.py:39-48).
KOKKOS_INLINE_FUNCTION State conserved(const State& W, const Gas& gas) {
    const double rho = W[0];
    const double kinetic = 0.5 * rho * (W[1] * W[1] + W[2] * W[2] + W[3] * W[3]);
    return {rho, rho * W[1], rho * W[2], rho * W[3], W[4] / (gas.gamma - 1.0) + kinetic};
}

KOKKOS_INLINE_FUNCTION double sound_speed(const State& W, const Gas& gas) { return Kokkos::sqrt(gas.gamma * W[4] / W[0]); }

// Gradients of the primitives: g[v] = grad W[v].
using Gradient = Kokkos::Array<Vec3, 5>;

// MUSCL-Hancock half step: W - dt/2 sum_d A_d(W) dW/dx_d, with the
// primitive-variable Jacobians of jacobian.py:84-97 summed over x, y, z in
// that order as muscl_hancock.py:105-116 does.
KOKKOS_INLINE_FUNCTION State half_step(const State& W, const Gradient& g, double dt, const Gas& gas) {
    const double rho = W[0], P = W[4], inv_rho = 1.0 / W[0], rho_c2 = gas.gamma * P;
    State A{0, 0, 0, 0, 0};
    for (int d = 0; d < 3; ++d) {
        auto component = [](const Vec3& v, int k) { return k == 0 ? v.x : k == 1 ? v.y : v.z; };
        const double vn = W[1 + d];
        A[0] += vn * component(g[0], d) + rho * component(g[1 + d], d);
        for (int k = 0; k < 3; ++k) {
            double row = vn * component(g[1 + k], d);
            if (k == d) row = row + inv_rho * component(g[4], d);
            A[1 + k] += row;
        }
        A[4] += rho_c2 * component(g[1 + d], d) + vn * component(g[4], d);
    }
    State half;
    for (int v = 0; v < 5; ++v) half[v] = W[v] - 0.5 * dt * A[v];
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
// through, which is LUNA's rho* (S_M, vt1, vt2) in box axes. CTest `hydro`
// checks both against LUNA's values and under random rotations.
namespace detail {

struct Side {
    Vec3 v;              // velocity relative to the face
    double vn, c, E;     // normal velocity, sound speed (0 if c^2 <= 0), energy
    State F, U;          // flux along n and conserved state
};

KOKKOS_INLINE_FUNCTION Side side(const State& W, Vec3 n, Vec3 w, const Gas& gas) {
    Side s;
    s.v = Vec3{W[1], W[2], W[3]} - w;
    s.vn = dot(s.v, n);
    const double c2 = gas.gamma * W[4] / W[0];
    s.c = c2 > 0 ? Kokkos::sqrt(c2) : 0.0;
    s.E = W[4] / (gas.gamma - 1.0) + W[0] * (0.5 * dot(s.v, s.v));
    const double m = W[0] * s.vn;
    s.F = {m, m * s.v.x + W[4] * n.x, m * s.v.y + W[4] * n.y, m * s.v.z + W[4] * n.z, (s.E + W[4]) * s.vn};
    s.U = {W[0], W[0] * s.v.x, W[0] * s.v.y, W[0] * s.v.z, s.E};
    return s;
}

// From the face frame to the lab frame, as AREPO next generation's
// flux_convert_to_lab_frame (mesh_fluxes.cc:492-498): energy + w.F_momentum
// + |w|^2 / 2 F_rho, momentum + w F_rho.
KOKKOS_INLINE_FUNCTION State to_lab(State F, Vec3 w) {
    F[4] += w.x * F[1] + w.y * F[2] + w.z * F[3] + 0.5 * dot(w, w) * F[0];
    F[1] += w.x * F[0];
    F[2] += w.y * F[0];
    F[3] += w.z * F[0];
    return F;
}

}  // namespace detail

// HLL, LUNA's _hll_core (fluxes.py:135-167).
struct HLL {
    KOKKOS_INLINE_FUNCTION static State flux(const State& L, const State& R, Vec3 n, Vec3 w, const Gas& gas) {
        const detail::Side l = detail::side(L, n, w, gas), r = detail::side(R, n, w, gas);
        const double SL = Kokkos::fmin(l.vn - l.c, r.vn - r.c);
        const double SR = Kokkos::fmax(l.vn + l.c, r.vn + r.c);
        State F;
        if (SL >= 0.0) {
            F = l.F;
        } else if (SR <= 0.0) {
            F = r.F;
        } else {
            const double inv_dS = 1.0 / (SR - SL);
            for (int v = 0; v < 5; ++v) F[v] = (SR * l.F[v] - SL * r.F[v] + SL * SR * (r.U[v] - l.U[v])) * inv_dS;
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
    KOKKOS_INLINE_FUNCTION static State flux(const State& L, const State& R, Vec3 n, Vec3 w, const Gas& gas) {
        // Wave speeds and the contact speed from the normal velocities and
        // sound speeds alone; the flux is then built for one side only.
        const double vnL = dot(Vec3{L[1], L[2], L[3]} - w, n), vnR = dot(Vec3{R[1], R[2], R[3]} - w, n);
        const double c2L = gas.gamma * L[4] / L[0], c2R = gas.gamma * R[4] / R[0];
        const bool ok = c2L >= 0 && c2R >= 0;
        const double cL = c2L > 0 ? Kokkos::sqrt(c2L) : 0.0, cR = c2R > 0 ? Kokkos::sqrt(c2R) : 0.0;
        const double SL = Kokkos::fmin(vnL - cL, vnR - cR);
        const double SR = Kokkos::fmax(vnL + cL, vnR + cR);
        const double denomL = L[0] * (SL - vnL), denomR = R[0] * (SR - vnR);
        const double SM = (R[4] - L[4] + denomL * vnL - denomR * vnR) / nonzero(denomL - denomR);
        // LUNA's branch order (fluxes.py:88-94): F_L, F*_L, F*_R, F_R.
        const bool left = ok && (SL >= 0 || SM >= 0);
        const bool star = ok && SL < 0 && (SM >= 0 || SR > 0);
        const State& W = left ? L : R;
        const detail::Side s = detail::side(W, n, w, gas);
        if (!star) return detail::to_lab(s.F, w);
        // The star state: rho*, rho* (v_t + S_M n), rho* E*.
        const double S = left ? SL : SR, denom = left ? denomL : denomR;
        const double rho_star = denom / nonzero(S - SM);
        const double E_star = s.E * (1.0 / W[0]) + (SM - s.vn) * (SM + W[4] * (1.0 / nonzero(denom)));
        const Vec3 v_star = s.v + n * (SM - s.vn);
        const State U_star{rho_star, rho_star * v_star.x, rho_star * v_star.y, rho_star * v_star.z, rho_star * E_star};
        State F;
        for (int v = 0; v < 5; ++v) F[v] = s.F[v] + S * (U_star[v] - s.U[v]);
        return detail::to_lab(F, w);
    }
};

}  // namespace hydro
