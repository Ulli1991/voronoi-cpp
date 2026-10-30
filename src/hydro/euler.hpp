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

// A face state after reconstruction, checked against the cell-centre state
// Wc: LUNA's RAMSES face rule (slopes.py:181-186): a density below smallr
// or a pressure below smallr smallc^2 takes the cell-centre value, then
// both are floored.
KOKKOS_INLINE_FUNCTION State face_guard(State Wf, const State& Wc, const Gas& gas) {
    const double small_p = gas.smallr * gas.smallc * gas.smallc;
    double rho = Wf[0] < gas.smallr ? Wc[0] : Wf[0];
    double p = Wf[4] < small_p ? Wc[4] : Wf[4];
    rho = Kokkos::fmax(rho, gas.smallr);
    Wf[0] = rho;
    Wf[4] = Kokkos::fmax(p, rho * (gas.smallc * gas.smallc / gas.gamma));
    return Wf;
}

// HLL (fluxes.py:135-167, _hll_core), solved directly along the unit
// normal n, in the frame of a face moving with velocity w: the states' velocities are
// taken relative to w, and the flux returned is the lab-frame flux through
// the moving face, F(U) - (w.n) U for equal states. Wave speeds after Davis
// (fluxes.py:146-147). Flux per unit area, from L to R.
struct HLL {
    KOKKOS_INLINE_FUNCTION static State flux(const State& L, const State& R, Vec3 n, Vec3 w, const Gas& gas) {
        const double inv_gm1 = 1.0 / (gas.gamma - 1.0);
        const Vec3 vL = Vec3{L[1], L[2], L[3]} - w, vR = Vec3{R[1], R[2], R[3]} - w;
        const double vnL = dot(vL, n), vnR = dot(vR, n);
        const double c2L = gas.gamma * L[4] / L[0], c2R = gas.gamma * R[4] / R[0];
        const double cL = c2L > 0 ? Kokkos::sqrt(c2L) : 0.0, cR = c2R > 0 ? Kokkos::sqrt(c2R) : 0.0;
        const double SL = Kokkos::fmin(vnL - cL, vnR - cR);
        const double SR = Kokkos::fmax(vnL + cL, vnR + cR);
        const double EL = L[4] * inv_gm1 + L[0] * (0.5 * dot(vL, vL));
        const double ER = R[4] * inv_gm1 + R[0] * (0.5 * dot(vR, vR));
        const double mL = L[0] * vnL, mR = R[0] * vnR;
        const State FL{mL, mL * vL.x + L[4] * n.x, mL * vL.y + L[4] * n.y, mL * vL.z + L[4] * n.z, (EL + L[4]) * vnL};
        const State FR{mR, mR * vR.x + R[4] * n.x, mR * vR.y + R[4] * n.y, mR * vR.z + R[4] * n.z, (ER + R[4]) * vnR};
        const State UL{L[0], L[0] * vL.x, L[0] * vL.y, L[0] * vL.z, EL};
        const State UR{R[0], R[0] * vR.x, R[0] * vR.y, R[0] * vR.z, ER};
        State F;
        if (SL >= 0.0) {
            F = FL;
        } else if (SR <= 0.0) {
            F = FR;
        } else {
            const double inv_dS = 1.0 / (SR - SL);
            for (int v = 0; v < 5; ++v) F[v] = (SR * FL[v] - SL * FR[v] + SL * SR * (UR[v] - UL[v])) * inv_dS;
        }
        // Back to the lab frame as AREPO's flux_convert_to_lab_frame
        // (finite_volume_solver.c:3434-3450): momentum + w F_rho, energy +
        // w.F_momentum + |w|^2 / 2 F_rho.
        F[4] += w.x * F[1] + w.y * F[2] + w.z * F[3] + 0.5 * dot(w, w) * F[0];
        F[1] += w.x * F[0];
        F[2] += w.y * F[0];
        F[3] += w.z * F[0];
        return F;
    }
};

}  // namespace hydro
