#pragma once

#include "finite_volume_1d.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

// LUNA's three 1D tests: scripts/luna_1d_run.py has the file:line of each.
namespace hydro::problems_1d {

constexpr double gamma_gas = 1.4;

// One of LUNA's problems: the line, its walls, t_end, and the exact average
// of the conserved initial state over [a, b] (in the problem's coordinates).
struct Problem {
    double L, origin, t_end;
    Walls walls;
    State<1> (*average)(double a, double b);
};

inline State<1> conserved(double rho, double u, double p) { return {rho, rho * u, p / (gamma_gas - 1) + 0.5 * rho * u * u}; }
inline double fraction_below(double x, double a, double b) { return std::clamp((x - a) / (b - a), 0.0, 1.0); }

// Sod: exact cell averages of the jump at t -> 0 (LUNA:
// er.cell_average(..., t=1e-12)); at t = 0 that is the volume fraction.
inline State<1> sod(double a, double b) {
    const double f = fraction_below(0.5, a, b);
    const State<1> l = conserved(1.0, 0.0, 1.0), r = conserved(0.125, 0.0, 0.1);
    return {f * l[0] + (1 - f) * r[0], f * l[1] + (1 - f) * r[1], f * l[2] + (1 - f) * r[2]};
}
// Shu-Osher, LUNA's exact cell averages (shu_osher.py:26-46) on [a, b].
inline State<1> shu_osher(double a, double b) {
    constexpr double amp = 0.2, k = 5, x0 = -4, rhoL = 3.857143, uL = 2.629369, pL = 10.33333;
    const double h = b - a, f = fraction_below(x0, a, b), lo = std::max(a, x0);
    double rho = f * rhoL + (1 - f);
    if (b > lo) rho += amp * (std::cos(k * lo) - std::cos(k * b)) / k / h;
    return {rho, f * rhoL * uL, f * (pL / (gamma_gas - 1) + 0.5 * rhoL * uL * uL) + (1 - f) / (gamma_gas - 1)};
}
// Woodward-Colella, LUNA's exact cell averages (woodward_colella.py:14-27).
inline State<1> woodward_colella(double a, double b) {
    const double f1 = fraction_below(0.1, a, b), f2 = fraction_below(0.9, a, b);
    const double p = f1 * 1000 + (f2 - f1) * 0.01 + (1 - f2) * 100;
    return {1.0, 0.0, p / (gamma_gas - 1)};
}

inline Problem problem(const std::string& name) {
    const Walls outflow{Wall::outflow, Wall::outflow};
    if (name == "sod") return {1.0, 0.0, 0.25, outflow, sod};
    if (name == "shu-osher") return {10.0, -5.0, 1.8, outflow, shu_osher};
    if (name == "woodward-colella")
        return {1.0, 0.0, 0.038, Walls{Wall::reflective, Wall::reflective}, woodward_colella};
    throw std::invalid_argument("hydro_1d: unknown problem " + name);
}

}  // namespace hydro::problems_1d
