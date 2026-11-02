#pragma once

// The exact solution of the 1D Riemann problem of an ideal gas (Toro 2009,
// ch. 4: the pressure function, Newton's method for the star pressure, the
// sampling), for tests: Sod's tube against the exact cell averages, as
// LUNA's exact_riemann.cell_average (src/luna/hydro/exact_riemann.py:105-113:
// conservative averages of sub-samples per cell).

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace exact_riemann {

struct Primitive {
    double rho, u, p;
};

class Solver {
public:
    Solver(Primitive left, Primitive right, double gamma) : L_(left), R_(right), g_(gamma) {
        cL_ = std::sqrt(g_ * L_.p / L_.rho);
        cR_ = std::sqrt(g_ * R_.p / R_.rho);
        if (2 * (cL_ + cR_) / (g_ - 1) <= R_.u - L_.u) throw std::invalid_argument("exact_riemann: vacuum");
        double p = std::max(1e-12, 0.5 * (L_.p + R_.p));  // Newton from the arithmetic mean
        for (int it = 0; it < 100; ++it) {
            double fL, dL, fR, dR;
            f(p, L_, cL_, fL, dL);
            f(p, R_, cR_, fR, dR);
            const double next = std::max(1e-14, p - (fL + fR + R_.u - L_.u) / (dL + dR));
            const bool done = 2 * std::abs(next - p) / (next + p) < 1e-15;
            p = next;
            if (done) break;
        }
        p_ = p;
        double fL, dL, fR, dR;
        f(p_, L_, cL_, fL, dL);
        f(p_, R_, cR_, fR, dR);
        u_ = 0.5 * (L_.u + R_.u + fR - fL);
    }

    // The solution at s = (x - x0) / t.
    Primitive sample(double s) const {
        const double g = g_;
        if (s <= u_) {  // left of the contact
            if (p_ > L_.p) {  // left shock
                const double SL = L_.u - cL_ * std::sqrt((g + 1) / (2 * g) * p_ / L_.p + (g - 1) / (2 * g));
                if (s <= SL) return L_;
                const double r = p_ / L_.p, k = (g - 1) / (g + 1);
                return {L_.rho * (r + k) / (k * r + 1), u_, p_};
            }
            const double head = L_.u - cL_, cs = cL_ * std::pow(p_ / L_.p, (g - 1) / (2 * g)), tail = u_ - cs;
            if (s <= head) return L_;
            if (s >= tail) return {L_.rho * std::pow(p_ / L_.p, 1 / g), u_, p_};
            const double c = 2 / (g + 1) * (cL_ + (g - 1) / 2 * (L_.u - s));
            const double u = 2 / (g + 1) * (cL_ + (g - 1) / 2 * L_.u + s);
            return {L_.rho * std::pow(c / cL_, 2 / (g - 1)), u, L_.p * std::pow(c / cL_, 2 * g / (g - 1))};
        }
        if (p_ > R_.p) {  // right shock
            const double SR = R_.u + cR_ * std::sqrt((g + 1) / (2 * g) * p_ / R_.p + (g - 1) / (2 * g));
            if (s >= SR) return R_;
            const double r = p_ / R_.p, k = (g - 1) / (g + 1);
            return {R_.rho * (r + k) / (k * r + 1), u_, p_};
        }
        const double head = R_.u + cR_, cs = cR_ * std::pow(p_ / R_.p, (g - 1) / (2 * g)), tail = u_ + cs;
        if (s >= head) return R_;
        if (s <= tail) return {R_.rho * std::pow(p_ / R_.p, 1 / g), u_, p_};
        const double c = 2 / (g + 1) * (cR_ - (g - 1) / 2 * (R_.u - s));
        const double u = 2 / (g + 1) * (-cR_ + (g - 1) / 2 * R_.u + s);
        return {R_.rho * std::pow(c / cR_, 2 / (g - 1)), u, R_.p * std::pow(c / cR_, 2 * g / (g - 1))};
    }

    // Conservative average over [a, b] at time t of the problem with its
    // jump at x0 (`samples` midpoints), returned as primitives.
    Primitive average(double a, double b, double x0, double t, int samples = 256) const {
        double m = 0, mom = 0, e = 0;
        for (int k = 0; k < samples; ++k) {
            const double x = a + (b - a) * (k + 0.5) / samples;
            const Primitive w = sample((x - x0) / t);
            m += w.rho;
            mom += w.rho * w.u;
            e += w.p / (g_ - 1) + 0.5 * w.rho * w.u * w.u;
        }
        m /= samples;
        mom /= samples;
        e /= samples;
        const double u = mom / m;
        return {m, u, (g_ - 1) * (e - 0.5 * m * u * u)};
    }
    double star_pressure() const { return p_; }
    double star_velocity() const { return u_; }

private:
    // Toro's pressure function f_K(p) and its derivative (eqs. 4.6, 4.7, 4.37).
    void f(double p, const Primitive& K, double c, double& value, double& derivative) const {
        const double g = g_;
        if (p > K.p) {
            const double A = 2 / ((g + 1) * K.rho), B = (g - 1) / (g + 1) * K.p, q = std::sqrt(A / (p + B));
            value = (p - K.p) * q;
            derivative = q * (1 - 0.5 * (p - K.p) / (B + p));
        } else {
            value = 2 * c / (g - 1) * (std::pow(p / K.p, (g - 1) / (2 * g)) - 1);
            derivative = 1 / (K.rho * c) * std::pow(p / K.p, -(g + 1) / (2 * g));
        }
    }
    Primitive L_, R_;
    double g_, cL_, cR_, p_ = 0, u_ = 0;
};

}  // namespace exact_riemann
