#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

// Exact geometric predicates on positions snapped to a grid of 2^30 steps
// across the longest side of the periodic box, for host and device.
//
// Differences are exact integers (wrapped to the nearest periodic image).
// Each predicate first evaluates its determinant in double together with a
// bound on the rounding error, and decides there whenever the sign is
// certain; otherwise it repeats the evaluation exactly, the orientation in
// 128-bit integers and the in-sphere determinant in a 256-bit signed sum
// (exact for coordinates below 2^40). In-sphere ties are broken by symbolic
// perturbation, so every caller agrees on one triangulation even for
// cospherical points.
namespace exact_predicates {

using i64 = long long;
__extension__ typedef __int128 i128;
__extension__ typedef unsigned __int128 u128;

constexpr double scale = 1073741824.0;  // 2^30
constexpr double epsilon = 1.1102230246251565e-16;  // 2^-53

struct Offset {
    i64 v[3];
    KOKKOS_INLINE_FUNCTION i64& operator[](int a) { return v[a]; }
    KOKKOS_INLINE_FUNCTION const i64& operator[](int a) const { return v[a]; }
};

constexpr i64 period = i64(1) << 30;  // grid steps across the (longest) side

// A periodic box [0, side[0]) x [0, side[1]) x [0, side[2]). One grid step,
// the longest side / 2^30, serves every axis, and each side is a whole
// number period[a] of steps: the grid is isotropic, so the integer in-sphere
// test is the in-sphere test of the snapped points themselves. (Per-axis
// steps, 2^30 on every side, would stretch the axes differently, and the
// in-sphere test on the stretched integers would not be the one of the
// points.) A side that is not a whole number of steps is rounded to one
// (relative change below 2^-31 longest / side; none for sides in ratios of
// powers of two, e.g. 2 x 2 x 1/16); side[] holds the rounded sides. A cubic
// box is 2^30 steps on every axis, as before.
struct Box {
    double side[3] = {1, 1, 1};
    double inverse[3] = {1, 1, 1};  // 1 / side
    double step = 1 / scale;
    i64 period[3] = {exact_predicates::period, exact_predicates::period, exact_predicates::period};

    Box() = default;
    Box(double cube) : Box(cube, cube, cube) {}  // NOLINT: a cube converts implicitly
    Box(double x, double y, double z) {
        const double given[3] = {x, y, z}, longest = std::max({x, y, z});
        if (!(x > 0 && y > 0 && z > 0) || !std::isfinite(longest))
            throw std::invalid_argument("exact_predicates::Box: sides must be positive and finite");
        step = longest / scale;
        for (int a = 0; a < 3; ++a) {
            period[a] = std::max(std::llround(given[a] / step), 2LL);
            side[a] = given[a] == longest ? longest : static_cast<double>(period[a]) * step;
            inverse[a] = 1.0 / side[a];
        }
    }
    bool cubic() const { return side[0] == side[1] && side[1] == side[2]; }
    double shortest() const { return std::min({side[0], side[1], side[2]}); }
    double volume() const { return side[0] * side[1] * side[2]; }
};

// The grid step of coordinate x along `axis` of `box`, in [0, period[axis]):
// the one definition of "on the grid" every caller uses. In a cube, x / step
// is x / side * 2^30 to the bit (scaling by 2^30 is exact).
KOKKOS_INLINE_FUNCTION int snap(double x, const Box& box, int axis) {
    const i64 s = static_cast<i64>(Kokkos::floor(x / box.step)), steps = box.period[axis];
    if ((steps & (steps - 1)) == 0) return static_cast<int>(s & (steps - 1));  // a power of two
    const i64 r = s % steps;
    return static_cast<int>(r < 0 ? r + steps : r);
}
// In a cube of side `box`, in [0, 2^30).
KOKKOS_INLINE_FUNCTION int snap(double x, double box) {
    return static_cast<int>(static_cast<i64>(Kokkos::floor(x / box * scale)) & (period - 1));
}

// b - a on the periodic grid of `steps` per side, to the nearest image; a
// and b are snapped coordinates in [0, steps), so one shift suffices (no
// 64-bit modulo).
KOKKOS_INLINE_FUNCTION i64 delta(i64 a, i64 b, i64 steps = period) {
    i64 d = b - a;
    if (d >= steps / 2) d -= steps;
    if (d < -steps / 2) d += steps;
    return d;
}

KOKKOS_INLINE_FUNCTION Offset minus(const Offset& a, const Offset& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

// Signed 256-bit integer as four 64-bit limbs, two's complement.
struct Wide {
    unsigned long long limb[4] = {0, 0, 0, 0};
};

// w += a * b for signed a, b with |a|, |b| < 2^127.
KOKKOS_INLINE_FUNCTION void add_product(Wide& w, i128 a, i128 b) {
    const bool negative = (a < 0) != (b < 0);
    const u128 ua = a < 0 ? -u128(a) : u128(a), ub = b < 0 ? -u128(b) : u128(b);
    const u128 a0 = static_cast<unsigned long long>(ua), a1 = ua >> 64;
    const u128 b0 = static_cast<unsigned long long>(ub), b1 = ub >> 64;
    // |a b| = a1 b1 2^128 + (a1 b0 + a0 b1) 2^64 + a0 b0, in limbs l[0..3].
    unsigned long long l[4];
    const u128 low = a0 * b0, cross1 = a1 * b0, cross2 = a0 * b1, high = a1 * b1;
    const u128 mid = (low >> 64) + static_cast<unsigned long long>(cross1) + static_cast<unsigned long long>(cross2);
    l[0] = static_cast<unsigned long long>(low);
    l[1] = static_cast<unsigned long long>(mid);
    const u128 top = (mid >> 64) + (cross1 >> 64) + (cross2 >> 64) + high;
    l[2] = static_cast<unsigned long long>(top);
    l[3] = static_cast<unsigned long long>(top >> 64);
    if (negative) {  // two's complement negate across four limbs
        bool carry = true;
        for (int i = 0; i < 4; ++i) {
            l[i] = ~l[i] + carry;
            carry = carry && l[i] == 0;
        }
    }
    unsigned long long carry = 0;
    for (int i = 0; i < 4; ++i) {
        const u128 sum = u128(w.limb[i]) + l[i] + carry;
        w.limb[i] = static_cast<unsigned long long>(sum);
        carry = static_cast<unsigned long long>(sum >> 64);
    }
}

KOKKOS_INLINE_FUNCTION int sign(const Wide& w) {
    if (static_cast<long long>(w.limb[3]) < 0) return -1;
    return (w.limb[0] | w.limb[1] | w.limb[2] | w.limb[3]) ? 1 : 0;
}

KOKKOS_INLINE_FUNCTION i128 det3(const Offset& a, const Offset& b, const Offset& c) {
    return i128(a[0]) * (i128(b[1]) * c[2] - i128(b[2]) * c[1]) - i128(a[1]) * (i128(b[0]) * c[2] - i128(b[2]) * c[0]) +
           i128(a[2]) * (i128(b[0]) * c[1] - i128(b[1]) * c[0]);
}

// r = p x q, exactly.
KOKKOS_INLINE_FUNCTION void cross(const Offset& p, const Offset& q, i128* r) {
    r[0] = i128(p[1]) * q[2] - i128(p[2]) * q[1];
    r[1] = i128(p[2]) * q[0] - i128(p[0]) * q[2];
    r[2] = i128(p[0]) * q[1] - i128(p[1]) * q[0];
}

KOKKOS_INLINE_FUNCTION i128 lift(const Offset& a) {
    return i128(a[0]) * a[0] + i128(a[1]) * a[1] + i128(a[2]) * a[2];
}

// The circumcentre of the origin and a, b, c is n / (2 det3(a, b, c)), with
// n = |a|^2 (b x c) + |b|^2 (c x a) + |c|^2 (a x b); returns det3(a, b, c).
KOKKOS_INLINE_FUNCTION i128 circumcentre(const Offset& a, const Offset& b, const Offset& c, i128* n) {
    i128 bc[3], ca[3], ab[3];
    cross(b, c, bc);
    cross(c, a, ca);
    cross(a, b, ab);
    const i128 la = lift(a), lb = lift(b), lc = lift(c);
    for (int k = 0; k < 3; ++k) n[k] = la * bc[k] + lb * ca[k] + lc * ab[k];
    return a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2];
}

// det3 in double and its permanent (the same sum with absolute values), which
// bounds the rounding error: |det - exact| <= 8 eps permanent.
KOKKOS_INLINE_FUNCTION double det3_float(const Offset& a, const Offset& b, const Offset& c, double& permanent) {
    const double ax = double(a[0]), ay = double(a[1]), az = double(a[2]);
    const double bx = double(b[0]), by = double(b[1]), bz = double(b[2]);
    const double cx = double(c[0]), cy = double(c[1]), cz = double(c[2]);
    const double m0 = by * cz - bz * cy, m1 = bx * cz - bz * cx, m2 = bx * cy - by * cx;
    permanent = Kokkos::abs(ax) * (Kokkos::abs(by * cz) + Kokkos::abs(bz * cy)) +
                Kokkos::abs(ay) * (Kokkos::abs(bx * cz) + Kokkos::abs(bz * cx)) +
                Kokkos::abs(az) * (Kokkos::abs(bx * cy) + Kokkos::abs(by * cx));
    return ax * m0 - ay * m1 + az * m2;
}

// Sign of the orientation of (o, a, b, c) given as offsets from o.
KOKKOS_INLINE_FUNCTION int orient(const Offset& a, const Offset& b, const Offset& c) {
    double permanent;
    const double d = det3_float(a, b, c, permanent);
    const double bound = 8 * epsilon * permanent;
    if (d > bound) return 1;
    if (d < -bound) return -1;
    const i128 exact = det3(a, b, c);
    return exact > 0 ? 1 : exact < 0 ? -1 : 0;
}

// Whether p[4] lies inside the circumsphere of the tetrahedron p[0..3] (any
// orientation, not flat), all in one frame; id are the points' particle ids.
// Ties are broken by symbolic perturbation: the lift |p|^2 of each point
// grows by eps^(its id), lower ids dominating. The perturbed test is never a
// tie, because the term of p[4] is the tetrahedron's own orientation.
KOKKOS_INLINE_FUNCTION bool in_sphere(const Offset* p, const int* id) {
    // D = det [1 p_i |p_i|^2] over the five points; e inside means
    // sign(D) = -orientation. Expanded along the lift column.
    const Offset a = minus(p[1], p[0]), b = minus(p[2], p[0]), c = minus(p[3], p[0]), e = minus(p[4], p[0]);
    const int o = orient(a, b, c);
    int d = 0;
    {
        auto lift_float = [](const Offset& x) {
            return double(x[0]) * double(x[0]) + double(x[1]) * double(x[1]) + double(x[2]) * double(x[2]);
        };
        double pa, pb, pc, pe;
        const double ta = det3_float(b, c, e, pa), tb = det3_float(a, c, e, pb), tc = det3_float(a, b, e, pc),
                     te = det3_float(a, b, c, pe);
        const double la = lift_float(a), lb = lift_float(b), lc = lift_float(c), le = lift_float(e);
        const double value = -la * ta + lb * tb - lc * tc + le * te;
        const double bound = 20 * epsilon * (la * pa + lb * pb + lc * pc + le * pe);
        if (value > bound) d = 1;
        if (value < -bound) d = -1;
    }
    if (d == 0) {
        Wide w;
        add_product(w, det3(b, c, e), -lift(a));
        add_product(w, det3(a, c, e), lift(b));
        add_product(w, det3(a, b, e), -lift(c));
        add_product(w, det3(a, b, c), lift(e));
        d = sign(w);
    }
    if (d == 0) {
        // Cofactor of lift i: (-1)^i times the orientation of the other four,
        // taken in order of increasing particle id.
        int order[5] = {0, 1, 2, 3, 4};
        for (int i = 1; i < 5; ++i)
            for (int j = i; j > 0 && id[order[j - 1]] > id[order[j]]; --j) {
                const int swap = order[j - 1];
                order[j - 1] = order[j];
                order[j] = swap;
            }
        for (int r = 0; r < 5 && d == 0; ++r) {
            const int i = order[r];
            Offset q[4];
            for (int j = 0, k = 0; j < 5; ++j)
                if (j != i) q[k++] = p[j];
            d = (i % 2 ? -1 : 1) * orient(minus(q[1], q[0]), minus(q[2], q[0]), minus(q[3], q[0]));
        }
    }
    return d * o < 0;
}

// Snapped particle positions, point(i, a) for a < 3, in [0, period[a]) so
// 32 bits hold them exactly; padded to 16 bytes (one aligned load per
// point). The periods are the box's (a unit cube unless given).
template <class MemorySpace>
struct Points {
    using View = Kokkos::View<int* [4], Kokkos::LayoutRight, MemorySpace>;
    View view;
    i64 period[3] = {exact_predicates::period, exact_predicates::period, exact_predicates::period};

    Points() = default;
    Points(const std::string& label, std::size_t n, const Box& box = {}) : view(label, n) { periods(box.period); }
    Points(const View& v, const i64* steps) : view(v) { periods(steps); }
    KOKKOS_INLINE_FUNCTION int& operator()(int i, int a) const { return view(i, a); }
    KOKKOS_INLINE_FUNCTION std::size_t extent(int r) const { return view.extent(r); }

private:
    void periods(const i64* steps) {
        for (int a = 0; a < 3; ++a) period[a] = steps[a];
    }
};

// point(i) = the snapped x(i) for positions x in `box` (point's box).
template <class ExecSpace, class X>
void snap(const ExecSpace& exec, const X& x, const Box& box, const Points<typename ExecSpace::memory_space>& point) {
    Kokkos::parallel_for(
        "exact_predicates::snap", Kokkos::RangePolicy<ExecSpace>(exec, 0, x.extent(0)), KOKKOS_LAMBDA(int i) {
            for (int a = 0; a < 3; ++a) point(i, a) = snap(x(i, a), box, a);
        });
}

// The predicates on particles, by index into snapped positions.
template <class P>
KOKKOS_INLINE_FUNCTION Offset offset(const P& point, int from, int to) {
    return {delta(point(from, 0), point(to, 0), point.period[0]), delta(point(from, 1), point(to, 1), point.period[1]),
            delta(point(from, 2), point(to, 2), point.period[2])};
}

// Orientation sign of tetrahedron v.
template <class P>
KOKKOS_INLINE_FUNCTION int orient(const P& point, const int* v) {
    return orient(offset(point, v[0], v[1]), offset(point, v[0], v[2]), offset(point, v[0], v[3]));
}

// Whether particle e lies inside the circumsphere of tetrahedron v, whose
// offsets from v[0] are o (o[0] = 0).
template <class P>
KOKKOS_INLINE_FUNCTION bool in_sphere(const P& point, const int* v, const Offset* o, int e) {
    const Offset p[5] = {o[0], o[1], o[2], o[3], offset(point, v[0], e)};
    const int id[5] = {v[0], v[1], v[2], v[3], e};
    return in_sphere(p, id);
}

template <class P>
KOKKOS_INLINE_FUNCTION bool in_sphere(const P& point, const int* v, int e) {
    const Offset o[4] = {{0, 0, 0}, offset(point, v[0], v[1]), offset(point, v[0], v[2]), offset(point, v[0], v[3])};
    return in_sphere(point, v, o, e);
}

}  // namespace exact_predicates
