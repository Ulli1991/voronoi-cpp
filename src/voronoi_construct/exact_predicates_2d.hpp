#pragma once

#include "exact_predicates.hpp"

// 2D counterpart of exact_predicates.hpp: used by polygon_clip (clipping) and tri_repair/tri_finish (repair).
// The exact predicates in 2D on offsets of snapped points (|offset| <= 2³⁰): orientation in 64 bits, in-circle in
// 128 bits (terms below 2¹²²), behind a double filter, with the same symbolic perturbation as 3D.
namespace exact_predicates {

using Offset2 = OffsetN<2>;

KOKKOS_INLINE_FUNCTION Offset2 minus(const Offset2& a, const Offset2& b) { return {a[0] - b[0], a[1] - b[1]}; }
KOKKOS_INLINE_FUNCTION i64 cross(const Offset2& a, const Offset2& b) { return a[0] * b[1] - a[1] * b[0]; }
KOKKOS_INLINE_FUNCTION i64 lift(const Offset2& a) { return a[0] * a[0] + a[1] * a[1]; }
KOKKOS_INLINE_FUNCTION int sign_of(i64 v) { return v > 0 ? 1 : v < 0 ? -1 : 0; }

// Whether a triangle with edges a, b from one corner fits one periodic image: all three edges under half a period.
// Wider ones have no meaning here, as no images are stored: their area depends on the corner they are read from.
KOKKOS_INLINE_FUNCTION bool one_image(const Offset2& a, const Offset2& b, const i64* period) {
    for (int k = 0; k < 2; ++k) {
        if (2 * Kokkos::abs(a[k]) >= period[k] || 2 * Kokkos::abs(b[k]) >= period[k] || 2 * Kokkos::abs(b[k] - a[k]) >= period[k])
            return false;
    }
    return true;
}

// Sign of det [a la; b lb; e le] of offsets from one point: -orientation(a, b) if e is inside the circle through
// the origin, a, b; +orientation(a, b) if outside; 0 on it.
// Used by: in_circle.
KOKKOS_INLINE_FUNCTION int circle_sign(const Offset2& a, const Offset2& b, const Offset2& e) {
    const double ax = double(a[0]), ay = double(a[1]), bx = double(b[0]), by = double(b[1]);
    const double ex = double(e[0]), ey = double(e[1]);
    const double la = ax * ax + ay * ay, lb = bx * bx + by * by, le = ex * ex + ey * ey;
    const double mx = la * bx - lb * ax, my = la * by - lb * ay, det = ax * by - ay * bx;
    const double value = ey * mx - ex * my + le * det;
    const double bound = 16 * epsilon *
                         (Kokkos::abs(ey) * (la * Kokkos::abs(bx) + lb * Kokkos::abs(ax)) +
                          Kokkos::abs(ex) * (la * Kokkos::abs(by) + lb * Kokkos::abs(ay)) +
                          le * (Kokkos::abs(ax * by) + Kokkos::abs(ay * bx)));
    if (value > bound) return 1;
    if (value < -bound) return -1;
    const i128 exact = i128(cross(b, e)) * lift(a) - i128(cross(a, e)) * lift(b) + i128(cross(a, b)) * lift(e);
    return exact > 0 ? 1 : exact < 0 ? -1 : 0;
}

// Whether p[3] lies inside the circle through p[0..2] (any orientation, not flat, one frame); id are particle ids.
// Perturbed, so never a tie: the term of p[3] is the triangle's own orientation.
// Used by: polygon_clip::beyond, tri_repair, tri_finish.
KOKKOS_INLINE_FUNCTION bool in_circle(const Offset2* p, const int* id) {
    const Offset2 a = minus(p[1], p[0]), b = minus(p[2], p[0]), e = minus(p[3], p[0]);
    const int o = sign_of(cross(a, b));
    int d = circle_sign(a, b, e);
    if (d == 0) {
        // Cofactor of lift i: -(-1)^i times the orientation of the other three, taken in order of increasing particle
        // id.
        int order[4] = {0, 1, 2, 3};
        for (int i = 1; i < 4; ++i)
            for (int j = i; j > 0 && id[order[j - 1]] > id[order[j]]; --j) {
                const int swap = order[j - 1];
                order[j - 1] = order[j];
                order[j] = swap;
            }
        for (int r = 0; r < 4 && d == 0; ++r) {
            const int i = order[r];
            Offset2 q[3];
            for (int j = 0, k = 0; j < 4; ++j)
                if (j != i) q[k++] = p[j];
            d = (i % 2 ? 1 : -1) * sign_of(cross(minus(q[1], q[0]), minus(q[2], q[0])));
        }
    }
    return d * o < 0;
}

}  // namespace exact_predicates
