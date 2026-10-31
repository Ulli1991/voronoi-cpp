// Exact predicates: orientation and in-sphere signs equal an independent
// integer evaluation, near-degenerate and large inputs included; ties are
// broken consistently; periodic offsets take the nearest image.

#include "check.hpp"
#include "exact_predicates.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>

namespace {

namespace ep = exact_predicates;
using ep::i128;
using ep::Offset;

i128 det(const Offset& a, const Offset& b, const Offset& c) {
    return i128(a[0]) * (i128(b[1]) * c[2] - i128(b[2]) * c[1]) + i128(a[1]) * (i128(b[2]) * c[0] - i128(b[0]) * c[2]) +
           i128(a[2]) * (i128(b[0]) * c[1] - i128(b[1]) * c[0]);
}

int sign(i128 v) { return v > 0 ? 1 : v < 0 ? -1 : 0; }

// Reference in-sphere for small coordinates, from the circumcentre: with p0
// at the origin, e is inside iff |e|^2 < 2 e.C, C = N / (2 det(a, b, c)).
// Returns +1 inside, -1 outside, 0 on the sphere.
int reference_in_sphere(const Offset* p) {
    Offset q[4];
    for (int k = 0; k < 4; ++k) q[k] = ep::minus(p[k + 1], p[0]);
    const Offset &a = q[0], &b = q[1], &c = q[2], &e = q[3];
    const i128 d = det(a, b, c);
    auto cross = [](const Offset& u, const Offset& v, int axis) {
        const int i = (axis + 1) % 3, j = (axis + 2) % 3;
        return i128(u[i]) * v[j] - i128(u[j]) * v[i];
    };
    i128 e_dot_n = 0;
    for (int axis = 0; axis < 3; ++axis)
        e_dot_n += e[axis] * (ep::lift(a) * cross(b, c, axis) + ep::lift(b) * cross(c, a, axis) +
                              ep::lift(c) * cross(a, b, axis));
    return sign(e_dot_n - ep::lift(e) * d) * sign(d);
}

Offset random_offset(std::mt19937_64& rng, long long range) {
    std::uniform_int_distribution<long long> u(-range, range);
    return {u(rng), u(rng), u(rng)};
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    Check check;
    std::mt19937_64 rng(5);

    // Periodic offsets on the 2^30 grid.
    const ep::i64 period = ep::period;
    check.expect(ep::snap(0.5, 1.0) == period / 2 && ep::snap(0.0, 1.0) == 0 && ep::snap(1.0, 2.0) == period / 2 &&
                     ep::snap(3.0 * 0.75, 3.0) == 3 * period / 4,
                 "snap maps the box [0, box) onto [0, 2^30) for any box");
    check.expect(ep::snap(2.0, 2.0) == 0 && ep::snap(-0.5 / ep::scale, 1.0) == period - 1,
                 "snap wraps positions on or just outside the box edge");
    check.expect(ep::delta(0, period - 1) == -1 && ep::delta(period - 1, 0) == 1 && ep::delta(5, 9) == 4,
                 "delta takes the nearest periodic image");

    // Boxes of three sides: one step for every axis (the longest side /
    // 2^30), each side a whole number of steps; a cube is 2^30 per side and
    // snaps as before.
    {
        const ep::Box slab(2, 2, 0.0625), odd(1, 2, 3), cube(3.0);
        check.expect(slab.period[0] == period && slab.period[2] == period / 32 && slab.side[2] == 0.0625 &&
                         slab.step == 2 / ep::scale,
                     "a 2 x 2 x 1/16 slab: 2^30, 2^30, 2^25 steps of one length, sides exact");
        check.expect(odd.period[0] == 357913941 && odd.period[1] == 715827883 && odd.period[2] == period &&
                         std::abs(odd.side[0] - 1) < 2e-9 && odd.side[2] == 3 && odd.step == 3 / ep::scale,
                     "a 1 x 2 x 3 box: sides rounded to whole steps of 3 / 2^30");
        bool same = cube.cubic() && !odd.cubic();
        std::uniform_real_distribution<double> in_cube(-0.1, 3.1);
        for (int i = 0; i < 10000; ++i) {
            const double x = in_cube(rng);
            for (int a = 0; a < 3; ++a) same = same && ep::snap(x, cube, a) == ep::snap(x, 3.0);
        }
        check.expect(same, "a cubic Box snaps as the cube's snap");
        bool wraps = true;
        for (int a = 0; a < 3; ++a) {
            const ep::i64 p = odd.period[a];
            wraps = wraps && ep::snap(odd.side[a], odd, a) == 0 && ep::snap(-0.5 * odd.step, odd, a) == p - 1 &&
                    ep::snap(5.5 * odd.step, odd, a) == 5 && ep::snap(odd.side[a] + 5.5 * odd.step, odd, a) == 5 &&
                    ep::delta(0, p - 1, p) == -1 && ep::delta(p - 1, 0, p) == 1;
        }
        check.expect(wraps, "per axis: snap wraps at the (rounded) side, delta at its period");
        bool exact = true;
        for (int i = 0; i < 10000; ++i)
            for (int a = 0; a < 3; ++a) {
                const int k = static_cast<int>(rng() % odd.period[a]);
                exact = exact && ep::snap(k * odd.step, odd, a) == k;
            }
        check.expect(exact, "a 1 x 2 x 3 box: grid positions snap to themselves");
    }

    // Orientation equals the exact determinant's sign, also for nearly
    // coplanar points with large coordinates where the double filter cannot
    // decide.
    int wrong = 0, coplanar = 0;
    for (int i = 0; i < 20000; ++i) {
        const Offset a = random_offset(rng, 1 << 29), b = random_offset(rng, 1 << 29);
        Offset c = random_offset(rng, 1 << 29);
        if (i % 2) {  // c = a + b, or one grid step off it
            const Offset s = random_offset(rng, 1);
            for (int k = 0; k < 3; ++k) c[k] = a[k] + b[k] + (i % 4 == 1 ? 0 : s[k]);
        }
        const int o = ep::orient(a, b, c);
        wrong += o != sign(det(a, b, c));
        coplanar += o == 0;
    }
    check.expect(wrong == 0, "orient equals the exact sign (" + std::to_string(coplanar) + " of 20000 coplanar)");

    // In-sphere equals the circumcentre reference wherever that is not a tie.
    wrong = 0;
    int ties = 0;
    for (int i = 0; i < 20000; ++i) {
        const long long range = i % 2 ? 8 : 1000;  // small range: many cospherical sets
        Offset p[5];
        for (auto& q : p) q = random_offset(rng, range);
        if (ep::orient(ep::minus(p[1], p[0]), ep::minus(p[2], p[0]), ep::minus(p[3], p[0])) == 0) continue;
        const int id[5] = {0, 1, 2, 3, 4};
        const int reference = reference_in_sphere(p);
        if (reference == 0) {
            ++ties;
            continue;
        }
        wrong += ep::in_sphere(p, id) != (reference > 0);
    }
    check.expect(wrong == 0, "in_sphere equals the reference off ties (" + std::to_string(ties) + " ties skipped)");

    // Large coordinates: one grid step inside or outside a sphere of radius 2^28.
    {
        const ep::i64 r = ep::i64(1) << 28;
        const int id[5] = {0, 1, 2, 3, 4};
        Offset p[5] = {{r, 0, 0}, {-r, 0, 0}, {0, r, 0}, {0, 0, r}, {0, -r + 1, 0}};
        const bool inside = ep::in_sphere(p, id);
        p[4] = {0, -r - 1, 0};
        const bool outside = !ep::in_sphere(p, id);
        check.expect(inside && outside, "in_sphere resolves one grid step at radius 2^28");
    }

    // Ties: for five cospherical points the perturbed answer must not depend
    // on the order of the tetrahedron's corners, and a triangle abc with d
    // and e on opposite sides is locally Delaunay from both sides alike.
    int order_dependent = 0, one_sided = 0, cospherical = 0;
    const Offset sphere[] = {{3, 0, 0}, {-3, 0, 0}, {0, 3, 0}, {0, -3, 0}, {0, 0, 3}, {0, 0, -3}, {1, 2, 2},
                             {-1, 2, 2}, {1, -2, 2}, {1, 2, -2}, {2, 1, -2}, {-2, -2, 1}, {2, -1, 2}};
    const int count = static_cast<int>(std::size(sphere));
    for (int i = 0; i < 5000; ++i) {
        int pick[5];
        for (int k = 0; k < 5; ++k) {
            bool fresh;
            do {
                pick[k] = static_cast<int>(rng() % count);
                fresh = std::find(pick, pick + k, pick[k]) == pick + k;
            } while (!fresh);
        }
        Offset p[5];
        int id[5];
        for (int k = 0; k < 5; ++k) {
            p[k] = sphere[pick[k]];
            id[k] = pick[k];
        }
        const Offset ab = ep::minus(p[1], p[0]), ac = ep::minus(p[2], p[0]);
        const int side_d = ep::orient(ab, ac, ep::minus(p[3], p[0])), side_e = ep::orient(ab, ac, ep::minus(p[4], p[0]));
        if (side_d == 0 || side_e == 0) continue;
        ++cospherical;
        const bool answer = ep::in_sphere(p, id);
        // Same tetrahedron, corners rotated.
        const Offset rotated[5] = {p[1], p[2], p[3], p[0], p[4]};
        const int rotated_id[5] = {id[1], id[2], id[3], id[0], id[4]};
        order_dependent += ep::in_sphere(rotated, rotated_id) != answer;
        if (side_d != side_e) {  // abcd with e, and abce with d
            const Offset other[5] = {p[0], p[1], p[2], p[4], p[3]};
            const int other_id[5] = {id[0], id[1], id[2], id[4], id[3]};
            one_sided += ep::in_sphere(other, other_id) != answer;
        }
    }
    check.expect(order_dependent == 0, "ties do not depend on the corner order (" + std::to_string(cospherical) +
                                           " cospherical sets)");
    check.expect(one_sided == 0, "ties agree on both sides of a shared triangle");

    // Particle-level predicates on snapped positions take the nearest image:
    // a tetrahedron across the periodic boundary keeps its orientation.
    {
        ep::Points<Kokkos::HostSpace> point("point", 5);
        const double x[5][3] = {{0.999, 0.5, 0.5}, {0.001, 0.5, 0.5}, {0.999, 0.502, 0.5}, {0.999, 0.5, 0.502},
                                {0.0005, 0.501, 0.501}};
        for (int i = 0; i < 5; ++i)
            for (int a = 0; a < 3; ++a) point(i, a) = ep::snap(x[i][a], 1.0);
        const int tet[4] = {0, 1, 2, 3}, flipped[4] = {0, 1, 3, 2};
        check.expect(ep::orient(point, tet) == 1 && ep::orient(point, flipped) == -1,
                     "orient on particles across the periodic boundary");
        check.expect(ep::in_sphere(point, tet, 4), "in_sphere on particles across the periodic boundary");

        // The same across the thin side of a slab (z period 2^25 steps): the
        // points with the axes rolled (y, z, x), x across the boundary as z;
        // the same signs as the copy shifted by half the slab, inside it.
        const ep::Box slab(2, 2, 0.0625);
        ep::Points<Kokkos::HostSpace> thin("thin", 5, slab), inside("inside", 5, slab);
        for (int i = 0; i < 5; ++i) {
            const double z = x[i][0] < 0.5 ? x[i][0] : slab.side[2] - (1 - x[i][0]);
            const double q[3] = {1.0 + x[i][1] - 0.5, 1.0 + x[i][2] - 0.5, z};
            for (int a = 0; a < 3; ++a) {
                thin(i, a) = ep::snap(q[a], slab, a);
                inside(i, a) = ep::snap(a == 2 ? std::fmod(q[a] + 0.5 * slab.side[2], slab.side[2]) : q[a], slab, a);
            }
        }
        check.expect(ep::orient(thin, tet) == 1 && ep::orient(inside, tet) == 1 && ep::in_sphere(thin, tet, 4) &&
                         ep::in_sphere(inside, tet, 4),
                     "orient and in_sphere across the thin side of a slab (nearest image per axis)");
    }
    return check.failed() != 0;
}
