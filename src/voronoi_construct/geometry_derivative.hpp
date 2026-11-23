#pragma once

#include "voronoi_mesh.hpp"

#include <cstddef>
#include <stdexcept>

// Geometry derivatives at fixed topology (#63): how the cell volumes V_i, cell centroids c_i, face area vectors A_f and
// face centroids x_f of a Voronoi mesh change with its generator positions x, for the triangles or tetrahedra, faces and
// cell lists that update() and update_cells() produced. A tangent maps position changes to geometry changes; the adjoint
// is its transpose. Users: shape and mesh optimisation (centroidal tessellations, energies of V and A), sensitivities and
// inverse problems on moving meshes. Entry point: voronoi::GeometryDerivative<Mesh>.
//
// The derivative is that of the formulas the mesh evaluates, from the offsets between generators in whole grid steps
// (taken as continuous; through a periodic image the offset's derivative is 1):
//  - a Voronoi vertex is the circumcentre c of a triangle or tetrahedron, d_k . c = |d_k|^2 / 2 for the offsets d_k of
//    its other corners, so d_k . dc = dd_k . (d_k - c) (Centre);
//  - a 3D face is the fan of triangles (v_0, v_{m-1}, v_m) of its ring of vertices: A = sum of their area vectors, x_f the
//    mean of their centroids weighted by |T_m . d| (tet_mesh::FaceGeometry); a 2D face is the segment between two vertices;
//  - a cell's volume is sum_f |A_f . d_f| / (2D), its centroid the pyramids' mean at D / (D + 1) of the way to each x_f.
// The weights |T_m . d| depend on the direction of d only through T_m: every T_m is parallel to d (the face lies in the
// bisector plane), so that term of the derivative is zero and is left out.
namespace voronoi {

namespace geometry_derivative {

using tet_mesh::Vec2;
using tet_mesh::Vec3;
// The vector of D dimensions.
template <int D>
using VecN = tet_mesh::VecN<D>;
using shell_voronoi::cross;
using shell_voronoi::dot;

// +1 for v >= 0, else -1: the derivative of |v|, the same in the tangent and the adjoint.
KOKKOS_INLINE_FUNCTION double sign_of(double v) { return v >= 0 ? 1.0 : -1.0; }

// The linearised circumcentre of one triangle (D = 2) or tetrahedron (D = 3) with offsets d_k of corners 1..D from corner
// 0 (grid steps) and centre c relative to corner 0: dc = sum_k [dd_k . (d_k - c)] n_k, n_k the columns of the inverse of
// the matrix with rows d_k. Degree 0 in the offsets, so the grid step cancels: dd in box units gives dc in box units.
template <int D>
struct Centre {
    VecN<D> e[D];  // d_k - c
    VecN<D> n[D];  // columns of the inverse

    KOKKOS_INLINE_FUNCTION Centre(const VecN<D>* d, VecN<D> c) {
        if constexpr (D == 3) {
            const double inverse = 1.0 / dot(d[0], cross(d[1], d[2]));
            n[0] = cross(d[1], d[2]) * inverse;
            n[1] = cross(d[2], d[0]) * inverse;
            n[2] = cross(d[0], d[1]) * inverse;
        } else {
            const double inverse = 1.0 / (d[0].x * d[1].y - d[0].y * d[1].x);
            n[0] = Vec2{d[1].y, -d[1].x} * inverse;
            n[1] = Vec2{-d[0].y, d[0].x} * inverse;
        }
        for (int k = 0; k < D; ++k) e[k] = d[k] - c;
    }
    // dc from the offsets' changes dd_k = dx_{k+1} - dx_0.
    KOKKOS_INLINE_FUNCTION VecN<D> tangent(const VecN<D>* dd) const {
        VecN<D> dc{};
        for (int k = 0; k < D; ++k) dc = dc + n[k] * dot(dd[k], e[k]);
        return dc;
    }
    // The transpose: the bars of dd_k from the bar of c.
    KOKKOS_INLINE_FUNCTION void adjoint(VecN<D> cbar, VecN<D>* ddbar) const {
        for (int k = 0; k < D; ++k) ddbar[k] = e[k] * dot(cbar, n[k]);
    }
    // The bar of corner `corner`'s position from the bar of the absolute centre x_0 + c.
    KOKKOS_INLINE_FUNCTION VecN<D> corner_bar(VecN<D> cbar, int corner) const {
        VecN<D> ddbar[D];
        adjoint(cbar, ddbar);
        if (corner > 0) return ddbar[corner - 1];
        VecN<D> sum = cbar;
        for (int k = 0; k < D; ++k) sum = sum - ddbar[k];
        return sum;
    }
};

// Row i of a (rows x D) view as a vector, or zero if the view is empty (an absent bar).
template <int D, class V>
KOKKOS_INLINE_FUNCTION VecN<D> row_or_zero(const V& v, int i) {
    if (v.extent(0) == 0) return VecN<D>{};
    return tet_mesh::row<D>(v, i);
}
// Adds w to row i of a (rows x D) view.
template <class V>
KOKKOS_INLINE_FUNCTION void add_row(const V& v, int i, Vec3 w) {
    v(i, 0) += w.x;
    v(i, 1) += w.y;
    v(i, 2) += w.z;
}
// The same in 2D.
template <class V>
KOKKOS_INLINE_FUNCTION void add_row(const V& v, int i, Vec2 w) {
    v(i, 0) += w.x;
    v(i, 1) += w.y;
}

// Tetrahedron t's linearised circumcentre, its corners v sorted as tet_mesh::stored_centre sorts them; the centre read
// from Mesh::centre (exact for slivers), relative to corner 0 in grid steps.
template <class X, class Vertex, class Centres>
KOKKOS_INLINE_FUNCTION Centre<3> tet_centre(const X& x, const exact_predicates::Box& box, const Vertex& vertex,
                                            const Centres& centre, int t, int* v) {
    for (int k = 0; k < 4; ++k) v[k] = vertex(t, k);
    tet_mesh::sort_corners(v);
    Vec3 d[3];
    tet_mesh::rounded_offsets(x, box, v, d);
    double c[3];
    for (int a = 0; a < 3; ++a) {
        const double e = centre(t, a) - Kokkos::nearbyint(x(v[0], a) * box.per_step);
        const double p = static_cast<double>(box.period[a]);
        c[a] = e > 0.5 * p ? e - p : e < -0.5 * p ? e + p : e;
    }
    return Centre<3>(d, Vec3{c[0], c[1], c[2]});
}

// Triangle t's linearised circumcentre (corners(t, 0..2), counter-clockwise, lowest id first), from the grid points as
// tri_repair::centres has them.
template <class Point, class Corners>
KOKKOS_INLINE_FUNCTION Centre<2> triangle_centre(const Point& point, const exact_predicates::BoxN<2>& box,
                                                 const Corners& corners, int t) {
    const int px = static_cast<int>(box.period[0]), py = static_cast<int>(box.period[1]);
    const int v0 = corners(t, 0), v1 = corners(t, 1), v2 = corners(t, 2);
    const int x0 = point(v0, 0), y0 = point(v0, 1);
    const Vec2 d[2] = {{tri_mesh::steps(x0, point(v1, 0), px), tri_mesh::steps(y0, point(v1, 1), py)},
                       {tri_mesh::steps(x0, point(v2, 0), px), tri_mesh::steps(y0, point(v2, 1), py)}};
    const tri_mesh::Circle circle(d[0].x, d[0].y, d[1].x, d[1].y);
    const double scale = 0.5 / circle.det;
    return Centre<2>(d, Vec2{circle.my * scale, -circle.mx * scale});
}

// One 3D face as the fan of its ring (tet_mesh::FaceGeometry::face_as, written arithmetic), with the sums its derivative
// needs. Filled by Faces3::walk.
struct Fan {
    int i = 0, j = 0, begin = 0, end = 0;  // owner, neighbour, ring entries [begin, end)
    Vec3 dj{}, area{}, centroid{}, weighted{};
    double weight = 0, sign = 1, projected = 0;  // the sum of |T_m . dj|, the orientation, area . dj (weighted: of
                                                 // the vertices relative to the first)
};

// The 3D faces and their derivatives, from the mesh's FaceGeometry. Copy into kernels.
template <class Memory>
struct Faces3 {
    tet_mesh::FaceGeometry<Memory> g;

    KOKKOS_INLINE_FUNCTION int slot(int f) const { return g.slot.data() ? g.slot(f) : f; }
    KOKKOS_INLINE_FUNCTION Vec3 vertex(int m, const tet_mesh::GridPoint& q) const {
        return tet_mesh::from_point(g.centre, g.ring(m), q, g.box.step);
    }
    // Face f's area, centroid and fan sums.
    KOKKOS_INLINE_FUNCTION Fan walk(int f) const {
        Fan fan;
        const int s = slot(f);
        fan.i = g.owner(s);
        fan.j = g.neighbour(s);
        fan.begin = g.ring_start(s);
        fan.end = fan.begin + g.ring_length(s);
        const tet_mesh::GridPoint q(g.x, fan.i, g.box);
        fan.dj = tet_mesh::snapped_difference<3>(tet_mesh::row(g.x, fan.j) - tet_mesh::row(g.x, fan.i), g.box);
        const Vec3 first = vertex(fan.begin, q);
        Vec3 a = vertex(fan.begin + 1, q) - first, twice{0, 0, 0};
        for (int m = fan.begin + 2; m < fan.end; ++m) {
            const Vec3 b = vertex(m, q) - first;
            const Vec3 t = cross(a, b);
            twice = twice + t;
            const double w = Kokkos::abs(dot(t, fan.dj));
            fan.weight += w;
            fan.weighted = fan.weighted + (a + b) * w;
            a = b;
        }
        fan.area = twice * 0.5;
        fan.projected = dot(fan.area, fan.dj);
        if (fan.projected < 0) {
            fan.sign = -1;
            fan.area = fan.area * -1.0;
            fan.projected = -fan.projected;
        }
        fan.centroid = fan.weight > 0 ? first + fan.weighted * (1.0 / (3.0 * fan.weight)) : first;
        return fan;
    }
    // dA and dx_f of face f from the vertices' changes dct (absolute, per tetrahedron) and dx (per cell).
    template <class DC, class DX>
    KOKKOS_INLINE_FUNCTION void tangent(int f, const DC& dct, const DX& dx, Vec3& dA, Vec3& dxf) const {
        const int s = slot(f);
        const int i = g.owner(s), j = g.neighbour(s), begin = g.ring_start(s), end = begin + g.ring_length(s);
        const tet_mesh::GridPoint q(g.x, i, g.box);
        const Vec3 dj = tet_mesh::snapped_difference<3>(tet_mesh::row(g.x, j) - tet_mesh::row(g.x, i), g.box);
        const Vec3 dxi = tet_mesh::row(dx, i);
        auto dvertex = [&](int m) { return tet_mesh::row(dct, g.ring(m)) - dxi; };
        // Vertices relative to the first: x_f = v_0 + sum w_m (a_m + b_m) / (3 sum w_m), the same function as the
        // mesh's (v_0 + sum w_m (v_{m-1} + v_m) / sum w_m) / 3, but its derivative does not cancel on small faces.
        const Vec3 first = vertex(begin, q), dfirst = dvertex(begin);
        Vec3 a = vertex(begin + 1, q) - first, da = dvertex(begin + 1) - dfirst;
        Vec3 twice{0, 0, 0}, dtwice{0, 0, 0}, weighted{0, 0, 0}, dweighted{0, 0, 0};
        double weight = 0, dweight = 0;
        for (int m = begin + 2; m < end; ++m) {
            const Vec3 b = vertex(m, q) - first, db = dvertex(m) - dfirst;
            const Vec3 t = cross(a, b), dt = cross(da, b) + cross(a, db);
            twice = twice + t;
            dtwice = dtwice + dt;
            const double projected = dot(t, dj), w = Kokkos::abs(projected), dw = sign_of(projected) * dot(dt, dj);
            weight += w;
            dweight += dw;
            weighted = weighted + (a + b) * w;
            dweighted = dweighted + (da + db) * w + (a + b) * dw;
            a = b;
            da = db;
        }
        dA = dtwice * (dot(twice, dj) < 0 ? -0.5 : 0.5);
        dxf = weight > 0 ? dfirst + (dweighted - weighted * (dweight / weight)) * (1.0 / (3.0 * weight)) : dfirst;
    }
    // The bar of the vertex at ring entry fan.begin + target (relative to the owner) from the bars of the face's area and
    // centroid: the transpose of tangent() for that vertex.
    KOKKOS_INLINE_FUNCTION Vec3 vertex_bar(const Fan& fan, int target, Vec3 Abar, Vec3 xfbar) const {
        const tet_mesh::GridPoint q(g.x, fan.i, g.box);
        const Vec3 tbar = Abar * (0.5 * fan.sign);  // the bar of each T_m through the area
        const bool weighted = fan.weight > 0;
        const Vec3 sbar = weighted ? xfbar * (1.0 / (3.0 * fan.weight)) : Vec3{0, 0, 0};  // of the weighted sum
        const Vec3 mean = weighted ? fan.weighted * (1.0 / fan.weight) : Vec3{0, 0, 0};
        Vec3 bar = target == 0 ? xfbar : Vec3{0, 0, 0};  // x_f = v_0 + ...
        const Vec3 first = vertex(fan.begin, q);
        Vec3 a = vertex(fan.begin + 1, q) - first;
        for (int m = 2; m < fan.end - fan.begin; ++m) {
            const Vec3 b = vertex(fan.begin + m, q) - first;
            if (target == 0 || target == m || target == m - 1) {
                const Vec3 t = cross(a, b);
                const double projected = dot(t, fan.dj), w = Kokkos::abs(projected);
                const double wbar = dot(sbar, a + b - mean);
                const Vec3 tb = tbar + fan.dj * (wbar * sign_of(projected));
                const Vec3 abar = cross(b, tb) + sbar * w,
                           bbar = cross(tb, a) + sbar * w;  // a = v_{m-1} - v_0, b = v_m - v_0
                if (target == m - 1) bar = bar + abar;
                if (target == m) bar = bar + bbar;
                if (target == 0) bar = bar - abar - bbar;
            }
            a = b;
        }
        return bar;
    }
};

// The forward face values a cell sum reads: 3D from the ring (FaceGeometry::face, the stored bits), 2D the stored rows.
template <class Memory>
struct RingFaces {
    tet_mesh::FaceGeometry<Memory> g;
    KOKKOS_INLINE_FUNCTION void operator()(int f, Vec3& A, Vec3& c) const { g.face(f, A, c); }
};
// The same from stored rows (2D).
template <class Rows>
struct StoredFaces {
    Rows area, centroid;
    KOKKOS_INLINE_FUNCTION void operator()(int f, Vec2& A, Vec2& c) const {
        A = tet_mesh::row<2>(area, f);
        c = tet_mesh::row<2>(centroid, f);
    }
};

// What the cell sums of the derivative read, D = 2 or 3: positions, faces and the cell lists of update_cells().
template <int D, class Memory, class X, class Faces>
struct Cells {
    using Ints = Kokkos::View<const int*, Memory>;
    X x;
    exact_predicates::BoxN<D> box;
    Faces faces;
    Ints owner, neighbour, start, list;
};

// The pyramid of cell i over face f, as tet_mesh's cell sums have it: the separation d to the other cell, the area,
// the face centroid relative to i, |A . d| and its sign.
template <int D>
struct Pyramid {
    int other;
    bool owned;
    VecN<D> d, A, xf;
    double p, s;
};
// Cell i's pyramid over face f.
template <int D, class C>
KOKKOS_INLINE_FUNCTION Pyramid<D> pyramid(const C& cells, int i, int f) {
    Pyramid<D> y;
    const int o = cells.owner(f);
    y.owned = o == i;
    y.other = o ^ cells.neighbour(f) ^ i;
    y.d = tet_mesh::snapped_difference<D>(tet_mesh::row<D>(cells.x, y.other) - tet_mesh::row<D>(cells.x, i), cells.box);
    cells.faces(f, y.A, y.xf);
    if (!y.owned) y.xf = y.xf + y.d;
    const double projected = dot(y.A, y.d);
    y.p = Kokkos::abs(projected);
    y.s = sign_of(projected);
    return y;
}

// The tangent's outputs: dV, dc (per cell), dA, dx_f (per face).
template <class DV, class DC, class DA, class DXF>
struct Changes {
    DV V;
    DC c;
    DA A;
    DXF xf;
};
// The adjoint's inputs: the bars of V, c (per cell), A, x_f (per face); an empty view counts as zero.
template <class VB, class CB, class AB, class XB>
struct Bars {
    VB V;
    CB c;
    AB A;
    XB xf;
};

// The tangent of the cell sums: dV and dc of every cell from dx and the faces' dA, dx_f (tet_mesh::CellLists::sum).
template <int D, class ExecSpace, class C, class DX, class Out>
void cell_tangent(const ExecSpace& exec, const C& cells, int n, const DX& dx, const Out& out) {
    constexpr double kappa = tet_mesh::detail::pyramid_centroid<D>, share = tet_mesh::detail::pyramid_volume<D>;
    const auto dA = out.A, dxf = out.xf;
    const auto dV = out.V;
    const auto dc = out.c;
    Kokkos::parallel_for(
        "geometry_derivative::cell_tangent", Kokkos::RangePolicy<ExecSpace>(exec, 0, n), KOKKOS_LAMBDA(int i) {
            double P = 0, dP = 0;
            VecN<D> M{}, dM{};
            const VecN<D> dxi = tet_mesh::row<D>(dx, i);
            for (int e = cells.start(i); e < cells.start(i + 1); ++e) {
                const int f = tet_mesh::list_face(cells.list(e));
                const Pyramid<D> y = pyramid<D>(cells, i, f);
                const VecN<D> ddx = tet_mesh::row<D>(dx, y.other) - dxi;
                const double dp = y.s * (dot(tet_mesh::row<D>(dA, f), y.d) + dot(y.A, ddx));
                VecN<D> dxfi = tet_mesh::row<D>(dxf, f);
                if (!y.owned) dxfi = dxfi + ddx;
                P += y.p;
                dP += dp;
                M = M + y.xf * y.p;
                dM = dM + dxfi * y.p + y.xf * dp;
            }
            dV(i) = dP * share;
            tet_mesh::store(dc, i, (dM - M * (dP / P)) * (kappa / P));
        });
}

// The transpose of the cell sums, first part: per cell the bars of P = sum p and M = sum p x_f (cellbar(i, 0) and
// cellbar(i, 1..D)) from the bars of V and c, with P and M summed as cell_tangent sums them.
template <int D, class ExecSpace, class C, class In, class Bar>
void cell_bars(const ExecSpace& exec, const C& cells, int n, const In& in, const Bar& cellbar) {
    constexpr double kappa = tet_mesh::detail::pyramid_centroid<D>, share = tet_mesh::detail::pyramid_volume<D>;
    const auto Vbar = in.V;
    const auto cbar = in.c;
    const bool with_volume = Vbar.extent(0) > 0;
    Kokkos::parallel_for(
        "geometry_derivative::cell_bars", Kokkos::RangePolicy<ExecSpace>(exec, 0, n), KOKKOS_LAMBDA(int i) {
            double P = 0;
            VecN<D> M{};
            for (int e = cells.start(i); e < cells.start(i + 1); ++e) {
                const Pyramid<D> y = pyramid<D>(cells, i, tet_mesh::list_face(cells.list(e)));
                P += y.p;
                M = M + y.xf * y.p;
            }
            const VecN<D> cb = row_or_zero<D>(cbar, i);
            // V = P share, c = kappa M / P
            const double Pbar = (with_volume ? Vbar(i) * share : 0.0) - kappa * dot(cb, M) / (P * P);
            const VecN<D> Mbar = cb * (kappa / P);
            cellbar(i, 0) = Pbar;
            cellbar(i, 1) = Mbar.x;
            cellbar(i, 2) = Mbar.y;
            if constexpr (D == 3) cellbar(i, 3) = Mbar.z;
        });
}

// A face's two pyramids under the bars of their cells: the totals of the area's and the centroid's bars (the caller's
// plus the cell sums'), and what lands on the owner and the neighbour directly (not through the vertices).
template <int D>
struct FaceBars {
    VecN<D> Abar, xfbar, owner, neighbour;
};
// The bars of face f with area A, centroid xf (relative to the owner), separation dj from owner o to neighbour j.
template <int D, class Bar, class In>
KOKKOS_INLINE_FUNCTION FaceBars<D> face_bars(const Bar& cellbar, const In& in, int f, int o, int j, VecN<D> A,
                                             VecN<D> xf, VecN<D> dj) {
    auto bar_of = [&](int i, double& P, VecN<D>& M) {
        P = cellbar(i, 0);
        M.x = cellbar(i, 1);
        M.y = cellbar(i, 2);
        if constexpr (D == 3) M.z = cellbar(i, 3);
    };
    double Po, Pj;
    VecN<D> Mo, Mj;
    bar_of(o, Po, Mo);
    bar_of(j, Pj, Mj);
    const VecN<D> dn = dj * -1.0;  // the neighbour's separation to the owner
    const double ao = dot(A, dj), an = dot(A, dn);
    const double so = sign_of(ao), sn = sign_of(an), po = Kokkos::abs(ao), pn = Kokkos::abs(an);
    const double pbar_o = Po + dot(Mo, xf), pbar_j = Pj + dot(Mj, xf + dn);
    FaceBars<D> b;
    b.Abar = row_or_zero<D>(in.A, f) + dj * (pbar_o * so) + dn * (pbar_j * sn);
    b.xfbar = row_or_zero<D>(in.xf, f) + Mo * po + Mj * pn;
    // p_o = |A . (x_j - x_o)|, p_j = |A . (x_o - x_j)|, x_f,j = x_f + x_o - x_j; x_f moves with the owner (- xfbar)
    const VecN<D> from_o = A * (pbar_o * so), from_j = A * (pbar_j * sn);
    b.neighbour = from_o - from_j - Mj * pn;
    b.owner = from_j - from_o + Mj * pn - b.xfbar;
    return b;
}

// 3D tangent: dct (absolute vertex changes per tetrahedron), then dA, dx_f per face, then dV, dc per cell.
template <class ExecSpace, class X, class Vertex, class Alive, class C, class DCT, class DX, class Out>
void tangent3(const ExecSpace& exec, const Faces3<typename ExecSpace::memory_space>& faces, const Vertex& vertex,
              const Alive& alive, int tets, const C& cells, int n, int face_count, const DCT& dct, const DX& dx,
              const Out& out) {
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const auto dA = out.A, dxf = out.xf;
    const X x = faces.g.x;
    const exact_predicates::Box box = faces.g.box;
    const auto centre = faces.g.centre;
    Kokkos::parallel_for(
        "geometry_derivative::vertex_tangent", Range(exec, 0, tets), KOKKOS_LAMBDA(int t) {
            if (!alive(t)) return;
            int v[4];
            const Centre<3> c = tet_centre(x, box, vertex, centre, t, v);
            const Vec3 d0 = tet_mesh::row(dx, v[0]);
            const Vec3 dd[3] = {tet_mesh::row(dx, v[1]) - d0, tet_mesh::row(dx, v[2]) - d0,
                                tet_mesh::row(dx, v[3]) - d0};
            tet_mesh::store(dct, t, d0 + c.tangent(dd));
        });
    Kokkos::parallel_for(
        "geometry_derivative::face_tangent", Range(exec, 0, face_count), KOKKOS_LAMBDA(int f) {
            Vec3 a, b;
            faces.tangent(f, dct, dx, a, b);
            tet_mesh::store(dA, f, a);
            tet_mesh::store(dxf, f, b);
        });
    cell_tangent<3>(exec, cells, n, dx, out);
}

// Cell lo's face to cell hi (lo < hi) whose ring holds tetrahedron t, and t's place in that ring; -1 if none. The list is
// sorted by the other cell (update_cells), so a bisection finds it.
template <class C, class G>
KOKKOS_INLINE_FUNCTION int face_of_edge(const C& cells, const G& g, int lo, int hi, int t, int& place) {
    int first = cells.start(lo), last = cells.start(lo + 1);
    auto other = [&](int e) {
        const int f = tet_mesh::list_face(cells.list(e));
        return cells.owner(f) ^ cells.neighbour(f) ^ lo;
    };
    while (first < last) {
        const int mid = (first + last) / 2;
        if (other(mid) < hi)
            first = mid + 1;
        else
            last = mid;
    }
    for (int e = first; e < cells.start(lo + 1) && other(e) == hi; ++e) {
        const int f = tet_mesh::list_face(cells.list(e));
        const int s = g.slot.data() ? g.slot(f) : f;
        for (int m = 0; m < g.ring_length(s); ++m)
            if (g.ring(g.ring_start(s) + m) == t) {
                place = m;
                return f;
            }
    }
    return -1;
}

// 3D adjoint: cell bars; per tetrahedron the bar of its vertex, gathered over its six faces; per cell x̄, gathered over
// its faces and the tetrahedra around it. No atomics: every sum is one thread's, in a fixed order.
template <class ExecSpace, class X, class Vertex, class Alive, class C, class Bar, class CTB, class In, class XBar>
void adjoint3(const ExecSpace& exec, const Faces3<typename ExecSpace::memory_space>& faces, const Vertex& vertex,
              const Alive& alive, int tets, const C& cells, int n, const Bar& cellbar, const CTB& ctbar, const In& in,
              const XBar& xbar) {
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const X x = faces.g.x;
    const exact_predicates::Box box = faces.g.box;
    const auto centre = faces.g.centre;
    const auto g = faces.g;
    cell_bars<3>(exec, cells, n, in, cellbar);
    Kokkos::parallel_for(
        "geometry_derivative::vertex_bars", Range(exec, 0, tets), KOKKOS_LAMBDA(int t) {
            if (!alive(t)) return;
            int v[4];
            for (int k = 0; k < 4; ++k) v[k] = vertex(t, k);
            tet_mesh::sort_corners(v);
            Vec3 bar{0, 0, 0};
            for (int a = 0; a < 3; ++a)
                for (int b = a + 1; b < 4; ++b) {
                    int place = 0;
                    const int f = face_of_edge(cells, g, v[a], v[b], t, place);
                    if (f < 0) continue;
                    const Fan fan = faces.walk(f);
                    const FaceBars<3> fb = face_bars<3>(cellbar, in, f, fan.i, fan.j, fan.area, fan.centroid, fan.dj);
                    bar = bar + faces.vertex_bar(fan, place, fb.Abar, fb.xfbar);
                }
            tet_mesh::store(ctbar, t, bar);
        });
    Kokkos::parallel_for(
        "geometry_derivative::cell_adjoint", Range(exec, 0, n), KOKKOS_LAMBDA(int i) {
            Vec3 bar{0, 0, 0};
            for (int e = cells.start(i); e < cells.start(i + 1); ++e) {
                const int f = tet_mesh::list_face(cells.list(e));
                const int o = cells.owner(f), j = cells.neighbour(f), other = o ^ j ^ i;
                Vec3 A, xf;
                cells.faces(f, A, xf);
                const Vec3 dj = tet_mesh::snapped_difference<3>(tet_mesh::row(x, j) - tet_mesh::row(x, o), box);
                const FaceBars<3> fb = face_bars<3>(cellbar, in, f, o, j, A, xf, dj);
                bar = bar + (o == i ? fb.owner : fb.neighbour);
                // The tetrahedra around edge i-other, each taken at its edge to the lowest of its other corners.
                const int s = g.slot.data() ? g.slot(f) : f;
                for (int m = g.ring_start(s); m < g.ring_start(s) + g.ring_length(s); ++m) {
                    const int t = g.ring(m);
                    int lowest = 0x7fffffff;
                    for (int k = 0; k < 4; ++k) {
                        const int w = vertex(t, k);
                        if (w != i && w < lowest) lowest = w;
                    }
                    if (lowest != other) continue;
                    int v[4];
                    const Centre<3> c = tet_centre(x, box, vertex, centre, t, v);
                    const int corner = v[0] == i ? 0 : v[1] == i ? 1 : v[2] == i ? 2 : 3;
                    bar = bar + c.corner_bar(tet_mesh::row(ctbar, t), corner);
                }
            }
            add_row(xbar, i, bar);
        });
}

// 2D tangent: dct per triangle, then dA, dx_f per face (one thread per triangle, as tri_mesh::faces), then the cells.
template <class ExecSpace, class Point, class Mesh2, class C, class DCT, class DX, class Out>
void tangent2(const ExecSpace& exec, const Point& point, const exact_predicates::BoxN<2>& box, const Mesh2& tri,
              const C& cells, int n, const DCT& dct, const DX& dx, const Out& out) {
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const auto dA = out.A, dxf = out.xf;
    const auto corners = tri.corners, links = tri.links;
    Kokkos::parallel_for(
        "geometry_derivative::vertex_tangent2", Range(exec, 0, tri.count()), KOKKOS_LAMBDA(int t) {
            const Centre<2> c = triangle_centre(point, box, corners, t);
            const Vec2 d0 = tet_mesh::row<2>(dx, corners(t, 0));
            const Vec2 dd[2] = {tet_mesh::row<2>(dx, corners(t, 1)) - d0, tet_mesh::row<2>(dx, corners(t, 2)) - d0};
            tet_mesh::store(dct, t, d0 + c.tangent(dd));
        });
    Kokkos::parallel_for(
        "geometry_derivative::face_tangent2", Range(exec, 0, tri.count()), KOKKOS_LAMBDA(int t) {
            const Vec2 ct = tet_mesh::row<2>(dct, t);
            for (int k = 0; k < 3; ++k) {
                if (corners(t, k) >= corners(t, 3 + k)) continue;
                const int a = corners(t, (k + 1) % 3), b = corners(t, (k + 2) % 3), f = links(t, 3 + k);
                const double sign = a < b ? 1.0 : -1.0;
                const Vec2 de = ct - tet_mesh::row<2>(dct, links(t, k));
                tet_mesh::store(dA, f, Vec2{sign * de.y, -sign * de.x});
                tet_mesh::store(dxf, f, ct - tet_mesh::row<2>(dx, a < b ? a : b) - de * 0.5);
            }
        });
    cell_tangent<2>(exec, cells, n, dx, out);
}

// The bars of a 2D face (face_bars) from its stored rows, seen from either triangle. Copy into kernels.
template <class C, class Bar, class In>
struct FaceBars2 {
    C cells;
    Bar cellbar;
    In in;
    KOKKOS_INLINE_FUNCTION FaceBars<2> operator()(int f) const {
        const int o = cells.owner(f), j = cells.neighbour(f);
        Vec2 A, xf;
        cells.faces(f, A, xf);
        const Vec2 dj =
            tet_mesh::snapped_difference<2>(tet_mesh::row<2>(cells.x, j) - tet_mesh::row<2>(cells.x, o), cells.box);
        return face_bars<2>(cellbar, in, f, o, j, A, xf, dj);
    }
};

// 2D adjoint: cell bars; the owning triangle of each face (3t + k, `owning`); per triangle the bar of its vertex over its
// three faces; per cell x̄ over its faces and the triangles around it. No atomics.
template <class ExecSpace, class Point, class Mesh2, class C, class Bar, class CTB, class Owning, class In, class XBar>
void adjoint2(const ExecSpace& exec, const Point& point, const exact_predicates::BoxN<2>& box, const Mesh2& tri,
              const C& cells, int n, const Bar& cellbar, const CTB& ctbar, const Owning& owning, const In& in,
              const XBar& xbar) {
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const auto corners = tri.corners, links = tri.links;
    cell_bars<2>(exec, cells, n, in, cellbar);
    const FaceBars2<C, Bar, In> bars{cells, cellbar, in};
    Kokkos::parallel_for(
        "geometry_derivative::vertex_bars2", Range(exec, 0, tri.count()), KOKKOS_LAMBDA(int t) {
            Vec2 bar{0, 0};
            for (int k = 0; k < 3; ++k) {
                const bool owns = corners(t, k) < corners(t, 3 + k);
                if (owns) owning(links(t, 3 + k)) = 3 * t + k;
                // A = sign (e_y, -e_x) and x_f = c_t - x_o - e / 2 with e = c_t - c_u, t the owning triangle
                const int a = corners(t, owns ? (k + 1) % 3 : (k + 2) % 3),
                          b = corners(t, owns ? (k + 2) % 3 : (k + 1) % 3);
                const double sign = a < b ? 1.0 : -1.0;
                const FaceBars<2> fb = bars(links(t, 3 + k));
                const Vec2 ebar = Vec2{-sign * fb.Abar.y, sign * fb.Abar.x} - fb.xfbar * 0.5;
                bar = owns ? bar + fb.xfbar + ebar : bar - ebar;
            }
            tet_mesh::store(ctbar, t, bar);
        });
    Kokkos::parallel_for(
        "geometry_derivative::cell_adjoint2", Range(exec, 0, n), KOKKOS_LAMBDA(int i) {
            Vec2 bar{0, 0};
            for (int e = cells.start(i); e < cells.start(i + 1); ++e) {
                const int f = tet_mesh::list_face(cells.list(e));
                const int other = cells.owner(f) ^ cells.neighbour(f) ^ i;
                const FaceBars<2> fb = bars(f);
                bar = bar + (cells.owner(f) == i ? fb.owner : fb.neighbour);
                // The face's two triangles, each taken at its edge from i to the next corner counter-clockwise.
                const int t0 = owning(f) / 3, k0 = owning(f) % 3;
                const int both[2] = {t0, links(t0, k0)};
                for (int t : both) {
                    const int corner = corners(t, 0) == i ? 0 : corners(t, 1) == i ? 1 : 2;
                    if (corners(t, (corner + 1) % 3) != other) continue;
                    bar = bar + triangle_centre(point, box, corners, t).corner_bar(tet_mesh::row<2>(ctbar, t), corner);
                }
            }
            add_row(xbar, i, bar);
        });
}

}  // namespace geometry_derivative

// The geometry derivatives of a Voronoi mesh (Mesh<2> or Mesh<3>) at its current topology: construct it on the mesh
// after update() and update_cells(); valid until the next update(). Rows are in the mesh's cell order (after a re-sort,
// cell i is the caller's particle permutation()(i)) and face order (f < faces()). Views are device views of doubles,
// one row per cell or face: dx, c, x_f, A as (rows x D), V as (rows).
//  - tangent(dx, dV, dc, dA, dxf): writes dV, dc (per cell) and dA, dx_f (per face) for the position changes dx;
//  - adjoint(Vbar, cbar, Abar, xfbar, xbar): adds to xbar (per cell) the transpose applied to the bars of V, c (per cell),
//    A, x_f (per face); an empty bar view counts as zero.
// The quantities are the mesh's: c relative to the generator, A from owner to neighbour, x_f relative to the owner.
// Deterministic: the adjoint gathers per vertex and per cell in a fixed order, with no atomics. The scratch it keeps,
// scratch_bytes(), is reused between calls (3D: 4 doubles per cell and 3 per tetrahedron; 2D: 3 per cell, 4 per cell for
// the triangles and an int per face).
template <class M>
class GeometryDerivative;

// The one for Mesh<D, ExecSpace, FaceReal>, D = 2 or 3.
template <int D, class ExecSpace, class FaceReal>
class GeometryDerivative<Mesh<D, ExecSpace, FaceReal>> {
public:
    static_assert(D == 2 || D == 3, "voronoi::GeometryDerivative: 2D or 3D meshes");
    using MeshType = Mesh<D, ExecSpace, FaceReal>;
    using Memory = typename ExecSpace::memory_space;

    explicit GeometryDerivative(const MeshType& mesh) : mesh_(mesh) {}

    // dx (per cell) -> dV, dc (per cell), dA, dx_f (per face).
    template <class DX, class DV, class DC, class DA, class DXF>
    void tangent(const DX& dx, const DV& dV, const DC& dc, const DA& dA, const DXF& dxf) const {
        check();
        const ExecSpace exec;
        if constexpr (D == 3) {
            const int tets = mesh_.repair_state().count;
            const auto dct = rows<3>(0, tets);
            geometry_derivative::tangent3<ExecSpace, typename MeshType::Positions>(
                exec, faces3(), mesh_.tetrahedra().vertex, mesh_.repair_state().alive, tets, cells(), mesh_.particles(),
                mesh_.faces(), dct, dx, geometry_derivative::Changes<DV, DC, DA, DXF>{dV, dc, dA, dxf});
        } else {
            const auto dct = rows<2>(0, mesh_.triangles().count());
            geometry_derivative::tangent2<ExecSpace>(exec, mesh_.points(), mesh_.sides(), mesh_.triangles(), cells(),
                                                     mesh_.particles(), dct, dx,
                                                     geometry_derivative::Changes<DV, DC, DA, DXF>{dV, dc, dA, dxf});
        }
    }

    // The transpose: from the bars of V, c (per cell) and A, x_f (per face), adds to xbar (per cell).
    template <class VB, class CB, class AB, class XB, class XBar>
    void adjoint(const VB& Vbar, const CB& cbar, const AB& Abar, const XB& xfbar, const XBar& xbar) const {
        check();
        const geometry_derivative::Bars<VB, CB, AB, XB> bars{Vbar, cbar, Abar, xfbar};
        const ExecSpace exec;
        const int n = mesh_.particles();
        const std::size_t after = static_cast<std::size_t>(n) * (D + 1);  // the cell bars, then the vertex bars
        if constexpr (D == 3) {
            const int tets = mesh_.repair_state().count;
            grow(after + static_cast<std::size_t>(tets) * 3);
            geometry_derivative::adjoint3<ExecSpace, typename MeshType::Positions>(
                exec, faces3(), mesh_.tetrahedra().vertex, mesh_.repair_state().alive, tets, cells(), n, rows<4>(0, n),
                rows<3>(after, tets), bars, xbar);
        } else {
            const int triangles = mesh_.triangles().count();
            const std::size_t ints = after + static_cast<std::size_t>(triangles) * 2;  // then the owning triangles
            const Kokkos::View<int*, Memory, Kokkos::MemoryTraits<Kokkos::Unmanaged>> owning(
                reinterpret_cast<int*>(grow(ints + (mesh_.faces() + 1) / 2) + ints), mesh_.faces());
            geometry_derivative::adjoint2<ExecSpace>(exec, mesh_.points(), mesh_.sides(), mesh_.triangles(), cells(), n,
                                                     rows<3>(0, n), rows<2>(after, triangles), owning, bars, xbar);
        }
    }

    // Bytes of scratch held now (grown by the calls, kept until this object goes).
    std::size_t scratch_bytes() const { return scratch_.extent(0) * sizeof(double); }

private:
    template <int Columns>
    using Rows = Kokkos::View<double* [Columns], Kokkos::LayoutRight, Memory, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    void check() const {
        if (!mesh_.cells_current())
            throw std::logic_error(
                "voronoi::GeometryDerivative: no current cells (call update_cells() after update())");
    }
    // Scratch of at least `doubles`, its start.
    double* grow(std::size_t doubles) const {
        if (scratch_.extent(0) < doubles)
            scratch_ = Kokkos::View<double*, Memory>(
                Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("geometry_derivative")), doubles);
        return scratch_.data();
    }
    // `count` rows of `Columns` doubles at double `at` of the scratch, grown to hold them (a call that takes several grows
    // to its total first: a growth frees the rows taken before).
    template <int Columns>
    Rows<Columns> rows(std::size_t at, int count) const {
        return Rows<Columns>(grow(at + static_cast<std::size_t>(count) * Columns) + at, count);
    }
    geometry_derivative::Faces3<Memory> faces3() const { return {mesh_.face_geometry()}; }
    // The cell sums' inputs: 3D faces from their rings (the stored bits, in every Layout), 2D the stored rows.
    auto cells() const {
        if constexpr (D == 3) {
            using Faces = geometry_derivative::RingFaces<Memory>;
            return geometry_derivative::Cells<3, Memory, typename MeshType::Positions, Faces>{
                mesh_.positions(), mesh_.sides(),     Faces{mesh_.face_geometry()},
                mesh_.owner(),     mesh_.neighbour(), mesh_.cell_face_start(),
                mesh_.cell_face()};
        } else {
            using Faces = geometry_derivative::StoredFaces<typename MeshType::Rows>;
            return geometry_derivative::Cells<2, Memory, typename MeshType::Positions, Faces>{
                mesh_.positions(), mesh_.sides(),     Faces{mesh_.area(), mesh_.face_centroid()},
                mesh_.owner(),     mesh_.neighbour(), mesh_.cell_face_start(),
                mesh_.cell_face()};
        }
    }

    const MeshType& mesh_;
    mutable Kokkos::View<double*, Memory> scratch_;
};

}  // namespace voronoi
