// voronoi::GeometryDerivative (#63), 2D and 3D: the tangent against central differences on the quantisation grid (whole
// grid steps), the dot test, the identities (sum of dV, closure, translation), the same bits twice; square, cubic and
// non-cubic periodic boxes. Exercises voronoi::GeometryDerivative.

#include "check.hpp"
#include "geometry_derivative.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Memory = Exec::memory_space;
template <int D>
using Rows = Kokkos::View<double* [D], Kokkos::LayoutRight, Memory>;
using Scalars = Kokkos::View<double*, Memory>;
template <int D>
using Vec = std::array<double, D>;
using Grid = std::vector<std::array<long long, 3>>;  // grid coordinates per point (the third unused in 2D)

// One test case: dimension, box sides, points, what the mesh went through before.
struct Case {
    std::string kind;  // jittered or uniform
    std::array<double, 3> side;
    std::array<int, 3> lattice;  // points per axis (jittered), product = n
    bool moved = false;          // the mesh is repaired into place from other positions, not built there
    bool layout = false;         // 3D: faces by owner, owner bit, faces on read, float centroids
};

template <int D>
exact_predicates::BoxN<D> box_of(const Case& c) {
    if constexpr (D == 3)
        return exact_predicates::BoxN<3>(c.side[0], c.side[1], c.side[2]);
    else
        return exact_predicates::BoxN<2>(c.side[0], c.side[1]);
}

// Points of the case on the box's grid.
template <int D>
Grid make_grid(const Case& c, const exact_predicates::BoxN<D>& box, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    int n = 1;
    for (int a = 0; a < D; ++a) n *= c.lattice[a];
    Grid g(n, {0, 0, 0});
    for (int i = 0; i < n; ++i) {
        int rest = i;
        for (int a = 0; a < D; ++a) {
            const int at = rest % c.lattice[a];
            rest /= c.lattice[a];
            const double period = static_cast<double>(box.period[a]);
            const double u = c.kind == "uniform" ? unit(rng) : (at + 0.5 + 0.3 * (unit(rng) - 0.5)) / c.lattice[a];
            g[i][a] = static_cast<long long>(std::floor(u * period)) % box.period[a];
        }
    }
    return g;
}

// Grid points g + k v (wrapped) as positions in box units.
template <int D>
Rows<D> positions(const Grid& g, const Grid& v, long long k, const exact_predicates::BoxN<D>& box) {
    Rows<D> x("x", g.size());
    auto h = Kokkos::create_mirror_view(x);
    for (std::size_t i = 0; i < g.size(); ++i)
        for (int a = 0; a < D; ++a) {
            const long long p = box.period[a];
            const long long at = ((g[i][a] + k * (v.empty() ? 0 : v[i][a])) % p + p) % p;
            h(i, a) = static_cast<double>(at) * box.step;
        }
    Kokkos::deep_copy(x, h);
    return x;
}

// The geometry of a mesh on the host, faces by (owner, neighbour).
template <int D>
struct Snapshot {
    std::vector<double> V;
    std::vector<Vec<D>> c;
    std::vector<std::pair<int, int>> pair;                          // of face f
    std::map<std::pair<int, int>, std::pair<Vec<D>, Vec<D>>> face;  // area, centroid
};

template <int D, class M>
Snapshot<D> snapshot(const M& mesh) {
    Snapshot<D> s;
    const int n = mesh.particles(), faces = mesh.faces();
    const auto V = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.volume());
    const auto c = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.centroid());
    const auto o = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.owner());
    const auto j = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.neighbour());
    const auto A = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.area());
    const auto x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.face_centroid());
    for (int i = 0; i < n; ++i) {
        s.V.push_back(V(i));
        Vec<D> ci;
        for (int a = 0; a < D; ++a) ci[a] = c(i, a);
        s.c.push_back(ci);
    }
    for (int f = 0; f < faces; ++f) {
        Vec<D> a, b;
        for (int d = 0; d < D; ++d) {
            a[d] = A(f, d);
            b[d] = x.extent(0) > 0 ? x(f, d) : 0.0;
        }
        s.pair.emplace_back(o(f), j(f));
        s.face[{o(f), j(f)}] = {a, b};
    }
    return s;
}

// The derivative's outputs on the host.
template <int D>
struct Tangent {
    std::vector<double> V;
    std::vector<Vec<D>> c, A, xf;  // c per cell; A, xf per face of the base mesh
};

template <int D, class G>
Tangent<D> tangent(const G& derivative, const Rows<D>& dx, int n, int faces) {
    Scalars dV("dV", n);
    Rows<D> dc("dc", n), dA("dA", faces), dxf("dxf", faces);
    derivative.tangent(dx, dV, dc, dA, dxf);
    const auto hV = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dV);
    const auto hc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dc);
    const auto hA = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dA);
    const auto hx = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dxf);
    Tangent<D> t;
    t.V.assign(hV.data(), hV.data() + n);
    t.c.resize(n);
    t.A.resize(faces);
    t.xf.resize(faces);
    for (int i = 0; i < n; ++i)
        for (int a = 0; a < D; ++a) t.c[i][a] = hc(i, a);
    for (int f = 0; f < faces; ++f)
        for (int a = 0; a < D; ++a) {
            t.A[f][a] = hA(f, a);
            t.xf[f][a] = hx(f, a);
        }
    return t;
}

// Largest |a - b| over largest |b|, and largest |b|.
struct Error {
    double worst = 0, scale = 0;
    void add(double a, double b) {
        worst = std::max(worst, std::abs(a - b));
        scale = std::max(scale, std::abs(b));
    }
    double relative() const { return scale > 0 ? worst / scale : worst; }
};

// The mesh of the case at positions x (and, if moved, repaired there from `from`).
template <int D>
voronoi::Mesh<D, Exec> make_mesh(const Case& c, const exact_predicates::BoxN<D>& box, const Rows<D>& x,
                                 const Rows<D>& from = {}) {
    voronoi::Mesh<D, Exec> mesh(Exec{}, box);
    if constexpr (D == 3) {
        voronoi::Layout layout;
        layout.every = 0;  // cell ids stay the caller's
        if (c.layout) {
            layout.owner_faces = layout.owner_bit = layout.faces_on_read = layout.float_centroids = true;
        }
        mesh.set_layout(layout);
    }
    if (from.extent(0) > 0) mesh.update(from);
    mesh.update(x);
    mesh.update_cells();
    return mesh;
}

// Random rows (normal), scaled.
template <int D>
Rows<D> random_rows(int count, double scale, std::mt19937_64& rng) {
    std::normal_distribution<double> normal(0.0, 1.0);
    Rows<D> v("random", count);
    auto h = Kokkos::create_mirror_view(v);
    for (int i = 0; i < count; ++i)
        for (int a = 0; a < D; ++a) h(i, a) = scale * normal(rng);
    Kokkos::deep_copy(v, h);
    return v;
}
Scalars random_scalars(int count, double scale, std::mt19937_64& rng) {
    std::normal_distribution<double> normal(0.0, 1.0);
    Scalars v("random", count);
    auto h = Kokkos::create_mirror_view(v);
    for (int i = 0; i < count; ++i) h(i) = scale * normal(rng);
    Kokkos::deep_copy(v, h);
    return v;
}
// The sum of the products of two views' entries (host).
template <class A, class B>
double inner(const A& a, const B& b) {
    const auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a);
    const auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);
    double sum = 0;
    const std::size_t count = a.size();
    for (std::size_t k = 0; k < count; ++k) sum += ha.data()[k] * hb.data()[k];
    return sum;
}
// Whether two views hold the same bits.
template <class A>
bool same_bits(const A& a, const A& b) {
    const auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a);
    const auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);
    return a.size() == b.size() && std::memcmp(ha.data(), hb.data(), a.size() * sizeof(double)) == 0;
}

// Runs every check of one case.
template <int D>
void run(const Case& c, Check& check, unsigned seed) {
    std::mt19937_64 rng(seed);
    const auto box = box_of<D>(c);
    const Grid g = make_grid<D>(c, box, rng);
    const int n = static_cast<int>(g.size());
    double spacing = 1;
    for (int a = 0; a < D; ++a) spacing *= c.side[a];
    spacing = std::pow(spacing / n, 1.0 / D);
    std::string name = std::to_string(D) + "D " + c.kind + " n=" + std::to_string(n) + " box";
    for (int a = 0; a < D; ++a) name += (a ? "x" : "=") + sci(c.side[a]);
    if (c.moved) name += " repaired";
    if (c.layout) name += " owner-faces/owner-bit/faces-on-read/float-centroids";
    std::cout << "== " << name << "\n";

    Grid offset;  // the start of a repaired mesh: up to 0.4 spacings away
    if (c.moved) {
        std::uniform_real_distribution<double> unit(-0.4, 0.4);
        offset.assign(n, {0, 0, 0});
        for (auto& o : offset)
            for (int a = 0; a < D; ++a) o[a] = static_cast<long long>(unit(rng) * spacing / box.step);
    }
    const Rows<D> x = positions<D>(g, {}, 0, box);
    const auto mesh = make_mesh<D>(c, box, x, c.moved ? positions<D>(g, offset, 1, box) : Rows<D>());
    const int faces = mesh.faces();
    const voronoi::GeometryDerivative<voronoi::Mesh<D, Exec>> derivative(mesh);
    const Snapshot<D> base = snapshot<D>(make_mesh<D>(Case{c.kind, c.side, c.lattice}, box, x));

    // Direction: up to 8 grid steps per point and axis; dx in box units.
    Grid v(n, {0, 0, 0});
    std::uniform_int_distribution<int> steps(-8, 8);
    for (auto& w : v)
        for (int a = 0; a < D; ++a) w[a] = steps(rng);
    Rows<D> dx("dx", n);
    {
        auto h = Kokkos::create_mirror_view(dx);
        for (int i = 0; i < n; ++i)
            for (int a = 0; a < D; ++a) h(i, a) = static_cast<double>(v[i][a]) * box.step;
        Kokkos::deep_copy(dx, h);
    }
    const Tangent<D> t = tangent<D>(derivative, dx, n, faces);
    const auto owner = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.owner());
    const auto neighbour = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.neighbour());

    // Central differences at k = 64, 16, 4 grid steps: the error falls as k² (truncation) down to the rounding. A few
    // faces in thousands flip within a thousand grid steps (near-cospherical corners), so the cells a flip touches in
    // the mesh at +k or -k, and their neighbours, and the faces of the cells it touches, are left out (counted).
    std::vector<std::vector<int>> adjacent(n);
    for (int f = 0; f < faces; ++f) {
        adjacent[owner(f)].push_back(neighbour(f));
        adjacent[neighbour(f)].push_back(owner(f));
    }
    struct Differences {  // central differences at one k, and the cells and faces away from flips
        std::vector<char> cell, face;
        std::vector<double> V;
        std::vector<Vec<D>> c, A, xf;
    } coarse, fine;  // k = 16 and 4
    for (long long k : {64LL, 16LL, 4LL}) {
        const Snapshot<D> plus =
            snapshot<D>(make_mesh<D>(Case{c.kind, c.side, c.lattice}, box, positions<D>(g, v, k, box)));
        const Snapshot<D> minus =
            snapshot<D>(make_mesh<D>(Case{c.kind, c.side, c.lattice}, box, positions<D>(g, v, -k, box)));
        std::vector<char> flipped(n, 0), touched(n, 0);
        auto mark = [&](const Snapshot<D>& a, const Snapshot<D>& b) {  // the pairs of a that b lacks
            for (const auto& entry : a.face)
                if (!b.face.count(entry.first)) flipped[entry.first.first] = flipped[entry.first.second] = 1;
        };
        mark(base, plus);
        mark(plus, base);
        mark(base, minus);
        mark(minus, base);
        for (int i = 0; i < n; ++i)  // a 3D flip's corners are its new edge's ends or their neighbours
            if (flipped[i]) {
                touched[i] = 1;
                for (int j : adjacent[i]) touched[j] = 1;
            }
        std::vector<char> clean(n, 1);
        int left_out = 0;
        for (int i = 0; i < n; ++i) {
            if (touched[i]) clean[i] = 0;
            for (int j : adjacent[i]) clean[i] = clean[i] && !touched[j];
            left_out += !clean[i];
        }
        Differences& d = k == 16 ? coarse : fine;
        d.cell = clean;
        d.face.assign(faces, 1);
        d.V.assign(n, 0);
        d.c.assign(n, Vec<D>{});
        d.A.assign(faces, Vec<D>{});
        d.xf.assign(faces, Vec<D>{});
        Error eV, ec, eA, ex;
        for (int i = 0; i < n; ++i) {
            d.V[i] = (plus.V[i] - minus.V[i]) / (2.0 * k);
            for (int a = 0; a < D; ++a) d.c[i][a] = (plus.c[i][a] - minus.c[i][a]) / (2.0 * k);
            if (!clean[i]) continue;
            eV.add(t.V[i], d.V[i]);
            for (int a = 0; a < D; ++a) ec.add(t.c[i][a], d.c[i][a]);
        }
        for (int f = 0; f < faces; ++f) {
            d.face[f] = !touched[owner(f)] && !touched[neighbour(f)];
            if (!d.face[f]) continue;
            const std::pair<int, int> p{owner(f), neighbour(f)};
            const auto& [Ap, xp] = plus.face.at(p);
            const auto& [Am, xm] = minus.face.at(p);
            for (int a = 0; a < D; ++a) {
                d.A[f][a] = (Ap[a] - Am[a]) / (2.0 * k);
                d.xf[f][a] = (xp[a] - xm[a]) / (2.0 * k);
                eA.add(t.A[f][a], d.A[f][a]);
                ex.add(t.xf[f][a], d.xf[f][a]);
            }
        }
        std::cout << "  k=" << k << " grid steps (at most " << 8 * k << " per axis; " << left_out
                  << " cells near a flip left out): dV " << sci(eV.relative()) << ", dc " << sci(ec.relative())
                  << ", dA " << sci(eA.relative()) << ", dx_f " << sci(ex.relative()) << " of the largest difference\n";
        if (k <= 16)
            check.expect(left_out < n / 20, name + ": k=" + std::to_string(k) + ": under 5 % of cells near a flip");
    }
    // Richardson's extrapolation of k = 16 and 4, (16 D_4 - D_16) / 15, cancels the k² term of the truncation (large on
    // the smallest faces of near-cospherical corners): what is left is the k^4 term and the rounding.
    {
        Error eV, ec, eA, ex;
        auto extrapolated = [](double c16, double c4) { return (16 * c4 - c16) / 15; };
        for (int i = 0; i < n; ++i) {
            if (!coarse.cell[i] || !fine.cell[i]) continue;
            eV.add(t.V[i], extrapolated(coarse.V[i], fine.V[i]));
            for (int a = 0; a < D; ++a) ec.add(t.c[i][a], extrapolated(coarse.c[i][a], fine.c[i][a]));
        }
        for (int f = 0; f < faces; ++f) {
            if (!coarse.face[f] || !fine.face[f]) continue;
            for (int a = 0; a < D; ++a) {
                eA.add(t.A[f][a], extrapolated(coarse.A[f][a], fine.A[f][a]));
                ex.add(t.xf[f][a], extrapolated(coarse.xf[f][a], fine.xf[f][a]));
            }
        }
        const double worst = std::max({eV.relative(), ec.relative(), eA.relative(), ex.relative()});
        std::cout << "  extrapolated from k=16 and 4: dV " << sci(eV.relative()) << ", dc " << sci(ec.relative())
                  << ", dA " << sci(eA.relative()) << ", dx_f " << sci(ex.relative()) << "\n";
        check.expect(worst < 1e-8, name + ": tangent against the extrapolated central differences: " + sci(worst) +
                                       " of the largest (gate 1e-8)");
    }

    // Dot test: <w, J v> = <J^T w, v> for random v and w (bars scaled so each quantity counts about alike), each bar
    // alone and all together.
    {
        const Rows<D> u = random_rows<D>(n, spacing, rng);
        Scalars dV("dV", n);
        Rows<D> dc("dc", n), dA("dA", faces), dxf("dxf", faces);
        derivative.tangent(u, dV, dc, dA, dxf);
        const Scalars Vbar = random_scalars(n, std::pow(spacing, 1 - D), rng);
        const Rows<D> cbar = random_rows<D>(n, 1.0, rng), Abar = random_rows<D>(faces, std::pow(spacing, 2 - D), rng),
                      xfbar = random_rows<D>(faces, 1.0, rng);
        const Scalars noV;
        const Rows<D> none;
        const char* label[5] = {"V", "c", "A", "x_f", "all"};
        const double parts[4] = {inner(Vbar, dV), inner(cbar, dc), inner(Abar, dA), inner(xfbar, dxf)};
        double worst = 0;
        for (int which = 0; which < 5; ++which) {
            Rows<D> xbar("xbar", n);
            const bool all = which == 4;
            derivative.adjoint(all || which == 0 ? Vbar : noV, all || which == 1 ? cbar : none,
                               all || which == 2 ? Abar : none, all || which == 3 ? xfbar : none, xbar);
            const double left = all ? parts[0] + parts[1] + parts[2] + parts[3] : parts[which], right = inner(xbar, u);
            const double e = std::abs(left - right) / std::max(std::abs(left), std::abs(right));
            std::cout << "  dot test " << label[which] << ": " << left << " vs " << right << ", " << sci(e) << "\n";
            worst = std::max(worst, e);
        }
        check.expect(worst < 1e-12, name + ": dot test " + sci(worst) + " (gate 1e-12)");

        // The same bits on a second call.
        Rows<D> once("once", n), twice("twice", n);
        derivative.adjoint(Vbar, cbar, Abar, xfbar, once);
        derivative.adjoint(Vbar, cbar, Abar, xfbar, twice);
        Scalars dV2("dV2", n);
        Rows<D> dc2("dc2", n), dA2("dA2", faces), dxf2("dxf2", faces);
        derivative.tangent(u, dV2, dc2, dA2, dxf2);
        check.expect(same_bits(once, twice) && same_bits(dV, dV2) && same_bits(dc, dc2) && same_bits(dA, dA2) &&
                         same_bits(dxf, dxf2),
                     name + ": tangent and adjoint give the same bits on a second call");

        // Identities: the volume changes sum to zero; each cell's area changes close (signed); a translation changes nothing.
        const Tangent<D> r = tangent<D>(derivative, u, n, faces);
        double sum = 0, magnitude = 0, largest = 0, open = 0;
        for (int i = 0; i < n; ++i) {
            sum += r.V[i];
            magnitude += std::abs(r.V[i]);
        }
        std::vector<Vec<D>> closure(n, Vec<D>{});
        for (int f = 0; f < faces; ++f)
            for (int a = 0; a < D; ++a) {
                closure[owner(f)][a] += r.A[f][a];
                closure[neighbour(f)][a] -= r.A[f][a];
                largest = std::max(largest, std::abs(r.A[f][a]));
            }
        for (const auto& w : closure)
            for (int a = 0; a < D; ++a) open = std::max(open, std::abs(w[a]));
        check.expect(std::abs(sum) < 1e-13 * magnitude,
                     name + ": the volume changes sum to zero: " + sci(std::abs(sum) / magnitude) + " of sum |dV|");
        check.expect(open < 1e-12 * largest,
                     name + ": every cell's area changes close: " + sci(open / largest) + " of the largest dA");
        Rows<D> shift("shift", n);
        {
            auto h = Kokkos::create_mirror_view(shift);
            for (int i = 0; i < n; ++i)
                for (int a = 0; a < D; ++a) h(i, a) = spacing * (0.3 + 0.2 * a);
            Kokkos::deep_copy(shift, h);
        }
        const Tangent<D> s = tangent<D>(derivative, shift, n, faces);
        // Per quantity the largest change under the translation over the largest under u.
        double moved = 0;
        auto compare = [&](const auto& under_shift, const auto& under_u) {
            double a = 0, b = 0;
            for (std::size_t k = 0; k < under_u.size(); ++k) {
                a = std::max(a, std::abs(under_shift[k]));
                b = std::max(b, std::abs(under_u[k]));
            }
            moved = std::max(moved, a / b);
        };
        auto flat = [](const std::vector<Vec<D>>& rows) {
            std::vector<double> out;
            for (const auto& w : rows) out.insert(out.end(), w.begin(), w.end());
            return out;
        };
        compare(s.V, r.V);
        compare(flat(s.c), flat(r.c));
        compare(flat(s.A), flat(r.A));
        compare(flat(s.xf), flat(r.xf));
        check.expect(moved < 1e-12, name + ": a translation changes nothing: " + sci(moved) +
                                        " of a random direction's changes of the same size");
    }
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    Check check;
    unsigned seed = 63;
    // 2D: 64 x 64 points in a unit square and a square of side 3, 64 x 128 in a 1 x 2 box; uniform; a repaired mesh.
    run<2>(Case{"jittered", {1, 1, 1}, {64, 64, 1}}, check, ++seed);
    run<2>(Case{"jittered", {3, 3, 1}, {64, 64, 1}}, check, ++seed);
    run<2>(Case{"jittered", {1, 2, 1}, {64, 128, 1}}, check, ++seed);
    run<2>(Case{"uniform", {1, 1, 1}, {64, 64, 1}}, check, ++seed);
    run<2>(Case{"jittered", {1, 1, 1}, {64, 64, 1}, true}, check, ++seed);
    // 3D: 16³ in a unit cube and a cube of side 3, 16 x 16 x 32 in a 1 x 1 x 2 box; uniform; a repaired mesh; the
    // other face layouts.
    run<3>(Case{"jittered", {1, 1, 1}, {16, 16, 16}}, check, ++seed);
    run<3>(Case{"jittered", {3, 3, 3}, {16, 16, 16}}, check, ++seed);
    run<3>(Case{"jittered", {1, 1, 2}, {16, 16, 32}}, check, ++seed);
    run<3>(Case{"uniform", {1, 1, 1}, {16, 16, 16}}, check, ++seed);
    run<3>(Case{"jittered", {1, 1, 1}, {16, 16, 16}, true}, check, ++seed);
    run<3>(Case{"jittered", {1, 1, 1}, {16, 16, 16}, true, true}, check, ++seed);
    std::cout << (check.failed() ? "FAILED: " : "passed: ") << check.failed() << " failures\n";
    return check.failed() != 0;
}
