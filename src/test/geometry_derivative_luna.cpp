// Optional (CMake -DVORONOI_LUNA_DIR=<luna-cpp checkout>): voronoi::GeometryDerivative behind luna-cpp's interface
// hydro::GeometryDerivative<2, Memory> against its test-only reference::Geometry2D (src/test/reference_geometry_2d.hpp):
// tangent and adjoint on random directions and bars, jittered and uniform 64 x 64 meshes on the grid.

// luna-cpp's interface (src/hydro/mesh_derivative.hpp, which includes this library's header) by path, then its reference.
#include LUNA_GEOMETRY_DERIVATIVE
#include LUNA_REFERENCE_2D

#include "check.hpp"
#include "geometry_derivative.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <string>

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Memory = Exec::memory_space;
using Mesh = voronoi::Mesh<2, Exec>;
using Interface = hydro::GeometryDerivative<2, Memory>;

// The library's derivative as luna-cpp's interface: a forwarding shim, nothing converted.
class Library : public Interface {
public:
    explicit Library(const Mesh& mesh) : derivative_(mesh) {}
    void tangent(const Points& dx, const Scalars& dV, const Points& dc, const Points& dA,
                 const Points& dxf) const override {
        derivative_.tangent(dx, dV, dc, dA, dxf);
    }
    void adjoint(const Scalars& Vbar, const Points& cbar, const Points& Abar, const Points& xfbar,
                 const Points& xbar) const override {
        derivative_.adjoint(Vbar, cbar, Abar, xfbar, xbar);
    }

private:
    voronoi::GeometryDerivative<Mesh> derivative_;
};

// Largest |a - b| over largest |b| of two device views.
template <class V>
double difference(const V& a, const V& b) {
    const auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a);
    const auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);
    double worst = 0, scale = 0;
    for (std::size_t k = 0; k < a.size(); ++k) {
        worst = std::max(worst, std::abs(ha.data()[k] - hb.data()[k]));
        scale = std::max(scale, std::abs(hb.data()[k]));
    }
    return worst / scale;
}

// Random entries (normal) times `scale`.
template <class V>
void fill(const V& v, double scale, std::mt19937_64& rng) {
    std::normal_distribution<double> normal(0.0, 1.0);
    const auto h = Kokkos::create_mirror_view(v);
    for (std::size_t k = 0; k < v.size(); ++k) h.data()[k] = scale * normal(rng);
    Kokkos::deep_copy(v, h);
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    Check check;
    for (const std::string kind : {"jittered", "uniform"}) {
        const exact_predicates::BoxN<2> box(1.0);
        const int side = 64, n = side * side;
        std::mt19937_64 rng(kind == "uniform" ? 2 : 1);
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        Kokkos::View<double* [2], Memory> x("x", n);
        auto hx = Kokkos::create_mirror_view(x);
        for (int i = 0; i < n; ++i)
            for (int a = 0; a < 2; ++a) {
                const int at = a == 0 ? i / side : i % side;
                const double u = kind == "uniform" ? unit(rng) : (at + 0.5 + 0.3 * (unit(rng) - 0.5)) / side;
                hx(i, a) = std::floor(u / box.step) * box.step;
            }
        Kokkos::deep_copy(x, hx);
        Mesh mesh(Exec{}, box);
        mesh.update(x);
        mesh.update_cells();
        const int faces = mesh.faces();
        const reference::Geometry2D<Mesh, Memory> luna(mesh);
        const Library library(mesh);
        const Interface& ours = library;
        const double spacing = 1.0 / side;
        Interface::Points dx("dx", n), dc[2] = {{"dc", n}, {"dc", n}}, dA[2] = {{"dA", faces}, {"dA", faces}},
                                       dxf[2] = {{"dxf", faces}, {"dxf", faces}};
        Interface::Scalars dV[2] = {{"dV", n}, {"dV", n}};
        fill(dx, spacing, rng);
        luna.tangent(dx, dV[0], dc[0], dA[0], dxf[0]);
        ours.tangent(dx, dV[1], dc[1], dA[1], dxf[1]);
        const double t = std::max(
            {difference(dV[1], dV[0]), difference(dc[1], dc[0]), difference(dA[1], dA[0]), difference(dxf[1], dxf[0])});
        std::cout << "  " << kind << " tangent: dV " << sci(difference(dV[1], dV[0])) << ", dc "
                  << sci(difference(dc[1], dc[0])) << ", dA " << sci(difference(dA[1], dA[0])) << ", dx_f "
                  << sci(difference(dxf[1], dxf[0])) << " of the largest\n";
        check.expect(t < 1e-12, kind + ": tangent equals luna-cpp's reference to " + sci(t) + " (gate 1e-12)");
        Interface::Scalars Vbar("Vbar", n);
        Interface::Points cbar("cbar", n), Abar("Abar", faces), xfbar("xfbar", faces),
            xbar[2] = {{"xbar", n}, {"xbar", n}};
        fill(Vbar, 1.0 / spacing, rng);
        fill(cbar, 1.0, rng);
        fill(Abar, 1.0, rng);
        fill(xfbar, 1.0, rng);
        luna.adjoint(Vbar, cbar, Abar, xfbar, xbar[0]);
        ours.adjoint(Vbar, cbar, Abar, xfbar, xbar[1]);
        const double a = difference(xbar[1], xbar[0]);
        std::cout << "  " << kind << " adjoint: x̄ " << sci(a) << " of the largest\n";
        check.expect(a < 1e-12, kind + ": adjoint equals luna-cpp's reference to " + sci(a) + " (gate 1e-12)");
    }
    std::cout << (check.failed() ? "FAILED: " : "passed: ") << check.failed() << " failures\n";
    return check.failed() != 0;
}
