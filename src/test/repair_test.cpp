// Repair of the persistent mesh as particles move: jittered points (ten
// steps), a lattice and a wave lattice (four steps; their exact starting
// meshes are decided by the perturbation alone) carried by a smooth flow
// at 0.03 and at 0.3 spacings per step, and icosahedral shells (two steps
// at 0.03; their first step needs the host's local rebuilds), and jittered
// points in a box of side 2 (snapping divides by the box). Every
// step runs the pipeline (screened device flips, host fallback when stuck,
// face rings and geometry), then the mesh must be Delaunay, its neighbours
// must equal those of an exact rebuild by clipping, and the volumes from
// its faces must match the clipped cells.

#include "check.hpp"
#include "support.hpp"

#include <sstream>

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    using namespace support;
    Check check;
    const Flow flow;
    struct Case {
        const char* kind;
        double displacement;
        int steps;
        double box = 1.0;
    };
    for (const Case c : {Case{"jittered", 0.03, 10}, Case{"jittered", 0.3, 10}, Case{"lattice", 0.03, 4},
                         Case{"lattice", 0.3, 4}, Case{"wave", 0.03, 4}, Case{"wave", 0.3, 4},
                         Case{"shells@1e-9", 0.03, 2}, Case{"jittered", 0.3, 3, 2.0}}) {
        const double displacement = c.displacement;
        const int steps = c.steps;
        std::cout << c.kind << ", displacement " << displacement << " spacings per step, box " << c.box << ":\n";
        Points points = make_points(c.kind, 4096);
        const int n = static_cast<int>(points.size());  // lattices round to their own sizes
        const double spacing = std::cbrt(1.0 / n);
        Positions x = to_device(scaled(points, c.box));
        Mesh mesh(Exec{}, c.box);
        check.expect(mesh.update(x).start.exact, "exact starting mesh");
        for (int step = 1; step <= steps; ++step) {
            flow.advect(points, displacement * spacing);
            x = to_device(scaled(points, c.box));
            const voronoi::Report report = mesh.update(x);
            const Comparison d = compare_with_rebuild(mesh, x);
            std::ostringstream what;
            what << "step " << step << ": flips " << report.flips << " stuck " << report.stuck << " host flips "
                 << report.host_flips << " rebuilt " << report.host_rebuilds << "; " << d;
            check.expect(d.failing == 0 && d.mismatched == 0 && d.faces == d.clipped_faces && d.volume < 1e-10 &&
                             std::abs(d.total - 1.0) < 1e-12,
                         what.str());
        }
    }
    return check.failed() != 0;
}
