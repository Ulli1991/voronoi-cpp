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
        Points points = in_slot_order(make_points(c.kind, 4096));
        const int n = static_cast<int>(points.size());  // lattices round to their own sizes
        const double spacing = std::cbrt(1.0 / n);
        Positions x = to_device(scaled(points, c.box));
        MovingMesh moving(x, c.box);
        check.expect(moving.start.exact, "exact starting mesh");
        for (int step = 1; step <= steps; ++step) {
            flow.advect(points, displacement * spacing);
            x = to_device(scaled(points, c.box));
            const auto device = moving.repair(x);
            const auto host = device.stuck > 0 ? moving.finish() : tet_flips::Stats{};
            const tet_flips::Mesh mesh = tet_flips::to_host<Exec>(moving.mesh, moving.state);
            const long long failing = tet_flips::failing(mesh);
            const int faces = moving.update_geometry();

            const Reference exact = clipped(x, c.box);
            const NeighbourSets neighbours = delaunay_neighbours(mesh, n);
            long long mismatched = 0, exact_faces = 0;
            for (int i = 0; i < n; ++i) {
                mismatched += neighbours[i] != exact.neighbours[i];
                exact_faces += static_cast<long long>(exact.neighbours[i].size());
            }
            const auto volume = to_host(moving.geometry.volume);
            double worst = 0, total = 0;
            for (int i = 0; i < n; ++i) {
                total += volume(i);
                worst = std::max(worst, std::abs(volume(i) - exact.volume[i]) / exact.volume[i]);
            }
            std::ostringstream what;
            what << "step " << step << ": flips " << device.flips23 + device.flips32 << " stuck " << device.stuck
                 << " host flips " << host.flips23 + host.flips32 << " rebuilt " << host.rebuilt << " failing "
                 << failing << " mismatched " << mismatched << " faces " << faces << "/" << exact_faces / 2
                 << " volume error " << worst << " total " << total;
            check.expect(failing == 0 && mismatched == 0 && faces == exact_faces / 2 && worst < 1e-10 &&
                             std::abs(total / (c.box * c.box * c.box) - 1.0) < 1e-12,
                         what.str());
        }
    }
    return check.failed() != 0;
}
