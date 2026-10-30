// Repair of the persistent mesh as particles move: jittered points carried
// by a smooth flow for ten steps of 0.03 and of 0.3 spacings each. Every
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
    const int n = 4096, steps = 10;
    const double spacing = std::cbrt(1.0 / n);
    const Flow flow;
    for (const double displacement : {0.03, 0.3}) {
        std::cout << "displacement " << displacement << " spacings per step:\n";
        Points points = in_slot_order(make_points("jittered", n));
        Positions x = to_device(points);
        MovingMesh moving(x, cells_with_rings(x));
        check.expect(moving.unresolved == 0, "starting mesh from rings");
        for (int step = 1; step <= steps; ++step) {
            flow.advect(points, displacement * spacing);
            x = to_device(points);
            const auto device = moving.repair(x);
            const auto host = device.stuck > 0 ? moving.finish() : tet_flips::Stats{};
            const tet_flips::Mesh mesh = tet_flips::to_host<Exec>(moving.mesh, moving.state);
            const long long failing = tet_flips::failing(mesh);
            const int faces = moving.update_geometry();

            const Reference exact = clipped(x);
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
                             std::abs(total - 1.0) < 1e-12,
                         what.str());
        }
    }
    return check.failed() != 0;
}
