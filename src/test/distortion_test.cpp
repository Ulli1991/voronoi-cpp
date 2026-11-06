// Mesh under flows that distort it (distortion.hpp): prescribed velocities, no hydro, then with a drift to the
// centroids.

#include "check.hpp"
#include "distortion.hpp"

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    using namespace distortion;
    DeviceBytes::start();
    Check check;
    std::vector<std::string> flows;
    Options base;
    base.n = 1000;
    base.steps = 80;
    base.oracle_every = 40;
    base.oracle_cells = 4;
    base.print = false;
    std::vector<double> drifts{0};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--steps=", 0) == 0) base.steps = std::stoi(a.substr(8));
        else if (a.rfind("--n=", 0) == 0) base.n = std::stoi(a.substr(4));
        else if (a.rfind("--d=", 0) == 0) base.displacement = std::stod(a.substr(4));
        else if (a.rfind("--every=", 0) == 0) base.oracle_every = std::stoi(a.substr(8));
        else if (a.rfind("--rebuild-every=", 0) == 0) base.rebuild_every = std::stoi(a.substr(16));
        else if (a.rfind("--cells=", 0) == 0) base.oracle_cells = std::stoi(a.substr(8));
        else if (a.rfind("--drift=", 0) == 0) drifts = {std::stod(a.substr(8))};
        else if (a.rfind("--seconds=", 0) == 0) base.seconds = std::stod(a.substr(10));
        else if (a.rfind("--shape-every=", 0) == 0) base.shape_every = std::stoi(a.substr(14));
        else if (a == "--final-oracle") base.final_oracle = true;
        else if (a == "--print") base.print = true;
        else if (a == "--dump") base.dump = true;
        else if (a == "--fallback=host") base.fallback = voronoi::Fallback::host;
        else if (a == "--fallback=rebuild") base.fallback = voronoi::Fallback::rebuild;
        else if (a == "--no-drift") drifts = {0};
        else flows.push_back(a);
    }
    const bool defaults = flows.empty();
    if (defaults) flows = all_flows();
    for (const auto& flow : flows)
        for (const double drift : drifts) {
            Options o = base;
            o.flow = flow;
            o.drift = drift;
            Kokkos::Timer timer;
            const Summary s = run(o);
            check.expect(s.wrong_step < 0, text(o, s) + " [" + sci(timer.seconds()) + " s]");
        }
    if (defaults)  // the drift of this driver, on three flows
        for (const std::string flow : {"shear", "turbulence", "thin"}) {
            Options o = base;
            o.flow = flow;
            o.drift = 0.1;
            const Summary s = run(o);
            check.expect(s.wrong_step < 0, text(o, s));
        }
    return check.failed() != 0;
}
