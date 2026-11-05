// Mesh under flows that distort it (distortion.hpp): prescribed velocities, no
// hydro, no regularisation, then with a drift to the centroids in this driver
// (the library layer has none). A run may end in a refusal; every mesh it
// returns must pass the oracle (sampled steps).
//   distortion_test [flow ...] [--steps=N] [--n=N] [--d=spacings per step]
//                   [--every=N oracle] [--rebuild-every=N] [--cells=N brute cells]
//                   [--no-drift | --drift=chi] [--fallback=host|rebuild] [--print]

#include "check.hpp"
#include "distortion.hpp"

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    using namespace distortion;
    DeviceBytes::start();
    Check check;
    std::vector<std::string> flows;
    Options base;
    base.n = 1728;
    base.steps = 240;
    base.oracle_every = 24;
    base.oracle_cells = 8;
    base.print = false;
    std::vector<double> drifts{0, 0.1};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--steps=", 0) == 0) base.steps = std::stoi(a.substr(8));
        else if (a.rfind("--n=", 0) == 0) base.n = std::stoi(a.substr(4));
        else if (a.rfind("--d=", 0) == 0) base.displacement = std::stod(a.substr(4));
        else if (a.rfind("--every=", 0) == 0) base.oracle_every = std::stoi(a.substr(8));
        else if (a.rfind("--rebuild-every=", 0) == 0) base.rebuild_every = std::stoi(a.substr(16));
        else if (a.rfind("--cells=", 0) == 0) base.oracle_cells = std::stoi(a.substr(8));
        else if (a.rfind("--drift=", 0) == 0) drifts = {std::stod(a.substr(8))};
        else if (a == "--print") base.print = true;
        else if (a == "--dump") base.dump = true;
        else if (a == "--fallback=host") base.fallback = voronoi::Fallback::host;
        else if (a == "--fallback=rebuild") base.fallback = voronoi::Fallback::rebuild;
        else if (a == "--no-drift") drifts = {0};
        else flows.push_back(a);
    }
    if (flows.empty()) flows = all_flows();
    for (const auto& flow : flows)
        for (const double drift : drifts) {
            Options o = base;
            o.flow = flow;
            o.drift = drift;
            Kokkos::Timer timer;
            const Summary s = run(o);
            check.expect(s.wrong_step < 0, text(o, s) + " [" + sci(timer.seconds()) + " s]");
        }
    return check.failed() != 0;
}
