// 3D special cases against the oracle: a case needs the exact mesh or a refusal, never a wrong mesh with no error.
// `list` prints the names, `<prefix> ...` runs matching cases; known open issues run with --known only.
// Exercises voronoi::Mesh<3> (start by tet_start, repair by tet_repair, fallbacks, update_cells) via mesh_check.hpp.

#include "mesh_cases/common.hpp"

using namespace mesh_cases;

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    coincident_cases();
    few_cases();
    contrast_cases();
    box_cases();
    degenerate_cases();
    valence_cases();
    large_cases();
    fallback_cases();
    predicate_cases();
    reproducibility_cases();
    std::vector<std::string> wanted;
    bool known = false, list = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--known")
            known = true;
        else if (a == "list")
            list = true;
        else
            wanted.push_back(a);
    }
    if (list) {
        for (const auto& c : cases()) std::cout << c.name << (c.issue ? " known" : "") << "\n";
        return 0;
    }
    // ctest runs a subset within its time budget; MESH_CASES_FULL=1 runs the whole matrix.
    const bool full = std::getenv("MESH_CASES_FULL") != nullptr;
    auto subset_skips = [&](const std::string& name) {
        static const std::vector<std::string> only_full{
            "box.void.centre.",     "box.void.empty.0.1",  "box.void.empty.0.22", "box.void.empty.0.25",
            "box.void.empty.0.3",   "box.void.empty.0.45", "box.slab.lattice.",   "box.slab.uniform.",
            "box.slab.jittered.7",  "box.slab.jittered.8", "box.thin.",           "contrast.ball.0.0",
            "contrast.ball.1e-05",  "contrast.ball.1e-06", "contrast.shell.",     "contrast.shell_centre.",
            "contrast.two_density.100",                    "contrast.two_density.1e+06",
            "contrast.grid_scale.1024",                    "contrast.grid_scale.4096",
            "contrast.grid_scale.16384",                   "contrast.grid_scale.65536",
            "contrast.sphere_status.1",                    "degenerate.noise.bcc.", "degenerate.noise.fcc.",
            "degenerate.noise.sc.16",                      "degenerate.noise.sc.1024",
            "degenerate.shear.smooth.", "degenerate.rigid.off_grid.", "large.random_walk.1", "large.flow.20",
            "fallback.large.", "fallback.growth.", "few.jittered.1", "few.lattice.1", "repro.dump."};
        if (full || std::find(wanted.begin(), wanted.end(), name) != wanted.end()) return false;
        return std::any_of(only_full.begin(), only_full.end(), [&](const std::string& p) { return name.rfind(p, 0) == 0; });
    };
    Check check;
    for (const auto& c : cases()) {
        if (subset_skips(c.name)) continue;
        if (!wanted.empty() &&
            std::none_of(wanted.begin(), wanted.end(), [&](const std::string& w) { return c.name.rfind(w, 0) == 0; }))
            continue;
        if (c.issue && !known) {
            std::cout << "skip  " << c.name << " (known, issue #" << c.issue << ")\n";
            continue;
        }
        std::cout << "case " << c.name << "\n" << std::flush;
        Kokkos::Timer timer;
        c.run(check);
        std::cout << "  time " << timer.seconds() << " s\n" << std::flush;
    }
    return check.failed() != 0;
}
