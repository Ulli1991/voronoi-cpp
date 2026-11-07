// Shared by the mesh_cases_test groups (one .cpp per group in this directory): the case registry, the checked build and
// step helpers, point generators and exact lattices. Used by: every group and main in mesh_cases_test.cpp.
#pragma once

#include "check.hpp"
#include "distortion.hpp"
#include "mesh_check.hpp"

#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <random>

namespace mesh_cases {

using namespace support;
namespace mc = mesh_check;

// A named test case; groups register them with add(), main runs the ones selected on the command line.
struct Case {
    std::string name;
    std::function<void(Check&)> run;
    int issue = 0;  // an open issue: runs with --known only
};
// The registry of all cases, in registration order. Used by: add, main.
inline std::vector<Case>& cases() {
    static std::vector<Case> all;
    return all;
}
// Marks a case as a known open issue on the CUDA build only (pass to add()); elsewhere it still runs.
#ifdef KOKKOS_ENABLE_CUDA
constexpr int on_cuda(int issue) { return issue; }
#else
constexpr int on_cuda(int) { return 0; }
#endif

// Registers a case. `issue` > 0 marks it known: it is skipped unless --known is given. Used by: every group.
inline void add(const std::string& name, std::function<void(Check&)> run, int issue = 0) {
    cases().push_back({name, std::move(run), issue});
}

// What build() accepts: an exact mesh, either an exact mesh or a refusal, or a refusal.
enum class Expect { exact, exact_or_refused, refused };

// A fresh build of `points`, judged by mesh_check::check on `cells` sampled cells. Returns whether there is a mesh.
// Used by: nearly every group (the start: tet_start::build plus repair).
inline bool build(Check& check, const std::string& name, Mesh& mesh, const Points& points, Expect expect,
                  int cells = 32) {
    try {
        const voronoi::Report r = mesh.update(to_device(points));
        mesh.update_cells();
        const mc::Result c = mc::check(mesh, cells);
        check.expect(expect != Expect::refused && c.ok(), name + ": built (patched " + std::to_string(r.start.patched) +
                                                              ", rounds " + std::to_string(r.start.rounds) + ", faces " +
                                                              std::to_string(r.faces) + "): " + c.text());
        return true;
    } catch (const std::exception& e) {
        check.expect(expect != Expect::exact, name + ": refused: " + e.what());
        return false;
    }
}

// One update to `points` (a step of the repair), judged by mesh_check::check. Used by: every group that moves points.
inline voronoi::Report step(Check& check, const std::string& name, Mesh& mesh, const Points& points, int cells = 16) {
    const voronoi::Report r = mesh.update(to_device(points));
    mesh.update_cells();
    const mc::Result c = mc::check(mesh, cells);
    std::ostringstream what;
    what << name << ": flips " << r.flips << " rounds " << r.rounds << " stuck " << r.stuck << " host " << r.host_flips
         << "/" << r.host_rebuilds << " rebuilt " << r.rebuilt << " overlapped " << r.overlapped << ": " << c.text();
    check.expect(c.ok(), what.str());
    return r;
}

// The same, accepting a refusal: `refused` is set and the report is empty. Used by: groups whose steps may refuse.
inline voronoi::Report step_or_refused(Check& check, const std::string& name, Mesh& mesh, const Points& points,
                                       bool& refused, int cells = 16) {
    try {
        return step(check, name, mesh, points, cells);
    } catch (const std::exception& e) {
        check.expect(true, name + ": refused: " + e.what());
        refused = true;
        return {};
    }
}

// A uniformly random unit vector. Used by: add_ball, add_shell and cases that place points on spheres.
inline Point on_sphere(std::mt19937_64& rng) {
    std::normal_distribution<double> g(0, 1);
    const Point p{g(rng), g(rng), g(rng)};
    const double r = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    return {p[0] / r, p[1] / r, p[2] / r};
}

// Appends n points uniform in the ball (wrapped into the box). Used by: contrast_cases.
inline void add_ball(Points& p, std::mt19937_64& rng, Point c, double radius, int n) {
    std::uniform_real_distribution<double> u(0, 1);
    for (int i = 0; i < n; ++i) {
        const Point d = on_sphere(rng);
        const double r = radius * std::cbrt(u(rng));
        p.push_back({wrap(c[0] + r * d[0]), wrap(c[1] + r * d[1]), wrap(c[2] + r * d[2])});
    }
}

// Appends n points on the sphere of that radius (wrapped). Used by: contrast_cases, valence_cases, fallback_cases.
inline void add_shell(Points& p, std::mt19937_64& rng, Point c, double radius, int n) {
    for (int i = 0; i < n; ++i) {
        const Point d = on_sphere(rng);
        p.push_back({wrap(c[0] + radius * d[0]), wrap(c[1] + radius * d[1]), wrap(c[2] + radius * d[2])});
    }
}

// Appends n points uniform in the unit box. Used by: most groups as background.
inline void add_uniform(Points& p, std::mt19937_64& rng, int n) {
    std::uniform_real_distribution<double> u(0, 1);
    for (int i = 0; i < n; ++i) p.push_back({u(rng), u(rng), u(rng)});
}

// One step of the predicates' grid in the unit box.
const double grid_step = 1.0 / exact_predicates::scale;

// A cubic lattice of side^3 cells with the basis, side a power of two, so
// every position is exact on the grid. Used by: box_cases, degenerate_cases, predicate_cases.
inline Points exact_lattice(int side, const Points& basis) {
    Points p;
    for (int i = 0; i < side; ++i)
        for (int j = 0; j < side; ++j)
            for (int k = 0; k < side; ++k)
                for (const auto& b : basis) p.push_back({(i + b[0]) / side, (j + b[1]) / side, (k + b[2]) / side});
    return p;
}
const Points sc_basis{{0.5, 0.5, 0.5}}, bcc_basis{{0.25, 0.25, 0.25}, {0.75, 0.75, 0.75}},
    fcc_basis{{0.25, 0.25, 0.25}, {0.75, 0.75, 0.25}, {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75}};

// The groups of cases, one per file here; main calls each to register its cases.
void coincident_cases();
void few_cases();
void contrast_cases();
void box_cases();
void degenerate_cases();
void valence_cases();
void large_cases();
void fallback_cases();
void predicate_cases();
void reproducibility_cases();

}  // namespace mesh_cases
