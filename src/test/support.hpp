#pragma once

// Inputs, the moving-mesh driver and reference results shared by the tests
// and the benchmark. Reference checks that compare against every generator
// are brute force: test oracles only.

#include "neighbor_search.hpp"
#include "shell_voronoi_warp.hpp"
#include "tet_flips.hpp"
#include "tet_mesh.hpp"
#include "tet_repair.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace support {

using Exec = Kokkos::DefaultExecutionSpace;
using Memory = Exec::memory_space;
using Point = std::array<double, 3>;
using Points = std::vector<Point>;
using Positions = Kokkos::View<double* [3], Memory>;
using NeighbourSets = std::vector<std::vector<int>>;

inline double wrap(double x) { return x - std::floor(x); }
inline double nearest_image(double d) { return d - std::nearbyint(d); }

template <class View>
auto to_host(const View& v) {
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
}

// Degenerate layouts, "<layout>@<perturbation>" with the perturbation in
// spacings (each coordinate moved uniformly within +-perturbation):
//   sc, bcc, fcc  cubic lattices (8, 4 and 6 cells meeting at vertices)
//   shells        icosahedral shells of 12 points without a centre point,
//                 so 12 cells meet at each shell centre
//   planes        random points on exactly spaced planes
//   pairs         random points, each with a twin `perturbation` away
inline Points make_degenerate(const std::string& layout, double perturbation, int n) {
    std::mt19937_64 rng(20261029);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    Points points;
    auto basis_lattice = [&](const Points& basis) {
        const int side = std::max(3, static_cast<int>(std::lround(std::cbrt(double(n) / basis.size()))));
        for (int i = 0; i < side; ++i)
            for (int j = 0; j < side; ++j)
                for (int k = 0; k < side; ++k)
                    for (const auto& b : basis) points.push_back({(i + b[0]) / side, (j + b[1]) / side, (k + b[2]) / side});
    };
    if (layout == "sc") {
        basis_lattice({{0.5, 0.5, 0.5}});
    } else if (layout == "bcc") {
        basis_lattice({{0.25, 0.25, 0.25}, {0.75, 0.75, 0.75}});
    } else if (layout == "fcc") {
        basis_lattice({{0.25, 0.25, 0.25}, {0.75, 0.75, 0.25}, {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75}});
    } else if (layout == "shells") {
        const double g = (1.0 + std::sqrt(5.0)) / 2.0, r = 0.3 / std::sqrt(1.0 + g * g);
        Points basis;
        for (int s1 : {-1, 1})
            for (int s2 : {-1, 1}) {
                basis.push_back({0.5, 0.5 + s1 * r, 0.5 + s2 * g * r});
                basis.push_back({0.5 + s1 * r, 0.5 + s2 * g * r, 0.5});
                basis.push_back({0.5 + s2 * g * r, 0.5, 0.5 + s1 * r});
            }
        basis_lattice(basis);
    } else if (layout == "planes") {
        const int layers = std::max(3, static_cast<int>(std::lround(std::cbrt(n))));
        for (int i = 0; i < n; ++i) points.push_back({unit(rng), unit(rng), (i % layers + 0.5) / layers});
    } else if (layout == "pairs") {
        const double spacing = std::cbrt(2.0 / n);
        std::uniform_real_distribution<double> direction(-1.0, 1.0);
        for (int i = 0; i < n / 2; ++i) {
            const Point a{unit(rng), unit(rng), unit(rng)};
            points.push_back(a);
            points.push_back({wrap(a[0] + perturbation * spacing * direction(rng)),
                              wrap(a[1] + perturbation * spacing * direction(rng)),
                              wrap(a[2] + perturbation * spacing * direction(rng))});
        }
        return points;
    } else {
        throw std::invalid_argument("unknown degenerate layout " + layout);
    }
    const double spacing = std::cbrt(1.0 / points.size());
    std::uniform_real_distribution<double> shift(-perturbation * spacing, perturbation * spacing);
    if (perturbation > 0)
        for (auto& p : points)
            for (double& x : p) x = wrap(x + shift(rng));
    return points;
}

// Points in the unit box: uniform, jittered (lattice moved by up to 0.1
// spacings), lattice, wave (lattice moved by 1e-5 spacings), clustered, halo,
// or a degenerate layout. Lattices round n to a cube.
inline Points make_points(const std::string& kind, int n) {
    if (const auto at = kind.find('@'); at != std::string::npos)
        return make_degenerate(kind.substr(0, at), std::stod(kind.substr(at + 1)), n);
    std::mt19937_64 rng(20261029);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    Points points;
    if (kind == "jittered" || kind == "lattice" || kind == "wave") {
        const int side = static_cast<int>(std::lround(std::cbrt(n)));
        const double spacing = 1.0 / side;
        const double jitter = kind == "lattice" ? 0.0 : kind == "wave" ? 1e-5 * spacing : 0.1 * spacing;
        std::uniform_real_distribution<double> shift(-jitter, jitter);
        for (int i = 0; i < side; ++i)
            for (int j = 0; j < side; ++j)
                for (int k = 0; k < side; ++k)
                    points.push_back({wrap((i + 0.5) * spacing + shift(rng)), wrap((j + 0.5) * spacing + shift(rng)),
                                      wrap((k + 0.5) * spacing + shift(rng))});
    } else if (kind == "clustered") {
        // Eight Gaussian blobs over a uniform background: about 1:100 density.
        std::normal_distribution<double> blob(0.0, 0.04);
        std::array<Point, 8> centres;
        for (auto& c : centres) c = {unit(rng), unit(rng), unit(rng)};
        for (int i = 0; i < n; ++i) {
            if (i % 5 == 0) {
                points.push_back({unit(rng), unit(rng), unit(rng)});
            } else {
                const auto& c = centres[i % 8];
                points.push_back({wrap(c[0] + blob(rng)), wrap(c[1] + blob(rng)), wrap(c[2] + blob(rng))});
            }
        }
    } else if (kind == "halo") {
        // Four halos with density 1 / (r^2 + c^2) out to r = 0.2, c = 0.002
        // (core to edge 1:10^4), over a 10 % uniform background: about
        // 1:10^5 from the voids to the halo cores.
        constexpr double edge = 0.2, core = 0.002;
        std::array<Point, 4> centres;
        for (auto& c : centres) c = {unit(rng), unit(rng), unit(rng)};
        // The enclosed mass m(r) = r - c atan(r / c), inverted by bisection.
        const double total = edge - core * std::atan(edge / core);
        std::normal_distribution<double> gauss(0.0, 1.0);
        for (int i = 0; i < n; ++i) {
            if (i % 10 == 0) {
                points.push_back({unit(rng), unit(rng), unit(rng)});
                continue;
            }
            const double target = total * unit(rng);
            double lo = 0, hi = edge;
            for (int k = 0; k < 60; ++k) {
                const double mid = 0.5 * (lo + hi);
                (mid - core * std::atan(mid / core) < target ? lo : hi) = mid;
            }
            const Point g{gauss(rng), gauss(rng), gauss(rng)};
            const double scale = 0.5 * (lo + hi) / std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
            const auto& c = centres[i % 4];
            points.push_back({wrap(c[0] + scale * g[0]), wrap(c[1] + scale * g[1]), wrap(c[2] + scale * g[2])});
        }
    } else if (kind == "uniform") {
        for (int i = 0; i < n; ++i) points.push_back({unit(rng), unit(rng), unit(rng)});
    } else {
        throw std::invalid_argument("unknown distribution " + kind);
    }
    return points;
}

inline Positions to_device(const Points& points) {
    Positions x("positions", points.size());
    auto h = Kokkos::create_mirror_view(x);
    for (std::size_t i = 0; i < points.size(); ++i)
        for (int a = 0; a < 3; ++a) h(i, a) = points[i][a];
    Kokkos::deep_copy(x, h);
    return x;
}

// A cell list over the unit box with about one particle per cell.
inline neighbor_search::CellList<Exec> cell_list(const Positions& x, double width_factor = 1.0, bool morton = false) {
    neighbor_search::CellList<Exec> list;
    list.morton = morton;
    const int n = static_cast<int>(x.extent(0));
    neighbor_search::configure(list, n, 1.0, width_factor * std::cbrt(1.0 / n));
    neighbor_search::build(Exec{}, list, x);
    return list;
}

// The points renumbered in slot order, so that a cell list built on them has
// the identity order and slots name particles from then on.
inline Points in_slot_order(const Points& points) {
    const auto order = to_host(cell_list(to_device(points)).order);
    Points sorted(points.size());
    for (std::size_t t = 0; t < points.size(); ++t) sorted[t] = points[order(t)];
    return sorted;
}

// A smooth periodic velocity field of unit-order speed: six random modes.
class Flow {
public:
    Flow() {
        std::mt19937_64 rng(11);
        std::uniform_real_distribution<double> unit(-1.0, 1.0);
        for (auto& m : modes_) {
            for (int a = 0; a < 3; ++a) m[a] = std::round(2.5 * unit(rng));
            for (int a = 3; a < 6; ++a) m[a] = unit(rng);
            m[6] = two_pi * unit(rng);
        }
    }

    Point velocity(const Point& x) const {
        Point v{0, 0, 0};
        for (const auto& m : modes_) {
            const double s = std::sin(two_pi * (m[0] * x[0] + m[1] * x[1] + m[2] * x[2]) + m[6]);
            for (int a = 0; a < 3; ++a) v[a] += m[3 + a] * s / 3.0;
        }
        return v;
    }

    // Move every point by `distance` times the velocity, wrapped into the box.
    void advect(Points& x, double distance) const {
        for (auto& p : x) {
            const Point v = velocity(p);
            for (int a = 0; a < 3; ++a) p[a] = wrap(p[a] + distance * v[a]);
        }
    }

private:
    static constexpr double two_pi = 6.283185307179586;
    std::array<std::array<double, 7>, 6> modes_;  // wave vector (3), direction (3), phase
};

template <class F>
double median_ms(int repeats, const F& f) {
    f();  // warm-up
    std::vector<double> times;
    for (int r = 0; r < repeats; ++r) {
        Kokkos::Timer timer;
        f();
        Exec{}.fence();
        times.push_back(timer.seconds() * 1e3);
    }
    std::nth_element(times.begin(), times.begin() + times.size() / 2, times.end());
    return times[times.size() / 2];
}

// Checks of a clipping build that need no other construction: every cell
// certified, volumes positive and summing to the box, area vectors closing
// each cell, every face seen from both sides (faces = 2 x owned), and each
// owned face centroid equidistant from its two generators with no generator
// closer (brute force over all generators for a sample of cells). Prints a
// one-line summary; returns the number of failed checks.
inline int check_cells(const shell_voronoi::Cells<Memory>& cells, const Positions& slot_positions, bool is_double,
                       std::ostream& out) {
    const double volume_tolerance = is_double ? 1e-12 : 1e-5, closure_tolerance = is_double ? 1e-12 : 1e-5,
                 distance_tolerance = is_double ? 1e-9 : 1e-4;
    const auto status = to_host(cells.status);
    const auto face_count = to_host(cells.face_count);
    const auto owned_count = to_host(cells.owned_count);
    const auto neighbour = to_host(cells.neighbour);
    const auto centroid = to_host(cells.face_centroid);
    const auto volume = to_host(cells.volume);
    const auto closure = to_host(cells.closure);
    const auto work = to_host(cells.work);
    const auto x = to_host(slot_positions);
    const int n = static_cast<int>(status.extent(0));
    std::array<int, shell_voronoi::status_count> statuses{};
    double total_volume = 0, min_volume = 1e300, max_closure = 0;
    long long faces = 0, owned = 0;
    std::array<double, 3> per_cell{};
    for (int t = 0; t < n; ++t) {
        ++statuses[status(t)];
        total_volume += volume(t);
        min_volume = std::min(min_volume, volume(t));
        max_closure = std::max(max_closure, closure(t));
        faces += face_count(t);
        owned += owned_count(t);
        for (int k = 0; k < 3; ++k) per_cell[k] += double(work(t, k)) / n;
    }
    // About 2.5e8 distances in all.
    const int sampled = static_cast<int>(std::clamp(2.5e8 / n, 16.0, double(n)));
    const int stride = std::max(1, n / sampled);
    long long checked = 0, violations = 0;
    double worst = 0;
    for (int t = 0; t < n; t += stride)
        for (int f = 0; f < owned_count(t); ++f) {
            Point c;
            for (int a = 0; a < 3; ++a) c[a] = x(t, a) + centroid(t, f, a);
            auto distance = [&](int k) {
                double d2 = 0;
                for (int a = 0; a < 3; ++a) {
                    const double d = nearest_image(x(k, a) - c[a]);
                    d2 += d * d;
                }
                return std::sqrt(d2);
            };
            const double own = distance(t), other = distance(neighbour(t, f));
            double closest = own;
            for (int k = 0; k < n; ++k) closest = std::min(closest, distance(k));
            const double error = std::max(std::abs(own - other), own - closest) / own;
            worst = std::max(worst, error);
            ++checked;
            violations += error > distance_tolerance;
        }
    const double volume_error = std::abs(total_volume - 1.0);
    int failures = 0;
    failures += statuses[shell_voronoi::ok] != n;
    failures += volume_error > volume_tolerance;
    failures += !(min_volume > 0);
    failures += max_closure > closure_tolerance;
    failures += faces != 2 * owned;
    failures += violations > 0;
    out << "  certified=" << statuses[shell_voronoi::ok] << "/" << n << " statuses=";
    for (int s : statuses) out << s << ",";
    out << " volume_error=" << volume_error << " min_volume=" << min_volume << " max_closure=" << max_closure
        << " faces_per_cell=" << double(faces) / n << " faces=2*owned:" << (faces == 2 * owned ? "yes" : "NO")
        << " centroid_checks=" << checked << " worst_centroid_error=" << worst << " violations=" << violations
        << " per_cell(rings,clipped,cuts)=" << per_cell[0] << "," << per_cell[1] << "," << per_cell[2]
        << (failures ? "  FAIL" : "  PASS") << "\n";
    return failures;
}

// Clipping build of x (in slot order) that records face rings, for
// tet_mesh::from_rings.
inline shell_voronoi::Cells<Memory> cells_with_rings(const Positions& x) {
    const int n = static_cast<int>(x.extent(0));
    const auto list = cell_list(x);
    shell_voronoi::Cells<Memory> cells(n, false);
    cells.rings = Kokkos::View<int**, Memory>("rings", n, 384);
    Kokkos::deep_copy(cells.rings, -1);
    shell_voronoi::construct<double>(Exec{}, shell_voronoi::make_grid(list), cells);
    return cells;
}

// The Voronoi cells of x by clipping, by particle: every particle's natural
// neighbours (from the owned faces), volume and build status.
struct Reference {
    NeighbourSets neighbours;
    std::vector<double> volume;
    std::vector<int> status;
};

inline Reference clipped(const Positions& x) {
    const int n = static_cast<int>(x.extent(0));
    const auto list = cell_list(x);
    shell_voronoi::Cells<Memory> cells(n, false);
    shell_voronoi::construct_warp<double>(Exec{}, shell_voronoi::make_grid(list), cells);
    const auto order = to_host(list.order);
    const auto owned = to_host(cells.owned_count);
    const auto neighbour = to_host(cells.neighbour);
    const auto volume = to_host(cells.volume);
    const auto status = to_host(cells.status);
    Reference r{NeighbourSets(n), std::vector<double>(n), std::vector<int>(n)};
    for (int t = 0; t < n; ++t) {
        const int i = order(t);
        r.volume[i] = volume(t);
        r.status[i] = status(t);
        for (int f = 0; f < owned(t); ++f) {
            const int j = order(neighbour(t, f));
            r.neighbours[i].push_back(j);
            r.neighbours[j].push_back(i);
        }
    }
    for (auto& s : r.neighbours) std::sort(s.begin(), s.end());
    return r;
}

// Every particle's Delaunay neighbours in a host mesh, including faces of
// zero area.
inline NeighbourSets delaunay_neighbours(const tet_flips::Mesh& m, int n) {
    NeighbourSets sets(n);
    for (int t = 0; t < m.count; ++t)
        if (m.alive(t))
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b)
                    if (a != b) sets[m.vertex(t, a)].push_back(m.vertex(t, b));
    for (auto& s : sets) {
        std::sort(s.begin(), s.end());
        s.erase(std::unique(s.begin(), s.end()), s.end());
    }
    return sets;
}

// The persistent mesh of moving particles, stepped as the pipeline does:
//   repair: pad positions, circumcentres, screen, flip on the device;
//   finish: the host fallback, only when the device repair left stuck entries;
//   update_geometry: circumcentres, face rings, faces and volumes.
// Particles are slots of the starting build (positions in slot order).
struct MovingMesh {
    Exec exec;
    int n = 0;
    tet_mesh::Mesh<Memory> mesh;
    tet_repair::State<Memory> state;
    tet_mesh::FaceRings<Memory> rings[2];
    int current = 0;  // rings[current] holds the last step's faces
    tet_mesh::Geometry<Memory> geometry;
    tet_mesh::Points4<Memory> padded;
    int unresolved = 0;  // from_rings could not resolve these

    MovingMesh(const Positions& x, const shell_voronoi::Cells<Memory>& cells) : n(static_cast<int>(x.extent(0))) {
        unresolved = tet_mesh::from_rings(exec, cells, mesh);
        const int capacity = 2 * static_cast<int>(mesh.vertex.extent(0));
        state = tet_repair::prepare(exec, x, mesh, capacity);
        const int face_capacity = 10 * n;
        for (auto& r : rings) r = tet_mesh::make_face_rings<Memory>(capacity, face_capacity, 8 * face_capacity);
        geometry = {Kokkos::View<double* [3], Memory>("face_area", face_capacity),
                    Kokkos::View<double* [3], Memory>("face_centroid", face_capacity),
                    Kokkos::View<double*, Memory>("cell_volume", n)};
        padded = tet_mesh::Points4<Memory>("padded_positions", n);
    }

    tet_repair::Stats repair(const Positions& x) {
        tet_mesh::pad(exec, x, padded);
        tet_mesh::circumcentres(exec, padded, 1.0, mesh, state.count);
        tet_repair::move_screened(exec, padded, 1.0, mesh, state);
        return tet_repair::repair(exec, mesh, state);
    }

    tet_flips::Stats finish() { return tet_flips::finish(exec, mesh, state); }

    // Returns the number of faces.
    int update_geometry() {
        tet_mesh::circumcentres(exec, padded, 1.0, mesh, state.count);
        const int faces = tet_mesh::update_face_rings(exec, mesh, state.count, state.touched, rings[current],
                                                      rings[1 - current]);
        current = 1 - current;
        tet_mesh::faces(exec, padded, 1.0, mesh, geometry);
        Kokkos::deep_copy(exec, state.touched, 0);
        return faces;
    }
};

}  // namespace support
