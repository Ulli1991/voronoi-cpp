#pragma once

// Inputs, reference results and comparisons shared by the tests
// and the benchmark. Reference checks that compare against every generator
// are brute force: test oracles only.

#include "neighbor_search.hpp"
#include "shell_voronoi_warp.hpp"
#include "tet_flips.hpp"
#include "tet_mesh.hpp"
#include "tet_repair.hpp"
#include "tet_start.hpp"
#include "voronoi_mesh.hpp"

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
using Box = exact_predicates::Box;
// Sides relative to the longest, (1, 1, 1) for a cube: lattices get about
// equal spacings along every axis.
using Shape = std::array<double, 3>;
inline Shape shape(const Box& box) {
    const double longest = std::max({box.side[0], box.side[1], box.side[2]});
    return {box.side[0] / longest, box.side[1] / longest, box.side[2] / longest};
}
// Lattice cells along each axis for about `cells` cells of equal spacing in
// a box of shape s (at least `fewest` per axis).
inline std::array<int, 3> lattice_sides(double cells, const Shape& s, int fewest) {
    const double per_unit = std::cbrt(cells / (s[0] * s[1] * s[2]));
    return {std::max(fewest, static_cast<int>(std::lround(per_unit * s[0]))),
            std::max(fewest, static_cast<int>(std::lround(per_unit * s[1]))),
            std::max(fewest, static_cast<int>(std::lround(per_unit * s[2])))};
}

inline double wrap(double x) { return x - std::floor(x); }

// Positions on the grid of the exact predicates (2^30 steps across the box).
// The mesh's topology is decided on snapped positions and its geometry
// computed from the positions themselves; off the grid, the two disagree on
// near-degenerate cells (volumes ~1e-9 apart) and for particles closer than
// ~1e3 grid steps. Every input and flow step here is kept on the grid.
inline void to_grid(Points& points) {
    for (auto& p : points)
        for (double& x : p) x = std::floor(x * exact_predicates::scale) / exact_predicates::scale;
}
inline double nearest_image(double d, double side = 1.0) { return d - side * std::nearbyint(d / side); }

// The same on the grid of a box (exact_predicates::snap): what voronoi::Mesh
// computes from them.
inline void to_grid(Points& points, const Box& box) {
    for (auto& p : points)
        for (int a = 0; a < 3; ++a) p[a] = exact_predicates::snap(p[a], box, a) * box.step;
}

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
inline Points make_degenerate(const std::string& layout, double perturbation, int n, const Shape& shape = {1, 1, 1}) {
    std::mt19937_64 rng(20261029);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    Points points;
    auto basis_lattice = [&](const Points& basis) {
        const auto side = lattice_sides(double(n) / basis.size(), shape, 3);
        for (int i = 0; i < side[0]; ++i)
            for (int j = 0; j < side[1]; ++j)
                for (int k = 0; k < side[2]; ++k)
                    for (const auto& b : basis)
                        points.push_back({(i + b[0]) / side[0], (j + b[1]) / side[1], (k + b[2]) / side[2]});
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
        const int layers = lattice_sides(n, shape, 3)[2];
        for (int i = 0; i < n; ++i) points.push_back({unit(rng), unit(rng), (i % layers + 0.5) / layers});
    } else if (layout == "pairs") {
        const double spacing = std::cbrt(2.0 * shape[0] * shape[1] * shape[2] / n);
        std::uniform_real_distribution<double> direction(-1.0, 1.0);
        for (int i = 0; i < n / 2; ++i) {
            const Point a{unit(rng), unit(rng), unit(rng)};
            points.push_back(a);
            points.push_back({wrap(a[0] + perturbation * spacing / shape[0] * direction(rng)),
                              wrap(a[1] + perturbation * spacing / shape[1] * direction(rng)),
                              wrap(a[2] + perturbation * spacing / shape[2] * direction(rng))});
        }
        return points;
    } else {
        throw std::invalid_argument("unknown degenerate layout " + layout);
    }
    const double spacing = std::cbrt(shape[0] * shape[1] * shape[2] / points.size());
    std::uniform_real_distribution<double> shift(-perturbation * spacing, perturbation * spacing);
    if (perturbation > 0)
        for (auto& p : points)
            for (int a = 0; a < 3; ++a) p[a] = wrap(p[a] + shift(rng) / shape[a]);
    return points;
}

// Points in the unit box: uniform, jittered (lattice moved by up to 0.1
// spacings), lattice, wave (lattice moved by 1e-5 spacings), clustered, halo,
// or a degenerate layout. Lattices round n to a cube, or with `shape` to a
// lattice of about equal spacings once scaled to a box of that shape.
inline Points make_points_off_grid(const std::string& kind, int n, const Shape& shape = {1, 1, 1}) {
    if (const auto at = kind.find('@'); at != std::string::npos)
        return make_degenerate(kind.substr(0, at), std::stod(kind.substr(at + 1)), n, shape);
    std::mt19937_64 rng(20261029);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    Points points;
    if (kind == "jittered" || kind == "lattice" || kind == "wave") {
        const auto side = lattice_sides(n, shape, 1);
        const double jitter = kind == "lattice" ? 0.0 : kind == "wave" ? 1e-5 : 0.1;  // spacings
        std::array<std::uniform_real_distribution<double>, 3> shift;
        Point spacing;
        for (int a = 0; a < 3; ++a) {
            spacing[a] = 1.0 / side[a];
            shift[a] = std::uniform_real_distribution<double>(-jitter * spacing[a], jitter * spacing[a]);
        }
        for (int i = 0; i < side[0]; ++i)
            for (int j = 0; j < side[1]; ++j)
                for (int k = 0; k < side[2]; ++k)
                    points.push_back({wrap((i + 0.5) * spacing[0] + shift[0](rng)),
                                      wrap((j + 0.5) * spacing[1] + shift[1](rng)),
                                      wrap((k + 0.5) * spacing[2] + shift[2](rng))});
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

// The same on the grid of the exact predicates (to_grid).
inline Points make_points(const std::string& kind, int n) {
    Points points = make_points_off_grid(kind, n);
    to_grid(points);
    return points;
}

// The points scaled from the unit box to `box` (side[a] along axis a).
inline Points scaled(Points points, const Box& box) {
    for (auto& p : points)
        for (int a = 0; a < 3; ++a) p[a] *= box.side[a];
    return points;
}

// Points in `box`, on its grid: lattices of about equal spacings along
// every axis, the other distributions stretched with the box.
inline Points make_points(const std::string& kind, int n, const Box& box) {
    if (box.cubic() && box.side[0] == 1.0) return make_points(kind, n);
    Points points = scaled(make_points_off_grid(kind, n, shape(box)), box);
    to_grid(points, box);
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

// A cell list over `box` whose base grid has about one particle per cell
// (times width_factor^-3).
template <class X>
neighbor_search::CellList<Exec> cell_list(const X& x, const Box& box = {}, double width_factor = 1.0) {
    neighbor_search::CellList<Exec> list;
    const int n = static_cast<int>(x.extent(0));
    const double width = box.cubic() ? box.side[0] * std::cbrt(1.0 / n) : std::cbrt(box.volume() / n);
    neighbor_search::configure(list, n, box, width_factor * width);
    neighbor_search::build(Exec{}, list, x);
    return list;
}

// The points renumbered in slot order, so that a cell list built on them has
// the identity order and slots name particles from then on.
inline Points in_slot_order(const Points& points, const Box& box = {}) {
    const auto order = to_host(cell_list(to_device(points), box).order);
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

    // Move every point by `distance` times the velocity, wrapped into the box
    // and kept on the grid. The field is periodic in the unit box; in another
    // box it is stretched with it (distances stay in box units).
    void advect(Points& x, double distance) const {
        for (auto& p : x) {
            const Point v = velocity(p);
            for (int a = 0; a < 3; ++a) p[a] = wrap(p[a] + distance * v[a]);
        }
        to_grid(x);
    }
    void advect(Points& x, double distance, const Box& box) const {
        if (box.cubic() && box.side[0] == 1.0) return advect(x, distance);
        for (auto& p : x) {
            const Point v = velocity({p[0] / box.side[0], p[1] / box.side[1], p[2] / box.side[2]});
            for (int a = 0; a < 3; ++a) p[a] = box.side[a] * wrap((p[a] + distance * v[a]) / box.side[a]);
        }
        to_grid(x, box);
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
inline int check_cells(const shell_voronoi::Cells<Memory>& cells, const Positions& positions, std::ostream& out,
                       const Box& box = {}) {
    constexpr double volume_tolerance = 1e-12, closure_tolerance = 1e-12, distance_tolerance = 1e-9;
    const auto status = to_host(cells.status);
    const auto face_count = to_host(cells.face_count);
    const auto owned_count = to_host(cells.owned_count);
    const auto start = to_host(cells.face_start);
    const auto neighbour = to_host(cells.neighbour);
    const auto centroid = to_host(cells.face_centroid);
    const auto volume = to_host(cells.volume);
    const auto closure = to_host(cells.closure);
    const auto work = to_host(cells.work);
    const auto x = to_host(positions);
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
            for (int a = 0; a < 3; ++a) c[a] = x(t, a) + centroid(start(t) + f, a);
            auto distance = [&](int k) {
                double d2 = 0;
                for (int a = 0; a < 3; ++a) {
                    const double d = nearest_image(x(k, a) - c[a], box.side[a]);
                    d2 += d * d;
                }
                return std::sqrt(d2);
            };
            const double own = distance(t), other = distance(neighbour(start(t) + f));
            double closest = own;
            for (int k = 0; k < n; ++k) closest = std::min(closest, distance(k));
            const double error = std::max(std::abs(own - other), own - closest) / own;
            worst = std::max(worst, error);
            ++checked;
            violations += error > distance_tolerance;
        }
    const double volume_error = std::abs(total_volume / box.volume() - 1.0);
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

// Clipping build of the particles of `list` that records
// face rings, for tet_mesh::from_rings and tet_start::build.
inline shell_voronoi::Cells<Memory> cells_with_rings(const neighbor_search::CellList<Exec>& list,
                                                     long long entries = 0) {
    auto cells = shell_voronoi::Cells<Memory>::with_rings(list.n, entries > 0 ? entries : 64LL * list.n);
    shell_voronoi::construct_warp(Exec{}, shell_voronoi::make_grid(list), cells);
    return cells;
}

inline shell_voronoi::Cells<Memory> cells_with_rings(const Positions& x, const Box& box = {}) {
    return cells_with_rings(cell_list(x, box));
}

// The exact starting tetrahedralization of x: clipping with
// rings, then tet_start::build.
inline tet_start::Stats start_mesh(const Positions& x, tet_mesh::Mesh<Memory>& mesh, const Box& box = {}) {
    const auto list = cell_list(x, box);
    auto cells = cells_with_rings(list);
    auto stats = tet_start::build(Exec{}, shell_voronoi::make_grid(list), cells, mesh);
    const long long needed = cells.needed();
    if (!stats.exact && needed > cells.capacity()) {  // the patches did not fit
        cells = cells_with_rings(list, needed);
        stats = tet_start::build(Exec{}, shell_voronoi::make_grid(list), cells, mesh);
    }
    return stats;
}

// The Voronoi cells of x by clipping, by particle: every particle's natural
// neighbours (from the owned faces), volume and build status.
struct Reference {
    NeighbourSets neighbours;
    std::vector<double> volume;
    std::vector<int> status;
};

// The cells of a clipping build, by particle.
inline Reference by_particle(const shell_voronoi::Cells<Memory>& cells) {
    const auto owned = to_host(cells.owned_count);
    const auto start = to_host(cells.face_start);
    const auto neighbour = to_host(cells.neighbour);
    const auto volume = to_host(cells.volume);
    const auto status = to_host(cells.status);
    const int n = static_cast<int>(owned.extent(0));
    Reference r{NeighbourSets(n), std::vector<double>(volume.data(), volume.data() + n),
                std::vector<int>(status.data(), status.data() + n)};
    for (int i = 0; i < n; ++i)
        for (int f = 0; f < owned(i); ++f) {
            const int j = neighbour(start(i) + f);
            r.neighbours[i].push_back(j);
            r.neighbours[j].push_back(i);
        }
    for (auto& s : r.neighbours) std::sort(s.begin(), s.end());
    return r;
}

inline Reference clipped(const Positions& x, const Box& box = {}) {
    const int n = static_cast<int>(x.extent(0));
    const auto list = cell_list(x, box);
    shell_voronoi::Cells<Memory> cells(n, false);
    shell_voronoi::construct_warp(Exec{}, shell_voronoi::make_grid(list), cells);
    return by_particle(cells);
}

// Cells of two builds that differ: particles with other neighbour sets, and
// the largest relative volume difference.
struct Difference {
    int neighbours = 0;
    double volume = 0;
};
inline Difference compare(const Reference& a, const Reference& b) {
    Difference d;
    for (std::size_t i = 0; i < a.volume.size(); ++i) {
        d.neighbours += a.neighbours[i] != b.neighbours[i];
        d.volume = std::max(d.volume, std::abs(a.volume[i] - b.volume[i]) / a.volume[i]);
    }
    return d;
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

using Mesh = voronoi::Mesh<Exec>;

// A host copy of the mesh's tetrahedra.
inline tet_flips::Mesh host_tetrahedra(const Mesh& mesh) {
    auto copy = [](const auto& v) {
        auto h = Kokkos::create_mirror(Kokkos::HostSpace{}, v);  // a copy even on host backends
        Kokkos::deep_copy(h, v);
        return h;
    };
    const auto& tets = mesh.tetrahedra();
    const auto& state = mesh.repair_state();
    return {copy(tets.vertex), copy(tets.adjacent), copy(state.alive), {copy(state.point.view), state.point.period},
            state.count};
}

// Inverted tetrahedra plus shared triangles that fail the local Delaunay
// test (each failing triangle counted from both sides); brute force over
// every tetrahedron, exact.
inline long long failing(const tet_flips::Mesh& m) {
    long long bad = 0;
    for (int t = 0; t < m.count; ++t) {
        if (!m.alive(t)) continue;
        const auto v = tet_flips::detail::tet(m, t);
        if (exact_predicates::orient(m.point, v.data()) <= 0) {
            ++bad;
            continue;
        }
        for (int k = 0; k < 4; ++k) {
            const int u = m.adjacent(t, k);
            bad += u >= 0 && tet_flips::detail::fails(m, t, u);
        }
    }
    return bad;
}

// The mesh's face area vectors on the host, rows [0, faces()) (packed on
// the device first: the face arrays hold more rows).
inline Kokkos::View<double* [3], Kokkos::LayoutRight, Kokkos::HostSpace> face_areas(const Mesh& mesh) {
    Kokkos::View<double* [3], Kokkos::LayoutRight, Memory> packed("face_areas", mesh.faces());
    const auto area = mesh.area();
    Kokkos::parallel_for(
        "support::face_areas", Kokkos::RangePolicy<Exec>(0, mesh.faces()), KOKKOS_LAMBDA(int f) {
            for (int a = 0; a < 3; ++a) packed(f, a) = area(f, a);
        });
    return to_host(packed);
}

// A mesh compared with an exact rebuild by clipping of the same positions
// (support::clipped): failing triangles and inverted tetrahedra (brute force
// over all tetrahedra, support::failing), cells with other Delaunay
// neighbours, clipping faces the mesh lacks, mesh faces the clipping lacks
// (and the largest of their areas, in spacings^2; clipping drops faces
// below rounding level, the mesh keeps every Delaunay edge), the largest
// relative volume difference and the total volume (box volumes).
struct Comparison {
    long long failing = 0, mismatched = 0, missing = 0, extra = 0, faces = 0, clipped_faces = 0;
    double extra_area = 0, volume = 0, total = 0;
    int worst_cell = -1;
    // Exact: no failing triangle, no missing face, extra faces only below
    // rounding level, volumes equal to 1e-10.
    bool pass() const { return failing == 0 && missing == 0 && extra_area < 1e-9 && volume < 1e-10; }
};

inline Comparison compare_with_rebuild(const Mesh& mesh, const Positions& x) {
    const int n = mesh.particles();
    const Box& box = mesh.sides();
    const double spacing = std::cbrt(box.volume() / n);
    Comparison c;
    const tet_flips::Mesh tets = host_tetrahedra(mesh);
    c.failing = failing(tets);
    const Reference exact = clipped(x, box);
    const NeighbourSets neighbours = delaunay_neighbours(tets, n);
    for (int i = 0; i < n; ++i) {
        c.mismatched += neighbours[i] != exact.neighbours[i];
        c.clipped_faces += static_cast<long long>(exact.neighbours[i].size());
        for (int j : exact.neighbours[i])
            c.missing += i < j && !std::binary_search(neighbours[i].begin(), neighbours[i].end(), j);
    }
    c.clipped_faces /= 2;
    const auto owner = to_host(mesh.owner());
    const auto neighbour = to_host(mesh.neighbour());
    const auto area = face_areas(mesh);
    c.faces = mesh.faces();
    for (int f = 0; f < c.faces; ++f) {
        const int i = owner(f), j = neighbour(f);
        if (std::binary_search(exact.neighbours[i].begin(), exact.neighbours[i].end(), j)) continue;
        ++c.extra;
        c.extra_area = std::max(c.extra_area, std::sqrt(area(f, 0) * area(f, 0) + area(f, 1) * area(f, 1) +
                                                        area(f, 2) * area(f, 2)) / (spacing * spacing));
    }
    const auto volume = to_host(mesh.volume());
    for (int i = 0; i < n; ++i) {
        c.total += volume(i) / box.volume();
        const double error = std::abs(volume(i) - exact.volume[i]) / exact.volume[i];
        if (exact.volume[i] > 0 && error > c.volume) {
            c.volume = error;
            c.worst_cell = i;
        }
    }
    return c;
}

inline std::ostream& operator<<(std::ostream& out, const Comparison& c) {
    return out << "failing=" << c.failing << " mismatched_cells=" << c.mismatched << " missing_faces=" << c.missing
               << " extra_faces=" << c.extra << " extra_max_area=" << c.extra_area << " faces=" << c.faces
               << " clipped_faces=" << c.clipped_faces << " total_volume=" << c.total
               << " max_relative_volume_difference=" << c.volume;
}

}  // namespace support
