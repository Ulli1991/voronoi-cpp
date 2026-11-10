#pragma once

// Point generators, reference builds and mesh comparisons shared by the tests and benchmarks.
// Exercises clipping (shell_voronoi), tet_start, tet_repair and voronoi::Mesh through their references.

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
// Box sides relative to the longest, so lattices get about equal spacings.
using Shape = std::array<double, 3>;
inline Shape shape(const Box& box) {
    const double longest = std::max({box.side[0], box.side[1], box.side[2]});
    return {box.side[0] / longest, box.side[1] / longest, box.side[2] / longest};
}
// Lattice cells per axis for about `cells` cells of equal spacing in a box of shape s, at least `fewest` per axis.
// Used by: make_points_off_grid and make_degenerate (non-cubic lattices).
inline std::array<int, 3> lattice_sides(double cells, const Shape& s, int fewest) {
    const double per_unit = std::cbrt(cells / (s[0] * s[1] * s[2]));
    return {std::max(fewest, static_cast<int>(std::lround(per_unit * s[0]))),
            std::max(fewest, static_cast<int>(std::lround(per_unit * s[1]))),
            std::max(fewest, static_cast<int>(std::lround(per_unit * s[2])))};
}

inline double wrap(double x) { return x - std::floor(x); }

// Snaps points to the unit-box grid of the exact predicates (2³⁰ steps), rounding down.
// Inputs must be on the grid: topology uses snapped positions, geometry the raw ones.
// Used by: make_points, Flow::advect, mesh_cases_test.
inline void to_grid(Points& points) {
    for (auto& p : points)
        for (double& x : p) x = std::floor(x * exact_predicates::scale) / exact_predicates::scale;
}
inline double nearest_image(double d, double side = 1.0) { return d - side * std::nearbyint(d / side); }

// The same on the grid of `box` (exact_predicates::snap), as voronoi::Mesh does.
// Used by: make_points(kind, n, box), Flow::advect(box).
inline void to_grid(Points& points, const Box& box) {
    for (auto& p : points)
        for (int a = 0; a < 3; ++a) p[a] = exact_predicates::snap(p[a], box, a) * box.step;
}

// Host copy of a device view. Used by: every test that reads results back.
template <class View>
auto to_host(const View& v) {
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
}

// Host copy of the first `rows` rows of v: a mesh array has capacity past what the mesh writes, and copying unwritten
// memory is what compute-sanitizer's initcheck reports (#47). Used by: tests reading mesh arrays.
template <class View>
auto to_host(const View& v, std::size_t rows) {
    Kokkos::View<typename View::non_const_data_type, typename View::memory_space> used(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("used_rows")), rows);
    const auto range = std::make_pair(std::size_t(0), rows);
    if constexpr (View::rank() == 1)
        Kokkos::deep_copy(used, Kokkos::subview(v, range));
    else
        Kokkos::deep_copy(used, Kokkos::subview(v, range, Kokkos::ALL));
    return to_host(used);
}

// The new starts of cell t's entries [start(t), start(t) + length[t]) of a clipping pool, packed in cell order (n + 1,
// on the device); the rest of a pool is capacity no build writes (#47). Used by: pack_to_host.
inline Kokkos::View<long long*, Memory> packed_starts(const std::vector<long long>& length,
                                                      std::vector<long long>& packed) {
    const std::size_t n = length.size();
    packed.assign(n + 1, 0);
    for (std::size_t t = 0; t < n; ++t) packed[t + 1] = packed[t] + length[t];
    Kokkos::View<long long*, Memory> offset("packed_offset", n + 1);
    Kokkos::deep_copy(offset, Kokkos::View<const long long*, Kokkos::HostSpace>(packed.data(), n + 1));
    return offset;
}

// A pool of one value per entry, packed on the host (see packed_starts). Used by: owned_faces, ring_entries.
template <class Pool, class Start>
Kokkos::View<int*, Kokkos::HostSpace> pack_to_host(const Pool& pool, const Start& start,
                                                   const std::vector<long long>& length,
                                                   std::vector<long long>& packed) {
    const auto offset = packed_starts(length, packed);
    Kokkos::View<int*, Memory> out(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("packed_pool")),
                                   packed.back());
    Kokkos::parallel_for(
        "support::pack", Kokkos::RangePolicy<Exec>(0, static_cast<int>(length.size())), KOKKOS_LAMBDA(int t) {
            for (long long f = 0; f < offset(t + 1) - offset(t); ++f) out(offset(t) + f) = pool(start(t) + f);
        });
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
}

// A pool of vectors, packed on the host (see packed_starts). Used by: owned_faces.
template <class Pool, class Start>
Kokkos::View<double* [3], Kokkos::LayoutRight, Kokkos::HostSpace> pack3_to_host(const Pool& pool, const Start& start,
                                                                                const std::vector<long long>& length,
                                                                                std::vector<long long>& packed) {
    const auto offset = packed_starts(length, packed);
    Kokkos::View<double* [3], Kokkos::LayoutRight, Memory> out(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("packed_pool")), packed.back());
    Kokkos::parallel_for(
        "support::pack3", Kokkos::RangePolicy<Exec>(0, static_cast<int>(length.size())), KOKKOS_LAMBDA(int t) {
            for (long long f = 0; f < offset(t + 1) - offset(t); ++f)
                for (int a = 0; a < 3; ++a) out(offset(t) + f, a) = pool(start(t) + f, a);
        });
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, out);
}

// The owned faces of a clipping build on the host: cell t's are [start[t], start[t + 1]) of the packed arrays.
// Filled by owned_faces. Used by: check_cells, by_particle, voronoi_mesh_test.
struct OwnedFaces {
    std::vector<long long> start;
    Kokkos::View<int*, Kokkos::HostSpace> neighbour;
    Kokkos::View<double* [3], Kokkos::LayoutRight, Kokkos::HostSpace> area, centroid;
};

inline OwnedFaces owned_faces(const shell_voronoi::Cells<Memory>& cells) {
    const auto owned = to_host(cells.owned_count);
    const std::vector<long long> length(owned.data(), owned.data() + owned.extent(0));
    OwnedFaces f;
    f.neighbour = pack_to_host(cells.neighbour, cells.face_start, length, f.start);
    f.area = pack3_to_host(cells.area, cells.face_start, length, f.start);
    f.centroid = pack3_to_host(cells.face_centroid, cells.face_start, length, f.start);
    return f;
}

// The face rings of a clipping build on the host, cell t's at [start[t], start[t + 1]); see owned_faces.
// Used by: mesh_test.
inline Kokkos::View<int*, Kokkos::HostSpace> ring_entries(const shell_voronoi::Cells<Memory>& cells,
                                                          std::vector<long long>& start) {
    const auto owned = cells.owned_count;
    const auto ring_start = cells.ring_start;
    const auto rings = cells.rings;
    Kokkos::View<long long*, Memory> entries("ring_entries", owned.extent(0));
    Kokkos::parallel_for(
        "support::ring_lengths", Kokkos::RangePolicy<Exec>(0, static_cast<int>(owned.extent(0))), KOKKOS_LAMBDA(int t) {
            long long length = 0;
            for (int f = 0; f < owned(t); ++f) length += 2 + rings(ring_start(t) + length + 1);
            entries(t) = length;
        });
    const auto length = to_host(entries);
    return pack_to_host(rings, ring_start, std::vector<long long>(length.data(), length.data() + length.extent(0)),
                        start);
}

// Degenerate layouts "<layout>@<perturbation>" (perturbation in spacings, uniform in each coordinate):
// sc, bcc, fcc lattices; shells of 12 points (12 cells meet at the centre); planes; pairs with a twin.
// Used by: make_points_off_grid, for the "sc@..." / "shells@..." kinds of start_test, repair_test, voronoi_mesh_test.
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

// Points in the unit box of the given `kind`: uniform, jittered, lattice, wave, clustered, halo or a degenerate layout.
// Used by: make_points, scaled; directly by voronoi_mesh_test (off-grid, snapped by the mesh).
// Lattices round n to a cube, or with `shape` to equal spacings once scaled to that box.
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
        // Four halos of density 1 / (r² + c²) out to r = 0.2, c = 0.002, over a 10 % uniform background.
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
// Used by: clipping_test, mesh_test, start_test, repair_test, mesh2d_test, mesh_cases_test, benchmarks.
inline Points make_points(const std::string& kind, int n) {
    Points points = make_points_off_grid(kind, n);
    to_grid(points);
    return points;
}

// The points scaled from the unit box to `box` (side[a] along axis a).
// Used by: make_points(kind, n, box), and by tests that build their own positions in a box.
inline Points scaled(Points points, const Box& box) {
    for (auto& p : points)
        for (int a = 0; a < 3; ++a) p[a] *= box.side[a];
    return points;
}

// Points in `box`, on its grid: lattices of about equal spacings along
// every axis, the other distributions stretched with the box.
// Used by: start_test, repair_test, voronoi_mesh_test, mesh_cases_test (non-cubic boxes).
inline Points make_points(const std::string& kind, int n, const Box& box) {
    if (box.cubic() && box.side[0] == 1.0) return make_points(kind, n);
    Points points = scaled(make_points_off_grid(kind, n, shape(box)), box);
    to_grid(points, box);
    return points;
}

// Uploads points to a device View. Used by: every test that feeds the library.
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
// Used by: clipping_test (coarser lists via width_factor), mesh_test, voronoi_mesh_test, mesh_cases_test.
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
// Used by: mesh_test, start_test, distortion.hpp.
inline Points in_slot_order(const Points& points, const Box& box = {}) {
    const auto order = to_host(cell_list(to_device(points), box).order);
    Points sorted(points.size());
    for (std::size_t t = 0; t < points.size(); ++t) sorted[t] = points[order(t)];
    return sorted;
}

// A smooth periodic velocity field of unit-order speed: six random modes.
// advect() moves points by it to make the steps that repair_test, voronoi_mesh_test and mesh_cases_test feed
// to voronoi::Mesh::update.
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

    // Moves every point by `distance` times the velocity, wrapped and kept on the grid.
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

// Median wall time in ms of f over `repeats` runs after one warm-up. Used by: voronoi_bench only.
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

// Checks a clipping build on its own: cells certified, volumes positive and summing to the box,
// area vectors closing, faces seen from both sides, owned face centroids equidistant from their
// two generators with none closer (brute force on a sample). Prints a summary; returns the failed checks.
// Exercises shell_voronoi clipping output. Used by: clipping_test, mesh2d_test, voronoi_bench.
inline int check_cells(const shell_voronoi::Cells<Memory>& cells, const Positions& positions, std::ostream& out,
                       const Box& box = {}) {
    constexpr double volume_tolerance = 1e-12, closure_tolerance = 1e-12, distance_tolerance = 1e-9;
    const auto status = to_host(cells.status);
    const auto face_count = to_host(cells.face_count);
    const auto owned_count = to_host(cells.owned_count);
    const OwnedFaces packed = owned_faces(cells);
    const auto& start = packed.start;
    const auto& neighbour = packed.neighbour;
    const auto& centroid = packed.centroid;
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
            for (int a = 0; a < 3; ++a) c[a] = x(t, a) + centroid(start[t] + f, a);
            auto distance = [&](int k) {
                double d2 = 0;
                for (int a = 0; a < 3; ++a) {
                    const double d = nearest_image(x(k, a) - c[a], box.side[a]);
                    d2 += d * d;
                }
                return std::sqrt(d2);
            };
            const double own = distance(t), other = distance(neighbour(start[t] + f));
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

// Clipping build of the particles of `list` that records face rings, the input of
// tet_mesh::from_rings and tet_start::build. Used by: mesh_test, mesh_cases_test, start_mesh, voronoi_bench.
inline shell_voronoi::Cells<Memory> cells_with_rings(const neighbor_search::CellList<Exec>& list,
                                                     long long entries = 0) {
    auto cells = shell_voronoi::Cells<Memory>::with_rings(list.n, entries > 0 ? entries : 64LL * list.n);
    shell_voronoi::construct_warp(Exec{}, shell_voronoi::make_grid(list), cells);
    return cells;
}

inline shell_voronoi::Cells<Memory> cells_with_rings(const Positions& x, const Box& box = {}) {
    return cells_with_rings(cell_list(x, box));
}

// The exact starting tetrahedralization of x: clipping with rings, then tet_start::build (once more with a
// bigger face pool if the patches did not fit). Returns tet_start's stats. Used by: start_test, mesh_cases_test.
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
// Used by: clipping_test (by_particle, compare), compare_with_rebuild (clipped), mesh_check.hpp.
struct Reference {
    NeighbourSets neighbours;
    std::vector<double> volume;
    std::vector<int> status;
};

// The cells of a clipping build, by particle.
inline Reference by_particle(const shell_voronoi::Cells<Memory>& cells) {
    const auto owned = to_host(cells.owned_count);
    const OwnedFaces packed = owned_faces(cells);
    const auto& start = packed.start;
    const auto& neighbour = packed.neighbour;
    const auto volume = to_host(cells.volume);
    const auto status = to_host(cells.status);
    const int n = static_cast<int>(owned.extent(0));
    Reference r{NeighbourSets(n), std::vector<double>(volume.data(), volume.data() + n),
                std::vector<int>(status.data(), status.data() + n)};
    for (int i = 0; i < n; ++i)
        for (int f = 0; f < owned(i); ++f) {
            const int j = neighbour(start[i] + f);
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
// the largest relative volume difference. Used by: clipping_test (kernel and list comparisons).
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
// zero area. Used by: compare_with_rebuild.
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

using Mesh = voronoi::Mesh<3, Exec>;

// A host copy of the mesh's tetrahedra. Used by: compare_with_rebuild, mesh_check.hpp, voronoi_bench.
template <class M>  // any voronoi::Mesh<3, Exec, ...>
tet_flips::Mesh host_tetrahedra(const M& mesh) {
    auto copy = [](const auto& v) {
        auto h = Kokkos::create_mirror(Kokkos::HostSpace{}, v);
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
// every tetrahedron, exact. Used by: compare_with_rebuild (so repair_test, voronoi_mesh_test).
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
// Used by: voronoi_mesh_test, mesh_cases_test.
template <class M>
Kokkos::View<double* [3], Kokkos::LayoutRight, Kokkos::HostSpace> face_areas(const M& mesh) {
    Kokkos::View<double* [3], Kokkos::LayoutRight, Memory> packed("face_areas", mesh.faces());
    const auto area = mesh.area();
    Kokkos::parallel_for(
        "support::face_areas", Kokkos::RangePolicy<Exec>(0, mesh.faces()), KOKKOS_LAMBDA(int f) {
            for (int a = 0; a < 3; ++a) packed(f, a) = area(f, a);
        });
    return to_host(packed);
}

// A mesh compared with an exact clipping rebuild of the same positions (support::clipped):
// failing (support::failing), mismatched (other Delaunay neighbours), missing (clipping faces the
// mesh lacks), extra (mesh faces clipping lacks, extra_area their largest area in spacings²;
// clipping drops faces below rounding level), volume (largest relative difference), total (box
// volumes), wrong_lists (per-cell list entries out of order or one-sided, after update_cells()).
// Filled by compare_with_rebuild. Used by: repair_test, voronoi_mesh_test, voronoi_bench.
struct Comparison {
    long long failing = 0, mismatched = 0, missing = 0, extra = 0, faces = 0, clipped_faces = 0, wrong_lists = 0;
    double extra_area = 0, volume = 0, total = 0;
    int worst_cell = -1;
    // Exact: no failing triangle, no missing face, extra faces only below
    // rounding level, volumes equal to 1e-10.
    bool pass() const { return failing == 0 && missing == 0 && extra_area < 1e-9 && volume < 1e-10 && wrong_lists == 0; }
};

// Rebuilds x exactly by clipping and compares the mesh's connectivity, faces and volumes with it.
template <class M>
Comparison compare_with_rebuild(const M& mesh, const Positions& x) {
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
    const auto owner = to_host(mesh.owner(), mesh.faces());
    const auto neighbour = to_host(mesh.neighbour(), mesh.faces());
    const auto area = face_areas(mesh);
    c.faces = mesh.faces();
    for (int f = 0; f < c.faces; ++f) {
        const int i = owner(f), j = neighbour(f);
        if (std::binary_search(exact.neighbours[i].begin(), exact.neighbours[i].end(), j)) continue;
        ++c.extra;
        c.extra_area = std::max(c.extra_area, std::sqrt(area(f, 0) * area(f, 0) + area(f, 1) * area(f, 1) +
                                                        area(f, 2) * area(f, 2)) / (spacing * spacing));
    }
    const auto start = to_host(mesh.cell_face_start());
    const auto list = to_host(mesh.cell_face(), mesh.cells_current() ? 2 * mesh.faces() : 0);
    c.wrong_lists = mesh.cells_current() ? std::abs(start(n) - 2 * mesh.faces()) : 0;
    for (int i = 0; i < n; ++i)
        for (int a = start(i), last = -1; mesh.cells_current() && a < start(i + 1) && c.wrong_lists == 0; ++a) {
            const int f = list(a), other = owner(f) == i ? neighbour(f) : owner(f);
            c.wrong_lists += (owner(f) != i && neighbour(f) != i) || other <= last;
            last = other;
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
               << " clipped_faces=" << c.clipped_faces << " wrong_lists=" << c.wrong_lists << " total_volume=" << c.total
               << " max_relative_volume_difference=" << c.volume;
}

}  // namespace support
