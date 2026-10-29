// Validate and time the fused natural-neighbour construction (shell_voronoi).
//
// Usage: shell_voronoi_check [n=32768] [uniform|jittered|lattice|wave|clustered|<layout>@<eps>]
//                            [variants=all] [repeats=5] [width_factor=1] [displacement=0]
// Variants are thread-, warp- and lean- with -double or -float; the
// argument selects every variant whose name contains it (e.g. warp, double).
//
// Checks, independent of any other construction: every cell certified, cell
// volumes positive and summing to the box volume, face area vectors closing
// each cell, every face seen from both sides (total faces = 2 x owned), and
// each owned face centroid equidistant from its two generators with no other
// generator closer. Exits non-zero if a check fails.

#include "neighbor_search.hpp"
#include "face_geometry.hpp"
#include "tet_flips.hpp"
#include "tet_mesh.hpp"
#include "shell_voronoi_lean.hpp"
#include "shell_voronoi_warp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Memory = Exec::memory_space;
using Points = std::vector<std::array<double, 3>>;

double wrap(double x) { return x - std::floor(x); }
double nearest_image(double d) { return d - std::nearbyint(d); }

// Degenerate layouts, written "<layout>@<perturbation>" with the perturbation
// in spacings (each coordinate moved uniformly within +-perturbation):
//   sc, bcc, fcc  cubic lattices (8, 4 and 6 cells meeting at vertices)
//   shells        icosahedral shells of 12 points without a centre point,
//                 so 12 cells meet at each shell centre
//   planes        random points on exactly spaced planes
//   pairs         random points, each with a twin `perturbation` away
Points make_degenerate(const std::string& layout, double perturbation, int n) {
    std::mt19937_64 rng(20261029);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    Points points;
    auto basis_lattice = [&](const std::vector<std::array<double, 3>>& basis) {
        const int side = std::max(3, static_cast<int>(std::lround(std::cbrt(double(n) / basis.size()))));
        for (int i = 0; i < side; ++i)
            for (int j = 0; j < side; ++j)
                for (int k = 0; k < side; ++k)
                    for (const auto& b : basis)
                        points.push_back({(i + b[0]) / side, (j + b[1]) / side, (k + b[2]) / side});
    };
    if (layout == "sc") {
        basis_lattice({{0.5, 0.5, 0.5}});
    } else if (layout == "bcc") {
        basis_lattice({{0.25, 0.25, 0.25}, {0.75, 0.75, 0.75}});
    } else if (layout == "fcc") {
        basis_lattice({{0.25, 0.25, 0.25}, {0.75, 0.75, 0.25}, {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75}});
    } else if (layout == "shells") {
        const double g = (1.0 + std::sqrt(5.0)) / 2.0, r = 0.3 / std::sqrt(1.0 + g * g);
        std::vector<std::array<double, 3>> basis;
        for (int s1 : {-1, 1})
            for (int s2 : {-1, 1}) {
                basis.push_back({0.5, 0.5 + s1 * r, 0.5 + s2 * g * r});
                basis.push_back({0.5 + s1 * r, 0.5 + s2 * g * r, 0.5});
                basis.push_back({0.5 + s2 * g * r, 0.5, 0.5 + s1 * r});
            }
        basis_lattice(basis);
    } else if (layout == "planes") {
        const int layers = std::max(3, static_cast<int>(std::lround(std::cbrt(n))));
        for (int i = 0; i < n; ++i)
            points.push_back({unit(rng), unit(rng), (i % layers + 0.5) / layers});
    } else if (layout == "pairs") {
        const double spacing = std::cbrt(2.0 / n);
        std::uniform_real_distribution<double> direction(-1.0, 1.0);
        for (int i = 0; i < n / 2; ++i) {
            const std::array<double, 3> a{unit(rng), unit(rng), unit(rng)};
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

Points make_points(const std::string& kind, int n) {
    if (const auto at = kind.find('@'); at != std::string::npos)
        return make_degenerate(kind.substr(0, at), std::stod(kind.substr(at + 1)), n);
    std::mt19937_64 rng(20261029);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    Points points;
    if (kind == "jittered" || kind == "lattice" || kind == "wave") {
        const int side = static_cast<int>(std::lround(std::cbrt(n)));
        const double spacing = 1.0 / side;
        // "wave": the near-degenerate lattice of an acoustic-wave start.
        const double jitter = kind == "lattice" ? 0.0 : kind == "wave" ? 1e-5 * spacing : 0.1 * spacing;
        std::uniform_real_distribution<double> shift(-jitter, jitter);
        for (int i = 0; i < side; ++i)
            for (int j = 0; j < side; ++j)
                for (int k = 0; k < side; ++k)
                    points.push_back({wrap((i + 0.5) * spacing + shift(rng)),
                                      wrap((j + 0.5) * spacing + shift(rng)),
                                      wrap((k + 0.5) * spacing + shift(rng))});
    } else if (kind == "clustered") {
        // Eight Gaussian blobs over a uniform background: about 1:100 density.
        std::normal_distribution<double> blob(0.0, 0.04);
        std::array<std::array<double, 3>, 8> centres;
        for (auto& c : centres) c = {unit(rng), unit(rng), unit(rng)};
        for (int i = 0; i < n; ++i) {
            if (i % 5 == 0) {
                points.push_back({unit(rng), unit(rng), unit(rng)});
            } else {
                const auto& c = centres[i % 8];
                points.push_back({wrap(c[0] + blob(rng)), wrap(c[1] + blob(rng)), wrap(c[2] + blob(rng))});
            }
        }
    } else {
        for (int i = 0; i < n; ++i) points.push_back({unit(rng), unit(rng), unit(rng)});
    }
    return points;
}

double median(std::vector<double> values) {
    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
    return values[values.size() / 2];
}

template <class F>
double median_ms(int repeats, const F& f) {
    f();  // warm-up
    std::vector<double> times;
    for (int r = 0; r < repeats; ++r) {
        Kokkos::Timer timer;
        f();
        times.push_back(timer.seconds() * 1e3);
    }
    return median(times);
}

struct HostCells {
    decltype(Kokkos::create_mirror_view(Kokkos::View<int*, Memory>())) status, face_count, owned_count;
    decltype(Kokkos::create_mirror_view(Kokkos::View<int**, Memory>())) neighbour;
    decltype(Kokkos::create_mirror_view(Kokkos::View<double** [3], Memory>())) area, face_centroid;
    decltype(Kokkos::create_mirror_view(Kokkos::View<double*, Memory>())) volume, closure;
    decltype(Kokkos::create_mirror_view(Kokkos::View<int* [3], Memory>())) work;
};

HostCells to_host(const shell_voronoi::Cells<Memory>& c) {
    auto copy = [](const auto& v) { return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v); };
    return {copy(c.status), copy(c.face_count), copy(c.owned_count), copy(c.neighbour),
            copy(c.area),   copy(c.face_centroid), copy(c.volume), copy(c.closure), copy(c.work)};
}

// Returns the number of failed checks and prints a one-line summary.
int validate(const HostCells& c, const std::vector<std::array<double, 3>>& slot_points,
             double volume_tolerance, double closure_tolerance, double distance_tolerance) {
    const int n = static_cast<int>(c.status.extent(0));
    std::array<int, shell_voronoi::status_count> statuses{};
    double total_volume = 0, min_volume = 1e300, max_closure = 0;
    long long faces = 0, owned = 0;
    std::array<double, 3> work{};
    for (int t = 0; t < n; ++t) {
        for (int k = 0; k < 3; ++k) work[k] += c.work(t, k);
        ++statuses[c.status(t)];
        total_volume += c.volume(t);
        min_volume = std::min(min_volume, c.volume(t));
        max_closure = std::max(max_closure, c.closure(t));
        faces += c.face_count(t);
        owned += c.owned_count(t);
    }

    // Voronoi property of owned face centroids against all generators; cells
    // are sampled evenly to bound the O(n^2) cost at about 2.5e8 distances.
    const int sampled = static_cast<int>(std::clamp(2.5e8 / n, 16.0, double(n)));
    const int stride = std::max(1, n / sampled);
    long long checked = 0, violations = 0;
    double worst = 0;
    for (int t = 0; t < n; t += stride) {
        for (int f = 0; f < c.owned_count(t); ++f) {
            std::array<double, 3> x;
            for (int a = 0; a < 3; ++a) x[a] = slot_points[t][a] + c.face_centroid(t, f, a);
            auto distance = [&](int k) {
                double d2 = 0;
                for (int a = 0; a < 3; ++a) {
                    const double d = nearest_image(slot_points[k][a] - x[a]);
                    d2 += d * d;
                }
                return std::sqrt(d2);
            };
            const double own = distance(t), other = distance(c.neighbour(t, f));
            double closest = own;
            for (int k = 0; k < n; ++k) closest = std::min(closest, distance(k));
            const double error = std::max(std::abs(own - other), own - closest) / own;
            worst = std::max(worst, error);
            ++checked;
            if (error > distance_tolerance) ++violations;
        }
    }

    const double volume_error = std::abs(total_volume - 1.0);
    int failures = 0;
    failures += statuses[shell_voronoi::ok] != n;
    failures += volume_error > volume_tolerance;
    failures += !(min_volume > 0);
    failures += max_closure > closure_tolerance;
    failures += faces != 2 * owned;
    failures += violations > 0;
    std::cout << "  certified=" << statuses[shell_voronoi::ok] << "/" << n << " statuses=";
    for (int s : statuses) std::cout << s << ",";
    std::cout << " volume_error=" << volume_error << " min_volume=" << min_volume
              << " max_closure=" << max_closure << " faces_per_cell=" << double(faces) / n
              << " faces=2*owned:" << (faces == 2 * owned ? "yes" : "NO") << " centroid_checks=" << checked
              << " worst_centroid_error=" << worst << " violations=" << violations
              << " per_cell(rings,clipped,cuts)=" << work[0] / n << "," << work[1] / n << "," << work[2] / n
              << (failures ? "  FAIL" : "  PASS") << "\n";
    return failures;
}

// Inverse of the grid's slot -> particle order.
Kokkos::View<int*, Memory> slot_of_particle(const Exec& exec, const Kokkos::View<int*, Memory>& order) {
    Kokkos::View<int*, Memory> rank("rank", order.extent(0));
    Kokkos::parallel_for("rank", Kokkos::RangePolicy<Exec>(exec, 0, order.extent(0)),
                         KOKKOS_LAMBDA(int q) { rank(order(q)) = q; });
    return rank;
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    int failures = 0;
    {
        const int requested = argc > 1 ? std::atoi(argv[1]) : 32768;
        const std::string kind = argc > 2 ? argv[2] : "uniform";
        const std::string selection = argc > 3 ? argv[3] : "all";
        const int repeats = argc > 4 ? std::max(1, std::atoi(argv[4])) : 5;
        const double width_factor = argc > 5 ? std::atof(argv[5]) : 1.0;
        const Exec exec{};

        const Points points = make_points(kind, requested);
        const int n = static_cast<int>(points.size());
        Kokkos::View<double* [3], Memory> positions("positions", n);
        Kokkos::View<double*, Memory> unused_radii("unused_radii", n);
        {
            auto h = Kokkos::create_mirror_view(positions);
            for (int i = 0; i < n; ++i)
                for (int a = 0; a < 3; ++a) h(i, a) = points[i][a];
            Kokkos::deep_copy(positions, h);
        }

        // About one particle per grid cell; only the cell index is built.
        neighbor_search::DenseCellSearch<Exec> search;
        const double spacing = std::cbrt(1.0 / n);
        // SHELL_MORTON=1: cells stored in Morton order, so consecutive slots form compact bricks.
        search.morton_cells = std::getenv("SHELL_MORTON") != nullptr;
        neighbor_search::configure(search, n, Kokkos::Array<double, 3>{1.0, 1.0, 1.0}, width_factor * spacing,
                                   true, 1, false);
        const double build_ms = median_ms(repeats, [&] {
            neighbor_search::build(exec, search, positions, unused_radii);
            exec.fence();
        });
        const shell_voronoi::Grid<Memory> grid{search.positions, search.cell_offsets, n, search.nx, search.hx, 1.0,
                                               search.morton_cells ? search.cell_rank : Kokkos::View<int*, Memory>{}};

        std::vector<std::array<double, 3>> slot_points(n);
        {
            auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, search.positions);
            for (int t = 0; t < n; ++t) slot_points[t] = {h(t, 0), h(t, 1), h(t, 2)};
        }

        std::cout << "# backend=" << Exec::name() << " n=" << n << " distribution=" << kind
                  << " grid=" << search.nx << "^3 repeats=" << repeats << "\n";
        const auto stencil = shell_voronoi::lean::make_stencil(exec, grid.cells_per_side, grid.cell_width);
        std::vector<double> reference_volumes;
        for (const std::string variant : {"thread-double", "thread-float", "warp-double", "warp-float",
                                          "lean-double", "lean-float"}) {
            if (selection != "all" && variant.find(selection) == std::string::npos) continue;
            shell_voronoi::Cells<Memory> cells(n, true);
            const double construct_ms = median_ms(repeats, [&] {
                if (variant == "thread-double") shell_voronoi::construct<double>(exec, grid, cells);
                if (variant == "thread-float") shell_voronoi::construct<float>(exec, grid, cells);
                if (variant == "warp-double") shell_voronoi::construct_warp<double>(exec, grid, cells);
                if (variant == "warp-float") shell_voronoi::construct_warp<float>(exec, grid, cells);
                if (variant == "lean-double") shell_voronoi::construct_lean<double>(exec, grid, cells, stencil);
                if (variant == "lean-float") shell_voronoi::construct_lean<float>(exec, grid, cells, stencil);
                exec.fence();
            });
            std::cout << variant << ": build_ms=" << build_ms << " construct_ms=" << construct_ms
                      << " construct_Mcells_s=" << n / construct_ms / 1e3
                      << " pipeline_Mcells_s=" << n / (build_ms + construct_ms) / 1e3 << "\n";
            const HostCells host = to_host(cells);
            const bool is_double = variant.find("double") != std::string::npos;
            failures += validate(host, slot_points, is_double ? 1e-12 : 1e-5, is_double ? 1e-12 : 1e-5,
                                 is_double ? 1e-9 : 1e-4);
            if (reference_volumes.empty()) {
                reference_volumes.assign(host.volume.data(), host.volume.data() + n);
            } else {
                double worst = 0;
                for (int t = 0; t < n; ++t)
                    worst = std::max(worst, std::abs(host.volume(t) - reference_volumes[t]) / reference_volumes[t]);
                std::cout << "  max_relative_volume_difference_to_first_variant=" << worst << "\n";
            }
        }

        // Warm rebuild: build once with neighbour lists, move every particle
        // by up to `displacement` spacings per axis, rebuild the grid (new
        // slots), then rebuild the cells starting from the old neighbours. The
        // result must match a cold build at the moved positions.
        const double displacement = argc > 6 ? std::atof(argv[6]) : 0.0;
        auto warm_experiment = [&]<class Real>(const char* label) {
            const int capacity = 40;
            auto restore = [&](double amplitude) {
                std::mt19937_64 rng(7);
                std::uniform_real_distribution<double> step(-amplitude * spacing, amplitude * spacing);
                auto h = Kokkos::create_mirror_view(positions);
                for (int i = 0; i < n; ++i)
                    for (int a = 0; a < 3; ++a) h(i, a) = wrap(points[i][a] + (amplitude > 0 ? step(rng) : 0.0));
                Kokkos::deep_copy(positions, h);
                neighbor_search::build(exec, search, positions, unused_radii);
                exec.fence();
            };

            restore(0.0);
            shell_voronoi::Cells<Memory> before(n, false);
            before.all_neighbours = Kokkos::View<int**, Memory>("all_neighbours", n, capacity);
            shell_voronoi::construct_lean<Real>(exec, grid, before, stencil);
            Kokkos::View<int**, Memory> seed_ids("seed_ids", n, capacity);
            Kokkos::View<int*, Memory> seed_count("seed_count", n);
            {
                auto order = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, search.order);
                auto lists = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, before.all_neighbours);
                auto counts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, before.face_count);
                auto ids = Kokkos::create_mirror_view(seed_ids);
                auto count = Kokkos::create_mirror_view(seed_count);
                for (int t = 0; t < n; ++t) {
                    const int particle = order(t);
                    count(particle) = counts(t);
                    for (int j = 0; j < counts(t); ++j) ids(particle, j) = order(lists(t, j));
                }
                Kokkos::deep_copy(seed_ids, ids);
                Kokkos::deep_copy(seed_count, count);
            }

            restore(displacement);
            const Kokkos::View<int*, Memory> rank = slot_of_particle(exec, search.order);
            const shell_voronoi::lean::Seeds<Memory> seeds{seed_ids, seed_count, search.order, rank};
            {
                auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, search.positions);
                for (int t = 0; t < n; ++t) slot_points[t] = {h(t, 0), h(t, 1), h(t, 2)};
            }

            // One output set at a time keeps 7M-particle runs within 40 GB.
            std::vector<double> cold_volumes(n);
            double cold_ms = 0, uncertified_ms = 0;
            {
                shell_voronoi::Cells<Memory> cold(n, false);
                cold_ms = median_ms(repeats, [&] {
                    shell_voronoi::construct_lean<Real>(exec, grid, cold, stencil);
                    exec.fence();
                });
                auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cold.volume);
                for (int t = 0; t < n; ++t) cold_volumes[t] = h(t);
                // Seeds only, no certification walk: not a valid mesh, but
                // the gap to the certified time bounds what a faster spatial
                // structure could save.
                uncertified_ms = median_ms(repeats, [&] {
                    shell_voronoi::construct_lean<Real>(exec, grid, cold, stencil, seeds, false);
                    exec.fence();
                });
            }
            shell_voronoi::Cells<Memory> warm(n, true);
            warm.all_neighbours = Kokkos::View<int**, Memory>("warm_neighbours", n, capacity);
            const double warm_ms = median_ms(repeats, [&] {
                shell_voronoi::construct_lean<Real>(exec, grid, warm, stencil, seeds);
                exec.fence();
            });
            std::cout << label << " warm(displacement=" << displacement << "): warm_ms=" << warm_ms
                      << " warm_Mcells_s=" << n / warm_ms / 1e3 << " seeds_only_ms=" << uncertified_ms
                      << " seeds_only_Mcells_s=" << n / uncertified_ms / 1e3 << " cold_ms=" << cold_ms
                      << " cold_Mcells_s=" << n / cold_ms / 1e3 << " grid_build_ms=" << build_ms << "\n";
            const HostCells host = to_host(warm);
            const bool is_double = std::is_same_v<Real, double>;
            failures += validate(host, slot_points, is_double ? 1e-12 : 1e-5, is_double ? 1e-12 : 1e-5,
                                 is_double ? 1e-9 : 1e-4);
            double worst = 0;
            for (int t = 0; t < n; ++t)
                worst = std::max(worst, std::abs(host.volume(t) - cold_volumes[t]) / cold_volumes[t]);
            std::cout << "  warm_vs_cold_max_relative_volume_difference=" << worst << "\n";
        };
        // Geometry from topology: rings from one exact build, then only the
        // face-parallel geometry pass is timed and checked against the build.
        // CPU replay (test only, brute force): clip every cell nearest-first
        // from all particles within 6 spacings and count how many plane tests
        // cheaper bounds on the current cell would have rejected.
        if (selection == "cullstats") {
            using shell_voronoi::Vec3;
            const int side = search.nx;
            const double width = search.hx;
            auto offsets = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, search.cell_offsets);
            const int reach = std::min((side - 1) / 2, static_cast<int>(std::ceil(6 * spacing / width)));
            const double half_width = std::min(0.25, ((side - 1) / 2 + 1) * width);
            std::vector<Vec3<double>> dirs;  // 26 unit directions: faces, edges, corners of a cube
            for (int x = -1; x <= 1; ++x)
                for (int y = -1; y <= 1; ++y)
                    for (int z = -1; z <= 1; ++z)
                        if (x || y || z) {
                            const double l = std::sqrt(double(x * x + y * y + z * z));
                            dirs.push_back({x / l, y / l, z / l});
                        }
            long long tests = 0, cuts = 0, by_octant = 0, by_dirs = 0, wrong = 0, vertices = 0, far = 0, open = 0;
            std::vector<std::pair<double, Vec3<double>>> candidates;
            for (int t = 0; t < n; ++t) {
                const auto& p = slot_points[t];
                int home[3];
                for (int a = 0; a < 3; ++a) home[a] = std::clamp(static_cast<int>(p[a] / width), 0, side - 1);
                candidates.clear();
                for (int dx = -reach; dx <= reach; ++dx)
                    for (int dy = -reach; dy <= reach; ++dy)
                        for (int dz = -reach; dz <= reach; ++dz) {
                            const int u[3] = {home[0] + dx, home[1] + dy, home[2] + dz};
                            int key = 0;
                            double image[3];
                            for (int a = 0; a < 3; ++a) {
                                key = key * side + (u[a] + side) % side;
                                image[a] = u[a] < 0 ? -1.0 : u[a] >= side ? 1.0 : 0.0;
                            }
                            for (int q = offsets(key); q < offsets(key + 1); ++q) {
                                if (q == t) continue;
                                const Vec3<double> d{slot_points[q][0] + image[0] - p[0], slot_points[q][1] + image[1] - p[1],
                                                     slot_points[q][2] + image[2] - p[2]};
                                candidates.push_back({dot(d, d), d});
                            }
                        }
                std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
                shell_voronoi::detail::Polyhedron<double> cell;
                shell_voronoi::detail::start_cube(cell, half_width);
                bool certified = false;
                for (const auto& [d2, d] : candidates) {
                    if (d2 >= 4 * cell.radius2) {
                        certified = true;
                        break;
                    }
                    // Bounds from the current vertices.
                    double octant[8] = {0, 0, 0, 0, 0, 0, 0, 0}, support[26];
                    for (int k = 0; k < 26; ++k) support[k] = -1e300;
                    for (int v = 0; v < cell.vertices; ++v) {
                        const Vec3<double>& x = cell.vertex[v];
                        const int o = (x.x < 0) | (x.y < 0) << 1 | (x.z < 0) << 2;
                        octant[o] = std::max(octant[o], std::sqrt(dot(x, x)));
                        for (int k = 0; k < 26; ++k) support[k] = std::max(support[k], dot(x, dirs[k]));
                        vertices += 1;
                        far += dot(x, x) >= 0.25 * d2;
                    }
                    const double half = 0.5 * d2, radius = std::sqrt(cell.radius2);
                    bool octant_cuts = false;
                    for (int o = 0; o < 8 && !octant_cuts; ++o) {
                        const double s[3] = {o & 1 ? -1.0 : 1.0, o & 2 ? -1.0 : 1.0, o & 4 ? -1.0 : 1.0};
                        const double c[3] = {std::max(s[0] * d.x, 0.0), std::max(s[1] * d.y, 0.0), std::max(s[2] * d.z, 0.0)};
                        octant_cuts = octant[o] * std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]) > half;
                    }
                    double best = radius * std::sqrt(d2);
                    for (int k = 0; k < 26; ++k) {
                        const double along = dot(d, dirs[k]);
                        if (along <= 0) continue;
                        const Vec3<double> across = d - dirs[k] * along;
                        best = std::min(best, radius * std::sqrt(dot(across, across)) + along * support[k]);
                    }
                    const bool dirs_cut = best > half;
                    const int planes = cell.planes;
                    if (shell_voronoi::detail::clip(cell, d, 0) != shell_voronoi::ok) break;
                    const bool cut = cell.planes > planes;
                    ++tests;
                    cuts += cut;
                    by_octant += !octant_cuts;
                    by_dirs += !dirs_cut;
                    wrong += cut && (!octant_cuts || !dirs_cut);
                }
                open += !certified;
            }
            std::cout << "cullstats: per_cell tests=" << double(tests) / n << " cuts=" << double(cuts) / n
                      << " rejected_by_octant_radii=" << double(by_octant) / n
                      << " rejected_by_26_directions=" << double(by_dirs) / n
                      << " vertices_per_test=" << double(vertices) / tests
                      << " far_vertices_per_test=" << double(far) / tests << " wrong_rejections=" << wrong
                      << " uncertified=" << open << "\n";
        }

        if (selection == "faces") {
            shell_voronoi::Cells<Memory> cells(n, false);
            // Room for every owned face's ring; a cell that still overflows is
            // rebuilt by the fallback, which records no ring, and is left out.
            cells.rings = Kokkos::View<int**, Memory>("rings", n, 384);
            Kokkos::deep_copy(cells.rings, -1);
            cells.all_neighbours = Kokkos::View<int**, Memory>("all_neighbours", n, 40);
            shell_voronoi::construct_lean<double>(exec, grid, cells, stencil);
            auto status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.status);
            auto owned = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.owned_count);
            auto neighbour = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.neighbour);
            auto rings = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.rings);
            auto built = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.volume);
            std::vector<int> owner, other, start{0}, ring;
            int skipped = 0;
            for (int t = 0; t < n; ++t) {
                if (status(t) != shell_voronoi::ok || (owned(t) > 0 && rings(t, 0) < 0)) {
                    ++skipped;  // failed, or built by the fallback, which records no rings
                    continue;
                }
                for (int f = 0, at = 0; f < owned(t); ++f) {
                    const int k = rings(t, at);
                    owner.push_back(t);
                    other.push_back(neighbour(t, f));
                    for (int m = 0; m < k; ++m) ring.push_back(rings(t, at + 1 + m));
                    start.push_back(static_cast<int>(ring.size()));
                    at += 1 + k;
                }
            }
            auto to_device = [](const std::vector<int>& v, const char* label) {
                Kokkos::View<int*, Memory> d(label, v.size());
                auto h = Kokkos::create_mirror_view(d);
                for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
                Kokkos::deep_copy(d, h);
                return d;
            };
            const face_geometry::Faces<Memory> faces{to_device(owner, "owner"), to_device(other, "neighbour"),
                                                      to_device(start, "ring_start"), to_device(ring, "ring")};
            const int face_count = static_cast<int>(owner.size());
            const face_geometry::Geometry<Memory> geometry{
                Kokkos::View<double* [3], Memory>("face_area", face_count),
                Kokkos::View<double* [3], Memory>("face_centroid", face_count),
                Kokkos::View<double*, Memory>("volume", n)};
            const double geometry_ms = median_ms(repeats, [&] {
                face_geometry::compute(exec, search.positions, 1.0, faces, geometry);
                exec.fence();
            });
            const double build_ms_lean = median_ms(repeats, [&] {
                shell_voronoi::construct_lean<double>(exec, grid, cells, stencil);
                exec.fence();
            });
            auto volume = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, geometry.volume);
            // Cells whose neighbour has no rings miss that neighbour's owned
            // faces, so only cells with complete face lists are compared.
            auto lists = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.all_neighbours);
            auto counts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.face_count);
            std::vector<char> complete(n, 1);
            for (int t = 0; t < n; ++t)
                if (status(t) != shell_voronoi::ok || (owned(t) > 0 && rings(t, 0) < 0)) {
                    complete[t] = 0;
                    for (int f = 0; f < counts(t); ++f) complete[lists(t, f)] = 0;
                }
            double worst = 0, total = 0;
            int compared = 0;
            for (int t = 0; t < n; ++t) {
                total += volume(t);
                if (!complete[t]) continue;
                ++compared;
                worst = std::max(worst, std::abs(volume(t) - built(t)) / built(t));
            }
            std::cout << "  compared " << compared << " of " << n << " cells\n";
            if (std::getenv("FACES_DEBUG")) {
                auto all = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.face_count);
                std::vector<int> seen(n, 0);
                for (std::size_t f = 0; f < owner.size(); ++f) {
                    ++seen[owner[f]];
                    ++seen[other[f]];
                }
                int shown = 0;
                for (int t = 0; t < n && shown < 6; ++t)
                    if (status(t) == shell_voronoi::ok && std::abs(volume(t) - built(t)) / built(t) > 1e-9) {
                        std::cout << "  cell " << t << " faces(build)=" << all(t) << " faces(list)=" << seen[t]
                                  << " owned=" << owned(t) << " volume(build)=" << built(t)
                                  << " volume(faces)=" << volume(t) << "\n";
                        ++shown;
                    }
            }
            std::cout << "faces: faces=" << face_count << " (" << double(face_count) / n
                      << " per cell) ring_entries_per_face=" << double(ring.size()) / face_count
                      << " geometry_ms=" << geometry_ms << " geometry_Mcells_s=" << n / geometry_ms / 1e3
                      << " clipping_build_ms=" << build_ms_lean << " clipping_Mcells_s=" << n / build_ms_lean / 1e3
                      << " cells_without_rings=" << skipped << " total_volume=" << total
                      << " max_relative_volume_difference=" << worst << "\n";
        }

        // Topology change per step: exact neighbour sets (by particle) before
        // and after one step's displacement of `displacement` spacings, for a
        // displacement field given as the seventh argument.
        if (selection == "topology") {
            const std::string field = argc > 7 ? argv[7] : "random";
            const int capacity = 40;
            auto neighbour_sets = [&](const std::vector<std::array<double, 3>>& at) {
                {
                    auto h = Kokkos::create_mirror_view(positions);
                    for (int i = 0; i < n; ++i)
                        for (int a = 0; a < 3; ++a) h(i, a) = at[i][a];
                    Kokkos::deep_copy(positions, h);
                }
                neighbor_search::build(exec, search, positions, unused_radii);
                shell_voronoi::Cells<Memory> cells(n, false);
                cells.all_neighbours = Kokkos::View<int**, Memory>("all_neighbours", n, capacity);
                shell_voronoi::construct_lean<double>(exec, grid, cells, stencil);
                auto order = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, search.order);
                auto lists = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.all_neighbours);
                auto counts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.face_count);
                auto status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.status);
                std::vector<std::vector<int>> sets(n);
                for (int t = 0; t < n; ++t) {
                    auto& set = sets[order(t)];
                    if (status(t) != shell_voronoi::ok) {
                        set = {-1};  // marks an unbuilt cell
                        continue;
                    }
                    for (int f = 0; f < counts(t); ++f) set.push_back(order(lists(t, f)));
                    std::sort(set.begin(), set.end());
                }
                return sets;
            };
            // Smooth fields have unit maximum speed; displacement = amplitude * spacing * field.
            const double two_pi = 2.0 * 3.14159265358979323846;
            std::mt19937_64 rng(11);
            std::uniform_real_distribution<double> unit(-1.0, 1.0);
            std::array<std::array<double, 7>, 6> modes;  // k (3), direction (3), phase
            for (auto& m : modes) {
                for (int a = 0; a < 3; ++a) m[a] = std::round(2.5 * unit(rng));
                for (int a = 3; a < 6; ++a) m[a] = unit(rng);
                m[6] = two_pi * unit(rng);
            }
            auto velocity = [&](const std::array<double, 3>& x) -> std::array<double, 3> {
                if (field == "shear") return {std::sin(two_pi * x[1]), 0.0, 0.0};
                if (field == "turbulent") {
                    std::array<double, 3> v{0, 0, 0};
                    for (const auto& m : modes) {
                        const double s = std::sin(two_pi * (m[0] * x[0] + m[1] * x[1] + m[2] * x[2]) + m[6]);
                        for (int a = 0; a < 3; ++a) v[a] += m[3 + a] * s / 3.0;
                    }
                    return v;
                }
                return {unit(rng), unit(rng), unit(rng)};  // random, incoherent
            };
            std::vector<std::array<double, 3>> moved(n);
            for (int i = 0; i < n; ++i) {
                const auto v = velocity(points[i]);
                for (int a = 0; a < 3; ++a) moved[i][a] = wrap(points[i][a] + displacement * spacing * v[a]);
            }
            const auto before = neighbour_sets(points);
            const auto after = neighbour_sets(moved);
            long long changed = 0, gained = 0, lost = 0, unbuilt = 0;
            for (int i = 0; i < n; ++i) {
                if (before[i] == std::vector<int>{-1} || after[i] == std::vector<int>{-1}) {
                    ++unbuilt;
                    continue;
                }
                if (before[i] == after[i]) continue;
                ++changed;
                std::vector<int> d;
                std::set_difference(after[i].begin(), after[i].end(), before[i].begin(), before[i].end(),
                                    std::back_inserter(d));
                gained += d.size();
                d.clear();
                std::set_difference(before[i].begin(), before[i].end(), after[i].begin(), after[i].end(),
                                    std::back_inserter(d));
                lost += d.size();
            }
            std::cout << "topology: field=" << field << " displacement=" << displacement
                      << " changed_cells=" << 100.0 * changed / n << "% faces_gained_per_changed_cell="
                      << (changed ? double(gained) / changed : 0.0)
                      << " faces_lost_per_changed_cell=" << (changed ? double(lost) / changed : 0.0)
                      << " unbuilt=" << unbuilt << "\n";
        }

        // Evolving mesh: particles carried by a smooth flow for 40 steps,
        // `displacement` spacings per step at the fastest point, and steered a
        // fraction of the way toward their cell centroid each step (mesh
        // regularisation; the eighth argument, default 0.3). Reports the share
        // of cells whose neighbour set changes in each step.
        if (selection == "evolve") {
            const std::string field = argc > 7 ? argv[7] : "turbulent";
            const double steer = argc > 8 ? std::atof(argv[8]) : 0.3;
            const int steps = 40, capacity = 40;
            const double two_pi = 2.0 * 3.14159265358979323846;
            std::mt19937_64 rng(11);
            std::uniform_real_distribution<double> unit(-1.0, 1.0);
            std::array<std::array<double, 7>, 6> modes;
            for (auto& m : modes) {
                for (int a = 0; a < 3; ++a) m[a] = std::round(2.5 * unit(rng));
                for (int a = 3; a < 6; ++a) m[a] = unit(rng);
                m[6] = two_pi * unit(rng);
            }
            auto velocity = [&](const std::array<double, 3>& x) -> std::array<double, 3> {
                if (field == "shear") return {std::sin(two_pi * x[1]), 0.0, 0.0};
                std::array<double, 3> v{0, 0, 0};
                for (const auto& m : modes) {
                    const double sv = std::sin(two_pi * (m[0] * x[0] + m[1] * x[1] + m[2] * x[2]) + m[6]);
                    for (int a = 0; a < 3; ++a) v[a] += m[3 + a] * sv / 3.0;
                }
                return v;
            };
            std::vector<std::array<double, 3>> x(points.begin(), points.end());
            std::vector<std::vector<int>> previous;
            for (int step = 0; step <= steps; ++step) {
                {
                    auto h = Kokkos::create_mirror_view(positions);
                    for (int i = 0; i < n; ++i)
                        for (int a = 0; a < 3; ++a) h(i, a) = x[i][a];
                    Kokkos::deep_copy(positions, h);
                }
                neighbor_search::build(exec, search, positions, unused_radii);
                shell_voronoi::Cells<Memory> cells(n, false);
                cells.all_neighbours = Kokkos::View<int**, Memory>("all_neighbours", n, capacity);
                shell_voronoi::construct_lean<double>(exec, grid, cells, stencil);
                auto order = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, search.order);
                auto lists = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.all_neighbours);
                auto counts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.face_count);
                auto centroid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.centroid);
                std::vector<std::vector<int>> sets(n);
                std::vector<std::array<double, 3>> offset(n);
                for (int t = 0; t < n; ++t) {
                    const int i = order(t);
                    for (int f = 0; f < counts(t); ++f) sets[i].push_back(order(lists(t, f)));
                    std::sort(sets[i].begin(), sets[i].end());
                    offset[i] = {centroid(t, 0), centroid(t, 1), centroid(t, 2)};
                }
                if (step > 0) {
                    int changed = 0;
                    for (int i = 0; i < n; ++i) changed += sets[i] != previous[i];
                    if (step <= 3 || step % 5 == 0)
                        std::cout << "evolve: field=" << field << " displacement=" << displacement
                                  << " steer=" << steer << " step=" << step
                                  << " changed_cells=" << 100.0 * changed / n << "%\n";
                }
                previous = std::move(sets);
                for (int i = 0; i < n; ++i) {
                    const auto v = velocity(x[i]);
                    for (int a = 0; a < 3; ++a)
                        x[i][a] = wrap(x[i][a] + displacement * spacing * v[a] + steer * offset[i][a]);
                }
            }
        }

        // Persistent tetrahedra: built once from an exact build, then the
        // per-step kernels (circumcentres, faces, local Delaunay check) are
        // timed, the geometry is checked against the build, and after moving
        // the particles by `displacement` spacings in a smooth turbulent field
        // the check's flagged cells are compared with the cells whose
        // neighbours really changed in an exact rebuild.
        if (selection == "tets") {
            const int capacity = 40;
            auto exact_sets = [&](const Kokkos::View<double* [3], Memory>& at, shell_voronoi::Cells<Memory>& cells) {
                neighbor_search::build(exec, search, at, unused_radii);
                shell_voronoi::construct_lean<double>(exec, grid, cells, stencil);
                auto order = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, search.order);
                auto lists = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.all_neighbours);
                auto counts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.face_count);
                std::vector<std::vector<int>> sets(n);
                for (int t = 0; t < n; ++t) {
                    for (int f = 0; f < counts(t); ++f) sets[order(t)].push_back(order(lists(t, f)));
                    std::sort(sets[order(t)].begin(), sets[order(t)].end());
                }
                return sets;
            };
            // Build 0 at the starting positions, in slot order: from here on a
            // particle is named by its slot of this build.
            neighbor_search::build(exec, search, positions, unused_radii);
            Kokkos::View<double* [3], Memory> x0("x0", n);
            Kokkos::deep_copy(x0, search.positions);
            shell_voronoi::Cells<Memory> cells(n, false);
            cells.rings = Kokkos::View<int**, Memory>("rings", n, 384);
            cells.all_neighbours = Kokkos::View<int**, Memory>("all_neighbours", n, capacity);
            Kokkos::deep_copy(cells.rings, -1);
            const auto before = exact_sets(x0, cells);  // positions already in slot order: order is identity
            auto status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.status);
            auto owned = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.owned_count);
            auto neighbour = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.neighbour);
            auto rings = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.rings);
            auto built = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.volume);
            auto order0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, search.order);

            // Tetrahedra from the face rings, keyed by their sorted particles.
            struct KeyHash {
                std::size_t operator()(const std::array<int, 4>& k) const {
                    std::size_t h = 0;
                    for (int v : k) h = h * 1000003u ^ std::size_t(v);
                    return h;
                }
            };
            std::unordered_map<std::array<int, 4>, int, KeyHash> tet_id;
            std::vector<std::array<int, 4>> tets;
            std::vector<int> owner, other, start{0}, ring;
            int skipped = 0;
            for (int t = 0; t < n; ++t) {
                const int i = order0(t);
                if (status(t) != shell_voronoi::ok || (owned(t) > 0 && rings(t, 0) < 0)) {
                    ++skipped;
                    continue;
                }
                for (int f = 0, at = 0; f < owned(t); ++f) {
                    const int k = rings(t, at);
                    const int j = order0(neighbour(t, f));
                    owner.push_back(i);
                    other.push_back(j);
                    for (int m = 0; m < k; ++m) {
                        std::array<int, 4> key{i, j, order0(rings(t, at + 1 + m)), order0(rings(t, at + 1 + (m + 1) % k))};
                        std::sort(key.begin(), key.end());
                        auto [it, fresh] = tet_id.emplace(key, static_cast<int>(tets.size()));
                        if (fresh) tets.push_back(key);
                        ring.push_back(it->second);
                    }
                    start.push_back(static_cast<int>(ring.size()));
                    at += 1 + k;
                }
            }
            // Neighbours across each face, from the shared triangles.
            const int tet_count = static_cast<int>(tets.size());
            std::vector<std::array<int, 4>> adjacent(tet_count, {-1, -1, -1, -1});
            {
                struct TriHash {
                    std::size_t operator()(const std::array<int, 3>& k) const {
                        return (std::size_t(k[0]) * 1000003u ^ std::size_t(k[1])) * 1000003u ^ std::size_t(k[2]);
                    }
                };
                std::unordered_map<std::array<int, 3>, std::pair<int, int>, TriHash> open;
                open.reserve(2 * tets.size());
                for (int t = 0; t < tet_count; ++t)
                    for (int k = 0; k < 4; ++k) {
                        std::array<int, 3> tri;
                        for (int a = 0, b = 0; a < 4; ++a)
                            if (a != k) tri[b++] = tets[t][a];
                        auto it = open.find(tri);
                        if (it == open.end()) {
                            open.emplace(tri, std::make_pair(t, k));
                        } else {
                            adjacent[t][k] = it->second.first;
                            adjacent[it->second.first][it->second.second] = t;
                            open.erase(it);
                        }
                    }
            }
            {
                long long open = 0;
                for (const auto& a : adjacent)
                    for (int k = 0; k < 4; ++k) open += a[k] < 0;
                std::cout << "  unpaired_triangles=" << open << " of " << 4LL * tet_count << "\n";
            }
            tet_mesh::Mesh<Memory> mesh{Kokkos::View<int* [4], Memory>("tet_vertex", tet_count),
                                        Kokkos::View<int* [4], Memory>("tet_adjacent", tet_count),
                                        Kokkos::View<double* [3], Memory>("tet_centre", tet_count),
                                        {}, {}, {}, {}};
            {
                auto hv = Kokkos::create_mirror_view(mesh.vertex);
                auto ha = Kokkos::create_mirror_view(mesh.adjacent);
                for (int t = 0; t < tet_count; ++t)
                    for (int k = 0; k < 4; ++k) {
                        hv(t, k) = tets[t][k];
                        ha(t, k) = adjacent[t][k];
                    }
                Kokkos::deep_copy(mesh.vertex, hv);
                Kokkos::deep_copy(mesh.adjacent, ha);
                auto to_device = [](const std::vector<int>& v, const char* label) {
                    Kokkos::View<int*, Memory> d(label, v.size());
                    auto h = Kokkos::create_mirror_view(d);
                    for (std::size_t q = 0; q < v.size(); ++q) h(q) = v[q];
                    Kokkos::deep_copy(d, h);
                    return d;
                };
                mesh.owner = to_device(owner, "face_owner");
                mesh.neighbour = to_device(other, "face_neighbour");
                mesh.ring_start = to_device(start, "face_ring_start");
                mesh.ring = to_device(ring, "face_ring");
            }
            const int face_count = static_cast<int>(owner.size());
            const face_geometry::Geometry<Memory> geometry{Kokkos::View<double* [3], Memory>("area", face_count),
                                                           Kokkos::View<double* [3], Memory>("centroid", face_count),
                                                           Kokkos::View<double*, Memory>("volume", n)};
            Kokkos::View<int*, Memory> invalid("invalid", tet_count);
            int failed = 0;
            const double centre_ms = median_ms(repeats, [&] {
                tet_mesh::circumcentres(exec, x0, 1.0, mesh);
                exec.fence();
            });
            const double face_ms = median_ms(repeats, [&] {
                tet_mesh::faces(exec, x0, 1.0, mesh, geometry);
                exec.fence();
            });
            const double check_ms = median_ms(repeats, [&] {
                failed = tet_mesh::check(exec, x0, 1.0, mesh, invalid);
            });
            auto volume = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, geometry.volume);
            double worst = 0;
            for (int t = 0; t < n; ++t)
                if (status(t) == shell_voronoi::ok) {
                    const int i = order0(t);
                    worst = std::max(worst, std::abs(volume(i) - built(t)) / built(t));
                }
            const double step_ms = centre_ms + face_ms + check_ms;
            std::cout << "tets: per_cell=" << double(tet_count) / n << " faces_per_cell=" << double(face_count) / n
                      << " skipped=" << skipped << " centres_ms=" << centre_ms << " faces_ms=" << face_ms
                      << " check_ms=" << check_ms << " step_Mcells_s=" << n / step_ms / 1e3
                      << " invalid_at_start=" << failed << " max_relative_volume_difference=" << worst << "\n";
            {
                // The same tetrahedra built on the device from the rings.
                tet_mesh::Mesh<Memory> built_mesh;
                int unresolved = 0;
                const double from_rings_ms = median_ms(repeats, [&] {
                    unresolved = tet_mesh::from_rings(exec, cells, built_mesh);
                    exec.fence();
                });
                tet_mesh::circumcentres(exec, x0, 1.0, built_mesh);
                tet_mesh::faces(exec, x0, 1.0, built_mesh, geometry);
                const int again = tet_mesh::check(exec, x0, 1.0, built_mesh, invalid);
                auto v2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, geometry.volume);
                double worst2 = 0;
                for (int t = 0; t < n; ++t)
                    if (status(t) == shell_voronoi::ok)
                        worst2 = std::max(worst2, std::abs(v2(order0(t)) - built(t)) / built(t));
                std::cout << "tets_from_rings: tets=" << built_mesh.vertex.extent(0) << " (host " << tet_count
                          << ") unresolved=" << unresolved << " invalid=" << again << " from_rings_ms=" << from_rings_ms
                          << " Mcells_s=" << n / from_rings_ms / 1e3
                          << " max_relative_volume_difference=" << worst2 << "\n";
            }

            // Move by a smooth turbulent field, check, and compare with an exact rebuild.
            const double two_pi = 2.0 * 3.14159265358979323846;
            std::mt19937_64 rng(11);
            std::uniform_real_distribution<double> unit(-1.0, 1.0);
            std::array<std::array<double, 7>, 6> modes;
            for (auto& m : modes) {
                for (int a = 0; a < 3; ++a) m[a] = std::round(2.5 * unit(rng));
                for (int a = 3; a < 6; ++a) m[a] = unit(rng);
                m[6] = two_pi * unit(rng);
            }
            Kokkos::View<double* [3], Memory> x1("x1", n);
            {
                auto h0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x0);
                auto h1 = Kokkos::create_mirror_view(x1);
                for (int i = 0; i < n; ++i) {
                    std::array<double, 3> v{0, 0, 0};
                    for (const auto& m : modes) {
                        const double sv = std::sin(two_pi * (m[0] * h0(i, 0) + m[1] * h0(i, 1) + m[2] * h0(i, 2)) + m[6]);
                        for (int a = 0; a < 3; ++a) v[a] += m[3 + a] * sv / 3.0;
                    }
                    for (int a = 0; a < 3; ++a) h1(i, a) = wrap(h0(i, a) + displacement * spacing * v[a]);
                }
                Kokkos::deep_copy(x1, h1);
            }
            Kokkos::View<int*, Memory> sign0("sign0", tet_count), sign1("sign1", tet_count);
            tet_mesh::orientations(exec, x0, 1.0, mesh, sign0);
            tet_mesh::orientations(exec, x1, 1.0, mesh, sign1);
            tet_mesh::circumcentres(exec, x1, 1.0, mesh);
            failed = tet_mesh::check(exec, x1, 1.0, mesh, invalid);
            std::vector<char> flagged(n, 0);
            int inverted = 0;
            {
                auto hi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, invalid);
                auto s0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sign0);
                auto s1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sign1);
                for (int t = 0; t < tet_count; ++t) {
                    const bool flip = s1(t) != s0(t);
                    inverted += flip;
                    if (hi(t) || flip)
                        for (int k = 0; k < 4; ++k) flagged[tets[t][k]] = 1;
                }
            }
            std::cout << "  inverted_tets=" << inverted << "\n";
            shell_voronoi::Cells<Memory> after_cells(n, false);
            after_cells.all_neighbours = Kokkos::View<int**, Memory>("all_neighbours_after", n, capacity);
            const auto after = exact_sets(x1, after_cells);  // indexed by particle = slot of build 0
            long long changed = 0, caught = 0, flagged_cells = 0;
            for (int i = 0; i < n; ++i) {
                flagged_cells += flagged[i];
                if (before[order0(i)].empty()) continue;
                // before[] is indexed through build 0's order, which is the identity here.
                const bool change = after[i] != before[i];
                changed += change;
                caught += change && flagged[i];
                if (change && !flagged[i] && std::getenv("TETS_DEBUG")) {
                    std::vector<int> gained, lost;
                    std::set_difference(after[i].begin(), after[i].end(), before[i].begin(), before[i].end(),
                                        std::back_inserter(gained));
                    std::set_difference(before[i].begin(), before[i].end(), after[i].begin(), after[i].end(),
                                        std::back_inserter(lost));
                    // Is each gained neighbour already joined to i by an edge of the
                    // tetrahedra, i.e. a face that only had rounding-level area?
                    int already = 0;
                    for (int g : gained) {
                        bool edge = false;
                        for (int t = 0; t < tet_count && !edge; ++t) {
                            bool hi = false, hg = false;
                            for (int k = 0; k < 4; ++k) {
                                hi |= tets[t][k] == i;
                                hg |= tets[t][k] == g;
                            }
                            edge = hi && hg;
                        }
                        already += edge;
                    }
                    std::cout << "  missed cell " << i << " faces " << before[i].size() << "->" << after[i].size()
                              << " gained " << gained.size() << " lost " << lost.size()
                              << " gained_already_in_tetrahedra=" << already << "\n";
                }
            }
            // Flip repair over several steps, each compared with an exact rebuild.
            if (argc > 7 && std::string(argv[7]) == "flips") {
                std::vector<tet_flips::Point> xs(n);
                {
                    auto h0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x0);
                    for (int i = 0; i < n; ++i) xs[i] = {h0(i, 0), h0(i, 1), h0(i, 2)};
                }
                tet_flips::Mesh fm = tet_flips::make_mesh(tet_flips::snap(xs), tets);
                auto previous = before;
                for (int step = 1; step <= 10; ++step) {
                    for (int i = 0; i < n; ++i) {
                        std::array<double, 3> v{0, 0, 0};
                        for (const auto& mode : modes) {
                            const double sv = std::sin(two_pi * (mode[0] * xs[i][0] + mode[1] * xs[i][1] +
                                                                 mode[2] * xs[i][2]) + mode[6]);
                            for (int a = 0; a < 3; ++a) v[a] += mode[3 + a] * sv / 3.0;
                        }
                        for (int a = 0; a < 3; ++a) xs[i][a] = wrap(xs[i][a] + displacement * spacing * v[a]);
                    }
                    Kokkos::Timer timer;
                    const auto grid_xs = tet_flips::snap(xs);
                    const auto stats = tet_flips::repair(fm, grid_xs);
                    const double repair_ms = timer.seconds() * 1e3;
                    const long long left = tet_flips::failing(fm, grid_xs);
                    if (left > 0 && std::getenv("TETS_DEBUG")) {
                        for (std::size_t t = 0; t < fm.vertex.size(); ++t) {
                            if (!fm.alive[t]) continue;
                            for (int k = 0; k < 4; ++k) {
                                const int u = fm.adjacent[t][k];
                                if (u < 0) continue;
                                int far = -1;
                                for (int v : fm.vertex[u])
                                    if (tet_flips::detail::position(fm.vertex[t], v) < 0) far = v;
                                if (far < 0 || !tet_flips::detail::in_sphere(grid_xs, fm.vertex[t], far)) continue;
                                int back = 0;
                                for (int j = 0; j < 4; ++j) back += fm.adjacent[u][j] == int(t);
                                std::cout << "  failing t=" << t << " u=" << u << " alive_u=" << int(fm.alive[u])
                                          << " back=" << back << " orient_t="
                                          << tet_flips::detail::orient_sign(grid_xs, fm.vertex[t])
                                          << " orient_u=" << tet_flips::detail::orient_sign(grid_xs, fm.vertex[u])
                                          << " reverse_fails="
                                          << tet_flips::detail::in_sphere(grid_xs, fm.vertex[u], fm.vertex[t][k]) << "\n";
                            }
                        }
                    }
                    const auto repaired = tet_flips::neighbours(fm, n);
                    Kokkos::View<double* [3], Memory> xd("xs", n);
                    {
                        auto h = Kokkos::create_mirror_view(xd);
                        for (int i = 0; i < n; ++i)
                            for (int a = 0; a < 3; ++a) h(i, a) = xs[i][a];
                        Kokkos::deep_copy(xd, h);
                    }
                    shell_voronoi::Cells<Memory> step_cells(n, false);
                    step_cells.all_neighbours = Kokkos::View<int**, Memory>("all_neighbours_step", n, capacity);
                    const auto exact = exact_sets(xd, step_cells);
                    long long mismatched = 0, changed = 0;
                    for (int i = 0; i < n; ++i) {
                        mismatched += repaired[i] != exact[i];
                        changed += exact[i] != previous[i];
                    }
                    if (mismatched && std::getenv("TETS_DEBUG")) {
                        // Size of each disputed face: spread of the circumcentres around edge i-j.
                        auto wrap = [](double d) { return d - std::nearbyint(d); };
                        auto centre = [&](const tet_flips::Tet& t) {
                            auto d = [&](int v) {
                                return shell_voronoi::Vec3<double>{wrap(xs[v][0] - xs[t[0]][0]),
                                                                   wrap(xs[v][1] - xs[t[0]][1]),
                                                                   wrap(xs[v][2] - xs[t[0]][2])};
                            };
                            return face_geometry::detail::circumcentre(d(t[1]), d(t[2]), d(t[3])) +
                                   shell_voronoi::Vec3<double>{xs[t[0]][0], xs[t[0]][1], xs[t[0]][2]};
                        };
                        auto step_status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, step_cells.status);
                        auto step_order = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, search.order);
                        std::vector<int> status_of(n);
                        for (int t = 0; t < n; ++t) status_of[step_order(t)] = step_status(t);
                        double volume = 0;
                        for (std::size_t t = 0; t < fm.vertex.size(); ++t) {
                            if (!fm.alive[t]) continue;
                            const auto& v = fm.vertex[t];
                            auto d = [&](int k) {
                                return shell_voronoi::Vec3<double>{wrap(xs[v[k]][0] - xs[v[0]][0]),
                                                                   wrap(xs[v[k]][1] - xs[v[0]][1]),
                                                                   wrap(xs[v[k]][2] - xs[v[0]][2])};
                            };
                            volume += dot(d(1), cross(d(2), d(3))) / 6.0;
                        }
                        std::cout << "  total_tet_volume=" << volume << "\n";
                        // The same positions through the plain thread-per-cell kernel.
                        shell_voronoi::Cells<Memory> plain(n, false);
                        plain.all_neighbours = Kokkos::View<int**, Memory>("all_neighbours_plain", n, capacity);
                        shell_voronoi::construct<double>(exec, grid, plain);
                        auto plain_lists = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, plain.all_neighbours);
                        auto plain_counts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, plain.face_count);
                        auto plain_status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, plain.status);
                        for (int i = 0; i < n; ++i) {
                            if (repaired[i] == exact[i]) continue;
                            int slot = 0;
                            while (step_order(slot) != i) ++slot;
                            std::vector<int> plain_set;
                            for (int f = 0; f < plain_counts(slot); ++f) plain_set.push_back(step_order(plain_lists(slot, f)));
                            std::sort(plain_set.begin(), plain_set.end());
                            std::cout << "  slot " << slot << " at " << xs[i][0] << " " << xs[i][1] << " " << xs[i][2] << "\n";
                            std::cout << "  cell " << i << " reference_status=" << status_of[i] << " mesh_faces="
                                      << repaired[i].size() << " reference_faces=" << exact[i].size()
                                      << " plain_status=" << plain_status(slot) << " plain_faces=" << plain_set.size()
                                      << " plain_matches_mesh=" << (plain_set == repaired[i]) << "\n";
                            std::vector<int> diff;
                            std::set_symmetric_difference(repaired[i].begin(), repaired[i].end(), exact[i].begin(),
                                                          exact[i].end(), std::back_inserter(diff));
                            for (int j : diff) {
                                std::vector<shell_voronoi::Vec3<double>> cs;
                                for (std::size_t t = 0; t < fm.vertex.size(); ++t)
                                    if (fm.alive[t] && tet_flips::detail::position(fm.vertex[t], i) >= 0 &&
                                        tet_flips::detail::position(fm.vertex[t], j) >= 0)
                                    {
                                        cs.push_back(centre(fm.vertex[t]));
                                        const auto& v = fm.vertex[t];
                                        auto d = [&](int k) {
                                            return shell_voronoi::Vec3<double>{wrap(xs[v[k]][0] - xs[v[0]][0]),
                                                                               wrap(xs[v[k]][1] - xs[v[0]][1]),
                                                                               wrap(xs[v[k]][2] - xs[v[0]][2])};
                                        };
                                        const auto c = face_geometry::detail::circumcentre(d(1), d(2), d(3));
                                        std::cout << "    tet " << t << " (" << v[0] << "," << v[1] << "," << v[2] << ","
                                                  << v[3] << ") volume/spacing^3=" << dot(d(1), cross(d(2), d(3))) / 6 / (spacing * spacing * spacing)
                                                  << " radius/spacing=" << std::sqrt(dot(c, c)) / spacing << "\n";
                                    }
                                double spread = 0;
                                for (auto& p : cs)
                                    for (auto& q : cs) {
                                        const shell_voronoi::Vec3<double> d{wrap(p.x - q.x), wrap(p.y - q.y), wrap(p.z - q.z)};
                                        spread = std::max(spread, std::sqrt(dot(d, d)));
                                    }
                                double area = 0;  // largest triangle of circumcentres: a lower bound on the face area
                                for (auto& p : cs)
                                    for (auto& q : cs)
                                        for (auto& r : cs) {
                                            const shell_voronoi::Vec3<double> u{wrap(q.x - p.x), wrap(q.y - p.y), wrap(q.z - p.z)},
                                                w{wrap(r.x - p.x), wrap(r.y - p.y), wrap(r.z - p.z)};
                                            const auto c = cross(u, w);
                                            area = std::max(area, 0.5 * std::sqrt(dot(c, c)));
                                        }
                                std::cout << "  area/spacing^2>=" << area / (spacing * spacing) << " ring=" << cs.size();
                                std::cout << "  disputed " << i << "-" << j << " in_mesh="
                                          << std::binary_search(repaired[i].begin(), repaired[i].end(), j)
                                          << " face_extent/spacing=" << spread / spacing << "\n";
                            }
                        }
                    }
                    previous = exact;
                    std::cout << "flips: step=" << step << " changed_cells=" << 100.0 * changed / n
                              << "% flips23=" << stats.flips23 << " flips32=" << stats.flips32
                              << " stuck=" << stats.stuck << " rebuilt=" << stats.rebuilt
                              << " unrepaired=" << stats.unrepaired << " failing_after=" << left
                              << " mismatched_cells=" << mismatched << " flips_per_changed_cell="
                              << (changed ? double(stats.flips23 + stats.flips32) / changed : 0.0)
                              << " host_repair_ms=" << repair_ms << "\n";
                }
            }
            std::cout << "tets moved: displacement=" << displacement << " invalid_tets=" << failed << " ("
                      << 100.0 * failed / tet_count << "%) flagged_cells=" << 100.0 * flagged_cells / n
                      << "% changed_cells=" << 100.0 * changed / n << "% changed_and_flagged=" << caught
                      << " missed=" << changed - caught << "\n";
        }

        if (displacement > 0 && selection != "topology" && selection != "evolve" && selection != "tets") {
            if (selection == "all" || std::string("lean-double").find(selection) != std::string::npos)
                warm_experiment.template operator()<double>("lean-double");
            if (selection == "all" || std::string("lean-float").find(selection) != std::string::npos)
                warm_experiment.template operator()<float>("lean-float");
        }
    }
    Kokkos::finalize();
    return failures == 0 ? 0 : 1;
}
