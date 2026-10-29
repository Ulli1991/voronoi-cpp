// Validate and time the fused natural-neighbour construction (shell_voronoi).
//
// Usage: shell_voronoi_check [n=32768] [uniform|jittered|lattice|wave|clustered|<layout>@<eps>]
//                            [variants=all] [repeats=5] [width_factor=1] [displacement=0]
// Variants are thread- and warp- with -double or -float; the
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
#include "tet_repair.hpp"
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
        const double displacement = argc > 6 ? std::atof(argv[6]) : 0.0;
        const Exec exec{};

        const Points points = make_points(kind, requested);
        const int n = static_cast<int>(points.size());
        Kokkos::View<double* [3], Memory> positions("positions", n);
        {
            auto h = Kokkos::create_mirror_view(positions);
            for (int i = 0; i < n; ++i)
                for (int a = 0; a < 3; ++a) h(i, a) = points[i][a];
            Kokkos::deep_copy(positions, h);
        }

        // About one particle per grid cell; only the cell index is built.
        neighbor_search::CellList<Exec> search;
        const double spacing = std::cbrt(1.0 / n);
        // SHELL_MORTON=1: cells stored in Morton order, so consecutive slots form compact bricks.
        search.morton = std::getenv("SHELL_MORTON") != nullptr;
        neighbor_search::configure(search, n, 1.0, width_factor * spacing);
        const double build_ms = median_ms(repeats, [&] {
            neighbor_search::build(exec, search, positions);
            exec.fence();
        });
        const shell_voronoi::Grid<Memory> grid = shell_voronoi::make_grid(search);

        std::vector<std::array<double, 3>> slot_points(n);
        {
            auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, search.positions);
            for (int t = 0; t < n; ++t) slot_points[t] = {h(t, 0), h(t, 1), h(t, 2)};
        }

        std::cout << "# backend=" << Exec::name() << " n=" << n << " distribution=" << kind
                  << " grid=" << search.cells_per_side << "^3 repeats=" << repeats << "\n";
        std::vector<double> reference_volumes;
        for (const std::string variant : {"thread-double", "thread-float", "warp-double", "warp-float"}) {
            if (selection != "all" && variant.find(selection) == std::string::npos) continue;
            shell_voronoi::Cells<Memory> cells(n, true);
            const double construct_ms = median_ms(repeats, [&] {
                if (variant == "thread-double") shell_voronoi::construct<double>(exec, grid, cells);
                if (variant == "thread-float") shell_voronoi::construct<float>(exec, grid, cells);
                if (variant == "warp-double") shell_voronoi::construct_warp<double>(exec, grid, cells);
                if (variant == "warp-float") shell_voronoi::construct_warp<float>(exec, grid, cells);
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

        // Persistent tetrahedra: built once from an exact build, then the
        // per-step kernels (circumcentres, faces, local Delaunay check) are
        // timed, the geometry is checked against the build, and after moving
        // the particles by `displacement` spacings in a smooth turbulent field
        // the check's flagged cells are compared with the cells whose
        // neighbours really changed in an exact rebuild.
        if (selection == "tets") {
            // Every cell's natural neighbours (by particle), from the owned faces.
            auto exact_sets = [&](const Kokkos::View<double* [3], Memory>& at, shell_voronoi::Cells<Memory>& cells) {
                neighbor_search::build(exec, search, at);
                shell_voronoi::construct<double>(exec, grid, cells);
                auto order = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, search.order);
                auto neighbour = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.neighbour);
                auto owned = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, cells.owned_count);
                std::vector<std::vector<int>> sets(n);
                for (int t = 0; t < n; ++t)
                    for (int f = 0; f < owned(t); ++f) {
                        const int i = order(t), j = order(neighbour(t, f));
                        sets[i].push_back(j);
                        sets[j].push_back(i);
                    }
                for (auto& set : sets) std::sort(set.begin(), set.end());
                return sets;
            };
            // Build 0 at the starting positions, in slot order: from here on a
            // particle is named by its slot of this build.
            neighbor_search::build(exec, search, positions);
            Kokkos::View<double* [3], Memory> x0("x0", n);
            Kokkos::deep_copy(x0, search.positions);
            shell_voronoi::Cells<Memory> cells(n, false);
            cells.rings = Kokkos::View<int**, Memory>("rings", n, 384);
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
            tet_mesh::Mesh<Memory> mesh{tet_mesh::Quads<Memory>("tet_vertex", tet_count),
                                        tet_mesh::Quads<Memory>("tet_adjacent", tet_count),
                                        tet_mesh::Points4<Memory>("tet_centre", tet_count),
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
            // Positions padded to 32 bytes for the random gathers of the tetrahedra kernels.
            auto padded = [&](const Kokkos::View<double* [3], Memory>& x) {
                tet_mesh::Points4<Memory> p("padded_positions", n);
                Kokkos::parallel_for("pad", Kokkos::RangePolicy<Exec>(exec, 0, n), KOKKOS_LAMBDA(int i) {
                    for (int a = 0; a < 3; ++a) p(i, a) = x(i, a);
                });
                return p;
            };
            const auto x0p = padded(x0);
            const double centre_ms = median_ms(repeats, [&] {
                tet_mesh::circumcentres(exec, x0p, 1.0, mesh);
                exec.fence();
            });
            const double face_ms = median_ms(repeats, [&] {
                tet_mesh::faces(exec, x0p, 1.0, mesh, geometry);
                exec.fence();
            });
            const double check_ms = median_ms(repeats, [&] {
                failed = tet_mesh::check(exec, x0p, 1.0, mesh, invalid);
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
                        for (int i = 0; i < n; ++i) {
                            if (repaired[i] == exact[i]) continue;
                            std::cout << "  cell " << i << " at " << xs[i][0] << " " << xs[i][1] << " " << xs[i][2]
                                      << " reference_status=" << status_of[i] << " mesh_faces=" << repaired[i].size()
                                      << " reference_faces=" << exact[i].size() << "\n";
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
            // Flip repair on the device, finished on the host (stuck triangles,
            // inverted tetrahedra), compared with an exact rebuild every step.
            if (argc > 7 && std::string(argv[7]) == "gpuflips") {
                tet_mesh::Mesh<Memory> dm;
                tet_mesh::from_rings(exec, cells, dm);
                const int capacity = 2 * static_cast<int>(dm.vertex.extent(0));
                auto state = tet_repair::prepare(exec, x0, dm, capacity);
                std::vector<tet_flips::Point> xs(n);
                {
                    auto h0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x0);
                    for (int i = 0; i < n; ++i) xs[i] = {h0(i, 0), h0(i, 1), h0(i, 2)};
                }
                Kokkos::View<double* [3], Memory> xd("xd", n);
                Kokkos::View<int* [2], Memory> stuck;
                // Faces rebuilt every step from the edge representatives.
                const int face_capacity = 10 * n;
                Kokkos::View<int*, Memory> face_start("face_start", capacity + 1);
                Kokkos::View<int* [2], Memory> face_pair("face_pair", face_capacity);
                tet_mesh::FaceRings<Memory> rings[2] = {
                    tet_mesh::make_face_rings<Memory>(capacity, face_capacity, 8 * face_capacity),
                    tet_mesh::make_face_rings<Memory>(capacity, face_capacity, 8 * face_capacity)};
                const face_geometry::Geometry<Memory> face_geometry{
                    Kokkos::View<double* [3], Memory>("face_area", face_capacity),
                    Kokkos::View<double* [3], Memory>("face_centroid", face_capacity),
                    Kokkos::View<double*, Memory>("cell_volume", n)};
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
                    {
                        auto h = Kokkos::create_mirror_view(xd);
                        for (int i = 0; i < n; ++i)
                            for (int a = 0; a < 3; ++a) h(i, a) = xs[i][a];
                        Kokkos::deep_copy(xd, h);
                    }
                    int stuck_count = 0;
                    Kokkos::Timer timer;
                    // GPUFLIPS_SCREEN=1: circumcentres first (the geometry needs them
                    // anyway), then only screened tetrahedra are checked exactly.
                    if (std::getenv("GPUFLIPS_SCREEN")) {
                        const auto xdp = padded(xd);
                        tet_mesh::circumcentres(exec, xdp, 1.0, dm, state.count);
                        tet_repair::move_screened(exec, xdp, 1.0, dm, state);
                    } else {
                        tet_repair::move(exec, xd, state);
                    }
                    const auto gpu = tet_repair::repair(exec, dm, state, stuck, stuck_count);
                    exec.fence();
                    const double gpu_ms = timer.seconds() * 1e3;
                    const auto xg = padded(xd);
                    Kokkos::Timer face_timer;
                    tet_mesh::circumcentres(exec, xg, 1.0, dm, state.count);
                    int faces = 0;
                    if (std::getenv("GPUFLIPS_WALK_ALL")) {  // every face walked around its edge
                        faces = tet_mesh::faces_from_edges(exec, xg, 1.0, dm, state.count, face_start, face_pair,
                                                           face_geometry);
                    } else {  // rings kept, only faces touched by flips walked
                        faces = tet_mesh::update_face_rings(exec, dm, state.count, state.touched, rings[step % 2],
                                                            rings[(step + 1) % 2]);
                        tet_mesh::faces(exec, xg, 1.0, dm, face_geometry);
                        Kokkos::deep_copy(exec, state.touched, 0);
                    }
                    exec.fence();
                    const double faces_ms = face_timer.seconds() * 1e3;
                    if (std::getenv("GPUFLIPS_FAST")) {  // timing only: no host check
                        std::cout << "gpuflips: step=" << step << " gpu_ms=" << gpu_ms
                                  << " gpu_Mcells_s=" << n / gpu_ms / 1e3 << " rounds=" << gpu.rounds
                                  << " flips23=" << gpu.flips23 << " flips32=" << gpu.flips32
                                  << " stuck=" << stuck_count << " tets=" << state.count << " faces=" << faces
                                  << " centres_faces_ms=" << faces_ms << " step_ms=" << gpu_ms + faces_ms
                                  << " step_Mcells_s=" << n / (gpu_ms + faces_ms) / 1e3 << "\n";
                        continue;
                    }
                    // Finish on the host and continue from its mesh.
                    auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dm.vertex);
                    auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dm.adjacent);
                    auto hl = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, state.alive);
                    tet_flips::Mesh fm;
                    for (int t = 0; t < state.count; ++t) {
                        fm.vertex.push_back({hv(t, 0), hv(t, 1), hv(t, 2), hv(t, 3)});
                        fm.adjacent.push_back({ha(t, 0), ha(t, 1), ha(t, 2), ha(t, 3)});
                        fm.alive.push_back(hl(t));
                    }
                    const auto grid_xs = tet_flips::snap(xs);
                    const long long failing_after_gpu = tet_flips::failing(fm, grid_xs);
                    auto adjacency_report = [&](const char* when) {
                        long long dead = 0, wrong_face = 0, no_back = 0;
                        for (std::size_t t = 0; t < fm.vertex.size(); ++t) {
                            if (!fm.alive[t]) continue;
                            for (int k = 0; k < 4; ++k) {
                                const int u = fm.adjacent[t][k];
                                if (u < 0) continue;
                                if (!fm.alive[u]) {
                                    ++dead;
                                    continue;
                                }
                                int shared = 0, back = 0;
                                for (int a = 0; a < 4; ++a)
                                    if (a != k && tet_flips::detail::position(fm.vertex[u], fm.vertex[t][a]) >= 0) ++shared;
                                for (int c = 0; c < 4; ++c) back += fm.adjacent[u][c] == int(t);
                                wrong_face += shared != 3;
                                no_back += back != 1;
                            }
                        }
                        std::cout << "  adjacency " << when << ": to_dead=" << dead << " wrong_face=" << wrong_face
                                  << " no_back=" << no_back << "\n";
                    };
                    if (std::getenv("TETS_DEBUG")) {
                        adjacency_report("after gpu");
                        auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, stuck);
                        long long inverted = 0, unflippable = 0, rejected = 0;
                        std::vector<char> seen(fm.vertex.size(), 0);
                        long long distinct_inverted = 0;
                        for (int i = 0; i < std::min<int>(stuck_count, hs.extent(0)); ++i) {
                            const int k = hs(i, 1);
                            inverted += k == -1;
                            rejected += k == -2;
                            unflippable += k >= 0;
                            const int t = hs(i, 0);
                            if (k == -1 && t >= 0 && t < int(seen.size()) && !seen[t]++) ++distinct_inverted;
                        }
                        std::cout << "  stuck by reason: inverted=" << inverted << " (distinct " << distinct_inverted
                                  << ") unflippable=" << unflippable << " rejected=" << rejected << "\n";
                    }
                    const auto host = tet_flips::repair(fm, grid_xs);
                    if (std::getenv("TETS_DEBUG")) adjacency_report("after host");
                    const long long left = tet_flips::failing(fm, grid_xs);
                    const auto repaired = tet_flips::neighbours(fm, n);
                    shell_voronoi::Cells<Memory> step_cells(n, false);
                    const auto exact = exact_sets(xd, step_cells);
                    long long mismatched = 0, exact_faces = 0;
                    for (int i = 0; i < n; ++i) {
                        mismatched += repaired[i] != exact[i];
                        exact_faces += static_cast<long long>(exact[i].size());
                    }
                    // Volumes from the rebuilt faces against the clipping reference.
                    double worst_volume = 0, total_volume = 0;
                    {
                        auto hv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, face_geometry.volume);
                        auto rv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, step_cells.volume);
                        auto ro = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, search.order);
                        auto rs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, step_cells.status);
                        int worst_t = -1;
                        for (int t = 0; t < n; ++t) {
                            total_volume += hv(ro(t));
                            if (rv(t) > 0 && std::abs(hv(ro(t)) - rv(t)) / rv(t) > worst_volume) {
                                worst_volume = std::abs(hv(ro(t)) - rv(t)) / rv(t);
                                worst_t = t;
                            }
                        }
                        if (worst_t >= 0 && std::getenv("TETS_DEBUG"))
                            std::cout << "  worst cell " << ro(worst_t) << " ours=" << hv(ro(worst_t))
                                      << " reference=" << rv(worst_t) << " reference_status=" << rs(worst_t)
                                      << " mean_volume=" << 1.0 / n << " reference_faces="
                                      << exact[ro(worst_t)].size() << " mesh_faces=" << repaired[ro(worst_t)].size() << "\n";
                    }
                    std::cout << "gpuflips_faces: step=" << step << " faces=" << faces << " exact_faces="
                              << exact_faces / 2 << " total_volume=" << total_volume
                              << " max_relative_volume_difference=" << worst_volume << " centres_faces_ms=" << faces_ms
                              << "\n";
                    std::cout << "gpuflips: step=" << step << " gpu_ms=" << gpu_ms
                              << " gpu_Mcells_s=" << n / gpu_ms / 1e3 << " rounds=" << gpu.rounds
                              << " flips23=" << gpu.flips23 << " flips32=" << gpu.flips32
                              << " stuck=" << stuck_count << " failing_after_gpu=" << failing_after_gpu
                              << " host_flips=" << host.flips23 + host.flips32 << " host_rebuilt=" << host.rebuilt
                              << " failing_after=" << left << " mismatched_cells=" << mismatched << "\n";
                    // Back to the device.
                    const int count = static_cast<int>(fm.vertex.size());
                    if (count > capacity) break;
                    // Tetrahedra the host changed count as touched for the next ring update.
                    auto ht = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, state.touched);
                    const int gpu_count = state.count;
                    for (int t = 0; t < count; ++t)
                        for (int k = 0; k < 4; ++k) {
                            if (t >= gpu_count || hv(t, k) != fm.vertex[t][k] || ha(t, k) != fm.adjacent[t][k] ||
                                hl(t) != fm.alive[t])
                                ht(t) = 1;
                            hv(t, k) = fm.vertex[t][k];
                            ha(t, k) = fm.adjacent[t][k];
                        }
                    Kokkos::deep_copy(state.touched, ht);
                    for (int t = 0; t < count; ++t) hl(t) = fm.alive[t];
                    Kokkos::deep_copy(dm.vertex, hv);
                    Kokkos::deep_copy(dm.adjacent, ha);
                    Kokkos::deep_copy(state.alive, hl);
                    {
                        // Edge representatives for the host's mesh: the lowest
                        // tetrahedron index around each edge (test only).
                        auto he = Kokkos::create_mirror_view(dm.edges);
                        for (std::size_t t = 0; t < he.extent(0); ++t) he(t) = 0;
                        for (int t = 0; t < count; ++t) {
                            if (!fm.alive[t]) continue;
                            for (int k = 0; k < 6; ++k) {
                                int ka, kb;
                                tet_mesh::edge_ends(k, ka, kb);
                                const int a = fm.vertex[t][ka], b = fm.vertex[t][kb];
                                int lowest = t;
                                for (int current = t, before = -1, steps = 0; steps < 64; ++steps) {
                                    int next = -1;
                                    for (int c = 0; c < 4 && next < 0; ++c) {
                                        const int v = fm.vertex[current][c];
                                        if (v != a && v != b && fm.adjacent[current][c] != before)
                                            next = fm.adjacent[current][c];
                                    }
                                    if (next < 0 || next == t) break;
                                    lowest = std::min(lowest, next);
                                    before = current;
                                    current = next;
                                }
                                if (lowest == t) he(t) |= 1 << k;
                            }
                        }
                        Kokkos::deep_copy(dm.edges, he);
                    }
                    state.count = count;
                    state.free_count = 0;
                }
            }
            std::cout << "tets moved: displacement=" << displacement << " invalid_tets=" << failed << " ("
                      << 100.0 * failed / tet_count << "%) flagged_cells=" << 100.0 * flagged_cells / n
                      << "% changed_cells=" << 100.0 * changed / n << "% changed_and_flagged=" << caught
                      << " missed=" << changed - caught << "\n";
        }

    }
    Kokkos::finalize();
    return failures == 0 ? 0 : 1;
}
