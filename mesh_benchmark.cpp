#include "local_voronoi.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Positions = Kokkos::View<double*[3]>;
using IntView = Kokkos::View<int*>;

int particle_id(int i, int j, int k, int side) {
    return (i * side + j) * side + k;
}

double periodic_delta(double d) {
    return d - std::floor(d + 0.5);
}

struct Input {
    int side;
    double reach;
    Positions positions;
    IntView offsets;
    IntView ids;
    std::vector<std::vector<int>> host_neighbors;
};

Input make_periodic_lattice(int side) {
    if (side < 12) throw std::runtime_error("side length must be at least 12");
    const int n = side * side * side;
    const double spacing = 1.0 / side;
    const double reach = 3.5 * spacing;
    const int stencil = static_cast<int>(std::ceil(reach / spacing + 0.21));
    std::mt19937_64 rng(20261027);
    std::uniform_real_distribution<double> jitter(-0.1 * spacing, 0.1 * spacing);

    Positions positions("positions", n);
    auto hp = Kokkos::create_mirror_view(positions);
    for (int i = 0; i < side; ++i)
        for (int j = 0; j < side; ++j)
            for (int k = 0; k < side; ++k) {
                const int id = particle_id(i, j, k, side);
                const int grid[3] = {i, j, k};
                for (int a = 0; a < 3; ++a) {
                    double x = (grid[a] + 0.5) * spacing + jitter(rng);
                    if (x < 0.0) x += 1.0;
                    if (x >= 1.0) x -= 1.0;
                    hp(id, a) = x;
                }
            }
    Kokkos::deep_copy(positions, hp);

    std::vector<std::vector<int>> neighbors(n);
    std::vector<std::vector<int>> ordered_neighbors(n);
    const double reach2 = reach * reach;
    for (int i = 0; i < side; ++i)
        for (int j = 0; j < side; ++j)
            for (int k = 0; k < side; ++k) {
                const int row = particle_id(i, j, k, side);
                auto& list = neighbors[row];
                for (int di = -stencil; di <= stencil; ++di)
                    for (int dj = -stencil; dj <= stencil; ++dj)
                        for (int dk = -stencil; dk <= stencil; ++dk) {
                            const int a = (i + di + side) % side;
                            const int b = (j + dj + side) % side;
                            const int c = (k + dk + side) % side;
                            const int other = particle_id(a, b, c, side);
                            if (other == row) continue;
                            double d2 = 0.0;
                            for (int axis = 0; axis < 3; ++axis) {
                                const double d = periodic_delta(hp(other, axis) - hp(row, axis));
                                d2 += d * d;
                            }
                            if (d2 <= reach2) list.push_back(other);
                        }
                std::sort(list.begin(), list.end());
                ordered_neighbors[row] = list;
                std::sort(ordered_neighbors[row].begin(), ordered_neighbors[row].end(),
                    [&](int lhs, int rhs) {
                        auto distance2 = [&](int other) {
                            double d2 = 0.0;
                            for (int axis = 0; axis < 3; ++axis) {
                                const double d = periodic_delta(hp(other, axis) - hp(row, axis));
                                d2 += d * d;
                            }
                            return d2;
                        };
                        const double dl = distance2(lhs), dr = distance2(rhs);
                        return dl == dr ? lhs < rhs : dl < dr;
                    });
            }

    IntView offsets("neighbor_offsets", n + 1);
    std::size_t edge_count = 0;
    for (const auto& row : neighbors) edge_count += row.size();
    IntView ids("neighbor_ids", edge_count);
    auto ho = Kokkos::create_mirror_view(offsets);
    auto hi = Kokkos::create_mirror_view(ids);
    int cursor = 0;
    for (int row = 0; row < n; ++row) {
        ho(row) = cursor;
        for (int id : ordered_neighbors[row]) hi(cursor++) = id;
    }
    ho(n) = cursor;
    Kokkos::deep_copy(offsets, ho);
    Kokkos::deep_copy(ids, hi);
    return {side, reach, positions, offsets, ids, std::move(neighbors)};
}

bool check_neighbor_graph(const Input& input) {
    const int n = input.side * input.side * input.side;
    if (n > 8192) return true;  // The structured stencil is exact by construction.
    auto hp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, input.positions);
    bool valid = true;
    for (int row = 0; row < n; ++row)
        for (int other = 0; other < n; ++other) {
            if (row == other) continue;
            double d2 = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                const double d = periodic_delta(hp(other, axis) - hp(row, axis));
                d2 += d * d;
            }
            const bool expected = d2 <= input.reach * input.reach;
            const auto& row_ids = input.host_neighbors[row];
            const bool found = std::binary_search(row_ids.begin(), row_ids.end(), other);
            if (expected != found) valid = false;
        }
    return valid;
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    int exit_code = 0;
    {
        const int side = argc > 1 ? std::max(12, std::atoi(argv[1])) : 12;
        Input input = make_periodic_lattice(side);
        const int n = side * side * side;
        Kokkos::View<double*> reach("reach", n);
        Kokkos::deep_copy(reach, input.reach);
        Kokkos::View<int*> face_counts("face_counts", n);
        Kokkos::View<int**> face_ids("face_ids", n, local_voronoi::max_faces);
        Kokkos::View<double***> face_areas("face_areas", n, local_voronoi::max_faces, 3);
        Kokkos::View<int**> face_vertex_counts("face_vertex_counts", n,
                                               local_voronoi::max_faces);
        Kokkos::View<double****> face_vertices("face_vertices", n,
            local_voronoi::max_faces, local_voronoi::max_face_vertices, 3);
        Kokkos::View<int*> status("row_status", n);

        Kokkos::Timer timer;
        local_voronoi::reconstruct(Exec{}, input.positions, input.offsets, input.ids,
            reach, Kokkos::Array<double, 3>{1.0, 1.0, 1.0}, face_counts, face_ids,
            face_areas, face_vertex_counts, face_vertices, status);
        const double mesh_ms = timer.seconds() * 1.0e3;

        auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, status);
        auto hcounts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, face_counts);
        auto hids = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, face_ids);
        auto hareas = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, face_areas);
        int good = 0, bad_faces = 0;
        std::array<int, 8> status_counts{};
        double max_closure = 0.0, max_pair_mismatch = 0.0;
        for (int row = 0; row < n; ++row) {
            if (hs(row) >= 0 && hs(row) < static_cast<int>(status_counts.size()))
                ++status_counts[hs(row)];
            if (hs(row) != local_voronoi::success) continue;
            ++good;
            std::array<double, 3> sum{0.0, 0.0, 0.0};
            for (int f = 0; f < hcounts(row); ++f)
                for (int a = 0; a < 3; ++a) sum[a] += hareas(row, f, a);
            const double closure = std::sqrt(sum[0]*sum[0] + sum[1]*sum[1] + sum[2]*sum[2]);
            max_closure = std::max(max_closure, closure);
            for (int f = 0; f < hcounts(row); ++f) {
                const int other = hids(row, f);
                int reverse = -1;
                for (int g = 0; g < hcounts(other); ++g)
                    if (hids(other, g) == row) { reverse = g; break; }
                if (reverse < 0) { ++bad_faces; continue; }
                double mismatch2 = 0.0;
                for (int a = 0; a < 3; ++a) {
                    const double d = hareas(row, f, a) + hareas(other, reverse, a);
                    mismatch2 += d * d;
                }
                max_pair_mismatch = std::max(max_pair_mismatch, std::sqrt(mismatch2));
            }
        }

        const bool graph_ok = check_neighbor_graph(input);
        std::cout << "backend=" << Kokkos::DefaultExecutionSpace::name()
                  << " side=" << side << " particles=" << n
                  << " reach=" << input.reach
                  << " mean_candidates=" << static_cast<double>(input.ids.extent(0)) / n
                  << " mesh_ms=" << mesh_ms << " valid_cells=" << good << '/' << n
                  << " graph_check=" << (graph_ok ? "pass" : "FAIL")
                  << " unmatched_faces=" << bad_faces
                  << " max_area_closure=" << max_closure
                  << " max_shared_face_mismatch=" << max_pair_mismatch
                  << " statuses=";
        for (int count : status_counts) std::cout << count << ',';
        std::cout << '\n';

        if (!graph_ok || good != n || bad_faces != 0 || max_closure > 1.0e-9 ||
            max_pair_mismatch > 1.0e-9) exit_code = 2;
    }
    Kokkos::finalize();
    return exit_code;
}
