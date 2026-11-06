#pragma once

#include "tri_repair.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <utility>
#include <vector>

// Host fallback of the 2D repair: patches around triangles stuck inverted are replaced by the Delaunay triangles of the
// patch's own corners (Bowyer-Watson, exact and perturbed by particle id); accepted if the boundary matches, else the patch grows.
namespace tri_finish {

namespace ep = exact_predicates;

constexpr int first_layers = 3, most_triangles = 2048, attempts = 7;

using Triple = std::array<int, 3>;
using Edge = std::pair<int, int>;

// The Delaunay triangles of p[3..] and the three far points p[0..2] (a
// counter-clockwise triangle around them), as indices, counter-clockwise.
inline std::vector<Triple> delaunay(const std::vector<ep::Offset2>& p, const std::vector<int>& id) {
    std::vector<Triple> triangles = {{0, 1, 2}}, kept;
    std::vector<Edge> rim;
    for (int e = 3; e < static_cast<int>(p.size()); ++e) {
        kept.clear();
        rim.clear();
        for (const Triple& t : triangles) {
            const ep::Offset2 q[4] = {p[t[0]], p[t[1]], p[t[2]], p[e]};
            const int ids[4] = {id[t[0]], id[t[1]], id[t[2]], id[e]};
            if (!ep::in_circle(q, ids)) {
                kept.push_back(t);
                continue;
            }
            for (int k = 0; k < 3; ++k) {  // an edge of two cavity triangles is no rim
                const Edge edge{t[(k + 1) % 3], t[(k + 2) % 3]};
                const auto twin = std::find(rim.begin(), rim.end(), Edge{edge.second, edge.first});
                if (twin == rim.end())
                    rim.push_back(edge);
                else
                    rim.erase(twin);
            }
        }
        triangles.swap(kept);
        for (const Edge& edge : rim) triangles.push_back({edge.first, edge.second, e});
    }
    return triangles;
}

struct Stats {
    long long patches = 0, triangles = 0;
    bool done = true;  // false: a patch could not be replaced
};

// Replaces patches around the inverted or flat triangles. Afterwards
// state.list holds the patches' edges and state.touched their slots
// (appended); run tri_repair::repair next.
template <class ExecSpace>
Stats finish(const ExecSpace& exec, const ep::Points<typename ExecSpace::memory_space, 2>& point,
             const tri_repair::Triangles<typename ExecSpace::memory_space>& mesh,
             tri_repair::State<typename ExecSpace::memory_space>& state) {
    using Memory = typename ExecSpace::memory_space;
    auto corners = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.corners);
    auto links = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.links);
    auto points = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, point.view);
    const ep::i64 px = point.period[0], py = point.period[1];
    const auto offset = [&](int from, int to) {
        return ep::Offset2{ep::delta(points(from, 0), points(to, 0), px), ep::delta(points(from, 1), points(to, 1), py)};
    };
    const auto positive = [&](int t) { return ep::cross(offset(corners(t, 0), corners(t, 1)), offset(corners(t, 0), corners(t, 2))) > 0; };

    Stats stats;
    std::vector<int> changed, edges, slots;        // triangles, edges and slots to send back
    std::map<int, Edge> faces;                     // faces to send back: their particles
    std::vector<char> in_patch(static_cast<std::size_t>(mesh.count()), 0);
    std::vector<int> patch, front;
    for (int seed = 0; seed < mesh.count(); ++seed) {
        if (positive(seed)) continue;
        patch.assign(1, seed);
        in_patch[seed] = 1;
        bool replaced = false;
        std::vector<int> grown = patch;
        for (int layers = first_layers, attempt = 0; attempt < attempts && !replaced; ++attempt, layers = 2) {
            for (int l = 0; l < layers; ++l) {
                front.clear();
                for (const int t : grown)
                    for (int k = 0; k < 3; ++k) {
                        const int u = links(t, k);
                        if (in_patch[u]) continue;
                        in_patch[u] = 1;
                        front.push_back(u);
                    }
                patch.insert(patch.end(), front.begin(), front.end());
                grown = front;
            }
            if (static_cast<int>(patch.size()) > most_triangles) break;

            // The patch's corners in a local frame, its boundary and inner faces.
            std::map<int, int> local;
            std::vector<ep::Offset2> p;
            std::vector<int> id;
            const int origin = corners(seed, 0);
            constexpr ep::i64 far = ep::i64(1) << 28;
            p = {{-far, -far}, {3 * far, -far}, {-far, 3 * far}};
            id = {-3, -2, -1};
            std::map<Edge, std::pair<int, int>> rim;  // directed edge: outer triangle, face
            std::vector<int> inner;
            bool fits = true;
            for (const int t : patch)
                for (int k = 0; k < 3; ++k) {
                    const int a = corners(t, (k + 1) % 3), b = corners(t, (k + 2) % 3), u = links(t, k);
                    if (local.emplace(a, static_cast<int>(p.size())).second) {
                        p.push_back(offset(origin, a));
                        id.push_back(a);
                        fits = fits && std::abs(p.back()[0]) < far / 2 && std::abs(p.back()[1]) < far / 2;
                    }
                    if (!in_patch[u])  // an edge twice (flips at inversions can leave that): not yet a disc
                        fits = rim.insert({{a, b}, {u, links(t, 3 + k)}}).second && fits;
                    else if (t < u)
                        inner.push_back(links(t, 3 + k));
                }
            if (!fits) continue;

            // The Delaunay triangles inside the boundary: flood from it.
            const std::vector<Triple> all = delaunay(p, id);
            std::map<Edge, int> holder;  // directed edge in particles: triangle of `all`
            for (int m = 0; m < static_cast<int>(all.size()); ++m)
                for (int k = 0; k < 3; ++k) holder[{id[all[m][(k + 1) % 3]], id[all[m][(k + 2) % 3]]}] = m;
            std::vector<char> inside(all.size(), 0);
            std::vector<int> stack;
            bool closed = true;
            for (const auto& [edge, outer] : rim) {
                (void)outer;
                const auto it = holder.find(edge);
                if (it == holder.end()) {
                    closed = false;
                    break;
                }
                if (!inside[it->second]) {
                    inside[it->second] = 1;
                    stack.push_back(it->second);
                }
            }
            if (!closed) continue;
            std::vector<int> chosen;
            while (!stack.empty() && closed) {
                const int m = stack.back();
                stack.pop_back();
                chosen.push_back(m);
                for (int k = 0; k < 3; ++k) {
                    const int a = id[all[m][(k + 1) % 3]], b = id[all[m][(k + 2) % 3]];
                    if (rim.count({a, b})) continue;
                    const auto it = holder.find({b, a});
                    if (it == holder.end() || a < 0 || b < 0) {
                        closed = false;
                        break;
                    }
                    if (!inside[it->second]) {
                        inside[it->second] = 1;
                        stack.push_back(it->second);
                    }
                }
            }
            if (!closed || chosen.size() != patch.size()) continue;
            bool flat = false;
            for (const int m : chosen) {
                const ep::Offset2 a = ep::minus(p[all[m][1]], p[all[m][0]]), b = ep::minus(p[all[m][2]], p[all[m][0]]);
                flat = flat || ep::cross(a, b) <= 0;
            }
            if (flat) continue;

            // Into the patch's slots, keeping every row's corner 0 the lowest.
            std::map<Edge, int> slot_of;
            std::vector<Triple> rows;
            for (std::size_t s = 0; s < chosen.size(); ++s) {
                Triple v = {id[all[chosen[s]][0]], id[all[chosen[s]][1]], id[all[chosen[s]][2]]};
                std::rotate(v.begin(), std::min_element(v.begin(), v.end()), v.end());
                rows.push_back(v);
                for (int k = 0; k < 3; ++k) slot_of[{v[(k + 1) % 3], v[(k + 2) % 3]}] = patch[s];
            }
            for (std::size_t s = 0; s < rows.size(); ++s) {
                const Triple& v = rows[s];
                const int t = patch[s];
                for (int k = 0; k < 3; ++k) corners(t, k) = v[k];
            }
            std::size_t next_face = 0;
            std::map<Edge, int> numbered;
            for (std::size_t s = 0; s < rows.size(); ++s) {
                const Triple& v = rows[s];
                const int t = patch[s];
                for (int k = 0; k < 3; ++k) {
                    const int a = v[(k + 1) % 3], b = v[(k + 2) % 3];
                    edges.push_back(t << 2 | k);
                    const auto outer = rim.find({a, b});
                    if (outer != rim.end()) {
                        const int x = outer->second.first, f = outer->second.second;
                        links(t, k) = x;
                        links(t, 3 + k) = f;
                        for (int j = 0; j < 3; ++j)
                            if (links(x, 3 + j) == f) {
                                corners(t, 3 + k) = corners(x, j);
                                links(x, j) = t;
                                corners(x, 3 + j) = v[k];
                            }
                        changed.push_back(x);
                        continue;
                    }
                    const int u = slot_of.at({b, a});
                    links(t, k) = u;
                    int apex = -1;
                    for (int j = 0; j < 3; ++j)
                        if (corners(u, j) != a && corners(u, j) != b) apex = corners(u, j);
                    corners(t, 3 + k) = apex;
                    const Edge key{std::min(a, b), std::max(a, b)};
                    auto number = numbered.find(key);
                    if (number == numbered.end()) {
                        number = numbered.emplace(key, inner[next_face++]).first;
                        faces[number->second] = key;
                    }
                    links(t, 3 + k) = number->second;
                }
                changed.push_back(t);
                slots.push_back(t);
            }
            replaced = true;
            ++stats.patches;
            stats.triangles += static_cast<long long>(patch.size());
        }
        for (const int t : patch) in_patch[t] = 0;
        if (!replaced) stats.done = false;
    }

    // Send the changed rows back.
    std::sort(changed.begin(), changed.end());
    changed.erase(std::unique(changed.begin(), changed.end()), changed.end());
    const int count = static_cast<int>(changed.size());
    Kokkos::View<int* [13], Kokkos::LayoutRight, Kokkos::HostSpace> rows("tri_finish_rows", count);
    for (int s = 0; s < count; ++s) {
        rows(s, 0) = changed[s];
        for (int c = 0; c < 6; ++c) {
            rows(s, 1 + c) = corners(changed[s], c);
            rows(s, 7 + c) = links(changed[s], c);
        }
    }
    Kokkos::View<int* [13], Kokkos::LayoutRight, Memory> device_rows("tri_finish_rows", count);
    Kokkos::deep_copy(exec, device_rows, rows);
    const auto device_corners = mesh.corners, device_links = mesh.links;
    Kokkos::parallel_for(
        "tri_finish::rows", Kokkos::RangePolicy<ExecSpace>(exec, 0, count), KOKKOS_LAMBDA(int s) {
            const int t = device_rows(s, 0);
            for (int c = 0; c < 6; ++c) {
                device_corners(t, c) = device_rows(s, 1 + c);
                device_links(t, c) = device_rows(s, 7 + c);
            }
        });
    const int face_count = static_cast<int>(faces.size());
    Kokkos::View<int* [3], Kokkos::LayoutRight, Kokkos::HostSpace> face_rows("tri_finish_faces", face_count);
    {
        int s = 0;
        for (const auto& [f, particles] : faces) {
            face_rows(s, 0) = f;
            face_rows(s, 1) = particles.first;
            face_rows(s, 2) = particles.second;
            ++s;
        }
    }
    Kokkos::View<int* [3], Kokkos::LayoutRight, Memory> device_faces("tri_finish_faces", face_count);
    Kokkos::deep_copy(exec, device_faces, face_rows);
    const auto owner = mesh.owner, neighbour = mesh.neighbour;
    Kokkos::parallel_for(
        "tri_finish::faces", Kokkos::RangePolicy<ExecSpace>(exec, 0, face_count), KOKKOS_LAMBDA(int s) {
            owner(device_faces(s, 0)) = device_faces(s, 1);
            neighbour(device_faces(s, 0)) = device_faces(s, 2);
        });

    // The next device repair starts from the patches' edges; their slots need centres.
    tri_repair::detail::reserve(state.list, edges.size(), false);
    tri_repair::detail::reserve(state.next, edges.size(), false);
    tri_repair::detail::reserve(state.touched, state.touched_count + slots.size(), true);
    const auto list_host = Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(edges.data(), edges.size());
    const auto slots_host = Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>(slots.data(), slots.size());
    Kokkos::deep_copy(exec, Kokkos::subview(state.list, std::make_pair(std::size_t(0), edges.size())), list_host);
    Kokkos::deep_copy(exec, Kokkos::subview(state.touched, std::make_pair(static_cast<std::size_t>(state.touched_count), state.touched_count + slots.size())),
                      slots_host);
    exec.fence();
    state.entries = static_cast<int>(edges.size());
    state.touched_count += static_cast<int>(slots.size());
    return stats;
}

}  // namespace tri_finish
