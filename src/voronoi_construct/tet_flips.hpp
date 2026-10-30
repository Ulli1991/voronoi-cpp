#pragma once

#include "tet_repair.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <deque>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_set>
#include <vector>

// Host fallback of tet_repair: restores the Delaunay property where the
// device repair got stuck.
//
// A shared triangle is locally Delaunay when the far vertex of either
// tetrahedron lies outside the other's circumsphere. Failing triangles are
// repaired by a 2->3 flip when the two tetrahedra form a convex pair, or by
// a 3->2 flip when one edge of the triangle has exactly three tetrahedra
// around it; otherwise the triangle waits until neighbouring flips change its
// surroundings, and what still fails is rebuilt locally (Bowyer-Watson on the
// particles around it). Every decision is an exact sign on the snapped
// positions shared with the device, so a tie is the same tie everywhere.
namespace tet_flips {

namespace ep = exact_predicates;
using Tet = std::array<int, 4>;

// A host copy of the device mesh, edited in place.
struct Mesh {
    tet_mesh::Quads<Kokkos::HostSpace> vertex, adjacent;  // capacity rows; positively oriented
    Kokkos::View<char*, Kokkos::HostSpace> alive;
    ep::Points<Kokkos::HostSpace> point;
    int count = 0;  // slots in use
};

struct Stats {
    long long checked = 0, flips23 = 0, flips32 = 0, stuck = 0, rebuilt = 0, unrepaired = 0;
};

namespace detail {

inline Tet tet(const Mesh& m, int t) { return {m.vertex(t, 0), m.vertex(t, 1), m.vertex(t, 2), m.vertex(t, 3)}; }

inline std::array<int, 3> face_key(const Tet& t, int k) {
    std::array<int, 3> f;
    tet_mesh::sorted_face(t.data(), k, f.data());
    return f;
}

// Replace the tetrahedra `old` by `fresh` (vertex sets, any orientation)
// covering the same region: orient the new ones, reuse the old slots first,
// and reconnect them to each other and to the outside through the old outer
// faces. Returns the new tetrahedra.
inline std::vector<int> replace(Mesh& m, const std::vector<int>& old, std::vector<Tet> fresh) {
    std::map<std::array<int, 3>, std::pair<int, int>> outside;  // outer face -> (outer tet, its face index)
    for (int t : old)
        for (int k = 0; k < 4; ++k) {
            const int o = m.adjacent(t, k);
            if (std::find(old.begin(), old.end(), o) != old.end()) continue;
            int back = -1;
            if (o >= 0)
                for (int j = 0; j < 4; ++j)
                    if (m.adjacent(o, j) == t) back = j;
            outside[face_key(tet(m, t), k)] = {o, back};
        }
    std::vector<int> slots;
    for (std::size_t i = 0; i < fresh.size(); ++i) {
        int slot;
        if (i < old.size()) {
            slot = old[i];
        } else {
            if (m.count == static_cast<int>(m.vertex.extent(0)))
                throw std::runtime_error("tet_flips: tetrahedra capacity exceeded");
            slot = m.count++;
        }
        if (ep::orient(m.point, fresh[i].data()) < 0) std::swap(fresh[i][2], fresh[i][3]);
        for (int k = 0; k < 4; ++k) {
            m.vertex(slot, k) = fresh[i][k];
            m.adjacent(slot, k) = -1;
        }
        m.alive(slot) = 1;
        slots.push_back(slot);
    }
    for (std::size_t i = fresh.size(); i < old.size(); ++i) m.alive(old[i]) = 0;
    std::map<std::array<int, 3>, std::pair<int, int>> inner;
    for (int s : slots)
        for (int k = 0; k < 4; ++k) {
            const auto key = face_key(tet(m, s), k);
            if (auto it = outside.find(key); it != outside.end()) {
                m.adjacent(s, k) = it->second.first;
                if (it->second.first >= 0) m.adjacent(it->second.first, it->second.second) = s;
            } else if (auto jt = inner.find(key); jt != inner.end()) {
                m.adjacent(s, k) = jt->second.first;
                m.adjacent(jt->second.first, jt->second.second) = s;
            } else {
                inner.emplace(key, std::make_pair(s, k));
            }
        }
    return slots;
}

// Delaunay tetrahedralization of a few points in one local frame by
// Bowyer-Watson insertion into an enclosing tetrahedron, whose corners are
// points n..n+3. Insert near-to-far: each point is located by walking from
// the previous cavity, so the cost stays near linear. Returns the live
// tetrahedra, or nothing if point location fails.
inline std::vector<Tet> delaunay(std::vector<ep::Offset> p, std::vector<int> id) {
    const int n = static_cast<int>(p.size());
    long long extent = 1;
    for (const auto& q : p) extent = std::max({extent, std::abs(q[0]), std::abs(q[1]), std::abs(q[2])});
    const long long k = 2 * extent;
    p.push_back({-k, -k, -k});
    p.push_back({5 * k, -k, -k});
    p.push_back({-k, 5 * k, -k});
    p.push_back({-k, -k, 5 * k});
    for (int c = 0; c < 4; ++c) id.push_back(std::numeric_limits<int>::max() - 3 + c);
    auto orient = [&](const Tet& t) {
        return ep::orient(ep::minus(p[t[1]], p[t[0]]), ep::minus(p[t[2]], p[t[0]]), ep::minus(p[t[3]], p[t[0]]));
    };
    auto inside = [&](const Tet& t, int q) {
        const ep::Offset five[5] = {p[t[0]], p[t[1]], p[t[2]], p[t[3]], p[q]};
        const int ids[5] = {id[t[0]], id[t[1]], id[t[2]], id[t[3]], id[q]};
        return ep::in_sphere(five, ids);
    };
    std::vector<Tet> tet{{n, n + 1, n + 2, n + 3}}, adj{{-1, -1, -1, -1}};
    std::vector<char> alive{1};
    if (orient(tet[0]) < 0) std::swap(tet[0][2], tet[0][3]);
    int last = 0;
    std::vector<int> cavity, stack;
    std::vector<char> in_cavity;
    std::map<std::pair<int, int>, std::pair<int, int>> rim;  // edge -> (new tet, face index)
    for (int q = 0; q < n; ++q) {
        // Walk towards q across any face that has q on its far side.
        int t = last;
        for (int steps = 0, moved = 1; moved && steps < 4 * static_cast<int>(tet.size()); ++steps) {
            moved = 0;
            for (int f = 0; f < 4 && !moved; ++f) {
                Tet s = tet[t];
                s[f] = q;
                if (orient(s) < 0 && adj[t][f] >= 0) {
                    t = adj[t][f];
                    moved = 1;
                }
            }
        }
        if (!inside(tet[t], q)) return {};
        // Cavity: the connected tetrahedra whose circumsphere holds q.
        in_cavity.resize(tet.size(), 0);
        cavity.assign(1, t);
        in_cavity[t] = 1;
        stack.assign(1, t);
        while (!stack.empty()) {
            const int c = stack.back();
            stack.pop_back();
            for (int f = 0; f < 4; ++f) {
                const int o = adj[c][f];
                if (o >= 0 && !in_cavity[o] && inside(tet[o], q)) {
                    in_cavity[o] = 1;
                    cavity.push_back(o);
                    stack.push_back(o);
                }
            }
        }
        // Connect q to every face of the cavity's boundary.
        rim.clear();
        for (int c : cavity)
            for (int f = 0; f < 4; ++f) {
                const int o = adj[c][f];
                if (o >= 0 && in_cavity[o]) continue;
                Tet s = tet[c];
                s[f] = q;  // same orientation as c
                const int fresh = static_cast<int>(tet.size());
                tet.push_back(s);
                adj.push_back({-1, -1, -1, -1});
                alive.push_back(1);
                adj[fresh][f] = o;
                if (o >= 0)
                    for (int j = 0; j < 4; ++j)
                        if (adj[o][j] == c) adj[o][j] = fresh;
                // The faces through q pair up along the boundary edges.
                for (int g = 0; g < 4; ++g) {
                    if (g == f) continue;
                    int a = -1, b = -1;
                    for (int h = 0; h < 4; ++h)
                        if (h != f && h != g) (a < 0 ? a : b) = s[h];
                    const auto key = std::minmax(a, b);
                    if (auto it = rim.find(key); it != rim.end()) {
                        adj[fresh][g] = it->second.first;
                        adj[it->second.first][it->second.second] = fresh;
                        rim.erase(it);
                    } else {
                        rim.emplace(key, std::make_pair(fresh, g));
                    }
                }
                last = fresh;
            }
        for (int c : cavity) {
            alive[c] = 0;
            in_cavity[c] = 0;
        }
    }
    std::vector<Tet> out;
    for (std::size_t t = 0; t < tet.size(); ++t)
        if (alive[t]) out.push_back(tet[t]);
    return out;
}

// Tetrahedra holding any of the particles in `seeds`, found from t through
// adjacency. `seeds` must be sorted.
inline std::vector<int> stars(const Mesh& m, int t, const std::vector<int>& seeds) {
    auto touches = [&](int s) {
        for (int k = 0; k < 4; ++k)
            if (std::binary_search(seeds.begin(), seeds.end(), m.vertex(s, k))) return true;
        return false;
    };
    std::vector<int> found{t}, stack{t};
    std::unordered_set<int> seen{t};
    while (!stack.empty()) {
        const int s = stack.back();
        stack.pop_back();
        for (int k = 0; k < 4; ++k) {
            const int o = m.adjacent(s, k);
            if (o >= 0 && seen.insert(o).second && touches(o)) {
                found.push_back(o);
                stack.push_back(o);
            }
        }
    }
    return found;
}

inline std::vector<int> vertices_of(const Mesh& m, const std::vector<int>& tets) {
    std::vector<int> v;
    for (int t : tets)
        for (int k = 0; k < 4; ++k) v.push_back(m.vertex(t, k));
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

// Fallback for a triangle t|u that no flip repairs, or an inverted
// tetrahedron t (u = -1): replace the stars of their particles by those of a
// fresh Delaunay tetrahedralization of the particles two rings around, if
// the new stars fit the old boundary exactly; otherwise add the particles of
// the faces that did not fit to the stars and retry. Returns the new
// tetrahedra, or nothing after eight tries.
inline std::vector<int> rebuild(Mesh& m, int t, int u) {
    std::vector<int> seeds = vertices_of(m, u < 0 ? std::vector<int>{t} : std::vector<int>{t, u});
    for (int attempt = 0; attempt < 8; ++attempt) {
        const std::vector<int> old = stars(m, t, seeds);
        const std::vector<int> ring = vertices_of(m, old);
        const std::vector<int> points = vertices_of(m, stars(m, t, ring));
        // Local frame centred on the first seed, nearest first.
        std::vector<std::pair<ep::i128, int>> order;
        for (int v : points) order.push_back({ep::lift(ep::offset(m.point, seeds[0], v)), v});
        std::sort(order.begin(), order.end());
        std::vector<ep::Offset> local;
        std::vector<int> id;
        for (auto [r, v] : order) {
            local.push_back(ep::offset(m.point, seeds[0], v));
            id.push_back(v);
        }
        std::vector<Tet> fresh;
        bool reaches_corner = false;
        for (Tet s : delaunay(local, id)) {
            bool keep = false;
            for (int& v : s) {
                if (v >= static_cast<int>(id.size())) {
                    v = -1;
                } else {
                    v = id[v];
                    keep |= std::binary_search(seeds.begin(), seeds.end(), v);
                }
            }
            if (!keep) continue;
            reaches_corner |= std::find(s.begin(), s.end(), -1) != s.end();
            fresh.push_back(s);
        }
        // The new stars must close against the old boundary face for face.
        std::map<std::array<int, 3>, int> boundary;
        for (int s : old)
            for (int k = 0; k < 4; ++k)
                if (std::find(old.begin(), old.end(), m.adjacent(s, k)) == old.end()) ++boundary[face_key(tet(m, s), k)];
        for (const Tet& s : fresh)
            for (int k = 0; k < 4; ++k) {
                const auto key = face_key(s, k);
                if (--boundary[key] == -2) boundary[key] = 0;  // inner face, seen twice
            }
        const bool fits = !fresh.empty() && !reaches_corner &&
                          std::all_of(boundary.begin(), boundary.end(), [](const auto& b) { return b.second == 0; });
        if (fits) return replace(m, old, fresh);
        // Widen the stars over the particles of the faces that did not fit
        // (over the whole ring if that adds none).
        const std::size_t before = seeds.size();
        for (const auto& [key, count] : boundary)
            if (count != 0) seeds.insert(seeds.end(), key.begin(), key.end());
        std::sort(seeds.begin(), seeds.end());
        seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());
        if (seeds.size() == before) seeds = ring;
    }
    return {};
}

// tet_repair::closes for host lists of up to three new tetrahedra.
inline bool closes(const Mesh& m, const std::vector<int>& old, const std::vector<Tet>& fresh) {
    int tets[3][4];
    for (std::size_t r = 0; r < fresh.size() && r < 3; ++r)
        for (int c = 0; c < 4; ++c) tets[r][c] = fresh[r][c];
    return fresh.size() <= 3 &&
           tet_repair::closes(m, old.data(), static_cast<int>(old.size()), tets, static_cast<int>(fresh.size()));
}

// Whether the triangle between neighbours t and u fails the local Delaunay test.
inline bool fails(const Mesh& m, int t, int u) {
    const int far = tet_mesh::far_vertex(m, t, u);
    return far >= 0 && ep::in_sphere(m.point, detail::tet(m, t).data(), far);
}

}  // namespace detail

// Restore the Delaunay property everywhere: flips first, local rebuilds for
// what they cannot repair.
inline Stats repair(Mesh& m) {
    Stats stats;
    std::deque<int> queue;
    std::vector<char> queued(m.count, 0);
    for (int t = 0; t < m.count; ++t)
        if (m.alive(t)) {
            queue.push_back(t);
            queued[t] = 1;
        }
    auto push = [&](int t) {
        if (t >= static_cast<int>(queued.size())) queued.resize(m.count, 0);
        if (!queued[t]) {
            queue.push_back(t);
            queued[t] = 1;
        }
    };
    long long budget = 50 * static_cast<long long>(m.count) + 1000;
    std::vector<std::pair<int, int>> stuck;
    for (int round = 0; round < 8; ++round) {
        while (!queue.empty() && budget-- > 0) {
            const int t = queue.front();
            queue.pop_front();
            queued[t] = 0;
            if (!m.alive(t)) continue;
            const Tet T = detail::tet(m, t);
            // Turned inside out: flips usually untangle it, else rebuild below.
            if (ep::orient(m.point, T.data()) <= 0) stuck.emplace_back(t, -1);
            for (int k = 0; k < 4; ++k) {
                const int u = m.adjacent(t, k);
                if (u < 0) continue;
                const int far = tet_mesh::far_vertex(m, t, u);
                ++stats.checked;
                if (far < 0 || !ep::in_sphere(m.point, T.data(), far)) continue;
                const int a = T[k];  // apex of t opposite the shared face
                int face[3];
                tet_mesh::sorted_face(T.data(), k, face);
                // 2->3: the segment a-far crosses the shared face exactly when the
                // three tetrahedra (a, far, edge of the face) share one orientation.
                std::vector<Tet> three;
                int positive = 0, negative = 0;
                for (int e = 0; e < 3; ++e) {
                    const Tet n{a, far, face[e], face[(e + 1) % 3]};
                    const int o = ep::orient(m.point, n.data());
                    positive += o > 0;
                    negative += o < 0;
                    three.push_back(n);
                }
                if ((positive == 3 || negative == 3) && detail::closes(m, {t, u}, three)) {
                    for (int s : detail::replace(m, {t, u}, three)) push(s);
                    ++stats.flips23;
                    break;
                }
                // 3->2: an edge of the face with exactly three tetrahedra around it.
                bool done = false;
                for (int e = 0; e < 3 && !done; ++e) {
                    const int p = face[e], q = face[(e + 1) % 3];
                    // Walk around edge p-q until the walk closes.
                    std::vector<int> ring{t};
                    bool closed = false;
                    for (int current = t, previous = -1; ring.size() <= 3;) {
                        int next = -1;
                        for (int j = 0; j < 4; ++j) {
                            const int v = m.vertex(current, j);
                            if (v == p || v == q) continue;
                            const int w = m.adjacent(current, j);
                            if (w >= 0 && w != previous) {
                                next = w;
                                break;
                            }
                        }
                        if (next < 0) break;
                        if (next == t) {
                            closed = true;
                            break;
                        }
                        ring.push_back(next);
                        previous = current;
                        current = next;
                    }
                    if (!closed) continue;
                    if (ring.size() != 3 || std::find(ring.begin(), ring.end(), u) == ring.end()) continue;
                    // The three tetrahedra share edge p-q; their other vertices
                    // form the triangle of the two new tetrahedra.
                    std::vector<int> others;
                    for (int r : ring)
                        for (int j = 0; j < 4; ++j) {
                            const int v = m.vertex(r, j);
                            if (v != p && v != q && std::find(others.begin(), others.end(), v) == others.end())
                                others.push_back(v);
                        }
                    if (others.size() != 3) continue;
                    const std::vector<Tet> two{{p, others[0], others[1], others[2]}, {q, others[0], others[1], others[2]}};
                    if (ep::orient(m.point, two[0].data()) == 0 || ep::orient(m.point, two[1].data()) == 0) continue;
                    // Valid only if p and q lie on opposite sides of the new triangle.
                    const Tet side_p{others[0], others[1], others[2], p}, side_q{others[0], others[1], others[2], q};
                    if (ep::orient(m.point, side_p.data()) * ep::orient(m.point, side_q.data()) >= 0) continue;
                    if (!detail::closes(m, ring, two)) continue;
                    for (int s : detail::replace(m, ring, two)) push(s);
                    ++stats.flips32;
                    done = true;
                }
                if (done) break;
                ++stats.stuck;
                stuck.emplace_back(t, u);
            }
        }
        // Triangles no flip could repair: rebuild their surroundings.
        bool rebuilt = false;
        for (auto [t, u] : stuck) {
            if (!m.alive(t)) continue;
            if (u < 0) {
                if (ep::orient(m.point, detail::tet(m, t).data()) > 0) continue;
            } else {
                bool adjacent = false;
                for (int k = 0; k < 4; ++k) adjacent |= m.adjacent(t, k) == u;
                if (!m.alive(u) || !adjacent || !detail::fails(m, t, u)) continue;
            }
            const auto fresh = detail::rebuild(m, t, u);
            if (fresh.empty()) {
                ++stats.unrepaired;
                continue;
            }
            ++stats.rebuilt;
            rebuilt = true;
            for (int s : fresh) push(s);
        }
        stuck.clear();
        if (!rebuilt) break;
    }
    return stats;
}

// Number of inverted tetrahedra plus shared triangles that fail the local
// Delaunay test (each failing triangle counted from both sides).
inline long long failing(const Mesh& m) {
    long long bad = 0;
    for (int t = 0; t < m.count; ++t) {
        if (!m.alive(t)) continue;
        if (ep::orient(m.point, detail::tet(m, t).data()) <= 0) {
            ++bad;
            continue;
        }
        for (int k = 0; k < 4; ++k) {
            const int u = m.adjacent(t, k);
            bad += u >= 0 && detail::fails(m, t, u);
        }
    }
    return bad;
}

// A host copy of the device mesh and its repair state.
template <class ExecSpace>
Mesh to_host(const tet_mesh::Mesh<typename ExecSpace::memory_space>& dm,
             const tet_repair::State<typename ExecSpace::memory_space>& state) {
    auto copy = [](const auto& v) {
        auto h = Kokkos::create_mirror(Kokkos::HostSpace{}, v);  // a copy even on host backends
        Kokkos::deep_copy(h, v);
        return h;
    };
    return {copy(dm.vertex), copy(dm.adjacent), copy(state.alive), copy(state.point), state.count};
}

// Finish a device repair that left stuck entries: repair a host copy, copy it
// back, mark every tetrahedron it changed as touched, and give each edge
// around them one representative (the lowest-numbered tetrahedron around
// it). The device free list stays valid: the host only rewrites live
// tetrahedra and appends new ones.
template <class ExecSpace>
Stats finish(const ExecSpace& exec, const tet_mesh::Mesh<typename ExecSpace::memory_space>& dm,
             tet_repair::State<typename ExecSpace::memory_space>& state) {
    exec.fence();
    const Mesh before = to_host<ExecSpace>(dm, state);
    Mesh m = to_host<ExecSpace>(dm, state);
    const Stats stats = repair(m);
    auto touched = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, state.touched);
    auto edges = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dm.edges);
    std::vector<int> changed;
    for (int t = 0; t < m.count; ++t) {
        bool same = t < before.count && m.alive(t) == before.alive(t);
        for (int k = 0; k < 4 && same; ++k)
            same = m.vertex(t, k) == before.vertex(t, k) && m.adjacent(t, k) == before.adjacent(t, k);
        if (same) continue;
        touched(t) = 1;
        changed.push_back(t);
        if (!m.alive(t)) edges(t) = 0;
    }
    for (int t : changed) {
        if (!m.alive(t)) continue;
        for (int k = 0; k < 6; ++k) {
            int ka, kb;
            tet_mesh::edge_ends(k, ka, kb);
            const int a = m.vertex(t, ka), b = m.vertex(t, kb);
            int ring[65];
            const int length = std::min(tet_mesh::walk_edge(m, t, a, b, ring), 65);
            const int lowest = *std::min_element(ring, ring + std::max(length, 1));
            for (int r = 0; r < length; ++r) {
                const int u = ring[r];
                const int pa = tet_mesh::position(m, u, a), pb = tet_mesh::position(m, u, b);
                if (pa < 0 || pb < 0) continue;
                const int bit = 1 << tet_mesh::edge_index(pa, pb);
                edges(u) = u == lowest ? edges(u) | bit : edges(u) & ~bit;
            }
        }
    }
    Kokkos::deep_copy(exec, dm.vertex, m.vertex);
    Kokkos::deep_copy(exec, dm.adjacent, m.adjacent);
    Kokkos::deep_copy(exec, state.alive, m.alive);
    Kokkos::deep_copy(exec, state.touched, touched);
    Kokkos::deep_copy(exec, dm.edges, edges);
    exec.fence();
    state.count = m.count;
    return stats;
}

}  // namespace tet_flips
