#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <unordered_set>
#include <vector>

#include "exact_predicates.hpp"

// Host prototype: restore the Delaunay property of a tetrahedralization of
// periodic particles after they moved, by bistellar flips.
//
// A shared triangle is locally Delaunay when the far vertex of either
// tetrahedron lies outside the other's circumsphere. Failing triangles are
// repaired by a 2->3 flip when the two tetrahedra form a convex pair, or by
// a 3->2 flip when one edge of the triangle has exactly three tetrahedra
// around it; otherwise the triangle waits until neighbouring flips change its
// surroundings. New tetrahedra are checked in turn, until none fails.
//
// Every decision is an exact sign on snapped positions (see `exact`), so a
// tie is the same tie wherever it is evaluated.
namespace tet_flips {

using Point = std::array<double, 3>;
using Tet = std::array<int, 4>;

struct Mesh {
    std::vector<Tet> vertex;    // positively oriented
    std::vector<Tet> adjacent;  // across the face opposite vertex k, -1 if none
    std::vector<char> alive;
};

struct Stats {
    long long checked = 0, flips23 = 0, flips32 = 0, stuck = 0, rebuilt = 0, unrepaired = 0;
};

namespace exact = exact_predicates;

namespace detail {

inline double wrap(double d) { return d - std::nearbyint(d); }

// Offset of q from p in the periodic unit box.
inline Point offset(const Point& p, const Point& q) { return {wrap(q[0] - p[0]), wrap(q[1] - p[1]), wrap(q[2] - p[2])}; }

inline double dot(const Point& a, const Point& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline Point cross(const Point& a, const Point& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

// Exact sign of the orientation of tetrahedron t (snapped positions).
inline int orient_sign(const std::vector<exact::Offset>& g, const Tet& t) {
    auto d = [&](int v) {
        return exact::Offset{exact::delta(g[t[0]][0], g[v][0]), exact::delta(g[t[0]][1], g[v][1]),
                             exact::delta(g[t[0]][2], g[v][2])};
    };
    return exact::orient(d(t[1]), d(t[2]), d(t[3]));
}

// Whether e lies inside the circumsphere of t (exact, ties perturbed).
inline bool in_sphere(const std::vector<exact::Offset>& g, const Tet& t, int e) {
    auto d = [&](int v) {
        return exact::Offset{exact::delta(g[t[0]][0], g[v][0]), exact::delta(g[t[0]][1], g[v][1]),
                             exact::delta(g[t[0]][2], g[v][2])};
    };
    const exact::Offset p[5] = {d(t[0]), d(t[1]), d(t[2]), d(t[3]), d(e)};
    const int id[5] = {t[0], t[1], t[2], t[3], e};
    return exact::in_sphere(p, id);
}

inline std::array<int, 3> face_key(const Tet& t, int k) {
    std::array<int, 3> f;
    for (int a = 0, b = 0; a < 4; ++a)
        if (a != k) f[b++] = t[a];
    std::sort(f.begin(), f.end());
    return f;
}

inline int position(const Tet& t, int v) {
    for (int k = 0; k < 4; ++k)
        if (t[k] == v) return k;
    return -1;
}

}  // namespace detail

// Orient every tetrahedron positively and connect the faces.
using Grid = std::vector<exact::Offset>;

// Positions snapped to the exact predicates' grid.
inline Grid snap(const std::vector<Point>& x) {
    Grid g(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        for (int a = 0; a < 3; ++a) g[i][a] = exact::snap(x[i][a]);
    return g;
}

inline Mesh make_mesh(const Grid& x, std::vector<Tet> tets) {
    Mesh m;
    for (auto& t : tets)
        if (detail::orient_sign(x, t) < 0) std::swap(t[2], t[3]);
    m.vertex = std::move(tets);
    m.adjacent.assign(m.vertex.size(), {-1, -1, -1, -1});
    m.alive.assign(m.vertex.size(), 1);
    std::map<std::array<int, 3>, std::pair<int, int>> open;
    for (int t = 0; t < static_cast<int>(m.vertex.size()); ++t)
        for (int k = 0; k < 4; ++k) {
            const auto key = detail::face_key(m.vertex[t], k);
            auto it = open.find(key);
            if (it == open.end()) {
                open.emplace(key, std::make_pair(t, k));
            } else {
                m.adjacent[t][k] = it->second.first;
                m.adjacent[it->second.first][it->second.second] = t;
                open.erase(it);
            }
        }
    return m;
}

namespace detail {

// Replace the tetrahedra `old` by `fresh` (vertex sets, any orientation)
// covering the same region: orient the new ones, reuse the old slots first,
// and reconnect them to each other and to the outside through the old outer
// faces. Returns the new tetrahedra.
inline std::vector<int> replace(Mesh& m, const Grid& x, const std::vector<int>& old,
                                std::vector<Tet> fresh) {
    std::map<std::array<int, 3>, std::pair<int, int>> outside;  // outer face -> (outer tet, its face index)
    for (int t : old)
        for (int k = 0; k < 4; ++k) {
            const int o = m.adjacent[t][k];
            if (std::find(old.begin(), old.end(), o) != old.end()) continue;
            int back = -1;
            if (o >= 0)
                for (int j = 0; j < 4; ++j)
                    if (m.adjacent[o][j] == t) back = j;
            outside[face_key(m.vertex[t], k)] = {o, back};
        }
    std::vector<int> slots;
    for (std::size_t i = 0; i < fresh.size(); ++i) {
        int slot;
        if (i < old.size()) {
            slot = old[i];
        } else {
            slot = static_cast<int>(m.vertex.size());
            m.vertex.push_back({});
            m.adjacent.push_back({-1, -1, -1, -1});
            m.alive.push_back(1);
        }
        if (orient_sign(x, fresh[i]) < 0) std::swap(fresh[i][2], fresh[i][3]);
        m.vertex[slot] = fresh[i];
        m.adjacent[slot] = {-1, -1, -1, -1};
        m.alive[slot] = 1;
        slots.push_back(slot);
    }
    for (std::size_t i = fresh.size(); i < old.size(); ++i) m.alive[old[i]] = 0;
    std::map<std::array<int, 3>, std::pair<int, int>> inner;
    for (int s : slots)
        for (int k = 0; k < 4; ++k) {
            const auto key = face_key(m.vertex[s], k);
            if (auto it = outside.find(key); it != outside.end()) {
                m.adjacent[s][k] = it->second.first;
                if (it->second.first >= 0) m.adjacent[it->second.first][it->second.second] = s;
            } else if (auto jt = inner.find(key); jt != inner.end()) {
                m.adjacent[s][k] = jt->second.first;
                m.adjacent[jt->second.first][jt->second.second] = s;
            } else {
                inner.emplace(key, std::make_pair(s, k));
            }
        }
    return slots;
}

}  // namespace detail

namespace detail {

// Delaunay tetrahedralization of a few points in one local frame by
// Bowyer-Watson insertion into an enclosing tetrahedron, whose corners are
// points n..n+3. Insert near-to-far: each point is located by walking from
// the previous cavity, so the cost stays near linear. Returns the live
// tetrahedra, or nothing if point location fails.
inline std::vector<Tet> delaunay(std::vector<exact::Offset> p, std::vector<int> id) {
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
        return exact::orient(exact::minus(p[t[1]], p[t[0]]), exact::minus(p[t[2]], p[t[0]]),
                             exact::minus(p[t[3]], p[t[0]]));
    };
    auto inside = [&](const Tet& t, int q) {
        const exact::Offset five[5] = {p[t[0]], p[t[1]], p[t[2]], p[t[3]], p[q]};
        const int ids[5] = {id[t[0]], id[t[1]], id[t[2]], id[t[3]], id[q]};
        return exact::in_sphere(five, ids);
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
        for (int v : m.vertex[s])
            if (std::binary_search(seeds.begin(), seeds.end(), v)) return true;
        return false;
    };
    std::vector<int> found{t}, stack{t};
    std::unordered_set<int> seen{t};
    while (!stack.empty()) {
        const int s = stack.back();
        stack.pop_back();
        for (int o : m.adjacent[s])
            if (o >= 0 && seen.insert(o).second && touches(o)) {
                found.push_back(o);
                stack.push_back(o);
            }
    }
    return found;
}

inline std::vector<int> vertices_of(const Mesh& m, const std::vector<int>& tets) {
    std::vector<int> v;
    for (int t : tets) v.insert(v.end(), m.vertex[t].begin(), m.vertex[t].end());
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

// Fallback for a triangle t|u that no flip repairs, or an inverted
// tetrahedron t (u = -1): replace the stars of their particles by those of a
// fresh Delaunay tetrahedralization of the particles two rings around, if
// the new stars fit the old boundary exactly; otherwise widen the stars and
// retry. Returns the new tetrahedra.
inline std::vector<int> rebuild(Mesh& m, const Grid& x, int t, int u) {
    std::vector<int> seeds = vertices_of(m, u < 0 ? std::vector<int>{t} : std::vector<int>{t, u});
    for (int attempt = 0; attempt < 3; ++attempt) {
        const std::vector<int> old = stars(m, t, seeds);
        const std::vector<int> ring = vertices_of(m, old);
        const std::vector<int> points = vertices_of(m, stars(m, t, ring));
        // Local frame centred on the first seed, nearest first.
        const exact::Offset& o = x[seeds[0]];
        std::vector<std::pair<exact::i128, int>> order;
        std::vector<exact::Offset> local;
        for (int v : points) {
            const exact::Offset d{exact::delta(o[0], x[v][0]), exact::delta(o[1], x[v][1]), exact::delta(o[2], x[v][2])};
            order.push_back({exact::lift(d), v});
        }
        std::sort(order.begin(), order.end());
        std::vector<int> id;
        for (auto [r, v] : order) {
            local.push_back({exact::delta(o[0], x[v][0]), exact::delta(o[1], x[v][1]), exact::delta(o[2], x[v][2])});
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
                if (std::find(old.begin(), old.end(), m.adjacent[s][k]) == old.end()) ++boundary[face_key(m.vertex[s], k)];
        for (const Tet& s : fresh)
            for (int k = 0; k < 4; ++k) {
                const auto key = face_key(s, k);
                if (--boundary[key] == -2) boundary[key] = 0;  // inner face, seen twice
            }
        const bool fits = !fresh.empty() && !reaches_corner &&
                          std::all_of(boundary.begin(), boundary.end(), [](const auto& b) { return b.second == 0; });
        if (fits) return replace(m, x, old, fresh);
        seeds = ring;
    }
    return {};
}

}  // namespace detail

// Restore the Delaunay property everywhere; `x` are the moved positions.
inline Stats repair(Mesh& m, const Grid& x) {
    Stats stats;
    std::deque<int> queue;
    std::vector<char> queued(m.vertex.size(), 0);
    for (int t = 0; t < static_cast<int>(m.vertex.size()); ++t)
        if (m.alive[t]) {
            queue.push_back(t);
            queued[t] = 1;
        }
    auto push = [&](int t) {
        if (t >= static_cast<int>(queued.size())) queued.resize(m.vertex.size(), 0);
        if (!queued[t]) {
            queue.push_back(t);
            queued[t] = 1;
        }
    };
    long long budget = 50 * static_cast<long long>(m.vertex.size()) + 1000;
    std::vector<std::pair<int, int>> stuck;
    for (int round = 0; round < 8; ++round) {
    while (!queue.empty() && budget-- > 0) {
        const int t = queue.front();
        queue.pop_front();
        queued[t] = 0;
        if (!m.alive[t]) continue;
        // Turned inside out: flips usually untangle it, else rebuild below.
        if (detail::orient_sign(x, m.vertex[t]) <= 0) stuck.emplace_back(t, -1);
        for (int k = 0; k < 4; ++k) {
            const int u = m.adjacent[t][k];
            if (u < 0) continue;
            const Tet& T = m.vertex[t];
            const Tet& U = m.vertex[u];
            int far = -1;
            for (int v : U)
                if (detail::position(T, v) < 0) far = v;
            ++stats.checked;
            if (far < 0 || !detail::in_sphere(x, T, far)) continue;

            const int a = T[k];  // apex of t opposite the shared face
            const auto face = detail::face_key(T, k);
            // 2->3: the segment a-far crosses the shared face exactly when the
            // three tetrahedra (a, far, edge of the face) share one orientation.
            std::vector<Tet> three;
            int positive = 0, negative = 0;
            for (int e = 0; e < 3; ++e) {
                const Tet n{a, far, face[e], face[(e + 1) % 3]};
                const int o = detail::orient_sign(x, n);
                positive += o > 0;
                negative += o < 0;
                three.push_back(n);
            }
            const bool convex = positive == 3 || negative == 3;
            if (convex) {
                for (int s : detail::replace(m, x, {t, u}, three)) push(s);
                ++stats.flips23;
                break;
            }
            // 3->2: an edge of the face with exactly three tetrahedra around it.
            bool done = false;
            for (int e = 0; e < 3 && !done; ++e) {
                const int p = face[e], q = face[(e + 1) % 3];
                // Walk around edge p-q: from each tetrahedron step across a face
                // that contains the edge (opposite one of its other vertices),
                // never back, until the walk closes.
                std::vector<int> ring{t};
                bool closed = false;
                for (int current = t, previous = -1; ring.size() <= 3;) {
                    int next = -1;
                    for (int j = 0; j < 4; ++j) {
                        const int v = m.vertex[current][j];
                        if (v == p || v == q) continue;
                        const int w = m.adjacent[current][j];
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
                    for (int v : m.vertex[r])
                        if (v != p && v != q && std::find(others.begin(), others.end(), v) == others.end())
                            others.push_back(v);
                if (others.size() != 3) continue;
                const Tet n1{p, others[0], others[1], others[2]}, n2{q, others[0], others[1], others[2]};
                if (detail::orient_sign(x, n1) == 0 || detail::orient_sign(x, n2) == 0) continue;
                // Valid only if p and q lie on opposite sides of the new triangle.
                const Tet side_p{others[0], others[1], others[2], p}, side_q{others[0], others[1], others[2], q};
                if (detail::orient_sign(x, side_p) * detail::orient_sign(x, side_q) >= 0) continue;
                for (int s : detail::replace(m, x, ring, {n1, n2})) push(s);
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
        if (!m.alive[t]) continue;
        if (u < 0) {
            if (detail::orient_sign(x, m.vertex[t]) > 0) continue;
        } else {
            if (!m.alive[u] || std::find(m.adjacent[t].begin(), m.adjacent[t].end(), u) == m.adjacent[t].end())
                continue;
            int far = -1;
            for (int v : m.vertex[u])
                if (detail::position(m.vertex[t], v) < 0) far = v;
            if (far < 0 || !detail::in_sphere(x, m.vertex[t], far)) continue;
        }
        const auto fresh = detail::rebuild(m, x, t, u);
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

// Every particle's Delaunay neighbours (the cells it shares a face with,
// including faces of zero area).
inline std::vector<std::vector<int>> neighbours(const Mesh& m, int n) {
    std::vector<std::vector<int>> sets(n);
    for (std::size_t t = 0; t < m.vertex.size(); ++t)
        if (m.alive[t])
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b)
                    if (a != b) sets[m.vertex[t][a]].push_back(m.vertex[t][b]);
    for (auto& s : sets) {
        std::sort(s.begin(), s.end());
        s.erase(std::unique(s.begin(), s.end()), s.end());
    }
    return sets;
}

// Number of inverted tetrahedra plus shared triangles that still fail the
// local Delaunay test.
inline long long failing(const Mesh& m, const Grid& x) {
    long long bad = 0;
    for (std::size_t t = 0; t < m.vertex.size(); ++t) {
        if (!m.alive[t]) continue;
        if (detail::orient_sign(x, m.vertex[t]) <= 0) {
            ++bad;
            continue;
        }
        for (int k = 0; k < 4; ++k) {
            const int u = m.adjacent[t][k];
            if (u < 0) continue;
            int far = -1;
            for (int v : m.vertex[u])
                if (detail::position(m.vertex[t], v) < 0) far = v;
            bad += far >= 0 && detail::in_sphere(x, m.vertex[t], far);
        }
    }
    return bad;
}

}  // namespace tet_flips
