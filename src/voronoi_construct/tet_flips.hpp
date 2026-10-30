#pragma once

#include "tet_repair.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <map>
#include <unordered_map>
#include <stdexcept>
#include <unordered_set>
#include <vector>

// Host fallback of tet_repair: restores the Delaunay property where the
// device repair got stuck.
//
// A shared triangle is locally Delaunay when the far vertex of either
// tetrahedron lies outside the other's circumsphere. The device repair
// flips failing triangles while a flip is valid and closes its cavity; the
// host rebuilds the surroundings of those it left (Bowyer-Watson on the
// particles around them), and the device flips again. Every decision is an
// exact sign on the snapped positions shared with the device, so a tie is
// the same tie everywhere.
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
    long long flips23 = 0, flips32 = 0, rebuilt = 0;  // device flips and local rebuilds
    int stuck = 0;                                    // entries finish() left
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

// Whether the triangle between neighbours t and u fails the local Delaunay test.
inline bool fails(const Mesh& m, int t, int u) {
    const int far = tet_mesh::far_vertex(m, t, u);
    return far >= 0 && ep::in_sphere(m.point, detail::tet(m, t).data(), far);
}

}  // namespace detail

// Rebuild the surroundings of every inverted tetrahedron and failing
// triangle of m locally, in up to eight passes while rebuilds succeed (the
// device repair has done every flip it could before, and flips what the
// rebuilds leave).
inline Stats repair(Mesh& m) {
    Stats stats;
    for (long long pass = 0, before = -1; pass < 8 && stats.rebuilt > before; ++pass) {
        before = stats.rebuilt;
        for (int t = 0; t < m.count; ++t) {
            if (!m.alive(t)) continue;
            int u = -1;  // the failing triangle t|u, or t inverted
            if (ep::orient(m.point, detail::tet(m, t).data()) > 0) {
                for (int k = 0; k < 4 && u < 0; ++k)
                    if (m.adjacent(t, k) >= 0 && detail::fails(m, t, m.adjacent(t, k))) u = m.adjacent(t, k);
                if (u < 0) continue;
            }
            stats.rebuilt += !detail::rebuild(m, t, u).empty();  // else stuck: the device lists it again
        }
    }
    return stats;
}

namespace detail {

// Mark the tetrahedra of stuck entry i dirty (both sides of a triangle).
template <class State, class Adjacent>
KOKKOS_INLINE_FUNCTION void mark_stuck(const State& st, const Adjacent& adjacent, int i) {
    const int t = st.stuck(i, 0), k = st.stuck(i, 1);
    if (t < 0 || !st.alive(t)) return;
    tet_repair::detail::mark(st, t);
    const int u = k >= 0 ? adjacent(t, k) : -1;
    if (u >= 0 && st.alive(u)) tet_repair::detail::mark(st, u);
}

// The live tetrahedra within `hops` adjacency steps of the stuck entries, in
// state.listed[0, returned); state.dirty is reset for them afterwards.
template <class ExecSpace>
int stuck_region(const ExecSpace& exec, const tet_mesh::Mesh<typename ExecSpace::memory_space>& dm,
                 tet_repair::State<typename ExecSpace::memory_space>& state, int stuck, int hops) {
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const auto& st = state;
    const auto adjacent = dm.adjacent;
    auto listed_count = [&] {
        Kokkos::deep_copy(exec, state.host_counters, state.counters);
        exec.fence();
        return state.host_counters(tet_repair::counter::listed);
    };
    Kokkos::deep_copy(exec, Kokkos::subview(state.counters, tet_repair::counter::listed), 0);
    Kokkos::parallel_for(
        "tet_flips::region_seeds", Range(exec, 0, stuck), KOKKOS_LAMBDA(int i) { mark_stuck(st, adjacent, i); });
    int begin = 0, end = listed_count();
    for (int h = 0; h < hops && end > begin; ++h) {
        Kokkos::parallel_for(
            "tet_flips::region_grow", Range(exec, begin, end), KOKKOS_LAMBDA(int i) {
                const int t = st.listed(i);
                for (int k = 0; k < 4; ++k) {
                    const int u = adjacent(t, k);
                    if (u >= 0 && st.alive(u)) tet_repair::detail::mark(st, u);
                }
            });
        begin = end;
        end = listed_count();
    }
    Kokkos::parallel_for(
        "tet_flips::region_unmark", Range(exec, 0, end), KOKKOS_LAMBDA(int i) { st.dirty(st.listed(i)) = 0; });
    return end;
}

}  // namespace detail

// Finish a device repair that left stuck entries. The tetrahedra within a
// few adjacency steps of them are copied to the host as a local mesh (its
// own particle numbering, in the same order, so every exact decision is the
// same; faces to the rest of the mesh are open), repaired there, and the
// changed and new ones written back; new ones are appended after
// state.count, so the device free list stays valid. Every tetrahedron the
// host changed is touched and listed, each edge around them gets one
// representative (the lowest-numbered tetrahedron around it; a tetrahedron
// whose edge bits change is touched too), and the device repair runs again
// from them and the stuck entries, to settle the region's border and to
// find what the host could not repair. If that still leaves stuck entries
// the region grows once (8, then 16 steps); what is left after that is
// stats.stuck (voronoi::Mesh then rebuilds).
template <class ExecSpace>
Stats finish(const ExecSpace& exec, tet_mesh::Mesh<typename ExecSpace::memory_space>& dm,
             tet_repair::State<typename ExecSpace::memory_space>& state) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    Stats total;
    // The entries of the last device repair (its counters are on the host).
    total.stuck = Kokkos::min(state.host_counters(tet_repair::counter::stuck), static_cast<int>(state.stuck.extent(0)));
    for (int hops = 8; hops <= 16 && total.stuck > 0; hops *= 2) {
        const int stuck = total.stuck;
        const int size = detail::stuck_region(exec, dm, state, stuck, hops);
        // The region's rows and particles to the host.
        Kokkos::View<int* [8], Memory> rows(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("region_rows")),
                                            size);
        const auto listed = state.listed;
        Kokkos::parallel_for(
            "tet_flips::region_rows", Range(exec, 0, size), KOKKOS_LAMBDA(int i) {
                const int t = listed(i);
                for (int k = 0; k < 4; ++k) {
                    rows(i, k) = dm.vertex(t, k);
                    rows(i, 4 + k) = dm.adjacent(t, k);
                }
            });
        const auto host_rows = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rows);
        const auto host_listed = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, Kokkos::subview(listed, std::make_pair(0, size)));
        std::vector<int> particle;  // local particle -> global, increasing
        for (int i = 0; i < size; ++i)
            for (int k = 0; k < 4; ++k) particle.push_back(host_rows(i, k));
        std::sort(particle.begin(), particle.end());
        particle.erase(std::unique(particle.begin(), particle.end()), particle.end());
        auto local_particle = [&](int v) {
            return static_cast<int>(std::lower_bound(particle.begin(), particle.end(), v) - particle.begin());
        };
        Kokkos::View<int*, Memory> particle_ids("region_particles", particle.size());
        Kokkos::deep_copy(exec, particle_ids,
                          Kokkos::View<const int*, Kokkos::HostSpace>(particle.data(), particle.size()));
        ep::Points<Memory> points("region_points", particle.size());
        const auto point = state.point;
        Kokkos::parallel_for(
            "tet_flips::region_points", Range(exec, 0, static_cast<int>(particle.size())), KOKKOS_LAMBDA(int i) {
                for (int a = 0; a < 3; ++a) points(i, a) = point(particle_ids(i), a);
            });

        // The local mesh; faces to tetrahedra outside the region are open (-1).
        std::unordered_map<int, int> local;
        for (int i = 0; i < size; ++i) local[host_listed(i)] = i;
        const int capacity = 2 * size + 1024;
        Mesh m{tet_mesh::Quads<Kokkos::HostSpace>("region_vertex", capacity),
               tet_mesh::Quads<Kokkos::HostSpace>("region_adjacent", capacity),
               Kokkos::View<char*, Kokkos::HostSpace>("region_alive", capacity),
               Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, points), size};
        std::map<std::array<int, 3>, int> outside;  // open face (local particles) -> outside tetrahedron
        for (int i = 0; i < size; ++i) {
            m.alive(i) = 1;
            for (int k = 0; k < 4; ++k) m.vertex(i, k) = local_particle(host_rows(i, k));
            for (int k = 0; k < 4; ++k) {
                const int u = host_rows(i, 4 + k);
                const auto it = local.find(u);
                m.adjacent(i, k) = it == local.end() ? -1 : it->second;
                if (it == local.end()) outside[detail::face_key(detail::tet(m, i), k)] = u;
            }
        }
        total.rebuilt += repair(m).rebuilt;

        // Changed and new tetrahedra back, with global numbers.
        const int added = m.count - size;
        if (state.count + added > static_cast<int>(state.alive.extent(0)))
            tet_repair::reserve(dm, state, state.count + added + (state.count + added) / 64);
        auto global = [&](int l) { return l < size ? host_listed(l) : state.count + l - size; };
        std::vector<int> changed;  // rows: slot, alive, 4 vertices, 4 neighbours, open-face bits
        for (int l = 0; l < m.count; ++l) {
            bool same = l < size && m.alive(l);
            for (int k = 0; k < 4 && same; ++k)
                same = m.vertex(l, k) == local_particle(host_rows(l, k)) &&
                       (m.adjacent(l, k) >= 0 ? global(m.adjacent(l, k)) == host_rows(l, 4 + k)
                                              : local.count(host_rows(l, 4 + k)) == 0);
            if (same) continue;
            int open = 0, neighbour[4];
            for (int k = 0; k < 4; ++k) {
                neighbour[k] = m.adjacent(l, k) >= 0 ? global(m.adjacent(l, k)) : -1;
                if (m.alive(l) && m.adjacent(l, k) < 0) {
                    // Host flips and rebuilds keep the region's open faces.
                    const auto it = outside.find(detail::face_key(detail::tet(m, l), k));
                    if (it == outside.end()) throw std::logic_error("tet_flips::finish: a region lost its border");
                    neighbour[k] = it->second;
                    open |= 1 << k;
                }
            }
            changed.insert(changed.end(), {global(l), m.alive(l), particle[m.vertex(l, 0)], particle[m.vertex(l, 1)],
                                           particle[m.vertex(l, 2)], particle[m.vertex(l, 3)], neighbour[0],
                                           neighbour[1], neighbour[2], neighbour[3], open});
        }
        const int count = static_cast<int>(changed.size()) / 11;
        Kokkos::View<int* [11], Kokkos::LayoutRight, Memory> update("region_update", count);
        Kokkos::deep_copy(exec, update,
                          Kokkos::View<const int* [11], Kokkos::HostSpace>(changed.data(), count));
        state.count += added;
        const auto vertex = dm.vertex, adjacent = dm.adjacent;
        const auto edges = dm.edges;
        const auto alive = state.alive, touched = state.touched;
        // Every tetrahedron touched here is listed in touched_list too
        // (the created count past its capacity: the list is unknown).
        const auto list = state.touched_list, counters = state.counters;
        const int list_capacity = static_cast<int>(list.extent(0));
        Kokkos::deep_copy(exec, Kokkos::subview(counters, tet_repair::counter::created),
                          state.touched_entries >= 0 ? state.touched_entries : list_capacity + 1);
        auto touch = KOKKOS_LAMBDA(int t) {
            touched(t) = 1;
            const int at = Kokkos::atomic_fetch_add(&counters(tet_repair::counter::created), 1);
            if (at < list_capacity) list(at) = t;
        };
        Kokkos::parallel_for(
            "tet_flips::region_write", Range(exec, 0, count), KOKKOS_LAMBDA(int i) {
                const int t = update(i, 0);
                alive(t) = static_cast<char>(update(i, 1));
                touch(t);
                edges(t) = 0;
                for (int k = 0; k < 4; ++k) {
                    vertex(t, k) = update(i, 2 + k);
                    adjacent(t, k) = update(i, 6 + k);
                }
            });
        // Outside neighbours point back to the new tetrahedra across the open faces.
        Kokkos::parallel_for(
            "tet_flips::region_border", Range(exec, 0, count), KOKKOS_LAMBDA(int i) {
                const int t = update(i, 0), open = update(i, 10);
                const int v[4] = {vertex(t, 0), vertex(t, 1), vertex(t, 2), vertex(t, 3)};
                for (int k = 0; k < 4; ++k) {
                    if (!(open >> k & 1)) continue;
                    const int u = adjacent(t, k);
                    int key[3];
                    tet_mesh::sorted_face(v, k, key);
                    adjacent(u, tet_mesh::opposite(dm, u, key)) = t;
                }
            });
        // One representative per edge around the changed tetrahedra.
        Kokkos::parallel_for(
            "tet_flips::region_edges", Range(exec, 0, count), KOKKOS_LAMBDA(int i) {
                const int t = update(i, 0);
                if (!update(i, 1)) return;
                for (int k = 0; k < 6; ++k) {
                    int ka, kb;
                    tet_mesh::edge_ends(k, ka, kb);
                    const int a = vertex(t, ka), b = vertex(t, kb);
                    int ring[tet_mesh::max_ring_length + 1];
                    const int length = Kokkos::min(tet_mesh::walk_edge(dm, t, a, b, ring), tet_mesh::max_ring_length + 1);
                    int lowest = t;
                    for (int r = 0; r < length; ++r) lowest = Kokkos::min(lowest, ring[r]);
                    for (int r = 0; r < length; ++r) {
                        const int u = ring[r];
                        const int pa = tet_mesh::position(dm, u, a), pb = tet_mesh::position(dm, u, b);
                        if (pa < 0 || pb < 0) continue;
                        const auto bit = static_cast<std::uint8_t>(1 << tet_mesh::edge_index(pa, pb));
                        const int before = u == lowest ? Kokkos::atomic_fetch_or(&edges(u), bit)
                                                       : Kokkos::atomic_fetch_and(&edges(u), static_cast<std::uint8_t>(~bit));
                        if ((before & bit) != (u == lowest ? bit : 0)) touch(u);  // renumbered faces
                    }
                }
            });
        // Repair on the device from the changed ones and the stuck ones (what
        // the host could not repair is stuck again).
        Kokkos::deep_copy(exec, Kokkos::subview(counters, tet_repair::counter::listed), 0);
        const auto& st = state;
        Kokkos::parallel_for(
            "tet_flips::region_dirty", Range(exec, 0, count), KOKKOS_LAMBDA(int i) {
                if (update(i, 1)) tet_repair::detail::mark(st, update(i, 0));
            });
        Kokkos::parallel_for(
            "tet_flips::region_recheck", Range(exec, 0, stuck),
            KOKKOS_LAMBDA(int i) { detail::mark_stuck(st, adjacent, i); });
        const tet_repair::Stats device = tet_repair::repair(exec, dm, state);
        total.flips23 += device.flips23;
        total.flips32 += device.flips32;
        total.stuck = device.stuck;
    }
    return total;
}

}  // namespace tet_flips
