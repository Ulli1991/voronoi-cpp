#pragma once

// Brute-force oracle sharing no code with the library: exact 128-bit predicates and long double clipping over all
// periodic images, on data only. Checks voronoi::Mesh tetrahedra and cells; reached through mesh_check.hpp
// and, directly, mesh_cases_test.

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace oracle {

using i64 = long long;
__extension__ typedef __int128 i128;
using Offset = std::array<i64, 3>;

// Grid positions: integer steps in [0, period[a]) and the step's length.
struct Grid {
    std::vector<Offset> p;
    Offset period{};
    long double step = 0;
    int size() const { return static_cast<int>(p.size()); }
};

// d reduced to the nearest periodic image, in [-period/2, period/2). Used by: difference.
inline i64 nearest(i64 d, i64 period) {
    d %= period;
    if (2 * d >= period) d -= period;
    if (2 * d < -period) d += period;
    return d;
}

// Nearest-image offset in grid steps from particle `from` to `to`. Used by: check, voronoi_cell.
inline Offset difference(const Grid& g, int from, int to) {
    Offset d;
    for (int a = 0; a < 3; ++a) d[a] = nearest(g.p[to][a] - g.p[from][a], g.period[a]);
    return d;
}

// 3x3 determinant, exact in 128 bits. Used by: check (orientation), in_sphere.
inline i128 det(const Offset& a, const Offset& b, const Offset& c) {
    return a[0] * (i128(b[1]) * c[2] - i128(b[2]) * c[1]) + a[1] * (i128(b[2]) * c[0] - i128(b[0]) * c[2]) +
           a[2] * (i128(b[0]) * c[1] - i128(b[1]) * c[0]);
}

// Squared length |a|² (the lifting coordinate). Used by: in_sphere.
inline i128 lift(const Offset& a) { return i128(a[0]) * a[0] + i128(a[1]) * a[1] + i128(a[2]) * a[2]; }

// Sign of s[0] t[0] + ... + s[3] t[3] for |s| < 2⁶⁶, |t| < 2⁹⁸: each t split at 2⁴⁸ so both sums fit 128 bits.
inline int sign_of_sum(const i128* s, const i128* t) {
    const i128 unit = i128(1) << 48;
    i128 high = 0, low = 0;
    for (int k = 0; k < 4; ++k) {
        i128 th = t[k] / unit, tl = t[k] % unit;
        if (tl < 0) {
            tl += unit;
            th -= 1;
        }
        high += s[k] * th;
        low += s[k] * tl;
    }
    i128 carry = low / unit, rest = low % unit;
    if (rest < 0) {
        rest += unit;
        carry -= 1;
    }
    high += carry;
    if (high != 0) return high > 0 ? 1 : -1;
    return rest > 0 ? 1 : 0;
}

// +1 when e lies strictly inside the circumsphere of (0, a, b, c), -1 outside, 0 on it
// (a, b, c not coplanar); with circumcentre C = N / (2 det), inside iff |e|² < 2 e.C. Used by: check.
inline int in_sphere(const Offset& a, const Offset& b, const Offset& c, const Offset& e) {
    const i128 d = det(a, b, c);
    const i128 lifts[4] = {lift(a), lift(b), lift(c), -lift(e)};
    const i128 dets[4] = {det(e, b, c), det(a, e, c), det(a, b, e), d};
    return sign_of_sum(lifts, dets) * (d > 0 ? 1 : -1);
}

// The tetrahedra of a mesh, indexed as stored; dead ones have alive = 0.
struct Tets {
    std::vector<std::array<int, 4>> vertex, adjacent;
    std::vector<char> alive;
    int count() const { return static_cast<int>(vertex.size()); }
};

// Result of check(): counts of each defect (zero when ok()), box_volume if tetrahedra fill the box exactly.
struct Verdict {
    long long tets = 0, ambiguous = 0, flat = 0, inverted = 0, unpaired = 0, folded = 0, wrong_adjacency = 0, unused = 0,
              not_empty = 0, oversized = 0;
    bool box_volume = false;
    long double radius = 0;           // largest circumradius / shortest side
    std::array<int, 5> witness{-1, -1, -1, -1, -1};  // the first failing tetrahedron and particle
    Offset witness_image{};           // that particle's image, in grid steps
    bool ok() const {
        return ambiguous + flat + inverted + unpaired + folded + wrong_adjacency + unused + not_empty + oversized == 0 &&
               box_volume;
    }
    std::string text() const {
        std::ostringstream o;
        o << "tets=" << tets << " ambiguous=" << ambiguous << " flat=" << flat << " inverted=" << inverted
          << " unpaired=" << unpaired << " folded=" << folded << " adjacency=" << wrong_adjacency << " unused=" << unused
          << " box_volume=" << (box_volume ? "yes" : "NO") << " not_empty=" << not_empty << " oversized=" << oversized
          << " radius=" << double(radius);
        if (witness[0] >= 0)
            o << " witness=" << witness[0] << "," << witness[1] << "," << witness[2] << "," << witness[3] << ":" << witness[4]
              << "@" << witness_image[0] << "," << witness_image[1] << "," << witness_image[2];
        return o.str();
    }
};

// Uniform bins of the particles, to find the images near a sphere. Used by: check (empty-sphere test).
class Bins {
public:
    explicit Bins(const Grid& g) : grid_(g) {
        const double per_side = std::cbrt(double(std::max(g.size(), 1)));
        for (int a = 0; a < 3; ++a) side_[a] = std::max(1, std::min(256, int(per_side)));
        start_.assign(side_[0] * side_[1] * side_[2] + 1, 0);
        for (int i = 0; i < g.size(); ++i) ++start_[index(g.p[i]) + 1];
        for (std::size_t b = 1; b < start_.size(); ++b) start_[b] += start_[b - 1];
        items_.resize(g.size());
        std::vector<int> cursor(start_.begin(), start_.end() - 1);
        for (int i = 0; i < g.size(); ++i) items_[cursor[index(g.p[i])]++] = i;
    }

    // Calls f(particle, image position) for every image in bins overlapping [lo, hi]; stops on false.
    template <class F>
    bool visit(const std::array<long double, 3>& lo, const std::array<long double, 3>& hi, const F& f) const {
        std::array<long long, 3> first, last;
        for (int a = 0; a < 3; ++a) {
            const long double width = (long double)(grid_.period[a]) / side_[a];
            first[a] = static_cast<long long>(std::floor(lo[a] / width));
            last[a] = static_cast<long long>(std::floor(hi[a] / width));
        }
        for (long long i = first[0]; i <= last[0]; ++i)
            for (long long j = first[1]; j <= last[1]; ++j)
                for (long long k = first[2]; k <= last[2]; ++k) {
                    const std::array<long long, 3> c{i, j, k};
                    std::array<long long, 3> wrapped, shift;
                    for (int a = 0; a < 3; ++a) {
                        wrapped[a] = ((c[a] % side_[a]) + side_[a]) % side_[a];
                        shift[a] = (c[a] - wrapped[a]) / side_[a] * grid_.period[a];
                    }
                    const long long b = (wrapped[0] * side_[1] + wrapped[1]) * side_[2] + wrapped[2];
                    for (int q = start_[b]; q < start_[b + 1]; ++q) {
                        const int e = items_[q];
                        const Offset image{grid_.p[e][0] + shift[0], grid_.p[e][1] + shift[1], grid_.p[e][2] + shift[2]};
                        if (!f(e, image)) return false;
                    }
                }
        return true;
    }

private:
    int index(const Offset& p) const {
        int b = 0;
        for (int a = 0; a < 3; ++a) b = b * side_[a] + int((i128(p[a]) * side_[a]) / grid_.period[a]);
        return b;
    }
    const Grid& grid_;
    std::array<int, 3> side_;
    std::vector<int> start_, items_;
};

// Exact validation of a tetrahedralization: orientation, pairing and adjacency, every particle used, volumes
// summing to the box, and empty circumspheres over all periodic images. Used by: mesh_check::check, mesh_cases_test.
inline Verdict check(const Grid& g, const Tets& m) {
    Verdict v;
    struct Side {
        std::array<int, 3> key;
        int tet, corner;
        bool operator<(const Side& o) const { return key < o.key; }
    };
    std::vector<Side> sides;
    std::vector<char> used(g.size(), 0);
    i128 volume6 = 0;
    const Bins bins(g);
    // Half the box diagonal: a sphere this large covers the torus.
    long double covering2 = 0;
    for (int a = 0; a < 3; ++a) covering2 += 0.25L * (long double)(g.period[a]) * (long double)(g.period[a]);
    for (int t = 0; t < m.count(); ++t) {
        if (!m.alive[t]) continue;
        ++v.tets;
        const auto& w = m.vertex[t];
        const Offset a = difference(g, w[0], w[1]), b = difference(g, w[0], w[2]), c = difference(g, w[0], w[3]);
        for (int k = 0; k < 4; ++k) used[w[k]] = 1;
        // Corners agree on the images of each other.
        const Offset ab = difference(g, w[1], w[2]), ac = difference(g, w[1], w[3]), bc = difference(g, w[2], w[3]);
        bool consistent = true;
        for (int k = 0; k < 3; ++k)
            consistent = consistent && b[k] - a[k] == ab[k] && c[k] - a[k] == ac[k] && c[k] - b[k] == bc[k];
        v.ambiguous += !consistent;
        for (int k = 0; k < 4; ++k) {
            std::array<int, 3> key;
            for (int j = 0, q = 0; j < 4; ++j)
                if (j != k) key[q++] = w[j];
            std::sort(key.begin(), key.end());
            sides.push_back({key, t, k});
        }
        const i128 d = det(a, b, c);
        if (d == 0) {
            ++v.flat;
            continue;
        }
        v.inverted += d < 0;
        volume6 += d < 0 ? -d : d;

        // The long double circumcentre selects candidate images, the exact sign decides.
        const long double la = (long double)(lift(a)), lb = (long double)(lift(b)), lc = (long double)(lift(c));
        const long double dd = (long double)(d);
        std::array<long double, 3> centre;
        long double r2 = 0;
        const std::array<long double, 3> A{(long double)a[0], (long double)a[1], (long double)a[2]},
            B{(long double)b[0], (long double)b[1], (long double)b[2]},
            C{(long double)c[0], (long double)c[1], (long double)c[2]};
        for (int k = 0; k < 3; ++k) {
            const int i = (k + 1) % 3, j = (k + 2) % 3;
            centre[k] = (la * (B[i] * C[j] - B[j] * C[i]) + lb * (C[i] * A[j] - C[j] * A[i]) +
                         lc * (A[i] * B[j] - A[j] * B[i])) / (2 * dd);
            r2 += centre[k] * centre[k];
        }
        v.radius = std::max(v.radius, std::sqrt(r2) / (long double)(std::min({g.period[0], g.period[1], g.period[2]})));
        if (!(r2 <= covering2 * (1 + 1e-12L))) {
            ++v.oversized;
            continue;
        }
        const long double radius = std::sqrt(r2) * (1 + 1e-12L) + 2;
        std::array<long double, 3> lo, hi;
        for (int k = 0; k < 3; ++k) {
            lo[k] = (long double)(g.p[w[0]][k]) + centre[k] - radius;
            hi[k] = (long double)(g.p[w[0]][k]) + centre[k] + radius;
        }
        bool empty = true;
        bins.visit(lo, hi, [&](int particle, const Offset& image) {
            const Offset e{image[0] - g.p[w[0]][0], image[1] - g.p[w[0]][1], image[2] - g.p[w[0]][2]};
            long double d2 = 0;
            for (int k = 0; k < 3; ++k) d2 += ((long double)e[k] - centre[k]) * ((long double)e[k] - centre[k]);
            if (d2 > r2 * (1 + 1e-12L) + 1) return true;
            if (in_sphere(a, b, c, e) > 0) {
                empty = false;
                if (v.witness[0] < 0) {
                    v.witness = {w[0], w[1], w[2], w[3], particle};
                    v.witness_image = image;
                }
            }
            return empty;
        });
        v.not_empty += !empty;
    }
    std::sort(sides.begin(), sides.end());
    for (std::size_t s = 0; s < sides.size();) {
        std::size_t e = s;
        while (e < sides.size() && sides[e].key == sides[s].key) ++e;
        if (e - s != 2) {
            ++v.unpaired;
        } else {
            const auto& p = sides[s];
            const auto& q = sides[s + 1];
            v.wrong_adjacency += m.adjacent[p.tet][p.corner] != q.tet || m.adjacent[q.tet][q.corner] != p.tet;
            const int ap = m.vertex[p.tet][p.corner], aq = m.vertex[q.tet][q.corner];
            const Offset x = difference(g, p.key[0], p.key[1]), y = difference(g, p.key[0], p.key[2]),
                         e1 = difference(g, p.key[0], ap), e2 = difference(g, p.key[0], aq);
            const i128 d1 = det(x, y, e1), d2 = det(x, y, e2);
            v.folded += !((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0));
        }
        s = e;
    }
    v.unused = g.size() - std::count(used.begin(), used.end(), 1);
    v.box_volume = volume6 == 6 * i128(g.period[0]) * g.period[1] * g.period[2];
    return v;
}

// Two meshes of one point set are the same mesh when their tetrahedra, as
// sorted quadruples, are equal. Returns how many quadruples are in only one.
// Used by: mesh_check::check (fresh build), mesh_cases_test.
inline long long different_tetrahedra(const Tets& a, const Tets& b) {
    auto quads = [](const Tets& m) {
        std::vector<std::array<int, 4>> q;
        for (int t = 0; t < m.count(); ++t)
            if (m.alive[t]) {
                q.push_back(m.vertex[t]);
                std::sort(q.back().begin(), q.back().end());
            }
        std::sort(q.begin(), q.end());
        return q;
    };
    const auto x = quads(a), y = quads(b);
    std::vector<std::array<int, 4>> only;
    std::set_symmetric_difference(x.begin(), x.end(), y.begin(), y.end(), std::back_inserter(only));
    return static_cast<long long>(only.size());
}

// One Voronoi cell by half-space clipping, in long double grid steps relative to its generator.
using Vec = std::array<long double, 3>;
// Small vector helpers for the clipping below.
inline Vec minus(const Vec& a, const Vec& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline long double dot(const Vec& a, const Vec& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline Vec cross(const Vec& a, const Vec& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

// One face of a cell: a polygon of corners, the neighbour particle and its image.
struct Polygon {
    std::vector<Vec> corners;
    int neighbour;  // -1: a side of the starting box (the particle's own image)
    int image = 0;  // numbers the periodic image of the neighbour
};

// Outward area vector of a polygon. Used by: voronoi_cell.
inline Vec area_vector(const Polygon& f) {
    Vec a{0, 0, 0};
    for (std::size_t k = 1; k + 1 < f.corners.size(); ++k) {
        const Vec c = cross(minus(f.corners[k], f.corners[0]), minus(f.corners[k + 1], f.corners[0]));
        for (int d = 0; d < 3; ++d) a[d] += c[d] / 2;
    }
    return a;
}

// A finished cell: faces, volume, moment, reach and per-neighbour area vectors. Used by: mesh_check::check and arbitrate, mesh_cases_test.
struct Cell {
    std::vector<Polygon> faces;
    long double volume = 0;  // grid steps^3
    Vec moment{0, 0, 0};     // integral of the position over the cell
    long double reach = 0;   // largest distance from the generator to a corner
    // Outward area vector (grid steps²) per neighbour, images summed, and the image count; own images under its own id.
    std::map<int, Vec> area;
    std::map<int, int> images;
};

// Cut the polyhedron by the half space x.q <= |q|^2 / 2. Used by: voronoi_cell.
inline void clip(std::vector<Polygon>& faces, const Vec& q, int neighbour, int image) {
    const long double limit = dot(q, q) / 2;
    std::vector<Vec> cut;
    std::vector<Polygon> kept;
    for (const Polygon& f : faces) {
        Polygon out{{}, f.neighbour, f.image};
        const int m = static_cast<int>(f.corners.size());
        for (int i = 0; i < m; ++i) {
            const Vec& p = f.corners[i];
            const Vec& r = f.corners[(i + 1) % m];
            const long double dp = dot(p, q) - limit, dr = dot(r, q) - limit;
            if (dp <= 0) out.corners.push_back(p);
            if ((dp < 0 && dr > 0) || (dp > 0 && dr < 0)) {
                const long double s = dp / (dp - dr);
                const Vec x{p[0] + s * (r[0] - p[0]), p[1] + s * (r[1] - p[1]), p[2] + s * (r[2] - p[2])};
                out.corners.push_back(x);
                cut.push_back(x);
            } else if (dp == 0) {
                cut.push_back(p);
            }
        }
        if (out.corners.size() >= 3) kept.push_back(std::move(out));
    }
    if (cut.size() >= 3) {  // the new face: the cut points by angle around their mean
        Vec mean{0, 0, 0};
        for (const Vec& c : cut)
            for (int d = 0; d < 3; ++d) mean[d] += c[d] / cut.size();
        const long double norm = std::sqrt(dot(q, q));
        const Vec n{q[0] / norm, q[1] / norm, q[2] / norm};
        Vec u = cross(n, std::abs(n[0]) < 0.6L ? Vec{1, 0, 0} : Vec{0, 1, 0});
        const long double ul = std::sqrt(dot(u, u));
        for (auto& c : u) c /= ul;
        const Vec v = cross(n, u);
        std::sort(cut.begin(), cut.end(), [&](const Vec& a, const Vec& b) {
            return std::atan2(dot(minus(a, mean), v), dot(minus(a, mean), u)) <
                   std::atan2(dot(minus(b, mean), v), dot(minus(b, mean), u));
        });
        cut.erase(std::unique(cut.begin(), cut.end()), cut.end());
        if (cut.size() >= 3) kept.push_back({std::move(cut), neighbour, image});
    }
    faces = std::move(kept);
}

// Largest distance from the generator to a corner; bounds which images can still cut. Used by: voronoi_cell.
inline long double reach_of(const std::vector<Polygon>& faces) {
    long double r2 = 0;
    for (const Polygon& f : faces)
        for (const Vec& c : f.corners) r2 = std::max(r2, dot(c, c));
    return std::sqrt(r2);
}

// The cell of particle i: the box around it cut by every other particle in every image that can reach it.
// A duplicate of i has no bisector and is skipped, so duplicates each get the whole shared cell.
// Used by: mesh_check::check and arbitrate, mesh_cases_test.
inline Cell voronoi_cell(const Grid& g, int i) {
    const Vec half{g.period[0] / 2.0L, g.period[1] / 2.0L, g.period[2] / 2.0L};
    std::vector<Polygon> faces;
    for (int a = 0; a < 3; ++a)
        for (const int s : {-1, 1}) {
            const int b = (a + 1) % 3, c = (a + 2) % 3;
            Polygon f{{}, -1};
            for (const auto& [ub, uc] : {std::pair{-1, -1}, std::pair{1, -1}, std::pair{1, 1}, std::pair{-1, 1}}) {
                Vec p;
                p[a] = s * half[a];
                p[b] = ub * half[b];
                p[c] = uc * half[c];
                f.corners.push_back(p);
            }
            if (s < 0) std::reverse(f.corners.begin(), f.corners.end());
            faces.push_back(std::move(f));
        }
    auto as_vec = [](const Offset& d) { return Vec{(long double)d[0], (long double)d[1], (long double)d[2]}; };
    auto image_number = [](int u, int v, int w) { return (u + 2) * 25 + (v + 2) * 5 + (w + 2); };
    // Nearest images first, nearest particles first: the cell shrinks fast.
    std::vector<std::pair<long double, int>> order;
    order.reserve(g.size());
    for (int j = 0; j < g.size(); ++j) {
        const Vec d = as_vec(difference(g, i, j));
        if (j != i && dot(d, d) > 0) order.push_back({dot(d, d), j});
    }
    std::sort(order.begin(), order.end());
    for (const auto& [d2, j] : order) {
        const long double r = reach_of(faces);
        if (d2 > 4 * r * r) break;
        clip(faces, as_vec(difference(g, i, j)), j, image_number(0, 0, 0));
    }
    // Every other image, and the particle's own beyond the sides of the box.
    const long double r = reach_of(faces);
    for (int j = 0; j < g.size(); ++j) {
        const Offset d = difference(g, i, j);
        if (j != i && d[0] == 0 && d[1] == 0 && d[2] == 0) continue;
        for (int u = -2; u <= 2; ++u)
            for (int v = -2; v <= 2; ++v)
                for (int w = -2; w <= 2; ++w) {
                    if (u == 0 && v == 0 && w == 0) continue;
                    if (j == i && std::abs(u) + std::abs(v) + std::abs(w) == 1) continue;
                    const Vec q{(long double)(d[0] + u * g.period[0]), (long double)(d[1] + v * g.period[1]),
                                (long double)(d[2] + w * g.period[2])};
                    if (dot(q, q) <= 4 * r * r) clip(faces, q, j, image_number(u, v, w));
                }
    }
    Cell cell;
    for (const Polygon& f : faces) {
        const Vec a = area_vector(f);
        cell.volume += dot(a, f.corners[0]) / 3;
        // The moment from a fan of tetrahedra over the face.
        for (std::size_t k = 1; k + 1 < f.corners.size(); ++k) {
            const Vec &p = f.corners[0], &q = f.corners[k], &s = f.corners[k + 1];
            const long double six = dot(p, cross(q, s));
            for (int d = 0; d < 3; ++d) cell.moment[d] += six / 24 * (p[d] + q[d] + s[d]);
        }
        const int key = f.neighbour >= 0 ? f.neighbour : i;
        auto& sum = cell.area[key];
        for (int d = 0; d < 3; ++d) sum[d] += a[d];
        ++cell.images[key];
    }
    cell.reach = reach_of(faces);
    cell.faces = std::move(faces);
    return cell;
}

}  // namespace oracle
