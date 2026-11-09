#pragma once

// Random mesh cases for mesh_fuzz_test: one integer seed gives a Script, a plain list of positions and API calls that can be
// saved, cut down and replayed by itself. Host only, the same on every backend; a seed's high digits choose the profile (`profile_of`).

#include "exact_predicates.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mesh_fuzz {

using Point = std::array<double, 3>;
using Points = std::vector<Point>;
using Box = exact_predicates::Box;

// xoshiro256** seeded by splitmix64: the same stream with every compiler.
class Rng {
public:
    explicit Rng(std::uint64_t seed) {
        for (auto& w : s_) {
            seed += 0x9e3779b97f4a7c15ULL;
            std::uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            w = z ^ (z >> 31);
        }
    }
    std::uint64_t next() {
        const std::uint64_t r = rotl(s_[1] * 5, 7) * 9, t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return r;
    }
    double unit() { return static_cast<double>(next() >> 11) * 0x1p-53; }
    int below(int k) { return static_cast<int>(unit() * k); }
    int range(int low, int high) { return low + below(high - low + 1); }  // both ends
    bool chance(double p) { return unit() < p; }
    double between(double low, double high) { return low + (high - low) * unit(); }
    double log_between(double low, double high) { return std::exp(between(std::log(low), std::log(high))); }
    double normal() {
        const double u = 1 - unit(), v = unit();
        return std::sqrt(-2 * std::log(u)) * std::cos(2 * M_PI * v);
    }
    Point direction() {
        Point d{normal(), normal(), normal()};
        const double r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        for (double& c : d) c /= r > 0 ? r : 1;
        return d;
    }
    template <class T>
    void shuffle(std::vector<T>& v) {
        for (int i = static_cast<int>(v.size()) - 1; i > 0; --i) std::swap(v[i], v[below(i + 1)]);
    }

private:
    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    std::uint64_t s_[4];
};

// One call on the mesh. A Script holds no randomness.
struct Op {
    enum Kind {
        update,    // update(x)
        rebuild,   // rebuild(x)
        cells,     // update_cells(), then the lists and centroids are read and checked
        clip,      // clip(x, list), checked against the oracle
        fallback,  // set_fallback(arg): 0 cheapest, 1 host, 2 rebuild
        rounds,    // set_repair_rounds(arg)
        scribble,  // scratch(arg bytes), every byte overwritten
        timed,     // the next update or rebuild passes Timings
        kinds
    };
    int kind = update;
    int arg = 0;
    std::vector<int> list;  // clip
    Points x;               // update, rebuild, clip
};

inline const char* op_name(int kind) {
    static const char* const names[Op::kinds] = {"update",   "rebuild", "cells",    "clip",
                                                 "fallback", "rounds",  "scribble", "timed"};
    return names[kind];
}

// Which Mesh a script runs on.
enum Variant { plain, float_rows, float_weights, double_weights, variants };

struct Script {
    std::string label;
    std::array<double, 3> sides{1, 1, 1};
    int variant = plain;
    std::vector<Op> ops;
    int particles() const {
        int n = 0;
        for (const Op& o : ops) n = std::max(n, static_cast<int>(o.x.size()));
        return n;
    }
    long long size() const {
        long long s = 0;
        for (const Op& o : ops) s += 1 + static_cast<long long>(o.x.size());
        return s;
    }
};

// Text form, doubles in hexadecimal: the exact bits.
inline void save(const Script& s, std::ostream& out) {
    char number[64];
    auto hex = [&](double v) {
        std::snprintf(number, sizeof number, "%a", v);
        return std::string(number);
    };
    out << "label " << s.label << "\nsides " << hex(s.sides[0]) << " " << hex(s.sides[1]) << " " << hex(s.sides[2])
        << "\nvariant " << s.variant << "\nops " << s.ops.size() << "\n";
    for (const Op& o : s.ops) {
        out << op_name(o.kind) << " " << o.arg << " " << o.list.size() << " " << o.x.size() << "\n";
        for (const int i : o.list) out << i << " ";
        if (!o.list.empty()) out << "\n";
        for (const Point& p : o.x) out << hex(p[0]) << " " << hex(p[1]) << " " << hex(p[2]) << "\n";
    }
}

inline Script load(std::istream& in) {
    Script s;
    std::string word, a, b, c;
    auto number = [](const std::string& t) { return std::strtod(t.c_str(), nullptr); };
    while (in >> word && word != "ops") {
        if (word == "label") {
            std::getline(in, s.label);
            if (!s.label.empty() && s.label[0] == ' ') s.label.erase(0, 1);
        } else if (word == "sides") {
            in >> a >> b >> c;
            s.sides = {number(a), number(b), number(c)};
        } else if (word == "variant") {
            in >> s.variant;
        } else {  // comments, such as the issue of a saved case
            std::getline(in, a);
        }
    }
    std::size_t ops = 0;
    in >> ops;
    s.ops.resize(ops);
    for (Op& o : s.ops) {
        std::size_t listed = 0, points = 0;
        in >> word >> o.arg >> listed >> points;
        o.kind = -1;
        for (int k = 0; k < Op::kinds; ++k)
            if (word == op_name(k)) o.kind = k;
        if (o.kind < 0) throw std::runtime_error("mesh_fuzz::load: unknown call " + word);
        o.list.resize(listed);
        for (int& i : o.list) in >> i;
        o.x.resize(points);
        for (Point& p : o.x) {
            in >> a >> b >> c;
            p = {number(a), number(b), number(c)};
        }
    }
    if (!in) throw std::runtime_error("mesh_fuzz::load: truncated case");
    return s;
}

inline void save(const Script& s, const std::string& path) {
    std::ofstream out(path);
    save(s, out);
}

inline Script load(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("mesh_fuzz::load: cannot read " + path);
    return load(in);
}

// ------------------------------------------------------------------ profiles

// A seed's digits above 10^12 choose the profile, so the one number is the case.
struct Profile {
    const char* name;
    int tiny, fewest, most;  // n: below `fewest` with chance 1 / tiny; else log uniform in [fewest, most]
    int steps;               // at most
};
inline Profile profile_of(std::uint64_t seed) {
    static const Profile profiles[] = {
        {"small", 8, 600, 2000, 6},                                                    // the long CPU runs
        {"api", 16, 600, 1000, 16},                                                    // long call sequences
        {"medium", 32, 2000, 32768, 4},                                                // long CPU runs and the A100
        {"large", 1000000, 32768, 262144, 3}, {"huge", 1000000, 1000000, 4000000, 2},  // the A100
    };
    return profiles[std::min<std::uint64_t>(seed / 1000000000000ULL, 4)];
}

// ---------------------------------------------------------------- point sets

// Fractions of the box in [0, 1). `spacing` of a lattice of about n points.
inline double frac(double x) { return x - std::floor(x); }

inline const std::vector<Point>& lattice_basis(int kind) {
    static const double q = 0.25;
    static const std::vector<Point> bases[] = {
        {{0, 0, 0}},                                                         // sc
        {{0, 0, 0}, {0.5, 0.5, 0.5}},                                        // bcc
        {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}},            // fcc
        {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 1.0 / 6, 0.5}, {0, 2.0 / 3, 0.5}},  // hcp, orthorhombic cell
        {{0, 0, 0},
         {0.5, 0.5, 0},
         {0.5, 0, 0.5},
         {0, 0.5, 0.5},  // diamond
         {q, q, q},
         {3 * q, 3 * q, q},
         {3 * q, q, 3 * q},
         {q, 3 * q, 3 * q}},
    };
    return bases[kind];
}
inline const char* lattice_name(int kind) {
    static const char* const names[] = {"sc", "bcc", "fcc", "hcp", "diamond"};
    return names[kind];
}

// A lattice of about n points in [0, 1)^3 scaled to `scale` around `origin`,
// per axis rounded to a power of two with chance `dyadic`: then it is exact on the grid.
inline Points lattice(Rng& rng, int kind, int n, double aspect[3], bool dyadic) {
    const auto& basis = lattice_basis(kind);
    const double per = std::cbrt(double(n) / basis.size() / (aspect[0] * aspect[1] * aspect[2]));
    int side[3];
    for (int a = 0; a < 3; ++a) {
        const double want = std::max(1.0, per * aspect[a]);
        side[a] = dyadic ? 1 << static_cast<int>(std::lround(std::log2(want))) : static_cast<int>(std::lround(want));
        side[a] = std::max(side[a], 1);
    }
    Points p;
    for (int i = 0; i < side[0]; ++i)
        for (int j = 0; j < side[1]; ++j)
            for (int k = 0; k < side[2]; ++k)
                for (const Point& b : basis)
                    p.push_back({(i + b[0]) / side[0], (j + b[1]) / side[1], (k + b[2]) / side[2]});
    (void)rng;
    return p;
}

// Quasi-periodic: the Kronecker sequence of the plastic number's powers, or
// a 3D cut of an irrational lattice.
inline Points quasi(Rng& rng, int n) {
    const double g = 1.32471795724474602596;
    const double a[3] = {1 / g, 1 / (g * g), 1 / (g * g * g)};
    const double offset = rng.unit();
    Points p;
    for (int i = 0; i < n; ++i)
        p.push_back({frac(offset + a[0] * i), frac(offset + a[1] * i), frac(offset + a[2] * i)});
    return p;
}

// Poisson-disc by dart throwing, as far as the distance allows. Brute force
// over a bucket grid: only in tests.
inline Points poisson_disc(Rng& rng, int n) {
    const double r = 0.7 * std::cbrt(1.0 / n);
    const int side = std::max(1, static_cast<int>(1 / r));
    std::vector<std::vector<int>> bucket(side * side * side);
    Points p;
    auto at = [&](int i, int j, int k) -> std::vector<int>& {
        return bucket[(((i + side) % side) * side + (j + side) % side) * side + (k + side) % side];
    };
    for (int tries = 30 * n; tries > 0 && static_cast<int>(p.size()) < n; --tries) {
        const Point c{rng.unit(), rng.unit(), rng.unit()};
        const int b[3] = {static_cast<int>(c[0] * side), static_cast<int>(c[1] * side), static_cast<int>(c[2] * side)};
        bool free = true;
        for (int i = -1; i <= 1 && free; ++i)
            for (int j = -1; j <= 1 && free; ++j)
                for (int k = -1; k <= 1 && free; ++k)
                    for (const int q : at(b[0] + i, b[1] + j, b[2] + k)) {
                        double d2 = 0;
                        for (int a = 0; a < 3; ++a) {
                            const double d = c[a] - p[q][a] - std::nearbyint(c[a] - p[q][a]);
                            d2 += d * d;
                        }
                        free = free && d2 >= r * r;
                    }
        if (!free) continue;
        at(b[0], b[1], b[2]).push_back(static_cast<int>(p.size()));
        p.push_back(c);
    }
    return p;
}

// Power-law clustering, a Soneira-Peebles tree: eta children at lambda times
// the scale at every level, xi(r) a power law as in cosmology.
inline Points power_law(Rng& rng, int n) {
    const int eta = rng.range(2, 6);
    const double lambda = rng.between(1.4, 4.0);
    const int levels = std::max(2, static_cast<int>(std::ceil(std::log(double(n)) / std::log(double(eta)))));
    Points now{{rng.unit(), rng.unit(), rng.unit()}};
    double scale = rng.between(0.1, 0.4);
    for (int l = 0; l < levels && static_cast<int>(now.size()) < n; ++l) {
        Points next;
        for (const Point& c : now)
            for (int e = 0; e < eta; ++e) {
                const Point d = rng.direction();
                const double r = scale * std::cbrt(rng.unit());
                next.push_back({frac(c[0] + r * d[0]), frac(c[1] + r * d[1]), frac(c[2] + r * d[2])});
            }
        now.swap(next);
        scale /= lambda;
    }
    if (static_cast<int>(now.size()) > n) {
        rng.shuffle(now);
        now.resize(n);
    }
    return now;
}

// Vertices of a regular polyhedron or one of its near relatives, on the unit sphere.
inline Points polyhedron(int kind) {
    const double g = (1 + std::sqrt(5.0)) / 2;
    Points p;
    auto signs = [&](double a, double b, double c) {
        for (const int u : {-1, 1})
            for (const int v : {-1, 1})
                for (const int w : {-1, 1}) p.push_back({u * a, v * b, w * c});
    };
    switch (kind) {
        case 0:
            p = {{1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}};
            break;  // tetrahedron
        case 1:
            signs(1, 1, 1);
            break;  // cube
        case 2:
            p = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
            break;  // octahedron
        case 3:     // icosahedron
            for (const int u : {-1, 1})
                for (const int v : {-1, 1}) {
                    p.push_back({0, double(u), v * g});
                    p.push_back({double(u), v * g, 0});
                    p.push_back({u * g, 0, double(v)});
                }
            break;
        case 4:  // dodecahedron
            signs(1, 1, 1);
            for (const int u : {-1, 1})
                for (const int v : {-1, 1}) {
                    p.push_back({0, u / g, v * g});
                    p.push_back({u / g, v * g, 0});
                    p.push_back({u * g, 0, v / g});
                }
            break;
        default:  // cuboctahedron
            for (const int u : {-1, 1})
                for (const int v : {-1, 1}) {
                    p.push_back({double(u), double(v), 0});
                    p.push_back({double(u), 0, double(v)});
                    p.push_back({0, double(u), double(v)});
                }
    }
    for (Point& q : p) {
        const double r = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
        for (double& c : q) c /= r;
    }
    return p;
}

// Integer points of a sphere x^2 + y^2 + z^2 = r2, nearest r2 upward with at
// least `want` of them, cut down at random to `want`.
inline std::vector<std::array<long long, 3>> sphere_points(Rng& rng, int want) {
    std::vector<std::array<long long, 3>> best;
    long long r2 = std::max<long long>(3, static_cast<long long>(want) * want / 8 + rng.below(50));
    for (int tries = 0; tries < 200 && static_cast<int>(best.size()) < want;
         ++tries, r2 += 1 + r2 / 20 + rng.below(5)) {
        // 7 mod 8 and multiples of 4 have few or no points.
        if ((r2 & 3) == 0 || r2 % 8 == 7) continue;
        std::vector<std::array<long long, 3>> here;
        const long long r = static_cast<long long>(std::sqrt(double(r2)));
        for (long long x = -r; x <= r; ++x)
            for (long long y = -r; y <= r; ++y) {
                const long long left = r2 - x * x - y * y;
                if (left < 0) continue;
                const long long z = std::llround(std::sqrt(double(left)));
                if (z * z != left) continue;
                here.push_back({x, y, z});
                if (z != 0) here.push_back({x, y, -z});
            }
        if (here.size() > best.size()) best.swap(here);
    }
    rng.shuffle(best);
    if (static_cast<int>(best.size()) > want) best.resize(want);
    return best;
}

// What one component of a point set adds, in fractions of the box; the
// integer components are added later on the grid (see `grid_parts`).
struct PointSet {
    Points x;           // in box fractions
    std::string label;  // short description
};

// A sub-box of relative size `size` at c, for scale-free clumps.
inline Points cloud(Rng& rng, const char* kind, int n, double size) {
    Points p;
    const Point c{rng.unit(), rng.unit(), rng.unit()};
    const std::string k = kind;
    const Point along = rng.direction();
    Point u = rng.direction();
    const double dot = u[0] * along[0] + u[1] * along[1] + u[2] * along[2];
    for (int a = 0; a < 3; ++a) u[a] -= dot * along[a];
    const double norm = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    for (double& v : u) v /= norm > 0 ? norm : 1;
    for (int i = 0; i < n; ++i) {
        Point q{};
        if (k == "gauss") {
            q = {size * rng.normal(), size * rng.normal(), size * rng.normal()};
        } else if (k == "cube") {
            q = {size * (rng.unit() - 0.5), size * (rng.unit() - 0.5), size * (rng.unit() - 0.5)};
        } else if (k == "shell") {
            const Point d = rng.direction();
            q = {size * d[0], size * d[1], size * d[2]};
        } else if (k == "fibonacci") {
            const double z = 1 - 2 * (i + 0.5) / n, phi = i * M_PI * (3 - std::sqrt(5.0)), r = std::sqrt(1 - z * z);
            q = {size * r * std::cos(phi), size * r * std::sin(phi), size * z};
        } else if (k == "filament") {
            const double t = rng.between(-0.5, 0.5);
            const double w = 1e-3 * size;
            q = {t * along[0] + w * rng.normal(), t * along[1] + w * rng.normal(), t * along[2] + w * rng.normal()};
        } else if (k == "sheet") {
            const double s = rng.between(-0.5, 0.5), t = rng.between(-0.5, 0.5);
            const double w = 1e-3 * size;
            q = {s * along[0] + t * u[0] + w * rng.normal(), s * along[1] + t * u[1] + w * rng.normal(),
                 s * along[2] + t * u[2] + w * rng.normal()};
        } else if (k == "spiral") {
            const double t = 8 * M_PI * i / n, r = size * std::exp(-t / 8);
            q = {r * std::cos(t), r * std::sin(t), size * 0.1 * t / (8 * M_PI)};
        } else if (k == "cusp") {
            const Point d = rng.direction();
            const double r = size * std::exp(-rng.unit() * 16);
            q = {r * d[0], r * d[1], r * d[2]};
        }
        p.push_back({frac(c[0] + q[0]), frac(c[1] + q[1]), frac(c[2] + q[2])});
    }
    return p;
}

// Every coincident particle moved to a free neighbouring grid position: a
// refusal is what a duplicate gets, and that is one case in many, not most.
inline void separate(Points& x, const Box& box, Rng& rng) {
    std::vector<std::pair<std::array<int, 3>, int>> order(x.size());
    auto at = [&](const Point& q) {
        return std::array<int, 3>{exact_predicates::snap(q[0], box, 0), exact_predicates::snap(q[1], box, 1),
                                  exact_predicates::snap(q[2], box, 2)};
    };
    for (std::size_t i = 0; i < x.size(); ++i) order[i] = {at(x[i]), static_cast<int>(i)};
    std::sort(order.begin(), order.end());
    std::vector<std::array<int, 3>> taken;
    taken.reserve(x.size());
    for (const auto& o : order) taken.push_back(o.first);
    bool any = false;
    for (std::size_t i = 1; i < order.size(); ++i) {
        if (order[i].first != order[i - 1].first) continue;
        any = true;
        std::array<int, 3> k = order[i].first;
        do {
            const int a = rng.below(3);
            const long long moved = k[a] + rng.range(-8, 8);
            k[a] = static_cast<int>(((moved % box.period[a]) + box.period[a]) % box.period[a]);
        } while (std::binary_search(taken.begin(), taken.end(), k));
        for (int a = 0; a < 3; ++a) x[order[i].second][a] = static_cast<double>(k[a]) * box.step;
    }
    (void)any;
}

// The fewest particles that leave the cells of an even set inside the reach
// limit of a box (shell_voronoi::level_reach): spacing below a fifth of the shortest side.
inline int fewest_particles(const Box& box) {
    const double s = box.shortest();
    return static_cast<int>(std::min(1e8, 125 * box.volume() / (s * s * s)));
}

// One seed's point set, in the box. Positions off the grid unless the case
// says so; the exact parts are built in whole grid steps.
inline Points make_points(Rng& rng, const Box& box, const Profile& profile, std::string& label) {
    const int low = std::max(profile.fewest, fewest_particles(box));
    int n = rng.below(profile.tiny) == 0 ? rng.range(2, std::max(3, profile.fewest))
                                         : static_cast<int>(rng.log_between(low, std::max(low, profile.most)));
    n = std::min(n, 8 * profile.most);
    double aspect[3];
    const double longest = std::max({box.side[0], box.side[1], box.side[2]});
    for (int a = 0; a < 3; ++a) aspect[a] = box.side[a] / longest;
    const double spacing = std::cbrt(aspect[0] * aspect[1] * aspect[2] / n);

    Points p;  // fractions of the box
    // A background that keeps cells small, nearly always.
    int kind = rng.below(10);
    if (kind == 9) {
        label += "nobg";
    } else if (kind < 4) {
        const int lat = rng.below(5);
        p = lattice(rng, lat, n, aspect, rng.chance(0.5));
        label += lattice_name(lat);
    } else if (kind < 7) {
        p.resize(n);
        for (Point& q : p) q = {rng.unit(), rng.unit(), rng.unit()};
        label += "uniform";
    } else if (kind < 8) {
        p = poisson_disc(rng, n);
        label += "poisson";
    } else {
        p = quasi(rng, n);
        label += "quasi";
    }
    // Jitter from one grid step to one spacing, or none.
    if (rng.chance(0.6) && !p.empty()) {
        const double amount =
            rng.chance(0.2) ? rng.range(1, 4) * box.step / longest : rng.log_between(1e-9, 1.0) * spacing;
        for (Point& q : p)
            for (int a = 0; a < 3; ++a) q[a] = frac(q[a] + rng.between(-amount, amount) / aspect[a]);
        label += "+jit";
    }

    // Further parts, in fractions of the box.
    static const char* const clouds[] = {"gauss", "cube", "shell", "fibonacci", "filament", "sheet", "spiral", "cusp"};
    const int parts = rng.below(4);
    for (int part = 0; part < parts; ++part) {
        const int k = std::max(4, static_cast<int>(n * rng.log_between(0.002, 0.5)));
        switch (rng.below(8)) {
            case 0:
            case 1:
            case 2: {
                const int c = rng.below(8);
                const double size = rng.log_between(1e-7, 0.3);
                const Points more = cloud(rng, clouds[c], k, size);
                p.insert(p.end(), more.begin(), more.end());
                label += std::string("+") + clouds[c];
                break;
            }
            case 3: {  // a density step: a half box or smaller cube with more particles
                const double size = rng.chance(0.5) ? 0.5 : rng.log_between(1e-4, 0.5);
                const Point c{rng.unit(), rng.unit(), rng.unit()};
                for (int i = 0; i < k; ++i)
                    p.push_back({frac(c[0] + size * rng.unit()), frac(c[1] + size * rng.unit()),
                                 frac(c[2] + size * rng.unit())});
                label += "+step";
                break;
            }
            case 4: {  // on faces, edges and corners of the box
                for (int i = 0; i < k; ++i) {
                    Point q{rng.unit(), rng.unit(), rng.unit()};
                    const int fixed = 1 + rng.below(3);
                    for (int a = 0; a < fixed; ++a) {
                        const int axis = rng.below(3);
                        const double near = rng.chance(0.5) ? 0 : rng.log_between(1e-10, 1e-3);
                        q[axis] = rng.chance(0.5) ? near : 1 - near;
                    }
                    p.push_back(q);
                }
                label += "+faces";
                break;
            }
            case 5: {  // a void, emptied
                const Point c{rng.unit(), rng.unit(), rng.unit()};
                const double r = rng.chance(0.15) ? rng.between(0.1, 0.45) : rng.between(0.02, 0.12);
                Points kept;
                for (const Point& q : p) {
                    double d2 = 0;
                    for (int a = 0; a < 3; ++a) {
                        const double d = (q[a] - c[a]) - std::nearbyint(q[a] - c[a]);
                        d2 += d * aspect[a] * d * aspect[a];
                    }
                    if (d2 >= r * r) kept.push_back(q);
                }
                p.swap(kept);
                label += "+void";
                break;
            }
            case 6: {
                const Points more = power_law(rng, k);
                p.insert(p.end(), more.begin(), more.end());
                label += "+powerlaw";
                break;
            }
            default: {  // polyhedra
                const Point c{rng.unit(), rng.unit(), rng.unit()};
                const double r = rng.log_between(1e-7, 0.2);
                const Points v = polyhedron(rng.below(6));
                for (const Point& d : v)
                    p.push_back({frac(c[0] + r * d[0]), frac(c[1] + r * d[1]), frac(c[2] + r * d[2])});
                if (rng.chance(0.5)) p.push_back(c);
                label += "+poly";
                break;
            }
        }
    }

    // To positions in the box, some of them on the grid.
    Points x(p.size());
    const bool on_grid = rng.chance(0.4);
    for (std::size_t i = 0; i < p.size(); ++i)
        for (int a = 0; a < 3; ++a) {
            const double v = p[i][a] * box.side[a];
            x[i][a] = on_grid ? static_cast<double>(exact_predicates::snap(v, box, a)) * box.step
                              : std::min(v, std::nextafter(box.side[a], 0.0));
        }

    // Parts in whole grid steps: cospherical and coplanar exactly, near-duplicates, duplicates.
    auto steps_of = [&](const Point& q) {
        std::array<long long, 3> k;
        for (int a = 0; a < 3; ++a) k[a] = exact_predicates::snap(q[a], box, a);
        return k;
    };
    auto add_steps = [&](long long i, long long j, long long k) {
        Point q{static_cast<double>(((i % box.period[0]) + box.period[0]) % box.period[0]) * box.step,
                static_cast<double>(((j % box.period[1]) + box.period[1]) % box.period[1]) * box.step,
                static_cast<double>(((k % box.period[2]) + box.period[2]) % box.period[2]) * box.step};
        x.push_back(q);
    };
    const int exact_parts = rng.below(3);
    for (int part = 0; part < exact_parts; ++part) {
        const Point c{rng.unit() * box.side[0], rng.unit() * box.side[1], rng.unit() * box.side[2]};
        const auto o = steps_of(c);
        switch (rng.below(5)) {
            case 0: {  // exact cospherical, 5 to 2000 points
                const int want = rng.chance(0.95) ? rng.range(5, 200) : rng.range(200, 2000);
                const auto s = sphere_points(rng, want);
                long long r = 1;
                for (const auto& q : s) r = std::max({r, std::llabs(q[0]), std::llabs(q[1]), std::llabs(q[2])});
                const double limit =
                    0.45 * static_cast<double>(std::min({box.period[0], box.period[1], box.period[2]})) / r;
                const long long scale = std::max(1LL, static_cast<long long>(rng.log_between(1, std::max(1.5, limit))));
                for (const auto& q : s) add_steps(o[0] + scale * q[0], o[1] + scale * q[1], o[2] + scale * q[2]);
                label += "+cosphere";
                if (rng.chance(0.3)) add_steps(o[0], o[1], o[2]);
                break;
            }
            case 1: {  // exact coplanar, 5 to 2000 points in a plane through o
                const int want = rng.chance(0.8) ? rng.range(5, 200) : rng.range(200, 2000);
                long long u[3] = {rng.range(-3, 3), rng.range(-3, 3), rng.range(-3, 3)},
                          v[3] = {rng.range(-3, 3), rng.range(-3, 3), rng.range(-3, 3)};
                if (u[0] == 0 && u[1] == 0 && u[2] == 0) u[0] = 1;
                if (v[1] * u[2] == v[2] * u[1] && v[2] * u[0] == v[0] * u[2] && v[0] * u[1] == v[1] * u[0]) {
                    v[(u[0] != 0) ? 1 : 0] += 1;
                }
                const long long stride = static_cast<long long>(rng.log_between(1, 1e6));
                const int side = std::max(2, static_cast<int>(std::sqrt(double(want))));
                for (int i = 0; i < want; ++i) {
                    const long long s = stride * (rng.range(0, side)), t = stride * (rng.range(0, side));
                    add_steps(o[0] + s * u[0] + t * v[0], o[1] + s * u[1] + t * v[1], o[2] + s * u[2] + t * v[2]);
                }
                label += "+coplanar";
                break;
            }
            case 2: {  // near-duplicates, 1 to 3 grid steps apart
                const int pairs = 1 + static_cast<int>(rng.log_between(1, 1 + x.size() / 4.0));
                const int count = static_cast<int>(x.size());
                for (int i = 0; i < pairs && count > 0; ++i) {
                    const auto q = steps_of(x[rng.below(count)]);
                    long long d[3] = {rng.range(-3, 3), rng.range(-3, 3), rng.range(-3, 3)};
                    if (d[0] == 0 && d[1] == 0 && d[2] == 0) d[rng.below(3)] = 1;
                    add_steps(q[0] + d[0], q[1] + d[1], q[2] + d[2]);
                }
                label += "+near";
                break;
            }
            case 3: {  // exact duplicates: a refusal expected
                if (!rng.chance(0.25) || x.empty()) break;
                const int pairs = rng.range(1, 4);
                const int count = static_cast<int>(x.size());
                for (int i = 0; i < pairs; ++i) x.push_back(x[rng.below(count)]);
                label += "+dup";
                break;
            }
            default: {  // points on one line
                const int want = rng.range(5, 200);
                long long u[3] = {rng.range(-2, 2), rng.range(-2, 2), rng.range(-2, 2)};
                if (u[0] == 0 && u[1] == 0 && u[2] == 0) u[2] = 1;
                const long long stride = static_cast<long long>(rng.log_between(1, 1e6));
                for (int i = 0; i < want; ++i)
                    add_steps(o[0] + i * stride * u[0], o[1] + i * stride * u[1], o[2] + i * stride * u[2]);
                label += "+line";
                break;
            }
        }
    }
    if (label.find("+dup") == std::string::npos) separate(x, box, rng);
    if (rng.chance(0.5)) rng.shuffle(x);
    return x;
}

// --------------------------------------------------------------------- boxes

inline std::array<double, 3> make_box(Rng& rng, std::string& label) {
    const double hcp[3] = {1, std::sqrt(3.0), 2 * std::sqrt(2.0 / 3)};
    switch (rng.below(16)) {
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
            label += "cube ";
            return {1, 1, 1};
        case 6: {
            const double s = std::ldexp(1.0, rng.range(-30, 30));
            label += "pow2 ";
            return {s, s, s};
        }
        case 7: {
            const double s = rng.between(1, 10) * std::pow(10.0, rng.range(-12, 12));  // a long mantissa
            label += "long ";
            return {s, s, s};
        }
        case 8: {
            const double s = rng.between(1, 10) * std::pow(10.0, rng.range(-140, 140));
            label += "extreme ";
            return {s, s, s};
        }
        case 9: {
            const double t = std::ldexp(1.0, -rng.range(1, 6));
            label += "slab ";
            return {1, 1, t};
        }
        case 10: {
            label += "rod ";
            const double t = std::ldexp(1.0, rng.range(1, 6));
            return {t, 1, 1};
        }
        case 11:
            label += "123 ";
            return {1, 2, 3};
        case 12:
            label += "hcp ";
            return {hcp[0], hcp[1], hcp[2]};
        case 13: {
            label += "ratio ";
            return {rng.between(0.2, 1), rng.between(0.2, 1), rng.between(0.2, 1)};
        }
        case 14: {
            const double s = rng.between(1, 10) * std::pow(10.0, rng.range(-12, 12));
            label += "ratiolong ";
            return {s * rng.between(0.25, 1), s * rng.between(0.25, 1), s};
        }
        default: {
            label += "thin ";
            return {1, rng.log_between(1e-3, 1), rng.log_between(1e-3, 1)};
        }
    }
}

// -------------------------------------------------------------------- motion

// Velocity at x of a field, in units of the speed, periodic in the box.
struct Field {
    int kind = 0;
    double k[8][3] = {}, phase[8] = {}, a[8][3] = {};
    Point centre{};
    Point velocity(const Point& q) const {  // q in fractions of the box
        constexpr double tau = 2 * M_PI;
        Point v{};
        const double x = tau * q[0], y = tau * q[1];
        switch (kind) {
            case 0:
                v = {std::sin(y), 0, 0};
                break;  // shear
            case 1:
                v = {-std::sin(y) * std::cos(x), std::sin(x) * std::cos(y), 0};
                break;  // vortex
            case 2:
                v = {std::sin(x), -std::sin(y), 0};
                break;  // strain
            case 3:     // turbulence
                for (int m = 0; m < 8; ++m) {
                    const double s = std::sin(tau * (k[m][0] * q[0] + k[m][1] * q[1] + k[m][2] * q[2]) + phase[m]);
                    for (int c = 0; c < 3; ++c) v[c] += a[m][c] * s;
                }
                break;
            case 4:  // collapse onto the centre, 5: expansion from it, periodic in each axis
            case 5: {
                const double sign = kind == 4 ? -1 : 1;
                for (int c = 0; c < 3; ++c)
                    v[c] = sign * std::sin(tau * (q[c] - centre[c]) / 2) * (q[c] < centre[c] ? -1 : 1);
                break;
            }
            default:  // collapse onto a plane
                v = {-std::sin(x - tau * centre[0]), 0, 0};
                break;
        }
        return v;
    }
};

inline Field make_field(Rng& rng) {
    Field f;
    f.kind = rng.below(7);
    f.centre = {rng.unit(), rng.unit(), rng.unit()};
    for (int m = 0; m < 8; ++m) {
        for (int c = 0; c < 3; ++c) f.k[m][c] = rng.range(-3, 3);
        f.phase[m] = rng.between(0, 6.28);
        const Point d = rng.direction();
        for (int c = 0; c < 3; ++c) f.a[m][c] = d[c] / std::sqrt(8.0);
    }
    return f;
}

inline void wrap_into(Point& q, const Box& box) {
    for (int a = 0; a < 3; ++a) {
        q[a] -= box.side[a] * std::floor(q[a] / box.side[a]);
        if (q[a] >= box.side[a]) q[a] = 0;
    }
}

// The next positions, from the current and the one before. `spacing`: mean,
// in length.
inline Points next_positions(Rng& rng, const Box& box, const Points& now, const Points& before, double spacing,
                             Field& field, bool on_grid, std::string& label) {
    Points x = now;
    const int n = static_cast<int>(x.size());
    if (n == 0) return x;
    // A step from one grid step to 3 spacings, sometimes of order the box.
    const double scale =
        rng.chance(0.07) ? rng.between(0.2, 1.1) * box.shortest() : rng.log_between(box.step, 3 * spacing);
    auto subset = [&](double fraction) {
        std::vector<int> s(n);
        std::iota(s.begin(), s.end(), 0);
        rng.shuffle(s);
        s.resize(std::max(1, static_cast<int>(fraction * n)));
        return s;
    };
    const int kind = rng.below(16);
    label += std::to_string(kind) + ",";
    switch (kind) {
        case 0: {  // noise on every particle
            for (Point& q : x)
                for (int a = 0; a < 3; ++a) q[a] += rng.between(-scale, scale);
            break;
        }
        case 1: {  // noise on a subset
            for (const int i : subset(rng.log_between(1.0 / n, 1)))
                for (int a = 0; a < 3; ++a) x[i][a] += rng.between(-scale, scale);
            break;
        }
        case 2:
        case 3:
        case 4:
        case 5: {  // a smooth field
            if (rng.chance(0.3)) field = make_field(rng);
            for (Point& q : x) {
                const Point v = field.velocity({q[0] / box.side[0], q[1] / box.side[1], q[2] / box.side[2]});
                for (int a = 0; a < 3; ++a) q[a] += scale * v[a];
            }
            break;
        }
        case 6: {  // kicks on a few
            for (const int i : subset(rng.log_between(1.0 / n, 0.1))) {
                const Point d = rng.direction();
                const double r = rng.log_between(spacing, box.shortest());
                for (int a = 0; a < 3; ++a) x[i][a] += r * d[a];
            }
            break;
        }
        case 7: {  // a bulk drift, wrapping through the box
            const Point d = rng.direction();
            for (Point& q : x)
                for (int a = 0; a < 3; ++a) q[a] += scale * d[a];
            break;
        }
        case 8:
            break;  // stopped
        case 9: {   // reversal
            if (before.size() == x.size()) x = before;
            break;
        }
        case 10: {  // reflection through now: the same step back past the start
            if (before.size() == x.size())
                for (int i = 0; i < n; ++i)
                    for (int a = 0; a < 3; ++a) x[i][a] += now[i][a] - before[i][a];
            break;
        }
        case 11: {  // swapping pairs
            auto s = subset(rng.log_between(2.0 / n, 1));
            for (std::size_t i = 0; i + 1 < s.size(); i += 2) std::swap(x[s[i]], x[s[i + 1]]);
            break;
        }
        case 12: {  // permuting ids
            rng.shuffle(x);
            break;
        }
        case 13: {  // one particle from the other in whole grid steps: a near pair appears
            for (const int i : subset(rng.log_between(1.0 / n, 0.1))) {
                const Point o = now[rng.below(n)];
                for (int a = 0; a < 3; ++a) x[i][a] = o[a] + rng.range(-3, 3) * box.step;
            }
            break;
        }
        case 14: {  // all back to an exact lattice
            double aspect[3];
            const double longest = std::max({box.side[0], box.side[1], box.side[2]});
            for (int a = 0; a < 3; ++a) aspect[a] = box.side[a] / longest;
            Points l = lattice(rng, rng.below(5), n, aspect, rng.chance(0.5));
            if (static_cast<int>(l.size()) >= n) {
                rng.shuffle(l);
                for (int i = 0; i < n; ++i)
                    for (int a = 0; a < 3; ++a) x[i][a] = l[i][a] * box.side[a];
            } else {
                for (int i = 0; i < static_cast<int>(l.size()); ++i)
                    for (int a = 0; a < 3; ++a) x[i][a] = l[i][a] * box.side[a];
            }
            break;
        }
        default: {  // all somewhere else
            for (Point& q : x)
                for (int a = 0; a < 3; ++a) q[a] = rng.unit() * box.side[a];
            break;
        }
    }
    const bool leave_outside = rng.chance(0.03);
    for (Point& q : x) {
        if (!leave_outside) wrap_into(q, box);
        if (on_grid)
            for (int a = 0; a < 3; ++a) q[a] = static_cast<double>(exact_predicates::snap(q[a], box, a)) * box.step;
    }
    if (!rng.chance(0.05)) separate(x, box, rng);  // a pair landing on one position, one step in 20
    return x;
}

// ------------------------------------------------------------------- scripts

inline Script make_script(std::uint64_t seed) {
    Rng rng(seed);
    const Profile profile = profile_of(seed);
    const bool api = profile.steps > 10;
    Script s;
    s.sides = make_box(rng, s.label);
    const Box box(s.sides[0], s.sides[1], s.sides[2]);
    s.sides = {box.side[0], box.side[1], box.side[2]};
    if (rng.chance(0.25)) s.variant = rng.range(1, variants - 1);
    Points x = make_points(rng, box, profile, s.label);
    if (x.empty()) x.push_back({0, 0, 0});
    const int n = static_cast<int>(x.size());
    const double spacing = std::cbrt(box.volume() / n);
    Field field = make_field(rng);
    const bool on_grid = rng.chance(0.3);
    auto add = [&](int kind, int arg = 0) {
        Op o;
        o.kind = kind;
        o.arg = arg;
        s.ops.push_back(std::move(o));
    };
    auto add_x = [&](int kind, const Points& p) {
        Op o;
        o.kind = kind;
        o.x = p;
        s.ops.push_back(std::move(o));
    };
    const double extras = api ? 0.5 : 0.15;
    if (rng.chance(0.2)) add(Op::fallback, n <= 400 && rng.chance(0.3) ? 1 : 2 * rng.below(2));
    add_x(rng.chance(0.2) ? Op::rebuild : Op::update, x);
    Points before = x, now = x;
    s.label += " steps:";
    const int steps = rng.range(1, profile.steps);
    for (int t = 0; t < steps; ++t) {
        if (rng.chance(0.5)) add(Op::cells);
        if (rng.chance(extras)) add(Op::scribble, rng.chance(0.5) ? rng.range(1, 4096) : rng.range(4096, 1 << 22));
        if (rng.chance(extras)) add(Op::fallback, n <= 400 && rng.chance(0.3) ? 1 : 2 * rng.below(2));
        if (rng.chance(extras / 3)) add(Op::rounds, rng.chance(0.5) ? rng.range(1, 4) : 1000);
        if (rng.chance(extras / 2)) add(Op::timed);
        if (rng.chance(extras / 2)) {
            // clip at the current positions, or at other positions.
            Op o;
            o.kind = Op::clip;
            o.x = now;
            if (rng.chance(0.3)) o.x = next_positions(rng, box, now, before, spacing, field, on_grid, s.label);
            const int count = rng.chance(0.05)  ? 0
                              : rng.chance(0.2) ? static_cast<int>(now.size())
                                                : rng.range(1, std::min<int>(32, now.size()));
            o.list.resize(count);
            std::iota(o.list.begin(), o.list.end(), 0);
            if (count > 0 && count < static_cast<int>(now.size())) {
                std::vector<int> all(now.size());
                std::iota(all.begin(), all.end(), 0);
                rng.shuffle(all);
                all.resize(count);
                o.list = all;
            }
            s.ops.push_back(std::move(o));
        }
        // Refused input, then the positions as they were.
        if (rng.chance(extras / 4)) {
            Points bad = now;
            const int i = rng.below(static_cast<int>(bad.size()));
            switch (rng.below(4)) {
                case 0:
                    bad[i][rng.below(3)] = std::numeric_limits<double>::quiet_NaN();
                    break;
                case 1:
                    bad[i][rng.below(3)] = std::numeric_limits<double>::infinity();
                    break;
                case 2:
                    bad[i][rng.below(3)] = 1e300;
                    break;
                default:
                    bad[i] = bad[rng.below(static_cast<int>(bad.size()))];
                    break;
            }
            add_x(Op::update, bad);
            add_x(Op::update, now);
        }
        // Another n.
        if (rng.chance(extras / 4)) {
            Points other = now;
            if (rng.chance(0.5) && other.size() > 4)
                other.resize(other.size() - 1 - rng.below(static_cast<int>(other.size()) / 4));
            else
                for (int i = rng.range(1, 16); i > 0; --i)
                    other.push_back({rng.unit() * box.side[0], rng.unit() * box.side[1], rng.unit() * box.side[2]});
            add_x(Op::update, other);
            now = other;
            before = other;
            continue;
        }
        Points next = next_positions(rng, box, now, before, spacing, field, on_grid, s.label);
        before = now;
        now = next;
        add_x(rng.chance(0.05) ? Op::rebuild : Op::update, now);
    }
    add(Op::cells);
    return s;
}

// ---------------------------------------------------------------- shrinking

// A copy of `s` with the particles marked in `drop` removed from every call.
inline Script without_particles(const Script& s, const std::vector<char>& drop) {
    const int n = static_cast<int>(drop.size());
    std::vector<int> renumber(n, -1);
    int kept = 0;
    for (int i = 0; i < n; ++i)
        if (!drop[i]) renumber[i] = kept++;
    Script out = s;
    for (Op& o : out.ops) {
        Points x;
        for (std::size_t i = 0; i < o.x.size(); ++i)
            if (i >= drop.size() || !drop[i]) x.push_back(o.x[i]);
        o.x.swap(x);
        std::vector<int> list;
        for (const int i : o.list)
            if (i < n && renumber[i] >= 0) list.push_back(renumber[i]);
        o.list.swap(list);
    }
    return out;
}

}  // namespace mesh_fuzz
