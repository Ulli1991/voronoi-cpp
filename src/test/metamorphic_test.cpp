// Metamorphic tests of voronoi::Mesh<D> (D = 3, 2, 1), exact on the 2^30 grid: a shift by whole grid steps, axis swaps and
// reflections, a permutation of the ids and a doubled box must give the transformed mesh. Jittered, clustered and lattice
// inputs, in a cube and in a 1:2:3 box. Exercises update, update_cells; a case with an open issue runs with --known only.

#include "check.hpp"
#include "voronoi_mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using Exec = Kokkos::DefaultExecutionSpace;
using Memory = Exec::memory_space;
namespace ep = exact_predicates;
using i64 = long long;

template <int D>
using Grid = std::vector<std::array<i64, D>>;  // positions in whole grid steps

template <int D, std::size_t... I>
ep::BoxN<D> box_of(const std::array<double, D>& s, std::index_sequence<I...>) {
    return ep::BoxN<D>(s[I]...);
}

// The box of the given sides. Used by: generate, the transforms.
template <int D>
ep::BoxN<D> make_box(const std::array<double, D>& sides) {
    return box_of<D>(sides, std::make_index_sequence<D>{});
}

// About n generators of a kind in the box, on its grid. The lattice has a power-of-two spacing in steps, so it is exactly
// degenerate. Used by: every case.
template <int D>
Grid<D> generate(const std::string& kind, int n, const ep::BoxN<D>& box) {
    std::mt19937_64 rng(20261109);
    std::uniform_real_distribution<double> unit(0.0, 1.0), jitter(-0.4, 0.4);
    const double spacing_steps = std::pow(box.volume() / n, 1.0 / D) / box.step;
    const int shift = static_cast<int>(std::lround(std::log2(spacing_steps)));
    const i64 h = 1LL << shift;
    std::array<i64, D> count;
    for (int a = 0; a < D; ++a) count[a] = box.period[a] / h;
    Grid<D> grid;
    std::array<i64, D> index{};
    if (kind != "clustered") {
        for (;;) {
            std::array<i64, D> k;
            for (int a = 0; a < D; ++a) {
                const double j = kind == "jittered" ? jitter(rng) * h : 0.0;
                k[a] =
                    ep::snap((static_cast<double>(index[a]) * h + 0.5 * static_cast<double>(h) + j) * box.step, box, a);
            }
            grid.push_back(k);
            int a = 0;
            while (a < D && ++index[a] == count[a]) index[a++] = 0;
            if (a == D) break;
        }
        return grid;
    }
    std::normal_distribution<double> blob(0.0, 0.04);
    std::array<std::array<double, D>, 8> centre;
    for (auto& c : centre)
        for (double& x : c) x = unit(rng);
    for (int i = 0; i < n; ++i) {
        std::array<i64, D> k;
        for (int a = 0; a < D; ++a) {
            const double u = i % 10 < 3 ? unit(rng) : centre[i % 8][a] + blob(rng);
            k[a] = ep::snap(u * box.side[a], box, a);
        }
        grid.push_back(k);
    }
    return grid;
}

// A mesh as plain numbers on the host, compared after update_cells(). Faces are keyed by (owner, neighbour).
template <int D>
struct Snapshot {
    struct Face {
        std::array<double, D> area, centre;
    };
    std::vector<double> volume;
    std::vector<std::array<double, D>> centroid, position;
    std::map<std::pair<int, int>, Face> faces;
    ep::BoxN<D> box;
};

template <class V>
auto host(const V& v) {
    auto h = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h, v);
    return h;
}

// Builds the mesh of a grid in a box and reads it back. Used by: every case. Throws what the mesh throws.
template <int D>
Snapshot<D> run(const ep::BoxN<D>& box, const Grid<D>& grid) {
    const int n = static_cast<int>(grid.size());
    Kokkos::View<double* [D], Memory> x("x", n);
    auto hx = Kokkos::create_mirror_view(x);
    Snapshot<D> s;
    s.box = box;
    for (int i = 0; i < n; ++i) {
        std::array<double, D> p;
        for (int a = 0; a < D; ++a) hx(i, a) = p[a] = static_cast<double>(grid[i][a]) * box.step;
        s.position.push_back(p);
    }
    Kokkos::deep_copy(x, hx);
    voronoi::Mesh<D, Exec> mesh(Exec{}, box);
    mesh.update(x);
    mesh.update_cells();
    const auto owner = host(mesh.owner()), neighbour = host(mesh.neighbour());
    const auto area = host(mesh.area()), fc = host(mesh.face_centroid());
    const auto volume = host(mesh.volume());
    const auto centroid = host(mesh.centroid());
    for (int i = 0; i < n; ++i) {
        s.volume.push_back(volume(i));
        std::array<double, D> c;
        for (int a = 0; a < D; ++a) c[a] = centroid(i, a);
        s.centroid.push_back(c);
    }
    for (int f = 0; f < mesh.faces(); ++f) {
        typename Snapshot<D>::Face face;
        for (int a = 0; a < D; ++a) {
            face.area[a] = area(f, a);
            face.centre[a] = fc(f, a);
        }
        s.faces[{owner(f), neighbour(f)}] = face;
    }
    return s;
}

// A transformation of the input: output axis a is source axis perm[a], mirrored if flip[a]; then a shift in steps, the
// ids renamed (new id rename[i] for old i) and the box scaled.
template <int D>
struct Transform {
    std::array<int, D> perm;
    std::array<bool, D> flip{};
    std::array<i64, D> shift{};
    std::vector<int> rename;
    double scale = 1;
    Transform() { std::iota(perm.begin(), perm.end(), 0); }
};

// The grid and box the transformation makes of a grid and box. Used by: every case.
template <int D>
std::pair<ep::BoxN<D>, Grid<D>> apply(const Transform<D>& t, const ep::BoxN<D>& box, const Grid<D>& grid) {
    std::array<double, D> sides;
    for (int a = 0; a < D; ++a) sides[a] = box.side[t.perm[a]] * t.scale;
    const ep::BoxN<D> out = make_box<D>(sides);
    Grid<D> moved(grid.size());
    for (std::size_t i = 0; i < grid.size(); ++i) {
        const int to = t.rename.empty() ? static_cast<int>(i) : t.rename[i];
        for (int a = 0; a < D; ++a) {
            const i64 period = box.period[t.perm[a]], k = grid[i][t.perm[a]];
            moved[to][a] = ((t.flip[a] ? period - k : k) + t.shift[a]) % period;
        }
    }
    return {out, moved};
}

// What the base mesh becomes under the transformation, from the base snapshot and the transformed positions.
template <int D>
Snapshot<D> expected(const Transform<D>& t, const Snapshot<D>& base, const Snapshot<D>& moved_positions) {
    Snapshot<D> e;
    e.box = moved_positions.box;
    e.position = moved_positions.position;
    const std::size_t n = base.volume.size();
    e.volume.resize(n);
    e.centroid.resize(n);
    const double length = t.scale, area_scale = std::pow(t.scale, D - 1), volume_scale = std::pow(t.scale, D);
    const auto vector = [&](const std::array<double, D>& v, double factor) {
        std::array<double, D> w;
        for (int a = 0; a < D; ++a) w[a] = (t.flip[a] ? -1 : 1) * v[t.perm[a]] * factor;
        return w;
    };
    const auto name = [&](int i) { return t.rename.empty() ? i : t.rename[i]; };
    for (std::size_t i = 0; i < n; ++i) {
        e.volume[name(static_cast<int>(i))] = base.volume[i] * volume_scale;
        e.centroid[name(static_cast<int>(i))] = vector(base.centroid[i], length);
    }
    for (const auto& [key, face] : base.faces) {
        int i = name(key.first), j = name(key.second);
        typename Snapshot<D>::Face out{vector(face.area, area_scale), vector(face.centre, length)};
        if (i > j) {  // owner and neighbour swap: the area flips, the centre is relative to the other generator
            std::swap(i, j);
            for (int a = 0; a < D; ++a) {
                const double d = e.position[j][a] - e.position[i][a];  // new neighbour minus new owner
                out.area[a] = -out.area[a];
                out.centre[a] += d - e.box.side[a] * std::nearbyint(d / e.box.side[a]);
            }
        }
        e.faces[{i, j}] = out;
    }
    return e;
}

// Largest differences of a mesh from an expected one, in units of the mean spacing to the power of the quantity's
// dimension (areas D - 1, volumes D, offsets 1); same_faces false if the sets of faces above rounding level differ.
// Faces below `rounding_area` (ties of a degenerate input, rounding-level faces) are not compared: the library may keep
// or drop them, and a lattice permuted in its ids picks other ones.
struct Difference {
    bool same_faces = true;
    double area = 0, volume = 0, centroid = 0, centre = 0;
    bool within(double tolerance) const {
        return same_faces && std::max({area, volume, centroid, centre}) <= tolerance;
    }
};

template <int D>
double norm(const std::array<double, D>& v) {
    double sum = 0;
    for (double x : v) sum += x * x;
    return std::sqrt(sum);
}

template <int D>
Difference compare(const Snapshot<D>& got, const Snapshot<D>& want) {
    Difference d;
    const double spacing = std::pow(want.box.volume() / static_cast<double>(want.volume.size()), 1.0 / D);
    const double area_unit = std::pow(spacing, D - 1), volume_unit = std::pow(spacing, D);
    const double significant = 1e-9 * area_unit;
    const auto large = [&](const Snapshot<D>& s) {
        std::map<std::pair<int, int>, typename Snapshot<D>::Face> faces;
        for (const auto& [key, face] : s.faces)
            if (norm<D>(face.area) > significant) faces[key] = face;
        return faces;
    };
    const auto have = large(got), need = large(want);
    d.same_faces = have.size() == need.size();
    for (const auto& [key, face] : need) {
        const auto found = have.find(key);
        if (found == have.end()) {
            d.same_faces = false;
            continue;
        }
        for (int a = 0; a < D; ++a) {
            d.area = std::max(d.area, std::abs(found->second.area[a] - face.area[a]) / area_unit);
            d.centre = std::max(d.centre, std::abs(found->second.centre[a] - face.centre[a]) / spacing);
        }
    }
    for (std::size_t i = 0; i < want.volume.size(); ++i) {
        d.volume = std::max(d.volume, std::abs(got.volume[i] - want.volume[i]) / volume_unit);
        for (int a = 0; a < D; ++a)
            d.centroid = std::max(d.centroid, std::abs(got.centroid[i][a] - want.centroid[i][a]) / spacing);
    }
    return d;
}

// Open issues by case name prefix (0: none). A known case runs with --known only.
int known_issue(const std::string&) { return 0; }

std::string describe(const Difference& d) {
    return std::string(d.same_faces ? "same faces" : "FACES DIFFER") + ", max area " + sci(d.area) + ", volume " +
           sci(d.volume) + ", centroid " + sci(d.centroid) + ", face centre " + sci(d.centre);
}

// All the transformations of one input, each within `tolerance` of the transformed mesh (the geometry is not
// bit-identical: see the README).
template <int D>
void cases(Check& check, bool known, const std::string& kind, int n, const std::array<double, D>& sides) {
    const ep::BoxN<D> box = make_box<D>(sides);
    const bool cube = box.cubic();
    const double tolerance = 1e-10;
    const std::string label = std::to_string(D) + "d." + kind + (cube ? ".cube." : ".box.");
    const Grid<D> grid = generate<D>(kind, n, box);
    Snapshot<D> base;
    try {
        base = run<D>(box, grid);
    } catch (const std::exception& e) {
        check.expect(false, label + "base mesh: " + e.what());
        return;
    }
    std::mt19937_64 rng(7);
    const auto test = [&](const std::string& name, const Transform<D>& t) {
        const std::string full = label + name;
        if (const int issue = known_issue(full); issue && !known) {
            std::cout << "skip  " << full << " (known, issue #" << issue << ")\n";
            return;
        }
        try {
            const auto [moved_box, moved_grid] = apply<D>(t, box, grid);
            Snapshot<D> positions;
            positions.box = moved_box;
            for (const auto& k : moved_grid) {
                std::array<double, D> p;
                for (int a = 0; a < D; ++a) p[a] = static_cast<double>(k[a]) * moved_box.step;
                positions.position.push_back(p);
            }
            const Difference d = compare<D>(run<D>(moved_box, moved_grid), expected<D>(t, base, positions));
            check.expect(d.within(tolerance), full + ": " + describe(d));
        } catch (const std::exception& e) {
            check.expect(false, full + ": " + e.what());
        }
    };
    Transform<D> shift;
    for (int a = 0; a < D; ++a) shift.shift[a] = static_cast<i64>(rng() % box.period[a]);
    test("shift.random", shift);
    Transform<D> wrap;  // every generator crosses the periodic boundary
    for (int a = 0; a < D; ++a) wrap.shift[a] = box.period[a] / 2 + 1;
    test("shift.half", wrap);
    Transform<D> one;
    for (int a = 0; a < D; ++a) one.shift[a] = 1;
    test("shift.one_step", one);
    for (int a = 0; a + 1 < D; ++a) {
        Transform<D> swap;
        std::swap(swap.perm[a], swap.perm[a + 1]);
        test("swap." + std::to_string(a), swap);
    }
    if (D > 1) {
        Transform<D> rotate;  // 90 degrees about the last axis: (x, y) -> (-y, x)
        std::swap(rotate.perm[0], rotate.perm[1]);
        rotate.flip[0] = true;
        test("rotate90", rotate);
    }
    Transform<D> mirror;
    mirror.flip[0] = true;
    test("mirror", mirror);
    Transform<D> permute;
    permute.rename.resize(grid.size());
    std::iota(permute.rename.begin(), permute.rename.end(), 0);
    std::shuffle(permute.rename.begin(), permute.rename.end(), rng);
    test("permute_ids", permute);
    Transform<D> doubled;
    doubled.scale = 2;
    test("double_box", doubled);
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    bool known = false;
    for (int i = 1; i < argc; ++i) known = known || std::string(argv[i]) == "--known";
    Check check;
    for (const char* kind : {"jittered", "clustered", "lattice"}) {
        cases<3>(check, known, kind, 4096, {1, 1, 1});
        cases<3>(check, known, kind, 16384, {1, 2, 3});
        cases<2>(check, known, kind, 4096, {1, 1});
        cases<2>(check, known, kind, 4096, {1, 3});
        cases<1>(check, known, kind, 256, {1});
    }
    return check.failed() != 0;
}
