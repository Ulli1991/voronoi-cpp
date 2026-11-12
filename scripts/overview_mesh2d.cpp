// Runs voronoi::Mesh<2> and its stages on one small point set and writes every array scripts/overview_figure.py draws.
// Usage: overview_mesh2d IN OUT. IN: blocks x0, x1, x2, late of n rows "x y"; OUT: named blocks "name rows cols".
// Kokkos::Serial, one core: the same arrays the GPU path writes, in a form a figure can read.

#include "voronoi_mesh.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace {

using Exec = Kokkos::Serial;
using Memory = Exec::memory_space;
using Mesh = voronoi::Mesh<2, Exec>;
using X = Kokkos::View<double* [2], Kokkos::LayoutRight, Memory>;
namespace ep = exact_predicates;

// Writes named blocks of numbers, one row per line. Used by: main.
struct Out {
    std::FILE* file;
    template <class V>
    void block(const std::string& name, const V& v, std::size_t rows) {
        const std::size_t cols = V::rank() == 1 ? 1 : v.extent(1);
        std::fprintf(file, "%s %zu %zu\n", name.c_str(), rows, cols);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t a = 0; a < cols; ++a) {
                double value;
                if constexpr (V::rank() == 1)
                    value = static_cast<double>(v(i));
                else
                    value = static_cast<double>(v(i, a));
                std::fprintf(file, a + 1 < cols ? "%.17g " : "%.17g\n", value);
            }
        }
    }
    template <class V>
    void block(const std::string& name, const V& v) {
        block(name, v, v.extent(0));
    }
    void scalar(const std::string& name, double value) { std::fprintf(file, "%s 1 1\n%.17g\n", name.c_str(), value); }
    void text(const std::string& name, const std::string& value) {
        std::fprintf(file, "%s text\n%s\n", name.c_str(), value.c_str());
    }
};

// The blocks of the input file. Used by: main.
std::map<std::string, X> read(const char* path) {
    std::ifstream in(path);
    std::map<std::string, X> sets;
    std::string name;
    int n;
    while (in >> name >> n) {
        X x(name, n);
        for (int i = 0; i < n; ++i) in >> x(i, 0) >> x(i, 1);
        sets[name] = x;
    }
    return sets;
}

// A deep copy of the triangles, so free stages can run on them while the mesh keeps its own. Used by: main.
tri_mesh::Triangles<Memory> copy(const tri_mesh::Triangles<Memory>& t) {
    tri_mesh::Triangles<Memory> c;
    c.corners = Kokkos::create_mirror(t.corners);
    c.links = Kokkos::create_mirror(t.links);
    c.centre = Kokkos::create_mirror(t.centre);
    c.owner = Kokkos::create_mirror(t.owner);
    c.neighbour = Kokkos::create_mirror(t.neighbour);
    Kokkos::deep_copy(c.corners, t.corners);
    Kokkos::deep_copy(c.links, t.links);
    Kokkos::deep_copy(c.centre, t.centre);
    Kokkos::deep_copy(c.owner, t.owner);
    Kokkos::deep_copy(c.neighbour, t.neighbour);
    return c;
}

// Report fields under a prefix. Used by: main.
void report(Out& out, const std::string& p, const voronoi::Report& r) {
    out.scalar(p + "_rebuilt", r.rebuilt);
    out.scalar(p + "_flips", static_cast<double>(r.flips));
    out.scalar(p + "_rounds", r.rounds);
    out.scalar(p + "_stuck", r.stuck);
    out.scalar(p + "_host_flips", static_cast<double>(r.host_flips));
    out.scalar(p + "_host_rebuilds", static_cast<double>(r.host_rebuilds));
}

// The mesh's triangles, faces and cells after update_cells(), under a prefix. Used by: main.
void geometry(Out& out, const std::string& p, Mesh& mesh) {
    mesh.update_cells();
    const auto& t = mesh.triangles();
    out.block(p + "_corners", t.corners);
    out.block(p + "_links", t.links);
    out.block(p + "_centre", t.centre);
    out.block(p + "_owner", mesh.owner());
    out.block(p + "_neighbour", mesh.neighbour());
    out.block(p + "_area", mesh.area());
    out.block(p + "_face_centroid", mesh.face_centroid());
    out.block(p + "_volume", mesh.volume());
    out.block(p + "_centroid", mesh.centroid());
    out.block(p + "_cell_face_start", mesh.cell_face_start());
    out.block(p + "_cell_face", mesh.cell_face(), mesh.cell_face_start()(mesh.particles()));
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    if (argc != 3) {
        std::cerr << "usage: overview_mesh2d IN OUT\n";
        return 2;
    }
    auto sets = read(argv[1]);
    Out out{std::fopen(argv[2], "w")};
    const Exec exec;
    const X x0 = sets.at("x0");
    const int n = static_cast<int>(x0.extent(0));

    // 1-5: the build. Snapped points, the cell list, every ring, the triangles of the rings and their checks.
    Mesh mesh(exec, 1.0);
    const ep::BoxN<2> box = mesh.sides();
    report(out, "build", mesh.update(x0));
    out.block("snap", mesh.points().view);
    out.scalar("period", static_cast<double>(box.period[0]));
    neighbor_search::CellList<Exec, 2> list;
    neighbor_search::configure(list, n, box, std::sqrt(box.volume() / n));
    neighbor_search::build(exec, list, mesh.positions());
    out.scalar("levels", list.levels);
    out.block("order", list.order);
    out.block("level", list.level);
    Kokkos::View<int*, Memory> all("all", n);
    std::iota(all.data(), all.data() + n, 0);
    const auto rings = mesh.clip(x0, all);
    out.block("ring_status", rings.status);
    out.block("ring_degree", rings.degree);
    out.block("ring_start", rings.start);
    out.block("ring", rings.ring);
    out.scalar("build_area", tri_repair::total_area(exec, mesh.points(), box, mesh.triangles()));
    geometry(out, "built", mesh);

    // 6-10: a step. The screen's list on a copy, then update() and update_cells() on the mesh itself.
    const X x1 = sets.at("x1");
    ep::Points<Memory, 2> p1("p1", n, box);
    ep::snap(exec, x1, box, p1);
    {
        auto t = copy(mesh.triangles());
        tri_repair::State<Memory> state(t.count(), n / 4 + 1024);
        tri_repair::screen(exec, p1, box, t, state);
        out.block("screen_list", state.list, state.entries);
    }
    out.block("snap1", p1.view);
    report(out, "step", mesh.update(x1));
    geometry(out, "moved", mesh);
    Mesh fresh(exec, 1.0);
    fresh.update(x1);
    fresh.update_cells();
    bool same = true;
    for (int i = 0; i < n; ++i) same = same && fresh.volume()(i) == mesh.volume()(i);
    out.scalar("moved_equals_fresh", same);

    // 11: a jump the flips cannot repair: the stuck triangles, the host finish, the rebuild, a refusal.
    const X x2 = sets.at("x2");
    ep::Points<Memory, 2> p2("p2", n, box);
    ep::snap(exec, x2, box, p2);
    {
        auto t = copy(mesh.triangles());
        tri_repair::State<Memory> state(t.count(), n / 4 + 1024);
        tri_repair::screen(exec, p2, box, t, state);
        const auto stats = tri_repair::repair(exec, p2, t, state);
        out.scalar("jump_stuck", stats.stuck);
        out.block("jump_corners", t.corners);
    }
    out.block("snap2", p2.view);
    for (const auto fallback : {voronoi::Fallback::cheapest, voronoi::Fallback::host}) {
        Mesh m(exec, 1.0);
        m.set_fallback(fallback);
        m.update(x1);
        report(out, fallback == voronoi::Fallback::host ? "host" : "cheapest", m.update(x2));
    }
    X twin("twin", n);
    Kokkos::deep_copy(twin, x0);
    twin(1, 0) = twin(0, 0);
    twin(1, 1) = twin(0, 1);
    try {
        Mesh m(exec, 1.0);
        m.update(twin);
        out.text("refusal", "none");
    } catch (const std::runtime_error& e) {
        out.text("refusal", e.what());
    }

    // 12: the same particles much later, for the re-sort (3D only; the figure orders them along a 2D curve).
    Mesh late(exec, 1.0);
    late.update(sets.at("late"));
    out.block("late_snap", late.points().view);
    out.block("late_owner", late.owner());
    out.block("late_neighbour", late.neighbour());
    std::fclose(out.file);
    return 0;
}
