#pragma once

// A voronoi::Mesh against the independent oracle (oracle.hpp): tetrahedra, faces, volumes, a sample of cells against
// brute-force clipping, and the same tetrahedra as a fresh build.

#include "oracle.hpp"
#include "support.hpp"

#include <map>
#include <sstream>

namespace mesh_check {

using namespace support;

template <class M>
oracle::Grid grid_of(const M& mesh) {
    const auto x = to_host(mesh.positions());
    const auto& box = mesh.sides();
    oracle::Grid g;
    g.step = box.step;
    for (int a = 0; a < 3; ++a) g.period[a] = box.period[a];
    g.p.resize(x.extent(0));
    for (std::size_t i = 0; i < x.extent(0); ++i)
        for (int a = 0; a < 3; ++a) g.p[i][a] = std::llround(x(i, a) / box.step);
    return g;
}

template <class M>
oracle::Tets tets_of(const M& mesh) {
    const tet_flips::Mesh m = host_tetrahedra(mesh);
    oracle::Tets t;
    t.vertex.resize(m.count);
    t.adjacent.resize(m.count);
    t.alive.resize(m.count);
    for (int i = 0; i < m.count; ++i) {
        t.alive[i] = m.alive(i) != 0;
        for (int k = 0; k < 4; ++k) {
            t.vertex[i][k] = m.vertex(i, k);
            t.adjacent[i][k] = m.adjacent(i, k);
        }
    }
    return t;
}

// The snapped positions of a mesh as an ordinary position view.
template <class M>
Positions snapped(const M& mesh) {
    const auto p = mesh.positions();
    Positions x("snapped", p.extent(0));
    Kokkos::parallel_for(
        "mesh_check::snapped", Kokkos::RangePolicy<Exec>(0, p.extent(0)), KOKKOS_LAMBDA(int i) {
            for (int a = 0; a < 3; ++a) x(i, a) = p(i, a);
        });
    return x;
}

struct Result {
    oracle::Verdict tets;
    long long not_finite = 0;     // faces or cells with NaN or infinity
    long long fresh_differs = 0;  // tetrahedra in only one of mesh and fresh build; -1: the fresh build refused
    std::string fresh_error;
    int sampled = 0;              // cells checked against brute clipping
    long long brute_missing = 0;  // faces of real area the mesh lacks
    long long brute_extra = 0;    // mesh faces of real area brute force lacks
    long long own_image = 0;      // sampled cells with a face on their own image
    long long two_images = 0;     // neighbours sharing faces through several images
    long long negative = 0;       // cells of volume <= 0
    long long duplicates = 0;     // particles on the grid position of another: no mesh should come back
    double closure = 0, total = 0, brute_volume = 0, brute_area = 0, fresh_volume = 0;
    // Relative volume error times the cell's size in grid steps: positions are doubles near 1, so geometry rounds at
    // about 4e-8 grid steps whatever the cell (measured, clumps 512 to 2^20 steps wide, contrast.grid_scale).
    double brute_floor = 0;
    bool brute_within_rounding() const { return brute_volume < 1e-9 || brute_floor < 2e-7; }
    bool ok() const {
        return tets.ok() && not_finite == 0 && fresh_differs <= 0 && brute_missing == 0 && brute_extra == 0 &&
               negative == 0 && duplicates == 0 && closure < 1e-11 && std::abs(total - 1) < 1e-11 &&
               brute_within_rounding() && (brute_area < 1e-9 || brute_floor < 2e-7) && fresh_volume < 1e-10;
    }
    std::string text() const {
        std::ostringstream o;
        o << tets.text() << " not_finite=" << not_finite << " fresh_differs=" << fresh_differs << fresh_error
          << " brute(cells=" << sampled << " missing=" << brute_missing << " extra=" << brute_extra
          << " volume=" << brute_volume << " area=" << brute_area << " floor=" << brute_floor << " own_image=" << own_image
          << " two_images=" << two_images << ") negative=" << negative << " duplicates=" << duplicates << " closure=" << closure
          << " total-1=" << total - 1 << " fresh_volume=" << fresh_volume;
        return o.str();
    }
};

// `cells`: how many cells to compare with brute clipping (evenly spread;
// every cell of a small mesh). check_fresh: build the same positions again.
template <class M>
Result check(const M& mesh, int cells = 32, bool check_fresh = true) {
    Result r;
    const int n = mesh.particles();
    const auto g = grid_of(mesh);
    const auto t = tets_of(mesh);
    r.tets = oracle::check(g, t);
    {
        std::vector<oracle::Offset> sorted = g.p;
        std::sort(sorted.begin(), sorted.end());
        for (std::size_t i = 1; i < sorted.size(); ++i) r.duplicates += sorted[i] == sorted[i - 1];
    }
    const auto owner = to_host(mesh.owner());
    const auto neighbour = to_host(mesh.neighbour());
    const auto area = face_areas(mesh);
    const auto volume = to_host(mesh.volume());
    const auto& box = mesh.sides();
    const auto face_centroid = to_host(mesh.face_centroid());

    std::vector<std::array<double, 3>> sum(n, {0, 0, 0});
    std::vector<double> magnitude(n, 0);
    for (int f = 0; f < mesh.faces(); ++f) {
        double m = 0;
        for (int a = 0; a < 3; ++a) {
            r.not_finite += !std::isfinite(area(f, a)) || !std::isfinite(face_centroid(f, a));
            sum[owner(f)][a] += area(f, a);
            sum[neighbour(f)][a] -= area(f, a);
            m += area(f, a) * area(f, a);
        }
        m = std::sqrt(m);
        magnitude[owner(f)] += m;
        magnitude[neighbour(f)] += m;
    }
    for (int i = 0; i < n; ++i) {
        r.not_finite += !std::isfinite(volume(i));
        r.negative += !(volume(i) > 0);
        r.total += volume(i) / box.volume();
        const double closing = std::sqrt(sum[i][0] * sum[i][0] + sum[i][1] * sum[i][1] + sum[i][2] * sum[i][2]);
        if (magnitude[i] > 0) r.closure = std::max(r.closure, closing / magnitude[i]);
    }

    // Brute clipping of a sample: volume and every face of real area, in units of the cell's surface.
    const int stride = std::max(1, n / std::max(cells, 1));
    std::vector<std::vector<std::pair<int, int>>> at(n);  // (other particle, face), sampled cells only
    for (int f = 0; f < mesh.faces(); ++f) {
        if (owner(f) % stride == 0) at[owner(f)].push_back({neighbour(f), f});
        if (neighbour(f) % stride == 0) at[neighbour(f)].push_back({owner(f), f});
    }
    const long double step2 = g.step * g.step, step3 = step2 * g.step;
    for (int i = 0; cells > 0 && i < n; i += stride) {
        const oracle::Cell cell = oracle::voronoi_cell(g, i);
        ++r.sampled;
        const double brute = double(cell.volume * step3);
        const double error = brute > 0 ? std::abs(volume(i) - brute) / brute : 1.0;
        r.brute_volume = std::max(r.brute_volume, error);
        r.brute_floor = std::max(r.brute_floor, error * std::cbrt(double(cell.volume)));
        double scale = 0;
        for (const auto& [j, a] : cell.area) scale += double(std::sqrt(oracle::dot(a, a)) * step2);
        const double real = 1e-9 * scale;
        std::map<int, std::array<double, 3>> mesh_faces;
        for (const auto& [j, f] : at[i]) {
            const double sign = owner(f) == i ? 1 : -1;
            auto& s = mesh_faces[j];
            for (int a = 0; a < 3; ++a) s[a] += sign * area(f, a);
        }
        for (const auto& [j, images] : cell.images) {
            r.own_image += j == i;
            r.two_images += j != i && images > 1;
        }
        for (const auto& [j, a] : cell.area) {
            const double size = double(std::sqrt(oracle::dot(a, a)) * step2);
            if (j == i || size < real) continue;
            const auto it = mesh_faces.find(j);
            if (it == mesh_faces.end()) {
                ++r.brute_missing;
                continue;
            }
            double d = 0;
            for (int k = 0; k < 3; ++k) d = std::max(d, std::abs(it->second[k] - double(a[k] * step2)));
            r.brute_area = std::max(r.brute_area, d / scale);
        }
        for (const auto& [j, s] : mesh_faces) {
            const double size = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
            r.brute_extra += size >= real * 1.001 && cell.area.find(j) == cell.area.end();
        }
    }

    if (check_fresh) {
        try {
            // Inside each grid cell: a position on its edge can snap to the
            // step below when the step is not dyadic (see box.snap_twice).
            Positions p("fresh", n);
            auto h = Kokkos::create_mirror_view(p);
            for (int i = 0; i < n; ++i)
                for (int a = 0; a < 3; ++a) h(i, a) = (double(g.p[i][a]) + 0.5) * box.step;
            Kokkos::deep_copy(p, h);
            voronoi::Mesh<3, Exec> fresh(Exec{}, box);
            fresh.update(p);
            r.fresh_differs = oracle::different_tetrahedra(t, tets_of(fresh));
            const auto v = to_host(fresh.volume());
            for (int i = 0; i < n; ++i) r.fresh_volume = std::max(r.fresh_volume, std::abs(v(i) - volume(i)) / v(i));
        } catch (const std::exception& e) {
            r.fresh_differs = -1;
            r.fresh_error = std::string(" fresh_error=\"") + e.what() + "\"";
        }
    }
    return r;
}

// A mesh and the clipping disagree on faces: who is right? Checks the largest faces the clipping lacks against brute-force
// clipping of their owner cell; true when brute force sides with the mesh on all (the plain clipping is the one wrong).
template <class M>
bool arbitrate(const M& mesh, std::ostream& out, int faces = 8) {
    const Positions x = snapped(mesh);
    const Reference reference = clipped(x, mesh.sides());
    long long bad_status = 0;
    for (const int s : reference.status) bad_status += s != shell_voronoi::ok;
    const auto owner = to_host(mesh.owner());
    const auto neighbour = to_host(mesh.neighbour());
    const auto area = face_areas(mesh);
    const auto volume = to_host(mesh.volume());
    std::vector<std::pair<double, int>> extra;
    for (int f = 0; f < mesh.faces(); ++f) {
        const auto& n = reference.neighbours[owner(f)];
        if (std::binary_search(n.begin(), n.end(), neighbour(f))) continue;
        extra.push_back({std::sqrt(area(f, 0) * area(f, 0) + area(f, 1) * area(f, 1) + area(f, 2) * area(f, 2)), f});
    }
    std::sort(extra.rbegin(), extra.rend());
    out << "arbitrate: extra faces " << extra.size() << ", cells of the clipping not ok " << bad_status << "\n";
    const auto g = grid_of(mesh);
    const long double step2 = g.step * g.step, step3 = step2 * g.step;
    bool mesh_right = true;
    for (int k = 0; k < std::min<int>(faces, extra.size()); ++k) {
        const int f = extra[k].second, i = owner(f), j = neighbour(f);
        const oracle::Cell cell = oracle::voronoi_cell(g, i);
        const auto it = cell.area.find(j);
        const double brute_volume = double(cell.volume * step3);
        mesh_right = mesh_right && it != cell.area.end() && std::abs(volume(i) / brute_volume - 1) < 1e-9;
        out << "  face " << f << " (" << i << "," << j << ") area " << extra[k].first << " status " << reference.status[i] << ","
            << reference.status[j] << " brute "
            << (it == cell.area.end() ? std::string("has no such face")
                                      : "has it, area " + sci(double(std::sqrt(oracle::dot(it->second, it->second)) * step2)))
            << "; volume of i: mesh " << volume(i) << " clipping " << reference.volume[i] << " brute "
            << brute_volume << "\n";
    }
    return mesh_right;
}

}  // namespace mesh_check
