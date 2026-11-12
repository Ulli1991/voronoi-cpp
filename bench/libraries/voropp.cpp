// voro++ (Rycroft 2009, Chaos 19, 041111) in the library benchmark (docs/library_benchmark.md). Two builds of this
// file:
//   voropp_bench     voro++ master (github.com/chr1shr/voro): container_periodic (native periodic unit cube), put
//                    every point (id = input index), then c_loop_all_periodic + compute_cell with a
//                    voronoicell_neighbor for every particle, and per cell volume(), face_areas() and neighbors()
//                    copied to our arrays (volumes in input order; face areas and neighbour ids, flat, per cell).
//   voropp_mt_bench  -DVOROPP_DEV: the multithreaded extension of Lu, Lazar and Rycroft (CPC 2023, 108832), on the
//                    dev branch of the same repository: container_3d periodic in x, y and z with nt voro_compute
//                    objects, add_parallel + put_reconcile_overflow for the insertion (put for one thread), then
//                    `omp parallel` + `omp for` over the container iterator with one voronoicell_neighbor_3d per
//                    thread (as in its examples/basic/mthread.cc), face output into per-thread arrays.
// The container uses about 5 particles per block (the voro++ documentation's guidance): nb = round((n/5)^(1/3))
// blocks per side (75 at 128^3, 150 at 256^3), 8 initial slots per block (it grows blocks as needed).
//   voropp_bench <points file> [--threads=1] [--repeats=3] [--volumes=file] [--blocks=nb]

#include "libbench.hpp"

#include <voro++.hh>

#include <cmath>
#include <memory>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// The library's output copied out of each computed cell.
struct Output {
    std::vector<double> volume;    // input order
    std::vector<double> centroid;  // input order, relative to the particle (voronoicell::centroid)
    std::vector<int> faces;        // input order
    // Per thread, flat over its cells in computation order: owner id, then face areas and neighbour ids.
    struct Part {
        std::vector<int> owner, neighbour;
        std::vector<double> area, normal;  // per face: area and unit outward normal (area vector = area x normal)
    };
    std::vector<Part> parts;
    void reset(std::uint64_t n, int threads) {
        volume.assign(n, NAN);
        centroid.assign(3 * n, NAN);
        faces.assign(n, 0);
        parts.assign(threads, Part{});
    }
};

template <class Cell>
inline void take(Cell& c, int id, Output& out, Output::Part& part, std::vector<double>& area, std::vector<int>& nb) {
    thread_local std::vector<double> normal;
    out.volume[id] = c.volume();
    c.centroid(out.centroid[3 * id], out.centroid[3 * id + 1], out.centroid[3 * id + 2]);
    c.face_areas(area);
    c.normals(normal);
    c.neighbors(nb);
    out.faces[id] = int(area.size());
    part.owner.push_back(id);
    part.area.insert(part.area.end(), area.begin(), area.end());
    part.normal.insert(part.normal.end(), normal.begin(), normal.end());
    part.neighbour.insert(part.neighbour.end(), nb.begin(), nb.end());
}

// Each build returns its container, freed after the clock stops.
#ifndef VOROPP_DEV
using Container = voro::container_periodic;
std::unique_ptr<Container> build(const double* xyz, std::uint64_t n, int nb, int, Output& out) {
    auto held = std::make_unique<Container>(1, 0, 1, 0, 0, 1, nb, nb, nb, 8);
    Container& con = *held;
    for (std::uint64_t i = 0; i < n; ++i) con.put(int(i), xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
    voro::c_loop_all_periodic loop(con);
    voro::voronoicell_neighbor c(con);  // tolerance scaled to the container (default ctor: 1000^2 box)
    std::vector<double> area;
    std::vector<int> neighbour;
    if (loop.start()) do
            if (con.compute_cell(c, loop)) take(c, loop.pid(), out, out.parts[0], area, neighbour);
        while (loop.inc());
    return held;
}
#else
using Container = voro::container_3d;
std::unique_ptr<Container> build(const double* xyz, std::uint64_t n, int nb, int threads, Output& out) {
    auto held = std::make_unique<Container>(0, 1, 0, 1, 0, 1, nb, nb, nb, true, true, true, 8, threads);
    Container& con = *held;
    if (threads > 1) {
        con.add_parallel(const_cast<double*>(xyz), int(n), threads);
        con.put_reconcile_overflow();
    } else {
        for (std::uint64_t i = 0; i < n; ++i) con.put(int(i), xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
    }
#pragma omp parallel num_threads(threads)
    {
        voro::voronoicell_neighbor_3d c(con);
        std::vector<double> area;
        std::vector<int> neighbour;
        Output::Part& part = out.parts[omp_get_thread_num()];
        voro::container_3d::iterator cli;
#pragma omp for
        for (cli = con.begin(); cli < con.end(); cli++)
            if (con.compute_cell(c, cli)) take(c, con.id[cli->ijk][cli->q], out, part, area, neighbour);
    }
    return held;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: voropp_bench <points file> [--threads=1] [--repeats=3] [--volumes=file]\n");
        return 1;
    }
    const std::string file = argv[1];
    const int threads = std::stoi(libbench::option(argc, argv, "threads", "1"));
    const int repeats = std::stoi(libbench::option(argc, argv, "repeats", "3"));
    const auto frames = libbench::read_frames(file);
    const std::uint64_t n = frames.n;
    const int nb = std::stoi(libbench::option(argc, argv, "blocks",
                                              std::to_string(int(std::lround(std::cbrt(double(n) / 5))))));
    libbench::Setup s;
    s.library = "voro++";
#ifndef VOROPP_DEV
    if (threads != 1) throw std::runtime_error("voro++ master is serial");
    s.variant = "container_periodic, c_loop_all_periodic + compute_cell(voronoicell_neighbor)";
    s.version = "master b0dac57 (2026-03-04; release v0.4.6 + fixes), github.com/chr1shr/voro";
#else
    s.variant = "multithreaded (Lu, Lazar, Rycroft 2023): container_3d periodic xyz, add_parallel, omp for + "
                "compute_cell(voronoicell_neighbor_3d)";
    s.version = "dev 61099f6 (2026-10-17), github.com/chr1shr/voro";
#endif
    s.device = "2x Xeon Platinum 8360Y";
    s.threads = threads;
    s.outputs = "volumes centroids faces areas (with unit normals) neighbours";
    s.periodic = "native";
    s.precision = "double floating point (no exact predicates; tolerance-based plane cuts)";
    s.timed = "points in host memory to container, every cell computed, volume, face areas and neighbour ids copied "
              "to host arrays";
    char text[128];
    std::snprintf(text, sizeof text, "blocks %d^3 (%.2f particles per block), 8 initial slots per block", nb,
                  double(n) / (double(nb) * nb * nb));
    s.note = text;
    Output out;
    const auto run = [&] {
        out.reset(n, threads);
        auto con = build(frames.frame(0), n, nb, threads, out);
    };
    const auto memory_run = [&] {
        libbench::HostPeak peak;
        peak.start();
        run();
        const double bytes = peak.stop();
        std::uint64_t missing = 0, faces = 0;
        double sum = 0;
        for (std::uint64_t i = 0; i < n; ++i) {
            if (std::isnan(out.volume[i])) ++missing;
            else sum += out.volume[i];
            faces += out.faces[i];
        }
        char text[256];
        std::snprintf(text, sizeof text, "; missing cells %llu, sum of volumes - 1 = %.3g, faces per cell %.3f",
                      (unsigned long long)missing, sum - 1, double(faces) / double(n));
        s.note += text;
        std::fprintf(stderr, "# voro++ %s threads %d: %s\n", file.c_str(), threads, s.note.c_str());
        libbench::write_volumes(libbench::option(argc, argv, "volumes"), out.volume);
        return bytes;
    };
    const auto timed_run = [&] {
        out.reset(n, threads);  // outside the clock: the output arrays exist before
        const double t0 = libbench::now();
        auto con = build(frames.frame(0), n, nb, threads, out);
        const double t1 = libbench::now();
        con.reset();
        return t1 - t0;
    };
    libbench::run_protocol(s, file, n, "build", repeats, memory_run, timed_run);
    return 0;
}
