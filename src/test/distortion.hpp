#pragma once

// Prescribed velocity fields that distort the mesh, without any hydro, and a
// driver that runs them through voronoi::Mesh and measures, per step,
// robustness (stuck flips, fallbacks, exact oracle at sampled steps), accuracy
// (volumes, closure, equal to a fresh build), speed (M cells/s of the mesh
// step, flips, rounds) and memory (faces, tetrahedra, B/cell, worst aspect
// ratio and faces per cell). Fields run on the device: 256^3 is fine.
//
// Fields live in the unit box (a cube of side 1), peak speed about 1, and are
// integrated with the midpoint rule. No hydro keeps them regular.

#include "mesh_check.hpp"

#include <cstdint>
#include <fstream>
#include <map>
#include <random>
#include <string>

namespace distortion {

using namespace support;

enum class Kind {
    shear,        // sin(2 pi z) along x: periodic shear
    sawtooth,     // uniform shear +-4 in two layers
    vortex,       // differential rotation around z, winding up
    taylor_green,
    turbulence,   // random-phase solenoidal modes with k^(-5/6) amplitudes, drifting phases
    collapse,     // converging to a point, homologous near it (Sedov reversed), within radius 0.16
    noh,          // converging at constant speed, through the centre (Noh reversed), within 0.16
    expansion,    // the reverse of collapse, from a dense ball
    pancake,      // a layer of half thickness 0.16 collapses onto a plane, shock-like
    counter,      // every other particle streams against the rest: all interpenetrate
    thin,         // compression onto four planes plus shear: long thin cells
    // The same with room for a void as wide as the box: these end in the
    // refusal of a cell that reaches too far.
    collapse_wide,
    pancake_wide,
    streams,      // two halves of the box stream against each other
};

inline Kind kind_of(const std::string& name) {
    const std::map<std::string, Kind> names{{"shear", Kind::shear},       {"sawtooth", Kind::sawtooth},
                                            {"vortex", Kind::vortex},     {"taylor_green", Kind::taylor_green},
                                            {"turbulence", Kind::turbulence}, {"collapse", Kind::collapse},
                                            {"noh", Kind::noh},           {"expansion", Kind::expansion},
                                            {"pancake", Kind::pancake},   {"streams", Kind::streams},
                                            {"thin", Kind::thin},         {"counter", Kind::counter},
                                            {"collapse_wide", Kind::collapse_wide},
                                            {"pancake_wide", Kind::pancake_wide}};
    return names.at(name);
}

inline const std::vector<std::string>& all_flows() {
    static const std::vector<std::string> names{"shear",  "sawtooth",  "vortex",  "taylor_green", "turbulence",
                                                "collapse", "noh",     "expansion", "pancake",    "counter",
                                                "thin",   "collapse_wide", "pancake_wide", "streams"};
    return names;
}

struct Field {
    static constexpr int modes = 32;
    static constexpr double pi = 3.141592653589793, two_pi = 2 * pi;
    Kind kind = Kind::shear;
    double k[modes][3], a[modes][3], phase[modes], omega[modes];

    explicit Field(Kind which, unsigned seed = 7) : kind(which) {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> unit(-1, 1);
        double sum = 0;
        for (int m = 0; m < modes; ++m) {
            double norm2 = 0;
            do {
                for (int d = 0; d < 3; ++d) k[m][d] = std::round(8 * unit(rng));
                norm2 = k[m][0] * k[m][0] + k[m][1] * k[m][1] + k[m][2] * k[m][2];
            } while (norm2 < 1 || norm2 > 64);
            // Solenoidal: the amplitude across the wave vector.
            const double r[3] = {unit(rng), unit(rng), unit(rng)};
            const double c[3] = {k[m][1] * r[2] - k[m][2] * r[1], k[m][2] * r[0] - k[m][0] * r[2], k[m][0] * r[1] - k[m][1] * r[0]};
            const double size = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]), amplitude = std::pow(norm2, -5.0 / 12);
            for (int d = 0; d < 3; ++d) a[m][d] = amplitude * c[d] / size;
            sum += amplitude;
            phase[m] = two_pi * unit(rng);
            omega[m] = two_pi * std::pow(norm2, 1.0 / 3);
        }
        for (int m = 0; m < modes; ++m)
            for (int d = 0; d < 3; ++d) a[m][d] *= 3 / sum;  // peak speed about 1: rms sums, not all modes in phase
    }

    // v at (x, y, z) in the unit box at time t; `side` is the particle's
    // stream (streams only).
    KOKKOS_INLINE_FUNCTION void velocity(const double* p, double t, int side, double* v) const {
        const double x = p[0], y = p[1], z = p[2];
        v[0] = v[1] = v[2] = 0;
        // Periodic position relative to the centre of the box.
        const double dx = x - 0.5, dy = y - 0.5, dz = z - 0.5;
        switch (kind) {
            case Kind::shear:
                v[0] = Kokkos::sin(two_pi * z);
                break;
            case Kind::sawtooth:
                v[0] = 4 * Kokkos::abs(z - 0.5) - 1;
                break;
            case Kind::vortex: {
                const double r = Kokkos::sqrt(dx * dx + dy * dy);
                if (r < 0.45) {
                    const double f = 1 - r / 0.45, omega = 8 * f * f;  // peak speed about 1 at r = 0.15
                    v[0] = -omega * dy;
                    v[1] = omega * dx;
                }
                break;
            }
            case Kind::taylor_green:
                v[0] = Kokkos::sin(two_pi * x) * Kokkos::cos(two_pi * y) * Kokkos::cos(two_pi * z);
                v[1] = -Kokkos::cos(two_pi * x) * Kokkos::sin(two_pi * y) * Kokkos::cos(two_pi * z);
                break;
            case Kind::turbulence:
                for (int m = 0; m < modes; ++m) {
                    const double s = Kokkos::sin(two_pi * (k[m][0] * x + k[m][1] * y + k[m][2] * z) + phase[m] + omega[m] * t);
                    for (int d = 0; d < 3; ++d) v[d] += a[m][d] * s;
                }
                break;
            case Kind::collapse:
            case Kind::collapse_wide:
            case Kind::expansion:
            case Kind::noh: {
                const double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
                const double outer = kind == Kind::collapse_wide ? 0.45 : 0.16;
                if (r > 0 && r < outer) {
                    const double taper = Kokkos::min(1.0, (outer - r) / 0.04);
                    double speed = taper;
                    if (kind != Kind::noh) speed *= Kokkos::min(1.0, r / 0.02);
                    if (kind != Kind::expansion) speed = -speed;
                    v[0] = speed * dx / r;
                    v[1] = speed * dy / r;
                    v[2] = speed * dz / r;
                }
                break;
            }
            case Kind::pancake:
            case Kind::pancake_wide: {
                const double outer = kind == Kind::pancake ? 0.16 : 0.45;
                if (Kokkos::abs(dz) < outer)
                    v[2] = -Kokkos::tanh(dz / 1e-4) * Kokkos::min(1.0, (outer - Kokkos::abs(dz)) / 0.04);
                break;
            }
            case Kind::streams:
            case Kind::counter:
                v[0] = side;
                break;
            case Kind::thin:
                v[0] = Kokkos::sin(two_pi * y);
                v[2] = -0.25 * Kokkos::sin(4 * two_pi * z);
                break;
        }
    }
};

// One midpoint step of `field` on x (kept in the box); `side`: streams. Other
// displacements (a drift) are added by the caller.
template <class X, class S>
void advance(const Field& field, const X& x, const S& side, double t, double dt) {
    Kokkos::parallel_for(
        "distortion::advance", Kokkos::RangePolicy<Exec>(0, x.extent(0)), KOKKOS_LAMBDA(int i) {
            double p[3] = {x(i, 0), x(i, 1), x(i, 2)}, v[3], mid[3];
            field.velocity(p, t, side(i), v);
            for (int a = 0; a < 3; ++a) mid[a] = p[a] + 0.5 * dt * v[a];
            field.velocity(mid, t + 0.5 * dt, side(i), v);
            for (int a = 0; a < 3; ++a) {
                const double q = p[a] + dt * v[a];
                x(i, a) = q - Kokkos::floor(q);
            }
        });
}

// Device memory through the Kokkos Tools callbacks.
struct DeviceBytes {
    static inline std::int64_t current = 0, peak = 0;
    static inline std::map<const void*, std::uint64_t> live;
    static void allocate(Kokkos_Profiling_SpaceHandle h, const char*, const void* p, std::uint64_t size) {
        if (std::string(h.name) == "Host" && std::string(Memory::name()) != "Host") return;
        live[p] = size;
        current += static_cast<std::int64_t>(size);
        peak = std::max(peak, current);
    }
    static void deallocate(Kokkos_Profiling_SpaceHandle, const char*, const void* p, std::uint64_t) {
        const auto it = live.find(p);
        if (it == live.end()) return;
        current -= static_cast<std::int64_t>(it->second);
        live.erase(it);
    }
    static void start() {
        Kokkos::Tools::Experimental::set_allocate_data_callback(allocate);
        Kokkos::Tools::Experimental::set_deallocate_data_callback(deallocate);
    }
};

// The shape of the mesh's cells.
struct Shape {
    double aspect = 0;        // largest, over cells, of reach R / half the distance to the nearest neighbour
    double reach = 0;         // largest 2R of any cell / the side of the box (3/8 is the limit of a build)
    int max_faces = 0;        // most faces of a cell
    double smallest_gap = 0;  // the smallest distance between neighbours, in initial spacings
    int max_corners = 0;      // most corners of a face (tet_mesh::max_ring_length caps the walk)
};

template <class M>
Shape shape_of(const M& mesh, double spacing) {
    const int n = mesh.particles();
    const auto& tets = mesh.tetrahedra();
    const auto& state = mesh.repair_state();
    const auto alive = state.alive;
    const auto centre = tets.centre;
    const auto vertex = tets.vertex;
    const auto owner = mesh.owner();
    const auto neighbour = mesh.neighbour();
    const auto x = mesh.positions();
    const auto box = mesh.sides();
    const auto start = mesh.cell_face_start();
    Kokkos::View<double*, Memory> far("far", n), near("near", n);
    Kokkos::deep_copy(near, 1e300);
    Kokkos::parallel_for(
        "distortion::reach", Kokkos::RangePolicy<Exec>(0, state.count), KOKKOS_LAMBDA(int t) {
            if (!alive(t)) return;
            const double r2 = centre(t, 3);
            for (int c = 0; c < 4; ++c) Kokkos::atomic_max(&far(vertex(t, c)), r2);
        });
    Kokkos::parallel_for(
        "distortion::gap", Kokkos::RangePolicy<Exec>(0, mesh.faces()), KOKKOS_LAMBDA(int f) {
            const int i = owner(f), j = neighbour(f);
            double d2 = 0;
            for (int a = 0; a < 3; ++a) {
                double d = x(i, a) - x(j, a);
                d -= box.side[a] * Kokkos::nearbyint(d * box.inverse[a]);
                d2 += d * d;
            }
            Kokkos::atomic_min(&near(i), d2);
            Kokkos::atomic_min(&near(j), d2);
        });
    double aspect = 0, largest = 0, smallest = 1e300;
    Kokkos::parallel_reduce(
        "distortion::aspect", Kokkos::RangePolicy<Exec>(0, n),
        KOKKOS_LAMBDA(int i, double& a, double& l, double& s) {
            const double ratio = near(i) > 0 ? 2 * Kokkos::sqrt(far(i) / near(i)) : 1e300;
            if (ratio > a) a = ratio;
            if (far(i) > l) l = far(i);
            if (near(i) < s) s = near(i);
        },
        Kokkos::Max<double>(aspect), Kokkos::Max<double>(largest), Kokkos::Min<double>(smallest));
    int faces = 0;
    Kokkos::parallel_reduce(
        "distortion::faces", Kokkos::RangePolicy<Exec>(0, n),
        KOKKOS_LAMBDA(int i, int& most) {
            const int c = start(i + 1) - start(i);
            if (c > most) most = c;
        },
        Kokkos::Max<int>(faces));
    int corners = 0;
    const auto length = tets.ring_length;
    Kokkos::parallel_reduce(
        "distortion::corners", Kokkos::RangePolicy<Exec>(0, mesh.faces()),
        KOKKOS_LAMBDA(int f, int& most) {
            if (length(f) > most) most = length(f);
        },
        Kokkos::Max<int>(corners));
    return {aspect, 2 * std::sqrt(largest) / box.shortest(), faces, std::sqrt(smallest) / spacing, corners};
}

struct Options {
    std::string flow = "shear";
    int n = 4096;
    double displacement = 0.3;  // initial spacings per step at speed 1
    int steps = 100;
    int oracle_every = 0;   // the full oracle (mesh_check::check); 0: never
    int rebuild_every = 0;  // compare_with_rebuild, cheaper; 0: never
    int oracle_cells = 16;
    double drift = 0;       // regularisation in this driver (the library has none): each step
                            // moves each generator this fraction of the way to its centroid
    voronoi::Fallback fallback = voronoi::Fallback::cheapest;
    double seconds = 0;     // stop after this much wall time; 0: run all steps
    bool final_oracle = false;  // the full oracle on the last step
    bool dump = false;      // write the grid positions of a failing step
    bool print = true;      // one line per step
    std::ostream* out = &std::cout;
};

struct Summary {
    int steps_done = 0;
    bool refused = false;
    std::string refusal;
    int refused_step = -1;
    int wrong_step = -1;     // first step the oracle rejected
    std::string wrong;
    long long stuck = 0, host_flips = 0, host_rebuilds = 0, rebuilds = 0, flips = 0;
    long long rounds_max = 0;
    double aspect = 0, reach = 0, gap = 1e300;
    int max_faces = 0, max_corners = 0;
    double faces_first = 0, faces_last = 0, faces_max = 0;
    double bytes_first = 0, bytes_peak = 0;  // B/cell
    double rate_first = 0, rate_worst = 1e300, rate_mean = 0;  // M cells/s of the mesh step
    double closure = 0, volume = 0;
    double strain = 0;  // time in units of the box crossing
};

inline void initial_positions(const Options& o, Points& p) {
    const Kind kind = kind_of(o.flow);
    p = in_slot_order(make_points("jittered", o.n));
    if (kind == Kind::expansion) {
        // A ball of radius 0.05 holding half the points.
        std::mt19937_64 rng(3);
        std::normal_distribution<double> g(0, 1);
        std::uniform_real_distribution<double> u(0, 1);
        for (std::size_t i = 0; i < p.size(); i += 2) {
            const Point d{g(rng), g(rng), g(rng)};
            const double size = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            const double r = 0.05 * std::cbrt(u(rng));
            for (int a = 0; a < 3; ++a) p[i][a] = 0.5 + r * d[a] / size;
        }
        p = in_slot_order(p);
    }
}

inline Summary run(const Options& o) {
    Summary s;
    const Exec exec;
    Points points;
    initial_positions(o, points);
    const int n = static_cast<int>(points.size());
    const double spacing = std::cbrt(1.0 / n), dt = o.displacement * spacing;
    Positions x = to_device(points);
    Kokkos::View<int*, Memory> side("side", n);
    {
        auto h = Kokkos::create_mirror_view(side);
        const bool mixed = kind_of(o.flow) == Kind::counter;
        for (int i = 0; i < n; ++i) h(i) = mixed ? (i % 2 ? 1 : -1) : points[i][0] < 0.5 ? 1 : -1;
        Kokkos::deep_copy(side, h);
    }
    const Field field(kind_of(o.flow));
    Mesh mesh(exec, 1.0);
    mesh.set_fallback(o.fallback);
    std::ostream& out = *o.out;
    try {
        mesh.update(x);
    } catch (const std::exception& e) {
        s.refused = true;
        s.refused_step = 0;
        s.refusal = e.what();
        return s;
    }
    mesh.update_cells();
    s.faces_first = s.faces_max = double(mesh.faces()) / n;
    s.bytes_first = double(DeviceBytes::current) / n;
    DeviceBytes::peak = DeviceBytes::current;
    double t = 0;
    int rated = 0;
    Kokkos::Timer wall;
    for (int step = 1; step <= o.steps; ++step) {
        if (o.seconds > 0 && wall.seconds() > o.seconds) break;
        advance(field, x, side, t, dt);
        t += dt;
        if (o.drift > 0) {
            // Generators towards their centroids (stored relative to the generator).
            const auto centroid = mesh.centroid();
            const double chi = o.drift;
            const auto y = x;
            Kokkos::parallel_for(
                "distortion::drift", Kokkos::RangePolicy<Exec>(0, n), KOKKOS_LAMBDA(int i) {
                    for (int a = 0; a < 3; ++a) {
                        const double q = y(i, a) + chi * centroid(i, a);
                        y(i, a) = q - Kokkos::floor(q);
                    }
                });
        }
        exec.fence();
        voronoi::Timings timing;
        voronoi::Report r;
        try {
            r = mesh.update(x, {}, &timing);
            mesh.update_cells(&timing);
        } catch (const std::exception& e) {
            s.refused = true;
            s.refused_step = step;
            s.refusal = e.what();
            if (o.print) out << "refused: step=" << step << " t=" << t << " " << e.what() << "\n";
            break;
        }
        exec.fence();
        s.steps_done = step;
        s.strain = t;
        const double ms = timing.repair + timing.geometry + timing.cells, rate = n / ms / 1e3;
        const double total_ms = ms + timing.fallback + timing.build;
        s.flips += r.flips;
        s.stuck += r.stuck;
        s.host_flips += r.host_flips;
        s.host_rebuilds += r.host_rebuilds;
        s.rebuilds += r.rebuilt;
        s.rounds_max = std::max<long long>(s.rounds_max, r.rounds);
        const Shape shape = shape_of(mesh, spacing);
        s.aspect = std::max(s.aspect, shape.aspect);
        s.reach = std::max(s.reach, shape.reach);
        s.max_faces = std::max(s.max_faces, shape.max_faces);
        s.max_corners = std::max(s.max_corners, shape.max_corners);
        s.gap = std::min(s.gap, shape.smallest_gap);
        s.faces_last = double(mesh.faces()) / n;
        s.faces_max = std::max(s.faces_max, s.faces_last);
        s.bytes_peak = std::max(s.bytes_peak, double(DeviceBytes::peak) / n);
        if (step == 1) s.rate_first = rate;
        if (!r.rebuilt) {
            s.rate_worst = std::min(s.rate_worst, n / total_ms / 1e3);
            s.rate_mean += n / total_ms / 1e3;
            ++rated;
        }
        std::string verdict;
        const bool oracle = o.oracle_every > 0 && step % o.oracle_every == 0;
        const bool rebuild = o.rebuild_every > 0 && step % o.rebuild_every == 0;
        if (oracle) {
            const mesh_check::Result c = mesh_check::check(mesh, o.oracle_cells);
            s.closure = std::max(s.closure, c.closure);
            s.volume = std::max(s.volume, c.fresh_volume);
            verdict = c.ok() ? " oracle=PASS radius=" + sci(double(c.tets.radius)) : " oracle=FAIL " + c.text();
            if (!c.ok() && o.dump) {
                const auto g = mesh_check::grid_of(mesh);
                std::ofstream file("oracle_fail_" + o.flow + "_" + std::to_string(step) + ".txt");
                file << g.period[0] << " " << g.period[1] << " " << g.period[2] << "\n";
                for (const auto& q : g.p) file << q[0] << " " << q[1] << " " << q[2] << "\n";
            }
            if (!c.ok() && s.wrong_step < 0) {
                s.wrong_step = step;
                s.wrong = c.text();
            }
        }
        if (rebuild) {
            const Comparison c = compare_with_rebuild(mesh, mesh_check::snapped(mesh));
            s.volume = std::max(s.volume, c.volume);
            verdict += c.pass() && std::abs(c.total - 1) < 1e-11 ? " rebuild=PASS" : " rebuild=FAIL";
            if ((!c.pass() || std::abs(c.total - 1) >= 1e-11) && s.wrong_step < 0) {
                s.wrong_step = step;
                std::ostringstream text;
                text << c;
                s.wrong = text.str();
            }
            verdict += " total-1=" + sci(c.total - 1) + " volume_error=" + sci(c.volume);
        }
        if (o.print)
            out << "step=" << step << " t=" << t << " flips=" << r.flips << " rounds=" << r.rounds << " stuck=" << r.stuck
                << " host=" << r.host_flips << "/" << r.host_rebuilds << " rebuilt=" << r.rebuilt
                << " Mcells_s=" << n / total_ms / 1e3 << " repair_ms=" << timing.repair << " geometry_ms=" << timing.geometry
                << " cells_ms=" << timing.cells << " fallback_ms=" << timing.fallback << " build_ms=" << timing.build
                << " faces_per_cell=" << s.faces_last << " tets_per_cell=" << double(mesh.repair_state().count) / n
                << " B_per_cell=" << double(DeviceBytes::current) / n << " aspect=" << shape.aspect
                << " reach=" << shape.reach << " max_faces=" << shape.max_faces << " max_corners=" << shape.max_corners
                << " gap=" << shape.smallest_gap << verdict
                << "\n"
                << std::flush;
    }
    if (rated > 0) s.rate_mean /= rated;
    if (o.final_oracle && !s.refused) {
        const mesh_check::Result c = mesh_check::check(mesh, o.oracle_cells);
        s.closure = std::max(s.closure, c.closure);
        s.volume = std::max(s.volume, c.fresh_volume);
        out << "final oracle: " << (c.ok() ? "PASS " : "FAIL ") << c.text() << "\n" << std::flush;
        if (!c.ok() && s.wrong_step < 0) {
            s.wrong_step = s.steps_done;
            s.wrong = c.text();
        }
    }
    return s;
}

inline std::string text(const Options& o, const Summary& s) {
    std::ostringstream t;
    t << o.flow << " n=" << o.n << " d=" << o.displacement << " drift=" << o.drift << ": steps=" << s.steps_done
      << " t=" << s.strain << (s.refused ? " REFUSED at " + std::to_string(s.refused_step) + " (" + s.refusal + ")" : "")
      << " flips=" << s.flips << " max_rounds=" << s.rounds_max << " stuck=" << s.stuck << " host=" << s.host_flips << "/"
      << s.host_rebuilds << " rebuilds=" << s.rebuilds << " aspect=" << s.aspect << " reach=" << s.reach
      << " max_faces=" << s.max_faces << " max_corners=" << s.max_corners << " gap=" << s.gap << " faces_per_cell=" << s.faces_first << "->" << s.faces_last
      << " (max " << s.faces_max << ") B_per_cell=" << s.bytes_first << " peak " << s.bytes_peak
      << " Mcells_s first=" << s.rate_first << " mean=" << s.rate_mean << " worst=" << s.rate_worst
      << " closure=" << s.closure << " volume=" << s.volume;
    if (s.wrong_step >= 0) t << " WRONG at step " << s.wrong_step << ": " << s.wrong;
    return t.str();
}

}  // namespace distortion
