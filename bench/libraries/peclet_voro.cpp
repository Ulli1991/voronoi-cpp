// peclet-voro (E.A.J.F. Peters, TU Eindhoven; doi:10.5281/zenodo.21132443) in the library benchmark: its Kokkos
// device engine called directly from C++ (no Python layer), on the CUDA backend (A100) or the OpenMP backend (CPU).
//   peclet_voro build <points file> [--variant=plain|resident] [--repeats=3] [--volumes=file]
//   peclet_voro step <trajectory file> [--volumes=file]
// File formats and the record of each timed run: libbench.hpp.
//
// What is called (include/peclet/voro/tessellator.hpp, repair.hpp):
//  - plain: buildTessellation<double, false>() = the library's cold build: counting-sort grid, per-seed worklist
//    gather + ConvexCell clipping, published CSR (cell volumes, per-facet neighbour ids and area vectors). Same
//    defaults as the Python Tessellation (search window kSearchWindow = 4, density from N), withForceGeom = false
//    (no dV / connector vectors: only what the other libraries give).
//  - resident: MovingTessellation::alloc + rebuild, which is what peclet.voro.Tessellation.build() runs: the cold
//    build plus the resident topology store, near-miss lists and edge adjacency that its incremental repair needs.
//  - step: MovingTessellation::step(pos) per trajectory frame (Tessellation.step()): re-evaluation over the
//    stored topology, convexity certificate, two-pass gather repair, verify, rebuild fallback; volumes in mt.vol.
//    Certificate tolerance and Verlet skin are the Python defaults (1e-4 and 0.25 mean spacings).
// Positions: the periodic unit box, native (origin-anchored, periodic on every axis), double precision.

#include "libbench.hpp"

#include <Kokkos_Core.hpp>
#include <cmath>

#include "peclet/core/common/view.hpp"
#include "peclet/voro/params.hpp"
#include "peclet/voro/repair.hpp"
#include "peclet/voro/tessellator.hpp"

namespace {

using Real = double;
using Mem = peclet::core::MemSpace;
using DView = Kokkos::View<Real*, Mem>;
using MT = peclet::voro::MovingTessellation<Real, peclet::voro::kMaxPlanes, peclet::voro::kMaxTriangles, false,
                                            peclet::voro::NoSdf>;

#ifdef KOKKOS_ENABLE_CUDA
using Peak = libbench::DevicePeak;
const char* const device_name = "A100-SXM4-40GB";
#else
using Peak = libbench::HostPeak;
const char* const device_name = "2x Xeon Platinum 8360Y";
#endif

const Real L[3] = {1, 1, 1};

libbench::Setup setup(const std::string& variant, const std::string& outputs, const std::string& timed,
                      const std::string& note) {
    libbench::Setup s;
    s.library = "peclet-voro";
    s.variant = variant;
    s.version = libbench::env("PECLET_VORO_VERSION", "v1.0.4-1-gd7371c9");
    s.device = device_name;
#ifdef KOKKOS_ENABLE_CUDA
    s.threads = 1;
#else
    s.threads = Kokkos::DefaultExecutionSpace().concurrency();
#endif
    s.outputs = outputs;
    s.periodic = "native";
    s.precision = "double, floating-point clipping (no exact predicates)";
    s.timed = timed;
    s.note = note;
    return s;
}

DView to_device(const double* xyz, std::uint64_t n) {
    DView d(Kokkos::view_alloc(std::string("pos"), Kokkos::WithoutInitializing), n * 3);
    Kokkos::deep_copy(d, Kokkos::View<const double*, Kokkos::HostSpace>(xyz, n * 3));
    return d;
}

// Status counts of a build (StatusBit), as extra JSON fields, and the volumes with NaN for cells the build did not
// produce (overflow or empty).
std::string status_counts(const Kokkos::View<int*, Mem>& status, const DView& vol, std::vector<double>* volume) {
    auto s = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, status);
    auto v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, vol);
    long overflow = 0, empty = 0, incomplete = 0, reach = 0;
    for (std::size_t i = 0; i < s.extent(0); ++i) {
        overflow += (s(i) & peclet::voro::kOverflow) != 0;
        empty += (s(i) & peclet::voro::kEmpty) != 0;
        incomplete += (s(i) & peclet::voro::kIncomplete) != 0;
        reach += (s(i) & peclet::voro::kReachExceeded) != 0;
        if (volume)
            (*volume)[i] = (s(i) & (peclet::voro::kOverflow | peclet::voro::kEmpty)) ? NAN : v(i);
    }
    std::ostringstream o;
    o << ",\"overflow\":" << overflow << ",\"empty\":" << empty << ",\"incomplete\":" << incomplete
      << ",\"reach_exceeded\":" << reach;
    return o.str();
}

void alloc(MT& mt, int n) {
    const Real spacing = std::cbrt(1.0 / n);
    mt.alloc(n, L, Real(peclet::voro::kCertificateTolerance) * spacing, Real(peclet::voro::kSkin) * spacing,
             peclet::voro::kSearchWindow, n);
}

int build(int argc, char** argv) {
    const std::string file = argv[2];
    const std::string variant = libbench::option(argc, argv, "variant", "plain");
    const int repeats = std::stoi(libbench::option(argc, argv, "repeats", "3"));
    const auto frames = libbench::read_frames(file);
    const std::uint64_t n = frames.n;
    const int N = static_cast<int>(n);
    const DView pos = to_device(frames.frame(0), n);
    std::vector<double> volume(n);
    std::string counts;
    libbench::Setup s;
    std::function<void(bool)> body;
    if (variant == "plain") {
        s = setup("cold build (buildTessellation)", "volumes faces areas",
                  "points on the device to cell volumes and facet CSR (neighbour ids, area vectors) on the device",
                  "search window 4, withForceGeom off");
        body = [&](bool copy) {
            auto res = peclet::voro::buildTessellation<Real, false>(pos, DView{}, N, L, peclet::voro::kSearchWindow,
                                                                   N, {}, peclet::voro::NoSdf{}, false);
            Kokkos::fence();
            if (copy) counts = status_counts(res.status, res.view.cellVolume, &volume);
        };
    } else {
        s = setup("resident build (MovingTessellation alloc + rebuild = Tessellation.build)",
                  "volumes faces (resident topology store)",
                  "points on the device to resident tessellation (topology store, volumes, adjacency) on the device",
                  "search window 4, tolerance 1e-4 and skin 0.25 mean spacings; includes alloc of the store");
        body = [&](bool copy) {
            MT mt;
            alloc(mt, N);
            mt.rebuild(pos);
            Kokkos::fence();
            if (copy) counts = status_counts(mt.lastStatus, mt.vol, &volume);
        };
    }
    body(true);  // warm-up, the volumes and the status counts
    libbench::write_volumes(libbench::option(argc, argv, "volumes"), volume);
    const double start = libbench::now();
    const double peak = [&] {
        Peak p;
        p.start();
        body(false);
        Kokkos::fence();
        return p.stop();
    }();
    while (libbench::now() - start < 2.0) body(false);  // warm-up as libbench::run_protocol: clocks ramp up
    for (int r = 1; r <= repeats; ++r) {
        const double t0 = libbench::now();
        body(false);
        const double seconds = libbench::now() - t0;
        libbench::record(s, file, libbench::input_name(file), n, "build", r, seconds, peak, counts);
    }
    return 0;
}

int step(int argc, char** argv) {
    const std::string file = argv[2];
    const auto frames = libbench::read_frames(file);
    const std::uint64_t n = frames.n;
    const int N = static_cast<int>(n);
    auto s = setup("moving step (MovingTessellation::step = Tessellation.step)", "volumes faces (resident topology store)",
                   "positions on the device to repaired resident tessellation and volumes on the device, per step",
                   "tolerance 1e-4 and skin 0.25 mean spacings (Python defaults), adaptive rebuild gate on");
    DView pos = to_device(frames.frame(0), n);
    MT mt;
    alloc(mt, N);
    // Warm-up as libbench::run_protocol (2 s of untimed rebuilds of frame 0), then the resident start.
    for (const double start = libbench::now(); libbench::now() - start < 2.0;) mt.rebuild(pos);
    mt.rebuild(pos);
    Kokkos::fence();
    Peak peak;
    peak.start();
    for (std::uint64_t f = 1; f < frames.frames; ++f) {
        Kokkos::deep_copy(pos, Kokkos::View<const double*, Kokkos::HostSpace>(frames.frame(f), n * 3));
        Kokkos::fence();
        const double t0 = libbench::now();
        const peclet::voro::RepairStats st = mt.step(pos);
        Kokkos::fence();
        const double seconds = libbench::now() - t0;
        std::ostringstream extra;
        extra << ",\"flagged\":" << st.pass1Raw << ",\"pass1\":" << st.pass1 << ",\"pass2\":" << st.pass2
              << ",\"extra\":" << st.extra << ",\"verify_passes\":" << st.verifyPasses
              << ",\"route\":" << static_cast<int>(st.route) << ",\"fell_back\":" << (st.fellBack ? 1 : 0);
        libbench::record(s, file, libbench::input_name(file), n, "step", static_cast<int>(f), seconds, -1,
                         extra.str());
    }
    std::cout << "{\"library\":\"peclet-voro\",\"variant\":\"" << s.variant << "\",\"step_peak_bytes\":" << peak.stop()
              << ",\"n\":" << n << "}" << std::endl;
    auto v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mt.vol);
    std::vector<double> volume(n);
    for (std::uint64_t i = 0; i < n; ++i) volume[i] = v(i);
    libbench::write_volumes(libbench::option(argc, argv, "volumes"), volume);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    const std::string mode = argc > 2 ? argv[1] : "";
    if (mode == "build") return build(argc, argv);
    if (mode == "step") return step(argc, argv);
    std::cerr << "usage: peclet_voro build <points file> [--variant=plain|resident] [--repeats=3] [--volumes=file]\n"
                 "       peclet_voro step <trajectory file> [--volumes=file]\n";
    return 2;
}
