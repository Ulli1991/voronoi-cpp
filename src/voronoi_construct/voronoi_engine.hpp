#pragma once

#include "voronoi_mesh.hpp"

#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>

// The run-time choice of where the mesh runs (docs/ideas.md, "CPU and GPU together"): voronoi::Engine<D> holds a
// Mesh<D, host space> or a Mesh<D, device space>, picked by Options at construction, behind one set of calls.
// Mesh<D, Exec> stays the compile-time interface; Engine adds nothing to it but the choice and the copies.
namespace voronoi {

// Where the mesh runs. cpu: Kokkos::DefaultHostExecutionSpace (OpenMP in a build with both backends); gpu:
// Kokkos::DefaultExecutionSpace, fused arithmetic (the fastest); hybrid: gpu in
// the host's bits (Layout::same_bits_as_host), with the host taking work as Options::tail says (until the hybrid tail
// exists, hybrid runs as gpu in the host's bits).
enum class Backend { cpu, gpu, hybrid };
// What the host takes in hybrid: nothing, the clipping's tail, or the tail when the last build's count of tail cells
// says so (a count, never a clock: the same input takes the same path).
enum class Tail { off, on, automatic };

// What Engine runs on. Used by: Engine, callers.
struct Options {
    Backend backend = Backend::gpu;
    Tail tail = Tail::off;
    int host_threads = 0;  // cpu: OpenMP threads of the mesh's instance; 0 = all of Kokkos' host threads
};

// "cpu", "gpu" or "hybrid"; anything else is a std::invalid_argument. Used by: callers' flags, modes_test.
inline Backend parse_backend(const std::string& name) {
    if (name == "cpu") return Backend::cpu;
    if (name == "gpu") return Backend::gpu;
    if (name == "hybrid") return Backend::hybrid;
    throw std::invalid_argument("voronoi: backend must be cpu, gpu or hybrid, not '" + name + "'");
}

// The backend's name, as parse_backend reads it. Used by: messages, modes_test.
inline const char* name(Backend b) { return b == Backend::cpu ? "cpu" : b == Backend::gpu ? "gpu" : "hybrid"; }

// Whether this build has a device space apart from the host one (gpu and hybrid need it).
constexpr bool has_device() {
    return !std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>;
}

// The mesh in D dimensions on the backend Options picks, with Mesh<D>'s calls. Positions may live in any memory
// space: they are copied to the mesh's when it cannot read them. Results come back as host copies;
// visit() hands the mesh itself to code that wants its views where they are.
template <int D>
class Engine {
public:
    using Host = Kokkos::DefaultHostExecutionSpace;
    using Device = Kokkos::DefaultExecutionSpace;
    using HostMesh = Mesh<D, Host>;
    using DeviceMesh = Mesh<D, Device>;
    using Any = std::conditional_t<has_device(), std::variant<HostMesh, DeviceMesh>, std::variant<HostMesh>>;
    using Box = exact_predicates::BoxN<D>;

    // A std::invalid_argument for gpu or hybrid in a build without a device space, or for negative host_threads.
    Engine(const Box& box, const Options& options = {}) : options_(options), mesh_(make(box, options)) {
        // hybrid: the GPU computes the geometry, and clips, in the host's bits (Layout::same_bits_as_host), so the
        // host's share is indistinguishable; cpu and gpu keep their fastest arithmetic. 1D has nothing to fuse.
        if constexpr (D > 1)
            if (options.backend == Backend::hybrid) std::visit([](auto& m) { m.set_same_bits_as_host(true); }, mesh_);
    }
    Engine(double side, const Options& options = {}) : Engine(Box(side), options) {}

    const Options& options() const { return options_; }
    Backend backend() const { return options_.backend; }

    // As Mesh<D>::update, x(i, a) in any memory space.
    template <class X>
    Report update(const X& x) {
        return std::visit([&](auto& m) { return m.update(on_space(m, x)); }, mesh_);
    }
    template <class X>
    Report rebuild(const X& x) {
        return std::visit([&](auto& m) { return m.rebuild(on_space(m, x)); }, mesh_);
    }
    void update_cells() {
        std::visit([](auto& m) { m.update_cells(); }, mesh_);
    }

    int particles() const {
        return std::visit([](const auto& m) { return m.particles(); }, mesh_);
    }
    int faces() const {
        return std::visit([](const auto& m) { return m.faces(); }, mesh_);
    }
    // Host views of the last update's results (see Mesh<D>); area() may have spare rows past faces().
    auto owner() const {
        return host([](const auto& m) { return m.owner(); });
    }
    auto neighbour() const {
        return host([](const auto& m) { return m.neighbour(); });
    }
    auto area() const {
        return host([](const auto& m) { return m.area(); });
    }
    auto face_centroid() const {
        return host([](const auto& m) { return m.face_centroid(); });
    }
    auto volume() const {
        return host([](const auto& m) { return m.volume(); });
    }
    auto centroid() const {
        return host([](const auto& m) { return m.centroid(); });
    }
    auto cell_face_start() const {
        return host([](const auto& m) { return m.cell_face_start(); });
    }
    auto cell_face() const {
        return host([](const auto& m) { return m.cell_face(); });
    }

    // f(mesh) with the Mesh<D, space> this engine runs; f must accept both kinds.
    template <class F>
    decltype(auto) visit(F&& f) {
        return std::visit(std::forward<F>(f), mesh_);
    }
    template <class F>
    decltype(auto) visit(F&& f) const {
        return std::visit(std::forward<F>(f), mesh_);
    }

private:
    static Any make(const Box& box, const Options& o) {
        if (o.host_threads < 0) throw std::invalid_argument("voronoi::Engine: host_threads must be 0 or more");
        if (o.backend == Backend::cpu) return Any(std::in_place_index<0>, host_space(o.host_threads), box);
        if constexpr (has_device())
            return Any(std::in_place_index<1>, Device{}, box);
        else
            throw std::invalid_argument(std::string("voronoi::Engine: no device backend in this build for ") +
                                        name(o.backend));
    }
    // The host instance: Kokkos' own, or (OpenMP) one of `threads` threads.
    static Host host_space(int threads) {
#if defined(KOKKOS_ENABLE_OPENMP)
        if constexpr (std::is_same_v<Host, Kokkos::OpenMP>)
            if (threads > 0 && threads < Host().concurrency()) return Kokkos::OpenMP(threads);
#endif
        (void)threads;
        return Host{};
    }
    // x as a view the mesh's space can read: itself, or a copy in the mesh's memory (kept between calls).
    template <class M, class X>
    auto on_space(const M& m, const X& x) {
        (void)m;
        using Memory = typename M::Memory;
        using Rows = Kokkos::View<double* [D], Memory>;
        if constexpr (Kokkos::SpaceAccessibility<Memory, typename X::memory_space>::accessible) {
            return x;
        } else {
            auto& copy = std::get<std::is_same_v<Memory, typename Host::memory_space> ? 0 : 1>(copies_);
            if (copy.extent(0) != x.extent(0))
                copy =
                    Rows(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("engine_positions")), x.extent(0));
            // Between spaces Kokkos copies only contiguous views of one layout (a host View<double*[D]> is LayoutRight,
            // a CUDA one LayoutLeft): otherwise through host staging, re-laid out on the host, then copied whole.
            if constexpr (std::is_same_v<typename X::array_layout, typename Rows::array_layout>) {
                if (x.span_is_contiguous()) {
                    Kokkos::deep_copy(copy, x);
                    return copy;
                }
            }
            const auto from = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
            const auto to = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, copy);
            Kokkos::deep_copy(Host{}, to, from);
            Host{}.fence();
            Kokkos::deep_copy(copy, to);
            return copy;
        }
    }
    // get(mesh) copied to a host view of one type for both meshes (row-major, writable).
    template <class Get>
    auto host(const Get& get) const {
        using V = decltype(get(std::get<0>(mesh_)));
        using Out = Kokkos::View<typename V::non_const_data_type, Kokkos::LayoutRight, Kokkos::HostSpace>;
        return std::visit(
            [&](const auto& m) {
                const auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, get(m));
                Out out(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(mirror.label())), mirror.extent(0));
                Kokkos::deep_copy(out, mirror);
                return out;
            },
            mesh_);
    }

    Options options_;
    Any mesh_;
    std::tuple<Kokkos::View<double* [D], typename Host::memory_space>,
               Kokkos::View<double* [D], typename Device::memory_space>>
        copies_;
};

}  // namespace voronoi
