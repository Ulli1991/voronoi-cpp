// mesh_cases_test group `reproducibility`: registers its cases with add(); shared helpers in common.hpp.

#include "mesh_cases/common.hpp"

namespace mesh_cases {

// ------------------------------------------------------------- reproducibility

// Runs `steps` flow steps on execution space E and returns a (topology, geometry) hash pair per step.
// Used by: reproducibility_cases (E = default space against Serial).
template <class E>
std::vector<std::uint64_t> trace(const Points& start, double displacement, int steps) {
    using Memory = typename E::memory_space;
    voronoi::Mesh<3, E> mesh(E{}, 1.0);
    Points p = start;
    const Flow flow;
    std::vector<std::uint64_t> hashes;
    auto put = [](std::uint64_t& h, std::uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    auto bits = [](double d) {
        std::uint64_t u;
        std::memcpy(&u, &d, sizeof u);
        return u;
    };
    Kokkos::View<double* [3], Memory> x("x", p.size());
    for (int s = 0; s <= steps; ++s) {
        auto h = Kokkos::create_mirror_view(x);
        for (std::size_t i = 0; i < p.size(); ++i)
            for (int a = 0; a < 3; ++a) h(i, a) = p[i][a];
        Kokkos::deep_copy(x, h);
        mesh.update(x);
        mesh.update_cells();
        const auto volume = to_host(mesh.volume());
        const auto centroid = to_host(mesh.centroid());
        const auto start_of = to_host(mesh.cell_face_start());
        const auto owner = to_host(mesh.owner());
        const auto neighbour = to_host(mesh.neighbour());
        const auto face = to_host(mesh.cell_face());
        std::uint64_t topology = 1469598103934665603ull, geometry = topology;
        for (std::size_t i = 0; i < p.size(); ++i) {
            put(geometry, bits(volume(i)));
            for (int a = 0; a < 3; ++a) put(geometry, bits(centroid(i, a)));
            for (int k = start_of(i); k < start_of(i + 1); ++k) {
                const int f = face(k);
                put(topology, static_cast<std::uint64_t>(owner(f) == static_cast<int>(i) ? neighbour(f) : owner(f)));
            }
        }
        hashes.push_back(topology);
        hashes.push_back(geometry);
        flow.advect(p, displacement / 16);
    }
    return hashes;
}

// The same run twice, against Serial, and a hash dump to compare builds and thread counts.
void reproducibility_cases() {
    const std::vector<std::pair<std::string, double>> runs{{"jittered", 0.3}, {"clustered", 0.3}, {"lattice", 0.1}, {"wave", 0.3}};
    for (const auto& [kind, displacement] : runs) {
        const std::string name = kind;
        const double d = displacement;
        add("repro.runs." + name, [=](Check& c) {
            const Points p = make_points(name, 4096);
            const auto a = trace<Exec>(p, d, 20), b = trace<Exec>(p, d, 20);
            c.expect(a == b, name + ": two runs give the same bits, 20 steps");
        });
#ifdef KOKKOS_ENABLE_SERIAL
        add("repro.serial." + name, [=](Check& c) {
            const Points p = make_points(name, 4096);
            const auto a = trace<Exec>(p, d, 20);
            const auto b = trace<Kokkos::Serial>(p, d, 20);
            long long topology = 0, geometry = 0;
            for (std::size_t i = 0; i < a.size(); i += 2) {
                topology += a[i] != b[i];
                geometry += a[i + 1] != b[i + 1];
            }
            c.expect(topology == 0, name + ": faces of " + Exec::name() + " and Serial agree (" + std::to_string(topology) + " of 21 steps differ)");
            c.expect(geometry == 0, name + ": bits of volumes and centroids agree (" + std::to_string(geometry) + " of 21 steps differ)");
        }, on_cuda(27));
#endif
        add("repro.dump." + name, [=](Check& c) {
            // Hashes to compare builds and thread counts: MESH_REPRO_DUMP=<file> appends "<case> <step> <topology> <geometry>".
            const char* path = std::getenv("MESH_REPRO_DUMP");
            const auto a = trace<Exec>(make_points(name, 4096), d, 20);
            if (path) {
                std::ofstream out(path, std::ios::app);
                for (std::size_t i = 0; i < a.size(); i += 2)
                    out << name << " " << i / 2 << " " << a[i] << " " << a[i + 1] << "\n";
            }
            c.expect(true, name + ": hashes written");
        });
    }
}

}  // namespace mesh_cases
