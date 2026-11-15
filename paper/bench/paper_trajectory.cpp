// A trajectory from a point file: the points in slot order, then `steps` frames of the bench flow, each moving them
// `spacings` mean spacings, as `library_bench trajectory` makes from a generator (the paper's sweep needs blue noise).
//   paper_trajectory <points file> <spacings> <steps> <trajectory file>

#include "../../bench/libraries/libbench.hpp"
#include "support.hpp"

// Flow::advect in blocks on up to 16 host threads (the same result, as voronoi_bench's advect).
void advect(const support::Flow& flow, support::Points& points, double distance) {
    const std::size_t threads = std::clamp(std::thread::hardware_concurrency(), 1u, 16u);
    const std::size_t block = (points.size() + threads - 1) / threads;
    std::vector<std::thread> pool;
    for (std::size_t begin = 0; begin < points.size(); begin += block)
        pool.emplace_back([&, begin] {
            support::Points part(points.begin() + begin, points.begin() + std::min(points.size(), begin + block));
            flow.advect(part, distance);
            std::copy(part.begin(), part.end(), points.begin() + begin);
        });
    for (auto& t : pool) t.join();
}

int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    if (argc < 5) {
        std::cerr << "usage: paper_trajectory <points file> <spacings> <steps> <trajectory file>\n";
        return 2;
    }
    using namespace support;
    const auto frames = libbench::read_frames(argv[1]);
    Points p(frames.n);
    for (std::uint64_t i = 0; i < frames.n; ++i)
        for (int a = 0; a < 3; ++a) p[i][a] = frames.frame(0)[3 * i + a];
    p = in_slot_order(p);
    const double distance = std::stod(argv[2]) * std::cbrt(1.0 / double(p.size()));
    const int steps = std::stoi(argv[3]);
    std::vector<double> xyz;
    xyz.reserve(3 * p.size() * (steps + 1));
    const Flow flow;
    for (int s = 0; s <= steps; ++s) {
        if (s > 0) advect(flow, p, distance);
        for (const auto& q : p) xyz.insert(xyz.end(), q.begin(), q.end());
    }
    libbench::write_frames(argv[4], p.size(), xyz);
    std::cout << "wrote " << steps + 1 << " frames of " << p.size() << " points to " << argv[4] << "\n";
    return 0;
}
