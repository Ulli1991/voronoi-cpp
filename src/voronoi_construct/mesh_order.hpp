#pragma once

#include "exact_predicates.hpp"
#include "neighbor_search.hpp"
#include "tet_mesh.hpp"
#include "tet_repair.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// Memory order of the 3D mesh (#34, ideas R15): cells sorted by a space-filling curve or by reverse Cuthill-McKee on the
// Voronoi graph, tetrahedra after their lowest cell. Called by voronoi::Mesh<3> (rebuild, and update when a re-sort is due).
namespace mesh_order {

// The order of cells: Morton (the cell list's curve), Hilbert, Moore (a closed Hilbert loop), Morton read as a Gray
// code, row-major (x fastest; a floor), or reverse Cuthill-McKee on the Voronoi neighbours.
enum class Curve { morton, hilbert, moore, gray, row_major, rcm };

// Bits per axis of a curve key: the top 21 of the 30-bit snapped coordinates, 63 bits in all.
constexpr int axis_bits = 21;

// The Hilbert index of (x, y, z) on a 2^b grid (Skilling 2004, "Programming the Hilbert curve": axes to transpose,
// then the transpose's bits interleaved). Used by: key, MooreTable::build.
KOKKOS_INLINE_FUNCTION std::uint64_t hilbert(std::uint32_t x, std::uint32_t y, std::uint32_t z, int b) {
    std::uint32_t X[3] = {x, y, z};
    for (std::uint32_t Q = 1u << (b - 1); Q > 1; Q >>= 1) {
        const std::uint32_t P = Q - 1;
        for (int i = 0; i < 3; ++i) {
            if (X[i] & Q) {
                X[0] ^= P;
            } else {
                const std::uint32_t t = (X[0] ^ X[i]) & P;
                X[0] ^= t;
                X[i] ^= t;
            }
        }
    }
    X[1] ^= X[0];
    X[2] ^= X[1];
    std::uint32_t t = 0;
    for (std::uint32_t Q = 1u << (b - 1); Q > 1; Q >>= 1)
        if (X[2] & Q) t ^= Q - 1;
    for (int i = 0; i < 3; ++i) X[i] ^= t;
    return neighbor_search::morton(X[0], X[1], X[2]);
}

// The rank r of Gray code g = r ^ (r >> 1). Used by: key (Curve::gray).
KOKKOS_INLINE_FUNCTION std::uint64_t gray_rank(std::uint64_t g) {
    for (int s = 1; s < 64; s <<= 1) g ^= g >> s;
    return g;
}

// The Moore curve: the eight octants in the loop 0, 1, 3, 2, 6, 7, 5, 4 (a Gray code), each a Hilbert curve turned so
// that it ends next to where the following one starts, the last next to the first. Built on the host by search.
struct MooreTable {
    int octant[8];            // octant (x bit 4, y bit 2, z bit 1) at loop place r
    int place[8];             // loop place of each octant
    int axis[8][3], flip[8];  // per octant: local axis a reads axis axis[a], reversed if bit a of flip

    // The search: each octant's turn so that the exit of place r and the entry of place r + 1 are neighbouring cells.
    static MooreTable build(int b) {
        MooreTable m{};
        const std::uint32_t top = (1u << b) - 1;
        int entry = 0, exit = 0;  // corners of the Hilbert curve, bits x 4, y 2, z 1
        std::uint64_t low = ~0ull, high = 0;
        for (int c = 0; c < 8; ++c) {
            const std::uint64_t h = hilbert(c & 4 ? top : 0, c & 2 ? top : 0, c & 1 ? top : 0, b);
            if (h < low) low = h, entry = c;
            if (h >= high) high = h, exit = c;
        }
        static constexpr int axes[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
        // The cell corner (bits) that a turn sends a curve corner to.
        auto corner = [](const int* axis, int flip, int c) {
            int out = 0;
            for (int a = 0; a < 3; ++a) out |= ((c >> (2 - a) & 1) ^ (flip >> a & 1)) << (2 - axis[a]);
            return out;
        };
        for (int r = 0; r < 8; ++r) {
            m.octant[r] = r ^ (r >> 1);
            m.place[m.octant[r]] = r;
        }
        // On the face two octants share, both corners sit at the bit opposite their own octant's, the rest agree.
        auto meets = [&](int r, int out, int in) {
            const int o = m.octant[r], p = m.octant[(r + 1) % 8], shared = o ^ p;
            return (out & shared) == (~o & shared) && (in & shared) == (~p & shared) &&
                   (out & ~shared & 7) == (in & ~shared & 7);
        };
        int choice[8], ends[8][2];
        auto search = [&](auto&& self, int r) -> bool {
            if (r == 8) return meets(7, ends[7][1], ends[0][0]);
            for (int k = 0; k < 48; ++k) {
                const int* axis = axes[k / 8];
                ends[r][0] = corner(axis, k % 8, entry);
                ends[r][1] = corner(axis, k % 8, exit);
                if (r > 0 && !meets(r - 1, ends[r - 1][1], ends[r][0])) continue;
                choice[r] = k;
                if (self(self, r + 1)) return true;
            }
            return false;
        };
        if (!search(search, 0)) throw std::logic_error("mesh_order::MooreTable: no closed loop");
        for (int r = 0; r < 8; ++r) {
            const int o = m.octant[r];
            for (int a = 0; a < 3; ++a) m.axis[o][a] = axes[choice[r] / 8][a];
            m.flip[o] = choice[r] % 8;
        }
        return m;
    }
};

// The curve key of a cell at grid coordinates c[0..2] (axis_bits each). Used by: cell_order, tests.
KOKKOS_INLINE_FUNCTION std::uint64_t key(Curve curve, const std::uint32_t* c, const MooreTable& moore) {
    constexpr int b = axis_bits;
    switch (curve) {
        case Curve::hilbert:
            return hilbert(c[0], c[1], c[2], b);
        case Curve::moore: {
            const int o = int(c[0] >> (b - 1)) << 2 | int(c[1] >> (b - 1)) << 1 | int(c[2] >> (b - 1));
            const std::uint32_t top = (1u << (b - 1)) - 1;
            std::uint32_t local[3];
            // Local axis a reads cell axis axis[a] (bit a of flip: reversed), as MooreTable::corner turns the corners.
            for (int a = 0; a < 3; ++a) {
                const std::uint32_t v = c[moore.axis[o][a]] & top;
                local[a] = (moore.flip[o] >> a & 1) ? top - v : v;
            }
            return static_cast<std::uint64_t>(moore.place[o]) << (3 * (b - 1)) |
                   hilbert(local[0], local[1], local[2], b - 1);
        }
        case Curve::gray:
            return gray_rank(neighbor_search::morton(c[0], c[1], c[2]));
        case Curve::row_major:
            return static_cast<std::uint64_t>(c[2]) << (2 * b) | static_cast<std::uint64_t>(c[1]) << b | c[0];
        default:
            return neighbor_search::morton(c[0], c[1], c[2]);
    }
}

// Stable sorts of (key, id) pairs of up to `capacity` entries, with the buffers allocated once: equal keys keep their
// order. Used by: cell_order, order_tetrahedra, rcm (once per level).
template <class ExecSpace>
class PairSort {
public:
    using Memory = typename ExecSpace::memory_space;
    using Keys = Kokkos::View<std::uint64_t*, Memory>;
    using Ids = Kokkos::View<int*, Memory>;

    PairSort(const ExecSpace& exec, int capacity) : exec_(exec) {
#if defined(KOKKOS_ENABLE_CUDA)
        if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
            keys_ = Keys(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("order_keys")), capacity);
            ids_ = Ids(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("order_ids")), capacity);
            cub::DoubleBuffer<std::uint64_t> k(nullptr, nullptr);
            cub::DoubleBuffer<int> v(nullptr, nullptr);
            std::size_t bytes = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, bytes, k, v, capacity, 0, 64, exec.cuda_stream());
            scratch_ = Kokkos::View<char*, Memory>(
                Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("order_sort")), bytes);
        }
#else
        (void)capacity;
#endif
    }

    // Sorts ids by the low `bits` bits of keys (equal extents, at most the capacity); both hold the result.
    void operator()(const Keys& keys, const Ids& ids, int bits) const {
        const int n = static_cast<int>(keys.extent(0));
        if (n < 2) return;
#if defined(KOKKOS_ENABLE_CUDA)
        if constexpr (std::is_same_v<ExecSpace, Kokkos::Cuda>) {
            cub::DoubleBuffer<std::uint64_t> k(keys.data(), keys_.data());
            cub::DoubleBuffer<int> v(ids.data(), ids_.data());
            std::size_t bytes = scratch_.extent(0);
            cub::DeviceRadixSort::SortPairs(scratch_.data(), bytes, k, v, n, 0, bits, exec_.cuda_stream());
            if (k.selector) {
                Kokkos::deep_copy(exec_, keys, Kokkos::subview(keys_, std::make_pair(0, n)));
                Kokkos::deep_copy(exec_, ids, Kokkos::subview(ids_, std::make_pair(0, n)));
            }
            return;
        }
#endif
        if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, Memory>::accessible) {
            exec_.fence();
            const std::uint64_t mask = bits >= 64 ? ~0ull : (1ull << bits) - 1;
            std::vector<int> at(n);
            std::iota(at.begin(), at.end(), 0);
            std::stable_sort(at.begin(), at.end(), [&](int a, int b) { return (keys(a) & mask) < (keys(b) & mask); });
            std::vector<std::uint64_t> k(n);
            std::vector<int> v(n);
            for (int q = 0; q < n; ++q) k[q] = keys(at[q]), v[q] = ids(at[q]);
            for (int q = 0; q < n; ++q) keys(q) = k[q], ids(q) = v[q];
        } else {
            throw std::runtime_error("mesh_order::PairSort: no sort for this backend");
        }
    }

private:
    ExecSpace exec_;
    Keys keys_;
    Ids ids_;
    Kokkos::View<char*, Memory> scratch_;
};

// order(i) = the cell that becomes cell i: cells sorted by `curve` (not rcm) of their snapped points, ties by id.
// Used by: voronoi::Mesh<3>::resort and rebuild.
template <class ExecSpace, class P>
void cell_order(const ExecSpace& exec, const P& point, int n, Curve curve,
                const Kokkos::View<int*, typename ExecSpace::memory_space>& order) {
    using Memory = typename ExecSpace::memory_space;
    Kokkos::View<std::uint64_t*, Memory> keys(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("order_keys")), n);
    const MooreTable moore = curve == Curve::moore ? MooreTable::build(axis_bits - 1) : MooreTable{};
    // Snapped coordinates are below 2^30 (the longest side): their top axis_bits bits.
    constexpr int drop = 30 - axis_bits;
    Kokkos::parallel_for(
        "mesh_order::keys", Kokkos::RangePolicy<ExecSpace>(exec, 0, n), KOKKOS_LAMBDA(int i) {
            const std::uint32_t c[3] = {static_cast<std::uint32_t>(point(i, 0)) >> drop,
                                        static_cast<std::uint32_t>(point(i, 1)) >> drop,
                                        static_cast<std::uint32_t>(point(i, 2)) >> drop};
            keys(i) = key(curve, c, moore);
            order(i) = i;
        });
    PairSort<ExecSpace>(exec, n)(keys, order, 3 * axis_bits);
}

// rank(order(i)) = i. Used by: voronoi::Mesh<3>::resort.
template <class ExecSpace>
void invert(const ExecSpace& exec, const Kokkos::View<const int*, typename ExecSpace::memory_space>& order,
            const Kokkos::View<int*, typename ExecSpace::memory_space>& rank) {
    Kokkos::parallel_for(
        "mesh_order::invert", Kokkos::RangePolicy<ExecSpace>(exec, 0, order.extent(0)),
        KOKKOS_LAMBDA(int i) { rank(order(i)) = i; });
}

// out(i) = in(order(i)), row by row (rank 1 or 2). Used by: voronoi::Mesh<3>::resort and permute.
template <class ExecSpace, class Order, class In, class Out>
void gather(const ExecSpace& exec, const Order& order, const In& in, const Out& out) {
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const int n = static_cast<int>(order.extent(0));
    if constexpr (In::rank() == 1) {
        Kokkos::parallel_for("mesh_order::gather", Range(exec, 0, n), KOKKOS_LAMBDA(int i) { out(i) = in(order(i)); });
    } else {
        Kokkos::parallel_for(
            "mesh_order::gather", Range(exec, 0, n), KOKKOS_LAMBDA(int i) {
                for (std::size_t c = 0; c < in.extent(1); ++c) out(i, c) = in(order(i), c);
            });
    }
}

// The share of faces whose cells are more than 64 ids apart. Used by: voronoi::Mesh<3>::geometry (Layout::far).
template <class ExecSpace, class Ints>
double far_share(const ExecSpace& exec, const Ints& owner, const Ints& neighbour) {
    int far = 0;
    const int faces = static_cast<int>(owner.extent(0));
    Kokkos::parallel_reduce(
        "mesh_order::far_faces", Kokkos::RangePolicy<ExecSpace>(exec, 0, faces),
        KOKKOS_LAMBDA(int f, int& sum) { sum += neighbour(f) - owner(f) > 64; }, far);
    return double(far) / std::max(1, faces);
}

// The Voronoi neighbours of every cell in CSR (start, adjacent), from the edges the live tetrahedra represent.
// Used by: rcm.
template <class ExecSpace>
void neighbours(const ExecSpace& exec, const tet_mesh::Mesh<typename ExecSpace::memory_space>& mesh,
                const Kokkos::View<const char*, typename ExecSpace::memory_space>& alive, int count, int n,
                Kokkos::View<int*, typename ExecSpace::memory_space>& start,
                Kokkos::View<int*, typename ExecSpace::memory_space>& adjacent) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    start = Kokkos::View<int*, Memory>("order_neighbour_start", n + 1);
    Kokkos::View<int*, Memory> cursor(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("order_cursor")), n);
    const auto vertex = mesh.vertex;
    const auto edges = mesh.edges;
    Kokkos::parallel_for(
        "mesh_order::degrees", Range(exec, 0, count), KOKKOS_LAMBDA(int t) {
            if (!alive(t)) return;
            for (unsigned bits = edges(t); bits; bits &= bits - 1) {
                int ka, kb;
                tet_mesh::edge_ends(Kokkos::Experimental::countr_zero_builtin(bits), ka, kb);
                Kokkos::atomic_inc(&start(vertex(t, ka)));
                Kokkos::atomic_inc(&start(vertex(t, kb)));
            }
        });
    int total = 0;
    Kokkos::parallel_scan(
        "mesh_order::neighbour_start", Range(exec, 0, n + 1),
        KOKKOS_LAMBDA(int i, int& sum, bool final) {
            const int degree = i < n ? start(i) : 0;
            if (final) {
                start(i) = sum;
                if (i < n) cursor(i) = sum;
            }
            sum += degree;
        },
        total);
    adjacent = Kokkos::View<int*, Memory>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("order_neighbours")), total);
    const auto adj = adjacent;
    Kokkos::parallel_for(
        "mesh_order::neighbour_fill", Range(exec, 0, count), KOKKOS_LAMBDA(int t) {
            if (!alive(t)) return;
            for (unsigned bits = edges(t); bits; bits &= bits - 1) {
                int ka, kb;
                tet_mesh::edge_ends(Kokkos::Experimental::countr_zero_builtin(bits), ka, kb);
                const int a = vertex(t, ka), b = vertex(t, kb);
                adj(Kokkos::atomic_fetch_inc(&cursor(a))) = b;
                adj(Kokkos::atomic_fetch_inc(&cursor(b))) = a;
            }
        });
}

// Reverse Cuthill-McKee on the Voronoi graph, level by level on the device: each level's cells sorted by the place of
// their first-placed neighbour in the level before, then by degree, then by id; the start is the lowest-degree cell of
// the last level of a search from cell 0 (one George-Liu step). order(i) = the cell that becomes cell i.
// Used by: voronoi::Mesh<3>::resort, for Curve::rcm.
template <class ExecSpace>
void rcm(const ExecSpace& exec, const tet_mesh::Mesh<typename ExecSpace::memory_space>& mesh,
         const Kokkos::View<const char*, typename ExecSpace::memory_space>& alive, int count, int n,
         const Kokkos::View<int*, typename ExecSpace::memory_space>& order) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    Kokkos::View<int*, Memory> start, adjacent;
    neighbours(exec, mesh, alive, count, n, start, adjacent);
    Kokkos::View<int*, Memory> place(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("order_place")), n);
    Kokkos::View<int*, Memory> parent(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("order_parent")), n);
    Kokkos::View<int*, Memory> cm(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("order_cm")), n);
    Kokkos::View<std::uint64_t*, Memory> keys(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("order_keys")), n);
    Kokkos::View<int, Memory> found("order_found");
    const auto host_found = Kokkos::create_mirror_view(found);
    const PairSort<ExecSpace> sort(exec, n);
    // One search from `first`: cm holds the cells in order, levels sorted if `sorted`; returns where the last level starts.
    auto search = [&](int first, bool sorted) {
        Kokkos::deep_copy(exec, place, -1);
        Kokkos::deep_copy(exec, parent, Kokkos::Experimental::finite_max_v<int>);
        Kokkos::parallel_for(
            "mesh_order::rcm_first", Range(exec, 0, 1), KOKKOS_LAMBDA(int) {
                cm(0) = first;
                place(first) = 0;
            });
        int begin = 0, end = 1, last = 0;
        while (end < n) {
            Kokkos::deep_copy(exec, found, 0);
            const int at = end;
            Kokkos::parallel_for(
                "mesh_order::rcm_level", Range(exec, begin, end), KOKKOS_LAMBDA(int p) {
                    const int v = cm(p);
                    for (int k = start(v); k < start(v + 1); ++k) {
                        const int u = adjacent(k);
                        if (place(u) >= 0 && place(u) < at) continue;
                        Kokkos::atomic_min(&parent(u), p);
                        if (Kokkos::atomic_compare_exchange(&place(u), -1, at) == -1)
                            cm(at + Kokkos::atomic_fetch_inc(&found())) = u;
                    }
                });
            Kokkos::deep_copy(exec, host_found, found);
            exec.fence();
            int fresh = host_found();
            if (fresh == 0) {  // another component: its lowest unplaced cell
                int lowest = n;
                Kokkos::parallel_reduce(
                    "mesh_order::rcm_unplaced", Range(exec, 0, n),
                    KOKKOS_LAMBDA(int i, int& low) {
                        if (place(i) < 0 && i < low) low = i;
                    },
                    Kokkos::Min<int>(lowest));
                Kokkos::parallel_for(
                    "mesh_order::rcm_restart", Range(exec, 0, 1), KOKKOS_LAMBDA(int) {
                        cm(at) = lowest;
                        place(lowest) = at;
                        parent(lowest) = -1;
                    });
                fresh = 1;
            }
            if (sorted && fresh > 1) {
                const auto level = std::make_pair(at, at + fresh);
                const Kokkos::View<int*, Memory> ids = Kokkos::subview(cm, level);
                const Kokkos::View<std::uint64_t*, Memory> k = Kokkos::subview(keys, level);
                Kokkos::parallel_for(
                    "mesh_order::rcm_ids", Range(exec, 0, fresh), KOKKOS_LAMBDA(int q) { k(q) = ids(q); });
                sort(k, ids, 32);
                Kokkos::parallel_for(
                    "mesh_order::rcm_keys", Range(exec, 0, fresh), KOKKOS_LAMBDA(int q) {
                        const int u = ids(q);
                        const int degree = Kokkos::min(start(u + 1) - start(u), 63);
                        k(q) = static_cast<std::uint64_t>(parent(u) + 1) << 6 | static_cast<std::uint64_t>(degree);
                    });
                sort(k, ids, 38);
            }
            last = at;
            begin = at;
            end = at + fresh;
        }
        return last;
    };
    // The search from cell 0 for the start, then the ordered search from it.
    const int last = search(0, false);
    int first = 0;
    {
        int best = Kokkos::Experimental::finite_max_v<int>;
        Kokkos::parallel_reduce(
            "mesh_order::rcm_start", Range(exec, last, n),
            KOKKOS_LAMBDA(int p, int& low) {
                const int u = cm(p);
                const int key = Kokkos::min(start(u + 1) - start(u), 127) << 24 | (u & 0xffffff);
                if (key < low) low = key;
            },
            Kokkos::Min<int>(best));
        Kokkos::View<int, Memory> chosen("order_first");
        Kokkos::parallel_for(
            "mesh_order::rcm_pick", Range(exec, last, n), KOKKOS_LAMBDA(int p) {
                const int u = cm(p);
                if ((Kokkos::min(start(u + 1) - start(u), 127) << 24 | (u & 0xffffff)) == best)
                    Kokkos::atomic_max(&chosen(), u);
            });
        Kokkos::deep_copy(exec, host_found, chosen);
        exec.fence();
        first = host_found();
    }
    search(first, true);
    Kokkos::parallel_for(
        "mesh_order::rcm_reverse", Range(exec, 0, n), KOKKOS_LAMBDA(int i) { order(i) = cm(n - 1 - i); });
}

// Renumbers the cells (rank(old) = new) of the tetrahedra and sorts the live ones by their lowest new cell, ties by
// slot; the slots past them are free, and every tetrahedron counts as created (state.touched_entries = -1).
// Used by: voronoi::Mesh<3>::resort.
template <class ExecSpace>
void order_tetrahedra(const ExecSpace& exec, const Kokkos::View<const int*, typename ExecSpace::memory_space>& rank,
                      tet_mesh::Mesh<typename ExecSpace::memory_space>& mesh,
                      tet_repair::State<typename ExecSpace::memory_space>& state) {
    using Memory = typename ExecSpace::memory_space;
    using Range = Kokkos::RangePolicy<ExecSpace>;
    const int count = state.count, capacity = static_cast<int>(mesh.vertex.extent(0));
    Kokkos::View<std::uint64_t*, Memory> keys(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("order_keys")), count);
    Kokkos::View<int*, Memory> slot(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("order_slots")), count);
    const auto vertex = mesh.vertex;
    const auto alive = state.alive;
    int live = 0;
    Kokkos::parallel_reduce(
        "mesh_order::tet_keys", Range(exec, 0, count),
        KOKKOS_LAMBDA(int t, int& sum) {
            slot(t) = t;
            if (!alive(t)) {
                keys(t) = 0xffffffffull;
                return;
            }
            ++sum;
            int low = rank(vertex(t, 0));
            for (int c = 1; c < 4; ++c) low = Kokkos::min(low, rank(vertex(t, c)));
            keys(t) = static_cast<std::uint64_t>(low);
        },
        live);
    PairSort<ExecSpace>(exec, count)(keys, slot, 32);
    // new_slot(old) for the adjacency, in the keys' room.
    const Kokkos::View<int*, Memory> new_slot(reinterpret_cast<int*>(keys.data()), count);
    Kokkos::parallel_for(
        "mesh_order::tet_rank", Range(exec, 0, count), KOKKOS_LAMBDA(int s) { new_slot(slot(s)) = s; });
    tet_mesh::Quads<Memory> v2(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("tet_vertex")), capacity);
    tet_mesh::Quads<Memory> a2(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string("tet_adjacent")), capacity);
    Kokkos::View<std::uint8_t*, Memory> e2("tet_edges", capacity);
    const auto adjacent = mesh.adjacent;
    const auto edges = mesh.edges;
    Kokkos::parallel_for(
        "mesh_order::tet_move", Range(exec, 0, live), KOKKOS_LAMBDA(int s) {
            const int t = slot(s);
            for (int c = 0; c < 4; ++c) {
                v2(s, c) = rank(vertex(t, c));
                const int u = adjacent(t, c);
                a2(s, c) = u < 0 ? -1 : new_slot(u);
            }
            e2(s) = edges(t);
        });
    mesh.vertex = v2;
    mesh.adjacent = a2;
    mesh.edges = e2;
    Kokkos::deep_copy(exec, Kokkos::subview(state.alive, std::make_pair(0, live)), char(1));
    Kokkos::deep_copy(exec, Kokkos::subview(state.alive, std::make_pair(live, capacity)), char(0));
    Kokkos::deep_copy(exec, state.touched, char(0));
    Kokkos::deep_copy(exec, state.dirty, 0);
    state.count = live;
    state.free_count = 0;
    state.touched_entries = -1;
    exec.fence();  // the old arrays go now
}

}  // namespace mesh_order
