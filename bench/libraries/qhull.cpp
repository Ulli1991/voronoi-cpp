// Qhull 2020.2 (Barber, Dobkin, Huhdanpaa 1996, ACM TOMS 22(4) 469) in the library benchmark
// (docs/library_benchmark.md): the 3D Delaunay triangulation (lower hull of the points lifted to the paraboloid) of
// the points padded with periodic images (Qhull has no periodic box), through the reentrant library:
// qh_new_qhull("qhull d Qbb Qz Qt"): d Delaunay, Qbb scale the last (lifted) coordinate as qdelaunay does, Qz add a
// point at infinity (for cospherical inputs), Qt triangulated output (simplicial facets). Then the Voronoi volumes of
// the originals from the lower-hull facets (images_dual.hpp). Single thread (Qhull is serial).
//   qhull_bench <points file> --variant=delaunay|dual --width=<mean spacings> [--repeats=3] [--volumes=file]

#include "images_dual.hpp"

extern "C" {
#include <libqhull_r/qhull_ra.h>
}

namespace {

struct Qhull {
    qhT qh_storage;
    qhT* qh = &qh_storage;
    bool built = false;
    void build(const images::Padded& p) {
        QHULL_LIB_CHECK
        qh_zero(qh, stderr);
        char cmd[] = "qhull d Qbb Qz Qt";
        // Qhull reads the coordinates in place (ismalloc false: not freed by Qhull).
        const int code = qh_new_qhull(qh, 3, int(p.size()), const_cast<coordT*>(p.xyz.data()), False, cmd, nullptr,
                                      stderr);
        built = true;
        if (code) throw std::runtime_error("qh_new_qhull failed: " + std::to_string(code));
    }
    template <class F>
    void each_tet(F&& f) {
        facetT* facet;
        vertexT *vertex, **vertexp;
        FORALLfacets {
            if (facet->upperdelaunay) continue;
            std::uint64_t q[4];
            int k = 0;
            FOREACHvertex_(facet->vertices) {
                if (k < 4) q[k] = std::uint64_t(qh_pointid(qh, vertex->point));
                ++k;
            }
            if (k != 4) throw std::runtime_error("non-simplicial facet");
            f(q);
        }
    }
    ~Qhull() {
        if (!built) return;
        int curlong, totlong;
        qh_freeqhull(qh, !qh_ALL);
        qh_memfreeshort(qh, &curlong, &totlong);
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: qhull_bench <points file> --variant=delaunay|dual --width=<spacings>\n");
        return 1;
    }
    libbench::Setup s;
    s.library = "Qhull";
    s.variant = "qh_new_qhull(d Qbb Qz Qt) of the padded points";
    s.version = "2020.2 (8.0.2, github.com/qhull/qhull tag v8.0.2, 613debe)";
    s.device = "2x Xeon Platinum 8360Y";
    s.threads = 1;
    s.precision = "double floating point, facet merging (Qt triangulates merged facets); dual volumes double";
    return images::drive<Qhull>(argc, argv, s);
}
