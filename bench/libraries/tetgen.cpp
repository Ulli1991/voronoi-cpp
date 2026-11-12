// TetGen 1.6.0 (Si 2015, ACM TOMS 41(2) 11) in the library benchmark (docs/library_benchmark.md): the Delaunay
// tetrahedralisation of the points padded with periodic images (TetGen has no periodic box), with
// tetrahedralize("Qz") (Q quiet, z indices from zero; no other switch: plain Delaunay of the point set, TetGen's exact
// predicates and its own Hilbert/BRIO insertion order), and the Voronoi volumes of the originals from the tetrahedra
// (images_dual.hpp; TetGen's -v Voronoi output has no volumes). Single thread (TetGen is serial).
//   tetgen_bench <points file> --variant=delaunay|dual --width=<mean spacings> [--repeats=3] [--volumes=file]

#include "images_dual.hpp"

#include <tetgen.h>

#include <cstring>

namespace {

struct TetGen {
    tetgenio in, out;
    void build(const images::Padded& p) {
        in.firstnumber = 0;
        in.numberofpoints = int(p.size());
        in.pointlist = new REAL[p.xyz.size()];  // freed by ~tetgenio
        std::memcpy(in.pointlist, p.xyz.data(), p.xyz.size() * sizeof(REAL));
        tetgenbehavior b;
        char switches[] = "Qz";
        b.parse_commandline(switches);
        tetrahedralize(&b, &in, &out);
    }
    template <class F>
    void each_tet(F&& f) const {
        const int c = out.numberofcorners;
        for (int t = 0; t < out.numberoftetrahedra; ++t) {
            const int* v = out.tetrahedronlist + std::size_t(c) * t;
            const std::uint64_t q[4] = {std::uint64_t(v[0]), std::uint64_t(v[1]), std::uint64_t(v[2]),
                                        std::uint64_t(v[3])};
            f(q);
        }
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: tetgen_bench <points file> --variant=delaunay|dual --width=<spacings>\n");
        return 1;
    }
    libbench::Setup s;
    s.library = "TetGen";
    s.variant = "tetrahedralize(Qz) of the padded points";
    s.version = "1.6.0 (github.com/TetGen/TetGen tag v1.6.0, 535f9c4)";
    s.device = "2x Xeon Platinum 8360Y";
    s.threads = 1;
    s.precision = "double, exact predicates (Shewchuk); dual volumes double floating point";
    return images::drive<TetGen>(argc, argv, s);
}
