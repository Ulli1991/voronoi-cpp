#pragma once

// "+ faces (ours)" for the libraries that return Voronoi neighbour lists (Paragram, votess): the full cell of a seed
// from its listed neighbours, in double, on the host or in a CUDA kernel. A cube of half side 0.5 around the seed is
// cut by the bisector of each listed neighbour; the cell is kept as dual triangles (each primal vertex the meeting of
// three planes, as Ray et al. 2018), and at the end each primal face y (a listed neighbour's bisector) is integrated
// from its directed edges (t -> o): the triangle (f_y, v_t, v_o) with f_y the plane's foot point (half the offset to
// the neighbour) adds 1/2 (v_t - f_y) x (v_o - f_y) to the face's area vector, and the tetrahedron (seed, f_y, v_t,
// v_o) its volume and moment. Exact if and only if the list holds every Voronoi neighbour (else the cell is too big)
// and the cell closes inside the cube (else it is reported unbounded). Shared by paragram.py (CUDA, through
// load_inline) and votess.cpp (OpenMP).

#include <cmath>

#if defined(__CUDACC__)
#define CLIP_HD __host__ __device__
#else
#define CLIP_HD
#endif

namespace clip {

constexpr int max_planes = 128;    // 6 cube planes + cutting neighbours
constexpr int max_vertices = 256;  // dual triangles

CLIP_HD inline void vertex3(const double (*n)[3], const double* nn, int a, int b, int c, double* v) {
    const double *A = n[a], *B = n[b], *C = n[c];
    const double bc[3] = {B[1] * C[2] - B[2] * C[1], B[2] * C[0] - B[0] * C[2], B[0] * C[1] - B[1] * C[0]};
    const double ca[3] = {C[1] * A[2] - C[2] * A[1], C[2] * A[0] - C[0] * A[2], C[0] * A[1] - C[1] * A[0]};
    const double ab[3] = {A[1] * B[2] - A[2] * B[1], A[2] * B[0] - A[0] * B[2], A[0] * B[1] - A[1] * B[0]};
    const double det = A[0] * bc[0] + A[1] * bc[1] + A[2] * bc[2];
    for (int k = 0; k < 3; ++k) v[k] = (nn[a] * bc[k] + nn[b] * ca[k] + nn[c] * ab[k]) / det;
}

// The dual triangle with directed edge y -> x, or -1.
CLIP_HD inline int across(const unsigned char* ta, const unsigned char* tb, const unsigned char* tc, int nt, int y,
                          int x) {
    for (int t = 0; t < nt; ++t)
        if ((ta[t] == y && tb[t] == x) || (tb[t] == y && tc[t] == x) || (tc[t] == y && ta[t] == x)) return t;
    return -1;
}

struct Result {
    double volume = 0, centroid[3] = {0, 0, 0}, rmax = 0;  // centroid relative to the seed
    int faces = 0;
    bool ok = false;  // false: unbounded by the list, or over the caps
};

// The cell of a seed with m listed neighbours; offset(j, d) writes neighbour j's position minus the seed's;
// face(j, a) receives each face: the list index j of its neighbour and its outward area vector a[3].
template <class Offset, class Face>
CLIP_HD inline Result cell(int m, const Offset& offset, const Face& face) {
    double pn[max_planes][3], nn[max_planes];
    int source[max_planes];
    unsigned char ta[max_vertices], tb[max_vertices], tc[max_vertices];
    double vx[max_vertices], vy[max_vertices], vz[max_vertices];
    Result out;
    const double r0 = 0.5;
    for (int k = 0; k < 6; ++k) {
        pn[k][0] = pn[k][1] = pn[k][2] = 0;
        pn[k][k / 2] = (k % 2 == 0) ? r0 : -r0;
        nn[k] = r0 * r0;
        source[k] = -1;
    }
    int np = 6, nt = 0;
    for (int c = 0; c < 8; ++c) {
        const int sx = c & 1, sy = (c >> 1) & 1, sz = (c >> 2) & 1;
        const bool even = ((sx + sy + sz) % 2) == 0;
        ta[nt] = static_cast<unsigned char>(sx);
        tb[nt] = static_cast<unsigned char>(even ? 2 + sy : 4 + sz);
        tc[nt] = static_cast<unsigned char>(even ? 4 + sz : 2 + sy);
        double v[3];
        vertex3(pn, nn, ta[nt], tb[nt], tc[nt], v);
        vx[nt] = v[0], vy[nt] = v[1], vz[nt] = v[2];
        ++nt;
    }
    for (int j = 0; j < m; ++j) {
        if (np >= max_planes) return out;
        double d[3];
        offset(j, d);
        const double fx = 0.5 * d[0], fy = 0.5 * d[1], fz = 0.5 * d[2];
        const double f2 = fx * fx + fy * fy + fz * fz;
        if (f2 == 0) continue;
        bool kill[max_vertices], any = false;
        for (int t = 0; t < nt; ++t) any |= (kill[t] = fx * vx[t] + fy * vy[t] + fz * vz[t] > f2);
        if (!any) continue;
        pn[np][0] = fx, pn[np][1] = fy, pn[np][2] = fz, nn[np] = f2;
        source[np] = j;
        unsigned char na[max_vertices], nb[max_vertices];
        int added = 0;
        for (int t = 0; t < nt; ++t) {
            if (!kill[t]) continue;
            const int vtx[3] = {ta[t], tb[t], tc[t]};
            for (int k = 0; k < 3; ++k) {
                const int x = vtx[k], y = vtx[(k + 1) % 3];
                const int o = across(ta, tb, tc, nt, y, x);
                if (o >= 0 && !kill[o]) {
                    if (added >= max_vertices) return out;
                    na[added] = static_cast<unsigned char>(x), nb[added] = static_cast<unsigned char>(y), ++added;
                }
            }
        }
        int w = 0;
        for (int t = 0; t < nt; ++t)
            if (!kill[t]) ta[w] = ta[t], tb[w] = tb[t], tc[w] = tc[t], vx[w] = vx[t], vy[w] = vy[t], vz[w] = vz[t], ++w;
        if (w + added > max_vertices) return out;
        nt = w;
        for (int k = 0; k < added; ++k) {
            ta[nt] = na[k], tb[nt] = nb[k], tc[nt] = static_cast<unsigned char>(np);
            double v[3];
            vertex3(pn, nn, na[k], nb[k], np, v);
            vx[nt] = v[0], vy[nt] = v[1], vz[nt] = v[2];
            ++nt;
        }
        ++np;
    }
    // Faces, volume and moment from the directed edges of each face.
    double area[max_planes][3];
    for (int y = 0; y < np; ++y) area[y][0] = area[y][1] = area[y][2] = 0;
    double six = 0, mx = 0, my = 0, mz = 0, r2 = 0;
    for (int t = 0; t < nt; ++t) {
        if (ta[t] < 6 || tb[t] < 6 || tc[t] < 6) return out;  // a cube face survived: unbounded by the list
        r2 = fmax(r2, vx[t] * vx[t] + vy[t] * vy[t] + vz[t] * vz[t]);
        const int vtx[3] = {ta[t], tb[t], tc[t]};
        for (int k = 0; k < 3; ++k) {
            const int x = vtx[k], y = vtx[(k + 1) % 3];
            const int o = across(ta, tb, tc, nt, y, x);
            if (o < 0) return out;
            // Edge (t -> o) of face y: triangle (f_y, v_t, v_o), tetrahedron (seed, f_y, v_t, v_o).
            const double* f = pn[y];
            const double a[3] = {vx[t] - f[0], vy[t] - f[1], vz[t] - f[2]};
            const double b[3] = {vx[o] - f[0], vy[o] - f[1], vz[o] - f[2]};
            area[y][0] += 0.5 * (a[1] * b[2] - a[2] * b[1]);
            area[y][1] += 0.5 * (a[2] * b[0] - a[0] * b[2]);
            area[y][2] += 0.5 * (a[0] * b[1] - a[1] * b[0]);
            const double v = (vy[t] * vz[o] - vz[t] * vy[o]) * f[0] + (vz[t] * vx[o] - vx[t] * vz[o]) * f[1] +
                             (vx[t] * vy[o] - vy[t] * vx[o]) * f[2];  // 6 x the tetrahedron's signed volume
            six += v;
            mx += v * (f[0] + vx[t] + vx[o]);
            my += v * (f[1] + vy[t] + vy[o]);
            mz += v * (f[2] + vz[t] + vz[o]);
        }
    }
    // Each directed edge appears once per face; the orientation of the dual triangles fixes one sign for all.
    const double sign = six < 0 ? -1.0 : 1.0;
    out.volume = sign * six / 6.0;
    out.centroid[0] = mx / (4.0 * six);
    out.centroid[1] = my / (4.0 * six);
    out.centroid[2] = mz / (4.0 * six);
    out.rmax = sqrt(r2);
    for (int y = 6; y < np; ++y) {
        const double a[3] = {sign * area[y][0], sign * area[y][1], sign * area[y][2]};
        if (a[0] == 0 && a[1] == 0 && a[2] == 0) continue;
        face(source[y], a);
        ++out.faces;
    }
    out.ok = true;
    return out;
}

}  // namespace clip
