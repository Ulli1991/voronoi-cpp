#!/usr/bin/env python3
"""Paragram (Taveira, Lindstroem, Fatemi, Hammarstrand, Kahl, SIGGRAPH 2026, arXiv:2605.06408,
doi:10.1145/3799902.3811229) in the library benchmark, from Python (it is a PyTorch CUDA extension).

    paragram.py <points file> [--shell=<mean spacings>] [--repeats=3] [--volumes=file]

What is called: paragram.voronoi_diagram(points) (cuBQL BVH + per-cell convex-cell clipping in float32, then a
symmetrised adjacency CSR). It returns only the Voronoi adjacency (neighbour lists, CSR offsets, per-cell status):
no volumes, faces or areas; it accepts float32 points only and is not periodic. So:
  - periodic box: the n points plus their periodic images in a shell of width w around the unit box, built on the
    device from the double points inside the clock; volumes and records count the n original cells only;
  - variant "adjacency": the library's own output, timed from double points on the device to the adjacency CSR;
  - variant "adjacency + volumes (ours)": the same plus the volumes of the n original cells computed in double from
    Paragram's neighbour lists (our kernel below: clip a cube by the bisector of each listed neighbour, volume from
    the dual-triangle cell; it uses the original double positions, so a volume is exact iff the neighbour list is
    complete). Its volumes go to the volumes file (oracle_check); a cell whose list leaves it unbounded or overflows
    our caps is NaN.
The shell is checked afterwards: every neighbour of an original cell lies within 2 R of its seed (R its largest
vertex distance), so w >= 2 max R makes the original cells exact; the record says whether it held.
Timing: torch.cuda.synchronize() before and after, points already on the device. Peak device memory: total minus
free (cudaMemGetInfo) sampled every 200 us from a native thread, as libbench::DevicePeak, after emptying the torch
cache. Records: the JSON keys and order of libbench.hpp record().
"""

import argparse
import itertools
import math
import os
import socket
import sys
import time

import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

VERSION = "4f36b1f (0.1.0, main 2026-09-01)"

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <thread>

constexpr int MAXP = 128;  // planes per cell: 6 box planes + listed neighbours
constexpr int MAXT = 256;  // dual triangles (primal vertices)

__device__ inline void vertex3(const double (*n)[3], const double* nn, int a, int b, int c, double* v) {
    // Solve n_a.x = nn_a, n_b.x = nn_b, n_c.x = nn_c (planes in foot-point form).
    const double* A = n[a]; const double* B = n[b]; const double* C = n[c];
    const double bc[3] = {B[1]*C[2]-B[2]*C[1], B[2]*C[0]-B[0]*C[2], B[0]*C[1]-B[1]*C[0]};
    const double ca[3] = {C[1]*A[2]-C[2]*A[1], C[2]*A[0]-C[0]*A[2], C[0]*A[1]-C[1]*A[0]};
    const double ab[3] = {A[1]*B[2]-A[2]*B[1], A[2]*B[0]-A[0]*B[2], A[0]*B[1]-A[1]*B[0]};
    const double det = A[0]*bc[0] + A[1]*bc[1] + A[2]*bc[2];
    for (int k = 0; k < 3; ++k) v[k] = (nn[a]*bc[k] + nn[b]*ca[k] + nn[c]*ab[k]) / det;
}

__device__ inline int across(const unsigned char* ta, const unsigned char* tb, const unsigned char* tc, int nt,
                             int y, int x) {  // the triangle with directed edge y -> x
    for (int t = 0; t < nt; ++t)
        if ((ta[t] == y && tb[t] == x) || (tb[t] == y && tc[t] == x) || (tc[t] == y && ta[t] == x)) return t;
    return -1;
}

__global__ void volumes_kernel(const double* __restrict__ p, const int* __restrict__ adj,
                               const int* __restrict__ off, int n, double* vol, double* rmax) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double pn[MAXP][3], nn[MAXP];
    unsigned char ta[MAXT], tb[MAXT], tc[MAXT];
    double vx[MAXT], vy[MAXT], vz[MAXT];
    const double R0 = 0.5;  // the starting cube, relative to the seed
    for (int k = 0; k < 6; ++k) {
        pn[k][0] = pn[k][1] = pn[k][2] = 0;
        pn[k][k / 2] = (k % 2 == 0) ? R0 : -R0;
        nn[k] = R0 * R0;
    }
    int np = 6, nt = 0;
    for (int s = 0; s < 8; ++s) {  // the cube's corners, triangles ordered by det > 0 (outward, counter-clockwise)
        const int sx = s & 1, sy = (s >> 1) & 1, sz = (s >> 2) & 1;
        const int a = sx, b = 2 + sy, c = 4 + sz;
        const bool even = ((sx + sy + sz) % 2) == 0;
        ta[nt] = a; tb[nt] = even ? b : c; tc[nt] = even ? c : b;
        double v[3];
        vertex3(pn, nn, ta[nt], tb[nt], tc[nt], v);
        vx[nt] = v[0]; vy[nt] = v[1]; vz[nt] = v[2];
        ++nt;
    }
    bool bad = false;
    const double px = p[3 * i], py = p[3 * i + 1], pz = p[3 * i + 2];
    for (int e = off[i]; e < off[i + 1] && !bad; ++e) {
        const int j = adj[e];
        if (j == i) continue;
        if (np >= MAXP) { bad = true; break; }
        const double fx = 0.5 * (p[3 * j] - px), fy = 0.5 * (p[3 * j + 1] - py), fz = 0.5 * (p[3 * j + 2] - pz);
        const double f2 = fx * fx + fy * fy + fz * fz;
        bool kill[MAXT];
        bool any = false;
        for (int t = 0; t < nt; ++t) {
            kill[t] = fx * vx[t] + fy * vy[t] + fz * vz[t] > f2;
            any |= kill[t];
        }
        if (!any) continue;
        pn[np][0] = fx; pn[np][1] = fy; pn[np][2] = fz; nn[np] = f2;
        unsigned char na[MAXT], nb[MAXT];
        int nnew = 0;
        for (int t = 0; t < nt && !bad; ++t) {
            if (!kill[t]) continue;
            const int vtx[3] = {ta[t], tb[t], tc[t]};
            for (int k = 0; k < 3; ++k) {
                const int x = vtx[k], y = vtx[(k + 1) % 3];
                const int o = across(ta, tb, tc, nt, y, x);
                if (o >= 0 && !kill[o]) {
                    if (nnew >= MAXT) { bad = true; break; }
                    na[nnew] = x; nb[nnew] = y; ++nnew;
                }
            }
        }
        int w = 0;
        for (int t = 0; t < nt; ++t)
            if (!kill[t]) { ta[w] = ta[t]; tb[w] = tb[t]; tc[w] = tc[t]; vx[w] = vx[t]; vy[w] = vy[t]; vz[w] = vz[t]; ++w; }
        if (w + nnew > MAXT) { bad = true; break; }
        nt = w;
        for (int k = 0; k < nnew; ++k) {
            ta[nt] = na[k]; tb[nt] = nb[k]; tc[nt] = np;
            double v[3];
            vertex3(pn, nn, na[k], nb[k], np, v);
            vx[nt] = v[0]; vy[nt] = v[1]; vz[nt] = v[2];
            ++nt;
        }
        ++np;
    }
    // V = 1/3 sum_faces n_f . A_f, A_f = 1/2 sum over the face's edges v_t x v_next; the edge x -> y of triangle t
    // leads to the next vertex of face y.
    double V = 0, r2 = 0;
    for (int t = 0; t < nt && !bad; ++t) {
        if (ta[t] < 6 || tb[t] < 6 || tc[t] < 6) bad = true;  // a box plane survives: unbounded by the list
        r2 = fmax(r2, vx[t] * vx[t] + vy[t] * vy[t] + vz[t] * vz[t]);
        const int vtx[3] = {ta[t], tb[t], tc[t]};
        for (int k = 0; k < 3; ++k) {
            const int x = vtx[k], y = vtx[(k + 1) % 3];
            const int o = across(ta, tb, tc, nt, y, x);
            if (o < 0) { bad = true; break; }
            const double cx = vy[t] * vz[o] - vz[t] * vy[o], cy = vz[t] * vx[o] - vx[t] * vz[o],
                         cz = vx[t] * vy[o] - vy[t] * vx[o];
            V += cx * pn[y][0] + cy * pn[y][1] + cz * pn[y][2];
        }
    }
    vol[i] = bad ? NAN : fabs(V) / 6.0;
    rmax[i] = sqrt(r2);
}

std::vector<torch::Tensor> cell_volumes(torch::Tensor points, torch::Tensor adj, torch::Tensor off, int64_t n) {
    auto opt = torch::TensorOptions().dtype(torch::kFloat64).device(points.device());
    auto vol = torch::empty({n}, opt), rmax = torch::empty({n}, opt);
    const int block = 64;
    volumes_kernel<<<(n + block - 1) / block, block>>>(points.data_ptr<double>(), adj.data_ptr<int>(),
                                                       off.data_ptr<int>(), (int)n, vol.data_ptr<double>(),
                                                       rmax.data_ptr<double>());
    return {vol, rmax};
}

// Device memory sampler, as libbench::DevicePeak (a native thread: the Python call holds the GIL).
static std::atomic<bool> running{false};
static std::atomic<long long> peak{0};
static long long base = 0;
static std::thread sampler;
static long long used() { size_t f = 0, t = 0; cudaMemGetInfo(&f, &t); return (long long)(t - f); }
void peak_start() {
    cudaDeviceSynchronize();
    base = used(); peak = base; running = true;
    int dev = 0; cudaGetDevice(&dev);
    sampler = std::thread([dev] {
        cudaSetDevice(dev);
        while (running) {
            const long long u = used();
            if (u > peak) peak = u;
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    });
}
double peak_stop() {
    cudaDeviceSynchronize();
    const long long u = used();
    if (u > peak) peak = u;
    running = false; sampler.join();
    return double(peak - base);
}
"""
CPP_SRC = """
std::vector<torch::Tensor> cell_volumes(torch::Tensor points, torch::Tensor adj, torch::Tensor off, int64_t n);
void peak_start();
double peak_stop();
"""


def ours():
    return load_inline(name="libbench_paragram_ours", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                       functions=["cell_volumes", "peak_start", "peak_stop"],
                       extra_cuda_cflags=["-O3"], verbose=False)


def read_frames(path):
    with open(path, "rb") as f:
        n, frames = np.frombuffer(f.read(16), dtype="<u8")
        xyz = np.frombuffer(f.read(), dtype="<f8")
    return int(n), int(frames), xyz.reshape(int(frames), int(n), 3)


def g9(x):
    return format(x, ".9g")


def record(s, points_file, n, phase, repeat, seconds, peak, extra=""):
    """One JSON line with the keys and order of libbench::record()."""
    inp = os.path.basename(points_file).split("_")[0]
    print("{" + f"\"library\":\"{s['library']}\",\"variant\":\"{s['variant']}\",\"version\":\"{s['version']}\","
          f"\"device\":\"{s['device']}\",\"threads\":{s['threads']},\"outputs\":\"{s['outputs']}\","
          f"\"periodic\":\"{s['periodic']}\",\"precision\":\"{s['precision']}\",\"timed\":\"{s['timed']}\","
          f"\"note\":\"{s['note']}\",\"points\":\"{points_file}\",\"input\":\"{inp}\",\"n\":{n},"
          f"\"phase\":\"{phase}\",\"repeat\":{repeat},\"seconds\":{g9(seconds)},\"Mcells_s\":{g9(n / seconds / 1e6)},"
          f"\"peak_bytes\":{g9(peak)},\"bytes_per_cell\":{g9(peak / n)},\"job\":\"{os.environ.get('SLURM_JOB_ID', '')}\","
          f"\"host\":\"{socket.gethostname()}\"{extra}" + "}", flush=True)


SHIFTS = [s for s in itertools.product((-1.0, 0.0, 1.0), repeat=3) if s != (0.0, 0.0, 0.0)]


def padded(x, w):
    """The n points, then their periodic images within w of the unit box (on the device)."""
    parts = [x]
    for s in SHIFTS:
        y = x + torch.tensor(s, dtype=x.dtype, device=x.device)
        parts.append(y[((y > -w) & (y < 1.0 + w)).all(dim=1)])
    return torch.cat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points")
    ap.add_argument("--shell", type=float, default=4.0, help="shell width in mean spacings n^(-1/3)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--volumes", default="")
    a = ap.parse_args()

    # This driver is itself named paragram.py and Python puts the script's directory first on sys.path, so a plain
    # `import paragram` would import the driver. Drop that entry so the library (PYTHONPATH) is found.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [q for q in sys.path if os.path.abspath(q or ".") != here]
    sys.modules.pop("paragram", None)
    import paragram
    assert hasattr(paragram, "voronoi_diagram"), paragram.__file__

    ext = ours()
    n, _, xyz = read_frames(a.points)
    x = torch.from_numpy(np.array(xyz[0])).cuda()
    w = a.shell * n ** (-1.0 / 3.0)

    def adjacency():
        p = padded(x, w)
        d = paragram.voronoi_diagram(p.float().contiguous())
        return p, d

    def with_volumes():
        p, d = adjacency()
        vol, rmax = ext.cell_volumes(p, d.adjacency, d.offsets, n)
        return p, d, vol, rmax

    # Warm-up (the lazy CUDA builds are loaded above), the volumes, the status counts and the shell check.
    p, d, vol, rmax = with_volumes()
    torch.cuda.synchronize()
    m = p.shape[0] - n
    status = d.status[:n]
    failed = int((status != 0).sum())
    nan = int(torch.isnan(vol).sum())
    closed = torch.where(torch.isnan(vol), torch.zeros_like(rmax), rmax)
    rmax_all = float(closed.max())  # over the closed cells
    rmax_ok = float(torch.where(status == 0, closed, torch.zeros_like(closed)).max())  # and those with status 0
    shell_ok = 2.0 * rmax_all <= w
    shell_ok0 = 2.0 * rmax_ok <= w
    vsum = float(torch.nansum(vol))
    degree = float((d.offsets[1:n + 1] - d.offsets[:n]).double().mean())
    codes, counts = torch.unique(status, return_counts=True)
    print("# status codes of the n cells:", dict(zip(codes.tolist(), counts.tolist())), file=sys.stderr)
    print(f"# {a.points}: n {n}, images {m}, w {w:.6g}, failed status {failed}, NaN volumes {nan}, "
          f"max R {rmax_all:.6g} (2R <= w: {shell_ok}), max R of status-0 cells {rmax_ok:.6g} (2R <= w: {shell_ok0}), sum {vsum:.15g}, mean neighbours {degree:.3f}",
          file=sys.stderr, flush=True)
    if a.volumes:
        vol.cpu().numpy().astype("<f8").tofile(a.volumes)
    del p, d, vol, rmax

    periodic = (f"images: shell of width {g9(w)} ({g9(a.shell)} mean spacings), {m} extra points"
                f" ({g9(m / n)} n); 2 max R <= w: {'yes' if shell_ok else 'NO'} over all closed cells,"
                f" {'yes' if shell_ok0 else 'NO'} over the cells of status 0")
    common = dict(library="Paragram", version=VERSION, device="A100-SXM4-40GB", threads=1, periodic=periodic)
    extra = f",\"status_failed\":{failed},\"images\":{m},\"nan_volumes\":{nan}"
    variants = [
        (dict(common, variant="voronoi_diagram (adjacency)", outputs="Voronoi adjacency (neighbour lists)",
              precision="float32 (the library takes float32 only; -use_fast_math, its default build)",
              timed="double points on the device to padded float32 set and adjacency CSR on the device",
              note="torch.cuda.synchronize around the Python call; chunking automatic (none at these sizes)"),
         adjacency),
        (dict(common, variant="voronoi_diagram + volumes from its adjacency (ours)",
              outputs="Voronoi adjacency, volumes (ours)",
              precision="float32 adjacency (library), volumes in double from the double points (ours)",
              timed="double points on the device to padded set, adjacency and volumes of the n cells on the device",
              note="our volume kernel: clip a cube by the bisectors of the listed neighbours (dual triangles)"),
         with_volumes),
    ]
    for s, body in variants:
        start = time.perf_counter()
        torch.cuda.empty_cache()
        ext.peak_start()
        r = body()
        torch.cuda.synchronize()
        peak = ext.peak_stop()
        del r
        while time.perf_counter() - start < 2.0:  # warm-up as libbench::run_protocol: clocks ramp up
            r = body()
            torch.cuda.synchronize()
            del r
        for rep in range(1, a.repeats + 1):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = body()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            del r
            record(s, a.points, n, "build", rep, seconds, peak, extra)


if __name__ == "__main__":
    main()
