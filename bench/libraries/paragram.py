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
  - variant "adjacency + faces (ours)": the same plus the full cells of the n originals computed in double from
    Paragram's neighbour lists (our kernel below, clip_cells.hpp: clip a cube by the bisector of each listed
    neighbour; volumes, centroids, faces with neighbours and areas; it uses the original double positions, so a cell
    is exact iff the neighbour list is complete). Its volumes go to the volumes file (oracle_check); a cell whose list
    leaves it unbounded or overflows our caps is NaN.
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

#include "clip_cells.hpp"

// The full cell of each original from Paragram's neighbour lists (clip_cells.hpp, double): volume, centroid
// (absolute), largest vertex distance, and its faces (neighbour's original label, outward area vector) appended to a
// pool at an atomically claimed offset (face_start, face_count per cell; `used` the pool's fill, may exceed `cap`:
// then nothing past cap is written and the caller reruns with a larger pool).
__global__ void cells_kernel(const double* __restrict__ p, const int* __restrict__ adj, const int* __restrict__ off,
                             const long long* __restrict__ label, int n, double* vol, double* cen, double* rmax,
                             int* face_count, long long* face_start, long long* face_nb, double* face_area,
                             unsigned long long* used, long long cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long nb[clip::max_planes];
    double ar[clip::max_planes][3];
    int k = 0;
    const int b = off[i], m = off[i + 1] - b;
    const double* s = p + 3 * (long long)i;
    const clip::Result r = clip::cell(
        m,
        [&](int j, double* d) {
            const double* q = p + 3 * (long long)adj[b + j];
            d[0] = q[0] - s[0], d[1] = q[1] - s[1], d[2] = q[2] - s[2];
        },
        [&](int j, const double* a) {
            nb[k] = label[adj[b + j]];
            ar[k][0] = a[0], ar[k][1] = a[1], ar[k][2] = a[2];
            ++k;
        });
    vol[i] = r.ok ? r.volume : NAN;
    rmax[i] = r.rmax;
    for (int a = 0; a < 3; ++a) cen[3 * (long long)i + a] = s[a] + r.centroid[a];
    const long long start = (long long)atomicAdd(used, (unsigned long long)k);
    face_count[i] = k;
    face_start[i] = start;
    if (start + k > cap) return;
    for (int f = 0; f < k; ++f) {
        face_nb[start + f] = nb[f];
        for (int a = 0; a < 3; ++a) face_area[3 * (start + f) + a] = ar[f][a];
    }
}

// Volumes, centroids, rmax and the face pool; reruns once with the pool's exact size if it overflowed.
std::vector<torch::Tensor> cell_faces(torch::Tensor points, torch::Tensor adj, torch::Tensor off, torch::Tensor label,
                                      int64_t n) {
    auto dopt = torch::TensorOptions().dtype(torch::kFloat64).device(points.device());
    auto iopt = torch::TensorOptions().dtype(torch::kInt32).device(points.device());
    auto lopt = torch::TensorOptions().dtype(torch::kInt64).device(points.device());
    auto vol = torch::empty({n}, dopt), cen = torch::empty({n, 3}, dopt), rmax = torch::empty({n}, dopt);
    auto count = torch::empty({n}, iopt), start = torch::empty({n}, lopt);
    long long cap = 20 * n;
    for (int attempt = 0; attempt < 2; ++attempt) {
        auto nb = torch::empty({cap}, lopt), area = torch::empty({cap, 3}, dopt), used = torch::zeros({1}, lopt);
        const int block = 64;
        cells_kernel<<<(n + block - 1) / block, block>>>(
            points.data_ptr<double>(), adj.data_ptr<int>(), off.data_ptr<int>(), label.data_ptr<int64_t>(), (int)n,
            vol.data_ptr<double>(), cen.data_ptr<double>(), rmax.data_ptr<double>(), count.data_ptr<int>(),
            (long long*)start.data_ptr<int64_t>(), (long long*)nb.data_ptr<int64_t>(), area.data_ptr<double>(),
            (unsigned long long*)used.data_ptr<int64_t>(), cap);
        const long long filled = used.item<int64_t>();
        if (filled <= cap) return {vol, cen, rmax, count, start, nb, area, used};
        cap = filled;
    }
    throw std::runtime_error("face pool overflow");
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
std::vector<torch::Tensor> cell_faces(torch::Tensor points, torch::Tensor adj, torch::Tensor off, torch::Tensor label,
                                      int64_t n);
void peak_start();
double peak_stop();
"""


def ours():
    here = os.path.dirname(os.path.abspath(__file__))
    return load_inline(name="libbench_paragram_faces", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                       functions=["cell_faces", "peak_start", "peak_stop"], extra_include_paths=[here],
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
    """The n points, then their periodic images within w of the unit box, and the original of each (on the device)."""
    index = torch.arange(x.shape[0], device=x.device)
    parts, labels = [x], [index]
    for s in SHIFTS:
        y = x + torch.tensor(s, dtype=x.dtype, device=x.device)
        keep = ((y > -w) & (y < 1.0 + w)).all(dim=1)
        parts.append(y[keep])
        labels.append(index[keep])
    return torch.cat(parts), torch.cat(labels)


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
        p, label = padded(x, w)
        d = paragram.voronoi_diagram(p.float().contiguous())
        return p, label, d

    def with_faces():
        p, label, d = adjacency()
        out = ext.cell_faces(p, d.adjacency, d.offsets, label, n)
        return p, d, out

    # Warm-up (the lazy CUDA builds are loaded above), the cells, the status counts and the shell check.
    p, d, out = with_faces()
    vol, rmax = out[0], out[2]
    faces_per_cell = float(out[7][0]) / n
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
          f"max R {rmax_all:.6g} (2R <= w: {shell_ok}), max R of status-0 cells {rmax_ok:.6g} (2R <= w: {shell_ok0}), sum {vsum:.15g}, mean neighbours {degree:.3f}, faces per cell {faces_per_cell:.3f}",
          file=sys.stderr, flush=True)
    if a.volumes:
        vol.cpu().numpy().astype("<f8").tofile(a.volumes)
    del p, d, vol, rmax, out

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
        (dict(common, variant="voronoi_diagram + faces from its adjacency (ours)",
              outputs="Voronoi adjacency; cells (ours): volumes centroids faces areas neighbours",
              precision="float32 adjacency (library), cells in double from the double points (ours)",
              timed="double points on the device to padded set, adjacency and the full cells of the n originals on "
                    "the device",
              note="our cell kernel: clip a cube by the bisectors of the listed neighbours (clip_cells.hpp)"),
         with_faces),
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
