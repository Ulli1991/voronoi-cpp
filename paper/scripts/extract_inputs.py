#!/usr/bin/env python3
"""paper/data/inputs.npz: for each benchmark input at 64³ (point files of library_bench points / lloyd, the generators
and seed of src/test/support.hpp), a slab one mean spacing thick projected to x-y, the histogram of nearest-neighbour
distances in mean spacings (periodic), and the density contrast (99.9 / 0.1 percentile of 1/r_32³). Usage: extract_inputs.py [points dir] (default /ptmp/uli/voronoi-paper/points)."""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

from paperfig import DATA

KINDS = ["jittered", "bluenoise", "uniform", "clustered", "halo", "lattice", "wave"]
N = 262144
BINS = np.logspace(-3, 1, 161)  # nearest-neighbour distance in mean spacings


def read(path):
    head = np.fromfile(path, dtype=np.uint64, count=2)
    return np.fromfile(path, dtype=np.float64, offset=16, count=int(head[0]) * 3).reshape(-1, 3)


def main(src):
    out, side = {}, round(N ** (1 / 3))
    for kind in KINDS:
        x = read(os.path.join(src, f"{kind}_{N}.pts"))
        d = cKDTree(x, boxsize=1.0).query(x, k=33)[0] * side
        nn, r32 = d[:, 1], d[:, 32]
        rho = 1 / r32 ** 3  # local density from the 32nd neighbour, for the density contrast
        out[f"{kind}_contrast"] = np.percentile(rho, 99.9) / np.percentile(rho, 0.1)
        counts = np.histogram(x[:, 2], bins=side, range=(0, 1))[0]
        z0 = (np.argmax(counts) if kind in ("clustered", "halo") else side // 2) / side
        slab = x[(x[:, 2] >= z0) & (x[:, 2] < z0 + 1 / side)][:, :2]
        out[f"{kind}_slab"], out[f"{kind}_z0"] = slab.astype(np.float32), z0
        out[f"{kind}_nn_hist"] = np.histogram(nn, bins=BINS)[0]
        print(f"{kind}: slab at z = {z0:.4f}, {len(slab)} points; nn p1 {np.percentile(nn, 1):.3g}, "
              f"median {np.median(nn):.3g}, p99 {np.percentile(nn, 99):.3g}; density contrast {out[kind + '_contrast']:.3g}")
    out["bins"] = BINS
    np.savez_compressed(os.path.join(DATA, "inputs.npz"), **out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/ptmp/uli/voronoi-paper/points")
