#!/usr/bin/env python3
"""paper/data/inputs.npz from the 64³ point files of the benchmark inputs (library_bench points / lloyd: the generators
and seed of src/test/support.hpp): per input a slab one mean spacing thick projected to x-y (for clustered and halo
through the densest cell, shifted periodically so it sits in the middle), the nearest-neighbour distance histogram in
mean spacings, and the density contrast (99.9 / 0.1 percentile of 1/r_32³). Usage: extract_inputs.py [points dir]."""
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
    out, side = {"bins": BINS}, round(N ** (1 / 3))
    for kind in KINDS:
        x = read(os.path.join(src, f"{kind}_{N}.pts"))
        d = cKDTree(x, boxsize=1.0).query(x, k=33)[0] * side
        rho = 1 / d[:, 32] ** 3
        out[f"{kind}_contrast"] = np.percentile(rho, 99.9) / np.percentile(rho, 0.1)
        out[f"{kind}_nn_hist"] = np.histogram(d[:, 1], bins=BINS)[0]
        centre = np.array([0.5, 0.5, 0.5 + 0.5 / side])
        if kind in ("clustered", "halo"):  # the densest of 16³ cells, moved to the middle of the box
            h, _ = np.histogramdd(x, bins=16, range=[(0, 1)] * 3)
            cell = np.array(np.unravel_index(np.argmax(h), h.shape))
            inside = np.all(np.floor(x * 16).astype(int) == cell, axis=1)
            centre = np.median(x[inside], axis=0)  # the core of that blob or halo
        y = (x - centre + 0.5) % 1.0
        out[f"{kind}_slab"] = y[np.abs(y[:, 2] - 0.5) < 0.5 / side][:, :2].astype(np.float32)
        print(f"{kind}: {len(out[kind + '_slab'])} slab points, contrast {out[kind + '_contrast']:.3g}")
    np.savez_compressed(os.path.join(DATA, "inputs.npz"), **out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/ptmp/uli/voronoi-paper/points")
