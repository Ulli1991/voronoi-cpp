#!/usr/bin/env python3
"""Compares the cells of two voronoi_bench runs by particle (--dump-cells): neighbour sets exactly, geometry to rounding.

Usage: resort_bits.py <dump a> <dump b>   (the path given to --dump-cells, without the .geometry.npy / .neighbours.npy ends)"""
import sys

import numpy as np


def main(a, b):
    ga, gb = np.load(a + ".geometry.npy"), np.load(b + ".geometry.npy")
    na, nb = np.load(a + ".neighbours.npy"), np.load(b + ".neighbours.npy")
    volume = np.abs(ga[:, 0] - gb[:, 0]) / ga[:, 0]
    centroid = np.abs(ga[:, 1:] - gb[:, 1:]).max(axis=1) / np.cbrt(ga[:, 0])
    print("cells=%d neighbour_sets_differ=%d volume_bits_differ=%d volume_rel_max=%.3g centroid_rel_max=%.3g"
          % (len(na), int((na != nb).sum()), int((ga[:, 0] != gb[:, 0]).sum()), volume.max(), centroid.max()))


if __name__ == "__main__":
    main(*sys.argv[1:3])
