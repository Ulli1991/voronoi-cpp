#!/usr/bin/env python3
"""Render the face polygons emitted by the Kokkos 128-particle example."""

import csv
import itertools
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = Path(__file__).resolve().parent.parent


def clip_polygon(poly, axis, bound, keep_greater):
    if not poly:
        return []
    out = []
    for a, b in zip(poly, poly[1:] + poly[:1]):
        da = (a[axis] - bound) * (1.0 if keep_greater else -1.0)
        db = (b[axis] - bound) * (1.0 if keep_greater else -1.0)
        ina, inb = da >= -1e-12, db >= -1e-12
        if ina:
            out.append(a)
        if ina != inb:
            t = da / (da - db)
            out.append([a[k] + t * (b[k] - a[k]) for k in range(3)])
    return out


def clip_to_box(poly):
    for axis in range(3):
        poly = clip_polygon(poly, axis, 0.0, True)
        poly = clip_polygon(poly, axis, 1.0, False)
    return poly


def main():
    with (HERE / "particles_128.csv").open() as f:
        particles = list(csv.DictReader(f))
    pos = {int(row["id"]): [float(row[k]) for k in ("x", "y", "z")] for row in particles}
    grouped = defaultdict(list)
    with (HERE / "faces_128.csv").open() as f:
        for row in csv.DictReader(f):
            key = (int(row["row"]), int(row["neighbor"]))
            grouped[key].append((int(row["vertex"]), [float(row[k]) for k in ("x", "y", "z")]))

    polygons = []
    for (i, j), entries in grouped.items():
        if i >= j:
            continue  # each shared face is drawn once
        entries.sort()
        poly = [point for _, point in entries]
        center = [sum(p[k] for p in poly) / len(poly) for k in range(3)]
        base_shift = [-int(center[k] // 1) for k in range(3)]
        for extra in itertools.product((-1, 0, 1), repeat=3):
            shift = [base_shift[k] + extra[k] for k in range(3)]
            shifted = [[p[k] + shift[k] for k in range(3)] for p in poly]
            clipped = clip_to_box(shifted)
            if len(clipped) >= 3:
                polygons.append(clipped)

    fig = plt.figure(figsize=(9, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    collection = Poly3DCollection(
        polygons, facecolors="#b8dce8", edgecolors="#47788c", linewidths=0.28, alpha=0.42
    )
    ax.add_collection3d(collection)
    xyz = [pos[i] for i in sorted(pos)]
    ax.scatter(
        *zip(*xyz),
        s=8,
        c="#d35f36",
        edgecolors="white",
        linewidths=0.25,
        depthshade=False,
        label="128 particles",
    )
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), xlabel="x", ylabel="y", zlabel="z")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=34)
    ax.set_title("Periodic local Voronoi mesh from the Kokkos reconstruction")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(False)
    fig.tight_layout()
    out = HERE / "voronoi_128.png"
    fig.savefig(out, dpi=220, facecolor="white", bbox_inches="tight")
    print(f"wrote {out} ({len(polygons)} clipped face polygons)")


if __name__ == "__main__":
    main()
