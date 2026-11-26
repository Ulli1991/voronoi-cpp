#!/usr/bin/env python3
"""Part 2 figure, 3D Yee vortex. Left: the density in the mid-plane z = 1.25 at the final time t = 10 on luna-cpp's
moving mesh (n_x = 64), each pixel coloured by the cell it lies in, the cell boundaries drawn (the plane's cut through
the 3D Voronoi mesh). Right: L1(ρ) and L1(v) of the cell averages against n_x for the moving mesh (two flux passes,
today's scheme), the static Voronoi mesh and LUNA MC (Cartesian). 190 mm. Data: paper/data/luna/ (vortex_slab_64.npz
from hydro_wave3d --out, cut to the slab around the plane by vortex_slab(); vortex_norms.csv from the luna-cpp job
logs; LUNA MC from luna-reference-solutions; see SOURCES.md)."""
import csv
import os

import numpy as np

from paperfig import DATA, DOUBLE, LW, figstyle, figure, save

NX = (32, 64, 128, 256)
COLUMN = 110.0  # mm: the two panels one above the other
SIDE, ORIGIN, PLANE = (10.0, 10.0, 2.5), (-5.0, -5.0, 0.0), 1.25
SLAB = os.path.join(DATA, "luna", "vortex_slab_64_t{t}.npz")  # t: 0 and 10


def vortex_slab(npy, t, nx=64):
    """Writes SLAB for time t from hydro_wave3d's --out rows (generator 0-2, centroid 3-5, volume 6, rho 7, v 8-10, P 11): the cells
    within three spacings of the plane, positions in the box [0, side), float32."""
    r = np.load(npy)
    x = (r[:, :3] - np.array(ORIGIN)) % np.array(SIDE)
    near = np.abs(x[:, 2] - PLANE) < 3 * SIDE[0] / nx
    np.savez_compressed(SLAB.format(t=t), x=x[near].astype(np.float32), rho=r[near, 7].astype(np.float32), nx=nx)


def picture(ax, t, limits, pixels=900):
    """The plane's cut through the mesh: every pixel takes the density of its nearest generator (its Voronoi cell)."""
    from scipy.spatial import cKDTree

    d = np.load(SLAB.format(t=t))
    x, rho = d["x"].astype(float), d["rho"]
    at = (np.arange(pixels) + 0.5) / pixels * SIDE[0]
    gx, gy = np.meshgrid(at, at, indexing="ij")
    query = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, PLANE)]
    # periodic in x and y; the slab does not reach the box's z faces
    tiles = np.concatenate([x + np.array([i * SIDE[0], j * SIDE[1], 0.0]) for i in (-1, 0, 1) for j in (-1, 0, 1)])
    cell = (cKDTree(tiles).query(query, workers=4)[1] % len(x)).reshape(pixels, pixels)
    image = ax.imshow(rho[cell].T, origin="lower", extent=(ORIGIN[0], ORIGIN[0] + SIDE[0], ORIGIN[1], ORIGIN[1] + SIDE[1]),
                      cmap="magma", interpolation="nearest", vmin=limits[0], vmax=limits[1])
    edge = np.zeros(cell.shape, bool)
    edge[:-1] |= cell[:-1] != cell[1:]
    edge[:, :-1] |= cell[:, :-1] != cell[:, 1:]
    lines = np.zeros((pixels, pixels, 4))
    lines[..., 3] = 0.35 * edge.T
    ax.imshow(lines, origin="lower", extent=image.get_extent(), interpolation="nearest")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(False)
    bar = ax.figure.colorbar(image, ax=ax, location="right", fraction=0.046, pad=0.03)
    bar.set_label(r"$\rho$")


def main():
    rows = {}
    for r in csv.DictReader(line for line in open(os.path.join(DATA, "luna", "vortex_norms.csv")) if line[0] != "#"):
        rows[r["series"], int(r["nx"])] = r
    d = np.load(os.path.join(DATA, "luna", "yee_vortex_convergence.npz"))
    pick = (d["dim"] == 3) & (d["limiter"] == 1) & (d["vx"] == 0) & (d["cfl"] == 0.8) & (d["ic_quad"] == 8)
    order = np.argsort(d["nx"][pick])
    # (a) the density in the plane through the vortex on the moving mesh after 10 time units; (b) the L1 error of the
    # density against resolution. A third panel, the code's cell updates per second against problem size on the A100 and
    # the MI300A, joins when those runs exist.
    fig, axes = figure(COLUMN, 185, nrows=2, gridspec_kw=dict(height_ratios=[1.25, 1.0]))
    picture(axes[0], 10, (float(np.load(SLAB.format(t=10))["rho"].min()), 1.0))
    axes[0].text(0.04, 0.96, "(a)", transform=axes[0].transAxes, va="top", color="white")
    for ax, key, label, v in ((axes[1], "L1_rho", r"$L_1(\rho)$", "rho"),):
        nx = d["nx"][pick][order]
        keep = nx <= 256
        ax.plot(nx[keep], d[f"avg_{v}_L1"][pick][order][keep], "o-", color="#1f77b4", lw=LW, ms=7,
                label="LUNA MC, Cartesian")
        ax.plot(NX, [float(rows["sta", n][key]) for n in NX], "s--", color="#ff7f0e", lw=LW, ms=7,
                label="static Voronoi mesh")
        ax.plot(NX, [float(rows["tp", n][key]) for n in NX], "D-", color="#2ca02c", lw=LW, ms=7,
                label="moving Voronoi mesh")
        ref = [float(rows["tp", 32][key]) * (32 / n) ** 2 for n in NX]
        ax.plot(NX, ref, ":", color="k", lw=1.6, label=r"$\propto n_x^{-2}$")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(NX, [str(n) for n in NX])
        ax.set_xlim(26, 310)
        ax.set_xlabel(r"$n_x$ (cells $n_x^3$)")
        ax.set_ylabel(label)
        figstyle.decorate(ax)
        ax.xaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
    axes[1].legend(loc="lower left", fontsize=11)
    axes[1].text(0.96, 0.96, "(b)", transform=axes[1].transAxes, va="top", ha="right")
    # the test and solver (3D Yee vortex, HLLC, MC limiter) are in the caption
    fig.tight_layout()
    save(fig, "luna_vortex", COLUMN)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:  # fig_vortex.py <hydro_wave3d --out file of the moving mesh at t = 0> <the same at t = 10> (n_x = 64)
        vortex_slab(sys.argv[1], 0)
        vortex_slab(sys.argv[2], 10)
    main()
