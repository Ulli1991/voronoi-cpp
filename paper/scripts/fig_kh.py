#!/usr/bin/env python3
"""Part 2 figure, 2D Kelvin-Helmholtz instability (McNally, Lyra & Passy 2012) with luna-cpp on the moving Voronoi mesh.
Top: the density at t = 2 on the moving mesh (N = 1024), each pixel coloured by the cell it lies in. Bottom: the amplitude M
of the v_y mode with k = 4 pi (McNally's diagnostic) against time, each code at its highest resolution (CODES), with the
slope of linear theory for this setup, sigma = 2.832 (kh_linear_mcnally.py: the linearised compressible Euler equations
about the same rho(y), u(y)). Data: paper/data/luna/kh/ (mode_<mesh>_<N>.txt: t, M, peak kinetic energy density, every
0.02, from hydro_kh2d on one MI300A; density_t2_1024.npz from kh_density())."""
import os
import sys

import numpy as np

from paperfig import DATA, DOUBLE, LW, figstyle, figure, save

KH = os.path.join(DATA, "luna", "kh")
SIGMA = 2.831857  # kh_linear_mcnally.py, 512 collocation points (2.831714 at 256)
SIZES = (256, 512, 1024)
WINDOW = (0.6, 1.2)  # the fit of the measured rate: after the start's transient, before saturation
COMPARE = os.path.join(KH, "compare")
# file stem, directory, legend label, colour; drawn in this order, a code without files is left out
CODES = [
    ("arepo", COMPARE, "Arepo", "#9467bd"),
    ("athenak_plm_rk2", COMPARE, "AthenaK", "#ff7f0e"),
    ("luna", COMPARE, "LUNA, fixed grid", "#2ca02c"),
    ("static", KH, "Kikkou + luna-cpp, static mesh (this work)", "#17becf"),
    ("moving", KH, "Kikkou + luna-cpp, moving mesh (this work)", "#1f77b4"),
]


def kh_density(prefix, pixels=1024):
    """density_t2_1024.npz from hydro_kh2d's field dump (rows: generator x, y; centroid x, y; volume; rho; rho v_y)."""
    from scipy.spatial import cKDTree

    rows = np.fromfile(prefix + ".bin").reshape(-1, 7)
    at = (np.arange(pixels) + 0.5) / pixels
    x, y = np.meshgrid(at, at, indexing="ij")
    cell = cKDTree(rows[:, :2] % 1.0, boxsize=1.0).query(np.c_[x.ravel(), y.ravel()], workers=4)[1]
    np.savez_compressed(os.path.join(KH, "density_t2_1024.npz"), rho=rows[cell, 5].reshape(pixels, pixels).astype(np.float32))


def rate(t, m):
    """Least-squares slope of ln M over WINDOW."""
    keep = (t >= WINDOW[0]) & (t <= WINDOW[1])
    return float(np.polyfit(t[keep], np.log(m[keep]), 1)[0])


def main():
    fig, (a, b) = figure(110.0, 185, nrows=2, gridspec_kw=dict(height_ratios=[1.25, 1.0]))
    rho = np.load(os.path.join(KH, "density_t2_1024.npz"))["rho"]
    image = a.imshow(rho.T, origin="lower", extent=(0, 1, 0, 1), cmap="magma", interpolation="nearest")
    a.set_xlabel("x")
    a.set_ylabel("y")
    a.grid(False)
    bar = fig.colorbar(image, ax=a, location="right", fraction=0.046, pad=0.03)
    bar.set_label(r"$\rho$")
    a.text(0.04, 0.96, "(a)", transform=a.transAxes, va="top", color="white")
    b.text(0.04, 0.96, "(b)", transform=b.transAxes, va="top")
    rates = {}
    for k, (stem, where, label, colour) in enumerate(CODES):
        runs = {n: np.loadtxt(f).T[:2] for n in SIZES if os.path.exists(f := os.path.join(where, f"mode_{stem}_{n}.txt"))}
        rates.update({(stem, n): rate(*r) for n, r in runs.items()})
        if runs:  # the highest resolution each code has; widths step down so every curve shows under the next
            top = max(runs)  # a code not yet run at the largest size says which size it shows
            b.plot(*runs[top], "-", color=colour, lw=LW * (2.6 - 0.5 * k), solid_capstyle="butt",
                   label=label if top == SIZES[-1] else f"{label}, {top}$^2$")
    t0 = WINDOW[0]
    t, m, _ = np.loadtxt(os.path.join(KH, "mode_moving_1024.txt")).T
    anchor = 0.45 * np.interp(t0, t, m)
    b.set_yscale("log")
    b.set_ylim(3e-3, 0.4)  # the theory line crosses the whole panel, below the codes, clipped at the edges
    line = np.array([0.0, 2.5])
    b.plot(line, anchor * np.exp(SIGMA * (line - t0)), ":", color="k", lw=1.8, label=r"linear theory, $\sigma = 2.83$")
    b.set_xlim(0, 2.5)
    b.set_xlabel("t")
    b.set_ylabel("mode amplitude M")
    figstyle.decorate(b)
    b.legend(loc="lower right", fontsize=11, frameon=False)
    fig.tight_layout()
    save(fig, "luna_kh", 110.0)
    for (code, n), s in rates.items():
        print(f"{code} {n}: measured rate {s:.3f} over t in {WINDOW}, linear theory {SIGMA:.3f}, {100 * (s / SIGMA - 1):+.1f} %")


if __name__ == "__main__":
    if len(sys.argv) > 1:  # fig_kh.py <prefix of hydro_kh2d's dump at t = 2, N = 1024>
        kh_density(sys.argv[1])
    main()
