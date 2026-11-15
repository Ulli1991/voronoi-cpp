#!/usr/bin/env python3
"""Part 2 figure, 3D Yee vortex: L1(ρ) and L1(v) of the cell averages against n_x for luna-cpp's moving mesh (two
flux passes, today's scheme), the static Voronoi mesh and LUNA MC (Cartesian), at 190 mm. Data: paper/data/luna/
(vortex_norms.csv from the luna-cpp job logs, LUNA MC from luna-reference-solutions; see SOURCES.md)."""
import csv
import os

import numpy as np

from paperfig import DATA, DOUBLE, LW, figstyle, figure, save

NX = (32, 64, 128, 256)


def main():
    rows = {}
    for r in csv.DictReader(line for line in open(os.path.join(DATA, "luna", "vortex_norms.csv")) if line[0] != "#"):
        rows[r["series"], int(r["nx"])] = r
    d = np.load(os.path.join(DATA, "luna", "yee_vortex_convergence.npz"))
    pick = (d["dim"] == 3) & (d["limiter"] == 1) & (d["vx"] == 0) & (d["cfl"] == 0.8) & (d["ic_quad"] == 8)
    order = np.argsort(d["nx"][pick])
    fig, axes = figure(DOUBLE, 72, ncols=2)
    for ax, key, label, v in ((axes[0], "L1_rho", r"$L_1(\rho)$", "rho"), (axes[1], "L1_v", r"$L_1(v)$", "v")):
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
    axes[0].legend(loc="lower left", fontsize=11)
    # the test and solver (3D Yee vortex, HLLC, MC limiter) are in the caption
    fig.tight_layout()
    save(fig, "luna_vortex", DOUBLE)


if __name__ == "__main__":
    main()
