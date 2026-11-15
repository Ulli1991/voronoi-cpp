#!/usr/bin/env python3
"""Figure inputs: the five benchmark inputs and the three application inputs (the generators and seed of the
benchmarks; 64³ points, the fcc solid 2^17 atoms), one square panel each, points of a slab one mean spacing thick in one
ink: a 16 x 16-spacing zoom for the even inputs and the fcc solid, the whole box for clustered, halo, cosmic web and
surface (clustered and halo shifted periodically so the densest blob or halo is centred); below each, the
nearest-neighbour distances on one shared log axis."""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from paperfig import DATA, DOUBLE, LABEL, MM, SCALE, figstyle, save

ROWS = [["jittered", "bluenoise", "uniform", "clustered"], ["halo", "cosmicweb", "fccsolid", "surface"]]
WHOLE = ("clustered", "halo", "cosmicweb", "surface")
INK = "#222222"
ZOOM = 16  # mean spacings shown by the zoomed panels
SIZE = 3.0  # marker area, design pt²


def contrast(c):
    if c < 1000:  # two significant digits
        return f"1 : {float(f'{c:.2g}'):.0f}"
    return f"1 : {c / 10 ** np.floor(np.log10(c)):.0f}·10$^{{{int(np.log10(c))}}}$"


def main():
    d = np.load(os.path.join(DATA, "inputs.npz"))
    figstyle.apply_style()
    fig = plt.figure(figsize=(DOUBLE * MM / SCALE, 132 * MM / SCALE))
    grid = fig.add_gridspec(4, 4, height_ratios=[1, 0.27, 1, 0.27], hspace=0.30, wspace=0.10,
                            left=0.02, right=0.98, top=0.99, bottom=0.075)
    bins = d["bins"]
    centres = np.sqrt(bins[1:] * bins[:-1])
    strips = []
    for row, kinds in enumerate(ROWS):
        for i, kind in enumerate(kinds):
            ax = fig.add_subplot(grid[2 * row, i])
            xy = d[f"{kind}_slab"]
            n = 131072 if kind == "fccsolid" else 262144
            side = n ** (1 / 3)
            whole = kind in WHOLE
            lo, hi = (0.0, 1.0) if whole else (0.5 - ZOOM / side / 2, 0.5 + ZOOM / side / 2)
            keep = (xy[:, 0] >= lo) & (xy[:, 0] < hi) & (xy[:, 1] >= lo) & (xy[:, 1] < hi)
            dense = kind == "halo"  # its core would be solid ink at the common marker size
            ax.scatter(xy[keep, 0], xy[keep, 1], s=SIZE / 3 if dense else SIZE, c=INK, alpha=0.5 if dense else 1,
                       linewidths=0, rasterized=True)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.minorticks_off()
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("0.5")
            span = hi - lo
            bar = (8 if whole else 1) / side
            ax.add_patch(Rectangle((0.015, 0.015), 0.6, 0.11, transform=ax.transAxes, fc="white", ec="0.6", lw=0.5,
                                   zorder=4))
            x0, y0 = lo + 0.05 * span, lo + 0.07 * span
            ax.plot([x0, x0 + bar], [y0, y0], color="#d62728", lw=2.2, solid_capstyle="butt", zorder=5)
            ax.text(x0 + bar + 0.03 * span, y0, "8 spacings" if whole else "1 spacing", color="#d62728",
                    ha="left", va="center", fontsize=11, zorder=5)
            ax.set_xlabel(f"{LABEL[kind]} · {contrast(float(d[kind + '_contrast']))}", fontsize=12, labelpad=4)
            h = d[f"{kind}_nn_hist"].astype(float)
            h /= h.max()
            sx = fig.add_subplot(grid[2 * row + 1, i], sharex=strips[0] if strips else None)
            strips.append(sx)
            sx.fill_between(centres, h, color=INK, alpha=0.85, lw=0, step="mid")
            sx.set_xscale("log")
            sx.set_xlim(3e-3, 3)
            sx.set_ylim(0, 1.05)
            sx.set_yticks([])
            sx.grid(False)
            sx.tick_params(axis="x", which="both", top=False, labelsize=11)
            sx.tick_params(axis="y", which="both", left=False, right=False)
            sx.set_xticks([0.01, 0.1, 1], ["0.01", "0.1", "1"])
            for s in ("top", "right", "left"):
                sx.spines[s].set_visible(False)
            sx.spines["bottom"].set_linewidth(0.6)
            sx.axvline(1.0, color="0.6", lw=0.8, ls=":")
    fig.text(0.5, 0.008, "nearest-neighbour distance, mean spacings", ha="center", va="bottom", fontsize=12)
    save(fig, "inputs", DOUBLE)


if __name__ == "__main__":
    main()
