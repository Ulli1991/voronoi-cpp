#!/usr/bin/env python3
"""Figure inputs: the five benchmark inputs and the three application inputs (the generators and seed of the
benchmarks; 64³ points, the fcc solid 2^17 atoms), one square panel each in the input's colour, points of a slab one
mean spacing thick: a 16 x 16-spacing zoom for the even inputs and the fcc solid, the whole box for clustered, halo,
cosmic web and surface (clustered and halo shifted periodically so the densest blob or halo is centred); below, one
shared panel of the nearest-neighbour distances of all eight."""
import os

import matplotlib.pyplot as plt
import numpy as np

from paperfig import COLOR, DATA, DOUBLE, LABEL, LW, MM, SCALE, figstyle, save, style

ROWS = [["jittered", "bluenoise", "uniform", "clustered"], ["halo", "cosmicweb", "fccsolid", "surface"]]
APPLICATION = ("cosmicweb", "fccsolid", "surface")
WHOLE = ("clustered", "halo", "cosmicweb", "surface")
ZOOM = 16  # mean spacings shown by the zoomed panels
SIZE = 3.0  # marker area, design pt²


def contrast(c):
    if c < 1000:  # two significant digits
        return f"1 : {float(f'{c:.2g}'):.0f}"
    mantissa, power = c / 10 ** np.floor(np.log10(c)), int(np.log10(c))
    return f"1 : 10$^{{{power}}}$" if round(mantissa) == 1 else f"1 : {mantissa:.0f}·10$^{{{power}}}$"


def points(fig, cell, kind, d):
    ax = fig.add_subplot(cell)
    xy = d[f"{kind}_slab"]
    side = (131072 if kind == "fccsolid" else 262144) ** (1 / 3)
    whole = kind in WHOLE
    lo, hi = (0.0, 1.0) if whole else (0.5 - ZOOM / side / 2, 0.5 + ZOOM / side / 2)
    keep = (xy[:, 0] >= lo) & (xy[:, 0] < hi) & (xy[:, 1] >= lo) & (xy[:, 1] < hi)
    dense = kind == "halo"  # at the common size the whole halo is solid ink and its core does not show
    size = SIZE / 3 if dense else SIZE if whole else 2.5 * SIZE  # the zooms hold few points: larger, so light colours show
    ax.scatter(xy[keep, 0], xy[keep, 1], s=size, c=COLOR[kind], alpha=0.5 if dense else 1,
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
    # the scale bar and its label without a box: a white outline keeps them readable on the points
    from matplotlib import patheffects
    halo = [patheffects.withStroke(linewidth=3.5, foreground="white")]
    x0, y0 = lo + 0.05 * span, lo + 0.07 * span
    ax.plot([x0, x0 + bar], [y0, y0], color="black", lw=2.2, solid_capstyle="butt", zorder=5, path_effects=halo)
    ax.text(x0 + bar + 0.03 * span, y0, "8 spacings" if whole else "1 spacing", color="black", ha="left",
            va="center", fontsize=11, zorder=5, path_effects=halo)
    ax.set_xlabel(f"{LABEL[kind]} · {contrast(float(d[kind + '_contrast']))}", fontsize=12, labelpad=4)


def main():
    d = np.load(os.path.join(DATA, "inputs.npz"))
    style()
    fig = plt.figure(figsize=(DOUBLE * MM / SCALE, 146 * MM / SCALE))
    grid = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.66], hspace=0.16, wspace=0.08,
                            left=0.075, right=0.99, top=0.995, bottom=0.075)
    for row, kinds in enumerate(ROWS):
        for i, kind in enumerate(kinds):
            points(fig, grid[row, i], kind, d)
    # one shared panel: the nearest-neighbour distances of all eight inputs
    ax = fig.add_subplot(grid[2, :])
    bins = d["bins"]
    for kind in ROWS[0] + ROWS[1]:
        h = d[f"{kind}_nn_hist"].astype(float)
        f = h / h.sum()
        ax.stairs(np.where(f > 0, f, np.nan), bins, color=COLOR[kind], lw=LW * 0.8, label=LABEL[kind],
                  ls="--" if kind in APPLICATION else "-")
    ax.axvline(1.0, color="0.4", lw=1.0, ls=":")
    ax.text(0.94, 0.92, "mean spacing", transform=ax.get_xaxis_transform(), fontsize=11, color="0.3", ha="right",
            va="top")  # in the headroom above the peaks
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(3e-3, 4)
    ax.set_ylim(1e-5, 40)
    ax.set_xticks([0.01, 0.1, 1], ["0.01", "0.1", "1"])
    ax.set_xlabel("nearest-neighbour distance / mean spacing")
    ax.set_ylabel("fraction of points")
    figstyle.decorate(ax)
    ax.legend(loc="upper left", ncol=4, fontsize=11)
    save(fig, "inputs", DOUBLE)


if __name__ == "__main__":
    main()
