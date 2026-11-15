#!/usr/bin/env python3
"""Figure inputs: the five benchmark inputs at 64³ (the generators and seed of the benchmarks), one square panel each,
points of a slab one mean spacing thick in one ink: a 16 x 16-spacing zoom for the even inputs, the whole box for
clustered and halo (shifted periodically so the densest blob or halo is centred); below, nearest-neighbour distances."""
import os

import matplotlib.pyplot as plt
import numpy as np

from paperfig import DATA, DOUBLE, LABEL, MM, SCALE, figstyle, save

MAIN = ["jittered", "bluenoise", "uniform", "clustered", "halo"]
INK = "#222222"
ZOOM = 16 / 64  # 16 mean spacings of the 64³ set
SIZE = 3.0  # marker area, design pt²


def contrast(c):
    return f"1 : {c:.0f}" if c < 100 else f"1 : {c / 10 ** np.floor(np.log10(c)):.0f}·10$^{{{int(np.log10(c))}}}$"


def main():
    d = np.load(os.path.join(DATA, "inputs.npz"))
    figstyle.apply_style()
    fig = plt.figure(figsize=(DOUBLE * MM / SCALE, 62 * MM / SCALE))
    grid = fig.add_gridspec(2, 5, height_ratios=[1, 0.3], hspace=0.36, wspace=0.14,
                            left=0.015, right=0.985, top=0.985, bottom=0.13)
    bins = d["bins"]
    centres = np.sqrt(bins[1:] * bins[:-1])
    for i, kind in enumerate(MAIN):
        ax = fig.add_subplot(grid[0, i])
        xy = d[f"{kind}_slab"]
        whole = kind in ("clustered", "halo")
        lo, hi = (0.0, 1.0) if whole else (0.5 - ZOOM / 2, 0.5 + ZOOM / 2)
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
        bar = 1 / 64 if not whole else 8 / 64
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((0.015, 0.015), 0.6, 0.13, transform=ax.transAxes, fc="white", ec="0.6", lw=0.5,
                               zorder=4))
        x0, y0 = lo + 0.05 * span, lo + 0.05 * span
        ax.plot([x0, x0 + bar], [y0, y0], color="#d62728", lw=2.2, solid_capstyle="butt", zorder=5)
        ax.text(x0 + bar + 0.03 * span, y0, "1 spacing" if not whole else "8 spacings", color="#d62728",
                ha="left", va="center", fontsize=11, zorder=5)
        ax.set_xlabel(f"{LABEL[kind]} · {contrast(float(d[kind + '_contrast']))}", fontsize=12, labelpad=4)
        h = d[f"{kind}_nn_hist"].astype(float)
        h /= h.max()
        sx = fig.add_subplot(grid[1, i])
        sx.fill_between(centres, h, color=INK, alpha=0.85, lw=0, step="mid")
        sx.set_xscale("log")
        sx.set_xlim(1e-2, 3)
        sx.set_ylim(0, 1.05)
        sx.set_yticks([])
        sx.grid(False)
        sx.tick_params(axis="x", which="both", top=False, labelsize=11)
        sx.tick_params(axis="y", which="both", left=False, right=False)
        sx.set_xticks([0.01, 0.1, 1], ["0.01", "0.1", "1"])
        for side in ("top", "right", "left"):
            sx.spines[side].set_visible(False)
        sx.spines["bottom"].set_linewidth(0.6)
        sx.axvline(1.0, color="0.6", lw=0.8, ls=":")
    fig.text(0.5, 0.02, "nearest-neighbour distance, mean spacings", ha="center", va="bottom", fontsize=12)
    save(fig, "inputs", DOUBLE)


if __name__ == "__main__":
    main()
