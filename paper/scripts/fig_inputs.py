#!/usr/bin/env python3
"""Figure inputs: the benchmark's point distributions at 64³ (262,144 points; the same generators and seed as the
benchmarks), a slab one mean spacing thick projected to x-y per input, and the nearest-neighbour distances in mean
spacings. Lattice and wave (secondary) in the bottom row. Data: paper/data/inputs.npz (extract_inputs.py)."""
import os

import matplotlib.pyplot as plt
import numpy as np

from paperfig import COLOR, DATA, DOUBLE, LABEL, LW, MM, SCALE, figstyle, save

MAIN = ["jittered", "bluenoise", "uniform", "clustered", "halo"]
SECONDARY = ["lattice", "wave"]


def slab(ax, d, kind, tag, size=0.6):
    xy = d[f"{kind}_slab"]
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=COLOR[kind], linewidths=0, rasterized=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.minorticks_off()
    ax.grid(False)
    ax.set_title(f"{tag} {LABEL[kind]}", fontsize=13, loc="left", pad=4)


def main():
    d = np.load(os.path.join(DATA, "inputs.npz"))
    figstyle.apply_style()
    fig = plt.figure(figsize=(DOUBLE * MM / SCALE, 100 * MM / SCALE))
    grid = fig.add_gridspec(2, 5, height_ratios=[1, 1], hspace=0.32, wspace=0.08)
    for i, kind in enumerate(MAIN):
        slab(fig.add_subplot(grid[0, i]), d, kind, f"({'abcde'[i]})")
    ax = fig.add_subplot(grid[1, :3])
    bins = d["bins"]
    centres = np.sqrt(bins[1:] * bins[:-1])
    for kind in MAIN + SECONDARY:
        h = d[f"{kind}_nn_hist"].astype(float)
        h /= h.sum() * np.diff(np.log10(bins))
        if kind in SECONDARY:
            ax.axvline(1.0, color=COLOR[kind], lw=1.4, ls=":", label=None)
            continue
        c = float(d[kind + "_contrast"])
        ratio = f"{c:.0f}" if c < 100 else f"{c / 10 ** np.floor(np.log10(c)):.0f}·10$^{{{int(np.log10(c))}}}$"
        ax.plot(centres, h, color=COLOR[kind], lw=LW, label=f"{LABEL[kind]}, 1 : {ratio}")
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 3)
    ax.set_ylim(0, None)
    ax.set_xlabel("nearest-neighbour distance, mean spacings")
    ax.set_ylabel("density per decade")
    ax.text(0.02, 0.96, "(f)", transform=ax.transAxes, va="top")
    ax.legend(loc="upper left", fontsize=11, bbox_to_anchor=(0.06, 1.0), handlelength=1.5,
              title="density contrast", title_fontsize=11)
    figstyle.decorate(ax)
    for j, kind in enumerate(SECONDARY):
        slab(fig.add_subplot(grid[1, 3 + j]), d, kind, f"({'gh'[j]})", size=1.2)
        fig.axes[-1].set_title(f"({'gh'[j]}) {LABEL[kind]} (secondary)", fontsize=13, loc="left", pad=4)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.94, bottom=0.11)
    save(fig, "inputs", DOUBLE)


if __name__ == "__main__":
    main()
