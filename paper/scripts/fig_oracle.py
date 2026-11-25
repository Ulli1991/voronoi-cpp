#!/usr/bin/env python3
"""Figure: relative volume error against the exact reference (src/test/oracle.hpp; 1000 sampled cells per input), voronoi-cpp's warp
rebuild and exact start on the A100 per input at 128³ (the paper's rerun, 55cd6b1), and the codes of the library benchmark on halo input at 128³.
Data: paper/data/oracle_errors.csv (paper/scripts/extract_oracle.py)."""
import matplotlib.ticker
import numpy as np

import matplotlib

from common import oracle_errors
from paperfig import COLOR, DOUBLE, INPUTS, LABEL, LW, figstyle, figure, save

ZERO = 2.2e-19  # errors of exactly 0: their own bin left of an axis break, labelled "0"
FLOOR = 1e-18  # nonzero errors below this go into the first log bin
BINS = np.concatenate([[1.2e-19, 4e-19], np.logspace(-18, -2, 65)])  # [4e-19, 1e-18) stays empty: the break
OTHERS = [  # (volumes prefix, label, colour); halo 2,097,152, the hardest input
    ("voronoi-cpp-warp-a100", "this work", "#000000"),
    ("voropp-mt-cpu", "voro++", "#1f77b4"),
    ("cgal-dt3-par-cpu", "CGAL + faces", "#2ca02c"),
    ("geogram-pdel-cpu", "geogram", "#9467bd"),
    ("peclet-voro-a100", "peclet-voro", "#ff7f0e"),
    ("basselin-white-a100", "Basselin et al.", "#8c564b"),
    ("vorogpu-k190-a100", "Ray et al. (float)", "#d62728"),
]


def curve(ax, e, color, label, ls="-"):
    finite = e[np.isfinite(e)]
    h, _ = np.histogram(np.where(finite == 0, ZERO, np.maximum(finite, FLOOR)), BINS)
    ax.stairs(h / len(e), BINS, color=color, lw=LW, ls=ls, label=label)


def main():
    err = oracle_errors()
    fig, axes = figure(DOUBLE, 72, ncols=3, gridspec_kw=dict(width_ratios=[1, 1, 1.45]))
    for ax, path, title in zip(axes[:2], ["warp", "exact"], ["rebuild", "exact start"]):
        for kind in INPUTS:
            name = f"voronoi-cpp-{path}-a100_{kind}_2097152.f64"
            if name in err:
                curve(ax, err[name][2], COLOR[kind], LABEL[kind])
        ax.text(0.04, 0.96, f"({'ab'[path == 'exact']}) {title}", transform=ax.transAxes, va="top")
        ax.set_xlim(1e-19, 3e-12)
        ax.set_ylim(0, 0.62)
    for prefix, label, color in OTHERS:
        n, missing, e = err[f"{prefix}_halo_2097152.f64"]
        curve(axes[2], e, color, label)  # cells missing: in the caption
    axes[2].text(0.04, 0.96, "(c) halo", transform=axes[2].transAxes, va="top")
    axes[2].set_xlim(1e-19, 1e-2)
    axes[2].set_ylim(0, 0.88)
    for ax, ticks in zip(axes, ([1e-17, 1e-15, 1e-13],) * 2 + ([1e-16, 1e-12, 1e-8, 1e-4],)):
        ax.set_xscale("log")
        ax.set_xticks([ZERO] + ticks, ["0"] + [f"$10^{{{int(np.log10(t))}}}$" for t in ticks])
        for x in (5.6e-19, 7.2e-19):  # the axis break between the exact-zero bin and the log axis
            ax.plot([x / 1.06, x * 1.06], [-0.012, 0.012], transform=ax.get_xaxis_transform(), color="black", lw=1.2,
                    clip_on=False, zorder=6)
        ax.set_xlabel(r"$|V - V_\mathrm{exact}|\,/\,V_\mathrm{exact}$")
        figstyle.decorate(ax)
        ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=30))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("fraction of sampled cells")
    axes[0].legend(loc="upper left", fontsize=11, bbox_to_anchor=(0.02, 0.9))
    axes[2].legend(loc="upper right", fontsize=11, bbox_to_anchor=(0.98, 0.9))
    fig.subplots_adjust(wspace=0.28)
    save(fig, "oracle_errors", DOUBLE)


if __name__ == "__main__":
    main()
