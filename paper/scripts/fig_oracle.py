#!/usr/bin/env python3
"""Figure: relative volume error against the brute-force oracle (1000 sampled cells per input), voronoi-cpp's warp
rebuild and exact start on the A100 per input at 256³, and the codes of the library benchmark on halo input at 128³.
Data: paper/data/oracle_errors.csv (paper/scripts/extract_oracle.py)."""
import matplotlib.ticker
import numpy as np

import matplotlib

from common import oracle_errors
from paperfig import COLOR, DOUBLE, INPUTS, LABEL, LW, figstyle, figure, save

FLOOR = 1e-18  # errors of exactly 0 are drawn in the first bin
BINS = np.logspace(-18, -2, 65)
EPS = 2.0 ** -53
OTHERS = [  # (volumes prefix, label, colour); halo 2,097,152, the hardest input
    ("voronoi-cpp-warp-a100", "voronoi-cpp", "#000000"),
    ("voropp-mt-cpu", "voro++", "#1f77b4"),
    ("cgal-dt3-par-cpu", "CGAL + our faces", "#2ca02c"),
    ("geogram-pdel-cpu", "geogram periodic", "#9467bd"),
    ("peclet-voro-a100", "peclet-voro", "#ff7f0e"),
    ("basselin-white-a100", "Basselin et al.", "#8c564b"),
    ("vorogpu-k190-a100", "VoroGPU (float)", "#d62728"),
]


def curve(ax, e, color, label, ls="-"):
    finite = e[np.isfinite(e)]
    h, _ = np.histogram(np.maximum(finite, FLOOR), BINS)
    ax.stairs(h / len(e), BINS, color=color, lw=LW, ls=ls, label=label)


def main():
    err = oracle_errors()
    fig, axes = figure(DOUBLE, 72, ncols=3, gridspec_kw=dict(width_ratios=[1, 1, 1.45]))
    for ax, path, title in zip(axes[:2], ["warp", "exact"], ["warp rebuild (A100)", "exact start (A100)"]):
        for kind in INPUTS:
            name = f"voronoi-cpp-{path}-a100_{kind}_16777216.f64"
            if name in err:
                curve(ax, err[name][2], COLOR[kind], LABEL[kind])
        ax.text(0.04, 0.95, f"({'ab'[path == 'exact']}) {title}\n256³", transform=ax.transAxes, va="top")
        ax.set_xlim(2e-19, 3e-12)
        ax.set_xticks([1e-17, 1e-15, 1e-13])
        ax.set_ylim(0, 0.62)
    for prefix, label, color in OTHERS:
        n, missing, e = err[f"{prefix}_halo_2097152.f64"]
        curve(axes[2], e, color, label + (f": {missing:,} missing" if missing else ""))
    axes[2].text(0.04, 0.95, "(c) halo, 128³", transform=axes[2].transAxes, va="top")
    axes[2].set_xlim(2e-19, 1e-2)
    axes[2].set_xticks([1e-16, 1e-12, 1e-8, 1e-4])
    axes[2].set_ylim(0, 0.88)
    axes[2].annotate("= 0", (1.4e-18, 0.34), xytext=(4e-18, 0.37), fontsize=11, arrowprops=dict(arrowstyle="-", lw=0.8))
    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(EPS, color="0.4", lw=1.0, ls=":")
        ax.set_xlabel(r"$|V - V_\mathrm{oracle}|\,/\,V_\mathrm{oracle}$")
        figstyle.decorate(ax)
        ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=30))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("fraction of sampled cells")
    axes[0].legend(loc="upper left", fontsize=11, bbox_to_anchor=(0.0, 0.84))
    axes[2].legend(loc="upper right", fontsize=11, bbox_to_anchor=(1.0, 0.86))
    fig.subplots_adjust(wspace=0.28)
    save(fig, "oracle_errors", DOUBLE)


if __name__ == "__main__":
    main()
