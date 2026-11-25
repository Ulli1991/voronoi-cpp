#!/usr/bin/env python3
"""Figures of the sweep (paper/data/throughput_vs_n.csv, extract_sweep.py): throughput_vs_n (rebuild, exact start, step at
0.1 spacings with update_cells, peak memory, against n per input) and step_vs_displacement (step rate at 128³ against
the displacement per step). Usage: fig_sweep.py [commit] (default: the newest commit in the file)."""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

from paperfig import COLOR, DATA, DOUBLE, INPUTS, LABEL, LW, MARKER, SINGLE, figstyle, figure, save

SIZES = [32768, 262144, 2097152, 16777216]
SIZE_LABEL = ["32³", "64³", "128³", "256³"]


def load(commit=None):
    rows = list(csv.DictReader(open(os.path.join(DATA, "throughput_vs_n.csv"))))
    commit = commit or rows[-1]["commit"]
    table = defaultdict(dict)
    for r in rows:
        if r["commit"] == commit:
            table[(r["input"], r["path"])][int(r["n"])] = r
    clocks = sorted({int(c) for r in rows if r["commit"] == commit for c in r["sm_clock_MHz"].split()})
    return commit, table, clocks


def clock_text(clocks):
    return f"{clocks[0]} MHz" if len(clocks) == 1 else f"{clocks[0]}–{clocks[-1]} MHz" if clocks else "clock ?"


def throughput(commit, table, clocks):
    fig, axes = figure(DOUBLE, 120, nrows=2, ncols=2)
    panels = [(axes[0, 0], "rebuild", "Mcells_s_median", "(a) rebuild"),
              (axes[0, 1], "start", "Mcells_s_median", "(b) exact start"),
              (axes[1, 0], "step 0.1", "Mcells_s_median", "(c) moving step"),
              (axes[1, 1], "start", "bytes_per_cell", "(d) peak memory, exact start")]
    for ax, path, col, text in panels:
        top = 0
        for kind in INPUTS:
            got = table.get((kind, path), {})
            pts = [(i, float(got[n][col])) for i, n in enumerate(SIZES) if n in got and got[n][col]]
            if pts:
                ax.plot(*zip(*pts), color=COLOR[kind], marker=MARKER[kind], lw=LW, ms=7, label=LABEL[kind])
                top = max(top, max(y for _, y in pts))
        ax.set_xticks(range(len(SIZES)), SIZE_LABEL)
        ax.set_xlim(-0.3, len(SIZES) - 0.7)
        ax.set_ylim(0, (1.55 if ax is axes[0, 0] else 1.2) * top if top else 1)  # headroom for the legend in (a)
        ax.text(0.03, 0.96, text, transform=ax.transAxes, va="top")
        ax.set_ylabel("B/cell" if col == "bytes_per_cell" else "M cells/s")
        figstyle.decorate(ax)
        ax.xaxis.set_minor_locator(matplotlib_null())
    for ax in axes[1]:
        ax.set_xlabel("cells n")
    axes[0, 0].legend(loc="upper right", ncol=4, fontsize=11, bbox_to_anchor=(1.0, 0.9))
    fig.tight_layout()
    save(fig, "throughput_vs_n", DOUBLE)


def matplotlib_null():
    import matplotlib.ticker
    return matplotlib.ticker.NullLocator()


def displacement(commit, table, clocks):
    fig, ax = figure(SINGLE, 75)
    for kind in INPUTS:
        pts = sorted((float(p.split()[1]), float(got[2097152]["Mcells_s_median"]))
                     for (k, p), got in table.items() if k == kind and p.startswith("step") and 2097152 in got)
        if pts:
            ax.plot(*zip(*pts), color=COLOR[kind], marker=MARKER[kind], lw=LW, ms=7, label=LABEL[kind])
    starts = [float(table[(k, "start")][2097152]["Mcells_s_median"]) for k in INPUTS
              if 2097152 in table.get((k, "start"), {})]
    if starts:
        ax.axhspan(min(starts), max(starts), color="0.85", zorder=0)
        ax.text(0.0105, min(starts) * 1.1, "exact start, all inputs", fontsize=11, color="0.3", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("displacement per step, mean spacings")
    ax.set_ylabel("moving step, M cells/s")
    ax.set_ylim(min(starts) * 0.8 if starts else None, 450)
    ax.legend(loc="lower left", ncol=2, fontsize=11, bbox_to_anchor=(0.0, 0.34))
    figstyle.decorate(ax)
    fig.tight_layout()
    save(fig, "step_vs_displacement", SINGLE)


def main():
    commit, table, clocks = load(sys.argv[1] if len(sys.argv) > 1 else None)
    throughput(commit, table, clocks)
    displacement(commit, table, clocks)


if __name__ == "__main__":
    main()
