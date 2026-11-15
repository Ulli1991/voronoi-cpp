#!/usr/bin/env python3
"""Figure: DRAM bytes per cell of each kernel group (Nsight Compute, 128³ jittered unless named) against the bytes per
cell that 5x and 10x the measured rate would allow at 80 % of the A100's 1555 GB/s. Data: paper/data/ceiling.json
(scripts/ceiling_tables.py on the jobs of scripts/ceiling_a100.sh); rows, groups and rates as scripts/ceiling_figure.py."""
import json
import os
import sys

import matplotlib.pyplot as plt

from paperfig import DATA, DOUBLE, REPO, figstyle, figure, save

sys.path.insert(0, os.path.join(REPO, "scripts"))
from ceiling_figure import GROUPS, ROWS, TARGET, run  # noqa: E402

COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#7f7f7f"]


def main():
    data = json.load(open(os.path.join(DATA, "ceiling.json")))
    fig, axes = figure(DOUBLE, 80, ncols=2)
    names = [g[0] for g in GROUPS] + ["other"]
    short = {"repair (find, plan, apply)": "repair", "screen + circumcentres": "screen, centres",
             "face numbers, rings": "numbers, rings"}
    for ax, rows, xmax, tag in ((axes[0], ROWS[:3], 6000, "(a) moving step"), (axes[1], ROWS[3:], 18000,
                                                                                 "(b) builds")):
        ys = list(range(len(rows)))[::-1]
        for y, (label, parts, base) in zip(ys, rows):
            per = dict.fromkeys(names, 0.0)
            for key, phase in parts:
                for r in run(data, key):
                    if r["phase"] == phase:
                        per[next((n for n, labels in GROUPS if r["label"] in labels), "other")] += r["B"]
            left = 0.0
            for n, c in zip(names, COLOURS):
                if per[n] > 0:
                    ax.barh(y, per[n], left=left, color=c, height=0.55, zorder=3, label=short.get(n, n))
                left += per[n]
            edge = max(left, TARGET / (5 * base * 1e6))
            ax.text(edge + 0.02 * xmax, y, f"{left:,.0f}", va="center")
            for mult, ls in ((5, "-"), (10, "--")):
                b = TARGET / (mult * base * 1e6)
                ax.plot([b, b], [y - 0.4, y + 0.4], color="k", lw=2.2, ls=ls, zorder=4,
                        label=f"bytes for {mult}× the rate")
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_locator(plt.MultipleLocator(2000 if xmax < 10000 else 5000))
        ax.set_ylim(-0.6, len(rows) + 2.0)
        ax.set_yticks(ys, [r[0] for r in rows])
        figstyle.decorate(ax)
        ax.tick_params(axis="y", which="both", left=False, right=False)
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel("DRAM bytes per cell, read + written")
        ax.text(0.02, 0.97, tag, transform=ax.transAxes, va="top")
        seen = {}
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
        ax.legend(list(seen.values()), list(seen.keys()), loc="upper right", fontsize=11, ncol=2, columnspacing=0.8, bbox_to_anchor=(1.0, 0.9),
                  handlelength=1.5)
    fig.tight_layout()
    save(fig, "ceiling", DOUBLE)


if __name__ == "__main__":
    main()
