#!/usr/bin/env python3
"""Figure stages: where a moving step's time goes at 128³ on one A100. (a) per input and displacement, the stages of
update() + update_cells() (median of steps 2 to 10); (b) jittered input, the kernels grouped as in the ceiling figure.
Data: paper/data/stages.csv, kernels.csv (extract_stages.py on stages_a100.sh kernels; bench copies left out)."""
import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

from paperfig import DATA, DOUBLE, LABEL, REPO, figstyle, figure, save

sys.path.insert(0, os.path.join(REPO, "scripts"))
from ceiling_figure import GROUPS  # noqa: E402

N = 2097152
INPUTS = ["jittered", "uniform", "clustered", "halo", "lattice", "wave"]
SPACINGS = ["0.03", "0.1", "0.3"]
STAGES = [("repair", "#d62728"), ("host fallback", "#ff9896"), ("rebuild", "#7f7f7f"), ("geometry", "#1f77b4"),
          ("cells", "#ff7f0e")]
GROUP_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#8c564b", "#e377c2", "#17becf", "#7f7f7f"]
BENCH = ("Kokkos::View::initialization", "deep_copy")  # the bench's host copy of the positions


def ns(ms):
    return ms * 1e6 / N


def main():
    stage = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(DATA, "stages.csv"))):
        if r["what"].startswith("step"):
            stage[r["input"], r["what"].split()[1]][r["stage"]] = float(r["ms"])
    kern = defaultdict(lambda: defaultdict(float))
    for r in csv.DictReader(open(os.path.join(DATA, "kernels.csv"))):
        if r["input"] == "jittered" and r["kernel"] not in BENCH:
            g = next((name for name, labels in GROUPS if r["kernel"] in labels), "other")
            kern[r["spacings"]][g] += float(r["ms_per_step"])
    fig, (a, b) = figure(DOUBLE, 92, ncols=2, gridspec_kw=dict(width_ratios=[1.35, 1]))
    y, ticks, labels = 0, [], []
    for kind in INPUTS:
        for d in SPACINGS:
            got = stage.get((kind, d))
            if not got:
                continue
            left = 0.0
            for name, colour in STAGES:
                v = ns(got.get(name, 0.0))
                a.barh(y, v, left=left, color=colour, height=0.75, label=name if y == 0 else None)
                left += v
            a.text(left + 0.3, y, f"{1e3 / left:.0f}", va="center", fontsize=11)
            ticks.append(y)
            labels.append(f"{LABEL[kind]}, {d}" if d == "0.03" else d)
            y += 1
        y += 0.6
    a.set_yticks(ticks, labels, fontsize=11)
    a.set_ylim(y - 0.4, -0.8)
    a.set_xlim(0, 26)
    a.set_xlabel("ns per cell (numbers: M cells/s)")
    a.legend(loc="lower right", fontsize=11)
    a.text(0.98, 0.98, "(a)", transform=a.transAxes, ha="right", va="top",
           fontsize=11)
    names = [g[0] for g in GROUPS] + ["other"]
    for i, d in enumerate(SPACINGS):
        left = 0.0
        for name, colour in zip(names, GROUP_COLOURS):
            v = ns(kern[d].get(name, 0.0))
            used = any(kern[e].get(name, 0.0) > 0 for e in SPACINGS)
            b.barh(i, v, left=left, color=colour, height=0.6, label=name if i == 0 and used else None)
            left += v
        b.text(left + 0.5, i, f"{left:.1f}", va="center", fontsize=11)
    b.set_yticks(range(3), [f"{d} spacings" for d in SPACINGS], fontsize=11)
    b.set_ylim(8.6, -0.6)
    b.set_xlim(0, 22)
    b.set_xlabel("kernel time, ns per cell")
    b.legend(loc="lower right", fontsize=11, handlelength=1.2)
    b.text(0.98, 0.98, "(b) jittered", transform=b.transAxes, ha="right", va="top", fontsize=11)
    for ax in (a, b):
        figstyle.decorate(ax)
        ax.tick_params(axis="y", which="both", left=False, right=False)
        ax.yaxis.set_minor_locator(plt.NullLocator())
    fig.tight_layout()
    save(fig, "stages", DOUBLE)


if __name__ == "__main__":
    main()
