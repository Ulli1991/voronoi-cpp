#!/usr/bin/env python3
"""Figure resort: step rate (update + update_cells, M cells/s, median of 11 steps around each) over 500 steps at 0.1
spacings, jittered, without re-sort and with a Hilbert re-sort every 50 updates, 128³ and 256³. Rates over the whole
run (re-sorts paid) in the legend. Data: paper/data/raw/long128-<job>.txt, long256-<job>.txt (stages_a100.sh)."""
import glob
import os
import re

import numpy as np

from paperfig import DATA, LW, SINGLE, figstyle, figure, save


def runs(path):
    """{flags: (n, steps, ms per step)} of one long-run log."""
    out, flags, n = {}, None, 0
    for line in open(path, errors="replace"):
        if line.startswith("== long"):
            n, flags = int(line.split()[2]), line.split("flags=")[1].strip() or "none"
            out[flags] = (n, [], [])
        elif line.startswith("gpuflips:") and flags and "all_ms=" in line:
            f = dict(re.findall(r"(\w+)=([-\w.+]+)", line))
            out[flags][1].append(int(f["step"]))
            out[flags][2].append(float(f["all_ms"]))
    return out


def main():
    fig, ax = figure(SINGLE, 70)
    for mode, colour in (("long128", "#1f77b4"), ("long256", "#d62728")):
        paths = sorted(glob.glob(os.path.join(DATA, "raw", f"{mode}-*.txt")))
        if not paths:
            continue
        for flags, (n, steps, ms) in runs(paths[-1]).items():
            ms = np.array(ms)
            rate = n / ms / 1e3
            smooth = [np.median(rate[max(0, i - 5):i + 6]) for i in range(len(rate))]
            over = n * len(ms) / ms.sum() / 1e3
            size = "128³" if n == 2097152 else "256³"
            label = f"{size}, {'no re-sort' if flags == 'none' else 're-sort every 50'}: {over:.0f}"
            ax.plot(steps, smooth, color=colour, lw=LW, ls="-" if flags == "none" else "--", label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("step + update_cells, M cells/s")
    ax.set_xlim(0, 500)
    lo, hi = ax.get_ylim()
    ax.set_ylim(0, hi * 1.6)  # headroom: the legend clears the curves
    ax.legend(loc="upper right", fontsize=11, title="over the run, M cells/s", title_fontsize=11)
    figstyle.decorate(ax)
    fig.tight_layout()
    save(fig, "resort", SINGLE)


if __name__ == "__main__":
    main()
