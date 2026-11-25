#!/usr/bin/env python3
"""Figure correctness: build speed against correctness for every library at 128³, one point per code and input.
(a) relative volume error of 1000 sampled cells against the exact reference (marker: median, line: up to the largest) on
the five synthetic inputs; (b)-(e) the fraction of cells missing on the synthetic inputs and on the three application
inputs, open markers where returned volumes are wrong. Data and rules: fig_quality.rows(), fig_application.rows()."""
import matplotlib.pyplot as plt

import fig_application as app
import fig_quality as quality
from paperfig import DOUBLE, LABEL, MARKER, figstyle, figure, save

CPU_TWIN = {"votess + faces (72 cores)": "#c5b0d5"}  # the CPU run of a code that also has an A100 run: a lighter shade
BAND = (1e-8, 2.5e-7)  # codes with no cell missing, spread by app.height() only to tell them apart


def colour(r):
    return CPU_TWIN.get(r["label"], r["colour"])


def main():
    synthetic, application = quality.rows(), app.rows()
    fig, _ = figure(DOUBLE, 150, nrows=2, ncols=4)
    fig.clf()
    grid = fig.add_gridspec(2, 4, height_ratios=[1.05, 1], hspace=0.30, wspace=0.07, left=0.085, right=0.99, top=0.985,
                            bottom=0.075)
    a = fig.add_subplot(grid[0, :3])
    key = fig.add_subplot(grid[0, 3])
    low = [fig.add_subplot(grid[1, 0])]
    low += [fig.add_subplot(grid[1, k], sharey=low[0]) for k in (1, 2, 3)]

    for r in synthetic:
        m = MARKER[r["kind"]]
        a.plot([r["rate"]] * 2, [r["median"], r["max"]], color=r["colour"], lw=1.0, alpha=0.6)
        a.plot(r["rate"], r["median"], m, color=r["colour"], ms=7, mec="white", mew=0.5)
        y = r["missing"] if r["missing"] > 0 else 1.7e-8 * 1.25 ** app.LEVEL[r["label"]]
        low[0].plot(r["rate"], y, m, color=r["colour"], ms=7, mec="white", mew=0.5)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(0.05, 100)
    a.set_ylim(3e-17, 1e0)
    a.set_xlabel("build rate, M cells/s")
    a.set_ylabel("|V − V$_\\mathrm{exact}$| / V$_\\mathrm{exact}$")
    a.text(0.012, 0.965, "(a)", transform=a.transAxes, va="top")
    figstyle.decorate(a)

    titles = ["(b) synthetic inputs"] + [f"({'cde'[k]}) {LABEL[kind]}" for k, (kind, _, _) in enumerate(app.PANELS)]
    for ax, (kind, _, _) in zip(low[1:], app.PANELS):
        for r in application:
            if r["kind"] != kind or r["state"] == "crashed":
                continue
            hollow = r["state"] == "wrong"  # open and on top, so a wrong code is never hidden behind a right one
            ax.plot(r["rate"], app.height(r), "o", color=colour(r), ms=9 if hollow else 8,
                    mfc="none" if hollow else colour(r), mec=colour(r) if hollow else "white",
                    mew=1.8 if hollow else 0.5, zorder=5 if hollow else 4)
    for ax, title in zip(low, titles):
        ax.axhspan(*BAND, color="0.92", zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.01, 100)
        ax.set_xticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
        ax.set_ylim(1e-8, 5)
        ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
        ax.set_xlabel("build rate, M cells/s")
        ax.set_title(title, fontsize=12, loc="left", pad=4)
        figstyle.decorate(ax)
    for ax in low[1:]:
        ax.tick_params(labelleft=False)
        ax.set_xticks([0.01, 0.1, 1, 10], ["", "0.1", "1", "10"])  # the first label would meet the neighbour's last
    low[0].set_ylabel("fraction of cells missing")

    # the key, in the cell right of (a): one entry per code, then the input markers of (a) and (b), then the open marker
    key.axis("off")
    seen = {}
    for r in synthetic + application:
        seen.setdefault(r["label"], colour(r))
    handles = [plt.Line2D([], [], color=c, marker="o", ls="none", ms=7, mec="white", mew=0.5) for c in seen.values()]
    names = list(seen)
    for kind in quality.INPUTS:
        handles.append(plt.Line2D([], [], color="0.4", marker=MARKER[kind], ls="none", ms=7, mec="white", mew=0.5))
        names.append(LABEL[kind])
    handles.append(plt.Line2D([], [], color="0.3", marker="o", ls="none", ms=8, mfc="none", mew=1.8))
    names.append("wrong volumes")
    key.legend(handles, names, loc="center left", bbox_to_anchor=(0.02, 0.5), fontsize=10.5, frameon=False,
               labelspacing=0.32, handletextpad=0.3)
    save(fig, "correctness", DOUBLE)


if __name__ == "__main__":
    main()
