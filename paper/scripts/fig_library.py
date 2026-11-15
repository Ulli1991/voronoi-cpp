#!/usr/bin/env python3
"""Figure: the like-for-like library benchmark (docs/library_benchmark.md) at 128³, from its raw records
(bench/libraries/results/*.jsonl, published.csv) through scripts/library_benchmark.py's readers; rows as its figure."""
import os
import sys

import numpy as np

from paperfig import DOUBLE, LW, REPO, figstyle, figure, save

sys.path.insert(0, os.path.join(REPO, "scripts"))
import library_benchmark as lb  # noqa: E402

LABELS = ["jittered", "blue\nnoise", "uniform", "clus-\ntered", "halo"]
COLOURS = ["#000000", "#555555", "#999999", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#1f77b4",
           "#17becf", "#e377c2"]
MARKERS = "osD^vP*Xh<>p"
SHORT = {"voronoi-cpp, warp rebuild, host to host": "voronoi-cpp, host to host",
         "voronoi-cpp, host to host, volumes + centroids out": "voronoi-cpp, host to host",
         "Paragram (float) + faces (ours)": "Paragram (float) + faces",
         "gDel3D + faces (ours)": "gDel3D + faces", "votess + faces (ours)": "votess + faces",
         "CGAL parallel + faces (ours)": "CGAL par. + faces", "geogram PDEL + faces (ours)": "geogram PDEL + faces",
         "geogram periodic + cells": "geogram periodic", "CGAL periodic + faces, 1 core": "CGAL, 1 core",
         "TetGen + faces, 1 core": "TetGen, 1 core", "Qhull + faces, 1 core": "Qhull, 1 core",
         "voronoi-cpp, warp rebuild": "voronoi-cpp, warp", "voro++ multithreaded": "voro++ mt",
         "peclet-voro (no centroids)": "peclet-voro"}


def panel(ax, summary, pub, rows, tag, text):
    xs = np.arange(len(lb.INPUTS))
    for j, (lib, variant, device, threads, label) in enumerate(rows):
        got = lb.curve(summary, lib, variant, device, threads, lb.SIZES[0])
        pts = [(x, got[i]) for x, i in zip(xs, lb.INPUTS) if i in got]
        if pts:
            style = "-" if device == "A100" else ("--" if threads > 1 else ":")
            ax.plot([x for x, _ in pts], [s["median"] for _, s in pts], color=COLOURS[j], marker=MARKERS[j], lw=LW,
                    ls=style, ms=7, label=SHORT.get(label, label))
    k = 0
    for code, n, dist, where in lb.FIGURE_PUBLISHED:
        for p in pub if where == tag else []:
            if p["code"].startswith(code) and p["n"] == n and p["distribution"].startswith(dist):
                name = p["code"].split(" (")[0].replace(" 2021", "").replace(" 2018", "").replace(" et al.", "")
                dist = p["distribution"].split(" (")[0].replace("white noise", "white").replace("blue noise", "blue")
                hw = lb.short_hardware(p["hardware"]).replace("2x Xeon E5-2650L v4", "2x Xeon")
                ax.plot([len(xs) + 0.25 * k], [float(p["rate_Mcells_s"])], marker="o^sDv"[k], mfc="none", mec="0.2",
                        mew=1.6, ms=9, ls="none", label=f"{name}{', ' + dist if 'Basselin' in name else ''} (publ.)")
                k += 1
                break
    ax.set_xticks(list(xs) + ([len(xs) + 0.25 * (k - 1) / 2] if k else []))
    ax.set_xticklabels(LABELS + (["pub-\nlished"] if k else []))
    ax.set_xlim(-0.4, len(xs) + (0.25 * k if k else -0.6))
    ax.set_yscale("log")
    ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top")
    figstyle.decorate(ax)
    ax.minorticks_off()


def main():
    records, _ = lb.load_records()
    summary = lb.summarise([r for r in records if r["library"] != "voronoi-cpp" or r["version"] == lb.OURS])
    pub = lb.published()
    fig, axes = figure(DOUBLE, 150, nrows=2, ncols=2)
    (a, b), (c, d) = axes
    panel(a, summary, pub, lb.FIGURE_GPU_FULL, "gpu_full", "(a) one A100, full cells")
    panel(b, summary, pub, lb.FIGURE_GPU_VOLUMES, "gpu_volumes", "(b) one A100, volumes and barycentres")
    panel(c, summary, pub, lb.FIGURE_CPU_FULL, "cpu", "(c) one CPU node, full cells")
    for ax in (a, b):
        ax.set_ylim(0.1, 3e4 if ax is a else 2e5)
        ax.legend(loc="upper right", fontsize=11, bbox_to_anchor=(1.0, 0.91), handlelength=1.6)
    c.set_ylim(0.008, 3e3)
    c.legend(loc="upper right", fontsize=11, ncol=2, columnspacing=0.6, handlelength=1.6, bbox_to_anchor=(1.0, 0.91))
    for ax in (a, c):
        ax.set_ylabel("build, M cells/s")
    for k, (lib, variant, device, threads, label) in enumerate(lb.FIGURE_STEP):
        got = lb.curve(summary, lib, variant, device, threads, lb.SIZES[0], phase="step").get("jittered")
        if got:
            d.bar(k, got["median"], color=COLOURS[[0, 3, 0, 3, 5, 5][k]], hatch="" if k < 2 else "//",
                  alpha=1 if device == "A100" else 0.55, width=0.7)
            d.text(k, got["median"] * 1.3, f"{got['median']:.3g}", ha="center", va="bottom", fontsize=11)
    d.set_yscale("log")
    d.set_ylim(0.003, 3e3)
    d.set_xticks(range(len(lb.FIGURE_STEP)))
    d.set_xticklabels(["ours\nA100", "peclet\nA100", "ours\n72 c.", "peclet\n72 c.", "CGAL\nrebuild", "CGAL\nmove"])
    d.set_ylabel("moving step, M cells/s")
    d.text(0.03, 0.97, "(d) moving mesh, 0.03 spacings per step,\njittered, full cells", transform=d.transAxes,
           va="top")
    figstyle.decorate(d)
    d.minorticks_off()
    fig.tight_layout()
    save(fig, "library_comparison", DOUBLE)


if __name__ == "__main__":
    main()
