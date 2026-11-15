#!/usr/bin/env python3
"""Library benchmark (docs/library_benchmark.md) at 128³ from its raw records (bench/libraries/results/*.jsonl,
published.csv) through scripts/library_benchmark.py's readers. library_headline: full cells on one A100, one marker per
library, ours highlighted, published apart; library_more: volumes only, one CPU node, the moving step."""
import glob
import json
import os
import sys

import numpy as np

from paperfig import DATA, DOUBLE, LW, REPO, SINGLE, figstyle, figure, save

sys.path.insert(0, os.path.join(REPO, "scripts"))
import library_benchmark as lb  # noqa: E402

LABELS = ["jittered", "blue\nnoise", "white\nnoise", "clus-\ntered", "halo"]
OURS = "#000000"
OTHERS = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#1f77b4", "#17becf", "#e377c2", "#7f7f7f", "#bcbd22"]
MARKERS = "^vPXsDh<>p*"
HEADLINE = [  # (library, variant prefix, device, threads, label)
    ("voronoi-cpp", "warp clipping rebuild (cell", "A100", 1, "ours, rebuild"),
    ("voronoi-cpp", "exact start", "A100", 1, "ours, exact start"),
    ("peclet-voro", "cold build", "A100", 1, "peclet-voro"),
    ("gDel3D", "Delaunay + faces", "A100", 1, "gDel3D + faces"),
    ("Paragram", "voronoi_diagram + faces", "A100", 1, "Paragram + faces"),
    ("votess", "tesellate + faces", "A100", 1, "votess + faces"),
]
SHORT = {"voronoi-cpp, warp rebuild, host to host": "voronoi-cpp, host to host",
         "voronoi-cpp, host to host, volumes + centroids out": "voronoi-cpp, host to host",
         "CGAL parallel + faces (ours)": "CGAL par. + faces", "geogram PDEL + faces (ours)": "geogram PDEL + faces",
         "geogram periodic + cells": "geogram periodic", "CGAL periodic + faces, 1 core": "CGAL, 1 core",
         "TetGen + faces, 1 core": "TetGen, 1 core", "Qhull + faces, 1 core": "Qhull, 1 core",
         "voronoi-cpp, warp rebuild": "voronoi-cpp", "voro++ multithreaded": "voro++ mt",
         "votess + faces (ours)": "votess + faces", "peclet-voro (no centroids)": "peclet-voro",
         "Basselin et al. 2021": "Basselin et al.", "Liu et al. 2022 (float)": "Liu et al. (float)",
         "Ray et al. 2018 (float)": "Ray et al. (float)",
         "voronoi-cpp, host to host, volumes + centroids out": "ours, host to host"}
SHORT["voronoi-cpp, warp rebuild"] = "ours"
SHORT["voronoi-cpp, host to host, volumes + centroids out"] = "ours, host to host"


def load():
    """The benchmark's records; voronoi-cpp's rows from the paper's rerun (paper/data/raw/libbench-vcpp-*.jsonl) when
    it exists, else the benchmark's own (lb.OURS)."""
    records, _ = lb.load_records()
    ours = []
    for path in sorted(glob.glob(os.path.join(DATA, "raw", "libbench-vcpp-*.jsonl"))):
        ours += [json.loads(line) for line in open(path) if line.startswith("{") and '"phase"' in line]
    keep = [r for r in records if r["library"] != "voronoi-cpp" or (not ours and r["version"] == lb.OURS)]
    return lb.summarise(keep + ours), lb.published()


def curves(ax, summary, rows, big=False):
    xs = np.arange(len(lb.INPUTS))
    k = 0
    for lib, variant, device, threads, label in rows:
        got = lb.curve(summary, lib, variant, device, threads, lb.SIZES[0])
        pts = [(x, got[i]["median"]) for x, i in zip(xs, lb.INPUTS) if i in got]
        if not pts:
            continue
        ours = lib == "voronoi-cpp"
        colour = OURS if ours else OTHERS[k % len(OTHERS)]
        marker = ("o" if "start" not in label else "s") if ours else MARKERS[k % len(MARKERS)]
        k += 0 if ours else 1
        ls = "-" if device == "A100" else ("--" if threads > 1 else ":")
        ax.plot(*zip(*pts), color=colour, marker=marker, ms=9 if ours and big else 7, lw=LW * (1.4 if ours else 1),
                ls=ls if not ours or "start" not in label else "--", label=SHORT.get(label, label), zorder=4 if ours else 3)


def published(ax, pub, tag, x0, annotate=False):
    """Published rates as open markers at x0 (+0.3 per marker, or one column with text labels); returns the count."""
    k = 0
    for code, n, dist, where in lb.FIGURE_PUBLISHED:
        for p in pub if where == tag else []:
            if p["code"].startswith(code) and p["n"] == n and p["distribution"].startswith(dist):
                name = p["code"].split(" (")[0].replace(" 2021", "").replace(" 2018", "")
                dist_s = p["distribution"].split(" (")[0].replace(" noise", "")
                label = f"{name}{', ' + dist_s if 'Basselin' in name else ''}"
                rate = float(p["rate_Mcells_s"])
                ax.plot([x0 if annotate else x0 + 0.3 * k], [rate], marker="o^sDv"[k], mfc="white", mec="0.25",
                        mew=1.6, ms=8, ls="none", zorder=5, label=None if annotate else label + " (publ.)")
                if annotate:
                    ax.text(x0 + 0.25, rate, label.replace(" et al.", ""), va="center", fontsize=11, color="0.25")
                k += 1
                break
    return k


def axes_inputs(ax, k, x0, short=False, width=None):
    width = 0.3 * max(k - 1, 0) if width is None else width
    ax.axvspan(x0 - 0.45, x0 + width + 0.45, color="0.93", zorder=0)
    ax.set_xticks(list(range(len(lb.INPUTS))) + ([x0 + width / 2] if k else []),
                  (["jittered", "blue noise", "white noise", "clustered", "halo"] if short else LABELS) + ((["publ."] if short else ["pub-\nlished"]) if k else []))
    ax.set_xlim(-0.4, x0 + width + 0.5 if k else len(lb.INPUTS) - 0.6)
    if short:
        ax.tick_params(axis="x", labelrotation=35)
        for t in ax.get_xticklabels():
            t.set_ha("right")
            t.set_rotation_mode("anchor")
    ax.set_yscale("log")
    figstyle.decorate(ax)
    ax.minorticks_off()
    ax.tick_params(axis="x", which="both", top=False, bottom=False)


def headline(summary, pub):
    fig, ax = figure(SINGLE, 85)
    curves(ax, summary, HEADLINE, big=True)
    x0 = len(lb.INPUTS) + 0.3
    k = published(ax, pub, "gpu_full", x0)
    axes_inputs(ax, k, x0)
    ax.set_ylim(0.1, 1000)
    ax.set_ylabel("full cells, M cells/s")
    ax.legend(loc="upper right", fontsize=11, ncol=2, columnspacing=0.6, handlelength=1.5)
    fig.tight_layout()
    save(fig, "library_headline", SINGLE)


def more(summary, pub):
    fig, axes = figure(DOUBLE, 80, ncols=3, gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    a, b, c = axes
    x0 = len(lb.INPUTS) + 0.3
    curves(a, summary, lb.FIGURE_GPU_VOLUMES)
    axes_inputs(a, published(a, pub, "gpu_volumes", x0, annotate=True), x0, short=True, width=2.7)
    a.set_ylim(0.5, 1000)
    a.set_ylabel("M cells/s")
    a.text(0.03, 0.97, "(a) A100, volumes", transform=a.transAxes, va="top")
    a.legend(loc="upper right", fontsize=11, handlelength=1.4, bbox_to_anchor=(1.0, 0.9))
    curves(b, summary, lb.FIGURE_CPU_FULL[:7])
    axes_inputs(b, published(b, pub, "cpu", x0), x0, short=True)
    b.set_ylim(0.05, 1000)
    b.text(0.03, 0.97, "(b) 72 cores, full cells", transform=b.transAxes, va="top")
    b.legend(loc="upper right", fontsize=11, handlelength=1.4, bbox_to_anchor=(1.0, 0.9))
    names = ["ours\nGPU", "peclet\nGPU", "ours\nCPU", "peclet\nCPU", "CGAL\n1 core"]
    for k, row in enumerate(lb.FIGURE_STEP[:5]):
        got = lb.curve(summary, *row[:4], lb.SIZES[0], phase="step").get("jittered")
        if got:
            c.bar(k, got["median"], color=OURS if row[0] == "voronoi-cpp" else "#ff7f0e" if k < 4 else "#2ca02c",
                  alpha=1 if row[2] == "A100" else 0.55, width=0.7)
            c.text(k, got["median"] * 1.3, f"{got['median']:.3g}", ha="center", va="bottom", fontsize=11)
    c.set_yscale("log")
    c.set_ylim(0.01, 1000)
    c.set_xlim(-0.6, 4.6)
    c.set_xticks(range(5), names)
    c.set_ylabel("moving step, M cells/s")
    c.text(0.96, 0.97, "(c)", transform=c.transAxes, ha="right", va="top")
    figstyle.decorate(c)
    c.minorticks_off()
    c.tick_params(axis="x", which="both", top=False, bottom=False)
    fig.tight_layout()
    save(fig, "library_more", DOUBLE)


def main():
    summary, pub = load()
    headline(summary, pub)
    more(summary, pub)


if __name__ == "__main__":
    main()
