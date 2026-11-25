#!/usr/bin/env python3
"""Figure quality: speed against quality, one point per library and input at 128³, from the library benchmark's own
runs: build rate (bench/libraries/results), relative volume error of 1000 sampled cells against the exact reference
(paper/data/oracle_errors.csv), cells missing, and the total sampled volume against the box (accuracy-*.jsonl)."""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from common import oracle_errors
from paperfig import DOUBLE, MARKER, REPO, figstyle, figure, save

sys.path.insert(0, os.path.join(REPO, "scripts"))
import library_benchmark as lb  # noqa: E402

INPUTS = ["jittered", "bluenoise", "uniform", "clustered", "halo"]
README = {"basselin": {"bluenoise": "blue", "jittered": "grid"}, "vorogpu": {"bluenoise": "k35", "jittered": "k90"},
          "liu": {"bluenoise": "p2", "jittered": "p1"}}
WHITE = {"basselin": "white", "vorogpu": "k190", "liu": "p0"}
# (label, volumes prefix with {s} for the README setting, library, variant prefix, device, threads, colour)
CODES = [
    ("Kikkou (this work), rebuild", "voronoi-cpp-warp-a100", "voronoi-cpp", "warp clipping rebuild (cell", "A100", 1, "#000000"),
    ("Kikkou (this work), exact start", "voronoi-cpp-exact-a100", "voronoi-cpp", "exact start", "A100", 1, "#555555"),
    ("peclet-voro", "peclet-voro-a100", "peclet-voro", "cold build", "A100", 1, "#ff7f0e"),
    ("gDel3D + faces", "gdel3d-a100", "gDel3D", "Delaunay + faces", "A100", 1, "#2ca02c"),
    ("Paragram + faces", "paragram-a100", "Paragram", "voronoi_diagram + faces", "A100", 1, "#d62728"),
    ("votess + faces", "votess-a100", "votess", "tesellate + faces", "A100", 1, "#9467bd"),
    ("Basselin et al.", "basselin-{s}-a100", "Basselin", "", "A100", 1, "#8c564b"),
    ("Ray et al. (float)", "vorogpu-{s}-a100", "VoroGPU", "", "A100", 1, "#e377c2"),
    ("Liu et al. (float)", "voronoi-gpu-liu-tet-{s}-a100", "Liu et al.", "tet-centric", "A100", 1, "#bcbd22"),
    ("voro++ (72 cores)", "voropp-mt-cpu", "voro++", "multithreaded", "2x Xeon", 72, "#1f77b4"),
    ("CGAL + faces (72 cores)", "cgal-dt3-par-cpu", "CGAL Delaunay_triangulation_3", "Delaunay + faces", "2x Xeon", 72,
     "#17becf"),
    ("geogram (72 cores)", "geogram-pdel-cpu", "geogram", "PeriodicDelaunay3d + Voronoi cells", "2x Xeon", 72,
     "#e6ab02"),
]
FLOOR = 1e-17


def totals():
    """{volumes file name: sum of volumes - box} from the benchmark's accuracy records."""
    out = {}
    for path in sorted(glob.glob(os.path.join(REPO, "bench", "libraries", "results", "accuracy-*.jsonl"))):
        for line in open(path):
            if line.startswith("{"):
                r = json.loads(line.replace(":null", ":NaN"))
                out[os.path.basename(r["volumes"])] = r["sum_minus_box"]
    return out


def rows():
    from fig_library import load
    summary = load()[0]
    err, tot = oracle_errors(), totals()
    out = []
    for label, prefix, lib, variant, device, threads, colour in CODES:
        speed = lb.curve(summary, lib, variant, device, threads, lb.SIZES[0])
        key = next((k for k in README if k in prefix), None)
        for kind in INPUTS:
            name = prefix.format(s=README[key].get(kind, WHITE[key]) if key else "") + f"_{kind}_2097152.f64"
            if kind not in speed or name not in err:
                continue
            n, missing, e = err[name]
            finite = e[np.isfinite(e)]
            out.append(dict(label=label, colour=colour, kind=kind, rate=speed[kind]["median"],
                            median=max(np.median(finite), FLOOR), max=max(finite.max(), FLOOR), missing=missing / n,
                            total=abs(tot.get(name, np.nan))))
    return out


def main():
    data = rows()
    fig, _ = figure(DOUBLE, 150, nrows=2, ncols=2)
    fig.clf()
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.32, wspace=0.28)
    a = fig.add_subplot(grid[0, :])
    b, c = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    axes = (a, b, c)
    for r in data:
        m = MARKER[r["kind"]]
        a.plot([r["rate"]] * 2, [r["median"], r["max"]], color=r["colour"], lw=1.0, alpha=0.6)
        a.plot(r["rate"], r["median"], m, color=r["colour"], ms=7, mec="white", mew=0.5)
        b.plot(r["rate"], max(r["missing"], 3e-8), m, color=r["colour"], ms=7, mec="white", mew=0.5)
        if np.isfinite(r["total"]):
            c.plot(r["rate"], max(r["total"], 1e-18), m, color=r["colour"], ms=7, mec="white", mew=0.5)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.05, 100)
        ax.set_xlabel("build rate, M cells/s")
        figstyle.decorate(ax)
    a.set_ylim(3e-17, 1e3)
    a.set_ylabel("|V − V$_\\mathrm{exact}$| / V$_\\mathrm{exact}$")
    a.text(0.01, 0.97, "(a)", transform=a.transAxes, va="top")
    b.set_ylim(1e-8, 2)
    b.axhspan(1e-8, 1e-7, color="0.92", zorder=0)
    b.text(0.06, 1.5e-7, "band: no cell missing", fontsize=11, va="bottom", color="0.3")
    b.set_ylabel("fraction of cells missing")
    b.text(0.03, 0.97, "(b) completeness", transform=b.transAxes, va="top")
    c.set_ylim(1e-18, 1e3)
    c.set_ylabel("|Σ V − box|")
    c.text(0.03, 0.97, "(c) total sampled volume", transform=c.transAxes, va="top")
    seen = {}
    for r in data:
        seen.setdefault(r["label"], r["colour"])
    a.legend([plt.Line2D([], [], color=v, marker="o", ls="none", ms=7) for v in seen.values()], list(seen),
             loc="upper center", fontsize=11, ncol=4)
    b.legend([plt.Line2D([], [], color="0.4", marker=MARKER[k], ls="none", ms=7) for k in INPUTS],
             ["jittered", "blue noise", "white noise", "clustered", "halo"], loc="upper right", fontsize=11,
             bbox_to_anchor=(0.98, 0.9), title="input", title_fontsize=11)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.08)
    save(fig, "quality", DOUBLE)


if __name__ == "__main__":
    main()
