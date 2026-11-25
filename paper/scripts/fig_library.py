#!/usr/bin/env python3
"""Library benchmark (docs/library_benchmark.md) at 128³ from its raw records (bench/libraries/results/*.jsonl,
published.csv) through scripts/library_benchmark.py's readers. library_headline: full cells on one A100, one marker per
library, ours highlighted; measured rows only (the published rates are in the captions, captions.md); library_more: volumes only, one CPU node, the moving step."""
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
    ("voronoi-cpp", "warp clipping rebuild (cell", "A100", 1, "Kikkou (this work), rebuild"),
    ("voronoi-cpp", "exact start", "A100", 1, "Kikkou (this work), exact start"),
    ("peclet-voro", "cold build", "A100", 1, "peclet-voro"),
    ("gDel3D", "Delaunay + faces", "A100", 1, "gDel3D + faces"),
    ("Paragram", "voronoi_diagram + faces", "A100", 1, "Paragram + faces"),
    ("votess", "tesellate + faces", "A100", 1, "votess + faces"),
]
SHORT = {"voronoi-cpp, warp rebuild, host to host": "Kikkou (this work), host to host",
         "voronoi-cpp, host to host, volumes + centroids out": "Kikkou (this work), host to host",
         "CGAL parallel + faces (ours)": "CGAL + faces", "geogram PDEL + faces (ours)": "geogram PDEL + faces",
         "geogram periodic + cells": "geogram periodic", "CGAL periodic + faces, 1 core": "CGAL, 1 core",
         "TetGen + faces, 1 core": "TetGen, 1 core", "Qhull + faces, 1 core": "Qhull, 1 core",
         "voronoi-cpp, warp rebuild": "Kikkou (this work)", "voro++ multithreaded": "voro++",
         "votess + faces (ours)": "votess + faces", "peclet-voro (no centroids)": "peclet-voro",
         "Basselin et al. 2021": "Basselin et al.", "Liu et al. 2022 (float)": "Liu et al. (float)",
         "Ray et al. 2018 (float)": "Ray et al. (float)",
         "voronoi-cpp, host to host, volumes + centroids out": "Kikkou (this work), full cells, host to host"}
SHORT["voronoi-cpp, warp rebuild"] = "Kikkou (this work)"
SHORT["voronoi-cpp, host to host, volumes + centroids out"] = "Kikkou (this work), full cells, host to host"


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
    """One curve per row over the five benchmark inputs; returns {label: its line}."""
    xs = np.arange(len(lb.INPUTS))
    k = 0
    drawn = {}
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
        drawn[label], = ax.plot(*zip(*pts), color=colour, marker=marker, ms=9 if ours and big else 7,
                                lw=LW * (1.4 if ours else 1), ls=ls if not ours or "start" not in label else "--",
                                label=SHORT.get(label, label), zorder=4 if ours else 3)
    return drawn


def axes_inputs(ax, short=False):
    ax.set_xticks(range(len(lb.INPUTS)), ["jittered", "blue noise", "white noise", "clustered", "halo"] if short
                  else LABELS)
    ax.set_xlim(-0.4, len(lb.INPUTS) - 0.6)
    if short:
        ax.tick_params(axis="x", labelrotation=35)
        for t in ax.get_xticklabels():
            t.set_ha("right")
            t.set_rotation_mode("anchor")
    ax.set_yscale("log")
    figstyle.decorate(ax)
    ax.minorticks_off()
    ax.tick_params(axis="x", which="both", top=False, bottom=False)


APPLICATION = [("cosmicweb", "cosmic\nweb"), ("fccsolid", "fcc\nsolid"), ("surface", "sur-\nface")]
HEADLINE_WIDTH = 130.0  # mm: eight inputs
# Panel (a) of library_more as it was first drawn: our full cells with the volumes and centroids copied out, then the
# three volumes-only codes (scripts/library_benchmark.py's FIGURE_GPU_VOLUMES has other rows of ours since).
VOLUMES = [("voronoi-cpp", "warp clipping rebuild, host to host, volumes and centroids copied out", "A100", 1,
            "Kikkou (this work), full cells, host to host")] + list(lb.FIGURE_GPU_VOLUMES[2:])


def application_inputs(ax, drawn, rows, ticks):
    """Continues each curve of `drawn` ({label: line}, drawn from `rows`) over the three application inputs: rates from
    the library benchmark's records of those inputs (ours: the application jobs' commit), an open marker where
    fig_application finds that code's volumes wrong or cells missing, a break where a code has no rate there."""
    import fig_application as fa

    records, _ = lb.load_records()
    size = {kind: n for kind, n, _ in fa.PANELS}
    summary = lb.summarise([r for r in records if r["input"] in size and (
        r["library"] != "voronoi-cpp" or r["version"] == lb.OURS_APPLICATION)])
    state = {(r["label"], r["kind"]): r for r in fa.rows() if r["state"] != "crashed"}
    known = {(lib, device, threads, variant): label for label, _, lib, variant, device, threads, _, _ in fa.CODES}
    first = len(lb.INPUTS)
    for lib, variant, device, threads, label in rows:
        line = drawn.get(label)
        if line is None:
            continue
        checked = next((name for (l, d, t, v), name in known.items()
                        if (l, d, t) == (lib, device, threads) and (variant.startswith(v) or v.startswith(variant))), None)
        if lib == "voronoi-cpp" and device == "A100":
            checked = "Kikkou (this work), rebuild"  # every A100 row of ours is the warp rebuild
        last = (line.get_xdata()[-1], line.get_ydata()[-1])
        run = [last] if last[0] == first - 1 else []
        for k, (kind, _) in enumerate(APPLICATION):
            got = lb.curve(summary, lib, variant, device, threads, size[kind]).get(kind)
            if got is None:
                run = []
                continue
            run.append((first + k, got["median"]))
            if len(run) > 1:
                ax.plot(*zip(*run[-2:]), color=line.get_color(), lw=line.get_linewidth(), ls=line.get_linestyle(),
                        zorder=line.get_zorder())
            r = state.get((checked, kind))
            incomplete = r is not None and (r["state"] == "wrong" or r["missing"] > 0)
            ax.plot(first + k, got["median"], line.get_marker(), color=line.get_color(), ms=line.get_markersize(),
                    mfc="white" if incomplete else line.get_color(), mew=1.8 if incomplete else 0,
                    zorder=line.get_zorder() + 1)
    ax.set_xticks(range(first + len(APPLICATION)), ticks)
    ax.set_xlim(-0.4, first + len(APPLICATION) - 0.6)


def headline(summary, pub):
    """Full cells on one A100 over the five benchmark inputs and the three application inputs."""
    fig, ax = figure(HEADLINE_WIDTH, 85)
    drawn = curves(ax, summary, HEADLINE, big=True)
    axes_inputs(ax)  # measured rows only; the published rates are in captions.md
    application_inputs(ax, drawn, HEADLINE, LABELS + [short for _, short in APPLICATION])
    ax.set_ylim(0.01, 1000)
    ax.set_ylabel("full cells, M cells/s")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(ax.plot([], [], "o", color="0.3", mfc="white", mew=1.8, ms=8)[0])
    labels.append("cells missing or wrong")
    ax.legend(handles, labels, loc="upper right", fontsize=11, ncol=3, columnspacing=0.6, handlelength=1.5)
    fig.tight_layout()
    save(fig, "library_headline", HEADLINE_WIDTH)


GAPS = ("gaps-chain-a100-32368078.jsonl", "gaps-cpu-pair-32367276.jsonl")  # branch close-gaps, voronoi-cpp 77055a4


def gaps():
    """Summary of the close-gaps jobs: our volumes-only path with the three volumes-only codes in one A100 job
    (blue noise only), and our CPU path with voro++ multithreaded on one node."""
    rs = []
    for name in GAPS:
        for line in open(os.path.join(DATA, "raw", name)):
            if line.startswith("{") and '"phase"' in line:
                r = json.loads(line)
                r["file"] = name
                rs.append(r)
    return lb.summarise(rs)


def more(summary, pub):
    g = gaps()
    fig, (a, b) = figure(DOUBLE, 80, ncols=2)
    names = ["jittered", "blue noise", "white noise", "clustered", "halo", "cosmic web", "fcc solid", "surface"]
    drawn = curves(a, summary, VOLUMES)
    axes_inputs(a, short=True)
    application_inputs(a, drawn, VOLUMES, names)
    # Our volumes-only path, host to host like the three codes, from the job that ran them on blue noise.
    vol = next(v for k, v in g.items() if k[0][1] == "volumes and centroids only, host to host"  # pageable buffers
               and k[2] == "bluenoise" and k[3] == lb.SIZES[0])
    others = [lb.curve(g, lib, var, "A100", 1, lb.SIZES[0])["bluenoise"]["median"]
              for lib, var in (("Basselin", ""), ("VoroGPU", ""), ("Liu et al.", "tet-centric"))]
    a.plot(1, vol["median"], "*", color=OURS, ms=15, mec="white", mew=0.5, zorder=6,
           label="Kikkou (this work), volumes only, host to host")
    a.text(1.2, vol["median"] * 1.5, "{:.1f}× Basselin\n{:.1f}× Ray\n{:.1f}× Liu (same job)".format(
        *(vol["median"] / o for o in others)), fontsize=10, va="center", ha="left")
    a.set_ylim(0.03, 5000)
    a.set_ylabel("M cells/s")
    a.text(0.03, 0.97, "(a) A100, volumes", transform=a.transAxes, va="top")
    a.legend(loc="upper right", fontsize=10, handlelength=1.4, bbox_to_anchor=(1.0, 0.9))
    # (b): ours and voro++ multithreaded from the same node and job; the other codes from the benchmark's CPU jobs.
    merged = {k: v for k, v in summary.items() if not (k[0][0] in ("voronoi-cpp", "voro++") and k[0][3] == 72)}
    merged.update({k: v for k, v in g.items() if k[0][3] == 72})
    cpu = [("voronoi-cpp", "warp clipping rebuild (cell", "2x Xeon", 72, "voronoi-cpp, warp rebuild")] + \
        [r for r in lb.FIGURE_CPU_FULL[1:7]]
    drawn = curves(b, merged, cpu)
    summary = merged
    axes_inputs(b, short=True)
    application_inputs(b, {SHORT.get(k, k) if k not in drawn else k: v for k, v in drawn.items()}, cpu, names)
    b.set_ylim(0.05, 5000)
    b.text(0.03, 0.97, "(b) 72 cores, full cells", transform=b.transAxes, va="top")
    b.legend(loc="upper right", fontsize=11, handlelength=1.4, bbox_to_anchor=(1.0, 0.9))
    fig.tight_layout()
    save(fig, "library_more", DOUBLE)
    step_codes(summary)


def step_codes(summary):
    """The moving step of the codes that have one, jittered input at 128³ (a figure of the moving-mesh part)."""
    fig, c = figure(SINGLE, 70)
    names = ["Kikkou\nGPU", "peclet\nGPU", "Kikkou\nCPU", "peclet\nCPU", "CGAL\n1 core"]
    for k, row in enumerate(lb.FIGURE_STEP[:5]):
        got = lb.curve(summary, *row[:4], lb.SIZES[0], phase="step").get("jittered")
        if got:
            c.bar(k, got["median"], color=OURS if row[0] == "voronoi-cpp" else "#ff7f0e" if k < 4 else "#2ca02c",
                  alpha=1 if row[2] == "A100" else 0.55, width=0.7)
            c.text(k, got["median"] * 1.3, f"{got['median']:.3g}", ha="center", va="bottom", fontsize=11)
    c.set_yscale("log")
    c.set_ylim(0.01, 1000)
    c.set_xlim(-0.6, 4.6)
    c.set_xticks(range(5), names, fontsize=11)
    c.set_ylabel("mesh update, M cells/s")
    figstyle.decorate(c)
    c.minorticks_off()
    c.tick_params(axis="x", which="both", top=False, bottom=False)
    fig.tight_layout()
    save(fig, "step_codes", SINGLE)


def main():
    summary, pub = load()
    headline(summary, pub)
    more(summary, pub)


if __name__ == "__main__":
    main()
