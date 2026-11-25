#!/usr/bin/env python3
"""Figure library_stages: where each code's build time goes, ns per cell at 128³ on jittered and white-noise input, from
the staged runs of the library benchmark (paper/data/raw/libstages-a100-32368965.jsonl, one A100;
libstages-cpu-32368966.jsonl, one 72-core node; scripts/library_stages.py on branch library-stages). Each code's
stages are markers on one log axis (a stacked bar would mislead on it); the bar is the plain runs' total.
Stages: neighbour search, cell kernel (with relaunches), exact Delaunay start, faces, copies host <-> device, and
the rest (preparation, padding, gaps between laps)."""
import json
import os
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt

from paperfig import DATA, DOUBLE, figstyle, figure, save

N = 2097152
INPUTS = [("jittered", "jittered"), ("uniform", "white noise")]
# (library prefix, variant prefix, label); the kNN codes run their README setting per input (grid / white noise).
GPU = [
    ("voronoi-cpp", "warp clipping rebuild (cell list", "voronoi-cpp, rebuild"),
    ("voronoi-cpp", "warp clipping rebuild, host to host (copy", "voronoi-cpp, rebuild, host to host"),
    ("voronoi-cpp", "exact start", "voronoi-cpp, exact start"),
    ("Basselin", "VoroAlgo", "Basselin et al."),
    ("VoroGPU", "kNN", "Ray et al. (float)"),
    ("Liu", "tet-centric", "Liu et al. (float)"),
    ("peclet-voro", "cold build", "peclet-voro"),
    ("Paragram", "voronoi_diagram + faces", "Paragram + faces"),
    ("votess", "tesellate + faces", "votess + faces"),
    ("gDel3D", "Delaunay + faces", "gDel3D + faces"),
]
CPU = [
    ("voronoi-cpp", "warp clipping rebuild (cell list", "voronoi-cpp, rebuild"),
    ("voronoi-cpp", "exact start", "voronoi-cpp, exact start"),
    ("voro++", "multithreaded", "voro++"),
    ("CGAL", "Delaunay + faces", "CGAL + faces"),
    ("geogram", "PeriodicDelaunay3d + Voronoi", "geogram periodic"),
]
GROUPS = [  # (name, stages of the record, marker, colour)
    ("neighbour search", ("search",), "o", "#1f77b4"),
    ("cell kernel (+ relaunch)", ("cells", "relaunch"), "s", "#d62728"),
    ("exact Delaunay start", ("exact",), "D", "#9467bd"),
    ("faces", ("faces",), "^", "#2ca02c"),
    ("copies host ↔ device", ("transfer_in", "transfer_out"), "v", "#ff7f0e"),
    ("rest", ("prepare", "other"), "x", "#7f7f7f"),
]


def load():
    plain, staged = defaultdict(list), defaultdict(list)
    for name in ("libstages-a100-32368965.jsonl", "libstages-cpu-32368966.jsonl"):
        for line in open(os.path.join(DATA, "raw", name)):
            if line.startswith("{"):
                r = json.loads(line)
                if "seconds" in r and r.get("n") == N:
                    key = (r["library"], r["variant"], r["input"], "A100" in r["device"])
                    (staged if r.get("staged") else plain)[key].append(r)
    return plain, staged


def find(plain, staged, lib, var, inp, gpu):
    for key, runs in staged.items():
        if key[0].startswith(lib) and key[1].startswith(var) and key[2] == inp and key[3] == gpu:
            total = statistics.median(r["seconds"] for r in plain.get(key, runs))
            stages = {s: statistics.median(r["stage_s"][s] for r in runs) for s in runs[0]["stage_s"]}
            whole = statistics.median(r["seconds"] for r in runs)
            ok = all(-1e-6 <= v <= 1.01 * whole for v in stages.values())
            return total * 1e9 / N, ({g: sum(stages[s] for s in st) * 1e9 / N for g, st, _, _ in GROUPS} if ok
                                     else None)
    return None, None


def main():
    plain, staged = load()
    fig, axes = figure(DOUBLE, 110, nrows=2, ncols=2, sharex=True,
                       gridspec_kw=dict(height_ratios=[len(GPU), len(CPU)]))
    for row, (codes, device) in enumerate(((GPU, "one A100"), (CPU, "72 cores"))):
        for col, (inp, name) in enumerate(INPUTS):
            ax = axes[row, col]
            for y, (lib, var, label) in enumerate(codes):
                total, parts = find(plain, staged, lib, var, inp, row == 0)
                if total is None:
                    ax.text(0.15, y, "not run", va="center", fontsize=10, color="0.4")
                    continue
                ax.barh(y, total, height=0.62, color="0.9", edgecolor="0.6", lw=0.5, zorder=1)
                if parts is None:
                    ax.text(0.13, y, "stages not shown: own timers inconsistent", va="center", fontsize=10,
                            color="0.3", zorder=2)
                    continue
                for k, (g, _, marker, colour) in enumerate(GROUPS):
                    if parts[g] >= 0.05:  # search and copies above, cell kernel and rest below the centre, so equal times both show
                        ax.plot(parts[g], y + (-0.17, 0.17, 0, 0, -0.17, 0.17)[k], marker, color=colour, ms=7, mew=1.6 if marker == "x" else 0.5,
                                mec=colour if marker == "x" else "white", zorder=3)
            ax.set_yticks(range(len(codes)), [c[2] for c in codes] if col == 0 else [])
            ax.set_ylim(len(codes) - 0.5, -0.5)
            ax.set_xscale("log")
            ax.set_xlim(0.1, 2e5)  # room on the right of the A100 panels for the legend
            ax.set_xticks([0.1, 1, 10, 100, 1e3, 1e4, 1e5], ["0.1", "1", "10", "100", "10³", "10⁴", "10⁵"])
            figstyle.decorate(ax)
            ax.tick_params(axis="y", which="both", left=False, right=False)
            ax.grid(False, axis="y")
            ax.set_title(f"{name}, {device}", fontsize=12)
            if row == 1:
                ax.set_xlabel("ns per cell, 128³")
    handles = [plt.Line2D([], [], marker=m, color=c, ls="none", ms=7, mec=c if m == "x" else "white",
                          mew=1.6 if m == "x" else 0.5) for _, _, m, c in GROUPS]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="0.9", ec="0.6", lw=0.5))
    # One shared legend inside the panel with the most room: right of the bars in jittered, one A100.
    axes[0, 0].legend(handles, [g[0] for g in GROUPS] + ["total (plain runs)"], loc="upper right", fontsize=10)
    fig.subplots_adjust(left=0.24, right=0.99, top=0.96, bottom=0.08, hspace=0.18, wspace=0.05)
    save(fig, "library_stages", DOUBLE)


if __name__ == "__main__":
    main()
