#!/usr/bin/env python3
"""Tables and figure of the library benchmark (docs/library_benchmark.md) from its raw data alone:
the timing records bench/libraries/results/*.jsonl (one JSON line per timed run, written by the drivers through
bench/libraries/libbench.hpp), the oracle checks bench/libraries/results/accuracy-*.jsonl and the published numbers
bench/libraries/published.csv. Writes bench/libraries/results/summary.md and docs/figures/library_benchmark.png.
    python3 scripts/library_benchmark.py
"""
import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import figstyle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "bench", "libraries", "results")
INPUTS = ["jittered", "uniform", "clustered", "halo"]
SIZES = [2097152, 16777216]
SIZE_NAME = {2097152: "128³", 16777216: "256³"}


def load_records():
    """Timing records of every results file; other JSON lines (memory side records) are kept apart."""
    records, extra = [], []
    for path in sorted(glob.glob(os.path.join(RESULTS, "*.jsonl"))):
        if os.path.basename(path).startswith("accuracy"):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                r = json.loads(line)
                r["file"] = os.path.basename(path)
                (records if "phase" in r else extra).append(r)
    return records, extra


def key(r):
    """One row of the tables: library, variant, device and threads."""
    return (r["library"], r["variant"], r["device"], int(r["threads"]))


def summarise(records):
    """Median, min and max rate per (row, phase, input, n) and the peak bytes per cell. Steps: frames 2 and later."""
    groups = defaultdict(list)
    for r in records:
        if r["phase"] == "step" and int(r["repeat"]) < 2:
            continue
        groups[(key(r), r["phase"], r["input"], int(r["n"]))].append(r)
    out = {}
    for g, rs in groups.items():
        rates = [float(r["Mcells_s"]) for r in rs]
        bpc = [float(r["bytes_per_cell"]) for r in rs if float(r["peak_bytes"]) >= 0]
        out[g] = dict(median=statistics.median(rates), lo=min(rates), hi=max(rates), runs=len(rates),
                      bytes_per_cell=max(bpc) if bpc else None, jobs=sorted({r["job"] for r in rs}),
                      record=rs[0])
    return out


def fmt(s):
    if s is None:
        return ""
    m = s["median"]
    digits = 3 if m < 1 else (2 if m < 10 else (1 if m < 100 else 0))
    return f"{m:.{digits}f} ({s['lo']:.{digits}f}–{s['hi']:.{digits}f})"


def rate_table(summary, phase):
    rows = sorted({g[0] for g in summary if g[1] == phase}, key=lambda k: (k[2].startswith("A100") is False, k))
    lines = []
    cols = [(n, i) for n in SIZES for i in INPUTS if any(g[1] == phase and g[2] == i and g[3] == n for g in summary)]
    lines.append("| library | variant | device | threads | " + " | ".join(f"{i} {SIZE_NAME[n]}" for n, i in cols)
                 + " | B/cell (jittered 128³) |")
    lines.append("|" + "---|" * (5 + len(cols)))
    for k in rows:
        cells = [fmt(summary.get((k, phase, i, n))) for n, i in cols]
        b = summary.get((k, phase, "jittered", SIZES[0]))
        bpc = f"{b['bytes_per_cell']:.0f}" if b and b["bytes_per_cell"] is not None else ""
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | " + " | ".join(cells) + f" | {bpc} |")
    return "\n".join(lines)


def provenance_table(summary):
    seen = {}
    for g, s in summary.items():
        r = s["record"]
        seen.setdefault(g[0], dict(version=r["version"], outputs=r["outputs"], periodic=r["periodic"],
                                   precision=r["precision"], timed=r["timed"], note=r.get("note", ""), jobs=set()))
        seen[g[0]]["jobs"].update(s["jobs"])
    lines = ["| library | variant | device | threads | version | outputs | periodic | precision | timed | note | jobs |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(seen.items()):
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {v['version']} | {v['outputs']} | {v['periodic']} | "
                     f"{v['precision']} | {v['timed']} | {v['note']} | {', '.join(sorted(v['jobs']))} |")
    return "\n".join(lines)


def accuracy_table():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "accuracy-*.jsonl"))):
        with open(path) as f:
            for line in f:
                if line.startswith("{"):
                    # Older checks wrote inf / nan for a missing sampled cell; read them as infinite.
                    a = json.loads(line.replace(":inf", ":Infinity").replace(":nan", ":NaN").replace(":-nan", ":NaN"))
                    for key in ("sum_minus_box", "median_rel_err", "max_rel_err"):
                        a[key] = float("inf") if a[key] is None else a[key]
                    a["label"] = os.path.basename(a["volumes"]).rsplit(".", 1)[0]
                    rows.append(a)
    lines = ["| volumes (library-device_input_n) | sample | missing or ≤ 0 | sum − 1 | median rel. error | max rel. error | "
             "cells > 1e-12 | > 1e-6 | > 1e-2 |", "|---|---|---|---|---|---|---|---|---|"]
    for a in sorted(rows, key=lambda a: a["label"]):
        lines.append(f"| {a['label']} | {a['sample']} | {a['missing_or_nonpositive']} | {a['sum_minus_box']:.1e} | "
                     f"{a['median_rel_err']:.1e} | {a['max_rel_err']:.1e} | {a['cells_rel_err_above_1e-12']} | "
                     f"{a['above_1e-6']} | {a['above_1e-2']} |")
    return "\n".join(lines)


def published():
    path = os.path.join(ROOT, "bench", "libraries", "published.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def published_table(rows):
    lines = ["| code | reference | hardware | n | distribution | precision | periodic | output | rate (M/s) | kind | where |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in rows:
        lines.append(f"| {p['code']} | {p['citation_key']} | {p['hardware']} | {p['n']} | {p['distribution']} | "
                     f"{p['precision']} | {p['periodic']} | {p['output']} | {p['rate_Mcells_s']} | {p['rate_kind']} | "
                     f"{p['source_location']} |")
    return "\n".join(lines)


# The figure: build rate (cell volumes included) against input at 128³, one curve per library, the A100 codes on the
# left and the CPU node on the right. Measured: filled markers and solid lines (72 cores: dashed; serial codes on one
# core: dotted). Published (not measured by us): open markers in a column of their own, each with its hardware.
# (library, variant prefix, device prefix, threads, label)
FIGURE_GPU = [
    ("voronoi-cpp", "warp clipping rebuild", "A100", 1, "voronoi-cpp, warp rebuild"),
    ("voronoi-cpp", "exact start", "A100", 1, "voronoi-cpp, exact start (Delaunay)"),
    ("peclet-voro", "cold build", "A100", 1, "peclet-voro"),
    ("Liu et al. 3D-Voronoi-GPU", "tet-centric", "A100", 1, "Liu et al. 2022 (float)"),
    ("VoroGPU", "kNN + cell clipping, K 90", "A100", 1, "Ray et al. 2018, K 90 (float)"),
    ("Paragram", "", "A100", 1, "Paragram + our volumes (float)"),
    ("gDel3D", "Delaunay + dual volumes", "A100", 1, "gDel3D + our volumes"),
    ("Basselin", "", "A100", 1, "Basselin et al. 2021"),
    ("votess", "", "A100", 1, "votess + our volumes"),
]
FIGURE_CPU = [
    ("voronoi-cpp", "warp clipping rebuild", "2x Xeon", 72, "voronoi-cpp, warp rebuild"),
    ("voro++", "multithreaded", "2x Xeon", 72, "voro++ multithreaded"),
    ("peclet-voro", "cold build", "2x Xeon", 72, "peclet-voro"),
    ("CGAL Delaunay_triangulation_3", "Delaunay + dual volumes", "2x Xeon", 72, "CGAL parallel + our volumes"),
    ("geogram", "PeriodicDelaunay3d + Voronoi cells", "2x Xeon", 72, "geogram periodic + cells"),
    ("votess", "", "2x Xeon", 72, "votess + our volumes"),
    ("voro++", "container_periodic", "2x Xeon", 1, "voro++, 1 core"),
    ("CGAL Periodic_3_Delaunay_triangulation_3", "Delaunay + dual volumes", "2x Xeon", 1, "CGAL periodic + our vol., 1 core"),
    ("TetGen", "", "2x Xeon", 1, "TetGen + our volumes, 1 core"),
    ("Qhull", "", "2x Xeon", 1, "Qhull + our volumes, 1 core"),
]
# Published numbers shown: (code prefix in published.csv, n) of the rows that give cells with volumes or Delaunay
# points per second, nearest to 2M uniform points.
FIGURE_PUBLISHED_GPU = [("Basselin et al. 2021 (", "10000000", "white noise"), ("Ray et al. 2018 (meshl", "10000000", "white noise"),
                        ("Paragram", "2000000", "white noise"), ("votess (GPU", "1000000", ""),
                        ("peclet-voro (cold", "2000000", "")]
FIGURE_PUBLISHED_CPU = [("Voro++ (multithreaded", "100000000", ""), ("votess (CPU", "1000000", "")]


def pick(summary, lib, variant, device, threads):
    """The summary rows of one library variant (build phase) at 128³, by input."""
    for g in summary:
        k = g[0]
        if g[1] == "build" and k[0].startswith(lib) and k[1].startswith(variant) and k[2].startswith(device) \
                and k[3] == threads:
            return {i: summary.get((k, "build", i, SIZES[0])) for i in INPUTS}
    return None


def figure(summary, pub, path):
    figstyle.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(2 * figstyle.FIGSIZE[0], 1.25 * figstyle.FIGSIZE[1]), sharey=True)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = "osD^vP*Xh<>"
    for ax, rows, published_rows, title in ((axes[0], FIGURE_GPU, FIGURE_PUBLISHED_GPU, "one A100"),
                                            (axes[1], FIGURE_CPU, FIGURE_PUBLISHED_CPU, "one CPU node (72 cores)")):
        figstyle.decorate(ax)
        xs = list(range(len(INPUTS)))
        j = 0
        for lib, variant, device, threads, label in rows:
            got = pick(summary, lib, variant, device, threads)
            if not got:
                continue
            pts = [(x, got[i]) for x, i in zip(xs, INPUTS) if got[i]]
            if not pts:
                continue
            c, m = colours[j % len(colours)], markers[j % len(markers)]
            j += 1
            style = "-" if device == "A100" else ("--" if threads > 1 else ":")
            ax.errorbar([x for x, _ in pts], [s["median"] for _, s in pts],
                        yerr=[[s["median"] - s["lo"] for _, s in pts], [s["hi"] - s["median"] for _, s in pts]],
                        color=c, marker=m, linewidth=2.2, linestyle=style, markersize=7, label=label)
        x0 = len(INPUTS)
        k = 0
        for code, n, dist in published_rows:
            for p in pub:
                if p["code"].startswith(code) and p["n"] == n and dist in p["distribution"]:
                    c = colours[(j + k) % len(colours)]
                    ax.plot([x0 + 0.1 * (k - (len(published_rows) - 1) / 2)], [float(p["rate_Mcells_s"])],
                            marker=markers[(j + k) % len(markers)], markerfacecolor="none", markeredgecolor=c,
                            markeredgewidth=1.6, markersize=9, linestyle="none",
                            label=f"{p['code'].split(' (')[0]}, {p['hardware']} (published)")
                    k += 1
                    break
        ax.set_xticks(xs + [x0])
        ax.set_xticklabels(INPUTS + ["published"])
        ax.set_yscale("log")
        ax.set_xlabel(f"input, {SIZES[0]:,} cells; {title}")
        figstyle.legend(ax, loc="upper right", fontsize=8.5)
    axes[0].set_ylabel("M cells/s, build with cell volumes")
    fig.savefig(path)


def main():
    records, extra = load_records()
    summary = summarise(records)
    pub = published()
    text = ["# Library benchmark: tables generated by scripts/library_benchmark.py (do not edit)", "",
            "Median of 3 timed runs (min–max in brackets), M cells/s; steps: median of steps 2 to 10.", "",
            "## Build", "", rate_table(summary, "build"), "", "## Moving-mesh step", "", rate_table(summary, "step"),
            "", "## Setup per row", "", provenance_table(summary), "", "## Accuracy against the oracle", "",
            accuracy_table(), "", "## Published numbers (not measured by us)", "", published_table(pub), "",
            "## Side records (memory)", ""]
    text += [f"- `{json.dumps(e)}`" for e in extra]
    with open(os.path.join(RESULTS, "summary.md"), "w") as f:
        f.write("\n".join(text) + "\n")
    figure(summary, pub, os.path.join(ROOT, "docs", "figures", "library_benchmark.png"))


if __name__ == "__main__":
    main()
