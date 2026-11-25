#!/usr/bin/env python3
"""Figure application: completeness against build rate on the three application inputs (cosmic web 128³, fcc solid 2²⁰
atoms, surface 128³), one point per code, from the library benchmark's own runs: rates from bench/libraries/results
(the "Application inputs" tables of scripts/library_benchmark.py), cells missing from accuracy-inputs-*.jsonl and the
per-cell errors of paper/data/oracle_errors_application.csv. Codes that crash are not drawn (CRASHED; named in the
caption), codes whose volumes are wrong are drawn open; the grey band holds the codes with no cell missing."""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from common import oracle_errors
from paperfig import DOUBLE, REPO, figstyle, figure, save

sys.path.insert(0, os.path.join(REPO, "scripts"))
import library_benchmark as lb  # noqa: E402

PANELS = [("cosmicweb", 2097152, "cosmic web, 128³"), ("fccsolid", 1048576, "fcc solid, 2²⁰ atoms"),
          ("surface", 2097152, "surface, 128³")]
# The kNN codes' README setting per input class (library_benchmark.README_SETTING): fcc solid as a perturbed grid.
SETTING = {"basselin": {"fccsolid": "grid"}, "vorogpu": {"fccsolid": "k90"}, "liu": {"fccsolid": "p1"}}
WHITE = {"basselin": "white", "vorogpu": "k190", "liu": "p0"}
# (label, volumes prefix with {s} for the README setting, library, variant prefix, device, threads, colour, marker)
CODES = [
    ("this work, rebuild", "voronoi-cpp-warp-a100", "voronoi-cpp", "warp clipping rebuild (cell", "A100", 1,
     "#000000", "o"),
    ("this work, exact start", "voronoi-cpp-exact-a100", "voronoi-cpp", "exact start", "A100", 1, "#555555", "o"),
    ("peclet-voro", "peclet-voro-a100", "peclet-voro", "cold build", "A100", 1, "#ff7f0e", "o"),
    ("gDel3D + faces", "gdel3d-a100", "gDel3D", "Delaunay + faces", "A100", 1, "#2ca02c", "o"),
    ("Paragram + faces", "paragram-a100", "Paragram", "voronoi_diagram + faces", "A100", 1, "#d62728", "o"),
    ("votess + faces", "votess-a100", "votess", "tesellate + faces", "A100", 1, "#9467bd", "o"),
    ("Basselin et al.", "basselin-{s}-a100", "Basselin", "", "A100", 1, "#8c564b", "o"),
    ("Ray et al. (float)", "vorogpu-{s}-a100", "VoroGPU", "", "A100", 1, "#e377c2", "o"),
    ("Liu et al. (float)", "voronoi-gpu-liu-tet-{s}-a100", "Liu et al.", "tet-centric", "A100", 1, "#bcbd22", "o"),
    ("voro++ (72 cores)", "voropp-mt-cpu", "voro++", "multithreaded", "2x Xeon", 72, "#1f77b4", "D"),
    ("CGAL + faces (72 cores)", "cgal-dt3-par-cpu", "CGAL Delaunay_triangulation_3", "Delaunay + faces", "2x Xeon", 72,
     "#17becf", "D"),
    ("geogram (72 cores)", "geogram-pdel-cpu", "geogram", "PeriodicDelaunay3d + Voronoi cells", "2x Xeon", 72,
     "#e6ab02", "D"),
    ("votess + faces (72 cores)", "votess-cpu", "votess", "tesellate + faces", "2x Xeon", 72, "#9467bd", "D"),
]
# Runs that end without volumes (docs/measurements.md, "Application inputs"): no rate, no record; listed in the panel.
CRASHED = {("votess + faces", "cosmicweb"): "CUDA error 700", ("votess + faces", "surface"): "CUDA error 700",
           ("votess + faces (72 cores)", "surface"): "length_error"}
ISSUE_60 = ("this work, rebuild", "this work, exact start")  # surface: cells refused, about 300x slower (#60 open)


def accuracy():
    """{volumes file name: (missing of all n, sum of volumes - box)} from the application inputs' oracle checks."""
    out = {}
    for path in sorted(glob.glob(os.path.join(REPO, "bench", "libraries", "results", "accuracy-inputs-*.jsonl"))):
        for line in open(path):
            if line.startswith("{"):
                r = json.loads(line.replace(":null", ":NaN"))
                out[os.path.basename(r["volumes"])] = (r["missing_or_nonpositive"] / r["n"], r["sum_minus_box"])
    return out


def summary():
    """The benchmark's summary with our rows of the application inputs from our own jobs (lb.OURS_APPLICATION)."""
    records, _ = lb.load_records()
    app = {i for _, i in lb.APPLICATION}
    return lb.summarise([r for r in records if r["input"] in app and (
        r["library"] != "voronoi-cpp" or (r["version"] == lb.OURS_APPLICATION
                                         and r["file"].startswith("voronoi-cpp-a100")))])


def rows():
    s, acc, err = summary(), accuracy(), oracle_errors()
    out = []
    for label, prefix, lib, variant, device, threads, colour, marker in CODES:
        key = next((k for k in WHITE if k in prefix), None)
        for kind, n, _ in PANELS:
            r = dict(label=label, colour=colour, marker=marker, kind=kind)
            if (label, kind) in CRASHED:
                out.append(dict(r, state="crashed", why=CRASHED[(label, kind)]))
                continue
            speed = lb.curve(s, lib, variant, device, threads, n)
            name = prefix.format(s=SETTING[key].get(kind, WHITE[key]) if key else "") + f"_{kind}_{n}.f64"
            if kind not in speed or name not in acc:
                continue
            missing, total = acc[name]
            e = err[name][2] if name in err else np.array([])
            # wrong: a sampled cell that has a volume is off by more than 1 %, or the cells cover more than the box
            wrong = bool(np.any(e[np.isfinite(e)] > 1e-2)) or total > 1e-3
            out.append(dict(r, rate=speed[kind]["median"], missing=missing, total=total,
                            state="wrong" if wrong else "ok", worst=float(e[np.isfinite(e)].max()) if e.size else 0))
    return out


# Level of each complete code in the band (steps of 25 %), chosen so codes of similar rate sit at least two levels apart.
LEVEL = {"this work, rebuild": 6, "this work, exact start": 1, "peclet-voro": 11, "gDel3D + faces": 0,
         "Paragram + faces": 8, "votess + faces": 7, "Basselin et al.": 4, "Ray et al. (float)": 9,
         "Liu et al. (float)": 6, "voro++ (72 cores)": 5, "CGAL + faces (72 cores)": 12,
         "geogram (72 cores)": 3, "votess + faces (72 cores)": 10}


def height(r):
    """The fraction missing, or for a complete code its level in the band."""
    return r["missing"] if r["missing"] > 0 else 1.7e-8 * 1.25 ** LEVEL[r["label"]]


def main():
    data = rows()
    fig, axes = figure(DOUBLE, 80, ncols=3, sharey=True, gridspec_kw=dict(width_ratios=[1, 1.3, 1]))
    for ax, (kind, n, title) in zip(axes, PANELS):
        here = [r for r in data if r["kind"] == kind]
        ax.axhspan(1e-8, 2.5e-7, color="0.92", zorder=0)
        for r in here:
            if r["state"] == "crashed":
                continue
            y = height(r)
            hollow = r["state"] == "wrong"  # open and on top, so a wrong code is never hidden behind a right one
            ax.plot(r["rate"], y, r["marker"], color=r["colour"], ms=9 if hollow else 8,
                    mfc="none" if hollow else r["colour"], mec=r["colour"] if hollow else "white",
                    mew=1.8 if hollow else 0.5, zorder=5 if hollow else 4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.01, 80)
        ax.set_xticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
        ax.set_ylim(1e-8, 5)
        ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
        ax.set_xlabel("build rate, M cells/s")
        ax.set_title(title, fontsize=13)
        figstyle.decorate(ax)
    axes[0].set_ylabel("fraction of cells missing")
    # One shared legend inside the panel with the most room (fcc solid: every code complete, the top is empty): one
    # entry per code (colour), then the keys for marker shape (where it ran) and open markers (wrong volumes).
    seen = {}
    for r in data:
        seen.setdefault(r["label"].replace(" (72 cores)", ""), r["colour"])
    handles = [plt.Line2D([], [], color=c, marker="o", ls="none", ms=8, mec="white", mew=0.5) for c in seen.values()]
    handles += [plt.Line2D([], [], color="0.3", marker="o", ls="none", ms=8),
                plt.Line2D([], [], color="0.3", marker="D", ls="none", ms=7),
                plt.Line2D([], [], color="0.3", marker="o", ls="none", ms=9, mfc="none", mew=1.8)]
    axes[1].legend(handles, list(seen) + ["one A100", "72 cores", "wrong volumes"], loc="center", ncol=2,
                   fontsize=10.5, handletextpad=0.2, columnspacing=0.6, bbox_to_anchor=(0.5, 0.56))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.14, wspace=0.08)
    save(fig, "application", DOUBLE)
    for r in data:
        print(r["kind"], r["label"], r["state"], f"{r.get('rate', 0):.3g}", f"{r.get('missing', 0):.2e}",
              f"{r.get('total', 0):.1e}", f"{r.get('worst', 0):.1e}")


if __name__ == "__main__":
    main()
