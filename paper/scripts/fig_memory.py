#!/usr/bin/env python3
"""Figure memory: (a) device B/cell by array role at the peak of the rebuild, the exact start and a mesh update
(Kokkos arrays, 128³ and 256³, jittered); (b) peak B/cell of each code in the library benchmark.
Data: paper/data/memory_by_array.csv, bench/libraries/results."""
import csv
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

from paperfig import DATA, DOUBLE, REPO, figstyle, figure, save

sys.path.insert(0, os.path.join(REPO, "scripts"))
import library_benchmark as lb  # noqa: E402

ROLES = [  # (role, array prefixes), stacking order
    ("positions", ("positions", "slot_positions", "tet_repair_point")),
    ("cell list", ("cell_offsets", "particle_order", "slot_fine_keys", "slot_levels")),
    ("tetrahedra", ("tet_vertex", "tet_adjacent", "tet_centre", "tet_edges")),
    ("faces", ("face_area", "face_centroid", "face_neighbour", "face_rows")),
    ("face rings", ("face_rings_",)),
    ("per-cell lists", ("cell_",)),
    ("repair scratch", ("tet_repair_",)),
]
COLOURS = ["#7f7f7f", "#bcbd22", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#d62728"]
GPU_ROWS = [  # (library prefix, variant prefix, label); full cells unless said
    ("voronoi-cpp", "warp clipping rebuild (cell", "Kikkou, rebuild"),
    ("voronoi-cpp", "exact start", "Kikkou, exact start"),
    ("peclet-voro", "cold build", "peclet-voro"),
    ("gDel3D", "Delaunay + faces", "gDel3D + faces"),
    ("Paragram", "voronoi_diagram + faces", "Paragram + faces"),
    ("votess", "tesellate + faces", "votess + faces"),
    ("Basselin", "", "Basselin et al. (volumes)"),
    ("Liu et al.", "tet-centric", "Liu et al. (volumes)"),
    ("VoroGPU", "", "Ray et al. (volumes)"),
]


def role(array):
    return next((r for r, prefixes in ROLES if any(array.startswith(p) for p in prefixes)), None)


def by_role():
    """{(n, phase): {role: B/cell}} for jittered input."""
    out = defaultdict(lambda: defaultdict(float))
    for r in csv.DictReader(open(os.path.join(DATA, "memory_by_array.csv"))):
        if r["input"] == "jittered" and role(r["array"]):
            out[int(r["n"]), r["phase"]][role(r["array"])] += float(r["bytes_per_cell"])
    return out


def main():
    data = by_role()
    fig, (a, b) = figure(DOUBLE, 82, ncols=2, gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    phases = ("rebuild", "exact start", "moving step")
    for i, p in enumerate(phases):
        left = 0.0
        for (name, _), c in zip(ROLES, COLOURS):
            v = data[2097152, p].get(name, 0.0)
            a.bar(i, v, bottom=left, color=c, width=0.7, label=name if i == 0 else None)
            left += v
        big = sum(data[16777216, p].values())
        a.text(i, left + 25, f"{left:.0f}", ha="center", va="bottom", fontsize=11)
    a.set_xticks(range(3), ["rebuild", "exact\nstart", "mesh\nupdate"])
    a.set_xlim(-0.6, 4.4)  # room right of the bars for the legend
    a.set_ylim(0, 2200)
    a.set_ylabel("device B/cell at the peak, 128³")
    handles, labels = a.get_legend_handles_labels()  # outside on the right, top role first as in the stack
    a.legend(handles[::-1], labels[::-1], loc="center right", fontsize=11)
    a.text(0.03, 0.98, "(a)", transform=a.transAxes, ha="left", va="top")
    figstyle.decorate(a)
    a.xaxis.set_minor_locator(plt.NullLocator())
    a.tick_params(axis="x", which="both", top=False, bottom=False)

    records, _ = lb.load_records()
    summary = lb.summarise([r for r in records if r["library"] != "voronoi-cpp" or r["version"] == lb.OURS])
    rows = []
    for lib, variant, label in GPU_ROWS:
        got = lb.curve(summary, lib, variant, "A100", 1, lb.SIZES[0])
        bpc = [s["bytes_per_cell"] for i, s in got.items() if s["bytes_per_cell"] and i in lb.INPUTS]  # the five inputs
        if bpc:
            rows.append((label, min(bpc), max(bpc)))
    steps = [float(r["bytes_per_cell"]) for k, r in csv_rows() if k == "moving step"]
    rows.insert(2, ("Kikkou, mesh update", min(steps), max(steps)))
    for y, (label, lo, hi) in enumerate(rows):
        ours = label.startswith("Kikkou")
        b.barh(y, hi, color="#666666" if ours else "#cfcfcf", height=0.6)
        b.barh(y, lo, color="#000000" if ours else "#8a8a8a", height=0.6)
        b.text(hi * 1.08, y, f"{lo:.0f}–{hi:.0f}" if hi > lo * 1.02 else f"{hi:.0f}", va="center", fontsize=11,
               zorder=5)
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.patches import Patch
    b.legend([(Patch(color="#000000"), Patch(color="#8a8a8a")), (Patch(color="#666666"), Patch(color="#cfcfcf"))],
             ["lowest input", "up to the highest"], handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
             handlelength=2.0, loc="lower right", bbox_to_anchor=(1.0, 0.13), fontsize=11)  # above the Ray et al. row
    b.set_xscale("log")
    b.set_xlim(60, 3e4)
    b.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=11)
    b.set_ylim(len(rows) - 0.4, -1.1)
    b.set_xlabel("device B/cell at the peak, 128³")
    b.text(0.02, 0.98, "(b)", transform=b.transAxes, ha="left", va="top")
    figstyle.decorate(b)
    b.tick_params(axis="y", which="both", left=False, right=False)
    b.yaxis.set_minor_locator(plt.NullLocator())
    fig.tight_layout()
    save(fig, "memory", DOUBLE)


def csv_rows():
    for r in csv.DictReader(open(os.path.join(DATA, "memory_by_array.csv"))):
        if r["array"] == "(peak, all arrays)" and r["n"] == "2097152":
            yield r["phase"], r


if __name__ == "__main__":
    main()
