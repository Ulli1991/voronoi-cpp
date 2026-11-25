#!/usr/bin/env python3
"""Figure where_time ("where the time goes"): (a) the control flow of Mesh::update(), its decisions and stages, in the
colours of the stages; (b) wall time per cell of a moving step in six stages, per input at 0.03, 0.1 and 0.3 mean
spacings per step (the number at each bar: M cells/s). 128³ on one A100. Data: paper/data/stages.csv (wall time per stage), kernels.csv (Kokkos Tools
kernel times); the kernel labels are mapped onto the six stages below (also in paper/data/SOURCES.md); "host and
other" is the wall time minus the mapped kernels. The flowchart follows scripts/flow_chart.py and flow_detail.py."""
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

from paperfig import DATA, DOUBLE, LABEL, figstyle, figure, save

N = 2097152
INPUTS = ["jittered", "uniform", "clustered", "halo", "lattice", "wave"]
SPACINGS = ["0.03", "0.1", "0.3"]
# (stage, colour, kernel labels): the owner's six stages, in the order of a step
STAGES = [
    ("check", "#ff7f0e", ["voronoi::snap", "exact_predicates::snap", "tet_repair::screen"]),
    ("flip", "#2ca02c", ["tet_repair::" + s for s in "find plan apply unlist release retry_stuck".split()]),
    ("reconnect", "#9467bd", ["tet_mesh::" + s for s in ("face_touched face_release face_offsets face_assign face_move "
                                                          "face_low face_numbered face_numbers face_ids dirty_faces "
                                                          "append_rings compact_rings ring_entries").split()]),
    ("geometry", "#1f77b4", ["tet_mesh::circumcentres", "tet_mesh::faces", "voronoi::total_volume"]),
    ("per-cell results", "#8c564b", ["tet_mesh::" + s for s in ("cell_face_counts cell_face_fill cell_face_start "
                                                                 "cell_sort cell_sums fresh_counts fresh_fill mark_cells "
                                                                 "marked_counts marked_fill").split()]),
    ("host and other", "#7f7f7f", []),  # the wall time no mapped kernel accounts for
]
COLOUR = {s: c for s, c, _ in STAGES}
BENCH = ("Kokkos::View::initialization", "deep_copy")  # the bench's host copy of the positions, outside the timed step
INK, FAINT = "#222222", "#6b6b6b"


def ns(ms):
    return ms * 1e6 / N


def data():
    """{(input, spacings): {stage: ns per cell}} with the wall total of stages.csv split into the six stages."""
    stage_of = {k: s for s, _, labels in STAGES for k in labels}
    kern = defaultdict(lambda: defaultdict(float))
    for r in csv.DictReader(open(os.path.join(DATA, "kernels.csv"))):
        if r["kernel"] not in BENCH:
            kern[r["input"], r["spacings"]][stage_of.get(r["kernel"], "host and other")] += float(r["ms_per_step"])
    wall = defaultdict(float)
    for r in csv.DictReader(open(os.path.join(DATA, "stages.csv"))):
        if r["what"].startswith("step"):
            wall[r["input"], r["what"].split()[1]] += float(r["ms"])
    out = {}
    for key, parts in kern.items():
        kernels = sum(v for s, v in parts.items() if s != "host and other")
        row = {s: ns(parts.get(s, 0.0)) for s, _, _ in STAGES[:-1]}
        row["host and other"] = ns(max(wall[key] - kernels, 0.0))
        out[key] = row
    return out


def tint(colour, f=0.22):
    r, g, b = to_rgb(colour)
    return (1 - f + f * r, 1 - f + f * g, 1 - f + f * b)


def flowchart(ax):
    ax.set_xlim(0, 160)
    ax.set_ylim(3.5, 54)
    ax.axis("off")
    ax.minorticks_off()
    top, mid, bus, low = 47, 33, 23.5, 11

    def box(x, y, w, h, title, stage=None, edge=None, fill=None, ls="-"):
        e = COLOUR[stage] if stage else (edge or "0.45")
        f = tint(COLOUR[stage]) if stage else (fill or "white")
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=f, ec=e,
                                    lw=1.8, ls=ls, zorder=3))
        ax.text(x, y, title, ha="center", va="center", fontsize=11, fontweight="bold", color=INK, zorder=4,
                linespacing=1.05)

    def diamond(x, y, w, h, text):
        ax.add_patch(Polygon([(x - w / 2, y), (x, y + h / 2), (x + w / 2, y), (x, y - h / 2)], closed=True, fc="white",
                             ec=INK, lw=1.4, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=11, color=INK, linespacing=1.0, zorder=4)

    def arrow(points, label=None, at=None):
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot(*zip(a, b), color=INK, lw=1.4, zorder=2, solid_capstyle="round")
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=13, lw=1.4, color=INK,
                                     shrinkA=0, shrinkB=0, zorder=2))
        if label:
            ax.text(*at, label, ha="center", va="center", fontsize=11, color=FAINT, style="italic", zorder=5,
                    bbox=dict(fc="white", ec="none", pad=0.6))

    def line(points):
        ax.plot(*zip(*points), color=INK, lw=1.4, zorder=2, solid_joinstyle="round")

    # the step (top row)
    ax.text(0.5, top, "positions", ha="left", va="center", fontsize=11, fontweight="bold", color=INK)
    diamond(22, top, 11, 12, "mesh\nbuilt?")
    box(39, top, 14, 9, "Check", "check")
    box(56, top, 14, 9, "Flip", "flip")
    diamond(72, top, 10, 11, "stuck?")
    box(89.5, top, 17, 9, "Reconnect", "reconnect")
    box(108, top, 16, 9, "Geometry", "geometry")
    diamond(124, top, 11, 12, "volume\ncheck")
    box(145, top, 19, 9, "Per-cell\nresults", "per-cell results")
    arrow([(14, top), (16.5, top)])
    arrow([(27.5, top), (32, top)], "yes", (29.7, top + 3.2))
    arrow([(46, top), (49, top)])
    arrow([(63, top), (67, top)])
    arrow([(77, top), (81, top)], "no", (79, top + 3.2))
    arrow([(98, top), (100, top)])
    arrow([(116, top), (118.5, top)])
    arrow([(129.5, top), (135.5, top)], "pass", (132.5, top + 3.2))
    # few stuck: the host finishes, then reconnect; still stuck: rebuild
    box(72, mid, 17, 7, "Host finish", "host and other")
    arrow([(72, top - 5.5), (72, mid + 3.5)], "few", (74.8, top - 8))
    arrow([(80.5, mid), (89.5, mid), (89.5, top - 4.5)], "done", (85, mid))
    line([(72, mid - 3.5), (72, bus)])
    ax.text(74, (mid - 3.5 + bus) / 2, "still stuck", ha="left", va="center", fontsize=11, color=FAINT, style="italic")
    line([(124, top - 6), (124, bus), (22, bus)])
    ax.text(126, (top - 6 + bus) / 2, "fail", ha="left", va="center", fontsize=11, color=FAINT, style="italic")
    ax.text(47, bus, "rebuild", ha="center", va="center", fontsize=11, color=FAINT, style="italic",
            bbox=dict(fc="white", ec="none", pad=0.6))
    for x in (72, 124):
        ax.add_patch(Circle((x, bus), 0.7, fc=INK, ec=INK, zorder=3))
    # the rebuild path (bottom row), counted in "host and other"
    line([(22, top - 6), (22, low)])
    ax.text(20, (top - 6 + bus) / 2 + 2, "no", ha="right", va="center", fontsize=11, color=FAINT, style="italic")
    arrow([(22, low), (31, low)])
    for x, w, name in ((38.5, 14, "Quantise"), (55, 13, "Cell list"), (69.5, 10, "Clip"), (85.5, 16, "Exact start")):
        box(x, low, w, 8, name, edge="#7f7f7f", fill=tint("#7f7f7f", 0.12), ls=(0, (4, 2)))
    for a, b in ((45.5, 48.5), (61.5, 64.5), (74.5, 77.5)):  # the arrows between them
        arrow([(a, low), (b, low)])
    diamond(108, low, 10, 11, "exact?")
    arrow([(93.5, low), (103, low)])
    arrow([(108, low + 5.5), (108, top - 4.5)], "yes", (110.3, low + 8))
    box(132, low, 13, 7, "Refuse", edge="#d62728", fill=tint("#d62728", 0.15))
    arrow([(113, low), (125.5, low)], "no", (119, low + 2.6))
    ax.text(0.0, 1.0, "(a)", transform=ax.transAxes, va="top", fontsize=12)


def bars(ax, rows, labels, values, numbers=True):
    for y, row in zip(rows, values):
        if row is None:
            continue
        left = 0.0
        for s, c, _ in STAGES:
            ax.barh(y, row[s], left=left, color=c, height=0.72, zorder=3)
            left += row[s]
        if numbers:
            ax.text(left + 0.25, y, f"{1e3 / left:.0f}", va="center", fontsize=11, zorder=4)
    ax.set_yticks(rows, labels)
    ax.tick_params(axis="y", which="both", left=False, right=False)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    figstyle.decorate(ax)


def main():
    d = data()
    fig = figure(DOUBLE, 170)[0]
    fig.clf()
    flowchart(fig.add_axes([0.005, 0.585, 0.99, 0.41]))  # the flowchart takes the full width
    b = fig.add_axes([0.2, 0.07, 0.6, 0.48])
    rows, labels, values, y = [], [], [], 0
    for kind in INPUTS:
        for k, s in enumerate(SPACINGS):
            rows.append(y)
            labels.append(f"{LABEL[kind]}, {s}" if k == 0 else s)
            values.append(d.get((kind, s)))
            y += 1
        y += 0.5
    bars(b, rows, labels, values)
    b.set_ylim(y - 0.3, -0.8)
    b.set_xlim(0, 26)
    b.xaxis.set_major_locator(plt.MultipleLocator(5))
    b.set_xlabel("wall time per cell, ns")
    b.text(-0.3, 1.0, "(b)", transform=b.transAxes, ha="left", va="top")
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for _, col, _ in STAGES]
    b.legend(handles, [s for s, _, _ in STAGES], loc="lower right", fontsize=11)
    save(fig, "where_time", DOUBLE)


if __name__ == "__main__":
    main()
