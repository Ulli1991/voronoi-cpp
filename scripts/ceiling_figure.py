#!/usr/bin/env python3
"""Draws docs/figures/measure_ceiling.png: DRAM bytes per cell of each kernel (Nsight Compute, 128³ jittered unless
named) against the bytes per cell that 5x and 10x the baseline rate would allow with every kernel at 80 % of the
A100's 1555 GB/s. Usage: ceiling_figure.py <ceiling.json from ceiling_tables.py> <out.png>"""
import json
import sys

import matplotlib.pyplot as plt

import figstyle

TARGET = 0.8 * 1555e9
# (row label, [(json run, phase)], baseline M cells/s on main 69de1d3)
ROWS = [
    ("step 0.03", [("step_0.03", "update")], 196),
    ("step + cells 0.03", [("step_0.03", "update"), ("step_0.03", "cells")], 154),
    ("step 0.1", [("step_0.1", "update")], 114),
    ("rebuild jittered", [("rebuild_jittered", "list"), ("rebuild_jittered", "clip")], 39.4),
    ("rebuild uniform", [("rebuild_uniform", "list"), ("rebuild_uniform", "clip")], 19.7),
    ("exact start", [("start", "start")], 17.7),
]
# Kernel groups, in stacking order; the rest goes to "other".
GROUPS = [
    ("faces", ("tet_mesh::faces",)),
    ("cell_sums", ("tet_mesh::cell_sums",)),
    ("screen + circumcentres", ("tet_repair::screen", "tet_mesh::circumcentres")),
    ("face numbers, rings", ("tet_mesh::face_numbers", "tet_mesh::dirty_faces", "tet_mesh::append_rings",
                             "tet_mesh::compact_rings", "tet_mesh::ring_entries", "tet_mesh::ring_start",
                             "tet_mesh::first_tet", "tet_mesh::face_start")),
    ("repair (find, plan, apply)", ("tet_repair::find", "tet_repair::plan", "tet_repair::apply",
                                    "tet_repair::unlist", "tet_repair::release", "tet_repair::retry_stuck")),
    ("cell lists", ("tet_mesh::cell_sort", "tet_mesh::cell_face_fill", "tet_mesh::cell_face_counts",
                    "tet_mesh::cell_face_start", "tet_mesh::marked_fill", "tet_mesh::marked_counts",
                    "tet_mesh::fresh_fill", "tet_mesh::fresh_counts", "tet_mesh::mark_cells")),
    ("clipping", ("shell_voronoi::construct_warp", "shell_voronoi::construct_warp_rings", "shell_voronoi::construct")),
    ("from_rings", ("tet_mesh::emit", "tet_mesh::adjacency", "tet_mesh::face_particles", "tet_mesh::represent",
                    "tet_mesh::represent_dropped", "tet_mesh::face_lengths", "tet_mesh::edge_bytes",
                    "Kokkos::ViewCopy-2D", "Kokkos::ViewCopy-1D", "cub::DeviceRadixSort")),
]
COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#7f7f7f"]


def run(data, key):
    return next(v for k, v in data.items() if k.startswith(key + "-"))


def main():
    data = json.load(open(sys.argv[1]))
    figstyle.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(2 * figstyle.FIGSIZE[0], figstyle.FIGSIZE[1]))
    names = [g[0] for g in GROUPS] + ["other"]
    for ax, rows, xmax in ((axes[0], ROWS[:3], 5600), (axes[1], ROWS[3:], 17500)):
        figstyle.decorate(ax)
        ys = list(range(len(rows)))[::-1]
        for y, (label, parts, base) in zip(ys, rows):
            per = dict.fromkeys(names, 0.0)
            for key, phase in parts:
                for r in run(data, key):
                    if r["phase"] == phase:
                        per[next((n for n, labels in GROUPS if r["label"] in labels), "other")] += r["B"]
            left = 0.0
            for n, c in zip(names, COLOURS):
                if per[n] > 0:
                    ax.barh(y, per[n], left=left, color=c, height=0.55, zorder=3, label=n)
                left += per[n]
            edge = max(left, TARGET / (5 * base * 1e6))
            ax.text(edge + 0.02 * xmax, y, f"{left:,.0f}", va="center", fontsize=11)
            for mult, ls in ((5, "-"), (10, "--")):
                b = TARGET / (mult * base * 1e6)
                ax.plot([b, b], [y - 0.4, y + 0.4], color="k", lw=2.4, ls=ls, zorder=4,
                        label=f"{mult}x at 80 % of peak")
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1000 if xmax < 10000 else 4000))
        ax.set_ylim(-0.6, len(rows) + 0.9)
        ax.set_yticks(ys, [r[0] for r in rows])
        ax.tick_params(axis="y", which="both", left=False, right=False)
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel("DRAM bytes per cell, read + written")
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        for h, lab in zip(handles, labels):
            seen.setdefault(lab, h)
        figstyle.legend(ax, handles=list(seen.values()), labels=list(seen.keys()), loc="upper right", fontsize=9.5,
                        ncol=2)
    fig.tight_layout()
    fig.savefig(sys.argv[2])


if __name__ == "__main__":
    main()
