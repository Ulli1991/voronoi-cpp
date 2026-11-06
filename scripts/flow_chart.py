#!/usr/bin/env python3
"""Draws docs/figures/flow_mesh.png, the control flow of Mesh::update, in LUNA's figure style."""
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("figstyle", os.path.join(HERE, "figstyle.py"))
figstyle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figstyle)
figstyle.apply_style()

INK, FAINT = "#222222", "#6b6b6b"
SETUP = ("#eef1f5", "#7c8796")
REPAIR = ("#dff0dc", "#2ca02c")
REBUILD = ("#dbe9f6", "#1f77b4")
TEST = ("#fde8d3", "#ff7f0e")
REFUSE = ("#f8dcdc", "#d62728")


def box(ax, x, y, w, h, title, text="", tone=SETUP, lw=1.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.1", fc=tone[0], ec=tone[1], lw=lw, zorder=3))
    if text:
        ax.text(x, y + 0.2 * h, title, ha="center", va="center", fontsize=11, fontweight="bold", color=INK, zorder=4)
        ax.text(x, y - 0.2 * h, text, ha="center", va="center", fontsize=9, color=INK, linespacing=1.15, zorder=4)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=11, fontweight="bold", color=INK, zorder=4)


def diamond(ax, x, y, w, h, text):
    ax.add_patch(Polygon([(x - w / 2, y), (x, y + h / 2), (x + w / 2, y), (x, y - h / 2)], closed=True, fc=TEST[0], ec=TEST[1], lw=1.5, zorder=3, joinstyle="round"))
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5, color=INK, linespacing=1.1, zorder=4)


def arrow(ax, start, end, label=None, at=None, bend=None, color=INK):
    if bend:
        points = [start, *bend]
        for a, b in zip(points, points[1:]):
            ax.plot(*zip(a, b), color=color, lw=1.6, solid_capstyle="round", zorder=2)
        start = points[-1]
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, lw=1.6, color=color, shrinkA=0, shrinkB=0, zorder=2))
    if label:
        ax.text(*at, label, ha="center", va="center", fontsize=10, color=FAINT, style="italic", zorder=5, bbox=dict(fc="white", ec="none", pad=1.2))


def line(ax, x, y):
    ax.plot(x, y, color=INK, lw=1.6, solid_capstyle="round", solid_joinstyle="round", zorder=2)


def note(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=10, color=FAINT, style="italic", zorder=5, bbox=dict(fc="white", ec="none", pad=1.2))


def main():
    fig, ax = plt.subplots(figsize=(17, 6.8))
    ax.set_xlim(0, 153)
    ax.set_ylim(0, 60)
    ax.axis("off")
    ax.minorticks_off()
    top, bus, low = 46, 24, 11

    box(ax, 9, top, 15, 10, "Positions x", "from the solver")
    diamond(ax, 25, top, 12, 13, "mesh built,\nsame n?")
    box(ax, 42, top, 14, 10, "Snap", "to the 2³⁰ grid", REPAIR)
    box(ax, 60, top, 16, 10, "Screen", "tetrahedra that may\nfail, in double", REPAIR)
    box(ax, 80, top, 18, 10, "Flip on the GPU", "2-3, 3-2 flips, decided\nby exact predicates", REPAIR, lw=2.4)
    diamond(ax, 97, top, 9, 12, "stuck?")
    box(ax, 113, top, 13, 10, "Geometry", "centres, faces,\nvolumes", REPAIR)
    diamond(ax, 129, top, 13, 13, "volumes sum\nto the box?")
    box(ax, 145, top, 14, 10, "Exact mesh", "update_cells():\nface lists")
    box(ax, 97, 32.5, 20, 7.5, "Finish on the host", "only if few are stuck", REPAIR)

    arrow(ax, (16.5, top), (19, top))
    arrow(ax, (31, top), (35, top), "yes", (33, top + 1.8))
    arrow(ax, (49, top), (52, top))
    arrow(ax, (68, top), (71, top))
    arrow(ax, (89, top), (92.5, top))
    arrow(ax, (101.5, top), (106.5, top), "no", (104, top + 1.8))
    arrow(ax, (119.5, top), (122.5, top))
    arrow(ax, (135.5, top), (138, top))
    note(ax, 136.6, top + 2.8, "yes")
    arrow(ax, (97, top - 6), (97, 36.25))
    note(ax, 99.6, 38.2, "yes")
    arrow(ax, (107, 32.5), (113, top - 5), "done", (110, 32.5), bend=[(113, 32.5)])

    line(ax, [97, 97], [28.75, bus])
    note(ax, 97, 26.5, "still stuck")
    line(ax, [129, 129, 25], [top - 6.5, bus, bus])
    note(ax, 129, 32, "no")
    note(ax, 62, bus, "rebuild")
    line(ax, [25, 25], [top - 6.5, low])
    note(ax, 25, 32, "no")
    ax.add_patch(Circle((25, bus), 0.6, fc=INK, ec=INK, zorder=3))
    ax.add_patch(Circle((97, bus), 0.6, fc=INK, ec=INK, zorder=3))

    box(ax, 42, low, 15, 10, "Snap, cell list", "Morton order,\nadaptive grid", REBUILD)
    box(ax, 60.5, low, 16, 10, "Clip on the GPU", "certified cells,\nface rings", REBUILD, lw=2.4)
    box(ax, 80, low, 17, 10, "Exact start", "tetrahedra, exact\ncheck and patches", REBUILD)
    diamond(ax, 97, low, 9, 12, "exact?")
    box(ax, 122, low, 17, 7.5, "Refuse", "clear error, no mesh", REFUSE)
    box(ax, 145, low, 14, 10, "Geometry", "repair state,\nfaces, volumes", REBUILD)
    arrow(ax, (25, low), (34.5, low))
    arrow(ax, (49.5, low), (52.5, low))
    arrow(ax, (68.5, low), (71.5, low))
    arrow(ax, (88.5, low), (92.5, low))
    arrow(ax, (101.5, low), (113.5, low), "no", (107.5, low + 1.8))
    arrow(ax, (97, low - 6), (145, low - 5), "yes", (120, 2), bend=[(97, 2), (145, 2)])
    arrow(ax, (145, low + 5), (145, top - 5))

    for k, (tone, label) in enumerate(((REPAIR, "each step: repair"), (REBUILD, "rebuild"), (TEST, "decision"), (REFUSE, "refusal"))):
        x = 46 + k * 22
        ax.add_patch(FancyBboxPatch((x, 56.2), 2.4, 1.6, boxstyle="round,pad=0,rounding_size=0.4", fc=tone[0], ec=tone[1], lw=1.2, zorder=3))
        ax.text(x + 3.2, 57, label, va="center", fontsize=10.5, color=INK)
    fig.savefig(os.path.join(HERE, "..", "docs", "figures", "flow_mesh.png"), dpi=200)


if __name__ == "__main__":
    main()
