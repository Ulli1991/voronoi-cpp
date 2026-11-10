"""Shared drawing helpers for the explanatory figures (flow_detail.py, data_figure.py, access_figures.py,
memory_figure.py): LUNA's figstyle, boxes with a GPU or host badge, arrows, wait markers. Units are inches."""
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "figures")
_spec = importlib.util.spec_from_file_location("figstyle", os.path.join(HERE, "figstyle.py"))
figstyle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(figstyle)
figstyle.apply_style()

INK, FAINT = "#222222", "#5f5f5f"
SETUP = ("#eef1f5", "#7c8796")
REPAIR = ("#dff0dc", "#2ca02c")
REBUILD = ("#dbe9f6", "#1f77b4")
TEST = ("#fde8d3", "#ff7f0e")
REFUSE = ("#f8dcdc", "#d62728")
HOSTC = ("#efe6f6", "#9467bd")
WAIT = "#d62728"
MONO = "DejaVu Sans Mono"


def canvas(width, height):
    """A figure of width x height inches whose data units are inches, axes hidden."""
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.minorticks_off()
    return fig, ax


def rect(ax, x, y, w, h, tone, lw=1.4, radius=0.08, z=3):
    """Rounded box with its lower-left corner at (x, y)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", fc=tone[0], ec=tone[1],
                                lw=lw, zorder=z))


def badge(ax, x, y, where):
    """GPU or host label, right-aligned at (x, y)."""
    tone = REBUILD if where == "GPU" else HOSTC
    ax.text(x, y, where, ha="right", va="top", fontsize=9.5, fontweight="bold", color=tone[1], zorder=6,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=tone[1], lw=1.0))


def card_height(code, lines, size=10.5):
    """Height in inches that card() needs for this text."""
    line = size * 1.3 / 72
    return 0.5 + (0.27 * (code.count("\n") + 1) if code else 0) + (line * (lines.count("\n") + 1) if lines else 0) + 0.06


def finish(fig, ax, bottom, top=None):
    """Crop the canvas to [bottom, top] inches (top: the current upper limit), keeping inches as data units."""
    top = ax.get_ylim()[1] if top is None else top
    ax.set_ylim(bottom, top)
    fig.set_size_inches(fig.get_size_inches()[0], top - bottom)


def card(ax, x, y, w, h, title, code, lines, tone, where=None, size=10.5):
    """Box at lower-left (x, y): bold title, a monospace code reference, then plain lines, top-aligned."""
    rect(ax, x, y, w, h, tone)
    ax.text(x + 0.12, y + h - 0.1, title, ha="left", va="top", fontsize=size + 1.5, fontweight="bold", color=INK,
            zorder=5)
    top = y + h - 0.42
    if code:
        ax.text(x + 0.12, top, code, ha="left", va="top", fontsize=size - 0.5, family=MONO, color=tone[1], zorder=5)
        top -= 0.27 * (code.count("\n") + 1)
    if lines:
        ax.text(x + 0.12, top, lines, ha="left", va="top", fontsize=size, color=INK, linespacing=1.3, zorder=5)
    if where:
        badge(ax, x + w - 0.08, y + h - 0.08, where)


def arrow(ax, points, color=INK, lw=1.6, label=None, at=None, size=10):
    """Polyline through `points` with an arrowhead at the last one; optional italic label at `at`."""
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot(*zip(a, b), color=color, lw=lw, solid_capstyle="round", zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=14, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=2))
    if label:
        ax.text(*at, label, ha="center", va="center", fontsize=size, color=FAINT, style="italic", zorder=5,
                bbox=dict(fc="white", ec="none", pad=1.0))


def wait(ax, x, y, text, size=9.5):
    """A host-waits marker: red bar symbol and text, left-aligned at (x, y)."""
    ax.add_patch(Polygon([(x, y - 0.09), (x + 0.05, y - 0.09), (x + 0.05, y + 0.09), (x, y + 0.09)], fc=WAIT, ec=WAIT,
                         zorder=6))
    ax.add_patch(Polygon([(x + 0.09, y - 0.09), (x + 0.14, y - 0.09), (x + 0.14, y + 0.09), (x + 0.09, y + 0.09)],
                         fc=WAIT, ec=WAIT, zorder=6))
    ax.text(x + 0.22, y, text, ha="left", va="center", fontsize=size, color=WAIT, zorder=6,
            bbox=dict(fc="white", ec="none", pad=0.6))


def diamond(ax, x, y, w, h, text, size=10):
    """Decision at centre (x, y)."""
    ax.add_patch(Polygon([(x - w / 2, y), (x, y + h / 2), (x + w / 2, y), (x, y - h / 2)], closed=True, fc=TEST[0],
                         ec=TEST[1], lw=1.4, zorder=3, joinstyle="round"))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.1, zorder=4)


def legend_chips(ax, x, y, items, step=2.2, size=10):
    """Colour chips with labels in one row from (x, y)."""
    for k, (tone, label) in enumerate(items):
        rect(ax, x + k * step, y - 0.08, 0.28, 0.18, tone, lw=1.0, radius=0.04)
        ax.text(x + k * step + 0.36, y, label, va="center", fontsize=size, color=INK)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
