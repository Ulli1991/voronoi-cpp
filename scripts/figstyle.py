# Copied unchanged from LUNA, src/luna/utils/figstyle.py at 60cf841 (file last changed in 15e1c02). Do not edit here.
"""Paper-figure style: apply_style() before plotting, decorate(ax) per axes; figures carry no titles."""
import matplotlib as mpl

# single-panel size; N-column multipanels use (N*W, H)
FIGSIZE = (6.4, 4.8)

_GRID = dict(color="0.80", linewidth=0.6, linestyle="-", alpha=0.7)


def apply_style():
    """Set global rcParams for the LUNA paper figure style."""
    mpl.rcParams.update({
        # ---- figure / saving ----
        "figure.figsize": FIGSIZE,
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        # ---- fonts ----
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        # ---- ticks: inward, on all four sides, with minor ticks ----
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "xtick.bottom": True,
        "ytick.left": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 6.0,
        "ytick.major.size": 6.0,
        "xtick.minor.size": 3.0,
        "ytick.minor.size": 3.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.7,
        "ytick.minor.width": 0.7,
        # ---- coarse background grid (major lines only) ----
        "axes.grid": True,
        "axes.grid.which": "major",
        "axes.axisbelow": True,
        "grid.color": _GRID["color"],
        "grid.linewidth": _GRID["linewidth"],
        "grid.linestyle": _GRID["linestyle"],
        "grid.alpha": _GRID["alpha"],
        # ---- legend: large, no box ----
        "legend.frameon": False,
        "legend.fontsize": 13,
        "legend.handlelength": 1.8,
        "legend.labelspacing": 0.4,
        # ---- axes / lines ----
        "axes.linewidth": 1.1,
        "lines.linewidth": 1.8,
    })


def decorate(ax):
    """Apply the tick and grid style to one axes, including twin axes (idempotent)."""
    ax.tick_params(which="both", direction="in",
                   top=True, bottom=True, left=True, right=True)
    ax.minorticks_on()
    ax.grid(True, which="major", color=_GRID["color"],
            linewidth=_GRID["linewidth"], linestyle=_GRID["linestyle"],
            alpha=_GRID["alpha"])
    ax.set_axisbelow(True)


def legend(ax, **kwargs):
    """Draw a frameless, large legend (kwargs override)."""
    opts = dict(frameon=False, fontsize=13)
    opts.update(kwargs)
    return ax.legend(**opts)
