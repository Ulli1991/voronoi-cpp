"""Paper sizes for LUNA's figstyle (scripts/figstyle.py, unchanged): a figure is drawn at a design width with
figstyle's fonts, and the saved PDF is scaled to CPC's column (90 mm) or page width (190 mm), which puts its 12-14 pt
text at 7.6-8.9 pt in print; the PNG has 600 dpi at print size. Data under paper/data, figures under paper/figures."""
import importlib.util
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(REPO, "paper", "data")
OUT = os.path.join(REPO, "paper", "figures")
_spec = importlib.util.spec_from_file_location("figstyle", os.path.join(REPO, "scripts", "figstyle.py"))
figstyle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(figstyle)

MM = 1 / 25.4
SINGLE, DOUBLE = 90.0, 190.0  # print widths, mm
SCALE = 0.633  # print / design: figstyle's 12 pt ticks print at 7.6 pt
LW = 2.2  # curves, design pt
# one colour and marker per input, the same in every figure
INPUTS = ["jittered", "bluenoise", "uniform", "clustered", "halo", "lattice", "wave"]
LABEL = {"jittered": "jittered", "bluenoise": "blue noise", "uniform": "white noise", "clustered": "clustered",
         "halo": "halo", "lattice": "lattice", "wave": "wave", "slab": "slab", "cosmicweb": "cosmic web",
         "fccsolid": "fcc solid", "surface": "surface"}
# One colour per input in every figure (owner, 2026-11-16), from the tab10 cycle that LUNA's figstyle uses, but surface
# in dark teal (owner): CIELAB distance to blue noise's cyan and jittered's blue 27-34 in normal, deuteranope and
# protanope simulation (Machado 2009), 8 to blue for tritanopes, where its dashed curve tells them apart. Curves also differ in marker or dash where inputs meet in
# one panel, for readers who cannot tell red from green.
COLOR = {"jittered": "#1f77b4", "bluenoise": "#17becf", "uniform": "#2ca02c", "clustered": "#ff7f0e",
         "halo": "#d62728", "lattice": "#9467bd", "wave": "#8c564b", "cosmicweb": "#e377c2", "fccsolid": "#bcbd22",
         "surface": "#00727a", "slab": "#7f7f7f"}
MARKER = {"jittered": "o", "bluenoise": "D", "uniform": "s", "clustered": "^", "halo": "v", "lattice": "P",
          "wave": "X", "slab": "h", "cosmicweb": "*", "fccsolid": "p", "surface": "d"}


# Compact legends, the paper's rule (PLAN.md): inside the axes, tight handles and spacing, an opaque white box without
# an edge so no grid or data line crosses the text.
LEGEND = {"legend.frameon": True, "legend.facecolor": "white", "legend.edgecolor": "none", "legend.framealpha": 1.0,
          "legend.fancybox": False, "legend.handlelength": 1.3, "legend.handletextpad": 0.4,
          "legend.columnspacing": 0.9, "legend.labelspacing": 0.25, "legend.borderpad": 0.2,
          "legend.borderaxespad": 0.4}


def style():
    """LUNA's figstyle (unchanged), then the compact legend settings."""
    figstyle.apply_style()
    plt.rcParams.update(LEGEND)


def figure(width_mm, height_mm, **kw):
    """A figure of the given print size, drawn at design size (print / SCALE)."""
    style()
    return plt.subplots(figsize=(width_mm * MM / SCALE, height_mm * MM / SCALE), **kw)


# The paper's order (paper/figures/README.md): figure name in the scripts -> file name. A letter marks the parts of one figure.
NUMBER = {
    "walkthrough": "fig01_build",
    "construction_grid": "fig02_grid",
    "construction_clipping": "fig03_clipping",
    "construction_exact": "fig04_exact_ties",
    "inputs": "fig05_inputs",
    "library_headline": "fig06_speed_full_cells",
    "correctness": "fig07_correctness",
    "throughput_vs_n": "fig08_build_vs_n",
    "memory": "fig09_memory",
    "walkthrough_step": "fig10_moving_step",
    "mesh_update": "fig11_mesh_update",
    "titarev_toro": "fig12_titarev_toro",
    "luna_kh": "fig13_luna_kh",
    "luna_vortex": "fig14_luna_vortex",
}


def save(fig, name, width_mm):
    """paper/figures/<numbered name>.pdf scaled to width_mm exactly, and the .png at 600 dpi of that width."""
    name = NUMBER[name]
    os.makedirs(OUT, exist_ok=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    from pypdf import PdfReader, PdfWriter

    page = PdfReader(buf).pages[0]
    page.scale_by(width_mm * MM * 72 / float(page.mediabox.width))
    writer = PdfWriter()
    writer.add_page(page)
    with open(os.path.join(OUT, name + ".pdf"), "wb") as f:
        writer.write(f)
    design = fig.get_size_inches()[0]
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=600 * width_mm * MM / design)
    plt.close(fig)
    print(f"{name}: {width_mm:.0f} mm x {float(page.mediabox.height) / 72 / MM:.0f} mm")
