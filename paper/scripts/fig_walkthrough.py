#!/usr/bin/env python3
"""Paper version of the walk-through (scripts/overview_figure.py, unchanged): one full-page figure "walkthrough", rows
1-2 the build (panels 1-6), rows 3-4 the moving step (7-12), labelled "Build" and "Step", from the library's own 2D
arrays; a 190 mm vector PDF and a 600 dpi PNG. The panel texts go to walkthrough_captions.md. Needs g++ 15 and Kokkos
(OpenMP) for the 2D driver, as overview_figure.py; one core."""
import io
import os
import sys

from matplotlib.transforms import Bbox

from paperfig import DOUBLE, MM, NUMBER, OUT, REPO

sys.path.insert(0, os.path.join(REPO, "scripts"))
import overview_figure as of  # noqa: E402

LEFT = 0.05  # inches left of the first column
PAIR = 0.20  # extra space between the build and the step rows, inches
SHEET = {}  # the one sheet both of overview_figure's figures draw on
_init = of.Sheet.__init__
# A brief title beside each panel's number (the full sentence stays in the caption).
TITLE = {"1": "Integer positions", "2": "Morton cell list", "3": "Clip one cell", "4": "All cells",
         "5": "Exact Delaunay mesh", "6": "Points move", "7": "Check", "8": "Flip", "9": "Geometry",
         "10": "Per-cell results", "11": "Fallbacks", "12": "Re-sort"}
CAPTIONS = []  # (panel number, title, text, code) of every panel, for the captions
_panel = of.Sheet.panel
WHERE = {}  # panel number -> where it runs (GPU, host, caller): for the caption


# The paper's terms (owner): positions are "quantised to integer coordinates on a 2^30 grid", then "quantised" or
# "integer positions"; the word "snap" is not used, not even for the functions' names.
WORDS = {
    "Snap to the grid": "Quantise positions",
    "Positions go down to an integer grid of 2³⁰ steps per side (inset: one grid cell, much enlarged). "
    "Every later decision is an exact test on these integers;":
        "Each coordinate is rounded down to an integer k in [0, 2³⁰), and the position is k × step: positions are "
        "quantised to integer coordinates on a 2³⁰ grid (inset: one grid cell, much enlarged). Every later decision "
        "is an exact test on these integer positions;",
    "exact_predicates.hpp: snap\nvoronoi_mesh_2d.hpp: Mesh::snap": "exact_predicates.hpp, voronoi_mesh_2d.hpp (quantisation)",
    "Two on one grid point: refused.": "Two at one integer position: refused.",
    "4 (16²) in the patch, 3 (8²) around.": "4 (16², dark blue) in the patch, 3 (8², light blue) around.",
}


def words(text):
    for old, new in WORDS.items():
        text = text.replace(old, new)
    return text


def panel(self, row, col, number, title, where, body, code, window=(0, 1, 0, 1)):
    """Sheet.panel with everything written left to the caption (title, where it runs, text, code): number and drawing only."""
    title, body, code = words(title), words(body), words(code)
    CAPTIONS.append((number, title, body, code))
    rowh = self.HEAD + self.S + self.TEXT + self.GAP
    if int(number) > 6:  # the step: rows 3-4, below a small extra gap
        row = row + 2 + PAIR / rowh
    WHERE[number] = where
    shown = number if int(number) <= 6 else str(int(number) - 6)  # the step is a figure of its own: its panels 1-6
    return _panel(self, row, col + LEFT / self.W, shown, TITLE[number], (), "", "", window)


def sheet(self, rows, cols):
    """Both figures draw on one sheet of four rows; the second Sheet() returns it."""
    if SHEET:
        self.__dict__ = SHEET["sheet"].__dict__
        return
    _init(self, 2 * rows, cols)
    self.height += PAIR
    self.width += LEFT
    self.fig.set_size_inches(self.width, self.height)
    SHEET["sheet"] = self


def save(self, name):
    """Sheet.save for print, once both halves are drawn: each half a PDF scaled to 190 mm and a PNG at 600 dpi of that width."""
    from pypdf import PdfReader, PdfWriter

    if name != "overview_step":
        return
    for ax in self.fig.axes:  # panel 2's level key sits on the drawing; the caption names the levels instead
        if ax.get_legend():
            ax.get_legend().remove()
    # Two figures from the one sheet: the build (rows 1-2) and the moving step (rows 3-4), cut between the pairs.
    cut = self.height - 2 * (self.HEAD + self.S + self.TEXT + self.GAP) - 0.8 * PAIR
    for key, low, high in (("walkthrough", cut, self.height), ("walkthrough_step", 0.0, cut)):
        box = Bbox([[0.0, low], [self.width, high]])
        buf = io.BytesIO()
        self.fig.savefig(buf, format="pdf", bbox_inches=box)
        page = PdfReader(buf).pages[0]
        page.scale_by(DOUBLE * MM * 72 / float(page.mediabox.width))
        writer = PdfWriter()
        writer.add_page(page)
        with open(os.path.join(OUT, NUMBER[key] + ".pdf"), "wb") as f:
            writer.write(f)
        self.fig.savefig(os.path.join(OUT, NUMBER[key] + ".png"), dpi=600 * DOUBLE * MM / self.width, bbox_inches=box)
        print(f"{NUMBER[key]}: {DOUBLE:.0f} mm x {float(page.mediabox.height) / 72 / MM:.0f} mm")
    of.plt.close(self.fig)


def captions():
    """paper/figures/walkthrough_captions.md: each panel's text (as drawn in docs/figures) and its code."""
    with open(os.path.join(OUT, "walkthrough_captions.md"), "w") as f:
        f.write("Panel texts of the walkthrough figure, moved out of the figure (generated by fig_walkthrough.py).\n\n")
        for number, title, body, code in CAPTIONS:
            f.write(f"- **({number}) {title}.** {body} *Code:* {code.replace(chr(10), '; ')}\n")


if __name__ == "__main__":
    of.BUILD = os.path.join(REPO, "build-paper-overview")
    of.Sheet.save = save
    of.Sheet.panel = panel
    of.Sheet.__init__ = sheet
    of.Sheet.HEAD, of.Sheet.TEXT, of.Sheet.GAP = 0.32, 0.02, 0.06
    of.main(sys.argv[1:])
    captions()
