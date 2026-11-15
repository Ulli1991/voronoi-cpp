#!/usr/bin/env python3
"""Paper versions of the walk-through (scripts/overview_figure.py, unchanged): walkthrough_build (panels 1-6) and
walkthrough_step (7-12), the library's own 2D arrays, as 190 mm vector PDFs (text 7.3-9.6 pt) and 600 dpi PNGs.
Needs g++ 15 and Kokkos (OpenMP) for the 2D driver, as overview_figure.py; one core."""
import io
import os
import sys

from paperfig import DOUBLE, MM, OUT, REPO

sys.path.insert(0, os.path.join(REPO, "scripts"))
import overview_figure as of  # noqa: E402

NAMES = {"overview_build": "walkthrough_build", "overview_step": "walkthrough_step"}


def save(self, name):
    """Sheet.save for print: the PDF scaled to 190 mm, the PNG at 600 dpi of that width."""
    from pypdf import PdfReader, PdfWriter

    name = NAMES[name]
    buf = io.BytesIO()
    self.fig.savefig(buf, format="pdf")
    page = PdfReader(buf).pages[0]
    page.scale_by(DOUBLE * MM * 72 / float(page.mediabox.width))
    writer = PdfWriter()
    writer.add_page(page)
    with open(os.path.join(OUT, name + ".pdf"), "wb") as f:
        writer.write(f)
    self.fig.savefig(os.path.join(OUT, name + ".png"), dpi=600 * DOUBLE * MM / self.width)
    of.plt.close(self.fig)
    print(f"{name}: {DOUBLE:.0f} mm x {float(page.mediabox.height) / 72 / MM:.0f} mm, scale {DOUBLE * MM / self.width:.3f}")


if __name__ == "__main__":
    of.BUILD = os.path.join(REPO, "build-paper-overview")
    of.Sheet.save = save
    of.main(sys.argv[1:])
