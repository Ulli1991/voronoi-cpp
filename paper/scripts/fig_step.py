#!/usr/bin/env python3
"""Figure mesh_update: the speed of the mesh update (the mesh alone, no solver), four panels. (a) against problem size,
per input, at 0.1 spacings per step; (b) at 128³ against how far the points move per step, with the exact start as the
cost of building again; (c) over 500 steps with and without the re-sort; (d) against the mesh update of the other codes
that have one. Data and drawing: fig_sweep.py, fig_resort.py, fig_library.py."""
import sys

import fig_library
import fig_resort
import fig_sweep
from paperfig import DOUBLE, figure, save


def main():
    _, table, _ = fig_sweep.load(sys.argv[1] if len(sys.argv) > 1 else None)
    fig, ((a, b), (c, d)) = figure(DOUBLE, 150, nrows=2, ncols=2)
    fig_sweep.step_vs_n(a, table)
    fig_sweep.displacement(b, table)
    fig_resort.resort(c)
    fig_library.step_codes(d, fig_library.load()[0])
    fig.tight_layout()
    save(fig, "mesh_update", DOUBLE)


if __name__ == "__main__":
    main()
