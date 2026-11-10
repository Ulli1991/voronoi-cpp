"""The memory order and numbering the figures name (data_figure.py, access_figures.py). The rules themselves are
written once, in docs/how_it_works.md, section 12 ("Memory order and numbering"); a change of order updates that
section and these labels, then the figures are rendered again. voronoi::Layout (#34) is off by default."""

PARTICLE = "particle id (caller's order; a curve's after a re-sort)"
TET = "tetrahedron slot (by lowest particle after a re-sort)"
FACE = "face number (scan over edge bits, or by owner)"
