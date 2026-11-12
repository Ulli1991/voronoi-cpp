#!/usr/bin/env python3
"""Draws docs/figures/memory_pools.png: device memory per cell at a rebuild, a start and a step against one A100, how
each pool is sized and grown and what happens when it is short, and where 32-bit offsets overflow. Rules are read
from the code of main; bytes per cell come from docs/measurements.md, docs/measure_2026-11-10.md and the README."""
from diagram import FAINT, INK, MONO, REBUILD, REFUSE, REPAIR, SETUP, TEST, canvas, figstyle, finish, rect, save

A100 = 40e9  # bytes of one A100 40 GB
SIZES = [(128, "128³"), (256, "256³"), (384, "384³"), (512, "512³"), (768, "768³"), (1024, "1024³")]

# Peak device bytes per cell. (label, B/cell, source)
PEAKS = [
    ("plain warp rebuild", 701, "docs/measurements.md, 128³"),
    ("start, mesh only", 1728, "README, mesh within a luna-cpp run"),
    ("step peak, mesh", 1950, "docs/measure_2026-11-10.md C, 128³"),
    ("step peak with the bench", 1998, "the same, + 48 B/cell of the bench's input"),
    ("luna-cpp moving run, double", 2319, "README, the mesh and its hydro"),
]

# 32-bit (int) offsets: entries per cell from the measured bytes per cell (section C of the note).
LIMITS = [
    ("ring pool offsets (FaceRings, int)", 501 / 2 / 4, "501 B/cell, two pools, 4 B per entry"),
    ("start's ring slots (assemble, int)", 174 / 4, "174 B/cell in use, 4 B per entry"),
    ("cell_face entries (int)", (74 - 8) / 4, "74 B/cell less start and cursor"),
    ("tetrahedron slots (int)", 250 / 32, "250 B/cell, 32 B per vertex and adjacent row"),
]


def between(cells):
    """The two round sizes a cell count falls between."""
    for (a, la), (b, lb) in zip(SIZES, SIZES[1:]):
        if a ** 3 <= cells < b ** 3:
            return f"between {la} and {lb}"
    return "past 1024³" if cells >= SIZES[-1][0] ** 3 else "below 128³"


def peaks_panel(fig, rect_):
    ax = fig.add_axes(rect_)
    figstyle.decorate(ax)
    ys = list(range(len(PEAKS)))[::-1]
    colours = [REBUILD[1], REBUILD[1], REPAIR[1], REPAIR[1], TEST[1]]
    ax.barh(ys, [p[1] for p in PEAKS], color=colours, height=0.6, zorder=3)
    ax.set_yticks(ys, [p[0] for p in PEAKS], fontsize=10.5)
    ax.tick_params(axis="y", which="both", left=False, right=False)
    ax.yaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
    for y, p in zip(ys, PEAKS):
        ax.text(p[1] + (70 if p[1] < 800 or p[1] > 2300 else 30), y, f"{p[1]}", va="center", fontsize=10, color=INK)
    for edge, name in ((256, "256³"), (384, "384³")):
        b = A100 / edge ** 3
        ax.axvline(b, color=INK, lw=2.2, ls="--" if edge == 384 else ":", zorder=4)
        ax.text(b + (25 if edge == 384 else -25), 5.55, f"{name} fits\nbelow {b:.0f}", fontsize=9.5, va="top",
                ha="left" if edge == 384 else "right", color=INK)
    ax.set_xlim(0, 2900)
    ax.set_ylim(-0.6, 5.7)
    ax.set_xlabel("peak device memory, B/cell")
    return ax


def limits_panel(fig, rect_):
    ax = fig.add_axes(rect_)
    figstyle.decorate(ax)
    ax.set_xscale("log", base=2)
    ys = list(range(len(LIMITS)))[::-1]
    for y, (name, per, _) in zip(ys, LIMITS):
        cells = 2 ** 31 / per
        ax.barh(y, cells, color=REFUSE[1] if cells < 384 ** 3 else TEST[1], height=0.55, zorder=3)
        ax.text(2 ** 21 * 1.15, y + 0.42, name, fontsize=9.5, va="center", color=INK)
        ax.text(cells * 1.1, y, between(cells), fontsize=9, va="center", color=INK)
    ax.set_xticks([s ** 3 for s, _ in SIZES], [l for _, l in SIZES])
    ax.xaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
    ax.set_yticks([])
    ax.set_xlim(2 ** 21, 2 ** 34)
    ax.set_ylim(-0.5, 3.95)
    ax.set_xlabel("cells where the offset passes 2³¹")
    return ax


POOLS = [
    ("Clipping ring pool", "Cells::rings  (start only)", REBUILD,
     "64 entries per cell at first, or the last start's size; n / 1024 chunks, a chunk per cell by hash",
     "a full chunk: face_overflow. until_pool_fits builds again with need + 1/8, 3 attempts;\n"
     "Mesh::rebuild once more with need + 1/16; then refuse"),
    ("Face ring pools, two", "FaceRings::pool[2]", REPAIR,
     "each 1.5 × the start's ring entries; dirty rings are appended at the end of the pool in use",
     "full: compact every ring into the spare (realloc to entries + 1/4 if short), swap; if the emptied\n"
     "pool is the smaller, move back (entries + 1/2). The spare is scratch: Mesh::scratch grows it"),
    ("Tetrahedra", "tet_vertex, _adjacent, _centre, ...", REPAIR,
     "the start's count + 1/8; a flip reuses its old slots, then pops free_slots, then appends",
     "past the end: the flip loses and retries next round; reserve grows every per-tetrahedron\n"
     "array by max(1/8, need) with resize (old and new alive at once); stuck: + count/64 + 1024"),
    ("Faces", "owner, neighbour, ring_*, area, ...", REPAIR,
     "the start's count + 1/32",
     "renumber resizes with 1/32 more; the number pool (ids, 1/4 spare) compacts in place when full"),
    ("Cell face lists", "cell_face  (update_cells)", SETUP,
     "2 × faces + faces / 16: a 1/32 spare", "realloc to the same rule (the lists are rebuilt)"),
    ("Repair lists", "work, plans, freed, stuck", REPAIR,
     "n / 4 + 1024 entries", "find counts past the end: grow by half, run find again; stuck keeps its entries"),
    ("Created list", "touched_list", REPAIR,
     "the tetrahedron capacity / 4 + 1024", "overflow: no list this step (every face walked, update_cells in "
     "full); next time created + 1/4"),
]


def main():
    W, H = 13.4, 30
    fig, ax = canvas(W, H)
    top = H - 0.2
    ax.text(0.2, top, "(a) Peak device memory per cell, against one A100 40 GB", fontsize=12.5, fontweight="bold",
            va="top", color=INK)
    ax.text(7.2, top, "(b) 32-bit offsets: where they overflow", fontsize=12.5, fontweight="bold", va="top",
            color=INK)
    ph = 3.3
    panels = [(peaks_panel, 2.55, 4.15), (limits_panel, 7.3, 5.8)]
    panel_y = top - 0.55 - ph
    y = top - 0.55 - ph - 0.7
    ax.text(0.2, y, "Lines: 40 × 10⁹ B / cells. The start's peak of the mesh is lower than a step's; the step peak is "
            "where the arrays of the step and update_cells()\nall live at once. Offsets: 2³¹ / entries per cell, the "
            "entries from the measured bytes per cell (docs/measure_2026-11-10.md C): 62.6 per\npool, 43.5 in use, "
            "16.5 list entries and 7.8 tetrahedra per cell. Red: overflows below 384³. ideas.md lists the first and "
            "third as open.", fontsize=9.3, va="top", color=FAINT, linespacing=1.3)
    y -= 1.15
    ax.text(0.2, y, "(c) Pools: how each is sized, grown, and what happens when it is short", fontsize=12.5,
            fontweight="bold", va="top", color=INK)
    y -= 0.5
    for name, code, tone, size, short in POOLS:
        h = 0.78 + 0.2 * short.count("\n")
        rect(ax, 0.2, y - h, 2.9, h, tone, lw=1.2)
        ax.text(0.32, y - 0.12, name, fontsize=11, fontweight="bold", va="top", color=INK)
        ax.text(0.32, y - 0.42, code, fontsize=8.6, family=MONO, va="top", color=tone[1])
        ax.text(3.35, y - 0.12, "sized:  " + size, fontsize=9.8, va="top", color=INK)
        ax.text(3.35, y - 0.4, "short:  " + short.replace("\n", "\n            "), fontsize=9.8, va="top", color=INK,
                linespacing=1.3)
        y -= h + 0.15
    ax.text(0.2, y - 0.1, "Kokkos::resize keeps the contents (a copy: old and new are allocated together); realloc "
            "drops them. Sources: voronoi_mesh.hpp (rebuild, scratch, geometry),\ntet_mesh.hpp (make_face_rings, "
            "update_face_rings), tet_repair.hpp (prepare, reserve, repair), tet_cells.hpp (cells), "
            "shell_voronoi.hpp (Cells, claim, until_pool_fits).", fontsize=9.3, va="top", color=FAINT,
            linespacing=1.3)
    bottom = y - 0.75
    finish(fig, ax, bottom)
    height = H - bottom
    for draw, x, w in panels:  # in figure fractions of the final size
        draw(fig, [x / W, (panel_y - bottom) / height, w / W, ph / height])
    save(fig, "memory_pools.png")


if __name__ == "__main__":
    main()
