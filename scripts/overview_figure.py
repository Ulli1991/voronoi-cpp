#!/usr/bin/env python3
"""Draws docs/figures/overview_build.{png,pdf} and overview_step.{png,pdf}: one small periodic point set taken through
every stage of voronoi::Mesh<2>, panel by panel. The arrays come from the library itself (scripts/overview_mesh2d.cpp,
built here against Kokkos on the host); the walk of one cell and the Morton cell list are the ports of
construction_figures.py, checked against the library's rings and slots. Usage: overview_figure.py [--data FILE]."""
import argparse
import importlib.util
import math
import os
import subprocess
import sys
import textwrap

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "docs", "figures")
BUILD = os.path.join(ROOT, "build-overview")
KOKKOS = os.environ.get("KOKKOS_ROOT", "/u/uli/kokkos-gcc15")
spec = importlib.util.spec_from_file_location("construction", os.path.join(HERE, "construction_figures.py"))
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)  # applies the LUNA figure style
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams["pdf.fonttype"] = 42  # TrueType in the PDF, as journals ask; the LUNA style is otherwise unchanged
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

INK, FAINT = cf.INK, cf.FAINT
BLUE, GREEN, ORANGE, RED = cf.REBUILD[1], cf.REPAIR[1], cf.TEST[1], cf.REFUSE[1]
N = 96  # 3 x 32: 64 in a dense patch (a node of 64 or more refines its level), 32 around it
SHIFTS = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]


# ---- the point set: ids along a Hilbert curve at the start, as a caller that sorts once would number them ----
def hilbert(x, y, bits):
    """Index of cell (x, y) on a 2D Hilbert curve of 2^bits cells per side (the 2D form of mesh_order.hpp's key)."""
    d, s = 0, 1 << (bits - 1)
    while s > 0:
        rx, ry = int(x & s > 0), int(y & s > 0)
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x, y = s - 1 - x, s - 1 - y
            x, y = y, x
        s >>= 1
    return d


def hilbert_order(xy):
    keys = [hilbert(int(x * 1024), int(y * 1024), 10) for x, y in xy]
    return sorted(range(len(xy)), key=lambda i: (keys[i], i))


def darts(rng, count, low, high, gap, inside):
    """`count` random points with no two closer than `gap`, from the region `inside` of [low, high)^2."""
    kept = []
    while len(kept) < count:
        p = low + (high - low) * rng.random(2)
        if inside(p) and all(np.sum(np.minimum(abs(p - q), 1 - abs(p - q)) ** 2) > gap * gap for q in kept):
            kept.append(p)
    return np.array(kept)


def point_set():
    rng = np.random.default_rng(12)
    patch = darts(rng, 64, 0.03, 0.47, 0.038, lambda p: True)
    rest = darts(rng, 32, 0.0, 1.0, 0.125, lambda p: not (p < 0.5).all() and all(np.hypot(*(p - q)) > 0.09 for q in patch))
    x0 = np.concatenate([patch, rest])
    return x0[hilbert_order(x0)]


def flow(x, amplitude):
    return np.stack([np.sin(2 * np.pi * x[:, 1] + 0.4), np.sin(2 * np.pi * x[:, 0] + 1.1)], axis=1) * amplitude


def positions():
    x0 = point_set()
    x1 = (x0 + flow(x0, 0.018)) % 1.0
    x2 = x1.copy()
    rng = np.random.default_rng(8)
    inner = [i for i in range(N) if ((x1[i] > 0.12) & (x1[i] < 0.38)).all()]
    jumped = rng.choice(inner, 2, replace=False)
    for i in jumped:  # two particles of the patch jump across their neighbours
        x2[i] = (x2[i] + 0.03 * rng.standard_normal(2)) % 1.0
    late = x0.copy()
    for _ in range(16):  # many steps of an area-preserving flow (two shears), ids kept
        late[:, 0] = (late[:, 0] + 0.09 * np.sin(2 * np.pi * late[:, 1] + 0.4)) % 1.0
        late[:, 1] = (late[:, 1] + 0.09 * np.sin(2 * np.pi * late[:, 0] + 1.1)) % 1.0
    return dict(x0=x0, x1=x1, x2=x2, late=late), sorted(int(i) for i in jumped)


# ---- the library run ----
def build_driver():
    exe = os.path.join(BUILD, "overview_mesh2d")
    source = os.path.join(HERE, "overview_mesh2d.cpp")
    headers = [os.path.join(ROOT, "src", d) for d in ("voronoi_construct", "neighbour_search")]
    newest = max(os.path.getmtime(os.path.join(d, f)) for d in headers for f in os.listdir(d))
    if os.path.exists(exe) and os.path.getmtime(exe) > max(newest, os.path.getmtime(source)):
        return exe
    os.makedirs(BUILD, exist_ok=True)
    cmd = [os.environ.get("CXX", "g++"), "-std=c++20", "-O2", "-fopenmp", "-Wall", "-Wextra", *("-I" + h for h in headers),
           "-I" + os.path.join(KOKKOS, "include"), source, "-o", exe, "-L" + os.path.join(KOKKOS, "lib64"), "-lkokkoscore", "-ldl"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return exe


def run_library(sets, data):
    exe = build_driver()
    given = os.path.join(BUILD, "overview_in.txt")
    with open(given, "w") as f:
        for name, x in sets.items():
            f.write(f"{name} {len(x)}\n")
            f.writelines(f"{a:.17g} {b:.17g}\n" for a, b in x)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    subprocess.run([exe, given, data, "--kokkos-num-threads=1"], check=True, env=env)


def read_data(path):
    blocks = {}
    with open(path) as f:
        lines = f.read().split("\n")
    i = 0
    while i < len(lines) and lines[i]:
        head = lines[i].split()
        if head[1] == "text":
            blocks[head[0]] = lines[i + 1]
            i += 2
            continue
        rows, cols = int(head[1]), int(head[2])
        values = np.array([[float(v) for v in lines[i + 1 + r].split()] for r in range(rows)]).reshape(rows, cols)
        blocks[head[0]] = values[0, 0] if (rows, cols) == (1, 1) else values[:, 0] if cols == 1 else values
        i += 1 + rows
    return blocks


def ints(v):
    return np.asarray(v).astype(np.int64)


# ---- geometry on the periodic grid ----
class Run:
    """The library's arrays, with the helpers every panel shares: wrapped offsets, triangles and cells in box units."""

    def __init__(self, d, jumped):
        self.d, self.jumped = d, jumped
        self.period = int(d["period"])
        self.k = ints(d["snap"])
        self.xy = self.k / self.period
        self.n = len(self.k)

    def offset(self, k, i, j):
        """Grid offset from particle i to the nearest image of j, at snapped positions k."""
        p = self.period
        return (k[j] - k[i] + p // 2) % p - p // 2

    def triangles(self, corners, k):
        """Each triangle in box units, unwrapped around its corner 0."""
        out = []
        for row in ints(corners):
            a, b, c = row[:3]
            base = k[a] / self.period
            out.append(np.array([base, base + self.offset(k, a, b) / self.period, base + self.offset(k, a, c) / self.period]))
        return out

    def edges(self, corners, k):
        """Each edge once, as (pair, segment in box units)."""
        seen = {}
        for row, tri in zip(ints(corners), self.triangles(corners, k)):
            for m in range(3):
                pair = tuple(sorted((row[(m + 1) % 3], row[(m + 2) % 3])))
                seen.setdefault(pair, np.array([tri[(m + 1) % 3], tri[(m + 2) % 3]]))
        return seen

    def cells(self, corners, centre, k):
        """Each cell's polygon in box units, from the circumcentres of the triangles around it."""
        rows = ints(corners)
        around = [[] for _ in range(self.n)]
        for t, row in enumerate(rows):
            for v in row[:3]:
                around[v].append(t)
        polygons = []
        for i in range(self.n):
            here = k[i] / self.period
            vs = np.array([here + (np.asarray(centre[t]) - here + 0.5) % 1.0 - 0.5 for t in around[i]])
            angle = np.arctan2(vs[:, 1] - here[1], vs[:, 0] - here[0])
            polygons.append(vs[np.argsort(angle)])
        return polygons


def tiled(shapes):
    return [s + np.array(shift) for s in shapes for shift in SHIFTS]


def circumcentre(p, q, r):
    a, b = q - p, r - p
    d = 2 * (a[0] * b[1] - a[1] * b[0])
    la, lb = a @ a, b @ b
    return p + np.array([b[1] * la - a[1] * lb, a[0] * lb - b[0] * la]) / d


def check_ports(run):
    """The ports of construction_figures.py against the library: slots, levels and the rings of every cell."""
    d = run.d
    cl = cf.CellList([tuple(int(v) for v in p) for p in run.k])
    assert cl.levels == int(d["levels"]), (cl.levels, d["levels"])
    assert cl.order == list(ints(d["order"])), "slot order"
    assert cl.level == list(ints(d["level"])), "levels"
    rings = library_rings(run)
    for t in range(run.n):
        clip = cf.Clipper(cl, t)
        mine = [clip.particle(s) for _, s in clip.edges]
        theirs = rings[clip.id]
        turn = theirs.index(mine[0])
        assert theirs[turn:] + theirs[:turn] == mine, ("ring", clip.id, mine, theirs)
    return cl


def library_rings(run):
    d = run.d
    start, degree, ring = ints(d["ring_start"]), ints(d["ring_degree"]), ints(d["ring"])
    assert (ints(d["ring_status"]) == 0).all()
    return [list(ring[start[i]:start[i] + degree[i]]) for i in range(run.n)]


# ---- panels ----
TITLE, BODY, CODE = 12.5, 10.5, 9.5
WRAP = 41


class Sheet:
    """A grid of panels: a title line with GPU/host badges, a square drawing, and three to four lines of text."""
    W, S, HEAD, TEXT, GAP = 3.25, 2.95, 0.36, 1.45, 0.16

    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols
        self.height = rows * (self.HEAD + self.S + self.TEXT + self.GAP)
        self.width = cols * self.W
        self.fig = plt.figure(figsize=(self.width, self.height))

    def panel(self, row, col, number, title, where, body, code, window=(0, 1, 0, 1)):
        x0 = col * self.W + (self.W - self.S) / 2
        top = self.height - row * (self.HEAD + self.S + self.TEXT + self.GAP)
        fig = self.fig
        head = fig.add_axes([x0 / self.width, (top - self.HEAD) / self.height, self.S / self.width, self.HEAD / self.height])
        head.axis("off")
        head.text(0, 0.45, f"{number}  {title}", fontsize=TITLE, fontweight="bold", va="center", color=INK)
        x = 1.0
        for label in reversed(where):
            dark = label == "GPU"
            w = 0.045 + 0.028 * len(label)
            x -= w
            head.add_patch(FancyBboxPatch((x, 0.18), w - 0.01, 0.54, boxstyle="round,pad=0,rounding_size=0.06",
                                          fc=INK if dark else "white", ec=INK, lw=1.0, transform=head.transAxes))
            head.text(x + (w - 0.01) / 2, 0.45, label, fontsize=CODE, fontweight="bold", ha="center", va="center",
                      color="white" if dark else INK)
            x -= 0.015
        ax = fig.add_axes([x0 / self.width, (top - self.HEAD - self.S) / self.height, self.S / self.width, self.S / self.height])
        cf.sketch(ax)
        ax.set_xlim(window[0], window[1])
        ax.set_ylim(window[2], window[3])
        for side in ax.spines.values():
            side.set_visible(True)
            side.set_color("0.55")
            side.set_linewidth(0.8)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.minorticks_off()
        ax.grid(False)
        text = fig.add_axes([x0 / self.width, (top - self.HEAD - self.S - self.TEXT) / self.height, self.S / self.width,
                             self.TEXT / self.height])
        text.axis("off")
        lines = textwrap.wrap(body, WRAP)
        if len(lines) > 5 or any(len(c) > 37 for c in code.split("\n")):
            raise ValueError(f"panel {number}: text too long: {lines} {code}")
        text.text(0, 0.93, "\n".join(lines), fontsize=BODY, va="top", color=INK, linespacing=1.25)
        text.text(0, 0.0, code, fontsize=CODE, va="bottom", color=FAINT, family="monospace", linespacing=1.2)
        return ax

    def save(self, name):
        for suffix in ("png", "pdf"):
            self.fig.savefig(os.path.join(OUT, f"{name}.{suffix}"), dpi=200)
        plt.close(self.fig)


def dots(ax, xy, size=3.2, color=INK, zorder=5, **kwargs):
    pts = np.concatenate([xy + np.array(s) for s in SHIFTS])
    ax.plot(*pts.T, marker="o", ms=size, mfc=color, mec="white", mew=0.4, ls="none", zorder=zorder, **kwargs)


def segments(ax, segs, **kwargs):
    ax.add_collection(LineCollection(tiled(list(segs)), **kwargs))


def polygons(ax, polys, **kwargs):
    ax.add_collection(PolyCollection(tiled(list(polys)), **kwargs))


def build_figure(run, cl, star):
    d, k, xy, n, P = run.d, run.k, run.xy, run.n, run.period
    sheet = Sheet(2, 3)
    rings = library_rings(run)
    built = run.triangles(d["built_corners"], k)

    # 1: snap
    ax = sheet.panel(0, 0, "1", "Snap to the grid", ["GPU"],
                     "Positions go down to an integer grid of 2³⁰ steps per side (inset: one grid cell, much enlarged). "
                     "Every later decision is an exact test on these integers; exact ties break by particle id.",
                     "exact_predicates.hpp: snap\nvoronoi_mesh_2d.hpp: Mesh::snap")
    dots(ax, xy, size=3.6)
    ax.add_patch(Rectangle((0.555, 0.555), 0.42, 0.42, fc="white", ec=INK, lw=1.0, zorder=7))
    inset = ax.inset_axes([0.555, 0.555, 0.42, 0.42])
    inset.set_zorder(8)
    cf.sketch(inset)
    steps = 8  # drawn grid: 8 steps (the code has 2^30)
    inset.set_xlim(-0.3, 2.3)
    inset.set_ylim(-0.3, 2.3)
    for v in range(3):
        inset.plot([v, v], [-0.3, 2.3], color="0.6", lw=0.8)
        inset.plot([-0.3, 2.3], [v, v], color="0.6", lw=0.8)
    true, grid = np.array([1.62, 1.35]), np.array([1.0, 1.0])
    inset.plot(*true, marker="o", ms=6, mfc="white", mec=INK, mew=1.2, zorder=4)
    cf.arrow(inset, true, grid + 0.06, lw=1.2, scale=10)
    inset.plot(*grid, marker="o", ms=6.5, mfc=INK, mec="white", zorder=5)
    inset.text(1.72, 1.5, "x", fontsize=BODY, ha="center", color=INK)
    inset.text(0.1, 0.35, "k = ⌊x/step⌋", fontsize=CODE, color=INK)
    del steps

    # 2: cell list
    levels = sorted(set(cl.level))
    shade = {lv: c for lv, c in zip(levels, ("#dbe9f6", "#9ec5e8", "#5b9bd5"))}
    ax = sheet.panel(0, 1, "2", "Morton cell list", ["GPU"],
                     f"Sorted along a Morton curve (line, slot 0 to {n - 1}): each tree node is one run of slots. Each "
                     f"particle's level has about one particle per cell: {levels[-1]} ({1 << levels[-1]}²) in the patch, "
                     f"{levels[0]} ({1 << levels[0]}²) around.",
                     "neighbor_search.hpp: configure, build")
    for q in sorted(range(n), key=lambda q: cl.level[q]):
        lv = cl.level[q]
        a, b = (int(v) >> (30 - lv) for v in k[cl.order[q]])
        w = 1.0 / (1 << lv)
        ax.add_patch(Rectangle((a * w, b * w), w, w, fc=shade[lv], ec="white", lw=0.6, zorder=1))
    pts = xy[cl.order]
    curve = LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), cmap="magma", norm=plt.Normalize(0, 1.25 * n),
                           linewidths=1.3, zorder=3)
    curve.set_array(np.arange(n - 1))
    ax.add_collection(curve)
    ax.plot(*pts.T, marker="o", ms=3.2, mfc=INK, mec="white", mew=0.4, ls="none", zorder=5)
    handles = [Rectangle((0, 0), 1, 1, fc=shade[lv], ec="none") for lv in levels]
    ax.legend(handles, [f"level {lv}" for lv in levels], loc="upper right", fontsize=CODE, handlelength=1.0,
              borderaxespad=0.3, framealpha=0.9, frameon=True, edgecolor="none")

    # 3: clip one cell
    t = cl.order.index(star)
    clip = cf.Clipper(cl, t)
    here = xy[star]
    half = 0.32
    window = (here[0] - half, here[0] + half, here[1] - half, here[1] + half)
    cuts = [s for s in clip.steps if s["cut"]]
    misses = [s for s in clip.steps if not s["cut"]]
    two_r = math.sqrt(clip.reach2) / cf.STEPS
    ax = sheet.panel(0, 2, "3", "Clip one cell", ["GPU"],
                     f"From a square of half-width L/4, cut by bisectors nearest first (numbers). The walk stops at the "
                     f"first grid cell 2R away (dashed), R the farthest corner: {len(cuts)} cuts, {len(misses)} miss "
                     f"(hollow), {n - 1 - len(clip.steps)} never read.",
                     "polygon_clip.hpp: construct_pass\n3D: shell_voronoi_warp.hpp", window)
    width = 1.0 / (1 << clip.level)
    for level, ux, uy, _ in clip.walked:
        w = 1.0 / (1 << level)
        ax.add_patch(Rectangle((ux * w, uy * w), w, w, fc="#eef1f5", ec="0.8", lw=0.5, zorder=0))
    if clip.stop:
        lv, sx, sy = clip.stop
        w = 1.0 / (1 << lv)
        ax.add_patch(Rectangle((sx * w, sy * w), w, w, fc="none", ec=INK, ls="--", lw=1.1, zorder=2))
    del width
    ax.add_patch(Rectangle(here - 0.25, 0.5, 0.5, fc="none", ec=FAINT, ls=":", lw=1.0, zorder=2))
    final = np.array(clip.vertices()) / cf.STEPS + here
    ax.add_patch(Polygon(final, closed=True, fc="#dbe9f6", ec=BLUE, lw=1.8, zorder=4))
    for j, s in enumerate(cuts):
        o = np.array(s["offset"], dtype=float) / cf.STEPS
        mid, tangent = here + o / 2, np.array([-o[1], o[0]]) / np.linalg.norm(o)
        ax.plot(*np.array([mid - 0.6 * tangent, mid + 0.6 * tangent]).T, color=ORANGE, lw=0.8, ls="--", zorder=3)
        ax.plot(*(here + o), marker="o", ms=5, mfc=ORANGE, mec="white", mew=0.5, zorder=6)
        cf.outlined(ax, *(here + o + np.array([0.012, 0.012])), str(j + 1), fontsize=CODE, zorder=9)
    for s in misses:
        o = np.array(s["offset"], dtype=float) / cf.STEPS
        ax.plot(*(here + o), marker="o", ms=5, mfc="white", mec=ORANGE, mew=1.3, zorder=6)
    dots(ax, xy, size=3.0, color="0.6", zorder=1)
    ax.add_patch(Circle(here, two_r, fc="none", ec=RED, lw=1.4, zorder=5))
    ax.text(here[0] - 0.8 * two_r, here[1] - 0.72 * two_r, "2R", color=RED, fontsize=BODY, fontweight="bold", zorder=9)
    ax.plot(*here, marker="o", ms=7, mfc=INK, mec="white", zorder=7)

    # 4: all cells, face rings
    ax = sheet.panel(1, 0, "4", "All cells: face rings", ["GPU"],
                     f"Every cell is clipped on its own, in parallel, and keeps its ring: its neighbours in order "
                     f"(q1 … q{len(rings[star])} here), in one pool. The rings are the whole topology; no corner is stored.",
                     "polygon_clip.hpp: construct\n3D: shell_voronoi.hpp, per face")
    polys = []
    for i in range(n):
        ring = rings[i]
        o = [run.offset(k, i, q) / P for q in ring]
        polys.append(np.array([xy[i] + circumcentre(np.zeros(2), o[m], o[(m + 1) % len(o)]) for m in range(len(o))]))
    polygons(ax, polys, facecolors="#eef3f9", edgecolors=BLUE, linewidths=0.8, zorder=1)
    ax.add_patch(Polygon(polys[star], closed=True, fc="#9ec5e8", ec=BLUE, lw=1.6, zorder=2))
    dots(ax, xy, size=3.0)
    for m, q in enumerate(rings[star]):
        p = xy[star] + run.offset(k, star, q) / P
        cf.outlined(ax, p[0] + 0.012, p[1] + 0.012, f"q{m + 1}", fontsize=CODE, zorder=9)
    ax.plot(*xy[star], marker="o", ms=6, mfc=INK, mec="white", zorder=7)

    # 5: exact start
    ax = sheet.panel(1, 1, "5", "Exact start", ["GPU"],
                     f"Consecutive ring entries give triangles (i, qₘ, qₘ₊₁), each once: {len(built)} = 2n, with "
                     f"3n = {3 * n} faces. Checked: the exact screen and flips change {int(d['build_flips'])} of them; "
                     f"the areas add up to the box once.",
                     "tri_mesh.hpp: from_rings\n3D: tet_start.hpp: build, check")
    segments(ax, [np.array(s) for s in run.edges(d["built_corners"], k).values()], colors=GREEN, linewidths=0.8, zorder=2)
    fan = [tri for row, tri in zip(ints(d["built_corners"]), built) if star in row[:3]]
    fan = [tri + (xy[star] - tri[list(row[:3]).index(star)]) for row, tri in
           zip([r for r in ints(d["built_corners"]) if star in r[:3]], fan)]
    for tri in fan:
        ax.add_patch(Polygon(tri, closed=True, fc="#dff0dc", ec=GREEN, lw=1.4, zorder=1))
    dots(ax, xy, size=3.0)
    ax.plot(*xy[star], marker="o", ms=6, mfc=INK, mec="white", zorder=7)
    assert abs(d["build_area"] / (2.0 * P * P) - 1) < 1e-12

    # 6: points move
    x1 = ints(d["snap1"]) / P
    move = (x1 - xy + 0.5) % 1.0 - 0.5
    spacing = 1 / math.sqrt(n)
    ax = sheet.panel(1, 2, "6", "Points move", ["caller"],
                     f"The solver calls update(x) with new positions, here up to "
                     f"{np.max(np.hypot(*move.T)) / spacing:.2f} mean spacings (arrows ×3). The triangles are kept "
                     f"(grey); a step needs no cell list and no clipping.",
                     "voronoi_mesh_2d.hpp: Mesh::update\n(the caller: luna-cpp, every step)")
    old = run.edges(d["built_corners"], k)
    segments(ax, [np.array(s) for s in old.values()], colors="0.75", linewidths=0.7, zorder=1)
    for p, m in zip(xy, move):
        for s in SHIFTS:
            q = p + np.array(s)
            if -0.05 < q[0] < 1.05 and -0.05 < q[1] < 1.05:
                cf.arrow(ax, q, q + 3 * m, color=ORANGE, lw=1.0, scale=7, zorder=4)
    dots(ax, xy, size=3.0)
    sheet.save("overview_build")


def step_figure(run, star):
    d, k, xy, n, P = run.d, run.k, run.xy, run.n, run.period
    k1 = ints(d["snap1"])
    x1 = k1 / P
    sheet = Sheet(2, 3)

    # 7: screen
    corners = ints(d["built_corners"])
    listed = set()
    for e in ints(d["screen_list"]):
        t, m = e >> 2, e & 3
        listed.add(tuple(sorted((corners[t][(m + 1) % 3], corners[t][(m + 2) % 3]))))
    old = run.edges(d["built_corners"], k1)
    ax = sheet.panel(0, 0, "7", "Screen", ["GPU"],
                     f"One pass over the old triangles at the new points: the in-circle test of each edge in double, with "
                     f"a rounding bound. Failing, unsure, and (in so small a box) long edges are listed: "
                     f"{len(listed)} of {3 * n}.",
                     "tri_repair.hpp: screen\n3D: tet_repair.hpp: move_screened")
    segments(ax, [s for p, s in old.items() if p not in listed], colors="0.7", linewidths=0.7, zorder=1)
    segments(ax, [s for p, s in old.items() if p in listed], colors=ORANGE, linewidths=2.0, zorder=2)
    dots(ax, x1, size=3.0)

    # 8: flips
    new = run.edges(d["moved_corners"], k1)
    gone, came = set(old) - set(new), set(new) - set(old)
    ax = sheet.panel(0, 1, "8", "Flip", ["GPU"],
                     f"The exact test decides each listed edge. Flips (dashed: gone, green: new) run in rounds; votes settle "
                     f"conflicts, so thread order is moot. {int(d['step_flips'])} flips in "
                     f"{int(d['step_rounds'])} rounds; volumes equal a fresh build's.",
                     "tri_repair.hpp: repair\n3D: tet_repair.hpp: flip23, flip32")
    segments(ax, [s for p, s in new.items() if p not in came], colors="0.7", linewidths=0.7, zorder=1)
    segments(ax, [old[p] for p in gone], colors=RED, linewidths=1.4, linestyles=(0, (3, 2)), zorder=2)
    segments(ax, [new[p] for p in came], colors=GREEN, linewidths=2.2, zorder=3)
    dots(ax, x1, size=3.0)

    # 9: geometry
    cells = run.cells(d["moved_corners"], d["moved_centre"], k1)
    owner, neighbour = ints(d["moved_owner"]), ints(d["moved_neighbour"])
    area, fcen = d["moved_area"], d["moved_face_centroid"]
    volume, centroid = d["moved_volume"], d["moved_centroid"]
    ax = sheet.panel(0, 2, "9", "Geometry: the dual", ["GPU"],
                     "Circumcentres (blue dots) are the cell corners: the Voronoi mesh is the dual of the triangles. "
                     "One thread per face writes its area vector (arrows) and centroid (orange).",
                     "tri_mesh.hpp: faces\n3D: tet_mesh.hpp: faces")
    polygons(ax, cells, facecolors="none", edgecolors=BLUE, linewidths=0.8, zorder=1)
    ax.add_patch(Polygon(cells[star], closed=True, fc="#dbe9f6", ec=BLUE, lw=1.6, zorder=2))
    centres = np.asarray(d["moved_centre"]) % 1.0
    dots(ax, centres, size=2.4, color=BLUE, zorder=3)
    dots(ax, x1, size=3.0, color="0.45")
    mine = [f for f in range(len(owner)) if star in (owner[f], neighbour[f])]
    for f in mine:
        base = x1[owner[f]] + fcen[f]
        sign = 1 if owner[f] == star else -1
        ax.plot(*base, marker="o", ms=4, mfc=ORANGE, mec="white", mew=0.4, zorder=6)
        cf.arrow(ax, base, base + 0.6 * sign * area[f], color=ORANGE, lw=1.2, scale=8, zorder=6)
    ax.plot(*x1[star], marker="o", ms=6, mfc=INK, mec="white", zorder=7)

    # 10: update_cells
    start, faces = ints(d["moved_cell_face_start"]), ints(d["moved_cell_face"])
    listed = list(faces[start[star]:start[star + 1]])
    others = [neighbour[f] if owner[f] == star else owner[f] for f in listed]
    assert others == sorted(others)
    here = x1[star]
    half = 0.17
    total = float(np.sum(volume))
    closure = "Σ volume = L² exactly here." if total == 1 else f"Σ volume = L² to {abs(total - 1):.0e}."
    ax = sheet.panel(1, 0, "10", "update_cells()", ["GPU"],
                     f"Per cell, a list of its faces sorted by the other particle's id (1 … {len(listed)}). Volume and "
                     f"centroid (×) are summed over the pyramids in that order: the same bits on every run. {closure}",
                     "tet_cells.hpp: cells\n(the same code in 2D and 3D)", (here[0] - half, here[0] + half, here[1] - half, here[1] + half))
    polygons(ax, cells, facecolors="none", edgecolors="0.75", linewidths=0.7, zorder=1)
    for m, (f, j) in enumerate(zip(listed, others)):
        base = x1[owner[f]] + fcen[f]
        middle = here + (base - here + 0.5) % 1.0 - 0.5
        a = np.array([-area[f][1], area[f][0]]) * 0.5
        tone = "#9ec5e8" if m % 2 else "#dbe9f6"
        ax.add_patch(Polygon([here, middle - a, middle + a], closed=True, fc=tone, ec=BLUE, lw=0.9, zorder=2))
        out = here + 1.18 * (middle - here)
        cf.outlined(ax, *out, str(m + 1), fontsize=CODE, ha="center", va="center", zorder=9)
        q = here + run.offset(k1, star, j) / P
        cf.outlined(ax, q[0] + 0.006, q[1] - 0.016, f"id {j}", fontsize=8.5, color=FAINT, zorder=9, clip_on=True)
    dots(ax, x1, size=3.4, color="0.45")
    ax.plot(*here, marker="o", ms=6, mfc=INK, mec="white", zorder=7)
    ax.plot(*(here + centroid[star]), marker="x", ms=8, mew=2.0, color=RED, zorder=8)

    # 11: fallbacks
    k2 = ints(d["snap2"])
    x2 = k2 / P
    stuck = [tri for tri in run.triangles(d["jump_corners"], k2) if cf.cross(tri[1] - tri[0], tri[2] - tri[0]) <= 0]
    assert len(stuck) == int(d["jump_stuck"])
    ax = sheet.panel(1, 1, "11", "Fallbacks", ["host", "GPU"],
                     f"Two particles jump (arrows); flips leave {len(stuck)} inverted triangle (red). Few stuck: the host "
                     f"redoes a patch ({int(d['host_host_flips'])} triangles), the GPU flips again. Many: a rebuild "
                     f"(1–5). Two on one grid point: refused.",
                     "tri_finish.hpp: finish (host)\n3D: tet_flips.hpp: finish", (0.05, 0.45, 0.05, 0.45))
    segments(ax, [np.array(s) for s in run.edges(d["jump_corners"], k2).values()], colors="0.7", linewidths=0.8, zorder=1)
    polygons(ax, stuck, facecolors="#f8dcdc", edgecolors=RED, linewidths=1.6, zorder=2)
    for i in run.jumped:
        a, b = x1[i], x1[i] + (x2[i] - x1[i] + 0.5) % 1.0 - 0.5
        cf.arrow(ax, a, b, color=RED, lw=1.4, scale=10, zorder=6)
    dots(ax, x2, size=4.0)
    assert d["host_rebuilt"] == 0 and d["cheapest_rebuilt"] == 1 and "one position" in d["refusal"]

    # 12: re-sort
    lk = ints(d["late_snap"])
    lxy = lk / P
    lo, ln = ints(d["late_owner"]), ints(d["late_neighbour"])
    order = hilbert_order(lxy)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    before = float(np.mean(np.abs(lo - ln)))
    after = float(np.mean(np.abs(rank[lo] - rank[ln])))
    start_gap = float(np.mean(np.abs(ints(d["built_owner"]) - ints(d["built_neighbour"]))))
    ax = sheet.panel(1, 2, "12", "Re-sort (3D)", ["GPU"],
                     f"Over many steps ids drift apart in space (grey: id order). Renumbering along a Hilbert curve "
                     f"(colour) restores locality: the mean id gap over faces is {start_gap:.0f} at the start, "
                     f"{before:.0f} now, {after:.0f} after.",
                     "mesh_order.hpp: cell_order\nvoronoi_mesh.hpp: Mesh::resort")

    def path(sequence, **kwargs):
        p = lxy[sequence]
        jumps = np.stack([p[:-1], p[1:]], axis=1)
        return LineCollection(jumps, **kwargs)
    ax.add_collection(path(list(range(n)), colors="0.7", linewidths=0.6, zorder=1))
    curve = path(order, cmap="magma", norm=plt.Normalize(0, 1.25 * n), linewidths=1.8, zorder=2)
    curve.set_array(np.arange(n - 1))
    ax.add_collection(curve)
    ax.plot(*lxy.T, marker="o", ms=3.0, mfc=INK, mec="white", mew=0.4, ls="none", zorder=5)
    sheet.save("overview_step")


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(BUILD, "overview_out.txt"))
    args = parser.parse_args(argv)
    sets, jumped = positions()
    run_library(sets, args.data)
    run = Run(read_data(args.data), jumped)
    cl = check_ports(run)
    star = int(np.argmin(np.sum((run.xy - np.array([0.74, 0.7])) ** 2, axis=1)))
    build_figure(run, cl, star)
    step_figure(run, star)


if __name__ == "__main__":
    main(sys.argv[1:])
