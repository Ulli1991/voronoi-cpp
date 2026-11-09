#!/usr/bin/env python3
"""Draws docs/figures/construction_*.png in LUNA's figure style. The geometry is computed by ports of the library's 2D
code (polygon_clip.hpp, neighbor_search.hpp, tri_repair.hpp, exact_predicates_2d.hpp), checked against scipy.spatial."""
import bisect
import importlib.util
import itertools
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
from scipy.spatial import Delaunay, Voronoi

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "figures")
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
HEAVY, THIN = 2.2, 0.7

STEPS = 1 << 30  # grid steps across the box
FINE_LEVELS = 10
MAX_REACH = 4  # shell_voronoi::max_reach
CANDIDATES = 16  # polygon_clip::candidates


# ---- exact_predicates_2d.hpp, in Python's exact integers ----
def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def minus(a, b):
    return (a[0] - b[0], a[1] - b[1])


def lift(a):
    return a[0] * a[0] + a[1] * a[1]


def sign(v):
    return (v > 0) - (v < 0)


def circle_exact(a, b, e):
    return cross(b, e) * lift(a) - cross(a, e) * lift(b) + cross(a, b) * lift(e)


def circle_rounded(a, b, e):
    """The double value of circle_sign and its rounding bound."""
    ax, ay, bx, by, ex, ey = map(float, (*a, *b, *e))
    la, lb, le = ax * ax + ay * ay, bx * bx + by * by, ex * ex + ey * ey
    mx, my, det = la * bx - lb * ax, la * by - lb * ay, ax * by - ay * bx
    value = ey * mx - ex * my + le * det
    bound = 16 * 2.0**-53 * (abs(ey) * (la * abs(bx) + lb * abs(ax)) + abs(ex) * (la * abs(by) + lb * abs(ay)) +
                             le * (abs(ax * by) + abs(ay * bx)))
    return value, bound


def in_circle(p, ids):
    """Whether p[3] lies inside the circle through p[0..2]; ties by symbolic perturbation, as in the C++."""
    a, b, e = minus(p[1], p[0]), minus(p[2], p[0]), minus(p[3], p[0])
    o = sign(cross(a, b))
    d = sign(circle_exact(a, b, e))
    if d == 0:
        for i in sorted(range(4), key=lambda j: ids[j]):
            q = [p[j] for j in range(4) if j != i]
            d = (1 if i % 2 else -1) * sign(cross(minus(q[1], q[0]), minus(q[2], q[0])))
            if d:
                break
    return d * o < 0


def wants_flip(p, ids, c, a, b, d):
    """tri_repair::wants_flip: the edge a-b between (c, a, b) and (d, b, a); at an inverted or flat one, if both new are positive."""
    A, B, D = minus(p[a], p[c]), minus(p[b], p[c]), minus(p[d], p[c])
    flip = True
    if cross(A, B) > 0 and cross(minus(B, D), minus(A, D)) > 0:
        flip = in_circle([(0, 0), A, B, D], [ids[c], ids[a], ids[b], ids[d]])
    return flip and cross(A, D) > 0 and cross(D, B) > 0


# ---- neighbor_search.hpp, D = 2, a square box of one tile ----
def spread(v):
    out = 0
    for bit in range(32):
        out |= (v >> bit & 1) << (2 * bit)
    return out


def morton(x, y):
    return spread(x) << 1 | spread(y)


def clamp(v, low, high):
    return max(low, min(v, high))


class CellList:
    """Slot q holds particle order[q]; level[q] is its grid level, as neighbor_search::build fills them."""

    def __init__(self, k):
        n = len(k)
        self.k = k
        self.levels = clamp(math.floor(math.log2(math.sqrt(n)) + 0.5), 2, 10)
        depth = self.levels + FINE_LEVELS
        shift = 30 - depth
        keys = [morton(int(x) >> shift, int(y) >> shift) for x, y in k]
        self.order = sorted(range(n), key=lambda i: (keys[i], i))
        self.keys = [keys[i] for i in self.order]
        cells = [key >> (2 * FINE_LEVELS) for key in self.keys]
        self.offsets = [bisect.bisect_left(cells, c) for c in range(4**self.levels + 1)]
        mask = (1 << (2 * FINE_LEVELS)) - 1
        self.fine = [key & mask for key in self.keys]
        self.level = [self.slot_level(q, cells[q], depth) for q in range(n)]

    def slot_level(self, q, cell, depth):
        enough, base = 64, self.levels
        begin, end = self.offsets[cell], self.offsets[cell + 1]
        node, count = base, end - begin
        if count >= enough:
            while node < depth:
                drop = 2 * (depth - node - 1)
                prefix = self.fine[q] >> drop
                lo = bisect.bisect_left(self.fine, prefix << drop, begin, q)
                hi = bisect.bisect_left(self.fine, (prefix + 1) << drop, q, end)
                if hi - lo < enough:
                    break
                node += 1
                count = hi - lo
        else:
            while node > 0 and count < enough:
                node -= 1
                span = 2 * (base - node)
                first = (cell >> span) << span
                count = self.offsets[first + (1 << span)] - self.offsets[first]
        finer = int(math.log2(max(count, 1)) / 2 + 0.5)
        return clamp(node + finer, 2, depth)

    def range(self, at, x, y):
        below = at - self.levels
        if below <= 0:
            first = morton(x, y) << (2 * (self.levels - at))
            return self.offsets[first], self.offsets[first + (1 << (-2 * below))]
        base = morton(x >> below, y >> below)
        begin, end = self.offsets[base], self.offsets[base + 1]
        if begin == end:
            return begin, end
        mask, drop = (1 << below) - 1, 2 * (FINE_LEVELS - below)
        key = morton(x & mask, y & mask) << drop
        begin = bisect.bisect_left(self.fine, key, begin, end)
        return begin, bisect.bisect_left(self.fine, key + (1 << drop), begin, end)


# ---- polygon_clip.hpp: one cell ----
def bisector_vertex(a, b):
    """Where the bisectors of offsets a and b meet."""
    la, lb, scale = lift(a), lift(b), 0.5 / cross(a, b)
    return ((la * b[1] - lb * a[1]) * scale, (lb * a[0] - la * b[0]) * scale)


def stencil():
    entries = []
    for dx in range(-MAX_REACH, MAX_REACH + 1):
        for dy in range(-MAX_REACH, MAX_REACH + 1):
            gaps = max(abs(dx) - 1, 0) ** 2 + max(abs(dy) - 1, 0) ** 2
            if gaps < MAX_REACH**2:
                entries.append((gaps, (dx + 8) | (dy + 8) << 5, dx, dy))
    return sorted(entries)


class Clipper:
    """The walk and clip of polygon_clip::construct_pass for one slot t, keeping every step for drawing."""

    def __init__(self, cl, t):
        self.cl, self.t, self.id = cl, t, cl.order[t]
        self.self = tuple(int(v) for v in cl.k[self.id])
        half = STEPS // 4
        self.half = half
        o = [(2 * half, 0), (0, 2 * half), (-2 * half, 0), (0, -2 * half)]
        self.edges = [(o[m], m - 4) for m in range(4)]  # offset of the particle beyond the edge, slot (negative: virtual)
        self.measure()
        self.start_reach2 = self.reach2
        self.steps, self.walked, self.buffer, self.gaps = [], [], [], 0
        self.walk()

    def vertices(self):
        e = self.edges
        return [bisector_vertex(e[m][0], e[(m + 1) % len(e)][0]) for m in range(len(e))]

    def measure(self):
        self.reach2 = 4 * max(v[0] ** 2 + v[1] ** 2 for v in self.vertices()) * (1 + 1e-9)

    def particle(self, slot):
        return slot if slot < 0 else self.cl.order[slot]

    def clip(self, offset, slot):
        e, m_count = self.edges, len(self.edges)
        ids = lambda m: [self.id, self.particle(e[m][1]), self.particle(e[(m + 1) % m_count][1]), self.particle(slot)]
        cut = [in_circle([(0, 0), e[m][0], e[(m + 1) % m_count][0], offset], ids(m)) for m in range(m_count)]
        before = self.vertices()
        if any(cut):
            assert not all(cut)
            first = next(m for m in range(m_count) if cut[m] and not cut[m - 1])
            keep = []
            m = (first + 1) % m_count
            while cut[m]:
                m = (m + 1) % m_count
            while True:  # edges kept: both ends of edge m are vertices m - 1, m
                keep.append(e[m])
                if m == first:
                    break
                m = (m + 1) % m_count
            self.edges = keep + [(offset, slot)]
            self.measure()
        self.steps.append(dict(slot=slot, offset=offset, cut=any(cut), before=before, after=self.vertices(), reach2=self.reach2,
                               gaps=self.gaps, edges_after=list(self.edges)))

    def flush(self):
        self.buffer.sort(key=lambda c: c[0])
        for d2, offset, slot in self.buffer:
            if d2 > self.reach2:
                break
            self.clip(offset, slot)
        self.buffer = []

    def walk(self):
        cl, entries = self.cl, stencil()
        level = cl.level[self.t]
        home = [int(self.self[a]) >> (30 - level) for a in range(2)]
        finer = -1
        certified = False
        while not certified:
            width = STEPS / (1 << level)
            reach = min(MAX_REACH, ((1 << level) - 1) // 2)
            ends = max((i + 1 for i, e in enumerate(entries) if e[0] < reach * reach), default=0)
            for i in range(ends):
                gaps, _, dx, dy = entries[i]
                self.gaps = gaps
                if gaps * width * width >= self.reach2:
                    self.stop = (level, home[0] + dx, home[1] + dy)
                    certified = True
                    break
                u = (home[0] + dx, home[1] + dy)
                gap2 = 0.0
                for a in range(2):
                    g = max(0.0, u[a] * width - self.self[a], self.self[a] - (u[a] + 1) * width)
                    gap2 += g * g
                skip = gap2 > self.reach2
                if not skip and finer >= 0:
                    near = 0
                    for a in range(2):
                        child = 2 * (u[a] - home[a]) - ((finer >> a) & 1)
                        near += max(max(abs(child) - 1, 0) ** 2, max(abs(child + 1) - 1, 0) ** 2)
                    covered = min(MAX_REACH, ((1 << (level + 1)) - 1) // 2)
                    skip = near < covered * covered
                if not skip:
                    self.visit(level, u)
                if i + 1 == ends or entries[i + 1][0] > gaps:
                    self.flush()
            if certified:
                break
            self.flush()
            if reach * width * reach * width >= self.reach2:
                self.stop = None
                certified = True
                break
            assert level > 2
            finer = (home[0] & 1) | (home[1] & 1) << 1
            home = [home[0] >> 1, home[1] >> 1]
            level -= 1
        self.flush()
        self.level = level
        self.home = home
        assert all(s >= 0 for _, s in self.edges)

    def visit(self, level, u):
        cl = self.cl
        cells = 1 << level
        shift = [(-1 if v < 0 else 1 if v >= cells else 0) for v in u]
        w = [u[a] - shift[a] * cells for a in range(2)]
        self.walked.append((level, u[0], u[1], self.gaps))
        begin, end = cl.range(level, w[0], w[1])
        for s in range(begin, end):
            if s == self.t:
                continue
            o = (int(cl.k[cl.order[s]][0]) + shift[0] * STEPS - self.self[0],
                 int(cl.k[cl.order[s]][1]) + shift[1] * STEPS - self.self[1])
            d2 = float(o[0]) ** 2 + float(o[1]) ** 2
            if not d2 <= self.reach2:
                continue
            if len(self.buffer) == CANDIDATES:
                self.flush()
            self.buffer.append((d2, o, s))


# ---- tri_repair.hpp: Lawson rounds with votes, on a tiled patch of the periodic box ----
def hash32(x):
    x = x * 0x9e3779b1 & 0xffffffff
    x ^= x >> 16
    x = x * 0x85ebca6b & 0xffffffff
    return x ^ x >> 13


class Triangulation:
    """Triangles (i, j, k) counter-clockwise over integer points; a flip rewrites two of them in place, as Apply does."""

    def __init__(self, p, ids, triangles):
        self.p, self.ids = p, ids
        self.tris = []
        for a, b, c in triangles:
            a, b, c = int(a), int(b), int(c)
            self.tris.append((a, b, c) if cross(minus(p[b], p[a]), minus(p[c], p[a])) > 0 else (a, c, b))
        self.flips, self.touched = [], set()
        self.rounds = 0

    def repair(self):
        while True:
            where = {}
            for t, (a, b, c) in enumerate(self.tris):
                where[(a, b)], where[(b, c)], where[(c, a)] = (t, c), (t, a), (t, b)
            lists = []
            for (a, b), (t, c) in where.items():
                if (b, a) in where:
                    u, d = where[(b, a)]
                    # Each edge once, from the side of the lower apex (Screen).
                    if self.ids[c] < self.ids[d] or (self.ids[c] == self.ids[d] and c < d):
                        if wants_flip(self.p, self.ids, c, a, b, d):
                            lists.append((t, u, c, a, b, d))
            if not lists:
                return
            self.rounds += 1
            lock, plans = {}, []
            for t, u, c, a, b, d in lists:
                outer = [where[e][0] for e in ((b, c), (c, a), (a, d), (d, b)) if e in where]
                if len(outer) < 4:
                    continue
                tag = sorted((self.ids[a], self.ids[b]))
                key = hash32(tag[0] * 65537 + tag[1] + self.rounds * 977)
                six = [t, u, *outer]
                for s in six:
                    lock[s] = max(lock.get(s, -1), key)
                plans.append((key, six, t, u, c, a, b, d))
            for key, six, t, u, c, a, b, d in plans:
                if all(lock[s] == key for s in six):
                    self.tris[t], self.tris[u] = (c, a, d), (c, d, b)
                    self.touched |= {t, u}
                    self.flips.append((self.rounds, t, u, c, a, b, d))


# ---- point sets and drawing helpers ----
def relaxed(n, seed, sweeps):
    """A random periodic point set after a few Lloyd sweeps, so that cells are well shaped."""
    x = np.random.default_rng(seed).random((n, 2))
    for _ in range(sweeps):
        tiled = np.concatenate([x + (i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)])
        v = Voronoi(tiled)
        moved = x.copy()
        for i in range(n):
            region = v.regions[v.point_region[4 * n + i]]
            w = v.vertices[region]
            area = cross_area(w)
            moved[i] = centroid(w, area)
        x = moved % 1.0
    return x


def cross_area(w):
    return 0.5 * np.sum(w[:, 0] * np.roll(w[:, 1], -1) - np.roll(w[:, 0], -1) * w[:, 1])


def centroid(w, area):
    c = w[:, 0] * np.roll(w[:, 1], -1) - np.roll(w[:, 0], -1) * w[:, 1]
    return np.array([np.sum((w[:, a] + np.roll(w[:, a], -1)) * c) for a in range(2)]) / (6 * area)


def snapped(x):
    return np.floor(x * STEPS).astype(np.int64)


def sketch(ax):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.minorticks_off()


def arrow(ax, start, end, color=INK, lw=1.6, scale=15, zorder=6):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=scale, lw=lw, color=color, shrinkA=0, shrinkB=0, zorder=zorder))


def outlined(ax, x, y, text, **kwargs):
    ax.text(x, y, text, color=kwargs.pop("color", INK), fontsize=kwargs.pop("fontsize", 14), zorder=kwargs.pop("zorder", 12),
            bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.85), **kwargs)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)


# ---- figure 1: one cell by clipping ----
def polygon_patch(points, **kwargs):
    return Polygon(np.asarray(points), closed=True, joinstyle="round", **kwargs)


def clipping_figure():
    x = relaxed(144, 1, 2)
    k = snapped(x)
    cl = CellList(k)
    i = int(np.argmin(((x - 0.5) ** 2).sum(1)))
    t = cl.order.index(i)
    c = Clipper(cl, t)
    g = k[i] / STEPS
    unit = 1.0 / STEPS
    level = cl.level[t]
    width = 1.0 / (1 << level)
    final = {s for _, s in c.edges}
    steps = c.steps
    first = [s for s in steps if s["gaps"] == 0]
    later = [s for s in steps if s["gaps"] > 0]
    tested = {s["slot"] for s in steps}
    cut_slots = {s["slot"] for s in steps if s["cut"]}

    # Checks: the cell is the Voronoi cell of the scipy construction, its ring the Delaunay neighbours.
    tiled = np.concatenate([k / STEPS + (a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)])
    vor = Voronoi(tiled)
    region = vor.vertices[vor.regions[vor.point_region[4 * len(k) + i]]]
    mine = np.array(c.vertices()) * unit + g
    assert abs(abs(cross_area(region)) - abs(cross_area(mine))) < 1e-12

    def rel(points):
        return np.asarray(points) * unit + g

    half = 0.25
    lo, hi = g - half - 0.035, g + half + 0.035
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 12.6))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.03, hspace=0.03)
    order_of = {s["slot"]: n + 1 for n, s in enumerate(steps)}

    def frame(ax, label):
        sketch(ax)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1] - 0.085, hi[1] + 0.07)
        x0, x1 = math.floor(lo[0] / width), math.ceil(hi[0] / width)
        y0, y1 = math.floor(lo[1] / width), math.ceil(hi[1] / width)
        grid = [[(a * width, lo[1]), (a * width, hi[1])] for a in range(x0, x1 + 1) if lo[0] <= a * width <= hi[0]]
        grid += [[(lo[0], b * width), (hi[0], b * width)] for b in range(y0, y1 + 1) if lo[1] <= b * width <= hi[1]]
        ax.add_collection(LineCollection(grid, colors="0.82", linewidths=THIN, zorder=1))
        ax.text(lo[0] + 0.004, hi[1] + 0.035, label, fontsize=15, color=INK, va="center", ha="left")

    def cells(ax, which, fc, ec="none", lw=0):
        for lv, a, b in which:
            w = 1.0 / (1 << lv)
            ax.add_patch(Rectangle((a * w, b * w), w, w, fc=fc, ec=ec, lw=lw, zorder=1.5 if ec == "none" else 6))

    def particles(ax, styles):
        inside = [j for j in range(len(k)) if j != i and np.all(np.abs(k[j] / STEPS - g) < half + 0.03)]
        for j in inside:
            q = k[j] / STEPS
            style = styles(j)
            ax.plot(*q, **style)
        ax.plot(*g, marker="o", ms=11, mfc=INK, mec="white", mew=1.6, zorder=9, ls="none")

    plain = dict(marker="o", ms=6.5, mfc=FAINT, mec="white", mew=0.8, zorder=8, ls="none")

    def cut_steps(ax, group):
        for s in group:
            if not s["cut"]:
                continue
            before, after = rel(s["before"]), rel(s["after"])
            ax.add_patch(polygon_patch(before, fc=TEST[0], ec="none", zorder=2))
        for s in group:
            if not s["cut"]:
                continue
            after = rel(s["after"])
            m = [e[1] for e in s["edges_after"]].index(s["slot"])
            a, b = after[m - 1], after[m]
            d = b - a
            ext = 0.03 / np.linalg.norm(d)
            ax.plot(*zip(a - d * ext, b + d * ext), color=TEST[1], lw=THIN, ls=(0, (3, 2)), zorder=3)
            ax.plot(*zip(a, b), color=TEST[1], lw=HEAVY, solid_capstyle="round", zorder=4)

    def label_order(ax, group, color=INK):
        for s in group:
            q = k[cl.order[s["slot"]]] / STEPS
            away = q - g
            away = away / np.linalg.norm(away)
            outlined(ax, *(q + away * 0.022), str(order_of[s["slot"]]), ha="center", va="center", fontsize=13,
                     color=color, fontweight="bold")

    # (a) the start square
    ax = axes[0, 0]
    frame(ax, "(a) Start: a square of half-width L/4")
    home = (level, *[int(v) >> (30 - level) for v in k[i]])
    cells(ax, [home], "none", INK, 1.2)
    square = rel(steps[0]["before"])
    ax.add_patch(polygon_patch(square, fc=REBUILD[0], ec=REBUILD[1], lw=HEAVY, zorder=2, alpha=0.9))
    particles(ax, lambda j: plain)
    corner = square[0]
    ax.plot(*zip(g, corner), color=INK, lw=THIN, zorder=7)
    outlined(ax, *(0.5 * (g + corner) + (0.012, -0.022)), "R = √2 L/4", fontsize=13)
    outlined(ax, *(g + (0.012, -0.03)), "home grid cell", fontsize=12, color=FAINT)
    ax.text(lo[0] + 0.004, lo[1] - 0.04, "2R = 0.71 L: every particle in view may cut.\n"
            f"Grid level {level}: {1 << level} × {1 << level} cells, about one particle each.", fontsize=13, va="center", color=INK)

    # (b) the first ring of grid cells, clipped nearest first
    ax = axes[0, 1]
    frame(ax, "(b) The 3 × 3 cells around home, nearest first")
    ring0 = [w[:3] for w in c.walked if w[3] == 0]
    cells(ax, ring0, "none", SETUP[1], 1.2)
    cut_steps(ax, first)
    ax.add_patch(polygon_patch(rel(first[-1]["after"]), fc=REBUILD[0], ec=REBUILD[1], lw=HEAVY, zorder=5, alpha=0.9))
    particles(ax, lambda j: dict(plain, mfc=TEST[1]) if cl.order.index(j) in {s["slot"] for s in first} else plain)
    label_order(ax, first)
    r = math.sqrt(first[-1]["reach2"]) / 2 / STEPS
    ax.text(lo[0] + 0.004, lo[1] - 0.04, f"Each bisector cuts off a corner (orange).\nR falls to {r:.2f} L.", fontsize=13,
            va="center", color=INK)

    # (c) the next rings
    ax = axes[1, 0]
    frame(ax, "(c) Next rings: only particles within 2R")
    later_cells = [w[:3] for w in c.walked if w[3] > 0]
    cells(ax, ring0, SETUP[0])
    cells(ax, later_cells, "#e4e8ee")
    ax.add_patch(polygon_patch(rel(first[-1]["after"]), fc="none", ec=REBUILD[1], lw=THIN, ls=(0, (4, 2)), zorder=5))
    cut_steps(ax, later)
    ax.add_patch(polygon_patch(rel(c.vertices()), fc=REBUILD[0], ec=REBUILD[1], lw=HEAVY, zorder=5, alpha=0.9))
    later_slots = {s["slot"] for s in later}

    def style_c(j):
        q = cl.order.index(j)
        if q in later_slots:
            return dict(plain, mfc=TEST[1]) if q in cut_slots else dict(plain, mfc="white", mec=TEST[1], mew=1.8)
        return plain
    particles(ax, style_c)
    label_order(ax, later)
    ax.text(lo[0] + 0.004, lo[1] - 0.04, "Hollow: tested, its bisector misses the cell.\nDashed: the cell after (b).",
            fontsize=13, va="center", color=INK)

    # (d) certified
    ax = axes[1, 1]
    frame(ax, "(d) Certified: nothing left within 2R")
    cells(ax, [w[:3] for w in c.walked], SETUP[0])
    if c.stop:
        lv, a, b = c.stop
        w = 1.0 / (1 << lv)
        ax.add_patch(Rectangle((a * w, b * w), w, w, fc="none", ec=INK, lw=1.4, ls=(0, (3, 2)), zorder=6))
        outlined(ax, (a + 1) * w + 0.006, (b + 0.5) * w, "walk stops: this cell\nis 2R from the home cell", fontsize=12,
                 va="center", color=INK)
    reach = math.sqrt(c.reach2) / STEPS
    ax.add_patch(Circle(g, reach, fc="none", ec=REFUSE[1], lw=HEAVY, zorder=6))
    ax.add_patch(polygon_patch(rel(c.vertices()), fc=REBUILD[0], ec=REBUILD[1], lw=HEAVY, zorder=5, alpha=0.9))
    angle = math.radians(140)
    outlined(ax, g[0] + reach * math.cos(angle) - 0.045, g[1] + reach * math.sin(angle) + 0.01, "2R", fontsize=14,
             color=REFUSE[1], fontweight="bold")

    def style_d(j):
        q = cl.order.index(j)
        if q in final:
            return dict(plain, mfc=REBUILD[1], ms=8)
        if q in tested:
            return dict(plain, mfc="white", mec=TEST[1], mew=1.8)
        return dict(plain, mfc="white", mec="0.6", mew=1.2, ms=5.5)
    particles(ax, style_d)
    handles = [plt.Line2D([], [], **dict(plain, mfc=REBUILD[1], ms=8)),
               plt.Line2D([], [], **dict(plain, mfc="white", mec=TEST[1], mew=1.8)),
               plt.Line2D([], [], **dict(plain, mfc="white", mec="0.6", mew=1.2, ms=5.5))]
    labels = ["neighbours: the cell's faces", f"tested, not a face ({len(tested) - len(final)})",
              f"never needed ({sum(1 for j in range(len(k)) if j != i and cl.order.index(j) not in tested and np.all(np.abs(k[j] / STEPS - g) < half + 0.03))} in view)"]
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=12.5, handletextpad=0.3,
              borderaxespad=0.2, ncol=1)
    save(fig, "construction_clipping.png")




# ---- figure 2: the cell list ----
def clustered_points():
    rng = np.random.default_rng(7)
    clump = 0.36 + 0.05 * rng.standard_normal((460, 2))
    small = np.array([0.74, 0.70]) + 0.022 * rng.standard_normal((110, 2))
    return np.concatenate([rng.random((230, 2)), clump, small]) % 1.0


def cell_list_figure():
    x = clustered_points()
    k = snapped(x)
    cl = CellList(k)
    n = len(k)
    levels = sorted(set(cl.level))
    colour = {lv: plt.cm.magma(0.25 + 0.6 * (lv - levels[0]) / max(1, levels[-1] - levels[0])) for lv in levels}
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.4))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995, wspace=0.03)
    for ax in axes:
        sketch(ax)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.2, 1.08)
        ax.add_patch(Rectangle((0, 0), 1, 1, fc="none", ec=INK, lw=1.1, zorder=7))
    pts = np.array([k[cl.order[q]] for q in range(n)]) / STEPS

    # (a) every particle's home cell at its own level, coarse first
    ax = axes[0]
    for q in sorted(range(n), key=lambda q: cl.level[q]):
        lv = cl.level[q]
        a, b = (int(v) >> (30 - lv) for v in k[cl.order[q]])
        w = 1.0 / (1 << lv)
        ax.add_patch(Rectangle((a * w, b * w), w, w, fc=colour[lv], ec="white", lw=0.3, zorder=2))
    ax.plot(*pts.T, marker="o", ms=1.7, mfc=INK, mec="none", ls="none", zorder=5)
    ax.text(0.0, 1.04, "(a) Each particle's grid level, its home cell shown", fontsize=15, va="center")
    handles = [Rectangle((0, 0), 1, 1, fc=colour[lv], ec="none") for lv in levels]
    labels = [f"level {lv}: {1 << lv}² cells" for lv in levels]
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=2, fontsize=12.5, handlelength=1.2,
              columnspacing=1.2, borderaxespad=0.3)

    # (b) slot order: the Morton curve, coloured from the first slot to the last
    ax = axes[1]
    base = 1 << cl.levels
    grid = [[(a / base, 0), (a / base, 1)] for a in range(1, base)] + [[(0, b / base), (1, b / base)] for b in range(1, base)]
    ax.add_collection(LineCollection(grid, colors="0.85", linewidths=THIN, zorder=1))
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    curve = LineCollection(segments, cmap="magma", norm=plt.Normalize(0, 1.25 * n), linewidths=1.6, zorder=3,
                           capstyle="round", joinstyle="round")
    curve.set_array(np.arange(n - 1))
    ax.add_collection(curve)
    ax.plot(*pts.T, marker="o", ms=1.7, mfc=INK, mec="none", ls="none", zorder=5)
    # One node of the implicit tree: its particles are one contiguous run of slots.
    node_level, node = 2, (1, 1)
    w = 1.0 / (1 << node_level)
    span = 2 * (cl.levels - node_level)
    first = morton(*node) << span
    begin, end = cl.offsets[first], cl.offsets[first + (1 << span)]
    ax.add_patch(Rectangle((node[0] * w, node[1] * w), w, w, fc="none", ec=REBUILD[1], lw=HEAVY, zorder=6))
    ax.text(0.0, 1.04, "(b) Slot order: the Morton curve", fontsize=15, va="center")
    up = cl.levels - node_level
    ax.text(0.0, -0.1, f"Base grid {base} × {base}, n = {n}; colour from slot 0 (dark) to {n - 1} (light).\n"
            f"The blue node, {up} levels above the base grid, holds slots {begin} to {end - 1}:\n"
            "every node of the tree is one run of slots.", fontsize=13, va="center", linespacing=1.3)
    save(fig, "construction_cell_list.png")
    return cl, levels


# ---- figure 3: repair by flips ----
def circumcentre(p, q, r):
    a, b = q - p, r - p
    d = 2 * (a[0] * b[1] - a[1] * b[0])
    la, lb = a @ a, b @ b
    return p + np.array([b[1] * la - a[1] * lb, a[0] * lb - b[0] * la]) / d


def moved_mesh(n=300, seed=9, amplitude=0.3):
    """A relaxed periodic set moved along a smooth flow, tiled so the box sees no hull; old and new integer positions."""
    x0 = relaxed(n, seed, 3)
    spacing = 1 / math.sqrt(n)
    flow = np.stack([np.sin(2 * np.pi * x0[:, 1]) + 0.5 * np.sin(4 * np.pi * x0[:, 0] + 1.0),
                     np.sin(2 * np.pi * x0[:, 0]) + 0.5 * np.cos(4 * np.pi * x0[:, 1])], axis=1)
    x1 = x0 + amplitude * spacing * flow  # unwrapped: the code takes every offset to the nearest image
    shifts = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
    k0, k1 = snapped(x0), snapped(x1)

    def tile(k):
        return [(int(k[i, 0]) + a * STEPS, int(k[i, 1]) + b * STEPS) for a, b in shifts for i in range(n)]
    p0, p1 = tile(k0), tile(k1)
    keep = [j for j in range(len(p0)) if all(-0.3 < v / STEPS < 1.3 for v in p0[j] + p1[j])]
    p0, p1 = [p0[j] for j in keep], [p1[j] for j in keep]
    ids = [keep[j] % n for j in range(len(keep))]
    moved = float(np.max(np.linalg.norm(x1 - x0, axis=1)) / spacing)
    return p0, p1, ids, n, moved


def flip_figure():
    p0, p1, ids, n, moved = moved_mesh()
    start = Delaunay(np.array(p0, dtype=float) / STEPS).simplices
    mesh = Triangulation(p1, ids, start)
    before = list(mesh.tris)
    mesh.repair()
    final = {tuple(sorted(tr)) for tr in mesh.tris}
    check = {tuple(sorted(tr)) for tr in Delaunay(np.array(p1, dtype=float) / STEPS).simplices}
    xy = np.array(p1, dtype=float) / STEPS

    def central(tr):
        return np.all((xy[list(tr)].mean(0) > 0.05) & (xy[list(tr)].mean(0) < 0.95))
    assert {tr for tr in final if central(tr)} == {tr for tr in check if central(tr)}
    spacing = 1 / math.sqrt(n)

    # The flip to show: inside the box, well shaped, clearly failing, and the only flip in its drawn patch.
    centres = {f: xy[[f[3], f[4], f[5], f[6]]].mean(0) for f in mesh.flips}
    best, score = None, -1
    for f in mesh.flips:
        rnd, t, u, c, a, b, d = f
        m = centres[f]
        if rnd != 1 or not np.all((m > 0.2) & (m < 0.8)):
            continue
        cc = circumcentre(xy[c], xy[a], xy[b])
        radius = np.linalg.norm(xy[c] - cc)
        depth = (radius - np.linalg.norm(xy[d] - cc)) / radius
        sides = [np.linalg.norm(xy[e] - xy[h]) for e, h in ((c, a), (a, d), (d, b), (b, c))]
        quality = min(sides) / max(sides) * min(1.0, radius / (1.4 * spacing))
        half = max(1.25 * radius, 2.3 * spacing)
        near = [j for j, tr in enumerate(before) if np.linalg.norm(xy[list(tr)].mean(0) - m) < 1.5 * half]
        after = list(before)
        after[t], after[u] = (c, a, d), (c, d, b)
        alone = all(tuple(sorted(after[j])) in final for j in near)
        if alone and depth > 0.06 and quality * depth > score:
            best, score = f, quality * depth
    rnd, t, u, c, a, b, d = best
    after = list(before)
    after[t], after[u] = (c, a, d), (c, d, b)
    middle = xy[[c, a, b, d]].mean(0)
    cc = circumcentre(xy[c], xy[a], xy[b])
    radius = np.linalg.norm(xy[c] - cc)
    half = max(1.25 * radius, 2.3 * spacing)
    window = (middle - half, middle + half)
    near = [j for j, tr in enumerate(before) if np.linalg.norm(xy[list(tr)].mean(0) - middle) < 1.5 * half]

    fig = plt.figure(figsize=(15.2, 10.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.55], left=0.01, right=0.99, bottom=0.07, top=0.95,
                          wspace=0.06, hspace=0.24)
    ax_b, ax_a = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_vb, ax_va = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    ax_m = fig.add_subplot(gs[:, 2])
    name = {c: "c", a: "a", b: "b", d: "d"}

    def patch_axes(ax, label):
        sketch(ax)
        ax.set_xlim(window[0][0], window[1][0])
        ax.set_ylim(window[0][1], window[1][1])
        ax.add_patch(Rectangle(window[0], 2 * half, 2 * half, fc="none", ec="0.6", lw=1.0, zorder=10, clip_on=False))
        ax.text(0, 1.04, label, fontsize=15, va="bottom", transform=ax.transAxes)
        return ax.patch

    def caption(ax, text):
        ax.text(0, -0.03, text, fontsize=13, va="top", transform=ax.transAxes, linespacing=1.3)

    def ring_mark(ax, segment, colour):
        mid = 0.5 * (segment[0] + segment[1])
        ax.add_patch(Circle(mid, 0.16 * half, fc="none", ec=colour, lw=1.4, zorder=6))

    def draw_tris(ax, tris, clip, highlight, tone):
        segments = []
        for j in near:
            tr = tris[j]
            for e in range(3):
                segments.append([xy[tr[e]], xy[tr[(e + 1) % 3]]])
        lines = LineCollection(segments, colors="0.55", linewidths=THIN, zorder=2)
        lines.set_clip_path(clip)
        ax.add_collection(lines)
        for j in highlight:
            ax.add_patch(polygon_patch(xy[list(tris[j])], fc=tone[0], ec=tone[1], lw=HEAVY, zorder=3))

    def draw_points(ax, clip):
        inside = [j for j in {v for jj in near for v in before[jj]} if np.all(np.abs(xy[j] - middle) < half)]
        ax.plot(*xy[inside].T, marker="o", ms=6, mfc=INK, mec="white", mew=0.8, ls="none", zorder=8)
        for v, label in name.items():
            away = xy[v] - middle
            away = away / np.linalg.norm(away)
            outlined(ax, *(xy[v] + away * 0.13 * half), label, fontsize=15, ha="center", va="center", fontweight="bold")

    # (a) before the flip
    clip = patch_axes(ax_b, "(a) After the move: edge a-b fails")
    draw_tris(ax_b, before, clip, [t, u], REBUILD)
    ax_b.plot(*zip(xy[a], xy[b]), color=REFUSE[1], lw=HEAVY + 0.6, zorder=4, solid_capstyle="round")
    ring = Circle(cc, radius, fc="none", ec=TEST[1], lw=HEAVY, zorder=5)
    ax_b.add_patch(ring)
    ring.set_clip_path(clip)
    draw_points(ax_b, clip)
    ax_b.plot(*xy[d], marker="o", ms=11, mfc=REFUSE[1], mec="white", mew=1.2, ls="none", zorder=9)
    caption(ax_b, "d lies inside the circle through c, a, b.")

    # (b) after the flip
    clip = patch_axes(ax_a, "(b) One 2-2 flip: edge c-d")
    draw_tris(ax_a, after, clip, [t, u], REPAIR)
    ax_a.plot(*zip(xy[c], xy[d]), color=REPAIR[1], lw=HEAVY + 0.6, zorder=4, solid_capstyle="round")
    draw_points(ax_a, clip)
    caption(ax_a, "Two triangles rewritten in place.")

    # (c), (d) the Voronoi faces of the same triangles
    def dual(ax, tris, label, gone, new, text):
        clip = patch_axes(ax, label)
        centre = {}
        where = {}
        for j in near:
            tr = tris[j]
            centre[j] = circumcentre(*xy[list(tr)])
            for e in range(3):
                where[(tr[e], tr[(e + 1) % 3])] = j
        plain, side, special = [], [], []
        for (p, q), j in where.items():
            if (q, p) not in where or p > q:
                continue
            segment = [centre[j], centre[where[(q, p)]]]
            pair = {p, q}
            if pair == gone or pair == new:
                special.append(segment)
            elif pair & {a, b} and pair & {c, d} and len(pair & {a, b, c, d}) == 2:
                side.append(segment)
            else:
                plain.append(segment)
        for segs, colour, width, z in ((plain, "0.6", 1.2, 2), (side, INK, HEAVY, 3),
                                       (special, REFUSE[1] if gone else REPAIR[1], HEAVY + 0.8, 4)):
            lines = LineCollection(segs, colors=colour, linewidths=width, zorder=z, capstyle="round")
            lines.set_clip_path(clip)
            ax.add_collection(lines)
        draw_points(ax, clip)
        for segment in special:
            ring_mark(ax, segment, REFUSE[1] if gone else REPAIR[1])
        caption(ax, text)

    dual(ax_vb, before, "(c) Its Voronoi faces before", {a, b}, None, "Face a|b (red) is inside out.")
    dual(ax_va, after, "(d) After: one face replaced", None, {c, d}, "New face c|d (green); four faces\nget a new end (black).")

    # (e) the whole moved mesh, the triangles the repair rewrote
    ax = ax_m
    sketch(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    box = ax.patch
    segments = [[xy[tr[e]], xy[tr[(e + 1) % 3]]] for tr in mesh.tris for e in range(3)]
    lines = LineCollection(segments, colors="0.6", linewidths=THIN, zorder=2)
    lines.set_clip_path(box)
    ax.add_collection(lines)
    touched = [polygon_patch(xy[list(mesh.tris[j])], fc=REPAIR[0], ec=REPAIR[1], lw=1.4, zorder=3) for j in mesh.touched]
    for patch in touched:
        ax.add_patch(patch)
        patch.set_clip_path(box)
    ax.add_patch(Rectangle((0, 0), 1, 1, fc="none", ec=INK, lw=1.1, zorder=6, clip_on=False))
    ax.add_patch(Rectangle(window[0], 2 * half, 2 * half, fc="none", ec=INK, lw=1.4, zorder=7))
    outlined(ax, window[1][0] + 0.01, window[0][1] + 0.01, "(a)-(d)", fontsize=13, va="bottom")
    flips_in_box = sum(1 for f in mesh.flips if np.all((centres[f] >= 0) & (centres[f] < 1)))
    ax.text(0, 1.02, "(e) The whole moved mesh", fontsize=15, va="bottom", transform=ax.transAxes)
    rounds = f"{mesh.rounds} round" + ("s" if mesh.rounds > 1 else "")
    ax.text(0, -0.02, f"n = {n}, periodic, particles moved up to {moved:.1f} spacings.\n"
            f"Green: the {2 * flips_in_box} of {2 * n} triangles that {flips_in_box} flips rewrote, in {rounds}.",
            fontsize=13, va="top", transform=ax.transAxes, linespacing=1.3)
    save(fig, "construction_flip.png")
    return mesh, best


# ---- figure 4: snapping and exact ties ----
def superscript(v):
    """A float as m×10ⁿ with Unicode superscripts."""
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    digits = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{v / 10**e:+.1f}×10{digits}".replace("-", "−")


def cocircular():
    """Four points on one lattice circle (the drawing), the same scaled onto the 2³⁰ grid, and rounded values that disagree."""
    centre, quad = (8, 8), [(13, 8), (11, 12), (4, 11), (5, 4)]
    for scale in range(99_000_001, 107_000_000, 2):
        big = [((1 << 29) + scale * (x - centre[0]), (1 << 29) + scale * (y - centre[1])) for x, y in quad]
        values = {}
        for order in itertools.permutations(range(4)):
            q = [big[j] for j in order]
            a, b, e = minus(q[1], q[0]), minus(q[2], q[0]), minus(q[3], q[0])
            assert circle_exact(a, b, e) == 0
            value, bound = circle_rounded(a, b, e)
            assert abs(value) <= bound
            values[order] = value * sign(cross(a, b))
        if max(values.values()) > 0 > min(values.values()):
            return centre, quad, big, scale, values
    raise RuntimeError("no scale found")


def tie_break(big, ids):
    """The diagonal the perturbed test keeps, checked from every order of the arguments."""
    kept = set()
    for diagonal in ((0, 2), (1, 3)):
        p, q = diagonal
        others = [j for j in range(4) if j not in diagonal]
        fine = True
        for r, opposite in ((others[0], others[1]), (others[1], others[0])):
            for order in itertools.permutations([p, q, r]):
                pts = [big[j] for j in order] + [big[opposite]]
                fine &= not in_circle(pts, [ids[j] for j in order] + [ids[opposite]])
        if fine:
            kept.add(diagonal)
    assert len(kept) == 1
    low = min(range(4), key=lambda j: ids[j])
    for order in itertools.permutations([j for j in range(4) if j != low]):
        assert not in_circle([big[j] for j in order] + [big[low]], [ids[j] for j in order] + [ids[low]])
    return kept.pop(), low


def exact_figure():
    centre, quad, big, scale, values = cocircular()
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 12.8))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.075, top=0.955, wspace=0.06, hspace=0.33)
    name = "ABCD"
    radius = math.dist(centre, quad[0])

    def lattice(ax, size, label):
        sketch(ax)
        ax.set_xlim(-0.5, size + 0.5)
        ax.set_ylim(-0.5, size + 0.5)
        grid = [[(v, 0), (v, size)] for v in range(size + 1)] + [[(0, v), (size, v)] for v in range(size + 1)]
        ax.add_collection(LineCollection(grid, colors="0.8", linewidths=THIN, zorder=1))
        ax.text(0, 1.03, label, fontsize=15, va="bottom", transform=ax.transAxes)

    def caption(ax, text):
        ax.text(0, -0.02, text, fontsize=13, va="top", transform=ax.transAxes, linespacing=1.3)

    # (a) snapping: down to the grid point below and to the left
    ax = axes[0, 0]
    lattice(ax, 8, "(a) Snapping to the grid")
    rng = np.random.default_rng(3)
    for x in rng.uniform(0.3, 7.7, (9, 2)):
        k = np.floor(x)
        arrow(ax, x, k + 0.06 * (x - k) / max(np.linalg.norm(x - k), 1e-9), color=FAINT, lw=1.3, scale=12)
        ax.plot(*x, marker="o", ms=8, mfc="white", mec=INK, mew=1.4, ls="none", zorder=7)
        ax.plot(*k, marker="o", ms=8, mfc=INK, mec="white", mew=1.0, ls="none", zorder=7)
    caption(ax, "Each position goes to the grid point below and left of it\n"
            "(floor). Drawn: 8 steps across; the code has 2³⁰.")

    # (b) the tie
    ax = axes[0, 1]
    lattice(ax, 16, "(b) Four points on one circle")
    ax.add_patch(Circle(centre, radius, fc="none", ec=INK, lw=HEAVY, zorder=3))
    for p, q, colour in ((0, 2, TEST[1]), (1, 3, REBUILD[1])):
        ax.plot(*zip(quad[p], quad[q]), color=colour, lw=HEAVY, ls=(0, (4, 2.5)), zorder=4)
    ax.add_patch(polygon_patch(quad, fc="none", ec=INK, lw=1.2, zorder=4))
    for j, q in enumerate(quad):
        ax.plot(*q, marker="o", ms=10, mfc=INK, mec="white", mew=1.2, ls="none", zorder=7)
        away = (np.array(q) - centre) / radius
        outlined(ax, *(np.array(q) + 0.9 * away), name[j], fontsize=15, ha="center", va="center", fontweight="bold")
    outlined(ax, centre[0] + 0.3, centre[1] + 0.5, "?", fontsize=22, ha="center", va="center", fontweight="bold")
    up = max(values.values())
    down = min(values.values())
    caption(ax, f"Scaled onto the 2³⁰ grid (×{scale:,}), the rounded in-circle value\n"
            f"is {superscript(up)} in one argument order and {superscript(down)} in another,\n"
            "inside its error bound, so double precision cannot decide.\n"
            "In exact integers it is 0: a true tie.")

    # (c), (d) the perturbation by particle id
    for ax, ids, label in ((axes[1, 0], [7, 3, 12, 5], "(c) Ties go by particle id"),
                           (axes[1, 1], [2, 9, 6, 4], "(d) Other ids, other diagonal")):
        lattice(ax, 16, label)
        (p, q), low = tie_break(big, ids)
        ax.add_patch(Circle(centre, radius, fc="none", ec="0.6", lw=1.2, zorder=3))
        others = [j for j in range(4) if j not in (p, q)]
        for r in others:
            ax.add_patch(polygon_patch([quad[p], quad[q], quad[r]], fc=REPAIR[0], ec=REPAIR[1], lw=HEAVY, zorder=4))
        dead = others
        ax.plot(*zip(quad[dead[0]], quad[dead[1]]), color="0.55", lw=1.2, ls=(0, (2, 2)), zorder=4)
        out = (np.array(quad[low]) - centre) / radius
        arrow(ax, np.array(quad[low]) + 0.35 * out, np.array(quad[low]) + 1.9 * out, color=REFUSE[1], lw=2.0, scale=16)
        for j, point in enumerate(quad):
            ax.plot(*point, marker="o", ms=10, mfc=REFUSE[1] if j == low else INK, mec="white", mew=1.2, ls="none", zorder=7)
            away = (np.array(point) - centre) / radius
            side = np.array([-away[1], away[0]])
            outlined(ax, *(np.array(point) + 1.0 * side + 0.25 * away), f"{name[j]}: id {ids[j]}", fontsize=13,
                     ha="center", va="center")
        caption(ax, f"{name[low]} has the lowest id. Its lift grows most (by εⁱᵈ),\n"
                f"so {name[low]} counts as outside the circle of the other three.\n"
                f"Kept: diagonal {name[p]}{name[q]}, from every argument order.")
    save(fig, "construction_exact.png")


# ---- figure 5: rings to simplices, and a face as the ring of circumcentres ----
def circumcentre3(p):
    """Circumcentre of four 3D points."""
    a = 2 * (p[1:] - p[0])
    rhs = np.sum(p[1:] ** 2 - p[0] ** 2, axis=1)
    return np.linalg.solve(a, rhs)


def rings_figure():
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.6))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.2, top=0.92, wspace=0.08)

    # (a) 2D: the ring of one clipped cell and the triangles it emits
    ax = axes[0]
    n = 60
    def owned(j):
        c = Clipper(cl, cl.order.index(j))
        r = [cl.order[sl] for _, sl in c.edges]
        return c, r, sum(j < r[m] and j < r[(m + 1) % len(r)] for m in range(len(r)))
    for seed in range(2, 40):
        x = relaxed(n, seed, 3)
        k = snapped(x)
        cl = CellList(k)
        central = sorted(range(n), key=lambda j: ((x[j] - 0.5) ** 2).sum())[:6]
        found = [j for j in central if len(owned(j)[1]) == 6 and 2 <= owned(j)[2] <= 3]
        if found:
            break
    i = found[0]
    clip, ring, _ = owned(i)
    tiled = np.concatenate([k / STEPS + (a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)])
    tri = Delaunay(tiled)
    neighbours = set(tri.vertex_neighbor_vertices[1][tri.vertex_neighbor_vertices[0][4 * n + i]:
                                                     tri.vertex_neighbor_vertices[0][4 * n + i + 1]] % n)
    assert neighbours == set(ring)
    g = k[i] / STEPS
    offsets = {q: np.array(clip.edges[m][0]) / STEPS for m, q in enumerate(ring)}
    pos = {q: g + offsets[q] for q in ring}
    vor = Voronoi(tiled)
    sketch(ax)
    half = 0.27
    ax.set_xlim(g[0] - half, g[0] + half)
    ax.set_ylim(g[1] - half, g[1] + half)
    segments = [vor.vertices[e] for e in vor.ridge_vertices if -1 not in e]
    ax.add_collection(LineCollection(segments, colors="0.8", linewidths=THIN, zorder=1))
    for m, q in enumerate(ring):
        r = ring[(m + 1) % len(ring)]
        mine = i < q and i < r
        ax.add_patch(polygon_patch([g, pos[q], pos[r]], fc=REPAIR[0] if mine else "none", ec=REPAIR[1] if mine else "0.55",
                                   lw=HEAVY if mine else 1.2, ls="-" if mine else (0, (3, 2)), zorder=3))
    ax.add_patch(polygon_patch(np.array(clip.vertices()) / STEPS + g, fc="none", ec=REBUILD[1], lw=HEAVY, zorder=4))
    others = np.array([tiled[j] for j in range(len(tiled)) if np.all(np.abs(tiled[j] - g) < half)])
    ax.plot(*others.T, marker="o", ms=6, mfc=FAINT, mec="white", mew=0.8, ls="none", zorder=5)
    for m, q in enumerate(ring):
        ax.plot(*pos[q], marker="o", ms=8, mfc=INK, mec="white", mew=1.0, ls="none", zorder=6)
        away = offsets[q] / np.linalg.norm(offsets[q])
        outlined(ax, *(pos[q] + away * 0.03), f"{q}", fontsize=13, ha="center", va="center")
    ax.plot(*g, marker="o", ms=11, mfc=INK, mec="white", mew=1.4, ls="none", zorder=7)
    outlined(ax, g[0] - 0.012, g[1] + 0.018, f"i = {i}", fontsize=13, fontweight="bold", ha="right")
    ax.text(0, 1.03, "(a) 2D: a cell's ring gives triangles", fontsize=15, va="bottom", transform=ax.transAxes)
    ax.text(0, -0.03, f"Cell {i} (blue) and its ring of neighbours, labelled by id.\n"
            f"Triangle (i, q, r) of two ring neighbours belongs to its lowest id:\n"
            f"cell {i} writes the green ones; the dashed ones are written\nby a neighbour with a lower id.",
            fontsize=13, va="top", transform=ax.transAxes, linespacing=1.3)

    # (b) 3D: the tetrahedra around edge i-j, seen along it; their circumcentres are the face of i and j
    ax = axes[1]
    rng = np.random.default_rng(11)
    pts = rng.random((400, 3))
    tet = Delaunay(pts)
    best = None
    for simplex in tet.simplices:
        for a, b in itertools.combinations(simplex, 2):
            if np.all(np.abs(pts[[a, b]] - 0.5) < 0.2):
                around = [t for t in range(len(tet.simplices)) if a in tet.simplices[t] and b in tet.simplices[t]]
                if len(around) == 6:
                    best = (min(a, b), max(a, b), around)
                    break
        if best:
            break
    i3, j3, around = best
    e = pts[j3] - pts[i3]
    e /= np.linalg.norm(e)
    u = np.cross(e, [0.3, 0.5, 0.8])
    u /= np.linalg.norm(u)
    v = np.cross(e, u)
    mid = 0.5 * (pts[i3] + pts[j3])

    def project(point):
        d = point - mid
        return np.array([d @ u, d @ v])
    centres = [circumcentre3(pts[tet.simplices[t]]) for t in around]
    for c in centres:
        assert abs((c - mid) @ e) < 1e-9
    ring3 = {q for t in around for q in tet.simplices[t]} - {i3, j3}
    angle = lambda q: math.atan2(*project(pts[q])[::-1])
    ring3 = sorted(ring3, key=angle)
    order = sorted(range(len(around)), key=lambda m: math.atan2(*project(centres[m])[::-1]))
    face = np.array([project(centres[m]) for m in order])
    sketch(ax)
    span = 1.35 * max(np.abs(np.array([project(pts[q]) for q in ring3])).max(), np.abs(face).max())
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    for m, q in enumerate(ring3):
        r = ring3[(m + 1) % len(ring3)]
        ax.add_patch(polygon_patch([(0, 0), project(pts[q]), project(pts[r])], fc="none", ec="0.6", lw=1.2, zorder=2))
    ax.add_patch(polygon_patch(face, fc=REBUILD[0], ec=REBUILD[1], lw=HEAVY, zorder=3, alpha=0.95))
    ax.plot(*face.T, marker="o", ms=8, mfc=REBUILD[1], mec="white", mew=1.0, ls="none", zorder=5)
    for q in ring3:
        pq = project(pts[q])
        ax.plot(*pq, marker="o", ms=8, mfc=INK, mec="white", mew=1.0, ls="none", zorder=6)
        outlined(ax, *(pq + pq / np.linalg.norm(pq) * 0.08 * span), f"q{ring3.index(q) + 1}", fontsize=13,
                 ha="center", va="center")
    ax.plot(0, 0, marker="o", ms=12, mfc=INK, mec="white", mew=1.4, ls="none", zorder=7)
    ax.plot([0, -0.27 * span], [0, 0.82 * span], color=INK, lw=THIN, zorder=4)
    outlined(ax, -0.55 * span, 0.84 * span, "edge i-j, seen end on", fontsize=12.5, va="bottom")
    ax.text(0, 1.03, "(b) 3D: the face of i and j", fontsize=15, va="bottom", transform=ax.transAxes)
    ax.text(0, -0.03, f"The {len(around)} tetrahedra (i, j, qₘ, qₘ₊₁) around edge i-j, seen along it.\n"
            "Their circumcentres (blue dots) are the corners of the Voronoi\n"
            "face of i and j, which lies in the plane halfway between them.\n"
            "Clipping writes the ring q1 ... qₖ; the start reads it back as these tetrahedra.",
            fontsize=13, va="top", transform=ax.transAxes, linespacing=1.3)
    save(fig, "construction_rings.png")

if __name__ == "__main__":
    clipping_figure()
    cell_list_figure()
    flip_figure()
    exact_figure()
    rings_figure()
