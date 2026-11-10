#!/usr/bin/env python3
"""Draws docs/figures/gpu_access_build.png and gpu_access_step.png: how GPU threads touch memory, on small examples
computed with the library's rules (Morton keys, make_stencil, pool_chunk, face numbering; see mesh_order.py) on random
points, with Delaunay tetrahedra from scipy.spatial."""
import itertools

import numpy as np
from matplotlib.patches import Rectangle
from scipy.spatial import Delaunay

import mesh_order
from diagram import FAINT, INK, MONO, REBUILD, REPAIR, SETUP, TEST, arrow, canvas, finish, save

RUN = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]


# ---- the library's rules, in Python ----
def spread(v, dims, bits=10):
    out = 0
    for k in range(bits):
        out |= ((v >> k) & 1) << (dims * k)
    return out


def morton3(x, y, z):  # neighbor_search.hpp: morton, x bit above y above z
    return spread(x, 3) << 2 | spread(y, 3) << 1 | spread(z, 3)


def morton2(x, y):  # neighbor_search.hpp: morton in 2D, x bit above y
    return spread(x, 2) << 1 | spread(y, 2)


def stencil(max_reach=4):  # shell_grid.hpp: make_stencil
    beyond = lambda d: max(abs(d) - 1, 0)
    entries = []
    for dx, dy, dz in itertools.product(range(-max_reach, max_reach + 1), repeat=3):
        gaps = beyond(dx) ** 2 + beyond(dy) ** 2 + beyond(dz) ** 2
        if gaps < max_reach ** 2:
            entries.append((gaps, (dx + 8) | (dy + 8) << 4 | (dz + 8) << 8, (dx, dy, dz)))
    entries.sort()
    return [e[2] for e in entries]


def pool_chunk(t, chunks):  # shell_voronoi.hpp: pool_chunk
    return (((t * 2654435761) & 0xFFFFFFFF) * chunks) >> 32


EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # tet_mesh.hpp: edge_ends


def label(ax, x, y, text, size=10, **kw):
    kw.setdefault("color", INK)
    ax.text(x, y, text, fontsize=size, **kw)


def panel_title(ax, x, y, letter, text):
    ax.text(x, y, f"({letter}) {text}", fontsize=12.5, fontweight="bold", va="top", color=INK)


def strip(ax, x, y, cells, width, height=0.3, colours=None, numbers=None, size=8.5, edge="#9a9a9a"):
    """A row of memory cells from (x, y), each width wide; colours and numbers per cell (None: blank)."""
    for k in range(cells):
        fc = colours[k] if colours and colours[k] else "white"
        ax.add_patch(Rectangle((x + k * width, y), width, height, fc=fc, ec=edge, lw=0.5, zorder=3))
        if numbers and numbers[k] is not None:
            ax.text(x + (k + 0.5) * width, y + height / 2, str(numbers[k]), ha="center", va="center", fontsize=size,
                    color=INK, zorder=4)


# ---- figure 1: the clipping (rebuild) ----
def build_figure():
    W, H = 13.0, 30
    fig, ax = canvas(W, H)
    rng = np.random.default_rng(7)
    top = H - 0.2

    # (a) Morton order in 2D, 4 x 4 grid, one particle per cell, caller ids random.
    panel_title(ax, 0.2, top, "a", "Morton order: slots of the cell list against the caller's particle ids")
    side, n = 3.2, 16
    gx, gy = 0.5, top - 0.6 - side
    ids = rng.permutation(n)
    cells = [(cx, cy) for cx in range(4) for cy in range(4)]
    pts = {}
    for k, (cx, cy) in enumerate(cells):
        pts[ids[k]] = ((cx + 0.25 + 0.5 * rng.random()) / 4, (cy + 0.25 + 0.5 * rng.random()) / 4)
    for k in range(5):
        ax.plot([gx, gx + side], [gy + k * side / 4] * 2, color="#bbbbbb", lw=0.8)
        ax.plot([gx + k * side / 4] * 2, [gy, gy + side], color="#bbbbbb", lw=0.8)
    order = sorted(pts, key=lambda i: (morton2(int(pts[i][0] * 4), int(pts[i][1] * 4)), i))
    path = [(gx + pts[i][0] * side, gy + pts[i][1] * side) for i in order]
    ax.plot(*zip(*path), color=REBUILD[1], lw=1.4, zorder=2)
    for q, i in enumerate(order):
        px, py = path[q]
        ax.plot(px, py, "o", ms=5, color=INK, zorder=4)
        label(ax, px + 0.06, py + 0.05, f"{i}", size=10, zorder=5, fontweight="bold")
        label(ax, px + 0.06, py - 0.17, f"q{q}", size=8, color=REBUILD[1], zorder=5)
    label(ax, gx, gy - 0.25, "bold: particle id i (caller's order); blue: slot q", size=9.5, va="top")
    sx, w = 4.4, 0.5
    label(ax, sx, top - 0.75, "particle_order(q): slot q holds particle", size=10, va="center")
    strip(ax, sx, top - 1.25, n, w, numbers=[int(i) for i in order],
          colours=[REBUILD[0]] * n)
    for q in range(n):
        label(ax, sx + (q + 0.5) * w, top - 1.4, f"{q}", size=8, ha="center", va="center", color=FAINT)
    label(ax, sx, top - 1.85, "positions(i): row i is particle i, wherever it lies", size=10, va="center")
    near = order[:4]
    strip(ax, sx, top - 2.35, n, w, numbers=list(range(n)),
          colours=[REPAIR[0] if i in near else None for i in range(n)])
    label(ax, sx, top - 2.7,
          "Slots 0 to 3 (one quadrant) are particles " + ", ".join(str(int(i)) for i in near) +
          ": neighbours in space sit apart in\nthe arrays indexed by particle (green). The clipping reads slot "
          "order; every step kernel reads\nparticle order, which is the caller's. voronoi_bench numbers particles "
          "along a Morton curve,\nso its steps see slot-like order; the mesh itself never re-sorts (#34).", size=9.5,
          va="top", linespacing=1.3)
    top = gy - 0.75

    # (b) Warp per cell: the first batch of the stencil and the runs it reads.
    panel_title(ax, 0.2, top, "b", "Clipping: one warp per cell, one grid cell per lane, nearest first")
    st = stencil()
    batch = st[:32]
    sgx, sgy, cs = 0.6, top - 0.7 - 2.6, 0.52
    label(ax, sgx, top - 0.55, "z = 0 slice of the stencil: lane of each grid cell", size=10, va="top")
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            x, y = sgx + (dx + 2) * cs, sgy + (dy + 2) * cs
            lane = batch.index((dx, dy, 0)) if (dx, dy, 0) in batch else None
            fc = REBUILD[0] if lane is not None else "white"
            if (dx, dy) == (0, 0):
                fc = TEST[0]
            ax.add_patch(Rectangle((x, y), cs, cs, fc=fc, ec="#9a9a9a", lw=0.6))
            if lane is not None:
                label(ax, x + cs / 2, y + cs / 2, str(lane), size=9, ha="center", va="center")
    label(ax, sgx, sgy - 0.15, "orange: home cell. Lanes 0 to 26 take the\n27 cells around home (3 × 3 × 3), lanes 27\n"
          "to 31 the first of the next shell; empty cells\nwait for the next batch of 32.", size=9, va="top",
          linespacing=1.25)
    # Real runs: 8³ base grid, 512 uniform points, Morton slots; the runs of the 32 lanes for a home cell.
    g = 8
    p = rng.random((512, 3))
    c = np.floor(p * g).astype(int)
    keys = np.array([morton3(*cc) for cc in c])
    slots = np.argsort(keys, kind="stable")
    sorted_keys = keys[slots]
    home = (3, 4, 3)
    runs = []
    for lane, (dx, dy, dz) in enumerate(batch):
        cc = ((home[0] + dx) % g, (home[1] + dy) % g, (home[2] + dz) % g)
        key = morton3(*cc)
        begin = int(np.searchsorted(sorted_keys, key, "left"))
        end = int(np.searchsorted(sorted_keys, key, "right"))
        runs.append((lane, begin, end))
    rx, rw = 4.4, 8.4
    per = rw / 512
    label(ax, rx, top - 0.55, "slot_positions of 512 uniform particles on an 8³ grid (one per cell on average):\n"
          "the run [begin, end) each lane reads, from cell_offsets", size=10, va="top", linespacing=1.3)
    ry = top - 1.55
    ax.add_patch(Rectangle((rx, ry), rw, 0.3, fc="white", ec="#9a9a9a", lw=0.6))
    for lane, begin, end in runs:
        if end > begin:
            ax.add_patch(Rectangle((rx + begin * per, ry), max((end - begin) * per, 0.02), 0.3,
                                   fc=RUN[lane % len(RUN)], ec="none", zorder=3))
    for s in (0, 128, 256, 384, 512):
        label(ax, rx + s * per, ry - 0.15, str(s), size=8.5, ha="center", va="center", color=FAINT)
    filled = sum(1 for _, b, e in runs if e > b)
    total = sum(e - b for _, b, e in runs)
    spans = max(e for _, b, e in runs if e > b) - min(b for _, b, e in runs if e > b)
    label(ax, rx, ry - 0.45,
          f"Here {filled} of the 32 lanes find particles, {total} in all, spread over {spans} slots. In step k every "
          "lane loads\nslot begin + k of its own run: 32 separate addresses per load, so the loads do not "
          "coalesce, but each run\nis contiguous, so later k hit the same sectors. A ballot collects the particles "
          "within 2R into a buffer of\n32 candidates in shared memory; a warp bitonic sort orders them, and "
          "each is broadcast (shuffle) to\nall lanes, which cut the polyhedron together: lane l owns vertices l "
          "and l + 32.", size=9.5, va="top", linespacing=1.3)
    label(ax, rx, ry - 1.85,
          "Measured (Nsight Compute, 128³): the clipping kernel waits on memory for 5 to 8 % of its stalls; it waits "
          "on its own\narithmetic and shared memory, and on uniform, clustered and halo input on instruction fetch "
          "(39 to 43 %).\n96 registers, 18 KB of shared memory per block of 4 warps; 20 of 64 warps active.",
          size=9.5, va="top", color=FAINT, linespacing=1.3)
    top = min(sgy - 1.1, ry - 2.65) - 0.35

    # (c) Claiming room in the ring pool.
    panel_title(ax, 0.2, top, "c", "Claiming room: the ring pool in n / 1024 chunks, a chunk per cell by hash")
    chunks, cells_shown = 8, 12
    px0, pw = 0.6, 11.6
    py = top - 1.4
    cw = pw / chunks
    for k in range(chunks):
        ax.add_patch(Rectangle((px0 + k * cw, py), cw, 0.4, fc="white", ec=INK, lw=0.9, zorder=3))
        label(ax, px0 + (k + 0.5) * cw, py - 0.18, f"chunk {k}", size=9, ha="center", va="center", color=FAINT)
    used = [0] * chunks
    for t in range(cells_shown):
        ch = pool_chunk(t, chunks)
        need = 0.18 + 0.1 * ((t * 5) % 3)
        x = px0 + ch * cw + used[ch]
        ax.add_patch(Rectangle((x, py), need * cw / 1.2, 0.4, fc=RUN[t % len(RUN)], ec="white", lw=0.6, zorder=4,
                               alpha=0.85))
        label(ax, x + need * cw / 2.4, py + 0.2, str(t), size=9, ha="center", va="center", zorder=5, color="white",
              fontweight="bold")
        used[ch] += need * cw / 1.2
    label(ax, px0, top - 0.55, f"cells (slots) 0 to {cells_shown - 1} of n = 8192 (8 chunks) and the chunk each "
          "claims in (shell_voronoi.hpp: pool_chunk, claim): neighbouring slots spread over the chunks.",
          size=9.5, va="top")
    label(ax, px0, py - 0.45,
          "Lane 0 of the warp claims all of a cell's entries with one atomic_fetch_add on ring_used(chunk); the lanes "
          "then write, for each owned face,\n[neighbour, ring length, ring particles ...] at offsets from an "
          "exclusive scan over lanes. A full chunk marks its cells face_overflow and\nthe build runs again with every "
          "chunk as large as the fullest + 1/8 (until_pool_fits). Block widths here are schematic.",
          size=9.5, va="top", linespacing=1.3)
    finish(fig, ax, py - 1.4)
    save(fig, "gpu_access_build.png")


# ---- a small real tetrahedral mesh with the library's numbering ----
def small_mesh():
    """Jittered 6³ points numbered along a Morton curve (as voronoi_bench), scipy's Delaunay tetrahedra; slots
    grouped by the two smallest particles (assemble), faces numbered by the scan over edge bits (number_faces)."""
    rng = np.random.default_rng(3)
    g = 6
    base = np.array(list(itertools.product(range(g), repeat=3)), float)
    p = (base + 0.5 + 0.3 * (rng.random(base.shape) - 0.5)) / g
    keys = [morton3(*(np.floor(q * 8).astype(int))) for q in p]
    p = p[np.argsort(keys, kind="stable")]
    tets = [tuple(sorted(t)) for t in Delaunay(p).simplices]
    tets.sort()
    slot = {t: k for k, t in enumerate(tets)}
    around = {}
    for t in tets:
        for a, b in EDGES:
            around.setdefault((t[a], t[b]), []).append(slot[t])
    # Representative: the lowest slot on the edge (represent_dropped's rule); faces by the scan over slots.
    face = {}
    f = 0
    for k, t in enumerate(tets):
        for a, b in EDGES:
            e = (t[a], t[b])
            if min(around[e]) == k:
                face[e] = f
                f += 1
    return p, tets, around, face


def ring_of(tets, around, e):
    """The tetrahedra around edge e in walking order, from the one whose other two ids are least
    (walk_edge_canonical); open rings at the hull are left in slot order."""
    members = around[e]
    others = {s: tuple(sorted(set(tets[s]) - set(e))) for s in members}
    adj = {s: [u for u in members if u != s and len(set(tets[s]) & set(tets[u])) == 3] for s in members}
    if any(len(v) != 2 for v in adj.values()):
        return sorted(members), False
    start = min(members, key=lambda s: others[s])
    nxt = min(adj[start], key=lambda s: others[s])
    out, before, cur = [start], start, nxt
    while cur != start:
        out.append(cur)
        before, cur = cur, [u for u in adj[cur] if u != before][0]
    return out, True


# ---- figure 2: a step ----
def step_figure():
    W, H = 13.0, 30
    fig, ax = canvas(W, H)
    p, tets, around, face = small_mesh()
    top = H - 0.2
    n = len(p)

    # Pick an interior particle with closed rings.
    i = next(v for v in range(n // 2 - 10, n) if all(ring_of(tets, around, e)[1]
                                                    for e in around if v in e))
    edges_i = sorted([e for e in face if i in e], key=lambda e: e[0] + e[1] - i)
    j = edges_i[2][0] + edges_i[2][1] - i
    e = (min(i, j), max(i, j))
    ring, _ = ring_of(tets, around, e)
    f = face[e]

    # (d) One face, computed once by one thread.
    panel_title(ax, 0.2, top, "d", f"Face geometry: thread f = {f} computes the face of particles {e[0]} and {e[1]} "
                "once (tet_mesh.hpp: faces)")
    label(ax, 0.2, top - 0.42, f"From jittered 6³ points numbered along a Morton curve: {n} particles, {len(tets)} "
          f"tetrahedra, {len(face)} faces ({mesh_order.FACE}; representative: lowest slot on the edge).", size=9.5, va="top", color=FAINT)
    y = top - 1.15
    x0, cw = 2.7, 0.62
    rows = [("owner, neighbour", [f - 1, f, f + 1], [None, f"{e[0]}, {e[1]}", None]),
            ("ring_start, _length", [f - 1, f, f + 1], [None, f"·, {len(ring)}", None])]
    for name, idx, vals in rows:
        label(ax, 0.2, y + 0.15, name, size=9.5, va="center", family=MONO)
        strip(ax, x0, y, 3, 1.3, numbers=[f"f={k}" if v is None else v for k, v in zip(idx, vals)],
              colours=[None, REPAIR[0], None], size=8.5)
        y -= 0.5
    label(ax, x0 + 4.1, top - 0.92, "thread f reads row f: neighbouring threads, neighbouring rows: coalesced",
          size=9.5, va="center")
    label(ax, 0.2, y + 0.15, "ring pool", size=9.5, va="center", family=MONO)
    strip(ax, x0, y, len(ring) + 2, cw, numbers=[None] + ring + [None], colours=[None] + [REPAIR[0]] * len(ring) + [None])
    label(ax, x0 + (len(ring) + 2) * cw + 0.15, y + 0.15, f"{len(ring)} tetrahedron slots, contiguous for this face; "
          "the next thread reads another run", size=9.5, va="center")
    y -= 0.85
    label(ax, 0.2, y + 0.15, "tet_centre rows", size=9.5, va="center", family=MONO)
    span = max(ring) - min(ring) + 1
    width = min(10.0 / span, 0.2)
    lo = min(ring)
    cols = [REBUILD[1] if lo + k in ring else None for k in range(span)]
    strip(ax, x0, y, span, width, height=0.3, colours=cols)
    label(ax, x0, y - 0.18, f"slots {lo} to {lo + span - 1}: a gather of {len(ring)} rows of 32 B (one sector "
          "each), as many as the ring is long", size=9.5, va="top")
    for k, s in enumerate(ring):
        arrow(ax, [(x0 + (k + 1.5) * cw, y + 0.82), (x0 + (s - lo + 0.5) * width, y + 0.33)], color="#9a9a9a", lw=0.8)
    y -= 0.85
    label(ax, 0.2, y + 0.15, "area, face_centroid", size=9.5, va="center", family=MONO)
    strip(ax, x0, y, 3, 1.3, numbers=["f-1", f"f={f}", "f+1"], colours=[None, REPAIR[0], None], size=8.5)
    label(ax, x0 + 4.1, y + 0.15, "written once, at row f: coalesced", size=9.5, va="center")
    y -= 0.5
    label(ax, 0.2, y + 0.15, "cell_volume", size=9.5, va="center", family=MONO)
    label(ax, x0, y + 0.15, f"atomic_add of the pyramid to rows {e[0]} and {e[1]}: a scatter, and atomics add in any "
          "order, so the last bits\ncan differ run to run; update_cells() sums in a fixed order (panel f).", size=9.5,
          va="center", linespacing=1.3)
    y -= 0.75
    lst = [k for k in edges_i]
    label(ax, 0.2, y + 0.15, f"cell_face of {i}", size=9.5, va="center", family=MONO)
    strip(ax, x0, y, len(lst), 0.62, numbers=[face[k] for k in lst],
          colours=[REPAIR[0] if face[k] == f else SETUP[0] for k in lst], size=8.5)
    label(ax, x0, y - 0.18, f"cell {i}'s faces, by the other particle ({', '.join(str(k[0] + k[1] - i) for k in lst)}): "
          f"\nits gather reads the area rows above, scattered over the faces; cell {j} reads row {f} too, with the sign "
          "and centroid\nturned to its side (detail::pyramid).", size=9.5,
          va="top", linespacing=1.3)
    top = y - 0.95

    # (e) A 2-3 flip as gather and scatter on the tetrahedron rows.
    panel_title(ax, 0.2, top, "e", "A flip: gathers to plan and vote, scatters to apply (tet_repair.hpp: repair)")
    t = next(s for s, tt in enumerate(tets) if i in tt)
    adj_t = [s for s, tt in enumerate(tets) if s != t and len(set(tt) & set(tets[t])) == 3]
    u = max(adj_t, key=lambda s: abs(s - t))
    neighbours = sorted({s for s, tt in enumerate(tets) for o in (t, u) if s not in (t, u)
                         and len(set(tt) & set(tets[o])) == 3})
    free_slot = len(tets)
    y = top - 0.75
    total = len(tets) + 1
    xs, span_w = 0.4, 12.0
    at = lambda s: xs + (s + 0.5) * span_w / total
    ax.add_patch(Rectangle((xs, y - 0.35), span_w, 0.35, fc="#f2f2f2", ec="#9a9a9a", lw=0.6))
    for s in neighbours:
        ax.plot([at(s)] * 2, [y - 0.35, y], color=TEST[1], lw=2.2, zorder=4)
    for s, colour in ((t, REPAIR[1]), (u, REPAIR[1]), (free_slot, REBUILD[1])):
        ax.plot([at(s)] * 2, [y - 0.45, y + 0.1], color=colour, lw=3, zorder=5)
    label(ax, xs, y + 0.25, f"tet_vertex and tet_adjacent rows, slots 0 to {total - 1} (int[4] each, 16 B: one sector "
          "per random read); green: t and u, orange: their outer neighbours, blue: the new slot", size=9.5,
          va="center")
    label(ax, at(t), y - 0.5, f"t={t}", size=9, ha="right" if t < u else "left", va="top", color=REPAIR[1],
          fontweight="bold")
    label(ax, at(u), y - 0.5, f"u={u}", size=9, ha="left" if t < u else "right", va="top", color=REPAIR[1],
          fontweight="bold")
    label(ax, at(free_slot), y - 0.5, "new", size=9, ha="right", va="top", color=REBUILD[1], fontweight="bold")
    label(ax, xs, y - 0.95,
          f"plan (one thread per failing triangle t|u, if it failed): reads rows t and u and their outer neighbours ({len(neighbours)} "
          "orange rows, some hidden next to t and u), votes with atomic_max on the\nlock of each of them. apply: the winner rewrites rows t and u, "
          "takes a third slot from free_slots or the end of the array (atomic counter),\nand writes the back pointer "
          "adjacent(w, ·) of each outer neighbour w: scattered single-int stores. Rows near in slot number are near "
          "in\nspace only as far as the start's order and the reuse of freed slots keep them.", size=9.5,
          va="top", linespacing=1.3)
    label(ax, xs, y - 2.05, "Measured (Nsight Compute, 128³ jittered 0.1): the step kernels wait on memory for 63 to "
          "93 % of their stalls, at 280 to 860 GB/s of\nthe A100's 1555; apply runs 6.3 threads per instruction, "
          "one flip per thread with the rest of the warp idle in its branches.", size=9.5, va="top", color=FAINT,
          linespacing=1.3)
    top = y - 2.85

    # (f) Fixed-order sums.
    panel_title(ax, 0.2, top, "f", "Fixed-order sums: CellSumsWarp, 8 cells per warp, 4 lanes per cell (tet_cells.hpp)")
    y = top - 0.8
    entries = len(lst)
    batches = (entries + 3) // 4
    lw_ = 0.33
    first = i - i % 8  # CellSumsWarp: cell = (league * warps + team) * 8 + lane / 4
    for cell in range(8):
        for lane in range(4):
            x = 0.6 + (cell * 4 + lane) * lw_
            fc = REPAIR[0] if first + cell == i else "white"
            ax.add_patch(Rectangle((x, y), lw_, 0.35, fc=fc, ec="#9a9a9a", lw=0.5))
            label(ax, x + lw_ / 2, y + 0.175, str(cell * 4 + lane), size=7.5, ha="center", va="center")
        label(ax, 0.6 + (cell * 4 + 2) * lw_, y + 0.5, f"cell {first + cell}", size=8.5, ha="center", va="center",
              color=FAINT)
    label(ax, 0.6 + 32 * lw_ + 0.25, y + 0.175, "lanes of one warp", size=9.5, va="center")
    y -= 0.55
    for b in range(batches):
        yy = y - 0.45 * b
        label(ax, 0.6, yy, f"batch {b}:", size=9.5, va="center")
        for lane in range(4):
            k = lane + 4 * b
            text = f"lane {lane}: entry {k} (face {face[lst[k]]})" if k < entries else f"lane {lane}: past the end, adds 0"
            label(ax, 1.5 + 2.75 * lane, yy, text, size=9, va="center", family=MONO)
    y -= 0.45 * batches + 0.1
    label(ax, 0.6, y, f"Cell {i} (lanes {4 * (i % 8)} to {4 * (i % 8) + 3}) has {entries} list entries, so {batches} batches. After each batch the 4 lanes "
          "broadcast their pyramids by shuffle in lane order 0, 1, 2, 3, and every\nlane adds them in that order: "
          "entries 0, 1, 2, ... in list order, the same bits as the sequential loop (CellLists::sum) and the same "
          "every run.\nCellSortWarp first sorts each changed list: a warp per cell, lane l holds entry l, a bitonic "
          "sort by the other particle (over 32 entries: lane 0\nsorts alone). The list rows of a cell are contiguous; "
          "the area rows they point to are a gather, as in (d).", size=9.5, va="top", linespacing=1.3)
    finish(fig, ax, y - 1.25)
    save(fig, "gpu_access_step.png")


if __name__ == "__main__":
    build_figure()
    step_figure()
