#!/usr/bin/env python3
"""Draws docs/figures/flow_step.png and flow_rebuild.png: each stage of Mesh<3>::update and of Mesh<3>::rebuild with
its kernels, what they read and write, GPU or host, where the host waits, and measured times. Times are copied from
docs/measure_2026-11-10.md; everything else is read from the code of main (cited file: function)."""
from diagram import (FAINT, HOSTC, INK, REBUILD, REFUSE, REPAIR, SETUP, TEST, WAIT, arrow, canvas, card, card_height,
                     diamond, finish, legend_chips, rect, save, wait)

# Kernel times per step, ms: 128³ jittered at 0.1 spacings per step, Kokkos Tools, median of steps 2 to 6
# (docs/measure_2026-11-10.md, "Cost of a moving step"). The rows sum to the table's total, 22.98 ms.
STEP_MS = {
    "snap": 0.37, "screen": 2.28, "repair": 6.38, "centres": 0.14, "numbers": 1.20, "dirty": 4.42,
    "compact": 0.79, "faces": 2.63, "lists": 1.75, "sort": 1.24, "sums": 1.79,
}
STEP_TOTAL = 22.98
# Optional layout (voronoi::Layout, off by default), docs/measure_resort_2026-11-10.md, 128³ at 0.1 spacings: a Hilbert
# re-sort, when due, and the numbering by owner, every step.
RESORT_MS, OWNER_MS = 11.0, 2.3
GAP = 0.5  # between boxes, room for a wait line


def time_bar(ax, x, y, ms, name=None, scale=0.22):
    """A bar of `ms` milliseconds at (x, y), labelled with ms and its share of the step."""
    rect(ax, x, y - 0.1, max(ms * scale, 0.04), 0.2, ("#c9c9c9", "#7f7f7f"), lw=0.8, radius=0.02)
    label = f"{ms:.2f} ms, {100 * ms / STEP_TOTAL:.0f} %" + (f"  {name}" if name else "")
    ax.text(x + ms * scale + 0.1, y, label, va="center", fontsize=10, color=INK)


class Column:
    """Boxes stacked down one column, each as tall as its text."""

    def __init__(self, ax, x, w, top):
        self.ax, self.x, self.w, self.top, self.boxes = ax, x, w, top, []

    def box(self, title, code, lines, tone, where="GPU", link=True):
        h = card_height(code, lines)
        y = self.top - h
        card(self.ax, self.x, y, self.w, h, title, code, lines, tone, where)
        if link and self.boxes:
            above = self.boxes[-1]
            arrow(self.ax, [(self.x + self.w - 0.35, above), (self.x + self.w - 0.35, self.top)])
        self.boxes.append(y)
        self.top = y - GAP
        return y, h

    def wait(self, y, text):
        wait(self.ax, self.x + 0.1, y - GAP / 2, text)


def step_figure():
    W, H = 12.7, 33
    fig, ax = canvas(W, H)
    x0, w, tx, sx, sw = 0.2, 7.5, 7.95, 10.15, 2.5
    ax.text(x0, H - 0.2, "One step: update(x) on a built mesh with the same n", fontsize=14, fontweight="bold",
            va="top", color=INK)
    ax.text(tx, H - 0.2, "kernel time per step", fontsize=11, va="top", color=INK)
    ax.text(tx, H - 0.47, "128³ jittered, 0.1 spacings", fontsize=10, va="top", color=FAINT)
    c = Column(ax, x0, w, H - 0.9)

    y, h = c.box("1  Snap", "voronoi_mesh.hpp: detail::snap_positions",
                 "one thread per particle; reads x (caller), writes positions (double[4], a padded row)", REPAIR)
    time_bar(ax, tx, y + h / 2, STEP_MS["snap"], "with total_volume")

    y, h = c.box("2  Screen", "tet_repair.hpp: move_screened  (kernel tet_repair::screen)",
                 "one thread per tetrahedron slot; reads vertex, adjacent, positions\n"
                 "writes tet_repair_point (snapped ints), the centre of every live tetrahedron,\n"
                 "dirty and listed (atomic append) for those that may fail: 2.0 % of them here", REPAIR)
    time_bar(ax, tx, y + h / 2, STEP_MS["screen"])
    c.wait(y, "host reads the sliver count (pinned copy, fence); their centres are then made exact")

    y, h = c.box("3  Flip rounds", "tet_repair.hpp: repair  (find, unlist, plan, apply, release)",
                 "8 rounds here, each:\n"
                 "find: a thread per listed tetrahedron, exact in_sphere across its 4 faces,\n"
                 "     atomic append of failing triangles to work\n"
                 "plan: a thread per failing triangle plans a 2-3 or 3-2 flip and votes (atomic_max)\n"
                 "     on the locks of the old tetrahedra and their neighbours\n"
                 "apply: winners rewrite vertex and adjacent rows: old slots, then free slots, then new;\n"
                 "     new tetrahedra go to touched_list and are dirty for the next round", REPAIR)
    time_bar(ax, tx, y + h / 2, STEP_MS["repair"])
    c.wait(y, "host reads counters twice a round: work size (grow, find again), then the rest")

    # Stuck? Decision, with the host finish and the rebuild in the side column.
    dy = c.top - 0.5
    dx = x0 + 1.5
    diamond(ax, dx, dy, 2.4, 0.9, "stuck entries?", size=10.5)
    arrow(ax, [(dx, c.boxes[-1]), (dx, dy + 0.45)])
    ax.text(dx + 1.35, dy - 0.32, "none in the measured steps at 0.03 to 0.3 spacings, except clustered 0.3 (2 of 6);\n"
            "at 1.0 spacings 6 to 12 entries in most steps", fontsize=9.5, color=FAINT, va="top")
    hf_h = card_height("tet_flips.hpp: finish", "a\nb\nc\nd\ne\nf", 9.5)
    card(ax, sx, dy - hf_h + 0.3, sw, hf_h, "Host finish", "tet_flips.hpp: finish",
         "if stuck × 200000 ≤ n: copy\nthe tetrahedra within 8 hops\nto the host, Bowyer-Watson,\nwrite back, flip on the\n"
         "GPU again. Measured: 4 to 9\nms for 6 to 8 entries", HOSTC, "host", size=9.5)
    rb_y = dy - hf_h + 0.3 - 0.55
    rb_h = card_height("flow_rebuild.png", "a\nb\nc\nd", 9.5)
    card(ax, sx, rb_y - rb_h, sw, rb_h, "Rebuild", "flow_rebuild.png",
         "more entries, host failed,\nout of rounds (1000), or the\nvolume check (10) fails.\n250 to 590 ms at 128³",
         REBUILD, "GPU", size=9.5)
    bend = sx - 0.35
    arrow(ax, [(dx + 1.2, dy), (bend, dy), (bend, dy - 0.2), (sx, dy - 0.2)], label="few", at=(bend - 0.6, dy))
    arrow(ax, [(bend, dy - 0.2), (bend, rb_y - rb_h / 2), (sx, rb_y - rb_h / 2)], label="many", at=(bend, rb_y - rb_h / 2 - 0.25),
          size=9.5)
    arrow(ax, [(sx + sw / 2, dy - hf_h + 0.3), (sx + sw / 2, rb_y)], label="still stuck", at=(sx + sw / 2 + 0.6,
                                                                                               rb_y + 0.27), size=9.5)
    c.top = dy - 1.4
    arrow(ax, [(dx, dy - 0.45), (dx, c.top)], label="no", at=(dx + 0.25, dy - 0.75))
    c.boxes.append(None)

    def box_after_decision(*args):
        c.boxes.pop()
        h = card_height(args[1], args[2])
        y = c.top - h
        card(ax, x0, y, w, h, *args, "GPU")
        c.boxes.append(y)
        c.top = y - GAP
        return y, h

    y, h = box_after_decision("4  Re-sort, when due (Layout::every or far; off by default)",
                              "voronoi_mesh.hpp: Mesh::resort → mesh_order.hpp",
                              "keys of a curve from tet_repair_point, radix sort; positions and points gathered;\n"
                              "tetrahedra sorted by their lowest new particle, adjacency renumbered; every face\n"
                              "is then numbered and walked anew; Report::resorted, permutation() for the caller", SETUP)
    ax.text(tx, y + h / 2, f"{RESORT_MS:.1f} ms a re-sort\n(Hilbert), when due", va="center", fontsize=10, color=INK)
    done_y = y + h - 0.3
    y, h = c.box("5  Centres of created tetrahedra", "tet_mesh.hpp: circumcentres (only = touched_list)",
                 "one thread per created tetrahedron (3.3 % of them here); reads vertex, positions;\n"
                 "writes centre (of every tetrahedron after a re-sort)", REPAIR)
    time_bar(ax, tx, y + h / 2 - 0.25, STEP_MS["centres"])
    arrow(ax, [(sx + 0.3, dy - hf_h + 0.3), (sx + 0.3, done_y), (x0 + w, done_y)], label="done",
          at=(x0 + w + 0.7, done_y), size=9.5)

    y, h = c.box("6  Number the faces", "tet_mesh.hpp: update_face_rings → detail::number_faces",
                 "a scan over all tetrahedron slots: face slot s = start(t) + rank of the edge among t's bits;\n"
                 "writes owner, neighbour, ring_start, ring_length; a kept face takes its old ring\n"
                 "through the renumber table (in the spare pool)", REPAIR)
    time_bar(ax, tx, y + h / 2, STEP_MS["numbers"])
    c.wait(y, "host reads the face count; over capacity: realloc with 1/32 more, scan again")

    y, h = c.box("7  Walk the dirty rings", "tet_mesh.hpp: update_face_rings  (dirty_faces, append_rings)",
                 "a thread per created tetrahedron: its 6 edges; atomic_exchange claims each face once\n"
                 "a thread per dirty face (10.4 % here): walk around the edge, claim room at the pool's\n"
                 "end (atomic_fetch_add), walk again in canonical order; old rings stay as gaps", REPAIR)
    time_bar(ax, tx, y + h / 2 + 0.2, STEP_MS["dirty"])
    time_bar(ax, tx, y + h / 2 - 0.25, STEP_MS["compact"], "compact")
    c.wait(y, "host reads both totals; pool full: copy every ring to the spare pool, swap")

    y, h = c.box("8  Number by owner (Layout::owner_faces; off by default)", "tet_mesh.hpp: number_by_owner",
                 "count per owner, scan, atomic fill, each owner's faces in slot order: face f by owner,\n"
                 "owned_start; the renumber table rewritten in face numbers", SETUP)
    ax.text(tx, y + h / 2, f"{OWNER_MS:.1f} ms when on", va="center", fontsize=10, color=INK)

    y, h = c.box("9  Face geometry", "tet_mesh.hpp: faces",
                 "one thread per face, run once for the pair: gathers the ring's centres, writes area,\n"
                 "face_centroid (and float face_rows); atomic_add of the pyramid to both cells' volume", REPAIR)
    time_bar(ax, tx, y + h / 2, STEP_MS["faces"])

    y, h = c.box("10  Volume check", "voronoi_mesh.hpp: detail::total_volume, Mesh::update",
                 "a reduction over volume: |sum / box − 1| < 1e-6, else rebuild (Report::overlapped)", TEST)
    c.wait(y, "host reads the sum; update() returns")

    y, h = c.box("11  update_cells(), when the caller asks", "tet_cells.hpp: cells_incremental or cells",
                 "incremental if created ≤ n/5 and the scratch fits (not here: 490,000 created > n/5)\n"
                 "lists: atomic count per cell, scan, atomic fill (cell_face_counts, _start, _fill)\n"
                 "sort: CellSortWarp, a warp per cell, bitonic by the other particle's id\n"
                 "sums: CellSumsWarp, 4 lanes per cell, volume and centroid in list order\n"
                 "(Layout::owner_bit: bit 31 in the entries of faces the cell does not own)", SETUP, link=False)
    arrow(ax, [(x0 + w - 0.35, c.boxes[-2]), (x0 + w - 0.35, y + h)], label="later", at=(x0 + w - 0.85, y + h + 0.25),
          size=9.5)
    for k, name in enumerate(("lists", "sort", "sums")):
        time_bar(ax, tx, y + h - 0.45 - 0.38 * k, STEP_MS[name], name)

    b = y - 0.35
    ax.text(x0, b, f"Times: Kokkos Tools kernel times, median of steps 2 to 6, 128³ jittered at 0.1 spacings per step; "
            f"the rows sum to {STEP_TOTAL} ms.\nNsight Systems on the same input: 61 launches and 33 host "
            "synchronisations per step, 21.75 ms on the GPU in 22.2 ms of wall time.\nThe fractions (2.0, 3.3, 10.4 %), "
            "rounds, stuck counts and fallback times are from the same note, docs/measure_2026-11-10.md;\nthe re-sort and the "
            "numbering by owner from docs/measure_resort_2026-11-10.md.",
            fontsize=9.5, color=FAINT, va="top", linespacing=1.35)
    b -= 1.2
    legend_chips(ax, x0, b, [(REPAIR, "step"), (TEST, "check"), (SETUP, "optional"), (HOSTC, "host"),
                             (REBUILD, "rebuild")], step=1.45)
    ax.text(x0 + 7.4, b, "▌▌", color=WAIT, fontsize=10, va="center")
    ax.text(x0 + 7.7, b, "the host waits for the GPU", fontsize=10, va="center", color=INK)
    finish(fig, ax, b - 0.25)
    save(fig, "flow_step.png")


def rebuild_figure():
    W, H = 12.7, 30
    fig, ax = canvas(W, H)
    x0, w, sx, sw = 0.2, 8.6, 9.25, 3.3
    ax.text(x0, H - 0.2, "A rebuild: the first update(x), a new n, or a step that cannot be repaired", fontsize=14,
            fontweight="bold", va="top", color=INK)
    c = Column(ax, x0, w, H - 0.9)

    c.box("1  Forget, snap", "voronoi_mesh.hpp: Mesh::rebuild, forget, snap",
          "fences, drops every persistent array; writes positions", REBUILD)
    c.box("2  Cell list", "neighbor_search.hpp: configure (host, when n changes), build",
          "Morton key per particle, radix sort, a level per slot; writes particle_order,\n"
          "slot_positions, cell_offsets (dense base grid), slot_fine_keys, slot_levels", REBUILD)
    y, h = c.box("3  Clip with face rings", "shell_voronoi_warp.hpp: construct_warp, warp::Kernel",
                 "one warp per cell in shared memory; persistent warps take slots from an atomic counter;\n"
                 "reads slot_positions through the stencil; claims ring room in one of n/1024 pool chunks;\n"
                 "writes status, owned_count, ring_start and the ring pool (64 entries per cell at first).\n"
                 "Cells over 48 planes or 64 vertices: the thread kernel at 128 and 256 (shell_voronoi.hpp:\n"
                 "construct_cells); exact cells over those on the host (construct_huge)", REBUILD)
    ph = card_height("until_pool_fits", "a\nb\nc\nd", 9.5)
    card(ax, sx, y + h - ph, sw, ph, "Pool short?", "shell_voronoi.hpp: until_pool_fits",
         "a full chunk marks its cells\nface_overflow; build again\nwith the fullest chunk's need\n+ 1/8, at most 3 attempts",
         TEST, "host", size=9.5)
    arrow(ax, [(x0 + w, y + h - 0.45), (sx, y + h - 0.45)], label="full", at=(x0 + w + 0.32, y + h - 0.25), size=9.5)
    arrow(ax, [(sx, y + h - ph + 0.3), (x0 + w, y + h - ph + 0.3)], label="again", at=(x0 + w + 0.32, y + h - ph + 0.5),
          size=9.5)
    c.wait(y, "host reads the off-base level count, the overflow count, each chunk's use")

    y, h = c.box("4  Exact start", "tet_start.hpp: build  (tet_mesh.hpp: assemble; tet_start::check)",
                 "assemble: scans count faces, ring slots, tetrahedra; a thread per face emits the tetrahedra\n"
                 "(i, j, q, q') whose two smallest particles are i < j, and their adjacency, without hashing\n"
                 "check: a thread per tetrahedron, exact orient and in_sphere, every particle used, volume\n"
                 "bad: mark their cells and every cell sharing a tetrahedron, clip those exactly (all of\n"
                 "them if more than n/8 or on the second try); 3 rounds at most", REBUILD)
    c.wait(y, "host reads each scan total and the check's counts, every round")

    dy = c.top - 0.5
    dx = x0 + 1.5
    diamond(ax, dx, dy, 2.2, 0.9, "exact?", size=11)
    arrow(ax, [(dx, c.boxes[-1]), (dx, dy + 0.45)])
    rh = card_height("not_exact", "a\nb\nc\nd\ne\nf", 9.5)
    card(ax, sx, dy + 0.4 - rh, sw, rh, "Refuse", "voronoi_mesh.hpp: detail::not_exact",
         "if the pool was short, clip\nonce more with need + 1/16;\nstill not exact: throw\nstd::runtime_error saying why:\n"
         "a coincident pair (sorted on\nthe host), uncertified, too large", REFUSE, "host", size=9.5)
    arrow(ax, [(dx + 1.1, dy), (sx, dy)], label="no", at=(x0 + 5.0, dy))
    c.top = dy - 1.3
    arrow(ax, [(dx, dy - 0.45), (dx, c.top)], label="yes", at=(dx + 0.3, dy - 0.75))
    c.box("5  Persistent arrays", "voronoi_mesh.hpp: Mesh::rebuild; tet_repair.hpp: prepare, reserve",
          "drops the start's face arrays; tetrahedra get 1/8 spare slots, faces 1/32, two ring pools\n"
          "of 1.5 × the start's ring entries (tet_mesh.hpp: make_face_rings), repair lists n/4 + 1024;\n"
          "tet_repair::orient makes every tetrahedron positive; writes tet_repair_point, alive", REBUILD,
          link=False)
    y, h = c.box("6  Geometry of every tetrahedron", "voronoi_mesh.hpp: Mesh::geometry",
                 "all circumcentres, every face numbered and walked (no old numbers), faces, volume\n"
                 "check: steps 4 to 8 of flow_step.png", REBUILD)

    b = y - 0.35
    ax.text(x0, b, "Time and frequency", fontsize=11, fontweight="bold", color=INK, va="top")
    ax.text(x0, b - 0.33,
            "A start at 128³ jittered takes about 200 ms in the bench's start line: clipping 58 ms (29 %), assemble, "
            "check and patch the rest.\nThe exact start runs at 17.7 M cells/s at 128³ and 11.7 at 256³ (jittered). "
            "Nsight Compute: the plain warp kernel, without\nrings, takes 54 ms on jittered and 133 to 136 ms on "
            "clustered and halo input at 128³; the thread kernel for the cells it gives\nup adds 0.5 % (jittered) to "
            "10 % (halo) of a rebuild. Steps rarely rebuild: none of the measured steps at 0.03 to 0.3 spacings\ndid. "
            "At 1.0 spacings, 12 stuck entries exceed n / 200000 at 128³ and the rebuild takes 250 to 590 ms, 7 to 15 "
            "steps' worth.\nSources: docs/measure_2026-11-10.md (A, ranked list) and the README table.",
            fontsize=9.5, color=FAINT, va="top", linespacing=1.35)
    b -= 1.9
    legend_chips(ax, x0, b, [(REBUILD, "rebuild"), (TEST, "decision"), (REFUSE, "refusal"), (HOSTC, "host")],
                 step=1.6)
    ax.text(x0 + 6.6, b, "▌▌", color=WAIT, fontsize=10, va="center")
    ax.text(x0 + 6.9, b, "the host waits for the GPU", fontsize=10, va="center", color=INK)
    finish(fig, ax, b - 0.25)
    save(fig, "flow_rebuild.png")


if __name__ == "__main__":
    step_figure()
    rebuild_figure()
