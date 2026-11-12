#!/usr/bin/env python3
"""Draws docs/figures/data_arrays.png: every persistent array of Mesh<3>, where it lives, its bytes per cell at the
peak of a moving step (docs/measure_2026-11-10.md, section C), who writes and reads it, and how long it is valid.
Writers, readers and lifetimes are read from the code of main; the index of each group is in mesh_order.py."""
import mesh_order
from diagram import FAINT, HOSTC, INK, MONO, canvas, finish, rect, save

KEPT = ("#dff0dc", "#2ca02c")
STEP = ("#dbe9f6", "#1f77b4")
CELLS = ("#eef1f5", "#7c8796")
SCRATCH = ("#fde8d3", "#ff7f0e")
IDLE = ("#f2f2f2", "#b0b0b0")
LIFE = {"kept": KEPT, "step": STEP, "cells": CELLS, "scratch": SCRATCH, "idle": IDLE}
LIFE_TEXT = [("kept", "kept between steps, edited where flips change it"),
             ("step", "rewritten by every update()"),
             ("cells", "rewritten by update_cells(), valid until the next update()"),
             ("scratch", "scratch: valid until the next update(), update_cells() or scratch()"),
             ("idle", "used by a rebuild and clip() only, kept in between")]
HYDRO = "luna-cpp"

# (labels, B/cell at the step peak, written by, read by, lifetimes). B/cell: section C of the note, job 32145254.
GROUPS = [
    (mesh_order.TET, [
        ("tet_vertex, tet_adjacent\nint[4] rows each", 250, "assemble (start), repair: apply,\ntet_flips::finish",
         "every topological kernel", ["kept"]),
        ("tet_centre\ndouble[4]: centre, r²", 250, "screen (every live one),\ncircumcentres (created ones)",
         "screen's own test, faces", ["step"]),
        ("tet_repair_dirty, _listed,\n_free, _lock (64-bit votes)", 157, "screen, repair: find, apply,\nrelease",
         "repair: find, plan, apply", ["step"]),
        ("tet_edges, alive, touched,\ntouched_list, work, plans, ...", 45, "assemble, repair: apply",
         "number_faces, dirty_faces,\nupdate_cells (touched_list)", ["kept"]),
        ("face_rings_start\nfirst face of each tetrahedron", 31, "number_faces",
         "edge_face (dirty_faces,\nappend_rings)", ["step"]),
    ]),
    (mesh_order.FACE, [
        ("face_rings_owner, _neighbour", 66, "number_faces", f"faces, update_cells, {HYDRO}", ["step"]),
        ("face_rings_ring_start,\n_ring_length", 66, "number_faces, append_rings,\ncompact_rings", "faces", ["step"]),
        ("face_area, face_centroid\ndouble[3] rows", 394, "faces", f"cell_sums, {HYDRO}", ["step"]),
        ("face_rings_pool, two:\nin use and spare", 501, "append_rings, compact_rings;\nspare: the temporaries",
         f"faces (ring entries); spare:\n{HYDRO} via Mesh::scratch", ["kept", "scratch"]),
    ]),
    (mesh_order.PARTICLE, [
        ("positions\ndouble[4], snapped", 32, "snap", f"every geometric kernel,\n{HYDRO}", ["step"]),
        ("tet_repair_point\nint[4], grid steps", 16, "snap (in move_screened)", "exact predicates of the repair",
         ["step"]),
        ("cell_volume, cell_centroid", 32, "faces (volume, atomics),\ncell_sums (both, fixed order)",
         f"{HYDRO}, total_volume", ["step", "cells"]),
        ("cell list: slot_positions,\nparticle_order, cell_offsets, ...", 37, "neighbor_search::build",
         "clipping: rebuild, clip()", ["idle"]),
    ]),
    ("list entry (2 per face)", [
        ("cell_face, cell_face_start,\ncell_face_cursor", 74, "update_cells: fill, sort",
         f"cell_sums, {HYDRO}", ["cells"]),
        ("face_rows (FaceReal = float)", None, "faces",
         f"{HYDRO}", ["step", "cells"]),
    ]),
]
TOTAL = 1950


def main():
    W, H = 14.2, 30
    fig, ax = canvas(W, H)
    cx = {"name": 0.2, "bar": 3.55, "write": 6.25, "read": 9.15, "life": 12.1}
    scale = 2.0 / 501
    ax.text(0.2, H - 0.2, "Where the data lives: the arrays of Mesh<3> between steps", fontsize=14,
            fontweight="bold", va="top", color=INK)
    ax.text(0.2, H - 0.55, "Every array is device memory, allocated through Kokkos, and none is mirrored to the host. "
            f"Bytes per cell at the peak of a step: {TOTAL} B/cell in all.", fontsize=10.5, va="top", color=INK)
    y = H - 1.15
    for key, label in (("name", "array (label)"), ("bar", "B/cell at a step's peak"), ("write", "written by"),
                       ("read", "read by"), ("life", "valid")):
        ax.text(cx[key], y, label, fontsize=10.5, fontweight="bold", va="center", color=INK)
    y -= 0.25
    ax.plot([0.2, W - 0.2], [y, y], color=INK, lw=1.0)
    for index, rows in GROUPS:
        y -= 0.38
        ax.text(0.2, y, f"indexed by {index}", fontsize=10.5, style="italic", color=FAINT, va="center")
        for name, bytes_, writes, reads, lives in rows:
            lines = max(name.count("\n"), writes.count("\n"), reads.count("\n")) + 1
            h = 0.2 + 0.2 * lines
            y -= h / 2 + 0.12
            ax.text(cx["name"], y, name, fontsize=9.5, family=MONO, va="center", color=INK, linespacing=1.25)
            if bytes_ is None:
                ax.text(cx["bar"], y, "not in the profile:\n32 B per face", fontsize=9,
                        va="center", color=FAINT, linespacing=1.2)
            else:
                rect(ax, cx["bar"], y - 0.11, bytes_ * scale, 0.22, LIFE[lives[0]], lw=0.9, radius=0.02)
                ax.text(cx["bar"] + bytes_ * scale + 0.08, y, f"{bytes_}", fontsize=10, va="center", color=INK)
            ax.text(cx["write"], y, writes, fontsize=9.5, va="center", color=INK, linespacing=1.25)
            ax.text(cx["read"], y, reads, fontsize=9.5, va="center", color=INK, linespacing=1.25)
            for k, life in enumerate(lives):
                rect(ax, cx["life"] + 0.75 * k, y - 0.11, 0.62, 0.22, LIFE[life], lw=0.9, radius=0.04)
                ax.text(cx["life"] + 0.75 * k + 0.31, y, life, fontsize=8.5, ha="center", va="center", color=INK)
            y -= h / 2
            ax.plot([0.2, W - 0.2], [y, y], color="#dddddd", lw=0.6)

    # The host side.
    y -= 0.5
    rect(ax, 0.2, y - 1.3, W - 0.4, 1.3, HOSTC, lw=1.2)
    ax.text(0.35, y - 0.12, "On the host", fontsize=11.5, fontweight="bold", va="top", color=INK)
    ax.text(0.35, y - 0.45,
            "Counts that size the next launch: n, faces, tetrahedra in use, free slots, ring pool use "
            "(Mesh, State, FaceRings members). They are read\nback from the device at each wait in flow_step.png.\n"
            "tet_repair_host_counters: 9 ints in SharedHostPinnedSpace, the one host copy of device data; "
            "each read is one asynchronous copy, then a fence.\n"
            "Temporary copies only: the region of a host finish (tet_flips::finish), cells that overflow on the "
            "device (construct_huge), a refusal's message.", fontsize=9.5, va="top", color=INK, linespacing=1.35)
    y -= 1.6
    ax.text(0.2, y, "Lifetime:", fontsize=10, va="center", color=INK, fontweight="bold")
    for k, (life, text) in enumerate(LIFE_TEXT):
        yy = y - 0.3 * k
        rect(ax, 1.25, yy - 0.1, 0.5, 0.2, LIFE[life], lw=0.9, radius=0.04)
        ax.text(1.85, yy, text, fontsize=9.5, va="center", color=INK)
    y -= 0.3 * len(LIFE_TEXT) + 0.15
    ax.text(0.2, y, "B/cell: Kokkos allocations at the peak of four steps, 128³ jittered at 0.1 spacings per step "
            "(job 32145254; 256³ agrees to 0.1 B), docs/measure_2026-11-10.md,\nsection C. Writers and readers: "
            f"the code of main; {HYDRO} reads through voronoi::Mesh. Bars are coloured by their first lifetime.",
            fontsize=9, va="top", color=FAINT, linespacing=1.3)
    finish(fig, ax, y - 0.5)
    save(fig, "data_arrays.png")


if __name__ == "__main__":
    main()
