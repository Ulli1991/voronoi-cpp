"""Throughput table from a scripts/mesh_speed_a100.sh speed log.

    python3 scripts/mesh_speed_table.py <log> [baseline build name]

Steps: M cells/s of the median step (steps 2-9, mean over rounds), update
alone (repair + geometry), update_cells alone, both; peak B/cell. Moving
hydro: M cell updates/s of the whole step and of its mesh, cells and hydro
phases, as hydro_sedov prints them. Ratios to the baseline (default main).
"""
import re
import statistics
import sys
from collections import defaultdict

text = open(sys.argv[1]).read()
base = sys.argv[2] if len(sys.argv) > 2 else "main"
blocks = re.split(r"^== ", text, flags=re.M)[1:]
steps = defaultdict(lambda: defaultdict(list))  # (n, d) -> build -> [(update, cells, peak)]
hydro = defaultdict(lambda: defaultdict(list))
builds = []
for b in blocks:
    head = b.splitlines()[0].split()
    build = head[0]
    if build not in builds:
        builds.append(build)
    if head[1] == "step":
        n, d = int(head[2]), head[4]
        rows = re.findall(r"gpuflips: step=(\d+) gpu_ms=([\d.]+).*? rounds=(\d+) flips=(\d+).*? centres_faces_ms=([\d.]+) cells_ms=([\d.]+)", b)
        rows = [r for r in rows if 2 <= int(r[0]) <= 9]
        if not rows:
            continue
        update = statistics.median(float(r[1]) + float(r[4]) for r in rows)
        cells = statistics.median(float(r[5]) for r in rows)
        total = statistics.median(float(r[1]) + float(r[4]) + float(r[5]) for r in rows)
        peak = re.findall(r"phase=steps .*?peak_bytes_per_cell=([\d.]+)", b)
        steps[(n, d)][build].append((n / update / 1e3, n / cells / 1e3, n / total / 1e3, float(peak[-1]) if peak else 0))
    if head[1] == "hydro":
        m = re.search(r"cell_updates_per_s=([\d.e+]+)", b)
        if m:
            row = {"whole": float(m.group(1)) / 1e6}
            p = re.search(r"peak_bytes_per_cell=([\d.]+)", b)
            row["peak"] = float(p.group(1)) if p else 0
            g = re.search(r"gpu_used_MB=([\d.]+)", b)
            row["gpu_MB"] = float(g.group(1)) if g else 0
            for phase in ("mesh", "cells", "hydro"):
                q = re.search(rf"\b{phase}_(?:Mcells_s|M_cells_s)=([\d.]+)", b)
                if q:
                    row[phase] = float(q.group(1))
            hydro[head[2]][build].append(row)

print("| n, spacings | build | update | update_cells | total | peak B/cell | total / " + base + " |")
print("|---|---|---|---|---|---|---|")
for (n, d), by in sorted(steps.items()):
    ref = statistics.mean(r[2] for r in by[base]) if base in by else float("nan")
    for build in builds:
        if build not in by:
            continue
        v = [statistics.mean(r[k] for r in by[build]) for k in range(4)]
        print(f"| {n} {d} | {build} | {v[0]:.1f} | {v[1]:.1f} | {v[2]:.1f} | {v[3]:.0f} | {v[2] / ref:.3f} |")
for side, by in hydro.items():
    print(f"\nmoving hydro {side}:")
    for build in builds:
        if build in by:
            print(build, {k: round(statistics.mean(r.get(k, 0) for r in by[build]), 1) for k in by[build][0]})
