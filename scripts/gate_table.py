"""The 3D gate table from a scripts/dim_gate_a100.sh log: MAIN against FIX.

    python gate_table.py <log>

Rebuild: pipeline M cells/s. Steps (2.1M jittered, 10 steps): M cells/s of
the median step of steps 2-9 (as research_notes/hydro_stage3), and of
update_cells; start ms; peak B/cell. Hydro: M cell updates/s of the median
step. Timed runs ran MAIN FIX MAIN FIX; the table gives the mean of the two.
Also cells_checksum equality and the hashes of the hydro states.
"""
import re
import statistics
import sys
from collections import defaultdict

text = open(sys.argv[1]).read()
blocks = re.split(r"^== ", text, flags=re.M)[1:]
values = defaultdict(list)  # (metric, variant) -> list
checksums = defaultdict(set)
verify = defaultdict(int)
for b in blocks:
    head = b.splitlines()[0]
    variant = head.split()[0]
    key = " ".join(head.split()[1:])
    m = re.search(r"pipeline_Mcells_s=([\d.]+)", b)
    if head.split()[1] == "rebuild" and m:
        values[(f"rebuild {head.split()[2]} (M cells/s)", variant)].append(float(m.group(1)))
    for c in re.findall(r"cells_checksum=(\d+)", b):
        checksums[key].add((variant, c))
    verify[variant] += len(re.findall(r"verify:.*PASS", b))
    if head.split()[1] == "step":
        d = head.split()[-1]
        steps = [float(x) for x in re.findall(r"step_Mcells_s=([\d.]+)", b)]
        cells = [float(x) for x in re.findall(r"gpuflips: step=\d+ .*? cells_ms=([\d.]+)", b)]
        if steps:
            values[(f"step {d} (M cells/s)", variant)].append(statistics.median(steps[1:9]))
            values[(f"update_cells at {d} (M cells/s)", variant)].append(2097152 / statistics.median(cells[1:9]) / 1e3)
        m = re.search(r"start: .*?start_ms=([\d.]+)", b)
        if m:
            values[("start (ms)", variant)].append(float(m.group(1)))
        m = re.findall(r"phase=steps .*?peak_bytes_per_cell=([\d.]+)", b)
        if m:
            values[(f"peak memory at {d} (B/cell)", variant)].append(float(m[-1]))
    if head.split()[1] == "hydro" and "--steps=40" in head:
        m = re.search(r"cell_updates_per_s=([\d.e+]+)", b)
        values[("moving hydro step 128^3 (M updates/s)", variant)].append(float(m.group(1)) / 1e6)
        m = re.search(r"peak_bytes_per_cell=([\d.]+)", b)
        values[("moving hydro peak (B/cell)", variant)].append(float(m.group(1)))

print("| metric | main 57201bc | branch | branch / main |")
print("|---|---|---|---|")
seen = []
for (metric, variant) in values:
    if metric not in seen:
        seen.append(metric)
for metric in seen:
    a = statistics.mean(values[(metric, "MAIN")])
    b = statistics.mean(values[(metric, "FIX")])
    print(f"| {metric} | {a:.4g} | {b:.4g} | {b / a:.4f} |")
same = all(len({c for _, c in v}) == 1 for v in checksums.values())
print(f"\ncells_checksum identical in all {len(checksums)} runs: {same}; verify PASS lines: MAIN {verify['MAIN']}, FIX {verify['FIX']}")
for line in re.findall(r"^hash: .*$", text, flags=re.M):
    print(line)
