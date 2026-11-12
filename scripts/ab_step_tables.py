"""Tables of scripts/ab_step_a100.sh logs: step rates (M cells/s, median of steps 2 to 10) and checksums per build.

python3 scripts/ab_step_tables.py <log> [<log> ...]
"""
import re
import statistics
import sys
from collections import defaultdict

steps = defaultdict(list)  # (tag, n, kind, d) -> [(step_ms, cells_ms)]
sums = defaultdict(dict)  # (n, kind, d, step) -> {tag: checksums}
tags = []
for path in sys.argv[1:]:
    for line in open(path):
        m = re.match(r"(\S+) step (\d+) (\S+) (\S+) (\d+)( --checksum)? \| (.*)", line)
        if not m:
            continue
        tag, n, kind, d, _, checked, rest = m.groups()
        if tag not in tags:
            tags.append(tag)
        s = re.search(r"step=(\d+)", rest)
        if rest.startswith("gpuflips") and not checked:
            f = dict(kv.split("=") for kv in rest.split()[1:] if "=" in kv)
            if int(f["step"]) >= 2 and f["rebuilt"] == "0":
                steps[(tag, int(n), kind, d)].append((float(f["step_ms"]), float(f["cells_ms"])))
        elif rest.startswith("geometry_checksum"):
            sums[(int(n), kind, d, int(s.group(1)))][tag] = rest.split(" ", 2)[2]
print("| cells | spacings | " + " | ".join(f"{t}: step, step + cells" for t in tags) + " | step B/A, with cells |")
print("|---|---|" + "---|" * (len(tags) + 1))
for n, kind, d in sorted({k[1:] for k in steps}):
    row, rates = [], []
    for t in tags:
        v = steps.get((t, n, kind, d), [])
        if not v:
            row.append("")
            continue
        a = n / statistics.median(x[0] for x in v) / 1e3
        b = n / statistics.median(x[0] + x[1] for x in v) / 1e3
        rates.append((a, b))
        row.append(f"{a:.1f}, {b:.1f} ({len(v)} steps)")
    ratio = f"{rates[-1][0] / rates[0][0]:.3f}, {rates[-1][1] / rates[0][1]:.3f}" if len(rates) > 1 else ""
    print(f"| {n} {kind} | {d} | " + " | ".join(row) + f" | {ratio} |")
# Per checksum (geometry, cells, faces): steps where every build agrees, and where they do not.
for field in range(3):
    same = differ = 0
    for key, by in sorted(sums.items()):
        vals = {v.split()[field].split("=")[-1] for v in by.values()}
        if len(by) > 1:
            same += len(vals) == 1
            differ += len(vals) > 1
    print(f"{['geometry', 'cells', 'faces'][field]}_checksum: {same} steps identical, {differ} differ")
