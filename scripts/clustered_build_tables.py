#!/usr/bin/env python3
"""Tables of scripts/clustered_build_a100.sh logs: rebuild and start rates (M cells/s, median over rounds), and whether
every checksum of `new` equals `ref` (all but the step's cells_checksum). Usage: clustered_build_tables.py <log>..."""
import re
import statistics
import sys
from collections import defaultdict

rate, start, sums = defaultdict(list), defaultdict(list), defaultdict(dict)
for path in sys.argv[1:]:
    mode = design = kind = None
    for line in open(path):
        m = re.match(r"== (\w+) (\w+):\S+ (\S+) (\d+)", line)
        if m:
            mode, design, kind, n = m.groups()
            kind = kind.split("/")[-1].split("_")[0].replace(".pts", "")
            continue
        if mode == "rate" and line.startswith("warp:"):
            rate[(kind, n, design)].append(int(n) / float(re.search(r"construct_ms=(\S+)", line).group(1)) / 1e3)
        if mode == "start" and line.startswith("start:"):
            start[(kind, n, design)].append(int(n) / float(re.search(r"start_ms=(\S+)", line).group(1)) / 1e3)
        if mode in ("checksum", "work", "start"):
            for key, value in re.findall(r"(\w*checksum\w*|rings_topology|clip_topology)[=:] ?(?:step=\S+ )?(\S+)", line):
                sums[(kind, n, design)][f"{mode}:{line.split(':')[0]}:{key}"] = value
for name, table in (("rebuild", rate), ("exact start", start)):
    if not table:
        continue
    print(f"\n{name}, M cells/s: kind n ref new change")
    for kind, n in sorted({k[:2] for k in table}, key=lambda k: (int(k[1]), k[0])):
        r, w = table.get((kind, n, "ref")), table.get((kind, n, "new"))
        if r and w:
            a, b = statistics.median(r), statistics.median(w)
            print(f"{kind:10s} {n:>9s} {a:7.2f} {b:7.2f} {100 * (b / a - 1):+6.1f} %")
for kind, n in sorted({k[:2] for k in sums}, key=lambda k: (int(k[1]), k[0])):
    r, w = sums.get((kind, n, "ref"), {}), sums.get((kind, n, "new"), {})
    keys = sorted(set(r) | set(w))
    # The step's cells_checksum hashes the face numbering, which differs between runs of main too.
    keys = [k for k in keys if k != "checksum:geometry_checksum:cells_checksum"]
    differ = [k for k in keys if r.get(k) != w.get(k)]
    print(f"checksums {kind:10s} {n:>9s}: {len(keys)} compared, " + ("all equal" if not differ else "DIFFER " + ", ".join(differ)))
