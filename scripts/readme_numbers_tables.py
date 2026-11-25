#!/usr/bin/env python3
"""The README's numbers from one run of scripts/readme_numbers_a100.sh (issue #6) and its CPU companion.

    python3 scripts/readme_numbers_tables.py <lines file> <library jsonl> [<cpu jsonl>]

Rebuild: n / construct_ms (the warp clipping, as the lean-cut tables). Exact start: n / start_ms. Ten steps: the mean of
all_ms over steps 2-10 of each run, the two rounds averaged (as scripts/step2x_tables.py). 500 steps: the rate over steps
2-500, re-sorts paid. Memory: the largest peak_bytes_per_cell of a run. Library rows: the median Mcells_s of the timed
repeats per (path, input, n), and bytes_per_cell. All rates in M cells/s.
"""
import json
import re
import statistics
import sys
from collections import OrderedDict, defaultdict


def fields(text):
    return dict(re.findall(r"(\w+)=(\S+)", text))


def main():
    lines_path, lib_path = sys.argv[1], sys.argv[2]
    cpu_path = sys.argv[3] if len(sys.argv) > 3 else None
    rebuild, start, memory = OrderedDict(), OrderedDict(), OrderedDict()
    steps = defaultdict(list)  # (n, kind, d, steps, flags) -> list of runs, each a list of all_ms by step
    current = {}
    twod = []
    for line in open(lines_path):
        if line.startswith("2d "):
            if "mesh2d" in line or "memory peak" in line:
                twod.append(line.strip())
            continue
        if " | " not in line:
            continue
        args, out = line.split(" | ", 1)
        a = args.split()
        f = fields(out)
        if a[0] == "rebuild" and out.startswith("warp:"):
            rebuild.setdefault((a[1], a[2]), []).append(int(a[1]) / float(f["construct_ms"]) / 1e3)
        if out.startswith("start:") and "start_ms" in f:
            start[(a[1], a[2])] = int(a[1]) / float(f["start_ms"]) / 1e3
        if out.startswith("memory:"):
            key = args.strip()
            memory[key] = max(memory.get(key, 0.0), float(f["peak_bytes_per_cell"]))
        if a[0] == "step" and out.startswith("gpuflips:"):
            key = tuple(a[1:])
            k = int(f["step"])
            if k == 1 or key not in current:
                steps[key].append([])
                current[key] = True
            steps[key][-1].append(float(f["all_ms"]))
    print("## Full rebuild (warp clipping), M cells/s (median of runs)")
    for (n, kind), rates in rebuild.items():
        print(f"  {kind:10s} {n:>9s}: {statistics.median(rates):7.1f}")
    print("## Exact start, M cells/s")
    for (n, kind), rate in start.items():
        print(f"  {kind:10s} {n:>9s}: {rate:7.1f}")
    print("## Steps (update + update_cells), M cells/s")
    for key, runs in steps.items():
        n = int(key[0])
        if key[3] == "10":
            means = [statistics.mean(r[1:10]) for r in runs if len(r) >= 10]
            print(f"  ten steps  {' '.join(key):45s}: {n / statistics.mean(means) / 1e3:7.1f} ({len(means)} rounds)")
        elif key[3] == "500":
            for r in runs:
                print(f"  500 steps  {' '.join(key):45s}: {n * (len(r) - 1) / sum(r[1:]) / 1e3:7.1f} ({len(r)} steps)")
    print("## Peak memory, B/cell")
    for key, peak in memory.items():
        print(f"  {key:60s}: {peak:7.0f}")
    print("## 2D")
    for line in twod:
        print("  " + line)
    for title, path in (("A100", lib_path), ("CPU, 72 threads", cpu_path)):
        if not path:
            continue
        rows = defaultdict(list)
        mem = {}
        for line in open(path):
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            if r.get("phase") not in ("build", "step") or r.get("repeat", 1) == 0:
                continue
            key = (r["variant"], r["input"], r["n"], r.get("threads", ""))
            rows[key].append(r["Mcells_s"])
            mem[key] = r.get("bytes_per_cell", 0)
        print(f"## Library benchmark rows, {title}: median M cells/s of the timed repeats; B/cell")
        for key, rates in rows.items():
            print(f"  {key[0][:60]:60s} {key[1]:10s} {key[2]:>9}: {statistics.median(rates):8.2f}  {mem[key]:7.0f}")


if __name__ == "__main__":
    main()
