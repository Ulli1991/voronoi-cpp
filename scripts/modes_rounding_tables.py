#!/usr/bin/env python3
"""Time ratios of scripts/modes_rounding_a100.sh's speed.log: gpu / main and same_bits / main, per input and phase.
Rebuild: construct_ms (median of 5 per run); start: total_ms of the exact start; steps: step_ms + cells_ms, the median
over steps 2 on. Each run's value, then the median over the rounds; per-round ratios in brackets.
usage: modes_rounding_tables.py <speed.log>"""
import re
import statistics as st
import sys
from collections import defaultdict


def load(path):
    values = defaultdict(lambda: defaultdict(list))  # (kind, n, phase) -> build -> per-round values
    steps = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for line in open(path):
        kv = dict(re.findall(r"(\w+)=([\w.-]+)", line))
        if "build" not in kv:
            continue
        key = (kv["kind"], kv["n"])
        if " warp: " in line:
            values[key + ("rebuild",)][kv["build"]].append(float(kv["construct_ms"]))
        elif " mesh: " in line:
            values[key + ("start " + kv["d"],)][kv["build"]].append(float(kv["total_ms"]))
        elif "gpuflips:" in line and int(kv["step"]) >= 2:
            steps[key + ("steps " + kv["d"],)][kv["build"]][kv["round"]].append(
                float(kv["step_ms"]) + float(kv["cells_ms"]))
    for key, builds in steps.items():
        for build, rounds in builds.items():
            for r in sorted(rounds):
                values[key][build].append(st.median(rounds[r]))
    return values


def main(path):
    values = load(path)
    ratios = defaultdict(list)
    print(f"{'case':36s} {'main ms':>9s} {'gpu/main':>9s} {'same_bits/main':>14s}  per-round gpu/main")
    for key in sorted(values, key=lambda k: (int(k[1]), k[0], k[2])):
        m, g, s = (values[key][b] for b in ("main", "gpu", "same_bits"))
        rg, rs = st.median(g) / st.median(m), st.median(s) / st.median(m)
        phase = key[2].split()[0]
        ratios["gpu " + phase].append(rg)
        ratios["same_bits " + phase].append(rs)
        rounds = " ".join(f"{a / b:.3f}" for a, b in zip(g, m))
        print(f"{' '.join(key):36s} {st.median(m):9.3f} {rg:9.4f} {rs:14.4f}  {rounds}")
    for name, r in sorted(ratios.items()):
        print(f"{name:16s} min {min(r):.4f} max {max(r):.4f} cases {len(r)}")


if __name__ == "__main__":
    main(sys.argv[1])
