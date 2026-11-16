#!/usr/bin/env python3
"""Phases of the exact start and of the steps in scripts/modes_rounding_a100.sh's speed.log, median over every run:
gpu / main and same_bits / main of build_ms, geometry_ms, cells_ms (start) and gpu_ms, centres_faces_ms, cells_ms
(steps 2 on). usage: modes_rounding_phases.py <speed.log>"""
import re
import statistics as st
import sys
from collections import defaultdict


def main(path):
    v = defaultdict(list)
    for line in open(path):
        kv = dict(re.findall(r"(\w+)=([\w.-]+)", line))
        if "build" not in kv:
            continue
        if " mesh: " in line:
            for f in ("build_ms", "geometry_ms", "cells_ms"):
                v[(kv["n"], kv["kind"], "start", f, kv["build"])].append(float(kv[f]))
        elif "gpuflips:" in line and int(kv["step"]) >= 2:
            for f in ("gpu_ms", "centres_faces_ms", "cells_ms"):
                v[(kv["n"], kv["kind"], "d=" + kv["d"], f, kv["build"])].append(float(kv[f]))
    for key in sorted({k[:4] for k in v}, key=lambda k: (int(k[0]), k[1], k[2], k[3])):
        m = st.median(v[key + ("main",)])
        g, s = st.median(v[key + ("gpu",)]), st.median(v[key + ("same_bits",)])
        print(f"{' '.join(key):48s} main {m:9.3f}  gpu {g / m:.3f}  same_bits {s / m:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
