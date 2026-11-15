#!/usr/bin/env python3
"""paper/data/memory_by_array.csv from stages_a100.sh memory output (paper/data/raw/memory-<job>.txt): B/cell of every
Kokkos array at the peak of the rebuild ("warp"), the exact start ("start_and_prepare") and the steps, per input and n."""
import csv
import glob
import os
import re

from paperfig import DATA

PHASES = {"warp": "rebuild", "start_and_prepare": "exact start", "steps": "moving step"}


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA, "raw", "memory-*.txt"))):
        job = re.findall(r"(\d+)", os.path.basename(path))[-1]
        kind = n = phase = None
        for line in open(path):
            if line.startswith("== memory"):
                _, _, _, kind, n = line.split()
                phase = None
            elif line.startswith("memory: phase="):
                f = dict(re.findall(r"(\w+)=([-\w.+]+)", line))
                phase = PHASES.get(f["phase"])
                if phase:
                    rows.append([job, kind, n, phase, "(peak, all arrays)", f["peak_bytes_per_cell"]])
            elif phase and line.startswith("  ") and line.rstrip().endswith("B/cell"):
                parts = line.split()
                rows.append([job, kind, n, phase, parts[0], parts[-2]])
    out = os.path.join(DATA, "memory_by_array.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job", "input", "n", "phase", "array", "bytes_per_cell"])
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
