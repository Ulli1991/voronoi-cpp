#!/usr/bin/env python3
"""paper/data/weak_scaling.csv from the sweep's records (paper/data/raw/weak-<job>.jsonl, weak_scaling_a100.sh): per
commit, input, n, path (warp rebuild, exact start, step at s spacings) the median, min and max M cells/s (steps: frames
2 to 10), peak B/cell, and the SM clock the job logged under load (paper/data/clocks/clock-<job>.csv)."""
import csv
import glob
import os
import statistics
from collections import defaultdict

from common import records
from paperfig import DATA


def clock(job):
    """Median SM clock (MHz) over the samples with the GPU busy (utilisation >= 50 %)."""
    path = os.path.join(DATA, "clocks", f"clock-{job}.csv")
    if not os.path.exists(path):
        return ""
    mhz = []
    for row in csv.reader(open(path)):
        try:
            if float(row[3].split()[0]) >= 50:
                mhz.append(float(row[1].split()[0]))
        except (IndexError, ValueError):
            pass
    return f"{statistics.median(mhz):.0f}" if mhz else ""


def path_of(r):
    v = r["variant"]
    if v.startswith("warp clipping rebuild (cell"):
        return "rebuild"
    if v.startswith("exact start"):
        return "start"
    if v.startswith("moving-mesh step"):
        return f"step {r['spacings']:g}"
    return None


def main():
    groups = defaultdict(list)
    for r in records():
        p = path_of(r)
        if p is None or (p.startswith("step") and int(r["repeat"]) < 2):
            continue
        groups[(r["version"], r["input"], int(r["n"]), p)].append(r)
    out = os.path.join(DATA, "weak_scaling.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["commit", "input", "n", "path", "Mcells_s_median", "Mcells_s_min", "Mcells_s_max", "runs",
                    "bytes_per_cell", "jobs", "hosts", "sm_clock_MHz"])
        for (commit, inp, n, p), rs in sorted(groups.items()):
            rates = [float(r["Mcells_s"]) for r in rs]
            bpc = [float(r["bytes_per_cell"]) for r in rs if float(r["peak_bytes"]) > 0]
            jobs = sorted({r["job"] for r in rs})
            w.writerow([commit, inp, n, p, f"{statistics.median(rates):.4g}", f"{min(rates):.4g}",
                        f"{max(rates):.4g}", len(rates), f"{max(bpc):.0f}" if bpc else "", " ".join(jobs),
                        " ".join(sorted({r["host"] for r in rs})), " ".join(clock(j) for j in jobs)])
    print(f"{len(groups)} rows -> {out}")


if __name__ == "__main__":
    main()
