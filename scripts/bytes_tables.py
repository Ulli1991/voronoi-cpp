#!/usr/bin/env python3
"""Per-kernel DRAM bytes per cell of the moving step, from the jobs of scripts/bytes_a100.sh: one column per build.
Usage: bytes_tables.py <jobs dir> <job id> <tag> [<tag> ...]"""
import csv
import sys
from collections import defaultdict

N = 2097152  # 128³ cells


def kernels(path):
    """Per (phase, label): launches, ms, B/cell read, B/cell written."""
    lines = [ln for ln in open(path) if ln.startswith('"')]
    rows = list(csv.reader(lines))
    head, out = rows[0], defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    phase_col = [h for h in head if h.startswith("thread Domain")][0]
    for r in rows[2:]:
        d = dict(zip(head, r))
        label = d["Kernel Name"].split("/")[0]
        phase = "cells" if "cells" in d[phase_col] else "update"
        num = lambda k: float(d.get(k, "0").replace(",", "") or 0)
        e = out[(phase, label)]
        e[0] += 1
        e[1] += num("gpu__time_duration.sum") / 1e6
        e[2] += num("dram__bytes_read.sum") / N
        e[3] += num("dram__bytes_write.sum") / N
    return out


def main():
    jobs, job, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
    for d in ("0.03", "0.1"):
        data = {t: kernels(f"{jobs}/ncu_{t}_{d}-{job}.csv") for t in tags}
        for phase in ("update", "cells"):
            keys = sorted({k for t in tags for k in data[t] if k[0] == phase},
                          key=lambda k: -max(sum(data[t][k][2:]) for t in tags if k in data[t]))
            print(f"\n{d} spacings, {phase}: B/cell read + written (ms)")
            print("| kernel | " + " | ".join(tags) + " |")
            print("|---|" + "---|" * len(tags))
            total = {t: [0.0, 0.0, 0.0] for t in tags}
            for k in keys:
                cells = []
                for t in tags:
                    e = data[t].get(k)
                    if not e:
                        cells.append("–")
                        continue
                    total[t][0] += e[1]
                    total[t][1] += e[2]
                    total[t][2] += e[3]
                    cells.append(f"{e[2]:.0f} + {e[3]:.0f} = {e[2] + e[3]:.0f} ({e[1]:.2f})")
                print(f"| `{k[1]}` | " + " | ".join(cells) + " |")
            print("| **total** | " + " | ".join(
                f"**{total[t][1] + total[t][2]:.0f}** ({total[t][0]:.2f})" for t in tags) + " |")


if __name__ == "__main__":
    main()
