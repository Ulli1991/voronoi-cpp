#!/usr/bin/env python3
"""docs/data/fallback_cases.csv from fallback_cases_a100.sh logs: per case the mean ms per step of each stage over steps
2 to the last (means, not medians, so rare fallbacks count), the steps that used it, and the rate. Columns as
paper/data/stages.csv (job, input, n, what, stage, ms) plus steps, steps_used, stuck_entries, Mcells_s; and
docs/data/fallback_cases_steps.jsonl, one line per step with the gpuflips fields and the case.
Usage: fallback_cases_tables.py <log> ..."""
import csv
import json
import os
import re
import sys

STAGES = [("repair", "gpu_ms"), ("host fallback", "host_ms"), ("rebuild", "rebuild_ms"), ("re-sort", "resort_ms"),
          ("geometry", "centres_faces_ms"), ("cells", "cells_ms")]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "data", "fallback_cases.csv")


def fields(line):
    return dict(re.findall(r"(\w+)=([-\w.+]+)", line))


def flush(rows, job, case, steps, refused):
    label, kind, n, d = case[0], case[1], int(case[2]), case[3]
    flags = " ".join(case[4:])
    what = f"{label} {flags}".strip()
    later = [s for s in steps if int(s["step"]) >= 2]
    if not later:
        rows.append([job, kind, n, what, "refused" if refused else "none", "", 0, 0, 0, ""])
        return
    k = len(later)
    total = sum(float(s["all_ms"]) for s in later) / k
    hit = {"host fallback": lambda s: float(s["host_ms"]) > 0, "rebuild": lambda s: s["rebuilt"] == "1"}
    cells = {"host fallback": lambda s: int(s["stuck"]) if float(s["host_ms"]) > 0 else 0,
             "rebuild": lambda s: int(s["stuck"]) if s["rebuilt"] == "1" else 0}
    for name, key in STAGES:
        ms = sum(float(s[key]) for s in later) / k
        used = sum(hit[name](s) for s in later) if name in hit else k
        stuck = sum(cells[name](s) for s in later) if name in cells else ""
        rows.append([job, kind, n, what, name, f"{ms:.4g}", k, used, stuck, f"{n / total / 1e3:.4g}"])
    if refused:
        rows.append([job, kind, n, what, "refused", "", k, 1, "", ""])


def main(paths):
    rows, per_step = [], []
    for path in paths:
        job = re.findall(r"(\d+)", os.path.basename(path))[-1]
        case, steps, refused = None, [], False
        for line in open(path, errors="replace"):
            if line.startswith("== case"):
                if case:
                    flush(rows, job, case, steps, refused)
                case, steps, refused = line.split()[2:], [], False
            elif line.startswith("gpuflips:"):
                steps.append(fields(line))
                per_step.append({"job": job, "what": " ".join([case[0]] + case[4:]), "input": case[1],
                                 "n": int(case[2]), "spacings": float(case[3]),
                                 **{k: float(v) for k, v in steps[-1].items()}})
            elif line.startswith("refused:"):
                refused = True
        if case:
            flush(rows, job, case, steps, refused)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job", "input", "n", "what", "stage", "ms", "steps", "steps_used", "stuck_entries", "Mcells_s"])
        w.writerows(rows)
    with open(OUT.replace(".csv", "_steps.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in per_step)
    print(f"{len(rows)} rows -> {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
