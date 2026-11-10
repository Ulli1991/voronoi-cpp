#!/usr/bin/env python3
"""The tables of docs/measure_resort_2026-11-10.md from job logs of scripts/resort_a100.sh (table and decay modes).

Usage: resort_tables.py <log>...   One row per run: rates over the whole run and early and late, re-sort cost, rebuild."""
import re
import statistics
import sys

CURVES = ["morton", "hilbert", "moore", "gray", "row", "rcm"]


def fields(line):
    return dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)


def numbers(line):
    """The numeric fields of a line; a line cut off by the end of a job is dropped."""
    out = {}
    for k, v in fields(line).items():
        try:
            out[k] = float(v)
        except ValueError:
            return None
    return out


def runs(path):
    """Splits a log into runs: the header, layout, step rows, order rows and the final rebuild of each."""
    out, flags = [], ""
    for line in open(path):
        if line.startswith("== run"):
            flags = line[6:].strip()
        elif line.startswith("# backend="):
            out.append({"head": fields(line), "flags": flags, "steps": [], "order": []})
        elif not out:
            continue
        elif line.startswith(("layout:", "mesh:", "final_rebuild:")):
            out[-1][line.split(":")[0]] = fields(line)
        elif line.startswith(("gpuflips:", "order:")) and numbers(line) is not None:
            out[-1]["steps" if line[0] == "g" else "order"].append(numbers(line))
    return [r for r in out if r["steps"]]


def name(run):
    lay = run.get("layout")
    if not lay:
        return "today (no layout)"
    every = int(lay["every"])
    text = CURVES[int(lay["curve"])]
    if every == 0 and float(lay["far"]) == 0:
        text = "today, no re-sort"
    elif every >= 100000:
        text += ", at build"
    elif every > 0:
        text += ", every %d" % every
    if float(lay["far"]) > 0:
        text += ", far > %g" % float(lay["far"])
    if lay["owner_faces"] == "1":
        text += ", owner faces"
    if lay["owner_bit"] == "1":
        text += ", owner bit"
    return text


def row(run):
    n = int(run["head"]["n"])
    steps = run["steps"]
    whole = lambda s: s["gpu_ms"] + s["host_ms"] + s["rebuild_ms"] + s.get("resort_ms", 0) + s["centres_faces_ms"] + s["cells_ms"]
    total = sum(whole(s) for s in steps)
    early = [s for s in steps if 2 <= s["step"] <= 11]
    late = [s for s in steps if s["step"] > len(steps) - 10]
    rate = lambda rows: n / statistics.median(whole(s) for s in rows) / 1e3
    resorted = [s for s in steps if s.get("resorted", 0) == 1]
    sort_ms = statistics.mean(s["resort_ms"] for s in resorted) if resorted else 0
    # The re-sort's whole cost: its own time, and what its step takes beyond the median of the steps on either side.
    extra = []
    for s in resorted:
        k = int(s["step"])
        around = [whole(t) for t in steps if 1 <= abs(int(t["step"]) - k) <= 3]
        if around:
            extra.append(whole(s) - statistics.median(around))
    rebuild = float(run["final_rebuild"]["Mcells_s"]) if "final_rebuild" in run else float("nan")
    return (name(run), len(steps), n * len(steps) / total / 1e3, rate(early), rate(late), len(resorted), sort_ms,
            statistics.mean(extra) if extra else 0, rebuild)


def main(paths):
    print("| run | steps | M cells/s over the run | steps 2-11 | last 10 | re-sorts | re-sort ms | re-sort step, ms over neighbours | rebuild M cells/s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for path in paths:
        for run in runs(path):
            r = row(run)
            print("| %s | %d | %.1f | %.1f | %.1f | %d | %.1f | %.1f | %.1f |" % r)
    for path in paths:
        for run in runs(path):
            if not run["order"]:
                continue
            print("\n%s (%s)" % (name(run), path))
            print("| step | owner step p50 | owner step <= 64 | pair p50 | pair <= 64 | list span p50 | lines per entry | tet step <= 64 | Morton place p50 |")
            print("|---|---|---|---|---|---|---|---|---|")
            for o in run["order"]:
                print("| %d | %g | %.3f | %g | %.3f | %g | %.3f | %.3f | %g |" % (
                    o["step"], o["owner_step_p50"], o["owner_step_within_64"], o["pair_p50"], o["pair_within_64"],
                    o["list_span_p50"], o["lines_per_entry"], o["tet_step_within_64"], o["morton_place_p50"]))


if __name__ == "__main__":
    main(sys.argv[1:])
