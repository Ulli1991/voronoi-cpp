#!/usr/bin/env python3
"""Tables of scripts/step2x_a100.sh logs: step rates per build, input, size and spacing; the 500-step runs; checksums.

Usage: step2x_tables.py <mode> <log>...   mode: rates, long, bits, kernels."""
import re
import statistics
import sys
from collections import defaultdict


def fields(text):
    return dict(kv.split("=", 1) for kv in text.split() if "=" in kv)


def lines(paths):
    """(build, args, kind, fields) of every bench line; args are the run's arguments."""
    for path in paths:
        for line in open(path):
            m = re.match(r"^(\S+) (step [^|]*)\| (\w+): ?(.*)$", line)
            if m:
                yield m.group(1), m.group(2).strip(), m.group(3), m.group(4)


def rates(paths):
    """Mean over steps 2..10 of all_ms per (build, n, kind, spacing), rounds averaged: M cells/s and ns per cell."""
    runs = defaultdict(list)
    for build, args, kind, text in lines(paths):
        if kind != "gpuflips":
            continue
        a = args.split()
        f = fields(text)
        if int(f["step"]) < 2:
            continue
        runs[(a[1], a[2], a[3], build)].append(float(f["all_ms"]))
    builds = sorted({k[3] for k in runs}, key=lambda b: (b != "main", b))
    print("| n | input | spacings | " + " | ".join(f"{b} M cells/s" for b in builds) + " | " +
          " | ".join(f"{b} / main" for b in builds if b != "main") + " | " + " | ".join(f"{b} ns/cell" for b in builds) + " |")
    print("|---" * (3 + 3 * len(builds) - 1) + "|")
    for key in sorted({k[:3] for k in runs}, key=lambda k: (int(k[0]), k[1], float(k[2]))):
        n = int(key[0])
        ms = {b: statistics.mean(runs[key + (b,)]) for b in builds if runs.get(key + (b,))}
        rate = {b: n / ms[b] / 1e3 for b in ms}
        size = round(n ** (1 / 3))
        row = [f"{size}³", key[1], key[2]]
        row += [f"{rate[b]:.1f}" if b in rate else "" for b in builds]
        row += [f"{rate[b] / rate['main']:.3f}" if b in rate and "main" in rate else "" for b in builds if b != "main"]
        row += [f"{ms[b] * 1e6 / n:.2f}" if b in ms else "" for b in builds]
        print("| " + " | ".join(row) + " |")


def long_runs(paths):
    """500-step runs: rate over the run (re-sorts paid), steps 2-11, the last 10, re-sorts and their cost."""
    runs = defaultdict(list)
    for build, args, kind, text in lines(paths):
        if kind == "gpuflips" and "advect_ms" in text:  # a line cut off by the end of a job is dropped
            runs[(build, args)].append(fields(text))
    print("| build | flags | n | spacings | M cells/s over the run | steps 2-11 | last 10 | re-sorts | re-sort ms (mean) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (build, args), steps in runs.items():
        a = args.split()
        n = int(a[1])
        flags = " ".join(a[5:]) or "(default)"
        total = sum(float(s["all_ms"]) for s in steps[1:])
        early = sum(float(s["all_ms"]) for s in steps[1:11])
        late = sum(float(s["all_ms"]) for s in steps[-10:])
        resorts = [float(s["resort_ms"]) for s in steps if s.get("resorted") == "1"]
        mean = f"{statistics.mean(resorts):.1f}" if resorts else ""
        print(f"| {build} | {flags} | {round(n ** (1 / 3))}³ | {a[3]} | {n * (len(steps) - 1) / total / 1e3:.1f} | "
              f"{n * 10 / early / 1e3:.1f} | {n * 10 / late / 1e3:.1f} | {len(resorts)} | {mean} |")


def bits(paths):
    """Checksums per (input, spacing, step): equal across builds or not."""
    sums = defaultdict(dict)
    for build, args, kind, text in lines(paths):
        if kind != "geometry_checksum":
            continue
        a = args.split()
        f = fields(text)
        key = (a[2], a[3], f["step"])
        sums[key][build] = (text.split()[1], f["faces_checksum"], f["lists_checksum"], f["cells_checksum"])
    # geometry, faces and lists do not depend on the face numbering; cells_checksum hashes the face numbers too
    same = differ = numbering = 0
    for key, by in sorted(sums.items()):
        values = {v[:3] for v in by.values()}
        if len(values) == 1 and len(by) > 1:
            same += 1
        else:
            differ += 1
            print("DIFFER", key, by)
        numbering += len({v[3] for v in by.values()}) > 1
    print(f"geometry, faces, lists equal across builds: {same} lines, differing: {differ}; "
          f"cells_checksum (face numbers) differing: {numbering}")


def kernels(paths):
    """Per-kernel ms per step, by build, input and spacing."""
    table = defaultdict(dict)
    for build, args, kind, text in lines(paths):
        if kind != "kernel":
            continue
        a = args.split()
        label, rest = text.rsplit(" ms_per_step=", 1)
        table[(a[2], a[3], build)][label] = float(rest.split()[0])
    for key, ks in sorted(table.items()):
        print(f"## {' '.join(key)}: {sum(ks.values()):.2f} ms in all")
        for label, ms in sorted(ks.items(), key=lambda kv: -kv[1])[:30]:
            print(f"  {ms:7.3f}  {label}")


{"rates": rates, "long": long_runs, "bits": bits, "kernels": kernels}[sys.argv[1]](sys.argv[2:])
