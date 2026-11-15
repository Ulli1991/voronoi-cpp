#!/usr/bin/env python3
"""Tables of scripts/fewer_clips_a100.sh logs: rate (M cells/s, kNN pass and clip ms), work (clips, cuts, fallbacks)
and measure (candidate reads). Usage: fewer_clips_tables.py <log>... (any mix of modes)."""
import re
import sys
from collections import defaultdict

rate = defaultdict(list)  # (design, kind, n) -> construct_ms
listed = {}  # (design, kind, n) -> dict of the listed line
work = {}  # (design, kind, n) -> {work1: (mean, p99), ...}
reads = {}  # (design, kind, n) -> (mean, p99) of measure: tested
kinds, designs = [], []
MEASURED = "new2"


def number(text):
    return float(text) if re.match(r"^-?[0-9.e+-]+$", text) else text


for path in sys.argv[1:]:
    key, mode = None, None
    for line in open(path):
        m = re.match(r"^== (\w+) (\S+) (\w+) (\d+)", line)
        if m:
            mode = m.group(1)
            design = re.sub(r"^measure:", MEASURED + ":", m.group(2))  # the measurement build counts for MEASURED
            key = (design, m.group(3), int(m.group(4)))
            if m.group(3) not in kinds:
                kinds.append(m.group(3))
            if design not in designs:
                designs.append(design)
            continue
        if key is None:
            continue
        fields = dict(f.split("=", 1) for f in line.split()[1:] if "=" in f)
        if line.startswith("warp:") and mode == "rate":
            rate[key].append(float(fields["construct_ms"]))
        elif line.startswith("listed:") and (mode == "rate" or key not in listed):
            listed[key] = {k: number(v) for k, v in fields.items()}
        elif re.match(r"^work\d:", line):
            work.setdefault(key, {})[line.split(":")[0]] = (float(fields["mean"]), int(fields["p99"]))
        elif line.startswith("measure: tested "):
            reads[key] = (float(fields["mean"]), int(fields["p99"]))

sizes = sorted({k[2] for k in list(rate) + list(work) + list(reads)})
for n in sizes:
    print(f"\n## n = {n}\n")
    print("| design | input | M cells/s | vs ref | kNN pass ms | clip ms | walk ms | fallback | clips mean/p99 |"
          " cuts | reads |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for kind in kinds:
        ref = rate.get(("ref:0", kind, n))
        base = min(ref) if ref else None
        for d in designs:
            key = (d, kind, n)
            if key not in rate and key not in work and key not in reads:
                continue
            ms = sorted(rate[key])[len(rate[key]) // 2] if key in rate else None
            speed = f"{n / ms / 1e3:.1f}" if ms else ""
            versus = f"{(base / ms - 1) * 100:+.0f} %" if ms and base else ""
            l = listed.get(key, {})
            fall = ""
            if l:
                fall = f"{int(l['walked'])} ({100 * l['walked'] / n:.1f} %)"
                if l.get("relisted", 0) > 0:
                    fall = f"{int(l['relisted'])} relisted, " + fall
            w = work.get(key, {})
            clips = f"{w['work1'][0]:.1f} / {w['work1'][1]}" if "work1" in w else ""
            cuts = f"{w['work2'][0]:.1f}" if "work2" in w else ""
            r = ""
            if key in reads:
                total = reads[key][0] + l.get("list_reads_per_cell", 0) + l.get("relist_reads_per_cell", 0)
                r = f"{total:.0f}"
            print(f"| {d} | {kind} | {speed} | {versus} | {l.get('list_ms', '')} | {l.get('clip_ms', '')} |"
                  f" {l.get('walk_ms', '')} | {fall} | {clips} | {cuts} | {r} |")
