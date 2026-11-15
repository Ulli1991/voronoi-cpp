#!/usr/bin/env python3
"""Tables of the exact start: per-kernel Nsight Compute metrics (ncu CSV of scripts/start_fast_a100.sh) and the
start lines of the start / kernels / checksum modes, main (ref) against start-fast (new).

  python3 scripts/start_fast_tables.py ncu <n> <csv>...      one table per CSV: kernels of one start, by label
  python3 scripts/start_fast_tables.py log <log>...          start_ms and the start's kernels per input and design
"""
import csv
import re
import sys
from collections import OrderedDict

STALLS = ["long_scoreboard", "short_scoreboard", "lg_throttle", "mio_throttle", "wait", "barrier", "math_pipe_throttle",
          "no_instruction", "not_selected", "selected", "dispatch_stall", "drain", "membar", "branch_resolving", "tex_throttle",
          "imc_miss", "sleeping", "misc"]


def number(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def ncu_table(n, path):
    rows = []
    with open(path) as f:
        lines = [l for l in f if l.startswith('"')]
    reader = csv.reader(lines)
    header = next(reader)
    units = next(reader)
    for r in reader:
        rows.append(dict(zip(header, r)))
    kernels = OrderedDict()
    for r in rows:
        name = r["Kernel Name"].split("/")[0]
        k = kernels.setdefault(name, {"launches": 0, "ms": 0.0, "read": 0.0, "write": 0.0, "regs": 0, "occ": [],
                                      "theo": 0, "tpi": [], "stall": {}, "dram": [], "l2": [], "issue": []})
        for key, metric in (("dram", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
                            ("l2", "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
                            ("issue", "sm__issue_active.avg.pct_of_peak_sustained_elapsed")):
            v = number(r.get(metric))
            if v is not None:
                k[key].append(v)
        k["launches"] += 1
        ns = number(r.get("gpu__time_duration.sum"))
        k["ms"] += (ns or 0) / 1e6 if units[header.index("gpu__time_duration.sum")] in ("nsecond", "ns") else (ns or 0) * 1e-3
        k["read"] += number(r.get("dram__bytes_read.sum")) or 0
        k["write"] += number(r.get("dram__bytes_write.sum")) or 0
        k["regs"] = int(number(r.get("launch__registers_per_thread")) or 0)
        occ = number(r.get("sm__warps_active.avg.pct_of_peak_sustained_active"))
        if occ is not None:
            k["occ"].append(occ)
        k["theo"] = number(r.get("sm__maximum_warps_per_active_cycle_pct")) or k["theo"]
        tpi = number(r.get("smsp__thread_inst_executed_per_inst_executed.ratio"))
        if tpi is not None:
            k["tpi"].append(tpi)
        for s in STALLS:
            v = number(r.get("smsp__average_warp_latency_issue_stalled_%s.ratio" % s)) or number(
                r.get("smsp__average_warps_issue_stalled_%s_per_issue_active.ratio" % s))
            if v:
                k["stall"][s] = k["stall"].get(s, 0) + v
    byte_unit = units[header.index("dram__bytes_read.sum")]
    scale = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}.get(byte_unit, 1)
    print("| kernel | launches | ms | B/cell read + written | GB/s | DRAM % | L2 % | issue active % | registers | warps active / possible % | threads per instruction | top stall |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    total = [0, 0.0, 0.0, 0.0]
    for name, k in sorted(kernels.items(), key=lambda kv: -kv[1]["ms"]):
        rd, wr = k["read"] * scale / n, k["write"] * scale / n
        gbs = (k["read"] + k["write"]) * scale / (k["ms"] * 1e-3) / 1e9 if k["ms"] else 0
        occ = sum(k["occ"]) / len(k["occ"]) if k["occ"] else 0
        tpi = sum(k["tpi"]) / len(k["tpi"]) if k["tpi"] else 0
        top = max(k["stall"].items(), key=lambda kv: kv[1])[0] if k["stall"] else ""
        mean = lambda xs: sum(xs) / len(xs) if xs else 0
        print("| `%s` | %d | %.2f | %.0f + %.0f | %.0f | %.0f | %.0f | %.0f | %d | %.0f / %.0f | %.1f | %s |" % (
            name, k["launches"], k["ms"], rd, wr, gbs, mean(k["dram"]), mean(k["l2"]), mean(k["issue"]), k["regs"], occ,
            k["theo"], tpi, top))
        total[0] += k["launches"]
        total[1] += k["ms"]
        total[2] += rd
        total[3] += wr
    print("| **total** | %d | %.2f | %.0f + %.0f | | | | | | | | |" % tuple(total))


def log_table(paths):
    runs = OrderedDict()  # (input, n) -> design -> {start_ms: [...], kernels: {label: ms}}
    for path in paths:
        key = None
        for line in open(path):
            m = re.match(r"== \w+ (\S+) (\S+) (\S+)", line)
            if m:
                design, kind, n = m.groups()
                key = (kind, n)
                runs.setdefault(key, OrderedDict()).setdefault(design, {"start_ms": [], "kernels": {}, "line": ""})
                continue
            if key is None:
                continue
            d = runs[key][design]
            m = re.match(r"start_kernel: (\S+) ms_per_step=(\S+)", line)
            if m:
                d["kernels"][m.group(1)] = float(m.group(2))
            m = re.search(r"start_ms=(\S+)", line)
            if line.startswith("start:") and m:
                d["start_ms"].append(float(m.group(1)))
                d["line"] = line
    for (kind, n), designs in runs.items():
        print("\n%s, n = %s" % (kind, n))
        for design, d in designs.items():
            ms = sorted(d["start_ms"])
            med = ms[len(ms) // 2] if ms else 0
            rate = int(n) / med / 1e3 if med else 0
            r = re.search(r"rounds=(\d+) bad_tets=(\d+) patched_cells=(\d+)", d["line"])
            print("  %s: start_ms %.1f (%s), %.1f M cells/s, %.1f ns/cell%s" % (
                design, med, ", ".join("%.1f" % x for x in ms), rate, med * 1e6 / int(n),
                (", rounds %s bad %s patched %s" % r.groups()) if r else ""))
        labels = []
        for d in designs.values():
            for label in d["kernels"]:
                if label not in labels and label.startswith(("tet_", "shell_voronoi::construct")):
                    labels.append(label)
        if labels:
            print("  | kernel | " + " | ".join("%s ms" % x for x in designs) + " |")
            for label in labels:
                print("  | `%s` | " % label + " | ".join(
                    "%.2f" % d["kernels"][label] if label in d["kernels"] else "" for d in designs.values()) + " |")


if __name__ == "__main__":
    if sys.argv[1] == "ncu":
        for p in sys.argv[3:]:
            print("\n" + p)
            ncu_table(int(sys.argv[2]), p)
    else:
        log_table(sys.argv[2:])
