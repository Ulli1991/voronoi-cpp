#!/usr/bin/env python3
"""Tables of docs/measure_2026-11-12.md from the jobs of scripts/ceiling_a100.sh: per kernel label DRAM bytes, bandwidth,
occupancy, registers and bound (Nsight Compute), and per phase launches, synchronisations and kernel time (Nsight
Systems). Usage: ceiling_tables.py <jobs dir> [--json out.json]"""
import csv
import glob
import json
import os
import sqlite3
import sys
from collections import defaultdict

N = 2097152  # 128³ cells
PEAK = 1555e9  # A100-SXM4-40GB DRAM, bytes per second
TARGET = 0.8 * PEAK
PHASE_COL = "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"


def ncu_rows(path):
    """The kernel rows of one ncu --csv --page raw file (program output lines dropped)."""
    lines = [ln for ln in open(path) if ln.startswith('"')]
    rows = list(csv.reader(lines))
    head = rows[0]
    out = []
    for r in rows[2:]:
        d = dict(zip(head, r))
        d["label"] = d["Kernel Name"].split("/")[0]
        d["phase"] = d[PHASE_COL].split(":")[1] if ":" in d[PHASE_COL] else ""
        out.append(d)
    return out


def num(d, k):
    v = d.get(k, "").replace(",", "")
    try:
        return float(v)
    except ValueError:
        return 0.0


def bound(dram_pct, sm_pct, mem_pct, ns):
    """Memory-bound at 60 % of peak DRAM, compute at 60 % of SM throughput, else latency (launch if under 10 us);
    L1/L2 when another memory unit is at 60 % or more."""
    if dram_pct >= 60:
        return "memory"
    if sm_pct >= 60:
        return "compute"
    if ns < 10e3:
        return "launch"
    return "latency" if mem_pct < 60 else "L1/L2"


def kernel_table(rows, calls=1):
    """Per (phase kind, label) and per call of the phase (`calls` profiled, e.g. warm-up and timed): launches, ns,
    bytes read and written, registers and occupancy (time-weighted)."""
    agg = defaultdict(lambda: defaultdict(float))
    for d in rows:
        phase = d["phase"].split("_", 1)[-1] if d["phase"].startswith("s") and d["phase"][1].isdigit() else d["phase"]
        a = agg[(phase, d["label"])]
        ns = num(d, "gpu__time_duration.sum")
        a["launches"] += 1
        a["ns"] += ns
        a["read"] += num(d, "dram__bytes_read.sum")
        a["write"] += num(d, "dram__bytes_write.sum")
        a["l2"] += num(d, "lts__t_bytes.sum")
        for k, m in (("regs", "launch__registers_per_thread"), ("occ", "sm__warps_active.avg.pct_of_peak_sustained_active"),
                     ("theo", "sm__maximum_warps_per_active_cycle_pct"), ("dram", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
                     ("sm", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
                     ("mem", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
                     ("issue", "smsp__issue_active.avg.pct_of_peak_sustained_active"),
                     ("tpi", "smsp__thread_inst_executed_per_inst_executed.ratio"),
                     ("clock", "sm__cycles_elapsed.avg.per_second"),
                     ("fp64", "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed"),
                     ("lsb", "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"),
                     ("lgt", "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct"),
                     ("wait", "smsp__warp_issue_stalled_wait_per_warp_active.pct"),
                     ("ssb", "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct"),
                     ("noi", "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct")):
            a[k] += num(d, m) * ns
        a["maxns"] = max(a["maxns"], ns)
    table = []
    for (phase, label), a in agg.items():
        ns = a["ns"] or 1
        w = {k: a[k] / ns for k in ("regs", "occ", "theo", "dram", "sm", "mem", "issue", "tpi", "clock", "fp64", "lsb", "lgt",
                                    "wait", "ssb", "noi")}
        bytes_ = a["read"] + a["write"]
        table.append(dict(phase=phase, label=label, launches=a["launches"] / calls, ms=a["ns"] / 1e6 / calls,
                          read_B=a["read"] / N / calls, write_B=a["write"] / N / calls, B=bytes_ / N / calls,
                          l2_B=a["l2"] / N / calls,
                          GBs=bytes_ / a["ns"] if a["ns"] else 0, pct_peak=100 * bytes_ / (a["ns"] * 1e-9) / PEAK if a["ns"] else 0,
                          floor_ms=bytes_ / TARGET * 1e3 / calls, bound=bound(w["dram"], w["sm"], w["mem"], a["ns"] / a["launches"]),
                          **w))
    table.sort(key=lambda r: (r["phase"], -r["ms"]))
    return table


def print_kernels(name, table):
    print(f"\n## {name}")
    print("| phase | kernel | launches | ms | B/cell read, write | GB/s | % peak | ms at 80 % | regs | "
          "occupancy achieved / theoretical | issue % | thr/inst | long sb % | bound |")
    for r in table:
        print(f"| {r['phase']} | {r['label']} | {r['launches']:g} | {r['ms']:.3f} | {r['read_B']:.1f}, {r['write_B']:.1f} | "
              f"{r['GBs']:.0f} | {r['pct_peak']:.0f} | {r['floor_ms']:.3f} | {r['regs']:.0f} | {r['occ']:.0f} / {r['theo']:.0f} | "
              f"{r['issue']:.0f} | {r['tpi']:.1f} | {r['lsb']:.0f} | {r['bound']} |")
    for ph in sorted({r["phase"] for r in table}):
        rs = [r for r in table if r["phase"] == ph]
        ms = sum(r["ms"] for r in rs)
        b = sum(r["B"] for r in rs)
        fl = sum(r["floor_ms"] for r in rs)
        clk = sum(r["clock"] * r["ms"] for r in rs) / ms if ms else 0
        print(f"TOTAL {name} {ph}: launches {sum(r['launches'] for r in rs):g} ms {ms:.2f} B/cell {b:.0f} "
              f"(read {sum(r['read_B'] for r in rs):.0f}) avg GB/s {b * N / ms / 1e6:.0f} ms at 80% {fl:.2f} "
              f"clock {clk / 1e6:.0f} MHz")


def nsys_phases(path):
    """Per phase kind: per occurrence launches, syncs, copies, memsets, kernel ms, wall ms (NVTX range)."""
    con = sqlite3.connect(path)
    strings = dict(con.execute("select id, value from StringIds"))
    nv = con.execute("select start, end, coalesce(text, (select value from StringIds where id = textId)) "
                     "from NVTX_EVENTS where end is not null").fetchall()
    rt = con.execute("select start, end, nameId, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()
    kern = {c: (s, e) for s, e, c in con.execute("select start, end, correlationId from CUPTI_ACTIVITY_KIND_KERNEL")}
    phases = [(s, e, t) for s, e, t in nv if t and (t in ("start", "start_cells", "list", "clip") or
                                                     (t[0] == "s" and t[1:2].isdigit()))]
    labels = sorted([(s, e, t) for s, e, t in nv if (s, e, t) not in set(phases)])
    out = defaultdict(list)
    for ps, pe, pt in phases:
        c = defaultdict(float)
        perlabel = defaultdict(float)
        for s, e, nid, corr in rt:
            if not (ps <= s <= pe):
                continue
            name = strings.get(nid, "")
            if "LaunchKernel" in name:
                c["launches"] += 1
                if corr in kern:
                    ks, ke = kern[corr]
                    c["kernel_ms"] += (ke - ks) / 1e6
                    lab = [t for ls, le, t in labels if ls <= s <= le]
                    perlabel[lab[-1] if lab else "?"] += (ke - ks) / 1e6
            elif "Synchronize" in name:
                c["syncs"] += 1
            elif "Memcpy" in name:
                c["copies"] += 1
            elif "Memset" in name:
                c["memsets"] += 1
        c["wall_ms"] = (pe - ps) / 1e6
        kind = pt.split("_", 1)[1] if pt[0] == "s" and pt[1:2].isdigit() else pt
        step = int(pt[1:].split("_")[0]) if kind != pt else 0
        out[kind].append((step, dict(c), dict(perlabel)))
    return out


def main():
    jobs = sys.argv[1]
    result = {}
    for f in sorted(glob.glob(os.path.join(jobs, "ncu_*.csv"))):
        name = os.path.basename(f)[4:-4]
        if name.startswith("smoke"):
            continue
        t = kernel_table(ncu_rows(f), 2 if name.startswith("rebuild") else 1)  # rebuild: warm-up and one timed
        result[name] = t
        print_kernels(name, t)
    for f in sorted(glob.glob(os.path.join(jobs, "nsys_*.sqlite"))):
        name = os.path.basename(f)[5:-7]
        ph = nsys_phases(f)
        print(f"\n## nsys {name}")
        nsys = {}
        for kind, occ in ph.items():
            sel = [c for s, c, _ in occ if s >= 2] or [c for _, c, _ in occ]
            avg = {k: sum(c.get(k, 0) for c in sel) / len(sel) for k in ("launches", "syncs", "copies", "memsets",
                                                                         "kernel_ms", "wall_ms")}
            labs = defaultdict(float)
            for s, _, pl in occ:
                if s >= 2 or len(sel) == len(occ):
                    for k, v in pl.items():
                        labs[k] += v / len(sel)
            nsys[kind] = dict(avg=avg, labels=dict(labs), count=len(sel))
            print(f"{kind} (mean of {len(sel)}): " + " ".join(f"{k}={v:.2f}" for k, v in avg.items()))
            for k, v in sorted(labs.items(), key=lambda x: -x[1])[:40]:
                print(f"    {k}: {v:.3f} ms")
        result["nsys_" + name] = nsys
    if "--json" in sys.argv:
        json.dump(result, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
