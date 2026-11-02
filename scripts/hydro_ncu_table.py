"""Per-kernel medians of an Nsight Compute CSV (--csv --page raw) of hydro_sedov.

    python3 scripts/hydro_ncu_table.py <ncu.csv> [cells] [faces]

Kernels are grouped by a short name (the hydro functor or lambda; mesh
kernels by their label in the demangled name); per group: launches, median
time (us), DRAM bytes per cell and per face (read + write), L2 bytes per
cell, active warps (% of peak), long-scoreboard stall (% of issue cycles),
registers per thread.
"""
import csv
import re
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
cells = float(sys.argv[2]) if len(sys.argv) > 2 else 2097152.0
faces = float(sys.argv[3]) if len(sys.argv) > 3 else 16680562.0
rows = [r for r in csv.reader(l for l in open(path) if l.startswith('"'))]
head, units, data = rows[0], rows[1], rows[2:]
col = {name: i for i, name in enumerate(head)}


def short(name):
    for pat in (r"Sweep<[^>]*>", r"advance<\d, \w+>\(double, \w+::Timings\*\)::\{lambda\(int(, long long&)?\)#\d\}",
                r"gradients\(\)", r"time_step\(double", r"point_velocities_kernel", r"move\(double",
                r"tet_mesh::\w+|tet_repair::\w+|voronoi::\w+|neighbor_search::\w+"):
        m = re.search(pat, name)
        if m:
            return m.group(0)
    return name[:60]


def value(r, metric):
    try:
        return float(r[col[metric]].replace(",", ""))
    except (KeyError, ValueError):
        return float("nan")


groups = defaultdict(list)
for r in data:
    groups[short(r[col["Kernel Name"]])].append(r)
scale = {"ns": 1e-3, "nsecond": 1e-3, "us": 1.0, "usecond": 1.0, "ms": 1e3, "msecond": 1e3}  # to us
t_unit = scale.get(units[col["gpu__time_duration.sum"]], 1.0)
b_scale = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "KB": 1e3, "MB": 1e6, "GB": 1e9}
print("| kernel | n | ms | DRAM B/cell | DRAM B/face | DRAM % of peak | sectors/request | L2 B/cell | active warps % | long-scoreboard % | regs |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for name, rs in sorted(groups.items(), key=lambda kv: -statistics.median(value(r, "gpu__time_duration.sum") for r in kv[1])):
    t = statistics.median(value(r, "gpu__time_duration.sum") for r in rs) * t_unit
    dram = statistics.median(
        value(r, "dram__bytes_read.sum") * b_scale.get(units[col["dram__bytes_read.sum"]], 1)
        + value(r, "dram__bytes_write.sum") * b_scale.get(units[col["dram__bytes_write.sum"]], 1)
        for r in rs)
    l2 = statistics.median(value(r, "lts__t_bytes.sum") * b_scale.get(units[col["lts__t_bytes.sum"]], 1) for r in rs)
    warps = statistics.median(value(r, "sm__warps_active.avg.pct_of_peak_sustained_active") for r in rs)
    stall = statistics.median(value(r, "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct") for r in rs)
    regs = statistics.median(value(r, "launch__registers_per_thread") for r in rs)
    if t < 50:  # us
        continue
    pct = statistics.median(value(r, "dram__throughput.avg.pct_of_peak_sustained_elapsed") for r in rs)
    spr = statistics.median(value(r, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
                            / max(value(r, "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"), 1) for r in rs)
    print(f"| {name} | {len(rs)} | {t / 1e3:.2f} | {dram / cells:.0f} | {dram / faces:.0f} | {pct:.0f} | {spr:.1f} | "
          f"{l2 / cells:.0f} | {warps:.0f} | {stall:.0f} | {regs:.0f} |")
