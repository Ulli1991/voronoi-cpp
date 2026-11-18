"""Nsight Compute rows per kernel label (research_notes/profile_2026-11-18.md).

    python3 profile_ncu.py <csv> <cells> [repeats] [--peak-tbs 1.37] [--peak-tflops 8.85]

<csv>: ncu --csv --page raw with --nvtx (kernels named by the innermost NVTX range of scripts/kp_profile.cpp). Launches of a
label are summed and divided by `repeats` (the windows the csv holds, e.g. two rebuilds). Per label: time, launches, DRAM
read + write per cell, achieved DRAM bandwidth and its share of the measured stream peak, L1 and L2 hit (time-weighted),
FP64 flops and their share of the FMA peak, FP64 pipe busy, warps active against the theoretical limit, registers, local
memory (spill) bytes per cell, threads per warp instruction, the three largest stall reasons, and the time the DRAM bytes
alone would take at the peak (t_bytes).
"""
import csv
import re
import sys
from collections import OrderedDict, defaultdict

STALLS = ["long_scoreboard", "short_scoreboard", "math_pipe_throttle", "lg_throttle", "mio_throttle", "tex_throttle",
          "branch_resolving", "no_instruction", "barrier", "wait", "not_selected", "selected", "dispatch_stall", "drain",
          "misc", "membar", "imc_miss"]


def num(x):
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return 0.0


def label(nvtx, kernel):
    names = re.findall(r"(?:<default domain>|CCCL):(.*?):none:", nvtx)
    names = [n for n in names if not re.fullmatch(r"s\d+_(update|cells)|list|clip|start|start_cells", n)]
    return names[-1] if names else "(unnamed) " + kernel[:50]


def main():
    path, cells = sys.argv[1], int(sys.argv[2])
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 1
    peak_tbs = float(sys.argv[sys.argv.index("--peak-tbs") + 1]) if "--peak-tbs" in sys.argv else 1.37
    peak_tf = float(sys.argv[sys.argv.index("--peak-tflops") + 1]) if "--peak-tflops" in sys.argv else 8.85
    with open(path) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"ID"'))
    rows = list(csv.reader(lines[start:]))
    head, data = rows[0], rows[2:]
    col = {h: i for i, h in enumerate(head)}
    nvtx_col = next(i for i, h in enumerate(head) if "Push/Pop" in h)
    g = OrderedDict()
    for r in data:
        if len(r) < len(head):
            continue
        lab = label(r[nvtx_col], r[col["Kernel Name"]])
        e = g.setdefault(lab, defaultdict(float))
        t = num(r[col["gpu__time_duration.sum"]])  # ns
        e["t"] += t
        e["n"] += 1
        e["rd"] += num(r[col["dram__bytes_read.sum"]])
        e["wr"] += num(r[col["dram__bytes_write.sum"]])
        e["l2b"] += num(r[col["lts__t_bytes.sum"]])
        for k, m in (("l1", "l1tex__t_sector_hit_rate.pct"), ("l2", "lts__t_sector_hit_rate.pct"),
                     ("fp64", "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed"),
                     ("warps", "sm__warps_active.avg.pct_of_peak_sustained_active"),
                     ("theo", "sm__maximum_warps_per_active_cycle_pct"),
                     ("tpi", "smsp__thread_inst_executed_per_inst_executed.ratio"),
                     ("issue", "smsp__issue_active.avg.pct_of_peak_sustained_active"),
                     ("clk", "sm__cycles_elapsed.avg.per_second")):
            e[k] += t * num(r[col[m]])
        for s in STALLS:
            e["st_" + s] += t * num(r[col[f"smsp__warp_issue_stalled_{s}_per_warp_active.pct"]])
        e["flop"] += 2 * num(r[col["sm__sass_thread_inst_executed_op_dfma_pred_on.sum"]]) + num(
            r[col["sm__sass_thread_inst_executed_op_dadd_pred_on.sum"]]) + num(r[col["sm__sass_thread_inst_executed_op_dmul_pred_on.sum"]])
        e["local"] += num(r[col["l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum"]]) + num(r[col["l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum"]])
        e["regs"] = max(e["regs"], num(r[col["launch__registers_per_thread"]]))
        e["inst"] += num(r[col["smsp__inst_executed.sum"]])
    total = sum(e["t"] for e in g.values())
    print(f"# {path}: {len(data)} launches, {repeats} window(s), kernel time per window {total / repeats / 1e6:.3f} ms, "
          f"cells={cells}; peaks {peak_tbs} TB/s, {peak_tf} TFLOP/s")
    print("| kernel | ms | share | launches | ns/cell | DRAM B/cell (rd+wr) | TB/s | % of peak BW | L1 hit | L2 hit | flop/cell | "
          "% FP64 peak | FP64 pipe | warps active / theoretical % | regs | local B/cell | threads/inst | top stalls (% of warp cycles) | t_bytes ms | x over t_bytes |")
    print("|" + "---|" * 20)
    cum = 0
    for lab, e in sorted(g.items(), key=lambda kv: -kv[1]["t"]):
        t = e["t"]
        if t <= 0:
            continue
        cum += t
        w = lambda k: e[k] / t
        byt = e["rd"] + e["wr"]
        tbs = byt / t / 1e3  # B/ns = GB/s -> TB/s
        stalls = sorted(((w("st_" + s), s) for s in STALLS), reverse=True)[:3]
        st = ", ".join(f"{s} {v:.0f}" for v, s in stalls)
        tb = byt / (peak_tbs * 1e12) * 1e3 / repeats
        print(f"| {lab} | {t / repeats / 1e6:.3f} | {100 * t / total:.1f} % ({100 * cum / total:.0f}) | {e['n'] / repeats:g} | "
              f"{t / repeats / cells:.3f} | {e['rd'] / repeats / cells:.0f}+{e['wr'] / repeats / cells:.0f} | {tbs:.2f} | {100 * tbs / peak_tbs:.0f} | "
              f"{w('l1'):.0f} | {w('l2'):.0f} | {e['flop'] / repeats / cells:.0f} | {100 * e['flop'] / t / 1e3 / peak_tf:.1f} | {w('fp64'):.0f} | "
              f"{w('warps'):.0f} / {w('theo'):.0f} | {e['regs']:.0f} | {e['local'] / repeats / cells:.1f} | {w('tpi'):.1f} | {st} | {tb:.3f} | {t / repeats / 1e6 / tb if tb else 0:.1f} |")
    print(f"clock (time-weighted SM cycles/s): {sum(e['clk'] for e in g.values()) / total / 1e6:.0f} MHz")


if __name__ == "__main__":
    main()
