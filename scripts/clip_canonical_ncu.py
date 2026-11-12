# Key numbers of each warp-kernel .ncu-rep: kernel ms, warp instructions per cell, spills per cell, and the shares of
# the stall samples. Usage: python3 clip_canonical_ncu.py <cells> <reports>
import csv, subprocess, sys

NCU = "/mpcdf/soft/SLE_15/packages/x86_64/nsight_compute/2026/ncu"
cells = float(sys.argv[1])
groups = {"fixed latency": ["wait"], "shared memory": ["short_scoreboard", "mio_throttle"],
          "instruction fetch": ["no_instructions", "branch_resolving"], "memory": ["long_scoreboard", "lg_throttle"],
          "issue": ["selected", "not_selected"], "math pipe": ["math_pipe_throttle"],
          "other": ["barrier", "dispatch_stall", "drain", "imc_miss", "membar", "misc", "sleeping", "tex_throttle"]}
for rep in sys.argv[2:]:
    out = subprocess.run([NCU, "--import", rep, "--page", "raw", "--csv"], capture_output=True, text=True).stdout
    rows = list(csv.reader(out.splitlines()))
    h, r = rows[0], rows[2]
    g = lambda m: float(r[h.index(m)].replace(",", "") or 0) if m in h else 0.0
    total = sum(g("smsp__pcsamp_warps_issue_stalled_" + s) for v in groups.values() for s in v) or 1
    shares = {k: 100 * sum(g("smsp__pcsamp_warps_issue_stalled_" + s) for s in v) / total for k, v in groups.items()}
    spill = (g("sass__inst_executed_register_spilling_mem_local_op_read") +
             g("sass__inst_executed_register_spilling_mem_local_op_write"))
    print(rep.split("/")[-1], f"ms={g('gpu__time_duration.sum'):.1f}",
          f"instructions/cell={g('smsp__inst_executed.sum') / cells:.0f}", f"spills/cell={spill / cells:.0f}",
          f"registers={g('launch__registers_per_thread'):.0f}",
          " ".join(f"{k}={v:.0f}%" for k, v in shares.items()))
