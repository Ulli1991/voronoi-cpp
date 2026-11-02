"""Throughput table of scripts/hydro_speed_a100.sh (speed128, speed256, builds) logs.

    python3 scripts/hydro_speed_table.py <log> [<log> ...]

Per run label (the "== " line without the round), the mean over rounds of
each phase in M cells/s (n / phase median); "sweep" includes the primitives
pass (main: time step + point velocities + gradients); "hydro" = sweep +
fluxes + update (per-cell loop: fluxes includes the update), also in G face
updates/s (faces stored once / hydro time); the whole step in M cell
updates/s; the peak device memory over the steps in bytes per cell.
"""
import re
import sys
from collections import OrderedDict, defaultdict

runs = OrderedDict()
label = None
for path in sys.argv[1:]:
    for line in open(path):
        if line.startswith("== "):
            label = re.sub(r" round \d+", "", line[3:].strip())
            runs.setdefault(label, defaultdict(list))
            continue
        if label is None:
            continue
        fields = dict(re.findall(r"(\w+)=([-+0-9.eE]+)", line))
        if line.startswith("mesh: n="):
            runs[label]["n"].append(float(fields["n"]))
            runs[label]["faces"].append(float(fields["faces"]))
        elif line.startswith("phases_ms_median"):
            f = {k: float(v) for k, v in fields.items()}
            if "sweep" not in f:  # main
                f["sweep"] = f["time_step"] + f["velocities"] + f["gradients"]
            f["sweep"] += f.get("primitives", 0.0)
            f["hydro"] = f["sweep"] + f["fluxes"] + f["update"]
            for k in ("sweep", "fluxes", "update", "mesh", "cells", "hydro", "step"):
                runs[label][k].append(f[k])
        elif line.startswith("memory: phase=steps"):
            runs[label]["memory"].append(float(fields["peak_bytes_per_cell"]))


def rate(ms, count):
    return count / (ms * 1e-3) / 1e6 if ms > 0 else float("nan")


cols = ("sweep", "fluxes", "update", "mesh", "cells", "hydro", "step")
print("| run | " + " | ".join(cols) + " | hydro G face/s | peak B/cell |")
print("|---" * (len(cols) + 3) + "|")
for label, r in runs.items():
    if not r.get("step"):
        continue
    cells, faces = r["n"][0], r["faces"][0]
    mean = {k: sum(v) / len(v) for k, v in r.items() if v}
    row = []
    for k in cols:
        row.append("" if mean.get(k, 0) == 0 else f"{rate(mean[k], cells):.1f}")
    memory = f"{max(r['memory']):.0f}" if r.get("memory") else ""
    print(f"| {label} | " + " | ".join(row) + f" | {rate(mean['hydro'], faces) / 1e3:.2f} | {memory} |")
