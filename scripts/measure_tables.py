#!/usr/bin/env python3
"""Tables and figures of docs/measure_2026-11-10.md from the job logs in build-measure/jobs (scripts/measure_a100.sh).
Usage: python scripts/measure_tables.py build-measure/jobs [figures-dir]. Needs numpy, scipy, matplotlib (python-waterboa)."""
import csv
import glob
import io
import os
import re
import statistics
import sys

import numpy as np

JOBS = sys.argv[1] if len(sys.argv) > 1 else "build-measure/jobs"
FIGS = sys.argv[2] if len(sys.argv) > 2 else None
COLUMNS = ("tested taken clipped cuts vertices_end vertices_peak walk_cells walk_open tested_outside taken_outside "
           "open_outside in_open_outside warp_status cycles batches").split()


def blocks(path, marker):
    """Split a log into (header, lines) at lines starting with `marker`."""
    out, head, body = [], None, []
    for line in open(path, errors="replace"):
        if line.startswith(marker):
            if head:
                out.append((head, body))
            head, body = line.strip(), []
        elif head:
            body.append(line.rstrip("\n"))
    if head:
        out.append((head, body))
    return out


def fields(line):
    return dict(re.findall(r"([\w%./]+)=(-?[\w.+-]+)", line))


def clip_tables():
    rates = {}
    for f in glob.glob(f"{JOBS}/rate*-*.out"):
        for head, body in blocks(f, "== rate"):
            _, _, kind, n, box = head.split()
            tag = ("slab" if box != "box=1" else kind) + "_" + n
            for line in body:
                if line.startswith("warp:"):
                    rates[tag] = float(fields(line)["construct_Mcells_s"])
    rows = {}
    for f in sorted(glob.glob(f"{JOBS}/clip*-*.out")):
        for head, body in blocks(f, "== clip"):
            tag = head.split()[2]
            d = {"rate": rates.get(tag)}
            for line in body:
                if line.startswith("measure: ") and "mean=" in line:
                    name = line.split()[1]
                    d[name] = fields(line)
                elif line.startswith(("measure: warp_status ok", "measure: cycle_share", "measure: ratios")):
                    d.update(fields(line))
                elif line.startswith("shuffle: ") and "per_cell" in line:
                    d["shuffle_" + line.split()[1]] = fields(line)
            rows[tag] = d
    print("\n### Clipping per cell (warp kernel)\n")
    print("| input | cells | M cells/s | clips mean, p50, p99, max | cuts mean, p99 | peak vertices p50, p99, max | "
          "tested mean, p99 | slowest 1 % / 10 % of cells: share of cycles | max / median cycles | to thread kernel |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for tag, d in rows.items():
        kind, n = tag.rsplit("_", 1)
        c, k, v, t, y = d["clipped"], d["cuts"], d["vertices_peak"], d["tested"], d["cycles"]
        sent = int(n) - int(d["ok"])
        print(f"| {kind} | {int(n):,} | {d['rate']:.1f} | {float(c['mean']):.1f}, {c['p50']}, {c['p99']}, {c['max']} | "
              f"{float(k['mean']):.1f}, {k['p99']} | {v['p50']}, {v['p99']}, {v['max']} | {float(t['mean']):.0f}, {t['p99']} | "
              f"{100 * float(d['slowest_1%']):.1f} % / {100 * float(d['slowest_10%']):.1f} % | "
              f"{int(y['max']) / int(y['p50']):.1f} | {sent} |")
    print("\n### Plume P1: work outside every vertex ball\n")
    print("| input | cells | taken / tested | clips that cut | tested outside every ball | taken outside | "
          "opened grid cells meeting no ball | their particles / tested |")
    print("|---|---|---|---|---|---|---|---|")
    for tag, d in rows.items():
        kind, n = tag.rsplit("_", 1)
        p = lambda key: f"{100 * float(d[key]):.0f} %"
        print(f"| {kind} | {int(n):,} | {p('taken/tested')} | {p('cuts/clipped')} | {p('tested_outside/tested')} | "
              f"{p('taken_outside/taken')} | {p('open_outside/open')} | {p('in_open_outside/tested')} |")
    print("\n### Shuffle test (V3): per cell, replayed on the host\n")
    print("| input | cells | clips: kernel, sorted, random mean, worst | cuts: kernel, sorted, random | "
          "vertex tests: kernel, sorted, best of 66 | spread of clips p50, p90 | spread of vertex tests p50, p90 |")
    print("|---|---|---|---|---|---|---|")
    for tag, d in rows.items():
        kind, n = tag.rsplit("_", 1)
        c, k, w = d["shuffle_clips"], d["shuffle_cuts"], d["shuffle_vertex_tests"]
        print(f"| {kind} | {int(n):,} | {c['kernel']}, {c['sorted']}, {c['random_mean']}, {c['worst']} | "
              f"{k['kernel']}, {k['sorted']}, {k['random_mean']} | {w['kernel']}, {w['sorted']}, {w['best']} | "
              f"{c['spread_p50']}, {c['spread_p90']} | {w['spread_p50']}, {w['spread_p90']} |")
    return rows


def attribution():
    from scipy.optimize import nnls
    print("\n### Cycles per cell attributed by non-negative least squares (128³ inputs)\n")
    print("| input | cycles per distance test | per clip that cuts nothing | per cut | per walked grid cell | R² | "
          "share: tests, empty clips, cuts, walk | P1 bound: empty clips + tests outside every ball |")
    print("|---|---|---|---|---|---|---|---|")
    data = {}
    for f in sorted(glob.glob(f"{JOBS}/clip_*.npy")):
        tag = os.path.basename(f)[5:-4]
        a = np.load(f).astype(float)
        d = {c: a[:, i] for i, c in enumerate(COLUMNS)}
        data[tag] = d
        ok = d["warp_status"] == 0
        X = np.column_stack([d["tested"], d["clipped"] - d["cuts"], d["cuts"], d["walk_cells"]])[ok]
        y = d["cycles"][ok]
        coef, _ = nnls(X, y)
        r2 = 1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        share = (X * coef).sum(0) / y.sum()
        p1 = (coef[0] * d["tested_outside"][ok] + coef[1] * (d["clipped"] - d["cuts"])[ok]).sum() / y.sum()
        print(f"| {tag.rsplit('_', 1)[0]} | {coef[0]:.0f} | {coef[1]:.0f} | {coef[2]:.0f} | {coef[3]:.0f} | {r2:.2f} | "
              + ", ".join(f"{100 * s:.0f} %" for s in share) + f" | {100 * p1:.0f} % |")
    return data


def figures(data):
    if not FIGS or not data:
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle

    figstyle.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(2 * figstyle.FIGSIZE[0], figstyle.FIGSIZE[1]))
    for tag, d in data.items():
        label = tag.rsplit("_", 1)[0]
        clips = d["clipped"]
        bins = np.arange(0, clips.max() + 2)
        h, _ = np.histogram(clips, bins=bins)
        axes[0].step(bins[:-1], h / len(clips), where="post", label=label)
        cyc = np.sort(d["cycles"])[::-1]
        share = np.cumsum(cyc) / cyc.sum()
        frac = np.arange(1, len(cyc) + 1) / len(cyc)
        axes[1].plot(frac, share, label=label)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("clips per cell")
    axes[0].set_ylabel("fraction of cells")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("fraction of cells, slowest first")
    axes[1].set_ylabel("share of warp cycles")
    axes[1].plot([1e-6, 1], [1e-6, 1], color="0.5", linestyle=":", linewidth=1)
    axes[1].set_xlim(1e-6, 1)
    for ax in axes:
        figstyle.decorate(ax)
    figstyle.legend(axes[0])
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(f"{FIGS}/measure_clip_tails.png")


def step_tables():
    print("\n### Moving steps (plain build: rate; measurement build: what changed)\n")
    print("| input | cells | spacings per step | step M cells/s (with update_cells) | flips per cell | rounds | "
          "tets screened | tets created | faces re-walked | cells whose neighbours changed | new neighbours outside 2-ring / 3-ring |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    kernels = {}
    for f in sorted(glob.glob(f"{JOBS}/step*-*.out")):
        for head, body in blocks(f, "== step"):
            _, _, kind, n, box, d = head.split()
            kind = "slab" if box != "box=1" else kind
            flips = [fields(l) for l in body if l.startswith("gpuflips")]
            if len(flips) < 6:
                continue
            plain = flips[1:6]  # steps 2-6 of the plain run
            n = int(n)
            rate = statistics.median(n / float(x["step_ms"]) / 1e3 for x in plain)
            whole = statistics.median(n / (float(x["step_ms"]) + float(x["cells_ms"])) / 1e3 for x in plain)
            fl = statistics.median(int(x["flips"]) for x in plain) / n
            rounds = statistics.median(int(x["rounds"]) for x in plain)
            m = [fields(l) for l in body if l.startswith("step_measure") and "step=0" not in l]
            if not m:
                continue
            last = m[-1]
            tets, faces = int(last["live_tets"]), int(last["faces"])
            pct = lambda a, b: f"{100 * a / b:.1f} %"
            created = int(last["created"])
            out2 = sum(int(x["new_neighbours"]) - int(x["in_2ring"]) for x in m)
            out3 = sum(int(x["new_neighbours"]) - int(x["in_3ring"]) for x in m)
            new = sum(int(x["new_neighbours"]) for x in m)
            print(f"| {kind} | {n:,} | {d[2:]} | {rate:.0f} ({whole:.0f}) | {fl:.3f} | {rounds:g} | "
                  f"{pct(int(last['screened']), tets)} | {pct(created, tets) if created >= 0 else 'overflow'} | "
                  f"{pct(int(last['faces_rewalked']), faces)} | {pct(int(last['cells_changed']), n)} | "
                  f"{out2} / {out3} of {new:,} |")
            ks = {}
            for l in body:
                if l.startswith("kernel: "):
                    label = l.split()[1] if "[" not in l else " ".join(l.split()[1:4])
                    ks[label] = float(fields(l)["ms_per_step"])
            kernels[(kind, n, d)] = ks
    return kernels


ALL_ELEMENTS = {"tet_repair::screen": "all tetrahedra", "tet_mesh::faces": "all faces", "tet_mesh::cell_sums": "all cells",
                "tet_mesh::face_numbers": "all tetrahedra", "voronoi::snap": "all cells", "exact_predicates::snap": "all cells",
                "voronoi::total_volume": "all cells", "tet_mesh::cell_face_start": "all cells",
                "tet_mesh::marked_fill": "all cells", "tet_mesh::marked_counts": "all cells",
                "tet_mesh::cell_face_counts": "all faces", "tet_mesh::cell_face_fill": "all faces",
                "tet_mesh::cell_sort": "listed cells", "tet_mesh::compact_rings": "all faces",
                "tet_mesh::ring_entries": "all faces"}
TOPOLOGY = {"tet_mesh::face_numbers", "tet_mesh::cell_face_start", "tet_mesh::marked_fill", "tet_mesh::marked_counts",
            "tet_mesh::cell_face_counts", "tet_mesh::cell_face_fill", "tet_mesh::cell_sort", "tet_mesh::compact_rings",
            "tet_mesh::ring_entries", "tet_mesh::dirty_faces", "tet_mesh::append_rings", "tet_mesh::fresh_fill",
            "tet_mesh::fresh_counts", "tet_mesh::mark_cells"}


def kernel_table(kernels):
    print("\n### Step kernels, ms per step (Kokkos Tools, fenced), 128³ jittered and clustered\n")
    keys = [k for k in kernels if k[1] in (2097152,) and k[0] in ("jittered", "clustered")]
    keys.sort()
    labels = sorted({l for k in keys for l in kernels[k] if not l.startswith(("Kokkos::View", "deep_copy"))},
                    key=lambda l: -kernels[keys[0]].get(l, 0))
    print("| kernel | runs over | " + " | ".join(f"{k[0]} {k[2][2:]}" for k in keys) + " |")
    print("|---|---|" + "---|" * len(keys))
    for l in labels:
        print(f"| `{l}` | {ALL_ELEMENTS.get(l, 'changed items')} | " +
              " | ".join(f"{kernels[k].get(l, 0):.2f}" for k in keys) + " |")
    tot = {k: sum(v for l, v in kernels[k].items() if not l.startswith(("Kokkos::View", "deep_copy"))) for k in keys}
    topo = {k: sum(v for l, v in kernels[k].items() if l in TOPOLOGY) for k in keys}
    print("| **mesh kernels, total** | | " + " | ".join(f"{tot[k]:.2f}" for k in keys) + " |")
    print("| topology bookkeeping (numbers, rings, lists) | | " + " | ".join(f"{topo[k]:.2f} ({100 * topo[k] / tot[k]:.0f} %)" for k in keys) + " |")


def ncu_rows(path):
    lines = [l for l in open(path, errors="replace") if l.startswith('"')]
    if len(lines) < 3:
        return []
    r = list(csv.reader(io.StringIO("".join(lines))))
    return [dict(zip(r[0], row)) for row in r[2:]]


def stalls(d):
    st = {k.split("stalled_")[1].split("_per")[0]: float(v) for k, v in d.items() if "issue_stalled" in k and v}
    tot = sum(v for k, v in st.items() if k != "selected") or 1
    return {k: 100 * v / tot for k, v in st.items()}


def short_name(name):
    if "Kernel<" in name:
        return "warp::Kernel" + re.search(r"Kernel<CudaSpace, (\d), (\d), (\d), (\d)>", name).group(0)[16:]
    m = re.search(r"(CellSortWarp|CellSumsWarp)", name)
    if m:
        return m.group(1)
    m = re.search(r"([\w:]+)(?:<(?:[^<>()]|<(?:[^<>()]|<[^<>()]*>)*>)*>)?\([^()]*\)::\[lambda\([^)]*\) \(instance (\d+)\)\]", name)
    return f"{m.group(1)}#{m.group(2)}" if m else name[:60]


def ncu_tables():
    print("\n### Nsight Compute: clipping kernel at 2.1M (4.2M slab), one launch\n")
    print("| input | ms | registers | stack B (cuobjdump) | local loads + stores per cell | warps active | threads per instruction | "
          "memory wait | fixed-latency wait | instruction fetch | shared-memory wait | issue |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob(f"{JOBS}/ncu_rebuild_*.csv")):
        rows = ncu_rows(f)
        if not rows:
            continue
        d = rows[0]
        kind, n = os.path.basename(f).split("_")[2:4]
        n = int(n.split("-")[0])
        s = stalls(d)
        local = (float(d["sm__sass_inst_executed_op_local_ld.sum"]) + float(d["sm__sass_inst_executed_op_local_st.sum"])) / n
        print(f"| {kind if n == 2097152 else 'slab'} | {float(d['gpu__time_duration.sum']) / 1e6:.1f} | "
              f"{d['launch__registers_per_thread']} | | {local:.0f} | {float(d['sm__warps_active.avg.pct_of_peak_sustained_active']):.0f} % | "
              f"{float(d['smsp__thread_inst_executed_per_inst_executed.ratio']):.1f} | "
              f"{s['long_scoreboard'] + s['lg_throttle']:.0f} % | {s['wait']:.0f} % | {s['no_instruction'] + s['branch_resolving']:.0f} % | "
              f"{s['short_scoreboard'] + s['mio_throttle']:.0f} % | {s['selected'] + s['not_selected']:.0f} % |")
    for f in sorted(glob.glob(f"{JOBS}/ncu_step_*.csv")):
        rows = ncu_rows(f)
        if not rows:
            continue
        print(f"\n### Nsight Compute: step kernels, {os.path.basename(f)}\n")
        agg = {}
        for d in rows:
            k = short_name(d["Kernel Name"])
            a = agg.setdefault(k, {"ms": 0, "n": 0, "rows": []})
            a["ms"] += float(d["gpu__time_duration.sum"]) / 1e6
            a["n"] += 1
            a["rows"].append(d)
        print("| kernel | launches | ms all launches | registers | warps active | threads per instruction | memory wait | "
              "fixed-latency wait | DRAM GB/s |")
        print("|---|---|---|---|---|---|---|---|---|")
        for k, a in sorted(agg.items(), key=lambda x: -x[1]["ms"])[:18]:
            w = [float(d["gpu__time_duration.sum"]) for d in a["rows"]]
            avg = lambda key: sum(float(d[key] or 0) * x for d, x in zip(a["rows"], w)) / sum(w)
            st = [stalls(d) for d in a["rows"]]
            mem = sum((s.get("long_scoreboard", 0) + s.get("lg_throttle", 0)) * x for s, x in zip(st, w)) / sum(w)
            fix = sum(s.get("wait", 0) * x for s, x in zip(st, w)) / sum(w)
            dram = sum(float(d["dram__bytes_read.sum"] or 0) + float(d["dram__bytes_write.sum"] or 0) for d in a["rows"]) / (sum(w) * 1e-9) / 1e9
            print(f"| `{k}` | {a['n']} | {a['ms']:.2f} | {a['rows'][0]['launch__registers_per_thread']} | "
                  f"{avg('sm__warps_active.avg.pct_of_peak_sustained_active'):.0f} % | "
                  f"{avg('smsp__thread_inst_executed_per_inst_executed.ratio'):.1f} | {mem:.0f} % | {fix:.0f} % | {dram:.0f} |")


if __name__ == "__main__":
    clip_tables()
    if not os.environ.get("LOGS_ONLY"):
        figures(attribution())
    kernel_table(step_tables())
    ncu_tables()
