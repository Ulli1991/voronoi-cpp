#!/usr/bin/env python3
"""Tables and figure of the library benchmark (docs/library_benchmark.md) from its raw data alone: the timing records
bench/libraries/results/*.jsonl (one JSON line per timed run, written through bench/libraries/libbench.hpp), the
oracle checks bench/libraries/results/accuracy-*.jsonl and the published numbers bench/libraries/published.csv.
Writes bench/libraries/results/summary.md and docs/figures/library_benchmark.png.
    python3 scripts/library_benchmark.py

Rows are grouped by what the library returns:
  full cells   volumes, centroids, and faces with areas and neighbours (the library's own, or "+ faces (ours)": our
               cells from its tetrahedra or neighbour lists, timed in);
  volumes      volumes (and barycentres) only: the cells are clipped but their faces are not returned;
  topology     Delaunay tetrahedra or neighbour lists only.
"""
import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import figstyle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "bench", "libraries", "results")
INPUTS = ["jittered", "bluenoise", "uniform", "clustered", "halo"]
SIZES = [2097152, 16777216]
SIZE_NAME = {2097152: "128³", 16777216: "256³", 1048576: "2²⁰", 8388608: "2²³"}
# The application inputs (docs/measurements.md, Inputs), in their own tables: the fcc solid has 4 l³ atoms, so it runs
# at l = 64 and 128 unit cells (2²⁰ and 2²³ atoms) rather than 128³ and 256³.
APPLICATION = [(2097152, "cosmicweb"), (16777216, "cosmicweb"), (2097152, "surface"), (16777216, "surface"),
               (1048576, "fccsolid"), (8388608, "fccsolid")]
# The voronoi-cpp commit of the tables (close-gaps: volumes-only output, parallel host sort, dynamic chunks); records of
# earlier commits serve only as the same-GPU references of the other libraries' jobs (same_gpu_table).
OURS = "77055a4"
OURS_APPLICATION = "631af46"  # main c42c230 + the application inputs: our rows of those inputs only


def load_records():
    """Timing records of every results file; other JSON lines (side records) are kept apart."""
    records, extra = [], []
    for path in sorted(glob.glob(os.path.join(RESULTS, "*.jsonl"))):
        if os.path.basename(path).startswith("accuracy"):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                r = json.loads(line)
                r["file"] = os.path.basename(path)
                (records if "phase" in r else extra).append(r)
    return records, extra


def key(r):
    """One table row: library, variant, device and threads."""
    return (r["library"], r["variant"], r["device"], int(r["threads"]))


def gpu(k):
    return k[2].startswith("A100")


def summarise(records):
    """Median, min and max rate per (row, phase, input, n), peak bytes per cell, hosts. Steps: frames 2 and later."""
    groups = defaultdict(list)
    for r in records:
        if r["phase"] == "step" and int(r["repeat"]) < 2:
            continue
        groups[(key(r), r["phase"], r["input"], int(r["n"]))].append(r)
    out = {}
    for g, rs in groups.items():
        rates = [float(r["Mcells_s"]) for r in rs]
        bpc = [float(r["bytes_per_cell"]) for r in rs if float(r["peak_bytes"]) >= 0]
        out[g] = dict(median=statistics.median(rates), lo=min(rates), hi=max(rates), runs=len(rates),
                      bytes_per_cell=max(bpc) if bpc else None, jobs=sorted({r["job"] for r in rs}),
                      hosts=sorted({r["host"] for r in rs}), record=rs[0])
    # A driver that times "X" and then "X + ..." in one process starts the second with the allocator's memory of the
    # first, so its host high-water mark above the resident set at start is too low (CGAL's padded Delaunay + cells:
    # 64 B/cell against 675 for the Delaunay alone). A "+" row reports at least its base row's peak.
    for g, v in out.items():
        k, phase, i, n = g
        if "+" not in k[1] or v["bytes_per_cell"] is None:
            continue
        for h, w in out.items():
            if h[1:] == (phase, i, n) and h[0][0] == k[0] and h[0][2:] == k[2:] and "+" not in h[0][1] \
                    and w["bytes_per_cell"] is not None and k[1].startswith(h[0][1].split(" (")[0].split(";")[0]):
                v["bytes_per_cell"] = max(v["bytes_per_cell"], w["bytes_per_cell"])
    return out


def group(record):
    o = record["outputs"].lower()
    if "faces" in o or "facet" in o:
        return "full"
    if "volume" in o:
        return "volumes"
    return "topology"


# The README's setting per input class for the kNN codes (their papers tune K per distribution): blue noise, perturbed
# grid (our jittered lattice) and white noise (the rest; at 256³ VoroGPU's K 190 overflows, so K 90).
README_SETTING = {
    "VoroGPU": {"bluenoise": "K 35", "jittered": "K 90", "white": "K 190", "white_256": "K 90"},
    "Liu et al.": {"bluenoise": "PRESET 2", "jittered": "PRESET 1", "white": "PRESET 0", "white_256": "PRESET 0"},
    "Basselin": {"bluenoise": "--K=20", "jittered": "--K=50", "white": "--K=70", "white_256": "--K=70"},
}


def readme_choice(lib, variant, inp, n):
    """Whether this row is the README's setting for the input (kNN codes); True for every other library."""
    if lib.startswith("Basselin"):  # its variant lists the defaults, then the setting given: the last --K counts
        variant = variant[variant.rfind("--K="):]
    for name, table in README_SETTING.items():
        if lib.startswith(name):
            cls = {"fccsolid": "jittered"}.get(inp, inp)  # a thermal crystal is a perturbed grid
            cls = cls if cls in ("bluenoise", "jittered") else ("white" if n == SIZES[0] else "white_256")
            return table[cls] in variant
    return True


def fmt(s):
    if s is None:
        return "–"
    m = s["median"]
    digits = 3 if m < 1 else (2 if m < 10 else (1 if m < 100 else 0))
    spread = "" if s["runs"] < 2 else f" ({s['lo']:.{digits}f}–{s['hi']:.{digits}f})"
    return f"{m:.{digits}f}{spread}"


def breaks(summary, k):
    """What a row does not share with the headline's rules: periodic box, double, exact predicates, device-resident
    timing, centroids. (Same hardware: see same_gpu_table.)"""
    out = []
    r = next(s["record"] for g, s in summary.items() if g[0] == k)
    if not r["periodic"].startswith("native"):
        out.append("images")
    prec = r["precision"].lower()
    if "float32" in prec or prec.startswith("float"):
        out.append("float")
    if "no exact" in prec or not any(w in prec for w in ("exact", "shewchuk", "sos")):
        out.append("no exact predicates")
    if gpu(k):
        if "host memory" in r["timed"]:
            out.append("host↔device copies timed")
    if group(r) == "full" and "centroid" not in r["outputs"].lower():
        out.append("no centroids")
    return "; ".join(out)


def rows_of(summary, phase, which):
    return sorted({g[0] for g, s in summary.items() if g[1] == phase and group(s["record"]) == which},
                  key=lambda k: (not gpu(k), -k[3], k[0], k[1]))


def rate_table(summary, phase, which, readme_only=False, columns=None):
    rows = [k for k in rows_of(summary, phase, which)
            if any(g[0] == k and (g[3], g[2]) in (columns or [(n, i) for n in SIZES for i in INPUTS]) for g in summary)]
    cols = [(n, i) for n, i in (columns or [(n, i) for n in SIZES for i in INPUTS])
            if any(g[1] == phase and g[2] == i and g[3] == n and group(s["record"]) == which for g, s in summary.items())]
    if not rows:
        return "(none)"
    lines = ["| library | variant | device | threads | " + " | ".join(f"{i} {SIZE_NAME[n]}" for n, i in cols) +
             " | B/cell (128³ max) | breaks |", "|" + "---|" * (6 + len(cols))]
    for k in rows:
        cells = []
        for n, i in cols:
            s = summary.get((k, phase, i, n))
            if s and readme_only and not readme_choice(k[0], k[1], i, n):
                s = None
            cells.append(fmt(s))
        small = [(n, i) for n, i in cols if n <= SIZES[0]]
        b = [summary[(k, phase, i, n)]["bytes_per_cell"] for n, i in small
             if (k, phase, i, n) in summary and summary[(k, phase, i, n)]["bytes_per_cell"] is not None]
        bpc = f"{max(b):.0f}" if b else ""
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | " + " | ".join(cells) + f" | {bpc} | {breaks(summary, k)} |")
    return "\n".join(lines)


def order(inp):
    names = INPUTS + [i for _, i in APPLICATION]
    return names.index(inp) if inp in names else len(names)


def same_gpu_table(records):
    """Per A100 job (one GPU: run_chain_a100.sh), each library's median against voronoi-cpp's warp rebuild in the same
    job on the same input: the ratio is free of node-to-node clock differences."""
    per = defaultdict(list)
    for r in records:
        if r["device"].startswith("A100") and r["phase"] == "build":
            per[(r["file"], key(r), r["input"], int(r["n"]))].append(float(r["Mcells_s"]))
    med = {g: statistics.median(v) for g, v in per.items()}
    lines = ["| job file | library | variant | input | n | M cells/s | voronoi-cpp warp, same job (commit) | ratio |",
             "|---|---|---|---|---|---|---|---|"]
    version = {r["file"]: r["version"] for r in records if r["library"] == "voronoi-cpp"}
    for (f, k, i, n), m in sorted(med.items(), key=lambda e: (e[0][0], e[0][1], order(e[0][2]), e[0][3])):
        if k[0] == "voronoi-cpp" or group({"outputs": next(r["outputs"] for r in records if key(r) == k)}) == "topology":
            continue
        ref = [v for (f2, k2, i2, n2), v in med.items()
               if f2 == f and k2[0] == "voronoi-cpp" and k2[1].startswith("warp clipping rebuild (cell") and i2 == i
               and n2 == n]
        if not ref:
            continue
        lines.append(f"| {f} | {k[0]} | {k[1]} | {i} | {n} | {m:.3g} | {ref[0]:.3g} ({version.get(f, '')}) | "
                     f"{ref[0] / m:.1f} |")
    return "\n".join(lines)


def provenance_table(summary):
    seen = {}
    for g, s in summary.items():
        r = s["record"]
        e = seen.setdefault(g[0], dict(version=r["version"], outputs=r["outputs"], periodic=r["periodic"],
                                       precision=r["precision"], timed=r["timed"], note=r.get("note", ""),
                                       jobs=set(), hosts=set()))
        e["jobs"].update(s["jobs"])
        e["hosts"].update(s["hosts"])
    lines = ["| library | variant | device | threads | version | outputs | periodic | precision | timed | note | jobs | hosts |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(seen.items()):
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {v['version']} | {v['outputs']} | {v['periodic']} | "
                     f"{v['precision']} | {v['timed']} | {v['note']} | {', '.join(sorted(v['jobs']))} | "
                     f"{', '.join(sorted(v['hosts']))} |")
    return "\n".join(lines)


def accuracy_rows():
    rows = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "accuracy-*.jsonl"))):
        with open(path) as f:
            for line in f:
                if line.startswith("{"):
                    a = json.loads(line.replace(":inf", ":Infinity").replace(":-nan", ":NaN").replace(":nan", ":NaN"))
                    for k in ("sum_minus_box", "median_rel_err", "max_rel_err"):
                        a[k] = float("inf") if a[k] is None else a[k]
                    a["label"] = os.path.basename(a["volumes"]).rsplit(".", 1)[0]
                    rows[a["label"]] = a  # the latest check of a file wins
    return rows


def accuracy_table(acc):
    lines = ["| volumes (library-device_input_n) | sample | missing or ≤ 0 (of n) | sum − 1 | median rel. error | "
             "max rel. error | sampled cells > 1e-12 | > 1e-6 | > 1e-2 |", "|---|---|---|---|---|---|---|---|---|"]
    for label in sorted(acc):
        a = acc[label]
        lines.append(f"| {label} | {a['sample']} | {a['missing_or_nonpositive']} | {a['sum_minus_box']:.1e} | "
                     f"{a['median_rel_err']:.1e} | {a['max_rel_err']:.1e} | {a['cells_rel_err_above_1e-12']} | "
                     f"{a['above_1e-6']} | {a['above_1e-2']} |")
    return "\n".join(lines)


def published():
    path = os.path.join(ROOT, "bench", "libraries", "published.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def published_table(rows):
    lines = ["| code | reference | hardware | n | distribution | precision | periodic | output | rate (M/s) | kind | where |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in rows:
        lines.append(f"| {p['code']} | {p['citation_key']} | {p['hardware']} | {p['n']} | {p['distribution']} | "
                     f"{p['precision']} | {p['periodic']} | {p['output']} | {p['rate_Mcells_s']} | {p['rate_kind']} | "
                     f"{p['source_location']} |")
    return "\n".join(lines)


# ---- figure ----------------------------------------------------------------------------------------------------
# (library prefix, variant prefix, device prefix, threads, label). Volumes-only codes take their README setting per
# input (README_SETTING); their "variant prefix" matches all their settings.
FIGURE_GPU_FULL = [
    ("voronoi-cpp", "warp clipping rebuild (cell", "A100", 1, "voronoi-cpp, warp rebuild"),
    ("voronoi-cpp", "warp clipping rebuild, host to host (copy", "A100", 1, "voronoi-cpp, warp rebuild, host to host"),
    ("voronoi-cpp", "exact start", "A100", 1, "voronoi-cpp, exact start"),
    ("peclet-voro", "cold build", "A100", 1, "peclet-voro (no centroids)"),
    ("gDel3D", "Delaunay + faces", "A100", 1, "gDel3D + faces (ours)"),
    ("Paragram", "voronoi_diagram + faces", "A100", 1, "Paragram (float) + faces (ours)"),
    ("votess", "tesellate + faces", "A100", 1, "votess + faces (ours)"),
]
FIGURE_GPU_VOLUMES = [
    ("voronoi-cpp", "volumes and centroids only, host to host$", "A100", 1,
     "voronoi-cpp, volumes only, host to host"),
    ("voronoi-cpp", "volumes and centroids only (", "A100", 1, "voronoi-cpp, volumes only, device"),
    ("Basselin", "", "A100", 1, "Basselin et al. 2021"),
    ("Liu et al.", "tet-centric", "A100", 1, "Liu et al. 2022 (float)"),
    ("VoroGPU", "", "A100", 1, "Ray et al. 2018 (float)"),
]
FIGURE_CPU_FULL = [
    ("voronoi-cpp", "warp clipping rebuild (cell", "2x Xeon", 72, "voronoi-cpp, warp rebuild"),
    ("voro++", "multithreaded", "2x Xeon", 72, "voro++ multithreaded"),
    ("peclet-voro", "cold build", "2x Xeon", 72, "peclet-voro (no centroids)"),
    ("CGAL Delaunay_triangulation_3", "Delaunay + faces", "2x Xeon", 72, "CGAL parallel + faces (ours)"),
    ("geogram", "PeriodicDelaunay3d + Voronoi cells", "2x Xeon", 72, "geogram periodic + cells"),
    ("geogram", "ParallelDelaunay3d PDEL on padded points + faces", "2x Xeon", 72, "geogram PDEL + faces (ours)"),
    ("votess", "tesellate + faces", "2x Xeon", 72, "votess + faces (ours)"),
    ("voro++", "container_periodic", "2x Xeon", 1, "voro++, 1 core"),
    ("CGAL Periodic_3_Delaunay_triangulation_3", "Delaunay + faces", "2x Xeon", 1, "CGAL periodic + faces, 1 core"),
    ("TetGen", "tetrahedralize(Qz) of the padded points; Delaunay + faces", "2x Xeon", 1, "TetGen + faces, 1 core"),
    ("Qhull", "qh_new_qhull(d Qbb Qz Qt) of the padded points; Delaunay + faces", "2x Xeon", 1, "Qhull + faces, 1 core"),
]
# Published numbers: (code prefix in published.csv, n, distribution fragment, panel).
FIGURE_PUBLISHED = [
    ("peclet-voro (cold", "2000000", "", "gpu_full"),
    ("Basselin et al. 2021 (", "10000000", "blue noise", "gpu_volumes"),
    ("Basselin et al. 2021 (", "10000000", "white noise", "gpu_volumes"),
    ("Ray et al. 2018 (meshl", "10000000", "blue noise", "gpu_volumes"),
    ("Paragram", "2000000", "white noise", "gpu_volumes"),
    ("votess (GPU", "1000000", "", "gpu_volumes"),
    ("Voro++ (multithreaded", "100000000", "", "cpu"),
]
FIGURE_STEP = [
    ("voronoi-cpp", "moving-mesh step", "A100", 1, "voronoi-cpp, A100"),
    ("peclet-voro", "moving step", "A100", 1, "peclet-voro, A100"),
    ("voronoi-cpp", "moving-mesh step", "2x Xeon", 72, "voronoi-cpp, 72 cores"),
    ("peclet-voro", "moving step", "2x Xeon", 72, "peclet-voro, 72 cores"),
    ("CGAL Periodic_3", "moving mesh: full rebuild", "2x Xeon", 1, "CGAL periodic, rebuild, 1 core"),
    ("CGAL Periodic_3", "moving mesh: remove + insert", "2x Xeon", 1, "CGAL periodic, move, 1 core"),
]


def curve(summary, lib, variant, device, threads, n, phase="build"):
    """{input: summary} of one figure row; volumes-only codes take their README setting per input."""
    got = {}
    for g, s in summary.items():
        k = g[0]
        same = k[1] == variant[:-1] if variant.endswith("$") else k[1].startswith(variant)  # "$": whole variant
        if g[1] == phase and g[3] == n and k[0].startswith(lib) and same and \
                k[2].startswith(device) and k[3] == threads and readme_choice(k[0], k[1], g[2], n):
            got[g[2]] = s
    return got


def short_hardware(h):
    for a, b in (("NVIDIA ", ""), ("GeForce ", ""), ("Tesla ", ""), ("2x Intel Xeon ", "2x Xeon "), ("-40GB", "")):
        h = h.replace(a, b)
    return h.split(" 1.7 GHz")[0].split(" 2.40 GHz")[0]


def figure(summary, pub, path):
    figstyle.apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(4 * figstyle.FIGSIZE[0], 1.3 * figstyle.FIGSIZE[1]),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 0.6]))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = "osD^vP*Xh<>p"
    open_markers = "o^sDvp<>h"
    xs = list(range(len(INPUTS)))
    panels = [(axes[0], FIGURE_GPU_FULL, "gpu_full", "one A100: full cells (faces, areas, neighbours)"),
              (axes[1], FIGURE_GPU_VOLUMES, "gpu_volumes", "one A100: volumes and barycentres only"),
              (axes[2], FIGURE_CPU_FULL, "cpu", "one CPU node: full cells")]
    for ax, rows, tag, title in panels:
        figstyle.decorate(ax)
        j = 0
        for lib, variant, device, threads, label in rows:
            got = curve(summary, lib, variant, device, threads, SIZES[0])
            pts = [(x, got[i]) for x, i in zip(xs, INPUTS) if i in got]
            if not pts:
                continue
            c, m = colours[j % len(colours)], markers[j % len(markers)]
            j += 1
            style = "-" if device == "A100" else ("--" if threads > 1 else ":")
            ax.errorbar([x for x, _ in pts], [s["median"] for _, s in pts],
                        yerr=[[s["median"] - s["lo"] for _, s in pts], [s["hi"] - s["median"] for _, s in pts]],
                        color=c, marker=m, linewidth=2.2, linestyle=style, markersize=7, label=label)
        x0 = len(INPUTS)
        k = 0
        shown = [(code, n, dist) for code, n, dist, panel in FIGURE_PUBLISHED if panel == tag]
        for code, n, dist in shown:
            for p in pub:
                if p["code"].startswith(code) and p["n"] == n and p["distribution"].startswith(dist):
                    ax.plot([x0 + 0.12 * (k - (len(shown) - 1) / 2)], [float(p["rate_Mcells_s"])],
                            marker=open_markers[k % len(open_markers)], markerfacecolor="none", markeredgecolor="0.25",
                            markeredgewidth=1.6, markersize=9, linestyle="none",
                            label=f"{p['code'].split(' (')[0]}, {short_hardware(p['hardware'])}, "
                                  f"{p['distribution'].split(' (')[0]}, n {float(n):.0e} (published)")
                    k += 1
                    break
        ax.set_xticks(xs + [x0])
        ax.set_xticklabels(INPUTS + ["published"], rotation=20)
        ax.set_yscale("log")
        ax.set_ylim(0.005, 3e4)
        ax.set_xlabel(f"{title}\n{SIZES[0]:,} cells")
        figstyle.legend(ax, loc="upper right", fontsize=8, ncol=1)
    axes[0].set_ylabel("M cells/s, build")
    ax = axes[3]
    figstyle.decorate(ax)
    for k, (lib, variant, device, threads, label) in enumerate(FIGURE_STEP):
        got = curve(summary, lib, variant, device, threads, SIZES[0], phase="step").get("jittered")
        if not got:
            continue
        ax.errorbar([k], [got["median"]], yerr=[[got["median"] - got["lo"]], [got["hi"] - got["median"]]],
                    color=colours[k % len(colours)], marker=markers[k % len(markers)], markersize=8, linestyle="none",
                    linewidth=2.2, label=label)
    ax.set_yscale("log")
    ax.set_ylim(0.001, 3e4)
    ax.set_xlim(-0.7, len(FIGURE_STEP) - 0.3)
    ax.set_xticks([])
    ax.set_ylabel("M cells/s, moving step with full cells")
    ax.set_xlabel("0.03 spacings per step\njittered, 2,097,152 cells")
    figstyle.legend(ax, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


def main():
    records, extra = load_records()
    app = {i for _, i in APPLICATION}
    # Our application rows come from our own jobs; the same-job references in the chains count only in "Same GPU".
    summary = summarise([r for r in records if r["library"] != "voronoi-cpp"
                         or (r["version"] == OURS_APPLICATION and r["file"].startswith("voronoi-cpp-a100")
                             if r["input"] in app else r["version"] == OURS)])
    pub = published()
    acc = accuracy_rows()
    text = ["# Library benchmark: tables generated by scripts/library_benchmark.py (do not edit)", "",
            "Median of 3 timed runs (min–max in brackets), M cells/s; steps: median (range) of steps 2 to 10. "
            "\"breaks\": what the row does not share with the headline rules (periodic box, double, exact predicates, "
            "device-resident timing, centroids). Same hardware: every A100 job ran voronoi-cpp's warp rebuild as a "
            "reference on its GPU (section \"Same GPU\"). Volumes-only codes: the README's setting per input, then "
            "every setting.", "",
            "## Full cells (volumes, centroids, faces with areas and neighbours): the headline", "",
            rate_table(summary, "build", "full"), "",
            "## Volumes and barycentres only: README setting per input", "",
            rate_table(summary, "build", "volumes", readme_only=True), "",
            "## Volumes and barycentres only: every setting", "",
            rate_table(summary, "build", "volumes"), "",
            "## Topology only (Delaunay tetrahedra or neighbour lists)", "",
            rate_table(summary, "build", "topology"), "",
            "## Moving-mesh step", "", rate_table(summary, "step", "full"), "",
            "## Application inputs: cosmic web, surface, fcc solid (full cells; volumes only, README setting; "
            "topology; moving-mesh step)", "",
            rate_table(summary, "build", "full", columns=APPLICATION), "",
            rate_table(summary, "build", "volumes", readme_only=True, columns=APPLICATION), "",
            rate_table(summary, "build", "topology", columns=APPLICATION), "",
            rate_table(summary, "step", "full", columns=APPLICATION), "",
            "## Same GPU: each A100 library against voronoi-cpp's warp rebuild in the same job", "",
            same_gpu_table(records), "",
            "## Setup per row", "", provenance_table(summary), "",
            "## Accuracy against the oracle", "", accuracy_table(acc), "",
            "## Published numbers (not measured by us)", "", published_table(pub), "",
            "## Side records", ""]
    text += [f"- `{json.dumps(e)}`" for e in extra]
    with open(os.path.join(RESULTS, "summary.md"), "w") as f:
        f.write("\n".join(text) + "\n")
    figure(summary, pub, os.path.join(ROOT, "docs", "figures", "library_benchmark.png"))


if __name__ == "__main__":
    main()
