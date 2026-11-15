#!/usr/bin/env python3
"""GPU busy and idle time per NVTX phase of voronoi_bench runs traced by scripts/overlap_a100.sh (or any nsys sqlite).

  python3 scripts/overlap_tables.py <job dir> [--kernels]       one row per run and phase

Busy: the union of kernels, copies and memsets on the GPU. Idle: the rest of the phase. Each idle interval is split by
what the issuing host thread did meanwhile: in a launch call, in a synchronising call, in a copy call, in an allocation
or free, or outside the CUDA API (host code). A read-back is a device-to-host copy; a round trip is the idle interval
that starts at its end. Times in ms."""
import os
import re
import sqlite3
import statistics
import sys
from collections import defaultdict

FUNC = re.compile(r"\b((?:shell_voronoi|tet_start|tet_mesh|tet_repair|tet_flips|tet_cells|mesh_order|neighbor_search|voronoi|"
                  r"warp|exact_predicates)(?:::[A-Za-z_][A-Za-z_0-9]*)+)")


def function(name):
    """The library function a Kokkos kernel's demangled name comes from (lambda scope or functor)."""
    found = [m for m in FUNC.findall(name) if "Kokkos" not in m]
    if not found:
        return name[:60]
    f = found[0]
    if "construct_pass" in name:  # which capacity the per-thread clipping ran with
        m = re.search(r"(RoundedCell|ExactCell)<(\d+), (\d+)>", name)
        if m:
            f += f"<{m.group(1)},{m.group(2)}>"
    return f


def category(api):
    if "Launch" in api:
        return "launch"
    if "Synchronize" in api or "Query" in api:
        return "sync"
    if "Memcpy" in api or "Memset" in api:
        return "copy"
    if "Malloc" in api or "Free" in api or "HostAlloc" in api or "HostRegister" in api:
        return "alloc"
    return "other"


def union(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def overlap(intervals, a, b):
    """Total length of the (sorted, disjoint) intervals inside [a, b]."""
    return sum(max(0, min(y, b) - max(x, a)) for x, y in intervals)


class Trace:
    def __init__(self, path):
        db = sqlite3.connect(path)
        s = dict(db.execute("select id, value from StringIds"))
        self.kernels = [(a, b, function(s[n])) for a, b, n in db.execute(
            "select start, end, demangledName from CUPTI_ACTIVITY_KIND_KERNEL")]
        tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        self.copies = list(db.execute("select start, end, copyKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY")) \
            if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables else []
        self.sets = list(db.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMSET")) \
            if "CUPTI_ACTIVITY_KIND_MEMSET" in tables else []
        api = list(db.execute("select start, end, nameId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"))
        count = defaultdict(int)
        for r in api:
            count[r[3]] += 1
        main = max(count, key=count.get)
        self.api = sorted((a, b, s[n]) for a, b, n, t in api if t == main)
        self.ranges = []
        if "NVTX_EVENTS" in tables:
            for a, b, text, tid in db.execute("select start, end, text, textId from NVTX_EVENTS where end is not null"):
                self.ranges.append((a, b, text if text is not None else s.get(tid, "")))
        self.busy = union([(a, b) for a, b, _ in self.kernels] + [(a, b) for a, b, _, _ in self.copies] + self.sets)
        self.by_cat = defaultdict(list)
        for a, b, name in self.api:
            self.by_cat[category(name)].append((a, b))
        self.dtoh_ends = sorted(b for a, b, k, _ in self.copies if k == 2)

    def phase(self, a, b):
        """Busy, idle and the split of idle and busy time by host activity in [a, b], ns."""
        busy = [(max(x, a), min(y, b)) for x, y in self.busy if y > a and x < b]
        idle, at = [], a
        for x, y in busy:
            if x > at:
                idle.append((at, x))
            at = max(at, y)
        if at < b:
            idle.append((at, b))
        r = {"wall": b - a, "busy": sum(y - x for x, y in busy), "idle": sum(y - x for x, y in idle)}
        cats = {c: union(v) for c, v in self.by_cat.items()}
        for c, iv in cats.items():
            r["idle_" + c] = sum(overlap(iv, x, y) for x, y in idle)
            r["busy_" + c] = sum(overlap(iv, x, y) for x, y in busy)
        in_api = union([iv for v in cats.values() for iv in v])
        r["idle_host"] = r["idle"] - sum(overlap(in_api, x, y) for x, y in idle)
        r["busy_host"] = r["busy"] - sum(overlap(in_api, x, y) for x, y in busy)
        # Read-back round trips: idle intervals starting where a device-to-host copy ended.
        ends = set(e for e in self.dtoh_ends if a <= e <= b)
        trips = [(x, y) for x, y in idle if x in ends]
        r["readbacks"] = len(ends)
        r["trip"] = sum(y - x for x, y in trips)
        r["syncs"] = sum(1 for x, y, n in self.api if a <= x < b and category(n) == "sync")
        r["launches"] = sum(1 for x, y, n in self.api if a <= x < b and category(n) == "launch")
        r["gaps"] = sorted(((y - x, x, y) for x, y in idle), reverse=True)
        k = defaultdict(float)
        for x, y, f in self.kernels:
            if y > a and x < b:
                k[f] += min(y, b) - max(x, a)
        r["kernels"] = dict(k)
        return r

    def next_kernel(self, t):
        for x, y, f in self.kernels:
            if x >= t:
                return f
        return "-"

    def prev_api(self, t):
        best = None
        for x, y, n in self.api:
            if y <= t:
                best = n
            else:
                break
        return best


def ms(ns):
    return ns / 1e6


def row(name, r, extra=""):
    print(f"{name:34s} wall={ms(r['wall']):9.2f} busy={ms(r['busy']):9.2f} idle={ms(r['idle']):8.2f} "
          f"({100 * r['idle'] / max(r['wall'], 1):4.1f}%) idle_host={ms(r['idle_host']):7.2f} "
          f"idle_launch={ms(r.get('idle_launch', 0)):6.2f} idle_sync={ms(r.get('idle_sync', 0)):6.2f} "
          f"idle_copy={ms(r.get('idle_copy', 0)):6.2f} idle_alloc={ms(r.get('idle_alloc', 0)):6.2f} "
          f"readbacks={r['readbacks']} trips={ms(r['trip']):6.2f} syncs={r['syncs']} launches={r['launches']} "
          f"busy_sync={ms(r.get('busy_sync', 0)):8.2f} busy_host={ms(r['busy_host']):7.2f}{extra}")


def merged(rs):
    """Median of each number over runs of the same phase."""
    out = {}
    for key in rs[0]:
        if key in ("gaps", "kernels"):
            continue
        out[key] = statistics.median(r.get(key, 0) for r in rs)
    out["gaps"] = rs[len(rs) // 2]["gaps"]
    ks = defaultdict(list)
    for r in rs:
        for f, v in r["kernels"].items():
            ks[f].append(v)
    out["kernels"] = {f: statistics.median(v + [0] * (len(rs) - len(v))) for f, v in ks.items()}
    return out


def report(path, kernels):
    t = Trace(path)
    tag = os.path.basename(path).replace(".sqlite", "")
    groups = defaultdict(list)
    for a, b, text in sorted(t.ranges):
        m = re.match(r"s(\d+)_(update|cells)", text)
        if m:
            if int(m.group(1)) >= 2:
                groups["step_" + m.group(2)].append((a, b))
        else:
            groups[text].append((a, b))
    # The whole step: update + cells of the same step.
    steps = defaultdict(list)
    for a, b, text in t.ranges:
        m = re.match(r"s(\d+)_(update|cells)", text)
        if m and int(m.group(1)) >= 2:
            steps[int(m.group(1))].append((a, b))
    if steps:
        groups["step_all"] = [(min(x for x, _ in v), max(y for _, y in v)) for v in steps.values()]
    for name in ("list", "clip", "start", "start_cells", "step_update", "step_cells", "step_all"):
        if name not in groups:
            continue
        rs = [t.phase(a, b) for a, b in groups[name]]
        r = merged(rs)
        row(f"{tag}:{name}", r, f" runs={len(rs)}")
        if name in ("clip", "start", "step_update", "step_all"):
            g = r["gaps"][:4]
            print("   largest gaps: " + "; ".join(
                f"{ms(d):.3f} ms before {t.next_kernel(y)} after {t.prev_api(x)}" for d, x, y in g))
        if kernels or name in ("clip", "start"):
            top = sorted(r["kernels"].items(), key=lambda kv: -kv[1])[:12 if kernels else 8]
            print("   kernels: " + "; ".join(f"{f} {ms(v):.3f}" for f, v in top))
        if name == "start":
            # When the cells for exact clipping are known: end of the first tet_start::check, from the range start.
            a, b = groups[name][len(groups[name]) // 2]
            checks = [(x, y) for x, y, f in t.kernels if a <= x < b and f.startswith("tet_start::check")]
            warp = [(x, y) for x, y, f in t.kernels if a <= x < b and f.startswith("shell_voronoi::warp")]
            exact = [(x, y) for x, y, f in t.kernels if a <= x < b and "ExactCell" in f]
            if warp:
                print(f"   timeline: warp kernel {ms(warp[0][0] - a):.2f}-{ms(warp[0][1] - a):.2f} ms", end="")
            if checks:
                print(f"; first check ends {ms(checks[0][1] - a):.2f} ms", end="")
            if exact:
                print(f"; exact clipping {ms(exact[0][0] - a):.2f}-{ms(exact[-1][1] - a):.2f} ms "
                      f"({ms(sum(y - x for x, y in exact)):.2f} ms in {len(exact)} kernels)", end="")
            print(f"; range {ms(b - a):.2f} ms")
    # Copy bandwidth (PCIe) of the large host-device copies.
    big = [(b - a, by, k) for a, b, k, by in t.copies if by >= 1 << 24]
    for kind, label in ((1, "H2D"), (2, "D2H")):
        sel = [(d, by) for d, by, k in big if k == kind]
        if sel:
            print(f"   pcie {label}: {len(sel)} copies of >=16 MB at {sum(by for _, by in sel) / sum(d for d, _ in sel):.2f} GB/s")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for d in args:
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else [d]:
            p = os.path.join(d, f) if os.path.isdir(d) else f
            if p.endswith(".sqlite"):
                report(p, "--kernels" in sys.argv)
