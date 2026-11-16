#!/usr/bin/env python3
"""GPU busy and idle time per NVTX phase of voronoi_bench runs traced by scripts/overlap_a100.sh (or any nsys sqlite).

  python3 scripts/overlap_tables.py <job dir> [--kernels]       one row per run and phase
  python3 scripts/overlap_tables.py --volumes <job dir>          library_bench warp-host-vc: one row per input, per build
  python3 scripts/overlap_tables.py --markdown <job dir>...      the tables of docs/measurements.md

Busy: the union of kernels, copies and memsets on the GPU. Idle: the rest of the phase. Each idle interval is split by
what the issuing host thread did meanwhile: in a launch call, in a synchronising call, in a copy call, in an allocation
or free, or outside the CUDA API (host code). A read-back is a device-to-host copy; a round trip is the idle interval
that starts at its end. Times in ms."""
import bisect
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
    if "ViewCopy" in name or "ViewFill" in name:
        return "Kokkos::deep_copy"
    if "::[lambda" in name:  # a lambda: the function it is defined in
        head = name[:name.index("::[lambda")]
        head = head[:head.index("(")] if "(" in head else head
        while head.endswith(">"):  # drop the function's template arguments
            depth, i = 0, len(head) - 1
            while i >= 0:
                depth += {">": 1, "<": -1}.get(head[i], 0)
                if depth == 0:
                    break
                i -= 1
            head = head[:i]
        f = re.split(r"[ <,]", head)[-1] or None
        if f:
            m = re.search(r"(RoundedCell|ExactCell)<\(int\)(\d+), \(int\)(\d+)>", name)
            return f + (f"<{m.group(1)},{m.group(2)}>" if m and "construct_pass" in f else "")
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


class Cover:
    """Sorted disjoint intervals with prefix sums: covered length inside [a, b] in O(log n)."""

    def __init__(self, intervals):
        self.iv = union(intervals)
        self.starts = [a for a, _ in self.iv]
        self.sum = [0]
        for a, b in self.iv:
            self.sum.append(self.sum[-1] + b - a)

    def upto(self, t):
        i = bisect.bisect_right(self.starts, t)
        if i == 0:
            return 0
        a, b = self.iv[i - 1]
        return self.sum[i - 1] + min(b, t) - a

    def inside(self, a, b):
        return self.upto(b) - self.upto(a) if b > a else 0


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
        self.kernels.sort()
        self.kstarts = [a for a, _, _ in self.kernels]
        self.busy = union([(a, b) for a, b, _ in self.kernels] + [(a, b) for a, b, _, _ in self.copies] + self.sets)
        self.bstarts = [a for a, _ in self.busy]
        by_cat = defaultdict(list)
        for a, b, name in self.api:
            by_cat[category(name)].append((a, b))
        self.cats = {c: Cover(v) for c, v in by_cat.items()}
        self.in_api = Cover([(a, b) for a, b, _ in self.api])
        self.astarts = [a for a, _, _ in self.api]
        self.aends = sorted((b, n) for a, b, n in self.api)
        self.dtoh_ends = sorted(b for a, b, k, _ in self.copies if k == 2)

    def phase(self, a, b):
        """Busy, idle and the split of idle and busy time by host activity in [a, b], ns."""
        lo = max(0, bisect.bisect_right(self.bstarts, a) - 1)
        hi = bisect.bisect_left(self.bstarts, b)
        busy = [(max(x, a), min(y, b)) for x, y in self.busy[lo:hi] if y > a and x < b]
        idle, at = [], a
        for x, y in busy:
            if x > at:
                idle.append((at, x))
            at = max(at, y)
        if at < b:
            idle.append((at, b))
        r = {"wall": b - a, "busy": sum(y - x for x, y in busy), "idle": sum(y - x for x, y in idle)}
        for c, cov in self.cats.items():
            r["idle_" + c] = sum(cov.inside(x, y) for x, y in idle)
            r["busy_" + c] = sum(cov.inside(x, y) for x, y in busy)
        r["idle_host"] = r["idle"] - sum(self.in_api.inside(x, y) for x, y in idle)
        r["busy_host"] = r["busy"] - sum(self.in_api.inside(x, y) for x, y in busy)
        # Read-back round trips: idle intervals starting where a device-to-host copy ended.
        ends = set(self.dtoh_ends[bisect.bisect_left(self.dtoh_ends, a):bisect.bisect_right(self.dtoh_ends, b)])
        trips = [(x, y) for x, y in idle if x in ends]
        r["readbacks"] = len(ends)
        r["trip"] = sum(y - x for x, y in trips)
        calls = self.api[bisect.bisect_left(self.astarts, a):bisect.bisect_left(self.astarts, b)]
        r["syncs"] = sum(1 for x, y, n in calls if category(n) == "sync")
        r["launches"] = sum(1 for x, y, n in calls if category(n) == "launch")
        r["gaps"] = sorted(((y - x, x, y) for x, y in idle), reverse=True)
        k = defaultdict(float)
        for x, y, f in self.kernels[max(0, bisect.bisect_left(self.kstarts, a) - 1):bisect.bisect_left(self.kstarts, b)]:
            if y > a and x < b:
                k[f] += min(y, b) - max(x, a)
        r["kernels"] = dict(k)
        return r

    def next_kernel(self, t):
        i = bisect.bisect_left(self.kstarts, t)
        return self.kernels[i][2] if i < len(self.kernels) else "-"

    def prev_api(self, t):
        i = bisect.bisect_right(self.aends, (t, "\uffff"))
        return self.aends[i - 1][1] if i > 0 else None


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


def volumes(path):
    """library_bench --path=warp-host-vc: each build runs from the copy of the points to the device (the largest
    host-to-device copies) to the end of the second device-to-host copy after it (volumes, centroids)."""
    t = Trace(path)
    h2d = [(a, b, by) for a, b, k, by in t.copies if k == 1]
    if not h2d:
        return
    big = max(by for _, _, by in h2d)
    starts = sorted(a for a, b, by in h2d if by == big)
    rs = []
    for a in starts[1:]:  # the first build is the warm-up
        outs = sorted(b for x, b, k, by in t.copies if k == 2 and x > a and by >= big // 3)
        if len(outs) >= 2:
            r = t.phase(a, outs[1])
            r["h2d"] = sum(min(y, outs[1]) - x for x, y, k, by in t.copies if k == 1 and a <= x < outs[1])
            r["d2h"] = sum(y - x for x, y, k, by in t.copies if k == 2 and a <= x < outs[1])
            r["kern"] = sum(min(y, outs[1]) - x for x, y, f in t.kernels if a <= x < outs[1])
            rs.append(r)
    if not rs:
        return
    r = merged(rs)
    row(f"{os.path.basename(path).replace('.sqlite', '')}:build", r,
        f" h2d={ms(r['h2d']):.2f} d2h={ms(r['d2h']):.2f} kernels={ms(r['kern']):.2f} runs={len(rs)}")
    print("   largest gaps: " + "; ".join(
        f"{ms(d):.3f} ms before {t.next_kernel(y)} after {t.prev_api(x)}" for d, x, y in r["gaps"][:4]))
    sel = [(b - a, by) for a, b, by in h2d if by == big]
    print(f"   points copy: {big / 1e6:.1f} MB at {sum(by for _, by in sel) / sum(d for d, _ in sel):.2f} GB/s")


def phases(path):
    """Median phase results of one bench trace, by phase name."""
    t = Trace(path)
    groups = defaultdict(list)
    steps = defaultdict(list)
    for a, b, text in t.ranges:
        m = re.match(r"s(\d+)_(update|cells)", text)
        if m:
            if int(m.group(1)) >= 2:
                groups["step_" + m.group(2)].append((a, b))
                steps[int(m.group(1))].append((a, b))
        else:
            groups[text].append((a, b))
    if steps:
        groups["step_all"] = [(min(x for x, _ in v), max(y for _, y in v)) for v in steps.values()]
    return {k: merged([t.phase(a, b) for a, b in v]) for k, v in groups.items()}


def log_value(path, key, line_start):
    if not os.path.exists(path):
        return None
    for line in open(path):
        if line.startswith(line_start):
            m = re.search(rf"\b{key}=([-0-9.e]+)", line)
            if m:
                return float(m.group(1))
    return None


def markdown(dirs):
    files = {}
    for d in dirs:
        for f in os.listdir(d):
            files[f] = os.path.join(d, f)
    kinds = ["jittered", "uniform", "clustered", "halo", "cosmicweb", "surface"]
    tail = lambda r: sum(v for f, v in r["kernels"].items() if "construct_pass" in f)
    print("Rebuild (warp clipping, median of 3):\n")
    print("| input | N | clipping ms | GPU idle ms (%) | tail cells | tail pass ms (% of clipping) |")
    print("|---|---|---|---|---|---|")
    for side in (128, 256):
        for k in kinds:
            f = files.get(f"rebuild_{k}_{side}.sqlite")
            if not f:
                continue
            r = phases(f)["clip"]
            m = files.get(f"measure_{k}_{side}.log")
            cells = "-"
            if m:
                for line in open(m):
                    if "warp_status ok=" in line:
                        v = dict(x.split("=") for x in line.split()[2:])
                        cells = str(sum(int(v[x]) for x in v if x != "ok"))
            print(f"| {k} | {side}³ | {ms(r['wall']):.1f} | {ms(r['idle']):.2f} ({100 * r['idle'] / r['wall']:.1f} %) | {cells} | "
                  f"{ms(tail(r)):.2f} ({100 * tail(r) / r['wall']:.1f} %) |")
    print("\nExact start (Mesh::update from scratch, one run):\n")
    print("| input | N | start ms | GPU idle ms (%) | of which allocation | read-back trips ms | tail pass ms | "
          "exact re-clipped cells | warp kernel ends, first check ends (ms) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for side in (128, 256):
        for k in kinds:
            f = files.get(f"step_{k}_{side}_0.03.sqlite")
            if not f:
                continue
            t = Trace(f)
            rng = [(a, b) for a, b, x in t.ranges if x == "start"]
            if not rng:
                continue
            a, b = rng[0]
            r = t.phase(a, b)
            warp = [y for x, y, fn in t.kernels if a <= x < b and fn.startswith("shell_voronoi::warp")]
            check = [y for x, y, fn in t.kernels if a <= x < b and fn.startswith("tet_start::check")]
            patched = log_value(files.get(f"step_{k}_{side}_0.03.log", ""), "patched_cells", "start:")
            print(f"| {k} | {side}³ | {ms(r['wall']):.1f} | {ms(r['idle']):.1f} ({100 * r['idle'] / r['wall']:.1f} %) | "
                  f"{ms(r.get('idle_alloc', 0)):.1f} | {ms(r['trip']):.2f} | {ms(tail(r)):.2f} | "
                  f"{int(patched) if patched is not None else '-'} | {ms(warp[0] - a) if warp else 0:.0f}, {ms(check[0] - a) if check else 0:.0f} |")
    print("\nMoving step (update + update_cells, median of steps 2 to the last):\n")
    print("| input | N | spacings | step ms | GPU idle ms (%) | read-backs | trips ms | launches | host code ms | "
          "stuck, host-fallback ms |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for side in (128, 256):
        for k in kinds:
            for d in ("0.03", "0.1", "0.3"):
                f = files.get(f"step_{k}_{side}_{d}.sqlite")
                if not f:
                    continue
                p = phases(f).get("step_all")
                if not p:
                    continue
                stuck = host = 0
                for line in open(files[f"step_{k}_{side}_{d}.log"]):
                    if line.startswith("gpuflips:"):
                        stuck += int(re.search(r"stuck=(\d+)", line).group(1))
                        host += float(re.search(r"host_ms=([-0-9.e]+)", line).group(1))
                print(f"| {k} | {side}³ | {d} | {ms(p['wall']):.2f} | {ms(p['idle']):.2f} ({100 * p['idle'] / p['wall']:.1f} %) | "
                      f"{p['readbacks']:.0f} | {ms(p['trip']):.2f} | {p['launches']:.0f} | {ms(p['idle_host']):.2f} | {stuck}, {host:.1f} |")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--markdown" in sys.argv:
        markdown(args)
        sys.exit(0)
    if "--volumes" in sys.argv:
        for d in args:
            for f in sorted(os.listdir(d)):
                if f.startswith("volumes_") and f.endswith(".sqlite"):
                    volumes(os.path.join(d, f))
        sys.exit(0)
    for d in args:
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else [d]:
            p = os.path.join(d, f) if os.path.isdir(d) else f
            if p.endswith(".sqlite"):
                report(p, "--kernels" in sys.argv)
