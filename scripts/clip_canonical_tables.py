# Tables of clip_canonical_a100.sh logs: rate and start (median ms per design and input, M cells/s, change against
# ref) and checksum (which checksums agree with ref's, per input). Usage: python3 clip_canonical_tables.py <logs>
import collections, re, statistics, sys

kinds = ["jittered", "uniform", "clustered", "halo", "lattice", "wave"]
ms = collections.defaultdict(list)
sums = collections.defaultdict(list)
for f in sys.argv[1:]:
    cur = None
    for line in open(f):
        m = re.match(r"== (rate|start|checksum) (\S+) (\S+) (\d+)", line)
        if m:
            cur = m.groups()
            continue
        if cur is None:
            continue
        mode, design, kind, n = cur
        key = {"rate": "construct_ms", "start": "start_ms"}.get(mode)
        m = key and re.search(key + r"=([\d.]+)", line)
        if m:
            ms[(mode, int(n), design, kind)].append(float(m.group(1)))
        if mode == "checksum":
            step = re.search(r"step=(\d+)", line)
            tag = line.split(":")[0] + ("@" + step.group(1) if step else "")
            m = re.match(r"geometry_checksum: step=\d+ (\d+)", line)
            if m:
                sums[(int(n), design, kind)].append(("geometry_checksum@" + step.group(1), m.group(1)))
            for name, value in re.findall(r"(\w+_checksum)=(\d+)", line):
                sums[(int(n), design, kind)].append((name + "@" + tag, value))
for mode in ("rate", "start"):
    for n in sorted({k[1] for k in ms if k[0] == mode}):
        designs = sorted({k[2] for k in ms if k[0] == mode and k[1] == n}, key=lambda d: (d != "ref", d))
        print(f"{mode} n={n}: M cells/s (change against ref), median of", min(len(v) for k, v in ms.items() if k[:2] == (mode, n)))
        for d in designs:
            row = []
            for kind in kinds:
                v, r = ms.get((mode, n, d, kind)), ms.get((mode, n, "ref", kind))
                if not v:
                    row.append("-")
                    continue
                row.append(f"{n / statistics.median(v) / 1e3:.1f}" +
                           (f" ({100 * (statistics.median(r) / statistics.median(v) - 1):+.0f} %)" if r and d != "ref" else ""))
            print("|", d, "|", " | ".join(row), "|")
for (n, d, kind), values in sorted(sums.items()):
    if d == "ref":
        continue
    ref = dict(sums.get((n, "ref", kind), []))
    same = [k for k, v in values if ref.get(k) == v]
    differ = [k for k, v in values if k in ref and ref[k] != v]
    print(f"checksum n={n} {d} {kind}: same as ref: {' '.join(same)}; differ: {' '.join(differ)}")
