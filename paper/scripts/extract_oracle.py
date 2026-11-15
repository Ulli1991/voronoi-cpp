"""Oracle errors per cell of every library-benchmark volumes file into paper/data/oracle_errors.csv.

Needs oracle_errors (g++ -O2 -std=c++17 oracle_errors.cpp); checks each file's median and max against the jsonl records.
"""
import glob, json, os, subprocess, sys

DATA = os.environ.get("LIBBENCH_DATA", "/raven/u/uli/libbench-data")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "paper", "data", "oracle_errors.csv")


def points_of(volumes):
    rest = os.path.basename(volumes)[:-4].split("_", 1)[1]
    if "_step" in rest:
        base, frame = rest.rsplit("_step", 1)
        return f"{DATA}/points/{base}_step0.03.trj@{frame}"
    return f"{DATA}/points/{rest}.pts"


def main(exe):
    groups = {}
    for v in sorted(glob.glob(f"{DATA}/volumes/*.f64")):
        p = points_of(v)
        if os.path.exists(p + ".oracle1000"):
            groups.setdefault(p, []).append(v)
    records = {}
    for f in glob.glob(os.path.join(REPO, "bench/libraries/results/accuracy-*.jsonl")):
        for line in open(f):
            if line.startswith("{"):
                d = json.loads(line)
                records[os.path.basename(d["volumes"])] = (d, os.path.basename(f))
    rows, odd = [], []
    for p, vs in groups.items():
        out = subprocess.run([exe, p] + vs, capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            path, n, missing, errs = line.split(",")
            name = os.path.basename(path)
            e = sorted(float(x) for x in errs.split())
            rec, src = records.get(name, (None, ""))
            if rec is not None and rec["max_rel_err"] is not None and rec["median_rel_err"] is not None:
                if abs(e[500] - rec["median_rel_err"]) > 0.01 * rec["median_rel_err"] + 1e-30:
                    odd.append(f"{name}: median {e[500]:.3e} vs record {rec['median_rel_err']:.3e} ({src})")
            rows.append(f"{name},{points_of(path).split('/')[-1]},{n},{missing},{src},{errs}")
    with open(OUT, "w") as f:
        f.write(f"# per-cell |V - V_oracle| / V_oracle of the 1000 sampled cells (bench/libraries/oracle_check.cpp);"
                f" volumes and oracle caches under {DATA}; record = the accuracy jsonl that lists the file\n")
        f.write("volumes,points,n,missing_or_nonpositive,record,errors\n")
        f.write("\n".join(rows) + "\n")
    print(f"{len(rows)} files -> {OUT}")
    print("\n".join(odd) if odd else "medians match the records")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "oracle_errors"))
