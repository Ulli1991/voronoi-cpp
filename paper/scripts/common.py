"""Readers of paper/data shared by the figure scripts."""
import csv
import json
import os

import numpy as np

from paperfig import DATA


def oracle_errors():
    """{volumes file name: (n, missing cells of all n, errors of the 1000 sampled cells; inf = no volume)}; the paper's
    rerun of voronoi-cpp (oracle_errors_vcpp_rerun.csv) replaces the benchmark's own rows of the same name."""
    out = {}
    for name in ("oracle_errors.csv", "oracle_errors_application.csv", "oracle_errors_vcpp_rerun.csv"):
        path = os.path.join(DATA, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for r in csv.DictReader(line for line in f if not line.startswith("#")):
                out[r["volumes"]] = (int(r["n"]), int(r["missing_or_nonpositive"]),
                                     np.array(r["errors"].split(), dtype=float))
    return out


def records(pattern_dir="raw", prefix="sweep-"):
    """All JSON records of paper/data/<dir>/<prefix>*.jsonl (Slurm epilogue lines skipped)."""
    out = []
    d = os.path.join(DATA, pattern_dir)
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if name.startswith(prefix) and name.endswith(".jsonl"):
            for line in open(os.path.join(d, name)):
                if line.startswith("{") and '"phase"' in line:
                    out.append(json.loads(line))
    return out
