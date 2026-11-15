"""Readers of paper/data shared by the figure scripts."""
import csv
import json
import os

import numpy as np

from paperfig import DATA


def oracle_errors():
    """{volumes file name: (n, missing cells of all n, errors of the 1000 sampled cells; inf = no volume)}."""
    out = {}
    with open(os.path.join(DATA, "oracle_errors.csv")) as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        for r in rows:
            out[r["volumes"]] = (int(r["n"]), int(r["missing_or_nonpositive"]),
                                 np.array(r["errors"].split(), dtype=float))
    return out


def records(pattern_dir="raw", prefix="weak-"):
    """All JSON records of paper/data/<dir>/<prefix>*.jsonl (Slurm epilogue lines skipped)."""
    out = []
    d = os.path.join(DATA, pattern_dir)
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if name.startswith(prefix) and name.endswith(".jsonl"):
            for line in open(os.path.join(d, name)):
                if line.startswith("{") and '"phase"' in line:
                    out.append(json.loads(line))
    return out
