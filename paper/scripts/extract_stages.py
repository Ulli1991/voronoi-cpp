#!/usr/bin/env python3
"""paper/data/stages.csv and kernels.csv from stages_a100.sh kernels output (paper/data/raw/kernels-<job>.txt): per
input and spacings the median ms per step of each stage (steps 2 to 10) and each kernel label (KernelTimes), and
the rebuild's and start's stages. Usage: extract_stages.py [kernels-<job>.txt ...] (default: all)."""
import csv
import glob
import os
import re
import statistics
import sys

from paperfig import DATA

STAGES = [("repair", "gpu_ms"), ("host fallback", "host_ms"), ("rebuild", "rebuild_ms"), ("re-sort", "resort_ms"),
          ("geometry", "centres_faces_ms"), ("cells", "cells_ms")]


def fields(line):
    return dict(re.findall(r"(\w+)=([-\w.+]+)", line))


def main(paths):
    stages, kernels = [], []
    for path in paths:
        job = re.findall(r"(\d+)", os.path.basename(path))[-1]
        what = steps = None
        for line in open(path, errors="replace"):
            if line.startswith("== "):
                what = line.split()[1:]
                steps = []
            elif what and what[0] == "rebuild" and line.startswith("warp:"):
                f = fields(line)
                for name, key in (("cell list", "build_ms"), ("clipping", "construct_ms")):
                    stages.append([job, what[1], int(what[2]), "rebuild", name, f[key]])
            elif what and what[0] == "step":
                kind, n, d = what[1], int(what[2]), what[3]
                if line.startswith("start:"):
                    f = fields(line)
                    stages.append([job, kind, n, "start", "clipping with rings", f["clip_ms"]])
                    stages.append([job, kind, n, "start", "exact start after clipping",
                                   f"{float(f['start_ms']) - float(f['clip_ms']):.4g}"])
                elif line.startswith("gpuflips:"):
                    steps.append(fields(line))
                elif line.startswith("kernel:"):
                    label = line.split()[1]
                    kernels.append([job, kind, n, d, label, fields(line)["ms_per_step"]])
                elif line.startswith("final_rebuild:") and steps:
                    later = [s for s in steps if int(s["step"]) >= 2]
                    for name, key in STAGES:
                        ms = statistics.median(float(s[key]) for s in later)
                        stages.append([job, kind, n, f"step {d}", name, f"{ms:.4g}"])
    with open(os.path.join(DATA, "stages.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job", "input", "n", "what", "stage", "ms"])
        w.writerows(stages)
    with open(os.path.join(DATA, "kernels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job", "input", "n", "spacings", "kernel", "ms_per_step"])
        w.writerows(kernels)
    print(f"{len(stages)} stage rows, {len(kernels)} kernel rows")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(glob.glob(os.path.join(DATA, "raw", "kernels-*.txt"))))
