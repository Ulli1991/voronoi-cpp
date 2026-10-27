"""Compare directed Voronoi face areas keyed by original particle IDs."""
import csv
import math
import sys


def load(path):
    faces = {}
    with open(path) as stream:
        for row in csv.reader(stream):
            key = (int(row[0]), int(row[1]))
            if key in faces:
                raise RuntimeError(f"duplicate face {key} in {path}")
            faces[key] = tuple(map(float, row[2:]))
    if not faces:
        raise RuntimeError(f"no faces in {path}")
    return faces


if __name__ == "__main__":
    cpu, gpu = map(load, sys.argv[1:])
    if cpu.keys() != gpu.keys():
        raise RuntimeError("CPU/GPU face neighbor IDs differ")
    max_error = 0.0
    for key in cpu:
        if len(cpu[key]) != len(gpu[key]):
            raise RuntimeError(f"CPU/GPU geometry dimensions differ at {key}")
        for a, b in zip(cpu[key], gpu[key]):
            if not math.isfinite(a) or not math.isfinite(b):
                raise RuntimeError(f"nonfinite face area at {key}")
            max_error = max(max_error, abs(a - b))
            if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
                raise RuntimeError(f"CPU/GPU area mismatch at {key}: {a} vs {b}")
    print(f"CPU/GPU comparison PASS faces={len(cpu)} max_area_component_error={max_error:.6g}")
