"""Reject wrong-kernel, missing, nonnumeric or nonfinite Nsight CSV captures."""

import argparse
import csv
import io
import json
import math
from pathlib import Path


def validate(text, required):
    lines = text.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if "ID" in line and "Kernel Name" in line),
        None,
    )
    if start is None:
        raise ValueError("No Nsight raw metric CSV header")
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    fields = reader.fieldnames or []
    if len(fields) != len(set(fields)):
        raise ValueError("Duplicate CSV columns")
    rows = list(reader)
    if "Metric Name" not in fields:
        # Raw-page exports use one column per metric and a separate units row.
        if not rows or rows[0].get("ID") or rows[0].get("Kernel Name"):
            raise ValueError("Missing raw CSV units row")
        units, captures_wide = rows[0], rows[1:]
        rows = []
        for capture in captures_wide:
            if not any(capture.values()):
                continue
            for name in required:
                if name not in fields:
                    raise ValueError(f"Missing counter: {name}")
                rows.append(
                    {
                        "ID": capture.get("ID"),
                        "Process ID": capture.get("Process ID"),
                        "Kernel Name": capture.get("Kernel Name", ""),
                        "Metric Name": name,
                        "Metric Value": capture[name],
                        "Metric Unit": units.get(name, ""),
                    }
                )
    captures = set()
    values = {}
    for row in rows:
        if not row.get("Metric Name"):
            continue
        kernel = row.get("Kernel Name", "")
        if "reconstruct_hydro<" not in kernel or "ParallelScan<" in kernel:
            raise ValueError(f"Unexpected kernel: {kernel}")
        captures.add((row.get("ID"), row.get("Process ID"), kernel))
        name = row["Metric Name"]
        if name in required:
            if name in values:
                raise ValueError(f"Duplicate metric: {name}")
            value = float(row["Metric Value"].replace(",", ""))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid counter {name}: {value}")
            values[name] = {"value": value, "unit": row.get("Metric Unit", "")}
    if len(captures) != 1:
        raise ValueError(f"Expected one main-kernel capture, got {len(captures)}")
    missing = set(required) - values.keys()
    if missing:
        raise ValueError(f"Missing counters: {sorted(missing)}")
    for name in ("gpu__time_duration.sum", "dram__bytes_read.sum", "smsp__inst_executed.sum"):
        if name in values and values[name]["value"] <= 0:
            raise ValueError(f"Empty kernel counter: {name}")
    return {"kernel": next(iter(captures))[2], "metrics": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--required", required=True, help="Comma-separated metric names")
    args = parser.parse_args()
    result = validate(args.report.read_text(), args.required.split(","))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
