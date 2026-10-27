"""Plot offline scaling CSVs; requires Matplotlib only for rendering."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, help="Output prefix for PNG/PDF")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with args.csv.open() as stream:
        rows = sorted(
            (row for row in csv.DictReader(stream) if row["status"] == "PASS"),
            key=lambda row: int(row["particles"]),
        )
    if not rows:
        raise SystemExit("No passing scaling measurements in CSV")
    n = [int(row["particles"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    for key, label in (
        ("search", "Neighbor rebuild (particles/s)"),
        ("reconstruction", "Reconstruction (cells/s)"),
        ("pipeline", "Combined (particles/s)"),
    ):
        rate_key = key + ("_Mcells_s" if key == "reconstruction" else "_Mparticles_s")
        axes[0].semilogx(n, [float(row[rate_key]) for row in rows], "o-", label=label)
        axes[1].loglog(n, [float(row[key + "_ms"]) for row in rows], "o-", label=label.split(" (")[0])
    axes[0].set_ylabel("Throughput (millions/s)")
    axes[1].set_ylabel("Median time (ms)")
    for key, label in (("estimated_array_bytes", "Array payload estimate"),
                       ("max_observed_device_used_bytes", "Observed device usage")):
        axes[2].semilogx(n, [float(row[key]) / 2**30 for row in rows], "o-", label=label)
    total = float(rows[-1]["total_device_bytes"]) / 2**30
    if total:
        axes[2].axhline(total, color="black", linestyle="--", label="Device capacity")
    axes[2].set_ylabel("Memory (GiB)")
    for axis in axes:
        axis.set_xlabel("Particles / cells")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Single-GPU local Voronoi construction: " + rows[-1]["policy"])
    output = args.output or args.csv.with_suffix("")
    fig.savefig(str(output) + ".png", dpi=180)
    fig.savefig(str(output) + ".pdf")


if __name__ == "__main__":
    main()
