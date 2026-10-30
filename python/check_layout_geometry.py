"""Compare unsorted/Morton geometry with the sorted Cartesian reference by particle ID."""

import subprocess
import sys
import tempfile
from pathlib import Path

from compare_faces import load


def main():
    executable, distribution = sys.argv[1:]
    compare = Path(__file__).with_name("compare_faces.py")
    with tempfile.TemporaryDirectory(prefix="voronoi-layout-") as directory:
        outputs = []
        for mode in ("team_bitonic_recompute", "unsorted", "morton", "morton_unsorted"):
            output = Path(directory) / f"{mode}.csv"
            subprocess.run(
                [
                    executable,
                    str(output),
                    "12",
                    mode,
                    "cached",
                    distribution,
                    "64",
                    "relative_early",
                ],
                check=True,
            )
            load(output)
            outputs.append(output)
        for output in outputs[1:]:
            for suffix in ("", ".hydro.csv"):
                subprocess.run(
                    [sys.executable, str(compare), str(outputs[0]) + suffix, str(output) + suffix],
                    check=True,
                )


if __name__ == "__main__":
    main()
