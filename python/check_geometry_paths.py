"""Compare full, reduced, cached and recomputed cell geometry on one dataset."""

import subprocess
import sys
import tempfile
from pathlib import Path

from compare_faces import load


def main():
    executable, distribution = sys.argv[1:]
    compare = Path(__file__).with_name("compare_faces.py")
    with tempfile.TemporaryDirectory() as directory:
        outputs = []
        for name, vertices, planes, policy, sort in (
            ("reference", "cached", "160", "relative", "insertion"),
            ("early", "cached", "64", "relative_early", "insertion"),
            ("recomputed", "recomputed", "160", "legacy", "insertion"),
            ("team", "cached", "64", "relative_early", "team_bitonic"),
            ("nearest_batch", "cached", "64", "relative_nearest_batch", "insertion"),
            # Small enough that every distribution refills across equal distances.
            ("nearest_batch_4", "cached", "64", "relative_nearest_batch_4", "insertion"),
        ):
            output = Path(directory) / f"{name}.csv"
            subprocess.run(
                [executable, str(output), "12", sort, vertices, distribution, planes, policy],
                check=True,
            )
            load(output)  # Also reject empty or duplicate output.
            outputs.append(output)
        for output in outputs[1:]:
            for suffix in ("", ".hydro.csv"):
                subprocess.run(
                    [sys.executable, str(compare), str(outputs[0]) + suffix, str(output) + suffix],
                    check=True,
                )


if __name__ == "__main__":
    main()
