"""Check stage-output schema and successful geometry validation for both sort paths."""

import csv
import math
import pathlib
import subprocess
import sys
import tempfile


def main():
    executable = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="voronoi-stage-check-") as temporary:
        root = pathlib.Path(temporary)
        timings, stages = root / "timings.csv", root / "stages.csv"
        policies = ("relative_early", "relative_early_scratch_sort")
        for policy in policies:
            subprocess.run([executable, "12", "3", policy, str(timings), str(stages)], check=True)
        with timings.open() as stream:
            results = list(csv.DictReader(stream))
        with stages.open() as stream:
            profiles = list(csv.DictReader(stream))
        assert len(results) == len(profiles) == len(policies)
        fields = (
            "key_count_ms",
            "scan_ms",
            "scatter_ms",
            "cell_order_gather_ms",
            "candidate_fill_ms",
            "nearest_order_ms",
            "overflow_check_ms",
            "instrumented_wall_ms",
        )
        for result, profile, policy in zip(results, profiles, policies):
            assert result["status"] == "PASS" and int(result["validation_errors"]) == 0
            assert profile["policy"] == policy and profile["backend"] == result["backend"]
            assert int(profile["side"]) == 12 and int(profile["repeats"]) == 3
            for field in fields:
                value = float(profile[field])
                assert math.isfinite(value) and value >= 0, (field, value)
            assert float(profile["instrumented_wall_ms"]) > 0
            assert float(profile["candidate_fill_ms"]) > 0
            assert float(profile["nearest_order_ms"]) > 0
        print("Search-stage schema, append behavior, timing values and geometry: PASS")


if __name__ == "__main__":
    main()
