"""Validation contracts independent of access to a GPU or Nsight installation."""

import csv
import io
import unittest

from check_ncu_capture import validate


class CaptureCheck(unittest.TestCase):
    def capture(self, values=("12", "1,024"), kernel="void reconstruct_hydro<Cached>()", ids=None):
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(
            ["ID", "Process ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"]
        )
        for index, (name, value) in enumerate(zip(self.required, values)):
            writer.writerow([0 if ids is None else ids[index], 100, kernel, name, "byte", value])
        return stream.getvalue()

    required = ("gpu__time_duration.sum", "dram__bytes_read.sum")

    def test_raw_wide(self):
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(["ID", "Process ID", "Kernel Name", *self.required])
        writer.writerow(["", "", "", "ms", "Mbyte"])
        writer.writerow(["0", "100", "void reconstruct_hydro<Cached>()", "6.08", "886.04"])
        result = validate(stream.getvalue(), self.required)
        self.assertEqual(result["metrics"][self.required[1]], {"value": 886.04, "unit": "Mbyte"})
        writer.writerow(["1", "100", "void reconstruct_hydro<Cached>()", "6.08", "886.04"])
        with self.assertRaises(ValueError):
            validate(stream.getvalue(), self.required)

    def test_valid(self):
        result = validate(self.capture(), self.required)
        self.assertEqual(result["metrics"][self.required[1]]["value"], 1024)

    def test_invalid_values(self):
        for value in ("nan", "inf", "-1", "(!) nan", "N/A", "0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate(self.capture((value, "12")), self.required)

    def test_wrong_kernel(self):
        for kernel in (
            "ParallelScan<foo>",
            "profile_workspace<foo>",
            "retry_compact_plane_capacity",
        ):
            with self.subTest(kernel=kernel), self.assertRaises(ValueError):
                validate(self.capture(kernel=kernel), self.required)

    def test_missing_metric(self):
        with self.assertRaises(ValueError):
            validate(self.capture(("12",)), self.required)

    def test_multiple_launches(self):
        with self.assertRaises(ValueError):
            validate(self.capture(ids=(0, 1)), self.required)

    def test_missing_header(self):
        with self.assertRaises(ValueError):
            validate("==ERROR== driver resource unavailable", self.required)


if __name__ == "__main__":
    unittest.main()
