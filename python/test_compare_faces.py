"""Exercise success and failure contracts of the geometry comparison CLI."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ComparisonTests(unittest.TestCase):
    def compare(self, left, right, success):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.csv", Path(directory) / "b.csv"
            a.write_text(left)
            b.write_text(right)
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("compare_faces.py")), str(a), str(b)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)

    def test_equal_and_reordered(self):
        self.compare("0,1,1,2,3\n1,0,-1,-2,-3\n", "1,0,-1,-2,-3\n0,1,1,2,3\n", True)

    def test_roundoff(self):
        self.compare("0,1,1,2,3\n", "0,1,1.0000000001,2,3\n", True)

    def test_empty(self):
        self.compare("", "", False)

    def test_duplicate(self):
        self.compare("0,1,1,2,3\n0,1,1,2,3\n", "0,1,1,2,3\n", False)

    def test_missing_face(self):
        self.compare("0,1,1,2,3\n", "0,2,1,2,3\n", False)

    def test_dimension_mismatch(self):
        self.compare("0,1,1,2,3\n", "0,1,1,2\n", False)

    def test_nonfinite(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                self.compare(f"0,1,{value},2,3\n", f"0,1,{value},2,3\n", False)

    def test_wrong_geometry(self):
        self.compare("0,1,1,2,3\n", "0,1,1.01,2,3\n", False)

    def test_malformed(self):
        self.compare("bad,data\n", "0,1,1,2,3\n", False)


if __name__ == "__main__":
    unittest.main()
