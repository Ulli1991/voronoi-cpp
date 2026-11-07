#!/usr/bin/env python3
"""Fail if clang-format would change a line this branch changed relative to BASE (default origin/main).
Only changed lines are checked, so code that predates .clang-format is left alone. Set CLANG_FORMAT to pick the binary."""
import os
import re
import subprocess
import sys

SUFFIXES = (".hpp", ".cpp", ".h", ".cu")
HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@")


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def changed_ranges(base, path):
    """Line ranges of `path` added or changed since the merge base with `base`."""
    ranges = []
    for line in git("diff", "-U0", base + "...HEAD", "--", path).split("\n"):
        m = HUNK.match(line)
        if m and m.group(2) != "0":
            start, count = int(m.group(1)), int(m.group(2) or 1)
            ranges.append((start, start + count - 1))
    return ranges


def main(argv):
    base = argv[0] if argv else "origin/main"
    tool = os.environ.get("CLANG_FORMAT", "clang-format")
    failed = 0
    for path in git("diff", "--name-only", "--diff-filter=AM", base + "...HEAD").split():
        if not path.startswith("src/") or not path.endswith(SUFFIXES):
            continue
        ranges = changed_ranges(base, path)
        if not ranges:
            continue
        cmd = [tool, "--style=file", path] + ["--lines=%d:%d" % r for r in ranges]
        formatted = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        diff = subprocess.run(["diff", "-u", path, "-"], input=formatted, capture_output=True, text=True).stdout
        if diff:
            print(diff)
            failed += 1
    if failed:
        print("%d file(s) need formatting: run `clang-format -i --style=file --lines=A:B FILE` on the changed lines" % failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
