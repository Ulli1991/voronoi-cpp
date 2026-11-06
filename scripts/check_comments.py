#!/usr/bin/env python3
"""Fail on a run of more than 3 full-line // comments unless it sits directly above a declaration."""
import re
import sys
from pathlib import Path

LIMIT = 3
CONTROL = ("return", "if", "for", "while", "else", "switch", "case", "}", "//", "throw", "do", "using", "#")
KEYWORD = re.compile(r"(template|struct|class|enum|inline|static|KOKKOS|constexpr|\[\[|namespace)\b")
SIGNATURE = re.compile(r"^[\w:<>,\*&\s]+[\s\*&]\**&?[\w:~]+\(")


def is_declaration(line):
    s = line.strip()
    if not s or s.startswith(CONTROL):
        return False
    return bool(KEYWORD.match(s)) or bool(SIGNATURE.match(s) and "=" not in s.split("(")[0])


def check(path):
    lines = path.read_text().split("\n")
    bad, i = [], 0
    while i < len(lines):
        if not lines[i].strip().startswith("//"):
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].strip().startswith("//"):
            j += 1
        following = lines[j] if j < len(lines) else ""
        if j - i > LIMIT and not is_declaration(following):
            bad.append((i + 1, j - i))
        i = j
    return bad


def main(argv):
    root = Path(__file__).resolve().parent.parent
    files = [Path(a) for a in argv] or sorted(p for p in (root / "src").rglob("*") if p.suffix in (".hpp", ".cpp", ".h", ".cu"))
    failed = 0
    for path in files:
        for line, length in check(path):
            print("%s:%d: %d-line comment run (max %d)" % (path, line, length, LIMIT))
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
