#!/usr/bin/env python3
"""Fail on a run of more than 3 full-line // comments unless it sits directly above a declaration, or on a file
without a header comment in its first 30 lines. `--report` lists multi-line library definitions without a docstring."""
import re
import sys
from pathlib import Path

LIMIT = 3
HEADER_LINES = 30
LIBRARY = ("neighbour_search", "voronoi_construct", "hydro")
CONTROL = re.compile(r"(return|if|for|while|else|switch|case|throw|do|using)\b|[}#]|//")
KEYWORD = re.compile(r"(template|struct|class|enum|inline|static|KOKKOS|constexpr|\[\[|namespace)\b")
SIGNATURE = re.compile(r"^[\w:<>,\*&\s]+[\s\*&]\**&?[\w:~]+\(")


def is_declaration(line):
    s = line.strip()
    if not s or CONTROL.match(s):
        return False
    return bool(KEYWORD.match(s)) or bool(SIGNATURE.match(s) and "=" not in s.split("(")[0])


def has_header(path):
    return any(line.startswith("//") for line in path.read_text().split("\n")[:HEADER_LINES])


def undocumented(path):
    """Multi-line definitions at namespace scope with no // line directly above (overload groups may share one)."""
    lines = path.read_text().split("\n")
    found = []
    for i, line in enumerate(lines):
        if not line[:1].isalpha() or line.startswith(("namespace", "using", "return", "if ", "for ")):
            continue
        if not is_declaration(line) or (i and lines[i - 1].startswith(("template", "//"))):
            continue
        if line.rstrip().endswith(";") or ("=" in line.split("(")[0] and not line.startswith("template")):
            continue
        end = i
        while end < len(lines) and not lines[end].rstrip().endswith(("{", "}", ";")):
            end += 1
        if end < len(lines) and lines[end].rstrip().endswith("}"):
            continue  # a one-line definition
        found.append(i + 1)
    return found


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
    report = "--report" in argv
    argv = [a for a in argv if a != "--report"]
    root = Path(__file__).resolve().parent.parent
    files = [Path(a) for a in argv] or sorted(p for p in (root / "src").rglob("*") if p.suffix in (".hpp", ".cpp", ".h", ".cu"))
    failed = 0
    for path in files:
        if not has_header(path):
            print("%s:1: no header comment in the first %d lines" % (path, HEADER_LINES))
            failed += 1
        for line, length in check(path):
            print("%s:%d: %d-line comment run (max %d)" % (path, line, length, LIMIT))
            failed += 1
        if report and path.parent.name in LIBRARY:
            for line in undocumented(path):
                print("%s:%d: no docstring" % (path, line))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
