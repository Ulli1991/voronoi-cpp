#!/usr/bin/env python3
"""Figure tried (design decisions) and its appendix table, both parsed from docs/tried.md: per row the measured rate
change (dots, coloured kept or dropped; inputs named in the row get their marker), then the effect as written and
the structural reason. Rows without a rate (memory, precision) are listed as text. Writes figures/tried.{pdf,png,md}."""
import os
import re

import matplotlib.pyplot as plt

from paperfig import DOUBLE, LW, MARKER, OUT, REPO, figstyle, figure, save

NUM = r"[+−-]?\d+(?:\.\d+)?"
INPUT = r"(jittered|blue noise|uniform|clustered|halo|lattice|wave)"
SKIP = ("kernel", "exact start", "syncs", "falls back", "stuck", "memory", "hydro")
LIMIT = -100  # rate changes below are drawn at the edge with their value


def number(s):
    return float(s.replace("−", "-"))


def sections(path):
    """[(section, [row dict])] of the tables of tried.md."""
    out, name, head = [], None, None
    for line in open(path):
        if line.startswith("## "):
            name, head = line[3:].strip(), None
            out.append((name, []))
        elif line.startswith("|") and name:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if head is None:
                head = cells
            elif not set(cells[0]) <= set("-: "):
                out[-1][1].append(dict(zip(head, cells)))
    return [(n, rows) for n, rows in out if rows]


def changes(effect):
    """Rate changes in % from an Effect cell: [(value, input or None)]; [] if it states no rate."""
    clauses = [c for c in re.split(r"[;,]", effect) if not any(w in c for w in SKIP)]
    text = " ".join(clauses)
    named = [(number(v), k) for k, v in re.findall(INPUT + r"\s*(" + NUM + r")\s*%", text)]
    if named:
        return named
    m = re.search(r"(" + NUM + r")\s*[–-]\s*(" + NUM + r")\s*×", text)
    if m:
        return [((number(m.group(1)) - 1) * 100, None), ((number(m.group(2)) - 1) * 100, None)]
    m = re.search(r"(" + NUM + r") against (" + NUM + r")", text)
    if m:
        return [((number(m.group(1)) / number(m.group(2)) - 1) * 100, None)]
    vals = [number(v) for v in re.findall(r"(" + NUM + r")(?=\s*(?:%|(?:to|/|–)\s*" + NUM + r"\s*%))", text)]
    if "worse" in text:
        vals = [-abs(v) for v in vals]
    if re.search(r"about (equal|0\b)", text):
        vals.append(0.0)
    return [(v, None) for v in vals]


def verdict(kept):
    """The Kept? cell, shortened, and its colour: kept green, dropped red, the rest grey."""
    k = kept.lower()
    word = brief(kept, 22) if kept.strip() not in ("—", "-", "") else "not built"
    if k.startswith("kept"):
        return word, "#2ca02c"
    if k.startswith("dropped") or k.startswith("rejected"):
        return word, "#d62728"
    return word, "0.45"


def short(text, width):
    text = re.sub(r"\s*\(.*?\)", "", text) if len(text) > width else text
    return text if len(text) <= width else text[:width - 1] + "…"


def brief(text, width=64, seps=(";", ": ", ", but", ", giving", ", which", " and raises", ", plus", ", ")):
    """The first clause of a cell, without parentheses, for the figure; the table keeps the whole cell."""
    text = re.sub(r"\s*\(.*?\)", "", text.replace("`", ""))
    for sep in seps:
        if len(text) > width and sep in text:
            text = text.split(sep)[0]
    return short(text, width)


# The compact main-text figure: the rebuild's key decisions, in this order (prefixes of tried.md's Change column).
KEY = ["Lean cut", "kNN-first", "kNN hybrids", "P1 per-vertex", "Split clipping", "Batched rounds",
       "Float side test", "Compact cell", "Delaunay-first: lattice", "Delaunay-first: batched"]


# Reasons in the owner's words where tried.md's first clause is too short to stand alone.
REASON = {"kNN-first": "find all, then clip (fixed k) vs the interleaved find-and-clip walk",
          "kNN hybrids": "the same: a separate find pass before the walk"}


def draw(rows, name, headers=True, legend=True, ncol=2):
    """One figure of tried.md rows [(section, row)]: dots, then the verdict, effect and reason as text."""
    extra = (3.2 if ncol == 2 else 2.2) if legend else 0.2
    fig, ax = figure(DOUBLE, 18 + 6.0 * len(rows) + ((12 if ncol == 2 else 8) if legend else 0))
    y, last = 0, None
    ticks, labels = [], []
    for s, r in rows:
        if headers and last != s:
            last = s
            y += 0.5 if ticks else 0.0
            ax.text(LIMIT - 2, y, s, fontweight="bold", ha="right", va="center", fontsize=12)
            y += 1.0
        effect, kept, reason = r["Effect"], r["Kept?"], r["Reason"]
        word, colour = verdict(kept)
        vals = [] if s.startswith("Precision") else changes(effect)
        if vals:
            v = [x for x, _ in vals]
            ax.plot([max(min(v), LIMIT), max(v)], [y, y], color=colour, lw=LW, alpha=0.5, solid_capstyle="round")
            for x, k in vals:
                ax.plot(max(x, LIMIT), y, MARKER.get(k.replace(" ", ""), "o") if k else "o", ms=7, mfc=colour,
                        mec="white", mew=0.6, zorder=3)
                if x < LIMIT:
                    ax.annotate(f"{x:.0f} %", (LIMIT, y), xytext=(LIMIT + 2, y - 0.38), fontsize=11, color=colour)
        lead = brief(effect) if brief(effect) != word else ""
        ax.text(1.03, y - 0.05, f"{word}: {lead}" if lead else word, va="center", fontsize=11, color=colour,
                transform=ax.get_yaxis_transform())
        why = next((v for k, v in REASON.items() if r["Change"].startswith(k)), None) if not headers else None
        ax.text(1.03, y + 0.42, why or brief(reason), va="center", fontsize=11, color="0.35",
                transform=ax.get_yaxis_transform())
        ticks.append(y)
        labels.append(brief(r["Change"], 62, seps=(";", ", plus")))
        y += 1.0
    ax.axvline(0, color="0.3", lw=1.0)
    ax.set_yticks(ticks, labels, fontsize=11)
    ax.set_ylim(y - 0.4 + extra, -0.6)
    ax.set_xlim(LIMIT - 2, 52)
    ax.set_xlabel("rate change against the code before, %")
    figstyle.decorate(ax)
    ax.tick_params(axis="y", which="both", left=False, right=False)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    if legend:
        for k in ("jittered", "uniform", "clustered", "halo", "lattice", "wave"):
            ax.plot([], [], MARKER[k], ms=7, mfc="0.45", mec="white", ls="none", label=k)
        ax.legend(loc="lower center", fontsize=11, ncol=ncol, columnspacing=0.6, handletextpad=0.2,
                  title="marker: the input a row names", title_fontsize=11)
    fig.subplots_adjust(left=0.29, right=0.52, top=0.99, bottom=0.12 if not legend else 0.035)
    save(fig, name, DOUBLE)


def main():
    secs = sections(os.path.join(REPO, "docs", "tried.md"))
    rows = [(s, r) for s, rs in secs for r in rs]
    draw(rows, "tried")
    key = [next((s, r) for s, r in rows if r["Change"].startswith(p)) for p in KEY]
    draw(key, "tried_key", headers=False)
    with open(os.path.join(OUT, "tried.md"), "w") as f:
        f.write("Appendix table, generated by paper/scripts/fig_tried.py from docs/tried.md.\n\n")
        for s, rs in secs:
            f.write(f"\n**{s}**\n\n| Change | Effect | Kept? | Reason |\n|---|---|---|---|\n")
            for r in rs:
                f.write(f"| {r['Change']} | {r['Effect']} | {r['Kept?']} | {r['Reason']} |\n")


if __name__ == "__main__":
    main()
