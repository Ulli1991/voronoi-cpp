#!/usr/bin/env python3
"""Tables of distortion_test logs (--print): one row per run, and with
--stages the same run at four points in its strain.
    python3 scripts/flow_table.py [--stages] <log>...
Rates are M cells/s of the whole update: repair, geometry, cells, fallbacks."""
import re
import sys


def fields(line):
    return dict(re.findall(r'([A-Za-z_0-9-]+)=([^ ]+)', line))


def runs(path):
    steps = []
    refused = None
    checks = [0, 0]
    for line in open(path, errors='replace'):
        if line.startswith('step='):
            f = fields(line)
            if f['step'] == '1' and steps:  # the last run ended without its summary
                yield '(aborted)', '?', steps, refused, checks, ''
                steps, refused, checks = [], None, [0, 0]
            steps.append(f)
            checks[0] += 'rebuild=PASS' in line
            checks[1] += 'rebuild=FAIL' in line
        elif line.startswith('refused:'):
            refused = line.split(' ', 3)[-1].strip()
        elif line.startswith('  ok') or line.startswith('  FAIL'):
            name = line.split()[1]
            yield name, line.split()[0], steps, refused, checks, line
            steps, refused, checks = [], None, [0, 0]
    if steps:
        yield '(unfinished)', '?', steps, refused, checks, ''


def median(values):
    values = sorted(values)
    return values[len(values) // 2] if values else 0.0


def row(name, status, steps, refused, checks):
    rate = [float(s['Mcells_s']) for s in steps]
    tail = steps[len(steps) // 2:]
    stuck = sum(int(s['stuck']) for s in steps)
    rebuilt = sum(int(s['rebuilt']) for s in steps)
    first = next((s['step'] for s in steps if int(s['stuck']) > 0), '-')
    bytes_cell = [float(s['B_per_cell']) for s in steps]
    return (f"{name:13s} {status:4s} steps {len(steps):5d} t {float(steps[-1]['t']):6.3f} "
            f"rate med {median(rate):7.1f} last-half {median([float(s['Mcells_s']) for s in tail]):7.1f} "
            f"worst {min(rate):8.3f} | flips/cell {sum(int(s['flips']) for s in steps) / max(len(steps), 1) / 1e6:6.2f}M/step "
            f"rounds max {max(int(s['rounds']) for s in steps):4d} | first stuck {first:>5s} stuck {stuck:9d} "
            f"rebuilt {rebuilt:4d} | aspect {max(float(s['aspect']) for s in steps):9.1f} "
            f"reach {max(float(s['reach']) for s in steps):.3f} faces {max(int(s['max_faces']) for s in steps):3d} "
            f"corners {max(int(s['max_corners']) for s in steps):3d} gap {min(float(s['gap']) for s in steps):.2e} "
            f"f/cell {float(steps[0]['faces_per_cell']):.2f}->{float(steps[-1]['faces_per_cell']):.2f} "
            f"B/cell {bytes_cell[0]:.0f}->{max(bytes_cell):.0f} "
            f"checks {checks[0]} pass {checks[1]} fail" + (f" | REFUSED: {refused[:50]}" if refused else ""))


def main():
    args = sys.argv[1:]
    stages = '--stages' in args
    for path in [a for a in args if a != '--stages']:
        print('#', path)
        for name, status, steps, refused, checks, line in runs(path):
            if not steps:
                print(f"{name:13s} {status:4s} no steps: {line[:150].strip()}")
                continue
            print(row(name, status, steps, refused, checks))
            if stages:
                n = len(steps)
                for a, b in ((0, n // 4), (n // 4, n // 2), (n // 2, 3 * n // 4), (3 * n // 4, n)):
                    part = steps[a:b]
                    if part:
                        print('   ' + row(f"{part[0]['step']}-{part[-1]['step']}", '', part, None, [0, 0]))


main()
