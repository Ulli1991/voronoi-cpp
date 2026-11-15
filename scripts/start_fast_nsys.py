#!/usr/bin/env python3
"""The start_once range of an nsys export (sqlite): wall time, kernel time, launches, and host API time by call.
  python3 scripts/start_fast_nsys.py <file.sqlite>..."""
import sqlite3
import sys

for path in sys.argv[1:]:
    db = sqlite3.connect(path)
    row = db.execute("select start, end from NVTX_EVENTS where text = 'start_once'").fetchone()
    if not row:
        print(path, "no start_once range")
        continue
    t0, t1 = row
    kernels = db.execute("select count(*), sum(end - start) from CUPTI_ACTIVITY_KIND_KERNEL where start >= ? and end <= ?",
                         (t0, t1)).fetchone()
    api = db.execute("""select s.value, count(*), sum(r.end - r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r
                        join StringIds s on r.nameId = s.id where r.start >= ? and r.end <= ?
                        group by s.value order by sum(r.end - r.start) desc limit 8""", (t0, t1)).fetchall()
    print("%s: wall %.1f ms, %d kernels %.1f ms, host gap %.1f ms" % (
        path, (t1 - t0) / 1e6, kernels[0], (kernels[1] or 0) / 1e6, (t1 - t0 - (kernels[1] or 0)) / 1e6))
    for name, count, ns in api:
        print("   %-40s %5d calls %8.1f ms" % (name, count, ns / 1e6))
