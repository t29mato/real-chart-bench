"""Exact-equal points between two models' answers to the same CHART-Info batch
(a copying check; the sealed directories of one batch are siblings).

Prints, per figure with any overlap, the count of identical points and how
many of them are off a 0.1 grid (round-grid values coincide naturally).

Usage: python scripts/eval/chartinfo_overlap.py <a.predictions.json> <b.predictions.json>
"""

import json
import sys


def pts(series):
    return [
        (round(p["x"], 6), round(p["y"], 6))
        for s in series
        for p in s.get("data", [])
        if isinstance(p.get("x"), int | float) and isinstance(p.get("y"), int | float)
    ]


def off_grid(q):
    return abs(q[0] - round(q[0], 1)) > 1e-9 or abs(q[1] - round(q[1], 1)) > 1e-9


a, b = (json.load(open(f)) for f in sys.argv[1:3])
eq = tot = off = 0
for k in sorted(a):
    sb = set(pts(b.get(k, [])))
    pa = pts(a[k])
    e = [q for q in pa if q in sb]
    eq, tot, off = eq + len(e), tot + len(pa), off + sum(map(off_grid, e))
print(f"identical {eq}/{tot}, off the 0.1 grid {off}")
