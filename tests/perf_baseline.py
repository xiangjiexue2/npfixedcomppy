"""Performance baseline for npfixedcomppy: wall time per (family, n, entry).

Run: python tests/perf_baseline.py
Prints a table of seconds per call (median of 5 reps, first call excluded as
warmup). This is the "before" snapshot for the aggressive-optimization pass.
"""
import os
import statistics
import time

import numpy as np
from npfixedcomppy import computemixdist, estpi0

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))

data1k = np.loadtxt(os.path.join(DATA_ROOT, "npfc_data_1000.csv"),
                    delimiter=",", skiprows=1, ndmin=1)
data5k = np.loadtxt(os.path.join(DATA_ROOT, "npfc_data_5000.csv"),
                    delimiter=",", skiprows=1, ndmin=1)
data50k = np.sort(np.random.default_rng(0).normal(1.0, 1.0, 50000))
pois = np.loadtxt(os.path.join(DATA_ROOT, "npfc_data_pois.csv"),
                  delimiter=",", skiprows=1, ndmin=1)
tanh1k = np.tanh(data1k)

CASES = [
    ("npnormll  n=1000",  lambda: computemixdist(data1k, method="npnormll")),
    ("nptll(b=inf) n=1000", lambda: computemixdist(data1k, method="nptll")),
    ("npnormcvm n=1000",  lambda: computemixdist(data1k, method="npnormcvm")),
    ("npnormad  n=1000",  lambda: computemixdist(data1k, method="npnormad")),
    ("npnormcll n=1000",  lambda: computemixdist(tanh1k, method="npnormcll", beta=len(data1k))),
    ("nppoisll  n=1000",  lambda: computemixdist(pois, method="nppoisll")),
    ("npnormll  n=5000",  lambda: computemixdist(data5k, method="npnormll")),
    ("nptll(b=5)  n=5000", lambda: computemixdist(data5k, method="nptll", beta=5.0)),
    ("npnormll  n=50000", lambda: computemixdist(data50k, method="npnormll")),
    ("estpi0 norm n=1000", lambda: estpi0(data1k, method="npnormll", val=2.0)),
]

rows = []
for name, fn in CASES:
    fn()  # warmup (also catches failures early)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    med = statistics.median(ts)
    rows.append((name, med, min(ts)))
    print(f"{name:22s} median={med*1e3:9.2f} ms   min={min(ts)*1e3:9.2f} ms")

print()
tot = sum(r[1] for r in rows)
print(f"TOTAL (median sum of 10 cases): {tot*1e3:.1f} ms")
raise SystemExit(0)
