"""Wall-time bench for the 4 binned ("...w") families (Python side).

Prints seconds per call (median of 5 reps, first call excluded as warmup)
for the binned families at n=5000 and n=20000, plus the un-binned
equivalents at the same n as reference rows ("[ref]") so the table can
show both the Py-vs-R speedup and the binned-vs-un-binned scaling that
is the point of the "large-scale" (R chapter 6) variants.
"""
import os
import statistics
import time

import numpy as np

from npfixedcomppy import computemixdist, estpi0
from npfixedcomppy.npfc import _bin

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))

data5k = np.loadtxt(os.path.join(DATA_ROOT, "npfc_data_5000.csv"),
                    delimiter=",", skiprows=1, ndmin=1)
data20k = np.loadtxt(os.path.join(DATA_ROOT, "npfc_bench_data_20000.csv"),
                     delimiter=",", skiprows=1, ndmin=1)[:, 0]

print(f"bin count @ order=-3: n=5000 -> {len(_bin(data5k, -3)[0])}, "
      f"n=20000 -> {len(_bin(data20k, -3)[0])}")

CASES_5K = [
    ("npnormllw n=5000",  lambda: computemixdist(data5k, method="npnormllw")),
    ("npnormcvmw n=5000", lambda: computemixdist(data5k, method="npnormcvmw")),
    ("npnormadw  n=5000", lambda: computemixdist(data5k, method="npnormadw")),
    ("nptllw(b=inf) n=5000", lambda: computemixdist(data5k, method="nptllw")),
    ("nptllw(b=5)  n=5000", lambda: computemixdist(data5k, method="nptllw", beta=5.0)),
    ("estpi0 llw  n=5000", lambda: estpi0(data5k, method="npnormllw", val=2.0)),
    ("[ref] npnormll n=5000", lambda: computemixdist(data5k, method="npnormll")),
    ("[ref] npnormcvm n=5000", lambda: computemixdist(data5k, method="npnormcvm")),
    ("[ref] npnormad  n=5000", lambda: computemixdist(data5k, method="npnormad")),
    ("[ref] nptll(b=5) n=5000", lambda: computemixdist(data5k, method="nptll", beta=5.0)),
]
CASES_20K = [
    ("npnormllw n=20000", lambda: computemixdist(data20k, method="npnormllw")),
    ("nptllw(b=inf) n=20000", lambda: computemixdist(data20k, method="nptllw")),
    ("nptllw(b=5)  n=20000", lambda: computemixdist(data20k, method="nptllw", beta=5.0)),
    ("[ref] npnormll n=20000", lambda: computemixdist(data20k, method="npnormll")),
    ("[ref] nptll(b=5) n=20000", lambda: computemixdist(data20k, method="nptll", beta=5.0)),
]

for label, cases in (("n=5000", CASES_5K), ("n=20000", CASES_20K)):
    print(f"--- {label} ---")
    for name, fn in cases:
        fn()  # warmup (also catches failures early)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        print(f"{name:24s} median={statistics.median(ts)*1e3:9.2f} ms   "
              f"min={min(ts)*1e3:9.2f} ms")
raise SystemExit(0)
