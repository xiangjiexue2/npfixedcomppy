"""Measure npnorm2Dll wall time (median of 5, warmup excluded) on the
reference data, for whichever objective path the environment selects
(default: fast; NPFIC_2D_EXACT=1: the R-identical exact path). No
asserts — this is a measurement script, not a gate (that is
verify_2d_same.py)."""

import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from npfixedcomppy import computemixdist  # noqa: E402

tag = "EXACT" if os.environ.get("NPFIC_2D_EXACT") else "FAST "
X = np.loadtxt(os.path.join(HERE, "parity_2d.csv"), delimiter=",")

r = computemixdist(X, method="npnorm2Dll")  # warmup
ts = []
for _ in range(5):
    t0 = time.perf_counter()
    r = computemixdist(X, method="npnorm2Dll")
    ts.append((time.perf_counter() - t0) * 1e3)
print("==%s== median=%.1f ms  all=%s  ll=%r  iter=%d  k=%d" %
      (tag, float(np.median(ts)), ["%.1f" % t for t in ts],
       r.ll, r.iter, len(r.pt)))
