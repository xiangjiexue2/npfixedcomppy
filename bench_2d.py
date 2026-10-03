import time
import numpy as np
from npfixedcomppy import computemixdist

base = r'C:/Users/xxjie/Documents/rebuild'
X = np.loadtxt(base + '/parity_2d.csv', delimiter=',')

# warmup
computemixdist(X, method="npnorm2Dll")
ts = []
for _ in range(5):
    t0 = time.perf_counter()
    r = computemixdist(X, method="npnorm2Dll")
    ts.append((time.perf_counter() - t0) * 1e3)
    # Default (fast) path ll; the NPFIC_2D_EXACT=1 R-identical path gives
    # 848.7472820349071 on the same data (see tests/verify_2d_same.py).
    assert abs(r.ll - 848.7611702303334) < 1e-6, r.ll
print("PY npnorm2Dll n=300 (2D): median = %.1f ms  (all: %s)" %
      (float(np.median(ts)), ["%.1f" % t for t in ts]))
print("k =", len(r.pr), " ll =", repr(r.ll), " iter =", r.iter,
      " conv =", r.convergence)
