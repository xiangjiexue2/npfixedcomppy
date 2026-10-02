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
    assert abs(r.ll - 848.7472820349067) < 1e-6, r.ll
print("PY npnorm2Dll n=300 (2D): median = %.1f ms  (all: %s)" %
      (float(np.median(ts)), ["%.1f" % t for t in ts]))
print("k =", len(r.pr), " ll =", repr(r.ll), " iter =", r.iter,
      " conv =", r.convergence)
