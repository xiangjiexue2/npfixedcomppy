"""Fixed-point-step A/B (task #4).

Question: does capping the per-candidate refinement inside brmin/dfmin
("less work per step, hopefully fewer outer iterations") beat the shipped
behaviour ("full refinement per step") in TOTAL wall time and ll?

Arms differ ONLY in NPFIC_REFINE_STEPS (default -1 = unlimited = shipped).
Data / initial points / grid / tol are identical across arms; each arm is a
fresh solver build (the env knob is read in the MixSolver constructor).

Usage: python tests/refine_ab.py [n]  (default n=30000)
"""

import os
import sys
import time

import numpy as np

from npfixedcomppy import computemixdist

n = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
rng = np.random.default_rng(42)
x = rng.standard_normal(n)

# refine steps per candidate: -1 = unlimited (shipped), 1..3 = capped arms
ARMS = [(-1, "unlimited (shipped)"), (3, "cap=3"), (2, "cap=2"), (1, "cap=1")]
METHODS = (("nptll", 3.0), ("npnormll", 1.0))

rows = []
for method, beta in METHODS:
    for cap, label in ARMS:
        if cap < 0:
            os.environ.pop("NPFIC_REFINE_STEPS", None)
        else:
            os.environ["NPFIC_REFINE_STEPS"] = str(cap)
        # 3 timed runs, take the median (machine-noise robust)
        walls = []
        for _ in range(3):
            t0 = time.perf_counter()
            r = computemixdist(x, method=method, beta=beta)
            walls.append(time.perf_counter() - t0)
        rows.append(
            (method, label, float(np.median(walls)), int(r.iter), len(r.pt),
             r.ll, int(r.convergence)))

os.environ.pop("NPFIC_REFINE_STEPS", None)
print(f"n={n}")
print(f"{'method':9s} {'arm':22s} {'wall_ms':>9s} {'iter':>5s} {'k':>3s} "
      f"{'ll':>18s} {'conv':>4s}")
for method, label, wall, iters, k, ll, conv in rows:
    print(f"{method:9s} {label:22s} {wall * 1e3:9.1f} {iters:5d} {k:3d} "
          f"{ll:18.6f} {conv:4d}")
