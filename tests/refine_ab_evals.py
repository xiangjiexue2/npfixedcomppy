"""evals capture for the NPFIC_REFINE_STEPS A/B (companion to refine_ab.py).

The gradient-evaluation count lives only in the C++ PROFILE line (stderr),
so each arm runs in a FRESH subprocess with NPFIXEDCOMPY_PROFILE=1. The
trajectory is kept identical to refine_ab.py (same seed 42, same data —
deliberately UNSORTED, same beta, 3 timed runs); medians are reported per
arm.

Usage: python tests/refine_ab_evals.py [n]  (default n=30000)
"""

import os
import re
import subprocess
import sys
from statistics import median

HERE = r"C:\Users\xxjie\Documents\rebuild\npfixedcomppy"
PY = HERE + r"\.venv\Scripts\python.exe"

n = int(sys.argv[1]) if len(sys.argv) > 1 else 30000

ARMS = [(-1, "unlimited (shipped)"), (3, "cap=3"), (2, "cap=2"), (1, "cap=1")]
METHODS = (("nptll", 3.0), ("npnormll", 1.0))

BODY = (
    "import time,numpy as np\n"
    "from npfixedcomppy import computemixdist\n"
    "rng=np.random.default_rng(42)\n"
    "x=rng.standard_normal(%d)\n"
    "beta=%s\n"
    "for _ in range(3):\n"
    "    t0=time.perf_counter()\n"
    "    r=computemixdist(x, method='%s', beta=beta)\n"
    "    print(f'WALL {time.perf_counter()-t0:.4f} ITER {r.iter} "
    "K {len(r.pt)} LL {r.ll:.6f}')\n"
)

EVALS = re.compile(r"evals=(\d+)")
WALL = re.compile(r"WALL ([\d.]+) ITER (\d+) K (\d+) LL ([\d.]+)")

print(f"n={n}")
print(f"{'method':9s} {'arm':22s} {'wall_ms':>9s} {'iter':>4s} "
      f"{'evals_med':>10s} {'evals (3 runs)':>24s} {'ll':>14s}")
for method, beta in METHODS:
    for cap, label in ARMS:
        env = {**os.environ, "NPFIXEDCOMPY_PROFILE": "1"}
        if cap < 0:
            env.pop("NPFIC_REFINE_STEPS", None)
        else:
            env["NPFIC_REFINE_STEPS"] = str(cap)
        code = BODY % (n, repr(beta), method)
        proc = subprocess.run([PY, "-c", code], cwd=HERE,
                              capture_output=True, text=True, env=env)
        walls = WALL.findall(proc.stdout)
        evals = [int(v) for v in EVALS.findall(proc.stderr)]
        if len(walls) != 3 or len(evals) != 3:
            print(proc.stdout, file=sys.stderr)
            print(proc.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"{method}/{label}: expected 3 WALL + 3 evals "
                             f"lines, got {len(walls)}/{len(evals)}")
        wm = median(float(w[0]) for w in walls) * 1e3
        it = walls[-1][1]
        ll = walls[-1][3]
        print(f"{method:9s} {label:22s} {wm:9.1f} {it:>4s} "
              f"{median(evals):10.0f} "
              f"{','.join(str(e) for e in evals):>24s} {ll:>14s}")
