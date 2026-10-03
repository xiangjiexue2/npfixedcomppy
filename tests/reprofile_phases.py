"""Re-measure the docs/PERF.md §1 phase table on the CURRENT build.

Each case runs in a fresh subprocess with NPFIXEDCOMPY_PROFILE=1, three
times; the medians of the C++ PROFILE line are printed. Same real datasets
as tests/perf_baseline.py, so the table stays consistent with §4.

Usage: python tests/reprofile_phases.py
"""

import os
import re
import subprocess
import sys
from statistics import median

HERE = r"C:\Users\xxjie\Documents\rebuild\npfixedcomppy"
PY = HERE + r"\.venv\Scripts\python.exe"

SETUP = (
    "import os,numpy as np;from npfixedcomppy import computemixdist;"
    "d=r'" + HERE + r"\\tests';"
    "d1=np.loadtxt(d+r'\\npfc_data_1000.csv',delimiter=',',skiprows=1,ndmin=1);"
    "d5=np.loadtxt(d+r'\\npfc_data_5000.csv',delimiter=',',skiprows=1,ndmin=1);"
)

CASES = [
    ("nptll b=5 n=5000", "computemixdist(d5, method='nptll', beta=5)"),
    ("npnormcll n=1000",
     "computemixdist(np.tanh(d1), method='npnormcll', beta=len(d1))"),
    ("npnormadw n=5000", "computemixdist(d5, method='npnormadw')"),
    ("nptllw b=inf n=5000", "computemixdist(d5, method='nptllw')"),
    ("npnormllw n=5000", "computemixdist(d5, method='npnormllw')"),
    ("nptll b=inf n=5000", "computemixdist(d5, method='nptll')"),
    ("npnormll n=5000", "computemixdist(d5, method='npnormll')"),
    ("npnormad n=1000", "computemixdist(d1, method='npnormad')"),
]

# C++ prints:
#   PROFILE iters=31 total=2379.0ms  solvegrad=2119.6  mapping=0.3  loss=0.8
#             weights=8.9  collapse=249.5 evals=12345 (ms)
PAT = re.compile(
    r"PROFILE iters=(\d+) total=([\d.]+)ms  solvegrad=([\d.]+)  "
    r"mapping=([\d.]+)  loss=([\d.]+)  weights=([\d.]+)  collapse=([\d.]+) "
    r"evals=(\d+) \(ms\)")

print(f"{'case':20s} {'iters':>5s} {'total':>8s} {'solvegrad':>9s} "
      f"{'mapping':>7s} {'loss':>6s} {'weights':>7s} {'collapse':>8s} "
      f"{'evals':>7s}")
for name, snippet in CASES:
    rows = []
    for _ in range(3):
        proc = subprocess.run(
            [PY, "-c", SETUP + snippet],
            cwd=HERE,
            capture_output=True,
            text=True,
            env={**os.environ, "NPFIXEDCOMPY_PROFILE": "1"},
        )
        m = PAT.search(proc.stderr)
        if not m:
            print(f"{name}: NO PROFILE LINE (rc={proc.returncode})",
                  file=sys.stderr)
            print(proc.stderr[-2000:], file=sys.stderr)
            sys.exit(1)
        rows.append(tuple(float(v) for v in m.groups()))
    med = [median(r[i] for r in rows) for i in range(8)]
    print(f"{name:20s} {med[0]:5.0f} {med[1]:8.1f} {med[2]:9.1f} "
          f"{med[3]:7.1f} {med[4]:6.1f} {med[5]:7.1f} {med[6]:8.1f} "
          f"{med[7]:7.0f}")
