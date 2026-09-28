"""Profile n=5000 computemixdist phases (NPFIXEDCOMPY_PROFILE=1).

Usage: set NPFIXEDCOMPY_PROFILE=1 && python tests\profile_n5000.py <norm|tinf>
Prints one PROFILE line (from the Rust engine) per run + a CASE summary.
"""
import os
import sys

os.environ.setdefault("NPFIXEDCOMPY_PROFILE", "1")
import numpy as np
from npfixedcomppy import computemixdist

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))

d = np.loadtxt(
    os.path.join(DATA_ROOT, "npfc_data_5000.csv"),
    delimiter=",", skiprows=1, ndmin=1,
)
case = sys.argv[1]
for rep in range(3):
    if case == "norm":
        r = computemixdist(d, method="npnormll")
    else:
        r = computemixdist(d, method="nptll", beta=float("inf"))
    print(f"CASE {case} rep{rep} ll={r.ll:.6f} iter={r.iter}", file=sys.stderr)
