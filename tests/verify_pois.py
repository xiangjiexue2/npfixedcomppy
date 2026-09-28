"""Verify npfixedcomppy (Rust) nppoisll + estpi0 against R npfixedcomp2.

Data: npfc_data_pois.csv (1000 draws from a Poisson mixture, means {0, 2},
50/50, R seed 12345 — generated once by probe_r_pois.R so both sides fit
IDENTICAL data; R's `rpois` and NumPy's `Generator.poisson` differ, so the
probe data must come from a shared file, not a shared seed).

R reference behaviour (deterministic, 2 runs identical on this data):
  - computemixdist nppoisll: ll=1369.9230505309843, npt=3, iter=13,
      pt=[0, 0.0165991179091, 2.15205466691],
      pr=[0, 0.510511051409, 0.489488948591], fam=nppois flag=d0.
  - estpi0 nppoisll (val=2):  ll=1371.9230567929201, npt=2, iter=2,
      pt=[0, 2.15806154493], pr=[0.534509825673, 0.465490174327].
    estpi0 refines the point-mass weight by bisection to hit the
    hypothesis statistic, so the last digits differ from R (Python lands
    3.9e-5 off on the free support point, ll relerr ~5e-9); the IDENTIFIED
    quantity is the invariant  ll_ep - ll_cm ~= val, gated at 1e-4.
"""
import os

import numpy as np
from npfixedcomppy import computemixdist, estpi0

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))

data = np.loadtxt(os.path.join(DATA_ROOT, "npfc_data_pois.csv"),
                  delimiter=",", skiprows=1, ndmin=1)

R_CM = dict(ll=1369.9230505309843, npt=3, it=13,
            pt=[0.0, 0.0165991179091, 2.15205466691],
            pr=[0.0, 0.510511051409, 0.489488948591])
R_EP = dict(ll=1371.9230567929201, npt=2, it=2,
            pt=[0.0, 2.15806154493],
            pr=[0.534509825673, 0.465490174327])
VAL = 2.0

def relerr(a, b):
    denom = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denom

bad = 0

def check(tag, cond, msg):
    global bad
    if not cond:
        bad += 1
        print(f"  [BAD] {tag}: {msg}")
    else:
        print(f"  [OK ] {tag}: {msg}")

def sanity(tag, r):
    pt = np.asarray(r.pt, dtype=float)
    pr = np.asarray(r.pr, dtype=float)
    check(tag, np.all(np.isfinite(pt)) and np.all(np.isfinite(pr)), "pt/pr finite")
    check(tag, np.all(np.diff(pt) >= -1e-12), "pt sorted")
    check(tag, np.all(pr >= -1e-12), "pr >= 0")
    check(tag, abs(pr.sum() - 1.0) < 1e-8, f"sum(pr)={pr.sum()!r} ~= 1")

# ---- 1. computemixdist nppoisll (R deterministic: strict gold) -----------
r = computemixdist(data, method="nppoisll")
print(f"== CM nppoisll  (R deterministic strict)")
print(f"   ll={r.ll!r} npt={len(r.pt)} iter={r.iter} conv={r.convergence} "
      f"mg={r.min_gradient!r} fam={r.family} flag={r.flag}")
check("CM", relerr(r.ll, R_CM["ll"]) < 1e-9, f"ll relerr={relerr(r.ll, R_CM['ll']):.3e}")
check("CM", len(r.pt) == R_CM["npt"], f"npt={len(r.pt)} == {R_CM['npt']}")
check("CM", r.iter == R_CM["it"], f"iter={r.iter} == {R_CM['it']}")
check("CM", r.family == "nppois" and r.flag == "d0", "family=nppois flag=d0")
check("CM", r.convergence == 0, "conv=0")
if len(r.pt) == R_CM["npt"]:
    dpt = max(abs(a - b) for a, b in zip(r.pt, R_CM["pt"]))
    dpr = max(abs(a - b) for a, b in zip(r.pr, R_CM["pr"]))
    check("CM", dpt < 1e-8, f"max|dpt|={dpt:.3e}")
    check("CM", dpr < 1e-8, f"max|dpr|={dpr:.3e}")
sanity("CM", r)

# ---- 2. estpi0 nppoisll (refined to hit val: ll + invariant gate) --------
e = estpi0(data, method="nppoisll", val=VAL)
print(f"== estpi0 nppoisll (val={VAL}; bisection to the statistic)")
print(f"   ll={e.ll!r} npt={len(e.pt)} iter={e.iter} conv={e.convergence} "
      f"mg={e.min_gradient!r} fam={e.family} flag={e.flag}")
check("EP", relerr(e.ll, R_EP["ll"]) < 1e-6, f"ll relerr={relerr(e.ll, R_EP['ll']):.3e}")
check("EP", len(e.pt) == R_EP["npt"], f"npt={len(e.pt)} == {R_EP['npt']}")
check("EP", e.iter == R_EP["it"], f"iter={e.iter} == {R_EP['it']}")
check("EP", e.family == "nppois" and e.flag == "d0", "family=nppois flag=d0")
check("EP", e.convergence == 0, "conv=0")
inv = e.ll - r.ll
check("EP", abs(inv - VAL) < 1e-4, f"ll_ep - ll_cm = {inv:.9g} ~= val={VAL} (d={abs(inv - VAL):.3e})")
sanity("EP", e)

# ---- 3. determinism re-runs (bit-identical) ------------------------------
r2 = computemixdist(data, method="nppoisll")
check("CM-rerun", (r2.ll == r.ll) and (r2.pt == r.pt) and (r2.pr == r.pr),
      "CM deterministic re-run bit-identical")
e2 = estpi0(data, method="nppoisll", val=VAL)
check("EP-rerun", (e2.ll == e.ll) and (e2.pt == e.pt) and (e2.pr == e.pr),
      "EP deterministic re-run bit-identical")

print()
print("TOTAL BAD:", bad)
raise SystemExit(1 if bad else 0)
