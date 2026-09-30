"""Verify npfixedcomppy posterior mean against R npfixedcomp2.

The posterior mean of fun(pt) under a fitted mixing distribution
G = sum_j pi_j Delta_{pt_j} and the family's kernel K(x; mu, beta) is

    out[i] = sum_j K(x_i; pt_j, beta) pi_j f(pt_j) / sum_j K(x_i; pt_j) pi_j

— the R `posteriormean` dispatcher (utility.R). The kernel follows the
family of the fit (npnorm / npt / npnormc / nppois).

Design: the posterior mean is a PURE function of (x, pt, pr, beta, family),
and R's npnormll fit is non-reproducible across runs (OpenMP reductions on a
flat NPMLE surface — see verify_npnormll.py). So the gold generator
(gen_posteriors_ref.R) records per case R's EXACT fit params
(family, beta, pt, pr) plus the posterior means R computed from them:

  npfc_pm_fit.csv : tag, family, beta, k, pt, pr   (pt/pr semicolon-joined)
  npfc_pm_ref.csv : tag, idx, idval, tanval

This test feeds R's exact params into its own posteriormean (validating the
code path in isolation, with a strict 1e-9 relative gate) and, for the
deterministic families (npt beta=Inf, nppois), additionally refits and
checks the end-to-end value against the gold.
"""
import csv
import os

import numpy as np
from npfixedcomppy import Npmix, computemixdist, posteriormean

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))

fits = {}
with open(os.path.join(DATA_ROOT, "npfc_pm_fit.csv"), newline="") as fh:
    for r in csv.DictReader(fh):
        fits[r["tag"]] = dict(
            family=r["family"],
            beta=float(r["beta"]),
            pt=[float(v) for v in r["pt"].split(";")],
            pr=[float(v) for v in r["pr"].split(";")],
        )

gold = {}
with open(os.path.join(DATA_ROOT, "npfc_pm_ref.csv"), newline="") as fh:
    for r in csv.DictReader(fh):
        gold.setdefault(r["tag"], {})[int(r["idx"])] = (
            float(r["idval"]),
            float(r["tanval"]),
        )

d = np.loadtxt(os.path.join(DATA_ROOT, "npfc_data_1000.csv"),
               delimiter=",", skiprows=1, ndmin=1)
p = np.loadtxt(os.path.join(DATA_ROOT, "npfc_data_pois.csv"),
               delimiter=",", skiprows=1, ndmin=1)

# tag -> (x, refit result or None)
CASES = [
    ("npnorm:cm",  d,      None),
    ("npnorm:ep",  d,      None),
    ("npt:cm",     d,      computemixdist(d, method="nptll")),
    ("npnormc:cm", np.tanh(d), None),
    ("nppois:cm",  p,      computemixdist(p, method="nppoisll")),
]

def relerr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)
    return np.abs(a - b) / denom

bad = 0

# ---- 1. pure-function parity: feed R's exact fit params ---------------
for tag, x, _ in CASES:
    f = fits[tag]
    r = Npmix(pt=f["pt"], pr=f["pr"], beta=f["beta"], family=f["family"])
    ref = np.array([gold[tag][i] for i in range(len(x))])
    got_id = posteriormean(x, r)
    got_tan = posteriormean(x, r, fun=np.tanh)
    re_id = float(np.max(relerr(got_id, ref[:, 0])))
    re_tan = float(np.max(relerr(got_tan, ref[:, 1])))
    ok = re_id < 1e-9 and re_tan < 1e-9
    print(f"  [{'OK ' if ok else 'BAD'}] {tag} pure: n={len(x)} k={len(r.pt)} "
          f"id relerr={re_id:.3e} tan relerr={re_tan:.3e} (tol 1e-9)")
    if not ok:
        bad += 1

# ---- 2. end-to-end: deterministic families refit ----------------------
for tag, x, refit in CASES:
    if refit is None:
        continue
    f = fits[tag]
    # 1e-6: this is a cross-run fit difference (the R gold was generated in a
    # separate R session; even the "deterministic" nptll drifts by ~6e-8 in a
    # support point between runs). The fit itself is strictly validated
    # against its own same-run gold in verify_nptll.py / verify_pois.py;
    # here we only need the refit close enough for the posterior-mean
    # comparison below to be meaningful.
    re_pt = float(np.max(np.abs(np.asarray(refit.pt) - np.asarray(f["pt"]))))
    ok_fit = re_pt < 1e-6
    print(f"  [{'OK ' if ok_fit else 'BAD'}] {tag} refit: max|dpt|={re_pt:.3e} "
          f"(cross-run drift, tol 1e-6)")
    if not ok_fit:
        bad += 1
        continue
    got_id = posteriormean(x, refit)
    ref_id = np.array([gold[tag][i][0] for i in range(len(x))])
    re_id = float(np.max(relerr(got_id, ref_id)))
    ok = re_id < 1e-6
    print(f"  [{'OK ' if ok else 'BAD'}] {tag} e2e: id relerr={re_id:.3e} (tol 1e-6)")
    if not ok:
        bad += 1

# ---- 3. determinism + monotonicity sanity -----------------------------
r0 = Npmix(pt=fits["npnorm:cm"]["pt"], pr=fits["npnorm:cm"]["pr"],
           beta=fits["npnorm:cm"]["beta"], family="npnorm")
a = posteriormean(d, r0)
b = posteriormean(d, r0)
if not np.array_equal(a, b):
    bad += 1
    print("  [BAD] determinism: two calls differ")
else:
    print("  [OK ] determinism: bit-identical re-run")

pm = posteriormean(np.sort(d), r0)
if np.any(np.diff(pm) < -1e-12):
    bad += 1
    print("  [BAD] monotonicity: identity posterior mean decreases on sorted x")
else:
    print("  [OK ] monotonicity: identity posterior mean non-decreasing")

print()
print("TOTAL BAD:", bad)
raise SystemExit(1 if bad else 0)
