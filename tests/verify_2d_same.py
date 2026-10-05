"""2-D family (`npnorm2Dll`) parity/similarity gate against R 4.6.1.

Contract (relaxed; see `docs/PERF.md` §4a): the per-cell objective has a
FAST path (default) — the explicit-inverse quadratic form, preallocated
buffers, the TRUE directional derivative of the objective (each point's
gradient contribution weighted by 1/(dens + pre)), the CNM working-set
warm start, and the negative-gain early stop. The R-identical
`NPFIC_2D_EXACT` path was removed when the 2-D family was generalized
to N dimensions (`npnormND`), so the gate is density-based, per the
acceptance criteria: the fitted density must agree with R's within
tolerance and the fitted ll must not be worse than R's.

Established facts (measured on this build; see `docs/PERF.md`):

* the multivariate **kernel** `dnpnormND` is **bit-exact** with R's —
  an independent numpy recomputation matches R's ll at PY's points to
  < 1e-12;
* R 4.6.1 (installed 1.1.0003) on this data: ll = 848.88775287688986
  (6 iters);
* the FAST path is bit-deterministic here: ll = 848.7472072261477
  (10 iters, 5 support points incl. the zero-weight fixed point);
* density similarity, FAST vs R at this data: relative L1 = 0.0158,
  max |log-density| difference = 0.169; the main components
  (w >= 0.02) sit in the same two modes, max location distance 0.675
  (the FAST fit keeps one extra small shoulder component — the
  components cross, so the location band is loose and the DENSITY is
  the primary check).

Gate:
  1. FAST (default): two runs bit-identical; converges; iter == 10; ll
     bit-exact against the recorded value; self-consistent
     (`ll == -sum log dnpnormND(...)`, 1e-9); not worse than R.
  2. KERNEL parity (independent of the trajectory): PY's `dnpnormND`
     at R's final points reproduces R's ll to 1e-9.
  3. DENSITY similarity vs R: relative L1 < 0.02, max |log diff| <
     0.25, main components (w >= 0.02) within 0.70 in both matching
     directions.
  4. CERTIFICATE: the grid_gain (the minimum gain over all (G-1)^2
     cell midpoints at the final mixture, docs/PERF.md §4a.3) is > -1.
     A LARGE negative value means the objective ran on a corrupted
     dataset: the row-major (n,2) data-read bug drove grid_gain to
     ~ -518 while the similarity band still passed, so the similarity
     check alone cannot catch it. Small negatives (~1e-3) are the
     documented early-stop hot-start artifact and are acceptable —
     hence the -1 gate, not 0.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from npfixedcomppy import computemixdist, dnpnormND  # noqa: E402

# R 4.6.1, `npnorm2Dll_` on the same data / initial mix / grid (3 runs
# bit-identical on this data).
R_LL = 848.88775287688986
# Recorded FAST-path result on this build (docs/PERF.md §4a.2).
FAST_LL = 848.7472072261477

X = np.loadtxt(os.path.join(HERE, "parity_2d.csv"), delimiter=",")
ipt = np.loadtxt(os.path.join(HERE, "py_initpt_2d.csv"), delimiter=",")
ipr = np.loadtxt(os.path.join(HERE, "py_initpr_2d.txt"), delimiter=",").ravel()
gp = np.loadtxt(os.path.join(HERE, "grid_2d.csv"), delimiter=",")
r3 = np.loadtxt(os.path.join(HERE, "r3_pts_2d.csv"), delimiter=",")
r3pr = np.loadtxt(os.path.join(HERE, "r3_pr_2d.txt"), delimiter=",").ravel()

bad = 0


def check(tag, cond, msg):
    global bad
    if not cond:
        bad += 1
        print(f"  [BAD] {tag}: {msg}")
    else:
        print(f"  [OK ] {tag}: {msg}")


def main_components(pt, pr, wmin=0.02):
    pr = np.asarray(pr)
    keep = pr >= wmin
    return np.asarray(pt)[keep], pr[keep]


mix = {"pt": [list(map(float, row)) for row in ipt],
       "pr": [float(v) for v in ipr]}
gpv = [list(map(float, row)) for row in gp]

# --- 1. fast path (default) -------------------------------------------
f1 = computemixdist(X, method="npnorm2Dll", mix=mix, gridpoints=gpv)
f2 = computemixdist(X, method="npnorm2Dll", mix=mix, gridpoints=gpv)
check("fast determinism",
      f1.ll == f2.ll and f1.pt == f2.pt and f1.pr == f2.pr,
      f"two runs bit-identical (ll={f1.ll!r})")
check("fast converged", f1.convergence == 0,
      f"convergence={f1.convergence} (0)")
check("fast iter", f1.iter == 10, f"iter={f1.iter} (R: 6)")
check("fast ll (bit-exact)", f1.ll == FAST_LL,
      f"ll={f1.ll!r} vs recorded {FAST_LL!r}")
check("fast grid_gain certificate",
      f1.grid_gain > -1.0,
      f"grid_gain={f1.grid_gain!r} (recorded ~ +4.7e-3; the "
      f"row-major data-read bug drove it to ~ -518 while the "
      f"similarity band still passed)")

ptf = np.array(f1.pt)
prf = np.array(f1.pr)
dens = dnpnormND(X, ptf, prf, np.array(f1.beta))
ll_self = -float(np.sum(np.log(dens)))
check("fast self-consistency",
      abs(ll_self - f1.ll) < 1e-9 * max(1.0, abs(f1.ll)),
      f"ll={f1.ll!r} vs recomputed {ll_self!r}")

# --- 2. kernel parity (trajectory-independent) ------------------------
r3f, r3pf = r3[:-1], r3pr[:-1]
dens_r = dnpnormND(X, r3f, r3pf, np.eye(2))
ll_r_at = -float(np.sum(np.log(dens_r)))
check("kernel@R-pts",
      abs(ll_r_at - R_LL) < 1e-9 * max(1.0, R_LL),
      f"PY kernel at R's points: ll={ll_r_at!r} (R {R_LL!r})")

check("fast no-worse-than-R",
      f1.ll <= R_LL + 1e-9,
      f"fast ll={f1.ll!r} <= R ll={R_LL!r}")

# --- 3. density similarity vs R ---------------------------------------
# The trajectory differs (10 vs 6 iters; the FAST fit keeps an extra
# small shoulder component), so the contract is on the DENSITY, not on
# the components. Measured on this build: rel-L1 = 0.0158,
# max |logdiff| = 0.169, main-component location distance 0.675.
d_fast = dnpnormND(X, ptf, prf, np.array(f1.beta))
d_r = dnpnormND(X, r3f, r3pf, np.eye(2))
rel_l1 = float(np.abs(d_fast - d_r).sum() / np.abs(d_r).sum())
logdiff = float(np.max(np.abs(np.log(d_fast) - np.log(d_r))))
check("density rel-L1 vs R",
      rel_l1 < 0.02,
      f"rel-L1={rel_l1:.6f} (measured 0.0158)")
check("density max |logdiff| vs R",
      logdiff < 0.25,
      f"max |logdiff|={logdiff:.6f} (measured 0.169)")
fp, _fw = main_components(ptf, prf)
ep, _ew = main_components(r3f, r3pf)
d = 0.0
if len(fp) and len(ep):
    for i in range(len(fp)):
        d = max(d, float(np.min(np.linalg.norm(ep - fp[i], axis=1))))
    for j in range(len(ep)):
        d = max(d, float(np.min(np.linalg.norm(fp - ep[j], axis=1))))
check("main components vs R",
      d < 0.70,
      f"n={len(fp)}/{len(ep)} max dmu={d:.6f} (measured 0.675)")

print(f"TOTAL BAD: {bad}")
sys.exit(0 if bad == 0 else 1)
