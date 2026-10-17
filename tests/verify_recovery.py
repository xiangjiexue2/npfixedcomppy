"""Self-contained recovery gate: does the fit recover the TRUE mixture?

Replaces the old R-parity fit-quality gate.  We sample from a KNOWN mixing
distribution (a 3-component normal mixture, fixed seed), fit the
non-parametric mixing distribution with the NPMLE (``npnormll``) and NPMDE
(``npnormcvm`` / ``npnormad``) estimators, and check that the returned
mixture is close to the truth we generated — independent of any R reference.

Per estimator we gate:
  1. the mixture DENSITY the fit induces,
     ``d_fit(x) = sum_j pr_j N(x; pt_j, beta)``, matches the true density on
     a grid (max absolute and L1 distance);
  2. the main fitted components (``pr >= 0.05``) sit near the three true
     modes — a data-layout or kernel regression would scatter them;
  3. the KKT certificate also holds (``convergence == 0`` and
     ``min_gradient >= -TOL``).
The fitted weights are reported for information (the non-parametric fit may
split or merge minor mass, so they are not hard-gated).
"""
import os
import sys

import numpy as np

from npfixedcomppy import computemixdist, dnpnorm

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260202
N = 4000

# ---- the TRUE mixing distribution (ground truth) -------------------------
TRUE_MEANS = np.array([-1.5, 0.0, 1.8])
TRUE_W = np.array([0.3, 0.4, 0.3])
BETA = 1.0          # the known within-component stdev (the family beta)

DENS_MAX_TOL = 0.03   # max |d_fit - d_true|
DENS_L1_TOL = 0.05    # integral |d_fit - d_true|
# Sanity check, not the strict gate (the density gate above is): the
# distance estimators (CVM/AD) may split a true mode into several support
# points, so this is kept loose — a data-layout / kernel regression scatters
# the components by >> this, and is still caught.
MODE_TOL = 0.45       # each true mode has a main component within this
KKT_TOL = 1e-3

bad = 0


def check(tag, cond, msg):
    global bad
    if not cond:
        bad += 1
        print(f"  [BAD] {tag}: {msg}")
    else:
        print(f"  [OK ] {tag}: {msg}")


# ---- sample from the known mixture (fixed seed) --------------------------
rng = np.random.default_rng(SEED)
comp = rng.choice(len(TRUE_MEANS), size=N, p=TRUE_W)
data = rng.normal(np.asarray(TRUE_MEANS)[comp], BETA)

# ---- the true density on a shared grid -----------------------------------
grid = np.linspace(-5.0, 5.0, 2001)
dx = grid[1] - grid[0]
d_true = np.zeros_like(grid)
for m, w in zip(TRUE_MEANS, TRUE_W):
    d_true += w * np.exp(-0.5 * ((grid - m) / BETA) ** 2) / (BETA * np.sqrt(2 * np.pi))

print("== recovery of the true mixing distribution ==")
print(f"   true means={TRUE_MEANS.tolist()} weights={TRUE_W.tolist()} "
      f"beta={BETA}  n={N}, seed={SEED}")
print()

for method in ("npnormll", "npnormcvm", "npnormad"):
    print(f"  -- {method} --")
    r = computemixdist(data, method=method, beta=BETA)
    pt = np.asarray(r.pt)
    pr = np.asarray(r.pr)
    d_fit = np.asarray(dnpnorm(grid, pt, pr, BETA))

    dmax = float(np.max(np.abs(d_fit - d_true)))
    dl1 = float(np.sum(np.abs(d_fit - d_true)) * dx)
    check(
        f"{method}.density",
        dmax < DENS_MAX_TOL and dl1 < DENS_L1_TOL,
        f"max|d_fit-d_true|={dmax:.4f} (<{DENS_MAX_TOL})  "
        f"L1={dl1:.4f} (<{DENS_L1_TOL})",
    )

    keep = pr >= 0.05
    main = pt[keep]
    dmax_mode = 0.0
    for m in TRUE_MEANS:
        if len(main):
            dmax_mode = max(dmax_mode, float(np.min(np.abs(main - m))))
    check(
        f"{method}.modes",
        len(main) >= 3 and dmax_mode < MODE_TOL,
        f"n_main={len(main)}  max dist to a true mode={dmax_mode:.4f} "
        f"(<{MODE_TOL})",
    )

    check(
        f"{method}.kkt",
        (r.convergence == 0) and (r.min_gradient >= -KKT_TOL),
        f"conv={r.convergence} min_gradient={r.min_gradient:+.3e} "
        f"(>= -{KKT_TOL:g})  npt={len(r.pt)} iter={r.iter}",
    )

    w_sorted = sorted(pr[keep])
    print(f"         info: fitted weights (main) = "
          f"{[f'{w:.3f}' for w in w_sorted]}  (true = "
          f"{[f'{w:.3f}' for w in sorted(TRUE_W)]})")

print()
print("TOTAL BAD:", bad)
sys.exit(0 if bad == 0 else 1)
