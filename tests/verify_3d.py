"""3-D family (`npnormND`, k = 3) gate for the N-d generalization.

There is no R reference for the 3-D case (the R package's `npnorm2Dll`
is 2-D only), so the gate is structural + self-consistency, per the
acceptance criteria (the fitted density must be what the family
computes, and the fit must be sane):

  1. two runs bit-identical (determinism);
  2. the fit converges and the ll is bit-exact against the recorded
     value (trajectory guard — a kernel or data-layout regression
     would move it);
  3. self-consistency: `ll == -sum log dnpnormND(...)` (1e-9) using
     the PUBLIC ND kernel — an independent recomputation of the
     mixture density from the returned support points / weights /
     covariance;
  4. the returned support points are (k x 3), `beta` a (3 x 3) matrix,
     `family == "npnormND"`, and the `grid_gain` certificate is
     finite and > -1 (a LARGE negative value means the objective ran
     on a corrupted dataset — the 2-D row-major data-read lesson);
  5. the main components (w >= 0.05) sit near the three known
     generator modes within 0.3 (the data is a 3-component Gaussian
     mixture, see below — a data-layout bug would scatter the fit).

Data: the fixed `tests/parity_3d.csv` — three Gaussian components in
R^3 (120 points each, generator means (0,0,0) / (2,2,2) /
(-1.5,0.5,1.8), sigmas 0.35 / 0.30 / 0.40, seed 20261005). Grid: the
explicit 18-per-axis tensor product recorded in this file's comment
(17^3 = 4913 cells). Recorded result: ll = 1430.711238303976
(11 iters, 4 support points incl. the zero-weight fixed point).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from npfixedcomppy import computemixdist, dnpnormND  # noqa: E402

LL_RECORD = 1430.711238303976

X = np.loadtxt(os.path.join(HERE, "parity_3d.csv"), delimiter=",")
assert X.shape == (360, 3)
gp = np.arange(18.0) - 9.0
gpm = [list(map(float, row)) for row in np.column_stack([gp, gp, gp])]

bad = 0


def check(tag, cond, msg):
    global bad
    if not cond:
        bad += 1
        print(f"  [BAD] {tag}: {msg}")
    else:
        print(f"  [OK ] {tag}: {msg}")


r1 = computemixdist(X, method="npnormND", gridpoints=gpm)
r2 = computemixdist(X, method="npnormND", gridpoints=gpm)
check("determinism",
      r1.ll == r2.ll and r1.pt == r2.pt and r1.pr == r2.pr,
      f"two runs bit-identical (ll={r1.ll!r})")
check("converged", r1.convergence == 0,
      f"convergence={r1.convergence} (0)")
check("ll (bit-exact)", r1.ll == LL_RECORD,
      f"ll={r1.ll!r} vs recorded {LL_RECORD!r}")

pt = np.array(r1.pt)
pr = np.array(r1.pr)
check("shapes",
      pt.shape == (len(r1.pr), 3) and len(pr) == len(r1.pr),
      f"pt {pt.shape}, n_pr={len(pr)}")
check("beta shape", np.asarray(r1.beta).shape == (3, 3),
      f"beta {np.asarray(r1.beta).shape}")
check("family", r1.family == "npnormND", f"family={r1.family!r}")
check("grid_gain certificate",
      np.isfinite(r1.grid_gain) and r1.grid_gain > -1.0,
      f"grid_gain={r1.grid_gain!r} (recorded ~ +1.9e-3)")

dens = dnpnormND(X, pt, pr, np.asarray(r1.beta))
ll_self = -float(np.sum(np.log(dens)))
check("self-consistency",
      abs(ll_self - r1.ll) < 1e-9 * max(1.0, abs(r1.ll)),
      f"ll={r1.ll!r} vs recomputed {ll_self!r}")

keep = pr >= 0.05
main = pt[keep]
means = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [-1.5, 0.5, 1.8]])
dmax = 0.0
for m in means:
    if len(main):
        dmax = max(dmax, float(np.min(np.linalg.norm(main - m, axis=1))))
check("main components near generator modes",
      len(main) >= 3 and dmax < 0.30,
      f"n_main={len(main)} max dist to a generator mode={dmax:.6f} "
      f"(measured 0.1925)")

print(f"TOTAL BAD: {bad}")
sys.exit(0 if bad == 0 else 1)
