"""2-D family (`npnorm2Dll`) parity gate against recorded R 4.6.1 results.

Established facts (measured on this build; see the 2-D section of
`docs/PERF.md`):

* the 2-D kernel `dnpnormND` is **bit-exact** with R's — R's kernel
  evaluated at PY's final points gives 848.7472820349071, PY's kernel at
  R's final points gives 848.8877528768896, and an independent numpy
  recomputation matches both to < 1e-12;
* the family's L-BFGS-B support-point search lands on different — both
  valid — local optima on each side (the search accepts a grid cell only
  on a strictly negative gradient, which is exquisitely sensitive to the
  floating-point trajectory): R ll = 848.88775287688986 (6 iters),
  PY ll = 848.747282034907 (6 iters, slightly *better* NLL). Both
  implementations are deterministic on this data.

Gate (no external paths; gold files live in this directory):
  1. PY run from the recorded initial mix is deterministic (two runs
     bit-identical) and converges (convergence == 0);
  2. PY's reported `ll` equals `-sum log dnpnormND(X, pt_free, pr_free)`
     (self-consistency, 1e-9);
  3. PY's kernel evaluated at R's final points reproduces R's ll to 1e-9
     (the kernel-parity check, independent of the optimizer trajectory);
  4. PY's final NLL is not worse than R's (valid local optimum; in this
     run it is slightly better).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from npfixedcomppy import computemixdist, dnpnormND  # noqa: E402

# R 4.6.1, `npnorm2Dll_` on the same data / initial mix / grid (3 runs
# bit-identical on this data).
R_LL = 848.88775287688986

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

mix = {"pt": [list(map(float, row)) for row in ipt],
       "pr": [float(v) for v in ipr]}
gpv = [list(map(float, row)) for row in gp]

r1 = computemixdist(X, method="npnorm2Dll", mix=mix, gridpoints=gpv)
r2 = computemixdist(X, method="npnorm2Dll", mix=mix, gridpoints=gpv)

check("determinism",
      r1.ll == r2.ll and r1.pt == r2.pt and r1.pr == r2.pr,
      "two runs bit-identical (ll=%r)" % r1.ll)
check("converged", r1.convergence == 0, f"convergence={r1.convergence} (0)")
check("iter", r1.iter == 6, f"iter={r1.iter} (R: 6)")

# self-consistency: ll == -sum log density at the returned (free) points
ptf = np.array(r1.pt)
prf = np.array(r1.pr)
dens = dnpnormND(X, ptf, prf, np.array(r1.beta))
ll_self = -float(np.sum(np.log(dens)))
check("self-consistency",
      abs(ll_self - r1.ll) < 1e-9 * max(1.0, abs(r1.ll)),
      f"ll={r1.ll!r} vs recomputed {ll_self!r}")

# kernel parity: PY kernel at R's final (free) points == R's ll.
# R's `r3` rows include the trailing fixed (0,0) point (weight 0).
r3f, r3pf = r3[:-1], r3pr[:-1]
dens_r = dnpnormND(X, r3f, r3pf, np.eye(2))
ll_r_at = -float(np.sum(np.log(dens_r)))
check("kernel@R-pts",
      abs(ll_r_at - R_LL) < 1e-9 * max(1.0, R_LL),
      f"PY kernel at R's points: ll={ll_r_at!r} (R {R_LL!r})")

# valid (no worse) local optimum
check("no-worse-than-R",
      r1.ll <= R_LL + 1e-9,
      f"PY ll={r1.ll!r} <= R ll={R_LL!r}")

print(f"TOTAL BAD: {bad}")
sys.exit(0 if bad == 0 else 1)
