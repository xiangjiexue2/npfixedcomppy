"""2-D family (`npnorm2Dll`) parity/similarity gate against R 4.6.1.

Contract (relaxed in 0.2.1 from bit-exact matching to *similarity*; see
`docs/PERF.md` §4a.2): the per-cell objective has a **fast path**
(default) and a **R-identical path** (`NPFIC_2D_EXACT=1`). The fast path
uses an explicit-inverse quadratic form, preallocated buffers, and the
TRUE directional derivative of the objective (each point's gradient
contribution weighted by 1/(dens + pre); the R-ported unweighted form is
the gradient of the unnormalised KDE, which only aligns when 1/F is
~constant). Not bit-identical to the exact path, so the per-cell
accept/reject trajectory differs (more iters, fewer evals per iter —
see docs/PERF.md §4a.4). Both paths must therefore land on a *similar*
mixture: ll within a tight band, no worse than R, and the main
components in the same locations.

Established facts (measured on this build; see `docs/PERF.md`):

* the 2-D **kernel** `dnpnormND` is still **bit-exact** with R's — R's
  kernel evaluated at PY's exact-path points gives
  848.7472820349071, PY's kernel at R's final points gives
  848.8877528768896, and an independent numpy recomputation matches both
  to < 1e-12;
* R 4.6.1 (installed 1.1.0003) on this data: ll = 848.88775287688986
  (6 iters);
* the R-identical path (NPFIC_2D_EXACT=1) is bit-deterministic here:
  ll = 848.7472820349071 (6 iters, 5 support points incl. the zero-weight
  fixed point);
* the fast path (default, true directional derivative) is
  bit-deterministic here: ll = 848.7472072261477 (10 iters, 5 support
  points incl. the zero-weight fixed point); the older unweighted
  ("KDE-direction") fast path gave 848.7479272517137 (6 iters).

Gate:
  1. FAST (default): two runs bit-identical; converges; iter == 10; ll
     within 1e-4 of the recorded fast-path ll; self-consistent
     (`ll == -sum log dnpnormND(...)`, 1e-9); not worse than R.
  2. KERNEL parity (independent of the trajectory): PY's `dnpnormND` at
     R's final points reproduces R's ll to 1e-9.
  3. EXACT (`NPFIC_2D_EXACT=1`): two runs bit-identical; converges;
     iter == 6; ll bit-exact against the recorded value; not worse than
     R.
  4. SIMILARITY: fast vs exact main components (weights >= 0.02) —
     max location distance < 0.08 in both matching directions — and
     |Δll| < 2e-2.
  5. CERTIFICATE: both paths' grid_gain (the minimum gain over all
     (G-1)^2 cell midpoints at the final mixture, docs/PERF.md 4a.3)
     is > -1. A LARGE negative value means the objective ran on a
     corrupted dataset: the row-major (n,2) data-read bug drove
     grid_gain to ~ -518 while the similarity band (dmu < 0.08) still
     passed, so the similarity check alone cannot catch it. Small
     negatives (~1e-3) are the documented early-stop hot-start
     artifact and are acceptable — hence the -1 gate, not 0.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

from npfixedcomppy import computemixdist, dnpnormND  # noqa: E402

# R 4.6.1, `npnorm2Dll_` on the same data / initial mix / grid (3 runs
# bit-identical on this data).
R_LL = 848.88775287688986
# Recorded path results on this build (docs/PERF.md §4a.2).
EXACT_LL = 848.7472820349071
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


def main_components(r, wmin=0.02):
    pr = np.array(r.pr)
    keep = pr >= wmin
    return (np.array(r.pt)[keep], pr[keep])


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
check("fast ll", abs(f1.ll - FAST_LL) < 1e-4,
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

# --- 3. exact (R-identical) path ---------------------------------------
os.environ["NPFIC_2D_EXACT"] = "1"
try:
    e1 = computemixdist(X, method="npnorm2Dll", mix=mix, gridpoints=gpv)
    e2 = computemixdist(X, method="npnorm2Dll", mix=mix, gridpoints=gpv)
finally:
    del os.environ["NPFIC_2D_EXACT"]
check("exact determinism",
      e1.ll == e2.ll and e1.pt == e2.pt and e1.pr == e2.pr,
      f"two runs bit-identical (ll={e1.ll!r})")
check("exact converged", e1.convergence == 0,
      f"convergence={e1.convergence} (0)")
check("exact iter", e1.iter == 6, f"iter={e1.iter} (R: 6)")
check("exact ll (bit-exact)", e1.ll == EXACT_LL,
      f"ll={e1.ll!r} vs recorded {EXACT_LL!r}")
check("exact no-worse-than-R",
      e1.ll <= R_LL + 1e-9,
      f"exact ll={e1.ll!r} <= R ll={R_LL!r}")
check("exact grid_gain certificate",
      e1.grid_gain > -1.0,
      f"grid_gain={e1.grid_gain!r} (recorded ~ +2.7e-7)")

# --- 4. fast vs exact similarity ---------------------------------------
# Match components by LOCATION (nearest neighbour in both directions),
# not by weight rank — the two small off-diagonal components cross in
# weight between the paths while staying in the same location.
fp, fw = main_components(f1)
ep, ew = main_components(e1)
d = 0.0
if len(fp) and len(ep):
    for i in range(len(fp)):
        d = max(d, float(np.min(np.linalg.norm(ep - fp[i], axis=1))))
    for j in range(len(ep)):
        d = max(d, float(np.min(np.linalg.norm(fp - ep[j], axis=1))))
# Measured on this data (fast path with the true directional
# derivative): max location difference 0.0066, |dll| = 7.5e-5 (the
# older unweighted fast path: 0.026 / 6.5e-4; the pre-fix corrupted
# fast fit drifted ~0.06 — caught by the certificate checks above).
# Bands: dmu < 0.08, |dll| < 2e-2.
check("fast~exact mixture",
      len(fp) == len(ep) and d < 0.08 and abs(f1.ll - e1.ll) < 2e-2,
      f"n={len(fp)}/{len(ep)} max dmu={d:.6f} |dll|="
      f"{abs(f1.ll - e1.ll):.6f}")

print(f"TOTAL BAD: {bad}")
sys.exit(0 if bad == 0 else 1)
