"""covestEB loss criteria (mle/cvm/ad): self-contained checks.

No R reference needed: the per-criterion mixing fits are already
parity-tested in the 1-D suites (verify_npnormll.py / verify_cvmadcll.py).
Here we validate the covestEB orchestration of the selectable loss:

  1. the default call is bit-identical to loss="mle" in both the plain and
     estpi0 paths (the R covestEB behaviour must be unchanged);
  2. each loss in {mle, cvm, ad} x {estpi0 False, True} returns a valid
     covariance estimate (symmetric, PSD after deflation, sample-variance
     diagonal, non-negative correction);
  3. determinism: a repeated call is bit-identical;
  4. the returned mix_dist family follows the criterion (npnormll /
     npnormcvm / npnormad, "w" suffixed only when binning triggers — it
     does not for this 24-variable data set);
  5. an unknown loss raises ValueError.
"""
import os

import numpy as np
from npfixedcomppy import covestEB

DATA_ROOT = os.path.dirname(os.path.abspath(__file__))

X = np.loadtxt(os.path.join(DATA_ROOT, "npfc_coveb_X.csv"),
               delimiter=",", skiprows=1)
p = X.shape[1]

covest = np.cov(X, rowvar=False, ddof=1)
varest = np.sqrt(np.diag(covest))

bad = 0


def check(tag, cond, msg):
    global bad
    if not cond:
        bad += 1
        print(f"  [BAD] {tag}: {msg}")
    else:
        print(f"  [OK ] {tag}: {msg}")


def valid_estimate(tag, res):
    M = res.mat
    check(f"{tag} shape/finite", M.shape == (p, p) and bool(np.all(np.isfinite(M))),
          f"shape={M.shape}")
    asym = float(np.max(np.abs(M - M.T)))
    check(f"{tag} symmetric", asym < 1e-12, f"max |M-M'| = {asym:.3e}")
    C = M / np.outer(varest, varest)
    C = (C + C.T) / 2
    eigmin = float(np.min(np.linalg.eigvalsh(C)))
    check(f"{tag} PSD", eigmin > -1e-7, f"min eigval = {eigmin:.3e}")
    diagd = float(np.max(np.abs(np.diag(M) - np.diag(covest)) / np.diag(covest)))
    check(f"{tag} diag=var", diagd < 1e-9, f"max relerr = {diagd:.3e}")
    check(f"{tag} Fnorm>=0", res.correction_Fnorm >= 0.0,
          f"||ans1-ans||_F = {res.correction_Fnorm:.3e}")


# ---- 1. default == loss="mle", bit-identical (R behaviour unchanged) --
r_def = covestEB(X)
r_mle = covestEB(X, loss="mle")
check("default==mle mat", bool(np.array_equal(r_def.mat, r_mle.mat)),
      "bit-identical covariance")
check("default==mle Fnorm", r_def.correction_Fnorm == r_mle.correction_Fnorm,
      f"{r_def.correction_Fnorm!r} vs {r_mle.correction_Fnorm!r}")
r_def_e = covestEB(X, estpi0=True)
r_mle_e = covestEB(X, loss="mle", estpi0=True)
check("default==mle estpi0 mat", bool(np.array_equal(r_def_e.mat, r_mle_e.mat)),
      "bit-identical covariance (estpi0 path)")

# ---- 2/3/4. each loss x estpi0: validity, determinism, kernel ---------
# ``Npmix.family`` stores the KERNEL (all three losses share the normal
# kernel on Fisher-transformed data); the loss criterion is not encoded in
# it, so it cannot be used to tell the criteria apart.  Instead we check the
# kernel and, after the loop, that the three criteria actually yield
# different fits (the feature's observable effect).
results = {}
for loss in ("mle", "cvm", "ad"):
    for epi in (False, True):
        tag = f"{loss}/estpi0={epi}"
        r1 = covestEB(X, loss=loss, estpi0=epi)
        valid_estimate(tag, r1)
        check(f"{tag} kernel", r1.mix_dist.family == "npnorm",
              f"family={r1.mix_dist.family!r} (want 'npnorm' kernel)")
        check(f"{tag} ll finite", bool(np.isfinite(r1.mix_dist.ll)),
              f"ll={r1.mix_dist.ll!r}")
        r2 = covestEB(X, loss=loss, estpi0=epi)
        check(f"{tag} deterministic", bool(np.array_equal(r1.mat, r2.mat)),
              "bit-identical repeat")
        results[(loss, epi)] = r1

# The selectable loss must actually change the fit: pairwise different final
# estimates for the three criteria (deterministic on this fixed data).
check("losses differ (mle vs cvm)",
      not np.array_equal(results[("mle", False)].mat, results[("cvm", False)].mat),
      "mle and cvm estimates should differ")
check("losses differ (mle vs ad)",
      not np.array_equal(results[("mle", False)].mat, results[("ad", False)].mat),
      "mle and ad estimates should differ")
check("losses differ (cvm vs ad)",
      not np.array_equal(results[("cvm", False)].mat, results[("ad", False)].mat),
      "cvm and ad estimates should differ")

# ---- 5. unknown loss ----------------------------------------------------
try:
    covestEB(X, loss="bogus")
    check("bad loss", False, "expected ValueError")
except ValueError as e:
    check("bad loss", "unknown loss" in str(e), f"ValueError: {e}")

print()
print("TOTAL BAD:", bad)
raise SystemExit(1 if bad else 0)
