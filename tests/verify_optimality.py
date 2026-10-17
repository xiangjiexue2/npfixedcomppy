"""Self-contained optimality (KKT / directional-derivative) gate.

Replaces the old R-parity gate: instead of matching values recorded by the R
reference implementation, we verify the property that ANY correct fit must
satisfy — the directional derivative (the gain of adding a new support point
at the returned estimate) is not negative within a tolerance.

For a concave maximum-likelihood (NPMLE) or minimum-distance (NPMDE)
estimate no new support point may improve the objective, so the minimum gain
over the support, ``min_gradient``, must be ``>= -TOL`` (``TOL`` absorbs
numerical noise; a solver that failed to converge would leave it clearly
negative).  ``grid_gain`` (the grid-level analogue) is reported for
information only — it is grid-resolution dependent and legitimately
slightly negative, so it does not gate.

Data is generated in memory from a fixed seed — no R, no external files:
  * a normal mixture        -> npnormll / npnormcvm / npnormad (+ binned ...w)
  * a non-central-t mixture -> nptll (+ binned nptllw)
  * Fisher correlations     -> npnormcll
  * a Poisson mixture       -> nppoisll
"""
import os
import sys

import numpy as np

from npfixedcomppy import computemixdist

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-3    # directional derivative must be >= -TOL
N = 2000
SEED = 20260101

bad = 0


def check(tag, cond, msg):
    global bad
    if not cond:
        bad += 1
        print(f"  [BAD] {tag}: {msg}")
    else:
        print(f"  [OK ] {tag}: {msg}")


# ---- data generation (fixed seed, in memory, vectorised) -----------------
def gen_normal(rng, means, weights, stdev, n):
    mus = np.asarray(means)[rng.choice(len(means), size=n, p=weights)]
    return rng.normal(mus, stdev)


def gen_t(rng, means, weights, df, n):
    mus = np.asarray(means)[rng.choice(len(means), size=n, p=weights)]
    return rng.normal(mus, 1.0) / np.sqrt(rng.chisquare(df, n) / df)


def gen_corr(rng, means, weights, n):
    # Fisher transform of a normal mixture: values strictly in (-1, 1).
    return np.tanh(gen_normal(rng, means, weights, 1.0, n))


def gen_pois(rng, lambdas, weights, n):
    lam = np.asarray(lambdas)[rng.choice(len(lambdas), size=n, p=weights)]
    return rng.poisson(lam).astype(float)


rng = np.random.default_rng(SEED)
data = {
    "norm": gen_normal(rng, [-1.5, 0.0, 1.8], [0.3, 0.4, 0.3], 1.0, N),
    "t": gen_t(rng, [-1.5, 0.0, 1.8], [0.3, 0.4, 0.3], 5, N),
    "corr": gen_corr(rng, [-0.6, 0.0, 0.45], [0.4, 0.3, 0.3], N),
    "pois": gen_pois(rng, [1.0, 4.0], [0.5, 0.5], N),
}

# (tag, method, data key, beta, group)
CASES = [
    ("npnormll", "npnormll", "norm", 1.0, "NPMLE"),
    ("nptll", "nptll", "t", 5.0, "NPMLE"),
    ("npnormcll", "npnormcll", "corr", 100.0, "NPMLE"),
    ("nppoisll", "nppoisll", "pois", None, "NPMLE"),
    ("npnormcvm", "npnormcvm", "norm", 1.0, "NPMDE"),
    ("npnormad", "npnormad", "norm", 1.0, "NPMDE"),
    ("npnormllw", "npnormllw", "norm", 1.0, "NPMLE binned"),
    ("nptllw", "nptllw", "t", 5.0, "NPMLE binned"),
    ("npnormcvmw", "npnormcvmw", "norm", 1.0, "NPMDE binned"),
    ("npnormadw", "npnormadw", "norm", 1.0, "NPMDE binned"),
]

print("== directional-derivative (KKT) certificate ==")
print(f"   gate: min_gradient >= -TOL  (TOL={TOL:g});  n={N}, seed={SEED}")
print()
for tag, method, dkey, beta, group in CASES:
    kw = {} if beta is None else {"beta": beta}
    r = computemixdist(data[dkey], method=method, **kw)
    ok = (r.convergence == 0) and (r.min_gradient >= -TOL)
    check(
        tag,
        ok,
        f"[{group}] conv={r.convergence} min_gradient={r.min_gradient:+.3e} "
        f"(>= -{TOL:g}) npt={len(r.pt)} iter={r.iter} "
        f"[grid_gain={r.grid_gain:+.3e}]",
    )

print()
print("TOTAL BAD:", bad)
sys.exit(0 if bad == 0 else 1)
