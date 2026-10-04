"""A/B benchmark for the `NPFIC_WARM` support-point hot-start.

Three arms, same data / initial mix / grid / tol:
  cold (NPFIC_WARM=0)  - the shipped, bit-identical behaviour (reference);
  seed (NPFIC_WARM=1)  - previous-call refined root seeds `brmin`/`dfmin`;
                         the search still converges (quality-preserving);
  aggr (NPFIC_WARM=2)  - aggressive CNM working-set re-verification
                         (previous root accepted with ONE eval; A/B reference).

Each arm is run `repeats` times (median wall reported). The
`NPFIXEDCOMPY_PROFILE` line is written to the C-level stderr, so it is
captured by redirecting fd 2 (Python's `redirect_stderr` cannot see it).

Gate (the relaxed contract, measured against the cold arm):
  * seed:  ll within O(tol) of cold, and never worse; mixture density on
           the raw data matches cold to ~1e-5 relative;
  * aggr:  reported for reference only (its quality is not guaranteed on
           families whose support roots drift between iterations).
"""
import os
import statistics
import sys
import time

import numpy as np

from npfixedcomppy import (
    computemixdist,
    dnpdisct,
    dnppois,
    dnpnorm,
    dnpnormc,
    dnpdiscnorm,
    dnpt,
)

os.environ["NPFIXEDCOMPY_PROFILE"] = "1"
REPEATS = 5
ARMS = [("cold", 0), ("seed", 1), ("aggr", 2)]


def with_c_stderr(fn):
    """Run `fn()` with fd 2 redirected to a pipe; return (result, text)."""
    r, w = os.pipe()
    old2 = os.dup(2)
    os.dup2(w, 2)
    os.close(w)
    try:
        res = fn()
    finally:
        os.dup2(old2, 2)
        os.close(old2)
    chunks = []
    while True:
        b = os.read(r, 65536)
        if not b:
            break
        chunks.append(b)
    os.close(r)
    return res, b"".join(chunks).decode(errors="replace")


def _run(method, data, beta, order, mode, repeats):
    os.environ["NPFIC_WARM"] = str(mode)
    times, res, prof = [], None, {}
    for _ in range(repeats):
        t0 = time.perf_counter()
        res, err = with_c_stderr(
            lambda: computemixdist(
                data, method=method, beta=beta, order=order, verbose=0
            )
        )
        times.append(time.perf_counter() - t0)
        for line in err.splitlines():
            if "PROFILE" in line:
                prof = _parse(line)
    return statistics.median(times), res, prof


def _parse(line):
    out = {}
    for tok in line.replace("PROFILE", "").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def _density(method, data, beta, order, res):
    h = 10.0**order
    if method == "npnormll":
        return dnpnorm(data, res.pt, res.pr, beta)
    if method == "nptll":
        return dnpt(data, res.pt, res.pr, beta)
    if method == "npnormcll":
        return dnpnormc(data, res.pt, res.pr, beta)
    if method == "npnormad":
        return dnpdiscnorm(data, res.pt, res.pr, beta, h)
    if method == "nptllw":
        return dnpdisct(data, res.pt, res.pr, beta, h)
    if method == "nppoisll":
        return dnppois(data, res.pt, res.pr, beta)
    raise ValueError(method)


def _mk_data(method, n, rng):
    if method == "npnormcll":
        # correlation-like values well inside (-1, 1)
        base = 0.45 * rng.standard_normal(n)
        return np.clip(base, -0.7, 0.7)
    if method == "nppoisll":
        return rng.poisson(2.5, n).astype(float)
    # a 3-component normal mixture (real-valued families)
    c = rng.integers(0, 3, size=n)
    x = np.empty(n)
    x[c == 0] = -1.5 + 0.8 * rng.standard_normal((c == 0).sum())
    x[c == 1] = 1.0 + 0.6 * rng.standard_normal((c == 1).sum())
    x[c == 2] = 3.5 + 0.5 * rng.standard_normal((c == 2).sum())
    return x


CASES = [
    # (method, beta, order)
    ("nptll", 5.0, -3),
    ("nptllw", 5.0, -3),
    ("npnormcll", 1000.0, -3),
    ("npnormll", 1.0, -3),
    ("npnormad", 1.0, -3),
    ("nppoisll", 1.0, -3),
]


def main():
    n = 5000
    rng = np.random.default_rng(42)
    data = {m: _mk_data(m, n, rng) for m, _, _ in CASES}

    hdr = (
        f"{'method':<10} {'arm':<5} {'wall_ms':>9} {'iters':>5} {'evals':>7} "
        f"{'fresh':>6} {'freshms':>8} {'ll':>14}"
    )
    print(hdr)
    print("-" * len(hdr))
    print(f"n={n} per case, repeats={REPEATS} (median wall), "
          "same data/init/grid/tol in all arms")
    print()
    ok_all = True
    for method, beta, order in CASES:
        x = data[method]
        runs = {}
        for name, mode in ARMS:
            runs[name] = _run(method, x, beta, order, mode, REPEATS)
        cold_t, cold_r, cold_p = runs["cold"]
        for name, _ in ARMS:
            t, r, p = runs[name]
            print(
                f"{method:<10} {name:<5} {t * 1e3:>9.1f} {r.iter:>5} "
                f"{p.get('evals', -1):>7.0f} {p.get('freshcols', 0):>6.0f} "
                f"{p.get('freshms', -1):>8.1f} {r.ll:>14.6f}"
            )
        d_c = _density(method, x, beta, order, cold_r)
        scale = max(1.0, abs(cold_r.ll))
        for name in ("seed", "aggr"):
            t, r, p = runs[name]
            d = _density(method, x, beta, order, r)
            absmax = float(np.max(np.abs(d - d_c))) if len(d) else 0.0
            denom = float(np.max(np.abs(d_c)))
            relmax = absmax / denom if denom > 0 else 0.0
            dll = r.ll - cold_r.ll
            speedup = cold_t / t if t > 0 else float("inf")
            line = (
                f"{'':10} {name:<5} A/B: speedup x{speedup:5.2f}  "
                f"evals {p.get('evals', 0):.0f} vs "
                f"{cold_p.get('evals', 0):.0f}  "
                f"max|Δdens|={absmax:.3e} (rel {relmax:.2e})  "
                f"Δll={dll:+.3e}  "
            )
            if name == "seed":
                ok = dll <= 1e-5 * scale and relmax < 1e-5
                ok_all &= bool(ok)
                line += "GATE " + ("OK" if ok else "FAIL")
            else:
                line += "ref: " + (
                    "OK" if dll <= 1e-5 * scale and relmax < 1e-5 else
                    "WORSE (expected on drifting d1 families)"
                )
            print(line)
        print()
    print("SEED GATE: ALL OK" if ok_all else "SEED GATE: FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
