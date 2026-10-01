"""User-facing API for npfixedcomppy.

This module mirrors the R package ``npfixedcomp2``: the two entry points
:func:`computemixdist` (estimate a mixing distribution, with optional
fixed components) and :func:`estpi0` (estimate a mixing distribution
together with the proportion of point mass at zero).

All heavy computation — the histogram-based initial mixing distribution
and support grid, the density/mapping/gradient evaluations, the
constrained non-negative least-squares subproblems, and the support-point
solvers (improved Brent / successive parabolic interpolation) — runs in
the bundled ``npfixedcomppy._core`` extension (C++/Eigen); this module is
a thin, R-compatible front-end. Results are designed to match the R
package to working precision (typically ``ll`` to relative error ``1e-9``
and support points to ``1e-6``).

Threading: the package contains no hand-written OpenMP loops. At build
time ``setup.py`` probe-compiles the compiler with the OpenMP flag and,
when it supports it, adds the flag so Eigen's own compile-time gate
(``EIGEN_HAS_OPENMP``) switches on its internal parallel GEMM/GEMV; when
the flag is unavailable the build is serial Eigen. Either way the package
owns no worker threads of its own.

Implemented families (the ``method`` argument)
----------------------------------------------
+-------------+----------------------------------------------------------+
| method      | model / structural parameter ``beta``                    |
+=============+==========================================================+
| ``npnormll``| normal kernel ``N(x; mu, beta)``; maximum likelihood     |
+-------------+----------------------------------------------------------+
| ``npnormcvm``| normal kernel; Cramer-von Mises distance                |
+-------------+----------------------------------------------------------+
| ``npnormad``| normal kernel; Anderson-Darling distance                |
+-------------+----------------------------------------------------------+
| ``nptll``   | non-central-t kernel ``t(df=beta, ncp=mu)``; maximum     |
|             | likelihood; ``beta = inf`` reduces to the normal kernel  |
+-------------+----------------------------------------------------------+
| ``npnormcll``| correlation kernel; maximum likelihood; ``beta`` =      |
|             | number of observations (must be supplied)                |
+-------------+----------------------------------------------------------+
| ``nppoisll``| Poisson kernel; maximum likelihood (count data)          |
+-------------+----------------------------------------------------------+
| ``npnormllw``| binned normal kernel (grid ``h = 10**order``); maximum |
|             | likelihood                                               |
+-------------+----------------------------------------------------------+
| ``npnormcvmw``| binned normal kernel; Cramer-von Mises distance        |
+-------------+----------------------------------------------------------+
| ``npnormadw``| binned normal kernel; Anderson-Darling distance         |
+-------------+----------------------------------------------------------+
| ``nptllw``   | binned non-central-t kernel; maximum likelihood          |
+-------------+----------------------------------------------------------+

The binned (``"...w"``) families pre-bin the observations onto the grid
``h = 10**order`` (default ``order = -3``, round-down, as in R's ``bin``)
and fit the binned kernel; see :func:`computemixdist`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from npfixedcomppy import _core

__version__ = _core.version()


@dataclass
class Npmix:
    """Result of a non-parametric mixing distribution estimate.

    Attributes
    ----------
    pt : list of float
        Support points of the mixing distribution, including any fixed
        components, sorted ascending. For :func:`estpi0` this includes the
        point mass at zero whenever the hypothesis test threshold was
        exceeded.
    pr : list of float
        Weights (probabilities) corresponding to :attr:`pt`; they sum to 1
        up to floating-point rounding.
    beta : float
        The structural parameter actually used (the normal scale, the t
        degrees of freedom, or the number of observations for the
        correlation family).
    family : str
        Family identifier: ``"npnorm"``, ``"npt"``, ``"npnormc"`` or
        ``"nppois"``.
    min_gradient : float
        The minimum of the gradient of the loss with respect to a new
        support point, evaluated at the final support points. At
        convergence this is ``<= 0``.
    ll : float
        The loss at the estimate — negative log-likelihood for the
        maximum-likelihood families, the chosen distance for the
        ``"npnormcvm"``/``"npnormad"`` families — plus any family-specific
        extra term.
    flag : str
        ``"d0"`` if the support-point search is derivative-free (successive
        parabolic interpolation) and ``"d1"`` if the gradient's derivative
        is available (improved Brent's method).
    iter : int
        Number of outer iterations performed.
    convergence : int
        ``0`` if the loss-change tolerance was met, ``1`` if the maximum
        iteration count was reached first.

    Notes
    -----
    Instances are returned by :func:`computemixdist` and :func:`estpi0`.
    They are plain dataclasses: use attribute access (``r.ll``) or the
    R-compatible :attr:`mix` accessor (``r.mix["pt"]``).
    """

    pt: list = field(default_factory=list)
    pr: list = field(default_factory=list)
    beta: float = 1.0
    family: str = "npnorm"
    min_gradient: float = 0.0
    ll: float = 0.0
    flag: str = "d0"
    iter: int = 0
    convergence: int = 0

    @property
    def mix(self) -> dict:
        """R-compatible accessor: ``{"pt": [...], "pr": [...]}``.

        Mirrors the ``$mix`` component of the R ``nspmix`` result object,
        so code written against the R package can read the support points
        and weights via ``r.mix["pt"]`` / ``r.mix["pr"]``.
        """
        return {"pt": self.pt, "pr": self.pr}

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"Npmix(family={self.family!r}, beta={self.beta}, "
            f"n_support={len(self.pt)}, ll={self.ll:.6g}, "
            f"convergence={self.convergence}, iter={self.iter})"
        )


def _to_vec(x: Optional[Sequence[float]]) -> np.ndarray:
    """Coerce an optional array-like to a 1-D float64 numpy array.

    The result is passed straight to the C++ entry points, whose
    ``py::array_t<double>`` arguments read the buffer in place — no
    intermediate Python list is built, so the (n x 8)-byte data vector
    crosses the boundary exactly once.
    """
    if x is None:
        return np.empty(0, dtype=float)
    return np.ascontiguousarray(x, dtype=float).ravel()


def _bin(data: np.ndarray, order: int):
    """R's ``bin(data, order)``: round observations down to the fixed grid
    ``..., -h, 0, h, ...`` with ``h = 10^order`` and count the (non-empty)
    bins.

    Returns ``(v, w, h)``: the bin representatives (centres, ascending),
    the integer counts per bin (as floats, aligned with ``v``), and ``h``.
    This is exactly the pre-binning the R ``computemixdist.<method>w`` /
    ``estpi0.<method>w`` wrappers do before calling the C++ binned
    families, so the Python front-end mirrors it in one place.
    """
    h = 10.0 ** order
    b = np.floor(data / h)
    rng = b - b.min()
    t = np.bincount(rng.astype(np.int64))
    idx = np.nonzero(t)[0]
    v = h * (b.min() + idx)
    w = t[idx].astype(float)
    return v, w, h


def computemixdist(
    v: Sequence[float],
    method: str = "npnormll",
    mu0=None,
    pi0=None,
    beta: Optional[float] = None,
    order: int = -3,
    mix: Optional[dict] = None,
    gridpoints: Optional[Sequence[float]] = None,
    tol: float = 1e-6,
    maxit: int = 100,
    verbose: int = 0,
) -> Npmix:
    """Compute a non-parametric mixing distribution estimate.

    Estimates a discrete mixing distribution ``G = sum_j pi_j Delta_{mu_j}``
    for the kernel indexed by ``method``, optionally with fixed components
    (support points/weights that are included but never updated), matching
    the R ``computemixdist`` dispatcher of the ``npfixedcomp2`` package.

    Parameters
    ----------
    v : array_like
        The observations (1-D, coerced to ``float64``). Type conventions:

        * ``"nppoisll"`` — non-negative integers (counts);
        * ``"npnormcll"`` — sample correlation coefficients in (-1, 1);
        * everything else — real values.
    method : str, default ``"npnormll"``
        One of ``"npnormll"``, ``"npnormcvm"``, ``"npnormad"``, ``"nptll"``,
        ``"npnormcll"``, ``"nppoisll"``. Kernels and losses:

        * ``"npnormll"`` — normal kernel ``N(x; mu, beta)``, maximum
          likelihood; the gradient's derivative is available (solver flag
          ``"d1"``).
        * ``"npnormcvm"`` — normal kernel, Cramer-von Mises distance (the
          support-point search is derivative-free, flag ``"d0"``).
        * ``"npnormad"`` — normal kernel, Anderson-Darling distance (flag
          ``"d0"``).
        * ``"nptll"`` — non-central-t kernel ``t(df=beta, ncp=mu)`` under
          the maximum likelihood (flag ``"d0"``). With ``beta = inf`` the
          kernel reduces exactly to the normal ``N(x; mu, 1)``.
        * ``"npnormcll"`` — kernel for sample correlation coefficients
          under the maximum likelihood; ``beta`` (the number of
          observations) must be supplied.
        * ``"nppoisll"`` — Poisson kernel under the maximum likelihood
          (counts).
    mu0 : array_like of float, optional
        Support points of the fixed components. Default ``[0.0]`` (a
        zero-weight degenerate fixed component at the origin, as in the R
        package; it appears in the returned support points).
    pi0 : array_like of float, optional
        Weights of the fixed components, paired with ``mu0``. Default
        ``[0.0]``. Must satisfy ``sum(pi0) < 1``; the estimated components
        are scaled to the remaining mass ``1 - sum(pi0)``.
    beta : float, optional
        The structural parameter. Defaults: ``1.0`` for the normal families
        and ``"nppoisll"``, ``inf`` for ``"nptll"`` (the normal limit).
        For ``"npnormcll"`` it is required (the number of observations) and
        a ``ValueError`` is raised when omitted.
    order : int, default ``-3``
        Binning level for the binned (``"...w"``) families only. The
        observations are pre-binned onto the grid ``h = 10^order``
        (round-down, as in R's ``bin``); ``-3`` is the R default. Ignored
        by the un-binned families.
    mix : dict, optional
        An initial mixing distribution ``{"pt": [...], "pr": [...]}`` for
        the non-fixed part. If omitted it is derived from a histogram of
        the data (exactly as in the R package: ``nspmix:::initial.npnorm``
        for the continuous kernels).
    gridpoints : array_like of float, optional
        Candidate support points at which the new-point search evaluates
        the gradient. If omitted it is derived from a histogram of the
        data (``nspmix:::gridpoints.npnorm``).
    tol : float, default ``1e-6``
        Convergence tolerance on the improvement of the loss between
        iterations.
    maxit : int, default ``100``
        Maximum number of outer iterations.
    verbose : int, default ``0``
        ``0`` for no output, ``1`` to print each iteration's support
        points and probabilities (stderr), ``2`` for full diagnostics.

    Returns
    -------
    Npmix
        The estimated mixing distribution and diagnostics. ``Npmix.pt`` /
        ``Npmix.pr`` include the fixed components; ``Npmix.ll`` is the
        negative log-likelihood (or distance) at the estimate;
        ``Npmix.min_gradient <= 0`` at a converged solution.

    Raises
    ------
    ValueError
        If ``method`` is not implemented, or a method requiring ``beta``
        (``"npnormcll"``) is called without it.

    Notes
    -----
    Deterministic: identical inputs produce identical results. Each kernel
    column (the data vector evaluated at one support value) is computed
    once per fit and reused by every consumer, so no re-evaluation can ever
    change a result; the only parallelism is Eigen's own internal GEMM/GEMV
    splitting, which is enabled solely by the build-time compiler flag and
    never by code in this package.

    Examples
    --------
    >>> import numpy as np
    >>> from npfixedcomppy import computemixdist
    >>> x = np.concatenate([np.random.randn(500), np.random.randn(500) + 2.0])
    >>> r = computemixdist(x, method="npnormll")
    >>> len(r.pt) == len(r.pr)
    True

    Fixed components and a t kernel with finite degrees of freedom:

    >>> r = computemixdist(x, method="nptll", beta=5, mu0=[-0.5], pi0=[0.3])
    """
    if method not in FAMILIES:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(FAMILIES)}")
    if beta is None:
        beta = FAMILIES[method].default_beta
    if FAMILIES[method].needs_beta and beta is None:
        raise ValueError(f"method {method!r} requires an explicit `beta`")
    data = np.ascontiguousarray(v, dtype=float).ravel()
    # R's computemixdist.<method> defaults: if (missing(mu0)) mu0 = 0;
    # if (missing(pi0)) pi0 = 0  (a zero-weight fixed component at the origin).
    mu0f = _to_vec(mu0) if mu0 is not None else np.array([0.0])
    pi0f = _to_vec(pi0) if pi0 is not None else np.array([0.0])
    gp = _to_vec(gridpoints)
    mix_pt, mix_pr = (
        (_to_vec(mix["pt"]), _to_vec(mix["pr"])) if mix is not None else (np.empty(0), np.empty(0))
    )
    spec = FAMILIES[method]
    if spec.is_w:
        # Binned family: pre-bin the observations exactly as the R
        # `bin(v, order)` wrapper does, then pass (bin centres, bin counts)
        # and `h = 10^order` to the C++ binned entry point.
        vbin, wbin, h = _bin(data, int(order))
        fn = getattr(_core, method)
        res = fn(
            vbin, wbin, mu0f, pi0f, float(beta), float(h), mix_pt, mix_pr, gp,
            float(tol), int(maxit), int(verbose),
        )
    else:
        fn = getattr(_core, method)
        res = fn(
            data, mu0f, pi0f, float(beta), mix_pt, mix_pr, gp,
            float(tol), int(maxit), int(verbose),
        )
    return _to_npmix(res)


def estpi0(
    v: Sequence[float],
    method: str = "npnormll",
    beta: Optional[float] = None,
    val: Optional[float] = None,
    order: int = -3,
    mix: Optional[dict] = None,
    gridpoints: Optional[Sequence[float]] = None,
    tol: float = 1e-6,
    verbose: int = 0,
    fast: bool = True,
    relax: bool = False,
    inner_tol: float = 1e-4,
) -> Npmix:
    """Compute a mixing distribution with the proportion at zero estimated.

    Fits :func:`computemixdist` with the point mass at the origin *estimated
    rather than fixed*: it searches for the fixed weight ``pi0`` of a point
    mass at 0 such that the family's hypothesis statistic (the loss
    difference between the fit with the point mass and the unconstrained
    fit) equals the threshold ``val`` — a likelihood-ratio-style test for
    the maximum-likelihood families and a distance statistic for the
    distance families. This mirrors the R ``estpi0`` dispatcher.

    Parameters
    ----------
    v : array_like
        The observations (same conventions as :func:`computemixdist`).
    method : str, default ``"npnormll"``
        One of ``"npnormll"``, ``"npnormcvm"``, ``"npnormad"``, ``"nptll"``,
        ``"npnormcll"``, ``"nppoisll"``. Note (as in the R package): the
        ``estpi0`` theory for ``"nptll"`` and ``"npnormcll"`` is not fully
        established — use those results with caution.
    beta : float, optional
        Structural parameter (defaults as in :func:`computemixdist`);
        required for ``"npnormcll"``.
    val : float, optional
        Threshold on the hypothesis statistic. Default ``2.0``. The
        *recommended* thresholds follow the R package examples: ``2`` for
        the likelihood families (``"npnormll"``, ``"nptll"``,
        ``"npnormcll"``, ``"nppoisll"``), ``0.1`` for ``"npnormcvm"`` and
        ``1`` for ``"npnormad"``.
    order : int, default ``-3``
        Binning level for the binned (``"...w"``) families only (as in
        :func:`computemixdist`). Ignored by the un-binned families.
    mix : dict, optional
        Initial mixing distribution (as in :func:`computemixdist`).
    gridpoints : array_like of float, optional
        Candidate support points (as in :func:`computemixdist`).
    tol : float, default ``1e-6``
        Tolerance on the statistic during the refinement of the point-mass
        weight, and on the first (unconstrained) fit.
    verbose : int, default ``0``
        ``0`` for no output, ``1`` to print the refinement iterations
        (bracket, current point-mass weight, statistic; stderr).
    fast : bool, default ``True``
        Fast refinement (the default): each refinement of the point-mass
        proportion performs a single inner solve, at the relaxed
        ``inner_tol``. The convergence tolerance ``tol`` on the statistic is
        preserved, so the reported point mass is still the threshold
        solution up to ``tol`` — typically several times faster than the
        legacy path. Pass ``fast=False`` to reproduce the R/C++ refinement
        bit-for-bit (two inner solves per iteration, strict inner tolerance
        ``1e-6``).
    relax : bool, default ``False``
        Only meaningful with ``fast=True``. Early stop: the statistic is
        monotonically increasing in the point-mass weight, so once it already
        exceeds ``val`` while the weight is still small, pushing it up can
        only move it further past the threshold. The solver then stops after
        one more inner solve and returns the "no point mass beyond the test"
        answer, in O(1) inner solves instead of the full bisection.
    inner_tol : float, default ``1e-4``
        Only meaningful with ``fast=True``. Tolerance of the inner
        :func:`computemixdist` calls during refinement (legacy uses ``1e-6``;
        ``1e-4`` is usually ample and several times cheaper per iteration).

    Returns
    -------
    Npmix
        The estimated mixing distribution; a point mass at 0 (support
        ``0.0``) is included among ``pt``/``pr`` when the test threshold is
        exceeded, otherwise the unconstrained estimate is returned.

    Notes
    -----
    Deterministic like :func:`computemixdist`. With ``fast=True`` the
    reported point mass is accurate to ``tol`` in the hypothesis
    statistic; with ``fast=False`` the refinement matches the R/C++
    legacy path bit-for-bit.

    Examples
    --------
    >>> import numpy as np
    >>> from npfixedcomppy import estpi0
    >>> x = np.random.randn(500)
    >>> r = estpi0(x, method="npnormll", val=2.0)
    """
    if method not in FAMILIES:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(FAMILIES)}")
    if beta is None:
        beta = FAMILIES[method].default_beta
    if FAMILIES[method].needs_beta and beta is None:
        raise ValueError(f"method {method!r} requires an explicit `beta`")
    data = np.ascontiguousarray(v, dtype=float).ravel()
    gp = _to_vec(gridpoints)
    mix_pt, mix_pr = (
        (_to_vec(mix["pt"]), _to_vec(mix["pr"])) if mix is not None else (np.empty(0), np.empty(0))
    )
    val = 2.0 if val is None else val
    fn = getattr(_core, f"{method}_estpi0")
    if FAMILIES[method].is_w:
        vbin, wbin, h = _bin(data, int(order))
        res = fn(vbin, wbin, float(beta), float(h), val,
                 mix_pt, mix_pr, gp, float(tol), int(verbose),
                 bool(fast), bool(relax), float(inner_tol))
    else:
        res = fn(data, float(beta), val,
                 mix_pt, mix_pr, gp, float(tol), int(verbose),
                 bool(fast), bool(relax), float(inner_tol))
    return _to_npmix(res)


def posteriormean(
    x: Sequence[float],
    result: Npmix,
    fun: Optional[Callable[[float], float]] = None,
) -> np.ndarray:
    """Posterior mean of ``fun(pt)`` at each observation, given a fitted mixture.

    For a fitted mixing distribution ``G = sum_j pi_j Delta_{mu_j}`` (the
    support points/weights of ``result``, including any fixed components)
    and the family's kernel ``K(x; mu, beta)``, this returns

    .. math::

        \\hat{f}(x_i) = \\frac{\\sum_j K(x_i; \\mu_j, \\beta)\\,\\pi_j\\,
        f(\\mu_j)}{\\sum_j K(x_i; \\mu_j, \\beta)\\,\\pi_j}

    i.e. the expectation of ``fun(pt)`` under the posterior distribution of
    the latent support point induced by the observation ``x_i``. This is the
    R ``posteriormean`` dispatcher: the kernel follows ``result.family``
    (``"npnorm"`` / ``"npt"`` / ``"npnormc"`` / ``"nppois"``).

    Parameters
    ----------
    x : array_like
        The observations (1-D, same conventions as :func:`computemixdist`).
    result : Npmix
        A fit from :func:`computemixdist` or :func:`estpi0`; its ``family``
        and ``beta`` select the kernel and its structural parameter.
    fun : callable, optional
        The (vectorised or scalar) function whose posterior mean is wanted,
        evaluated at the support points. Default: the identity
        (``fun = lambda x: x``), so the result is the posterior mean of the
        support point itself. Only ``k = len(result.pt)`` evaluations happen
        in Python; the ``n x k`` kernel and the row sums run in C++.

    Returns
    -------
    numpy.ndarray
        ``out[i]`` as above for each observation.

    Examples
    --------
    >>> import numpy as np
    >>> from npfixedcomppy import computemixdist, posteriormean
    >>> x = np.concatenate([np.random.randn(500), np.random.randn(500) + 2])
    >>> r = computemixdist(x, method="npnormll")
    >>> pm = posteriormean(x, r)               # posterior mean of the point
    >>> pm2 = posteriormean(x, r, fun=np.tanh) # posterior mean of tanh(point)
    """
    if fun is None:
        fpt = np.ascontiguousarray(result.pt, dtype=float).ravel()
    else:
        fpt = np.ascontiguousarray([float(fun(v)) for v in result.pt], dtype=float)
    return _core.posteriormean(
        result.family,
        np.ascontiguousarray(x, dtype=float).ravel(),
        np.ascontiguousarray(result.pt, dtype=float).ravel(),
        np.ascontiguousarray(result.pr, dtype=float).ravel(),
        float(result.beta),
        fpt,
    )


@dataclass
class CovEBResult:
    """Result of an Empirical-Bayes covariance estimate (``covestEB``).

    Attributes
    ----------
    mat : numpy.ndarray
        The ``p x p`` covariance estimate (a valid correlation matrix
        rescaled back to the original variances).
    correction_Fnorm : float
        The Frobenius norm of the difference between the corrected
        correlation matrix and the (unprojected) one; measures how far the
        sample correlations were from the correlation-matrix cone.
    mix_dist : Npmix
        The mixing distribution fit to the (Fisher-transformed) sample
        correlations, from which the posterior means were derived.
    """

    mat: np.ndarray
    correction_Fnorm: float
    mix_dist: Npmix

    def __repr__(self) -> str:  # pragma: no cover - trivial
        p = self.mat.shape[0]
        return (
            f"CovEBResult(p={p}, correction_Fnorm={self.correction_Fnorm:.6g}, "
            f"mix.family={self.mix_dist.family!r}, "
            f"mix.n_support={len(self.mix_dist.pt)})"
        )


# R's ``.Machine$double.eps`` (the variance-zero threshold in covestEB).
_MACHINE_EPS = float(np.finfo(np.float64).eps)


def _extract_lower_colmajor(M: np.ndarray) -> np.ndarray:
    """Strict lower triangle of a symmetric matrix, in R column-major order.

    R's ``A[lower.tri(A)]`` reads columns left-to-right, so element ``k`` is
    ``(i, j)`` for ``j`` ascending and ``i > j``. This is the exact order the
    R package feeds to the fit, so matching it keeps the Python estimate
    bit-close to R's (same data order into the solver).
    """
    p = M.shape[0]
    out = np.empty(p * (p - 1) // 2, dtype=float)
    k = 0
    for j in range(p):
        for i in range(j + 1, p):
            out[k] = M[i, j]
            k += 1
    return out


def _return_lower(v: np.ndarray, p2: int) -> np.ndarray:
    """Build a symmetric matrix from its strict lower triangle (unit diagonal).

    The inverse of :func:`_extract_lower_colmajor`; places ``v`` into the
    strict lower triangle in the same R column-major order, mirrors it, and
    sets the diagonal to 1 (R's ``returnlower``).
    """
    L = np.zeros((p2, p2), dtype=float)
    k = 0
    for j in range(p2):
        for i in range(j + 1, p2):
            L[i, j] = v[k]
            k += 1
    return L + L.T + np.eye(p2)


def _corr_from_cov(covest: np.ndarray) -> np.ndarray:
    """R's ``cov2cor``: scale a covariance matrix to a correlation matrix."""
    sd = np.sqrt(np.diag(covest))
    sd = np.where(sd > 0, sd, 1.0)
    return covest / np.outer(sd, sd)


def covestEB(
    X: Sequence[Sequence[float]],
    estpi0: bool = False,
    order: int = -3,
    verbose: bool = False,
    force_nonbin: bool = False,
) -> CovEBResult:
    """Estimate a covariance matrix using Empirical Bayes (Fisher transform).

    Mirrors the R ``covestEB`` (utility.R). The strict lower triangle of the
    sample correlation matrix is Fisher-transformed (``atanh``), a
    non-parametric mixing distribution is fit to those transformed
    correlations (with standard error ``sqrt(1 / (n - 3))``), and the
    posterior mean of ``tanh(.)`` gives the shrunk correlations; the result
    is projected onto the correlation-matrix cone and rescaled by the sample
    standard deviations.

    Parameters
    ----------
    X : array_like
        An ``n x p`` matrix (rows = observations, columns = variables).
    estpi0 : bool, default ``False``
        If ``True``, fit with ``estpi0`` (the point-mass proportion at the
        null is estimated to the threshold ``val = 1``) rather than
        ``computemixdist``.
    order : int, default ``-3``
        Binning level used only when the number of pairwise correlations
        exceeds 5000 (the binned ``npnormllw`` family).
    verbose : bool, default ``False``
        Passed through to the inner fit.
    force_nonbin : bool, default ``False``
        If ``True``, use the non-binned fit even for very large ``p``.

    Returns
    -------
    CovEBResult
        ``mat`` (the covariance estimate), ``correction_Fnorm`` (Frobenius
        correction size), and ``mix_dist`` (the mixing fit).

    Notes
    -----
    Variables whose sample variance is ``<= .Machine$double.eps`` are
    dropped from the correlation computation (as in R) and their rows/columns
    in ``mat`` stay at the sample covariance diagonal.
    """
    A = np.asarray(X, dtype=float)
    if A.ndim != 2:
        raise ValueError("covestEB: X must be a 2-D matrix (n x p)")
    n, p = A.shape
    covest = np.cov(A, rowvar=False, ddof=1)
    index = np.diag(covest) > _MACHINE_EPS
    p2 = int(index.sum())
    C = covest[np.ix_(index, index)]
    Ccor = _corr_from_cov(C)
    fisherdata = np.arctanh(_extract_lower_colmajor(Ccor))
    beta = float(np.sqrt(1.0 / (n - 3)))
    v = int(verbose)

    use_bin = (len(fisherdata) > 5000) and (not force_nonbin)
    if estpi0:
        if use_bin:
            r = estpi0(fisherdata, method="npnormllw", order=order,
                       beta=beta, verbose=v, val=1.0)
        else:
            r = estpi0(fisherdata, beta=beta, verbose=v, val=1.0)
    else:
        if use_bin:
            r = computemixdist(fisherdata, method="npnormllw", order=order,
                               beta=beta, verbose=v)
        else:
            r = computemixdist(fisherdata, beta=beta, verbose=v)

    postmean = posteriormean(fisherdata, r, fun=np.tanh)
    ans = np.eye(p)
    if p2 >= 2:
        ans[np.ix_(index, index)] = _return_lower(postmean, p2)
    ans1 = np.asarray(_core.correlationmatrixcpp(ans.tolist(), tau=0.0, tol=1e-3))
    varest = np.sqrt(np.diag(covest))
    mat = ans1 * np.outer(varest, varest)
    return CovEBResult(
        mat=mat,
        correction_Fnorm=float(np.linalg.norm(ans1 - ans, "fro")),
        mix_dist=r,
    )


def covestEB_cor(
    X: Sequence[Sequence[float]],
    verbose: bool = False,
) -> CovEBResult:
    """Estimate a covariance matrix using Empirical Bayes (raw correlations).

    Mirrors the R ``covestEB.cor`` (utility.R): a one-parameter-normal
    (``npnormcll``) mixing distribution is fit directly to the sample
    correlation coefficients (``beta = n``), the posterior mean (identity)
    gives the shrunk correlations, and the result is projected onto the
    correlation-matrix cone and rescaled by the sample standard deviations.

    Parameters
    ----------
    X : array_like
        An ``n x p`` matrix (rows = observations, columns = variables).
    verbose : bool, default ``False``
        Passed through to the inner fit.

    Returns
    -------
    CovEBResult
        ``mat``, ``correction_Fnorm``, and ``mix_dist``.
    """
    A = np.asarray(X, dtype=float)
    if A.ndim != 2:
        raise ValueError("covestEB_cor: X must be a 2-D matrix (n x p)")
    n, p = A.shape
    covest = np.cov(A, rowvar=False, ddof=1)
    index = np.diag(covest) > _MACHINE_EPS
    p2 = int(index.sum())
    C = covest[np.ix_(index, index)]
    Ccor = _corr_from_cov(C)
    data = _extract_lower_colmajor(Ccor)
    r = computemixdist(data, beta=float(n), method="npnormcll", verbose=int(verbose))
    postmean = posteriormean(data, r)
    ans = np.eye(p)
    if p2 >= 2:
        ans[np.ix_(index, index)] = _return_lower(postmean, p2)
    ans1 = np.asarray(_core.correlationmatrixcpp(ans.tolist(), tau=0.0, tol=1e-3))
    varest = np.sqrt(np.diag(covest))
    mat = ans1 * np.outer(varest, varest)
    return CovEBResult(
        mat=mat,
        correction_Fnorm=float(np.linalg.norm(ans1 - ans, "fro")),
        mix_dist=r,
    )


def _to_npmix(res) -> Npmix:
    """Convert a ``_core`` result mapping into an :class:`Npmix`."""
    if isinstance(res, Npmix):
        return res
    return Npmix(
        pt=list(res["pt"]),
        pr=list(res["pr"]),
        beta=float(res["beta"]),
        family=str(res["family"]),
        min_gradient=float(res["min_gradient"]),
        ll=float(res["ll"]),
        flag=str(res["flag"]),
        iter=int(res["iter"]),
        convergence=int(res["convergence"]),
    )


class _FamilySpec:
    """Per-family defaults.

    Attributes
    ----------
    default_beta : float
        The structural parameter used when the caller does not supply one.
    needs_beta : bool
        If ``True``, an explicit ``beta`` is mandatory and a missing one
        raises ``ValueError`` (currently only ``"npnormcll"``).
    is_w : bool
        If ``True`` this is a binned ("``...w``") family: the observations
        are pre-binned (``order``) into bin centres/counts and the C++
        binned entry point is called with ``h = 10^order``.
    """

    def __init__(self, default_beta: float, needs_beta: bool, is_w: bool = False):
        self.default_beta = default_beta
        self.needs_beta = needs_beta
        self.is_w = is_w


#: Implemented families: ``method`` name -> :class:`_FamilySpec`.
#:
#: +-------------+--------------+-------------------------------+
#: | method      | default beta | meaning of ``beta``           |
#: +=============+==============+===============================+
#: | ``npnormll``| ``1.0``      | normal scale                  |
#: +-------------+--------------+-------------------------------+
#: | ``npnormcvm``| ``1.0``     | normal scale                  |
#: +-------------+--------------+-------------------------------+
#: | ``npnormad`` | ``1.0``      | normal scale                  |
#: +-------------+--------------+-------------------------------+
#: | ``nptll``    | ``inf``      | t degrees of freedom          |
#: |             |              | (``inf`` = the normal kernel) |
#: +-------------+--------------+-------------------------------+
#: | ``npnormcll``| required     | number of observations        |
#: +-------------+--------------+-------------------------------+
#: | ``nppoisll`` | ``1.0``      | Poisson scale                 |
#: +-------------+--------------+-------------------------------+
#: | ``npnormllw``| ``1.0``      | normal scale (binned)         |
#: +-------------+--------------+-------------------------------+
#: | ``npnormcvmw``| ``1.0``    | normal scale (binned)         |
#: +-------------+--------------+-------------------------------+
#: | ``npnormadw``| ``1.0``      | normal scale (binned)         |
#: +-------------+--------------+-------------------------------+
#: | ``nptllw``   | ``inf``      | t degrees of freedom (binned) |
#: +-------------+--------------+-------------------------------+
FAMILIES = {
    "npnormll": _FamilySpec(1.0, False),
    "npnormcvm": _FamilySpec(1.0, False),
    "npnormad": _FamilySpec(1.0, False),
    "nptll": _FamilySpec(float("inf"), False),
    "npnormcll": _FamilySpec(0.0, True),   # user must supply beta
    "nppoisll": _FamilySpec(1.0, False),
    # binned ("...w") families: observations pre-binned with `order`.
    "npnormllw": _FamilySpec(1.0, False, True),
    "npnormcvmw": _FamilySpec(1.0, False, True),
    "npnormadw": _FamilySpec(1.0, False, True),
    "nptllw": _FamilySpec(float("inf"), False, True),
}
