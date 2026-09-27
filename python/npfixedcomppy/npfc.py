"""User-facing API for npfixedcomppy.

This module mirrors the R package ``npfixedcomp2``: the two entry points
:func:`computemixdist` (estimate a mixing distribution, with optional
fixed components) and :func:`estpi0` (estimate a mixing distribution
together with the proportion of point mass at zero).

All heavy computation — the histogram-based initial mixing distribution
and support grid, the density/mapping/gradient evaluations, the
constrained non-negative least-squares subproblems, and the support-point
solvers (improved Brent / successive parabolic interpolation) — runs in
the bundled Rust extension ``npfixedcomppy._core``; this module is a thin,
R-compatible front-end. Results are designed to match the R package to
working precision (typically ``ll`` to relative error ``1e-9`` and support
points to ``1e-6``).

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

The binned (``"...w"``, ``order = -k``) variants from the R package are
not ported yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

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
        """``{"pt": [...], "pr": [...]}`` (R-compatible accessor)."""
        return {"pt": self.pt, "pr": self.pr}

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"Npmix(family={self.family!r}, beta={self.beta}, "
            f"n_support={len(self.pt)}, ll={self.ll:.6g}, "
            f"convergence={self.convergence}, iter={self.iter})"
        )


def _to_vec(x: Optional[Sequence[float]]) -> list:
    if x is None:
        return []
    return [float(v) for v in np.asarray(x, dtype=float).ravel()]


def computemixdist(
    v,
    method: str = "npnormll",
    mu0=None,
    pi0=None,
    beta: Optional[float] = None,
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
    data = np.asarray(v, dtype=float).ravel().tolist()
    # R's computemixdist.<method> defaults: if (missing(mu0)) mu0 = 0;
    # if (missing(pi0)) pi0 = 0  (a zero-weight fixed component at the origin).
    mu0f = _to_vec(mu0) if mu0 is not None else [0.0]
    pi0f = _to_vec(pi0) if pi0 is not None else [0.0]
    gp = _to_vec(gridpoints)
    mix_pt, mix_pr = (
        (_to_vec(mix["pt"]), _to_vec(mix["pr"])) if mix is not None else ([], [])
    )
    fn = getattr(_core, method)
    res = fn(
        data, mu0f, pi0f, float(beta), mix_pt, mix_pr, gp,
        float(tol), int(maxit), int(verbose),
    )
    return _to_npmix(res)


def estpi0(
    v,
    method: str = "npnormll",
    beta: Optional[float] = None,
    val: Optional[float] = None,
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
    data = np.asarray(v, dtype=float).ravel().tolist()
    gp = _to_vec(gridpoints)
    mix_pt, mix_pr = (
        (_to_vec(mix["pt"]), _to_vec(mix["pr"])) if mix is not None else ([], [])
    )
    fn = getattr(_core, f"{method}_estpi0")
    res = fn(data, float(beta), val if val is not None else 2.0,
             mix_pt, mix_pr, gp, float(tol), int(verbose),
             bool(fast), bool(relax), float(inner_tol))
    return _to_npmix(res)


def _to_npmix(res) -> Npmix:
    """Convert a Rust-returned dict into an :class:`Npmix`."""
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
    def __init__(self, default_beta: float, needs_beta: bool):
        self.default_beta = default_beta
        self.needs_beta = needs_beta


FAMILIES = {
    "npnormll": _FamilySpec(1.0, False),
    "npnormcvm": _FamilySpec(1.0, False),
    "npnormad": _FamilySpec(1.0, False),
    "nptll": _FamilySpec(float("inf"), False),
    "npnormcll": _FamilySpec(0.0, True),   # user must supply beta
    "nppoisll": _FamilySpec(1.0, False),
}
