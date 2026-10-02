"""npfixedcomppy: non-parametric mixing distribution estimation.

Python port of the R package ``npfixedcomp2`` (Wang, 2007, extended):
non-parametric maximum-likelihood (and distance-based) estimation of
mixing distributions with optional fixed components, plus estimation of
the point-mass proportion at zero (``estpi0``).

All heavy computation runs in a bundled C++/Eigen extension
(``npfixedcomppy._core``, built with pybind11); the Python layer is a
thin, R-compatible front-end. Results are designed to match the R
package to working precision for the families below. There are no
hand-written OpenMP loops: at build time ``setup.py`` detects whether the
compiler supports the OpenMP flag and, if so, enables Eigen's own
compile-time-parallel GEMM/GEMV through that flag alone (otherwise the
build is serial Eigen).

Public API
----------
computemixdist : estimate a mixing distribution (with optional fixed
    components).
estpi0         : estimate a mixing distribution together with the
    proportion of point mass at zero.
Npmix          : result container (support points, weights, ll, ...).

Implemented families (``method`` argument)
------------------------------------------
+-------------+-------------------------------------------------+
| method      | model / structural parameter ``beta``           |
+=============+=================================================+
| ``npnormll``| normal kernel ``N(x; mu, beta)``; MLE           |
+-------------+-------------------------------------------------+
| ``npnormcvm``| normal kernel; Cramer-von Mises distance       |
+-------------+-------------------------------------------------+
| ``npnormad``| normal kernel; Anderson-Darling distance       |
+-------------+-------------------------------------------------+
| ``nptll``   | non-central-t kernel ``t(df=beta, ncp=mu)``;    |
|             | MLE; ``beta = inf`` reduces to the normal       |
+-------------+-------------------------------------------------+
| ``npnormcll``| correlation kernel; MLE; ``beta`` = number of  |
|             | observations (user-supplied)                    |
+-------------+-------------------------------------------------+
| ``nppoisll``| Poisson kernel; MLE (count data)                |
+-------------+-------------------------------------------------+
| ``npnormllw``| normal kernel; MLE (binned, ``order = -k``)   |
+-------------+-------------------------------------------------+
| ``npnormcvmw``| normal kernel; Cramér–von Mises (binned)      |
+-------------+-------------------------------------------------+
| ``npnormadw``| normal kernel; Anderson–Darling (binned)      |
+-------------+-------------------------------------------------+
| ``nptllw``   | non-central-t kernel; MLE (binned)             |
+-------------+-------------------------------------------------+

The binned (``"...w"``, ``order = -k``) families pre-bin the observations
onto the grid ``h = 10^order`` (round-down, as in the R ``bin``) and are
intended for large samples; see ``computemixdist`` / ``estpi0`` (the
``order`` argument).
"""

from npfixedcomppy.npfc import (  # noqa: F401
    FAMILIES,
    Npmix,
    computemixdist,
    covestEB,
    covestEB_cor,
    CovEBResult,
    dnppois,
    dnpt,
    dnpdisct,
    dnpdiscnorm,
    dnpnorm,
    dnpnormND,
    dnpnormc,
    dnormNDarray,
    estpi0,
    pnpdiscnorm,
    pnpnorm,
    pnppois,
    pnpt,
    posteriormean,
    __version__,
)

__all__ = [
    "computemixdist",
    "estpi0",
    "posteriormean",
    "covestEB",
    "covestEB_cor",
    "CovEBResult",
    "Npmix",
    "FAMILIES",
    "dnpnorm",
    "pnpnorm",
    "dnpnormc",
    "dnpt",
    "pnpt",
    "dnpdiscnorm",
    "pnpdiscnorm",
    "dnppois",
    "pnppois",
    "dnpdisct",
    "dnpnormND",
    "dnormNDarray",
    "__version__",
]
