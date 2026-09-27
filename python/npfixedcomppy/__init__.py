"""npfixedcomppy: non-parametric mixing distribution estimation.

Python port of the R package ``npfixedcomp2`` (Wang, 2007, extended):
non-parametric maximum-likelihood (and distance-based) estimation of
mixing distributions with optional fixed components, plus estimation of
the point-mass proportion at zero (``estpi0``).

All heavy computation runs in a bundled Rust extension
(``npfixedcomppy._core``); the Python layer is a thin, R-compatible
front-end. Results are designed to match the R package to working
precision for the families below.

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

The binned ("``w``", ``order = -k``) variants from the R package are not
ported yet.
"""

from npfixedcomppy.npfc import (  # noqa: F401
    FAMILIES,
    Npmix,
    computemixdist,
    estpi0,
    __version__,
)

__all__ = [
    "computemixdist",
    "estpi0",
    "Npmix",
    "FAMILIES",
    "__version__",
]
