# npfixedcomppy

A Rust + Python reimplementation of the R package
[`npfixedcomp2`](https://CRAN) for computing **non-parametric mixing
distribution** estimates for several parametric families, with support for

- estimating the mixing distribution with some components **fixed**;
- estimating the mixing distribution **together with the proportion of
  point mass at zero** (`estpi0`);
- multiple **loss functions** (maximum likelihood, Cramér–von Mises
  distance, Anderson–Darling distance).

All numerically heavy work (initial mixing distribution, grid points,
density / CDF / mapping / gradient evaluation, the constrained non-negative
least-squares subproblem, and the support-point solvers) is implemented in
Rust; the Python package is a thin, ergonomic front-end.

The algorithms follow Wang (2007) and the extensions in the `npfixedcomp2`
R package. The un-binned families are currently supported:

| `method`     | family    | loss function                | structural parameter `beta`            |
|--------------|-----------|------------------------------|----------------------------------------|
| `npnormll`   | normal    | maximum likelihood           | scale (default `1`)                    |
| `npnormcvm`  | normal    | Cramér–von Mises distance    | scale (default `1`)                    |
| `npnormad`   | normal    | Anderson–Darling distance    | scale (default `1`)                    |
| `nptll`      | t         | maximum likelihood           | degrees of freedom (default `inf`)     |
| `npnormcll`  | 1-param normal | maximum likelihood       | number of observations (user supplied) |
| `nppoisll`   | poisson   | maximum likelihood           | none (unused)                          |

## Installation

```bash
pip install -e .            # builds the Rust extension via maturin
```

## Quick start

```python
import numpy as np
from npfixedcomppy import computemixdist, estpi0

rng = np.random.default_rng(123)
x = rng.normal(loc=[0, 2], scale=1, size=1000)

# non-parametric maximum-likelihood mixing distribution (normal family)
res = computemixdist(x, method="npnormll")
print(res)                 # Npmix(family='npnorm', beta=1.0, n_support=..., ...)
print(res.pt)              # support points
print(res.pr)              # weights

# estimate the proportion at 0 simultaneously
res0 = estpi0(x, method="npnormll", val=2)
```

## Tests

```bash
python -m pytest
```
