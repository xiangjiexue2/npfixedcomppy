"""Shared benchmark data for the npfixedcomppy vs R perf bench.

Usage: python bench_data.py [n]   (default 200000)
Writes tests/npfc_bench_data_{n}.csv with two columns (normal mixture,
Poisson counts).
"""
import sys

import numpy as np

n = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
rng = np.random.default_rng(20240529)
v_norm = np.concatenate(
    [rng.standard_normal(n // 2), rng.standard_normal(n // 2) + 2.0]
)
v_pois = rng.poisson(np.where(rng.random(n) < 0.5, 1.5, 4.5))
np.savetxt(
    f"tests/npfc_bench_data_{n}.csv",
    np.column_stack([v_norm, v_pois]),
    delimiter=",",
)
print(f"wrote tests/npfc_bench_data_{n}.csv ({n} rows x 2 cols)")
