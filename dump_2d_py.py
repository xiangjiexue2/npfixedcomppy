import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
import numpy as np
from npfixedcomppy import _core

X = np.loadtxt('C:/Users/xxjie/Documents/rebuild/parity_2d.csv', delimiter=',')
beta = np.eye(2).ravel()
col1 = X[:, 0].copy(); col2 = X[:, 1].copy()
w = np.ones(len(X))
g1 = _core.gridpoints_npnorm(col1, w, 1.0, 100)
g2 = _core.gridpoints_npnorm(col2, w, 1.0, 100)
LLL = max(len(g1), len(g2))
g1s = sorted(list(np.repeat(g1, LLL // len(g1) + 1)[:LLL]))
g2s = sorted(list(np.repeat(g2, LLL // len(g2) + 1)[:LLL]))
print(f"PY default: g1={len(g1)} g2={len(g2)} LLL={LLL} grid={2*LLL//2}")
print("PY g1s[:8]:", [f"{v:.17g}" for v in g1s[:8]])
print("PY g2s[:8]:", [f"{v:.17g}" for v in g2s[:8]])
# init
for tag, col in [("m1", col1), ("m2", col2)]:
    r = _core.initial_npnorm(col, w, 1.0, [], [])
    print(f"PY init {tag}: k={len(r[1])} pts={[f'{v:.6g}' for v in r[1]]}")
