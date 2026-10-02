import numpy as np
from npfixedcomppy import _core

base = r'C:/Users/xxjie/Documents/rebuild'
gp = np.loadtxt(f'{base}/small_grid.csv', delimiter=',')
ipt = np.array([[0.5, -0.5], [-1.0, 0.0], [0.0, 1.0], [1.5, 0.3]])
ipr = np.array([0.2, 0.3, 0.2, 0.3])
mu0 = np.array([0.0, 0.0]); pi0 = np.array([0.0]); beta = np.eye(2).ravel()

for seed in (11, 22, 33):
    X = np.loadtxt(f'{base}/small_data_{seed}.csv', delimiter=',')
    res = _core.npnorm2Dll(np.ascontiguousarray(X), np.ascontiguousarray(mu0),
                           np.ascontiguousarray(pi0), np.ascontiguousarray(beta),
                           np.ascontiguousarray(ipt), np.ascontiguousarray(ipr),
                           np.ascontiguousarray(gp), 1e-6, 100, 0)
    pt = np.array(res['pt']); pr = np.array(res['pr'])
    print(f"PY seed={seed} k={len(pr)} ll={res['ll']!r}")
    # write Python's mixture (pt rows, pr) for R to score
    with open(f'{base}/py_mix_{seed}.txt', 'w') as f:
        f.write(f"{len(pt)}\n")
        for j in range(len(pt)):
            f.write(f"{float(pt[j,0]):.17g} {float(pt[j,1]):.17g} {float(pr[j]):.17g}\n")
    print(f"   pt={pt.round(6).tolist()}")
    print(f"   pr={pr.round(6).tolist()}")
