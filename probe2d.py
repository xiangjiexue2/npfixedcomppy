import numpy as np
from npfixedcomppy import _core

base = r'C:/Users/xxjie/Documents/rebuild'
X = np.loadtxt(base + '/parity_2d.csv', delimiter=',')
gp = np.loadtxt(base + '/grid_2d.csv', delimiter=',')
r3 = np.loadtxt(base + '/r3_pts_2d.csv', delimiter=',')
r3pr = np.loadtxt(base + '/r3_pr_2d.txt', delimiter=',').ravel()

# 1) independent numpy ll at R's final free points (drop the fixed (0,0) row)
free = r3[:-1]; freepr = r3pr[:-1]
ll = 0.0
for i in range(X.shape[0]):
    s = 0.0
    for j in range(len(free)):
        d = X[i] - free[j]
        s += freepr[j] * np.exp(-0.5 * d @ d) / (2 * np.pi)
    ll += np.log(s)
print('numpy ll @ R-pts  =', repr(-ll))
print('R    ll @ R-pts   =', 848.88775287688986)

# 2) PY C++ kernel (dnpnormND) on R's points via npnorm2Dll_ single-iter path:
#    simpler: call the exported kernel directly if present
print('has dnpnormND:', hasattr(_core, 'dnpnormND'))
if hasattr(_core, 'dnpnormND'):
    dens = _core.dnpnormND(np.ascontiguousarray(X), np.ascontiguousarray(free),
                           np.ascontiguousarray(freepr), np.ascontiguousarray(np.eye(2)),
                           False)
    print('PY kernel ll @ R-pts =', repr(-float(np.sum(np.log(dens)))))

# 3) feed R's final points as PY start: expect immediate convergence (ll ~ R)
res = _core.npnorm2Dll(
    np.ascontiguousarray(X), np.ascontiguousarray([0.0, 0.0]),
    np.ascontiguousarray([0.0]), np.ascontiguousarray(np.eye(2).ravel()),
    np.ascontiguousarray(free), np.ascontiguousarray(freepr),
    np.ascontiguousarray(gp), 1e-6, 100, 0,
)
print('PY start@R-pts: k=%d ll=%r iter=%d ming=%r' % (
    len(res['pr']), res['ll'], res['iter'], res['min_gradient']))

# 4) PY determinism (fresh init, twice)
ipt = np.loadtxt(base + '/initpt_2d.csv', delimiter=',')
ipr = np.loadtxt(base + '/initpr_2d.txt', delimiter=',').ravel()
runs = []
for _ in range(2):
    r = _core.npnorm2Dll(
        np.ascontiguousarray(X), np.ascontiguousarray([0.0, 0.0]),
        np.ascontiguousarray([0.0]), np.ascontiguousarray(np.eye(2).ravel()),
        np.ascontiguousarray(ipt), np.ascontiguousarray(ipr),
        np.ascontiguousarray(gp), 1e-6, 100, 0,
    )
    runs.append(r)
print('PY fresh run1: k=%d ll=%r iter=%d' % (len(runs[0]['pr']), runs[0]['ll'], runs[0]['iter']))
print('PY fresh run2: k=%d ll=%r iter=%d' % (len(runs[1]['pr']), runs[1]['ll'], runs[1]['iter']))
print('deterministic:', abs(runs[0]['ll'] - runs[1]['ll']) < 1e-12)
