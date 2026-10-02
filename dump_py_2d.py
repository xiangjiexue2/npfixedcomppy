import os, sys
import numpy as np
from npfixedcomppy import _core

base = r'C:/Users/xxjie/Documents/rebuild'
X = np.loadtxt(os.path.join(base, 'parity_2d.csv'), delimiter=',')
gp = np.loadtxt(os.path.join(base, 'grid_2d.csv'), delimiter=',')
ipt = np.loadtxt(os.path.join(base, 'initpt_2d.csv'), delimiter=',')
ipr = np.loadtxt(os.path.join(base, 'initpr_2d.txt'), delimiter=',').ravel()

res = _core.npnorm2Dll(
    np.ascontiguousarray(X), np.ascontiguousarray([0.0, 0.0]),
    np.ascontiguousarray([0.0]), np.ascontiguousarray(np.eye(2).ravel()),
    np.ascontiguousarray(ipt), np.ascontiguousarray(ipr),
    np.ascontiguousarray(gp), 1e-6, 100, 0,
)
pt = np.array(res['pt']); pr = np.array(res['pr'])
np.savetxt(os.path.join(base, 'py_pts_2d.csv'), pt, delimiter=',')
np.savetxt(os.path.join(base, 'py_pr_2d.txt'), pr, delimiter=',')
np.savetxt(os.path.join(base, 'py_initpt_2d.csv'), ipt, delimiter=',')
np.savetxt(os.path.join(base, 'py_initpr_2d.txt'), ipr, delimiter=',')
print('PY k   =', len(pr))
print('PY ll  =', repr(res['ll']))
print('PY iter=', res['iter'], ' conv=', res['convergence'])
print('PY ming=', repr(res['min_gradient']))
print('wrote py_pts_2d.csv / py_pr_2d.txt / py_initpt_2d.csv / py_initpr_2d.txt')
