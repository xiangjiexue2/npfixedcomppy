import os, sys
import numpy as np
from npfixedcomppy import _core

base = r'C:/Users/xxjie/Documents/rebuild'
X = np.loadtxt(os.path.join(base, 'parity_2d.csv'), delimiter=',')
gp = np.loadtxt(os.path.join(base, 'grid_2d.csv'), delimiter=',')
ipt = np.loadtxt(os.path.join(base, 'initpt_2d.csv'), delimiter=',')
ipr = np.loadtxt(os.path.join(base, 'initpr_2d.txt'), delimiter=',').ravel()
ref = [float(t) for t in open(os.path.join(base, 'res_2d.txt')).read().split()]

print('data', X.shape, 'grid', gp.shape, 'initpt', ipt.shape, 'initpr', ipr.shape)
res = _core.npnorm2Dll(
    np.ascontiguousarray(X), np.ascontiguousarray([0.0, 0.0]),
    np.ascontiguousarray([0.0]), np.ascontiguousarray(np.eye(2).ravel()),
    np.ascontiguousarray(ipt), np.ascontiguousarray(ipr),
    np.ascontiguousarray(gp), 1e-6, 100, 0,
)
print('PY k   =', len(res['pr']))
print('PY ll  =', repr(res['ll']))
print('PY iter=', res['iter'], ' conv=', res['convergence'])
print('PY ming=', repr(res['min_gradient']))
print('R  ll  =', ref[0], ' iter=', ref[1], ' conv=', ref[2], ' ming=', ref[3])
print('PY pt:'); print(np.array(res['pt']))
print('PY pr:', np.array(res['pr']))
