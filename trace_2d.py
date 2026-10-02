import os, sys
import numpy as np
from npfixedcomppy import computemixdist

base = r'C:/Users/xxjie/Documents/rebuild'
X = np.loadtxt(os.path.join(base, 'parity_2d.csv'), delimiter=',')
gp = np.loadtxt(os.path.join(base, 'grid_2d.csv'), delimiter=',')
ipt = np.loadtxt(os.path.join(base, 'initpt_2d.csv'), delimiter=',')
ipr = np.loadtxt(os.path.join(base, 'initpr_2d.txt'), delimiter=',').ravel()

r = computemixdist(
    X, method='npnorm2Dll',
    mu0=[0.0, 0.0], pi0=[0.0], beta=[[1.0, 0.0], [0.0, 1.0]],
    mix={'pt': ipt.tolist(), 'pr': ipr.tolist()},
    gridpoints=gp.tolist(), tol=1e-6, maxit=100, verbose=1,
)
print('PY FINAL k=%d ll=%.12f iter=%d conv=%d ming=%g' % (
    len(r.pt), r.ll, r.iter, r.convergence, r.min_gradient), flush=True)
