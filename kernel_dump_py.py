import numpy as np
from npfixedcomppy import _core

base = r'C:/Users/xxjie/Documents/rebuild'
X = np.loadtxt(f'{base}/parity_2d.csv', delimiter=',')
ipt = np.loadtxt(f'{base}/initpt_2d.csv', delimiter=',')
ipr = np.loadtxt(f'{base}/initpr_2d.txt', delimiter=',').ravel()
Sigma = np.eye(2)

# 1) dnpnormND at the 36 init points (the mapping used at iter 0)
py_mix = _core.dnpnormND(X, ipt, ipr, Sigma, False)
np.save(f'{base}/py_dnpnormND_init.npy', py_mix)

# 2) dnormNDarray at a mid-range support set (2 points), lg both ways
mu2 = np.array([[0.5, -0.3], [-1.2, 0.8]])
Sigma2 = np.array([[1.0, 0.2], [0.2, 1.0]])
np.save(f'{base}/py_dnormNDarray_lg.npy', _core.dnormNDarray(X, mu2, Sigma2, True))
np.save(f'{base}/py_dnormNDarray_nolg.npy', _core.dnormNDarray(X, mu2, Sigma2, False))
print('PY dnpnormND_init: n=', len(py_mix), ' sum=', py_mix.sum(),
      ' min=', py_mix.min(), ' max=', py_mix.max(), ' any0=', int((py_mix <= 0).sum()))
print('PY dnormNDarray  : shape', _core.dnormNDarray(X, mu2, Sigma2, False).shape)
