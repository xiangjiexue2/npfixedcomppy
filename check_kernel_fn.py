import numpy as np
import sys
sys.path.insert(0, r'C:/Users/xxjie/Documents/rebuild/npfixedcomppy/python')
from npfixedcomppy import _core

base = r'C:/Users/xxjie/Documents/rebuild'
X = np.loadtxt(f'{base}/parity_2d.csv', delimiter=',')
# Use a small subset + a simple mu0/beta to compare kernel values
x = X[:50]
mu0 = np.array([[0.5, -0.3], [-1.2, 0.8]])
Sigma = np.array([[1.0, 0.2], [0.2, 1.0]])
# Call the C++ kernel via the 2D solver's mapping — but that's internal.
# Instead, call the public _core.npnorm2Dll with a trivial 1-point init and
# read back the loss at iteration 0, which is -log(dens) where dens = dnpnormND.
# Simpler: use the kernel directly if exposed. Check what _core exposes:
fns = [f for f in dir(_core) if 'norm' in f.lower() or 'dnp' in f.lower()]
print('_core norm/dnp functions:', fns)
