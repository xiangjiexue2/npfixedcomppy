import numpy as np

base = r'C:/Users/xxjie/Documents/rebuild'
X = np.loadtxt(base + '/parity_2d.csv', delimiter=',')
pt = np.loadtxt(base + '/py_pts_2d.csv', delimiter=',')
pr = np.loadtxt(base + '/py_pr_2d.txt', delimiter=',').ravel()

# independent 2-D standard-normal mixture log-likelihood, Sigma = I
# dmvn(x; m, I) = (2*pi)^-1 * exp(-0.5 * ||x-m||^2)
ll = 0.0
for i in range(X.shape[0]):
    s = 0.0
    for j in range(pt.shape[0]):
        d = X[i] - pt[j]
        s += pr[j] * np.exp(-0.5 * d @ d) / (2 * np.pi)
    ll += np.log(s)
ll = -ll
print('numpy ll on PY pts =', repr(ll))
print('PY   ll           =', repr(848.747282034907))
print('R    ll (dnpnormND on PY pts) =', repr(1219.2788358969024))
print()
print('free pts (first 4):')
print(pt[:4])
print('free pr:', pr[:4])
print('fixed pt:', pt[4], 'fixed pr:', pr[4])
