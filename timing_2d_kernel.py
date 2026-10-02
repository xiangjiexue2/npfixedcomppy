import time
import numpy as np
from npfixedcomppy import dnpnormND, dnormNDarray

base = r'C:/Users/xxjie/Documents/rebuild'
X = np.loadtxt(base + '/parity_2d.csv', delimiter=',')
pt = np.loadtxt(base + '/py_pts_2d.csv', delimiter=',')
pr = np.loadtxt(base + '/py_pr_2d.txt', delimiter=',').ravel()
gp = np.loadtxt(base + '/grid_2d.csv', delimiter=',')
dnpnormND(X, pt, pr, np.eye(2))
dnormNDarray(X, gp, np.eye(2))
t0 = time.perf_counter()
for _ in range(200):
    dnpnormND(X, pt, pr, np.eye(2))
t1 = time.perf_counter()
for _ in range(200):
    dnormNDarray(X, gp, np.eye(2))
t2 = time.perf_counter()
print('dnpnormND      (n=300, k=5) : %.4f ms/call' % ((t1 - t0) * 10))
print('dnormNDarray   (n=300, k=104): %.4f ms/call' % ((t2 - t1) * 10))
