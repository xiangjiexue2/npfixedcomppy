import numpy as np

base = r'C:/Users/xxjie/Documents/rebuild'

def load_txt(p):
    return np.array([float(x) for x in open(p).read().split()])

# 1) init-mix density (n values, same order)
r_mix = load_txt(f'{base}/r_dnpnormND_init.txt')
py_mix = np.load(f'{base}/py_dnpnormND_init.npy')
assert r_mix.shape == py_mix.shape, (r_mix.shape, py_mix.shape)
d = np.abs(r_mix - py_mix)
rel = d / np.maximum(np.abs(r_mix), 1e-300)
print('=== dnpnormND (init mix, n=300) ===')
print('n_bit_exact  =', int((r_mix == py_mix).sum()), '/', r_mix.size)
print('max_abs_diff =', d.max(), ' max_rel_diff =', rel.max())
print('max|logdiff| =', np.abs(np.log(py_mix) - np.log(r_mix)).max())
# where are the biggest differences?
i = np.argmax(d)
print(f'worst i={i}: R={r_mix[i]!r} PY={py_mix[i]!r}')

# 2) dnormNDarray lg/nolg. R as.vector = column-major: [col0(all n), col1(all n)]
r_lg = load_txt(f'{base}/r_dnormNDarray_lg.txt')
r_nolg = load_txt(f'{base}/r_dnormNDarray_nolg.txt')
py_lg = np.load(f'{base}/py_dnormNDarray_lg.npy')     # (300, 2) row-major
py_nolg = np.load(f'{base}/py_dnormNDarray_nolg.npy')
r_lg_m = r_lg.reshape(2, 300, order='F')              # -> (2, 300) = (k, n)
r_nolg_m = r_nolg.reshape(2, 300, order='F')
py_lg_m = py_lg.T                                      # (k, n)
py_nolg_m = py_nolg.T
for name, rm, pm in [('lg', r_lg_m, py_lg_m), ('nolg', r_nolg_m, py_nolg_m)]:
    dd = np.abs(rm - pm)
    rr = dd / np.maximum(np.abs(rm), 1e-300)
    print(f'=== dnormNDarray {name} ===')
    print('bit_exact   =', int((rm == pm).sum()), '/', rm.size)
    print('max_abs_diff=', dd.max(), ' max_rel_diff =', rr.max())
