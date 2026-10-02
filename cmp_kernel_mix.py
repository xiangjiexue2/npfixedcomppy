import numpy as np

base = r'C:/Users/xxjie/Documents/rebuild'
r_mix = np.array([float(x) for x in open(f'{base}/r_dnpnormND_init.txt').read().split()])
py_mix = np.load(f'{base}/py_dnpnormND_init.npy')
assert r_mix.shape == py_mix.shape, (r_mix.shape, py_mix.shape)
d = np.abs(r_mix - py_mix)
rel = d / np.maximum(np.abs(r_mix), 1e-300)
print('=== dnpnormND (init mix, n=300) after constant-grouping fix ===')
print('bit_exact   =', int((r_mix == py_mix).sum()), '/', r_mix.size)
print('max_abs_diff=', d.max(), ' max_rel_diff =', rel.max())
print('max|logdiff|=', np.abs(np.log(py_mix) - np.log(r_mix)).max())
i = int(np.argmax(d)); print(f'worst i={i}: R={r_mix[i]!r} PY={py_mix[i]!r}')
