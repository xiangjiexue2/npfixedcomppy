import os, filecmp

r_base = r'C:/Users/xxjie/Documents/rebuild/npfixedcomp2/inst/include'
p_base = r'C:/Users/xxjie/Documents/rebuild/npfixedcomppy/cpp'

# 1) LBFGSpp directory, file by file
r_l = os.path.join(r_base, 'LBFGSpp')
p_l = os.path.join(p_base, 'LBFGSpp')
r_files = sorted(os.listdir(r_l))
p_files = sorted(os.listdir(p_l))
print('R LBFGSpp files:', r_files)
print('PY LBFGSpp files:', p_files)
print('only in R:', set(r_files) - set(p_files))
print('only in PY:', set(p_files) - set(r_files))
for f in sorted(set(r_files) & set(p_files)):
    same = filecmp.cmp(os.path.join(r_l, f), os.path.join(p_l, f), shallow=False)
    print(f'  {f}: {"IDENTICAL" if same else "DIFFERS"}')

# 2) LBFGSB.h
same = filecmp.cmp(os.path.join(r_base, 'LBFGSB.h'), os.path.join(p_base, 'LBFGSB.h'), shallow=False)
print('LBFGSB.h:', 'IDENTICAL' if same else 'DIFFERS')
