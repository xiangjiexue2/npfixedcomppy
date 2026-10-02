import difflib

r = open(r'C:/Users/xxjie/Documents/rebuild/npfixedcomp2/inst/include/LBFGSB.h', encoding='utf-8', errors='replace').read().splitlines()
p = open(r'C:/Users/xxjie/Documents/rebuild/npfixedcomppy/cpp/LBFGSB.h', encoding='utf-8', errors='replace').read().splitlines()
print('R lines:', len(r), ' PY lines:', len(p))
for line in difflib.unified_diff(r, p, fromfile='R/LBFGSB.h', tofile='PY/LBFGSB.h', n=1):
    print(line)
