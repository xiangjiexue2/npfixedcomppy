import numpy as np
from npfixedcomppy import pnppois

base = r"C:/Users/xxjie/Documents/rebuild"
xp = np.loadtxt(f"{base}/verify_new_xp.txt")
mup = np.array([0.5, 1.5, 3.0, 5.0]); pip = np.array([0.1, 0.4, 0.3, 0.2])

lines = open(f"{base}/ref_new.txt").read().splitlines()
i = 0
rv = {}
while i < len(lines):
    name = lines[i].strip(); cnt = int(lines[i + 1].strip())
    rv[name] = np.array([float(v) for v in lines[i + 2 : i + 2 + cnt]])
    i += 2 + cnt

for name in ["pnppois_ft", "pnppois_flt", "pnppois_lt", "pnppois_llt"]:
    r = rv[name]
    lt, lg = name[7] == "t", name[7] == "l" and name.endswith("t") is False
    # name enc: _ft = lower.tail=T? verify: R w("pnppois_ft", pnppois(xp,mup,pip,1,TRUE,FALSE))
    # so 4th arg = lower.tail, 5th = log.p; ft: lt=F? no: ft = (TRUE, FALSE)?
    pass

# R side (verify_newr.R): pnppois_ft = (TRUE, FALSE) i.e. lt=T,lg=F
#                         pnppois_lt = (FALSE, FALSE) lt=F,lg=F
#                         pnppois_flt= (TRUE, TRUE)  lt=T,lg=T
#                         pnppois_llt= (FALSE, TRUE) lt=F,lg=T
cases = [("pnppois_ft", True, False), ("pnppois_lt", False, False),
         ("pnppois_flt", True, True), ("pnppois_llt", False, True)]
for name, lt, lg in cases:
    p = np.asarray(pnppois(xp, mup, pip, lt, lg), dtype=float)
    r = rv[name]
    print(f"=== {name} (lt={lt}, lg={lg}) ===")
    for k in range(len(xp)):
        flag = "" if (np.isnan(r[k]) == np.isnan(p[k])) else "   <<NAN-PATTERN"
        if not (np.isnan(r[k]) and np.isnan(p[k])) and abs(r[k]-p[k]) > 1e-9:
            flag += "   <<DIFF"
        print(f"  x={xp[k]:4.0f}  R={r[k]!r:26} PY={p[k]!r:26}{flag}")
    print()
