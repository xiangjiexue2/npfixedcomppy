import os
import sys

import numpy as np

from npfixedcomppy import computemixdist

tag = "EXACT" if os.environ.get("NPFIC_2D_EXACT") else "FAST "
path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\xxjie\Documents\rebuild\itercmp\iterdata_2d.csv"
x = np.loadtxt(path, delimiter=",")
r = computemixdist(x, method="npnorm2Dll")
pr = np.array(r.pr)
order = np.argsort(pr)[::-1]
print("==%s== ll=%r iter=%d k=%d" % (tag, r.ll, r.iter, len(r.pt)))
for i in order:
    print("  w=%.6f mu=(%.6f,%.6f)" % (pr[i], r.pt[i][0], r.pt[i][1]))
