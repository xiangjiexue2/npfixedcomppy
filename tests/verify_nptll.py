"""Verify npfixedcomppy (Rust) nptll against R npfixedcomp2 references."""
import numpy as np
from npfixedcomppy import computemixdist, estpi0

data1000 = np.loadtxt("C:/Users/xxjie/Documents/rebuild/npfc_data_1000.csv", delimiter=",", skiprows=1, ndmin=1)
data5000 = np.loadtxt("C:/Users/xxjie/Documents/rebuild/npfc_data_5000.csv", delimiter=",", skiprows=1, ndmin=1)

REF = {
    "CM_T_INF": dict(
        beta=float("inf"),
        pt=[0, 0.14734546289474806, 1.9718641806855033, 2.9959353188328626],
        pr=[0, 0.53041975408846131, 0.45782007233600264, 0.011760173575535994],
        ll=1722.5203042042685, mg=-9.5284352028102148e-05, it=24, conv=0,
    ),
    "CM_T_5": dict(
        beta=5.0,
        pt=[0, 0.13446866536461727, 1.4727685574951701],
        pr=[0, 0.4309642518858855, 0.56903574811411473],
        ll=1758.3177224453912, mg=-8.3097984315827489e-07, it=11, conv=0,
    ),
    "CM_T_10": dict(
        beta=10.0,
        pt=[0, 0.11319635727224081, 1.6898524671118309],
        pr=[0, 0.46920163072389293, 0.53079836927610702],
        ll=1735.636790396472, mg=-1.842806796048535e-07, it=11, conv=0,
    ),
    "CM_T_INF_FIX": dict(
        beta=float("inf"),
        pt=[-0.5, 1.0982971600707707, 2.3078505261791173],
        pr=[0.29999999999999999, 0.45040047507115044, 0.24959952492884954],
        ll=1733.2284824169278, mg=-1.3348873377813577e-06, it=14, conv=0,
    ),
    "CM_T5_BIG": dict(
        beta=5.0,
        pt=[0, 0.052037754484351266, 1.5691169090352],
        pr=[0, 0.44802334278956746, 0.55197665721043254],
        ll=8955.1284878289407, mg=-1.2211958164698444e-06, it=31, conv=0,
    ),
    "EP_T_INF": dict(
        beta=float("inf"),
        pt=[0, 1.8955324542347831, 2.9717272463614295],
        pr=[0.50501022838764875, 0.47542195962476114, 0.019567811987590125],
        ll=1724.5202875776058, mg=-4.1520565722592302e-06, it=8, conv=0,
    ),
    "EP_T_5": dict(
        beta=5.0,
        pt=[0, 1.5068840539634916],
        pr=[0.43378430633902265, 0.56621569366097735],
        ll=1760.3177224223598, mg=1.3789930017237057e-13, it=4, conv=0,
    ),
}

res = {
    "CM_T_INF": computemixdist(data1000, method="nptll"),
    "CM_T_5": computemixdist(data1000, method="nptll", beta=5),
    "CM_T_10": computemixdist(data1000, method="nptll", beta=10),
    "CM_T_INF_FIX": computemixdist(data1000, method="nptll", mu0=[-0.5], pi0=[0.3]),
    "CM_T5_BIG": computemixdist(data5000, method="nptll", beta=5),
    # fast=False pins the legacy (R-bit-identical) refinement: the REF
    # values below were produced by the R package's legacy path.
    "EP_T_INF": estpi0(data1000, method="nptll", val=2.0, fast=False),
    "EP_T_5": estpi0(data1000, method="nptll", beta=5, val=2.0, fast=False),
}

def relerr(a, b):
    denom = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denom

bad = 0
for tag, r in REF.items():
    got = res[tag]
    print(f"== {tag}  (got family={got.family} flag={got.flag})")
    n = len(r["pt"])
    assert len(got.pt) == n, f"{tag}: npt {len(got.pt)} != {n}"
    for i in range(n):
        e_pt = abs(got.pt[i] - r["pt"][i])
        e_pr = abs(got.pr[i] - r["pr"][i])
        ok = e_pt < 1e-6 and e_pr < 1e-6
        if not ok:
            bad += 1
        print(f"  [{'OK ' if ok else 'BAD'}] pt[{i}]={got.pt[i]!r:22} (ref {r['pt'][i]!r}) d={e_pt:.3e}   pr d={e_pr:.3e}")
    e_ll = relerr(got.ll, r["ll"])
    e_mg = abs(got.min_gradient - r["mg"])
    it_ok = got.iter == r["it"]
    fam_ok = got.family == "npt" and got.flag == "d0"
    conv_ok = got.convergence == r["conv"]
    if e_ll > 1e-9 or e_mg > 1e-8 or not it_ok or not fam_ok or not conv_ok:
        bad += 1
    print(f"  [{'OK ' if e_ll <= 1e-9 else 'BAD'}] ll={got.ll!r} (ref {r['ll']!r}) relerr={e_ll:.3e}")
    print(f"  [{'OK ' if e_mg <= 1e-8 else 'BAD'}] min_gradient={got.min_gradient!r} (ref {r['mg']!r}) d={e_mg:.3e}")
    print(f"  [{'OK ' if it_ok else 'BAD'}] iter={got.iter} (ref {r['it']})  [{'OK ' if conv_ok else 'BAD'}] conv={got.convergence} (ref {r['conv']})")
    print(f"  [{'OK ' if fam_ok else 'BAD'}] family=npt flag=d0")

print()
print("TOTAL BAD:", bad)
raise SystemExit(1 if bad else 0)
