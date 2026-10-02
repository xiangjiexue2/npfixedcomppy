import os, sys, shutil

src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python", "npfixedcomppy")
print("SRC:", src)
print("  exists:", os.path.isdir(src))

cands = {}
try:
    import npfixedcomppy as _m
    cands["current"] = os.path.dirname(os.path.abspath(_m.__file__))
except Exception as e:
    print("  current import failed:", e)

sitepkg = os.path.join(sys.base_prefix, "Lib", "site-packages", "npfixedcomppy")
print("SITE:", sitepkg, "exists:", os.path.isdir(sitepkg))

# sync every .py from src into the site-packages copy (keeps .py and .pyd in
# lockstep; the .pyd was already synced, the .py was stale)
if os.path.isdir(sitepkg):
    for f in sorted(os.listdir(src)):
        if not f.endswith(".py"):
            continue
        s = os.path.join(src, f)
        d = os.path.join(sitepkg, f)
        same = os.path.exists(d) and open(s, "rb").read() == open(d, "rb").read()
        if not same:
            shutil.copy2(s, d)
            print("  copied ->", f)
        else:
            print("  same   ->", f)
else:
    print("  (site-packages copy absent; nothing to sync)")
