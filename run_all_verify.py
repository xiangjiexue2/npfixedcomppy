"""Run every tests/verify_*.py as a subprocess, capture output, summarize."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, "tests")
PY = sys.executable

names = sorted(f for f in os.listdir(TESTS)
               if f.startswith("verify_") and f.endswith(".py"))
print("python:", PY)
print("tests :", ", ".join(names))
failed = []
for name in names:
    p = os.path.join(TESTS, name)
    r = subprocess.run([PY, p], capture_output=True, text=True, timeout=600)
    out = (r.stdout or "") + (r.stderr or "")
    tail = "\n".join(out.strip().splitlines()[-4:])
    ok = r.returncode == 0
    print(f"\n===== {name}  exit={r.returncode}  {'OK' if ok else 'FAILED'}")
    print(tail)
    if not ok:
        failed.append(name)
print("\n" + ("ALL PASS" if not failed else f"FAILED: {failed}"))
