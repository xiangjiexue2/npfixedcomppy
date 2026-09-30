"""Build the C++/Eigen extension ``npfixedcomppy._core`` with setuptools +
pybind11.

The whole compute stack lives in C++ (cpp/): the pybind11 module, the
mixing-distribution engine, the per-family kernels and the NNLS solver,
with Eigen (vendored in eigen/) for the linear algebra.

Compiler flags:

* SIMD width is decided at BUILD time by the build machine's own ISA:
  MSVC ``/arch:`` auto and gcc/clang ``-march=native`` enable the widest
  instruction set the compiler's host supports (AVX2/AVX512/FMA on x86-64,
  NEON/SVE on aarch64), so Eigen picks its widest packet traits
  automatically. Note the build artifact only runs on hardware with the
  same or a superset of that instruction set.

Threading: this package has NO hand-written OpenMP. The only multi-threading
comes from Eigen's own compile-time-gated parallel GEMM/GEMV: at build time
the compiler is probe-compiled with the OpenMP flag and, when it supports
the flag, the flag is added to the extension build so Eigen's
``EIGEN_HAS_OPENMP`` gate turns on its internal parallel loops. When the
compiler (or the flag) is unavailable, the build falls back to serial Eigen.
No project source file is ever modified for threading — only this build-time
flag decision.
"""

import os
import platform
import shutil
import subprocess
import tempfile

from setuptools import Extension, setup

try:
    import pybind11
except ImportError:  # pragma: no cover - build-time only
    import sys

    sys.exit("pybind11 is required to build npfixedcomppy (pip install pybind11)")

from pybind11.setup_helpers import build_ext  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INCLUDE_DIRS = [
    os.path.join(HERE, "cpp"),
    os.path.join(HERE, "eigen"),
    pybind11.get_include(),
]

if platform.system() == "Windows":
    # /utf-8: the headers carry non-ASCII comment text; MSVC's default
    # code page (936 here) would flag C4819 and can mis-decode it.
    cxx_flags = ["/std:c++17", "/O2", "/MD", "/J", "/EHsc", "/utf-8"]
    link_flags = []
else:
    # -march=native: widest ISA the build machine supports (x86-64: AVX2/
    # AVX512/FMA as available; aarch64: NEON/SVE); Eigen picks packet traits
    # from it automatically. The artifact only runs on hardware with the
    # same or a superset of this instruction set.
    cxx_flags = ["-std=c++17", "-O3", "-march=native"]
    link_flags = []

# ---------------------------------------------------------------------------
# OpenMP auto-detection (build-time flag probe only).
#
# Per the project constraint there are NO hand-written OpenMP loops in this
# codebase; threading exists only through Eigen's compile-time gate
# (``EIGEN_HAS_OPENMP`` in eigen/Eigen/src/Core/util/Macros.h, set from
# ``_OPENMP``) that the compiler flag itself switches on. So the whole
# threading decision is one build-time flag: if the C++ compiler the build
# uses accepts an OpenMP flag, that flag is added to the extension;
# otherwise the build is exactly as it was (serial Eigen).
#
# The probe actually test-compiles a trivial program with the candidate
# flag (the C API the flag is for is tiny, but MSVC/MSYS2/MinGW each behave
# differently, so a real compile is the only reliable answer).
# ---------------------------------------------------------------------------
_OMP_FLAG = {
    "msvc": "/openmp",
    "unix": "-fopenmp",
}


def _openmp_supported() -> bool:
    """True when a test-compile with the OpenMP flag succeeds."""
    if platform.system() == "Windows":
        compiler = os.environ.get("CC") or shutil.which("cl")
        kind = "msvc"
    else:
        compiler = (
            os.environ.get("CXX")
            or os.environ.get("CC")
            or shutil.which("g++")
            or shutil.which("clang++")
        )
        kind = "unix"
    if not compiler:
        return False
    flag = _OMP_FLAG[kind]
    src = (
        "#include <omp.h>\n"
        "int main() { return omp_get_max_threads() > 0 ? 0 : 1; }\n"
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "openmp_probe.cxx")
            out_path = os.path.join(tmp, "openmp_probe.bin")
            with open(src_path, "w") as f:
                f.write(src)
            cmd = [compiler, flag]
            if kind == "msvc":
                cmd += ["/nologo", src_path, "/Fe" + out_path, "/link",
                        "/NOLOGO"]
            else:
                cmd += [src_path, "-o", out_path]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
            return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


_omp_flag = _OMP_FLAG["msvc" if platform.system() == "Windows" else "unix"]
if _openmp_supported():
    # The OpenMP flag goes on the compile (and, for the unix linkers, the
    # link) line: it defines _OPENMP, which is the whole of Eigen's
    # threading gate. No project source changes are involved.
    cxx_flags.append(_omp_flag)
    if platform.system() != "Windows":
        link_flags.append(_omp_flag)
    print(f"npfixedcomppy: OpenMP detected, building Eigen with {_omp_flag}")
else:
    print(
        "npfixedcomppy: OpenMP not detected, building serial Eigen "
        "(no OpenMP flags)"
    )

ext = Extension(
    "npfixedcomppy._core",
    sources=[
        "cpp/npfc_py.cpp",
        "cpp/npfc_nnls.cpp",
    ],
    include_dirs=INCLUDE_DIRS,
    language="c++",
    extra_compile_args=cxx_flags,
    extra_link_args=link_flags,
)

setup(
    name="npfixedcomppy",
    version="0.1.0",
    description=(
        "Non-parametric estimation of mixing distributions with fixed "
        "components (C++/Eigen + pybind11 port of the R package npfixedcomp2)"
    ),
    packages=["npfixedcomppy"],
    package_dir={"": "python"},
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.9",
    zip_safe=False,
)
