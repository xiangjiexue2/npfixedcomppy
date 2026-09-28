"""Build the C++/Eigen extension ``npfixedcomppy._core`` with setuptools +
pybind11.

The whole compute stack lives in C++ (cpp/): the pybind11 module, the
mixing-distribution engine, the per-family kernels and the NNLS solver,
with Eigen (vendored in eigen/) for the linear algebra.

Compiler flags:

* ``/arch:AVX2``  -- SIMD width decided at BUILD time; override with the
  environment variable ``NPFIC_ARCH`` (e.g. ``NPFIC_ARCH=`` for plain x86-64)
  when targeting hosts without AVX2. The same source also compiles for
  gcc/clang (``-mavx2 -O3``).

Threading: this package has NO hand-written OpenMP. The only multi-threading
comes from Eigen's own compile-time-gated parallel GEMM/GEMV, which is
disabled everywhere (Eigen serial) so that all parallelism in the process is
owned by a single, known runtime.
"""

import os
import platform

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

arch = os.environ.get("NPFIC_ARCH", "AVX2")

if platform.system() == "Windows":
    # /utf-8: the headers carry non-ASCII comment text; MSVC's default
    # code page (936 here) would flag C4819 and can mis-decode it.
    cxx_flags = ["/std:c++17", "/O2", "/MD", "/J", "/EHsc", "/utf-8"]
    if arch:
        cxx_flags.append("/arch:" + arch)
    link_flags = []
else:
    cxx_flags = ["-std=c++17", "-O3", "-march=native"]
    if arch == "AVX2":
        cxx_flags = [f for f in cxx_flags if f != "-march=native"]
        cxx_flags.append("-mavx2")
    link_flags = []

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
