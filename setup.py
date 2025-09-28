from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os, glob

PDQ_CPP_ROOT = os.path.normpath("ThreatExchange/pdq/cpp")
if not os.path.exists(os.path.join(PDQ_CPP_ROOT, "hashing", "pdqhashing.cpp")):
    raise RuntimeError(
        f"PDQ sources not found under {PDQ_CPP_ROOT}. "
        "If it's a submodule, run: git submodule update --init --recursive"
    )

all_cpp = glob.glob(os.path.join(PDQ_CPP_ROOT, "**", "*.cpp"), recursive=True)
cpp_files = []
for p in all_cpp:
    p = os.path.normpath(p).replace("\\", "/")
    if "/bin/" in p or "/io/" in p:
        continue
    cpp_files.append(p)

include_dirs = {
    ".", "ThreatExchange",
    os.path.normpath("ThreatExchange/pdq").replace("\\", "/"),
    os.path.normpath("ThreatExchange/pdq/cpp").replace("\\", "/"),
    np.get_include(),
}

is_windows = (os.name == "nt")
if is_windows:
    extra_compile_args = ["/std:c++17", "/EHsc", "/O2", "/bigobj"]
    extra_link_args    = ["/LTCG"]
    define_macros = [
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        ("WIN32_LEAN_AND_MEAN", "1"),
        ("NOMINMAX", "1"),
        ("_USE_MATH_DEFINES", "1"),
        ("_CRT_SECURE_NO_WARNINGS", "1"),
    ]
    libraries = []
else:
    extra_compile_args = ["-std=c++17", "-O2", "-fPIC"]
    extra_link_args    = []
    define_macros = [
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        ("__STDC_FORMAT_MACROS", None),
    ]
    libraries = []

ext = Extension(
    "pdqhash.bindings",
    sources=["pdqhash/bindings.pyx"] + cpp_files,
    include_dirs=sorted(include_dirs),
    language="c++",
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    define_macros=define_macros,
    libraries=libraries,
)

setup(
    ext_modules=cythonize([ext], compiler_directives={"language_level": 3}),
)