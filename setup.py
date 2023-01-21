# -*- coding: utf-8 -*-
import os 
import sysconfig
import distutils.sysconfig
from typing import Any, Dict
from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

## Get base path to package
base_path = os.path.dirname(__file__)

## Print the platform- and compiler-dependent flags 
flags = distutils.sysconfig.get_config_var("CFLAGS")
print(f"COMPILER FLAGS: { str(flags) }")

## Configure additional compiler flags
compile_args = sysconfig.get_config_var('CFLAGS').split()
# compile_args = ["-O3" if (arg[:2] == "-O" and int(arg[2]) in [0,1,2,3]) else arg for arg in compile_args]
compile_args += ["-std=c++20", "-Wall", "-Wextra" "-O0"]
# compile_args += ["-march=native", "-O3", "-fopenmp"] ## If optimizing for performance "-fopenmp"
# extra_compile_args += "-O0" ## debug mode  
# extra_compile_args = list(set(extra_compile_args))

## Configure the native extension modules
ext_modules = [
  Pybind11Extension(
    '_boundary', 
    sources = ['src/pbsig/boundary.cpp'], 
    # include_dirs=['/Users/mpiekenbrock/diameter/extern/pybind11/include'], 
    extra_compile_args=compile_args,
    language='c++17', 
    cxx_std=1
  ), 
  Pybind11Extension(
    '_lanczos', 
    sources = ['src/pbsig/lanczos_spectra.cpp'], 
    include_dirs=[
      '/Users/mpiekenbrock/pbsig/extern/eigen',
      '/Users/mpiekenbrock/pbsig/extern/pybind11/include',
      '/Users/mpiekenbrock/pbsig/extern/spectra/include'
    ], 
    extra_compile_args=compile_args,
    language='c++17', 
    cxx_std=1
  ),
  Pybind11Extension(
    '_laplacian', 
    sources = ['src/pbsig/laplacian.cpp'], 
    include_dirs=[
      '/Users/mpiekenbrock/pbsig/extern/pybind11/include', 
      '/Users/mpiekenbrock/pbsig/extern/pthash/include',
      '/Users/mpiekenbrock/pbsig/extern/pthash/external', 
      '/Users/mpiekenbrock/pbsig/src/pbsig/'
    ], 
    extra_compile_args=compile_args,
    language='c++17', 
    cxx_std=1
  ), 
  Pybind11Extension(
    '_combinatorial', 
    sources = ['src/pbsig/combinatorial.cpp'], 
    include_dirs=[
      '/Users/mpiekenbrock/pbsig/extern/pybind11/include'
    ], 
    extra_compile_args=compile_args,
    language='c++17', 
    cxx_std=1
  ),
  Pybind11Extension(
    '_pbn', 
    sources = ['src/pbsig/pbn.cpp'], 
    extra_compile_args=compile_args,
    language='c++17', 
    cxx_std=1
  ), 
  Pybind11Extension(
    '_persistence', 
    sources = ['src/pbsig/persistence.cpp'], 
    include_dirs=[
      '/Users/mpiekenbrock/pbsig/extern/eigen'
    ], 
    extra_compile_args=compile_args,
    language='c++20', 
    cxx_std=1
  )
]

# Build: python -m build --skip-dependency-check --no-isolation --wheel
setup(
  name="pbsig",
  author="Matt Piekenbrock",
  author_email="matt.piekenbrock@gmail.com",
  description="Persistent Betti Signatures",
  long_description="",
  ext_modules=ext_modules,
  cmdclass={'build_ext': build_ext},
  zip_safe=False, # needed for platform-specific wheel 
  python_requires=">=3.8",
  package_dir={'': 'src'}, # < root >/src/* contains packages
  packages=['pbsig', 'pbsig.ext'],
  package_data={'pbsig': ['data/*.bsp', 'data/*.txt', 'data/*.csv']},
)