# -*- coding: utf-8 -*-
import os 
import sysconfig
import distutils.sysconfig
from typing import Any, Dict
from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
from os.path import normpath

## Get base path to package
base_path = os.path.dirname(__file__)

## Print the platform- and compiler-dependent flags 
flags = distutils.sysconfig.get_config_var("CFLAGS")
print(f"COMPILER FLAGS: { str(flags) }")

## Configure additional compiler flags
compile_args = sysconfig.get_config_var('CFLAGS').split()
compile_args += ["-std=c++20", "-Wall", "-Wextra", "-O2"]
# compile_args += ["-std=c++20", "-Wall", "-Wextra", "-march=native", "-O3", "-fopenmp"]
link_args = []
# link_args = ["-fopenmp"]

## Configure includes + extension modules
extensions = ['laplacian', 'combinatorial', 'persistence', 'landmark']
include_dirs = [
  normpath(base_path + '/include'), 
  normpath(base_path + '/extern/pybind11/include'), 
  normpath(base_path + '/extern/eigen'),
  normpath(base_path + '/pbsig/src/pbsig/')
]

## Configure the native extension modules
ext_modules = []
for ext in extensions:
  ext_module = Pybind11Extension(
    f"_{ext}", 
    sources = [normpath(f"src/pbsig/{ext}.cpp")], # setuptools require relative paths 
    include_dirs=include_dirs, 
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    language='c++20', 
    cxx_std=1
  )
  ext_modules.append(ext_module)

# Develop: pip install --editable . --no-deps --no-build-isolation
# Build: python -m build --skip-dependency-check --no-isolation --wheel
setup(
  name="pbsig",
  author="Matt Piekenbrock",
  author_email="matt.piekenbrock@gmail.com",
  description="Spectral Rank Invariant",
  long_description="",
  ext_modules=ext_modules,
  cmdclass={'build_ext': build_ext},
  zip_safe=False, # needed for platform-specific wheel 
  python_requires=">=3.8",
  package_dir={'': 'src'}, # < root >/src/* contains packages
  packages=['pbsig', 'pbsig.ext'],
  package_data={'pbsig': ['data/*.bsp', 'data/*.txt', 'data/*.csv']},
)