# -*- coding: utf-8 -*-
# python setup.py install
from setuptools import setup
from glob import glob
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
	Pybind11Extension(
	"boundary",
		sorted(glob("boundary.cpp")),
	)
]

setup(
  name="boundary",
  author="Matt Piekenbrock",
  author_email="matt.piekenbrock@gmail.com",
  description="Persistent Betti optimization",
  long_description="",
  ext_modules=ext_modules,
  cmdclass={"build_ext": build_ext},
  zip_safe=False,
  python_requires=">=3.8",
)

# package_dir = \
# {'': 'src'}

# packages = \
# ['set_cover', 'set_cover.sc_ext']

# package_data = \
# {'': ['*'],
#  'set_cover.sc_ext': ['extern/pybind11/*',
#                       'extern/pybind11/detail/*',
#                       'extern/pybind11/stl/*']}

# setup_kwargs = {
#     'name': 'set-cover',
#     'version': '0.1.0',
#     'description': 'My Package with C++ Extensions',
#     'long_description': None,
#     'author': 'Matt Piekenbrock',
#     'author_email': None,
#     'maintainer': None,
#     'maintainer_email': None,
#     'url': None,
#     'package_dir': package_dir,
#     'packages': packages,
#     'package_data': package_data,
# }
# from build import *
# build(setup_kwargs)

# setup(**setup_kwargs)
