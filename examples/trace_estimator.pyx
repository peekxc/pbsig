# cython: language_level=3
# distutils: language = c++
from imate._trace_estimator import trace_estimator
from imate._trace_estimator cimport trace_estimator
from imate.functions import pyFunction
from imate.functions cimport pyFunction, Function, Identity
from imate.__version__ import __version__

def slq_method_f(
  A,
  gram=False,
  p=1.0,
  return_info=False,
  parameters=None,
  min_num_samples=10,
  max_num_samples=50,
  error_atol=None,
  error_rtol=1e-2,
  confidence_level=0.95,
  outlier_significance_level=0.001,
  lanczos_degree=20,
  lanczos_tol=None,
  orthogonalize=0,
  num_threads=0,
  num_gpu_devices=0,
  verbose=False,
  plot=False,
  gpu=False
):
  cdef Function* matrix_function = new Identity()
  py_matrix_function = pyFunction()
  py_matrix_function.set_function(matrix_function)

  trace, info = trace_estimator(
    A,
    parameters,
    py_matrix_function,
    gram,
    p,
    min_num_samples,
    max_num_samples,
    error_atol,
    error_rtol,
    confidence_level,
    outlier_significance_level,
    lanczos_degree,
    lanczos_tol,
    orthogonalize,
    num_threads,
    num_gpu_devices,
    verbose,
    plot,
    gpu
  )

  del matrix_function
  if return_info:
    return trace, info
  else:
    return trace