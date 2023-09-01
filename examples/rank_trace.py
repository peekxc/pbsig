from typing import * 
import numpy as np 
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import coo_array, random
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._expm_multiply import traceest

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook()

## For PROPACK Lanczos bi-orthogonalization, see: scipy.sparse.linalg._svdp
## Contains: Lanczos bidiagonalization algorithm with partial reorthogonalization (but returns singular vectors/values)

## TODO: 
## First implement Stochastic Lanczos quadrature  (works for any f)
## Then add Hutch++ trace estimator to quadratic form of tridiagonal/Lanczos eigvectors 
## this should just be ew,ev = eig(T); 
## == v.T @ (ev @ f(diag(ew)) @ ev.T) @ v 
## == traceest( (ev @ f(diag(ew)) @ ev.T) )


## % Testing expectation 
outer = lambda v: np.outer(v,v)
Z = np.zeros(shape=(10,10))
for i in range(50000):
  # Z += outer(np.random.normal(size=10, loc=0, scale=1)) 
  Z += outer(np.random.choice([-1.0, +1.0], size=10))
Z / 50000

## TODO: make this a generator that yields pairs (Y,Theta)
## This should be cheapest Lanczos possible that returns n-1 ritz pairs
def lanczos_eigh(A: LinearOperator, v0: np.ndarray = None, info: bool = False, **kwargs):
  v0 = np.random.normal(size=A.shape[1], loc=0.0, scale=1.0) if v0 is None else v0
  _primme_params = dict(k=A.shape[0]-1, ncv=3, maxiter=A.shape[0], method="PRIMME_Arnoldi", tol=np.inf, return_unconverged=True, raise_for_unconverged=False, return_eigenvectors=True)
  _primme_params |= kwargs
  _primme_params['v0'] = v0[:,np.newaxis] if v0.ndim == 1 else v0
  _primme_params['return_stats'] = info
  return primme.eigsh(A, **_primme_params)

## Random PSD matrix 
from math import prod
from scipy.sparse.linalg import svds
from imate import schatten, trace
n = 500
A = random(n, 40, density=0.050, random_state=0)
P = A @ A.T
sv = np.sort(np.abs(svds(A, k = min(A.shape)-1)[1]))
ev = np.sort(np.linalg.eigh(P.todense())[0])

print(f"Density of P: {P.nnz/prod(P.shape)}")
print(f"(A) Schatten-1 / nuclear norm: {np.sum(sv[sv > 1e-16])}")
print(f"(A) Schatten-2 / frobenius norm: {np.sum(sv[sv > 1e-16]**2)**(1/2)}")
# print(f"Numpy verified fro norm: {np.linalg.norm(A.todense(), 'fro')}")
print(f"(P) Schatten-1 / nuclear norm: {np.sum(ev[ev > 1e-16])}")
print(f"(P) Schatten-2 / frobenius norm: {np.sum(ev[ev > 1e-16]**2)**(1/2)}")

## 
print(f"(A) Schatten-1 / nuclear norm, derived from P: {np.sum(ev[ev > 1e-16]**(1/2))}")
print(f"(A) Schatten-2 / frobenius norm, derived from P: {np.sum(ev[ev > 1e-16])**(1/2)}")


schatten(P, p=1.0, method="exact")*P.shape[0]
schatten(P, p=2.0, method="exact")*P.shape[0]**(1/2)
((1/n)*(P @ P).trace())**(1/2)

schatten(A, gram=True, p=2.0, method="slq")
schatten(A, gram=True, p=1.0, method="slq")

## Rank / matrix power 
n = 500
A = random(n, 40, density=0.050, random_state=0)
P = A @ A.T
sv = np.sort(np.abs(svds(A, k = min(A.shape)-1)[1]))
ev = np.sort(np.linalg.eigh(P.todense())[0])

np.sum(ev > 1e-12)
np.sum(np.abs(ev)**(1e-6))

## This works: matches the nuclear norm about
schatten(P, p=1, min_num_samples=100, max_num_samples=200, gram=False, method="slq")*A.shape[0]
np.sum()


schatten(A, p=1, min_num_samples=100, max_num_samples=200, gram=True, method="slq")*A.shape[0]

from imate import InterpolateTrace
from splex import * 
from pbsig.linalg import up_laplacian
from bokeh.models import Range1d
from bokeh.layouts import row, column

X = np.random.uniform(size=(30,2))
S = delaunay_complex(X)
L = up_laplacian(S, p = 1, normed = True)
L_ew = np.linalg.eigh(L.todense())[0]
numerical_rank = np.sum(np.where(L_ew > 1e-8, 1.0, 0.0))
np.histogram(L_ew)


normalize = lambda x: (x - min(x))/(max(x) - min(x))
p = figure(width=500, height=250, x_axis_type="log")
p.line(L_ew, normalize(np.cumsum(L_ew)), color="red")
p.scatter(L_ew, normalize(np.cumsum(L_ew)), size=1.5)
p.x_range = Range1d(1e-6, 3.1)
# show(p)

ti = np.geomspace(1e-6, 3.0, 150)
f = InterpolateTrace(L, p=2.0, kind='spl', ti=list(ti))
ft = np.array([f(t) for t in ti])

# q = figure(width=500, height=200, x_axis_type="log")
p.line(ti, normalize(ft), color="blue")
p.scatter(ti, normalize(ft), color="blue",  size=1.5)
# p.x_range = Range1d(1e-6, 3.1)
show(p)

show(column(p,q))


## Custom function

from imate._trace_estimator import trace_estimator
from imate.functions import pyFunction




trace(A, p = 0.01, method="slq",  return_info=True)

trace(P, p = 1e-16, method="slq")



np.linalg.matrix_rank(A.todense())

schatten(A, p=1.0, method="slq")*A.shape[0]
np.sum(A.todense() == 0)/(1500*1500)

from imate import schatten
schatten(A, p=1.0, method="exact")

schatten(A, p=1.0, method="exact")*A.shape[0]
schatten(A, p=1.0, method="exact")*A.shape[0]

import timeit
timeit.timeit(lambda: np.sum(np.sort(np.linalg.eigh(A.todense())[0])), number=100)/100
timeit.timeit(lambda: schatten(A, p=1.0)*A.shape[0], number=100)/100

schatten(A, p=0, method="cholesky")

np.sum(np.abs(true_ew))

true_ew = np.sort(np.linalg.eigh(A.todense())[0])
v = np.random.choice([-1.0, 1.0], size = A.shape[0])
# v /= np.linalg.norm(v)
ew,ev,info = lanczos_eigh(A, info=True, return_unconverged=True, ncv=15, v0=v, tol=0, maxiter=10*A.shape[0])
np.histogram(ew)
np.sum(f(ew))

v = np.random.choice([-1.0, 1.0], size = A.shape[0])
v /= np.linalg.norm(v)
A @ v

from scipy.sparse.linalg import eigsh
# eigsh(A, k=A.shape[0]-1, ncv=)

## Stochastic Lanczos quadrature
from numbers import Integral
from pbsig.linalg import eigsh_block_shifted
f = lambda x: np.where(x >= 1e-8, 1.0, 0.0)
def SLQ(A: LinearOperator, f: Callable, nv: int = 30, dist: str = "rademacher", seed: int = None, **kwargs):
  """Stochastic Lanczos quadrature"""
  if seed is not None:
    np.random.set_state(int(100*np.random.rand()))
  # rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)
  n = A.shape[0]
  tr_est = 0.0
  running_theta, running_weights = [], []
  sample_v = lambda: np.random.choice([-1.0, 1.0], size = n) if dist == "rademacher" else np.random.normal(loc=0, scale=1.0, size = n)
  for j in range(nv):
    v = sample_v()
    v /= np.linalg.norm(v)
    Theta, Y = lanczos_eigh(A, v0 = v, k = k, **kwargs)
    # Theta, Y = lanczos_eigh(A, return_unconverged=True, ncv=15, v0=v, tol=0, maxiter=10*A.shape[0])
    print(np.sum(f(Theta)))
    running_weights.append(np.sum(f(Theta)))
    running_theta.append(np.sum((Y[0,:]**2)))
    # running_avg.append(np.sum(f(Theta)))
    # tr_est += n * np.sum((Y[0,:]**2) * f(Theta)) ## THIS ISNT CORRECT
    ## np.trace(Y @ np.diag(f(Theta)) @ Y.T) ## THIS WORKS 
    # e1 = np.zeros(n)[:,np.newaxis]
    # e1[0] = 1.0
    ## e1.T @ Y @ np.diag(f(Theta)) @ Y.T @ e1 ## THIS DOES NOT 
    # tr_est += (Y @ np.diag(f(Theta)) @ Y.T)[0,0]## THIS DOES NOT 
  # tr_est *= n/nv if dist == "rademacher" else 1.0/nv
  return running_weights, running_theta

true_ew = np.sort(np.linalg.eigh(A.todense())[0])
true_rank = np.linalg.matrix_rank(A.todense())
assert np.sum(f(true_ew)) == true_rank

## SO maxiter is just determining the magnitude basically, up to the order of matrix 
f = lambda x: np.abs(x)
# convtest(eval, evec, resNorm)
rw,rt = SLQ(A, f, nv=250, dist="rademacher", ncv=15, maxiter=150*A.shape[0], tol=1e-6)
np.sum(true_ew[true_ew >= 1e-8])

## UNABLE TO REPLICATE
trace_est_arr = np.array(rw)
p = figure(width=250, height=200)
# p.line(np.arange(len(trace_ests)), np.array(trace_ests)*A.shape[0])
p.line(np.arange(len(trace_est_arr)), (np.cumsum(trace_est_arr))/(np.arange(len(trace_est_arr))+1)) # n*(cnt_est/ii);
show(p)



from scipy.sparse.linalg import eigsh
np.sum(np.abs(eigsh(A, k=A.shape[0]-1, tol=0)[0]))


## Yep this is return basically just the order of the matrix
v = np.random.choice([-1.0, 1.0], size = A.shape[0])
Theta, Y = lanczos_eigh(A, v0 = v)
Y_op = aslinearoperator(Y)
D_op = aslinearoperator(np.diag(f(Theta)))
traceest(Y_op @ D_op @ Y_op.T, 10)


f = lambda x: np.sum(np.abs(x))

# 15 seems good here
# slq_vars = [np.var([SLQ(A, f, nv=nv) for i in range(30)]) for nv in range(1,100,10)]


np.sum(np.abs(true_ew))



np.sort(lanczos_eigh(A)[0])

np.linalg.matrix_rank(A.todense())


## True rank 
print(np.linalg.matrix_rank(A.todense()))
print(np.sum(A.diagonal()))

## See: https://curvlinops.readthedocs.io/en/latest/basic_usage/example_verification_spectral_density.html#
# from curvlinops.papyan2020traces.spectrum import fast_lanczos
A_lo = aslinearoperator(A)

## Main Lanczos parameters
# maxiter: Maximum number of iterations.
# v0: (N x i) Initial guesses to the eigenvectors.
# ncv: The maximum size of the basis
# maxBlockSize: Maximum number of vectors added at every iteration.
# minRestartSize: Number of approximate eigenvectors kept during restart.
# maxPrevRetain: Number of approximate eigenvectors kept from previous iteration in restart. Also referred as +k vectors in GD+k.
# if (method == PRIMME_Arnoldi) {
#   primme->restartingParams.maxPrevRetain      = 0;
#   primme->correctionParams.precondition       = 0;
#   primme->correctionParams.maxInnerIterations = 0;
# %% This seems to be a good setup for getting Lanczos eigenvalues + vectors
import primme
np.random.seed(45)
v0 = np.random.normal(size=A.shape[0], loc=0, scale=1.0)
primme.eigsh(
  A_lo, k=A.shape[0]-1,
  v0=v0[:,np.newaxis],
  ncv=3, which='LM', 
  maxiter=A.shape[0],
  maxBlockSize=0,         # defaults to 0, must be < ncv
  minRestartSize=5,       # defaults to 0, must be < ncv 
  maxPrevRetain=0,        # defaults to 0.
  method="PRIMME_Arnoldi",
  # convtest=lambda eval, evec, resNorm: True, 
  tol = 0.01,
  return_stats=True, 
  return_unconverged=True, 
  raise_for_unconverged=False,
  return_eigenvectors=False
)



lanczos_eigh(A)

## quadrature over this 
# https://github.com/f-dangel/curvlinops/blob/45c13bac2b71304a5d77c52267612cbd1d276280/curvlinops/papyan2020traces/spectrum.py#L74
# https://github.com/stdogpkg/emate/blob/97965001e14d2c8689df99393036179cfec1124b/emate/symmetric/tfops/slq.py#L49C18-L49C18

## NOTES: 
## Increasing num Lanczos vectors monotonically improves accuracy, though has strong diminishing returns 
## Increasing minRestartSize seemed to improve accuracy of extrema Ritz values, but possibly sacrifices accuracy of interior
## Though it seems to improve accuracy well if close to ncv. Drops the number of outer iterations and the number restarts. 
## Keeps matvecs the same. 
## maxPrevRetain means nothing if minRestartSize = 0, and seems to mean nothing for Lanczos
## maxBlockSize means nothing if minRestartSize = 0, and seems to mean nothing for Lanczos 
## Increasing maxiter has no effect if convtest is always True
## Setting tol = np.inf is equiv to setting convtest to always return True


#%% 
traceest(A_lo, 10)

# Perform SciPy quad integration
integral = 0.0

for k in range(num_iterations):
  integral += alpha[k] * quad(func, -np.inf, np.inf)[0]




# %% 
def lanczos(A: LinearOperator, k: int, ncv: int) -> tuple[np.ndarray, np.ndarray]:
  """Lanczos iterations for large-scale problems (no reorthogonalization step).

  Implements algorithm 2 of Papyan, 2020 (https://jmlr.org/papers/v21/20-933.html).

  Params:
    A: Symmetric linear operator.
    k: Number of iterations. 
    ncv: Number of Lanczos vectors.

  Returns:
    alpha: diagonal terms of the tridiagonal matrix. 
    beta: off-diagonal terms of the tridiagonal matrix
  """
  assert ncv >= 3, "Need at least 3 Lanczos vectors to "
  alphas, betas = np.zeros(ncv), np.zeros(ncv - 1)
  dim = A.shape[1]
  v, v_prev = None, None
  for m in range(ncv):
    if m == 0:
      v = np.randn(dim)
      v /= np.norm(v)
      v_next = A @ v
    else:
      v_next = A @ v - betas[m - 1] * v_prev
      alphas[m] = np.inner(v_next, v)
      v_next -= alphas[m] * v
      last = m == ncv - 1
      if not last:
        betas[m] = np.linalg.norm(v_next)
        v_next /= betas[m]
        v_prev = v
        v = v_next
  
traceest(A_lo, 10)

def create_matrix(dim: int = 2000) -> ndarray:
    """Draw a matrix from the matrix distribution used in papyan2020traces, Figure 15a.

    Args:
        dim: Matrix dimension.

    Returns:
        A sample from the matrix distribution.k
    """
    X = zeros((dim, dim))
    X[0, 0] = 5
    X[1, 1] = 4
    X[2, 2] = 3

    Z = randn(dim, dim)

    return X + 1 / dim * matmul(Z, Z.transpose())