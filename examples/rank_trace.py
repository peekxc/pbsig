import numpy as np 
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import coo_array, random
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._expm_multiply import traceest

## For PROPACK Lanczos bi-orthogonalization, see: scipy.sparse.linalg._svdp
## Contains: Lanczos bidiagonalization algorithm with partial reorthogonalization (but returns singular vectors/values)

## Random PSD matrix 
A = random(40, 40, density=0.05)
A = A @ A.T

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
# %% This seems to be a good setup for getting Lanczos eigenvalues + vectors
import primme
np.random.seed(1234)
v0 = np.random.normal(size=A.shape[0], loc=0)
primme.eigsh(
  A_lo, k=A.shape[0]-1,
  v0=v0[:,np.newaxis],
  ncv=15, which='LM', 
  maxiter=A.shape[0]*2,
  maxBlockSize=0,         # defaults to 0, must be < ncv
  minRestartSize=14,       # defaults to 0, must be < ncv 
  maxPrevRetain=0,        # defaults to 0.
  method="PRIMME_Arnoldi",
  # convtest=lambda eval, evec, resNorm: True, 
  tol = 0.01,
  return_stats=True, 
  return_unconverged=True, 
  raise_for_unconverged=False,
  return_eigenvectors=False
)

## NOTES: 
## Increasing num Lanczos vectors monotonically improves accuracy, though has strong diminishing returns 
## Increasing minRestartSize seemed to improve accuracy of extrema Ritz values, but possibly sacrifices accuracy of interior
## Though it seems to improve accuracy well if close to ncv. Drops the number of outer iterations and the number restarts. 
## Keeps matvecs the same. 
## maxPrevRetain means nothing if minRestartSize = 0, and seems to mean nothing for Lanczos
## maxBlockSize means nothing if minRestartSize = 0, and seems to mean nothing for Lanczos 
## Increasing maxiter has no effect if convtest is always True
## Setting tol = np.inf is equiv to setting convtest to always return True

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

