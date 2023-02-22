from typing import * 
from numpy.typing import ArrayLike
from numbers import *

import numpy as np 
import primme
from math import *
from scipy.sparse import *
from scipy.sparse.linalg import * 
from scipy.sparse.csgraph import *
from splex.combinatorial import rank_combs

## Local imports
from .meta import *
from .simplicial import * 
from .combinatorial import * 
import _lanczos as lanczos
import _laplacian as laplacian
from splex import *

from scipy.sparse.linalg import eigs as truncated_eig
from scipy.linalg import eigh, eig as dense_eig
from scipy.spatial import KDTree
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components, floyd_warshall

def cmds(D: ArrayLike, d: int = 2, coords: bool = True, pos: bool = True):
  ''' 
  Computes classical MDS (cmds) 
    D := squared distance matrix
  '''
  n = D.shape[0]
  H = np.eye(n) - (1.0/n)*np.ones(shape=(n,n)) # centering matrix
  evals, evecs = np.linalg.eigh(-0.5 * H @ D @ H)
  evals, evecs = evals[(n-d):n], evecs[:,(n-d):n]

  # Compute the coordinates using positive-eigenvalued components only     
  if coords:               
    w = np.flip(np.maximum(evals, np.repeat(0.0, d)))
    Y = np.fliplr(evecs) @ np.diag(np.sqrt(w))
    return(Y)
  else: 
    ni = np.setdiff1d(np.arange(d), np.flatnonzero(evals > 0))
    if pos:
      evecs[:,ni], evals[ni] = 1.0, 0.0
    return(np.flip(evals), np.fliplr(evecs))

def pca(x: ArrayLike, d: int = 2, center: bool = False, coords: bool = True) -> ArrayLike:
	''' PCA embedding '''
	if is_pairwise_distances(x) or is_distance_matrix(x):
		return(cmds(x, d))
	assert is_point_cloud(x), "Input should be a point cloud, not a distance matrix."
	if center: x -= x.mean(axis = 0)
	evals, evecs = np.linalg.eigh(np.cov(x, rowvar=False))
	idx = np.argsort(evals)[::-1] # descending order to pick the largest components first 
	if coords:
		return(np.dot(x, evecs[:,idx[range(d)]]))
	else: 
		return(np.flip(evals)[range(d)], np.fliplr(evecs)[:,range(d)])


# Neumanns majorization doesn't apply? rank_lb = sum(np.sort(diagonal(A)) >= tol)
## TODO: replace with new operator norm bounds for the Laplacian?
def rank_bound(A: Union[ArrayLike, spmatrix, LinearOperator], upper: bool = True) -> int:
  assert A.shape[0] == A.shape[1], "Matrix 'A' must be square"
  if upper: 
    k_ = structural_rank(A) if isinstance(A, spmatrix) else A.shape[0]
    bound = min([A.shape[0], A.shape[1], sum(np.array(diagonal(A)) != 0), k_])  
  else:
    ## https://konradswanepoel.wordpress.com/2014/03/04/the-rank-lemma/
    n, m = A.shape
    # np.eye(5,1,k=0)
    col_sum = lambda j: sum((A @ np.array([1 if i == j else 0 for i in range(m)]))**2)
    fn2 = sum([col_sum(j) for j in range(m)])
    bound = min([A.shape[0], A.shape[1], np.ceil(sum(diagonal(A))**2 / fn2)])
  return int(bound)
  
def sgn_approx(x: ArrayLike = None, eps: float = 0.0, p: float = 1.0, method: int = 0, normalize: bool = False) -> ArrayLike:
  """Applies a function phi(x) to 'x' where phi smoothly approximates the positive sgn+ function.

  If x = (x1, x2, ..., xn), where each x >= 0, then this function returns a vector (y1, y2, ..., yn) satisfying

  1. 0 <= yi <= 1
  2. sgn+(xi) = sgn+(yi)
  3. yi -> sgn+(xi) as eps -> 0+
  4. phi(xi, eps) <= phi(xi, eps') for all eps >= eps'

  Parameters:
    sigma = array of non-negative values
    eps = smoothing parameter. Values close to 0 mimick the sgn function more closely. Larger values smooth out the relaxation. 
    p = float 
    method = the type of smoothing to apply. Should be integer in [0,3]. Defaults to 0. 
    normalize = whether to normalize the output such that eps in [0,1] interpolates the integrated area of phi() in the unit interval
  """
  assert method >= 0 and method <= 3, "Invalid method given"
  assert eps >= 0.0, "Epsilon must be non-negative"
  # assert isinstance(sigma, np.ndarray), "Only numpy arrays supported for now"
  if eps == 0 and (x is not None): return np.sign(x)
  _f = 0
  if normalize:
    s1 = np.vectorize(lambda t, e: t**p / (t**p + e**p))
    s2 = np.vectorize(lambda t, e: t**p / (t**p + e))
    s3 = np.vectorize(lambda t, e: t / (t**p + e**p)**(1/p))
    s4 = np.vectorize(lambda t, e: 1 - np.exp(-t/e))
    _phi = [s1,s2,s3,s4][method]
    from scipy.interpolate import interp1d
    alpha = np.copy(eps).item()
    assert alpha >= 0.0 and alpha <= 1.0,"Must be in [0,1]"
    dom = np.abs(np.linspace(0, 1, 100))
    EPS = np.linspace(1e-16, 10, 50)
    area = np.array([2*np.trapz(_phi(dom, eps), dom) for eps in EPS])
    area[0] = 2.0
    area[-1] = 0.0
    _f = lambda z: interp1d(2-area, EPS)(z*2)
  else:
    _f = lambda z: z
  
  ## Ridiculous syntax due to python: see https://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture
  method = int(method)
  if method == 0: 
    phi = lambda x, eps=eps, p=p, f=_f: np.sign(x) * (np.abs(x)**p / (np.abs(x)**p + f(eps)**p))
  elif method == 1: 
    phi = lambda x, eps=eps, p=p, f=_f: np.sign(x) * (np.abs(x)**p / (np.abs(x)**p + f(eps)))
  elif method == 2: 
    phi = lambda x, eps=eps, p=p, f=_f: np.sign(x) * (np.abs(x) / (np.abs(x)**p + f(eps)**p)**(1/p))
  else: 
    phi = lambda x, eps=eps, p=p, f=_f: np.sign(x) * (1.0 - np.exp(-np.abs(x)/f(eps)))
  return phi if x is None else phi(x)
  
def soft_threshold(x: ArrayLike = None, t: float = 1.0) -> ArrayLike:
  def _st(x: ArrayLike):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0)
  return _st if x is None else _st(x)

def moreau(x: ArrayLike = None, t: float = 1.0) -> ArrayLike:
  def _moreau(x: ArrayLike):
    x = np.array(x)
    #return sum(x) + sum(np.where(x >= t, 0.5, x)**2)
    return x + np.where(x >= t, 0.5, x)**2
  return _moreau if x is None else _moreau(np.array(x))

def huber(x: ArrayLike = None, delta: float = 1.0) -> ArrayLike:
  def _huber(x: ArrayLike): 
    return np.where(np.abs(x) <= delta, 0.5 * (x ** 2), delta * (np.abs(x) - 0.5*delta))
  return _huber if x is None else _huber(x)

## Great advice: https://gist.github.com/denis-bz/2658f671cee9396ac15cfe07dcc6657d 
def polymorphic_psd_solver(A: Union[ArrayLike, spmatrix, LinearOperator], pp: float = 1.0, solver: str = 'default', laplacian: bool = True,  return_eigenvectors: bool = False, **kwargs):
  """Configures an eigen-solver for a symmetric positive semi-definite linear operator _A_.

  _A_ can be a dense np.array, any of the sparse arrays from SciPy, or a _LinearOperator_. The default solver 'default' chooses 
  the solver dynamically based on the size, structure, and estimated rank of _A_. In particular, if 'A' is dense, then the divide-and-conquer 
  approach is chosen. 

  The solver can also be specified explicitly as one of:
    'dac' <=> Divide-and-conquer (via LAPACK routine 'syevd')
    'irl' <=> Implicitly Restarted Lanczos (via ARPACK)
    'lanczos' <=> Lanczos Iteration (non-restarted, w/ PRIMME)
    'gd' <=> Generalized Davidson w/ robust shifting (via PRIMME)
    'jd' <=> Jacobi Davidson w/ Quasi Minimum Residual (via PRIMME)
    'lobpcg' <=> Locally Optimal Block Preconditioned Conjugate Gradient (via PRIMME)

  Parameters: 
    A = array, sparse matrix, or linear operator to compute the rank of
    pp = proportional part of the spectrum to compute. 
    solver = One of 'default', 'dac', 'irl', 'lanczos', 'gd', 'jd', or 'lobpcg'.
    laplacian = whether to assume _A_ comes from a Laplacian. Defaults to True. 
    return_eigenvector = whether to return both eigen values and eigenvector pairs, or just eigenvalues. Defaults to False. 
  Returns:
    a callable f(A, ...) accepting operators of the same type as _A_ and returns spectral information

  Notes:
    For fast parameterization, _A_ should support efficient access to its diagonal via a method A.diagonal()
  """
  assert A.shape[0] == A.shape[1], "A must be square"
  assert pp >= 0.0 and pp <= 1.0, "Proportion 'pp' must be between [0,1]"
  assert solver is not None, "Invalid solver"
  tol = kwargs['tol'] if 'tol' in kwargs.keys() else np.finfo(A.dtype).eps
  if pp == 0.0: return 0.0
  if solver == 'dac':
    assert isinstance(A, np.ndarray), "Cannot use divide-and-conquer with linear operators"
  if isinstance(A, np.ndarray) and solver == 'default' or solver == 'dac':
    params = kwargs
    solver = np.linalg.eigh if return_eigenvectors else np.linalg.eigvalsh
  elif isinstance(A, spmatrix) or isinstance(A, LinearOperator):
    if isinstance(A, spmatrix) and np.allclose(A.data, 0.0): return(lambda A: np.zeros(1))
    if A.shape[0] == 0 or A.shape[1] == 0: return(lambda A: np.zeros(1))
    nev = trace_threshold(A, pp) if pp != 1.0 else rank_bound(A, upper=True)
    nev = min(nev, A.shape[0] - 1) if laplacian else nev
    if nev == 0: return(lambda A: np.zeros(1))
    if nev == A.shape[0] and (solver == 'irl' or solver == 'default'):
      import warnings
      warnings.warn("Switching to PRIMME, as ARPACK cannot estimate all eigenvalues without shift-invert")
      solver = 'jd'
    solver = 'irl' if solver == 'default' else solver
    assert isinstance(solver, str) and solver in ['default', 'irl', 'lanczos', 'gd', 'jd', 'lobpcg']
    if solver == 'irl':
      params = dict(k=nev, which='LM', tol=tol, return_eigenvectors=return_eigenvectors) | kwargs
      solver = eigsh
    else:
      import primme
      n = A.shape[0]
      ncv = min(2*nev + 1, 20) # use modification of scipy rule
      methods = { 'lanczos' : 'PRIMME_Arnoldi', 'gd': "PRIMME_GD" , 'jd' : "PRIMME_JDQR", 'lobpcg' : 'PRIMME_LOBPCG_OrthoBasis', 'default' : 'PRIMME_DEFAULT_MIN_TIME' }
      params = dict(ncv=ncv, maxiter=pp*n*100, tol=tol, k=nev, which='LM', return_eigenvectors=return_eigenvectors, method=methods[solver]) | kwargs
      solver = primme.eigsh
  else: 
    raise ValueError(f"Invalid solver / operator-type {solver}/{str(type(A))} given")
  def _call_solver(A, **kwargs):
    ew = solver(A, **(params | kwargs))
    if len(ew) == 0: 
      raise RuntimeError("Solver did not return any eigenvalues!")
    return ew
  return _call_solver

def eigh_solver(A: Union[ArrayLike, spmatrix, LinearOperator], **kwargs):
  return polymorphic_psd_solver(A, return_eigenvectors=True, **kwargs)

def eigvalsh_solver(A: Union[ArrayLike, spmatrix, LinearOperator], **kwargs) -> Callable:
  return polymorphic_psd_solver(A, return_eigenvectors=False, **kwargs)

def spectral_rank(ew: ArrayLike, method: int = 0, shape: tuple = None, prec: float = None) -> int:
  if len(ew) == 0: return 0
  ew = np.array(ew) if not isinstance(ew, np.ndarray) else ew
  prec = np.finfo(ew.dtype).eps if prec is None else prec
  m,n = shape if (shape is not None and len(shape) == 2) else (len(ew), len(ew))
  if method == 0:
    ## Default: identifies only those as positive cannot have been introduced by numerical errors 
    tol = ew.max() * max(m,n) * prec
    return sum(ew > tol)
  elif method == 1: 
    ## Minimizes expected roundoff error
    tol = ew.max() * prec / 2. * np.sqrt(m + n + 1.)
    return sum(ew > tol)
  elif method == 2:
    ## Dynamic method: tests multiple thresholds for detecting rank deficiency up to the prescribed precision, 
    ## choosing the one that appears the most frequently 
    C = 1.0/np.exp(np.linspace(1,36,100))
    i = len(C)-np.searchsorted(np.flip(C), prec)
    C = C[:i]
    nr = [sum(ew/(ew + c)) for c in C]
    values, counts = np.unique(np.round(nr).astype(int), return_counts=True)
    return values[np.argmax(counts)]
  else:
    raise ValueError("Invalid method chosen")

def stable_rank(A: Union[ArrayLike, spmatrix, LinearOperator], method: int = 0, prec: float = None, **kwargs):
  """Computes the rank of an operator _A_ using by counting non-zero eigenvalues.
  
  This method attempts to use various means with which to estimate the rank of _A_ in a stable fashion. 
  """
  solver = eigvalsh_solver(A)
  ew = solver(A)
  return spectral_rank(ew, method=method, prec=prec)

def smooth_rank(A: Union[ArrayLike, spmatrix, LinearOperator], smoothing: tuple = (0.5, 1.0, 0), symmetric: bool = True, sqrt: bool = False, raw: bool = False, solver: Optional[Callable] = None, **kwargs) -> float:
  """ 
  Computes a smoothed-version of the rank of a symmetric positive semi-definite 'A'

  The form of 'A' can be a dense or sparse array, or a [matrix-free] linear operator.

  Parameters: 
    A := array, sparse matrix, or linear operator to compute the rank of
    pp := proportional part of the spectrum to compute.
    smoothing := tuple (eps, p, method) of args to pass to 'sgn_approx'
    symmetric := whether the 'A' is symmetric. Should always be true; non-symmetric matrices are not yet supported. 
    sqrt := whether to compute sqrt the singular values
    solver := Optional callable or string. If the latter, one of 'default', 'dac', 'irl', 'lanczos', 'gd', 'jd', or 'lobpcg'. See 'parameterize_solver' for more details. 
    kwargs := keyword arguments to pass to the parameterize_solver.
  """
  if solver is None: 
    solver = eigvalsh_solver(A, pp=1.0)
  elif isinstance(solver, str):
    solver = eigvalsh_solver(A, pp=1.0, solver=solver)
  else:
    assert isinstance(solver, Callable), "Solver must callable, if not string or None"
  ew = solver(A)
  ew = np.maximum(ew, 0.0)
  assert all(np.isreal(ew)) and all(ew >= 0.0), "Negative or non-real eigenvalues detected. This method only works with symmetric PSD matrices."
  ## Note that eigenvalues of PSD 'A' *are* its singular values via Schur theorem. 
  if smoothing is not None:
    assert isinstance(smoothing, tuple) and len(smoothing) == 3, "Smoothing must a tuple of the parameters to pass"
    eps, p, method = smoothing 
    ew = sgn_approx(np.sqrt(ew), eps, p, method) if sqrt else sgn_approx(ew, eps, p, method)
  else: 
    ew = np.sqrt(ew) if sqrt else ew
  return ew if raw else sum(ew)

def prox_nuclear(x: ArrayLike, t: float = 1.0):
  """Prox operator on the nuclear norm.
  
  prox_f returns the (unique) point that achieves the infimum defining the _Moreau envelope_ _M_f_.

  prox_{t*f}(x) = argmin_z { f(z) + (1/2t) || x - z ||^2 }
                = argmin_z { ||z||_* + (1/2t) || x - z ||_F^2 }
                = \sum\limits_{i=1}^r max(si - t, 0) + ui vi^T

  where X = U S V^T

  Smaller values of _t_ will yield operators closer to the original nuclear norm.
  """
  solver = eigh_solver(x)
  ew, ev = solver(x)
  sv = np.sqrt(np.maximum(ew, 0.0))                     ## singular values 
  sw_prox = np.maximum(sv - t, 0.0)                     ## soft-thresholded singular values
  A = ev @ diags(sw_prox) @ ev.T                        ## prox operator 
  proj_d = (1/2*t)*np.linalg.norm(A - x, 'fro')**2      ## projection distance (rhs)
  me = sum(sw_prox) + proj_d                            ## Moreau envelope value
  return A, me, ew

def numerical_rank(A: Union[ArrayLike, spmatrix, LinearOperator], tol: float = None, solver: str = 'default', **kwargs) -> int:
  """ 
  Computes the numerical rank of a positive semi-definite 'A' via thresholding its eigenvalues.

  The eps-numerical rank r_eps(A) is the largest integer 'r' such that: 
  
  s1 >= s2 >= ... >= sr >= eps >= ... >= 0

  where si represent the i'th largest singular value of A. 

  See: https://math.stackexchange.com/questions/2238798/numerically-determining-rank-of-a-matrix
  """
  if isinstance(A, np.ndarray):
    if np.allclose(A, 0.0) or prod(A.shape): return 0
    return np.linalg.matrix_rank(A, tol, hermitian=True)
  elif isinstance(A, spmatrix):
    # import importlib.util
    # if importlib.util.find_spec('sksparse') is not None: 
    #   from sksparse.cholmod import cholesky
    #   factor = cholesky(L, beta=tol)
    #   PD = type('cholmod_solver', (type(LO),), { '_matvec' : lambda self, b: factor(b) })
    #   solver = PD(L)
    ## Use tolerance + Neumans trace inequality / majorization result to lower bound on rank / upper bound nullity
    ## this doesn't work for some reason
    # rank_lb = sum(np.sort(diagonal(A)) >= tol)
    if np.allclose(A.data, 0.0) or prod(A.shape) == 0: return 0

    ## https://konradswanepoel.wordpress.com/2014/03/04/the-rank-lemma/
    rank_lb = rank_bound(A, upper=False)
    k = A.shape[0]-rank_lb

    ## Get numerical epsilon tolerance 
    if tol is None: 
      sn = eigsh(A, k=1, which='LM', return_eigenvectors=False).item() # spectral norm
      tol = sn * max(A.shape) * np.finfo(float).eps
      if np.isclose(sn, 0.0): return 0
    assert isinstance(tol, float), "Invalid tolerance"
    
    ## Use un-shifted operator 'A' + shifted solver to estimate the dimension of the nullspace w/ shift-invert mode
    ## TODO: Preconditioning advice: https://caam37830.github.io/book/02_linear_algebra/sparse_linalg.html
    smallest_k = eigsh(A + tol*eye(A.shape[0]), k=k, which='LM', sigma=0.0, tol=tol, return_eigenvectors=False)
    dim_nullspace = sum(np.isclose(np.maximum(smallest_k - tol, 0.0), 0.0))
    
    ## Use rank-nullity to obtain the numerical rank
    return max(A.shape) - dim_nullspace
  elif isinstance(A, LinearOperator):
    if prod(A.shape) == 0: return 0 

    ## Get numerical epsilon tolerance 
    if tol is None: 
      sn = eigsh(A, k=1, which='LM', return_eigenvectors=False).item() # spectral norm
      tol = sn * max(A.shape) * np.finfo(float).eps
      if np.isclose(sn, 0.0): return 0
    assert isinstance(tol, float), "Invalid tolerance"
    cI = aslinearoperator(tol * eye(A.shape[0]))

    ## Resolve the Ax = b solver
    if solver == 'default' or solver == 'cg':
      solver = lambda b: cg(A+cI, b, tol=tol, atol=tol)[0]
    elif solver == 'lgmres':
      solver = lambda b: lgmres(A+cI, b, tol=tol, atol=tol)[0]
    else: 
      raise ValueError(f"Unknown Ax=b solver '{solver}'")

    ## Construct an object with .shape and .matvec attributes
    Ax_solver = aslinearoperator(type('Ax_solver', (object,), { 
      'matvec' : solver, 
      'shape' : A.shape
    }))

    ## Get lower bound on rank
    k = A.shape[0]-rank_bound(A, upper=False)

    ## Use un-shifted operator 'A' + shifted solver to estimate the dimension of the nullspace w/ shift-invert mode
    smallest_k = eigsh(A, k=k, which='LM', sigma=0.0, OPinv = Ax_solver, tol=tol, return_eigenvectors=False)
    dim_nullspace = sum(np.isclose(np.maximum(smallest_k - tol, 0.0), 0.0))
    
    ## Use rank-nullity to obtain the numerical rank
    return max(A.shape) - dim_nullspace
  else: 
    raise ValueError("Invalid input type supplied")

## is_positive_definite
# First check if symmetric 
# then see if strictly diagonally dominant 
# then compute eigenvalues
# For linear operators: row sums == L @ np.repeat(1, L.shape[0]) 


def sparse_pinv(A, method: str = "pinv"):
  if method == "psvd_sqrt":
    u,s,vt = np.linalg.svd(A.todense(), hermitian=True)
    tol = s.max() * max(A.shape) * np.finfo(float).eps
    r = sum(s > tol)
    LS = vt[:(r-1),:].T @ np.diag(1/np.sqrt(s[:(r-1)])) @ u[:,:(r-1)].T
  elif method == "pinv":
    from scipy.linalg import pinvh
    LS, r = pinvh(A.todense(), return_rank=True)
  elif method == "psvd":
    u,s,vt = np.linalg.svd(A.todense(), hermitian=True)
    tol = s.max() * max(A.shape) * np.finfo(float).eps
    r = np.sum(abs(s) > tol)
    LS = vt[:(r-1),:].T @ np.diag(1/s[:(r-1)]) @ u[:,:(r-1)].T
  else: 
    raise ValueError("Unknown method")
  return LS 

## Given a graph Laplacian, calculate the effective resistance of every edge
def effective_resistance(L: ArrayLike, method: str = "pinv"):
  eff_resist = lambda x: x
  if method == "psvd_sqrt":
    u,s,vt = np.linalg.svd(L.todense(), hermitian=True)
    tol = s.max() * max(L.shape) * np.finfo(float).eps
    r = sum(s > tol)
    LS = vt[:(r-1),:].T @ np.diag(1/np.sqrt(s[:(r-1)])) @ u[:,:(r-1)].T
    eff_resist = lambda x: np.linalg.norm(LS @ x)**2
  elif method == "pinv":
    from scipy.linalg import pinvh
    LP, r = pinvh(L.todense(), return_rank=True)
    eff_resist = lambda x: (x.T @ LP @ x).item()
  elif method == "psvd":
    u,s,vt = np.linalg.svd(L.todense(), hermitian=True)
    tol = s.max() * max(L.shape) * np.finfo(float).eps
    r = np.sum(abs(s) > tol)
    LS = vt[:(r-1),:].T @ np.diag(1/s[:(r-1)]) @ u[:,:(r-1)].T
    eff_resist = lambda x: (x.T @ LS @ x).item()
  elif method == "randomized":
    raise NotImplementedError("Haven't done yet")
  else: 
    raise ValueError("Unknown method")
  nv = L.shape[0]
  cv, er = np.zeros(shape=(nv,1)), []
  for (i,j) in edge_iterator(L):
    cv[i] = 1
    cv[j] = -1
    er.append(eff_resist(cv))
    cv[i] = cv[j] = 0
  return np.array(er)

def eigsh_block_shifted(A, k: int, b: int = 10, **kwargs):
  """
  Calculates all eigenvalues in a block-wise manner 
  """
  ni = int(np.ceil(k/b))
  f_args = dict(tol=np.finfo(np.float32).eps, which='CGT', return_eigenvectors=True, raise_for_unconverged=False, return_history=False)
  f_args = f_args | kwargs
  evals = np.zeros(ni*b)
  for i in range(ni):
    if i == 0:
      ew, ev = primme.eigsh(A, k=b, **f_args)
    else: 
      #f_args['which'] = min(ew)
      f_args['sigma'] = min(ew)
      ew, ev = primme.eigsh(A, k=b, lock=ev, **f_args)
    # print(ew)
    evals[(i*b):((i+1)*b)] = ew
  return(evals[-k:])


# from pbsig.utility import timeout
# from pbsig.precon import Minres, Jacobi, ssor, ShiftedJacobi

# def as_positive_definite(A: Union[ArrayLike, LinearOperator], c: float = 0.0):

def diagonal(A: Union[ArrayLike, LinearOperator]):
  """ Generic to get the diagonal of a sparse matrix, an array, or a linear operator """
  if hasattr(A, 'diagonal'):
    return A.diagonal()
  else: 
    assert hasattr(A, 'shape')
    I = np.zeros(A.shape[0])
    d = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
      I[i] = 1
      I[i-1] = 0
      d[i] = (A @ I)[i]
    return d 

## Returns number of largest eigenvalues needed to capture t% of the spectrum 
def trace_threshold(A, pp=0.80) -> int:
  assert pp >= 0.0 and pp <= 1.0, "Proportion must be in [0,1]"
  d = diagonal(A)
  ev_cs = np.cumsum(d)
  if ev_cs[-1] == 0.0: return 0
  thresh_ind = np.flatnonzero(ev_cs/ev_cs[-1] >= pp)[0] # index needed to obtain 80% of the spectrum
  num_pos = len(ev_cs) - sum(np.isclose(ev_cs, 0.0))
  return min(num_pos, thresh_ind+1)# 0-based conversion

def eigsh_family(M: Iterable[ArrayLike], p: float = 0.25, reduce: Callable[ArrayLike, float] = None, **kwargs):
  """
  Given an iterable of n symmetric real-valued matrices M1, M2, ..., Mn, this function computes an 'R' such that: 

  R[i] = reduce(eigenvalues(Mi, p))

  where 'eigenvalues(*,p)' computes the k largest eigenvalues that make up p% of the spectra of each Mi.

  Parameters: 
    - M := an iterable of symmetric real-valued matrices, sparse matrices, arrays, or LinearOperator's
    - p := the percentage of the spectra to compute, for each Mi in M 
    - generator := whether to return a generator (default) or store the results as-is in a container 
    - reduce := optional scalar-valued aggregation function, to call on the eigenvalues each iteration. Must return a float, if supplied. 
    - kwargs := global/'default' keyword arguments to supply to primme.eigsh 
  
  """
  default_opts = dict(tol=1e-6, printLevel=0, return_eigenvectors=False, return_stats=False, return_history=False) | kwargs
  args = {} 
  reduce_f, is_aggregate = (lambda x: x, False) if reduce is None else (reduce, True)
  for A in M: 
    A, args = (A[0], args | A[1]) if isinstance(A, tuple) and len(A) == 2 else (A, args)
    nev = trace_threshold(A, p)
    if default_opts['return_stats']:
      if default_opts['return_eigenvectors']:
        ew, ev, stats = primme.eigsh(A=A, k=nev, **default_opts)
        yield reduce_f(ew), ev, stats
      else:
        ew, stats = primme.eigsh(A=A, k=nev, **default_opts)
        yield reduce_f(ew), stats
    else: 
      if default_opts['return_eigenvectors']:
        ew, ev, stats = primme.eigsh(A=A, k=nev, **default_opts)
        yield reduce_f(ew), ev
      else:
        ew = primme.eigsh(A=A, k=nev, **default_opts)
        yield reduce_f(ew)
  # else: 
  #   #n = len(M)
  #   return list(yield from eigsh_family(M, p, False, reduce, **kwargs))
  #   # result = np.zeros(shape=(n,), dtype=float) if aggregate else [None]*n
  #   # for i, A in enumerate(M): 
  #   #   nev = trace_threshold(A, p)
  #   #   ew = primme.eigsh(A=A, k=nev, **default_opts)
  #   #   result[i] = reduce(ew)
  #   return result
    
def as_linear_operator(A, stats=True):
  from scipy.sparse.linalg import aslinearoperator, LinearOperator
  class LO(LinearOperator):
    def __init__(self, A):
      self.n_calls = 0
      self.A = A
      self.dtype = A.dtype
      self.shape = A.shape

    def _matvec(self, x):
      self.n_calls += 1
      return self.A @ x

    def _matmat(self, X):
      self.n_calls += X.shape[1]
      return self.A @ X
  lo = LO(A)
  return lo


## TODO: make generalized up- and down- adjacency matrices for simplicial complexes

class UpLaplacianPy(LinearOperator):
  """ 
  Linear operator for weighted p up-Laplacians of simplicial complexes. 
  
  In matrix notation, these take the form:

  W_p^l @ D_{p+1} @ W_{p+1} D_{p+1}.T @ W_p^r 

  where W* represents diagonal weight matrices on the p-th (or p+1) simplices and Dp represents 
  the p-th oriented boundary matrix of the simplicial complex. 

  The operator is always matrix-free in the sense that no matrix is actually stored in this class. 

  """
  identity_seq = type("One", (), { '__getitem__' : lambda self, x: 1.0 })()

  def __init__(self, S: Iterable['SimplexLike'], F: Sequence['SimplexLike'], dtype = None, orientation: Union[int, tuple] = 0):
    ## Ensure S is repeatable and faces 'F' is indexable
    assert not(S is iter(S)) and not(F is iter(F)), "Simplex iterables must be repeatable (a generator is not sufficient!)"
    assert isinstance(F, Sequence), "Faces must be a valid Sequence (supporting .index(*) with SimplexLike objects!)"
    p: int = len(F[0]) # 0 => F are vertices, build graph Laplacian

    ## Resolve the sign pattern
    if isinstance(orientation, Integral):
      sgn_pattern = (1,-1) if orientation == 0 else (-1,1)
      orientation = tuple(islice(cycle(sgn_pattern), p+2))
    assert isinstance(orientation, tuple)
    self.sgn_pattern = np.array([s1*s2 for (s1,s2) in combinations(orientation,2)])
    self._v = np.zeros(len(F)) # workspace
    self.simplices = S
    self.faces = F
    self.shape = (len(F), len(F))
    self.dtype = np.dtype(float) if dtype is None else dtype
    self._wfl = self._wfr = self._ws = UpLaplacianPy.identity_seq
    self.prepare()
    self._precompute_degree()
    self.precompute()
    self.P = None

  def diagonal(self) -> ArrayLike:
    return self.degree

  @property 
  def face_left_weights(self): 
    return self._wfl

  @face_left_weights.setter
  def face_left_weights(self, value: ArrayLike) -> None:
    if value is None: 
      self._wfl = UpLaplacian.identity_seq
    else: 
      assert len(value) == self.shape[0], "Invalid value given. Must match shape."
      assert isinstance(value, np.ndarray)
      self._wfl = value
    self._precompute_degree()

  @property 
  def face_right_weights(self): 
    return self._wfr

  @face_right_weights.setter
  def face_right_weights(self, value: ArrayLike) -> None:
    if value is None: 
      self._wfr = UpLaplacian.identity_seq
    else: 
      assert len(value) == self.shape[0], "Invalid value given. Must match shape."
      assert isinstance(value, np.ndarray)
      self._wfr = value
    self._precompute_degree()

  ## S-weights
  @property
  def simplex_weights(self): 
    return self._ws
  
  @simplex_weights.setter
  def simplex_weights(self, value: ArrayLike):
    if value is None: 
      self._ws = UpLaplacian.identity_seq
    else:
      ## Soft-check to help debugging
      if hasattr(self.simplices, '__len__') and hasattr(value, '__len__'): 
        assert len(self.simplices) == len(value), "Invalid weight vector given."
      #assert isinstance(value, np.ndarray) 
      self._ws = value ## length is potentially unchecked! 
    self._precompute_degree()

  def set_weights(self, lw = None, cw = None, rw = None):
    self.face_left_weights = lw
    self.simplex_weights = cw
    self.face_right_weights = rw
    self._precompute_degree()
    return self 

  def _precompute_degree(self):
    self.degree = np.zeros(self.shape[1])
    for s_ind, s in enumerate(self.simplices):
      for f in s.boundary():
        ii = self.index(f)
        self.degree[ii] += self._wfl[ii] * self._ws[s_ind] * self._wfr[ii]

  ## Prepares the indexing function, precomputes the degree, etc.
  def prepare(self, level: int = 0) -> None: 
    # self.index = type('_indexer', (), { '__getitem__' : lambda })
    if level == 0:
      ## Faces must be a Sequence anyways
      self.index = lambda s: self.faces.index(s)
    elif level == 1:
      self._index = { f: i for i, f in enumerate(self.faces) }
      self.index = lambda s: self._index[s]
    else: 
      raise NotImplementedError("Not implemented yet")

  ## Computes all of the expressions that do not directly involve x, including input and output indices
  ## This essentially moves the looping structures in _matvec over to numpy, at the expense of a lot of memory  
  def precompute(self):
    q = len(self.simplices[0])-1
    N = 2*len(self.simplices)*comb(q+1, 2)
    self.P = np.zeros(shape=N, dtype=[('weights', 'f4'), ('xi', 'i2'), ('vo', 'i2')])
    cc = 0 
    for s_ind, s in enumerate(self.simplices):
      for (f1, f2), sgn_ij in zip(combinations(s.boundary(), 2), self.sgn_pattern):
        ii, jj = self.index(f1), self.index(f2)
        d1 = sgn_ij * self._wfl[ii] * self._ws[s_ind] * self._wfr[jj]
        d2 = sgn_ij * self._wfl[jj] * self._ws[s_ind] * self._wfr[ii]
        self.P[cc] = d1, jj, ii
        self.P[cc+1] = d2, ii, jj
        cc += 2
    return self
    #   np.add.at(v, P['vo'], x[P['xi']]*P['weights'])

  def _matvec_precompute(self, x: ArrayLike) -> ArrayLike:
    x = x.reshape(-1)
    self._v.fill(0)
    self._v += self.degree * x
    np.add.at(self._v, self.P['vo'], x[self.P['xi']]*self.P['weights'])
    return self._v
  
  ## Define the matvec operator using the precomputed degree, the pre-allocate workspace, and the pre-determined sgn pattern
  def _matvec_full(self, x: ArrayLike) -> ArrayLike:
    x = x.reshape(-1)
    self._v.fill(0)
    self._v += self.degree * x
    for s_ind, s in enumerate(self.simplices):
      for (f1, f2), sgn_ij in zip(combinations(s.boundary(), 2), self.sgn_pattern):
        ii, jj = self.index(f1), self.index(f2)
        self._v[ii] += sgn_ij * x[jj] * self._wfl[ii] * self._ws[s_ind] * self._wfr[jj]
        self._v[jj] += sgn_ij * x[ii] * self._wfl[jj] * self._ws[s_ind] * self._wfr[ii]
    return self._v
    
  def _matvec(self, x: ArrayLike) -> ArrayLike:
    assert x.ndim == 1 or (x.ndim == 2 and 1 in x.shape)
    return self._matvec_precompute(x) if self.P is not None else self._matvec_full(x)

  def _matmat(self, X: ArrayLike) -> ArrayLike:
    n,k = X.shape
    m = self.shape[0]
    Y = np.empty(shape=(m,k))
    for j in range(X.shape[1]):
      Y[:,j] = self._matvec(X[:,j])
    return Y

## From: https://github.com/cvxpy/cvxpy/blob/master/cvxpy/interface/matrix_utilities.py
def is_symmetric(A) -> bool:
  """ Check if a real-valued matrix is symmetric up to a given tolerance (via np.isclose) """
  from scipy.sparse import issparse
  if isinstance(A, np.ndarray):
    return np.allclose(A, A.T)
  assert issparse(A)
  if A.shape[0] != A.shape[1]:
    raise ValueError('m must be a square matrix')
  A = coo_array(A) if not isinstance(A, coo_array) else A

  r, c, v = A.row, A.col, A.data
  tril_no_diag = r > c
  triu_no_diag = c > r

  if not np.isclose(triu_no_diag.sum(), tril_no_diag.sum()):
    return False

  rl, cl, vl = r[tril_no_diag], c[tril_no_diag], v[tril_no_diag]
  ru, cu, vu = r[triu_no_diag], c[triu_no_diag], v[triu_no_diag]

  sortl = np.lexsort((cl, rl))
  sortu = np.lexsort((ru, cu))
  vl = vl[sortl]
  vu = vu[sortu]
  return np.allclose(vl, vu)

from scipy.sparse import coo_array
def adjacency_matrix(S: ComplexLike, p: int = 0, weights: ArrayLike = None):
  assert len(S) > p, "Empty simplicial complex"
  if dim(S) <= (p+1):
    return coo_array((card(S,p), card(S,p)), dtype=int)
  weights = np.ones(card(S,p+1)) if weights is None else weights
  assert len(weights) == card(S,p+1), "Invalid weight array, must match length of p+1 simplices"
  V = list(faces(S, p))
  ind = []
  for s in faces(S, p+1):
    VI = [V.index(f) for f in faces(s, p)]
    for i,j in combinations(VI, 2):
      ind.append((i,j))
  IJ = np.array(ind)
  A = coo_array((weights, (IJ[:,0], IJ[:,1])), shape=(card(S,p), card(S,p)))
  A = A + A.T
  return A

## TODO: change weight to optionally be a string when attr system added to SC's
def up_laplacian(S: ComplexLike, p: int = 0, weight: Optional[Callable] = None, normed=False, return_diag=False, form='array', symmetric: bool = True, dtype=None, **kwargs):
    """Returns the weighted combinatorial p-th up-laplacian of an abstract simplicial complex S. 

    Given D = boundary_matrix(S, p+1), this function parameterizes any (generic) weighted p-th up-laplacian L of the form: 

    L := Wl(fp) @ D @ W(fq) @ D^T @ Wr(fp)
    
    Where  W(fp)^{l,r}, W(fq) are diagonal matrices weighting the p and q=p+1 simplices, respectively. If the weights supplied via 
    a weight callable w: S -> R are positive, then the Laplacian operator returned corresponds to equipping a scalar product on the 
    coboundary vector spaces of S. By default, if symmetric = True, then the exact form of the combinatorial laplacian returned is: 

    L_sym := Wl(fp)^{+/2} @ D @ W(fq) @ D^T @ Wr(fp)^{+/2}

    where A^{+/2} denotes the pseudo-inverse of A^(1/2). This operator is compact and symmetric, has real eigenvalues, and is 
    spectrally-similar to the asymmetric form (where symmetric=False) given by 
    
    L_asym := Wl(fp)^{+} @ D @ W(fq) @ D^T 

    If normed = True, then L is referred to as the _normalized_ combinatorial Laplacian, which has the form: 

    L_norm := Wl(deg(fp))^{+/2} @ D @ W(fq) @ D^T Wr(deg(fp))^{+/2} 

    where deg(...) denotes the weighted degrees of the p-simplices. Note that L_norm has a bounded spectrum in the interval [0, p+2].

    Summary of the specializations:
      - (p=0)                               <=> graph laplacian  
      - (p=0, weight=...)                   <=> weighted graph laplacian
      - (p=0, weight=..., normed=True)      <=> normalized weighted graph Laplacian 
      - (p>0, weight=...)                   <=> weighted combinatorial up-laplacian
      - (p>0, weight=..., normed=True)      <=> normalized weighted combinatorial up-laplacian 
      - (p>0, weight=..., symmetric=False)  <=> asymmetric weighted combinatorial up-laplacian 
      ...

    Parameters:
      S = _ComplexLike_ or _FiltrationLike_ instance
      p = dimension of the up Laplacian. Defaults to 0 (graph Laplacian).
      weight = Callable weight function to be evaluated on each simplex in S. Must return either a value or a 2-tuple (wl, wr). 
      normed = Whether to degree-normalize the corresponding up-laplacian. See details. 
      return_diag = whether to return diagonal degree matrix along with the laplacian
      form = return type. One of ['array', 'lo', 'function']
      symmetric = boolean whether to use the spectrally-similar symmetric form. Defaults to True.
      dtype := dtype of associated laplacian. 
      kwargs := unused. 
    
    The argument names for this function is loosely based on SciPy 'laplacian' interface in the sparse.csgraph module. 
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.laplacian.html. However, unlike the 
    LinearOperator returned by csgraph.laplacian, the operator returned here is always matrix-free.  
    (see https://github.com/scipy/scipy/blob/main/scipy/sparse/csgraph/_laplacian.py)
    """
    # assert isinstance(K, ComplexLike), "K must be a Simplicial Complex for now"
    assert isinstance(weight, Callable) if weight is not None else True
    pseudo = lambda x: np.reciprocal(x, where=~np.isclose(x, 0)) # scalar pseudo-inverse
    weight = (lambda s: 1.0) if weight is None else weight
    _ = weight(next(faces(S, p)))
    if not isinstance(_, Number) and len(_) == 2:
      W_LR = np.array([weight(s) for s in faces(S, p)]).astype(float)
      wpl = W_LR[:,0]
      wpr = W_LR[:,1]
    else: 
      wpl = np.array([float(weight(s)) for s in faces(S, p)])
      wpr = wpl
    wq = np.array([float(weight(s)) for s in faces(S, p+1)])
    assert len(wpl) == card(S,p) and len(wq) == card(S,p+1), "Invalid weight arrays given."
    assert all(wq >= 0.0) and all(wpl >= 0.0), "Weight function must be non-negative"
    if form == 'array':
      B = boundary_matrix(S, p = p+1)
      if normed: 
        L = B @ diags(wq) @ B.T
        deg = L.diagonal()
        L = (diags(pseudo(np.sqrt(deg))) @ L @ diags(pseudo(np.sqrt(deg)))).tocoo()
      else:
        L = (diags(pseudo(np.sqrt(wpl))) @ B @ diags(wq) @ B.T @ diags(pseudo(np.sqrt(wpr)))).tocoo()
      return (L, L.diagonal()) if return_diag else L
    elif form == 'lo':
      p_faces = list(faces(S, p))        ## need to make a view 
      p_simplices = list(faces(S, p+1))  ## need to make a view 
      _lap_cls = eval(f"UpLaplacian{int(p)}D")
      lo = _lap_cls(p_simplices, p_faces, card(S, 0))
      if normed:
        if not(symmetric):
          import warnings
          warnings.warn("symmetric = False is not a valid option when normed = True")
        lo.set_weights(None, wq, None)
        deg = lo.diagonal()
        lo.set_weights(pseudo(np.sqrt(deg)), wq, pseudo(np.sqrt(deg))) # normalized weighted symmetric psd version 
        # deg = np.zeros(len(p_faces))
        # for cc, qs in enumerate(p_simplices):
        #   face_ind = np.array([lo.index(ps) for ps in boundary(qs)])
        #   deg[face_ind] += wq[cc]
        # lo.set_weights(pseudo(np.sqrt(deg)), wq, pseudo(np.sqrt(deg))) 
      else: 
        if symmetric:
          lo.set_weights(pseudo(np.sqrt(wpl)), wq, pseudo(np.sqrt(wpr)))  # weighted symmetric psd version 
        else:
          lo.set_weights(pseudo(wpl), wq, None)                           # asymmetric version
      return lo
      # f = _up_laplacian_matvec_p(p_simplices, p_faces, w0, w1, p, "default")
      # lo = f if form == 'function' else LinearOperator(shape=(ns[p],ns[p]), matvec=f, dtype=np.dtype(float))
    elif form == 'function':
      raise NotImplementedError("function form not implemented yet")
    else: 
      raise ValueError(f"Unknown form '{form}'.")


# for cls_nm in dir(laplacian):
#   if cls_nm[0] != "_":
#     #eval(f"laplacian.{cls_nm}")
#     type(cls_nm, (eval(f"laplacian.{cls_nm}"), LinearOperator), {})


class UpLaplacianBase(LinearOperator):
  """ 
  Linear operator for weighted p up-Laplacians of simplicial complexes. 
  
  In matrix notation, these take the form:

  W_p^l @ D_{p+1} @ W_{p+1} D_{p+1}.T @ W_p^r 

  where W* represents diagonal weight matrices on the p-th (or p+1) simplices and Dp represents 
  the p-th oriented boundary matrix of the simplicial complex. 

  This operator is always matrix-free in the sense that no matrix is actually stored in the instance. 
  """
  # __slots__ = ('shape', 'dtype')
  # identity_seq = type("One", (), { '__getitem__' : lambda self, x: 1.0 })()
  # if p == 0:
  #   _lap_cls = laplacian.UpLaplacian0D
  # elif p == 1: 
  #   _lap_cls = laplacian.UpLaplacian1D
  # elif p == 2: 
  #   _lap_cls = laplacian.UpLaplacian2D
  # else: 
  #   raise ValueError("Laplacian extension modules has not been compiled for p > 2.")
  # self.m = _lap_cls(q_ranks, nv, _np)

  def __init__(S: Iterable['SimplexLike'], F: Sequence['SimplexLike'], dtype = None):
    ## Ensure S is repeatable and faces 'F' is indexable
    assert not(S is iter(S)) and not(F is iter(F)), "Simplex iterables must be repeatable (a generator is not sufficient!)"
    assert isinstance(F, Sequence), "Faces must be a valid Sequence (supporting .index(*) with SimplexLike objects!)"
    p: int = len(next(iter(F)))-1 # 0 => F are vertices, build graph Laplacian
    assert p == 0 or p == 1 or p == 2, "Only p in {0,1} supported for now"
    assert (len(next(iter(S))) == p+2), "Invalid length of simplices/faces"
    # super().__init__(*args, **kwargs)

  def diagonal(self) -> ArrayLike:
    return self.degrees

  @property 
  def face_left_weights(self): 
    return self.fpl

  @face_left_weights.setter
  def face_left_weights(self, value: ArrayLike) -> None:
    assert len(value) == self.shape[0] and value.ndim == 1, "Invalid value given. Must match shape."
    assert isinstance(value, np.ndarray)
    self.fpl = value.astype(self.dtype)

  @property 
  def face_right_weights(self): 
    return self.fpr

  @face_right_weights.setter
  def face_right_weights(self, value: ArrayLike) -> None:
    assert len(value) == self.shape[0] and value.ndim == 1, "Invalid value given. Must match shape."
    assert isinstance(value, np.ndarray)
    self.fpr = value.astype(self.dtype)

  @property
  def simplex_weights(self): 
    return self.fq
  
  @simplex_weights.setter
  def simplex_weights(self, value: ArrayLike):
    assert len(value) == len(self.fq) and value.ndim == 1, "Invalid value given. Must match shape."
    assert isinstance(value, np.ndarray)
    self.fq = value.astype(self.dtype)

  def set_weights(self, lw = None, cw = None, rw = None):
    #print(len(cw), self.nq, len(self.fq))
    self.face_left_weights = lw if lw is not None else np.repeat(1.0, self.np)
    self.simplex_weights = cw if cw is not None else np.repeat(1.0, self.nq)
    self.face_right_weights = rw if rw is not None else np.repeat(1.0, self.np)
    self.precompute_degree()
    return self 

class UpLaplacian0D(laplacian.UpLaplacian0D, UpLaplacianBase):
  def __init__(self, S: Iterable['SimplexLike'], F: Sequence['SimplexLike'], nv: int):
    _np = len(F)
    q_ranks = rank_combs(S, n=nv, order="lex")
    UpLaplacianBase.__init__(S, F)
    laplacian.UpLaplacian0D.__init__(self, q_ranks, nv, _np)
    self.compute_indexes()
    self.precompute_degree()

class UpLaplacian0F(laplacian.UpLaplacian0F, UpLaplacianBase):
  def __init__(self, S: Iterable['SimplexLike'], F: Sequence['SimplexLike'], nv: int):
    _np = len(F)
    q_ranks = rank_combs(S, n=nv, order="lex")
    UpLaplacianBase.__init__(S, F)
    laplacian.UpLaplacian0F.__init__(self, q_ranks, nv, _np)
    self.compute_indexes()
    self.precompute_degree()

class UpLaplacian1D(laplacian.UpLaplacian1D, UpLaplacianBase):
  def __init__(self, S: Iterable['SimplexLike'], F: Sequence['SimplexLike'], nv: int):
    _np = len(F)
    q_ranks = rank_combs(S, n=nv, order="lex")
    UpLaplacianBase.__init__(S, F)
    laplacian.UpLaplacian1D.__init__(self, q_ranks, nv, _np)
    self.compute_indexes()
    self.precompute_degree()

class UpLaplacian1F(laplacian.UpLaplacian1F, UpLaplacianBase):
  def __init__(self, S: Iterable['SimplexLike'], F: Sequence['SimplexLike'], nv: int):
    _np = len(F)
    q_ranks = rank_combs(S, n=nv, order="lex")
    UpLaplacianBase.__init__(S, F)
    laplacian.UpLaplacian1F.__init__(self, q_ranks, nv, _np)
    self.compute_indexes()
    self.precompute_degree()
    
class UpLaplacian2D(laplacian.UpLaplacian2D, UpLaplacianBase):
  def __init__(self, S: Iterable['SimplexLike'], F: Sequence['SimplexLike'], nv: int):
    _np = len(F)
    q_ranks = rank_combs(S, n=nv, order="lex")
    UpLaplacianBase.__init__(S, F)
    laplacian.UpLaplacian1D.__init__(self, q_ranks, nv, _np)
    self.compute_indexes()
    self.precompute_degree()

class UpLaplacian2F(laplacian.UpLaplacian2F, UpLaplacianBase):
  def __init__(self, S: Iterable['SimplexLike'], F: Sequence['SimplexLike'], nv: int):
    _np = len(F)
    q_ranks = rank_combs(S, n=nv, order="lex")
    UpLaplacianBase.__init__(S, F)
    laplacian.UpLaplacian1F.__init__(self, q_ranks, nv, _np)
    self.compute_indexes()
    self.precompute_degree()

## From: https://github.com/cvxpy/cvxpy/blob/master/cvxpy/interface/matrix_utilities.py
def is_symmetric(A) -> bool:
  """ Check if a real-valued matrix is symmetric up to a given tolerance (via np.isclose) """
  from scipy.sparse import issparse
  if isinstance(A, np.ndarray):
    return np.allclose(A, A.T)
  assert issparse(A)
  if A.shape[0] != A.shape[1]:
    raise ValueError('m must be a square matrix')
  A = coo_array(A) if not isinstance(A, coo_array) else A

  r, c, v = A.row, A.col, A.data
  tril_no_diag = r > c
  triu_no_diag = c > r

  if not np.isclose(triu_no_diag.sum(), tril_no_diag.sum()):
    return False

  rl, cl, vl = r[tril_no_diag], c[tril_no_diag], v[tril_no_diag]
  ru, cu, vu = r[triu_no_diag], c[triu_no_diag], v[triu_no_diag]

  sortl = np.lexsort((cl, rl))
  sortu = np.lexsort((ru, cu))
  vl = vl[sortl]
  vu = vu[sortu]
  return np.allclose(vl, vu)

# from enum import Enum
def projector_intersection(A, B, space: str = "RR", method: str = "neumann", eps: float = 100*np.finfo(float).resolution):
  """ 
  Creates a projector that projects onto the intersection of spaces (A,B), where
  A and B are linear subspace.
  
  method := one of 'neumann', 'anderson', or 'israel' 
  """
  assert space in ["RR", "RN", "NR", "NN"], "Invalid space argument"
  # if A.shape
  U, E, Vt = np.linalg.svd(A.A, full_matrices=False)
  X, S, Yt = np.linalg.svd(B.A, full_matrices=False)
  PA = U @ U.T if space[0] == 'R' else np.eye(A.shape[1]) - Vt.T @ Vt
  PB = X @ X.T if space[1] == 'R' else np.eye(B.shape[1]) - Yt.T @ Yt
  if method == "neumann":
    PI = np.linalg.matrix_power(PA @ PB, 125)# von neumanns identity
  elif method == "anderson":
    PI = 2*(PA @ np.linalg.pinv(PA + PB) @ PB)
  elif method == "israel":
    L, s, _ = np.linalg.svd(PA @ PB, full_matrices=False)
    s_ind = np.flatnonzero(abs(s - 1.0) <= eps)
    PI = np.zeros(shape=PA.shape)
    for si in s_ind:
      PI += L[:,[si]] @ L[:,[si]].T
  else:
    raise ValueError("Invalid method given.")
  return(PI)
  # np.sum(abs(np.linalg.svd(np.c_[PI @ D1.T, PI @ D2])[1]))