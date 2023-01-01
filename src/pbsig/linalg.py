from typing import * 
from numpy.typing import ArrayLike
from numbers import *

import numpy as np 
import primme
from math import *
from scipy.sparse import *
from scipy.sparse.linalg import * 
from scipy.sparse.csgraph import structural_rank

## Local imports
from .simplicial import * 
import _lanczos as lanczos
import _laplacian as laplacian

# Neumanns majorization doesn't apply? rank_lb = sum(np.sort(diagonal(A)) >= tol)
def rank_bound(A: Union[ArrayLike, spmatrix, LinearOperator], upper: bool = True) -> int:
  assert A.shape[0] == A.shape[1], "Matrix 'A' must be square"
  if upper: 
    k_ = structural_rank(A) if isinstance(A, spmatrix) else A.shape[0]
    bound = min([A.shape[0], A.shape[1], sum(diagonal(A) != 0), k_])  
  else:
    ## https://konradswanepoel.wordpress.com/2014/03/04/the-rank-lemma/
    n, m = A.shape
    col_sum = lambda j: sum((A @ np.array([1 if i == j else 0 for i in range(m)]))**2)
    fn2 = sum([col_sum(j) for j in range(m)])
    bound = min([A.shape[0], A.shape[1], np.ceil(sum(diagonal(A))**2 / fn2)])
  return int(bound)
  
def sgn_approx(sigma: ArrayLike, eps: float = 0.0, p: float = 1.0, method: int = 0) -> ArrayLike:
  """ 
  Approximates the positive sgn+ function with a smooth relaxation

  If sigma = (x1, x2, ..., xn), where each x >= 0, then this function returns a vector (y1, y2, ..., yn) satisfying

  1. 0 <= yi <= 1
  2. sgn+(yi) = sgn+(xi)
  3. yi = sgn+(xi) as eps -> 0+
  4. sgn(xi, eps) <= sgn(xi, eps') for all eps >= eps'

  Parameters:
    sigma := array of non-negative values
    eps := smoothing parameter. Values close to 0 mimick the sgn function more closely. Larger values smooth out the relaxation. 
    p := float 
  """
  assert isinstance(method, Integral) and method >= 0 and method <= 3, "Invalid method given"
  # assert isinstance(sigma, np.ndarray), "Only numpy arrays supported for now"
  sigma = np.array(sigma)
  s1 = np.vectorize(lambda x: x**p / (x**p + eps**p))
  s2 = np.vectorize(lambda x: x**p / (x**p + eps))
  s3 = np.vectorize(lambda x: x / (x**p + eps**p)**(1/p))
  s4 = np.vectorize(lambda x: 1 - np.exp(-x/eps))
  phi = [s1,s2,s3,s4][method]
  return np.array([0.0 if np.isclose(s, 0.0) else phi(s) for s in sigma], dtype=sigma.dtype)

def smooth_rank(A: Union[ArrayLike, spmatrix, LinearOperator], pp: float = 1.0, solver: str = 'default', smoothing: tuple = (0.5, 1.5, 0), symmetric: bool = True, **kwargs) -> float:
  """ 
  Computes a smoothed-version of the rank of a positive semi-definite 'A' 
  
  The default solver 'default' chooses depends on the size, structure, and estimated rank of 'A'.

  Otherwise, if solver is 'irl' then 'dac'  

  Parameters: 
    A := array, sparse matrix, or linear operator to compute the rank of
    pp := proportional part of the spectrum to compute
    solver := One of 'default', 'dac', 'irl', 'lanczos', 'gd', 'jd', or 'lobpcg'.
    smoothing := tuple (eps, p, method) of args to pass to 'sgn_approx'
    symmetric := whether the 'A' is symmetric. Should always be true; non-symmetric matrices are not yet supported. 
    kwargs := keyword arguments to pass to the solver. Ignored if solver = 'default'.
  """
  assert A.shape[0] == A.shape[1], "A must be square"
  assert symmetric, "Haven't implemented non-symmetric yet"
  assert pp >= 0.0 and pp <= 1.0, "Proportion 'pp' must be between [0,1]"
  assert isinstance(smoothing, tuple) and len(smoothing) == 3, "Smoothing must a tuple of the parameters to pass"
  if pp == 0.0: return 0.0
  eps, p, method = smoothing 
  if solver == 'dac':
    assert isinstance(A, np.ndarray) or isinstance(A, spmatrix), "Cannot use divide-and-conquer with linear operators"
    if isinstance(A, spmatrix):
      import warnings
      warnings.warn("Converting sparse matrix to a dense matrix to use LAPACK divide-and-conquer routine.")
      A = A.todense()
  if isinstance(A, np.ndarray) and solver == 'default' or solver == 'dac':
    ew = np.linalg.eigvalsh(A, **kwargs)
  elif isinstance(A, spmatrix) or isinstance(A, LinearOperator):
    nev = trace_threshold(A, pp) if pp != 1.0 else rank_bound(A, upper=True)
    if nev == A.shape[0] and solver == 'irl':
      import warnings
      warnings.warn("Switching to PRIMME, as ARPACK cannot estimate all eigenvalues without shift-invert")
      solver = 'default' if solver == 'irl' else solver
    assert isinstance(solver, str) and solver in ['default', 'irl', 'lanczos', 'gd', 'jd', 'lobpcg']
    if solver == 'irl':
      ew = eigsh(A, k=nev, which='LM', return_eigenvectors=False, **kwargs)
    else:
      import primme
      methods = { 'lanczos' : 'PRIMME_Arnoldi', 'gd': "PRIMME_GD" , 'jd' : "PRIMME_JDQR", 'lobpcg' : 'PRIMME_LOBPCG_OrthoBasis', 'default' : 'PRIMME_DEFAULT_MIN_TIME' }
      ew = primme.eigsh(A, k=nev, which='LM', return_eigenvectors=False, method=methods[solver], **kwargs)
  else: 
    raise ValueError(f"Invalid solver / operator-type {solver}/{str(type(A))} given")
  ew = np.array([0.0 if np.isclose(v, 0.0) else v for v in ew], dtype=ew.dtype)
  assert all(np.isreal(ew)) and all(ew >= 0.0), "Negative or non-real eigenvalues detected. This method only works with symmetric PSD matrices."
  return sum(sgn_approx(np.sqrt(ew), eps, p, method))


def numerical_rank(A: Union[ArrayLike, spmatrix, LinearOperator], tol: float = None, solver: str = 'default', **kwargs) -> int:
  """ 
  Computes the numerical rank of a positive semi-definite 'A' via thresholding its eigenvalues.

  The eps-numerical rank r_eps(A) is the largest integer 'r' such that: 

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
  ev_cs = np.cumsum(diagonal(A))
  thresh_ind = np.flatnonzero(ev_cs/max(ev_cs) >= pp)[0] # index needed to obtain 80% of the spectrum
  return thresh_ind+1 # 0-based conversion

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


class UpLaplacian(LinearOperator):
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
    self._wfl = self._wfr = self._ws = UpLaplacian.identity_seq
    self.prepare()
    self._precompute_degree()
    
  
  ## Face weights
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

  ## Define the matvec operator using the precomputed degree, the pre-allocate workspace, and the pre-determined sgn pattern
  def _matvec(self, x: ArrayLike):
    self._v.fill(0)
    self._v += self.degree * x.reshape(-1)
    for s_ind, s in enumerate(self.simplices):
      for (f1, f2), sgn_ij in zip(combinations(s.boundary(), 2), self.sgn_pattern):
        ii, jj = self.index(f1), self.index(f2)
        self._v[ii] += sgn_ij * x[jj] * self._wfl[ii] * self._ws[s_ind] * self._wfr[jj]
        self._v[jj] += sgn_ij * x[ii] * self._wfl[jj] * self._ws[s_ind] * self._wfr[ii]
    return self._v

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



## TODO: change weight to optionally be a string when attr system added to SC's
def up_laplacian(K: SimplicialComplex, p: int = 0, weight: Optional[Callable] = None, normed=False, return_diag=False, form='array', dtype=None, **kwargs):
    """
    Returns the weighted combinatorial p-th up-laplacian of an abstract simplicial complex K. 

    Given D_p = boundary_matrix(K, p), the weighted p-th up-laplacian L is defined as: 

    Lp := W_p @ D_{p+1} @ W_{p+1} @ D_{p+1}^T @ W_p 
    
    Where W_p, W_{p+1} are diagonal matrices weighting the p and p+1 simplices, respectively. Note p = 0 (default), the results represents 
    the graph laplacians operator. This function is loosely based on SciPy 'laplacian' interface. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.laplacian.html.

    Parameters:
      S := Simplicial Complex 
      p := dimension of the up Laplacian. Defaults to 0 (graph Laplacian).
      weight := Callable weight function to be evaluated on each simplex in S. Must return either a value or a 2-tuple (wl, wr). 
      normed := 
      return_diag := whether to return diagonal degree matrix along with the laplacian
      form := return type. One of ['array', 'lo', 'function']
      dtype := dtype of associated laplacian. 
      kwargs := unused. 
    """
    assert isinstance(K, SimplicialComplex), "K must be a Simplicial Complex for now"
    assert isinstance(weight, Callable) if weight is not None else True
    ns = K.shape
    weight = (lambda s: 1.0) if weight is None else weight
    _ = weight(next(K.faces(p)))
    if not isinstance(_, Number) and len(_) == 2:
      W_LR = np.array([weight(s) for s in K.faces(p)]).astype(float)
      wpl = W_LR[:,0]
      wpr = W_LR[:,1]
    else: 
      wpl = np.array([float(weight(s)) for s in K.faces(p)])
      wpr = wpl
    wq = np.array([float(weight(s)) for s in K.faces(p+1)])
    assert len(wpl) == ns[p] and len(wq) == ns[p+1], "Invalid weight arrays given."
    if form == 'array':
      B = boundary_matrix(K, p = p+1)
      L = (diags(wpl) @ B @ diags(wq) @ B.T @ diags(wpr)).tocoo()
      return (L, L.diagonal()) if return_diag else L
    elif form == 'lo':
      p_faces = list(K.faces(p))        ## need to make a view 
      p_simplices = list(K.faces(p+1))  ## need to make a view 
      lo = UpLaplacian(p_simplices, p_faces)
      lo._wfl, lo._wfr, lo._ws = wpl, wpr, wq
      lo._precompute_degree()
      lo.prepare(1)
      return lo
      # f = _up_laplacian_matvec_p(p_simplices, p_faces, w0, w1, p, "default")
      # lo = f if form == 'function' else LinearOperator(shape=(ns[p],ns[p]), matvec=f, dtype=np.dtype(float))
    elif form == 'function':
      raise NotImplementedError("function form not implemented yet")
    else: 
      raise ValueError(f"Unknown form '{form}'.")


