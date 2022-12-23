from typing import * 
from numpy.typing import ArrayLike
from numbers import *

from math import prod 
from scipy.sparse import *
from scipy.sparse.linalg import * 
import numpy as np
import _lanczos as lanczos
import _laplacian as laplacian
import primme

## Local imports
from .simplicial import * 

# def testing_l():
#   lanczos.lower_star_lanczos()
#   print(dir(lanczos))
# eigh(L.todense())

# def bound_rank(A: Union[ArrayLike, spmatrix, LinearOperator], tol: float):
#   ## Use tolerance + Neumans trace inequality / majorization result to lower bound on rank / upper bound nullity
#   # return sum(np.sort(diagonal(A)) >= tol) # rank lower bound



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
    rank_lb = int(np.ceil((sum(diagonal(A))**2)/sum(A.data**2)))
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
      solver = lambda self, b: cg(A+cI, b, tol=tol, atol=tol)[0]
    elif solver == 'lgmres':
      solver = lambda self, b: lgmres(A+cI, b, tol=tol, atol=tol)[0]
    else: 
      raise ValueError(f"Unknown Ax=b solver '{solver}'")
    solver = aslinearoperator(type('Ax_solver', (type(A),), { '_matvec' : solver })(A))

    ## Use tolerance + Neumans trace inequality / majorization result to lower bound on rank / upper bound nullity
    rank_lb = sum(np.sort(diagonal(A)) >= tol)
    k = A.shape[0]-rank_lb

    ## Use un-shifted operator 'A' + shifted solver to estimate the dimension of the nullspace w/ shift-invert mode
    smallest_k = eigsh(A, k=k, which='LM', sigma=0.0, OPinv = solver, tol=tol, return_eigenvectors=False)
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
  Calculates all 
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
def trace_threshold(A, p=0.80):
  assert p >= 0.0 and p <= 1.0, "Proportion must be in [0,1]"
  ev_cs = np.cumsum(diagonal(A))
  thresh_ind = np.flatnonzero(ev_cs/max(ev_cs) >= p)[0] # index needed to obtain 80% of the spectrum
  return thresh_ind

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

from pbsig.simplicial import SimplicialComplex
## TODO: make generalized up- and down- adjacency matrices for simplicial complexes

# class UpLaplacian(LinearOperator):
#   def __init__(self):

#   def __matvec__(self):
# from types import MappingProxyType

# No F needed because faces are assumed to be vertices in sorted order, V = [n]
def _up_laplacian_matvec_1(S: Collection['Simplex'], w0: ArrayLike, w1: ArrayLike):
  # .shape and .matvec and .dtype attributes
  n,m = len(w0), len(S)
  v = np.zeros(n) # note the O(n) memory
  def _matvec(x: ArrayLike): 
    nonlocal v
    v.fill(0)
    for cc, (i,j) in enumerate(S):
      v[i] += w1[cc]**2
      v[j] += w1[cc]**2
    v *= w0**2 * x
    for cc, (i,j) in enumerate(S):
      v[i] -= x[j]*w0[i]*w0[j]*w1[cc]**2
      v[j] -= x[i]*w0[i]*w0[j]*w1[cc]**2
    return v
  return _matvec


class UpLaplacian(LinearOperator):
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
    self._wf = None
    self._ws = None
    self.prepare()
  
  ## Face weights
  @property 
  def face_weights(self): 
    return self._wf

  @face_weights.setter
  def face_weights(self, value: ArrayLike) -> None:
    if value is None: 
      self._wf = None 
    else: 
      assert len(value) == self.shape[0], "Invalid value given. Must match shape."
      assert isinstance(value, np.ndarray)
      self._wf = value

  ## S-weights
  @property
  def simplex_weights(self): 
    return self._ws
  
  @simplex_weights.setter
  def simplex_weights(self, value: ArrayLike):
    if value is None: 
      self._ws = None
    else:
      # if hasattr(self.S, '__len__'): assert len(self.S) == len(value), "Invalid weight vector given."
      assert isinstance(value, np.ndarray) 
      self._ws = value ## length is unchecked! 

  def prepare(self, level: int = 0) -> None: 
    # self.index = type('_indexer', (), { '__getitem__' : lambda })
    if level == 0:
      self.index = lambda s: self.faces.index(s)
    elif level == 1:
      self._index = { f: i for i, f in enumerate(self.faces) }
      self.index = lambda s: self._index[s]
    else: 
      raise NotImplementedError("Not implemented yet")

  ## Define the matvec operator as a closure
  def _matvec(self, x: ArrayLike):
    v = self._v
    v.fill(0)
    if self._wf is None and self._wf is None:
      for s_ind, s in enumerate(self.simplices):
        for (f1, f2), sgn_ij in zip(combinations(s.boundary(), 2), self.sgn_pattern):
          ii, jj = self.index(f1), self.index(f2)
          v[ii] += x[jj] * sgn_ij
          v[jj] += x[ii] * sgn_ij
      for s_ind, s in enumerate(self.simplices):
        for f in s.boundary():
          ii = self.index(f)
          v[ii] += x[ii]
    elif (self._ws is None) and not(self._wf is None):
      for s_ind, s in enumerate(self.simplices):
        for (f1, f2), sgn_ij in zip(combinations(s.boundary(), 2), self.sgn_pattern):
          ii, jj = self.index(f1), self.index(f2)
          c = self._wf[ii] * self._wf[jj]
          v[ii] += x[jj] * c * sgn_ij
          v[jj] += x[ii] * c * sgn_ij
      for s_ind, s in enumerate(self.simplices):
        for f in s.boundary():
          ii = self.index(f)
          v[ii] += x[ii] * self._wf[ii]**2
    elif not(self._ws is None) and (self._wf is None):
      for s_ind, s in enumerate(self.simplices):
        for (f1, f2), sgn_ij in zip(combinations(s.boundary(), 2), self.sgn_pattern):
          ii, jj = self.index(f1), self.index(f2)
          c = self._ws[s_ind]
          v[ii] += x[jj] * c * sgn_ij
          v[jj] += x[ii] * c * sgn_ij
      for s_ind, s in enumerate(self.simplices):
        for f in s.boundary():
          ii = self.index(f)
          v[ii] += x[ii] * self._ws[s_ind]
    else:
      for s_ind, s in enumerate(self.simplices):
        for (f1, f2), sgn_ij in zip(combinations(s.boundary(), 2), self.sgn_pattern):
          ii, jj = self.index(f1), self.index(f2)
          c = self._wf[ii] * self._wf[jj] * self._ws[s_ind]
          v[ii] += x[jj] * c * sgn_ij
          v[jj] += x[ii] * c * sgn_ij
      for s_ind, s in enumerate(self.simplices):
        for f in s.boundary():
          ii = self.index(f)
          v[ii] += x[ii] * self._ws[s_ind] * self._wf[ii]**2
    return v

## From: https://github.com/cvxpy/cvxpy/blob/master/cvxpy/interface/matrix_utilities.py
def is_symmetric(A) -> bool:
  """Check if a matrix is Hermitian and/or symmetric.
  """
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

    Given D = boundary_matrix(K, p), the weighted p-th up-laplacian L is defined as: 

    Lp := W_p @ D @ W_{p+1} @ D^T @ W_p 
    
    Where W_p, W_{p+1} are diagonal matrices weighting the p and p+1 simplices, respectively.

    Note p = 0 (default), the results represents the graph laplacians operator. 

    *Note to self*: You need weights as parameters. If the form = 'lo' or 'function', then you can't post-compose the resulting matrix-free matvec product w/ weights. 

    SciPy accepts weights by accepting potentially sparse adjacency matrix input. 

    Based on SciPy 'laplacian' interface. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.laplacian.html.
    """
    assert isinstance(K, SimplicialComplex), "K must be a Simplicial Complex for now"
    assert isinstance(weight, Callable) if weight is not None else True
    ns = K.dim() 
    w0 = np.array([float(weight(s)) for s in K.faces(p)]) if weight is not None else np.repeat(1, ns[p]) 
    w1 = np.array([float(weight(s)) for s in K.faces(p+1)]) if weight is not None else np.repeat(1, ns[p+1])
    assert len(w0) == ns[p] and len(w1) == ns[p+1], "Invalid weight arrays given."
    if form == 'array':
      B = boundary_matrix(K, p = p+1)
      L = (diags(w0) @ B @ diags(w1) @ B.T @ diags(w0)).tocoo()
      return L
    elif form == 'lo' or form == 'function':
      p_faces = list(K.faces(p))        ## need to make a view 
      p_simplices = list(K.faces(p+1))  ## need to make a view 
      lo = UpLaplacian(p_simplices, p_faces)
      lo.simplex_weights = w1
      lo.simplex_weights = w0
      # f = _up_laplacian_matvec_p(p_simplices, p_faces, w0, w1, p, "default")
      # lo = f if form == 'function' else LinearOperator(shape=(ns[p],ns[p]), matvec=f, dtype=np.dtype(float))
      return lo
    else: 
      raise ValueError(f"Unknown form '{form}'.")
