import numpy as np
import _lanczos as lanczos
from numpy.typing import ArrayLike
from pbsig.simplicial import edge_iterator
import primme
from typing import * 
from numpy.typing import ArrayLike
from numbers import *

def testing_l():
  lanczos.lower_star_lanczos()
  print(dir(lanczos))

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


## Returns number of largest eigenvalues needed to capture t% of the spectrum 
def trace_threshold(A, p=0.80):
  assert p >= 0.0 and p <= 1.0, "Proportion must be in [0,1]"
  ev_cs = np.cumsum(A.diagonal())
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
  reduce, aggregate = (lambda x: x, False) if reduce is None else (reduce, True)
  for A in M: 
    A, args = (A[0], args | A[1]) if isinstance(A, tuple) and len(A) == 2 else (A, args)
    nev = trace_threshold(A, p)
    if default_opts['return_stats']:
      if default_opts['return_eigenvectors']:
        ew, ev, stats = primme.eigsh(A=A, k=nev, **default_opts)
        yield reduce(ew), ev, stats
      else:
        ew, stats = primme.eigsh(A=A, k=nev, **default_opts)
        yield reduce(ew), stats
    else: 
      if default_opts['return_eigenvectors']:
        ew, ev, stats = primme.eigsh(A=A, k=nev, **default_opts)
        yield reduce(ew), ev
      else:
        ew = primme.eigsh(A=A, k=nev, **default_opts)
        yield reduce(ew)
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
  lo = LO(L)
  return lo

from pbsig.simplicial import SimplicialComplex
## TODO: make generalized up- and down- adjacency matrices for simplicial complexes

# class UpLaplacian(LinearOperator):
#   def __init__(self):

#   def __matvec__(self):
# from types import MappingProxyType

# No F needed because faces are assumed to be vertices in sorted order, V = [n]
def _up_laplacian_matvec_1(S: Collection['Simplex'],  w0: ArrayLike, w1: ArrayLike):
  # .shape and .matvec and .dtype attributes
  n,m = len(F), len(S)
  v = np.zeros(n) # note the O(n) memory
  def _matvec(x: ArrayLike): 
    nonlocal v
    v.fill(0)
    for cc, (i,j) in enumerate(S):
      v[i] += w1[cc]**2
      v[j] += w1[cc]**2
    v = w0**2 * v * x
    for cc, (i,j) in enumerate(S):
      v[i] -= x[j]*w0[i]*w0[j]*w1[cc]**2
      v[j] -= x[i]*w0[i]*w0[j]*w1[cc]**2
    return v
  return _matvec

  def _up_laplacian_matvec_p(S: Iterable['Simplex'], F: Sequence['Simplex'],  w0: ArrayLike, w1: ArrayLike, p: int, sgn_pattern = "default"):
    sgn_pattern = list(islice(cycle([1,-1]), p+2)) if sgn_pattern == "default" else sgn_pattern
    lap_sgn_pattern = np.array([s1*s2 for (s1,s2) in combinations(sgn_pattern,2)])
    n = len(F) # F must have an index method, so it must be at least a Sequence
    assert n == len(w0), "Invalid w0"
    if hasattr(S, '__len__'): assert len(S) == len(w1), "Invalid w1"
    #n,m = len(F), len(S)
    #assert len(w0) == n and len(w1) == m, "Invalid weight array"
    v = np.zeros(n) # workspace
    def _matvec(x: ArrayLike):
      nonlocal v
      v.fill(0)
      for t_ind, t in enumerate(S):
        for (e1, e2), s_ij in zip(combinations(t.boundary(), 2), lap_sgn_pattern):
          ii, jj = F.index(e1), F.index(e2)
          c = w0[ii] * w0[jj] * w1[t_ind]**2
          v[ii] += x[jj] * c * s_ij
          v[jj] += x[ii] * c * s_ij
      for t_ind, t in enumerate(S):
        for e in t.boundary():
          ii = F.index(e)
          v[ii] += x[ii] * w1[t_ind]**2 * w0[ii]**2
      return v
    return _matvec


def up_laplacian(K: SimplicialComplex, w0 = None, w1 = None, p: int = 0, normed=False, return_diag=False, form='array', dtype=None, **kwargs):
    """
    Returns the weighted combinatorial p-th up-laplacian of an abstract simplicial complex K. 

    Given D = boundary_matrix(K, p), the weighted p-th up-laplacian L is defined as: 

    Lp := W0 @ D @ W1 @ D^T @ W0 
    
    Where W0, W1 = diag(w0), diag(w1) are diagonal matrices weighting the p and p+1 simplices, respectively.

    Note p = 0 (default), the results represents the graph laplacians operator. 

    *Note to self*: You need weights as parameters. If the form = 'lo' or 'function', then you can't post-compose the resulting 
    matrix-free matvec product w/ weights. 

    SciPy accepts weights by accepting potentially sparse adjacency matrix input. 

    Based on SciPy 'laplacian' interface. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.laplacian.html.
    """
    assert isinstance(K, SimplicialComplex), "K must be a Simplicial Complex for now"
    ns = K.dim() 
    w0 = np.repeat(1, ns[p]) if w0 is None else w0
    w1 = np.repeat(1, ns[p+1]) if w1 is None else w1
    assert len(w0) == ns[p] and len(w1) == ns[p+1], "Invalid weight arrays given."
    if form == 'array':
      B = boundary_matrix(K, p = p+1)
      L = (diags(w0) @ B @ diags(w1) @ B.T @ diags(w0)).tocoo()
      return L
    elif form == 'lo' or form == 'function':
      p_faces = list(K.faces(1)) ## need to make a view 
      p_simplices = list(K.faces(2)) ## need to make a view 
      f = _up_laplacian_matvec_p(p_simplices, p_faces, w0, w1, p, "default")
      lo = f if form == 'function' else LinearOperator(shape=(ns[p],ns[p]), matvec=f,dtype=np.dtype(float))
      return lo
