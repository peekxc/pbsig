## Typing support 
from typing import *
from numpy.typing import ArrayLike 

## Module imports
import numpy as np
import math
import numbers 
import scipy.sparse as sps

## Function/structure imports
from scipy.sparse import *
from array import array
from itertools import *
from scipy.special import *
from scipy.spatial.distance import *

## Relative package imports
# import _boundary as boundary
import _persistence as pm
from .apparent_pairs import *
from .utility import *
from .simplicial import *

_perf = {
  "n_col_adds" : 0,
  "n_field_ops": 0
}

def persistent_betti(D1, D2, i: int, j: int, summands: bool = False):
  """
  Computes the p-th persistent Betti number Bij of a pair of boundary matrices. 

  Bij = dim(Zp(Ki)) - dim(Zp(Ki) \cap Bp(Kj))

  Parameters:
    D1 := p-th filtered boundary matrix 
    D2 := (p+1) filtered boundary matrix 
    i := Birth index (integer)
    j := Death index (integer)
    summands := if True, returns a tuple (a, b, c) where the persistent Betti number B = a - b - c. Defaults to False. See details. 

  Given filtered boundary matrices (D1, D2) of dimensions (n,m) and (m,l), respectively, and 
  indices i \in [1, m] and j \in [1, l], returns the persistent Betti number 
  representing the number of p-dimensional persistent homology groups born at 
  or before index i (in D1) that were not destroyed by inclusion of the first j 
  (p+1)-simplices (D2). Note that (i,j) are given here as 1-based, index-coordinates. 
  Typically, the indices i and j are derived from geometric scaling parameters. 

  If summands = True, then a tuple (a, b, c) is returned where: 
    a := dim(Cp(Ki))
    b := dim(B{p-1}(Ki))
    c := dim(Zp(Ki) \cap Bp(Kj))
  The corresponding persistent Betti number is Bij = a - (b + c). 
  This can be useful for assessing the size of the group in the denominator (c).
  """
  #assert i < j, "i must be less than j"
  i, j = int(i), int(j)
  if (i == 0): 
    return((0, 0, 0, 0) if summands else 0)
  t1 = D1[:,:i].shape[1] # i, D1.shape[1]
  t2 = 0 if np.prod(D1[:,:i].shape) == 0 else np.linalg.matrix_rank(D1[:,:i].A)
  D2_tmp = D2[:,:j]
  t3_1 = 0 if np.prod(D2_tmp.shape) == 0 else np.linalg.matrix_rank(D2_tmp.A)
  D2_tmp = D2_tmp[i:,:] # note i is 1-based, so this is like (i+1)
  t3_2 = 0 if np.prod(D2_tmp.shape) == 0 else np.linalg.matrix_rank(D2_tmp.A)
  return((t1, t2, t3_1, t3_2) if summands else t1 - t2 - t3_1 + t3_2)

def low_entry(D: lil_array, j: Optional[int] = None):
  """ 
  Provides fast access to all the low entries of a given matrix 'D'
  
  If D is CSC, low(D,j) takes O(1) time
  
  """
  from scipy.sparse import sparray
  #assert isinstance(D, csc_matrix)
  if j is None: 
    assert (isinstance(D, spmatrix) or isinstance(D, sparray)) or isinstance(D, np.ndarray)
    m = np.repeat(-1, D.shape[1])
    r,c = D.nonzero() if isinstance(D, sparray) else np.argwhere(D != 0).T
    np.maximum.at(m,c,r)
    return m
    # return(np.array([low_entry(D, j) for j in range(D.shape[1])], dtype=int))
  else: 
    if isinstance(D, csc_matrix):
      nnz_j = np.abs(D.indptr[j+1]-D.indptr[j]) 
      return(D.indices[D.indptr[j]+nnz_j-1] if nnz_j > 0 else -1)
    elif isinstance(D, spmatrix) or isinstance(D, sparray): 
      return(-1 if D[:,[j]].getnnz() == 0 else max(D[:,[j]].nonzero()[0]))
      # max(zip(*R[:,[1]].nonzero()), key=lambda rc: (rc[0], rc[1] != 0))
    elif isinstance(D, np.ndarray):
      return(-1 if all(D[:,j] == 0) else max(np.flatnonzero(D[:,j] != 0)))
    else: 
      raise ValueError(f"Invalid input type {type(D)}for D ")

def pHcol(R: lil_array, V: lil_array, I: Optional[Iterable] = None):
  # assert isinstance(R, csc_matrix) and isinstance(V, csc_matrix), "Invalid inputs"
  assert R.shape[1] == V.shape[0], "Must be matching boundary matrices"
  m = R.shape[1]
  I = range(1, m) if I is None else I
  piv = np.array([low_entry(R, j) for j in range(m)], dtype=int)
  for j in I:
    while( piv[j] != -1 and np.any(J := piv[:j] == piv[j]) ):
      i = np.flatnonzero(J)[0] # i < j
      c = R[piv[j],j] / R[piv[i],i]
      R[:,[j]] -= c*R[:,[i]] 
      V[:,[j]] -= c*V[:,[i]] 
      _perf['n_col_adds'] += 2
      # print(np.max(np.linalg.svd(R.A)[1]))
      if isinstance(R, csc_matrix): 
        R.eliminate_zeros() # needed to changes the indices
      # piv[j] = -1 if len(R[:,j].indices) == 0 else R[:,j].indices[-1] # update pivot array
      piv[j] = -1 if R[:,[j]].getnnz() == 0 else max(R[:,[j]].nonzero()[0]) # update pivot array
  return(None)



def reduction_pHcol(D1: lil_array, D2: lil_array, clearing: bool = False):
  # assert isinstance(D1, csc_matrix) and isinstance(D2, csc_matrix), "Invalid boundary matrices"
  assert D1.shape[1] == D2.shape[0], "Must be matching boundary matrices"
  V1, V2 = sps.identity(D1.shape[1]).tolil(), sps.identity(D2.shape[1]).tolil()
  R1, R2 = D1.copy(), D2.copy()
  if not(clearing):
    pHcol(R1, V1)
    pHcol(R2, V2)
  else: 
    pHcol(R2, V2)
    cleared = np.array(list(filter(lambda p: p != -1, low_entry(R2))))
    uncleared = np.setdiff1d(np.array(list(range(R1.shape[1]))), cleared)
    pHcol(R1, V1, uncleared)
    R1[:,cleared] = 0
    #R1.eliminate_zeros()
    cleared_domain = np.flatnonzero(low_entry(R2) != -1)
    V1[:,cleared] = R2[:,cleared_domain] # set Vi = Rj
  return((R1, R2, V1, V2))

def reduction_pHcol_clearing(D: Iterable[lil_array], implicit: bool = False):
  """ 
  D must be given as an iterable of boundary matrices given in decreasing dimensions
  """
  R, V = [], []
  cleared = np.array([], dtype=int)
  for Dp in D:
    m_p = Dp.shape[1] 
    Rp, Vp = Dp.copy(), sps.identity(m_p).tocsc()
    uncleared = np.array(list(filter(lambda p: not(p in cleared), range(m_p))))
    pHcol(Rp, Vp, uncleared) # is_reduced(Rp[:,uncleared]) == true
    R.append(Rp)
    V.append(Vp)
    cleared = np.array(list(filter(lambda p: p != -1, low_entry(Rp))))
  return((R, V))

def is_reduced(R: lil_array) -> bool:
  """ Checks if a matrix R is indeed reduced. """
  # assert isinstance(R, csc_matrix), "R must be a CSC sparse matrix."
  if isinstance(R, csc_matrix): 
    R.eliminate_zeros() # fix indices if need be
  low_ind = np.array([low_entry(R, j) for j in range(R.shape[1])], dtype=int)
  low_ind = low_ind[low_ind != -1]
  return(len(np.unique(low_ind)) == len(low_ind))

def as_dgm(dgm) -> np.ndarray:
  # dgm = dgms[1]
  if dgm.dtype.names is not None and 'birth' in dgm.dtype.names and 'death' in dgm.dtype.names:
    return dgm
  dgm = np.atleast_2d(dgm)
  dgm_dtype = [('birth', 'f4'), ('death', 'f4')]
  # return np.array(*dgm, dtype=dgm_dtype)
  return np.array([tuple(pair) for pair in dgm], dtype=dgm_dtype)

def validate_decomp(D: sparray, R: sparray, V: sparray, epsilon: float = 10*np.finfo(float).eps, warn: bool = True):
  tol = 10*np.finfo(R.dtype).eps if not np.issubdtype(R.dtype, np.integer) else 0
  _is_reduced = is_reduced(R)
  dv_r_diff = abs(((D @ V) - R).data)
  v_upper_diff = abs((V - triu(V)).data)
  _is_rv_decomp = len(dv_r_diff) == 0 or np.isclose(sum(dv_r_diff.flatten()), tol)
  _is_upper_tri = len(v_upper_diff) == 0 or np.isclose(np.sum(v_upper_diff), tol)
  valid = _is_reduced and _is_rv_decomp and _is_upper_tri
  if not(valid) and warn:
    import warnings 
    warnings.warn(f"Decomposition failed validation checks.\n 1. R is reduced? {_is_reduced}\n 2. R=DV? {_is_rv_decomp} \n 3. V is upper-triangular? {_is_upper_tri}")
  return(valid)

def generate_dgm(K: FiltrationLike, R: sparray, collapse: bool = True, simplex_pairs: bool = False, essential: float = float('inf')) -> ArrayLike :
  """ Returns the persistence diagram from (K, R) """
  rlow = low_entry(R)
  sdim = np.array([dim(s) for s in faces(K)])
  
  ## Get the indices of the creators and destroyers
  if any(rlow == -1):
    creator_mask = rlow == -1
    creator_dim = sdim[creator_mask]
    birth = np.flatnonzero(creator_mask)      ## indices of zero columns 
    death = np.repeat(essential, len(birth))  ## indices of destroyer columns
    death[np.searchsorted(birth, rlow[~creator_mask])] = np.flatnonzero(~creator_mask)
  else:
    creator_dim = np.empty(0)
    birth = np.empty(shape=(0, 1))
    death = np.empty(shape=(0, 1))
  
  ## Trivial case
  if len(creator_dim) == 0: return dict()

  ## Match the barcodes with the index set of the filtration
  # from more_itertools import collapse
  filter_vals = np.ravel(np.array([fval for fval, s in K])) # np.ravel(np.fromiter(K.indices(), dtype=key_type)) if hasattr(K, "indices") else 
  index2fv = dict(zip(np.arange(len(K)), filter_vals)) | { essential : essential} | { -essential : -essential}

  ## Assemble the diagram
  if simplex_pairs:
    ## At this point, birth/death are integer indices!
    # assert isinstance(K, Sequence), "Filtration-like object must support be Sequence semantics to attach simplex pairs"
    assert hasattr(K, "__getitem__"), "Filtration-like object must support be Sequence semantics to attach simplex pairs"
    dgm_dtype = [('birth', 'f4'), ('death', 'f4'), ('creators', 'O'), ('destroyers', 'O')]
    # if is_filtration_like(K):
    #   print(K[birth[0]])
    #   creators = (Simplex(K[b][1]) for b in birth) 
    #   destroyers = (Simplex(K[int(d)][1]) if not(np.isinf(d)) else Simplex([np.nan]) for d in death)
    # else:
    creators = (Simplex(K[b]) for b in birth)
    destroyers = (Simplex(K[int(d)]) if not(np.isinf(d)) else Simplex([np.nan]) for d in death)
    birth = np.array([index2fv[i] for i in birth])
    death = np.array([index2fv[i] for i in death])
    dgm = np.fromiter(zip(birth, death, creators, destroyers), dtype=dgm_dtype)
  else:
    dgm_dtype = [('birth', 'f4'), ('death', 'f4')]
    birth = np.array([index2fv[i] for i in birth])
    death = np.array([index2fv[i] for i in death])
    dgm = np.fromiter(zip(birth, death), dtype=dgm_dtype)
  
  ## Collapse zero-persistence pairs if warranted
  if collapse:
    nonzero_ind = ~np.isclose(dgm['death'], dgm['birth'], equal_nan=True) 
    dgm = dgm[nonzero_ind]
    creator_dim = creator_dim[nonzero_ind]

  ## Split diagram based on homology dimension
  dgm = { p : np.take(dgm, np.flatnonzero(creator_dim == p)) for p in np.sort(np.unique(creator_dim)) }
  for p in dgm.keys():
    if len(dgm[p]) == 0:
      dgm[p] = np.empty((0,2),dtype=[('birth', 'f4'), ('death', 'f4')])
  return dgm 

def cycle_generators(K: FiltrationLike, V: sparray, R: sparray = None, collapse: bool = True):
  G = []
  piv = np.flatnonzero(low_entry(R))
  # G = dict(zip(np.flatnonzero(piv == -1), repeat([])))
  # V = V.tocsr()
  # V.sort_indices()
  GR = {}
  for r,c in zip(*V.nonzero()): 
    if c == r or c in piv:
      GR.setdefault(r, []).append(c) ## simplex at index 'r' contributes to cycle at index 'c' 

  S = list(faces(K))
  G = {}
  for k,v in GR.items():
    for c in v: 
      G.setdefault(c, []).append(S[k])

  if collapse: 
    G = { k : v for k,v in G.items() if len(v) > 1 }
  return G
  # sel = np.zeros(len(K))
  # sel[piv] = 1
  
  # G = {}
  # for (k,v),s in zip(GR.items(), compress(faces(K), sel)):
  #   for c in v: 
  #     G.setdefault(c, []).append(s)

  # for i, s in enumerate(faces(K)):
  #   pass
  # pass 
from splex import boundary_matrix

## TODO: redo with filtration class at some point
def ph(K: FiltrationLike, p: Optional[int] = None, output: str = "dgm", engine: str = ["python", "cpp", "dionysus"], field="reals", validate: bool = True,  **kwargs):
  """
  Given a filtered simplicial complex 'K' and optionally an integer p >= 0, generate p-dimensional barcodes

  If p is not specified, all p-dimensional barcodes are generated up to the dimension of the filtration.

  Parameters: 
    K: filtered simplicial complex. 
    p: Homology dimension to compute. If None (default), computes all of them.  
    output: the preferred output. Can be any combination of ["dgm", "generators", "RV"]
    engine: the persistence implementation to use. 
  """
  assert isinstance(K, FiltrationLike), "Only accepts filtration objects for now"
  engine = "cpp" if isinstance(engine, list) and engine == ["python", "cpp", "dionysus"] else engine
  assert isinstance(engine, str), f"Supplied engine '{engine}' must be string argument"
  if p is None:
    if engine == "python":
      R, V = boundary_matrix(K), sps.identity(len(K), dtype=np.int64).tolil()
      pHcol(R, V)
      assert validate_decomp(boundary_matrix(K), R, V) if validate else True
      return generate_dgm(K, R, **kwargs) if output == "dgm" else (R,V)
    elif engine == "cpp": 
      D, V = boundary_matrix(K).astype(np.float32), sps.identity(len(K)).astype(np.float32) # .astype(np.int64)
      R, V = pm.phcol(D, V, range(len(K)))  
      assert validate_decomp(D, R, V) if validate else True
      return generate_dgm(K, R, **kwargs) if output == "dgm" else (R,V)
    elif engine == "dionysus":
      assert output == "dgm", "Dionysus only produces diagram outputs. Please use a different engine."
      dgm = ph_dionysus(K)
      return dgm
    else: 
      raise ValueError("Unknown engine type")
  else: 
    raise NotImplementedError("Haven't implement yet")
  

# def persistent_pairs(R: List[csc_matrix], K: Dict, F: List[ArrayLike] = None, cohomology: bool = False):
# p_birth = np.where(low_entry(R1) == -1)
# p_death = 
#ind = np.setdiff1d(np.fromiter(range(Rp.shape[1]), dtype=int), cleared)
#low_p = np.array([low_entry(Rp, j) for j in range(Rp.shape[1])], dtype=int)
#cleared = low_p[low_p != -1]
# piv1 = np.array([low_entry(D1,j) for j in range(m)], dtype=int
## Reduce D1
# for j in range(1, m):
#   while( piv1[j] != -1 and np.any(J := piv1[:j] == piv1[j]) ):
#     i = np.flatnonzero(J)[0] # i < j
#     c = D1[piv1[j],j] / D1[piv1[i],i]
#     D1[:,j] -= c*D1[:,i] # real-valued case
#     V1[:,j] -= c*V1[:,i] # also do V1
#     D1.eliminate_zeros() # needed to changes the indices
#     piv1[j] = -1 if len(D1[:,j].indices) == 0 else D1[:,j].indices[-1]
## Reduce D2
# piv2 = np.array([low_entry(D2,j) for j in range(k)], dtype=int)
# for j in range(1, k):
#   while( piv2[j] != -1 and np.any(J := piv2[:j] == piv2[j]) ):
#     i = np.flatnonzero(J)[0] # i < j
#     c = D2[piv2[j],j] / D2[piv2[i],i]
#     D2[:,j] -= c*D2[:,i] # real-valued case
#     V2[:,j] -= c*V2[:,i] # also do V1
#     D2.eliminate_zeros() # needed to changes the indices
#     piv2[j] = -1 if len(D2[:,j].indices) == 0 else D2[:,j].indices[-1]

## Add persistent pairs
# I1 = D1.indices[D1.indptr[1]:D1.indptr[2]]
# I2 = D1.indices[D1.indptr[2]:D1.indptr[3]]
# np.bitwise_xor
# np.setxor1d(I1, I2)
# D1[:,2] = D1[:,2] + D1[:,3]
# raise ValueError("test")

# class Reduction():
#   def __init__(D0: csc_matrix, D1: csc_matrix): # boundary matrix


# def boundary_matrix(S: Iterable, F: Iterable):
#   """
#   Creates a p-boundary matrix (CSC format) from an iterable of simplices.

#   Assumes S only enumerates p-simplices.

#   Parameters:
#     S := Iterable over p-simplices
#     F := Iterable over (p-1)-faces
#   """ 
#   array('I')


def bisection_tree_top_down(mu_q: Callable, ind: tuple, mu: int, splitrows: bool = True, verbose: bool = False):
  assert len(ind) == 4 and tuple(sorted(ind)) == ind, "Invalid box given. Must be in upper-half plane."
  i,j,k,l = ind 
  if mu == 1 and (i == j if splitrows else k == l):
    piv = i if splitrows else k
    if verbose: print(f"[{mu}]: ({i},{j},{k},{l}): creator = {i}, destroyer = {k} ({piv})")
    yield piv
  else:
    p = int(np.floor((i+j)/2)) if splitrows else int(np.floor((k+l)/2)) # pivot
    ind_lc = (i,p,k,l) if splitrows else (i,j,k,p)
    ind_rc = (p+1,j,k,l) if splitrows else (i,j,p+1,l)
    mu_lc = mu_q(*ind_lc) #np.linalg.matrix_rank(M[i:(piv+1),k:(l+1)].todense())
    mu_rc = mu_q(*ind_rc) #np.linalg.matrix_rank(M[(piv+1):(j+1),k:(l+1)].todense())
    if verbose: print(f"[{mu}]: ({i},{j},{k},{l}), L:{mu_lc}, R:{mu_rc}")
    assert mu == mu_lc + mu_rc
    if mu_lc > 0:
      yield from bisection_tree_top_down(mu_q,ind_lc,mu_lc,splitrows)
    if mu_rc > 0:
      yield from bisection_tree_top_down(mu_q,ind_rc,mu_rc,splitrows)

def pairs_in_box(box: tuple, mu_rank: Callable):
  assert len(box) == 4 and tuple(sorted(box)) == box
  dgm_res = []
  creators = list(bisection_tree_top_down(mu_rank, ind=box, mu=mu_rank(*box), splitrows=True))
  for idx in creators:
    d = list(bisection_tree_top_down(mu_rank, ind=(idx, idx, box[2], box[3]), mu=mu_rank(idx, idx, box[2], box[3]), splitrows=False))
    assert len(d) == 1
    dgm_res.append([idx, d[0]])
  return dgm_res 

## Generate all the boxes between [a,b] into the list 'res'
def generate_boxes(a: int, b: int, res: list = []):
  res.append((a, (a+b) // 2, (a+b) // 2, b))
  if abs(a-b) <= 1: 
    return  
  else:
    generate_boxes(a, (a+b) // 2, res)
    generate_boxes((a+b) // 2, b, res)

from more_itertools import collapse, chunked, unique_everseen
from scipy.sparse import spmatrix, sparray 

def ph_rank(D: sparray = None):
  """Chens rank-based persistence algorithm.
  
  Computes the persistence pairing constituting a diagram from only rank-based multplicity queries. 

  Currently limited to dense, full boundary matrix inputs and naive rank computations.
  """
  assert isinstance(D, sparray) or isinstance(D, spmatrix), "Must be one of sparse matrix or array"
  ## If not given, use a default dense multiplicity algorithm
  B = D.todense()
  rrank = lambda i,j,k,l: 0 if len(B[i:j,k:l]) == 0 else np.linalg.matrix_rank(B[i:j,k:l])
  mu_rank = lambda i,j,k,l: rrank(i,l+1,i,l+1) - rrank(i,k,i,k) - rrank(j+1,l+1,j+1,l+1) + rrank(j+1,k,j+1,k)

  ## Generate all 2n-1 boxes 
  boxes = []
  generate_boxes(0, B.shape[1], boxes) ## these are square boxes; need way of generating rectangular

  ## Compute all the persistence pairs 
  pairs = [pairs_in_box(box, mu_rank) for box in boxes]

  ## Post-process to produce a nice diagram 
  pairs = list(chunked(collapse(pairs), 2))
  pairs = np.fromiter(iter(set(map(tuple, pairs))), dtype=[('birth', 'f4'), ('death', 'f4')])
  return pairs


def sw_parameters(bounds: Tuple, d: int = None, tau: float = None,  w: float = None, L: int = None):
  """ 
  Computes 'nice' slidding window parameters. 
  
  Given an interval [a,b] and any two slidding window parameters (d, w, tau, L), 
  returns the necessary parameters (d, tau) needed to construct Takens slidding window embedding 
  
  """
  if not(d is None) and not(tau is None):
    return(d, tau)
  elif not(d is None) and not(w is None):
    return(d, w/d)
  elif not(d is None) and not(L is None):
    w = bounds[1]*d/(L*(d+1))
    return(d, w/d)
  elif not(tau is None) and not(w is None):
    d = int(w/tau)
    return(d, tau)
  elif not(tau is None) and not(L is None):
    d = bounds[1]/(L*tau) - 1
    return(d, tau)
  elif not(w is None) and not(L is None):
    d = np.round(1/((bounds[1] - w*L)/(w*L)))
    assert d > 0, "window*L must be less than the entire time duration!"
    return(d, w/d)
  else:
    raise ValueError("Invalid combination of parameters given.")


def sliding_window(f: Union[ArrayLike, Callable], bounds: Tuple = (0, 1)):
  ''' 
  Slidding Window Embedding of a time series.
  
  Returns a function which generates a n-point slidding window point cloud of a fixed time-series/function _f_. 

  The returned function has the parameters
    - n := number of point to generate the embedding
    - d := dimension-1 of the resulting embedding 
    - w := (optional) window size each (d+1)-dimensional delay coordinate is derived from
    - tau := (optional) step size 
    - L := (optional) expected number of periods, if known
  The parameter n and d must be supplied, along with exactly one of 'w', 'tau' or 'L'.  
  '''
  assert isinstance(f, Callable) or isinstance(f, np.ndarray), "Time series must be function or array like"
  # Assume function like, defined on [0, 1]
  if isinstance(f, Callable): # Construct a continuous interpolation via e.g. cubic spline
    def sw(n: int, d: int = None, tau: float = None, w: float = None, L: int = None):
      ''' Creates a slidding window point cloud over 'n' windows '''
      d, tau = sw_parameters(bounds, d=d, tau=tau, w=w, L=L)
      T = np.linspace(bounds[0], bounds[1] - d*tau, n)
      delay_coord = lambda t: np.array([f(t + di*tau) for di in range(d+1)])
      X = np.array([delay_coord(t) for t in T])
      return(X)
    return(sw)
  else:
    d, tau = sw_parameters(bounds, d=d, tau=tau, w=w, L=L)
    T = np.linspace(bounds[0], bounds[1] - d*tau, n).astype(int)
    delay_coord = lambda t: np.array([f[int(t + di*tau)] for di in range(d+1)])
  return(sw)

## TODO: generalize beyond lower stars! 
# TODO: change E to be iterable, add flag to accept pre-sorted edge iterables
def ph0_lower_star(fv: ArrayLike, E: ArrayLike, collapse: bool = True, lex_sort: bool = True, max_death: Any = ["inf", "max"]) -> ArrayLike:
  """
  Computes the 0-dim persistence diagram of a lower-star filtration. 

  Parameters: 
    fv := 1d np.array of vertex function values, i.e. f(v_i) = fv[i] (index ordered)
    E := (m x 2) np.array of edges in the underlying complex
    collapse := whether to collapse 0-persistence pairs. Defaults to True. 
    lex_sort := whether to return the pairs in lexicographical birth-death order. Default to True.

  Return: 
    barcodes 
  """
  from scipy.cluster.hierarchy import DisjointSet
  if isinstance(max_death, list) and max_death == ["inf", "max"]:
    max_death = np.inf
  elif max_death == "max":
    max_death = max(fv)
  else:
    assert isinstance(max_death, float)

  ## Data structures + variables
  nv = len(fv)
  ds = DisjointSet(range(nv)) # Set representatives are minimum by *index*
  elders = np.fromiter(range(nv), dtype=int) # Elder map: needed to maintain minimum set representatives
  paired = np.zeros(nv, dtype=bool)
  insert_rule = lambda a,b: not(np.isclose(a,b)) if collapse else True # when to insert a pair
  
  ## Compute lower-star edge values; prepare to traverse in order
  ne = E.shape[0]
  ei = np.fromiter(sorted(range(ne), key=lambda i: (max(fv[E[i,:]]), min(fv[E[i,:]]))), dtype=int)
  
  ## Proceed to union components via elder rule
  dgm = []
  for (i,j), f in zip(E[ei,:], fv[E[ei,:]].max(axis=1)):
    ## The 'elder' was born first before the child: has smaller function value
    elder, child = (i, j) if fv[i] <= fv[j] else (j, i)
    if not ds.connected(i,j):
      if not paired[child]: # child unpaired => merged instantly by (i,j)
        dgm += [(fv[child], f)] if insert_rule(fv[child], f) else []
        paired[child] = True # kill the child
      else: # child already paired in component, use elder rule (keep elder alive)
        creator = elders[ds[elder]] if fv[elders[ds[child]]] <= fv[elders[ds[elder]]] else elders[ds[child]]
        dgm += [(fv[creator], f)] if insert_rule(fv[creator], f) else []
        paired[creator] = True
      
    ## Merge (i,j) and update elder map
    elder_i, elder_j = elders[ds[i]], elders[ds[j]]
    ds.merge(i,j)
    elders[ds[i]] = elders[ds[j]] = elder_i if fv[elder_i] <= fv[elder_j] else elder_j

  ## Post-processing 
  eb = fv[np.flatnonzero(paired == False)]  ## essential births
  ep = list(zip(eb, [max_death]*len(eb)))   ## essential pairs 
  dgm = np.array(dgm + ep)                  ## append essential pairs

  ## If warranted, lexicographically sort output
  return dgm if not(lex_sort) else dgm[np.lexsort(np.rot90(dgm)),:]

## For reference: TODO replace with generic function that call engine
def lower_star_ph_dionysus(f: ArrayLike, E: ArrayLike, T: ArrayLike):
  """
  Compute the p=(0,1) persistence diagrams of the lower-star filtration with vertex values 'f' and triangles 'T'
  """
  import dionysus as d
  #n = len(f)
  vertices = [([i], w) for i,w in enumerate(f)]
  edges = [(list(e), np.max(f[e])) for e in E]
  triangles = [(list(t), np.max(f[t])) for t in T]
  F = []
  for v in vertices: F.append(v)
  for e in edges: F.append(e) 
  for t in triangles: F.append(t)

  
  filtration = d.Filtration()
  for v_ind, time in F: filtration.append(d.Simplex(v_ind, time))
  filtration.sort()
  m = d.homology_persistence(filtration)
  dgms = d.init_diagrams(m, filtration)
  DGM0 = np.array([[pt.birth, pt.death] for pt in dgms[0]])
  DGM1 = np.array([[pt.birth, pt.death] for pt in dgms[1]])
  return([DGM0, DGM1])

def ph_dionysus(K: FiltrationLike):
  import dionysus as d
  F = d.Filtration([d.Simplex(s,f) for f,s in K])
  m = d.homology_persistence(F)
  dgms = d.init_diagrams(m, F)
  bd_dtype = [('birth', 'f4'), ('death', 'f4')]
  dgms = { i : np.fromiter(((pt.birth,pt.death) for pt in dgm), dtype=bd_dtype) for i,dgm in enumerate(dgms) }
  return dgms