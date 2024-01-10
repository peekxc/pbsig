import numpy as np
import splex as sx
from typing import *
from itertools import * 
from more_itertools import * 
from numpy.typing import ArrayLike
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.sparse import issparse, sparray, coo_array, diags
from scipy.sparse.linalg import LinearOperator, aslinearoperator
import _laplacian as laplacian


def deflate_sparse(A: sparray):
  """'Deflates' a given sparse array 'A' by removing it's rows and columns that are all zero. 
  
  Returns a new sparse matrix with the same data but potentially diffferent shape. 
  """
  from hirola import HashTable
  A = A if (hasattr(A, "row") and hasattr(A, "col") and hasattr(A, "data")) else A.tocoo()
  h = HashTable(1.1*len(A.data), np.int32)
  h.add(A.row)
  h.add(A.col)
  return coo_array((A.data, (h[A.row], h[A.col])), shape=(h.length, h.length))


def adjacency_matrix(S: sx.ComplexLike, p: int = 0, weights: ArrayLike = None):
  assert len(S) > p, "Empty simplicial complex"
  if sx.dim(S) <= (p+1):
    return coo_array((sx.card(S,p), sx.card(S,p)), dtype=int)
  weights = np.ones(sx.card(S,p+1)) if weights is None else weights
  assert len(weights) == sx.card(S,p+1), "Invalid weight array, must match length of p+1 simplices"
  V = list(sx.faces(S, p))
  ind = []
  for s in sx.faces(S, p+1):
    VI = [V.index(f) for f in sx.faces(s, p)]
    for i,j in combinations(VI, 2):
      ind.append((i,j))
  IJ = np.array(ind)
  A = coo_array((weights, (IJ[:,0], IJ[:,1])), shape=(sx.card(S,p), sx.card(S,p)))
  A = A + A.T
  return A

def knn(X: np.ndarray, k: int, method: str = 'auto', sort: bool = True, **kwargs):
  if method == 'kdtree' or method == 'auto' and X.shape[1] <= 16: 
    X_kdtree = KDTree(X)
    return X_kdtree.query(X, k=k)
  else: 
    knn_dist = squareform(pdist(X, **kwargs))
    knn_indices = knn_dist.argpartition(kth=k, axis=1)[:,:k] if not sort else knn_dist.argsort(axis=1)[:,:k]
    knn_dist = np.array([knn_dist[cc,ind] for cc, ind in enumerate(knn_indices)])
    return knn_dist, knn_indices

def neighborhood_graph(X: np.ndarray, k: int, weighted: bool = False, **kwargs) -> sparray:
  """Returns the (possibly weighted) adjacency matrix representing the k-nearest neighborhood graph of X."""
  from scipy.sparse import coo_array
  from array import array
  knn_dist, knn_ind = knn(X, k+1)
  nv = X.shape[0]
  I, J = np.repeat(np.arange(len(X)), k), np.ravel(knn_ind[:,1:])
  if weighted: 
    A = coo_array((np.ravel(knn_dist[:,1:]), (I, J)), shape=(nv,nv), dtype=float)
    return A 
  else:
    A = coo_array((np.repeat(True, len(I)), (I, J)), shape=(nv,nv), dtype=bool)
    return A

## TODO: change weight to optionally be a string when attr system added to SC's
def up_laplacian(S:sx.ComplexLike, p: int = 0, weight: Union[Callable, ArrayLike] = None, normed=False, return_diag=False, form='array', symmetric: bool = True, isometric: bool = True, dtype=None, **kwargs):
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
      - (p>0, weight=...)                   <=> weighted combinatorial up p-laplacian
      - (p>0, weight=..., normed=True)      <=> normalized weighted combinatorial up p-laplacian 
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
      isometric = 
      dtype := dtype of associated laplacian. 
      kwargs := unused. 
    
    The argument names for this function is loosely based on SciPy 'laplacian' interface in the sparse.csgraph module. 
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.laplacian.html. However, unlike the 
    LinearOperator returned by csgraph.laplacian, the operator returned here is always matrix-free.  
    (see https://github.com/scipy/scipy/blob/main/scipy/sparse/csgraph/_laplacian.py)
    """
    # assert isinstance(K,sx.ComplexLike), "K must be a Simplicial Complex for now"
    if isinstance(weight, Callable): 
      weights = np.array(weight(S), dtype=dtype)
    elif isinstance(weight, Iterable):
      weights = np.fromiter(iter(weight), dtype=dtype)
    elif weight is None: 
      weights = np.ones(len(S))
    else: 
      raise ValueError(f"Invalid weighting '{type(weight)}' given")
    
    ## Get the dimensions and the weights
    f_dim = np.array([sx.dim(s) for s in sx.faces(S)], dtype=np.uint16)
    wq, wpl, wpr = weights[f_dim == (p+1)], weights[f_dim == p], weights[f_dim == p]
    scale_ip = (lambda x: np.reciprocal(x, where=~np.isclose(x, 0))) if isometric else lambda x: x
    assert len(wpl) == sx.card(S,p) and len(wq) == sx.card(S,p+1), "Invalid weight arrays given."
    assert all(wq >= 0.0) and all(wpl >= 0.0), "Weight function must be non-negative"
    
    ## Form the operator
    if form == 'array':
      B = sx.boundary_matrix(S, p = p+1)
      L = B @ diags(wq) @ B.T
      if normed: 
        deg = (diags(np.sign(wpl)) @ L @ diags(np.sign(wpr))).diagonal() ## retain correct nullspace
        L = (diags(np.sqrt(scale_ip(deg))) @ L @ diags(np.sqrt(scale_ip(deg)))).tocoo()
      else:
        L = (diags(np.sqrt(scale_ip(wpl))) @ L @ diags(np.sqrt(scale_ip(wpr)))).tocoo()
      return (L, L.diagonal()) if return_diag else L
    elif form == 'lo':
      # p_faces = list(sx.faces(S, p))        ## need to make a view 
      p_simplices = list(sx.faces(S, p+1))  ## need to make a view 
      _lap_cls = eval(f"UpLaplacian{int(p)}D")
      lo = _lap_cls(p_simplices, sx.card(S, 0), sx.card(S, p))
      if normed:
        if not(symmetric):
          import warnings
          warnings.warn("symmetric = False is not a valid option when normed = True")
        lo.set_weights(np.sign(wpl), wq, np.sign(wpr))
        deg = np.array(lo.degrees)
        lo.set_weights(scale_ip(np.sqrt(deg)), wq, scale_ip(np.sqrt(deg))) # normalized weighted symmetric psd version 
      else: 
        if symmetric:
          lo.set_weights(scale_ip(np.sqrt(wpl)), wq, scale_ip(np.sqrt(wpr)))  # weighted symmetric psd version 
        else:
          lo.set_weights(scale_ip(wpl), wq, None)                           # asymmetric version
      return lo
      # f = _up_laplacian_matvec_p(p_simplices, p_faces, w0, w1, p, "default")
      # lo = f if form == 'function' else LinearOperator(shape=(ns[p],ns[p]), matvec=f, dtype=np.dtype(float))
    elif form == 'function':
      raise NotImplementedError("function form not implemented yet")
    else: 
      raise ValueError(f"Unknown form '{form}'.")

# elif isinstance(weight, Iterable):
#   assert len(weight) == sx.card(S, p+1) or len(weight) == len(S), "If weights given, they must match size of complex or p-faces"
#   weight = np.asarray(weight)
#   if len(weight) == sx.card(S, p+1):
#     wq, wpl, wpr = weight, np.ones(sx.card(S, p)), np.ones(sx.card(S, p))
#   else:
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

  def __init__(S: Iterable['SimplexLike'], n: int, dtype = None):
    from more_itertools import peekable
    S_peek = peekable(S)
    p: int = len(S_peek.peek())-1
    assert p == 0 or p == 1 or p == 2, "Only p in {0,1} supported for now"

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
  def __init__(self, S: Iterable['SimplexLike'], nv: int, _np: int = 0):
    UpLaplacianBase.__init__(S, nv) ## does error checking on S 
    S = np.fromiter(collapse(S), dtype=np.uint16)
    laplacian.UpLaplacian0D.__init__(self, S, nv, _np)
    self.precompute_degree()

class UpLaplacian0F(laplacian.UpLaplacian0F, UpLaplacianBase):
  def __init__(self, S: Iterable['SimplexLike'], nv: int, _np: int = 0):
    S = np.fromiter(collapse(S), dtype=np.uint16)
    laplacian.UpLaplacian0F.__init__(self, S, nv, _np)
    self.precompute_degree()

class UpLaplacian1D(laplacian.UpLaplacian1D, UpLaplacianBase):
  def __init__(self, S: Iterable['SimplexLike'], nv: int, _np: int = 0):
    UpLaplacianBase.__init__(S, nv)
    S = np.fromiter(collapse(S), dtype=np.uint16)
    laplacian.UpLaplacian1D.__init__(self, S, nv, _np)
    self.precompute_degree()

class UpLaplacian1F(laplacian.UpLaplacian1F, UpLaplacianBase):
  def __init__(self, S: Iterable['SimplexLike'], nv: int, _np: int = 0):
    UpLaplacianBase.__init__(S, nv)
    S = np.fromiter(collapse(S), dtype=np.uint16)
    laplacian.UpLaplacian1F.__init__(self, S, nv, _np)
    self.precompute_degree()
    
class UpLaplacian2D(laplacian.UpLaplacian2D, UpLaplacianBase):
  def __init__(self, S: Iterable['SimplexLike'], nv: int, _np: int = 0):
    UpLaplacianBase.__init__(S, nv)
    S = np.fromiter(collapse(S), dtype=np.uint16)
    laplacian.UpLaplacian2D.__init__(self, S, nv, _np)
    self.precompute_degree()
    
class UpLaplacian2F(laplacian.UpLaplacian2F, UpLaplacianBase):
  def __init__(self, S: Iterable['SimplexLike'], nv: int, _np: int = 0):
    UpLaplacianBase.__init__(S, nv)
    S = np.fromiter(collapse(S), dtype=np.uint16)
    laplacian.UpLaplacian2F.__init__(self, S, nv, _np)
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

def param_laplacian(
  L: Union[sparray, LinearOperator],
  p_weights: np.ndarray, 
  q_weights: np.ndarray, 
  normed: bool = False, 
  boundary: bool = False, 
  isometric: bool = True
):
  """Parameterizes the given boundary or Laplacian operator/matrix 'L' with weights on the (p, p+1)-simplices in-place.
  
  Returns: 
    Weighted up-Laplacian of the same type passed in
  """
  is_linear_op = isinstance(L, LinearOperator)
  p_func = (lambda x: np.reciprocal(x.astype(float), where=~np.isclose(x, 0.0))) if isometric else lambda x: x
  if is_linear_op and not boundary:
    assert hasattr(L, "face_left_weights") and hasattr(L, "face_right_weights"), "Must be a Laplacian instance"
    assert len(p_weights) == len(L.p_faces) and len(q_weights) == len(L.q_faces), "Invalid weights given. Must match length of faces."
    I = np.where(np.isclose(p_weights, 0.0), 0.0, 1.0)
    if normed: 
      L.set_weights(I,q_weights,I)
      d = np.sqrt(p_func(L.degrees))
      L.face_right_weights = d
      L.face_left_weights = d
      L.precompute_degree()
    else:
      p_inv_sqrt = np.sqrt(p_func(p_weights))
      L.set_weights(p_inv_sqrt, q_weights, p_inv_sqrt)
  elif is_linear_op and boundary:
    raise NotImplementedError("Haven't implemented re-weighted boudnary operators yet")
  else:
    assert isinstance(L, np.ndarray) or issparse(L), "If given explicit matrix, it must be numpy array or Scipy sparse matrix."
    if boundary: 
      B = L
      if (B.shape[0] == 0):
        return B @ B.T 
      L = B @ diags(q_weights) @ B.T
      if normed: 
        I = np.where(np.isclose(p_weights, 0.0), 0.0, 1.0)## retain correct nullspace
        # deg = (diags(np.sign(p_weights)) @ L @ diags(np.sign(p_weights))).diagonal() ## retain correct nullspace
        deg = (diags(I) @ L @ diags(I)).diagonal() # degrees 
        L = (diags(np.sqrt(p_func(deg))) @ L @ diags(np.sqrt(p_func(deg)))).tocoo()
      else:
        L = (diags(np.sqrt(p_func(p_weights))) @ L @ diags(np.sqrt(p_func(p_weights)))).tocoo()
    else:
      raise NotImplementedError("Haven't implemented re-weighted sparse Laplacian matrices yet")
    return L

class WeightedLaplacian:
  """Represents a combinatorial p-laplacian that can be easily re-weighted."""
  def __init__(self, S: sx.ComplexLike, p: int = 0, dtype: Any = None, **kwargs):
    # self.p_faces = np.array(list(sx.faces(S, p)))
    # self.q_faces = np.array(list(sx.faces(S, p+1))) ## to remove
    self.p = p
    # self.complex = S.copy() if hasattr(S, "copy") else S
    self.complex = S
    self.dtype = np.dtype('float32') if dtype is None else dtype
    self.shape = (sx.card(S, p), sx.card(S, p))
    print("normed: " + str(kwargs.get("normed", False)))
    self.normed = kwargs.get("normed", False)
    self.isometric = kwargs.get("isometric", True)
    self.sign_width = kwargs.get("sign_width", 0.0)
    self.form = kwargs.get("form", "array") # do this last
    self.reweight() # apply weights as nececsary
    
  @property
  def form(self) -> str:
    return self._form

  @form.setter
  def form(self, x: str) -> None:
    self._form = x
    if x == "array":
      self.bm = sx.boundary_matrix(self.complex, p = self.p+1)
      self.reweight()
      # self.laplacian = up_laplacian(self.complex, p = self.p, form = "array")
      self.__dict__.pop('op', None)
    elif x == "lo":
      self.op = up_laplacian(self.complex, p = self.p, form = "lo", isometric=self.isometric)
      self.__dict__.pop('laplacian', None)
      self.__dict__.pop('bm', None)
    else:
      raise ValueError(f"Invalid given form '{x}'")

  def _matmat(self, X: np.ndarray) -> np.ndarray:
    Y = self.laplacian @ X if self.form == "array" else self.op @ X
    return Y.astype(self.dtype)

  def operator(self, deflate: bool = False):
    if self.form == "array": 
      return deflate_sparse(self.laplacian) if deflate else self.laplacian
    else: 
      return self.op 
    # return self.laplacian if self.form == "array" else self.op

  # Scipy doesn't like this 
  def _matvec(self, x: np.ndarray) -> np.ndarray:
    y = self.laplacian @ x if self.form == "array" else self.op @ x
    return y.astype(self.dtype).copy()

  def sign(self, x: np.ndarray):
    """Smoothed version of the non-negative sign function. """
    from pbsig.linalg import smooth_upstep
    return smooth_upstep(lb=0, ub=self.sign_width)(x)

  def reweight(self, q_weights: ArrayLike = None, p_weights: ArrayLike = None):
    """Sets the laplacian attribute appropriately """
    p_weights = np.ones(sx.card(self.complex, self.p)) if p_weights is None else p_weights
    q_weights = np.ones(sx.card(self.complex, self.p+1)) if q_weights is None else q_weights
    assert len(p_weights) == sx.card(self.complex, self.p) and len(q_weights) == sx.card(self.complex, self.p+1), "Invalid weights given. Must match length of faces."
    assert np.all(p_weights >= 0) and np.all(q_weights >= 0), "Invalid weights supplied; must be all non-negative."
    assert self.form in ["array", "lo"]
    scale_ip = (lambda x: np.reciprocal(x, where=~np.isclose(x, 0))) if self.isometric else lambda x: x
    if self.form == "array":
      L = self.bm @ diags(q_weights) @ self.bm.T
      if self.normed: 
        P_diag = diags(np.squeeze(np.ravel(self.sign(p_weights))))
        deg = (P_diag @ L @ P_diag).diagonal() ## retain correct nullspace
        P_diag_scaled = diags(np.squeeze(np.ravel(np.sqrt(scale_ip(deg)))))
        self.laplacian = (P_diag_scaled @ L @ P_diag_scaled).tocoo()
      else:
        P_diag = diags(np.squeeze(np.ravel(np.sqrt(scale_ip(p_weights)))))
        self.laplacian = (P_diag @ L @ P_diag).tocoo()
    else:
      assert hasattr(self, "op"), "No operator attribute set"
      # I = np.where(np.isclose(p_weights, 0.0), 0.0, 1.0)
      I = np.squeeze(np.ravel(self.sign(p_weights)))
      if self.normed: 
        print("normed")
        self.op.set_weights(I, q_weights, I)
        d = np.sqrt(scale_ip(self.op.degrees))
        self.op.set_weights(d, q_weights, d)
      else:
        print("not normed")
        p_inv_sqrt = np.sqrt(scale_ip(p_weights))
        self.op.set_weights(p_inv_sqrt, q_weights, p_inv_sqrt)
    return self

  # def __call__(self, t: Union[float, int], **kwargs):
  #   assert t >= self.domain_[0] and t <= self.domain_[1], f"Invalid time point 't'; must in the domain [{self.domain_[0]},{self.domain_[1]}]"
  #   # assert hasattr(self, "domain_") and hasattr(self, "q_splines_"), "Cannot interpolate without calling interpolat fmaily first! "
  #   assert isinstance(self.family, Callable), "Parameterized family must be callable!"
  #   # wp = np.array([pf(t) for pf in self.p_splines_])
  #   # wq = np.array([qf(t) for qf in self.q_splines_])
  #   wp = np.array(self.family(t, p=0))
  #   wq = np.array(self.family(t, p=1))
  #   self.param_weights(wq, wp, **kwargs)
  #   return self.laplacian
  
  # def __iter__(self) -> Generator:
  #   """Iterates through the family, yielding the parameterized operators"""
  #   for f in self.family:
  #     wp, wq = self.post_p(f(self.p_faces)), self.post_q(f(self.q_faces))
  #     self.param_weights(wq, wp)
  #     yield self.laplacian
  # def interpolate_family(self, interval=(0.0, 1.0)) -> None: 
  #   """Interpolates the stored 1-parameter family of filtration values."""
  #   filter_values = [[f(self.p_faces), f(self.q_faces)] for f in self.family]
  #   n_parameters = len(filter_values)
  #   p_filter_values = np.array([self.post_p(f[0]) for f in filter_values], dtype=np.float32)
  #   q_filter_values = np.array([self.post_q(f[1]) for f in filter_values], dtype=np.float32)
  #   del filter_values

  #   ## Interpolate the filter path of each simplex (simplexwise) with a polynomial curve
  #   self.domain_ = interval
  #   domain_points = np.linspace(*interval, num=n_parameters) # knot vector
  #   self.p_splines_ = [CubicSpline(domain_points, p_fv) for p_fv in p_filter_values.T]
  #   self.q_splines_ = [CubicSpline(domain_points, q_fv) for q_fv in q_filter_values.T]