import numpy as np 
from numbers import Number
from typing import *
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import structural_rank
from .persistence import * 
from .apparent_pairs import *
from .linalg import *
from .utility import progressbar, smooth_upstep, smooth_dnstep
from splex.geometry import flag_filter
from itertools import *
from more_itertools import spy 
import copy
from array import array
# from tqdm import tqdm
# from tqdm.notebook import tqdm

def intervals_intersect(i1: tuple, i2: tuple) -> bool: 
  """Detects whether two intervals of the form (x1,x2) intersect."""
  i1, i2 = np.sort(i1), np.sort(i2)
  return not(i1[1] < i2[0] or i2[1] < i1[0])

def rect_intersect(r1,r2) -> bool:
  """Detects whether two rectangles of the form (x1,x2,y1,y2) intersect."""
  return intervals_intersect(r1[[0,1]], r2[[0,1]]) and intervals_intersect(r1[[2,3]], r2[[2,3]])

## Generate a random set of rectangles in the upper half plane 
def sample_rect_halfplane(n: int, lb: float = 0.0, ub: float = 1.0, area: tuple = (0.01, 0.05), max_asp = 5, disjoint: bool = False):  
  """Generate random rectangles with min/max _area_ and maximum aspect ratio _max asp_ in upper-half plane via rejection sampling.

  Parameters: 
    n: number of rectangles to generate 
    lb: lower bound on the sampling area (x and y)
    ub: upper bound on the sampling area (x and y)
    area: relative min/max proportiate area each rectangle should have relative to the total area.
    max_asp: maximum aspect ratio allowed for any given rectangle. Supplying 1.0 is equivalent to sampling squares. 
    disjoint: whether to require the rectangles to be disjoint. 

  Returns: 
    rectangles: a np.array of rectangles [a,b]x[c,d] given as rows (a,b,c,d) 
  """
  assert max_asp >= 1.0, "Aspect ratio must be >= 1.0"
  cc, n_tries = 0, 0
  def r_sample(max_asp):
    if max_asp == 1.0:
      i,j,k = np.sort(np.random.uniform(size=3, low=lb, high=ub)) 
      l = k + (j-i)
      return i,j,k,l
    else:
      return np.sort(np.random.uniform(size=4, low=lb, high=ub))
  R = []
  area_min = area[0] * abs(ub-lb)**2
  area_max = area[1] * abs(ub-lb)**2
  while cc < n:
    r = r_sample(max_asp)
    dx, dy = abs(r[1]-r[0]), abs(r[3]-r[2])
    asp, ra = max(dx/dy, dy/dx), dx*dy
    area_check = ra >= area_min and ra <= area_max and asp <= (1.0 + 1e-6)*max_asp
    if area_check and (not(disjoint) or not(any([rect_intersect(r, r_) for r_ in R]))):
      R.append(r)
    else: 
      n_tries += 1
      continue
    cc += 1
  return np.array(R)

def lipshitz_constant(f: ArrayLike, x: ArrayLike):
  """ 
  Estimate Lipshitz constant K such that: 
  
  (|f(x) - f(x')|) / (|x - x'|) <= K 
  
  """
  assert len(f) == len(x)
  return(np.max(np.abs(np.diff(f))/np.abs(np.diff(x))))

def make_smooth_step(b, bp):
  def S(x):
    if x <= b: return(1.0)
    if x >= bp: return(0.0)
    n = (x-b)/(bp-b)
    return(0.5*n**3 - 1.5*n**2 + 1)
  return(S)

def smooth_grad(b, bp):
  def sg(x):
    if x <= b: return(0)
    if x >= bp: return(0)
    return(-(3*(b-bp)*(bp-x))/(bp-b)**3)
  return(sg)

# if smoothing is None: # use nuclear norm
#   sig += self._Terms[0].data.sum(axis=1)
#   sig -= self._Terms[1].data.sum(axis=1)
#   sig -= self._Terms[2].data.sum(axis=1)
#   sig += self._Terms[3].data.sum(axis=1)
# elif isinstance(smoothing, tuple):
#   eps,p,method = smoothing
#   sv_op = sgn_approx(eps=eps, p=p, method=method)
#   sig += elementwise_row_sum(self._Terms[0].data, sv_op)
#   sig -= elementwise_row_sum(self._Terms[1].data, sv_op)
#   sig -= elementwise_row_sum(self._Terms[2].data, sv_op)
#   sig += elementwise_row_sum(self._Terms[3].data, sv_op)
# elif isinstance(smoothing, str):
#   if smoothing == "huber":asda
#     raise NotImplemented("Haven't done hinge loss yet")
#   elif smoothing == "soft-thresholding":
#     soft_threshold = lambda s: np.sign(s) * np.maximum(np.abs(s) - t, 0)
#     sig += elementwise_row_sum(self._Terms[0].data, soft_threshold)
#     sig -= elementwise_row_sum(self._Terms[1].data, soft_threshold)
#     sig -= elementwise_row_sum(self._Terms[2].data, soft_threshold)
#     sig += elementwise_row_sum(self._Terms[3].data, soft_threshold)
# else:
#   raise ValueError("Invalid")

def prox_moreau(X: Union[ArrayLike, Tuple], alpha: float = 1.0, mu: Union[float, ArrayLike] = 0.0):
  """
  Evaluates the proximal operator of a matrix (or SVD) 'X' under the nuclear norm scaled by 'alpha', and 'mu' is a smoothing parameter. 
  
  As mu -> 0, Mf(X) converges to the nuclear norm of X. 
  
  If mu < 1, the term |X - Y| tends to be larger and so the optimization will tend prefer matrices with lower nuclear norms

  If mu > 1, the term |X - Y| tends to be smaller and so the optimization will tend to prefer 'closer' matrices in the neighborhood of X that happen to have small nuclear norm. 

  As mu -> C for some very large constant C, the gradient of should become smoother as the expense of |f(x) - Mf(x)| becoming larger.
  """
  ## alpha-scaled prox operator 
  if isinstance(X, np.ndarray):
    usv = np.linalg.svd(X, compute_uv=True, full_matrices=False)
  else: 
    assert isinstance(X, Tuple)
    usv = X
    X = usv[0] @ np.diag(usv[1]) @ usv[2]
  # s_threshold = soft_threshold((1.0/np.sqrt(alpha))*usv[1], mu=mu)
  s_threshold = soft_threshold(usv[1], mu=alpha*mu)
  prox_af = alpha * (usv[0] @ np.diag(s_threshold) @ usv[2])

  ## Moreau envelope 
  Ma = alpha**2 * np.sum(s_threshold) # to check
  Mb = 0 if mu == 0 else (1/(2*mu))*np.linalg.norm(X - prox_af, 'fro')**2
  return(Ma + Mb, prox_af)

# def plot_direction(V: ArrayLike, T: ArrayLike, W: ArrayLike, cmap: str = 'jet'):
#   import matplotlib
#   assert V.shape[0] == len(W)
#   import plotly.graph_objects as go
#   from tallem.color import bin_color
#   cm = matplotlib.cm.get_cmap(cmap)
#   colors = list(np.array([cm(v) for v in np.linspace(0, 1, endpoint=True)]))
#   TW = np.array([np.max(W[t]) for t in T])
#   face_colors = bin_color(TW, colors)
#   axis = dict(showbackground=True,backgroundcolor="rgb(230, 230,230)",gridcolor="rgb(255, 255, 255)",zerolinecolor="rgb(255, 255, 255)")
#   layout = go.Layout(scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis), aspectmode='data', aspectratio=dict(x=1, y=1, z=1)))
#   mesh = go.Mesh3d(x=V[:,0], y=V[:,1], z=V[:,2], i=T[:,0],j=T[:,1],k=T[:,2],facecolor=face_colors) #  intensity=W+np.min(W), colorscale='Jet'
#   tri_points = V[T]
#   Xe, Ye, Ze = [], [], []
#   for T in tri_points:
#     Xe.extend([T[k%3][0] for k in range(4)]+[ None])
#     Ye.extend([T[k%3][1] for k in range(4)]+[ None])
#     Ze.extend([T[k%3][2] for k in range(4)]+[ None])
#   lines = go.Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines', name='', line=dict(color= 'rgb(70,70,70)', width=1))  
#   fig = go.Figure(data=[mesh, lines], layout=layout)
#   fig.show()

def lower_star_boundary(weights: ArrayLike, threshold: Optional[float] = np.inf, simplices: Optional[ArrayLike] = None, dim: Optional[int] = 1):
  """
  Given vertex weights and either 1. a threshold + dimension or 2. a (m x k) set of simplices, returns a tuple (D, sw) where:
    D := an (l x m) boundary matrix of the given simplices, in lex order
    sw := simplex weights, given by the maximal weight of its lower stars
  If (1) is supplied, all 'dim'-dimensional simplices are computed. 
  """
  nv = len(weights)
  if not(simplices is None):
    assert isinstance(simplices, np.ndarray) # TODO: accept tuple (E, T) and use as overload
    d = simplices.shape[1]
    assert d == 2 or d == 3
    simplices.sort(axis=1)
    S = simplices[np.lexsort(np.rot90(simplices))]  ## ensure sorted lex order
    SR = np.sort(rank_combs(S, k=d, n=nv))
    if d == 2:
      cdata, cindices, cindptr, ew = boundary.lower_star_boundary_1_sparse(weights, SR)
      D1 = csc_matrix((cdata, cindices, cindptr), shape=(len(weights), len(cindptr)-1))
      return(D1, ew)
    else:
      ER = np.sort(rank_combs(edges_from_triangles(S, nv), k=2, n=nv))
      cdata, cindices, cindptr, tw = boundary.lower_star_boundary_2_sparse(weights, ER, SR)
      D2 = csc_matrix((cdata, cindices, cindptr), shape=(len(ER), len(cindptr)-1))
      return(D2, tw)
  else:
    assert (dim == 1 or dim == 2) and isinstance(threshold, float)
    from math import comb
    cdata, cindices, cindptr, sw = boundary.lower_star_boundary_1(weights, threshold) if dim == 1 else boundary.lower_star_boundary_2(weights, threshold)
    D = csc_matrix((cdata, cindices, cindptr), shape=(int(comb(len(weights), dim)), len(cindptr)-1))
    return(D, sw)


def _pb_relaxations(D0, D1, D2, H, type, terms, **kwargs):
  if isinstance(type, list) or type == "norm_nuc" or type == "nuc":
    from scipy.sparse.linalg import aslinearoperator, svds
    type = "norm_nuc" if isinstance(type, list) else type
    d0_sv = D0.diagonal()
    d1_sv = svds(D1, k=np.min(D1.shape)-1, return_singular_vectors=False)
    d2_sv = svds(D2, k=np.min(D2.shape)-1, return_singular_vectors=False)
    dh_sv = svds(H, k=np.min(H.shape)-1, return_singular_vectors=False)
    if type == "norm_nuc":
      d0_t = np.sum(D0.diagonal()/np.max(d0_sv))
      d1_t = np.sum(d1_sv/np.max(d1_sv))
      d2_t = np.sum(d2_sv/np.max(d2_sv))
      dh_t = np.sum(dh_sv/np.max(dh_sv))
    else: 
      d0_t = np.sum(D0.diagonal())
      d1_t = np.sum(d1_sv)
      d2_t = np.sum(d2_sv)
      dh_t = np.sum(dh_sv)
  elif type == 'fro': 
    d0_t = np.sqrt(np.sum(D0.diagonal()**2))
    d1_t = np.sqrt(np.sum(D1.data**2))
    d2_t = np.sqrt(np.sum(D2.data**2))
    dh_t = np.sqrt(np.sum(H.data**2))
  elif type=="approx": 
    from sksparse.cholmod import analyze, cholesky, analyze_AAt, cholesky_AAt
    D1_m, D2_m, H_m = D1 @ D1.T, D2 @ D2.T, H @ H.T
    d1_factor = cholesky(D1_m, beta=kwargs['beta'])
    d2_factor = cholesky(D2_m, beta=kwargs['beta'])
    h_factor = cholesky(H_m, beta=kwargs['beta'])
    d0_t = np.sum((D0.diagonal()**2)/(D0.diagonal()**2 + kwargs['beta']))
    d1_t = np.sum([d1_factor(D1_m[:,j])[j][0,0] for j in range(D1_m.shape[1])])
    d2_t = np.sum([d2_factor(D2_m[:,j])[j][0,0] for j in range(D2_m.shape[1])])
    dh_t = np.sum([h_factor(H_m[:,j])[j][0,0] for j in range(H_m.shape[1])])
  elif type=="rank":
    from scipy.linalg.interpolative import estimate_rank
    from scipy.sparse.linalg import aslinearoperator, svds
    d0_t = np.sum(abs(D0.diagonal()) > 0.0)
    sval = lambda X: np.append(svds(X, k=np.min(X.shape)-1, return_singular_vectors=False, which='SM'), svds(X, k=1, return_singular_vectors=False, which='LM'))
    s1,s2,sh = sval(D1), sval(D2), sval(H)
    d1_t = np.sum(s1 > np.max(s1)*np.max(D1.shape)*np.finfo(D1.dtype).eps)
    d2_t = np.sum(s2 > np.max(s2)*np.max(D2.shape)*np.finfo(D2.dtype).eps)
    dh_t = np.sum(sh > np.max(sh)*np.max(H.shape)*np.finfo(H.dtype).eps)
  return(d0_t - d1_t - d2_t + dh_t if not(terms) else (d0_t, d1_t, d2_t, dh_t))

def tolerance(m: int, n: int, dtype: type = float):
  _machine_eps, _min_res = np.finfo(dtype).eps, np.finfo(dtype).resolution*100
  def _tol(spectral_radius):
    return np.max([_machine_eps, spectral_radius * np.max([m,n]) * _min_res])
  return _tol

from pbsig.csgraph import param_laplacian, WeightedLaplacian
def betti_query(
  S: Union[LinearOperator, ComplexLike], 
  f: Callable[SimplexConvertible, float],
  matrix_func: Callable[np.ndarray, float],
  i: Union[float, np.ndarray], 
  j: Union[float, np.ndarray], 
  p: int = 0, 
  w: float = 0.0, 
  terms: bool = False, 
  solver: Callable = None, 
  **kwargs
) -> Generator:
  # from splex.predicates import is_complex_like
  # L = S if isinstance(S, UpLaplacian) else up_laplacian(S, p=p, form='lo')
  # assert isinstance(L, UpLaplacian)
  # assert is_complex_like(S)
  # B0, B1 = boundary_matrix(S, p = p), boundary_matrix(S, p = p+1)
  # pseudo = lambda x: np.reciprocal(x, where=~np.isclose(x, 0, atol=atol)) # scalar pseudo-inverse 
  
  yw, fw, sw = f(faces(S, p-1)), f(faces(S, p)), f(faces(S, p+1))
  delta = np.finfo(float).eps 
  atol = kwargs['tol'] if 'tol' in kwargs else 1e-5     
  p_solver = PsdSolver(k = int(card(S, p-1)-1)) if solver is None else solver
  q_solver = PsdSolver(k = int(card(S, p)-1)) if solver is None else solver
  I, J = (np.array([i]), np.array([j])) if isinstance(i, Number) and isinstance(j, Number) else (i,j)
  inc_all = smooth_upstep(0, w)
  
  ## Get the Weighted Laplacians
  L_kwargs = dict(normed = True, isometric = False, sign_width = w, form="array") | kwargs
  Lp = WeightedLaplacian(S, p = p-1, **L_kwargs)
  Lq = WeightedLaplacian(S, p = p, **L_kwargs)

  for ii, jj in zip(I, J):
    assert ii <= jj, f"Invalid point ({ii:.2f}, {jj:.2f}): must be in the upper half-plane"
    fi_inc = smooth_dnstep(lb = ii-w, ub = ii+delta)
    fi_exc = smooth_upstep(lb = ii, ub = ii+w)         
    fj_inc = smooth_dnstep(lb = jj-w, ub = jj+delta)
    
    # T1 = param_laplacian(B0, p_weights=inc_all(yw), q_weights=fi_inc(fw), normed=True, boundary=True, **kwargs)
    # T2 = param_laplacian(B1, p_weights=inc_all(fw), q_weights=fj_inc(sw), normed=True, boundary=True, **kwargs)
    # T3 = param_laplacian(B1, p_weights=fi_exc(fw), q_weights=fj_inc(sw), normed=True, boundary=True, **kwargs)
    
    t0 = matrix_func(fi_inc(fw)) # instead of solver 
    
    Lp.reweight(fi_inc(fw), inc_all(yw))
    t1 = matrix_func(p_solver(Lp.operator()))

    Lq.reweight(fj_inc(sw), inc_all(fw)) # this one
    t2 = matrix_func(q_solver(Lq.operator()))
    
    Lq.reweight(fj_inc(sw), fi_exc(fw))
    t3 = matrix_func(q_solver(Lq.operator()))
    
    yield (t0, t1, t2, t3) if terms else t0 - t1 - t2 + t3

    # ii = 3.49905344
    # ii = 3.50068702

## Cone the complex
def cone_filter(f: Callable, vid: int = -1, v_birth: float = -np.inf, collapse_weight: float = np.inf):
  def _cone_weight(s):
    s = Simplex(s)
    if s == Simplex([vid]):
      return v_birth
    elif vid in s:
      return collapse_weight
    else: 
      return f(s)
  return _cone_weight

def cone_complex(S: ComplexLike, vid: int = -1):
  assert hasattr(S, "add"), f"simplicial complex '{type(S)}' must be mutable"
  for s in S: 
    S.add(Simplex([vid]) + Simplex(s))
  return S


## Changes the eigenvalues! 
def update_weights(S: ComplexLike, f: Callable[SimplexConvertible, float], p: int, R: ArrayLike, w: float = 0.0, biased: bool = True):
  """Updates the smooth step functions to reflect the scalar-product _f_ on _R_ smoothed by _w_"""
  if isinstance(f, Callable):
    fw = np.array([f(s) for s in faces(S, p)])
    sw = np.array([f(s) for s in faces(S, p+1)])
  elif isinstance(f, np.ndarray):
    d = np.array([dim(s) for s in S], dtype=np.uint16)
    fw = f[d == p]
    sw = f[d == (p+1)]
  else: 
    raise ValueError(f"Invalid form '{type(f)}' for f")
  assert len(R) == 4 and is_sorted(R)
  i,j,k,l = R 
  assert i < j and j <= k and k < l, f"Invalid rectangle ({i:.2f}, {j:.2f}, {k:.2f}, {l:.2f}): each rectangle must have positive measure"
  delta = np.finfo(float).eps # TODO: use bound on spectral norm to get tol instead of eps?        
  if biased:
    fi = smooth_upstep(lb = i, ub = i+w)(fw)          # STEP UP:   0 (i-w) -> 1 (i), includes (i, infty)
    fj = smooth_upstep(lb = j, ub = j+w)(fw)          # STEP UP:   0 (j-w) -> 1 (j), includes (j, infty)
    fk = smooth_dnstep(lb = k-w, ub = k+delta)(sw)    # STEP DOWN: 1 (k-w) -> 0 (k), includes (-infty, k]
    fl = smooth_dnstep(lb = l-w, ub = l+delta)(sw)    # STEP DOWN: 1 (l-w) -> 0 (l), includes (-infty, l]
  else:
    x = w/2
    fi = smooth_upstep(lb = i-x, ub = i+x)(fw)          # STEP UP:   0 (i-w) -> 1 (i), includes (i, infty)
    fj = smooth_upstep(lb = j-x, ub = j+x)(fw)          # STEP UP:   0 (j-w) -> 1 (j), includes (j, infty)
    fk = smooth_dnstep(lb = k-x, ub = k+x+delta)(sw)    # STEP DOWN: 1 (k-w) -> 0 (k), includes (-infty, k]
    fl = smooth_dnstep(lb = l-x, ub = l+x+delta)(sw)    # STEP DOWN: 1 (l-w) -> 0 (l), includes (-infty, l]
  return fi,fj,fk,fl

def _transform_mult_ew(EW: list[ArrayLike], smoothing: Callable = None, terms: bool = False, shape: tuple = None, **kwargs) -> Union[tuple, float, int]:
  n, m = (len(EW[0]), len(EW[0])) if shape is None else shape
  if smoothing is None: 
    mu = np.zeros(4)
    for cc,(ew,s) in enumerate(zip(EW, [1,-1,-1,1])):
      mu[cc] += s*spectral_rank(EW[cc], shape=(n,m), **kwargs)
    return mu.astype(int) if terms else int(sum(mu))
  else:
    if isinstance(smoothing, bool):
      smoothing = lambda x: x if smoothing else huber()
    assert isinstance(smoothing, Callable)
    mu = np.array([sum(smoothing(EW[0])), -sum(smoothing(EW[1])), -sum(smoothing(EW[2])), sum(smoothing(EW[3]))])
    return mu if terms else sum(mu)

def weighted_degree(S: ComplexLike, p: int, weights: ArrayLike = None) -> ArrayLike:
  """Computes the weighted degree of the p-simplices of _S_.

  The degree of a p-simplex _s_ is defined as the sum of the weights of its codimension-1 faces. 
  """
  if weights is None: 
    weights = np.ones(card(S,p+1))
  elif isinstance(weights, Iterable) and len(weights) == card(S,p+1):
    weights = np.fromiter(weights, dtype=np.float64)
  elif isinstance(weights, Iterable) and len(weights) == len(S):
    d = np.array([dim(s) for s in faces(S)])
    weights = np.fromiter(weights, dtype=np.float64)[d == p+1]
  else: 
    raise ValueError(f"Invalid weights array of type '{type(weights)}' given.")
  assert len(weights) == card(S,p+1), "Invalid weights array. Must match the length of the number of p+1 simplices of S."
  import hirola 
  h_sz = card(S,p)*1.25 if card(S,p) > 1 else 2
  h = hirola.HashTable(h_sz, dtype=np.uint64)
  h.add(np.asarray(rank_combs(faces(S,p), order='colex'), dtype=np.uint64))
  degrees = np.zeros(card(S,p), dtype=np.float64)
  for j, sigma in enumerate(faces(S,p+1)):
    for tau in boundary(sigma):
      i = h[np.uint64(rank_colex(tau))]
      degrees[i] += weights[j] # the definition of degree 
  return degrees

class MuQuery():
  def __init__(self, S: Union[FiltrationLike, ComplexLike], p: int, R: tuple, w: float = 0.0, smoothing=None):
    assert isinstance(S, ComplexLike) or isinstance(S, FiltrationLike), f"Invalid complex type '{type(S)}'"
    assert len(R) == 4 and is_sorted(R), f"Invalid rectangle"
    self.S: ComplexLike = S
    self.R: tuple = R
    self.w: float = w
    self.p: int = p
    self.biased: bool = True
    self.smoothing = smoothing
    self.terms: bool = False
    self.choose_solver()

  def choose_solver(self, solver='default', form = 'array', normed: bool = True, **kwargs):
    if form == "lo":
      self.form = "lo"
      self.normed = normed
      self.L = up_laplacian(self.S, p=self.p, normed=normed, form="lo", dtype=np.float64)
      self.D = None
      self.solver = eigvalsh_solver(self.L, solver=solver, **kwargs)
    elif form == "array": 
      self.form = "array"
      self.normed = normed
      self.D = boundary_matrix(self.S, p=self.p+1).astype(np.float64).tocsr()
      self.L = None
      if solver == "dac":
        self.D = self.D.todense()
      self.solver = eigvalsh_solver(self.D, solver=solver, **kwargs)
    else: 
      raise ValueError(f"Invalid form '{str(form)}'.")

  def _grad_scalar_product(self, f: Callable[float, Callable], **kwargs) -> Callable:
    import numdifftools as nd
    def _f(alpha: float):
      wf = f(alpha)
      return np.array([wf(s) for s in self.S])
    return nd.Derivative(_f, n=1, **kwargs)

  def _grad_laplacian(self): #f: Callable[float, Callable], t0: float): ## normed + array only
    import hirola
    L = self.D @ self.D.T
    I,J = L.nonzero()
    ht = hirola.HashTable(int(len(I)*1.25), dtype=(np.uint16, 2))
    ht.add(np.c_[I,J].astype(np.uint16))
    data_vec = L.data.copy()
    def _grad(f: ArrayLike, as_matrix: bool = False, term: str = "i"):
      """Accepts a scalar product _f_ over K and returns its gradient parameterizing _L_."""
      fi,fj,fk,fl = update_weights(self.S, f, self.p, self.R, self.w, self.biased)
      data_vec.fill(0)
      L_tmp = up_laplacian(self.S, p=self.p, weight=f, form="array")
      data_vec[ht[np.c_[L_tmp.nonzero()].astype(np.uint16)]] = L_tmp.data
      if as_matrix: 
        L.data = data_vec
        return L
      else:
        return data_vec
   
    # for cc, (I,J) in enumerate([(fj, fk), (fi, fk), (fj, fl), (fi, fl)]):
    #   I_sgn = np.sign(abs(I)) 
    #   L = self.D @ diags(J) @ self.D.T
    #   di = (diags(I_sgn) @ L @ diags(I_sgn)).diagonal()
    #   I_norm = pseudoinverse(np.sqrt(di))
    #   L = diags(I_norm) @ L @ diags(I_norm)
    #   L.data 

  def __call__(self, f: Callable[SimplexConvertible, float], **kwargs):
    fi,fj,fk,fl = update_weights(self.S, f, self.p, self.R, self.w, self.biased)
    EW = [[None] for _ in range(4)]
    if self.form == "lo":
      for cc, (I,J) in enumerate([(fj, fk), (fi, fk), (fj, fl), (fi, fl)]):
        if self.normed:
          I_sgn = np.sign(abs(I)) # I 
          self.L.set_weights(I_sgn, J, I_sgn)
          I_norm = pseudoinverse(np.sqrt(self.L.diagonal())) # degrees
          self.L.set_weights(I_norm, J, I_norm)
        else:
          I_norm = pseudoinverse(np.sqrt(I))
          self.L.set_weights(I_norm, J, I_norm)
        EW[cc] = self.solver(self.L, **kwargs)
    else:
      for cc, (I,J) in enumerate([(fj, fk), (fi, fk), (fj, fl), (fi, fl)]):
        if self.normed: 
          I_sgn = np.sign(abs(I)) # I 
          L = self.D @ diags(J) @ self.D.T
          di = (diags(I_sgn) @ L @ diags(I_sgn)).diagonal()
          I_norm = pseudoinverse(np.sqrt(di))
          L = diags(I_norm) @ L @ diags(I_norm)
        else:
          I_norm = pseudoinverse(np.sqrt(I))
          L = diags(I_norm) @ self.D @ diags(J) @ self.D.T @ diags(I_norm)
        EW[cc] = self.solver(L, **kwargs)
    return _transform_mult_ew(EW, smoothing=self.smoothing, terms=self.terms)

def mu_query(S: Union[FiltrationLike, ComplexLike], f: Callable[SimplexConvertible, float], p: int, R: tuple, w: float = 0.0, biased: bool = True, solver=None, form = 'array', normed: bool = False, **kwargs):
  """
  Parameterizes a multiplicity (mu) query restricting the persistence diagram of a simplicial complex to box 'R'
  
  Parameters: 
    S = Simplicial complex, or its corresponding Laplacian operator 
    R = box (i,j,k,l,w) to restrict to, where i < j <= k < l, and w > 0 is a smoothing parameter
    f = filter function on S (or equivalently, scalar product on its Laplacian operator)
    p = homology dimension 
    smoothing = parameters for singular values
  """
  assert isinstance(S, ComplexLike) or isinstance(S, FiltrationLike), f"Invalid complex type '{type(S)}'"
  fi,fj,fk,fl = update_weights(S, f, p, R, w, biased)

  ## Compute the multiplicities 
  # kwargs['sqrt'] = True if 'sqrt' not in kwargs.keys() else kwargs['sqrt']
  pseudo = lambda x: np.reciprocal(x, where=~np.isclose(x, 0)) # scalar pseudo-inverse
  EW = [None]*4
  if form == "lo":
    L = up_laplacian(S, p=p, form="lo")
    for cc, (I,J) in enumerate([(fj, fk), (fi, fk), (fj, fl), (fi, fl)]):
      if normed:
        I_sgn = np.where(np.isclose(I, 0.0), 0.0, 1.0) #np.sign(abs(I)) # I 
        L.set_weights(I_sgn, J, I_sgn)
        I_norm = pseudo(np.sqrt(L.diagonal())) # degrees; multiplying by I shouldn't matter?
        L.set_weights(I_norm, J, I_norm)
      else:
        I_norm = pseudo(np.sqrt(I))
        L.set_weights(I_norm, J, I_norm)
      solver = eigvalsh_solver(L)
      EW[cc] = solver(L)
  elif form == "array": 
    # I_norm = pseudo(np.sqrt(I * di))
    D = boundary_matrix(S, p=p+1)
    for cc, (I,J) in enumerate([(fj, fk), (fi, fk), (fj, fl), (fi, fl)]):
      if normed: 
        I_sgn = np.sign(abs(I))  # I
        di = (diags(I_sgn) @ D @ diags(J) @ D.T @ diags(I_sgn)).diagonal()
        I_norm = pseudo(np.sqrt(di))
        L = diags(I_norm) @ D @ diags(J) @ D.T @ diags(I_norm)
      else:
        I_norm = pseudo(np.sqrt(I))
        L = diags(I_norm) @ D @ diags(J) @ D.T @ diags(I_norm)
      solver = eigvalsh_solver(L)
      EW[cc] = solver(L)
      ## (p+2)*max((D @ diags(J) @ D.T).diagonal())/min(I[I > 0])
  else: 
    raise ValueError(f"Invalid form '{form}'.")
  return _transform_mult_ew(EW, shape=(card(S,p), card(S,p+1)), **kwargs)

# def spri_query(L: Union[sparray, LinearOperator], matrix_f: Callable[float, float], p: int, R: tuple):
#   """
#   """



def mu_query_mat(S: Union[FiltrationLike, ComplexLike], f: Callable[SimplexConvertible, float], p: int, R: tuple, w: float = 0.0, biased: bool = True,  form = 'array', normed=False, **kwargs):
  """
  Parameterizes a multiplicity (mu) query restricting the persistence diagram of a simplicial complex to box 'R'
  
  Parameters: 
    S = Simplicial complex, or its corresponding Laplacian operator 
    R = box (i,j,k,l,w) to restrict to, where i < j <= k < l, and w > 0 is a smoothing parameter
    f = filter function on S (or equivalently, scalar product on its Laplacian operator)
    p = homology dimension 
    smoothing = parameters for singular values
  """
  assert isinstance(S, ComplexLike) or isinstance(S, FiltrationLike), f"Invalid complex type '{type(S)}'"
  fi,fj,fk,fl = update_weights(S, f, p, R, w, biased)
  pseudo = lambda x: np.reciprocal(x, where=~np.isclose(x, 0)) # scalar pseudo-inverse
  L = [None]*4
  if form == "lo":
    for cc, (I,J) in enumerate([(fj, fk), (fi, fk), (fj, fl), (fi, fl)]):
      L[cc] = up_laplacian(S, p=p, form="lo")
      if normed:
        I_sgn = np.sign(abs(I))
        L[cc].set_weights(I_sgn, J, I_sgn)
        I_norm = pseudo(np.sqrt(I * L.diagonal())) # degrees
        L[cc].set_weights(I_norm, J, I_norm)
      else:
        I_norm = pseudo(np.sqrt(I))
        L[cc].set_weights(I_norm, J, I_norm)
  elif form == "array": 
    # I_norm = pseudo(np.sqrt(I * di))
    D = boundary_matrix(S, p=p+1)
    for cc, (I,J) in enumerate([(fj, fk), (fi, fk), (fj, fl), (fi, fl)]):
      if normed: 
        I_sgn = np.sign(abs(I)) 
        di = (diags(I_sgn) @ D @ diags(J) @ D.T @ diags(I_sgn)).diagonal()
        I_norm = pseudo(np.sqrt(di))
        L[cc] = diags(I_norm) @ D @ diags(J) @ D.T @ diags(I_norm)
      else:
        I_norm = pseudo(np.sqrt(I))
        L[cc] = diags(I_norm) @ D @ diags(J) @ D.T @ diags(I_norm)
  else: 
    raise ValueError(f"Invalid form choice {form}")
  return L

class MuFamily:
  """ 
  A multiplicity (mu) family M is a multiplicity statistic M(i) generated over a parameterized family of F = { f1, f2, ..., fk }

  Given a pair (S, r) where 'S' is simplicial complex and 'r' a box in the upper half-plane, 
  the quantity M(i) is real number representing the cardinality of dgm(S, fi) restricted to 'r'. To obtain this cardinality, 
  the multiplicity is characterized via a set of numerical rank computations performed on certain sub-matrices of the boundary operator. 
  Since these quantities are derived spectrally, this class precomputes the eigenvalues of these linear operators, storing them for 
  efficient (vectorized) access via the __call__ operator. 

  Constructor parameters: 
    S := Fixed simplicial complex 
    F := Iterable of filter functions, each representing scalar-products equipped to S
    p := dimension of persistence to restrict R too

  Methods: 
    precompute := precomputes the signature
  """
  def __init__(self, S: ComplexLike, family: Iterable[Callable], p: int = 0, form: str = "array"):
    # assert isinstance(S, ComplexLike)
    assert not(family is iter(family)), "Iterable 'family' must be repeateable; a generator is not sufficient!"
    self.S = S
    self.family = family
    self.np = card(S, p)
    self.nq = card(S, p+1)
    self.p = p
    self.form = form
  
  ## Does not change eigenvalues!
  def precompute(self, R: ArrayLike, w: float = 0.0, biased: bool = True, normed: bool = False, progress: bool = False, **kwargs) -> None:
    """Precomputes the signature across a parameterized family.

    Parameters: 
      R := Rectangle in the upper half-plane 
      w = smoothing parameter associated to _R_. Defaults to 0.  
      normed = whether to normalize the laplacian(s). Defaults to false. 
      kwargs = additional arguments to pass to eigvalsh_solver.
    """
    assert len(R) == 4 and is_sorted(R)
    k = len(self.family)
    self._terms = list([[None]*k for _ in range(4)])
    pseudo = lambda x: np.reciprocal(x, where=~np.isclose(x, 0)) # scalar pseudo-inverse
    family_it = progressbar(enumerate(self.family), count=len(self.family)) if progress else enumerate(self.family)
    
    ## Construct a single operator or matrix for iteration
    if self.form == "array":
      D = boundary_matrix(self.S, p=self.p+1).astype(np.float64) if self.form == "array" else None
      solver = eigvalsh_solver(D @ D.T)
    elif self.form == "lo":
      L = up_laplacian(self.S, p=self.p, form="lo")
      solver = eigvalsh_solver(L) 
    else: 
      raise ValueError("Unknown type") 
    

    ## Traverse the family 
    for i, f in family_it:
      assert isinstance(f, Callable), "f must be a simplex-wise weight function f: S -> float !"
      self._fi, self._fj, self._fk, self._fl = update_weights(self.S, f=f, p=self.p, R=R, w=w, biased=biased) # updates self._fi, ..., self._fl
      if self.form == "array":
        for cc, (I,J) in enumerate([(self._fj, self._fk), (self._fi, self._fk), (self._fj, self._fl), (self._fi, self._fl)]):
          if normed:
            I_sgn = np.sign(abs(I)) # I
            di = (diags(I_sgn) @ D @ diags(J) @ D.T @ diags(I_sgn)).diagonal()
            I_norm = pseudo(np.sqrt(di)) # used to be I * di 
            #I_norm = pseudo(np.sqrt(I * di)) # used to be I * di 
            L = diags(I_norm) @ D @ diags(J) @ D.T @ diags(I_norm)
          else:
            I_norm = pseudo(np.sqrt(I))
            L = diags(I_norm) @ D @ diags(J) @ D.T @ diags(I_norm)
            # L = D @ diags(J) @ D.T
          self._terms[cc][i] = solver(L, **kwargs)
          # ew_ext = np.sort(np.append(np.repeat(0, len(I)-len(ew)), ew))
          # self._terms[cc][i] = ew_ext * np.sort(I)
      elif self.form == "lo":
        for cc, (I,J) in enumerate([(self._fj, self._fk), (self._fi, self._fk), (self._fj, self._fl), (self._fi, self._fl)]):
          if normed:
            I_sgn = np.sign(abs(I)) # I 
            # I_sgn = np.where(np.isclose(I, 0), 0.0, 1.0)
            L.set_weights(I_sgn, J, I_sgn) ## TODO: PERHAPS don't set face weights to None, as that defaults to identity. Use entries in {0,1}
            I_norm = pseudo(np.sqrt(L.diagonal())) # degree computation
            L.set_weights(I_norm, J, I_norm)
          else:
            I_norm = pseudo(np.sqrt(I))
            L.set_weights(I_norm, J, I_norm)
          self._terms[cc][i] = solver(L, **kwargs)
      else: 
        raise ValueError("Unknown type")  
    
    # Compress the eigenvalues into sparse matrices, where each row _i_ contains the 
    # set of non-negative eigenvalues for the scalar-product fi
    self._Terms = [None]*4
    for ti in range(4):
      self._Terms[ti] = lil_array((len(self.family), card(self.S, self.p)), dtype=np.float64)
      for cc, ev in enumerate(self._terms[ti]):
        self._Terms[ti][cc,:len(ev)] = ev
      self._Terms[ti] = self._Terms[ti].tocsr()
      self._Terms[ti].eliminate_zeros() 
  
  def make_callable(self, R: ArrayLike, w: float = 0.0, biased: bool = True, normed: bool = False, form: str = "array") -> Callable:
    assert form == "array" or form == "lo", f"Invalid form argument '{str(form)}'"
    _terms = list([[None]*k for _ in range(4)])
    pseudo = lambda x: np.reciprocal(x, where=~np.isclose(x, 0)) # scalar pseudo-inverse
    if self.form == "array":
      D = boundary_matrix(self.S, p=self.p+1) if self.form == "array" else None
    else:
      L = up_laplacian(self.S, p=self.p, form="lo")
    def _mu_smooth(f: Callable):
      self._fi, self._fj, self._fk, self._fl = update_weights(self.S, f=f, p=self.p, R=R, w=w, biased=biased) 
      if self.form == "array":
        for cc, (I,J) in enumerate([(self._fj, self._fk), (self._fi, self._fk), (self._fj, self._fl), (self._fi, self._fl)]):
          if normed:
            I_sgn = np.sign(abs(I))
            di = (diags(I_sgn) @ D @ diags(J) @ D.T @ diags(I_sgn)).diagonal()
            I_norm = pseudo(np.sqrt(di)) # used to be I * di 
            L = diags(I_norm) @ D @ diags(J) @ D.T @ diags(I_norm)
          else:
            I_norm = pseudo(np.sqrt(I))
            L = diags(I_norm) @ D @ diags(J) @ D.T @ diags(I_norm)
          solver = eigvalsh_solver(L, **kwargs) ## TODO: change behavior to not be matrix specific! or to transform based on matrix
      else:
        for cc, (I,J) in enumerate([(self._fj, self._fk), (self._fi, self._fk), (self._fj, self._fl), (self._fi, self._fl)]):
          if normed:
            I_sgn = np.sign(abs(I))
            L.set_weights(I_sgn, J, I_sgn) 
            I_norm = pseudo(np.sqrt(L.diagonal())) # degree computation
            L.set_weights(I_norm, J, I_norm)
          else:
            I_norm = pseudo(np.sqrt(I))
            L.set_weights(I_norm, J, I_norm)
            solver = eigvalsh_solver(L, **kwargs)
        _terms[cc] = solver(L)
      
  @staticmethod
  def elementwise_row_sum(T: spmatrix, f: Callable):
    rs = np.ravel(np.add.reduceat(np.append(f(T.data), 0), T.indptr)[:-1])
    rs[np.ravel(np.isclose(T.sum(axis=1), 0))] = 0
    return rs

  ## Vectorized version 
  def __call__(self, smoothing: Callable = None, terms: bool = False, **kwargs) -> Union[float, ArrayLike]:
    """Evaluates the precomputed eigenvalues to yield the (possibly smoothed) multiplicities. 
    
    Vectorized evaluation of various reductions on a precomputed set of eigenvalues.
    Smoothing has a few effects depending on its usage: 
      - smoothing = None      <=> compute the numerical rank              ||*||_0
      - smoothing = False     <=> compute the nuclear norm                ||*||_1
      - smoothing = True      <=> compute the huber (hinge)               balance between ||*||_1 and ||*||_2 
      - smoothing = Callable  <=> apply an elementwise operation to the eigenvalues 
    
    Parameters: 
      smoothing = Element-wise real-valued callable, boolean, or None. Defaults to None, which returns the numerical rank.
      terms = bool indicating whether to return the multiplicities or the 4 constitutive terms themselves. Defaults to false. 

    Returns:
      ndarray of shape (k,) where k := size of the family if terms = False, otherwise of shape (4,k)
    """
    if smoothing is None: 
      boundary_shape = card(self.S, self.p), card(self.S, self.p+1)
      if terms:
        sig = np.zeros((4, len(self.family)))
        for cc,(T,s) in enumerate(zip(self._Terms, [1,-1,-1,1])):
          sig[cc,:] += s*np.array([spectral_rank(T[[i],:].data, shape=boundary_shape, **kwargs) for i in range(T.shape[0])])
      else:
        sig = np.zeros(len(self.family))
        for T,s in zip(self._Terms, [1,-1,-1,1]):
          sig += s*np.array([spectral_rank(T[[i],:].data, shape=boundary_shape, **kwargs) for i in range(T.shape[0])])
      sig = sig.astype(int)
    else:
      if isinstance(smoothing, bool):
        smoothing = lambda x: x if smoothing else huber()
      assert isinstance(smoothing, Callable)
      sig = np.zeros(len(self.family))
      if terms == False:
        sig += self.elementwise_row_sum(self._Terms[0], smoothing)
        sig -= self.elementwise_row_sum(self._Terms[1], smoothing)
        sig -= self.elementwise_row_sum(self._Terms[2], smoothing)
        sig += self.elementwise_row_sum(self._Terms[3], smoothing)
      else: 
        sig = np.zeros((4, len(self.family)))
        for cc,(T,s) in enumerate(zip(self._Terms, [1,-1,-1,1])):
          sig[cc,:] += s*self.elementwise_row_sum(T, smoothing)
    return sig

# from pbsig.linalg import ParameterizedLaplacian

class SpectralRankInvariant:
  """ 
  A sieve M is a statistic M(i) generated over a parameterized family of F = { f1, f2, ..., fk }
  
  Parameters: 
    S: Fixed simplicial complex 
    F: Iterable of filter functions. Each f should respect the face poset. (optional)
    p: Homology dimension of interest. Defaults to 0 (connected components).
    form: Representation of the Laplacian operator. Defaults to 'lo' (LinearOperator).

  Attributes:
    p_faces: the stored simplicial complex (ComplexLike).
    p_faces: the stored simplicial complex (ComplexLike).

  Methods: 
    precompute: precomputes the signature
  """
  def __init__(self, S: ComplexLike, family: Union[Iterable, Callable] = None, p: int = 0, **kwargs):
    from pbsig.linalg import PsdSolver
    from pbsig.csgraph import WeightedLaplacian
    assert isinstance(S, ComplexLike), "S must be ComplexLike"
    self.family = family if family is not None else [sx.generic_filter(lambda x: 1.0)]
    self.y_faces = np.array([s for s in faces(S,p-1)]).astype(np.uint16)
    self.p_faces = np.array([s for s in faces(S,p)]).astype(np.uint16)
    self.q_faces = np.array([s for s in faces(S,p+1)]).astype(np.uint16)
    # self.p = p
    # self.np = card(S, p)
    # self.nq = card(S, p+1)
    # self.form = form
    # self.laplacian = up_laplacian(S, p=self.p, form=self.form, normed=True)
    self.p_laplacian = WeightedLaplacian(S, p = p-1, **kwargs)
    self.q_laplacian = WeightedLaplacian(S, p = p, **kwargs)
    # self.q_operators = ParameterizedLaplacian(S, family, p, form=form)
    # if p > 0: 
    #   self.p_operators = ParameterizedLaplacian(S, family, p-1, form=form)
    self.solver = PsdSolver(eigenvectors=False)
    self.delta = np.sqrt(np.finfo(self.q_laplacian.dtype).eps)
    ## Note: breaking away from the corner-point per evaluation might not be good due to irregular shapes
    self._sieve = np.empty(shape=0, dtype=[('i', float), ('j', float), ('sign', int), ('index', int), ('finite', bool)])
    self.bounds_domain = (0, 1) # bounds on the domain of the family, if Callable
    self.bounds_range = (0, 1)  # bounds on the range of the filter functions
    self._default_val = { "eigenvalues" : array('d'), "lengths" : array('I') }

  # @property
  # def family(self): 
  #   """The _family_ refers to the parameter space of filter functions. 

  #   (1) an Iterable of Callables, the inner of which are filter function 
  #   (2) a Callable itself, in which case bounds should be properly set 
  #   """
  #   return self._family
  
  # @family.setter
  # def family(self, family: Union[Callable, Iterable], *args):
  #   """ Sets the parameterized family to the product """
  #   from more_itertools import spy
  #   if isinstance(family, Iterable):
  #     assert not(family is iter(family)), "Iterable 'family' must be repeatable; a generator is not sufficient!"
  #     assert isinstance(family, Sized), "The family of iterables must be sized"
  #     f, _ = spy(family)
  #     assert isinstance(f[0], Callable), "Iterable family must be a callables!"
  #     self._family = family
  #   else: 
  #     assert isinstance(family, Callable), "family must be iterable or Callable"
  #     self._family = [family]

  @property
  def sieve(self):
    """ Sets the sieve to the union of rectangles given by _R_. """
    return self._sieve
  
  @sieve.setter
  def sieve(self, R: ArrayLike = None):
    """ Sets the sieve to the union of rectangles given by _R_. """
    R = np.sort(np.atleast_2d(R), axis=1)
    if R is None: 
      return self._sieve 
    elif isinstance(R, np.ndarray) and R.shape[1] == 2:
      I,J,SGN,IND = [],[],[],[]
      for cc, (i,j) in enumerate(R): # (x1,x2,y1,y2)
        I.append(i)
        J.append(j)
        SGN.append(1)
        IND.append(cc)
      self._sieve = np.fromiter(zip(I,J,SGN,IND), dtype=self._sieve.dtype)
    else:
      assert isinstance(R, np.ndarray) and R.shape[1] == 4, "Invalid rectangle set given"
      I,J,SGN,IND,FINITE = [],[],[],[],[]
      for cc, (i,j,k,l) in enumerate(R): # (x1,x2,y1,y2)
        if np.isinf(i) and np.isinf(l):
          I.append(j)
          J.append(k)
          SGN.append(1)
          IND.append(cc)
          FINITE.append(False)
        else: 
          for pp, s, idx in zip(product([i,j], [k,l]), [-1,1,1,-1], repeat(cc, 4)):
            I.append(pp[0])
            J.append(pp[1])
            SGN.append(s)
            IND.append(idx)
            FINITE.append(True)
      self._sieve = np.fromiter(zip(I,J,SGN,IND,FINITE), dtype=self._sieve.dtype)
  
  def randomize_sieve(self, n_rect: int = 1, plot: bool = False, **kwargs):
    from bokeh.io import show
    kwargs = dict(area=(0.0025, 0.050), disjoint=True) | kwargs 
    self.sieve = sample_rect_halfplane(n_rect, **kwargs)  # must be disjoint 
    if plot: show(self.figure_sieve())

  def figure_sieve(self, p = None, **kwargs):
    from pbsig.vis import figure_dgm, valid_parameters
    from bokeh.models import Rect
    p = figure_dgm(**kwargs) if p is None else p
    indices = self._sieve['index']
    # rect_opts = { idx : {} for idx in np.unique(indices) } | 
    rect_kwargs = valid_parameters(Rect, **kwargs)
    for idx in np.unique(indices):
      r = self._sieve[indices == idx]
      x,y = np.mean(r['i']), np.mean(r['j'])
      w,h = np.max(np.diff(np.sort(r['i']))), np.max(np.diff(np.sort(r['j'])))
      p.rect(x,y,width=w,height=h, **rect_kwargs)
    # lb, ub = min(self._sieve['i']) - 1, max(self._sieve['j']) + 1
    # p.x_range = Range1d(lb, ub)
    # p.y_range = Range1d(lb, ub)
    return p

  # def restrict_family(self, i: float, j: float, w: float, **kwargs) -> ArrayLike:
  #   """ Restricts the weights supplied to the Laplacian family of operators to point (i,j) in the upper half plane. 
     
  #   Specifically, this function parameterizes the family of Laplacian operators by a post composition of the supplied filter function 'f' 

  #   """
  #   si, sj = smooth_upstep(lb=i, ub=i+w), smooth_dnstep(lb=j-w, ub=j+self.delta)
  #   self.q_operators.post_p = si
  #   self.q_operators.post_q = sj  
  # self.operators = ParameterizedLaplacian(S, family, self.operators.p, form=self.operators.form)
  # 
  # fp, fq = f(self.operators.p_faces), f(self.operators.q_faces)
  # fp, fq = si(fp), sj(fq) # for benchmarking purposes, these are not combined above
  # self.operators.param_weights(fq, fp)
  
  def sift(self, w: float = 0.0, progress: bool = True, **kwargs):
    """Sifts through the supplied _family_ of functions, computing the spectra of the stored _laplacian_ operator."""
    if len(self._sieve) == 0:
      raise ValueError("No rectilinear sieve has been chosen! Consider using 'randomize_sieve'.")
    
    ## Setup the main iterator
    n_corner_pts, n_family = len(self.sieve), len(self.family)
    corner_iter = zip(self.sieve['i'], self.sieve['j'], self.sieve['finite'])
    
    ## Setup the default values in the dict
    self.spectra = { corner_id : copy.deepcopy(self._default_val) for corner_id in range(n_corner_pts) }
    
    ## Projects each point (i,j) of the sieve onto a Krylov subspace
    delta = np.finfo(self.q_laplacian.dtype).eps # should should stay quite small
    for cc, (i,j,finite) in progressbar(enumerate(corner_iter), len(self.sieve)):
      for f in self.family:
        yw, pw, qw = f(self.y_faces), f(self.p_faces), f(self.q_faces) # p-1, p, p+1
        if not finite:
          inc_all = smooth_upstep(0, w)
          fi_inc = smooth_dnstep(lb = i-w, ub = i+delta)
          fi_exc = smooth_upstep(lb = i, ub = i+w)
          fj_inc = smooth_dnstep(lb = j-w, ub = j+delta)
          
          ## Get the four terms 
          # np.sum(pw <= i) == spectral_rank(fi_inc(pw))
          ew0 = fi_inc(pw)
          ew1 = self.solver(self.p_laplacian.reweight(fi_inc(pw), inc_all(yw)).operator())
          ew2 = self.solver(self.q_laplacian.reweight(fj_inc(qw), inc_all(pw)).operator())
          ew3 = self.solver(self.q_laplacian.reweight(fj_inc(qw), fi_exc(pw)).operator())

          ## Normally spectra is indexed for each corner point, but for unbounded we just 
          ## collect everything together for now 
          for ew in [ew0, ew1, ew2, ew3]:
            self.spectra[cc]['eigenvalues'].extend(ew)
            self.spectra[cc]['lengths'].extend([len(ew)])
        else:
          si, sj = smooth_upstep(lb=i, ub=i+w), smooth_dnstep(lb=j-w, ub=j+self.delta)
          self.q_laplacian.reweight(sj(qw), si(pw))
          q_ew = self.solver(self.q_laplacian.operator())
          self.spectra[cc]['eigenvalues'].extend(q_ew)
          self.spectra[cc]['lengths'].extend([len(q_ew)])

  ## TODO: add ability to handle not just length-dependent function, but pure elementwise ufunc
  def summarize(self, spectral_func: Callable[Number, ArrayLike] = None, **kwargs) -> ArrayLike:
    """Applies a spectral function to the precomputed eigenvalues.
    
    The spectral function can either be vector-valued (in which case the sum is used as a reduction) or scalar-valued. 
    """
    n_pts, n_family = len(self.sieve), len(self.family)
    f = np.sum if spectral_func is None else spectral_func
    assert isinstance(f, Callable), "reduce function must be Callable"
    r, vector_valued = f(np.zeros(10)), False

    ## Handle if f is vector valued or scalar-valued
    if isinstance(r, Container) and len(r) == 10:
      assert np.allclose(r, 0.0), "elementwise function must map [0,...,0] -> [0,...,0]"
      vector_valued = True
    elif isinstance(r, Number):
      assert np.isclose(r, 0.0), "reduction function must map [0,...,0] -> 0"
      vector_valued = False
    else: 
      raise ValueError("Unknown type of output of function _f_.")
    
    ## Apply f to the precomputed eigenvalues
    values = np.zeros(shape=(n_pts, n_family))
    for ii, d in enumerate(self.spectra.values()):
      if self.sieve['finite'][ii]:
        ew_split = np.split(d['eigenvalues'], np.cumsum(d['lengths']))[:-1]
        for jj, ew in enumerate(ew_split):
          values[ii,jj] = np.sum(f(ew)) if vector_valued else f(ew)
      else: 
        ew_split = np.split(d['eigenvalues'], np.cumsum(d['lengths']))[:-1]
        for jj, (t0,t1,t2,t3) in enumerate(chunked(ew_split, 4)):
          t0 = np.sum(f(t0)) if vector_valued else f(t0)
          t1 = np.sum(f(t1)) if vector_valued else f(t1)
          t2 = np.sum(f(t2)) if vector_valued else f(t2)
          t3 = np.sum(f(t3)) if vector_valued else f(t3)
          values[ii,jj] = t0 - t1 - t2 + t3
          # print(f"t0, t1, t2, t3")

    ## Apply inclusion-exclusion to add the appropriately signed corner points together
    n_summaries = len(np.unique(self._sieve['index']))
    summary = np.zeros(shape=(n_summaries, n_family))
    np.add.at(summary, self._sieve['index'], np.c_[self._sieve['sign']]*values) # should be fixed based on inclusion/exclusion
    return summary
  
  def figure_summary(self, func: Union[np.ndarray, Callable] = None, **kwargs):
    from pbsig.vis import valid_parameters, bin_color
    from bokeh.models import Line
    from bokeh.plotting import figure
    nt, ns = len(self.family), len(np.unique(self._sieve['index']))
    fig_kwargs = valid_parameters(figure, **kwargs)
    p = kwargs.get('figure', figure(width=300, height=300, **fig_kwargs))
    summary = self.summarize(func) if (isinstance(func, Callable) or func is None) else func
    assert isinstance(summary, np.ndarray) and summary.shape == (ns, nt)
    sample_index = np.arange(0, nt)
    pt_color = (bin_color(sample_index, 'viridis')*255).astype(np.uint8)
    for f_summary in summary:
      p.line(sample_index, f_summary, color='red')
      p.scatter(sample_index, f_summary, color=pt_color)
    return p

  # def restrict(self, i: float, j: float, w: float, fp: ArrayLike, fq: ArrayLike,  **kwargs):
  #   si, sj = smooth_upstep(lb=i, ub=i+w), smooth_dnstep(lb=j-w, ub=j+self.delta)
  #   fp, fq = si(fp), sj(fq) 
  #   if self.form == 'lo':
  #     self._project(fp, fq, **kwargs)
  #   else:
  #     raise NotImplementedError("Array form of projection not implemented")
  
  def compute_dgms(self, S: ComplexLike, **kwargs):
    """Constructs all the persistence diagrams in the parameterized family.
    
    Parameters: 
      S : the simplicial complex used to construct the parameterized filter
      method: persistence algorithm to use. Currently ignored. 
    """
    ph_dgm = lambda filter_f: ph(filtration(S, f=filter_f), output="dgm", **kwargs)
    self.dgms = [ph_dgm(filter_f) for filter_f in self.family]
    return self.dgms

  # def estimate_step(self, f: Callable, phi: Callable, alpha: List[float]) -> float:
  #   import numdifftools as nd
  #   np.array([self.gradient_fd(f=f, phi=phi, alpha0=a0, n_coeff=4, obj=True) for a0 in alpha])
  #   # nd.Derivative(exp, full_output=True)

  def __call__(self, filter_f: Callable, phi: Callable, i: float = None, j: float = None, w: float = 0.0, **kwargs):
    """Computes the spectral rank invariant for a specified filter function _filter_. 
    
    Optionally regularized by matrix function _phi_. If a corner point _(i,j)_ are supplied, then the spectral function is evaluated 
    at that point; otherwise all of the spectral invariants are computed for the stored sieve and then they are 
    combined according to the inclusion/exclusion aggregation rule. 
    """
    if isinstance(i, Number) and isinstance(j, Number):
      return np.sum(phi(self.restrict(i=i, j=j, w=w, f=filter_f, **kwargs)))
    elif i is None and j is None:
      corner_it = zip(self.sieve['i'], self.sieve['j'])
      F = np.array([np.sum(phi(self.restrict(i=i, j=j, w=w, f=filter_f, **kwargs))) for i,j in corner_it])
      f_obj = np.zeros(len(np.unique(self._sieve['index'])))
      np.add.at(f_obj, self._sieve['index'], self._sieve['sign']*F)
      return f_obj
    else: 
      raise ValueError(f"Invalid parameters {i}, {j}")



  def gradient_fd(self, f: Callable, phi: Callable, alpha0: float, da: float = 1e-5, n_coeff: int = 2, obj: bool = False, i: float = None, j: float = None, w: float = 0.0, **kwargs) -> np.ndarray:
    """Computes a uniform finite-difference approximation of the gradient of _f_ at the point _alpha0_ w.r.t fixed parameters (i,j,w).
    
    Parameters:
      f: alpha-parameterized Callable that returns a filter-function anywhere in the neighborhood of _alpha0_
      phi: vector- or scalar-valued regularizer to apply to the spectra of the Laplacian operators. 
      alpha0: coordinate to compute the gradient at
      i,j: fixed upper half-plane points to restrict the Laplacians too. If not supplied (or None), all of gradients of sieve are computed and coombined.
      w: smoothing/continuity parameter, see project(...) for more details. 
      da: uniform-spacing value for the finite-difference calculation(s). Defaults to 1e-5. 
      n_coeff: number of coefficients to use in the finite-difference calculation(s). Must be positive even number (defaults to 2). Larger values yield more precise differences. 
      obj: boolean whether to return the gradient alone (default) or both the objective and the gradient. 

    Returns: 
      The gradient _g_, if obj == False, else a pair (f,g) where _f_ is the objective and _g_ is the gradient. 
    """
    # from numdifftools import Derivative, Jacobian
    assert isinstance(f, Callable), "_f_ must be a Callable."
    assert isinstance(n_coeff, Integral) and n_coeff % 2 == 0, f"Number of coefficients must be even."
    assert isinstance(f(alpha0), Callable), "_f_ must yield a filter function (callable)"
    increments = np.setdiff1d(np.arange(-(n_coeff >> 1), (n_coeff >> 1)+1), [0])
    coeffs = []  
    if n_coeff == 2:
      coeffs.extend([-1/2, 1/2])
    elif n_coeff == 4: 
      coeffs.extend([1/12, -2/3, 2/3, -1/12])
    elif n_coeff == 6: 
      coeffs.extend([-1/60, 3/20, -3./4., 3/4, -3/20, 1/60])
    elif n_coeff == 8: 
      coeffs.extend([1/280, -4/105, 1/5, -4/5, 4/5, -1/5, 4/105, -1/280])
    else: 
      raise ValueError(f"Invalid number of coefficients '{n_coeff}' supplied ")
    
    ## Objective + gradient
    def _f_grad(_i: float, _j: float):
      grad = np.sum([coeff*np.sum(phi(self.project(i=_i, j=_j, w=w, f=f(float(alpha0 + s*da)), **kwargs))) for s, coeff in zip(increments, coeffs)])
      if obj: 
        f_val = np.sum(phi(self.project(i=_i, j=_j, w=w, f=f(alpha0), **kwargs)))
        return f_val, grad
      else: 
        return grad

    ## Combine the gradients per rectlinear sieve, if warranted
    if isinstance(i, Number) and isinstance(j, Number):
      return _f_grad(i,j)
    elif i is None and j is None:
      corner_it = zip(self.sieve['i'], self.sieve['j'])
      G = np.array([_f_grad(i,j) for i,j in corner_it])
      n_grads = len(np.unique(self._sieve['index']))
      if obj: 
        f_obj, grads = np.zeros(n_grads), np.zeros(n_grads)
        np.add.at(f_obj, self._sieve['index'], self._sieve['sign']*G[:,0])
        np.add.at(grads, self._sieve['index'], self._sieve['sign']*G[:,1])
        return f_obj, grads 
      else: 
        grads = np.zeros(n_grads)
        np.add.at(grads, self._sieve['index'], self._sieve['sign']*np.ravel(G))
        return grads
    else: 
      raise ValueError(f"Invalid parameters {i}, {j}")

def lower_star_multiplicity(F: Iterable[ArrayLike], S: ComplexLike, R: Collection[tuple], p: int = 0, method: str = ["exact", "rank"], **kwargs):
  """
  Returns the multiplicity values of a set of rectangles in the upper half-plane evaluated on the 0-th dim. persistence 
  diagram of a sequence of vertex functions F = [f1, f2, ..., f_n]
 
  Each r = (a,b,c,d) should satisfy a < b <= c < d so as to parameterize a box [a,b] x [c,d] in the upper half-plane. 
  
  Specialization/overloads: 
    - Use (-inf,a,a,inf) to calculate the Betti number at index a
    - Use (-inf,a,b,inf) to calculate the persistent Betti number at (a,b), for any a < b 

  Parameters: 
    F := Iterable of vertex functions. Each item should be a collection of length 'n'.
    S := Fixed simplicial complex. 
    R := Collection of 4-tuples representing rectangles r=(a,b,c,d) in the upper half-plane
    p := Homology dimension to compute. Must be non-negative. Default to 0. 
    method := either "exact" or "singular"
  """
  from pbsig.persistence import ph0_lower_star
  if p != 0: raise NotImplemented("p > 0 hasn't been implemented yet")
  R = np.array(R)

  if method == "exact":
    E = np.array(list(faces(S, p=1)))
    for i, f in enumerate(F):
      dgm = ph0_lower_star(f, E, max_death="max") # O(m*a(n) + mlogm) since E is unsorted
      if len(dgm) > 0:
        for j, (a,b,c,d) in enumerate(R):
          assert a < b and b <= c and c < d, f"Invalid rectangle ({a:.2f}, {b:.2f}, {c:.2f}, {d:.2f}): each rectangle must have positive measure"
          born_mu = np.logical_and(dgm[:,0] >= a, dgm[:,0] <= b)
          died_mu = np.logical_and(dgm[:,1] > c, dgm[:,1] <= d) # < d?
          yield sum(np.logical_and(born_mu, died_mu))
  else: 
    # assert isinstance(method, str) and method in ["rank", "nuclear", "generic", "frobenius"], f"Invalid method input {method}"
    for c0, f in enumerate(F):
      weight = lambda s: max(f[s])
      for c1, (i,j,k,l) in enumerate(R):
        assert i < j and j <= k and k < l, f"Invalid rectangle ({i:.2f}, {j:.2f}, {k:.2f}, {l:.2f}): each rectangle must have positive measure"
        t1 = rank_ll(S, j, k, p=p+1, weight=weight, **kwargs)
        t2 = rank_ll(S, i, k, p=p+1, weight=weight, **kwargs)
        t3 = rank_ll(S, j, l, p=p+1, weight=weight, **kwargs)
        t4 = rank_ll(S, i, l, p=p+1, weight=weight, **kwargs)
        print(f"{t1-t2-t3+t4}: {t1},{t2},{t3},{t4}")
        yield t1-t2-t3+t4
    
# def mu_query(S: ComplexLike, i: float, j: float, p: int = 1, weight: Optional[Callable] = None, w: float = 0.0):
#   """
#   Returns the rank of the lower-left portion of the p-th boundary matrix of 'S' with real-valued coefficients.

#   The rank is computed by evaluating the given weight function on all p and (p+1)-simplices of S, and then 
#   restricting the entries to non-zero if (p+1)-simplices have weight <= j and p-simplices have weight > than i. 

#   S := an abstract simplicial complex. 
#   i := (p-1)-face threshold. Only (p-1)-simplices with weight strictly greater than i are considered.  
#   j := p-face threshold. Only p-simplices with weight less than or equal to j are considered. 
#   p := dimension of the chain group on S with which to construct the boundary matrix.
#   w := non-negative smoothing parameter. If 0, non-zero entries will have value 1, and otherwise they will be in [0,1]. 
#   weight := real-valued weight function on S. 
#   """
#   assert i <= j, "i must be <= j"
#   assert w >= 0, "smoothing parameter mut be non-negative."
#   assert p >= 0, "Invalid homology dimension."
#   assert isinstance(weight, Callable)
#   delta = np.finfo(float).eps # TODO: use bound on spectral norm to get tol instead of eps?
#   ss_ic = smooth_upstep(lb = i, ub = i+w)         # STEP UP:   0 (i-w) -> 1 (i), includes (i, infty)
#   ss_j = smooth_dnstep(lb = j-w, ub = j+delta)    # STEP DOWN: 1 (j-w) -> 0 (j), includes (-infty, j]
#   smoothed_weight = lambda s: float(ss_ic(weight(s)) if len(s) == p else ss_j(weight(s)))
#   # print(p)
#   LS = up_laplacian(S, p = p-1, weight = smoothed_weight, form = 'lo') # 0th up laplacian = D1 @ D1.T

#   # print(LS)
#   # print(LS.data)
#   return numerical_rank(LS)
# if i is None: 
#   sig = np.array([self.__call__(i, smoothing) for i in range(self.k)])
#   return sig
# else: 
#   assert isinstance(i, Integral) and i >= 0 and i < self.k, "operator(i) defined for i in [0, 1, ..., k]"
#   eps,p,method = smoothing
#   t1 = sum(sgn_approx(self._T1[i], eps, p, method))
#   t2 = sum(sgn_approx(self._T2[i], eps, p, method))
#   t3 = sum(sgn_approx(self._T3[i], eps, p, method))
#   t4 = sum(sgn_approx(self._T4[i], eps, p, method))
#   return t1 - t2 - t3 + t4