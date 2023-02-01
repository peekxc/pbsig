import numpy as np 
from typing import *
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import structural_rank
from .persistence import * 
from .apparent_pairs import *
from .linalg import *
from .utility import progressbar, smooth_upstep, smooth_dnstep

def rank_C2(i: int, j: int, n: int):
  i, j = (j, i) if j < i else (i, j)
  return(int(n*i - i*(i+1)/2 + j - i - 1))

def unrank_C2(x: int, n: int):
  i = int(n - 2 - np.floor(np.sqrt(-8*x + 4*n*(n-1)-7)/2.0 - 0.5))
  j = int(x + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2)
  return(i,j) 

## Generate a random set of rectangles in the upper half plane 
def sample_rect_halfplane(n: int, area: tuple = (0, 0.05), disjoint: bool = False):  
  """ 
  Generate random rectilinear boxes with area between [lb,ub] in upper-half plane via rejection sampling 
  """
  cc = 0
  R = []
  while cc < n:
    r = np.sort(np.random.uniform(size=4, low = -1.0, high=1.0))
    ra = (r[1]-r[0])*(r[3]-r[2])
    if ra >= area[0] and ra <= area[1]:
      R.append(r)
    else: 
      continue
    cc += 1
  return np.array(R)

def Lipshitz(f: ArrayLike, x: ArrayLike):
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

def soft_threshold(s: ArrayLike, mu: Union[float, ArrayLike] = 1.0):
  s = s.copy()
  if isinstance(mu, float):
    mu = np.repeat(mu, len(s))
  gt, et, lt = s >= mu, np.logical_and(-mu <= s, s <= mu), s <= -mu
  s[gt] = s[gt] - mu[gt]
  s[et] = 0.0
  s[lt] = s[lt] + mu[lt]
  return(s)

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

def relaxed_pb(X: ArrayLike, b: float, d: float, bp: float, m1: float, m2: float, summands: bool = False, p: Tuple = (1.0, 1.0), rank_adjust: bool = False, **kwargs):
  """ Relaxed Persistent betti objective """
  assert is_point_cloud(X) or is_pairwise_distances(X), "Wrong format"
  XD = pdist(X) if is_point_cloud(X) else X
  eps = 100*np.finfo(float).resolution
  S1, R1, R2 = persistent_betti_rips(XD, b, d, summands=True)
  
  ## Term 1 plt.plot(np.sort(s_extra), relax_identity(np.sort(s_extra), p=2, negative=True))
  #S = make_smooth_step(b, bp)
  #t0 = np.sum([S(x) for x in XD])
  t0 = np.sum(XD <= b) 
  s_extra = (XD[np.bitwise_and(XD > b, XD <= bp)]-b)/(bp - b)
  assert np.all(s_extra <= 1)
  t0 += np.sum(relax_identity(s_extra, p=p[0], negative=True))

  ## ~ || D1 ||_* <= rank(D1) = dim(B_{p-1})
  D1, (vw, ew) = rips_boundary(XD, p=1, diam=d, sorted=False) ## keep at d, as rank is unaffected by next statement
  b_tmp = np.maximum(b - np.repeat(ew, 2), 0.0)
  #b_tmp = b*(b_tmp/b)**(1/p)
  D1.data = np.sign(D1.data) * b_tmp
  sv1 = np.linalg.svd(D1.A, compute_uv=False)
  tol = np.max([np.finfo(float).resolution*100, np.max(sv1) * np.max(D1.shape) * np.finfo(float).eps])
  # R1 = np.sum(sv1 > tol)

  nsv1 = ( (1/m1) * sv1[sv1 > tol] ) # should all be <= 1 w/ appropriate m1 
  assert np.all(nsv1 <= 1.0) # must be true! 
  nsv1 = relax_identity(nsv1, p=p[1])
  t1 = np.sum(nsv1/R1) + np.max([R1 - 1, 0]) if rank_adjust else np.sum(nsv1)

  ## ~ || ||_*  
  D2, (ew, tw) = rips_boundary(XD, p=2, diam=d, sorted=False)
  d_tmp = np.maximum(d - np.repeat(tw, 3), 0.0)
  #d_tmp = d*(d_tmp/d)**(1/p)
  D2.data = np.sign(D2.data) * d_tmp
  
  ## Need to ensure rank is matched!
  Z, B = D1[:,ew <= b], D2[ew <= b,:]
  # Z, B = D1, D2
  PI = projector_intersection(Z, B, space='NR', method=kwargs.get('method', 'neumann')) # Projector onto null(D1) \cap range(D2)
  spanning_set = PI @ B
  # spanning_set = np.c_[PI @ Z.T, PI @ B] # don't use: Z spans more than nullspace! 
  sv2 = np.linalg.svd(spanning_set, compute_uv=False)
  tol = np.max([np.finfo(float).resolution*100, np.max(sv2) * np.max(spanning_set.shape) * np.finfo(float).eps])
  # R2 = np.sum(sv2 > tol)
  
  nsv2 = ( (1/m2) * sv2[sv2 > tol] )
  assert np.all(nsv2 <= 1.0) # must be true! 
  nsv2 = relax_identity(nsv2, p=p[1])
  t2 = np.sum(nsv2/R2) + np.max([R2 - 1, 0]) if rank_adjust else np.sum(nsv2)

  return((t0, t1, t2) if summands else t0 - (t1 + t2))

def nuclear_constants(n: int, b: float, d: float, bound: str = "global", f: Optional[Callable] = None, T: Optional[List] = None):
  """
  f := (optional) single function which generates a point cloud
  """
  import math
  if bound == "global":
    m1 = np.sqrt(2 * b**2 * math.comb(n,2))
    m2 = np.sqrt(3 * d**2 * math.comb(n,3))
  elif bound == "tight":
    assert not(f is None), "f must be supplied if a tight bound is required"
    assert not(T is None), "T must be supplied if a tight bound is required"
    ## If best constants desired
    m1, m2 = 0.0, 0.0
    for t in T: 
      XD = pdist(f(t))
      D1, (vw, ew) = rips_boundary(XD, p=1, diam=d, sorted=False)
      D1.data = np.sign(D1.data) * np.maximum(b - np.repeat(ew, 2), 0.0)
      D2, (ew, tw) = rips_boundary(XD, p=2, diam=d, sorted=False)
      D2.data = np.sign(D2.data) * np.maximum(d - np.repeat(tw, 3), 0.0)
      D1 = D1[:,ew <= b]
      D2 = D2[ew <= b,:]
      m1_tmp = 0 if np.prod(D1.shape) == 0 else np.max(np.linalg.svd(D1.A, compute_uv=False))
      m2_tmp = 0 if np.prod(D2.shape) == 0 else np.max(np.linalg.svd(D2.A, compute_uv=False))
      m1, m2 = np.max([m1, m1_tmp]), np.max([m2, m2_tmp])
  else: 
    raise ValueError("Invalid bound type given.")
  return(m1, m2)

def relax_identity(X: ArrayLike, q: float = 1.0, complement: bool = False, ub: float = 250.0):
  """
  Relaxes the identity function f(x) = x on the unit interval [0,1] such that the relaxation f', for q \in [0, 1]:
    a. approaches f'(x) = x as q -> 0 
    b. approaches f'(x) = 1 as q -> 1
  If complement=True, then f(x) = 1 - x is relaxed such that f' approaches f'(x) = 0 as q -> 1.

  Parameters: 
    X := values in [0, 1]
    q := scaling parameter between [0, 1].
    complement := whether to relax f(x) = x or f(x) = 1 - x in the unit interval. Defaults to false. 
    ub := upper bound on exponential scaling. Larger values increase sharpness of relaxation. Defaults to 250. 

  """
  assert (0 <= q) and (q <= 1), "q must be <= 1"
  assert np.all(X >= 0) and np.all(X <= 1), "x values must like in [0,1]"
  p = 1/np.exp(q*np.log(ub)) if not(complement) else np.exp(q*np.log(ub))
  return(X**p if not(complement) else (1.0 - X**p))
  # beta = 1.0 # todo: make a parameter
  # if not(negative):
  #   tl = (1-beta)*np.array([0.5, 0.5]) + beta*np.array([0.0, 1.0])
  #   x = np.array([0.0, tl[0], 1.0])
  #   y = np.array([0.0, tl[1], 1.0])
  # else:
  #   tl = (1-beta)*np.array([0.5, 0.5]) + beta*np.array([0.0, 0.0])
  #   x = np.array([0.0, tl[0], 1.0])
  #   y = np.array([1.0, tl[1], 0.0])
  # p0, p1, p2 = np.array([x[0], y[0]]), np.array([x[1], y[1]]), np.array([x[2], y[2]])
  # T = np.sqrt(X**p)
  # C = np.array([(p1 + (1-ti) * (p0 - p1) + ti * (p2 - p1))[1] for ti in T])
  # return(C)




def plot_direction(V: ArrayLike, T: ArrayLike, W: ArrayLike, cmap: str = 'jet'):
  import matplotlib
  assert V.shape[0] == len(W)
  import plotly.graph_objects as go
  from tallem.color import bin_color
  cm = matplotlib.cm.get_cmap(cmap)
  colors = list(np.array([cm(v) for v in np.linspace(0, 1, endpoint=True)]))
  TW = np.array([np.max(W[t]) for t in T])
  face_colors = bin_color(TW, colors)
  axis = dict(showbackground=True,backgroundcolor="rgb(230, 230,230)",gridcolor="rgb(255, 255, 255)",zerolinecolor="rgb(255, 255, 255)")
  layout = go.Layout(scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis), aspectmode='data', aspectratio=dict(x=1, y=1, z=1)))
  mesh = go.Mesh3d(x=V[:,0], y=V[:,1], z=V[:,2], i=T[:,0],j=T[:,1],k=T[:,2],facecolor=face_colors) #  intensity=W+np.min(W), colorscale='Jet'
  tri_points = V[T]
  Xe, Ye, Ze = [], [], []
  for T in tri_points:
    Xe.extend([T[k%3][0] for k in range(4)]+[ None])
    Ye.extend([T[k%3][1] for k in range(4)]+[ None])
    Ze.extend([T[k%3][2] for k in range(4)]+[ None])
  lines = go.Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines', name='', line=dict(color= 'rgb(70,70,70)', width=1))  
  fig = go.Figure(data=[mesh, lines], layout=layout)
  fig.show()

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

def mu_query(L: Union[LinearOperator, SimplicialComplex], R: tuple, f: Callable, smoothing: tuple = (0.5, 1.5, 0), w: float = 0.0):
  """
  Given a weighted up laplacian 'UL', computes
  """
  assert len(R) == 4, "Must be a rectangle"
  if isinstance(L, UpLaplacian):
    i,j,k,l = R
    assert i < j and j <= k and k < l, f"Invalid rectangle ({i:.2f}, {j:.2f}, {k:.2f}, {l:.2f}): each rectangle must have positive measure"
    delta = np.finfo(float).eps # TODO: use bound on spectral norm to get tol instead of eps?
    sd_k = smooth_dnstep(lb = k-w, ub = k+delta)    # STEP DOWN: 1 (k-w) -> 0 (k), includes (-infty, k]
    sd_l = smooth_dnstep(lb = l-w, ub = l+delta)    # STEP DOWN: 1 (l-w) -> 0 (l), includes (-infty, l]
    su_i = smooth_upstep(lb = i, ub = i+w)          # STEP UP:   0 (i-w) -> 1 (i), includes (i, infty)
    su_j = smooth_upstep(lb = j, ub = j+w)          # STEP UP:   0 (j-w) -> 1 (j), includes (j, infty)
    
    ## Get initial set of weights
    fw = np.array([f(s) for s in L.faces])
    sw = np.array([f(s) for s in L.simplices])

    ## Multiplicity formula 
    fi,fj,fk,fl = np.sqrt(su_i(fw)), np.sqrt(su_j(fw)), sd_k(sw), sd_l(sw)
    t1 = smooth_rank(L.set_weights(fj, fk, fj))
    t2 = smooth_rank(L.set_weights(fi, fk, fi))
    t3 = smooth_rank(L.set_weights(fj, fl, fj))
    t4 = smooth_rank(L.set_weights(fi, fl, fi))
    return t1 - t2 - t3 + t4
  else: 
    raise ValueError("Invlaid input")
  return 0 

def mu_sig(S: SimplicialComplex, R: tuple, f: Callable, p: int = 0, w: float = 0.0, **kwargs):
  assert isinstance(f, Callable), "f must be a simplex-wise weight function f: S -> float !"
  assert len(R) == 4, "Must be a rectangle"
  i,j,k,l = R
  assert i < j and j <= k and k < l, f"Invalid rectangle ({i:.2f}, {j:.2f}, {k:.2f}, {l:.2f}): each rectangle must have positive measure"
  pseudo = lambda x: np.reciprocal(x, where=~np.isclose(x, 0)) # scalar pseudo-inverse
  
  L = up_laplacian(S, p=p, form='lo')
  eigh_solver = parameterize_solver(L, **kwargs)
  fw = np.array([f(s) for s in L.faces])
  sw = np.array([f(s) for s in L.simplices])
  delta = np.finfo(float).eps # TODO: use bound on spectral norm to get tol instead of eps?        
  fi = smooth_upstep(lb = i, ub = i+w)(fw)          # STEP UP:   0 (i-w) -> 1 (i), includes (i, infty)
  fj = smooth_upstep(lb = j, ub = j+w)(fw)          # STEP UP:   0 (j-w) -> 1 (j), includes (j, infty)
  fk = smooth_dnstep(lb = k-w, ub = k+delta)(sw)    # STEP DOWN: 1 (k-w) -> 0 (k), includes (-infty, k]
  fl = smooth_dnstep(lb = l-w, ub = l+delta)(sw)    # STEP DOWN: 1 (l-w) -> 0 (l), includes (-infty, l]

  atol = kwargs['tol'] if 'tol' in kwargs else 1e-5
  EW = [None]*4
  # min(eigh_solver(L)) vs min(eigh_solver(LM))
  for cc, (I,J) in enumerate([(fj, fk), (fi, fk), (fj, fl), (fi, fl)]):
    L.set_weights(None, J, None)
    # L.precompute_degree()
    I_norm = I * L.diagonal() # degrees
    L.set_weights(pseudo(np.sqrt(I_norm)), J, pseudo(np.sqrt(I_norm)))
    EW[cc] = eigh_solver(L)
    #EW[cc][np.isclose(abs(EW[cc]), 0.0, atol=atol)] = 0.0 # compress
    # if not(all(np.isreal(ew)) and all(ew >= 0.0)):
    #   print(f"Negative or non-real eigenvalues detected [{min(ew)}]")
    #EW[cc] = np.maximum(EW[cc], 0.0)

  def _transform(smoothing: tuple = (0.5, 1.0, 0)):
    eps,p,method = smoothing
    S = sgn_approx(eps=eps, p=p, method=method)
    sig = np.sum(S(EW[0])) if len(EW[0]) > 0 else  0
    sig -= np.sum(S(EW[1])) if len(EW[1]) > 0 else  0
    sig -= np.sum(S(EW[2])) if len(EW[2]) > 0 else  0
    sig += np.sum(S(EW[3])) if len(EW[3]) > 0 else  0
    return sig
  return _transform


class MuSignature:
  """ 
  A multiplicity (mu) signature M is a shape statistic M(i) generated from a parameterized family of F = { f1, f2, ..., fk }

  Given a pair (S, r) where 'S' is simplicial complex and 'r' a box in the upper half-plane, 
  the quantity M(i) is real number representing the *smoothed* cardinality of dgm(S, fi) restricted to 'r'

  Constructor parameters: 
    S := Fixed simplicial complex 
    F := Iterable of filter functions, each representing scalar-products equipped to S
    R := Rectangle in the upper half-plane 
    p := dimension of persistence to restrict R too

  Methods: 
    precompute := precomputes the signature
  """
  def __init__(self, S: SimplicialComplex, family: Iterable[Callable], R: ArrayLike, p: int = 0, **kwargs):
    assert len(R) == 4 and is_sorted(R)
    # assert isinstance(S, SimplicialComplex)
    assert not(family is iter(family)), "Iterable 'family' must be repeateable; a generator is not sufficient!"
    self.L = up_laplacian(S, p, form='lo', **kwargs)
    self.R = R
    self.family = family
    self.np = S.shape[p]
    self.nq = S.shape[p+1]
  
  ## Does not change eigenvalues!
  def precompute(self, w: float = 0.0, normed: bool = False, **kwargs) -> None:
    """
    pp: 
    """
    k = len(self.family)
    self._T1 = [None]*k
    self._T2 = [None]*k
    self._T3 = [None]*k
    self._T4 = [None]*k
    i,j,k,l = self.R
    eigh_solver = parameterize_solver(self.L, **kwargs) 
    pseudo = lambda x: np.reciprocal(x, where=~np.isclose(x, 0)) # scalar pseudo-inverse
    # as-is? self.L.set_weights(self._fj, self._fl, self._fj)
    for i, f in progressbar(enumerate(self.family), count=len(self.family)):
      assert isinstance(f, Callable), "f must be a simplex-wise weight function f: S -> float !"
      self.update_weights(f, self.R, w=w)
      if not(normed): 
        self.L.set_weights(pseudo(np.sqrt(self._fj)), self._fk, pseudo(np.sqrt(self._fj)))
        # print(f"LW: {sum(self.L.face_left_weights)}, CW: {sum(self.L.simplex_weights)}, RW: {sum(self.L.face_right_weights)}")
        self._T1[i] = eigh_solver(self.L)
        self.L.set_weights(pseudo(np.sqrt(self._fi)), self._fk, pseudo(np.sqrt(self._fi)))
        self._T2[i] = eigh_solver(self.L)
        self.L.set_weights(pseudo(np.sqrt(self._fj)), self._fl, pseudo(np.sqrt(self._fj)))
        self._T3[i] = eigh_solver(self.L)
        self.L.set_weights(pseudo(np.sqrt(self._fi)), self._fl, pseudo(np.sqrt(self._fi)))
        self._T4[i] = eigh_solver(self.L)
        self.L.set_weights(None, None, None)
      else: ## Normalize and solve
        
        ## 1: k, j
        self.L.set_weights(None, self._fk, None)
        self.L.precompute_degree()
        fj_norm = self._fj * self.L.diagonal() # degrees
        self.L.set_weights(pseudo(np.sqrt(fj_norm)), self._fk, pseudo(np.sqrt(fj_norm)))
        self._T1[i] = eigh_solver(self.L)

        ## 2: k, i 
        self.L.set_weights(None, self._fk, None)
        self.L.precompute_degree()
        fi_norm = self._fi * self.L.diagonal() # degrees
        self.L.set_weights(pseudo(np.sqrt(fi_norm)), self._fk, pseudo(np.sqrt(fi_norm)))
        self._T2[i] = eigh_solver(self.L)

        ## 3: l, j 
        self.L.set_weights(None, self._fl, None)
        self.L.precompute_degree()
        fi_norm = self._fi * self.L.diagonal() # degrees
        self.L.set_weights(pseudo(np.sqrt(fj_norm)), self._fl, pseudo(np.sqrt(fj_norm)))
        self._T3[i] = eigh_solver(self.L)

        ## 4: l, i 
        self.L.set_weights(None, self._fl, None)
        self.L.precompute_degree()
        fi_norm = self._fi * self.L.diagonal() # degrees
        self.L.set_weights(pseudo(np.sqrt(fi_norm)), self._fl, pseudo(np.sqrt(fi_norm)))
        self._T4[i] = eigh_solver(self.L)
        
        ## For some reason, this is needed for determinism...
        self.L.set_weights(None, None, None)
    def _fix_ew(ew): 
      atol = kwargs['tol'] if 'tol' in kwargs else 1e-5
      ew[np.isclose(abs(ew), 0.0, atol=atol)] = 0.0 # compress
      #if not(all(np.isreal(ew)) and all(ew >= 0.0)):
        #print(f"Negative or non-real eigenvalues detected [{min(ew)}]")
        # print(ew)
        # print(min(ew))
      #assert all(np.isreal(ew)) and all(ew >= 0.0), "Negative or non-real eigenvalues detected."
      return np.maximum(ew, 0.0)
    self._T1 = [_fix_ew(t) for t in self._T1]
    self._T2 = [_fix_ew(t) for t in self._T2]
    self._T3 = [_fix_ew(t) for t in self._T3]
    self._T4 = [_fix_ew(t) for t in self._T4]
    self._T1 = csr_array(self._T1)
    self._T2 = csr_array(self._T2)
    self._T3 = csr_array(self._T3)
    self._T4 = csr_array(self._T4)
    self._T1.eliminate_zeros() 
    self._T2.eliminate_zeros() 
    self._T3.eliminate_zeros() 
    self._T4.eliminate_zeros() 
  
  ## Changes the eigenvalues! 
  def update_weights(self, f: Callable[SimplexLike, float], R: ArrayLike, w: float):
    assert len(R) == 4 and is_sorted(R)
    fw = np.array([f(s) for s in self.L.faces])
    sw = np.array([f(s) for s in self.L.simplices])
    i,j,k,l = R 
    delta = np.finfo(float).eps # TODO: use bound on spectral norm to get tol instead of eps?        
    self._fi = smooth_upstep(lb = i, ub = i+w)(fw)          # STEP UP:   0 (i-w) -> 1 (i), includes (i, infty)
    self._fj = smooth_upstep(lb = j, ub = j+w)(fw)          # STEP UP:   0 (j-w) -> 1 (j), includes (j, infty)
    self._fk = smooth_dnstep(lb = k-w, ub = k+delta)(sw)    # STEP DOWN: 1 (k-w) -> 0 (k), includes (-infty, k]
    self._fl = smooth_dnstep(lb = l-w, ub = l+delta)(sw)    # STEP DOWN: 1 (l-w) -> 0 (l), includes (-infty, l]


  ## Changes the eigenvalues! 
  ## TODO: allow multiple rectangles
  def _update_rectangle(self, R: ArrayLike, defer: bool = False):
    pass

  ## Vectorized version 
  def __call__(self, i: int = None, smoothing: tuple = (0.5, 1.0, 0)) -> Union[float, ArrayLike]:
    eps,p,method = smoothing
    S = sgn_approx(eps=eps, p=p, method=method)
    sig = np.zeros(len(self.family))
    sig += np.add.reduceat(S(self._T1.data), self._T1.indptr[:-1]) if len(self._T1.data) > 0 else 0
    sig -= np.add.reduceat(S(self._T2.data), self._T2.indptr[:-1]) if len(self._T2.data) > 0 else 0
    sig -= np.add.reduceat(S(self._T3.data), self._T3.indptr[:-1]) if len(self._T3.data) > 0 else 0
    sig += np.add.reduceat(S(self._T4.data), self._T4.indptr[:-1]) if len(self._T4.data) > 0 else 0
    return sig
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



#   ## Prep smooth-step functions for rectangle R = [i,j] x [k,l]
#   i,j,k,l = R 
#   sd_k = smooth_dnstep(lb = k-w, ub = k+delta)    # STEP DOWN: 1 (k-w) -> 0 (k), includes (-infty, k]
#   sd_l = smooth_dnstep(lb = l-w, ub = l+delta)    # STEP DOWN: 1 (l-w) -> 0 (l), includes (-infty, l]
#   su_i = smooth_upstep(lb = i, ub = i+w)          # STEP UP:   0 (i-w) -> 1 (i), includes (i, infty)
#   su_j = smooth_upstep(lb = j, ub = j+w)          # STEP UP:   0 (j-w) -> 1 (j), includes (j, infty)
  
#   ## Get initial set of weights based on weight function
#   weight = (lambda s: 1) if weight is None else weight 
#   L = up_laplacian(S, p=0, form='lo')
#   fw = np.array([weight(s) for s in L.faces])
#   sw = np.array([weight(s) for s in L.simplices])
  
#   ## Multiplicity formula 
#   fi,fj,fk,fl = np.sqrt(su_i(fw)), np.sqrt(su_j(fw)), sd_k(sw), sd_l(sw)
  
#   ## Choose a eigenvalue solver 
#   eigh_solver = parameterize_solver(L) 
#   # pairs = [(i,j), (), (), ()
  
#   ## Precompute the eigenvalues for each box r \in R 
#   EW1 = [] ## eigenvalues for first pair

#   pass

# def mu_query(S: SimplicialComplex, i: float, j: float, p: int = 1, weight: Optional[Callable] = None, w: float = 0.0):
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

# E: Union[ArrayLike, Iterable],
def lower_star_multiplicity(F: Iterable[ArrayLike], S: SimplicialComplex, R: Collection[tuple], p: int = 0, method: str = ["exact", "rank"], **kwargs):
  """
  Returns the multiplicity values of a set of rectangles in the upper half-plane evaluated on the 0-th dim. persistence 
  diagram of a sequence of vertex functions F = [f1, f2, ..., f_n]
 
  Each r = (a,b,c,d) should satisfy a < b <= c < d so as to parameterize a box [a,b] x [c,d] in the upper half-plane. 
  
  Specialization/overloads: 
    * Use (-inf,a,a,inf) to calculate the Betti number at index a
    * Use (-inf,a,b,inf) to calculate the persistent Betti number at (a,b), for any a < b 

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
    E = np.array(list(S.faces(p=1)))
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
    



class Laplacian_DT_2D:
  def __init__(self, X: ArrayLike, K, nd: 132):
    self.theta = np.linspace(0, 2*np.pi, nd, endpoint=False)
    self.X = X 
    self.cc = 0
    self.D1 = boundary_matrix(K, p=1).tocsc()
    self.D1.sort_indices() # ensure !
    self.W = np.zeros(shape=(1,self.D1.shape[0]))
  
  def __len__(self) -> int: 
    return len(self.theta)

  def __iter__(self):
    self.cc = 0
    return self

  def __next__(self):
    if self.cc >= len(self.theta):
      raise StopIteration
    v = np.cos(self.theta[self.cc]), np.sin(self.theta[self.cc])
    self.W = diags(self.X @ v)
    self.cc += 1
    return self.W @ self.D1 @ self.D1.T @ self.W




  # A_exc = ss_ac(f[E]).flatten()
  # B_inc = np.repeat(ss_b(edge_f), 2)
  # D1.data = np.array([s*af*bf if af*bf > 0 else 0.0 for (s, af, bf) in zip(D1_nz_pattern, A_exc, B_inc)])
  # L = D1 @ D1.T
  # k = structural_rank(L)
  # if k > 0: 
  #   k = k - 1 if k == min(D1.shape) else k
  #   T4 = eigsh(L, return_eigenvectors=False, k=k)
  #   terms[3] = relax_f(np.array(T4))
  # else: 
  #   terms[3] = 0.0
  #   nv, ne = shape
  # r, rz = np.zeros(ne), np.zeros(nv)
  # def _ab_mat_vec(x: ArrayLike): # x ~ O(V)
  #   r.fill(0) # r = Ax ~ O(E)
  #   rz.fill(0)# rz = Ar ~ O(V)
  #   for cc, (i,j) in enumerate(E):
  #     ew = max(fv[i], fv[j])
  #     r[cc] = ss_ac(fv[i])*ss_b(ew)*x[i] - ss_ac(fv[j])*ss_b(ew)*x[j]
  #   for cc, (i,j) in enumerate(E):
  #     ew = max(fv[i], fv[j])
  #     rz[i] += ew*r[cc] #?? 
  #     rz[j] -= ew*r[cc]
  #   return(rz)
    
# def lower_star_multiplicity(F: Iterable[ArrayLike], E: ArrayLike, R: Collection[tuple], p: int = 0, **kwargs):
#   a,b,c,d = next(iter(R))
#   Ef = [f[E].max(axis=1) for f in F]

