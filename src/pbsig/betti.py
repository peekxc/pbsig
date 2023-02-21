import numpy as np 
from typing import *
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import structural_rank
from .persistence import * 
from .apparent_pairs import *
from .linalg import *
from .utility import progressbar, smooth_upstep, smooth_dnstep
from splex.geometry import flag_weight

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
#   if smoothing == "huber":
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

## TODO: This actually does need to be done
def betti_query(S: Union[LinearOperator, ComplexLike], i: float, j: float, smoothing: tuple = (0.5, 1.5, 0), solver=None, **kwargs):
  pass 
  # L = S if isinstance(S, UpLaplacian) else up_laplacian(S, p=0, form='lo')
  # assert isinstance(L, UpLaplacian)
  # assert i <= j, f"Invalid point ({i:.2f}, {j:.2f}): must be in the upper half-plane"
  # fw = np.array([f(s) for s in L.faces])
  # sw = np.array([f(s) for s in L.simplices])
  # delta = np.finfo(float).eps                      
  # fi = smooth_upstep(lb = i, ub = i+w)(fw)          # STEP UP:   0 (i-w) -> 1 (i), includes (i, infty)
  # fj = smooth_upstep(lb = j, ub = j+w)(fw)          # STEP UP:   0 (j-w) -> 1 (j), includes (j, infty)
  # fk = smooth_dnstep(lb = k-w, ub = k+delta)(sw)    # STEP DOWN: 1 (k-w) -> 0 (k), includes (-infty, k]
  # fl = smooth_dnstep(lb = l-w, ub = l+delta)(sw)    # STEP DOWN: 1 (l-w) -> 0 (l), includes (-infty, l]
  # pseudo = lambda x: np.reciprocal(x, where=~np.isclose(x, 0)) # scalar pseudo-inverse
  # atol = kwargs['tol'] if 'tol' in kwargs else 1e-5
  # EW = [None]*4
  # ## Multiplicity formula 
  # for cc, (I,J) in enumerate([(fj, fk), (fi, fk), (fj, fl), (fi, fl)]):
  #   L.set_weights(None, J, None)
  #   I_norm = I * L.diagonal() # degrees
  #   EW[cc] = smooth_rank(L.set_weights(pseudo(np.sqrt(I_norm)), J, pseudo(np.sqrt(I_norm))), smoothing=smoothing, **kwargs)
  # return EW[0] - EW[1] - EW[2] + EW[3]


## Cone the complex
def cone_weight(x: ArrayLike, vid: int = -1, v_birth: float = -np.inf, collapse_weight: float = np.inf):
  from scipy.spatial.distance import pdist
  flag_w = flag_weight(x)
  def _cone_weight(s):
    s = Simplex(s)
    if s == Simplex([vid]):
      return v_birth
    elif vid in s:
      return collapse_weight
    else: 
      return flag_w(s)
  return _cone_weight

def mu_query(S: Union[FiltrationLike, ComplexLike], R: tuple, f: Callable[SimplexConvertible, float], p: int = 0, smoothing: Callable = None, solver=None, terms: bool = False, form = 'array', **kwargs):
  """
  Parameterizes a multiplicity (mu) query restricting the persistence diagram of a simplicial complex to box 'R'
  
  Parameters: 
    S = Simplicial complex, or its corresponding Laplacian operator 
    R = box (i,j,k,l,w) to restrict to, where i < j <= k < l, and w > 0 is a smoothing parameter
    f = filter function on S (or equivalently, scalar product on its Laplacian operator)
    p = homology dimension 
    smoothing = parameters for singular values
  """
  assert len(R) == 4 or len(R) == 5, "Must be a rectangle"
  # L = S if issubclass(type(S), UpLaplacianBase) else up_laplacian(S, p=p, form=form)
  assert isinstance(S, ComplexLike) or isinstance(S, FiltrationLike), f"Invalid complex type '{type(S)}'"
  #assert issubclass(type(L), UpLaplacianBase), f"Type '{type(L)}' be derived from UpLaplacianBase"
  (i,j,k,l), w = (R[:4], 0.0) if len(R) == 4 else (R[:4], R[4])
  assert i < j and j <= k and k < l, f"Invalid rectangle ({i:.2f}, {j:.2f}, {k:.2f}, {l:.2f}): each rectangle must have positive measure"
  # fw = np.array([f(s) for s in L.faces])
  # sw = np.array([f(s) for s in L.simplices]) ## TODO: replace 
  pw = np.array([f(s) for s in faces(S, p)])
  qw = np.array([f(s) for s in faces(S, p+1)])
  delta = np.finfo(float).eps                       # TODO: use bound on spectral norm to get tol instead of eps?
  fi = smooth_upstep(lb = i, ub = i+w)(pw)          # STEP UP:   0 (i-w) -> 1 (i), includes (i, infty)
  fj = smooth_upstep(lb = j, ub = j+w)(pw)          # STEP UP:   0 (j-w) -> 1 (j), includes (j, infty)
  fk = smooth_dnstep(lb = k-w, ub = k+delta)(qw)    # STEP DOWN: 1 (k-w) -> 0 (k), includes (-infty, k]
  fl = smooth_dnstep(lb = l-w, ub = l+delta)(qw)    # STEP DOWN: 1 (l-w) -> 0 (l), includes (-infty, l]
  
  ## Compute the multiplicities 
  kwargs['sqrt'] = True if 'sqrt' not in kwargs.keys() else kwargs['sqrt']
  pseudo = lambda x: np.reciprocal(x, where=~np.isclose(x, 0)) # scalar pseudo-inverse
  EW = [None]*4
  if form == "lo":
    L = up_laplacian(S, p=p, form="lo")
    for cc, (I,J) in enumerate([(fj, fk), (fi, fk), (fj, fl), (fi, fl)]):
      # print(L.shape)
      L.set_weights(None, J, None)
      I_norm = pseudo(np.sqrt(I * L.diagonal())) # degrees
      L.set_weights(I_norm, J, I_norm)
      EW[cc] = stable_rank(L, method=2) if not smooth else smooth_rank(L, pp=1.0, smoothing=smoothing, **kwargs)
  elif form == "array":
    D = boundary_matrix(S, p=p+1)
    for cc, (I,J) in enumerate([(fj, fk), (fi, fk), (fj, fl), (fi, fl)]):
      di = (D @ diags(J) @ D.T).diagonal()
      I_norm = pseudo(np.sqrt(I * di))
      L = diags(I_norm) @ D @ diags(J) @ D.T @ diags(I_norm)
      EW[cc] = stable_rank(L, method=2) if not smooth else smooth_rank(L, pp=1.0, smoothing=smoothing, **kwargs)
  else: 
    raise ValueError(f"Invalid form '{form}'.")
  return EW[0] - EW[1] - EW[2] + EW[3] if not terms else EW

# def mu_query_mat(S: Union[FiltrationLike, ComplexLike], R: tuple, f: Callable[SimplexConvertible, float], p: int = 0, terms: bool = False, form = 'lo', **kwargs):
#   """
#   Parameterizes a multiplicity (mu) query restricting the persistence diagram of a simplicial complex to box 'R'
  
#   Parameters: 
#     S = Simplicial complex, or its corresponding Laplacian operator 
#     R = box (i,j,k,l,w) to restrict to, where i < j <= k < l, and w > 0 is a smoothing parameter
#     f = filter function on S (or equivalently, scalar product on its Laplacian operator)
#     p = homology dimension 
#     smoothing = parameters for singular values
#   """
#   assert len(R) == 4 or len(R) == 5, "Must be a rectangle"
#   assert isinstance(S, ComplexLike) or isinstance(S, FiltrationLike), f"Invalid complex type '{type(S)}'"
#   (i,j,k,l), w = (R[:4], 0.0) if len(R) == 4 else (R[:4], R[4])
#   assert i < j and j <= k and k < l, f"Invalid rectangle ({i:.2f}, {j:.2f}, {k:.2f}, {l:.2f}): each rectangle must have positive measure"
#   pw = np.array([f(s) for s in faces(S, p)])
#   qw = np.array([f(s) for s in faces(S, p+1)])
#   delta = np.finfo(float).eps                       # TODO: use bound on spectral norm to get tol instead of eps?
#   fi = smooth_upstep(lb = i, ub = i+w)(pw)          # STEP UP:   0 (i-w) -> 1 (i), includes (i, infty)
#   fj = smooth_upstep(lb = j, ub = j+w)(pw)          # STEP UP:   0 (j-w) -> 1 (j), includes (j, infty)
#   fk = smooth_dnstep(lb = k-w, ub = k+delta)(qw)    # STEP DOWN: 1 (k-w) -> 0 (k), includes (-infty, k]
#   fl = smooth_dnstep(lb = l-w, ub = l+delta)(qw)    # STEP DOWN: 1 (l-w) -> 0 (l), includes (-infty, l]
#   pseudo = lambda x: np.reciprocal(x, where=~np.isclose(x, 0)) # scalar pseudo-inverse
#   L = [None]*4
#   if form == "lo":
#     for cc, (I,J) in enumerate([(fj, fk), (fi, fk), (fj, fl), (fi, fl)]):
#       L[cc] = up_laplacian(S, p=p, form="lo")
#       L[cc].set_weights(None, J, None)
#       I_norm = pseudo(np.sqrt(I * L.diagonal())) # degrees
#       L[cc].set_weights(I_norm, J, I_norm)
#   elif form == "array":
#     # D = boundary_matrix(S, p=p+1)
#     #for cc, (I,J) in enumerate([(fj, fk), (fi, fk), (fj, fl), (fi, fl)]):
#       # LM = D @ diags(J) @ D.T
#       # di = LM.diagonal()
#       # I_norm = pseudo(np.sqrt(I * di))
#       # L[cc] = diags(I_norm) @ LM @ diags(I_norm)
#     for cc, (I,J) in enumerate([(j, k), (i, k), (j, l), (i, l)]):
#       face_f = smooth_upstep(lb = I, ub = I+w)
#       coface_f = smooth_dnstep(lb = J-w, ub = J+delta)
#       weight_f = lambda s: face_f(f(s)).item() if dim(s) == p else coface_f(f(s)).item()
#       L[cc] = up_laplacian(S, p=p, form="array", normed=True, weight=weight_f)
#   else: 
#     raise ValueError(f"Invalid form '{form}'.")
#   return L

def mu_sig(S: ComplexLike, R: tuple, f: Callable, p: int = 0, w: float = 0.0, **kwargs):
  """Creates a multiplicity signature for some box _R_ over a 1-parameter family _f_. 
  """
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
  def __init__(self, S: ComplexLike, family: Iterable[Callable], R: ArrayLike, p: int = 0, form: str = "array"):
    assert len(R) == 4 and is_sorted(R)
    # assert isinstance(S, ComplexLike)
    assert not(family is iter(family)), "Iterable 'family' must be repeateable; a generator is not sufficient!"
    self.S = S
    self.R = R
    self.family = family
    self.np = card(S, p)
    self.nq = card(S, p+1)
    self.p = p
    self.form = form
  
  ## Changes the eigenvalues! 
  def update_weights(self, f: Callable[SimplexConvertible, float], R: ArrayLike, w: float):
    """Updates the smooth step functions to reflect the scalar-product _f_ on _R_ smoothed by _w_.
    """
    assert len(R) == 4 and is_sorted(R)
    fw = np.array([f(s) for s in faces(self.S, self.p)])
    sw = np.array([f(s) for s in faces(self.S, self.p+1)])
    i,j,k,l = R 
    delta = np.finfo(float).eps # TODO: use bound on spectral norm to get tol instead of eps?        
    self._fi = smooth_upstep(lb = i, ub = i+w)(fw)          # STEP UP:   0 (i-w) -> 1 (i), includes (i, infty)
    self._fj = smooth_upstep(lb = j, ub = j+w)(fw)          # STEP UP:   0 (j-w) -> 1 (j), includes (j, infty)
    self._fk = smooth_dnstep(lb = k-w, ub = k+delta)(sw)    # STEP DOWN: 1 (k-w) -> 0 (k), includes (-infty, k]
    self._fl = smooth_dnstep(lb = l-w, ub = l+delta)(sw)    # STEP DOWN: 1 (l-w) -> 0 (l), includes (-infty, l]

  ## Does not change eigenvalues!
  def precompute(self, w: float = 0.0, normed: bool = False, progress: bool = False, **kwargs) -> None:
    """Precomputes the signature across a parameterized family.

    Parameters: 
      w = smoothing parameter. Defaults to 0.  
      normed = whether to normalize the laplacian(s). Defaults to false. 
      kwargs = additional arguments to pass to eigvalsh_solver.
    """
    k = len(self.family)
    self._terms = [[None]*k for i in range(4)] 
    i,j,k,l = self.R
    pseudo = lambda x: np.reciprocal(x, where=~np.isclose(x, 0)) # scalar pseudo-inverse
    # as-is? self.L.set_weights(self._fj, self._fl, self._fj)
    family_it = progressbar(enumerate(self.family), count=len(self.family)) if progress else enumerate(self.family)
    for i, f in family_it:
      assert isinstance(f, Callable), "f must be a simplex-wise weight function f: S -> float !"
      self.update_weights(f=f, R=self.R, w=w) # updates self._fi, ..., self._fl
      if self.form == "array":
        D = boundary_matrix(self.S, p=self.p+1)
        for cc, (I,J) in enumerate([(self._fj, self._fk), (self._fi, self._fk), (self._fj, self._fl), (self._fi, self._fl)]):
          if normed:
            di = (D @ diags(J) @ D.T).diagonal()
            I_norm = pseudo(np.sqrt(I * di))
            L = diags(I_norm) @ D @ diags(J) @ D.T @ diags(I_norm)
          else:
            L = diags(np.sqrt(I)) @ D @ diags(J) @ D.T @ diags(np.sqrt(I))
          solver = eigvalsh_solver(L, **kwargs) ## TODO: change behavior to not be matrix specific! or to transform based on matrix
          self._terms[cc][i] = solver(L)
      elif self.form == "lo":
        L = up_laplacian(self.S, p=self.p, form="lo")
        for cc, (I,J) in enumerate([(self._fj, self._fk), (self._fi, self._fk), (self._fj, self._fl), (self._fi, self._fl)]):
          if normed:
            L.set_weights(None, J, None) ## adjusts degree
            I_norm = pseudo(np.sqrt(I * L.diagonal())) # degrees
            L.set_weights(I_norm, J, I_norm)
          else:
            L.set_weights(pseudo(np.sqrt(I)), J, pseudo(np.sqrt(I)))
          solver = eigvalsh_solver(L, **kwargs)
          self._terms[cc][i] = solver(L)
      else: 
        raise ValueError("Unknown type")  
    
    ## Compress the eigenvalues into sparse matrices 
    self._Terms = [None]*4
    for ti in range(4):
      self._Terms[ti] = lil_array((len(self.family), card(self.S, self.p)), dtype=np.float64)
      for cc, ev in enumerate(self._terms[ti]):
        self._Terms[ti][cc,:len(ev)] = ev
      self._Terms[ti] = self._Terms[ti].tocsr()
      self._Terms[ti].eliminate_zeros() 
    

  ## Changes the eigenvalues! 
  ## TODO: allow multiple rectangles
  def _update_rectangle(self, R: ArrayLike, defer: bool = False):
    pass
  
  @staticmethod
  def elementwise_row_sum(T: spmatrix, f: Callable):
    rs = np.ravel(np.add.reduceat(np.append(f(T.data), 0), T.indptr)[:-1])
    rs[np.ravel(np.isclose(T.sum(axis=1), 0))] = 0
    return rs
    
  ## Vectorized version 
  def __call__(self, smoothing: Callable = None, terms: bool = False, **kwargs) -> Union[float, ArrayLike]:
    """Evaluates the precomputed eigenvalues to yield the (possibly smoothed) multiplicities. 
    
    Vectorized evaluation of various reductions on a precomputed set of eigenvalues.
    
    Parameters: 
      smoothing = Element-wise real-valued callable, boolean, or None. Defaults to None, which returns the numerical rank.
      terms = bool indicating whether to return the multiplicities or the 4 constitutive terms themselves. Defaults to false. 
    """
    if smoothing is None: 
      boundary_shape = card(self.S, self.p), card(self.S, self.p+1)
      if terms:
        sig = np.zeros((4, len(self.family)))
        for cc,(T,s) in enumerate(self._Terms, [1,-1,-1,1]):
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
        for cc,(T,s) in enumerate(self._Terms, [1,-1,-1,1]):
          sig[cc,:] += s*self.elementwise_row_sum(T, smoothing)
    return sig

# E: Union[ArrayLike, Iterable],
def lower_star_multiplicity(F: Iterable[ArrayLike], S: ComplexLike, R: Collection[tuple], p: int = 0, method: str = ["exact", "rank"], **kwargs):
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


# def _fix_ew(ew): 
#   atol = kwargs['tol'] if 'tol' in kwargs else 1e-5
#   ew[np.isclose(abs(ew), 0.0, atol=atol)] = 0.0 # compress
#   #if not(all(np.isreal(ew)) and all(ew >= 0.0)):
#     #print(f"Negative or non-real eigenvalues detected [{min(ew)}]")
#     # print(ew)
#     # print(min(ew))
#   #assert all(np.isreal(ew)) and all(ew >= 0.0), "Negative or non-real eigenvalues detected."
#   return np.maximum(ew, 0.0)
# self._T1 = [_fix_ew(t) for t in self._T1]
# self._T2 = [_fix_ew(t) for t in self._T2]
# self._T3 = [_fix_ew(t) for t in self._T3]
# self._T4 = [_fix_ew(t) for t in self._T4]
# self._T1 = csr_array(self._T1)
# self._T2 = csr_array(self._T2)
# self._T3 = csr_array(self._T3)
# self._T4 = csr_array(self._T4)
# self._T1.eliminate_zeros() 
# self._T2.eliminate_zeros() 
# self._T3.eliminate_zeros() 
# self._T4.eliminate_zeros() 

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