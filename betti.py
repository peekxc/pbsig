from typing import *
import numpy as np 
from persistence import * 
from apparent_pairs import *

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

def subdifferential(A): 
  U, _, Vt = np.linalg.svd(A, full_matrices=False)
  return(U @ Vt)

def pb_gradient(f: Callable, t: Any, b: float, d: float, Sg: Callable, summands: bool = False, h: float = 100*np.finfo(float).resolution): 
  K = rips(f(t), diam=d, p=2)
  D1 = boundary_matrix(K, p=1)
  D2 = boundary_matrix(K, p=2)

  ## Jacobian finite difference for D1 
  e_sgn = np.tile([-1, 1], len(K['edges']))
  ew_p = e_sgn * np.maximum(b - np.repeat(rips_weights(f(t + h), K['edges']), 2), 0.0)
  ew_m = e_sgn * np.maximum(b - np.repeat(rips_weights(f(t - h), K['edges']), 2), 0.0)
  # csc_matrix(((tw_p - tw_m)/(2*h), D2.indices, D2.indptr), shape=D2.shape)
  D1_jac = D1.copy()
  D1_jac.data = (ew_p - ew_m)/(2*h)

  ## Jacobian finite difference for D2
  # Wrong! 
  t_sgn = np.tile([1, -1, 1], len(K['triangles']))
  tw_p = t_sgn * np.maximum(d - np.repeat(rips_weights(f(t + h), K['triangles']), 3), 0.0)
  tw_m = t_sgn * np.maximum(d - np.repeat(rips_weights(f(t - h), K['triangles']), 3), 0.0)
  D2_jac = D2.copy()
  # csc_matrix(((tw_p - tw_m)/(2*h), D2.indices, D2.indptr), shape=D2.shape)
  D2_jac.data = (tw_p - tw_m)/(2*h)

  ## Gradients! 
  # from autograd import grad
  # import warnings
  # gs = grad(S)
  # with warnings.catch_warnings():
  #   warnings.simplefilter("ignore")
  # g0 = np.sum([gs(ew) for ew in pdist(f(t))])
  g0 = np.sum([Sg(ew) for ew in pdist(f(t))])
  g1 = np.matrix(subdifferential(D1.A).flatten('F')) @ np.matrix(D1_jac.A.flatten('F')).T
  g2 = np.matrix(subdifferential(D2.A).flatten('F')) @ np.matrix(D2_jac.A.flatten('F')).T

  ## 1D-case
  if isinstance(t, float): 
    g1 = g1.item()
    g2 = g2.item()
  return((g0, g1, g2) if summands else g0 - (g1 + g2))