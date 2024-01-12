from typing import * 
from numpy.typing import ArrayLike
from numbers import *

import numpy as np 
import inspect 
from math import *
from scipy.sparse import *
from scipy.sparse.linalg import * 
from scipy.sparse.csgraph import *
from more_itertools import collapse
from scipy.interpolate import CubicSpline
from scipy.sparse import issparse, sparray
from math import prod


## Local imports
from .meta import *
from .simplicial import * 
from .combinatorial import * 
from .distance import dist 
from .csgraph import * 

# import _lanczos as lanczos
import _laplacian as laplacian
from splex import *

from scipy.sparse.linalg import eigs as truncated_eig
from scipy.linalg import eigh, eig as dense_eig
from scipy.spatial import KDTree
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components, floyd_warshall
from scipy.sparse.linalg import LinearOperator

def cmds(D: ArrayLike, d: int = 2, coords: bool = True, pos: bool = True):
  '''Classical Multidimensional Scaling (CMDS).

  Parameters:  
    D: _squared_ dense distance matrix.
    d: target dimension of the embedding.
    coords: whether to produce a coordinitization of 'D' or just return the eigen-sets.
    pos: keep only eigenvectors whose eigenvalues are positive. Defaults to True. 
  '''
  n = D.shape[0]
  H = np.eye(n) - (1.0/n)*np.ones(shape=(n,n)) # centering matrix
  evals, evecs = np.linalg.eigh(-0.5 * H @ D @ H)
  evals, evecs = evals[(n-d):n], evecs[:,(n-d):n]

  ## Compute the coordinates using positive-eigenvalued components only     
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
	'''Principal Component Analysis (PCA).
  '''
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
    return x + np.where(x >= t, t, x)**2
  return _moreau if x is None else _moreau(np.array(x))

def moreau_loss(x: ArrayLike, sf: Callable, t: float = 1.0):
  """
  Parameters:
    x:  ndarray, spmatrix, or LinearOperator
    sf: element-wise function to apply to the eigenvalues 
    t:  proximal scaling operator 
  """
  from scipy.sparse import spmatrix
  from scipy.optimize import minimize
  if x.ndim == 0: return 0
  if isinstance(x, np.ndarray) and all(np.ravel(x == 0)): return 0
  elif isinstance(x, spmatrix) and (len(x.data) == 0 or x.nnz == 0): return 0
  #x_ew = np.linalg.eigvalsh(x)
  x_ew = eigvalsh_solver(x)(x)
  def sgn_approx_cost(ew_hat: ArrayLike, t: float):
    return sum(sf(ew_hat)) + (1/(t*2)) * np.linalg.norm(sf(ew_hat) - sf(x_ew))**2
  w = minimize(sgn_approx_cost, x0=x_ew, args=(t), tol=1e-15, method="Powell")
  if w.status != 0:
    import warnings
    warnings.warn("Did not converge to sign vector prox")
  ew = w.x if w.status == 0 else x_ew
  return sgn_approx_cost(ew, t)

def huber(x: ArrayLike = None, delta: float = 1.0, nonnegative: bool = True) -> ArrayLike:
  """Huber loss function."""
  def _huber(x: ArrayLike): 
    x = np.maximum(x, 0.0) if nonnegative else np.abs(x)
    return np.where(x <= delta, 0.5 * (x ** 2), delta * (x - 0.5*delta))
  return _huber if x is None else _huber(x)

def tikhonov(x: ArrayLike = None, eps: float = 1.0, truncate: Union[str, float] = 0.0, nonnegative: bool = True) -> ArrayLike:
  """Tikhonov regularization function."""
  if isinstance(truncate, str) and truncate == "numrank":
    prec = np.finfo(np.float64).eps
    trunc = lambda x: np.where(x < x.max() * len(x) * prec, 0.0, x)
  elif isinstance(truncate, Number): 
    trunc = lambda x: np.where(x < truncate, 0.0, x)
  else: 
    raise ValueError("Invalid")
  def _tikhonov(x: ArrayLike):
    num = trunc(np.maximum(x, 0.0) if nonnegative else np.abs(x))
    den = num + eps 
    return num / den
  return _tikhonov if x is None else _tikhonov(x)

def heat(x: ArrayLike = None, t: float = 1.0, nonnegative: bool = True, complement: bool = False) -> ArrayLike:
  """Huber loss function."""
  def _heat(x: ArrayLike): 
    x = np.maximum(x, 0.0) if nonnegative else np.abs(x)
    return np.exp(-x * t) if not complement else 1.0 - np.exp(-x * t)
  return _heat if x is None else _heat(x)

def nuclear_interp(x: ArrayLike = None, eps: float = 1.0, rho: float = 0.5, beta: float = 0.5, nonnegative: bool = True):
  """Spectral approximation function to the nuclear norm. Satisfies f(x) -> sgn+(x) as eps -> 0 and f(x) -> abs(x) as eps -> +inf"""
  c: float = (1 + eps**rho)
  def _interp(x: ArrayLike):
    x = np.maximum(x, 0.0) if nonnegative else np.abs(x)
    return (c * np.power(x, 2*beta)) / np.power(np.power(x, 2) + eps, beta)
  return _interp if x is None else _interp(x)

def timepoint_heuristic(n: int, L: LinearOperator, A: LinearOperator, locality: tuple = (0, 1), **kwargs):
  """Constructs _n_ positive time points equi-distant in log-space for use in the map exp(-t).
  
  This uses the heuristic from "A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion" to determine 
  adequete time-points for generating a "nice" heat kernel signature, with a tuneable locality parameter. 

  Parameters: 
    n: number of time point to generate 
    L: Laplacian operator used in the the generalized eigenvalue problem.
    A: Mass matrix used in the the generalized eigenvalue problem. 
    locality: tuple indicating how to modify the lower and upper bounds of the time points to adjust for locality. 
  """
  # d = A.diagonal()
  # d_min, d_max = np.min(d), np.max(d)
  # lb_approx = (1.0/d_max)*1e-8
  # tmin_approx = 4 * np.log(10) / (2.0 / d_min)
  # tmax_approx = 4 * np.log(10) / (1e-8 / d_max)
  # TODO: revisit randomized Rayleigh quotient or use known bounds idea
  # XR = np.random.normal(size=(L_sim.shape[0],15), loc=0.0).T
  # np.max([(x.T @ L_sim @ x)/(np.linalg.norm(x)**2) for x in XR])
  lb, ub = eigsh(L, M=A, k=4, which="BE", return_eigenvectors=False, **kwargs)[np.array((1,3))] # checks out
  tmin = 4 * np.log(10) / ub
  tmax = 4 * np.log(10) / lb
  # tdiff = np.abs(tmax-tmin)
  # tmin, tmax = tmin+locality[0]*tdiff, tmax+locality[1]*tdiff
  tmin *= (1.0+locality[0])
  tmax *= locality[1]
  timepoints = np.geomspace(tmin, tmax, n)
  return timepoints 



def logsample(start: float, end: float, num: Union[int, np.ndarray] = 50, endpoint: bool = True, base: int = 2, dtype=None, axis=0):
  """Generate samples on a logarithmic scale within the interval [start, end].

  If 'num' is an integer, the samples are uniformly spaced, matching the behavior of np.logspace. 

  If 'num' is an ndarray, its values are used as relative proportions in log-scale. This can be helpful for procedures 
  seeking to generate e.g. random values that uniformly-sampled in log-scale, i.e. 

  x = logsample(1, 100, np.random.uniform(0,1,size=10))

  Yields 10 random points in the interval [1, 100] that are uniformly-sampled in log-salce.

  Parameters:
    start: The start of the interval.
    end: The end of the interval.
    num: The number of samples to generate, or an array of proportions relative to [start, end].
    endpoint: whether to include end, in the case where num is an integer.
    base: The logarithmic base. Default is 2.
    dtype: passed to np.linspace
    axis: passed to np.linspace 
     
  Returns:
    np.ndarray: An array of logarithmically spaced samples.
  """
  log_start, log_end = np.log(start) / np.log(base),  np.log(end) / np.log(base)
  if isinstance(num, np.ndarray):
    log_samples = log_start + num * np.abs(log_end-log_start)
  else: 
    log_samples = np.linspace(log_start, log_end, num, endpoint=endpoint, dtype=dtype, axis=axis)
  samples = np.power(base, log_samples)
  return samples

# def logspaced_timepoints(n: int, lb: float = 1e-6, ub: float = 2.0) -> np.ndarray:
#   """Constructs _n_ non-negative time points equi-distant in log-space for use in the map exp(-t).
  
#   If an upper-bound for time is known, it may be specified so as to map the interval [0, ub] to the range
#   such that np.exp(-t0) = 1 and np.exp(-tn) = epsilon, which epsilon is the machine epsilon.
#   """
#   ## TODO: revisit the full heuristic! The localized method works so much better
#   # min_t = 13.815510475347063 # 32-bit float 
#   # min_t = 34.53877627071313  # 64-bit floats
#   # if method == "full":
#   #   # tmin = 1e-3 / ub
#   #   # tmax = min_t / max(1e-3, lb)
#   #   tmin = 4 * np.log(10) / 2.0 
#   #   tmax = 4 * np.log(10) / 1e-6
#   #   timepoints = np.geomspace(tmin, tmax, n)
#   # elif method == "local":
#   #   assert lb != 0.0, "Local heuristic require positive lower-bound for spectral gap"
#   tmin = 4 * np.log(10) / ub
#   tmax = 4 * np.log(10) / lb
#   timepoints = np.geomspace(tmin, tmax, n)
#   return timepoints
  # else:
  #   raise ValueError(f"Unknown heuristic method '{method}' passed. Must be one 'local' or 'full'")
  # return timepoints

def vertex_masses(S: ComplexLike, X: ArrayLike, use_triangles: bool = True) -> np.ndarray:
  """Computes the cumulative area or 'mass' around every vertex"""
  # TODO: allow area to be computed via edge lengths?
  vertex_mass = np.zeros(card(S,0))
  if use_triangles:
    from pbsig.shape import triangle_areas
    areas = triangle_areas(S, X)
    for t, t_area in zip(faces(S,2), areas):
      vertex_mass[t] += t_area / 3.0
  else:
    for i,j in faces(S,1):
      edge_weight = np.linalg.norm(X[i] - X[j])
      vertex_mass[i] += edge_weight / 2.0
      vertex_mass[j] += edge_weight / 2.0
  return vertex_mass


def gauss_similarity(S: ComplexLike, X: ArrayLike, sigma: Union[str, float] = "default", **kwargs) -> np.ndarray:
  """Gaussian kernel similarity function"""
  E_ind = np.array(list(faces(S,1)))
  E_dist = dist(X[E_ind[:,0]], X[E_ind[:,1]], paired=True, **kwargs)
  o = np.mean(E_dist) if sigma == "default" else float(sigma)
  a = 1.0 / (4 * np.pi * o**2)
  w = a * np.exp(-E_dist**2 / (4.0*o))
  return w

def geomspace(start: float, stop: float, p: Union[int, np.ndarray] = 50, base: float = 2.0, reverse: bool = False):
  """Returns a subset of numbers in the interval [start,stop] spaced in log scale (a geometric progression).

  If p is an integer, then this function matches the output of np.geomspace, returning p points between [start, stop] 
  evenly spaced in logscale (for an arbitrary base). If p is an array, its values are used to specify proportions in 
  log-space to select the samples the [start,stop] interval. 
  
  If reverse = True, then the sequence returned is logarithimically scaled from the stop to start. 
  """
  p = np.linspace(0, 1, p, endpoint=True) if isinstance(p, Integral) else p
  ub = np.emath.logn(base, stop)
  lb = np.emath.logn(base, start) if start != 0.0 else np.emath.logn(base, np.finfo(np.float32).eps)
  logspaced = base ** (lb + p*(ub-lb))
  if start == 0.0:
    logspaced[0] = 0.0
  return logspaced if not reverse else start + (stop - logspaced)

class HeatKernel:
  ## "any parameter that can have a value assigned prior to having access to the data should be an __init__ keyword argument."
  ## TODO: Replace the laplacian solver parameters with actual parameters, e.g. ntime, bound, approx 
  def __init__(self, 
    timepoints: int = 32,
    approx: str = "unweighted",
    solver: str = "default",
    bounds: str = "informative"
  ): 
    self.timepoints = timepoints
    self.approx = approx 
    self.solver = solver 
    self.bounds = bounds

  def param_timepoints(self, timepoints: Union[int, np.ndarray] = None, bounds: str = "informative"):
    """Parameterizes the actual timepoints to evaluate the heat kernel over."""
    if isinstance(timepoints, Integral):
      lb, ub = self.time_bounds(bounds)
      # timepoints = dict(ntime=timepoints, bound="informative") # default 
      return geomspace(lb, ub, timepoints)
    elif isinstance(timepoints, Iterable):
      timepoints = np.fromiter(iter(timepoints), dtype=np.float64)
      assert np.all(timepoints >= 0), "Time points must be non-negative!"
      return timepoints
    else:
      raise ValueError(f"Invalid timepoints argument type '{type(timepoints)}' supplied")
      
  ## Parameterize the laplacian approximation
  @staticmethod
  def param_laplacian(approx: Union[str, LinearOperator] = "unweighted", X: ArrayLike = None, S: ComplexLike = None, use_triangles: bool = True, **kwargs):
    """Parameterizes the laplacian and mass operators, with input validation."""
    from scipy.sparse import spmatrix, sparray
    laplacian_kwargs = function_kwargs(up_laplacian, **kwargs)
    if isinstance(approx, LinearOperator) or isinstance(approx, sparray) or isinstance(approx, spmatrix):
      assert hasattr(approx, "diagonal"), f"Laplacian approximation operator '{approx}' must has .diagonal() method!"
      laplacian_ = approx
      d = laplacian_.diagonal()
      mass_matrix_ = diags(d) 
      return laplacian_, mass_matrix_
    elif approx == "unweighted":
      laplacian_, d = up_laplacian(S, p=0, return_diag=True, **laplacian_kwargs)
      mass_matrix_ = diags(d) 
      return laplacian_, mass_matrix_
    elif approx == "mesh":
      assert isinstance(X, np.ndarray), "Vertex positions must be supplied ('X') if 'mesh' approximation is used."
      fw = sx.lower_star_filter(gauss_similarity(S, X))
      laplacian_ = up_laplacian(S, p=0, weight=fw, **laplacian_kwargs)
      mass_matrix_ = diags(vertex_masses(S, X, use_triangles))     
      return laplacian_, mass_matrix_
    elif approx == "cotangent":
      raise NotImplementedError("Haven't implemented yet")
    else:
      raise ValueError("Invalid approximation choice {approx}; must be one of 'mesh', 'cotangent', or 'unweighted.'")

  ## Parameterize the eigenvalue solver
  @staticmethod
  def param_solver(solver: Union[str, Callable] = "default", shift_invert: bool = True, precon: str = None, **kwargs):
    # assert hasattr(self, "laplacian_"), "Must parameterize Laplacian first"
    if shift_invert and "sigma" not in kwargs: 
      kwargs |= dict(sigma=1e-6, which="LM")
    solver_kwargs = function_kwargs(PsdSolver, **(dict(method=solver, shift_invert=shift_invert, precon = None) | kwargs))
    return PsdSolver(laplacian=True, eigenvectors=True, **solver_kwargs)

  ## Fit + all the param_< attribute > methods enact side-effects, but should be idempotent! 
  def fit(self, X: ArrayLike = None, y: np.ndarray = None, **kwargs):
    """Computes the eigensets to use to construct (or approximate) the heat kernel
  
    Parameters: 
      X: positions of an (n x d) point cloud embedding in Euclidean space. Required for "mesh"  or "cotangent" approximation. 
      y: unused. 
      kwargs: additional keyword-arguments are forwarded to the param_laplacian(...) and param_solver(...) methods. 
    
    Returns: 
      self: returns the self instance modified with fitted 'laplacian_', 'mass_matrix_', 'eigvecs_', and 'eigvals_'
    """
    
    ## Parameterize the solver -- note we just pass kwargs as-is and leave terminal leaves to handle restrictions
    solver_kwargs = dict(solver=self.solver) | kwargs
    # solver_kwargs.update(function_kwargs(self.param_solver, kwargs))
    # solver_kwargs.update(function_kwargs(self.PsdSolver, kwargs))
    self.solver_ = self.param_solver(**solver_kwargs)                                  ## should be idempotent for fixed params + kwargs
    
    ## Parameterize the laplacian
    laplacian_kwargs = dict(approx=self.approx) | kwargs
    # laplacian_kwargs.update(function_kwargs(self.param_laplacian, kwargs))
    # laplacian_kwargs.update(function_kwargs(self.up_laplacian, kwargs))
    self.laplacian_, self.mass_matrix_ = self.param_laplacian(X=X, **laplacian_kwargs) ## should be idempotent for fixed params + kwargs
    assert hasattr(self, "laplacian_"), "Heat kernel must have laplacian parameterized"
    # self.approx_ = laplacian_kwargs.pop("approx", "unweighted")

    ## Do the call to the solver 
    solver_kwargs = dict(A=self.laplacian_, M=self.mass_matrix_) | function_kwargs(self.solver_, **kwargs)
    ew, ev = self.solver_(**solver_kwargs) ## solve Ax = \lambda M x 
    self.eigvecs_ = ev
    self.eigvals_ = np.maximum(ew, 0.0) ## enforce positive semi-definiteness
    return self

  def time_bounds(self, bound: str, interval: tuple = (0.0, 1.0), dtype = None):
    """Returns lower and upper bounds on the time parameter of the heat kernel, based on various heuristics.
    
    The heat kernel, and the various summary representations derived from it, are heavily dependent on the choice of time parameter, 
    which loosely translates into the time the system is diffuses heat from some initial condition. 

    This function tries to deduce (loose) lower and upper bounds to help set the time parameter effectively. Heuristics include: 
      "absolute" = returns [0, t_max], where t_max is the largest t one can expect to have any heat above machine precision.
      "laplacian" = uses gerschgorin theorem to bound spectral radius and spectral gap, then uses "heuristic"
    """
    # assert len(interval) == 2 and interval[0] >= 0.0 and interval[1] <= 1.0, "If supplied, interval bounds must be in [0,1]"
    if bound == "absolute":
      # assert hasattr(self, "laplacian_")
      # dtype = self.laplacian_.dtype
      dtype = np.float64 if dtype is None else dtype
      t_max = -np.log(np.finfo(dtype).eps)/1e-6 # 1e-6 assumed smallest eigenvalue can be found
      t_min = 1.0 # since log(0) will be -inf, though this could be go down to 0
      return t_min, t_max
    elif bound == "laplacian":
      assert hasattr(self, "laplacian_")
      L, dtype = self.laplacian_, self.laplacian_.dtype
      min_ew, max_ew = np.inf, 0.0
      if np.allclose(L.diagonal(), 1.0):
        max_ew = 2.0  ## the normalized laplacian is bounded in [0, p+1]
        min_ew = 1e-6 ## heuristic on spectral gap
      else:
        ## use gerschgorin theorem
        cv = np.zeros(L.shape[0])
        for i in range(L.shape[0]):
          cv[i-1], cv[i] = 0, 1
          row = L @ cv
          max_ew = max(row[i] + np.sum(np.abs(np.delete(row, i))), max_ew)
          min_ew = min(row[i] - np.sum(np.abs(np.delete(row, i))), min_ew)
        min_ew = 1e-6 if min_ew <= 0 else min_ew 
      l_min = min_ew / np.max(self.mass_matrix_.diagonal())
      l_max = max_ew / np.min(self.mass_matrix_.diagonal())
      t_min = 4 * np.log(10) / l_max
      t_max = min(4 * np.log(10) / l_min, -np.log(np.finfo(dtype).eps) / min_ew)
      return t_min, t_max
    elif bound == "effective":
      assert hasattr(self, "eigvals_"), "Must call .fit() first!"
      machine_eps = np.finfo(self.laplacian_.dtype).eps
      l_min, l_max = np.min(self.eigvals_[~np.isclose(self.eigvals_, 0.0)]), np.max(self.eigvals_)
      t_max = -np.log(machine_eps)/l_min
      t_min = -np.log(machine_eps)/(l_max - l_min)
      return t_min, t_max
    elif bound == "informative":
      assert hasattr(self, "eigvals_"), "Must call .fit() first!"
      machine_eps = np.finfo(self.laplacian_.dtype).eps
      # ew = np.sort(self.eigvals_)
      # l_min, l_max = ew[1], ew[-1]
      l_min, l_max = np.min(self.eigvals_[~np.isclose(self.eigvals_, 0.0)]), np.max(self.eigvals_)
      # l_min, l_max = np.quantile(self.eigvals_[1:], interval) ## TODO: revisit, if its a linearly spaced interval use could do themselves
      t_max = -np.log2(machine_eps) / l_min
      t_min = -np.log2(machine_eps) / (l_max - l_min)
      lmi, lmx = np.log2(t_min), np.log2(t_max)
      t_min = 2.0 ** (lmi + interval[0] * (lmx - lmi))
      t_max = 2.0 ** (lmi + interval[1] * (lmx - lmi))
      return t_min, t_max
    elif bound == "heuristic":
      assert hasattr(self, "eigvals_"), "Must call .fit() first!"
      l_min, l_max = np.min(self.eigvals_[~np.isclose(self.eigvals_, 0.0)]), np.max(self.eigvals_) #np.quantile(self.eigvals_, interval)
      t_min = 4 * np.log(10) / l_max
      t_max = 4 * np.log(10) / l_min
      return t_min, t_max
    else: 
      raise ValueError(f"Unknown time bound method '{bound}' supplied. Must be one ['absolute', 'laplacian', 'effective', 'informative', 'heuristic']")
    
  def diffuse(self, timepoints: ArrayLike = None, subset: ArrayLike = None, **kwargs) -> Generator:
    assert hasattr(self, "eigvecs_"), "Must call .fit() first!"
    I = np.arange(self.eigvecs_.shape[0]) if subset is None else np.array(subset)
    T = self.param_timepoints(self.timepoints, self.bounds) if timepoints is None else np.asarray(timepoints)
    for t in T:
      Ht = self.eigvecs_ @ np.diag(np.exp(-t*self.eigvals_)) @ self.eigvecs_.T
      yield Ht[:,I]

  def trace(self, timepoints: ArrayLike = None) -> np.ndarray:
    assert hasattr(self, "eigvals_"), "Must call .fit() first!"
    cind_nz = np.flatnonzero(~np.isclose(self.eigvals_, 0.0, atol=1e-14))
    T = self.param_timepoints(self.timepoints, self.bounds) if timepoints is None else np.asarray(timepoints)
    ht = np.array([np.sum(np.exp(-t * self.eigvals_[cind_nz])) for t in T])
    return ht

  def content(self, timepoints: ArrayLike = None, subset: ArrayLike = None, **kwargs) -> np.ndarray:
    # heat_content = np.array([hk.sum() for hk in self.diffuse(timepoints, subset, **kwargs)])
    assert hasattr(self, "eigvals_"), "Must call .fit() first!"
    T = self.param_timepoints(self.timepoints, self.bounds) if timepoints is None else np.asarray(timepoints)
    hkc = self.trace(T)
    hkc = np.append(hkc[0], hkc[0] + np.cumsum(np.abs(np.diff(hkc))))
    return hkc

  def signature(self, timepoints: ArrayLike = None, scaled: bool = True, subset: ArrayLike = None) -> np.ndarray:
    assert hasattr(self, "eigvecs_"), "Must call .fit() first!"
    T = self.param_timepoints(self.timepoints, self.bounds) if timepoints is None else np.asarray(timepoints)
    I = np.arange(self.eigvecs_.shape[0]) if subset is None else np.array(subset)
    cind_nz = np.flatnonzero(~np.isclose(self.eigvals_, 0.0, atol=1e-14))
    ev_subset = np.square(self.eigvecs_[np.ix_(I, cind_nz)]) if subset is not None else np.square(self.eigvecs_[:,cind_nz])
    hks_matrix = np.array([ev_subset @ np.exp(-t*self.eigvals_[cind_nz]) for t in T]).T
    if scaled: 
      ht = np.array([np.sum(np.exp(-t * self.eigvals_[cind_nz])) for t in T]) # heat trace
      ht = np.reciprocal(ht, where=~np.isclose(ht, 0.0, atol=1e-14))
      hks_matrix = hks_matrix @ diags(ht)
    return hks_matrix
  
  # def estimator(self, method: str = "signature"):
  #   """Constructs an sklearn-compatible e""

  def sihks(self, timepoints: ArrayLike = None, alpha: float = 2.0):
    assert hasattr(self, "eigvecs_"), "Must call .fit() first!"
    ## TODO: choose alpha such that alpha**max(timepoints) is representeable as a number, or maybe such that 
    ## -max(ew)*alpha*max(t) ~= -37, say
    T = self.param_timepoints(self.timepoints, self.bounds) if timepoints is None else np.asarray(timepoints)
    n, c = self.eigvecs_.shape[0], np.log(alpha)
    sc = []
    ew, ev = self.eigvals_, np.square(self.eigvecs_) ## note the elementwise squaring
    for a_t in (alpha ** T):
      # scaling = np.tile(alpha**t * ew.T * np.exp(-alpha**t * ew.T), (n, 1))
      scaling = a_t * ew * np.exp(-a_t * ew)
      if all(np.abs(scaling) < 1e-15) or any(np.isnan(scaling)):
        ## alpha**t can very quickly go to zero, so break if that happens
        break
      sc.append(-c * np.sum(ev * scaling, axis=1) / np.sum(ev * np.exp(-a_t * ew), axis=1))
    sc = np.array(sc).T
    si = np.abs(np.fft.fft(sc, axis=1))
    return si, sc

  def __repr__(self) -> str:
    # " ".join([f"{t:.2e}" for t in self.timepoints]).replace("+","")
    s = f"Heat kernel (approx = {self.approx}, timepoints = {self.timepoints}, bounds = {self.bounds})\n"
    s += f"Solver: {str(self.solver)}"
    return s 

  
  def clone(self):
    """Returns a clone of this instance with the same parameters, discarding fitted information (if any).""" 
    hk_clone = HeatKernel() # shallow copy
    # hk_clone.timepoints = self.timepoints
    hk_clone.timepoints = self.timepoints
    # hk_clone.laplacian = self.laplacian 
    hk_clone.solver = self.solver 
    hk_clone.bounds = self.bounds 
    hk_clone.approx = self.approx
    return hk_clone


# def heat_kernel_signature(L: LinearOperator, A: LinearOperator, timepoints: Union[int, ArrayLike] = 10, scaled: bool = True, subset = None, **kwargs):
#   """Constructs the heat kernel signature of a given Laplacian operator at time points T. 
  
#   For a subset S \subseteq V of vertices, this returns a  |S| x |T| signature. 
#   """
#   from pbsig.linalg import eigh_solver
#   solver = eigh_solver(L, laplacian=True)
#   ew, ev = solver(L, M=A, which="SM", **kwargs) ## solve generalized eigenvalue problem
#   timepoints = logspaced_timepoints(timepoints) if isinstance(timepoints, Integral) else timepoints
#   assert isinstance(timepoints, np.ndarray), "timepoints must be an array."
#   cind_nz = np.flatnonzero(~np.isclose(ew, 0.0, atol=1e-14))
#   ev_subset = np.square(ev[np.array(subset), cind_nz]) if subset is not None else np.square(ev[:,cind_nz])
#   hks_matrix = np.array([ev_subset @ np.exp(-t*ew[cind_nz]) for t in timepoints]).T
#   if scaled: 
#     ht = np.array([np.sum(np.exp(-t * ew[cind_nz])) for t in timepoints]) # heat trace
#     ht = np.reciprocal(ht, where=~np.isclose(ht, 0.0))
#     hks_matrix = hks_matrix @ diags(ht)
#   return hks_matrix
    
# def diffuse_heat(evecs: ArrayLike, evals: ArrayLike, timepoints: Union[int, ArrayLike] = 10, subset = None):
#   timepoints = logspaced_timepoints(timepoints) if isinstance(timepoints, Integral) else timepoints
#   I = np.arange(evecs.shape[0]) if subset is None else np.array(subset)
#   for t in timepoints:
#     Ht = evecs @ np.diag(np.exp(-t*evals)) @ evecs.T
#     yield Ht[:,I]

# def heat_kernel_trace(L: LinearOperator, timepoints: Union[int, ArrayLike] = 10, **kwargs):
#   """Computes the heat kernel trace of a gievn Laplacian operators at time points T.
#   """
#   from pbsig.linalg import eigvalsh_solver
#   ew = eigvalsh_solver(L)(L, sigma=1e-6, which="LM", **kwargs)
#   timepoints = logspaced_timepoints(timepoints) if isinstance(timepoints, Integral) else timepoints
#   assert isinstance(timepoints, np.ndarray), "timepoints must be an array." 
#   return np.array([np.sum(np.exp(-t*ew)) for t in timepoints])

class PsdSolver:
  def __init__(self, method: str = 'default', k: Union[int, str, float, list] = "auto", laplacian: bool = False, eigenvectors: bool = False, tol: float = 0.0, **kwargs):
    assert method is not None, f"Invalid solver method '{method}'"
    # self._rank_bound = int(rank_ub) if isinstance(rank_ub, Integral) else lambda A: rank_bound(A, upper=True)
    self.method: Union[tuple, str, Callable] = method
    self.k: Union[str, int] = k ## have to infer this, these are solver parms and prob should go in solver! 
    self.tol = np.max([tol, kwargs.pop("tolerance", 0.0)])
    # self.tol = tol if tol is not None else np.sqrt(np.finfo(np.float64).eps)
    self.laplacian = laplacian
    self.eigenvectors = eigenvectors | kwargs.pop("return_eigenvectors", False)
    # if "return_eigenvectors" in kwargs:
    #   self.eigenvectors = (eigenvectors | kwargs['return_eigenvectors']) 
    #   kwargs.pop("return_eigenvectors", None)
    # else: 
    #   self.eigenvectors = eigenvectors
    self.params = kwargs
    # for key, value in kwargs.items():
    #   setattr(self, key, value)
    # dict(tol=self.tolerance, return_eigenvectors=eigenvectors, k=)
    
  ## Turns a data-independent parameter setting + data into a realized
  def param_method(self, A: Union[ArrayLike, spmatrix, LinearOperator], method: Union[str, Callable] = 'default'):
    # if solver != 'dac': assert isinstance(A, np.ndarray), f"Cannot use iterative methods with dense of type '{type(A)}'"
    if isinstance(A, np.ndarray) and method == 'default' or method == 'dac':
      if method == 'dac': assert isinstance(A, np.ndarray), f"Cannot use divide-and-conquer with operators of type '{type(A)}'"
      method = np.linalg.eigh if self.eigenvectors else np.linalg.eigvalsh
      return method, {}
    elif issparse(A) or isinstance(A, LinearOperator):
      # if nev == A.shape[0] and (solver == 'irl' or solver == 'default'):
      #   import warnings
      #   warnings.warn("Switching to PRIMME, as ARPACK cannot estimate all eigenvalues without shift-invert")
      #   solver = 'jd'
      method = 'irl' if method == 'default' else method
      assert isinstance(method, str) and method in ['default', 'irl', 'lanczos', 'gd', 'jd', 'lobpcg']
      if method == 'irl':
        from scipy.sparse.linalg import eigsh
        return eigsh, dict(which='LM', tol=self.tol, return_eigenvectors=self.eigenvectors)
      else:
        from primme import eigsh as primme_eigsh
        methods = { 'lanczos' : 'PRIMME_Arnoldi', 'gd': "PRIMME_GD" , 'jd' : "PRIMME_JDQR", 'lobpcg' : 'PRIMME_LOBPCG_OrthoBasis', 'default' : 'PRIMME_DEFAULT_MIN_TIME' }
        return primme_eigsh, dict(tol=self.tol, which='LM', return_eigenvectors=self.eigenvectors, method=methods[method])
    else: 
      raise ValueError(f"Invalid solver method / operator-type combination ({method}/{str(type(A))}) given")
  
  ## Parameterize 
  def param_k(self, A: Union[ArrayLike, spmatrix, LinearOperator], k: Union[int, str, float, list] = "auto") -> int:
    if isinstance(k, Integral): 
      return k
    elif isinstance(k, Number) and k >= 0.0 and k <= 1.0:
      if k == 0.0: return 0
      k = trace_threshold(A, k) if k != 1.0 else rank_bound(A, upper=True)
      return min(self.k, A.shape[0] - 1) # if self.laplacian else nev
    elif isinstance(k, str) and k == "auto":
      k = rank_bound(A, upper=True)
      return min(k, A.shape[0] - 1) # if self.laplacian else nev
    else: 
      raise ValueError("Couldn't deduce number of eigenvalues to compute.")


  ## Equivalent of .fit() 
  def __call__(self, A: Union[ArrayLike, sparray, LinearOperator], **kwargs) -> Union[ArrayLike, tuple]:
    assert A.shape[0] == A.shape[1], "A must be square"
    n = A.shape[0]
    if np.all(np.array(A.shape) == 0): 
      return (np.zeros(1), np.c_[np.zeros(n)]) if self.eigenvectors else np.zeros(1)
    self.k_ = self.param_k(A, kwargs.pop("k", self.k))
    self.solver_, defaults_ = self.param_method(A, kwargs.pop("method", self.method))
    if (issparse(A) and len(A.data) == 0) or (n == 0 or A.shape[1] == 0) or self.k_ == 0: 
      return (np.zeros(1), np.c_[np.zeros(n)]) if self.eigenvectors else np.zeros(1)
    if isinstance(A, spmatrix) or isinstance(A, LinearOperator): 
      defaults_ |= dict(k=self.k_, maxiter=n*100, ncv=None)
      # params = (default_params | params) | kwargs  # if 'scipy' in self.solver.__module__ else None #min(2*nev + 1, 20) 
    self.params_ = (defaults_ | self.params) | kwargs
    return self.solver_(A, **self.params_)
  
  def __repr__(self):
    format_val = lambda v: str(v) if isinstance(v, Integral) else (f"{v:.2e}" if isinstance(v, Number) else str(v))
    format_dict = lambda d: "(" + ", ".join([f"{str(k)} = {format_val(v)}" for k,v in d.items()]) + ")"
    s = "Laplacian" if self.laplacian else ""
    print_params = self.params | dict(method=self.method, k=self.k, tol=self.tol, eigenvectors=self.eigenvectors)
    s += f"PsdSolver{format_dict(print_params)}:\n"
    if hasattr(self, "solver_"):
      s += f"Solver type: (module = {str(self.solver_.__module__)}, name = {str(self.solver_.__name__)})\n"
    if hasattr(self, "params_") and self.params_ != {}:
      s += f"Called with: {format_dict(self.params_)}\n"
    return s

  def test_accuracy(self, A: Union[ArrayLike, spmatrix, LinearOperator] = None, **kwargs):
    assert self.eigenvectors, "Solver must be parameterized with eigenvectors=True to test accuracy."
    ew, ev = self(A, **kwargs)
    diffs = A @ ev - ev * ew
    maxdiffs = np.linalg.norm(diffs, axis=0, ord=np.inf)
    print(f"|Av - λv|_max: {float(np.max(maxdiffs)):.6f}")
    return diffs

  # def test_speed(self, A: Union[ArrayLike, spmatrix, LinearOperator], **kwargs):



# def eigsort(w: ArrayLike, v: ArrayLike = None, ascending: bool = True):
#   """
  
#   Based on: https://web.cecs.pdx.edu/~gerry/nmm/mfiles/eigen/eigSort.m
#   """
#   np.argsort(w)


## Great advice: https://gist.github.com/denis-bz/2658f671cee9396ac15cfe07dcc6657d 
## TODO: change p to be accepted at runtime from returned Callable, accept argument that has heuristic 'rank_estimate'  
## that parameterizes how to compute the rank bound 
# def polymorphic_psd_solver(A: Union[ArrayLike, spmatrix, LinearOperator], pp: float = 1.0, solver: str = 'default', laplacian: bool = True, return_eigenvectors: bool = False, **kwargs):
#   """Configures an eigen-solver for a symmetric positive semi-definite linear operator _A_ (optionally coming from a laplacian).

#   _A_ can be a dense np.array, any of the sparse arrays from SciPy, or a _LinearOperator_. The default solver 'default' chooses 
#   the solver dynamically based on the size, structure, and estimated rank of _A_. In particular, if 'A' is dense, then the divide-and-conquer 
#   approach is chosen. 

#   The solver can also be specified explicitly as one of:
#     'dac' <=> Divide-and-conquer (via LAPACK routine 'syevd')
#     'irl' <=> Implicitly Restarted Lanczos (via ARPACK)
#     'lanczos' <=> Lanczos Iteration (non-restarted, w/ PRIMME)
#     'gd' <=> Generalized Davidson w/ robust shifting (via PRIMME)
#     'jd' <=> Jacobi Davidson w/ Quasi Minimum Residual (via PRIMME)
#     'lobpcg' <=> Locally Optimal Block Preconditioned Conjugate Gradient (via PRIMME)

#   Parameters: 
#     A: array, sparse matrix, or linear operator to compute the rank of
#     pp: proportional part of the spectrum to compute. 
#     solver: One of 'default', 'dac', 'irl', 'lanczos', 'gd', 'jd', or 'lobpcg'.
#     laplacian: whether _A_ represents a Laplacian operator. Defaults to True. 
#     return_eigenvector: whether to return both eigen values and eigenvector pairs, or just eigenvalues. Defaults to False. 
#   Returns:
#     a callable f(A, ...) that returns spectral information for psd operators _A_

#   Notes:
#     For fast parameterization, _A_ should support efficient access to its diagonal via a method A.diagonal()
#   """
#   assert A.shape[0] == A.shape[1], "A must be square"
#   assert pp >= 0.0 and pp <= 1.0, "Proportion 'pp' must be between [0,1]"
#   assert solver is not None, "Invalid solver"
#   tol = kwargs['tol'] if 'tol' in kwargs.keys() else np.finfo(A.dtype).eps
#   if pp == 0.0: 
#     if return_eigenvectors:
#       return lambda x: (np.zeros(1), np.c_[np.zeros(A.shape[0])])
#     else: 
#       return lambda x: np.zeros(1)
#   if solver == 'dac':
#     assert isinstance(A, np.ndarray), "Cannot use divide-and-conquer with linear operators"
#   if isinstance(A, np.ndarray) and solver == 'default' or solver == 'dac':
#     params = kwargs
#     solver = np.linalg.eigh if return_eigenvectors else np.linalg.eigvalsh
#   elif isinstance(A, spmatrix) or isinstance(A, LinearOperator):
#     if isinstance(A, spmatrix) and np.allclose(A.data, 0.0): 
#       if return_eigenvectors:
#         return lambda x: (np.zeros(1), np.c_[np.zeros(A.shape[0])])
#       else:
#         return(lambda A: np.zeros(1))
#     if A.shape[0] == 0 or A.shape[1] == 0: 
#       if return_eigenvectors:
#         return lambda x: (np.zeros(1), np.c_[np.zeros(A.shape[0])])
#       else:
#         return(lambda A: np.zeros(1))
#     nev = trace_threshold(A, pp) if pp != 1.0 else rank_bound(A, upper=True)
#     nev = min(nev, A.shape[0] - 1) if laplacian else nev
#     if nev == 0: return(lambda A: np.zeros(1))
#     if nev == A.shape[0] and (solver == 'irl' or solver == 'default'):
#       import warnings
#       warnings.warn("Switching to PRIMME, as ARPACK cannot estimate all eigenvalues without shift-invert")
#       solver = 'jd'
#     solver = 'irl' if solver == 'default' else solver
#     assert isinstance(solver, str) and solver in ['default', 'irl', 'lanczos', 'gd', 'jd', 'lobpcg']
#     if solver == 'irl':
#       params = dict(k=nev, which='LM', tol=tol, return_eigenvectors=return_eigenvectors) | kwargs
#       solver = eigsh
#     else:
#       import primme
#       n = A.shape[0]
#       ncv = min(2*nev + 1, 20) # use modification of scipy rule
#       methods = { 'lanczos' : 'PRIMME_Arnoldi', 'gd': "PRIMME_GD" , 'jd' : "PRIMME_JDQR", 'lobpcg' : 'PRIMME_LOBPCG_OrthoBasis', 'default' : 'PRIMME_DEFAULT_MIN_TIME' }
#       params = dict(ncv=ncv, maxiter=pp*n*100, tol=tol, k=nev, which='LM', return_eigenvectors=return_eigenvectors, method=methods[solver]) | kwargs
#       solver = primme.eigsh
#   else: 
#     raise ValueError(f"Invalid solver / operator-type {solver}/{str(type(A))} given")
#   def _call_solver(A, **kwargs):
#     return solver(A, **(params | kwargs)) 
#   return _call_solver

# def eigh_solver(A: Union[ArrayLike, spmatrix, LinearOperator], **kwargs):
#   #return polymorphic_psd_solver(A, return_eigenvectors=True, **kwargs)
#   return PsdSolver(A, return_eigenvectors=True, **kwargs)

# def eigvalsh_solver(**kwargs) -> Callable:
#   #return polymorphic_psd_solver(A, return_eigenvectors=False, **kwargs)
  
#   return PsdSolver(A, return_eigenvectors=False, **kwargs)

def eigen_dist(x: np.ndarray, y: np.ndarray, p: int = 2, method: str = "relative"):
  """Computes a variety of distances between sets of eigenvalues. """
  if method == "relative":
    n = max(len(x), len(y))
    a,b = np.zeros(n), np.zeros(n)
    a[:len(x)] = np.sort(x)
    b[:len(y)] = np.sort(y)
    denom = (np.abs(a)**p + np.abs(b)**p)**(1/p)
    return np.sum(np.where(np.isclose(denom, 0, atol=1e-15), 0, np.abs(a-b)/denom))
  else: 
    raise NotImplementedError("Haven't implemented other distances yet.")

def spectral_rank(ew: ArrayLike = None, method: int = 0, shape: tuple = None, prec: float = None, vector_valued: bool = False) -> int:
  ## Closure on the parameters
  prec = np.finfo(np.float64).eps if prec is None else prec
  reso = np.finfo(np.float64).resolution
  def _numerical_rank(x: np.ndarray):
    x = np.atleast_1d(x) if not isinstance(x, np.ndarray) else x
    if len(x) == 0: return 0
    m, n = shape if (shape is not None and len(shape) == 2) else (len(x), len(x))
    if method == 0:
      ## Default: identifies only those as positive cannot have been introduced by numerical errors 
      x = np.where(x < reso, 0.0, x) ## NOTE: this differs from numpy
      tol = x.max() * max(m,n) * prec
      return sum(x > tol) if not vector_valued else np.array(x > tol, dtype=int)
    elif method == 1: 
      ## Minimizes expected roundoff error
      x = np.where(x < reso, 0.0, x) ## NOTE: this differs from numpy
      tol = x.max() * prec / 2. * np.sqrt(m + n + 1.)
      return sum(x > tol) if not vector_valued else np.array(x > tol, dtype=int)
    elif method == 2:
      ## Dynamic method: tests multiple thresholds for detecting rank deficiency up to the prescribed precision, 
      ## choosing the one that appears the most frequently / seems to be the most stable
      x = np.where(x < reso, 0.0, x) ## NOTE: this differs from numpy
      C = 1.0/np.exp(np.linspace(1,36,100))
      i = len(C)-np.searchsorted(np.flip(C), prec)
      C = C[:i]
      nr = [sum(x/(x + c)) for c in C]
      values, counts = np.unique(np.round(nr).astype(int), return_counts=True)
      return values[np.argmax(counts)] if not vector_valued else (x / (x + C[np.argmax(counts)]))
    else:
      raise ValueError("Invalid method chosen")
  return _numerical_rank(ew) if ew is not None else _numerical_rank

def pseudoinverse(x: ArrayLike, **kwargs) -> np.ndarray:
  x = np.array(x) if not(isinstance(x, np.ndarray)) else x
  if x.ndim == 2:
    return np.linalg.pinv(x)
  else:
    close_to_zero = np.isclose(x, 0.0, **kwargs)
    x = np.reciprocal(x, where=~close_to_zero)
    x[close_to_zero] = 0.0
    return x
  # pseudo = lambda x: np.reciprocal(x, where=~np.isclose(x, 0)) # scalar pseudo-inverse
  
  # if isinstance(x, np.ndarray):
  #   return np.linalg.pinv(x) if x.ndim == 2 else pseudo(x)
  # else: 
  #   return np.array([pseudo(xi) for xi in x])
  

def prox_nuclear(x: ArrayLike, t: float = 1.0):
  """Prox operator on the nuclear norm.
  
  prox_f returns the (unique) point that achieves the infimum defining the _Moreau envelope_ _Mf_.

  prox_{t*f}(x) = argmin_z { f(z) + (1/2t) || x - z ||^2 }
                = argmin_z { ||z||_* + (1/2t) || x - z ||_F^2 }

  where X = U S V^T

  Smaller values of _t_ will yield operators closer to the original nuclear norm.
  """
  solver = eigh_solver(x)
  ew, ev = solver(x)
  # sv = np.maximum(ew, 0.0)                  ## singular values (sqrt?)
  sw_prox = np.maximum(ew - t, 0.0)                     ## soft-thresholded singular values
  A = ev @ diags(np.array(sw_prox)) @ ev.T                        ## prox operator 
  proj_d = (1/(2*t))*np.linalg.norm(A - x, 'fro')**2      ## projection distance (rhs)
  me = sum(abs(sw_prox)) + proj_d                            ## Moreau envelope value
  return A, me, ew

def prox_sgn_approx(x: ArrayLike, t: float = 1.0, **kwargs):
  """ Proximal operator for sgn approximation function"""
  from scipy.optimize import minimize
  sf = sgn_approx(**kwargs)  
  solver = eigh_solver(x)
  ew, ev = solver(x)
  minimize(rosen, x0)


def numerical_rank(A: Union[ArrayLike, spmatrix, LinearOperator], tol: float = None, solver: str = 'default', **kwargs) -> int:
  """ 
  Computes the numerical rank of a positive semi-definite 'A' via thresholding its eigenvalues.

  The eps-numerical rank r_eps(A) is the largest integer 'r' such that: 
  
  s1 >= s2 >= ... >= sr >= eps >= ... >= 0

  where si represent the i'th largest singular value of A. 

  See: https://math.stackexchange.com/questions/2238798/numerically-determining-rank-of-a-matrix
  """
  if isinstance(A, np.ndarray):
    if np.allclose(A, 0.0) or prod(A.shape) == 0: return 0
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

from typing import * 
def eigsh_block_shifted(A, k: int, b: int = 10, locking: bool = False, **kwargs) -> Generator:
  """
  Calculates all eigenvalues in a block-wise manner 
  """
  from primme import eigsh as primme_eigsh
  ni = int(np.ceil(k/b))
  f_args = dict(which='CGT', return_eigenvectors=True, raise_for_unconverged=False)
  f_args = f_args | kwargs
  for i in range(ni):
    if i == 0:
      ew, ev = primme_eigsh(A, k=b, **f_args)
    else: 
      #f_args['which'] = min(ew)
      f_args['sigma'] = min(ew)
      f_args['lock'] = ev if locking else None
      ew, ev = primme_eigsh(A, k=b, **f_args)
    yield ew, ev


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
  if pp == 0.0: return 0
  d = np.array(diagonal(A))
  d = d[d > 1e-15]
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
  from primme import eigsh as primme_eigsh
  default_opts = dict(tol=1e-6, printLevel=0, return_eigenvectors=False, return_stats=False, return_history=False) | kwargs
  args = {} 
  reduce_f, is_aggregate = (lambda x: x, False) if reduce is None else (reduce, True)
  for A in M: 
    A, args = (A[0], args | A[1]) if isinstance(A, tuple) and len(A) == 2 else (A, args)
    nev = trace_threshold(A, p)
    if default_opts['return_stats']:
      if default_opts['return_eigenvectors']:
        ew, ev, stats = primme_eigsh(A=A, k=nev, **default_opts)
        yield reduce_f(ew), ev, stats
      else:
        ew, stats = primme_eigsh(A=A, k=nev, **default_opts)
        yield reduce_f(ew), stats
    else: 
      if default_opts['return_eigenvectors']:
        ew, ev, stats = primme_eigsh(A=A, k=nev, **default_opts)
        yield reduce_f(ew), ev
      else:
        ew = primme_eigsh(A=A, k=nev, **default_opts)
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
  Creates a projector that projects onto the intersection of spaces (A,B), where A and B are linear subspaces.
  
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




