from typing import * 
from numpy.typing import ArrayLike
from numbers import *

import numpy as np 
from math import *
from scipy.sparse import *
from scipy.sparse.linalg import * 
from scipy.sparse.csgraph import *
from splex.combinatorial import rank_combs
from more_itertools import collapse

## Local imports
from .meta import *
from .simplicial import * 
from .combinatorial import * 
from .distance import dist 

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

def huber(x: ArrayLike = None, delta: float = 1.0) -> ArrayLike:
  def _huber(x: ArrayLike): 
    return np.where(np.abs(x) <= delta, 0.5 * (x ** 2), delta * (np.abs(x) - 0.5*delta))
  return _huber if x is None else _huber(x)

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


def logspaced_timepoints(n: int, lb: float = 0.0, ub: float = 1.0, method: str = "full") -> np.ndarray:
  """Constructs _n_ non-negative time points equi-distant in log-space for use in the map exp(-t).
  
  If an upper-bound for time is known, it may be specified so as to map the interval [0, ub] to the range
  such that np.exp(-t0) = 1 and np.exp(-tn) = epsilon, which epsilon is the machine epsilon.
  """
  ## TODO: revisit the full heuristic! The localized method works so much better
  if method == "full":
    timepoints = np.geomspace(lb+1.0, 37/ub, num=n)-1.0 ## -1 to include 0
  elif method == "local":
    assert lb != 0.0, "Local heuristic require positive lower-bound for spectral gap"
    tmin = 4 * np.log(10) / ub
    tmax = 4 * np.log(10) / lb
    timepoints = np.geomspace(tmin, tmax, n)
  else:
    raise ValueError(f"Unknown heuristic method '{method}' passed. Must be one 'local' or 'full'")
  return timepoints

def vertex_masses(S: ComplexLike, X: ArrayLike) -> np.ndarray:
  """Computes the cumulative area or 'mass' around every vertex"""
  # TODO: allow area to be computed via edge lengths?
  from pbsig.shape import triangle_areas
  areas = triangle_areas(S, X)
  vertex_mass = np.zeros(card(S,0))
  for t, t_area in zip(faces(S,2), areas):
    vertex_mass[t] += t_area / 3.0
  return vertex_mass


def gauss_similarity(S: ComplexLike, X: ArrayLike, sigma: Union[str, float] = "default", **kwargs) -> np.ndarray:
  """Gaussian kernel similarity function"""
  E_ind = np.array(list(faces(S,1)))
  E_dist = dist(X[E_ind[:,0]], X[E_ind[:,1]], paired=True, **kwargs)
  o = np.mean(E_dist) if sigma == "default" else float(sigma)
  a = 1.0 / (4 * np.pi * o**2)
  w = a * np.exp(-E_dist**2 / (4.0*o))
  return w

class HeatKernel:
  def __init__(self, complex: ComplexLike):
    self.complex = complex    ## TODO: make the simplicial complex shallow-copeable      
    self._timepoints = (32, "log-spaced")
    # self.laplacian = None 
    # self.solver = None 
    # self._solver = None  ## TODO: make the PsdSolver *not* specific a matrix, despite previous contracts! 

  @property
  def timepoints(self) -> np.ndarray:
    if isinstance(self._timepoints, np.ndarray):
      return self._timepoints
    else:
      assert hasattr(self, "eigvals_"), "Must call .fit() first!"
      ntime, heuristic = self._timepoints
      lb, ub = np.sort(self.eigvals_)[1], np.max(self.eigvals_)
      return logspaced_timepoints(ntime, lb, ub, method="local")

  @timepoints.setter
  def timepoints(self, timepoints): 
    if isinstance(timepoints, Container):
      timepoints = np.array(timepoints)
      assert np.all(timepoints >= 0), "Time points must be non-negative!"
      self._timepoints = timepoints
    elif isinstance(timepoints, Integral):
      self._timepoints = (timepoints, "log-spaced")
    elif isinstance(timepoints, tuple):
      assert len(timepoints) == 2 and isinstance(timepoints[0], Integral) and isinstance(timepoints, str), "timepoints must be a tuple (num_time_points, heuristic)" 
      self._timepoints = timepoints
    else:
      raise ValueError(f"Invalid timepoints form {type(timepoints)} given. Must be array, integer, or tuple (num timepoints, heuristic)")

  ## TODO: consider revisiting variant of the property pattern
  ## Note: JS version using @property not possible due to this: https://stackoverflow.com/questions/13029254/python-property-callable
  # @property
  # def laplacian(self):
  #   return self._laplacian

  # @laplacian.setter(self, value):
  #   self._laplacian = value 

  ## Parameterize the laplacian approximation
  def param_laplacian(self, approx: str = "unweighted", X: ArrayLike = None, **kwargs):
    if approx == "unweighted":
      self.laplacian_, d = up_laplacian(self.complex, p=0, return_diag=True, **kwargs)
      self.mass_matrix_ = diags(d) 
      return self
    elif approx == "mesh":
      assert isinstance(X, np.ndarray), "Vertex positions must be supplied ('X') if 'mesh' approximation is used."
      self.laplacian_ = up_laplacian(self.complex, p=0, weight=gauss_similarity(self.complex, X), **kwargs)
      self.mass_matrix_ = diags(vertex_masses(self.complex, X))     
      return self 
    elif approx == "cotangent":
      raise NotImplementedError("Haven't implemented yet")
    else:
      raise ValueError("Invalid approximation choice {approx}; must be one of 'mesh', 'cotangent', or 'unweighted.'")

  ## Parameterize the eigenvalue solver
  def param_solver(self, shift_invert: bool = True, precon: str = None, **kwargs):
    assert hasattr(self, "laplacian_"), "Must parameterize Laplacian first"
    if shift_invert: 
      kwargs |= dict(sigma=1e-6, which="LM")
    self.solver_ = PsdSolver(laplacian=True, eigenvectors=True, **kwargs)
    return self

  def fit(self, **kwargs):
    """Computes the eigensets to use to construct (or approximate) the heat kernel
    
    Parameters: 
      X: positions of an (n x d) point cloud embedding in Euclidean space. 
    """
    if not hasattr(self, "laplacian_"):
      self.param_laplacian("unweighted")
    if not hasattr(self, "solver_"):
      self.param_solver(shift_invert=True)
    # self.laplacian_ = self.param_laplacian("unweighted")  else self.laplacian_
    # self.solver_ =  if not hasattr(self, "solver_") else self.solver_
    assert hasattr(self, "mass_matrix_"), "Heat kernel must have laplacian parameterized"
    ew, ev = self.solver_(A=self.laplacian_, M=self.mass_matrix_, **kwargs) ## solve Ax = \lambda M x 
    self.eigvecs_ = ev
    self.eigvals_ = np.maximum(ew, 0.0)
    return self
  
  def diffuse(self, timepoints: ArrayLike = None, subset: ArrayLike = None, **kwargs) -> Generator:
    assert hasattr(self, "eigvecs_"), "Must call .fit() first!"
    I = np.arange(self.eigvecs_.shape[0]) if subset is None else np.array(subset)
    T = self.timepoints if timepoints is None else timepoints
    for t in T:
      Ht = self.eigvecs_ @ np.diag(np.exp(-t*self.eigvals_)) @ self.eigvecs_.T
      yield Ht[:,I]

  def trace(self, timepoints: ArrayLike = None) -> np.ndarray:
    assert hasattr(self, "eigvals_"), "Must call .fit() first!"
    cind_nz = np.flatnonzero(~np.isclose(self.eigvals_, 0.0, atol=1e-14))
    T = self.timepoints if timepoints is None else timepoints
    ht = np.array([np.sum(np.exp(-t * self.eigvals_[cind_nz])) for t in T])
    return ht

  def content(self, timepoints: ArrayLike = None, subset: ArrayLike = None, **kwargs) -> np.ndarray:
    assert hasattr(self, "eigvals_"), "Must call .fit() first!"
    hkc = self.trace(timepoints)
    hkc = np.append(hkc[0], hkc[0] + np.cumsum(np.abs(np.diff(hkc))))
    # heat_content = np.array([hk.sum() for hk in self.diffuse(timepoints, subset, **kwargs)])
    return hkc

  def signature(self, timepoints: ArrayLike = None, scaled: bool = True, subset: ArrayLike = None) -> np.ndarray:
    assert hasattr(self, "eigvecs_"), "Must call .fit() first!"
    T = self.timepoints if timepoints is None else timepoints
    I = np.arange(self.eigvecs_.shape[0]) if subset is None else np.array(subset)
    cind_nz = np.flatnonzero(~np.isclose(self.eigvals_, 0.0, atol=1e-14))
    ev_subset = np.square(self.eigvecs_[np.ix_(I, cind_nz)]) if subset is not None else np.square(self.eigvecs_[:,cind_nz])
    hks_matrix = np.array([ev_subset @ np.exp(-t*self.eigvals_[cind_nz]) for t in T]).T
    if scaled: 
      ht = np.array([np.sum(np.exp(-t * self.eigvals_[cind_nz])) for t in T]) # heat trace
      ht = np.reciprocal(ht, where=~np.isclose(ht, 0.0, atol=1e-14))
      hks_matrix = hks_matrix @ diags(ht)
    return hks_matrix

  def __repr__(self) -> str:
    return f"Heat kernel for {str(self.complex)}"
  
  def clone(self):
    """Returns a clone of this instance with the same parameters, discarding fitted information (if any).""" 
    hk_clone = HeatKernel(self.complex) # shallow copy
    hk_clone._timepoints = self._timepoints
    hk_clone.solver = self.solver
    return hk_clone


def heat_kernel_signature(L: LinearOperator, A: LinearOperator, timepoints: Union[int, ArrayLike] = 10, scaled: bool = True, subset = None, **kwargs):
  """Constructs the heat kernel signature of a given Laplacian operator at time points T. 
  
  For a subset S \subseteq V of vertices, this returns a  |S| x |T| signature. 
  """
  from pbsig.linalg import eigh_solver
  solver = eigh_solver(L, laplacian=True)
  ew, ev = solver(L, M=A, which="SM", **kwargs) ## solve generalized eigenvalue problem
  timepoints = logspaced_timepoints(timepoints) if isinstance(timepoints, Integral) else timepoints
  assert isinstance(timepoints, np.ndarray), "timepoints must be an array."
  cind_nz = np.flatnonzero(~np.isclose(ew, 0.0, atol=1e-14))
  ev_subset = np.square(ev[np.array(subset), cind_nz]) if subset is not None else np.square(ev[:,cind_nz])
  hks_matrix = np.array([ev_subset @ np.exp(-t*ew[cind_nz]) for t in timepoints]).T
  if scaled: 
    ht = np.array([np.sum(np.exp(-t * ew[cind_nz])) for t in timepoints]) # heat trace
    ht = np.reciprocal(ht, where=~np.isclose(ht, 0.0))
    hks_matrix = hks_matrix @ diags(ht)
  return hks_matrix
    
def diffuse_heat(evecs: ArrayLike, evals: ArrayLike, timepoints: Union[int, ArrayLike] = 10, subset = None):
  timepoints = logspaced_timepoints(timepoints) if isinstance(timepoints, Integral) else timepoints
  I = np.arange(evecs.shape[0]) if subset is None else np.array(subset)
  for t in timepoints:
    Ht = evecs @ np.diag(np.exp(-t*evals)) @ evecs.T
    yield Ht[:,I]

def heat_kernel_trace(L: LinearOperator, timepoints: Union[int, ArrayLike] = 10, **kwargs):
  """Computes the heat kernel trace of a gievn Laplacian operators at time points T.
  """
  from pbsig.linalg import eigvalsh_solver
  ew = eigvalsh_solver(L)(L, sigma=1e-6, which="LM", **kwargs)
  timepoints = logspaced_timepoints(timepoints) if isinstance(timepoints, Integral) else timepoints
  assert isinstance(timepoints, np.ndarray), "timepoints must be an array." 
  return np.array([np.sum(np.exp(-t*ew)) for t in timepoints])

class PsdSolver:
  def __init__(self, solver: str = 'default', k: Union[int, str, float, list] = "auto", laplacian: bool = False, eigenvectors: bool = False, tol: float = None, **kwargs):
    assert solver is not None, "Invalid solver"
    # self._rank_bound = int(rank_ub) if isinstance(rank_ub, Integral) else lambda A: rank_bound(A, upper=True)
    self.solver = solver
    self.k = k ## have to infer this 
    self.tol = tol if tol is not None else np.sqrt(np.finfo(np.float64).eps)
    self.laplacian = laplacian
    if "return_eigenvectors" in kwargs:
      self.eigenvectors = (eigenvectors | kwargs['return_eigenvectors']) 
      kwargs.pop("return_eigenvectors", None)
    else: 
      self.eigenvectors = eigenvectors
    self.fixed_params = kwargs
    # dict(tol=self.tolerance, return_eigenvectors=eigenvectors, k=)
    
  def configure_solver(self, A: Union[ArrayLike, spmatrix, LinearOperator], solver: str = 'default') -> tuple:
    if solver == 'dac': assert isinstance(A, np.ndarray), f"Cannot use divide-and-conquer with operators of type '{type(A)}'"
    # if solver != 'dac': assert isinstance(A, np.ndarray), f"Cannot use iterative methods with dense of type '{type(A)}'"
    if isinstance(A, np.ndarray) and solver == 'default' or solver == 'dac':
      solver = np.linalg.eigh if self.eigenvectors else np.linalg.eigvalsh
      return solver, {}
    elif isinstance(A, spmatrix) or isinstance(A, LinearOperator):
      # if nev == A.shape[0] and (solver == 'irl' or solver == 'default'):
      #   import warnings
      #   warnings.warn("Switching to PRIMME, as ARPACK cannot estimate all eigenvalues without shift-invert")
      #   solver = 'jd'
      solver = 'irl' if solver == 'default' else solver
      assert isinstance(solver, str) and solver in ['default', 'irl', 'lanczos', 'gd', 'jd', 'lobpcg']
      if solver == 'irl':
        from scipy.sparse.linalg import eigsh
        return eigsh, dict(which='LM', tol=self.tol, return_eigenvectors=self.eigenvectors)
      else:
        from primme import eigsh as primme_eigsh
        methods = { 'lanczos' : 'PRIMME_Arnoldi', 'gd': "PRIMME_GD" , 'jd' : "PRIMME_JDQR", 'lobpcg' : 'PRIMME_LOBPCG_OrthoBasis', 'default' : 'PRIMME_DEFAULT_MIN_TIME' }
        return primme_eigsh, dict(tol=self.tol, which='LM', return_eigenvectors=self.eigenvectors, method=methods[solver])
    else: 
      raise ValueError(f"Invalid solver / operator-type {solver}/{str(type(A))} given")
    
  def configure_k(self, A: Union[ArrayLike, spmatrix, LinearOperator], k: Union[int, str, float, list] = "auto") -> int:
    if isinstance(k, Integral): 
      return k
    elif isinstance(k, Number) and k >= 0.0 and k <= 1.0:
      if k == 0.0: return 0
      nev = trace_threshold(A, k) if k != 1.0 else rank_bound(A, upper=True)
      nev = min(nev, A.shape[0] - 1) # if self.laplacian else nev
      return nev 
    elif isinstance(k, str) and k == "auto":
      nev = rank_bound(A, upper=True)
      nev = min(nev, A.shape[0] - 1) # if self.laplacian else nev
      return nev
    else: 
      raise ValueError("Couldn't deduce number of eigenvalues to compute.")

  def __call__(self, A: Union[ArrayLike, spmatrix, LinearOperator], **kwargs) -> Union[ArrayLike, tuple]:
    assert A.shape[0] == A.shape[1], "A must be square"
    # assert pp >= 0.0 and pp <= 1.0, "Proportion 'pp' must be between [0,1]"
    n = A.shape[0]
    nev = self.configure_k(A, kwargs.pop("k", self.k))
    self.solver_, defaults_ = self.configure_solver(A, kwargs.pop("solver", self.solver))
    kwargs.pop("solver", None)
    if (isinstance(A, spmatrix) and len(A.data) == 0) or (n == 0 or A.shape[1] == 0) or nev == 0: 
      return (np.zeros(1), np.c_[np.zeros(n)]) if self.eigenvectors else np.zeros(1)
    if isinstance(A, spmatrix) or isinstance(A, LinearOperator): 
      defaults_ |= dict(k=nev, maxiter=n*100, ncv=None)
      # params = (default_params | params) | kwargs  # if 'scipy' in self.solver.__module__ else None #min(2*nev + 1, 20) 
    self.params_ = (defaults_ | self.fixed_params) | kwargs
    return self.solver_(A, **self.params_)
  
  def __repr__(self):
    format_val = lambda v: str(v) if isinstance(v, Integral) else (f"{v:.2e}" if isinstance(v, Number) else str(v))
    format_dict = lambda d: "(" + ", ".join([f"{str(k)} = {format_val(v)}" for k,v in d.items()]) + ")"
    s = "Laplacian" if self.laplacian else ""
    print_params = self.fixed_params | dict(solver=self.solver, k=self.k, tol=self.tol, eigenvectors=self.eigenvectors)
    s += f"PsdSolver{format_dict(print_params)}:\n"
    # if (self.fixed_params is not None and self.fixed_params != {}):
    #   s += f"Fixed params: {format_dict(self.fixed_params)}\n"
    s += f"Solver type: ({str(self.solver_)})\n"
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

def eigh_solver(A: Union[ArrayLike, spmatrix, LinearOperator], **kwargs):
  #return polymorphic_psd_solver(A, return_eigenvectors=True, **kwargs)
  return PsdSolver(A, return_eigenvectors=True, **kwargs)

def eigvalsh_solver(A: Union[ArrayLike, spmatrix, LinearOperator], **kwargs) -> Callable:
  #return polymorphic_psd_solver(A, return_eigenvectors=False, **kwargs)
  return PsdSolver(A, return_eigenvectors=False, **kwargs)


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

def spectral_rank(ew: ArrayLike, method: int = 0, shape: tuple = None, prec: float = None) -> int:
  if len(ew) == 0: return 0
  ew = np.array(ew) if not isinstance(ew, np.ndarray) else ew
  prec = np.finfo(ew.dtype).eps if prec is None else prec
  m,n = shape if (shape is not None and len(shape) == 2) else (len(ew), len(ew))
  if method == 0:
    ## Default: identifies only those as positive cannot have been introduced by numerical errors 
    tol = ew.max() * max(m,n) * prec
    return sum(ew > tol)
  elif method == 1: 
    ## Minimizes expected roundoff error
    tol = ew.max() * prec / 2. * np.sqrt(m + n + 1.)
    return sum(ew > tol)
  elif method == 2:
    ## Dynamic method: tests multiple thresholds for detecting rank deficiency up to the prescribed precision, 
    ## choosing the one that appears the most frequently 
    C = 1.0/np.exp(np.linspace(1,36,100))
    i = len(C)-np.searchsorted(np.flip(C), prec)
    C = C[:i]
    nr = [sum(ew/(ew + c)) for c in C]
    values, counts = np.unique(np.round(nr).astype(int), return_counts=True)
    return values[np.argmax(counts)]
  else:
    raise ValueError("Invalid method chosen")

def stable_rank(A: Union[ArrayLike, spmatrix, LinearOperator], method: int = 0, prec: float = None, **kwargs):
  """Computes the rank of an operator _A_ using by counting non-zero eigenvalues.
  
  This method attempts to use various means with which to estimate the rank of _A_ in a stable fashion. 
  """
  solver = eigvalsh_solver(A)
  ew = solver(A)
  return spectral_rank(ew, method=method, prec=prec)

def smooth_rank(A: Union[ArrayLike, spmatrix, LinearOperator], smoothing: tuple = (0.5, 1.0, 0), symmetric: bool = True, sqrt: bool = False, raw: bool = False, solver: Optional[Callable] = None, **kwargs) -> float:
  """ 
  Computes a smoothed-version of the rank of a symmetric positive semi-definite 'A'

  The form of 'A' can be a dense or sparse array, or a [matrix-free] linear operator.

  Parameters: 
    A := array, sparse matrix, or linear operator to compute the rank of
    pp := proportional part of the spectrum to compute.
    smoothing := tuple (eps, p, method) of args to pass to 'sgn_approx'
    symmetric := whether the 'A' is symmetric. Should always be true; non-symmetric matrices are not yet supported. 
    sqrt := whether to compute sqrt the singular values
    solver := Optional callable or string. If the latter, one of 'default', 'dac', 'irl', 'lanczos', 'gd', 'jd', or 'lobpcg'. See 'parameterize_solver' for more details. 
    kwargs := keyword arguments to pass to the parameterize_solver.
  """
  if solver is None: 
    solver = eigvalsh_solver(A, pp=1.0)
  elif isinstance(solver, str):
    solver = eigvalsh_solver(A, pp=1.0, solver=solver)
  else:
    assert isinstance(solver, Callable), "Solver must callable, if not string or None"
  ew = solver(A)
  ew = np.maximum(ew, 0.0)
  assert all(np.isreal(ew)) and all(ew >= 0.0), "Negative or non-real eigenvalues detected. This method only works with symmetric PSD matrices."
  ## Note that eigenvalues of PSD 'A' *are* its singular values via Schur theorem. 
  if smoothing is not None:
    assert isinstance(smoothing, tuple) and len(smoothing) == 3, "Smoothing must a tuple of the parameters to pass"
    eps, p, method = smoothing 
    ew = sgn_approx(np.sqrt(ew), eps, p, method) if sqrt else sgn_approx(ew, eps, p, method)
  else: 
    ew = np.sqrt(ew) if sqrt else ew
  return ew if raw else sum(ew)

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

def eigsh_block_shifted(A, k: int, b: int = 10, **kwargs):
  """
  Calculates all eigenvalues in a block-wise manner 
  """
  from primme import eigsh as primme_eigsh
  ni = int(np.ceil(k/b))
  f_args = dict(tol=np.finfo(np.float32).eps, which='CGT', return_eigenvectors=True, raise_for_unconverged=False, return_history=False)
  f_args = f_args | kwargs
  evals = np.zeros(ni*b)
  for i in range(ni):
    if i == 0:
      ew, ev = primme_eigsh(A, k=b, **f_args)
    else: 
      #f_args['which'] = min(ew)
      f_args['sigma'] = min(ew)
      ew, ev = primme_eigsh(A, k=b, lock=ev, **f_args)
    # print(ew)
    evals[(i*b):((i+1)*b)] = ew
  return(evals[-k:])


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


## TODO: make generalized up- and down- adjacency matrices for simplicial complexes

class UpLaplacianPy(LinearOperator):
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
    self._wfl = self._wfr = self._ws = UpLaplacianPy.identity_seq
    self.prepare()
    self._precompute_degree()
    self.precompute()
    self.P = None

  def diagonal(self) -> ArrayLike:
    return self.degree

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

  def set_weights(self, lw = None, cw = None, rw = None):
    self.face_left_weights = lw
    self.simplex_weights = cw
    self.face_right_weights = rw
    self._precompute_degree()
    return self 

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

  ## Computes all of the expressions that do not directly involve x, including input and output indices
  ## This essentially moves the looping structures in _matvec over to numpy, at the expense of a lot of memory  
  def precompute(self):
    q = len(self.simplices[0])-1
    N = 2*len(self.simplices)*comb(q+1, 2)
    self.P = np.zeros(shape=N, dtype=[('weights', 'f4'), ('xi', 'i2'), ('vo', 'i2')])
    cc = 0 
    for s_ind, s in enumerate(self.simplices):
      for (f1, f2), sgn_ij in zip(combinations(s.boundary(), 2), self.sgn_pattern):
        ii, jj = self.index(f1), self.index(f2)
        d1 = sgn_ij * self._wfl[ii] * self._ws[s_ind] * self._wfr[jj]
        d2 = sgn_ij * self._wfl[jj] * self._ws[s_ind] * self._wfr[ii]
        self.P[cc] = d1, jj, ii
        self.P[cc+1] = d2, ii, jj
        cc += 2
    return self
    #   np.add.at(v, P['vo'], x[P['xi']]*P['weights'])

  def _matvec_precompute(self, x: ArrayLike) -> ArrayLike:
    x = x.reshape(-1)
    self._v.fill(0)
    self._v += self.degree * x
    np.add.at(self._v, self.P['vo'], x[self.P['xi']]*self.P['weights'])
    return self._v
  
  ## Define the matvec operator using the precomputed degree, the pre-allocate workspace, and the pre-determined sgn pattern
  def _matvec_full(self, x: ArrayLike) -> ArrayLike:
    x = x.reshape(-1)
    self._v.fill(0)
    self._v += self.degree * x
    for s_ind, s in enumerate(self.simplices):
      for (f1, f2), sgn_ij in zip(combinations(s.boundary(), 2), self.sgn_pattern):
        ii, jj = self.index(f1), self.index(f2)
        self._v[ii] += sgn_ij * x[jj] * self._wfl[ii] * self._ws[s_ind] * self._wfr[jj]
        self._v[jj] += sgn_ij * x[ii] * self._wfl[jj] * self._ws[s_ind] * self._wfr[ii]
    return self._v
    
  def _matvec(self, x: ArrayLike) -> ArrayLike:
    assert x.ndim == 1 or (x.ndim == 2 and 1 in x.shape)
    return self._matvec_precompute(x) if self.P is not None else self._matvec_full(x)

  def _matmat(self, X: ArrayLike) -> ArrayLike:
    n,k = X.shape
    m = self.shape[0]
    Y = np.empty(shape=(m,k))
    for j in range(X.shape[1]):
      Y[:,j] = self._matvec(X[:,j])
    return Y


def nuclear_norm(X: ArrayLike) -> float:
  if X.shape[0] != X.shape[1]:
    X = X @ X.T
    use_sqrt = True 
  else: 
    assert is_symmetric(X)
    use_sqrt = False
  solver = eigvalsh_solver(X)
  return sum(abs(solver(X))) if not(use_sqrt) else sum(np.sqrt(np.abs(solver(X))))

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

from scipy.sparse import coo_array
def adjacency_matrix(S: ComplexLike, p: int = 0, weights: ArrayLike = None):
  assert len(S) > p, "Empty simplicial complex"
  if dim(S) <= (p+1):
    return coo_array((card(S,p), card(S,p)), dtype=int)
  weights = np.ones(card(S,p+1)) if weights is None else weights
  assert len(weights) == card(S,p+1), "Invalid weight array, must match length of p+1 simplices"
  V = list(faces(S, p))
  ind = []
  for s in faces(S, p+1):
    VI = [V.index(f) for f in faces(s, p)]
    for i,j in combinations(VI, 2):
      ind.append((i,j))
  IJ = np.array(ind)
  A = coo_array((weights, (IJ[:,0], IJ[:,1])), shape=(card(S,p), card(S,p)))
  A = A + A.T
  return A

## TODO: change weight to optionally be a string when attr system added to SC's
def up_laplacian(S: ComplexLike, p: int = 0, weight: Union[Callable, ArrayLike] = None, normed=False, return_diag=False, form='array', symmetric: bool = True, dtype=None, **kwargs):
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
      - (p>0, weight=...)                   <=> weighted combinatorial up-laplacian
      - (p>0, weight=..., normed=True)      <=> normalized weighted combinatorial up-laplacian 
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
      dtype := dtype of associated laplacian. 
      kwargs := unused. 
    
    The argument names for this function is loosely based on SciPy 'laplacian' interface in the sparse.csgraph module. 
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.laplacian.html. However, unlike the 
    LinearOperator returned by csgraph.laplacian, the operator returned here is always matrix-free.  
    (see https://github.com/scipy/scipy/blob/main/scipy/sparse/csgraph/_laplacian.py)
    """
    # assert isinstance(K, ComplexLike), "K must be a Simplicial Complex for now"
    weight = np.ones(card(S, p+1)) if weight is None else weight
    if isinstance(weight, Callable):
      _ = weight(faces(S, p))
      if not isinstance(_, Number) and len(_) == 2:
        W_LR = np.array([weight(s) for s in faces(S, p)]).astype(float)
        wpl = W_LR[:,0]
        wpr = W_LR[:,1]
      else: 
        wpl = np.array([float(weight(s)) for s in faces(S, p)])
        wpr = wpl
      wq = np.array([float(weight(s)) for s in faces(S, p+1)])
    elif isinstance(weight, Iterable):
      assert len(weight) == card(S, p+1) or len(weight) == len(S), "If weights given, they must match size of complex or p-faces"
      weight = np.asarray(weight)
      if len(weight) == card(S, p+1):
        wq, wpl, wpr = weight, np.ones(card(S, p)), np.ones(card(S, p))
      else:
        f_dim = np.array([dim(s) for s in faces(S)], dtype=np.uint16)
        wq = weight[f_dim == (p+1)]
        wpr = weight[f_dim == p]
        wpl = wpr
    else: 
      raise ValueError("Invalid weight function given")
    pseudo = lambda x: np.reciprocal(x, where=~np.isclose(x, 0)) # scalar pseudo-inverse
    assert len(wpl) == card(S,p) and len(wq) == card(S,p+1), "Invalid weight arrays given."
    assert all(wq >= 0.0) and all(wpl >= 0.0), "Weight function must be non-negative"
    if form == 'array':
      B = boundary_matrix(S, p = p+1)
      L = B @ diags(wq) @ B.T
      if normed: 
        deg = (diags(np.sign(wpl)) @ L @ diags(np.sign(wpr))).diagonal() ## retain correct nullspace
        L = (diags(np.sqrt(pseudo(deg))) @ L @ diags(np.sqrt(pseudo(deg)))).tocoo()
      else:
        L = (diags(np.sqrt(pseudo(wpl))) @ L @ diags(np.sqrt(pseudo(wpr)))).tocoo()
      return (L, L.diagonal()) if return_diag else L
    elif form == 'lo':
      # p_faces = list(faces(S, p))        ## need to make a view 
      p_simplices = list(faces(S, p+1))  ## need to make a view 
      _lap_cls = eval(f"UpLaplacian{int(p)}D")
      lo = _lap_cls(p_simplices, card(S, 0), card(S, p))
      if normed:
        if not(symmetric):
          import warnings
          warnings.warn("symmetric = False is not a valid option when normed = True")
        lo.set_weights(np.sign(wpl), wq, np.sign(wpr))
        deg = np.array(lo.degrees)
        lo.set_weights(pseudo(np.sqrt(deg)), wq, pseudo(np.sqrt(deg))) # normalized weighted symmetric psd version 
      else: 
        if symmetric:
          lo.set_weights(pseudo(np.sqrt(wpl)), wq, pseudo(np.sqrt(wpr)))  # weighted symmetric psd version 
        else:
          lo.set_weights(pseudo(wpl), wq, None)                           # asymmetric version
      return lo
      # f = _up_laplacian_matvec_p(p_simplices, p_faces, w0, w1, p, "default")
      # lo = f if form == 'function' else LinearOperator(shape=(ns[p],ns[p]), matvec=f, dtype=np.dtype(float))
    elif form == 'function':
      raise NotImplementedError("function form not implemented yet")
    else: 
      raise ValueError(f"Unknown form '{form}'.")


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


## Wrapper around the more efficient operator above to simplify things
class LaplacianOperator:
  def __init__(self, S: ComplexLike, lo: LinearOperator, normalized: bool = False):
    assert 'Laplacian' in type(lo).__name__
    self.operator = lo
    self.normalized = normalized
    self.complex = S
    self.p = int(type(lo).__name__[11])
    
  def update_scalar_product(self, f: Callable) -> None:
    assert isinstance(f, Callable), "Supplied f must be a function"
    fp = np.array([f(s) for s in faces(self.complex, self.p)], dtype=self.operator.dtype)
    fq = np.array([f(s) for s in faces(self.complex, self.p+1)], dtype=self.operator.dtype)
    assert all(fp >= 0.0)
    if self.normalized:
      self.operator.set_weights(np.sign(fp), fq, np.sign(fp))
      deg = np.array(self.operator.degrees)
      self.operator.set_weights(np.sqrt(pseudoinverse(deg)), fq, np.sqrt(pseudoinverse(deg)))
    else:
      self.operator.set_weights(np.sqrt(pseudoinverse(fq)), fq, np.sqrt(pseudoinverse(fq)))


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




