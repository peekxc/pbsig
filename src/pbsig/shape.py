

## Shape descriptors
from typing import *
import numpy as np
from numbers import Number, Integral, Complex
from numpy.typing import ArrayLike
from itertools import * 
from more_itertools import *
from splex.predicates import * 
import _landmark as lm

def cart2pol(x, y):
  rho = np.sqrt(x**2 + y**2)
  phi = np.arctan2(y, x)
  return(rho, phi)

def shells(X: ArrayLike, k: int, center: Optional[ArrayLike] = None, **kwargs):
  """a histogram of distances from the center of mass to points on the surface"""
  barycenter = X.mean(axis=0) if center is None else center
  return np.histogram(np.linalg.norm(X - barycenter, axis=1), bins=k, **kwargs)[0]

def sectors_2d(X: ArrayLike, k: int, center: Optional[ArrayLike] = None, **kwargs):
  """a histogram of distances from the center of mass to points on the surface"""
  assert isinstance(X, np.ndarray) and X.shape[1] == 2, "Invalid object; must be point cloud in 2D."
  barycenter = X.mean(axis=0) if center is None else center
  Y = X - barycenter # origin must be (0,0) for polar equation to work
  _,Phi = cart2pol(Y[:,0], Y[:,1])
  return np.histogram(Phi, bins=k, **kwargs)[0]

def _archi_alpha(L: float, a_max: float) -> float:
  ''' Given length 'L', returns the corresponding alpha on the Archimedean spiral '''
  from scipy.special import ellipeinc
  from scipy.optimize import minimize
  archi_arc_len = lambda alpha: ellipeinc((np.pi*alpha)/a_max, -(a_max**2) / (2*np.pi))
  l2_error = lambda a, L: abs(archi_arc_len(a) - L)**2
  res = minimize(l2_error, 1.0, args=(L), method='Nelder-Mead', tol=1e-6)
  return(res.x[0])

def archimedean_sphere(n: int, nr: int):
  ''' Gives an n-point uniform sampling over 'nr' rotations around the 2-sphere '''
  n = int(n/2)
  a_max = nr*(2*np.pi)
  alpha = np.linspace(0, a_max, n)
  max_len = _archi_alpha(alpha[-1], a_max)
  alpha_equi = np.array([_archi_alpha(L, a_max) for L in np.linspace(0.0, max_len, n)])
  p = -np.pi/2 + (alpha_equi*np.pi)/(a_max)
  x = np.cos(alpha_equi)*np.cos(p)
  y = np.sin(alpha_equi)*np.cos(p)
  z = -np.sin(p)
  X = np.vstack((np.c_[x,y,z], np.flipud(np.c_[-x,-y, z])))
  return(X)


def complex2points(x): 
  return(np.c_[np.real(x), np.imag(x)])


# %% Definitions
def landmarks(a: ArrayLike, k: Optional[int] = 15, eps: Optional[float] = -1.0, seed: int = 0, diameter: bool = False, metric = "euclidean"):
	'''
	Computes landmarks points for a point set or set of distance 'a' using the 'maxmin' method. 

	Parameters:
		a := (n x d) matrix of *n* points in *d* dimensions, a distance matrix, or a set of pairwise distances
		k := (optional) number of landmarks requested. Defaults to 15. 
		eps := (optional) covering radius to stop finding landmarks at. Defaults to -1.0, using *k* instead.
		seed := index of the initial point to be the first landmark. Defaults to 0.
		diameter := whether to include the diameter as the first insertion radius (see details). Defaults to False. 
		metric := metric distance to use. Ignored if *a* is a set of distances. See details. 

	Details: 
		- The first radius is always the diameter of the point set, which can be expensive to compute for high dimensions, so by default "inf" is used as the first covering radius 
		- If the 'metric' parameter is "euclidean", the point set is used directly, otherwise the pairwise distances are computed first via 'dist'
		- If 'k' is specified an 'eps' is not (or it's -1.0), then the procedure stops when 'k' landmarks are found. The converse is true if k = 0 and eps > 0. 
			If both are specified, the both are used as stopping criteria for the procedure (whichever becomes true first).
		- Given a fixed seed, this procedure is deterministic. 

	Returns a pair (indices, radii) where:
		indices := the indices of the points defining the landmarks; the prefix of the *greedy permutation* (up to 'k' or 'eps')
		radii := the insertion radii whose values 'r' yield a cover of 'a' when balls of radius 'r' are places at the landmark points.

	The maxmin method yields a logarithmic approximation to the geometric set cover problem.

	'''
	a = np.asanyarray(a)
	k = 0 if k is None else int(k)
	eps = -1.0 if eps is None else float(eps)
	seed = int(seed)
	if is_dist_like(a):
		if is_distance_matrix(a):
			a = a[np.triu_indices(a.shape[0], k=1)]
		if a.dtype != np.float64:
			a = a.astype(np.float64)
		indices, radii = lm.maxmin(a, eps, k, True, seed)
	elif metric == "euclidean" and is_point_cloud(a):
		indices, radii = lm.maxmin(a.T, eps, k, False, seed)
		radii = np.sqrt(radii)
	else:
		raise ValueError("Unknown input type detected. Must be a matrix of points, a distance matrix, or a set of pairwise distances.")
	
	## Check is a valid cover 
	is_monotone = np.all(np.diff(-np.array(radii)) >= 0.0)
	assert is_monotone, "Invalid metric: non-monotonically decreasing radii found."

	return(indices, radii)

def PL_path(path, k: int, close_path: bool = False, output_path: bool = False): 
  """Convert a SVG collection of paths into a set k-1 piecewise-linear line segments.
  
  The sample points interpolated along the given paths are equi-spaced in arclength.
  """
  from svgpathtools import parse_path, Line, Path, wsvg
  arc_lengths = np.array([np.linalg.norm(seg.length()) for seg in path])
  if any(arc_lengths == 0):
    path = list(filter(lambda p: p.length() > 0.0, path))
    for j in range(len(path)-1):
      if path[j].end != path[j+1].start:
        path[j].end = path[j+1].start
    path = Path(*path)
  arc_lengths = np.array([np.linalg.norm(seg.length())for seg in path])
  assert(all(arc_lengths > 0)), "Invalid shape detected: lines with 0-length found and not handled"
  A = np.cumsum(arc_lengths)
  p_cc = np.linspace(0, max(A), k)
  idx = np.digitize(p_cc, A+np.sqrt(np.finfo(float).resolution))
  L_points = []
  for i, pp in zip(idx, p_cc):
    t = pp/A[i] if i == 0 else (pp-A[i-1])/(A[i]-A[i-1])
    L_points.append(path[i].point(t))
  assert len(L_points) == k
  if isinstance(L_points[0], Complex):
    complex2pt = lambda x: (np.real(x), np.imag(x))
    L_points = list(map(complex2pt, L_points))
  ## Connect them PL 
  connect_the_dots = [Line(p0, p1) for p0, p1 in pairwise(L_points)]
  if close_path:
    connect_the_dots.append(Line(connect_the_dots[-1].end, connect_the_dots[0].start))
  new_path = Path(*connect_the_dots)
  return new_path if output_path else np.array([s.start for s in new_path])

def simplify_outline(S: ArrayLike, k: int):
  from svgpathtools import parse_path, Line, Path, wsvg
  Lines = [Line(p, q) for p,q in pairwise(S)]
  Lines.append(Line(Lines[-1].end, Lines[0].start))
  return(PL_path(Path(*Lines), k))

def offset_curve(path, offset_distance, steps=1000):
  """
  Takes in a Path object, `path`, and a distance, `offset_distance`, and outputs an piecewise-linear approximation of the 'parallel' offset curve.
  """
  from svgpathtools import parse_path, Line, Path, wsvg
  nls = []
  for seg in path:
    ct = 1
    for k in range(steps):
      t = k / steps
      offset_vector = offset_distance * seg.normal(t)
      nl = Line(seg.point(t), seg.point(t) + offset_vector)
      nls.append(nl)
  connect_the_dots = [Line(nls[k].end, nls[k+1].end) for k in range(len(nls)-1)]
  if path.isclosed():
    connect_the_dots.append(Line(nls[-1].end, nls[0].end))
  offset_path = Path(*connect_the_dots)
  return offset_path

def stratify_sphere(d: int, n: int, **kwargs) -> np.ndarray:
  """Partitions the d-sphere into a set of _n_ unit vectors in dimension (d+1)."""
  assert d == 1 or d == 2, "Only d == 1 or d == 2 are implemented."
  if d == 1:
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)+(np.pi/2)
    V = np.array(list(zip(np.cos(theta), np.sin(theta))))
    return V
  elif d == 2:
    args = dict(nr = 5) | kwargs
    V = archimedean_sphere(n, **args)
    return V
  else: 
    ## TODO: do maxmin sampling for d > 2, accept option to support paths/Iterables
    raise NotImplementedError("Haven't done d-spheres larger than 2 yet")

def normalize_shape(X: ArrayLike, V: Iterable[np.ndarray] = None, scale = "directions", translate: str = "directions", **kwargs) -> ArrayLike:
  """Performs a variety of shape normalizations, such as mean-centering and scaling, with respect to a set of direction vectors _V_."""
  u = shape_center(X, method=translate, V=V, **kwargs)
  X = X - u 
  if scale == "directions":
    assert V is not None, "Direction vectors must be supplied when scaling is directions based!"
    L = -sum([min(X @ vi[:,np.newaxis]) for vi in V])
    return (len(V)/L)*X
  elif scale == "diameter":
    raise NotImplementedError("Haven't implemented yet")
    diam = np.max(pdist(X))
    return X 
  else: 
    raise ValueError(f"Invalid scale given '{scale}'")
                               
def shape_center(X: ArrayLike, method: str = ["pointwise", "bbox", "hull", "polygon", "directions"], V: Optional[ArrayLike] = None, atol: float = 1e-12):
  """
  Given a set of (n x d) points 'X' in d dimensions, returns the (1 x d) 'center' of the shape, suitably defined. 

  Each type of center varies in cost and definition, but each can effectively be interpreted as some kind of 'barycenter'. In particular: 

  barycenter := point-wise barycenter of 'X', equivalent to arithmetic mean of the points 
  box := barycenter of bounding box of 'X' 
  hull := barycenter of convex hull of 'X'
  polygon := barycenter of 'X', when 'X' traces out the boundary of a closed polygon in R^2*
  directions := barycenter of 'X' projected onto a set of rotationally-symmetric direction vectors 'V' in S^1
  
  The last options 'directions' uses an iterative process (akin to meanshift) to find the barycenter 'u' satisfying: 

  np.array([min((X-u) @ v)*v for v in V]).sum(axis=0) ~= [0.0, 0.0]
  """
  import warnings
  n, d = X.shape[0], X.shape[1]
  method = "directions" if V is not None and method == ["barycenter", "directions", "bbox", "hull"] else method 
  if method == "barycenter" or method == ["barycenter", "directions", "bbox", "hull"]:
    return(X.mean(axis=0))
  elif method == "hull":
    from scipy.spatial import ConvexHull
    return(X[ConvexHull(X).vertices,:].mean(axis=0))
  elif method == "directions":
    assert V is not None and isinstance(V, np.ndarray) and V.shape[1] == d
    ## Check V is rotationally symmetric by seeing if there is an opposite vector that exists for each v
    rot_sym = np.allclose([min(np.linalg.norm(V + v, axis=1)) for v in V], 0.0, atol=atol)
    if not rot_sym:
      warnings.warn(f"Warning: supplied vectors 'V' not rotationally symmetric up to supplied tolerance {atol}")
    original_center = X.mean(axis=0)
    cost: float = np.inf
    while not(np.isclose(cost, 0.0, atol=atol)):
      Lambda = [np.min(X @ vi[:,np.newaxis]) for vi in V]
      U = np.vstack([s*vi for s, vi in zip(Lambda, V)])
      center_diff = U.mean(axis=0)
      X = X - center_diff
      cost = min([
        np.linalg.norm(np.array([min(X @ vi[:,np.newaxis])*vi for vi in V]).sum(axis=0)),
        np.linalg.norm(center_diff)
      ])
    return(original_center - X.mean(axis=0))
  elif method == "bbox":
    min_x, max_x = X.min(axis=0), X.max(axis=0)
    return((min_x + (max_x-min_x)/2))
  elif method == "polygon":
    from shapely.geometry import Polygon
    P = Polygon(X)
    return np.array(list(P.centroid.coords)).flatten()
  else: 
    raise ValueError("Invalid method supplied")


def winding_distance(X: ArrayLike, Y: ArrayLike):
  """ 
  Returns the minimum 'winding' distance between sequences of points (X,Y) defining closed curves in the plane homeomorphic to S^1
  
  Given two arrays (X, Y) of size (n,m) = (|X|,|Y|) representing 'circle-ish outlines', i.e. representing shapes whose edge sets EX satisfy:
  
  EX = [(X[0,:], X[1,:]), ..., (X[-1,], X[0])] (and similar with Y)
  
  This function: 
  1. finds a set of N=max(n,m) equi-spaced points along the boundaries (EX,EY) 
  2. cyclically rotates the N points from (1) along EX to best-align the N points along EY (i.e. w/ minimum squared euclidean distance)
  3. computes the integrated distances between the PL-curves traced by (EX, EY) 

  Since the curves are PL, (3) reduces to a sequence of irregular quadrilateral area calculations. 
  """ 
  from scipy.spatial import ConvexHull
  from shapely.geometry import Polygon
  from itertools import cycle
  from pyflubber.closed import prepare
  A, B = prepare(X, Y)
  N = A.shape[0] # should match B 
  edge_pairs = cycle(zip(pairwise(A), pairwise(B)))
  int_diff = 0.0
  for (p1,p2), (q1,q2) in islice(edge_pairs, N+1):
    ind = ConvexHull([p1,p2,q1,q2]).vertices
    int_diff += Polygon(np.vstack([p1,p2,q1,q2])[ind,:]).area
  return(int_diff)

def shift_curve(line, reference, objective: Optional[Callable] = None):
  min_, best_offset = np.inf, 0
  if objective is None: 
    objective = lambda X,Y: np.linalg.norm(X-Y)
  for offset in range(len(line)):
    deviation = objective(reference, np.roll(line, offset, axis=0))
    if deviation < min_:
      min_ = deviation
      best_offset = offset
  if best_offset:
    line = np.roll(line, best_offset, axis=0)
  return line

def Kabsch(A, B):
  H = A.T @ B
  R = (H.T @ H)**(1/2) @ np.linalg.inv(H)
  u,s,vt = np.linalg.svd(H)
  d = np.sign(np.linalg.det(vt.T @ u.T))
  R = vt.T @ np.diag(np.append(np.repeat(1, A.shape[1]-1), d)) @ u.T
  return R

from scipy.linalg import orthogonal_procrustes
def procrustes_cc(A: ArrayLike, B: ArrayLike, do_reflect: bool = True, matched: bool = False, preprocess: bool = False):
  """ Procrustes distance between closed curves """
  # from procrustes.rotational import rotational
  # from pbsig.pht import pht_preprocess_pc
  from pyflubber.closed import prepare
  # if preprocess:
  #   A, B = pht_preprocess_pc(A), pht_preprocess_pc(B)
  if do_reflect:
    R_refl = np.array([[-1, 0], [0, 1]])
    A1,B1 = procrustes_cc(A, B, do_reflect=False, matched=matched, preprocess=False)
    A2,B2 = procrustes_cc(A @ R_refl, B, do_reflect=False, matched=matched, preprocess=False)
    return (A1, B1) if np.linalg.norm(A1-B1, 'fro') < np.linalg.norm(A2-B2, 'fro') else (A2, B2)
  else:
    if matched:
      # R = rotational(A, B, pad=False, translate=False, scale=False)
      # from scipy.spatial import procrustes
      R, err = orthogonal_procrustes(A, B)
      #return np.linalg.norm((A @ R.t) - B, 'fro')
      return A @ R, B
    else: 
      ## Fix clockwise / counter-clockwise orientation of points
      A_p, B_p = prepare(A, B)
      
      ## Shift points along shape until minimum procrustes distance is obtained
      #pro_error = lambda X,Y: rotational(X, Y, pad=False, translate=False, scale=False).error
      def pro_error(X, Y):
        R, _ = orthogonal_procrustes(X, Y)[1]
        return np.linalg.norm((X @ R) - Y, 'fro')
      pro_error = lambda X,Y: np.linalg.norm((X @ orthogonal_procrustes(X, Y)[0]) - Y, 'fro')
      os = np.argmin([pro_error(A_p, np.roll(B_p, offset, axis=0)) for offset in range(A_p.shape[0])])
      A, B = A_p, np.roll(B_p, os, axis=0)
      # plt.plot(*A.T);plt.plot(*B.T)
      return procrustes_cc(A, B, matched=True)

def procrustes_dist_cc(A: ArrayLike, B: ArrayLike, **kwargs):
  A_p, B_p = procrustes_cc(A, B, **kwargs)
  return np.linalg.norm(A_p - B_p, 'fro')


# def sphere_ext(X: ArrayLike):

