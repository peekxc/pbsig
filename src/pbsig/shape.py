

## Shape descriptors
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike

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


def PL_path(path, k: int, close_path: bool = False): 
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
  return(new_path)

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

# def sphere_ext(X: ArrayLike):