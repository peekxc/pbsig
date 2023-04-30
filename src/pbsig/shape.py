

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



# def sphere_ext(X: ArrayLike):