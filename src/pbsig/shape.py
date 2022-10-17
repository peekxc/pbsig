

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

# def sphere_ext(X: ArrayLike):