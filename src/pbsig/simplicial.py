import numpy as np 

from typing import *
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay
from .utility import edges_from_triangles

def delaunay_complex(X: ArrayLike):
  dt = Delaunay(X)
  E = edges_from_triangles(dt.simplices, X.shape[0])
  T = dt.simplices
  K = {
    'vertices' : np.fromiter(range(X.shape[0]), dtype=np.int32),
    'edges' : E,
    'triangles' : T
  }
  return(K)
  
@runtime_checkable
class Simplex(Collection, Protocol):
  def boundary(self) -> Iterable['Simplex']: 
    raise NotImplementedError 
  def dimension(self) -> int: 
    raise NotImplementedError

@runtime_checkable
class Sequence(Collection, Sized, Protocol):
  def __getitem__(self, index): raise NotImplementedError

@runtime_checkable
class MutableSequence(Sequence, Protocol):
  def __delitem__(self, index): raise NotImplementedError
  def __setitem__(self, key, newvalue): raise NotImplementedError

@runtime_checkable
class Filtration(MutableSequence, Protocol):
  def sort(self, key: Callable[[Simplex, Simplex], bool]) -> None: raise NotImplementedError 
  def rearrange(self, indices: Collection) -> None: raise NotImplementedError



def lower_star_filtration():
  return(0)