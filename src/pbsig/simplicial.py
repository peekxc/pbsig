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
class SimplexLike(Collection, Protocol):
  def boundary(self) -> Iterable['SimplexLike']: 
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

## Filtrations need not have delitem / be MutableSequences to match dionysus...

@runtime_checkable
class FiltrationLike(MutableSequence[SimplexLike], Protocol):
  """ """
  # def sort(self, key: Callable[[Simplex, Simplex], bool]) -> None: raise NotImplementedError 
  # def rearrange(self, indices: Collection) -> None: raise NotImplementedError


def as_simplex(vertices: Collection) -> SimplexLike:
  class _simplex(Collection):
    def __init__(self, v: Collection) -> None:
      self.vertices = v
    def __len__(self):
      return len(self.vertices)
    def __contains__(self, __x: object) -> bool:
      return self.vertices.__contains__(__x)
    def __iter__(self) -> Iterator:
      return iter(self.vertices)
    def boundary(self) -> Iterable['SimplexLike']: 
      raise NotImplementedError 
    def dimension(self) -> int: 
      return len(vertices)
    def __repr__(self):
      return str(self.vertices)
  return(_simplex(vertices))
    

def lower_star_filtration(simplices: Collection[SimplexLike], heights: Collection[float]) -> FiltrationLike:
  """
  Returns a Filtration 
  """
  S = [np.fromiter(s, dtype=int) for s in simplices]
  return(0)