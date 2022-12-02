import numpy as np 

from typing import *
from itertools import *
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay
from .utility import edges_from_triangles, cycle_window
import networkx as nx
from networkx import Graph 
from scipy.sparse import diags, csc_matrix, issparse
from scipy.linalg import issymmetric
from pbsig.utility import lexsort_rows


def edge_iterator(A):
  assert issparse(A), "Must be a sparse matrix"
  for i,j in zip(*A.nonzero()):
    if i < j: 
      yield (i,j)

def delaunay_complex(X: ArrayLike):
  dt = Delaunay(X)
  E = edges_from_triangles(dt.simplices, X.shape[0])
  T = dt.simplices
  K = {
    'vertices' : np.fromiter(range(X.shape[0]), dtype=np.int32),
    'edges' : lexsort_rows(E),
    'triangles' : lexsort_rows(T)
  }
  return(K)

def cycle_graph(X: ArrayLike):
  "Creates a cycle graph from a set of ordered points" 
  E = np.array(list(cycle_window(range(X.shape[0]))))
  K = {
    'vertices' : np.fromiter(range(X.shape[0]), dtype=np.int32),
    'edges' : lexsort_rows(E),
    'triangles' : np.empty(shape=(0,3))
  }
  return(K)

def complete_graph(n: int):
  "Creates the complete graph from an integer n" 
  from itertools import combinations
  K = {
    'vertices' : np.fromiter(range(n), dtype=np.int32),
    'edges' : np.array(list(combinations(range(n), 2)), dtype=np.int32),
    'triangles' : np.empty(shape=(0,3), dtype=np.int32)
  }
  return(K)


def is_symmetric(A):
  if issparse(A):
    r, c = A.nonzero()
    ri = np.argsort(r)
    ci = np.argsort(c)
    return all(r[ri] == c[ci]) and not(any((A != A.T).data))
  else: 
    return issymmetric(A)

def laplacian_DA(L):
  """ Converts graph Laplacian L to Adjacency + diagonal degree """
  D = diags(L.diagonal())
  A = L - D
  return D, A

def graph_laplacian(G: Union[Graph, Tuple[Iterable, int]], normalize: bool = False):
  if isinstance(G, Graph):
    A = nx.adjacency_matrix(G)
    D = diags([G.degree(i) for i in range(G.number_of_nodes())])
    L = D - A
  elif isinstance(G, dict) and 'vertices' in G.keys():
    E, n = G['edges'], len(G['vertices'])
    L = np.zeros(shape=(n,n))
    for i,j in E: 
      L[i,j] = L[j,i] = -1
    L = diags(abs(L.sum(axis=0))) + csc_matrix(L)
  elif isinstance(G, np.ndarray) or issparse(G): # adjacency matrix 
    L = diags(np.ravel(G.sum(axis=0))) - G
  elif isinstance(G, Iterable):
    E, n = G
    L = np.zeros(shape=(n,n))
    for i,j in E: 
      L[i,j] = L[j,i] = -1
    L = diags(abs(L.sum(axis=0))) + csc_matrix(L)
  else: 
    raise ValueError("Invalid input")
  if normalize: 
    D = diags(1.0/np.sqrt(L.diagonal()))
    L = D @ L @ D
  return L

@runtime_checkable
class SimplexLike(Collection, Protocol):
  def boundary(self) -> Iterable['SimplexLike']: 
    raise NotImplementedError 
  def dimension(self) -> int: 
    raise NotImplementedError

## Simplicial complex duck type
@runtime_checkable
class ComplexLike(Collection, Protocol):
  def __iter__(self) -> Iterable['SimplexLike']: 
    raise NotImplementedError 
  def __next__(self) -> SimplexLike:
    raise NotImplementedError
  def dimension(self) -> int: 
    raise NotImplementedError
  def simplices(self, p: int) -> Iterable['SimplexLike']:
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
  def sort(self, key: Callable[[SimplexLike, SimplexLike], bool]) -> None: raise NotImplementedError 
  def rearrange(self, indices: Collection) -> None: raise NotImplementedError

from itertools import combinations
import numpy as np
class Simplex(Collection):
  def __init__(self, v: Collection) -> None:
    self.vertices = tuple(v)
  # def __hash__() -> np.int_:
  #   return(0)
  def __eq__(self, other) -> bool: 
    return(all(v == w for (v,w) in zip(iter(self.vertices), iter(other))))
  def __len__(self):
    return len(self.vertices)
  def __lt__(self, other: SimplexLike) -> bool:
    ''' Returns whether self is a face of other '''
    if len(self) >= len(other): 
      return(False)
    # elif len(self) == len(other)-1: 
    #   return(self in other.boundary())
    else:
      #return(any(self < face for face in other.boundary()))
      return(all([v in other.vertices for v in self.vertices]))
  def __contains__(self, __x: int) -> bool:
    """ Reports vertex-wise inclusion """
    return self.vertices.__contains__(__x)
  def __iter__(self) -> Iterator:
    return iter(self.vertices)
  def boundary(self) -> Iterable['SimplexLike']: 
    if len(self.vertices) == 0: 
      return self.vertices
    for s in combinations(self.vertices, len(self.vertices)-1):
      yield as_simplex(s) 
  def dimension(self) -> int: 
    return len(self.vertices)-1
  def __repr__(self):
    return str(self.vertices).replace(',','') if self.dimension() == 0 else str(self.vertices).replace(' ','')

def as_simplex(vertices: Collection) -> SimplexLike:
  return(Simplex(vertices))
    
def less_lexicographic_refinement(S1: Tuple[SimplexLike, Any], S2: Tuple[SimplexLike, Any]) -> bool:
  (s1,i1), (s2,i2) = S1, S2
  if i1 != i2: return(i1 < i2)
  if len(s1) != len(s2): return(len(s1) < len(s2))
  return(tuple(iter(s1)) < tuple(iter(s2)))

class Filtration(Collection):
  def __init__(self, simplices: MutableSequence[SimplexLike], I: Optional[Collection] = None) -> None:
    # isinstance([0,1,2], MutableSequence)
    assert all([isinstance(s, SimplexLike) for s in simplices]), "Must all be simplex-like"
    self.simplices = simplices
    self.indices = range(len(simplices)) if I is None else I
  def __getitem__(self, index) -> SimplexLike:
    return(self.simplices[index])
  def __delitem__(self, index): 
    # raise NotImplementedError
    if index < len(self.simplices):
      del self.simplices[index]
  def __setitem__(self, key, newvalue): 
    #raise NotImplementedError
    assert isinstance(newvalue, SimplexLike), "Value-type must be simplex-like"
    self.simplices[key] = newvalue 
  def __len__(self):
    return len(self.simplices)
  def __contains__(self, __x: object) -> bool:
    return self.simplices.__contains__(__x)
  def __iter__(self) -> Iterator:
    return iter(self.simplices)
  def sort(self, key: Optional[Callable[[SimplexLike, SimplexLike], bool]] = None) -> None: 
    #raise NotImplementedError 
    if key is None: 
      key = less_lexicographic_refinement 
    self.simplices = sorted(self.simplices, key=key)
  def rearrange(self, indices: Collection) -> None:
    #self.simplices = sorted(self.simplices, key=lambda s1, s2: )
    raise NotImplementedError
  def __validate__(self):
    """ Checks the face poset is valid, and asserts that the corresponding indices align as well """
    for (i, s), idx in zip(enumerate(self.simplices), self.indices):
      if s.dimension() == 0: 
        continue
      else: 
        ## Check: all faces in filtration, all at indices < coface index, all have stored indices < coface stored index
        assert all([f in self.simplices for f in s.boundary()]), "Not all faces contained in filtration"
        face_ind = np.array([self.simplices.index(f) for f in s.boundary()])
        assert all(face_ind < i), "There exist faces that come after their coface in filtration order"
        assert all([self.indices[fi] < idx for fi in face_ind]), "Index set not consistent with face poset"
        # assert all([f in self.simplices and self.simplices.index(f) < i for f in s.boundary()])
    return(True)
  def __repr__(self):
    return "F: "+str(self.simplices)

def as_filtration(simplices: Collection[SimplexLike]) -> SimplexLike:
  ## Construct a mutable sequence (list in this case) of simplices
  F = [as_simplex(s) for s in simplices]
  assert isinstance(F, MutableSequence), "simplices must be a mutable sequence"
  return(Filtration(F))

def lower_star_filtration(simplices: Collection[SimplexLike], heights: Collection[float]) -> FiltrationLike:
  """
  Returns a Filtration 
  """
  # filtration([as_simplex(s) for s in [[0], [1], [0,1]]])
  as_filtration(simplices)
  S = [np.fromiter(s, dtype=int) for s in simplices]
  return(0)