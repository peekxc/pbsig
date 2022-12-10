import numpy as np 
from numbers import Number
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

def graph2complex(G):
  K = {
    'vertices' : np.sort(np.array([i for i in G.nodes()], dtype=int)),
    'edges' : lexsort_rows(np.array([[i,j] for i,j in G.edges()], dtype=int)),
    'triangles' : np.empty(shape=(0,3), dtype=int)
  }
  return K


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



## Contains definitions and utilities for prescribing a structural type system on the 
## space of abstract simplicial complexes and on [essential simplexwise] simplicial filtrations
## 
## In practice, since Python is interpreted, this is typically implemented as a duck-typing system enforced via Protocol classes. 
## In the future, a true structural subtype system is preferable via e.g. mypy to enable optimizations such as mypy-c


@runtime_checkable
class SimplexLike(Collection, Protocol):
  ''' 
  An object is *simplex-like* if it (minimally) implements a boundary operator and has a dimension  

  Inherited Abstract Methods: __contains__, __iter__, __len__
  '''
  def boundary(self) -> Iterable['SimplexLike']: 
    raise NotImplementedError 
  def dimension(self) -> int: 
    raise NotImplementedError

## Simplicial complex duck type
@runtime_checkable
class ComplexLike(Collection, Protocol):
  ''' 
  An object is *complex-like* if it is a collection and it can iterate through [p- or all-] simplex-like objects and it has a dimension 
  
  Inherits Collection => ComplexLike are Sized, Iterable, Containers 
  Implements:
    __contains__, __iter__, __len__
  
  '''
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
class FiltrationLike(ComplexLike, Sequence[SimplexLike], Protocol):
  """ """
  def __reversed__(self):
    raise NotImplementedError
  def index(self, other: SimplexLike):
    raise NotImplementedError

@runtime_checkable
class MutableSequence(Sequence, Protocol):
  def __delitem__(self, index): raise NotImplementedError
  def __setitem__(self, key, newvalue): raise NotImplementedError

## Filtrations need not have delitem / be MutableSequences to match dionysus...
@runtime_checkable
class MutableFiltrationLike(MutableSequence[SimplexLike], Protocol):
  """ """
  def sort(self, key: Callable[[SimplexLike, SimplexLike], bool]) -> None: raise NotImplementedError 
  def rearrange(self, indices: Collection) -> None: raise NotImplementedError

from itertools import combinations
from functools import total_ordering
import numpy as np

# @total_ordering
class Simplex(Set):
  '''
  Implements: 
    __contains__(self, v: int) <=> Returns whether integer 'v' is a vertex in 'self'
  '''
  def __init__(self, v: Collection[int]) -> None:
    if isinstance(v, Number):
      self.vertices = tuple([int(v)])
    else:
      self.vertices = tuple(np.unique(np.sort(np.ravel(tuple(v)))))
  def __eq__(self, other) -> bool: 
    return(all(v == w for (v,w) in zip(iter(self.vertices), iter(other))))
  def __len__(self):
    return len(self.vertices)
  def __lt__(self, other: Collection[int]) -> bool:
    ''' Returns whether self is a face of other '''
    if len(self) >= len(other): 
      return(False)
    else:
      return(all([v in other for v in self.vertices]))
  def __le__(self, other: Collection) -> bool: 
    if len(self) > len(other): 
      return(False)
    elif len(self) == len(other):
      return self.__eq__(other)
    else:
      return self < other
  def __ge__(self, other: Collection[int]) -> bool:
    if len(self) < len(other): 
      return(False)
    elif len(self) == len(other):
      return self.__eq__(other)
    else:
      return self > other
  def __gt__(self, other: Collection[int]) -> bool:
    if len(self) <= len(other): 
      return(False)
    else:
      return(all([v in self.vertices for v in other]))
  def __contains__(self, __x: int) -> bool:
    """ Reports vertex-wise inclusion """
    if not isinstance(__x, Number): 
      return False
    return self.vertices.__contains__(__x)
  
  def __iter__(self) -> Iterable[int]:
    return iter(self.vertices)

  def faces(self, p: Optional[int] = None) -> Iterable['Simplex']:
    dim = len(self.vertices)
    if p is None:
      yield from map(Simplex, chain(*[combinations(self.vertices, d) for d in range(1, dim+1)]))
    else: 
      yield from filter(lambda s: len(s) == p+1, self.faces(None))

  def boundary(self) -> Iterable['Simplex']: 
    if len(self.vertices) == 0: 
      return self.vertices
    yield from map(Simplex, combinations(self.vertices, len(self.vertices)-1))

  def dimension(self) -> int: 
    return len(self.vertices)-1
  def __repr__(self):
    return str(self.vertices).replace(',','') if self.dimension() == 0 else str(self.vertices).replace(' ','')
  def __getitem__(self, index: int) -> int:
    return self.vertices[index] # auto handles IndexError exception 
  def __sub__(self, other) -> 'Simplex':
    return Simplex(set(self.vertices) - set(other))
  def __hash__(self):
    # Because Python has no idea about mutability of an object.
    return hash(self.vertices)
    
from collections.abc import Set, Hashable

# https://stackoverflow.com/questions/798442/what-is-the-correct-or-best-way-to-subclass-the-python-set-class-adding-a-new
class SimplicialComplex(set):
  """ Abstract Simplicial Complex"""
  __hash__ = Set._hash
  wrapped_methods = ('difference', 'intersection', 'symmetric_difference', 'union', 'copy')
  
  @classmethod
  def _wrap_methods(cls, names):
    def wrap_method_closure(name):
      def inner(self, *args):
        result = getattr(super(cls, self), name)(*args)
        result = cls(result) if isinstance(result, set) else result 
        return result
      inner.fn_name = name
      setattr(cls, name, inner)
    for name in names: wrap_method_closure(name)

  def __new__(cls, iterable=None):
    selfobj = super(SimplicialComplex, cls).__new__(SimplicialComplex)
    cls._wrap_methods(cls.wrapped_methods)
    return selfobj

  def __init__(self, iterable = None):
    """"""
    self_set = super(SimplicialComplex, self)
    self_set.__init__()
    if iterable is not None: 
      self.update(iterable)

  def update(self, iterable):
    self_set = super(SimplicialComplex, self)
    for item in iterable:
      self.add(item)

  def add(self, item: Collection[int]):
    self_set = super(SimplicialComplex, self)
    self_set.update(Simplex(item).faces())
  
  def remove(self, item: Collection[int]):
    self_set = super(SimplicialComplex, self)
    self_set.difference_update(set(self.cofaces(item)))
  
  def discard(self, item: Collection[int]):
    self_set = super(SimplicialComplex, self)
    self_set.discard(Simplex(item))

  def cofaces(self, item: Collection[int]):
    s = Simplex(item)
    yield from filter(lambda t: t >= s, iter(self))

  def __contains__(self, item: Collection[int]):
    self_set = super(SimplicialComplex, self)
    return self_set.__contains__(Simplex(item))

  def faces(self, p: Optional[int] = None) -> Iterable['Simplex']:
    if p is None:
      yield from iter(self)
    else: 
      assert isinstance(p, Number)
      yield from filter(lambda s: len(s) == p + 1, iter(self))

  def dimension(self) -> int: 
    return max([s.dimension() for s in iter(self)])

  def __repr__(self) -> str:
    self_set = super(SimplicialComplex, self)
    if self_set.__len__() <= 10:
      return str(self_set)
    else:
      return "Large complex"

  def print(self) -> None:
    d = self.dimension()
    ST = np.zeros(shape=(self.__len__(), d+1), dtype='<U15')
    ST.fill(' ')
    for i,s in enumerate(self):
      ST[i,:len(s)] = str(s)[1:-1].split(',')
    SC = np.apply_along_axis(lambda x: ' '.join(x), axis=0, arr=ST)
    for s in SC: print(s, sep='', end='\n')

# def less_lexicographic_refinement(S1: Tuple[SimplexLike, Any], S2: Tuple[SimplexLike, Any]) -> bool:
#   (s1,i1), (s2,i2) = S1, S2
#   if i1 != i2: return(i1 < i2)
#   if len(s1) != len(s2): return(len(s1) < len(s2))
#   return(tuple(iter(s1)) < tuple(iter(s2)))

class Filtration(Mapping):
  """
  Simplicial Filtration 

  Implements: __getitem__, __iter__, __len__, __contains__, keys, items, values, get, __eq__, and __ne__
  """
  def __init__(self, simplices: Sequence[SimplexLike], I: Optional[Collection] = None) -> None:

    # assert all([isinstance(s, SimplexLike) for s in simplices]), "Must all be simplex-like"
    if I is not None: assert len(simplices) == len(I)
    self.simplices = [Simplex(s) for s in simplices]
    self.index_set = np.arange(0, len(simplices)) if I is None else np.asarray(I)
    # self.dtype = [('s', Simplex), ('index', I.dtype)]


    # np.fromiter(zip(), self.dtype)
    # self.simplices = simplices
    # self.indices = range(len(simplices)) if I is None else I

  ## --- Collection methods ---
  def __contains__(self, __x: Collection[int]) -> bool:
    return self.simplices.__contains__(Simplex(__x))
  def __iter__(self) -> Iterator:
    return iter(self.simplices)
  def __len__(self):
    return len(self.simplices)

  ## --- Mapping methods ---
  def __getitem__(self, index: Any) -> Tuple[Simplex, Any]:
    idx = self.index_set.searchsorted(index)
    return self.simplices[idx]

  def keys(self):
    return self.index_set
  
  def values(self):
    return self.simplices

  def items(self):
    return zip(self.keys, self.simplices)

  # def __eq__(self, other):
  #   return self.
  def get(self, key, value = None):
    return self[key] if key in self.index_set else value

  # def __delitem__(self, index): 
  #   # raise NotImplementedError
  #   if index < len(self.simplices):
  #     del self.simplices[index]
  # def __setitem__(self, key, newvalue): 
  #   #raise NotImplementedError
  #   assert isinstance(newvalue, SimplexLike), "Value-type must be simplex-like"
  #   self.simplices[key] = newvalue 

  # def sort(self, key: Optional[Callable[[SimplexLike, SimplexLike], bool]] = None) -> None: 
  #   #raise NotImplementedError 
  #   if key is None: 
  #     key = less_lexicographic_refinement 
  #   self.simplices = sorted(self.simplices, key=key)
  # def rearrange(self, indices: Collection) -> None:
  #   #self.simplices = sorted(self.simplices, key=lambda s1, s2: )
  #   raise NotImplementedError
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

# def as_filtration(simplices: Collection[SimplexLike]) -> SimplexLike:
#   ## Construct a mutable sequence (list in this case) of simplices
#   F = [Simplex(s) for s in simplices]
#   assert isinstance(F, MutableSequence), "simplices must be a mutable sequence"
#   return(Filtration(F))

# def lower_star_filtration(simplices: Collection[SimplexLike], heights: Collection[float]) -> FiltrationLike:
#   """
#   Returns a Filtration 
#   """
#   # filtration([as_simplex(s) for s in [[0], [1], [0,1]]])
#   as_filtration(simplices)
#   S = [np.fromiter(s, dtype=int) for s in simplices]
#   return(0)