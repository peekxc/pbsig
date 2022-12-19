from __future__ import annotations

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
from pbsig.utility import lexsort_rows, pairwise
from itertools import chain 

from abc import abstractmethod
from typing import MutableSequence, Protocol, TypeVar

@runtime_checkable
class Comparable(Protocol):
  """Protocol for annotating comparable types."""
  @abstractmethod
  def __lt__(self, other) -> bool:
      pass

@runtime_checkable
class SetLike(Comparable, Protocol):
  """Protocol for annotating set-like types."""
  @abstractmethod
  def __contains__(self, item: Any) -> bool: pass
  
  @abstractmethod
  def __iter__(self) -> Iterator[Any]: pass
  
  @abstractmethod
  def __len__(self) -> int: pass
    
# isinstance(tuple([0,1,2]), SetLike)

def edge_iterator(A):
  assert issparse(A), "Must be a sparse matrix"
  for i,j in zip(*A.nonzero()):
    if i < j: 
      yield (i,j)

def delaunay_complex(X: ArrayLike):
  dt = Delaunay(X)
  # E = edges_from_triangles(dt.simplices, X.shape[0])
  T = dt.simplices
  V = np.fromiter(range(X.shape[0]), dtype=np.int32)
  K = SimplicialComplex(chain(V, T))
  # K = {
  #   'vertices' : np.fromiter(range(X.shape[0]), dtype=np.int32),
  #   'edges' : lexsort_rows(E),
  #   'triangles' : lexsort_rows(T)
  # }
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


# def is_symmetric(A):
#   if issparse(A):
#     r, c = A.nonzero()
#     ri = np.argsort(r)
#     ci = np.argsort(c)
#     return all(r[ri] == c[ci]) and not(any((A != A.T).data))
#   else: 
#     return issymmetric(A)

def laplacian_DA(L):
  """ Converts graph Laplacian L to Adjacency + diagonal degree """
  D = diags(L.diagonal())
  A = L - D
  return D, A

## TODO: remove/replace with up_laplacian interface 
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
class SimplexLike(SetLike, Hashable, Protocol):
  ''' 
  An object is *simplex-like* if it is a Hashable Set

  By definition, this implies a simplex is sized, iterable, and acts as a container (supports vertex containment queries)

  Moreover, a simplex is also immutable and Set-like: 
  
  a boundary operator and has a dimension  

  Inherited Abstract Methods: __hash__, __contains__, __iter__, __len__

  
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

# ## Filtrations need not have delitem / be MutableSequences to match dionysus...
# @runtime_checkable
# class MutableFiltrationLike(MutableSequence[SimplexLike], Protocol):
#   """ """
#   def sort(self, key: Callable[[SimplexLike, SimplexLike], bool]) -> None: raise NotImplementedError 
#   def rearrange(self, indices: Collection) -> None: raise NotImplementedError

from itertools import combinations
from functools import total_ordering
import numpy as np

from numbers import Integral

## The Python data model doesn't exactly allow for immutable, hashable, ordered set-like objects
## https://stackoverflow.com/questions/66874287/python-data-model-type-protocols-magic-methods
## A few typles come close but have limitations: 
## - bytes are comparable and immutable, but values are limited to [0, 255], and not set-like
## - array & np.array are homogeous collections, but they are mutable not set-like
## - tuples are comparable and immutable, but they are not set-like
## - frozensets are comparable, immutable, have unique entries, and are set-like, but are *unordered*. Order affects the simplices orientation.   
## We conclude there is no base Python representation implementation. 
## SortedSet is close, however it is mutable and not hashable. 
## So we make our own Simplex representation!
class Simplex(Set, Hashable):
  '''
  Implements: 
    __contains__(self, v: int) <=> Returns whether integer 'v' is a vertex in 'self'
  '''
  def __init__(self, v: Collection[Integral]) -> None:
    self.vertices = tuple([int(v)]) if isinstance(v, Number) else tuple(np.unique(np.sort(np.ravel(tuple(v)))))
    assert all([isinstance(v, Integral) for v in self.vertices]), "Simplex must be comprised of integral types."
  def __eq__(self, other) -> bool: 
    if len(self) != len(other):
      return False
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
    return Simplex(set(self.vertices) - set(Simplex(other).vertices))
  def __add__(self, other) -> 'Simplex':
    return Simplex(set(self.vertices) | set(Simplex(other).vertices))
  def __hash__(self) -> int:
    # Because Python has no idea about mutability of an object.
    return hash(self.vertices)
    
from collections.abc import Set, Hashable

## TODO: implement a simplex |-> attribute system like networkx graphs
# https://stackoverflow.com/questions/798442/what-is-the-correct-or-best-way-to-subclass-the-python-set-class-adding-a-new
class SimplicialComplex(MutableSet):
  """ Abstract Simplicial Complex"""
  __hash__ = Set._hash
  # wrapped_methods = ('difference', 'intersection', 'symmetric_difference', 'union', 'copy')
  
  # @classmethod
  # def _wrap_methods(cls, names):
  #   def wrap_method_closure(name):
  #     def inner(self, *args):
  #       result = getattr(super(cls, self), name)(*args)
  #       result = cls(result) if isinstance(result, set) else result 
  #       return result
  #     inner.fn_name = name
  #     setattr(cls, name, inner)
  #   for name in names: wrap_method_closure(name)

  # def __new__(cls, iterable=None):
  #   selfobj = super(SimplicialComplex, cls).__new__(SimplicialComplex)
  #   cls._wrap_methods(cls.wrapped_methods)
  #   return selfobj

  def __init__(self, iterable = None):
    """"""
    self.data = SortedSet([], key=lambda s: (len(s), tuple(s), s)) # for now, just use the lex/dim/face order 
    self.update(iterable)

  def __iter__(self) -> Iterator:
    return iter(self.data)
  
  def __len__(self) -> int:
    return self.data.__len__()

  def update(self, iterable):
    for item in iterable:
      self.add(item)

  def add(self, item: Collection[int]):
    # self_set = super(SimplicialComplex, self)
    self.data.update(Simplex(item).faces())
  
  def remove(self, item: Collection[int]):
    self.data.difference_update(set(self.cofaces(item)))
  
  def discard(self, item: Collection[int]):
    self.data.discard(Simplex(item))

  def cofaces(self, item: Collection[int]):
    s = Simplex(item)
    yield from filter(lambda t: t >= s, iter(self))

  def __contains__(self, item: Collection[int]):
    return self.data.__contains__(Simplex(item))

  def faces(self, p: Optional[int] = None) -> Iterable['Simplex']:
    if p is None:
      yield from iter(self)
    else: 
      assert isinstance(p, Number)
      yield from filter(lambda s: len(s) == p + 1, iter(self))

  def dimension(self) -> int: 
    return max([s.dimension() for s in iter(self)])

  def __repr__(self) -> str:
    if self.data.__len__() <= 15:
      return repr(self.data)
    else:
      from collections import Counter
      cc = Counter([s.dimension() for s in iter(self)])
      cc = dict(sorted(cc.items()))
      return f"{max(cc)}-d complex with {tuple(cc.values())}-simplices of dimension {tuple(cc.keys())}"

  def __format__(self, format_spec = "default") -> str:
    from io import StringIO
    s = StringIO()
    self.print(file=s)
    res = s.getvalue()
    s.close()
    return res

  def dim(self) -> tuple:
    from collections import Counter
    cc = Counter([len(s)-1 for s in self.data])
    cc = dict(sorted(cc.items()))
    return cc

  def print(self, **kwargs) -> None:
    d = self.dimension()
    ST = np.zeros(shape=(self.__len__(), d+1), dtype='<U15')
    ST.fill(' ')
    lex_iter = sorted(iter(self), key=lambda s: (len(s), tuple(iter(s))))
    for i,s in enumerate(lex_iter):
      ST[i,:len(s)] = str(s)[1:-1].split(',')
    SC = np.apply_along_axis(lambda x: ' '.join(x), axis=0, arr=ST)
    for i, s in enumerate(SC): 
      ending = '\n' if i != (len(SC)-1) else ''
      print(s, sep='', end=ending, **kwargs)

# def less_lexicographic_refinement(S1: Tuple[SimplexLike, Any], S2: Tuple[SimplexLike, Any]) -> bool:
#   (s1,i1), (s2,i2) = S1, S2
#   if i1 != i2: return(i1 < i2)
#   if len(s1) != len(s2): return(len(s1) < len(s2))
#   return(tuple(iter(s1)) < tuple(iter(s2)))

# Pythonic version: https://grantjenks.com/docs/sortedcontainers/#features

from sortedcontainers import SortedDict, SortedSet
# SortedDict()

from collections import OrderedDict # nah 
from typing import Any 

# The OrderedDict was designed to be good at reordering operations. Space efficiency, iteration speed, and the performance of update operations were secondary.
# A mapping object maps hashable values to arbitrary objects

# MutableMapping abc 
# Requires: __getitem__, __delitem__, __setitem__ , __iter__, and __len__ a
# Inferred: pop, clear, update, and setdefault
# https://treyhunner.com/2019/04/why-you-shouldnt-inherit-from-list-and-dict-in-python/
class MutableFiltration(MutableMapping):
  """
  Simplicial Filtration 

  Implements: __getitem__, __iter__, __len__, __contains__, keys, items, values, get, __eq__, and __ne__
  """
  wrapped_methods = ('_check', '_key', '_list', '_list_add', '_list_clear', '_list_iter', '_list_pop', '_list_remove', '_list_reversed', '_list_update', '_reset', '_setitem', '_update')
  
  # @classmethod
  # def _wrap_methods(cls, names):
  #   def wrap_method_closure(name):
  #     def inner(self, *args):
  #       result = getattr(super(cls, self), name)(*args)
  #       result = cls(result) if isinstance(result, SortedDict) else result 
  #       return result
  #     inner.fn_name = name
  #     setattr(cls, name, inner)
  #   for name in names: wrap_method_closure(name)

  # def __new__(cls, iterable=None):
  #   selfobj = super(MutableFiltration, cls).__new__(MutableFiltration)
  #   cls._wrap_methods(cls.wrapped_methods)
  #   return selfobj

  # _key = int.__le__
  @classmethod
  def _key_dim_lex_poset(cls, s: Simplex) -> bool:
    return (len(s), tuple(s), s)

  ## Returns a newly allocated Sorted Set w/ lexicographical poset ordering
  def _sorted_set(self, iterable: Iterable[Collection[Integral]] = None) -> SortedSet:
    if iterable is None:
      return SortedSet(None, key=MutableFiltration._key_dim_lex_poset)
    else:
      return SortedSet(iter(map(Simplex, iterable)), key=MutableFiltration._key_dim_lex_poset)

  # simplices: Sequence[SimplexLike], I: Optional[Collection] = None
  def __init__(self, iterable: Union[SimplicialComplex, Iterable] = None, f: Optional[Callable] = None) -> None:
    self.data = SortedDict()
    if isinstance(iterable, SimplicialComplex):
      if isinstance(f, Callable):
        self += ((f(s), s) for s in iterable)
      elif f is None:
        index_set = np.arange(len(iterable))  
        iterable = sorted(iter(iterable), key=lambda s: (len(s), tuple(s), s)) # dimension, lex, face poset
        self += zip(iter(index_set), iterable)
      else:
        raise ValueError("Invalid input for simplicial complex")
    elif isinstance(iterable, Iterable):
      # if index_set is None:
      #   index_set = np.arange(len(iterable))  
      # assert len(index_set) == len(iterable)
      #self.data.update(zip(iter(index_set), iterable))
      self += iterable ## accept pairs, like a normal dict
    elif iterable is None:
      pass
    else: 
      raise ValueError("Invalid input")


    # self_dict.__init__(*args, **kwargs)
    # np.fromiter(zip(), self.dtype)
    # self.simplices = simplices
    # self.indices = range(len(simplices)) if I is None else I
    # assert all([isinstance(s, SimplexLike) for s in simplices]), "Must all be simplex-like"
    # if I is not None: assert len(simplices) == len(I)
    # self.simplices = [Simplex(s) for s in simplices]
    # self.index_set = np.arange(0, len(simplices)) if I is None else np.asarray(I)
    # self.dtype = [('s', Simplex), ('index', I.dtype)]
  ## TODO: maybe this shouldn't add, but retain dict-like properties
  ## delegate new behavior to new methods: __iadd__, __isub__
  def update(self, other: Iterable[Tuple[Any, Collection[Integral]]]):
    for k,v in other:
      #print(f"key={str(k)}, val={str(v)}")
      # self._sorted_set()
      self.data.__setitem__(k, self._sorted_set(v))
      # ss.(Simplex(v))



  def __getitem__(self, key: Any) -> Simplex: 
    return self.data.__getitem__(key)

  def __setitem__(self, k: Any, v: Union[Collection[Integral], SortedSet]):
    self.data.__setitem__(k, self._sorted_set(v))
    # if isinstance(v, SortedSet):
    #   self.data.__setitem__(k, v)
    # else:
    #   assert isinstance(v, Collection)
      # self.data.setdefault(k, self._sorted_set())
      # print("here")
      # ss = self._empty_set()
      # ss.add(Simplex(v))
      # print(ss)
      # self.data.__setitem__(k, ss)
  
  ## Returns the value of the item with the specified key.
  ## If key doesn't exist, set's F[key] = default and returns default
  def setdefault(self, key, default=None):
    if key in self.data:
      return self[key] # value type 
    else:
      self.__setitem__(key, default)
      return self[key]   # value type   

  def __delitem__(self, k: Any):
    self.data.__del__(k)
  
  def __iter__(self) -> Iterator:
    return iter(self.data.keys())

  def __len__(self) -> int:
    return sum((len(v) for v in self.data.values()))
    #return self.data.__len__()

  # https://peps.python.org/pep-0584/
  def __or__(self, other: Union[Iterable[Tuple[int, int]], Mapping]):
    new = self.copy()
    new.update(SortedDict(other))
    return new

  def __ror__(self, other: Union[Iterable[Tuple[int, int]], Mapping]):
    new = SortedDict(other)
    new.update(self.data)
    return new

  ## In-place union '|=' operator 
  # TODO: map Collection[Integral] -> SimplexLike? 
  def __ior__(self, other: Union[Iterable[Tuple[int, int]], Mapping]):
    self.data.update(other)
    return self
  
  ## In-place append '+=' operator ; true dict union/merge, retaining values
  def __iadd__(self, other: Iterable[Tuple[int, int]]):
    for k,v in other:
      if len(Simplex(v)) >= 1:
        # print(f"key={str(k)}, val={str(v)}")
        self.setdefault(k, self._sorted_set()).add(Simplex(v))
    return self

  ## Copy-add '+' operator 
  def __add__(self, other: Iterable[Tuple[int, int]]):
    new = self.copy()
    new += other 
    return new

  ## Simple copy operator 
  def copy(self) -> 'MutableFiltration':
    new = MutableFiltration()
    new.data = self.data.copy()
    return new 

  ## Keys yields the index set. Set expand = True to get linearized order. 
  ## TODO: Make view objects
  def keys(self, expand: bool = True):
    if not expand:
      return self.data.keys()
    else:
      it_keys = chain()
      for k,v in self.data.items():
        it_keys = chain(it_keys, repeat(k, len(v)))
      return it_keys
  def values(self, expand: bool = True):
    if not expand:
      return self.data.values()
    else: 
      it_vals = chain()
      for v in self.data.values():
        it_vals = chain(it_vals, iter(v))
      return it_vals
  def items(self, expand: bool = True):
    if not expand:
      return self.data.items()
    else:
      it_keys, it_vals = chain(), chain()
      for k,v in self.data.items():
        it_keys = chain(it_keys, repeat(k, len(v)))
        it_vals = chain(it_vals, iter(v))
      return zip(it_keys, it_vals)

  def __repr__(self) -> str:
    from collections import Counter
    cc = Counter([len(s)-1 for s in self.values(expand=True)])
    cc = dict(sorted(cc.items()))
    return f"{max(cc)}-d filtered complex with {tuple(cc.values())}-simplices of dimension {tuple(cc.keys())}"

  def print(self, **kwargs) -> None:
    import sys
    fv_s, fs_s = [], []
    for k,v in self.items(expand=True):
      ks = len(str(v))
      fv_s.append(f"{str(k):<{ks}.{ks}}")
      fs_s.append(f"{str(v): <{ks}}")
      assert len(fv_s[-1]) == len(fs_s[-1])
    sym_le, sym_inc = (' ≤ ', ' ⊆ ') if sys.getdefaultencoding()[:3] == 'utf' else (' <= ', ' <= ') 
    print(repr(self))
    print("I: " + sym_le.join(fv_s[:5]) + sym_le + ' ... ' + sym_le + sym_le.join(fv_s[-2:]), **kwargs)
    print("S: " + sym_inc.join(fs_s[:5]) + sym_inc + ' ... ' + sym_inc + sym_inc.join(fs_s[-2:]), **kwargs)

  def validate(self, light: bool = True) -> bool:
    fs = list(self.values(expand=True))
    for i, s in enumerate(fs): 
      p = s.dimension() - 1 if light and len(s) >= 2 else None
      assert all([fs.index(face) <= i for face in s.faces(p)])
    assert all([k1 <= k2 for k1, k2 in pairwise(self.keys(expand=False))])

  def __format__(self, format_spec = "default") -> str:
    from io import StringIO
    s = StringIO()
    self.print(file=s)
    res = s.getvalue()
    s.close()
    return res

  # def update(self, E, *args):


  # ## --- Collection methods ---
  # def __contains__(self, __x: Collection[int]) -> bool:
  #   return self.simplices.__contains__(Simplex(__x))
  # def __iter__(self) -> Iterator:
  #   return iter(self.simplices)
  # def __len__(self):
  #   return len(self.simplices)

  # ## --- Mapping methods ---
  # def __getitem__(self, index: Any) -> Tuple[Simplex, Any]:
  #   idx = self.index_set.searchsorted(index)
  #   return self.simplices[idx]

  # def keys(self):
  #   return self.index_set
  
  # def values(self):
  #   return self.simplices

  # def items(self):
  #   return zip(self.keys, self.simplices)

  # # def __eq__(self, other):
  # #   return self.
  # def get(self, key, value = None):
  #   return self[key] if key in self.index_set else value

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
  # def __validate__(self):
  #   """ Checks the face poset is valid, and asserts that the corresponding indices align as well """
  #   for (i, s), idx in zip(enumerate(self.simplices), self.indices):
  #     if s.dimension() == 0: 
  #       continue
  #     else: 
  #       ## Check: all faces in filtration, all at indices < coface index, all have stored indices < coface stored index
  #       assert all([f in self.simplices for f in s.boundary()]), "Not all faces contained in filtration"
  #       face_ind = np.array([self.simplices.index(f) for f in s.boundary()])
  #       assert all(face_ind < i), "There exist faces that come after their coface in filtration order"
  #       assert all([self.indices[fi] < idx for fi in face_ind]), "Index set not consistent with face poset"
  #       # assert all([f in self.simplices and self.simplices.index(f) < i for f in s.boundary()])
  #   return(True)
  # def __repr__(self):
  #   return "F: "+str(self.simplices)

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


# See: https://stackoverflow.com/questions/70381559/ensure-that-an-argument-can-be-iterated-twice
from scipy.sparse import coo_array
def _boundary(S: Iterable[SimplexLike], F: Optional[Sequence[SimplexLike]] = None):

  ## Load faces. If not given, by definition, the given p-simplices contain their boundary faces.
  if F is None: 
    assert not(S is iter(S)), "Simplex iterable must be repeatable (a generator is not sufficient!)"
    F = list(map(Simplex, set(chain.from_iterable([combinations(s, len(s)-1) for s in S]))))
  
  ## Ensure faces 'F' is indexable
  assert isinstance(F, Sequence), "Faces must be a valid Sequence (supporting .index(*) with SimplexLike objects!)"

  ## Build the boundary matrix from the sequence
  m = 0
  I,J,X = [],[],[] # row, col, data 
  for (j,s) in enumerate(map(Simplex, S)):
    if s.dimension() > 0:
      I.extend([F.index(f) for f in s.faces(s.dimension()-1)])
      J.extend(repeat(j, s.dimension()+1))
      X.extend(islice(cycle([1,-1]), s.dimension()+1))
    m += 1
  D = coo_array((X, (I,J)), shape=(len(F), m)).tolil(copy=False)
  return D 

def boundary_matrix(K: Union[SimplicialComplex, MutableFiltration, Iterable[tuple]], p: Optional[Union[int, tuple]] = None):
  """
  Returns the ordered p-th boundary matrix of a simplicial complex 'K'

  Return: 
    D := sparse matrix representing either the full or p-th boundary matrix (as List-of-Lists format)
  """
  from collections.abc import Sized
  if isinstance(p, tuple):
    return (boundary_matrix(K, pi) for pi in p)
  else: 
    assert p is None or isinstance(p, Integral), "p must be integer, or None"
    if isinstance(K, MutableFiltration):
      assert p is None
      simplices = list(K.values())
      D = _boundary(simplices, simplices)
    elif isinstance(K, SimplicialComplex):
      if p is None:
        simplices = list(iter(K))
        D = _boundary(simplices, simplices)
      else:
        p_simplices = K.faces(p=p)
        p_faces = list(K.faces(p=p-1))
        D = _boundary(p_simplices, p_faces)
    else: 
      raise ValueError("Invalid input")
    return D