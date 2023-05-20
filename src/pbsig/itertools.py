import numpy as np 
from itertools import *
from typing import * 
from numpy.typing import ArrayLike
from numbers import Complex, Integral, Number
from more_itertools import * 
from math import prod
from functools import lru_cache
from splex.predicates import *

# Simple counter mechanism to count evaluation calls
class Counter():
  def __init__(self): self.cc = 0
  def __call__(self, *args, **kwargs): self.cc += 1
  def __repr__(self): return f"Number of calls: {self.cc}"
  def num_calls(self): return self.cc

## Caches the least-recently used value
## An iterable which supports efficient element access using integer indices via the __getitem__() special method and 
## defines a __len__() method that returns the length of the sequence.
class LazyIterable(Sequence, Sized):
  """ Constructs of a lazily-evaluated value iterable whose contents are accessed via a Callable """
  def __init__(self, accessor: Callable, count: int) -> None:
    assert isinstance(accessor, Callable) and isinstance(count, Integral), "Invalid argument types supplied to constructor"
    self._accessor = accessor
    self._count = count

  @lru_cache(maxsize=1, typed=True)
  def __getitem__(self, index: int, *args: Hashable, **kwargs: Hashable) -> Any:
    return self._accessor(index, *args, **kwargs)
  
  def __len__(self) -> int:
    return self._count


class RepeatableIterable(Sequence, Sized):
  def __init__(self, it: Iterable, count: int):
    self._items = seekable(it)
    self._count = count 
  def __iter__(self):
    self._items.seek(0)
    return iter(self._items)
  def __getitem__(self, i: int):
    self._items.seek(i)
    return self._items.peek()
  def __len__(self) -> int: 
    return self._count

## See: https://stackoverflow.com/questions/62970581/what-exactly-is-a-sequence
## TODO: Maybe figure out inheritance solution here
class sequence(Sequence):
  """Wrapped Sequence-registered type for a duck-typed sequence""" 
  def __init__(self, seq: Any):
    assert hasattr(seq, "__getitem__") and hasattr(seq, "__len__")
    self._seq = seq
  def __getitem__(self, index: Union[slice, int]):
    return self._seq.__getitem__(index)
  def __len__(self) -> int:
    return len(self._seq)
  def __iter__(self) -> Iterator:
    return iter(self._seq)
  def __contains__(self, x: Any) -> bool:
    return self._seq.__contains__(x)

# does not perform any checking
def rank_tuple(t: tuple, N: tuple) -> int:
  """Ranks the integer-tuple _t_ in the lexicographical ordering of the product given by product(*map(range, N)) """
  ind = np.append([1], np.cumprod(np.flip(N))[:-1])
  return sum(np.array(t)*np.flip(ind))

# does not perform any checking
def unrank_tuple(r: int, N: tuple) -> tuple: 
  """Unranks an integer rank _r_ into the _r_-th tuple in the product(*map(range, N)) """
  t = np.zeros(len(N), dtype=int)
  for i, n in enumerate(reversed(N)):
    t[-(i+1)] = r % n
    r = r // n
  return tuple(t)

def rproduct(*iterables: Iterable[Sequence]):
  """Repeatable product operator."""
  class _repeatable_product(Sequence, Sized):
    def __init__(self, *iterables):
      self.views = [SequenceView(sequence(s)) for s in iterables]
    def __len__(self) -> int: 
      return prod([len(view) for view in self.views])
    def __iter__(self) -> Iterator:
      return iter(product(*self.views))
    def __getitem__(self, index: int) -> tuple:
      # if index < 0 or index > len(self): raise ValueError("Invalid index supplied")
      ind = unrank_tuple(index, tuple([len(view) for view in self.views]))
      return tuple((v[i] for i,v in zip(ind, self.views)))
  return _repeatable_product(*iterables)

## TODO: handle args somehow
def rmap(function: Callable, iterable: Iterable):
  class _repeatable_map(Sequence, Sized):
    def __init__(self, f, I):
      self.function = f
      self.view = SequenceView(sequence(I))
      # self.args = [SequenceView(sequence(arg)) for arg in args]
    def __len__(self) -> int: 
      return len(self.view)
    def __iter__(self) -> Iterator:
      return iter(map(self.function, self.view))
    def __getitem__(self, index: int) -> tuple:
      # if index < 0 or index >= len(self): raise ValueError("Invalid index supplied")
      return self.function(self.view[index])
  return _repeatable_map(function, iterable)


def rstarmap(function: Callable, iterable: Iterable):
  class _repeatable_starmap(Sequence, Sized):
    def __init__(self, f, args):
      self.function = f
      self.args_view = SequenceView(sequence(args))
      # self.args = [SequenceView(sequence(arg)) for arg in args]
    def __len__(self) -> int: 
      return len(self.args_view)
    def __iter__(self) -> Iterator:
      return iter(starmap(self.function, self.args_view))
    def __getitem__(self, index: int) -> tuple:
      # if index < 0 or index > len(self): raise ValueError("Invalid index supplied")
      return self.function(*self.args_view[index])
  return _repeatable_starmap(function, iterable)

class IndexedIterable(Sequence, Sized):
  """Constructs a repeatable sequence in a functional way.
  
  The corresponding iterable is sized and sequence-like. Adds structure (cache and length information) to generators 
  that allows for more structured enumeration of common types. 

  Parameters:
    x: index set or value generator. If accessor is provided, treats _x_ as domain of _accessor_ (index set). Otherwise, _x_ is assumed to represent the values to repeat. 
    count: the size of _x_, if known ahead-of-time.
    accessor: the equivalent of __getitem__ defined over _x_,  

  Examples: 

    ## Usage case 0: simple in-memory iterable convertor from generator
    S = RepeatableIterable(range(10)) # caches values in generator + deduces size
    for value in S:
      print(value) # this points to the values in 'data'
    assert is_repeatable(S) # true

    ## Usage case 1: simple in-memory domain/iterable convertor
    S = RepeatableIterable(['a', 'b', 'c'], count=3, accessor=lambda s: data[s]) # where data is dict storing things 
    for value in S:
      print(value) # this points to the values in 'data'

    ## Usage case 2: Custom domain iterable + accessor function
    data_dir = os.path()
    filenames = ['shape1.obj', 'shark2.obj', 'penguin.obj']
    def load_model(fn: str) -> ArrayLike: 
      with open(fn, 'r') as f:
        mesh = ... # preprocess f 
        return mesh 
    S = RepeatableIterable(filenames, accessor=load_model)

  """
  def __init__(self, index_set: Iterable, accessor: Callable, count: int = None) -> None:
    assert isinstance(accessor, Callable), f"Invalid accessor type given '{type(accessor)}' (must be Callable)"
    assert isinstance(is_repeatable(index_set)), "'index_set' iterable must be a repeatable" # also checks sized 
    assert isinstance(index_set, Sequence), "'index_set' must be Sequence type with valid .index(value) method"

    self.accessor = accessor
    self.index_set = index_set
    starmap(self.accessor, self.index_set)
    if count is None: # either precompute or deduce the size
      self._count = sum(1 for _ in self._index_set) if not isinstance(x, Sized) else len(index_set) 
    else: 
      assert isinstance(count, Integral) 
      self._count = int(count)
    # self._index_set = seekable(range(self._count)) if self.accessor is None else self._items
    self._c_index = None
    self._c_value = None

  def __iter__(self) -> Iterator[Any]:
    self._index_set.seek(0)
    for index in self._index_set:
      value = self.__getitem__(index)
      yield value

  def __getitem__(self, *args, **kwargs) -> Any:
    index_set.index(*args, ) 
    return self.accessor(index_set[0])
    # ## Use _index_set as the domain
    # ind = self._index_set.index(i)
    # if ind == self._c_index: # use item cache
    #   return self._c_value
    # self._c_index = ind
    # self._c_value = self.accessor(ind) 
    # return self._c_value
  # else: 
    #   assert isinstance(i, Integral)
    #   self._items.seek(i) # items with be fully seekable 
    #   return self._items.peek() 

  def __len__(self) -> int:
    return self._count


