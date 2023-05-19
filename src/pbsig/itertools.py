import numpy as np 
from itertools import *
from typing import * 
from numpy.typing import ArrayLike
from numbers import Complex, Integral, Number
from more_itertools import * 

# Simple counter mechanism to count evaluation calls
class Counter():
  def __init__(self): self.cc = 0
  def __call__(self, *args, **kwargs): self.cc += 1
  def __repr__(self): return f"Number of calls: {self.cc}"
  def num_calls(self): return self.cc


## Caches the least-recently used value
class LazyIterable(Sized):
  """ Constructs of a lazily-evaluated value iterable whose contents are accessed via a Callable """
  def __init__(self, accessor: Callable, count: int) -> None:
    self.accessor = accessor
    self._count = count

  @lru_cache(maxsize=1, typed=False)
  def __getitem__(self, *args: Hashable, **kwargs: Hashable) -> Any:
    return self.accessor(*args, **kwargs)
  
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
    self.accessor = accessor
    self._index_set = list(index_set) 
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

  def __getitem__(self, i: Any) -> Any:
    ## Use _index_set as the domain
    ind = self._index_set.index(i)
    if ind == self._c_index: # use item cache
      return self._c_value
    self._c_index = ind
    self._c_value = self.accessor(ind) 
    return self._c_value
  # else: 
    #   assert isinstance(i, Integral)
    #   self._items.seek(i) # items with be fully seekable 
    #   return self._items.peek() 

  def __len__(self) -> int:
    return self._count


