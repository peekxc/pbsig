import sys
import numpy as np 
from itertools import *
from operator import le
from typing import * 
from numpy.typing import ArrayLike
from math import comb
from numbers import Complex, Integral, Number
from more_itertools import * 


## From: https://stackoverflow.com/questions/492519/timeout-on-a-function-call
def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
  import signal
  class TimeoutError(Exception): pass
  def handler(signum, frame): raise TimeoutError()
  # set the timeout handler
  signal.signal(signal.SIGALRM, handler) 
  signal.alarm(timeout_duration)
  result = default
  try:
    result = func(*args, **kwargs)
  except TimeoutError as exc:
    result = default
  finally:
    signal.alarm(0)
  return result

def lexsort_rows(A):
  """ Returns A with its rows in sorted, lexicographical order. Each row is also ordered. """
  A = np.array([np.sort(a) for a in A])
  return A[np.lexsort(np.rot90(A))] 

def window(S: Iterable, n: int = 2):
  "Returns a sliding window (of width n) over data from the iterable"
  "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
  it = iter(S)
  result = tuple(islice(it, n))
  if len(result) == n:
    yield result
  for elem in it:
    result = result[1:] + (elem,)
    yield result

def rotate(S: Iterable, n: int = 1):
  from collections import deque
  items = deque(S)
  items.rotate(n)
  return iter(items)

def cycle_window(S: Iterable, offset: int = 1, w: int = 2):
  return window(islice(cycle(S), len(S)+offset), w)

def partition_envelope(f: Callable, threshold: float, interval: Tuple = (0, 1), lower: bool = False):
  """
  Partitions the domain of a real-valued function 'f' into intervals by evaluating the pointwise maximum of 'f' and the constant function g(x) = threshold. 
  The name 'envelope' is because this can be seen as intersecting the upper-envelope of 'f' with 'g'

  Parameters: 
    f := Callable that supports f.operator(), f.derivative(), and f.roots() (such as the Splines from scipy)
    threshold := cutoff-threshold
    interval := interval to evalulate f over 
    lower := whether to evaluate the lower envelope instead of the upper 

  Return: 
    intervals := (m x 3) nd.array giving the intervals that partition the corresponding envelope
  
  Each row (b,e,s) of intervals has the form: 
    b := beginning of interval
    e := ending of interval
    s := 1 if in the envelope, 0 otherwise 

  """
  assert isinstance(interval, Tuple) and len(interval) == 2

  ## Partition a curve into intervals at some threshold
  in_interval = lambda x: x >= interval[0] and x <= interval[1]
  crossings = np.fromiter(filter(in_interval, f.solve(threshold)), float)

  ## Determine the partitioning of the upper-envelope (1 indicates above threshold)
  intervals = []
  if len(crossings) == 0:
    is_above = f(0.50).item() >= threshold
    intervals.append((0.0, 1.0, 1 if is_above else 0))
  else:
    if crossings[-1] != 1.0: 
      crossings = np.append(crossings, 1.0)
    b = 0.0
    df = f.derivative(1)
    df2 = f.derivative(2)
    for c in crossings:
      grad_sign = np.sign(df(c))
      if grad_sign == -1:
        intervals.append((b, c, 1))
      elif grad_sign == 1:
        intervals.append((b, c, 0))
      else: 
        accel_sign = np.sign(df2(c).item())
        if accel_sign > 0: # concave 
          intervals.append((b, c, 1))
        elif accel_sign < 0: 
          intervals.append((b, c, 0))
        else: 
          raise ValueError("Unable to detect")
      b = c
  
  ### Finish up and return 
  intervals = np.array(intervals)
  if lower: 
    intervals[:,2] = 1 - intervals[:,2]
  return(intervals)
  
## From: https://stackoverflow.com/questions/3160699/python-progress-bar
def progressbar(it, count=None, prefix="", size=60, out=sys.stdout, f: Callable = None, newline: bool = True): # Python3.6+
  count = len(it) if count == None else count 
  f = (lambda j: "") if f is None else f
  def show(j):
    x = int(size*j/count)
    print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}" + f(j), end='\r', file=out, flush=True)
  show(0)
  for i, item in enumerate(it):
    yield item
    show(i+1)
  print("\n" if newline else "", flush=True, file=out)

def is_sorted(L: Iterable, compare: Callable = le):
  a, b = tee(L)
  next(b, None)
  return all(map(compare, a, b))

# def is_sorted(A: Iterable, key=lambda x: x):
#   for i, el in enumerate(A[1:]):
#     if key(el) < key(A[i]): # i is the index of the previous element
#       return False
#   return True


def smoothstep(lb: float = 0.0, ub: float = 1.0, order: int = 1, down: bool = False):
  """
  Maps [lb, ub] -> [0, 1] via a smoother version of a step function. When down=False and lb=ub, the step functions look like: 

  down = False       |    down = True
  1:      o------    |    1: -----* 
  0: -----*          |    0:      o------  

  When |lb - ub| > 0, the returned function is a differentiable relaxation the above step function(s). 

  Parameters: 
    lb := lower bound in the domain to begin the step 
    ub := upper bound in the domain to end the step 
    order := smoothness parameter 
    down := whether to make the step a down step. Default to False (makes an up-step).
  
  Returns a vectorized function S(x) satisfying one of: 
    1. S(x) = 0 for all x <= lb, 0 < S(x) < 1 for all lb < x < ub, and S(x) = 1 for all x >= ub, if down = False
    2. S(x) = 1 for all x <= lb, 0 < S(x) < 1 for all lb < x < ub, and S(x) = 0 for all x >= ub, if down = True
  """
  assert ub >= lb, "Invalid input"
  if lb == ub:
    if down: 
      def _ss(x: float): return np.where(x <= lb, 1.0, 0.0)
      return _ss 
    else:
      def _ss(x: float): return np.where(x <= lb, 0.0, 1.0)
      return _ss 
  elif order == 1: 
    d = (ub-lb)
    assert d > 0, "Must be positive distance"
    if down: 
      def _ss(x: np.array):
        y = np.minimum(np.maximum((x-lb)/d, 0), 1.0)
        return np.clip((1.0 - (3*y**2 - 2*y**3)), 0.0, 1.0)
      return _ss 
    else:       
      def _ss(x: float):
        y = np.minimum(np.maximum((x-lb)/d, 0), 1.0)
        return np.clip(3*y**2 - 2*y**3, 0.0, 1.0)
      return _ss 
  elif order == 2: 
    d = (ub-lb)
    assert d > 0, "Must be positive distance"
    if down: 
      def _ss(x: np.array):
        y = np.clip((x-lb)/d, 0.0, 1.0)
        return np.clip((1.0 - (6*y**5 - 15*y**4 + 10*y**3)), 0.0, 1.0)
      return _ss 
    else:       
      def _ss(x: float):
        y = np.clip((x-lb)/d, 0.0, 1.0)
        return np.clip(6*y**5 - 15*y**4 + 10*y**3, 0.0, 1.0)
      return _ss 
  else: 
    raise ValueError("Smooth step only supports order parameters in [0,2]")

## Convenience wrapper
def smooth_upstep(lb: float = 0.0, ub: float = 1.0, order: int = 1):
  return smoothstep(lb=lb, ub=ub, order=order, down=False)

## Convenience wrapper
def smooth_dnstep(lb: float = 0.0, ub: float = 1.0, order: int = 1):
  return smoothstep(lb=lb, ub=ub, order=order, down=True)

from array import array 
class BlockReduce():
  def __init__(self, blocks: np.ndarray = None, lengths: np.ndarray = None, dtype: np.dtype = None):
    self.dtype = np.float32 if blocks is None else np.array([np.take(blocks,0)]).dtype
    self.blocks = array(np.sctype2char(self.dtype)) 
    self.lengths = array('I')
    if blocks is not None:
      assert len(blocks) == len(lengths)
      self.blocks.extend(blocks)
      self.lengths.extend(lengths)

  def __getitem__(self, key: int):
    index = int(np.sum(self.lengths[:key]))
    arr_len = int(self.lengths[key])
    return self.blocks[index:(index+arr_len)]

  def __iadd__(self, x: np.ndarray):
    x = np.ravel(x).astype(self.dtype)
    self.blocks.extend(x)
    self.lengths.extend([len(x)])
    return self

  def reduce(self, transform: Callable = None, blockwise: bool = False, reduction: Callable = np.add):
    """Reduces contiguous ranges of elements 'blocks' via 'reduce' after transforming with 'f'. """
    assert isinstance(reduction, np.ufunc), "Reduce must be a 'ufunc'"
    c_lengths = np.concatenate(([0], np.array(self.lengths[:-1]).cumsum()), dtype=np.int64)
    if transform is not None:
      f_ufunc = np.frompyfunc(transform, nin=1, nout=1) if not(isinstance(transform, np.ufunc)) else transform
      is_vector_valued = len(f_ufunc(np.array([0,0,0,0,0]))) == 5
      assert is_vector_valued, "Transform must be a vector-valued function"
      if blockwise:
        t_blocks = np.concatenate([f_ufunc(l).astype(self.dtype) for l in np.split(self.blocks, c_lengths[1:])])
      else: 
        t_blocks = transform(np.array(self.blocks))
    else: 
      t_blocks = self.blocks
    return reduction.reduceat(t_blocks, c_lengths, dtype=self.dtype)

  def __repr__(self):
    return f'BlockReduce with {len(self.lengths)} blocks'

sorting_nets = { 
  1 : [[0]],
  2 : [[0,1]],
  3 : [[0,1],[0,2],[1,2]],
  4 : [[0,1],[2,3],[0,2],[1,3],[1,2]],
  5 : [[0,1],[2,3],[0,2],[1,4],[0,1],[2,3],[1,2],[3,4],[2,3]],
  6 : [[0,1],[2,3],[4,5],[0,2],[1,4],[3,5],[0,1],[2,3],[4,5],[1,2],[3,4],[2,3]]
}

def parity(X: Collection):
  """ 
  Finds the number of inversions of any collection of comparable numbers of size <= 6. 
  
  On the implementation side, sorts 'X' using sorting networks to find the number of inversions. 
  """
  X = np.array(X, copy=True)
  if len(X) == 1: return 0
  nt: int = 0
  for i,j in sorting_nets[len(X)]:
    X[[i,j]], nt = (X[[j,i]], nt+abs(i-j)) if X[i] > X[j] else (X[[i,j]], nt)
  return nt

# https://math.stackexchange.com/questions/415970/how-to-determine-the-parity-of-a-permutation-by-its-cycle-decomposition
# TODO: find number of cycles w/o Sympy
def sgn_permutation(p):
  return int((-1)**(len(p)-Permutation(p).cycles))

def edges_from_triangles(triangles: ArrayLike, nv: int):
  ER = np.array([[rank_C2(*t[[0,1]], n=nv), rank_C2(*t[[0,2]], n=nv), rank_C2(*t[[1,2]], n=nv)] for t in triangles])
  ER = np.unique(ER.flatten())
  E = np.array([unrank_C2(r, n=nv) for r in ER])
  return(E)

def expand_triangles(nv: int, E: ArrayLike):
  ''' k=2 Expansion: requires Gudhi '''
  from gudhi import SimplexTree
  st = SimplexTree()
  for v in range(nv): st.insert([int(v)], 0.0)
  for e in E: st.insert(e)
  st.expansion(2)
  triangles = np.array([s[0] for s in st.get_simplices() if len(s[0]) == 3])
  return(triangles)

def scale_diameter(X: ArrayLike, diam: float = 1.0):
  from scipy.spatial.distance import pdist
  vec_mag = np.reshape(np.linalg.norm(X, axis=1), (X.shape[0], 1))
  vec_mag[vec_mag == 0] = 1 
  VM = np.reshape(np.repeat(vec_mag, X.shape[1]), (len(vec_mag), X.shape[1]))
  Xs = (X / VM) * diam*(VM/(np.max(vec_mag)))
  return(Xs)



# def interpolate_PL_path(path: Sequence, k: int): 
#   """ Interpolates a Path of Line objects """
#   from svgpathtools import parse_path, Line, Path, wsvg
#   arc_lengths = np.array([np.linalg.norm(seg.length()) for seg in path])
#   if any(arc_lengths == 0):
#     path = list(filter(lambda p: p.length() > 0.0, path))
#     for j in range(len(path)-1):
#       if path[j].end != path[j+1].start:
#         path[j].end = path[j+1].start
#     path = Path(*path)
#   arc_lengths = np.array([np.linalg.norm(seg.length())for seg in path])
#   assert(all(arc_lengths > 0)), "Invalid shape detected: lines with 0-length found and not handled"
#   A = np.cumsum(arc_lengths)
#   p_cc = np.linspace(0, max(A), k)
#   idx = np.digitize(p_cc, A+np.sqrt(np.finfo(float).resolution))
#   L_points = []
#   for i, pp in zip(idx, p_cc):
#     t = pp/A[i] if i == 0 else (pp-A[i-1])/(A[i]-A[i-1])
#     L_points.append(path[i].point(t))
#   connect_the_dots = [Line(p0, p1) for p0, p1 in pairwise(L_points)]
#   if any(connect_the_dots[-1].end != connect_the_dots[0].start): #not(path.iscontinuous()) or path.isclosed():
#     #connect_the_dots.append(Line(L_points[-1], L_points[0]))
#     connect_the_dots.append(Line(connect_the_dots[-1].end, connect_the_dots[0].start))
#   new_path = Path(*connect_the_dots)
#   return(new_path)

## From: https://stackoverflow.com/questions/1376438/how-to-make-a-repeating-generator-in-python
def multigen(gen_func):
  class _multigen(object):
    def __init__(self, *args, **kwargs):
      self.__args = args
      self.__kwargs = kwargs
    def __iter__(self):
      return gen_func(*self.__args, **self.__kwargs)
  return _multigen


def split_train_test(n: int, split: tuple = (0.80,0.20), labels: ArrayLike = None):
  """Partitions the range [0, n-1] into disjoint training/testing indices, optionally stratified. """
  assert sum(split) == 1, "Split must sum to 1."
  if labels is not None:
    train_ind, test_ind = [], []
    for cl in np.unique(labels):
      n = np.sum(labels == cl)
      indices = np.random.permutation(n)
      cut_ind = int(np.floor(n * split[0]))
      train_ind.extend(np.flatnonzero(labels == cl)[indices[:cut_ind]])
      test_ind.extend(np.flatnonzero(labels == cl)[indices[cut_ind:]])
    train_ind, test_ind = np.ravel(train_ind), np.ravel(test_ind)
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    return train_ind,test_ind
  else:
    indices = np.random.permutation(n)
    cut_ind = int(np.floor(n * split[0]))
    training_idx, test_idx = indices[:cut_ind], indices[cut_ind:]
    return training_idx, test_idx

import heapq
import numpy as np

# From: https://stats.stackexchange.com/questions/459601/are-these-any-existing-implementation-of-l1-isotonic-regression-in-python
def isotonic_regress(y: ArrayLike, w: Optional[ArrayLike] = None):
  """Finds a non-decreasing fit for the specified `y` under L1 norm.

  The O(n log n) algorithm is described in:
  "Isotonic Regression by Dynamic Programming", Gunter Rote, SOSA@SODA 2019.

  Args:
    y: The values to be fitted, 1d-numpy array.
    w: The loss weights vector, 1d-numpy array.

  Returns:
    An isotonic fit for the specified `y` which minimizies the weighted
    L1 norm of the fit's residual.
  """
  if w is None: 
    w = np.ones(len(y))
  h = []  # max heap of values
  p = np.zeros_like(y)  # breaking position
  for i in range(y.size):
    a_i = y[i]
    w_i = w[i]
    heapq.heappush(h, (-a_i, 2 * w_i))
    s = -w_i
    b_position, b_value = h[0]
    while s + b_value <= 0:
      s += b_value
      heapq.heappop(h)
      b_position, b_value = h[0]
    b_value += s
    h[0] = (b_position, b_value)
    p[i] = -b_position
  z = np.flip(np.minimum.accumulate(np.flip(p)))  # right_to_left_cumulative_min
  return z

def profile_f(f: Callable, *args):
  import line_profiler
  profile = line_profiler.LineProfiler()
  profile.add_function(f)
  profile.enable_by_count()
  f(*args)
  profile.print_stats(output_unit=1e-3, stripzeros=True)




def spectral_bound(V, E, type: str = ["graph", "laplacian"]):
  """
  Returns bounds on the spectral radius of either the Graph G = (V, E) or the Graph Laplacian L = D(G) - A(G)
  Based on the simple bounds from: "Bounds on the (Laplacian) spectral radius of graphs"
  """
  import networkx as nx
  G = nx.Graph()
  G.add_nodes_from(V)
  G.add_edges_from(E)
  bound = 0
  if (isinstance(type, list) and type == ["graph", "laplacian"]) or type == "graph":
    bound = np.sqrt(max([sum([G.degree(u) for u in G.neighbors(v)]) for v in G.nodes()]))
  elif type == "laplacian":
    bound = np.sqrt(2)*np.sqrt(max([G.degree(v)**2 + sum([G.degree(u) for u in G.neighbors(v)]) for v in G.nodes()])) 
  else: 
    raise ValueError("invalid graph type")
  return(bound)

