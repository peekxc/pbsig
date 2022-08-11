import sys
import numpy as np 

from itertools import combinations, tee
from operator import le
from typing import * 
from numpy.typing import ArrayLike
from math import comb


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
def progressbar(it, count=None, prefix="", size=60, out=sys.stdout): # Python3.6+
  count = len(it) if count == None else count 
  def show(j):
    x = int(size*j/count)
    print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}", end='\r', file=out, flush=True)
  show(0)
  for i, item in enumerate(it):
    yield item
    show(i+1)
  print("\n", flush=True, file=out)

def is_sorted(L: Iterable, compare: Callable = le):
  a, b = tee(L)
  next(b, None)
  return all(map(compare, a, b))

# def is_sorted(A: Iterable, key=lambda x: x):
#   for i, el in enumerate(A[1:]):
#     if key(el) < key(A[i]): # i is the index of the previous element
#       return False
#   return True

def smoothstep(lb: float = 0.0, ub: float = 1.0, order: int = 1, reverse: bool = False):
  """
  Maps [lb, ub] -> [0, 1] via a smoother version of a step function
  """
  assert ub > lb, "Invalid input"
  d = (ub-lb)
  def _ss(x: float):
    y = (x-lb)/d if not(reverse) else 1.0 - (x-lb)/d
    if (y <= 0.0): return(0.0)
    if (y >= 1.0): return(1.0)
    return(3*y**2 - 2*y**3)
  return(np.vectorize(_ss))

def rank_C2(i: int, j: int, n: int):
  i, j = (j, i) if j < i else (i, j)
  return(int(n*i - i*(i+1)/2 + j - i - 1))

def unrank_C2(x: int, n: int):
  i = int(n - 2 - np.floor(np.sqrt(-8*x + 4*n*(n-1)-7)/2.0 - 0.5))
  j = int(x + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2)
  return(i,j) 

def unrank_comb(r: int, k: int, n: int):
  result = np.zeros(k, dtype=int)
  x = 1
  for i in range(1, k+1):
    while(r >= comb(n-x, k-i)):
      r -= comb(n-x, k-i)
      x += 1
    result[i-1] = (x - 1)
    x += 1
  return(result)

def unrank_combs(R: Iterable, k: int, n: int):
  if k == 2: 
    return(np.array([unrank_C2(r, n) for r in R], dtype=int))
  else: 
    return(np.array([unrank_comb(r, k, n) for r in R], dtype=int))

def rank_comb(c: Tuple, k: int, n: int):
  c = np.array(c, dtype=int)
  #index = np.sum(comb((n-1)-c, np.flip(range(1, k+1))))
  index = np.sum([comb(cc, kk) for cc,kk in zip((n-1)-c, np.flip(range(1, k+1)))])
  return(int((comb(n, k)-1) - int(index)))

def rank_combs(C: Iterable, k: int, n: int):
  if k == 2: 
    return(np.array([rank_C2(c[0], c[1], n) for c in C], dtype=int))
  else: 
    return(np.array([rank_comb(c, k, n) for c in C], dtype=int))

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