import numpy as np 
from itertools import combinations

from typing import * 
from numpy.typing import ArrayLike
from math import comb

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